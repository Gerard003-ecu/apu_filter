# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Thermal Gradient Laws (Leyes y Gradientes de Convección Térmica)    ║
║ Ruta   : app/physics/thermal_gradient_laws.py                                ║
║ Versión: 3.1.0-Doctoral-Lanczos-ItohAbe-AdaptiveCD-Fourier-Carnot-KBN-CSMD   ║
║                                                                              ║
║ SINOPSIS (rigor doctoral; lo que SÍ se computa):                             ║
║                                                                              ║
║   (Fase 1)  Deflación espectral de Weyl / Lanczos–Krylov                     ║
║             λ_min, λ_max, κ₂(K) extraídos por ARPACK (eigsh, both-ends)      ║
║             cuando n > n_krylov. Higham: K ← K + (α − λ_min)_+ I.            ║
║             Opcional: K ≈ Σ_{i=1}^{k} λ_i v_i v_iᵀ + γ (I − P_k)             ║
║             (rango k + ridge; k ≪ n). Complejidad O(k n²) vs O(n³).          ║
║                                                                              ║
║   (Fase 2)  Gradiente discreto de Itoh–Abe de Ē(p) = pᵀ κ p                  ║
║             ⟨∇̄_IA Ē(0, p), p⟩ = Ē(p) − Ē(0)     (Tellegen / cadena discreta) ║
║             ⟨q, dT⟩ := −⟨∇̄_IA Ē(0, ∇T), ∇T⟩                                  ║
║             q_Fourier = −κ ∇T  (constitutiva); q_IA audita el pairing.       ║
║             Carnot local: η_C = L|dT|_g / (T + L|dT|_g)  (unidades legales). ║
║                                                                              ║
║   (Fase 3)  Umbral CD adaptativo (envolvente de transitorio)                 ║
║             b = | |dT|_g − s_t | ,  s_t = ρ s_{t−1} + (1−ρ) |dT|_g           ║
║             ν(b) = log(1 + b/(s_t+ε))     (valuación ultramétrica-surrogate) ║
║             τ_CD(t) = τ₀ exp(−ν(b))                                          ║
║             Dilata el margen ante transitorios; NO es un potencial de        ║
║             Gromov–Witten ni una ecuación de Maurer–Cartan en Λ_K.           ║
║             Es un dead-zone / histéresis con filtración de escala.           ║
║                                                                              ║
║ ARQUITECTURA: In --F1→ Phase1 --F2→ Phase2 --F3→ Phase3 → dict               ║
║ Crowbar/GPIO SIMULADO. No hay acceso a silicio.                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

try:
    from scipy.sparse.linalg import eigsh
except ImportError:  # pragma: no cover
    eigsh = None  # type: ignore[assignment]

logger = logging.getLogger("APU.Physics.ThermalGradientLaws")

# ---------------------------------------------------------------------------
# Constantes metrológicas (Wilkinson / IEEE-754 binary64)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_SPD_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_SPD_REL_TOL: Final[float] = 1e-8
_ENTROPY_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_LANDAUER_LN2: Final[float] = float(np.log(2.0))

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = 380.0
_CROWBAR_LATENCY_CEIL_NS: Final[float] = 420.0

_DEFAULT_SAFETY_MARGIN: Final[float] = 1.0
_DEGRADED_FOURIER_THRESHOLD: Final[float] = 1e-6
_DEGRADED_CONDITION: Final[float] = 1e8
_KRYLOV_FULL_EIGH_CAP: Final[int] = 24
_KRYLOV_K_DEFAULT: Final[int] = 8
_NOVIKOV_WINDOW: Final[int] = 16
_NOVIKOV_RHO: Final[float] = 0.85
_CD_TAU0: Final[float] = 1e-4


# =============================================================================
# Retículo de Heyting
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)


# =============================================================================
# Cartas y contratos de fase
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""
    Factorización / carta extremal de un SPD.

    Si `lanczos_used`, `eigenvalues` contiene solo los k extremos (both-ends)
    y `eigenvectors` las bases asociadas; `inv_metric` puede ser None y la
    inversa se aplica por CG. `deflated_operator` es K de rango-k + ridge
    cuando `rank_reduced` es verdadero.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inv_metric: Optional[np.ndarray]
    sqrt_metric: Optional[np.ndarray]
    condition_number: float
    operator_norm: float
    frobenius_norm: float
    spectral_gap: float
    volume_density: float
    regularized: bool
    tikhonov_alpha: float
    kernel_dimension: int
    min_eigenvalue: float
    max_eigenvalue: float
    lanczos_used: bool
    rank_reduced: bool
    krylov_k: int
    deflated_operator: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))
        if self.inv_metric is not None:
            object.__setattr__(self, "inv_metric", _freeze_array(self.inv_metric))
        if self.sqrt_metric is not None:
            object.__setattr__(self, "sqrt_metric", _freeze_array(self.sqrt_metric))
        if self.deflated_operator is not None:
            object.__setattr__(self, "deflated_operator", _freeze_array(self.deflated_operator))


@dataclass(frozen=True, slots=True)
class ItohAbeChart:
    r"""Carta del gradiente discreto de Itoh–Abe de Ē(p) = pᵀ κ p."""

    discrete_gradient: np.ndarray
    tellegen_residual: float
    pairing: float
    dirichlet_energy: float
    fourier_pairing: float
    coordinate_order_defect: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "discrete_gradient", _freeze_array(self.discrete_gradient))


@dataclass(frozen=True, slots=True)
class TransientFiltration:
    r"""
    Envolvente de transitorio (surrogate de valuación tipo Novikov).

    ν ≥ 0 crece con el contenido de alta frecuencia de |dT|_g.
    τ_CD = −τ₀·safety·exp(−ν)  (más negativo ⇒ margen más holgado).
    """

    envelope: float
    high_frequency: float
    valuation: float
    tau_cd: float
    window_ultrametric: float
    mc_polynomial: float


@dataclass(frozen=True, slots=True)
class Phase1ThermalState:
    K_reg: np.ndarray
    grad_T: np.ndarray
    T_sys: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    conductivity_chart: Optional[SpectralChart] = None
    weyl_skew_residual: float = 0.0
    temperature_clamped: bool = False
    third_law_flag: bool = False
    onsager_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "K_reg", _freeze_array(self.K_reg))
        object.__setattr__(self, "grad_T", _freeze_array(self.grad_T))


@dataclass(frozen=True, slots=True)
class Phase2ThermalState:
    Q_vector: np.ndarray
    Q_down: np.ndarray
    clausius_duhem_residual: float
    thermal_entropy_production: float
    carnot_efficiency: float
    carnot_density: float
    entropy_production_rate: float
    exergy_potential: float
    gouy_stodola: float
    fourier_residual: float
    dissipation_identity_residual: float
    dirichlet_energy: float
    csmd_error: float
    landauer_gap: float
    pairing_q_dT: float
    tellegen_residual: float
    T_hot: float
    T_cold: float
    itoh_abe: Optional[ItohAbeChart] = None
    metric_chart: Optional[SpectralChart] = None
    conductivity_up: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "Q_vector", _freeze_array(self.Q_vector))
        object.__setattr__(self, "Q_down", _freeze_array(self.Q_down))
        if self.conductivity_up is not None:
            object.__setattr__(self, "conductivity_up", _freeze_array(self.conductivity_up))


@dataclass(frozen=True, slots=True)
class Phase3HeytingDecision:
    heyting_verdict: str
    heyting_rank: int
    heyting_score: float
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    conservation_residual: float
    adaptive_tau_cd: float
    transient_valuation: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# FASE 1 — INGESTA + LANCZOS / HIGHAM
# =============================================================================
class Phase1ThermalIngestionMixin:
    r"""
    FASE 1 — OBSERVE + ORIENT.

    Proyección π_Sym(K) = ½(K+Kᵀ) y regularización de Higham sobre {A ⪰ α I},
    con espectro extremal por Lanczos–Krylov cuando n > n_krylov.

    Codominio del último método: `Phase1ThermalState`
    = dominio de `phase2_simulate_carnot_from_phase1`.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = _DEFAULT_SAFETY_MARGIN,
        *,
        rng_seed: Optional[int] = None,
        krylov_k: int = _KRYLOV_K_DEFAULT,
        rank_reduce: bool = False,
        novikov_rho: float = _NOVIKOV_RHO,
        cd_tau0: float = _CD_TAU0,
    ) -> None:
        self._n = self._validate_positive_int("dimension_n", dimension_n)
        self._safety_margin = self._validate_positive_finite("safety_margin", safety_margin)
        self._reg = _WILKINSON_FLOOR
        self._krylov_k = int(max(1, krylov_k))
        self._rank_reduce = bool(rank_reduce)
        self._novikov_rho = float(min(max(novikov_rho, 0.0), 1.0 - _MACHINE_EPS))
        self._cd_tau0 = self._validate_positive_finite("cd_tau0", cd_tau0)
        self._rng = np.random.default_rng(rng_seed)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False
        self._transient_envelope = 0.0
        self._transient_window: Deque[float] = deque(maxlen=_NOVIKOV_WINDOW)

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_positive_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente mayor que cero.")
        return value_f

    @staticmethod
    def _validate_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f):
            raise ValueError(f"{name} debe ser finito.")
        return value_f

    def _validate_temperature(self, value: Any, name: str = "T_sys") -> float:
        return self._validate_finite(name, value)

    @staticmethod
    def _ensure_finite_array(arr: np.ndarray, name: str) -> None:
        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise ValueError(f"{name} contiene tipos no numéricos.") from exc
        if not finite:
            raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")

    def _as_numeric_vector(
        self,
        values: Any,
        name: str,
        *,
        expected_size: Optional[int] = None,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim == 0:
            raw = raw.reshape(1)
        elif raw.ndim > 1:
            raw = raw.ravel()
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(f"{name} posee componente imaginaria no despreciable.")
            raw = raw.real
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a vector float64.") from exc
        self._ensure_finite_array(arr, name)
        if expected_size is not None and arr.size != expected_size:
            raise ValueError(
                f"{name} debe tener tamaño estricto {expected_size}. Se recibió {arr.size}."
            )
        return arr

    def _as_numeric_matrix(
        self,
        values: Any,
        name: str,
        *,
        expected_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(f"{name} posee componente imaginaria no despreciable.")
            raw = raw.real
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a matriz float64.") from exc
        self._ensure_finite_array(arr, name)
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(
                f"{name} debe tener forma estricta {expected_shape}. Se recibió {arr.shape}."
            )
        return arr

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def kahan_sum(self, arr: np.ndarray) -> float:
        r"""Kahan–Babuška–Neumaier. |fl(∑x)−∑x| ≤ (2u+O(u²)) ∑|x|."""
        vec = self._as_numeric_vector(arr, "arr")
        return float(np.real(self._neumaier_sum(vec)))

    @staticmethod
    def _weyl_sym_project(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """π_Sym(K)=½(K+Kᵀ). Residuo Onsager = ‖K−π(K)‖_F."""
        sym = 0.5 * (matrix + matrix.T)
        residual = float(la.norm(matrix - sym, ord="fro"))
        return sym, residual

    def _lanczos_both_ends(
        self,
        spd: np.ndarray,
        name: str,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        r"""
        Autovalores extremos por Lanczos simétrico (ARPACK both-ends).

        k se recorta a [2, n−1]. Si ARPACK no está o no converge, se
        degrada a `eigh` denso (O(n³)).
        """
        n = int(spd.shape[0])
        k_use = int(max(2, min(k, n - 1)))
        if eigsh is None or n <= _KRYLOV_FULL_EIGH_CAP:
            evals, evecs = la.eigh(spd, check_finite=True)
            return np.real(evals), evecs, False
        try:
            evals, evecs = eigsh(spd, k=k_use, which="BE", maxiter=max(4 * n, 40 * k_use))
            order = np.argsort(np.real(evals))
            return np.real(evals)[order], evecs[:, order], True
        except Exception as exc:
            logger.warning(
                "Lanczos/ARPACK falló sobre %s (%s). Fallback a eigh denso.",
                name, exc,
            )
            evals, evecs = la.eigh(spd, check_finite=True)
            return np.real(evals), evecs, False

    def _low_rank_ridge(
        self,
        evals: np.ndarray,
        evecs: np.ndarray,
        n: int,
        gamma: float,
        take_largest: bool,
        k: int,
    ) -> np.ndarray:
        r"""
        K_defl = Σ_{i∈I} λ_i v_i v_iᵀ + γ (I − P_I).

        I = k autovalores de mayor |λ| si `take_largest` (modos dominantes
        de conductividad); el complemento se sustituye por el piso γ.
        Esto ES una aproximación: el espectro medio se pierde. Se usa
        sólo si `rank_reduce=True`.
        """
        k_use = int(max(1, min(k, evals.size, n)))
        if take_largest:
            idx = np.argsort(np.abs(evals))[-k_use:]
        else:
            idx = np.argsort(evals)[:k_use]
        V = np.asarray(evecs[:, idx], dtype=np.float64)
        lam = np.maximum(np.asarray(evals[idx], dtype=np.float64), gamma)
        P_term = (V * lam) @ V.T
        gram = V @ V.T
        ridge = gamma * (np.eye(n, dtype=np.float64) - gram)
        K = P_term + ridge
        return 0.5 * (K + K.T)

    def _factorize_spd(
        self,
        matrix: np.ndarray,
        name: str,
        *,
        mode: str = "floor",
        allow_negative: bool = False,
        rank_reduce: Optional[bool] = None,
    ) -> SpectralChart:
        r"""
        Carta espectral.

        mode="floor"    : Higham λ ↦ max(λ, α) vía shift (α−λ_min)_+ I.
        mode="tikhonov" : (A + α I)⁻¹ cuando hay base completa.

        Si Lanczos no trae la base completa, la inversa densa se omite
        (`inv_metric=None`) y Fase 2 usará CG / el operador original.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")
        if matrix.shape[0] != self._n:
            raise ValueError(f"{name} debe ser {self._n} × {self._n}, se recibió {matrix.shape}")

        sym, _ = self._weyl_sym_project(np.asarray(matrix, dtype=np.float64))
        do_reduce = self._rank_reduce if rank_reduce is None else bool(rank_reduce)

        evals, evecs, lanczos = self._lanczos_both_ends(sym, name, self._krylov_k)
        self._ensure_finite_array(evals, f"{name}.eigenvalues")
        if evals.size == 0:
            raise ValueError(f"{name} no puede ser una matriz vacía.")

        min_eig = float(np.min(evals))
        max_eig = float(np.max(evals))
        max_abs = float(max(abs(min_eig), abs(max_eig)))
        tol = max(_SPD_ABS_TOL, _SPD_REL_TOL * max(1.0, max_abs))
        if min_eig < -tol and not allow_negative:
            raise ValueError(
                f"{name} no es semidefinido positivo dentro de tolerancia. "
                f"Autovalor mínimo: {min_eig:.18e}, tolerancia: {tol:.18e}."
            )

        regularized = bool(min_eig <= self._reg)
        if regularized:
            logger.warning(
                "%s mal condicionada o casi singular (λ_min=%.3e); α=%.3e, lanczos=%s.",
                name, min_eig, self._reg, lanczos,
            )

        full_basis = (not lanczos) or (evals.size == self._n)
        shift = max(self._reg - min_eig, 0.0) if mode == "floor" else 0.0
        tik_shift = self._reg if mode == "tikhonov" else 0.0

        deflated: Optional[np.ndarray] = None
        if do_reduce and evals.size >= 1:
            deflated = self._low_rank_ridge(
                evals, evecs, self._n, max(self._reg, _WILKINSON_FLOOR),
                take_largest=True, k=self._krylov_k,
            )

        working = sym + (shift + tik_shift) * np.eye(self._n, dtype=np.float64)
        working = 0.5 * (working + working.T)
        if deflated is not None:
            working = deflated

        inv_metric: Optional[np.ndarray] = None
        sqrt_metric: Optional[np.ndarray] = None
        volume = float("nan")
        frob = float(la.norm(working, ord="fro"))
        kernel_dim = 0

        if full_basis and deflated is None:
            shifted = np.maximum(evals + shift + tik_shift, _WILKINSON_FLOOR)
            inv_metric = (evecs * (1.0 / shifted)) @ evecs.T
            inv_metric = 0.5 * (inv_metric + inv_metric.T)
            sqrt_metric = (evecs * np.sqrt(shifted)) @ evecs.T
            sqrt_metric = 0.5 * (sqrt_metric + sqrt_metric.T)
            log_vol = 0.5 * float(self.kahan_sum(np.log(np.maximum(np.abs(shifted), _WILKINSON_FLOOR))))
            volume = float(np.exp(log_vol))
            frob = float(np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2))))
            kernel_dim = int(np.sum(np.abs(evals) <= max(self._reg, _WILKINSON_FLOOR)))
        elif deflated is not None:
            try:
                d_e, d_v = la.eigh(working, check_finite=True)
                d_e = np.maximum(np.real(d_e), _WILKINSON_FLOOR)
                inv_metric = (d_v * (1.0 / d_e)) @ d_v.T
                inv_metric = 0.5 * (inv_metric + inv_metric.T)
                sqrt_metric = (d_v * np.sqrt(d_e)) @ d_v.T
                sqrt_metric = 0.5 * (sqrt_metric + sqrt_metric.T)
            except la.LinAlgError:
                inv_metric = None
                sqrt_metric = None

        lam_floor = max(min_eig + shift + tik_shift, _WILKINSON_FLOOR)
        lam_ceil = max(max_eig + shift + tik_shift, lam_floor)
        cond = float(lam_ceil / lam_floor)
        op_norm = float(lam_ceil)
        gap = float(max(lam_ceil - lam_floor, 0.0)) if evals.size >= 2 else float(lam_floor)

        return SpectralChart(
            eigenvalues=evals,
            eigenvectors=evecs,
            inv_metric=inv_metric,
            sqrt_metric=sqrt_metric,
            condition_number=cond,
            operator_norm=op_norm,
            frobenius_norm=frob,
            spectral_gap=gap,
            volume_density=volume,
            regularized=regularized,
            tikhonov_alpha=self._reg,
            kernel_dimension=kernel_dim,
            min_eigenvalue=min_eig,
            max_eigenvalue=max_eig,
            lanczos_used=lanczos,
            rank_reduced=bool(deflated is not None),
            krylov_k=int(min(self._krylov_k, evals.size)),
            deflated_operator=deflated,
        )

    def _apply_inverse_spd(
        self,
        chart: SpectralChart,
        original: np.ndarray,
        vec: np.ndarray,
    ) -> np.ndarray:
        """Aplica A⁻¹: base espectral si existe, si no CG sobre A+αI."""
        if chart.inv_metric is not None:
            return np.asarray(chart.inv_metric, dtype=np.float64) @ vec
        A = 0.5 * (np.asarray(original, dtype=np.float64) + np.asarray(original, dtype=np.float64).T)
        A = A + self._reg * np.eye(self._n, dtype=np.float64)
        try:
            x, *_ = la.cg(A, vec, atol=1e-12, maxiter=max(4 * self._n, 64)) if hasattr(la, "cg") else (None,)
        except Exception:
            x = None
        if x is None:
            x = la.solve(A, vec, assume_a="pos")
        return np.asarray(x, dtype=np.float64)

    def _regularize_spd_matrix(
        self,
        matrix: np.ndarray,
        name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any], SpectralChart]:
        chart = self._factorize_spd(matrix, name, mode="floor")
        if chart.deflated_operator is not None:
            reconstructed = np.array(chart.deflated_operator, copy=True)
        else:
            shift = max(self._reg - chart.min_eigenvalue, 0.0)
            reconstructed = 0.5 * (matrix + matrix.T) + shift * np.eye(self._n, dtype=np.float64)
        diagnostics = {
            "min_eigenvalue": chart.min_eigenvalue,
            "max_eigenvalue": chart.max_eigenvalue,
            "regularization_floor": self._reg,
            "spd_tolerance": max(
                _SPD_ABS_TOL,
                _SPD_REL_TOL * max(1.0, abs(chart.max_eigenvalue), abs(chart.min_eigenvalue)),
            ),
            "condition_number": chart.condition_number,
            "spectral_gap": chart.spectral_gap,
            "lanczos_used": chart.lanczos_used,
            "rank_reduced": chart.rank_reduced,
            "krylov_k": chart.krylov_k,
        }
        return reconstructed, diagnostics, chart

    def phase1_ingest_and_purify(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
    ) -> Phase1ThermalState:
        r"""
        [FASE 1 — ÚLTIMO MORFISMO]

        Codominio: `Phase1ThermalState`
        = dominio de `phase2_simulate_carnot_from_phase1`.
        """
        K = self._as_numeric_matrix(K_raw, "K_raw", expected_shape=(self._n, self._n))
        grad_T = self._as_numeric_vector(grad_T_raw, "grad_T_raw", expected_size=self._n)
        T = self._validate_temperature(T_sys)

        diagnostics: Dict[str, Any] = {"temperature_clamped": False, "third_law_flag": False}
        third_law = bool(T <= 0.0)
        clamped = bool(T < _WILKINSON_FLOOR)
        if clamped:
            diagnostics["temperature_clamped"] = True
            diagnostics["T_sys_raw"] = T
            if third_law:
                diagnostics["third_law_flag"] = True
                logger.warning(
                    "T_sys=%.6e ≤ 0: 3ª ley / Nernst. Se proyecta al piso de Wilkinson; Fase 3 vetará.",
                    T,
                )
            T_clamped = _WILKINSON_FLOOR
        else:
            T_clamped = T

        K_sym, onsager_res = self._weyl_sym_project(K)
        K_reg, spd_diagnostics, chart = self._regularize_spd_matrix(K_sym, "K_raw")
        diagnostics.update(spd_diagnostics)
        diagnostics["onsager_skew_residual"] = onsager_res

        return Phase1ThermalState(
            K_reg=K_reg,
            grad_T=grad_T,
            T_sys=float(T_clamped),
            diagnostics=diagnostics,
            conductivity_chart=chart,
            weyl_skew_residual=float(onsager_res),
            temperature_clamped=clamped,
            third_law_flag=third_law,
            onsager_residual=float(onsager_res),
        )

    def execute_phase_1_ingestion(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """[COMPATIBILIDAD 1.X — FASE 1]"""
        phase1 = self.phase1_ingest_and_purify(K_raw, grad_T_raw, T_sys)
        return np.array(phase1.K_reg, copy=True), np.array(phase1.grad_T, copy=True), phase1.T_sys


# =============================================================================
# FASE 2 — FOURIER + ITOH–ABE + CD + CARNOT
# Dominio = Phase1ThermalState.
# =============================================================================
class Phase2CarnotSimulationMixin(Phase1ThermalIngestionMixin):
    r"""
    FASE 2 — DECIDE FÍSICO.

    Itoh–Abe de Ē(p)=pᵀκp garantiza Tellegen:
        ⟨∇̄_IA Ē(0,p), p⟩ = Ē(p)
    y el emparejamiento de dualidad ⟨q,dT⟩ = −Ē(p) es geométricamente fiel
    (independiente del orden de redondeo de q^T ∇T plano).
    """

    @staticmethod
    def _validate_entropy_production(value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError("entropy_production_rate no debe ser booleano.")
        try:
            sigma = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("entropy_production_rate debe ser numérico.") from exc
        if not math.isfinite(sigma):
            raise ValueError("entropy_production_rate debe ser finito.")
        if sigma < -_ENTROPY_ABS_TOL:
            raise ValueError(
                "entropy_production_rate no puede ser negativo fuera de tolerancia."
            )
        return max(0.0, sigma)

    def _inverse_spd_matrix(self, matrix: np.ndarray, name: str) -> np.ndarray:
        chart = self._factorize_spd(matrix, name, mode="tikhonov")
        if chart.inv_metric is not None:
            return np.array(chart.inv_metric, copy=True)
        A = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
        A = A + self._reg * np.eye(self._n, dtype=np.float64)
        return np.array(la.inv(A), copy=True)

    def _raise_conductivity(
        self,
        K_reg: np.ndarray,
        metric_chart: SpectralChart,
        metric_tensor: np.ndarray,
        conductivity_type: str,
    ) -> np.ndarray:
        ginv = metric_chart.inv_metric
        if ginv is None:
            ginv = self._inverse_spd_matrix(metric_tensor, "metric_tensor")
        ginv = np.asarray(ginv, dtype=np.float64)
        K = np.asarray(K_reg, dtype=np.float64)
        kind = str(conductivity_type).strip().lower()
        if kind == "covariant":
            kappa = ginv @ K @ ginv
        elif kind == "contravariant":
            kappa = K
        elif kind == "mixed":
            kappa = K @ ginv
        elif kind in ("isotropic_metric", "isotropic"):
            k_scalar = float(np.trace(K) / max(self._n, 1))
            kappa = k_scalar * ginv
        else:
            raise ValueError(
                "conductivity_type debe ser 'covariant', 'contravariant', "
                "'mixed' o 'isotropic_metric'."
            )
        return 0.5 * (kappa + kappa.T)

    def _quadratic_energy(self, kappa: np.ndarray, p: np.ndarray) -> float:
        """Ē(p) = pᵀ κ p evaluado con KBN sobre p ⊙ (κp)."""
        Kp = np.asarray(kappa, dtype=np.float64) @ np.asarray(p, dtype=np.float64)
        return float(self.kahan_sum(np.asarray(p, dtype=np.float64) * Kp))

    def _itoh_abe_discrete_gradient(
        self,
        energy: Callable[[np.ndarray], float],
        x: np.ndarray,
        y: np.ndarray,
        analytic_partial: Optional[Callable[[np.ndarray, int], float]] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Gradiente discreto de Itoh–Abe (orden de coordenadas 0..n−1):

            z^{(0)} = y,   z^{(i)} = (x_1..x_i, y_{i+1}..y_n)
            (∇̄H)_i = [H(z^{(i)}) − H(z^{(i−1)})] / (x_i − y_i)

        Identidad de Tellegen (aritmética exacta):
            ⟨∇̄H(x,y), x−y⟩ = H(x) − H(y).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = x.size
        g = np.zeros(n, dtype=np.float64)
        z = y.copy()
        h_prev = float(energy(z))
        for i in range(n):
            z_next = z.copy()
            z_next[i] = x[i]
            h_next = float(energy(z_next))
            dx = float(x[i] - y[i])
            if abs(dx) > _WILKINSON_FLOOR:
                g[i] = (h_next - h_prev) / dx
            elif analytic_partial is not None:
                g[i] = float(analytic_partial(0.5 * (x + y), i))
            else:
                g[i] = 0.0
            z = z_next
            h_prev = h_next
        pairing = float(self.kahan_sum(g * (x - y)))
        tellegen = float(abs(pairing - (float(energy(x)) - float(energy(y)))))
        return g, tellegen

    def _gonzalez_midpoint_gradient(self, kappa: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""González: ∇̄Ē(x,y) = κ(x+y) para Ē(p)=pᵀκp (Hessiano constante)."""
        return np.asarray(kappa, dtype=np.float64) @ (np.asarray(x) + np.asarray(y))

    def _complex_step_dirichlet(
        self,
        kappa: np.ndarray,
        grad_T: np.ndarray,
        step: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""CSMD de I(p)=½ pᵀκp. Im(I(p+ih e_λ))/h = (κp)_λ. Conjugado prohibido."""
        K = np.asarray(kappa, dtype=np.float64)
        p = np.asarray(grad_T, dtype=np.complex128)
        sens = np.zeros(self._n, dtype=np.float64)
        for lam in range(self._n):
            pert = p.copy()
            pert[lam] += 1.0j * step
            value = 0.5 * pert @ K @ pert
            sens[lam] = float(np.imag(value) / step)
        return sens

    def phase2_simulate_carnot_from_phase1(
        self,
        phase1_state: Phase1ThermalState,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        *,
        conductivity_type: str = "covariant",
        length_scale: float = 1.0,
        ambient_temperature: Optional[float] = None,
        info_erasure_rate: float = 0.0,
    ) -> Phase2ThermalState:
        r"""
        [FASE 2 — ÚLTIMO MORFISMO]

        Dominio : `Phase1ThermalState`
        Codominio: `Phase2ThermalState` → dominio de Fase 3.

        Flujo constitutivo q = −κ ∇T.
        Pairing Tellegen: ⟨q,dT⟩_IA = −⟨∇̄_IA Ē(0,∇T), ∇T⟩ = −Ē(∇T).
        Φ = σ − ⟨q,dT⟩ / T²  ≥ 0.
        """
        if not isinstance(phase1_state, Phase1ThermalState):
            raise TypeError("phase1_state debe ser Phase1ThermalState.")

        K_reg = np.asarray(phase1_state.K_reg, dtype=np.float64)
        grad_T = np.asarray(phase1_state.grad_T, dtype=np.float64)
        T_sys = float(phase1_state.T_sys)
        self._ensure_finite_array(K_reg, "phase1_state.K_reg")
        self._ensure_finite_array(grad_T, "phase1_state.grad_T")
        if not math.isfinite(T_sys) or T_sys <= 0.0:
            raise ValueError("phase1_state.T_sys debe ser finita y positiva.")

        sigma = self._validate_entropy_production(entropy_production_rate)
        L = self._validate_positive_finite("length_scale", length_scale)
        Gamma = self._validate_finite("info_erasure_rate", info_erasure_rate)
        if Gamma < -_ENTROPY_ABS_TOL:
            raise ValueError("info_erasure_rate no puede ser negativo.")
        Gamma = max(0.0, Gamma)

        metric_chart = self._factorize_spd(metric_tensor, "metric_tensor", mode="tikhonov")
        kappa = self._raise_conductivity(K_reg, metric_chart, metric_tensor, conductivity_type)

        Q_vector = np.asarray(-kappa @ grad_T, dtype=np.float64)
        self._ensure_finite_array(Q_vector, "heat_flux_vector")

        g_mat = 0.5 * (
            np.asarray(metric_tensor, dtype=np.float64)
            + np.asarray(metric_tensor, dtype=np.float64).T
        )
        Q_down = np.asarray(g_mat @ Q_vector, dtype=np.float64)

        def energy_E(p: np.ndarray) -> float:
            return self._quadratic_energy(kappa, p)

        def partial_E(p: np.ndarray, i: int) -> float:
            Kp = kappa @ np.asarray(p, dtype=np.float64)
            return float(Kp[i] + Kp[i])  # ∂(pᵀκp)/∂p_i = 2 (κp)_i

        ia_grad, tellegen = self._itoh_abe_discrete_gradient(
            energy_E, grad_T, np.zeros(self._n, dtype=np.float64), analytic_partial=partial_E,
        )
        pairing_ia = float(-self.kahan_sum(ia_grad * grad_T))
        pairing_fourier = float(self.kahan_sum(Q_vector * grad_T))
        gonzalez = self._gonzalez_midpoint_gradient(kappa, grad_T, np.zeros(self._n))
        order_defect = float(la.norm(ia_grad - gonzalez))

        itoh_chart = ItohAbeChart(
            discrete_gradient=ia_grad,
            tellegen_residual=tellegen,
            pairing=pairing_ia,
            dirichlet_energy=0.5 * energy_E(grad_T),
            fourier_pairing=pairing_fourier,
            coordinate_order_defect=order_defect,
        )

        # Pairing físico de CD: Tellegen (fiel) con fallback Fourier si IA no finito.
        pairing = pairing_ia if math.isfinite(pairing_ia) else pairing_fourier

        ginv = metric_chart.inv_metric
        if ginv is None:
            ginv = self._inverse_spd_matrix(metric_tensor, "metric_tensor")
        dirichlet_g = float(np.real(grad_T @ np.asarray(ginv, dtype=np.float64) @ grad_T))
        dirichlet_g = max(dirichlet_g, 0.0)
        grad_norm_g = float(np.sqrt(dirichlet_g))

        T_squared = T_sys * T_sys
        conduction_term = pairing / T_squared
        phi = float(sigma - conduction_term)
        sigma_th = float(-pairing / T_squared)
        quadratic = float(energy_E(grad_T) / T_squared)
        identity_res = float(abs(sigma_th - quadratic))

        fourier_res = float(la.norm(Q_vector + kappa @ grad_T))
        csmd = self._complex_step_dirichlet(kappa, grad_T)
        csmd_err = float(la.norm(csmd - (kappa @ grad_T)))

        T_cold = T_sys
        T_hot = T_sys + L * grad_norm_g
        if math.isfinite(T_hot) and T_hot > T_cold > 0.0:
            carnot_efficiency = float(np.clip(1.0 - T_cold / T_hot, 0.0, 1.0))
        else:
            carnot_efficiency = 0.0
        carnot_density = float(grad_norm_g / T_sys)

        T0 = float(ambient_temperature) if ambient_temperature is not None else T_sys
        if not math.isfinite(T0) or T0 <= 0.0:
            raise ValueError("ambient_temperature debe ser finita y positiva.")
        gouy = float(T0 * phi)

        landauer_floor = float(Gamma * _LANDAUER_LN2)
        landauer_gap = float(sigma - landauer_floor)

        cd_gate = 1.0 if (math.isfinite(phi) and phi >= -_WILKINSON_FLOOR) else 0.0
        fourier_gate = float(1.0 / (1.0 + fourier_res / max(_DEGRADED_FOURIER_THRESHOLD, _WILKINSON_FLOOR)))
        exergy_potential = float(np.clip(carnot_efficiency * cd_gate * fourier_gate, 0.0, 1.0))

        diagnostics = dict(phase1_state.diagnostics)
        diagnostics.update(
            {
                "heat_flux_norm": float(la.norm(Q_vector)),
                "q_dot_grad_T": pairing,
                "q_dot_grad_T_fourier": pairing_fourier,
                "q_dot_grad_T_itoh_abe": pairing_ia,
                "tellegen_residual": tellegen,
                "itoh_abe_order_defect": order_defect,
                "entropy_production_rate": sigma,
                "thermal_entropy_production": sigma_th,
                "conduction_term": conduction_term,
                "clausius_duhem_phi": phi,
                "dirichlet_energy": dirichlet_g,
                "grad_norm_g": grad_norm_g,
                "fourier_residual": fourier_res,
                "dissipation_identity_residual": identity_res,
                "csmd_error": csmd_err,
                "T_hot": T_hot,
                "T_cold": T_cold,
                "length_scale": L,
                "carnot_efficiency": carnot_efficiency,
                "carnot_density": carnot_density,
                "exergy_potential": exergy_potential,
                "gouy_stodola": gouy,
                "landauer_floor": landauer_floor,
                "landauer_gap": landauer_gap,
                "conductivity_type": conductivity_type,
                "metric_condition": metric_chart.condition_number,
                "metric_lanczos": metric_chart.lanczos_used,
                "ambient_temperature": T0,
            }
        )

        phi_value = float(phi) if math.isfinite(phi) else (float(phi) if math.isinf(phi) else float("nan"))
        if not math.isfinite(pairing):
            diagnostics["heat_flux_inner_product_nonfinite"] = True
            phi_value = float("nan")

        return Phase2ThermalState(
            Q_vector=Q_vector,
            Q_down=Q_down,
            clausius_duhem_residual=phi_value,
            thermal_entropy_production=sigma_th,
            carnot_efficiency=carnot_efficiency,
            carnot_density=carnot_density,
            entropy_production_rate=sigma,
            exergy_potential=exergy_potential,
            gouy_stodola=gouy,
            fourier_residual=fourier_res,
            dissipation_identity_residual=identity_res,
            dirichlet_energy=dirichlet_g,
            csmd_error=csmd_err,
            landauer_gap=landauer_gap,
            pairing_q_dT=pairing,
            tellegen_residual=tellegen,
            T_hot=float(T_hot),
            T_cold=float(T_cold),
            itoh_abe=itoh_chart,
            metric_chart=metric_chart,
            conductivity_up=kappa,
            diagnostics=diagnostics,
        )

    def execute_phase_2_carnot_simulation(
        self,
        K_reg: np.ndarray,
        grad_T: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """[COMPATIBILIDAD 1.X — FASE 2]"""
        K = self._as_numeric_matrix(K_reg, "K_reg", expected_shape=(self._n, self._n))
        grad = self._as_numeric_vector(grad_T, "grad_T", expected_size=self._n)
        T = self._validate_temperature(T_sys)
        if T < _WILKINSON_FLOOR:
            T = _WILKINSON_FLOOR
        phase1_state = Phase1ThermalState(
            K_reg=K, grad_T=grad, T_sys=T, diagnostics={"source": "compatibility_api"},
        )
        phase2_state = self.phase2_simulate_carnot_from_phase1(
            phase1_state=phase1_state,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
        )
        return (
            np.array(phase2_state.Q_vector, copy=True),
            phase2_state.clausius_duhem_residual,
            phase2_state.carnot_efficiency,
            phase2_state.entropy_production_rate,
            phase2_state.exergy_potential,
        )


# =============================================================================
# FASE 3 — UMBRAL ADAPTATIVO + HEYTING + CROWBAR SIMULADO
# Dominio = Phase2ThermalState.
# =============================================================================
class Phase3HeytingVetoMixin(Phase2CarnotSimulationMixin):
    r"""
    FASE 3 — ACT.

    Umbral CD adaptativo. Sea x = |dT|_g (carnot_density · T = |dT|_g):

        s_t = ρ s_{t−1} + (1−ρ) x          (envolvente lenta)
        b_t = |x − s_t|                    (contenido rápido)
        ν_t = log(1 + b_t / (s_t + ε))     (valuación-surrogate)
        u_t = max_{w∈ventana} |log(x_{t−j}+ε) − log(x_{t−j−1}+ε)|
        τ_CD(t) = −τ₀ · safety · exp(−ν_t)

    Ante transitorios (ν grande) el piso se hace más negativo y no veta
    un spike de Φ numérico. Un desequilibrio persistente (ν→0 y Φ<τ₀)
    sí colapsa a VETOED.

    El polinomio m₁(b)+m₂(b²)+m₃(b³) es un *marcador* de amplitud, no
    un potencial de Gromov–Witten ni una MC-ecuación en el anillo de Novikov.
    """

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def _act_thermal_interlock(self, verdict: str) -> Tuple[bool, float]:
        normalized_verdict = str(verdict).strip().upper()
        if normalized_verdict != HeytingVerdict.VETOED.value:
            return False, 0.0
        swapped = self._cas_interlock(expected=False, desired=True)
        if not swapped:
            logger.warning(
                "CAS: el interlock ya estaba enclavado. Se reconoce el veto "
                "y se registra la latencia de actuación SIMULADA."
            )
        jitter = float(self._rng.normal(loc=0.0, scale=3.5))
        actuation_latency_ns = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )
        logger.critical(
            "RUPTURA DE CLAUSIUS–DUHEM / 3ª LEY / FOURIER. "
            "Disyuntor Crowbar BT151 [GPIO14] SIMULADO. Latencia ficticia: %.2f ns.",
            actuation_latency_ns,
        )
        return True, actuation_latency_ns

    def _update_transient_filtration(self, grad_norm_g: float) -> TransientFiltration:
        x = float(max(grad_norm_g, 0.0))
        with self._interlock_lock:
            rho = self._novikov_rho
            s = rho * self._transient_envelope + (1.0 - rho) * x
            self._transient_envelope = s
            self._transient_window.append(x)
            window = tuple(self._transient_window)
        hf = abs(x - s)
        valuation = float(math.log1p(hf / (s + _WILKINSON_FLOOR)))
        ultra = 0.0
        if len(window) >= 2:
            logs = [math.log(v + _WILKINSON_FLOOR) for v in window]
            ultra = float(max(abs(logs[j] - logs[j - 1]) for j in range(1, len(logs))))
        b = hf
        mc_poly = float(abs(b + 0.5 * b * b + (1.0 / 6.0) * b * b * b))
        tau = float(-self._cd_tau0 * self._safety_margin * math.exp(-valuation))
        return TransientFiltration(
            envelope=float(s),
            high_frequency=float(hf),
            valuation=valuation,
            tau_cd=tau,
            window_ultrametric=ultra,
            mc_polynomial=mc_poly,
        )

    def _phase3_conservation_audit(self, phase2_state: Phase2ThermalState) -> float:
        phi = phase2_state.clausius_duhem_residual
        if math.isfinite(phi):
            cd_inconsistent = float(-phi) if phi < -_WILKINSON_FLOOR else 0.0
        else:
            cd_inconsistent = 1.0
        eta = phase2_state.carnot_efficiency
        eta_res = 0.0 if 0.0 <= eta <= 1.0 + 1e-12 else 1.0
        tellegen = phase2_state.tellegen_residual
        terms = np.asarray(
            [
                phase2_state.fourier_residual,
                phase2_state.dissipation_identity_residual,
                phase2_state.csmd_error,
                eta_res,
                cd_inconsistent,
                max(0.0, -phase2_state.dirichlet_energy),
                max(0.0, -phase2_state.landauer_gap) if math.isfinite(phase2_state.landauer_gap) else 1.0,
                tellegen if math.isfinite(tellegen) else 1.0,
            ],
            dtype=np.float64,
        )
        return float(np.real(self._neumaier_sum(terms)))

    def _classify_thermal_heyting(
        self,
        phase2_state: Phase2ThermalState,
        filtration: TransientFiltration,
    ) -> Tuple[HeytingVerdict, float, Tuple[str, ...], Tuple[str, ...]]:
        phi = phase2_state.clausius_duhem_residual
        eta = phase2_state.carnot_efficiency
        allowed_cd_floor = float(filtration.tau_cd)

        veto: List[str] = []
        degraded: List[str] = []

        if not math.isfinite(phi):
            veto.append("clausius_duhem_residual_nonfinite")
            cd_score = 0.0
        elif phi < allowed_cd_floor:
            veto.append("clausius_duhem_second_law_violated")
            cd_score = 0.0
        elif phi < 0.0:
            degraded.append("clausius_duhem_residual_negative_numerical")
            cd_score = 0.5
        else:
            cd_score = 1.0

        if not math.isfinite(eta) or eta < 0.0 or eta > 1.0 + 1e-12:
            veto.append("carnot_efficiency_out_of_physical_range")
            eta_score = 0.0
        else:
            eta_score = 1.0

        fourier_res = phase2_state.fourier_residual
        if not math.isfinite(fourier_res) or fourier_res > 1e-3 * self._safety_margin:
            veto.append("fourier_constitutive_broken")
            fourier_score = 0.0
        elif fourier_res > _DEGRADED_FOURIER_THRESHOLD:
            degraded.append("fourier_residual_degraded")
            fourier_score = float(np.exp(-fourier_res / _DEGRADED_FOURIER_THRESHOLD))
        else:
            fourier_score = float(np.exp(-fourier_res / max(_DEGRADED_FOURIER_THRESHOLD, _WILKINSON_FLOOR)))

        csmd = phase2_state.csmd_error
        if not math.isfinite(csmd) or csmd > 1e-8:
            veto.append("csmd_identity_broken")
            csmd_score = 0.0
        else:
            csmd_score = float(np.exp(-csmd / 1e-12))

        if phase2_state.landauer_gap < -_ENTROPY_ABS_TOL:
            veto.append("landauer_bound_violated")
            landauer_score = 0.0
        else:
            landauer_score = 1.0

        ident = phase2_state.dissipation_identity_residual
        ident_score = float(np.exp(-ident / 1e-10)) if math.isfinite(ident) else 0.0
        if ident > 1e-6:
            degraded.append("dissipation_identity_degraded")

        tellegen = phase2_state.tellegen_residual
        if math.isfinite(tellegen) and tellegen > 1e-8:
            degraded.append("tellegen_identity_degraded")
        tellegen_score = float(np.exp(-tellegen / 1e-10)) if math.isfinite(tellegen) else 0.0

        cond = (
            float(phase2_state.metric_chart.condition_number)
            if phase2_state.metric_chart is not None
            else 1.0
        )
        if cond > 1e12:
            veto.append("metric_condition_explosive")
            cond_score = 0.0
        elif cond > _DEGRADED_CONDITION:
            degraded.append("metric_ill_conditioned")
            cond_score = 1.0 / (1.0 + np.log10(cond))
        else:
            cond_score = 1.0 / (1.0 + np.log10(max(cond, 1.0)))

        raw_T_clamped = bool(phase2_state.diagnostics.get("temperature_clamped", False))
        third_law = bool(phase2_state.diagnostics.get("third_law_flag", False))
        if third_law:
            veto.append("third_law_nonpositive_temperature")
            temp_score = 0.0
        elif raw_T_clamped:
            degraded.append("temperature_clamped_to_wilkinson_floor")
            temp_score = 0.5
        else:
            temp_score = 1.0

        pairing = phase2_state.pairing_q_dT
        if math.isfinite(pairing) and pairing > _WILKINSON_FLOOR:
            veto.append("anti_fourier_heat_up_gradient")
            pairing_score = 0.0
        else:
            pairing_score = 1.0

        meet = min(
            cd_score, eta_score, fourier_score, csmd_score, landauer_score,
            ident_score, tellegen_score, cond_score, temp_score, pairing_score,
        )

        if veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99 and not degraded:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90 and not degraded:
            verdict = HeytingVerdict.COHERENT
        elif degraded or meet >= 0.50:
            verdict = HeytingVerdict.DEGRADED if meet >= 0.50 else HeytingVerdict.VETOED
            if meet < 0.50 and not veto:
                veto.append("heyting_meet_collapsed")
                verdict = HeytingVerdict.VETOED
        else:
            verdict = HeytingVerdict.VETOED
        return verdict, float(meet), tuple(veto), tuple(degraded)

    def _classify_thermal_heyting_scalars(
        self,
        clausius_duhem_residual: float,
        carnot_efficiency: float,
        exergy_potential: float,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        allowed_cd_floor = -self._cd_tau0 * self._safety_margin
        veto_reasons: List[str] = []
        degraded_reasons: List[str] = []
        if not math.isfinite(clausius_duhem_residual):
            veto_reasons.append("clausius_duhem_residual_nonfinite")
        elif clausius_duhem_residual < allowed_cd_floor:
            veto_reasons.append("clausius_duhem_residual_below_allowed_floor")
        elif clausius_duhem_residual < 0.0:
            degraded_reasons.append("clausius_duhem_residual_negative")
        if not math.isfinite(carnot_efficiency):
            veto_reasons.append("carnot_efficiency_nonfinite")
        elif carnot_efficiency < 0.0 or carnot_efficiency > 1.0:
            veto_reasons.append("carnot_efficiency_out_of_physical_range")
        if not math.isfinite(exergy_potential):
            degraded_reasons.append("exergy_potential_nonfinite")
        if veto_reasons:
            verdict = HeytingVerdict.VETOED.value
        elif degraded_reasons:
            verdict = HeytingVerdict.DEGRADED.value
        else:
            verdict = HeytingVerdict.COHERENT.value
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def phase3_evaluate_heyting_from_phase2(
        self,
        phase2_state: Phase2ThermalState,
    ) -> Phase3HeytingDecision:
        r"""
        [FASE 3 — ÚLTIMO MORFISMO: ACT]

        Dominio: `Phase2ThermalState`
        ← continuación formal de `phase2_simulate_carnot_from_phase1`.
        """
        if not isinstance(phase2_state, Phase2ThermalState):
            raise TypeError("phase2_state debe ser Phase2ThermalState.")

        grad_norm = float(math.sqrt(max(phase2_state.dirichlet_energy, 0.0)))
        filtration = self._update_transient_filtration(grad_norm)
        verdict, score, veto_reasons, degraded_reasons = self._classify_thermal_heyting(
            phase2_state, filtration,
        )
        conservation = self._phase3_conservation_audit(phase2_state)
        interlock_fired, latency = self._act_thermal_interlock(verdict.value)

        diagnostics = dict(phase2_state.diagnostics)
        diagnostics.update(
            {
                "heyting_verdict": verdict.value,
                "heyting_rank": verdict.rank,
                "heyting_score": score,
                "veto_reasons": veto_reasons,
                "degraded_reasons": degraded_reasons,
                "hardware_interlock_fired": interlock_fired,
                "actuation_latency_ns": latency,
                "conservation_residual": conservation,
                "adaptive_tau_cd": filtration.tau_cd,
                "transient_valuation": filtration.valuation,
                "transient_envelope": filtration.envelope,
                "transient_high_frequency": filtration.high_frequency,
                "window_ultrametric": filtration.window_ultrametric,
                "mc_polynomial_surrogate": filtration.mc_polynomial,
            }
        )
        logger.debug(
            "Fase 3: veredicto=%s score=%.4f Φ=%.6e τ_CD=%.3e ν=%.3e cons=%.3e",
            verdict.value, score, phase2_state.clausius_duhem_residual,
            filtration.tau_cd, filtration.valuation, conservation,
        )
        return Phase3HeytingDecision(
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            heyting_score=score,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            conservation_residual=conservation,
            adaptive_tau_cd=filtration.tau_cd,
            transient_valuation=filtration.valuation,
            diagnostics=diagnostics,
        )

    def execute_phase_3_heyting_veto(
        self,
        clausius_duhem_residual: float,
        carnot_efficiency: float,
        exergy_potential: float,
    ) -> Dict[str, Any]:
        """[COMPATIBILIDAD 1.X — FASE 3]. Ya NO veta exergía baja."""
        verdict, veto_reasons, degraded_reasons = self._classify_thermal_heyting_scalars(
            clausius_duhem_residual=float(clausius_duhem_residual),
            carnot_efficiency=float(carnot_efficiency),
            exergy_potential=float(exergy_potential),
        )
        interlock_fired, latency = self._act_thermal_interlock(verdict)
        return {
            "heyting_verdict": verdict,
            "clausius_duhem_residual": float(clausius_duhem_residual),
            "carnot_efficiency": float(carnot_efficiency),
            "exergy_potential": float(exergy_potential),
            "hardware_interlock_fired": interlock_fired,
            "actuation_latency_ns": latency,
            "veto_reasons": veto_reasons,
            "degraded_reasons": degraded_reasons,
        }

    def execute_phase_3_hetying_veto(
        self,
        clausius_duhem_residual: float,
        carnot_efficiency: float,
        exergy_potential: float,
    ) -> Dict[str, Any]:
        """[COMPATIBILIDAD 1.X — typo original conservado]."""
        return self.execute_phase_3_heyting_veto(
            clausius_duhem_residual=clausius_duhem_residual,
            carnot_efficiency=carnot_efficiency,
            exergy_potential=exergy_potential,
        )


# =============================================================================
# CLASE PÚBLICA — T = Act ∘ Decide ∘ Observe
# =============================================================================
class ThermalGradientLaws(Phase3HeytingVetoMixin):
    r"""
    Resolvedor de gradientes de calor sobre (M, g).

        In --F1→ Phase1ThermalState --F2→ Phase2ThermalState
           --F3→ Phase3HeytingDecision → dict
    """

    def execute_thermal_cycle(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        *,
        conductivity_type: str = "covariant",
        length_scale: float = 1.0,
        ambient_temperature: Optional[float] = None,
        info_erasure_rate: float = 0.0,
    ) -> Dict[str, Any]:
        phase1 = self.phase1_ingest_and_purify(K_raw, grad_T_raw, T_sys)
        phase2 = self.phase2_simulate_carnot_from_phase1(
            phase1_state=phase1,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
            conductivity_type=conductivity_type,
            length_scale=length_scale,
            ambient_temperature=ambient_temperature,
            info_erasure_rate=info_erasure_rate,
        )
        phase3 = self.phase3_evaluate_heyting_from_phase2(phase2)
        return {
            "heyting_verdict": phase3.heyting_verdict,
            "clausius_duhem_residual": phase2.clausius_duhem_residual,
            "carnot_efficiency": phase2.carnot_efficiency,
            "exergy_potential": phase2.exergy_potential,
            "hardware_interlock_fired": phase3.hardware_interlock_fired,
            "actuation_latency_ns": phase3.actuation_latency_ns,
            "heat_flux_vector": np.array(phase2.Q_vector, copy=True),
            "T_sys_clamped": phase1.T_sys,
            "phase": "G_THERMAL_CYCLE_SUTURATED",
            "veto_reasons": phase3.veto_reasons,
            "degraded_reasons": phase3.degraded_reasons,
            "thermal_entropy_production": phase2.thermal_entropy_production,
            "gouy_stodola": phase2.gouy_stodola,
            "fourier_residual": phase2.fourier_residual,
            "csmd_error": phase2.csmd_error,
            "dirichlet_energy": phase2.dirichlet_energy,
            "carnot_density": phase2.carnot_density,
            "landauer_gap": phase2.landauer_gap,
            "conservation_residual": phase3.conservation_residual,
            "heyting_score": phase3.heyting_score,
            "T_hot": phase2.T_hot,
            "T_cold": phase2.T_cold,
            "tellegen_residual": phase2.tellegen_residual,
            "adaptive_tau_cd": phase3.adaptive_tau_cd,
            "transient_valuation": phase3.transient_valuation,
            "diagnostics": {
                "phase1": dict(phase1.diagnostics),
                "phase2": dict(phase2.diagnostics),
                "phase3": dict(phase3.diagnostics),
            },
        }


if __name__ == "__main__":
    print("Autocomprobación Thermal Gradient Laws v3.1 (Lanczos / Itoh–Abe / τ adaptativo)...")
    engine = ThermalGradientLaws(dimension_n=3, rng_seed=0)
    K = np.eye(3)
    G = np.eye(3)
    grad = np.array([1.0, 0.0, 0.0])
    stamp = engine.execute_thermal_cycle(K, grad, 300.0, G, 0.0)
    print("Veredicto:", stamp["heyting_verdict"], "score=", stamp["heyting_score"])
    print("Q (esperado [-1,0,0]):", stamp["heat_flux_vector"])
    print("Φ CD:", stamp["clausius_duhem_residual"])
    print("Tellegen:", stamp["tellegen_residual"])
    print("η_C (esperado 1/301):", stamp["carnot_efficiency"])
    print("τ_CD adaptativo:", stamp["adaptive_tau_cd"], "ν:", stamp["transient_valuation"])
    print("Fourier/CSMD:", stamp["fourier_residual"], stamp["csmd_error"])

    iso = engine.execute_thermal_cycle(K, np.zeros(3), 300.0, G, 0.0)
    print("Isotermo:", iso["heyting_verdict"], "η=", iso["carnot_efficiency"])

    spike = engine.execute_thermal_cycle(K, np.array([40.0, 0.0, 0.0]), 300.0, G, 0.0)
    print("Spike |∇T| (τ dilatado):", spike["adaptive_tau_cd"], spike["heyting_verdict"])

    big = ThermalGradientLaws(dimension_n=48, rng_seed=1, krylov_k=6, rank_reduce=False)
    Kb = np.eye(48) + 0.01 * np.ones((48, 48))
    Gb = np.eye(48)
    gb = np.zeros(48); gb[0] = 1.0
    st = big.execute_thermal_cycle(Kb, gb, 300.0, Gb, 0.0)
    print("n=48 Lanczos veredicto:", st["heyting_verdict"],
          "lanczos K:", st["diagnostics"]["phase1"].get("lanczos_used"),
          "κ:", st["diagnostics"]["phase1"].get("condition_number"))


__all__ = [
    "Phase1ThermalState",
    "Phase2ThermalState",
    "Phase3HeytingDecision",
    "Phase1ThermalIngestionMixin",
    "Phase2CarnotSimulationMixin",
    "Phase3HeytingVetoMixin",
    "ThermalGradientLaws",
    "SpectralChart",
    "ItohAbeChart",
    "TransientFiltration",
    "HeytingVerdict",
]