# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Gauge Projection Armory (Arsenal de Proyección de Calibre)          ║
║ Ruta   : app/agents/core/inmune_system/gauge_projection_armory.py            ║
║ Versión: 3.0.0-Doctoral-Hermitian-Higham-Duchi-DK-Connes-Heyting-Secure      ║
║                                                                              ║
║ ARQUITECTURA DE FASES ANIDADAS (morfismos, no meras secciones):              ║
║                                                                              ║
║   Fase 1  --η-->  Fase 2  --χ-->  Fase 3                                     ║
║   Observe+Orient    Decide (Ω₃)    Act + sello de telemetría                 ║
║                                                                              ║
║   η  = _phase1_terminal_morphism  = objeto inicial de la Fase 2              ║
║   χ  = clasificador de subobjetos = objeto inicial de la Fase 3              ║
║                                                                              ║
║ MATEMÁTICA (sin metáfora suelta):                                            ║
║   • Π_H(ρ) = (ρ+ρ†)/2     projector hermítico (Frobenius-óptimo).            ║
║   • Higham: λ ↦ π_Δ(λ)    proyección euclídea al símplice {x≥0, Σx=1}.       ║
║   • Tikhonov / despolarizante: Φ_γ(ρ) = (ρ+γI)/(1+nγ) con                    ║
║         γ = (μ−λ_min)/(1−nμ)   ⇒   λ_min(Φ_γ(ρ)) ≥ μ < 1/n.                  ║
║   • Seminorma tipo Connes: L(X) = ‖[ρ^{-1/2}, π(X)]‖_{B(ℋ)}.                ║
║     En la base propia: [D,X]_{ij} = (λ_i^{-1/2}−λ_j^{-1/2}) X_{ij}.          ║
║   • Daletskii–Krein: f^{[1]}(λ,μ) = (f(λ)−f(μ))/(λ−μ),                       ║
║         f(λ)=λ^{-1/2} ⇒ f^{[1]} = −1/(√(λμ)(√λ+√μ)),                         ║
║         Lip(f)|_{[μ,∞)} = ½ μ^{-3/2}.                                        ║
║   • Ω₃ = {COHERENT < DEGRADED < VETOED} cadena de Heyting.                   ║
║   • Fase 3: ISR *simulada*. No hay GPIO, ESP32 ni BT151 reales.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import scipy.linalg as la
from scipy.special import xlogy


logger = logging.getLogger("APU.Agents.GaugeProjectionArmory")

# =============================================================================
# Constantes espectrales (IEEE-754 float64, Wilkinson–Higham–Kahan)
# =============================================================================
_EPS64: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 50.0 * _EPS64
_DEFAULT_REGULARIZER: Final[float] = _WILKINSON_FLOOR
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

_HERMITIAN_RTOL: Final[float] = 1e-12
_PSD_EIGEN_FLOOR: Final[float] = -1e-12
_TRACE_ATOL: Final[float] = 1e-10
_SPECTRAL_RESIDUAL_TOL: Final[float] = 1e-10
_DIRAC_RESIDUAL_TOL: Final[float] = 1e-8
_SIMPLEX_SUM_TOL: Final[float] = 1e-12
_CONDITION_MARGINAL: Final[float] = 1.0e8
_SKEW_MARGINAL: Final[float] = 1e-8
_TIKHONOV_MOVE_MARGINAL: Final[float] = 1e-3
_DIMLESS_COMM_MARGINAL: Final[float] = 0.5  # fracción de la cota 2‖D‖‖X‖

_DEFAULT_LIPSCHITZ_TOLERANCE: Final[float] = 50.0
_DEFAULT_MU_FLOOR: Final[float] = 1e-12
_DEFAULT_RNG_SEED: Final[int] = 42
_DIGEST_ROUND_DECIMALS: Final[int] = 12


# =============================================================================
# Tipos ordinales del clasificador
# =============================================================================
class HeytingVerdict(str, Enum):
    """Cadena de Heyting Ω₃: COHERENT ⊥ < DEGRADED < VETOED ⊤."""

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def rank(self) -> int:
        return {self.COHERENT: 0, self.DEGRADED: 1, self.VETOED: 2}[self]


class HeytingOmega3:
    r"""
    Ω₃ = {0 = COHERENT, ½ = DEGRADED, 1 = VETOED} con el orden de cadena.

        a ∧ b = min(a, b)
        a ∨ b = max(a, b)
        a → b = ⊤  si a ≤ b,  else b
        ¬a    = a → ⊥
    """

    @classmethod
    def meet(cls, a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
        return a if a.rank <= b.rank else b

    @classmethod
    def join(cls, a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
        return a if a.rank >= b.rank else b

    @classmethod
    def implies(cls, a: HeytingVerdict, b: HeytingVerdict) -> HeytingVerdict:
        return HeytingVerdict.VETOED if a.rank <= b.rank else b

    @classmethod
    def neg(cls, a: HeytingVerdict) -> HeytingVerdict:
        return cls.implies(a, HeytingVerdict.COHERENT)

    @classmethod
    def join_all(cls, elems: Sequence[HeytingVerdict]) -> HeytingVerdict:
        acc = HeytingVerdict.COHERENT
        for e in elems:
            acc = cls.join(acc, e)
        return acc

    @classmethod
    def classify(
        cls,
        critical: Sequence[bool],
        marginal: Sequence[bool],
    ) -> HeytingVerdict:
        atoms: List[HeytingVerdict] = []
        for flag in critical:
            atoms.append(HeytingVerdict.VETOED if flag else HeytingVerdict.COHERENT)
        for flag in marginal:
            atoms.append(HeytingVerdict.DEGRADED if flag else HeytingVerdict.COHERENT)
        return cls.join_all(atoms)


# =============================================================================
# Utilidades C* / Banach / IEEE-754
# =============================================================================
def _as_c128(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.size == 0:
        raise ValueError(f"{name} está vacío")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o Inf")
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_f64(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.size == 0:
        raise ValueError(f"{name} está vacío")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene NaN o Inf")
    return np.array(arr, dtype=np.float64, copy=True)


def _freeze(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    out.setflags(write=False)
    return out


def _hermite(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def _scaled_tol(norm: float, rtol: float, atol: float) -> float:
    return float(max(atol, rtol * max(norm, 1.0)))


def _safe_xlogx(x: np.ndarray) -> np.ndarray:
    """x log x con convención 0 log 0 = 0."""
    xr = np.real(np.asarray(x, dtype=np.float64))
    return xlogy(xr, xr)


def _project_simplex(vec: np.ndarray) -> np.ndarray:
    r"""
    Proyección euclídea sobre el símplice de probabilidad
        Δ^{n−1} = {x ∈ ℝ^n : x ≥ 0,  1ᵀx = 1}
    (Duchi et al., 2008; es el contenido espectral de Higham para
    la matriz densidad Frobenius-más-cercana a un hermítico dado).
    """
    v = np.real(np.asarray(vec, dtype=np.float64)).copy()
    n = int(v.size)
    if n == 0:
        raise ValueError("símplice de dimensión 0")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    j = np.arange(1, n + 1, dtype=np.float64)
    cond = u + (1.0 - cssv) / j > 0.0
    if not np.any(cond):
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(v))] = 1.0
        return out
    rho = int(np.max(np.flatnonzero(cond)))
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(v - theta, 0.0)
    total = float(np.sum(projected))
    if total <= _WILKINSON_FLOOR:
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(v))] = 1.0
        return out
    projected /= total
    return projected


# =============================================================================
# Paquetes inmutables (objetos de las fases)
# =============================================================================
@dataclass(frozen=True, slots=True)
class PhaseOneArmoryPacket:
    """
    Objeto terminal de la Fase 1 / objeto inicial de la Fase 2.

    Contiene el projector hermítico, el estado de Higham (símplice),
    el estado Tikhonov (estrictamente PD) y los residuos certificados.
    """

    rho_weyl_toeplitz: np.ndarray
    rho_higham: np.ndarray
    rho_regularized: np.ndarray
    lambda_min: float
    lambda_min_regularized: float
    scale_factor: float
    depolarizing_gamma: float
    skew_hermitian_residual: float
    toeplitz_residual: float
    spectral_residual: float
    trace_residual: float
    condition_number: float
    higham_tikhonov_distance: float
    purity_regularized: float
    von_neumann_entropy: float

    def __post_init__(self) -> None:
        if self.rho_regularized.ndim != 2:
            raise ValueError("rho_regularized debe ser una matriz")
        if self.rho_weyl_toeplitz.shape != self.rho_regularized.shape:
            raise ValueError("Π_H(ρ) y ρ_μ tienen forma distinta")


@dataclass(frozen=True, slots=True)
class PhaseTwoArmoryPacket:
    """
    Objeto terminal de la Fase 2 / objeto inicial de la Fase 3.

    Transporta la seminorma de Connes, la constante de Daletskii–Krein,
    el veredicto Ω₃ y el clasificador de subobjetos χ (True ⇔ ⊤).
    """

    lipschitz_constant: float
    lipschitz_normalized: float
    lipschitz_dimensionless: float
    daleckii_krein_lipschitz: float
    dirac_residual: float
    allowed_tolerance: float
    heyting_verdict: str
    lipschitz_violation: bool
    dirac_violation: bool
    condition_violation: bool
    skew_violation: bool
    subobject_classifier: bool
    implication_coherent_to_verdict: str
    join_of_atoms: str


@dataclass(frozen=True, slots=True)
class ArmoryTelemetry:
    """
    Sello inmutable de telemetría (Fase 3).
    `to_dict()` exporta exactamente las claves de la v2 (compatibilidad ABI).
    """

    heyting_verdict: str
    lambda_min: float
    scale_factor: float
    lipschitz_constant: float
    allowed_tolerance: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    purified_operator: Optional[np.ndarray]
    daleckii_krein_lipschitz: float = 0.0
    lipschitz_dimensionless: float = 0.0
    condition_number: float = 0.0
    forensic_digest: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Interfaz original: ocho claves, ni una más."""
        return {
            "heyting_verdict": self.heyting_verdict,
            "lambda_min": self.lambda_min,
            "scale_factor": self.scale_factor,
            "lipschitz_constant": self.lipschitz_constant,
            "allowed_tolerance": self.allowed_tolerance,
            "hardware_interlock_fired": self.hardware_interlock_fired,
            "actuation_latency_ns": self.actuation_latency_ns,
            "purified_operator": self.purified_operator,
        }

    def to_dict_extended(self) -> Dict[str, Any]:
        base = self.to_dict()
        base.update(
            {
                "daleckii_krein_lipschitz": self.daleckii_krein_lipschitz,
                "lipschitz_dimensionless": self.lipschitz_dimensionless,
                "condition_number": self.condition_number,
                "forensic_digest": self.forensic_digest,
            }
        )
        return base


# =============================================================================
# Arsenal — tres fases anidadas
# =============================================================================
class GaugeProjectionArmory:
    r"""
    Aduana espectral entre el Patio de Armas y el Pretorio.

    Métodos públicos (contrato v2):

        kahan_sum(x)                          → Σ̂ xᵢ   (KBN)
        weyl_toeplitz_symmetrization(ρ)       → Π_H(ρ)
        higham_tikhonov_regularization(ρ, μ)  → (ρ_μ, λ_min, α)
        connes_daleckii_krein_filter(ρ, X)    → (L, τ, veredicto)
        execute_armory_cycle(ρ, X, μ)         → dict   (ABI v2)

    Fases anidadas:

        _phase1_observe_orient  →  η(_phase1_terminal_morphism)
                └──────────────────────────────────────────┐
        _phase2_decide            ←  η(packet)             ┘
                └─ χ
        _phase3_act_and_telemetry ← χ
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        lipschitz_tolerance: Optional[float] = None,
        rng_seed: Optional[int] = _DEFAULT_RNG_SEED,
    ) -> None:
        if dimension_n <= 0:
            raise ValueError(
                "La dimensión del espacio de Hilbert debe ser estrictamente positiva."
            )
        if safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser > 0")

        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = _DEFAULT_REGULARIZER
        self._lipschitz_tol: Final[float] = (
            float(lipschitz_tolerance)
            if lipschitz_tolerance is not None
            else _DEFAULT_LIPSCHITZ_TOLERANCE
        )
        if self._lipschitz_tol <= 0.0:
            raise ValueError("lipschitz_tolerance debe ser > 0")
        self._rng = np.random.default_rng(rng_seed)

        logger.debug(
            "GaugeProjectionArmory n=%d  margen=%.3f  τ_Lip=%.3f",
            self._n,
            self._safety_margin,
            self._lipschitz_tol,
        )

    # -------------------------------------------------------------------------
    # Accesores
    # -------------------------------------------------------------------------
    @property
    def dimension(self) -> int:
        return self._n

    @property
    def allowed_lipschitz_tolerance(self) -> float:
        return float(self._lipschitz_tol * self._safety_margin)

    # =========================================================================
    # Utilidades de precisión numérica
    # =========================================================================
    @staticmethod
    def kahan_sum(arr: np.ndarray) -> float:
        r"""
        Sumación compensada de Kahan–Babuška–Neumaier (KBN) sobre un
        vector real 1-D.  Neutraliza la deriva de Wilkinson en la mantisa:

            s ← s + x,   c ← c + ((s_old − t) + x)   (rama |s| ≥ |x|)

        Más estable que Kahan clásico cuando los sumandos cambian de signo
        y de magnitud.  Para n típico de M_n(ℂ) el error de `np.trace` ya
        es O(n ε‖x‖_∞); KBN se conserva como contrato público y como
        verificador independiente de Σ diag(ρ).
        """
        vec = np.asarray(arr)
        if vec.ndim != 1:
            raise ValueError(f"kahan_sum espera un vector 1-D; se recibió {vec.shape}")
        if vec.size == 0:
            return 0.0
        if not np.all(np.isfinite(vec)):
            raise ValueError("kahan_sum: el vector contiene NaN o Inf")
        total = 0.0
        compensator = 0.0
        for raw in vec:
            x = float(np.real(raw))
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return float(total + compensator)

    # =========================================================================
    # FASE 1 — OBSERVE + ORIENT
    #   1.1  validación C* de la matriz cruda
    #   1.2  projector hermítico Π_H  (API: weyl_toeplitz_symmetrization)
    #   1.3  espectro certificado (Weyl–Wilkinson)
    #   1.4  Higham: proyección al símplice (Duchi)
    #   1.5  Tikhonov / despolarizante con suelo μ *normalizado*
    #   1.6  residuos (skew, Toeplitz, traza, κ)
    #   1.7  morfismo terminal η  →  inicio formal de la Fase 2
    # =========================================================================
    def _validate_matrix_square(self, mat: np.ndarray, name: str) -> np.ndarray:
        arr = _as_c128(mat, name)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada; se recibió {arr.shape}")
        if arr.shape[0] != self._n:
            raise ValueError(
                f"dim({name})={arr.shape[0]} no coincide con n={self._n}"
            )
        return arr

    def _validate_hermitian(
        self,
        mat: np.ndarray,
        name: str,
        tolerance: float = _HERMITIAN_RTOL,
    ) -> np.ndarray:
        arr = self._validate_matrix_square(mat, name)
        herm_res = float(np.linalg.norm(arr - arr.conj().T, ord="fro"))
        herm_tol = _scaled_tol(
            float(np.linalg.norm(arr, ord="fro")), tolerance, _WILKINSON_FLOOR
        )
        if herm_res > herm_tol:
            raise ValueError(
                f"{name} no es hermítica: ‖A−A†‖_F={herm_res:.3e} > {herm_tol:.3e}"
            )
        return _hermite(arr)

    def _certified_eigh(
        self, herm: np.ndarray, name: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """A = UΛU† con residuo de Wilkinson r = ‖AU−UΛ‖_F / max(1,‖A‖_F)."""
        eigvals, eigvecs = la.eigh(herm)
        recon = herm @ eigvecs - eigvecs @ np.diag(eigvals)
        denom = max(float(np.linalg.norm(herm, ord="fro")), 1.0)
        residual = float(np.linalg.norm(recon, ord="fro") / denom)
        if residual > _SPECTRAL_RESIDUAL_TOL:
            logger.warning("Residuo espectral elevado en %s: r=%.3e.", name, residual)
        return np.real(eigvals), eigvecs, residual

    def _certified_spectral_norm(self, mat: np.ndarray) -> float:
        """‖A‖_{B(ℋ)} = σ_max(A)."""
        if mat.size == 0:
            return 0.0
        singular = la.svd(mat, compute_uv=False, overwrite_a=False)
        if singular.size == 0:
            return 0.0
        return float(np.real(singular[0]))

    def _toeplitz_residual(self, mat: np.ndarray) -> float:
        """
        Residuo del projector de Toeplitz (promedio por diagonales):
            ‖A − P_Toep(A)‖_F / max(1, ‖A‖_F).
        Cero ⇔ A es Toeplitz en esta base.  Diagnóstico, no se impone.
        """
        n = mat.shape[0]
        if n < 2:
            return 0.0
        proj = np.zeros_like(mat, dtype=np.complex128)
        for k in range(-n + 1, n):
            diag = np.diag(mat, k=k)
            mean = np.mean(diag) if diag.size else 0.0
            proj += np.diag(np.full(diag.shape, mean, dtype=np.complex128), k=k)
        denom = max(float(np.linalg.norm(mat, ord="fro")), 1.0)
        return float(np.linalg.norm(mat - proj, ord="fro") / denom)

    def _von_neumann_entropy(self, eigenvalues: np.ndarray) -> float:
        lam = np.real(np.asarray(eigenvalues, dtype=np.float64))
        lam = lam[lam > _WILKINSON_FLOOR]
        if lam.size == 0:
            return 0.0
        return float(-np.sum(_safe_xlogx(lam)))

    def _clamp_mu_floor(self, mu_floor: float) -> float:
        r"""
        El suelo μ debe cumplir 0 < μ < 1/n:  1/n = λ_min(I/n) es el máximo
        λ_min admisible en un estado.  μ ≥ 1/n no es realizable.
        """
        mu = float(mu_floor)
        if not np.isfinite(mu) or mu < 0.0:
            raise ValueError(f"mu_floor debe ser ≥ 0 y finito; se recibió {mu_floor}")
        mu_max = (1.0 / float(self._n)) * (1.0 - 1.0e-12)
        if mu >= mu_max:
            raise ValueError(
                f"mu_floor={mu:.3e} ≥ 1/n={1.0 / self._n:.3e}: "
                "ningún estado tiene λ_min tan grande (salvo I/n, inalcanzable "
                "con desigualdad estricta)."
            )
        return max(mu, self._reg)

    def _depolarizing_gamma(self, lambda_min: float, mu: float) -> float:
        r"""
        γ ≥ 0 tal que, partiendo de un estado (Tr=1),

            λ_min( (ρ + γ I)/(1 + nγ) ) = (λ_min + γ)/(1 + nγ)  ≥  μ.

        Despeje exacto:  γ = (μ − λ_min)/(1 − nμ)   si λ_min < μ,  else 0.
        El v2 usaba γ = max(0, μ − λ_min), que deja
            λ_min' = (λ_min + γ)/(1+nγ) < μ   siempre que n>0 y γ>0.
        """
        if lambda_min + _WILKINSON_FLOOR >= mu:
            return 0.0
        denom = 1.0 - float(self._n) * mu
        if denom <= _WILKINSON_FLOOR:
            raise ValueError("1 − nμ numéricamente nulo; mu_floor demasiado grande")
        return float((mu - lambda_min) / denom)

    def weyl_toeplitz_symmetrization(self, rho: np.ndarray) -> np.ndarray:
        r"""
        [ARSENAL 1 — PROJECTOR HERMÍTICO]

        Nombre histórico «Weyl–Toeplitz».  El operador implementado es el
        projector ortogonal (Frobenius) sobre el subespacio real de
        hermíticos:

            Π_H(ρ) = (ρ + ρ†)/2.

        Es el primer paso del algoritmo de Higham para la matriz densidad
        más cercana.  No es una cuantización de Weyl ni un operador de
        Toeplitz; el residuo de Toeplitz se reporta aparte en la Fase 1.
        """
        raw = self._validate_matrix_square(rho, "rho_raw")
        return _hermite(raw)

    def higham_tikhonov_regularization(
        self,
        rho_wt: np.ndarray,
        mu_floor: float = _DEFAULT_MU_FLOOR,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        [ARSENAL 2 — HIGHAM (SÍMPLICE) + TIKHONOV (DESPOLARIZANTE)]

        1. Certifica hermiticidad y diagonaliza (Wilkinson).
        2. Higham: proyecta el espectro al símplice Δ^{n−1} (Duchi).
           El estado resultante ρ_★ es la matriz densidad Frobenius-
           más-cercana a ρ_WT en la órbita unitaria.
        3. Tikhonov / canal despolarizante con suelo *normalizado* μ:

               ρ_μ = (ρ_★ + γ I) / (1 + nγ),
               γ = max{0, (μ − λ_min(ρ_★)) / (1 − nμ)}.

        Returns:
            (ρ_μ, λ_min(ρ_WT), α)  con α = 1/(1+nγ) ∈ (0, 1]
            (peso de ρ_★ en la mezcla; contrato v2: `scale_factor`).
        """
        herm = self._validate_hermitian(rho_wt, "rho_wt")
        mu = self._clamp_mu_floor(mu_floor)

        eigvals, eigvecs, _ = self._certified_eigh(herm, "rho_wt")
        lambda_min = float(np.min(eigvals))

        spec_higham = _project_simplex(eigvals)
        if abs(float(np.sum(spec_higham)) - 1.0) > _SIMPLEX_SUM_TOL:
            spec_higham = spec_higham / max(float(np.sum(spec_higham)), _WILKINSON_FLOOR)

        gamma = self._depolarizing_gamma(float(np.min(spec_higham)), mu)
        scale_factor = 1.0 / (1.0 + float(self._n) * gamma)
        spec_reg = (spec_higham + gamma) * scale_factor

        rho_reg = eigvecs @ np.diag(spec_reg) @ eigvecs.conj().T
        rho_reg = _hermite(rho_reg)

        diag = np.real(np.diagonal(rho_reg))
        trace_kbn = self.kahan_sum(diag)
        if abs(trace_kbn - 1.0) > _TRACE_ATOL:
            logger.warning(
                "Traza KBN tras regularización: %.12f; se renormaliza.",
                trace_kbn,
            )
            if abs(trace_kbn) <= _WILKINSON_FLOOR:
                raise ValueError("traza numéricamente nula tras Higham–Tikhonov")
            rho_reg = rho_reg / trace_kbn
            scale_factor = float(scale_factor / trace_kbn)

        return rho_reg, lambda_min, float(scale_factor)

    def _phase1_observe_orient(
        self,
        rho_raw: np.ndarray,
        mu_floor: float = _DEFAULT_MU_FLOOR,
    ) -> PhaseOneArmoryPacket:
        """
        [FASE 1 · Observe + Orient]

        Π_H → Higham → Tikhonov, con residuos certificados.
        Emite el paquete que **es** el objeto inicial de la Fase 2 vía η.
        """
        logger.info(
            "Fase 1: projector hermítico, Higham-símplice y despolarizante Tikhonov."
        )
        mu = self._clamp_mu_floor(mu_floor)

        raw = self._validate_matrix_square(rho_raw, "rho_raw")
        skew = float(np.linalg.norm(raw - raw.conj().T, ord="fro") / 2.0)

        rho_h = self.weyl_toeplitz_symmetrization(raw)
        eig_h, vec_h, spec_res = self._certified_eigh(rho_h, "rho_wt")
        lambda_min = float(np.min(eig_h))

        spec_star = _project_simplex(eig_h)
        rho_star = _hermite(vec_h @ np.diag(spec_star) @ vec_h.conj().T)

        rho_mu, lambda_min_api, scale = self.higham_tikhonov_regularization(
            rho_h, mu_floor=mu
        )
        eig_mu, _, spec_res_mu = self._certified_eigh(rho_mu, "rho_regularized")
        spec_res = max(spec_res, spec_res_mu)

        gamma = self._depolarizing_gamma(float(np.min(spec_star)), mu)
        lam_min_mu = float(np.min(eig_mu))
        lam_max_mu = float(np.max(eig_mu))
        cond = (
            lam_max_mu / max(lam_min_mu, _WILKINSON_FLOOR)
            if lam_min_mu > 0.0
            else float("inf")
        )
        dist = float(np.linalg.norm(rho_star - rho_mu, ord="fro"))
        trace_res = abs(self.kahan_sum(np.real(np.diagonal(rho_mu))) - 1.0)
        purity = float(np.real(np.sum(np.square(eig_mu))))
        entropy = self._von_neumann_entropy(eig_mu)
        toep = self._toeplitz_residual(rho_h)

        packet = PhaseOneArmoryPacket(
            rho_weyl_toeplitz=_freeze(rho_h),
            rho_higham=_freeze(rho_star),
            rho_regularized=_freeze(rho_mu),
            lambda_min=float(lambda_min_api),
            lambda_min_regularized=lam_min_mu,
            scale_factor=float(scale),
            depolarizing_gamma=float(gamma),
            skew_hermitian_residual=float(skew),
            toeplitz_residual=float(toep),
            spectral_residual=float(spec_res),
            trace_residual=float(trace_res),
            condition_number=float(cond),
            higham_tikhonov_distance=float(dist),
            purity_regularized=purity,
            von_neumann_entropy=entropy,
        )
        logger.debug(
            "Fase 1: λ_min(H)=%.3e  λ_min(μ)=%.3e  α=%.6f  γ=%.3e  κ=%.3e  skew=%.3e",
            packet.lambda_min,
            packet.lambda_min_regularized,
            packet.scale_factor,
            packet.depolarizing_gamma,
            packet.condition_number,
            packet.skew_hermitian_residual,
        )
        # ---- morfismo terminal η: cierra Fase 1 y abre formalmente Fase 2 ----
        return self._phase1_terminal_morphism(packet)

    def _phase1_terminal_morphism(
        self,
        packet: PhaseOneArmoryPacket,
    ) -> PhaseOneArmoryPacket:
        r"""
        [FASE 1 · morfismo terminal η]  ≡  [FASE 2 · objeto inicial]

        Unidad idempotente Spec → Spec.  Todo método de la Fase 2
        **comienza** reaplicando η: recertifica invariantes y no muta.
        """
        if packet.rho_regularized.shape != (self._n, self._n):
            raise ValueError("η: dim(ρ_μ) corrupta")
        if packet.rho_weyl_toeplitz.shape != (self._n, self._n):
            raise ValueError("η: dim(Π_H ρ) corrupta")
        if packet.lambda_min_regularized < -_TRACE_ATOL:
            raise ValueError("η: λ_min(ρ_μ) negativo (inconsistencia numérica)")
        if packet.purity_regularized < -_TRACE_ATOL or packet.purity_regularized > 1.0 + _TRACE_ATOL:
            raise ValueError("η: pureza fuera de [0, 1]")
        if packet.scale_factor <= 0.0 or packet.scale_factor > 1.0 + _TRACE_ATOL:
            raise ValueError("η: scale_factor fuera de (0, 1]")
        if not np.isfinite(
            [
                packet.lambda_min,
                packet.lambda_min_regularized,
                packet.scale_factor,
                packet.condition_number,
                packet.skew_hermitian_residual,
                packet.von_neumann_entropy,
            ]
        ).all():
            raise ValueError("η: observables no finitos")
        return packet

    # =========================================================================
    # FASE 2 — DECIDE  (continúa η; produce χ ∈ Ω₃)
    #   2.0  η(packet)                         ← último morfismo de Fase 1
    #   2.1  seminorma de Connes ‖[ρ^{-1/2}, X]‖ en la base propia
    #   2.2  diferencias divididas de Daletskii–Krein para f(λ)=λ^{-1/2}
    #   2.3  predicados atómicos + operaciones de Heyting
    #   2.4  clasificador de subobjetos χ
    # =========================================================================
    def _daleckii_krein_divided_differences(
        self,
        eigenvalues: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Matriz de diferencias divididas de f(λ) = λ^{-1/2}:

            f^{[1]}(λ,μ) = −1 / (√(λμ) (√λ + √μ))

        (la fórmula cubre también la diagonal: f'(λ) = −½ λ^{-3/2}).
        Lip(f) = max |f^{[1]}| = ½ λ_min^{-3/2} sobre el espectro.
        """
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        sqrt_l = np.sqrt(lam)
        denom = sqrt_l[:, None] * sqrt_l[None, :] * (sqrt_l[:, None] + sqrt_l[None, :])
        dk = -1.0 / denom
        lip = float(np.max(np.abs(dk)))
        return dk, lip

    def _connes_commutator_in_eigenbasis(
        self,
        eigenvalues: np.ndarray,
        eigvecs: np.ndarray,
        pi_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        [D, X] en la base propia de ρ, D = ρ^{-1/2}:

            [D, X]_{ij} = (λ_i^{-1/2} − λ_j^{-1/2}) ⟨i|X|j⟩.

        La norma espectral es unitariamente invariante, así que no hace
        falta reconstruir D ni volver a la base original.
        """
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        d_spec = 1.0 / np.sqrt(lam)
        x_eig = eigvecs.conj().T @ pi_x @ eigvecs
        comm_eig = (d_spec[:, None] - d_spec[None, :]) * x_eig
        return comm_eig, d_spec

    def _dirac_residual(
        self,
        eigenvalues: np.ndarray,
        eigvecs: np.ndarray,
        rho: np.ndarray,
    ) -> float:
        """‖ρ D² − I‖_F / √n   con D² = ρ^{-1} en la base propia."""
        lam = np.clip(np.real(np.asarray(eigenvalues, dtype=np.float64)), self._reg, None)
        d2 = eigvecs @ np.diag(1.0 / lam) @ eigvecs.conj().T
        resid = rho @ d2 - np.eye(self._n, dtype=np.complex128)
        return float(np.linalg.norm(resid, ord="fro") / np.sqrt(float(self._n)))

    def connes_daleckii_krein_filter(
        self,
        rho_reg: np.ndarray,
        pi_X: np.ndarray,
    ) -> Tuple[float, float, str]:
        r"""
        [ARSENAL 3 — SEMINORMA DE CONNES + LIPSCHITZ DE DALETSKII–KREIN]

        No se construye un triple espectral (𝒜, ℋ, 𝐷̸).  Se evalúa la
        seminorma de Lipschitz no conmutativa del observable π(X) respecto
        del operador derivado D_ρ = ρ^{-1/2}:

            L(X) = ‖[D_ρ, π(X)]‖_{B(ℋ)}.

        Daletskii–Krein aporta Lip(f) de f(λ)=λ^{-1/2} sobre spec(ρ),
        cota de la derivada de Fréchet de ρ ↦ ρ^{-1/2}.

        El veredicto *local* (contrato v2) usa solo L(X) frente a
        τ = τ₀ · margen:

            L > τ          → VETOED
            L > 0.1 τ      → DEGRADED
            en otro caso   → COHERENT

        El ciclo OODA (Fase 2) *junta* este veredicto con los predicados
        estructurales de η (κ, skew, residuo de D, …).
        """
        herm = self._validate_hermitian(rho_reg, "rho_reg")
        pi = self._validate_hermitian(pi_X, "pi_X")

        eigvals, eigvecs, _ = self._certified_eigh(herm, "rho_reg")
        if np.any(eigvals <= self._reg * 0.5):
            raise ValueError(
                "rho_reg no es estrictamente positiva definida para D=ρ^{-1/2} "
                f"(λ_min={float(np.min(eigvals)):.3e})"
            )

        comm_eig, _d_spec = self._connes_commutator_in_eigenbasis(eigvals, eigvecs, pi)
        lipschitz_const = self._certified_spectral_norm(comm_eig)
        allowed_tolerance = self.allowed_lipschitz_tolerance

        if lipschitz_const > allowed_tolerance:
            verdict = HeytingVerdict.VETOED.value
        elif lipschitz_const > allowed_tolerance * 0.1:
            verdict = HeytingVerdict.DEGRADED.value
        else:
            verdict = HeytingVerdict.COHERENT.value

        return float(lipschitz_const), float(allowed_tolerance), verdict

    def _classify_heyting_atoms(
        self,
        packet: PhaseOneArmoryPacket,
        lipschitz_const: float,
        lipschitz_dimless: float,
        dk_lip: float,
        dirac_res: float,
        allowed: float,
    ) -> Dict[str, bool]:
        """
        Críticos → join VETOED.  Marginales → join DEGRADED.
        El umbral absoluto L > τ se conserva (ABI del filtro).
        """
        return {
            "lipschitz_violation": lipschitz_const > allowed,
            "dirac_violation": dirac_res > _DIRAC_RESIDUAL_TOL,
            "spectral_violation": packet.spectral_residual > _SPECTRAL_RESIDUAL_TOL,
            "trace_violation": packet.trace_residual > _TRACE_ATOL,
            "pd_violation": packet.lambda_min_regularized <= self._reg * 0.5,
            "lipschitz_marginal": lipschitz_const > 0.1 * allowed,
            "dimless_marginal": lipschitz_dimless > _DIMLESS_COMM_MARGINAL,
            "condition_marginal": packet.condition_number > _CONDITION_MARGINAL,
            "skew_marginal": packet.skew_hermitian_residual > _SKEW_MARGINAL,
            "tikhonov_marginal": packet.higham_tikhonov_distance > _TIKHONOV_MOVE_MARGINAL,
            "dk_marginal": dk_lip > 0.5 * (packet.lambda_min_regularized + self._reg) ** (-1.5),
        }

    def _phase2_decide(
        self,
        packet1: PhaseOneArmoryPacket,
        pi_X: np.ndarray,
    ) -> PhaseTwoArmoryPacket:
        """
        [FASE 2 · Decide]

        Continuación formal de `_phase1_terminal_morphism`: re-certifica η,
        evalúa L(X) y Lip(f) de Daletskii–Krein, y clasifica en

            Ω₃ = {COHERENT < DEGRADED < VETOED}.

        χ = ⊤  ⇔  veredicto = VETOED.
        """
        certified = self._phase1_terminal_morphism(packet1)  # ← nido Fase 1 → 2
        logger.info("Fase 2: seminorma de Connes y Lipschitz de Daletskii–Krein.")

        pi = self._validate_hermitian(pi_X, "pi_X")
        eigvals, eigvecs, _ = self._certified_eigh(certified.rho_regularized, "rho_μ")

        comm_eig, d_spec = self._connes_commutator_in_eigenbasis(eigvals, eigvecs, pi)
        l_raw = self._certified_spectral_norm(comm_eig)
        norm_d = float(np.max(d_spec))
        norm_x = self._certified_spectral_norm(pi)
        l_norm = l_raw / max(norm_x, _WILKINSON_FLOOR)
        l_dimless = l_raw / max(norm_d * norm_x, _WILKINSON_FLOOR)

        _dk_mat, dk_lip = self._daleckii_krein_divided_differences(eigvals)
        dirac_res = self._dirac_residual(eigvals, eigvecs, certified.rho_regularized)
        allowed = self.allowed_lipschitz_tolerance

        atoms = self._classify_heyting_atoms(
            certified, l_raw, l_dimless, dk_lip, dirac_res, allowed
        )
        critical = (
            atoms["lipschitz_violation"],
            atoms["dirac_violation"],
            atoms["spectral_violation"],
            atoms["trace_violation"],
            atoms["pd_violation"],
        )
        marginal = (
            atoms["lipschitz_marginal"],
            atoms["dimless_marginal"],
            atoms["condition_marginal"],
            atoms["skew_marginal"],
            atoms["tikhonov_marginal"],
            atoms["dk_marginal"],
        )
        verdict = HeytingOmega3.classify(critical, marginal)
        chi_top = verdict is HeytingVerdict.VETOED
        impl = HeytingOmega3.implies(HeytingVerdict.COHERENT, verdict)

        decision = PhaseTwoArmoryPacket(
            lipschitz_constant=float(l_raw),
            lipschitz_normalized=float(l_norm),
            lipschitz_dimensionless=float(l_dimless),
            daleckii_krein_lipschitz=float(dk_lip),
            dirac_residual=float(dirac_res),
            allowed_tolerance=float(allowed),
            heyting_verdict=verdict.value,
            lipschitz_violation=atoms["lipschitz_violation"],
            dirac_violation=atoms["dirac_violation"],
            condition_violation=atoms["condition_marginal"],
            skew_violation=atoms["skew_marginal"],
            subobject_classifier=bool(chi_top),
            implication_coherent_to_verdict=impl.value,
            join_of_atoms=verdict.value,
        )
        logger.debug(
            "Fase 2: Ω₃=%s  L=%.4f (τ=%.2f)  L/‖D‖‖X‖=%.3f  Lip(f)=%.3e  ‖ρD²−I‖=%.3e",
            decision.heyting_verdict,
            decision.lipschitz_constant,
            decision.allowed_tolerance,
            decision.lipschitz_dimensionless,
            decision.daleckii_krein_lipschitz,
            decision.dirac_residual,
        )
        return decision

    # =========================================================================
    # FASE 3 — ACT  (continúa χ; ISR simulada + sello)
    #   3.0  consume PhaseTwoArmoryPacket
    #   3.1  interlock crowbar *simulado*
    #   3.2  digest SHA-256 y sello ABI-compatible
    # =========================================================================
    def _simulate_crowbar_interlock(self, verdict: str) -> Tuple[bool, float]:
        """
        [FASE 3 · ISR simulada]

        Modelo N(400, 4) ns recortado a [380, 420].
        No toca GPIO, ESP32 ni el tiristor BT151 reales.
        """
        if verdict == HeytingVerdict.VETOED.value:
            jitter = float(self._rng.normal(0.0, 4.0))
            latency = float(np.clip(_CROWBAR_IRAM_LATENCY_NS + jitter, 380.0, 420.0))
            logger.critical(
                "VETO Ω₃=⊤: interlock *simulado* (GPIO14/BT151 modelo) en %.2f ns.",
                latency,
            )
            return True, latency
        return False, 0.0

    def _round_obs(self, x: float) -> float:
        if not np.isfinite(x):
            raise ValueError("observable no finito en el digest")
        return float(round(float(x), _DIGEST_ROUND_DECIMALS))

    def _forensic_digest(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, allow_nan=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _phase3_act_and_telemetry(
        self,
        decision: PhaseTwoArmoryPacket,
        packet1: PhaseOneArmoryPacket,
    ) -> ArmoryTelemetry:
        """
        [FASE 3 · Act]

        Recibe χ (Fase 2) y η(packet) (Fase 1), actúa el interlock
        simulado si χ=⊤ y emite el sello con digest SHA-256.
        """
        certified = self._phase1_terminal_morphism(packet1)
        interlock_fired, latency = self._simulate_crowbar_interlock(
            decision.heyting_verdict
        )

        if decision.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "Fase 3: VETO. L=%.4f  τ=%.2f  λ_min=%.3e  κ=%.3e  skew=%.3e",
                decision.lipschitz_constant,
                decision.allowed_tolerance,
                certified.lambda_min,
                certified.condition_number,
                certified.skew_hermitian_residual,
            )

        digest_payload = {
            "verdict": decision.heyting_verdict,
            "lmin": self._round_obs(certified.lambda_min),
            "lmin_mu": self._round_obs(certified.lambda_min_regularized),
            "scale": self._round_obs(certified.scale_factor),
            "L": self._round_obs(decision.lipschitz_constant),
            "L_dim": self._round_obs(decision.lipschitz_dimensionless),
            "dk": self._round_obs(decision.daleckii_krein_lipschitz),
            "tau": self._round_obs(decision.allowed_tolerance),
            "kappa": self._round_obs(certified.condition_number),
            "chi": bool(decision.subobject_classifier),
            "interlock": bool(interlock_fired),
        }

        telemetry = ArmoryTelemetry(
            heyting_verdict=decision.heyting_verdict,
            lambda_min=certified.lambda_min,
            scale_factor=certified.scale_factor,
            lipschitz_constant=decision.lipschitz_constant,
            allowed_tolerance=decision.allowed_tolerance,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            purified_operator=_freeze(np.array(certified.rho_regularized, copy=True)),
            daleckii_krein_lipschitz=decision.daleckii_krein_lipschitz,
            lipschitz_dimensionless=decision.lipschitz_dimensionless,
            condition_number=certified.condition_number,
            forensic_digest=self._forensic_digest(digest_payload),
        )
        logger.debug(
            "Fase 3: interlock=%s  latencia=%.2f ns  digest=%s",
            "ON" if interlock_fired else "OFF",
            latency,
            telemetry.forensic_digest[:16],
        )
        return telemetry

    # =========================================================================
    # ORQUESTACIÓN OODA  (composición η ; χ ; Act)
    # =========================================================================
    def execute_armory_cycle(
        self,
        rho_raw: np.ndarray,
        pi_X: np.ndarray,
        mu_floor: float = _DEFAULT_MU_FLOOR,
    ) -> Dict[str, Any]:
        r"""
        Compone las tres fases anidadas:

            sello = Act( χ( η( Observe(ρ, μ) ), π(X) ) )

        Cualquier excepción no contemplada colapsa a VETOED preventivo
        (fail-closed) y devuelve el dict ABI v2.
        """
        try:
            packet1 = self._phase1_observe_orient(rho_raw, mu_floor=mu_floor)
            decision = self._phase2_decide(packet1, pi_X)
            telemetry = self._phase3_act_and_telemetry(decision, packet1)
            return telemetry.to_dict()
        except Exception:
            logger.exception(
                "Excepción en el ciclo del Arsenal: VETO preventivo (fail-closed)."
            )
            return {
                "heyting_verdict": HeytingVerdict.VETOED.value,
                "lambda_min": float("nan"),
                "scale_factor": float("nan"),
                "lipschitz_constant": float("nan"),
                "allowed_tolerance": self.allowed_lipschitz_tolerance,
                "hardware_interlock_fired": True,
                "actuation_latency_ns": _CROWBAR_IRAM_LATENCY_NS,
                "purified_operator": None,
            }


# -----------------------------------------------------------------------------
# Exportación de firmas de calibre
# -----------------------------------------------------------------------------
__all__ = [
    "GaugeProjectionArmory",
    "PhaseOneArmoryPacket",
    "PhaseTwoArmoryPacket",
    "ArmoryTelemetry",
    "HeytingVerdict",
    "HeytingOmega3",
]