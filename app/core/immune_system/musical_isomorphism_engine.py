# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Musical Isomorphism Engine (Fibrador de Dualidad Categórica)               ║
║  Ruta   : app/core/immune_system/musical_isomorphism_engine.py                       ║
║  Versión: 4.0.0-Topos-Spectral-Categorical-Nested-Wilkinson                          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Diferencial
══════════════════════════════════════════════════════════════════════════════════════
Meta-funtor de dualidad entre Γ(TM) y Γ(T*M) sobre la variedad Riemanniana de la MIC.
Tres fases anidadas con contratos algebraicos formales, auditoría espectral extendida
y verificación de axiomas de emparejamiento métrico.

CAMBIOS ESTRUCTURALES RESPECTO A v3.0.0
----------------------------------------
1. FASES ANIDADAS con DTOs de continuación formal (PreconditionedMetric →
   FlatIsomorphism → SharpIsomorphism → MusicalIsomorphismEngine).

2. RESIDUALES BILATERALES DE INVERSIÓN (Wilkinson):
   \[
     r_\pm=\frac{\|G G^{-1}-I\|_F}{\sqrt n},\quad
     r=\max(r_+,r_-),\quad
     \mathrm{tol}\propto \kappa\,\varepsilon_{\mathrm{mach}}\,n.
   \]

3. DOBLE ROUNDTRIP: ♯∘♭ = id_{TM} y ♭∘♯ = id_{T*M}.

4. AXIOMAS DE EMPAREJAMIENTO:
   \[
     \langle ♭v,\,w\rangle = G(v,w),\qquad
     \|v\|_G=\sqrt{v^\top G v},\qquad
     \|\omega\|_{G^{-1}}=\sqrt{\omega^\top G^{-1}\omega}.
   \]

5. TIKHONOV CON κ OBJETIVO: ε adaptativo hasta κ_reg ≤ κ_target (iterativo acotado).

6. RUTA CHOLESKY de verificación cruzada cuando G es SPD bien condicionada.

7. DTOs de informe: InversionAudit, RoundtripReport, PairingReport, FullCycleReport.

8. STUBS de ecosistema para tests aislados.

9. NORMA-G en TangentVector / CotangentVector (opcional vía motor).

10. PROYECCIÓN al ortocomplemento del kernel reportada en diagnósticos.

Fases anidadas
--------------
.. code-block:: text

    Phase1_MetricSpectralPreconditioner
        │  precondition(G)          ──►  PreconditionedMetric
        ▼
    Phase2_FlatIsomorphism
        │  apply_flat / pairing    ──►  CotangentVector (+ PairingReport)
        ▼
    Phase3_SharpIsomorphism + Engine
        │  apply_sharp / roundtrips ──►  TangentVector (+ RoundtripReport)
        │  audit_functor_composition──►  Z₂ × topos compatibility

FUNDAMENTO
----------
§1 ♭: v_i = G_{ij} v^j.
§2 ♯: ω^i = G^{ij} ω_j.
§3 ♯∘♭ = id, ♭∘♯ = id (equivalencia de fibrados).
§4 ⟨♭v, w⟩ = G(v,w) (musical = Riesz).
§5 κ(G)=λ_max/λ_min; Tikhonov IR ε·I.
§6 Var: (Z₂,×) sobre funtores Cov/Cont.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# =============================================================================
# DEPENDENCIAS DEL ECOSISTEMA (stubs para ejecución aislada)
# =============================================================================
try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:  # pragma: no cover
    G_PHYSICS = np.eye(4, dtype=np.float64)

try:
    from app.core.mic_algebra import (
        Functor,
        FunctorialityError,
        NumericalInstabilityError,
    )
except ImportError:  # pragma: no cover

    class FunctorialityError(Exception):
        """Stub: fallo de funtorialidad / colapso dimensional."""

    class NumericalInstabilityError(Exception):
        """Stub: inestabilidad numérica."""

    class Functor:  # type: ignore[no-redef]
        """Stub base funtorial."""

        @property
        def variance(self) -> Any:
            raise NotImplementedError

        def map_object(self, state: Any) -> Any:
            return state

        def map_morphism(self, f: Any) -> Callable[[Any], Any]:
            if callable(f):
                return f
            return lambda x: x


logger = logging.getLogger("MIC.ImmuneSystem.MusicalIsomorphism")

# =============================================================================
# CONSTANTES NUMÉRICAS (Wilkinson / espectral)
# =============================================================================
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)  # ≈ 2.22e-16
_SYMMETRY_TOLERANCE: float = 1.0e-12
_CONDITION_THRESHOLD: float = 1.0e12
_TIKHONOV_EPSILON_RATIO: float = 1.0e-8
_INVERSION_BASE_TOLERANCE: float = 1.0e-14
_ROUNDTRIP_TOLERANCE_FACTOR: float = 100.0
_WILKINSON_SAFETY: float = 100.0
_TARGET_KAPPA: float = 1.0e10
_MAX_TIKHONOV_ITERS: int = 8
_ORTHO_TOL: float = 1.0e-10
_PAIRING_TOL_FACTOR: float = 50.0
_SPECTRAL_ZERO_FLOOR: float = 1.0e-14

S = TypeVar("S")
T = TypeVar("T")


# =============================================================================
# §A — TIPOS ALGEBRAICOS FUNDAMENTALES
# =============================================================================


class CategoricalVariance(Enum):
    r"""
    Dualidad topológica sobre el grupo \((\mathbb{Z}_2,\times)\).

    \[
      \varphi(\mathrm{COVARIANT})=+1,\quad
      \varphi(\mathrm{CONTRAVARIANT})=-1,
    \]
    con \(\varphi(v_1*v_2)=\varphi(v_1)\cdot\varphi(v_2)\).
    """

    COVARIANT = 1
    CONTRAVARIANT = -1

    def __mul__(self, other: CategoricalVariance) -> CategoricalVariance:
        if not isinstance(other, CategoricalVariance):
            raise TypeError(
                f"Composición de varianza requiere CategoricalVariance; "
                f"recibido {type(other).__name__}."
            )
        return CategoricalVariance(self.value * other.value)

    def __repr__(self) -> str:
        return f"CategoricalVariance.{self.name}(value={self.value:+d})"


@dataclass(frozen=True, slots=True)
class TangentVector:
    r"""
    Vector contravariante \(v=v^i\partial_i\in T_p\mathcal{M}\).

    Invariantes: ``float64``, 1-D, shape ``(n,)``, finito.
    """

    coordinates: NDArray[np.float64]

    def __post_init__(self) -> None:
        coords = object.__getattribute__(self, "coordinates")
        if not isinstance(coords, np.ndarray):
            raise TypeError(
                f"TangentVector.coordinates debe ser NDArray; "
                f"recibido {type(coords).__name__}."
            )
        if coords.ndim != 1:
            raise ValueError(
                f"TangentVector.coordinates debe ser 1-D; shape={coords.shape}."
            )
        if coords.shape[0] < 1:
            raise ValueError("TangentVector: n ≥ 1 requerido.")
        if coords.dtype != np.float64:
            object.__setattr__(
                self, "coordinates", coords.astype(np.float64)
            )
            coords = object.__getattribute__(self, "coordinates")
        if not np.all(np.isfinite(coords)):
            raise ValueError("TangentVector: coordenadas no finitas.")

    @property
    def dim(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def norm(self) -> float:
        """‖v‖₂ euclídea."""
        return float(np.linalg.norm(self.coordinates))

    def copy_coords(self) -> NDArray[np.float64]:
        return np.array(self.coordinates, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class CotangentVector:
    r"""1-forma \(\omega=\omega_i\,dx^i\in T_p^*\mathcal{M}\)."""

    coordinates: NDArray[np.float64]

    def __post_init__(self) -> None:
        coords = object.__getattribute__(self, "coordinates")
        if not isinstance(coords, np.ndarray):
            raise TypeError(
                f"CotangentVector.coordinates debe ser NDArray; "
                f"recibido {type(coords).__name__}."
            )
        if coords.ndim != 1:
            raise ValueError(
                f"CotangentVector.coordinates debe ser 1-D; shape={coords.shape}."
            )
        if coords.shape[0] < 1:
            raise ValueError("CotangentVector: n ≥ 1 requerido.")
        if coords.dtype != np.float64:
            object.__setattr__(
                self, "coordinates", coords.astype(np.float64)
            )
            coords = object.__getattribute__(self, "coordinates")
        if not np.all(np.isfinite(coords)):
            raise ValueError("CotangentVector: coordenadas no finitas.")

    @property
    def dim(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.coordinates))

    def copy_coords(self) -> NDArray[np.float64]:
        return np.array(self.coordinates, dtype=np.float64, copy=True)


# =============================================================================
# §B — FUNTORES BASE
# =============================================================================


class CovariantFunctor(Functor, Generic[S, T]):
    r"""Funtor covariante \(F_*\): preserva composición (push-forward)."""

    @property
    def variance(self) -> CategoricalVariance:
        return CategoricalVariance.COVARIANT

    @property
    def domain_category(self) -> str:
        return "C"

    @property
    def codomain_category(self) -> str:
        return "D"

    def map_object(self, state: Any) -> Any:
        return state

    def map_morphism(self, f: Any) -> Callable[[Any], Any]:
        if callable(f):
            return f
        return lambda x: x


class ContravariantFunctor(Functor, Generic[S, T]):
    r"""Funtor contravariante \(F^*\): invierte composición (pull-back)."""

    @property
    def variance(self) -> CategoricalVariance:
        return CategoricalVariance.CONTRAVARIANT

    @property
    def domain_category(self) -> str:
        return "C^{op}"

    @property
    def codomain_category(self) -> str:
        return "D"

    def map_object(self, state: Any) -> Any:
        return state

    def map_morphism(self, f: Any) -> Callable[[Any], Any]:
        if callable(f):
            return f
        return lambda x: x


# =============================================================================
# §C — DTOs DE AUDITORÍA / CONTINUACIÓN FORMAL
# =============================================================================


@dataclass(frozen=True, slots=True)
class InversionAudit:
    r"""Auditoría bilateral \(G G^{-1}\approx I\approx G^{-1}G\)."""

    residual_left: float
    residual_right: float
    residual_max: float
    tolerance: float
    passed: bool
    method: str  # "spectral" | "cholesky_crosscheck"

    def summary(self) -> Dict[str, Any]:
        return {
            "residual_left": self.residual_left,
            "residual_right": self.residual_right,
            "residual_max": self.residual_max,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class RoundtripReport:
    r"""Informe ♯∘♭ o ♭∘♯."""

    direction: str  # "sharp_flat" | "flat_sharp"
    residual: float
    tolerance: float
    passed: bool
    input_norm: float
    kappa_reg: float

    def summary(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "input_norm": self.input_norm,
            "kappa_reg": self.kappa_reg,
        }


@dataclass(frozen=True, slots=True)
class PairingReport:
    r"""
    Axioma de Riesz / emparejamiento musical:

    \[
      \langle ♭v,\,w\rangle_{\mathrm{eucl}} = v^\top G w.
    \]
    """

    residual: float
    tolerance: float
    passed: bool
    G_inner: float
    flat_pairing: float


@dataclass(frozen=True, slots=True)
class PreconditionedMetric:
    r"""
    **Salida terminal de la Fase 1** — precondición formal de
    ``FlatIsomorphism`` / ``MusicalIsomorphismEngine``.

    Invariantes
    -----------
    - \(G,G^{-1}\in\mathrm{Sym}^+(n)\), autovalores reg. \(>0\).
    - Residual de inversión dentro de cota Wilkinson.
    - Formas \((n,n)\) / \((n,)\).
    """

    G: NDArray[np.float64]
    G_inv: NDArray[np.float64]
    eigenvalues_raw: NDArray[np.float64]
    eigenvalues_reg: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    condition_number_raw: float
    condition_number_reg: float
    spectral_gap_absolute: float
    spectral_gap_cheeger: float
    null_space_dim: int
    tikhonov_epsilon: float
    regularization_applied: bool
    matrix_dimension: int
    inversion_residual: float = 0.0
    inversion_tolerance: float = 0.0
    cholesky_crosscheck_residual: Optional[float] = None
    frobenius_norm_G: float = 0.0
    frobenius_norm_G_inv: float = 0.0
    operator_norm_G: float = 0.0  # λ_max_reg
    operator_norm_G_inv: float = 0.0  # 1/λ_min_reg

    def __post_init__(self) -> None:
        n = self.matrix_dimension
        if n < 1:
            raise ValueError("matrix_dimension debe ser ≥ 1.")
        for attr_name, arr in (
            ("G", self.G),
            ("G_inv", self.G_inv),
            ("eigenvectors", self.eigenvectors),
        ):
            if arr.shape != (n, n):
                raise ValueError(
                    f"PreconditionedMetric.{attr_name} shape={arr.shape} "
                    f"≠ ({n},{n})."
                )
        for attr_name, arr in (
            ("eigenvalues_raw", self.eigenvalues_raw),
            ("eigenvalues_reg", self.eigenvalues_reg),
        ):
            if arr.shape != (n,):
                raise ValueError(
                    f"PreconditionedMetric.{attr_name} shape={arr.shape} "
                    f"≠ ({n},)."
                )
        if not np.all(self.eigenvalues_reg > 0):
            raise ValueError(
                "eigenvalues_reg contiene valores ≤ 0; Tikhonov insuficiente."
            )
        if self.condition_number_reg < 1.0 - 1e-12:
            raise ValueError(
                f"condition_number_reg={self.condition_number_reg} < 1."
            )

    def spectral_summary(self) -> Dict[str, Any]:
        return {
            "matrix_dimension": self.matrix_dimension,
            "null_space_dim": self.null_space_dim,
            "condition_number_raw": self.condition_number_raw,
            "condition_number_reg": self.condition_number_reg,
            "spectral_gap_absolute": self.spectral_gap_absolute,
            "spectral_gap_cheeger": self.spectral_gap_cheeger,
            "tikhonov_epsilon": self.tikhonov_epsilon,
            "regularization_applied": self.regularization_applied,
            "lambda_min_raw": float(self.eigenvalues_raw[0]),
            "lambda_max_raw": float(self.eigenvalues_raw[-1]),
            "lambda_min_reg": float(self.eigenvalues_reg[0]),
            "lambda_max_reg": float(self.eigenvalues_reg[-1]),
            "inversion_residual": self.inversion_residual,
            "inversion_tolerance": self.inversion_tolerance,
            "cholesky_crosscheck_residual": self.cholesky_crosscheck_residual,
            "frobenius_norm_G": self.frobenius_norm_G,
            "frobenius_norm_G_inv": self.frobenius_norm_G_inv,
            "operator_norm_G": self.operator_norm_G,
            "operator_norm_G_inv": self.operator_norm_G_inv,
        }


# =============================================================================
# FASE 1 — MetricSpectralPreconditioner
# =============================================================================


class MetricSpectralPreconditioner:
    r"""
    **Fase 1** — Preacondicionamiento espectral de \(G\).

    Pipeline
    --------
    .. code-block:: text

        G_raw
          → validate → symmetrize → eigh
          → diagnostics → Tikhonov(κ_target)
          → (G_reg, G_inv) espectral
          → auditoría bilateral (+ Cholesky cross-check)
          → PreconditionedMetric

    Continuación formal
    -------------------
    ``precondition`` produce ``PreconditionedMetric``, objeto inicial de
    ``FlatIsomorphism.__init__`` (Fase 2).
    """

    SYMMETRY_TOLERANCE: float = _SYMMETRY_TOLERANCE
    CONDITION_THRESHOLD: float = _CONDITION_THRESHOLD
    TIKHONOV_EPSILON_RATIO: float = _TIKHONOV_EPSILON_RATIO
    INVERSION_BASE_TOLERANCE: float = _INVERSION_BASE_TOLERANCE
    TARGET_KAPPA: float = _TARGET_KAPPA
    MAX_TIKHONOV_ITERS: int = _MAX_TIKHONOV_ITERS

    def __init__(
        self,
        *,
        symmetry_tolerance: float = _SYMMETRY_TOLERANCE,
        condition_threshold: float = _CONDITION_THRESHOLD,
        tikhonov_epsilon_ratio: float = _TIKHONOV_EPSILON_RATIO,
        inversion_base_tolerance: float = _INVERSION_BASE_TOLERANCE,
        target_kappa: float = _TARGET_KAPPA,
        enable_cholesky_crosscheck: bool = True,
    ) -> None:
        if symmetry_tolerance <= 0.0:
            raise ValueError("symmetry_tolerance debe ser > 0.")
        if condition_threshold < 1.0:
            raise ValueError("condition_threshold debe ser ≥ 1.")
        if tikhonov_epsilon_ratio <= 0.0:
            raise ValueError("tikhonov_epsilon_ratio debe ser > 0.")
        if target_kappa < 1.0:
            raise ValueError("target_kappa debe ser ≥ 1.")
        self.SYMMETRY_TOLERANCE = float(symmetry_tolerance)
        self.CONDITION_THRESHOLD = float(condition_threshold)
        self.TIKHONOV_EPSILON_RATIO = float(tikhonov_epsilon_ratio)
        self.INVERSION_BASE_TOLERANCE = float(inversion_base_tolerance)
        self.TARGET_KAPPA = float(target_kappa)
        self.enable_cholesky_crosscheck = bool(enable_cholesky_crosscheck)

    # ----- método terminal Fase 1 -------------------------------------------

    def precondition(
        self, raw_metric: NDArray[np.float64]
    ) -> PreconditionedMetric:
        r"""
        **Método terminal de la Fase 1.**

        Retorna
        -------
        PreconditionedMetric
            Precondición formal de ``FlatIsomorphism`` (Fase 2).
        """
        n = self._validate_matrix_structure(raw_metric)
        G_sym = self._enforce_symmetry(raw_metric, n)
        eigenvalues_raw, eigenvectors = self._spectral_decomposition(
            G_sym, n
        )
        cond_raw, gap_abs, gap_chg, null_dim = (
            self._compute_spectral_diagnostics(eigenvalues_raw)
        )
        eigenvalues_reg, epsilon, reg_applied = self._adaptive_tikhonov(
            eigenvalues_raw, cond_raw
        )
        G_reg, G_inv = self._build_regularized_pair(
            eigenvectors, eigenvalues_reg
        )
        cond_reg = float(eigenvalues_reg[-1] / eigenvalues_reg[0])

        audit = self._verify_inversion(G_reg, G_inv, cond_reg, n)
        chol_res: Optional[float] = None
        if self.enable_cholesky_crosscheck and not reg_applied:
            chol_res = self._cholesky_crosscheck(G_reg, G_inv, n)

        pm = PreconditionedMetric(
            G=G_reg,
            G_inv=G_inv,
            eigenvalues_raw=eigenvalues_raw,
            eigenvalues_reg=eigenvalues_reg,
            eigenvectors=eigenvectors,
            condition_number_raw=float(cond_raw),
            condition_number_reg=cond_reg,
            spectral_gap_absolute=gap_abs,
            spectral_gap_cheeger=gap_chg,
            null_space_dim=null_dim,
            tikhonov_epsilon=float(epsilon),
            regularization_applied=reg_applied,
            matrix_dimension=n,
            inversion_residual=audit.residual_max,
            inversion_tolerance=audit.tolerance,
            cholesky_crosscheck_residual=chol_res,
            frobenius_norm_G=float(la.norm(G_reg, "fro")),
            frobenius_norm_G_inv=float(la.norm(G_inv, "fro")),
            operator_norm_G=float(eigenvalues_reg[-1]),
            operator_norm_G_inv=float(1.0 / eigenvalues_reg[0]),
        )
        logger.info(
            "[Fase1] Precondicionamiento: %s | inv=%s",
            pm.spectral_summary(),
            audit.summary(),
        )
        # Continuación formal → FlatIsomorphism(pm)
        return pm

    # ----- pipeline privado -------------------------------------------------

    def _validate_matrix_structure(self, G: NDArray[np.float64]) -> int:
        if not isinstance(G, np.ndarray):
            raise TypeError(
                f"Se requiere NDArray; recibido {type(G).__name__}."
            )
        if G.ndim != 2:
            raise ValueError(
                f"Tensor métrico debe ser 2-D; ndim={G.ndim}."
            )
        if G.shape[0] != G.shape[1]:
            raise ValueError(
                f"Tensor métrico debe ser cuadrado; shape={G.shape}."
            )
        n = int(G.shape[0])
        if n < 1:
            raise ValueError("dimensión n ≥ 1 requerida.")
        if not np.all(np.isfinite(G)):
            n_nan = int(np.sum(np.isnan(G)))
            n_inf = int(np.sum(np.isinf(G)))
            raise ValueError(
                f"G no finita: {n_nan} NaN, {n_inf} Inf."
            )
        return n

    def _enforce_symmetry(
        self, G: NDArray[np.float64], n: int
    ) -> NDArray[np.float64]:
        G = np.asarray(G, dtype=np.float64)
        asym = float(la.norm(G - G.T, "fro"))
        if asym > self.SYMMETRY_TOLERANCE:
            logger.warning(
                "[Fase1] Asimetría ‖G−Gᵀ‖_F=%.3e > tol=%.3e; proyectando a Sym(%d).",
                asym,
                self.SYMMETRY_TOLERANCE,
                n,
            )
            G_sym = 0.5 * (G + G.T)
            return G_sym
        return G

    def _spectral_decomposition(
        self, G_sym: NDArray[np.float64], n: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        try:
            eigenvalues, eigenvectors = la.eigh(G_sym)
        except la.LinAlgError as exc:
            raise NumericalInstabilityError(
                f"eigh falló en Mat({n}×{n}): {exc}"
            ) from exc
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
        ortho_err = float(
            la.norm(eigenvectors.T @ eigenvectors - np.eye(n), "fro")
        )
        if ortho_err > _ORTHO_TOL:
            logger.warning(
                "[Fase1] Ortogonalidad degradada ‖VᵀV−I‖_F=%.3e (n=%d).",
                ortho_err,
                n,
            )
        return eigenvalues, eigenvectors

    def _spectral_zero_tol(
        self, eigenvalues: NDArray[np.float64]
    ) -> float:
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        return max(
            _WILKINSON_SAFETY * _MACHINE_EPSILON * scale,
            _SPECTRAL_ZERO_FLOOR,
        )

    def _compute_spectral_diagnostics(
        self, eigenvalues: NDArray[np.float64]
    ) -> Tuple[float, float, float, int]:
        lam_min = float(eigenvalues[0])
        lam_max = float(eigenvalues[-1])
        ztol = self._spectral_zero_tol(eigenvalues)

        if lam_max <= 0.0:
            logger.error(
                "[Fase1] λ_max=%.3e ≤ 0: G no define métrica Riemanniana.",
                lam_max,
            )
            cond_raw = float(np.inf)
        elif lam_min <= ztol:
            cond_raw = float(np.inf)
        else:
            cond_raw = lam_max / lam_min

        gap_abs = (lam_min / lam_max) if lam_max > 0.0 else 0.0

        positive = eigenvalues[eigenvalues > ztol]
        if len(positive) >= 2 and lam_max > 0.0:
            gap_chg = float(positive[1] / lam_max)
        elif len(positive) == 1 and lam_max > 0.0:
            gap_chg = float(positive[0] / lam_max)
        else:
            gap_chg = 0.0

        null_dim = int(np.sum(eigenvalues <= ztol))
        return cond_raw, float(gap_abs), float(gap_chg), null_dim

    def _adaptive_tikhonov(
        self,
        eigenvalues: NDArray[np.float64],
        condition_number: float,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        Tikhonov iterativo hasta \(\kappa_{\mathrm{reg}}\le\kappa_{\mathrm{target}}\)
        o agotar iteraciones.

        \[
          \lambda_i^{(k+1)}=\lambda_i+\varepsilon_k,\quad
          \varepsilon_0=\rho\cdot\max(|\lambda|),\quad
          \varepsilon_{k+1}\leftarrow 10\,\varepsilon_k.
        \]
        """
        ztol = self._spectral_zero_tol(eigenvalues)
        needs = (
            (not math.isfinite(condition_number))
            or condition_number > self.CONDITION_THRESHOLD
            or bool(np.any(eigenvalues <= ztol))
        )
        if not needs:
            return eigenvalues.copy(), 0.0, False

        scale = float(np.max(np.abs(eigenvalues)))
        if scale <= 0.0:
            scale = 1.0
        eps = self.TIKHONOV_EPSILON_RATIO * scale
        lam = eigenvalues.astype(np.float64, copy=True)
        total_eps = 0.0

        for it in range(self.MAX_TIKHONOV_ITERS):
            lam = lam + eps
            total_eps += eps
            cond = float(lam[-1] / lam[0])
            if math.isfinite(cond) and cond <= self.TARGET_KAPPA and lam[0] > 0.0:
                logger.warning(
                    "[Fase1] Tikhonov: iters=%d, ε_total=%.3e, κ: %.3e → %.3e.",
                    it + 1,
                    total_eps,
                    condition_number,
                    cond,
                )
                return lam, total_eps, True
            eps *= 10.0

        cond_final = float(lam[-1] / max(lam[0], _MACHINE_EPSILON))
        logger.warning(
            "[Fase1] Tikhonov agotó iters=%d; κ_final=%.3e (target=%.3e).",
            self.MAX_TIKHONOV_ITERS,
            cond_final,
            self.TARGET_KAPPA,
        )
        return lam, total_eps, True

    def _build_regularized_pair(
        self,
        eigenvectors: NDArray[np.float64],
        eigenvalues_reg: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        V = eigenvectors
        # G = V diag(λ) Vᵀ sin materializar diag completo cuando es posible
        G_reg = (V * eigenvalues_reg) @ V.T
        G_inv = (V * (1.0 / eigenvalues_reg)) @ V.T
        G_reg = 0.5 * (G_reg + G_reg.T)
        G_inv = 0.5 * (G_inv + G_inv.T)
        return G_reg, G_inv

    def _inversion_tolerance(self, kappa_reg: float, n: int) -> float:
        return max(
            self.INVERSION_BASE_TOLERANCE,
            float(kappa_reg) * _MACHINE_EPSILON * max(n, 1) * _WILKINSON_SAFETY,
        )

    def _verify_inversion(
        self,
        G_reg: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        condition_number_reg: float,
        n: int,
    ) -> InversionAudit:
        I = np.eye(n, dtype=np.float64)
        scale = math.sqrt(float(n))
        r_left = float(la.norm(G_reg @ G_inv - I, "fro")) / scale
        r_right = float(la.norm(G_inv @ G_reg - I, "fro")) / scale
        r_max = max(r_left, r_right)
        tol = self._inversion_tolerance(condition_number_reg, n)
        passed = r_max <= tol
        if not passed:
            raise NumericalInstabilityError(
                f"Inversión fallida: r_max={r_max:.3e} > tol={tol:.3e} "
                f"(r_left={r_left:.3e}, r_right={r_right:.3e}, "
                f"κ_reg={condition_number_reg:.3e}, n={n})."
            )
        return InversionAudit(
            residual_left=r_left,
            residual_right=r_right,
            residual_max=r_max,
            tolerance=tol,
            passed=True,
            method="spectral",
        )

    def _cholesky_crosscheck(
        self,
        G_reg: NDArray[np.float64],
        G_inv_spectral: NDArray[np.float64],
        n: int,
    ) -> Optional[float]:
        """Compara \(G^{-1}_{\mathrm{spectral}}\) con inversa vía Cholesky."""
        try:
            L = la.cholesky(G_reg, lower=True)
        except la.LinAlgError:
            logger.debug("[Fase1] Cholesky cross-check omitido (no SPD numérico).")
            return None
        I = np.eye(n, dtype=np.float64)
        Y = la.solve_triangular(L, I, lower=True, check_finite=False)
        G_inv_chol = Y.T @ Y
        G_inv_chol = 0.5 * (G_inv_chol + G_inv_chol.T)
        diff = float(
            la.norm(G_inv_chol - G_inv_spectral, "fro")
        ) / math.sqrt(float(n))
        logger.debug(
            "[Fase1] Cholesky cross-check ‖ΔG⁻¹‖_F/√n=%.3e.", diff
        )
        return diff


# =============================================================================
# FASE 2 — FlatIsomorphism  (inicio = continuación de PreconditionedMetric)
# =============================================================================


class FlatIsomorphism:
    r"""
    **Fase 2** — Isomorfismo musical bemol \(\flat:TM\to T^*M\).

    Continuación formal de la Fase 1
    --------------------------------
    Entrada: ``PreconditionedMetric`` de
    ``MetricSpectralPreconditioner.precondition``.

    \[
      \omega_i = G_{ij} v^j \quad\Leftrightarrow\quad \omega = G v.
    \]
    """

    def __init__(self, preconditioned_metric: PreconditionedMetric) -> None:
        if not isinstance(preconditioned_metric, PreconditionedMetric):
            raise TypeError(
                f"FlatIsomorphism requiere PreconditionedMetric; "
                f"recibido {type(preconditioned_metric).__name__}."
            )
        self._pm: PreconditionedMetric = preconditioned_metric
        self._G: NDArray[np.float64] = preconditioned_metric.G
        self._G_inv: NDArray[np.float64] = preconditioned_metric.G_inv
        self._n: int = preconditioned_metric.matrix_dimension
        logger.debug(
            "[Fase2] FlatIsomorphism n=%d κ_reg=%.3e.",
            self._n,
            preconditioned_metric.condition_number_reg,
        )

    # ----- normas métricas --------------------------------------------------

    def g_norm(self, v: Union[TangentVector, NDArray[np.float64]]) -> float:
        r"""\(\|v\|_G=\sqrt{v^\top G v}\)."""
        coords = (
            v.coordinates
            if isinstance(v, TangentVector)
            else np.asarray(v, dtype=np.float64).reshape(-1)
        )
        if coords.shape != (self._n,):
            raise FunctorialityError(
                f"g_norm: dim={coords.shape[0]} ≠ n={self._n}."
            )
        q = float(coords @ self._G @ coords)
        return math.sqrt(max(q, 0.0))

    def g_inv_norm(
        self, w: Union[CotangentVector, NDArray[np.float64]]
    ) -> float:
        r"""\(\|\omega\|_{G^{-1}}=\sqrt{\omega^\top G^{-1}\omega}\)."""
        coords = (
            w.coordinates
            if isinstance(w, CotangentVector)
            else np.asarray(w, dtype=np.float64).reshape(-1)
        )
        if coords.shape != (self._n,):
            raise FunctorialityError(
                f"g_inv_norm: dim={coords.shape[0]} ≠ n={self._n}."
            )
        q = float(coords @ self._G_inv @ coords)
        return math.sqrt(max(q, 0.0))

    def metric_inner(
        self,
        v: TangentVector,
        w: TangentVector,
    ) -> float:
        r"""\(G(v,w)=v^\top G w\)."""
        self._validate_vector_dimension(v, "TangentVector", "G(·,·)")
        self._validate_vector_dimension(w, "TangentVector", "G(·,·)")
        return float(v.coordinates @ self._G @ w.coordinates)

    # ----- ♭ ----------------------------------------------------------------

    def apply_flat_isomorphism(self, vector: TangentVector) -> CotangentVector:
        r"""\(\omega = ♭v = G v\)."""
        if not isinstance(vector, TangentVector):
            raise TypeError(
                f"♭ requiere TangentVector; recibido {type(vector).__name__}."
            )
        self._validate_vector_dimension(vector, "TangentVector", "♭")
        omega = self._G @ vector.coordinates
        if not np.all(np.isfinite(omega)):
            raise NumericalInstabilityError(
                f"♭ no finito: ‖v‖₂={vector.norm:.3e}, "
                f"‖G‖_F={self._pm.frobenius_norm_G:.3e}."
            )
        logger.debug(
            "♭: ‖v‖₂=%.3e → ‖ω‖₂=%.3e; ‖v‖_G=%.3e.",
            vector.norm,
            float(np.linalg.norm(omega)),
            self.g_norm(vector),
        )
        return CotangentVector(coordinates=np.asarray(omega, dtype=np.float64))

    def verify_riesz_pairing(
        self,
        v: TangentVector,
        w: TangentVector,
    ) -> PairingReport:
        r"""
        Axioma musical / Riesz:

        \[
          \langle ♭v,\, w\rangle_{\mathbb{R}^n}
            = \sum_i (♭v)_i w^i
            = v^\top G w.
        \]
        """
        omega = self.apply_flat_isomorphism(v)
        self._validate_vector_dimension(w, "TangentVector", "pairing")
        flat_pairing = float(omega.coordinates @ w.coordinates)
        G_inner = self.metric_inner(v, w)
        residual = abs(flat_pairing - G_inner)
        scale = max(abs(G_inner), v.norm * w.norm, 1.0)
        tol = (
            _PAIRING_TOL_FACTOR
            * self._pm.condition_number_reg
            * _MACHINE_EPSILON
            * scale
        )
        passed = residual <= tol
        if not passed:
            raise NumericalInstabilityError(
                f"Riesz pairing fallido: |⟨♭v,w⟩−G(v,w)|={residual:.3e} "
                f"> tol={tol:.3e}."
            )
        return PairingReport(
            residual=residual,
            tolerance=tol,
            passed=True,
            G_inner=G_inner,
            flat_pairing=flat_pairing,
        )

    def diagnostics_report(self) -> Dict[str, Any]:
        report = self._pm.spectral_summary()
        report["phase"] = "FlatIsomorphism (Fase 2)"
        report["isomorphism"] = "♭ : TM → T*M"
        return report

    def _validate_vector_dimension(
        self,
        vector: Union[TangentVector, CotangentVector],
        vector_type: str,
        iso_symbol: str,
    ) -> None:
        if vector.dim != self._n:
            raise FunctorialityError(
                f"Colapso dimensional en {iso_symbol}: "
                f"{vector_type} ∈ ℝ^{vector.dim} incompatible con "
                f"G ∈ Mat({self._n}×{self._n})."
            )


# =============================================================================
# FASE 3 — SharpIsomorphism + auditoría categórica
#           (continuación de FlatIsomorphism / Fase 2)
# =============================================================================


class SharpIsomorphism(FlatIsomorphism):
    r"""
    **Fase 3** — Isomorfismo sostenido \(\sharp:T^*M\to TM\) + auditoría.

    Continuación formal de la Fase 2
    --------------------------------
    Hereda \(G,G^{-1}\) y añade:

    \[
      v^i = G^{ij}\omega_j,\qquad
      ♯\circ♭=\mathrm{id}_{TM},\qquad
      ♭\circ♯=\mathrm{id}_{T^*M}.
    \]
    """

    def apply_sharp_isomorphism(
        self, covector: CotangentVector
    ) -> TangentVector:
        r"""\(v = ♯\omega = G^{-1}\omega\)."""
        if not isinstance(covector, CotangentVector):
            raise TypeError(
                f"♯ requiere CotangentVector; "
                f"recibido {type(covector).__name__}."
            )
        self._validate_vector_dimension(covector, "CotangentVector", "♯")
        v = self._G_inv @ covector.coordinates
        if not np.all(np.isfinite(v)):
            raise NumericalInstabilityError(
                f"♯ no finito: ‖ω‖₂={covector.norm:.3e}, "
                f"‖G⁻¹‖_F={self._pm.frobenius_norm_G_inv:.3e}."
            )
        logger.debug(
            "♯: ‖ω‖₂=%.3e → ‖v‖₂=%.3e; ‖ω‖_{G⁻¹}=%.3e.",
            covector.norm,
            float(np.linalg.norm(v)),
            self.g_inv_norm(covector),
        )
        return TangentVector(coordinates=np.asarray(v, dtype=np.float64))

    def _roundtrip_tolerance(self, input_norm: float) -> float:
        kappa = self._pm.condition_number_reg
        return max(
            _ROUNDTRIP_TOLERANCE_FACTOR
            * kappa
            * _MACHINE_EPSILON
            * max(input_norm, 1.0),
            _INVERSION_BASE_TOLERANCE,
        )

    def verify_roundtrip_identity(
        self, vector: TangentVector
    ) -> RoundtripReport:
        r"""Verifica \(\|♯(♭(v))-v\|_2 \le \mathrm{tol}\)."""
        if not isinstance(vector, TangentVector):
            raise TypeError("verify_roundtrip_identity requiere TangentVector.")
        omega = self.apply_flat_isomorphism(vector)
        v_prime = self.apply_sharp_isomorphism(omega)
        residual = float(
            np.linalg.norm(v_prime.coordinates - vector.coordinates)
        )
        tol = self._roundtrip_tolerance(vector.norm)
        passed = residual <= tol
        report = RoundtripReport(
            direction="sharp_flat",
            residual=residual,
            tolerance=tol,
            passed=passed,
            input_norm=vector.norm,
            kappa_reg=self._pm.condition_number_reg,
        )
        if not passed:
            raise NumericalInstabilityError(
                f"♯∘♭=id fallida: residual={residual:.3e} > tol={tol:.3e} "
                f"(κ_reg={self._pm.condition_number_reg:.3e})."
            )
        logger.debug(
            "Roundtrip ♯∘♭ OK: residual=%.3e ≤ tol=%.3e.", residual, tol
        )
        return report

    def verify_cotroundtrip_identity(
        self, covector: CotangentVector
    ) -> RoundtripReport:
        r"""Verifica \(\|♭(♯(\omega))-\omega\|_2 \le \mathrm{tol}\)."""
        if not isinstance(covector, CotangentVector):
            raise TypeError(
                "verify_cotroundtrip_identity requiere CotangentVector."
            )
        v = self.apply_sharp_isomorphism(covector)
        omega_prime = self.apply_flat_isomorphism(v)
        residual = float(
            np.linalg.norm(omega_prime.coordinates - covector.coordinates)
        )
        tol = self._roundtrip_tolerance(covector.norm)
        passed = residual <= tol
        report = RoundtripReport(
            direction="flat_sharp",
            residual=residual,
            tolerance=tol,
            passed=passed,
            input_norm=covector.norm,
            kappa_reg=self._pm.condition_number_reg,
        )
        if not passed:
            raise NumericalInstabilityError(
                f"♭∘♯=id fallida: residual={residual:.3e} > tol={tol:.3e}."
            )
        logger.debug(
            "Roundtrip ♭∘♯ OK: residual=%.3e ≤ tol=%.3e.", residual, tol
        )
        return report

    def verify_isometry_flat(
        self, vector: TangentVector
    ) -> Dict[str, Any]:
        r"""
        \(\|v\|_G = \|♭v\|_{G^{-1}}\) (isometría musical).
        """
        omega = self.apply_flat_isomorphism(vector)
        n_g = self.g_norm(vector)
        n_gi = self.g_inv_norm(omega)
        residual = abs(n_g - n_gi)
        tol = (
            _PAIRING_TOL_FACTOR
            * self._pm.condition_number_reg
            * _MACHINE_EPSILON
            * max(n_g, 1.0)
        )
        passed = residual <= tol
        if not passed:
            raise NumericalInstabilityError(
                f"Isometría ♭ fallida: |‖v‖_G−‖♭v‖_{{G⁻¹}}|={residual:.3e}."
            )
        return {
            "g_norm": n_g,
            "g_inv_norm_flat": n_gi,
            "residual": residual,
            "tolerance": tol,
            "passed": passed,
        }

    # ----- auditoría Z₂ / topos ---------------------------------------------

    @staticmethod
    def audit_functor_composition(
        f1: Functor,
        f2: Functor,
        *,
        verify_domain_compatibility: bool = True,
    ) -> Dict[str, Any]:
        r"""
        Audita \(f_1\circ f_2\):

        - Varianza: \(\mathrm{Var}(f_1)\otimes\mathrm{Var}(f_2)\) en \(\mathbb{Z}_2\).
        - Topos: \(\mathrm{dom}(f_1)\cong\mathrm{cod}(f_2)\) si está disponible.
        """
        var_1 = SharpIsomorphism._extract_variance(f1, "f1")
        var_2 = SharpIsomorphism._extract_variance(f2, "f2")
        result_variance = var_1 * var_2

        f1_domain = getattr(f1, "domain_category", "?")
        f2_codomain = getattr(f2, "codomain_category", "?")
        domain_compatible = True

        if (
            verify_domain_compatibility
            and f1_domain != "?"
            and f2_codomain != "?"
        ):
            domain_compatible = f1_domain == f2_codomain
            if not domain_compatible:
                raise FunctorialityError(
                    f"Composición inválida en el topos MIC: "
                    f"dom(f1)='{f1_domain}' ≇ cod(f2)='{f2_codomain}'."
                )

        report = {
            "result_variance": result_variance,
            "var_f1": var_1,
            "var_f2": var_2,
            "domain_compatible": domain_compatible,
            "f1_domain": f1_domain,
            "f2_codomain": f2_codomain,
            "composition_valid": domain_compatible,
            "phase": "SharpIsomorphism.audit_functor_composition (Fase 3)",
        }
        logger.debug(
            "Auditoría: %s ⊗ %s ↦ %s | dom(f1)=%s cod(f2)=%s ok=%s.",
            var_1.name,
            var_2.name,
            result_variance.name,
            f1_domain,
            f2_codomain,
            domain_compatible,
        )
        return report

    @staticmethod
    def _extract_variance(functor: Functor, label: str) -> CategoricalVariance:
        variance = getattr(functor, "variance", None)
        if variance is None:
            raise TypeError(
                f"El funtor {label} ({type(functor).__name__}) no expone "
                f"'variance'."
            )
        if not isinstance(variance, CategoricalVariance):
            raise TypeError(
                f"'variance' de {label} debe ser CategoricalVariance; "
                f"recibido {type(variance).__name__}."
            )
        return variance


# =============================================================================
# ORQUESTADOR — MusicalIsomorphismEngine (Fases 1+2+3)
# =============================================================================


class MusicalIsomorphismEngine(SharpIsomorphism):
    r"""
    Árbitro geométrico unificado del fibrado de dualidad.

    Cadena
    ------
    .. code-block:: text

        MetricSpectralPreconditioner.precondition
            → PreconditionedMetric
        FlatIsomorphism (♭, normas G, Riesz)
            → SharpIsomorphism (♯, roundtrips, Z₂)
                → MusicalIsomorphismEngine
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
    ) -> None:
        _pre = (
            preconditioner
            if preconditioner is not None
            else MetricSpectralPreconditioner()
        )
        pm = _pre.precondition(metric_tensor)
        super().__init__(pm)
        self._preconditioner = _pre
        logger.info(
            "[MusicalIsomorphismEngine] v4 n=%d κ_reg=%.3e reg=%s inv_res=%.3e.",
            pm.matrix_dimension,
            pm.condition_number_reg,
            pm.regularization_applied,
            pm.inversion_residual,
        )

    @property
    def preconditioned_metric(self) -> PreconditionedMetric:
        return self._pm

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        return self._G.copy()

    @property
    def metric_inverse(self) -> NDArray[np.float64]:
        return self._G_inv.copy()

    def full_cycle_report(self, vector: TangentVector) -> Dict[str, Any]:
        r"""
        Ciclo unificado:

        1. \(\omega=♭v\)
        2. \(v'=♯\omega\)
        3. roundtrip \(♯∘♭\)
        4. isometría \(\|v\|_G=\|♭v\|_{G^{-1}}\)
        5. pairing de Riesz \(G(v,v)=\langle♭v,v\rangle\)
        """
        if not isinstance(vector, TangentVector):
            raise TypeError("full_cycle_report requiere TangentVector.")
        omega = self.apply_flat_isomorphism(vector)
        v_prime = self.apply_sharp_isomorphism(omega)
        rt = self.verify_roundtrip_identity(vector)
        iso = self.verify_isometry_flat(vector)
        pair = self.verify_riesz_pairing(vector, vector)
        # cotroundtrip sobre ω
        crt = self.verify_cotroundtrip_identity(omega)

        report: Dict[str, Any] = {
            **self._pm.spectral_summary(),
            "input_vector_norm": vector.norm,
            "input_vector_g_norm": self.g_norm(vector),
            "flat_covector_norm": omega.norm,
            "flat_covector_g_inv_norm": self.g_inv_norm(omega),
            "sharp_vector_norm": v_prime.norm,
            "roundtrip_sharp_flat": rt.summary(),
            "roundtrip_flat_sharp": crt.summary(),
            "isometry": iso,
            "riesz_pairing": {
                "residual": pair.residual,
                "tolerance": pair.tolerance,
                "passed": pair.passed,
                "G_inner": pair.G_inner,
                "flat_pairing": pair.flat_pairing,
            },
            "engine": "MusicalIsomorphismEngine v4.0.0 (Fases 1+2+3)",
        }
        logger.info("[Engine] Ciclo completo ♭→♯ auditado.")
        return report


# =============================================================================
# EXPORTACIÓN CANÓNICA
# =============================================================================

__all__ = [
    # Constantes útiles a tests
    "_MACHINE_EPSILON",
    "_MACHINE_EPS",  # alias compat levi_civita stubs
    # Tipos
    "CategoricalVariance",
    "TangentVector",
    "CotangentVector",
    # Funtores
    "CovariantFunctor",
    "ContravariantFunctor",
    # DTOs
    "InversionAudit",
    "RoundtripReport",
    "PairingReport",
    "PreconditionedMetric",
    # Fase 1
    "MetricSpectralPreconditioner",
    # Fase 2
    "FlatIsomorphism",
    # Fase 3
    "SharpIsomorphism",
    # Orquestador
    "MusicalIsomorphismEngine",
    # Errores (reexport si stub o real)
    "FunctorialityError",
    "NumericalInstabilityError",
]

# Alias de compatibilidad con levi_civita_agent stubs
_MACHINE_EPS = _MACHINE_EPSILON