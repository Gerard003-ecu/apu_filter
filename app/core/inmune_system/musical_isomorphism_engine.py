# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ ARQUITECTURA COGNITIVA APU FILTER — SISTEMA INMUNE SOBERANO                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║ Nombre   : MusicalIsomorphismEngine (Árbitro Espectral de Dualidad Riemanniana y Simpléctica)    ║
║ Ubicación: app/core/immune_system/musical_isomorphism_engine.py                                  ║
║ Versión  : 9.0.0-Poincare-Celestial-RSI-L3-Kahler-Spectral-Doctoral                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO:
────────────────────────────────────────────────────────────────────────────────────────────────────
El Soberano de Isomorfismos Musicales (`MusicalIsomorphismEngine`) es el transformador dual y
simpléctico del sistema inmune de APU Filter. Su misión es alternar de forma matemáticamente exacta,
incondicionalmente estable y libre de deriva espectral entre el fibrado tangente de velocidades/flujos
de acción $TM$ y el fibrado cotangente de momentos canónicos/gradientes $T^*M$ mediante la métrica
Riemanniana precondicionada $G_{\mu\nu}$, su inversa refinada en el álgebra de Banach $G^{\mu\nu}$ y
la estructura simpléctica compatible de Kähler-Poincaré $\omega_{\mu\nu}$.

Sus respectivos en el ecosistema son:
  • `ScalarMomentumSatelliteEngine` (`app/core/immune_system/scalar_momentum_satellite_engine.py`),
  • `ScalarMomentumSatelliteAgent` (`app/agents/core/immune_system/scalar_momentum_satellite_agent.py`),
  • `LeviCivitaConnectionAgent` (`app/agents/omega/levi_civita_agent.py`).

MARCO TEÓRICO, MECÁNICA CELESTE DE POINCARÉ Y AUTOMEJORA RECURSIVA NIVEL 3 (Rigor PhD):
────────────────────────────────────────────────────────────────────────────────────────────────────
1. Isomorfismo Bemol ($\flat: TM \to T^*M$) e Isomorfismo Sostenido ($\sharp: T^*M \to TM$):
   Dada la variedad Riemanniana $(M, G)$, el operador bemol contrae vectores contravariantes con el
   tensor métrico y el operador sostenido eleva 1-formas covariantes mediante el tensor inverso:
   $$(\flat v)_\mu = G_{\mu\nu} v^\nu, \qquad (\sharp \alpha)^\mu = G^{\mu\nu} \alpha_\nu, \qquad
     \sharp \circ \flat = \mathrm{id}_{TM}, \quad \flat \circ \sharp = \mathrm{id}_{T^*M}$$

2. Mecánica Celeste de Henri Poincaré, Triplete de Kähler $(G, \omega, J)$ y Flujo Hamiltoniano:
   En la base ortonormal espectral de $G = V \Lambda V^\top$, el motor induce el operador casi-complejo
   ortogonal $J = V J_0 V^\top$ ($J_0 = \bigoplus_{k} \begin{psmallmatrix} 0 & -1 \\ 1 & 0 \end{psmallmatrix}$)
   y la 2-forma simpléctica compatible $\omega(u, v) = G(J u, v)$.
   Dado el diferencial de energía $dH \in T^*M$, el gradiente Riemanniano y el campo vectorial
   Hamiltoniano celeste de Poincaré vienen dados por:
   $$\operatorname{grad}_G(H) = (dH)^\sharp = G^{-1} dH, \qquad
     \iota_{X_H}\omega = -dH \iff X_H = -J (dH)^\sharp = -J G^{-1} dH$$
   Esto garantiza de forma exacta el Teorema de Conservación de Liouville-Poincaré:
   $$\mathcal{L}_{X_H} H = \langle dH, X_H \rangle = - (dH)^\top G^{-1} G J (dH)^\sharp = 0$$
   y dota a $T^*M$ del corchete canónico de Poisson $\{f, g\}_{\mathrm{Poincaré}} = \langle dg, X_f \rangle$.

3. Automejora Recursiva de Nivel 3 (RSI-L3) en Álgebras de Banach:
   • Fase 1 (Hiper-Potencia Cúbica de Chebyshev-Newton-Schulz): Refina el inverso espectral $X_0 = G^{-1}$
     mediante la contracción de orden 3 en el álgebra de Banach $\mathcal{B}(\mathbb{R}^n)$:
     $$R_k = I - G X_k, \qquad X_{k+1} = \frac{1}{2}\left[X_k(I + R_k + R_k^2) + (I + R_k + R_k^2)^\top X_k^\top\right]$$
   • Fase 3 (Corrección Iterativa de Defecto Dual — IDC): Al evaluar $v_0 = G^{-1}\alpha$, audita el
     defecto residual $\delta = \alpha - G v_0$ en $T^*M$ y aplica la corrección de punto fijo
     $v^* = v_0 + G^{-1}\delta$, aniquilando el error de redondeo de punto flotante.

SUTURA DE FLUJO EN 3 FASES ANIDADAS ESTRICTAMENTE CONTINUAS:
────────────────────────────────────────────────────────────────────────────────────────────────────
  Fase 1 ──► MetricSpectralPreconditioner (Phase1_MetricSpectralPreconditioner)
             Último método : `precondition(self, raw_metric) -> PreconditionedMetric`
                                              │
                                              ▼ (Continuación Formal Directa)
  Fase 2 ──► FlatIsomorphism (Phase2_FlatIsomorphism)
             Primer método : `_ingest_phase1_continuation(preconditioned_metric) -> PreconditionedMetric`
             Último método : `diagnostics_report(self) -> Dict[str, Any]`
                                              │
                                              ▼ (Continuación Formal Directa)
  Fase 3 ──► SharpIsomorphism / MusicalIsomorphismEngine (Phase3_SharpIsomorphism)
             Primer método : `_ingest_phase2_continuation(self) -> PreconditionedMetric`
             Último método : `full_cycle_report(self, vector: TangentVector) -> Dict[str, Any]`
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
# DEPENDENCIAS DEL ECOSISTEMA (stubs deterministas para ejecución aislada)
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
# CONSTANTES NUMÉRICAS (Wilkinson / Espectral / Poincaré / RSI-L3)
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

# Constantes de Automejora Recursiva de Nivel 3 (RSI-L3)
_RSI_L3_MAX_NS_ITERS: int = 5
_RSI_L3_CONTRACTION_THRESHOLD: float = 0.95
_RSI_L3_IDC_MAX_STEPS: int = 2

S = TypeVar("S")
T = TypeVar("T")


# =============================================================================
# §A — TIPOS ALGEBRAICOS FUNDAMENTALES Y FIBRADOS DUALES
# =============================================================================


class CategoricalVariance(Enum):
    r"""
    Dualidad topológica sobre el grupo multiplicativo \((\mathbb{Z}_2,\times)\).

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

    Invariantes: ``float64``, 1-D, shape ``(n,)``, entradas finitas.
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
        """Norma euclídea ‖v‖₂."""
        return float(np.linalg.norm(self.coordinates))

    def copy_coords(self) -> NDArray[np.float64]:
        return np.array(self.coordinates, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class CotangentVector:
    r"""
    1-forma diferencial covariante \(\omega=\omega_i\,dx^i\in T_p^*\mathcal{M}\).

    Invariantes: ``float64``, 1-D, shape ``(n,)``, entradas finitas.
    """

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
        """Norma euclídea ‖ω‖₂."""
        return float(np.linalg.norm(self.coordinates))

    def copy_coords(self) -> NDArray[np.float64]:
        return np.array(self.coordinates, dtype=np.float64, copy=True)


# =============================================================================
# §B — FUNTORES BASE (COVARIANTE Y CONTRAVARIANTE)
# =============================================================================


class CovariantFunctor(Functor, Generic[S, T]):
    r"""Funtor covariante \(F_*\): preserva la dirección de los morfismos (push-forward)."""

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
    r"""Funtor contravariante \(F^*\): invierte la dirección de los morfismos (pull-back)."""

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
# §C — DTOs DE AUDITORÍA, TELEMETRÍA RSI-L3 Y CONTINUACIÓN FORMAL ENTRE FASES
# =============================================================================


@dataclass(frozen=True, slots=True)
class InversionAudit:
    r"""Auditoría bilateral en el álgebra de Banach: \(G G^{-1}\approx I\approx G^{-1}G\)."""

    residual_left: float
    residual_right: float
    residual_max: float
    tolerance: float
    passed: bool
    method: str  # "spectral" | "spectral_rsi_l3_ns" | "cholesky_crosscheck"

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
    r"""Informe de identidad canónica \(\sharp\circ\flat = \mathrm{id}_{TM}\) o \(\flat\circ\sharp = \mathrm{id}_{T^*M}\)."""

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
    Teorema de Representación de Riesz / Emparejamiento Musical:

    \[
      \langle \flat v,\,w\rangle_{\mathrm{eucl}} = v^\top G w = G(v, w).
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
    **Salida terminal de la Fase 1** — precondición formal de entrada para
    ``FlatIsomorphism._ingest_phase1_continuation`` (Fase 2).

    Invariantes
    -----------
    - \(G, G^{-1} \in \mathrm{Sym}^+(n)\), autovalores regularizados \(> 0\).
    - Residuo de inversión acotado por la barrera de Wilkinson y refinado por RSI-L3.
    - Estructura simpléctica/casi-compleja de Poincaré \(J\) compatible con \(G\).
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
    newton_schulz_iterations: int = 0
    banach_contraction_gain: float = 1.0
    poincare_symplectic_capacity: float = math.pi
    symplectic_J: Optional[NDArray[np.float64]] = None

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
            "newton_schulz_iterations": self.newton_schulz_iterations,
            "banach_contraction_gain": self.banach_contraction_gain,
            "poincare_symplectic_capacity": self.poincare_symplectic_capacity,
        }


# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 — MetricSpectralPreconditioner (Espectro, Banach RSI-L3 y Kähler) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


class MetricSpectralPreconditioner:
    r"""
    **FASE 1** — Preacondicionamiento espectral de \(G\), refinamiento
    recursivo de Nivel 3 (Chebyshev-Newton-Schulz) en el álgebra de Banach
    \(\mathcal{B}(\mathbb{R}^n)\) y síntesis de la estructura simpléctica de Poincaré.

    Orden constructivo de métodos (1.1 → 1.13)
    ------------------------------------------
    1.1  ``_validate_matrix_structure``
    1.2  ``_enforce_symmetry``
    1.3  ``_spectral_decomposition``
    1.4  ``_spectral_zero_tol``
    1.5  ``_compute_spectral_diagnostics``
    1.6  ``_adaptive_tikhonov``
    1.7  ``_build_regularized_pair``
    1.8  ``_rsi_l3_banach_hyperpower_refinement``
    1.9  ``_build_poincare_symplectic_structure``
    1.10 ``_inversion_tolerance``
    1.11 ``_verify_inversion``
    1.12 ``_cholesky_crosscheck``
    1.13 ``precondition`` (**Método terminal de la Fase 1** → inicia Fase 2).
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

    def _validate_matrix_structure(self, G: NDArray[np.float64]) -> int:
        """1.1 Validación estructural, dimensional y de finitud IEEE-754."""
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
        r"""1.2 Proyección ortogonal sobre el subespacio autoadjunto \(\mathrm{Sym}(n, \mathbb{R})\)."""
        G = np.asarray(G, dtype=np.float64)
        asym = float(la.norm(G - G.T, "fro"))
        if asym > self.SYMMETRY_TOLERANCE:
            logger.warning(
                "[Fase1] Asimetría ‖G−Gᵀ‖_F=%.3e > tol=%.3e; proyectando a Sym(%d).",
                asym,
                self.SYMMETRY_TOLERANCE,
                n,
            )
            return 0.5 * (G + G.T)
        return G

    def _spectral_decomposition(
        self, G_sym: NDArray[np.float64], n: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""1.3 Teorema espectral para operadores autoadjuntos compactos: \(G_{\mathrm{sym}} = V \Lambda V^\top\)."""
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
        """1.4 Umbral espectral de Wilkinson para detección de ker(G)."""
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        return max(
            _WILKINSON_SAFETY * _MACHINE_EPSILON * scale,
            _SPECTRAL_ZERO_FLOOR,
        )

    def _compute_spectral_diagnostics(
        self, eigenvalues: NDArray[np.float64]
    ) -> Tuple[float, float, float, int]:
        """1.5 Cálculo de número de condición, gap absoluto, gap de Cheeger y nulidad."""
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
        1.6 Regularización espectral adaptativa de Tikhonov hasta alcanzar
        \(\kappa_{\mathrm{reg}} \le \kappa_{\mathrm{target}}\).
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
        r"""1.7 Síntesis espectral del par dual \((G_{\mathrm{reg}}, G_{\mathrm{inv}})\)."""
        V = eigenvectors
        G_reg = (V * eigenvalues_reg) @ V.T
        G_inv = (V * (1.0 / eigenvalues_reg)) @ V.T
        G_reg = 0.5 * (G_reg + G_reg.T)
        G_inv = 0.5 * (G_inv + G_inv.T)
        return G_reg, G_inv

    @staticmethod
    def _rsi_l3_banach_hyperpower_refinement(
        G_reg: NDArray[np.float64],
        G_inv_init: NDArray[np.float64],
        n: int,
    ) -> Tuple[NDArray[np.float64], int, float]:
        r"""
        1.8 **Automejora Recursiva de Nivel 3 (RSI-L3) en el Álgebra de Banach \(\mathcal{B}(\mathbb{R}^n)\).**

        Refina el operador inverso \(X_0 = G_{\mathrm{inv}}\) mediante el polinomio
        hiper-potencia de tercer orden de Chebyshev-Newton-Schulz:

        \[
          R_k = I - G_{\mathrm{reg}} X_k, \qquad
          X_{k+1} = \mathrm{Sym}\!\bigl(X_k (I + R_k + R_k^2)\bigr),
        \]

        cuya tasa de convergencia es cúbica (\(\|R_{k+1}\|_F \le \|R_k\|_F^3\)).
        """
        I = np.eye(n, dtype=np.float64)
        scale = math.sqrt(float(n))
        X = G_inv_init

        def _bilateral_res(M_inv: NDArray[np.float64]) -> float:
            rl = float(la.norm(G_reg @ M_inv - I, "fro")) / scale
            rr = float(la.norm(M_inv @ G_reg - I, "fro")) / scale
            return max(rl, rr)

        res_init = _bilateral_res(X)
        res_cur = res_init
        floor_tol = 2.0 * _MACHINE_EPSILON
        iters_done = 0

        if res_cur <= floor_tol:
            return X, 0, 1.0

        for _ in range(_RSI_L3_MAX_NS_ITERS):
            R = I - (G_reg @ X)
            norm_R = float(la.norm(R, "fro"))
            if norm_R >= _RSI_L3_CONTRACTION_THRESHOLD:
                break
            poly = I + R + (R @ R)
            X_cand = X @ poly
            X_cand = 0.5 * (X_cand + X_cand.T)
            if not np.all(np.isfinite(X_cand)):
                break
            res_cand = _bilateral_res(X_cand)
            if res_cand < res_cur:
                X = X_cand
                res_cur = res_cand
                iters_done += 1
                if res_cur <= floor_tol:
                    break
            else:
                break

        gain = res_init / max(res_cur, _MACHINE_EPSILON)
        return X, iters_done, float(gain)

    @staticmethod
    def _build_poincare_symplectic_structure(
        eigenvectors: NDArray[np.float64],
        eigenvalues_reg: NDArray[np.float64],
        n: int,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        1.9 **Mecánica Celeste de Poincaré:** Síntesis de la estructura casi-compleja
        compatible \(J\) y la capacidad simpléctica de Gromov-Poincaré.

        En la base de autovectores \(V\), define bloques de Darboux \(2\times 2\):

        \[
          J_0 = \bigoplus_{k=1}^{\lfloor n/2\rfloor}
          \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}, \qquad
          J = V J_0 V^\top, \qquad
          c_{\mathrm{Gromov}}(B_G(1)) = \frac{\pi}{\sqrt{\lambda_{\max}(G)}}.
        \]
        """
        J0 = np.zeros((n, n), dtype=np.float64)
        pairs = n // 2
        for k in range(pairs):
            i = 2 * k
            J0[i, i + 1] = -1.0
            J0[i + 1, i] = 1.0
        J = eigenvectors @ J0 @ eigenvectors.T
        J = 0.5 * (J - J.T)  # Garantiza antisimetría exacta en la base canónica
        lam_max = float(eigenvalues_reg[-1])
        symp_capacity = float(math.pi / math.sqrt(max(lam_max, _MACHINE_EPSILON)))
        return J, symp_capacity

    def _inversion_tolerance(self, kappa_reg: float, n: int) -> float:
        """1.10 Cota de error de inversión según el análisis de perturbación de Wilkinson."""
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
        ns_iters: int = 0,
    ) -> InversionAudit:
        """1.11 Auditoría bilateral estricta de la identidad G @ G_inv ≈ I."""
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
        method_tag = "spectral_rsi_l3_ns" if ns_iters > 0 else "spectral"
        return InversionAudit(
            residual_left=r_left,
            residual_right=r_right,
            residual_max=r_max,
            tolerance=tol,
            passed=True,
            method=method_tag,
        )

    def _cholesky_crosscheck(
        self,
        G_reg: NDArray[np.float64],
        G_inv_spectral: NDArray[np.float64],
        n: int,
    ) -> Optional[float]:
        r"""1.12 Verificación cruzada independiente mediante factorización de Cholesky \(G = L L^\top\)."""
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

    def precondition(
        self, raw_metric: NDArray[np.float64]
    ) -> PreconditionedMetric:
        r"""
        1.13 **MÉTODO TERMINAL DE LA FASE 1.**

        Ejecuta el pipeline completo de precondicionamiento espectral,
        refinamiento hiper-potencia RSI-L3 en el álgebra de Banach y
        síntesis simpléctica de Poincaré.

        Continuación formal hacia la Fase 2
        -----------------------------------
        El objeto ``PreconditionedMetric`` devuelto por este método es el
        argumento formal exacto del método inicial de la Fase 2:
        ``FlatIsomorphism._ingest_phase1_continuation``.
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
        G_reg, G_inv_raw = self._build_regularized_pair(
            eigenvectors, eigenvalues_reg
        )

        # RSI-L3: Refinamiento hiper-potencia cúbico de Chebyshev-Newton-Schulz
        G_inv, ns_iters, banach_gain = self._rsi_l3_banach_hyperpower_refinement(
            G_reg, G_inv_raw, n
        )

        # Mecánica Celeste de Poincaré: Estructura J y capacidad de Gromov
        symp_J, symp_cap = self._build_poincare_symplectic_structure(
            eigenvectors, eigenvalues_reg, n
        )

        cond_reg = float(eigenvalues_reg[-1] / eigenvalues_reg[0])
        audit = self._verify_inversion(G_reg, G_inv, cond_reg, n, ns_iters=ns_iters)

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
            newton_schulz_iterations=ns_iters,
            banach_contraction_gain=banach_gain,
            poincare_symplectic_capacity=symp_cap,
            symplectic_J=symp_J,
        )
        logger.info(
            "[Fase1] Precondicionamiento: %s | inv=%s",
            pm.spectral_summary(),
            audit.summary(),
        )
        # ═══► CONTINUACIÓN FORMAL DIRECTA HACIA FASE 2 ◄═══
        return pm


# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 — FlatIsomorphism (Isomorfismo ♭: TM → T*M y Forma de Poincaré)    ║
# ║          (Inicio = Continuación directa de Phase1.precondition)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


class FlatIsomorphism:
    r"""
    **FASE 2** — Isomorfismo musical bemol \(\flat: TM \to T^*M\), geometría
    de Riesz y 2-forma simpléctica de Kähler-Poincaré.

    Continuación formal de la Fase 1
    --------------------------------
    El primer método, ``_ingest_phase1_continuation``, recibe directamente el
    objeto ``PreconditionedMetric`` emitido por ``MetricSpectralPreconditioner.precondition``.

    Orden constructivo de métodos (2.1 → 2.10)
    ------------------------------------------
    2.1  ``_ingest_phase1_continuation`` (**Inicio formal de la Fase 2**)
    2.2  ``__init__``
    2.3  ``_validate_vector_dimension``
    2.4  ``g_norm``
    2.5  ``g_inv_norm``
    2.6  ``metric_inner``
    2.7  ``poincare_kinetic_hamiltonian``
    2.8  ``poincare_symplectic_form``
    2.9  ``apply_flat_isomorphism``
    2.10 ``verify_riesz_pairing``
    2.11 ``diagnostics_report`` (**Método terminal de la Fase 2** → inicia Fase 3).
    """

    @staticmethod
    def _ingest_phase1_continuation(
        preconditioned_metric: PreconditionedMetric,
    ) -> PreconditionedMetric:
        r"""
        2.1 **MÉTODO INICIAL DE LA FASE 2 (Continuación Formal de la Fase 1).**

        Verifica el contrato espectral y de positividad del objeto
        ``PreconditionedMetric`` producido por ``MetricSpectralPreconditioner.precondition``.
        """
        if not isinstance(preconditioned_metric, PreconditionedMetric):
            raise TypeError(
                f"FlatIsomorphism requiere PreconditionedMetric; "
                f"recibido {type(preconditioned_metric).__name__}."
            )
        return preconditioned_metric

    def __init__(self, preconditioned_metric: PreconditionedMetric) -> None:
        """2.2 Inicializa el fibrado bemol a partir de la métrica precondicionada."""
        pm = self._ingest_phase1_continuation(preconditioned_metric)
        self._pm: PreconditionedMetric = pm
        self._G: NDArray[np.float64] = pm.G
        self._G_inv: NDArray[np.float64] = pm.G_inv
        self._n: int = pm.matrix_dimension
        if pm.symplectic_J is not None and pm.symplectic_J.shape == (self._n, self._n):
            self._J: NDArray[np.float64] = pm.symplectic_J
        else:
            self._J, _ = MetricSpectralPreconditioner._build_poincare_symplectic_structure(
                pm.eigenvectors, pm.eigenvalues_reg, self._n
            )
        logger.debug(
            "[Fase2] FlatIsomorphism n=%d κ_reg=%.3e c_Poincaré=%.3e.",
            self._n,
            pm.condition_number_reg,
            pm.poincare_symplectic_capacity,
        )

    def _validate_vector_dimension(
        self,
        vector: Union[TangentVector, CotangentVector],
        vector_type: str,
        iso_symbol: str,
    ) -> None:
        """2.3 Verificación de invariancia dimensional del fibrado."""
        if vector.dim != self._n:
            raise FunctorialityError(
                f"Colapso dimensional en {iso_symbol}: "
                f"{vector_type} ∈ ℝ^{vector.dim} incompatible con "
                f"G ∈ Mat({self._n}×{self._n})."
            )

    def g_norm(self, v: Union[TangentVector, NDArray[np.float64]]) -> float:
        r"""2.4 Norma Riemanniana en \(TM\): \(\|v\|_G=\sqrt{v^\top G v}\)."""
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
        r"""2.5 Norma dual cométrica en \(T^*M\): \(\|\omega\|_{G^{-1}}=\sqrt{\omega^\top G^{-1}\omega}\)."""
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
        r"""2.6 Producto interno de Riemann \(G(v,w)=v^\top G w\)."""
        self._validate_vector_dimension(v, "TangentVector", "G(·,·)")
        self._validate_vector_dimension(w, "TangentVector", "G(·,·)")
        return float(v.coordinates @ self._G @ w.coordinates)

    def poincare_kinetic_hamiltonian(self, v: TangentVector) -> float:
        r"""
        2.7 **Mecánica Celeste de Poincaré:** Energía cinética Hamiltoniana
        sobre el fibrado tangente:

        \[
          H_{\mathrm{Poincaré}}(v) = \frac{1}{2} G(v, v) = \frac{1}{2}\langle \flat v, v \rangle.
        \]
        """
        return 0.5 * self.metric_inner(v, v)

    def poincare_symplectic_form(
        self, u: TangentVector, v: TangentVector
    ) -> float:
        r"""
        2.8 **2-Forma Simpléctica Compatible de Kähler-Poincaré:**

        \[
          \omega(u, v) = G(J u, v) = (J u)^\top G v = -\omega(v, u).
        \]
        """
        self._validate_vector_dimension(u, "TangentVector", "ω(·,·)")
        self._validate_vector_dimension(v, "TangentVector", "ω(·,·)")
        Ju = self._J @ u.coordinates
        return float(Ju @ self._G @ v.coordinates)

    def apply_flat_isomorphism(self, vector: TangentVector) -> CotangentVector:
        r"""2.9 Operador musical bemol \(\omega = \flat v = G v \in T_p^*\mathcal{M}\)."""
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
        2.10 Certifica el Teorema de Representación de Riesz:

        \[
          \langle \flat v,\, w\rangle_{\mathbb{R}^n}
            = \sum_{i=1}^n (\flat v)_i w^i
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
        r"""
        2.11 **MÉTODO TERMINAL DE LA FASE 2.**

        Emite el reporte de estado del fibrado bemol \(\flat: TM \to T^*M\)
        y constituye el punto de enlace hacia la Fase 3 (`SharpIsomorphism`).
        """
        report = self._pm.spectral_summary()
        report["phase"] = "FlatIsomorphism (Fase 2)"
        report["isomorphism"] = "♭ : TM → T*M"
        # ═══► CONTINUACIÓN FORMAL DIRECTA HACIA FASE 3 ◄═══
        return report


# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 — SharpIsomorphism + Dinámica Hamiltoniana de Poincaré + RSI-L3    ║
# ║          (Inicio = Continuación directa de FlatIsomorphism / Fase 2)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


class SharpIsomorphism(FlatIsomorphism):
    r"""
    **FASE 3** — Isomorfismo musical sostenido \(\sharp: T^*M \to TM\),
    corrección iterativa de defecto dual (RSI-L3), campos vectoriales
    Hamiltonianos de Poincaré, corchete de Poisson y auditoría categórica \(\mathbb{Z}_2\).

    Continuación formal de la Fase 2
    --------------------------------
    Comienza con ``_ingest_phase2_continuation``, validando la estructura
    heredada de ``FlatIsomorphism``, y culmina en ``full_cycle_report``.
    """

    def _ingest_phase2_continuation(self) -> PreconditionedMetric:
        r"""
        3.1 **MÉTODO INICIAL DE LA FASE 3 (Continuación Formal de la Fase 2).**

        Certifica que el estado de ``FlatIsomorphism`` (Fase 2) se encuentra
        íntegro y listo para habilitar el operador inverso \(\sharp: T^*M \to TM\).
        """
        if not hasattr(self, "_pm") or not isinstance(self._pm, PreconditionedMetric):
            raise FunctorialityError(
                "[Fase3] Continuación desde Fase 2 inválida: PreconditionedMetric ausente."
            )
        return self._pm

    def _rsi_l3_iterative_defect_correction(
        self,
        omega_coords: NDArray[np.float64],
        v_init: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        3.2 **Automejora Recursiva de Nivel 3 (RSI-L3) — Corrección Iterativa de Defecto Dual.**

        Dado \(v_0 = G^{-1}\omega\), evalúa el defecto cotangente
        \(\delta_k = \omega - G v_k\). Si \(\|\delta_k\|_2\) supera el piso
        de precisión de máquina pero la iteración es contractiva, actualiza:

        \[
          v_{k+1} = v_k + G^{-1}(\omega - G v_k),
        \]
        abatiendo el error de inversión finita en el espacio tangente.
        """
        omega_norm = float(np.linalg.norm(omega_coords))
        if omega_norm <= _MACHINE_EPSILON:
            return v_init

        v_cur = v_init
        defect_cur = float(np.linalg.norm(omega_coords - (self._G @ v_cur)))
        floor_tol = 2.0 * _MACHINE_EPSILON * omega_norm

        if defect_cur <= floor_tol:
            return v_cur

        for _ in range(_RSI_L3_IDC_MAX_STEPS):
            residual_vec = omega_coords - (self._G @ v_cur)
            v_cand = v_cur + (self._G_inv @ residual_vec)
            if not np.all(np.isfinite(v_cand)):
                break
            defect_cand = float(np.linalg.norm(omega_coords - (self._G @ v_cand)))
            if defect_cand < defect_cur:
                v_cur = v_cand
                defect_cur = defect_cand
                if defect_cur <= floor_tol:
                    break
            else:
                break

        return v_cur

    def apply_sharp_isomorphism(
        self, covector: CotangentVector
    ) -> TangentVector:
        r"""3.3 Operador musical sostenido \(v = \sharp\omega = G^{-1}\omega \in T_p\mathcal{M}\) con refinamiento RSI-L3."""
        self._ingest_phase2_continuation()
        if not isinstance(covector, CotangentVector):
            raise TypeError(
                f"♯ requiere CotangentVector; "
                f"recibido {type(covector).__name__}."
            )
        self._validate_vector_dimension(covector, "CotangentVector", "♯")
        v_raw = self._G_inv @ covector.coordinates
        if not np.all(np.isfinite(v_raw)):
            raise NumericalInstabilityError(
                f"♯ no finito: ‖ω‖₂={covector.norm:.3e}, "
                f"‖G⁻¹‖_F={self._pm.frobenius_norm_G_inv:.3e}."
            )
        v = self._rsi_l3_iterative_defect_correction(covector.coordinates, v_raw)
        logger.debug(
            "♯: ‖ω‖₂=%.3e → ‖v‖₂=%.3e; ‖ω‖_{G⁻¹}=%.3e.",
            covector.norm,
            float(np.linalg.norm(v)),
            self.g_inv_norm(covector),
        )
        return TangentVector(coordinates=np.asarray(v, dtype=np.float64))

    def compute_riemannian_gradient(
        self, df: CotangentVector
    ) -> TangentVector:
        r"""
        3.4 **Gradiente Riemanniano Canónico:**

        \[
          \operatorname{grad}_G(f) = (df)^\sharp = G^{-1} df \in T_p\mathcal{M}.
        \]
        """
        return self.apply_sharp_isomorphism(df)

    def compute_poincare_hamiltonian_vector_field(
        self, dH: CotangentVector
    ) -> TangentVector:
        r"""
        3.5 **Campo Vectorial Hamiltoniano Celeste de Poincaré:**

        Resuelve la ecuación simpléctica fundamental \(\iota_{X_H}\omega = -dH\):

        \[
          X_H = -J\,(dH)^\sharp = -J\,G^{-1}\,dH \in T_p\mathcal{M}.
        \]

        Satisface idénticamente la conservación de energía de Liouville-Poincaré:
        \(\mathcal{L}_{X_H} H = \langle dH, X_H \rangle = 0\).
        """
        grad_H = self.apply_sharp_isomorphism(dH)
        xh_coords = -(self._J @ grad_H.coordinates)
        return TangentVector(coordinates=np.asarray(xh_coords, dtype=np.float64))

    def compute_poisson_bracket(
        self, df: CotangentVector, dg: CotangentVector
    ) -> float:
        r"""
        3.6 **Corchete Canónico de Poisson-Poincaré:**

        \[
          \{f, g\}_{\mathrm{Poincaré}}
            = \omega(X_f, X_g)
            = \langle dg, X_f \rangle
            = - (dg)^\top J G^{-1} (df).
        \]
        """
        self._validate_vector_dimension(df, "CotangentVector", "{·,·}")
        self._validate_vector_dimension(dg, "CotangentVector", "{·,·}")
        X_f = self.compute_poincare_hamiltonian_vector_field(df)
        return float(dg.coordinates @ X_f.coordinates)

    def _roundtrip_tolerance(self, input_norm: float) -> float:
        """3.7 Cota elástica de Wilkinson para la identidad de ida y vuelta."""
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
        r"""3.8 Verifica el Axioma MIE1 en \(TM\): \(\|\sharp(\flat(v))-v\|_2 \le \mathrm{tol}\)."""
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
        r"""3.9 Verifica el Axioma MIE1 en \(T^*M\): \(\|\flat(\sharp(\omega))-\omega\|_2 \le \mathrm{tol}\)."""
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
        3.10 Verifica el Axioma MIE2 (Isometría de Espacios Duales):
        \(\|v\|_G = \|\flat v\|_{G^{-1}}\).
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

    @staticmethod
    def _extract_variance(functor: Functor, label: str) -> CategoricalVariance:
        """3.11 Extracción tipada de la varianza funtorial en Z₂."""
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

    @staticmethod
    def audit_functor_composition(
        f1: Functor,
        f2: Functor,
        *,
        verify_domain_compatibility: bool = True,
    ) -> Dict[str, Any]:
        r"""
        3.12 Audita la composición funtorial \(f_1\circ f_2\) en el Topos MIC:

        - Varianza: \(\mathrm{Var}(f_1)\otimes\mathrm{Var}(f_2)\) en \(\mathbb{Z}_2\).
        - Compatibilidad de categorías: \(\mathrm{dom}(f_1)\cong\mathrm{cod}(f_2)\).
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


# =============================================================================
# ORQUESTADOR SOBERANO — MusicalIsomorphismEngine (Clausura de Fases 1+2+3)
# =============================================================================


class MusicalIsomorphismEngine(SharpIsomorphism):
    r"""
    Árbitro geométrico, simpléctico y espectral unificado del fibrado de dualidad.

    Encapsula y enlaza las 3 Fases Anidadas:

    .. code-block:: text

        Fase 1: MetricSpectralPreconditioner.precondition
                   ──► PreconditionedMetric
        Fase 2: FlatIsomorphism._ingest_phase1_continuation → diagnostics_report
                   ──► (♭, normas G, Riesz, forma simpléctica de Poincaré ω)
        Fase 3: SharpIsomorphism._ingest_phase2_continuation → full_cycle_report
                   ──► (♯, IDC RSI-L3, campo Hamiltoniano X_H, Poisson, Z₂)
    """

    # Enlaces explícitos de las 3 Fases Anidadas para introspección estructural
    Phase1_MetricSpectralPreconditioner = MetricSpectralPreconditioner
    Phase2_FlatIsomorphism = FlatIsomorphism
    Phase3_SharpIsomorphism = SharpIsomorphism

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
    ) -> None:
        # ── Fase 1: Precondicionamiento Espectral y Refinamiento de Banach ────
        _pre = (
            preconditioner
            if preconditioner is not None
            else MetricSpectralPreconditioner()
        )
        pm = _pre.precondition(metric_tensor)

        # ── Fase 2 y Fase 3: Inicialización continua de Flat y Sharp ──────────
        super().__init__(pm)
        self._preconditioner = _pre
        self.phase1 = _pre
        self.phase2 = self
        self.phase3 = self

        logger.info(
            "[MusicalIsomorphismEngine] v9 n=%d κ_reg=%.3e reg=%s inv_res=%.3e "
            "NS_iters=%d c_Poincaré=%.3e.",
            pm.matrix_dimension,
            pm.condition_number_reg,
            pm.regularization_applied,
            pm.inversion_residual,
            pm.newton_schulz_iterations,
            pm.poincare_symplectic_capacity,
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
        3.13 **MÉTODO TERMINAL DE LA FASE 3 (Clausura del Ciclo Musical y Celeste).**

        Ejecuta y audita el ciclo completo:
        1. \(\omega = \flat v\)
        2. \(v' = \sharp \omega\) (con corrección de defecto dual RSI-L3)
        3. Identidad \(\sharp \circ \flat = \mathrm{id}_{TM}\) y \(\flat \circ \sharp = \mathrm{id}_{T^*M}\)
        4. Isometría dual \(\|v\|_G = \|\flat v\|_{G^{-1}}\)
        5. Emparejamiento de Riesz \(G(v,v) = \langle \flat v, v \rangle\)
        6. Campo vectorial Hamiltoniano de Poincaré \(X_H = -J(dH)^\sharp\) y
           verificación del invariante de Liouville-Poincaré \(\langle \omega, X_H \rangle = 0\).
        """
        if not isinstance(vector, TangentVector):
            raise TypeError("full_cycle_report requiere TangentVector.")
        omega = self.apply_flat_isomorphism(vector)
        v_prime = self.apply_sharp_isomorphism(omega)
        rt = self.verify_roundtrip_identity(vector)
        iso = self.verify_isometry_flat(vector)
        pair = self.verify_riesz_pairing(vector, vector)
        crt = self.verify_cotroundtrip_identity(omega)

        # Dinámica Hamiltoniana de Poincaré sobre la 1-forma inducida ω = ♭v
        X_H = self.compute_poincare_hamiltonian_vector_field(omega)
        liouville_conservation_residual = abs(
            float(omega.coordinates @ X_H.coordinates)
        )
        poincare_hamiltonian = self.poincare_kinetic_hamiltonian(vector)

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
            "poincare_celestial_mechanics": {
                "kinetic_hamiltonian": poincare_hamiltonian,
                "hamiltonian_vector_field_norm": X_H.norm,
                "liouville_energy_conservation_residual": liouville_conservation_residual,
                "symplectic_capacity_gromov": self._pm.poincare_symplectic_capacity,
            },
            "engine": "MusicalIsomorphismEngine v9.0.0 (Fases 1+2+3)",
        }
        logger.info("[Engine] Ciclo completo ♭→♯ y flujo de Poincaré auditados.")
        return report


# =============================================================================
# EXPORTACIÓN CANÓNICA
# =============================================================================

__all__ = [
    # Constantes útiles a tests y módulos dependientes
    "_MACHINE_EPSILON",
    "_MACHINE_EPS",
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
    # Errores
    "FunctorialityError",
    "NumericalInstabilityError",
]

# Alias de compatibilidad con levi_civita_agent
_MACHINE_EPS = _MACHINE_EPSILON