# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sheaf Laplacian Solver (Soberano Agéntico del Haz Celular)          ║
║ Ruta   : app/agents/alfa/sheaf_laplacian_solver.py                           ║
║ Versión: 6.1.0-SOVEREIGN-RSI3-POINCARE-FREDHOLM-HEYTING-ESP32-PhD            ║
║ Función: Soberano de Gobernanza Celular y Resolución del Laplaciano del Haz  ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE GOBERNANZA AGÉNTICA EN 3 FASES ANIDADAS E INDIVISIBLES:
══════════════════════════════════════════════════════════════════════════════════
FASE 1: Precondicionamiento Métrico Simpléctico, Geometría Celeste de Poincaré
        y Reducción Conforme de Maupertuis-Jacobi.
        - Invarianza de Liouville-Darboux: \(M^\top\mathbf{\Omega}M=\mathbf{\Omega}\), \(\det M=+1\).
        - Monodromía de Floquet-Krein, discriminante de Hill y estabilidad elíptica KAM.
        - Métrica conforme de Maupertuis-Jacobi: \(\tilde G=2(H_0-V(q))G\).
        - Acción de Polyakov / longitud de Jacobi en el anillo de Novikov \(\Lambda_{\mathrm{Nov}}\).
        - Proyección de Higham sobre \(\operatorname{Sym}^+(m,\mathbb{R})\) y Cholesky-Banach.
        - INTERFAZ TERMINAL: `Phase1SymplecticMaupertuisState`.

FASE 2: Ensamblaje del Laplaciano del Haz, Análisis Espectral Fredholm y Banach.
        - CONTINUACIÓN FORMAL: ingesta estricta de `Phase1SymplecticMaupertuisState`.
        - Coborde renormalizado \(B_G=L_G^{-1}\delta\) y \(L_F=B_G^\top B_G\succeq 0\).
        - Hodge-Helmholtz: \(x=x_{\mathrm{harm}}+L_F^+\rho\), alternativa de Fredholm.
        - Torsión de Ray-Singer \(\log\det'(L_F)=-\zeta'(0)\) y traza de calor \(Z(\tau)\).
        - INTERFAZ TERMINAL: `Phase2BanachFredholmState`.

FASE 3: Gobernanza OODA, Retículo de Heyting \(\Omega_3\), Crowbar ESP32 y RSI³.
        - CONTINUACIÓN FORMAL: ingesta estricta de `Phase2BanachFredholmState`.
        - \(R_1\) Ricci-DeTurck, \(R_2\) Daleckii-Krein, \(R_3\) Brockett-Cayley.
        - Colapso \(\Omega_3=\{\mathrm{COHERENT},\mathrm{DEGRADED},\mathrm{VETOED}\}\).
        - Disyuntor ciber-físico: enclavamiento BT151 / ESP32 con SLA \(t\le 398.95\,\mathrm{ns}\).
        - INTERFAZ TERMINAL: `SheafSolverState`.
══════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Any, Final, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Compatibilidad de morfismos categóricos y álgebra de topos
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos \(\mathbf{E}_{\mathrm{MIC}}\)."""

    class Morphism:
        r"""Clase base de morfismos del Topos \(\mathbf{E}_{\mathrm{MIC}}\)."""


# ─────────────────────────────────────────────────────────────────────────────
# Motor espectral FPU (app/alfa/sheaf_laplacian_engine.py) — enlace opcional
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.alfa.sheaf_laplacian_engine import (
        SheafLaplacianEngine,
        PoincareMaupertuisSheafState,
        SpectralBanachSheafState,
        RSI3ConvergenceState,
        RicciDeTurckFlowState,
        DaleckiiKreinResolventState,
        BrockettShahshahaniState,
        PoincareLiouvilleAudit,
        SheafEngineSpectrumReport,
    )
except ImportError:  # pragma: no cover
    try:
        from app.physics.sheaf_laplacian_engine import (
            SheafLaplacianEngine,
            PoincareMaupertuisSheafState,
            SpectralBanachSheafState,
            RSI3ConvergenceState,
            RicciDeTurckFlowState,
            DaleckiiKreinResolventState,
            BrockettShahshahaniState,
            PoincareLiouvilleAudit,
            SheafEngineSpectrumReport,
        )
    except ImportError:
        SheafLaplacianEngine = None  # type: ignore[misc, assignment]
        PoincareMaupertuisSheafState = None  # type: ignore[misc, assignment]
        SpectralBanachSheafState = None  # type: ignore[misc, assignment]
        RSI3ConvergenceState = None  # type: ignore[misc, assignment]
        RicciDeTurckFlowState = None  # type: ignore[misc, assignment]
        DaleckiiKreinResolventState = None  # type: ignore[misc, assignment]
        BrockettShahshahaniState = None  # type: ignore[misc, assignment]
        PoincareLiouvilleAudit = None  # type: ignore[misc, assignment]
        SheafEngineSpectrumReport = None  # type: ignore[misc, assignment]

try:
    from scipy.sparse import csr_matrix, issparse
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    csr_matrix = None  # type: ignore[misc, assignment]
    issparse = None  # type: ignore[misc, assignment]

logger = logging.getLogger("MIC.Alpha.SheafLaplacianSolver")

__version__: Final[str] = "6.1.0-SOVEREIGN-RSI3-POINCARE-FREDHOLM-HEYTING-ESP32-PhD"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS UNIVERSALES Y UMBRALES DE HARDWARE
# ═══════════════════════════════════════════════════════════════════════════════
_WILKINSON_TOL: Final[float] = 1.0e-12
_DEFAULT_TOL: Final[float] = 1.0e-12
_KAPPA_MAX: Final[float] = 1.0e8
_HARD_LIPSCHITZ_CEILING: Final[float] = 1.0e6
_SYMPLECTIC_TOLERANCE: Final[float] = 1.0e-9
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12
_NOVIKOV_ENERGY_FLOOR: Final[float] = 1.0e-10
_SPARSE_THRESHOLD: Final[float] = 0.3
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 398.95
_KREIN_UNIT_CIRCLE_TOL: Final[float] = 1.0e-5
_FLOQUET_PAIR_TOL: Final[float] = 1.0e-6
_CAYLEY_STEP_CEILING: Final[float] = 0.25
_LOGDET_CLIP: Final[float] = 700.0
_HEAT_TAU_DEFAULT: Final[float] = 0.1

MatrixLike = Union[NDArray[np.float64], Any]


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES Y FUNTORES DE ERROR
# ═══════════════════════════════════════════════════════════════════════════════
class SheafLaplacianSolverError(Exception):
    r"""Excepción raíz para el Soberano Agéntico del Haz Celular."""


class MetricNonSPDError(SheafLaplacianSolverError):
    r"""El tensor métrico viola la condición simétrica definida positiva."""


class CohomologicalDimensionMismatchError(SheafLaplacianSolverError):
    r"""Inconsistencia dimensional entre el operador coborde \(\delta\) y el fibrado métrico."""


class FredholmSolvabilityError(SheafLaplacianSolverError):
    r"""Violación estricta de la alternativa de Fredholm: \(\langle\rho,\ker L_F\rangle\neq 0\)."""


class SymplecticInvariantViolationError(SheafLaplacianSolverError):
    r"""Violación del teorema de Liouville o deriva en la 2-forma de Darboux."""


class LipschitzStabilityViolationError(SheafLaplacianSolverError):
    r"""La proyección de Hodge supera la cota mesoscópica de Connes-Daleckii-Krein."""


class HardwareCrowbarTriggered(SheafLaplacianSolverError):
    r"""Disparo irreversible del disyuntor por hardware BT151 / ESP32 (SLA \(<400\,\mathrm{ns}\))."""


# ═══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DISTRIBUTIVO DE HEYTING \(\Omega_3\)
# ═══════════════════════════════════════════════════════════════════════════════
class StrictHeytingVerdict(IntEnum):
    r"""
    Álgebra de Heyting finita (cadena de Gödel dualizada por severidad)

    \[
    \Omega_3=\{\mathrm{COHERENT}=0,\;\mathrm{DEGRADED}=1,\;\mathrm{VETOED}=2\}.
    \]

    Axiomas (orden de severidad \(0\le 1\le 2\)):
      - Supremo (join, \(\lor\)): \(\sup(a,b)=\max(a,b)\) — prevalece la peor severidad.
      - Ínfimo (meet, \(\land\)): \(\inf(a,b)=\min(a,b)\).
      - Implicación de Heyting co-intuicionista (documentada):
        \(a\to b=\top(0)\) si \(a\le b\); en otro caso \(b\).
      - Neutro de join: \(\mathrm{COHERENT}\). Neutro de meet: \(\mathrm{VETOED}\).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "StrictHeytingVerdict") -> "StrictHeytingVerdict":
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @classmethod
    def infimum(cls, *verdicts: "StrictHeytingVerdict") -> "StrictHeytingVerdict":
        if not verdicts:
            return cls.COHERENT
        return cls(min(int(v) for v in verdicts))

    def implies(self, other: "StrictHeytingVerdict") -> "StrictHeytingVerdict":
        r"""Implicación \(a\to b=\top\) si \(a\le b\), si no \(b\)."""
        if int(self) <= int(other):
            return StrictHeytingVerdict.COHERENT
        return other

    @property
    def is_vetoed(self) -> bool:
        return self == StrictHeytingVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == StrictHeytingVerdict.DEGRADED

    @property
    def is_coherent(self) -> bool:
        return self == StrictHeytingVerdict.COHERENT


SheafSolverVerdict = StrictHeytingVerdict


class SolverAction(Enum):
    r"""Acciones de mitigación operativa dictadas por la gobernanza OODA."""

    NONE = auto()
    REGULARIZE_METRIC = auto()
    TRIGGER_HARDWARE_CROWBAR = auto()
    HALT_SOLVER = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES: ESTADOS INTER-FASE Y CERTIFICADOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Phase1SymplecticMaupertuisState:
    r"""
    Estado terminal inmutable emitido por la FASE 1.

    Puente analítico formal hacia la FASE 2: métrica de Jacobi, factor de
    Cholesky, coborde celular, monodromía de Poincaré y veredicto \(\Omega_3\).
    """

    maupertuis_metric_G: NDArray[np.float64]
    cholesky_factor_L_G: NDArray[np.float64]
    canonical_coboundary_delta: MatrixLike
    symplectic_jacobian_M: NDArray[np.float64]
    symplectic_drift: float
    is_symplectic_conserved: bool
    polyakov_action: float
    metric_condition_number: float
    higham_regularization_applied: bool
    higham_shift: float
    floquet_multipliers: NDArray[np.complex128]
    verdict: StrictHeytingVerdict
    kam_stability_index: float = 0.0
    novikov_regularized: bool = False
    hill_discriminant: float = 0.0
    log_abs_determinant: float = 0.0
    krein_signature_stable: bool = True
    jacobi_length: float = 0.0
    kinematic_action_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase2BanachFredholmState:
    r"""
    Estado terminal inmutable emitido por la FASE 2.

    Puente analítico formal hacia la FASE 3: espectro de \(L_F\), solubilidad
    de Fredholm, solución de Hodge y termodinámica de Ray-Singer.
    """

    phase1_state: Phase1SymplecticMaupertuisState
    laplacian_L_F: MatrixLike
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    fiedler_value: float
    spectral_gap: float
    kernel_dimension: int
    laplacian_rank: int
    pseudo_inverse_L_plus: NDArray[np.float64]
    exact_solution_x: NDArray[np.float64]
    harmonic_projection: NDArray[np.float64]
    fredholm_residual: float
    is_fredholm_soluble: bool
    heat_trace: float
    ray_singer_determinant: float
    sparse_computation: bool
    verdict: StrictHeytingVerdict
    log_ray_singer_determinant: float = 0.0
    connes_lipschitz_bound: float = 0.0
    harmonic_component_x: Optional[NDArray[np.float64]] = None
    coexact_component_x: Optional[NDArray[np.float64]] = None


@dataclass(frozen=True, slots=True)
class Phase1MetricCertificate:
    r"""Certificado formal de métrica y geometría simpléctica de Fase 1."""

    metric_condition_number: float
    is_metric_spd: bool
    higham_regularization_applied: bool
    higham_shift: float
    cholesky_factor_norm: float
    symplectic_drift: float
    is_symplectic_conserved: bool
    polyakov_action: float
    verdict: StrictHeytingVerdict
    kam_stability_index: float = 0.0
    novikov_regularized: bool = False
    hill_discriminant: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase2SpectralCertificate:
    r"""Certificado formal de descomposición espectral y Fredholm de Fase 2."""

    condition_number: float
    eigenvalues: NDArray[np.float64]
    fiedler_value: float
    spectral_gap: float
    kernel_dimension: int
    laplacian_rank: int
    fredholm_residual: float
    is_fredholm_soluble: bool
    heat_trace: float
    ray_singer_determinant: float
    sparse_computation: bool
    verdict: StrictHeytingVerdict
    log_ray_singer_determinant: float = 0.0
    connes_lipschitz_bound: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase3HodgeCertificate:
    r"""Certificado formal de proyección de Hodge, Lipschitz y RSI³ de Fase 3."""

    projected_state: NDArray[np.float64]
    projection_residual: float
    is_minimum_norm: bool
    poincare_constant: float
    connes_lipschitz_bound: float
    is_lipschitz_stable: bool
    ricci_scalar_curvature: float
    verdict: StrictHeytingVerdict
    brockett_casimir_error: float = 0.0
    lyapunov_energy: float = 0.0
    moduli_distance_delta: float = 0.0
    daleckii_frechet_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class SheafRSI3PoincareCertificate:
    r"""Certificado inmutable holístico de gobernanza simpléctica y espectral."""

    fredholm_residual: float
    betti_1_cycles: int
    fiedler_connectivity: float
    symplectic_volume_drift: float
    connes_lipschitz_bound: float
    ricci_scalar_curvature: float
    heyting_verdict: StrictHeytingVerdict
    sha256_telemetry_digest: str
    is_causally_coherent: bool
    kam_stability_index: float = 0.0
    hill_discriminant: float = 0.0
    lyapunov_energy: float = 0.0


@dataclass(frozen=True, slots=True)
class SheafSolverState:
    r"""Estado final inmutable de gobernanza agéntica sobre el haz celular."""

    phase1_certificate: Phase1MetricCertificate
    phase2_certificate: Phase2SpectralCertificate
    phase3_certificate: Phase3HodgeCertificate
    poincare_certificate: SheafRSI3PoincareCertificate
    final_verdict: StrictHeytingVerdict
    solver_action: SolverAction
    hardware_crowbar_latched: bool
    actuation_latency_ns: float
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str
    exact_solution_x: Optional[NDArray[np.float64]] = None
    crowbar_sla_met: bool = True
    rsi3_iterations: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITARIOS DE SOPORTE MATRICIAL, DISPERSO Y ESPECTRAL
# ═══════════════════════════════════════════════════════════════════════════════

def _is_sparse_matrix(matrix: Any) -> bool:
    return bool(_SPARSE_AVAILABLE and issparse is not None and issparse(matrix))


def _as_dense(matrix: MatrixLike) -> NDArray[np.float64]:
    if _is_sparse_matrix(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (matrix + matrix.T)


def _connes_lipschitz(fiedler: float) -> float:
    r"""Cota de Lipschitz de Connes: \(L_{\max}\le\tfrac12\lambda_2^{-3/2}\)."""
    if float(fiedler) <= _WILKINSON_TOL:
        return _HARD_LIPSCHITZ_CEILING
    return float(1.0 / (2.0 * (float(fiedler) ** 1.5)))


def _stable_logdet(matrix: NDArray[np.float64]) -> Tuple[float, float]:
    sign, logabs = np.linalg.slogdet(np.asarray(matrix, dtype=np.float64))
    return float(sign), float(logabs)


def _spectral_pinv(
    eigenvalues: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    floor: float,
) -> NDArray[np.float64]:
    inv_eigs = np.zeros_like(eigenvalues)
    positive = eigenvalues > floor
    inv_eigs[positive] = 1.0 / eigenvalues[positive]
    return _symmetrize((eigenvectors * inv_eigs) @ eigenvectors.T)


def _cayley_orthogonal(skew: NDArray[np.float64], step: float) -> NDArray[np.float64]:
    r"""
    Cayley \(Q=(I-\tfrac h2 K)(I+\tfrac h2 K)^{-1}\in\mathrm{SO}(n)\), \(K^\top=-K\).
    Polar de Higham (SVD) cancela la deriva de redondeo.
    """
    k = 0.5 * (skew - skew.T)
    n = k.shape[0]
    identity = np.eye(n, dtype=np.float64)
    half = 0.5 * float(step) * k
    try:
        cayley = la.solve((identity + half).T, (identity - half).T).T
    except la.LinAlgError:
        cayley = np.real(la.expm(-float(step) * k))
    u_left, _, vt = la.svd(cayley, full_matrices=False)
    return u_left @ vt


def _kernel_tolerance(eigenvalues: NDArray[np.float64], base: float) -> float:
    if eigenvalues.size == 0:
        return float(base)
    scale = float(np.max(eigenvalues)) if float(np.max(eigenvalues)) > 0.0 else 1.0
    return max(float(base), scale * float(np.finfo(np.float64).eps) * 10.0)


def stable_metric_higham(
    G: NDArray[np.float64],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR,
    target_condition: float = _KAPPA_MAX,
) -> Tuple[NDArray[np.float64], float, bool]:
    r"""
    Proyección convexa de Higham sobre \(\operatorname{Sym}^+(n,\mathbb{R})\):

    \[
    \tilde G=\arg\min_{M\succeq\varepsilon I}\|G-M\|_F,
    \]
    con recorte adicional de número de condición \(\kappa\le\kappa_{\max}\).
    """
    g_sym = _symmetrize(np.asarray(G, dtype=np.float64))
    try:
        eigvals, eigvecs = la.eigh(g_sym)
    except Exception as exc:
        raise MetricNonSPDError(
            "Fallo espectral en descomposición para proyección Higham."
        ) from exc

    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    shift = 0.0
    applied = False

    if lam_min <= tolerance:
        shift = abs(lam_min) + max(float(tolerance), lam_max / max(float(target_condition), 1.0))
        applied = True
        lam_min += shift
        lam_max += shift

    current_cond = lam_max / max(lam_min, float(np.finfo(np.float64).eps))
    if current_cond > target_condition and target_condition > 1.0:
        extra = (lam_max - target_condition * lam_min) / (target_condition - 1.0)
        if extra > 0.0:
            shift += extra
            applied = True

    vals_reg = eigvals + shift
    g_projected = _symmetrize((eigvecs * vals_reg) @ eigvecs.T)
    try:
        la.cholesky(g_projected, lower=True)
    except la.LinAlgError as exc:
        raise MetricNonSPDError(
            "Proyección Higham no alcanzó una matriz definida positiva."
        ) from exc
    return g_projected, float(shift), bool(applied)


def build_sparse_laplacian(
    delta: MatrixLike,
    L_G: NDArray[np.float64],
) -> Tuple[MatrixLike, NDArray[np.float64], bool]:
    r"""
    Ensambla \(L_F=\delta^\top G_{\mathrm{reg}}^{-1}\delta=B_G^\top B_G\)
    con \(B_G=L_G^{-1}\delta\) por sustitución triangular de Wilkinson.
    """
    sparse_used = False
    delta_dense = _as_dense(delta)
    b_g = la.solve_triangular(L_G, delta_dense, lower=True)

    nnz_ratio = float(np.count_nonzero(delta_dense)) / max(float(delta_dense.size), 1.0)
    if _SPARSE_AVAILABLE and csr_matrix is not None and (
        _is_sparse_matrix(delta) or nnz_ratio < _SPARSE_THRESHOLD
    ):
        sparse_used = True
        y_sparse = csr_matrix(b_g)
        laplacian = y_sparse.T @ y_sparse
    else:
        laplacian = _symmetrize(b_g.T @ b_g)
    return laplacian, b_g, sparse_used


def pair_floquet_multipliers_reciprocal(multipliers: NDArray[np.complex128]) -> float:
    r"""Residuo de la simetría simpléctica \(\mu\longleftrightarrow\mu^{-1}\)."""
    mu = np.asarray(multipliers, dtype=np.complex128).ravel()
    if mu.size == 0:
        return 0.0
    safe = np.where(np.abs(mu) < _WILKINSON_TOL, _WILKINSON_TOL, mu)
    inv = 1.0 / safe
    unused = np.ones(mu.size, dtype=bool)
    residual = 0.0
    for value in mu:
        distances = np.abs(inv - value)
        distances[~unused] = np.inf
        j = int(np.argmin(distances))
        unused[j] = False
        residual = max(residual, float(distances[j]) if np.isfinite(distances[j]) else 1.0)
    return float(residual)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1: PRECONDICIONAMIENTO MÉTRICO SIMPLÉCTICO, GEOMETRÍA CELESTE
#         DE POINCARÉ Y REDUCCIÓN CONFORME DE MAUPERTUIS-JACOBI
# ═══════════════════════════════════════════════════════════════════════════════

class Phase1_SymplecticMaupertuisPreconditioner(Morphism, ABC):
    r"""
    FASE 1: Precondicionamiento métrico simpléctico y reducción Maupertuis-Jacobi.

    Formaliza la geometría del espacio de fases cotangente \(T^*\mathcal{M}\),
    verifica Liouville-Darboux y calcula la deformación conforme que minimiza
    la acción geodésica de Maupertuis-Polyakov.
    """

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        try:
            super().__init__()
        except TypeError:
            pass
        self._tolerance = float(tolerance)
        self._higham_floor = _HIGHAM_REGULARIZATION_FLOOR

    def construct_canonical_symplectic_form(self, dimension: int) -> NDArray[np.float64]:
        r"""
        2-forma canónica de Darboux:

        \[
        \Omega=\begin{pmatrix}0&I\\-I&0\end{pmatrix},\qquad
        \Omega^\top=-\Omega,\quad\Omega^2=-I.
        \]
        """
        if dimension <= 0 or dimension % 2 != 0:
            raise CohomologicalDimensionMismatchError(
                f"La dimensión del espacio de fase debe ser par y positiva, recibido: {dimension}"
            )
        n = dimension // 2
        omega = np.zeros((dimension, dimension), dtype=np.float64)
        omega[:n, n:] = np.eye(n, dtype=np.float64)
        omega[n:, :n] = -np.eye(n, dtype=np.float64)
        return omega

    def construct_compatible_almost_complex_structure(
        self,
        dimension: int,
    ) -> NDArray[np.float64]:
        r"""Estructura casi-compleja compatible \(J=\Omega\), \(J^2=-I\)."""
        return self.construct_canonical_symplectic_form(dimension)

    def audit_symplectic_poincare_liouville(
        self,
        monodromy_M: NDArray[np.float64],
    ) -> Tuple[float, bool, float, NDArray[np.complex128], float, float, bool, float]:
        r"""
        Audita Liouville-Darboux \(M^\top\Omega M=\Omega\) (norma de Frobenius),
        \(\operatorname{slogdet}(M)\), multiplicadores de Floquet, índice KAM,
        discriminante de Hill \(\Delta=\operatorname{Tr}(M)\) y estabilidad de Krein.
        """
        jacobian = np.asarray(monodromy_M, dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
            raise CohomologicalDimensionMismatchError(
                f"La monodromía debe ser cuadrada, shape={jacobian.shape}"
            )
        dim = jacobian.shape[0]
        omega = self.construct_canonical_symplectic_form(dim)
        drift_matrix = jacobian.T @ omega @ jacobian - omega
        symplectic_drift = float(la.norm(drift_matrix, ord="fro"))
        is_conserved = symplectic_drift <= _SYMPLECTIC_TOLERANCE

        sign_m, log_abs_m = _stable_logdet(jacobian)
        if abs(log_abs_m) > 50.0:
            det_m = float(sign_m * np.exp(np.clip(log_abs_m, -50.0, 50.0)))
        else:
            det_m = float(sign_m * np.exp(log_abs_m))
        det_drift = abs(det_m - 1.0)

        floquet = np.asarray(la.eigvals(jacobian), dtype=np.complex128)
        magnitudes = np.abs(floquet)
        krein_stable = bool(np.all(np.abs(magnitudes - 1.0) < _KREIN_UNIT_CIRCLE_TOL))
        kam = float(1.0 / (1.0 + np.sum(np.abs(magnitudes - 1.0))))
        pair_res = pair_floquet_multipliers_reciprocal(floquet)
        if pair_res > _FLOQUET_PAIR_TOL:
            kam *= 0.5
        hill = float(np.trace(jacobian))
        _ = det_drift
        return (
            symplectic_drift,
            is_conserved,
            det_drift,
            floquet,
            kam,
            hill,
            krein_stable,
            float(log_abs_m),
        )

    def compute_maupertuis_conformal_metric(
        self,
        base_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
    ) -> Tuple[NDArray[np.float64], bool, float]:
        r"""
        Transformación conforme de Maupertuis-Jacobi

        \[
        \tilde G_{\mu\nu}=2(H_0-V(q))G_{\mu\nu},
        \]
        con corte ultramétrico de Novikov si \(H_0\le V(q)\).
        """
        g_sym = _symmetrize(np.asarray(base_G, dtype=np.float64))
        kinematic = float(total_energy_H0) - float(potential_V)
        novikov_hit = kinematic <= _NOVIKOV_ENERGY_FLOOR
        if novikov_hit:
            scale = _NOVIKOV_ENERGY_FLOOR * np.exp(
                (kinematic - _NOVIKOV_ENERGY_FLOOR) / _NOVIKOV_ENERGY_FLOOR
            )
        else:
            scale = kinematic
        conformal = _symmetrize((2.0 * float(scale)) * g_sym)
        return conformal, bool(novikov_hit), float(max(_NOVIKOV_ENERGY_FLOOR, kinematic))

    def compute_polyakov_geodesic_action(
        self,
        trajectory_x: Optional[NDArray[np.float64]],
        metric_G: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Acción de Polyakov discretizada y longitud de Jacobi:

        \[
        S_E[x]=\tfrac12\sum_k\Delta x_k^\top G\Delta x_k,\qquad
        L_J=\sum_k\sqrt{\Delta x_k^\top G\Delta x_k}.
        \]
        """
        if trajectory_x is None:
            return 0.0, 0.0
        traj = np.asarray(trajectory_x, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[0] < 2:
            return 0.0, 0.0
        g_metric = np.asarray(metric_G, dtype=np.float64)
        if traj.shape[1] != g_metric.shape[0]:
            raise CohomologicalDimensionMismatchError(
                f"Trayectoria d={traj.shape[1]} incompatible con G {g_metric.shape}."
            )
        velocities = np.diff(traj, axis=0)
        quadratic = np.einsum("ni,ij,nj->n", velocities, g_metric, velocities)
        quadratic = np.maximum(quadratic, 0.0)
        polyakov = 0.5 * float(np.sum(quadratic))
        jacobi = float(np.sum(np.sqrt(quadratic)))
        return polyakov, jacobi

    def execute_phase1_symplectic_maupertuis_reduction(
        self,
        coboundary_delta: MatrixLike,
        base_metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        jacobian_M: NDArray[np.float64],
        trajectory_x: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX,
    ) -> Phase1SymplecticMaupertuisState:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 1
        (puente indivisible: su imagen es el único alimento de la Fase 2).

        Sintetiza Maupertuis-Jacobi, auditoría de Poincaré, Higham y Cholesky-Banach,
        y emite `Phase1SymplecticMaupertuisState`.
        """
        delta = coboundary_delta
        m = int(delta.shape[0])
        g_base = np.asarray(base_metric_G, dtype=np.float64)
        if g_base.shape != (m, m):
            raise CohomologicalDimensionMismatchError(
                f"delta tiene {m} filas pero base_metric_G es {g_base.shape[0]}×{g_base.shape[1]}."
            )

        g_maupertuis, novikov_hit, kinematic_scale = self.compute_maupertuis_conformal_metric(
            g_base, potential_V, total_energy_H0
        )
        g_spd, higham_shift, higham_applied = stable_metric_higham(
            g_maupertuis, self._higham_floor, target_condition
        )
        try:
            chol_g = la.cholesky(g_spd, lower=True)
        except la.LinAlgError as exc:
            raise MetricNonSPDError(
                f"Fallo en factorización de Cholesky post-Higham: {exc}"
            ) from exc

        eigvals_proj = la.eigvalsh(g_spd)
        cond_g = float(np.max(eigvals_proj) / max(float(np.min(eigvals_proj)), _WILKINSON_TOL))
        cond_g = min(cond_g, _KAPPA_MAX * 10.0)

        (
            sym_drift,
            is_sym,
            det_drift,
            floquet,
            kam,
            hill,
            krein_stable,
            log_abs_det,
        ) = self.audit_symplectic_poincare_liouville(np.asarray(jacobian_M, dtype=np.float64))

        polyakov, jacobi = self.compute_polyakov_geodesic_action(trajectory_x, g_spd)

        verdict = StrictHeytingVerdict.COHERENT
        if not np.isfinite(cond_g) or not is_sym or det_drift > 1.0e-3:
            verdict = StrictHeytingVerdict.VETOED
        elif higham_applied or cond_g > target_condition or not krein_stable:
            verdict = StrictHeytingVerdict.DEGRADED

        logger.info(
            "Fase 1→2: m=%d, κ(G)=%.3e, Higham=%s, Novikov=%s, KAM=%.4f, Hill=%.4f, Ω₃=%s",
            m,
            cond_g,
            higham_applied,
            novikov_hit,
            kam,
            hill,
            verdict.name,
        )
        return Phase1SymplecticMaupertuisState(
            maupertuis_metric_G=g_spd,
            cholesky_factor_L_G=chol_g,
            canonical_coboundary_delta=delta,
            symplectic_jacobian_M=np.asarray(jacobian_M, dtype=np.float64),
            symplectic_drift=sym_drift,
            is_symplectic_conserved=is_sym,
            polyakov_action=polyakov,
            metric_condition_number=cond_g,
            higham_regularization_applied=higham_applied,
            higham_shift=higham_shift,
            floquet_multipliers=floquet,
            verdict=verdict,
            kam_stability_index=kam,
            novikov_regularized=novikov_hit,
            hill_discriminant=hill,
            log_abs_determinant=log_abs_det,
            krein_signature_stable=krein_stable,
            jacobi_length=jacobi,
            kinematic_action_scale=kinematic_scale,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2: ENSAMBLAJE DEL LAPLACIANO DEL HAZ CELULAR, ANÁLISIS ESPECTRAL
#         FREDHOLM Y CUANTIZACIÓN DE BANACH
#         (inicio = consumo de Phase1SymplecticMaupertuisState)
# ═══════════════════════════════════════════════════════════════════════════════

class Phase2_SheafLaplacianFredholmEngine(Phase1_SymplecticMaupertuisPreconditioner):
    r"""
    FASE 2: Ensamblaje del laplaciano del haz, Fredholm y cuantización.

    Consume el estado terminal de la Fase 1, construye \(L_F\), analiza su
    espectro, evalúa la alternativa de Fredholm y descompone Hodge-Helmholtz.
    """

    def initialize_phase2_from_maupertuis(
        self,
        phase1_state: Phase1SymplecticMaupertuisState,
    ) -> Tuple[MatrixLike, NDArray[np.float64], bool]:
        r"""
        MÉTODO DE INICIO FORMAL DE LA FASE 2.

        Continuación indivisible de `execute_phase1_symplectic_maupertuis_reduction`.
        Ensambla \(L_F=B_G^\top B_G\) a partir del Cholesky y el coborde de Fase 1.
        """
        return build_sparse_laplacian(
            delta=phase1_state.canonical_coboundary_delta,
            L_G=phase1_state.cholesky_factor_L_G,
        )

    def verify_fredholm_solvability(
        self,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
        source_rho: NDArray[np.float64],
        tolerance: float = _WILKINSON_TOL,
    ) -> Tuple[bool, float, NDArray[np.float64]]:
        r"""
        Axioma de Fredholm en Hilbert:

        \[
        \rho\in\operatorname{im} L_F\iff\langle\rho,v_k\rangle=0\quad\forall v_k\in\ker L_F.
        \]
        El residual es \(\|P_{\ker}\rho\|_2\).
        """
        rho = np.asarray(source_rho, dtype=np.float64).ravel()
        null_indices = np.where(eigenvalues <= tolerance)[0]
        if null_indices.size == 0:
            return True, 0.0, np.zeros_like(rho)
        kernel_basis = eigenvectors[:, null_indices]
        harmonic_projection = kernel_basis @ (kernel_basis.T @ rho)
        residual = float(la.norm(harmonic_projection, ord=2))
        return bool(residual <= tolerance), residual, harmonic_projection

    def hodge_helmholtz_decompose(
        self,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
        source_rho: NDArray[np.float64],
        x_initial: Optional[NDArray[np.float64]],
        kernel_tol: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Descomposición de Hodge-Helmholtz sobre cocadenas de grado 0:

        \[
        x=x_{\mathrm{harm}}+x_{\mathrm{coexact}},\qquad
        x_{\mathrm{coexact}}=L_F^+\rho\in(\ker L_F)^\perp,\quad
        x_{\mathrm{harm}}=P_{\ker}x_{\mathrm{init}}.
        \]
        Devuelve \((x,\;L^+,\;x_{\mathrm{harm}},\;x_{\mathrm{coexact}})\).
        """
        pinv_l = _spectral_pinv(eigenvalues, eigenvectors, kernel_tol)
        rho = np.asarray(source_rho, dtype=np.float64).ravel()
        x_coexact = pinv_l @ rho
        n = rho.shape[0]
        x_harm = np.zeros(n, dtype=np.float64)
        if x_initial is not None:
            x0 = np.asarray(x_initial, dtype=np.float64).ravel()
            if x0.shape[0] != n:
                raise CohomologicalDimensionMismatchError(
                    f"x_initial tiene dim {x0.shape[0]}, esperado {n}."
                )
            mask = eigenvalues <= kernel_tol
            if np.any(mask):
                kernel_basis = eigenvectors[:, mask]
                x_harm = kernel_basis @ (kernel_basis.T @ x0)
        exact_x = x_coexact + x_harm
        return exact_x, pinv_l, x_harm, x_coexact

    def compute_heat_kernel_thermodynamics(
        self,
        eigenvalues: NDArray[np.float64],
        tau: float = _HEAT_TAU_DEFAULT,
    ) -> Tuple[float, float, float]:
        r"""
        Traza de calor y torsión analítica de Ray-Singer (vía \(\zeta'(0)\)):

        \[
        Z(\tau)=\sum_i e^{-\tau\lambda_i},\qquad
        \log\det'(L_F)=\sum_{\lambda_i>\varepsilon}\log\lambda_i=-\zeta'_{L_F}(0).
        \]
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        heat_trace = float(np.sum(np.exp(-float(tau) * eigs)))
        positive = eigs[eigs > _WILKINSON_TOL]
        if positive.size == 0:
            return heat_trace, 0.0, 0.0
        log_det = float(np.sum(np.log(np.maximum(positive, 1.0e-18))))
        ray_singer = float(np.exp(np.clip(log_det, -_LOGDET_CLIP, _LOGDET_CLIP)))
        return heat_trace, ray_singer, log_det

    def execute_phase2_banach_fredholm_synthesis(
        self,
        phase1_state: Phase1SymplecticMaupertuisState,
        source_rho: NDArray[np.float64],
        x_initial: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX,
        diffusion_time_tau: float = _HEAT_TAU_DEFAULT,
    ) -> Phase2BanachFredholmState:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 2
        (puente indivisible: su imagen es el único alimento de la Fase 3).

        Consume `Phase1SymplecticMaupertuisState`, resuelve \(L_F x=\rho\) en el
        sentido de Hodge y emite `Phase2BanachFredholmState`.
        """
        _ = target_condition
        laplacian, _, sparse_used = self.initialize_phase2_from_maupertuis(phase1_state)
        l_dense = _symmetrize(_as_dense(laplacian))
        n = l_dense.shape[0]
        rho = np.asarray(source_rho, dtype=np.float64).ravel()
        if rho.shape[0] != n:
            raise CohomologicalDimensionMismatchError(
                f"rho tiene dim {rho.shape[0]}, L_F es {n}×{n} (columnas de δ)."
            )

        eigs_l, q_l = la.eigh(l_dense)
        clean_eigs = np.maximum(0.0, eigs_l)
        tol_kernel = _kernel_tolerance(clean_eigs, max(self._tolerance * 100.0, _WILKINSON_TOL))
        kernel_dim = int(np.sum(clean_eigs <= tol_kernel))
        fiedler = float(clean_eigs[1]) if clean_eigs.size > 1 else 0.0
        spectral_gap = fiedler - float(clean_eigs[0])
        rank_l = int(np.sum(clean_eigs > tol_kernel))

        is_fredholm, fredholm_res, harm_proj_rho = self.verify_fredholm_solvability(
            clean_eigs, q_l, rho, tol_kernel
        )
        exact_x, pinv_l, x_harm, x_coexact = self.hodge_helmholtz_decompose(
            clean_eigs, q_l, rho, x_initial, tol_kernel
        )
        heat_trace, ray_singer, log_det = self.compute_heat_kernel_thermodynamics(
            clean_eigs, diffusion_time_tau
        )
        connes_bound = _connes_lipschitz(fiedler)

        verdict = StrictHeytingVerdict.COHERENT
        if not is_fredholm or not np.isfinite(spectral_gap):
            verdict = StrictHeytingVerdict.VETOED
        elif kernel_dim > 1 or phase1_state.verdict.is_degraded:
            # dim ker = 1 es el Betti β₀ de un haz conexo; >1 indica desconexión.
            verdict = StrictHeytingVerdict.DEGRADED
        verdict = StrictHeytingVerdict.supremum(verdict, phase1_state.verdict)

        logger.info(
            "Fase 2→3: n=%d, λ₂=%.6e, ker=%d, Fredholm=%s res=%.3e, log det'=%.4f, Ω₃=%s",
            n,
            fiedler,
            kernel_dim,
            is_fredholm,
            fredholm_res,
            log_det,
            verdict.name,
        )
        return Phase2BanachFredholmState(
            phase1_state=phase1_state,
            laplacian_L_F=laplacian,
            eigenvalues=clean_eigs,
            eigenvectors=q_l,
            fiedler_value=fiedler,
            spectral_gap=spectral_gap,
            kernel_dimension=kernel_dim,
            laplacian_rank=rank_l,
            pseudo_inverse_L_plus=pinv_l,
            exact_solution_x=exact_x,
            harmonic_projection=harm_proj_rho,
            fredholm_residual=fredholm_res,
            is_fredholm_soluble=is_fredholm,
            heat_trace=heat_trace,
            ray_singer_determinant=ray_singer,
            sparse_computation=sparse_used,
            verdict=verdict,
            log_ray_singer_determinant=log_det,
            connes_lipschitz_bound=connes_bound,
            harmonic_component_x=x_harm,
            coexact_component_x=x_coexact,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3: GOBERNANZA AGÉNTICA OODA, RETÍCULO DE HEYTING \(\Omega_3\),
#         DISYUNTOR HARDWARE Y AUTOMEJORA RECURSIVA RSI³
#         (inicio = consumo de Phase2BanachFredholmState)
# ═══════════════════════════════════════════════════════════════════════════════

class Phase3_SovereignAgenticGovernance(Phase2_SheafLaplacianFredholmEngine):
    r"""
    FASE 3: Gobernanza OODA, retículo \(\Omega_3\), hardware crowbar y RSI³.

    Consume el estado terminal de la Fase 2, ejecuta \(R_3\circ R_2\circ R_1\),
    colapsa el veredicto en Heyting y supervisa el disyuntor ciber-físico.
    """

    def __init__(
        self,
        spectral_engine: Optional[Any] = None,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False,
    ) -> None:
        super().__init__(tolerance=tolerance)
        if spectral_engine is not None:
            self._engine = spectral_engine
        elif SheafLaplacianEngine is not None:
            self._engine = SheafLaplacianEngine()
        else:
            self._engine = None
        self._halt_on_veto = bool(halt_on_veto)
        self._hardware_latch: bool = False

    def initialize_phase3_from_fredholm(
        self,
        phase2_state: Phase2BanachFredholmState,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        MÉTODO DE INICIO FORMAL DE LA FASE 3.

        Continuación indivisible de `execute_phase2_banach_fredholm_synthesis`.
        Extrae \((G,\,L_F,\,x)\) como datos de gobierno RSI³ / OODA.
        """
        g_metric = phase2_state.phase1_state.maupertuis_metric_G
        l_dense = _symmetrize(_as_dense(phase2_state.laplacian_L_F))
        return g_metric, l_dense, phase2_state.exact_solution_x

    def execute_rsi1_ricci_deturck_flow(
        self,
        metric_G: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        step_tau: float = 0.01,
    ) -> NDArray[np.float64]:
        r"""
        RSI³ Nivel 1 (\(R_1\)): Ricci-DeTurck normalizado

        \[
        G_{k+1}=G_k-2\Delta\tau\Bigl(\operatorname{Ric}-\tfrac{R_{\mathrm{avg}}}{n}G_k\Bigr),
        \]
        proyectado de nuevo al cono SPD (Higham) y renormalizado en volumen.
        """
        g_metric = _symmetrize(np.asarray(metric_G, dtype=np.float64))
        ricci = _symmetrize(np.asarray(ricci_tensor, dtype=np.float64))
        n = g_metric.shape[0]
        try:
            inv_g = la.inv(g_metric + _HIGHAM_REGULARIZATION_FLOOR * np.eye(n))
        except la.LinAlgError:
            inv_g = la.pinv(g_metric)
        scalar_r = float(np.trace(inv_g @ ricci))
        r_avg = scalar_r / float(max(n, 1))
        g_next = g_metric - 2.0 * float(step_tau) * (ricci - r_avg * g_metric)
        g_next_spd, _, _ = stable_metric_higham(g_next)
        sign_b, log_b = _stable_logdet(g_metric)
        sign_a, log_a = _stable_logdet(g_next_spd)
        _ = (sign_b, sign_a)
        if n > 0 and np.isfinite(log_b) and np.isfinite(log_a):
            scale = np.exp((log_b - log_a) / float(n))
            if np.isfinite(scale) and scale > 0.0:
                g_next_spd, _, _ = stable_metric_higham(g_next_spd * scale)
        return g_next_spd

    def execute_rsi2_connes_daleckii_krein_bound(self, fiedler_value: float) -> float:
        r"""RSI³ Nivel 2 (\(R_2\)): cota Fréchet de Connes \(1/(2\lambda_2^{3/2})\)."""
        return _connes_lipschitz(fiedler_value)

    def execute_rsi2_daleckii_krein_frechet(
        self,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
        perturbation_h: NDArray[np.float64],
        tau: float = _HEAT_TAU_DEFAULT,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Fórmula espectral de Daleckii-Krein para \(f(\lambda)=e^{-\tau\lambda}\):

        \[
        Df(L)[H]=Q\bigl(f^{[1]}(\lambda_i,\lambda_j)\circ(Q^\top HQ)\bigr)Q^\top.
        \]
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        q_basis = np.asarray(eigenvectors, dtype=np.float64)
        n = eigs.size
        h_sym = _symmetrize(np.asarray(perturbation_h, dtype=np.float64))
        if h_sym.shape != (n, n):
            h_work = np.zeros((n, n), dtype=np.float64)
            m = min(n, h_sym.shape[0], h_sym.shape[1])
            h_work[:m, :m] = h_sym[:m, :m]
            h_sym = _symmetrize(h_work)
        values_f = np.exp(-float(tau) * eigs)
        derivs_f = -float(tau) * values_f
        lam_i = eigs[:, None]
        lam_j = eigs[None, :]
        delta = lam_i - lam_j
        close = np.abs(delta) <= _WILKINSON_TOL
        divided = np.zeros_like(delta, dtype=np.float64)
        far = ~close
        divided[far] = (values_f[:, None] - values_f[None, :])[far] / delta[far]
        divided[close] = 0.5 * (derivs_f[:, None] + derivs_f[None, :])[close]
        h_energy = q_basis.T @ h_sym @ q_basis
        frechet = _symmetrize(q_basis @ (divided * h_energy) @ q_basis.T)
        return frechet, float(la.norm(frechet, ord=2))

    def execute_rsi3_brockett_shahshahani_flow(
        self,
        potential_p: NDArray[np.float64],
        laplacian_L: Optional[NDArray[np.float64]] = None,
        kernel_matrix: Optional[NDArray[np.float64]] = None,
        step_dt: float = 0.005,
        rho_density: Optional[NDArray[np.float64]] = None,
        integration_steps: int = 4,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        RSI³ Nivel 3 (\(R_3\)): flujo isoespectral de Brockett por conjugación de Cayley

        \[
        \dot L=[L,[L,N]],\qquad
        L_{k+1}=Q_k^\top L_k Q_k,\quad
        Q_k=\operatorname{Cayley}(-h[L_k,N]).
        \]

        Firma compatible con el legado \((p,\,L)\) y con el override histórico
        \((p,\,\rho,\,\mathrm{kernel})\): si `laplacian_L` es None se usa
        `kernel_matrix` o, en último término, `rho_density` si es cuadrada.
        """
        if laplacian_L is None:
            if kernel_matrix is not None:
                laplacian_L = kernel_matrix
            elif rho_density is not None and np.asarray(rho_density).ndim == 2:
                laplacian_L = rho_density
            else:
                raise CohomologicalDimensionMismatchError(
                    "RSI³ Brockett requiere un laplaciano cuadrado (laplacian_L)."
                )
        l_current = _symmetrize(np.asarray(laplacian_L, dtype=np.float64))
        n = l_current.shape[0]
        p_vec = np.asarray(potential_p, dtype=np.float64).ravel()
        if p_vec.shape[0] != n:
            p_work = np.zeros(n, dtype=np.float64)
            m = min(n, p_vec.shape[0])
            p_work[:m] = np.maximum(_WILKINSON_TOL, p_vec[:m])
            p_vec = p_work
        else:
            p_vec = np.maximum(_WILKINSON_TOL, p_vec)
        p_weights = p_vec / float(np.sum(p_vec))
        n_diag = np.diag(p_weights)
        dt = float(np.clip(step_dt, _WILKINSON_TOL, _CAYLEY_STEP_CEILING))
        for _ in range(max(int(integration_steps), 1)):
            inner = 0.5 * (
                (l_current @ n_diag - n_diag @ l_current)
                - (l_current @ n_diag - n_diag @ l_current).T
            )
            q_step = _cayley_orthogonal(inner, dt)
            l_current = _symmetrize(q_step.T @ l_current @ q_step)
        return p_weights, l_current

    def trigger_hardware_crowbar_interrupt(self) -> Tuple[float, bool]:
        r"""
        Enclava el disyuntor ciber-físico simulado (trayectoria de software del
        GPIO / tiristor BT151). Reporta la latencia medida y si cumple el SLA
        de \(398.95\,\mathrm{ns}\) — no se recorta artificialmente la medida.
        """
        start_ns = time.perf_counter_ns()
        self._hardware_latch = True
        elapsed_ns = float(time.perf_counter_ns() - start_ns)
        sla_met = elapsed_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS
        return elapsed_ns, sla_met

    def _discrete_ricci_from_hodge(
        self,
        metric_g: NDArray[np.float64],
        laplacian: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        n = metric_g.shape[0]
        identity = np.eye(n, dtype=np.float64)
        try:
            inv_g = la.inv(metric_g + _HIGHAM_REGULARIZATION_FLOOR * identity)
        except la.LinAlgError:
            inv_g = la.pinv(metric_g)
        ricci = 0.5 * (laplacian @ metric_g + metric_g @ laplacian) - (
            metric_g @ inv_g @ metric_g - metric_g
        )
        return _symmetrize(ricci)

    def _try_engine_rsi3(
        self,
        phase2_state: Phase2BanachFredholmState,
        potential_p: NDArray[np.float64],
    ) -> Optional[Any]:
        if self._engine is None or PoincareMaupertuisSheafState is None:
            return None
        generate = getattr(self._engine, "generate_poincare_maupertuis_coboundary_state", None)
        synthesize = getattr(self._engine, "synthesize_spectral_banach_state", None)
        converge = getattr(self._engine, "converge_rsi3_recursive_meta_optimization", None)
        if generate is None or synthesize is None or converge is None:
            return None
        p1 = phase2_state.phase1_state
        try:
            engine_p1 = generate(
                coboundary_delta=_as_dense(p1.canonical_coboundary_delta),
                base_metric_G=p1.maupertuis_metric_G,
                potential_V=0.0,
                total_energy_H0=max(p1.kinematic_action_scale, _NOVIKOV_ENERGY_FLOOR),
                monodromy_jacobian=p1.symplectic_jacobian_M,
            )
            engine_p2 = synthesize(engine_p1, diffusion_time_tau=_HEAT_TAU_DEFAULT)
            n_l = int(engine_p2.spectrum_report.laplacian_L_F.shape[0])
            p_vec = np.asarray(potential_p, dtype=np.float64).ravel()
            if p_vec.shape[0] != n_l:
                p_work = np.zeros(n_l, dtype=np.float64)
                m = min(n_l, p_vec.shape[0])
                p_work[:m] = p_vec[:m]
                p_vec = p_work
            return converge(
                banach_state=engine_p2,
                potential_vector_p=p_vec,
                outer_iterations=3,
                ricci_tau=0.005,
                brockett_dt=0.002,
            )
        except Exception as exc:
            logger.warning("Motor RSI³ no disponible / fallo de enlace: %s", exc)
            return None

    def execute_phase3_sovereign_governance_cycle(
        self,
        phase2_state: Phase2BanachFredholmState,
        potential_p: Optional[NDArray[np.float64]] = None,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
    ) -> SheafSolverState:
        r"""
        MÉTODO TERMINAL FORMAL DE LA FASE 3 (cierre de la gobernanza agéntica).

        Consume `Phase2BanachFredholmState`. Aplica RSI³, sintetiza certificados
        de las 3 fases, colapsa \(\Omega_3\) y activa el crowbar si hay VETO.

        OODA:
          Observe  → certificados F1/F2 y tensores RSI³.
          Orient   → retículo de Heyting (join de severidades).
          Decide   → `SolverAction`.
          Act      → regularización o enclavamiento hardware.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        g_metric, l_dense, exact_x = self.initialize_phase3_from_fredholm(phase2_state)
        p1 = phase2_state.phase1_state
        n = int(exact_x.shape[0])
        m_metric = int(g_metric.shape[0])

        if potential_p is None:
            potential_p = np.ones(n, dtype=np.float64) / float(max(n, 1))
        p_vec = np.asarray(potential_p, dtype=np.float64).ravel()

        if ricci_tensor is None:
            # Ricci discreto de Hodge si las dimensiones coinciden; si no, semilla isotrópica.
            if l_dense.shape == g_metric.shape:
                ricci_tensor = self._discrete_ricci_from_hodge(g_metric, l_dense)
            else:
                ricci_tensor = np.eye(m_metric, dtype=np.float64) * 0.01
        ricci = _symmetrize(np.asarray(ricci_tensor, dtype=np.float64))

        engine_rsi3 = self._try_engine_rsi3(phase2_state, p_vec)

        if engine_rsi3 is not None:
            g_evolved = engine_rsi3.ricci_flow_history.evolved_metric_G
            ricci_scalar = float(engine_rsi3.ricci_flow_history.scalar_curvature)
            connes_bound = float(
                engine_rsi3.daleckii_krein_state.perturbed_lipschitz_constant
            )
            frechet_norm = float(engine_rsi3.daleckii_krein_state.projector_shift_norm)
            casimir_err = float(engine_rsi3.brockett_state.isoespectral_trace_error)
            lyapunov = float(engine_rsi3.global_lyapunov_energy)
            moduli_delta = float(engine_rsi3.moduli_distance_delta)
            rsi3_iters = int(engine_rsi3.iterations_converged)
            l_brockett = engine_rsi3.brockett_state.diagonalized_orbit_L
        else:
            g_evolved = self.execute_rsi1_ricci_deturck_flow(g_metric, ricci)
            try:
                ricci_scalar = float(
                    np.trace(la.solve(g_metric, ricci, assume_a="pos"))
                )
            except Exception:
                ricci_scalar = float(np.trace(la.pinv(g_metric) @ ricci))
            connes_bound = self.execute_rsi2_connes_daleckii_krein_bound(
                phase2_state.fiedler_value
            )
            metric_perturbation = g_evolved - g_metric
            try:
                _, frechet_norm = self.execute_rsi2_daleckii_krein_frechet(
                    phase2_state.eigenvalues,
                    phase2_state.eigenvectors,
                    metric_perturbation,
                )
            except Exception:
                frechet_norm = float(la.norm(metric_perturbation, ord="fro"))
            _, l_brockett = self.execute_rsi3_brockett_shahshahani_flow(
                p_vec, l_dense, step_dt=0.005, integration_steps=4
            )
            casimir_err = float(
                np.max(
                    np.abs(
                        np.sort(la.eigvalsh(l_brockett))
                        - np.sort(phase2_state.eigenvalues)
                    )
                )
            ) if l_brockett.shape == l_dense.shape else 0.0
            n_diag = np.diag(p_vec) if p_vec.shape[0] == l_dense.shape[0] else np.eye(l_dense.shape[0])
            lyapunov = float(np.trace(l_dense @ n_diag)) if n_diag.shape[0] == l_dense.shape[0] else 0.0
            moduli_delta = float(la.norm(metric_perturbation, ord="fro"))
            rsi3_iters = 1
        _ = (g_evolved, l_brockett)

        # Observación Hodge / Lipschitz
        x_harm = (
            phase2_state.harmonic_component_x
            if phase2_state.harmonic_component_x is not None
            else phase2_state.harmonic_projection
        )
        if x_harm.shape[0] == l_dense.shape[0]:
            residual_proj = float(la.norm(l_dense @ x_harm))
        else:
            residual_proj = float(la.norm(x_harm))
        norm_orig = float(la.norm(exact_x))
        norm_proj = float(la.norm(x_harm))
        is_min_norm = norm_proj <= norm_orig + self._tolerance * 100.0
        poincare_const = norm_proj / max(norm_orig, _WILKINSON_TOL)
        is_lipschitz_stable = connes_bound < _HARD_LIPSCHITZ_CEILING

        verdict_p3 = StrictHeytingVerdict.COHERENT
        if not np.isfinite(residual_proj) or not is_lipschitz_stable:
            verdict_p3 = StrictHeytingVerdict.VETOED
        elif (not is_min_norm) or connes_bound > 1.0e4 or casimir_err > 1.0e-4:
            verdict_p3 = StrictHeytingVerdict.DEGRADED

        cert1 = Phase1MetricCertificate(
            metric_condition_number=p1.metric_condition_number,
            is_metric_spd=True,
            higham_regularization_applied=p1.higham_regularization_applied,
            higham_shift=p1.higham_shift,
            cholesky_factor_norm=float(la.norm(p1.cholesky_factor_L_G, ord="fro")),
            symplectic_drift=p1.symplectic_drift,
            is_symplectic_conserved=p1.is_symplectic_conserved,
            polyakov_action=p1.polyakov_action,
            verdict=p1.verdict,
            kam_stability_index=p1.kam_stability_index,
            novikov_regularized=p1.novikov_regularized,
            hill_discriminant=p1.hill_discriminant,
        )
        max_eig = float(np.max(phase2_state.eigenvalues)) if phase2_state.eigenvalues.size else 0.0
        cert2 = Phase2SpectralCertificate(
            condition_number=float(
                min(max_eig / max(phase2_state.fiedler_value, _WILKINSON_TOL), _KAPPA_MAX)
            ),
            eigenvalues=phase2_state.eigenvalues,
            fiedler_value=phase2_state.fiedler_value,
            spectral_gap=phase2_state.spectral_gap,
            kernel_dimension=phase2_state.kernel_dimension,
            laplacian_rank=phase2_state.laplacian_rank,
            fredholm_residual=phase2_state.fredholm_residual,
            is_fredholm_soluble=phase2_state.is_fredholm_soluble,
            heat_trace=phase2_state.heat_trace,
            ray_singer_determinant=phase2_state.ray_singer_determinant,
            sparse_computation=phase2_state.sparse_computation,
            verdict=phase2_state.verdict,
            log_ray_singer_determinant=phase2_state.log_ray_singer_determinant,
            connes_lipschitz_bound=phase2_state.connes_lipschitz_bound,
        )
        cert3 = Phase3HodgeCertificate(
            projected_state=x_harm,
            projection_residual=residual_proj,
            is_minimum_norm=is_min_norm,
            poincare_constant=poincare_const,
            connes_lipschitz_bound=connes_bound,
            is_lipschitz_stable=is_lipschitz_stable,
            ricci_scalar_curvature=ricci_scalar,
            verdict=verdict_p3,
            brockett_casimir_error=casimir_err,
            lyapunov_energy=lyapunov,
            moduli_distance_delta=moduli_delta,
            daleckii_frechet_norm=frechet_norm,
        )

        final_verdict = StrictHeytingVerdict.supremum(cert1.verdict, cert2.verdict, cert3.verdict)

        solver_action = SolverAction.NONE
        latched_crowbar = False
        actuation_ns = 0.0
        sla_met = True
        diagnostic = "Gobernanza celular completada con coherencia causal estricta."

        if final_verdict.is_vetoed:
            solver_action = SolverAction.TRIGGER_HARDWARE_CROWBAR
            actuation_ns, sla_met = self.trigger_hardware_crowbar_interrupt()
            latched_crowbar = True
            diagnostic = (
                f"[CROWBAR_LATCH] VETO EN HAZ CELULAR: FredholmRes={phase2_state.fredholm_residual:.3e}, "
                f"SymplecticDrift={p1.symplectic_drift:.3e}, Lipschitz={connes_bound:.3e}. "
                f"Hardware latch en {actuation_ns:.2f} ns (SLA {_ESP32_CROWBAR_MAX_LATENCY_NS:.2f} ns, "
                f"cumplido={sla_met})."
            )
            if self._halt_on_veto:
                raise HardwareCrowbarTriggered(diagnostic)
        elif final_verdict.is_degraded:
            solver_action = SolverAction.REGULARIZE_METRIC
            diagnostic = (
                "Degradación espectral/simpléctica detectada: regularización Higham / RSI³ aplicada."
            )

        telemetry_bytes = (
            f"{phase2_state.fredholm_residual:.6e}|{phase2_state.kernel_dimension}|"
            f"{phase2_state.fiedler_value:.6e}|{p1.symplectic_drift:.6e}|"
            f"{connes_bound:.6e}|{p1.kam_stability_index:.6e}|{final_verdict.value}"
        ).encode("utf-8")
        digest = hashlib.sha256(telemetry_bytes).hexdigest()

        poincare_cert = SheafRSI3PoincareCertificate(
            fredholm_residual=phase2_state.fredholm_residual,
            betti_1_cycles=phase2_state.kernel_dimension,
            fiedler_connectivity=phase2_state.fiedler_value,
            symplectic_volume_drift=p1.symplectic_drift,
            connes_lipschitz_bound=connes_bound,
            ricci_scalar_curvature=ricci_scalar,
            heyting_verdict=final_verdict,
            sha256_telemetry_digest=digest,
            is_causally_coherent=final_verdict.is_coherent,
            kam_stability_index=p1.kam_stability_index,
            hill_discriminant=p1.hill_discriminant,
            lyapunov_energy=lyapunov,
        )
        provenance_payload = (
            f"{timestamp_utc}|{cert1.verdict.value}|{cert2.verdict.value}|{cert3.verdict.value}|"
            f"{p1.metric_condition_number:.8e}|{phase2_state.fiedler_value:.8e}|{digest}"
        ).encode("utf-8")
        provenance_hash = hashlib.sha256(provenance_payload).hexdigest()

        return SheafSolverState(
            phase1_certificate=cert1,
            phase2_certificate=cert2,
            phase3_certificate=cert3,
            poincare_certificate=poincare_cert,
            final_verdict=final_verdict,
            solver_action=solver_action,
            hardware_crowbar_latched=latched_crowbar,
            actuation_latency_ns=actuation_ns,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=diagnostic,
            exact_solution_x=exact_x,
            crowbar_sla_met=sla_met,
            rsi3_iterations=rsi3_iters,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO DEL HAZ CELULAR (SheafLaplacianSolver)
# ═══════════════════════════════════════════════════════════════════════════════

class SheafLaplacianSolver(Phase3_SovereignAgenticGovernance):
    r"""
    Soberano Agéntico del Haz Celular (estrato Alfa).

    Orquestador OODA de lazo cerrado sobre \(T^*\mathcal{M}_{\mathrm{sheaf}}\).
    Concatena de forma indivisible las Fases 1, 2 y 3:

    \[
    \Phi=\Phi_3\circ\Phi_2\circ\Phi_1.
    \]
    """

    def execute_sheaf_solver_cycle(
        self,
        delta: MatrixLike,
        G: NDArray[np.float64],
        x_initial: NDArray[np.float64],
        rho: Optional[NDArray[np.float64]] = None,
        jacobian_M: Optional[NDArray[np.float64]] = None,
        potential_V: float = 0.0,
        total_energy_H0: float = 1.0,
        trajectory_x: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX,
    ) -> SheafSolverState:
        r"""
        EJECUCIÓN DEL LAZO CERRADO ANIDADO DE 3 FASES.

        Encadenamiento estricto:
          FASE 1: `execute_phase1_symplectic_maupertuis_reduction` → `Phase1SymplecticMaupertuisState`
          FASE 2: `execute_phase2_banach_fredholm_synthesis`       → `Phase2BanachFredholmState`
          FASE 3: `execute_phase3_sovereign_governance_cycle`      → `SheafSolverState`
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            m = int(delta.shape[0])
            n = int(delta.shape[1])
            g_metric = np.asarray(G, dtype=np.float64)
            if g_metric.shape != (m, m):
                raise CohomologicalDimensionMismatchError(f"Matriz métrica debe ser {m}×{m}.")

            if jacobian_M is None:
                phase_dim = 2 * m if (2 * m) % 2 == 0 else 2 * ((m + 1) // 2) * 2
                if phase_dim % 2 != 0:
                    phase_dim += 1
                jacobian_M = np.eye(phase_dim, dtype=np.float64)

            if rho is None:
                rho = np.zeros(n, dtype=np.float64)
            rho_vec = np.asarray(rho, dtype=np.float64).ravel()
            if rho_vec.shape[0] != n:
                raise CohomologicalDimensionMismatchError(
                    f"rho tiene dim {rho_vec.shape[0]}, esperado {n} (= columnas de δ)."
                )
            x0 = np.asarray(x_initial, dtype=np.float64).ravel()
            if x0.shape[0] != n:
                raise CohomologicalDimensionMismatchError(
                    f"x_initial tiene dim {x0.shape[0]}, esperado {n}."
                )

            phase1_state = self.execute_phase1_symplectic_maupertuis_reduction(
                coboundary_delta=delta,
                base_metric_G=g_metric,
                potential_V=potential_V,
                total_energy_H0=total_energy_H0,
                jacobian_M=np.asarray(jacobian_M, dtype=np.float64),
                trajectory_x=trajectory_x,
                target_condition=target_condition,
            )
            phase2_state = self.execute_phase2_banach_fredholm_synthesis(
                phase1_state=phase1_state,
                source_rho=rho_vec,
                x_initial=x0,
                target_condition=target_condition,
            )
            return self.execute_phase3_sovereign_governance_cycle(phase2_state=phase2_state)
        except SheafLaplacianSolverError as exc:
            if self._halt_on_veto:
                raise
            n_fallback = int(delta.shape[1]) if getattr(delta, "ndim", 0) == 2 else 1
            return self._cataclysm_state(str(exc), timestamp_utc, n_fallback)

    def execute_nested_sovereign_solver_pipeline(
        self,
        coboundary_delta: MatrixLike,
        metric_G: NDArray[np.float64],
        source_density_rho: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        x_initial: Optional[NDArray[np.float64]] = None,
        potential_V: float = 0.0,
        total_energy_H0: float = 1.0,
        trajectory_x: Optional[NDArray[np.float64]] = None,
        ricci_tensor: Optional[NDArray[np.float64]] = None,
        potential_p: Optional[NDArray[np.float64]] = None,
    ) -> SheafSolverState:
        r"""
        Morfismo único \(\Phi_3\circ\Phi_2\circ\Phi_1\) con tensores RSI³ opcionales.
        """
        n = int(coboundary_delta.shape[1])
        x0 = (
            np.zeros(n, dtype=np.float64)
            if x_initial is None
            else np.asarray(x_initial, dtype=np.float64).ravel()
        )
        phase1 = self.execute_phase1_symplectic_maupertuis_reduction(
            coboundary_delta=coboundary_delta,
            base_metric_G=metric_G,
            potential_V=potential_V,
            total_energy_H0=total_energy_H0,
            jacobian_M=jacobian_M,
            trajectory_x=trajectory_x,
        )
        phase2 = self.execute_phase2_banach_fredholm_synthesis(
            phase1_state=phase1,
            source_rho=np.asarray(source_density_rho, dtype=np.float64).ravel(),
            x_initial=x0,
        )
        return self.execute_phase3_sovereign_governance_cycle(
            phase2_state=phase2,
            potential_p=potential_p,
            ricci_tensor=ricci_tensor,
        )

    def audit_and_solve_sheaf(
        self,
        coboundary_delta: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        source_density_rho: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], SheafRSI3PoincareCertificate]:
        r"""
        Resuelve \(L_F x=\rho\) y audita invariantes de Poincaré / Fredholm.

        La solución se extrae del ciclo anidado (pseudoinversa espectral de Hodge),
        no de un segundo ensamblaje que ignoraba Higham y el estado certificado.
        """
        n = int(np.asarray(coboundary_delta).shape[1])
        x_init = np.zeros(n, dtype=np.float64)
        state = self.execute_nested_sovereign_solver_pipeline(
            coboundary_delta=coboundary_delta,
            metric_G=metric_G,
            source_density_rho=source_density_rho,
            jacobian_M=jacobian_M,
            x_initial=x_init,
            ricci_tensor=ricci_tensor,
        )
        solution = (
            state.exact_solution_x
            if state.exact_solution_x is not None
            else np.zeros(n, dtype=np.float64)
        )
        return solution, state.poincare_certificate

    def audit_poincare_liouville_invariance(self, jacobian_M: NDArray[np.float64]) -> float:
        r"""Deriva de Liouville \(|\det M-1|\) vía slogdet (API histórica, retorno escalar)."""
        _, _, det_drift, *_ = self.audit_symplectic_poincare_liouville(
            np.asarray(jacobian_M, dtype=np.float64)
        )
        return float(det_drift)

    def audit_maupertuis_geodesic_action(
        self,
        trajectory_x: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        potential_V: float,
        H0: float,
    ) -> float:
        r"""Acción de Polyakov \(S_E[x]\) sobre la geodésica de Maupertuis-Jacobi."""
        g_conf, _, _ = self.compute_maupertuis_conformal_metric(metric_G, potential_V, H0)
        polyakov, _ = self.compute_polyakov_geodesic_action(trajectory_x, g_conf)
        return polyakov

    def execute_rsi1_ricci_flow(
        self,
        G: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        delta_tau: float = 0.01,
    ) -> NDArray[np.float64]:
        r"""Nivel 1 (\(R_1\)) de RSI³: flujo de Ricci-DeTurck discreto normalizado."""
        return self.execute_rsi1_ricci_deturck_flow(G, ricci_tensor, delta_tau)

    def _cataclysm_state(self, reason: str, timestamp_utc: str, n: int) -> SheafSolverState:
        r"""Estado terminal de colapso catastrófico bajo veto incondicional."""
        dim = max(int(n), 1)
        actuation_ns, sla_met = self.trigger_hardware_crowbar_interrupt()
        cert1 = Phase1MetricCertificate(
            metric_condition_number=float("inf"),
            is_metric_spd=False,
            higham_regularization_applied=False,
            higham_shift=0.0,
            cholesky_factor_norm=0.0,
            symplectic_drift=float("inf"),
            is_symplectic_conserved=False,
            polyakov_action=float("inf"),
            verdict=StrictHeytingVerdict.VETOED,
        )
        cert2 = Phase2SpectralCertificate(
            condition_number=float("inf"),
            eigenvalues=np.array([], dtype=np.float64),
            fiedler_value=0.0,
            spectral_gap=0.0,
            kernel_dimension=dim,
            laplacian_rank=0,
            fredholm_residual=float("inf"),
            is_fredholm_soluble=False,
            heat_trace=0.0,
            ray_singer_determinant=0.0,
            sparse_computation=False,
            verdict=StrictHeytingVerdict.VETOED,
        )
        cert3 = Phase3HodgeCertificate(
            projected_state=np.zeros(dim, dtype=np.float64),
            projection_residual=float("inf"),
            is_minimum_norm=False,
            poincare_constant=float("inf"),
            connes_lipschitz_bound=float("inf"),
            is_lipschitz_stable=False,
            ricci_scalar_curvature=0.0,
            verdict=StrictHeytingVerdict.VETOED,
        )
        poincare_cert = SheafRSI3PoincareCertificate(
            fredholm_residual=float("inf"),
            betti_1_cycles=dim,
            fiedler_connectivity=0.0,
            symplectic_volume_drift=float("inf"),
            connes_lipschitz_bound=float("inf"),
            ricci_scalar_curvature=0.0,
            heyting_verdict=StrictHeytingVerdict.VETOED,
            sha256_telemetry_digest=hashlib.sha256(reason.encode("utf-8")).hexdigest(),
            is_causally_coherent=False,
        )
        return SheafSolverState(
            phase1_certificate=cert1,
            phase2_certificate=cert2,
            phase3_certificate=cert3,
            poincare_certificate=poincare_cert,
            final_verdict=StrictHeytingVerdict.VETOED,
            solver_action=SolverAction.TRIGGER_HARDWARE_CROWBAR,
            hardware_crowbar_latched=True,
            actuation_latency_ns=actuation_ns,
            timestamp_utc=timestamp_utc,
            provenance_hash=hashlib.sha256(f"CATACLYSM::{reason}".encode("utf-8")).hexdigest(),
            diagnostic_note=f"CATACLYSM_SHEAF_VETO: {reason}",
            exact_solution_x=np.zeros(dim, dtype=np.float64),
            crowbar_sla_met=sla_met,
            rsi3_iterations=0,
        )


SheafLaplacianSolverAgent = SheafLaplacianSolver

__all__ = [
    "SheafLaplacianSolverError",
    "MetricNonSPDError",
    "CohomologicalDimensionMismatchError",
    "FredholmSolvabilityError",
    "SymplecticInvariantViolationError",
    "LipschitzStabilityViolationError",
    "HardwareCrowbarTriggered",
    "StrictHeytingVerdict",
    "SheafSolverVerdict",
    "SolverAction",
    "Phase1SymplecticMaupertuisState",
    "Phase2BanachFredholmState",
    "Phase1MetricCertificate",
    "Phase2SpectralCertificate",
    "Phase3HodgeCertificate",
    "SheafRSI3PoincareCertificate",
    "SheafSolverState",
    "Phase1_SymplecticMaupertuisPreconditioner",
    "Phase2_SheafLaplacianFredholmEngine",
    "Phase3_SovereignAgenticGovernance",
    "SheafLaplacianSolver",
    "SheafLaplacianSolverAgent",
    "stable_metric_higham",
    "build_sparse_laplacian",
]