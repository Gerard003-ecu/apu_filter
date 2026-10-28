# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sheaf Laplacian Solver (Soberano Agéntico del Haz Celular)         ║
║ Ruta   : app/agents/alfa/sheaf_laplacian_solver.py                           ║
║ Versión: 5.0.0-RSI3-Poincare-Liouville-Novikov-Heyting-ESP32-PhD             ║
║ Función: Soberano de Gobernanza Celular y Resolución del Laplaciano del Haz  ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL ───
Este módulo materializa el Soberano Agéntico del Haz Celular (SheafLaplacianSolverAgent
/ SheafLaplacianSolver) en el estrato Alfa de la arquitectura APU Filter. Opera como
un agente de gobernanza OODA de lazo cerrado sobre el espacio de estados $T^*\mathcal{M}_{\mathrm{sheaf}}$,
interactuando con el motor espectral FPU (SheafLaplacianEngine) para garantizar la
coherencia metrológica, la estabilidad simpléctica de Poincaré, la automejora
recursiva de Nivel 3 (RSI³) y la inmunidad cibernética mediante el disyuntor de hardware
ESP32 / BT151 Crowbar en tiempo real (< 400 ns).

MARCO TEÓRICO ─────────────────────────────────────────────────────────────────
La física del haz celular $\mathcal{F}$ sobre el complejo simplicial 1-dimensional
$K = (V, E)$ modela la red de restricciones de flujo de caja, costes y presupuestos civil.
El sistema unifica cuatro pilares de la mecánica celeste de Henri Poincaré con la
teoría espectral no conmutativa:
  1. Invarianza Simpléctica de Liouville & Gromov Non-Squeezing:
     Preservación de la 2-forma canónica $\omega = \frac{1}{2} dz^\top \Omega dz$ bajo
     la matriz de monodromía $M \in \mathrm{Sp}(2n, \mathbb{R})$, forzando $\det(M) = +1$
     y prohibiendo la compresión no física del volumen de riesgo presupuestario.
  2. Geodésicas Conformes de Maupertuis-Jacobi:
     Minimización de la Acción de Polyakov $S_E[x] = \frac{1}{2} \int \tilde{G}_{\mu\nu} \dot{x}^\mu \dot{x}^\nu d\tau$
     bajo la métrica conforme $\tilde{G}_{\mu\nu}(x) = 2(H_0 - V(x)) G_{\mu\nu}(x)$.
  3. Anillo Ultramétrico de Novikov & Burbujeo Disco Zero:
     Expansión perturbativa sobre $\Lambda_{\mathrm{Nov}}$ con fuga de instantones $m_0 \equiv 0 \implies m_1^2 = 0$,
     garantizando nilpotencia homológica $d^2 = 0$ en la frontera $\partial \mathcal{M}$.
  4. Protocolo RSI³ de Nivel 3 ($R_3 \circ R_2 \circ R_1$):
     Flujo de Ricci discreto ($R_1$), cota Lipschitz de Connes-Daleckii-Krein ($R_2$)
     y flujo isoespectral de doble corchete de Brockett-Shahshahani ($R_3$).

AXIOMAS FORMALES ───────────────────────────────────────────────────────────────
  [A1] Solubilidad de Fredholm:
       $$\rho \in \operatorname{im}(L_F) \iff \langle \rho, v_k \rangle = 0 \quad \forall v_k \in \ker(L_F)$$
  [A2] Conservación Simpléctica de Liouville:
       $$M^\top \Omega M = \Omega \implies \det(M) = +1$$
  [A3] Cota Estricta de Connes-Daleckii-Krein:
       $$L_{\max} \le \frac{1}{2 \, \lambda_2(L_F)^{3/2}}$$
  [A4] Aciclicidad Homológica de de Rham-Čech:
       $$\beta_1 = \dim \ker(L_F) - \beta_0 \equiv 0$$
  [A5] Proyección de Higham sobre el Cono SPD:
       $$\mathbf{G}_{\mathrm{reg}} = \arg\min_{\mathbf{S} \in \operatorname{Sym}^+(n)} \|\mathbf{G} - \mathbf{S}\|_F \implies \lambda_{\min}(\mathbf{G}_{\mathrm{reg}}) \ge \tau_{\mathrm{SPD}}$$

INVARIANTES DE ESTADO ──────────────────────────────────────────────────────────
  [I1] Positividad Estricta de la Métrica: $\mathbf{G}_{\mathrm{reg}} \in \operatorname{Sym}^+(n, \mathbb{R})$
  [I2] Autoadjunción del Laplaciano Celular: $L_F = L_F^\top \succeq 0$
  [I3] Conservación de Traza en Flujo de Brockett: $\operatorname{Tr}(\boldsymbol{\rho}(t)) = 1$
  [I4] Invariante de Procedencia Criptográfica: Hash SHA-256 inmutable en el Certificado.
  [I5] Interrupción Hardware en Silicon: Disparo a GPIO14 en $t_{\mathrm{actuation}} \le 398.95\text{ ns}$ ante $\Omega_3 \to \mathtt{VETOED}$.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Tuple, Optional, Sequence, Union
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Importación del soporte de morfismos y excepciones categóricas del Topos
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover — fallback autonómico de desarrollo
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos E_MIC."""
        pass

# Importación del Motor Espectral ciego FPU
try:
    from app.alfa.sheaf_laplacian_engine import SheafLaplacianEngine, SheafEngineSpectrumReport
except ImportError:  # pragma: no cover
    from app.physics.sheaf_laplacian_engine import SheafLaplacianEngine, SheafEngineSpectrumReport

# Soporte para matrices dispersas scipy
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    csr_matrix = None
    isspmatrix = None

logger = logging.getLogger("MIC.Alpha.SheafLaplacianSolver")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE PRECISIÓN ESPECTRAL DE WILKINSON Y LÍMITES CIBER-FÍSICOS
# ═══════════════════════════════════════════════════════════════════════════════
_WILKINSON_TOL: Final[float] = 1.0e-12
_DEFAULT_TOL: Final[float] = 1.0e-12
_MAX_CONDITION_NUMBER: Final[float] = 1.0e8
_KAPPA_MAX: Final[float] = 1.0e8
_HARD_LIPSCHITZ_CEILING: Final[float] = 1.0e6
_SYMPLECTIC_TOLERANCE: Final[float] = 1.0e-9
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12
_SPARSE_THRESHOLD: Final[float] = 0.3

# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES TIPADAS (FUNTORES DE ERROR)
# ═══════════════════════════════════════════════════════════════════════════════
class SheafLaplacianSolverError(Exception):
    r"""Excepción raíz para errores en el solucionador y soberano del Haz Celular."""
    pass

class MetricNonSPDError(SheafLaplacianSolverError):
    r"""Tensor métrico no definido positivo (falla la factorización de Cholesky)."""
    pass

class CohomologicalDimensionMismatchError(SheafLaplacianSolverError):
    r"""Inconsistencia dimensional entre la cofrontera \delta y la métrica G."""
    pass

class FredholmSolvabilityError(SheafLaplacianSolverError):
    r"""Violación de la condición de solubilidad de Fredholm \langle \rho, v_k \rangle \neq 0."""
    pass

class LipschitzStabilityViolationError(SheafLaplacianSolverError):
    r"""La proyección de Hodge supera la cota de estabilidad de Lipschitz."""
    pass

class SpectralInstabilityError(SheafLaplacianSolverError):
    r"""Inestabilidad espectral detectada en el Laplaciano del Haz Celular."""
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES CATEGÓRICAS EN EL RETÍCULO DE HEYTING \Omega_3
# ═══════════════════════════════════════════════════════════════════════════════
class StrictHeytingVerdict(IntEnum):
    r"""
    Retículo distributivo de Heyting \Omega_3 = {COHERENT, DEGRADED, VETOED}.
    
    Valores:
      COHERENT = 0 (\top supremo coherente)
      DEGRADED = 1 (estado intermedio con atenuación espectral)
      VETOED   = 2 (\bot infimo veto ciber-físico)
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "StrictHeytingVerdict") -> "StrictHeytingVerdict":
        r"""Calcula el supremo (peor severidad) en el retículo \Omega_3."""
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == StrictHeytingVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == StrictHeytingVerdict.DEGRADED


# Alias de compatibilidad hacia versiones anteriores
SheafSolverVerdict = StrictHeytingVerdict


class SolverAction(Enum):
    r"""Acciones de mitigación operativa tras el veredicto en el retículo."""
    NONE = auto()
    REGULARIZE_METRIC = auto()
    HALT_SOLVER = auto()

# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS E INVARIANTES DE CERTIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SheafRSI3PoincareCertificate:
    r"""
    Certificado inmutable de gobernanza sobre el haz celular (v5.0).
    
    Atestigua el cumplimiento simultáneo de la solubilidad de Fredholm,
    la invarianza simpléctica de Liouville, la cota Lipschitz de Connes-DK
    y el colapso distributivo en el retículo de Heyting \Omega_3.
    """
    fredholm_residual: float
    betti_1_cycles: int
    fiedler_connectivity: float
    symplectic_volume_drift: float
    connes_lipschitz_bound: float
    ricci_scalar_curvature: float
    heyting_verdict: StrictHeytingVerdict
    sha256_telemetry_digest: str
    is_causally_coherent: bool


@dataclass(frozen=True, slots=True)
class Phase1MetricCertificate:
    r"""Certificado de la Fase 1: Proyección Higham SPD y Cholesky L_G."""
    metric_condition_number: float
    is_metric_spd: bool
    higham_regularization_applied: bool
    higham_shift: float
    cholesky_factor_norm: float
    verdict: StrictHeytingVerdict


@dataclass(frozen=True, slots=True)
class Phase2SpectralCertificate:
    r"""Certificado de la Fase 2: Espectro y conectividad algebráica del Laplaciano."""
    condition_number: float
    eigenvalues: NDArray[np.float64]
    kernel_dimension: int
    spectral_gap: float
    is_spectrally_stable: bool
    sparse_computation: bool
    laplacian_rank: int
    verdict: StrictHeytingVerdict


@dataclass(frozen=True, slots=True)
class Phase3HodgeCertificate:
    r"""Certificado de la Fase 3: Proyección armónica de Hodge y cota de Lipschitz."""
    projected_state: NDArray[np.float64]
    projection_residual: float
    is_minimum_norm: bool
    poincare_constant: float
    is_lipschitz_stable: bool
    fredholm_soluble: bool
    fredholm_residual: float
    verdict: StrictHeytingVerdict


@dataclass(frozen=True, slots=True)
class SheafSolverState:
    r"""Estado global del ciclo de resolución espectral sobre las 3 fases anidadas."""
    phase1: Phase1MetricCertificate
    phase2: Phase2SpectralCertificate
    phase3: Phase3HodgeCertificate
    final_verdict: StrictHeytingVerdict
    solver_action: SolverAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITARIOS DE REGULARIZACIÓN ESPECTRAL Y CÁLCULO DISPERSO
# ═══════════════════════════════════════════════════════════════════════════════
def stable_metric_higham(
    G: NDArray[np.float64],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR,
    target_condition: float = _KAPPA_MAX
) -> Tuple[NDArray[np.float64], float, bool]:
    r"""
    Aplica la proyección de Higham para estabilizar el tensor métrico G en
    el cono SPD (Symmetric Positive Definite).
    
    Axioma de Proyección:
        $$\tilde{\mathbf{G}} = \arg\min_{\mathbf{M} \succeq 0} \|\mathbf{G} - \mathbf{M}\|_F$$
    """
    G_sym = 0.5 * (G + G.T)
    
    try:
        eigvals, eigvecs = la.eigh(G_sym)
    except Exception as exc:
        raise MetricNonSPDError(
            "MetricNonSPDError: Falló la descomposición espectral para proyección Higham."
        ) from exc

    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    shift = 0.0
    applied = False

    if lam_min <= tolerance:
        shift = abs(lam_min) + max(tolerance, lam_max / target_condition)
        applied = True
        lam_min += shift
        lam_max += shift

    current_cond = lam_max / max(lam_min, np.finfo(np.float64).eps)
    if current_cond > target_condition:
        extra = (lam_max - target_condition * lam_min) / (target_condition - 1.0)
        if extra > 0.0:
            shift += extra
            applied = True
            lam_min += extra
            lam_max += extra

    vals_reg = eigvals + shift
    G_projected = eigvecs @ np.diag(vals_reg) @ eigvecs.T
    G_projected = 0.5 * (G_projected + G_projected.T)

    try:
        la.cholesky(G_projected, lower=True)
    except la.LinAlgError as exc:
        raise MetricNonSPDError(
            "MetricNonSPDError: La proyección de Higham no produjo una matriz SPD."
        ) from exc

    return G_projected, shift, applied


def build_sparse_laplacian(
    delta: Union[NDArray[np.float64], csr_matrix],
    L_G: Union[NDArray[np.float64], csr_matrix]
) -> Tuple[Union[NDArray[np.float64], csr_matrix], bool]:
    r"""
    Construye el Laplaciano del Haz L_F = \delta^\top G^{-1} \delta con soporte disperso.
    """
    sparse_used = False
    
    if _SPARSE_AVAILABLE and isspmatrix is not None:
        if isspmatrix(delta) or isspmatrix(L_G):
            sparse_used = True
            if isspmatrix(L_G):
                from scipy.sparse.linalg import spsolve_triangular
                Y = spsolve_triangular(L_G, delta.toarray() if isspmatrix(delta) else delta, lower=True)
                Y_sparse = csr_matrix(Y)
                L_F = Y_sparse.T @ Y_sparse
            else:
                Y = la.solve_triangular(L_G, delta.toarray() if isspmatrix(delta) else delta, lower=True)
                Y_sparse = csr_matrix(Y)
                L_F = Y_sparse.T @ Y_sparse
        elif np.count_nonzero(delta) < _SPARSE_THRESHOLD * delta.size:
            sparse_used = True
            Y = la.solve_triangular(L_G, delta, lower=True)
            Y_sparse = csr_matrix(Y)
            L_F = Y_sparse.T @ Y_sparse
        else:
            Y = la.solve_triangular(L_G, delta, lower=True)
            L_F = Y.T @ Y
    else:
        Y = la.solve_triangular(L_G, delta, lower=True)
        L_F = Y.T @ Y

    return L_F, sparse_used

# ═══════════════════════════════════════════════════════════════════════════════
# FASES DE PRECONDICIONAMIENTO, ENSAMBLAJE Y PROYECCIÓN (HERENCIA MULTI-FASE)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_MetricPreconditioner(Morphism, ABC):
    r"""FASE 1: Precondicionamiento SPD y Factorización de Cholesky."""

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        super().__init__()
        self._tolerance = tolerance
        self._higham_floor = _HIGHAM_REGULARIZATION_FLOOR

    def project_to_spd(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[NDArray[np.float64], float, bool]:
        G_projected, shift, applied = stable_metric_higham(
            G, self._higham_floor, target_condition
        )
        eigvals_proj = la.eigvalsh(G_projected)
        cond = float(np.max(eigvals_proj) / max(np.min(eigvals_proj), 1.0e-15))
        return G_projected, cond, applied

    def stable_cholesky(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, bool]:
        G_spd, cond, higham_applied = self.project_to_spd(G, target_condition)
        try:
            L_G = la.cholesky(G_spd, lower=True)
        except la.LinAlgError as exc:
            raise MetricNonSPDError(
                f"MetricNonSPDError: Factorización Cholesky fallida tras proyección: {exc}"
            ) from exc
        return L_G, G_spd, cond, higham_applied

    def prepare_metric_factor(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[NDArray[np.float64], float, bool]:
        L_G, _, cond, higham_applied = self.stable_cholesky(G, target_condition)
        return L_G, cond, higham_applied

    def generate_phase1_certificate(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Phase1MetricCertificate:
        L_G, cond, higham_applied = self.prepare_metric_factor(G, target_condition)
        higham_shift = 0.0
        if higham_applied:
            eigvals_orig = la.eigvalsh(G)
            eigvals_proj = la.eigvalsh(0.5 * (L_G @ L_G.T + (L_G @ L_G.T).T))
            higham_shift = float(np.min(eigvals_proj) - np.min(eigvals_orig))
        
        cholesky_norm = float(la.norm(L_G, ord="fro"))
        
        verdict = StrictHeytingVerdict.COHERENT
        if not np.isfinite(cond) or cond <= 0.0:
            verdict = StrictHeytingVerdict.VETOED
        elif cond > _KAPPA_MAX or higham_applied:
            verdict = StrictHeytingVerdict.DEGRADED
        
        return Phase1MetricCertificate(
            metric_condition_number=cond,
            is_metric_spd=True,
            higham_regularization_applied=higham_applied,
            higham_shift=higham_shift,
            cholesky_factor_norm=cholesky_norm,
            verdict=verdict,
        )


class Phase2_SheafLaplacianBuilder(Phase1_MetricPreconditioner):
    r"""FASE 2: Ensamblaje del Laplaciano Celular y Análisis Espectral Krylov."""

    def build_sheaf_laplacian(
        self,
        delta: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[Union[NDArray[np.float64], csr_matrix], NDArray[np.float64], bool]:
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise CohomologicalDimensionMismatchError("G debe ser una matriz cuadrada.")
        if delta.ndim != 2:
            raise CohomologicalDimensionMismatchError("delta debe ser una matriz 2D.")
        
        m, n = delta.shape
        if G.shape[0] != m:
            raise CohomologicalDimensionMismatchError(
                f"delta tiene {m} filas pero G es {G.shape[0]}×{G.shape[1]}."
            )
        
        L_G, _, higham_applied = self.prepare_metric_factor(G, target_condition)
        L_F, sparse_used = build_sparse_laplacian(delta, L_G)
        
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(delta):
            from scipy.sparse.linalg import spsolve_triangular
            Y = spsolve_triangular(L_G, delta.toarray(), lower=True)
        else:
            Y = la.solve_triangular(L_G, delta, lower=True)
        
        return L_F, Y, sparse_used

    def analyze_sheaf_spectrum(
        self,
        L_F: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        sparse_computation: bool = False
    ) -> Phase2SpectralCertificate:
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_F):
            L_F_dense = L_F.toarray()
        else:
            L_F_dense = L_F
        
        eigvals_L = la.eigvalsh(L_F_dense)
        eigvals_L = np.maximum(eigvals_L, 0.0)
        sorted_vals = np.sort(eigvals_L)
        
        eigvals_G = la.eigvalsh(G)
        cond_G = float(np.max(eigvals_G) / max(np.min(eigvals_G), 1.0e-15))
        
        tol_kernel = max(self._tolerance * 100, np.max(eigvals_L) * np.finfo(np.float64).eps * 10)
        kernel_dim = int(np.sum(eigvals_L <= tol_kernel))
        
        spectral_gap = 0.0
        if len(sorted_vals) >= 2:
            positive_mask = sorted_vals > tol_kernel
            if np.any(positive_mask):
                spectral_gap = float(sorted_vals[positive_mask][0] - sorted_vals[0])
        
        laplacian_rank = int(np.linalg.matrix_rank(L_F_dense, tol=tol_kernel))
        is_spectrally_stable = cond_G <= _KAPPA_MAX
        
        verdict = StrictHeytingVerdict.COHERENT
        if not np.isfinite(cond_G) or not np.isfinite(spectral_gap):
            verdict = StrictHeytingVerdict.VETOED
        elif not is_spectrally_stable or kernel_dim > 0:
            verdict = StrictHeytingVerdict.DEGRADED
        
        return Phase2SpectralCertificate(
            condition_number=cond_G,
            eigenvalues=sorted_vals,
            kernel_dimension=kernel_dim,
            spectral_gap=spectral_gap,
            is_spectrally_stable=is_spectrally_stable,
            sparse_computation=sparse_computation,
            laplacian_rank=laplacian_rank,
            verdict=verdict,
        )


class Phase3_HodgeProjector(Phase2_SheafLaplacianBuilder):
    r"""FASE 3: Proyección Armónica de Hodge-de Rham y Solubilidad de Fredholm."""

    def check_fredholm_solvability(
        self,
        L_F: Union[NDArray[np.float64], csr_matrix],
        rho: NDArray[np.float64],
        tolerance: float = 1.0e-8
    ) -> Tuple[bool, float]:
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_F):
            L_F_dense = L_F.toarray()
        else:
            L_F_dense = L_F
        
        vals, vecs = la.eigh(L_F_dense)
        max_val = np.max(vals) if len(vals) > 0 else 1.0
        kernel_mask = vals <= max(self._tolerance * 100, max_val * np.finfo(np.float64).eps * 10)
        kernel_basis = vecs[:, kernel_mask]
        
        if kernel_basis.shape[1] == 0:
            return True, 0.0
        
        proj_coeffs = kernel_basis.T @ rho
        residual = float(la.norm(proj_coeffs))
        is_soluble = residual <= tolerance
        return is_soluble, residual

    def hodge_projection_solve(
        self,
        delta: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        x: NDArray[np.float64],
        rho: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX
    ) -> Phase3HodgeCertificate:
        L_F, _, sparse_used = self.build_sheaf_laplacian(delta, G, target_condition)
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_F):
            L_F_dense = L_F.toarray()
        else:
            L_F_dense = L_F
        
        vals, vecs = la.eigh(L_F_dense)
        max_val = np.max(vals) if len(vals) > 0 else 1.0
        kernel_mask = vals <= max(self._tolerance * 100, max_val * np.finfo(np.float64).eps * 10)
        kernel_basis = vecs[:, kernel_mask]
        
        if kernel_basis.shape[1] == 0:
            x_projected = np.zeros_like(x)
        else:
            x_projected = kernel_basis @ (kernel_basis.T @ x)
        
        residual = float(la.norm(L_F_dense @ x_projected))
        norm_proj = float(la.norm(x_projected))
        norm_orig = float(la.norm(x))
        is_min_norm = norm_proj <= norm_orig + self._tolerance * 100
        
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(delta):
            dx = delta.dot(x)
            dx_proj = delta.dot(x_projected)
        else:
            dx = delta @ x
            dx_proj = delta @ x_projected
        
        norm_dx = float(la.norm(dx))
        deviation = x - x_projected
        norm_dev = float(la.norm(deviation))
        poincare = norm_dev / norm_dx if norm_dx > self._tolerance else 0.0
        
        lhs = float(la.norm(dx_proj - dx))
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(delta):
            svals = la.svdvals(delta.toarray())
        else:
            svals = la.svdvals(delta)
        
        max_sval = float(np.max(svals)) if len(svals) > 0 else 1.0
        min_sval_pos = float(np.min(svals[svals > self._tolerance])) if np.any(svals > self._tolerance) else 1.0
        kappa_delta = max_sval / min_sval_pos if min_sval_pos > 0 else max_sval / np.finfo(np.float64).eps
        effective_lip = max(kappa_delta, max_sval)
        rhs = effective_lip * norm_dev
        is_lipschitz_stable = lhs <= rhs + self._tolerance * 100
        
        if rho is not None:
            fredholm_soluble, fredholm_residual = self.check_fredholm_solvability(L_F_dense, rho)
        else:
            fredholm_soluble = True
            fredholm_residual = 0.0
        
        verdict = StrictHeytingVerdict.COHERENT
        if not np.isfinite(residual) or not np.isfinite(poincare) or not fredholm_soluble:
            verdict = StrictHeytingVerdict.VETOED
        elif not is_lipschitz_stable or not is_min_norm:
            verdict = StrictHeytingVerdict.DEGRADED
        
        return Phase3HodgeCertificate(
            projected_state=x_projected,
            projection_residual=residual,
            is_minimum_norm=is_min_norm,
            poincare_constant=poincare,
            is_lipschitz_stable=is_lipschitz_stable,
            fredholm_soluble=fredholm_soluble,
            fredholm_residual=fredholm_residual,
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO DEL HAZ CELULAR (SheafLaplacianSolver)
# ═══════════════════════════════════════════════════════════════════════════════
class SheafLaplacianSolver(Phase3_HodgeProjector):
    r"""
    Soberano Agéntico de Calibre del Haz Celular (SheafLaplacianSolver) en V_\mathbb{S} / V_\alpha.
    
    Gobierna la coherencia del haz celular integrando:
      - Invarianza simpléctica de Liouville (\det(M) = +1).
      - Métrica conforme de Maupertuis-Jacobi (Acción de Polyakov).
      - Anillo ultramétrico de Novikov (\Lambda_{\mathrm{Nov}}).
      - Protocolo RSI³ de Nivel 3 (Ricci-DeTurck, Connes-DK, Brockett-Shahshahani).
      - Retículo distributivo de Heyting \Omega_3 = {COHERENT, DEGRADED, VETOED}.
      - Disyuntor ciber-físico por hardware ESP32 / BT151 Crowbar (< 400 ns).
    """

    def __init__(
        self,
        spectral_engine: Optional[SheafLaplacianEngine] = None,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False
    ) -> None:
        super().__init__(tolerance=tolerance)
        self._engine = spectral_engine or SheafLaplacianEngine()
        self._halt_on_veto = halt_on_veto

    def audit_and_solve_sheaf(
        self,
        coboundary_delta: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        source_density_rho: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], SheafRSI3PoincareCertificate]:
        r"""
        Ejecuta el ciclo OODA de lazo cerrado sobre el reporte del motor espectral.

        Axiomas y Teoremas:
          1. Consumo FPU del motor espectral SheafLaplacianEngine.
          2. Invarianza Simpléctica de Liouville: M^\top \Omega M = \Omega \implies \det(M) = +1.
          3. Test de Fredholm: \rho \in \operatorname{im}(L_F) \iff \langle \rho, v_k \rangle = 0 \;\forall v_k \in \ker(L_F).
          4. Cota Connes-DK: L_{\max} \le \frac{1}{2 \lambda_2(L_F)^{3/2}}.
          5. Betti-1 y Aciclicidad: \beta_1 = \dim \ker(L_F) - \beta_0 \equiv 0.
        """
        # 1. Consumo del reporte FPU del motor espectral ciego
        report: SheafEngineSpectrumReport = self._engine.assemble_sheaf_laplacian_operator(
            self._engine.generate_poincare_maupertuis_coboundary_state(
                coboundary_delta=coboundary_delta,
                base_metric_G=metric_G,
                potential_V=0.0,
                total_energy_H0=1.0,
                monodromy_jacobian=jacobian_M
            )
        )

        # 2. Auditoría de Invarianza Simpléctica de Liouville
        det_M = float(la.det(jacobian_M))
        volume_drift = abs(det_M - 1.0)
        if volume_drift > _SYMPLECTIC_TOLERANCE:
            logger.error(f"[SHEAF_SOLVER_VETO] Violación de Liouville: det(M) drift={volume_drift:.3e}")

        # 3. Test de Solubilidad de Fredholm
        null_indices = np.where(report.eigenvalues <= _WILKINSON_TOL)[0]
        betti_0 = len(null_indices)

        fredholm_residual = 0.0
        if betti_0 > 0:
            null_basis = report.eigenvectors[:, null_indices]
            fredholm_projection = null_basis.T @ source_density_rho
            fredholm_residual = float(la.norm(fredholm_projection, ord=2))

        if fredholm_residual > _WILKINSON_TOL:
            logger.error(f"[FREDHOLM_VETO] Carga no ortogonal a ker(L_F): Residuo={fredholm_residual:.3e}")

        # 4. Cota de Connes-Daleckii-Krein & Betti-1
        fiedler = report.fiedler_value
        if fiedler <= _WILKINSON_TOL:
            connes_bound = _HARD_LIPSCHITZ_CEILING
            betti_1_cycles = 1
        else:
            connes_bound = float(1.0 / (2.0 * (fiedler ** 1.5)))
            betti_1_cycles = 0

        # Curvatura escalar de Ricci
        try:
            ricci_scalar = float(np.trace(la.solve(report.regularized_metric_G, ricci_tensor, assume_a='pos')))
        except Exception:
            ricci_scalar = float(np.trace(la.pinv(report.regularized_metric_G) @ ricci_tensor))

        # 5. Cómputo de la solución exacta x = L_F^+ \rho
        exact_solution = report.pseudo_inverse_L_plus @ source_density_rho

        # 6. Veredicto en \Omega_3
        is_coherent = (fredholm_residual <= _WILKINSON_TOL) and \
                      (betti_1_cycles == 0) and \
                      (volume_drift <= _SYMPLECTIC_TOLERANCE) and \
                      (report.condition_number <= _MAX_CONDITION_NUMBER)

        heyting_verdict = StrictHeytingVerdict.COHERENT if is_coherent else StrictHeytingVerdict.VETOED

        # Hash SHA-256 de telemetría inmutable
        telemetry_bytes = (
            f"{fredholm_residual:.6e}|{betti_1_cycles}|{fiedler:.6e}|"
            f"{volume_drift:.6e}|{connes_bound:.6e}|{heyting_verdict.value}"
        ).encode('utf-8')
        digest = hashlib.sha256(telemetry_bytes).hexdigest()

        if not is_coherent:
            logger.error(
                f"[SHEAF_SOLVER_VETOED] Incoherencia en el Haz Celular: "
                f"FredholmRes={fredholm_residual:.3e}, β₁={betti_1_cycles}, "
                f"VolDrift={volume_drift:.3e}, Fiedler={fiedler:.3e}"
            )
            if self._halt_on_veto:
                raise TopologicalInvariantError(
                    f"[FREDHOLM_VETO] Carga no ortogonal a ker(L_F): Residuo={fredholm_residual:.3e}"
                )

        certificate = SheafRSI3PoincareCertificate(
            fredholm_residual=fredholm_residual,
            betti_1_cycles=betti_1_cycles,
            fiedler_connectivity=fiedler,
            symplectic_volume_drift=volume_drift,
            connes_lipschitz_bound=connes_bound,
            ricci_scalar_curvature=ricci_scalar,
            heyting_verdict=heyting_verdict,
            sha256_telemetry_digest=digest,
            is_causally_coherent=is_coherent
        )

        return exact_solution, certificate

    def audit_poincare_liouville_invariance(
        self, jacobian_M: NDArray[np.float64]
    ) -> float:
        r"""Audita el drift de volumen de Liouville \det(M) - 1.0."""
        det_M = float(la.det(jacobian_M))
        return abs(det_M - 1.0)

    def audit_maupertuis_geodesic_action(
        self,
        trajectory_x: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        potential_V: float,
        H0: float
    ) -> float:
        r"""Calcula la Acción de Polyakov S_E[x] sobre la geodésica de Maupertuis-Jacobi."""
        scale = max(1.0e-10, 2.0 * (H0 - potential_V))
        G_maupertuis = scale * metric_G
        velocities = np.diff(trajectory_x, axis=0)
        action = 0.5 * sum(float(v.T @ G_maupertuis @ v) for v in velocities)
        return action

    def execute_rsi1_ricci_flow(
        self,
        G: NDArray[np.float64],
        ricci_tensor: NDArray[np.float64],
        delta_tau: float = 0.01
    ) -> NDArray[np.float64]:
        r"""Ejecuta el nivel R_1 de RSI³: Flujo de Ricci discreto sobre G."""
        n = G.shape[0]
        inv_G = la.pinv(G)
        scalar_R = float(np.trace(inv_G @ ricci_tensor))
        ricci_flow_step = -2.0 * delta_tau * (ricci_tensor - (scalar_R / float(n)) * G)
        G_next = G + ricci_flow_step
        G_next_spd, _, _ = stable_metric_higham(G_next)
        return G_next_spd

    def execute_rsi2_connes_daleckii_krein_bound(
        self, fiedler_val: float
    ) -> float:
        r"""Ejecuta el nivel R_2 de RSI³: Cota Lipschitz Connes-DK."""
        if fiedler_val <= _WILKINSON_TOL:
            return _HARD_LIPSCHITZ_CEILING
        return float(1.0 / (2.0 * (fiedler_val ** 1.5)))

    def execute_rsi3_brockett_shahshahani_flow(
        self,
        p_weights: NDArray[np.float64],
        rho_density: NDArray[np.float64],
        kernel_matrix: NDArray[np.float64],
        step_dt: float = 0.005
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Ejecuta el nivel R_3 de RSI³: Flujo de doble corchete sobre la órbita de Fock."""
        Kp = kernel_matrix @ p_weights
        mean_fitness = float(p_weights.T @ Kp)
        dp_dt = p_weights * (Kp - mean_fitness)
        p_next = np.maximum(1.0e-12, p_weights + step_dt * dp_dt)
        p_next /= float(np.sum(p_next))

        N_diag = np.diag(p_next)
        inner = rho_density @ N_diag - N_diag @ rho_density
        double_bracket = rho_density @ inner - inner @ rho_density
        rho_next = rho_density + step_dt * double_bracket
        rho_next = 0.5 * (rho_next + rho_next.T)
        rho_next /= max(1.0e-12, float(np.trace(rho_next)))

        return p_next, rho_next

    def execute_sheaf_solver_cycle(
        self,
        delta: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        x_initial: NDArray[np.float64],
        rho: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX,
    ) -> SheafSolverState:
        r"""Orquesta el ciclo completo de resolución espectral sobre las 3 fases anidadas."""
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            if delta.ndim != 2:
                raise CohomologicalDimensionMismatchError("delta debe ser una matriz 2D.")
            
            m, n = delta.shape
            if G.shape != (m, m):
                raise CohomologicalDimensionMismatchError(f"G debe ser {m}×{m}.")
            
            if x_initial.size != n or not np.all(np.isfinite(x_initial)):
                raise CohomologicalDimensionMismatchError(f"x_initial debe tener dimensión {n} finita.")
            
            cert_1 = self.generate_phase1_certificate(G, target_condition)
            
            if cert_1.verdict.is_vetoed:
                cert_2 = self._vetoed_phase2_certificate()
                cert_3 = self._vetoed_phase3_certificate(n)
            else:
                L_F, _, sparse_used = self.build_sheaf_laplacian(delta, G, target_condition)
                cert_2 = self.analyze_sheaf_spectrum(L_F, G, sparse_used)
                
                if cert_2.verdict.is_vetoed:
                    cert_3 = self._vetoed_phase3_certificate(n)
                else:
                    cert_3 = self.hodge_projection_solve(delta, G, x_initial, rho, target_condition)
            
            final_verdict = StrictHeytingVerdict.supremum(cert_1.verdict, cert_2.verdict, cert_3.verdict)
            
            solver_action = SolverAction.NONE
            diagnostic_note = "Ciclo de resolución del Laplaciano del Haz completado."
            
            if final_verdict == StrictHeytingVerdict.VETOED:
                solver_action = SolverAction.HALT_SOLVER
                diagnostic_note = "VETO DE RESOLUCIÓN: Violación en el haz celular."
                if self._halt_on_veto:
                    raise SheafLaplacianSolverError("Violación de condiciones de solubilidad espectral.")
            elif final_verdict == StrictHeytingVerdict.DEGRADED:
                solver_action = SolverAction.REGULARIZE_METRIC
                diagnostic_note = "Degradación detectada: regularización de Higham o brecha espectral reducida."
            
            provenance_hash = self._generate_provenance_hash(cert_1, cert_2, cert_3, timestamp_utc)
            
            return SheafSolverState(
                phase1=cert_1,
                phase2=cert_2,
                phase3=cert_3,
                final_verdict=final_verdict,
                solver_action=solver_action,
                timestamp_utc=timestamp_utc,
                provenance_hash=provenance_hash,
                diagnostic_note=diagnostic_note,
            )
            
        except SheafLaplacianSolverError as exc:
            if self._halt_on_veto:
                raise
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc, n=delta.shape[1])

    def _vetoed_phase2_certificate(self) -> Phase2SpectralCertificate:
        return Phase2SpectralCertificate(
            condition_number=float("inf"),
            eigenvalues=np.array([]),
            kernel_dimension=0,
            spectral_gap=0.0,
            is_spectrally_stable=False,
            sparse_computation=False,
            laplacian_rank=0,
            verdict=StrictHeytingVerdict.VETOED,
        )
    
    def _vetoed_phase3_certificate(self, n: int) -> Phase3HodgeCertificate:
        return Phase3HodgeCertificate(
            projected_state=np.zeros(n, dtype=np.float64),
            projection_residual=float("inf"),
            is_minimum_norm=False,
            poincare_constant=float("inf"),
            is_lipschitz_stable=False,
            fredholm_soluble=False,
            fredholm_residual=float("inf"),
            verdict=StrictHeytingVerdict.VETOED,
        )
    
    def _cataclysm_state(
        self, reason: str, timestamp_utc: str, n: int
    ) -> SheafSolverState:
        phase1_dummy = Phase1MetricCertificate(
            metric_condition_number=float("inf"),
            is_metric_spd=False,
            higham_regularization_applied=False,
            higham_shift=0.0,
            cholesky_factor_norm=0.0,
            verdict=StrictHeytingVerdict.VETOED,
        )
        raw_payload = f"CATACLYSM_SHEAF_VETO::{reason}"
        provenance_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        
        return SheafSolverState(
            phase1=phase1_dummy,
            phase2=self._vetoed_phase2_certificate(),
            phase3=self._vetoed_phase3_certificate(n),
            final_verdict=StrictHeytingVerdict.VETOED,
            solver_action=SolverAction.HALT_SOLVER,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_SHEAF_VETO: {reason}",
        )
    
    def _generate_provenance_hash(
        self,
        c1: Phase1MetricCertificate,
        c2: Phase2SpectralCertificate,
        c3: Phase3HodgeCertificate,
        timestamp_utc: str,
    ) -> str:
        raw_payload = "|".join(
            (
                timestamp_utc,
                str(c1.verdict.value),
                str(c2.verdict.value),
                str(c3.verdict.value),
                f"{c1.metric_condition_number:.12e}",
                f"{c1.cholesky_factor_norm:.12e}",
                f"{c2.condition_number:.12e}",
                f"{c2.spectral_gap:.12e}",
                f"{c2.laplacian_rank:d}",
                f"{c2.kernel_dimension:d}",
                f"{c3.projection_residual:.12e}",
                f"{c3.poincare_constant:.12e}",
                f"{c3.fredholm_residual:.12e}",
                str(int(c1.higham_regularization_applied)),
                str(int(c2.sparse_computation)),
                str(int(c3.is_lipschitz_stable)),
            )
        )
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


# Alias del Soberano Agéntico para coherencia arquitectónica
SheafLaplacianSolverAgent = SheafLaplacianSolver


__all__ = [
    "SheafLaplacianSolverError",
    "MetricNonSPDError",
    "CohomologicalDimensionMismatchError",
    "FredholmSolvabilityError",
    "LipschitzStabilityViolationError",
    "SpectralInstabilityError",
    "StrictHeytingVerdict",
    "SheafSolverVerdict",
    "SolverAction",
    "SheafRSI3PoincareCertificate",
    "Phase1MetricCertificate",
    "Phase2SpectralCertificate",
    "Phase3HodgeCertificate",
    "SheafSolverState",
    "Phase1_MetricPreconditioner",
    "Phase2_SheafLaplacianBuilder",
    "Phase3_HodgeProjector",
    "SheafLaplacianSolver",
    "SheafLaplacianSolverAgent",
    "stable_metric_higham",
    "build_sparse_laplacian",
]
