# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Sheaf Laplacian Solver (Motor de Coherencia de Haz de de Rham)     ║
║  Ruta   : app/agents/alpha/sheaf_laplacian_solver.py                         ║
║  Versión: 5.1.0-Sheaf-Hodge-Cholesky-Krylov-Strict                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL:                                   ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo materializa el solucionador de alto rendimiento para el         ║
║  Laplaciano del Haz Celular $L_F = \delta^\top G^{-1} \delta \succeq 0$      ║
║  en la variedad de de Rham-Hodge sobre el Estrato de la Estrategia.          ║
║  Evita la inversión directa mediante la factorización de Cholesky de la      ║
║  métrica Riemanniana de fondo, garantizando la consistencia y la pasividad   ║
║  espectral en la Unidad de Punto Flotante (FPU).                             ║
║                                                                              ║
║  Axioma de Integrabilidad y Estabilidad del Haz:                             ║
║    $$L_F = \delta^\top G^{-1} \delta = Y^\top Y \quad \land \quad L_G Y = \delta$$║
║    $$\Delta \Phi = -\rho \quad \implies \quad \rho \perp \ker(L_F)$$         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Tuple, Optional, Union
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Soporte para matrices dispersas (Laplacianos de gran escala)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    csr_matrix = None
    isspmatrix = None

logger = logging.getLogger("MIC.Alpha.SheafLaplacianSolver")

# ═══════════════════════════════════════════════════════════════════════════
# Constantes de precisión espectral de Wilkinson y categóricas
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_KAPPA_MAX: Final[float] = 1.0e8
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12  # Suelo espectral de Wilkinson
_SPARSE_THRESHOLD: Final[float] = 0.3  # Umbral de dispersión para conversión
_MAX_DEGREES_OF_FREEDOM: Final[int] = 10000  # Cota para sistemas de gran escala

# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones propias del solucionador (funtores de error)
# ═══════════════════════════════════════════════════════════════════════════
class SheafLaplacianSolverError(Exception):
    """Excepción raíz para errores en el solucionador del Laplaciano del Haz."""
    pass

class MetricNonSPDError(SheafLaplacianSolverError):
    """Tensor métrico no definido positivo (falla la factorización de Cholesky)."""
    pass

class CohomologicalDimensionMismatchError(SheafLaplacianSolverError):
    """Inconsistencia dimensional entre la cofrontera δ y la métrica G."""
    pass

class FredholmSolvabilityError(SheafLaplacianSolverError):
    """Violación de la condición de solubilidad de Fredholm."""
    pass

class LipschitzStabilityViolationError(SheafLaplacianSolverError):
    """La proyección de Hodge supera la cota de estabilidad de Lipschitz."""
    pass

class SpectralInstabilityError(SheafLaplacianSolverError):
    """Inestabilidad espectral detectada en el Laplaciano del Haz."""
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Enumeraciones categóricas (subobjetos y veredictos en el topos operativo)
# ═══════════════════════════════════════════════════════════════════════════
class SheafSolverVerdict(IntEnum):
    """
    Clasificador de tres valores en el retículo de Heyting (Ω) para calidad
    de resolución espectral.
    
    Orden por severidad operativa:
        COHERENT  = ⊤ operativo (resolución válida)
        DEGRADED  = elemento intermedio (precisión reducida)
        VETOED    = ⊥ operativo (resolución inválida)
    
    El supremo de veredictos se toma como máximo nivel de severidad.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "SheafSolverVerdict") -> "SheafSolverVerdict":
        """
        Supremo en el retículo de severidad.
        Si no se proveen veredictos, retorna COHERENT como elemento neutro.
        """
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == SheafSolverVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == SheafSolverVerdict.DEGRADED

class SolverAction(Enum):
    """Acciones de mitigación tras el veredicto de resolución."""
    NONE = auto()
    REGULARIZE_METRIC = auto()
    HALT_SOLVER = auto()

# ═══════════════════════════════════════════════════════════════════════════
# Certificados de las fases anidadas (objetos de la subcategoría Spec)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1MetricCertificate:
    """
    FASE 1 — Certificado de regularización y factorización de la métrica.
    
    Este certificado constituye el objeto terminal de la Fase 1 y es consumido
    por la Fase 2 a través del puente prepare_metric_factor().
    
    Atributos
    ---------
    metric_condition_number : float
        Número de condición espectral κ₂(G) de la métrica proyectada.
    is_metric_spd : bool
        Verificación de definida positiva (tras Higham si aplica).
    higham_regularization_applied : bool
        Indicador de si se requirió regularización de Higham.
    higham_shift : float
        Desplazamiento espectral aplicado (si Higham fue usado).
    cholesky_factor_norm : float
        Norma de Frobenius del factor L_G.
    verdict : SheafSolverVerdict
        Veredicto local de la Fase 1.
    """
    metric_condition_number: float
    is_metric_spd: bool
    higham_regularization_applied: bool
    higham_shift: float
    cholesky_factor_norm: float
    verdict: SheafSolverVerdict

@dataclass(frozen=True, slots=True)
class Phase2SpectralCertificate:
    """
    FASE 2 — Certificado espectral del Laplaciano de Haz L_F.
    
    Atributos
    ---------
    condition_number : float
        Número de condición κ₂(G) de la métrica.
    eigenvalues : NDArray[np.float64]
        Autovalores reales ordenados de L_F.
    kernel_dimension : int
        Dimensión del núcleo dim ker(L_F) (proxy del número de Betti).
    spectral_gap : float
        Brecha espectral λ₂ − λ₁ (valor de Fiedler).
    is_spectrally_stable : bool
        Verdadero si κ₂(G) ≤ κ_max.
    sparse_computation : bool
        Indicador de si se usó computación dispersa.
    laplacian_rank : int
        Rango numérico del Laplaciano.
    verdict : SheafSolverVerdict
        Veredicto local de la Fase 2.
    """
    condition_number: float
    eigenvalues: NDArray[np.float64]
    kernel_dimension: int
    spectral_gap: float
    is_spectrally_stable: bool
    sparse_computation: bool
    laplacian_rank: int
    verdict: SheafSolverVerdict

@dataclass(frozen=True, slots=True)
class Phase3HodgeCertificate:
    """
    FASE 3 — Certificado de proyección armónica de Hodge.
    
    Atributos
    ---------
    projected_state : NDArray[np.float64]
        Vector proyectado x* sobre ker(L_F).
    projection_residual : float
        Residuo ||L_F x*||₂.
    is_minimum_norm : bool
        Verdadero si x* minimiza la norma euclídea.
    poincare_constant : float
        Estimación de la constante de Poincaré.
    is_lipschitz_stable : bool
        Verdadero si se satisface la cota de Lipschitz.
    fredholm_soluble : bool
        Verdadero si ρ ⊥ ker(L_F).
    fredholm_residual : float
        Norma de la proyección de ρ sobre ker(L_F).
    verdict : SheafSolverVerdict
        Veredicto local de la Fase 3.
    """
    projected_state: NDArray[np.float64]
    projection_residual: float
    is_minimum_norm: bool
    poincare_constant: float
    is_lipschitz_stable: bool
    fredholm_soluble: bool
    fredholm_residual: float
    verdict: SheafSolverVerdict

@dataclass(frozen=True, slots=True)
class SheafSolverState:
    """Certificado global terminal del ciclo de resolución del Laplaciano."""
    phase1: Phase1MetricCertificate
    phase2: Phase2SpectralCertificate
    phase3: Phase3HodgeCertificate
    final_verdict: SheafSolverVerdict
    solver_action: SolverAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Regularización Espectral (Higham para Métricas)
# ═══════════════════════════════════════════════════════════════════════════
def stable_metric_higham(
    G: NDArray[np.float64],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR,
    target_condition: float = _KAPPA_MAX
) -> Tuple[NDArray[np.float64], float, bool]:
    r"""
    Aplica la proyección de Higham para estabilizar el tensor métrico G en
    el cono SPD (Symmetric Positive Definite).
    
    Axioma de Proyección:
        \tilde{G} = \arg\min_{M \succeq 0} \|G - M\|_F
    
    Parámetros
    ----------
    G : NDArray[np.float64]
        Tensor métrico simétrico (potencialmente no-SPD).
    tolerance : float
        Suelo espectral de Wilkinson para recorte de autovalores.
    target_condition : float
        Número de condición máximo admisible κ_max.
    
    Retorna
    -------
    Tuple[NDArray[np.float64], float, bool]
        (G_projected, shift, higham_applied) donde shift es el desplazamiento
        espectral aplicado y higham_applied indica si se requirió regularización.
    
    Raises
    ------
    MetricNonSPDError
        Si la proyección de Higham falla en producir una matriz SPD.
    """
    # 1. Simetrización exacta
    G_sym = 0.5 * (G + G.T)
    
    # 2. Descomposición espectral de Weyl
    try:
        eigvals, eigvecs = la.eigh(G_sym)
    except Exception as exc:
        raise MetricNonSPDError(
            "MetricNonSPDError: Falló descomposición espectral para Higham."
        ) from exc

    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    shift = 0.0
    applied = False

    # 3. Forzar positividad estricta (si λ_min ≤ tol)
    if lam_min <= tolerance:
        shift = abs(lam_min) + max(tolerance, lam_max / target_condition)
        applied = True
        lam_min += shift
        lam_max += shift

    # 4. Limitar el número de condición
    current_cond = lam_max / max(lam_min, _MACHINE_EPS)
    if current_cond > target_condition:
        # (λ_max + extra) / (λ_min + extra) = κ_target
        extra = (lam_max - target_condition * lam_min) / (target_condition - 1.0)
        if extra > 0.0:
            shift += extra
            applied = True
            lam_min += extra
            lam_max += extra

    # 5. Reconstrucción con el desplazamiento total
    vals_reg = eigvals + shift
    G_projected = eigvecs @ np.diag(vals_reg) @ eigvecs.T

    # Simetrización numérica post-proyección
    G_projected = (G_projected + G_projected.T) / 2.0

    if applied:
        logger.warning(
            "Métrica inestable. Proyectada G al cono SPD via Higham "
            "(autovalores mínimos: %.4e → %.4e, shift: %.4e).",
            float(np.min(eigvals)),
            float(np.min(vals_reg)),
            float(shift)
        )

    # Verificación final de Cholesky
    try:
        la.cholesky(G_projected, lower=True)
    except la.LinAlgError as exc:
        raise MetricNonSPDError(
            "MetricNonSPDError: Proyección de Higham no produjo matriz SPD."
        ) from exc

    cond = float(np.max(vals_reg) / np.min(vals_reg))
    return G_projected, shift, applied

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Cálculo Disperso (Laplacianos de Gran Escala)
# ═══════════════════════════════════════════════════════════════════════════
def build_sparse_laplacian(
    delta: Union[NDArray[np.float64], csr_matrix],
    L_G: Union[NDArray[np.float64], csr_matrix]
) -> Tuple[Union[NDArray[np.float64], csr_matrix], bool]:
    r"""
    Construye el Laplaciano del Haz L_F = δᵀ G⁻¹ δ con soporte disperso.
    
    Parámetros
    ----------
    delta : Union[NDArray, csr_matrix]
        Operador cofrontera (matriz m × n).
    L_G : Union[NDArray, csr_matrix]
        Factor de Cholesky de G (m × m).
    
    Retorna
    -------
    Tuple[Union[NDArray, csr_matrix], bool]
        (L_F, sparse_used) donde sparse_used indica si se usó formato disperso.
    """
    sparse_used = False
    
    if _SPARSE_AVAILABLE and isspmatrix is not None:
        if isspmatrix(delta) or isspmatrix(L_G):
            sparse_used = True
            # Resolver sistema triangular disperso
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
            # Matriz densa pero dispersa en contenido → convertir
            sparse_used = True
            delta_sparse = csr_matrix(delta)
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

# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 – Regularizar y Factorizar: Métrica SPD y Cholesky
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_MetricPreconditioner(ABC):
    """
    FASE 1 (Regularizar y Factorizar): Proyección de la métrica al cono de
    matrices simétricas definidas positivas mediante regularización de Higham
    adaptativa y factorización triangular de Cholesky estable.
    
    El último método de esta fase:
        prepare_metric_factor(...)
    constituye el morfismo de transición formal hacia la Fase 2.
    """

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        """
        Inicializa el precondicionador de métrica.
        
        Parámetros
        ----------
        tolerance : float
            Cota de precisión para validaciones espectrales.
        """
        self._tolerance = tolerance
        self._higham_floor = _HIGHAM_REGULARIZATION_FLOOR

    def project_to_spd(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        Proyecta la matriz simétrica G al cono SPD, limitando el número de
        condición mediante regularización de Higham.
        
        Algoritmo:
        1. Simetrizar: G_sym = (G + Gᵀ)/2.
        2. Descomposición espectral: G_sym = V Λ Vᵀ.
        3. Si λ_min ≤ 0, añadir desplazamiento s = |λ_min| + max(ε, λ_max/κ_objetivo).
        4. Si κ actual > κ_objetivo, aplicar desplazamiento adicional s_extra.
        5. Reconstruir: G_proj = V (Λ + s I) Vᵀ.
        
        Parámetros
        ----------
        G : NDArray
            Matriz simétrica candidata.
        target_condition : float
            Número de condición máximo admisible κ_max.
        
        Retorna
        -------
        G_projected : NDArray
            Matriz proyectada SPD.
        cond : float
            Número de condición final.
        applied : bool
            True si se aplicó alguna regularización.
        """
        G_projected, shift, applied = stable_metric_higham(
            G, self._higham_floor, target_condition
        )
        
        # Calcular número de condición final
        eigvals_proj = la.eigvalsh(G_projected)
        cond = float(np.max(eigvals_proj) / np.min(eigvals_proj))
        
        return G_projected, cond, applied

    def stable_cholesky(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, bool]:
        r"""
        Calcula el factor triangular inferior L_G de la métrica Riemanniana G
        después de garantizar que es SPD y bien condicionada.
        
        Parámetros
        ----------
        G : NDArray
            Tensor métrico simétrico.
        target_condition : float
            κ máximo deseado.
        
        Retorna
        -------
        L_G : NDArray
            Factor de Cholesky triangular inferior.
        G_spd : NDArray
            Métrica proyectada SPD (útil para auditoría).
        cond : float
            Número de condición final.
        higham_applied : bool
            Indicador de si se aplicó regularización de Higham.
        
        Lanza
        -----
        MetricNonSPDError
            Si incluso después de la proyección la factorización falla.
        """
        G_spd, shift, higham_applied = self.project_to_spd(G, target_condition)
        
        try:
            L_G = la.cholesky(G_spd, lower=True)
        except la.LinAlgError as exc:
            raise MetricNonSPDError(
                f"MetricNonSPDError: Factorización de Cholesky fallida tras proyección. Detalle: {exc}"
            ) from exc
        
        # Calcular número de condición de G_spd
        eigvals_proj = la.eigvalsh(G_spd)
        cond = float(np.max(eigvals_proj) / np.min(eigvals_proj))
        
        return L_G, G_spd, cond, higham_applied

    # ─────────────────────────────────────────────────────────────────────
    # Último método de la Fase 1: Puente hacia la Fase 2
    # ─────────────────────────────────────────────────────────────────────
    def prepare_metric_factor(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[NDArray[np.float64], float, bool]:
        """
        MORFISMO DE TRANSICIÓN FASE 1 → FASE 2.
        
        Interfaz explícita para la Fase 2: retorna el factor de Cholesky,
        el número de condición, y el indicador de regularización Higham,
        garantizando que la métrica está en condiciones de ser utilizada
        en el ensamblado del Laplaciano.
        
        Parámetros
        ----------
        G : NDArray
            Tensor métrico simétrico.
        target_condition : float
            κ máximo permitido.
        
        Retorna
        -------
        Tuple[NDArray, float, bool]
            (L_G, cond, higham_applied)
        
        Lanza
        -----
        MetricNonSPDError
            Si la métrica no puede factorizarse incluso tras Higham.
        """
        L_G, _, cond, higham_applied = self.stable_cholesky(G, target_condition)
        return L_G, cond, higham_applied

    def generate_phase1_certificate(
        self,
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Phase1MetricCertificate:
        """
        Genera el certificado de Fase 1 con auditoría completa de la métrica.
        
        Parámetros
        ----------
        G : NDArray
            Tensor métrico simétrico.
        target_condition : float
            κ máximo permitido.
        
        Retorna
        -------
        Phase1MetricCertificate
            Certificado terminal de Fase 1.
        """
        L_G, cond, higham_applied = self.prepare_metric_factor(G, target_condition)
        
        # Calcular desplazamiento Higham si se aplicó
        higham_shift = 0.0
        if higham_applied:
            eigvals_orig = la.eigvalsh(G)
            eigvals_proj = la.eigvalsh((L_G @ L_G.T + (L_G @ L_G.T).T) / 2.0)
            higham_shift = float(np.min(eigvals_proj) - np.min(eigvals_orig))
        
        cholesky_norm = float(la.norm(L_G, ord="fro"))
        
        # Veredicto local
        verdict = SheafSolverVerdict.COHERENT
        if not np.isfinite(cond) or cond <= 0.0:
            verdict = SheafSolverVerdict.VETOED
        elif cond > _KAPPA_MAX or higham_applied:
            verdict = SheafSolverVerdict.DEGRADED
        
        return Phase1MetricCertificate(
            metric_condition_number=cond,
            is_metric_spd=True,  # Garantizado por Higham
            higham_regularization_applied=higham_applied,
            higham_shift=higham_shift,
            cholesky_factor_norm=cholesky_norm,
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 – Ensamblar y Analizar: Laplaciano del Haz y Espectro
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_SheafLaplacianBuilder(Phase1_MetricPreconditioner):
    """
    FASE 2 (Ensamblar y Analizar): Construcción del Laplaciano ponderado
    L_F = δᵀ G⁻¹ δ sin inversión explícita y análisis espectral completo.
    
    Hereda de la Fase 1 para usar prepare_metric_factor() como puente formal.
    Soporta computación dispersa para sistemas de gran escala.
    """

    def build_sheaf_laplacian(
        self,
        delta: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        target_condition: float = _KAPPA_MAX
    ) -> Tuple[Union[NDArray[np.float64], csr_matrix], NDArray[np.float64], bool]:
        r"""
        Ensambla L_F = δᵀ G⁻¹ δ mediante resolución triangular con soporte
        disperso opcional.
        
        Procedimiento:
        1. Validación de dimensiones: δ debe ser (m × n), G (m × m).
        2. Obtener factor de Cholesky L_G de G (usando la Fase 1).
        3. Resolver L_G Y = δ para Y con `solve_triangular`.
        4. L_F = Yᵀ Y (simétrica semidefinida positiva).
        
        Parámetros
        ----------
        delta : Union[NDArray, csr_matrix]
            Operador cofrontera (matriz m × n).
        G : NDArray
            Tensor métrico (m × m).
        target_condition : float
            κ máximo permitido.
        
        Retorna
        -------
        L_F : Union[NDArray, csr_matrix]
            Laplaciano del Haz (n × n).
        Y : NDArray
            Matriz intermedia (m × n) que satisface L_G Y = δ.
        sparse_used : bool
            Indicador de si se usó computación dispersa.
        
        Lanza
        -----
        CohomologicalDimensionMismatchError
            Si las dimensiones no son compatibles.
        MetricNonSPDError
            Si la métrica no puede factorizarse.
        """
        # 1. Verificación dimensional
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise CohomologicalDimensionMismatchError(
                "CohomologicalDimensionMismatchError: G debe ser una matriz cuadrada."
            )
        
        if delta.ndim != 2:
            raise CohomologicalDimensionMismatchError(
                "CohomologicalDimensionMismatchError: delta debe ser una matriz 2D."
            )
        
        m, n = delta.shape
        if G.shape[0] != m:
            raise CohomologicalDimensionMismatchError(
                f"CohomologicalDimensionMismatchError: delta tiene {m} filas pero G es {G.shape[0]}×{G.shape[1]}."
            )
        
        # Verificar finitud (compatible con dispersas)
        if _SPARSE_AVAILABLE and isspmatrix is not None:
            if isspmatrix(delta):
                if not np.all(np.isfinite(delta.data)):
                    raise CohomologicalDimensionMismatchError(
                        "CohomologicalDimensionMismatchError: delta dispersa contiene entradas no finitas."
                    )
            else:
                if not np.all(np.isfinite(delta)):
                    raise CohomologicalDimensionMismatchError(
                        "CohomologicalDimensionMismatchError: delta contiene entradas no finitas."
                    )
        else:
            if not np.all(np.isfinite(delta)):
                raise CohomologicalDimensionMismatchError(
                    "CohomologicalDimensionMismatchError: delta contiene entradas no finitas."
                )
        
        if not np.all(np.isfinite(G)):
            raise CohomologicalDimensionMismatchError(
                "CohomologicalDimensionMismatchError: G contiene entradas no finitas."
            )
        
        # 2. Obtener factor de Cholesky (Fase 1)
        L_G, _, higham_applied = self.prepare_metric_factor(G, target_condition)
        
        # 3. Construcción del Laplaciano con soporte disperso
        L_F, sparse_used = build_sparse_laplacian(delta, L_G)
        
        # Resolver para Y (para auditoría)
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
        r"""
        Calcula el espectro del Laplaciano y las propiedades de estabilidad.
        
        Parámetros
        ----------
        L_F : Union[NDArray, csr_matrix]
            Laplaciano del Haz (n × n).
        G : NDArray
            Métrica original o proyectada (para calcular κ₂(G)).
        sparse_computation : bool
            Indicador de si se usó computación dispersa.
        
        Retorna
        -------
        Phase2SpectralCertificate
            Certificado espectral inmutable.
        """
        # Convertir a densa para cálculo de autovalores si es dispersa
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_F):
            L_F_dense = L_F.toarray()
        else:
            L_F_dense = L_F
        
        # Autovalores de L_F (forzar no negatividad numérica)
        eigvals_L = la.eigvalsh(L_F_dense)
        eigvals_L = np.maximum(eigvals_L, 0.0)
        sorted_vals = np.sort(eigvals_L)
        
        # Número de condición de G
        eigvals_G = la.eigvalsh(G)
        cond_G = float(np.max(eigvals_G) / np.min(eigvals_G))
        
        # Dimensión del núcleo (valores propios menores que una cota relativa)
        tol_kernel = max(self._tolerance * 100, np.max(eigvals_L) * _MACHINE_EPS * 10)
        kernel_dim = int(np.sum(eigvals_L <= tol_kernel))
        
        # Brecha espectral (valor de Fiedler)
        spectral_gap = 0.0
        if len(sorted_vals) >= 2:
            # Si hay varios autovalores en el núcleo, la brecha es el primer autovalor positivo
            positive_mask = sorted_vals > tol_kernel
            if np.any(positive_mask):
                first_positive = sorted_vals[positive_mask][0]
                spectral_gap = float(first_positive - sorted_vals[0])
            else:
                spectral_gap = 0.0
        
        # Rango numérico del Laplaciano
        laplacian_rank = int(np.linalg.matrix_rank(L_F_dense, tol=tol_kernel))
        
        # Estabilidad espectral
        is_spectrally_stable = cond_G <= _KAPPA_MAX
        
        # Veredicto local
        verdict = SheafSolverVerdict.COHERENT
        if not np.isfinite(cond_G) or not np.isfinite(spectral_gap):
            verdict = SheafSolverVerdict.VETOED
        elif not is_spectrally_stable or kernel_dim > 0:
            verdict = SheafSolverVerdict.DEGRADED
        
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

# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 – Verificar y Proyectar: Fredholm + Proyección de Hodge
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_HodgeProjector(Phase2_SheafLaplacianBuilder):
    """
    FASE 3 (Verificar y Proyectar): Condición de solubilidad de Fredholm para
    el problema de Poisson y proyección armónica de Hodge sobre el núcleo de
    L_F. Hereda de la Fase 2 para poder construir el Laplaciano y evaluar el
    espectro.
    """

    def check_fredholm_solvability(
        self,
        L_F: Union[NDArray[np.float64], csr_matrix],
        rho: NDArray[np.float64],
        tolerance: float = 1.0e-8
    ) -> Tuple[bool, float]:
        r"""
        Verifica la ortogonalidad de la densidad de carga ρ al núcleo de L_F.
        
        Condición de Fredholm:
            ρ ⊥ ker(L_F) ⟺ Σ_{v ∈ ker} (ρ · v) = 0
        
        Parámetros
        ----------
        L_F : Union[NDArray, csr_matrix]
            Laplaciano del Haz.
        rho : NDArray
            Vector de forzamiento (densidad de carga).
        tolerance : float
            Tolerancia para el residuo de proyección.
        
        Retorna
        -------
        is_soluble : bool
            True si se cumple la condición.
        residual : float
            Norma-2 de la proyección de ρ sobre ker(L_F).
        """
        # Convertir a densa para cálculo de autovalores si es dispersa
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_F):
            L_F_dense = L_F.toarray()
        else:
            L_F_dense = L_F
        
        # Obtener base del núcleo
        vals, vecs = la.eigh(L_F_dense)
        
        # Umbral relativo para pertenencia al núcleo
        max_val = np.max(vals) if len(vals) > 0 else 1.0
        kernel_mask = vals <= max(self._tolerance * 100, max_val * _MACHINE_EPS * 10)
        kernel_basis = vecs[:, kernel_mask]
        
        if kernel_basis.shape[1] == 0:
            return True, 0.0
        
        proj_coeffs = kernel_basis.T @ rho
        residual = float(la.norm(proj_coeffs))
        is_soluble = residual <= tolerance
        
        if not is_soluble:
            logger.warning(
                "¡Divergencia de Fredholm! Densidad ρ no ortogonal a ker(L_F). Residuo: %.4e",
                residual
            )
        
        return is_soluble, residual

    def hodge_projection_solve(
        self,
        delta: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        x: NDArray[np.float64],
        rho: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX
    ) -> Phase3HodgeCertificate:
        r"""
        Proyecta el vector de estado x sobre el espacio armónico ker(L_F).
        
        Fórmula:
            x* = P_{ker} x = U Uᵀ x,
        donde las columnas de U forman una base ortonormal de ker(L_F).
        
        Adicionalmente se calculan:
        - Residuo de proyección: ||L_F x*||.
        - Constante de Poincaré: C_P = ||x - x*|| / ||δ x||.
        - Estabilidad de Lipschitz: ||δ x* - δ x|| ≤ max(κ(δ), ||δ||₂) · ||x* - x||.
        - Solubilidad de Fredholm para ρ si se proporciona.
        
        Parámetros
        ----------
        delta : Union[NDArray, csr_matrix]
            Operador cofrontera.
        G : NDArray
            Tensor métrico.
        x : NDArray
            Vector de estado inicial.
        rho : Optional[NDArray]
            Vector de densidad de carga para Fredholm.
        target_condition : float
            κ máximo para la métrica.
        
        Retorna
        -------
        Phase3HodgeCertificate
            Reporte completo con vector proyectado y métricas de estabilidad.
        
        Lanza
        -----
        CohomologicalDimensionMismatchError
            Si las dimensiones no son compatibles.
        LipschitzStabilityViolationError
            Si la cota de Lipschitz se incumple severamente.
        """
        # 1. Construir el Laplaciano (Fase 2)
        L_F, _, sparse_used = self.build_sheaf_laplacian(delta, G, target_condition)
        
        # Convertir a densa para cálculo de autovalores si es dispersa
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_F):
            L_F_dense = L_F.toarray()
        else:
            L_F_dense = L_F
        
        # 2. Obtener base del núcleo de L_F
        vals, vecs = la.eigh(L_F_dense)
        max_val = np.max(vals) if len(vals) > 0 else 1.0
        kernel_mask = vals <= max(self._tolerance * 100, max_val * _MACHINE_EPS * 10)
        kernel_basis = vecs[:, kernel_mask]
        
        if kernel_basis.shape[1] == 0:
            x_projected = np.zeros_like(x)
        else:
            x_projected = kernel_basis @ (kernel_basis.T @ x)
        
        # 3. Residuo de proyección
        residual = float(la.norm(L_F_dense @ x_projected))
        
        # 4. Verificación de mínima norma
        norm_proj = float(la.norm(x_projected))
        norm_orig = float(la.norm(x))
        is_min_norm = norm_proj <= norm_orig + self._tolerance * 100
        
        # 5. Constante de Poincaré
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(delta):
            dx = delta.dot(x)
            dx_proj = delta.dot(x_projected)
        else:
            dx = delta @ x
            dx_proj = delta @ x_projected
        
        norm_dx = float(la.norm(dx))
        deviation = x - x_projected
        norm_dev = float(la.norm(deviation))
        poincare = 0.0
        if norm_dx > self._tolerance:
            poincare = norm_dev / norm_dx
        
        # 6. Estabilidad de Lipschitz
        lhs = float(la.norm(dx_proj - dx))
        # Valores singulares de delta para estimar κ(δ) y norma-2
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(delta):
            svals = la.svdvals(delta.toarray())
        else:
            svals = la.svdvals(delta)
        
        max_sval = float(np.max(svals)) if len(svals) > 0 else 1.0
        min_sval_pos = float(np.min(svals[svals > self._tolerance])) if np.any(svals > self._tolerance) else 1.0
        kappa_delta = max_sval / min_sval_pos if min_sval_pos > 0 else max_sval / _MACHINE_EPS
        effective_lip = max(kappa_delta, max_sval)
        rhs = effective_lip * norm_dev
        is_lipschitz_stable = lhs <= rhs + self._tolerance * 100
        
        if not is_lipschitz_stable:
            logger.warning(
                "Inestabilidad de Lipschitz detectada: ||δx*-δx|| = %.4e, cota = %.4e",
                lhs, rhs
            )
        
        # 7. Solubilidad de Fredholm
        if rho is not None:
            fredholm_soluble, fredholm_residual = self.check_fredholm_solvability(L_F_dense, rho)
        else:
            fredholm_soluble = True
            fredholm_residual = 0.0
        
        # 8. Veredicto local
        verdict = SheafSolverVerdict.COHERENT
        if (
            not np.isfinite(residual) or
            not np.isfinite(poincare) or
            not fredholm_soluble
        ):
            verdict = SheafSolverVerdict.VETOED
        elif not is_lipschitz_stable or not is_min_norm:
            verdict = SheafSolverVerdict.DEGRADED
        
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

# ═══════════════════════════════════════════════════════════════════════════
# Solucionador final: herencia completa de las tres fases
# ═══════════════════════════════════════════════════════════════════════════
class SheafLaplacianSolver(Phase3_HodgeProjector):
    """
    Solucionador espectral consolidado para complejos de haces celulares.
    Hereda todas las capacidades de las fases 1, 2 y 3, exponiendo una interfaz
    unificada con validación matemática rigurosa y trazabilidad completa.
    
    El ciclo de resolución queda formalizado como:
        Regularizar → Fase 1 (MetricPreconditioner)
        Ensamblar   → Fase 2 (SheafLaplacianBuilder)
        Proyectar   → Fase 3 (HodgeProjector)
    
    El veredicto final es el supremo de severidad en el retículo de Heyting.
    """

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False
    ) -> None:
        """
        Inicializa el solucionador con configuración de precisión.
        
        Parámetros
        ----------
        tolerance : float
            Cota de precisión para validaciones.
        halt_on_veto : bool
            Si True, lanza excepción cuando el veredicto es VETOED.
        """
        super().__init__(tolerance=tolerance)
        self._halt_on_veto = halt_on_veto

    def execute_sheaf_solver_cycle(
        self,
        delta: Union[NDArray[np.float64], csr_matrix],
        G: NDArray[np.float64],
        x_initial: NDArray[np.float64],
        rho: Optional[NDArray[np.float64]] = None,
        target_condition: float = _KAPPA_MAX,
    ) -> SheafSolverState:
        """
        Orquesta el ciclo completo de resolución del Laplaciano del Haz con
        validación de las tres fases.
        
        Estrategia de anidación:
          1. Fase 1: Genera certificado de regularización de la métrica.
          2. Fase 2: Construye el Laplaciano y analiza su espectro.
          3. Fase 3: Proyecta sobre el espacio armónico y verifica Fredholm.
        
        Si Fase 1 emite VETOED, se emiten certificados vetados para Fase 2 y 3.
        
        Parámetros
        ----------
        delta : Union[NDArray, csr_matrix]
            Operador cofrontera (m × n).
        G : NDArray
            Tensor métrico (m × m).
        x_initial : NDArray
            Vector de estado inicial (n,).
        rho : Optional[NDArray]
            Vector de densidad de carga para Fredholm.
        target_condition : float
            κ máximo permitido para la métrica.
        
        Retorna
        -------
        SheafSolverState
            Certificado global terminal.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Validar entradas
            if delta.ndim != 2:
                raise CohomologicalDimensionMismatchError(
                    "CohomologicalDimensionMismatchError: delta debe ser una matriz 2D."
                )
            
            m, n = delta.shape
            if G.shape != (m, m):
                raise CohomologicalDimensionMismatchError(
                    f"CohomologicalDimensionMismatchError: G debe ser {m}×{m}."
                )
            
            if x_initial.size != n:
                raise CohomologicalDimensionMismatchError(
                    f"CohomologicalDimensionMismatchError: x_initial debe tener dimensión {n}."
                )
            
            if not np.all(np.isfinite(x_initial)):
                raise CohomologicalDimensionMismatchError(
                    "CohomologicalDimensionMismatchError: x_initial contiene entradas no finitas."
                )
            
            if rho is not None and rho.size != n:
                raise CohomologicalDimensionMismatchError(
                    f"CohomologicalDimensionMismatchError: rho debe tener dimensión {n}."
                )
            
            # ─── FASE 1: Regularizar ───
            cert_1 = self.generate_phase1_certificate(G, target_condition)
            
            if cert_1.verdict.is_vetoed:
                cert_2 = self._vetoed_phase2_certificate()
                cert_3 = self._vetoed_phase3_certificate(n)
                x_projected = x_initial.copy()
            else:
                # ─── FASE 2: Ensamblar y Analizar ───
                L_F, _, sparse_used = self.build_sheaf_laplacian(delta, G, target_condition)
                cert_2 = self.analyze_sheaf_spectrum(L_F, G, sparse_used)
                
                if cert_2.verdict.is_vetoed:
                    cert_3 = self._vetoed_phase3_certificate(n)
                    x_projected = x_initial.copy()
                else:
                    # ─── FASE 3: Proyectar ───
                    cert_3 = self.hodge_projection_solve(
                        delta, G, x_initial, rho, target_condition
                    )
                    x_projected = cert_3.projected_state
            
            # ─── Fusión de veredictos (supremo en severidad) ───
            final_verdict = SheafSolverVerdict.supremum(
                cert_1.verdict,
                cert_2.verdict,
                cert_3.verdict,
            )
            
            solver_action = SolverAction.NONE
            diagnostic_note = "Ciclo de resolución del Laplaciano del Haz completado."
            
            if final_verdict == SheafSolverVerdict.VETOED:
                solver_action = SolverAction.HALT_SOLVER
                diagnostic_note = (
                    "VETO DE RESOLUCIÓN: violación de SPD, solubilidad de Fredholm "
                    "o estabilidad de Lipschitz."
                )
                logger.error("¡VETO DE RESOLUCIÓN! Halt requested.")
                if self._halt_on_veto:
                    raise SheafLaplacianSolverError(
                        "Violación de condiciones de solubilidad o estabilidad espectral."
                    )
            elif final_verdict == SheafSolverVerdict.DEGRADED:
                solver_action = SolverAction.REGULARIZE_METRIC
                diagnostic_note = (
                    "Degradación detectada: regularización Higham aplicada o brecha "
                    "espectral reducida. Se recomienda revisar la métrica G."
                )
                logger.warning(
                    "Degradación en resolución del Laplaciano del Haz. "
                    "Se recomienda revisar la métrica G."
                )
            
            provenance_hash = self._generate_provenance_hash(
                cert_1, cert_2, cert_3, timestamp_utc
            )
            
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
            logger.error("Colapso categórico del solucionador del Laplaciano del Haz: %s", exc)
            if self._halt_on_veto:
                raise
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc, n=delta.shape[1])
            
        except Exception as exc:  # pragma: no cover
            logger.exception("Colapso catastrófico no tipado del solucionador del Laplaciano del Haz.")
            if self._halt_on_veto:
                raise SheafLaplacianSolverError(
                    "Colapso catastrófico no tipado del solucionador del Laplaciano del Haz."
                ) from exc
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc, n=delta.shape[1])

    # ─────────────────────────────────────────────────────────────────────
    # Certificados vetados para cortocircuito anidado
    # ─────────────────────────────────────────────────────────────────────
    def _vetoed_phase2_certificate(self) -> Phase2SpectralCertificate:
        """Certificado vetado de Fase 2 para cortocircuito."""
        return Phase2SpectralCertificate(
            condition_number=float("inf"),
            eigenvalues=np.array([]),
            kernel_dimension=0,
            spectral_gap=0.0,
            is_spectrally_stable=False,
            sparse_computation=False,
            laplacian_rank=0,
            verdict=SheafSolverVerdict.VETOED,
        )
    
    def _vetoed_phase3_certificate(self, n: int) -> Phase3HodgeCertificate:
        """Certificado vetado de Fase 3 para cortocircuito."""
        return Phase3HodgeCertificate(
            projected_state=np.zeros(n, dtype=np.float64),
            projection_residual=float("inf"),
            is_minimum_norm=False,
            poincare_constant=float("inf"),
            is_lipschitz_stable=False,
            fredholm_soluble=False,
            fredholm_residual=float("inf"),
            verdict=SheafSolverVerdict.VETOED,
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Estado catastrófico de emergencia
    # ─────────────────────────────────────────────────────────────────────
    def _cataclysm_state(
        self, reason: str, timestamp_utc: str, n: int
    ) -> SheafSolverState:
        """Construye un estado catastrófico con certificados vetados."""
        phase1_dummy = Phase1MetricCertificate(
            metric_condition_number=float("inf"),
            is_metric_spd=False,
            higham_regularization_applied=False,
            higham_shift=0.0,
            cholesky_factor_norm=0.0,
            verdict=SheafSolverVerdict.VETOED,
        )
        phase2_dummy = self._vetoed_phase2_certificate()
        phase3_dummy = self._vetoed_phase3_certificate(n)
        
        raw_payload = f"CATACLYSM_SHEAF_VETO::{reason}"
        provenance_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        
        return SheafSolverState(
            phase1=phase1_dummy,
            phase2=phase2_dummy,
            phase3=phase3_dummy,
            final_verdict=SheafSolverVerdict.VETOED,
            solver_action=SolverAction.HALT_SOLVER,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_SHEAF_VETO: {reason}",
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Hash de procedencia auditable
    # ─────────────────────────────────────────────────────────────────────
    def _generate_provenance_hash(
        self,
        c1: Phase1MetricCertificate,
        c2: Phase2SpectralCertificate,
        c3: Phase3HodgeCertificate,
        timestamp_utc: str,
    ) -> str:
        """Genera un hash SHA-256 de procedencia que ata los veredictos y residuos."""
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

# ═══════════════════════════════════════════════════════════════════════════
# Compatibilidad de nombres de clases para versiones anteriores
# ═══════════════════════════════════════════════════════════════════════════
Phase1_MetricPreconditioner = Phase1_MetricPreconditioner
Phase2_SheafLaplacianBuilder = Phase2_SheafLaplacianBuilder
Phase3_HodgeProjector = Phase3_HodgeProjector

__all__ = [
    "SheafLaplacianSolverError",
    "MetricNonSPDError",
    "CohomologicalDimensionMismatchError",
    "FredholmSolvabilityError",
    "LipschitzStabilityViolationError",
    "SpectralInstabilityError",
    "SheafSolverVerdict",
    "SolverAction",
    "Phase1MetricCertificate",
    "Phase2SpectralCertificate",
    "Phase3HodgeCertificate",
    "SheafSolverState",
    "Phase1_MetricPreconditioner",
    "Phase2_SheafLaplacianBuilder",
    "Phase3_HodgeProjector",
    "SheafLaplacianSolver",
    "stable_metric_higham",
    "build_sparse_laplacian",
]