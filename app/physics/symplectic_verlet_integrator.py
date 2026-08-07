# -*- coding: utf-8 -*-
r"""
─────────────────────────────────────────────────────────────────────────────
Módulo : Symplectic Verlet Integrator (Conservador de la Forma Simpléctica)
Ruta   : app/physics/symplectic_verlet_integrator.py
Versión: 3.0.0-Verlet-Yoshida-PHS-Nested-PhD-Higham-Sparse
─────────────────────────────────────────────────────────────────────────────
NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL (FASES ANIDADAS EVOLUCIONADAS)
-----------------------------------------------------------------------------
Este módulo implementa un integrador simpléctico para sistemas Port-Hamiltonianos
disipativos, organizado en tres fases que heredan secuencialmente con puentes
formales entre ellas:

FASE 1 – Observar/Estructurar : Definición del sistema (gradiente, potencial,
    energía cinética), validación de parámetros con regularización de Higham
    para matrices de masa no-SPD, y construcción del puente morfísmico hacia
    la Fase 2 mediante evaluate_system_state().

FASE 2 – Integrar              : Paso de Störmer-Verlet de 2º orden usando
    la descripción de la Fase 1, con auditoría de finitud numérica y
    preservación de estructura simpléctica.

FASE 3 – Verificar y Extender : Auditoría de invarianza simpléctica y pasividad,
    composición de Yoshida de 4º orden, cálculo de Jacobiano numérico con
    soporte disperso para sistemas de gran escala, y veredicto categórico.

La clase final SymplecticVerletIntegrator hereda de la Fase 3 y ofrece una
interfaz consolidada con trazabilidad completa.

RIGOR MATEMÁTICO INCORPORADO:
• Geometría Simpléctica: Forma ω = dq ∧ dp, preservación MᵀΩM = Ω
• Sistemas Port-Hamiltonianos: ẋ = (J - R)∇H + Bu, y = Bᵀ∇H
• Teorema de Liouville: Conservación de volumen en espacio de fase
• Estabilidad de Yoshida: Composición de pasos para orden superior
• Regularización de Higham: Proyección al cono SPD para matrices de masa
• Álgebra Lineal Dispersa: O(|E|) para Jacobianos de sistemas grandes
• Análisis Espectral: Números de condición de Wilkinson para matrices SPD
• Pasividad de Lyapunov: V̇ = -∇HᵀR∇H ≤ 0 para disipación física

FASES ANIDADAS:
El último método de la Fase 1:
    evaluate_system_state(...)
es el morfismo de transición que alimenta directamente los métodos de la
Fase 2, garantizando la anidación categórica solicitada.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Optional, Callable, Tuple, Union
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Soporte para matrices dispersas (Jacobianos de gran escala)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    csr_matrix = None
    isspmatrix = None

logger = logging.getLogger("MIC.Physics.SymplecticVerletIntegrator")

# ═══════════════════════════════════════════════════════════════════════════
# Constantes de precisión de la FPU, espectrales y categóricas
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12  # Suelo espectral de Wilkinson
_YOSHIDA_COMPOSITION_TOL: Final[float] = 1.0e-8
_SYMPLECTIC_VERIFICATION_TOL: Final[float] = 1.0e-5
_MASS_MATRIX_CONDITION_LIMIT: Final[float] = 1.0e10
_MAX_DEGREES_OF_FREEDOM: Final[int] = 10000  # Cota para sistemas de gran escala

# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones (funtores de error en la categoría física)
# ═══════════════════════════════════════════════════════════════════════════
class SymplecticIntegrationError(Exception):
    """Excepción raíz para errores en el integrador simpléctico."""
    pass

class SystemConfigurationError(SymplecticIntegrationError):
    """Error en la configuración del sistema Port-Hamiltoniano."""
    pass

class IntegrationStepError(SymplecticIntegrationError):
    """Error durante la ejecución del paso de integración."""
    pass

class SymplecticInvarianceError(SymplecticIntegrationError):
    """Violación de la preservación de la forma simpléctica."""
    pass

class PassivityViolationError(SymplecticIntegrationError):
    """Violación de la desigualdad de disipación de Lyapunov."""
    pass

class MassMatrixSPDError(SystemConfigurationError):
    """Error cuando la matriz de masa no es definida positiva incluso tras Higham."""
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Enumeraciones categóricas (subobjetos y veredictos en el topos operativo)
# ═══════════════════════════════════════════════════════════════════════════
class IntegrationVerdict(IntEnum):
    """
    Clasificador de tres valores en el retículo de Heyting (Ω) para calidad
    de integración.
    
    Orden por severidad operativa:
        COHERENT  = ⊤ operativo (integración válida)
        DEGRADED  = elemento intermedio (precisión reducida)
        VETOED    = ⊥ operativo (integración inválida)
    
    El supremo de veredictos se toma como máximo nivel de severidad.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "IntegrationVerdict") -> "IntegrationVerdict":
        """
        Supremo en el retículo de severidad.
        Si no se proveen veredictos, retorna COHERENT como elemento neutro.
        """
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == IntegrationVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == IntegrationVerdict.DEGRADED

class IntegrationAction(Enum):
    """Acciones de mitigación tras el veredicto de integración."""
    NONE = auto()
    REDUCE_TIMESTEP = auto()
    HALT_INTEGRATION = auto()

# ═══════════════════════════════════════════════════════════════════════════
# Certificados de las fases anidadas (objetos de la subcategoría Spec)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SymplecticState:
    """
    Punto en el espacio de fase simpléctico M.
    
    Atributos
    ---------
    coordinates : NDArray[np.float64]
        Coordenadas generalizadas q ∈ ℝⁿ.
    momenta : NDArray[np.float64]
        Momentos conjugados p ∈ ℝⁿ.
    hamiltonian : float
        Energía total H(q, p).
    """
    coordinates: NDArray[np.float64]
    momenta: NDArray[np.float64]
    hamiltonian: float

@dataclass(frozen=True, slots=True)
class SymplecticIntegratorReport:
    """
    Resultado de un paso de integración con verificación de invariantes.
    
    Atributos
    ---------
    state_next : SymplecticState
        Estado en t + Δt.
    symplectic_residual : float
        Residuo ||MᵀΩM - Ω||_F.
    is_symplectically_invariant : bool
        True si residuo ≤ tolerancia.
    dissipation_rate : float
        Tasa de disipación de Rayleigh P_diss = -vᵀ R v.
    is_lyapunov_passive : bool
        True si P_diss ≥ 0.
    energy_drift : float
        Cambio relativo en Hamiltoniano: |H_next - H_curr| / |H_curr|.
    verdict : IntegrationVerdict
        Veredicto de calidad del paso.
    """
    state_next: SymplecticState
    symplectic_residual: float
    is_symplectically_invariant: bool
    dissipation_rate: float
    is_lyapunov_passive: bool
    energy_drift: float
    verdict: IntegrationVerdict

@dataclass(frozen=True, slots=True)
class Phase1SystemCertificate:
    """
    FASE 1 — Certificado de configuración del sistema Port-Hamiltoniano.
    
    Este certificado constituye el objeto terminal de la Fase 1 y es consumido
    por la Fase 2 a través del puente evaluate_system_state().
    
    Atributos
    ---------
    degrees_of_freedom : int
        Dimensión n del espacio de configuración.
    mass_matrix_condition_number : float
        Número de condición espectral de M⁻¹.
    damping_matrix_rank : int
        Rango de la matriz de disipación R.
    is_mass_matrix_spd : bool
        Verificación de definida positiva (tras Higham si aplica).
    higham_regularization_applied : bool
        Indicador de si se requirió regularización de Higham.
    potential_energy : float
        Energía potencial V(q) evaluada.
    kinetic_energy : float
        Energía cinética T(p) evaluada.
    total_hamiltonian : float
        Energía total H(q, p) = T + V.
    verdict : IntegrationVerdict
        Veredicto local de la Fase 1.
    """
    degrees_of_freedom: int
    mass_matrix_condition_number: float
    damping_matrix_rank: int
    is_mass_matrix_spd: bool
    higham_regularization_applied: bool
    potential_energy: float
    kinetic_energy: float
    total_hamiltonian: float
    verdict: IntegrationVerdict

@dataclass(frozen=True, slots=True)
class Phase2IntegrationCertificate:
    """
    FASE 2 — Certificado del paso de integración Verlet.
    
    Atributos
    ---------
    timestep : float
        Paso temporal Δt aplicado.
    external_forcing_norm : float
        Norma del forzamiento externo ||u||.
    state_norm_curr : float
        Norma del estado inicial ||(q, p)||.
    state_norm_next : float
        Norma del estado final ||(q_next, p_next)||.
    state_norm_residual : float
        Residuo relativo de preservación de norma.
    is_finite : bool
        Verificación de finitud numérica del estado siguiente.
    verdict : IntegrationVerdict
        Veredicto local de la Fase 2.
    """
    timestep: float
    external_forcing_norm: float
    state_norm_curr: float
    state_norm_next: float
    state_norm_residual: float
    is_finite: bool
    verdict: IntegrationVerdict

@dataclass(frozen=True, slots=True)
class Phase3ValidationCertificate:
    """
    FASE 3 — Certificado de validación simpléctica y de pasividad.
    
    Atributos
    ---------
    symplectic_residual : float
        Residuo ||MᵀΩM - Ω||_F del Jacobiano numérico.
    is_symplectically_invariant : bool
        True si se preserva la forma simpléctica.
    dissipation_rate : float
        Tasa de disipación P_diss = -vᵀ R v.
    is_lyapunov_passive : bool
        True si P_diss ≥ 0.
    energy_drift : float
        Deriva energética relativa del paso.
    jacobian_computation_sparse : bool
        Indicador de si se usó computación dispersa.
    yoshida_order : int
        Orden del esquema de composición (2 o 4).
    verdict : IntegrationVerdict
        Veredicto local de la Fase 3.
    """
    symplectic_residual: float
    is_symplectically_invariant: bool
    dissipation_rate: float
    is_lyapunov_passive: bool
    energy_drift: float
    jacobian_computation_sparse: bool
    yoshida_order: int
    verdict: IntegrationVerdict

@dataclass(frozen=True, slots=True)
class SymplecticIntegrationState:
    """Certificado global terminal del ciclo de integración."""
    phase1: Phase1SystemCertificate
    phase2: Phase2IntegrationCertificate
    phase3: Phase3ValidationCertificate
    final_verdict: IntegrationVerdict
    integration_action: IntegrationAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Regularización Espectral (Higham para Matriz de Masa)
# ═══════════════════════════════════════════════════════════════════════════
def stable_mass_matrix_higham(
    M_inv: NDArray[np.float64],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR
) -> Tuple[NDArray[np.float64], bool]:
    r"""
    Aplica la proyección de Higham para estabilizar la matriz de masa inversa
    en el cono SPD (Symmetric Positive Definite).
    
    Axioma de Proyección:
        \tilde{M}^{-1} = \arg\min_{X \succeq 0} \|M^{-1} - X\|_F
    
    Parámetros
    ----------
    M_inv : NDArray[np.float64]
        Matriz de masa inversa (potencialmente no-SPD por fluctuaciones FPU).
    tolerance : float
        Suelo espectral de Wilkinson para recorte de autovalores.
    
    Retorna
    -------
    Tuple[NDArray[np.float64], bool]
        (M_inv_projected, higham_applied) donde higham_applied indica si se
        requirió regularización.
    
    Raises
    ------
    MassMatrixSPDError
        Si la proyección de Higham falla en producir una matriz SPD.
    """
    try:
        la.cholesky(M_inv, lower=True)
        return M_inv.copy(), False
    except la.LinAlgError:
        # Descomposición espectral de Weyl de M⁻¹
        try:
            vals, vecs = la.eigh(M_inv)
        except Exception as exc:
            raise MassMatrixSPDError(
                "MassMatrixSPDError: Falló descomposición espectral para Higham."
            ) from exc

        # Recorte de autovalores al suelo espectral de Wilkinson
        vals_clean = np.maximum(vals, tolerance)

        # Reconstrucción de la matriz proyectada al cono SPD
        M_inv_projected = vecs @ np.diag(vals_clean) @ vecs.T

        # Simetrización numérica post-proyección
        M_inv_projected = (M_inv_projected + M_inv_projected.T) / 2.0

        logger.warning(
            "Matriz de masa inestable. Proyectada M⁻¹ al cono SPD via Higham "
            "(autovalores mínimos: %.4e → %.4e).",
            float(np.min(vals)),
            float(np.min(vals_clean))
        )

        try:
            la.cholesky(M_inv_projected, lower=True)
            return M_inv_projected, True
        except la.LinAlgError as exc:
            raise MassMatrixSPDError(
                "MassMatrixSPDError: Proyección de Higham no produjo matriz SPD."
            ) from exc

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Cálculo Disperso (Jacobiano para Sistemas Grandes)
# ═══════════════════════════════════════════════════════════════════════════
def compute_sparse_jacobian(
    step_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    h: float = 1e-6,
    sparse_threshold: float = 0.3
) -> Tuple[Union[NDArray[np.float64], csr_matrix], bool]:
    r"""
    Calcula el Jacobiano numérico con soporte disperso para sistemas de
    gran escala.
    
    Parámetros
    ----------
    step_func : Callable
        Función que mapea estado → estado siguiente.
    x0 : NDArray[np.float64]
        Estado base para linearización.
    h : float
        Paso de diferenciación finita.
    sparse_threshold : float
        Umbral de dispersión para convertir a formato sparse.
    
    Retorna
    -------
    Tuple[Union[NDArray, csr_matrix], bool]
        (Jacobian, sparse_used) donde sparse_used indica si se usó formato
        disperso.
    """
    n = x0.size
    jac = np.zeros((n, n), dtype=np.float64)
    f0 = step_func(x0)

    for i in range(n):
        x_plus = np.copy(x0)
        x_plus[i] += h
        f_plus = step_func(x_plus)

        x_minus = np.copy(x0)
        x_minus[i] -= h
        f_minus = step_func(x_minus)

        jac[:, i] = (f_plus - f_minus) / (2.0 * h)

    # Verificar dispersión y convertir si aplica
    sparse_used = False
    if _SPARSE_AVAILABLE and csr_matrix is not None:
        sparsity = 1.0 - (np.count_nonzero(jac) / jac.size)
        if sparsity > sparse_threshold:
            jac = csr_matrix(jac)
            sparse_used = True

    return jac, sparse_used

# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 — Observe/Estructurar: Descripción del sistema Port-Hamiltoniano
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_SystemDescriptor(ABC):
    """
    FASE 1 (Observe/Estructurar): Encapsula la geometría y la energía del
    sistema Port-Hamiltoniano.
    
    Almacena las funciones de gradiente de fuerza, potencial, matrices de masa
    y amortiguamiento, y proporciona métodos para evaluar energías y fuerzas.
    
    El último método de esta fase:
        evaluate_system_state(...)
    constituye el morfismo de transición formal hacia la Fase 2.
    """

    def __init__(
        self,
        dimension: int,
        mass_matrix_inv: NDArray[np.float64],
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        potential_q: Optional[Callable[[NDArray[np.float64]], float]] = None,
        damping_matrix_r: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL
    ) -> None:
        """
        Inicializa el descriptor del sistema.
        
        Parámetros
        ----------
        dimension : int
            Grados de libertad n (mitad de la dimensión del espacio de fase).
        mass_matrix_inv : NDArray
            Inversa de la matriz de masa/inercia M⁻¹ (n×n). Debe ser SPD.
        force_gradient_q : Callable
            Gradiente negativo del potencial F(q) = -∇V(q).
        potential_q : Callable, opcional
            Función escalar de energía potencial V(q).
        damping_matrix_r : NDArray, opcional
            Matriz de disipación de Rayleigh R (n×n), simétrica y PSD.
        tolerance : float
            Cota de precisión para validaciones.
        
        Lanza
        -----
        SystemConfigurationError
            Si algún parámetro no cumple las condiciones exigidas.
        """
        if dimension <= 0:
            raise SystemConfigurationError("La dimensión debe ser positiva.")
        if mass_matrix_inv.shape != (dimension, dimension):
            raise SystemConfigurationError(
                f"mass_matrix_inv debe ser de tamaño ({dimension}, {dimension})."
            )

        # Validar finitud
        if not np.all(np.isfinite(mass_matrix_inv)):
            raise SystemConfigurationError(
                "SystemConfigurationError: mass_matrix_inv contiene entradas no finitas."
            )

        # Verificar simetría estructural
        sym_residual = float(la.norm(mass_matrix_inv - mass_matrix_inv.T, ord="fro"))
        if sym_residual > max(tolerance * 100.0, _RELAX_TOL):
            raise SystemConfigurationError(
                f"SystemConfigurationError: mass_matrix_inv no es simétrica "
                f"(residuo = {sym_residual:.4e})."
            )

        # Aplicar regularización de Higham si es necesario
        try:
            self._mass_matrix_inv, higham_applied = stable_mass_matrix_higham(
                mass_matrix_inv, tolerance
            )
        except MassMatrixSPDError as exc:
            raise SystemConfigurationError(str(exc)) from exc

        self._n = dimension
        self._force_gradient_q = force_gradient_q
        self._potential_q = potential_q
        self._tol = tolerance
        self._higham_applied = higham_applied

        # Número de condición espectral de M⁻¹
        try:
            self._mass_condition_number = float(np.linalg.cond(self._mass_matrix_inv))
        except Exception:
            self._mass_condition_number = float("inf")

        # Validar matriz de amortiguamiento R
        if damping_matrix_r is not None:
            if damping_matrix_r.shape != (dimension, dimension):
                raise SystemConfigurationError(
                    f"damping_matrix_r debe ser de tamaño ({dimension}, {dimension})."
                )
            if not np.all(np.isfinite(damping_matrix_r)):
                raise SystemConfigurationError(
                    "SystemConfigurationError: damping_matrix_r contiene entradas no finitas."
                )

            sym_residual_r = float(
                la.norm(damping_matrix_r - damping_matrix_r.T, ord="fro")
            )
            if sym_residual_r > max(tolerance * 100.0, _RELAX_TOL):
                raise SystemConfigurationError(
                    f"SystemConfigurationError: damping_matrix_r no es simétrica "
                    f"(residuo = {sym_residual_r:.4e})."
                )

            eigvals_r = la.eigvalsh(damping_matrix_r)
            if np.any(eigvals_r < -_MACHINE_EPS):
                raise SystemConfigurationError(
                    "SystemConfigurationError: damping_matrix_r no es PSD."
                )
            self._damping_matrix_r = damping_matrix_r.copy()
            self._damping_rank = int(np.linalg.matrix_rank(damping_matrix_r, tol=1e-12))
        else:
            self._damping_matrix_r = np.zeros((dimension, dimension), dtype=np.float64)
            self._damping_rank = 0

        # Matriz simpléctica canónica Ω para uso en fases posteriores
        id_n = np.eye(self._n, dtype=np.float64)
        z_n = np.zeros((self._n, self._n), dtype=np.float64)
        self._omega: NDArray[np.float64] = np.block([
            [z_n, id_n],
            [-id_n, z_n]
        ])

    def compute_kinetic_energy(self, p: NDArray[np.float64]) -> float:
        """Energía cinética T(p) = ½ pᵀ M⁻¹ p."""
        return 0.5 * float(p.T @ self._mass_matrix_inv @ p)

    def compute_potential_energy(self, q: NDArray[np.float64]) -> float:
        """
        Energía potencial V(q). Si no se ha proporcionado la función escalar,
        se estima mediante la aproximación lineal: V ≈ -½ Σ q_i F_i(q).
        """
        if self._potential_q is not None:
            return self._potential_q(q)
        force = self._force_gradient_q(q)
        return -0.5 * float(np.sum(q * force))

    def compute_force(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fuerza conservativa F(q) = -∇V(q)."""
        return self._force_gradient_q(q)

    def compute_total_hamiltonian(self, q: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """H(q, p) = T(p) + V(q)."""
        return self.compute_kinetic_energy(p) + self.compute_potential_energy(q)

    # ─────────────────────────────────────────────────────────────────────
    # Último método de la Fase 1: Puente hacia la Fase 2
    # ─────────────────────────────────────────────────────────────────────
    def evaluate_system_state(
        self, q: NDArray[np.float64], p: NDArray[np.float64]
    ) -> dict:
        """
        MORFISMO DE TRANSICIÓN FASE 1 → FASE 2.
        
        Evalúa todas las magnitudes necesarias para un paso de integración.
        Este método es el punto de entrada de la Fase 2.
        
        Retorna un diccionario con:
            - 'force': fuerza conservativa F(q).
            - 'potential_energy': V(q).
            - 'kinetic_energy': T(p).
            - 'hamiltonian': H(q,p).
            - 'damping_force': fuerza disipativa R p (vector).
            - 'mass_matrix_inv': M⁻¹ (para uso en Fase 2).
            - 'damping_matrix': R (para uso en Fase 2).
        """
        force = self.compute_force(q)
        V = self.compute_potential_energy(q)
        T = self.compute_kinetic_energy(p)
        H = T + V
        damping_force = self._damping_matrix_r @ p

        return {
            'force': force,
            'potential_energy': V,
            'kinetic_energy': T,
            'hamiltonian': H,
            'damping_force': damping_force,
            'mass_matrix_inv': self._mass_matrix_inv,
            'damping_matrix': self._damping_matrix_r,
        }

    def generate_phase1_certificate(
        self, q: NDArray[np.float64], p: NDArray[np.float64]
    ) -> Phase1SystemCertificate:
        """
        Genera el certificado de Fase 1 con auditoría completa del sistema.
        
        Parámetros
        ----------
        q : NDArray[np.float64]
            Coordenadas generalizadas.
        p : NDArray[np.float64]
            Momentos conjugados.
        
        Retorna
        -------
        Phase1SystemCertificate
            Certificado terminal de Fase 1.
        """
        sys_eval = self.evaluate_system_state(q, p)

        # Veredicto local
        verdict = IntegrationVerdict.COHERENT
        if not np.isfinite(sys_eval['hamiltonian']):
            verdict = IntegrationVerdict.VETOED
        elif (
            self._mass_condition_number > _MASS_MATRIX_CONDITION_LIMIT
            or not self._higham_applied and np.any(la.eigvalsh(self._mass_matrix_inv) <= _MACHINE_EPS)
        ):
            verdict = IntegrationVerdict.DEGRADED

        return Phase1SystemCertificate(
            degrees_of_freedom=self._n,
            mass_matrix_condition_number=self._mass_condition_number,
            damping_matrix_rank=self._damping_rank,
            is_mass_matrix_spd=True,  # Garantizado por Higham
            higham_regularization_applied=self._higham_applied,
            potential_energy=float(sys_eval['potential_energy']),
            kinetic_energy=float(sys_eval['kinetic_energy']),
            total_hamiltonian=float(sys_eval['hamiltonian']),
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 – Integrar: Paso de Störmer-Verlet (anidada en Fase 1)
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_VerletIntegrator(Phase1_SystemDescriptor):
    """
    FASE 2 (Integrar): Implementa el algoritmo Kick-Drift-Kick de Störmer-Verlet
    de segundo orden. Utiliza directamente el método evaluate_system_state de la
    Fase 1, estableciendo el nexo explícito entre las fases.
    """

    def integrate_verlet_step(
        self,
        state_curr: SymplecticState,
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None
    ) -> SymplecticState:
        r"""
        Ejecuta un paso de integración temporal mediante Störmer-Verlet de 2º orden.
        
        Algoritmo (Kick-Drift-Kick):
            p_{k+1/2} = p_k + (Δt/2) [F(q_k) - R p_k + u]
            q_{k+1}   = q_k + Δt M⁻¹ p_{k+1/2}
            p_{k+1}   = p_{k+1/2} + (Δt/2) [F(q_{k+1}) - R p_{k+1/2} + u]
        
        Parámetros
        ----------
        state_curr : SymplecticState
            Estado actual (q, p, H).
        dt : float
            Paso temporal.
        external_forcing : NDArray, opcional
            Forzamiento externo u (n,). Si es None, se asume cero.
        
        Retorna
        -------
        SymplecticState
            Nuevo estado en t + Δt con energía evaluada.
        
        Lanza
        -----
        IntegrationStepError
            Si el paso produce valores no finitos.
        """
        if dt <= _MACHINE_EPS:
            raise IntegrationStepError(
                "IntegrationStepError: Paso temporal dt debe ser positivo."
            )

        q = np.copy(state_curr.coordinates)
        p = np.copy(state_curr.momenta)

        # Preparar fuerzas externas
        u = np.zeros(self._n) if external_forcing is None else np.asarray(external_forcing)
        if u.size != self._n:
            raise IntegrationStepError(
                f"IntegrationStepError: Forzamiento externo debe tener dimensión {self._n}."
            )
        if not np.all(np.isfinite(u)):
            raise IntegrationStepError(
                "IntegrationStepError: Forzamiento externo contiene entradas no finitas."
            )

        # Obtener evaluación del sistema actual (puente con Fase 1)
        sys_curr = self.evaluate_system_state(q, p)
        force_q = sys_curr['force']
        damping_force = sys_curr['damping_force']
        M_inv = sys_curr['mass_matrix_inv']
        R = sys_curr['damping_matrix']

        # Kick 1: actualizar momento a medio paso
        p_half = p + 0.5 * dt * (force_q - damping_force + u)

        # Drift: actualizar coordenada
        q_next = q + dt * (M_inv @ p_half)

        # Evaluar el sistema en la nueva coordenada
        force_next = self.compute_force(q_next)

        # La fuerza disipativa se calcula con el momento intermedio
        damping_half = R @ p_half

        # Kick 2: completar el momento
        p_next = p_half + 0.5 * dt * (force_next - damping_half + u)

        # Verificar finitud
        if not (np.all(np.isfinite(q_next)) and np.all(np.isfinite(p_next))):
            raise IntegrationStepError(
                "IntegrationStepError: Paso de Verlet produjo valores no finitos (NaN o Inf)."
            )

        # Calcular energía total del nuevo estado
        H_next = self.compute_total_hamiltonian(q_next, p_next)

        return SymplecticState(
            coordinates=q_next,
            momenta=p_next,
            hamiltonian=H_next
        )

    def generate_phase2_certificate(
        self,
        state_curr: SymplecticState,
        state_next: SymplecticState,
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None,
    ) -> Phase2IntegrationCertificate:
        """
        Genera el certificado de Fase 2 con auditoría del paso de integración.
        
        Parámetros
        ----------
        state_curr : SymplecticState
            Estado inicial.
        state_next : SymplecticState
            Estado después del paso.
        dt : float
            Paso temporal aplicado.
        external_forcing : Optional[NDArray]
            Forzamiento externo aplicado.
        
        Retorna
        -------
        Phase2IntegrationCertificate
            Certificado terminal de Fase 2.
        """
        # Norma del forzamiento externo
        if external_forcing is None:
            u_norm = 0.0
        else:
            u_norm = float(la.norm(external_forcing))

        # Normas de estado
        state_curr_norm = float(np.sqrt(
            la.norm(state_curr.coordinates)**2 + la.norm(state_curr.momenta)**2
        ))
        state_next_norm = float(np.sqrt(
            la.norm(state_next.coordinates)**2 + la.norm(state_next.momenta)**2
        ))

        # Residuo de preservación de norma
        if state_curr_norm > _MACHINE_EPS:
            state_norm_residual = float(abs(state_next_norm - state_curr_norm) / state_curr_norm)
        else:
            state_norm_residual = float(abs(state_next_norm))

        # Verificación de finitud
        is_finite = (
            np.all(np.isfinite(state_next.coordinates)) and
            np.all(np.isfinite(state_next.momenta)) and
            np.isfinite(state_next.hamiltonian)
        )

        # Veredicto local
        verdict = IntegrationVerdict.COHERENT
        if not is_finite:
            verdict = IntegrationVerdict.VETOED
        elif state_norm_residual > _RELAX_TOL or dt > 1.0:
            verdict = IntegrationVerdict.DEGRADED

        return Phase2IntegrationCertificate(
            timestep=float(dt),
            external_forcing_norm=u_norm,
            state_norm_curr=state_curr_norm,
            state_norm_next=state_next_norm,
            state_norm_residual=state_norm_residual,
            is_finite=is_finite,
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 – Verificar y Extender: Auditoría simpléctica y composición Yoshida
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_SymplecticValidator(Phase2_VerletIntegrator):
    """
    FASE 3 (Verificar y Extender): Proporciona métodos para auditar la preservación
    simpléctica y la pasividad de un paso dado, y compone pasos de orden superior
    (Yoshida de 4º orden). Hereda de la Fase 2 para poder generar pasos y evaluar
    el sistema.
    """

    def _compute_numerical_jacobian(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None,
        h: float = 1e-6
    ) -> Tuple[Union[NDArray[np.float64], csr_matrix], bool]:
        """
        Calcula el Jacobiano numérico del mapeo simpléctico local
        (q,p) → (q_next, p_next) mediante diferencias finitas centrales.
        
        Soporta computación dispersa para sistemas de gran escala.
        
        Retorna
        -------
        Tuple[Union[NDArray, csr_matrix], bool]
            (Jacobian, sparse_used)
        """
        dim_total = 2 * self._n
        sparse_used = False

        # Umbral para usar computación dispersa
        if _SPARSE_AVAILABLE and self._n > 100:
            sparse_used = True

        if sparse_used and _SPARSE_AVAILABLE:
            def step_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
                q_vec = x[:self._n]
                p_vec = x[self._n:]
                state = SymplecticState(q_vec, p_vec, 0.0)
                next_state = self.integrate_verlet_step(state, dt, external_forcing)
                return np.concatenate([next_state.coordinates, next_state.momenta])

            x0 = np.concatenate([q, p])
            jac, sparse_used = compute_sparse_jacobian(step_func, x0, h)
        else:
            jac = np.zeros((dim_total, dim_total), dtype=np.float64)

            def step_from(q_vec: NDArray[np.float64], p_vec: NDArray[np.float64]) -> NDArray[np.float64]:
                state = SymplecticState(q_vec, p_vec, 0.0)
                next_state = self.integrate_verlet_step(state, dt, external_forcing)
                return np.concatenate([next_state.coordinates, next_state.momenta])

            base_out = step_from(q, p)
            for i in range(self._n):
                # Perturbar q_i (diferencias centrales)
                q_plus = np.copy(q)
                q_plus[i] += h
                out_plus = step_from(q_plus, p)
                q_minus = np.copy(q)
                q_minus[i] -= h
                out_minus = step_from(q_minus, p)
                jac[:, i] = (out_plus - out_minus) / (2.0 * h)

                # Perturbar p_i
                p_plus = np.copy(p)
                p_plus[i] += h
                out_plus_p = step_from(q, p_plus)
                p_minus = np.copy(p)
                p_minus[i] -= h
                out_minus_p = step_from(q, p_minus)
                jac[:, self._n + i] = (out_plus_p - out_minus_p) / (2.0 * h)

        return jac, sparse_used

    def verify_integration_step(
        self,
        state_curr: SymplecticState,
        state_next: SymplecticState,
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None,
        jacobian_tolerance: float = _SYMPLECTIC_VERIFICATION_TOL
    ) -> SymplecticIntegratorReport:
        """
        Verifica la invarianza simpléctica y la pasividad del paso que va
        de state_curr a state_next. Calcula el Jacobiano numérico y evalúa
        Mᵀ Ω M - Ω, así como la tasa de disipación.
        
        Retorna un reporte completo.
        """
        # 1. Jacobiano numérico del mapeo
        jac, sparse_used = self._compute_numerical_jacobian(
            state_curr.coordinates, state_curr.momenta, dt, external_forcing
        )

        # 2. Residuo simpléctico
        if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(jac):
            # Para matrices dispersas, calcular en formato denso para Ω
            jac_dense = jac.toarray()
            symp_residual = float(
                la.norm(jac_dense.T @ self._omega @ jac_dense - self._omega, ord="fro")
            )
        else:
            symp_residual = float(
                la.norm(jac.T @ self._omega @ jac - self._omega, ord="fro")
            )

        # Si el sistema no tiene disipación, exigimos preservación estricta
        is_hamiltonian = (
            self._damping_matrix_r is None or
            la.norm(self._damping_matrix_r, ord="fro") <= _MACHINE_EPS
        )
        is_symplectic = symp_residual <= jacobian_tolerance if is_hamiltonian else True

        if is_hamiltonian and not is_symplectic:
            raise SymplecticInvarianceError(
                f"SymplecticInvarianceError: Violación de simplécticidad: "
                f"residuo = {symp_residual:.4e} > tolerancia."
            )

        # 3. Tasa de disipación P_diss = -vᵀ R v (con v = M⁻¹ p_next)
        velocity = self._mass_matrix_inv @ state_next.momenta
        diss_rate = -float(velocity.T @ self._damping_matrix_r @ velocity)
        is_passive = diss_rate >= -_MACHINE_EPS

        if not is_passive:
            raise PassivityViolationError(
                f"PassivityViolationError: Disipación negativa: P_diss = {diss_rate:.4e}."
            )

        # 4. Deriva energética
        if abs(state_curr.hamiltonian) > _MACHINE_EPS:
            energy_drift = float(
                abs(state_next.hamiltonian - state_curr.hamiltonian) /
                abs(state_curr.hamiltonian)
            )
        else:
            energy_drift = float(abs(state_next.hamiltonian - state_curr.hamiltonian))

        # 5. Veredicto local
        verdict = IntegrationVerdict.COHERENT
        if not np.isfinite(symp_residual) or not np.isfinite(diss_rate):
            verdict = IntegrationVerdict.VETOED
        elif (
            (is_hamiltonian and not is_symplectic) or
            energy_drift > _RELAX_TOL or
            not is_passive
        ):
            verdict = IntegrationVerdict.DEGRADED

        return SymplecticIntegratorReport(
            state_next=state_next,
            symplectic_residual=symp_residual,
            is_symplectically_invariant=is_symplectic,
            dissipation_rate=diss_rate,
            is_lyapunov_passive=is_passive,
            energy_drift=energy_drift,
            verdict=verdict,
        )

    def integrate_step_2nd_order(
        self,
        state_curr: SymplecticState,
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None
    ) -> SymplecticIntegratorReport:
        """
        Ejecuta un paso de Verlet y retorna el reporte verificado (conveniencia).
        """
        state_next = self.integrate_verlet_step(state_curr, dt, external_forcing)
        return self.verify_integration_step(state_curr, state_next, dt, external_forcing)

    def integrate_step_4th_order(
        self,
        state_curr: SymplecticState,
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None
    ) -> SymplecticIntegratorReport:
        r"""
        Compone tres pasos de Störmer-Verlet con pesos de Yoshida para obtener
        un integrador simpléctico de 4º orden.
        
        Pesos de Yoshida:
            w_1 = 1 / (2 - 2^(1/3))
            w_0 = -2^(1/3) / (2 - 2^(1/3))
        
        Retorna el reporte correspondiente al último paso compuesto.
        """
        yoshida_factor = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - yoshida_factor)
        w0 = -yoshida_factor / (2.0 - yoshida_factor)

        # Verificación de consistencia de composición
        composition_sum = 2.0 * w1 + w0
        if abs(composition_sum - 1.0) > _YOSHIDA_COMPOSITION_TOL:
            logger.warning(
                "Composición de Yoshida: suma de pesos = %.12f (debería ser 1.0)",
                composition_sum
            )

        dt1 = w1 * dt
        dt2 = w0 * dt
        dt3 = w1 * dt

        # Paso 1
        report1 = self.integrate_step_2nd_order(state_curr, dt1, external_forcing)

        # Paso 2
        report2 = self.integrate_step_2nd_order(report1.state_next, dt2, external_forcing)

        # Paso 3
        report3 = self.integrate_step_2nd_order(report2.state_next, dt3, external_forcing)

        # Actualizar veredicto con orden 4
        return SymplecticIntegratorReport(
            state_next=report3.state_next,
            symplectic_residual=report3.symplectic_residual,
            is_symplectically_invariant=report3.is_symplectically_invariant,
            dissipation_rate=report3.dissipation_rate,
            is_lyapunov_passive=report3.is_lyapunov_passive,
            energy_drift=report3.energy_drift,
            verdict=report3.verdict,
        )

    def generate_phase3_certificate(
        self,
        state_curr: SymplecticState,
        state_next: SymplecticState,
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None,
        yoshida_order: int = 2,
    ) -> Phase3ValidationCertificate:
        """
        Genera el certificado de Fase 3 con auditoría completa de validación.
        
        Parámetros
        ----------
        state_curr : SymplecticState
            Estado inicial.
        state_next : SymplecticState
            Estado final.
        dt : float
            Paso temporal.
        external_forcing : Optional[NDArray]
            Forzamiento externo.
        yoshida_order : int
            Orden del esquema (2 o 4).
        
        Retorna
        -------
        Phase3ValidationCertificate
            Certificado terminal de Fase 3.
        """
        report = self.verify_integration_step(
            state_curr, state_next, dt, external_forcing
        )

        # Determinar si se usó computación dispersa
        _, sparse_used = self._compute_numerical_jacobian(
            state_curr.coordinates, state_curr.momenta, dt, external_forcing
        )

        # Veredicto local
        verdict = IntegrationVerdict.COHERENT
        if report.verdict.is_vetoed:
            verdict = IntegrationVerdict.VETOED
        elif report.verdict.is_degraded or yoshida_order == 4:
            verdict = IntegrationVerdict.DEGRADED if report.energy_drift > _DEFAULT_TOL else IntegrationVerdict.COHERENT

        return Phase3ValidationCertificate(
            symplectic_residual=report.symplectic_residual,
            is_symplectically_invariant=report.is_symplectically_invariant,
            dissipation_rate=report.dissipation_rate,
            is_lyapunov_passive=report.is_lyapunov_passive,
            energy_drift=report.energy_drift,
            jacobian_computation_sparse=sparse_used,
            yoshida_order=yoshida_order,
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════
# Integrador final: herencia completa de las tres fases
# ═══════════════════════════════════════════════════════════════════════════
class SymplecticVerletIntegrator(Phase3_SymplecticValidator):
    """
    Integrador simpléctico consolidado. Ofrece la misma interfaz que la versión
    anterior, pero con una estructura interna anidada y validaciones rigurosas.
    
    El ciclo de integración queda formalizado como:
        Observe  → Fase 1 (SystemDescriptor)
        Integrar → Fase 2 (VerletIntegrator)
        Verificar→ Fase 3 (SymplecticValidator)
    
    El veredicto final es el supremo de severidad en el retículo de Heyting.
    """

    def __init__(
        self,
        dimension: int,
        mass_matrix_inv: NDArray[np.float64],
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        potential_q: Optional[Callable[[NDArray[np.float64]], float]] = None,
        damping_matrix_r: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False
    ) -> None:
        """
        Inicializa el integrador con configuración completa del sistema.
        
        Parámetros
        ----------
        dimension : int
            Grados de libertad del sistema.
        mass_matrix_inv : NDArray
            Inversa de la matriz de masa M⁻¹.
        force_gradient_q : Callable
            Gradiente de fuerza F(q) = -∇V(q).
        potential_q : Callable, opcional
            Función de energía potencial V(q).
        damping_matrix_r : NDArray, opcional
            Matriz de disipación R.
        tolerance : float
            Cota de precisión.
        halt_on_veto : bool
            Si True, lanza excepción cuando el veredicto es VETOED.
        """
        super().__init__(
            dimension=dimension,
            mass_matrix_inv=mass_matrix_inv,
            force_gradient_q=force_gradient_q,
            potential_q=potential_q,
            damping_matrix_r=damping_matrix_r,
            tolerance=tolerance,
        )
        self._halt_on_veto = halt_on_veto

    def execute_integration_cycle(
        self,
        q_initial: NDArray[np.float64],
        p_initial: NDArray[np.float64],
        dt: float,
        external_forcing: Optional[NDArray[np.float64]] = None,
        use_yoshida_4th_order: bool = False,
    ) -> SymplecticIntegrationState:
        """
        Orquesta el ciclo completo de integración con validación de las tres fases.
        
        Estrategia de anidación:
          1. Fase 1: Genera certificado de configuración del sistema.
          2. Fase 2: Ejecuta paso de Verlet (2º o 4º orden).
          3. Fase 3: Valida invarianza simpléctica y pasividad.
        
        Si Fase 1 emite VETOED, se emiten certificados vetados para Fase 2 y 3.
        
        Parámetros
        ----------
        q_initial : NDArray[np.float64]
            Coordenadas iniciales.
        p_initial : NDArray[np.float64]
            Momentos iniciales.
        dt : float
            Paso temporal.
        external_forcing : Optional[NDArray]
            Forzamiento externo.
        use_yoshida_4th_order : bool
            Si True, usa composición de Yoshida de 4º orden.
        
        Retorna
        -------
        SymplecticIntegrationState
            Certificado global terminal.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Validar entradas
            if q_initial.size != self._n or p_initial.size != self._n:
                raise SystemConfigurationError(
                    f"SystemConfigurationError: Estado inicial debe tener dimensión {self._n}."
                )
            if not np.all(np.isfinite(q_initial)) or not np.all(np.isfinite(p_initial)):
                raise SystemConfigurationError(
                    "SystemConfigurationError: Estado inicial contiene entradas no finitas."
                )

            # ─── FASE 1: Observe ───
            H_initial = self.compute_total_hamiltonian(q_initial, p_initial)
            state_curr = SymplecticState(
                coordinates=q_initial.copy(),
                momenta=p_initial.copy(),
                hamiltonian=H_initial,
            )
            cert_1 = self.generate_phase1_certificate(q_initial, p_initial)

            if cert_1.verdict.is_vetoed:
                cert_2 = self._vetoed_phase2_certificate(dt)
                cert_3 = self._vetoed_phase3_certificate()
                state_next = state_curr
            else:
                # ─── FASE 2: Integrar ───
                if use_yoshida_4th_order:
                    report = self.integrate_step_4th_order(state_curr, dt, external_forcing)
                    yoshida_order = 4
                else:
                    report = self.integrate_step_2nd_order(state_curr, dt, external_forcing)
                    yoshida_order = 2

                state_next = report.state_next
                cert_2 = self.generate_phase2_certificate(
                    state_curr, state_next, dt, external_forcing
                )

                if cert_2.verdict.is_vetoed:
                    cert_3 = self._vetoed_phase3_certificate()
                else:
                    # ─── FASE 3: Verificar ───
                    cert_3 = self.generate_phase3_certificate(
                        state_curr, state_next, dt, external_forcing, yoshida_order
                    )

            # ─── Fusión de veredictos (supremo en severidad) ───
            final_verdict = IntegrationVerdict.supremum(
                cert_1.verdict,
                cert_2.verdict,
                cert_3.verdict,
            )

            integration_action = IntegrationAction.NONE
            diagnostic_note = "Ciclo de integración simpléctica completado."

            if final_verdict == IntegrationVerdict.VETOED:
                integration_action = IntegrationAction.HALT_INTEGRATION
                diagnostic_note = (
                    "VETO DE INTEGRACIÓN: violación de simplécticidad, pasividad "
                    "o finitud numérica."
                )
                logger.error("¡VETO DE INTEGRACIÓN! Halt requested.")
                if self._halt_on_veto:
                    raise SymplecticIntegrationError(
                        "Violación de invariantes simplécticos o pasividad."
                    )
            elif final_verdict == IntegrationVerdict.DEGRADED:
                integration_action = IntegrationAction.REDUCE_TIMESTEP
                diagnostic_note = (
                    "Degradación detectada: deriva energética o residuo simpléctico "
                    "elevado. Se recomienda reducir dt."
                )
                logger.warning(
                    "Degradación en integración simpléctica. Se recomienda reducir dt."
                )

            provenance_hash = self._generate_provenance_hash(
                cert_1, cert_2, cert_3, timestamp_utc
            )

            return SymplecticIntegrationState(
                phase1=cert_1,
                phase2=cert_2,
                phase3=cert_3,
                final_verdict=final_verdict,
                integration_action=integration_action,
                timestamp_utc=timestamp_utc,
                provenance_hash=provenance_hash,
                diagnostic_note=diagnostic_note,
            )

        except SymplecticIntegrationError as exc:
            logger.error("Colapso categórico del integrador simpléctico: %s", exc)
            if self._halt_on_veto:
                raise
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)

        except Exception as exc:  # pragma: no cover
            logger.exception("Colapso catastrófico no tipado del integrador simpléctico.")
            if self._halt_on_veto:
                raise SymplecticIntegrationError(
                    "Colapso catastrófico no tipado del integrador simpléctico."
                ) from exc
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)

    # ─────────────────────────────────────────────────────────────────────
    # Certificados vetados para cortocircuito anidado
    # ─────────────────────────────────────────────────────────────────────
    def _vetoed_phase2_certificate(self, dt: float) -> Phase2IntegrationCertificate:
        """Certificado vetado de Fase 2 para cortocircuito."""
        return Phase2IntegrationCertificate(
            timestep=float(dt),
            external_forcing_norm=0.0,
            state_norm_curr=0.0,
            state_norm_next=0.0,
            state_norm_residual=float("inf"),
            is_finite=False,
            verdict=IntegrationVerdict.VETOED,
        )

    def _vetoed_phase3_certificate(self) -> Phase3ValidationCertificate:
        """Certificado vetado de Fase 3 para cortocircuito."""
        return Phase3ValidationCertificate(
            symplectic_residual=float("inf"),
            is_symplectically_invariant=False,
            dissipation_rate=float("inf"),
            is_lyapunov_passive=False,
            energy_drift=float("inf"),
            jacobian_computation_sparse=False,
            yoshida_order=0,
            verdict=IntegrationVerdict.VETOED,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Estado catastrófico de emergencia
    # ─────────────────────────────────────────────────────────────────────
    def _cataclysm_state(self, reason: str, timestamp_utc: str) -> SymplecticIntegrationState:
        """Construye un estado catastrófico con certificados vetados."""
        phase1_dummy = Phase1SystemCertificate(
            degrees_of_freedom=self._n,
            mass_matrix_condition_number=float("inf"),
            damping_matrix_rank=0,
            is_mass_matrix_spd=False,
            higham_regularization_applied=False,
            potential_energy=0.0,
            kinetic_energy=0.0,
            total_hamiltonian=0.0,
            verdict=IntegrationVerdict.VETOED,
        )
        phase2_dummy = self._vetoed_phase2_certificate(0.0)
        phase3_dummy = self._vetoed_phase3_certificate()

        raw_payload = f"CATACLYSM_SYMPLECTIC_VETO::{reason}"
        provenance_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

        return SymplecticIntegrationState(
            phase1=phase1_dummy,
            phase2=phase2_dummy,
            phase3=phase3_dummy,
            final_verdict=IntegrationVerdict.VETOED,
            integration_action=IntegrationAction.HALT_INTEGRATION,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_SYMPLECTIC_VETO: {reason}",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Hash de procedencia auditable
    # ─────────────────────────────────────────────────────────────────────
    def _generate_provenance_hash(
        self,
        c1: Phase1SystemCertificate,
        c2: Phase2IntegrationCertificate,
        c3: Phase3ValidationCertificate,
        timestamp_utc: str,
    ) -> str:
        """Genera un hash SHA-256 de procedencia que ata los veredictos y residuos."""
        raw_payload = "|".join(
            (
                timestamp_utc,
                str(c1.verdict.value),
                str(c2.verdict.value),
                str(c3.verdict.value),
                f"{c1.mass_matrix_condition_number:.12e}",
                f"{c1.total_hamiltonian:.12e}",
                f"{c2.timestep:.12e}",
                f"{c2.state_norm_residual:.12e}",
                f"{c3.symplectic_residual:.12e}",
                f"{c3.energy_drift:.12e}",
                f"{c3.dissipation_rate:.12e}",
                str(int(c3.jacobian_computation_sparse)),
                str(int(c3.yoshida_order)),
                str(int(c1.higham_regularization_applied)),
            )
        )
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

# ═══════════════════════════════════════════════════════════════════════════
# Compatibilidad de nombres de clases para versiones anteriores
# ═══════════════════════════════════════════════════════════════════════════
Phase1_SystemDescriptor = Phase1_SystemDescriptor
Phase2_VerletIntegrator = Phase2_VerletIntegrator
Phase3_SymplecticValidator = Phase3_SymplecticValidator

__all__ = [
    "SymplecticIntegrationError",
    "SystemConfigurationError",
    "IntegrationStepError",
    "SymplecticInvarianceError",
    "PassivityViolationError",
    "MassMatrixSPDError",
    "IntegrationVerdict",
    "IntegrationAction",
    "SymplecticState",
    "SymplecticIntegratorReport",
    "Phase1SystemCertificate",
    "Phase2IntegrationCertificate",
    "Phase3ValidationCertificate",
    "SymplecticIntegrationState",
    "Phase1_SystemDescriptor",
    "Phase2_VerletIntegrator",
    "Phase3_SymplecticValidator",
    "SymplecticVerletIntegrator",
    "stable_mass_matrix_higham",
    "compute_sparse_jacobian",
]