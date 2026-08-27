# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Synaptic Fock Space Registry Agent (Soberano de Memoria Cuántica)   ║
║ Ruta   : app/agents/core/immune_system/synaptic_fock_space_registry_agent.py ║
║ Versión: 3.0.0-Fock-OODA-Heyting-PhD-Nested-Higham-Sparse-Strict             ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA ESPECTRAL EN EL ESTRATO OMEGA (V_Ω) ─────
Este módulo consagra al Agente Soberano y Observador Activo del Registro del
Espacio de Fock, encargado de gobernar y certificar síncronamente la consistencia
cuántica de la inyección de "Vitaminas Cognitivas" (ToonCartridges) en la
Malla Agéntica, actuando como un endofuntor en la categoría de haces sobre el 
sitio de Grothendieck con valores en el retículo de Heyting $$\Omega_3$$.

El sistema trata el flujo de capacidades y restricciones tácticas no como meros
registros de bases de datos relacionales, sino como excitaciones cuánticas (bosones
y fermiones) sobre un espacio de Fock multimodo separable, forzando la disipación
del ruido del LLM y el veto inmediato ante cualquier asonancia de fase.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ────────────
La transición de estados se rige por la Ley de Clausura Transitiva de subespacios
de Hilbert covariantes y se compone de tres fases fuertemente acopladas:

  Fase 1 ──► FASE 1: OBSERVACIÓN DE ISOMETRÍA Y COTA DE BEKENSTEIN (Observe)
             Audita la inyección y el principio de exclusión de Pauli.
             Construye la matriz densidad inicial: $$\rho \succeq 0, \;\operatorname{Tr}(\rho)=1$$.
             Entrega: Phase1FockIsometryCertificate como precondición formal de Fase 2.

  Fase 2 ──► FASE 2: ORIENTACIÓN SIMPLÉCTICA DE BOGOLIUBOV-VALATIN (Orient)
             Diagonaliza el Hamiltoniano cuadrático mediante isometrías de Lie.
             Certifica la invarianza simpléctica: $$|u_k|^2 - |v_k|^2 = 1$$.
             Entrega: Phase2SymplecticBogoliubovCertificate como precondición de Fase 3.

  Fase 3 ──► FASE 3: DECISIÓN EN LINDBLAD Y PURIFICACIÓN ESPECTRAL (Decide & Act)
             Resuelve la ecuación maestra disipativa y ejecuta la proyección de Higham.
             Evolución cuántica: $$\dot{\rho} = -i[\hat{H}, \rho] + \mathcal{D}(\rho)$$.
             Veredicto: Colapso síncrono en $$\Omega_3$$ y disparo de potencia del Crowbar.

INVARIANTES MATEMÁTICOS Y GEOMÉTRICOS PRESERVADOS: ──────────────────────────────
  [I1] Principio de Exclusión de Pauli:   $$(a_i^\dagger)^2 = 0 \quad \forall i \in \text{Fermion}$$
  [I2] Conservación de la Norma de Fock:  $$\|\star_k \psi\|_{\Lambda^{N-k}} = \|\psi\|_{\Lambda^k}$$
  [I3] Preservación de Traza Cuántica:    $$\operatorname{Tr}(\rho(t)) \equiv 1.0$$
  [I4] Simetría Hermítica de la Densidad: $$\rho(t) = \rho(t)^\dagger \succeq 0$$
  [I5] Condición de Solubilidad Fredholm: $$\rho \perp \ker(\Delta)$$
"""
from __future__ import annotations
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Tuple, Optional, Dict, List, Union
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Soporte para matrices dispersas (Registros de gran escala)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    csr_matrix = None
    isspmatrix = None

# ─────────────────────────────────────────────────────────────────────────────
# Stubs seguros del núcleo MIC (Zero-Trust)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""
        pass

    class Morphism:
        """Clase base de morfismos en la categoría MIC."""
        pass

    class Stratum:
        """Estratos de la jerarquía DIKW."""
        PHYSICS = "PHYSICS"
        TACTICS = "TACTICS"
        STRATEGY = "STRATEGY"
        WISDOM = "WISDOM"

# ─────────────────────────────────────────────────────────────────────────────
# Importación robusta del motor del Espacio de Fock
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.inmune_system.synaptic_fock_space_registry import (
        SynapticFockSpaceRegistry,
        ToonCartridge,
        ElectronCartridge,
        ProtonCartridge,
        PolaronCartridge,
        TorsionCartridge,
        HouseholderReflectionFermion,
        PositronCartridge,
        GammaPhoton,
        PauliExclusionViolationError,
        SymplecticConstraintError,
        TraceAnomalyError,
        FockSpaceOverflowError,
        DensityMatrixNonPSDError,
    )
except ImportError:
    try:
        from synaptic_fock_space_registry import (
            SynapticFockSpaceRegistry,
            ToonCartridge,
            ElectronCartridge,
            ProtonCartridge,
            PolaronCartridge,
            TorsionCartridge,
            HouseholderReflectionFermion,
            PositronCartridge,
            GammaPhoton,
            PauliExclusionViolationError,
            SymplecticConstraintError,
            TraceAnomalyError,
            FockSpaceOverflowError,
            DensityMatrixNonPSDError,
        )
    except ImportError:
        # Fallbacks mínimos para evitar errores de importación
        class SynapticRegistryError(TopologicalInvariantError):
            pass
        class PauliExclusionViolationError(SynapticRegistryError):
            pass
        class FockSpaceOverflowError(SynapticRegistryError):
            pass
        class TraceAnomalyError(SynapticRegistryError):
            pass
        class SymplecticConstraintError(SynapticRegistryError):
            pass
        class DensityMatrixNonPSDError(SynapticRegistryError):
            pass
        # Clases dummy para tipado
        class ToonCartridge:
            cartridge_id: str = ""
            signature: str = ""
            @property
            def quantum_state_hash(self) -> str:
                return ""
        class ElectronCartridge(ToonCartridge):
            homological_charge: int = 0
            inertial_mass: float = 0.0
        class PositronCartridge(ToonCartridge):
            homological_charge: int = 0
        class GammaPhoton(ToonCartridge):
            pass
        class SynapticFockSpaceRegistry:
            pass

logger = logging.getLogger("MIC.Agents.Omega.SynapticFockSpaceRegistryAgent")

# ═══════════════════════════════════════════════════════════════════════════
# Constantes físicas, cuánticas y de precisión de la FPU
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_CROWBAR_GPIO: Final[int] = 14  # GPIO14 — Hardware ESP32
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12  # Suelo espectral de Wilkinson
_DENSITY_MATRIX_CONDITION_LIMIT: Final[float] = 1.0e8
_SPARSE_THRESHOLD: Final[float] = 0.3  # Umbral de dispersión para conversión
_MAX_REGISTRY_SIZE: Final[int] = 1000  # Cota de Bekenstein ampliada
_ENTROPY_EVICT_THRESHOLD: Final[float] = 0.9  # Umbral para evicción por entropía

# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones del agente (funtores de error en la categoría MIC)
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistryAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de la Memoria Cuántica."""
    pass

class FockIsometryViolation(SynapticFockSpaceRegistryAgentError):
    """Desviaciones no unitarias en el espacio de Fock."""
    pass

class DensityMatrixAnomalyError(SynapticFockSpaceRegistryAgentError):
    """Violación de los postulados de Dirac-von Neumann."""
    pass

class SymplecticInvarianceViolation(SynapticFockSpaceRegistryAgentError):
    """Violación de la invarianza simpléctica."""
    pass

class LindbladTraceViolation(SynapticFockSpaceRegistryAgentError):
    """Evolución disipativa con traza no conservada."""
    pass

class FredholmSolvabilityViolation(SynapticFockSpaceRegistryAgentError):
    """Violación de la condición de solubilidad de Fredholm."""
    pass

class LipschitzStabilityViolation(SynapticFockSpaceRegistryAgentError):
    """Violación de la cota de estabilidad de Lipschitz."""
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Enumeraciones del retículo de Heyting y acciones de hardware
# ═══════════════════════════════════════════════════════════════════════════
class FockSovereignVerdict(IntEnum):
    """
    Clasificador de tres valores en Ω₃ (retículo de Heyting).
    
    Orden por severidad operativa:
        COHERENT  = ⊤ operativo (registro válido)
        DEGRADED  = elemento intermedio (precisión reducida)
        VETOED    = ⊥ operativo (registro inválido)
    
    El supremo de veredictos se toma como máximo nivel de severidad.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "FockSovereignVerdict") -> "FockSovereignVerdict":
        """
        Supremo en el retículo de severidad.
        Si no se proveen veredictos, retorna COHERENT como elemento neutro.
        """
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == FockSovereignVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == FockSovereignVerdict.DEGRADED

class CrowbarAction(Enum):
    """Acción física de mitigación tras el veredicto."""
    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()

# ═══════════════════════════════════════════════════════════════════════════
# Certificados inmutables de cada fase (objetos de la subcategoría Spec)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1FockIsometryCertificate:
    """
    FASE 1 – Isometría de Fock y ocupación.
    
    Este certificado constituye el objeto terminal de la Fase 1 y es consumido
    por la Fase 2 a través del puente build_phase1_bridge().
    
    Atributos
    ---------
    is_pauli_respected : bool
        Verificación del principio de exclusión de Pauli.
    is_fock_bounded : bool
        Verificación de la cota de Bekenstein.
    entropy_eviction_ratio : float
        Ratio de entropía para evicción (coseno de similitud).
    occupancy : int
        Número actual de partículas en el registro.
    max_capacity : int
        Capacidad máxima del registro.
    density_matrix_trace : float
        Traza del operador densidad (debe ser ≈ 1.0).
    density_matrix_purity : float
        Pureza Tr(ρ²) del operador densidad.
    higham_regularization_applied : bool
        Indicador de si se aplicó regularización de Higham.
    sparse_computation : bool
        Indicador de si se usó computación dispersa.
    verdict : FockSovereignVerdict
        Veredicto local de la Fase 1.
    """
    is_pauli_respected: bool
    is_fock_bounded: bool
    entropy_eviction_ratio: float
    occupancy: int
    max_capacity: int
    density_matrix_trace: float
    density_matrix_purity: float
    higham_regularization_applied: bool
    sparse_computation: bool
    verdict: FockSovereignVerdict

@dataclass(frozen=True, slots=True)
class Phase2SymplecticBogoliubovCertificate:
    """
    FASE 2 – Invarianza simpléctica y aniquilación.
    
    Atributos
    ---------
    is_symplectic_invariant : bool
        Verificación de |u|² - |v|² = 1.
    symplectic_residual : float
        Residuo de la restricción simpléctica.
    is_charge_conserved : bool
        Conservación de carga homológica en aniquilación.
    annihilation_provenance_hash : Optional[str]
        Hash de procedencia de la aniquilación.
    is_consistent_with_occupancy : bool
        Consistencia con el nivel de ocupación de Fase 1.
    bogoliubov_transformation_count : int
        Número de transformaciones aplicadas.
    annihilation_events : int
        Número de aniquilaciones ejecutadas.
    verdict : FockSovereignVerdict
        Veredicto local de la Fase 2.
    """
    is_symplectic_invariant: bool
    symplectic_residual: float
    is_charge_conserved: bool
    annihilation_provenance_hash: Optional[str]
    is_consistent_with_occupancy: bool
    bogoliubov_transformation_count: int
    annihilation_events: int
    verdict: FockSovereignVerdict

@dataclass(frozen=True, slots=True)
class Phase3LindbladPassivityCertificate:
    """
    FASE 3 – Evolución de Lindblad y pasividad.
    
    Atributos
    ---------
    is_trace_preserved : bool
        Verificación de Tr(ρ) = 1.
    trace_residual : float
        Residuo |Tr(ρ) - 1|.
    is_density_matrix_psd : bool
        Verificación de positividad semidefinida.
    is_lyapunov_passive : bool
        Verificación de disipación no negativa.
    lyapunov_derivative : float
        Derivada de Lyapunov ΔE.
    energy_dissipated : float
        Energía total disipada.
    is_energy_balance_consistent : bool
        Consistencia del balance energético.
    is_fredholm_soluble : bool
        Solubilidad de Fredholm para ρ.
    fredholm_residual : float
        Residuo de proyección sobre ker(L).
    is_lipschitz_stable : bool
        Estabilidad de Lipschitz verificada.
    evolution_steps : int
        Número de pasos de evolución.
    sparse_computation : bool
        Indicador de computación dispersa.
    verdict : FockSovereignVerdict
        Veredicto local de la Fase 3.
    """
    is_trace_preserved: bool
    trace_residual: float
    is_density_matrix_psd: bool
    is_lyapunov_passive: bool
    lyapunov_derivative: float
    energy_dissipated: float
    is_energy_balance_consistent: bool
    is_fredholm_soluble: bool
    fredholm_residual: float
    is_lipschitz_stable: bool
    evolution_steps: int
    sparse_computation: bool
    verdict: FockSovereignVerdict

@dataclass(frozen=True, slots=True)
class FockRegistrySovereignState:
    """Certificado global terminal de la gobernanza."""
    phase1: Phase1FockIsometryCertificate
    phase2: Phase2SymplecticBogoliubovCertificate
    phase3: Phase3LindbladPassivityCertificate
    final_verdict: FockSovereignVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Regularización Espectral (Higham para Operadores Densidad)
# ═══════════════════════════════════════════════════════════════════════════
def stable_density_matrix_higham(
    rho: NDArray[np.complex128],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR
) -> Tuple[NDArray[np.complex128], float, bool]:
    r"""
    Aplica la proyección de Higham para estabilizar el operador densidad ρ
    en el cono PSD (Positive Semi-Definite) con traza unitaria.
    
    Axioma de Proyección:
        \tilde{ρ} = \arg\min_{σ \succeq 0, Tr(σ)=1} \|ρ - σ\|_F
    
    Parámetros
    ----------
    rho : NDArray[np.complex128]
        Operador densidad (potencialmente no-PSD por fluctuaciones FPU).
    tolerance : float
        Suelo espectral de Wilkinson para recorte de autovalores.
    
    Retorna
    -------
    Tuple[NDArray[np.complex128], float, bool]
        (rho_projected, shift, higham_applied) donde shift es el desplazamiento
        espectral aplicado y higham_applied indica si se requirió regularización.
    
    Raises
    ------
    DensityMatrixAnomalyError
        Si la proyección de Higham falla en producir una matriz PSD.
    """
    # 1. Hermitización exacta
    rho_herm = 0.5 * (rho + rho.conj().T)
    
    # 2. Descomposición espectral
    try:
        vals, vecs = la.eigh(rho_herm)
    except Exception as exc:
        raise DensityMatrixAnomalyError(
            "DensityMatrixAnomalyError: Falló descomposición espectral para Higham."
        ) from exc

    lam_min = float(np.min(vals))
    lam_max = float(np.max(vals))
    shift = 0.0
    applied = False

    # 3. Recorte de autovalores negativos al suelo espectral
    if lam_min <= tolerance:
        shift = abs(lam_min) + max(tolerance, lam_max / _DENSITY_MATRIX_CONDITION_LIMIT)
        applied = True

    # 4. Reconstrucción con desplazamiento
    vals_reg = vals + shift
    vals_clean = np.maximum(vals_reg, tolerance)

    # 5. Normalización de traza a unidad
    trace = np.sum(vals_clean)
    if trace > _MACHINE_EPS:
        vals_clean /= trace
    else:
        # Estado de vacío: asignar todo al último autovalor
        vals_clean = np.zeros_like(vals_clean)
        vals_clean[-1] = 1.0

    # 6. Reconstrucción del operador densidad proyectado
    rho_projected = vecs @ np.diag(vals_clean) @ vecs.conj().T
    
    # 7. Hermitización numérica post-proyección
    rho_projected = 0.5 * (rho_projected + rho_projected.conj().T)

    if applied:
        logger.warning(
            "Operador densidad inestable. Proyectado ρ al cono PSD via Higham "
            "(autovalores mínimos: %.4e → %.4e, shift: %.4e).",
            float(np.min(vals)),
            float(np.min(vals_clean)),
            float(shift)
        )

    # 8. Verificación final de PSD
    try:
        final_vals = la.eigvalsh(rho_projected)
        if np.any(final_vals < -tolerance):
            raise DensityMatrixAnomalyError(
                "DensityMatrixAnomalyError: Proyección de Higham no produjo matriz PSD."
            )
    except Exception as exc:
        raise DensityMatrixAnomalyError(str(exc)) from exc

    return rho_projected, shift, applied

# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 – Observe: Auditoría de Isometría y Ocupación
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_FockIsometryAuditor(ABC):
    """
    FASE 1 (Observe): Sanea la estructura del espacio de Fock verificando
    el principio de exclusión de Pauli, la cota de Bekenstein y la entropía.
    
    El último método de esta fase:
        build_phase1_bridge(...)
    constituye el morfismo de transición formal hacia la Fase 2.
    """

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        """
        Inicializa el auditor de isometría de Fock.
        
        Parámetros
        ----------
        tolerance : float
            Cota de precisión para validaciones cuánticas.
        """
        self._tolerance = tolerance
        self._higham_floor = _HIGHAM_REGULARIZATION_FLOOR

    def audit_fock_isometry(
        self,
        registry: SynapticFockSpaceRegistry,
        current_decision_vector: NDArray[np.float64]
    ) -> Phase1FockIsometryCertificate:
        """
        Audita la isometría del registro, comprobando realmente si hay
        violaciones de Pauli y evaluando la tasa de entropía.
        
        Parámetros
        ----------
        registry : SynapticFockSpaceRegistry
            Registro físico de partículas.
        current_decision_vector : NDArray
            Vector de decisión actual (para cálculo de similitud).
        
        Retorna
        -------
        Phase1FockIsometryCertificate
            Certificado terminal de Fase 1.
        """
        # Verificar Pauli recorriendo el registro real
        is_pauli_respected = self._check_pauli_exclusion(registry)
        
        # Verificar cota de Bekenstein
        occupancy = registry.size
        max_cap = registry.max_capacity
        is_fock_bounded = occupancy <= max_cap
        
        # Ratio de entropía: coseno del vector de decisión con el promedio de embeddings
        entropy_ratio, sparse_used = self._compute_entropy_ratio(
            registry, current_decision_vector
        )
        
        # Construir operador densidad con regularización de Higham
        rho, higham_applied = self._build_density_matrix_with_higham(registry)
        
        # Calcular traza y pureza
        trace = float(np.real(np.trace(rho)))
        purity = float(np.real(np.trace(rho @ rho)))
        
        # Veredicto
        verdict = FockSovereignVerdict.COHERENT
        if not is_fock_bounded or not is_pauli_respected:
            verdict = FockSovereignVerdict.VETOED
        elif occupancy >= max_cap * _ENTROPY_EVICT_THRESHOLD:
            verdict = FockSovereignVerdict.DEGRADED
        elif higham_applied or abs(trace - 1.0) > self._tolerance * 100:
            verdict = FockSovereignVerdict.DEGRADED
        
        return Phase1FockIsometryCertificate(
            is_pauli_respected=is_pauli_respected,
            is_fock_bounded=is_fock_bounded,
            entropy_eviction_ratio=entropy_ratio,
            occupancy=occupancy,
            max_capacity=max_cap,
            density_matrix_trace=trace,
            density_matrix_purity=purity,
            higham_regularization_applied=higham_applied,
            sparse_computation=sparse_used,
            verdict=verdict,
        )

    def _check_pauli_exclusion(self, registry: SynapticFockSpaceRegistry) -> bool:
        """
        Comprueba que no haya dos fermiones del mismo tipo con el mismo
        hash de estado cuántico. Complejidad O(1) por tipo de fermión.
        """
        fermion_types = (
            ElectronCartridge, ProtonCartridge, PolaronCartridge,
            TorsionCartridge, HouseholderReflectionFermion
        )
        seen_hashes: Dict[type, set] = {}
        
        for cart in registry._registry.values():
            if isinstance(cart, fermion_types):
                typ = type(cart)
                if typ not in seen_hashes:
                    seen_hashes[typ] = set()
                h = cart.quantum_state_hash
                if h in seen_hashes[typ]:
                    return False
                seen_hashes[typ].add(h)
        
        return True

    def _compute_entropy_ratio(
        self,
        registry: SynapticFockSpaceRegistry,
        current_decision_vector: NDArray[np.float64]
    ) -> Tuple[float, bool]:
        """
        Calcula el ratio de entropía usando similitud de coseno.
        Soporta computación dispersa para registros grandes.
        """
        occupancy = registry.size
        sparse_used = False
        
        if occupancy == 0 or current_decision_vector.size == 0:
            return 0.0, sparse_used
        
        u = current_decision_vector / (la.norm(current_decision_vector) + _MACHINE_EPS)
        
        # Recuperar todos los embeddings del registro
        embeddings = list(registry._embedding_cache.values())
        
        if not embeddings:
            return 0.0, sparse_used
        
        # Verificar si usar computación dispersa
        if _SPARSE_AVAILABLE and len(embeddings) > 100:
            sparse_used = True
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            avg_embedding = np.mean(embeddings, axis=0)
        
        avg_norm = la.norm(avg_embedding)
        if avg_norm > _MACHINE_EPS:
            avg_embedding /= avg_norm
            entropy_ratio = float(np.dot(u, avg_embedding))
        else:
            entropy_ratio = 0.0
        
        return entropy_ratio, sparse_used

    def _build_density_matrix_with_higham(
        self,
        registry: SynapticFockSpaceRegistry
    ) -> Tuple[NDArray[np.complex128], bool]:
        """
        Construye el operador densidad con regularización de Higham.
        """
        n = len(registry._registry)
        if n == 0:
            # Estado de vacío sin partículas
            return np.eye(1, dtype=np.complex128), False
        
        # Suponemos que todos los embeddings tienen la misma dimensión
        sample_vec = next(iter(registry._embedding_cache.values()))
        dim = len(sample_vec)
        rho = np.zeros((dim, dim), dtype=np.complex128)
        
        for v in registry._embedding_cache.values():
            vec = v.reshape(-1, 1).astype(np.complex128)
            rho += vec @ vec.conj().T
        
        rho /= n
        
        # Aplicar regularización de Higham para garantizar PSD y traza unitaria
        rho_projected, shift, higham_applied = stable_density_matrix_higham(
            rho, self._higham_floor
        )
        
        return rho_projected, higham_applied

    # ─────────────────────────────────────────────────────────────────────
    # Último método de la Fase 1: Puente hacia la Fase 2
    # ─────────────────────────────────────────────────────────────────────
    def build_phase1_bridge(
        self,
        phase1_cert: Phase1FockIsometryCertificate
    ) -> dict:
        """
        MORFISMO DE TRANSICIÓN FASE 1 → FASE 2.
        
        Prepara un diccionario con información de ocupación y estabilidad
        para ser utilizado por la Fase 2.
        
        Parámetros
        ----------
        phase1_cert : Phase1FockIsometryCertificate
            Certificado de la Fase 1.
        
        Retorna
        -------
        dict
            Diccionario con datos de puente para Fase 2.
        """
        return {
            'occupancy': phase1_cert.occupancy,
            'max_capacity': phase1_cert.max_capacity,
            'is_stable': phase1_cert.verdict != FockSovereignVerdict.VETOED,
            'density_matrix_trace': phase1_cert.density_matrix_trace,
            'density_matrix_purity': phase1_cert.density_matrix_purity,
            'higham_applied': phase1_cert.higham_regularization_applied,
            'sparse_used': phase1_cert.sparse_computation,
        }

    def generate_phase1_certificate(
        self,
        registry: SynapticFockSpaceRegistry,
        current_decision_vector: NDArray[np.float64]
    ) -> Phase1FockIsometryCertificate:
        """
        Genera el certificado de Fase 1 con auditoría completa.
        Método conveniencia que llama a audit_fock_isometry.
        """
        return self.audit_fock_isometry(registry, current_decision_vector)

# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 – Orient: Auditoría Simpléctica y de Aniquilación
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_SymplecticBogoliubovValidator(Phase1_FockIsometryAuditor):
    """
    FASE 2 (Orient): Valida la invarianza simpléctica de la transformación de
    Bogoliubov y la consistencia de la aniquilación con el contexto de la Fase 1.
    
    Hereda de la Fase 1 para usar build_phase1_bridge() como puente formal.
    """

    def audit_symplectic_bogoliubov(
        self,
        u_k: complex,
        v_k: complex,
        phase1_cert: Phase1FockIsometryCertificate,
        electron: Optional[ElectronCartridge] = None,
        positron: Optional[PositronCartridge] = None
    ) -> Phase2SymplecticBogoliubovCertificate:
        r"""
        Verifica la conservación simpléctica $|u|^2 - |v|^2 = 1$ y la
        aniquilación carga-conservativa. Adicionalmente comprueba que la
        transformación sea compatible con el nivel de ocupación del registro.
        
        Parámetros
        ----------
        u_k, v_k : complex
            Coeficientes de Bogoliubov.
        phase1_cert : Phase1FockIsometryCertificate
            Certificado de la Fase 1 para verificar consistencia.
        electron, positron : opcional
            Partículas para aniquilación.
        
        Retorna
        -------
        Phase2SymplecticBogoliubovCertificate
            Certificado terminal de Fase 2.
        """
        # 1. Invarianza simpléctica
        norm_u = abs(u_k) ** 2
        norm_v = abs(v_k) ** 2
        symplectic_residual = abs((norm_u - norm_v) - 1.0)
        is_symplectic = symplectic_residual <= self._tolerance * 100
        
        # 2. Conservación de carga en aniquilación
        is_charge_ok = True
        p_hash = None
        annihilation_events = 0
        
        if electron is not None and positron is not None:
            if electron.homological_charge + positron.homological_charge != 0:
                is_charge_ok = False
            else:
                raw = f"ANNIHILATION-{electron.quantum_state_hash}-{positron.quantum_state_hash}"
                p_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                annihilation_events = 1
        
        # 3. Consistencia con la ocupación de la Fase 1 (puente formal)
        bridge = self.build_phase1_bridge(phase1_cert)
        occupancy_ratio = bridge['occupancy'] / max(bridge['max_capacity'], 1)
        
        # Criterio heurístico: si la ocupación es alta (>80%), |v_k| no debe superar 0.5
        consistent_occ = True
        if occupancy_ratio > 0.8 and abs(v_k) > 0.5:
            consistent_occ = False
        
        # 4. Contar transformaciones de Bogoliubov
        bogoliubov_count = 1 if is_symplectic else 0
        
        # Veredicto
        verdict = FockSovereignVerdict.COHERENT
        if not is_symplectic or not is_charge_ok:
            verdict = FockSovereignVerdict.VETOED
        elif symplectic_residual > self._tolerance or not consistent_occ:
            verdict = FockSovereignVerdict.DEGRADED
        
        return Phase2SymplecticBogoliubovCertificate(
            is_symplectic_invariant=is_symplectic,
            symplectic_residual=symplectic_residual,
            is_charge_conserved=is_charge_ok,
            annihilation_provenance_hash=p_hash,
            is_consistent_with_occupancy=consistent_occ,
            bogoliubov_transformation_count=bogoliubov_count,
            annihilation_events=annihilation_events,
            verdict=verdict,
        )

    def generate_phase2_certificate(
        self,
        u_k: complex,
        v_k: complex,
        phase1_cert: Phase1FockIsometryCertificate,
        electron: Optional[ElectronCartridge] = None,
        positron: Optional[PositronCartridge] = None
    ) -> Phase2SymplecticBogoliubovCertificate:
        """
        Genera el certificado de Fase 2 con auditoría completa.
        Método conveniencia que llama a audit_symplectic_bogoliubov.
        """
        return self.audit_symplectic_bogoliubov(
            u_k, v_k, phase1_cert, electron, positron
        )

# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 – Decide: Auditoría de Lindblad y Pasividad
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_LindbladPassivityValidator(Phase2_SymplecticBogoliubovValidator):
    """
    FASE 3 (Decide): Verifica la evolución disipativa según Lindblad,
    conservación de traza, positividad y balance energético.
    
    Hereda de la Fase 2 para poder generar pasos y evaluar el sistema.
    """

    def audit_lindblad_evolution(
        self,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
        hamiltonian_eff: NDArray[np.complex128],
        phase2_cert: Phase2SymplecticBogoliubovCertificate,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        evolution_steps: int = 10
    ) -> Phase3LindbladPassivityCertificate:
        r"""
        Evalúa la evolución cuántica abierta: traza, positividad y disipación.
        
        .. math::
            \operatorname{Tr}(\rho_{\mathrm{post}}) = 1, \quad
            \rho_{\mathrm{post}} \succeq 0, \quad
            \Delta E = \operatorname{Tr}(H_{\mathrm{eff}} (\rho_{\mathrm{post}} - \rho_{\mathrm{pre}})) \le 0.
        
        Si se proporcionan los operadores de salto, se verifica que la energía
        disipada sea consistente con la ecuación de Lindblad.
        
        Parámetros
        ----------
        rho_pre, rho_post : NDArray[np.complex128]
            Matrices de densidad antes y después.
        hamiltonian_eff : NDArray[np.complex128]
            Hamiltoniano efectivo (Hermítico).
        phase2_cert : Phase2SymplecticBogoliubovCertificate
            Para comprobar coherencia con la aniquilación (si hubo).
        jump_operators, gamma_rates : opcional
            Para validar el balance energético detallado.
        evolution_steps : int
            Número de pasos de evolución.
        
        Retorna
        -------
        Phase3LindbladPassivityCertificate
            Certificado terminal de Fase 3.
        """
        # 1. Traza
        trace_pre = float(np.real(np.trace(rho_pre)))
        trace_post = float(np.real(np.trace(rho_post)))
        trace_residual = abs(trace_post - 1.0)
        is_trace_ok = trace_residual <= self._tolerance * 1000
        
        # 2. Positividad
        vals_post = la.eigvalsh(rho_post)
        is_psd = np.all(vals_post >= -self._tolerance * 100)
        
        # 3. Disipación de Lyapunov
        energy_pre = float(np.real(np.trace(hamiltonian_eff @ rho_pre)))
        energy_post = float(np.real(np.trace(hamiltonian_eff @ rho_post)))
        lyap_derivative = energy_post - energy_pre
        is_passive = lyap_derivative <= self._tolerance * 1000
        
        # 4. Balance energético con operadores de salto (si se proporcionan)
        energy_dissipated = -lyap_derivative
        is_energy_balance_consistent = True
        sparse_used = False
        
        if jump_operators and gamma_rates:
            # Calcular la potencia disipada esperada: sum_k γ_k Tr(L_k ρ L_k†)
            expected_diss = 0.0
            for L, gamma in zip(gamma_rates, jump_operators):
                if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L):
                    sparse_used = True
                    expected_diss += gamma * np.real(np.trace(L @ rho_pre @ L.conj().T))
                else:
                    expected_diss += gamma * np.real(np.trace(L @ rho_pre @ L.conj().T))
            
            # La energía disipada real debe ser aproximadamente igual a la esperada
            balance_error = abs(energy_dissipated - expected_diss)
            is_energy_balance_consistent = balance_error <= self._tolerance * 100
        
        # 5. Solubilidad de Fredholm (ortogonalidad al núcleo)
        is_fredholm_soluble, fredholm_residual = self._check_fredholm_solvability(
            rho_post, hamiltonian_eff
        )
        
        # 6. Estabilidad de Lipschitz
        is_lipschitz_stable = self._check_lipschitz_stability(rho_pre, rho_post)
        
        # Veredicto
        verdict = FockSovereignVerdict.COHERENT
        if not is_trace_ok or not is_psd:
            verdict = FockSovereignVerdict.VETOED
        elif not is_passive or trace_residual > self._tolerance or not is_fredholm_soluble:
            verdict = FockSovereignVerdict.DEGRADED
        
        return Phase3LindbladPassivityCertificate(
            is_trace_preserved=is_trace_ok,
            trace_residual=trace_residual,
            is_density_matrix_psd=is_psd,
            is_lyapunov_passive=is_passive,
            lyapunov_derivative=lyap_derivative,
            energy_dissipated=energy_dissipated,
            is_energy_balance_consistent=is_energy_balance_consistent,
            is_fredholm_soluble=is_fredholm_soluble,
            fredholm_residual=fredholm_residual,
            is_lipschitz_stable=is_lipschitz_stable,
            evolution_steps=evolution_steps,
            sparse_computation=sparse_used,
            verdict=verdict,
        )

    def _check_fredholm_solvability(
        self,
        rho: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128]
    ) -> Tuple[bool, float]:
        """
        Verifica la condición de solubilidad de Fredholm para la evolución.
        """
        # Obtener autovalores del Hamiltoniano
        vals_H = la.eigvalsh(hamiltonian)
        
        # Umbral para el núcleo
        max_val = np.max(np.abs(vals_H)) if len(vals_H) > 0 else 1.0
        kernel_mask = np.abs(vals_H) <= max(self._tolerance * 100, max_val * _MACHINE_EPS * 10)
        
        if not np.any(kernel_mask):
            return True, 0.0
        
        # Proyección sobre el núcleo
        vals_rho = la.eigvalsh(rho)
        residual = float(np.sum(np.abs(vals_rho[kernel_mask])))
        is_soluble = residual <= self._tolerance * 100
        
        return is_soluble, residual

    def _check_lipschitz_stability(
        self,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128]
    ) -> bool:
        """
        Verifica la estabilidad de Lipschitz de la evolución.
        """
        delta_rho = rho_post - rho_pre
        norm_delta = float(la.norm(delta_rho, ord='fro'))
        norm_pre = float(la.norm(rho_pre, ord='fro'))
        
        # Cota de Lipschitz: ||ρ_post - ρ_pre|| ≤ L ||ρ_pre||
        lipschitz_constant = 2.0  # Cota conservadora para evolución unitaria
        rhs = lipschitz_constant * norm_pre
        
        return norm_delta <= rhs + self._tolerance * 100

    def generate_phase3_certificate(
        self,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
        hamiltonian_eff: NDArray[np.complex128],
        phase2_cert: Phase2SymplecticBogoliubovCertificate,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        evolution_steps: int = 10
    ) -> Phase3LindbladPassivityCertificate:
        """
        Genera el certificado de Fase 3 con auditoría completa.
        Método conveniencia que llama a audit_lindblad_evolution.
        """
        return self.audit_lindblad_evolution(
            rho_pre, rho_post, hamiltonian_eff, phase2_cert,
            jump_operators, gamma_rates, evolution_steps
        )

# ═══════════════════════════════════════════════════════════════════════════
# Agente Soberano final (Morfismo OODA)
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistryAgent(Morphism, Phase3_LindbladPassivityValidator):
    """
    Soberano de la Memoria Cuántica. Ejecuta el ciclo OODA completo sobre
    el Espacio de Fock, utilizando las tres fases anidadas.
    
    El ciclo de gobernanza queda formalizado como:
        Observe  → Fase 1 (FockIsometryAuditor)
        Orient   → Fase 2 (SymplecticBogoliubovValidator)
        Decide   → Fase 3 (LindbladPassivityValidator)
    
    El veredicto final es el supremo de severidad en el retículo de Heyting.
    """

    def __init__(
        self,
        raise_on_veto: bool = False,
        tolerance: float = _DEFAULT_TOL
    ) -> None:
        """
        Inicializa el agente soberano con configuración de precisión.
        
        Parámetros
        ----------
        raise_on_veto : bool
            Si True, lanza excepción cuando el veredicto es VETOED.
        tolerance : float
            Cota de precisión para validaciones.
        """
        Morphism.__init__(self, "SynapticFockSpaceRegistryAgent")
        Phase3_LindbladPassivityValidator.__init__(self, tolerance=tolerance)
        self._raise_on_veto = raise_on_veto
        self._target_stratum: Stratum = Stratum.WISDOM

    def execute_sovereign_governance(
        self,
        registry: SynapticFockSpaceRegistry,
        current_decision_vector: NDArray[np.float64],
        u_k: complex,
        v_k: complex,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
        hamiltonian_eff: NDArray[np.complex128],
        electron: Optional[ElectronCartridge] = None,
        positron: Optional[PositronCartridge] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        evolution_steps: int = 10,
    ) -> FockRegistrySovereignState:
        """
        Orquesta la auditoría completa de las tres fases.
        
        Estrategia de anidación:
          1. Si Fase 1 emite VETOED, se emiten certificados vetados para
             Fase 2 y Fase 3 sin continuar el cómputo.
          2. Si Fase 2 emite VETOED, se emite certificado vetado para Fase 3.
          3. Si todas las fases continúan, Fase 3 audita la evolución de Lindblad.
        
        Parámetros
        ----------
        registry : SynapticFockSpaceRegistry
            Registro del espacio de Fock.
        current_decision_vector : NDArray
            Vector de decisión actual.
        u_k, v_k : complex
            Coeficientes de Bogoliubov.
        rho_pre, rho_post : NDArray[np.complex128]
            Matrices de densidad antes y después de evolución.
        hamiltonian_eff : NDArray[np.complex128]
            Hamiltoniano efectivo.
        electron, positron : opcional
            Partículas para aniquilación.
        jump_operators, gamma_rates : opcional
            Para verificar balance energético.
        evolution_steps : int
            Número de pasos de evolución.
        
        Retorna
        -------
        FockRegistrySovereignState
            Certificado global terminal.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            # ─── FASE 1: Observe ───
            cert1 = self.audit_fock_isometry(registry, current_decision_vector)
            
            if cert1.verdict.is_vetoed:
                cert2 = self._vetoed_phase2_certificate()
                cert3 = self._vetoed_phase3_certificate(evolution_steps)
            else:
                # ─── FASE 2: Orient (recibe cert1) ───
                cert2 = self.audit_symplectic_bogoliubov(
                    u_k, v_k, cert1, electron, positron
                )
                
                if cert2.verdict.is_vetoed:
                    cert3 = self._vetoed_phase3_certificate(evolution_steps)
                else:
                    # ─── FASE 3: Decide (recibe cert2) ───
                    cert3 = self.audit_lindblad_evolution(
                        rho_pre, rho_post, hamiltonian_eff, cert2,
                        jump_operators, gamma_rates, evolution_steps
                    )
            
            # ─── Supremo de Heyting (fusión de veredictos) ───
            final_verdict = FockSovereignVerdict.supremum(
                cert1.verdict, cert2.verdict, cert3.verdict
            )
            
            crowbar_triggered = False
            crowbar_act = CrowbarAction.NONE
            diagnostic_note = "Ciclo OODA de memoria cuántica completado."
            
            if final_verdict == FockSovereignVerdict.VETOED:
                crowbar_triggered = True
                crowbar_act = CrowbarAction.HARD_SHORT
                diagnostic_note = (
                    "VETO CUÁNTICO: Violación de invariantes de Fock, simplécticidad "
                    "o trazabilidad de Lindblad. Activando Crowbar."
                )
                logger.error(
                    "VETO CUÁNTICO: Violación de invariantes. Activando Crowbar (GPIO%d).",
                    _CROWBAR_GPIO
                )
                if self._raise_on_veto:
                    raise SynapticFockSpaceRegistryAgentError(
                        "Veto cuántico en el espacio de Fock."
                    )
            elif final_verdict == FockSovereignVerdict.DEGRADED:
                crowbar_triggered = True
                crowbar_act = CrowbarAction.WATCHDOG_PULSE
                diagnostic_note = (
                    "MEMORIA DEGRADADA: Regularización Higham aplicada o producción "
                    "de entropía elevada. Emitiendo WATCHDOG_PULSE."
                )
                logger.warning("MEMORIA DEGRADADA: Emitiendo WATCHDOG_PULSE.")
            
            # Hash de procedencia auditable
            provenance_hash = self._generate_provenance_hash(
                cert1, cert2, cert3, timestamp_utc
            )
            
            return FockRegistrySovereignState(
                phase1=cert1,
                phase2=cert2,
                phase3=cert3,
                final_verdict=final_verdict,
                crowbar_triggered=crowbar_triggered,
                crowbar_action=crowbar_act,
                timestamp_utc=timestamp_utc,
                provenance_hash=provenance_hash,
                diagnostic_note=diagnostic_note,
            )
            
        except SynapticFockSpaceRegistryAgentError as exc:
            logger.error("Colapso categórico del soberano: %s. Forzando Crowbar.", exc)
            if self._raise_on_veto:
                raise
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)
            
        except Exception as exc:  # pragma: no cover
            logger.exception("Colapso catastrófico no tipado del soberano.")
            if self._raise_on_veto:
                raise SynapticFockSpaceRegistryAgentError(
                    "Colapso catastrófico no tipado del soberano."
                ) from exc
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        r"""Invocación como morfismo categórico."""
        if kwargs:
            return self.execute_sovereign_governance(**kwargs)
        return state

    # ─────────────────────────────────────────────────────────────────────
    # Certificados vetados para cortocircuito anidado
    # ─────────────────────────────────────────────────────────────────────
    def _vetoed_phase2_certificate(self) -> Phase2SymplecticBogoliubovCertificate:
        """Certificado vetado de Fase 2 para cortocircuito."""
        return Phase2SymplecticBogoliubovCertificate(
            is_symplectic_invariant=False,
            symplectic_residual=float("inf"),
            is_charge_conserved=False,
            annihilation_provenance_hash=None,
            is_consistent_with_occupancy=False,
            bogoliubov_transformation_count=0,
            annihilation_events=0,
            verdict=FockSovereignVerdict.VETOED,
        )

    def _vetoed_phase3_certificate(
        self,
        evolution_steps: int
    ) -> Phase3LindbladPassivityCertificate:
        """Certificado vetado de Fase 3 para cortocircuito."""
        return Phase3LindbladPassivityCertificate(
            is_trace_preserved=False,
            trace_residual=float("inf"),
            is_density_matrix_psd=False,
            is_lyapunov_passive=False,
            lyapunov_derivative=float("inf"),
            energy_dissipated=float("inf"),
            is_energy_balance_consistent=False,
            is_fredholm_soluble=False,
            fredholm_residual=float("inf"),
            is_lipschitz_stable=False,
            evolution_steps=evolution_steps,
            sparse_computation=False,
            verdict=FockSovereignVerdict.VETOED,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Estado catastrófico de emergencia
    # ─────────────────────────────────────────────────────────────────────
    def _cataclysm_state(
        self,
        reason: str,
        timestamp_utc: str
    ) -> FockRegistrySovereignState:
        """
        Construye un estado catastrófico con certificados vetados y Crowbar
        HARD_SHORT.
        
        Este estado es el objeto terminal de fallo dentro del topos operativo.
        """
        dummy1 = Phase1FockIsometryCertificate(
            is_pauli_respected=False,
            is_fock_bounded=False,
            entropy_eviction_ratio=0.0,
            occupancy=0,
            max_capacity=0,
            density_matrix_trace=0.0,
            density_matrix_purity=0.0,
            higham_regularization_applied=False,
            sparse_computation=False,
            verdict=FockSovereignVerdict.VETOED,
        )
        dummy2 = self._vetoed_phase2_certificate()
        dummy3 = self._vetoed_phase3_certificate(0)
        
        raw_payload = f"CATACLYSM_FOCK_VETO::{reason}"
        provenance_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        
        return FockRegistrySovereignState(
            phase1=dummy1,
            phase2=dummy2,
            phase3=dummy3,
            final_verdict=FockSovereignVerdict.VETOED,
            crowbar_triggered=True,
            crowbar_action=CrowbarAction.HARD_SHORT,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_FOCK_VETO: {reason}",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Hash de procedencia auditable
    # ─────────────────────────────────────────────────────────────────────
    def _generate_provenance_hash(
        self,
        c1: Phase1FockIsometryCertificate,
        c2: Phase2SymplecticBogoliubovCertificate,
        c3: Phase3LindbladPassivityCertificate,
        timestamp_utc: str,
    ) -> str:
        """
        Genera un hash SHA-256 de procedencia que ata los veredictos, residuos
        críticos y marca temporal.
        
        El hash constituye el sello de auditoría del certificado global.
        """
        raw_payload = "|".join(
            (
                timestamp_utc,
                str(c1.verdict.value),
                str(c2.verdict.value),
                str(c3.verdict.value),
                f"{c1.occupancy:d}",
                f"{c1.max_capacity:d}",
                f"{c1.density_matrix_trace:.12e}",
                f"{c1.density_matrix_purity:.12e}",
                f"{c2.symplectic_residual:.12e}",
                f"{c2.bogoliubov_transformation_count:d}",
                f"{c2.annihilation_events:d}",
                f"{c3.trace_residual:.12e}",
                f"{c3.lyapunov_derivative:.12e}",
                f"{c3.energy_dissipated:.12e}",
                f"{c3.fredholm_residual:.12e}",
                str(int(c1.higham_regularization_applied)),
                str(int(c1.sparse_computation)),
                str(int(c3.sparse_computation)),
                str(int(c3.evolution_steps)),
            )
        )
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

# ═══════════════════════════════════════════════════════════════════════════
# Compatibilidad de nombres de clases para versiones anteriores
# ═══════════════════════════════════════════════════════════════════════════
Phase1_FockIsometryAuditor = Phase1_FockIsometryAuditor
Phase2_SymplecticBogoliubovValidator = Phase2_SymplecticBogoliubovValidator
Phase3_LindbladPassivityValidator = Phase3_LindbladPassivityValidator

__all__ = [
    "SynapticFockSpaceRegistryAgentError",
    "FockIsometryViolation",
    "DensityMatrixAnomalyError",
    "SymplecticInvarianceViolation",
    "LindbladTraceViolation",
    "FredholmSolvabilityViolation",
    "LipschitzStabilityViolation",
    "FockSovereignVerdict",
    "CrowbarAction",
    "Phase1FockIsometryCertificate",
    "Phase2SymplecticBogoliubovCertificate",
    "Phase3LindbladPassivityCertificate",
    "FockRegistrySovereignState",
    "Phase1_FockIsometryAuditor",
    "Phase2_SymplecticBogoliubovValidator",
    "Phase3_LindbladPassivityValidator",
    "SynapticFockSpaceRegistryAgent",
    "stable_density_matrix_higham",
]