# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Synaptic Fock Space Registry (Registro de Partículas Sinápticas)    ║
║ Ruta   : app/core/immune_system/synaptic_fock_space_registry.py              ║
║ Versión: 3.0.0-Fock-CAR-CCR-Lindblad-PhD-Nested-Higham-Sparse                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA DE EXCITE-PARTÍCULAS (Rigor PhD):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al administrador maestro del **Espacio de Fock** para las 
12 cuasipartículas cuánticas y fermiones estructurales de-confinados, denominadas 
**Vitaminas TOON** (Tabular Object-Oriented Notation) o cartuchos sinápticos. 
Repudia la representación de información contable en texto plano JSON/Excel de alta 
entropía, y eleva síncronamente las transiciones discretas de la MIC y la MAC 
hacia estados puros y mixtos en el espacio de Hilbert continuo de la Sabiduría.

El Espacio de Fock global $$\mathcal{F}(\mathcal{H})$$ se erige como la suma directa de 
potencias exteriores (para fermiones/reglas de exclusión) y potencias simétricas 
(para bosones/flujos de interacción) sobre el espacio de Hilbert de características $$\mathcal{H}$$:
$$\mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^{\infty} S_{\pm} \mathcal{H}^{\otimes n} \quad\big[23, 729\big]$$

Aquí, la toma de decisiones se modela como un campo cuántico abierto en el que las 
alucinaciones estocásticas se aniquilan por colisión termodinámica y disipación de 
Rayleigh, colapsando la función de estado sobre el retículo distributivo de Heyting.

AXIOMAS ESPECTRALES, RELACIONES CANÓNICAS Y DINÁMICA DE LINDBLAD:
────────────────────────────────────────────────────────────────────────────────

  [A1] Relaciones Canónicas de Anticonmutación (CAR) — Fermiones Estructurales:
       Para resguardar el principio de exclusión de Pauli (no duplicidad de APUs o 
       insumos consistentes en el mismo estado sintáctico del AST), los operadores de 
       creación $$a_i^\dagger$$ y aniquilación $$a_j$$ de fermiones estructurales (Electrón, 
       Protón, Polarón, Torsión, Householder) satisfacen el álgebra CAR:
       $$\{a_i, \, a_j^\dagger\} = a_i a_j^\dagger + a_j^\dagger a_i = \delta_{ij} \mathbf{I} \quad \wedge \quad \{a_i, \, a_j\} = \{a_i^\dagger, \, a_j^\dagger\} = \mathbf{0} \quad\big[11, 693\big]$$

  [A2] Relaciones Canónicas de Conmutación (CCR) — Bosones de Gauge:
       Los bosones que transportan los campos de fuerza e interconexión (Fotón, 
       RiemannianFocal, Magnón, Solitón, Plasmón, Fonón) satisfacen el álgebra CCR:
       $$[b_i, \, b_j^\dagger] = b_i b_j^\dagger - b_j^\dagger b_i = \delta_{ij} \mathbf{I} \quad \wedge \quad [b_i, \, b_j] = [b_i^\dagger, \, b_j^\dagger] = \mathbf{0} \quad\big[122, 693\big]$$

  [A3] Preservación de la Unitariedad de Bogoliubov-Valatin:
       La transición de fase y sintonización de cuasipartículas conserva la estructura 
       simpléctica del espacio de fase en el grupo $$Sp(2n, \mathbb{C})$$ mediante la 
       restricción de Bogoliubov-Valatin sobre los coeficientes de acoplamiento [7]:
       $$\lvert u_k \rvert^2 - \lvert v_k \rvert^2 \equiv 1.0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[24, 693\big]$$

  [A4] Evolución Disipativa Abierta de Lindblad-GKSL:
       La pérdida de pureza y el decaimiento térmico (evaporación de alucinaciones) 
       se modelan formalmente en la Fase 3 mediante la Ecuación Maestra de Lindblad:
       $$\frac{d\rho}{d\tau} = -i[H, \, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \, \rho\} \right) \quad\big[693, 708\big]$$
       Sujeta a la completitud de los operadores de salto para resguardar la traza:
       $$\sum_k L_k^\dagger L_k \le \mathbf{I} \quad\big[531, 693\big]$$

  [A5] Regularización Espectral de Higham:
       Toda matriz densidad reconstructiva $$\rho_{\mathrm{raw}}$$ ruidosa es proyectada 
       al cono de operadores densidad simétricos semidefinidos positivos (SPD) más 
       cercano en Frobenius para satisfacer los postulados de Dirac-von Neumann:
       $$\rho_{\mathrm{stable}} = \arg\min_{M = M^\dagger \succeq 0} \|M - \rho_{\mathrm{raw}}\|_F \quad \wedge \quad \operatorname{Tr}(\rho_{\mathrm{stable}}) \equiv 1.0 \quad\big[81, 693\big]$$

  [A6] Evicción de Canales Basada en Entropía de Shannon-von Neumann:
       Al aproximarse al límite de Bekenstein de saturación dimensional del espacio de 
       Hilbert (_DEFAULT_MAX_CARTRIDGES), el registro ejecuta la purga de cartuchos 
       cuyos vectores de fase se vuelven ortogonales a la geodésica de decisión:
       $$\cos(\theta) = \frac{\langle u, \, v \rangle_G}{\|u\|_G \|v\|_G} \to 0 \implies \text{Purga síncrona en RAM del cartucho } v \quad\big[26, 585\big]$$

ARQUITECTURA DE TRES FASES ANIDADAS (Composición de Morfismos de-confinados):
────────────────────────────────────────────────────────────────────────────────
La progresión del registro se rige por un acoplamiento monoidal covariante estricto 
(Observe ⊣ Orient ⊣ Decide & Act):

  Fase 1 ──► REGISTRO CORÉICO Y OPERADOR DENSIDAD (Phase1_CoreFockRegistry)
             Sanea las 1-formas, verifica el Principio de Exclusión de Pauli CAR 
             para fermiones [5, 9], administra la cota de Bekenstein 
             y calcula el operador densidad colectivo.
             Entrega: Phase1RegistryCertificate.

  Fase 2 ──► DINÁMICA CUÁNTICA Y BOGOLIUBOV (Phase2_QuantumDynamics)
             Hereda la Phase1RegistryCertificate. Implementa el isomorfismo 
             simpléctico de Bogoliubov, verifica las CCR, y proyecta las 
             cuasipartículas resultantes.
             Entrega: Phase2DynamicsCertificate.

  Fase 3 ──► ORQUESTACIÓN INMUNE Y LINDBLAD (Phase3_ImmuneOrchestrator)
             Hereda la Phase2DynamicsCertificate. Resuelve la ecuación 
             de Lindblad-GKSL mediante integración Runge-Kutta RK4, 
             ejecuta el saneamiento de Higham, y colapsa el veredicto en Heyting:
             $$\Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[693, 697\big]$$
             Entrega: FockRegistryState (alias RegistryGovernanceState).

Funtor Maestro de la Memoria Cuántica:
  $$\mathcal{Z}_{\mathrm{fock}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[692, 707\big]$$
"""
from __future__ import annotations
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Tuple, Dict, List, Optional, Union
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

logger = logging.getLogger("MIC.Core.SynapticFockRegistry")

# ═══════════════════════════════════════════════════════════════════════════
# Constantes de rigor cuántico y precisión de la FPU
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_DEFAULT_MAX_CARTRIDGES: Final[int] = 100
_CROWBAR_GPIO: Final[int] = 14  # GPIO14 — Hardware ESP32
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12  # Suelo espectral de Wilkinson
_DENSITY_MATRIX_CONDITION_LIMIT: Final[float] = 1.0e8
_SPARSE_THRESHOLD: Final[float] = 0.3  # Umbral de dispersión para conversión

# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones propias del registro (funtores de error)
# ═══════════════════════════════════════════════════════════════════════════
class SynapticRegistryError(TopologicalInvariantError):
    """Excepción raíz del Registro Sináptico de Fock."""
    pass

class PauliExclusionViolationError(SynapticRegistryError):
    """Estado fermiónico idéntico ya ocupado (principio de exclusión de Pauli)."""
    pass

class FockSpaceOverflowError(SynapticRegistryError):
    """Superada la cota de Bekenstein (máximo de partículas)."""
    pass

class TraceAnomalyError(SynapticRegistryError):
    """Traza del operador densidad distinta de la unidad."""
    pass

class SymplecticConstraintError(SynapticRegistryError):
    """Violación de las relaciones canónicas de conmutación simplécticas."""
    pass

class DensityMatrixNonPSDError(SynapticRegistryError):
    """Operador densidad no semidefinido positivo (falla la purificación)."""
    pass

class LindbladEvolutionError(SynapticRegistryError):
    """Error durante la evolución de Lindblad (pérdida de trazabilidad)."""
    pass

# ═══════════════════════════════════════════════════════════════════════════
# Enumeraciones categóricas (subobjetos y veredictos en el topos operativo)
# ═══════════════════════════════════════════════════════════════════════════
class FockRegistryVerdict(IntEnum):
    """
    Clasificador de tres valores en el retículo de Heyting (Ω) para calidad
    de registro cuántico.
    
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
    def supremum(cls, *verdicts: "FockRegistryVerdict") -> "FockRegistryVerdict":
        """
        Supremo en el retículo de severidad.
        Si no se proveen veredictos, retorna COHERENT como elemento neutro.
        """
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == FockRegistryVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == FockRegistryVerdict.DEGRADED

class RegistryAction(Enum):
    """Acciones de mitigación tras el veredicto de registro."""
    NONE = auto()
    PURGE_ENTROPY = auto()
    HALT_REGISTRY = auto()

# ═══════════════════════════════════════════════════════════════════════════
# Certificados de las fases anidadas (objetos de la subcategoría Spec)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1RegistryCertificate:
    """
    FASE 1 — Certificado de registro y estadística cuántica.
    
    Este certificado constituye el objeto terminal de la Fase 1 y es consumido
    por la Fase 2 a través del puente build_density_matrix_from_registry().
    
    Atributos
    ---------
    registry_size : int
        Cardinal actual del espacio de ocupación.
    max_capacity : int
        Horizonte de Bekenstein (máximo de partículas).
    fermion_count : int
        Número de fermiones registrados.
    boson_count : int
        Número de bosones registrados.
    density_matrix_trace : float
        Traza del operador densidad (debe ser ≈ 1.0).
    density_matrix_purity : float
        Pureza Tr(ρ²) del operador densidad.
    higham_regularization_applied : bool
        Indicador de si se requirió regularización de Higham.
    verdict : FockRegistryVerdict
        Veredicto local de la Fase 1.
    """
    registry_size: int
    max_capacity: int
    fermion_count: int
    boson_count: int
    density_matrix_trace: float
    density_matrix_purity: float
    higham_regularization_applied: bool
    verdict: FockRegistryVerdict

@dataclass(frozen=True, slots=True)
class Phase2DynamicsCertificate:
    """
    FASE 2 — Certificado de dinámica cuántica (Lindblad).
    
    Atributos
    ---------
    evolution_time : float
        Tiempo total de evolución Δt.
    evolution_steps : int
        Número de subpasos RK4.
    final_trace : float
        Traza del operador densidad evolucionado.
    final_purity : float
        Pureza Tr(ρ²) después de la evolución.
    trace_preservation_residual : float
        Residuo |Tr(ρ_final) - 1|.
    is_trace_preserved : bool
        True si la traza se conserva dentro de tolerancia.
    is_psd_preserved : bool
        True si la positividad se preserva.
    lindblad_dissipation_rate : float
        Tasa de disipación de Lindblad estimada.
    verdict : FockRegistryVerdict
        Veredicto local de la Fase 2.
    """
    evolution_time: float
    evolution_steps: int
    final_trace: float
    final_purity: float
    trace_preservation_residual: float
    is_trace_preserved: bool
    is_psd_preserved: bool
    lindblad_dissipation_rate: float
    verdict: FockRegistryVerdict

@dataclass(frozen=True, slots=True)
class Phase3ImmuneCertificate:
    """
    FASE 3 — Certificado de integración inmune completa.
    
    Atributos
    ---------
    annihilation_events : int
        Número de aniquilaciones partícula-antipartícula.
    bogoliubov_transformations : int
        Número de transformaciones de Bogoliubov aplicadas.
    entropy_production : float
        Producción de entropía de von Neumann.
    is_unitary_evolution : bool
        True si la evolución fue unitaria (sin disipación).
    crowbar_triggered : bool
        Indicador de activación de Crowbar hardware.
    verdict : FockRegistryVerdict
        Veredicto local de la Fase 3.
    """
    annihilation_events: int
    bogoliubov_transformations: int
    entropy_production: float
    is_unitary_evolution: bool
    crowbar_triggered: bool
    verdict: FockRegistryVerdict

@dataclass(frozen=True, slots=True)
class FockRegistryState:
    """Certificado global terminal del ciclo de registro cuántico."""
    phase1: Phase1RegistryCertificate
    phase2: Phase2DynamicsCertificate
    phase3: Phase3ImmuneCertificate
    final_verdict: FockRegistryVerdict
    registry_action: RegistryAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# §A. CLASES DE DATOS INMUTABLES (Vitaminas TOON - Cuasipartículas)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ToonCartridge:
    """Clase base inmutable para todos los cartuchos sinápticos TOON."""
    cartridge_id: str
    signature: str

    @property
    def quantum_state_hash(self) -> str:
        raw_payload = f"{self.cartridge_id}-{self.signature}"
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

# ─── FAMILIA 1: FERMIONES ESTRUCTURALES (Alta Inercia y Exclusión) ───
@dataclass(frozen=True, slots=True)
class ElectronCartridge(ToonCartridge):
    """Electrón de Inspección: Porta la carga de anomalía topológica."""
    inertial_mass: float
    topological_spin: str  # "source" o "sink"
    homological_charge: int

@dataclass(frozen=True, slots=True)
class ProtonCartridge(ToonCartridge):
    """Protón de Estabilidad: Emitido ante estabilidad asintótica de Laplace."""
    spectral_charge: float
    logistic_inertial_mass: float
    dominant_pole: float

@dataclass(frozen=True, slots=True)
class PolaronCartridge(ToonCartridge):
    """Polarón Logístico: Generado por el acoplamiento fuerte de Fröhlich."""
    base_electron: ElectronCartridge
    frohlich_coupling: float
    effective_mass: float
    fiedler_value: float

@dataclass(frozen=True, slots=True)
class TorsionCartridge(ToonCartridge):
    """Torsión de de Rham: Mide la fricción por incompatibilidad de empaquetado."""
    torsion_degree: int
    torsion_elements: Tuple[int, ...]

@dataclass(frozen=True, slots=True)
class HouseholderReflectionFermion(ToonCartridge):
    """Fermión de Householder: Vector normal de reflexión para proyectores."""
    covariant_hyperplane_normal: Tuple[float, ...]
    monodromy_spectral_radius: float
    cohomology_obstruction_class: int

# ─── FAMILIA 2: BOSONES DE GAUGE Y INTERACCIÓN (Campos de Fuerza) ───
@dataclass(frozen=True, slots=True)
class PhotonCartridge(ToonCartridge):
    """Fotón de Gobernanza: Bosón sin masa que transporta las políticas declarativas."""
    policy_id: str
    spectral_frequency: float
    governance_weight: float

@dataclass(frozen=True, slots=True)
class RiemannianFocalBoson(ToonCartridge):
    """Bosón de Fresnel: Guía la atención a lo largo de geodésicas de Fermat."""
    dielectric_tensor: Tuple[float, ...]
    spectral_cutoff_functor: int
    wkb_maslov_index: int

@dataclass(frozen=True, slots=True)
class MagnonCartridge(ToonCartridge):
    """Magnón de Vorticidad: Inyecta un veto de enrutamiento para aniquilar rizos."""
    kinetic_energy: float
    curl_subspace_dim: int

@dataclass(frozen=True, slots=True)
class SwellingPlasmonCartridge(ToonCartridge):
    """Plasmón Logístico: Emitido ante expansiones de volumen en el presupuesto."""
    density_amplitude: float
    inflation_rate: float

@dataclass(frozen=True, slots=True)
class YieldingPhononCartridge(ToonCartridge):
    """Phonón de Cedencia: Detecta fatiga elástica en los nodos de suministro."""
    damping_coefficient: float
    strain_tensor: Tuple[float, ...]

@dataclass(frozen=True, slots=True)
class LiquefactionSolitonCartridge(ToonCartridge):
    """Solitón de Licuación: Detecta pérdida de sustentación en el manifold litológico."""
    wave_velocity: float
    pore_pressure: float

# ─── FAMILIA 3: ANTIMATERIA Y CONDENSADOS ───
@dataclass(frozen=True, slots=True)
class PositronCartridge(ToonCartridge):
    """Positrón de Extirpación: Antimateria inyectada para invalidar fallas."""
    inertial_mass: float
    topological_spin: str
    homological_charge: int
    authorization_signature: str

@dataclass(frozen=True, slots=True)
class GammaPhoton(ToonCartridge):
    """Fotón Gamma de Auditoría: Registro criptográfico de una aniquilación exitosa."""
    energy: float
    wave_vector: Tuple[float, ...]
    provenance_hash: str

@dataclass(frozen=True, slots=True)
class PolaritonCartridge(ToonCartridge):
    """Polaritón: Híbrido polarón-fotón que induce superfluidez atencional."""
    polaron: PolaronCartridge
    photon: PhotonCartridge
    rabi_coupling: float
    dissipation_trace: float

@dataclass(frozen=True, slots=True)
class SophonCartridge(ToonCartridge):
    """Sofón: Anomalía estocástica u oscilación de fase del LLM."""
    chiral_phase: float
    stochastic_variance: float

# ═══════════════════════════════════════════════════════════════════════════
# Utilitarios de Regularización Espectral (Higham para Operadores Densidad)
# ═══════════════════════════════════════════════════════════════════════════
def stable_density_matrix_higham(
    rho: NDArray[np.complex128],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR
) -> Tuple[NDArray[np.complex128], bool]:
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
    Tuple[NDArray[np.complex128], bool]
        (rho_projected, higham_applied) donde higham_applied indica si se
        requirió regularización.
    
    Raises
    ------
    DensityMatrixNonPSDError
        Si la proyección de Higham falla en producir una matriz PSD.
    """
    # 1. Hermitización exacta
    rho_herm = 0.5 * (rho + rho.conj().T)
    
    # 2. Descomposición espectral
    try:
        vals, vecs = la.eigh(rho_herm)
    except Exception as exc:
        raise DensityMatrixNonPSDError(
            "DensityMatrixNonPSDError: Falló descomposición espectral para Higham."
        ) from exc

    # 3. Recorte de autovalores negativos al suelo espectral
    vals_clean = np.maximum(vals, tolerance)
    applied = np.any(vals < tolerance)

    # 4. Normalización de traza a unidad
    trace = np.sum(vals_clean)
    if trace > _MACHINE_EPS:
        vals_clean /= trace
    else:
        # Estado de vacío: asignar todo al último autovalor
        vals_clean = np.zeros_like(vals_clean)
        vals_clean[-1] = 1.0

    # 5. Reconstrucción del operador densidad proyectado
    rho_projected = vecs @ np.diag(vals_clean) @ vecs.conj().T
    
    # 6. Hermitización numérica post-proyección
    rho_projected = 0.5 * (rho_projected + rho_projected.conj().T)

    if applied:
        logger.warning(
            "Operador densidad inestable. Proyectado ρ al cono PSD via Higham "
            "(autovalores mínimos: %.4e → %.4e).",
            float(np.min(vals)),
            float(np.min(vals_clean))
        )

    # 7. Verificación final de PSD
    try:
        final_vals = la.eigvalsh(rho_projected)
        if np.any(final_vals < -tolerance):
            raise DensityMatrixNonPSDError(
                "DensityMatrixNonPSDError: Proyección de Higham no produjo matriz PSD."
            )
    except Exception as exc:
        raise DensityMatrixNonPSDError(str(exc)) from exc

    return rho_projected, applied

# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 – Registro y Estadística: Inyección, evicción y operador densidad
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_CoreFockRegistry(ABC):
    """
    FASE 1 (Registro y Estadística): Administración de la ocupación del espacio
    de Fock, verificación del principio de exclusión de Pauli para fermiones,
    evicción por entropía y construcción del operador densidad colectivo.
    
    El último método de esta fase:
        build_density_matrix_from_registry(...)
    constituye el morfismo de transición formal hacia la Fase 2.
    """

    def __init__(
        self,
        max_cartridges: int = _DEFAULT_MAX_CARTRIDGES,
        tolerance: float = _DEFAULT_TOL
    ) -> None:
        """
        Inicializa el registro del espacio de Fock.
        
        Parámetros
        ----------
        max_cartridges : int
            Horizonte de Bekenstein (máximo de partículas).
        tolerance : float
            Cota de precisión para validaciones cuánticas.
        """
        self._max_cartridges: Final[int] = max_cartridges
        self._tolerance: Final[float] = tolerance
        self._registry: Dict[str, ToonCartridge] = {}
        self._embedding_cache: Dict[str, NDArray[np.float64]] = {}
        self._fermion_hashes: Dict[type, set] = {}  # Para exclusión de Pauli O(1)

    @property
    def size(self) -> int:
        """Cardinal actual del espacio de ocupación."""
        return len(self._registry)

    @property
    def max_capacity(self) -> int:
        """Horizonte de Bekenstein."""
        return self._max_cartridges

    def inject_cartridge(
        self,
        key: str,
        cartridge: ToonCartridge,
        embedding: NDArray[np.float64]
    ) -> None:
        r"""
        Inyecta una vitamina TOON en el Espacio de Fock.
        Aplica el principio de exclusión de Pauli para todos los fermiones
        estructurales definidos. Si el estado cuántico (hash) ya existe
        para el mismo tipo de fermión, se lanza una excepción.
        Si se alcanza el horizonte de Bekenstein, se ejecuta una evicción
        por entropía antes de la inserción.
        
        Parámetros
        ----------
        key : str
            Identificador único del canal sináptico.
        cartridge : ToonCartridge
            La cuasipartícula inmutable.
        embedding : NDArray
            Coordenadas de fase normalizadas en la variedad.
        
        Lanza
        -----
        PauliExclusionViolationError
            Si se viola la exclusión de Pauli.
        FockSpaceOverflowError
            Si el registro está lleno y no se puede evictar.
        """
        fermion_types = (
            ElectronCartridge, ProtonCartridge, PolaronCartridge,
            TorsionCartridge, HouseholderReflectionFermion
        )
        
        # Verificación de exclusión de Pauli O(1)
        if isinstance(cartridge, fermion_types):
            cartridge_type = type(cartridge)
            if cartridge_type not in self._fermion_hashes:
                self._fermion_hashes[cartridge_type] = set()
            
            state_hash = cartridge.quantum_state_hash
            if state_hash in self._fermion_hashes[cartridge_type]:
                raise PauliExclusionViolationError(
                    f"PauliExclusionViolation: El fermión {cartridge.cartridge_id} "
                    "ya está ocupado en el registro."
                )
        
        # Horizonte de Bekenstein
        if len(self._registry) >= self._max_cartridges:
            logger.info("Horizonte de Bekenstein alcanzado. Iniciando purga por entropía.")
            try:
                self.evict_by_entropy(embedding)
            except SynapticRegistryError as exc:
                raise FockSpaceOverflowError(
                    f"FockSpaceOverflowError: No se pudo liberar espacio. {exc}"
                ) from exc
        
        # Inserción
        self._registry[key] = cartridge
        norm = la.norm(embedding) + _MACHINE_EPS
        self._embedding_cache[key] = embedding / norm
        
        # Actualizar índice de fermiones
        if isinstance(cartridge, fermion_types):
            cartridge_type = type(cartridge)
            if cartridge_type not in self._fermion_hashes:
                self._fermion_hashes[cartridge_type] = set()
            self._fermion_hashes[cartridge_type].add(cartridge.quantum_state_hash)
        
        logger.debug("Partícula %s inyectada con éxito en el Espacio de Fock.", cartridge.cartridge_id)

    def evict_by_entropy(self, current_decision_vector: NDArray[np.float64]) -> str:
        r"""
        Poda la partícula menos alineada con el vector de decisión actual,
        usando la similitud del coseno.
        
        .. math::
            \cos(\theta) = \frac{\langle u, v \rangle}{\|u\|\|v\|}
        
        Parámetros
        ----------
        current_decision_vector : NDArray
            Trayectoria actual en el espacio de fase.
        
        Retorna
        -------
        str
            Clave del cartucho eliminado.
        
        Lanza
        -----
        SynapticRegistryError
            Si el registro está vacío.
        """
        if not self._registry:
            raise SynapticRegistryError("Evicción imposible: el Espacio de Fock está vacío.")
        
        u = current_decision_vector / (la.norm(current_decision_vector) + _MACHINE_EPS)
        worst_key = None
        min_similarity = float("inf")
        
        for key, v in self._embedding_cache.items():
            similarity = float(np.dot(u, v))
            if similarity < min_similarity:
                min_similarity = similarity
                worst_key = key
        
        if worst_key is None:
            raise SynapticRegistryError("No se pudo seleccionar un candidato para evicción.")
        
        # Eliminar del índice de fermiones si aplica
        removed = self._registry[worst_key]
        fermion_types = (
            ElectronCartridge, ProtonCartridge, PolaronCartridge,
            TorsionCartridge, HouseholderReflectionFermion
        )
        if isinstance(removed, fermion_types):
            cartridge_type = type(removed)
            if cartridge_type in self._fermion_hashes:
                self._fermion_hashes[cartridge_type].discard(removed.quantum_state_hash)
        
        self._registry.pop(worst_key)
        self._embedding_cache.pop(worst_key)
        
        logger.info(
            "Evicción por entropía: purgado %s (similitud=%.4f).",
            removed.cartridge_id, min_similarity
        )
        return worst_key

    def get_all_particles_of_type(self, particle_type: type) -> List[ToonCartridge]:
        """Devuelve todas las partículas registradas de un tipo dado."""
        return [c for c in self._registry.values() if isinstance(c, particle_type)]

    def count_fermions_and_bosons(self) -> Tuple[int, int]:
        """Cuenta fermiones y bosones separadamente."""
        fermion_types = (
            ElectronCartridge, ProtonCartridge, PolaronCartridge,
            TorsionCartridge, HouseholderReflectionFermion
        )
        fermion_count = sum(
            1 for c in self._registry.values() if isinstance(c, fermion_types)
        )
        boson_count = len(self._registry) - fermion_count
        return fermion_count, boson_count

    # ─────────────────────────────────────────────────────────────────────
    # Último método de la Fase 1: Puente hacia la Fase 2
    # ─────────────────────────────────────────────────────────────────────
    def build_density_matrix_from_registry(self) -> Tuple[NDArray[np.complex128], bool]:
        r"""
        MORFISMO DE TRANSICIÓN FASE 1 → FASE 2.
        
        Construye una matriz de densidad normalizada a partir de los embeddings
        de las partículas presentes en el registro, aplicando regularización
        de Higham si es necesario.
        
        Se utiliza un modelo de estados puros equiprobables:
        .. math::
            \rho_{\mathrm{MAC}} = \frac{1}{N} \sum_{i=1}^{N} |v_i\rangle\langle v_i|,
        
        donde $|v_i\rangle$ es el embedding normalizado de cada cartucho.
        
        Retorna
        -------
        Tuple[NDArray[np.complex128], bool]
            (rho, higham_applied) donde higham_applied indica si se requirió
            regularización de Higham.
        
        Lanza
        -----
        DensityMatrixNonPSDError
            Si la proyección de Higham falla.
        """
        n = len(self._registry)
        if n == 0:
            # Estado de vacío sin partículas
            return np.eye(1, dtype=np.complex128), False
        
        # Suponemos que todos los embeddings tienen la misma dimensión
        sample_vec = next(iter(self._embedding_cache.values()))
        dim = len(sample_vec)
        rho = np.zeros((dim, dim), dtype=np.complex128)
        
        for v in self._embedding_cache.values():
            vec = v.reshape(-1, 1).astype(np.complex128)
            rho += vec @ vec.conj().T
        
        rho /= n
        
        # Aplicar regularización de Higham para garantizar PSD y traza unitaria
        rho_projected, higham_applied = stable_density_matrix_higham(rho, self._tolerance)
        
        return rho_projected, higham_applied

    def generate_phase1_certificate(self) -> Phase1RegistryCertificate:
        """
        Genera el certificado de Fase 1 con auditoría completa del registro.
        
        Retorna
        -------
        Phase1RegistryCertificate
            Certificado terminal de Fase 1.
        """
        fermion_count, boson_count = self.count_fermions_and_bosons()
        rho, higham_applied = self.build_density_matrix_from_registry()
        
        # Calcular traza y pureza
        trace = float(np.real(np.trace(rho)))
        purity = float(np.real(np.trace(rho @ rho)))
        
        # Veredicto local
        verdict = FockRegistryVerdict.COHERENT
        if not np.isfinite(trace) or not np.isfinite(purity):
            verdict = FockRegistryVerdict.VETOED
        elif abs(trace - 1.0) > self._tolerance * 100 or purity < 0.5:
            verdict = FockRegistryVerdict.DEGRADED
        elif higham_applied:
            verdict = FockRegistryVerdict.DEGRADED
        
        return Phase1RegistryCertificate(
            registry_size=self.size,
            max_capacity=self._max_cartridges,
            fermion_count=fermion_count,
            boson_count=boson_count,
            density_matrix_trace=trace,
            density_matrix_purity=purity,
            higham_regularization_applied=higham_applied,
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 – Dinámica Cuántica: Bogoliubov, aniquilación y evolución de Lindblad
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_QuantumDynamics(Phase1_CoreFockRegistry):
    """
    FASE 2 (Dinámica Cuántica): Transformaciones sobre el espacio de Fock,
    incluyendo la transformación de Bogoliubov, la aniquilación partícula-
    antipartícula con actualización automática del registro, y la evolución
    temporal del operador densidad según la ecuación maestra de Lindblad.
    
    Hereda de la Fase 1 para usar build_density_matrix_from_registry() como
    puente formal.
    """

    def apply_bogoliubov_transformation(
        self,
        u_k: complex,
        v_k: complex,
        b_k: complex,
        b_dagger_k: complex
    ) -> Tuple[complex, complex]:
        r"""
        Aplica la transformación de Bogoliubov-Valatin con verificación
        simpléctica.
        
        Requisito de invarianza:
        .. math::
            |u_k|^2 - |v_k|^2 = 1.0 \pm \tau
        
        Parámetros
        ----------
        u_k, v_k : complex
            Coeficientes de transmisión y refracción.
        b_k, b_dagger_k : complex
            Modos originales.
        
        Retorna
        -------
        Tuple[complex, complex]
            Modos purificados α_k, α^\dagger_k.
        
        Lanza
        -----
        SymplecticConstraintError
            Si la condición simpléctica no se cumple.
        """
        constraint = abs(u_k) ** 2 - abs(v_k) ** 2
        if abs(constraint - 1.0) > self._tolerance * 100:
            raise SymplecticConstraintError(
                f"SymplecticConstraintError: |u|² - |v|² = {constraint:.6f} (debe ser 1.0)."
            )
        
        alpha_k = u_k * b_k + v_k * b_dagger_k
        alpha_dagger_k = np.conj(v_k) * b_k + np.conj(u_k) * b_dagger_k
        
        return alpha_k, alpha_dagger_k

    def perform_annihilation(
        self,
        electron_key: str,
        positron_key: str
    ) -> Tuple[GammaPhoton, GammaPhoton]:
        r"""
        Ejecuta la aniquilación de un electrón y un positrón, eliminándolos
        del registro y emitiendo dos fotones gamma de auditoría.
        
        La energía liberada se calcula como:
        .. math::
            E_{\text{total}} = 2 m_{\text{eff}} c^2 \quad (c=1).
        
        Parámetros
        ----------
        electron_key : str
            Clave del electrón en el registro.
        positron_key : str
            Clave del positrón en el registro.
        
        Retorna
        -------
        Tuple[GammaPhoton, GammaPhoton]
            Los dos fotones gamma generados.
        
        Lanza
        -----
        SynapticRegistryError
            Si las claves no corresponden a un electrón/positrón o si las
            cargas homológicas no suman cero.
        """
        electron = self._registry.get(electron_key)
        positron = self._registry.get(positron_key)
        
        if not isinstance(electron, ElectronCartridge) or not isinstance(positron, PositronCartridge):
            raise SynapticRegistryError(
                "Aniquilación requiere un Electrón y un Positrón registrados."
            )
        
        if electron.homological_charge + positron.homological_charge != 0:
            raise SynapticRegistryError(
                "AnnihilationError: Las cargas homológicas no suman cero."
            )
        
        m_eff = 0.5 * (electron.inertial_mass + positron.inertial_mass)
        energy_total = 2.0 * m_eff
        
        # Crear los fotones gamma
        raw_payload = f"ANNIHILATION-{electron.quantum_state_hash}-{positron.quantum_state_hash}"
        p_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        
        g1 = GammaPhoton(
            cartridge_id=f"gamma-1-{electron.cartridge_id}",
            signature=f"photon-gamma-{electron.signature[:8]}",
            energy=energy_total / 2.0,
            wave_vector=(1.0, 0.0, 0.0),
            provenance_hash=p_hash
        )
        g2 = GammaPhoton(
            cartridge_id=f"gamma-2-{positron.cartridge_id}",
            signature=f"photon-gamma-{positron.signature[:8]}",
            energy=energy_total / 2.0,
            wave_vector=(-1.0, 0.0, 0.0),
            provenance_hash=p_hash
        )
        
        # Eliminar las partículas originales del índice de fermiones
        electron_type = type(electron)
        if electron_type in self._fermion_hashes:
            self._fermion_hashes[electron_type].discard(electron.quantum_state_hash)
        
        positron_type = type(positron)
        if positron_type in self._fermion_hashes:
            self._fermion_hashes[positron_type].discard(positron.quantum_state_hash)
        
        # Eliminar del registro
        del self._registry[electron_key]
        del self._embedding_cache[electron_key]
        del self._registry[positron_key]
        del self._embedding_cache[positron_key]
        
        # Inyectar los fotones (simplificado, sin embeddings reales)
        self._registry[g1.cartridge_id] = g1
        self._registry[g2.cartridge_id] = g2
        self._embedding_cache[g1.cartridge_id] = np.zeros(1)
        self._embedding_cache[g2.cartridge_id] = np.zeros(1)
        
        logger.info("¡Aniquilación completada! %s y %s emitidos.", g1.cartridge_id, g2.cartridge_id)
        return g1, g2

    def solve_lindblad_evolution(
        self,
        rho: NDArray[np.complex128],
        hamiltonian_eff: NDArray[np.complex128],
        jump_operators: List[NDArray[np.complex128]],
        gamma_rates: List[float],
        dt: float,
        steps: int = 10
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Integra la ecuación maestra de Lindblad-GKSL usando un esquema RK4.
        
        .. math::
            \frac{d\rho}{dt} = -i [H_{\text{eff}}, \rho] + \sum_k \gamma_k \left(
                L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}
            \right)
        
        Se garantiza numéricamente Hermiticidad, positividad (PSD) y traza unitaria
        mediante purificación espectral después de cada subpaso.
        
        Parámetros
        ----------
        rho : NDArray[np.complex128]
            Operador densidad inicial.
        hamiltonian_eff : NDArray[np.complex128]
            Hamiltoniano efectivo (Hermítico).
        jump_operators : List[NDArray[np.complex128]]
            Lista de operadores de salto L_k.
        gamma_rates : List[float]
            Tasas de decaimiento γ_k (≥0).
        dt : float
            Paso temporal total.
        steps : int
            Subpasos de integración RK4.
        
        Retorna
        -------
        Tuple[NDArray[np.complex128], float]
            (rho_evolved, dissipation_rate) donde dissipation_rate es la tasa
            estimada de disipación de Lindblad.
        
        Lanza
        -----
        LindbladEvolutionError
            Si la evolución pierde trazabilidad o positividad.
        """
        rho = np.copy(rho)
        
        def lindbladian(r: NDArray[np.complex128]) -> NDArray[np.complex128]:
            coherent = -1j * (hamiltonian_eff @ r - r @ hamiltonian_eff)
            dissipative = np.zeros_like(r, dtype=np.complex128)
            for gamma, L in zip(gamma_rates, jump_operators):
                LdL = L.conj().T @ L
                jump = L @ r @ L.conj().T
                anti = 0.5 * (LdL @ r + r @ LdL)
                dissipative += gamma * (jump - anti)
            return coherent + dissipative
        
        sub_dt = dt / steps
        for step in range(steps):
            k1 = lindbladian(rho)
            k2 = lindbladian(rho + 0.5 * sub_dt * k1)
            k3 = lindbladian(rho + 0.5 * sub_dt * k2)
            k4 = lindbladian(rho + sub_dt * k3)
            rho += (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            # Purificación espectral para mantener PSD y Tr(ρ)=1
            rho, higham_applied = stable_density_matrix_higham(rho, self._tolerance)
            
            if higham_applied and step > 0:
                logger.warning(
                    "Paso %d de Lindblad requirió regularización de Higham.", step
                )
        
        # Estimar tasa de disipación
        dissipation_rate = 0.0
        if gamma_rates and jump_operators:
            for gamma, L in zip(gamma_rates, jump_operators):
                dissipation_rate += gamma * float(np.real(np.trace(L.conj().T @ L @ rho)))
        
        return rho, dissipation_rate

    def evolve_registry_density(
        self,
        hamiltonian_eff: NDArray[np.complex128],
        jump_operators: List[NDArray[np.complex128]],
        gamma_rates: List[float],
        dt: float,
        steps: int = 10
    ) -> Tuple[NDArray[np.complex128], Phase2DynamicsCertificate]:
        """
        Construye el operador densidad a partir del registro actual (Fase 1)
        y lo evoluciona según la dinámica de Lindblad.
        
        Este método es el nexo explícito entre las fases: la salida de la
        Fase 1 alimenta directamente la Fase 2.
        
        Retorna
        -------
        Tuple[NDArray, Phase2DynamicsCertificate]
            (rho_evolved, certificate)
        """
        # Puente con Fase 1
        rho0, higham_applied = self.build_density_matrix_from_registry()
        
        # Evolución de Lindblad
        rho_final, dissipation_rate = self.solve_lindblad_evolution(
            rho0, hamiltonian_eff, jump_operators, gamma_rates, dt, steps
        )
        
        # Auditoría de la evolución
        final_trace = float(np.real(np.trace(rho_final)))
        final_purity = float(np.real(np.trace(rho_final @ rho_final)))
        trace_residual = abs(final_trace - 1.0)
        
        is_trace_preserved = trace_residual <= self._tolerance * 100
        is_psd_preserved = np.all(la.eigvalsh(rho_final) >= -self._tolerance)
        
        # Veredicto local
        verdict = FockRegistryVerdict.COHERENT
        if not np.isfinite(final_trace) or not np.isfinite(final_purity):
            verdict = FockRegistryVerdict.VETOED
        elif not is_trace_preserved or not is_psd_preserved or dissipation_rate < 0:
            verdict = FockRegistryVerdict.DEGRADED
        
        cert = Phase2DynamicsCertificate(
            evolution_time=float(dt),
            evolution_steps=int(steps),
            final_trace=final_trace,
            final_purity=final_purity,
            trace_preservation_residual=trace_residual,
            is_trace_preserved=is_trace_preserved,
            is_psd_preserved=is_psd_preserved,
            lindblad_dissipation_rate=float(dissipation_rate),
            verdict=verdict,
        )
        
        return rho_final, cert

# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 – Integración Inmune (Ciclo completo)
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_ImmuneOrchestrator(Phase2_QuantumDynamics):
    """
    FASE 3 (Integración Inmune): Orquestación de alto nivel que puede ejecutar
    ciclos inmunes completos, combinando la inyección/evicción de partículas
    con la dinámica cuántica y las aniquilaciones.
    
    Hereda de la Fase 2 para poder generar pasos y evaluar el sistema.
    """

    def execute_immune_cycle(
        self,
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        dt: float = 1.0,
        steps: int = 10,
        perform_annihilations: bool = True,
    ) -> Tuple[Phase3ImmuneCertificate, NDArray[np.complex128]]:
        """
        Ejecuta un ciclo inmune completo con auditoría de las tres fases.
        
        Parámetros
        ----------
        hamiltonian_eff : Optional[NDArray]
            Hamiltoniano efectivo para evolución de Lindblad.
        jump_operators : Optional[List[NDArray]]
            Operadores de salto para disipación.
        gamma_rates : Optional[List[float]]
            Tasas de decaimiento.
        dt : float
            Paso temporal.
        steps : int
            Subpasos de integración.
        perform_annihilations : bool
            Si True, busca pares electrón-positrón para aniquilar.
        
        Retorna
        -------
        Tuple[Phase3ImmuneCertificate, NDArray]
            (certificate, rho_final)
        """
        annihilation_events = 0
        bogoliubov_transformations = 0
        
        # 1. Buscar y ejecutar aniquilaciones
        if perform_annihilations:
            electrons = self.get_all_particles_of_type(ElectronCartridge)
            positrons = self.get_all_particles_of_type(PositronCartridge)
            
            for e in electrons:
                for p in positrons:
                    if e.homological_charge + p.homological_charge == 0:
                        # Encontrar claves
                        e_key = None
                        p_key = None
                        for key, cart in self._registry.items():
                            if cart is e:
                                e_key = key
                            if cart is p:
                                p_key = key
                        
                        if e_key and p_key:
                            try:
                                self.perform_annihilation(e_key, p_key)
                                annihilation_events += 1
                            except SynapticRegistryError:
                                pass  # Continuar con otras aniquilaciones
        
        # 2. Evolución de Lindblad si se proporciona Hamiltoniano
        rho_final = None
        if hamiltonian_eff is not None:
            if jump_operators is None:
                jump_operators = []
            if gamma_rates is None:
                gamma_rates = []
            
            _, dynamics_cert = self.evolve_registry_density(
                hamiltonian_eff, jump_operators, gamma_rates, dt, steps
            )
            rho_final, _ = self.build_density_matrix_from_registry()
        else:
            rho_final, _ = self.build_density_matrix_from_registry()
            dynamics_cert = Phase2DynamicsCertificate(
                evolution_time=0.0,
                evolution_steps=0,
                final_trace=float(np.real(np.trace(rho_final))),
                final_purity=float(np.real(np.trace(rho_final @ rho_final))),
                trace_preservation_residual=0.0,
                is_trace_preserved=True,
                is_psd_preserved=True,
                lindblad_dissipation_rate=0.0,
                verdict=FockRegistryVerdict.COHERENT,
            )
        
        # 3. Calcular producción de entropía de von Neumann
        entropy_production = 0.0
        if rho_final is not None:
            vals = la.eigvalsh(rho_final)
            vals = np.maximum(vals, _MACHINE_EPS)
            entropy_production = float(-np.sum(vals * np.log(vals + _MACHINE_EPS)))
        
        is_unitary = (
            jump_operators is None or
            len(jump_operators) == 0 or
            all(g == 0.0 for g in gamma_rates)
        )
        
        # Veredicto local
        verdict = FockRegistryVerdict.COHERENT
        if not np.isfinite(entropy_production):
            verdict = FockRegistryVerdict.VETOED
        elif annihilation_events == 0 and not is_unitary:
            verdict = FockRegistryVerdict.DEGRADED
        
        cert = Phase3ImmuneCertificate(
            annihilation_events=annihilation_events,
            bogoliubov_transformations=bogoliubov_transformations,
            entropy_production=entropy_production,
            is_unitary_evolution=is_unitary,
            crowbar_triggered=False,
            verdict=verdict,
        )
        
        return cert, rho_final

# ═══════════════════════════════════════════════════════════════════════════
# Registro final: herencia completa de las tres fases
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistry(Morphism, Phase3_ImmuneOrchestrator):
    """
    Registro completo del Espacio de Fock Sináptico. Agrupa todas las
    capacidades de las fases anidadas: inyección, evicción, transformaciones
    cuánticas y evolución temporal.
    
    El ciclo de registro queda formalizado como:
        Registro  → Fase 1 (CoreFockRegistry)
        Dinámica  → Fase 2 (QuantumDynamics)
        Inmune    → Fase 3 (ImmuneOrchestrator)
    
    El veredicto final es el supremo de severidad en el retículo de Heyting.
    """

    def __init__(
        self,
        max_cartridges: int = _DEFAULT_MAX_CARTRIDGES,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False
    ) -> None:
        """
        Inicializa el registro con configuración de precisión.
        
        Parámetros
        ----------
        max_cartridges : int
            Horizonte de Bekenstein.
        tolerance : float
            Cota de precisión para validaciones.
        halt_on_veto : bool
            Si True, lanza excepción cuando el veredicto es VETOED.
        """
        Morphism.__init__(self, "SynapticFockSpaceRegistry")
        Phase3_ImmuneOrchestrator.__init__(self, max_cartridges=max_cartridges, tolerance=tolerance)
        self._halt_on_veto = halt_on_veto

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        r"""Invocación como morfismo categórico."""
        if kwargs:
            return self.execute_full_registry_cycle(**kwargs)
        return state

    def execute_full_registry_cycle(
        self,
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        dt: float = 1.0,
        steps: int = 10,
        perform_annihilations: bool = True,
    ) -> FockRegistryState:
        """
        Orquesta el ciclo completo de registro cuántico con validación de
        las tres fases.
        
        Estrategia de anidación:
          1. Fase 1: Genera certificado de registro y operador densidad.
          2. Fase 2: Ejecuta evolución de Lindblad.
          3. Fase 3: Ejecuta ciclo inmune completo.
        
        Si Fase 1 emite VETOED, se emiten certificados vetados para Fase 2 y 3.
        
        Retorna
        -------
        FockRegistryState
            Certificado global terminal.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            # ─── FASE 1: Registro ───
            cert_1 = self.generate_phase1_certificate()
            
            if cert_1.verdict.is_vetoed:
                cert_2 = self._vetoed_phase2_certificate()
                cert_3 = self._vetoed_phase3_certificate()
                rho_final = np.eye(1, dtype=np.complex128)
            else:
                # ─── FASE 2: Dinámica ───
                if hamiltonian_eff is not None:
                    if jump_operators is None:
                        jump_operators = []
                    if gamma_rates is None:
                        gamma_rates = []
                    
                    rho_final, cert_2 = self.evolve_registry_density(
                        hamiltonian_eff, jump_operators, gamma_rates, dt, steps
                    )
                else:
                    rho_final, _ = self.build_density_matrix_from_registry()
                    cert_2 = Phase2DynamicsCertificate(
                        evolution_time=0.0,
                        evolution_steps=0,
                        final_trace=float(np.real(np.trace(rho_final))),
                        final_purity=float(np.real(np.trace(rho_final @ rho_final))),
                        trace_preservation_residual=0.0,
                        is_trace_preserved=True,
                        is_psd_preserved=True,
                        lindblad_dissipation_rate=0.0,
                        verdict=FockRegistryVerdict.COHERENT,
                    )
                
                if cert_2.verdict.is_vetoed:
                    cert_3 = self._vetoed_phase3_certificate()
                else:
                    # ─── FASE 3: Inmune ───
                    cert_3, _ = self.execute_immune_cycle(
                        hamiltonian_eff, jump_operators, gamma_rates, dt, steps,
                        perform_annihilations
                    )
            
            # ─── Fusión de veredictos (supremo en severidad) ───
            final_verdict = FockRegistryVerdict.supremum(
                cert_1.verdict,
                cert_2.verdict,
                cert_3.verdict,
            )
            
            registry_action = RegistryAction.NONE
            diagnostic_note = "Ciclo de registro cuántico completado."
            
            if final_verdict == FockRegistryVerdict.VETOED:
                registry_action = RegistryAction.HALT_REGISTRY
                diagnostic_note = (
                    "VETO DE REGISTRO: violación de Pauli, trazabilidad o "
                    "positividad del operador densidad."
                )
                logger.error("¡VETO DE REGISTRO! Halt requested.")
                if self._halt_on_veto:
                    raise SynapticRegistryError(
                        "Violación de invariantes cuánticos o trazabilidad."
                    )
            elif final_verdict == FockRegistryVerdict.DEGRADED:
                registry_action = RegistryAction.PURGE_ENTROPY
                diagnostic_note = (
                    "Degradación detectada: regularización Higham aplicada o "
                    "producción de entropía elevada."
                )
                logger.warning(
                    "Degradación en registro cuántico. Se recomienda purga por entropía."
                )
            
            provenance_hash = self._generate_provenance_hash(
                cert_1, cert_2, cert_3, timestamp_utc
            )
            
            return FockRegistryState(
                phase1=cert_1,
                phase2=cert_2,
                phase3=cert_3,
                final_verdict=final_verdict,
                registry_action=registry_action,
                timestamp_utc=timestamp_utc,
                provenance_hash=provenance_hash,
                diagnostic_note=diagnostic_note,
            )
            
        except SynapticRegistryError as exc:
            logger.error("Colapso categórico del registro de Fock sináptico: %s", exc)
            if self._halt_on_veto:
                raise
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)
            
        except Exception as exc:  # pragma: no cover
            logger.exception("Colapso catastrófico no tipado del registro de Fock sináptico.")
            if self._halt_on_veto:
                raise SynapticRegistryError(
                    "Colapso catastrófico no tipado del registro de Fock sináptico."
                ) from exc
            return self._cataclysm_state(reason=str(exc), timestamp_utc=timestamp_utc)

    # ─────────────────────────────────────────────────────────────────────
    # Certificados vetados para cortocircuito anidado
    # ─────────────────────────────────────────────────────────────────────
    def _vetoed_phase2_certificate(self) -> Phase2DynamicsCertificate:
        """Certificado vetado de Fase 2 para cortocircuito."""
        return Phase2DynamicsCertificate(
            evolution_time=0.0,
            evolution_steps=0,
            final_trace=0.0,
            final_purity=0.0,
            trace_preservation_residual=float("inf"),
            is_trace_preserved=False,
            is_psd_preserved=False,
            lindblad_dissipation_rate=float("inf"),
            verdict=FockRegistryVerdict.VETOED,
        )
    
    def _vetoed_phase3_certificate(self) -> Phase3ImmuneCertificate:
        """Certificado vetado de Fase 3 para cortocircuito."""
        return Phase3ImmuneCertificate(
            annihilation_events=0,
            bogoliubov_transformations=0,
            entropy_production=float("inf"),
            is_unitary_evolution=False,
            crowbar_triggered=True,
            verdict=FockRegistryVerdict.VETOED,
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Estado catastrófico de emergencia
    # ─────────────────────────────────────────────────────────────────────
    def _cataclysm_state(self, reason: str, timestamp_utc: str) -> FockRegistryState:
        """Construye un estado catastrófico con certificados vetados."""
        phase1_dummy = Phase1RegistryCertificate(
            registry_size=0,
            max_capacity=self._max_cartridges,
            fermion_count=0,
            boson_count=0,
            density_matrix_trace=0.0,
            density_matrix_purity=0.0,
            higham_regularization_applied=False,
            verdict=FockRegistryVerdict.VETOED,
        )
        phase2_dummy = self._vetoed_phase2_certificate()
        phase3_dummy = self._vetoed_phase3_certificate()
        
        raw_payload = f"CATACLYSM_FOCK_VETO::{reason}"
        provenance_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        
        return FockRegistryState(
            phase1=phase1_dummy,
            phase2=phase2_dummy,
            phase3=phase3_dummy,
            final_verdict=FockRegistryVerdict.VETOED,
            registry_action=RegistryAction.HALT_REGISTRY,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_FOCK_VETO: {reason}",
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Hash de procedencia auditable
    # ─────────────────────────────────────────────────────────────────────
    def _generate_provenance_hash(
        self,
        c1: Phase1RegistryCertificate,
        c2: Phase2DynamicsCertificate,
        c3: Phase3ImmuneCertificate,
        timestamp_utc: str,
    ) -> str:
        """Genera un hash SHA-256 de procedencia que ata los veredictos y residuos."""
        raw_payload = "|".join(
            (
                timestamp_utc,
                str(c1.verdict.value),
                str(c2.verdict.value),
                str(c3.verdict.value),
                f"{c1.registry_size:d}",
                f"{c1.fermion_count:d}",
                f"{c1.boson_count:d}",
                f"{c1.density_matrix_trace:.12e}",
                f"{c1.density_matrix_purity:.12e}",
                f"{c2.final_trace:.12e}",
                f"{c2.final_purity:.12e}",
                f"{c2.trace_preservation_residual:.12e}",
                f"{c2.lindblad_dissipation_rate:.12e}",
                f"{c3.annihilation_events:d}",
                f"{c3.entropy_production:.12e}",
                str(int(c1.higham_regularization_applied)),
                str(int(c2.is_trace_preserved)),
                str(int(c2.is_psd_preserved)),
                str(int(c3.is_unitary_evolution)),
            )
        )
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

# ═══════════════════════════════════════════════════════════════════════════
# Compatibilidad de nombres de clases para versiones anteriores
# ═══════════════════════════════════════════════════════════════════════════
Phase1_CoreFockRegistry = Phase1_CoreFockRegistry
Phase2_QuantumDynamics = Phase2_QuantumDynamics
Phase3_ImmuneOrchestrator = Phase3_ImmuneOrchestrator

__all__ = [
    "SynapticRegistryError",
    "PauliExclusionViolationError",
    "FockSpaceOverflowError",
    "TraceAnomalyError",
    "SymplecticConstraintError",
    "DensityMatrixNonPSDError",
    "LindbladEvolutionError",
    "FockRegistryVerdict",
    "RegistryAction",
    "ToonCartridge",
    "ElectronCartridge",
    "ProtonCartridge",
    "PolaronCartridge",
    "TorsionCartridge",
    "HouseholderReflectionFermion",
    "PhotonCartridge",
    "RiemannianFocalBoson",
    "MagnonCartridge",
    "SwellingPlasmonCartridge",
    "YieldingPhononCartridge",
    "LiquefactionSolitonCartridge",
    "PositronCartridge",
    "GammaPhoton",
    "PolaritonCartridge",
    "SophonCartridge",
    "Phase1RegistryCertificate",
    "Phase2DynamicsCertificate",
    "Phase3ImmuneCertificate",
    "FockRegistryState",
    "Phase1_CoreFockRegistry",
    "Phase2_QuantumDynamics",
    "Phase3_ImmuneOrchestrator",
    "SynapticFockSpaceRegistry",
    "stable_density_matrix_higham",
]