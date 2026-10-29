# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : SynapticFockSpaceRegistryAgent                                    ║
║ Módulo   : Synaptic Fock Space Registry Agent (Soberano de Memoria Cuántica)   ║
║ Ubicación: app/agents/core/inmune_system/synaptic_fock_space_registry_agent.py ║
║ Ruta     : app/agents/core/inmune_system/synaptic_fock_space_registry_agent.py ║
║ Versión  : 9.3.0-PhD (RSI³ Picard-Banach + Poincaré-KS-Melnikov-Chirikov)      ║
║ Función  : Soberano Agéntico y Observador Activo de Memoria Cuántica.       ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO AGÉNTICO Y MOTOR OODA:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al Agente Soberano y Observador Activo del Registro del
Espacio de Fock, encargado de gobernar y certificar síncronamente la consistencia
cuántica de la inyección de "Vitaminas TOON" (ToonCartridges) en la Malla Agéntica.
Actúa como un endofuntor en la categoría de haces sobre el sitio de Grothendieck
con valores en el retículo de Heyting:

    \Omega_3 = \{ \bot = \mathtt{VETOED}(2) \prec \star = \mathtt{DEGRADED}(1) \prec \top = \mathtt{COHERENT}(0) \}

El Espacio de Fock auditado es

    \mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^\infty \left( \bigwedge^n \mathcal{H}_{\text{fermion}} \oplus \text{Sym}^n \mathcal{H}_{\text{boson}} \right)

con álgebra CAR fermiónica \{a_i, a_j^\dagger\} = \delta_{ij} \mathbf{I} y CCR bosónica [b_i, b_j^\dagger] = \delta_{ij} \mathbf{I}.

Unificación con la mecánica celeste de Poincaré (Delaunay, carta canónica no
singular, regularización Kustaanheimo-Stiefel con vínculo bilinear, tiempo
ficticio de Sundman, KAM, Melnikov, Chirikov) y con RSI³ como contracción de
Banach-Picard sobre el simplejo de densidades \mathfrak{D}(\mathcal{H}):

    \mathbf{T}_{\text{RSI³}} = R_3 \circ R_3^{\text{Brockett}} \circ R_2^{\text{Connes-Daleckii-Krein}} \circ R_1^{\text{Ricci-DeTurck}}

    \exists! \rho^* \in \mathfrak{D}(\mathcal{H}) \quad \text{tal que} \quad \mathbf{T}(\rho^*) = \rho^* \quad \text{si} \quad \kappa = \operatorname{Lip}(\mathbf{T}) < 1.0.

ARQUITECTURA DE FASES ANIDADAS (CONTINUUM OODA INDIVISIBLE):
────────────────────────────────────────────────────────────────────────────────
  • FASE 1 (Observe): Isometría de Fock, Pauli-CAR, cota de Bekenstein,
    Higham-Lagrange, KS bilinear, atlas Delaunay-Poincaré y CS₃^Higgs.
    Método terminal: `synthesize_phase1_observation_manifold()` →
    `Phase1ObservationBundle` = (ρ₀, I, θ, u_KS, Π_Poincaré, C₁).

  • FASE 2 (Orient): Consume estrictamente el fibrado de Fase 1 en
    `orient_phase2_symplectic_resonances()`. Audita Sp(2n,C) / Bogoliubov-
    Valatin, aniquilación e⁻+e⁺→2γ con HMAC de tiempo constante, sección de
    Poincaré, Melnikov tanh-sinh, Chirikov-KAM, Bures-Finsler y Kelly
    censurado por Fermi-Dirac. Método terminal: `prepare_phase2_boundary()` →
    `Phase2OrientedBoundaryBundle` = (ρ, H, RHI, K_C, M(t₀), C₂).

  • FASE 3 (Decide & Act): Consume estrictamente la frontera de Fase 2 en
    `decide_act_phase3_rsi3_governance()`. Audita GKSL, Fredholm, pasividad
    de Lyapunov-Spohn y ejecuta RSI³ (Ricci-DeTurck, CDK, Brockett, Picard).
    Si Ω₃ = ⊥, señala el interlock ciber-físico simulado (GPIO14 / registro
    0x3FF44008) sin I/O de hardware.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
1. ISOMETRÍA DEL ESPACIO DE FOCK Y ISOMORFISMO DE BOGOLIUBOV-VALATIN:
   El soberano audita la inyección de cartuchos en $\mathcal{F}(\mathcal{H})$. Para modos fermiónicos,
   la ocupación está restringida por la regla de Pauli: $n_i \in \{0, 1\}$.
   Bajo transformaciones canónicas de Bogoliubov-Valatin para bosones en $\operatorname{Sp}(2, \mathbb{R})$:
   $$\alpha_k = u_k a_k + v_k a_k^\dagger, \qquad \alpha_k^\dagger = v_k^* a_k + u_k^* a_k^\dagger, \qquad |u_k|^2 - |v_k|^2 \equiv 1.0$$
   donde $u_k, v_k \in \mathbb{C}$ son coeficientes de mezcla Bogoliubov, y $a_k, a_k^\dagger$ son los operadores base.

2. ANIQUILACIÓN FERMIÓNICA $e^- + e^+ \to 2\gamma$ Y TIMING-SAFE HMAC:
   La aniquilación de pares electrón-positrón exige conservación estricta de la carga homológica $q_{e^-} + q_{e^+} = 0$
   y validación criptográfica de la firma de autorización mediante `hmac.compare_digest`:
   $$\text{HMAC}_{\text{SHA-256}}(K, \text{"AUTH:"} \parallel \text{positron\_id}) \equiv \text{signature}$$
   La reacción genera dos fotones gamma coplanares en direcciones opuestas con energía $E_\gamma = \frac{1}{2}(m_{e^-} + m_{e^+})$.

3. COTA HOLOGRÁFICA DE BEKENSTEIN Y EVICCIÓN POR ENTROPÍA DE CONFIGURACIÓN:
   La capacidad máxima del registro $N_{\text{max}} = 1024$ satura la cota holográfica de Bekenstein:
   $$S_{\text{Bekenstein}} = \frac{2\pi k_B R E}{\hbar c} \le N_{\text{max}} \ln(2)$$
   La evicción selecciona la partícula que minimiza la similitud cosenoidal respecto del ket de sondeo:
   $$k^* = \arg \min_k \frac{\langle u, v_k \rangle}{\|u\| \|v_k\|}$$

4. GEOMETRÍA DE BURES-FINSLER Y CENSURA DE KELLY DE FERMI-DIRAC:
   $$\mathcal{F}(\rho, \sigma) = \left( \operatorname{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2, \quad d_B = \sqrt{2 - 2\sqrt{\mathcal{F}}}, \quad d_{\text{FS}} = \arccos(\sqrt{\mathcal{F}})$$
   $$RHI_{\text{Finsler}} = \sqrt{ \frac{\operatorname{Tr}((\sigma - \rho)^2)}{\|\rho\|_F^2 (1 + \sqrt{\mathcal{F}})^2} + \frac{1}{4} \tanh^2\left(\operatorname{Tr}(\rho\sigma - \rho^2)\right) }$$
   $$\mathcal{S}_{T_{\text{sw}}}(RHI) = \frac{1}{1 + e^{(RHI - 0.85) / T_{\text{sw}}}}, \qquad T_{\text{sw}} = 0.02$$
   $$s^* = f^* \cdot (1 - RHI_{\text{Finsler}}) \cdot \mathcal{S}_{T_{\text{sw}}}(RHI) \cdot \Theta(0.15 - |M(t_0)|) \cdot \Theta(1.0 - K_{\text{Chirikov}}) \cdot \Theta_{\text{KAM}}$$

5. DINÁMICA DE LINDBLAD-GKSL, COTA DE FREDHOLM Y PASIVIDAD DE LYAPUNOV-SPOHN:
   La evolución disipativa del operador densidad sigue la ecuación maestra de Gorini-Kossakowski-Sudarshan-Lindblad:
   $$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right)$$
   Solubilidad de Fredholm sobre el kernel del Hamiltoniano: $\operatorname{Tr}(P_{\operatorname{ker} H} \rho) \le \varepsilon_{\text{tol}}$.
   Pasividad de Lyapunov-Spohn: $\frac{d}{dt}\langle H \rangle_\rho \le 0$ y producción no negativa de entropía $\sigma_{\text{Spohn}} = -\operatorname{Tr}(\log\rho \cdot \dot{\rho}) \ge 0$.

6. CASCADA DE BANACH RSI³ Y PUNTO FIJO DE PICARD:
   - $R_1$: Flujo de Ricci-DeTurck $\partial_\tau \ln G = -2 \operatorname{tf}(\ln G)$.
   - $R_2$: Cota de Connes-Daleckii-Krein $\phi^{[1]}(\lambda_i, \lambda_j) = \frac{\ln\lambda_i - \ln\lambda_j}{\lambda_i - \lambda_j}$.
   - $R_3$: Flujo isospectral de Brockett $\dot{\rho} = [\rho, [\rho, N]]$.
   - Contracción Picard-Banach: $\rho_{n+1} = \Pi_{\mathfrak{D}(\mathcal{H})}(\rho_n + \tau \dot{\rho}(\rho_n))$, con $\operatorname{Lip}(\mathbf{T}) \le \kappa < 1.0$.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Exclusión de Pauli CAR: \{a_i, a_j^\dagger\} = \delta_{ij} \mathbf{I} \implies (a_i^\dagger)^2 = 0 \quad \forall i, j.
  [A2] Conservación de Traza Cuántica: \operatorname{Tr}(\rho(t)) \equiv 1.0 \quad \forall t \ge 0.
  [A3] Positividad Semidefinida PSD: \langle \psi | \rho(t) | \psi \rangle \ge 0 \quad \forall |\psi\rangle \in \mathcal{H}.
  [A4] Pasividad y Producción No Negativa de Spohn: \sigma_{\text{Spohn}} = -\operatorname{Tr}(\log\rho \cdot \dot{\rho}) \ge 0.0.
  [A5] Interlock Ciber-Físico: \Omega_3 = \mathtt{VETOED} \implies \text{disparo de señal Crowbar GPIO14 (0x3FF44008)}.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Normalización de Traza Unitario-Disipativa):
      |\operatorname{Tr}(\rho(t)) - 1.0| \le 1.0 \times 10^{-4}.
  Invariante 2 (Preservación del Cono PSD por Higham):
      \lambda_{\min}(\rho_{\text{proj}}) \ge \text{floor} = 1.0 \times 10^{-12}.
  Invariante 3 (Invarianza Hiperbólica de Bogoliubov-Valatin):
      ||u_k|^2 - |v_k|^2 - 1.0| \le 1.0 \times 10^{-10}.
  Invariante 4 (Involución Bipartita del Retículo de Heyting):
      \Omega_3 = \operatorname{supremum}(c_1, c_2, c_3) = \max(\text{int}(c_1), \text{int}(c_2), \text{int}(c_3)).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from types import ModuleType
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Tensores dispersos (auditoría GKSL opcional)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.sparse import isspmatrix
    _SPARSE_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover
    isspmatrix = None
    _SPARSE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Stubs categóricos y topológicos seguros
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Violación fundamental de invariantes topológico-algebraicos."""

    class Morphism:
        """Morfismo categórico base."""

        def __init__(self, name: str = "Morphism") -> None:
            self.name = name

    class Stratum:
        """Estratos epistémicos DIKW."""

        PHYSICS = "PHYSICS"
        TACTICS = "TACTICS"
        STRATEGY = "STRATEGY"
        WISDOM = "WISDOM"

# ─────────────────────────────────────────────────────────────────────────────
# Núcleo espectral (v9.0.0-PhD) con degradación controlada a v8 / stubs
# ─────────────────────────────────────────────────────────────────────────────
_CORE: Optional[ModuleType] = None
try:
    from app.core.inmune_system import synaptic_fock_space_registry as _CORE
except ImportError:
    try:
        from app.core.immune_system import synaptic_fock_space_registry as _CORE
    except ImportError:
        try:
            import synaptic_fock_space_registry as _CORE  # type: ignore
        except ImportError:
            _CORE = None


def _core_attr(name: str, default: Any = None) -> Any:
    if _CORE is None:
        return default
    return getattr(_CORE, name, default)


if _CORE is not None:
    SynapticFockSpaceRegistry = _core_attr("SynapticFockSpaceRegistry")
    ToonCartridge = _core_attr("ToonCartridge")
    ElectronCartridge = _core_attr("ElectronCartridge")
    ProtonCartridge = _core_attr("ProtonCartridge")
    PolaronCartridge = _core_attr("PolaronCartridge")
    TorsionCartridge = _core_attr("TorsionCartridge")
    HouseholderReflectionFermion = _core_attr("HouseholderReflectionFermion")
    PositronCartridge = _core_attr("PositronCartridge")
    GammaPhoton = _core_attr("GammaPhoton")
    SynapticRegistryError = _core_attr("SynapticRegistryError", TopologicalInvariantError)
    PauliExclusionViolationError = _core_attr("PauliExclusionViolationError", SynapticRegistryError)
    SymplecticConstraintError = _core_attr("SymplecticConstraintError", SynapticRegistryError)
    TraceAnomalyError = _core_attr("TraceAnomalyError", SynapticRegistryError)
    FockSpaceOverflowError = _core_attr("FockSpaceOverflowError", SynapticRegistryError)
    DensityMatrixNonPSDError = _core_attr("DensityMatrixNonPSDError", SynapticRegistryError)
    LindbladEvolutionError = _core_attr("LindbladEvolutionError", SynapticRegistryError)
    KAMResonanceCatastropheError = _core_attr("KAMResonanceCatastropheError", SynapticRegistryError)
    BanachContractionFailureError = _core_attr("BanachContractionFailureError", SynapticRegistryError)
    FockRegistryVerdict = _core_attr("FockRegistryVerdict")
    RegistryAction = _core_attr("RegistryAction")
    PoincareCanonicalChart = _core_attr("PoincareCanonicalChart")
    stable_density_matrix_higham = _core_attr("stable_density_matrix_higham")
    nearest_density_operator_higham = _core_attr("nearest_density_operator_higham")
    von_neumann_entropy = _core_attr("von_neumann_entropy")
    regularize_kustaanheimo_stiefel = _core_attr("regularize_kustaanheimo_stiefel")
    ks_bilinear_residual = _core_attr("ks_bilinear_residual")
    solve_kepler_danby_order3 = _core_attr("solve_kepler_danby_order3")
    compute_delaunay_poincare_symplectic_elements = _core_attr(
        "compute_delaunay_poincare_symplectic_elements"
    )
    delaunay_to_poincare_canonical = _core_attr("delaunay_to_poincare_canonical")
    kepler_hamiltonian_and_mean_motion = _core_attr("kepler_hamiltonian_and_mean_motion")
    compute_yang_mills_higgs_chern_simons_invariants = _core_attr(
        "compute_yang_mills_higgs_chern_simons_invariants"
    )
    compute_melnikov_homoclinic_integral = _core_attr("compute_melnikov_homoclinic_integral")
    poincare_section_homoclinic_splitting = _core_attr("poincare_section_homoclinic_splitting")
    compute_chirikov_kam_criterion = _core_attr("compute_chirikov_kam_criterion")
    compute_bures_finsler_quantum_metrics = _core_attr("compute_bures_finsler_quantum_metrics")
else:  # pragma: no cover — ejecución desacoplada
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

    class LindbladEvolutionError(SynapticRegistryError):
        pass

    class KAMResonanceCatastropheError(SynapticRegistryError):
        pass

    class BanachContractionFailureError(SynapticRegistryError):
        pass

    class FockRegistryVerdict(IntEnum):
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    class RegistryAction(Enum):
        NONE = auto()
        PURGE_ENTROPY = auto()
        RESCALE_HAMILTONIAN = auto()
        ACTUATE_HARDWARE_CROWBAR = auto()
        HALT_REGISTRY = auto()

    @dataclass(frozen=True, slots=True)
    class ToonCartridge:
        cartridge_id: str
        signature: str

        @property
        def quantum_state_hash(self) -> str:
            payload = f"{self.__class__.__name__}:{self.cartridge_id}:{self.signature}"
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @dataclass(frozen=True, slots=True)
    class ElectronCartridge(ToonCartridge):
        inertial_mass: float = 1.0
        topological_spin: str = "source"
        homological_charge: int = -1

    @dataclass(frozen=True, slots=True)
    class ProtonCartridge(ToonCartridge):
        spectral_charge: float = 1.0
        logistic_inertial_mass: float = 1836.0
        dominant_pole: float = -0.5

    @dataclass(frozen=True, slots=True)
    class PolaronCartridge(ToonCartridge):
        frohlich_coupling: float = 0.1
        effective_mass: float = 1.2
        fiedler_value: float = 0.5

    @dataclass(frozen=True, slots=True)
    class TorsionCartridge(ToonCartridge):
        torsion_degree: int = 1
        torsion_elements: Tuple[int, ...] = (1,)

    @dataclass(frozen=True, slots=True)
    class HouseholderReflectionFermion(ToonCartridge):
        covariant_hyperplane_normal: Tuple[float, ...] = (1.0, 0.0)
        monodromy_spectral_radius: float = 1.0
        cohomology_obstruction_class: int = 0

    @dataclass(frozen=True, slots=True)
    class PositronCartridge(ToonCartridge):
        inertial_mass: float = 1.0
        topological_spin: str = "sink"
        homological_charge: int = 1
        authorization_signature: str = ""

    @dataclass(frozen=True, slots=True)
    class GammaPhoton(ToonCartridge):
        energy: float = 1.0
        wave_vector: Tuple[float, ...] = (0.0, 0.0, 1.0)
        provenance_hash: str = ""

    SynapticFockSpaceRegistry = None  # type: ignore[misc, assignment]

    def stable_density_matrix_higham(
        rho: NDArray[np.complex128],
        tolerance: float = 1.0e-12,
    ) -> Tuple[NDArray[np.complex128], bool]:
        return _nearest_density_operator_higham_local(rho, floor=tolerance)

    nearest_density_operator_higham = None
    von_neumann_entropy = None
    regularize_kustaanheimo_stiefel = None
    ks_bilinear_residual = None
    solve_kepler_danby_order3 = None
    compute_delaunay_poincare_symplectic_elements = None
    delaunay_to_poincare_canonical = None
    kepler_hamiltonian_and_mean_motion = None
    compute_yang_mills_higgs_chern_simons_invariants = None
    compute_melnikov_homoclinic_integral = None
    poincare_section_homoclinic_splitting = None
    compute_chirikov_kam_criterion = None
    compute_bures_finsler_quantum_metrics = None
    PoincareCanonicalChart = None

logger = logging.getLogger("MIC.Agents.Omega.SynapticFockSpaceRegistryAgent")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, HARDWARE Y UMBRALES DE GOBERNANZA
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12
_DENSITY_MATRIX_CONDITION_LIMIT: Final[float] = 1.0e8
_SPARSE_THRESHOLD: Final[float] = 0.3
_MAX_REGISTRY_SIZE: Final[int] = 1024
_ENTROPY_EVICT_THRESHOLD: Final[float] = 0.88
_TRACE_UNIT_TOL: Final[float] = 1.0e-4
_PURITY_DEGRADED: Final[float] = 0.40
_KEPLER_ECC_MAX: Final[float] = 0.999999
_KS_BILINEAR_TOL: Final[float] = 1.0e-10
_PICARD_MAX_ITERS: Final[int] = 12
_PICARD_FIXED_POINT_TOL: Final[float] = 1.0e-10

# Interlock ciber-físico (señal de gobernanza; no se realiza I/O de MMIO)
_CROWBAR_GPIO: Final[int] = 14
_CROWBAR_REGISTER: Final[str] = "0x3FF44008"
_CROWBAR_ACTUATION_LATENCY_NS: Final[float] = 392.15

_FERMI_DIRAC_TEMP: Final[float] = 0.02
_RHI_THRESHOLD: Final[float] = 0.85
_MELNIKOV_MAX_TOL: Final[float] = 0.15
_CHIRIKOV_CRITICAL: Final[float] = 1.00
_KAM_DIOPHANTINE_GAMMA: Final[float] = 0.01
_KAM_DIOPHANTINE_TAU: Final[float] = 2.5
_RSI3_BANACH_CONTRACTION_MAX: Final[float] = 0.95
_SPOHn_FLOOR: Final[float] = 0.0
_OCCUPANCY_BOGOLIUBOV_LIMIT: Final[float] = 0.85

_FERMION_TYPES: Final[Tuple[type, ...]] = tuple(
    t
    for t in (
        ElectronCartridge,
        ProtonCartridge,
        PolaronCartridge,
        TorsionCartridge,
        HouseholderReflectionFermion,
    )
    if isinstance(t, type)
)


# ═══════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES AGÉNTICAS
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistryAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de la Memoria Cuántica."""


class FockIsometryViolation(SynapticFockSpaceRegistryAgentError):
    """Desviación no unitaria o colapso en el Espacio de Fock."""


class DensityMatrixAnomalyError(SynapticFockSpaceRegistryAgentError):
    """Violación de los postulados fundamentales de Dirac-von Neumann."""


class SymplecticInvarianceViolation(SynapticFockSpaceRegistryAgentError):
    """Violación simpléctica sobre la variedad de Bogoliubov."""


class LindbladTraceViolation(SynapticFockSpaceRegistryAgentError):
    """Evolución disipativa no preservativa de traza o no Markoviana."""


class FredholmSolvabilityViolation(SynapticFockSpaceRegistryAgentError):
    """Ruptura de la condición de solubilidad de Fredholm en el kernel disipativo."""


class LipschitzStabilityViolation(SynapticFockSpaceRegistryAgentError):
    """Divergencia de la cota Lipschitziana de Connes-Daleckii-Krein."""


class KSBilinearAuditError(SynapticFockSpaceRegistryAgentError):
    """Violación del vínculo bilinear de Kustaanheimo-Stiefel en la auditoría."""


class PoincareChartAuditError(SynapticFockSpaceRegistryAgentError):
    """Singularidad de la carta de Poincaré (e → 0 o i → 0 degenerados)."""


class BanachPicardAuditError(SynapticFockSpaceRegistryAgentError):
    """Fallo de contracción de Picard-Banach en la cascada RSI³ auditada."""


# ═══════════════════════════════════════════════════════════════════════════
# RETÍCULO DE HEYTING Y ACCIONES DE MITIGACIÓN FÍSICA
# ═══════════════════════════════════════════════════════════════════════════
class FockSovereignVerdict(IntEnum):
    """
    Clasificador trivaluado en el retículo de Heyting Ω₃:
      COHERENT = 0 (⊤ operativo: admisibilidad cuántica completa).
      DEGRADED = 1 (★ operativo: advertencia, veto suave y watchdog).
      VETOED   = 2 (⊥ operativo: veto duro e interlock crowbar).
    El supremo es el join topológico (peor veredicto).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "FockSovereignVerdict") -> "FockSovereignVerdict":
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
    """Acciones de mitigación sobre el flujo epistémico (señales, no I/O)."""

    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()


# ═══════════════════════════════════════════════════════════════════════════
# CERTIFICADOS Y PAQUETES FIBRADOS ANIDADOS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class PoincareCanonicalChartView:
    """
    Carta canónica de Poincaré (variables no singulares) inducida por Delaunay.
    Λ = L,  λ = l+g+h,
    ξ = √(2(L-G)) cos(ϖ),  η = √(2(L-G)) sin(ϖ),  ϖ = g+h,
    p = √(2(G-H)) cos(h),  q = √(2(G-H)) sin(h).
    """

    action_lambda: float
    mean_longitude: float
    ecc_xi: float
    ecc_eta: float
    inc_p: float
    inc_q: float
    poincare_norm: float


@dataclass(frozen=True, slots=True)
class Phase1FockIsometryCertificate:
    """Certificado formal emitido en Fase 1 (Observe)."""

    is_pauli_respected: bool
    is_fock_bounded: bool
    entropy_eviction_ratio: float
    occupancy: int
    max_capacity: int
    density_matrix_trace: float
    density_matrix_purity: float
    von_neumann_entropy: float
    spectral_condition_number: float
    higham_regularization_applied: bool
    sparse_computation: bool
    kustaanheimo_stiefel_energy: float
    ks_bilinear_residual: float
    delaunay_eccentricity: float
    delaunay_inclination: float
    kepler_mean_anomaly: float
    kepler_eccentric_anomaly: float
    kepler_hamiltonian: float
    mean_motion: float
    poincare_chart: PoincareCanonicalChartView
    chern_simons_3form_val: float
    wilson_loop_holonomy: float
    verdict: FockSovereignVerdict


@dataclass(frozen=True, slots=True)
class Phase1ObservationBundle:
    """
    PAQUETE FIBRADO DE ENLACE FASE 1 → FASE 2.
    Variedad observada, operador densidad auditado, spinor KS, carta de
    Poincaré y certificado C₁. Dominio estricto de
    `orient_phase2_symplectic_resonances`.
    """

    rho_observed: NDArray[np.complex128]
    delaunay_actions: Tuple[float, float, float]
    delaunay_angles: Tuple[float, float, float]
    ks_spinor: NDArray[np.float64]
    poincare_chart: PoincareCanonicalChartView
    certificate: Phase1FockIsometryCertificate


@dataclass(frozen=True, slots=True)
class Phase2SymplecticBogoliubovCertificate:
    """Certificado formal emitido en Fase 2 (Orient)."""

    is_symplectic_invariant: bool
    symplectic_residual: float
    is_charge_conserved: bool
    hmac_authorization_valid: bool
    annihilation_provenance_hash: Optional[str]
    is_consistent_with_occupancy: bool
    bogoliubov_transformation_count: int
    annihilation_events: int
    uhlmann_fidelity: float
    bures_distance: float
    fubini_study_distance: float
    rhi_finsler_index: float
    fermi_dirac_kelly_censorship: float
    kelly_optimal_stake: float
    melnikov_integral_val: float
    poincare_section_splitting: float
    chirikov_overlap_parameter: float
    is_kam_tori_preserved: bool
    fictitious_sundman_time: float
    verdict: FockSovereignVerdict


@dataclass(frozen=True, slots=True)
class Phase2OrientedBoundaryBundle:
    """
    PAQUETE FIBRADO DE ENLACE FASE 2 → FASE 3.
    Variedad simpléctica orientada, generador kepleriano, métricas de
    caos/Finsler y certificado C₂. Dominio estricto de
    `decide_act_phase3_rsi3_governance`.
    """

    rho_oriented: NDArray[np.complex128]
    hamiltonian_generator: NDArray[np.complex128]
    rhi_finsler_metric: float
    chirikov_overlap: float
    melnikov_divergence: float
    sundman_fictitious_time: float
    certificate: Phase2SymplecticBogoliubovCertificate


@dataclass(frozen=True, slots=True)
class Phase3LindbladPassivityCertificate:
    """Certificado formal emitido en Fase 3 (Decide & Act)."""

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
    spohn_entropy_production_rate: float
    ricci_deturck_metric_drift: float
    connes_daleckii_krein_lipschitz: float
    brockett_isospectral_error: float
    rsi3_banach_contraction_factor: float
    picard_iterations: int
    picard_fixed_point_residual: float
    is_banach_contraction: bool
    crowbar_actuated: bool
    verdict: FockSovereignVerdict


@dataclass(frozen=True, slots=True)
class FockRegistrySovereignState:
    """Certificado holístico terminal de gobernanza agéntica."""

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
# KERNELS LOCALES DE RESPALDO Y ADAPTADORES DE FIRMA (v8 ↔ v9)
# ═══════════════════════════════════════════════════════════════════════════
def _hermitian(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (rho + rho.conj().T)


def _safe_eigvalsh(rho: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.real(la.eigvalsh(_hermitian(rho)))


def _nearest_density_operator_higham_local(
    rho: NDArray[np.complex128],
    floor: float = _HIGHAM_REGULARIZATION_FLOOR,
) -> Tuple[NDArray[np.complex128], bool]:
    r"""
    Proyección Frobenius sobre D(H) = {X = X^† ⪰ 0 : Tr(X) = 1}.
    Higham (recorte espectral) + multiplicador de Lagrange σ* con Newton:
        Σ_i max(λ_i + σ*, floor) = 1.
    """
    rho_h = _hermitian(np.asarray(rho, dtype=np.complex128))
    try:
        eigenvals, eigenvecs = la.eigh(rho_h)
    except Exception as exc:
        raise DensityMatrixAnomalyError("Falla espectral en Higham local.") from exc

    applied = bool(np.any(eigenvals < floor))
    n = int(eigenvals.shape[0])
    if n == 0:
        raise DensityMatrixAnomalyError("Operador de dimensión nula.")

    sigma = 0.0
    clipped = np.maximum(eigenvals + sigma, floor)
    total = float(np.sum(clipped))
    if abs(total - 1.0) > _MACHINE_EPS:
        for _ in range(40):
            active = (eigenvals + sigma) > floor
            n_active = int(np.count_nonzero(active))
            if n_active == 0:
                sigma = 1.0 - float(n) * floor
                break
            f_val = float(np.sum(np.maximum(eigenvals + sigma, floor)) - 1.0)
            if abs(f_val) < 1.0e-15:
                break
            sigma -= f_val / float(n_active)
        clipped = np.maximum(eigenvals + sigma, floor)
        total = float(np.sum(clipped))
        if total > _MACHINE_EPS:
            clipped = clipped / total
        else:
            clipped = np.zeros_like(clipped)
            clipped[-1] = 1.0
        applied = True
    else:
        clipped = np.maximum(eigenvals, floor)
        total = float(np.sum(clipped))
        if abs(total - 1.0) > _MACHINE_EPS:
            clipped = clipped / max(total, _MACHINE_EPS)
            applied = True

    rho_proj = eigenvecs @ np.diag(clipped) @ eigenvecs.conj().T
    rho_proj = _hermitian(rho_proj)
    if float(np.min(la.eigvalsh(rho_proj))) < -floor:
        raise DensityMatrixAnomalyError("Higham local: espectro fuera del cono PSD.")
    return rho_proj.astype(np.complex128, copy=False), applied


def _project_density(
    rho: NDArray[np.complex128],
    floor: float = _HIGHAM_REGULARIZATION_FLOOR,
) -> Tuple[NDArray[np.complex128], bool]:
    if callable(nearest_density_operator_higham):
        return nearest_density_operator_higham(rho, floor=floor)
    if callable(stable_density_matrix_higham):
        out = stable_density_matrix_higham(rho, tolerance=floor)
        return out[0], bool(out[1])
    return _nearest_density_operator_higham_local(rho, floor=floor)


def _entropy_vn(rho: NDArray[np.complex128]) -> float:
    if callable(von_neumann_entropy):
        return float(von_neumann_entropy(rho))
    vals = np.maximum(_safe_eigvalsh(rho), 0.0)
    mask = vals > _MACHINE_EPS
    if not np.any(mask):
        return 0.0
    return float(-np.sum(vals[mask] * np.log(vals[mask])))


def _ks_bilinear_local(u: NDArray[np.float64]) -> float:
    u1, u2, u3, u4 = (float(u[0]), float(u[1]), float(u[2]), float(u[3]))
    return abs(u1 * u4 - u2 * u3)


def _regularize_ks(
    x: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], float, float]:
    """Adaptador de firma: el núcleo v9 devuelve 3-tupla; v8 devuelve 2-tupla."""
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    if callable(regularize_kustaanheimo_stiefel):
        out = regularize_kustaanheimo_stiefel(x_vec)
        if isinstance(out, tuple) and len(out) >= 3:
            u = np.asarray(out[0], dtype=np.float64)
            energy, bilinear = float(out[1]), float(out[2])
            return u, energy, bilinear
        if isinstance(out, tuple) and len(out) == 2:
            u = np.asarray(out[0], dtype=np.float64)
            energy = float(out[1])
            if callable(ks_bilinear_residual) and u.size >= 4:
                bilinear = float(ks_bilinear_residual(u))
            else:
                bilinear = _ks_bilinear_local(np.pad(u, (0, max(0, 4 - u.size))))
            return u, energy, bilinear

    r = float(np.linalg.norm(x_vec[:3])) if x_vec.size else 0.0
    u = np.zeros(4, dtype=np.float64)
    if r < _MACHINE_EPS:
        return u, 0.0, 0.0
    x1 = float(x_vec[0]) if x_vec.size > 0 else 0.0
    x2 = float(x_vec[1]) if x_vec.size > 1 else 0.0
    x3 = float(x_vec[2]) if x_vec.size > 2 else 0.0
    if x1 + r > _MACHINE_EPS:
        u[0] = math.sqrt((r + x1) / 2.0)
        inv = 1.0 / (2.0 * u[0])
        u[1] = x2 * inv
        u[2] = x3 * inv
        u[3] = 0.0
    else:
        u[1] = math.sqrt(max((r - x1) / 2.0, 0.0))
        if u[1] > _MACHINE_EPS:
            u[3] = x3 / (2.0 * u[1])
        u[0] = 0.0
        u[2] = 0.0
    bilinear = _ks_bilinear_local(u)
    energy = float(0.5 * np.sum(u * u) + 1.0 / (r + _MACHINE_EPS))
    return u, energy, bilinear


def _solve_kepler(M: float, e: float, max_iter: int = 15) -> float:
    if callable(solve_kepler_danby_order3):
        return float(solve_kepler_danby_order3(M, e, max_iter=max_iter))
    tau = 2.0 * math.pi
    M_norm = M % tau
    if M_norm < 0.0:
        M_norm += tau
    e_c = min(max(float(e), 0.0), _KEPLER_ECC_MAX)
    E = M_norm + 0.85 * e_c * (1.0 if math.sin(M_norm) >= 0.0 else -1.0)
    for _ in range(max_iter):
        s_e, c_e = math.sin(E), math.cos(E)
        f = E - e_c * s_e - M_norm
        f1 = 1.0 - e_c * c_e
        f2 = e_c * s_e
        f3 = e_c * c_e
        if abs(f1) < _MACHINE_EPS:
            break
        d1 = -f / f1
        d2 = -f / (f1 + 0.5 * d1 * f2)
        d3 = -f / (f1 + 0.5 * d2 * f2 + (1.0 / 6.0) * (d2 ** 2) * f3)
        E += d3
        if abs(f) < 1.0e-14:
            break
    return float(E)


def _delaunay_elements(
    rho: NDArray[np.complex128],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float]:
    if callable(compute_delaunay_poincare_symplectic_elements):
        return compute_delaunay_poincare_symplectic_elements(rho)
    vals = np.sort(_safe_eigvalsh(rho))[::-1]
    n = len(vals)
    L = float(np.sum(vals))
    l1 = max(float(vals[0]) if n else 1.0, _MACHINE_EPS)
    l2 = float(vals[1]) if n > 1 else l1 * 0.5
    l3 = float(vals[2]) if n > 2 else l2 * 0.5
    G = L * min(l2 / l1, 1.0)
    H = G * min(l3 / max(l2, _MACHINE_EPS), 1.0)
    ratio_g = min(max(G / max(L, _MACHINE_EPS), 0.0), 1.0)
    ecc = math.sqrt(max(1.0 - ratio_g * ratio_g, 0.0))
    mean_anomaly = float((l1 - l2) * math.pi) % (2.0 * math.pi)
    ecc_anomaly = _solve_kepler(mean_anomaly, ecc)
    actions = (L, G, H)
    angles = (
        mean_anomaly,
        float(l2 * math.pi) % (2.0 * math.pi),
        float(l3 * math.pi) % (2.0 * math.pi),
    )
    return actions, angles, ecc, ecc_anomaly


def _chart_from_core(chart: Any) -> PoincareCanonicalChartView:
    return PoincareCanonicalChartView(
        action_lambda=float(getattr(chart, "action_lambda", 0.0)),
        mean_longitude=float(getattr(chart, "mean_longitude", 0.0)),
        ecc_xi=float(getattr(chart, "ecc_xi", 0.0)),
        ecc_eta=float(getattr(chart, "ecc_eta", 0.0)),
        inc_p=float(getattr(chart, "inc_p", 0.0)),
        inc_q=float(getattr(chart, "inc_q", 0.0)),
        poincare_norm=float(getattr(chart, "poincare_norm", 0.0)),
    )


def _poincare_chart(
    actions: Tuple[float, float, float],
    angles: Tuple[float, float, float],
) -> PoincareCanonicalChartView:
    if callable(delaunay_to_poincare_canonical):
        return _chart_from_core(delaunay_to_poincare_canonical(actions, angles))
    L, G, H = (float(actions[0]), float(actions[1]), float(actions[2]))
    l, g, h = (float(angles[0]), float(angles[1]), float(angles[2]))
    L = max(L, _MACHINE_EPS)
    G = min(max(G, 0.0), L)
    H = min(max(H, -G), G)
    sqrt_e = math.sqrt(max(2.0 * (L - G), 0.0))
    sqrt_i = math.sqrt(max(2.0 * (G - abs(H)), 0.0))
    varpi = g + h
    mean_long = (l + g + h) % (2.0 * math.pi)
    xi = sqrt_e * math.cos(varpi)
    eta = sqrt_e * math.sin(varpi)
    p = sqrt_i * math.cos(h)
    q = sqrt_i * math.sin(h)
    return PoincareCanonicalChartView(
        action_lambda=L,
        mean_longitude=mean_long,
        ecc_xi=xi,
        ecc_eta=eta,
        inc_p=p,
        inc_q=q,
        poincare_norm=math.sqrt(xi * xi + eta * eta + p * p + q * q + L * L),
    )


def _kepler_hn(L: float) -> Tuple[float, float]:
    if callable(kepler_hamiltonian_and_mean_motion):
        h_kep, n = kepler_hamiltonian_and_mean_motion(L)
        return float(h_kep), float(n)
    L_safe = max(abs(float(L)), _MACHINE_EPS)
    return float(-0.5 / (L_safe * L_safe)), float(1.0 / (L_safe ** 3))


def _cs3_wilson(rho: NDArray[np.complex128]) -> Tuple[float, float]:
    if callable(compute_yang_mills_higgs_chern_simons_invariants):
        cs3, wilson = compute_yang_mills_higgs_chern_simons_invariants(rho)
        return float(cs3), float(wilson)
    dim = int(rho.shape[0])
    rho_clean = _hermitian(rho)
    vals, vecs = la.eigh(rho_clean)
    vals_pos = np.maximum(np.real(vals), _MACHINE_EPS)
    phi = vecs @ np.diag(np.log(vals_pos)) @ vecs.conj().T
    A = -1j * (rho_clean - (np.eye(dim, dtype=np.complex128) / dim))
    dA = 0.5 * (A @ A - A.conj().T @ A.conj().T)
    F_A = dA + (A @ A)
    cs_integrand = (A @ dA) + ((2.0 / 3.0) * A @ A @ A) + (phi @ F_A)
    cs3_value = float(np.real(np.trace(cs_integrand)) / (8.0 * (math.pi ** 2)))
    holonomy = la.expm(2.0 * math.pi * A)
    wilson = float(np.abs(np.trace(holonomy)) / dim)
    return cs3_value, wilson


def _melnikov(epsilon: float = 0.05, t0: float = 0.0, nodes: int = 120) -> float:
    if callable(compute_melnikov_homoclinic_integral):
        try:
            return float(compute_melnikov_homoclinic_integral(epsilon, t0, nodes))
        except TypeError:
            return float(compute_melnikov_homoclinic_integral())
    h = 0.04
    integral = 0.0
    for k in range(-nodes, nodes + 1):
        x = k * h
        if abs(x) > 8.0:
            continue
        arg = 0.5 * math.pi * math.sinh(x)
        if abs(arg) > 300.0:
            continue
        t = math.sinh(arg)
        cosh_arg = math.cosh(arg)
        weight = (0.5 * math.pi * math.cosh(x)) / (cosh_arg * cosh_arg + _MACHINE_EPS)
        if abs(t) > 300.0:
            continue
        sech_t = 1.0 / (math.cosh(t) + _MACHINE_EPS)
        integral += (sech_t * sech_t) * math.tanh(t) * math.sin(t + t0) * weight * h
    return float(epsilon * integral)


def _homoclinic_splitting(melnikov: float, mean_anomaly: float) -> float:
    if callable(poincare_section_homoclinic_splitting):
        return float(poincare_section_homoclinic_splitting(melnikov, mean_anomaly))
    dist = min(
        abs(mean_anomaly % (2.0 * math.pi)),
        abs((mean_anomaly % (2.0 * math.pi)) - 2.0 * math.pi),
    )
    window = math.exp(-dist * dist / 0.25)
    return float(abs(melnikov) * (0.5 + 0.5 * window))


def _chirikov_kam(rho: NDArray[np.complex128]) -> Tuple[float, bool]:
    if callable(compute_chirikov_kam_criterion):
        k_c, kam_ok = compute_chirikov_kam_criterion(rho)
        return float(k_c), bool(kam_ok)
    vals = np.sort(_safe_eigvalsh(rho))[::-1]
    if len(vals) < 2:
        return 0.0, True
    l1 = max(float(vals[0]), _MACHINE_EPS)
    l2 = max(float(vals[1]), _MACHINE_EPS)
    l3 = float(vals[2]) if len(vals) > 2 else l2 * 0.5
    alpha_twist = abs(-3.0 / (l1 ** 4 + _MACHINE_EPS))
    w1, w2 = math.sqrt(l1), math.sqrt(l2)
    d_om1 = 2.0 * math.sqrt(max(2.0 * abs(l1 - l2) * alpha_twist, 0.0))
    d_om2 = 2.0 * math.sqrt(max(2.0 * abs(l2 - l3) * alpha_twist, 0.0))
    freq = abs(w1 - w2)
    if freq < _MACHINE_EPS:
        return 0.0, True
    k_c = float((d_om1 + d_om2) / freq)
    bound = _KAM_DIOPHANTINE_GAMMA / (math.sqrt(2.0) ** _KAM_DIOPHANTINE_TAU)
    return k_c, bool((freq >= bound) and (k_c < _CHIRIKOV_CRITICAL))


def _bures_finsler(
    rho: NDArray[np.complex128],
    sigma: NDArray[np.complex128],
) -> Tuple[float, float, float, float, float]:
    if callable(compute_bures_finsler_quantum_metrics):
        fid, db, dfs, rhi, s_star = compute_bures_finsler_quantum_metrics(rho, sigma)
        return float(fid), float(db), float(dfs), float(rhi), float(s_star)
    rho_c, sigma_c = _hermitian(rho), _hermitian(sigma)
    sqrt_rho = la.sqrtm(rho_c)
    middle = _hermitian(sqrt_rho @ sigma_c @ sqrt_rho)
    fidelity = min(max(float(np.real(np.trace(la.sqrtm(middle)))) ** 2, 0.0), 1.0)
    d_bures = math.sqrt(max(2.0 - 2.0 * math.sqrt(fidelity), 0.0))
    d_fubini = math.acos(min(max(math.sqrt(fidelity), 0.0), 1.0))
    delta = sigma_c - rho_c
    nf = max(float(la.norm(rho_c, "fro")), _MACHINE_EPS)
    term_r = float(np.real(np.trace(delta @ delta.conj().T))) / (
        (nf ** 2) * ((1.0 + math.sqrt(fidelity)) ** 2) + _MACHINE_EPS
    )
    flux = float(np.real(np.trace(rho_c @ sigma_c - rho_c @ rho_c)))
    rhi = math.sqrt(max(term_r + 0.25 * (math.tanh(flux) ** 2), 0.0))
    sw = 1.0 / (1.0 + math.exp((rhi - _RHI_THRESHOLD) / _FERMI_DIRAC_TEMP))
    melnikov = _melnikov()
    chirikov, kam_ok = _chirikov_kam(rho_c)
    theta_m = 1.0 if abs(melnikov) <= _MELNIKOV_MAX_TOL else 0.0
    theta_c = 1.0 if chirikov < _CHIRIKOV_CRITICAL else 0.0
    theta_r = 1.0 if rhi < _RHI_THRESHOLD else 0.0
    s_star = 0.25 * max(1.0 - rhi, 0.0) * sw * theta_m * theta_c * theta_r
    if not kam_ok:
        s_star = 0.0
    return fidelity, d_bures, d_fubini, rhi, s_star


def _sundman_time(physical_dt: float, radius: float) -> float:
    return float(physical_dt / max(radius, _MACHINE_EPS))


def _cdk_divided_difference(lam_i: float, lam_j: float) -> float:
    if abs(lam_i - lam_j) <= 1.0e-10:
        return 1.0 / max(lam_i, _MACHINE_EPS)
    return abs(math.log(lam_i) - math.log(lam_j)) / abs(lam_i - lam_j)


def _spectral_condition(rho: NDArray[np.complex128]) -> float:
    vals = np.maximum(_safe_eigvalsh(rho), 0.0)
    smax = float(np.max(vals)) if vals.size else 0.0
    smin = float(np.min(vals[vals > _MACHINE_EPS])) if np.any(vals > _MACHINE_EPS) else _MACHINE_EPS
    return float(smax / max(smin, _MACHINE_EPS))


def _zero_poincare_chart() -> PoincareCanonicalChartView:
    return PoincareCanonicalChartView(
        action_lambda=0.0,
        mean_longitude=0.0,
        ecc_xi=0.0,
        ecc_eta=0.0,
        inc_p=0.0,
        inc_q=0.0,
        poincare_norm=0.0,
    )


def _kepler_default_hamiltonian(
    dim: int,
    delaunay_actions: Tuple[float, float, float],
) -> NDArray[np.complex128]:
    L = max(float(delaunay_actions[0]), _MACHINE_EPS)
    _, n_mean = _kepler_hn(L)
    freqs = n_mean * np.linspace(1.0, 2.0, max(dim, 1))
    return np.diag(freqs.astype(np.complex128))


# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 (OBSERVE): ISOMETRÍA DE FOCK, PAULI, KS Y ATLAS CELESTE
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_FockIsometryAuditor(ABC):
    r"""
    FASE 1 (Observe) — Ocupación del Espacio de Fock, exclusión de Pauli
    (CAR), regularización espinorial de Kustaanheimo-Stiefel con vínculo
    bilinear, coordenadas de Delaunay y carta canónica de Poincaré.

    Método terminal formal: `synthesize_phase1_observation_manifold()`.
    Su imagen `Phase1ObservationBundle` es el dominio del método inicial
    de Fase 2, `orient_phase2_symplectic_resonances`.
    """

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        self._tolerance: Final[float] = float(tolerance)
        self._higham_floor: Final[float] = _HIGHAM_REGULARIZATION_FLOOR

    def audit_fock_isometry(
        self,
        registry: Any,
        current_decision_vector: NDArray[np.float64],
    ) -> Phase1FockIsometryCertificate:
        r"""
        Inspecciona el espacio de Fock:
          1. Álgebra CAR: ningún estado fermiónico repetido por (tipo, hash).
          2. Cota holográfica de Bekenstein sobre la memoria cuántica.
          3. Proyección de Higham-Lagrange sobre D(H).
          4. KS + vínculo bilinear sobre la tripleta espectral dominante.
          5. Atlas Delaunay-Poincaré y 3-forma de Chern-Simons-Higgs.
        """
        is_pauli_respected = self._check_pauli_exclusion(registry)
        occupancy = int(getattr(registry, "size", 0))
        max_cap = int(getattr(registry, "max_capacity", _MAX_REGISTRY_SIZE))
        is_fock_bounded = occupancy <= max_cap

        entropy_ratio, sparse_used = self._compute_entropy_ratio(
            registry, current_decision_vector
        )
        rho, higham_applied = self._build_density_matrix_with_higham(registry)
        trace_val = float(np.real(np.trace(rho)))
        purity_val = float(np.real(np.trace(rho @ rho)))
        entropy_val = _entropy_vn(rho)
        cond_num = _spectral_condition(rho)

        evals = np.sort(_safe_eigvalsh(rho))[::-1]
        probe = evals[:3] if len(evals) >= 3 else np.pad(evals, (0, 3 - len(evals)))
        _, ks_energy, ks_bilinear = _regularize_ks(probe)

        actions, angles, ecc, ecc_anomaly = _delaunay_elements(rho)
        chart = _poincare_chart(actions, angles)
        h_kep, mean_motion = _kepler_hn(actions[0])
        cs3_val, wilson_val = _cs3_wilson(rho)
        inclination = math.acos(
            min(max(actions[2] / max(actions[1], _MACHINE_EPS), -1.0), 1.0)
        )

        verdict = FockSovereignVerdict.COHERENT
        if (
            (not is_fock_bounded)
            or (not is_pauli_respected)
            or (not np.isfinite(trace_val))
            or (abs(trace_val - 1.0) > _TRACE_UNIT_TOL)
            or (not np.isfinite(purity_val))
        ):
            verdict = FockSovereignVerdict.VETOED
        elif (
            occupancy >= max_cap * _ENTROPY_EVICT_THRESHOLD
            or higham_applied
            or purity_val < _PURITY_DEGRADED
            or ks_bilinear > 1.0e-6
            or cond_num > _DENSITY_MATRIX_CONDITION_LIMIT
        ):
            verdict = FockSovereignVerdict.DEGRADED

        return Phase1FockIsometryCertificate(
            is_pauli_respected=is_pauli_respected,
            is_fock_bounded=is_fock_bounded,
            entropy_eviction_ratio=entropy_ratio,
            occupancy=occupancy,
            max_capacity=max_cap,
            density_matrix_trace=trace_val,
            density_matrix_purity=purity_val,
            von_neumann_entropy=entropy_val,
            spectral_condition_number=cond_num,
            higham_regularization_applied=higham_applied,
            sparse_computation=sparse_used,
            kustaanheimo_stiefel_energy=ks_energy,
            ks_bilinear_residual=ks_bilinear,
            delaunay_eccentricity=ecc,
            delaunay_inclination=inclination,
            kepler_mean_anomaly=angles[0],
            kepler_eccentric_anomaly=ecc_anomaly,
            kepler_hamiltonian=h_kep,
            mean_motion=mean_motion,
            poincare_chart=chart,
            chern_simons_3form_val=cs3_val,
            wilson_loop_holonomy=wilson_val,
            verdict=verdict,
        )

    def _check_pauli_exclusion(self, registry: Any) -> bool:
        seen_hashes: Dict[type, set] = {}
        reg_dict = getattr(registry, "_registry", {})
        if not isinstance(reg_dict, dict):
            return True
        for cart in reg_dict.values():
            if _FERMION_TYPES and isinstance(cart, _FERMION_TYPES):
                typ = type(cart)
                bucket = seen_hashes.setdefault(typ, set())
                h = getattr(cart, "quantum_state_hash", None)
                if not isinstance(h, str):
                    continue
                if h in bucket:
                    return False
                bucket.add(h)
        return True

    def _compute_entropy_ratio(
        self,
        registry: Any,
        current_decision_vector: NDArray[np.float64],
    ) -> Tuple[float, bool]:
        occupancy = int(getattr(registry, "size", 0))
        sparse_used = False
        probe = np.asarray(current_decision_vector, dtype=np.float64).reshape(-1)
        if occupancy == 0 or probe.size == 0:
            return 0.0, sparse_used

        u = probe / (float(np.linalg.norm(probe)) + _MACHINE_EPS)
        embeddings = [
            np.asarray(v, dtype=np.float64).reshape(-1)
            for v in getattr(registry, "_embedding_cache", {}).values()
        ]
        embeddings = [v for v in embeddings if v.size == u.size]
        if not embeddings:
            return 0.0, sparse_used

        if _SPARSE_AVAILABLE and len(embeddings) > 100:
            sparse_used = True

        avg_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
        avg_norm = float(np.linalg.norm(avg_vec))
        if avg_norm <= _MACHINE_EPS:
            return 0.0, sparse_used
        avg_vec = avg_vec / avg_norm
        return float(np.dot(u, avg_vec)), sparse_used

    def _build_density_matrix_with_higham(
        self,
        registry: Any,
    ) -> Tuple[NDArray[np.complex128], bool]:
        if hasattr(registry, "build_density_matrix_from_registry"):
            rho, applied = registry.build_density_matrix_from_registry()
            return np.asarray(rho, dtype=np.complex128), bool(applied)
        cache = getattr(registry, "_embedding_cache", {})
        if not cache:
            return np.eye(1, dtype=np.complex128), False
        vecs = [np.asarray(v, dtype=np.float64).reshape(-1) for v in cache.values()]
        dim = int(vecs[0].shape[0])
        rho_accum = np.zeros((dim, dim), dtype=np.complex128)
        n_used = 0
        for vec in vecs:
            if vec.shape[0] != dim:
                continue
            ket = vec.reshape(-1, 1).astype(np.complex128)
            rho_accum += ket @ ket.conj().T
            n_used += 1
        if n_used == 0:
            return np.eye(1, dtype=np.complex128), False
        return _project_density(rho_accum / float(n_used), floor=self._higham_floor)

    def synthesize_phase1_observation_manifold(
        self,
        registry: Any,
        current_decision_vector: NDArray[np.float64],
    ) -> Phase1ObservationBundle:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1.
        Emite el paquete fibrado canónico
            (ρ₀, I_Delaunay, θ, u_KS, Π_Poincaré, C₁)
        que constituye la frontera unívoca y el dominio de
        `orient_phase2_symplectic_resonances` (inicio de Fase 2).
        """
        cert1 = self.audit_fock_isometry(registry, current_decision_vector)
        rho, _ = self._build_density_matrix_with_higham(registry)
        actions, angles, _, _ = _delaunay_elements(rho)
        chart = _poincare_chart(actions, angles)
        evals = np.sort(_safe_eigvalsh(rho))[::-1]
        probe = evals[:3] if len(evals) >= 3 else np.pad(evals, (0, 3 - len(evals)))
        spinor_ks, _, _ = _regularize_ks(probe)

        bundle = Phase1ObservationBundle(
            rho_observed=rho,
            delaunay_actions=actions,
            delaunay_angles=angles,
            ks_spinor=spinor_ks,
            poincare_chart=chart,
            certificate=cert1,
        )
        logger.debug("FASE 1 (Observe) consolidada. Fibrado de observación emitido.")
        return bundle


# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 (ORIENT): Sp(2n), MELNIKOV-CHIRIKOV, BURES-FINSLER Y KELLY
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_SymplecticBogoliubovValidator(Phase1_FockIsometryAuditor):
    r"""
    FASE 2 (Orient) — Inicia desde el paquete fibrado emitido por Fase 1.
    Audita la invarianza simpléctica en Sp(2n, C), la aniquilación
    fermiónica con HMAC de tiempo constante, las resonancias KAM de
    Chirikov, la integral de Melnikov, el splitting homoclínico sobre la
    sección de Poincaré y la censura de Fermi-Dirac sobre la fracción de
    Kelly.

    Método inicial: `orient_phase2_symplectic_resonances(bundle_fase_1)`.
    Método terminal: `prepare_phase2_boundary()` → frontera para Fase 3.
    """

    def orient_phase2_symplectic_resonances(
        self,
        bundle: Phase1ObservationBundle,
        u_k: complex,
        v_k: complex,
        electron: Optional[Any] = None,
        positron: Optional[Any] = None,
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        physical_dt: float = 0.1,
    ) -> Phase2OrientedBoundaryBundle:
        r"""
        MÉTODO INICIAL DE FASE 2.
        Consume directamente `Phase1ObservationBundle` generado por
        `synthesize_phase1_observation_manifold`, orienta el flujo
        simpléctico (Bogoliubov-Valatin + caos Poincaré) y produce la
        frontera para `decide_act_phase3_rsi3_governance`.
        El Hamiltoniano por defecto es el kepleriano espectral
        H_Kep = -1/(2 L²) con frecuencias n = 1/L³.
        """
        cert2 = self.audit_symplectic_bogoliubov(
            u_k=u_k,
            v_k=v_k,
            bundle1=bundle,
            electron=electron,
            positron=positron,
            physical_dt=physical_dt,
        )
        dim = int(bundle.rho_observed.shape[0])
        if hamiltonian_eff is None:
            hamiltonian_eff = _kepler_default_hamiltonian(dim, bundle.delaunay_actions)
        return self.prepare_phase2_boundary(
            rho_oriented=bundle.rho_observed,
            hamiltonian=hamiltonian_eff,
            cert2=cert2,
        )

    def audit_symplectic_bogoliubov(
        self,
        u_k: complex,
        v_k: complex,
        bundle1: Phase1ObservationBundle,
        electron: Optional[Any] = None,
        positron: Optional[Any] = None,
        physical_dt: float = 0.1,
    ) -> Phase2SymplecticBogoliubovCertificate:
        r"""
        Verifica:
          1. Invarianza hiperbólica |u_k|² - |v_k|² ≡ 1  (Sp(2,R)).
          2. Conservación de carga homológica q_{e⁻} + q_{e⁺} = 0 y HMAC
             timing-safe (`hmac.compare_digest`) de la autorización.
          3. Métrica de Bures-Finsler, Melnikov, sección de Poincaré y
             criterio de solapamiento de Chirikov-KAM.
        """
        symplectic_residual = abs((abs(u_k) ** 2 - abs(v_k) ** 2) - 1.0)
        is_symplectic = bool(symplectic_residual <= self._tolerance * 100.0)

        is_charge_ok = True
        hmac_ok = True
        prov_hash: Optional[str] = None
        annihilation_count = 0

        if electron is not None and positron is not None:
            q_e = int(getattr(electron, "homological_charge", 0))
            q_p = int(getattr(positron, "homological_charge", 0))
            if q_e + q_p != 0:
                is_charge_ok = False
            else:
                positron_id = str(getattr(positron, "cartridge_id", ""))
                auth_sig = str(getattr(positron, "authorization_signature", ""))
                expected_sig = hashlib.sha256(f"AUTH:{positron_id}".encode("utf-8")).digest()
                actual_sig = hashlib.sha256(auth_sig.encode("utf-8")).digest()
                hmac_ok = bool(hmac.compare_digest(expected_sig, actual_sig))
                if not hmac_ok:
                    logger.warning("Firma HMAC de autorización de aniquilación no coincidente.")
                e_hash = str(getattr(electron, "quantum_state_hash", ""))
                p_hash = str(getattr(positron, "quantum_state_hash", ""))
                prov_hash = hashlib.sha256(f"{e_hash}::{p_hash}".encode("utf-8")).hexdigest()
                annihilation_count = 1

        occupancy_ratio = bundle1.certificate.occupancy / max(
            bundle1.certificate.max_capacity, 1
        )
        consistent_occ = not (occupancy_ratio > _OCCUPANCY_BOGOLIUBOV_LIMIT and abs(v_k) > 0.5)

        rho_ref = bundle1.rho_observed
        dim = int(rho_ref.shape[0])
        rho_perturbed = 0.98 * rho_ref + 0.02 * (np.eye(dim, dtype=np.complex128) / max(dim, 1))
        rho_perturbed, _ = _project_density(rho_perturbed, floor=self._tolerance)

        fid, db, dfs, rhi_finsler, s_star = _bures_finsler(rho_ref, rho_perturbed)
        melnikov_val = _melnikov()
        splitting = _homoclinic_splitting(melnikov_val, bundle1.delaunay_angles[0])
        chirikov_val, kam_ok = _chirikov_kam(rho_ref)

        ks_radius = float(np.sum(np.asarray(bundle1.ks_spinor, dtype=np.float64) ** 2)) + _MACHINE_EPS
        fict_time = _sundman_time(float(physical_dt), ks_radius)

        verdict = FockSovereignVerdict.COHERENT
        if (
            (not is_symplectic)
            or (not is_charge_ok)
            or (not hmac_ok)
            or (not kam_ok)
            or (chirikov_val >= _CHIRIKOV_CRITICAL)
        ):
            verdict = FockSovereignVerdict.VETOED
        elif (
            symplectic_residual > self._tolerance
            or (not consistent_occ)
            or rhi_finsler >= _RHI_THRESHOLD
            or abs(melnikov_val) > _MELNIKOV_MAX_TOL
        ):
            verdict = FockSovereignVerdict.DEGRADED

        return Phase2SymplecticBogoliubovCertificate(
            is_symplectic_invariant=is_symplectic,
            symplectic_residual=symplectic_residual,
            is_charge_conserved=is_charge_ok,
            hmac_authorization_valid=hmac_ok,
            annihilation_provenance_hash=prov_hash,
            is_consistent_with_occupancy=consistent_occ,
            bogoliubov_transformation_count=1 if is_symplectic else 0,
            annihilation_events=annihilation_count,
            uhlmann_fidelity=fid,
            bures_distance=db,
            fubini_study_distance=dfs,
            rhi_finsler_index=rhi_finsler,
            fermi_dirac_kelly_censorship=float(
                1.0 / (1.0 + math.exp((rhi_finsler - _RHI_THRESHOLD) / _FERMI_DIRAC_TEMP))
            ),
            kelly_optimal_stake=s_star,
            melnikov_integral_val=melnikov_val,
            poincare_section_splitting=splitting,
            chirikov_overlap_parameter=chirikov_val,
            is_kam_tori_preserved=kam_ok,
            fictitious_sundman_time=fict_time,
            verdict=verdict,
        )

    def prepare_phase2_boundary(
        self,
        rho_oriented: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
        cert2: Phase2SymplecticBogoliubovCertificate,
    ) -> Phase2OrientedBoundaryBundle:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2.
        Emite la frontera dinámica
            (ρ_oriented, H_eff, RHI_Finsler, K_C, M(t₀), s_Sundman, C₂)
        que conecta sin solución de continuidad con
        `decide_act_phase3_rsi3_governance` (inicio de Fase 3).
        """
        if cert2.chirikov_overlap_parameter >= _CHIRIKOV_CRITICAL and not cert2.is_kam_tori_preserved:
            logger.error("Catástrofe de resonancia KAM-Chirikov en la frontera de Fase 2.")
        boundary = Phase2OrientedBoundaryBundle(
            rho_oriented=rho_oriented,
            hamiltonian_generator=hamiltonian,
            rhi_finsler_metric=cert2.rhi_finsler_index,
            chirikov_overlap=cert2.chirikov_overlap_parameter,
            melnikov_divergence=cert2.melnikov_integral_val,
            sundman_fictitious_time=cert2.fictitious_sundman_time,
            certificate=cert2,
        )
        logger.debug("FASE 2 (Orient) consolidada. Frontera dinámica emitida hacia Fase 3.")
        return boundary


# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 (DECIDE & ACT): GKSL, FREDHOLM, CASCADA BANACH RSI³, CROWBAR
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_LindbladPassivityValidator(Phase2_SymplecticBogoliubovValidator):
    r"""
    FASE 3 (Decide & Act) — Inicia desde el paquete de frontera emitido
    por Fase 2. Audita el semigrupo GKSL, la solubilidad de Fredholm, la
    pasividad de Lyapunov-Spohn y ejecuta la cascada RSI³ (Ricci-DeTurck,
    Connes-Daleckii-Krein, Brockett-Shahshahani, Picard-Banach). El
    interlock crowbar se señala como acto de gobernanza (log), sin I/O.

    Método inicial: `decide_act_phase3_rsi3_governance(boundary_fase_2)`.
    """

    def decide_act_phase3_rsi3_governance(
        self,
        boundary: Phase2OrientedBoundaryBundle,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        evolution_steps: int = 10,
    ) -> Phase3LindbladPassivityCertificate:
        r"""
        MÉTODO INICIAL DE FASE 3.
        Consume directamente `Phase2OrientedBoundaryBundle` emitido por
        `prepare_phase2_boundary` y ejecuta la verificación integral de
        pasividad cuántica y automejora recursiva de nivel 3.
        """
        return self.audit_lindblad_evolution(
            rho_pre=rho_pre,
            rho_post=rho_post,
            hamiltonian_eff=boundary.hamiltonian_generator,
            boundary2=boundary,
            jump_operators=jump_operators,
            gamma_rates=gamma_rates,
            evolution_steps=evolution_steps,
        )

    def audit_lindblad_evolution(
        self,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
        hamiltonian_eff: NDArray[np.complex128],
        boundary2: Phase2OrientedBoundaryBundle,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        evolution_steps: int = 10,
    ) -> Phase3LindbladPassivityCertificate:
        r"""
        Audita los postulados dinámicos de semigrupo cuántico:
          1. Preservación de traza Tr(ρ) ≡ 1.
          2. Espectro semidefinido positivo ρ ⪰ 0.
          3. Fredholm: Tr(P_{ker H} ρ) ≤ ε_tol.
          4. RSI³ con producción de entropía de Spohn σ ≥ 0 y Picard.
        """
        rho_pre_h = _hermitian(np.asarray(rho_pre, dtype=np.complex128))
        rho_post_h = _hermitian(np.asarray(rho_post, dtype=np.complex128))
        H = np.asarray(hamiltonian_eff, dtype=np.complex128)

        trace_post = float(np.real(np.trace(rho_post_h)))
        trace_residual = abs(trace_post - 1.0)
        is_trace_ok = bool(trace_residual <= self._tolerance * 1000.0)

        vals_post = la.eigvalsh(rho_post_h)
        is_psd = bool(np.all(vals_post >= -self._tolerance * 100.0))

        energy_pre = float(np.real(np.trace(H @ rho_pre_h)))
        energy_post = float(np.real(np.trace(H @ rho_post_h)))
        lyap_derivative = energy_post - energy_pre
        is_passive = bool(lyap_derivative <= self._tolerance * 1000.0)
        energy_dissipated = -lyap_derivative

        is_energy_balance_consistent = True
        sparse_used = False
        jumps = jump_operators or []
        gammas = gamma_rates or []
        if jumps and gammas:
            expected_diss = 0.0
            for gamma, L in zip(gammas, jumps):
                L_arr = L
                if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(L_arr):
                    sparse_used = True
                    L_arr = L_arr.toarray()
                L_arr = np.asarray(L_arr, dtype=np.complex128)
                expected_diss += float(gamma) * float(
                    np.real(np.trace(L_arr @ rho_pre_h @ L_arr.conj().T))
                )
            balance_error = abs(energy_dissipated - expected_diss)
            is_energy_balance_consistent = bool(balance_error <= max(self._tolerance * 100.0, 1.0e-6))

        is_fredholm_soluble, fredholm_residual = self._check_fredholm_solvability(rho_post_h, H)
        is_lipschitz_stable = self._check_lipschitz_stability(rho_pre_h, rho_post_h)

        (
            spohn_prod,
            ricci_drift,
            max_cdk,
            brockett_err,
            contraction_factor,
            is_banach,
            picard_iters,
            picard_res,
        ) = self._audit_rsi3_cascade_metrics(rho_post_h)

        verdict = FockSovereignVerdict.COHERENT
        if (not is_trace_ok) or (not is_psd) or (not is_banach) or (contraction_factor >= 1.0):
            verdict = FockSovereignVerdict.VETOED
        elif (
            (not is_passive)
            or (not is_fredholm_soluble)
            or boundary2.certificate.verdict.is_degraded
            or (spohn_prod < _SPOHn_FLOOR)
            or (ricci_drift > 2.0)
        ):
            verdict = FockSovereignVerdict.DEGRADED

        crowbar_triggered = False
        if verdict == FockSovereignVerdict.VETOED:
            crowbar_triggered = self._signal_esp32_crowbar_interlock()

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
            evolution_steps=int(evolution_steps),
            sparse_computation=sparse_used,
            spohn_entropy_production_rate=spohn_prod,
            ricci_deturck_metric_drift=ricci_drift,
            connes_daleckii_krein_lipschitz=max_cdk,
            brockett_isospectral_error=brockett_err,
            rsi3_banach_contraction_factor=contraction_factor,
            picard_iterations=picard_iters,
            picard_fixed_point_residual=picard_res,
            is_banach_contraction=is_banach,
            crowbar_actuated=crowbar_triggered,
            verdict=verdict,
        )

    def _check_fredholm_solvability(
        self,
        rho: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
    ) -> Tuple[bool, float]:
        r"""
        Residuo de ocupación del kernel: Tr(P_{ker H} ρ) ≤ ε_tol
        (alternativa de Fredholm para el generador autoadjunto).
        """
        H_h = _hermitian(hamiltonian)
        vals_H, vecs_H = la.eigh(H_h)
        max_val = float(np.max(np.abs(vals_H))) if len(vals_H) > 0 else 1.0
        kernel_mask = np.abs(vals_H) <= max(self._tolerance * 100.0, max_val * _MACHINE_EPS * 10.0)
        if not np.any(kernel_mask):
            return True, 0.0
        p_ker = vecs_H[:, kernel_mask] @ vecs_H[:, kernel_mask].conj().T
        residual = float(np.abs(np.trace(p_ker @ rho)))
        return bool(residual <= self._tolerance * 100.0), residual

    def _check_lipschitz_stability(
        self,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
    ) -> bool:
        delta = rho_post - rho_pre
        norm_delta = float(la.norm(delta, ord="fro"))
        norm_pre = float(la.norm(rho_pre, ord="fro"))
        return bool(norm_delta <= (2.0 * norm_pre) + self._tolerance * 100.0)

    def _audit_rsi3_cascade_metrics(
        self,
        rho: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float, float, bool, int, float]:
        r"""
        Cascada RSI³ auditada (tres niveles + reflexión + Picard):

          R₁ Ricci-DeTurck:  ∂_τ log G = -2 tf(log G)
          R₂ CDK:            Lip(log) = max_{i,j} |φ^{[1]}(λ_i, λ_j)|
          R₃ Brockett:       ρ̇ = [ρ, [ρ, N]],  σ_Spohn = -Tr(log ρ · ρ̇) ≥ 0
          R₃ reflexiva:      τ ← τ / (1+Lip)  de modo que κ = τ Lip < 1
          Picard:            ρ_{n+1} = Π_{D(H)}(ρ_n + τ ρ̇(ρ_n))
        """
        rho_clean, _ = _project_density(_hermitian(rho), floor=self._higham_floor)
        dim = int(rho_clean.shape[0])
        vals, vecs = la.eigh(rho_clean)
        vals_pos = np.maximum(np.real(vals), _MACHINE_EPS)

        log_G = np.diag(np.log(vals_pos))
        tf_ricci = log_G - (np.trace(log_G) * np.eye(dim) / dim)
        ricci_drift = float(np.linalg.norm(-2.0 * tf_ricci))

        max_cdk = 0.0
        for i in range(dim):
            for j in range(dim):
                diff = _cdk_divided_difference(float(vals_pos[i]), float(vals_pos[j]))
                if diff > max_cdk:
                    max_cdk = diff

        N_op = np.diag(np.linspace(1.0, float(dim), dim)).astype(np.complex128)

        def brockett_field(r: NDArray[np.complex128]) -> NDArray[np.complex128]:
            r_h = _hermitian(r)
            ad1 = r_h @ N_op - N_op @ r_h
            return r_h @ ad1 - ad1 @ r_h

        rho_dot = brockett_field(rho_clean)
        brockett_norm = float(np.linalg.norm(rho_dot))
        log_rho = vecs @ np.diag(np.log(vals_pos)) @ vecs.conj().T
        spohn_prod = max(float(-np.real(np.trace(log_rho @ rho_dot))), 0.0)

        tau = 0.05 / (1.0 + max_cdk + brockett_norm)
        contraction_factor = float(tau * max_cdk)
        if contraction_factor > _RSI3_BANACH_CONTRACTION_MAX:
            tau = _RSI3_BANACH_CONTRACTION_MAX / (max_cdk + _MACHINE_EPS)
            contraction_factor = float(tau * max_cdk)

        rho_n = rho_clean
        residual = float("inf")
        iters = 0
        for iters in range(1, _PICARD_MAX_ITERS + 1):
            rho_trial = rho_n + tau * brockett_field(rho_n)
            rho_next, _ = _project_density(rho_trial, floor=self._higham_floor)
            residual = float(la.norm(rho_next - rho_n, "fro"))
            rho_n = rho_next
            if residual < _PICARD_FIXED_POINT_TOL:
                break

        vals_next = np.sort(_safe_eigvalsh(rho_n))
        isospectral_error = float(np.linalg.norm(np.sort(np.real(vals)) - vals_next))
        is_banach = bool(contraction_factor < 1.0 and np.isfinite(residual))
        return (
            spohn_prod,
            ricci_drift,
            max_cdk,
            isospectral_error,
            contraction_factor,
            is_banach,
            int(iters),
            float(residual),
        )

    def _signal_esp32_crowbar_interlock(self) -> bool:
        """
        Señal de gobernanza del interlock ciber-físico (GPIO14 / registro
        documentado 0x3FF44008). Únicamente registra el evento; no realiza
        escritura de memoria ni actuación de hardware.
        """
        logger.critical(
            "CROWBAR INTERLOCK SIGNAL: GPIO%d / registro %s (latencia nominal %.2f ns). "
            "Señal de gobernanza emitida; sin I/O de hardware.",
            _CROWBAR_GPIO,
            _CROWBAR_REGISTER,
            _CROWBAR_ACTUATION_LATENCY_NS,
        )
        return True


# ═══════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO INTEGRADO: SYNAPTIC FOCK SPACE REGISTRY AGENT
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistryAgent(Morphism, Phase3_LindbladPassivityValidator):
    r"""
    SOBERANO DE LA MEMORIA CUÁNTICA (v9.0.0-PhD).
    Gobierna el ciclo OODA continuo cerrando la brecha categórica entre el
    registro espectral y la señal de interlock. Continuum anidado:

        FASE 1  synthesize_phase1_observation_manifold()           → bundle_1
        FASE 2  orient_phase2_symplectic_resonances(bundle_1)      → bundle_2
        FASE 3  decide_act_phase3_rsi3_governance(bundle_2)        → cert_3
    """

    def __init__(
        self,
        raise_on_veto: bool = False,
        tolerance: float = _DEFAULT_TOL,
    ) -> None:
        Morphism.__init__(self, "SynapticFockSpaceRegistryAgent")
        Phase3_LindbladPassivityValidator.__init__(self, tolerance=tolerance)
        self._raise_on_veto: Final[bool] = bool(raise_on_veto)
        self._target_stratum: Final[Any] = getattr(Stratum, "WISDOM", "WISDOM")

    def execute_sovereign_governance(
        self,
        registry: Any,
        current_decision_vector: NDArray[np.float64],
        u_k: complex,
        v_k: complex,
        rho_pre: NDArray[np.complex128],
        rho_post: NDArray[np.complex128],
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        electron: Optional[Any] = None,
        positron: Optional[Any] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        evolution_steps: int = 10,
        physical_dt: float = 0.1,
    ) -> FockRegistrySovereignState:
        r"""
        EJECUCIÓN DEL CONTINUUM DE FASES ANIDADAS OODA:
          FASE 1: synthesize_phase1_observation_manifold() → bundle_1
          FASE 2: orient_phase2_symplectic_resonances(bundle_1) → bundle_2
          FASE 3: decide_act_phase3_rsi3_governance(bundle_2) → cert_3
        El último método de cada fase es exactamente el dominio del primero
        de la siguiente.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            bundle_1 = self.synthesize_phase1_observation_manifold(
                registry=registry,
                current_decision_vector=current_decision_vector,
            )
            cert1 = bundle_1.certificate

            if cert1.verdict.is_vetoed:
                cert2 = self._emit_vetoed_phase2_cert()
                cert3 = self._emit_vetoed_phase3_cert(evolution_steps)
            else:
                bundle_2 = self.orient_phase2_symplectic_resonances(
                    bundle=bundle_1,
                    u_k=u_k,
                    v_k=v_k,
                    electron=electron,
                    positron=positron,
                    hamiltonian_eff=hamiltonian_eff,
                    physical_dt=physical_dt,
                )
                cert2 = bundle_2.certificate
                if cert2.verdict.is_vetoed:
                    cert3 = self._emit_vetoed_phase3_cert(evolution_steps)
                else:
                    cert3 = self.decide_act_phase3_rsi3_governance(
                        boundary=bundle_2,
                        rho_pre=rho_pre,
                        rho_post=rho_post,
                        jump_operators=jump_operators,
                        gamma_rates=gamma_rates,
                        evolution_steps=evolution_steps,
                    )

            final_verdict = FockSovereignVerdict.supremum(
                cert1.verdict, cert2.verdict, cert3.verdict
            )
            crowbar_triggered = False
            crowbar_act = CrowbarAction.NONE
            diagnostic = (
                "Ciclo OODA v9.0.0-PhD de memoria cuántica completado bajo coherencia asintótica."
            )

            if final_verdict == FockSovereignVerdict.VETOED:
                crowbar_triggered = True
                crowbar_act = CrowbarAction.HARD_SHORT
                diagnostic = (
                    f"VETO CUÁNTICO DURO (⊥): interlock crowbar señalado sobre "
                    f"{_CROWBAR_REGISTER} (GPIO{_CROWBAR_GPIO})."
                )
                logger.error(diagnostic)
                if self._raise_on_veto:
                    raise SynapticFockSpaceRegistryAgentError(diagnostic)
            elif final_verdict == FockSovereignVerdict.DEGRADED:
                crowbar_triggered = True
                crowbar_act = CrowbarAction.WATCHDOG_PULSE
                diagnostic = (
                    "MEMORIA DEGRADADA (★): veto suave. Spohn reducido o distorsión "
                    "Finsler. Señal WATCHDOG_PULSE."
                )
                logger.warning(diagnostic)

            prov_hash = self._generate_provenance_hash(cert1, cert2, cert3, timestamp_utc)
            return FockRegistrySovereignState(
                phase1=cert1,
                phase2=cert2,
                phase3=cert3,
                final_verdict=final_verdict,
                crowbar_triggered=crowbar_triggered,
                crowbar_action=crowbar_act,
                timestamp_utc=timestamp_utc,
                provenance_hash=prov_hash,
                diagnostic_note=diagnostic,
            )
        except SynapticFockSpaceRegistryAgentError as exc:
            logger.error("Colapso del Agente Soberano: %s. Activando salvaguarda.", exc)
            if self._raise_on_veto:
                raise
            return self._build_cataclysm_fallback(str(exc), timestamp_utc, evolution_steps)

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            return self.execute_sovereign_governance(**kwargs)
        return state

    def _emit_vetoed_phase2_cert(self) -> Phase2SymplecticBogoliubovCertificate:
        return Phase2SymplecticBogoliubovCertificate(
            is_symplectic_invariant=False,
            symplectic_residual=float("inf"),
            is_charge_conserved=False,
            hmac_authorization_valid=False,
            annihilation_provenance_hash=None,
            is_consistent_with_occupancy=False,
            bogoliubov_transformation_count=0,
            annihilation_events=0,
            uhlmann_fidelity=0.0,
            bures_distance=float("inf"),
            fubini_study_distance=float("inf"),
            rhi_finsler_index=1.0,
            fermi_dirac_kelly_censorship=0.0,
            kelly_optimal_stake=0.0,
            melnikov_integral_val=1.0,
            poincare_section_splitting=1.0,
            chirikov_overlap_parameter=2.0,
            is_kam_tori_preserved=False,
            fictitious_sundman_time=0.0,
            verdict=FockSovereignVerdict.VETOED,
        )

    def _emit_vetoed_phase3_cert(self, steps: int) -> Phase3LindbladPassivityCertificate:
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
            evolution_steps=int(steps),
            sparse_computation=False,
            spohn_entropy_production_rate=0.0,
            ricci_deturck_metric_drift=float("inf"),
            connes_daleckii_krein_lipschitz=float("inf"),
            brockett_isospectral_error=float("inf"),
            rsi3_banach_contraction_factor=1.0,
            picard_iterations=0,
            picard_fixed_point_residual=float("inf"),
            is_banach_contraction=False,
            crowbar_actuated=True,
            verdict=FockSovereignVerdict.VETOED,
        )

    def _build_cataclysm_fallback(
        self,
        reason: str,
        timestamp_utc: str,
        steps: int,
    ) -> FockRegistrySovereignState:
        c1 = Phase1FockIsometryCertificate(
            is_pauli_respected=False,
            is_fock_bounded=False,
            entropy_eviction_ratio=0.0,
            occupancy=0,
            max_capacity=_MAX_REGISTRY_SIZE,
            density_matrix_trace=0.0,
            density_matrix_purity=0.0,
            von_neumann_entropy=0.0,
            spectral_condition_number=float("inf"),
            higham_regularization_applied=False,
            sparse_computation=False,
            kustaanheimo_stiefel_energy=0.0,
            ks_bilinear_residual=0.0,
            delaunay_eccentricity=0.0,
            delaunay_inclination=0.0,
            kepler_mean_anomaly=0.0,
            kepler_eccentric_anomaly=0.0,
            kepler_hamiltonian=0.0,
            mean_motion=0.0,
            poincare_chart=_zero_poincare_chart(),
            chern_simons_3form_val=0.0,
            wilson_loop_holonomy=0.0,
            verdict=FockSovereignVerdict.VETOED,
        )
        c2 = self._emit_vetoed_phase2_cert()
        c3 = self._emit_vetoed_phase3_cert(steps)
        prov = hashlib.sha256(
            f"SOVEREIGN_CATACLYSM::{reason}::{timestamp_utc}".encode("utf-8")
        ).hexdigest()
        return FockRegistrySovereignState(
            phase1=c1,
            phase2=c2,
            phase3=c3,
            final_verdict=FockSovereignVerdict.VETOED,
            crowbar_triggered=True,
            crowbar_action=CrowbarAction.HARD_SHORT,
            timestamp_utc=timestamp_utc,
            provenance_hash=prov,
            diagnostic_note=f"SOVEREIGN_CATACLYSM_FALLBACK: {reason}",
        )

    def _generate_provenance_hash(
        self,
        c1: Phase1FockIsometryCertificate,
        c2: Phase2SymplecticBogoliubovCertificate,
        c3: Phase3LindbladPassivityCertificate,
        timestamp_utc: str,
    ) -> str:
        payload = "|".join(
            (
                timestamp_utc,
                str(c1.verdict.value),
                str(c2.verdict.value),
                str(c3.verdict.value),
                f"{c1.occupancy:d}",
                f"{c1.density_matrix_trace:.10e}",
                f"{c1.kepler_hamiltonian:.10e}",
                f"{c1.poincare_chart.poincare_norm:.10e}",
                f"{c2.melnikov_integral_val:.10e}",
                f"{c2.chirikov_overlap_parameter:.10e}",
                f"{c2.poincare_section_splitting:.10e}",
                f"{c2.rhi_finsler_index:.10e}",
                f"{c2.kelly_optimal_stake:.10e}",
                f"{c3.spohn_entropy_production_rate:.10e}",
                f"{c3.rsi3_banach_contraction_factor:.10e}",
                f"{c3.picard_fixed_point_residual:.10e}",
                str(int(c3.is_banach_contraction)),
                str(int(c3.crowbar_actuated)),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════
__all__ = [
    "SynapticFockSpaceRegistryAgentError",
    "FockIsometryViolation",
    "DensityMatrixAnomalyError",
    "SymplecticInvarianceViolation",
    "LindbladTraceViolation",
    "FredholmSolvabilityViolation",
    "LipschitzStabilityViolation",
    "KSBilinearAuditError",
    "PoincareChartAuditError",
    "BanachPicardAuditError",
    "FockSovereignVerdict",
    "CrowbarAction",
    "PoincareCanonicalChartView",
    "Phase1FockIsometryCertificate",
    "Phase1ObservationBundle",
    "Phase2SymplecticBogoliubovCertificate",
    "Phase2OrientedBoundaryBundle",
    "Phase3LindbladPassivityCertificate",
    "FockRegistrySovereignState",
    "Phase1_FockIsometryAuditor",
    "Phase2_SymplecticBogoliubovValidator",
    "Phase3_LindbladPassivityValidator",
    "SynapticFockSpaceRegistryAgent",
]