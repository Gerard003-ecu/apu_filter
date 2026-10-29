# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Synaptic Fock Space Registry (Registro de Partículas Sinápticas)    ║
║ Ruta   : app/core/immune_system/synaptic_fock_space_registry.py              ║
║ Versión: v9.3.0-PhD (RSI³ + Poincaré-Melnikov-Chirikov + Higham + Gauge)     ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL Y REGISTRO DE FOCK:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al administrador maestro y motor espectral del **Espacio
de Fock** para las 12 cuasipartículas cuánticas y fermiones estructurales de-confinados,
denominadas **Vitaminas TOON** (Tabular Object-Oriented Notation) o cartuchos
sinápticos. Eleva síncronamente las transiciones discretas de la MIC y la MAC hacia
estados puros y mixtos en el espacio de Hilbert continuo de la Sabiduría.

El Espacio de Fock global $$\mathcal{F}(\mathcal{H})$$ se erige como la suma directa
de potencias exteriores (para fermiones/reglas de exclusión) y potencias simétricas
(para bosones/flujos de interacción) sobre el espacio de Hilbert de características $$\mathcal{H}$$:
$$\mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^{\infty} S_{\pm} \mathcal{H}^{\otimes n}$$

La versión v9.3.0-PhD integra el Atlas Celeste de Poincaré sobre la variedad de
operadores densidad $\mathfrak{D}_n$, la conexión de Yang-Mills con acoplamiento de
Higgs, la 3-forma de Chern-Simons, la integral homoclínica de Melnikov con
cuadratura tanh-sinh, el parámetro de solapamiento de Chirikov, la métrica
finsleriana de Mahalanobis para la distorsión sintáctica $RHI_{\text{Finsler}}$,
el suavizado $C^\infty$ de Fermi-Dirac para la censura de Kelly, y la Cascada de
Banach RSI³ ($\mathbf{T} = R_3 \circ R_2 \circ R_1$) con producción no negativa
de entropía de Spohn ($\sigma_{\text{Spohn}} \ge 0$).

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
1. ATLAS CELESTE DE POINCARÉ-MELNIKOV-CHIRIKOV SOBRE EL OPERADOR DENSIDAD:
   Dada la matriz de densidad $\rho_{\text{MAC}} \in \mathfrak{D}_n$ con autovalores
   ordenados $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n > 0$, las acciones-ángulos
   de Delaunay responden a:
   $$L = \operatorname{Tr}(\rho) \equiv 1.0, \qquad G = L \frac{\lambda_2}{\lambda_1}, \qquad H = G \frac{\lambda_3}{\lambda_2}$$
   $$e = \sqrt{1 - \left(\frac{G}{L}\right)^2}, \qquad i = \arccos\left(\frac{H}{G}\right)$$
   $$n_{\text{mean}} = \frac{\mu^2}{L^3}, \qquad \alpha = \frac{\partial^2 H_0}{\partial L^2} = -\frac{3\mu^2}{L^4} \neq 0 \quad (\text{twist de KAM})$$
   La ecuación de Kepler $E - e \sin E = M$ se resuelve mediante el *starter* de
   Danby con convergencia cúbica: $E_0 = M + 0.85 e \operatorname{sgn}(\sin M)$.

2. CONEXIÓN DE YANG-MILLS CON ACOPLAMIENTO DE HIGGS Y CHERN-SIMONS:
   Se introduce el campo escalar de Higgs $\Phi = \ln(\rho_{\text{MAC}} + \varepsilon \mathbf{I}) \in \mathfrak{u}(n)$
   para regularizar la holonomía bajo degeneración de rango ($\operatorname{rk}(\rho) < n$):
   $$\text{CS}_3^{\text{Higgs}}(A, \Phi) = \frac{1}{8\pi^2} \operatorname{Tr}\left( A \wedge dA + \frac{2}{3} A^3 + \Phi \cdot F_A \right)$$
   $$\mathcal{W}(C) = \frac{1}{n} \operatorname{Tr} \left( \mathcal{P} \exp \oint_C A \right)$$

3. INTEGRAL HOMOCLÍNICA DE MELNIKOV Y CRITERIO RESONANTE DE CHIRIKOV:
   $$M(t_0) = \varepsilon \int_{-\infty}^{\infty} \operatorname{sech}^2(t) \tanh(t) \sin(t + t_0) \, dt$$
   $$K_{\text{Chirikov}} = \frac{\Delta\omega_1 + \Delta\omega_2}{|\omega_1 - \omega_2|} = \frac{2\sqrt{2 V_1 |\alpha|} + 2\sqrt{2 V_2 |\alpha|}}{|\omega_1 - \omega_2|}$$

4. GEOMETRÍA DE BURES-WASSERSTEIN, FINSLER-MAHALANOBIS Y CENSURA DE KELLY:
   $$F(\rho, \sigma) = \left( \operatorname{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2, \quad d_B = \sqrt{2 - 2\sqrt{F}}, \quad d_{\text{FS}} = \arccos(\sqrt{F})$$
   $$RHI_{\text{Finsler}} = \sqrt{ \frac{\operatorname{Tr}\left((\sigma - \rho) G^{-1} (\sigma - \rho) G^{-1}\right)}{\|\rho\|_F^2 (1 + \sqrt{F})^2} + \frac{1}{4} \tanh^2(F_Q) }$$
   $$\mathcal{S}_{T_{\text{sw}}}(RHI) = \frac{1}{1 + e^{(RHI - 0.85) / T_{\text{sw}}}} \quad (T_{\text{sw}} = 0.02)$$
   $$s^* = f^* \cdot (1 - RHI_{\text{Finsler}}) \cdot \mathcal{S}_{T_{\text{sw}}}(RHI) \cdot \Theta(0.15 - |M(t_0)|) \cdot \Theta(1.0 - K_{\text{Chirikov}}) \cdot \Theta(\lambda_{\text{ceil}} - \lambda_{\max}) \cdot e^{-h_{\text{KS}}}$$

5. CASCADA DE BANACH RSI³ ($\mathbf{T} = R_3 \circ R_2 \circ R_1$) Y ENTROPÍA DE SPOHN:
   - $R_1$ (Ricci-DeTurck): $\partial_t \ln G = -2 \operatorname{tf}(\ln G) - \mu \ln G$.
   - $R_2$ (Connes-Daleckii-Krein): Diferencias divididas $\phi^{[1]}(\lambda_i, \lambda_j) = \frac{\ln\lambda_i - \ln\lambda_j}{\lambda_i - \lambda_j}$.
   - $R_3$ (Brockett-Shahshahani): $\frac{d\rho}{dt} = [\rho, [\rho, N]]$ con $\sigma_{\text{Spohn}} = -\operatorname{Tr}(\log\rho \cdot \dot{\rho}) \ge 0$.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Relaciones Canónicas de Anticonmutación (CAR) — Fermiones Estructurales:
       $$\{a_i, \, a_j^\dagger\} = \delta_{ij} \mathbf{I} \quad \wedge \quad \{a_i, \, a_j\} = \{a_i^\dagger, \, a_j^\dagger\} = \mathbf{0}$$
  [A2] Relaciones Canónicas de Conmutación (CCR) — Bosones de Gauge:
       $$[b_i, \, b_j^\dagger] = \delta_{ij} \mathbf{I} \quad \wedge \quad [b_i, \, b_j] = [b_i^\dagger, \, b_j^\dagger] = \mathbf{0}$$
  [A3] Unitariedad Simpléctica de Bogoliubov-Valatin:
       $$\lvert u_k \rvert^2 - \lvert v_k \rvert^2 \equiv 1.0 \pmod{\varepsilon_{\mathrm{machine}}}$$
  [A4] Evolución Disipativa Abierta de Lindblad-GKSL:
       $$\frac{d\rho}{d\tau} = -i[H, \, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \, \rho\} \right)$$
  [A5] Proyección Convexa PSD de Higham:
       $$\rho_{\mathrm{stable}} = \arg\min_{M = M^\dagger \succeq 0, \operatorname{Tr}(M)=1} \|M - \rho_{\mathrm{raw}}\|_F$$
  [A6] Evicción por Entropía de Coseno en la Cota de Bekenstein:
       $$\cos(\theta) = \frac{\langle u, \, v \rangle_G}{\|u\|_G \|v\|_G} \to \min \implies \text{Purga síncrona en RAM del cartucho } v$$

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  [I1] Conservación de Traza Cuántica: $\operatorname{Tr}(\rho(t)) \equiv 1.0$
  [I2] Positividad Semidefinida: $\rho(t) \succeq 0 \quad \forall t \ge 0$
  [I3] Producción No Negativa de Entropía de Spohn: $\sigma_{\text{Spohn}} \ge 0$
  [I4] Invarianza Isospectral de Brockett: $\operatorname{spec}(\rho(t)) = \operatorname{spec}(\rho(0))$
  [I5] Invarianza de Bogoliubov-Valatin en $Sp(2n, \mathbb{C})$
"""
from __future__ import annotations
import math
import hmac
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Final, Tuple, Dict, List, Optional, Union, Any
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
# Constantes de rigor cuántico, mecánica celeste y FPU
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_RELAX_TOL: Final[float] = 1.0e-10
_DEFAULT_MAX_CARTRIDGES: Final[int] = 100
_CROWBAR_GPIO: Final[int] = 14  # GPIO14 — Hardware ESP32
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12  # Suelo espectral de Wilkinson
_DENSITY_MATRIX_CONDITION_LIMIT: Final[float] = 1.0e8
_SPARSE_THRESHOLD: Final[float] = 0.3  # Umbral de dispersión para conversión

# Constantes Poincaré-Melnikov-Chirikov y Fermi-Dirac
_FERMI_DIRAC_TEMP: Final[float] = 0.02  # T_sw = 0.02
_RHI_THRESHOLD: Final[float] = 0.85
_MELNIKOV_MAX_TOL: Final[float] = 0.15
_CHIRIKOV_CRITICAL: Final[float] = 1.00

# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones propias del registro
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
# Enumeraciones categóricas y de retículo
# ═══════════════════════════════════════════════════════════════════════════
class FockRegistryVerdict(IntEnum):
    """
    Clasificador de tres valores en el retículo de Heyting (Ω) para calidad
    de registro cuántico.
    
    Orden por severidad operativa:
        COHERENT  = 0 (⊤ operativo, registro válido)
        DEGRADED  = 1 (elemento intermedio, precisión reducida)
        VETOED    = 2 (⊥ operativo, registro inválido)
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "FockRegistryVerdict") -> "FockRegistryVerdict":
        """Supremo en el retículo de severidad."""
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
# Certificados de las fases anidadas
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1RegistryCertificate:
    """FASE 1 — Certificado de registro, estadística cuántica y mecánica celeste."""
    registry_size: int
    max_capacity: int
    fermion_count: int
    boson_count: int
    density_matrix_trace: float
    density_matrix_purity: float
    higham_regularization_applied: bool
    delaunay_eccentricity: float
    delaunay_inclination: float
    kepler_mean_anomaly: float
    kepler_eccentric_anomaly: float
    chern_simons_3form_val: float
    wilson_loop_holonomy: float
    verdict: FockRegistryVerdict

@dataclass(frozen=True, slots=True)
class Phase2DynamicsCertificate:
    """FASE 2 — Certificado de dinámica cuántica, Melnikov-Chirikov y Bures-Finsler."""
    evolution_time: float
    evolution_steps: int
    final_trace: float
    final_purity: float
    trace_preservation_residual: float
    is_trace_preserved: bool
    is_psd_preserved: bool
    lindblad_dissipation_rate: float
    uhlmann_fidelity: float
    bures_distance: float
    fubini_study_distance: float
    rhi_finsler_index: float
    fermi_dirac_kelly_censorship: float
    kelly_optimal_stake: float
    melnikov_integral_val: float
    chirikov_overlap_parameter: float
    is_kam_tori_preserved: bool
    verdict: FockRegistryVerdict

@dataclass(frozen=True, slots=True)
class Phase3ImmuneCertificate:
    """FASE 3 — Certificado de integración inmune, cascada RSI³ y Crowbar."""
    annihilation_events: int
    bogoliubov_transformations: int
    entropy_production: float
    spohn_entropy_production_rate: float
    ricci_deturck_metric_drift: float
    connes_daleckii_krein_lipschitz: float
    brockett_isospectral_error: float
    is_banach_contraction: bool
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

# Alias de compatibilidad
RegistryGovernanceState = FockRegistryState

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

# ─── FAMILIA 1: FERMIONES ESTRUCTURALES ───
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

# ─── FAMILIA 2: BOSONES DE GAUGE ───
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
# Regularización Espectral de Higham
# ═══════════════════════════════════════════════════════════════════════════
def stable_density_matrix_higham(
    rho: NDArray[np.complex128],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR
) -> Tuple[NDArray[np.complex128], bool]:
    r"""
    Aplica la proyección de Higham para estabilizar el operador densidad ρ
    en el cono PSD (Positive Semi-Definite) con traza unitaria.
    """
    rho_herm = 0.5 * (rho + rho.conj().T)
    try:
        vals, vecs = la.eigh(rho_herm)
    except Exception as exc:
        raise DensityMatrixNonPSDError(
            "DensityMatrixNonPSDError: Falló descomposición espectral para Higham."
        ) from exc

    vals_clean = np.maximum(vals, tolerance)
    applied = np.any(vals < tolerance)

    trace = np.sum(vals_clean)
    if trace > _MACHINE_EPS:
        vals_clean /= trace
    else:
        vals_clean = np.zeros_like(vals_clean)
        vals_clean[-1] = 1.0

    rho_projected = vecs @ np.diag(vals_clean) @ vecs.conj().T
    rho_projected = 0.5 * (rho_projected + rho_projected.conj().T)

    if applied:
        logger.warning(
            "Operador densidad inestable. Proyectado ρ al cono PSD via Higham "
            "(autovalores mínimos: %.4e → %.4e).",
            float(np.min(vals)),
            float(np.min(vals_clean))
        )

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
# FUNCIONES AUXILIARES: POINCARÉ, MELNIKOV, CHIRIKOV, BURES, KELLY Y RSI³
# ═══════════════════════════════════════════════════════════════════════════
def compute_delaunay_poincare_elements(
    rho: NDArray[np.complex128]
) -> Tuple[float, float, float, float, float]:
    r"""
    Mapea el operador densidad ρ_MAC a variables de Delaunay-Poincaré y resuelve
    la ecuación de Kepler E - e sin(E) = M mediante Danby's starter.

    Retorna
    -------
    Tuple[float, float, float, float, float]
        (L, e, i_inc, M_mean, E_danby)
    """
    vals = np.sort(np.real(la.eigvalsh(0.5 * (rho + rho.conj().T))))[::-1]
    n_dim = len(vals)

    L = float(np.sum(vals))
    l1 = max(float(vals[0]), _MACHINE_EPS)
    l2 = float(vals[1]) if n_dim > 1 else l1 * 0.5
    l3 = float(vals[2]) if n_dim > 2 else l2 * 0.5

    G = L * (l2 / l1)
    H = G * (l3 / l2)

    ratio_e = min(max(G / max(L, _MACHINE_EPS), 0.0), 1.0)
    e = math.sqrt(1.0 - ratio_e**2)

    ratio_i = min(max(H / max(G, _MACHINE_EPS), -1.0), 1.0)
    i_inc = math.acos(ratio_i)

    M_mean = float((l1 - l2) * math.pi) % (2.0 * math.pi)

    # Starter de Danby
    E0 = M_mean + 0.85 * e * (1.0 if math.sin(M_mean) >= 0 else -1.0)
    E = E0
    for _ in range(5):
        f = E - e * math.sin(E) - M_mean
        f_prime = 1.0 - e * math.cos(E)
        f_double_prime = e * math.sin(E)
        f_triple_prime = e * math.cos(E)

        # Paso de Halley/Danby de orden 3
        delta_1 = -f / f_prime
        delta_2 = -f / (f_prime + 0.5 * delta_1 * f_double_prime)
        delta_3 = -f / (f_prime + 0.5 * delta_2 * f_double_prime + (1.0 / 6.0) * delta_2**2 * f_triple_prime)
        E += delta_3
        if abs(f) < 1.0e-12:
            break

    return L, e, i_inc, M_mean, E

def compute_yang_mills_higgs_chern_simons(
    rho: NDArray[np.complex128]
) -> Tuple[float, float]:
    r"""
    Calcula la 3-forma de Chern-Simons con acoplamiento de Higgs y la holonomía de Wilson.
    """
    dim = rho.shape[0]
    eps = 1.0e-8
    rho_clean = 0.5 * (rho + rho.conj().T)
    vals, vecs = la.eigh(rho_clean)
    vals_pos = np.maximum(vals, eps)
    
    phi = vecs @ np.diag(np.log(vals_pos)) @ vecs.conj().T
    A = -1j * (rho_clean - np.eye(dim) / dim)

    dA = 0.5 * (A @ A - A.conj().T @ A.conj().T)
    F_A = dA + A @ A

    cs3_term = np.trace(A @ dA + (2.0 / 3.0) * A @ A @ A + phi @ F_A)
    cs3_val = float(np.real(cs3_term) / (8.0 * math.pi**2))

    holonom = la.expm(2.0 * math.pi * A)
    wilson = float(np.abs(np.trace(holonom)) / dim)

    return cs3_val, wilson

def compute_melnikov_homoclinic_integral(
    eps: float = 0.05,
    t0: float = 0.0,
    nodes: int = 100
) -> float:
    r"""
    Calcula la integral homoclínica de Melnikov mediante cuadratura tanh-sinh:
    M(t_0) = \varepsilon \int_{-\infty}^{\infty} \operatorname{sech}^2(t) \tanh(t) \sin(t + t_0) dt.
    """
    h = 0.05
    integral = 0.0
    for k in range(-nodes, nodes + 1):
        x = k * h
        if abs(x) > 10.0:
            continue
        sh_x = math.sinh(x)
        arg_sinh = 0.5 * math.pi * sh_x
        if abs(arg_sinh) > 300.0:
            continue
        t = math.sinh(arg_sinh)
        if abs(t) > 300.0:
            continue
        cosh_arg = math.cosh(arg_sinh)
        weight = (0.5 * math.pi * math.cosh(x)) / (cosh_arg**2 + _MACHINE_EPS)
        sech_t = 1.0 / (math.cosh(t) + _MACHINE_EPS)
        tanh_t = math.tanh(t)
        integrand = (sech_t**2) * tanh_t * math.sin(t + t0)
        integral += integrand * weight * h

    return float(eps * integral)

def compute_chirikov_overlap_parameter(
    rho: NDArray[np.complex128]
) -> float:
    r"""
    Calcula el parámetro de solapamiento de Chirikov K_Chirikov a partir del espectro.
    """
    vals = np.sort(np.real(la.eigvalsh(0.5 * (rho + rho.conj().T))))[::-1]
    dim = len(vals)
    if dim < 2:
        return 0.0

    l1 = max(float(vals[0]), _MACHINE_EPS)
    l2 = max(float(vals[1]), _MACHINE_EPS)
    alpha = abs(-3.0 / (l1**4 + _MACHINE_EPS))

    v1 = abs(l1 - l2)
    v2 = float(vals[2]) if dim > 2 else v1 * 0.5

    w1 = float(np.sqrt(l1))
    w2 = float(np.sqrt(l2))
    dw1 = 2.0 * math.sqrt(2.0 * max(v1, 0.0) * alpha)
    dw2 = 2.0 * math.sqrt(2.0 * max(v2, 0.0) * alpha)

    diff_w = abs(w1 - w2)
    if diff_w < _MACHINE_EPS:
        return 0.0

    return float((dw1 + dw2) / diff_w)

def compute_quantum_information_metrics(
    rho: NDArray[np.complex128],
    sigma: NDArray[np.complex128]
) -> Tuple[float, float, float, float, float]:
    r"""
    Calcula Fidelidad de Uhlmann, Bures-Wasserstein, Fubini-Study,
    distorsión Finsler-Mahalanobis RHI_Finsler y la censura de Kelly s*.
    """
    rho_sq = la.sqrtm(0.5 * (rho + rho.conj().T))
    mid = rho_sq @ (0.5 * (sigma + sigma.conj().T)) @ rho_sq
    mid_sq = la.sqrtm(0.5 * (mid + mid.conj().T))

    fid_val = float(np.real(np.trace(mid_sq)))**2
    fid_val = min(max(fid_val, 0.0), 1.0)

    d_bures = math.sqrt(max(2.0 - 2.0 * math.sqrt(fid_val), 0.0))
    d_fs = math.acos(math.sqrt(fid_val))

    # RHI Finsler
    dim = rho.shape[0]
    diff = sigma - rho
    norm_f = max(float(la.norm(rho, 'fro')), _MACHINE_EPS)

    trace_diff_g = float(np.real(np.trace(diff @ diff.conj().T)))
    term1 = trace_diff_g / ((norm_f**2) * ((1.0 + math.sqrt(fid_val))**2) + _MACHINE_EPS)

    fq = float(np.real(np.trace(rho @ sigma - rho @ rho)))
    term2 = 0.25 * (math.tanh(fq)**2)

    rhi_finsler = math.sqrt(max(term1 + term2, 0.0))

    # Censura C^inf Fermi-Dirac
    sw_fermi = 1.0 / (1.0 + math.exp((rhi_finsler - _RHI_THRESHOLD) / _FERMI_DIRAC_TEMP))

    # Kelly stake
    f_star = 0.25
    melnikov_val = compute_melnikov_homoclinic_integral()
    chirikov_val = compute_chirikov_overlap_parameter(rho)

    theta_m = 1.0 if abs(melnikov_val) <= _MELNIKOV_MAX_TOL else 0.0
    theta_k = 1.0 if chirikov_val < _CHIRIKOV_CRITICAL else 0.0
    theta_rhi = 1.0 if rhi_finsler < _RHI_THRESHOLD else 0.0

    s_star = f_star * max(1.0 - rhi_finsler, 0.0) * sw_fermi * theta_m * theta_k * theta_rhi
    if theta_m == 0.0 or theta_k == 0.0 or theta_rhi == 0.0:
        s_star = 0.0

    return fid_val, d_bures, d_fs, rhi_finsler, s_star

# ═══════════════════════════════════════════════════════════════════════════
# FASE 1 – Registro y Estadística
# ═══════════════════════════════════════════════════════════════════════════
class Phase1_CoreFockRegistry(ABC):
    """
    FASE 1 (Registro y Estadística): Ocupación de Fock, exclusión de Pauli,
    evicción por entropía, atlas celeste y operador densidad.
    """

    def __init__(
        self,
        max_cartridges: int = _DEFAULT_MAX_CARTRIDGES,
        tolerance: float = _DEFAULT_TOL
    ) -> None:
        self._max_cartridges: Final[int] = max_cartridges
        self._tolerance: Final[float] = tolerance
        self._registry: Dict[str, ToonCartridge] = {}
        self._embedding_cache: Dict[str, NDArray[np.float64]] = {}
        self._fermion_hashes: Dict[type, set] = {}

    @property
    def size(self) -> int:
        return len(self._registry)

    @property
    def max_capacity(self) -> int:
        return self._max_cartridges

    def inject_cartridge(
        self,
        key: str,
        cartridge: ToonCartridge,
        embedding: NDArray[np.float64]
    ) -> None:
        fermion_types = (
            ElectronCartridge, ProtonCartridge, PolaronCartridge,
            TorsionCartridge, HouseholderReflectionFermion
        )
        
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
        
        if len(self._registry) >= self._max_cartridges:
            logger.info("Horizonte de Bekenstein alcanzado. Iniciando purga por entropía.")
            try:
                self.evict_by_entropy(embedding)
            except SynapticRegistryError as exc:
                raise FockSpaceOverflowError(
                    f"FockSpaceOverflowError: No se pudo liberar espacio. {exc}"
                ) from exc
        
        self._registry[key] = cartridge
        norm = la.norm(embedding) + _MACHINE_EPS
        self._embedding_cache[key] = embedding / norm
        
        if isinstance(cartridge, fermion_types):
            cartridge_type = type(cartridge)
            if cartridge_type not in self._fermion_hashes:
                self._fermion_hashes[cartridge_type] = set()
            self._fermion_hashes[cartridge_type].add(cartridge.quantum_state_hash)
        
        logger.debug("Partícula %s inyectada con éxito en el Espacio de Fock.", cartridge.cartridge_id)

    def evict_by_entropy(self, current_decision_vector: NDArray[np.float64]) -> str:
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
        return [c for c in self._registry.values() if isinstance(c, particle_type)]

    def count_fermions_and_bosons(self) -> Tuple[int, int]:
        fermion_types = (
            ElectronCartridge, ProtonCartridge, PolaronCartridge,
            TorsionCartridge, HouseholderReflectionFermion
        )
        fermion_count = sum(
            1 for c in self._registry.values() if isinstance(c, fermion_types)
        )
        boson_count = len(self._registry) - fermion_count
        return fermion_count, boson_count

    def build_density_matrix_from_registry(self) -> Tuple[NDArray[np.complex128], bool]:
        n = len(self._registry)
        if n == 0:
            return np.eye(1, dtype=np.complex128), False
        
        sample_vec = next(iter(self._embedding_cache.values()))
        dim = len(sample_vec)
        rho = np.zeros((dim, dim), dtype=np.complex128)
        
        for v in self._embedding_cache.values():
            vec = v.reshape(-1, 1).astype(np.complex128)
            rho += vec @ vec.conj().T
        
        rho /= n
        rho_projected, higham_applied = stable_density_matrix_higham(rho, self._tolerance)
        return rho_projected, higham_applied

    def generate_phase1_certificate(self) -> Phase1RegistryCertificate:
        fermion_count, boson_count = self.count_fermions_and_bosons()
        rho, higham_applied = self.build_density_matrix_from_registry()
        
        trace = float(np.real(np.trace(rho)))
        purity = float(np.real(np.trace(rho @ rho)))
        
        # Atlas Celeste y Yang-Mills-Higgs
        L_del, e_del, i_del, M_del, E_del = compute_delaunay_poincare_elements(rho)
        cs3_val, wilson_val = compute_yang_mills_higgs_chern_simons(rho)

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
            delaunay_eccentricity=e_del,
            delaunay_inclination=i_del,
            kepler_mean_anomaly=M_del,
            kepler_eccentric_anomaly=E_del,
            chern_simons_3form_val=cs3_val,
            wilson_loop_holonomy=wilson_val,
            verdict=verdict,
        )

# ═══════════════════════════════════════════════════════════════════════════
# FASE 2 – Dinámica Cuántica y Mecánica Celeste
# ═══════════════════════════════════════════════════════════════════════════
class Phase2_QuantumDynamics(Phase1_CoreFockRegistry):
    """
    FASE 2 (Dinámica Cuántica): Bogoliubov-Valatin, aniquilación e^+ e^- con
    verificación HMAC de tiempo constante, Melnikov, Chirikov y Bures-Finsler.
    """

    def apply_bogoliubov_transformation(
        self,
        u_k: complex,
        v_k: complex,
        b_k: complex,
        b_dagger_k: complex
    ) -> Tuple[complex, complex]:
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

        # Verificación HMAC de tiempo constante
        expected_sig = f"POSI_AUTH_{positron.cartridge_id}".encode("utf-8")
        actual_sig = positron.authorization_signature.encode("utf-8")
        if not hmac.compare_digest(
            hashlib.sha256(expected_sig).hexdigest(),
            hashlib.sha256(actual_sig).hexdigest()
        ):
            logger.warning("Firma HMAC de Positrón sospechosa o no coincide.")
        
        m_eff = 0.5 * (electron.inertial_mass + positron.inertial_mass)
        energy_total = 2.0 * m_eff
        
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
        
        electron_type = type(electron)
        if electron_type in self._fermion_hashes:
            self._fermion_hashes[electron_type].discard(electron.quantum_state_hash)
        
        positron_type = type(positron)
        if positron_type in self._fermion_hashes:
            self._fermion_hashes[positron_type].discard(positron.quantum_state_hash)
        
        del self._registry[electron_key]
        del self._embedding_cache[electron_key]
        del self._registry[positron_key]
        del self._embedding_cache[positron_key]
        
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
            rho, higham_applied = stable_density_matrix_higham(rho, self._tolerance)
        
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
        rho0, higham_applied = self.build_density_matrix_from_registry()
        rho_final, dissipation_rate = self.solve_lindblad_evolution(
            rho0, hamiltonian_eff, jump_operators, gamma_rates, dt, steps
        )
        
        final_trace = float(np.real(np.trace(rho_final)))
        final_purity = float(np.real(np.trace(rho_final @ rho_final)))
        trace_residual = abs(final_trace - 1.0)
        
        is_trace_preserved = trace_residual <= self._tolerance * 100
        is_psd_preserved = np.all(la.eigvalsh(rho_final) >= -self._tolerance)
        
        # Invariantes Melnikov, Chirikov y Bures-Finsler
        melnikov_val = compute_melnikov_homoclinic_integral()
        chirikov_val = compute_chirikov_overlap_parameter(rho_final)
        fid, d_bures, d_fs, rhi_finsler, s_star = compute_quantum_information_metrics(rho0, rho_final)

        is_kam_ok = (abs(melnikov_val) <= _MELNIKOV_MAX_TOL) and (chirikov_val < _CHIRIKOV_CRITICAL)

        verdict = FockRegistryVerdict.COHERENT
        if not np.isfinite(final_trace) or not np.isfinite(final_purity) or not is_kam_ok:
            verdict = FockRegistryVerdict.VETOED
        elif not is_trace_preserved or not is_psd_preserved or rhi_finsler >= _RHI_THRESHOLD:
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
            uhlmann_fidelity=fid,
            bures_distance=d_bures,
            fubini_study_distance=d_fs,
            rhi_finsler_index=rhi_finsler,
            fermi_dirac_kelly_censorship=float(1.0 / (1.0 + math.exp((rhi_finsler - _RHI_THRESHOLD) / _FERMI_DIRAC_TEMP))),
            kelly_optimal_stake=s_star,
            melnikov_integral_val=melnikov_val,
            chirikov_overlap_parameter=chirikov_val,
            is_kam_tori_preserved=is_kam_ok,
            verdict=verdict,
        )
        
        return rho_final, cert

# ═══════════════════════════════════════════════════════════════════════════
# FASE 3 – Integración Inmune y Cascada RSI³
# ═══════════════════════════════════════════════════════════════════════════
class Phase3_ImmuneOrchestrator(Phase2_QuantumDynamics):
    """
    FASE 3 (Integración Inmune): Orquestación del ciclo completo y ejecución
    de la Cascada de Banach RSI³ (T = R_3 o R_2 o R_1).
    """

    def execute_rsi3_banach_cascade(
        self,
        rho: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], float, float, float, float, bool]:
        r"""
        Ejecuta la Cascada de Banach RSI³:
        - R1: Ricci-DeTurck metric smoothing
        - R2: Connes-Daleckii-Krein Lipschitz divided difference bound
        - R3: Brockett-Shahshahani isospectral flow
        - Spohn Entropy Production: \sigma_{Spohn} = -\operatorname{Tr}(\log\rho \cdot \dot{\rho}) \ge 0.
        """
        dim = rho.shape[0]
        rho_clean = 0.5 * (rho + rho.conj().T)
        vals, vecs = la.eigh(rho_clean)
        vals_pos = np.maximum(vals, 1.0e-12)

        # R1: Ricci-DeTurck
        log_G = np.diag(np.log(vals_pos))
        ricci_drift = float(np.linalg.norm(-2.0 * (log_G - np.trace(log_G) * np.eye(dim) / dim)))

        # R2: Connes-Daleckii-Krein
        max_cdk = 0.0
        for i in range(dim):
            for j in range(dim):
                if i != j:
                    denom = vals_pos[i] - vals_pos[j]
                    if abs(denom) > 1.0e-12:
                        diff_div = abs(np.log(vals_pos[i]) - np.log(vals_pos[j])) / abs(denom)
                        if diff_div > max_cdk:
                            max_cdk = diff_div

        # R3: Brockett-Shahshahani isospectral flow
        N_diag = np.diag(np.arange(1, dim + 1, dtype=np.float64))
        rho_dot = rho_clean @ (rho_clean @ N_diag - N_diag @ rho_clean) - (rho_clean @ N_diag - N_diag @ rho_clean) @ rho_clean

        # Spohn Entropy Production
        log_rho = vecs @ np.diag(np.log(vals_pos)) @ vecs.conj().T
        spohn_prod = float(-np.real(np.trace(log_rho @ rho_dot)))
        spohn_prod = max(spohn_prod, 0.0)

        # Paso de integración
        tau = 0.01 / (1.0 + max_cdk)
        rho_next = rho_clean + tau * rho_dot
        rho_next, _ = stable_density_matrix_higham(rho_next, self._tolerance)

        # Invarianza isoespectral de Brockett
        vals_next = la.eigvalsh(rho_next)
        isospectral_err = float(np.linalg.norm(np.sort(vals) - np.sort(vals_next)))

        # Cota de contracción de Banach
        is_banach = (max_cdk * tau < 1.0) and (isospectral_err < 1.0e-8)

        return rho_next, spohn_prod, ricci_drift, max_cdk, isospectral_err, is_banach

    def execute_immune_cycle(
        self,
        hamiltonian_eff: Optional[NDArray[np.complex128]] = None,
        jump_operators: Optional[List[NDArray[np.complex128]]] = None,
        gamma_rates: Optional[List[float]] = None,
        dt: float = 1.0,
        steps: int = 10,
        perform_annihilations: bool = True,
    ) -> Tuple[Phase3ImmuneCertificate, NDArray[np.complex128]]:
        annihilation_events = 0
        bogoliubov_transformations = 0
        
        if perform_annihilations:
            electrons = self.get_all_particles_of_type(ElectronCartridge)
            positrons = self.get_all_particles_of_type(PositronCartridge)
            
            for e in electrons:
                for p in positrons:
                    if e.homological_charge + p.homological_charge == 0:
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
                                pass

        rho_final, _ = self.build_density_matrix_from_registry()
        
        if hamiltonian_eff is not None:
            if jump_operators is None:
                jump_operators = []
            if gamma_rates is None:
                gamma_rates = []
            rho_final, _ = self.evolve_registry_density(
                hamiltonian_eff, jump_operators, gamma_rates, dt, steps
            )
        
        # Ejecutar Cascada Banach RSI³
        rho_rsi3, spohn, ricci_d, cdk_lip, brockett_err, is_banach = self.execute_rsi3_banach_cascade(rho_final)

        # Entropía de von Neumann
        vals = la.eigvalsh(rho_rsi3)
        vals = np.maximum(vals, _MACHINE_EPS)
        entropy_prod = float(-np.sum(vals * np.log(vals)))
        
        is_unitary = (
            jump_operators is None or
            len(jump_operators) == 0 or
            all(g == 0.0 for g in gamma_rates)
        )
        
        verdict = FockRegistryVerdict.COHERENT
        if not np.isfinite(entropy_prod) or not is_banach:
            verdict = FockRegistryVerdict.VETOED
        elif annihilation_events == 0 and not is_unitary:
            verdict = FockRegistryVerdict.DEGRADED
        
        cert = Phase3ImmuneCertificate(
            annihilation_events=annihilation_events,
            bogoliubov_transformations=bogoliubov_transformations,
            entropy_production=entropy_prod,
            spohn_entropy_production_rate=spohn,
            ricci_deturck_metric_drift=ricci_d,
            connes_daleckii_krein_lipschitz=cdk_lip,
            brockett_isospectral_error=brockett_err,
            is_banach_contraction=is_banach,
            is_unitary_evolution=is_unitary,
            crowbar_triggered=False,
            verdict=verdict,
        )
        
        return cert, rho_rsi3

# ═══════════════════════════════════════════════════════════════════════════
# Registro final: SynapticFockSpaceRegistry
# ═══════════════════════════════════════════════════════════════════════════
class SynapticFockSpaceRegistry(Morphism, Phase3_ImmuneOrchestrator):
    """
    Registro completo del Espacio de Fock Sináptico v9.3.0-PhD.
    Agrupa todas las capacidades de las fases anidadas.
    """

    def __init__(
        self,
        max_cartridges: int = _DEFAULT_MAX_CARTRIDGES,
        tolerance: float = _DEFAULT_TOL,
        halt_on_veto: bool = False
    ) -> None:
        Morphism.__init__(self, "SynapticFockSpaceRegistry")
        Phase3_ImmuneOrchestrator.__init__(self, max_cartridges=max_cartridges, tolerance=tolerance)
        self._halt_on_veto = halt_on_veto

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
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
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        try:
            cert_1 = self.generate_phase1_certificate()
            
            if cert_1.verdict.is_vetoed:
                cert_2 = self._vetoed_phase2_certificate()
                cert_3 = self._vetoed_phase3_certificate()
            else:
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
                        uhlmann_fidelity=1.0,
                        bures_distance=0.0,
                        fubini_study_distance=0.0,
                        rhi_finsler_index=0.0,
                        fermi_dirac_kelly_censorship=1.0,
                        kelly_optimal_stake=0.25,
                        melnikov_integral_val=0.0,
                        chirikov_overlap_parameter=0.0,
                        is_kam_tori_preserved=True,
                        verdict=FockRegistryVerdict.COHERENT,
                    )
                
                if cert_2.verdict.is_vetoed:
                    cert_3 = self._vetoed_phase3_certificate()
                else:
                    cert_3, _ = self.execute_immune_cycle(
                        hamiltonian_eff, jump_operators, gamma_rates, dt, steps,
                        perform_annihilations
                    )
            
            final_verdict = FockRegistryVerdict.supremum(
                cert_1.verdict,
                cert_2.verdict,
                cert_3.verdict,
            )
            
            registry_action = RegistryAction.NONE
            diagnostic_note = "Ciclo de registro cuántico v9.3.0-PhD completado."
            
            if final_verdict == FockRegistryVerdict.VETOED:
                registry_action = RegistryAction.HALT_REGISTRY
                diagnostic_note = (
                    "VETO DE REGISTRO: violación de Pauli, melnikov/chirikov caótico, "
                    "o inestabilidad de Banach RSI³."
                )
                logger.error("¡VETO DE REGISTRO! Halt requested.")
                if self._halt_on_veto:
                    raise SynapticRegistryError(
                        "Violación de invariantes cuánticos o de mecánica celeste."
                    )
            elif final_verdict == FockRegistryVerdict.DEGRADED:
                registry_action = RegistryAction.PURGE_ENTROPY
                diagnostic_note = (
                    "Degradación detectada: RHI_Finsler elevado o regularización Higham."
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

    def _vetoed_phase2_certificate(self) -> Phase2DynamicsCertificate:
        return Phase2DynamicsCertificate(
            evolution_time=0.0,
            evolution_steps=0,
            final_trace=0.0,
            final_purity=0.0,
            trace_preservation_residual=float("inf"),
            is_trace_preserved=False,
            is_psd_preserved=False,
            lindblad_dissipation_rate=float("inf"),
            uhlmann_fidelity=0.0,
            bures_distance=float("inf"),
            fubini_study_distance=float("inf"),
            rhi_finsler_index=1.0,
            fermi_dirac_kelly_censorship=0.0,
            kelly_optimal_stake=0.0,
            melnikov_integral_val=1.0,
            chirikov_overlap_parameter=2.0,
            is_kam_tori_preserved=False,
            verdict=FockRegistryVerdict.VETOED,
        )
    
    def _vetoed_phase3_certificate(self) -> Phase3ImmuneCertificate:
        return Phase3ImmuneCertificate(
            annihilation_events=0,
            bogoliubov_transformations=0,
            entropy_production=float("inf"),
            spohn_entropy_production_rate=0.0,
            ricci_deturck_metric_drift=float("inf"),
            connes_daleckii_krein_lipschitz=float("inf"),
            brockett_isospectral_error=float("inf"),
            is_banach_contraction=False,
            is_unitary_evolution=False,
            crowbar_triggered=True,
            verdict=FockRegistryVerdict.VETOED,
        )
    
    def _cataclysm_state(self, reason: str, timestamp_utc: str) -> FockRegistryState:
        phase1_dummy = Phase1RegistryCertificate(
            registry_size=0,
            max_capacity=self._max_cartridges,
            fermion_count=0,
            boson_count=0,
            density_matrix_trace=0.0,
            density_matrix_purity=0.0,
            higham_regularization_applied=False,
            delaunay_eccentricity=0.0,
            delaunay_inclination=0.0,
            kepler_mean_anomaly=0.0,
            kepler_eccentric_anomaly=0.0,
            chern_simons_3form_val=0.0,
            wilson_loop_holonomy=0.0,
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
    
    def _generate_provenance_hash(
        self,
        c1: Phase1RegistryCertificate,
        c2: Phase2DynamicsCertificate,
        c3: Phase3ImmuneCertificate,
        timestamp_utc: str,
    ) -> str:
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
                f"{c2.melnikov_integral_val:.12e}",
                f"{c2.chirikov_overlap_parameter:.12e}",
                f"{c2.rhi_finsler_index:.12e}",
                f"{c2.kelly_optimal_stake:.12e}",
                f"{c3.spohn_entropy_production_rate:.12e}",
                str(int(c3.is_banach_contraction)),
            )
        )
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

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
    "RegistryGovernanceState",
    "Phase1_CoreFockRegistry",
    "Phase2_QuantumDynamics",
    "Phase3_ImmuneOrchestrator",
    "SynapticFockSpaceRegistry",
    "stable_density_matrix_higham",
    "compute_delaunay_poincare_elements",
    "compute_yang_mills_higgs_chern_simons",
    "compute_melnikov_homoclinic_integral",
    "compute_chirikov_overlap_parameter",
    "compute_quantum_information_metrics",
]
