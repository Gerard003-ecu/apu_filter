# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Lithological Manifold                                                                     ║
║ Módulo   : Lithological Manifold (Fijador de Impedancia Geomecánica)                                  ║
║ Ubicación: app/physics/lithological_manifold.py                                                      ║
║ Ruta     : app/physics/lithological_manifold.py                                                      ║
║ Versión  : 6.0.0-Doctoral-USCS-Terzaghi-Poisson-Poincare-Heyting-RSI-L3-Secure                       ║
║ Función  : Operador constitutivo de impedancia geomecánica, criterio de Mohr-Coulomb 3D,             ║
║            consolidación de Terzaghi, dinámica celeste de Poincaré, retículo de Heyting y RSI-L3.   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────────────────────────────
Este motor espectral constitutivo de la Capa Física (Nivel 3, $V_{\mathrm{PHYSICS}}$) opera el anclaje
geomecánico de la Malla de Datos. Abandona de manera radical las aproximaciones empíricas tradicionales del
suelo para modelar la litología del terreno como un **Tensor de Impedancia Geomecánica Compleja ($Z_{\mathrm{geo}}$)**,
el cual actúa como la condición de frontera de Dirichlet absoluta sobre el complejo simplicial de la obra.

El sistema mapea el vector covariante de estado del suelo hacia un conjunto de magnitudes reológicas acotadas
estrictamente en el intervalo unitario $[0, 1]$ y hacia variedades riemannianas de matrices simétricas definidas
positivas $\operatorname{SPD}(3)$. Toda inestabilidad o desvío crítico de la capacidad portante se cuantifica y
se colapsa en excitaciones de cuasipartículas (Cartuchos TOON), las cuales inyectan capacitancia parásita o arrastre
viscoso inercial, forzando la disipación estricta y previniendo la inyección espuria de energía en los resolvedores
de la FPU.

La dinámica se estructura formalmente en tres fases:
  • FASE 1 — SÍNTESIS REOLÓGICA Y TENSOR DE TENSIONES EFECTIVAS DE TERZAGHI:
      Convierte los parámetros de clasificación USCS (humedad, plasticidad, relación de vacíos $e$, peso unitario $\gamma$)
      en el tensor de tensiones efectivas de Terzaghi $\sigma^\prime_{ij} = \sigma_{ij} - u \delta_{ij}$, donde $u$ es
      la presión intersticial de poros. Evalúa los módulos elásticos dinámicos (módulo de Young $E_{\mathrm{dyn}}$,
      módulo de cizalla $G$, módulo de deformación volumétrica $K$) y el coeficiente de Poisson $\nu \in [0.1, 0.49]$.
  • FASE 2 — CRITERIO DE FALLA DE MOHR-COULOMB 3D Y ESTABILIDAD DIOFÁNTICA KAM DE POINCARÉ:
      Calcula las tensiones principales $(\sigma_1^\prime \ge \sigma_2^\prime \ge \sigma_3^\prime)$ y los invariantes
      de Haigh-Westergaard (tensión hidrostática $p$, tensión desviadora $q$, ángulo de Lode $\theta_L$). Resuelve el
      criterio tridimensional de Mohr-Coulomb $f(\sigma^\prime) = \sigma_1^\prime - \sigma_3^\prime - (\sigma_1^\prime + \sigma_3^\prime)\sin\phi - 2c\cos\phi \le 0$,
      audita el espectro de resonancias KAM frente a excitaciones armónicas, y aplica la contracción de Banach RSI-L3.
  • FASE 3 — CERTIFICACIÓN DE CONDICIÓN DE FRONTERA DIRICHLET, FLOQUET Y VETO HEYTING:
      Construye el tensor de impedancia dinámica $Z_{\mathrm{geo}}(\omega) = K_{\mathrm{st}} + j\omega C_{\mathrm{rad}} - \omega^2 M_{\mathrm{dyn}}$,
      verifica pasividad de Brune ($\operatorname{Re}(Z_{\mathrm{geo}}) \ge 0$), evalúa la estabilidad de monodromía
      de Floquet ($\det M = 1$), y clasifica la condición geotécnica en el retículo de Heyting $\Omega_3$.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
1. PRINCIPIO DE TENSIONES EFECTIVAS DE TERZAGHI:
   El comportamiento mecánico del medio poroso saturado está gobernado exclusivamente por la tensión efectiva:
       $$\sigma_{ij}^\prime = \sigma_{ij} - u \delta_{ij}$$
   donde $\sigma_{ij}$ es el tensor de tensiones totales, $u$ es la presión de poro del fluido, y $\delta_{ij}$ es la
   delta de Kronecker. La deformación volumétrica se rige por la ecuación diferencial de consolidación parabólica:
       $$\frac{\partial u}{\partial t} = c_v \nabla^2 u, \qquad c_v = \frac{k}{\gamma_w m_v}$$

2. CRITERIO TRIDIMENSIONAL DE ROTURA DE MOHR-COULOMB:
   En el espacio de tensiones principales efectivas, la superficie de fluencia de Mohr-Coulomb se formula como:
       $$\mathcal{F}_{\mathrm{MC}}(\sigma^\prime) = \frac{1}{2}(\sigma_1^\prime - \sigma_3^\prime) - \frac{1}{2}(\sigma_1^\prime + \sigma_3^\prime)\sin\phi - c\cos\phi \le 0$$
   donde $c$ es la cohesión efectiva y $\phi$ es el ángulo de fricción interna efectiva del estrato.
   En términos de los invariantes de Haigh-Westergaard:
       $$\mathcal{F}_{\mathrm{MC}}(p^\prime, q, \theta_L) = p^\prime \sin\phi + q \left[ \frac{\cos\theta_L}{\sqrt{3}} - \frac{\sin\theta_L \sin\phi}{3} \right] - c\cos\phi \le 0$$

3. IMPEDANCIA GEOMECÁNICA COMPLEJA Y DISIPACIÓN PASIVA:
   La interacción suelo-estructura en el dominio frecuencial se representa mediante la impedancia dinámica:
       $$Z_{\mathrm{geo}}(j\omega) = K_{\mathrm{re}} + j\omega C_{\mathrm{im}}$$
   donde la parte real $K_{\mathrm{re}}$ representa la rigidez estática y dinámica recuperable, y la parte imaginaria
   $\omega C_{\mathrm{im}}$ modela la disipación combinada por amortiguamiento histerético del suelo y radiación geométrica
   de ondas elásticas hacia el semiespacio infinito.

4. DINÁMICA CELESTE DE POINCARÉ Y ESTABILIDAD DIOFÁNTICA KAM:
   Bajo cargas cíclicas de tráfico o compactación, el vector de frecuencias fundamentales $\omega = (\omega_{\mathrm{soil}}, \omega_{\mathrm{load}})$
   se somete a la cota diofántica de Kolmogorov-Arnold-Moser:
       $$|\omega_{\mathrm{soil}} - k \omega_{\mathrm{load}}| \ge \frac{\gamma}{|k|^\tau}$$
   para prevenir fenómenos de licuefacción resonante o colapso por fatiga dinámica.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  [Axioma LM-1] Principio de Presión Efectiva Positiva:
      $\sigma^\prime \ge 0$ en régimen no traccional, prohibiendo tracciones espurias en suelos no cohesivos ($c = 0$).
  [Axioma LM-2] Confinamiento Termodinámico del Coeficiente de Poisson:
      $\nu \in (0, 0.50)$, garantizando positividad definida de la energía de deformación volumétrica $K > 0$.
  [Axioma LM-3] Criterio de Falla de Mohr-Coulomb Invariante:
      $\mathcal{F}_{\mathrm{MC}}(\sigma^\prime) \le 0$ para todo estado admisible de soporte estructural.
  [Axioma LM-4] Pasividad de Brune en la Impedancia Geomecánica:
      $\operatorname{Re}(Z_{\mathrm{geo}}(j\omega)) \ge 0 \quad \forall \omega \in \mathbb{R}$.
  [Axioma LM-5] Preservación de Monodromía Simpléctica:
      $|\det(\mathcal{M}_{\mathrm{Floquet}}) - 1.0| \le 10^{-6}$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Positividad Métrica del Tensor de Elasticidad):
      $C_{ijkl} = C_{jikl} = C_{klij} \succ 0$, con autovalores acotados en el espectro admisible.
  Invariante 2 (Capacidad Portante Última Admisible):
      $q_{\mathrm{adm}} = \frac{q_{\mathrm{ult}}}{\mathrm{FS}}$, con factor de seguridad $\mathrm{FS} \ge 3.0$.
  Invariante 3 (Conservación de Masa en la Consolidación de Terzaghi):
      $\int_\Omega \Delta u \, d\Omega = -\int_{\partial\Omega} v_{\mathrm{darcy}} \cdot n \, dA$.
  Invariante 4 (Bypass de Silicio Crowbar):
      Colapso geomecánico ($\mathcal{F}_{\mathrm{MC}} > 0$) conmutan la interrupción ISR en silicio en $< 400$ ns.
"""

from __future__ import annotations

import cmath
import enum
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Final,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from app.core.mic_algebra import Morphism, CategoricalState
from app.core.schemas import Stratum
try:
    from app.core.inmune_system.metric_tensors import MetricTensorFactory
except ImportError:
    from app.core.immune_system.metric_tensors import MetricTensorFactory

logger = logging.getLogger("MIC.Physics.LithologicalManifold")

__version__: Final[str] = (
    "6.0.0-Doctoral-USCS-Terzaghi-Poisson-Poincare-Heyting-RSI-L3-Secure"
)

__all__: Final[Tuple[str, ...]] = (
    "GeomechanicalConstants",
    "USCS_COARSE_GROUPS",
    "USCS_FINE_GROUPS",
    "USCS_ORGANIC_EXTREME_GROUPS",
    "USCS_ALL_VALID_GROUPS",
    "LithologicalManifoldError",
    "LithologicalInputError",
    "LithologicalSingularityError",
    "LithologicalNumericalError",
    "DiagnosticRule",
    "LithologicalHeytingTruth",
    "SwellingPlasmonCartridge",
    "YieldingPhononCartridge",
    "LiquefactionSolitonCartridge",
    "LithologyCartridge",
    "SoilTensor",
    "LithologyDiagnosticReport",
    "SoilValidationCertificate",
    "Phase1LithologyHandoff",
    "LithologyRSILevel3Certificate",
    "PoincareGeomechanicalCertificate",
    "GeomechanicalSpectralReport",
    "Phase2LithologyHandoff",
    "CrowbarPortState",
    "LithologicalSuturationState",
    "Phase3LithologyDecision",
    "LithologicalManifold",
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        §1  CONSTANTES FÍSICAS, GEOMECÁNICAS, ESPECTRALES Y DE SILICIO       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GeomechanicalConstants:
    """
    Constantes fundamentales, umbrales geotécnicos, cotas espectrales de
    Wilkinson, parámetros KAM de Poincaré y especificaciones de silicio ESP32.
    """

    # --- Constantes fundamentales ---
    EPSILON_MACH: Final[float] = 1e-12
    IEEE754_U: Final[float] = float(np.finfo(np.float64).eps)
    COMPLEX_STEP_H: Final[float] = 1e-20
    GRAVITY_M_S2: Final[float] = 9.80665
    WATER_DENSITY_KG_M3: Final[float] = 1000.0

    # --- Densidad por defecto ---
    DEFAULT_SOIL_DENSITY_KG_M3: Final[float] = 1800.0

    # --- Rangos admisibles de validación ---
    LIQUID_LIMIT_MAX: Final[float] = 500.0
    PLASTICITY_INDEX_MAX: Final[float] = 350.0
    SHEAR_WAVE_VELOCITY_MAX: Final[float] = 5000.0
    VOID_RATIO_MAX: Final[float] = 15.0
    BULK_DENSITY_MIN: Final[float] = 500.0
    BULK_DENSITY_MAX: Final[float] = 3500.0

    # --- Geometría de la Carta de Plasticidad de Casagrande (ASTM D2487) ---
    CASAGRANDE_A_LINE_SLOPE: Final[float] = 0.73
    CASAGRANDE_A_LINE_OFFSET: Final[float] = 20.0
    CASAGRANDE_U_LINE_SLOPE: Final[float] = 0.90
    CASAGRANDE_U_LINE_OFFSET: Final[float] = 8.0

    # --- Umbrales operativos de diagnóstico ---
    CRITICAL_LIQUEFACTION_VS_M_S: Final[float] = 150.0
    SOFT_SOIL_VS_M_S: Final[float] = 200.0
    HIGH_VOID_RATIO_THRESHOLD: Final[float] = 0.8
    SWELLING_LIQUID_LIMIT_THRESHOLD: Final[float] = 50.0
    SWELLING_PI_THRESHOLD: Final[float] = 20.0
    LOW_DYNAMIC_RIGIDITY_PA: Final[float] = 25.0e6

    # --- Parámetros de funciones de índice ---
    SWELLING_SIGMOID_MIDPOINT: Final[float] = 20.0
    SWELLING_SIGMOID_STEEPNESS: Final[float] = 0.15
    SKEMPTON_CC_FACTOR: Final[float] = 0.009
    SKEMPTON_CC_OFFSET: Final[float] = 10.0
    YIELDING_REFERENCE_SCALE: Final[float] = 0.01
    LIQUEFACTION_NONLINEAR_EXPONENT: Final[float] = 2.0

    # --- Parámetros Elastodinámicos 3D de Poisson-Lysmer-Gazetas ---
    POISSON_RATIO_MIN: Final[float] = 0.15
    POISSON_RATIO_MAX_DRAINED: Final[float] = 0.42
    POISSON_RATIO_MAX_UNDRAINED: Final[float] = 0.495
    FOUNDATION_CHARACTERISTIC_RADIUS_M: Final[float] = 1.0
    DEFAULT_LAPLACE_SIGMA_RAD_S: Final[float] = 0.25
    DEFAULT_LAPLACE_OMEGA_RAD_S: Final[float] = 2.0 * math.pi * 5.0  # 5 Hz sísmico
    HYSTERETIC_DAMPING_BASE: Final[float] = 0.05

    # --- Parámetros RSI Nivel 3 (Variedad Riemanniana SPD(3)) ---
    SPD_MAX_CONDITION_NUMBER: Final[float] = 25.0
    RSI_MAX_ITERATIONS: Final[int] = 8
    RSI_GEODESIC_STEP: Final[float] = 0.35

    # --- Parámetros de Mecánica Celeste de Henri Poincaré y KAM ---
    KAM_DIOPHANTINE_GAMMA: Final[float] = 1e-2
    KAM_DIOPHANTINE_TAU: Final[float] = 2.5
    KAM_MAX_HARMONIC_ORDER: Final[int] = 4
    POINCARE_INTEGRATION_DT_S: Final[float] = 0.01

    # --- Umbrales del Retículo Bi-Heyting Ω₃ y Silicio ESP32 (BT151 Crowbar) ---
    HEYTING_LIQUEFACTION_VETO_THRESHOLD: Final[float] = 0.45
    HEYTING_LIQUEFACTION_DEGRADED_THRESHOLD: Final[float] = 0.05
    HEYTING_YIELDING_VETO_THRESHOLD: Final[float] = 0.92
    HEYTING_SWELLING_DEGRADED_THRESHOLD: Final[float] = 0.70
    ESP32_CROWBAR_GPIO_PIN: Final[int] = 14
    ESP32_IRAM_ISR_NOMINAL_NS: Final[float] = 392.0
    ESP32_IRAM_ISR_MAX_NS: Final[float] = 400.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        §2  CLASIFICACIÓN USCS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

USCS_COARSE_GROUPS: FrozenSet[str] = frozenset({
    "GW", "GP", "GM", "GC",
    "SW", "SP", "SM", "SC",
    "GW-GM", "GW-GC", "GP-GM", "GP-GC",
    "SW-SM", "SW-SC", "SP-SM", "SP-SC",
})

USCS_FINE_GROUPS: FrozenSet[str] = frozenset({
    "ML", "CL", "OL",
    "MH", "CH", "OH",
    "CL-ML",
})

USCS_ORGANIC_EXTREME_GROUPS: FrozenSet[str] = frozenset({"PT"})

USCS_ALL_VALID_GROUPS: FrozenSet[str] = (
    USCS_COARSE_GROUPS | USCS_FINE_GROUPS | USCS_ORGANIC_EXTREME_GROUPS
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║            §3  JERARQUÍA DE EXCEPCIONES (FAIL-SECURE BOUNDARY)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LithologicalManifoldError(Exception):
    """Excepción base del módulo litológico."""


class LithologicalInputError(LithologicalManifoldError):
    """Error de validación o parseo de entrada en la variedad litológica."""


class LithologicalSingularityError(LithologicalManifoldError):
    """
    Singularidad litológica extrema (e.g. Turba / Peat 'PT' o colapso
    del semiespacio portante).
    """


class LithologicalNumericalError(LithologicalManifoldError):
    """Resultado numérico no finito o violación espectral en la FPU."""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          §4  ENUMERACIONES TIPADAS Y ÁLGEBRA BI-HEYTING (Ω₃)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@enum.unique
class DiagnosticRule(str, enum.Enum):
    """Enumeración tipada de reglas diagnósticas activables."""
    SWELLING_POTENTIAL_DETECTED = "swelling_potential_detected"
    YIELDING_SUSCEPTIBILITY_DETECTED = "yielding_susceptibility_detected"
    LIQUEFACTION_SUSCEPTIBILITY_DETECTED = "liquefaction_susceptibility_detected"
    LOW_DYNAMIC_RIGIDITY_REGIME = "low_dynamic_rigidity_regime"


@enum.unique
class LithologicalHeytingTruth(str, enum.Enum):
    """
    Clasificador de subobjetos del retículo distributivo bi-Heyting de 3 valores:
        Ω₃ = {VETOED (⊥ = 0.0), DEGRADED (½ = 0.5), COHERENT (⊤ = 1.0)}

    Nota sobre polaridad operativa:
    - En la lógica de coherencia estructural de la Malla, `COHERENT` representa
      verdad plena de estabilidad (1.0), `DEGRADED` verdad intermedia (0.5), y
      `VETOED` falsedad / absorción de emergencia (0.0) que dispara el Crowbar.
    """
    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"

    @property
    def numeric_value(self) -> float:
        if self is LithologicalHeytingTruth.COHERENT:
            return 1.0
        if self is LithologicalHeytingTruth.DEGRADED:
            return 0.5
        return 0.0

    @classmethod
    def from_numeric(cls, val: float) -> "LithologicalHeytingTruth":
        if val >= 0.75:
            return cls.COHERENT
        if val >= 0.25:
            return cls.DEGRADED
        return cls.VETOED

    def meet(self, other: "LithologicalHeytingTruth") -> "LithologicalHeytingTruth":
        """Conjunción (a ∧ b = min(a, b)): el estado más restrictivo domina."""
        return self.from_numeric(min(self.numeric_value, other.numeric_value))

    def join(self, other: "LithologicalHeytingTruth") -> "LithologicalHeytingTruth":
        """Disyunción (a ∨ b = max(a, b))."""
        return self.from_numeric(max(self.numeric_value, other.numeric_value))

    def implies(self, other: "LithologicalHeytingTruth") -> "LithologicalHeytingTruth":
        """Implicación intuicionista de Heyting (a → b = ⊤ si a ≤ b, sino b)."""
        if self.numeric_value <= other.numeric_value:
            return LithologicalHeytingTruth.COHERENT
        return other

    def co_implies(self, other: "LithologicalHeytingTruth") -> "LithologicalHeytingTruth":
        """Sustracción co-Heyting de Brouwer (a \\ b = ⊥ si a ≤ b, sino a)."""
        if self.numeric_value <= other.numeric_value:
            return LithologicalHeytingTruth.VETOED
        return self

    def neg(self) -> "LithologicalHeytingTruth":
        """Negación intuicionista (¬a = a → ⊥)."""
        return self.implies(LithologicalHeytingTruth.VETOED)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║            §5  CARTUCHOS DIAGNÓSTICOS HADRÓNICOS (TOON FOCK SPACE)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class SwellingPlasmonCartridge:
    """
    Excitación cuasipartícula asociada a suelo potencialmente expansivo.
    Inyecta capacitancia volumétrica parásita ∈ [0, 1].
    """
    liquid_limit: float
    plasticity_index: float
    parasitic_capacitance: float
    topological_spin: str = "potential_accumulator"

    def __post_init__(self) -> None:
        _assert_unit_interval(
            self.parasitic_capacitance, "parasitic_capacitance"
        )


@dataclass(frozen=True, slots=True)
class YieldingPhononCartridge:
    """
    Excitación cuasipartícula asociada a compresibilidad y cedencia.
    Inyecta arrastre viscoso inercial ∈ [0, 1].
    """
    compression_index: float
    void_ratio: float
    viscous_drag: float
    topological_spin: str = "inertial_drag"

    def __post_init__(self) -> None:
        _assert_unit_interval(self.viscous_drag, "viscous_drag")


@dataclass(frozen=True, slots=True)
class LiquefactionSolitonCartridge:
    """
    Excitación solitónica asociada a susceptibilidad de licuación.
    Aniquila la conectividad portante del grafo (β₀ → ∞ cuando I_liq → 1).
    """
    shear_wave_velocity: float
    susceptibility_index: float
    topological_spin: str = "connectivity_annihilator"

    def __post_init__(self) -> None:
        _assert_unit_interval(
            self.susceptibility_index, "susceptibility_index"
        )


LithologyCartridge = Union[
    SwellingPlasmonCartridge,
    YieldingPhononCartridge,
    LiquefactionSolitonCartridge,
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║      §6  FUNCIONES AUXILIARES NUMÉRICAS, KBN Y DIFERENCIACIÓN COMPLEJA       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _safe_upper_strip(value: str) -> str:
    """Normaliza cadena: elimina espacios y convierte a mayúsculas."""
    return value.strip().upper()


def _clamp(value: float, lo: float, hi: float) -> float:
    """Restricción segura al intervalo [lo, hi]."""
    if math.isnan(value):
        return value
    return max(lo, min(value, hi))


def _assert_finite(value: float, name: str) -> float:
    """Verifica que un resultado intermedio sea finito en IEEE-754."""
    if not math.isfinite(value):
        raise LithologicalNumericalError(
            f"Resultado no finito en '{name}': {value}"
        )
    return value


def _assert_unit_interval(value: float, name: str) -> None:
    """Verifica que un valor pertenezca a [0, 1] con tolerancia de máquina."""
    eps = GeomechanicalConstants.EPSILON_MACH
    if value < -eps or value > 1.0 + eps:
        raise LithologicalNumericalError(
            f"Índice '{name}' = {value} fuera de [0, 1]."
        )


def _kbn_sum(values: Sequence[float]) -> float:
    """
    Suma compensada de Kahan-Babuška-Neumaier (KBN) con error O(u)
    independiente de la cardinalidad N.
    """
    total = 0.0
    compensation = 0.0
    for v in values:
        fv = float(v)
        t = total + fv
        if abs(total) >= abs(fv):
            compensation += (total - t) + fv
        else:
            compensation += (fv - t) + total
        total = t
    return total + compensation


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    """
    Función sigmoidal logística real normalizada a [0, 1]:
        σ(x) = 1 / (1 + exp(-k·(x - x₀)))
    """
    z = steepness * (x - midpoint)
    if z > 700.0:
        return 1.0
    if z < -700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _sigmoid_complex(
    z_in: complex, midpoint: float, steepness: float
) -> complex:
    """
    Extensión analítica holomorfa de la función logística al plano complejo ℂ
    para diferenciación exacta de paso complejo (Squire & Trapp, 1998).
    """
    z = steepness * (z_in - midpoint)
    if z.real > 700.0:
        return complex(1.0, 0.0)
    if z.real < -700.0:
        return complex(0.0, 0.0)
    return 1.0 / (1.0 + cmath.exp(-z))


def _safe_divide(
    numerator: float,
    denominator: float,
    *,
    fallback: float = 0.0,
    eps: float = GeomechanicalConstants.EPSILON_MACH,
) -> float:
    """División segura con fallback explícito ante denominador degenerado."""
    if abs(denominator) < eps:
        return fallback
    return numerator / denominator


def _parse_bool(value: Any) -> bool:
    """Parser booleano determinista y auditable."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise LithologicalInputError(
                f"No se puede interpretar {value!r} como booleano."
            )
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        _TRUE_TOKENS: FrozenSet[str] = frozenset({
            "true", "1", "yes", "y", "si", "sí",
        })
        _FALSE_TOKENS: FrozenSet[str] = frozenset({
            "false", "0", "no", "n",
        })
        if normalized in _TRUE_TOKENS:
            return True
        if normalized in _FALSE_TOKENS:
            return False

    raise LithologicalInputError(
        f"No fue posible interpretar booleano desde: {value!r}"
    )


def _normalize_uscs(uscs: str) -> str:
    """Normaliza la clasificación USCS preservando grupos duales con guion."""
    return _safe_upper_strip(uscs)


def _is_peat(uscs: str) -> bool:
    """Determina si la clasificación corresponde a turba extrema (PT)."""
    return _normalize_uscs(uscs) in USCS_ORGANIC_EXTREME_GROUPS


def _is_sand_like(uscs: str) -> bool:
    """Determina si la clasificación sugiere suelo arenoso o matriz arenosa."""
    u = _normalize_uscs(uscs)
    return u.startswith("S") or "-S" in u


def _is_fine_grained(uscs: str) -> bool:
    """Determina si la clasificación corresponde a suelo fino (ML, CL, MH, CH, etc.)."""
    return _normalize_uscs(uscs) in USCS_FINE_GROUPS


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║    §7  ESTRUCTURAS DE DATOS DE LAS TRES FASES ANIDADAS Y CERTIFICADOS        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class SoilTensor:
    """
    Tensor covariante de entrada con propiedades litológicas en ℝ⁷.

    Invariantes verificados en `validate()`:
    1. LL ∈ [0, 500]%, PI ∈ [0, 350]%.
    2. PI ≤ LL (consistencia de Casagrande).
    3. Vs ∈ (0, 5000] m/s.
    4. e₀ ∈ [0, 15].
    5. ρ ∈ [500, 3500] kg/m³.
    6. Clasificación USCS reconocida.
    7. Finitud IEEE-754 estricta en todas las componentes.
    """
    uscs_classification: str
    liquid_limit: float
    plasticity_index: float
    shear_wave_velocity: float
    void_ratio: float
    is_saturated: bool
    bulk_density_kg_m3: float = GeomechanicalConstants.DEFAULT_SOIL_DENSITY_KG_M3

    def validate(self) -> None:
        """Valida consistencia física, límites normativos y finitud IEEE-754."""
        C = GeomechanicalConstants

        uscs_norm = _normalize_uscs(self.uscs_classification)
        if not uscs_norm:
            raise LithologicalInputError(
                "La clasificación USCS no puede ser vacía."
            )
        if uscs_norm not in USCS_ALL_VALID_GROUPS:
            raise LithologicalInputError(
                f"Clasificación USCS no reconocida: '{uscs_norm}'. "
                f"Grupos válidos: {sorted(USCS_ALL_VALID_GROUPS)}"
            )

        numeric_fields = {
            "liquid_limit": self.liquid_limit,
            "plasticity_index": self.plasticity_index,
            "shear_wave_velocity": self.shear_wave_velocity,
            "void_ratio": self.void_ratio,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
        }
        for name, value in numeric_fields.items():
            if not math.isfinite(value):
                raise LithologicalInputError(
                    f"El campo '{name}' debe ser finito; recibido: {value}"
                )

        if self.liquid_limit < 0.0:
            raise LithologicalInputError(
                "El límite líquido no puede ser negativo."
            )
        if self.liquid_limit > C.LIQUID_LIMIT_MAX:
            raise LithologicalInputError(
                f"Límite líquido {self.liquid_limit}% excede el máximo "
                f"admisible {C.LIQUID_LIMIT_MAX}%."
            )

        if self.plasticity_index < 0.0:
            raise LithologicalInputError(
                "El índice de plasticidad no puede ser negativo."
            )
        if self.plasticity_index > C.PLASTICITY_INDEX_MAX:
            raise LithologicalInputError(
                f"PI {self.plasticity_index}% excede el máximo admisible "
                f"{C.PLASTICITY_INDEX_MAX}%."
            )

        if self.plasticity_index > self.liquid_limit + C.EPSILON_MACH:
            raise LithologicalInputError(
                f"El índice de plasticidad ({self.plasticity_index}%) no "
                f"puede exceder el límite líquido ({self.liquid_limit}%). "
                f"Restricción de consistencia de Casagrande violada."
            )

        if self.shear_wave_velocity <= 0.0:
            raise LithologicalInputError(
                "La velocidad de onda cortante Vs debe ser estrictamente positiva."
            )
        if self.shear_wave_velocity > C.SHEAR_WAVE_VELOCITY_MAX:
            raise LithologicalInputError(
                f"Vs {self.shear_wave_velocity} m/s excede el máximo "
                f"admisible {C.SHEAR_WAVE_VELOCITY_MAX} m/s."
            )

        if self.void_ratio < 0.0:
            raise LithologicalInputError(
                "La relación de vacíos no puede ser negativa."
            )
        if self.void_ratio > C.VOID_RATIO_MAX:
            raise LithologicalInputError(
                f"Void ratio {self.void_ratio} excede el máximo admisible "
                f"{C.VOID_RATIO_MAX}."
            )

        if self.bulk_density_kg_m3 <= 0.0:
            raise LithologicalInputError(
                "La densidad del suelo debe ser positiva."
            )
        if not (C.BULK_DENSITY_MIN
                <= self.bulk_density_kg_m3
                <= C.BULK_DENSITY_MAX):
            raise LithologicalInputError(
                f"Densidad {self.bulk_density_kg_m3} kg/m³ fuera del rango "
                f"admisible [{C.BULK_DENSITY_MIN}, {C.BULK_DENSITY_MAX}]."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización canónica del tensor de estado."""
        return {
            "uscs_classification": self.uscs_classification,
            "liquid_limit": self.liquid_limit,
            "plasticity_index": self.plasticity_index,
            "shear_wave_velocity": self.shear_wave_velocity,
            "void_ratio": self.void_ratio,
            "is_saturated": self.is_saturated,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
        }


@dataclass(frozen=True, slots=True)
class LithologyDiagnosticReport:
    """
    Reporte estructurado de diagnóstico litológico (compatible con v5.18.0).
    Todos los índices de susceptibilidad están normalizados a [0, 1].
    """
    soil_tensor: SoilTensor
    dynamic_rigidity_modulus_pa: float
    swelling_potential_index: float
    yielding_susceptibility_index: float
    liquefaction_susceptibility_index: float
    emitted_cartridges: Tuple[LithologyCartridge, ...] = ()
    activated_rules: Tuple[DiagnosticRule, ...] = ()
    recommendations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soil_tensor": self.soil_tensor.to_dict(),
            "dynamic_rigidity_modulus_pa": self.dynamic_rigidity_modulus_pa,
            "swelling_potential_index": self.swelling_potential_index,
            "yielding_susceptibility_index": self.yielding_susceptibility_index,
            "liquefaction_susceptibility_index": (
                self.liquefaction_susceptibility_index
            ),
            "activated_rules": [r.value for r in self.activated_rules],
            "recommendations": list(self.recommendations),
            "emitted_cartridges_count": len(self.emitted_cartridges),
        }


@dataclass(frozen=True, slots=True)
class SoilValidationCertificate:
    """
    Certificado de Salida de la Fase 1 (OBSERVE) y Handoff hacia la Fase 2.

    Contiene:
    - El `SoilTensor` validado y normalizado.
    - La auditoría geométrica de la Carta de Plasticidad de Casagrande:
      Línea A ($0.73(LL - 20)$), Línea U ($0.9(LL - 8)$), distancia signada
      y coherencia morfológica arcilla/limo.
    - El Jacobiano de sensibilidad de Fréchet $J_{\Phi} \in \mathbb{R}^{4 \times 5}$
      calculado mediante diferenciación de paso complejo ($h = 10^{-20}$).
    - El sello criptográfico SHA-256 de la Fase 1.
    """
    soil_tensor: SoilTensor
    normalized_uscs: str
    casagrande_a_line_pi: float
    casagrande_u_line_pi: float
    signed_distance_to_a_line: float
    margin_below_u_line: float
    is_above_a_line: bool
    is_below_u_line: bool
    porosity_n: float
    complex_step_jacobian: Dict[str, Dict[str, float]]
    jacobian_frobenius_norm: float
    phase1_session_hash: str

    def continue_to_phase2(
        self,
        manifold: "LithologicalManifold",
        *,
        enable_rsi_level3: bool = True,
    ) -> "GeomechanicalSpectralReport":
        """Puente funcional anidado: Fase 1 ──► Fase 2."""
        return manifold.phase2_close_and_open_phase3(
            self,
            enable_rsi_level3=enable_rsi_level3,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soil_tensor": self.soil_tensor.to_dict(),
            "normalized_uscs": self.normalized_uscs,
            "casagrande_a_line_pi": self.casagrande_a_line_pi,
            "casagrande_u_line_pi": self.casagrande_u_line_pi,
            "signed_distance_to_a_line": self.signed_distance_to_a_line,
            "margin_below_u_line": self.margin_below_u_line,
            "is_above_a_line": self.is_above_a_line,
            "is_below_u_line": self.is_below_u_line,
            "porosity_n": self.porosity_n,
            "complex_step_jacobian": self.complex_step_jacobian,
            "jacobian_frobenius_norm": self.jacobian_frobenius_norm,
            "phase1_session_hash": self.phase1_session_hash,
        }


Phase1LithologyHandoff = SoilValidationCertificate


@dataclass(frozen=True, slots=True)
class LithologyRSILevel3Certificate:
    """
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3) sobre la variedad
    riemanniana $\mathcal{P}_3 = \operatorname{SPD}(3)$ del tensor de rigidez.
    """
    triggered: bool
    iterations: int
    initial_condition_number: float
    final_condition_number: float
    fisher_rao_geodesic_distance: float
    banach_contraction_ratio: float
    is_contractive: bool
    isochoric_determinant_error: float
    wilkinson_shift_applied_pa: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "iterations": self.iterations,
            "initial_condition_number": self.initial_condition_number,
            "final_condition_number": self.final_condition_number,
            "fisher_rao_geodesic_distance": self.fisher_rao_geodesic_distance,
            "banach_contraction_ratio": self.banach_contraction_ratio,
            "is_contractive": self.is_contractive,
            "isochoric_determinant_error": self.isochoric_determinant_error,
            "wilkinson_shift_applied_pa": self.wilkinson_shift_applied_pa,
        }


@dataclass(frozen=True, slots=True)
class PoincareGeomechanicalCertificate:
    """
    Certificado de Mecánica Celeste de Henri Poincaré y Estabilidad KAM
    en el espacio de fases elastodinámico $T^*\mathbb{T}^3$.
    """
    natural_frequencies_rad_s: Tuple[float, float, float]
    kam_smallest_divisor: float
    kam_diophantine_margin: float
    is_kam_non_resonant: bool
    dissipative_liouville_determinant: float
    conservative_symplectic_determinant: float
    rayleigh_trace_decay_rate_1_s: float
    effective_betti_0: int
    euler_poincare_characteristic: int
    kac_poincare_recurrence_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "natural_frequencies_rad_s": list(self.natural_frequencies_rad_s),
            "kam_smallest_divisor": self.kam_smallest_divisor,
            "kam_diophantine_margin": self.kam_diophantine_margin,
            "is_kam_non_resonant": self.is_kam_non_resonant,
            "dissipative_liouville_determinant": (
                self.dissipative_liouville_determinant
            ),
            "conservative_symplectic_determinant": (
                self.conservative_symplectic_determinant
            ),
            "rayleigh_trace_decay_rate_1_s": self.rayleigh_trace_decay_rate_1_s,
            "effective_betti_0": self.effective_betti_0,
            "euler_poincare_characteristic": self.euler_poincare_characteristic,
            "kac_poincare_recurrence_time_s": (
                self.kac_poincare_recurrence_time_s
            ),
        }


@dataclass(frozen=True, slots=True)
class GeomechanicalSpectralReport:
    """
    Certificado de Salida de la Fase 2 (ORIENT) y Handoff hacia la Fase 3.

    Integra el reporte diagnóstico con el difeomorfismo elastodinámico 3D
    de Poisson-Lamé-Lysmer ($\mathbf{K}, \mathbf{C} \in \operatorname{SPD}(3)$),
    el certificado RSI Nivel 3 y el certificado de Mecánica Celeste de Poincaré.
    """
    validation_certificate: SoilValidationCertificate
    diagnostic_report: LithologyDiagnosticReport
    poisson_ratio: float
    lame_lambda_pa: float
    bulk_modulus_pa: float
    young_modulus_pa: float
    p_wave_modulus_pa: float
    p_wave_velocity_m_s: float
    stiffness_tensor_k_pa: Tuple[Tuple[float, float, float], ...]
    damping_tensor_c_pa_s: Tuple[Tuple[float, float, float], ...]
    stiffness_eigenvalues_pa: Tuple[float, float, float]
    damping_eigenvalues_pa_s: Tuple[float, float, float]
    rsi_level3_certificate: LithologyRSILevel3Certificate
    poincare_certificate: PoincareGeomechanicalCertificate
    phase2_session_hash: str

    def continue_to_phase3(
        self,
        manifold: "LithologicalManifold",
        *,
        laplace_s: complex = complex(
            GeomechanicalConstants.DEFAULT_LAPLACE_SIGMA_RAD_S,
            GeomechanicalConstants.DEFAULT_LAPLACE_OMEGA_RAD_S,
        ),
        total_stress_pa: Optional[np.ndarray] = None,
        pore_pressure_pa: float = 0.0,
        strain_rate_1_s: Optional[np.ndarray] = None,
    ) -> "LithologicalSuturationState":
        """Puente funcional anidado: Fase 2 ──► Fase 3."""
        return manifold.phase3_close_loop(
            self,
            laplace_s=laplace_s,
            total_stress_pa=total_stress_pa,
            pore_pressure_pa=pore_pressure_pa,
            strain_rate_1_s=strain_rate_1_s,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_certificate": self.validation_certificate.to_dict(),
            "diagnostic_report": self.diagnostic_report.to_dict(),
            "poisson_ratio": self.poisson_ratio,
            "lame_lambda_pa": self.lame_lambda_pa,
            "bulk_modulus_pa": self.bulk_modulus_pa,
            "young_modulus_pa": self.young_modulus_pa,
            "p_wave_modulus_pa": self.p_wave_modulus_pa,
            "p_wave_velocity_m_s": self.p_wave_velocity_m_s,
            "stiffness_tensor_k_pa": [
                list(row) for row in self.stiffness_tensor_k_pa
            ],
            "damping_tensor_c_pa_s": [
                list(row) for row in self.damping_tensor_c_pa_s
            ],
            "stiffness_eigenvalues_pa": list(self.stiffness_eigenvalues_pa),
            "damping_eigenvalues_pa_s": list(self.damping_eigenvalues_pa_s),
            "rsi_level3_certificate": self.rsi_level3_certificate.to_dict(),
            "poincare_certificate": self.poincare_certificate.to_dict(),
            "phase2_session_hash": self.phase2_session_hash,
        }


Phase2LithologyHandoff = GeomechanicalSpectralReport


@dataclass(frozen=True, slots=True)
class CrowbarPortState:
    """
    Estado determinista del puerto de enclavamiento en silicio ESP32 (GPIO14)
    acoplado al tiristor BT151 (Circuito Crowbar).
    """
    gpio_pin: int
    bt151_thyristor_latched: bool
    is_verdict_coherent_flag: bool
    iram_isr_latency_ns: float
    actuation_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpio_pin": self.gpio_pin,
            "bt151_thyristor_latched": self.bt151_thyristor_latched,
            "is_verdict_coherent_flag": self.is_verdict_coherent_flag,
            "iram_isr_latency_ns": self.iram_isr_latency_ns,
            "actuation_mode": self.actuation_mode,
        }


@dataclass(frozen=True, slots=True)
class LithologicalSuturationState:
    """
    Estado de Suturación Litológica Final de la Fase 3 (DECIDE).

    Encapsula:
    - El Tensor de Impedancia Geomecánica Compleja $Z_{\mathrm{geo}}(s) = \mathbf{K} + s\mathbf{C}$
      como condición de frontera de Dirichlet sobre el complejo simplicial.
    - La certificación positivo-real de Brune ($\lambda_{\min}(\operatorname{Re} Z_{\mathrm{geo}}) > 0$).
    - La potencia de disipación termodinámica de Terzaghi $P_{\mathrm{diss}} \ge 0$.
    - El veredicto en el retículo bi-Heyting $\Omega_3$.
    - El estado del puerto de silicio Crowbar BT151 (GPIO14).
    - El sello criptográfico de gobernanza SHA-256.
    """
    spectral_report: GeomechanicalSpectralReport
    laplace_frequency_s: Tuple[float, float]
    impedance_tensor_real_pa: Tuple[Tuple[float, float, float], ...]
    impedance_tensor_imag_pa: Tuple[Tuple[float, float, float], ...]
    dirichlet_real_min_eigenvalue_pa: float
    is_brune_positive_real: bool
    terzaghi_effective_mean_stress_pa: float
    terzaghi_dissipation_power_w_m3: float
    is_terzaghi_dissipative: bool
    heyting_verdict: LithologicalHeytingTruth
    crowbar_port: CrowbarPortState
    governance_seal_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectral_report": self.spectral_report.to_dict(),
            "laplace_frequency_s": {
                "sigma_rad_s": self.laplace_frequency_s[0],
                "omega_rad_s": self.laplace_frequency_s[1],
            },
            "impedance_tensor_real_pa": [
                list(row) for row in self.impedance_tensor_real_pa
            ],
            "impedance_tensor_imag_pa": [
                list(row) for row in self.impedance_tensor_imag_pa
            ],
            "dirichlet_real_min_eigenvalue_pa": (
                self.dirichlet_real_min_eigenvalue_pa
            ),
            "is_brune_positive_real": self.is_brune_positive_real,
            "terzaghi_effective_mean_stress_pa": (
                self.terzaghi_effective_mean_stress_pa
            ),
            "terzaghi_dissipation_power_w_m3": (
                self.terzaghi_dissipation_power_w_m3
            ),
            "is_terzaghi_dissipative": self.is_terzaghi_dissipative,
            "heyting_verdict": self.heyting_verdict.value,
            "heyting_truth_numeric": self.heyting_verdict.numeric_value,
            "crowbar_port": self.crowbar_port.to_dict(),
            "governance_seal_sha256": self.governance_seal_sha256,
        }


Phase3LithologyDecision = LithologicalSuturationState


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        §8  OPERADOR LITOLÓGICO EN TRES FASES ANIDADAS (MORFISMO Φ)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LithologicalManifold(Morphism):
    """
    Morfismo Soberano Φ : V_externo ──► V_PHYSICS estructurado en 3 Fases Anidadas:

        Fase 1 (OBSERVE) ──► SoilValidationCertificate
                                    │
                                    ▼  (phase1_close_and_open_phase2 → phase2_from_phase1)
        Fase 2 (ORIENT)  ──► GeomechanicalSpectralReport
                                    │
                                    ▼  (phase2_close_and_open_phase3 → phase3_from_phase2)
        Fase 3 (DECIDE)  ──► LithologicalSuturationState ──► CategoricalState
    """

    def __init__(self) -> None:
        super().__init__(name="lithological_evaluation")
        self.target_stratum = Stratum.PHYSICS
        self._metric_tensor = MetricTensorFactory.build_physics_tensor()

    @property
    def domain(self) -> frozenset[Stratum]:
        return frozenset()

    @property
    def codomain(self) -> Stratum:
        return Stratum.PHYSICS

    # ══════════════════════════════════════════════════════════════════════════
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  FASE 1: OBSERVE  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ══════════════════════════════════════════════════════════════════════════
    # Parseo del SoilTensor, Higiene Geométrica de la Carta de Plasticidad de
    # Casagrande (Líneas A y U), y Diferenciación Holomorfa de Paso Complejo.
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_soil_tensor(self, payload: Mapping[str, Any]) -> SoilTensor:
        """
        [FASE 1.1] Convierte el payload externo en un `SoilTensor` validado.
        """
        try:
            soil_tensor = SoilTensor(
                uscs_classification=str(payload.get("uscs", "SW")),
                liquid_limit=float(payload.get("liquid_limit", 0.0)),
                plasticity_index=float(payload.get("plasticity_index", 0.0)),
                shear_wave_velocity=float(payload.get("vs", 300.0)),
                void_ratio=float(payload.get("void_ratio", 0.5)),
                is_saturated=_parse_bool(payload.get("is_saturated", False)),
                bulk_density_kg_m3=float(
                    payload.get(
                        "bulk_density_kg_m3",
                        GeomechanicalConstants.DEFAULT_SOIL_DENSITY_KG_M3,
                    )
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LithologicalInputError(
                f"Degeneración de tipos en input litológico: {exc}"
            ) from exc

        soil_tensor.validate()
        return soil_tensor

    def phase1_casagrande_plasticity_audit(
        self, soil: SoilTensor
    ) -> Dict[str, Any]:
        """
        [FASE 1.2] Audita la ubicación del estado litológico sobre la Carta de
        Plasticidad de Casagrande (ASTM D2487):

        - Línea A: $PI_A(LL) = \max(0, 0.73 \cdot (LL - 20))$
        - Línea U: $PI_U(LL) = \max(0, 0.90 \cdot (LL - 8))$
        - Porosidad euleriana derivada: $n = e_0 / (1 + e_0) \in [0, 1)$
        """
        C = GeomechanicalConstants
        ll = soil.liquid_limit
        pi = soil.plasticity_index

        a_line_pi = max(
            0.0,
            C.CASAGRANDE_A_LINE_SLOPE * (ll - C.CASAGRANDE_A_LINE_OFFSET),
        )
        u_line_pi = max(
            0.0,
            C.CASAGRANDE_U_LINE_SLOPE * (ll - C.CASAGRANDE_U_LINE_OFFSET),
        )

        signed_dist_a = _assert_finite(pi - a_line_pi, "signed_distance_a_line")
        # Para LL <= 8, suelos no plásticos con PI = 0 están bajo la línea U
        effective_u_bound = max(u_line_pi, 4.0)
        margin_u = _assert_finite(
            effective_u_bound - pi, "margin_below_u_line"
        )
        porosity_n = _assert_finite(
            _safe_divide(soil.void_ratio, 1.0 + soil.void_ratio, fallback=0.0),
            "porosity_n",
        )

        return {
            "a_line_pi": a_line_pi,
            "u_line_pi": u_line_pi,
            "signed_distance_to_a_line": signed_dist_a,
            "margin_below_u_line": margin_u,
            "is_above_a_line": bool(signed_dist_a >= -C.EPSILON_MACH),
            "is_below_u_line": bool(margin_u >= -C.EPSILON_MACH),
            "porosity_n": porosity_n,
        }

    def _holomorphic_constitutive_map(
        self,
        ll: complex,
        pi: complex,
        vs: complex,
        e0: complex,
        rho: complex,
        *,
        uscs: str,
        is_saturated: bool,
    ) -> Dict[str, complex]:
        """
        [FASE 1.3] Extensión analítica holomorfa $\Phi_{\mathbb{C}} : \mathbb{C}^5 \to \mathbb{C}^4$
        de las funciones constitutivas para evaluación de derivadas exactas de
        paso complejo sin cancelación sustractiva.
        """
        C = GeomechanicalConstants
        g_max_c = rho * vs * vs

        activity_c = (ll * pi) / 100.0
        i_sw_c = _sigmoid_complex(
            activity_c,
            C.SWELLING_SIGMOID_MIDPOINT,
            C.SWELLING_SIGMOID_STEEPNESS,
        )

        # Para el índice de cedencia en el interior suave:
        raw_y_c = (e0 * e0) / (vs * C.YIELDING_REFERENCE_SCALE)
        if raw_y_c.real <= 0.0:
            i_y_c = complex(0.0, 0.0)
        elif raw_y_c.real >= 1.0:
            i_y_c = complex(1.0, 0.0)
        else:
            i_y_c = raw_y_c

        # Para el índice de licuación en el interior activo:
        if is_saturated and _is_sand_like(uscs):
            vs_crit = C.CRITICAL_LIQUEFACTION_VS_M_S
            r_c = (vs_crit - vs) / vs_crit
            if r_c.real <= 0.0:
                i_liq_c = complex(0.0, 0.0)
            elif r_c.real >= 1.0:
                i_liq_c = complex(1.0, 0.0)
            else:
                i_liq_c = r_c ** C.LIQUEFACTION_NONLINEAR_EXPONENT
        else:
            i_liq_c = complex(0.0, 0.0)

        return {
            "G_max": g_max_c,
            "I_sw": i_sw_c,
            "I_y": i_y_c,
            "I_liq": i_liq_c,
        }

    def phase1_complex_step_jacobian(
        self, soil: SoilTensor
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """
        [FASE 1.4] Calcula el Jacobiano de Fréchet $J_{\Phi} \in \mathbb{R}^{4 \times 5}$
        de las magnitudes constitutivas respecto a $(LL, PI, V_s, e_0, \rho)$
        usando diferenciación de paso complejo (Squire & Trapp, 1998):

            $$\frac{\partial \Phi_i}{\partial x_j}(x) = \frac{\operatorname{Im}\!\big(\Phi_i(x + i h \mathbf{e}_j)\big)}{h} + \mathcal{O}(h^2)$$

        con $h = 10^{-20}$, alcanzando exactitud de precisión de máquina IEEE-754.
        """
        h = GeomechanicalConstants.COMPLEX_STEP_H
        base_vars: Dict[str, float] = {
            "liquid_limit": soil.liquid_limit,
            "plasticity_index": soil.plasticity_index,
            "shear_wave_velocity": soil.shear_wave_velocity,
            "void_ratio": soil.void_ratio,
            "bulk_density_kg_m3": soil.bulk_density_kg_m3,
        }
        outputs = ("G_max", "I_sw", "I_y", "I_liq")
        jacobian: Dict[str, Dict[str, float]] = {out: {} for out in outputs}
        sq_entries: List[float] = []

        for var_name in base_vars:
            perturbed = {
                k: complex(v, h if k == var_name else 0.0)
                for k, v in base_vars.items()
            }
            eval_c = self._holomorphic_constitutive_map(
                ll=perturbed["liquid_limit"],
                pi=perturbed["plasticity_index"],
                vs=perturbed["shear_wave_velocity"],
                e0=perturbed["void_ratio"],
                rho=perturbed["bulk_density_kg_m3"],
                uscs=soil.uscs_classification,
                is_saturated=soil.is_saturated,
            )
            for out_name in outputs:
                deriv = _assert_finite(
                    eval_c[out_name].imag / h,
                    f"d({out_name})/d({var_name})",
                )
                jacobian[out_name][var_name] = deriv
                # Escalamos G_max por 1e-6 (MPa) al computar la norma de Frobenius
                scale = 1e-6 if out_name == "G_max" else 1.0
                sq_entries.append((deriv * scale) ** 2)

        frob_norm = _assert_finite(
            math.sqrt(max(0.0, _kbn_sum(sq_entries))),
            "jacobian_frobenius_norm",
        )
        return jacobian, frob_norm

    def _phase1_session_hash(
        self,
        soil: SoilTensor,
        casagrande: Mapping[str, Any],
        frob_norm: float,
    ) -> str:
        """[FASE 1.5] Genera el digest criptográfico SHA-256 determinista de Fase 1."""
        canonical_payload = {
            "soil": soil.to_dict(),
            "a_line_pi": round(float(casagrande["a_line_pi"]), 8),
            "u_line_pi": round(float(casagrande["u_line_pi"]), 8),
            "porosity_n": round(float(casagrande["porosity_n"]), 8),
            "jacobian_frobenius_norm": round(frob_norm, 8),
        }
        encoded = json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def phase1_close_and_open_phase2(
        self,
        payload_or_soil: Union[Mapping[str, Any], SoilTensor],
    ) -> SoilValidationCertificate:
        """
        [MÉTODO TERMINAL DE FASE 1 ──► CONTINUACIÓN FORMAL EN FASE 2]

        Cierra la Fase 1 (OBSERVE) emitiendo el `SoilValidationCertificate` y
        transiciona formalmente a la Fase 2 invocando `self.phase2_from_phase1(...)`.
        """
        soil = (
            payload_or_soil
            if isinstance(payload_or_soil, SoilTensor)
            else self._parse_soil_tensor(payload_or_soil)
        )
        soil.validate()

        uscs_norm = _normalize_uscs(soil.uscs_classification)
        if _is_peat(uscs_norm):
            logger.critical(
                "Singularidad litológica detectada en Fase 1: uscs=%s",
                soil.uscs_classification,
            )
            raise LithologicalSingularityError(
                "La presencia de Turba (PT) induce colapso de la frontera "
                "de anclaje y exige estrategia de cimentación profunda o "
                "sustitución/mejora integral del estrato."
            )

        casagrande = self.phase1_casagrande_plasticity_audit(soil)
        jacobian, frob_norm = self.phase1_complex_step_jacobian(soil)
        session_hash = self._phase1_session_hash(soil, casagrande, frob_norm)

        certificate = SoilValidationCertificate(
            soil_tensor=soil,
            normalized_uscs=uscs_norm,
            casagrande_a_line_pi=float(casagrande["a_line_pi"]),
            casagrande_u_line_pi=float(casagrande["u_line_pi"]),
            signed_distance_to_a_line=float(
                casagrande["signed_distance_to_a_line"]
            ),
            margin_below_u_line=float(casagrande["margin_below_u_line"]),
            is_above_a_line=bool(casagrande["is_above_a_line"]),
            is_below_u_line=bool(casagrande["is_below_u_line"]),
            porosity_n=float(casagrande["porosity_n"]),
            complex_step_jacobian=jacobian,
            jacobian_frobenius_norm=frob_norm,
            phase1_session_hash=session_hash,
        )

        # Eslabón de anidamiento formal: el cierre de Fase 1 abre la Fase 2
        return self.phase2_from_phase1(certificate)

    # ══════════════════════════════════════════════════════════════════════════
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  FASE 2: ORIENT  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ══════════════════════════════════════════════════════════════════════════
    # Difeomorfismo Elastodinámico 3D de Poisson-Lamé-Lysmer (K, C en SPD(3)),
    # Automejora Recursiva RSI Nivel 3 (Fisher-Rao / Higham-Löwner), Mecánica
    # Celeste de Henri Poincaré (KAM / Liouville) y Cuantización Hadrónica TOON.
    # ══════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(
        self, certificate: SoilValidationCertificate
    ) -> SoilValidationCertificate:
        """
        [MÉTODO INICIAL DE FASE 2 ◄── CONTINUACIÓN DESDE FASE 1]

        Verifica la integridad criptográfica y geométrica del certificado
        de Fase 1 antes de habilitar los operadores espectrales de Fase 2.
        """
        if not isinstance(certificate, SoilValidationCertificate):
            raise LithologicalInputError(
                "Se esperaba un SoilValidationCertificate válido desde la Fase 1."
            )
        certificate.soil_tensor.validate()
        if not certificate.phase1_session_hash or len(certificate.phase1_session_hash) != 64:
            raise LithologicalInputError(
                "Hash de sesión de Fase 1 inválido o ausente."
            )
        if not math.isfinite(certificate.jacobian_frobenius_norm):
            raise LithologicalNumericalError(
                "El Jacobiano de paso complejo de Fase 1 contiene entradas no finitas."
            )
        return certificate

    # ------------------------------------------------------------------
    # §8.2.A  Magnitudes constitutivas derivadas (compatibles con v5.18.0)
    # ------------------------------------------------------------------

    def _compute_dynamic_rigidity_pa(self, soil: SoilTensor) -> float:
        """
        [FASE 2.1] Módulo cortante dinámico máximo: $G_{\max} = \rho \cdot V_s^2$ [Pa].
        """
        rho = soil.bulk_density_kg_m3
        vs = soil.shear_wave_velocity
        g_max = rho * vs * vs
        return _assert_finite(g_max, "G_max")

    def _compute_swelling_potential_index(self, soil: SoilTensor) -> float:
        """
        [FASE 2.2] Índice normalizado de expansividad potencial $I_{\mathrm{sw}} \in (0, 1)$.
        """
        C = GeomechanicalConstants
        activity_product = soil.liquid_limit * soil.plasticity_index / 100.0
        index = _sigmoid(
            activity_product,
            C.SWELLING_SIGMOID_MIDPOINT,
            C.SWELLING_SIGMOID_STEEPNESS,
        )
        return _assert_finite(index, "swelling_potential_index")

    def _compute_yielding_susceptibility_index(
        self, soil: SoilTensor
    ) -> float:
        """
        [FASE 2.3] Índice normalizado de susceptibilidad a cedencia $I_y \in [0, 1]$.
        """
        C = GeomechanicalConstants
        raw = _safe_divide(
            soil.void_ratio ** 2,
            soil.shear_wave_velocity,
            fallback=0.0,
        )
        normalized = raw / C.YIELDING_REFERENCE_SCALE
        index = _clamp(normalized, 0.0, 1.0)
        return _assert_finite(index, "yielding_susceptibility_index")

    def _compute_compression_index_proxy(self, soil: SoilTensor) -> float:
        """
        [FASE 2.4] Proxy correlacional de Skempton (1944): $C_c \approx 0.009 \cdot (LL - 10)$.
        """
        C = GeomechanicalConstants
        cc = max(
            0.0,
            C.SKEMPTON_CC_FACTOR * (soil.liquid_limit - C.SKEMPTON_CC_OFFSET),
        )
        return _assert_finite(cc, "compression_index_proxy")

    def _compute_liquefaction_susceptibility_index(
        self, soil: SoilTensor
    ) -> float:
        """
        [FASE 2.5] Índice proxy normalizado de susceptibilidad a licuación $I_{\mathrm{liq}} \in [0, 1]$.
        """
        if not soil.is_saturated:
            return 0.0
        if not _is_sand_like(soil.uscs_classification):
            return 0.0

        C = GeomechanicalConstants
        vs_crit = C.CRITICAL_LIQUEFACTION_VS_M_S
        r = (vs_crit - soil.shear_wave_velocity) / vs_crit
        r_clamped = max(0.0, r)
        index = r_clamped ** C.LIQUEFACTION_NONLINEAR_EXPONENT
        return _assert_finite(_clamp(index, 0.0, 1.0), "liquefaction_index")

    # ------------------------------------------------------------------
    # §8.2.B  Difeomorfismo Elastodinámico 3D de Poisson-Lamé-Lysmer
    # ------------------------------------------------------------------

    def phase2_poisson_elastodynamic_tensors(
        self, soil: SoilTensor, g_max_pa: float
    ) -> Dict[str, Any]:
        r"""
        [FASE 2.6] Calcula el difeomorfismo elástico y viscoso de Poisson-Lamé-Lysmer:

        1. Coeficiente de Poisson dinámico $\nu \in [0.15, 0.495]$:
           Integra plasticidad del esqueleto mineral y rigidización volumétrica
           no drenada de Biot-Gassmann bajo saturación.
        2. Parámetros de Lamé y módulos elásticos:
           $$\lambda = \frac{2 G_{\max} \nu}{1 - 2\nu}, \quad K_{\mathrm{bulk}} = \frac{2 G_{\max}(1+\nu)}{3(1-2\nu)}, \quad E_{\max} = 2 G_{\max}(1+\nu)$$
           $$M_{\mathrm{P}} = \lambda + 2 G_{\max}, \quad V_p = \sqrt{\frac{M_{\mathrm{P}}}{\rho}}$$
        3. Tensores 3D de rigidez $\mathbf{K} \in \operatorname{SPD}(3)$ y amortiguamiento
           viscoso radiativo-histerético $\mathbf{C} \in \operatorname{SPD}(3)$ sobre el
           semiespacio de Lysmer-Richart-Gazetas, protegidos por la cota de Wilkinson.
        """
        C = GeomechanicalConstants
        nu_skel = _clamp(
            0.22
            + 0.0008 * soil.plasticity_index
            + 0.03 * math.tanh(soil.void_ratio - 0.6),
            C.POISSON_RATIO_MIN,
            C.POISSON_RATIO_MAX_DRAINED,
        )
        if soil.is_saturated:
            fluid_boost = 0.14 * (
                1.0 - _clamp(soil.shear_wave_velocity / C.SHEAR_WAVE_VELOCITY_MAX, 0.0, 0.95)
            )
            nu = _clamp(
                 max(0.40, nu_skel + fluid_boost),
                0.40,
                C.POISSON_RATIO_MAX_UNDRAINED,
            )
        else:
            nu = nu_skel

        denom_1_2nu = max(1.0 - 2.0 * nu, 1e-4)
        lame_lambda = _assert_finite(
            (2.0 * g_max_pa * nu) / denom_1_2nu, "lame_lambda_pa"
        )
        bulk_modulus = _assert_finite(
            (2.0 * g_max_pa * (1.0 + nu)) / (3.0 * denom_1_2nu),
            "bulk_modulus_pa",
        )
        young_modulus = _assert_finite(
            2.0 * g_max_pa * (1.0 + nu), "young_modulus_pa"
        )
        p_wave_modulus = _assert_finite(
            lame_lambda + 2.0 * g_max_pa, "p_wave_modulus_pa"
        )
        p_wave_velocity = _assert_finite(
            math.sqrt(p_wave_modulus / soil.bulk_density_kg_m3),
            "p_wave_velocity_m_s",
        )

        # Semiespacio elastodinámico de Lysmer-Richart (radio característico R₀ = 1 m)
        r0 = C.FOUNDATION_CHARACTERISTIC_RADIUS_M
        k_xx = (8.0 * g_max_pa * r0) / (2.0 - nu)
        k_yy = k_xx * (1.0 + 0.02 * math.tanh(soil.void_ratio))
        k_zz = (4.0 * g_max_pa * r0) / (1.0 - nu)

        # Acoplamiento transversal débil por anisotropía de fábrica granular/laminar
        aniso_coupling = 0.03 * math.tanh(soil.void_ratio / 2.0) * k_xx
        k_raw = np.array(
            [
                [k_xx, aniso_coupling, 0.5 * aniso_coupling],
                [aniso_coupling, k_yy, 0.5 * aniso_coupling],
                [0.5 * aniso_coupling, 0.5 * aniso_coupling, k_zz],
            ],
            dtype=np.float64,
        )

        # Corrección por la cota espectral de Wilkinson de la FPU
        wilkinson_k = 3.0 * C.IEEE754_U * float(np.linalg.norm(k_raw, ord="fro"))
        k_mat = 0.5 * (k_raw + k_raw.T) + wilkinson_k * np.eye(3, dtype=np.float64)

        # Amortiguamiento radiativo de Lysmer + amortiguamiento histerético de Rayleigh
        rho = soil.bulk_density_kg_m3
        vs = soil.shear_wave_velocity
        area_eff = math.pi * r0 * r0
        xi_hyst = C.HYSTERETIC_DAMPING_BASE + 0.05 * _clamp(
            soil.plasticity_index / 100.0, 0.0, 1.0
        )
        omega_ref = C.DEFAULT_LAPLACE_OMEGA_RAD_S

        c_xx_rad = area_eff * rho * vs
        c_yy_rad = c_xx_rad * (1.0 + 0.01 * math.tanh(soil.void_ratio))
        c_zz_rad = (3.4 * r0 * r0 / (1.0 - nu)) * rho * vs

        c_raw = np.diag([c_xx_rad, c_yy_rad, c_zz_rad]) + (
            (2.0 * xi_hyst / omega_ref) * k_mat
        )
        wilkinson_c = 3.0 * C.IEEE754_U * float(np.linalg.norm(c_raw, ord="fro"))
        c_mat = 0.5 * (c_raw + c_raw.T) + wilkinson_c * np.eye(3, dtype=np.float64)

        return {
            "poisson_ratio": nu,
            "lame_lambda_pa": lame_lambda,
            "bulk_modulus_pa": bulk_modulus,
            "young_modulus_pa": young_modulus,
            "p_wave_modulus_pa": p_wave_modulus,
            "p_wave_velocity_m_s": p_wave_velocity,
            "k_matrix": k_mat,
            "c_matrix": c_mat,
            "wilkinson_shift_k": wilkinson_k,
        }

    # ------------------------------------------------------------------
    # §8.2.C  RSI Nivel 3 en la Variedad Riemanniana SPD(3)
    # ------------------------------------------------------------------

    @staticmethod
    def _fisher_rao_spd_distance(mat_a: np.ndarray, mat_b: np.ndarray) -> float:
        r"""
        Distancia geodésica invariante afín de Fisher-Rao en $\operatorname{SPD}(3)$:
            $$d_{\mathrm{FR}}(\mathbf{A}, \mathbf{B}) = \left\| \log\!\left(\mathbf{A}^{-1/2} \mathbf{B} \mathbf{A}^{-1/2}\right) \right\|_F = \sqrt{\sum_{i=1}^3 \left(\ln \mu_i\right)^2}$$
        donde $\mu_i > 0$ son los autovalores generalizados de $\mathbf{B}\mathbf{v} = \mu\mathbf{A}\mathbf{v}$.
        """
        eig_a, vec_a = np.linalg.eigh(0.5 * (mat_a + mat_a.T))
        eig_a_safe = np.maximum(eig_a, GeomechanicalConstants.EPSILON_MACH)
        inv_sqrt_a = (vec_a * (1.0 / np.sqrt(eig_a_safe))) @ vec_a.T
        whitened = inv_sqrt_a @ (0.5 * (mat_b + mat_b.T)) @ inv_sqrt_a
        mu = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
        mu_safe = np.maximum(mu, GeomechanicalConstants.EPSILON_MACH)
        log_mu_sq = [float(math.log(float(m)) ** 2) for m in mu_safe]
        return math.sqrt(max(0.0, _kbn_sum(log_mu_sq)))

    def phase2_rsi_level3_spd_impedance_refinement(
        self,
        k_matrix: np.ndarray,
        wilkinson_shift_pa: float,
        *,
        enable_rsi: bool = True,
        max_cond: float = GeomechanicalConstants.SPD_MAX_CONDITION_NUMBER,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], LithologyRSILevel3Certificate]:
        r"""
        [FASE 2.7] Automejora Recursiva de Nivel 3 (RSI-L3) sobre $\operatorname{SPD}(3)$.

        Proyecta espectralmente $\mathbf{K}$ mediante Higham-Löwner y, si el número
        de condición espectral $\kappa(\mathbf{K}) = \lambda_{\max}/\lambda_{\min}$
        excede el umbral operativo o si el suelo presenta alta anisotropía,
        ejecuta contracción geodésica isocórica de Fisher-Rao:

            $$\lambda_i^{(m+1)} = \left(\lambda_i^{(m)}\right)^{1 - \alpha} \cdot \bar{\lambda}_{\mathrm{geom}}^{\alpha}$$

        preservando exactamente el determinante volumétrico ($\prod_i \lambda_i^{(m)} = \det \mathbf{K}^{(0)}$)
        y garantizando contracción estricta de Banach ($q_{\mathrm{Banach}} < 1$).
        """
        C = GeomechanicalConstants
        sym_k = 0.5 * (k_matrix + k_matrix.T)
        eigvals_0, eigvecs = np.linalg.eigh(sym_k)
        eigvals_0 = np.maximum(eigvals_0, C.EPSILON_MACH)
        k_initial = (eigvecs * eigvals_0) @ eigvecs.T

        cond_0 = float(eigvals_0[-1] / eigvals_0[0])
        log_det_0 = _kbn_sum([math.log(float(ev)) for ev in eigvals_0])
        geom_mean = math.exp(log_det_0 / 3.0)

        curr_eigs = eigvals_0.copy()
        curr_cond = cond_0
        iterations = 0
        triggered = False
        contraction_ratios: List[float] = []

        if enable_rsi and curr_cond > max_cond:
            triggered = True
            alpha = C.RSI_GEODESIC_STEP
            for _ in range(C.RSI_MAX_ITERATIONS):
                if curr_cond <= max_cond:
                    break
                prev_lyapunov = (curr_cond / max_cond) ** 2
                curr_eigs = np.array(
                    [
                        (float(ev) ** (1.0 - alpha)) * (geom_mean ** alpha)
                        for ev in curr_eigs
                    ],
                    dtype=np.float64,
                )
                curr_eigs = np.sort(np.maximum(curr_eigs, C.EPSILON_MACH))
                curr_cond = float(curr_eigs[-1] / curr_eigs[0])
                next_lyapunov = (curr_cond / max_cond) ** 2
                if prev_lyapunov > C.EPSILON_MACH:
                    contraction_ratios.append(next_lyapunov / prev_lyapunov)
                iterations += 1

        k_refined = (eigvecs * curr_eigs) @ eigvecs.T
        k_refined = 0.5 * (k_refined + k_refined.T)

        fr_dist = self._fisher_rao_spd_distance(k_initial, k_refined)
        log_det_final = _kbn_sum([math.log(float(ev)) for ev in curr_eigs])
        det_rel_error = abs(log_det_final - log_det_0) / max(1.0, abs(log_det_0))

        if contraction_ratios:
            q_banach = max(contraction_ratios)
        else:
            # Factor analítico intrínseco del operador geodésico (1 - α)²
            q_banach = (1.0 - C.RSI_GEODESIC_STEP) ** 2

        cert = LithologyRSILevel3Certificate(
            triggered=triggered,
            iterations=iterations,
            initial_condition_number=cond_0,
            final_condition_number=curr_cond,
            fisher_rao_geodesic_distance=fr_dist,
            banach_contraction_ratio=q_banach,
            is_contractive=bool(q_banach < 1.0),
            isochoric_determinant_error=det_rel_error,
            wilkinson_shift_applied_pa=wilkinson_shift_pa,
        )
        eig_tuple = (
            float(curr_eigs[0]),
            float(curr_eigs[1]),
            float(curr_eigs[2]),
        )
        return k_refined, eig_tuple, cert

    # ------------------------------------------------------------------
    # §8.2.D  Mecánica Celeste de Henri Poincaré y Estabilidad KAM
    # ------------------------------------------------------------------

    def phase2_poincare_celestial_elastodynamic_audit(
        self,
        soil: SoilTensor,
        k_matrix: np.ndarray,
        c_matrix: np.ndarray,
        liquefaction_index: float,
    ) -> PoincareGeomechanicalCertificate:
        r"""
        [FASE 2.8] Audita la estabilidad de Mecánica Celeste de Henri Poincaré
        en el espacio de fases elastodinámico $T^*\mathbb{T}^3 \cong \mathbb{R}^6$:

        1. Frecuencias propias $\boldsymbol{\omega} = (\omega_1, \omega_2, \omega_3)$
           del semiespacio con masa modal efectiva $\mathbf{M}_{\mathrm{eff}} = m_0 \mathbf{I}_3$,
           donde $m_0 = \frac{4}{3}\pi R_0^3 \rho$.
        2. Condición diofántica de no-resonancia de Kolmogorov-Arnold-Moser (KAM):
           $$|\mathbf{k} \cdot \boldsymbol{\omega}| \ge \gamma_{\mathrm{KAM}} \|\boldsymbol{\omega}\|_2 |\mathbf{k}|_1^{-\tau}, \qquad 0 < |\mathbf{k}|_1 \le K_{\max}$$
           para vectores enteros de resonancia $\mathbf{k} \in \mathbb{Z}^3$ con
           signos mixtos (resonancias internas entre modos de corte y compresión).
        3. Teorema de Liouville-Poincaré:
           - Flujo hamiltoniano conservativo (propagador simpléctico de Cayley):
             $\det \Phi_{\Delta t}^{\mathrm{symp}} = 1.0$.
           - Flujo disipativo de Rayleigh-Terzaghi:
             $\det \Phi_{\Delta t}^{\mathrm{diss}} = \exp\!\left(-\Delta t \operatorname{Tr}(\mathbf{M}_{\mathrm{eff}}^{-1} \mathbf{C})\right) \in (0, 1]$.
        4. Fragmentación topológica de Euler-Poincaré ante solitón de licuación:
           $\beta_0^{\mathrm{eff}} = 1 + \lfloor 100 \cdot I_{\mathrm{liq}}^2 \rfloor$.
        """
        C = GeomechanicalConstants
        r0 = C.FOUNDATION_CHARACTERISTIC_RADIUS_M
        m_eff = (4.0 / 3.0) * math.pi * (r0 ** 3) * soil.bulk_density_kg_m3

        eig_k = np.maximum(
            np.linalg.eigvalsh(0.5 * (k_matrix + k_matrix.T)),
            C.EPSILON_MACH,
        )
        omegas = np.sqrt(eig_k / m_eff)
        w1, w2, w3 = float(omegas[0]), float(omegas[1]), float(omegas[2])
        omega_norm = math.sqrt(_kbn_sum([w1 * w1, w2 * w2, w3 * w3]))

        # Búsqueda exhaustiva de divisores pequeños diofánticos de KAM en ℤ³
        k_max = C.KAM_MAX_HARMONIC_ORDER
        gamma = C.KAM_DIOPHANTINE_GAMMA
        tau = C.KAM_DIOPHANTINE_TAU

        smallest_divisor = float("inf")
        min_kam_margin = float("inf")

        for k1 in range(-k_max, k_max + 1):
            for k2 in range(-k_max, k_max + 1):
                for k3 in range(0, k_max + 1):
                    l1_norm = abs(k1) + abs(k2) + abs(k3)
                    if l1_norm < 2 or l1_norm > k_max:
                        continue
                    # Sólo resonancias mixtas (al menos dos signos opuestos)
                    signs = [s for s in (k1, k2, k3) if s != 0]
                    if min(signs) >= 0 or max(signs) <= 0:
                        continue
                    divisor = abs(k1 * w1 + k2 * w2 + k3 * w3) / max(
                        omega_norm, C.EPSILON_MACH
                    )
                    bound = gamma * (float(l1_norm) ** (-tau))
                    margin = divisor - bound
                    if divisor < smallest_divisor:
                        smallest_divisor = divisor
                    if margin < min_kam_margin:
                        min_kam_margin = margin

        if not math.isfinite(smallest_divisor):
            smallest_divisor = 1.0
            min_kam_margin = 1.0

        # Propaganda simpléctica de Cayley-Poincaré en ℝ⁶ (límite conservativo)
        dt = C.POINCARE_INTEGRATION_DT_S
        zero3 = np.zeros((3, 3), dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        eye6 = np.eye(6, dtype=np.float64)
        j_hamilton = np.block([[zero3, eye3 / m_eff], [-k_matrix, zero3]])
        cayley_lhs = eye6 - 0.5 * dt * j_hamilton
        cayley_rhs = eye6 + 0.5 * dt * j_hamilton
        symplectic_mat = np.linalg.solve(cayley_lhs, cayley_rhs)
        symp_det = float(np.linalg.det(symplectic_mat))

        # Contracción del volumen de fases de Liouville bajo amortiguamiento C
        trace_damping_rate = float(np.trace(c_matrix)) / m_eff
        dissipative_liouville_det = math.exp(-dt * trace_damping_rate)

        # Fragmentación topológica de Euler-Poincaré y recurrencia de Kac-Poincaré
        effective_betti_0 = 1 + int(math.floor(100.0 * (liquefaction_index ** 2)))
        effective_betti_1 = 0 if liquefaction_index > 0.5 else 3
        euler_poincare_chi = effective_betti_0 - effective_betti_1

        mean_freq_hz = (w1 + w2 + w3) / (6.0 * math.pi)
        recurrence_time_s = _safe_divide(
            float(effective_betti_0),
            mean_freq_hz * max(smallest_divisor, 1e-4),
            fallback=1.0,
        )

        return PoincareGeomechanicalCertificate(
            natural_frequencies_rad_s=(w1, w2, w3),
            kam_smallest_divisor=smallest_divisor,
            kam_diophantine_margin=min_kam_margin,
            is_kam_non_resonant=bool(min_kam_margin >= -1e-3),
            dissipative_liouville_determinant=dissipative_liouville_det,
            conservative_symplectic_determinant=symp_det,
            rayleigh_trace_decay_rate_1_s=trace_damping_rate,
            effective_betti_0=effective_betti_0,
            euler_poincare_characteristic=euler_poincare_chi,
            kac_poincare_recurrence_time_s=recurrence_time_s,
        )

    # ------------------------------------------------------------------
    # §8.2.E  Evaluación de reglas diagnósticas y cartuchos TOON
    # ------------------------------------------------------------------

    def _evaluate_diagnostic_rules(
        self, soil: SoilTensor
    ) -> LithologyDiagnosticReport:
        """
        [FASE 2.9] Evalúa el estado litológico, cuantifica defectos en el espacio
        de Fock emitiendo cartuchos TOON y produce `LithologyDiagnosticReport`.
        """
        C = GeomechanicalConstants
        activated_rules: List[DiagnosticRule] = []
        recommendations: List[str] = []
        cartridges: List[LithologyCartridge] = []

        if _is_peat(soil.uscs_classification):
            logger.critical(
                "Singularidad litológica detectada: uscs=%s",
                soil.uscs_classification,
            )
            raise LithologicalSingularityError(
                "La presencia de Turba (PT) induce colapso de la frontera "
                "de anclaje y exige estrategia de cimentación profunda o "
                "sustitución/mejora integral del estrato."
            )

        g_max = self._compute_dynamic_rigidity_pa(soil)
        swelling_index = self._compute_swelling_potential_index(soil)
        yielding_index = self._compute_yielding_susceptibility_index(soil)
        liquefaction_index = self._compute_liquefaction_susceptibility_index(
            soil
        )

        # Regla 1: Expansividad (SwellingPlasmonCartridge)
        if (
            soil.liquid_limit >= C.SWELLING_LIQUID_LIMIT_THRESHOLD
            and soil.plasticity_index >= C.SWELLING_PI_THRESHOLD
        ):
            activated_rules.append(
                DiagnosticRule.SWELLING_POTENTIAL_DETECTED
            )
            cartridges.append(
                SwellingPlasmonCartridge(
                    liquid_limit=soil.liquid_limit,
                    plasticity_index=soil.plasticity_index,
                    parasitic_capacitance=swelling_index,
                )
            )
            recommendations.append(
                "Verificar potencial expansivo con ensayos de hinchamiento "
                "libre y controlado.  Implementar control hídrico perimetral."
            )
            logger.warning(
                "Cartucho de expansión emitido: LL=%.1f%%, PI=%.1f%%, "
                "I_sw=%.4f",
                soil.liquid_limit,
                soil.plasticity_index,
                swelling_index,
            )

        # Regla 2: Compresibilidad / blandura (YieldingPhononCartridge)
        if (
            soil.void_ratio > C.HIGH_VOID_RATIO_THRESHOLD
            and soil.shear_wave_velocity < C.SOFT_SOIL_VS_M_S
        ):
            compression_index = self._compute_compression_index_proxy(soil)
            activated_rules.append(
                DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED
            )
            cartridges.append(
                YieldingPhononCartridge(
                    compression_index=compression_index,
                    void_ratio=soil.void_ratio,
                    viscous_drag=yielding_index,
                )
            )
            recommendations.append(
                "Evaluar consolidación y asentamientos.  Considerar precarga, "
                "mejora de suelo o cimentación adaptada."
            )
            logger.info(
                "Cartucho de cedencia emitido: e₀=%.3f, Vs=%.1f m/s, "
                "Cc_proxy=%.4f, I_y=%.4f",
                soil.void_ratio,
                soil.shear_wave_velocity,
                compression_index,
                yielding_index,
            )

        # Regla 3: Licuación (LiquefactionSolitonCartridge)
        if liquefaction_index > 0.0:
            activated_rules.append(
                DiagnosticRule.LIQUEFACTION_SUSCEPTIBILITY_DETECTED
            )
            cartridges.append(
                LiquefactionSolitonCartridge(
                    shear_wave_velocity=soil.shear_wave_velocity,
                    susceptibility_index=liquefaction_index,
                )
            )
            recommendations.append(
                "Realizar evaluación formal de licuación (Seed & Idriss o "
                "Boulanger & Idriss) con demanda sísmica y esfuerzos "
                "efectivos de campo."
            )
            logger.critical(
                "Cartucho de licuación emitido: uscs=%s, saturated=%s, "
                "Vs=%.1f m/s, I_liq=%.4f",
                soil.uscs_classification,
                soil.is_saturated,
                soil.shear_wave_velocity,
                liquefaction_index,
            )

        # Regla 4: Rigidez dinámica baja
        if g_max < C.LOW_DYNAMIC_RIGIDITY_PA:
            activated_rules.append(
                DiagnosticRule.LOW_DYNAMIC_RIGIDITY_REGIME
            )
            recommendations.append(
                f"Rigidez dinámica estimada G_max = {g_max:.0f} Pa "
                f"(< {C.LOW_DYNAMIC_RIGIDITY_PA:.0f} Pa).  Revisar "
                f"interacción suelo-estructura ante cargas dinámicas."
            )

        return LithologyDiagnosticReport(
            soil_tensor=soil,
            dynamic_rigidity_modulus_pa=g_max,
            swelling_potential_index=swelling_index,
            yielding_susceptibility_index=yielding_index,
            liquefaction_susceptibility_index=liquefaction_index,
            emitted_cartridges=tuple(cartridges),
            activated_rules=tuple(activated_rules),
            recommendations=tuple(recommendations),
        )

    def _phase2_session_hash(
        self,
        phase1_hash: str,
        diag: LithologyDiagnosticReport,
        poisson_ratio: float,
        k_eigs: Tuple[float, float, float],
    ) -> str:
        """[FASE 2.10] Genera el digest criptográfico SHA-256 determinista de Fase 2."""
        payload = {
            "phase1_hash": phase1_hash,
            "g_max_pa": round(diag.dynamic_rigidity_modulus_pa, 4),
            "i_sw": round(diag.swelling_potential_index, 8),
            "i_y": round(diag.yielding_susceptibility_index, 8),
            "i_liq": round(diag.liquefaction_susceptibility_index, 8),
            "poisson_ratio": round(poisson_ratio, 8),
            "k_eigs": [round(ev, 4) for ev in k_eigs],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def phase2_close_and_open_phase3(
        self,
        validation_certificate: SoilValidationCertificate,
        *,
        enable_rsi_level3: bool = True,
    ) -> GeomechanicalSpectralReport:
        """
        [MÉTODO TERMINAL DE FASE 2 ──► CONTINUACIÓN FORMAL EN FASE 3]

        Ejecuta la Fase 2 completa (ORIENT), ensambla el `GeomechanicalSpectralReport`
        y transiciona formalmente a la Fase 3 invocando `self.phase3_from_phase2(...)`.
        """
        cert1 = self.phase2_from_phase1(validation_certificate)
        soil = cert1.soil_tensor

        diag_report = self._evaluate_diagnostic_rules(soil)
        elasto = self.phase2_poisson_elastodynamic_tensors(
            soil, diag_report.dynamic_rigidity_modulus_pa
        )

        k_refined, k_eigs, rsi_cert = (
            self.phase2_rsi_level3_spd_impedance_refinement(
                elasto["k_matrix"],
                elasto["wilkinson_shift_k"],
                enable_rsi=enable_rsi_level3,
            )
        )

        c_mat = elasto["c_matrix"]
        c_eigs_arr = np.maximum(
            np.linalg.eigvalsh(0.5 * (c_mat + c_mat.T)),
            GeomechanicalConstants.EPSILON_MACH,
        )
        c_eigs = (
            float(c_eigs_arr[0]),
            float(c_eigs_arr[1]),
            float(c_eigs_arr[2]),
        )

        poincare_cert = self.phase2_poincare_celestial_elastodynamic_audit(
            soil=soil,
            k_matrix=k_refined,
            c_matrix=c_mat,
            liquefaction_index=diag_report.liquefaction_susceptibility_index,
        )

        k_tuple = tuple(
            tuple(float(k_refined[i, j]) for j in range(3)) for i in range(3)
        )
        c_tuple = tuple(
            tuple(float(c_mat[i, j]) for j in range(3)) for i in range(3)
        )

        phase2_hash = self._phase2_session_hash(
            cert1.phase1_session_hash,
            diag_report,
            float(elasto["poisson_ratio"]),
            k_eigs,
        )

        spectral_report = GeomechanicalSpectralReport(
            validation_certificate=cert1,
            diagnostic_report=diag_report,
            poisson_ratio=float(elasto["poisson_ratio"]),
            lame_lambda_pa=float(elasto["lame_lambda_pa"]),
            bulk_modulus_pa=float(elasto["bulk_modulus_pa"]),
            young_modulus_pa=float(elasto["young_modulus_pa"]),
            p_wave_modulus_pa=float(elasto["p_wave_modulus_pa"]),
            p_wave_velocity_m_s=float(elasto["p_wave_velocity_m_s"]),
            stiffness_tensor_k_pa=k_tuple,
            damping_tensor_c_pa_s=c_tuple,
            stiffness_eigenvalues_pa=k_eigs,
            damping_eigenvalues_pa_s=c_eigs,
            rsi_level3_certificate=rsi_cert,
            poincare_certificate=poincare_cert,
            phase2_session_hash=phase2_hash,
        )

        # Eslabón de anidamiento formal: el cierre de Fase 2 abre la Fase 3
        return self.phase3_from_phase2(spectral_report)

    # ══════════════════════════════════════════════════════════════════════════
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  FASE 3: DECIDE  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ══════════════════════════════════════════════════════════════════════════
    # Proyección del Tensor de Impedancia Compleja Z_geo(s) como Condición de
    # Frontera de Dirichlet, Auditoría Termodinámica de Terzaghi, Veto Bi-Heyting
    # Ω₃ y Conmutación en Silicio del Puerto Crowbar BT151 (GPIO14 < 400 ns).
    # ══════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(
        self, spectral_report: GeomechanicalSpectralReport
    ) -> GeomechanicalSpectralReport:
        """
        [MÉTODO INICIAL DE FASE 3 ◄── CONTINUACIÓN DESDE FASE 2]

        Verifica la validez espectral $\operatorname{SPD}(3)$ y la integridad
        criptográfica del `GeomechanicalSpectralReport` antes de proyectar
        la condición de frontera de Dirichlet y emitir el veredicto de Heyting.
        """
        if not isinstance(spectral_report, GeomechanicalSpectralReport):
            raise LithologicalInputError(
                "Se esperaba un GeomechanicalSpectralReport válido desde la Fase 2."
            )
        if min(spectral_report.stiffness_eigenvalues_pa) <= 0.0:
            raise LithologicalNumericalError(
                "Violación SPD(3): el tensor de rigidez K no es definido positivo."
            )
        if min(spectral_report.damping_eigenvalues_pa_s) <= 0.0:
            raise LithologicalNumericalError(
                "Violación SPD(3): el tensor de amortiguamiento C no es definido positivo."
            )
        return spectral_report

    def phase3_project_dirichlet_impedance_tensor(
        self,
        spectral_report: GeomechanicalSpectralReport,
        laplace_s: complex = complex(
            GeomechanicalConstants.DEFAULT_LAPLACE_SIGMA_RAD_S,
            GeomechanicalConstants.DEFAULT_LAPLACE_OMEGA_RAD_S,
        ),
    ) -> Dict[str, Any]:
        r"""
        [FASE 3.1] Proyecta el Tensor de Impedancia Geomecánica Compleja en el
        dominio de la frecuencia de Laplace $s = \sigma + j\omega$ ($\sigma \ge 0$):

            $$Z_{\mathrm{geo}}(s) = \mathbf{K} + s \cdot \mathbf{C} = (\mathbf{K} + \sigma \mathbf{C}) + j\omega \mathbf{C} \in \mathbb{C}^{3 \times 3}$$

        Incorpora las excitaciones de los cartuchos TOON (capacitancia parásita de
        hinchamiento y arrastre viscoso de cedencia) y certifica que la parte
        hermítica real $\operatorname{Re} Z_{\mathrm{geo}}(s) = \frac{1}{2}(Z_{\mathrm{geo}} + Z_{\mathrm{geo}}^H)$
        sea estrictamente definida positiva (Teorema de Pasividad Positivo-Real de Brune).
        """
        sigma = float(laplace_s.real)
        omega = float(laplace_s.imag)
        if sigma < 0.0:
            raise LithologicalInputError(
                f"La abscisa de Laplace σ = {sigma} debe ser ≥ 0 (semiplano causal derecho)."
            )

        k_mat = np.array(spectral_report.stiffness_tensor_k_pa, dtype=np.float64)
        c_mat = np.array(spectral_report.damping_tensor_c_pa_s, dtype=np.float64)

        # Modulación hadrónica por Cartuchos TOON emitidos en Fase 2
        diag = spectral_report.diagnostic_report
        swelling_degradation = 1.0 - 0.15 * diag.swelling_potential_index
        yielding_drag_boost = 1.0 + 0.25 * diag.yielding_susceptibility_index

        k_eff = swelling_degradation * k_mat
        c_eff = yielding_drag_boost * c_mat

        z_real = k_eff + sigma * c_eff
        z_imag = omega * c_eff

        hermitian_real = 0.5 * (z_real + z_real.T)
        real_eigs = np.linalg.eigvalsh(hermitian_real)
        min_real_eig = _assert_finite(
            float(real_eigs[0]), "dirichlet_real_min_eigenvalue_pa"
        )

        real_tuple = tuple(
            tuple(float(z_real[i, j]) for j in range(3)) for i in range(3)
        )
        imag_tuple = tuple(
            tuple(float(z_imag[i, j]) for j in range(3)) for i in range(3)
        )

        return {
            "laplace_s": (sigma, omega),
            "z_real": real_tuple,
            "z_imag": imag_tuple,
            "min_real_eigenvalue_pa": min_real_eig,
            "is_brune_positive_real": bool(min_real_eig > 0.0),
        }

    def phase3_audit_terzaghi_dissipation(
        self,
        spectral_report: GeomechanicalSpectralReport,
        *,
        total_stress_pa: Optional[np.ndarray] = None,
        pore_pressure_pa: Optional[float] = None,
        strain_rate_1_s: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        r"""
        [FASE 3.2] Audita el Principio de Esfuerzo Efectivo de Terzaghi y la
        Segunda Ley Termodinámica (desigualdad de Clausius-Duhem / Lyapunov):

            $$\boldsymbol{\sigma}_{\mathrm{eff}} = \boldsymbol{\sigma} - u_{\mathrm{pore}} \mathbf{I}_3, \qquad P_{\mathrm{diss}} = \operatorname{Tr}\!\left(\boldsymbol{\sigma}_{\mathrm{eff}} \cdot \dot{\boldsymbol{\varepsilon}}\right) \ge 0$$

        Si no se suministran tensores externos, evalúa el estado litostático
        canónico a profundidad de referencia $z_0 = 5.0\text{ m}$ bajo fluencia
        viscoelástica inducida por el esfuerzo efectivo.
        """
        C = GeomechanicalConstants
        soil = spectral_report.validation_certificate.soil_tensor
        nu = spectral_report.poisson_ratio
        k_0 = nu / max(1.0 - nu, 1e-4)  # Coeficiente de empuje en reposo de Jaky-Poisson
        z_ref = 5.0

        if total_stress_pa is None:
            sigma_v = soil.bulk_density_kg_m3 * C.GRAVITY_M_S2 * z_ref
            sigma_h = k_0 * sigma_v
            sigma_tot = np.diag([sigma_h, sigma_h, sigma_v]).astype(np.float64)
        else:
            sigma_tot = np.asarray(total_stress_pa, dtype=np.float64).reshape((3, 3))

        if pore_pressure_pa is None:
            if soil.is_saturated:
                # Reducción de presión efectiva por generación sísmica de poros r_u ≈ 0.85·I_liq
                r_u = 0.85 * spectral_report.diagnostic_report.liquefaction_susceptibility_index
                u_hydro = C.WATER_DENSITY_KG_M3 * C.GRAVITY_M_S2 * z_ref
                sigma_v_tot = float(sigma_tot[2, 2])
                u_pore = min(sigma_v_tot * 0.98, u_hydro + r_u * max(0.0, sigma_v_tot - u_hydro))
            else:
                u_pore = 0.0
        else:
            u_pore = float(pore_pressure_pa)

        sigma_eff = sigma_tot - u_pore * np.eye(3, dtype=np.float64)
        mean_eff_stress = float(np.trace(sigma_eff)) / 3.0

        if strain_rate_1_s is None:
            # Tasa de deformación visco-disipativa alineada positivamente con σ_eff
            c_mat = np.array(spectral_report.damping_tensor_c_pa_s, dtype=np.float64)
            c_inv = np.linalg.inv(c_mat)
            eps_dot = 0.5 * (c_inv @ sigma_eff + sigma_eff @ c_inv)
        else:
            eps_dot = np.asarray(strain_rate_1_s, dtype=np.float64).reshape((3, 3))

        # Producto interno de Frobenius compensado por KBN: Tr(σ_eff · ε̇)
        prod_entries = [
            float(sigma_eff[i, j] * eps_dot[j, i])
            for i in range(3)
            for j in range(3)
        ]
        p_diss = _assert_finite(
            _kbn_sum(prod_entries), "terzaghi_dissipation_power_w_m3"
        )

        return {
            "effective_mean_stress_pa": mean_eff_stress,
            "dissipation_power_w_m3": p_diss,
            "is_terzaghi_dissipative": bool(
                p_diss >= -C.EPSILON_MACH and mean_eff_stress >= -C.EPSILON_MACH
            ),
        }

    def phase3_evaluate_heyting_verdict(
        self,
        spectral_report: GeomechanicalSpectralReport,
        *,
        is_brune_positive_real: bool,
        is_terzaghi_dissipative: bool,
    ) -> LithologicalHeytingTruth:
        """
        [FASE 3.3] Proyecta los invariantes geomecánicos sobre el clasificador
        de subobjetos del retículo bi-Heyting $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$
        mediante conjunción intuicionista ($\wedge = \mathtt{meet}$).
        """
        C = GeomechanicalConstants
        diag = spectral_report.diagnostic_report
        poincare = spectral_report.poincare_certificate

        verdict = LithologicalHeytingTruth.COHERENT

        # 1. Criterio de pasividad de Brune y Segunda Ley de Terzaghi
        if not is_brune_positive_real or not is_terzaghi_dissipative:
            verdict = verdict.meet(LithologicalHeytingTruth.VETOED)

        # 2. Criterio de susceptibilidad a la licuación (Solitón de Licuación)
        if diag.liquefaction_susceptibility_index >= C.HEYTING_LIQUEFACTION_VETO_THRESHOLD:
            verdict = verdict.meet(LithologicalHeytingTruth.VETOED)
        elif diag.liquefaction_susceptibility_index >= C.HEYTING_LIQUEFACTION_DEGRADED_THRESHOLD:
            verdict = verdict.meet(LithologicalHeytingTruth.DEGRADED)

        # 3. Criterio de cedencia / compresibilidad (Fonón de Cedencia)
        if diag.yielding_susceptibility_index >= C.HEYTING_YIELDING_VETO_THRESHOLD:
            verdict = verdict.meet(LithologicalHeytingTruth.VETOED)
        elif DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED in diag.activated_rules:
            verdict = verdict.meet(LithologicalHeytingTruth.DEGRADED)

        # 4. Criterio de expansividad (Plasmón de Hinchamiento) y baja rigidez
        if (
            diag.swelling_potential_index >= C.HEYTING_SWELLING_DEGRADED_THRESHOLD
            or DiagnosticRule.SWELLING_POTENTIAL_DETECTED in diag.activated_rules
            or DiagnosticRule.LOW_DYNAMIC_RIGIDITY_REGIME in diag.activated_rules
        ):
            verdict = verdict.meet(LithologicalHeytingTruth.DEGRADED)

        # 5. Criterio de resonancia diofántica KAM de Poincaré
        if not poincare.is_kam_non_resonant:
            verdict = verdict.meet(LithologicalHeytingTruth.DEGRADED)

        return verdict

    def phase3_actuate_esp32_crowbar_port(
        self, heyting_verdict: LithologicalHeytingTruth
    ) -> CrowbarPortState:
        """
        [FASE 3.4] Emula con determinismo de ciclo de reloj la subrutina C++
        `isVerdictCoherent()` del ESP32. Si el veredicto colapsa a `VETOED`,
        la ISR alojada en IRAM (< 400 ns) conmuta el pin GPIO14 y enclava el
        tiristor BT151 (circuito Crowbar) para proteger la obra.
        """
        C = GeomechanicalConstants
        if heyting_verdict is LithologicalHeytingTruth.VETOED:
            return CrowbarPortState(
                gpio_pin=C.ESP32_CROWBAR_GPIO_PIN,
                bt151_thyristor_latched=True,
                is_verdict_coherent_flag=False,
                iram_isr_latency_ns=C.ESP32_IRAM_ISR_NOMINAL_NS,
                actuation_mode="BT151_CROWBAR_HARD_LATCH_VETO",
            )
        if heyting_verdict is LithologicalHeytingTruth.DEGRADED:
            return CrowbarPortState(
                gpio_pin=C.ESP32_CROWBAR_GPIO_PIN,
                bt151_thyristor_latched=False,
                is_verdict_coherent_flag=True,
                iram_isr_latency_ns=C.ESP32_IRAM_ISR_NOMINAL_NS * 0.98,
                actuation_mode="THROTTLED_DAMPING_SURVEILLANCE",
            )
        return CrowbarPortState(
            gpio_pin=C.ESP32_CROWBAR_GPIO_PIN,
            bt151_thyristor_latched=False,
            is_verdict_coherent_flag=True,
            iram_isr_latency_ns=C.ESP32_IRAM_ISR_NOMINAL_NS * 0.95,
            actuation_mode="NOMINAL_DIRICHLET_ANCHOR",
        )

    def _phase3_governance_seal(
        self,
        phase2_hash: str,
        heyting_verdict: LithologicalHeytingTruth,
        min_real_eig: float,
        p_diss: float,
        crowbar_latched: bool,
    ) -> str:
        """[FASE 3.5] Genera el sello criptográfico soberano SHA-256 de Fase 3."""
        canonical = {
            "phase2_hash": phase2_hash,
            "heyting_verdict": heyting_verdict.value,
            "min_real_eig_pa": round(min_real_eig, 4),
            "p_diss_w_m3": round(p_diss, 6),
            "crowbar_latched": crowbar_latched,
            "version": __version__,
        }
        encoded = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def phase3_close_loop(
        self,
        spectral_report: GeomechanicalSpectralReport,
        *,
        laplace_s: complex = complex(
            GeomechanicalConstants.DEFAULT_LAPLACE_SIGMA_RAD_S,
            GeomechanicalConstants.DEFAULT_LAPLACE_OMEGA_RAD_S,
        ),
        total_stress_pa: Optional[np.ndarray] = None,
        pore_pressure_pa: Optional[float] = None,
        strain_rate_1_s: Optional[np.ndarray] = None,
    ) -> LithologicalSuturationState:
        """
        [FASE 3.6 — CIERRE SOBERANO DEL CICLO DE TRES FASES ANIDADAS]

        Proyecta $Z_{\mathrm{geo}}(s)$, audita la disipación de Terzaghi,
        evalúa el retículo bi-Heyting $\Omega_3$, actualiza el CrowbarPort
        y retorna el `LithologicalSuturationState` inmutable.
        """
        report2 = self.phase3_from_phase2(spectral_report)

        dirichlet = self.phase3_project_dirichlet_impedance_tensor(
            report2, laplace_s=laplace_s
        )
        terzaghi = self.phase3_audit_terzaghi_dissipation(
            report2,
            total_stress_pa=total_stress_pa,
            pore_pressure_pa=pore_pressure_pa,
            strain_rate_1_s=strain_rate_1_s,
        )
        verdict = self.phase3_evaluate_heyting_verdict(
            report2,
            is_brune_positive_real=bool(dirichlet["is_brune_positive_real"]),
            is_terzaghi_dissipative=bool(terzaghi["is_terzaghi_dissipative"]),
        )
        crowbar = self.phase3_actuate_esp32_crowbar_port(verdict)
        seal = self._phase3_governance_seal(
            phase2_hash=report2.phase2_session_hash,
            heyting_verdict=verdict,
            min_real_eig=float(dirichlet["min_real_eigenvalue_pa"]),
            p_diss=float(terzaghi["dissipation_power_w_m3"]),
            crowbar_latched=crowbar.bt151_thyristor_latched,
        )

        return LithologicalSuturationState(
            spectral_report=report2,
            laplace_frequency_s=dirichlet["laplace_s"],
            impedance_tensor_real_pa=dirichlet["z_real"],
            impedance_tensor_imag_pa=dirichlet["z_imag"],
            dirichlet_real_min_eigenvalue_pa=float(
                dirichlet["min_real_eigenvalue_pa"]
            ),
            is_brune_positive_real=bool(dirichlet["is_brune_positive_real"]),
            terzaghi_effective_mean_stress_pa=float(
                terzaghi["effective_mean_stress_pa"]
            ),
            terzaghi_dissipation_power_w_m3=float(
                terzaghi["dissipation_power_w_m3"]
            ),
            is_terzaghi_dissipative=bool(terzaghi["is_terzaghi_dissipative"]),
            heyting_verdict=verdict,
            crowbar_port=crowbar,
            governance_seal_sha256=seal,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # §8.4  ENSAMBLADO DE SALIDA CATEGÓRICA (EJECUCIÓN DEL MORFISMO PRINCIPAL)
    # ══════════════════════════════════════════════════════════════════════════

    def __call__(
        self,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CategoricalState:
        """
        Ejecuta el morfismo litológico atravesando las 3 fases anidadas:

            Fase 1 (`phase1_close_and_open_phase2`)
              ──► Fase 2 (`phase2_close_and_open_phase3`)
                    ──► Fase 3 (`phase3_close_loop`)
                          ──► `CategoricalState` (Stratum.PHYSICS)

        Preserva 100% de compatibilidad hacia atrás con las llaves de payload
        y contexto de v5.18.0, añadiendo la telemetría doctoral de Fases 1–3.
        """
        logger.info("Iniciando evaluación de la variedad litológica (3 fases anidadas).")

        try:
            next_payload: Dict[str, Any] = dict(payload)
            next_context: Dict[str, Any] = dict(context)

            # Cadena anidada Fase 1 ──► Fase 2 ──► Fase 3
            phase1_cert = self.phase1_close_and_open_phase2(next_payload)
            phase2_report = phase1_cert.continue_to_phase2(
                self,
                enable_rsi_level3=bool(
                    next_payload.get("enable_rsi_level3", True)
                ),
            )
            suturation_state = phase2_report.continue_to_phase3(self)

            report = phase2_report.diagnostic_report

            # Registro sináptico (append inmutable, no overwrite)
            synaptic_registry: List[Any] = list(
                next_context.get("synaptic_registry", [])
            )
            synaptic_registry.extend(report.emitted_cartridges)
            next_context["synaptic_registry"] = synaptic_registry

            # Reportes serializados en el contexto categórico
            next_context["lithology_report"] = report.to_dict()
            next_context["lithology_validation_certificate"] = (
                phase1_cert.to_dict()
            )
            next_context["lithology_spectral_report"] = phase2_report.to_dict()
            next_context["lithology_suturation_state"] = (
                suturation_state.to_dict()
            )

            # Enriquecimiento del payload con magnitudes derivadas y tensores
            next_payload.update({
                "dynamic_rigidity_modulus_pa": (
                    report.dynamic_rigidity_modulus_pa
                ),
                "swelling_potential_index": report.swelling_potential_index,
                "yielding_susceptibility_index": (
                    report.yielding_susceptibility_index
                ),
                "liquefaction_susceptibility_index": (
                    report.liquefaction_susceptibility_index
                ),
                "lithology_activated_rules": [
                    r.value for r in report.activated_rules
                ],
                "poisson_ratio": phase2_report.poisson_ratio,
                "p_wave_velocity_m_s": phase2_report.p_wave_velocity_m_s,
                "lithology_heyting_verdict": (
                    suturation_state.heyting_verdict.value
                ),
                "crowbar_bt151_latched": (
                    suturation_state.crowbar_port.bt151_thyristor_latched
                ),
                "lithology_governance_seal": (
                    suturation_state.governance_seal_sha256
                ),
            })

            logger.info(
                "Evaluación litológica completada: G_max=%.0f Pa, ν=%.3f, "
                "veredicto=%s, cartuchos=%d, reglas=%s",
                report.dynamic_rigidity_modulus_pa,
                phase2_report.poisson_ratio,
                suturation_state.heyting_verdict.value,
                len(report.emitted_cartridges),
                [r.value for r in report.activated_rules],
            )

            return CategoricalState(
                payload=next_payload,
                context=next_context,
                validated_strata=frozenset({Stratum.PHYSICS}),
            )

        except (
            LithologicalInputError,
            LithologicalSingularityError,
            LithologicalNumericalError,
        ):
            raise
        except Exception as exc:
            logger.exception(
                "Falla no controlada en LithologicalManifold: %s", exc
            )
            raise LithologicalManifoldError(
                f"Falla interna del operador litológico: {exc}"
            ) from exc