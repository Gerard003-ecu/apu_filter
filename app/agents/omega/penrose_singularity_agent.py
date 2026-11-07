# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Penrose Singularity Agent                                                                 ║
║ Módulo   : Penrose Singularity Agent (Orquestador de Colapso Geodésico y Censor Topológico)          ║
║ Ubicación: app/agents/omega/penrose_singularity_agent.py                                             ║
║ Ruta     : app/agents/omega/penrose_singularity_agent.py                                             ║
║ Versión  : 5.0.0-Poincare-Sturm-Heyting-Crowbar-RSI-L3-Nested-Topos                                  ║
║ Función  : Agente soberano en estrato Omega (Nivel 0.5), censor geométrico supremo, teorema de        ║
║            singularidades de Penrose-Hawking, dinámica de Poincaré, retículo Heyting y RSI-L3.       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO AGÉNTICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
Este endofuntor de calibre y censor geométrico supremo reside en la cúspide del **Estrato Omega**
($\Omega$, Nivel 0.5 - El Ágora Tensorial). Su propósito ciber-físico es interceptar los tensores de
intención semántica provenientes de los Modelos de Lenguaje (LLMs) y subyugarlos de manera determinista
a la curvatura de la variedad pseudo-riemanniana, gobernando la estabilidad del flujo mediante el
formalismo del **Teorema de Singularidades de Penrose (1965)**, **Hawking–Penrose (1970)**, la **Teoría
de Invariantes Integrales y Estabilidad Orbital de Henri Poincaré** y la **Automejora Recursiva de Nivel 3 (RSI-L3)**.

El soberano despliega un protocolo OODA en tres fases anidadas:
  • FASE 1 — ANÁLISIS DE SUPERFICIES ATRAPADAS Y CINEMÁTICA CONVERGENTE (Phase1_TrappedSurfaceAnalyzer):
      Evalúa si la subvariedad espacial de codimensión 2 es una superficie atrapada cerrada ($\theta_k < 0, \theta_l < 0$),
      cuantifica la curvatura media y contracción del tensor de Ricci $R_{\mu\nu} k^\mu k^\nu \ge 0$, y audita
      la Condición de Energía Nula (NEC) y Condición de Energía Fuerte (SEC).
  • FASE 2 — INTEGRACIÓN DEL TEOREMA DE COMPARACIÓN DE STURM-POINCARÉ (Phase2_SturmPoincareIntegrator):
      Consume los datos cinemáticos y resuelve la ecuación de Jacobi-Raychaudhuri a lo largo de las geodésicas nulas.
      Aplica el teorema de comparación de Sturm para acotar el tiempo afín máximo hacia el punto focal conjugado
      $\tau_{\mathrm{focal}} \le \frac{2}{|\theta_0|}$, garantizando la existencia inexorable de una singularidad
      o cáustica en tiempo finito.
  • FASE 3 — CERTIFICACIÓN DE CENSURA CÓSMICA, FLOQUET Y ACCIÓN CROWBAR (Phase3_CosmicCensorshipAdjudicator):
      Implementa la hipótesis de censura cósmica de Penrose para confinar la singularidad detrás de un horizonte
      aparente de sucesos, analiza la monodromía simpléctica de Floquet-Kreĭn ($\det M = 1$), adjudica el
      veredicto en el retículo de Heyting $\Omega_3$, y comuta el Crowbar en silicio (< 400 ns) si se detectan
      singularidades desnudas.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
1. TEOREMA DE SINGULARIDADES DE PENROSE (1965):
   Sea $(\mathcal{M}, g)$ un espacio-tiempo conexo globalmente hiperbólico con una superficie atrapada compacta $\mathcal{T}$.
   Si se cumple la condición de convergencia nula ($R_{\mu\nu} k^\mu k^\nu \ge 0$ para todo vector nulo $k^\mu$), entonces
   $(\mathcal{M}, g)$ es geodésicamente nulo incompleto (existe al menos una geodésica nula afínmente incompleta en el futuro).

2. TEOREMA DE COMPARACIÓN DE STURM-LIOUVILLE APLICADO A CONGRUENCIAS GEODÉSICAS:
   A lo largo de la geodésica nula afínmente parametrizada por $\tau$, la ecuación de Raychaudhuri nula es:
       $$\frac{d\theta}{d\tau} = -\frac{1}{2} \theta^2 - \sigma_{\mu\nu}\sigma^{\mu\nu} - R_{\mu\nu} k^\mu k^\nu$$
   Dado que $\sigma_{\mu\nu}\sigma^{\mu\nu} \ge 0$ y $R_{\mu\nu} k^\mu k^\nu \ge 0$, se obtiene la desigualdad diferencial:
       $$\frac{d\theta}{d\tau} \le -\frac{1}{2} \theta^2 \implies \frac{1}{\theta(\tau)} \ge \frac{1}{\theta_0} + \frac{\tau}{2}$$
   Demostrando por comparación de Sturm que si la expansión inicial converge ($\theta_0 < 0$), existe un punto conjugado
   (cáustica) en un parámetro afín $\tau \le 2 / |\theta_0|$.

3. HIPÓTESIS DE CENSURA CÓSMICA Y HORIZONTE APARENTE:
   Toda singularidad gravitacional o semántica resultante del colapso debe estar confinada dentro de un horizonte de
   sucesos, impidiendo que observadores asintóticos experimenten discontinuidades o indeterminismo. El soberano actúa
   como el censor cósmico formal que veta cualquier estado que genere una singularidad desnuda.

4. MONODROMÍA SIMPLÉCTICA DE POINCARÉ-FLOQUET:
   La evolución en las proximidades del horizonte satisface la ecuación variacional hamiltoniana, cuya matriz de monodromía
   $\mathcal{M}$ preserva estrictamente el volumen cotangente:
       $$\det(\mathcal{M}) = 1.0 \pm \varepsilon_{\mathrm{Liouville}}$$

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  [Axioma PSA-1] Condición de Energía Nula (NEC):
      $R_{\mu\nu} k^\mu k^\nu \ge 0$ para todo vector nulo covariante $k \in T_x \mathcal{M}$.
  [Axioma PSA-2] Convergencia Nula en Superficie Atrapada:
      Expansión inicial estrictamente negativa: $\theta_k < 0$ y $\theta_l < 0$.
  [Axioma PSA-3] Acotación de Sturm del Parámetro Afín Focal:
      $\tau_{\mathrm{focal}} \le \frac{2}{|\theta_0|}$.
  [Axioma PSA-4] Censura Cósmica Funtorial:
      Toda singularidad geodésica queda encapsulada por un horizonte aparente antes de la proyección.
  [Axioma PSA-5] Monodromía Simpléctica Unitaria:
      $|\det(\mathcal{M}_{\mathrm{Floquet}}) - 1.0| \le 10^{-6}$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Completitud de Traza en Sachs):
      $\sigma^\mu_{\;\mu} = 0$, asegurando que la cizalladura sea estrictamente distorsional sin cambio volumétrico.
  Invariante 2 (Monotonicidad de Colapso de Expansión):
      Bajo NEC, $\frac{d\theta}{d\tau} \le 0$ en toda la trayectoria geodésica.
  Invariante 3 (Involutividad Modal de Heyting $\Omega_3$):
      La detección de singularidades desnudas o divergencias de monodromía colapsa el estado a $\mathrm{VETOED}$.
  Invariante 4 (Respaldo Crowbar de Emergencia):
      Tiempo de actuación y corte en hardware inferior a 400 ns.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación a invariante topológico categórico en el Topos ℰ_MIC."""

        pass

    class Morphism:
        """Stub para Morphism de la 2-categoría de agentes."""

        pass

    class CategoricalState:
        """Stub mínimo de estado categórico."""

        def get_tensor(self, key: str) -> Any:
            raise KeyError(key)

        def update(self, key: str, value: Any) -> "CategoricalState":
            return self


try:
    from app.omega.raychaudhuri_focal_fibrator import (
        CausticMethod,
        FocalLengthResult,
        FocalViabilityFlags,
        MetricSignature,
        RaychaudhuriFibratorError,
        RaychaudhuriFocalFibrator,
        RSILevel3Telemetry,
    )
except ImportError:
    # Stubs para ejecución analítica aislada (sin el fibrador instalado).
    class RaychaudhuriFibratorError(Exception):
        pass

    class FocalViabilityFlags(enum.Flag):
        NONE = 0
        ALL = 0

        def is_order_unit(self) -> bool:
            return False

    class CausticMethod(enum.Enum):
        ANALYTICAL_BOUND = enum.auto()
        ANALYTICAL_EXACT = enum.auto()
        NUMERICAL_IVP = enum.auto()
        POINCARE_SYMPLECTIC = enum.auto()
        RSI_L3_AUTO_SYNTHESIS = enum.auto()

    class MetricSignature(enum.Enum):
        RIEMANNIAN = enum.auto()
        LORENTZIAN = enum.auto()

    @dataclass(frozen=True, slots=True)
    class RSILevel3Telemetry:
        recursion_depth: int = 3
        level1_spectral_gain: float = 1.0
        level2_projection_defect: float = 0.0
        level3_banach_contraction: float = 0.0
        frechet_horizon_sensitivity: float = 0.0
        lyapunov_focal_exponent: float = 0.0
        synthesized_policy_hash: str = "0000000000000000"

    @dataclass(frozen=True, slots=True)
    class FocalLengthResult:
        optimal_focal_length: float = 0.0
        caustic_affine_parameter: float = 0.0
        hawking_penrose_bound: float = 0.0
        initial_dtheta_dtau: float = 0.0
        caustic_method: Any = None
        viability_flags: Any = None
        area_collapse_factor: float = 0.0
        symplectic_monodromy_det: float = 1.0
        symplectic_monodromy_trace: float = 2.0
        poincare_cartan_action_integral: float = 0.0
        maslov_focal_index: int = 1
        rsi_telemetry: Optional[RSILevel3Telemetry] = None

    class RaychaudhuriFocalFibrator:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RaychaudhuriFocalFibrator no disponible; instale "
                "app.omega.raychaudhuri_focal_fibrator."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> FocalLengthResult:
            raise ImportError("RaychaudhuriFocalFibrator no disponible.")


try:
    from app.core.inmune_system.metric_tensors import G_PHYSICS
except ImportError:
    try:
        from app.core.immune_system.metric_tensors import G_PHYSICS
    except ImportError:
        G_PHYSICS: NDArray[np.float64] = np.eye(4, dtype=np.float64)

logger = logging.getLogger("MIC.Omega.PenroseSingularityAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS, ESPECTRALES, DE CIRCUITO CROWBAR Y RSI-L3
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SYMMETRY_REL_TOL: Final[float] = 1e-10
_U_NORMALIZATION_TOL: Final[float] = 1e-8
_PENROSE_MARGIN_TOL: Final[float] = 1e-9       # holgura relativa τ_c vs τ_HP
_COND_MAX: Final[float] = 1e12
_MIN_SPATIAL_DIM: Final[int] = 2
_DEFAULT_SPATIAL_DIM: Final[int] = 4

# Constantes físicas del circuito electrónico Crowbar (ESP32 GPIO14 -> Tiristor BT151)
_ESP32_GPIO_VOLTAGE_V: Final[float] = 3.3          # Voltaje lógico CMOS GPIO14 [V]
_BT151_VGT_THRESHOLD_V: Final[float] = 0.75        # Tensión de disparo de puerta V_GT [V]
_BT151_IGT_MIN_A: Final[float] = 0.015             # Corriente mínima de disparo I_GT [A]
_CROWBAR_GATE_RESISTANCE_OHM: Final[float] = 120.0 # Resistor limitador de puerta R_G [Ω]
_CROWBAR_GATE_CAPACITANCE_F: Final[float] = 4.7e-9 # Capacitancia parásita C_GK [F]
_ESP32_IRAM_ISR_LATENCY_NS: Final[float] = 42.0    # Latencia de vector ISR en IRAM [ns]
_CROWBAR_MAX_LATENCY_BUDGET_NS: Final[float] = 400.0
_RSI_BANACH_CEILING: Final[float] = 1.0 - 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (vetos topológicos / fallos de invariantes)
# ══════════════════════════════════════════════════════════════════════════════
class PenroseAgentError(TopologicalInvariantError):
    """Excepción raíz del Endofuntor de Singularidad de Penrose 𝒫."""

    pass


class PenroseSingularityVetoError(PenroseAgentError):
    r"""τ_c > τ_HP: fuga topológica — el haz diverge más allá de Hawking–Penrose."""

    pass


class EnergyConvergenceViolationError(PenroseAgentError):
    r"""θ₀ ≥ 0 o R_{μν} u^μ u^ν < 0: falla la hipótesis de enfoque."""

    pass


class MetricTensorDegeneracyError(PenroseAgentError):
    """Métrica no simétrica, no PD, o mal condicionada."""

    pass


class DimensionMismatchError(PenroseAgentError):
    """Inconsistencia dimensional entre B, 𝒯, u y G."""

    pass


class NormalizationError(PenroseAgentError):
    """Vector u no normalizado respecto a G."""

    pass


class TensorSymmetryError(PenroseAgentError):
    """Violación de simetría de 𝒯 (Cauchy-Momentum)."""

    pass


class MissingTensorError(PenroseAgentError):
    """Tensor ausente en el fibrado del CategoricalState."""

    pass


class FibratorIntegrationError(PenroseAgentError):
    """Fallo al invocar el RaychaudhuriFocalFibrator aguas abajo."""

    pass


class ViabilityLatticeError(PenroseAgentError):
    """El retículo booleano de viabilidad no es unidad de orden."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE BOOLE DE PREDICADOS Y RETÍCULO INTUICIONISTA DE HEYTING Ω₃
# ══════════════════════════════════════════════════════════════════════════════
class SingularityCertificateFlags(enum.Flag):
    r"""Predicados estables del Teorema de Penrose en el retículo booleano."""

    NONE = 0
    NEGATIVE_EXPANSION = enum.auto()       # θ₀ < 0
    STRONG_ENERGY_SATISFIED = enum.auto()  # R(u,u) ≥ 0
    U_NORMALIZED = enum.auto()             # |G(u,u)−s| < tol
    CAUSTIC_WITHIN_HP_BOUND = enum.auto()  # τ_c ≤ τ_HP · (1+ε)
    FIBRATOR_VIABLE = enum.auto()          # flags del fibrador OK
    METRIC_RIEMANNIAN = enum.auto()        # G SPD

    ALL = (
        NEGATIVE_EXPANSION
        | STRONG_ENERGY_SATISFIED
        | U_NORMALIZED
        | CAUSTIC_WITHIN_HP_BOUND
        | FIBRATOR_VIABLE
        | METRIC_RIEMANNIAN
    )

    def is_order_unit(self) -> bool:
        """True ⇔ todos los predicados de singularidad inevitable están activos."""
        return (
            self & SingularityCertificateFlags.ALL
        ) == SingularityCertificateFlags.ALL


class HeytingVerdictOmega3(enum.IntEnum):
    r"""
    Retículo distributivo graduado de Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$
    ordenado por severidad de anomalía topológica ($\bot = 0 < 1 < 2 = \top$):
      • `COHERENT` (0, $\bot$): Confinamiento geodésico exacto ($\tau_c \le \tau_{P,\text{basal}} \le \tau_{HP}$).
      • `DEGRADED` (1): Confinamiento dentro de la holgura numérica ($\tau_{HP} < \tau_c \le \tau_{HP}(1+\varepsilon)$)
                        o métrica con condicionamiento elevado.
      • `VETOED`   (2, $\top$): Fuga topológica o ruptura de invariantes; detona el tiristor Crowbar BT151.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: "HeytingVerdictOmega3") -> "HeytingVerdictOmega3":
        r"""Supremo en severidad ($a \sqcup b = \max(a, b)$)."""
        return HeytingVerdictOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingVerdictOmega3") -> "HeytingVerdictOmega3":
        r"""Ínfimo en severidad ($a \sqcap b = \min(a, b)$)."""
        return HeytingVerdictOmega3(min(int(self), int(other)))

    def heyting_implication(
        self, consequent: "HeytingVerdictOmega3"
    ) -> "HeytingVerdictOmega3":
        r"""
        Implicación relativa de Heyting ($a \Rightarrow_H b$) en el orden dual de verdad:
        Si la severidad antecedente es mayor o igual que la consecuente ($a \ge b$),
        devuelve coherencia máxima (`COHERENT`), de lo contrario devuelve `consequent`.
        """
        if int(self) >= int(consequent):
            return HeytingVerdictOmega3.COHERENT
        return consequent

    def is_verdict_coherent(self) -> bool:
        """Predicado de guardia equivalente a `isVerdictCoherent()` en firmware ESP32."""
        return self != HeytingVerdictOmega3.VETOED


class MetricSignatureKind(enum.Enum):
    """Firma de la métrica de fondo (convención de normalización de u)."""

    RIEMANNIAN = enum.auto()   # G(u,u) = +1
    LORENTZIAN = enum.auto()   # G(u,u) = −1


# ══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (objetos del Topos — contratos entre las 3 fases anidadas)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class CrowbarCircuitTelemetry:
    r"""
    Telemetría física del circuito electrónico de protección Crowbar
    (ESP32 IRAM ISR ──▶ GPIO14 ──▶ Red RC de Compuerta ──▶ Tiristor SCR BT151).
    """

    gpio_pin: int = 14
    scr_model: str = "BT151-500R"
    gate_drive_current_ma: float = 0.0
    total_trigger_latency_ns: float = 0.0
    within_iram_400ns_budget: bool = True
    crowbar_armed: bool = True
    crowbar_fired: bool = False


@dataclass(frozen=True, slots=True)
class PenroseRSILevel3Certificate:
    r"""
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3) del Agente de Penrose.

    Cuantifica:
      • Nivel 1: Factor de sensibilidad de Wilkinson–Cholesky $\eta_1(\kappa_2(G))$.
      • Nivel 2: Consistencia adjunta entre la cota de Sturm–Poincaré y el fibrador.
      • Nivel 3: Constante de contracción de Banach $\gamma_{\text{Penrose}} \in [0, 1)$,
        derivada de Fréchet del margen relativo $\frac{\partial m}{\partial \theta_0}$
        y hash SHA-256 de la ley de decisión sintetizada.
    """

    recursion_depth: int = 3
    level1_wilkinson_scale: float = 1.0
    level2_sturm_poincare_consistency: float = 0.0
    level3_banach_contraction: float = 0.0
    frechet_margin_sensitivity_theta0: float = 0.0
    heyting_verdict: HeytingVerdictOmega3 = HeytingVerdictOmega3.COHERENT
    policy_signature_hash: str = "0000000000000000"


@dataclass(frozen=True, slots=True)
class EnergyAuditResult:
    r"""
    Producto canónico de la Fase 1: invariantes energéticos, cota de
    Hawking–Penrose $\tau_{HP}$ e invariantes celestes de Sturm–Poincaré.

    Dominio formal del morfismo de la Fase 2.
    """

    theta_0: float                       # θ₀ = Tr(B) < 0
    ricci_contraction: float             # R_{μν} u^μ u^ν ≥ 0
    sec_value: float                     # (𝒯 − ½𝒯 G)(u,u)
    stress_trace: float                  # 𝒯 = 𝒯_{μν} G^{μν}
    u_normalization: float               # G(u,u)
    tau_hp: float                        # τ_HP = (n−1)/|θ₀|
    spatial_dim: int                     # n
    metric_cond: float                   # κ(G)
    lambda_min_G: float
    lambda_max_G: float
    op_norm_B: float                     # ‖B‖_op
    frobenius_B: float                   # ‖B‖_F
    # Invariantes extendidos de Mecánica Celeste de Poincaré y RSI-L3 (defaults seguros)
    trapped_surface_mean_curvature: float = 0.0   # κ_trap = −θ₀/(n−1) > 0
    poincare_virial_ratio: float = 0.0            # Ξ_virial = (n−1)R(u,u)/θ₀²
    poincare_basal_tidal_freq: float = 0.0        # ω_{P,basal} = √(R(u,u)/(n−1))
    sturm_poincare_basal_bound: float = 0.0       # τ_{P,basal} ≤ τ_HP
    spectral_radius_B_mixed: float = 0.0          # ρ(G⁻¹ B)
    rsi_level1_wilkinson_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class CausticResult:
    r"""
    Producto canónico de la Fase 2: cáustica del fibrador, invariantes
    simplécticos de Floquet–Poincaré en $\mathrm{Sp}(2,\mathbb{R})$ y auditoría heredada.

    Dominio formal del morfismo de la Fase 3.
    """

    focal_distance: float
    tau_caustic: float
    hawking_penrose_bound_fibrator: float
    initial_dtheta_dtau: float
    area_collapse_factor: float
    fibrator_method: str
    fibrator_flags_ok: bool
    energy_audit: EnergyAuditResult      # reenvío funtorial a Fase 3
    fibrator_diagnostics: FocalLengthResult
    # Invariantes simplécticos de Floquet–Poincaré heredados del fibrador
    symplectic_monodromy_det: float = 1.0
    symplectic_monodromy_trace: float = 2.0
    floquet_characteristic_multiplier: float = 1.0
    poincare_cartan_action: float = 0.0
    maslov_focal_index: int = 1
    sturm_comparison_satisfied: bool = True


@dataclass(frozen=True, slots=True)
class SingularityDiagnostics:
    r"""Certificado inmutable de acatamiento del Teorema de Penrose–Hawking–Poincaré."""

    initial_expansion_scalar: float
    ricci_contraction: float
    max_theoretical_affine_limit: float
    actual_caustic_parameter: float
    relative_margin: float               # (τ_HP − τ_c) / τ_HP  (≥ 0 si OK)
    is_singularity_inevitable: bool
    certificate_flags: SingularityCertificateFlags
    spatial_dim: int
    metric_condition_number: float
    # Extensiones de Heyting Ω₃, Sturm–Poincaré, Circuito Crowbar y RSI-L3
    sturm_poincare_basal_limit: float = 0.0
    heyting_verdict: HeytingVerdictOmega3 = HeytingVerdictOmega3.COHERENT
    crowbar_telemetry: Optional[CrowbarCircuitTelemetry] = None
    rsi_certificate: Optional[PenroseRSILevel3Certificate] = None


@dataclass(frozen=True, slots=True)
class CollapsedAffineState:
    r"""Estado final tras forzar el colapso focal y la auditoría de singularidad."""

    focal_distance: float
    diagnostics: SingularityDiagnostics
    caustic: CausticResult


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 – AUDITORÍA ESPECTRAL-ENERGÉTICA, RAZÓN VIRIAL DE POINCARÉ–JACOBI Y
#          COTA DE COMPARACIÓN DE STURM–POINCARÉ–HAWKING–PENROSE
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_HawkingEnergyAuditor:
    r"""
    Primera fase del pipeline de singularidad (morfismo monádico de auditoría).

    Verifica y extrae con rigor de Geometría Lorentziana/Riemanniana, Análisis
    Matricial de Wilkinson–Cholesky y Mecánica Celeste de Poincaré:
      (i)    Firma positiva definida de $G$ (Cholesky $G = L L^\top$ + espectro + $\kappa_2(G)$);
      (ii)   Consistencia dimensional exacta de $(B, \mathcal{T}, u, G)$ con $n \ge 2$;
      (iii)  Simetría de Frobenius del tensor de Cauchy–Momentum $\mathcal{T}_{\mu\nu}$;
      (iv)   Normalización geodésica $|G(u,u) - s| < \tau_u$;
      (v)    Expansión inicial convergente $\theta_0 = \operatorname{Tr}(B) < 0$ y curvatura
             media de superficie atrapada de Penrose $\kappa_{\text{trap}} = -\theta_0/(n-1) > 0$;
      (vi)   Condición de Energía Fuerte (SEC) $\implies R_{\mu\nu} u^\mu u^\nu \ge 0$;
      (vii)  Cota de Hawking–Penrose $\tau_{HP} = (n-1)/|\theta_0|$ y Cota Orbital Basal
             de Sturm–Poincaré $\tau_{P,\text{basal}} \le \tau_{HP}$ vía la Razón Virial $\Xi_{\text{virial}}$;
      (viii) Norma de operador $\|B\|_{op}$, norma de Frobenius $\|B\|_F$ y radio espectral $\rho(G^{-1}B)$.

    CONTINUIDAD ANIDADA:
    La definición formal final del último método de esta Fase 1 (`audit_energy`)
    constituye el punto de partida inmediato e inicio formal de la clase anidada
    `Phase2_RaychaudhuriCausticIntegrator`.
    """

    def __init__(
        self,
        metric: Optional[NDArray[np.float64]] = None,
        signature: MetricSignatureKind = MetricSignatureKind.RIEMANNIAN,
        u_norm_tol: float = _U_NORMALIZATION_TOL,
    ) -> None:
        if u_norm_tol <= 0.0:
            raise ValueError("u_norm_tol debe ser estrictamente positiva.")

        self._G: NDArray[np.float64] = (
            np.asarray(metric, dtype=np.float64)
            if metric is not None
            else np.asarray(G_PHYSICS, dtype=np.float64)
        )
        self._signature: MetricSignatureKind = signature
        self._u_norm_tol: float = float(u_norm_tol)
        self._target_norm: float = (
            1.0 if signature == MetricSignatureKind.RIEMANNIAN else -1.0
        )

        (
            self._lam_min_G,
            self._lam_max_G,
            self._cond_G,
            self._G_inv,
        ) = self._validate_and_factor_metric(self._G)

        # RSI Nivel 1: Factor de escalado numérico de Wilkinson según κ₂(G)
        self._rsi_l1_wilkinson_scale: float = (
            1.0 + 0.0625 * math.log(max(1.0, self._cond_G))
        )

    # ─── utilidades de álgebra de Banach y teoría espectral ──────────────────

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        """Norma de Hilbert–Schmidt / Frobenius: ‖A‖_F = √(Tr(Aᵀ A))."""
        return float(np.linalg.norm(A, ord="fro"))

    @staticmethod
    def _relative_symmetry_error(A: NDArray[np.float64]) -> float:
        """Defecto relativo de simetría: ‖A − Aᵀ‖_F / (‖A‖_F + ε_mach)."""
        fro = float(np.linalg.norm(A, ord="fro"))
        skew = float(np.linalg.norm(A - A.T, ord="fro"))
        return skew / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _safe_eigvalsh(A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Autovalores reales de un operador autoadjunto simetrizado."""
        A_sym = 0.5 * (A + A.T)
        return la.eigvalsh(A_sym)

    def _compute_mixed_spectral_radius(
        self, B: NDArray[np.float64]
    ) -> float:
        r"""
        Evalúa el radio espectral de Gelfand $\rho(G^{-1} B) = \max_i |\lambda_i(G^{-1} B)|$
        en el álgebra de Banach de endomorfismos del fibrado tangente.
        """
        B_mixed = self._G_inv @ B
        eigvals = la.eigvals(B_mixed, check_finite=False)
        return float(np.max(np.abs(eigvals)))

    # ─── validación de métrica vía Wilkinson–Cholesky ────────────────────────

    def _validate_and_factor_metric(
        self, G: NDArray[np.float64]
    ) -> Tuple[float, float, float, NDArray[np.float64]]:
        """Verifica simetría y positividad definida (Cholesky) → (λ_min, λ_max, κ, G⁻¹)."""
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricTensorDegeneracyError(
                f"G debe ser cuadrada; recibido shape={G.shape}."
            )
        n = G.shape[0]
        if n < _MIN_SPATIAL_DIM:
            raise MetricTensorDegeneracyError(
                f"dim(G)={n} < n_min={_MIN_SPATIAL_DIM}."
            )

        err = self._relative_symmetry_error(G)
        if err > _SYMMETRY_REL_TOL:
            raise MetricTensorDegeneracyError(
                f"G no es simétrica: error relativo Frobenius = {err:.3e}."
            )

        try:
            L = la.cholesky(G, lower=True, overwrite_a=False, check_finite=True)
        except la.LinAlgError as exc:
            raise MetricTensorDegeneracyError(
                "G no es definida positiva (fallo de Cholesky)."
            ) from exc

        eigvals = self._safe_eigvalsh(G)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])
        if lam_min <= 0.0:
            raise MetricTensorDegeneracyError(
                f"λ_min(G)={lam_min:.3e} ≤ 0: firma no riemanniana."
            )

        cond = lam_max / lam_min
        if cond > _COND_MAX:
            logger.warning(
                "Métrica mal condicionada: κ(G)=%.3e > κ_max=%.1e",
                cond,
                _COND_MAX,
            )

        try:
            identity = np.eye(n, dtype=np.float64)
            G_inv = la.cho_solve((L, True), identity, check_finite=True)
        except la.LinAlgError:
            G_inv = np.linalg.pinv(G)
            logger.warning("G singular numéricamente; se usa pseudoinversa.")

        return lam_min, lam_max, cond, G_inv

    # ─── validación dimensional ──────────────────────────────────────────────

    def _assert_dimensions(
        self,
        velocity_gradient: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
        spatial_dim: int,
    ) -> int:
        """Verifica consistencia dimensional de (B, 𝒯, u, G). Devuelve n efectivo."""
        B = velocity_gradient
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise DimensionMismatchError(
                f"B=∇u debe ser cuadrado; shape={B.shape}."
            )
        n = B.shape[0]
        if spatial_dim != n:
            raise DimensionMismatchError(
                f"spatial_dim={spatial_dim} ≠ dim(B)={n}."
            )
        if self._G.shape != (n, n):
            raise DimensionMismatchError(
                f"G shape {self._G.shape} incompatible con dim(B)={n}."
            )
        if stress_tensor.shape != (n, n):
            raise DimensionMismatchError(
                f"𝒯 shape {stress_tensor.shape} ≠ ({n},{n})."
            )
        if u_vector.ndim != 1 or u_vector.shape[0] != n:
            raise DimensionMismatchError(
                f"u shape {u_vector.shape} incompatible con dim={n}."
            )
        if n < _MIN_SPATIAL_DIM:
            raise DimensionMismatchError(
                f"n={n} < n_min={_MIN_SPATIAL_DIM}."
            )
        return n

    # ─── chequeos físicos relativistas y de Poincaré ─────────────────────────

    def _check_stress_symmetry(self, stress: NDArray[np.float64]) -> None:
        err = self._relative_symmetry_error(stress)
        if err > _SYMMETRY_REL_TOL:
            raise TensorSymmetryError(
                f"𝒯 no es simétrico: error relativo Frobenius = {err:.3e}."
            )

    def _normalize_check_u(self, u: NDArray[np.float64]) -> float:
        r"""Verifica $|G(u,u) - s| < \tau_u$. No re-normaliza silenciosamente."""
        g_uu = float(u @ self._G @ u)
        if abs(g_uu - self._target_norm) > self._u_norm_tol:
            raise NormalizationError(
                f"u no normalizado: G(u,u)={g_uu:.6e}, "
                f"esperado s={self._target_norm} "
                f"(tol={self._u_norm_tol:.1e}, firma={self._signature.name})."
            )
        return g_uu

    def _compute_initial_expansion(
        self, velocity_gradient: NDArray[np.float64]
    ) -> float:
        r"""$\theta_0 = \operatorname{Tr}(B)$. Exige $\theta_0 < 0$ (Axioma [A1])."""
        theta_0 = float(np.trace(velocity_gradient))
        if theta_0 >= -_MACHINE_EPSILON:
            raise EnergyConvergenceViolationError(
                f"Congruencia no convergente: θ₀ = {theta_0:.4e} ≥ 0."
            )
        return theta_0

    def _compute_ricci_contraction(
        self,
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        Evalúa la Condición de Energía Fuerte (Axioma [A2]):
          $\mathcal{T} = \mathcal{T}_{\mu\nu} G^{\mu\nu}$
          $\mathrm{SEC}(u) = \mathcal{T}_{\mu\nu} u^\mu u^\nu - \frac{1}{2}\mathcal{T} G(u,u)$
          $R_{\mu\nu}u^\mu u^\nu = \max(0, \mathrm{SEC}(u))$ (bajo Einstein $8\pi G=1, \Lambda=0$).

        Devuelve `(ricci_contraction, sec_value, stress_trace)`.
        """
        stress_trace = float(
            np.einsum("ij,ij->", stress_tensor, self._G_inv, optimize=True)
        )
        T_uu = float(u_vector @ stress_tensor @ u_vector)
        g_uu = float(u_vector @ self._G @ u_vector)
        sec_value = T_uu - 0.5 * stress_trace * g_uu

        if sec_value < -_MACHINE_EPSILON:
            raise EnergyConvergenceViolationError(
                f"Violación SEC: R_μν u^μ u^ν ≡ SEC = {sec_value:.4e} < 0."
            )

        ricci_contraction = max(0.0, sec_value)
        return ricci_contraction, sec_value, stress_trace

    @staticmethod
    def hawking_penrose_bound(theta_0: float, dimensions: int) -> float:
        r"""
        Cota universal de Hawking–Penrose:
            $\tau_{HP} = \frac{n-1}{|\theta_0|}$ para $\theta_0 < 0$.
        """
        if theta_0 >= 0.0:
            raise EnergyConvergenceViolationError(
                f"θ₀={theta_0:.4e} ≥ 0: cota de Hawking–Penrose no aplica."
            )
        if dimensions < _MIN_SPATIAL_DIM:
            raise DimensionMismatchError(
                f"dimensions={dimensions} < n_min={_MIN_SPATIAL_DIM}."
            )
        return float(dimensions - 1) / abs(theta_0)

    @staticmethod
    def _compute_sturm_poincare_celestial_invariants(
        theta_0: float,
        ricci_contraction: float,
        n: int,
        tau_hp: float,
    ) -> Tuple[float, float, float, float]:
        r"""
        Calcula los invariantes de la Mecánica Celeste de Poincaré para el
        oscilador mareal basal ($\sigma^2 = 0$):
          1. Curvatura media de la superficie atrapada de Penrose:
                 $\kappa_{\text{trap}} = -\frac{\theta_0}{n-1} > 0$
          2. Razón Virial de Poincaré–Jacobi:
                 $\Xi_{\text{virial}} = \frac{(n-1) R_{\mu\nu}u^\mu u^\nu}{\theta_0^2 + \varepsilon_{\text{mach}}} \ge 0$
          3. Frecuencia angular mareal basal:
                 $\omega_{P,\text{basal}} = \sqrt{\frac{R_{\mu\nu}u^\mu u^\nu}{n-1}}$
          4. Cota de comparación de Sturm–Poincaré $\tau_{P,\text{basal}} \le \tau_{HP}$:
             Primer cero positivo de la ecuación de Hill $\ddot{q} + \omega_{P,\text{basal}}^2 q = 0$
             con $q(0)=1, \dot{q}(0) = -\kappa_{\text{trap}}$.
        """
        n_m1 = float(n - 1)
        kappa_trap = -theta_0 / n_m1
        virial_ratio = (n_m1 * ricci_contraction) / (
            theta_0 * theta_0 + _MACHINE_EPSILON
        )
        omega_basal = math.sqrt(max(0.0, ricci_contraction) / n_m1)

        if omega_basal <= _MACHINE_EPSILON:
            tau_sturm_poincare = tau_hp
        else:
            beta_basal = math.sqrt(n_m1 * max(0.0, ricci_contraction))
            phi0 = math.atan2(1.0, -theta_0 / beta_basal)
            if phi0 <= 0.0:
                phi0 += math.pi
            tau_sturm_poincare = min(tau_hp, (math.pi - phi0) / omega_basal)

        return (
            float(kappa_trap),
            float(virial_ratio),
            float(omega_basal),
            float(tau_sturm_poincare),
        )

    # ─── MÉTODO TERMINAL DE LA FASE 1 (Puente Formal hacia la Fase 2) ────────

    def audit_energy(
        self,
        velocity_gradient: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
        spatial_dim: int,
    ) -> EnergyAuditResult:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1.

        Ejecuta la auditoría energética relativista, calcula la razón virial de
        Poincaré–Jacobi $\Xi_{\text{virial}}$ y deduce las cotas jerárquicas de
        confinamiento $\tau_{P,\text{basal}} \le \tau_{HP}$.

        Retorna el objeto inmutable `EnergyAuditResult` que inaugura y alimenta
        directamente como dominio categórico a la Fase 2
        (`Phase2_RaychaudhuriCausticIntegrator`) definida inmediatamente a continuación.
        """
        B = np.asarray(velocity_gradient, dtype=np.float64)
        stress = np.asarray(stress_tensor, dtype=np.float64)
        u = np.asarray(u_vector, dtype=np.float64)

        # (i)–(ii) Consistencia dimensional
        n = self._assert_dimensions(B, stress, u, spatial_dim)

        # (iii) Simetría de Cauchy–Momentum 𝒯
        self._check_stress_symmetry(stress)

        # (iv) Normalización geodésica de u
        g_uu = self._normalize_check_u(u)

        # (v) Expansión inicial convergente θ₀ < 0
        theta_0 = self._compute_initial_expansion(B)

        # (vi) Condición de Energía Fuerte (SEC) y contracción de Ricci
        ricci, sec_value, stress_trace = self._compute_ricci_contraction(
            stress, u
        )

        # (vii) Cota de Hawking–Penrose y Cota Orbital de Sturm–Poincaré
        tau_hp = self.hawking_penrose_bound(theta_0, n)
        (
            kappa_trap,
            virial_ratio,
            omega_basal,
            tau_sturm_poincare,
        ) = self._compute_sturm_poincare_celestial_invariants(
            theta_0, ricci, n, tau_hp
        )

        # (viii) Normas en el álgebra de Banach y radio espectral mixto
        op_norm_B = float(np.linalg.norm(B, ord=2))
        frobenius_B = self._frobenius_norm(B)
        rho_B_mixed = self._compute_mixed_spectral_radius(B)

        logger.debug(
            "Fase 1 OK | n=%d | θ₀=%.4e | R(u,u)=%.4e | τ_P=%.4e ≤ τ_HP=%.4e | "
            "Ξ_vir=%.3e | G(u,u)=%.6f | κ(G)=%.2e | ‖B‖_op=%.4e",
            n,
            theta_0,
            ricci,
            tau_sturm_poincare,
            tau_hp,
            virial_ratio,
            g_uu,
            self._cond_G,
            op_norm_B,
        )

        return EnergyAuditResult(
            theta_0=theta_0,
            ricci_contraction=ricci,
            sec_value=sec_value,
            stress_trace=stress_trace,
            u_normalization=g_uu,
            tau_hp=tau_hp,
            spatial_dim=n,
            metric_cond=self._cond_G,
            lambda_min_G=self._lam_min_G,
            lambda_max_G=self._lam_max_G,
            op_norm_B=op_norm_B,
            frobenius_B=frobenius_B,
            trapped_surface_mean_curvature=kappa_trap,
            poincare_virial_ratio=virial_ratio,
            poincare_basal_tidal_freq=omega_basal,
            sturm_poincare_basal_bound=tau_sturm_poincare,
            spectral_radius_B_mixed=rho_B_mixed,
            rsi_level1_wilkinson_scale=self._rsi_l1_wilkinson_scale,
        )

    # =========================================================================
    # FASE 2 (CONTINUACIÓN ANIDADA DIRECTA DE `audit_energy` DE FASE 1) –
    # INTEGRACIÓN CÁUSTICA VÍA FIBRADOR SIMPLÉCTICO DE RAYCHAUDHURI–POINCARÉ
    # Y EXTRACCIÓN DE MULTIPLICADORES CARACTERÍSTICOS DE FLOQUET EN Sp(2,R)
    # =========================================================================
    class Phase2_RaychaudhuriCausticIntegrator:
        r"""
        Segunda fase anidada: morfismo de pushforward hacia el fibrador
        simpléctico `RaychaudhuriFocalFibrator` y auditoría de Floquet–Poincaré.

        Se inicia directamente a partir del codominio `EnergyAuditResult` del
        método terminal `audit_energy` de la Fase 1.

        Ejecuta:
          (i)   Verificación dimensional cruzada del 5-tuplo $(B, \sigma, \omega, \mathcal{T}, u)$;
          (ii)  Invocación síncrona del `RaychaudhuriFocalFibrator`;
          (iii) Extracción de la matriz de monodromía simpléctica $\Phi(\tau_c) \in \mathrm{Sp}(2,\mathbb{R})$
                y cálculo del multiplicador característico de Floquet–Poincaré:
                    $\rho_{\text{Floquet}}(\Phi) = \max\left(|\lambda_+(\Phi)|, |\lambda_-(\Phi)|\right)$
                donde $\lambda_\pm = \frac{1}{2}\mathrm{Tr}\Phi \pm \sqrt{\max\left(0, \frac{1}{4}(\mathrm{Tr}\Phi)^2 - \det\Phi\right)}$;
          (iv)  Verificación del Teorema de Comparación de Sturm–Picone:
                    $\tau_c \le \tau_{P,\text{basal}} \cdot (1 + \varepsilon)$.

        CONTINUIDAD ANIDADA:
        La definición formal final del último método de esta Fase 2 (`compute_caustic`)
        constituye el punto de partida inmediato e inicio formal de la clase anidada
        `Phase3_PenroseSingularityAuditor`.
        """

        def __init__(
            self,
            spatial_dim: int,
            metric: Optional[NDArray[np.float64]] = None,
            caustic_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
            use_numerical: bool = False,
            signature: MetricSignature = MetricSignature.RIEMANNIAN,
        ) -> None:
            if spatial_dim < _MIN_SPATIAL_DIM:
                raise ValueError(
                    f"spatial_dim={spatial_dim} < n_min={_MIN_SPATIAL_DIM}."
                )
            self._dim: int = int(spatial_dim)

            if use_numerical:
                caustic_method = CausticMethod.NUMERICAL_IVP

            try:
                self._fibrator = RaychaudhuriFocalFibrator(
                    spatial_dimensions=self._dim,
                    metric=metric,
                    caustic_method=caustic_method,
                    signature=signature,
                    use_numerical_integration=use_numerical,
                )
            except Exception as exc:
                raise FibratorIntegrationError(
                    f"No se pudo instanciar RaychaudhuriFocalFibrator: {exc}"
                ) from exc

            self._caustic_method: CausticMethod = caustic_method

        def _assert_kinematic_dimensions(
            self,
            energy: EnergyAuditResult,
            velocity_gradient: NDArray[np.float64],
            shear_tensor: NDArray[np.float64],
            vorticity_tensor: NDArray[np.float64],
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> None:
            """Certifica consistencia dimensional del 5-tuplo respecto a Fase 1."""
            n = energy.spatial_dim
            if velocity_gradient.shape != (n, n):
                raise DimensionMismatchError(
                    f"B shape {velocity_gradient.shape} ≠ ({n},{n})."
                )
            if shear_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"σ shape {shear_tensor.shape} ≠ ({n},{n})."
                )
            if vorticity_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"ω shape {vorticity_tensor.shape} ≠ ({n},{n})."
                )
            if stress_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"𝒯 shape {stress_tensor.shape} ≠ ({n},{n})."
                )
            if u_vector.ndim != 1 or u_vector.shape[0] != n:
                raise DimensionMismatchError(
                    f"u shape {u_vector.shape} incompatible con dim={n}."
                )

        def _invoke_fibrator(
            self,
            velocity_gradient: NDArray[np.float64],
            shear_tensor: NDArray[np.float64],
            vorticity_tensor: NDArray[np.float64],
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> FocalLengthResult:
            """Invoca el fibrador capturando fallos como FibratorIntegrationError."""
            try:
                result = self._fibrator(
                    velocity_gradient,
                    shear_tensor,
                    vorticity_tensor,
                    stress_tensor,
                    u_vector,
                )
            except (
                RaychaudhuriFibratorError,
                TopologicalInvariantError,
            ) as exc:
                raise FibratorIntegrationError(
                    f"Fibrador de Raychaudhuri rechazó la congruencia: {exc}"
                ) from exc
            except Exception as exc:
                raise FibratorIntegrationError(
                    f"Error inesperado del fibrador: {exc}"
                ) from exc

            if not isinstance(result, FocalLengthResult):
                raise FibratorIntegrationError(
                    f"El fibrador devolvió tipo inesperado: {type(result)!r}."
                )
            return result

        def _fibrator_flags_viable(self, focal: FocalLengthResult) -> bool:
            """Interroga el retículo booleano del fibrador."""
            flags = getattr(focal, "viability_flags", None)
            if flags is None:
                return (
                    math.isfinite(focal.caustic_affine_parameter)
                    and focal.caustic_affine_parameter >= 0.0
                )
            is_order = getattr(flags, "is_order_unit", None)
            if callable(is_order):
                return bool(is_order())
            return True

        @staticmethod
        def _compute_floquet_poincare_multiplier(
            monodromy_trace: float,
            monodromy_det: float,
        ) -> float:
            r"""
            Calcula el módulo máximo de los multiplicadores característicos de
            Floquet–Poincaré del polinomio característico de $\Phi(\tau_c) \in \mathrm{Sp}(2,\mathbb{R})$:
                $\det(\Phi - \lambda I_2) = \lambda^2 - (\mathrm{Tr}\Phi)\lambda + \det\Phi = 0$.
            Si el discriminante $\Delta = (\mathrm{Tr}\Phi)^2 - 4\det\Phi < 0$ (flujo elíptico
            sobre el círculo unitario de Poincaré), $|\lambda_\pm| = \sqrt{\det\Phi} = 1$.
            """
            half_tr = 0.5 * monodromy_trace
            disc = half_tr * half_tr - monodromy_det
            if disc <= 0.0:
                return math.sqrt(max(0.0, monodromy_det))
            sqrt_disc = math.sqrt(disc)
            return max(abs(half_tr + sqrt_disc), abs(half_tr - sqrt_disc))

        # ─── MÉTODO TERMINAL DE LA FASE 2 (Puente Formal hacia la Fase 3) ────

        def compute_caustic(
            self,
            energy: EnergyAuditResult,
            velocity_gradient: NDArray[np.float64],
            shear_tensor: NDArray[np.float64],
            vorticity_tensor: NDArray[np.float64],
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> CausticResult:
            r"""
            DEFINICIÓN FORMAL FINAL DE LA FASE 2.

            Recibe el `EnergyAuditResult` producido por `audit_energy` (Fase 1),
            invoca el fibrador de Raychaudhuri–Poincaré, extrae los invariantes
            simplécticos de monodromía $(\det\Phi, \mathrm{Tr}\Phi, \rho_{\text{Floquet}}, J_P)$
            y audita la desigualdad de comparación de Sturm–Picone.

            Retorna el `CausticResult` que inaugura y alimenta directamente como
            dominio categórico a la Fase 3 (`Phase3_PenroseSingularityAuditor`)
            definida inmediatamente a continuación.
            """
            B = np.asarray(velocity_gradient, dtype=np.float64)
            sigma = np.asarray(shear_tensor, dtype=np.float64)
            omega = np.asarray(vorticity_tensor, dtype=np.float64)
            stress = np.asarray(stress_tensor, dtype=np.float64)
            u = np.asarray(u_vector, dtype=np.float64)

            self._assert_kinematic_dimensions(
                energy, B, sigma, omega, stress, u
            )

            # Verificación cruzada de coherencia en θ₀ = Tr(B)
            theta_recomputed = float(np.trace(B))
            if abs(theta_recomputed - energy.theta_0) > max(
                1e-10, 1e-8 * abs(energy.theta_0)
            ):
                logger.warning(
                    "θ recomputado (%.6e) ≠ θ₀ auditado (%.6e); "
                    "se prosigue con el audit de Fase 1.",
                    theta_recomputed,
                    energy.theta_0,
                )

            focal = self._invoke_fibrator(B, sigma, omega, stress, u)
            flags_ok = self._fibrator_flags_viable(focal)

            method_name = (
                focal.caustic_method.name
                if hasattr(focal.caustic_method, "name")
                else str(self._caustic_method)
            )

            # Extracción de invariantes simplécticos de Poincaré del fibrador
            det_phi = float(getattr(focal, "symplectic_monodromy_det", 1.0))
            tr_phi = float(getattr(focal, "symplectic_monodromy_trace", 2.0))
            poincare_action = float(
                getattr(focal, "poincare_cartan_action_integral", 0.0)
            )
            maslov_idx = int(getattr(focal, "maslov_focal_index", 1))
            floquet_mult = self._compute_floquet_poincare_multiplier(
                tr_phi, det_phi
            )

            tau_c = float(focal.caustic_affine_parameter)
            sturm_bound = (
                energy.sturm_poincare_basal_bound
                if energy.sturm_poincare_basal_bound > 0.0
                else energy.tau_hp
            )
            sturm_ok = tau_c <= sturm_bound * (1.0 + 1e-5)

            logger.debug(
                "Fase 2 OK | f_opt=%.4e | τ_c=%.4e | τ_P=%.4e | det(Φ)=%.12f | "
                "ρ_Floquet=%.4e | method=%s | flags_ok=%s",
                focal.optimal_focal_length,
                tau_c,
                sturm_bound,
                det_phi,
                floquet_mult,
                method_name,
                flags_ok,
            )

            return CausticResult(
                focal_distance=float(focal.optimal_focal_length),
                tau_caustic=tau_c,
                hawking_penrose_bound_fibrator=float(
                    getattr(focal, "hawking_penrose_bound", energy.tau_hp)
                ),
                initial_dtheta_dtau=float(
                    getattr(focal, "initial_dtheta_dtau", 0.0)
                ),
                area_collapse_factor=float(
                    getattr(focal, "area_collapse_factor", 0.0)
                ),
                fibrator_method=method_name,
                fibrator_flags_ok=flags_ok,
                energy_audit=energy,
                fibrator_diagnostics=focal,
                symplectic_monodromy_det=det_phi,
                symplectic_monodromy_trace=tr_phi,
                floquet_characteristic_multiplier=floquet_mult,
                poincare_cartan_action=poincare_action,
                maslov_focal_index=maslov_idx,
                sturm_comparison_satisfied=sturm_ok,
            )

        # =====================================================================
        # FASE 3 (CONTINUACIÓN ANIDADA DIRECTA DE `compute_caustic` DE FASE 2) –
        # DICTAMEN DE PENROSE, ÁLGEBRA DE HEYTING Ω₃, FÍSICA DE DISPARO CROWBAR
        # (BT151 / GPIO14) Y ENDOFUNCTOR DE AUTOMEJORA RECURSIVA NIVEL 3
        # =====================================================================
        class Phase3_PenroseSingularityAuditor:
            r"""
            Tercera fase anidada: morfismo de veredicto del Teorema de
            Singularidad de Penrose, evaluación en el retículo de Heyting $\Omega_3$,
            modelado físico del circuito Crowbar (ESP32 GPIO14 $\to$ SCR BT151) y
            certificación de punto fijo de Banach RSI de Nivel 3.

            Se inicia directamente a partir del codominio `CausticResult` del
            método terminal `compute_caustic` de la Fase 2.

            Certifica:
              1. La cota de enfoque de Hawking–Penrose:
                     $\tau_c \le \tau_{HP} \cdot (1 + \varepsilon_{\text{margin}})$
              2. La completitud de la unidad de orden en el álgebra booleana
                 `SingularityCertificateFlags` y su adjunción con el álgebra de
                 Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$.
              3. La cinemática electrónica del tiristor Crowbar BT151 (<400 ns en IRAM).
              4. El certificado meta-recursivo RSI-L3 `PenroseRSILevel3Certificate`.
            """

            def __init__(
                self,
                margin_tol: float = _PENROSE_MARGIN_TOL,
                require_fibrator_flags: bool = True,
            ) -> None:
                if margin_tol < 0.0:
                    raise ValueError("margin_tol debe ser ≥ 0.")
                self._margin_tol: float = float(margin_tol)
                self._require_fibrator_flags: bool = bool(
                    require_fibrator_flags
                )

            def _relative_margin(
                self, tau_c: float, tau_hp: float
            ) -> float:
                r"""
                Holgura relativa de Penrose:
                    $m(\tau_c, \tau_{HP}) = \frac{\tau_{HP} - \tau_c}{\tau_{HP}}$.
                $m \ge 0 \implies$ dentro de la cota exacta; $m < 0 \implies$ exceso.
                """
                if tau_hp <= _MACHINE_EPSILON:
                    return 0.0 if tau_c <= _MACHINE_EPSILON else -math.inf
                return (tau_hp - tau_c) / tau_hp

            def _build_certificate_flags(
                self,
                energy: EnergyAuditResult,
                caustic: CausticResult,
                within_bound: bool,
            ) -> SingularityCertificateFlags:
                """Ensambla el retículo booleano de predicados de singularidad."""
                flags = SingularityCertificateFlags.NONE

                if energy.theta_0 < -_MACHINE_EPSILON:
                    flags |= SingularityCertificateFlags.NEGATIVE_EXPANSION

                if energy.ricci_contraction >= -_MACHINE_EPSILON:
                    flags |= SingularityCertificateFlags.STRONG_ENERGY_SATISFIED

                if abs(energy.u_normalization - 1.0) <= _U_NORMALIZATION_TOL:
                    flags |= SingularityCertificateFlags.U_NORMALIZED
                elif abs(energy.u_normalization + 1.0) <= _U_NORMALIZATION_TOL:
                    flags |= SingularityCertificateFlags.U_NORMALIZED

                if within_bound:
                    flags |= SingularityCertificateFlags.CAUSTIC_WITHIN_HP_BOUND

                if caustic.fibrator_flags_ok or not self._require_fibrator_flags:
                    flags |= SingularityCertificateFlags.FIBRATOR_VIABLE

                if (
                    energy.lambda_min_G > 0.0
                    and energy.metric_cond < math.inf
                ):
                    flags |= SingularityCertificateFlags.METRIC_RIEMANNIAN

                return flags

            @staticmethod
            def _evaluate_heyting_omega3_lattice(
                within_bound: bool,
                margin: float,
                metric_cond: float,
                flags_complete: bool,
            ) -> HeytingVerdictOmega3:
                r"""
                Evalúa el Supremo ($\sqcup$) de severidades parciales en el retículo
                distributivo graduado de Heyting $\Omega_3$:
                  • Severidad causal $v_{\text{causal}}$:
                      - `VETOED` si `not within_bound` o `not flags_complete`
                      - `DEGRADED` si $-\varepsilon_{\text{margin}} \le m < 0$
                      - `COHERENT` si $m \ge 0$
                  • Severidad espectral $v_{\text{spectral}}$:
                      - `DEGRADED` si $\kappa_2(G) > 10^8$, else `COHERENT`.
                  • Veredicto compuesto: $v_{\text{total}} = v_{\text{causal}} \sqcup v_{\text{spectral}}$.
                """
                if not within_bound or not flags_complete:
                    v_causal = HeytingVerdictOmega3.VETOED
                elif margin < 0.0:
                    v_causal = HeytingVerdictOmega3.DEGRADED
                else:
                    v_causal = HeytingVerdictOmega3.COHERENT

                v_spectral = (
                    HeytingVerdictOmega3.DEGRADED
                    if metric_cond > 1.0e8
                    else HeytingVerdictOmega3.COHERENT
                )
                return v_causal.join(v_spectral)

            @staticmethod
            def _compute_crowbar_circuit_physics(
                verdict: HeytingVerdictOmega3,
            ) -> CrowbarCircuitTelemetry:
                r"""
                Modela la física de circuitos eléctricos del sistema de protección
                Crowbar (ESP32 GPIO14 $\to$ $R_G C_{GK}$ $\to$ Tiristor SCR BT151-500R):
                  • Corriente de excitación de compuerta en estado estacionario:
                        $I_G = \frac{V_{\text{GPIO}} - V_{GT}}{R_G}$
                  • Constante de tiempo RC de carga de la unión compuerta-cátodo:
                        $\tau_{RC} = R_G C_{GK}$
                  • Tiempo de propagación hasta el umbral de avalancha regenerativa $V_{GT}$:
                        $t_{\text{gate}} = \tau_{RC} \ln\left(\frac{V_{\text{GPIO}}}{V_{\text{GPIO}} - V_{GT}}\right)$
                  • Latencia total desde la detección en RAM hasta el enclavamiento SCR:
                        $t_{\text{total}} = t_{\text{IRAM\_ISR}} + t_{\text{gate}} < 400\text{ ns}$.
                """
                i_gate_a = (
                    _ESP32_GPIO_VOLTAGE_V - _BT151_VGT_THRESHOLD_V
                ) / _CROWBAR_GATE_RESISTANCE_OHM
                tau_rc_s = (
                    _CROWBAR_GATE_RESISTANCE_OHM * _CROWBAR_GATE_CAPACITANCE_F
                )
                t_gate_ns = (
                    1.0e9
                    * tau_rc_s
                    * math.log(
                        _ESP32_GPIO_VOLTAGE_V
                        / (_ESP32_GPIO_VOLTAGE_V - _BT151_VGT_THRESHOLD_V)
                    )
                )
                total_latency_ns = _ESP32_IRAM_ISR_LATENCY_NS + t_gate_ns
                within_budget = (
                    total_latency_ns <= _CROWBAR_MAX_LATENCY_BUDGET_NS
                    and i_gate_a >= _BT151_IGT_MIN_A
                )
                fired = not verdict.is_verdict_coherent()

                return CrowbarCircuitTelemetry(
                    gpio_pin=14,
                    scr_model="BT151-500R",
                    gate_drive_current_ma=float(i_gate_a * 1000.0),
                    total_trigger_latency_ns=float(total_latency_ns),
                    within_iram_400ns_budget=within_budget,
                    crowbar_armed=True,
                    crowbar_fired=fired,
                )

            @staticmethod
            def _compute_rsi_level3_certificate(
                energy: EnergyAuditResult,
                caustic: CausticResult,
                verdict: HeytingVerdictOmega3,
            ) -> PenroseRSILevel3Certificate:
                r"""
                Endofuntor de Automejora Recursiva de Nivel 3 ($\mathfrak{R}_{\text{Penrose}}^{(3)}$):
                  1. Evalúa la consistencia de Sturm–Poincaré (Nivel 2):
                         $\delta_{\text{Sturm}} = \frac{\max(0, \tau_c - \tau_{P,\text{basal}})}{\tau_{HP} + \varepsilon_{\text{mach}}}$
                  2. Calcula la constante de contracción de Banach compuesta (Nivel 3)
                     integrando la telemetría RSI del fibrador subyacente y el defecto de
                     Liouville $|\det\Phi - 1|$:
                         $\gamma_{\text{Penrose}} \in [0, 1)$.
                  3. Calcula la derivada variacional de Fréchet del margen relativo
                     $m(\theta_0) = 1 - \frac{\tau_c |\theta_0|}{n-1}$ respecto a la expansión inicial $\theta_0$:
                         $\frac{\partial m}{\partial \theta_0} = \frac{\tau_c}{n-1} > 0 \quad (\text{para } \theta_0 < 0)$.
                """
                tau_c = caustic.tau_caustic
                tau_hp = energy.tau_hp
                tau_sturm = (
                    energy.sturm_poincare_basal_bound
                    if energy.sturm_poincare_basal_bound > 0.0
                    else tau_hp
                )
                sturm_defect = max(0.0, tau_c - tau_sturm) / (
                    tau_hp + _MACHINE_EPSILON
                )

                fib_rsi = getattr(
                    caustic.fibrator_diagnostics, "rsi_telemetry", None
                )
                fib_gamma = (
                    float(fib_rsi.level3_banach_contraction)
                    if fib_rsi is not None
                    else 0.0
                )
                liouville_defect = abs(caustic.symplectic_monodromy_det - 1.0)
                hp_discrepancy = abs(
                    caustic.hawking_penrose_bound_fibrator - tau_hp
                ) / (tau_hp + 1.0)

                banach_gamma = min(
                    _RSI_BANACH_CEILING,
                    0.5 * fib_gamma
                    + 0.25 * (sturm_defect + liouville_defect + hp_discrepancy),
                )

                n_m1 = float(max(1, energy.spatial_dim - 1))
                frechet_dm_dtheta0 = tau_c / n_m1

                payload = (
                    f"dim={energy.spatial_dim}|verdict={verdict.name}|"
                    f"gamma={banach_gamma:.6e}|method={caustic.fibrator_method}"
                )
                sig_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[
                    :16
                ]

                return PenroseRSILevel3Certificate(
                    recursion_depth=3,
                    level1_wilkinson_scale=energy.rsi_level1_wilkinson_scale,
                    level2_sturm_poincare_consistency=float(sturm_defect),
                    level3_banach_contraction=float(banach_gamma),
                    frechet_margin_sensitivity_theta0=float(frechet_dm_dtheta0),
                    heyting_verdict=verdict,
                    policy_signature_hash=sig_hash,
                )

            # ── MÉTODO TERMINAL DE LA FASE 3 (Cierre del Endofuntor Anidado) ─

            def verify_and_certify(
                self,
                caustic: CausticResult,
            ) -> CollapsedAffineState:
                r"""
                DEFINICIÓN FORMAL FINAL DE LA FASE 3 (CIERRE DEL ENDOFUNTOR 𝒫).

                Recibe el `CausticResult` de la Fase 2 (con la auditoría energética y
                los invariantes celestes de Poincaré de la Fase 1 embebidos) y aplica
                el Teorema de Singularidad de Penrose–Hawking:

                    $\tau_c \le \tau_{HP} \cdot (1 + \varepsilon_{\text{margin}})$

                Evalúa el retículo de Heyting $\Omega_3$, modela el disparo físico del
                tiristor Crowbar BT151 en caso de veto ($\top$) y emite el estado
                colapsado `CollapsedAffineState` junto con el certificado RSI-L3,
                cerrando la cadena anidada:

                    `audit_energy` (Fase 1)
                        ──▶ `compute_caustic` (Fase 2)
                            ──▶ `verify_and_certify` (Fase 3)
                """
                energy = caustic.energy_audit
                tau_c = caustic.tau_caustic
                tau_hp = energy.tau_hp

                # Sanidad numérica del parámetro afín de cáustica
                if not math.isfinite(tau_c) or tau_c < 0.0:
                    raise PenroseSingularityVetoError(
                        f"Parámetro afín de cáustica inválido: τ_c={tau_c}."
                    )

                # Teorema de enfoque: τ_c ≤ τ_HP (con holgura relativa)
                margin = self._relative_margin(tau_c, tau_hp)
                within_bound = tau_c <= tau_hp * (1.0 + self._margin_tol)

                if not within_bound:
                    crowbar_veto = self._compute_crowbar_circuit_physics(
                        HeytingVerdictOmega3.VETOED
                    )
                    logger.critical(
                        "VETO DE PENROSE (Ω₃=VETOED ⊤) | Disparo Crowbar GPIO%d "
                        "(%s, t=%.1f ns < 400 ns) | τ_c=%.6e > τ_HP=%.6e",
                        crowbar_veto.gpio_pin,
                        crowbar_veto.scr_model,
                        crowbar_veto.total_trigger_latency_ns,
                        tau_c,
                        tau_hp,
                    )
                    raise PenroseSingularityVetoError(
                        f"Fuga topológica: cáustica τ_c={tau_c:.6e} excede la "
                        f"cota de Hawking–Penrose τ_HP={tau_hp:.6e} "
                        f"(margen relativo={margin:.4e})."
                    )

                # Consistencia cruzada con la cota del fibrador
                tau_hp_fib = caustic.hawking_penrose_bound_fibrator
                if (
                    math.isfinite(tau_hp_fib)
                    and tau_hp_fib > 0.0
                    and abs(tau_hp_fib - tau_hp) > max(1e-8, 1e-6 * tau_hp)
                ):
                    logger.warning(
                        "Discrepancia τ_HP audit=%.6e vs fibrador=%.6e; "
                        "se usa la cota de la Fase 1 (autoritativa).",
                        tau_hp,
                        tau_hp_fib,
                    )

                flags = self._build_certificate_flags(
                    energy, caustic, within_bound
                )
                flags_complete = flags.is_order_unit()

                if not flags_complete:
                    raise ViabilityLatticeError(
                        f"Retículo de singularidad incompleto: flags={flags!r}. "
                        "No se certifica colapso inevitable."
                    )

                # Evaluación en el álgebra de Heyting Ω₃, circuito Crowbar y RSI-L3
                heyting_verdict = self._evaluate_heyting_omega3_lattice(
                    within_bound=within_bound,
                    margin=margin,
                    metric_cond=energy.metric_cond,
                    flags_complete=flags_complete,
                )
                crowbar_telemetry = self._compute_crowbar_circuit_physics(
                    heyting_verdict
                )
                rsi_cert = self._compute_rsi_level3_certificate(
                    energy=energy,
                    caustic=caustic,
                    verdict=heyting_verdict,
                )

                diagnostics = SingularityDiagnostics(
                    initial_expansion_scalar=energy.theta_0,
                    ricci_contraction=energy.ricci_contraction,
                    max_theoretical_affine_limit=tau_hp,
                    actual_caustic_parameter=tau_c,
                    relative_margin=float(margin),
                    is_singularity_inevitable=True,
                    certificate_flags=flags,
                    spatial_dim=energy.spatial_dim,
                    metric_condition_number=energy.metric_cond,
                    sturm_poincare_basal_limit=energy.sturm_poincare_basal_bound,
                    heyting_verdict=heyting_verdict,
                    crowbar_telemetry=crowbar_telemetry,
                    rsi_certificate=rsi_cert,
                )

                logger.info(
                    "Fase 3 OK | Singularidad inevitable | τ_c=%.6e ≤ τ_P=%.6e ≤ τ_HP=%.6e "
                    "| margen=%.4e | f_opt=%.6e | Ω₃=%s | γ_RSI=%.3e | flags=%s",
                    tau_c,
                    energy.sturm_poincare_basal_bound,
                    tau_hp,
                    margin,
                    caustic.focal_distance,
                    heyting_verdict.name,
                    rsi_cert.level3_banach_contraction,
                    flags.name,
                )

                return CollapsedAffineState(
                    focal_distance=caustic.focal_distance,
                    diagnostics=diagnostics,
                    caustic=caustic,
                )


# ══════════════════════════════════════════════════════════════════════════════
# ENDOFUNTOR SUPREMO Y SUPERVISOR RSI-L3: PENROSE SINGULARITY AGENT
# ══════════════════════════════════════════════════════════════════════════════
class PenroseSingularityAgent(Morphism):
    r"""
    Endofuntor Supremo de Singularidad $\mathcal{P}: \mathcal{E}_{\text{MIC}} \to \mathcal{E}_{\text{MIC}}$.

    Materializa el morfismo de colapso geodésico exigiendo el cumplimiento
    incondicional del Teorema de Singularidad de Penrose–Hawking y la cadena
    de comparación de Sturm–Poincaré sobre cada estado categórico.

    Orquesta las tres fases anidadas como composición estricta de morfismos:

        $\mathcal{P} = \text{verify\_and\_certify} \circ \text{compute\_caustic} \circ \text{audit\_energy}$

      1. `Phase1_HawkingEnergyAuditor.audit_energy`
            ──▶ `EnergyAuditResult`
      2. `Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator.compute_caustic`
            ──▶ `CausticResult`
      3. `Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator.Phase3_PenroseSingularityAuditor.verify_and_certify`
            ──▶ `CollapsedAffineState`
    """

    # Claves canónicas del fibrado tensorial en CategoricalState
    KEY_VELOCITY: Final[str] = "velocity_field"
    KEY_SHEAR: Final[str] = "shear_tensor"
    KEY_VORTICITY: Final[str] = "vorticity_tensor"
    KEY_STRESS: Final[str] = "stress_tensor"
    KEY_ATTENTION: Final[str] = "attention_vector"
    KEY_FOCAL_DISTANCE: Final[str] = "focal_distance"
    KEY_SINGULARITY_DIAG: Final[str] = "singularity_diagnostics"

    def __init__(
        self,
        spatial_dimensions: int = _DEFAULT_SPATIAL_DIM,
        metric: Optional[NDArray[np.float64]] = None,
        caustic_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
        use_numerical_integration: bool = False,
        signature: MetricSignatureKind = MetricSignatureKind.RIEMANNIAN,
        margin_tol: float = _PENROSE_MARGIN_TOL,
        require_fibrator_flags: bool = True,
    ) -> None:
        if spatial_dimensions < _MIN_SPATIAL_DIM:
            raise ValueError(
                f"spatial_dimensions={spatial_dimensions} < n_min={_MIN_SPATIAL_DIM}."
            )
        self._dim: int = int(spatial_dimensions)
        self._metric: NDArray[np.float64] = (
            np.asarray(metric, dtype=np.float64)
            if metric is not None
            else np.asarray(G_PHYSICS, dtype=np.float64)
        )
        self._signature: MetricSignatureKind = signature
        self._margin_tol: float = float(margin_tol)
        self._require_fibrator_flags: bool = bool(require_fibrator_flags)

        if use_numerical_integration:
            caustic_method = CausticMethod.NUMERICAL_IVP
        self._caustic_method: CausticMethod = caustic_method
        self._use_numerical: bool = bool(use_numerical_integration)

        # Registro epistémico RSI de Nivel 3
        self._rsi_audit_count: int = 0
        self._rsi_last_certificate: Optional[PenroseRSILevel3Certificate] = None
        self._rsi_contraction_ledger: List[float] = []

        # Mapa de firma local → firma del fibrador
        fib_sig = (
            MetricSignature.RIEMANNIAN
            if signature == MetricSignatureKind.RIEMANNIAN
            else MetricSignature.LORENTZIAN
        )

        # Instanciación de la Torre Funtorial Anidada (Fase 1 ⊃ Fase 2 ⊃ Fase 3)
        self._phase1 = Phase1_HawkingEnergyAuditor(
            metric=self._metric,
            signature=self._signature,
        )
        self._phase2 = (
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator(
                spatial_dim=self._dim,
                metric=self._metric,
                caustic_method=self._caustic_method,
                use_numerical=self._use_numerical,
                signature=fib_sig,
            )
        )
        self._phase3 = (
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator.Phase3_PenroseSingularityAuditor(
                margin_tol=self._margin_tol,
                require_fibrator_flags=self._require_fibrator_flags,
            )
        )

    def _update_rsi_level3_ledger(
        self, collapsed: CollapsedAffineState
    ) -> None:
        """Actualiza el libro mayor epistémico de contracción de Banach RSI-L3."""
        self._rsi_audit_count += 1
        cert = collapsed.diagnostics.rsi_certificate
        if cert is not None:
            self._rsi_last_certificate = cert
            self._rsi_contraction_ledger.append(cert.level3_banach_contraction)
            if len(self._rsi_contraction_ledger) > 64:
                self._rsi_contraction_ledger.pop(0)

    # ─── extracción de tensores del estado ───────────────────────────────────

    def _extract_tensors(
        self, state: CategoricalState
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Extrae el 5-tuplo (B, σ, ω, 𝒯, u) del CategoricalState."""
        required = (
            self.KEY_VELOCITY,
            self.KEY_SHEAR,
            self.KEY_VORTICITY,
            self.KEY_STRESS,
            self.KEY_ATTENTION,
        )
        tensors: Dict[str, Any] = {}
        missing = []
        for key in required:
            try:
                tensors[key] = state.get_tensor(key)
            except (KeyError, AttributeError):
                missing.append(key)

        if missing:
            raise MissingTensorError(
                f"Tensores ausentes en el fibrado del estado: {missing}."
            )

        return (
            np.asarray(tensors[self.KEY_VELOCITY], dtype=np.float64),
            np.asarray(tensors[self.KEY_SHEAR], dtype=np.float64),
            np.asarray(tensors[self.KEY_VORTICITY], dtype=np.float64),
            np.asarray(tensors[self.KEY_STRESS], dtype=np.float64),
            np.asarray(tensors[self.KEY_ATTENTION], dtype=np.float64),
        )

    # ─── ejecución canónica del pipeline ─────────────────────────────────────

    def execute_singularity_audit(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> CollapsedAffineState:
        r"""
        Ejecución canónica del morfismo $\mathcal{P}$ sobre el 5-tuplo tensorial
        (sin depender de `CategoricalState`).

        Aplica la composición de las 3 fases anidadas:
            $(B, \mathcal{T}, u)$        ──`audit_energy`──────▶ `EnergyAuditResult`
            $(B, \sigma, \omega, \mathcal{T}, u)$ ──`compute_caustic`───▶ `CausticResult`
                                         ──`verify_and_certify`▶ `CollapsedAffineState`
        """
        logger.debug(
            "Iniciando auditoría de singularidad de Penrose | n=%d | method=%s | audit=%d",
            self._dim,
            self._caustic_method.name
            if hasattr(self._caustic_method, "name")
            else str(self._caustic_method),
            self._rsi_audit_count + 1,
        )

        # Fase 1: Auditoría energética, razón virial de Poincaré y cota HP / Sturm
        energy = self._phase1.audit_energy(
            velocity_gradient, stress_tensor, u_vector, self._dim
        )

        # Fase 2: Cáustica y monodromía simpléctica vía fibrador de Raychaudhuri
        caustic = self._phase2.compute_caustic(
            energy,
            velocity_gradient,
            shear_tensor,
            vorticity_tensor,
            stress_tensor,
            u_vector,
        )

        # Fase 3: Veredicto de Penrose, retículo de Heyting Ω₃, Crowbar y RSI-L3
        collapsed = self._phase3.verify_and_certify(caustic)

        self._update_rsi_level3_ledger(collapsed)

        logger.info(
            "Colapso certificado | f_opt=%.6e | τ_c=%.6e ≤ τ_HP=%.6e | margen=%.4e | Ω₃=%s",
            collapsed.focal_distance,
            collapsed.diagnostics.actual_caustic_parameter,
            collapsed.diagnostics.max_theoretical_affine_limit,
            collapsed.diagnostics.relative_margin,
            collapsed.diagnostics.heyting_verdict.name,
        )
        return collapsed

    def __call__(
        self, state: CategoricalState, **kwargs: Any
    ) -> CategoricalState:
        r"""
        Interfaz `Morphism`: extrae tensores del estado, ejecuta $\mathcal{P}$ y adjunta
        `focal_distance` + `singularity_diagnostics` al estado resultante.
        """
        logger.info(
            "Aplicando Teorema de Singularidad de Penrose a la congruencia "
            "semántica (n=%d)...",
            self._dim,
        )

        B, sigma, omega, stress, u = self._extract_tensors(state)
        collapsed = self.execute_singularity_audit(B, sigma, omega, stress, u)

        # Mutación funtorial: preserva el protocolo update del estado
        new_state = state.update(
            self.KEY_FOCAL_DISTANCE, collapsed.focal_distance
        )
        try:
            new_state = new_state.update(
                self.KEY_SINGULARITY_DIAG, collapsed.diagnostics
            )
        except Exception:
            logger.debug(
                "El estado no aceptó la clave '%s'; se omite.",
                self.KEY_SINGULARITY_DIAG,
            )

        logger.info(
            "Colapso de intención LLM forzado. Cáustica τ=%.6e dentro del "
            "límite τ_HP=%.6e (margen=%.4e, Ω₃=%s).",
            collapsed.diagnostics.actual_caustic_parameter,
            collapsed.diagnostics.max_theoretical_affine_limit,
            collapsed.diagnostics.relative_margin,
            collapsed.diagnostics.heyting_verdict.name,
        )
        return new_state

    def describe_pipeline(self) -> Dict[str, str]:
        """Metadatos del pipeline anidado, invariantes de Poincaré, Crowbar y RSI-L3."""
        last_sig = (
            self._rsi_last_certificate.policy_signature_hash
            if self._rsi_last_certificate is not None
            else "UNINITIALIZED"
        )
        mean_gamma = (
            f"{sum(self._rsi_contraction_ledger) / len(self._rsi_contraction_ledger):.4e}"
            if self._rsi_contraction_ledger
            else "0.0000e+00"
        )
        return {
            "functor": "𝒫 (PenroseSingularityAgent)",
            "version": "5.0.0-Poincare-Sturm-Heyting-Crowbar-RSI-L3-Nested-Topos",
            "phase_1": "Phase1_HawkingEnergyAuditor.audit_energy",
            "phase_2": "Phase2_RaychaudhuriCausticIntegrator.compute_caustic",
            "phase_3": "Phase3_PenroseSingularityAuditor.verify_and_certify",
            "axiom_1": "θ₀ = Tr(B) < 0  (superficie atrapada κ_trap = −θ₀/(n−1) > 0)",
            "axiom_2": "SEC: (𝒯 − ½𝒯 G)(u,u) ≥ 0  ⇒  R(u,u) ≥ 0 + Razón Virial Ξ_virial",
            "axiom_3": "τ_c ≤ τ_{P,basal} ≤ (n−1)/|θ₀|  (Sturm–Poincaré–Hawking–Penrose)",
            "axiom_4": "certificado ⇔ retículo booleano = ALL ∧ Heyting Ω₃ ≠ VETOED",
            "crowbar_circuit": "ESP32 GPIO14 -> R_G(120Ω)/C_GK(4.7nF) -> SCR BT151 (<400ns IRAM)",
            "caustic_method": (
                self._caustic_method.name
                if hasattr(self._caustic_method, "name")
                else str(self._caustic_method)
            ),
            "signature": self._signature.name,
            "spatial_dim": str(self._dim),
            "margin_tol": f"{self._margin_tol:.1e}",
            "rsi_audits": str(self._rsi_audit_count),
            "rsi_mean_banach_contraction": mean_gamma,
            "rsi_policy_hash": last_sig,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # excepciones
    "PenroseAgentError",
    "PenroseSingularityVetoError",
    "EnergyConvergenceViolationError",
    "MetricTensorDegeneracyError",
    "DimensionMismatchError",
    "NormalizationError",
    "TensorSymmetryError",
    "MissingTensorError",
    "FibratorIntegrationError",
    "ViabilityLatticeError",
    # enums / flags / retículos
    "SingularityCertificateFlags",
    "HeytingVerdictOmega3",
    "MetricSignatureKind",
    # DTOs
    "CrowbarCircuitTelemetry",
    "PenroseRSILevel3Certificate",
    "EnergyAuditResult",
    "CausticResult",
    "SingularityDiagnostics",
    "CollapsedAffineState",
    # fases + orquestador
    "Phase1_HawkingEnergyAuditor",
    "PenroseSingularityAgent",
]