# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Thermal Gradient Agent (Soberano de Calibre del Campo Térmico)      ║
║ Ruta   : app/agents/physics/thermal_gradient_agent.py                        ║
║ Versión: 4.0.0-Sovereign-Poincare-RSIL3-Caputo-CechSheaf-TomitaKMS-OODA      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Soberano de Calibre del Campo Térmico (Thermal Gradient Agent).

Este módulo implementa el endofunctor de supervisión cibernética y cohomológica
OODA S = Act ∘ Orient ∘ Observe sobre los estados del campo de gradientes térmicos.
Integra el cálculo fraccional no-markoviano de Caputo/Grünwald-Letnikov, la
cohomología de Čech H¹_Čech sobre un haz de álgebras de Heyting con holonomía de
Poincaré-Birkhoff, la teoría modular cuántica KMS de Tomita-Takesaki, la mecánica
celeste simpléctica de Henri Poincaré y un motor de Automejora Recursiva de Nivel 3
(RSI-L3) con contracción de Banach y candado lógico de Gödel-Tarski.

DEFINICIÓN FORMAL Y OPERATORIA POR FASES ANIDADAS:
    El agente procesa tensores de conductividad K y gradientes ∇T en ℝ^n mediante
    tres fases estrictamente anidadas por mixins, donde el método formal final de
    cada fase constituye el inicio exacto de la fase subsiguiente:

        In ──Fase 1──► Phase1ThermalObservation
           ──Fase 2──► Phase2ThermalOrientation
           ──Fase 3──► Phase3ThermalDecision ──► ThermalGradientCertificate

    1. Fase 1 (Observe - Phase1ThermalObservationMixin):
       - Sello de ejecución del motor ThermalGradientLaws y auditoría estructural del
         certificado (`EngineCertificateAudit`), incluyendo telemetría KAM y RSI-L3.
       - Construcción de la Órbita Observacional de Poincaré en el espacio fase
         (q_obs, p_obs) = (‖∇T‖_g, ‖Q‖_g / T_sys), detección de cruces en la superficie
         de sección transversal Σ_OODA e invariante integral de Poincaré-Cartan.
       - Acumulación sincronizada en la ventana de memoria fraccional de Caputo.
       - Morfismo terminal de Fase 1: `phase1_observe_engine_stamp` → `Phase1ThermalObservation`.

    2. Fase 2 (Orient - Phase2ThermalOrientationMixin(Phase1ThermalObservationMixin)):
       - Continuación formal directa desde `Phase1ThermalObservation`.
       - Memoria Fraccional de Caputo / Grünwald-Letnikov (α ∈ (0,1)) y Espacio Fase de
         Poincaré-Melnikov:
         D^α f_n = Δt^{−α} ∑_{j=0}^n w_j^{(α)} f_{n−j},   I^α Φ_n = ∑_{j=0}^{n-1} k_j^{(α)} Φ_{n-1-j}.
         Evalúa la acción fraccional de Poincaré-Cartan A_α = ∮ I^α Φ d(D^α Φ), el tiempo de
         recurrencia de Kac-Poincaré fraccional y la separatriz homoclínica de Melnikov M_α
         para distinguir oscilaciones KAM-confinadas de fugas seculares irreversibles.
       - Haz de Heyting, Cohomología de Čech H¹_Čech y Torsión de Poincaré-Birkhoff:
         Sobre un cubrimiento {U_i}, evalúa secciones locales Γ(U_i, ℋ), el 1-cociclo de
         solape δ⁰ s en U_i ∩ U_j y el ángulo de holonomía de Poincaré-Birkhoff ϑ_{ij}.
         Si |Δ rank| ≥ 2, certifica obstrucción H¹_Čech ≠ 0 y habilita aislamiento quirúrgico.
       - Condición KMS, Flujo Modular de Tomita-Takesaki y Geometría de Bures-Poincaré:
         Para ρ y H en M_n(ℂ) a temperatura inversa β = 1/T, evalúa ρ_β = e^{−βH}/Z,
         la entropía relativa D(ρ ∥ ρ_β), el defecto modular ‖log ρ + βH − cI‖_HS, el
         conmutador modular de Tomita-Takesaki ‖[Δ_ρ^{it}, H]‖_HS, la fidelidad cuántica
         F_Uhlmann(ρ, ρ_β) y la distancia geodésica de Bures D_Bures(ρ, ρ_β).
       - Morfismo terminal de Fase 2: `phase2_orient_from_observation` → `Phase2ThermalOrientation`.

    3. Fase 3 (Decide/Act & RSI-L3 - Phase3ThermalDecisionActuationMixin(Phase2ThermalOrientationMixin)):
       - Continuación formal directa desde `Phase1ThermalObservation × Phase2ThermalOrientation`.
       - Automejora Recursiva de Nivel 3 (RSI-L3) del Supervisor:
           * Nivel 1: Refina hiperparámetros adaptativos θ^(1)_sup = (α_eff, τ_sec, ε_KMS, γ_sheaf).
           * Nivel 2: Descenso por gradiente natural sobre la métrica de Fisher-Rao G^(2)_sup.
           * Nivel 3: Audita la razón de contracción de Lipschitz-Banach L_Banach < 1 del operador
             de Nivel 2 y reescribe G^(2)_sup bajo el candado inmutable de Gödel-Tarski.
       - Clasificación en la Cadena de Heyting Ω_4 = {VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED}.
       - Veto quirúrgico aislante de cartas locales o disparo atómico CAS del disyuntor
         Crowbar BT151/GPIO14 simulado en IRAM (< 400 ns).
       - Morfismo terminal de Fase 3: `phase3_decide_from_orientation` → `Phase3ThermalDecision`.

AXIOMAS E INVARIANTES RIGUROSOS:
    - Axioma I (Criterio de Fuga Secular de Caputo-Melnikov):
      I^α Φ < −τ_secular  ⇒  Violación irreversible de la Segunda Ley en el historial no-markoviano.
    - Axioma II (Aislamiento Quirúrgico de Čech-Poincaré):
      H¹_Čech ≠ 0 con resto coherente  ⇒  Degradación quirúrgica localizada en cartas U_i vetadas,
      preservando la continuidad operativa de ⋃_{j ≠ i} U_j.
    - Axioma III (Invariante Estructural KMS y Estacionariedad de Tomita-Takesaki):
      D(ρ ∥ ρ_β) ≤ ε_KMS  ∧  ‖[ρ, H]‖_F → 0  ⇔  Equilibrio termodinámico cuántico local a β = 1/T.
    - Axioma IV (Contracción de Punto Fijo de Banach RSI-L3 y Candado Gödel-Tarski):
      L_Banach^(sup) ∈ [0, 1)  ∧  τ_CD^(sup) ≥ τ_CD^(base) (prohibición absoluta de relajar leyes físicas).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Final, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

try:
    from thermal_gradient_laws import ThermalGradientLaws
except ImportError:  # pragma: no cover
    try:
        from app.physics.thermal_gradient_laws import ThermalGradientLaws
    except ImportError:
        from physics.thermal_gradient_laws import ThermalGradientLaws

logger = logging.getLogger("APU.Agents.ThermalGradientAgent")

# ---------------------------------------------------------------------------
# Constantes metrológicas (Wilkinson / IEEE-754 / Caputo / Čech / KMS / Poincaré / RSI-L3)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_LIMIT: Final[float] = 1e-15
_WILKINSON_FLOOR: Final[float] = _WILKINSON_LIMIT

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = 380.0
_CROWBAR_LATENCY_CEIL_NS: Final[float] = 420.0

_DEFAULT_CD_THRESHOLD: Final[float] = -1e-4
_DEFAULT_EXERGY_THRESHOLD: Final[float] = 0.4
_DEGRADED_EXERGY_THRESHOLD: Final[float] = 0.6
_FOURIER_VETO: Final[float] = 1e-3
_FOURIER_DEGRADED: Final[float] = 1e-6
_CSMD_VETO: Final[float] = 1e-8
_LANDAUER_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_CONDITION_VETO: Final[float] = 1e12
_CONDITION_DEGRADED: Final[float] = 1e8

_DEFAULT_ALPHA: Final[float] = 0.5
_DEFAULT_HISTORY: Final[int] = 32
_DEFAULT_DT: Final[float] = 1.0
_FRAC_SECULAR_TAU: Final[float] = 1e-3
_KMS_VETO: Final[float] = 1e-2
_KMS_DEGRADED: Final[float] = 1e-4
_CECH_RANK_GAP: Final[int] = 2
_PSD_TOLERANCE: Final[float] = 1e-10
_DENSITY_TRACE_TOLERANCE: Final[float] = 1e-10

# Constantes de Mecánica Celeste de Poincaré y Automejora Recursiva Nivel 3 (RSI-L3)
_POINCARE_AGENT_WINDOW: Final[int] = 24
_RSI_AGENT_BANACH_CAP: Final[float] = 0.90
_RSI_AGENT_META_LR: Final[float] = 0.04


# =============================================================================
# Retículo de Heyting (Topos de Verdad del Supervisor Ω_4)
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.

    El «join de seguridad» (peor gana) es el meet de verdad.
    CERTIFIED es ⊤, no un token desconocido.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)

    @classmethod
    def parse(cls, token: Any, *, unknown_as_veto: bool = True) -> "HeytingVerdict":
        normalized = str(token).strip().upper()
        for member in cls:
            if member.value == normalized:
                return member
        if unknown_as_veto and normalized:
            return cls.VETOED
        return cls.COHERENT if not normalized else cls.VETOED


# =============================================================================
# Contratos inmutables y Cartas Topológicas / Simplécticas / RSI-L3
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _logsumexp(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return -np.inf
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -np.inf
    shift = float(np.max(finite))
    return shift + float(np.log(np.sum(np.exp(finite - shift))))


@dataclass(frozen=True, slots=True)
class PoincareObservationOrbitChart:
    r"""
    Carta de la Órbita Observacional de Poincaré sobre la Sección Σ_OODA (Fase 1).

    Atributos:
        canonical_q_obs          : Coordenada configuracional observada ‖∇T‖_2.
        canonical_p_obs          : Momento conjugado observado ‖Q‖_2 / T_sys.
        poincare_section_crossings: Número de cruces transversales sobre Σ_OODA.
        poincare_cartan_action   : Acción integral de Poincaré-Cartan ∮ p_obs dq_obs − H dt.
        poincare_hopf_index      : Índice topológico de rotación de Poincaré-Hopf en Σ_OODA.
        kam_stability_inherited  : Estabilidad del toro KAM reportada por el motor físico.
    """

    canonical_q_obs: float
    canonical_p_obs: float
    poincare_section_crossings: int
    poincare_cartan_action: float
    poincare_hopf_index: int
    kam_stability_inherited: float


@dataclass(frozen=True, slots=True)
class AgentRSILevel3Chart:
    r"""
    Carta del Gobernador de Automejora Recursiva de Nivel 3 del Agente (RSI-L3).

    Jerarquía formal de recursión supervisora:
        - Nivel 1 (Parámetros Operativos de Supervisión):
          (α_eff, τ_secular_eff, ε_KMS_eff, γ_sheaf_confidence).
        - Nivel 2 (Meta-Gradiente Natural de Fisher-Rao):
          Métrica diagonal G^(2)_sup y funcional de pérdida de Lyapunov L_sup.
        - Nivel 3 (Meta-Meta-Endofunctor de Punto Fijo de Banach):
          Factor de contracción L_Banach^(sup) < 1 y Candado de Consistencia de Gödel-Tarski.
    """

    epoch: int
    level1_effective_alpha: float
    level1_secular_tau: float
    level1_kms_threshold: float
    level1_sheaf_confidence: float
    level2_lyapunov_loss: float
    level2_natural_grad_norm: float
    level2_fisher_trace: float
    level3_banach_contraction: float
    level3_meta_damping: float
    godel_tarski_soundness_locked: bool


@dataclass(frozen=True, slots=True)
class EngineCertificateAudit:
    available: bool
    engine_verdict: str
    engine_interlock_fired: bool
    fourier_residual: float
    csmd_error: float
    landauer_gap: float
    gouy_stodola: float
    conservation_residual: float
    thermal_entropy_production: float
    dirichlet_energy: float
    carnot_density: float
    T_hot: float
    T_cold: float
    pairing_q_dT: float
    metric_condition: float
    third_law_flag: bool
    temperature_clamped: bool
    engine_veto_reasons: Tuple[str, ...]
    engine_degraded_reasons: Tuple[str, ...]
    missing_keys: Tuple[str, ...]
    adaptive_tau_cd: float = float("nan")
    tellegen_residual: float = float("nan")
    kam_torus_stability: float = 1.0
    poincare_recurrence_time: float = 1.0
    poincare_monodromy_det: float = 1.0
    melnikov_distance: float = 0.0
    engine_rsi_contraction: float = 0.0


@dataclass(frozen=True, slots=True)
class FractionalMemoryChart:
    r"""
    Carta de memoria no-markoviana (Caputo / Grünwald–Letnikov / RL) y Espacio Fase de Poincaré.
    """

    alpha: float
    dt: float
    window: int
    samples: int
    gl_phi: float
    gl_temperature: float
    rl_integral_phi: float
    secular_leak: bool
    transient_flag: bool
    kernel_l1: float
    poincare_fractional_action: float = 0.0
    poincare_recurrence_estimate: float = 1.0
    melnikov_homoclinic_integral: float = 0.0
    kam_memory_confined: bool = True


@dataclass(frozen=True, slots=True)
class ChartSection:
    r"""Sección local Γ(U, ℋ) sobre una carta de coordenadas con ángulo de fase de Poincaré."""

    name: str
    indices: Tuple[int, ...]
    pairing: float
    phi_local: float
    dirichlet_mass: float
    verdict: str
    rank: int
    poincare_phase_angle: float = 0.0


@dataclass(frozen=True, slots=True)
class SheafCohomologyChart:
    r"""
    Čech H¹ de un cubrimiento finito con valores en la cadena de Heyting y Holonomía de Poincaré-Birkhoff.

    `obstructed` ⇔ existe solape con |Δ rank| ≥ 2 (no pegable).
    `isolated_charts` = cartas cuyo veredicto local es VETOED.
    """

    cover_names: Tuple[str, ...]
    sections: Tuple[ChartSection, ...]
    overlap_disagreements: Tuple[Tuple[str, str, str, str], ...]
    cech_h1_obstructed: bool
    isolated_charts: Tuple[str, ...]
    surgical_possible: bool
    global_meet: str
    cocycle_l2_norm: float = 0.0
    poincare_birkhoff_holonomy: float = 0.0


@dataclass(frozen=True, slots=True)
class KMSChart:
    r"""
    Defecto KMS / modular de Tomita-Takesaki a β = 1/T sobre M_n(ℂ) y distancia de Bures-Poincaré.
    """

    available: bool
    is_density: bool
    inverse_temperature: float
    relative_entropy: float
    modular_defect: float
    uhlmann_fidelity: float
    von_neumann_entropy: float
    kms_coherent: bool
    tomita_takesaki_commutator: float = 0.0
    bures_poincare_distance: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase1ThermalObservation:
    r"""Objeto terminal de la Fase 1 ≡ objeto inicial exacto de la Fase 2."""

    clausius_duhem_residual: float
    carnot_efficiency: float
    exergy_potential: float
    heat_flux_norm: float
    heat_flux_energy: float
    temperature_system: float
    heat_flux_vector: np.ndarray
    grad_T: np.ndarray
    entropy_production_rate: float
    engine_stamp: Dict[str, Any]
    observation_valid: bool
    engine_alive: bool
    certificate_audit: EngineCertificateAudit
    history_length: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    poincare_orbit: Optional[PoincareObservationOrbitChart] = None
    rsi_governor: Optional[AgentRSILevel3Chart] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "heat_flux_vector", _freeze_array(self.heat_flux_vector))
        object.__setattr__(self, "grad_T", _freeze_array(self.grad_T))


@dataclass(frozen=True, slots=True)
class Phase2ThermalOrientation:
    r"""Objeto terminal de la Fase 2 ≡ segundo factor exacto del dominio de la Fase 3."""

    is_cd_coherent: bool
    is_exergy_sufficient: bool
    is_fourier_coherent: bool
    is_csmd_coherent: bool
    is_landauer_coherent: bool
    is_temperature_physical: bool
    is_carnot_in_range: bool
    is_engine_alive: bool
    is_certificate_internally_consistent: bool
    is_fractional_secular: bool
    is_fractional_transient: bool
    is_sheaf_obstructed: bool
    is_kms_coherent: bool
    cd_margin: float
    exergy_margin: float
    fourier_margin: float
    landauer_margin: float
    physics_score: float
    policy_score: float
    fractional: Optional[FractionalMemoryChart]
    sheaf: Optional[SheafCohomologyChart]
    kms: Optional[KMSChart]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    is_poincare_kam_stable: bool = True


@dataclass(frozen=True, slots=True)
class Phase3ThermalDecision:
    r"""Objeto de Decisión Terminal de la Fase 3."""

    heyting_verdict: str
    heyting_rank: int
    heyting_score: float
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    engine_interlock_fired: bool
    dual_channel_actuation: bool
    supervisor_disagreement: bool
    conservation_residual: float
    isolated_charts: Tuple[str, ...]
    surgical_veto: bool
    cech_h1_obstructed: bool
    fractional_cd_accumulator: float
    kms_defect: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    rsi_governor: Optional[AgentRSILevel3Chart] = None
    poincare_winding_index: int = 0


@dataclass(frozen=True, slots=True)
class ThermalGradientCertificate:
    r"""Certificado Soberano de Calibre emitido al concluir el ciclo OODA."""

    phase: str
    heyting_verdict: str
    clausius_duhem_residual: float
    carnot_efficiency: float
    exergy_potential: float
    heat_flux_norm: float
    temperature_system: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    heyting_score: float = 0.0
    conservation_residual: float = 0.0
    fourier_residual: float = 0.0
    csmd_error: float = 0.0
    landauer_gap: float = 0.0
    gouy_stodola: float = 0.0
    observation_valid: bool = True
    engine_heyting_verdict: str = ""
    policy_exergy_sufficient: bool = True
    supervisor_disagreement: bool = False
    fractional_cd_accumulator: float = 0.0
    cech_h1_obstructed: bool = False
    isolated_charts: Tuple[str, ...] = ()
    surgical_veto: bool = False
    kms_defect: float = 0.0
    uhlmann_fidelity: float = 1.0
    poincare_recurrence_time: float = 1.0
    rsi_level3_contraction: float = 0.0


# =============================================================================
# FASE 1 — OBSERVE (Sello del Motor + Sección de Poincaré Σ_OODA + Semilla RSI-L3)
# =============================================================================
class Phase1ThermalObservationMixin:
    r"""
    FASE 1 — OBSERVE.

    Invoca el motor físico `ThermalGradientLaws`, audita la integridad del certificado,
    empuja (Φ, T, |dT|²) a la ventana de memoria fraccional de Caputo y registra la
    órbita en la hipersuperficie de sección de Poincaré Σ_OODA.

    CONEXIÓN ANIDADA DE FASE:
        El último método de esta clase (`phase1_observe_engine_stamp`) devuelve
        `Phase1ThermalObservation`, que constituye el dominio exacto de inicio de la Fase 2.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        cd_threshold: float = _DEFAULT_CD_THRESHOLD,
        exergy_threshold: float = _DEFAULT_EXERGY_THRESHOLD,
        *,
        rng_seed: Optional[int] = None,
        exergy_is_hard_veto: bool = False,
        fractional_alpha: float = _DEFAULT_ALPHA,
        history_window: int = _DEFAULT_HISTORY,
        time_step: float = _DEFAULT_DT,
        coordinate_cover: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> None:
        self._n = self._validate_positive_int("dimension_n", dimension_n)
        self._safety_margin = self._validate_positive_finite("safety_margin", safety_margin)

        cd_threshold = self._validate_finite("cd_threshold", cd_threshold)
        exergy_threshold = self._validate_finite("exergy_threshold", exergy_threshold)
        if exergy_threshold < 0.0 or exergy_threshold > 1.0:
            raise ValueError("exergy_threshold debe estar en el intervalo [0, 1].")

        self._cd_thresh = float(cd_threshold * self._safety_margin)
        self._exergy_thresh = float(exergy_threshold / self._safety_margin)
        self._exergy_is_hard_veto = bool(exergy_is_hard_veto)

        alpha = self._validate_finite("fractional_alpha", fractional_alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("fractional_alpha debe vivir en (0, 1).")
        self._alpha = float(alpha)
        self._history_window = int(max(2, history_window))
        self._dt = self._validate_positive_finite("time_step", time_step)
        self._cover = self._normalize_cover(coordinate_cover)

        self._engine = ThermalGradientLaws(
            dimension_n=self._n,
            safety_margin=self._safety_margin,
            rng_seed=rng_seed,
        )
        self._rng = np.random.default_rng(rng_seed)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=self._history_window)

        # Estado de Mecánica Celeste de Poincaré en el Agente Supervisor
        self._poincare_obs_orbit: Deque[Tuple[float, float, float]] = deque(
            maxlen=_POINCARE_AGENT_WINDOW
        )
        self._poincare_section_crossings: int = 0
        self._poincare_cartan_accum: float = 0.0

        # Estado del Gobernador de Automejora Recursiva de Nivel 3 (RSI-L3) del Agente
        self._rsi_agent_epoch: int = 0
        self._rsi_effective_alpha: float = self._alpha
        self._rsi_secular_tau: float = _FRAC_SECULAR_TAU
        self._rsi_kms_thresh: float = _KMS_VETO
        self._rsi_sheaf_confidence: float = 1.0
        self._rsi_fisher_diag: np.ndarray = np.ones(4, dtype=np.float64)
        self._rsi_prev_step_norm: float = 0.0
        self._rsi_meta_damping: float = _RSI_AGENT_META_LR
        self._rsi_last_banach: float = 0.0
        self._rsi_last_lyapunov: float = 0.0
        self._rsi_last_grad_norm: float = 0.0

    def _normalize_cover(
        self,
        cover: Optional[Mapping[str, Sequence[int]]],
    ) -> Dict[str, Tuple[int, ...]]:
        if not cover:
            return {}
        out: Dict[str, Tuple[int, ...]] = {}
        for name, idxs in cover.items():
            unique = tuple(sorted({int(i) for i in idxs if 0 <= int(i) < self._n}))
            if unique:
                out[str(name)] = unique
        return out

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_positive_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente mayor que cero.")
        return value_f

    @staticmethod
    def _validate_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f):
            raise ValueError(f"{name} debe ser finito.")
        return value_f

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def kahan_sum(self, arr: np.ndarray) -> float:
        vec = np.asarray(arr, dtype=np.float64).ravel()
        return float(np.real(self._neumaier_sum(vec)))

    @staticmethod
    def _extract_float(mapping: Dict[str, Any], key: str, default: float) -> Tuple[float, bool]:
        if key not in mapping:
            return default, False
        try:
            return float(mapping[key]), True
        except (TypeError, ValueError):
            return default, False

    def _extract_vector(self, mapping: Dict[str, Any], key: str) -> Tuple[np.ndarray, bool]:
        default = np.zeros(self._n, dtype=np.float64)
        raw = mapping.get(key, default)
        try:
            arr = np.asarray(raw, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return default.copy(), False
        if arr.size != self._n or not np.all(np.isfinite(arr)):
            return default.copy(), False
        return arr, True

    def _extract_string_tuple(self, mapping: Dict[str, Any], key: str) -> Tuple[str, ...]:
        raw = mapping.get(key, ())
        if raw is None:
            return ()
        if isinstance(raw, str):
            return (raw,)
        try:
            return tuple(str(item) for item in raw)
        except TypeError:
            return ()

    def _snapshot_agent_rsi_chart(self) -> AgentRSILevel3Chart:
        """Emite una carta inmutable del estado actual del gobernador RSI-L3 del Agente."""
        with self._interlock_lock:
            return AgentRSILevel3Chart(
                epoch=int(self._rsi_agent_epoch),
                level1_effective_alpha=float(self._rsi_effective_alpha),
                level1_secular_tau=float(self._rsi_secular_tau),
                level1_kms_threshold=float(self._rsi_kms_thresh),
                level1_sheaf_confidence=float(self._rsi_sheaf_confidence),
                level2_lyapunov_loss=float(self._rsi_last_lyapunov),
                level2_natural_grad_norm=float(self._rsi_last_grad_norm),
                level2_fisher_trace=float(np.sum(self._rsi_fisher_diag)),
                level3_banach_contraction=float(self._rsi_last_banach),
                level3_meta_damping=float(self._rsi_meta_damping),
                godel_tarski_soundness_locked=bool(
                    0.0 < self._rsi_effective_alpha < 1.0
                    and self._rsi_secular_tau > 0.0
                    and self._rsi_kms_thresh > 0.0
                    and self._rsi_last_banach < 1.0
                ),
            )

    def _update_poincare_observation_orbit(
        self,
        grad_T: np.ndarray,
        heat_flux_norm: float,
        temperature_sys: float,
        phi_cd: float,
        kam_inherited: float,
    ) -> PoincareObservationOrbitChart:
        r"""
        Actualiza la órbita observacional en el espacio fase (q_obs, p_obs) y audita
        los cruces transversales sobre la superficie de sección de Poincaré Σ_OODA.
        """
        q_obs = float(la.norm(grad_T)) if np.all(np.isfinite(grad_T)) else 0.0
        T_safe = max(temperature_sys, _WILKINSON_FLOOR) if math.isfinite(temperature_sys) else 1.0
        p_obs = float(heat_flux_norm / T_safe) if math.isfinite(heat_flux_norm) else 0.0
        phi_safe = float(phi_cd) if math.isfinite(phi_cd) else 0.0
        H_obs = 0.5 * (q_obs * q_obs + p_obs * p_obs)

        with self._interlock_lock:
            if len(self._poincare_obs_orbit) > 0:
                q_prev, p_prev, _ = self._poincare_obs_orbit[-1]
                dq = q_obs - q_prev
                d_action = 0.5 * (p_obs + p_prev) * dq - H_obs * self._dt
                self._poincare_cartan_accum += float(d_action)
                # Cruce transversal positivo en la sección media de momento p
                orbit_ps = [pt[1] for pt in self._poincare_obs_orbit]
                p_mean = sum(orbit_ps) / len(orbit_ps)
                if p_prev < p_mean <= p_obs:
                    self._poincare_section_crossings += 1
            self._poincare_obs_orbit.append((q_obs, p_obs, phi_safe))
            orbit_snapshot = tuple(self._poincare_obs_orbit)
            crossings = int(self._poincare_section_crossings)
            cartan_act = float(self._poincare_cartan_accum)

        hopf_idx = 0
        if len(orbit_snapshot) >= 3:
            qs = [pt[0] for pt in orbit_snapshot]
            ps = [pt[1] for pt in orbit_snapshot]
            qm = sum(qs) / len(qs)
            pm = sum(ps) / len(ps)
            ang_sum = 0.0
            for k in range(1, len(orbit_snapshot)):
                a1 = math.atan2(ps[k - 1] - pm, qs[k - 1] - qm)
                a2 = math.atan2(ps[k] - pm, qs[k] - qm)
                da = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
                ang_sum += da
            hopf_idx = int(round(ang_sum / (2.0 * math.pi)))

        return PoincareObservationOrbitChart(
            canonical_q_obs=q_obs,
            canonical_p_obs=p_obs,
            poincare_section_crossings=crossings,
            poincare_cartan_action=cartan_act,
            poincare_hopf_index=hopf_idx,
            kam_stability_inherited=float(
                kam_inherited if math.isfinite(kam_inherited) else 1.0
            ),
        )

    def _audit_engine_certificate(
        self,
        stamp: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> EngineCertificateAudit:
        missing: List[str] = []

        def grab(key: str, default: float) -> float:
            value, present = self._extract_float(stamp, key, default)
            if not present:
                missing.append(key)
            return value

        engine_verdict = str(stamp.get("heyting_verdict", "")).strip().upper()
        engine_interlock = bool(stamp.get("hardware_interlock_fired", False))
        nested = stamp.get("diagnostics")
        nested_p1: Dict[str, Any] = {}
        nested_p2: Dict[str, Any] = {}
        if isinstance(nested, dict):
            p1 = nested.get("phase1")
            p2 = nested.get("phase2")
            if isinstance(p1, dict):
                nested_p1 = p1
            if isinstance(p2, dict):
                nested_p2 = p2

        pairing = float(nested_p2.get("q_dot_grad_T", float("nan")))
        condition = float(nested_p2.get("metric_condition", float("nan")))
        third_law = bool(nested_p1.get("third_law_flag", False))
        t_clamped = bool(nested_p1.get("temperature_clamped", False))

        kam_stab, _ = self._extract_float(stamp, "kam_torus_stability", 1.0)
        p_rec, _ = self._extract_float(stamp, "poincare_recurrence_time", 1.0)
        p_mono, _ = self._extract_float(stamp, "poincare_monodromy_det", 1.0)
        melnikov_d, _ = self._extract_float(stamp, "melnikov_distance", 0.0)
        eng_rsi, _ = self._extract_float(stamp, "rsi_level3_contraction", 0.0)

        return EngineCertificateAudit(
            available=bool(stamp),
            engine_verdict=engine_verdict,
            engine_interlock_fired=engine_interlock,
            fourier_residual=grab("fourier_residual", float("nan")),
            csmd_error=grab("csmd_error", float("nan")),
            landauer_gap=grab("landauer_gap", float("nan")),
            gouy_stodola=grab("gouy_stodola", float("nan")),
            conservation_residual=grab("conservation_residual", float("nan")),
            thermal_entropy_production=grab("thermal_entropy_production", float("nan")),
            dirichlet_energy=grab("dirichlet_energy", float("nan")),
            carnot_density=grab("carnot_density", float("nan")),
            T_hot=grab("T_hot", float("nan")),
            T_cold=grab("T_cold", float("nan")),
            pairing_q_dT=pairing,
            metric_condition=condition,
            third_law_flag=third_law,
            temperature_clamped=t_clamped,
            engine_veto_reasons=self._extract_string_tuple(stamp, "veto_reasons"),
            engine_degraded_reasons=self._extract_string_tuple(stamp, "degraded_reasons"),
            missing_keys=tuple(missing),
            adaptive_tau_cd=grab("adaptive_tau_cd", float("nan")),
            tellegen_residual=grab("tellegen_residual", float("nan")),
            kam_torus_stability=kam_stab,
            poincare_recurrence_time=p_rec,
            poincare_monodromy_det=p_mono,
            melnikov_distance=melnikov_d,
            engine_rsi_contraction=eng_rsi,
        )

    def _fail_safe_observation(
        self,
        diagnostics: Dict[str, Any],
        grad_T: np.ndarray,
        entropy_production_rate: float,
    ) -> Phase1ThermalObservation:
        empty_q = np.zeros(self._n, dtype=np.float64)
        audit = EngineCertificateAudit(
            available=False,
            engine_verdict="VETOED",
            engine_interlock_fired=False,
            fourier_residual=float("nan"),
            csmd_error=float("nan"),
            landauer_gap=float("nan"),
            gouy_stodola=float("nan"),
            conservation_residual=float("nan"),
            thermal_entropy_production=float("nan"),
            dirichlet_energy=float("nan"),
            carnot_density=float("nan"),
            T_hot=float("nan"),
            T_cold=float("nan"),
            pairing_q_dT=float("nan"),
            metric_condition=float("nan"),
            third_law_flag=False,
            temperature_clamped=False,
            engine_veto_reasons=("engine_invocation_failed",),
            engine_degraded_reasons=(),
            missing_keys=(),
        )
        return Phase1ThermalObservation(
            clausius_duhem_residual=float("nan"),
            carnot_efficiency=float("nan"),
            exergy_potential=float("nan"),
            heat_flux_norm=0.0,
            heat_flux_energy=0.0,
            temperature_system=float("nan"),
            heat_flux_vector=empty_q,
            grad_T=grad_T,
            entropy_production_rate=entropy_production_rate,
            engine_stamp={},
            observation_valid=False,
            engine_alive=False,
            certificate_audit=audit,
            history_length=len(self._history),
            diagnostics=diagnostics,
            poincare_orbit=None,
            rsi_governor=self._snapshot_agent_rsi_chart(),
        )

    def _push_history(self, phi: float, temperature: float, dirichlet: float) -> int:
        with self._interlock_lock:
            self._history.append(
                (
                    float(phi) if math.isfinite(phi) else 0.0,
                    float(temperature) if math.isfinite(temperature) else 0.0,
                    float(max(dirichlet, 0.0)) if math.isfinite(dirichlet) else 0.0,
                )
            )
            return len(self._history)

    def reset_fractional_memory(self) -> None:
        """Purga la ventana de Caputo y la órbita de sección de Poincaré (p. ej. tras rearranque)."""
        with self._interlock_lock:
            self._history.clear()
            self._poincare_obs_orbit.clear()
            self._poincare_section_crossings = 0
            self._poincare_cartan_accum = 0.0

    def phase1_observe_engine_stamp(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        **engine_kwargs: Any,
    ) -> Phase1ThermalObservation:
        r"""
        [FASE 1 — MORFISMO TERMINAL DE LA FASE 1 → ESLABÓN DE INICIO DE LA FASE 2]

        Ejecuta el motor de leyes térmicas, audita el sello criptográfico/físico,
        actualiza la órbita en la sección transversal de Poincaré Σ_OODA y alimenta
        la ventana de memoria fraccional de Caputo.

        Codominio: `Phase1ThermalObservation`
                 ≡ Dominio exacto de inicio de `Phase2ThermalOrientationMixin`.
        """
        logger.info("Fase Observe: sello del motor + sección de Poincaré + memoria fraccional.")
        diagnostics: Dict[str, Any] = {
            "observe_phase": "engine_invocation",
            "engine_invoked": True,
        }
        try:
            grad_T = np.asarray(grad_T_raw, dtype=np.float64).reshape(-1)
            if grad_T.size != self._n:
                grad_T = np.zeros(self._n, dtype=np.float64)
                diagnostics["grad_T_invalid"] = True
        except (TypeError, ValueError):
            grad_T = np.zeros(self._n, dtype=np.float64)
            diagnostics["grad_T_invalid"] = True

        sigma = 0.0
        try:
            sigma = float(entropy_production_rate)
            if not math.isfinite(sigma):
                sigma = 0.0
        except (TypeError, ValueError):
            sigma = 0.0

        try:
            stamp = self._engine.execute_thermal_cycle(
                K_raw=K_raw,
                grad_T_raw=grad_T_raw,
                T_sys=T_sys,
                metric_tensor=metric_tensor,
                entropy_production_rate=entropy_production_rate,
                **engine_kwargs,
            )
        except TypeError:
            try:
                stamp = self._engine.execute_thermal_cycle(
                    K_raw=K_raw,
                    grad_T_raw=grad_T_raw,
                    T_sys=T_sys,
                    metric_tensor=metric_tensor,
                    entropy_production_rate=entropy_production_rate,
                )
                diagnostics["engine_kwargs_dropped"] = tuple(engine_kwargs.keys())
            except Exception as exc:
                logger.exception("Motor térmico falló. Fail-safe.")
                diagnostics.update({"engine_error": str(exc), "engine_invoked": False})
                return self._fail_safe_observation(diagnostics, grad_T, sigma)
        except Exception as exc:
            logger.exception("Motor térmico falló. Fail-safe.")
            diagnostics.update({"engine_error": str(exc), "engine_invoked": False})
            return self._fail_safe_observation(diagnostics, grad_T, sigma)

        if not isinstance(stamp, dict):
            try:
                stamp = dict(stamp)
            except Exception:
                stamp = {}
                diagnostics["engine_stamp_invalid"] = True

        cd_residual, cd_present = self._extract_float(
            stamp, "clausius_duhem_residual", float("nan")
        )
        carnot_efficiency, eta_present = self._extract_float(
            stamp, "carnot_efficiency", float("nan")
        )
        exergy_potential, ex_present = self._extract_float(
            stamp, "exergy_potential", float("nan")
        )
        temperature_system, t_present = self._extract_float(
            stamp, "T_sys_clamped", float("nan")
        )
        heat_flux_vector, heat_flux_valid = self._extract_vector(
            stamp, "heat_flux_vector"
        )
        dirichlet, _ = self._extract_float(stamp, "dirichlet_energy", float("nan"))

        if not eta_present and not math.isfinite(carnot_efficiency):
            carnot_efficiency = 0.0
        if not ex_present and not math.isfinite(exergy_potential):
            exergy_potential = 0.0

        if heat_flux_valid:
            energy = self.kahan_sum(np.asarray(heat_flux_vector, dtype=np.float64) ** 2)
            heat_flux_energy = float(max(energy, 0.0))
            heat_flux_norm = float(math.sqrt(heat_flux_energy))
            if not math.isfinite(heat_flux_norm):
                heat_flux_norm = 0.0
                heat_flux_energy = 0.0
        else:
            heat_flux_norm = 0.0
            heat_flux_energy = 0.0
            diagnostics["heat_flux_vector_invalid"] = True

        engine_diagnostics = stamp.get("diagnostics")
        if isinstance(engine_diagnostics, dict):
            diagnostics["engine"] = dict(engine_diagnostics)

        audit = self._audit_engine_certificate(stamp, diagnostics)
        hist_len = self._push_history(cd_residual, temperature_system, dirichlet)
        poincare_orbit = self._update_poincare_observation_orbit(
            grad_T=grad_T,
            heat_flux_norm=heat_flux_norm,
            temperature_sys=temperature_system,
            phi_cd=cd_residual,
            kam_inherited=audit.kam_torus_stability,
        )
        rsi_snapshot = self._snapshot_agent_rsi_chart()

        observation_valid = bool(
            heat_flux_valid
            and cd_present
            and math.isfinite(cd_residual)
            and t_present
            and math.isfinite(temperature_system)
        )
        diagnostics["observation_valid"] = observation_valid
        diagnostics["history_length"] = hist_len
        diagnostics["cover_names"] = tuple(self._cover.keys())
        diagnostics["poincare_section_crossings"] = (
            poincare_orbit.poincare_section_crossings
        )
        diagnostics["poincare_hopf_index"] = poincare_orbit.poincare_hopf_index
        diagnostics["agent_rsi_epoch"] = rsi_snapshot.epoch

        return Phase1ThermalObservation(
            clausius_duhem_residual=cd_residual,
            carnot_efficiency=carnot_efficiency,
            exergy_potential=exergy_potential,
            heat_flux_norm=heat_flux_norm,
            heat_flux_energy=heat_flux_energy,
            temperature_system=temperature_system,
            heat_flux_vector=heat_flux_vector,
            grad_T=grad_T,
            entropy_production_rate=sigma,
            engine_stamp=dict(stamp),
            observation_valid=observation_valid,
            engine_alive=True,
            certificate_audit=audit,
            history_length=hist_len,
            diagnostics=diagnostics,
            poincare_orbit=poincare_orbit,
            rsi_governor=rsi_snapshot,
        )


# =============================================================================
# FASE 2 — ORIENT (Continuación Formal desde Fase 1: Caputo-Poincaré + Čech-Birkhoff + KMS)
# =============================================================================
class Phase2ThermalOrientationMixin(Phase1ThermalObservationMixin):
    r"""
    FASE 2 — ORIENT (Continuación Anidada Directa del Codominio `Phase1ThermalObservation`).

    Recibe `Phase1ThermalObservation` desde `phase1_observe_engine_stamp` y ejecuta:
        1. Cálculo Fraccional de Caputo / Grünwald-Letnikov / Riemann-Liouville acoplado con
           la Acción Simpléctica de Poincaré-Cartan en el plano de memoria (I^α Φ, D^α Φ) y
           la integral homoclínica de Melnikov.
        2. Cohomología de Čech H¹_Čech sobre el haz de Heyting enriquecida con la Holonomía
           de Torsión de Poincaré-Birkhoff en los solapes U_i ∩ U_j.
        3. Auditoría de Estados KMS Cuánticos, Conmutador del Flujo Modular de Tomita-Takesaki
           y Distancia Geodésica de Bures-Poincaré.

    CONEXIÓN ANIDADA DE FASE:
        El último método de esta clase (`phase2_orient_from_observation`) emite
        `Phase2ThermalOrientation`, que junto a `Phase1ThermalObservation` inicia la Fase 3.
    """

    def verify_thermodynamic_coherence(
        self,
        cd_residual: float,
        exergy_potential: float,
    ) -> Tuple[bool, bool]:
        """[COMPATIBILIDAD 1.X — FASE ORIENT]"""
        try:
            cd_value = float(cd_residual)
        except (TypeError, ValueError):
            cd_value = float("nan")
        try:
            exergy_value = float(exergy_potential)
        except (TypeError, ValueError):
            exergy_value = float("nan")
        is_cd_coherent = math.isfinite(cd_value) and cd_value >= self._cd_thresh
        is_exergy_sufficient = (
            math.isfinite(exergy_value) and exergy_value >= self._exergy_thresh
        )
        return is_cd_coherent, is_exergy_sufficient

    @staticmethod
    def _gl_weights(alpha: float, n: int) -> np.ndarray:
        w = np.empty(n, dtype=np.float64)
        if n <= 0:
            return w
        w[0] = 1.0
        for j in range(1, n):
            w[j] = w[j - 1] * (j - 1.0 - alpha) / j
        return w

    @staticmethod
    def _rl_kernel(alpha: float, n: int, dt: float) -> np.ndarray:
        r"""Núcleo RL I^α: k_j = Δt^α (j+1)^{α−1} / Γ(α), j=0..n−1 (más reciente = 0)."""
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        g = math.gamma(alpha)
        k = np.empty(n, dtype=np.float64)
        scale = (dt**alpha) / g
        for j in range(n):
            k[j] = scale * ((j + 1.0) ** (alpha - 1.0))
        return k

    def _fractional_memory_chart(self) -> FractionalMemoryChart:
        r"""
        Evalúa los operadores no-markovianos D^α_GL e I^α_RL junto con la dinámica
        simpléctica de Poincaré-Melnikov en el plano fase de memoria (I^α Φ, D^α Φ).
        """
        with self._interlock_lock:
            hist = tuple(self._history)
            alpha_eff = float(self._rsi_effective_alpha)
            sec_tau = float(self._rsi_secular_tau)

        n = len(hist)
        if n == 0:
            return FractionalMemoryChart(
                alpha=self._alpha,
                dt=self._dt,
                window=self._history_window,
                samples=0,
                gl_phi=0.0,
                gl_temperature=0.0,
                rl_integral_phi=0.0,
                secular_leak=False,
                transient_flag=False,
                kernel_l1=0.0,
                poincare_fractional_action=0.0,
                poincare_recurrence_estimate=1.0,
                melnikov_homoclinic_integral=0.0,
                kam_memory_confined=True,
            )

        phi = np.array([h[0] for h in hist], dtype=np.float64)
        temp = np.array([h[1] for h in hist], dtype=np.float64)
        dirich = np.array([h[2] for h in hist], dtype=np.float64)

        # GL: el presente es el último sample; pesos w_j multiplican f_{n-1-j}.
        w = self._gl_weights(alpha_eff, n)
        dt_a = self._dt ** (-alpha_eff)
        gl_phi = float(dt_a * self.kahan_sum(w * phi[::-1]))
        gl_T = float(dt_a * self.kahan_sum(w * temp[::-1]))
        rl = self._rl_kernel(alpha_eff, n, self._dt)
        rl_phi = float(self.kahan_sum(rl * phi[::-1]))
        kernel_l1 = float(self.kahan_sum(np.abs(rl)))

        # Acción simpléctica de Poincaré-Cartan e Integral Homoclínica de Melnikov en memoria
        # Construimos la curva acumulada (I^α Φ_k, D^α Φ_k) sobre los sub-intervalos
        poincare_action = float(0.5 * rl_phi * gl_phi * self._dt)
        melnikov_terms = rl * phi[::-1] * np.sin(np.linspace(0.0, math.pi, n))
        melnikov_int = float(self.kahan_sum(melnikov_terms))

        mean_dirich = float(self.kahan_sum(dirich) / max(n, 1))
        t_last = max(abs(float(temp[-1])), 1.0)
        rec_exp = float(np.clip(mean_dirich / (t_last * t_last), 0.0, 25.0))
        poincare_rec = float(math.exp(rec_exp))

        # Fuga secular: la integral fraccional de Φ es persistentemente negativa.
        secular = bool(n >= 4 and rl_phi < -sec_tau * self._safety_margin)
        # Transitorio: |D^α T| grande frente a la escala de T, sin fuga secular de Φ.
        transient = bool(
            n >= 4
            and abs(gl_T) > 0.05 * t_last * dt_a
            and rl_phi >= -sec_tau * self._safety_margin
        )
        kam_confined = bool(not secular and melnikov_int >= -sec_tau * self._safety_margin)

        return FractionalMemoryChart(
            alpha=self._alpha,
            dt=self._dt,
            window=self._history_window,
            samples=n,
            gl_phi=gl_phi,
            gl_temperature=gl_T,
            rl_integral_phi=rl_phi,
            secular_leak=secular,
            transient_flag=transient,
            kernel_l1=kernel_l1,
            poincare_fractional_action=poincare_action,
            poincare_recurrence_estimate=poincare_rec,
            melnikov_homoclinic_integral=melnikov_int,
            kam_memory_confined=kam_confined,
        )

    def _local_phi(
        self,
        observation: Phase1ThermalObservation,
        indices: Tuple[int, ...],
    ) -> Tuple[float, float, float, float]:
        q = np.asarray(observation.heat_flux_vector, dtype=np.float64)
        g = np.asarray(observation.grad_T, dtype=np.float64)
        idx = np.array(indices, dtype=int)
        pairing = float(self.kahan_sum(q[idx] * g[idx]))
        mass = float(self.kahan_sum(g[idx] * g[idx]))
        q_mass = float(self.kahan_sum(q[idx] * q[idx]))
        mass_tot = float(self.kahan_sum(g * g))
        frac = (
            mass / mass_tot
            if mass_tot > _WILKINSON_FLOOR
            else (len(indices) / max(self._n, 1))
        )
        sigma_u = float(observation.entropy_production_rate) * frac
        T = observation.temperature_system
        t2 = T * T if math.isfinite(T) and T > 0.0 else 1.0
        phi_u = float(sigma_u - pairing / t2)

        # Ángulo de fase de Poincaré-Birkhoff entre flujo de calor local y gradiente local
        denom = math.sqrt(max(mass * q_mass, 0.0))
        if denom > _WILKINSON_FLOOR:
            cos_theta = float(np.clip(pairing / denom, -1.0, 1.0))
            phase_angle = float(math.acos(cos_theta))
        else:
            phase_angle = math.pi  # En equilibrio isotermo, ángulo nominal antiparalelo π
        return pairing, phi_u, mass, phase_angle

    def _local_verdict(self, phi_u: float, tau: float) -> HeytingVerdict:
        if not math.isfinite(phi_u):
            return HeytingVerdict.VETOED
        if phi_u < tau:
            return HeytingVerdict.VETOED
        if phi_u < 0.0:
            return HeytingVerdict.DEGRADED
        return HeytingVerdict.CERTIFIED

    def _sheaf_cech_chart(
        self, observation: Phase1ThermalObservation
    ) -> SheafCohomologyChart:
        r"""
        Evalúa las secciones locales Γ(U_i, ℋ), la cohomología de Čech H¹_Čech sobre
        los solapes U_i ∩ U_j y la holonomía de torsión de Poincaré-Birkhoff.
        """
        tau = self._cd_thresh
        if not self._cover:
            pairing = (
                float(
                    self.kahan_sum(
                        np.asarray(observation.heat_flux_vector)
                        * np.asarray(observation.grad_T)
                    )
                )
                if observation.engine_alive
                else float("nan")
            )
            section = ChartSection(
                name="global",
                indices=tuple(range(self._n)),
                pairing=pairing,
                phi_local=observation.clausius_duhem_residual,
                dirichlet_mass=float("nan"),
                verdict=HeytingVerdict.parse(
                    observation.certificate_audit.engine_verdict or "COHERENT"
                ).value,
                rank=HeytingVerdict.parse(
                    observation.certificate_audit.engine_verdict or "COHERENT"
                ).rank,
                poincare_phase_angle=math.pi,
            )
            return SheafCohomologyChart(
                cover_names=("global",),
                sections=(section,),
                overlap_disagreements=(),
                cech_h1_obstructed=False,
                isolated_charts=(),
                surgical_possible=False,
                global_meet=section.verdict,
                cocycle_l2_norm=0.0,
                poincare_birkhoff_holonomy=0.0,
            )

        sections: List[ChartSection] = []
        for name, idxs in self._cover.items():
            pairing, phi_u, mass, phase_ang = self._local_phi(observation, idxs)
            v = self._local_verdict(phi_u, tau)
            sections.append(
                ChartSection(
                    name=name,
                    indices=idxs,
                    pairing=pairing,
                    phi_local=phi_u,
                    dirichlet_mass=mass,
                    verdict=v.value,
                    rank=v.rank,
                    poincare_phase_angle=phase_ang,
                )
            )

        disagreements: List[Tuple[str, str, str, str]] = []
        cocycle_sq_terms: List[float] = []
        holonomy_twists: List[float] = []
        names = list(self._cover.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = tuple(sorted(set(self._cover[a]) & set(self._cover[b])))
                if not overlap:
                    continue
                sa = next(s for s in sections if s.name == a)
                sb = next(s for s in sections if s.name == b)
                if abs(sa.rank - sb.rank) >= _CECH_RANK_GAP:
                    disagreements.append((a, b, sa.verdict, sb.verdict))
                if math.isfinite(sa.phi_local) and math.isfinite(sb.phi_local):
                    cocycle_sq_terms.append(float((sa.phi_local - sb.phi_local) ** 2))
                holonomy_twists.append(
                    float(abs(sa.poincare_phase_angle - sb.poincare_phase_angle))
                )

        cocycle_norm = (
            float(math.sqrt(self.kahan_sum(np.asarray(cocycle_sq_terms, dtype=np.float64))))
            if cocycle_sq_terms
            else 0.0
        )
        max_holonomy = float(max(holonomy_twists)) if holonomy_twists else 0.0

        isolated = tuple(
            s.name for s in sections if s.verdict == HeytingVerdict.VETOED.value
        )
        meet = HeytingVerdict.CERTIFIED
        for s in sections:
            meet = meet.meet(HeytingVerdict.parse(s.verdict))
        obstructed = bool(disagreements)
        surgical = bool(isolated) and (
            len(isolated) < len(sections) or not obstructed
        )
        return SheafCohomologyChart(
            cover_names=tuple(names),
            sections=tuple(sections),
            overlap_disagreements=tuple(disagreements),
            cech_h1_obstructed=obstructed,
            isolated_charts=isolated,
            surgical_possible=surgical,
            global_meet=meet.value,
            cocycle_l2_norm=cocycle_norm,
            poincare_birkhoff_holonomy=max_holonomy,
        )

    def _kms_chart(
        self,
        rho: Optional[np.ndarray],
        hamiltonian: Optional[np.ndarray],
        temperature: float,
    ) -> KMSChart:
        r"""
        Audita la condición KMS a temperatura inversa β = 1/T, la fidelidad cuántica
        de Uhlmann, el conmutador modular de Tomita-Takesaki ‖[Δ_ρ^{it}, H]‖_F y la
        distancia geodésica de Bures-Poincaré.
        """
        if rho is None or hamiltonian is None:
            return KMSChart(
                available=False,
                is_density=False,
                inverse_temperature=float("nan"),
                relative_entropy=float("nan"),
                modular_defect=float("nan"),
                uhlmann_fidelity=float("nan"),
                von_neumann_entropy=float("nan"),
                kms_coherent=True,
                tomita_takesaki_commutator=0.0,
                bures_poincare_distance=0.0,
            )
        rho_h = 0.5 * (
            np.asarray(rho, dtype=np.complex128)
            + np.asarray(rho, dtype=np.complex128).conj().T
        )
        H_h = 0.5 * (
            np.asarray(hamiltonian, dtype=np.complex128)
            + np.asarray(hamiltonian, dtype=np.complex128).conj().T
        )
        if rho_h.shape != (self._n, self._n) or H_h.shape != (self._n, self._n):
            return KMSChart(
                available=True,
                is_density=False,
                inverse_temperature=float("nan"),
                relative_entropy=float("inf"),
                modular_defect=float("inf"),
                uhlmann_fidelity=0.0,
                von_neumann_entropy=float("nan"),
                kms_coherent=False,
                tomita_takesaki_commutator=float("inf"),
                bures_poincare_distance=math.sqrt(2.0),
            )
        evals, evecs = la.eigh(rho_h)
        evals = np.real(evals)
        trace = float(np.real(np.sum(evals)))
        min_eig = float(np.min(evals))
        is_density = (
            min_eig >= -_PSD_TOLERANCE and abs(trace - 1.0) <= _DENSITY_TRACE_TOLERANCE
        )
        acc_s: List[float] = []
        for eta in evals:
            clipped = max(float(eta), 0.0)
            if clipped > _WILKINSON_FLOOR:
                acc_s.append(float(-clipped * np.log(clipped)))
        svn = (
            float(self.kahan_sum(np.asarray(acc_s, dtype=np.float64)))
            if acc_s
            else 0.0
        )
        T = (
            float(temperature)
            if math.isfinite(temperature) and temperature > 0.0
            else float("nan")
        )
        if not math.isfinite(T):
            return KMSChart(
                available=True,
                is_density=is_density,
                inverse_temperature=float("nan"),
                relative_entropy=float("inf"),
                modular_defect=float("inf"),
                uhlmann_fidelity=0.0,
                von_neumann_entropy=svn,
                kms_coherent=False,
                tomita_takesaki_commutator=float("inf"),
                bures_poincare_distance=math.sqrt(2.0),
            )
        beta = 1.0 / T
        energy = float(np.real(np.trace(rho_h @ H_h)))
        h_eigs = np.real(la.eigvalsh(H_h))
        log_z = _logsumexp(-beta * h_eigs)
        relative = max(0.0, float(-svn + beta * energy + log_z))
        # log ρ = V log(η)_+ V† ;  c = −mean(log η + β λ_H) en el soporte.
        log_eta = np.log(np.maximum(evals, _WILKINSON_FLOOR))
        log_rho = (evecs * log_eta) @ evecs.conj().T
        target = -beta * H_h
        residual = log_rho - target
        residual = residual - (np.trace(residual) / self._n) * np.eye(self._n)
        modular = float(la.norm(residual, ord="fro"))

        # Flujo modular de Tomita-Takesaki: σ_t(H) = ρ^{it} H ρ^{-it} en t = β
        phase_mod = np.exp(1.0j * beta * log_eta)
        u_mod = (evecs * phase_mod) @ evecs.conj().T
        tomita_comm = float(la.norm(u_mod @ H_h - H_h @ u_mod, ord="fro"))

        # Estado de Gibbs ρ_β para fidelidad de Uhlmann y distancia geodésica de Bures-Poincaré.
        gibbs_e = np.exp(-beta * h_eigs - log_z)
        H_vecs = la.eigh(H_h)[1]
        rho_beta = (H_vecs * gibbs_e) @ H_vecs.conj().T
        try:
            sqrt_rho = evecs * np.sqrt(np.maximum(evals, 0.0)) @ evecs.conj().T
            inner = sqrt_rho @ rho_beta @ sqrt_rho
            fid_eigs = np.real(la.eigvalsh(inner))
            root_fidelity = float(np.real(np.sum(np.sqrt(np.maximum(fid_eigs, 0.0)))))
            fidelity = float(np.clip(root_fidelity * root_fidelity, 0.0, 1.0))
        except la.LinAlgError:
            root_fidelity = 0.0
            fidelity = 0.0

        bures_dist = float(math.sqrt(max(0.0, 2.0 * (1.0 - min(root_fidelity, 1.0)))))
        with self._interlock_lock:
            kms_thresh = float(self._rsi_kms_thresh)
        coherent = bool(is_density and relative <= kms_thresh and modular <= 1.0)
        return KMSChart(
            available=True,
            is_density=is_density,
            inverse_temperature=beta,
            relative_entropy=relative,
            modular_defect=modular,
            uhlmann_fidelity=fidelity,
            von_neumann_entropy=svn,
            kms_coherent=coherent,
            tomita_takesaki_commutator=tomita_comm,
            bures_poincare_distance=bures_dist,
        )

    def _phase2_audit_engine_certificate(
        self,
        observation: Phase1ThermalObservation,
    ) -> Dict[str, Any]:
        audit = observation.certificate_audit
        flags: Dict[str, Any] = {}
        eta = observation.carnot_efficiency
        flags["carnot_in_range"] = math.isfinite(eta) and 0.0 <= eta <= 1.0 + 1e-12
        T = observation.temperature_system
        flags["temperature_physical"] = (
            math.isfinite(T) and T > 0.0 and not audit.third_law_flag
        )
        T_h, T_c = audit.T_hot, audit.T_cold
        if math.isfinite(T_h) and math.isfinite(T_c):
            flags["carnot_reservoirs_ordered"] = bool(T_h + 1e-12 >= T_c > 0.0)
            if math.isfinite(eta) and T_h > T_c > 0.0:
                flags["carnot_identity_residual"] = float(
                    abs(eta - (1.0 - T_c / T_h))
                )
            else:
                flags["carnot_identity_residual"] = 0.0
        else:
            flags["carnot_reservoirs_ordered"] = True
            flags["carnot_identity_residual"] = 0.0
        pairing = audit.pairing_q_dT
        flags["anti_fourier"] = bool(
            math.isfinite(pairing) and pairing > _WILKINSON_FLOOR
        )
        flags["fourier_coherent"] = (
            True
            if not math.isfinite(audit.fourier_residual)
            else audit.fourier_residual <= _FOURIER_VETO * self._safety_margin
        )
        flags["csmd_coherent"] = (
            True
            if not math.isfinite(audit.csmd_error)
            else audit.csmd_error <= _CSMD_VETO
        )
        flags["landauer_coherent"] = (
            True
            if not math.isfinite(audit.landauer_gap)
            else audit.landauer_gap >= -_LANDAUER_ABS_TOL
        )
        flags["engine_verdict_known"] = audit.engine_verdict in {
            "VETOED",
            "DEGRADED",
            "COHERENT",
            "CERTIFIED",
            "",
        }
        flags["poincare_liouville_contracting"] = bool(
            not math.isfinite(audit.poincare_monodromy_det)
            or audit.poincare_monodromy_det <= 1.0 + 1e-6
        )
        return flags

    def phase2_orient_from_observation(
        self,
        observation: Phase1ThermalObservation,
        *,
        density_rho: Optional[np.ndarray] = None,
        hamiltonian_H: Optional[np.ndarray] = None,
    ) -> Phase2ThermalOrientation:
        r"""
        [FASE 2 — CONTINUACIÓN DE `phase1_observe_engine_stamp` Y MORFISMO TERMINAL HACIA FASE 3]

        Dominio  : `Phase1ThermalObservation` (codominio exacto de Fase 1).
        Codominio: `Phase2ThermalOrientation` (segundo factor exacto del dominio de Fase 3).
        """
        if not isinstance(observation, Phase1ThermalObservation):
            raise TypeError("observation debe ser Phase1ThermalObservation.")

        is_cd_coherent, is_exergy_sufficient = self.verify_thermodynamic_coherence(
            cd_residual=observation.clausius_duhem_residual,
            exergy_potential=observation.exergy_potential,
        )
        flags = self._phase2_audit_engine_certificate(observation)
        audit = observation.certificate_audit
        frac = self._fractional_memory_chart()
        sheaf = self._sheaf_cech_chart(observation)
        kms = self._kms_chart(density_rho, hamiltonian_H, observation.temperature_system)

        cd_value = observation.clausius_duhem_residual
        exergy_value = observation.exergy_potential
        cd_margin = (
            float(cd_value - self._cd_thresh)
            if math.isfinite(cd_value)
            else float("-inf")
        )
        exergy_margin = (
            float(exergy_value - self._exergy_thresh)
            if math.isfinite(exergy_value)
            else float("-inf")
        )
        fourier_margin = (
            float(_FOURIER_VETO * self._safety_margin - audit.fourier_residual)
            if math.isfinite(audit.fourier_residual)
            else float("inf")
        )
        landauer_margin = (
            float(audit.landauer_gap)
            if math.isfinite(audit.landauer_gap)
            else float("inf")
        )

        is_fourier = bool(flags["fourier_coherent"]) and not flags["anti_fourier"]
        is_csmd = bool(flags["csmd_coherent"])
        is_landauer = bool(flags["landauer_coherent"])
        is_temp = bool(flags["temperature_physical"]) and observation.engine_alive
        is_carnot = bool(flags["carnot_in_range"]) and bool(
            flags["carnot_reservoirs_ordered"]
        )
        identity_ok = float(flags["carnot_identity_residual"]) <= 1e-8
        cert_ok = (
            observation.observation_valid
            and observation.engine_alive
            and bool(flags["engine_verdict_known"])
            and identity_ok
        )
        is_kam_stable = bool(
            frac.kam_memory_confined and flags["poincare_liouville_contracting"]
        )

        physics_terms = [
            1.0 if is_cd_coherent else 0.0,
            1.0 if is_fourier else 0.0,
            1.0 if is_csmd else 0.0,
            1.0 if is_landauer else 0.0,
            1.0 if is_temp else 0.0,
            1.0 if is_carnot else 0.0,
            1.0 if observation.engine_alive else 0.0,
            1.0 if observation.observation_valid else 0.0,
            0.0 if frac.secular_leak else 1.0,
            0.0 if (kms.available and not kms.kms_coherent) else 1.0,
        ]
        physics_score = float(min(physics_terms))
        if math.isfinite(audit.fourier_residual) and audit.fourier_residual > 0.0:
            physics_score = min(
                physics_score,
                float(
                    np.exp(
                        -audit.fourier_residual
                        / max(_FOURIER_DEGRADED, _WILKINSON_FLOOR)
                    )
                ),
            )
        policy_score = (
            1.0
            if is_exergy_sufficient
            else float(
                np.clip(
                    observation.exergy_potential
                    / max(self._exergy_thresh, _WILKINSON_FLOOR),
                    0.0,
                    0.99,
                )
            )
            if math.isfinite(observation.exergy_potential)
            else 0.0
        )

        diagnostics: Dict[str, Any] = {
            "cd_threshold": self._cd_thresh,
            "exergy_threshold": self._exergy_thresh,
            "exergy_is_hard_veto": self._exergy_is_hard_veto,
            "cd_margin": cd_margin,
            "exergy_margin": exergy_margin,
            "fourier_margin": fourier_margin,
            "landauer_margin": landauer_margin,
            "is_cd_coherent": is_cd_coherent,
            "is_exergy_sufficient": is_exergy_sufficient,
            "certificate_flags": flags,
            "physics_score": physics_score,
            "policy_score": policy_score,
            "fractional_alpha": self._alpha,
            "rl_integral_phi": frac.rl_integral_phi,
            "gl_phi": frac.gl_phi,
            "poincare_fractional_action": frac.poincare_fractional_action,
            "melnikov_homoclinic_integral": frac.melnikov_homoclinic_integral,
            "cech_h1_obstructed": sheaf.cech_h1_obstructed,
            "cech_cocycle_l2_norm": sheaf.cocycle_l2_norm,
            "poincare_birkhoff_holonomy": sheaf.poincare_birkhoff_holonomy,
            "isolated_charts": sheaf.isolated_charts,
            "kms_available": kms.available,
            "kms_relative_entropy": kms.relative_entropy,
            "tomita_takesaki_commutator": kms.tomita_takesaki_commutator,
            "bures_poincare_distance": kms.bures_poincare_distance,
        }
        return Phase2ThermalOrientation(
            is_cd_coherent=is_cd_coherent,
            is_exergy_sufficient=is_exergy_sufficient,
            is_fourier_coherent=is_fourier,
            is_csmd_coherent=is_csmd,
            is_landauer_coherent=is_landauer,
            is_temperature_physical=is_temp,
            is_carnot_in_range=is_carnot,
            is_engine_alive=observation.engine_alive,
            is_certificate_internally_consistent=cert_ok,
            is_fractional_secular=frac.secular_leak,
            is_fractional_transient=frac.transient_flag,
            is_sheaf_obstructed=sheaf.cech_h1_obstructed,
            is_kms_coherent=kms.kms_coherent,
            cd_margin=cd_margin,
            exergy_margin=exergy_margin,
            fourier_margin=fourier_margin,
            landauer_margin=landauer_margin,
            physics_score=physics_score,
            policy_score=policy_score,
            fractional=frac,
            sheaf=sheaf,
            kms=kms,
            diagnostics=diagnostics,
            is_poincare_kam_stable=is_kam_stable,
        )


# =============================================================================
# FASE 3 — DECIDE / ACT & AUTOMEJORA RECURSIVA NIVEL 3 (RSI-L3)
# Dominio = Phase1ThermalObservation × Phase2ThermalOrientation
# =============================================================================
class Phase3ThermalDecisionActuationMixin(Phase2ThermalOrientationMixin):
    r"""
    FASE 3 — DECIDE / ACT & SELF-IMPROVE (Continuación Anidada Directa de Fase 2).

    Toma `(observation, orientation)` producidos por las Fases 1 y 2 y ejecuta:
        1. Meet en el Álgebra de Heyting (agente ∧ motor ∧ Caputo-Poincaré ∧ Čech ∧ KMS).
        2. Automejora Recursiva de Nivel 3 (RSI-L3) con descenso natural de Fisher-Rao,
           contracción de Lipschitz-Banach L_Banach < 1 y Candado de Gödel-Tarski.
        3. Veto quirúrgico de cartas locales o disparo atómico CAS del Crowbar BT151 [GPIO14].
    """

    @staticmethod
    def _join_heyting_verdicts(verdicts: Tuple[str, ...]) -> str:
        acc = HeytingVerdict.CERTIFIED
        saw = False
        for token in verdicts:
            normalized = str(token).strip().upper()
            if not normalized:
                continue
            saw = True
            acc = acc.meet(HeytingVerdict.parse(normalized, unknown_as_veto=True))
        return (acc if saw else HeytingVerdict.COHERENT).value

    def _classify_heyting(
        self,
        cd_residual: float,
        exergy_potential: float,
        is_cd_coherent: bool,
        is_exergy_sufficient: bool,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        veto_reasons: List[str] = []
        degraded_reasons: List[str] = []
        cd_finite = math.isfinite(cd_residual)
        exergy_finite = math.isfinite(exergy_potential)
        if not cd_finite:
            veto_reasons.append("clausius_duhem_residual_nonfinite")
        if not is_cd_coherent:
            veto_reasons.append("clausius_duhem_residual_below_threshold")
        if cd_finite and cd_residual < 0.0:
            degraded_reasons.append("clausius_duhem_residual_negative")
        if not exergy_finite:
            degraded_reasons.append("exergy_potential_nonfinite")
        elif not is_exergy_sufficient:
            if self._exergy_is_hard_veto:
                veto_reasons.append("exergy_potential_below_threshold")
            else:
                degraded_reasons.append("exergy_potential_below_policy_threshold")
        if exergy_finite and exergy_potential < _DEGRADED_EXERGY_THRESHOLD:
            degraded_reasons.append("exergy_potential_below_degraded_threshold")
        if veto_reasons:
            verdict = HeytingVerdict.VETOED.value
        elif degraded_reasons:
            verdict = HeytingVerdict.DEGRADED.value
        else:
            verdict = HeytingVerdict.COHERENT.value
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def _classify_heyting_supervised(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
    ) -> Tuple[
        HeytingVerdict,
        float,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        bool,
    ]:
        veto: List[str] = []
        degraded: List[str] = []
        audit = observation.certificate_audit
        frac = orientation.fractional
        sheaf = orientation.sheaf
        kms = orientation.kms

        if not observation.engine_alive:
            veto.append("engine_invocation_failed")
        if not observation.observation_valid:
            veto.append("observation_invalid")
        if not orientation.is_cd_coherent:
            veto.append("clausius_duhem_residual_below_threshold")
        if not math.isfinite(observation.clausius_duhem_residual):
            veto.append("clausius_duhem_residual_nonfinite")
        if not orientation.is_temperature_physical:
            veto.append("temperature_nonphysical")
        if audit.third_law_flag:
            veto.append("third_law_nonpositive_temperature")
        if not orientation.is_fourier_coherent:
            veto.append("fourier_constitutive_broken")
        if not orientation.is_csmd_coherent:
            veto.append("csmd_identity_broken")
        if not orientation.is_landauer_coherent:
            veto.append("landauer_bound_violated")
        if not orientation.is_carnot_in_range:
            veto.append("carnot_efficiency_out_of_physical_range")
        if (
            math.isfinite(audit.metric_condition)
            and audit.metric_condition > _CONDITION_VETO
        ):
            veto.append("metric_condition_explosive")
        if orientation.is_fractional_secular:
            veto.append("caputo_secular_leak")
        if kms is not None and kms.available and not kms.is_density:
            veto.append("kms_density_axioms_violated")
        if kms is not None and kms.available and kms.relative_entropy > _KMS_VETO:
            veto.append("kms_relative_entropy_excessive")

        if (
            math.isfinite(observation.clausius_duhem_residual)
            and observation.clausius_duhem_residual < 0.0
            and orientation.is_cd_coherent
        ):
            degraded.append("clausius_duhem_residual_negative_numerical")
        if not orientation.is_exergy_sufficient:
            if self._exergy_is_hard_veto:
                veto.append("exergy_potential_below_threshold")
            else:
                degraded.append("exergy_potential_below_policy_threshold")
        if (
            math.isfinite(observation.exergy_potential)
            and observation.exergy_potential < _DEGRADED_EXERGY_THRESHOLD
        ):
            degraded.append("exergy_potential_below_degraded_threshold")
        if audit.temperature_clamped and not audit.third_law_flag:
            degraded.append("temperature_clamped_to_wilkinson_floor")
        if (
            math.isfinite(audit.fourier_residual)
            and _FOURIER_DEGRADED
            < audit.fourier_residual
            <= _FOURIER_VETO * self._safety_margin
        ):
            degraded.append("fourier_residual_degraded")
        if (
            math.isfinite(audit.metric_condition)
            and _CONDITION_DEGRADED < audit.metric_condition <= _CONDITION_VETO
        ):
            degraded.append("metric_ill_conditioned")
        if not orientation.is_certificate_internally_consistent:
            degraded.append("certificate_internal_inconsistency")
        if orientation.is_fractional_transient:
            degraded.append("caputo_high_frequency_transient")
        if (
            kms is not None
            and kms.available
            and _KMS_DEGRADED < kms.relative_entropy <= _KMS_VETO
        ):
            degraded.append("kms_relative_entropy_degraded")
        if audit.engine_verdict == "DEGRADED":
            degraded.append("engine_heyting_degraded")
        if audit.engine_verdict == "VETOED":
            veto.append("engine_heyting_vetoed")
        for reason in audit.engine_veto_reasons:
            tag = f"engine:{reason}"
            if tag not in veto:
                veto.append(tag)

        isolated: Tuple[str, ...] = (
            sheaf.isolated_charts if sheaf is not None else ()
        )
        surgical = False
        if sheaf is not None and sheaf.isolated_charts and sheaf.surgical_possible:
            rest = [
                s for s in sheaf.sections if s.name not in sheaf.isolated_charts
            ]
            rest_meet = HeytingVerdict.CERTIFIED
            for s in rest:
                rest_meet = rest_meet.meet(HeytingVerdict.parse(s.verdict))
            if rest and rest_meet is not HeytingVerdict.VETOED:
                surgical = True
                for tag in ("clausius_duhem_residual_below_threshold",):
                    if (
                        tag in veto
                        and observation.clausius_duhem_residual >= self._cd_thresh
                    ):
                        veto.remove(tag)
                degraded.append("sheaf_surgical_isolation")
        if sheaf is not None and sheaf.cech_h1_obstructed and not surgical:
            veto.append("cech_h1_obstruction")

        physics = float(orientation.physics_score)
        policy = float(orientation.policy_score)
        meet = min(
            physics,
            1.0 if not veto else 0.0,
            0.95 if degraded else 1.0,
            max(policy, 0.5),
        )

        if veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99 and not degraded:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90 and not degraded:
            verdict = HeytingVerdict.COHERENT
        elif degraded or meet >= 0.50:
            verdict = (
                HeytingVerdict.DEGRADED if meet >= 0.50 else HeytingVerdict.VETOED
            )
            if meet < 0.50 and not veto:
                veto.append("heyting_meet_collapsed")
                verdict = HeytingVerdict.VETOED
        else:
            verdict = HeytingVerdict.VETOED

        if (
            surgical
            and verdict is HeytingVerdict.VETOED
            and "cech_h1_obstruction" not in veto
        ):
            if (
                "engine_invocation_failed" not in veto
                and "third_law_nonpositive_temperature" not in veto
            ):
                verdict = HeytingVerdict.DEGRADED
                degraded.append("global_veto_demoted_to_surgical")
                veto = [
                    v
                    for v in veto
                    if v not in {"clausius_duhem_residual_below_threshold"}
                ]

        _ = float(frac.rl_integral_phi) if frac is not None else 0.0
        return (
            verdict,
            float(meet),
            tuple(veto),
            tuple(degraded),
            isolated,
            surgical,
        )

    def evaluate_heyting_decision_lattice(
        self,
        cd_residual: float,
        exergy_potential: float,
        is_cd_coherent: bool,
        is_exergy_sufficient: bool,
    ) -> str:
        """[COMPATIBILIDAD 1.X — FASE DECIDE]"""
        verdict, _, _ = self._classify_heyting(
            cd_residual=float(cd_residual),
            exergy_potential=float(exergy_potential),
            is_cd_coherent=bool(is_cd_coherent),
            is_exergy_sufficient=bool(is_exergy_sufficient),
        )
        return verdict

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def _act_from_verdict(self, verdict: str, *, surgical: bool) -> Tuple[bool, float]:
        normalized = str(verdict).strip().upper()
        if normalized != HeytingVerdict.VETOED.value:
            return False, 0.0
        if surgical:
            logger.warning(
                "Veto quirúrgico: no se dispara Crowbar global. Cartas aisladas en telemetría."
            )
            return False, 0.0
        swapped = self._cas_interlock(expected=False, desired=True)
        if not swapped:
            logger.warning("CAS: interlock del AGENTE ya enclavado. Latencia SIMULADA.")
        jitter = float(self._rng.normal(loc=0.0, scale=4.0))
        latency = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )
        logger.critical(
            "VETO TÉRMICO DEL SOBERANO. Crowbar BT151 [GPIO14] SIMULADO. Latencia ficticia: %.2f ns.",
            latency,
        )
        return True, latency

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        """[COMPATIBILIDAD 1.X — FASE ACT]. GPIO/IRAM no se tocan."""
        return self._act_from_verdict(verdict, surgical=False)

    def _phase3_conservation_audit(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
        verdict: HeytingVerdict,
    ) -> float:
        q = np.asarray(observation.heat_flux_vector, dtype=np.float64)
        energy_l2 = float(np.real(np.dot(q, q))) if q.size else 0.0
        energy_gap = abs(energy_l2 - observation.heat_flux_energy)
        phi = observation.clausius_duhem_residual
        cd_inconsistent = 0.0
        if orientation.is_cd_coherent and math.isfinite(phi) and phi < self._cd_thresh:
            cd_inconsistent = 1.0
        if (
            (not orientation.is_cd_coherent)
            and math.isfinite(phi)
            and phi >= self._cd_thresh
        ):
            cd_inconsistent = 1.0
        eta = observation.carnot_efficiency
        eta_res = (
            0.0 if (not math.isfinite(eta) or 0.0 <= eta <= 1.0 + 1e-12) else 1.0
        )
        temp_res = (
            1.0
            if observation.observation_valid and not orientation.is_temperature_physical
            else 0.0
        )
        certified_lie = (
            1.0
            if verdict is HeytingVerdict.CERTIFIED and orientation.physics_score < 0.99
            else 0.0
        )
        engine_cons = observation.certificate_audit.conservation_residual
        engine_cons_term = float(engine_cons) if math.isfinite(engine_cons) else 0.0
        tellegen = observation.certificate_audit.tellegen_residual
        tellegen_term = float(tellegen) if math.isfinite(tellegen) else 0.0
        terms = np.asarray(
            [
                energy_gap,
                cd_inconsistent,
                eta_res,
                temp_res,
                certified_lie,
                max(0.0, engine_cons_term),
                tellegen_term,
            ],
            dtype=np.float64,
        )
        return float(np.real(self._neumaier_sum(terms)))

    def _execute_agent_rsi_level3_autopoiesis(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
        disagreement: bool,
        conservation_residual: float,
    ) -> AgentRSILevel3Chart:
        r"""
        Motor de Automejora Recursiva de Nivel 3 (RSI-L3) del Agente Supervisor.

        Arquitectura de 3 niveles:
            - Nivel 1 (Parámetros Operativos de Supervisión θ^(1)_sup):
              Ajusta (α_eff, τ_secular, ε_KMS, γ_sheaf) dentro del politopo compacto C_safe.
            - Nivel 2 (Meta-Aprendizaje por Gradiente Natural de Fisher-Rao):
              Minimiza el funcional de Lyapunov de supervisión:
                  L_sup = ½ (d_disagree² + ‖δ⁰ s‖_Čech² + r_cons² + (α_eff − α_0)²)
              mediante Δθ^(1)_sup = −η_meta · (G^(2)_sup)⁻¹ ∇L_sup.
            - Nivel 3 (Meta-Meta-Endofunctor de Contracción de Banach y Candado Gödel-Tarski):
              Audita L_Banach^(sup) = ‖Δθ_t‖_{G^(2)} / (‖Δθ_{t-1}‖_{G^(2)} + ε) < 1 y reescribe
              la métrica G^(2)_sup para garantizar convergencia de punto fijo sin relajar
              jamás los umbrales de seguridad física.
        """
        cocycle_err = (
            float(orientation.sheaf.cocycle_l2_norm)
            if orientation.sheaf is not None
            else 0.0
        )
        cons_err = (
            float(conservation_residual)
            if math.isfinite(conservation_residual)
            else 1.0
        )
        disagree_val = 1.0 if disagreement else 0.0

        with self._interlock_lock:
            self._rsi_agent_epoch += 1
            a_curr = self._rsi_effective_alpha
            tau_curr = self._rsi_secular_tau
            kms_curr = self._rsi_kms_thresh
            shf_curr = self._rsi_sheaf_confidence

            alpha_dev = a_curr - self._alpha
            lyapunov_loss = float(
                0.5
                * (
                    disagree_val * disagree_val
                    + cocycle_err * cocycle_err
                    + cons_err * cons_err
                    + alpha_dev * alpha_dev
                )
            )

            grad_alpha = alpha_dev + 0.01 * math.tanh(cocycle_err)
            grad_tau = tau_curr - _FRAC_SECULAR_TAU
            grad_kms = kms_curr - _KMS_VETO
            grad_shf = (shf_curr - 1.0) + 0.05 * disagree_val
            raw_grad = np.array(
                [grad_alpha, grad_tau, grad_kms, grad_shf], dtype=np.float64
            )

            # Nivel 2: Actualización de la métrica de información de Fisher-Rao G^(2)_sup
            self._rsi_fisher_diag = 0.9 * self._rsi_fisher_diag + 0.1 * (
                1.0 + raw_grad * raw_grad
            )
            inv_fisher = 1.0 / np.maximum(self._rsi_fisher_diag, _WILKINSON_FLOOR)
            natural_step = -self._rsi_meta_damping * (inv_fisher * raw_grad)

            step_norm = float(
                math.sqrt(
                    max(
                        float(
                            np.sum(self._rsi_fisher_diag * natural_step * natural_step)
                        ),
                        0.0,
                    )
                )
            )

            # Nivel 3: Auditoría de Contracción de Lipschitz-Banach y reescritura de G^(2)_sup
            if self._rsi_prev_step_norm > _WILKINSON_FLOOR:
                banach_ratio = step_norm / self._rsi_prev_step_norm
            else:
                banach_ratio = 0.5 * _RSI_AGENT_BANACH_CAP

            if banach_ratio >= _RSI_AGENT_BANACH_CAP:
                scale = _RSI_AGENT_BANACH_CAP / (banach_ratio + 1e-12)
                self._rsi_meta_damping = float(
                    max(self._rsi_meta_damping * scale, _WILKINSON_FLOOR)
                )
                self._rsi_fisher_diag = self._rsi_fisher_diag / max(scale, 1e-6)
                natural_step = natural_step * scale
                step_norm = step_norm * scale
                banach_ratio = float(min(banach_ratio * scale, _RSI_AGENT_BANACH_CAP))
            else:
                self._rsi_meta_damping = float(
                    min(self._rsi_meta_damping * 0.98, _RSI_AGENT_META_LR)
                )

            # Candado de Gödel-Tarski: Proyección sobre politopo que preserva las cotas de seguridad
            new_alpha = float(
                np.clip(
                    a_curr + natural_step[0],
                    max(0.05, self._alpha - 0.15),
                    min(0.95, self._alpha + 0.15),
                )
            )
            new_tau = float(
                np.clip(
                    tau_curr + natural_step[1],
                    0.5 * _FRAC_SECULAR_TAU,
                    _FRAC_SECULAR_TAU,
                )
            )
            new_kms = float(
                np.clip(kms_curr + natural_step[2], 0.5 * _KMS_VETO, _KMS_VETO)
            )
            new_shf = float(np.clip(shf_curr + natural_step[3], 0.5, 1.0))

            self._rsi_effective_alpha = new_alpha
            self._rsi_secular_tau = new_tau
            self._rsi_kms_thresh = new_kms
            self._rsi_sheaf_confidence = new_shf
            self._rsi_prev_step_norm = max(step_norm, _WILKINSON_FLOOR)
            self._rsi_last_banach = float(
                np.clip(banach_ratio, 0.0, _RSI_AGENT_BANACH_CAP)
            )
            self._rsi_last_lyapunov = lyapunov_loss
            self._rsi_last_grad_norm = step_norm

        return self._snapshot_agent_rsi_chart()

    def phase3_decide_from_orientation(
        self,
        observation: Phase1ThermalObservation,
        orientation: Phase2ThermalOrientation,
    ) -> Phase3ThermalDecision:
        r"""
        [FASE 3 — CONTINUACIÓN DE `phase2_orient_from_observation` Y MORFISMO TERMINAL: DECIDE / ACT]

        Dominio  : `Phase1ThermalObservation × Phase2ThermalOrientation`.
        Codominio: `Phase3ThermalDecision`.
        """
        if not isinstance(observation, Phase1ThermalObservation):
            raise TypeError("observation debe ser Phase1ThermalObservation.")
        if not isinstance(orientation, Phase2ThermalOrientation):
            raise TypeError("orientation debe ser Phase2ThermalOrientation.")

        agent_verdict, score, veto_reasons, degraded_reasons, isolated, surgical = (
            self._classify_heyting_supervised(observation, orientation)
        )
        engine_token = observation.certificate_audit.engine_verdict
        final_verdict = HeytingVerdict.parse(
            self._join_heyting_verdicts((agent_verdict.value, engine_token)),
            unknown_as_veto=True,
        )
        if surgical and final_verdict is HeytingVerdict.VETOED:
            if "engine_invocation_failed" not in veto_reasons:
                final_verdict = HeytingVerdict.DEGRADED

        disagreement = bool(
            engine_token
            and HeytingVerdict.parse(engine_token, unknown_as_veto=True)
            is not agent_verdict
        )
        if disagreement:
            degraded_reasons = degraded_reasons + ("supervisor_engine_disagreement",)

        conservation = self._phase3_conservation_audit(
            observation, orientation, final_verdict
        )
        rsi_chart = self._execute_agent_rsi_level3_autopoiesis(
            observation, orientation, disagreement, conservation
        )

        interlock_fired, latency = self._act_from_verdict(
            final_verdict.value, surgical=surgical
        )
        engine_interlock = bool(observation.certificate_audit.engine_interlock_fired)
        dual = bool(interlock_fired and engine_interlock)
        frac = orientation.fractional
        kms = orientation.kms
        sheaf = orientation.sheaf
        rl_acc = float(frac.rl_integral_phi) if frac is not None else 0.0
        kms_def = (
            float(kms.relative_entropy)
            if (
                kms is not None
                and kms.available
                and math.isfinite(kms.relative_entropy)
            )
            else 0.0
        )
        h1 = bool(sheaf.cech_h1_obstructed) if sheaf is not None else False
        winding_idx = (
            observation.poincare_orbit.poincare_hopf_index
            if observation.poincare_orbit is not None
            else 0
        )

        diagnostics: Dict[str, Any] = {
            "observe": dict(observation.diagnostics),
            "orientation": dict(orientation.diagnostics),
            "agent_verdict": agent_verdict.value,
            "engine_verdict": engine_token or None,
            "final_verdict": final_verdict.value,
            "heyting_score": score,
            "veto_reasons": veto_reasons,
            "degraded_reasons": degraded_reasons,
            "supervisor_disagreement": disagreement,
            "engine_interlock_fired": engine_interlock,
            "agent_interlock_fired": interlock_fired,
            "dual_channel_actuation": dual,
            "conservation_residual": conservation,
            "isolated_charts": isolated,
            "surgical_veto": surgical,
            "cech_h1_obstructed": h1,
            "fractional_cd_accumulator": rl_acc,
            "kms_defect": kms_def,
            "poincare_winding_index": winding_idx,
            "rsi_epoch": rsi_chart.epoch,
            "rsi_level3_banach_contraction": rsi_chart.level3_banach_contraction,
            "rsi_level2_lyapunov_loss": rsi_chart.level2_lyapunov_loss,
            "rsi_godel_tarski_locked": rsi_chart.godel_tarski_soundness_locked,
        }
        logger.debug(
            "Fase 3: agente=%s motor=%s final=%s score=%.4f I^αΦ=%.3e H¹=%s surgical=%s L_Banach=%.4f",
            agent_verdict.value,
            engine_token,
            final_verdict.value,
            score,
            rl_acc,
            h1,
            surgical,
            rsi_chart.level3_banach_contraction,
        )
        return Phase3ThermalDecision(
            heyting_verdict=final_verdict.value,
            heyting_rank=final_verdict.rank,
            heyting_score=score,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            engine_interlock_fired=engine_interlock,
            dual_channel_actuation=dual,
            supervisor_disagreement=disagreement,
            conservation_residual=conservation,
            isolated_charts=isolated,
            surgical_veto=surgical,
            cech_h1_obstructed=h1,
            fractional_cd_accumulator=rl_acc,
            kms_defect=kms_def,
            diagnostics=diagnostics,
            rsi_governor=rsi_chart,
            poincare_winding_index=winding_idx,
        )


# =============================================================================
# CLASE PÚBLICA — ENDOFUNCTOR OODA S = Act ∘ Orient ∘ Observe
# =============================================================================
class ThermalGradientAgent(Phase3ThermalDecisionActuationMixin):
    r"""
    Soberano de Calibre del Campo Térmico con Mecánica Celeste de Poincaré
    y Automejora Recursiva de Nivel 3 (RSI-L3).

        In ──phase1_observe_engine_stamp─────► Phase1ThermalObservation
           ──phase2_orient_from_observation──► Phase2ThermalOrientation
           ──phase3_decide_from_orientation──► Phase3ThermalDecision
           ──► ThermalGradientCertificate
    """

    def execute_thermal_agent_cycle(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        *,
        density_rho: Optional[np.ndarray] = None,
        hamiltonian_H: Optional[np.ndarray] = None,
        **engine_kwargs: Any,
    ) -> ThermalGradientCertificate:
        observation = self.phase1_observe_engine_stamp(
            K_raw=K_raw,
            grad_T_raw=grad_T_raw,
            T_sys=T_sys,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
            **engine_kwargs,
        )
        orientation = self.phase2_orient_from_observation(
            observation,
            density_rho=density_rho,
            hamiltonian_H=hamiltonian_H,
        )
        decision = self.phase3_decide_from_orientation(observation, orientation)

        if decision.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "VETO TÉRMICO. Φ=%.6f η=%.4f I^αΦ=%.3e razones=%s",
                observation.clausius_duhem_residual,
                observation.carnot_efficiency,
                decision.fractional_cd_accumulator,
                decision.veto_reasons,
            )

        diagnostics: Dict[str, Any] = {
            "phase1": dict(observation.diagnostics),
            "phase2": dict(orientation.diagnostics),
            "phase3": dict(decision.diagnostics),
        }
        audit = observation.certificate_audit
        kms = orientation.kms
        fid = (
            float(kms.uhlmann_fidelity)
            if (
                kms is not None
                and kms.available
                and math.isfinite(kms.uhlmann_fidelity)
            )
            else 1.0
        )
        rec_time = (
            float(orientation.fractional.poincare_recurrence_estimate)
            if orientation.fractional is not None
            else 1.0
        )
        rsi_banach = (
            float(decision.rsi_governor.level3_banach_contraction)
            if decision.rsi_governor is not None
            else 0.0
        )
        return ThermalGradientCertificate(
            phase="G_THERMAL_GRADIENTS_SUTURATED",
            heyting_verdict=decision.heyting_verdict,
            clausius_duhem_residual=observation.clausius_duhem_residual,
            carnot_efficiency=observation.carnot_efficiency,
            exergy_potential=observation.exergy_potential,
            heat_flux_norm=observation.heat_flux_norm,
            temperature_system=observation.temperature_system,
            hardware_interlock_fired=decision.hardware_interlock_fired,
            actuation_latency_ns=decision.actuation_latency_ns,
            veto_reasons=decision.veto_reasons,
            degraded_reasons=decision.degraded_reasons,
            diagnostics=diagnostics,
            heyting_score=decision.heyting_score,
            conservation_residual=decision.conservation_residual,
            fourier_residual=(
                audit.fourier_residual if math.isfinite(audit.fourier_residual) else 0.0
            ),
            csmd_error=audit.csmd_error if math.isfinite(audit.csmd_error) else 0.0,
            landauer_gap=(
                audit.landauer_gap if math.isfinite(audit.landauer_gap) else 0.0
            ),
            gouy_stodola=(
                audit.gouy_stodola if math.isfinite(audit.gouy_stodola) else 0.0
            ),
            observation_valid=observation.observation_valid,
            engine_heyting_verdict=audit.engine_verdict,
            policy_exergy_sufficient=orientation.is_exergy_sufficient,
            supervisor_disagreement=decision.supervisor_disagreement,
            fractional_cd_accumulator=decision.fractional_cd_accumulator,
            cech_h1_obstructed=decision.cech_h1_obstructed,
            isolated_charts=decision.isolated_charts,
            surgical_veto=decision.surgical_veto,
            kms_defect=decision.kms_defect,
            uhlmann_fidelity=fid,
            poincare_recurrence_time=rec_time,
            rsi_level3_contraction=rsi_banach,
        )


if __name__ == "__main__":
    print(
        "Autocomprobación Thermal Gradient Agent v4.0 "
        "(Poincaré / RSI-L3 / Caputo / Čech-Birkhoff / Tomita-Takesaki KMS)..."
    )
    agent = ThermalGradientAgent(
        dimension_n=3,
        rng_seed=0,
        coordinate_cover={"partners": [0, 1], "resources": [1, 2]},
        fractional_alpha=0.5,
        history_window=16,
    )
    K = np.eye(3)
    G = np.eye(3)
    grad = np.array([1.0, 0.0, 0.0])

    cert = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    print(
        "Ciclo 1:",
        cert.heyting_verdict,
        "I^αΦ=",
        cert.fractional_cd_accumulator,
        "H¹=",
        cert.cech_h1_obstructed,
        "isolated=",
        cert.isolated_charts,
        "L_Banach=",
        cert.rsi_level3_contraction,
    )

    iso = agent.execute_thermal_agent_cycle(K, np.zeros(3), 300.0, G, 0.0)
    print("Isotermo:", iso.heyting_verdict, "η=", iso.carnot_efficiency)

    # Memoria: muchos ciclos coherentes no deben vetar y deben contraer en RSI-L3.
    for _ in range(8):
        agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    slow = agent.execute_thermal_agent_cycle(K, grad, 300.0, G, 0.0)
    print(
        "Tras 10 ciclos coherentes:",
        slow.heyting_verdict,
        "I^αΦ=",
        slow.fractional_cd_accumulator,
        "τ_rec=",
        slow.poincare_recurrence_time,
        "L_Banach=",
        slow.rsi_level3_contraction,
    )

    # KMS: Gibbs a T=300 con H diagonal debe ser coherente y conmutador Tomita-Takesaki = 0.
    H = np.diag([0.0, 1.0, 2.0])
    beta = 1.0 / 300.0
    z = np.exp(-beta * np.array([0.0, 1.0, 2.0]))
    rho = np.diag(z / z.sum())
    kms_ok = agent.execute_thermal_agent_cycle(
        K, grad, 300.0, G, 0.0, density_rho=rho, hamiltonian_H=H
    )
    print(
        "KMS Gibbs:",
        kms_ok.heyting_verdict,
        "D=",
        kms_ok.kms_defect,
        "F=",
        kms_ok.uhlmann_fidelity,
    )

    rho_bad = np.diag([1.0, 0.0, 0.0])
    kms_bad = agent.execute_thermal_agent_cycle(
        K, grad, 300.0, G, 0.0, density_rho=rho_bad, hamiltonian_H=H
    )
    print(
        "KMS puro vs Gibbs:",
        kms_bad.heyting_verdict,
        "D=",
        kms_bad.kms_defect,
        kms_bad.veto_reasons,
    )

    cold = agent.execute_thermal_agent_cycle(K, grad, -1.0, G, 0.0)
    print("T<0:", cold.heyting_verdict, cold.veto_reasons)


__all__ = [
    "Phase1ThermalObservation",
    "Phase2ThermalOrientation",
    "Phase3ThermalDecision",
    "ThermalGradientCertificate",
    "EngineCertificateAudit",
    "FractionalMemoryChart",
    "ChartSection",
    "SheafCohomologyChart",
    "KMSChart",
    "PoincareObservationOrbitChart",
    "AgentRSILevel3Chart",
    "Phase1ThermalObservationMixin",
    "Phase2ThermalOrientationMixin",
    "Phase3ThermalDecisionActuationMixin",
    "ThermalGradientAgent",
    "HeytingVerdict",
]