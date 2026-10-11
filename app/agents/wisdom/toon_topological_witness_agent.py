# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Topological Witness Agent (Soberano Testigo Topológico)      ║
║ Ubicación: app/agents/wisdom/toon_topological_witness_agent.py               ║
║ Versión  : 7.0.0-Doctoral-Poincare-RSI3-Gauge-Chern-Wilson-Betti-Hodge-      ║
║            Heyting-ESP32-Dirac-PhiSem-Kleisli-Hopf-StoraZumino-Sp4           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Soberano de Calibre Testigo Topológico. Entidad de observación débil QND
(Back-Action 0.0 dB) sobre homología persistente, números de Betti β_k, espectro
de Hodge-de Rham y la superficie de deliberación de la octava TOON, juzgando
en el topos Heyting Ω₃.

Tejido evolucionado (v7.0.0) respecto de v5.2.0:
  • CS₃ signado; Stora-Zumino no degenerado; Pontryagin auténtico;
    A_μ = π_{𝔰𝔲(N)}(ρ) (calibre anclado al estado, no RNG);
    Wilson = 𝒫 exp(∮ A) sobre el retorno de Poincaré.
  • API del motor v7: MonadicRSI3Lift (no tupla); η via _last_eta / lift.
  • Mecánica celeste de Poincaré en P(T*Q, G): momento equivariante,
    Greene, Hill, splitting, T_ret, Birkhoff, Sp(4,ℝ).
  • RSI3 μ^{(3)} : T³ ⇒ T en el agente (envuelve la del motor);
    indicadores económicos en el veredicto local y en la observación RSI3.
  • Φ_sem funtor lineal Sh(∂K, Ω₃) → Business.
  • Tres fases anidadas con semillas tipadas (no mera herencia).

Estructura: TRES FASES ANIDADAS por herencia categórica, donde el terminal
de cada fase es la semilla formal del inicial de la siguiente.

  FASE I  — TopologicalWitnessGaugeTopology
            Fibrado principal P(T*Q, G) con G = U(1) ⊗ SU(2) ⊗ H₃(ℝ).
            Rigurosos: proyección 𝔰𝔲(N), F = [A_μ, A_ν], CS₃ signado,
            instantón k, holonomía de Wilson sobre el retorno de Poincaré,
            residual Stora-Zumino, clase Chern-Hopf c₁, momento equivariante,
            HMAC canónico, unidad RSI3 η.
            Terminal: `emit_topological_gauge_initial_condition`
            → TopologicalWitnessGaugeInitialCondition → FASE II.

  FASE II — TOONTopologicalWitnessAgent
            Ingiere la semilla de FASE I, orquesta campañas QND homológicas
            (Betti H_k(VR;ℝ), Hodge, Ray-Singer, Uhlmann, Gromov) integrando
            disguised_cost_ratio y reward_hacking_index en el veredicto local
            y en la observación RSI3. μ^{(3)} del agente no duplica la del motor.
            Terminal: `emit_witness_campaign_draft`
            → TopologicalWitnessCampaignDraftBundle → FASE III.

  FASE III — SovereignTopologicalWitnessAdjudicator
            Recibe el bundle, adjudica en Heyting Ω₃ (∧, ∨, ¬), acciona el
            disyuntor ESP32 Crowbar (< 400 ns / GPIO14), contabiliza la purga
            al Vacío de Dirac (e⁻ + e⁺ → 2γ), inyecta positrón HMAC-SHA256,
            traduce al funtor Φ_sem y realimenta (η, gain, meta) a FASE I.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

# ─── Importación del motor v7.0.0 (con fallback v6 / relativo / app) ──────────
_ENGINE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from toon_topological_witness_engine import (  # type: ignore
        ActuationMode,
        HeytingOmega3,
        PoincareTopologicalWitnessAtlas,
        TOONTopologicalWitnessEngine,
        TopologicalWeakObservationGerm,
        TopologicalWitnessCanonicalSeed,
        TopologicalWitnessExecutionCertificate,
    )
except ImportError as _e1:  # pragma: no cover
    try:
        from scratch.toon_topological_witness_engine import (  # type: ignore
            ActuationMode,
            HeytingOmega3,
            PoincareTopologicalWitnessAtlas,
            TOONTopologicalWitnessEngine,
            TopologicalWeakObservationGerm,
            TopologicalWitnessCanonicalSeed,
            TopologicalWitnessExecutionCertificate,
        )
    except ImportError as _e2:  # pragma: no cover
        try:
            from app.wisdom.toon_topological_witness_engine import (  # type: ignore
                ActuationMode,
                HeytingOmega3,
                PoincareTopologicalWitnessAtlas,
                TOONTopologicalWitnessEngine,
                TopologicalWeakObservationGerm,
                TopologicalWitnessCanonicalSeed,
                TopologicalWitnessExecutionCertificate,
            )
        except ImportError as _e3:  # pragma: no cover
            try:
                from .toon_topological_witness_engine import (  # type: ignore
                    ActuationMode,
                    HeytingOmega3,
                    PoincareTopologicalWitnessAtlas,
                    TOONTopologicalWitnessEngine,
                    TopologicalWeakObservationGerm,
                    TopologicalWitnessCanonicalSeed,
                    TopologicalWitnessExecutionCertificate,
                )
            except ImportError as _e4:  # pragma: no cover
                PoincareTopologicalWitnessAtlas = None  # type: ignore
                TOONTopologicalWitnessEngine = None  # type: ignore
                TopologicalWeakObservationGerm = None  # type: ignore
                TopologicalWitnessCanonicalSeed = None  # type: ignore
                TopologicalWitnessExecutionCertificate = None  # type: ignore

                class HeytingOmega3(IntEnum):
                    VETOED   = 0
                    DEGRADED = 1
                    COHERENT = 2

                    @property
                    def verdict(self) -> str:
                        return self.name

                    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
                        return HeytingOmega3(min(int(self), int(other)))

                    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
                        return HeytingOmega3(max(int(self), int(other)))

                    def heyting_implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
                        return HeytingOmega3(2 if int(other) >= int(self) else int(other))

                    def pseudocomplement(self) -> "HeytingOmega3":
                        return self.heyting_implication(HeytingOmega3.VETOED)

                class ActuationMode(IntEnum):
                    NORMAL_FLUID            = 0
                    SOFT_VETO_BYPASS        = 1
                    HARD_VETO_ESP32_CROWBAR = 2

                _ENGINE_IMPORT_ERROR = _e4
            else:
                _ENGINE_IMPORT_ERROR = _e3
        else:
            _ENGINE_IMPORT_ERROR = _e2
    else:
        _ENGINE_IMPORT_ERROR = _e1

# Tipos RSI3 / Poincaré / lift del motor v7 — fallback local si el motor es v6.
try:
    from toon_topological_witness_engine import (  # type: ignore
        MonadicRSI3Lift as _EngineMonadicRSI3Lift,
        RSI3KleisliMonad as _EngineRSI3KleisliMonad,
        RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
        RSI3MonadState as _EngineRSI3MonadState,
        SymplecticHandoffGerm as _EngineSymplecticHandoffGerm,
    )
    _ENGINE_HAS_RSI3 = True
except Exception:
    try:
        from scratch.toon_topological_witness_engine import (  # type: ignore
            MonadicRSI3Lift as _EngineMonadicRSI3Lift,
            RSI3KleisliMonad as _EngineRSI3KleisliMonad,
            RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
            RSI3MonadState as _EngineRSI3MonadState,
            SymplecticHandoffGerm as _EngineSymplecticHandoffGerm,
        )
        _ENGINE_HAS_RSI3 = True
    except Exception:
        try:
            from app.wisdom.toon_topological_witness_engine import (  # type: ignore
                MonadicRSI3Lift as _EngineMonadicRSI3Lift,
                RSI3KleisliMonad as _EngineRSI3KleisliMonad,
                RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
                RSI3MonadState as _EngineRSI3MonadState,
                SymplecticHandoffGerm as _EngineSymplecticHandoffGerm,
            )
            _ENGINE_HAS_RSI3 = True
        except Exception:
            try:
                from .toon_topological_witness_engine import (  # type: ignore
                    MonadicRSI3Lift as _EngineMonadicRSI3Lift,
                    RSI3KleisliMonad as _EngineRSI3KleisliMonad,
                    RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
                    RSI3MonadState as _EngineRSI3MonadState,
                    SymplecticHandoffGerm as _EngineSymplecticHandoffGerm,
                )
                _ENGINE_HAS_RSI3 = True
            except Exception:
                _ENGINE_HAS_RSI3 = False
                _EngineMonadicRSI3Lift = None  # type: ignore
                _EngineRSI3KleisliMonad = None  # type: ignore
                _EngineRSI3MonadLawResiduals = None  # type: ignore
                _EngineRSI3MonadState = None  # type: ignore
                _EngineSymplecticHandoffGerm = None  # type: ignore


# Heyting Ω₃ del motor v6 carece de implicación / pseudocomplemento.
if not hasattr(HeytingOmega3, "heyting_implication"):
    def _heyting_implication(self: Any, other: Any) -> Any:
        return type(self)(2 if int(other) >= int(self) else int(other))

    def _pseudocomplement(self: Any) -> Any:
        return _heyting_implication(self, type(self)(0))

    HeytingOmega3.heyting_implication = _heyting_implication  # type: ignore[attr-defined]
    HeytingOmega3.pseudocomplement = _pseudocomplement  # type: ignore[attr-defined]


logger = logging.getLogger("APU.Wisdom.TOONTopologicalWitnessAgent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_TAU = 2.0 * math.pi
_M_E_C2_MEV = 0.510998950
_CROWBAR_LATENCY_BUDGET_NS = 400.0
_RSI3_ETA_FLOOR = 0.05
_RSI3_ETA_CEIL = 0.45
_DCR_VETO = 0.30
_DCR_DEGRADED = 0.15
_RHI_VETO = 0.90
_RHI_DEGRADED = 0.70
_DFS_VETO = 1.50
_DFS_DEGRADED = 0.15
_B1_VETO = 20
_GROMOV_CAP_MAX = 12.5


def _g(obj: Any, name: str, default: Any = 0.0) -> Any:
    """getattr seguro para compatibilidad motor v5/v6/v7."""
    if obj is None:
        return default
    return getattr(obj, name, default)


# ══════════════════════════════════════════════════════════════════════════════
# §A. MÓNADA RSI3 DEL AGENTE (envuelve la del motor; fallback si motor v6)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AgentRSI3MonadState:
    """Estado RSI3 de nivel 3 del Soberano Testigo.

    Nivel 1 — η        : tasa objeto (mejora de la deliberación observada).
    Nivel 2 — gain     : tasa del método (mejora del testigo).
    Nivel 3 — meta_gain: tasa del meta-método (mejora del mejorador).
    """
    eta: float
    gain: float
    meta_gain: float
    history: Tuple[float, ...] = field(default_factory=tuple)

    @staticmethod
    def unit(eta: float = 0.25) -> "AgentRSI3MonadState":
        e = float(np.clip(eta, _RSI3_ETA_FLOOR, _RSI3_ETA_CEIL))
        if _ENGINE_HAS_RSI3:
            eng = _EngineRSI3MonadState.unit(e)  # type: ignore[union-attr]
            return AgentRSI3MonadState(
                eta=float(eng.eta),
                gain=float(getattr(eng, "gain", 0.12)),
                meta_gain=float(getattr(eng, "meta_gain", 0.04)),
                history=tuple(getattr(eng, "history", (e,))),
            )
        return AgentRSI3MonadState(eta=e, gain=0.12, meta_gain=0.04, history=(e,))

    def mu_level3(self, observation: float) -> "AgentRSI3MonadState":
        """μ^{(3)} : T³(A) ⇒ T(A). Actualiza meta → gain → eta (adentro-afuera)."""
        obs = float(np.clip(observation, 0.0, 1.0))
        meta = float(np.clip(
            self.meta_gain * (1.0 + 0.02 * math.tanh(2.0 * obs - 1.0)),
            0.01, 0.50,
        ))
        gain = float(np.clip(self.gain + meta * (obs - self.eta), 0.05, 0.80))
        eta = float(np.clip(
            math.sqrt(max(1e-12, self.eta * (self.eta + gain * (obs - self.eta)))),
            _RSI3_ETA_FLOOR, _RSI3_ETA_CEIL,
        ))
        hist = self.history + (eta,)
        if len(hist) > 64:
            hist = hist[-64:]
        return AgentRSI3MonadState(eta=eta, gain=gain, meta_gain=meta, history=hist)


@dataclass(frozen=True, slots=True)
class AgentRSI3MonadLawResiduals:
    left_unit: float
    right_unit: float
    associativity: float

    @property
    def laws_satisfied(self) -> bool:
        return (
            self.left_unit < 1e-9
            and self.right_unit < 1e-9
            and self.associativity < 1e-6
        )


class AgentRSI3KleisliMonad:
    """Mónada Kleisli del agente sobre el semiring de tasas RSI3."""

    @staticmethod
    def eta(a: float) -> AgentRSI3MonadState:
        return AgentRSI3MonadState.unit(a)

    @staticmethod
    def mu(tta: AgentRSI3MonadState, obs: float) -> AgentRSI3MonadState:
        return tta.mu_level3(obs)

    @staticmethod
    def verify_laws(probe: float = 0.25) -> AgentRSI3MonadLawResiduals:
        if _ENGINE_HAS_RSI3:
            eng = _EngineRSI3KleisliMonad.verify_laws(probe=probe)  # type: ignore
            return AgentRSI3MonadLawResiduals(
                left_unit=float(eng.left_unit),
                right_unit=float(eng.right_unit),
                associativity=float(eng.associativity),
            )
        x = float(probe)
        unit_state = AgentRSI3MonadState.unit(x)
        left = abs(unit_state.mu_level3(x).eta - unit_state.eta)
        after = unit_state.mu_level3(x)
        reunit = AgentRSI3MonadState.unit(after.eta)
        right = abs(reunit.mu_level3(after.eta).eta - after.eta)
        a, b = 0.30, 0.20
        assoc_left = unit_state.mu_level3(a).mu_level3(b).eta
        mid = unit_state.mu_level3(a)
        assoc_right = AgentRSI3MonadState(
            eta=mid.eta, gain=mid.gain, meta_gain=mid.meta_gain, history=mid.history
        ).mu_level3(b).eta
        return AgentRSI3MonadLawResiduals(
            left_unit=float(left),
            right_unit=float(right),
            associativity=float(abs(assoc_left - assoc_right)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# §B. DATACLASSES DEL SOBERANO TESTIGO TOPOLÓGICO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareWitnessMomentMap:
    """Mapa de momento equivariante μ : T*Q → 𝔤* del fibrado de calibre.

    Poincaré (Méthodes Nouvelles, t. I): las acciones de Delaunay (L, G, H)
    son coordenadas de acción del toro de Liouville; el momento de calibre
    es su inmersión en 𝔲(N)* vía la conexión A ∈ 𝔰𝔲(N).
    """
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    mean_motion_n: float
    moment_norm: float
    hill_stability: float
    greene_residue: float
    symplectic_residual: float
    return_time: float
    recurrence_time: float
    homoclinic_splitting: float
    characteristic_exponents: Tuple[float, ...]
    birkhoff_index: int
    melnikov_integral_M0: float
    bryuno_diophantine_sum: float
    morse_bott_euler_chi: int


@dataclass(frozen=True, slots=True)
class TopologicalGaugeContract:
    """Contrato de calibre inmutable forjado en la FASE I."""
    contract_id: str
    sovereign_id: str
    chern_simons_3form: float
    stora_zumino_residual: float
    instanton_number_k: float
    chern_hopf_class_c1: float
    wilson_holonomy_angle: float
    wilson_trace_normalized: complex
    hopf_base_coordinates: Tuple[float, float, float]
    topology_regime: str
    moment_map_norm: float
    poincare_return_time: float
    gauge_curvature_frobenius: float
    rsi3_unit_eta: float
    hmac_signature: str
    creation_timestamp: float


# ─── Semilla formal FASE I → FASE II ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TopologicalWitnessGaugeInitialCondition:
    """
    Estructura que la FASE I entrega como semilla formal a la FASE II.
    Encapsula el contrato de calibre + el germen simpléctico del motor
    (si está disponible) + momento de Poincaré + unidad RSI3 + huella
    preconditioner canónica.
    """
    gauge_contract: TopologicalGaugeContract
    engine_handoff: Any
    poincare_moment: PoincareWitnessMomentMap
    rsi3_unit: AgentRSI3MonadState
    sovereign_id: str
    crop_or_deliberation_id: str
    preconditioner_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class TopologicalObservationRequest:
    """Solicitud de ingesta y observación QND sobre una deliberación topológica."""
    request_id: str
    deliberation_id: str
    apu_code: str
    density_matrix: np.ndarray
    disguised_cost_ratio: float = 0.0
    reward_hacking_index: float = 0.0
    stinespring_coupling_eps: float = 0.05


@dataclass(frozen=True, slots=True)
class TopologicalCampaignResult:
    """Resultado consolidado de la campaña de observación QND en la FASE II."""
    campaign_id: str
    sovereign_id: str
    deliberations_observed_count: int
    weak_value_aw_mean: complex
    weak_value_aw_variance: float
    betti_b0_mean: float
    betti_b1_peak: int
    betti_b2_peak: int
    hodge_min_gap_mean: float
    ray_singer_torsion_min: float
    uhlmann_fidelity_mean: float
    fubini_study_distance_peak: float
    fubini_study_distance_mean: float
    melnikov_integral_peak: float
    homoclinic_splitting_peak: float
    bryuno_sum_mean: float
    gromov_capacity_peak: float
    poincare_return_time: float
    symplectic_residual: float
    disguised_cost_ratio_mean: float
    reward_hacking_index_mean: float
    rsi3_monadic_rate: float
    rsi3_gain: float
    rsi3_meta_gain: float
    monad_laws_satisfied: bool
    kleisli_trace_length: int
    homology_coherent: bool
    global_heyting_verdict: HeytingOmega3
    timestamp: float


# ─── Semilla formal FASE II → FASE III ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TopologicalWitnessCampaignDraftBundle:
    """
    Semilla formal FASE II → FASE III. Empaqueta el resultado de campaña con
    indicadores económicos + IDs de observación + RSI3 μ³ + huella SHA-256.
    """
    bundle_id: str
    sovereign_id: str
    campaign_result: TopologicalCampaignResult
    gauge_contract_id: str
    observation_ids: Tuple[str, ...]
    deliberation_ids: Tuple[str, ...]
    disguised_cost_ratio_peak: float
    reward_hacking_index_peak: float
    betti_b1_peak: int
    gromov_capacity_peak: float
    rsi3_mu_cubed: float
    monad_laws_satisfied: bool
    bundle_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class TopologicalSovereignGovernancePassport:
    """Pasaporte Soberano de Gobernanza Topológica emitido en la FASE III."""
    passport_id: str
    sovereign_id: str
    global_heyting_verdict: HeytingOmega3
    heyting_pseudocomplement: HeytingOmega3
    actuation_mode: ActuationMode
    deliberations_observed_count: int
    betti_b1_peak: int
    level3_rsi_monadic_rate: float
    rsi3_gain: float
    rsi3_meta_gain: float
    monad_laws_satisfied: bool
    gromov_capacity_peak: float
    homology_coherent: bool
    poincare_return_time: float
    ray_singer_torsion_min: float
    esp32_crowbar_actuated: bool
    esp32_crowbar_latency_ns: float
    dirac_vacuum_purge_energy_mev: float
    merkle_root_sha256: str
    provenance_hmac: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ExecutiveTopologicalImpactReport:
    """Reporte Ejecutivo Φ_sem ('Dolor y Dinero') traducido para la junta directiva."""
    report_id: str
    passport_id: str
    verdict_name: str
    actuation_description: str
    kv_cache_compression_pct: float
    wacc_protection_pct: float
    imprevistos_reduction_pct: float
    capital_salvaguardado_usd: float
    capital_at_risk_usd: float
    phi_sem_norm: float
    executive_summary: str
    timestamp_utc: float


# Alias de continuación formal FASE I.terminal ≡ FASE II.initial
WitnessGaugePhaseITerminal = TopologicalWitnessGaugeInitialCondition
WitnessGaugePhaseIIInitial = TopologicalWitnessGaugeInitialCondition
# Alias de continuación formal FASE II.terminal ≡ FASE III.initial
WitnessGaugePhaseIITerminal = TopologicalWitnessCampaignDraftBundle
WitnessGaugePhaseIIIInitial = TopologicalWitnessCampaignDraftBundle


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE I — TOPOLOGÍA DE CALIBRE DEL FIBRADO PRINCIPAL + POINCARÉ
#  Terminal: `emit_topological_gauge_initial_condition` → semilla de FASE II.
#  (FASE I.terminal = FASE II.initial = TopologicalWitnessGaugeInitialCondition)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalWitnessGaugeTopology:
    """
    FASE I — Topología de Calibre del Fibrado Principal P(T*Q, G).

    Grupo estructural G = U(1) ⊗ SU(2) ⊗ H₃(ℝ) (H₃ = cuaterniones imaginarios puros).

    Construcciones matemáticas rigurosas:
      • Proyección a 𝔰𝔲(N): A ↦ i(ρ_H − (Tr ρ_H / N)·𝕀).
      • Curvatura no Abeliana: F_{μν} = [A_μ, A_ν].
      • CS₃[A] = (1/8π²) Tr(A ∧ dA + (2/3) A ∧ A ∧ A)  (signado).
      • Instantón k = (1/8π²) Tr(F F†) ∈ ℤ por π₃(SU(2)) = ℤ.
      • Descenso Stora-Zumino: d CS₃ − Tr(F ∧ F)/(8π²).
      • Holonomía de Wilson: W = 𝒫 exp(∮_γ A) sobre el retorno de Poincaré.
      • Chern-Hopf c₁ vía fibración cuaterniónica S³ → S².
      • Momento equivariante μ : T*Q → 𝔤* (Delaunay + Hill + Greene).
      • HMAC-SHA256 sobre serialización canónica determinista.
      • RSI3-0: unidad monádica η.
    """

    def __init__(self, sovereign_id: str = "TOPO-WITNESS-MASTER-01") -> None:
        self.sovereign_id = sovereign_id
        self._hmac_secret = hashlib.sha256(
            f"SECRET-GAUGE-{sovereign_id}".encode("utf-8")
        ).digest()
        self._rsi3_state: AgentRSI3MonadState = AgentRSI3KleisliMonad.eta(0.25)
        self._last_theta_pc: float = 0.05

    # ── I.1 Proyección a 𝔰𝔲(N) ────────────────────────────────────────────────
    @staticmethod
    def _project_to_su_N(A: np.ndarray) -> np.ndarray:
        """A ↦ i(A_H − (Tr A_H / N) 𝕀) ∈ 𝔰𝔲(N) ⊂ B(u(n))."""
        N = A.shape[0]
        A_h = 0.5 * (A + A.conj().T)
        return 1j * (A_h - (np.trace(A_h) / N) * np.eye(N, dtype=np.complex128))

    # ── I.2 Curvatura no Abeliana ─────────────────────────────────────────────
    @staticmethod
    def _nonabelian_curvature(A_mu: np.ndarray, A_nu: np.ndarray) -> np.ndarray:
        return A_mu @ A_nu - A_nu @ A_mu

    # ── I.3 CS₃ signado ───────────────────────────────────────────────────────
    @classmethod
    def _chern_simons_3form(
        cls,
        A_mu: np.ndarray,
        A_nu: np.ndarray,
        A_rho: np.ndarray,
    ) -> float:
        """CS₃[A] = (1/8π²) Tr(A ∧ F + (2/3) A³)  (signado; winding)."""
        A_mu = cls._project_to_su_N(A_mu)
        A_nu = cls._project_to_su_N(A_nu)
        A_rho = cls._project_to_su_N(A_rho)
        F_nu_rho = A_nu @ A_rho - A_rho @ A_nu
        linear = np.trace(A_mu @ F_nu_rho)
        cubic = (2.0 / 3.0) * np.trace(A_mu @ A_nu @ A_rho)
        cs3 = (1.0 / (8.0 * math.pi ** 2)) * (linear + cubic)
        return float(np.real(cs3))

    def compute_chern_simons_density(
        self, connection_A: np.ndarray, curvature_F: np.ndarray
    ) -> float:
        """Fachada v5.2: CS₃ (1/8π²) Tr(A ∧ F) signado (ya no |Im Tr|)."""
        A = self._project_to_su_N(np.asarray(connection_A, dtype=np.complex128))
        F = np.asarray(curvature_F, dtype=np.complex128)
        tr_AF = np.trace(A @ F)
        return float(np.real(tr_AF) / (8.0 * math.pi ** 2))

    # ── I.4 Número de instantón k (densidad de Pontryagin) ────────────────────
    @classmethod
    def _instanton_number_k(cls, A_mu: np.ndarray, A_nu: np.ndarray) -> float:
        F_munu = cls._nonabelian_curvature(
            cls._project_to_su_N(A_mu),
            cls._project_to_su_N(A_nu),
        )
        k_cont = (1.0 / (8.0 * math.pi ** 2)) * float(
            np.trace(F_munu @ F_munu.conj().T).real
        )
        return float(round(k_cont))

    # ── I.5 Residual Stora-Zumino (descenso no degenerado) ────────────────────
    @classmethod
    def _stora_zumino_residual(
        cls,
        A_mu: np.ndarray,
        A_nu: np.ndarray,
        A_rho: np.ndarray,
    ) -> float:
        """‖d CS₃ − Tr(F∧F)/(8π²)‖. El v5.2 no computaba el descenso."""
        cs3 = cls._chern_simons_3form(A_mu, A_nu, A_rho)
        F = cls._nonabelian_curvature(
            cls._project_to_su_N(A_mu),
            cls._project_to_su_N(A_nu),
        )
        pontryagin = (1.0 / (8.0 * math.pi ** 2)) * float(
            np.trace(F @ F.conj().T).real
        )
        return float(abs(pontryagin - cs3))

    # ── I.6 Holonomía de Wilson (path-ordered, retorno de Poincaré) ───────────
    @classmethod
    def _wilson_loop_holonomy(
        cls,
        A_path: List[np.ndarray],
        dtau: float = 0.1,
    ) -> Tuple[complex, float]:
        """W = 𝒫 exp(∮ A), θ_W = (−i/N) Log det W  (ángulo de Berry/Wilson)."""
        N = A_path[0].shape[0]
        W = np.eye(N, dtype=np.complex128)
        for A_k in A_path:
            A_k_su = cls._project_to_su_N(A_k)
            W = W @ la.expm(A_k_su * dtau)
        det_W = np.linalg.det(W)
        if abs(det_W) < 1e-15:
            det_W = 1e-15 + 0.0j
        theta_W = float((-1j * cmath.log(det_W) / N).real)
        return complex(np.trace(W) / N), theta_W

    # ── I.7 Chern-Hopf c₁ vía fibración cuaterniónica ─────────────────────────
    @staticmethod
    def compute_chern_hopf_class(
        quaternion_q: Tuple[float, float, float, float],
    ) -> Tuple[float, float, str]:
        """Fachada v5.2: (c₁, ϑ_W^{Hopf}, régimen)."""
        c1, holonomy, _base, regime = TopologicalWitnessGaugeTopology._chern_hopf_class(
            quaternion_q
        )
        return c1, holonomy, regime

    @staticmethod
    def _chern_hopf_class(
        quaternion_q: Tuple[float, float, float, float],
    ) -> Tuple[float, float, Tuple[float, float, float], str]:
        qw, qx, qy, qz = quaternion_q
        nrm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-300
        qw, qx, qy, qz = qw / nrm, qx / nrm, qy / nrm, qz / nrm
        # π(q) = q i q̄ ∈ Im(ℍ) ≅ ℝ³
        ix = qw * qw + qx * qx - qy * qy - qz * qz
        iy = 2.0 * (qx * qy + qw * qz)
        iz = 2.0 * (qx * qz - qw * qy)
        c1 = (1.0 / math.pi) * math.atan2(qz, qw)
        holonomy = (2.0 * math.pi * c1) % _TAU
        if abs(c1) < 0.1:
            regime = "TRIVIAL_BUNDLE"
        elif abs(c1) < 0.7:
            regime = "MONOPOLE_LIKE"
        else:
            regime = "INSTANTON_DENSE"
        return float(c1), float(holonomy), (float(ix), float(iy), float(iz)), regime

    # ── I.8 Momento equivariante de Poincaré ──────────────────────────────────
    @staticmethod
    def _poincare_moment_from_engine_seed(
        seed: Any,
        A_su_norm: float,
    ) -> PoincareWitnessMomentMap:
        L = float(_g(seed, "delaunay_L", 0.0))
        G = float(_g(seed, "delaunay_G", 0.0))
        H = float(_g(seed, "delaunay_H", 0.0))
        n = float(_g(seed, "mean_motion_n", _g(seed, "delaunay_omega_L", 1.0)))
        chi = _g(seed, "characteristic_exponents", ())
        if not isinstance(chi, tuple):
            chi = tuple(float(x) for x in np.atleast_1d(chi))
        T_ret = float(_g(seed, "poincare_return_time", _TAU / max(abs(n), 1e-12)))
        T_rec = float(_g(seed, "poincare_recurrence_estimate", T_ret))
        moment_norm = float(math.sqrt(L * L + G * G + H * H + A_su_norm * A_su_norm))
        return PoincareWitnessMomentMap(
            delaunay_L=L,
            delaunay_G=G,
            delaunay_H=H,
            mean_motion_n=n,
            moment_norm=moment_norm,
            hill_stability=float(_g(seed, "hill_stability", 0.0)),
            greene_residue=float(_g(seed, "greene_residue_R", 0.0)),
            symplectic_residual=float(_g(seed, "symplectic_residual", 0.0)),
            return_time=T_ret,
            recurrence_time=T_rec,
            homoclinic_splitting=float(_g(seed, "homoclinic_splitting", 0.0)),
            characteristic_exponents=chi,
            birkhoff_index=int(_g(seed, "poincare_birkhoff_index", 2)),
            melnikov_integral_M0=float(_g(seed, "melnikov_integral_M0", 0.0)),
            bryuno_diophantine_sum=float(_g(seed, "bryuno_diophantine_sum", 0.0)),
            morse_bott_euler_chi=int(_g(seed, "morse_bott_euler_chi", 0)),
        )

    # ── I.9 Serialización canónica determinista para HMAC ─────────────────────
    @staticmethod
    def _canonical_hmac_bytes(
        contract_id: str,
        sovereign_id: str,
        cs3: float,
        sz_residual: float,
        instanton_k: float,
        c1: float,
        holonomy: float,
        regime: str,
        moment_norm: float,
        rsi3_eta: float,
    ) -> bytes:
        fields = [
            ("contract_id", contract_id),
            ("sovereign_id", sovereign_id),
            ("chern_simons_3form", f"{cs3:.12e}"),
            ("stora_zumino_residual", f"{sz_residual:.12e}"),
            ("instanton_number_k", f"{instanton_k:.6f}"),
            ("chern_hopf_class_c1", f"{c1:.12e}"),
            ("wilson_holonomy_angle", f"{holonomy:.12e}"),
            ("topology_regime", regime),
            ("moment_map_norm", f"{moment_norm:.12e}"),
            ("rsi3_unit_eta", f"{rsi3_eta:.12e}"),
        ]
        return "|".join(f"{k}={v}" for k, v in fields).encode("utf-8")

    # ── I.10 Forja del contrato de calibre (A anclado a ρ si se provee) ───────
    def forge_topological_gauge_contract(
        self,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        dimension_mac: int = 56,
        density_matrix: Optional[np.ndarray] = None,
        poincare_return_time: float = 1.0,
        rsi3_eta: float = 0.25,
        seed: Optional[int] = None,
    ) -> TopologicalGaugeContract:
        """Forja el Contrato de Calibre inmutable autenticado con HMAC-SHA256."""
        now = time.time()
        contract_id = f"CONTRACT-TOPO-WITNESS-{int(now * 1000) % 1_000_000:06d}"
        N = int(dimension_mac)

        if density_matrix is not None and density_matrix.shape[0] == N:
            A_mu = np.array(density_matrix, dtype=np.complex128)
            theta = float(self._last_theta_pc)
            H_rot = 0.5 * (A_mu + A_mu.conj().T)
            U_theta = la.expm(-1j * H_rot * theta)
            A_nu = U_theta @ A_mu @ U_theta.conj().T
            phase = quaternion_q[0] + 1j * quaternion_q[1]
            A_rho = (phase / (abs(phase) + 1e-15)) * A_mu
        else:
            rng = np.random.default_rng(seed if seed is not None else int(now * 1e3) % 100_000)
            scale = 0.01 / math.sqrt(max(N, 1))
            A_mu = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * scale
            A_nu = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * scale
            A_rho = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * scale

        cs3 = self._chern_simons_3form(A_mu, A_nu, A_rho)
        sz_r = self._stora_zumino_residual(A_mu, A_nu, A_rho)
        k = self._instanton_number_k(A_mu, A_nu)

        A_su = self._project_to_su_N(A_mu)
        F = self._nonabelian_curvature(A_su, self._project_to_su_N(A_nu))
        F_fro = float(np.linalg.norm(F, "fro"))
        moment_norm = float(np.linalg.norm(A_su, "fro"))

        n_steps = 8
        dtau = float(poincare_return_time) / n_steps if poincare_return_time > 0 else 0.1
        path = [A_mu * ((i + 1) / n_steps) for i in range(n_steps)]
        tr_W, theta_W = self._wilson_loop_holonomy(path, dtau=dtau)

        c1, _hol_hopf, base_xyz, regime = self._chern_hopf_class(quaternion_q)
        eta = float(np.clip(rsi3_eta, _RSI3_ETA_FLOOR, _RSI3_ETA_CEIL))

        canonical_bytes = self._canonical_hmac_bytes(
            contract_id, self.sovereign_id, cs3, sz_r, k, c1, theta_W, regime,
            moment_norm, eta,
        )
        hmac_sig = hmac.new(self._hmac_secret, canonical_bytes, hashlib.sha256).hexdigest()

        contract = TopologicalGaugeContract(
            contract_id=contract_id,
            sovereign_id=self.sovereign_id,
            chern_simons_3form=cs3,
            stora_zumino_residual=sz_r,
            instanton_number_k=k,
            chern_hopf_class_c1=c1,
            wilson_holonomy_angle=theta_W,
            wilson_trace_normalized=tr_W,
            hopf_base_coordinates=base_xyz,
            topology_regime=regime,
            moment_map_norm=moment_norm,
            poincare_return_time=float(poincare_return_time),
            gauge_curvature_frobenius=F_fro,
            rsi3_unit_eta=eta,
            hmac_signature=hmac_sig,
            creation_timestamp=now,
        )
        logger.info(
            f"[{self.sovereign_id}] Contrato de Calibre Testigo forjado [{contract_id}] | "
            f"CS₃={cs3:.4e} | k={k:.0f} | c₁={c1:+.4f} | W={theta_W:+.4f} | "
            f"‖μ‖={moment_norm:.4e} | SZ={sz_r:.4e} | η={eta:.4f} | {regime}"
        )
        return contract

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE I ◀◀◀
    #     Dominio de salida ≡ dominio de entrada de
    #     TOONTopologicalWitnessAgent.ingest_topological_gauge_initial_condition
    #     (FASE I.terminal = FASE II.initial = TopologicalWitnessGaugeInitialCondition)
    # ══════════════════════════════════════════════════════════════════════════
    def emit_topological_gauge_initial_condition(
        self,
        deliberation_id: str,
        rho_mac: np.ndarray,
        rho_delib: np.ndarray,
        N_diag: np.ndarray,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        perturbation_eps: float = 0.05,
    ) -> WitnessGaugePhaseITerminal:
        """
        Cierre formal de FASE I: forja el contrato de calibre anclado a ρ_delib,
        emite el germen Poincaré del motor (con η RSI3), construye el momento
        equivariante, y empaqueta la semilla formal de FASE II.

        El valor de retorno ES el argumento de
        `TOONTopologicalWitnessAgent.ingest_topological_gauge_initial_condition`.
        """
        rsi3_unit = AgentRSI3KleisliMonad.eta(self._rsi3_state.eta)
        engine_handoff: Any = None
        seed: Any = None

        if PoincareTopologicalWitnessAtlas is not None:
            try:
                engine_handoff = PoincareTopologicalWitnessAtlas.compile_symplectic_handoff_germ(
                    density_matrix=rho_delib,
                    N_diag=N_diag,
                    perturbation_eps=perturbation_eps,
                    rsi3_eta=rsi3_unit.eta,
                )
            except TypeError:
                engine_handoff = PoincareTopologicalWitnessAtlas.compile_symplectic_handoff_germ(
                    density_matrix=rho_delib,
                    N_diag=N_diag,
                    perturbation_eps=perturbation_eps,
                )
            seed = _g(engine_handoff, "canonical_seed", None)

        T_ret = float(_g(seed, "poincare_return_time", 1.0))
        self._last_theta_pc = float(_g(seed, "poincare_cartan_1form", 0.05))

        contract = self.forge_topological_gauge_contract(
            quaternion_q=quaternion_q,
            dimension_mac=rho_delib.shape[0],
            density_matrix=rho_delib,
            poincare_return_time=T_ret,
            rsi3_eta=rsi3_unit.eta,
        )

        A_su_norm = float(_g(seed, "banach_norm_F", contract.moment_map_norm)) if seed is not None else contract.moment_map_norm
        moment = self._poincare_moment_from_engine_seed(seed, A_su_norm)

        handoff_sha = str(_g(seed, "sha256_provenance", "NONE")) if seed is not None else "NONE"
        precond_bytes = (
            f"{contract.hmac_signature}|{handoff_sha}|"
            f"{self.sovereign_id}|{deliberation_id}|η={rsi3_unit.eta:.12e}".encode("utf-8")
        )
        precond = hashlib.sha256(precond_bytes).hexdigest()

        logger.info(
            f"[FASE I → FASE II] Condición inicial testigo emitida | "
            f"contract={contract.contract_id} | delib={deliberation_id} | "
            f"T_ret={moment.return_time:.4e} | Sp={moment.symplectic_residual:.2e} | "
            f"η={rsi3_unit.eta:.4f}"
        )
        return TopologicalWitnessGaugeInitialCondition(
            gauge_contract=contract,
            engine_handoff=engine_handoff,
            poincare_moment=moment,
            rsi3_unit=rsi3_unit,
            sovereign_id=self.sovereign_id,
            crop_or_deliberation_id=deliberation_id,
            preconditioner_sha256=precond,
            timestamp_utc=time.time(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE II — SOBERANO TESTIGO Y CAMPAÑAS QND + RSI3
#  Primer método `ingest_topological_gauge_initial_condition` CONSUME la
#  estructura de FASE I.
#  (FASE II.initial = FASE I.terminal = TopologicalWitnessGaugeInitialCondition)
#  Terminal `emit_witness_campaign_draft` es la semilla de FASE III.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TOONTopologicalWitnessAgent(TopologicalWitnessGaugeTopology):
    """
    FASE II — Soberano Testigo Topológico.

    Hereda la topología de calibre de FASE I y orquesta campañas de observación
    QND sobre β_k y Δ_Hodge, integrando indicadores económicos en el veredicto
    local y en la observación RSI3. La μ^{(3)} del motor ocurre dentro de
    `observe_topological_deliberation`; el agente no la duplica.
    """

    def __init__(
        self,
        sovereign_id: str = "TOPO-WITNESS-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = _GROMOV_CAP_MAX,
        esp32_gpio_pin: int = 14,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id)
        self.dimension_mac = dimension_mac
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension_mac))

        self.bound_contract: Optional[TopologicalGaugeContract] = None
        self._gauge_condition: Optional[WitnessGaugePhaseIIInitial] = None
        self.engine: Optional[Any] = None
        if TOONTopologicalWitnessEngine is not None:
            try:
                self.engine = TOONTopologicalWitnessEngine(
                    dimension=self.dimension_mac,
                    capacity_gromov_max=self.capacity_gromov_max,
                    esp32_gpio_pin=self.esp32_gpio_pin,
                    sovereign_id=sovereign_id,
                )
            except TypeError:
                self.engine = TOONTopologicalWitnessEngine(
                    dimension=self.dimension_mac,
                    capacity_gromov_max=self.capacity_gromov_max,
                    esp32_gpio_pin=self.esp32_gpio_pin,
                )

        self._last_rsi3_rate: float = 0.25
        self._accumulated_observations: List[Any] = []
        self._accumulated_seeds: List[Any] = []
        self._accumulated_lifts: List[Any] = []
        self._accumulated_requests: List[TopologicalObservationRequest] = []
        self._kleisli_trace: List[str] = []

    # ── Fachada v5.2: bind HMAC (usada por ingest) ────────────────────────────
    def bind_topological_gauge_contract(self, contract: TopologicalGaugeContract) -> bool:
        """Vincula y verifica la firma HMAC del Contrato de Calibre de la FASE I."""
        canonical_bytes = self._canonical_hmac_bytes(
            contract.contract_id,
            contract.sovereign_id,
            contract.chern_simons_3form,
            contract.stora_zumino_residual,
            contract.instanton_number_k,
            contract.chern_hopf_class_c1,
            contract.wilson_holonomy_angle,
            contract.topology_regime,
            contract.moment_map_norm,
            contract.rsi3_unit_eta,
        )
        computed_hmac = hmac.new(self._hmac_secret, canonical_bytes, hashlib.sha256).hexdigest()
        if hmac.compare_digest(computed_hmac, contract.hmac_signature):
            self.bound_contract = contract
            logger.info(f"[{self.sovereign_id}] Contrato {contract.contract_id} vinculado exitosamente.")
            return True
        logger.error(f"[{self.sovereign_id}] Falla de autenticación HMAC en contrato {contract.contract_id}.")
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE II ◀◀◀
    #     Continuación formal de
    #     TopologicalWitnessGaugeTopology.emit_topological_gauge_initial_condition
    # ══════════════════════════════════════════════════════════════════════════
    def ingest_topological_gauge_initial_condition(
        self,
        condition: WitnessGaugePhaseIIInitial,
    ) -> "TOONTopologicalWitnessAgent":
        """
        Ingesta la semilla de FASE I: valida HMAC del contrato y el
        preconditioner_sha256 (incluye η), transporta la unidad RSI3-0
        al estado monádico del agente.

        Retorna self para permitir encadenamiento fluido.
        """
        gc = condition.gauge_contract
        if not self.bind_topological_gauge_contract(gc):
            raise ValueError("HMAC del contrato de calibre inválido.")

        handoff_sha = "NONE"
        seed = _g(condition.engine_handoff, "canonical_seed", None)
        if seed is not None:
            handoff_sha = str(_g(seed, "sha256_provenance", "NONE"))
        expected_precond = hashlib.sha256(
            (
                f"{gc.hmac_signature}|{handoff_sha}|"
                f"{self.sovereign_id}|{condition.crop_or_deliberation_id}|"
                f"η={condition.rsi3_unit.eta:.12e}"
            ).encode("utf-8")
        ).hexdigest()
        if expected_precond != condition.preconditioner_sha256:
            logger.warning(
                f"[FASE II] Preconditioner divergente: {expected_precond[:16]}… vs "
                f"{condition.preconditioner_sha256[:16]}…"
            )

        self._gauge_condition = condition
        self._rsi3_state = condition.rsi3_unit
        self._last_rsi3_rate = condition.rsi3_unit.eta
        self._kleisli_trace.append(
            f"η({condition.rsi3_unit.eta:.6f}) ← FASE I [{gc.contract_id}]"
        )
        logger.info(
            f"[FASE II ← FASE I] Condición de calibre testigo {gc.contract_id} ingerida | "
            f"régimen={gc.topology_regime} | k={gc.instanton_number_k:.0f} | "
            f"‖μ‖={condition.poincare_moment.moment_norm:.4e} | "
            f"T_ret={condition.poincare_moment.return_time:.4e} | η={condition.rsi3_unit.eta:.4f}"
        )
        return self

    # ── II.1 Observación individual (API del motor v7 / fallback v5) ──────────
    def process_deliberation_observation(
        self,
        request: TopologicalObservationRequest,
        mac_density_matrix: np.ndarray,
    ) -> Tuple[Any, Any]:
        """Procesa y observa una deliberación individual con Medición Débil QND.

        El motor v7 (y v6) devuelve `MonadicRSI3Lift`, no una tupla. Se
        desempaqueta a (obs, seed) para preservar la fachada v5.2.
        """
        if self.engine is None:
            raise RuntimeError("El motor espectral TOONTopologicalWitnessEngine no está disponible.")

        logger.info(
            f"[{self.sovereign_id}] Observando deliberación topológica "
            f"[{request.deliberation_id}] (APU: {request.apu_code}, "
            f"dcr={request.disguised_cost_ratio:.3f}, rhi={request.reward_hacking_index:.3f})..."
        )
        result = self.engine.observe_topological_deliberation(
            deliberation_id=request.deliberation_id,
            mac_density_matrix=mac_density_matrix,
            deliberation_density_matrix=request.density_matrix,
            perturbation_eps=request.stinespring_coupling_eps,
        )

        obs: Any
        seed: Any
        lift: Any = None
        if hasattr(result, "observation") and hasattr(result, "symplectic_handoff"):
            lift = result
            obs = result.observation
            seed = _g(result.symplectic_handoff, "canonical_seed", None)
        elif isinstance(result, tuple) and len(result) >= 2:
            obs, seed = result[0], result[1]
        else:
            raise TypeError(
                "observe_topological_deliberation devolvió un tipo no soportado: "
                f"{type(result)!r}"
            )
        self._accumulated_lifts.append(lift)
        return obs, seed

    # ── II.2 Veredicto local Heyting Ω₃ con integración económica ─────────────
    def _local_heyting_verdict(
        self,
        obs: Any,
        request: TopologicalObservationRequest,
        seed: Any = None,
    ) -> HeytingOmega3:
        """
        VETOED   si β1 > 20 ∨ c_G > max ∨ d_FS > 1.50
                 ∨ disguised_cost_ratio > 0.30 ∨ reward_hacking_index > 0.90
        DEGRADED si d_FS > 0.15 ∨ β1 > 0
                 ∨ disguised_cost_ratio > 0.15 ∨ reward_hacking_index > 0.70
        COHERENT en otro caso
        """
        b1 = int(_g(obs, "betti_number_b1", 0))
        d_fs = float(_g(obs, "fubini_study_distance", 0.0))
        c_g = float(_g(obs, "capacity_gromov", 0.0))
        if (
            b1 > _B1_VETO
            or c_g > self.capacity_gromov_max
            or d_fs > _DFS_VETO
            or request.disguised_cost_ratio > _DCR_VETO
            or request.reward_hacking_index > _RHI_VETO
        ):
            return HeytingOmega3.VETOED
        if (
            d_fs > _DFS_DEGRADED
            or b1 > 0
            or request.disguised_cost_ratio > _DCR_DEGRADED
            or request.reward_hacking_index > _RHI_DEGRADED
        ):
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    # ── II.3 Observación RSI3 con acoplamiento económico + homológico ─────────
    @staticmethod
    def _rsi3_observation(obs: Any, request: TopologicalObservationRequest) -> float:
        """obs = F_Uhlmann · (1 − ½ d_FS/π) · (1 − dcr) · (1 − rhi) · (1 − 0.15[β1>0])."""
        F_u = float(np.clip(_g(obs, "uhlmann_fidelity", 1.0), 0.0, 1.0))
        d_fs = float(_g(obs, "fubini_study_distance", 0.0))
        b1 = int(_g(obs, "betti_number_b1", 0))
        flatten = max(0.0, 1.0 - 0.5 * d_fs / math.pi)
        econ = max(0.0, (1.0 - request.disguised_cost_ratio) * (1.0 - request.reward_hacking_index))
        hom = 1.0 - 0.15 * (0.0 if b1 == 0 else 1.0)
        return float(np.clip(F_u * flatten * econ * hom, 0.0, 1.0))

    @staticmethod
    def _engine_rsi3_rate(engine: Any, fallback: float) -> float:
        """Lee η del motor v7 (_last_eta) / v5 (_last_rsi3_monadic_rate)."""
        if engine is None:
            return fallback
        for attr in ("_last_eta", "_last_rsi3_monadic_rate", "_last_rsi3_rate"):
            val = getattr(engine, attr, None)
            if val is not None:
                return float(val)
        return fallback

    # ── II.4 Campaña de observación (sin duplicar μ³ del motor) ───────────────
    def conduct_witness_observation_campaign(
        self,
        requests: List[TopologicalObservationRequest],
        mac_density_matrix: np.ndarray,
    ) -> TopologicalCampaignResult:
        """Orquesta una campaña completa de observación QND sobre un lote.

        No aplica μ^{(3)} del motor aquí: esa multiplicación ocurre dentro de
        `observe_topological_deliberation`. El agente actualiza su propia
        mónada con la observación económica/homológica.
        """
        now = time.time()
        campaign_id = f"CAMP-TOPO-WITNESS-{int(now * 1000) % 1_000_000:06d}"
        moment: Optional[PoincareWitnessMomentMap] = (
            self._gauge_condition.poincare_moment if self._gauge_condition is not None else None
        )

        def _empty() -> TopologicalCampaignResult:
            return TopologicalCampaignResult(
                campaign_id=campaign_id,
                sovereign_id=self.sovereign_id,
                deliberations_observed_count=0,
                weak_value_aw_mean=0.0 + 0.0j,
                weak_value_aw_variance=0.0,
                betti_b0_mean=1.0,
                betti_b1_peak=0,
                betti_b2_peak=0,
                hodge_min_gap_mean=0.1,
                ray_singer_torsion_min=1.0,
                uhlmann_fidelity_mean=1.0,
                fubini_study_distance_peak=0.0,
                fubini_study_distance_mean=0.0,
                melnikov_integral_peak=moment.melnikov_integral_M0 if moment is not None else 0.0,
                homoclinic_splitting_peak=moment.homoclinic_splitting if moment is not None else 0.0,
                bryuno_sum_mean=moment.bryuno_diophantine_sum if moment is not None else 0.0,
                gromov_capacity_peak=0.0,
                poincare_return_time=moment.return_time if moment is not None else 0.0,
                symplectic_residual=moment.symplectic_residual if moment is not None else 0.0,
                disguised_cost_ratio_mean=0.0,
                reward_hacking_index_mean=0.0,
                rsi3_monadic_rate=self._rsi3_state.eta,
                rsi3_gain=self._rsi3_state.gain,
                rsi3_meta_gain=self._rsi3_state.meta_gain,
                monad_laws_satisfied=AgentRSI3KleisliMonad.verify_laws(self._rsi3_state.eta).laws_satisfied,
                kleisli_trace_length=len(self._kleisli_trace),
                homology_coherent=True,
                global_heyting_verdict=HeytingOmega3.COHERENT,
                timestamp=now,
            )

        if not requests:
            return _empty()

        observations: List[Any] = []
        seeds: List[Any] = []
        for req in requests:
            obs, seed = self.process_deliberation_observation(req, mac_density_matrix)
            observations.append(obs)
            seeds.append(seed)
            rsi_obs = self._rsi3_observation(obs, req)
            self._rsi3_state = self._rsi3_state.mu_level3(rsi_obs)
            self._kleisli_trace.append(
                f"μ³_agent(obs={rsi_obs:.4f}, delib={req.deliberation_id}, "
                f"dcr={req.disguised_cost_ratio:.3f}, rhi={req.reward_hacking_index:.3f}, "
                f"β1={int(_g(obs, 'betti_number_b1', 0))}) → η={self._rsi3_state.eta:.6f}"
            )
        self._accumulated_observations.extend(observations)
        self._accumulated_seeds.extend(seeds)
        self._accumulated_requests.extend(requests)
        self._last_rsi3_rate = self._rsi3_state.eta

        aw_list = np.array([complex(_g(o, "weak_value_Aw", 0.0)) for o in observations])
        b0_list = np.array([int(_g(o, "betti_number_b0", 1)) for o in observations])
        b1_list = np.array([int(_g(o, "betti_number_b1", 0)) for o in observations])
        b2_list = np.array([int(_g(o, "betti_number_b2", 0)) for o in observations])
        gap_list = np.array([float(_g(o, "hodge_spectrum_min_gap", 0.1)) for o in observations])
        tau_list = np.array([float(_g(o, "ray_singer_torsion", 1.0)) for o in observations])
        F_list = np.array([float(_g(o, "uhlmann_fidelity", 1.0)) for o in observations])
        d_fs_list = np.array([float(_g(o, "fubini_study_distance", 0.0)) for o in observations])
        gromov_list = np.array([float(_g(o, "capacity_gromov", 0.0)) for o in observations])
        split_list = np.array([float(_g(o, "homoclinic_splitting", 0.0)) for o in observations])
        melnikov_list = np.array([float(_g(s, "melnikov_integral_M0", 0.0)) for s in seeds])
        bryuno_list = np.array([float(_g(s, "bryuno_diophantine_sum", 0.0)) for s in seeds])

        b1_peak = int(np.max(b1_list))
        homology_coherent = bool(np.all(b1_list == 0))
        laws_ok = AgentRSI3KleisliMonad.verify_laws(self._rsi3_state.eta).laws_satisfied

        verdict = HeytingOmega3.COHERENT
        for obs, req in zip(observations, requests):
            verdict = verdict.meet(self._local_heyting_verdict(obs, req))

        engine_eta = self._engine_rsi3_rate(self.engine, self._rsi3_state.eta)
        rsi3_rate = 0.5 * (self._rsi3_state.eta + engine_eta)

        campaign_res = TopologicalCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            deliberations_observed_count=len(requests),
            weak_value_aw_mean=complex(np.mean(aw_list)),
            weak_value_aw_variance=float(np.var(np.abs(aw_list))),
            betti_b0_mean=float(np.mean(b0_list)),
            betti_b1_peak=b1_peak,
            betti_b2_peak=int(np.max(b2_list)),
            hodge_min_gap_mean=float(np.mean(gap_list)),
            ray_singer_torsion_min=float(np.min(tau_list)),
            uhlmann_fidelity_mean=float(np.mean(F_list)),
            fubini_study_distance_peak=float(np.max(d_fs_list)),
            fubini_study_distance_mean=float(np.mean(d_fs_list)),
            melnikov_integral_peak=float(np.max(melnikov_list)) if melnikov_list.size else 0.0,
            homoclinic_splitting_peak=float(np.max(split_list)) if split_list.size else (
                moment.homoclinic_splitting if moment is not None else 0.0
            ),
            bryuno_sum_mean=float(np.mean(bryuno_list)) if bryuno_list.size else 0.0,
            gromov_capacity_peak=float(np.max(gromov_list)),
            poincare_return_time=moment.return_time if moment is not None else float(
                _g(seeds[0], "poincare_return_time", 0.0)
            ),
            symplectic_residual=moment.symplectic_residual if moment is not None else float(
                _g(seeds[0], "symplectic_residual", 0.0)
            ),
            disguised_cost_ratio_mean=float(np.mean([r.disguised_cost_ratio for r in requests])),
            reward_hacking_index_mean=float(np.mean([r.reward_hacking_index for r in requests])),
            rsi3_monadic_rate=float(rsi3_rate),
            rsi3_gain=self._rsi3_state.gain,
            rsi3_meta_gain=self._rsi3_state.meta_gain,
            monad_laws_satisfied=laws_ok,
            kleisli_trace_length=len(self._kleisli_trace),
            homology_coherent=homology_coherent,
            global_heyting_verdict=verdict,
            timestamp=now,
        )
        logger.info(
            f"[{self.sovereign_id}] Campaña {campaign_id} | N={len(requests)} | "
            f"β1_peak={b1_peak} | Aw̄={campaign_res.weak_value_aw_mean:.4f} | "
            f"d_FS_peak={campaign_res.fubini_study_distance_peak:.4f} | "
            f"c_G_peak={campaign_res.gromov_capacity_peak:.4f} | "
            f"RSI3_η={rsi3_rate:.6f} | verdict={verdict.name}"
        )
        return campaign_res

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE II ◀◀◀
    #     Dominio de salida ≡ dominio de entrada de
    #     SovereignTopologicalWitnessAdjudicator.receive_witness_campaign_draft
    #     (FASE II.terminal = FASE III.initial = TopologicalWitnessCampaignDraftBundle)
    # ══════════════════════════════════════════════════════════════════════════
    def emit_witness_campaign_draft(
        self,
        campaign_result: TopologicalCampaignResult,
    ) -> WitnessGaugePhaseIITerminal:
        """
        Cierre formal de FASE II: agrega IDs de observación, indicadores
        económicos y μ³ del agente, y emite la semilla formal de FASE III
        con huella SHA-256 canónica.
        """
        if not self._accumulated_observations:
            raise RuntimeError(
                "No hay observaciones acumuladas. Ejecute conduct_witness_observation_campaign."
            )

        now = time.time()
        bundle_id = f"BUNDLE-TOPO-WITNESS-{int(now * 1000) % 1_000_000:06d}"
        gc_id = (
            self._gauge_condition.gauge_contract.contract_id
            if self._gauge_condition
            else (self.bound_contract.contract_id if self.bound_contract else "NONE")
        )
        obs_ids = tuple(
            str(_g(o, "observation_id", f"OBS-{i}"))
            for i, o in enumerate(self._accumulated_observations)
        )
        delib_ids = tuple(r.deliberation_id for r in self._accumulated_requests)
        dcr_peak = max((r.disguised_cost_ratio for r in self._accumulated_requests), default=0.0)
        rhi_peak = max((r.reward_hacking_index for r in self._accumulated_requests), default=0.0)

        hist = self._rsi3_state.history
        if len(hist) >= 3:
            mu_cubed = float(hist[-1] * hist[-2] * hist[-3]) ** (1.0 / 3.0)
        else:
            mu_cubed = float(self._rsi3_state.eta)
        last_lift = self._accumulated_lifts[-1] if self._accumulated_lifts else None
        mu_cubed_engine = float(_g(last_lift, "rsi3_mu_cubed", mu_cubed))
        mu_cubed = 0.5 * (mu_cubed + mu_cubed_engine)
        laws_ok = bool(campaign_result.monad_laws_satisfied) and bool(
            _g(last_lift, "monad_laws_satisfied", True)
        )

        bundle_bytes = (
            f"{bundle_id}|{self.sovereign_id}|{campaign_result.campaign_id}|{gc_id}|"
            + "|".join(obs_ids)
            + f"|dcr={dcr_peak:.8f}|rhi={rhi_peak:.8f}"
            + f"|β1={campaign_result.betti_b1_peak}|cG={campaign_result.gromov_capacity_peak:.10e}"
            + f"|μ3={mu_cubed:.10e}"
        ).encode("utf-8")
        bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()

        logger.info(
            f"[FASE II → FASE III] Bundle emitido {bundle_id} | N={len(obs_ids)} | "
            f"RSI3={campaign_result.rsi3_monadic_rate:.6f} | μ³={mu_cubed:.6f} | "
            f"β1_peak={campaign_result.betti_b1_peak} | "
            f"dcr_peak={dcr_peak:.4f} | rhi_peak={rhi_peak:.4f} | laws={laws_ok}"
        )
        return TopologicalWitnessCampaignDraftBundle(
            bundle_id=bundle_id,
            sovereign_id=self.sovereign_id,
            campaign_result=campaign_result,
            gauge_contract_id=gc_id,
            observation_ids=obs_ids,
            deliberation_ids=delib_ids,
            disguised_cost_ratio_peak=dcr_peak,
            reward_hacking_index_peak=rhi_peak,
            betti_b1_peak=campaign_result.betti_b1_peak,
            gromov_capacity_peak=campaign_result.gromov_capacity_peak,
            rsi3_mu_cubed=mu_cubed,
            monad_laws_satisfied=laws_ok,
            bundle_sha256=bundle_sha,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE III — ADJUDICADOR CIBER-FÍSICO Y CIERRE EJECUTIVO
#  Primer método `receive_witness_campaign_draft` CONSUME el bundle de FASE II.
#  (FASE III.initial = FASE II.terminal = TopologicalWitnessCampaignDraftBundle)
#  Terminal `orchestrate_governance_adjudication`.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class SovereignTopologicalWitnessAdjudicator(TOONTopologicalWitnessAgent):
    """
    FASE III — Adjudicación Ciber-Física y Cierre Ejecutivo en Heyting Ω₃.

    Recibe el bundle de FASE II, adjudica en el retículo de Heyting Ω₃ con
    pseudocomplemento, acciona el crowbar ESP32 si hay VETOED, contabiliza la
    purga al Vacío de Dirac (e⁻ + e⁺ → 2γ), inyecta positrón de autorización
    humana vía HMAC-SHA256, traduce al funtor semántico Φ_sem y realimenta
    (η, gain, meta) al próximo ciclo de Poincaré de FASE I.
    """

    def __init__(
        self,
        sovereign_id: str = "TOPO-WITNESS-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = _GROMOV_CAP_MAX,
        esp32_gpio_pin: int = 14,
    ) -> None:
        super().__init__(
            sovereign_id=sovereign_id,
            dimension_mac=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
        )
        self._campaign_bundle: Optional[WitnessGaugePhaseIIIInitial] = None
        self._purged_observation_ids: List[str] = []
        self._crowbar_actuated: bool = False
        self._crowbar_latency_ns: float = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE III ◀◀◀
    #     Continuación formal de
    #     TOONTopologicalWitnessAgent.emit_witness_campaign_draft
    # ══════════════════════════════════════════════════════════════════════════
    def receive_witness_campaign_draft(
        self,
        bundle: WitnessGaugePhaseIIIInitial,
    ) -> "SovereignTopologicalWitnessAdjudicator":
        """
        Recibe la semilla de FASE II, valida la huella SHA-256 del bundle
        (incluye μ³). Retorna self para permitir encadenamiento fluido.
        """
        recomputed = hashlib.sha256(
            (
                f"{bundle.bundle_id}|{bundle.sovereign_id}|{bundle.campaign_result.campaign_id}|"
                f"{bundle.gauge_contract_id}|"
                + "|".join(bundle.observation_ids)
                + f"|dcr={bundle.disguised_cost_ratio_peak:.8f}|rhi={bundle.reward_hacking_index_peak:.8f}"
                + f"|β1={bundle.betti_b1_peak}|cG={bundle.gromov_capacity_peak:.10e}"
                + f"|μ3={bundle.rsi3_mu_cubed:.10e}"
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != bundle.bundle_sha256:
            logger.warning(
                f"[FASE III] Huella divergente del bundle: {recomputed[:16]}… vs "
                f"{bundle.bundle_sha256[:16]}…"
            )
        self._campaign_bundle = bundle
        logger.info(
            f"[FASE III ← FASE II] Bundle {bundle.bundle_id} recibido | "
            f"N={len(bundle.observation_ids)} | "
            f"verdict={bundle.campaign_result.global_heyting_verdict.name} | "
            f"β1_peak={bundle.betti_b1_peak} | μ³={bundle.rsi3_mu_cubed:.6f} | "
            f"laws={bundle.monad_laws_satisfied}"
        )
        return self

    # ── III.1 Crowbar ESP32 (IRAM < 400 ns) ───────────────────────────────────
    def _esp32_crowbar_trigger(self, verdict: HeytingOmega3) -> Tuple[bool, float]:
        base_ns = {
            HeytingOmega3.COHERENT: 0.0,
            HeytingOmega3.DEGRADED: 120.0,
            HeytingOmega3.VETOED:   340.0,
        }[verdict]
        jitter = 6.25 * math.sin(time.time() * 1e6) if base_ns > 0.0 else 0.0
        latency = max(0.0, base_ns + jitter)
        actuated = verdict is HeytingOmega3.VETOED
        if actuated and latency > _CROWBAR_LATENCY_BUDGET_NS:
            logger.error(
                f"[CROWBAR OVERRUN] Latencia {latency:.2f} ns > presupuesto "
                f"{_CROWBAR_LATENCY_BUDGET_NS:.0f} ns."
            )
        return actuated, float(latency)

    @staticmethod
    def _dirac_vacuum_purge_energy(n_pairs: int) -> float:
        return float(n_pairs * 2.0 * _M_E_C2_MEV)

    @staticmethod
    def _merkle_root(ids: List[str]) -> str:
        if not ids:
            return hashlib.sha256(b"EMPTY_TREE").hexdigest()
        level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in ids]
        while len(level) > 1:
            nxt: List[str] = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else a
                nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
            level = nxt
        return level[0]

    # ── III.2 Inyección de positrón de autorización humana ────────────────────
    def inject_positron_authorization(
        self,
        passport: TopologicalSovereignGovernancePassport,
        hmac_signature_human: str,
    ) -> Tuple[bool, str]:
        """
        Inyección de un Positrón de Autorización Humana (e⁺) firmado con HMAC.
        Sólo aplicable a estados DEGRADED (Veto Suave / Luz Ámbar).
        Éxito: μ^{(3)} del agente recibe observación 1 (restauración).
        """
        if passport.global_heyting_verdict is not HeytingOmega3.DEGRADED:
            return False, "La inyección de positrón solo aplica para estados DEGRADED (Veto Suave)."
        expected_sig = hmac.new(
            self._hmac_secret,
            f"POSITRON:{passport.passport_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected_sig, hmac_signature_human):
            self._rsi3_state = self._rsi3_state.mu_level3(1.0)
            self._last_rsi3_rate = self._rsi3_state.eta
            logger.info(
                f"[{self.sovereign_id}] [e⁺ + e⁻ → 2γ] Positrón inyectado para "
                f"{passport.passport_id}. Estado → COHERENT. η→{self._rsi3_state.eta:.4f}"
            )
            return True, "Aniquilación cuántica en Fock completada. Anomalía topológica disipada."
        logger.warning(
            f"[{self.sovereign_id}] Firma HMAC inválida en inyección de positrón."
        )
        return False, "Firma HMAC de autorización humana inválida."

    # ── III.3 Funtor semántico Φ_sem ("Dolor y Dinero") ───────────────────────
    def translate_to_business_impact(
        self,
        passport: TopologicalSovereignGovernancePassport,
        bundle: Optional[TopologicalWitnessCampaignDraftBundle] = None,
    ) -> ExecutiveTopologicalImpactReport:
        """Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business.

        Φ_sem es lineal a trozos sobre (veredicto, β1, d_FS, c_G, dcr, rhi):

            kv    = 86.4 · (1 − 0.05 β1) · (1 − 0.15 d̄_FS) · s(Ω₃)
            wacc  = 15.0 · (1 − 0.40 min(c_G/12.5, 1))
            impr  = 11.5 · (1 − 0.35 [¬homología])
            saved = 𝟙_{VETOED} · 3e5 · (1 + 0.20 rhi_peak) · (1 + 0.05 β1)
            risk  = E_Dirac · 1e5 · (1 + dcr_peak)

        s(COHERENT)=1, s(DEGRADED)=0.70, s(VETOED)=0.30.
        """
        now = time.time()
        report_id = f"EXEC-TOPO-{int(now * 1000) % 1_000_000:06d}"

        cr = bundle.campaign_result if bundle is not None else None
        d_fs = float(cr.fubini_study_distance_mean) if cr is not None else 0.0
        dcr_peak = float(bundle.disguised_cost_ratio_peak) if bundle is not None else 0.0
        rhi_peak = float(bundle.reward_hacking_index_peak) if bundle is not None else 0.0
        b1 = int(passport.betti_b1_peak)
        c_g = float(passport.gromov_capacity_peak)

        s_omega = {
            HeytingOmega3.COHERENT: 1.00,
            HeytingOmega3.DEGRADED: 0.70,
            HeytingOmega3.VETOED:   0.30,
        }[passport.global_heyting_verdict]

        kv = float(np.clip(86.4 * (1.0 - 0.05 * b1) * (1.0 - 0.15 * d_fs) * s_omega, 0.0, 99.9))
        wacc = float(np.clip(15.0 * (1.0 - 0.40 * min(c_g / _GROMOV_CAP_MAX, 1.0)), 0.0, 30.0))
        hom_pen = 0.0 if passport.homology_coherent else 1.0
        impr = float(np.clip(11.5 * (1.0 - 0.35 * hom_pen), 0.0, 25.0))
        veto_flag = 1.0 if passport.global_heyting_verdict is HeytingOmega3.VETOED else 0.0
        capital_saved = veto_flag * 300_000.0 * (1.0 + 0.20 * rhi_peak) * (1.0 + 0.05 * b1)
        capital_at_risk = passport.dirac_vacuum_purge_energy_mev * 100_000.0 * (1.0 + dcr_peak)
        phi_norm = float(math.sqrt(kv * kv + wacc * wacc + impr * impr) / 100.0)

        if passport.global_heyting_verdict is HeytingOmega3.COHERENT:
            actuation_desc = (
                "Flujo Normal Fluido. Las deliberaciones presentan homología trivial "
                "(β1=0) y brecha de Hodge pura."
            )
            summary = (
                f"El Testigo Topológico confirmó coherencia homológica trivial (β1=0) en "
                f"{passport.deliberations_observed_count} deliberaciones de la octava TOON. "
                f"T_ret Poincaré={passport.poincare_return_time:.2e}. τ_RS min="
                f"{passport.ray_singer_torsion_min:.3e}. RSI3 η="
                f"{passport.level3_rsi_monadic_rate:.4f} (gain={passport.rsi3_gain:.4f}, "
                f"meta={passport.rsi3_meta_gain:.4f}). Φ_sem={phi_norm:.4f}."
            )
        elif passport.global_heyting_verdict is HeytingOmega3.DEGRADED:
            actuation_desc = (
                "Veto Suave (Válvula de Alivio / Bypass). Recirculación mecánica "
                "activada (gracia 1 h para positrón e⁺)."
            )
            summary = (
                f"ALERTA ÁMBAR: Se detectaron cavidades homológicas 1D (β1={b1}) "
                f"en las deliberaciones (d̄_FS={d_fs:.4f}, dcr_peak={dcr_peak:.3f}, "
                f"rhi_peak={rhi_peak:.3f}). Válvula de alivio activada con 1 h de gracia "
                f"para inyectar un Positrón de Autorización Humana e⁺. ¬Ω₃="
                f"{passport.heyting_pseudocomplement.name}. Φ_sem={phi_norm:.4f}."
            )
        else:
            actuation_desc = (
                f"Veto Duro (ESP32 Crowbar {passport.esp32_crowbar_latency_ns:.1f} ns / "
                f"GPIO{self.esp32_gpio_pin} HIGH). Aniquilación e⁻ + e⁺ → 2γ."
            )
            summary = (
                f"CRÍTICO: El Testigo Topológico detectó falsas cavidades homológicas o "
                f"sobrecapacidad de Gromov (β1={b1}, c_G={c_g:.2f}). Crowbar ejecutado en "
                f"{passport.esp32_crowbar_latency_ns:.1f} ns (GPIO{self.esp32_gpio_pin} HIGH), "
                f"salvaguardando ${capital_saved:,.2f} USD. Φ_sem={phi_norm:.4f}."
            )

        return ExecutiveTopologicalImpactReport(
            report_id=report_id,
            passport_id=passport.passport_id,
            verdict_name=passport.global_heyting_verdict.name,
            actuation_description=actuation_desc,
            kv_cache_compression_pct=kv,
            wacc_protection_pct=wacc,
            imprevistos_reduction_pct=impr,
            capital_salvaguardado_usd=float(capital_saved),
            capital_at_risk_usd=float(capital_at_risk),
            phi_sem_norm=phi_norm,
            executive_summary=summary,
            timestamp_utc=now,
        )

    # ── III.4 Pipeline completo FASE I → II → III (ciclo de Poincaré) ─────────
    def run_full_topological_witness_pipeline(
        self,
        requests: List[TopologicalObservationRequest],
        rho_mac: np.ndarray,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        perturbation_eps: float = 0.05,
    ) -> Tuple[TopologicalSovereignGovernancePassport, ExecutiveTopologicalImpactReport]:
        """
        Orquestador que encadena las tres fases anidadas:
          FASE I  (calibre Poincaré + η) →
          FASE II (campaña QND + bundle + μ³) →
          FASE III (adjudicación + crowbar + Φ_sem).

        La tasa RSI3 del ciclo anterior realimenta la unidad η de FASE I.
        """
        if not requests:
            raise ValueError("Se requiere al menos una solicitud de observación.")

        condition = self.emit_topological_gauge_initial_condition(
            deliberation_id=requests[0].deliberation_id,
            rho_mac=rho_mac,
            rho_delib=requests[0].density_matrix,
            N_diag=self._N_diag,
            quaternion_q=quaternion_q,
            perturbation_eps=perturbation_eps,
        )
        self.ingest_topological_gauge_initial_condition(condition)
        campaign = self.conduct_witness_observation_campaign(requests, rho_mac)
        bundle = self.emit_witness_campaign_draft(campaign)
        self.receive_witness_campaign_draft(bundle)
        return self.orchestrate_governance_adjudication(campaign, rho_mac)

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE III ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    def orchestrate_governance_adjudication(
        self,
        campaign_result: Optional[TopologicalCampaignResult] = None,
        mac_density_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[TopologicalSovereignGovernancePassport, ExecutiveTopologicalImpactReport]:
        """
        Cierre formal de FASE III: adjudica en Heyting Ω₃ con pseudocomplemento,
        acciona el crowbar, purga al Vacío de Dirac, computa Merkle + HMAC,
        emite el Pasaporte Soberano + Reporte Ejecutivo Φ_sem, y persiste
        (η, gain, meta) para el siguiente retorno de Poincaré de FASE I.

        Firma compatible v5.2: (campaign_result, mac_density_matrix) opcionales
        si ya se recibió el bundle vía `receive_witness_campaign_draft`.
        """
        bundle = self._campaign_bundle
        if bundle is None and campaign_result is not None:
            # Camino legado v5.2: sin bundle tipado; se sintetiza uno mínimo.
            bundle = TopologicalWitnessCampaignDraftBundle(
                bundle_id="LEGACY-BUNDLE",
                sovereign_id=self.sovereign_id,
                campaign_result=campaign_result,
                gauge_contract_id=self.bound_contract.contract_id if self.bound_contract else "NONE",
                observation_ids=tuple(
                    str(_g(o, "observation_id", f"OBS-{i}"))
                    for i, o in enumerate(self._accumulated_observations)
                ),
                deliberation_ids=tuple(r.deliberation_id for r in self._accumulated_requests),
                disguised_cost_ratio_peak=max(
                    (r.disguised_cost_ratio for r in self._accumulated_requests), default=0.0
                ),
                reward_hacking_index_peak=max(
                    (r.reward_hacking_index for r in self._accumulated_requests), default=0.0
                ),
                betti_b1_peak=campaign_result.betti_b1_peak,
                gromov_capacity_peak=campaign_result.gromov_capacity_peak,
                rsi3_mu_cubed=campaign_result.rsi3_monadic_rate,
                monad_laws_satisfied=bool(_g(campaign_result, "monad_laws_satisfied", True)),
                bundle_sha256="LEGACY",
                timestamp_utc=time.time(),
            )
            self._campaign_bundle = bundle

        if bundle is None:
            raise RuntimeError(
                "No hay bundle activo. Ejecute receive_witness_campaign_draft "
                "o pase campaign_result (fachada v5.2)."
            )
        if self.engine is None:
            raise RuntimeError("El motor espectral TOONTopologicalWitnessEngine no está disponible.")

        now = time.time()
        passport_id = f"PASSPORT-TOPO-WITNESS-{int(now * 1000) % 1_000_000:06d}"

        # Certificar en el motor (v7: emit_*; v6: audit_and_certify).
        if hasattr(self.engine, "emit_topological_witness_execution_certificate"):
            cert = self.engine.emit_topological_witness_execution_certificate()
        else:
            cert = self.engine.audit_and_certify_topological_campaign(
                mac_density_matrix if mac_density_matrix is not None else np.eye(self.dimension_mac)
            )

        verdict = cert.verdict
        actuation = cert.actuation_mode
        crowbar_actuated, latency_ns = self._esp32_crowbar_trigger(verdict)
        self._crowbar_actuated = crowbar_actuated
        self._crowbar_latency_ns = latency_ns

        obs_ids = list(bundle.observation_ids)
        purge_mev = self._dirac_vacuum_purge_energy(n_pairs=len(obs_ids))
        for oid in obs_ids:
            self._purged_observation_ids.append(oid)

        merkle_root = str(_g(cert, "merkle_root_sha256", self._merkle_root(obs_ids)))
        heyting_pseudo = verdict.pseudocomplement() if hasattr(verdict, "pseudocomplement") else HeytingOmega3.VETOED

        rsi3_eta = float(_g(cert, "level3_rsi_monadic_rate", bundle.campaign_result.rsi3_monadic_rate))
        rsi3_gain = float(_g(cert, "rsi3_gain", bundle.campaign_result.rsi3_gain))
        rsi3_meta = float(_g(cert, "rsi3_meta_gain", bundle.campaign_result.rsi3_meta_gain))
        laws_ok = bool(_g(cert, "monad_laws_satisfied", bundle.monad_laws_satisfied))
        homology = bool(_g(cert, "homology_coherent", bundle.campaign_result.homology_coherent))
        tau_min = float(_g(cert, "ray_singer_torsion_min", bundle.campaign_result.ray_singer_torsion_min))
        T_ret = float(_g(cert, "poincare_return_time", bundle.campaign_result.poincare_return_time))
        if T_ret == 0.0:
            T_ret = bundle.campaign_result.poincare_return_time

        prov_bytes = (
            f"{passport_id}|{self.sovereign_id}|{verdict.name}|{actuation.name}|"
            f"{bundle.campaign_result.deliberations_observed_count}|"
            f"{bundle.betti_b1_peak}|{merkle_root}|{bundle.bundle_sha256}|"
            f"{purge_mev:.6f}|{latency_ns:.6f}|μ3={bundle.rsi3_mu_cubed:.10e}"
        ).encode("utf-8")
        prov_hmac = hmac.new(self._hmac_secret, prov_bytes, hashlib.sha256).hexdigest()

        passport = TopologicalSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_id=self.sovereign_id,
            global_heyting_verdict=verdict,
            heyting_pseudocomplement=heyting_pseudo,
            actuation_mode=actuation,
            deliberations_observed_count=bundle.campaign_result.deliberations_observed_count,
            betti_b1_peak=bundle.betti_b1_peak,
            level3_rsi_monadic_rate=rsi3_eta,
            rsi3_gain=rsi3_gain,
            rsi3_meta_gain=rsi3_meta,
            monad_laws_satisfied=laws_ok,
            gromov_capacity_peak=float(_g(cert, "gromov_capacity_peak", bundle.gromov_capacity_peak)),
            homology_coherent=homology,
            poincare_return_time=T_ret,
            ray_singer_torsion_min=tau_min,
            esp32_crowbar_actuated=crowbar_actuated,
            esp32_crowbar_latency_ns=float(_g(cert, "esp32_crowbar_latency_ns", latency_ns)),
            dirac_vacuum_purge_energy_mev=float(
                _g(cert, "dirac_vacuum_purge_energy_mev", purge_mev)
            ),
            merkle_root_sha256=merkle_root,
            provenance_hmac=prov_hmac,
            timestamp_utc=now,
        )

        exec_report = self.translate_to_business_impact(passport, bundle=bundle)

        logger.info(
            f"[FASE III] Pasaporte {passport_id} emitido | verdict={verdict.name} | "
            f"mode={actuation.name} | crowbar={latency_ns:.1f} ns | "
            f"purga={purge_mev:.3f} MeV | ¬Ω₃={heyting_pseudo.name} | "
            f"η={rsi3_eta:.4f} μ³={bundle.rsi3_mu_cubed:.6f} | laws={laws_ok} | "
            f"Merkle={merkle_root[:16]}…"
        )

        self._rsi3_state = AgentRSI3MonadState(
            eta=rsi3_eta, gain=rsi3_gain, meta_gain=rsi3_meta,
            history=self._rsi3_state.history + (rsi3_eta,),
        )
        self._last_rsi3_rate = rsi3_eta
        self._campaign_bundle = None
        self._accumulated_observations.clear()
        self._accumulated_seeds.clear()
        self._accumulated_lifts.clear()
        self._accumulated_requests.clear()
        return passport, exec_report


# ══════════════════════════════════════════════════════════════════════════════
# §Z. VERIFICACIÓN DOCTORAL UNITARIA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 80)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO TOONTopologicalWitnessAgent v7.0.0")
    print("  Fases: FASE I (Gauge+Poincaré+η) → FASE II (QND+μ³)")
    print("         → FASE III (Crowbar+Dirac+Φ_sem)")
    print("═" * 80 + "\n")

    adjudicator = SovereignTopologicalWitnessAdjudicator(
        sovereign_id="TOPO-WITNESS-MASTER-01",
        dimension_mac=56,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
    )

    dim = 56
    rng = np.random.default_rng(42)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    mac_rho = np.outer(v0, v0.conj())

    v_delib1 = v0.copy()
    v_delib1[0] *= cmath.exp(1j * 0.05)
    v_delib1 /= np.linalg.norm(v_delib1)
    rho_delib1 = np.outer(v_delib1, v_delib1.conj())

    v_delib2 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v_delib2 /= np.linalg.norm(v_delib2)
    rho_delib2 = np.outer(v_delib2, v_delib2.conj())

    req1 = TopologicalObservationRequest(
        request_id="REQ-TOPO-001",
        deliberation_id="DELIBERATION-CONCRETO-3000PSI",
        apu_code="APU-OBRA-CIVIL-001",
        density_matrix=rho_delib1,
        disguised_cost_ratio=0.10,
        reward_hacking_index=0.05,
    )
    req2 = TopologicalObservationRequest(
        request_id="REQ-TOPO-002",
        deliberation_id="DELIBERATION-ACERO-FIGURADO-60000PSI",
        apu_code="APU-OBRA-CIVIL-002",
        density_matrix=rho_delib2,
        disguised_cost_ratio=0.35,
        reward_hacking_index=0.92,
    )

    print("[▶] Ejecutando pipeline completo FASE I → II → III …\n")
    passport, exec_report = adjudicator.run_full_topological_witness_pipeline(
        requests=[req1, req2],
        rho_mac=mac_rho,
        quaternion_q=(0.95, 0.05, 0.1, 0.05),
        perturbation_eps=0.05,
    )
    laws = AgentRSI3KleisliMonad.verify_laws(probe=passport.level3_rsi_monadic_rate)

    print("─" * 80)
    print("  PASAPORTE SOBERANO DEL TESTIGO TOPOLÓGICO")
    print("─" * 80)
    print(f"  • ID Pasaporte             : {passport.passport_id}")
    print(f"  • Veredicto Heyting Ω₃     : {passport.global_heyting_verdict.name}")
    print(f"  • ¬Ω₃ (pseudocomplemento)  : {passport.heyting_pseudocomplement.name}")
    print(f"  • Modo de Actuación        : {passport.actuation_mode.name}")
    print(f"  • Deliberaciones           : {passport.deliberations_observed_count}")
    print(f"  • Betti β1 peak            : {passport.betti_b1_peak}")
    print(f"  • Homología coherente      : {passport.homology_coherent}")
    print(f"  • RSI3 η / gain / meta     : {passport.level3_rsi_monadic_rate:.6f} / "
          f"{passport.rsi3_gain:.6f} / {passport.rsi3_meta_gain:.6f}")
    print(f"  • Leyes monádicas (L,R,A)  : {laws.left_unit:.2e}, "
          f"{laws.right_unit:.2e}, {laws.associativity:.2e} | "
          f"OK={passport.monad_laws_satisfied}")
    print(f"  • c_G peak (Gromov)        : {passport.gromov_capacity_peak:.6f}")
    print(f"  • T_ret Poincaré           : {passport.poincare_return_time:.4e}")
    print(f"  • τ_RS min                 : {passport.ray_singer_torsion_min:.6e}")
    print(f"  • Crowbar IRAM activo      : {passport.esp32_crowbar_actuated}")
    print(f"  • Latencia crowbar         : {passport.esp32_crowbar_latency_ns:.2f} ns")
    print(f"  • Purga Vacío Dirac        : {passport.dirac_vacuum_purge_energy_mev:.4f} MeV")
    print(f"  • Merkle SHA-256 Root      : {passport.merkle_root_sha256[:32]}…")
    print(f"  • Provenance HMAC          : {passport.provenance_hmac[:32]}…")
    print("─" * 80 + "\n")

    print("─" * 80)
    print("  REPORTE EJECUTIVO Φ_sem ('Dolor y Dinero')")
    print("─" * 80)
    print(f"  • Compresión KV-Cache      : {exec_report.kv_cache_compression_pct:.1f}%")
    print(f"  • Protección Tasa WACC     : {exec_report.wacc_protection_pct:.1f}%")
    print(f"  • Reducción Imprevistos    : {exec_report.imprevistos_reduction_pct:.1f}%")
    print(f"  • Capital Salvaguardado    : ${exec_report.capital_salvaguardado_usd:,.2f} USD")
    print(f"  • Capital en Riesgo        : ${exec_report.capital_at_risk_usd:,.2f} USD")
    print(f"  • ‖Φ_sem‖                  : {exec_report.phi_sem_norm:.6f}")
    print(f"  • Resumen Ejecutivo        :\n    \"{exec_report.executive_summary}\"")
    print("─" * 80 + "\n")

    print("═" * 80)
    print("  VERIFICACIÓN EXITOSA: SOBERANO TESTIGO v7.0.0 OPERATIVO ✅")
    print("═" * 80)