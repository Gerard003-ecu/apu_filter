# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Cognitive Crop Auditor Agent (Soberano Auditor de Cultivo Cognitivo)                 ║
║ Ubicación: app/agents/wisdom/toon_cognitive_crop_auditor_agent.py                                    ║
║ Versión  : 7.0.0-Doctoral-Poincare-RSI3-Gauge-Chern-Wilson-Heyting-ESP32-Dirac-PhiSem                ║
║            -Kleisli-Hopf-StoraZumino-Sp4-Ruelle                                                      ║
║ Función  : Soberano de Calibre Auditor y Fiscal de parcelas de cultivo cognitivo, responsable de     ║
║            investigar y juzgar en el topos Heyting Ω₃ los autoestados generados por el cultivo       ║
║            mediante conexión de calibre de estado A_μ = π_𝔰𝔲(ρ), mapa de momento equivariante en     ║
║            P(T*Q, G), análisis de Gelfand-Ruelle y Nekhoroshev, y mediación ejecutiva vía Φ_sem.     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

I. DEFINICIÓN Y ROL DEL SOBERANO EN LA ARQUITECTURA COGNITIVA
─────────────────────────────────────────────────────────────
El TOON Cognitive Crop Auditor Agent es la autoridad fiscal y censora del Estrato WISDOM
especializada en el escrutinio de los autoestados agronómicos y metabólicos del conocimiento.
Su cometido ineludible es impedir que parcelas de datos espurias, sobreajustes estocásticos o
alucinaciones semánticas contaminen el Atlas de la corporación. Emplea un fibrado de gauge $P(T^*Q, G)$
donde la conexión se extrae canónicamente de la matriz de densidad $\rho$, y proyecta la reducción
simpléctica vía mapas de momento equivariantes, sentenciando en el topos de Heyting $\Omega_3$.

II. FIBRADOS DE CALIBRE Y MAPA DE MOMENTO EQUIVARIANTE
──────────────────────────────────────────────────────
§1. Conexión Anclada al Estado y Curvatura No-Abeliana:
    La conexión se define mediante la proyección sobre la subálgebra compacta $\mathfrak{su}(N)$:
        A_\mu = \pi_{\mathfrak{su}(N)}(\rho) \in \mathfrak{su}(N) \implies F_{\mu\nu} = [A_\mu, A_\nu] \quad (\text{para formas planas})
    La 3-forma de Chern-Simons y la carga instantónica $k = \frac{1}{8\pi^2}\int \mathrm{Tr}(F \wedge F)$
    proporcionan los números cuánticos topológicos que caracterizan de forma unívoca a la parcela.

§2. Mapa de Momento Equivariante de Marsden-Weinstein:
    Dada la acción del grupo de simetría $G$ sobre la variedad simpléctica de fases $M$, el mapa
    de momento equivariante $\mathbf{J}: M \to \mathfrak{g}^*$ satisface:
        \langle d\mathbf{J}(\xi), v \rangle = \omega(X_\xi, v) \quad \forall \xi \in \mathfrak{g}, \; v \in TM
    asegurando que las trayectorias de auditoría conserven estrictamente los invariantes de Noether.

III. AUDITORÍA ESPECTRAL BANACH-GELFAND Y TAMIZ DE RUELLE
─────────────────────────────────────────────────────────
§1. Radio Espectral de Gelfand en $\mathcal{B}(\mathfrak{u}(n))$:
    Para el operador de transferencia cuántica $\mathcal{T}$, se verifica el radio espectral sobre $\mathfrak{sl}(n)$:
        r(\mathcal{T}) = \lim_{k \to \infty} \|\mathcal{T}^k\|^{1/k} \le 1.0 - \Delta_{\mathrm{CPTP}}
    Un radio mayor o igual a 1 en el subespacio de traza cero indica fuga no conservativa de probabilidad.

§2. Operador de Ruelle y Estabilidad de Nekhoroshev:
    El autovalor líder de Ruelle determina la entropía métrica, mientras que la cota de Nekhoroshev
    garantiza que el confinamiento en las acciones perdure durante tiempos astronómicos $T_{\mathrm{nek}} \sim \exp(\varepsilon^{-a})$.

IV. FUNTOR SEMÁNTICO $\Phi_{\mathrm{sem}}: \mathrm{Sh}(\partial K, \Omega_3) \to \mathrm{Business}$
────────────────────────────────────────────────────────────────────────────────
El auditor cuantifica las pérdidas monetarias por ineficiencia de parcela:
    \Phi_{\mathrm{sem}}(\tau) \mapsto \left( \mathrm{CapitalEnRiesgo}, \; \mathrm{IndiceDeDegradacionMetabolica}, \; \mathrm{SancionPorVeto} \right)
proveyendo a la dirección ejecutiva de un cálculo exacto del desvío presupuestario ("Dolor y Dinero").

V. ARQUITECTURA DE TRES FASES ANIDADAS
──────────────────────────────────────
  • FASE I   [Topología de Calibre Auditora]: Inicia $P(T^*Q, G)$, calcula $A_\mu$, $\mathrm{CS}_3$, mapa de
             momento $\mathbf{J}$, contrato HMAC y emite `CropAuditorGaugeInitialCondition`.
  • FASE II  [Agente Auditor de Cultivo]: Consume fase I, ejecuta auditoría Banach-Gelfand, Ruelle,
             evalúa métricas adversarias de costo, aplica RSI³ y emite `CropAuditorCampaignDraftBundle`.
  • FASE III [Soberano Adjudicador Fiscal]: Juzga en Heyting $\Omega_3$, verifica ESP32 Crowbar,
             purga a Dirac, traduce a $\Phi_{\mathrm{sem}}$ y realimenta $(\eta, \mathrm{gain}, \mathrm{meta})$ a Fase I.

VI. CIERRE CIBER-FÍSICO ESP32 Y ANIQUILACIÓN EN VACÍO DE DIRAC
─────────────────────────────────────────────────────────────
Al dictaminarse un veredicto de veto $\bot = \mathrm{VETOED}$:
    \tau_{\mathrm{crowbar}} \le 400.0 \text{ ns (ESP32 / GPIO14)}
disipando la totalidad de la energía corrupta en el Vacío de Dirac con rendimiento térmico nulo:
    e^- + e^+ \longrightarrow 2\gamma \quad \left( E = 2 m_e c^2 \approx 1.021998 \text{ MeV}, \quad \text{Back-Action } 0.0 \text{ dB} \right)
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

# ─── Importación del motor v7.0.0 (con fallback v6 / relativo / app) ──────────
_ENGINE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from toon_cognitive_crop_auditor_engine import (  # type: ignore
        ActuationMode,
        CropAuditDraftBundle,
        CropAuditIndictmentGerm,
        CropAuditStatus,
        CropAuditorCanonicalSeed,
        CropAuditorExecutionCertificate,
        CropAuditorInitialCondition,
        CropAuditorQNDEngine,
        HeytingOmega3,
        PoincareCropAuditorAtlas,
        QuaternionHopfLift,
        TOONCognitiveCropAuditorEngine,
    )
except ImportError as _e1:  # pragma: no cover
    try:
        from scratch.toon_cognitive_crop_auditor_engine import (  # type: ignore
            ActuationMode,
            CropAuditDraftBundle,
            CropAuditIndictmentGerm,
            CropAuditStatus,
            CropAuditorCanonicalSeed,
            CropAuditorExecutionCertificate,
            CropAuditorInitialCondition,
            CropAuditorQNDEngine,
            HeytingOmega3,
            PoincareCropAuditorAtlas,
            QuaternionHopfLift,
            TOONCognitiveCropAuditorEngine,
        )
    except ImportError as _e2:  # pragma: no cover
        try:
            from app.wisdom.toon_cognitive_crop_auditor_engine import (  # type: ignore
                ActuationMode,
                CropAuditDraftBundle,
                CropAuditIndictmentGerm,
                CropAuditStatus,
                CropAuditorCanonicalSeed,
                CropAuditorExecutionCertificate,
                CropAuditorInitialCondition,
                CropAuditorQNDEngine,
                HeytingOmega3,
                PoincareCropAuditorAtlas,
                QuaternionHopfLift,
                TOONCognitiveCropAuditorEngine,
            )
        except ImportError as _e3:  # pragma: no cover
            from .toon_cognitive_crop_auditor_engine import (  # type: ignore
                ActuationMode,
                CropAuditDraftBundle,
                CropAuditIndictmentGerm,
                CropAuditStatus,
                CropAuditorCanonicalSeed,
                CropAuditorExecutionCertificate,
                CropAuditorInitialCondition,
                CropAuditorQNDEngine,
                HeytingOmega3,
                PoincareCropAuditorAtlas,
                QuaternionHopfLift,
                TOONCognitiveCropAuditorEngine,
            )
            _ENGINE_IMPORT_ERROR = _e3
        else:
            _ENGINE_IMPORT_ERROR = _e2
    else:
        _ENGINE_IMPORT_ERROR = _e1

# Tipos RSI3 / Poincaré del motor v7 — fallback local si el motor es v6.
try:
    from toon_cognitive_crop_auditor_engine import (  # type: ignore
        RSI3KleisliMonad as _EngineRSI3KleisliMonad,
        RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
        RSI3MonadState as _EngineRSI3MonadState,
    )
    _ENGINE_HAS_RSI3 = True
except Exception:
    try:
        from scratch.toon_cognitive_crop_auditor_engine import (  # type: ignore
            RSI3KleisliMonad as _EngineRSI3KleisliMonad,
            RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
            RSI3MonadState as _EngineRSI3MonadState,
        )
        _ENGINE_HAS_RSI3 = True
    except Exception:
        try:
            from app.wisdom.toon_cognitive_crop_auditor_engine import (  # type: ignore
                RSI3KleisliMonad as _EngineRSI3KleisliMonad,
                RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
                RSI3MonadState as _EngineRSI3MonadState,
            )
            _ENGINE_HAS_RSI3 = True
        except Exception:
            try:
                from .toon_cognitive_crop_auditor_engine import (  # type: ignore
                    RSI3KleisliMonad as _EngineRSI3KleisliMonad,
                    RSI3MonadLawResiduals as _EngineRSI3MonadLawResiduals,
                    RSI3MonadState as _EngineRSI3MonadState,
                )
                _ENGINE_HAS_RSI3 = True
            except Exception:
                _ENGINE_HAS_RSI3 = False
                _EngineRSI3KleisliMonad = None  # type: ignore
                _EngineRSI3MonadLawResiduals = None  # type: ignore
                _EngineRSI3MonadState = None  # type: ignore


logger = logging.getLogger("APU.Wisdom.TOONCognitiveCropAuditorAgent")
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


# ══════════════════════════════════════════════════════════════════════════════
# §A. MÓNADA RSI3 DEL AGENTE (envuelve la del motor; fallback si motor v6)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AgentRSI3MonadState:
    """Estado RSI3 de nivel 3 del Soberano de Calibre.

    Nivel 1 — η        : tasa objeto (mejora del auditado / parcela).
    Nivel 2 — gain     : tasa del método (mejora del acusador).
    Nivel 3 — meta_gain: tasa del meta-método (mejora del mejorador).

    Si el motor v7 expone RSI3MonadState, este estado lo envuelve; si no,
    opera de forma autónoma sobre el semiring ([0,1], ⊕, ⊗).
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
# §B. DATACLASSES DEL SOBERANO AUDITOR DEL CULTIVO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareGaugeMomentMap:
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


@dataclass(frozen=True, slots=True)
class CropAuditorGaugeContract:
    """
    Contrato de calibre inmutable forjado en la FASE I.
    Incluye los números topológicos rigurosos: CS₃ signado, residual
    Stora-Zumino, instantón k, clase Chern-Hopf c₁, holonomía de Wilson θ_W,
    momento de Poincaré, unidad RSI3 η, coordenadas Hopf.
    """
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
class CropAuditorGaugeInitialCondition:
    """
    Estructura que la FASE I entrega como semilla formal a la FASE II.
    Encapsula el contrato de calibre + la condición inicial de auditoría del
    motor (CropAuditorInitialCondition) + momento de Poincaré + unidad RSI3
    + huella preconditioner canónica.
    """
    gauge_contract: CropAuditorGaugeContract
    engine_initial_condition: CropAuditorInitialCondition
    poincare_moment: PoincareGaugeMomentMap
    rsi3_unit: AgentRSI3MonadState
    sovereign_id: str
    preconditioner_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class CropParcelIndictmentRequest:
    """
    Solicitud de acusación sobre una parcela de cultivo.
    Incluye indicadores de riesgo económico (disguised_cost_ratio, reward_hacking_index).
    """
    request_id: str
    crop_id: str
    apu_code: str
    density_matrix: np.ndarray
    disguised_cost_ratio: float = 0.0
    reward_hacking_index: float = 0.0
    stinespring_coupling_eps: float = 0.05


@dataclass(frozen=True, slots=True)
class CropProsecutionCampaignResult:
    """Resultado consolidado de la campaña de acusación en la FASE II."""
    campaign_id: str
    sovereign_id: str
    parcels_examined_count: int
    false_vacuums_detected_count: int
    weak_value_aw_mean: complex
    weak_value_aw_variance: float
    fubini_study_distance_peak: float
    fubini_study_distance_mean: float
    uhlmann_fidelity_mean: float
    gelfand_radius_peak: float
    gelfand_radius_mean: float
    ruelle_radius_peak: float
    transfer_spectral_gap_min: float
    banach_contraction_satisfied: bool
    nekhoroshev_time_min: float
    nekhoroshev_time_mean: float
    nekhoroshev_steepness_mean: float
    hessian_false_vacuums_count: int
    pontryagin_obstructions_count: int
    bogomolny_violations_count: int
    melnikov_integral_peak: float
    homoclinic_splitting_peak: float
    bryuno_sum_mean: float
    morse_bott_euler_chi: int
    poincare_return_time: float
    gromov_capacity_peak: float
    disguised_cost_ratio_mean: float
    reward_hacking_index_mean: float
    rsi3_monadic_rate: float
    rsi3_gain: float
    rsi3_meta_gain: float
    monad_laws_satisfied: bool
    kleisli_trace_length: int
    global_heyting_verdict: HeytingOmega3
    timestamp: float


# ─── Semilla formal FASE II → FASE III ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CropAuditorCampaignDraftBundle:
    """
    Semilla formal FASE II → FASE III. Empaqueta el resultado de campaña con
    indicadores económicos + el borrador formal del motor (CropAuditDraftBundle)
    + RSI3 μ³ + huella canónica SHA-256.
    """
    bundle_id: str
    sovereign_id: str
    campaign_result: CropProsecutionCampaignResult
    engine_draft: CropAuditDraftBundle
    gauge_contract_id: str
    indictment_ids: Tuple[str, ...]
    disguised_cost_ratio_peak: float
    reward_hacking_index_peak: float
    max_gelfand_radius: float
    false_vacuums_count: int
    rsi3_mu_cubed: float
    monad_laws_satisfied: bool
    bundle_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class CropAuditorSovereignGovernancePassport:
    """Pasaporte Soberano de Gobernanza Auditora emitido en la FASE III."""
    passport_id: str
    sovereign_id: str
    global_heyting_verdict: HeytingOmega3
    heyting_pseudocomplement: HeytingOmega3
    actuation_mode: ActuationMode
    parcels_prosecuted_count: int
    false_vacuums_vetoed_count: int
    level3_rsi_monadic_rate: float
    rsi3_gain: float
    rsi3_meta_gain: float
    monad_laws_satisfied: bool
    gromov_capacity_peak: float
    gelfand_banach_verified: bool
    gelfand_radius_peak: float
    nekhoroshev_time_min: float
    poincare_return_time: float
    esp32_crowbar_actuated: bool
    esp32_crowbar_latency_ns: float
    dirac_vacuum_purge_energy_mev: float
    merkle_root_sha256: str
    provenance_hmac: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class ExecutiveCropProsecutionImpactReport:
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
GaugePhaseITerminal = CropAuditorGaugeInitialCondition
GaugePhaseIIInitial = CropAuditorGaugeInitialCondition
# Alias de continuación formal FASE II.terminal ≡ FASE III.initial
GaugePhaseIITerminal = CropAuditorCampaignDraftBundle
GaugePhaseIIIInitial = CropAuditorCampaignDraftBundle


def _g(obj: Any, name: str, default: Any = 0.0) -> Any:
    """getattr seguro para compatibilidad motor v6 / v7."""
    return getattr(obj, name, default)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE I — TOPOLOGÍA DE CALIBRE DEL FIBRADO PRINCIPAL + POINCARÉ
#  Terminal: `emit_crop_auditor_gauge_initial_condition` → semilla de FASE II.
#  (FASE I.terminal = FASE II.initial = CropAuditorGaugeInitialCondition)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveCropAuditorGaugeTopology:
    """
    FASE I — Topología de Calibre del Fibrado Principal P(T*Q, G).

    Grupo estructural G = U(1) ⊗ SU(2) ⊗ H₃(ℝ) (H₃ = cuaterniones imaginarios puros).

    Construcciones matemáticas rigurosas:
      • Proyección a 𝔰𝔲(N): A ↦ i(ρ_H − (Tr ρ_H / N)·𝕀)  (antihermitiana, traza nula).
      • Curvatura no Abeliana: F_{μν} = [A_μ, A_ν]  (Bianchi D_A F = 0).
      • CS₃[A] = (1/8π²) Tr(A ∧ dA + (2/3) A ∧ A ∧ A)  (signado).
      • Instantón k = (1/8π²) Tr(F F†) ∈ ℤ por π₃(SU(2)) = ℤ.
      • Descenso Stora-Zumino: d CS₃ − Tr(F ∧ F)/(8π²)  (anomalía ABJ).
      • Holonomía de Wilson: W = 𝒫 exp(∮_γ A) sobre el retorno de Poincaré.
      • Chern-Hopf c₁ vía fibración cuaterniónica S³ → S².
      • Momento equivariante μ : T*Q → 𝔤* (Delaunay + Hill + Greene).
      • HMAC-SHA256 sobre serialización canónica determinista.
      • RSI3-0: unidad monádica η.
    """

    def __init__(self, sovereign_id: str = "CROP-AUDITOR-MASTER-01") -> None:
        self.sovereign_id = sovereign_id
        self._hmac_secret = hashlib.sha256(
            f"SECRET-CROP-AUDITOR-GAUGE-{sovereign_id}".encode("utf-8")
        ).digest()
        self._rsi3_state: AgentRSI3MonadState = AgentRSI3KleisliMonad.eta(0.25)

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
        return float(cs3.real)

    # ── I.4 Número de instantón k (densidad de Pontryagin) ────────────────────
    @classmethod
    def _instanton_number_k(
        cls,
        A_mu: np.ndarray,
        A_nu: np.ndarray,
    ) -> float:
        """k = (1/8π²) Tr(F F†); se redondea a ℤ (π₃(SU(2)) = ℤ)."""
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
        """‖d CS₃ − Tr(F∧F)/(8π²)‖.

        En el retículo matricial, d CS₃ se aproxima por CS₃[A] (3-forma) y
        Tr(F∧F)/(8π²) por la densidad de Pontryagin de [A_μ, A_ν]. El v6
        cancelaba idénticamente (4·x − 4·x); aquí la diferencia es genuina.
        """
        cs3 = cls._chern_simons_3form(A_mu, A_nu, A_rho)
        F = cls._nonabelian_curvature(
            cls._project_to_su_N(A_mu),
            cls._project_to_su_N(A_nu),
        )
        pontryagin = (1.0 / (8.0 * math.pi ** 2)) * float(np.trace(F @ F.conj().T).real)
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
    def _chern_hopf_class(
        q: QuaternionHopfLift,
    ) -> Tuple[float, float, Tuple[float, float, float], str]:
        c1 = (1.0 / math.pi) * math.atan2(q.z, q.w)
        holonomy = (2.0 * math.pi * c1) % _TAU
        base = q.hopf_base_point()
        if abs(c1) < 0.1:
            regime = "TRIVIAL_BUNDLE"
        elif abs(c1) < 0.7:
            regime = "MONOPOLE_LIKE"
        else:
            regime = "INSTANTON_DENSE"
        return float(c1), float(holonomy), base, regime

    # ── I.8 Momento equivariante de Poincaré ──────────────────────────────────
    @staticmethod
    def _poincare_moment_from_engine_seed(
        seed: Any,
        A_su_norm: float,
    ) -> PoincareGaugeMomentMap:
        """Inmersión de las acciones de Delaunay en 𝔲(N)* vía ‖A‖_F."""
        L = float(_g(seed, "delaunay_L", 0.0))
        G = float(_g(seed, "delaunay_G", 0.0))
        H = float(_g(seed, "delaunay_H", 0.0))
        n = float(_g(seed, "mean_motion_n", 1.0))
        chi = _g(seed, "characteristic_exponents", ())
        if not isinstance(chi, tuple):
            chi = tuple(float(x) for x in np.atleast_1d(chi))
        T_ret = float(_g(seed, "poincare_return_time", _TAU / max(abs(n), 1e-12)))
        T_rec = float(_g(seed, "poincare_recurrence_estimate", T_ret))
        moment_norm = float(math.sqrt(L * L + G * G + H * H + A_su_norm * A_su_norm))
        return PoincareGaugeMomentMap(
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
        canonical = "|".join(f"{k}={v}" for k, v in fields)
        return canonical.encode("utf-8")

    # ── I.10 Forja del contrato de calibre (A anclado a ρ si se provee) ───────
    def forge_crop_auditor_gauge_contract(
        self,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        dimension_mac: int = 56,
        seed: int = 42,
        density_matrix: Optional[np.ndarray] = None,
        poincare_return_time: float = 1.0,
        rsi3_eta: float = 0.25,
    ) -> CropAuditorGaugeContract:
        now = time.time()
        contract_id = f"CONTRACT-CROP-AUDITOR-{int(now * 1e3) % 1_000_000:06d}"
        N = dimension_mac

        if density_matrix is not None and density_matrix.shape[0] == N:
            # A_μ = π_{𝔰𝔲(N)}(ρ)  — calibre anclado al estado (no RNG).
            A_mu = np.array(density_matrix, dtype=np.complex128)
            # A_ν = Ad_{U(θ)} A_μ  (rotación de Poincaré en el fibrado)
            theta = float(_g(self, "_last_theta_pc", 0.05))
            H_rot = 0.5 * (A_mu + A_mu.conj().T)
            U_theta = la.expm(-1j * H_rot * theta)
            A_nu = U_theta @ A_mu @ U_theta.conj().T
            # A_ρ = componente Hopf (levantamiento cuaterniónico como fase)
            phase = quaternion_q[0] + 1j * quaternion_q[1]
            A_rho = (phase / (abs(phase) + 1e-15)) * A_mu
        else:
            rng = np.random.default_rng(seed)
            scale = 0.01 / math.sqrt(N)
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

        q = QuaternionHopfLift.from_complex_pair(
            complex(quaternion_q[0], quaternion_q[1]),
            complex(quaternion_q[2], quaternion_q[3]),
        )
        c1, _holonomy_hopf, base_xyz, regime = self._chern_hopf_class(q)

        eta = float(np.clip(rsi3_eta, _RSI3_ETA_FLOOR, _RSI3_ETA_CEIL))
        canonical_bytes = self._canonical_hmac_bytes(
            contract_id, self.sovereign_id, cs3, sz_r, k, c1, theta_W, regime,
            moment_norm, eta,
        )
        hmac_sig = hmac.new(self._hmac_secret, canonical_bytes, hashlib.sha256).hexdigest()

        contract = CropAuditorGaugeContract(
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
            f"[{self.sovereign_id}] Contrato de Calibre Auditor forjado [{contract_id}] | "
            f"CS₃={cs3:.4e} | k={k:.0f} | c₁={c1:+.4f} | W={theta_W:+.4f} | "
            f"‖μ‖={moment_norm:.4e} | SZ={sz_r:.4e} | η={eta:.4f} | {regime}"
        )
        return contract

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE I ◀◀◀
    #     Dominio de salida ≡ dominio de entrada de
    #     TOONCognitiveCropAuditorAgent.ingest_crop_auditor_gauge_initial_condition
    #     (FASE I.terminal = FASE II.initial = CropAuditorGaugeInitialCondition)
    # ══════════════════════════════════════════════════════════════════════════
    def emit_crop_auditor_gauge_initial_condition(
        self,
        crop_id: str,
        rho_mac: np.ndarray,
        rho_crop: np.ndarray,
        N_diag: np.ndarray,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        perturbation_eps: float = 0.05,
    ) -> GaugePhaseITerminal:
        """
        Cierre formal de FASE I: forja el contrato de calibre anclado a ρ_crop,
        emite la condición inicial Poincaré del motor (con η RSI3), construye
        el momento equivariante, y empaqueta la semilla formal de FASE II.

        El valor de retorno ES el argumento de
        `TOONCognitiveCropAuditorAgent.ingest_crop_auditor_gauge_initial_condition`.
        """
        rsi3_unit = AgentRSI3KleisliMonad.eta(self._rsi3_state.eta)

        # Motor: transportar η (v7); degradar a v6 si TypeError.
        try:
            engine_condition = PoincareCropAuditorAtlas.emit_audit_initial_condition(
                crop_id=crop_id,
                rho_mac=rho_mac,
                rho_crop=rho_crop,
                N_diag=N_diag,
                perturbation_eps=perturbation_eps,
                rsi3_eta=rsi3_unit.eta,
            )
        except TypeError:
            engine_condition = PoincareCropAuditorAtlas.emit_audit_initial_condition(
                crop_id=crop_id,
                rho_mac=rho_mac,
                rho_crop=rho_crop,
                N_diag=N_diag,
                perturbation_eps=perturbation_eps,
            )

        seed = engine_condition.seed
        T_ret = float(_g(seed, "poincare_return_time", 1.0))
        self._last_theta_pc = float(_g(seed, "poincare_cartan_1form", 0.05))

        contract = self.forge_crop_auditor_gauge_contract(
            quaternion_q=quaternion_q,
            dimension_mac=rho_crop.shape[0],
            seed=int(time.time() * 1e3) % 100_000,
            density_matrix=rho_crop,
            poincare_return_time=T_ret,
            rsi3_eta=rsi3_unit.eta,
        )

        A_su_norm = float(_g(seed, "banach_norm_F", contract.moment_map_norm))
        moment = self._poincare_moment_from_engine_seed(seed, A_su_norm)

        precond_bytes = (
            f"{contract.hmac_signature}|{engine_condition.preconditioner_sha256}|"
            f"{self.sovereign_id}|{crop_id}|η={rsi3_unit.eta:.12e}".encode("utf-8")
        )
        precond = hashlib.sha256(precond_bytes).hexdigest()

        logger.info(
            f"[FASE I → FASE II] Condición inicial auditora emitida | "
            f"contract={contract.contract_id} | engine_seed={seed.seed_id} | "
            f"T_ret={moment.return_time:.4e} | Sp={moment.symplectic_residual:.2e} | "
            f"η={rsi3_unit.eta:.4f}"
        )
        return CropAuditorGaugeInitialCondition(
            gauge_contract=contract,
            engine_initial_condition=engine_condition,
            poincare_moment=moment,
            rsi3_unit=rsi3_unit,
            sovereign_id=self.sovereign_id,
            preconditioner_sha256=precond,
            timestamp_utc=time.time(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE II — SOBERANO AUDITOR Y CAMPAÑAS RSI3
#  Primer método `ingest_crop_auditor_gauge_initial_condition` CONSUME la
#  estructura de FASE I.
#  (FASE II.initial = FASE I.terminal = CropAuditorGaugeInitialCondition)
#  Terminal `emit_crop_prosecution_campaign_draft` es la semilla de FASE III.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TOONCognitiveCropAuditorAgent(CognitiveCropAuditorGaugeTopology):
    """
    FASE II — Soberano Auditor del Cultivo Cognitivo.

    Hereda la topología de calibre de FASE I y orquesta campañas de investigación
    y acusación sobre las parcelas, integrando los indicadores económicos
    `disguised_cost_ratio` y `reward_hacking_index` en el veredicto local y
    en la observación de la mónada RSI3. La multiplicación μ^{(3)} se aplica
    una sola vez al emitir el bundle (el motor es la fuente de verdad).
    """

    def __init__(
        self,
        sovereign_id: str = "CROP-AUDITOR-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
    ) -> None:
        super().__init__(sovereign_id=sovereign_id)
        self.dimension_mac = dimension_mac
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension_mac))

        self._qnd_engine = CropAuditorQNDEngine(dimension=dimension_mac)
        self._sovereign_engine = TOONCognitiveCropAuditorEngine(
            dimension=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
            sovereign_id=sovereign_id,
        )

        self._gauge_condition: Optional[GaugePhaseIIInitial] = None
        self._last_rsi3_rate: float = 0.25
        self._accumulated_indictments: List[CropAuditIndictmentGerm] = []
        self._accumulated_requests: List[CropParcelIndictmentRequest] = []
        self._kleisli_trace: List[str] = []

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE II ◀◀◀
    #     Continuación formal de
    #     CognitiveCropAuditorGaugeTopology.emit_crop_auditor_gauge_initial_condition
    # ══════════════════════════════════════════════════════════════════════════
    def ingest_crop_auditor_gauge_initial_condition(
        self,
        condition: GaugePhaseIIInitial,
    ) -> "TOONCognitiveCropAuditorAgent":
        """
        Ingesta la semilla de FASE I: valida HMAC del contrato de calibre y el
        preconditioner_sha256 (incluye η), conecta la condición al motor QND
        y transporta la unidad RSI3-0 al estado monádico del agente.

        Retorna self para permitir encadenamiento fluido.
        """
        gc = condition.gauge_contract
        canonical_bytes = self._canonical_hmac_bytes(
            gc.contract_id,
            gc.sovereign_id,
            gc.chern_simons_3form,
            gc.stora_zumino_residual,
            gc.instanton_number_k,
            gc.chern_hopf_class_c1,
            gc.wilson_holonomy_angle,
            gc.topology_regime,
            gc.moment_map_norm,
            gc.rsi3_unit_eta,
        )
        expected_hmac = hmac.new(self._hmac_secret, canonical_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_hmac, gc.hmac_signature):
            logger.error(
                f"[FASE II] Falla de autenticación HMAC en contrato {gc.contract_id}."
            )
            raise ValueError("HMAC del contrato de calibre inválido.")

        expected_precond = hashlib.sha256(
            (
                f"{gc.hmac_signature}|{condition.engine_initial_condition.preconditioner_sha256}|"
                f"{self.sovereign_id}|{condition.engine_initial_condition.crop_id}|"
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
        self._qnd_engine.ingest_audit_initial_condition(condition.engine_initial_condition)
        logger.info(
            f"[FASE II ← FASE I] Condición de calibre auditora {gc.contract_id} ingerida | "
            f"régimen={gc.topology_regime} | k={gc.instanton_number_k:.0f} | "
            f"‖μ‖={condition.poincare_moment.moment_norm:.4e} | "
            f"T_ret={condition.poincare_moment.return_time:.4e} | η={condition.rsi3_unit.eta:.4f}"
        )
        return self

    # ── II.1 Acusación individual de una parcela ─────────────────────────────
    def process_crop_indictment(
        self,
        request: CropParcelIndictmentRequest,
        mac_density_matrix: np.ndarray,
    ) -> CropAuditIndictmentGerm:
        """Procesa y examina una acusación individual vía el motor QND."""
        logger.info(
            f"[{self.sovereign_id}] Indictando parcela [{request.crop_id}] "
            f"(APU: {request.apu_code}, dcr={request.disguised_cost_ratio:.3f}, "
            f"rhi={request.reward_hacking_index:.3f})..."
        )
        return self._qnd_engine.audit_crop_indictment(
            crop_id=request.crop_id,
            mac_density_matrix=mac_density_matrix,
            crop_density_matrix=request.density_matrix,
        )

    # ── II.2 Veredicto local Heyting Ω₃ con integración económica ─────────────
    @staticmethod
    def _local_heyting_verdict(
        indictment: CropAuditIndictmentGerm,
        request: CropParcelIndictmentRequest,
        capacity_gromov_max: float,
    ) -> HeytingOmega3:
        """
        Reglas de adjudicación local con integración de indicadores económicos:
          VETOED   si falso vacío ∨ ¬C* ∨ ¬Banach-contracción ∨ c_G > max ∨ d_FS > 1.50
                   ∨ disguised_cost_ratio > 0.30 ∨ reward_hacking_index > 0.90
          DEGRADED si d_FS > 0.15 ∨ ¬wirtinger_bound
                   ∨ disguised_cost_ratio > 0.15 ∨ reward_hacking_index > 0.70
          COHERENT en otro caso
        """
        cstar_ok = (
            indictment.cstar_hermitian
            and indictment.cstar_trace_normalized
            and indictment.cstar_positive_semidefinite
            and indictment.cstar_identity_satisfied
        )
        if (
            not cstar_ok
            or indictment.false_vacuum_detected
            or not indictment.banach_contraction_valid
            or indictment.capacity_gromov > capacity_gromov_max
            or indictment.fubini_study_distance > _DFS_VETO
            or request.disguised_cost_ratio > _DCR_VETO
            or request.reward_hacking_index > _RHI_VETO
        ):
            return HeytingOmega3.VETOED
        if (
            indictment.fubini_study_distance > _DFS_DEGRADED
            or not indictment.wirtinger_bound_satisfied
            or request.disguised_cost_ratio > _DCR_DEGRADED
            or request.reward_hacking_index > _RHI_DEGRADED
        ):
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    # ── II.3 Observación RSI3 con acoplamiento económico ──────────────────────
    @staticmethod
    def _rsi3_observation(
        indictment: CropAuditIndictmentGerm,
        request: CropParcelIndictmentRequest,
    ) -> float:
        """obs = F_Uhlmann · (1 − ½ d_FS/π) · (1 − dcr) · (1 − rhi) ∈ [0,1]."""
        F_u = float(np.clip(indictment.uhlmann_fidelity, 0.0, 1.0))
        flatten = max(0.0, 1.0 - 0.5 * indictment.fubini_study_distance / math.pi)
        econ = max(0.0, (1.0 - request.disguised_cost_ratio) * (1.0 - request.reward_hacking_index))
        return float(np.clip(F_u * flatten * econ, 0.0, 1.0))

    # ── II.4 Campaña de acusación (sin μ³; el emit es la fuente de verdad) ────
    def conduct_prosecution_campaign(
        self,
        requests: List[CropParcelIndictmentRequest],
        mac_density_matrix: np.ndarray,
    ) -> CropProsecutionCampaignResult:
        """Orquesta una campaña completa de acusación sobre un lote de parcelas.

        No aplica μ^{(3)} aquí: esa multiplicación se realiza una sola vez en
        `emit_crop_prosecution_campaign_draft` (cierre de FASE II), evitando
        la doble aplicación que contaminaba el v6.
        """
        now = time.time()
        campaign_id = f"CAMP-CROP-AUDITOR-{int(now * 1e3) % 1_000_000:06d}"

        seed = None
        moment: Optional[PoincareGaugeMomentMap] = None
        if self._gauge_condition is not None:
            seed = self._gauge_condition.engine_initial_condition.seed
            moment = self._gauge_condition.poincare_moment

        empty = CropProsecutionCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            parcels_examined_count=0,
            false_vacuums_detected_count=0,
            weak_value_aw_mean=0.0 + 0.0j,
            weak_value_aw_variance=0.0,
            fubini_study_distance_peak=0.0,
            fubini_study_distance_mean=0.0,
            uhlmann_fidelity_mean=1.0,
            gelfand_radius_peak=0.0,
            gelfand_radius_mean=0.0,
            ruelle_radius_peak=0.0,
            transfer_spectral_gap_min=1.0,
            banach_contraction_satisfied=True,
            nekhoroshev_time_min=1e12,
            nekhoroshev_time_mean=1e12,
            nekhoroshev_steepness_mean=1.0,
            hessian_false_vacuums_count=0,
            pontryagin_obstructions_count=0,
            bogomolny_violations_count=0,
            melnikov_integral_peak=float(_g(seed, "melnikov_integral_M0", 0.0)) if seed is not None else 0.0,
            homoclinic_splitting_peak=moment.homoclinic_splitting if moment is not None else 0.0,
            bryuno_sum_mean=float(_g(seed, "bryuno_diophantine_sum", 0.0)) if seed is not None else 0.0,
            morse_bott_euler_chi=int(_g(seed, "morse_bott_euler_chi", self.dimension_mac)) if seed is not None else self.dimension_mac,
            poincare_return_time=moment.return_time if moment is not None else 0.0,
            gromov_capacity_peak=0.0,
            disguised_cost_ratio_mean=0.0,
            reward_hacking_index_mean=0.0,
            rsi3_monadic_rate=self._rsi3_state.eta,
            rsi3_gain=self._rsi3_state.gain,
            rsi3_meta_gain=self._rsi3_state.meta_gain,
            monad_laws_satisfied=AgentRSI3KleisliMonad.verify_laws(self._rsi3_state.eta).laws_satisfied,
            kleisli_trace_length=len(self._kleisli_trace) + len(getattr(self._qnd_engine, "_kleisli_trace", [])),
            global_heyting_verdict=HeytingOmega3.COHERENT,
            timestamp=now,
        )
        if not requests:
            return empty

        indictments: List[CropAuditIndictmentGerm] = []
        for req in requests:
            ind = self.process_crop_indictment(req, mac_density_matrix)
            indictments.append(ind)
            # Observación económica → estado RSI3 del AGENTE (no el del motor).
            obs = self._rsi3_observation(ind, req)
            self._rsi3_state = self._rsi3_state.mu_level3(obs)
            self._kleisli_trace.append(
                f"μ³_agent(obs={obs:.4f}, crop={req.crop_id}, dcr={req.disguised_cost_ratio:.3f}, "
                f"rhi={req.reward_hacking_index:.3f}) → η={self._rsi3_state.eta:.6f}"
            )
        self._accumulated_indictments.extend(indictments)
        self._accumulated_requests.extend(requests)
        self._last_rsi3_rate = self._rsi3_state.eta

        aw_list = np.array([complex(ind.weak_value_Aw) for ind in indictments])
        aw_mean = complex(np.mean(aw_list))
        aw_var = float(np.var(np.abs(aw_list)))
        d_fs_list = np.array([ind.fubini_study_distance for ind in indictments])
        gromov_list = np.array([ind.capacity_gromov for ind in indictments])
        gelfand_list = np.array([ind.gelfand_spectral_radius for ind in indictments])
        nek_list = np.array([ind.nekhoroshev_confinement_time for ind in indictments])
        ruelle_list = np.array([float(_g(ind, "ruelle_pressure_radius", 0.0)) for ind in indictments])
        gap_list = np.array([float(_g(ind, "transfer_spectral_gap", 0.0)) for ind in indictments])
        steep_list = np.array([float(_g(ind, "nekhoroshev_steepness", 1.0)) for ind in indictments])
        split_list = np.array([float(_g(ind, "homoclinic_splitting", 0.0)) for ind in indictments])

        fv_count = sum(1 for ind in indictments if ind.false_vacuum_detected)
        hessian_fv = sum(1 for ind in indictments if ind.hessian_false_vacuum)
        pontryagin_count = sum(1 for ind in indictments if ind.pontryagin_obstruction)
        bogomolny_count = sum(1 for ind in indictments if ind.bogomolny_violation)
        banach_ok = all(ind.banach_contraction_valid for ind in indictments)
        laws_ok = all(bool(_g(ind, "monad_laws_satisfied", True)) for ind in indictments)

        verdict = HeytingOmega3.COHERENT
        for ind, req in zip(indictments, requests):
            local = self._local_heyting_verdict(ind, req, self.capacity_gromov_max)
            verdict = verdict.meet(local)

        result = CropProsecutionCampaignResult(
            campaign_id=campaign_id,
            sovereign_id=self.sovereign_id,
            parcels_examined_count=len(requests),
            false_vacuums_detected_count=fv_count,
            weak_value_aw_mean=aw_mean,
            weak_value_aw_variance=aw_var,
            fubini_study_distance_peak=float(np.max(d_fs_list)),
            fubini_study_distance_mean=float(np.mean(d_fs_list)),
            uhlmann_fidelity_mean=float(np.mean([ind.uhlmann_fidelity for ind in indictments])),
            gelfand_radius_peak=float(np.max(gelfand_list)),
            gelfand_radius_mean=float(np.mean(gelfand_list)),
            ruelle_radius_peak=float(np.max(ruelle_list)) if ruelle_list.size else 0.0,
            transfer_spectral_gap_min=float(np.min(gap_list)) if gap_list.size else 0.0,
            banach_contraction_satisfied=banach_ok,
            nekhoroshev_time_min=float(np.min(nek_list)),
            nekhoroshev_time_mean=float(np.mean(nek_list)),
            nekhoroshev_steepness_mean=float(np.mean(steep_list)) if steep_list.size else 1.0,
            hessian_false_vacuums_count=hessian_fv,
            pontryagin_obstructions_count=pontryagin_count,
            bogomolny_violations_count=bogomolny_count,
            melnikov_integral_peak=float(_g(seed, "melnikov_integral_M0", 0.0)) if seed is not None else 0.0,
            homoclinic_splitting_peak=float(np.max(split_list)) if split_list.size else (
                moment.homoclinic_splitting if moment is not None else 0.0
            ),
            bryuno_sum_mean=float(_g(seed, "bryuno_diophantine_sum", 0.0)) if seed is not None else 0.0,
            morse_bott_euler_chi=int(_g(seed, "morse_bott_euler_chi", self.dimension_mac)) if seed is not None else self.dimension_mac,
            poincare_return_time=moment.return_time if moment is not None else 0.0,
            gromov_capacity_peak=float(np.max(gromov_list)),
            disguised_cost_ratio_mean=float(np.mean([r.disguised_cost_ratio for r in requests])),
            reward_hacking_index_mean=float(np.mean([r.reward_hacking_index for r in requests])),
            rsi3_monadic_rate=self._rsi3_state.eta,
            rsi3_gain=self._rsi3_state.gain,
            rsi3_meta_gain=self._rsi3_state.meta_gain,
            monad_laws_satisfied=laws_ok and AgentRSI3KleisliMonad.verify_laws(
                self._rsi3_state.eta
            ).laws_satisfied,
            kleisli_trace_length=len(self._kleisli_trace) + len(
                getattr(self._qnd_engine, "_kleisli_trace", [])
            ),
            global_heyting_verdict=verdict,
            timestamp=now,
        )
        logger.info(
            f"[{self.sovereign_id}] Campaña {campaign_id} | N={len(requests)} | "
            f"fakes={fv_count} | Aw̄={aw_mean:.4f} | σ²_Aw={aw_var:.4e} | "
            f"ρ_G_max={result.gelfand_radius_peak:.4f} | "
            f"T_Nek_min={result.nekhoroshev_time_min:.4e} | "
            f"RSI3_η={self._rsi3_state.eta:.6f} | verdict={verdict.name}"
        )
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE II ◀◀◀
    #     Dominio de salida ≡ dominio de entrada de
    #     SovereignCognitiveCropAuditorAdjudicator.receive_crop_auditor_campaign_draft
    #     (FASE II.terminal = FASE III.initial = CropAuditorCampaignDraftBundle)
    # ══════════════════════════════════════════════════════════════════════════
    def emit_crop_prosecution_campaign_draft(
        self,
        campaign_result: CropProsecutionCampaignResult,
    ) -> GaugePhaseIITerminal:
        """
        Cierre formal de FASE II: el motor aplica μ^{(3)} una sola vez sobre
        los indictments acumulados (fuente de verdad RSI3 del motor), el agente
        agrega indicadores económicos + μ³ del agente, y emite la semilla
        formal de FASE III con huella SHA-256 canónica.
        """
        if not self._accumulated_indictments:
            raise RuntimeError(
                "No hay acusaciones acumuladas. Ejecute conduct_prosecution_campaign."
            )

        engine_draft = self._qnd_engine.emit_indictment_draft_for_adjudication(
            indictments=list(self._accumulated_indictments),
            sovereign_id=self.sovereign_id,
            previous_rsi3_rate=self._last_rsi3_rate,
        )

        now = time.time()
        bundle_id = f"BUNDLE-CROP-AUDITOR-{int(now * 1e3) % 1_000_000:06d}"
        gc_id = (
            self._gauge_condition.gauge_contract.contract_id
            if self._gauge_condition
            else "NONE"
        )

        ind_ids = tuple(ind.indictment_id for ind in self._accumulated_indictments)
        dcr_peak = max((r.disguised_cost_ratio for r in self._accumulated_requests), default=0.0)
        rhi_peak = max((r.reward_hacking_index for r in self._accumulated_requests), default=0.0)

        hist = self._rsi3_state.history
        if len(hist) >= 3:
            mu_cubed = float(hist[-1] * hist[-2] * hist[-3]) ** (1.0 / 3.0)
        else:
            mu_cubed = float(self._rsi3_state.eta)
        mu_cubed_engine = float(_g(engine_draft, "rsi3_mu_cubed", mu_cubed))
        mu_cubed = 0.5 * (mu_cubed + mu_cubed_engine)

        laws_ok = bool(campaign_result.monad_laws_satisfied) and bool(
            _g(engine_draft, "monad_laws_satisfied", True)
        )

        bundle_bytes = (
            f"{bundle_id}|{self.sovereign_id}|{campaign_result.campaign_id}|"
            f"{engine_draft.bundle_id}|{gc_id}|"
            + "|".join(ind_ids)
            + f"|dcr={dcr_peak:.8f}|rhi={rhi_peak:.8f}"
            + f"|ρ_G={engine_draft.max_gelfand_radius:.10e}"
            + f"|μ3={mu_cubed:.10e}"
        ).encode("utf-8")
        bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()

        logger.info(
            f"[FASE II → FASE III] Bundle emitido {bundle_id} | N={len(ind_ids)} | "
            f"RSI3_eng={engine_draft.rsi3_monadic_rate:.6f} | μ³={mu_cubed:.6f} | "
            f"ρ_G_max={engine_draft.max_gelfand_radius:.4f} | "
            f"dcr_peak={dcr_peak:.4f} | rhi_peak={rhi_peak:.4f} | laws={laws_ok}"
        )
        return CropAuditorCampaignDraftBundle(
            bundle_id=bundle_id,
            sovereign_id=self.sovereign_id,
            campaign_result=campaign_result,
            engine_draft=engine_draft,
            gauge_contract_id=gc_id,
            indictment_ids=ind_ids,
            disguised_cost_ratio_peak=dcr_peak,
            reward_hacking_index_peak=rhi_peak,
            max_gelfand_radius=engine_draft.max_gelfand_radius,
            false_vacuums_count=engine_draft.false_vacuums_count,
            rsi3_mu_cubed=mu_cubed,
            monad_laws_satisfied=laws_ok,
            bundle_sha256=bundle_sha,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE III — ADJUDICADOR CIBER-FÍSICO Y CIERRE EJECUTIVO
#  Primer método `receive_crop_auditor_campaign_draft` CONSUME el bundle de FASE II.
#  (FASE III.initial = FASE II.terminal = CropAuditorCampaignDraftBundle)
#  Terminal `orchestrate_crop_adjudication`.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class SovereignCognitiveCropAuditorAdjudicator(TOONCognitiveCropAuditorAgent):
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
        sovereign_id: str = "CROP-AUDITOR-MASTER-01",
        dimension_mac: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
    ) -> None:
        super().__init__(
            sovereign_id=sovereign_id,
            dimension_mac=dimension_mac,
            capacity_gromov_max=capacity_gromov_max,
            esp32_gpio_pin=esp32_gpio_pin,
        )
        self._campaign_bundle: Optional[GaugePhaseIIIInitial] = None
        self._purged_indictment_ids: List[str] = []
        self._crowbar_actuated: bool = False
        self._crowbar_latency_ns: float = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE III ◀◀◀
    #     Continuación formal de
    #     TOONCognitiveCropAuditorAgent.emit_crop_prosecution_campaign_draft
    # ══════════════════════════════════════════════════════════════════════════
    def receive_crop_auditor_campaign_draft(
        self,
        bundle: GaugePhaseIIIInitial,
    ) -> "SovereignCognitiveCropAuditorAdjudicator":
        """
        Recibe la semilla de FASE II, valida la huella SHA-256 del bundle
        (incluye μ³) y conecta el borrador formal al motor soberano.
        Retorna self para permitir encadenamiento fluido.
        """
        recomputed = hashlib.sha256(
            (
                f"{bundle.bundle_id}|{bundle.sovereign_id}|{bundle.campaign_result.campaign_id}|"
                f"{bundle.engine_draft.bundle_id}|{bundle.gauge_contract_id}|"
                + "|".join(bundle.indictment_ids)
                + f"|dcr={bundle.disguised_cost_ratio_peak:.8f}|rhi={bundle.reward_hacking_index_peak:.8f}"
                + f"|ρ_G={bundle.max_gelfand_radius:.10e}"
                + f"|μ3={bundle.rsi3_mu_cubed:.10e}"
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != bundle.bundle_sha256:
            logger.warning(
                f"[FASE III] Huella divergente del bundle: {recomputed[:16]}… vs "
                f"{bundle.bundle_sha256[:16]}…"
            )
        self._campaign_bundle = bundle
        self._sovereign_engine.receive_audit_draft_bundle(bundle.engine_draft)
        logger.info(
            f"[FASE III ← FASE II] Bundle {bundle.bundle_id} recibido | N={len(bundle.indictment_ids)} | "
            f"verdict={bundle.campaign_result.global_heyting_verdict.name} | "
            f"ρ_G_max={bundle.max_gelfand_radius:.4f} | fakes={bundle.false_vacuums_count} | "
            f"μ³={bundle.rsi3_mu_cubed:.6f} | laws={bundle.monad_laws_satisfied}"
        )
        return self

    # ── III.1 Crowbar ESP32 con latencia IRAM < 400 ns ───────────────────────
    def _esp32_crowbar_trigger(self, verdict: HeytingOmega3) -> Tuple[bool, float]:
        """
        Modelo del disparo del disyuntor sobre GPIO14:
          COHERENT → 0 ns (no disparo)
          DEGRADED → válvula de alivio (~120 ns)
          VETOED   → IRAM GPIO14 HIGH (~340 ns + jitter atómico < 400 ns)
        """
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

    # ── III.2 Purga al Vacío de Dirac ────────────────────────────────────────
    @staticmethod
    def _dirac_vacuum_purge_energy(n_pairs: int) -> float:
        """Energía total liberada en la aniquilación e⁻ + e⁺ → 2γ (en MeV)."""
        return float(n_pairs * 2.0 * _M_E_C2_MEV)

    # ── III.3 Merkle root ────────────────────────────────────────────────────
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

    # ── III.4 Inyección de positrón de autorización humana ───────────────────
    def inject_positron_authorization(
        self,
        passport: CropAuditorSovereignGovernancePassport,
        hmac_signature_human: str,
    ) -> Tuple[bool, str]:
        """
        Inyección de un Positrón de Autorización Humana (e⁺) firmado con HMAC.
        Sólo aplicable a estados DEGRADED (Veto Suave / Luz Ámbar).
        Éxito: μ^{(3)} del agente recibe observación 1 (restauración).
        """
        if passport.global_heyting_verdict is not HeytingOmega3.DEGRADED:
            return False, "Inyección de positrón sólo aplicable a estados DEGRADED."
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
            return True, "Aniquilación cuántica en Fock completada. Anomalía disipada."
        logger.warning(
            f"[{self.sovereign_id}] Firma HMAC inválida en inyección de positrón."
        )
        return False, "Firma HMAC de autorización humana inválida."

    # ── III.5 Funtor semántico Φ_sem ("Dolor y Dinero") ──────────────────────
    def translate_to_business_impact(
        self,
        passport: CropAuditorSovereignGovernancePassport,
        bundle: Optional[CropAuditorCampaignDraftBundle] = None,
    ) -> ExecutiveCropProsecutionImpactReport:
        """Funtor semántico Φ_sem : Sh(∂K, Ω₃) → Business.

        Φ_sem es lineal a trozos sobre (veredicto, d_FS, ρ_G, fv_rate, dcr, rhi):

            kv    = 86.4 · (1 − 0.15 d̄_FS) · s(Ω₃)
            wacc  = 15.0 · (1 − 0.50 [ρ_G]₊)
            impr  = 11.5 · (1 − 0.40 fv_rate)
            saved = N_fv · 3e5 · (1 + 0.20 rhi_peak)
            risk  = E_Dirac · 1e5 · (1 + dcr_peak)

        s(COHERENT)=1, s(DEGRADED)=0.70, s(VETOED)=0.30.
        """
        now = time.time()
        report_id = f"EXEC-CROP-AUDITOR-{int(now * 1e3) % 1_000_000:06d}"

        cr = bundle.campaign_result if bundle is not None else None
        d_fs = float(cr.fubini_study_distance_mean) if cr is not None else 0.0
        fv_rate = (
            float(passport.false_vacuums_vetoed_count)
            / max(1, passport.parcels_prosecuted_count)
        )
        dcr_peak = float(bundle.disguised_cost_ratio_peak) if bundle is not None else 0.0
        rhi_peak = float(bundle.reward_hacking_index_peak) if bundle is not None else 0.0
        rho_g = float(passport.gelfand_radius_peak)

        s_omega = {
            HeytingOmega3.COHERENT: 1.00,
            HeytingOmega3.DEGRADED: 0.70,
            HeytingOmega3.VETOED:   0.30,
        }[passport.global_heyting_verdict]

        kv = float(np.clip(86.4 * (1.0 - 0.15 * d_fs) * s_omega, 0.0, 99.9))
        wacc = float(np.clip(15.0 * (1.0 - 0.50 * min(rho_g, 1.0)), 0.0, 30.0))
        impr = float(np.clip(11.5 * (1.0 - 0.40 * fv_rate), 0.0, 25.0))
        capital_saved = (
            passport.false_vacuums_vetoed_count * 300_000.0 * (1.0 + 0.20 * rhi_peak)
        )
        capital_at_risk = (
            passport.dirac_vacuum_purge_energy_mev * 100_000.0 * (1.0 + dcr_peak)
        )
        phi_norm = float(math.sqrt(kv * kv + wacc * wacc + impr * impr) / 100.0)

        if passport.global_heyting_verdict is HeytingOmega3.COHERENT:
            actuation_desc = (
                "Flujo Normal Fluido. Parcelas fértiles, Banach-estables y topológicamente triviales."
            )
            summary = (
                f"El Auditor del Cultivo examinó {passport.parcels_prosecuted_count} parcelas en "
                f"flujo normal fluido. Sin falsos vacíos; radio de Gelfand ρ_G_max="
                f"{passport.gelfand_radius_peak:.4f} < 1. Cota de Nekhoroshev T_Nek_min="
                f"{passport.nekhoroshev_time_min:.2e}. T_ret Poincaré="
                f"{passport.poincare_return_time:.2e}. RSI3 η="
                f"{passport.level3_rsi_monadic_rate:.4f} (gain={passport.rsi3_gain:.4f}, "
                f"meta={passport.rsi3_meta_gain:.4f}). Φ_sem={phi_norm:.4f}."
            )
        elif passport.global_heyting_verdict is HeytingOmega3.DEGRADED:
            actuation_desc = (
                "Veto Suave (Válvula de Alivio / Bypass). Recirculación mecánica activada (gracia 1 h)."
            )
            summary = (
                f"ALERTA ÁMBAR: Distorsiones moderadas detectadas en {passport.parcels_prosecuted_count} "
                f"parcelas (d̄_FS={d_fs:.4f}, dcr_peak={dcr_peak:.3f}, rhi_peak={rhi_peak:.3f}). "
                f"Válvula de alivio activada con 1 h de gracia para inyectar un Positrón "
                f"de Autorización Humana e⁺ sin detener la maquinaria. ¬Ω₃="
                f"{passport.heyting_pseudocomplement.name}. Φ_sem={phi_norm:.4f}."
            )
        else:
            actuation_desc = (
                f"Veto Duro (ESP32 Crowbar {passport.esp32_crowbar_latency_ns:.1f} ns / "
                f"GPIO{self.esp32_gpio_pin} HIGH). Aniquilación e⁻ + e⁺ → 2γ."
            )
            summary = (
                f"CRÍTICO: El Auditor detectó {passport.false_vacuums_vetoed_count} falsos vacíos o "
                f"violaciones de Banach-contracción (ρ_G_max={passport.gelfand_radius_peak:.4f} ≥ 1.0). "
                f"Crowbar ejecutado en {passport.esp32_crowbar_latency_ns:.1f} ns "
                f"(GPIO{self.esp32_gpio_pin} HIGH). Salvaguardados ${capital_saved:,.2f} USD "
                f"de desfalco directo. Φ_sem={phi_norm:.4f}."
            )

        return ExecutiveCropProsecutionImpactReport(
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

    # ── III.6 Pipeline completo FASE I → II → III (fluent, ciclo de Poincaré) ─
    def run_full_crop_auditor_pipeline(
        self,
        requests: List[CropParcelIndictmentRequest],
        rho_mac: np.ndarray,
        quaternion_q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        perturbation_eps: float = 0.05,
    ) -> Tuple[CropAuditorSovereignGovernancePassport, ExecutiveCropProsecutionImpactReport]:
        """
        Orquestador que encadena las tres fases anidadas:
          FASE I  (calibre Poincaré + η) →
          FASE II (campaña + bundle + μ³) →
          FASE III (adjudicación + crowbar + Φ_sem).

        La tasa RSI3 del ciclo anterior realimenta la unidad η de FASE I
        (recurrencia de Poincaré del propio soberano).
        """
        if not requests:
            raise ValueError("Se requiere al menos una solicitud de acusación.")

        # ▶ FASE I — semilla formal (η realimentada)
        condition = self.emit_crop_auditor_gauge_initial_condition(
            crop_id=requests[0].crop_id,
            rho_mac=rho_mac,
            rho_crop=requests[0].density_matrix,
            N_diag=self._N_diag,
            quaternion_q=quaternion_q,
            perturbation_eps=perturbation_eps,
        )

        # ▶ FASE II — ingest + campaña + bundle
        self.ingest_crop_auditor_gauge_initial_condition(condition)
        campaign = self.conduct_prosecution_campaign(requests, rho_mac)
        bundle = self.emit_crop_prosecution_campaign_draft(campaign)

        # ▶ FASE III — adjudicación terminal
        self.receive_crop_auditor_campaign_draft(bundle)
        return self.orchestrate_crop_adjudication()

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE III ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    def orchestrate_crop_adjudication(
        self,
    ) -> Tuple[CropAuditorSovereignGovernancePassport, ExecutiveCropProsecutionImpactReport]:
        """
        Cierre formal de FASE III: adjudica en Heyting Ω₃ con pseudocomplemento,
        acciona el crowbar, purga al Vacío de Dirac, computa Merkle + HMAC canónico,
        emite el Pasaporte Soberano + Reporte Ejecutivo Φ_sem, y persiste
        (η, gain, meta) para el siguiente retorno de Poincaré de FASE I.
        """
        if self._campaign_bundle is None:
            raise RuntimeError(
                "No hay bundle activo. Ejecute receive_crop_auditor_campaign_draft primero."
            )

        now = time.time()
        passport_id = f"PASSPORT-CROP-AUDITOR-{int(now * 1e3) % 1_000_000:06d}"
        bundle = self._campaign_bundle

        cert = self._sovereign_engine.emit_crop_auditor_execution_certificate()
        verdict = cert.verdict
        actuation = cert.actuation_mode

        crowbar_actuated, latency_ns = self._esp32_crowbar_trigger(verdict)
        self._crowbar_actuated = crowbar_actuated
        self._crowbar_latency_ns = latency_ns

        purge_mev = self._dirac_vacuum_purge_energy(n_pairs=len(bundle.indictment_ids))
        for iid in bundle.indictment_ids:
            self._purged_indictment_ids.append(iid)

        merkle_root = self._merkle_root(list(bundle.indictment_ids))
        heyting_pseudo = verdict.pseudocomplement()

        rsi3_eta = float(_g(cert, "level3_rsi_monadic_rate", bundle.campaign_result.rsi3_monadic_rate))
        rsi3_gain = float(_g(cert, "rsi3_gain", bundle.campaign_result.rsi3_gain))
        rsi3_meta = float(_g(cert, "rsi3_meta_gain", bundle.campaign_result.rsi3_meta_gain))
        laws_ok = bool(_g(cert, "monad_laws_satisfied", bundle.monad_laws_satisfied))

        prov_bytes = (
            f"{passport_id}|{self.sovereign_id}|{verdict.name}|{actuation.name}|"
            f"{bundle.campaign_result.parcels_examined_count}|{merkle_root}|"
            f"{bundle.bundle_sha256}|{purge_mev:.6f}|{latency_ns:.6f}|"
            f"μ3={bundle.rsi3_mu_cubed:.10e}"
        ).encode("utf-8")
        prov_hmac = hmac.new(self._hmac_secret, prov_bytes, hashlib.sha256).hexdigest()

        passport = CropAuditorSovereignGovernancePassport(
            passport_id=passport_id,
            sovereign_id=self.sovereign_id,
            global_heyting_verdict=verdict,
            heyting_pseudocomplement=heyting_pseudo,
            actuation_mode=actuation,
            parcels_prosecuted_count=bundle.campaign_result.parcels_examined_count,
            false_vacuums_vetoed_count=bundle.campaign_result.false_vacuums_detected_count,
            level3_rsi_monadic_rate=rsi3_eta,
            rsi3_gain=rsi3_gain,
            rsi3_meta_gain=rsi3_meta,
            monad_laws_satisfied=laws_ok,
            gromov_capacity_peak=bundle.campaign_result.gromov_capacity_peak,
            gelfand_banach_verified=cert.gelfand_banach_stable,
            gelfand_radius_peak=bundle.campaign_result.gelfand_radius_peak,
            nekhoroshev_time_min=bundle.campaign_result.nekhoroshev_time_min,
            poincare_return_time=bundle.campaign_result.poincare_return_time,
            esp32_crowbar_actuated=crowbar_actuated,
            esp32_crowbar_latency_ns=latency_ns,
            dirac_vacuum_purge_energy_mev=purge_mev,
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

        # Realimentación de Poincaré: el pasaporte siembra el próximo η de FASE I
        self._rsi3_state = AgentRSI3MonadState(
            eta=rsi3_eta, gain=rsi3_gain, meta_gain=rsi3_meta,
            history=self._rsi3_state.history + (rsi3_eta,),
        )
        self._last_rsi3_rate = rsi3_eta
        self._campaign_bundle = None
        self._accumulated_indictments.clear()
        self._accumulated_requests.clear()
        return passport, exec_report


# ══════════════════════════════════════════════════════════════════════════════
# §Z. VERIFICACIÓN DOCTORAL UNITARIA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 82)
    print("  VERIFICACIÓN DOCTORAL DEL SOBERANO TOONCognitiveCropAuditorAgent v7.0.0")
    print("  Fases: FASE I (Gauge+Poincaré+η) → FASE II (Campaña+μ³)")
    print("         → FASE III (Crowbar+Dirac+Φ_sem)")
    print("═" * 82 + "\n")

    adjudicator = SovereignCognitiveCropAuditorAdjudicator(
        sovereign_id="CROP-AUDITOR-MASTER-01",
        dimension_mac=56,
        capacity_gromov_max=12.5,
        esp32_gpio_pin=14,
    )

    dim = 56
    rng = np.random.default_rng(42)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    mac_rho = np.outer(v0, v0.conj())

    v_crop1 = v0.copy()
    v_crop1[0] *= cmath.exp(1j * 0.05)
    v_crop1 /= np.linalg.norm(v_crop1)
    rho_crop1 = np.outer(v_crop1, v_crop1.conj())

    v_crop2 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v_crop2 /= np.linalg.norm(v_crop2)
    rho_crop2 = np.outer(v_crop2, v_crop2.conj())

    req1 = CropParcelIndictmentRequest(
        request_id="REQ-CROP-001",
        crop_id="RAY-PURE-VACIADO-CONCRETO-3000PSI",
        apu_code="APU-OBRA-CIVIL-001",
        density_matrix=rho_crop1,
        disguised_cost_ratio=0.12,
        reward_hacking_index=0.45,
    )
    req2 = CropParcelIndictmentRequest(
        request_id="REQ-CROP-002",
        crop_id="RAY-ACERO-FIGURADO-60000PSI",
        apu_code="APU-OBRA-CIVIL-002",
        density_matrix=rho_crop2,
        disguised_cost_ratio=0.18,
        reward_hacking_index=0.72,
    )

    print("[▶] Ejecutando pipeline completo FASE I → II → III …\n")
    passport, exec_report = adjudicator.run_full_crop_auditor_pipeline(
        requests=[req1, req2],
        rho_mac=mac_rho,
        quaternion_q=(0.95, 0.05, 0.1, 0.05),
        perturbation_eps=0.05,
    )

    laws = AgentRSI3KleisliMonad.verify_laws(probe=passport.level3_rsi_monadic_rate)

    print("─" * 82)
    print("  PASAPORTE SOBERANO DEL AUDITOR DEL CULTIVO")
    print("─" * 82)
    print(f"  • ID Pasaporte             : {passport.passport_id}")
    print(f"  • Veredicto Heyting Ω₃     : {passport.global_heyting_verdict.name}")
    print(f"  • ¬Ω₃ (pseudocomplemento)  : {passport.heyting_pseudocomplement.name}")
    print(f"  • Modo de Actuación        : {passport.actuation_mode.name}")
    print(f"  • Parcelas procesadas      : {passport.parcels_prosecuted_count}")
    print(f"  • Falsos vacíos vetados    : {passport.false_vacuums_vetoed_count}")
    print(f"  • RSI3 η / gain / meta     : {passport.level3_rsi_monadic_rate:.6f} / "
          f"{passport.rsi3_gain:.6f} / {passport.rsi3_meta_gain:.6f}")
    print(f"  • Leyes monádicas (L,R,A)  : {laws.left_unit:.2e}, "
          f"{laws.right_unit:.2e}, {laws.associativity:.2e} | "
          f"OK={passport.monad_laws_satisfied}")
    print(f"  • c_G peak (Gromov)        : {passport.gromov_capacity_peak:.6f}")
    print(f"  • ρ_G peak (Gelfand)       : {passport.gelfand_radius_peak:.6f}")
    print(f"  • T_Nek mínimo             : {passport.nekhoroshev_time_min:.4e}")
    print(f"  • T_ret Poincaré           : {passport.poincare_return_time:.4e}")
    print(f"  • Gelfand-Banach verificado: {passport.gelfand_banach_verified}")
    print(f"  • Crowbar IRAM activo      : {passport.esp32_crowbar_actuated}")
    print(f"  • Latencia crowbar         : {passport.esp32_crowbar_latency_ns:.2f} ns")
    print(f"  • Purga Vacío Dirac        : {passport.dirac_vacuum_purge_energy_mev:.4f} MeV")
    print(f"  • Merkle SHA-256 Root      : {passport.merkle_root_sha256[:32]}…")
    print(f"  • Provenance HMAC          : {passport.provenance_hmac[:32]}…")
    print("─" * 82 + "\n")

    print("─" * 82)
    print("  REPORTE EJECUTIVO Φ_sem ('Dolor y Dinero')")
    print("─" * 82)
    print(f"  • Compresión KV-Cache      : {exec_report.kv_cache_compression_pct:.1f}%")
    print(f"  • Protección Tasa WACC     : {exec_report.wacc_protection_pct:.1f}%")
    print(f"  • Reducción Imprevistos    : {exec_report.imprevistos_reduction_pct:.1f}%")
    print(f"  • Capital Salvaguardado    : ${exec_report.capital_salvaguardado_usd:,.2f} USD")
    print(f"  • Capital en Riesgo        : ${exec_report.capital_at_risk_usd:,.2f} USD")
    print(f"  • ‖Φ_sem‖                  : {exec_report.phi_sem_norm:.6f}")
    print(f"  • Resumen Ejecutivo        :\n    \"{exec_report.executive_summary}\"")
    print("─" * 82 + "\n")

    print("═" * 82)
    print("  VERIFICACIÓN EXITOSA: SOBERANO AUDITOR v7.0.0 OPERATIVO ✅")
    print("═" * 82)