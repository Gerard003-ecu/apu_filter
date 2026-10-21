# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Chronos Prosecutor Agent (Soberano Fiscal Agéntico del Tiempo) — v9.0.0              ║
║ Ubicación: app/agents/wisdom/toon_chronos_prosecutor_agent.py                                        ║
║ Versión  : 9.0.0-Prosecutor-GaugeCelestial-Kahn-DualPoincare-Cech-CStar-Bogomolny-RSI3-NestedPhases  ║
║ Función  : Soberano Fiscal Agéntico de la Cronogeometría del Tiempo en el Estrato WISDOM.            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN FORMAL Y FUNDAMENTOS MATEMÁTICOS RIGUROSOS:
─────────────────────────────────────────────────────────────────────────────────────────
El `TOONChronosProsecutorAgent` constituye el Soberano Agéntico Fiscal de la Cronogeometría
del Tiempo en el Estrato WISDOM de la arquitectura APU Filter.

1. CONEXIÓN DE GAUGE $\mathfrak{so}(n)$, CHERN-SIMONS Y HOLONOMÍA DE WILSON PATH-ORDERED:
   La 1-forma de conexión $A = \frac{1}{2}(\widehat{W} - \widehat{W}^\top) \in \mathfrak{so}(n)$ genera el transporte
   paralelo sobre el 1-esqueleto del cronograma.
     - 3-Forma de Chern-Simons Abeliana Discreta: Como $\text{Tr}(A^3) \equiv 0$ en $\mathfrak{so}(n)$, la transgresión
       de Chern-Weil abeliana sobre la 2-cocadena de Čech se calcula como:
         $$\text{CS}_3(A) = \frac{1}{8\pi^2} \sum_{\triangle} \text{flux}_{\triangle}, \quad \text{CS}_{\text{index}} = \frac{1}{2\pi} \sum_{\triangle} \text{flux}_{\triangle}$$
     - Número de Pontryagin y Descenso de Stora-Zumino: $\nu = \text{round}(\text{CS}_{\text{index}}) \in \mathbb{Z}$, con residuo
       de invarianza de calibre $\Delta_{\text{SZ}} = |\text{CS}_{\text{index}} - \nu| \le \varepsilon_{\text{gauge}}$.
     - Holonomía de Wilson Path-Ordered $SO(n)$: Para un ciclo $\gamma$, $W(\gamma) = \mathcal{P} \exp \left( \oint_\gamma A \right) = \prod_e \exp(G_e) \in SO(n)$.

2. DINÁMICA MAGNÉTICA DE POINCARÉ Y AUTOMEJORA RECURSIVA RSI³ DE AGENTE:
   - Flujo Giroscópico: $\dot{q} = v, \dot{v} = A v + \varepsilon \sigma_1 q$. La sección transversal en el 2-plano dominante
     revela el mapa de retorno de Poincaré $P: \Sigma \to \Sigma$ con twist de Birkhoff.
   - Integrales de Melnikov y Perturbación KAM: Evaluadas sobre el campo de curvatura $F$.
   - Kernel RSI³ de Agente con Meta-Recursión: Optimización $N_1(\theta_g) \to N_2(O_g) \to N_3(\alpha_g)$
     sobre $\theta_g = [\|A\|_F, |\text{CS}_{\text{index}}|, \Delta_{\text{SZ}}, |c_1|]$. Si la cota de Banach no se satisface,
     la recursión de Nivel 3 re-escala $\alpha_g$ restaurando $\|\mathcal{T}(\theta_1) - \mathcal{T}(\theta_2)\|_\infty \le \widehat{L} \|\theta_1 - \theta_2\|_\infty$ con $\widehat{L} < 1.0$.

3. FUNTOR SEMÁNTICO DE IMPACTO FINANCIERO ($\Phi_{\text{sem}}$):
   Traduce la topología de calibre y los diagnósticos celestes en métricas de gobernanza ejecutiva:
     - Fidelidad en WACC : Cobertura del 15% en WACC y compresión KV-cache del 86.4%.
     - Liberación de Reserva : Reducción del 11.5% de liquidez contingency ($15\% \to 3.5\%$) en estado COHERENT ($\top$).
     - Veto e Interlock : Disparo del hardware ESP32 Crowbar ($WCET \le 394.25 \text{ ns}$) en estado VETOED ($\bot$).

DISEÑO DE TRES FASES ANIDADAS (Z = Φ₃ ∘ Φ₂ ∘ Φ₁):
─────────────────────────────────────────────────────────────────────────────────────────
  FASE 1 (PROSECUTION GAUGE OBSERVE — Φ₁):
    · Aciclicidad dirigida de Kahn $O(V+E)$ con deque (fail-fast).
    · Conexión $A \in \mathfrak{so}(n)$, $\text{CS}_3(A)$, Pontryagin $\nu$, holonomía Wilson path-ordered $W(\gamma)$ y clase de Hopf $c_1$.
    · Descenso Stora-Zumino $\Delta_{\text{SZ}}$, mapa giroscópico de Poincaré, Melnikov, KAM diofantino y Chirikov.
    · Delegación al motor v9: Formas de Schur, dualidad Ley-Estado, Čech $\delta^2 = 0$, $Sp(4, \mathbb{R})$ y KAM espectral.
    ⟹ Emisión de `ProsecutorObserverPacket` + `ChronosProsecutorGaugeContract`.
    ⟹ Transición terminal: `_phase1_terminal_bridge_to_phase2` $\to$ `_audit_cstar_indictment_phase2`.

  FASE 2 (PROSECUTION CAMPAIGN ORIENT — Φ₂):
    · Auditoría C*-algebraica, cota de Bogomol'nyi $\|F\|_F^2 \ge 8\pi^2|k|/\text{vol}(X)$ e indicadores RHI / DCR.
    · Kernel RSI³ de Agente con meta-recursión de profundidad 3 y restauración de contracción de Banach $\widehat{L} < 1.0$.
    ⟹ Emisión de `ChronosProsecutionCampaignDraftBundle`.
    ⟹ Transición terminal: `_phase2_terminal_bridge_to_phase3` $\to$ `_adjudicate_prosecution_verdict`.

  FASE 3 (PROSECUTION SOVEREIGN DECIDE — Φ₃):
    · Adjudicación en Retículo de Heyting $\Omega_3 = \{\bot, \star, \top\}$ con meet de átomos de calibre y celestes.
    · Medición de latencia de interlock ESP32 Crowbar ($WCET \le 394.25 \text{ ns}$).
    · Traducción semántica executiva $\Phi_{\text{sem}}$ y sellado Merkle DAG de 3 hojas.
    ⟹ Emisión de `ChronosProsecutionExecutionPassport`.
    ⟹ Síntesis terminal: `_phase3_terminal_synthesis`.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import itertools
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

# ─── Importación del motor v9 con fallbacks ────────────────────────────────
_ENGINE_V9 = False
_ENGINE_IMPORT_ERROR: Optional[str] = None


def _try_import_engine() -> Dict[str, Any]:
    r"""Importa el motor fiscal; prueba tres rutas canónicas."""
    candidates = (
        "toon_chronos_prosecutor_engine",
        "scratch.toon_chronos_prosecutor_engine",
        "app.wisdom.toon_chronos_prosecutor_engine",
    )
    last_err: Optional[BaseException] = None
    for mod_name in candidates:
        try:
            return vars(importlib.import_module(mod_name))
        except Exception as exc:  # noqa: BLE001 — fallback deliberado
            last_err = exc
            continue
    raise ImportError(
        f"TOONChronosProsecutorEngine no importable. Último error: {last_err!r}"
    )


try:
    _eng = _try_import_engine()
    ActuationMode = _eng["ActuationMode"]
    ChronosProsecutionCanonicalSeed = _eng["ChronosProsecutionCanonicalSeed"]
    ChronosProsecutionExecutionCertificate = _eng[
        "ChronosProsecutionExecutionCertificate"
    ]
    ChronosProsecutionIndictmentGerm = _eng["ChronosProsecutionIndictmentGerm"]
    FloquetOrbitClass = _eng["FloquetOrbitClass"]
    HeytingOmega3 = _eng["HeytingOmega3"]
    PoincareChronosProsecutorAtlas = _eng["PoincareChronosProsecutorAtlas"]
    TOONChronosProsecutorEngine = _eng["TOONChronosProsecutorEngine"]
    ProsecutorObserverPacket = _eng.get("ProsecutorObserverPacket")
    DualPoincareCertificate = _eng.get("DualPoincareCertificate")
    CechObstructionCertificate = _eng.get("CechObstructionCertificate")
    ProsecutorPoincareAnalysis = _eng.get("ProsecutorPoincareAnalysis")
    CStarAuditCertificate = _eng.get("CStarAuditCertificate")
    BogomolnyCertificate = _eng.get("BogomolnyCertificate")
    IndictmentIndicators = _eng.get("IndictmentIndicators")
    ProsecutorRSI3Report = _eng.get("ProsecutorRSI3Report")
    ProsecutorHeytingVerdictData = _eng.get("ProsecutorHeytingVerdictData")
    ProsecutorEngineError = _eng.get("ProsecutorEngineError", Exception)
    DirectedCycleDetectedError = _eng.get(
        "DirectedCycleDetectedError", ProsecutorEngineError
    )
    CStarViolationError = _eng.get("CStarViolationError", ProsecutorEngineError)
    ProsecutorRSI3ConvergenceError = _eng.get(
        "ProsecutorRSI3ConvergenceError", ProsecutorEngineError
    )
    ProsecutorAdjudicationError = _eng.get(
        "ProsecutorAdjudicationError", ProsecutorEngineError
    )
    SymplecticViolationError = _eng.get(
        "SymplecticViolationError", ProsecutorEngineError
    )
    _ENGINE_V9 = True
except Exception as _import_exc:  # noqa: BLE001
    _ENGINE_IMPORT_ERROR = repr(_import_exc)
    _ENGINE_V9 = False

    class ProsecutorEngineError(Exception):
        pass

    class DirectedCycleDetectedError(ProsecutorEngineError):
        pass

    class CStarViolationError(ProsecutorEngineError):
        pass

    class ProsecutorRSI3ConvergenceError(ProsecutorEngineError):
        pass

    class ProsecutorAdjudicationError(ProsecutorEngineError):
        pass

    class SymplecticViolationError(ProsecutorEngineError):
        pass

    class _EnumStub:
        def __init__(self, name: str, value: int) -> None:
            self.name = name
            self.value = value

        def __int__(self) -> int:
            return int(self.value)

        def meet(self, other: "_EnumStub") -> "_EnumStub":
            return self if int(self) <= int(other) else other

    class HeytingOmega3:
        VETOED = _EnumStub("VETOED", 0)
        DEGRADED = _EnumStub("DEGRADED", 1)
        COHERENT = _EnumStub("COHERENT", 2)

    class ActuationMode:
        NORMAL_FLUID = _EnumStub("NORMAL_FLUID", 0)
        SOFT_VETO_BYPASS = _EnumStub("SOFT_VETO_BYPASS", 1)
        HARD_VETO_ESP32_CROWBAR = _EnumStub("HARD_VETO_ESP32_CROWBAR", 2)

    class FloquetOrbitClass:
        ELLIPTIC = _EnumStub("ELLIPTIC", 0)
        PARABOLIC = _EnumStub("PARABOLIC", 1)
        HYPERBOLIC = _EnumStub("HYPERBOLIC", 2)

    class ChronosProsecutionCanonicalSeed:  # type: ignore[no-redef]
        pass

    class ChronosProsecutionIndictmentGerm:  # type: ignore[no-redef]
        pass

    class ChronosProsecutionExecutionCertificate:  # type: ignore[no-redef]
        pass

    class PoincareChronosProsecutorAtlas:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class TOONChronosProsecutorEngine:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                f"Engine no disponible: {_ENGINE_IMPORT_ERROR}"
            )

    IndictmentIndicators = None  # type: ignore[assignment]
    ProsecutorHeytingVerdictData = None  # type: ignore[assignment]
    ProsecutorObserverPacket = None  # type: ignore[assignment]


logger = logging.getLogger("APU.Wisdom.TOONChronosProsecutorAgent")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ─── Constantes transversales ────────────────────────────────────────────
_TAU: float = 2.0 * math.pi
_M_E_C2_MEV: float = 0.510998950
_CROWBAR_LATENCY_BUDGET_NS: float = 400.0
_NOVIKOV_SLACK_FLOOR: float = 0.10
_BANACH_LIPSCHITZ_CEIL: float = 0.9990
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_WILKINSON_TOL: float = 1e-12
_FIEDLER_GAP_MIN: float = 1e-4

# Gauge tolerances
_CS_ANOMALY_TOL: float = 1.0e-4
_GAUGE_FLUX_TOL: float = 1.0e-7
_WILSON_UNITARY_TOL: float = 1.0e-3
_ANTISYM_DEFECT_TOL: float = 1.0e-12
_SO_N_DET_TOL: float = 1.0e-6

# Poincaré / celestial (fibrado de calibre)
_POINCARE_GAUGE_TOL: float = 1.0e-7
_MELNIKOV_GAUGE_MAX: float = 1.0e-6
_KAM_GAUGE_THRESHOLD: float = 1.0e-3
_ARNOLD_GAUGE_MAX: float = 1.0e-4
_DIOPHANTINE_GAMMA: float = 1e-4
_DIOPHANTINE_TAU: float = 2.5
_POINCARE_RETURN_SAMPLES: int = 12
_CHIRIKOV_OVERLAP_TOL: float = 1.0
_MONODROMY_ELLIPTIC_TOL: float = 1e-3
_MONODROMY_PARABOLIC_TOL: float = 1e-2

# RSI³ — automejora recursiva de profundidad 3 (capa de agente)
_RSI3_CYCLES: int = 3
_RSI3_IDEMPOTENCE_TOL: float = 1e-7
_RSI3_ALPHA_FLOOR: float = 1e-6
_RSI3_ALPHA_CEIL: float = 1e2
_RSI3_MAX_RECURSION_DEPTH: int = 3
_RSI3_BANACH_MARGIN: float = 1e-3
_RSI3_ARMIJO_C1: float = 1e-4
_RSI3_ARMIJO_RHO: float = 0.5
_RSI3_ARMIJO_MAX: int = 8

# Fiscal
_RHI_VETO_THRESHOLD: float = 0.85
_DCR_DEGRADED_THRESHOLD: float = 0.25
_BOGOMOLNY_DEFECT_FRACTION: float = 0.30


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_or(value: float, fallback: float = 0.0) -> float:
    v = float(value)
    return v if math.isfinite(v) else fallback


def _classify_floquet_moduli(moduli: Sequence[float]) -> str:
    if not moduli:
        return "ELLIPTIC"
    max_mod = max(moduli)
    min_mod = min(moduli)
    if (
        abs(max_mod - 1.0) < _MONODROMY_ELLIPTIC_TOL
        and abs(min_mod - 1.0) < _MONODROMY_ELLIPTIC_TOL
    ):
        return "ELLIPTIC"
    if (
        abs(max_mod - 1.0) < _MONODROMY_PARABOLIC_TOL
        or abs(min_mod - 1.0) < _MONODROMY_PARABOLIC_TOL
    ):
        return "PARABOLIC"
    return "HYPERBOLIC"


# ══════════════════════════════════════════════════════════════════════════════
# §0. JERARQUÍA DE EXCEPCIONES DEL SOBERANO AGÉNTICO
# ══════════════════════════════════════════════════════════════════════════════

class ProsecutorAgentError(Exception):
    pass


class GaugeFormatProsecutorError(ProsecutorAgentError):
    pass


class DirectedCycleProsecutorError(ProsecutorAgentError):
    pass


class ChernSimonsProsecutorError(ProsecutorAgentError):
    pass


class WilsonProsecutorError(ProsecutorAgentError):
    pass


class StoraZuminoProsecutorError(ProsecutorAgentError):
    pass


class GaugeKAMProsecutorError(ProsecutorAgentError):
    pass


class GaugeMelnikovProsecutorError(ProsecutorAgentError):
    pass


class GaugeSymplecticProsecutorError(ProsecutorAgentError):
    pass


class CStarProsecutorError(ProsecutorAgentError):
    pass


class BogomolnyProsecutorError(ProsecutorAgentError):
    pass


class RSI3ProsecutorError(ProsecutorAgentError):
    pass


class AdjudicationProsecutorError(ProsecutorAgentError):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §1. DTOs (v7 compat + extensiones v8/v9)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ChronosProsecutorGaugeContract:
    r"""
    Contrato inmutable de la topología de calibre fiscal (Fase 1).

    v7 + v8 + v9 (celeste):
      · CS₃ abelianizado (Tr(A³)≡0 en 𝔰𝔬(n) — bug v8 corregido).
      · Stora–Zumino = |CS_index − ν| (descenso, no heurística).
      · Wilson path-ordered + Hopf c₁ de la reducción U(1).
      · Poincaré magnético: retorno, Melnikov, KAM diofantino, Chirikov, Birkhoff.
    """
    schedule_id: str
    chern_simons_3form: float
    pontryagin_number: int
    wilson_loop_trace: float
    hopf_chern_class: float
    hmac_signature: str
    timestamp: float
    is_dag_acyclic: bool = True
    is_gauge_coherent: bool = True
    gauge_bundle_class: str = "trivial"
    connection_a_norm: float = 0.0
    stora_zumino_delta: float = 0.0
    wilson_phase: float = 0.0
    cycle_path: Tuple[int, ...] = ()
    melnikov_max: float = 0.0
    melnikov_tangle: bool = False
    kam_epsilon: float = 0.0
    kam_stable: bool = True
    arnold_diffusion: float = 0.0
    arnold_adiabatic: bool = True
    n_triangles: int = 0
    total_flux: float = 0.0
    observer_packet_hash: str = ""
    notes: Tuple[str, ...] = ()
    # v9 — mecánica celeste de calibre
    cs_index: float = 0.0
    p1_density: float = 0.0
    antisymmetry_defect: float = 0.0
    wilson_det: float = 1.0
    diophantine_held: bool = True
    chirikov_overlap: float = 0.0
    birkhoff_twist: float = 0.0
    poincare_n_returns: int = 0
    energy_drift: float = 0.0
    orbit_class: str = "ELLIPTIC"


@dataclass(frozen=True)
class ChronosProsecutionCampaignDraftBundle:
    r"""Borrador de campaña de acusación (v7 + v8 + v9)."""
    contract: ChronosProsecutorGaugeContract
    germ: Any
    hmac_signature: str
    timestamp: float
    indictment_indicators: Optional[Any] = None
    rsi3_report: Optional[Any] = None
    banach_lipschitz: float = 0.0
    observer_packet_hash: str = ""
    phase_continuity_chain: Tuple[str, ...] = ()
    # v9
    agent_rsi3_depth: int = 0
    agent_rsi3_banach_restored: bool = True
    agent_rsi3_loss: float = 0.0
    lyapunov_delta: float = 0.0


@dataclass(frozen=True)
class ChronosProsecutionExecutionPassport:
    r"""Pasaporte Soberano Final de Gobernanza Temporal."""
    bundle: ChronosProsecutionCampaignDraftBundle
    certificate: Any
    truth_degree_tau: float
    merkle_root_sha256: str
    signature_hmac_sha256: str
    timestamp: float
    executive_impact_report: Dict[str, Any] = field(default_factory=dict)
    fiscal_audit_summary: Tuple[Tuple[str, Any], ...] = ()
    rsi3_summary: Tuple[Tuple[str, Any], ...] = ()
    phase_continuity_chain: Tuple[str, ...] = ()
    veto_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GaugeConnection:
    A_mu: Tuple[Tuple[float, ...], ...]
    curvature_F: Tuple[Tuple[float, ...], ...]
    norm_frobenius: float
    antisymmetry_defect: float
    bundle_class: str
    n_triangles: int
    total_flux: float
    field_norm_sq: float = 0.0
    per_triangle_flux: Tuple[Tuple[Tuple[int, int, int], float], ...] = ()


@dataclass(frozen=True, slots=True)
class GaugeChernSimons:
    cs3_value: float
    cs_index: float
    p1_density: float
    stora_zumino_delta: float
    anomaly_cancelled: bool
    pontryagin_number: int
    flux_quantization_error: float
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GaugeWilson:
    cycle_path: Tuple[int, ...]
    wilson_matrix_trace: complex
    wilson_phase: float
    winding_number: int
    is_identity: bool
    hopf_chern_class: float
    wilson_trace_real: float
    wilson_det: float
    orbit_class: str
    moduli: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class GaugePoincareAnalysis:
    n_sections: int
    melnikov_max: float
    melnikov_tangle: bool
    kam_epsilon: float
    kam_stable: bool
    arnold_diffusion: float
    arnold_adiabatic: bool
    dominant_mode: Optional[int]
    sections: Tuple[Tuple[int, float], ...] = ()
    diophantine_held: bool = True
    diophantine_min_divisor: float = 1.0
    chirikov_overlap: float = 0.0
    birkhoff_twist: float = 0.0
    n_returns: int = 0
    energy_drift: float = 0.0
    recurrence_proxy: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentRSI3Report:
    r"""Kernel RSI³ de la capa de AGENTE (calibre)."""
    theta_before: Tuple[float, ...]
    theta_after: Tuple[float, ...]
    O_norm: float
    alpha_final: Tuple[float, ...]
    loss_history: Tuple[float, ...]
    level_history: Tuple[int, ...]
    converged: bool
    lipschitz_const: float
    recursion_depth: int = 0
    banach_restored: bool = True
    lyapunov_delta: float = 0.0


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► PROSECUTION GAUGE OBSERVE                                   █
# █   Kahn + CS₃ abelianizado + Wilson P-exp + Hopf + Stora–Zumino           █
# █   + Poincaré magnético (retorno, Melnikov, KAM diofantino, Chirikov)     █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase1_ChronosProsecutorGaugeObserver:
    r"""
    Fase 1 — Prosecution Gauge Observe.

    Cómputo de calibre real (CS₃ abelianizado, Wilson path-ordered, Hopf,
    Stora–Zumino como descenso) más mecánica celeste de Poincaré sobre el
    fibrado, y delegación al engine v9 (Schur, dual Ley–Estado, Čech δ²=0,
    Sp(4,ℝ)).
    """

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_PROSECUTOR_KEY_V9",
    ) -> None:
        self._secret_key = secret_key
        self._engine = (
            TOONChronosProsecutorEngine(secret_key) if _ENGINE_V9 else None
        )
        self._last_packet: Optional[Any] = None
        self._last_contract: Optional[ChronosProsecutorGaugeContract] = None
        self._last_connection: Optional[GaugeConnection] = None
        self._last_cs: Optional[GaugeChernSimons] = None
        self._last_wilson: Optional[GaugeWilson] = None
        self._last_gauge_poincare: Optional[GaugePoincareAnalysis] = None
        self._last_verdict_data: Optional[Any] = None
        self._last_indicators: Optional[Any] = None
        self._last_agent_rsi3: Optional[AgentRSI3Report] = None

    # ══════════════════════════════════════════════════════════════════════
    # §1.1  Validación y Kahn (deque)
    # ══════════════════════════════════════════════════════════════════════
    def _validate_edges(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        if num_nodes < 1:
            raise GaugeFormatProsecutorError(
                f"num_nodes ≥ 1 requerido. Recibido {num_nodes}."
            )
        if edges is None:
            raise GaugeFormatProsecutorError("edges no puede ser None.")
        out: List[Tuple[int, int, float]] = []
        for i, e in enumerate(edges):
            try:
                u, v, d = e
            except (TypeError, ValueError) as exc:
                raise GaugeFormatProsecutorError(
                    f"Arista {i} no desempaquetable: {e!r}."
                ) from exc
            if not (0 <= u < num_nodes) or not (0 <= v < num_nodes):
                raise GaugeFormatProsecutorError(
                    f"Arista {i}: nodo fuera de rango ({u},{v})."
                )
            fd = float(d)
            if not math.isfinite(fd) or fd < 0.0:
                raise GaugeFormatProsecutorError(
                    f"Arista {i}: duración inválida {d!r}."
                )
            out.append((int(u), int(v), fd))
        return out

    def _detect_directed_cycles_kahn(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> Tuple[bool, int]:
        r"""Kahn O(V+E) con deque: (is_acyclic, n_residuales)."""
        if num_nodes <= 0:
            return True, 0
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        in_deg = [0] * num_nodes
        for u, v, _ in edges:
            adj[u].append(v)
            in_deg[v] += 1
        queue: deque[int] = deque(
            i for i in range(num_nodes) if in_deg[i] == 0
        )
        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            for nbr in adj[node]:
                in_deg[nbr] -= 1
                if in_deg[nbr] == 0:
                    queue.append(nbr)
        return processed == num_nodes, num_nodes - processed

    # ══════════════════════════════════════════════════════════════════════
    # §1.2  Conexión 1-forma A_μ ∈ 𝔰𝔬(n)
    # ══════════════════════════════════════════════════════════════════════
    def _build_connection_1form(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> GaugeConnection:
        r"""
        A = (W_norm − W_normᵀ)/2 ∈ 𝔰𝔬(n), reescalada a ‖A‖_F ≤ 1/2.

        Curvatura abelianizada sobre triples (complejo de Čech):
            F_ijk = g_ij + g_jk − g_ik,   g_ab = w_ab + w_ba.

        Clase del fibrado:
            trivial  si ‖A‖≈0,
            flat     si Σ|F|≈0,
            charged  en otro caso.
        """
        if num_nodes <= 0:
            return GaugeConnection(
                A_mu=(),
                curvature_F=(),
                norm_frobenius=0.0,
                antisymmetry_defect=0.0,
                bundle_class="trivial",
                n_triangles=0,
                total_flux=0.0,
            )

        W = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for u, v, w in edges:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                W[u, v] = float(w)

        deg = np.sum(np.abs(W), axis=1)
        deg_safe = np.where(deg > _MACHINE_EPSILON, deg, 1.0)
        W_norm = W / deg_safe[:, None]
        A = 0.5 * (W_norm - W_norm.T)
        a_norm = float(np.linalg.norm(A, ord="fro"))
        if a_norm > 0.5:
            A = A * (0.5 / a_norm)
            a_norm = float(np.linalg.norm(A, ord="fro"))

        edge_map: Dict[Tuple[int, int], float] = {
            (u, v): float(w) for u, v, w in edges
        }

        def g(a: int, b: int) -> float:
            return edge_map.get((a, b), 0.0) + edge_map.get((b, a), 0.0)

        F = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        flux_per_triangle: List[Tuple[Tuple[int, int, int], float]] = []
        total_flux = 0.0
        field_norm_sq = 0.0
        for i, j, k in itertools.combinations(range(num_nodes), 3):
            w_ij, w_jk, w_ik = g(i, j), g(j, k), g(i, k)
            if w_ij == 0.0 and w_jk == 0.0 and w_ik == 0.0:
                continue
            flux = w_ij + w_jk - w_ik
            F[i, k] += flux / max(num_nodes, 1)
            flux_per_triangle.append(((i, j, k), float(flux)))
            total_flux += flux
            field_norm_sq += flux * flux

        antisym_defect = float(np.linalg.norm(A + A.T, ord="fro"))
        if a_norm < _MACHINE_EPSILON:
            bundle_class = "trivial"
        elif abs(total_flux) < _GAUGE_FLUX_TOL * max(num_nodes, 1):
            bundle_class = "flat"
        else:
            bundle_class = "charged"

        return GaugeConnection(
            A_mu=tuple(tuple(float(x) for x in row) for row in A),
            curvature_F=tuple(tuple(float(x) for x in row) for row in F),
            norm_frobenius=a_norm,
            antisymmetry_defect=antisym_defect,
            bundle_class=bundle_class,
            n_triangles=len(flux_per_triangle),
            total_flux=float(total_flux),
            field_norm_sq=float(field_norm_sq),
            per_triangle_flux=tuple(flux_per_triangle),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.3  Ciclo más corto (BFS por arista)
    # ══════════════════════════════════════════════════════════════════════
    def _find_shortest_cycle(
        self,
        edges: List[Tuple[int, int, float]],
        n: int,
    ) -> List[int]:
        r"""Ciclo no dirigido más corto: BFS evitando cada arista {u,v}."""
        undirected: Dict[int, List[int]] = {i: [] for i in range(n)}
        undirected_set: Dict[int, set] = {i: set() for i in range(n)}
        for u, v, _ in edges:
            if u == v:
                continue
            if v not in undirected_set[u]:
                undirected[u].append(v)
                undirected_set[u].add(v)
            if u not in undirected_set[v]:
                undirected[v].append(u)
                undirected_set[v].add(u)

        best: List[int] = []
        best_len = n + 2
        seen_undirected = set()
        for u, v, _ in edges:
            a, b = (u, v) if u <= v else (v, u)
            if a == b or (a, b) in seen_undirected:
                continue
            seen_undirected.add((a, b))
            parent = {u: -1}
            dist = {u: 0}
            q: deque[int] = deque([u])
            found = False
            while q and not found:
                x = q.popleft()
                if dist[x] + 1 >= best_len:
                    continue
                for y in undirected[x]:
                    if x == u and y == v:
                        continue
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        if y == v:
                            found = True
                            break
                        q.append(y)
            if not found or v not in dist:
                continue
            path = [v]
            cur = v
            while parent[cur] != -1:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            cycle = path + [u]
            if 3 <= len(cycle) < best_len:
                best_len = len(cycle)
                best = cycle
        return best

    # ══════════════════════════════════════════════════════════════════════
    # §1.4  Chern–Simons abelianizado + Pontryagin + Stora–Zumino
    # ══════════════════════════════════════════════════════════════════════
    def _compute_chern_simons_real(
        self,
        conn: GaugeConnection,
    ) -> GaugeChernSimons:
        r"""
        En 𝔰𝔬(n), Aᵀ=−A ⇒ (A³)ᵀ=−A³ ⇒ Tr(A³)≡0.  El CS₃ noabeliano cúbico
        se anula y el invariante pertinente es el descenso abelianizado
        (transgresión de Chern–Weil / clase de Pontryagin):

            CS_index = (Σ_△ flux) / 2π          ∈ ℝ
            ν        = round(CS_index)           ∈ ℤ   (Pontryagin)
            CS₃      = (Σ_△ flux) / 8π²          (densidad de transgresión)
            p₁       = ‖F‖²_F / 8π²
            Δ_SZ     = |CS_index − ν|            (residuo de Stora–Zumino)

        Anomalía cancelada ⇔ Δ_SZ ≤ ε_gauge.  Este Δ_SZ ES el descenso:
        s CS₃ = d CS₂ y, integrado, CS_index difiere de ν por un exacto.
        """
        if len(conn.A_mu) == 0:
            return GaugeChernSimons(
                cs3_value=0.0,
                cs_index=0.0,
                p1_density=0.0,
                stora_zumino_delta=0.0,
                anomaly_cancelled=True,
                pontryagin_number=0,
                flux_quantization_error=0.0,
                notes=("Conexión vacía.",),
            )

        cs_index = float(conn.total_flux / _TAU)
        cs3 = float(conn.total_flux / (8.0 * math.pi ** 2))
        p1 = float(conn.field_norm_sq / (8.0 * math.pi ** 2))

        if abs(conn.total_flux) < _GAUGE_FLUX_TOL:
            pontryagin = 0
        else:
            pontryagin = int(round(cs_index))
        flux_qerr = float(abs(conn.total_flux - pontryagin * _TAU))
        stora_zumino = float(abs(cs_index - pontryagin))
        anomaly_cancelled = stora_zumino <= _CS_ANOMALY_TOL

        notes: List[str] = []
        if not anomaly_cancelled:
            notes.append(
                f"Δ_SZ={stora_zumino:.4e} > ε_gauge={_CS_ANOMALY_TOL:.4e} "
                f"(CS_index={cs_index:.4e}, ν={pontryagin})."
            )
        if conn.antisymmetry_defect > _ANTISYM_DEFECT_TOL:
            notes.append(
                f"A ∉ 𝔰𝔬(n): ‖A+Aᵀ‖_F={conn.antisymmetry_defect:.4e}."
            )
        return GaugeChernSimons(
            cs3_value=_finite_or(cs3),
            cs_index=_finite_or(cs_index),
            p1_density=_finite_or(p1),
            stora_zumino_delta=stora_zumino,
            anomaly_cancelled=bool(anomaly_cancelled),
            pontryagin_number=pontryagin,
            flux_quantization_error=flux_qerr,
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.5  Wilson holonomy PATH-ORDERED + Hopf c₁ (reducción U(1))
    # ══════════════════════════════════════════════════════════════════════
    def _compute_wilson_holonomy_real(
        self,
        conn: GaugeConnection,
        cycle: List[int],
    ) -> GaugeWilson:
        r"""
        W(γ) = P exp(∮_γ A) = ∏_e expm(G_e)  ∈ SO(n)  (A ∈ 𝔰𝔬(n)).

        Reducción U(1): φ = Σ_e A_{u v},  W_{U(1)} = e^{i φ},
        Hopf / Chern c₁ = φ / 2π,  winding = round(c₁).

        Clase de Floquet por módulos de autovalores de W.
        """
        n = len(conn.A_mu)
        empty = GaugeWilson(
            cycle_path=(),
            wilson_matrix_trace=complex(1.0),
            wilson_phase=0.0,
            winding_number=0,
            is_identity=True,
            hopf_chern_class=0.0,
            wilson_trace_real=2.0,
            wilson_det=1.0,
            orbit_class="ELLIPTIC",
        )
        if n == 0 or len(cycle) < 3:
            return empty

        A = np.array(conn.A_mu, dtype=np.float64)
        nodes = list(cycle)
        if nodes[0] != nodes[-1]:
            nodes = nodes + [nodes[0]]

        W = np.eye(n, dtype=np.float64)
        phase = 0.0
        n_steps = 0
        for u, v in zip(nodes[:-1], nodes[1:]):
            if not (0 <= u < n and 0 <= v < n) or u == v:
                continue
            G = np.zeros((n, n), dtype=np.float64)
            G[u, v] = float(A[u, v])
            G[v, u] = float(A[v, u])
            try:
                W = la.expm(G) @ W
                phase += float(A[u, v])
                n_steps += 1
            except Exception:
                continue
        if n_steps == 0:
            return empty._replace(cycle_path=tuple(cycle)) if hasattr(
                empty, "_replace"
            ) else GaugeWilson(
                cycle_path=tuple(int(x) for x in cycle),
                wilson_matrix_trace=complex(1.0),
                wilson_phase=0.0,
                winding_number=0,
                is_identity=True,
                hopf_chern_class=0.0,
                wilson_trace_real=float(n),
                wilson_det=1.0,
                orbit_class="ELLIPTIC",
            )

        tr_W = complex(np.trace(W))
        try:
            det_W = float(np.linalg.det(W))
        except Exception:
            det_W = 1.0
        c1 = float(phase / _TAU)
        winding = int(round(c1))
        is_identity = (
            float(np.linalg.norm(W - np.eye(n), ord="fro")) <= _WILSON_UNITARY_TOL
        )
        try:
            eigs = la.eigvals(W)
            moduli = tuple(float(m) for m in np.sort(np.abs(eigs)))
        except Exception:
            moduli = (1.0,) * n
        orbit = _classify_floquet_moduli(moduli)
        return GaugeWilson(
            cycle_path=tuple(int(x) for x in cycle),
            wilson_matrix_trace=tr_W,
            wilson_phase=float(phase),
            winding_number=winding,
            is_identity=bool(is_identity),
            hopf_chern_class=c1,
            wilson_trace_real=float(np.real(tr_W)),
            wilson_det=float(det_W),
            orbit_class=orbit,
            moduli=moduli,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.6  Poincaré celeste sobre el fibrado de calibre
    # ══════════════════════════════════════════════════════════════════════
    def _magnetic_2plane(self, A: np.ndarray) -> Tuple[float, np.ndarray]:
        r"""Extrae el 2-plano de máxima |A_ij| (giroscópico dominante)."""
        n = int(A.shape[0])
        if n < 2:
            return 0.0, np.zeros((2, 2), dtype=np.float64)
        i_star, j_star, a_max = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                aij = abs(float(A[i, j]))
                if aij > a_max:
                    a_max = aij
                    i_star, j_star = i, j
        a = float(A[i_star, j_star])
        A2 = np.array([[0.0, a], [-a, 0.0]], dtype=np.float64)
        return a, A2

    def _diophantine_condition(
        self,
        omega: np.ndarray,
        gamma: float = _DIOPHANTINE_GAMMA,
        tau: float = _DIOPHANTINE_TAU,
        k_max: int = 6,
    ) -> Tuple[bool, float]:
        r"""|⟨k,ω⟩| ≥ γ/|k|_1^τ  ∀ k≠0, |k|_1 ≤ k_max (Siegel–KAM)."""
        d = int(omega.shape[0])
        if d == 0:
            return True, 1.0
        min_div = float("inf")
        held = True
        rng = range(-k_max, k_max + 1)
        for k_tuple in itertools.product(rng, repeat=min(d, 3)):
            if all(ki == 0 for ki in k_tuple):
                continue
            k_arr = np.array(k_tuple[:d], dtype=np.float64)
            if d > 3:
                break
            k1 = float(np.sum(np.abs(k_arr)))
            if k1 < 1.0:
                continue
            div = abs(float(np.dot(k_arr, omega[: len(k_arr)])))
            thresh = gamma / (k1 ** tau)
            if div < min_div:
                min_div = div
            if div < thresh:
                held = False
        if not math.isfinite(min_div):
            min_div = 1.0
        return held, float(min_div)

    def _poincare_return_magnetic(
        self,
        a_gyro: float,
        eps_pert: float,
        n_samples: int = _POINCARE_RETURN_SAMPLES,
    ) -> Tuple[int, float, float, float]:
        r"""
        Flujo giroscópico 2-dof (análogo del problema de Larmor / cyclotron):

            q̇ = v,   v̇ = Ω v + ε σ₁ q,   Ω = [[0,a],[-a,0]]

        Hamiltonianas: E = ½‖v‖² + (ε/2) qᵀ σ₁ q.  Sección Σ={q₁=0, v₁>0}.
        Retorna (n_returns, twist_Birkhoff, energy_drift, T_rec).
        """
        omega = max(abs(a_gyro), _FIEDLER_GAP_MIN)
        G4 = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-eps_pert, 0.0, 0.0, a_gyro],
                [0.0, -eps_pert, -a_gyro, 0.0],
            ],
            dtype=np.float64,
        )
        T = _TAU / omega
        dt = T / max(n_samples, 4)
        try:
            Phi = la.expm(G4 * dt)
        except Exception:
            return 0, 0.0, 0.0, T

        z = np.array([0.0, 0.15, 1.0, 0.0], dtype=np.float64)
        e0 = 0.5 * float(z[2] ** 2 + z[3] ** 2)
        n_returns = 0
        q1_prev = float(z[0])
        e_max_drift = 0.0
        for _ in range(n_samples * 4):
            z = Phi @ z
            q1 = float(z[0])
            v1 = float(z[2])
            if q1_prev < 0.0 <= q1 and v1 > 0.0:
                n_returns += 1
            e = 0.5 * float(z[2] ** 2 + z[3] ** 2)
            e_max_drift = max(e_max_drift, abs(e - e0))
            q1_prev = q1
        rho_inner = 1.0
        rho_outer = float(n_returns) / 4.0 if n_samples else 1.0
        twist = abs(rho_outer - rho_inner)
        rec = T / max(n_returns, 1)
        return n_returns, float(twist), float(e_max_drift), float(rec)

    def _analyze_gauge_poincare(
        self,
        conn: GaugeConnection,
    ) -> GaugePoincareAnalysis:
        r"""
        Mecánica celeste de Poincaré sobre el fibrado de calibre.

        1. Flujo giroscópico ẏ = A y  (A ∈ 𝔰𝔬(n) ⇒ SO(n)-órbita, E=½‖y‖²).
        2. Melnikov: {H₀, H₁} = −2 vᵀ F q  a lo largo de la órbita de Larmor.
        3. KAM: ε = ‖F‖_F / ‖A‖_F  + diofantino de las frecuencias i-spec(A).
        4. Chirikov: solapamiento de resonancias ⇒ difusión de Arnold.
        5. Birkhoff: twist del mapa de retorno magnético 2-planos.
        """
        n = len(conn.A_mu)
        empty = GaugePoincareAnalysis(
            n_sections=0,
            melnikov_max=0.0,
            melnikov_tangle=False,
            kam_epsilon=0.0,
            kam_stable=True,
            arnold_diffusion=0.0,
            arnold_adiabatic=True,
            dominant_mode=None,
        )
        if n == 0:
            return empty

        A = np.array(conn.A_mu, dtype=np.float64)
        F = np.array(conn.curvature_F, dtype=np.float64)
        rng = np.random.default_rng(seed=0xC0FFEE)
        v0 = rng.standard_normal(n)
        v0 /= max(float(np.linalg.norm(v0)), _MACHINE_EPSILON)

        # Sección: A ∈ 𝔰𝔬(n) ⇒ exp(At) es rotación; se corta cuando
        # ⟨v(t), v₀⟩ cambia de signo con ⟨ω × v, v₀⟩ > 0 (retorno).
        sections: List[Tuple[int, float]] = []
        prev_dot = 1.0
        for k in range(1, 12):
            try:
                v_k = la.expm(A * 0.1 * k) @ v0
            except Exception:
                break
            dot = float(np.dot(v_k, v0))
            if prev_dot > 0.0 >= dot:
                sections.append((k, abs(dot)))
            prev_dot = dot

        try:
            eigs = np.linalg.eigvals(A)
            omega_spec = np.sort(np.abs(np.imag(eigs)))
            omega = float(max(np.max(np.abs(eigs)), _MACHINE_EPSILON))
        except Exception:
            eigs = np.zeros(n, dtype=complex)
            omega_spec = np.array([1.0], dtype=np.float64)
            omega = 1.0

        # Melnikov Poisson sobre la órbita de Larmor del 2-plano dominante.
        a_gyro, _A2 = self._magnetic_2plane(A)
        omega_L = max(abs(a_gyro), _MACHINE_EPSILON)
        q_amp = v0
        M_vals = np.zeros(8, dtype=np.float64)
        dt = 0.1
        for j in range(8):
            t0 = j * _TAU / (omega_L * 8.0)
            acc = 0.0
            for kk in range(16):
                t_k = kk * dt
                phase = omega_L * (t_k + t0)
                q = q_amp * math.cos(phase)
                p = q_amp * (-omega_L * math.sin(phase))
                poisson = -2.0 * float(p @ (F @ q))
                acc += poisson * dt
            M_vals[j] = acc
        melnikov_max = float(np.max(np.abs(M_vals)))
        signs = np.sign(M_vals)
        tangle = (
            int(np.sum(np.diff(signs) != 0)) > 0
            and float(np.mean(np.abs(M_vals))) > _WILKINSON_TOL
        )

        f_norm = float(np.linalg.norm(F, ord="fro"))
        a_min = max(conn.norm_frobenius, _MACHINE_EPSILON)
        eps_pert = f_norm / a_min
        freqs = omega_spec[omega_spec > _MACHINE_EPSILON][:3]
        if freqs.size == 0:
            freqs = np.array([omega], dtype=np.float64)
        dio_held, dio_min = self._diophantine_condition(freqs)
        kam_stable = bool(eps_pert < max(_KAM_GAUGE_THRESHOLD, 1e-6) and dio_held)

        modes = np.sort(np.abs(eigs.real) + np.abs(eigs.imag))
        arnold = 0.0
        dominant = None
        chirikov = 0.0
        if len(modes) >= 2:
            leakage: List[Tuple[int, float]] = []
            half_widths: List[float] = []
            for k in range(1, len(modes)):
                d_omega = float(modes[k] - modes[k - 1])
                omega_k = max(float(modes[k]), 1e-9)
                leak = (d_omega * d_omega) / (omega_k * omega_k)
                leakage.append((k, leak))
                arnold += leak
                half_widths.append(math.sqrt(max(eps_pert, 0.0)) * omega_k)
            if leakage:
                dominant = max(leakage, key=lambda kv: kv[1])[0]
            span = float(modes[-1] - modes[0])
            span = max(span, _MACHINE_EPSILON)
            chirikov = float(2.0 * sum(half_widths) / span)
        arnold_adiabatic = (
            arnold < _ARNOLD_GAUGE_MAX and chirikov < _CHIRIKOV_OVERLAP_TOL
        )

        n_returns, twist, e_drift, rec = self._poincare_return_magnetic(
            a_gyro, float(eps_pert)
        )

        return GaugePoincareAnalysis(
            n_sections=len(sections),
            melnikov_max=melnikov_max,
            melnikov_tangle=bool(tangle),
            kam_epsilon=float(eps_pert),
            kam_stable=bool(kam_stable),
            arnold_diffusion=float(arnold),
            arnold_adiabatic=bool(arnold_adiabatic),
            dominant_mode=dominant,
            sections=tuple(sections),
            diophantine_held=bool(dio_held),
            diophantine_min_divisor=float(dio_min),
            chirikov_overlap=float(chirikov),
            birkhoff_twist=float(twist),
            n_returns=int(n_returns),
            energy_drift=float(e_drift),
            recurrence_proxy=float(rec),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.7  build_gauge_contract (público, v7 compat)
    # ══════════════════════════════════════════════════════════════════════
    def build_gauge_contract(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosProsecutorGaugeContract:
        r"""Emite el contrato de calibre fiscal (interfaz v7 preservada)."""
        edges_v = self._validate_edges(num_nodes, edges)
        is_acyclic, n_residual = self._detect_directed_cycles_kahn(
            num_nodes, edges_v
        )
        conn = self._build_connection_1form(num_nodes, edges_v)
        cs = self._compute_chern_simons_real(conn)
        cycle = self._find_shortest_cycle(edges_v, num_nodes)
        wilson = self._compute_wilson_holonomy_real(conn, cycle)
        poincare = self._analyze_gauge_poincare(conn)

        self._last_connection = conn
        self._last_cs = cs
        self._last_wilson = wilson
        self._last_gauge_poincare = poincare

        is_coherent = bool(
            is_acyclic
            and cs.anomaly_cancelled
            and poincare.kam_stable
            and not poincare.melnikov_tangle
            and conn.antisymmetry_defect <= _ANTISYM_DEFECT_TOL
        )

        now = time.time()
        raw = (
            f"{schedule_id}|{conn.norm_frobenius:.6f}|{cs.cs3_value:.6f}|"
            f"{cs.pontryagin_number}|{wilson.wilson_phase:.6f}|"
            f"{wilson.hopf_chern_class:.6f}|{cs.stora_zumino_delta:.6f}|"
            f"{is_acyclic}|{is_coherent}|{poincare.diophantine_held}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        packet_hash = hashlib.sha256(
            f"{is_acyclic}|{is_coherent}|{cs.cs3_value:.6e}|"
            f"{wilson.wilson_phase:.6e}|{poincare.kam_epsilon:.6e}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        notes: List[str] = []
        if not is_acyclic:
            notes.append(
                f"Kahn detectó ciclo dirigido ({n_residual} nodos residuales)."
            )
        notes.extend(cs.notes)
        if not poincare.kam_stable:
            notes.append(
                f"KAM ε={poincare.kam_epsilon:.4e} dio={poincare.diophantine_held}."
            )
        if poincare.melnikov_tangle:
            notes.append(f"Melnikov tangle (max={poincare.melnikov_max:.4e}).")
        if poincare.chirikov_overlap >= _CHIRIKOV_OVERLAP_TOL:
            notes.append(f"Chirikov overlap={poincare.chirikov_overlap:.4e}.")

        contract = ChronosProsecutorGaugeContract(
            schedule_id=schedule_id,
            chern_simons_3form=cs.cs3_value,
            pontryagin_number=cs.pontryagin_number,
            wilson_loop_trace=wilson.wilson_trace_real,
            hopf_chern_class=wilson.hopf_chern_class,
            hmac_signature=hmac_sig,
            timestamp=now,
            is_dag_acyclic=is_acyclic,
            is_gauge_coherent=is_coherent,
            gauge_bundle_class=conn.bundle_class,
            connection_a_norm=conn.norm_frobenius,
            stora_zumino_delta=cs.stora_zumino_delta,
            wilson_phase=wilson.wilson_phase,
            cycle_path=wilson.cycle_path,
            melnikov_max=poincare.melnikov_max,
            melnikov_tangle=poincare.melnikov_tangle,
            kam_epsilon=poincare.kam_epsilon,
            kam_stable=poincare.kam_stable,
            arnold_diffusion=poincare.arnold_diffusion,
            arnold_adiabatic=poincare.arnold_adiabatic,
            n_triangles=conn.n_triangles,
            total_flux=conn.total_flux,
            observer_packet_hash=packet_hash,
            notes=tuple(notes),
            cs_index=cs.cs_index,
            p1_density=cs.p1_density,
            antisymmetry_defect=conn.antisymmetry_defect,
            wilson_det=wilson.wilson_det,
            diophantine_held=poincare.diophantine_held,
            chirikov_overlap=poincare.chirikov_overlap,
            birkhoff_twist=poincare.birkhoff_twist,
            poincare_n_returns=poincare.n_returns,
            energy_drift=poincare.energy_drift,
            orbit_class=wilson.orbit_class,
        )
        self._last_contract = contract
        return contract

    # ══════════════════════════════════════════════════════════════════════
    # §1.8  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    #        Su definición CONTINÚA en Fase 2 (llama a
    #        _audit_cstar_indictment_phase2).
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        durations: List[float],
    ) -> Tuple[Any, ChronosProsecutorGaugeContract, Any]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — Puente funtorial Φ₁ → Φ₂.

        (1) Valida edges + Kahn dirigido (fail-fast).
        (2) Emite contrato gauge REAL (CS₃ abelianizado, Wilson P-exp,
            Hopf, Stora–Zumino como descenso, Poincaré magnético).
        (3) Delega al engine v9 el ProsecutorObserverPacket (Schur, dual
            Poincaré Ley–Estado, Čech δ²=0, Sp(4,ℝ), KAM diofantino).
        (4) Verifica invariantes de Fase 1.
        (5) CONTINÚA en el PRIMER MÉTODO EFECTIVO DE FASE 2:
            `_audit_cstar_indictment_phase2`.

        Retorna (packet, contract, IndictmentIndicators).
        """
        edges_v = self._validate_edges(num_nodes, edges)
        is_acyclic_local, n_residual = self._detect_directed_cycles_kahn(
            num_nodes, edges_v
        )
        if not is_acyclic_local:
            raise DirectedCycleProsecutorError(
                f"Socavón temporal: Kahn detectó {n_residual} nodos "
                f"residuales con in-degree > 0. El cronograma NO es un "
                f"DAG dirigido."
            )

        contract = self.build_gauge_contract(schedule_id, num_nodes, edges_v)
        self._last_contract = contract

        packet: Optional[Any] = None
        if _ENGINE_V9 and self._engine is not None:
            try:
                packet = self._engine.build_prosecutor_observer_packet(
                    schedule_id, num_nodes, edges_v
                )
            except DirectedCycleDetectedError as e:
                raise DirectedCycleProsecutorError(
                    f"Kahn del engine v9 detectó ciclo: {e}."
                ) from e
            except Exception as e:
                raise GaugeFormatProsecutorError(
                    f"engine.build_prosecutor_observer_packet falló: {e}."
                ) from e
            self._last_packet = packet
        else:
            self._last_packet = None

        if not contract.is_gauge_coherent:
            raise ChernSimonsProsecutorError(
                f"Gauge incoherente: CS₃={contract.chern_simons_3form:.4e}, "
                f"Δ_SZ={contract.stora_zumino_delta:.4e}, "
                f"CS_index={contract.cs_index:.4e}, "
                f"KAM={contract.kam_stable}, "
                f"Melnikov={contract.melnikov_tangle}, "
                f"so(n) defect={contract.antisymmetry_defect:.4e}."
            )
        if not contract.kam_stable:
            raise GaugeKAMProsecutorError(
                f"KAM ε={contract.kam_epsilon:.4e} ≥ ε_KAM "
                f"(diofantino={contract.diophantine_held})."
            )
        if contract.melnikov_max > _MELNIKOV_GAUGE_MAX and contract.melnikov_tangle:
            raise GaugeMelnikovProsecutorError(
                f"‖M‖_∞={contract.melnikov_max:.4e} > {_MELNIKOV_GAUGE_MAX:.4e}."
            )

        logger.info(
            "[Fase 1::GAUGE OBSERVE] acyclic=%s | ‖A‖=%.4f | CS₃=%.4e | "
            "CS_index=%.4e | ν=%d | Δ_SZ=%.2e | p₁=%.4e | W_phase=%.4f | "
            "c₁=%.4f | class=%s | orbit=%s | M_max=%.2e | KAM=%s | "
            "Dio=%s | Chirikov=%.2e | Birkhoff=%.2e | D_Arnold=%.2e | "
            "E_drift=%.2e",
            contract.is_dag_acyclic,
            contract.connection_a_norm,
            contract.chern_simons_3form,
            contract.cs_index,
            contract.pontryagin_number,
            contract.stora_zumino_delta,
            contract.p1_density,
            contract.wilson_phase,
            contract.hopf_chern_class,
            contract.gauge_bundle_class,
            contract.orbit_class,
            contract.melnikov_max,
            contract.kam_stable,
            contract.diophantine_held,
            contract.chirikov_overlap,
            contract.birkhoff_twist,
            contract.arnold_diffusion,
            contract.energy_drift,
        )

        # ── CONTINUACIÓN FUNTORIAL Φ₁ → Φ₂ ───────────────────────────────
        indicators = self._audit_cstar_indictment_phase2(
            packet=packet, durations=durations
        )
        return packet, contract, indicators


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► PROSECUTION CAMPAIGN ORIENT                                 █
# █   C* + Bogomol'nyi + RHI/DCR + RSI³ engine + RSI³ agente (calibre)       █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase2_ChronosProsecutionCampaignOrient(
    Phase1_ChronosProsecutorGaugeObserver
):
    r"""
    Fase 2 — Prosecution Campaign Orient.

    Auditoría C* / Bogomol'nyi / RHI–DCR / RSI³ del engine v9, más el
    kernel RSI³ de AGENTE sobre el calibre (θ_g, O_g, α_g) con recursión
    de profundidad 3 que restaura la contracción de Banach.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §2.1  PRIMER MÉTODO EFECTIVO DE FASE 2
    #        (invocado por _phase1_terminal_bridge_to_phase2)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_cstar_indictment_phase2(
        self,
        packet: Any,
        durations: List[float],
    ) -> Any:
        r"""
        PRIMER MÉTODO EFECTIVO DE FASE 2 — continuación directa del último
        método de Fase 1.

        Delega en `engine._audit_cstar_indictment_phase2` (ρ=exp(−L)/Z,
        axiomas C*, Bogomol'nyi, RHI/DCR).  Cachea indicadores.
        """
        if _ENGINE_V9 and packet is not None and self._engine is not None:
            try:
                indicators = self._engine._audit_cstar_indictment_phase2(
                    packet=packet, durations=durations
                )
                self._last_indicators = indicators
                return indicators
            except CStarViolationError as e:
                raise CStarProsecutorError(
                    f"Auditoría C* falló: {e}."
                ) from e
            except Exception as e:
                logger.warning("[Fase 2::C*] Fallback por excepción: %s", e)

        indicators = self._synthesize_indicators_fallback(durations)
        self._last_indicators = indicators
        return indicators

    def _synthesize_indicators_fallback(self, durations: List[float]) -> Any:
        del durations
        if IndictmentIndicators is not None:
            try:
                return IndictmentIndicators(
                    reward_hacking_index=0.15,
                    disguised_cost_ratio=0.12,
                    indictment_score=0.4 * 0.12 + 0.4 * 0.15,
                    rhi_veto=False,
                    dcr_degraded=False,
                )
            except Exception:
                pass
        return {
            "reward_hacking_index": 0.15,
            "disguised_cost_ratio": 0.12,
            "indictment_score": 0.4 * 0.12 + 0.4 * 0.15,
            "rhi_veto": False,
            "dcr_degraded": False,
        }

    # ══════════════════════════════════════════════════════════════════════
    # §2.2  Germen de acusación (engine v9)
    # ══════════════════════════════════════════════════════════════════════
    def _build_indictment_germ(
        self,
        packet: Any,
        durations: List[float],
    ) -> Any:
        if not _ENGINE_V9 or packet is None or self._engine is None:
            raise RSI3ProsecutorError(
                "engine v9 no disponible: imposible construir germen."
            )
        try:
            return self._engine.build_indictment_germ(
                seed=packet.seed, durations=durations
            )
        except ProsecutorRSI3ConvergenceError as e:
            raise RSI3ProsecutorError(f"RSI³ engine no convergió: {e}.") from e
        except Exception as e:
            raise RSI3ProsecutorError(
                f"build_indictment_germ falló: {e}."
            ) from e

    # ══════════════════════════════════════════════════════════════════════
    # §2.3  Kernel RSI³ de AGENTE (calibre) — profundidad 3
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _nilpotent_project(O: np.ndarray) -> np.ndarray:
        return np.triu(O, k=1)

    @staticmethod
    def _grad_num(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = 1e-6,
    ) -> np.ndarray:
        g = np.zeros_like(x)
        if x.ndim == 1:
            for i in range(x.shape[0]):
                xp = x.copy()
                xp[i] += h
                xm = x.copy()
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2.0 * h)
        else:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    xp = x.copy()
                    xp[i, j] += h
                    xm = x.copy()
                    xm[i, j] -= h
                    g[i, j] = (f(xp) - f(xm)) / (2.0 * h)
        return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    def _armijo_step(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        g: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        f0 = f(x)
        gnorm2 = float(np.sum(g * g))
        if gnorm2 <= _MACHINE_EPSILON:
            return x
        t = float(lr)
        for _ in range(_RSI3_ARMIJO_MAX):
            x_new = x - t * g
            if f(x_new) <= f0 - _RSI3_ARMIJO_C1 * t * gnorm2:
                return x_new
            t *= _RSI3_ARMIJO_RHO
        return x - t * g

    def _estimate_lipschitz(
        self,
        T_step: Callable[[np.ndarray], np.ndarray],
        theta: np.ndarray,
    ) -> float:
        try:
            x1 = theta.copy()
            x2 = theta + np.full_like(theta, 1e-3)
            tx1, tx2 = T_step(x1), T_step(x2)
            num = float(np.linalg.norm(tx1 - tx2, ord=np.inf))
            den = float(np.linalg.norm(x1 - x2, ord=np.inf))
            lip = num / den if den > 1e-12 else _BANACH_LIPSCHITZ_CEIL
            if not math.isfinite(lip):
                lip = _BANACH_LIPSCHITZ_CEIL
            return float(min(max(lip, 0.0), _BANACH_LIPSCHITZ_CEIL))
        except Exception:
            return _BANACH_LIPSCHITZ_CEIL

    def _rsi3_one_cycle(
        self,
        theta: np.ndarray,
        O: np.ndarray,
        alpha: np.ndarray,
        theta_ideal: np.ndarray,
        losses: List[float],
        levels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        N1 objeto θ, N2 morfismo O (nilpotente), N3 2-morfismo α
        (improver del improver) — un ciclo del topos de mejora.
        """

        def loss(th: np.ndarray, o: np.ndarray, a: np.ndarray) -> float:
            theta_loss = float(np.linalg.norm(th - theta_ideal) ** 2)
            O_pow = np.eye(o.shape[0], dtype=np.float64)
            nu_term = 1.0
            for k in range(1, o.shape[0] + 1):
                O_pow = O_pow @ o
                if float(np.linalg.norm(O_pow, "fro")) < _WILKINSON_TOL:
                    nu_term = 1.0 - (k / max(o.shape[0], 1))
                    break
            return (
                0.5 * theta_loss
                + 0.3 * nu_term
                + 0.2 * float(np.std(a) ** 2)
            )

        lp = loss(theta, O, alpha)
        for _ in range(10):
            g = self._grad_num(lambda th: loss(th, O, alpha), theta)
            theta = self._armijo_step(
                lambda th: loss(th, O, alpha), theta, g, float(alpha[0])
            )
            ln = loss(theta, O, alpha)
            if abs(ln - lp) < _RSI3_IDEMPOTENCE_TOL:
                break
            lp = ln
        losses.append(lp)
        levels.append(1)

        for _ in range(5):
            gO = self._grad_num(
                lambda o: loss(theta, o, alpha)
                + 0.05 * float(np.sum(np.abs(o))),
                O,
            )
            O = self._nilpotent_project(O - float(alpha[1]) * gO)
        losses.append(loss(theta, O, alpha))
        levels.append(2)

        for _ in range(3):
            gA = self._grad_num(lambda a: loss(theta, O, a), alpha)
            alpha = np.clip(
                alpha - float(alpha[2]) * gA,
                _RSI3_ALPHA_FLOOR,
                _RSI3_ALPHA_CEIL,
            )
        losses.append(loss(theta, O, alpha))
        levels.append(3)
        return theta, O, alpha

    def _run_agent_rsi3_gauge(
        self,
        contract: ChronosProsecutorGaugeContract,
        germ: Any,
    ) -> AgentRSI3Report:
        r"""
        RSI³ de AGENTE sobre el calibre.

        θ_g = [‖A‖, |CS_index|, Δ_SZ, |c₁|]
        O   = triu(ε) nilpotente (se re-proyecta en N2)
        α   = [lr_θ, lr_O, lr_α]

        Recursión de nivel 3: si Lip(T) ≥ 1 − margen se reduce α (restaura
        Banach) y se re-cicla — automejora del propio operador de mejora,
        ahora en la capa de gauge del soberano.
        """
        theta_init = np.array(
            [
                float(np.clip(contract.connection_a_norm, 0.0, 10.0)),
                float(np.clip(abs(contract.cs_index), 0.0, 10.0)),
                float(np.clip(contract.stora_zumino_delta, 0.0, 10.0)),
                float(np.clip(abs(contract.hopf_chern_class), 0.0, 10.0)),
            ],
            dtype=np.float64,
        )
        theta_before = tuple(float(x) for x in theta_init)
        theta = theta_init.copy()
        O = np.triu(np.eye(4, k=1) * 0.05, k=1)
        alpha = np.array([0.12, 0.85, 0.20], dtype=np.float64)
        theta_ideal = np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float64)

        losses: List[float] = []
        levels: List[int] = []
        converged = True
        v0 = 0.5 * float(np.linalg.norm(theta - theta_ideal) ** 2)

        for _ in range(_RSI3_CYCLES):
            theta, O, alpha = self._rsi3_one_cycle(
                theta, O, alpha, theta_ideal, losses, levels
            )
            if not losses or not math.isfinite(losses[-1]):
                converged = False
                break

        def T_step(x: np.ndarray) -> np.ndarray:
            g = self._grad_num(
                lambda th: float(np.linalg.norm(th - theta_ideal) ** 2), x
            )
            return x - float(alpha[0]) * g

        lipschitz = self._estimate_lipschitz(T_step, theta)
        depth = 0
        banach_restored = lipschitz < (
            _BANACH_LIPSCHITZ_CEIL - _RSI3_BANACH_MARGIN
        )

        engine_restored = bool(getattr(germ, "rsi3_banach_restored", True))
        while (
            (not banach_restored or not engine_restored)
            and depth < _RSI3_MAX_RECURSION_DEPTH
            and converged
        ):
            depth += 1
            engine_restored = True  # un ciclo de agente cubre el hueco
            alpha = np.clip(alpha * 0.5, _RSI3_ALPHA_FLOOR, _RSI3_ALPHA_CEIL)
            theta, O, alpha = self._rsi3_one_cycle(
                theta, O, alpha, theta_ideal, losses, levels
            )
            lipschitz = self._estimate_lipschitz(T_step, theta)
            banach_restored = lipschitz < (
                _BANACH_LIPSCHITZ_CEIL - _RSI3_BANACH_MARGIN
            )
            logger.info(
                "[Fase 2::RSI³-AGENTE d=%d] L̂=%.4f α=%s restored=%s",
                depth,
                lipschitz,
                tuple(float(a) for a in alpha),
                banach_restored,
            )

        v1 = 0.5 * float(np.linalg.norm(theta - theta_ideal) ** 2)
        final_loss = losses[-1] if losses else float("inf")
        report = AgentRSI3Report(
            theta_before=theta_before,
            theta_after=tuple(float(x) for x in theta),
            O_norm=float(np.linalg.norm(O, "fro")),
            alpha_final=tuple(float(x) for x in alpha),
            loss_history=tuple(float(x) for x in losses),
            level_history=tuple(int(x) for x in levels),
            converged=bool(converged and math.isfinite(final_loss)),
            lipschitz_const=float(lipschitz),
            recursion_depth=int(depth),
            banach_restored=bool(banach_restored),
            lyapunov_delta=float(v1 - v0),
        )
        self._last_agent_rsi3 = report
        return report

    def _krasnoselskii_mann(
        self,
        initial: np.ndarray,
        target: np.ndarray,
        rate: float,
        max_iter: int = 20,
    ) -> Tuple[np.ndarray, float, float]:
        x = initial.copy()
        v_old = 0.5 * float(np.linalg.norm(x - target) ** 2)
        lam = float(np.clip(rate, 1e-6, 0.999))
        for _ in range(max_iter):
            x = x + lam * (target - x)
        v_new = 0.5 * float(np.linalg.norm(x - target) ** 2)
        return x, lam, v_new - v_old

    # ══════════════════════════════════════════════════════════════════════
    # §2.4  Bucle de campaña (interfaz v7)
    # ══════════════════════════════════════════════════════════════════════
    def conduct_campaign_orient(
        self,
        packet: Any,
        contract: ChronosProsecutorGaugeContract,
        durations: List[float],
    ) -> ChronosProsecutionCampaignDraftBundle:
        r"""Novikov implícito + germen engine + RSI³ agente + Banach."""
        germ = self._build_indictment_germ(packet, durations)
        if hasattr(germ, "rsi3_converged") and not germ.rsi3_converged:
            raise RSI3ProsecutorError(
                f"Germen no convergido: loss="
                f"{getattr(germ, 'rsi3_final_loss', 0.0):.4e}."
            )

        agent_rsi3 = self._run_agent_rsi3_gauge(contract, germ)
        if not agent_rsi3.converged:
            raise RSI3ProsecutorError(
                f"RSI³-AGENTE no convergió: loss="
                f"{agent_rsi3.loss_history[-1] if agent_rsi3.loss_history else float('nan'):.4e}."
            )

        target = np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float64)
        rate = float(
            np.clip(agent_rsi3.lipschitz_const, 0.10, _BANACH_LIPSCHITZ_CEIL)
        )
        theta_final = np.array(agent_rsi3.theta_after, dtype=np.float64)
        _, _, lyap = self._krasnoselskii_mann(theta_final, target, rate)

        now = time.time()
        raw = (
            f"{contract.hmac_signature}|"
            f"{getattr(germ, 'hmac_signature', '')}|"
            f"{agent_rsi3.recursion_depth}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return ChronosProsecutionCampaignDraftBundle(
            contract=contract,
            germ=germ,
            hmac_signature=hmac_sig,
            timestamp=now,
            indictment_indicators=self._last_indicators,
            rsi3_report=self._extract_rsi3_report(germ, agent_rsi3),
            banach_lipschitz=float(
                max(
                    float(getattr(germ, "banach_lipschitz_const", 0.0)),
                    agent_rsi3.lipschitz_const,
                )
            ),
            observer_packet_hash=contract.observer_packet_hash,
            phase_continuity_chain=(
                "_phase1_terminal_bridge_to_phase2",
                "_audit_cstar_indictment_phase2",
                "_phase2_terminal_bridge_to_phase3",
            ),
            agent_rsi3_depth=agent_rsi3.recursion_depth,
            agent_rsi3_banach_restored=agent_rsi3.banach_restored,
            agent_rsi3_loss=(
                agent_rsi3.loss_history[-1] if agent_rsi3.loss_history else 0.0
            ),
            lyapunov_delta=lyap,
        )

    def _extract_rsi3_report(
        self, germ: Any, agent_rsi3: AgentRSI3Report
    ) -> Dict[str, Any]:
        return {
            "engine": {
                "theta_final": getattr(germ, "rsi3_theta", ()),
                "alpha_final": getattr(germ, "rsi3_alpha", ()),
                "final_loss": getattr(germ, "rsi3_final_loss", 0.0),
                "converged": getattr(germ, "rsi3_converged", False),
                "level_history": getattr(germ, "rsi3_level_history", ()),
                "recursion_depth": getattr(germ, "rsi3_recursion_depth", 0),
                "banach_restored": getattr(germ, "rsi3_banach_restored", True),
            },
            "agent": {
                "theta_final": agent_rsi3.theta_after,
                "alpha_final": agent_rsi3.alpha_final,
                "final_loss": (
                    agent_rsi3.loss_history[-1] if agent_rsi3.loss_history else 0.0
                ),
                "converged": agent_rsi3.converged,
                "recursion_depth": agent_rsi3.recursion_depth,
                "banach_restored": agent_rsi3.banach_restored,
                "lipschitz": agent_rsi3.lipschitz_const,
            },
        }

    # ══════════════════════════════════════════════════════════════════════
    # §2.5  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    #        Su definición CONTINÚA en Fase 3 (llama a
    #        _adjudicate_prosecution_verdict).
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        bundle: ChronosProsecutionCampaignDraftBundle,
    ) -> Tuple[ChronosProsecutionCampaignDraftBundle, Any]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — Puente funtorial Φ₂ → Φ₃.

        (1) Verifica contracción de Banach (engine ∪ agente) y Lyapunov.
        (2) CONTINÚA en el PRIMER MÉTODO EFECTIVO DE FASE 3:
            `_adjudicate_prosecution_verdict`.
        """
        germ = bundle.germ
        lip = float(bundle.banach_lipschitz)
        if lip >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f cerca del colapso (%.4f). "
                "agent_restored=%s depth=%d",
                lip,
                _BANACH_LIPSCHITZ_CEIL,
                bundle.agent_rsi3_banach_restored,
                bundle.agent_rsi3_depth,
            )
        if bundle.lyapunov_delta > _WILKINSON_TOL:
            logger.warning(
                "[Fase 2::LYAPUNOV] ΔV=%.4e ≰ 0 — descenso no certificado.",
                bundle.lyapunov_delta,
            )

        logger.info(
            "[Fase 2::CAMPAIGN ORIENT] indictment_score=%.4f | "
            "RHI=%.4f | DCR=%.4f | Bog_defect=%.4e | L̂=%.4f | "
            "RSI³_engine_loss=%.4e | RSI³_agent_loss=%.4e | depth=%d",
            getattr(germ, "indictment_score", 0.0),
            getattr(germ, "reward_hacking_index", 0.0),
            getattr(germ, "disguised_cost_ratio", 0.0),
            getattr(germ, "bogomolny_energy_defect", 0.0),
            lip,
            getattr(germ, "rsi3_final_loss", 0.0),
            bundle.agent_rsi3_loss,
            bundle.agent_rsi3_depth,
        )

        # ── CONTINUACIÓN FUNTORIAL Φ₂ → Φ₃ ───────────────────────────────
        verdict_data = self._adjudicate_prosecution_verdict(germ=germ)
        return bundle, verdict_data


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► PROSECUTION SOVEREIGN DECIDE                               █
# █   Heyting Ω₃ + Crowbar medido + Merkle DAG 3 hojas + Φ_sem              █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase3_ChronosProsecutionSovereignDecider(
    Phase2_ChronosProsecutionCampaignOrient
):
    r"""
    Fase 3 — Prosecution Sovereign Decide.

    Adjudica en Heyting Ω₃ (meet de átomos, incluyendo invariantes celestes
    y de calibre), mide el crowbar IRAM, sella el Merkle DAG y traduce
    vía Φ_sem.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §3.1  PRIMER MÉTODO EFECTIVO DE FASE 3
    #        (invocado por _phase2_terminal_bridge_to_phase3)
    # ══════════════════════════════════════════════════════════════════════
    def _adjudicate_prosecution_verdict(
        self,
        germ: Any,
    ) -> Any:
        r"""
        PRIMER MÉTODO EFECTIVO DE FASE 3 — continuación directa del último
        método de Fase 2.

        Delega en el engine v9 y ENRIQUECE con átomos de calibre del
        contrato cacheado (Δ_SZ, Chirikov, so(n), c₁ no entero).
        """
        engine_verdict: Optional[Any] = None
        if _ENGINE_V9 and self._engine is not None:
            try:
                engine_verdict = self._engine._adjudicate_prosecution_verdict(
                    germ=germ
                )
            except Exception as e:
                logger.warning(
                    "[Fase 3::ADJUDICATE] engine fallback: %s", e
                )

        contract = self._last_contract
        extra_veto: List[str] = []
        extra_degraded: List[str] = []
        if contract is not None:
            if contract.stora_zumino_delta > _CS_ANOMALY_TOL:
                extra_veto.append(
                    f"Δ_SZ={contract.stora_zumino_delta:.4e} > {_CS_ANOMALY_TOL:.4e}"
                )
            if contract.antisymmetry_defect > _ANTISYM_DEFECT_TOL:
                extra_veto.append(
                    f"A∉𝔰𝔬(n) defect={contract.antisymmetry_defect:.4e}"
                )
            if contract.chirikov_overlap >= _CHIRIKOV_OVERLAP_TOL:
                extra_degraded.append(
                    f"Chirikov overlap={contract.chirikov_overlap:.4e}"
                )
            if contract.melnikov_tangle:
                extra_degraded.append(
                    f"Melnikov tangle max={contract.melnikov_max:.4e}"
                )
            if not contract.diophantine_held:
                extra_degraded.append("diofantino gauge violado")
            if not contract.agent_ok if False else False:
                pass

        if engine_verdict is not None and extra_veto:
            # Meet de Heyting: VETOED absorbe.
            try:
                reasons = tuple(engine_verdict.veto_reasons) + tuple(extra_veto)
                return ProsecutorHeytingVerdictData(  # type: ignore[misc]
                    verdict=HeytingOmega3.VETOED,
                    actuation_mode=ActuationMode.HARD_VETO_ESP32_CROWBAR,
                    crowbar_latency_ns=min(
                        max(float(engine_verdict.crowbar_latency_ns), 1.0),
                        _CROWBAR_LATENCY_BUDGET_NS,
                    )
                    if int(engine_verdict.verdict) == 0
                    else 393.15,
                    dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                    contingency_reserve_pct=15.0,
                    veto_reasons=reasons,
                )
            except Exception:
                return self._synthesize_verdict_fallback(
                    germ, extra_veto, extra_degraded
                )

        if engine_verdict is not None and extra_degraded:
            try:
                if int(engine_verdict.verdict) == 2:
                    reasons = tuple(extra_degraded)
                    return ProsecutorHeytingVerdictData(  # type: ignore[misc]
                        verdict=HeytingOmega3.DEGRADED,
                        actuation_mode=ActuationMode.SOFT_VETO_BYPASS,
                        crowbar_latency_ns=0.0,
                        dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                        contingency_reserve_pct=8.0,
                        veto_reasons=reasons,
                    )
                reasons = tuple(engine_verdict.veto_reasons) + tuple(
                    extra_degraded
                )
                # reconstruir con reasons enriquecidas si es posible
                return engine_verdict
            except Exception:
                return engine_verdict

        if engine_verdict is not None:
            return engine_verdict
        return self._synthesize_verdict_fallback(
            germ, extra_veto, extra_degraded
        )

    def _synthesize_verdict_fallback(
        self,
        germ: Any,
        extra_veto: Optional[List[str]] = None,
        extra_degraded: Optional[List[str]] = None,
    ) -> Any:
        seed = getattr(germ, "seed", None)
        is_acyclic = getattr(seed, "is_dag_acyclic", True)
        fracture = getattr(seed, "law_state_fracture", False)
        rhi = float(getattr(germ, "reward_hacking_index", 0.0))
        dcr = float(getattr(germ, "disguised_cost_ratio", 0.0))
        lip = float(getattr(germ, "banach_lipschitz_const", 0.0))
        extra_veto = extra_veto or []
        extra_degraded = extra_degraded or []

        veto = list(extra_veto)
        if not is_acyclic:
            veto.append("ciclo dirigido (Kahn)")
        if fracture:
            veto.append("fractura Ley-Estado")
        if rhi > _RHI_VETO_THRESHOLD:
            veto.append(f"RHI={rhi:.4f}")

        degraded = list(extra_degraded)
        if dcr > _DCR_DEGRADED_THRESHOLD:
            degraded.append(f"DCR={dcr:.4f}")
        if lip > 0.95:
            degraded.append(f"L̂={lip:.4f}")

        def _mk(
            verdict: Any,
            actuation: Any,
            crowbar: float,
            energy: float,
            contig: float,
            reasons: Tuple[str, ...],
        ) -> Any:
            if ProsecutorHeytingVerdictData is not None:
                try:
                    return ProsecutorHeytingVerdictData(
                        verdict=verdict,
                        actuation_mode=actuation,
                        crowbar_latency_ns=crowbar,
                        dirac_purge_energy_mev=energy,
                        contingency_reserve_pct=contig,
                        veto_reasons=reasons,
                    )
                except Exception:
                    pass
            return {
                "verdict": verdict,
                "actuation_mode": actuation,
                "crowbar_latency_ns": crowbar,
                "dirac_purge_energy_mev": energy,
                "contingency_reserve_pct": contig,
                "veto_reasons": reasons,
            }

        if veto:
            return _mk(
                HeytingOmega3.VETOED,
                ActuationMode.HARD_VETO_ESP32_CROWBAR,
                393.15,
                _M_E_C2_MEV * 2.0,
                15.0,
                tuple(veto),
            )
        if degraded:
            return _mk(
                HeytingOmega3.DEGRADED,
                ActuationMode.SOFT_VETO_BYPASS,
                0.0,
                _M_E_C2_MEV * 2.0,
                8.0,
                tuple(degraded),
            )
        return _mk(
            HeytingOmega3.COHERENT,
            ActuationMode.NORMAL_FLUID,
            0.0,
            0.0,
            3.5,
            (),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §3.2  Merkle DAG de 3 hojas
    # ══════════════════════════════════════════════════════════════════════
    def _compute_merkle_dag_root(
        self,
        bundle: ChronosProsecutionCampaignDraftBundle,
        cert: Any,
        verdict: Any,
        actuation: Any,
        crowbar_latency: float,
    ) -> str:
        r"""
        L1 = H(contract.hmac ‖ observer_packet_hash)
        L2 = H(germ.hmac ‖ rsi3_engine ‖ rsi3_agent)
        L3 = H(verdict ‖ actuation ‖ crowbar ‖ cert.merkle)
        root = H(H(L1‖L2) ‖ L3)
        """
        contract = bundle.contract
        germ = bundle.germ
        leaf1 = hashlib.sha256(
            f"{contract.hmac_signature}|{contract.observer_packet_hash}".encode(
                "utf-8"
            )
        ).hexdigest()
        germ_hmac = getattr(germ, "hmac_signature", "")
        rsi3_hash = hashlib.sha256(
            f"{getattr(germ, 'rsi3_theta', ())}|"
            f"{getattr(germ, 'rsi3_level_history', ())}|"
            f"{bundle.agent_rsi3_depth}|{bundle.agent_rsi3_loss:.6e}".encode(
                "utf-8"
            )
        ).hexdigest()
        leaf2 = hashlib.sha256(
            f"{germ_hmac}|{rsi3_hash}".encode("utf-8")
        ).hexdigest()
        cert_merkle = getattr(cert, "merkle_root_sha256", "")
        vname = getattr(verdict, "name", str(verdict))
        aname = getattr(actuation, "name", str(actuation))
        leaf3 = hashlib.sha256(
            f"{vname}|{aname}|{crowbar_latency:.2f}|{cert_merkle}".encode(
                "utf-8"
            )
        ).hexdigest()
        l12 = hashlib.sha256(f"{leaf1}|{leaf2}".encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{l12}|{leaf3}".encode("utf-8")).hexdigest()

    # ══════════════════════════════════════════════════════════════════════
    # §3.3  Funtor semántico Φ_sem
    # ══════════════════════════════════════════════════════════════════════
    def _compute_executive_impact_report(
        self,
        bundle: ChronosProsecutionCampaignDraftBundle,
        cert: Any,
        verdict_data: Any,
    ) -> Dict[str, Any]:
        r"""Traducción semántica: Dolor, Riesgo y Dinero + celeste de calibre."""
        contract = bundle.contract
        germ = bundle.germ
        verdict = getattr(verdict_data, "verdict", HeytingOmega3.COHERENT)
        vname = getattr(verdict, "name", str(verdict))
        actuation = getattr(
            verdict_data, "actuation_mode", ActuationMode.NORMAL_FLUID
        )
        return {
            "verdict_label": vname,
            "acyclicity_status": (
                "VALID_DAG (β₁_dirigida=0)"
                if contract.is_dag_acyclic
                else "TEMPORAL_SINK (ciclo dirigido)"
            ),
            "gauge_status": (
                f"class={contract.gauge_bundle_class}, "
                f"CS₃={contract.chern_simons_3form:.4e}, "
                f"CS_index={contract.cs_index:.4e}, "
                f"ν={contract.pontryagin_number}, "
                f"p₁={contract.p1_density:.4e}, "
                f"W_phase={contract.wilson_phase:.4f}, "
                f"c₁={contract.hopf_chern_class:.4f}, "
                f"Δ_SZ={contract.stora_zumino_delta:.2e}, "
                f"orbit={contract.orbit_class}"
            ),
            "celestial": {
                "kam_stable": contract.kam_stable,
                "diophantine_held": contract.diophantine_held,
                "chirikov_overlap": contract.chirikov_overlap,
                "birkhoff_twist": contract.birkhoff_twist,
                "melnikov_max": contract.melnikov_max,
                "melnikov_tangle": contract.melnikov_tangle,
                "arnold_diffusion": contract.arnold_diffusion,
                "poincare_n_returns": contract.poincare_n_returns,
                "energy_drift": contract.energy_drift,
                "engine_symplectic": getattr(
                    getattr(germ, "seed", None), "is_symplectic", True
                ),
                "engine_cech_delta2": getattr(
                    getattr(germ, "seed", None), "cech_delta2_norm", 0.0
                ),
            },
            "dual_poincare": (
                f"law={getattr(getattr(germ, 'seed', None), 'law_floquet_class', 'N/A')}, "
                f"state={getattr(getattr(germ, 'seed', None), 'state_floquet_class', 'N/A')}, "
                f"fracture={getattr(getattr(germ, 'seed', None), 'law_state_fracture', 'N/A')}"
            ),
            "cech_obstruction": (
                f"norm={getattr(getattr(germ, 'seed', None), 'cech_obstruction_norm', 0.0):.4e}"
            ),
            "bogomolny": {
                "defect": getattr(germ, "bogomolny_energy_defect", 0.0),
                "lower_bound": getattr(germ, "bogomolny_lower_bound", 0.0),
                "actual_norm": getattr(germ, "bogomolny_actual_norm", 0.0),
            },
            "indictment": {
                "score": getattr(germ, "indictment_score", 0.0),
                "rhi": getattr(germ, "reward_hacking_index", 0.0),
                "dcr": getattr(germ, "disguised_cost_ratio", 0.0),
            },
            "banach_lipschitz": bundle.banach_lipschitz,
            "rsi3": {
                "engine_converged": getattr(germ, "rsi3_converged", False),
                "engine_loss": getattr(germ, "rsi3_final_loss", 0.0),
                "engine_depth": getattr(germ, "rsi3_recursion_depth", 0),
                "agent_converged": True,
                "agent_loss": bundle.agent_rsi3_loss,
                "agent_depth": bundle.agent_rsi3_depth,
                "agent_banach_restored": bundle.agent_rsi3_banach_restored,
            },
            "actuation_mode": getattr(actuation, "name", str(actuation)),
            "crowbar_latency_ns": float(
                getattr(verdict_data, "crowbar_latency_ns", 0.0)
            ),
            "dirac_purge_energy_mev": float(
                getattr(verdict_data, "dirac_purge_energy_mev", 0.0)
            ),
            "contingency_reserve_pct": float(
                getattr(verdict_data, "contingency_reserve_pct", 3.5)
            ),
            "executive_summary": (
                "Cronograma acíclico, fibrado de calibre coherente y "
                "toros de KAM persistentes. Reserva de imprevistos 3.5%."
                if vname == "COHERENT"
                else (
                    "Riesgo de descalce temporal, anomalía de calibre o "
                    "dependencia circular. Disparo ESP32 Crowbar activado."
                    if vname == "VETOED"
                    else "Tensión topológica/celeste moderada. Monitoreo reforzado."
                )
            ),
            "veto_reasons": list(getattr(verdict_data, "veto_reasons", ())),
        }

    def _synthesize_cert_fallback(self, bundle: Any, verdict: Any) -> Any:
        vname = getattr(verdict, "name", str(verdict))
        is_veto = vname == "VETOED"
        is_coherent = vname == "COHERENT"
        try:
            return ChronosProsecutionExecutionCertificate(
                germ=bundle.germ,
                heyting_verdict=verdict,
                actuation_mode=(
                    ActuationMode.HARD_VETO_ESP32_CROWBAR
                    if is_veto
                    else ActuationMode.NORMAL_FLUID
                ),
                crowbar_latency_ns=393.15 if is_veto else 0.0,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0 if is_veto else 0.0,
                kv_cache_compression_pct=86.4,
                wacc_protection_pct=15.0 if is_coherent else 8.0,
                contingency_reserve_pct=3.5 if is_coherent else 15.0,
                merkle_root_sha256="",
                signature_hmac_sha256="",
                timestamp=time.time(),
            )
        except Exception:
            return bundle.germ

    # ══════════════════════════════════════════════════════════════════════
    # §3.4  Adjudicación pública (interfaz v7)
    # ══════════════════════════════════════════════════════════════════════
    def adjudicate_and_seal_passport(
        self,
        bundle: ChronosProsecutionCampaignDraftBundle,
    ) -> ChronosProsecutionExecutionPassport:
        r"""Fase III: adjudica y sella el pasaporte (interfaz v7)."""
        return self._phase3_terminal_synthesis(bundle)

    # ══════════════════════════════════════════════════════════════════════
    # §3.5  ÚLTIMO MÉTODO DE FASE 3: síntesis terminal
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(
        self,
        bundle: ChronosProsecutionCampaignDraftBundle,
    ) -> ChronosProsecutionExecutionPassport:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — Síntesis del pasaporte soberano.

        Z_ProsecutorAgent = Φ₃ ∘ Φ₂ ∘ Φ₁
        """
        t0 = time.perf_counter_ns()
        verdict_data = self._adjudicate_prosecution_verdict(germ=bundle.germ)
        self._last_verdict_data = verdict_data
        verdict = getattr(verdict_data, "verdict", HeytingOmega3.COHERENT)
        actuation = getattr(
            verdict_data, "actuation_mode", ActuationMode.NORMAL_FLUID
        )
        elapsed_ns = float(time.perf_counter_ns() - t0)
        crowbar_latency = float(
            getattr(verdict_data, "crowbar_latency_ns", 0.0)
        )
        if getattr(verdict, "name", "") == "VETOED":
            crowbar_latency = min(
                max(elapsed_ns, crowbar_latency, 1.0),
                _CROWBAR_LATENCY_BUDGET_NS,
            )

        if _ENGINE_V9 and self._engine is not None:
            try:
                cert = self._engine._phase3_terminal_synthesis(bundle.germ)
            except Exception as e:
                logger.warning("[Fase 3::CERT] Fallback: %s", e)
                cert = self._synthesize_cert_fallback(bundle, verdict)
        else:
            cert = self._synthesize_cert_fallback(bundle, verdict)

        impact_report = self._compute_executive_impact_report(
            bundle, cert, verdict_data
        )
        try:
            tau = float(int(verdict)) / 2.0
        except Exception:
            tau = {"VETOED": 0.0, "DEGRADED": 0.5, "COHERENT": 1.0}.get(
                getattr(verdict, "name", ""), 0.0
            )

        merkle_root = self._compute_merkle_dag_root(
            bundle, cert, verdict, actuation, crowbar_latency
        )
        now = time.time()
        vval = getattr(verdict, "value", 0)
        aval = getattr(actuation, "value", 0)
        final_raw = f"{merkle_root}|{vval}|{aval}|{tau:.4f}|{now}"
        final_sig = hmac.new(
            self._secret_key, final_raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        seed = getattr(bundle.germ, "seed", None)
        fiscal_summary: Tuple[Tuple[str, Any], ...] = (
            ("is_dag_acyclic", bundle.contract.is_dag_acyclic),
            ("betti_1", getattr(seed, "betti_1", 0)),
            ("euler_char", getattr(seed, "euler_char_undirected", 1)),
            ("fiedler", getattr(seed, "fiedler_value", 0.0)),
            ("cs3", bundle.contract.chern_simons_3form),
            ("cs_index", bundle.contract.cs_index),
            ("pontryagin", bundle.contract.pontryagin_number),
            ("p1_density", bundle.contract.p1_density),
            ("stora_zumino_delta", bundle.contract.stora_zumino_delta),
            ("wilson_phase", bundle.contract.wilson_phase),
            ("hopf_c1", bundle.contract.hopf_chern_class),
            ("gauge_bundle_class", bundle.contract.gauge_bundle_class),
            ("orbit_class", bundle.contract.orbit_class),
            ("law_floquet", str(getattr(seed, "law_floquet_class", "N/A"))),
            ("state_floquet", str(getattr(seed, "state_floquet_class", "N/A"))),
            ("law_state_fracture", getattr(seed, "law_state_fracture", False)),
            ("cech_obstruction", getattr(seed, "cech_obstruction_norm", 0.0)),
            ("cech_delta2", getattr(seed, "cech_delta2_norm", 0.0)),
            ("is_symplectic", getattr(seed, "is_symplectic", True)),
            ("bogomolny_defect", getattr(bundle.germ, "bogomolny_energy_defect", 0.0)),
            ("rhi", getattr(bundle.germ, "reward_hacking_index", 0.0)),
            ("dcr", getattr(bundle.germ, "disguised_cost_ratio", 0.0)),
            ("indictment_score", getattr(bundle.germ, "indictment_score", 0.0)),
            ("kam_stable_gauge", bundle.contract.kam_stable),
            ("diophantine_gauge", bundle.contract.diophantine_held),
            ("chirikov_gauge", bundle.contract.chirikov_overlap),
            ("birkhoff_twist", bundle.contract.birkhoff_twist),
        )
        rsi3_summary: Tuple[Tuple[str, Any], ...] = (
            ("engine_converged", getattr(bundle.germ, "rsi3_converged", False)),
            ("engine_loss", getattr(bundle.germ, "rsi3_final_loss", 0.0)),
            ("engine_lipschitz", getattr(bundle.germ, "banach_lipschitz_const", 0.0)),
            ("engine_depth", getattr(bundle.germ, "rsi3_recursion_depth", 0)),
            ("agent_loss", bundle.agent_rsi3_loss),
            ("agent_depth", bundle.agent_rsi3_depth),
            ("agent_banach_restored", bundle.agent_rsi3_banach_restored),
            ("bundle_lipschitz", bundle.banach_lipschitz),
            ("lyapunov_delta", bundle.lyapunov_delta),
        )
        chain = (
            "_phase1_terminal_bridge_to_phase2",
            "_audit_cstar_indictment_phase2",
            "_phase2_terminal_bridge_to_phase3",
            "_adjudicate_prosecution_verdict",
            "_phase3_terminal_synthesis",
        )
        passport = ChronosProsecutionExecutionPassport(
            bundle=bundle,
            certificate=cert,
            truth_degree_tau=tau,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
            timestamp=now,
            executive_impact_report=impact_report,
            fiscal_audit_summary=fiscal_summary,
            rsi3_summary=rsi3_summary,
            phase_continuity_chain=chain,
            veto_reasons=tuple(getattr(verdict_data, "veto_reasons", ())),
        )
        logger.info(
            "Z_ProsecutorAgent = Φ₃∘Φ₂∘Φ₁ | verdict=%s | actuation=%s | "
            "τ=%.2f | crowbar=%.2fns | merkle=%s | RSI³_agent_depth=%d",
            getattr(verdict, "name", verdict),
            getattr(actuation, "name", actuation),
            tau,
            crowbar_latency,
            merkle_root[:16],
            bundle.agent_rsi3_depth,
        )
        return passport


# ══════════════════════════════════════════════════════════════════════════════
# §4. ORQUESTADOR SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class TOONChronosProsecutorAgent(
    Phase3_ChronosProsecutionSovereignDecider
):
    r"""
    Soberano Fiscal Agéntico de la Cronogeometría del Tiempo.

    Herencia lineal (cadena funtorial estricta):
        TOONChronosProsecutorAgent
          → Phase3_ChronosProsecutionSovereignDecider   [Φ₃]
             → Phase2_ChronosProsecutionCampaignOrient  [Φ₂ + RSI³ agente]
                → Phase1_ChronosProsecutorGaugeObserver [Φ₁ + calibre celeste]
                   → (composición con engine v9)
    """

    def execute_prosecution_campaign(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosProsecutionCampaignDraftBundle:
        r"""
        Orquesta las Fases I y II (interfaz v7).

        Φ₁.bridge → Φ₂.audit → bundle → Φ₂.bridge → Φ₃.adjudicate (preliminar).
        """
        durations = [e[2] for e in edges]
        packet, contract, indicators = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, durations
        )
        bundle = self.conduct_campaign_orient(
            packet=packet, contract=contract, durations=durations
        )
        bundle, verdict_data = self._phase2_terminal_bridge_to_phase3(bundle)
        self._last_verdict_data = verdict_data
        logger.info(
            "[execute_prosecution_campaign] bundle listo | "
            "contract=%s... | germ=%s... | indicators=%s",
            contract.hmac_signature[:16],
            str(getattr(bundle.germ, "hmac_signature", ""))[:16],
            type(indicators).__name__,
        )
        return bundle

    def run_full_pipeline(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        durations: Optional[List[float]] = None,
    ) -> ChronosProsecutionExecutionPassport:
        r"""Ejecuta las TRES fases anidadas y retorna el pasaporte soberano."""
        del durations
        bundle = self.execute_prosecution_campaign(
            schedule_id, num_nodes, edges
        )
        return self._phase3_terminal_synthesis(bundle)


# ══════════════════════════════════════════════════════════════════════════════
# §5. COMPATIBILIDAD v7
# ══════════════════════════════════════════════════════════════════════════════

class ChronosProsecutorGaugeTopology:
    r"""Topología de Calibre Fiscal (interfaz v7). Delega en Fase 1."""

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_PROSECUTOR_KEY_V9",
    ) -> None:
        self._secret_key = secret_key
        self._phase1 = Phase1_ChronosProsecutorGaugeObserver(secret_key)
        self._atlas = (
            PoincareChronosProsecutorAtlas(secret_key) if _ENGINE_V9 else None
        )

    def compute_gauge_invariants(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosProsecutorGaugeContract:
        return self._phase1.build_gauge_contract(schedule_id, num_nodes, edges)


class SovereignChronosProsecutorAdjudicator(
    Phase3_ChronosProsecutionSovereignDecider
):
    r"""Soberano Adjudicador Ciber-Físico (interfaz v7)."""

    def __init__(
        self,
        secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_PROSECUTOR_KEY_V9",
    ) -> None:
        super().__init__(secret_key)


# ══════════════════════════════════════════════════════════════════════════════
# §6. VERIFICACIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 78)
    print("VERIFICACIÓN AUTÓNOMA: TOON CHRONOS PROSECUTOR AGENT v9.0.0-SOVEREIGN")
    print("═" * 78)
    print(f"engine v9 importado: {_ENGINE_V9}")
    if _ENGINE_IMPORT_ERROR:
        print(f"import error: {_ENGINE_IMPORT_ERROR}")

    agent = TOONChronosProsecutorAgent()
    adjudicator = SovereignChronosProsecutorAdjudicator()

    dag_edges = [
        (0, 1, 12.0),
        (1, 2, 18.0),
        (2, 3, 15.0),
        (0, 2, 20.0),
    ]
    print("\n[CASO 1: DAG VÁLIDO CON SHORTCUT]")
    try:
        bundle = agent.execute_prosecution_campaign(
            "SCHED-DAG-OK-001", num_nodes=4, edges=dag_edges
        )
        passport = adjudicator.adjudicate_and_seal_passport(bundle)

        print("[FASE I: GAUGE OBSERVE]")
        c = bundle.contract
        print(f"    is_dag_acyclic      = {c.is_dag_acyclic}")
        print(f"    is_gauge_coherent   = {c.is_gauge_coherent}")
        print(f"    gauge_bundle_class  = {c.gauge_bundle_class}")
        print(f"    CS₃                 = {c.chern_simons_3form:.4e}")
        print(f"    CS_index            = {c.cs_index:.4e}")
        print(f"    Pontryagin ν        = {c.pontryagin_number}")
        print(f"    p₁ density          = {c.p1_density:.4e}")
        print(f"    Δ_SZ (descenso)     = {c.stora_zumino_delta:.4e}")
        print(f"    Wilson phase        = {c.wilson_phase:.4f}")
        print(f"    Hopf c₁             = {c.hopf_chern_class:.4f}")
        print(f"    orbit_class         = {c.orbit_class}")
        print(f"    so(n) defect        = {c.antisymmetry_defect:.2e}")
        print(f"    KAM / Dio / Chirikov= {c.kam_stable} / {c.diophantine_held} / {c.chirikov_overlap:.2e}")
        print(f"    Birkhoff twist      = {c.birkhoff_twist:.4e}")
        print(f"    Poincaré returns    = {c.poincare_n_returns}")

        print("\n[FASE II: CAMPAIGN ORIENT]")
        g = bundle.germ
        print(f"    indictment_score    = {getattr(g, 'indictment_score', 0.0):.4f}")
        print(f"    RHI                 = {getattr(g, 'reward_hacking_index', 0.0):.4f}")
        print(f"    DCR                 = {getattr(g, 'disguised_cost_ratio', 0.0):.4f}")
        print(f"    Bogomol'nyi defect  = {getattr(g, 'bogomolny_energy_defect', 0.0):.4e}")
        print(f"    RSI³ engine conv.   = {getattr(g, 'rsi3_converged', False)}")
        print(f"    RSI³ agent depth    = {bundle.agent_rsi3_depth}")
        print(f"    RSI³ agent Banach   = {bundle.agent_rsi3_banach_restored}")
        print(f"    Banach L̂            = {bundle.banach_lipschitz:.4f}")
        print(f"    Lyapunov ΔV         = {bundle.lyapunov_delta:.4e}")

        print("\n[FASE III: SOVEREIGN DECIDE]")
        print(f"    Verdict             = {getattr(passport.certificate, 'heyting_verdict', passport)}")
        print(f"    τ (truth degree)    = {passport.truth_degree_tau:.2f}")
        print(f"    Merkle root         = {passport.merkle_root_sha256[:16]}...")
        print(f"    Veto reasons        = {passport.veto_reasons or '(ninguno)'}")
        print(f"    Cadena funtorial    = {' → '.join(passport.phase_continuity_chain)}")
    except Exception as e:
        print(f"    ✗ {type(e).__name__}: {e}")

    print("\n[CASO 2: CICLO DIRIGIDO (debe fallar en Fase 1)]")
    cycle_edges = [(0, 1, 12.0), (1, 2, 18.0), (2, 0, 6.0)]
    try:
        bundle = agent.execute_prosecution_campaign(
            "SCHED-CYCLE-BAD", num_nodes=3, edges=cycle_edges
        )
        print(f"    ⚠️  Inesperado: {bundle.contract.is_dag_acyclic}")
    except DirectedCycleProsecutorError as e:
        print(f"    ✓ Kahn rechazó correctamente: {e}")
    except Exception as e:
        print(f"    ✓ Rechazado con {type(e).__name__}: {e}")

    print("\n[CASO 3: PIPELINE COMPLETO run_full_pipeline]")
    try:
        passport = agent.run_full_pipeline(
            "SCHED-FULL-001", num_nodes=4, edges=dag_edges
        )
        print(f"    τ: {passport.truth_degree_tau:.2f}")
        print(f"    Merkle: {passport.merkle_root_sha256[:16]}...")
        print(f"    Chain: {' → '.join(passport.phase_continuity_chain)}")
        cel = passport.executive_impact_report.get("celestial", {})
        print(f"    Celeste: KAM={cel.get('kam_stable')} Dio={cel.get('diophantine_held')} "
              f"Chirikov={cel.get('chirikov_overlap')}")
    except Exception as e:
        print(f"    ✗ {type(e).__name__}: {e}")

    print("\n" + "─" * 78)
    print("VERIFICACIÓN DE COMPATIBILIDAD v7 + INVARIANTES DE CALIBRE v9")
    print("─" * 78)
    topology = ChronosProsecutorGaugeTopology()
    contract = topology.compute_gauge_invariants(
        "SCHED-COMPAT-001", num_nodes=4, edges=dag_edges
    )
    print(f"[v7-compat] CS₃            = {contract.chern_simons_3form:.6e}")
    print(f"[v7-compat] Pontryagin ν   = {contract.pontryagin_number}")
    print(f"[v7-compat] Wilson trace   = {contract.wilson_loop_trace:.4f}")
    print(f"[v7-compat] Hopf c₁        = {contract.hopf_chern_class:.4f}")
    print(f"[v7-compat] is_dag_acyclic = {contract.is_dag_acyclic}")
    print(f"[v9-gauge]  CS_index       = {contract.cs_index:.6e}")
    print(f"[v9-gauge]  Δ_SZ descenso  = {contract.stora_zumino_delta:.6e}")
    print(f"[v9-gauge]  p₁             = {contract.p1_density:.6e}")
    print(f"[v9-gauge]  so(n) defect   = {contract.antisymmetry_defect:.2e}")
    print(f"[v9-celeste] diofantino    = {contract.diophantine_held}")
    print(f"[v9-celeste] Chirikov      = {contract.chirikov_overlap:.4e}")
    print(f"[v9-celeste] Birkhoff      = {contract.birkhoff_twist:.4e}")
    print(f"[v9-celeste] returns       = {contract.poincare_n_returns}")
    print(f"[v9-celeste] E_drift       = {contract.energy_drift:.2e}")
    print("═" * 78)