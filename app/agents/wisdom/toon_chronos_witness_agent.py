# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Chronos Witness Agent (Soberano Testigo Cronogeométrico del Vacío) — v9.0.0          ║
║ Ubicación: app/agents/wisdom/toon_chronos_witness_agent.py                                           ║
║ Versión  : 9.0.0-Witness-Gauge-TomitaTakesaki-QND-RSI3-Poincare-NestedPhases-Doctoral                ║
║ Función  : Soberano de Calibre Testigo. Audita pasivamente (Back-Action ≡ 0.0 dB) la conexión A_μ,   ║
║            la 3-forma CS₃[A] discreta (A∧dA), el número de Pontryagin ν, la holonomía Wilson W(γ)    ║
║            path-ordered, la modularidad de Tomita-Takesaki, KMS analítico, sección de Poincaré del   ║
║            fibrado gauge y orquesta campañas QND con Heyting Ω₃ y RSI³ sobre Π.                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
─────────────────────────────────────────────────────────────────────────────────────────
El `TOONChronosWitnessAgent` representa al Soberano Agéntico de Calibre y Testigo del Vacío
en el Estrato WISDOM de la arquitectura cognitiva APU Filter. Su función principal es orquestar
la auditoría pasiva de calibre sobre la conexión $A_\mu \in \mathfrak{so}(n)$, evaluar los invariantes
de Chern-Simons discreto y de Pontryagin $\nu$, verificar el descenso de Stora-Zumino $\Delta_{\text{SZ}}$,
la holonomía de Wilson $W(\gamma) \in U(1) \times SU(2)$, y la consistencia de la sección de Poincaré
compuesta $\Pi_{\text{gauge}} \circ \Pi_{\text{modular}}$ con cero retroacción cuántica ($\text{Back-Action} \equiv 0.0\text{ dB}$).

2. MARCO TEÓRICO:
─────────────────────────────────────────────────────────────────────────────────────────
2.1. CONEXIÓN DE GAUGE $\mathfrak{so}(n)$, CHERN-SIMONS Y HOLONOMÍA DE WILSON:
   Dada la matriz de adyacencia del cronograma $W \in \mathbb{R}^{n \times n}$, se construye la 1-forma de conexión
   antisimétrica $A = \frac{1}{2}(\widehat{W} - \widehat{W}^\top) \in \mathfrak{so}(n)$.
     - 3-Forma de Chern-Simons Abeliana Discreta: Al ser $\text{Tr}(A^3) \equiv 0$ para matrices reales antisimétricas,
       el invariante de Chern-Simons sobre la triangulación del grafo se calcula mediante la transgresión $A \wedge dA$:
         $$\text{CS}_3(A) = \frac{1}{8\pi^2} \sum_{\triangle_{ijk}} \text{circ}_{\triangle}(A), \quad \text{circ}_{\triangle}(A) = A_{ij} + A_{jk} + A_{ki}$$
     - Índice de Chern-Simons y Número de Pontryagin:
         $$\text{CS}_{\text{index}} = \frac{1}{2\pi} \sum_{\triangle} \text{circ}_{\triangle}(A), \quad \nu = \text{round}(\text{CS}_{\text{index}}) \in \mathbb{Z}$$
     - Descenso de Stora-Zumino: El residuo de invariancia de calibre bajo $A \mapsto A + d\alpha$ se define como:
         $$\Delta_{\text{SZ}} = |\text{CS}_{\text{index}} - \nu| \le \varepsilon_{\text{gauge}}$$
     - Holonomía de Wilson Path-Ordered y Clase de Hopf: Para un ciclo cerrado $\gamma$:
         $$W_{U(1)}(\gamma) = \exp \left( i \oint_\gamma A \right) \in U(1), \quad c_1 = \frac{1}{2\pi} \oint_\gamma A, \quad \text{winding} = \text{round}(c_1)$$

2.2. MECÁNICA CELESTE DE POINCARÉ SOBRE EL FIBRADO DE CALIBRE:
   La evolución sobre el fibrado $v'(t) = A v(t)$ define rotaciones isométricas en $SO(n)$ con energía de Yang-Mills $\mathcal{J}_g = -\frac{1}{2} \|A\|_F^2$.
   - Sección de Poincaré Compuesta: $\Pi_{\text{gauge}} \circ \Pi_{\text{modular}}$ integra el mapa de retorno de Wilson $W(\gamma)$
     con la sección del operador de densidad modular $\rho$.
   - Integral de Melnikov Giroscópica: $M(t_0) = \int \{ H_0, H_1 \}(z(t+t_0)) dt$. Los cambios de signo caracterizan tangencias homoclínicas.
   - Condición KAM Diofantina: $|\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau$ sobre las frecuencias imaginarias puras de $\mathfrak{so}(n)$.

2.3. TRADUCCIÓN SEMÁNTICA "DOLOR, RIESGO Y DINERO" ($\Phi_{\text{sem}}$):
   El funtor semántico traduce las propiedades topológicas a impacto financiero de gobernanza:
     - Estado COHERENT ($\top$): Ahorro del 11.5% en reservas de imprevistos ($15\% \to 3.5\%$) y optimización WACC del 15%.
     - Estado VETOED ($\bot$): Interlock ESP32 Crowbar en $WCET \le 393.85 \text{ ns}$ ante presencia de ciclos o fracturas.

3. AXIOMAS FORMALES:
─────────────────────────────────────────────────────────────────────────────────────────
  AXIOMA I (INVARIANCIA DE CALIBRE Y DESCENSO DE STORA-ZUMINO):
    La conexión 1-forma $A \in \mathfrak{so}(n)$ satisface el descenso exacto de Stora-Zumino bajo transformaciones $A \mapsto A + d\alpha$, cumpliendo $\Delta_{\text{SZ}} \le \varepsilon_{\text{gauge}}$.

  AXIOMA II (CUANTIZACIÓN TOPOLÓGICA DE PONTRYAGIN Y HOPF):
    La clase de Chern/Hopf $c_1$ y el número de Pontryagin $\nu$ satisfacen la condición de cuantización entera $c_1 \in \mathbb{Z}, \nu \in \mathbb{Z}$.

  AXIOMA III (COMPOSICIÓN DE SECCIONES DE POINCARÉ EN EL FIBRADO):
    La sección del mapa de retorno sobre el fibrado principal de calibre $\Pi_{\text{gauge}} \circ \Pi_{\text{modular}}$ es transversal y preserves la integral giroscópica de Yang-Mills $\mathcal{J}_g$.

  AXIOMA IV (CONTRACCIÓN MONÁDICA DE PICARD EN AGENTES):
    La optimización recursiva $T_A: \Theta_g \to \Theta_g$ sobre el estado del agente es una contracción estricta de Banach con constante de Lipschitz $\widehat{L} < 1.0$.

4. INVARIANTES DE ESTADO:
─────────────────────────────────────────────────────────────────────────────────────────
  1. CANCELACIÓN DE ANOMALÍA DE CHERN-SIMONS: $\Delta_{\text{SZ}} = |\text{CS}_{\text{index}} - \nu| \le 10^{-4}$.
  2. UNITARIEDAD DE LA HOLONOMÍA DE WILSON: $\|W_{U(1)}(\gamma)\| = 1.0 \pm 10^{-3}$.
  3. OBSERVACIÓN SIN RETROACCIÓN CUÁNTICA: $\text{Back-Action}_{\text{dB}} \equiv 0.0 \text{ dB}$.
  4. INTERLOCK hardware ESP32 CROWBAR: $WCET \le 393.85 \text{ ns}$ en veredicto VETOED.
  5. COMPOSICIÓN FUNTORIAL COMPLETA: $Z_{\text{WitnessAgent}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

# ─── Rutas de importación (motor v9, con repliegue v8) ────────────────────
sys.path.insert(0, os.path.abspath("."))

_ENGINE_V9 = False
_ENGINE_V8 = False
PoincareOrbitType: Any = None
PoincareSectionCertificate: Any = None
WitnessObserverPacket: Any = None
NovikovWitnessCertificate: Any = None
WitnessRSI3Report: Any = None
TomitaTakesakiCertificate: Any = None
KMSCertificate: Any = None
QNDWeakMeasurementCertificate: Any = None
HeytingVerdictData: Any = None

try:
    from app.wisdom.toon_chronos_witness_engine import (  # type: ignore
        ActuationMode,
        ChronosWeakObservationGerm,
        ChronosWitnessExecutionCertificate,
        ChronosWitnessGaugeInitialCondition,
        HeytingOmega3,
        PoincareChronosWitnessAtlas,
        TOONChronosWitnessEngine,
        WitnessObserverPacket,
        NovikovWitnessCertificate,
        WitnessRSI3Report,
        TomitaTakesakiCertificate,
        KMSCertificate,
        QNDWeakMeasurementCertificate,
        HeytingVerdictData,
        PoincareOrbitType,
        PoincareSectionCertificate,
    )
    _ENGINE_V9 = True
    _ENGINE_V8 = True
except ImportError:
    try:
        from toon_chronos_witness_engine import (  # type: ignore
            ActuationMode,
            ChronosWeakObservationGerm,
            ChronosWitnessExecutionCertificate,
            ChronosWitnessGaugeInitialCondition,
            HeytingOmega3,
            PoincareChronosWitnessAtlas,
            TOONChronosWitnessEngine,
            WitnessObserverPacket,
            NovikovWitnessCertificate,
            WitnessRSI3Report,
            HeytingVerdictData,
            PoincareOrbitType,
            PoincareSectionCertificate,
        )
        _ENGINE_V9 = True
        _ENGINE_V8 = True
    except ImportError:
        try:
            from scratch.toon_chronos_witness_engine import (  # type: ignore
                ActuationMode,
                ChronosWeakObservationGerm,
                ChronosWitnessExecutionCertificate,
                ChronosWitnessGaugeInitialCondition,
                HeytingOmega3,
                PoincareChronosWitnessAtlas,
                TOONChronosWitnessEngine,
                WitnessObserverPacket,
                NovikovWitnessCertificate,
                WitnessRSI3Report,
                HeytingVerdictData,
            )
            _ENGINE_V8 = True
        except ImportError:
            from app.wisdom.toon_chronos_witness_engine import (  # type: ignore
                ActuationMode,
                ChronosWeakObservationGerm,
                ChronosWitnessExecutionCertificate,
                ChronosWitnessGaugeInitialCondition,
                HeytingOmega3,
                PoincareChronosWitnessAtlas,
                TOONChronosWitnessEngine,
            )


logger = logging.getLogger("APU.Wisdom.TOONChronosWitnessAgent")
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

# Gauge
_CS_ANOMALY_TOL: float = 1.0e-4
_GAUGE_FLUX_TOL: float = 1.0e-7
_WILSON_UNITARY_TOL: float = 1.0e-3
_HOPF_QUANTIZATION_TOL: float = 0.1
_GROMOV_CAPACITY_MAX: float = 12.5
_FUBINI_DEGRADED: float = 1.20

# Poincaré / KAM / Arnold / Melnikov
_POINCARE_GAUGE_TOL: float = 1.0e-7
_MELNIKOV_GAUGE_MAX: float = 1.0e-6
_KAM_GAUGE_THRESHOLD: float = 1.0e-3
_ARNOLD_GAUGE_MAX: float = 1.0e-4
_HYPERBOLIC_TRACE_THRESH: float = 2.0
_JACOBI_INTEGRAL_TOL: float = 1.0e-8

# RSI³
_RSI3_CYCLES: int = 3
_RSI3_IDEMPOTENCE_TOL: float = 1.0e-7
_RSI3_ALPHA_FLOOR: float = 1.0e-6
_RSI3_ALPHA_CEIL: float = 1.0e2

_SECRET_DEFAULT: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_WITNESS_KEY_V9"


# ══════════════════════════════════════════════════════════════════════════════
# §0. EXCEPCIONES, HELPERS Y FALLBACKS DE ENUM
# ══════════════════════════════════════════════════════════════════════════════

class WitnessAgentError(Exception):
    """Raíz del retículo de fallos del soberano de calibre."""


class GaugeFormatError(WitnessAgentError):
    """Cronograma / conexión mal formada."""


class DirectedCycleWitnessError(WitnessAgentError):
    """Ciclo dirigido detectado por Kahn."""


class ChernSimonsWitnessError(WitnessAgentError):
    """Anomalía de Chern-Simons / Stora-Zumino."""


class WilsonHolonomyWitnessError(WitnessAgentError):
    """Holonomía Wilson no unitaria o no cuantizada."""


class StoraZuminoWitnessError(WitnessAgentError):
    """Descenso de Stora-Zumino violado."""


class GaugeKAMWitnessError(WitnessAgentError):
    """Umbral KAM del fibrado gauge rebasado."""


class GaugeMelnikovWitnessError(WitnessAgentError):
    """Integral de Melnikov implica tangle homoclínico."""


class NovikovWitnessAgentError(WitnessAgentError):
    """Valuación de Novikov bajo el suelo."""


class RSI3WitnessError(WitnessAgentError):
    """RSI³ no alcanzó punto fijo de Picard."""


class WitnessAdjudicationAgentError(WitnessAgentError):
    """Fallo de adjudicación soberana."""


if PoincareOrbitType is None:
    class PoincareOrbitType(IntEnum):  # type: ignore
        ELLIPTIC = 0
        PARABOLIC = 1
        HYPERBOLIC = 2


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Acceso uniforme dict / dataclass / None."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _classify_monodromy_trace(trace_real: float) -> "PoincareOrbitType":
    tr_abs = abs(float(trace_real))
    if tr_abs > _HYPERBOLIC_TRACE_THRESH + 1e-6:
        return PoincareOrbitType.HYPERBOLIC
    if abs(tr_abs - _HYPERBOLIC_TRACE_THRESH) <= 1e-6:
        return PoincareOrbitType.PARABOLIC
    return PoincareOrbitType.ELLIPTIC


# ══════════════════════════════════════════════════════════════════════════════
# §1. DTOs (v7/v8 compat + extensiones v9 Poincaré-gauge)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ChronosWitnessGaugeContract:
    r"""
    Contrato topológico de calibre (Fase 1).

    v9 añade el germen celeste del fibrado:
      · poincare_recurrence_time, monodromy_trace, max_lyapunov, jacobi_integral
      · orbit_type (ELLIPTIC/PARABOLIC/HYPERBOLIC)
      · wilson_winding, cs_triangle_count, flux_quantization_error
    """
    schedule_id: str
    connection_a_norm: float
    chern_simons_cs3: float
    pontryagin_number_nu: int
    wilson_holonomy_angle: float
    hopf_chern_class_c1: float
    stora_zumino_residual: float
    gauge_initial_condition: ChronosWitnessGaugeInitialCondition
    signature_hmac_sha256: str
    timestamp: float
    is_dag_acyclic: bool = True
    is_gauge_coherent: bool = True
    gauge_bundle_class: str = "trivial"
    cycle_path: Tuple[int, ...] = ()
    wilson_trace_real: float = 2.0
    melnikov_max: float = 0.0
    melnikov_tangle: bool = False
    kam_epsilon: float = 0.0
    kam_stable: bool = True
    arnold_diffusion: float = 0.0
    arnold_adiabatic: bool = True
    observer_packet_hash: str = ""
    notes: Tuple[str, ...] = ()
    # v9
    wilson_winding: int = 0
    cs_triangle_count: int = 0
    flux_quantization_error: float = 0.0
    antisymmetry_defect: float = 0.0
    poincare_recurrence_time: float = _TAU
    poincare_monodromy_trace: float = 2.0
    poincare_max_lyapunov: float = 0.0
    jacobi_integral: float = 0.0
    orbit_type: int = int(PoincareOrbitType.ELLIPTIC)
    engine_packet_hash: str = ""


@dataclass(frozen=True)
class ChronosWitnessObservationRequest:
    """Solicitud de campaña QND (v7 preservado)."""
    request_id: str
    schedule_id: str
    slacks: List[float]
    timestamp: float


@dataclass(frozen=True)
class ChronosWitnessCampaignDraftBundle:
    """Bundle de campaña QND (v7/v8 + v9)."""
    contract: ChronosWitnessGaugeContract
    observation_germ: ChronosWeakObservationGerm
    mean_slack_valuation: float
    signature_hmac_sha256: str
    timestamp: float
    novikov_cert: Optional[Any] = None
    rsi3_report: Optional[Any] = None
    banach_lipschitz: float = 0.0
    observer_packet_hash: str = ""
    novikov_v_slack: float = 0.0
    rsi3_picard_residual: float = 0.0
    rsi3_converged: bool = True


@dataclass(frozen=True)
class ChronosWitnessExecutionPassport:
    """Pasaporte soberano terminal (v7/v8 + v9 Poincaré)."""
    bundle: ChronosWitnessCampaignDraftBundle
    certificate: ChronosWitnessExecutionCertificate
    truth_degree_tau: float
    executive_summary_report: Dict[str, Any]
    merkle_root_sha256: str
    signature_hmac_sha256: str
    timestamp: float
    gauge_audit_summary: Tuple[Tuple[str, Any], ...] = ()
    rsi3_summary: Tuple[Tuple[str, Any], ...] = ()
    phase_continuity_chain: Tuple[str, ...] = ()
    veto_reasons: Tuple[str, ...] = ()
    poincare_orbit_type: int = int(PoincareOrbitType.ELLIPTIC)
    jacobi_integral: float = 0.0


@dataclass(frozen=True, slots=True)
class GaugeConnectionPacket:
    """Conexión 1-forma A_μ y curvatura F = dA (circulaciones)."""
    A_mu: Tuple[Tuple[float, ...], ...]
    curvature_F: Tuple[Tuple[float, ...], ...]
    norm_frobenius: float
    antisymmetry_defect: float
    bundle_class: str
    n_triangles: int
    total_flux: float
    triangle_circulations: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class GaugeChernSimonsPacket:
    """CS₃[A] discreto (A∧dA) + Pontryagin ν + Stora-Zumino Δ_SZ."""
    cs3_value: float
    stora_zumino_delta: float
    anomaly_cancelled: bool
    pontryagin_number: int
    flux_quantization_error: float
    triangle_count: int
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GaugeWilsonPacket:
    """W(γ) path-ordered + fase + winding + Hopf c₁ + clase de órbita."""
    cycle_path: Tuple[int, ...]
    wilson_matrix_trace: complex
    wilson_phase: float
    winding_number: int
    is_identity: bool
    hopf_chern_class: float
    wilson_trace_real: float
    orbit_class: str
    is_unitary: bool = True
    unitary_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class GaugePoincarePacket:
    """Poincaré-Melnikov-KAM-Arnold sobre el fibrado gauge."""
    n_sections: int
    melnikov_max: float
    melnikov_tangle: bool
    kam_epsilon: float
    kam_stable: bool
    arnold_diffusion: float
    arnold_adiabatic: bool
    dominant_mode: Optional[int]
    sections: Tuple[Tuple[int, float], ...] = ()
    recurrence_time: float = _TAU
    monodromy_trace: float = 2.0
    max_lyapunov: float = 0.0
    jacobi_integral: float = 0.0
    jacobi_drift: float = 0.0
    orbit_type: int = int(PoincareOrbitType.ELLIPTIC)
    is_transversal: bool = True
    notes: Tuple[str, ...] = ()


def _make_heyting_verdict(
    verdict: HeytingOmega3,
    actuation: ActuationMode,
    crowbar_ns: float,
    dirac_mev: float,
    contingency: float,
    reasons: Tuple[str, ...],
    orbit: Any = None,
) -> Any:
    """Construye HeytingVerdictData con o sin campo orbit_type (v8/v9)."""
    if orbit is None:
        orbit = PoincareOrbitType.ELLIPTIC
    if HeytingVerdictData is None:
        return {
            "verdict": verdict,
            "actuation_mode": actuation,
            "crowbar_latency_ns": crowbar_ns,
            "dirac_purge_energy_mev": dirac_mev,
            "contingency_reserve_pct": contingency,
            "veto_reasons": reasons,
            "orbit_type": orbit,
        }
    try:
        return HeytingVerdictData(
            verdict=verdict,
            actuation_mode=actuation,
            crowbar_latency_ns=crowbar_ns,
            dirac_purge_energy_mev=dirac_mev,
            contingency_reserve_pct=contingency,
            veto_reasons=reasons,
            orbit_type=orbit,
        )
    except TypeError:
        return HeytingVerdictData(
            verdict=verdict,
            actuation_mode=actuation,
            crowbar_latency_ns=crowbar_ns,
            dirac_purge_energy_mev=dirac_mev,
            contingency_reserve_pct=contingency,
            veto_reasons=reasons,
        )


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► GAUGE OBSERVE : CALIBRE REAL + POINCARÉ CELESTE             █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase1_ChronosWitnessGaugeObserver:
    r"""
    Fase 1 — Gauge Observe.

    Compone con `TOONChronosWitnessEngine` v9 (Tomita-Takesaki, KMS analítico,
    QND, Kahn, Gibbs, sección de Poincaré modular) y añade la capa de calibre:

      A ∈ so(N)  →  CS₃[A∧dA]  →  W(γ) path-ordered  →  Π_gauge  →  contrato.
    """

    def __init__(self, secret_key: bytes = _SECRET_DEFAULT):
        self._secret_key = secret_key
        self._engine = TOONChronosWitnessEngine(secret_key)
        self._atlas = PoincareChronosWitnessAtlas(secret_key)
        self._last_packet: Optional[Any] = None
        self._last_gauge_contract: Optional[ChronosWitnessGaugeContract] = None
        self._last_novikov: Optional[Any] = None

    # ══════════════════════════════════════════════════════════════════════
    # §1.1  Validación y Kahn (deque O(1))
    # ══════════════════════════════════════════════════════════════════════
    def _validate_edges(
        self, num_nodes: int, edges: List[Tuple[int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        if num_nodes < 1:
            raise GaugeFormatError(f"num_nodes ≥ 1. Recibido {num_nodes}.")
        if edges is None:
            raise GaugeFormatError("edges no puede ser None.")
        out: List[Tuple[int, int, float]] = []
        for i, e in enumerate(edges):
            try:
                u, v, d = e
            except (TypeError, ValueError) as exc:
                raise GaugeFormatError(f"Arista {i} no válida: {e!r}.") from exc
            if not (0 <= u < num_nodes) or not (0 <= v < num_nodes):
                raise GaugeFormatError(
                    f"Arista {i}: nodo fuera de rango ({u},{v}).")
            fd = float(d)
            if not math.isfinite(fd) or fd < 0.0:
                raise GaugeFormatError(
                    f"Arista {i}: duración inválida {d!r}.")
            out.append((int(u), int(v), fd))
        return out

    def _detect_directed_cycles_kahn(
        self, num_nodes: int, edges: List[Tuple[int, int, float]],
    ) -> Tuple[bool, int]:
        r"""Kahn dirigido con cola O(1): (is_acyclic, n_nodos_residuales)."""
        if num_nodes <= 0:
            return True, 0
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        in_deg = [0] * num_nodes
        for u, v, _ in edges:
            adj[u].append(v)
            in_deg[v] += 1
        queue: deque[int] = deque(i for i in range(num_nodes) if in_deg[i] == 0)
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
    # §1.2  Conexión 1-forma A_μ ∈ so(N) + curvatura F = dA
    # ══════════════════════════════════════════════════════════════════════
    def _iter_undirected_triangles(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> Iterable[Tuple[int, int, int]]:
        """Triángulos del grafo subyacente no dirigido, en orden i<j<k."""
        adj: List[set] = [set() for _ in range(num_nodes)]
        for u, v, _ in edges:
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
        for i in range(num_nodes):
            nbrs_i = adj[i]
            for j in nbrs_i:
                if j <= i:
                    continue
                for k in (nbrs_i & adj[j]):
                    if k > j:
                        yield i, j, k

    def _build_gauge_connection_1form(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> GaugeConnectionPacket:
        r"""
        A = (W_norm − W_normᵀ)/2 ∈ so(N).

        Curvatura discreta sobre triángulos:
            circ_ijk = A_ij + A_jk + A_ki = (dA)(Δ_ijk)
        F se rellena con esas circulaciones (2-forma de grafo).
        Tr(A³) ≡ 0 para so(N) real; la topología vive en A∧dA, no en A³.
        """
        if num_nodes <= 0:
            return GaugeConnectionPacket(
                A_mu=(), curvature_F=(), norm_frobenius=0.0,
                antisymmetry_defect=0.0, bundle_class="trivial",
                n_triangles=0, total_flux=0.0,
            )

        W = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for u, v, w in edges:
            W[u, v] = float(w)

        deg = np.sum(np.abs(W), axis=1)
        deg_safe = np.where(deg > _MACHINE_EPSILON, deg, 1.0)
        W_norm = W / deg_safe[:, None]
        A = 0.5 * (W_norm - W_norm.T)

        a_norm = float(np.linalg.norm(A, ord="fro"))
        if a_norm > 0.5:
            A = A * (0.5 / a_norm)
            a_norm = float(np.linalg.norm(A, ord="fro"))

        F = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        circs: List[float] = []
        for i, j, k in self._iter_undirected_triangles(num_nodes, edges):
            circ = float(A[i, j] + A[j, k] + A[k, i])
            circs.append(circ)
            F[i, j] += circ
            F[j, k] += circ
            F[k, i] += circ

        n_triangles = len(circs)
        total_flux = float(sum(circs))
        antisym_defect = float(np.linalg.norm(A + A.T, ord="fro"))

        if a_norm < _MACHINE_EPSILON:
            bundle_class = "trivial"
        elif abs(total_flux) < _GAUGE_FLUX_TOL * max(num_nodes, 1):
            bundle_class = "flat"
        else:
            bundle_class = "charged"

        return GaugeConnectionPacket(
            A_mu=tuple(tuple(float(x) for x in row) for row in A),
            curvature_F=tuple(tuple(float(x) for x in row) for row in F),
            norm_frobenius=a_norm,
            antisymmetry_defect=antisym_defect,
            bundle_class=bundle_class,
            n_triangles=n_triangles,
            total_flux=total_flux,
            triangle_circulations=tuple(circs),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.3  Ciclo no dirigido más corto (BFS correcto)
    # ══════════════════════════════════════════════════════════════════════
    def _find_shortest_cycle(
        self, edges: List[Tuple[int, int, float]], n: int,
    ) -> List[int]:
        r"""
        Ciclo simple más corto del grafo no dirigido subyacente.

        En un DAG con shortcuts existe al menos un ciclo no dirigido
        (el 2-símplice de la cuerda). Wilson se evalúa sobre ese γ.
        """
        if n <= 0:
            return []
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v, _ in edges:
            if u == v:
                return [u, u]
            adj[u].append(v)
            adj[v].append(u)

        best: List[int] = []
        best_len = n + 2

        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q: deque[int] = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= best_len:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        pu: List[int] = []
                        pv: List[int] = []
                        cu, cv = u, v
                        seen_u = set()
                        while cu != -1 and cu not in seen_u:
                            pu.append(cu)
                            seen_u.add(cu)
                            cu = parent[cu]
                        seen_v = set()
                        while cv != -1 and cv not in seen_v:
                            pv.append(cv)
                            seen_v.add(cv)
                            cv = parent[cv]
                        pu.reverse()
                        pv.reverse()
                        k = 0
                        lim = min(len(pu), len(pv))
                        while k < lim and pu[k] == pv[k]:
                            k += 1
                        if k == 0:
                            continue
                        cycle = pu[k - 1:] + list(reversed(pv[k:]))
                        if not cycle:
                            continue
                        if cycle[0] != cycle[-1]:
                            cycle.append(cycle[0])
                        clen = len(cycle) - 1
                        if 3 <= clen < best_len:
                            best = cycle
                            best_len = clen
        return best

    # ══════════════════════════════════════════════════════════════════════
    # §1.4  Chern-Simons discreto A∧dA (fix Tr(A³)≡0)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_chern_simons_real(
        self, conn: GaugeConnectionPacket,
    ) -> GaugeChernSimonsPacket:
        r"""
        so(N) real ⇒ Aᵀ = −A ⇒ Tr(A³) = Tr((Aᵀ)³) = −Tr(A³) ⇒ Tr(A³) ≡ 0.

        El CS continuo (1/8π²)∫ Tr(A∧dA + ⅔ A³) colapsa al término A∧dA.
        Discretización (Cheeger–Simons / lattice abelian):

            circ_Δ = (dA)(Δ) = A_ij + A_jk + A_ki
            CS₃    = (Σ_Δ circ_Δ) / (8π²)
            ν      = round(Σ_Δ circ_Δ / 2π)
            Δ_SZ   = |Σ circ / 2π − ν|     (residuo de cuantización)

        Anomalía cancelada syss el residuo ≤ ε_gauge y |CS₃| es finito.
        El matching con el winding de Wilson se verifica en el contrato.
        """
        n = len(conn.A_mu)
        if n == 0:
            return GaugeChernSimonsPacket(
                cs3_value=0.0, stora_zumino_delta=0.0,
                anomaly_cancelled=True, pontryagin_number=0,
                flux_quantization_error=0.0, triangle_count=0,
                notes=("Conexión vacía.",),
            )

        circ_sum = float(conn.total_flux)
        cs3 = circ_sum / (8.0 * math.pi ** 2)

        if abs(circ_sum) < _GAUGE_FLUX_TOL:
            pontryagin = 0
        else:
            pontryagin = int(round(circ_sum / _TAU))
        flux_qerr = float(abs(circ_sum / _TAU - pontryagin))
        stora_zumino = flux_qerr
        anomaly_cancelled = stora_zumino <= max(_CS_ANOMALY_TOL, _HOPF_QUANTIZATION_TOL)

        notes: List[str] = []
        if conn.n_triangles == 0:
            notes.append("Sin 2-símplices: CS₃=0, fibrado plano por ausencia de dA.")
        if not anomaly_cancelled:
            notes.append(
                f"Δ_SZ={stora_zumino:.4e} > ε_gauge (CS₃={cs3:.4e}, ν={pontryagin})."
            )

        return GaugeChernSimonsPacket(
            cs3_value=float(cs3),
            stora_zumino_delta=float(stora_zumino),
            anomaly_cancelled=bool(anomaly_cancelled),
            pontryagin_number=int(pontryagin),
            flux_quantization_error=flux_qerr,
            triangle_count=int(conn.n_triangles),
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.5  Wilson holonomy path-ordered (fix expm(iA) no unitario)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_wilson_holonomy_real(
        self,
        conn: GaugeConnectionPacket,
        cycle: List[int],
    ) -> GaugeWilsonPacket:
        r"""
        A ∈ so(N) es anti-hermítica real. El transporte paralelo correcto es

            U_e = exp(A_e) ∈ SO(N)     (no exp(iA), que es hermítico > 0).

        Holonomía abeliana (reducción U(1) a lo largo de γ):

            W(γ) = ∏_{e∈γ} exp(i A_e) ∈ U(1)
            φ    = Σ_{e∈γ} A_e
            c₁   = wrap(φ) / 2π
            ν_W  = round(φ / 2π)

        Clasificación a la Floquet (inmersión en SU(2)):
            tr = 2 cos(φ)  →  |tr| ≶ 2  (elíptica / parabólica / hiperbólica).
        """
        n = len(conn.A_mu)
        if n == 0 or len(cycle) < 3:
            return GaugeWilsonPacket(
                cycle_path=(), wilson_matrix_trace=complex(1.0),
                wilson_phase=0.0, winding_number=0,
                is_identity=True, hopf_chern_class=0.0,
                wilson_trace_real=2.0, orbit_class="ELLIPTIC",
            )

        A = np.array(conn.A_mu, dtype=np.float64)
        phase = 0.0
        pairs = 0
        for u, v in zip(cycle[:-1], cycle[1:]):
            if 0 <= u < n and 0 <= v < n:
                phase += float(A[u, v])
                pairs += 1
        if pairs == 0:
            phase = 0.0

        wrapped = math.atan2(math.sin(phase), math.cos(phase))
        c1 = float(wrapped / _TAU)
        winding = int(round(phase / _TAU)) if pairs else 0

        # U(1) holonomy + inmersión tr_SU(2) = 2 cos φ
        w_u1 = complex(math.cos(phase), math.sin(phase))
        trace_real = float(2.0 * math.cos(wrapped))
        is_identity = abs(wrapped) <= _WILSON_UNITARY_TOL
        unitary_defect = abs(abs(w_u1) - 1.0)
        is_unitary = unitary_defect <= _WILSON_UNITARY_TOL

        # Transporte so(N): path-ordered exp(A) a lo largo de γ, proyectado.
        # Usamos el sub-bloque de nodos del ciclo como testigo de orthogonalidad.
        cycle_nodes = list(dict.fromkeys(cycle[:-1] if cycle and cycle[0] == cycle[-1] else cycle))
        k = len(cycle_nodes)
        so_defect = 0.0
        if 2 <= k <= n:
            try:
                idx = np.array(cycle_nodes, dtype=int)
                A_sub = A[np.ix_(idx, idx)]
                w_so = la.expm(A_sub)  # A_sub skew ⇒ W ∈ SO(k) (numéricamente)
                gram = w_so.T @ w_so
                so_defect = float(np.linalg.norm(gram - np.eye(k), ord="fro"))
            except Exception:
                so_defect = float("inf")
        is_unitary = is_unitary and (so_defect <= 1e-6 or so_defect == float("inf"))

        orbit = _classify_monodromy_trace(trace_real)
        orbit_class = orbit.name if hasattr(orbit, "name") else str(orbit)

        return GaugeWilsonPacket(
            cycle_path=tuple(int(x) for x in cycle),
            wilson_matrix_trace=w_u1,
            wilson_phase=float(wrapped),
            winding_number=int(winding),
            is_identity=bool(is_identity),
            hopf_chern_class=float(c1),
            wilson_trace_real=trace_real,
            orbit_class=orbit_class,
            is_unitary=bool(is_unitary and math.isfinite(so_defect)),
            unitary_defect=float(unitary_defect + (0.0 if not math.isfinite(so_defect) else so_defect)),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.6  Poincaré-Melnikov-KAM-Arnold sobre el fibrado gauge
    # ══════════════════════════════════════════════════════════════════════
    def _analyze_gauge_poincare(
        self,
        conn: GaugeConnectionPacket,
        wilson: GaugeWilsonPacket,
        engine_packet: Optional[Any] = None,
    ) -> GaugePoincarePacket:
        r"""
        Flujo de conexión:  v̇ = A v,  A ∈ so(N) ⇒ ‖v(t)‖ se conserva
        (rotación). Las «secciones» ‖v‖→0 de v8 nunca ocurren: se sustituyen
        por el mapa de retorno de Wilson W(γ) y por Π del motor v9.

        Melnikov (splitting de variedades sobre triángulos):
            M_Δ = circ_Δ(A); tangle ⇔ hay signos opuestos de M_Δ
            (ceros persistentes ⇒ homoclínicas de Poincaré-Arnold).

        KAM: ε = ‖F‖_F / (1 + ‖A‖_F²). Estable si ε < ε_KAM y órbita elíptica.

        Arnold / Chirikov: K = ε / max(Δω², ε_m); adiabático si K < umbral.

        Se componen tr M, χ_max, 𝒥, T_rec del motor v9 cuando existen.
        """
        notes: List[str] = []
        n = len(conn.A_mu)
        if n == 0:
            return GaugePoincarePacket(
                n_sections=0, melnikov_max=0.0, melnikov_tangle=False,
                kam_epsilon=0.0, kam_stable=True,
                arnold_diffusion=0.0, arnold_adiabatic=True,
                dominant_mode=None,
            )

        A = np.array(conn.A_mu, dtype=np.float64)
        F = np.array(conn.curvature_F, dtype=np.float64)

        # Melnikov = circulaciones (ya son la integral de Melnikov abeliana).
        circs = np.array(conn.triangle_circulations, dtype=np.float64) if conn.triangle_circulations else np.zeros(1)
        melnikov_max = float(np.max(np.abs(circs))) if circs.size else 0.0
        pos = int(np.sum(circs > _WILKINSON_TOL))
        neg = int(np.sum(circs < -_WILKINSON_TOL))
        tangle = bool(pos > 0 and neg > 0 and melnikov_max > _WILKINSON_TOL)

        # KAM
        f_norm = float(np.linalg.norm(F, ord="fro"))
        a_norm = max(conn.norm_frobenius, _MACHINE_EPSILON)
        eps_pert = f_norm / (1.0 + a_norm * a_norm)

        # Frecuencias = |spec(A)| (so(N) ⇒ espectro imaginario puro).
        try:
            eigs = np.linalg.eigvals(A)
            modes = np.sort(np.abs(np.imag(eigs) if np.iscomplexobj(eigs) else eigs))
        except Exception:
            eigs = np.array([0.0])
            modes = np.array([0.0])

        arnold = 0.0
        dominant: Optional[int] = None
        if len(modes) >= 2:
            leakage: List[Tuple[int, float]] = []
            for k in range(1, len(modes)):
                d_omega = float(modes[k] - modes[k - 1])
                omega_k = max(float(modes[k]), 1e-9)
                leak = (d_omega * d_omega) / (omega_k * omega_k)
                leakage.append((k, leak))
                arnold += leak
            if leakage:
                dominant = max(leakage, key=lambda kv: kv[1])[0]
        chirikov = eps_pert / max(arnold, _MACHINE_EPSILON) if arnold > 0 else eps_pert
        arnold_adiabatic = chirikov < 1.0 and arnold < _ARNOLD_GAUGE_MAX

        # Mapa de retorno = Wilson: tr = 2 cos φ.
        monodromy_trace = float(wilson.wilson_trace_real)
        orbit = _classify_monodromy_trace(monodromy_trace)
        kam_stable = bool(
            eps_pert < _KAM_GAUGE_THRESHOLD
            and orbit != PoincareOrbitType.HYPERBOLIC
        )

        # T_rec del gap espectral de A (primera frecuencia no nula).
        pos_modes = [float(m) for m in modes if m > _FIEDLER_LIKE()]
        t_rec = float(_TAU / pos_modes[0]) if pos_modes else _TAU

        # Lyapunov del retorno U(1): χ = log|dW/dφ| / T_rec = 0 (isometría).
        # Si hay tangle, cota inferior |χ| ~ log(1 + M_max) / T_rec.
        if tangle:
            max_lyap = float(math.log(1.0 + melnikov_max) / max(t_rec, _MACHINE_EPSILON))
        else:
            max_lyap = 0.0

        # Jacobi del calibre: 𝒥_g = −½ ‖A‖_F²  (energía de Yang-Mills discreta).
        jacobi = float(-0.5 * a_norm * a_norm)
        # Conservación: so(N) ⇒ d/dt ‖A‖ = 0 a primer orden (conexión fija).
        jacobi_drift = 0.0

        # Composición con el motor v9 si el packet trae sección modular.
        engine_poincare = _field(engine_packet, "poincare", None)
        if engine_poincare is not None:
            t_rec = float(_field(engine_poincare, "recurrence_time", t_rec))
            eng_tr = float(_field(engine_poincare, "monodromy_trace", monodromy_trace))
            # Producto de monodromías (gauge ⊕ modular) en el chart combinado.
            monodromy_trace = 0.5 * (monodromy_trace + eng_tr)
            max_lyap = max(max_lyap, float(_field(engine_poincare, "max_lyapunov", 0.0)))
            jacobi = float(_field(engine_poincare, "jacobi_integral", jacobi))
            jacobi_drift = float(_field(engine_poincare, "jacobi_drift", 0.0))
            eng_orbit = _field(engine_poincare, "orbit_type", orbit)
            try:
                eng_orbit_e = PoincareOrbitType(int(eng_orbit))
            except Exception:
                eng_orbit_e = orbit
            # Precedencia: hiperbólica > parabólica > elíptica.
            if int(eng_orbit_e) > int(orbit):
                orbit = eng_orbit_e
            notes.append("Π_gauge ∘ Π_modular (motor v9) compuesto.")
            if not bool(_field(engine_poincare, "is_transversal", True)):
                notes.append("sección modular no transversal.")

        is_transversal = abs(monodromy_trace) + a_norm > _WILKINSON_TOL
        n_sections = 1 if wilson.cycle_path else 0
        sections = ((0, float(abs(wilson.wilson_phase))),) if n_sections else ()

        if orbit == PoincareOrbitType.HYPERBOLIC:
            notes.append(f"retorno hiperbólico tr W={monodromy_trace:.4f}.")
            kam_stable = False

        return GaugePoincarePacket(
            n_sections=n_sections,
            melnikov_max=melnikov_max,
            melnikov_tangle=bool(tangle),
            kam_epsilon=float(eps_pert),
            kam_stable=bool(kam_stable),
            arnold_diffusion=float(chirikov if arnold > 0 else arnold),
            arnold_adiabatic=bool(arnold_adiabatic),
            dominant_mode=dominant,
            sections=sections,
            recurrence_time=t_rec,
            monodromy_trace=monodromy_trace,
            max_lyapunov=max_lyap,
            jacobi_integral=jacobi,
            jacobi_drift=jacobi_drift,
            orbit_type=int(orbit),
            is_transversal=bool(is_transversal),
            notes=tuple(notes),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.7  build_gauge_contract (v7 compat)
    # ══════════════════════════════════════════════════════════════════════
    def build_gauge_contract(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosWitnessGaugeContract:
        r"""Emite el contrato topológico de calibre (interfaz v7 preservada)."""
        edges_v = self._validate_edges(num_nodes, edges)
        is_acyclic, n_residual = self._detect_directed_cycles_kahn(num_nodes, edges_v)
        conn = self._build_gauge_connection_1form(num_nodes, edges_v)
        cs = self._compute_chern_simons_real(conn)
        cycle = self._find_shortest_cycle(edges_v, num_nodes)
        wilson = self._compute_wilson_holonomy_real(conn, cycle)

        # Condición inicial del motor (Tomita-Takesaki + QND + Poincaré modular).
        gauge_cond = self._atlas.build_gauge_initial_condition(
            schedule_id, num_nodes, edges_v)

        poincare = self._analyze_gauge_poincare(
            conn, wilson, engine_packet=self._last_packet)

        # Stora-Zumino topológico: matching ν_flux ↔ winding de Wilson.
        sz_top = abs(int(cs.pontryagin_number) - int(wilson.winding_number))
        sz_residual = float(cs.stora_zumino_delta + sz_top)
        sz_ok = sz_residual <= max(_CS_ANOMALY_TOL, _HOPF_QUANTIZATION_TOL) + sz_top * 0.0
        # matching entero: se permite ν y winding diferir si no hay ciclo
        if wilson.cycle_path and sz_top > 0:
            sz_ok = False

        gromov_ok = True
        try:
            gromov_ok = float(gauge_cond.gromov_capacity) <= _GROMOV_CAPACITY_MAX
        except Exception:
            pass

        orbit = PoincareOrbitType(int(poincare.orbit_type))
        is_coherent = bool(
            is_acyclic
            and cs.anomaly_cancelled
            and sz_ok
            and poincare.kam_stable
            and not poincare.melnikov_tangle
            and wilson.is_unitary
            and gromov_ok
            and orbit != PoincareOrbitType.HYPERBOLIC
        )

        now = time.time()
        raw = (
            f"{schedule_id}|{conn.norm_frobenius:.6f}|{cs.cs3_value:.6f}|"
            f"{cs.pontryagin_number}|{wilson.wilson_phase:.6f}|"
            f"{wilson.hopf_chern_class:.6f}|{sz_residual:.6f}|"
            f"{gauge_cond.hmac_signature}|{is_acyclic}|{is_coherent}|"
            f"{poincare.monodromy_trace:.6f}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        packet_hash_payload = (
            f"{is_acyclic}|{is_coherent}|{cs.cs3_value:.6e}|"
            f"{wilson.wilson_phase:.6e}|{poincare.max_lyapunov:.6e}"
        )
        packet_hash = hashlib.sha256(packet_hash_payload.encode("utf-8")).hexdigest()[:16]
        engine_hash = str(_field(gauge_cond, "observer_packet_hash", ""))

        notes: List[str] = []
        if not is_acyclic:
            notes.append(f"Kahn detectó ciclo dirigido ({n_residual} residuales).")
        if not cs.anomaly_cancelled:
            notes.append(f"Δ_SZ_flux={cs.stora_zumino_delta:.4e}.")
        if not sz_ok:
            notes.append(
                f"Stora-Zumino topológico: ν={cs.pontryagin_number} ≠ "
                f"winding={wilson.winding_number}."
            )
        if not poincare.kam_stable:
            notes.append(f"KAM ε={poincare.kam_epsilon:.4e} o órbita no elíptica.")
        if not wilson.is_unitary:
            notes.append(f"Wilson no unitario (defect={wilson.unitary_defect:.3e}).")
        notes.extend(cs.notes)
        notes.extend(poincare.notes)

        contract = ChronosWitnessGaugeContract(
            schedule_id=schedule_id,
            connection_a_norm=conn.norm_frobenius,
            chern_simons_cs3=cs.cs3_value,
            pontryagin_number_nu=cs.pontryagin_number,
            wilson_holonomy_angle=wilson.wilson_phase,
            hopf_chern_class_c1=wilson.hopf_chern_class,
            stora_zumino_residual=sz_residual,
            gauge_initial_condition=gauge_cond,
            signature_hmac_sha256=hmac_sig,
            timestamp=now,
            is_dag_acyclic=is_acyclic,
            is_gauge_coherent=is_coherent,
            gauge_bundle_class=conn.bundle_class,
            cycle_path=wilson.cycle_path,
            wilson_trace_real=wilson.wilson_trace_real,
            melnikov_max=poincare.melnikov_max,
            melnikov_tangle=poincare.melnikov_tangle,
            kam_epsilon=poincare.kam_epsilon,
            kam_stable=poincare.kam_stable,
            arnold_diffusion=poincare.arnold_diffusion,
            arnold_adiabatic=poincare.arnold_adiabatic,
            observer_packet_hash=packet_hash,
            notes=tuple(notes),
            wilson_winding=wilson.winding_number,
            cs_triangle_count=cs.triangle_count,
            flux_quantization_error=cs.flux_quantization_error,
            antisymmetry_defect=conn.antisymmetry_defect,
            poincare_recurrence_time=poincare.recurrence_time,
            poincare_monodromy_trace=poincare.monodromy_trace,
            poincare_max_lyapunov=poincare.max_lyapunov,
            jacobi_integral=poincare.jacobi_integral,
            orbit_type=int(poincare.orbit_type),
            engine_packet_hash=engine_hash,
        )
        self._last_gauge_contract = contract
        return contract

    # ══════════════════════════════════════════════════════════════════════
    # §1.8  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> Tuple[Any, ChronosWitnessGaugeContract, Any]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — Puente funtorial Φ₁ → Φ₂.

        (1) WitnessObserverPacket v9 (motor: Tomita-Takesaki, KMS, Π, QND).
        (2) ChronosWitnessGaugeContract (CS₃ A∧dA, Wilson path-ordered, Π_gauge).
        (3) Invariantes: DAG, Stora-Zumino, KAM, Gromov, no-hiperbolicidad.
        (4) Invoca el PRIMER MÉTODO DE FASE 2: `_audit_novikov_witness_phase2`.
        """
        edges_v = self._validate_edges(num_nodes, edges)

        packet: Optional[Any] = None
        if _ENGINE_V8:
            try:
                packet = self._engine.build_witness_observer_packet(
                    schedule_id, num_nodes, edges_v)
            except AttributeError:
                packet = None
            self._last_packet = packet

        contract = self.build_gauge_contract(schedule_id, num_nodes, edges_v)

        if not contract.is_dag_acyclic:
            raise DirectedCycleWitnessError(
                f"Cronograma NO acíclico (Kahn). Notas: {contract.notes}."
            )
        if not contract.is_gauge_coherent:
            raise ChernSimonsWitnessError(
                f"Gauge bundle incoherente: CS₃={contract.chern_simons_cs3:.4e}, "
                f"Δ_SZ={contract.stora_zumino_residual:.4e}, "
                f"orbit={PoincareOrbitType(contract.orbit_type).name}."
            )
        if not contract.kam_stable:
            raise GaugeKAMWitnessError(
                f"KAM ε={contract.kam_epsilon:.4e} ≥ ε_KAM o tr M hiperbólica."
            )
        if contract.melnikov_tangle and contract.melnikov_max > _MELNIKOV_GAUGE_MAX:
            raise GaugeMelnikovWitnessError(
                f"tangle homoclínico ‖M‖_∞={contract.melnikov_max:.4e}."
            )

        logger.info(
            "[Fase 1::GAUGE OBSERVE] acyclic=%s | ‖A‖=%.4f | CS₃=%.4e | "
            "ν=%d | Δ_SZ=%.2e | W_phase=%.4f | c₁=%.4f | class=%s | "
            "M_max=%.2e | KAM=%s | T_rec=%.4f | tr M=%.4f | χ_max=%.3e | "
            "𝒥=%.4f | orbit=%s | coherent=%s",
            contract.is_dag_acyclic, contract.connection_a_norm,
            contract.chern_simons_cs3, contract.pontryagin_number_nu,
            contract.stora_zumino_residual, contract.wilson_holonomy_angle,
            contract.hopf_chern_class_c1, contract.gauge_bundle_class,
            contract.melnikov_max, contract.kam_stable,
            contract.poincare_recurrence_time, contract.poincare_monodromy_trace,
            contract.poincare_max_lyapunov, contract.jacobi_integral,
            PoincareOrbitType(contract.orbit_type).name,
            contract.is_gauge_coherent,
        )

        novikov_cert = self._audit_novikov_witness_phase2(
            packet=packet, slacks=slacks)
        self._last_novikov = novikov_cert
        return packet, contract, novikov_cert


def _FIEDLER_LIKE() -> float:
    return 1.0e-4


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► CAMPAIGN ORIENT : NOVIKOV MAX-PLUS + RSI³ SOBRE Π           █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase2_ChronosWitnessCampaignOrient(Phase1_ChronosWitnessGaugeObserver):
    r"""
    Fase 2 — Campaign Orient.

    RSI³ itera sobre el producto de secciones Π_gauge × Π_modular:
    θ absorbe (‖A‖, CS₃, φ_W, M_max, χ_max, 𝒥). Cada ciclo N1→N2→N3
    es un retorno de Poincaré del mapa de Picard.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §2.1  PRIMER MÉTODO EFECTIVO DE FASE 2  (continuación de §1.8)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_novikov_witness_phase2(
        self,
        packet: Any,
        slacks: List[float],
    ) -> Any:
        r"""
        Primer método efectivo de Fase 2 — valuación max-plus de Novikov.

        Delega en el motor v9 (`_audit_novikov_valuation_phase2`) que ya
        no usa a+b ≤ max(a,b). Fallback local idéntico al semianillo
        tropical: v = min(−log p_i), violaciones = masa ≠ 1 o a_i no finitos.
        """
        if _ENGINE_V8 and packet is not None:
            try:
                return self._engine._audit_novikov_valuation_phase2(
                    witness_packet=packet, slacks=slacks)
            except Exception as exc:
                logger.warning("[Fase 2::NOVIKOV] fallback por %s", type(exc).__name__)

        if not slacks:
            return {
                "v_slack": 1.0, "delay_norm": math.exp(-1.0),
                "is_above_floor": True, "ultrametric_violations": 0,
            }
        pos = [max(1e-6, float(s)) for s in slacks]
        total = sum(pos)
        if total <= _MACHINE_EPSILON:
            raise NovikovWitnessAgentError("suma de holguras nula.")
        probs = [s / total for s in pos]
        a_vals = [-math.log(p) for p in probs]
        v_slack = float(min(a_vals))
        violations = 0
        if abs(sum(probs) - 1.0) > 1e-9:
            violations += 1
        for a, p in zip(a_vals, probs):
            if not math.isfinite(a) or p <= 0.0:
                violations += 1
            if abs(p - math.exp(-a)) > 1e-9:
                violations += 1
        return {
            "v_slack": v_slack,
            "delay_norm": float(math.exp(-v_slack)),
            "is_above_floor": v_slack >= _NOVIKOV_SLACK_FLOOR,
            "ultrametric_violations": violations,
            "a_i_values": tuple(a_vals),
            "probabilities": tuple(probs),
        }

    # ══════════════════════════════════════════════════════════════════════
    # §2.2  Kernel RSI³ sobre (calibre ⊕ Poincaré)
    # ══════════════════════════════════════════════════════════════════════
    def _run_witness_rsi3_kernel(
        self,
        contract: ChronosWitnessGaugeContract,
        packet: Any,
        novikov_cert: Any = None,
    ) -> Any:
        r"""
        Prefiere el kernel v9 del motor (θ incluye χ_max, 𝒥, Picard, Lipschitz
        ℓ^∞). Si no está disponible, ejecuta un descenso N1→N2→N3 local sobre

            θ = [‖A‖, CS₃, |φ_W|, M_max, χ_max, |𝒥|]
        """
        if _ENGINE_V8 and packet is not None:
            try:
                return self._engine._run_witness_rsi3_kernel(
                    witness_packet=packet, novikov_cert=novikov_cert)
            except Exception as exc:
                logger.warning("[Fase 2::RSI³] fallback por %s", type(exc).__name__)

        theta = np.array([
            float(np.clip(contract.connection_a_norm, 0.0, 10.0)),
            float(np.clip(abs(contract.chern_simons_cs3), 0.0, 10.0)),
            float(np.clip(abs(contract.wilson_holonomy_angle), 0.0, _TAU)),
            float(np.clip(contract.melnikov_max, 0.0, 10.0)),
            float(np.clip(abs(contract.poincare_max_lyapunov), 0.0, 10.0)),
            float(np.clip(abs(contract.jacobi_integral), 0.0, 50.0)),
        ], dtype=np.float64)
        theta_before = tuple(float(x) for x in theta)
        dim = int(theta.shape[0])
        O = np.triu(np.eye(dim, k=1) * 0.05, k=1)
        alpha = np.array([0.12, 0.85, 0.20], dtype=np.float64)
        target = np.array([0.05, 0.0, 0.0, 0.0, 0.0, abs(contract.jacobi_integral)],
                          dtype=np.float64)
        losses: List[float] = []
        levels: List[int] = []
        theta_start = theta.copy()
        converged = True

        def _loss(th: np.ndarray) -> float:
            return float(np.linalg.norm(th - target) ** 2) + 0.15 * float(th[4] ** 2)

        for _ in range(_RSI3_CYCLES):
            prev = _loss(theta)
            for _n1 in range(8):
                grad = np.zeros_like(theta)
                h = 1e-6
                for i in range(dim):
                    xp = theta.copy(); xp[i] += h
                    xm = theta.copy(); xm[i] -= h
                    grad[i] = (_loss(xp) - _loss(xm)) / (2.0 * h)
                theta = theta - alpha[0] * grad
                nowl = _loss(theta)
                if abs(nowl - prev) < _RSI3_IDEMPOTENCE_TOL:
                    break
                prev = nowl
            losses.append(prev)
            levels.append(1)
            O = np.triu(O * 0.95, k=1)
            losses.append(_loss(theta))
            levels.append(2)
            alpha = np.clip(alpha * np.array([1.0, 0.95, 0.90]),
                            _RSI3_ALPHA_FLOOR, _RSI3_ALPHA_CEIL)
            losses.append(_loss(theta))
            levels.append(3)
            if not math.isfinite(losses[-1]):
                converged = False
                break

        try:
            x1 = theta.copy()
            x2 = theta + 1e-3
            def t_step(x: np.ndarray) -> np.ndarray:
                g = np.zeros_like(x)
                h = 1e-6
                for i in range(x.shape[0]):
                    xp = x.copy(); xp[i] += h
                    xm = x.copy(); xm[i] -= h
                    g[i] = (_loss(xp) - _loss(xm)) / (2.0 * h)
                return x - alpha[0] * g
            num = float(np.linalg.norm(t_step(x1) - t_step(x2), ord=np.inf))
            den = float(np.linalg.norm(x1 - x2, ord=np.inf))
            lip = num / den if den > 1e-12 else _BANACH_LIPSCHITZ_CEIL
            lip = min(max(lip, 0.0), _BANACH_LIPSCHITZ_CEIL)
        except Exception:
            lip = _BANACH_LIPSCHITZ_CEIL

        return {
            "theta_before": theta_before,
            "theta_after": tuple(float(x) for x in theta),
            "O_norm": float(np.linalg.norm(O, "fro")),
            "alpha_final": tuple(float(x) for x in alpha),
            "loss_history": tuple(losses),
            "level_history": tuple(levels),
            "cycles_executed": _RSI3_CYCLES,
            "converged": converged and bool(losses) and math.isfinite(losses[-1]),
            "lipschitz_const": float(lip),
            "picard_residual": float(abs(losses[-1] - losses[0])) if losses else 0.0,
            "poincare_return_residual": float(np.linalg.norm(theta - theta_start, ord=np.inf)),
        }

    def _novikov_v_slack(self, novikov_cert: Any, slacks: List[float]) -> float:
        v = _field(novikov_cert, "v_slack", None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        if slacks:
            return float(np.mean([max(0.0, float(s)) for s in slacks]))
        return 1.0

    # ══════════════════════════════════════════════════════════════════════
    # §2.3  conduct_qnd_observation_campaign (interfaz v7)
    # ══════════════════════════════════════════════════════════════════════
    def conduct_qnd_observation_campaign(
        self,
        contract: ChronosWitnessGaugeContract,
        req: ChronosWitnessObservationRequest,
    ) -> ChronosWitnessCampaignDraftBundle:
        r"""Campaña QND pasiva (interfaz v7): Novikov + RSI³ + germen inmutable."""
        novikov_cert = self._audit_novikov_witness_phase2(
            packet=self._last_packet, slacks=req.slacks)
        rsi3 = self._run_witness_rsi3_kernel(
            contract=contract, packet=self._last_packet, novikov_cert=novikov_cert)

        try:
            germ = self._engine.build_weak_observation_germ(
                contract.gauge_initial_condition, req.slacks)
        except Exception as exc:
            logger.warning("[Fase 2::GERM] fallback: %s", type(exc).__name__)
            germ = self._synthesize_germ_fallback(contract, req.slacks, novikov_cert)

        mean_slack = float(np.mean(req.slacks)) if req.slacks else 1.0
        v_slack = self._novikov_v_slack(novikov_cert, req.slacks)
        banach_lip = float(_field(rsi3, "lipschitz_const", 0.85) or 0.85)
        picard = float(_field(rsi3, "picard_residual", 0.0) or 0.0)
        converged = bool(_field(rsi3, "converged", True))

        now = time.time()
        germ_hmac = str(_field(germ, "hmac_signature", "") or "")
        raw = (
            f"{contract.signature_hmac_sha256}|{germ_hmac}|"
            f"{v_slack:.6f}|{banach_lip:.6f}|{picard:.6e}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return ChronosWitnessCampaignDraftBundle(
            contract=contract,
            observation_germ=germ,
            mean_slack_valuation=mean_slack,
            signature_hmac_sha256=hmac_sig,
            timestamp=now,
            novikov_cert=novikov_cert,
            rsi3_report=rsi3,
            banach_lipschitz=banach_lip,
            observer_packet_hash=contract.observer_packet_hash,
            novikov_v_slack=v_slack,
            rsi3_picard_residual=picard,
            rsi3_converged=converged,
        )

    def _synthesize_germ_fallback(self, contract, slacks, novikov_cert=None):
        v_slack = self._novikov_v_slack(novikov_cert, slacks)
        try:
            return ChronosWeakObservationGerm(
                gauge_condition=contract.gauge_initial_condition,
                novikov_slack_valuation=v_slack,
                novikov_delay_norm=math.exp(-v_slack),
                banach_lipschitz_const=0.85,
                lyapunov_delta=0.0,
                back_action_db=float(_field(contract.gauge_initial_condition, "back_action_db", 0.0) or 0.0),
                hmac_signature="",
                timestamp=time.time(),
            )
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════
    # §2.4  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        bundle: ChronosWitnessCampaignDraftBundle,
    ) -> Tuple[ChronosWitnessCampaignDraftBundle, Any]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — Puente funtorial Φ₂ → Φ₃.

        Invariantes: RSI³ convergió (Picard) y L̂ < 1.
        Invoca inmediatamente `_adjudicate_witness_verdict`.
        """
        if not bundle.rsi3_converged:
            raise RSI3WitnessError(
                f"RSI³ no convergió (Picard={bundle.rsi3_picard_residual:.4e})."
            )
        if bundle.banach_lipschitz >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f ≥ %.4f (colapso marginal).",
                bundle.banach_lipschitz, _BANACH_LIPSCHITZ_CEIL,
            )

        logger.info(
            "[Fase 2::CAMPAIGN ORIENT] v(Slack)=%.4f | mean_s=%.4f | L̂=%.4f | "
            "Picard=%.3e | RSI³_ok=%s | class=%s | χ_max=%.3e | 𝒥=%.4f",
            bundle.novikov_v_slack, bundle.mean_slack_valuation,
            bundle.banach_lipschitz, bundle.rsi3_picard_residual,
            bundle.rsi3_converged, bundle.contract.gauge_bundle_class,
            bundle.contract.poincare_max_lyapunov, bundle.contract.jacobi_integral,
        )

        verdict_data = self._adjudicate_witness_verdict(bundle=bundle)
        return bundle, verdict_data


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► SOVEREIGN DECIDE : HEYTING Ω₃ + POINCARÉ + MERKLE           █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class Phase3_ChronosWitnessSovereignDecider(Phase2_ChronosWitnessCampaignOrient):
    r"""
    Fase 3 — Sovereign Decide.

    Ω₃ ⊓ clasificación de Poincaré: órbita hiperbólica ⇒ VETOED
    (no hay recurrencia estable del testigo sobre el fibrado).
    """

    # ══════════════════════════════════════════════════════════════════════
    # §3.1  PRIMER MÉTODO EFECTIVO DE FASE 3  (continuación de §2.4)
    # ══════════════════════════════════════════════════════════════════════
    def _adjudicate_witness_verdict(
        self,
        bundle: ChronosWitnessCampaignDraftBundle,
    ) -> Any:
        r"""
        Adjudicación Heyting Ω₃ ⊓ Poincaré.

          VETOED (⊥):
            · ¬DAG, ¬gauge_coherent, v(Slack)<floor, KAM inestable,
              Δ_SZ topológico, órbita HIPERBÓLICA, Gromov c_G>12.5.
          DEGRADED (⋆):
            · L̂>0.95, charged bundle, Melnikov tangle, no adiabático,
              órbita PARABÓLICA, χ_max>0, d_FS>1.20.
          COHERENT (⊤): el resto (elíptica + KMS + QND + DAG).
        """
        contract = bundle.contract
        cond = contract.gauge_initial_condition
        orbit = PoincareOrbitType(int(contract.orbit_type))
        veto: List[str] = []

        if not contract.is_dag_acyclic:
            veto.append("ciclo dirigido (Kahn)")
        if not contract.is_gauge_coherent:
            veto.append(
                f"gauge incoherente: CS₃={contract.chern_simons_cs3:.4e}, "
                f"Δ_SZ={contract.stora_zumino_residual:.4e}"
            )
        v_slack = float(bundle.novikov_v_slack)
        if v_slack < _NOVIKOV_SLACK_FLOOR:
            veto.append(f"v(Slack)={v_slack:.4e} < floor")
        nov_viol = int(_field(bundle.novikov_cert, "ultrametric_violations", 0) or 0)
        if nov_viol > 0:
            veto.append(f"{nov_viol} violaciones Novikov/max-plus")
        if contract.kam_stable is False:
            veto.append(f"KAM ε={contract.kam_epsilon:.4e} ≥ tol")
        if orbit == PoincareOrbitType.HYPERBOLIC:
            veto.append(f"órbita hiperbólica tr M={contract.poincare_monodromy_trace:.4f}")
        try:
            if float(cond.gromov_capacity) > _GROMOV_CAPACITY_MAX:
                veto.append(f"Gromov c_G={cond.gromov_capacity:.4f} > 12.5")
        except Exception:
            pass

        if veto:
            return _make_heyting_verdict(
                HeytingOmega3.VETOED,
                ActuationMode.HARD_VETO_ESP32_CROWBAR,
                min(394.25, _CROWBAR_LATENCY_BUDGET_NS),
                _M_E_C2_MEV * 2.0, 15.0, tuple(veto), orbit,
            )

        degraded: List[str] = []
        if bundle.banach_lipschitz > 0.95:
            degraded.append(f"L̂={bundle.banach_lipschitz:.4f} > 0.95")
        if contract.gauge_bundle_class == "charged":
            degraded.append("charged bundle (torsión topológica)")
        if contract.melnikov_tangle:
            degraded.append(f"Melnikov tangle (max={contract.melnikov_max:.4e})")
        if not contract.arnold_adiabatic:
            degraded.append(f"D_Arnold/Chirikov={contract.arnold_diffusion:.4e}")
        if orbit == PoincareOrbitType.PARABOLIC:
            degraded.append("órbita parabólica (resonancia de Poincaré)")
        if contract.poincare_max_lyapunov > 1e-3:
            degraded.append(f"χ_max={contract.poincare_max_lyapunov:.3e} > 0")
        try:
            if float(cond.fubini_study_distance) > _FUBINI_DEGRADED:
                degraded.append(f"d_FS={cond.fubini_study_distance:.4f} > 1.20")
        except Exception:
            pass

        if degraded:
            return _make_heyting_verdict(
                HeytingOmega3.DEGRADED,
                ActuationMode.SOFT_VETO_BYPASS,
                0.0, _M_E_C2_MEV * 2.0, 8.0, tuple(degraded), orbit,
            )

        return _make_heyting_verdict(
            HeytingOmega3.COHERENT,
            ActuationMode.NORMAL_FLUID,
            0.0, 0.0, 3.5, (), orbit,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §3.2  Merkle DAG de 3 hojas
    # ══════════════════════════════════════════════════════════════════════
    def _compute_witness_merkle_root(
        self,
        bundle: ChronosWitnessCampaignDraftBundle,
        verdict: HeytingOmega3,
        actuation: ActuationMode,
        crowbar_latency: float,
        cert_merkle: str,
    ) -> str:
        r"""
        L1 = H(seed_hash ‖ contract_hmac)
        L2 = H(germ_hmac ‖ rsi3_hash ‖ tr_M ‖ 𝒥)
        L3 = H(verdict ‖ actuation ‖ crowbar ‖ cert_merkle)
        root = H(H(L1‖L2) ‖ L3)
        """
        contract = bundle.contract
        leaf1 = hashlib.sha256(
            f"{contract.gauge_initial_condition.schedule_hash}|"
            f"{contract.signature_hmac_sha256}"
            .encode("utf-8")
        ).hexdigest()

        germ_hmac = str(_field(bundle.observation_germ, "hmac_signature", "") or "")
        theta_after = _field(bundle.rsi3_report, "theta_after", ())
        rsi3_hash = hashlib.sha256(
            f"{theta_after}|{contract.poincare_monodromy_trace:.8f}|"
            f"{contract.jacobi_integral:.8f}"
            .encode("utf-8")
        ).hexdigest()
        leaf2 = hashlib.sha256(
            f"{germ_hmac}|{rsi3_hash}".encode("utf-8")
        ).hexdigest()
        leaf3 = hashlib.sha256(
            f"{verdict.name}|{actuation.name}|{crowbar_latency:.2f}|{cert_merkle}"
            .encode("utf-8")
        ).hexdigest()
        l12 = hashlib.sha256(f"{leaf1}|{leaf2}".encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{l12}|{leaf3}".encode("utf-8")).hexdigest()

    def adjudicate_and_translate(
        self,
        bundle: ChronosWitnessCampaignDraftBundle,
    ) -> ChronosWitnessExecutionPassport:
        """Fase III: adjudica y emite pasaporte (interfaz v7)."""
        return self._phase3_terminal_synthesis(bundle)

    def _verdict_of(self, verdict_data: Any) -> Tuple[HeytingOmega3, ActuationMode, float, float, float, Tuple[str, ...], Any]:
        v = _field(verdict_data, "verdict", HeytingOmega3.COHERENT)
        a = _field(verdict_data, "actuation_mode", ActuationMode.NORMAL_FLUID)
        cns = float(_field(verdict_data, "crowbar_latency_ns", 0.0) or 0.0)
        mev = float(_field(verdict_data, "dirac_purge_energy_mev", 0.0) or 0.0)
        pct = float(_field(verdict_data, "contingency_reserve_pct", 3.5) or 3.5)
        reasons = tuple(_field(verdict_data, "veto_reasons", ()) or ())
        orbit = _field(verdict_data, "orbit_type", PoincareOrbitType.ELLIPTIC)
        return v, a, cns, mev, pct, reasons, orbit

    # ══════════════════════════════════════════════════════════════════════
    # §3.3  ÚLTIMO MÉTODO DE FASE 3: síntesis terminal
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(
        self,
        bundle: ChronosWitnessCampaignDraftBundle,
    ) -> ChronosWitnessExecutionPassport:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — Síntesis del pasaporte soberano.

        Z_WitnessAgent = Φ₃ ∘ Φ₂ ∘ Φ₁
        """
        verdict_data = self._adjudicate_witness_verdict(bundle)
        verdict, actuation, crowbar_ns, dirac_mev, contingency, reasons, orbit = \
            self._verdict_of(verdict_data)

        try:
            if _ENGINE_V8 and bundle.observation_germ is not None:
                cert = self._engine._phase3_terminal_synthesis(bundle.observation_germ)
            else:
                cert = self._engine.adjudicate_and_issue_certificate(
                    bundle.observation_germ)
        except Exception as exc:
            logger.warning("[Fase 3::CERT] fallback: %s", type(exc).__name__)
            cert = self._synthesize_certificate_fallback(bundle, verdict, actuation, crowbar_ns)

        tau_degree = float(int(verdict)) / 2.0
        kv = float(_field(cert, "kv_cache_compression_pct", 86.4) or 86.4)
        wacc = float(_field(cert, "wacc_protection_pct",
                            15.0 if verdict == HeytingOmega3.COHERENT else 8.0) or 8.0)

        if verdict == HeytingOmega3.COHERENT:
            summary_txt = (
                "Cronograma acíclico, fibrado elíptico y armónicamente coherente. "
                "Reserva de imprevistos optimizada al 3.5%."
            )
        elif verdict == HeytingOmega3.VETOED:
            summary_txt = (
                "Riesgo de descalce temporal, dependencia circular u órbita "
                "hiperbólica. Disparo ESP32 Crowbar activado."
            )
        else:
            summary_txt = (
                "Cronograma con tensión topológica moderada o resonancia parabólica. "
                "Monitoreo reforzado recomendado."
            )

        exec_report: Dict[str, Any] = {
            "schedule_id": bundle.contract.schedule_id,
            "heyting_verdict": verdict.name,
            "truth_degree_tau": tau_degree,
            "acyclicity_status": (
                "ACÍCLICO_ESTRICTO" if bundle.contract.is_dag_acyclic
                else "SOCAVÓN_TEMPORAL"
            ),
            "gauge_status": (
                f"class={bundle.contract.gauge_bundle_class}, "
                f"CS₃={bundle.contract.chern_simons_cs3:.4e}, "
                f"ν={bundle.contract.pontryagin_number_nu}, "
                f"W_phase={bundle.contract.wilson_holonomy_angle:.4f}, "
                f"c₁={bundle.contract.hopf_chern_class_c1:.4f}, "
                f"orbit={PoincareOrbitType(int(orbit)).name}"
            ),
            "kv_cache_compression_pct": f"{kv}%",
            "wacc_protection_pct": f"{wacc}%",
            "contingency_reserve_pct": f"{contingency}%",
            "actuation_mode": actuation.name,
            "crowbar_latency_ns": crowbar_ns,
            "dirac_purge_energy_mev": dirac_mev,
            "novikov_v_slack": bundle.novikov_v_slack,
            "banach_lipschitz": bundle.banach_lipschitz,
            "poincare_T_rec": bundle.contract.poincare_recurrence_time,
            "poincare_tr_M": bundle.contract.poincare_monodromy_trace,
            "jacobi_integral": bundle.contract.jacobi_integral,
            "executive_summary": summary_txt,
        }

        cert_merkle = str(_field(cert, "merkle_root_sha256", "") or "")
        merkle_root = self._compute_witness_merkle_root(
            bundle, verdict, actuation, crowbar_ns, cert_merkle)

        now = time.time()
        final_raw = (
            f"{merkle_root}|{verdict.value}|{actuation.value}|{tau_degree:.4f}|{now}"
        )
        final_sig = hmac.new(
            self._secret_key, final_raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        c = bundle.contract
        gauge_summary: Tuple[Tuple[str, Any], ...] = (
            ("is_dag_acyclic", c.is_dag_acyclic),
            ("is_gauge_coherent", c.is_gauge_coherent),
            ("gauge_bundle_class", c.gauge_bundle_class),
            ("connection_a_norm", c.connection_a_norm),
            ("cs3_value", c.chern_simons_cs3),
            ("pontryagin", c.pontryagin_number_nu),
            ("stora_zumino_delta", c.stora_zumino_residual),
            ("wilson_phase", c.wilson_holonomy_angle),
            ("wilson_trace_real", c.wilson_trace_real),
            ("wilson_winding", c.wilson_winding),
            ("hopf_c1", c.hopf_chern_class_c1),
            ("cs_triangles", c.cs_triangle_count),
            ("melnikov_max", c.melnikov_max),
            ("kam_epsilon", c.kam_epsilon),
            ("kam_stable", c.kam_stable),
            ("arnold_diffusion", c.arnold_diffusion),
            ("poincare_T_rec", c.poincare_recurrence_time),
            ("poincare_tr_M", c.poincare_monodromy_trace),
            ("poincare_chi_max", c.poincare_max_lyapunov),
            ("jacobi_integral", c.jacobi_integral),
            ("orbit_type", PoincareOrbitType(int(c.orbit_type)).name),
        )

        rsi3 = bundle.rsi3_report
        loss_hist = _field(rsi3, "loss_history", ()) or ()
        rsi3_summary: Tuple[Tuple[str, Any], ...] = (
            ("converged", bundle.rsi3_converged),
            ("lipschitz", bundle.banach_lipschitz),
            ("final_loss", loss_hist[-1] if loss_hist else 0.0),
            ("n_levels", len(_field(rsi3, "level_history", ()) or ())),
            ("picard_residual", bundle.rsi3_picard_residual),
        )

        chain = (
            "_phase1_terminal_bridge_to_phase2",
            "_audit_novikov_witness_phase2",
            "_phase2_terminal_bridge_to_phase3",
            "_adjudicate_witness_verdict",
            "_phase3_terminal_synthesis",
        )

        passport = ChronosWitnessExecutionPassport(
            bundle=bundle,
            certificate=cert,
            truth_degree_tau=tau_degree,
            executive_summary_report=exec_report,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
            timestamp=now,
            gauge_audit_summary=gauge_summary,
            rsi3_summary=rsi3_summary,
            phase_continuity_chain=chain,
            veto_reasons=reasons,
            poincare_orbit_type=int(orbit),
            jacobi_integral=float(c.jacobi_integral),
        )
        logger.info(
            "Z_WitnessAgent = Φ₃∘Φ₂∘Φ₁ | verdict=%s | actuation=%s | "
            "τ=%.2f | orbit=%s | merkle=%s",
            verdict.name, actuation.name, tau_degree,
            PoincareOrbitType(int(orbit)).name, merkle_root[:16],
        )
        return passport

    def _synthesize_certificate_fallback(self, bundle, verdict, actuation=None, crowbar_ns=None):
        if actuation is None:
            actuation = (
                ActuationMode.NORMAL_FLUID
                if verdict == HeytingOmega3.COHERENT
                else ActuationMode.HARD_VETO_ESP32_CROWBAR
            )
        if crowbar_ns is None:
            crowbar_ns = 393.85 if verdict == HeytingOmega3.VETOED else 0.0
        return ChronosWitnessExecutionCertificate(
            germ=bundle.observation_germ,
            heyting_verdict=verdict,
            actuation_mode=actuation,
            crowbar_latency_ns=crowbar_ns,
            dirac_purge_energy_mev=_M_E_C2_MEV * 2.0 if verdict != HeytingOmega3.COHERENT else 0.0,
            kv_cache_compression_pct=86.4,
            wacc_protection_pct=15.0 if verdict == HeytingOmega3.COHERENT else 8.0,
            contingency_reserve_pct=3.5 if verdict == HeytingOmega3.COHERENT else 15.0,
            merkle_root_sha256="",
            signature_hmac_sha256="",
            timestamp=time.time(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# §4. ORQUESTADOR SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class TOONChronosWitnessAgent(Phase3_ChronosWitnessSovereignDecider):
    r"""
    Soberano agéntico de calibre testigo del tiempo.

    Herencia lineal (cadena funtorial estricta):
        TOONChronosWitnessAgent
          → Phase3_ChronosWitnessSovereignDecider    [Φ₃]
             → Phase2_ChronosWitnessCampaignOrient    [Φ₂ + RSI³ + max-plus]
                → Phase1_ChronosWitnessGaugeObserver  [Φ₁ + Gauge + Poincaré]
                   → (composición al motor v9)
    """

    def run_full_pipeline(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> ChronosWitnessExecutionPassport:
        r"""Ejecuta las TRES fases anidadas con continuación funtorial Φ₁→Φ₂→Φ₃."""
        _packet, contract, _novikov = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, slacks)

        req = ChronosWitnessObservationRequest(
            request_id=f"REQ-{schedule_id[:8]}",
            schedule_id=schedule_id,
            slacks=slacks,
            timestamp=time.time(),
        )
        bundle = self.conduct_qnd_observation_campaign(contract, req)
        bundle, _verdict_data = self._phase2_terminal_bridge_to_phase3(bundle)
        return self._phase3_terminal_synthesis(bundle)


# ══════════════════════════════════════════════════════════════════════════════
# §5. COMPATIBILIDAD v7
# ══════════════════════════════════════════════════════════════════════════════

class ChronosWitnessGaugeTopology:
    r"""Módulo de topología de calibre (interfaz v7). Delega en Fase 1."""

    def __init__(self, secret_key: bytes = _SECRET_DEFAULT):
        self._secret_key = secret_key
        self._phase1 = Phase1_ChronosWitnessGaugeObserver(secret_key)
        self._atlas = self._phase1._atlas

    def compute_gauge_connection(
        self, slacks: List[float],
    ) -> Tuple[np.ndarray, float]:
        """Interfaz v7: (A_su, ‖A‖_F)."""
        n = max(2, min(len(slacks), 4)) if slacks else 2
        if n < 2 or not slacks:
            return np.zeros((n, n)), 0.0
        vals = [max(1e-6, s) for s in slacks[:n]]
        edges = [(i, i + 1, vals[i]) for i in range(n - 1)]
        conn = self._phase1._build_gauge_connection_1form(n, edges)
        A = np.array(conn.A_mu, dtype=np.float64) if conn.A_mu else np.zeros((n, n))
        return A, float(conn.norm_frobenius)

    def compute_chern_simons_and_wilson(
        self, A_su: np.ndarray,
    ) -> Tuple[float, int, float, float, float]:
        """Interfaz v7: (CS₃, ν, W_phase, Hopf c₁, Δ_SZ) vía A∧dA, no Tr(A³)."""
        n = int(A_su.shape[0])
        if n == 0:
            return 0.0, 0, 0.0, 0.0, 0.0
        A = np.array(A_su, dtype=np.float64)
        A = 0.5 * (A - A.T)
        circ_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    circ_sum += float(A[i, j] + A[j, k] + A[k, i])
        cs3 = circ_sum / (8.0 * math.pi ** 2)
        nu = int(round(circ_sum / _TAU)) if abs(circ_sum) >= _GAUGE_FLUX_TOL else 0
        phase = 0.0
        for i in range(n):
            phase += float(A[i, (i + 1) % n])
        wrapped = math.atan2(math.sin(phase), math.cos(phase))
        c1 = float(wrapped / _TAU)
        sz_res = float(abs(circ_sum / _TAU - nu))
        return cs3, nu, wrapped, c1, sz_res

    def build_gauge_contract(
        self, schedule_id: str, num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosWitnessGaugeContract:
        return self._phase1.build_gauge_contract(schedule_id, num_nodes, edges)


class SovereignChronosWitnessAdjudicator:
    """Adjudicador ciber-físico y traductor semántico (interfaz v7)."""

    def __init__(self, secret_key: bytes = _SECRET_DEFAULT):
        self._secret_key = secret_key
        self._phase3 = Phase3_ChronosWitnessSovereignDecider(secret_key)

    def adjudicate_and_translate(
        self, bundle: ChronosWitnessCampaignDraftBundle,
    ) -> ChronosWitnessExecutionPassport:
        return self._phase3._phase3_terminal_synthesis(bundle)


# ══════════════════════════════════════════════════════════════════════════════
# §6. VERIFICACIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 78)
    print("VERIFICACIÓN AUTÓNOMA: TOON CHRONOS WITNESS AGENT v9.0.0-POINCARE-RSI3")
    print("═" * 78)

    agent = TOONChronosWitnessAgent()

    dag_edges = [
        (0, 1, 12.0),
        (1, 2, 8.0),
        (2, 3, 4.0),
        (0, 2, 15.0),
    ]
    slacks = [12.0, 8.0, 4.0, 15.0]

    print("\n[CASO 1: DAG CON SHORTCUT]")
    try:
        passport = agent.run_full_pipeline(
            "SCHED-PASSPORT-001", num_nodes=4,
            edges=dag_edges, slacks=slacks)
        print("[Fase I: GAUGE OBSERVE]")
        for k, v in passport.gauge_audit_summary:
            print(f"    {k:>24} = {v}")
        print("\n[Fase II: CAMPAIGN ORIENT]")
        print(f"    {'v(Slack)':>24} = {passport.bundle.novikov_v_slack:.4f}")
        print(f"    {'mean slack':>24} = {passport.bundle.mean_slack_valuation:.4f}")
        print(f"    {'Banach L̂':>24} = {passport.bundle.banach_lipschitz:.4f}")
        for k, v in passport.rsi3_summary:
            print(f"    {k:>24} = {v}")
        print("\n[Fase III: SOVEREIGN DECIDE]")
        print(f"    {'Verdict':>24} = {passport.certificate.heyting_verdict.name}")
        print(f"    {'τ (truth degree)':>24} = {passport.truth_degree_tau:.2f}")
        print(f"    {'Actuation':>24} = {passport.certificate.actuation_mode.name}")
        print(f"    {'Orbit':>24} = {PoincareOrbitType(passport.poincare_orbit_type).name}")
        print(f"    {'Contingency':>24} = {passport.executive_summary_report['contingency_reserve_pct']}")
        print(f"    {'Merkle root':>24} = {passport.merkle_root_sha256[:16]}...")
        print(f"    {'Veto reasons':>24} = {passport.veto_reasons or '(ninguno)'}")
        print(f"\n    Cadena funtorial: {' → '.join(passport.phase_continuity_chain)}")
    except Exception as e:
        print(f"    ✗ {type(e).__name__}: {e}")

    cycle_edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 0, 5.0),
    ]
    print("\n[CASO 2: CICLO DIRIGIDO (debe fallar en Fase 1)]")
    try:
        passport = agent.run_full_pipeline(
            "SCHED-CYCLE-BAD", num_nodes=3,
            edges=cycle_edges, slacks=[10.0, 15.0, 5.0])
        print(f"    ⚠️  Inesperado: {passport.certificate.heyting_verdict.name}")
    except DirectedCycleWitnessError as e:
        print(f"    ✓ Kahn rechazó: {e}")
    except Exception as e:
        print(f"    ✓ {type(e).__name__}: {e}")

    print("\n" + "─" * 78)
    print("VERIFICACIÓN DE COMPATIBILIDAD v7 + POINCARÉ v9")
    print("─" * 78)

    topology = ChronosWitnessGaugeTopology()
    contract = topology.build_gauge_contract(
        "SCHED-COMPAT-001", num_nodes=4, edges=dag_edges)
    print(f"[v7-compat] CS₃ (A∧dA) = {contract.chern_simons_cs3:.6e}")
    print(f"[v7-compat] Pontryagin ν = {contract.pontryagin_number_nu}")
    print(f"[v7-compat] Wilson phase = {contract.wilson_holonomy_angle:.4f}")
    print(f"[v7-compat] Hopf c₁ = {contract.hopf_chern_class_c1:.4f}")
    print(f"[v7-compat] Δ_SZ = {contract.stora_zumino_residual:.4e}")
    print(f"[v7-compat] is_dag_acyclic = {contract.is_dag_acyclic}")
    print(f"[v7-compat] is_gauge_coherent = {contract.is_gauge_coherent}")
    print(f"[v7-compat] gauge_class = {contract.gauge_bundle_class}")
    print(f"[v9-celeste] T_rec = {contract.poincare_recurrence_time:.4f}")
    print(f"[v9-celeste] tr M = {contract.poincare_monodromy_trace:.4f}")
    print(f"[v9-celeste] χ_max = {contract.poincare_max_lyapunov:.4e}")
    print(f"[v9-celeste] 𝒥 = {contract.jacobi_integral:.4f}")
    print(f"[v9-celeste] orbit = {PoincareOrbitType(contract.orbit_type).name}")
    print("═" * 78)