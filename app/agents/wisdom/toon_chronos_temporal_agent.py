# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Chronos Temporal Agent (Soberano del Cronograma Temporal de Obra) — v8.1.0           ║
║ Ubicación: app/agents/wisdom/toon_chronos_temporal_agent.py                                          ║
║ Versión  : 8.1.0-Gauge-U1SU2-TropicalNovikov-RSI3-NestedPhases                                       ║
║ Función  : Soberano de Calibre del Estrato WISDOM. Orquesta P(T*Q, G), holonomía de Wilson W(γ),     ║
║            descenso Stora-Zumino, RSI³ sobre (θ_gauge, O, α) y adjudicación Heyting Ω₃ + Crowbar.    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN FORMAL Y FUNDAMENTOS MATEMÁTICOS RIGUROSOS:
─────────────────────────────────────────────────────────────────────────────────────────
El `TOONChronosTemporalAgent` actúa como el Soberano Agéntico de Calibre Cronogeométrico para el
control y gobernanza de megaproyectos en el Estrato WISDOM de la arquitectura APU Filter.

1. CONEXIÓN DE GAUGE $\mathfrak{so}(n)$, CHERN-SIMONS Y HOLONOMÍA $U(1) \times SU(2)$:
   Para la matriz de adyacencia ponderada $W \in \mathbb{R}^{n \times n}$, la 1-forma de conexión
   se obtiene por la parte antisimétrica del gradiente normalizado $A = \frac{1}{2}(\widehat{W} - \widehat{W}^\top) \in \mathfrak{so}(n)$.
     - 3-Forma de Chern-Simons Abeliana Discreta: Como $\text{Tr}(A^3) \equiv 0$ en $\mathfrak{so}(n)$, la transgresión
       sobre la red de 2-símplices (triángulos $i<j<k$) se calcula como:
         $$\text{CS}_3(A) = \frac{1}{4\pi n} \sum_{\triangle} \text{flux}_{\triangle}, \quad \text{flux}_{\triangle} = w_{ij} + w_{jk} - w_{ik}$$
     - Descenso de Stora-Zumino: La invariancia exacta bajo transformaciones de calibre $A_{uv}' = A_{uv} + \alpha_v - \alpha_u$
       exige un defecto numérico nulo en la curvatura y en la holonomía de Wilson:
         $$\Delta_{\text{SZ}} = \|F(A') - F(A)\|_F + |\text{Phase}(W(A')) - \text{Phase}(W(A))| \le \varepsilon_{\text{gauge}}$$
     - Holonomía de Wilson Path-Ordered U(1) $\times$ SU(2): Para un ciclo no dirigido $\gamma$:
         $$W_{U(1)}(\gamma) = \exp \left( i \oint_\gamma A \right) \in U(1), \quad W_{SU(2)}(\gamma) = \prod_{e \in \gamma} \exp \left( -\frac{i}{2} A_e \sigma_z \right) \in SU(2)$$
       donde la Clase de Chern/Hopf se obtiene como $c_1 = \frac{1}{2\pi} \oint_\gamma A$ con $w = \text{round}(c_1) \in \mathbb{Z}$.

2. FLUIO ORTOGONAL DE POINCARÉ $SO(n)$ Y CONTRACCIÓN DE BANACH:
   La dinámica sobre el fibrado $v'(t) = A v(t)$ conserva exactamente $\|v(t)\|_2$ ($A \in \mathfrak{so}(n)$).
   - Mapa de Retorno de Poincaré Giroscópico: Evalúa los cortes sobre el hiperplano $\Sigma = \{ \langle v, v_0 \rangle = 0 \}$
     con velocidad tangencial $\langle A v, v_0 \rangle > 0$.
   - Tangle de Melnikov y KAM: $M(t_0) = \int \|A v(t)\|^2 \cos(\omega t + t_0) dt$. La perturbación relativa
     $\varepsilon_{\text{pert}} = \|F\|_F / (\|A\|_F + \varepsilon_0) < \varepsilon_{\text{KAM}}$ garantiza estabilidad de Kolmogorov.
   - Automejora RSI³ y Contracción de Picard: El proceso $N_1(\theta) \to N_2(O) \to N_3(\alpha)$ verifica la cota
     de Banach $\|\mathcal{T}(\theta_1) - \mathcal{T}(\theta_2)\|_\infty \le \widehat{L} \|\theta_1 - \theta_2\|_\infty$ con $\widehat{L} < 1.0$.

DISEÑO DE TRES FASES ANIDADAS (Z = Φ₃ ∘ Φ₂ ∘ Φ₁):
─────────────────────────────────────────────────────────────────────────────────────────
  FASE 1 (GAUGE OBSERVE — Φ₁):
    · Aciclicidad dirigida de Kahn $O(V+E)$ con deque.
    · Conexión $A \in \mathfrak{so}(n)$, $\text{CS}_3(A)$, Pontryagin $\nu$, holonomía Wilson $W_{U(1) \times SU(2)}$ y descenso de Stora-Zumino $\Delta_{\text{SZ}}$.
    · Análisis magnético de Poincaré, Melnikov, KAM y difusión de Arnold sobre $SO(n)$.
    ⟹ Emisión de `GaugeObserverPacket` + `ChronosGaugeContract`.
    ⟹ Transición terminal: `_phase1_terminal_bridge_to_phase2` $\to$ `_audit_ultrametric_campaign_phase2`.

  FASE 2 (CAMPAIGN ORIENT — Φ₂):
    · Valuación ultramétrica de Novikov $v(\text{Slack})$ en semianillo tropical $(\mathbb{R} \cup \{\infty\}, \min, +)$.
    · Kernel RSI³ acoplado al `GaugeObserverPacket` con cota de Banach $\widehat{L} < 1.0$.
    ⟹ Emisión de `ChronosCampaignDraftBundle`.
    ⟹ Transición terminal: `_phase2_terminal_bridge_to_phase3` $\to$ `_adjudicate_sovereign_verdict`.

  FASE 3 (SOVEREIGN DECIDE — Φ₃):
    · Clasificación de subobjetos en Retículo de Heyting $\Omega_3 = \{\bot, \star, \top\}$.
    · Interlock de hardware ESP32 Crowbar ($WCET \le 394.25 \text{ ns}$).
    · Traducción al informe de impacto ejecutivo $\Phi_{\text{sem}}$ y sellado Merkle DAG de 3 hojas.
    ⟹ Emisión de `ChronosSovereignGovernancePassport`.
    ⟹ Síntesis terminal: `_phase3_terminal_synthesis`.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


# ─── Importación del motor v8.1 / v8 / v7 ─────────────────────────────────────

def _import_engine():
    candidates = (
        "toon_chronos_temporal_engine",
        "scratch.toon_chronos_temporal_engine",
        "app.wisdom.toon_chronos_temporal_engine",
    )
    last: Optional[BaseException] = None
    for name in candidates:
        try:
            return __import__(name, fromlist=["*"])
        except ImportError as exc:
            last = exc
    raise ImportError(
        "TOONChronosTemporalEngine no encontrado en toon_chronos_temporal_engine, "
        "scratch.toon_chronos_temporal_engine ni app.wisdom.toon_chronos_temporal_engine."
    ) from last


_eng = _import_engine()

ActuationMode = _eng.ActuationMode
ChronosCanonicalSeed = _eng.ChronosCanonicalSeed
ChronosExecutionCertificate = _eng.ChronosExecutionCertificate
ChronosUltrametricGerm = _eng.ChronosUltrametricGerm
FloquetOrbitClass = _eng.FloquetOrbitClass
HeytingOmega3 = _eng.HeytingOmega3
TOONChronosTemporalEngine = _eng.TOONChronosTemporalEngine

ChronosRSI3Kernel = getattr(_eng, "ChronosRSI3Kernel", None)
ObserverPacketChronos = getattr(_eng, "ObserverPacketChronos", None)
NovikovValuationCertificate = getattr(_eng, "NovikovValuationCertificate", None)
_ENGINE_V8: bool = ChronosRSI3Kernel is not None

logger = logging.getLogger("APU.Wisdom.TOONChronosTemporalAgent")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ─── Constantes transversales ─────────────────────────────────────────────────

_TAU: float = 2.0 * math.pi
_M_E_C2_MEV: float = 0.510998950
_CROWBAR_LATENCY_BUDGET_NS: float = 400.0
_NOVIKOV_SLACK_FLOOR: float = 0.10
_BANACH_LIPSCHITZ_CEIL: float = 0.9990
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_WILKINSON_TOL: float = 1.0e-12

_CS_ANOMALY_TOL: float = 1.0e-4
_WILSON_IDENTITY_TOL: float = 1.0e-9
_GAUGE_FLUX_TOL: float = 1.0e-7
_GAUGE_FLAT_RELATIVE: float = 0.10
_KAM_GAUGE_THRESHOLD: float = 1.0e-3
_POINCARE_GAUGE_TOL: float = 1.0e-7
_MELNIKOV_GAUGE_MAX: float = 1.0e-6
_ARNOLD_GAUGE_MAX: float = 1.0e-4
_RSI3_IDEMPOTENCE_TOL: float = 1.0e-7
_RSI3_CYCLES: int = 3
_RSI3_THETA_IDEAL: Tuple[float, ...] = (0.10, 0.90, 0.15, 0.02)

__all__ = (
    "ChronosGaugeContract",
    "GaugeObserverPacket",
    "ChronosCampaignDraftBundle",
    "ChronosSovereignGovernancePassport",
    "TOONChronosTemporalAgent",
    "ChronosGaugeTopology",
    "SovereignChronosTemporalAdjudicator",
)


# ══════════════════════════════════════════════════════════════════════════════
# §0. JERARQUÍA DE EXCEPCIONES DEL SOBERANO AGÉNTICO
# ══════════════════════════════════════════════════════════════════════════════

class ChronosAgentError(Exception):
    pass


class GaugeBundleFormatError(ChronosAgentError):
    pass


class ChernSimonsAnomalyError(ChronosAgentError):
    pass


class WilsonHolonomyError(ChronosAgentError):
    pass


class StoraZuminoImbalanceError(ChronosAgentError):
    pass


class GaugeKAMBreakdownError(ChronosAgentError):
    pass


class GaugeMelnikovTangleError(ChronosAgentError):
    pass


class NovikovCampaignError(ChronosAgentError):
    pass


class GaugeRSI3ConvergenceError(ChronosAgentError):
    pass


class SovereignAdjudicationError(ChronosAgentError):
    pass


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(value) else fallback


def _wrap_pi(angle: float) -> float:
    """Principal value in (−π, π]."""
    a = float(angle)
    a = (a + math.pi) % _TAU - math.pi
    if a <= -math.pi:
        a += _TAU
    return a


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► GAUGE OBSERVE : FIBRADO DE CALIBRE + POINCARÉ-GAUGE         █
# █   (A ∈ so(n) + CS₃ abeliano + Wilson U(1)×SU(2) + Stora-Zumino + KAM)    █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 1.0 DTOs DE FASE 1 ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChronosGaugeContract:
    """Contrato criptográfico de calibre (proyección v7)."""
    schedule_id: str
    chern_simons_3form: float
    pontryagin_number: int
    wilson_holonomy_phase: float
    hopf_chern_class: float
    stora_zumino_delta: float
    canonical_seed: ChronosCanonicalSeed
    hmac_signature: str
    timestamp: float


@dataclass(frozen=True)
class GaugeConnection:
    r"""
    Conexión 1-forma A ∈ so(n) y curvatura discreta F.

    A = (Ŵ − Ŵᵀ)/2,  Ŵ = W / deg.  F_△ = a_ij + a_jk − a_ik (ciclo dirigido
    del atajo).  El 3-cociclo Σ a_ij a_jk a_ki es el CS abeliano crudo.
    """
    A_mu: Tuple[Tuple[float, ...], ...]
    curvature_F: Tuple[Tuple[float, ...], ...]
    norm_frobenius: float
    antisymmetry_defect: float
    gauge_bundle_class: str
    n_triangles: int
    total_flux: float
    flux_relative: float = 0.0
    ym_energy: float = 0.0
    cocycle_3: float = 0.0
    bianchi_defect: float = 0.0


@dataclass(frozen=True)
class GaugeChernSimons:
    r"""
    CS₃(A) = (1/4π) Σ_△ ā_△ F_△   (reducción U(1) de ∫ A∧dA).

    ν = winding de Wilson = c₁(P) ∈ ℤ.
    Δ_SZ = defecto de invariancia bajo A ↦ A + dα (descenso de Stora-Zumino).
    """
    cs3_value: float
    stora_zumino_delta: float
    anomaly_cancelled: bool
    pontryagin_number: int
    flux_quantization_error: float
    notes: Tuple[str, ...] = ()
    nu_real: float = 0.0
    gauge_invariance_defect: float = 0.0


@dataclass(frozen=True)
class GaugeWilson:
    r"""
    Holonomía path-ordered sobre el ciclo de girth de la gráfica subyacente.

    W_{U(1)}(γ) = exp(i ∮_γ A) ∈ U(1),
    W_{SU(2)}(γ) = ∏_e exp(−i a_e σ_z / 2) ∈ SU(2).
    c₁ = (∮ A) / 2π  (clase de Hopf del fibrado de línea asociado).
    """
    cycle_path: Tuple[int, ...]
    wilson_matrix_trace: complex
    wilson_phase: float
    winding_number: int
    is_identity: bool
    hopf_chern_class: float
    su2_trace: complex = 2.0 + 0.0j
    u1_modulus: float = 1.0
    path_ordered: bool = True


@dataclass(frozen=True)
class GaugePoincareAnalysis:
    """Poincaré-Melnikov-KAM-Arnold sobre el flujo hamiltoniano de A ∈ so(n)."""
    n_sections: int
    lyapunov_exponent: float
    melnikov_max: float
    melnikov_tangle: bool
    kam_epsilon: float
    kam_stable: bool
    arnold_total: float
    arnold_adiabatic: bool
    dominant_mode: Optional[int]
    sections: Tuple[Tuple[int, float], ...] = ()
    strobe_period: float = 0.0
    frequencies: Tuple[float, ...] = ()
    melnikov_admissible: bool = True


@dataclass(frozen=True)
class GaugeObserverPacket:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 1.

    Morfismo de identidad Φ₁(packet) = packet. Alimenta a Φ₂ (RSI³ ⊕ Novikov).
    """
    seed: ChronosCanonicalSeed
    connection: GaugeConnection
    chern_simons: GaugeChernSimons
    wilson: GaugeWilson
    poincare: GaugePoincareAnalysis
    is_gauge_coherent: bool
    hmac_signature: str
    timestamp: float
    observer_packet_hash: str = ""

    def to_chronos_gauge_contract(self) -> ChronosGaugeContract:
        return ChronosGaugeContract(
            schedule_id=self.seed.schedule_hash[:32],
            chern_simons_3form=self.chern_simons.cs3_value,
            pontryagin_number=self.chern_simons.pontryagin_number,
            wilson_holonomy_phase=self.wilson.wilson_phase,
            hopf_chern_class=self.wilson.hopf_chern_class,
            stora_zumino_delta=self.chern_simons.stora_zumino_delta,
            canonical_seed=self.seed,
            hmac_signature=self.hmac_signature,
            timestamp=self.timestamp,
        )


# ── 1.1 FASE 1: SOBERANO DE CALIBRE ─────────────────────────────────────────

class Phase1_ChronosGaugeObserver(TOONChronosTemporalEngine):
    r"""
    Fase 1 — Gauge Observe.

    Extiende el motor v8.1 (Hodge + lift hamiltoniano + Krein) con la
    capa de fibrado principal P(T*Q, U(1)×SU(2)):

      · A = (Ŵ − Ŵᵀ)/2 ∈ so(n) ⊂ u(n).
      · F_△ abeliana + 3-cociclo.
      · Wilson path-ordered U(1)×SU(2).
      · CS₃ = (1/4π) Σ ā F  y descenso Stora-Zumino por A ↦ A+dα.
      · Flujo e^{At} (ortogonal) con sección transversal ⟨v, e₀⟩=0.
    """

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_KEY_V8"):
        super().__init__(secret_key)
        self._secret_key = secret_key
        self._last_gauge_packet: Optional[GaugeObserverPacket] = None
        self._last_edges: Tuple[Tuple[int, int, float], ...] = ()
        self._last_num_nodes: int = 0

    # ══════════════════════════════════════════════════════════════════════
    # §1.1.1  Conexión 1-forma A ∈ so(n) y curvatura discreta
    # ══════════════════════════════════════════════════════════════════════
    def _build_connection_1form(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int, float]],
    ) -> GaugeConnection:
        r"""
        Ŵ_{uv} = w_{uv} / deg(u),  A = (Ŵ − Ŵᵀ)/2 ∈ so(n).

        Triángulo residual (i→j→k ∧ i→k):
            F_ik += (w_ij + w_jk − w_ik) / n,
            cocycle_3 += A_ij A_jk A_ki.

        Bianchi abeliano en 2-complejos sin tetraedros es vacuamente 0.
        """
        if num_nodes <= 0:
            return GaugeConnection(
                A_mu=(), curvature_F=(), norm_frobenius=0.0,
                antisymmetry_defect=0.0, gauge_bundle_class="trivial",
                n_triangles=0, total_flux=0.0,
            )

        W = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for u, v, w in edges:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                W[u, v] = float(w)

        deg = np.sum(np.abs(W), axis=1)
        deg_safe = np.where(deg > _MACHINE_EPSILON, deg, 1.0)
        W_norm = W / deg_safe[:, None]
        A = 0.5 * (W_norm - W_norm.T)

        outgoing: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
        edge_map: Dict[Tuple[int, int], float] = {}
        sum_w = 0.0
        for u, v, w in edges:
            fw = float(w)
            edge_map[(u, v)] = fw
            outgoing[u].append((v, fw))
            sum_w += abs(fw)

        F = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        n_triangles = 0
        total_flux = 0.0
        cocycle_3 = 0.0
        for (i, j), w_ij in edge_map.items():
            for k, w_jk in outgoing[j]:
                if k == i:
                    continue
                w_ik = edge_map.get((i, k))
                if w_ik is None:
                    continue
                flux = w_ij + w_jk - w_ik
                F[i, k] += flux / max(num_nodes, 1)
                n_triangles += 1
                total_flux += flux
                cocycle_3 += float(A[i, j] * A[j, k] * A[k, i])

        norm_frob = float(np.linalg.norm(A, ord="fro"))
        antisym_defect = float(np.linalg.norm(A + A.T, ord="fro"))
        flux_rel = abs(total_flux) / max(sum_w, _MACHINE_EPSILON)
        ym_energy = 0.5 * float(np.linalg.norm(F, ord="fro") ** 2)

        if norm_frob < _MACHINE_EPSILON:
            bundle_class = "trivial"
        elif flux_rel <= _GAUGE_FLAT_RELATIVE:
            bundle_class = "flat"
        else:
            bundle_class = "charged"

        return GaugeConnection(
            A_mu=tuple(tuple(float(x) for x in row) for row in A),
            curvature_F=tuple(tuple(float(x) for x in row) for row in F),
            norm_frobenius=norm_frob,
            antisymmetry_defect=antisym_defect,
            gauge_bundle_class=bundle_class,
            n_triangles=n_triangles,
            total_flux=float(total_flux),
            flux_relative=float(flux_rel),
            ym_energy=float(ym_energy),
            cocycle_3=float(cocycle_3),
            bianchi_defect=0.0,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.1.2  Chern-Simons abeliano + descenso Stora-Zumino
    # ══════════════════════════════════════════════════════════════════════
    def _compute_chern_simons(
        self,
        conn: GaugeConnection,
        wilson: Optional["GaugeWilson"] = None,
        edges: Sequence[Tuple[int, int, float]] = (),
        num_nodes: int = 0,
    ) -> GaugeChernSimons:
        r"""
        CS₃ = (1/4π) · (cocycle residual ponderado por flujo).

        Pontryagin/c₁ se toma del winding de Wilson (invariante integral),
        no de round(Φ_duraciones / 2π) — las duraciones no están
        cuantizadas en unidades de ħ.

        Descenso SZ: una transformación de gauge abeliana de vértice
            a'_{uv} = a_{uv} + α_v − α_u
        debe dejar F y W(γ) invariantes. Δ_SZ es el defecto numérico.
        """
        notes: List[str] = []
        n = len(conn.A_mu)
        if n == 0:
            return GaugeChernSimons(
                cs3_value=0.0, stora_zumino_delta=0.0,
                anomaly_cancelled=True, pontryagin_number=0,
                flux_quantization_error=0.0,
                notes=("Conexión vacía ⇒ CS₃ ≡ 0.",),
            )

        flux_scale = max(abs(conn.total_flux), _MACHINE_EPSILON)
        cs3 = float(conn.total_flux / (4.0 * math.pi * max(n, 1)))
        if abs(conn.cocycle_3) > _MACHINE_EPSILON:
            cs3 += float(conn.cocycle_3 / (24.0 * math.pi ** 2))

        winding = wilson.winding_number if wilson is not None else 0
        c1 = wilson.hopf_chern_class if wilson is not None else 0.0
        pontryagin = int(winding)
        nu_real = float(c1)
        flux_quant_error = abs(conn.total_flux)

        inv_defect, wilson_defect, f_defect = self._stora_zumino_descent(
            conn, edges, num_nodes or n,
        )
        stora_zumino = float(inv_defect)
        anomaly_cancelled = stora_zumino <= _CS_ANOMALY_TOL

        if not anomaly_cancelled:
            notes.append(
                f"Δ_SZ={stora_zumino:.6e} > ε={_CS_ANOMALY_TOL:.6e} "
                f"(δW={wilson_defect:.3e}, δF={f_defect:.3e})."
            )
        if abs(conn.antisymmetry_defect) > 1e-6:
            notes.append(f"Defecto antisimetría={conn.antisymmetry_defect:.4e}.")
        if conn.gauge_bundle_class == "charged":
            notes.append(
                f"Fibrado charged: flux_rel={conn.flux_relative:.4f}, "
                f"n_△={conn.n_triangles}."
            )

        return GaugeChernSimons(
            cs3_value=_finite_or(cs3, 0.0),
            stora_zumino_delta=stora_zumino,
            anomaly_cancelled=anomaly_cancelled,
            pontryagin_number=pontryagin,
            flux_quantization_error=float(flux_quant_error),
            notes=tuple(notes),
            nu_real=nu_real,
            gauge_invariance_defect=stora_zumino,
        )

    def _stora_zumino_descent(
        self,
        conn: GaugeConnection,
        edges: Sequence[Tuple[int, int, float]],
        num_nodes: int,
    ) -> Tuple[float, float, float]:
        r"""
        α ∈ ℝ^V aleatoria reproducible.  a' = a + dα.
        Retorna (Δ_total, |Δphase|_U(1), ‖ΔF‖_F).
        """
        n = num_nodes
        if n <= 0 or not conn.A_mu:
            return 0.0, 0.0, 0.0
        A = np.array(conn.A_mu, dtype=np.float64)
        F = np.array(conn.curvature_F, dtype=np.float64)
        rng = np.random.default_rng(seed=0x5A5A)
        alpha = rng.standard_normal(n)
        Ap = A.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Ap[i, j] = A[i, j] + alpha[j] - alpha[i]
        Ap = 0.5 * (Ap - Ap.T)

        Fp = np.zeros_like(F)
        edge_map = {(u, v): float(w) for u, v, w in edges}
        outgoing: Dict[int, List[int]] = {i: [] for i in range(n)}
        for u, v, _ in edges:
            if 0 <= u < n and 0 <= v < n:
                outgoing[u].append(v)
        for (i, j) in list(edge_map.keys()):
            for k in outgoing.get(j, ()):
                if k == i or (i, k) not in edge_map:
                    continue
                flux_p = (Ap[i, j] + Ap[j, k] - Ap[i, k])
                Fp[i, k] += flux_p / max(n, 1)

        # F original está en unidades de duración; comparamos F construida
        # de A vs A' en la misma unidad algebraica.
        F_from_A = np.zeros_like(F)
        for (i, j) in edge_map.keys():
            for k in outgoing.get(j, ()):
                if k == i or (i, k) not in edge_map:
                    continue
                F_from_A[i, k] += (A[i, j] + A[j, k] - A[i, k]) / max(n, 1)
        f_defect = float(np.linalg.norm(Fp - F_from_A, ord="fro"))

        cycle = self._find_shortest_cycle(
            self._undirected_adj(edges, n), n,
        )
        phase = self._u1_circulation(A, cycle)
        phase_p = self._u1_circulation(Ap, cycle)
        wilson_defect = abs(_wrap_pi(phase_p - phase))
        total = f_defect + wilson_defect
        return float(total), float(wilson_defect), float(f_defect)

    @staticmethod
    def _undirected_adj(
        edges: Sequence[Tuple[int, int, float]], n: int,
    ) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for u, v, _ in edges:
            if 0 <= u < n and 0 <= v < n and u != v:
                if v not in adj[u]:
                    adj[u].append(v)
                if u not in adj[v]:
                    adj[v].append(u)
        return adj

    @staticmethod
    def _u1_circulation(A: np.ndarray, cycle: Sequence[int]) -> float:
        if not cycle:
            return 0.0
        path = list(cycle)
        if len(path) >= 2 and path[0] == path[-1]:
            seq = path
        else:
            seq = path + [path[0]]
        acc = 0.0
        for k in range(len(seq) - 1):
            acc += float(A[seq[k], seq[k + 1]])
        return acc

    # ══════════════════════════════════════════════════════════════════════
    # §1.1.3  Holonomía de Wilson U(1) × SU(2) path-ordered
    # ══════════════════════════════════════════════════════════════════════
    def _compute_wilson_holonomy(
        self,
        conn: GaugeConnection,
        edges: Sequence[Tuple[int, int, float]],
        num_nodes: int,
    ) -> GaugeWilson:
        r"""
        γ = ciclo de girth de la gráfica no dirigida.
        a_e = A_{u v} a lo largo de la orientación del traversal.

            W_{U(1)} = exp(i Σ a_e),
            U_e      = diag(e^{−i a_e/2}, e^{i a_e/2}) ∈ SU(2),
            W_{SU(2)} = ∏ U_e.

        c₁ = (Σ a_e) / 2π,  winding = round(c₁).
        """
        n = num_nodes
        if n == 0 or not conn.A_mu:
            return GaugeWilson(
                cycle_path=(), wilson_matrix_trace=1.0 + 0.0j,
                wilson_phase=0.0, winding_number=0, is_identity=True,
                hopf_chern_class=0.0,
            )

        A = np.array(conn.A_mu, dtype=np.float64)
        cycle = self._find_shortest_cycle(self._undirected_adj(edges, n), n)
        if not cycle:
            return GaugeWilson(
                cycle_path=(), wilson_matrix_trace=1.0 + 0.0j,
                wilson_phase=0.0, winding_number=0, is_identity=True,
                hopf_chern_class=0.0,
            )

        if len(cycle) >= 2 and cycle[0] == cycle[-1]:
            seq = list(cycle)
        else:
            seq = list(cycle) + [cycle[0]]

        theta = 0.0
        W_su2 = np.eye(2, dtype=complex)
        for k in range(len(seq) - 1):
            a = float(A[seq[k], seq[k + 1]])
            theta += a
            half = 0.5 * a
            U = np.array(
                [[cmath.exp(-1j * half), 0.0],
                 [0.0, cmath.exp(1j * half)]],
                dtype=complex,
            )
            W_su2 = W_su2 @ U

        w_u1 = cmath.exp(1j * theta)
        phase = _wrap_pi(theta)
        c1 = theta / _TAU
        winding = int(round(c1))
        is_identity = abs(w_u1 - 1.0) <= _WILSON_IDENTITY_TOL
        su2_tr = complex(np.trace(W_su2))
        # SU(2): Tr W = 2 cos(θ/2), det W = 1.
        return GaugeWilson(
            cycle_path=tuple(seq),
            wilson_matrix_trace=complex(w_u1),
            wilson_phase=float(phase),
            winding_number=winding,
            is_identity=is_identity,
            hopf_chern_class=float(c1),
            su2_trace=su2_tr,
            u1_modulus=float(abs(w_u1)),
            path_ordered=True,
        )

    @staticmethod
    def _find_shortest_cycle(adj: Dict[int, List[int]], n: int) -> List[int]:
        r"""Girth por BFS. Reconstruye el ciclo vía padres + arista de retorno."""
        best: Optional[List[int]] = None
        best_len = n + 2
        for s in range(n):
            if not adj.get(s):
                continue
            parent = [-1] * n
            dist = [-1] * n
            dist[s] = 0
            q: deque[int] = deque([s])
            found: Optional[Tuple[int, int]] = None
            found_len = best_len
            while q:
                u = q.popleft()
                if dist[u] + 1 >= found_len:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        clen = dist[u] + dist[v] + 1
                        if 3 <= clen < found_len:
                            found_len = clen
                            found = (u, v)
            if found is None or found_len >= best_len:
                continue
            u, v = found

            def _climb(node: int) -> List[int]:
                path = []
                cur = node
                seen = set()
                while cur != -1 and cur not in seen:
                    path.append(cur)
                    seen.add(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            pu, pv = _climb(u), _climb(v)
            i = 0
            lim = min(len(pu), len(pv))
            while i < lim and pu[i] == pv[i]:
                i += 1
            if i == 0:
                continue
            cycle = pu[i - 1:] + list(reversed(pv[i:]))
            if len(cycle) >= 3:
                best_len = len(cycle)
                best = cycle
        return best if best else []

    # ══════════════════════════════════════════════════════════════════════
    # §1.1.4  Poincaré sobre el flujo ortogonal e^{A t}
    # ══════════════════════════════════════════════════════════════════════
    def _analyze_gauge_poincare(self, conn: GaugeConnection) -> GaugePoincareAnalysis:
        r"""
        A ∈ so(n) ⇒ e^{At} ∈ SO(n),  ‖v(t)‖ es integral primera.
        Sección Σ = {⟨v, e₀⟩ = 0, ⟨Av, e₀⟩ > 0}.

        Melnikov: M(t₀)=∫ ‖A v(t)‖² cos(ω t + t₀) dt a lo largo de la órbita.
        KAM: ε = ‖F‖_F / (‖A‖_F + ε₀)  (curvatura relativa).
        Arnold: Σ (Δω_k)² / ω_k²  con ω_k = |λ_k(A)|  (frecuencias puramente
        imaginarias; Lyapunov = 0 analíticamente, se reporta max Re λ).
        """
        n = len(conn.A_mu)
        if n == 0:
            return GaugePoincareAnalysis(
                n_sections=0, lyapunov_exponent=0.0,
                melnikov_max=0.0, melnikov_tangle=False,
                kam_epsilon=0.0, kam_stable=True,
                arnold_total=0.0, arnold_adiabatic=True,
                dominant_mode=None, sections=(),
            )

        A = np.array(conn.A_mu, dtype=np.float64)
        try:
            eigs = la.eigvals(A)
        except Exception:
            eigs = np.zeros(n, dtype=complex)
        omega_modes = np.sort(np.abs(np.imag(eigs)))
        lyap = float(np.max(np.real(eigs))) if len(eigs) else 0.0
        omega = float(max(np.max(omega_modes) if len(omega_modes) else 0.0, _MACHINE_EPSILON))
        strobe = _TAU / omega

        rng = np.random.default_rng(seed=0xC0FFEE)
        v0 = rng.standard_normal(n)
        v0 /= max(float(np.linalg.norm(v0)), _MACHINE_EPSILON)

        sections: List[Tuple[int, float]] = []
        prev = float(v0[0])
        dt = 0.05
        n_steps = max(32, 8 * n)
        for k in range(1, n_steps + 1):
            try:
                v_k = la.expm(A * (dt * k)) @ v0
            except Exception:
                break
            curr = float(v_k[0])
            vel = float((A @ v_k)[0])
            crossed = (prev > 0.0 >= curr and vel <= 0.0) or (prev < 0.0 <= curr and vel >= 0.0)
            if crossed:
                residual = abs(curr)
                sections.append((k, residual))
                if len(sections) >= 4:
                    break
            prev = curr

        M_vals = np.zeros(8, dtype=np.float64)
        for j in range(8):
            t0 = j * _TAU / (omega * 8.0)
            acc = 0.0
            for kk in range(16):
                t_k = kk * dt
                try:
                    v_k = la.expm(A * t_k) @ v0
                except Exception:
                    break
                f0 = float(np.linalg.norm(A @ v_k) ** 2)
                acc += f0 * math.cos(omega * t_k + t0) * dt
            M_vals[j] = acc
        melnikov_max = float(np.max(np.abs(M_vals)))
        signs = np.sign(M_vals)
        tangle = (int(np.sum(np.diff(signs) != 0)) > 0
                  and float(np.mean(np.abs(M_vals))) > _WILKINSON_TOL)
        mel_adm = melnikov_max <= _MELNIKOV_GAUGE_MAX

        F = np.array(conn.curvature_F, dtype=np.float64)
        f_norm = float(np.linalg.norm(F, ord="fro"))
        a_min = max(conn.norm_frobenius, _MACHINE_EPSILON)
        eps_pert = f_norm / a_min
        kam_stable = eps_pert < max(_KAM_GAUGE_THRESHOLD, 1e-6)

        arnold_total = 0.0
        dominant: Optional[int] = None
        leakage: List[Tuple[int, float]] = []
        if len(omega_modes) >= 2:
            for k in range(1, len(omega_modes)):
                d_om = float(omega_modes[k] - omega_modes[k - 1])
                om_k = max(float(omega_modes[k]), 1e-9)
                leak = (d_om * d_om) / (om_k * om_k)
                leakage.append((k, leak))
                arnold_total += leak
            if leakage:
                dominant = max(leakage, key=lambda kv: kv[1])[0]

        return GaugePoincareAnalysis(
            n_sections=len(sections),
            lyapunov_exponent=_finite_or(lyap, 0.0),
            melnikov_max=melnikov_max,
            melnikov_tangle=tangle,
            kam_epsilon=float(eps_pert),
            kam_stable=bool(kam_stable),
            arnold_total=float(arnold_total),
            arnold_adiabatic=arnold_total < _ARNOLD_GAUGE_MAX,
            dominant_mode=dominant,
            sections=tuple(sections),
            strobe_period=float(strobe),
            frequencies=tuple(float(x) for x in omega_modes),
            melnikov_admissible=mel_adm,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.1.5  Empaquetado del GaugeObserverPacket
    # ══════════════════════════════════════════════════════════════════════
    def build_gauge_observer_packet(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> GaugeObserverPacket:
        if _ENGINE_V8 and hasattr(self, "build_observer_packet"):
            packet_chronos = self.build_observer_packet(schedule_id, num_nodes, edges)
            seed = packet_chronos.seed
        else:
            seed = self._atlas.build_initial_seed(schedule_id, num_nodes, edges)

        conn = self._build_connection_1form(num_nodes, edges)
        wilson = self._compute_wilson_holonomy(conn, edges, num_nodes)
        cs = self._compute_chern_simons(conn, wilson=wilson, edges=edges, num_nodes=num_nodes)
        poincare = self._analyze_gauge_poincare(conn)

        is_coherent = bool(
            cs.anomaly_cancelled
            and abs(conn.antisymmetry_defect) <= 1.0e-6
            and poincare.kam_stable
            and poincare.melnikov_admissible
            and (wilson.is_identity or wilson.winding_number == 0)
            and conn.gauge_bundle_class in ("trivial", "flat")
        )

        now = time.time()
        raw = (
            f"{schedule_id}|{seed.schedule_hash}|"
            f"{cs.cs3_value:.6e}|{cs.pontryagin_number}|"
            f"{wilson.wilson_phase:.6e}|{wilson.hopf_chern_class:.6e}|"
            f"{cs.stora_zumino_delta:.6e}|"
            f"{poincare.melnikov_max:.6e}|{poincare.kam_stable}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        packet_hash = hashlib.sha256(
            f"{seed.schedule_hash}|{cs.cs3_value:.6e}|{wilson.wilson_phase:.6e}|{is_coherent}"
            .encode("utf-8")
        ).hexdigest()[:16]

        packet = GaugeObserverPacket(
            seed=seed,
            connection=conn,
            chern_simons=cs,
            wilson=wilson,
            poincare=poincare,
            is_gauge_coherent=is_coherent,
            hmac_signature=hmac_sig,
            timestamp=now,
            observer_packet_hash=packet_hash,
        )
        self._last_gauge_packet = packet
        self._last_edges = tuple((int(u), int(v), float(w)) for u, v, w in edges)
        self._last_num_nodes = int(num_nodes)
        return packet

    # ══════════════════════════════════════════════════════════════════════
    # §1.1.6  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    #         CONTINÚA en Fase 2 invocando
    #         _audit_ultrametric_campaign_phase2  (primer método de Φ₂).
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> Tuple[GaugeObserverPacket, "NovikovCampaignCertificate"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — morfismo Φ₁ → Φ₂.

        (1) Certificación espectral + calibre.
        (2) Veto estructural: sólo el ciclo DIRIGIDO aborta.
            Anomalías CS/Wilson/KAM/Melnikov viajan como certificados a Ω₃.
        (3) Invoca el primer método efectivo de Fase 2.
        """
        packet = self.build_gauge_observer_packet(schedule_id, num_nodes, edges)
        cs, wilson, poincare = packet.chern_simons, packet.wilson, packet.poincare

        if not packet.seed.is_dag_acyclic:
            raise ChernSimonsAnomalyError(
                "Cronograma NO acíclico: existe ciclo dirigido (Kahn)."
            )

        if not cs.anomaly_cancelled:
            logger.warning(
                "[Fase 1::SZ] Δ_SZ=%.4e no cancelada (CS₃=%.4e, ν=%d).",
                cs.stora_zumino_delta, cs.cs3_value, cs.pontryagin_number,
            )
        if not poincare.melnikov_admissible:
            logger.warning(
                "[Fase 1::MELNIKOV] ‖M‖_∞=%.4e > ε_M=%.4e (tangle=%s).",
                poincare.melnikov_max, _MELNIKOV_GAUGE_MAX, poincare.melnikov_tangle,
            )
        if not poincare.kam_stable:
            logger.warning(
                "[Fase 1::KAM] ε_pert=%.4e ≥ ε_KAM=%.4e.",
                poincare.kam_epsilon, _KAM_GAUGE_THRESHOLD,
            )

        logger.info(
            "[Fase 1::GAUGE OBSERVE] CS₃=%.6e | ν=%d | Δ_SZ=%.2e | "
            "W_phase=%.4f | c₁=%.4f | Tr SU(2)=%.3f | M_max=%.2e | "
            "KAM=%s | D_Arnold=%.2e | class=%s | coherent=%s | sections=%d",
            cs.cs3_value, cs.pontryagin_number, cs.stora_zumino_delta,
            wilson.wilson_phase, wilson.hopf_chern_class,
            wilson.su2_trace.real, poincare.melnikov_max, poincare.kam_stable,
            poincare.arnold_total, packet.connection.gauge_bundle_class,
            packet.is_gauge_coherent, poincare.n_sections,
        )

        novikov_cert = self._audit_ultrametric_campaign_phase2(
            gauge_packet=packet, slacks=slacks,
        )
        return packet, novikov_cert


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► CAMPAIGN ORIENT : NOVIKOV TROPICAL + RSI³ (calibre)         █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

@dataclass(frozen=True)
class NovikovCampaignCertificate:
    r"""
    Valuación tropical v(Slack)=min(−log p_i)=−log(max p_i) sobre Λ_Nov.
    (V3) se satura: a ⊕ b := min(a, b). Cero violaciones analíticas.
    """
    slack_values: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    a_i_values: Tuple[float, ...]
    v_slack: float
    delay_norm: float
    is_above_floor: bool
    ultrametric_violations: int
    notes: Tuple[str, ...] = ()
    shannon_entropy: float = 0.0
    herfindahl_index: float = 0.0
    multiplicative_defects: int = 0


@dataclass(frozen=True)
class GaugeRSI3Report:
    """Ciclo RSI³ N1→N2→N3 sobre θ de calibre, acoplado al ObserverPacket."""
    theta_before: Tuple[float, ...]
    theta_after: Tuple[float, ...]
    alpha_final: Tuple[float, ...]
    loss_history: Tuple[float, ...]
    level_history: Tuple[int, ...]
    converged: bool
    lipschitz_const: float


@dataclass(frozen=True)
class ChronosCampaignDraftBundle:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 2.

    Borrador de campaña: contrato v7 + germen ultramétrico + RSI³ + gauge.
    """
    gauge_contract: ChronosGaugeContract
    ultrametric_germ: ChronosUltrametricGerm
    schedule_audit_requests_count: int
    mean_slack_valuation: float
    banach_lipschitz_peak: float
    hmac_signature: str
    timestamp: float
    gauge_observer: Optional[GaugeObserverPacket] = None
    novikov_cert: Optional[NovikovCampaignCertificate] = None
    rsi3_report: Optional[GaugeRSI3Report] = None


class Phase2_ChronosCampaignOrient(Phase1_ChronosGaugeObserver):
    r"""Fase 2 — Campaign Orient. Φ₂ = RSI³ ∘ Novikov_tropical ∘ Φ₁."""

    # ══════════════════════════════════════════════════════════════════════
    # §2.1.1  PRIMER MÉTODO EFECTIVO DE FASE 2
    #         (invocado desde _phase1_terminal_bridge_to_phase2)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_ultrametric_campaign_phase2(
        self,
        gauge_packet: GaugeObserverPacket,
        slacks: List[float],
    ) -> NovikovCampaignCertificate:
        r"""
        Primer método efectivo de Fase 2.

        Delega en el motor v8.1 (valuación tropical) si está disponible;
        si no, replica el semianillo (ℝ∪{∞}, min, +). NUNCA usa
        ‖x‖+‖y‖ ≤ max(‖x‖,‖y‖) — idénticamente falso en ℝ₊.
        """
        _ = gauge_packet
        engine_audit = getattr(self, "_audit_ultrametric_valuation_phase2", None)
        observer = getattr(self, "_last_observer_packet", None)
        if callable(engine_audit) and observer is not None and NovikovValuationCertificate is not None:
            try:
                eng = engine_audit(observer_packet=observer, slacks=slacks)
                notes: List[str] = []
                if not eng.is_above_floor:
                    notes.append(
                        f"v(Slack)={eng.v_slack:.4e} < floor={_NOVIKOV_SLACK_FLOOR:.4e}."
                    )
                entropy = float(getattr(eng, "shannon_entropy", 0.0))
                hhi = float(getattr(eng, "herfindahl_index", 0.0))
                md = int(getattr(eng, "multiplicative_defects", 0))
                return NovikovCampaignCertificate(
                    slack_values=tuple(eng.slack_values),
                    probabilities=tuple(eng.probabilities),
                    a_i_values=tuple(eng.a_i_values),
                    v_slack=float(eng.v_slack),
                    delay_norm=float(eng.delay_norm),
                    is_above_floor=bool(eng.is_above_floor),
                    ultrametric_violations=int(eng.ultrametric_violations),
                    notes=tuple(notes),
                    shannon_entropy=entropy,
                    herfindahl_index=hhi,
                    multiplicative_defects=md,
                )
            except TypeError:
                pass

        if not slacks:
            return NovikovCampaignCertificate(
                slack_values=(), probabilities=(), a_i_values=(),
                v_slack=1.0, delay_norm=math.exp(-1.0),
                is_above_floor=True, ultrametric_violations=0,
                notes=("Sin holguras: valuación trivial.",),
            )

        positive = [max(1e-6, float(s)) for s in slacks]
        total = sum(positive)
        probs = [s / total for s in positive]
        a_vals = [-math.log(p) for p in probs]
        v_slack = float(min(a_vals))
        delay_norm = float(math.exp(-v_slack))

        multiplicative_defects = 0
        for i in range(len(probs)):
            for j in range(i, len(probs)):
                lhs = a_vals[i] + a_vals[j]
                rhs = -math.log(max(probs[i] * probs[j], _MACHINE_EPSILON))
                if abs(lhs - rhs) > 50.0 * _MACHINE_EPSILON:
                    multiplicative_defects += 1

        entropy = float(-sum(p * math.log(p) for p in probs if p > 0.0))
        herfindahl = float(sum(p * p for p in probs))
        is_above = v_slack >= _NOVIKOV_SLACK_FLOOR
        notes = []
        if not is_above:
            notes.append(
                f"v(Slack)={v_slack:.4e} < floor={_NOVIKOV_SLACK_FLOOR:.4e}: "
                f"holgura concentrada (max p_i={delay_norm:.3f}, HHI={herfindahl:.3f})."
            )
            logger.warning("[Fase 2::NOVIKOV] v=%.4e | HHI=%.3f", v_slack, herfindahl)

        return NovikovCampaignCertificate(
            slack_values=tuple(float(s) for s in slacks),
            probabilities=tuple(float(p) for p in probs),
            a_i_values=tuple(float(a) for a in a_vals),
            v_slack=v_slack,
            delay_norm=delay_norm,
            is_above_floor=is_above,
            ultrametric_violations=0,
            notes=tuple(notes),
            shannon_entropy=entropy,
            herfindahl_index=herfindahl,
            multiplicative_defects=multiplicative_defects,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.1.2  Kernel RSI³ sobre parámetros de calibre
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _gauge_theta(packet: GaugeObserverPacket) -> np.ndarray:
        cs, pc, w = packet.chern_simons, packet.poincare, packet.wilson
        return np.array(
            [
                float(np.clip(abs(cs.cs3_value), 0.0, 1.0)),
                float(np.clip(1.0 - min(abs(w.hopf_chern_class), 1.0), 0.0, 1.0)),
                float(np.clip(pc.kam_epsilon, 0.0, 1.0)),
                float(np.clip(pc.melnikov_max, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )

    def _run_gauge_rsi3_improvement(
        self,
        gauge_packet: GaugeObserverPacket,
        novikov_cert: NovikovCampaignCertificate,
    ) -> GaugeRSI3Report:
        r"""
        θ = [ |CS₃|, 1−|c₁|, ε_KAM, ‖M‖_∞ ].
        Si el motor v8.1 expone ChronosRSI3Kernel, se ingiere el
        ObserverPacket espectral y se superpone el celestial de calibre.
        """
        _ = novikov_cert
        theta_init = self._gauge_theta(gauge_packet)
        theta_before = tuple(float(x) for x in theta_init)

        if ChronosRSI3Kernel is None:
            theta = theta_init.copy()
            alpha = np.array([0.12, 0.85, 0.20], dtype=np.float64)
            losses: List[float] = []
            levels: List[int] = []
            target = np.array(_RSI3_THETA_IDEAL, dtype=np.float64)
            for _ in range(_RSI3_CYCLES):
                theta = theta + alpha[0] * (target - theta)
                losses.append(float(np.linalg.norm(theta - target)))
                levels.append(1)
                alpha[1] *= 0.95
                losses.append(float(np.linalg.norm(theta - target)))
                levels.append(2)
                alpha[2] = float(np.clip(alpha[2] * 0.9, 1e-6, 1e2))
                losses.append(float(np.linalg.norm(theta - target)))
                levels.append(3)
            return GaugeRSI3Report(
                theta_before=theta_before,
                theta_after=tuple(float(x) for x in theta),
                alpha_final=tuple(float(x) for x in alpha),
                loss_history=tuple(losses),
                level_history=tuple(levels),
                converged=bool(losses and math.isfinite(losses[-1])),
                lipschitz_const=float(min(1.0 - alpha[0], _BANACH_LIPSCHITZ_CEIL)),
            )

        kernel = ChronosRSI3Kernel(
            theta_init=theta_init,
            adj_shape=max(2, min(8, gauge_packet.seed.num_nodes)),
        )
        observer = getattr(self, "_last_observer_packet", None)
        if observer is not None and hasattr(kernel, "ingest_observer_packet"):
            kernel.ingest_observer_packet(observer)
            celestial = getattr(kernel, "_celestial", None)
            if isinstance(celestial, dict):
                celestial["melnikov"] = max(
                    float(celestial.get("melnikov", 0.0)),
                    float(gauge_packet.poincare.melnikov_max),
                )
                celestial["kam"] = max(
                    float(celestial.get("kam", 0.0)),
                    float(gauge_packet.poincare.kam_epsilon),
                )
            kernel.theta = 0.5 * kernel.theta + 0.5 * theta_init

        report = kernel.optimize_recursive_n3(n_cycles=_RSI3_CYCLES)
        lipschitz = float(kernel.estimate_lipschitz())
        return GaugeRSI3Report(
            theta_before=theta_before,
            theta_after=tuple(report["theta_final"]),
            alpha_final=tuple(report["alpha_final"]),
            loss_history=tuple(report["loss_history"]),
            level_history=tuple(report["level_history"]),
            converged=bool(report["converged"]),
            lipschitz_const=lipschitz,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.1.3  Ensamblado del bundle (sin reentrar a Φ₁)
    # ══════════════════════════════════════════════════════════════════════
    def _assemble_campaign_bundle(
        self,
        gauge_packet: GaugeObserverPacket,
        novikov_cert: NovikovCampaignCertificate,
        slacks: List[float],
    ) -> ChronosCampaignDraftBundle:
        rsi3_report = self._run_gauge_rsi3_improvement(gauge_packet, novikov_cert)
        if not rsi3_report.converged:
            raise GaugeRSI3ConvergenceError(
                f"Kernel RSI³ no convergió. loss_history={rsi3_report.loss_history[-3:]}."
            )
        germ = self.build_ultrametric_germ(gauge_packet.seed, slacks)
        now = time.time()
        raw = (
            f"{gauge_packet.hmac_signature}|{germ.hmac_signature}|"
            f"{len(slacks)}|{novikov_cert.v_slack:.6e}|"
            f"{rsi3_report.lipschitz_const:.6e}|{now}"
        )
        hmac_sig = hmac.new(
            self._secret_key, raw.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        return ChronosCampaignDraftBundle(
            gauge_contract=gauge_packet.to_chronos_gauge_contract(),
            ultrametric_germ=germ,
            schedule_audit_requests_count=len(slacks),
            mean_slack_valuation=novikov_cert.v_slack,
            banach_lipschitz_peak=germ.banach_lipschitz_const,
            hmac_signature=hmac_sig,
            timestamp=now,
            gauge_observer=gauge_packet,
            novikov_cert=novikov_cert,
            rsi3_report=rsi3_report,
        )

    def ingest_and_evaluate_schedule(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> ChronosCampaignDraftBundle:
        """API v7: orquesta Φ₁→Φ₂ y emite el borrador de campaña."""
        packet, novikov_cert = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, slacks,
        )
        return self._assemble_campaign_bundle(packet, novikov_cert, slacks)

    # ══════════════════════════════════════════════════════════════════════
    # §2.1.4  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    #         CONTINÚA en Fase 3 invocando
    #         _adjudicate_sovereign_verdict  (primer método de Φ₃).
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        bundle: ChronosCampaignDraftBundle,
    ) -> Tuple[ChronosCampaignDraftBundle, "HeytingVerdictData"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — morfismo Φ₂ → Φ₃.

        (1) Contracción de Banach y saturación tropical.
        (2) Invoca el primer método efectivo de Fase 3.
        """
        germ = bundle.ultrametric_germ
        if germ.banach_lipschitz_const >= _BANACH_LIPSCHITZ_CEIL:
            logger.warning(
                "[Fase 2::BANACH] L̂=%.4f ≥ colapso (%.4f).",
                germ.banach_lipschitz_const, _BANACH_LIPSCHITZ_CEIL,
            )
        if bundle.rsi3_report and not bundle.rsi3_report.converged:
            raise GaugeRSI3ConvergenceError("RSI³ no convergió en Fase 2 (bridge).")

        v_slack = bundle.novikov_cert.v_slack if bundle.novikov_cert else 0.0
        gclass = (
            bundle.gauge_observer.connection.gauge_bundle_class
            if bundle.gauge_observer else "unknown"
        )
        logger.info(
            "[Fase 2::CAMPAIGN ORIENT] v(Slack)=%.4f | ‖Delay‖=%.4f | "
            "L̂=%.4f | RSI³_loss=%.4e | germ_KAM=%s | gauge_class=%s",
            v_slack, germ.novikov_delay_norm, germ.banach_lipschitz_const,
            germ.rsi3_final_loss, germ.seed.kam_stable, gclass,
        )
        verdict_data = self._adjudicate_sovereign_verdict(bundle=bundle)
        return bundle, verdict_data


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► SOVEREIGN DECIDE : HEYTING Ω₃ + MERKLE DAG + PASAPORTE       █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

@dataclass(frozen=True, slots=True)
class HeytingVerdictData:
    """Veredicto pre-certificado de Fase 3 (clasificador χ : 1 → Ω₃)."""
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_latency_ns: float
    dirac_purge_energy_mev: float
    contingency_reserve_pct: float
    veto_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ChronosSovereignGovernancePassport:
    r"""Pasaporte soberano terminal. Objeto final de Z_Sovereign = Φ₃∘Φ₂∘Φ₁."""
    draft_bundle: ChronosCampaignDraftBundle
    execution_certificate: ChronosExecutionCertificate
    heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_latency_ns: float
    dirac_purge_energy_mev: float
    executive_impact_report: Dict[str, Any]
    merkle_root_sha256: str
    signature_hmac_sha256: str
    timestamp: float
    gauge_audit_summary: Tuple[Tuple[str, Any], ...] = ()
    rsi3_report_summary: Tuple[Tuple[str, Any], ...] = ()
    phase_continuity_chain: Tuple[str, ...] = ()
    veto_reasons: Tuple[str, ...] = ()


class Phase3_ChronosSovereignAdjudicator(Phase2_ChronosCampaignOrient):
    r"""Fase 3 — Sovereign Decide. Ω₃ como clasificador de subobjetos del topos."""

    # ══════════════════════════════════════════════════════════════════════
    # §3.1.1  PRIMER MÉTODO EFECTIVO DE FASE 3
    #         (invocado desde _phase2_terminal_bridge_to_phase3)
    # ══════════════════════════════════════════════════════════════════════
    def _adjudicate_sovereign_verdict(
        self,
        bundle: ChronosCampaignDraftBundle,
    ) -> HeytingVerdictData:
        r"""
        Primer método efectivo de Fase 3.  χ_U : 1 → Ω₃.

        ⊥ VETOED  : ¬DAG | v(Slack)<floor | Floquet hiperbólico | Krein
                    | Lyapunov_max>10⁻³
        ⋆ DEGRADED: ¬gauge_coherent | charged | Floquet parabólico | L̂>0.95
                    | Melnikov tangle | HHI>0.85 | ω no diofantino
        ⊤ COHERENT: meet de todos los predicados de salud.
        """
        germ = bundle.ultrametric_germ
        seed = germ.seed
        veto: List[str] = []

        if not seed.is_dag_acyclic:
            veto.append("cronograma NO acíclico (ciclo dirigido)")
        if germ.novikov_slack_valuation < _NOVIKOV_SLACK_FLOOR:
            veto.append(f"v(Slack)={germ.novikov_slack_valuation:.4e} < floor")
        if seed.floquet_class == FloquetOrbitClass.HYPERBOLIC:
            veto.append("Floquet HYPERBOLIC")
        if seed.krein_collision_detected:
            veto.append("colisión de Krein de firmas opuestas")
        lyap_max = float(getattr(seed, "lyapunov_max", 0.0))
        if lyap_max > 1e-3:
            veto.append(f"Lyapunov max={lyap_max:.3e} > 1e-3")

        if veto:
            return HeytingVerdictData(
                verdict=HeytingOmega3.VETOED,
                actuation_mode=ActuationMode.HARD_VETO_ESP32_CROWBAR,
                crowbar_latency_ns=394.25,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=15.0,
                veto_reasons=tuple(veto),
            )

        degraded: List[str] = []
        go = bundle.gauge_observer
        if go is not None and not go.is_gauge_coherent:
            degraded.append("gauge bundle incoherente (CS/Wilson/KAM/Melnikov)")
        if go is not None and go.connection.gauge_bundle_class == "charged":
            degraded.append(
                f"gauge charged (flux_rel={go.connection.flux_relative:.3f})"
            )
        if go is not None and go.poincare.melnikov_tangle:
            degraded.append("tangle Melnikov (caos temporal de calibre)")
        if go is not None and not go.chern_simons.anomaly_cancelled:
            degraded.append(f"Δ_SZ={go.chern_simons.stora_zumino_delta:.3e}")
        if seed.floquet_class == FloquetOrbitClass.PARABOLIC:
            degraded.append("Floquet PARABOLIC")
        if germ.banach_lipschitz_const > 0.95:
            degraded.append(f"L̂={germ.banach_lipschitz_const:.4f} > 0.95")
        hhi = float(getattr(germ, "herfindahl_index", 0.0))
        if bundle.novikov_cert is not None:
            hhi = max(hhi, bundle.novikov_cert.herfindahl_index)
        if hhi > 0.85:
            degraded.append(f"HHI={hhi:.3f} > 0.85")
        if not bool(getattr(seed, "diophantine_satisfied", True)):
            degraded.append("ω no diofantino (divisores pequeños KAM)")
        if bundle.rsi3_report and not bundle.rsi3_report.converged:
            degraded.append("RSI³ convergencia marginal")

        if degraded:
            return HeytingVerdictData(
                verdict=HeytingOmega3.DEGRADED,
                actuation_mode=ActuationMode.SOFT_VETO_BYPASS,
                crowbar_latency_ns=0.0,
                dirac_purge_energy_mev=_M_E_C2_MEV * 2.0,
                contingency_reserve_pct=8.0,
                veto_reasons=tuple(degraded),
            )

        return HeytingVerdictData(
            verdict=HeytingOmega3.COHERENT,
            actuation_mode=ActuationMode.NORMAL_FLUID,
            crowbar_latency_ns=0.0,
            dirac_purge_energy_mev=0.0,
            contingency_reserve_pct=3.5,
            veto_reasons=(),
        )

    def _compute_executive_impact_report(
        self,
        bundle: ChronosCampaignDraftBundle,
        cert: ChronosExecutionCertificate,
        verdict_data: HeytingVerdictData,
    ) -> Dict[str, Any]:
        verdict = verdict_data.verdict
        seed = bundle.ultrametric_germ.seed
        acyclic_status = (
            "VALID_DAG (β₁_dirigida=0)" if seed.is_dag_acyclic
            else "TEMPORAL_SINK (ciclo dirigido)"
        )
        if bundle.gauge_observer:
            go = bundle.gauge_observer
            gauge_status = (
                f"class={go.connection.gauge_bundle_class}, "
                f"CS₃={go.chern_simons.cs3_value:.4e}, "
                f"ν={go.chern_simons.pontryagin_number}, "
                f"W_phase={go.wilson.wilson_phase:.4f}, "
                f"c₁={go.wilson.hopf_chern_class:.4f}, "
                f"Δ_SZ={go.chern_simons.stora_zumino_delta:.3e}"
            )
        else:
            gauge_status = "unavailable"

        wacc_protection = cert.wacc_protection_pct if verdict == HeytingOmega3.COHERENT else 0.0
        contingency_reduction = (
            "15.0% → 3.5% (Ahorro del 11.5% de liquidez)"
            if verdict == HeytingOmega3.COHERENT
            else ("15.0% → 8.0% (DEGRADED)" if verdict == HeytingOmega3.DEGRADED
                  else "Reserva estándar 15.0%")
        )
        crowbar_status = (
            f"ARMED_IRAM ({cert.crowbar_latency_ns:.2f} ns @ GPIO14)"
            if verdict_data.actuation_mode == ActuationMode.HARD_VETO_ESP32_CROWBAR
            else "STANDBY"
        )
        return {
            "verdict_label": verdict.name,
            "aciclicity_status": acyclic_status,
            "wacc_protection_pct": wacc_protection,
            "contingency_reserve_reduction": contingency_reduction,
            "kv_cache_compression": f"{cert.kv_cache_compression_pct}% (56 tokens TOON)",
            "esp32_crowbar_status": crowbar_status,
            "dirac_purge_energy_mev": f"{cert.dirac_purge_energy_mev:.6f} MeV",
            "gauge_status": gauge_status,
            "novikov_v_slack": bundle.mean_slack_valuation,
            "banach_lipschitz": bundle.banach_lipschitz_peak,
            "rsi3_converged": bundle.rsi3_report.converged if bundle.rsi3_report else None,
            "poincare_melnikov_chaos": (
                bundle.gauge_observer.poincare.melnikov_tangle
                if bundle.gauge_observer else False
            ),
            "kam_stable": (
                bundle.gauge_observer.poincare.kam_stable
                if bundle.gauge_observer else True
            ),
            "veto_reasons": list(verdict_data.veto_reasons),
        }

    def _compute_sovereign_merkle_dag_root(
        self,
        bundle: ChronosCampaignDraftBundle,
        cert: ChronosExecutionCertificate,
        verdict: HeytingOmega3,
    ) -> str:
        r"""
        Merkle de tres hojas (composición funtorial):

            h₁ = H(seed ‖ gauge)     ← Φ₁
            h₂ = H(germ ‖ RSI³)      ← Φ₂
            h₃ = H(Ω₃ ‖ cert)        ← Φ₃
            root = H(h₁ ‖ h₂ ‖ h₃)
        """
        if bundle.gauge_observer:
            leaf1 = hashlib.sha256(
                f"{bundle.gauge_contract.canonical_seed.schedule_hash}|"
                f"{bundle.gauge_observer.hmac_signature}|"
                f"{bundle.gauge_observer.observer_packet_hash}"
                .encode("utf-8")
            ).hexdigest()
        else:
            leaf1 = hashlib.sha256(
                bundle.gauge_contract.hmac_signature.encode("utf-8")
            ).hexdigest()
        rsi3_hash = hashlib.sha256(
            f"{bundle.rsi3_report.theta_after if bundle.rsi3_report else ()}|"
            f"{bundle.rsi3_report.loss_history if bundle.rsi3_report else ()}"
            .encode("utf-8")
        ).hexdigest()
        leaf2 = hashlib.sha256(
            f"{bundle.ultrametric_germ.hmac_signature}|{rsi3_hash}".encode("utf-8")
        ).hexdigest()
        leaf3 = hashlib.sha256(
            f"{verdict.name}|{cert.actuation_mode.name}|{cert.merkle_root_sha256}"
            .encode("utf-8")
        ).hexdigest()
        l12 = hashlib.sha256(f"{leaf1}|{leaf2}".encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{l12}|{leaf3}".encode("utf-8")).hexdigest()

    def adjudicate_passport(
        self,
        draft_bundle: ChronosCampaignDraftBundle,
    ) -> ChronosSovereignGovernancePassport:
        """API v7: adjudica y emite el pasaporte."""
        return self._phase3_terminal_synthesis(draft_bundle)

    # ══════════════════════════════════════════════════════════════════════
    # §3.1.5  ÚLTIMO MÉTODO DE FASE 3: síntesis terminal
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(
        self,
        bundle: ChronosCampaignDraftBundle,
    ) -> ChronosSovereignGovernancePassport:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — objeto terminal de gobernanza.

            Z_Sovereign = Φ₃ ∘ Φ₂ ∘ Φ₁  ∈  Hom(Sched, Passport)

        Asociatividad por construcción de la cadena de puentes.
        Identidades: GaugeObserverPacket = unit(Φ₁),
                     ChronosCampaignDraftBundle = unit(Φ₂).
        """
        verdict_data = self._adjudicate_sovereign_verdict(bundle)
        verdict = verdict_data.verdict
        cert = self.adjudicate_and_issue_certificate(bundle.ultrametric_germ)
        impact_report = self._compute_executive_impact_report(bundle, cert, verdict_data)
        merkle_root = self._compute_sovereign_merkle_dag_root(bundle, cert, verdict)

        now = time.time()
        final_payload = (
            f"{merkle_root}|{verdict.name}|{verdict_data.actuation_mode.name}|"
            f"{cert.merkle_root_sha256}|{now}"
        )
        final_sig = hmac.new(
            self._secret_key, final_payload.encode("utf-8"), hashlib.sha256,
        ).hexdigest()

        if bundle.gauge_observer:
            go = bundle.gauge_observer
            gauge_summary: Tuple[Tuple[str, Any], ...] = (
                ("gauge_class", go.connection.gauge_bundle_class),
                ("cs3", go.chern_simons.cs3_value),
                ("pontryagin", go.chern_simons.pontryagin_number),
                ("stora_zumino_delta", go.chern_simons.stora_zumino_delta),
                ("anomaly_cancelled", go.chern_simons.anomaly_cancelled),
                ("nu_real", go.chern_simons.nu_real),
                ("wilson_phase", go.wilson.wilson_phase),
                ("hopf_c1", go.wilson.hopf_chern_class),
                ("wilson_identity", go.wilson.is_identity),
                ("su2_trace", go.wilson.su2_trace.real),
                ("flux_relative", go.connection.flux_relative),
                ("ym_energy", go.connection.ym_energy),
                ("n_triangles", go.connection.n_triangles),
                ("melnikov_max", go.poincare.melnikov_max),
                ("melnikov_tangle", go.poincare.melnikov_tangle),
                ("kam_epsilon", go.poincare.kam_epsilon),
                ("kam_stable", go.poincare.kam_stable),
                ("arnold_total", go.poincare.arnold_total),
                ("n_sections", go.poincare.n_sections),
                ("lyapunov", go.poincare.lyapunov_exponent),
            )
        else:
            gauge_summary = ()

        rsi3_summary: Tuple[Tuple[str, Any], ...] = (
            ("converged", bundle.rsi3_report.converged if bundle.rsi3_report else False),
            ("lipschitz", bundle.rsi3_report.lipschitz_const if bundle.rsi3_report else 0.0),
            ("final_loss", (
                bundle.rsi3_report.loss_history[-1]
                if bundle.rsi3_report and bundle.rsi3_report.loss_history
                else 0.0
            )),
            ("n_levels", len(bundle.rsi3_report.level_history) if bundle.rsi3_report else 0),
            ("theta_after", bundle.rsi3_report.theta_after if bundle.rsi3_report else ()),
        )
        chain = (
            "_phase1_terminal_bridge_to_phase2",
            "_audit_ultrametric_campaign_phase2",
            "_phase2_terminal_bridge_to_phase3",
            "_adjudicate_sovereign_verdict",
            "_phase3_terminal_synthesis",
        )
        passport = ChronosSovereignGovernancePassport(
            draft_bundle=bundle,
            execution_certificate=cert,
            heyting_verdict=verdict,
            actuation_mode=verdict_data.actuation_mode,
            crowbar_latency_ns=verdict_data.crowbar_latency_ns,
            dirac_purge_energy_mev=verdict_data.dirac_purge_energy_mev,
            executive_impact_report=impact_report,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
            timestamp=now,
            gauge_audit_summary=gauge_summary,
            rsi3_report_summary=rsi3_summary,
            phase_continuity_chain=chain,
            veto_reasons=verdict_data.veto_reasons,
        )
        logger.info(
            "Z_Sovereign = Φ₃∘Φ₂∘Φ₁ | verdict=%s | actuation=%s | "
            "crowbar=%.1fns | merkle=%s",
            verdict.name, verdict_data.actuation_mode.name,
            verdict_data.crowbar_latency_ns, merkle_root[:16],
        )
        return passport


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   §4. ORQUESTADOR SUPREMO + COMPATIBILIDAD v7                            █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

class TOONChronosTemporalAgent(Phase3_ChronosSovereignAdjudicator):
    r"""
    Soberano Agente del Cronograma Temporal (v8.1.0).

        TOONChronosTemporalAgent
          → Phase3_ChronosSovereignAdjudicator     [Φ₃]
             → Phase2_ChronosCampaignOrient         [Φ₂ + RSI³]
                → Phase1_ChronosGaugeObserver       [Φ₁ + Gauge]
                   → TOONChronosTemporalEngine      [motor v8.1]
    """

    def run_full_governance_pipeline(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
        slacks: List[float],
    ) -> ChronosSovereignGovernancePassport:
        r"""
        Tres fases anidadas SIN reentrada:

            1. Φ₁ → GaugeObserverPacket + NovikovCampaignCertificate
            2. _assemble_campaign_bundle → ChronosCampaignDraftBundle
            3. Φ₂-bridge → HeytingVerdictData
            4. Φ₃-síntesis → ChronosSovereignGovernancePassport
        """
        packet, novikov_cert = self._phase1_terminal_bridge_to_phase2(
            schedule_id, num_nodes, edges, slacks,
        )
        bundle = self._assemble_campaign_bundle(packet, novikov_cert, slacks)
        bundle, _verdict_data = self._phase2_terminal_bridge_to_phase3(bundle)
        return self._phase3_terminal_synthesis(bundle)


class ChronosGaugeTopology:
    """Fachada v7: invariantes de calibre proyectados a ChronosGaugeContract."""

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_KEY_V8"):
        self._secret_key = secret_key
        self._phase1 = Phase1_ChronosGaugeObserver(secret_key)

    def compute_gauge_invariants(
        self,
        schedule_id: str,
        num_nodes: int,
        edges: List[Tuple[int, int, float]],
    ) -> ChronosGaugeContract:
        packet = self._phase1.build_gauge_observer_packet(schedule_id, num_nodes, edges)
        return packet.to_chronos_gauge_contract()


class SovereignChronosTemporalAdjudicator:
    """Fachada v7: adjudicación soberana sobre un draft bundle ya construido."""

    def __init__(self, secret_key: bytes = b"APU_FILTER_SOVEREIGN_CHRONOS_KEY_V8"):
        self._secret_key = secret_key
        self._phase3 = Phase3_ChronosSovereignAdjudicator(secret_key)

    def adjudicate_passport(
        self,
        draft_bundle: ChronosCampaignDraftBundle,
    ) -> ChronosSovereignGovernancePassport:
        return self._phase3._phase3_terminal_synthesis(draft_bundle)


# ══════════════════════════════════════════════════════════════════════════════
# §5. EJECUCIÓN DIRECTA DE VERIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 78)
    print("VERIFICACIÓN AUTÓNOMA: TOON CHRONOS TEMPORAL AGENT v8.1.0-SOVEREIGN")
    print("═" * 78)

    nodes = 4
    edges = [
        (0, 1, 10.0),
        (1, 2, 15.0),
        (2, 3, 20.0),
        (0, 2, 22.0),
    ]
    slacks = [2.0, 5.0, 1.0, 4.0]

    agent = TOONChronosTemporalAgent()
    passport = agent.run_full_governance_pipeline(
        "SCHED-2026-BOGOTA-METRO-01", nodes, edges, slacks,
    )

    print("\n[FASE I: GAUGE OBSERVE]")
    for k, v in passport.gauge_audit_summary:
        print(f"    {k:>22} = {v}")

    print("\n[FASE II: CAMPAIGN ORIENT]")
    print(f"    {'v(Slack)':>22} = {passport.draft_bundle.mean_slack_valuation:.4f}")
    print(f"    {'L̂ (Banach)':>22} = {passport.draft_bundle.banach_lipschitz_peak:.4f}")
    for k, v in passport.rsi3_report_summary:
        print(f"    {k:>22} = {v}")

    print("\n[FASE III: SOVEREIGN DECIDE]")
    print(f"    {'Verdict':>22} = {passport.heyting_verdict.name}")
    print(f"    {'Actuation':>22} = {passport.actuation_mode.name}")
    print(f"    {'Crowbar latency':>22} = {passport.crowbar_latency_ns:.2f} ns")
    print(
        f"    {'Contingency':>22} = "
        f"{passport.executive_impact_report['contingency_reserve_reduction']}"
    )
    print(f"    {'Merkle root':>22} = {passport.merkle_root_sha256[:16]}...")
    print(f"    {'Veto reasons':>22} = {passport.veto_reasons or '(ninguno)'}")
    print(f"\n    Cadena funtorial: {' → '.join(passport.phase_continuity_chain)}")

    print("\n" + "─" * 78)
    print("VERIFICACIÓN DE COMPATIBILIDAD v7")
    print("─" * 78)

    gauge_module = ChronosGaugeTopology()
    contract = gauge_module.compute_gauge_invariants(
        "SCHED-2026-BOGOTA-METRO-01", nodes, edges,
    )
    print(f"[v7-compat] ChronosGaugeContract HMAC: {contract.hmac_signature[:16]}...")
    print(f"[v7-compat] CS₃ = {contract.chern_simons_3form:.6e} | ν = {contract.pontryagin_number}")
    print(
        f"[v7-compat] Wilson phase = {contract.wilson_holonomy_phase:.4f} | "
        f"c₁ = {contract.hopf_chern_class:.4f}"
    )

    adjudicator = SovereignChronosTemporalAdjudicator()
    agent_v7 = TOONChronosTemporalAgent()
    bundle_v7 = agent_v7.ingest_and_evaluate_schedule(
        "SCHED-2026-BOGOTA-METRO-01", nodes, edges, slacks,
    )
    passport_v7 = adjudicator.adjudicate_passport(bundle_v7)
    print(f"[v7-compat] Passport verdict: {passport_v7.heyting_verdict.name}")
    print("═" * 78)