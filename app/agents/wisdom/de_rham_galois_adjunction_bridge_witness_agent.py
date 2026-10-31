r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : De Rham-Galois Adjunction Bridge Witness Agent v12.1.0-SOVEREIGN-POINCARE-N3-OODA         ║
║ Ubicación: app/agents/wisdom/de_rham_galois_adjunction_bridge_witness_agent.py                       ║
║ Versión  : 12.1.0-SOVEREIGN-POINCARE-N3-OODA-PhD                                                     ║
║ Función  : Soberano Agéntico Testigo Silencioso que orquesta OODA pasivo sobre el isomorfismo F ⊣ G, ║
║            enriquecido con Mecánica Celeste de Poincaré (secciones, Lindstedt, Birkhoff, KAM,        ║
║            twist, no-degeneración isoenergética, Oseledets) y Automejora Recursiva N3               ║
║            (N1: veredicto · N2: umbrales · N3: meta-tasa · N3-celeste: recurrencia del OODA).       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico Testigo Silencioso de Adjunción De Rham-Galois ()
ejecuta la atestación observacional pasiva sin demolición cuántica (QND AAV, Back-Action ≡ 0.0 dB)
sobre el flujo de inferencia de la adjunción F ⊣ G gobernando el motor espectral
 (app/wisdom/de_rham_galois_adjunction_bridge_witness_engine.py).
Orquesta un ciclo OODA trifásico (Observe → Orient → Decide&Act) compuesto por 9 subcapas de análisis
espectral, computa invariantes integrales de Poincaré-Cartan, descompone coordenadas canónicas de acción-ángulo,
audita la estabilidad de toros KAM diofánticos, y modula adaptativamente los umbrales de monitoreo mediante
el operador de automejora recursiva Nivel 3.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. Observación Débil Cuántica QND AAV sin Retroacción:
   La interacción pasiva entre el aparato observador y el estado del filtro garantiza:
       [A_{\mathrm{witness}}, H_{\mathrm{free}}] = 0 \implies \text{Back-Action} \equiv 0.0\text{ dB}.
   El valor débil de Aharonov-Albert-Vaidman para observables canónicos se evalúa mediante:
       A_w = \frac{\langle \phi | A | \psi \rangle}{\langle \phi | \psi \rangle}.
   Fidelidad cuántica de Uhlmann: F(\rho, \sigma) = (\mathrm{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2 \ge 0.95.

2. Mecánica Celeste de Poincaré y Estabilidad KAM:
   Sección transversal de Poincaré: \Sigma = \{ \rho \in \mathcal{D}_N : \mathrm{Re}\mathrm{Tr}(H_{\mathrm{probe}} \rho) - c = 0 \}.
   El mapa de Poincaré P: \Sigma \to \Sigma preserva el invariante integral de Hilbert-Poincaré \oint p dq.
   La condición diofántica de Kolmogorov-Arnold-Moser (KAM) para la persistencia de toros exige:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{||k||_1^\tau}, \quad \forall k \in \mathbb{Z}^n \setminus \{0\}, \quad \tau > n - 1.

3. Espectro de Oseledets y Formas Normales de Birkhoff:
   El exponente máximo de Lyapunov se obtiene mediante el cociente de crecimiento de variación tangente:
       \lambda_{\max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{||\delta z(t)||_2}{||\delta z(0)||_2}.
   La forma normal de Birkhoff linealiza la vecindad del punto fijo hasta orden cuadrático en acciones J.

4. Automejora Recursiva N3 y Modulación OODA:
   El agente aplica una política de autotuning adaptativo:
     N1: clasificación de veredicto sobre el retículo intuicionista \Omega_3.
     N2: adaptación de umbrales espectrales (kam_floor, fidelity_floor).
     N3: meta-tasa de aprendizaje y contracción de Banach acoplada a la recurrencia del ledger.

AXIOMAS FORMALES
────────────────
Axioma DRGWA-A1 (Pasividad Espectral QND):
    La medición del agente testigo preserva la pureza y entropía cuántica de von Neumann del sistema bajo prueba.
Axioma DRGWA-A2 (No Resonancia de Kolmogorov):
    El sistema permanece en régimen cuasi-periódico estable siempre que la estimación diofántica γ_KAM ≥ 0.01.
Axioma DRGWA-A3 (Simetría Simpléctica de Retorno):
    La transformación tangente del mapa de Poincaré satisface M^T J M = J con error de norma Frobenius inferior a 1.0e-6.
Axioma DRGWA-A4 (Convergencia Monótona del Ledger):
    La función de Lyapunov del ledger de estados V_t satisface V_{t+1} ≤ V_t con contracción Lipschitz κ < 1.0.

INVARIANTES DE ESTADO
─────────────────────
Invariante DRGWA-I1: Back-Action QND ≡ 0.0 dB idénticamente certificado en todo snapshot emitido.
Invariante DRGWA-I2: Normalización de traza cuántica Tr(ρ) ≡ 1.0 con piso de densidad Wilkinson floor ≥ 1.0e-12.
Invariante DRGWA-I3: Fidelidad de Uhlmann F(ρ, ρ_ref) ≥ 0.95 requerida para la certificación COHERENT.
Invariante DRGWA-I4: El agente testigo opera en estricta pasividad y NUNCA dispara el disyuntor ESP32 Crowbar.
Invariante DRGWA-I5: Firma criptográfica HMAC-SHA256 en cada certificado .
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Importación del motor v12.1 (con fallback a v11 y a stub) ──
try:
    from app.wisdom.de_rham_galois_adjunction_bridge_witness_engine import (
        ActionAngleDecomposition,
        DeRhamGaloisAdjunctionWitnessEngine,
        DeRhamGaloisAdjunctionWitnessEngineV12,
        DeRhamGaloisWitnessSnapshot,
        DeRhamGaloisWitnessSnapshotV12,
        HeytingOmega3,
        MetaWitnessReport,
        PoincareReturnAnalysis,
        TuningPolicy,
    )
    _ENGINE_V12_AVAILABLE: Final[bool] = True
except ImportError:
    try:
        from de_rham_galois_adjunction_bridge_witness_engine import (
            ActionAngleDecomposition,
            DeRhamGaloisAdjunctionWitnessEngine,
            DeRhamGaloisAdjunctionWitnessEngineV12,
            DeRhamGaloisWitnessSnapshot,
            DeRhamGaloisWitnessSnapshotV12,
            HeytingOmega3,
            MetaWitnessReport,
            PoincareReturnAnalysis,
            TuningPolicy,
        )
        _ENGINE_V12_AVAILABLE: Final[bool] = True
    except ImportError:
        from enum import IntEnum

        class HeytingOmega3(IntEnum):  # type: ignore[no-redef]
            COHERENT = 0x00
            DEGRADED = 0x01
            VETOED = 0x03

            def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(max(int(self), int(other)))

            def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
                return HeytingOmega3(min(int(self), int(other)))

        DeRhamGaloisAdjunctionWitnessEngine = None  # type: ignore
        DeRhamGaloisAdjunctionWitnessEngineV12 = None  # type: ignore
        DeRhamGaloisWitnessSnapshot = None  # type: ignore
        DeRhamGaloisWitnessSnapshotV12 = None  # type: ignore
        ActionAngleDecomposition = None  # type: ignore
        PoincareReturnAnalysis = None  # type: ignore
        TuningPolicy = None  # type: ignore
        MetaWitnessReport = None  # type: ignore
        _ENGINE_V12_AVAILABLE = False  # type: ignore[misc]

logger = logging.getLogger("APU.Wisdom.DeRhamGaloisAdjunctionBridgeWitnessAgentV12")

# ── Constantes HMAC (dominios v12.1, separados por fase) ──
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_DERHAM_GALOIS_WITNESS_SOVEREIGN_KEY_v12.1.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"DERHAM-GALOIS-WITNESS-P1v12.1-POINCARE-OBSERVE|"
_HMAC_DOMAIN_P2: Final[bytes] = b"DERHAM-GALOIS-WITNESS-P2v12.1-ORIENT-POINCARE|"
_HMAC_DOMAIN_P3: Final[bytes] = b"DERHAM-GALOIS-WITNESS-P3v12.1-DECIDE-N3-CELESTE|"
_CHAIN_HEX_LEN: Final[int] = 32
_AGENT_VERSION: Final[str] = "12.1.0-SOVEREIGN-POINCARE-N3-OODA"

# ── Umbrales de adjudicación celestial v12.1 ──
_KAM_VETO_THRESHOLD: Final[float] = 0.50
_KAM_ESCALATE_THRESHOLD: Final[float] = 0.75
_META_CONSISTENCY_VETO: Final[float] = 0.50
_META_CONSISTENCY_ESCALATE: Final[float] = 0.75
_TWIST_FLOOR: Final[float] = 1.0e-12
_POINCARE_LYAP_VETO: Final[float] = 1.0e-1
_POINCARE_LYAP_ESCALATE: Final[float] = 1.0e-3
_LIOUVILLE_PRESERVATION_MAX: Final[float] = 1.0e-8
_LIOUVILLE_DRIFT_MAX: Final[float] = 1.0e-4
_BACK_ACTION_MAX_DB: Final[float] = 1.0e-12
_N3_TRAJECTORY_WINDOW: Final[int] = 8
_N3_LEDGER_CAP: Final[int] = 64
_WILKINSON_EPSILON: Final[float] = 1.0e-15


def _canonical_complex(z: complex) -> str:
    """Serialización canónica de ℂ: evita (1+0j) vs (1+0.0j) que rompe HMAC.compare_digest."""
    c = complex(z)
    return f"{float(c.real):.16e}{float(c.imag):+.16e}j"


def _hmac_hex(key: bytes, domain: bytes, payload: str) -> str:
    return hmac.new(key, domain + payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _chain_hex(key: bytes, seed: bytes) -> str:
    return hmac.new(key, seed, hashlib.sha256).hexdigest()[:_CHAIN_HEX_LEN]


def _attr_float(obj: Any, name: str, default: float = 0.0) -> float:
    if obj is None or not hasattr(obj, name):
        return float(default)
    return float(getattr(obj, name))


def _attr_int(obj: Any, name: str, default: int = 0) -> int:
    if obj is None or not hasattr(obj, name):
        return int(default)
    return int(getattr(obj, name))


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 1 — OBSERVE (3 sub-capas): sanear · extraer F1 · emitir P1v12.1
# ██╔══██╗ Certificate enriquecido con acción-ángulo, H_ent, Kolmogorov
# ██████╔╝ [HANDOFF] continue_to_phase2_witness_gate_v12 → Φ₁₂ → execute_phase2_v12
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class OODACelestialLedger:
    """
    FASE 1 — Ledger N3-celeste del propio ciclo OODA del agente
    (complementario al RecursiveSelfImprovementLedger interno del motor).

    El ciclo OODA es un mapa discreto
        Φ : (κ, χ, ϑ, τ) ↦ (κ', χ', ϑ', τ')
    sobre (índice KAM, consistencia meta, veredicto Heyting, twist).
    Se miden:
      · recurrencia de Poincaré de (κ, χ) en ℝ²;
      · exponente de Lyapunov discreto de la serie κ;
      · entropía de la órbita de veredictos.
    """
    kam_history: List[float] = field(default_factory=list)
    consistency_history: List[float] = field(default_factory=list)
    verdict_history: List[int] = field(default_factory=list)
    twist_history: List[float] = field(default_factory=list)
    cycle_count: int = 0

    def observe(
        self,
        kam: float,
        consistency: float,
        verdict: HeytingOmega3,
        twist: float,
    ) -> None:
        self.kam_history.append(float(kam))
        self.consistency_history.append(float(consistency))
        self.verdict_history.append(int(verdict))
        self.twist_history.append(float(twist))
        self.cycle_count += 1
        if len(self.kam_history) > _N3_LEDGER_CAP:
            self.kam_history = self.kam_history[-_N3_LEDGER_CAP:]
            self.consistency_history = self.consistency_history[-_N3_LEDGER_CAP:]
            self.verdict_history = self.verdict_history[-_N3_LEDGER_CAP:]
            self.twist_history = self.twist_history[-_N3_LEDGER_CAP:]

    def recurrence_index(self) -> float:
        """r = exp(−d_min / σ) ∈ (0, 1] sobre (κ, χ); cercano a 1 ⇒ retorno próximo."""
        n = len(self.kam_history)
        if n < 4:
            return 0.0
        pts = np.column_stack([
            np.array(self.kam_history, dtype=float),
            np.array(self.consistency_history, dtype=float),
        ])
        last = pts[-1]
        prior = pts[:-1]
        dist = np.sqrt(np.sum((prior - last) ** 2, axis=1))
        d_min = float(np.min(dist))
        sigma = float(np.std(pts, axis=0).mean()) + _WILKINSON_EPSILON
        return float(math.exp(-d_min / sigma))

    def lyapunov(self) -> float:
        """λ_Φ ≈ ⟨ log |Δκ_{k+1}| − log |Δκ_k| ⟩ sobre el índice KAM del OODA."""
        series = np.array(self.kam_history, dtype=float)
        if series.size < 4:
            return 0.0
        d1 = np.maximum(np.abs(np.diff(series)), _WILKINSON_EPSILON)
        logs = np.log(d1[1:] / d1[:-1])
        return float(np.mean(logs))

    def verdict_entropy(self) -> float:
        window = self.verdict_history[-_N3_TRAJECTORY_WINDOW:]
        if len(window) < 2:
            return 0.0
        _uniq, counts = np.unique(np.array(window, dtype=int), return_counts=True)
        probs = counts / float(counts.sum())
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def trajectory_window(self) -> Tuple[int, ...]:
        return tuple(self.verdict_history[-_N3_TRAJECTORY_WINDOW:])


@dataclass(frozen=True, slots=True)
class DeRhamGaloisWitnessPhase1CertificateV12:
    """
    FASE 1 — Certificate enriquecido v12.1 (Observe + Poincaré F1).
    Invariantes isométricos v11 + observables acción-ángulo + H_ent + Kolmogorov.
    """
    # v11 heredado
    witness_id: str
    trace_preservation: float
    liouville_drift: float
    unit_eta_norm: float
    counit_epsilon_norm: float
    galois_gap_delta: float
    isometry_verdict: HeytingOmega3
    chain_hash: str
    hmac_p1_signature: str
    timestamp_utc: float
    # v12 · FASE 1 (Poincaré action-angle)
    action_total: float
    phase_space_volume: float
    lyapunov_max: float
    lyapunov_min: float
    n3_reflections_preexisting: int
    engine_v12_active: bool
    # v12.1 · FASE 1
    von_neumann_entropy: float
    kolmogorov_nondegeneracy: float


class DeRhamGaloisWitnessPhase1ObserverV12:
    """
    FASE 1 — OBSERVE con 3 sub-capas anidadas.

    Sub-capa 1.1: `_invoke_engine_v12_and_cache` — invocación ÚNICA del motor v12.1.
                  Garantiza que el ledger N3 del motor avanza solo una vez por ciclo OODA.
    Sub-capa 1.2: `_extract_phase1_invariants` — extrae ActionAngleDecomposition / H_ent / Kolmogorov.
    Sub-capa 1.3: `execute_phase1_v12` — emite P1 con HMAC-SHA256 canónico.
    Handoff:      `continue_to_phase2_witness_gate_v12` — revalida HMAC P1  (Φ₁₂ → FASE 2).
    """

    def __init__(
        self,
        dimension: int = 56,
        engine: Optional[Any] = None,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.dimension: Final[int] = max(4, int(dimension))
        if engine is not None:
            self.engine = engine
        elif _ENGINE_V12_AVAILABLE and DeRhamGaloisAdjunctionWitnessEngineV12 is not None:
            self.engine = DeRhamGaloisAdjunctionWitnessEngineV12(
                dimension=dimension, secret_key=secret_key
            )
        elif DeRhamGaloisAdjunctionWitnessEngine is not None:
            self.engine = DeRhamGaloisAdjunctionWitnessEngine(
                dimension=dimension, secret_key=secret_key
            )
        else:
            self.engine = None
        self._session_snapshot: Optional[Any] = None
        self._ooda_ledger: OODACelestialLedger = OODACelestialLedger()

    def _compute_hmac(self, domain: bytes, payload: Any) -> str:
        p = payload if isinstance(payload, str) else str(payload)
        return _hmac_hex(self.secret_key, domain, p)

    def _is_engine_v12(self) -> bool:
        return (
            _ENGINE_V12_AVAILABLE
            and DeRhamGaloisAdjunctionWitnessEngineV12 is not None
            and self.engine is not None
            and isinstance(self.engine, DeRhamGaloisAdjunctionWitnessEngineV12)
        )

    # ── Sub-capa 1.1: invocación única del motor ──
    def _invoke_engine_v12_and_cache(
        self,
        cartridges: Sequence[Any],
        target_density_matrix: Optional[NDArray[Any]],
        phi_initial: Optional[NDArray[Any]],
        phi_final: Optional[NDArray[Any]],
    ) -> Any:
        """
        Sub-capa 1.1 — Ejecuta el motor v12.1 EXACTAMENTE UNA VEZ por ciclo OODA.
        Cachea el snapshot maestro. NO re-extrae ActionAngleDecomposition: el
        snapshot v12.1 ya porta todos los invariantes F1 (evita un eigh 56×56 extra
        y cualquier tentación de doble-contabilidad del ledger N3).
        """
        if self.engine is None:
            self._session_snapshot = None
            return None

        if self._is_engine_v12():
            snapshot = self.engine.evaluate_adjunction_witness_v12(
                cartridges=cartridges,
                target_density_matrix=target_density_matrix,
                phi_initial=phi_initial,
                phi_final=phi_final,
            )
        elif hasattr(self.engine, "evaluate_adjunction_witness"):
            snapshot = self.engine.evaluate_adjunction_witness(
                cartridges=cartridges,
                target_density_matrix=target_density_matrix,
                phi_initial=phi_initial,
                phi_final=phi_final,
            )
        else:
            snapshot = None

        self._session_snapshot = snapshot
        return snapshot

    # ── Sub-capa 1.2: extracción de invariantes F1 ──
    def _extract_phase1_invariants(self, snapshot: Optional[Any]) -> Dict[str, Any]:
        """
        Sub-capa 1.2 — Extrae invariantes de Fase 1 (isometría + acción-ángulo + H_ent).
        Campos ausentes (motor v11 / stub) se anulan a neutro isométrico.
        """
        if snapshot is None:
            return {
                "trace": 1.0,
                "liouville": 0.0,
                "eta": 0.0,
                "eps": 0.0,
                "gap": 0.0,
                "verdict": HeytingOmega3.COHERENT,
                "action_total": 0.0,
                "phase_volume": 0.0,
                "lyap_max": 0.0,
                "lyap_min": 0.0,
                "n3_reflections": 0,
                "engine_v12": False,
                "von_neumann": 0.0,
                "kolmogorov": 0.0,
            }
        is_v12 = hasattr(snapshot, "action_total")
        return {
            "trace": float(snapshot.trace_density),
            "liouville": float(snapshot.liouville_drift),
            "eta": float(snapshot.unit_eta_norm),
            "eps": float(snapshot.counit_epsilon_norm),
            "gap": float(snapshot.galois_gap_delta),
            "verdict": snapshot.coherence_verdict,
            "action_total": _attr_float(snapshot, "action_total"),
            "phase_volume": _attr_float(snapshot, "phase_space_volume"),
            "lyap_max": _attr_float(snapshot, "lyapunov_max"),
            "lyap_min": _attr_float(snapshot, "lyapunov_min"),
            "n3_reflections": _attr_int(snapshot, "level_3_reflections"),
            "engine_v12": is_v12,
            "von_neumann": _attr_float(snapshot, "von_neumann_entropy"),
            "kolmogorov": _attr_float(snapshot, "kolmogorov_nondegeneracy"),
        }

    def _serialize_p1(
        self,
        witness_id: str,
        trace: float,
        liouville: float,
        eta: float,
        eps: float,
        gap: float,
        action_total: float,
        phase_volume: float,
        lyap_max: float,
        lyap_min: float,
        n3_reflections: int,
        engine_v12: bool,
        von_neumann: float,
        kolmogorov: float,
        verdict_value: int,
        chain_hash: str,
    ) -> str:
        """Fuente única de serialización P1 (firma y verificación idénticas)."""
        return (
            f"{witness_id}:{trace:.8f}:{liouville:.8f}:"
            f"{eta:.8f}:{eps:.8f}:{gap:.8e}:"
            f"{action_total:.8f}:{phase_volume:.8f}:"
            f"{lyap_max:+.6e}:{lyap_min:+.6e}:"
            f"{n3_reflections}:{int(engine_v12)}:"
            f"{von_neumann:.8f}:{kolmogorov:.8e}:"
            f"{verdict_value}:{chain_hash}"
        )

    # ── Sub-capa 1.3: emisión del certificado P1 ──
    def execute_phase1_v12(
        self,
        cartridges: Sequence[Any] = (),
        target_density_matrix: Optional[NDArray[Any]] = None,
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> DeRhamGaloisWitnessPhase1CertificateV12:
        """
        Sub-capa 1.3 — Emite Phase1CertificateV12 firmado con HMAC-SHA256 canónico.
        Retorna el certificado que alimenta la Fase 2 vía la compuerta homomórfica Φ₁₂.
        """
        now = time.time()
        snapshot = self._invoke_engine_v12_and_cache(
            cartridges, target_density_matrix, phi_initial, phi_final
        )
        inv = self._extract_phase1_invariants(snapshot)

        wit_id = f"DERHAM-WIT-P1v12-{int(now * 1000.0) % 1000000:06d}"
        chain_seed = (
            f"{wit_id}:{inv['trace']:.8f}:{inv['gap']:.8e}:"
            f"{inv['action_total']:.8f}:{inv['kolmogorov']:.8e}:{inv['verdict'].value}"
        ).encode("utf-8")
        chain_hash = _chain_hex(self.secret_key, chain_seed)

        payload = self._serialize_p1(
            witness_id=wit_id,
            trace=inv["trace"],
            liouville=inv["liouville"],
            eta=inv["eta"],
            eps=inv["eps"],
            gap=inv["gap"],
            action_total=inv["action_total"],
            phase_volume=inv["phase_volume"],
            lyap_max=inv["lyap_max"],
            lyap_min=inv["lyap_min"],
            n3_reflections=inv["n3_reflections"],
            engine_v12=inv["engine_v12"],
            von_neumann=inv["von_neumann"],
            kolmogorov=inv["kolmogorov"],
            verdict_value=int(inv["verdict"].value),
            chain_hash=chain_hash,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P1, payload)

        return DeRhamGaloisWitnessPhase1CertificateV12(
            witness_id=wit_id,
            trace_preservation=inv["trace"],
            liouville_drift=inv["liouville"],
            unit_eta_norm=inv["eta"],
            counit_epsilon_norm=inv["eps"],
            galois_gap_delta=inv["gap"],
            isometry_verdict=inv["verdict"],
            chain_hash=chain_hash,
            hmac_p1_signature=sig,
            timestamp_utc=now,
            action_total=inv["action_total"],
            phase_space_volume=inv["phase_volume"],
            lyapunov_max=inv["lyap_max"],
            lyapunov_min=inv["lyap_min"],
            n3_reflections_preexisting=inv["n3_reflections"],
            engine_v12_active=inv["engine_v12"],
            von_neumann_entropy=inv["von_neumann"],
            kolmogorov_nondegeneracy=inv["kolmogorov"],
        )

    # ── Handoff FASE 1 → FASE 2  (último método de FASE 1 ≡ primer acto de FASE 2) ──
    def continue_to_phase2_witness_gate_v12(
        self, cert1: DeRhamGaloisWitnessPhase1CertificateV12
    ) -> bool:
        """
        Compuerta Homomórfica Φ₁₂ — Revalida la firma HMAC canónica de P1v12.1.

        ADMITE isometry_verdict ∈ {COHERENT, DEGRADED, VETOED} (pasividad QND:
        el testigo observa, nunca veta por demolición).

        Este método ES el gozne formal F1 ▷ F2: `execute_phase2_v12` lo invoca
        como primera instrucción, de modo que la definición final de FASE 1
        es el inicio operacional de FASE 2.
        """
        payload = self._serialize_p1(
            witness_id=cert1.witness_id,
            trace=cert1.trace_preservation,
            liouville=cert1.liouville_drift,
            eta=cert1.unit_eta_norm,
            eps=cert1.counit_epsilon_norm,
            gap=cert1.galois_gap_delta,
            action_total=cert1.action_total,
            phase_volume=cert1.phase_space_volume,
            lyap_max=cert1.lyapunov_max,
            lyap_min=cert1.lyapunov_min,
            n3_reflections=cert1.n3_reflections_preexisting,
            engine_v12=cert1.engine_v12_active,
            von_neumann=cert1.von_neumann_entropy,
            kolmogorov=cert1.kolmogorov_nondegeneracy,
            verdict_value=int(cert1.isometry_verdict.value),
            chain_hash=cert1.chain_hash,
        )
        expected = self._compute_hmac(_HMAC_DOMAIN_P1, payload)
        return hmac.compare_digest(cert1.hmac_p1_signature, expected)

    # ─────────── (fin FASE 1 · handoff a FASE 2 · execute_phase2_v12 vía Φ₁₂) ───────────


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 2 — ORIENT (3 sub-capas): extraer F2 Poincaré · extraer N2/N3 · emitir P2v12.1
# ██╔══██╗ Certificate enriquecido con KAM, Lindstedt, Birkhoff, twist, Oseledets, TuningPolicy
# ██████╔╝ [HANDOFF] continue_to_phase3_witness_gate_v12 → Φ₂₃ → execute_phase3_v12
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class DeRhamGaloisWitnessPhase2CertificateV12:
    """
    FASE 2 — Certificate enriquecido v12.1 (Orient + Poincaré F2 + Tuning N3-celeste).
    Observables QND v11 + análisis de retorno de Poincaré + política adaptativa N2/N3.
    """
    # v11 heredado
    witness_id: str
    phase1_ref_id: str
    qnd_weak_value_aav: complex
    weak_value_modulus: float
    back_action_db: float
    uhlmann_fidelity: float
    bures_distance: float
    connection_A_norm: float
    qnd_verdict: HeytingOmega3
    chain_hash: str
    hmac_p2_signature: str
    timestamp_utc: float
    # v12 · FASE 2 (Poincaré + KAM)
    poincare_crossings_count: int
    action_drift_max: float
    action_drift_mean: float
    lindstedt_correction_norm: float
    birkhoff_quadratic_coefficient: float
    kam_stability_index: float
    kam_resonance_violations: int
    kam_worst_ratio: float
    # v12 · FASE 2 (Tuning N2/N3)
    adaptive_fidelity_floor: float
    adaptive_gap_ceiling: float
    adaptation_rate: float
    meta_adaptation_rate: float
    level_3_entropy: float
    # v12.1 · FASE 2 (celeste)
    birkhoff_matrix_frobenius: float
    twist_index: float
    rotation_number: float
    isoenergetic_nondegeneracy: float
    poincare_lyapunov_max: float
    liouville_preservation_error: float
    mean_return_time: float
    n3_recurrence_index: float
    n3_adaptation_lyapunov: float


class DeRhamGaloisWitnessPhase2OrientorV12(DeRhamGaloisWitnessPhase1ObserverV12):
    """
    FASE 2 — ORIENT con 3 sub-capas anidadas.

    Continúa FASE 1: la primera instrucción de `execute_phase2_v12` ES
    `continue_to_phase2_witness_gate_v12` (Φ₁₂), último método formal de FASE 1.

    Sub-capa 2.1: `_extract_phase2_poincare_invariants` — Poincaré/KAM/Lindstedt/Birkhoff/twist/Oseledets.
    Sub-capa 2.2: `_extract_phase2_n3_tuning` — TuningPolicy N2/N3/N3-celeste.
    Sub-capa 2.3: `execute_phase2_v12` — emite P2 con HMAC-SHA256 canónico.
    Handoff:      `continue_to_phase3_witness_gate_v12` — revalida HMAC P2  (Φ₂₃ → FASE 3).
    """

    # ── Sub-capa 2.1 ──
    def _extract_phase2_poincare_invariants(self, snapshot: Optional[Any]) -> Dict[str, Any]:
        """Extrae invariantes Poincaré F2 del snapshot v12.1 (ceros / neutros si v11)."""
        if snapshot is None:
            return {
                "crossings": 0,
                "drift_max": 0.0,
                "drift_mean": 0.0,
                "lindstedt": 0.0,
                "birkhoff": 0.0,
                "kam_index": 1.0,
                "kam_violations": 0,
                "kam_worst": 1.0e9,
                "birkhoff_fro": 0.0,
                "twist": 0.0,
                "rotation": 0.0,
                "isoenergetic": 0.0,
                "lyap_p": 0.0,
                "liouville_u": 0.0,
                "mean_T": 0.0,
            }
        return {
            "crossings": _attr_int(snapshot, "poincare_crossings_count"),
            "drift_max": _attr_float(snapshot, "action_drift_max"),
            "drift_mean": _attr_float(snapshot, "action_drift_mean"),
            "lindstedt": _attr_float(snapshot, "lindstedt_correction_norm"),
            "birkhoff": _attr_float(snapshot, "birkhoff_quadratic_coefficient"),
            "kam_index": _attr_float(snapshot, "kam_stability_index", 1.0),
            "kam_violations": _attr_int(snapshot, "kam_resonance_violations"),
            "kam_worst": _attr_float(snapshot, "kam_worst_ratio", 1.0e9),
            "birkhoff_fro": _attr_float(snapshot, "birkhoff_matrix_frobenius"),
            "twist": _attr_float(snapshot, "twist_index"),
            "rotation": _attr_float(snapshot, "rotation_number"),
            "isoenergetic": _attr_float(snapshot, "isoenergetic_nondegeneracy"),
            "lyap_p": _attr_float(snapshot, "poincare_lyapunov_max"),
            "liouville_u": _attr_float(snapshot, "liouville_preservation_error"),
            "mean_T": _attr_float(snapshot, "mean_return_time"),
        }

    # ── Sub-capa 2.2 ──
    def _extract_phase2_n3_tuning(self, snapshot: Optional[Any]) -> Dict[str, float]:
        """Extrae TuningPolicy N2/N3/N3-celeste del snapshot v12.1."""
        if snapshot is None or not hasattr(snapshot, "adaptive_fidelity_floor"):
            return {
                "fidelity_floor": 0.95,
                "gap_ceiling": 1.0e-6,
                "adapt_rate": 0.05,
                "meta_adapt_rate": 0.01,
                "entropy": 0.0,
                "n3_recurrence": 0.0,
                "n3_lyapunov": 0.0,
            }
        return {
            "fidelity_floor": float(snapshot.adaptive_fidelity_floor),
            "gap_ceiling": float(snapshot.adaptive_gap_ceiling),
            "adapt_rate": float(snapshot.adaptation_rate),
            "meta_adapt_rate": float(snapshot.meta_adaptation_rate),
            "entropy": float(snapshot.meta_verdict_entropy),
            "n3_recurrence": _attr_float(snapshot, "n3_recurrence_index"),
            "n3_lyapunov": _attr_float(snapshot, "n3_adaptation_lyapunov"),
        }

    def _serialize_p2(
        self,
        witness_id: str,
        phase1_ref_id: str,
        a_w: complex,
        w_mod: float,
        back_action: float,
        fidelity: float,
        bures: float,
        conn_A: float,
        crossings: int,
        drift_max: float,
        drift_mean: float,
        lindstedt: float,
        birkhoff: float,
        kam_index: float,
        kam_violations: int,
        kam_worst: float,
        fidelity_floor: float,
        gap_ceiling: float,
        adapt_rate: float,
        meta_adapt_rate: float,
        entropy: float,
        birkhoff_fro: float,
        twist: float,
        rotation: float,
        isoenergetic: float,
        lyap_p: float,
        liouville_u: float,
        mean_T: float,
        n3_recurrence: float,
        n3_lyapunov: float,
        verdict_value: int,
        chain_hash: str,
    ) -> str:
        """Fuente única de serialización P2 (complex canónico; firma ≡ verificación)."""
        return (
            f"{witness_id}:{phase1_ref_id}:{_canonical_complex(a_w)}:"
            f"{w_mod:.8f}:{back_action:.1f}:"
            f"{fidelity:.8f}:{bures:.8f}:{conn_A:.8f}:"
            f"{crossings}:{drift_max:.8e}:{drift_mean:.8e}:"
            f"{lindstedt:.8e}:{birkhoff:.8e}:"
            f"{kam_index:.8f}:{kam_violations}:{kam_worst:.6e}:"
            f"{fidelity_floor:.8f}:{gap_ceiling:.8e}:"
            f"{adapt_rate:.6f}:{meta_adapt_rate:.6f}:{entropy:.6f}:"
            f"{birkhoff_fro:.8e}:{twist:.8e}:{rotation:.8f}:"
            f"{isoenergetic:.8e}:{lyap_p:+.8e}:{liouville_u:.8e}:{mean_T:.8e}:"
            f"{n3_recurrence:.8f}:{n3_lyapunov:+.8e}:"
            f"{verdict_value}:{chain_hash}"
        )

    # ── Sub-capa 2.3 ──
    def execute_phase2_v12(
        self,
        cert1: DeRhamGaloisWitnessPhase1CertificateV12,
        cartridges: Sequence[Any] = (),
        target_density_matrix: Optional[NDArray[Any]] = None,
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> DeRhamGaloisWitnessPhase2CertificateV12:
        """
        Sub-capa 2.3 — Emite Phase2CertificateV12 firmado con HMAC-SHA256 canónico.

        Primera instrucción: Φ₁₂ = `continue_to_phase2_witness_gate_v12` (último
        método de FASE 1). Consume el snapshot cacheado por 1.1; NUNCA re-invoca
        al motor (protegería el ledger N3 contra doble-contabilidad).
        """
        now = time.time()
        if not self.continue_to_phase2_witness_gate_v12(cert1):
            raise ValueError("Falla de autenticidad en Compuerta Φ₁₂ del Testigo v12.1")

        snapshot = self._session_snapshot
        if snapshot is None and self.engine is not None:
            raise RuntimeError(
                "Invariante OODA violado: snapshot maestro ausente tras Fase 1. "
                "Abortando para proteger el ledger N3 (no se re-invoca el motor)."
            )

        if snapshot is not None:
            a_w = complex(getattr(snapshot, "qnd_weak_value_aav", 0.0))
            w_mod = float(getattr(snapshot, "weak_value_modulus", 0.0))
            back_action = float(getattr(snapshot, "back_action_db", 0.0))
            fidelity = float(getattr(snapshot, "uhlmann_fidelity", 1.0))
            bures = float(getattr(snapshot, "bures_distance", 0.0))
            conn_A = float(getattr(snapshot, "connection_A_norm", 0.0))
            qnd_verdict = getattr(snapshot, "coherence_verdict", cert1.isometry_verdict)
        else:
            a_w, w_mod, back_action = 0.0 + 0.0j, 0.0, 0.0
            fidelity, bures, conn_A = 1.0, 0.0, 0.0
            qnd_verdict = cert1.isometry_verdict

        poinc = self._extract_phase2_poincare_invariants(snapshot)
        tune = self._extract_phase2_n3_tuning(snapshot)

        wit_id = f"DERHAM-WIT-P2v12-{int(now * 1000.0) % 1000000:06d}"
        chain_seed = (
            f"{cert1.chain_hash}:{wit_id}:{fidelity:.8f}:{bures:.8f}:"
            f"{poinc['kam_index']:.8f}:{poinc['twist']:.8e}:{qnd_verdict.value}"
        ).encode("utf-8")
        chain_hash = _chain_hex(self.secret_key, chain_seed)

        payload = self._serialize_p2(
            witness_id=wit_id,
            phase1_ref_id=cert1.witness_id,
            a_w=a_w,
            w_mod=w_mod,
            back_action=back_action,
            fidelity=fidelity,
            bures=bures,
            conn_A=conn_A,
            crossings=poinc["crossings"],
            drift_max=poinc["drift_max"],
            drift_mean=poinc["drift_mean"],
            lindstedt=poinc["lindstedt"],
            birkhoff=poinc["birkhoff"],
            kam_index=poinc["kam_index"],
            kam_violations=poinc["kam_violations"],
            kam_worst=poinc["kam_worst"],
            fidelity_floor=tune["fidelity_floor"],
            gap_ceiling=tune["gap_ceiling"],
            adapt_rate=tune["adapt_rate"],
            meta_adapt_rate=tune["meta_adapt_rate"],
            entropy=tune["entropy"],
            birkhoff_fro=poinc["birkhoff_fro"],
            twist=poinc["twist"],
            rotation=poinc["rotation"],
            isoenergetic=poinc["isoenergetic"],
            lyap_p=poinc["lyap_p"],
            liouville_u=poinc["liouville_u"],
            mean_T=poinc["mean_T"],
            n3_recurrence=tune["n3_recurrence"],
            n3_lyapunov=tune["n3_lyapunov"],
            verdict_value=int(qnd_verdict.value),
            chain_hash=chain_hash,
        )
        sig = self._compute_hmac(_HMAC_DOMAIN_P2, payload)

        return DeRhamGaloisWitnessPhase2CertificateV12(
            witness_id=wit_id,
            phase1_ref_id=cert1.witness_id,
            qnd_weak_value_aav=a_w,
            weak_value_modulus=w_mod,
            back_action_db=back_action,
            uhlmann_fidelity=fidelity,
            bures_distance=bures,
            connection_A_norm=conn_A,
            qnd_verdict=qnd_verdict,
            chain_hash=chain_hash,
            hmac_p2_signature=sig,
            timestamp_utc=now,
            poincare_crossings_count=poinc["crossings"],
            action_drift_max=poinc["drift_max"],
            action_drift_mean=poinc["drift_mean"],
            lindstedt_correction_norm=poinc["lindstedt"],
            birkhoff_quadratic_coefficient=poinc["birkhoff"],
            kam_stability_index=poinc["kam_index"],
            kam_resonance_violations=poinc["kam_violations"],
            kam_worst_ratio=poinc["kam_worst"],
            adaptive_fidelity_floor=tune["fidelity_floor"],
            adaptive_gap_ceiling=tune["gap_ceiling"],
            adaptation_rate=tune["adapt_rate"],
            meta_adaptation_rate=tune["meta_adapt_rate"],
            level_3_entropy=tune["entropy"],
            birkhoff_matrix_frobenius=poinc["birkhoff_fro"],
            twist_index=poinc["twist"],
            rotation_number=poinc["rotation"],
            isoenergetic_nondegeneracy=poinc["isoenergetic"],
            poincare_lyapunov_max=poinc["lyap_p"],
            liouville_preservation_error=poinc["liouville_u"],
            mean_return_time=poinc["mean_T"],
            n3_recurrence_index=tune["n3_recurrence"],
            n3_adaptation_lyapunov=tune["n3_lyapunov"],
        )

    # ── Handoff FASE 2 → FASE 3  (último método de FASE 2 ≡ primer acto de FASE 3) ──
    def continue_to_phase3_witness_gate_v12(
        self, cert2: DeRhamGaloisWitnessPhase2CertificateV12
    ) -> bool:
        """
        Compuerta Homomórfica Φ₂₃ — Revalida la firma HMAC canónica de P2v12.1.

        Este método ES el gozne formal F2 ▷ F3: `execute_phase3_v12` lo invoca
        como primera instrucción.
        """
        payload = self._serialize_p2(
            witness_id=cert2.witness_id,
            phase1_ref_id=cert2.phase1_ref_id,
            a_w=cert2.qnd_weak_value_aav,
            w_mod=cert2.weak_value_modulus,
            back_action=cert2.back_action_db,
            fidelity=cert2.uhlmann_fidelity,
            bures=cert2.bures_distance,
            conn_A=cert2.connection_A_norm,
            crossings=cert2.poincare_crossings_count,
            drift_max=cert2.action_drift_max,
            drift_mean=cert2.action_drift_mean,
            lindstedt=cert2.lindstedt_correction_norm,
            birkhoff=cert2.birkhoff_quadratic_coefficient,
            kam_index=cert2.kam_stability_index,
            kam_violations=cert2.kam_resonance_violations,
            kam_worst=cert2.kam_worst_ratio,
            fidelity_floor=cert2.adaptive_fidelity_floor,
            gap_ceiling=cert2.adaptive_gap_ceiling,
            adapt_rate=cert2.adaptation_rate,
            meta_adapt_rate=cert2.meta_adaptation_rate,
            entropy=cert2.level_3_entropy,
            birkhoff_fro=cert2.birkhoff_matrix_frobenius,
            twist=cert2.twist_index,
            rotation=cert2.rotation_number,
            isoenergetic=cert2.isoenergetic_nondegeneracy,
            lyap_p=cert2.poincare_lyapunov_max,
            liouville_u=cert2.liouville_preservation_error,
            mean_T=cert2.mean_return_time,
            n3_recurrence=cert2.n3_recurrence_index,
            n3_lyapunov=cert2.n3_adaptation_lyapunov,
            verdict_value=int(cert2.qnd_verdict.value),
            chain_hash=cert2.chain_hash,
        )
        expected = self._compute_hmac(_HMAC_DOMAIN_P2, payload)
        return hmac.compare_digest(cert2.hmac_p2_signature, expected)

    # ─────────── (fin FASE 2 · handoff a FASE 3 · execute_phase3_v12 vía Φ₂₃) ───────────


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# ██████╗  FASE 3 — DECIDE&ACT (3 sub-capas): extraer meta F3 · adjudicar celestial · emitir Estado
# ██╔══██╗ Estado soberano enriquecido con órbita Poincaré del OODA + PASSIVE_SILENT
# ██████╔╝ [SALIDA] TopologicalDeRhamGaloisWitnessSovereignStateV12
# ██╔═══╝
# ╚═╝
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TopologicalDeRhamGaloisWitnessSovereignStateV12:
    """
    FASE 3 — Estado soberano enriquecido v12.1.
    Certificados P1/P2, veredicto global celestial, auditoría meta N3,
    órbita Poincaré del propio OODA y reporte visceral derivado (no inventado).
    """
    # v11 heredado
    sovereign_state_id: str
    phase1_cert: DeRhamGaloisWitnessPhase1CertificateV12
    phase2_cert: DeRhamGaloisWitnessPhase2CertificateV12
    global_verdict: HeytingOmega3
    back_action_db: float
    merkle_sha256_root: str
    hmac_provenance_signature: str
    visceral_report: Dict[str, Any]
    timestamp_utc: float
    # v12 · FASE 3 (meta N3)
    meta_verdict_entropy: float
    meta_consistency_score: float
    meta_fidelity_std: float
    meta_gap_std: float
    kam_adjusted_verdict: HeytingOmega3
    n3_verdict_trajectory: Tuple[int, ...]
    n3_cycle_count: int
    passive_silent_invariant_verified: bool
    # v12.1 · FASE 3 (OODA celeste)
    ooda_recurrence_index: float
    ooda_lyapunov: float
    ooda_verdict_entropy: float
    twist_index: float
    isoenergetic_nondegeneracy: float
    poincare_lyapunov_max: float
    liouville_preservation_error: float
    kolmogorov_nondegeneracy: float
    von_neumann_entropy: float
    agent_version: str


class DeRhamGaloisWitnessPhase3DeciderV12(DeRhamGaloisWitnessPhase2OrientorV12):
    """
    FASE 3 — DECIDE & ACT con 3 sub-capas anidadas.

    Continúa FASE 2: la primera instrucción de `execute_phase3_v12` ES
    `continue_to_phase3_witness_gate_v12` (Φ₂₃), último método formal de FASE 2.

    Sub-capa 3.1: `_extract_phase3_meta_witness` — MetaWitnessReport F3 + ledger del motor.
    Sub-capa 3.2: `_adjudicate_celestial` — adjudicación Heyting R1–R9.
    Sub-capa 3.3: `execute_phase3_v12` — Estado soberano + órbita OODA + PASSIVE_SILENT.
    """

    # ── Sub-capa 3.1 ──
    def _engine_ledger_stats(self) -> Dict[str, float]:
        """Lee desviaciones del ledger N3 del motor (campos no expuestos en el snapshot)."""
        ledger = getattr(self.engine, "_ledger", None) if self.engine is not None else None
        if ledger is None:
            return {"fid_std": 0.0, "gap_std": 0.0}
        fid = getattr(ledger, "fidelity_history", []) or []
        gap = getattr(ledger, "gap_history", []) or []
        fid_w = fid[-_N3_TRAJECTORY_WINDOW:]
        gap_w = gap[-_N3_TRAJECTORY_WINDOW:]
        fid_std = float(np.std(fid_w)) if len(fid_w) >= 2 else 0.0
        gap_std = float(np.std(gap_w)) if len(gap_w) >= 2 else 0.0
        return {"fid_std": fid_std, "gap_std": gap_std}

    def _extract_phase3_meta_witness(self, snapshot: Optional[Any]) -> Dict[str, Any]:
        """
        Sub-capa 3.1 — Extrae observables meta del snapshot v12.1.
        Completa fid_std / gap_std desde el ledger del motor si el snapshot no los porta.
        """
        stats = self._engine_ledger_stats()
        if snapshot is None or not hasattr(snapshot, "meta_consistency_score"):
            return {
                "entropy": 0.0,
                "consistency": 1.0,
                "fid_std": stats["fid_std"],
                "gap_std": stats["gap_std"],
            }
        return {
            "entropy": float(snapshot.meta_verdict_entropy),
            "consistency": float(snapshot.meta_consistency_score),
            "fid_std": _attr_float(snapshot, "meta_fidelity_std", stats["fid_std"]),
            "gap_std": _attr_float(snapshot, "meta_gap_std", stats["gap_std"]),
        }

    # ── Sub-capa 3.2 ──
    def _adjudicate_celestial(
        self,
        base_verdict: HeytingOmega3,
        kam_index: float,
        meta_consistency: float,
        twist_index: float,
        isoenergetic: float,
        poincare_lyapunov: float,
        liouville_u: float,
        kolmogorov: float,
    ) -> HeytingOmega3:
        """
        Sub-capa 3.2 — Adjudicación Heyting celestial.

        Reglas (orden de aplicación; meet = peor):
          R1: base_verdict = isometry ⊓ qnd                         (heredado v11)
          R2: meta_consistency < VETO            → VETOED           (evidencia meta insuficiente)
          R3: KAM_index < VETO                   → VETOED           (observación no diofantina)
          R4: |det U|−1 > ε_Liouville            → VETOED           (rompe QND / unitariedad)
          R5: λ_max(P) > λ_VETO                  → VETOED           (sección caótica, no certificable)
          R6: meta_consistency < ESCALATE        → DEGRADED
          R7: KAM_index < ESCALATE               → DEGRADED
          R8: twist = 0  (Poincaré–Birkhoff falla)→ DEGRADED
          R9: isoenergética = 0 o Kolmogorov = 0 → DEGRADED
          R10: λ_max(P) > λ_ESCALATE             → DEGRADED
        """
        verdict = base_verdict
        if meta_consistency < _META_CONSISTENCY_VETO:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        if kam_index < _KAM_VETO_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        if liouville_u > _LIOUVILLE_PRESERVATION_MAX:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        if poincare_lyapunov > _POINCARE_LYAP_VETO:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        if meta_consistency < _META_CONSISTENCY_ESCALATE:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if kam_index < _KAM_ESCALATE_THRESHOLD:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if twist_index <= _TWIST_FLOOR:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if isoenergetic <= _TWIST_FLOOR or kolmogorov <= _TWIST_FLOOR:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if poincare_lyapunov > _POINCARE_LYAP_ESCALATE:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        return verdict

    def _build_visceral_report(
        self,
        cert1: DeRhamGaloisWitnessPhase1CertificateV12,
        cert2: DeRhamGaloisWitnessPhase2CertificateV12,
        meta: Dict[str, Any],
        passive_silent_ok: bool,
        ooda_rec: float,
        ooda_lyap: float,
    ) -> Dict[str, Any]:
        """
        Reporte visceral derivado de observables espectrales reales.
        No se emiten porcentajes financieros no derivados (WACC, liquidity, …).
        """
        isometry_margin = float(max(0.0, 1.0 - min(1.0, cert1.liouville_drift / _LIOUVILLE_DRIFT_MAX)))
        qnd_integrity = 1.0 if abs(cert2.back_action_db) <= _BACK_ACTION_MAX_DB else 0.0
        liouville_integrity = 1.0 if cert2.liouville_preservation_error <= _LIOUVILLE_PRESERVATION_MAX else 0.0
        return {
            "witness_qnd_status": (
                "OBSERVACIÓN DE CERO DEMOLICIÓN CUÁNTICA DE ADJUNCIÓN (0.0 dB)"
                if passive_silent_ok
                else "ALERTA: INVARIANTE PASSIVE_SILENT VIOLADO"
            ),
            "uhlmann_fidelity": cert2.uhlmann_fidelity,
            "bures_distance": cert2.bures_distance,
            "galois_gap_delta": cert1.galois_gap_delta,
            "unit_eta_norm": cert1.unit_eta_norm,
            "counit_epsilon_norm": cert1.counit_epsilon_norm,
            "connection_A_norm": cert2.connection_A_norm,
            "poincare_crossings": cert2.poincare_crossings_count,
            "mean_return_time": cert2.mean_return_time,
            "kam_stability_index": cert2.kam_stability_index,
            "kam_resonance_violations": cert2.kam_resonance_violations,
            "kam_worst_ratio": cert2.kam_worst_ratio,
            "birkhoff_anharmonicity": cert2.birkhoff_quadratic_coefficient,
            "birkhoff_frobenius": cert2.birkhoff_matrix_frobenius,
            "lindstedt_correction_norm": cert2.lindstedt_correction_norm,
            "twist_index": cert2.twist_index,
            "rotation_number": cert2.rotation_number,
            "isoenergetic_nondegeneracy": cert2.isoenergetic_nondegeneracy,
            "poincare_lyapunov_max": cert2.poincare_lyapunov_max,
            "liouville_preservation_error": cert2.liouville_preservation_error,
            "phase_space_volume": cert1.phase_space_volume,
            "von_neumann_entropy": cert1.von_neumann_entropy,
            "kolmogorov_nondegeneracy": cert1.kolmogorov_nondegeneracy,
            "meta_consistency": meta["consistency"],
            "meta_entropy": meta["entropy"],
            "n3_cycle_count": self._ooda_ledger.cycle_count,
            "n3_trajectory": list(self._ooda_ledger.trajectory_window()),
            "n3_engine_recurrence": cert2.n3_recurrence_index,
            "n3_engine_lyapunov": cert2.n3_adaptation_lyapunov,
            "ooda_recurrence_index": ooda_rec,
            "ooda_lyapunov": ooda_lyap,
            "adaptive_fidelity_floor": cert2.adaptive_fidelity_floor,
            "adaptive_gap_ceiling": cert2.adaptive_gap_ceiling,
            "adaptation_rate": cert2.adaptation_rate,
            "meta_adaptation_rate": cert2.meta_adaptation_rate,
            "isometry_margin": isometry_margin,
            "qnd_integrity": qnd_integrity,
            "liouville_integrity": liouville_integrity,
            "kam_certified_fraction": cert2.kam_stability_index,
            "executive_action": (
                "CERTIFICACIÓN SILENCIOSA DE ADJUNCIÓN EN RAM (0.0 dB)"
                if passive_silent_ok
                else "ESCALADO EJECUTIVO: REVISAR INVARIANTE QND"
            ),
            "agent_version": _AGENT_VERSION,
        }

    # ── Sub-capa 3.3 ──
    def execute_phase3_v12(
        self,
        cert1: DeRhamGaloisWitnessPhase1CertificateV12,
        cert2: DeRhamGaloisWitnessPhase2CertificateV12,
    ) -> TopologicalDeRhamGaloisWitnessSovereignStateV12:
        """
        Sub-capa 3.3 — Emite el Estado Soberano v12.1 + órbita Poincaré del OODA
        + PASSIVE_SILENT verificado (back-action ∧ deriva de Liouville ∧ |det U|−1).

        Primera instrucción: Φ₂₃ = `continue_to_phase3_witness_gate_v12`.
        """
        now = time.time()
        if not self.continue_to_phase3_witness_gate_v12(cert2):
            raise ValueError("Falla de autenticidad en Compuerta Φ₂₃ del Testigo v12.1")

        snapshot = self._session_snapshot
        meta = self._extract_phase3_meta_witness(snapshot)

        base_verdict = cert1.isometry_verdict.meet(cert2.qnd_verdict)
        final_verdict = self._adjudicate_celestial(
            base_verdict=base_verdict,
            kam_index=cert2.kam_stability_index,
            meta_consistency=float(meta["consistency"]),
            twist_index=cert2.twist_index,
            isoenergetic=cert2.isoenergetic_nondegeneracy,
            poincare_lyapunov=cert2.poincare_lyapunov_max,
            liouville_u=cert2.liouville_preservation_error,
            kolmogorov=cert1.kolmogorov_nondegeneracy,
        )

        passive_silent_ok = (
            abs(cert2.back_action_db) <= _BACK_ACTION_MAX_DB
            and cert1.liouville_drift <= _LIOUVILLE_DRIFT_MAX
            and cert2.liouville_preservation_error <= _LIOUVILLE_PRESERVATION_MAX
        )

        self._ooda_ledger.observe(
            kam=cert2.kam_stability_index,
            consistency=float(meta["consistency"]),
            verdict=final_verdict,
            twist=cert2.twist_index,
        )
        ooda_rec = self._ooda_ledger.recurrence_index()
        ooda_lyap = self._ooda_ledger.lyapunov()
        ooda_ent = self._ooda_ledger.verdict_entropy()

        state_id = f"DERHAM-WITNESS-V12-STATE-{int(now * 1000.0) % 1000000:06d}"
        merkle_seed = (
            f"{state_id}:{cert1.hmac_p1_signature}:{cert2.hmac_p2_signature}:"
            f"{int(final_verdict)}:{int(base_verdict)}:{meta['consistency']:.6f}:"
            f"{cert2.kam_stability_index:.6f}:{cert2.twist_index:.8e}:"
            f"{int(passive_silent_ok)}"
        ).encode("utf-8")
        merkle_root = hmac.new(self.secret_key, merkle_seed, hashlib.sha256).hexdigest()

        prov_payload = (
            f"{merkle_root}:{cert2.back_action_db:.1f}:{final_verdict.value}:"
            f"{meta['consistency']:.6f}:{cert2.kam_stability_index:.6f}:"
            f"{ooda_rec:.6f}:{ooda_lyap:+.6e}:{self._ooda_ledger.cycle_count}"
        )
        prov_sig = self._compute_hmac(_HMAC_DOMAIN_P3, prov_payload)

        visceral_report = self._build_visceral_report(
            cert1, cert2, meta, passive_silent_ok, ooda_rec, ooda_lyap
        )

        return TopologicalDeRhamGaloisWitnessSovereignStateV12(
            sovereign_state_id=state_id,
            phase1_cert=cert1,
            phase2_cert=cert2,
            global_verdict=final_verdict,
            back_action_db=cert2.back_action_db,
            merkle_sha256_root=merkle_root,
            hmac_provenance_signature=prov_sig,
            visceral_report=visceral_report,
            timestamp_utc=now,
            meta_verdict_entropy=float(meta["entropy"]),
            meta_consistency_score=float(meta["consistency"]),
            meta_fidelity_std=float(meta["fid_std"]),
            meta_gap_std=float(meta["gap_std"]),
            kam_adjusted_verdict=final_verdict,
            n3_verdict_trajectory=self._ooda_ledger.trajectory_window(),
            n3_cycle_count=self._ooda_ledger.cycle_count,
            passive_silent_invariant_verified=passive_silent_ok,
            ooda_recurrence_index=ooda_rec,
            ooda_lyapunov=ooda_lyap,
            ooda_verdict_entropy=ooda_ent,
            twist_index=cert2.twist_index,
            isoenergetic_nondegeneracy=cert2.isoenergetic_nondegeneracy,
            poincare_lyapunov_max=cert2.poincare_lyapunov_max,
            liouville_preservation_error=cert2.liouville_preservation_error,
            kolmogorov_nondegeneracy=cert1.kolmogorov_nondegeneracy,
            von_neumann_entropy=cert1.von_neumann_entropy,
            agent_version=_AGENT_VERSION,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO TESTIGO — ORQUESTADOR OODA v12.1
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class DeRhamGaloisAdjunctionBridgeWitnessAgentV12(DeRhamGaloisWitnessPhase3DeciderV12):
    """
    Soberano Agéntico Testigo Silencioso v12.1.
    Orquesta el ciclo OODA pasivo completo sobre el isomorfismo F ⊣ G,
    con Mecánica Celeste de Poincaré y Automejora Recursiva N3 / N3-celeste.

    Cadena formal:
        FASE 1.1 → snapshot cacheado
        FASE 1.2 → invariantes acción-ángulo
        FASE 1.3 → P1 HMAC
        Φ₁₂ = continue_to_phase2_witness_gate_v12     ← último método F1 = primer acto F2
        FASE 2.1/2.2/2.3 → P2 HMAC
        Φ₂₃ = continue_to_phase3_witness_gate_v12     ← último método F2 = primer acto F3
        FASE 3.1/3.2/3.3 → SovereignStateV12
    """

    def witness_adjunction_bridge_v12(
        self,
        cartridges: Sequence[Any] = (),
        target_density_matrix: Optional[NDArray[Any]] = None,
        phi_initial: Optional[NDArray[Any]] = None,
        phi_final: Optional[NDArray[Any]] = None,
    ) -> TopologicalDeRhamGaloisWitnessSovereignStateV12:
        """
        Ejecuta el ciclo OODA pasivo completo del testigo v12.1:
          FASE 1 (Observe) → Φ₁₂ → FASE 2 (Orient) → Φ₂₃ → FASE 3 (Decide&Act).
        El motor v12.1 se invoca EXACTAMENTE UNA VEZ (sub-capa 1.1).
        """
        cert1 = self.execute_phase1_v12(
            cartridges=cartridges,
            target_density_matrix=target_density_matrix,
            phi_initial=phi_initial,
            phi_final=phi_final,
        )
        cert2 = self.execute_phase2_v12(
            cert1=cert1,
            cartridges=cartridges,
            target_density_matrix=target_density_matrix,
            phi_initial=phi_initial,
            phi_final=phi_final,
        )
        return self.execute_phase3_v12(cert1=cert1, cert2=cert2)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# Alias de compatibilidad v11 → v12.1 (el código v11 sigue funcionando)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

DeRhamGaloisWitnessPhase1Observer = DeRhamGaloisWitnessPhase1ObserverV12
DeRhamGaloisWitnessPhase2Orientor = DeRhamGaloisWitnessPhase2OrientorV12
DeRhamGaloisWitnessPhase3Decider = DeRhamGaloisWitnessPhase3DeciderV12
DeRhamGaloisAdjunctionBridgeWitnessAgent = DeRhamGaloisAdjunctionBridgeWitnessAgentV12


if __name__ == "__main__":
    banner = "═" * 92
    print(banner)
    print("  VERIFICACIÓN DOCTORAL v12.1 · AGENTE OODA + POINCARÉ + N3 (3 fases × 3 sub-capas)")
    print(banner)

    agent = DeRhamGaloisAdjunctionBridgeWitnessAgentV12(dimension=56)
    rng = np.random.default_rng(2026)
    state = rng.normal(0.0, 1.0, size=(56, 56)) + 1j * rng.normal(0.0, 1.0, size=(56, 56))
    rho = state @ state.conj().T
    rho /= np.trace(rho)

    last_state = None
    for cycle in range(3):
        last_state = agent.witness_adjunction_bridge_v12(target_density_matrix=rho)
        assert last_state is not None
        print(f"\n  ─── CICLO N3 #{cycle + 1} ───")
        print(f"  Sovereign State ID   : {last_state.sovereign_state_id}")
        print(f"  Versión agente       : {last_state.agent_version}")
        print(f"  Veredicto base       : {last_state.phase1_cert.isometry_verdict.name} ⊓ "
              f"{last_state.phase2_cert.qnd_verdict.name}")
        print(f"  Veredicto celestial  : {last_state.kam_adjusted_verdict.name}")
        print(f"  PASSIVE_SILENT ✓     : {last_state.passive_silent_invariant_verified}")
        print(f"  Back-Action QND      : {last_state.back_action_db:.1f} dB")
        print(f"  |det U|−1 Liouville  : {last_state.liouville_preservation_error:.6e}")
        print(f"  Fidelidad Uhlmann    : {last_state.phase2_cert.uhlmann_fidelity:.6f}")
        print(f"  Bures d_B            : {last_state.phase2_cert.bures_distance:.6f}")
        print(f"  Galois Gap Δ         : {last_state.phase1_cert.galois_gap_delta:.8e}")
        print(f"  Acción total ΣJ_i    : {last_state.phase1_cert.action_total:.6f}")
        print(f"  S_vonNeumann         : {last_state.von_neumann_entropy:.6f}")
        print(f"  Kolmogorov det B     : {last_state.kolmogorov_nondegeneracy:.6e}")
        print(f"  Lyapunov (max,min)   : ({last_state.phase1_cert.lyapunov_max:+.4e}, "
              f"{last_state.phase1_cert.lyapunov_min:+.4e})")
        print(f"  Poincaré crossings   : {last_state.phase2_cert.poincare_crossings_count}")
        print(f"  T_retorno medio      : {last_state.phase2_cert.mean_return_time:.6e}")
        print(f"  KAM index            : {last_state.phase2_cert.kam_stability_index:.6f} "
              f"(viol={last_state.phase2_cert.kam_resonance_violations})")
        print(f"  Twist Poincaré       : {last_state.twist_index:.6e}")
        print(f"  Nº rotación ϖ        : {last_state.phase2_cert.rotation_number:.6f}")
        print(f"  Isoenergética Δ      : {last_state.isoenergetic_nondegeneracy:.6e}")
        print(f"  λ_max(P) Oseledets   : {last_state.poincare_lyapunov_max:+.6e}")
        print(f"  Birkhoff anarm.      : {last_state.phase2_cert.birkhoff_quadratic_coefficient:.6e}")
        print(f"  Meta consistencia    : {last_state.meta_consistency_score:.6f}")
        print(f"  Meta entropía        : {last_state.meta_verdict_entropy:.6f}")
        print(f"  Recurrencia OODA     : {last_state.ooda_recurrence_index:.6f}")
        print(f"  λ_Φ OODA             : {last_state.ooda_lyapunov:+.6e}")
        print(f"  Entropía órbita OODA : {last_state.ooda_verdict_entropy:.6f}")
        print(f"  N3 trayectoria       : {last_state.n3_verdict_trajectory}")
        print(f"  N3 ciclos totales    : {last_state.n3_cycle_count}")
        print(f"  Merkle HMAC          : {last_state.merkle_sha256_root[:40]}...")
        print(f"  HMAC Provenance      : {last_state.hmac_provenance_signature[:40]}...")
    print(banner)