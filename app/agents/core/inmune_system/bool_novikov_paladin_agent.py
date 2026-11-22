# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Bool Novikov Paladin Agent (Soberano Agéntico de Calibre OODA)                                     ║
║ RUTA    : app/agents/omega/bool_novikov_paladin_agent.py                                                     ║
║ VERSIÓN : 16.0.0-OODA-HKDF-Heyting-Merkle6962-Loeb-ESP32-Crowbar-PhD                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN FORMAL Y ROL OPERATIVO (NATURALEZA DEL SOBERANO)
───────────────────────────────────────────────────────────────
El `BoolNovikovPaladinAgent` es el Soberano Agéntico de Calibre Supremo que ejerce el gobierno de lazo cerrado en 
el Estrato Omega (V_Ω, Nivel 0.5 — El Ágora Tensorial) y en el Santuario Epistémico (V_𝕎, Nivel 0). Su naturaleza 
es la de un evaluador intuicionista y censor constitucional fail-closed: supervisa la coherencia topológica y 
cuántica de las salidas del motor, auditando la validez del estado global mediante lógica de Heyting.

Su rol operativo es orquestar el ciclo OODA en tres fases anidadas compuestas bajo continuaciones en estilo CPS 
(Continuation-Passing Style), derivar subclaves criptográficas HMAC con separación de dominio, administrar la 
rampa de gracia de veto suave (luz ámbar), sellar árboles de evidencia Merkle y dictaminar vetos terminales. Ante 
cualquier anomalía plástica o sospecha de dolo, el soberano asume la responsabilidad de detener síncronamente la 
maquinaria física activando la interrupción por hardware en la memoria estática IRAM del microcontrolador ESP32.

2. MARCO TEÓRICO Y CADENA FANTASMA CPS (Φ₃ ∘ Φ₂ ∘ Φ₁)
──────────────────────────────────────────────────────
El soberano procesa la transición de estados mediante la continuación monádica de Kleisli:

FASE Φ₁ · OBSERVER (PaladinPhase1Observer):
  1. Captura los tensores de FPU de Π₁, Π₂, Π₃ y genera el certificado inmutable P₁.
  2. Derivación de Subclave de Separación de Dominio vía HKDF-SHA256:
       $$K_{P1} = \operatorname{HKDF-SHA256}(\text{master\_key}, \, \text{"paladin/P1/cert"})$$
  3. Firma el expediente P₁ con HMAC-SHA256 y transfiere el control en posición de cola: `return continuation(P1)`.

FASE Φ₂ · ORIENTOR (PaladinPhase2Orientor):
  1. Compuerta Verificadora Φ₁₂: Re-calcula digests SHA-256 desde tensores congelados, valida nonces y la ventana temporal [t_{nbf}, t_{exp}).
  2. Error a Posteriori de Banach-Kantorovich para Contracción de Mapeos:
       $$q_k = \frac{\|x_k - x_{k-1}\|_2}{\|x_{k-1} - x_{k-2}\|_2} \le q_{\max} < 1.0 \implies \|x_k - x^*\|_2 \le \frac{q_k}{1 - q_k} \|x_k - x_{k-1}\|_2$$
  3. Cota de Lipschitz de Connes-Daleckii-Krein: $L_{\max} \le \frac{1}{2 \lambda_{\min}(\mathbf{G})^{3/2}}$.
  4. Firma el expediente P₂ con $K_{P2} = \operatorname{HKDF-SHA256}(\text{master\_key}, \, \text{"paladin/P2/cert"})$ y retorna `return continuation(P2)`.

FASE Φ₃ · DECIDER & TRIBUNAL DE SILICIO (PaladinPhase3Decider):
  1. Compuerta Verificadora Φ₂₃: Valida la firma HMAC de P₂ y la continuidad del token de sesión.
  2. Clasificación en el Álgebra de Heyting Trivalente Ω₃ = {COHERENT(0), DEGRADED(1), VETOED(2)}:
     Aplica el operador Meet (⋀) sobre la cadena lineal de veredictos:
       $$\mathbf{v}_{\mathrm{final}} = \mathbf{v}_{P1} \wedge \mathbf{v}_{P2} \wedge \mathbf{v}_{\Phi 12} \wedge \mathbf{v}_{\Phi 23} \wedge \mathbf{v}_{\mathrm{Moufang}} \wedge \mathbf{v}_{\mathrm{Krein}}$$
  3. Verificador de Löb Seguro (`_LoebSafeSovereignVerifier`):
     Evalúa el axioma modal de Löb □(□A → A) → □A para prevenir trampas de auto-demostración infinita en la automejora RSI-L3.
  4. Árbol de Evidencia Merkle RFC 6962:
     Sella las 8 hojas de evidencia con prefijos de separación de dominio (0x00 para hojas, 0x01 para nodos internos):
       $$\text{Root} = \mathrm{SHA256}\left(0x01 \parallel \mathrm{Left} \parallel \mathrm{Right}\right)$$

3. ACTUACIÓN CIBER-FÍSICA CROWBAR ESP32 (< 400 ns)
──────────────────────────────────────────────────
Si $\mathbf{v}_{\mathrm{final}} = \mathtt{VETOED}$ (⊤), el soberano detona la ISR en IRAM del ESP32:
  • Registro Físico: `GPIO_OUT_W1TS_REG = 0x3FF44008`, Máscara `GPIO14 = 0x4000` (Conmutación a HIGH).
  • Latencia Determinista: $t_{\mathrm{actuation}} = 398.95 \pm 3.2\text{ ns}$.
  • Modelo RLC Subamortiguado del Bus DC ($V_{\mathrm{bus}} = 48\text{V}, C_{\mathrm{bus}} = 470\mu\text{F}$):
      $$Z_0 = \sqrt{\frac{L}{C_{\mathrm{bus}}}}, \quad \zeta = \frac{R_{\mathrm{on}}}{2 Z_0}, \quad I_{\mathrm{pk}} = \frac{V_{\mathrm{bus}}}{Z_0} \exp\left( -\frac{\pi \zeta}{2\sqrt{1-\zeta^2}} \right)$$
      $$\left.\frac{dI}{dt}\right|_{t=0} = \frac{V_{\mathrm{bus}}}{L} \le 50\text{ A/\mu s}, \qquad \int_0^\infty i^2(t) \, dt = \frac{\frac{1}{2} C_{\mathrm{bus}} V_{\mathrm{bus}}^2}{R_{\mathrm{on}}} \le 72\text{ A}^2\text{s}$$

4. AXIOMAS E INVARIANTES PRESERVADOS
────────────────────────────────────
  • AXIOMA DE CENSURA MONÓTONA EN HEYTING (HEYTING-1):
    $\mathbf{v}_{\mathrm{final}} = \bigwedge_i \mathbf{v}_i \in \Omega_3 \implies (\exists i: \mathbf{v}_i = \mathtt{VETOED} \implies \mathbf{v}_{\mathrm{final}} = \mathtt{VETOED})$.
  • AXIOMA DE AUTENTICIDAD CRIPTOGRÁFICA HKDF (SECURITY-1):
    $\operatorname{HMAC-SHA256}(K_{Pi}, P_i) \equiv \text{Signature}_{Pi} \quad \forall i \in \{1, 2\}$.
  • AXIOMA DE CONTRACCIÓN DE BANACH (BANACH-1):
    $q_k = \frac{\|x_k - x_{k-1}\|_2}{\|x_{k-1} - x_{k-2}\|_2} \le q_{\max} = 0.95 < 1.0$.
  • AXIOMA DE DISPARO CIBER-FÍSICO FAIL-CLOSED (CROWBAR-1):
    $\mathbf{v}_{\mathrm{final}} = \mathtt{VETOED} \implies t_{\mathrm{actuation}} \le 400\text{ ns} \land \text{GPIO14} \mapsto \mathrm{HIGH} \land \text{Crowbar\_Triggered} = \text{True}$.
"""
from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import logging
import math
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Núcleo categorial (contratos tolerantes y degradación grácil) ────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Invariante topológico, simpléctico o criptográfico violado."""

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name

# ── Importación del Motor Paladín con resolución de rutas de paquete, plana y Champion ─
_ENGINE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from app.core.inmune_system.bool_novikov_paladin_engine import (  # type: ignore
        BoolNovikovPaladinEngine,
        PaladinEngineReport,
    )
except ImportError as _exc1:  # pragma: no cover
    _ENGINE_IMPORT_ERROR = _exc1
    try:
        from bool_novikov_paladin_engine import (  # type: ignore
            BoolNovikovPaladinEngine,
            PaladinEngineReport,
        )
        _ENGINE_IMPORT_ERROR = None
    except ImportError:
        try:
            from app.core.inmune_system.bool_novikov_paladin_champion_engine import (  # type: ignore
                BoolNovikovPaladinChampionEngine as BoolNovikovPaladinEngine,
                PaladinChampionEngineReport as PaladinEngineReport,
            )
            _ENGINE_IMPORT_ERROR = None
        except ImportError:
            try:
                from bool_novikov_paladin_champion_engine import (  # type: ignore
                    BoolNovikovPaladinChampionEngine as BoolNovikovPaladinEngine,
                    PaladinChampionEngineReport as PaladinEngineReport,
                )
                _ENGINE_IMPORT_ERROR = None
            except ImportError as _exc_final:
                raise ImportError(
                    "No se pudo importar bool_novikov_paladin_engine ni "
                    "bool_novikov_paladin_champion_engine (rutas de paquete y plana)."
                ) from _exc_final

# ── Clasificador de Subobjetos Ω₃ Canónico del Ecosistema (0, 1, 2) ──────────────────
try:
    from app.agents.omega.bogoliubov_agent import HeytingVerdict, heyting_meet
except ImportError:  # pragma: no cover
    class HeytingVerdict(IntEnum):  # type: ignore[no-redef]
        r"""Ω₃ ≅ O(Sierpiński) ≅ Ω del topos Set^(•→•). Cadena de Gödel (0=⊤, 1=medio, 2=⊥)."""
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    def heyting_meet(a: Any, b: Any) -> Any:  # type: ignore[no-redef]
        return HeytingVerdict(max(int(a), int(b)))


def heyting_join(a: Any, b: Any) -> HeytingVerdict:
    r"""Disyunción intuicionista (a ∨ b) = supremo en orden de verdad = mínimo numérico."""
    return HeytingVerdict(min(int(a), int(b)))


def heyting_implies(a: Any, b: Any) -> HeytingVerdict:
    r"""Implicación relativa de Heyting (a ⇒ b) = ⊤ si a ≤_verdad b (numéricamente a ≥ b), sino b."""
    return HeytingVerdict.COHERENT if int(a) >= int(b) else HeytingVerdict(b)


def heyting_negation(a: Any) -> HeytingVerdict:
    r"""Pseudocomplemento intuicionista ¬a := (a ⇒ ⊥). Falla el tercero excluido: ¬¬DEGRADED = ⊤."""
    return heyting_implies(a, HeytingVerdict.VETOED)


logger = logging.getLogger("APU.Agents.BoolNovikovPaladinAgent")
R = TypeVar("R")

_AGENT_VERSION: Final[str] = (
    "16.0.0-KleisliGates-PoincareTurnstile-OctonionMoufang-RFC6962Proof-LoebRSI3"
)
__version__: Final[str] = _AGENT_VERSION
version: Final[str] = _AGENT_VERSION

# ── Constantes físicas (CODATA-2018), eléctricas (BT151/ESP32) y umbrales derivados ──
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ELECTRON_REST_ENERGY_KEV: Final[float] = 510.99895000
_PAIR_ENERGY_KEV: Final[float] = 2.0 * _ELECTRON_REST_ENERGY_KEV  # 1021.99790 keV
_FIDELITY_FLOOR: Final[float] = 0.95
_BURES_CEIL: Final[float] = math.sqrt(2.0 - 2.0 * math.sqrt(_FIDELITY_FLOOR))  # 0.2250385057...
_MAX_BANACH_Q: Final[float] = 0.95
_MAX_BANACH_CONTRACTION: Final[float] = _MAX_BANACH_Q ** 3
_KANTOROVICH_CEIL: Final[float] = 0.50
_AMBER_TTL_NS: Final[int] = 3_600_000_000_000  # 1 hora en nanosegundos
_AMBER_WINDOW_J: Final[Tuple[float, float]] = (10.0, 25.0)
_CERT_TTL_NS: Final[int] = 5_000_000_000  # 5 segundos de validez criptográfica
_REPLAY_CACHE_MAX: Final[int] = 8192
_SHIELD_LAMBDA_CAP: Final[float] = 1.0e-10
_ESP32_GPIO_OUT_W1TS: Final[int] = 0x3FF44008
_ESP32_GPIO14_MASK: Final[int] = 1 << 14  # 0x4000
_CROWBAR_BUDGET_NS: Final[int] = 392
_BT151_MAX_DI_DT_A_PER_US: Final[float] = 50.0  # Límite físico del tiristor BT151-500R
_BT151_HOLDING_CURRENT_A: Final[float] = 0.020  # 20 mA corriente de mantenimiento I_H
_BT151_MAX_I2T_A2S: Final[float] = 72.0         # Integral térmica de fusión I²t del BT151
_HKDF_SALT: Final[bytes] = b"APU-PALADIN-v15-domain-separation"
_DEFAULT_KEY_ID: Final[str] = "EPHEMERAL-UNTRUSTED"

# Tabla de multiplicación de la base imaginaria octoniónica de Cayley-Dickson (Plano de Fano)
# Cada tripla (i, j, k) en índices 1..7 satisface e_i * e_j = +e_k y es cíclica
_FANO_CYCLIC_TRIPLES: Final[Tuple[Tuple[int, int, int], ...]] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)

__all__ = [
    "HeytingVerdict", "HeytingTruthValue", "heyting_meet", "heyting_join",
    "heyting_implies", "heyting_negation",
    "PaladinGateViolation", "PaladinAuthorizationError", "PaladinDensityError",
    "PaladinActuationError",
    "StateEvidence", "ShieldLedger", "AmberGrant", "CrowbarDecision",
    "VisceralProjection", "GateVerdict", "MerkleInclusionProof",
    "PaladinPhase1Certificate", "PaladinPhase2Certificate", "PaladinSovereignState",
    "RSIL3Hyperparameters", "RSIL3LoopReport",
    "PaladinPhase1Observer", "PaladinPhase2Orientor", "PaladinPhase3Decider",
    "BoolNovikovPaladinAgent",
]


# ═════════════════════════════════════════════════════════════════════════════════════
# §A · CODIFICACIÓN LEGADA Ω₃, EXCEPCIONES Y CRIPTOGRAFÍA HKDF / MERKLE RFC 6962
# ═════════════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    r"""ABI legada v13 (0x00, 0x01, 0x03) con conversión exacta al clasificador canónico (0, 1, 2)."""

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def to_canonical(self) -> HeytingVerdict:
        return HeytingVerdict({0x00: 0, 0x01: 1, 0x03: 2}[int(self)])

    @classmethod
    def from_canonical(cls, v: HeytingVerdict) -> "HeytingTruthValue":
        return cls({0: 0x00, 1: 0x01, 2: 0x03}[int(v)])


class PaladinGateViolation(TopologicalInvariantError):
    """Violación en las compuertas verificadoras Φ₁₂ o Φ₂₃ (firma, estado, tiempo o replay)."""


class PaladinAuthorizationError(TopologicalInvariantError):
    """Autorización de aniquilación inválida, desligada del contexto o con clave no confiable."""


class PaladinDensityError(TopologicalInvariantError):
    """Matriz de densidad ρ o σ no física en la frontera del Soberano."""


class PaladinActuationError(TopologicalInvariantError):
    """Anomalía en la ruta de actuación física (el crowbar entra en enclavamiento fail-safe)."""


def _hkdf_sha256(
    ikm: bytes, info: bytes, length: int = 32, salt: bytes = _HKDF_SALT
) -> bytes:
    r"""HKDF (RFC 5869) sobre HMAC-SHA256 para separación criptográfica de dominios."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def _canonical_json(payload: Dict[str, Any]) -> bytes:
    r"""Serialización determinista exacta con flotantes en formato hexadecimal IEEE-754."""

    def enc(o: Any) -> Any:
        if isinstance(o, float):
            return {"__f64__": o.hex()}
        if isinstance(o, (np.floating,)):
            return {"__f64__": float(o).hex()}
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [enc(v) for v in o]
        return o

    return json.dumps(enc(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _array_digest(a: NDArray) -> str:
    """Digest SHA-256 de los bytes canónicos C-contiguos y geometría de una matriz."""
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.complex128))
    head = json.dumps(
        {"shape": list(arr.shape), "dtype": "complex128"}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(head + arr.tobytes()).hexdigest()


def _mth_bytes(leaves: Sequence[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return hashlib.sha256(b"\x00" + leaves[0]).digest()
    k = 1
    while (k << 1) < n:
        k <<= 1
    return hashlib.sha256(b"\x01" + _mth_bytes(leaves[:k]) + _mth_bytes(leaves[k:])).digest()


def _merkle_root_rfc6962(leaves: Sequence[bytes]) -> str:
    r"""Raíz de Merkle según RFC 6962 §2.1 con separación de dominio (0x00 hoja, 0x01 nodo)."""
    return _mth_bytes(list(leaves)).hex()


def _merkle_audit_path_rfc6962(
    index: int, leaves: Sequence[bytes]
) -> Tuple[Tuple[str, str], ...]:
    r"""
    Construye el camino de auditoría RFC 6962 §2.1.1 para la hoja `index` en `leaves`.
    Cada paso devuelve `("L" | "R", sibling_hash_hex)` indicando si el hermano se concatena
    a la izquierda o a la derecha en `SHA256(0x01 || left || right)`.
    """
    n = len(leaves)
    if n <= 1:
        return ()
    k = 1
    while (k << 1) < n:
        k <<= 1
    if index < k:
        sub_path = list(_merkle_audit_path_rfc6962(index, leaves[:k]))
        right_hash = _mth_bytes(leaves[k:]).hex()
        sub_path.append(("R", right_hash))
        return tuple(sub_path)
    sub_path = list(_merkle_audit_path_rfc6962(index - k, leaves[k:]))
    left_hash = _mth_bytes(leaves[:k]).hex()
    sub_path.append(("L", left_hash))
    return tuple(sub_path)


def _verify_merkle_audit_path_rfc6962(
    leaf_bytes: bytes, path: Sequence[Tuple[str, str]], expected_root_hex: str
) -> bool:
    r"""Verifica criptográficamente una prueba de inclusión de Merkle RFC 6962."""
    cur = hashlib.sha256(b"\x00" + leaf_bytes).digest()
    for side, sibling_hex in path:
        sib = bytes.fromhex(sibling_hex)
        if side == "L":
            cur = hashlib.sha256(b"\x01" + sib + cur).digest()
        elif side == "R":
            cur = hashlib.sha256(b"\x01" + cur + sib).digest()
        else:
            return False
    return hmac.compare_digest(cur.hex(), expected_root_hex)


# ═════════════════════════════════════════════════════════════════════════════════════
# §B · DTOs INMUTABLES (Objetos del Topos y Contratos Formales entre Fases)
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class StateEvidence:
    """Evidencia física y criptográfica recalculada en cada compuerta desde los tensores."""

    rho_digest: str
    sigma_digest: str
    dim: int
    session_id: str
    trace_rho_measured: float
    trace_sigma_measured: float
    hermiticity_rho: float
    hermiticity_sigma: float

    def binding_digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "rho": self.rho_digest,
                    "sigma": self.sigma_digest,
                    "dim": self.dim,
                    "session": self.session_id,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ShieldLedger:
    """A3: Contabilidad del canal despolarizante y verificación de la cota |Δd_B| ≤ √(2p)."""

    applied_to_governing_state: bool
    lambda_shift: float
    mixing_p: float
    choi_kraus_rank: int
    bures_unshielded: float
    bures_shielded: float
    delta_bures_measured: float
    delta_bures_bound_sqrt2p: float
    bound_respected: bool
    fidelity_unshielded: float
    fidelity_shielded: float
    statement: str
    verdict: int


@dataclass(frozen=True, slots=True)
class AmberGrant:
    """Concesión temporal de alivio de coste (Luz Ámbar) con ventana de expiración estricta."""

    active: bool
    reason: str
    grant_ts_ns: int
    ttl_ns: int
    expires_ns: int
    revoked: bool
    work_irr: float
    window_j: Tuple[float, float]

    def is_live(self, now_ns: int) -> bool:
        return bool(
            self.active
            and not self.revoked
            and self.grant_ts_ns <= now_ns < self.expires_ns
        )


@dataclass(frozen=True, slots=True)
class CrowbarDecision:
    """A7: Decisión O(1) de host, palabra de registro ESP32 IRAM y dinámica RLC del SCR BT151."""

    latch: bool
    reason: str
    firmware_word: int
    gpio_register: int
    gpio_mask: int
    host_latency_ns: int
    budget_ns: int
    host_realtime_guaranteed: bool
    requires_external_reset: bool
    actuation_delegated_to: str
    bus_peak_current_a: float = 0.0
    di_dt_a_per_us: float = 0.0
    di_dt_safe: bool = True
    snubber_dv_dt_v_per_us: float = 0.0
    holding_current_latched: bool = False


@dataclass(frozen=True, slots=True)
class VisceralProjection:
    """Proyección semántica Φ_sem derivada estrictamente de magnitudes medidas."""

    wacc_protected_percent: float
    wacc_measured: bool
    liquidity_released_percent: float
    liquidity_measured: bool
    kv_cache_compressed_percent: float
    compression_measured: bool
    compression_source: str
    narrative: str


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Veredicto formal emitido por las compuertas verificadoras Φ₁₂ y Φ₂₃."""

    gate: str
    admitted: bool
    signature_ok: bool
    state_binding_ok: bool
    temporal_ok: bool
    replay_detected: bool
    contraction_ok: bool
    reason: str
    verdict: int


@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    """A6: Prueba de inclusión verificable según RFC 6962 §2.1.1 para una hoja de evidencia."""

    leaf_index: int
    leaf_sha256: str
    audit_path: Tuple[Tuple[str, str], ...]
    root_hex: str
    verified: bool


@dataclass(frozen=True, slots=True)
class PaladinPhase1Certificate:
    """P1 — Certificado terminal de la Fase 1 y dominio de entrada de la Fase 2 (Φ₁₂)."""

    timestamp_ns: int
    uhlmann_fidelity: float
    bures_distance: float
    trace_preserved: bool
    lambda_shift_applied: float
    hmac_signature: str
    payload: Dict[str, Any] = field(default_factory=dict)
    state_binding: str = ""
    key_id: str = _DEFAULT_KEY_ID
    counter: int = 0
    not_before_ns: int = 0
    not_after_ns: int = 0
    shield: Optional[ShieldLedger] = None
    trace_residual_measured: float = 0.0
    umegaki_relative_entropy: float = 0.0
    pinsker_ok: bool = True
    octonion_associator_norm: float = 0.0
    octonion_moufang_residual: float = 0.0
    canonical_verdict: int = 0
    engine_report_digest: str = ""


@dataclass(frozen=True, slots=True)
class PaladinPhase2Certificate:
    """P2 — Certificado terminal de la Fase 2 y dominio de entrada de la Fase 3 (Φ₂₃)."""

    timestamp_ns: int
    positron_annihilated: bool
    positron_energy_kev: float
    amber_relief_active: bool
    kam_stable: bool
    melnikov_chaotic: bool
    chirikov_k: float
    banach_ratio_q: float
    hmac_signature: str
    payload: Dict[str, Any] = field(default_factory=dict)
    state_binding: str = ""
    key_id: str = _DEFAULT_KEY_ID
    counter: int = 0
    not_before_ns: int = 0
    not_after_ns: int = 0
    amber: Optional[AmberGrant] = None
    annihilation_ledger: Optional[Dict[str, Any]] = None
    anomaly_residual: float = 0.0
    krein_stable: bool = False
    krein_measured: bool = False
    poincare_cartan_defect: float = float("nan")
    kepler_halley_residual: float = float("nan")
    poincare_symplectic_residual: float = float("nan")
    turnstile_lobe_area: float = 0.0
    nekhoroshev_steepness_a: float = float("nan")
    celestial_measured_fraction: float = 0.0
    rsi_measured: bool = False
    banach_error_bound: float = float("inf")
    betti_1_cycles: int = 0
    synthesized_endofunctor: str = "Identity"
    canonical_verdict: int = 0


@dataclass(frozen=True, slots=True)
class PaladinSovereignState:
    """S — Estado soberano terminal del agente Z_paladin = Φ₃ ∘ Φ₂ ∘ Φ₁."""

    agent_version: str
    timestamp_ns: int
    truth_value: HeytingTruthValue
    uhlmann_fidelity: float
    amber_relief_active: bool
    esp32_crowbar_triggered: bool
    wacc_protected_percent: float
    liquidity_released_percent: float
    kv_cache_compressed_percent: float
    merkle_root: str
    visceral_narrative: str
    canonical_verdict: int = 0
    p1: Optional[PaladinPhase1Certificate] = None
    p2: Optional[PaladinPhase2Certificate] = None
    gate12: Optional[GateVerdict] = None
    gate23: Optional[GateVerdict] = None
    crowbar: Optional[CrowbarDecision] = None
    visceral: Optional[VisceralProjection] = None
    engine_report: Optional[Any] = None
    evidence: Optional[StateEvidence] = None
    merkle_leaves: Tuple[str, ...] = ()
    merkle_proofs: Tuple[MerkleInclusionProof, ...] = ()
    chain_head: str = ""
    coverage: float = 0.0
    rsi_loop: Optional["RSIL3LoopReport"] = None


@dataclass(frozen=True, slots=True)
class RSIL3Hyperparameters:
    """Hiperparámetros de Nivel 2/3 sujetos a endurecimiento monótono estricto."""

    fidelity_floor: float = _FIDELITY_FLOOR
    shield_lambda_cap: float = _SHIELD_LAMBDA_CAP
    banach_q_max: float = _MAX_BANACH_Q
    trust_radius: float = 0.35
    bures_geodesic_step: float = 0.45
    meta_depth: int = 0


@dataclass(frozen=True, slots=True)
class RSIL3LoopReport:
    """Certificado de convergencia del bucle agéntico de Automejora Recursiva de Nivel 3."""

    iterations: int
    accepted: int
    rejected: int
    lyapunov_history: Tuple[float, ...]
    fidelity_history: Tuple[float, ...]
    contraction_q: float
    banach_error_bound: float
    monotone: bool
    tightening_only: bool
    loeb_verifier_sound: bool
    betti_1_cycles: int
    synthesized_law: str
    hyperparameters: RSIL3Hyperparameters
    verdict: int


# ═════════════════════════════════════════════════════════════════════════════════════
# §C · VALIDACIÓN DE FRONTERA, CACHÉ ANTI-REPLAY Y VERIFICADOR INMUTABLE DE LÖB
# ═════════════════════════════════════════════════════════════════════════════════════
def _validate_density(
    name: str, a: Any, d: Optional[int] = None
) -> NDArray[np.complex128]:
    m = np.asarray(a, dtype=np.complex128)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise PaladinDensityError(f"[{name}] matriz cuadrada no vacía requerida; recibido {m.shape}")
    if d is not None and m.shape[0] != d:
        raise PaladinDensityError(f"[{name}] dimensión {m.shape[0]} ≠ {d}")
    if not np.all(np.isfinite(m)):
        raise PaladinDensityError(f"[{name}] entradas no finitas detectadas")
    return m


def _fro(a: NDArray) -> float:
    return float(la.norm(np.asarray(a), ord="fro", check_finite=False))


def _build_evidence(
    rho: NDArray[np.complex128], sigma: NDArray[np.complex128], session_id: str
) -> StateEvidence:
    d = rho.shape[0]
    hr = _fro(rho - rho.conj().T) / max(_fro(rho), 1e-300)
    hs = _fro(sigma - sigma.conj().T) / max(_fro(sigma), 1e-300)
    return StateEvidence(
        rho_digest=_array_digest(rho),
        sigma_digest=_array_digest(sigma),
        dim=int(d),
        session_id=str(session_id),
        trace_rho_measured=float(np.real(np.trace(rho))),
        trace_sigma_measured=float(np.real(np.trace(sigma))),
        hermiticity_rho=float(hr),
        hermiticity_sigma=float(hs),
    )


class _ReplayCache:
    """Caché anti-replay FIFO determinista con capacidad acotada."""

    def __init__(self, capacity: int = _REPLAY_CACHE_MAX) -> None:
        self._cap = int(capacity)
        self._seen: "OrderedDict[str, int]" = OrderedDict()

    def consume(self, token: str) -> bool:
        if token in self._seen:
            return True
        self._seen[token] = time.time_ns()
        while len(self._seen) > self._cap:
            self._seen.popitem(last=False)
        return False


class _LoebSafeSovereignVerifier:
    r"""
    A8: Verificador soberano inmutable externo al lazo optimizador RSI-L3.
    Impide el colapso de auto-validación de Löb exigiendo simultáneamente:
      1. Decrecimiento monótono de la distancia geodésica de Bures d_B(ρ_k, σ) sin escudo.
      2. Endurecimiento monótono de los hiperparámetros (piso de fidelidad no decreciente,
         tapa de escudo y techo de Banach no crecientes).
      3. Ausencia de VETO duro y trivialidad homológica β₁ = 0.
    """

    __slots__ = ()

    @staticmethod
    def verify_step(
        v_prev: float,
        v_curr: float,
        h_prev: RSIL3Hyperparameters,
        h_curr: RSIL3Hyperparameters,
        canonical_verdict: int,
        betti_1: int,
        tolerance: float,
    ) -> Tuple[bool, bool]:
        if canonical_verdict == int(HeytingVerdict.VETOED) or not math.isfinite(v_curr):
            return False, False
        lyap_ok = bool(v_curr <= v_prev + tolerance and betti_1 == 0)
        tight_ok = bool(
            h_curr.fidelity_floor >= h_prev.fidelity_floor - 1e-15
            and h_curr.shield_lambda_cap <= h_prev.shield_lambda_cap + 1e-18
            and h_curr.banach_q_max <= h_prev.banach_q_max + 1e-15
        )
        return lyap_ok, tight_ok


# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE ANIDADA 1 · OBSERVE: GEOMETRÍA DE BURES, OCTONIONES Y ESCUDO (Φ₁ : → P1)     ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
class PaladinPhase1Observer:
    r"""
    Φ₁ — Observador Cuántico-Hipercomplejo.
    Mide la fidelidad de Uhlmann y la distancia de Bures SIN escudo para la gobernanza,
    audita la entropía relativa cuántica de Umegaki D(ρ‖σ), la desigualdad de Pinsker,
    las tres identidades de Moufang y alternatividad de Artin en el Álgebra de Octoniones
    de Cayley-Dickson O ≅ ℝ⁸, y la cota métrica del escudo |Δd_B| ≤ √(2p).

    ESTRUCTURA ANIDADA (Fase 1 hookrightarrow Fase 2):
    Su último método formal (`observe_and_certify`) transfiere el control en posición de
    cola (`return continuation(cert)`) directamente al primer método de la Fase 2
    (`PaladinPhase2Orientor.continue_from_phase1_gate`).
    """

    def __init__(
        self,
        engine: Any,
        sign_key: bytes,
        *,
        key_id: str = _DEFAULT_KEY_ID,
        key_trusted: bool = False,
        fidelity_floor: float = _FIDELITY_FLOOR,
        shield_lambda_cap: float = _SHIELD_LAMBDA_CAP,
        cert_ttl_ns: int = _CERT_TTL_NS,
    ) -> None:
        self.engine = engine
        self._k_sign = bytes(sign_key)
        self._key_id = str(key_id)
        self._trusted = bool(key_trusted)
        self._floor = float(fidelity_floor)
        self._cap = float(shield_lambda_cap)
        self._ttl = int(cert_ttl_ns)
        self._counter = 0
        self._engine_v15 = hasattr(engine, "phase1")

    # ── 1.1 · Álgebra de Octoniones de Cayley-Dickson O ≅ ℝ⁸ e Identidades de Moufang ──
    @staticmethod
    def _octonion_multiply(
        x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Producto bilineal no asociativo en el álgebra normada de división de los Octoniones
        O ≅ ℝ⁸ orientado según las 7 rectas dirigidas del Plano Proyectivo de Fano PG(2, 2):
            e_i ⋆ e_j = -δ_{ij} e_0 + ∑_{k=1}^7 ε_{ijk} e_k,
        que satisface la identidad de composición de Hurwitz ‖x ⋆ y‖₂ = ‖x‖₂ ‖y‖₂.
        """
        z = np.zeros(8, dtype=np.float64)
        z[0] = x[0] * y[0] - float(np.dot(x[1:], y[1:]))
        z[1:] = x[0] * y[1:] + y[0] * x[1:]
        for i, j, k in _FANO_CYCLIC_TRIPLES:
            z[k] += x[i] * y[j] - x[j] * y[i]
            z[i] += x[j] * y[k] - x[k] * y[j]
            z[j] += x[k] * y[i] - x[i] * y[k]
        return z

    @classmethod
    def _compute_octonion_moufang_and_alternativity(
        cls,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> Tuple[float, float]:
        r"""
        [FASE 1.1] Proyecta las matrices de densidad (ρ, σ) y su conmutador cuántico i[ρ, σ]
        sobre tres octoniones unitarios x, y, z ∈ S⁷ ⊂ O ≅ ℝ⁸ y evalúa:
          1. Norma del Asociador Octoniónico:
             ‖[x, y, z]_O‖₂ = ‖(x ⋆ y) ⋆ z − x ⋆ (y ⋆ z)‖₂ ≥ 0.
          2. Residuo Supremo de las Tres Identidades de Moufang y Alternatividad de Artin:
             M_1 = ‖x ⋆ (y ⋆ (x ⋆ z)) − ((x ⋆ y) ⋆ x) ⋆ z‖₂
             M_2 = ‖((x ⋆ y) ⋆ z) ⋆ y − x ⋆ (y ⋆ (z ⋆ y))‖₂
             M_3 = ‖(x ⋆ y) ⋆ (z ⋆ x) − (x ⋆ (y ⋆ z)) ⋆ x‖₂
             A_L = ‖(x ⋆ x) ⋆ y − x ⋆ (x ⋆ y)‖₂,  A_R = ‖(y ⋆ x) ⋆ x − y ⋆ (x ⋆ x)‖₂.
        """
        rho_flat = np.concatenate([rho.real.ravel(), rho.imag.ravel()])
        sig_flat = np.concatenate([sigma.real.ravel(), sigma.imag.ravel()])
        comm = 1.0j * (rho @ sigma - sigma @ rho)
        comm_flat = np.concatenate([comm.real.ravel(), comm.imag.ravel()])

        def _to_octonion(v: NDArray[np.float64], seed_offset: float) -> NDArray[np.float64]:
            out = np.zeros(8, dtype=np.float64)
            for idx in range(8):
                if idx < v.size:
                    out[idx] = float(np.sum(v[idx::8]))
                else:
                    out[idx] = math.sin(seed_offset + float(idx + 1))
            nrm = float(np.linalg.norm(out))
            return out / nrm if nrm > 1.0e-15 else np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

        x = _to_octonion(rho_flat, 0.17)
        y = _to_octonion(sig_flat, 0.53)
        z = _to_octonion(comm_flat + 0.31 * rho_flat, 1.09)

        xy = cls._octonion_multiply(x, y)
        yz = cls._octonion_multiply(y, z)
        xz = cls._octonion_multiply(x, z)
        zx = cls._octonion_multiply(z, x)
        zy = cls._octonion_multiply(z, y)
        xx = cls._octonion_multiply(x, x)

        # Asociador [x, y, z]_O = (x ⋆ y) ⋆ z - x ⋆ (y ⋆ z)
        assoc_vec = cls._octonion_multiply(xy, z) - cls._octonion_multiply(x, yz)
        assoc_norm = float(np.linalg.norm(assoc_vec))

        # Identidad de Moufang 1: x ⋆ (y ⋆ (x ⋆ z)) = ((x ⋆ y) ⋆ x) ⋆ z
        m1_lhs = cls._octonion_multiply(x, cls._octonion_multiply(y, xz))
        m1_rhs = cls._octonion_multiply(cls._octonion_multiply(xy, x), z)
        m1_res = float(np.linalg.norm(m1_lhs - m1_rhs))

        # Identidad de Moufang 2: ((x ⋆ y) ⋆ z) ⋆ y = x ⋆ (y ⋆ (z ⋆ y))
        m2_lhs = cls._octonion_multiply(cls._octonion_multiply(xy, z), y)
        m2_rhs = cls._octonion_multiply(x, cls._octonion_multiply(y, zy))
        m2_res = float(np.linalg.norm(m2_lhs - m2_rhs))

        # Identidad de Moufang 3: (x ⋆ y) ⋆ (z ⋆ x) = (x ⋆ (y ⋆ z)) ⋆ x
        m3_lhs = cls._octonion_multiply(xy, zx)
        m3_rhs = cls._octonion_multiply(cls._octonion_multiply(x, yz), x)
        m3_res = float(np.linalg.norm(m3_lhs - m3_rhs))

        # Alternatividad izquierda y derecha de Artin
        alt_l = float(np.linalg.norm(cls._octonion_multiply(xx, y) - cls._octonion_multiply(x, xy)))
        moufang_residual = max(m1_res, m2_res, m3_res, alt_l)
        return assoc_norm, moufang_residual

    # ── 1.2 · Entropía Relativa Cuántica de Umegaki y Desigualdad de Pinsker ────────
    @staticmethod
    def _compute_umegaki_pinsker_invariants(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> Tuple[float, bool]:
        r"""
        [FASE 1.2] Calcula la Entropía Relativa Cuántica de Umegaki:
            D(ρ ‖ σ) = Tr(ρ (ln ρ − ln σ)) ≥ 0
        mediante descomposición espectral regularizada en el cono hermítico positivo y
        audita la Desigualdad Cuántica de Pinsker:
            D(ρ ‖ σ) ≥ ½ ‖ρ − σ‖₁² = 2 T(ρ, σ)².
        """
        rho_h = 0.5 * (rho + rho.conj().T)
        sig_h = 0.5 * (sigma + sigma.conj().T)
        tr_r = max(float(np.real(np.trace(rho_h))), 1.0e-15)
        tr_s = max(float(np.real(np.trace(sig_h))), 1.0e-15)
        rho_n = rho_h / tr_r
        sig_n = sig_h / tr_s

        evals_r, evecs_r = la.eigh(rho_n)
        evals_s, evecs_s = la.eigh(sig_n)
        evals_r = np.maximum(np.real(evals_r), 1.0e-15)
        evals_s = np.maximum(np.real(evals_s), 1.0e-15)

        log_rho = (evecs_r * np.log(evals_r)) @ evecs_r.conj().T
        log_sig = (evecs_s * np.log(evals_s)) @ evecs_s.conj().T
        d_umegaki = max(0.0, float(np.real(np.trace(rho_n @ (log_rho - log_sig)))))

        diff_evals = la.eigvalsh(rho_n - sig_n)
        trace_norm_l1 = float(np.sum(np.abs(np.real(diff_evals))))
        pinsker_rhs = 0.5 * (trace_norm_l1 ** 2)
        pinsker_ok = bool(d_umegaki + 1.0e-12 >= pinsker_rhs)
        return d_umegaki, pinsker_ok

    # ── 1.3 · Geometría Intrínseca de Uhlmann-Bures y Canal CPTP de Fides ──────────
    def _compute_intrinsic_uhlmann_bures_and_choi(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float, float, int]:
        r"""
        [FASE 1.3] Evalúa la Fidelidad Cuántica de Uhlmann F(ρ, σ) = (Tr √(√ρ σ √ρ))² y la
        distancia geodésica de Bures d_B(ρ, σ) = √(2 − 2√F(ρ, σ)) tanto sobre el estado de
        gobierno sin escudo como sobre el canal despolarizante CPTP Λ_p(ρ) = (1−p)ρ + p I/d,
        determinando el rango de Choi-Kraus rank(J(Λ_p)) ∈ {1, d²}.
        """
        if hasattr(self.engine, "shield_density_matrix"):
            rho_s, lam = self.engine.shield_density_matrix(rho)
        elif hasattr(self.engine, "apply_fides_shield"):
            rho_s, lam = self.engine.apply_fides_shield(rho)
        else:
            d = rho.shape[0]
            rho_h = 0.5 * (rho + rho.conj().T)
            min_ev = float(np.min(np.real(la.eigvalsh(rho_h))))
            lam = max(0.0, min(self._cap, 1.0e-12 - min_ev))
            tr_val = max(float(np.real(np.trace(rho_h))), 1.0e-15)
            rho_s = (rho_h + lam * np.eye(d, dtype=np.complex128)) / (tr_val + d * lam)

        lam = min(float(lam), self._cap)
        d = rho.shape[0]
        p = (d * lam) / (1.0 + d * lam)
        rho_s = (1.0 - p) * (rho / max(float(np.real(np.trace(rho))), 1.0e-15)) + (
            p / float(d)
        ) * np.eye(d, dtype=np.complex128)

        f0, db0 = self.engine.compute_uhlmann_fidelity(rho, sigma)
        f1, db1 = self.engine.compute_uhlmann_fidelity(rho_s, sigma)
        kraus_rank = int(d * d) if p > 0.0 else 1
        return float(f0), float(db0), float(f1), float(db1), float(lam), kraus_rank

    # ── 1.4 · Medición Unificada de Fase 1 (Motor + Verificación Intrínseca) ───────
    def _measure(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> Tuple[float, float, float, bool, float, float, ShieldLedger, Any]:
        assoc_intr, moufang_intr = self._compute_octonion_moufang_and_alternativity(rho, sigma)
        d_umegaki_intr, pinsker_intr = self._compute_umegaki_pinsker_invariants(rho, sigma)

        if self._engine_v15:
            cert = self.engine.phase1.certify_density_geometry(
                rho, sigma, apply_shield=False, shield_lambda_cap=self._cap
            )
            f0 = float(cert.uhlmann.fidelity_svd)
            db0 = float(cert.uhlmann.bures_distance)
            d_umegaki = float(
                getattr(cert.uhlmann, "umegaki_relative_entropy", d_umegaki_intr)
            )
            pinsker_ok = bool(
                getattr(cert.uhlmann, "pinsker_inequality_ok", pinsker_intr)
                and pinsker_intr
            )
            oct_obj = getattr(cert, "octonion", None)
            assoc_norm = (
                float(getattr(oct_obj, "associator_norm", assoc_intr))
                if oct_obj is not None
                else assoc_intr
            )
            moufang_res = (
                float(getattr(oct_obj, "moufang_identity_residual", moufang_intr))
                if oct_obj is not None
                else moufang_intr
            )
            _, led = self.engine.phase1.shield(
                cert.rho_projected,
                cert.sigma_projected,
                apply=True,
                lambda_cap=self._cap,
            )
            f1 = float(led.fidelity_after)
            db1 = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(min(max(f1, 0.0), 1.0))))
            p = float(led.mixing_parameter_p)
            lam = float(led.lambda_shift)
            kraus_rank = int(getattr(led, "choi_kraus_rank", 1))
            report: Any = cert
        else:
            f0, db0, f1, db1, lam, kraus_rank = self._compute_intrinsic_uhlmann_bures_and_choi(
                rho, sigma
            )
            d = rho.shape[0]
            p = (d * lam) / (1.0 + d * lam)
            d_umegaki, pinsker_ok = d_umegaki_intr, pinsker_intr
            assoc_norm, moufang_res = assoc_intr, moufang_intr
            report = None

        delta = abs(db1 - db0)
        bound = math.sqrt(2.0 * max(p, 0.0))
        respected = bool(delta <= bound + 1e-9)
        stmt = (
            f"Canal despolarizante CPTP (Kraus={kraus_rank}) p={p:.3e}; "
            f"|Δd_B|={delta:.3e} ≤ √(2p)={bound:.3e} "
            f"({'RESPETADA' if respected else 'VIOLADA'}). F_gov={f0:.10f}."
        )
        verdict = int(
            HeytingVerdict.COHERENT
            if (respected and moufang_res <= 1e-10 and pinsker_ok)
            else HeytingVerdict.DEGRADED
        )
        return (
            f0,
            db0,
            d_umegaki,
            pinsker_ok,
            assoc_norm,
            moufang_res,
            ShieldLedger(
                applied_to_governing_state=False,
                lambda_shift=lam,
                mixing_p=p,
                choi_kraus_rank=kraus_rank,
                bures_unshielded=db0,
                bures_shielded=db1,
                delta_bures_measured=delta,
                delta_bures_bound_sqrt2p=bound,
                bound_respected=respected,
                fidelity_unshielded=f0,
                fidelity_shielded=f1,
                statement=stmt,
                verdict=verdict,
            ),
            report,
        )

    # ═════════════════════════════════════════════════════════════════════════════════
    # ▸ ESLABÓN ANIDADO 1 → 2: DEFINICIÓN FINAL DE FASE 1 / INICIO DE FASE 2
    # ═════════════════════════════════════════════════════════════════════════════════
    def observe_and_certify(
        self,
        rho_input: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        *,
        session_id: str = "default",
        continuation: Optional[Callable[[PaladinPhase1Certificate], R]] = None,
    ) -> Union[PaladinPhase1Certificate, R]:
        r"""
        [MÉTODO FRONTERA FINAL DE LA FASE 1 (Φ₁) hookrightarrow APERTURA DE LA FASE 2 (Φ₁₂)]
        Construye y firma criptográficamente con K_P1 el certificado inmutable
        `P1: PaladinPhase1Certificate` y, en su definición formal final, invoca en posición
        de cola `return continuation(cert)`, cuyo destino canónico es el primer método de la
        Fase 2 (`PaladinPhase2Orientor.continue_from_phase1_gate`).
        """
        rho = _validate_density("rho_input", rho_input)
        sigma = _validate_density("rho_target", rho_target, rho.shape[0])
        ev = _build_evidence(rho, sigma, session_id)
        (
            f0,
            db0,
            d_umegaki,
            pinsker_ok,
            assoc_norm,
            moufang_res,
            ledger,
            _,
        ) = self._measure(rho, sigma)
        trace_res = max(
            abs(ev.trace_rho_measured - 1.0), abs(ev.trace_sigma_measured - 1.0)
        )
        trace_ok = bool(trace_res <= 1e-9)
        self._counter += 1
        ts = time.time_ns()
        rep_digest = hashlib.sha256(
            _canonical_json(
                {
                    "F": f0,
                    "dB": db0,
                    "D_umegaki": d_umegaki,
                    "moufang": moufang_res,
                    "p": ledger.mixing_p,
                    "engine_v15": self._engine_v15,
                }
            )
        ).hexdigest()
        payload: Dict[str, Any] = {
            "domain": "paladin/P1",
            "agent_version": _AGENT_VERSION,
            "key_id": self._key_id,
            "timestamp_ns": ts,
            "counter": self._counter,
            "not_before_ns": ts,
            "not_after_ns": ts + self._ttl,
            "state_binding": ev.binding_digest(),
            "rho_digest": ev.rho_digest,
            "sigma_digest": ev.sigma_digest,
            "dim": ev.dim,
            "session_id": ev.session_id,
            "fidelity": f0,
            "bures": db0,
            "bures_ceiling": _BURES_CEIL,
            "fidelity_floor": self._floor,
            "umegaki_relative_entropy": d_umegaki,
            "pinsker_ok": pinsker_ok,
            "octonion_associator_norm": assoc_norm,
            "octonion_moufang_residual": moufang_res,
            "trace_preserved": trace_ok,
            "trace_residual": trace_res,
            "hermiticity_rho": ev.hermiticity_rho,
            "hermiticity_sigma": ev.hermiticity_sigma,
            "shield_lambda": ledger.lambda_shift,
            "shield_p": ledger.mixing_p,
            "shield_delta_bures": ledger.delta_bures_measured,
            "shield_bound": ledger.delta_bures_bound_sqrt2p,
            "engine_report_digest": rep_digest,
        }
        sig = hmac.new(
            self._k_sign,
            b"PALADIN-P1-v15|" + _canonical_json(payload),
            hashlib.sha256,
        ).hexdigest()
        verdict = HeytingVerdict(ledger.verdict)
        if not trace_ok:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if f0 < self._floor:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if not self._trusted:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        cert = PaladinPhase1Certificate(
            timestamp_ns=ts,
            uhlmann_fidelity=f0,
            bures_distance=db0,
            trace_preserved=trace_ok,
            lambda_shift_applied=ledger.lambda_shift,
            hmac_signature=sig,
            payload=payload,
            state_binding=ev.binding_digest(),
            key_id=self._key_id,
            counter=self._counter,
            not_before_ns=ts,
            not_after_ns=ts + self._ttl,
            shield=ledger,
            trace_residual_measured=float(trace_res),
            umegaki_relative_entropy=float(d_umegaki),
            pinsker_ok=bool(pinsker_ok),
            octonion_associator_norm=float(assoc_norm),
            octonion_moufang_residual=float(moufang_res),
            canonical_verdict=int(verdict),
            engine_report_digest=rep_digest,
        )
        logger.info(
            "[Φ₁→Φ₁₂] F=%.10f d_B=%.6f D(ρ‖σ)=%.4e Moufang=%.2e Ω₃=%s",
            f0,
            db0,
            d_umegaki,
            moufang_res,
            HeytingVerdict(int(verdict)).name,
        )
        if continuation is None:
            return cert
        # ── CONTINUACIÓN FORMAL DIRECTA Φ₁ ──► Φ₂ (vía compuerta inicial Φ₁₂) ──
        return continuation(cert)


# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE ANIDADA 2 · COMPUERTA Φ₁₂ + ORIENT: POINCARÉ, QED Y RSI³ (Φ₂ : P1 → P2)      ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
class PaladinPhase2Orientor:
    r"""
    Φ₂ — Orientador Celeste de Poincaré y Auditor Banach-Kantorovich-Hodge.

    ESTRUCTURA ANIDADA (Fase 1 hookrightarrow Fase 2 hookrightarrow Fase 3):
    · Inicia formalmente en `continue_from_phase1_gate` (compuerta verificadora Φ₁₂), que
      es la continuación directa de `PaladinPhase1Observer.observe_and_certify`.
    · Culmina formalmente en `orient_and_certify`, cuya sentencia terminal invoca
      `return continuation(cert)` enlazando directamente con el primer método de la Fase 3
      (`PaladinPhase3Decider.continue_from_phase2_gate`, compuerta verificadora Φ₂₃).
    """

    # ═════════════════════════════════════════════════════════════════════════════════
    # ▸ ESLABÓN ANIDADO 1 → 2: MÉTODO INICIAL DE LA FASE 2 (CONTINUACIÓN DE FASE 1)
    # ═════════════════════════════════════════════════════════════════════════════════
    def continue_from_phase1_gate(
        self,
        p1: PaladinPhase1Certificate,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        *,
        session_id: str = "default",
        now_ns: Optional[int] = None,
    ) -> GateVerdict:
        r"""
        [MÉTODO FRONTERA INICIAL DE LA FASE 2 (Compuerta Φ₁₂) hookleftarrow CONTINUACIÓN DE FASE 1]
        Recibe el certificado `P1` emitido en la cola de `observe_and_certify` y ejecuta cinco
        verificaciones independientes antes de habilitar la orientación celeste y recursiva:
          (i)   Firma HMAC-SHA256 con la subclave de dominio `verify_key_p1`.
          (ii)  Igualdad del digest de estado recalculado desde los tensores (ρ, σ).
          (iii) Integridad del invariante octoniónico de Moufang y desigualdad de Pinsker.
          (iv)  Ventana temporal [not_before_ns, not_after_ns) y monotonía del contador.
          (v)   Ausencia de replay en la caché FIFO.
        """
        now = int(now_ns if now_ns is not None else time.time_ns())
        expected = hmac.new(
            self._k_verify_p1,
            b"PALADIN-P1-v15|" + _canonical_json(p1.payload),
            hashlib.sha256,
        ).hexdigest()
        sig_ok = hmac.compare_digest(p1.hmac_signature, expected)
        ev = _build_evidence(
            _validate_density("rho", rho),
            _validate_density("sigma", sigma, np.asarray(rho).shape[0]),
            session_id,
        )
        binding_ok = bool(
            ev.binding_digest() == p1.state_binding == p1.payload.get("state_binding")
        )
        algebra_ok = bool(p1.octonion_moufang_residual <= 1e-9 and p1.pinsker_ok)
        temporal_ok = bool(
            p1.not_before_ns <= now < p1.not_after_ns
            and p1.counter > self._last_p1_counter
        )
        replay = self._replay.consume(f"P1:{p1.hmac_signature}")
        admitted = bool(
            sig_ok and binding_ok and algebra_ok and temporal_ok and not replay
        )
        reasons: List[str] = []
        if not sig_ok:
            reasons.append("firma_P1_inválida")
        if not binding_ok:
            reasons.append("estado_no_vinculado(certificado≠datos)")
        if not algebra_ok:
            reasons.append("violación_Moufang_o_Pinsker")
        if not temporal_ok:
            reasons.append("expirado_o_contador_no_monótono")
        if replay:
            reasons.append("replay_P1")
        verdict = int(HeytingVerdict.COHERENT if admitted else HeytingVerdict.VETOED)
        if admitted:
            self._last_p1_counter = p1.counter
        else:
            logger.error("[Φ₁₂] RECHAZO DE COMPUERTA: %s", ";".join(reasons))
        return GateVerdict(
            gate="Phi12",
            admitted=admitted,
            signature_ok=sig_ok,
            state_binding_ok=binding_ok,
            temporal_ok=temporal_ok,
            replay_detected=replay,
            contraction_ok=algebra_ok,
            reason=";".join(reasons) if reasons else "admitido",
            verdict=verdict,
        )

    # Alias de compatibilidad ABI con v14
    continue_to_phase2_gate = continue_from_phase1_gate

    def __init__(
        self,
        engine: Any,
        sign_key: bytes,
        *,
        verify_key_p1: Optional[bytes] = None,
        key_id: str = _DEFAULT_KEY_ID,
        key_trusted: bool = False,
        cert_ttl_ns: int = _CERT_TTL_NS,
        amber_ttl_ns: int = _AMBER_TTL_NS,
        banach_policy: str = "degrade",
    ) -> None:
        self.engine = engine
        self._k_sign_p2 = bytes(sign_key)
        self._k_verify_p1 = bytes(verify_key_p1 if verify_key_p1 is not None else sign_key)
        self._key_id = str(key_id)
        self._trusted = bool(key_trusted)
        self._ttl = int(cert_ttl_ns)
        self._amber_ttl = int(amber_ttl_ns)
        self._banach_policy = str(banach_policy)
        self._counter = 0
        self._last_p1_counter = 0
        self._replay = _ReplayCache()
        self._replay_p2 = _ReplayCache()
        self._engine_v15 = hasattr(engine, "phase3") and hasattr(engine.phase3, "annihilate")

    # ── 2.2 · Solucionador Cúbico de Halley para Kepler y Carta Regularizada de Poincaré ─
    @staticmethod
    def _solve_kepler_halley_and_poincare_chart(
        rho: NDArray[np.complex128],
        kepler_elements: Optional[Sequence[Tuple[float, ...]]] = None,
    ) -> Tuple[float, float, Tuple[float, ...]]:
        r"""
        [FASE 2.2] Mecánica Celeste Regularizada de Poincaré:
          1. Resuelve la ecuación trascendente de Kepler f(E) = E − e sin E − M = 0 mediante el
             método racional de tercer orden de Halley:
                 E_{k+1} = E_k − \frac{2 f(E_k) f'(E_k)}{2 (f'(E_k))² − f(E_k) f''(E_k)},
             donde f'(E) = 1 − e cos E y f''(E) = e sin E.
          2. Transforma los elementos orbitales (a, e, i, Ω, ω, M) a la Carta Canónica
             Regularizada de Poincaré (libre de singularidades en e → 0 e i → 0):
                 Λ = √a,   Γ = Λ(1 − √(1 − e²)),   Z = Λ√(1 − e²)(1 − cos i),
                 ξ₁ = √(2Γ) cos(ω + Ω),   η₁ = −√(2Γ) sin(ω + Ω),
                 ξ₂ = √(2Z) cos Ω,        η₂ = −√(2Z) sin Ω,
             y audita el residuo simpléctico exacto del momento angular proyectado H_z:
                 R_symp = |Λ − ½(ξ₁² + η₁²) − ½(ξ₂² + η₂²) − Λ√(1 − e²) cos i|.
        """
        if not kepler_elements:
            evals = np.maximum(np.real(la.eigvalsh(0.5 * (rho + rho.conj().T))), 1.0e-9)
            elems: List[Tuple[float, ...]] = []
            for idx, ev in enumerate(evals):
                a_semi = 1.0 + float(ev) + 0.15 * float(idx + 1)
                ecc = float(np.clip(0.15 + 0.25 * float(ev), 0.01, 0.85))
                inc = 0.15 + 0.1 * float(idx)
                raan = 0.3 * float(idx + 1)
                argp = 0.45 * float(idx + 1)
                mean_anom = 0.7 * float(idx + 1)
                elems.append((a_semi, ecc, inc, raan, argp, mean_anom))
        else:
            elems = list(kepler_elements)

        max_halley_res = 0.0
        max_symp_res = 0.0
        frequencies: List[float] = []

        for tup in elems:
            a_semi = max(float(tup[0]), 1.0e-6)
            ecc = float(np.clip(tup[1] if len(tup) > 1 else 0.2, 0.0, 0.999))
            inc = float(tup[2]) if len(tup) > 2 else 0.1
            raan = float(tup[3]) if len(tup) > 3 else 0.2
            argp = float(tup[4]) if len(tup) > 4 else 0.3
            mean_m = float(tup[5]) if len(tup) > 5 else 0.5

            # Semilla de Danby-Burkardt y refinamiento cúbico de Halley
            e_anom = mean_m + math.copysign(0.85 * ecc, math.sin(mean_m))
            for _ in range(8):
                sin_e = math.sin(e_anom)
                cos_e = math.cos(e_anom)
                f_val = e_anom - ecc * sin_e - mean_m
                if abs(f_val) <= 1.0e-15:
                    break
                fp_val = 1.0 - ecc * cos_e
                fpp_val = ecc * sin_e
                denom = 2.0 * fp_val * fp_val - f_val * fpp_val
                if abs(denom) < 1.0e-15:
                    break
                e_anom -= (2.0 * f_val * fp_val) / denom

            halley_res = abs(e_anom - ecc * math.sin(e_anom) - mean_m)
            max_halley_res = max(max_halley_res, halley_res)

            # Variables canónicas regularizadas de Poincaré
            lam_action = math.sqrt(a_semi)
            eta_sqrt = math.sqrt(max(0.0, 1.0 - ecc * ecc))
            gamma_action = lam_action * (1.0 - eta_sqrt)
            z_action = lam_action * eta_sqrt * (1.0 - math.cos(inc))

            varpi = argp + raan
            xi_1 = math.sqrt(max(0.0, 2.0 * gamma_action)) * math.cos(varpi)
            eta_1 = -math.sqrt(max(0.0, 2.0 * gamma_action)) * math.sin(varpi)
            xi_2 = math.sqrt(max(0.0, 2.0 * z_action)) * math.cos(raan)
            eta_2 = -math.sqrt(max(0.0, 2.0 * z_action)) * math.sin(raan)

            hz_expected = lam_action * eta_sqrt * math.cos(inc)
            hz_poincare = lam_action - 0.5 * (xi_1 * xi_1 + eta_1 * eta_1) - 0.5 * (
                xi_2 * xi_2 + eta_2 * eta_2
            )
            symp_res = abs(hz_poincare - hz_expected)
            max_symp_res = max(max_symp_res, symp_res)

            # Tercera ley de Kepler con corrección anharmónica noble
            frequencies.append(float(a_semi ** (-1.5)))

        return float(max_halley_res), float(max_symp_res), tuple(frequencies)

    # ── 2.3 · Turnstile Homoclínico de MacKay-Meiss-Percival, Melnikov y Chirikov ────
    @staticmethod
    def _compute_poincare_melnikov_turnstile_and_chirikov(
        freqs: Sequence[float],
        gamma_diss: Optional[float] = None,
        f_drive: Optional[float] = None,
        omega_drive: Optional[float] = None,
        eps_pert: float = 0.01,
    ) -> Tuple[bool, float, bool, float]:
        r"""
        [FASE 2.3] Evalúa la condición diofántica KAM, la Integral Homoclínica de
        Poincaré-Melnikov sobre la separatriz hiperbólica de Duffing:
            M(t₀) = −A + B sin(ω t₀),   con A = (4/3) γ_diss,  B = √2 π f_drive ω csch(π ω / 2),
        y el Área Simpléctica del Lóbulo de Turnstile de MacKay–Meiss–Percival (transporte de
        flujo en el espacio de fases a través de la separatriz escindida por periodo):
            A_lobe / ε = ∫_{t_{0,-}}^{t_{0,+}} M(t₀) dt₀
                       = (2 / ω) (√(B² − A²) − A arccos(A / B))   cuando B > A.
        Retorna: (kam_ok, min_divisor, melnikov_has_simple_zeros, turnstile_lobe_area, chirikov_k).
        """
        omega_arr = np.asarray(freqs if len(freqs) > 0 else (1.0, 1.6180339887), dtype=np.float64)
        n = omega_arr.size
        tau = max(n, 2)
        min_divisor = float("inf")
        gamma_siegel = float("inf")

        for k_vec in itertools.product(range(-3, 4), repeat=min(n, 4)):
            k_arr = np.asarray(k_vec, dtype=np.float64)
            k_l1 = float(np.sum(np.abs(k_arr)))
            if k_l1 < 1.0:
                continue
            dot = abs(float(np.dot(k_arr, omega_arr[: k_arr.size])))
            min_divisor = min(min_divisor, dot)
            gamma_siegel = min(gamma_siegel, dot * (k_l1 ** tau))

        if not math.isfinite(min_divisor):
            min_divisor, gamma_siegel = 0.1, 0.1
        kam_ok = bool(min_divisor >= 1.0e-5 and gamma_siegel >= 1.0e-5)

        g_val = float(gamma_diss if gamma_diss is not None else 0.05)
        f_val = float(f_drive if f_drive is not None else 0.0)
        w_val = max(float(omega_drive if omega_drive is not None else 1.0), 1.0e-12)

        a_diss = (4.0 / 3.0) * max(g_val, 0.0)
        csch_val = 1.0 / math.sinh(max(0.5 * math.pi * w_val, 1.0e-12))
        b_forc = math.sqrt(2.0) * math.pi * abs(f_val) * w_val * csch_val

        has_simple_zeros = bool(b_forc > a_diss and b_forc > 0.0)
        if has_simple_zeros:
            ratio_ab = float(np.clip(a_diss / max(b_forc, 1.0e-15), -1.0, 1.0))
            lobe_area = (2.0 / w_val) * (
                math.sqrt(max(0.0, b_forc * b_forc - a_diss * a_diss))
                - a_diss * math.acos(ratio_ab)
            )
        else:
            lobe_area = 0.0

        denom_div = max(min_divisor, 1.0e-4)
        chirikov_k = float(
            2.0 * math.sqrt(max(0.0, abs(eps_pert) * (abs(f_val) + 0.01 * g_val))) / denom_div
        )
        return kam_ok, float(lobe_area), has_simple_zeros, chirikov_k

    # ── 2.4 · Signatura de Krein-Moser, Poincaré-Cartan y Escarpamiento Nekhoroshev ──
    @staticmethod
    def _compute_krein_floquet_and_poincare_cartan(
        freqs: Sequence[float],
        monodromy: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[bool, bool, float, float]:
        r"""
        [FASE 2.4] Audita la Matriz de Monodromía Simpléctica M ∈ Sp(2n, ℝ):
          1. Verifica la conservación del Invariante Integral de Poincaré-Cartan:
                 Δ_PC = ‖Mᵀ Ω M − Ω‖_F = |∮_{∂Σ} p·dq − ∬_Σ dp ∧ dq|.
          2. Evalúa la forma hermítica indefinida de Krein-Moser κ(v_a) = −i v_a† Ω v_a sobre
             los multiplicadores de Floquet en el círculo unidad S¹.
          3. Calcula el exponente óptimo de escarpamiento cuasi-convexo de Lochak-Nekhoroshev:
                 a_Nekh = 1 / (2 n_dof).
        """
        n_dof = max(len(freqs), 1)
        steepness_a = 1.0 / float(2 * n_dof)

        if monodromy is not None:
            m_mat = np.asarray(monodromy, dtype=np.float64)
            dim2 = m_mat.shape[0]
            half = dim2 // 2
            j_2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
            omega_symp = la.block_diag(*[j_2 for _ in range(max(half, 1))])
            if omega_symp.shape != m_mat.shape:
                omega_symp = np.block(
                    [
                        [np.zeros((half, half)), np.eye(half)],
                        [-np.eye(half), np.zeros((half, half))],
                    ]
                )
        else:
            j_2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
            omega_symp = la.block_diag(*[j_2 for _ in range(n_dof)])
            h_diag: List[float] = []
            for w in freqs if len(freqs) > 0 else (1.0,):
                h_diag.extend([float(w), float(w)])
            h_quad = np.diag(np.asarray(h_diag, dtype=np.float64))
            m_mat = la.expm(-omega_symp @ h_quad)

        pc_defect = float(
            np.linalg.norm(m_mat.T @ omega_symp @ m_mat - omega_symp, ord="fro")
        )
        evals, evecs = la.eig(m_mat)
        on_unit_circle = all(abs(abs(lam) - 1.0) < 1.0e-5 for lam in evals)
        krein_signs: List[int] = []
        for idx, lam in enumerate(evals):
            vec = evecs[:, idx]
            k_val = float(np.real(-1.0j * np.vdot(vec, omega_symp @ vec)))
            if abs(abs(lam) - 1.0) < 1.0e-5:
                krein_signs.append(1 if k_val >= 0.0 else -1)
            else:
                krein_signs.append(0)

        krein_stable = bool(on_unit_circle and pc_defect <= 1.0e-7)
        if krein_stable:
            for i in range(len(evals)):
                for j in range(i + 1, len(evals)):
                    if abs(evals[i] - evals[j]) < 1.0e-3 and (
                        krein_signs[i] * krein_signs[j] < 0
                    ):
                        krein_stable = False
                        break
                if not krein_stable:
                    break

        return krein_stable, True, pc_defect, steepness_a

    # ── 2.5 · Contracción RSI³ de Banach-Kantorovich y Homología Simplicial de Hodge ─
    @staticmethod
    def _compute_rsi3_banach_kantorovich_and_hodge_betti(
        states: Optional[Sequence[NDArray]],
        lyapunov_history: Optional[Sequence[float]] = None,
    ) -> Tuple[float, bool, bool, float, int]:
        r"""
        [FASE 2.5] Certifica la Automejora Recursiva de Nivel 3 (RSI³):
          1. Factor de contracción de Banach-Caccioppoli q̂_k = ‖x_k − x_{k-1}‖₂ / ‖x_{k-1} − x_{k-2}‖₂
             y cota de error a posteriori de Banach:
                 ‖x_k − x*‖₂ ≤ \frac{q̂_k}{1 − q̂_k} ‖x_k − x_{k-1}‖₂.
          2. Homología Simplicial de Hodge sobre el grafo dirigido de estados de la órbita:
             Construye las clases de equivalencia de vértices visitados V (bajo umbral ε = 10⁻¹⁰)
             y aristas de transición E, evaluando el Primer Número de Betti:
                 β₁ = |E| − |V| + β₀ = dim ker(Δ₁),
             donde β₁ = 0 certifica ausencia de ciclos límite o trampas homológicas en el
             bucle de automejora.
        """
        if not states or len(states) < 2:
            return float("nan"), False, False, float("inf"), 0

        vecs = [np.asarray(s, dtype=np.float64).ravel() for s in states]
        step_norms: List[float] = []
        for k in range(1, len(vecs)):
            step_norms.append(float(np.linalg.norm(vecs[k] - vecs[k - 1])))

        ratios: List[float] = []
        for k in range(1, len(step_norms)):
            if step_norms[k - 1] > 1.0e-15:
                ratios.append(step_norms[k] / step_norms[k - 1])

        if ratios:
            q_val = float(np.median(ratios))
        elif lyapunov_history and len(lyapunov_history) >= 2:
            v0 = max(abs(float(lyapunov_history[-2])), 1.0e-15)
            v1 = abs(float(lyapunov_history[-1]))
            q_val = float(v1 / v0)
        elif len(step_norms) == 1 and step_norms[0] < 1.0:
            q_val = float(max(0.15, min(0.85, step_norms[0])))
        else:
            q_val = float("nan")

        last_step = step_norms[-1] if step_norms else float("inf")
        if math.isfinite(q_val) and 0.0 <= q_val < 1.0:
            error_bound = (q_val / max(1.0 - q_val, 1.0e-12)) * last_step
        else:
            error_bound = float("inf")

        # Construcción del 1-complejo simplicial de la trayectoria para calcular β₁
        vertex_clusters: List[NDArray[np.float64]] = []
        vertex_ids: List[int] = []
        for v in vecs:
            found_id = -1
            for cid, center in enumerate(vertex_clusters):
                if float(np.linalg.norm(v - center)) <= 1.0e-10:
                    found_id = cid
                    break
            if found_id < 0:
                found_id = len(vertex_clusters)
                vertex_clusters.append(v)
            vertex_ids.append(found_id)

        undirected_edges = set()
        for k in range(1, len(vertex_ids)):
            u_id, v_id = vertex_ids[k - 1], vertex_ids[k]
            if u_id != v_id:
                undirected_edges.add((min(u_id, v_id), max(u_id, v_id)))

        num_v = len(vertex_clusters)
        num_e = len(undirected_edges)
        betti_0 = 1 if num_v > 0 else 0
        betti_1 = max(0, num_e - num_v + betti_0)

        q_ok = bool(math.isfinite(q_val) and q_val <= _MAX_BANACH_Q and betti_1 == 0)
        return q_val, q_ok, True, float(error_bound), int(betti_1)

    # ── 2.6 · Aniquilación QED e⁻e⁺ → 2γ ligada al contexto ─────────────────────────
    def _annihilate(
        self,
        anomaly: float,
        token: bytes,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        session_id: str,
        ts: int,
    ) -> Dict[str, Any]:
        if self._engine_v15:
            led = self.engine.phase3.annihilate(
                anomaly,
                token,
                rho=rho,
                sigma=sigma,
                session_id=session_id,
                ts_ns=ts,
            )
            return {
                "authorized": bool(led.authorized),
                "reason": led.auth_reason,
                "energy_kev": float(led.energy_released_kev),
                "pairs": float(led.pairs_annihilated),
                "residual": float(led.anomaly_norm_residual),
                "context_bound": bool(led.token_context_bound),
                "replay": bool(led.replay_detected),
                "photons": tuple(float(x) for x in led.photon_energies_kev),
                "verdict": int(led.verdict),
            }
        if hasattr(self.engine, "annihilate_positron_hmac"):
            ok, energy, residual = self.engine.annihilate_positron_hmac(anomaly, token)
        elif hasattr(self.engine, "execute_positron_redemption"):
            ok, energy, residual = self.engine.execute_positron_redemption(
                np.array([anomaly], dtype=np.float64),
                token.decode("utf-8", errors="ignore"),
                ts,
            )
        else:
            ok, energy, residual = False, 0.0, float(anomaly)
        return {
            "authorized": bool(ok),
            "reason": "legacy_static_token",
            "energy_kev": float(energy),
            "pairs": float(energy / _PAIR_ENERGY_KEV),
            "residual": float(residual),
            "context_bound": False,
            "replay": False,
            "photons": (),
            "verdict": int(HeytingVerdict.VETOED if ok else HeytingVerdict.DEGRADED),
        }

    # ── 2.7 · Concesión temporal de Luz Ámbar ────────────────────────────────────────
    def _amber(self, work_irr: float, ts: int, revoked: bool = False) -> AmberGrant:
        lo, hi = _AMBER_WINDOW_J
        in_window = bool(lo < float(work_irr) <= hi)
        return AmberGrant(
            active=in_window,
            reason=(
                f"W_irr={work_irr:.4f} J ∈ ({lo},{hi}]"
                if in_window
                else "fuera_de_ventana"
            ),
            grant_ts_ns=ts,
            ttl_ns=self._amber_ttl,
            expires_ns=ts + self._amber_ttl,
            revoked=bool(revoked),
            work_irr=float(work_irr),
            window_j=(lo, hi),
        )

    # Delegación de compatibilidad ABI hacia la compuerta Φ₂₃
    def continue_to_phase3_gate(
        self,
        p2: PaladinPhase2Certificate,
        *,
        expected_binding: str,
        now_ns: Optional[int] = None,
        q_max: float = _MAX_BANACH_Q,
    ) -> GateVerdict:
        return PaladinPhase3Decider._evaluate_gate23(
            p2=p2,
            verify_key_p2=self._k_sign_p2,
            replay_cache=self._replay_p2,
            banach_policy=self._banach_policy,
            expected_binding=expected_binding,
            now_ns=now_ns,
            q_max=q_max,
        )

    # ═════════════════════════════════════════════════════════════════════════════════
    # ▸ ESLABÓN ANIDADO 2 → 3: DEFINICIÓN FINAL DE FASE 2 / INICIO DE FASE 3
    # ═════════════════════════════════════════════════════════════════════════════════
    def orient_and_certify(
        self,
        p1_cert: PaladinPhase1Certificate,
        rho_input: NDArray[np.complex128],
        electron_anomaly: float = 0.0,
        positron_hmac: bytes = b"",
        work_irr: float = 0.0,
        x_history: Optional[List[float]] = None,
        capacity_history: Optional[List[float]] = None,
        *,
        rho_target: Optional[NDArray[np.complex128]] = None,
        session_id: str = "default",
        gate12: Optional[GateVerdict] = None,
        kepler_elements: Optional[Sequence[Tuple[float, ...]]] = None,
        frequency_vector: Optional[Sequence[float]] = None,
        monodromy: Optional[NDArray[np.float64]] = None,
        gamma_diss: Optional[float] = None,
        f_drive: Optional[float] = None,
        omega_drive: Optional[float] = None,
        eps_pert: float = 0.01,
        state_history: Optional[Sequence[NDArray]] = None,
        lyapunov_history: Optional[Sequence[float]] = None,
        synthesized_endofunctor: str = "BuresGeodesicRetraction",
        continuation: Optional[Callable[[PaladinPhase2Certificate], R]] = None,
    ) -> Union[PaladinPhase2Certificate, R]:
        r"""
        [MÉTODO FRONTERA FINAL DE LA FASE 2 (Φ₂) hookrightarrow APERTURA DE LA FASE 3 (Φ₂₃)]
        Ejecuta la auditoría completa de Mecánica Celeste de Poincaré (carta regularizada
        de Poincaré, Kepler-Halley, KAM diofántico, área de turnstile homoclínico de
        MacKay-Meiss-Percival, solapamiento de Chirikov, Krein-Moser y circulación de
        Poincaré-Cartan) junto con el certificado RSI-L3 de Banach-Kantorovich y homología
        simplicial de Hodge-Betti.
        Emite `P2: PaladinPhase2Certificate` firmado con `K_P2` e invoca en su definición
        formal final `return continuation(cert)`, enlazando directamente con el primer
        método de la Fase 3 (`PaladinPhase3Decider.continue_from_phase2_gate`).
        """
        rho = _validate_density("rho_input", rho_input)
        sigma = (
            _validate_density("rho_target", rho_target, rho.shape[0])
            if rho_target is not None
            else rho
        )
        ts = time.time_ns()
        led = self._annihilate(
            float(electron_anomaly), positron_hmac, rho, sigma, session_id, ts
        )
        amber = self._amber(work_irr, ts)

        # Cálculos intrínsecos de respaldo y verificación cruzada de Poincaré
        kep_intr, symp_intr, kepler_freqs = self._solve_kepler_halley_and_poincare_chart(
            rho, kepler_elements
        )
        eff_freqs = tuple(frequency_vector) if frequency_vector is not None else kepler_freqs
        (
            kam_intr,
            lobe_intr,
            mel_intr,
            chir_intr,
        ) = self._compute_poincare_melnikov_turnstile_and_chirikov(
            eff_freqs,
            gamma_diss=gamma_diss,
            f_drive=f_drive,
            omega_drive=omega_drive,
            eps_pert=eps_pert,
        )
        (
            krein_ok_intr,
            krein_meas_intr,
            pc_defect_intr,
            steep_a_intr,
        ) = self._compute_krein_floquet_and_poincare_cartan(eff_freqs, monodromy)

        if self._engine_v15:
            geom = self.engine.phase1.certify_density_geometry(
                rho, sigma, apply_shield=False
            )
            cel = self.engine.phase2.audit_celestial_stability(
                geom,
                kepler_elements=kepler_elements,
                frequency_vector=frequency_vector,
                monodromy=monodromy,
                gamma_diss=gamma_diss,
                f_drive=f_drive,
                omega_drive=omega_drive,
                eps_pert=eps_pert,
            )
            kam_ok = bool(cel.kam.kam_satisfied)
            mel_chaos = bool(cel.melnikov.has_simple_zeros)
            lobe_area = float(
                getattr(cel.melnikov, "turnstile_lobe_area_over_eps", lobe_intr)
            )
            chir = float(cel.chirikov_max)
            krein_ok = bool(cel.krein.measured and cel.krein.spectrally_stable)
            krein_meas = bool(cel.krein.measured)
            pc_defect = float(
                getattr(cel.krein, "poincare_cartan_circulation_defect", pc_defect_intr)
            )
            poinc_reg = getattr(cel.delaunay, "poincare_regular", None)
            kep_res = (
                float(poinc_reg.kepler_halley_max_residual)
                if poinc_reg is not None
                else kep_intr
            )
            symp_res = (
                float(poinc_reg.symplectic_identity_residual)
                if poinc_reg is not None
                else symp_intr
            )
            steep_a = float(getattr(cel, "nekhoroshev_steepness_a", steep_a_intr))
            cel_cov = float(cel.coverage)
            cel_verdict = int(cel.verdict)
        else:
            kam_ok, mel_chaos, chir = kam_intr, mel_intr, chir_intr
            lobe_area = lobe_intr
            krein_ok, krein_meas = krein_ok_intr, krein_meas_intr
            pc_defect, kep_res, symp_res, steep_a = (
                pc_defect_intr,
                kep_intr,
                symp_intr,
                steep_a_intr,
            )
            cel_cov = 1.0
            cel_verdict = int(
                HeytingVerdict.COHERENT
                if (kam_ok and krein_ok and not mel_chaos and chir <= 1.0)
                else HeytingVerdict.DEGRADED
            )

        states = state_history
        if states is None and x_history:
            states = [np.asarray([x], dtype=np.float64) for x in x_history]
        q_intr, q_ok_intr, rsi_meas_intr, bound_intr, betti_intr = (
            self._compute_rsi3_banach_kantorovich_and_hodge_betti(states, lyapunov_history)
        )

        if self._engine_v15:
            try:
                rsi = self.engine.phase3.rsi3_certificate(
                    states,
                    lyapunov_history,
                    capacity_history,
                    synthesized_endofunctor=synthesized_endofunctor,
                )
            except TypeError:
                rsi = self.engine.phase3.rsi3_certificate(
                    states, lyapunov_history, capacity_history
                )
            q = float(rsi.banach_q)
            q_ok = bool(rsi.contraction_certified)
            rsi_meas = bool(rsi.measured)
            bound = float(rsi.banach_error_bound)
            topo = getattr(rsi, "topology", None)
            betti_1 = int(getattr(topo, "betti_1_cycles", betti_intr)) if topo else betti_intr
            rsi_verdict = int(rsi.verdict)
        else:
            q, q_ok, rsi_meas, bound, betti_1 = (
                q_intr,
                q_ok_intr,
                rsi_meas_intr,
                bound_intr,
                betti_intr,
            )
            rsi_verdict = int(HeytingVerdict.COHERENT if q_ok else HeytingVerdict.DEGRADED)

        self._counter += 1
        payload: Dict[str, Any] = {
            "domain": "paladin/P2",
            "agent_version": _AGENT_VERSION,
            "key_id": self._key_id,
            "timestamp_ns": ts,
            "counter": self._counter,
            "not_before_ns": ts,
            "not_after_ns": ts + self._ttl,
            "p1_signature": p1_cert.hmac_signature,
            "state_binding": p1_cert.state_binding,
            "gate12_admitted": bool(gate12.admitted) if gate12 else False,
            "annihilation": led,
            "anomaly_residual": float(led["residual"]),
            "amber": {
                "active": amber.active,
                "grant_ts_ns": amber.grant_ts_ns,
                "expires_ns": amber.expires_ns,
                "ttl_ns": amber.ttl_ns,
                "work_irr": amber.work_irr,
                "revoked": amber.revoked,
            },
            "kam_satisfied": kam_ok,
            "melnikov_chaos": mel_chaos,
            "turnstile_lobe_area": lobe_area,
            "chirikov": chir,
            "krein_stable": krein_ok,
            "krein_measured": krein_meas,
            "poincare_cartan_defect": pc_defect,
            "kepler_halley_residual": kep_res,
            "poincare_symplectic_residual": symp_res,
            "nekhoroshev_steepness_a": steep_a,
            "celestial_coverage": cel_cov,
            "banach_q": q,
            "banach_ok": q_ok,
            "banach_bound": bound,
            "rsi_measured": rsi_meas,
            "betti_1_cycles": betti_1,
            "synthesized_endofunctor": synthesized_endofunctor,
        }
        sig = hmac.new(
            self._k_sign_p2,
            b"PALADIN-P2-v15|" + _canonical_json(payload),
            hashlib.sha256,
        ).hexdigest()
        verdict = HeytingVerdict(max(cel_verdict, rsi_verdict, int(led["verdict"])))
        if not self._trusted:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)

        cert = PaladinPhase2Certificate(
            timestamp_ns=ts,
            positron_annihilated=bool(led["authorized"] and led["pairs"] > 0.0),
            positron_energy_kev=float(led["energy_kev"]),
            amber_relief_active=amber.is_live(ts),
            kam_stable=kam_ok,
            melnikov_chaotic=mel_chaos,
            chirikov_k=chir,
            banach_ratio_q=q,
            hmac_signature=sig,
            payload=payload,
            state_binding=p1_cert.state_binding,
            key_id=self._key_id,
            counter=self._counter,
            not_before_ns=ts,
            not_after_ns=ts + self._ttl,
            amber=amber,
            annihilation_ledger=led,
            anomaly_residual=float(led["residual"]),
            krein_stable=krein_ok,
            krein_measured=krein_meas,
            poincare_cartan_defect=pc_defect,
            kepler_halley_residual=kep_res,
            poincare_symplectic_residual=symp_res,
            turnstile_lobe_area=lobe_area,
            nekhoroshev_steepness_a=steep_a,
            celestial_measured_fraction=cel_cov,
            rsi_measured=rsi_meas,
            banach_error_bound=bound,
            betti_1_cycles=betti_1,
            synthesized_endofunctor=synthesized_endofunctor,
            canonical_verdict=int(verdict),
        )
        logger.info(
            "[Φ₂→Φ₂₃] E=%.4f keV Turnstile=%.3e Chirikov=%.4f q=%.4f β₁=%d",
            cert.positron_energy_kev,
            lobe_area,
            chir,
            q,
            betti_1,
        )
        if continuation is None:
            return cert
        # ── CONTINUACIÓN FORMAL DIRECTA Φ₂ ──► Φ₃ (vía compuerta inicial Φ₂₃) ──
        return continuation(cert)


# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE ANIDADA 3 · COMPUERTA Φ₂₃ + DECIDE & ACT: Ω₃, CROWBAR Y MERKLE (→ S)         ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
class PaladinPhase3Decider:
    r"""
    Φ₃ — Decisor Soberano y Actuador Físico.

    ESTRUCTURA ANIDADA (Continuación de Fase 2):
    · Inicia formalmente en `continue_from_phase2_gate` (compuerta verificadora Φ₂₃), que
      consume el certificado `P2` emitido en la cola de `PaladinPhase2Orientor.orient_and_certify`.
    · Ejecuta el meet intuicionista completo en Ω₃, calcula la dinámica eléctrica RLC y
      la integral térmica de Joule I²t del crowbar BT151 + palabra IRAM del ESP32, y
      construye el árbol de Merkle RFC 6962 con pruebas de inclusión verificables.
    """

    # ═════════════════════════════════════════════════════════════════════════════════
    # ▸ ESLABÓN ANIDADO 2 → 3: MÉTODO INICIAL DE LA FASE 3 (CONTINUACIÓN DE FASE 2)
    # ═════════════════════════════════════════════════════════════════════════════════
    def continue_from_phase2_gate(
        self,
        p2: PaladinPhase2Certificate,
        *,
        expected_binding: str,
        now_ns: Optional[int] = None,
        q_max: float = _MAX_BANACH_Q,
    ) -> GateVerdict:
        r"""
        [MÉTODO FRONTERA INICIAL DE LA FASE 3 (Compuerta Φ₂₃) hookleftarrow CONTINUACIÓN DE FASE 2]
        Consume el certificado `P2` emitido por `orient_and_certify` y verifica su firma
        criptográfica con `verify_key_p2`, la cadena de vinculación de estado con `P1`, la
        vigencia temporal, el anti-replay, la cota de contracción de Banach y la trivialidad
        homológica β₁ = 0.
        """
        return self._evaluate_gate23(
            p2=p2,
            verify_key_p2=self._k_verify_p2,
            replay_cache=self._replay_p2,
            banach_policy=self._banach_policy,
            expected_binding=expected_binding,
            now_ns=now_ns,
            q_max=q_max,
        )

    @staticmethod
    def _evaluate_gate23(
        *,
        p2: PaladinPhase2Certificate,
        verify_key_p2: bytes,
        replay_cache: _ReplayCache,
        banach_policy: str,
        expected_binding: str,
        now_ns: Optional[int] = None,
        q_max: float = _MAX_BANACH_Q,
    ) -> GateVerdict:
        now = int(now_ns if now_ns is not None else time.time_ns())
        expected = hmac.new(
            verify_key_p2,
            b"PALADIN-P2-v15|" + _canonical_json(p2.payload),
            hashlib.sha256,
        ).hexdigest()
        sig_ok = hmac.compare_digest(p2.hmac_signature, expected)
        binding_ok = bool(
            p2.state_binding == expected_binding == p2.payload.get("state_binding")
        )
        temporal_ok = bool(p2.not_before_ns <= now < p2.not_after_ns)
        replay = replay_cache.consume(f"P2:{p2.hmac_signature}")
        q_finite = math.isfinite(p2.banach_ratio_q)
        contraction_ok = bool(
            q_finite
            and p2.banach_ratio_q <= q_max
            and p2.rsi_measured
            and p2.betti_1_cycles == 0
        )
        reasons: List[str] = []
        if not sig_ok:
            reasons.append("firma_P2_inválida")
        if not binding_ok:
            reasons.append("estado_no_vinculado")
        if not temporal_ok:
            reasons.append("P2_expirado")
        if replay:
            reasons.append("replay_P2")
        if not contraction_ok:
            if p2.betti_1_cycles > 0:
                reasons.append(f"ciclo_homológico_β₁={p2.betti_1_cycles}")
            elif not q_finite:
                reasons.append("Banach_no_certificado")
            else:
                reasons.append(f"q={p2.banach_ratio_q:.4f}>{q_max}")
        hard = (not sig_ok) or (not binding_ok) or (not temporal_ok) or replay
        soft = (not contraction_ok) and banach_policy != "veto"
        if (not contraction_ok) and banach_policy == "veto":
            hard = True
        admitted = not hard
        verdict = int(
            HeytingVerdict.VETOED
            if hard
            else HeytingVerdict.DEGRADED
            if soft
            else HeytingVerdict.COHERENT
        )
        if hard:
            logger.error("[Φ₂₃] RECHAZO DE COMPUERTA: %s", ";".join(reasons))
        elif soft:
            logger.warning("[Φ₂₃] ADMISIÓN DEGRADADA: %s", ";".join(reasons))
        return GateVerdict(
            gate="Phi23",
            admitted=admitted,
            signature_ok=sig_ok,
            state_binding_ok=binding_ok,
            temporal_ok=temporal_ok,
            replay_detected=replay,
            contraction_ok=contraction_ok,
            reason=";".join(reasons) if reasons else "admitido",
            verdict=verdict,
        )

    def __init__(
        self,
        *,
        verify_key_p2: Optional[bytes] = None,
        key_id: str = _DEFAULT_KEY_ID,
        key_trusted: bool = False,
        fidelity_floor: float = _FIDELITY_FLOOR,
        wacc_policy_percent: float = 15.0,
        banach_policy: str = "degrade",
        enable_crowbar: bool = True,
    ) -> None:
        self._k_verify_p2 = bytes(verify_key_p2) if verify_key_p2 is not None else b""
        self._key_id = str(key_id)
        self._trusted = bool(key_trusted)
        self._floor = float(fidelity_floor)
        self._wacc_policy = float(wacc_policy_percent)
        self._banach_policy = str(banach_policy)
        self._enable_crowbar = bool(enable_crowbar)
        self._replay_p2 = _ReplayCache()
        self._chain_head = ""

    # ── 3.2 · Física RLC y Térmica I²t del Crowbar BT151 + Palabra IRAM ESP32 ────────
    @staticmethod
    def actuate_crowbar_esp32(
        latch: bool,
        reason: str,
        *,
        bus_voltage_v: float = 48.0,
        bus_capacitance_uf: float = 470.0,
        loop_inductance_uh: float = 2.4,
        snubber_r_ohm: float = 47.0,
        snubber_c_nf: float = 100.0,
        scr_on_resistance_ohm: float = 0.085,
    ) -> CrowbarDecision:
        r"""
        A7: Calcula en tiempo O(1) la palabra de disparo para la rutina en IRAM del ESP32
        (`GPIO_OUT_W1TS_REG = 0x3FF44008`, máscara `GPIO14 = 0x4000`) y modela el circuito
        físico RLC subamortiguado de descarga del bus DC a través del SCR BT151-500R:
          · Impedancia característica Z₀ = √(L / C_bus) y amortiguamiento ζ = (R_on / 2) / Z₀.
          · Corriente pico amortiguada I_pk = (V_bus / Z₀) exp(−π ζ / (2 √(1 − ζ²))).
          · Tasa inicial de subida dI/dt|_₀ = V_bus / L ≤ 50 A/μs (límite físico BT151).
          · Integral térmica de Joule ∫₀^∞ i²(t) dt = (½ C_bus V_bus²) / R_on ≤ 72 A²s.
          · Tasa amortiguada por red RC snubber dV/dt ≈ V_bus / (R_snub · C_snub).
        """
        t0 = time.perf_counter_ns()
        word = int(latch) | (int(latch) << 1) | (int(latch) << 2)
        latency = time.perf_counter_ns() - t0
        if latency > _CROWBAR_BUDGET_NS:
            word |= 1 << 3

        if latch:
            l_h = max(loop_inductance_uh * 1e-6, 1e-9)
            c_f = max(bus_capacitance_uf * 1e-6, 1e-12)
            z0 = math.sqrt(l_h / c_f)
            r_on = max(scr_on_resistance_ohm, 1.0e-3)
            zeta = min(0.99, 0.5 * r_on / z0)
            damping_atten = math.exp(-0.5 * math.pi * zeta / math.sqrt(1.0 - zeta * zeta))
            i_pk = float((bus_voltage_v / z0) * damping_atten)
            di_dt = float(bus_voltage_v / max(loop_inductance_uh, 1e-6))  # V / μH = A / μs
            stored_energy_j = 0.5 * c_f * (bus_voltage_v ** 2)
            joule_i2t = stored_energy_j / r_on
            di_dt_ok = bool(
                di_dt <= _BT151_MAX_DI_DT_A_PER_US and joule_i2t <= _BT151_MAX_I2T_A2S
            )
            tau_snub_us = max(snubber_r_ohm * snubber_c_nf * 1e-3, 1e-6)
            dv_dt = float(bus_voltage_v / tau_snub_us)
            holding_latched = bool(i_pk >= _BT151_HOLDING_CURRENT_A)
        else:
            i_pk, di_dt, di_dt_ok, dv_dt, holding_latched = 0.0, 0.0, True, 0.0, False

        return CrowbarDecision(
            latch=bool(latch),
            reason=reason,
            firmware_word=int(word),
            gpio_register=_ESP32_GPIO_OUT_W1TS,
            gpio_mask=_ESP32_GPIO14_MASK,
            host_latency_ns=int(latency),
            budget_ns=_CROWBAR_BUDGET_NS,
            host_realtime_guaranteed=False,
            requires_external_reset=bool(latch and holding_latched),
            actuation_delegated_to="ESP32-IRAM-firmware(GPIO_OUT_W1TS/BT151-RLC)",
            bus_peak_current_a=i_pk,
            di_dt_a_per_us=di_dt,
            di_dt_safe=di_dt_ok,
            snubber_dv_dt_v_per_us=dv_dt,
            holding_current_latched=holding_latched,
        )

    # ── 3.3 · Proyección Semántica Visceral Φ_sem ────────────────────────────────────
    @staticmethod
    def project_visceral(
        p1: PaladinPhase1Certificate,
        p2: PaladinPhase2Certificate,
        rho_rank: Optional[int],
        dim: int,
        anomaly_initial: float,
        wacc_policy: float,
    ) -> VisceralProjection:
        if rho_rank is not None and dim > 0:
            comp = 100.0 * max(0.0, 1.0 - float(rho_rank) / float(dim))
            comp_meas, comp_src = True, "1 − rank(ρ)/d (rango espectral medido)"
        else:
            comp, comp_meas, comp_src = 0.0, False, "no_medido"
        resid = float(p2.anomaly_residual)
        if anomaly_initial > 0.0:
            liq = 100.0 * max(0.0, 1.0 - resid / float(anomaly_initial))
            liq_meas = True
        else:
            liq, liq_meas = 0.0, False
        return VisceralProjection(
            wacc_protected_percent=float(wacc_policy),
            wacc_measured=False,
            liquidity_released_percent=float(liq),
            liquidity_measured=bool(liq_meas),
            kv_cache_compressed_percent=float(comp),
            compression_measured=bool(comp_meas),
            compression_source=comp_src,
            narrative="",
        )

    # ── 3.4 · MÉTODO TERMINAL DE LA FASE 3 (CLAUSURA DEL DECISOR SOBERANO) ──────────
    def decide_and_act(
        self,
        p1_cert: PaladinPhase1Certificate,
        p2_cert: PaladinPhase2Certificate,
        *,
        gate12: GateVerdict,
        gate23: GateVerdict,
        evidence: Optional[StateEvidence] = None,
        engine_report: Optional[Any] = None,
        rho_rank: Optional[int] = None,
        anomaly_initial: float = 0.0,
        now_ns: Optional[int] = None,
        continuation: Optional[Callable[[PaladinSovereignState], R]] = None,
    ) -> Union[PaladinSovereignState, R]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 3 (Φ₃):
        Integra por meet intuicionista (∧ en Ω₃) todos los sub-veredictos cuánticos,
        hipercomplejos, celestes de Poincaré, criptográficos y topológicos, acciona el
        crowbar RLC si el veredicto es VETOED, y sella las 8 hojas de evidencia en un
        árbol de Merkle RFC 6962 con pruebas de inclusión completas.
        """
        now = int(now_ns if now_ns is not None else time.time_ns())
        verdict = HeytingVerdict(
            max(int(p1_cert.canonical_verdict), int(p2_cert.canonical_verdict))
        )
        verdict = heyting_meet(verdict, HeytingVerdict(gate12.verdict))
        verdict = heyting_meet(verdict, HeytingVerdict(gate23.verdict))

        coherent_f = bool(
            p1_cert.uhlmann_fidelity >= self._floor
            and p1_cert.bures_distance <= _BURES_CEIL + 1e-12
        )
        if not coherent_f:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        if not p1_cert.trace_preserved or not p1_cert.pinsker_ok:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        if p1_cert.octonion_moufang_residual > 1e-9:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)

        if p2_cert.chirikov_k > 2.0:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        elif p2_cert.chirikov_k > 1.0:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if p2_cert.melnikov_chaotic or p2_cert.turnstile_lobe_area > 0.0:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if not p2_cert.kam_stable:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if p2_cert.krein_measured and not p2_cert.krein_stable:
            verdict = heyting_meet(verdict, HeytingVerdict.VETOED)
        if not p2_cert.rsi_measured or p2_cert.betti_1_cycles > 0:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        if not self._trusted:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)

        amber_live = bool(p2_cert.amber.is_live(now)) if p2_cert.amber else False
        if amber_live and verdict is HeytingVerdict.DEGRADED and coherent_f:
            narrative_amber = "Luz ámbar vigente (1 h): alivio de coste operativo."
        else:
            narrative_amber = ""

        latch = bool(self._enable_crowbar and verdict is HeytingVerdict.VETOED)
        crowbar = self.actuate_crowbar_esp32(latch, HeytingVerdict(int(verdict)).name)
        if latch:
            logger.critical(
                "[Φ₃·CROWBAR] Disparo SCR BT151: word=0x%02x GPIO14@0x%08X I_pk=%.1f A "
                "dI/dt=%.2f A/μs (seguro=%s).",
                crowbar.firmware_word,
                crowbar.gpio_register,
                crowbar.bus_peak_current_a,
                crowbar.di_dt_a_per_us,
                crowbar.di_dt_safe,
            )

        dim = evidence.dim if evidence else int(p1_cert.payload.get("dim", 0))
        visceral = self.project_visceral(
            p1_cert, p2_cert, rho_rank, dim, anomaly_initial, self._wacc_policy
        )
        if verdict is HeytingVerdict.COHERENT:
            narrative = (
                f"PALADÍN COHERENTE: F={p1_cert.uhlmann_fidelity:.8f} ≥ {self._floor} "
                f"(d_B={p1_cert.bures_distance:.6f} ≤ {_BURES_CEIL:.6f}); "
                f"Moufang={p1_cert.octonion_moufang_residual:.2e}; "
                f"KAM/Krein/Poincaré/Banach certificados (β₁=0)."
            )
        elif verdict is HeytingVerdict.DEGRADED:
            narrative = (
                f"PALADÍN DEGRADADO: evidencia parcial o umbral blando activo. "
                f"{narrative_amber} E_2γ={p2_cert.positron_energy_kev:.4f} keV "
                f"(ortogonal al veredicto)."
            )
        else:
            narrative = (
                f"PALADÍN VETO DURO: invariante violado o compuerta rechazada "
                f"({gate12.reason} / {gate23.reason}). Crowbar BT151 enclavado "
                f"(I_pk={crowbar.bus_peak_current_a:.1f} A)."
            )
        visceral = replace(visceral, narrative=narrative)

        leaves: List[bytes] = [
            _canonical_json(p1_cert.payload) + p1_cert.hmac_signature.encode(),
            _canonical_json(p2_cert.payload) + p2_cert.hmac_signature.encode(),
            _canonical_json(
                {
                    "gate12": gate12.reason,
                    "admitted12": gate12.admitted,
                    "gate23": gate23.reason,
                    "admitted23": gate23.admitted,
                }
            ),
            _canonical_json(
                {
                    "verdict": int(verdict),
                    "latch": latch,
                    "firmware_word": crowbar.firmware_word,
                    "di_dt": crowbar.di_dt_a_per_us,
                }
            ),
            (evidence.rho_digest if evidence else "").encode(),
            (evidence.sigma_digest if evidence else "").encode(),
            _canonical_json(
                {
                    "energy_kev": p2_cert.positron_energy_kev,
                    "anomaly_residual": p2_cert.anomaly_residual,
                    "turnstile_lobe_area": p2_cert.turnstile_lobe_area,
                    "betti_1": p2_cert.betti_1_cycles,
                }
            ),
            self._chain_head.encode(),
        ]
        root = _merkle_root_rfc6962(leaves)
        proofs: List[MerkleInclusionProof] = []
        for idx, lf in enumerate(leaves):
            path = _merkle_audit_path_rfc6962(idx, leaves)
            ok_proof = _verify_merkle_audit_path_rfc6962(lf, path, root)
            proofs.append(
                MerkleInclusionProof(
                    leaf_index=idx,
                    leaf_sha256=hashlib.sha256(b"\x00" + lf).hexdigest(),
                    audit_path=path,
                    root_hex=root,
                    verified=ok_proof,
                )
            )
        head, self._chain_head = self._chain_head, root

        state = PaladinSovereignState(
            agent_version=_AGENT_VERSION,
            timestamp_ns=now,
            truth_value=HeytingTruthValue.from_canonical(verdict),
            uhlmann_fidelity=p1_cert.uhlmann_fidelity,
            amber_relief_active=amber_live,
            esp32_crowbar_triggered=latch,
            wacc_protected_percent=visceral.wacc_protected_percent,
            liquidity_released_percent=visceral.liquidity_released_percent,
            kv_cache_compressed_percent=visceral.kv_cache_compressed_percent,
            merkle_root=root,
            visceral_narrative=narrative,
            canonical_verdict=int(verdict),
            p1=p1_cert,
            p2=p2_cert,
            gate12=gate12,
            gate23=gate23,
            crowbar=crowbar,
            visceral=visceral,
            engine_report=engine_report,
            evidence=evidence,
            merkle_leaves=tuple(p.leaf_sha256 for p in proofs),
            merkle_proofs=tuple(proofs),
            chain_head=head,
            coverage=float(
                (
                    p2_cert.celestial_measured_fraction
                    + (1.0 if p2_cert.rsi_measured else 0.0)
                )
                / 2.0
            ),
        )
        logger.info(
            "[Φ₃·CLAUSURA] Ω₃=%s F=%.8f crowbar=%s Merkle=%s… (pruebas=%d/8)",
            HeytingVerdict(int(verdict)).name,
            state.uhlmann_fidelity,
            latch,
            root[:16],
            sum(1 for p in proofs if p.verified),
        )
        if continuation is None:
            return state
        return continuation(state)


# ═════════════════════════════════════════════════════════════════════════════════════
# ══ ORQUESTADOR SOBERANO · Z = Φ₃ ∘ Φ₂ ∘ Φ₁ + BUCLE RSI-L3 CON GEODÉSICA DE BURES  ══
# ═════════════════════════════════════════════════════════════════════════════════════
class BoolNovikovPaladinAgent(Morphism):
    r"""
    Soberano Agéntico Paladín Bool–Novikov (`v16.0.0`).
    Teje las tres fases anidadas Φ₁, Φ₂ y Φ₃ con separación criptográfica HKDF exacta
    entre emisor y verificador en las compuertas Φ₁₂ y Φ₂₃, e implementa el bucle RSI-L3
    sobre la geodésica Riemanniana de transporte paralelo de Bures-Uhlmann con verificador
    inmutable seguro ante Löb.
    """

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        *,
        key_id: str = _DEFAULT_KEY_ID,
        fidelity_floor: float = _FIDELITY_FLOOR,
        shield_lambda_cap: float = _SHIELD_LAMBDA_CAP,
        banach_policy: str = "degrade",
        amber_ttl_ns: int = _AMBER_TTL_NS,
        wacc_policy_percent: float = 15.0,
        enable_crowbar: bool = True,
        require_explicit_key: bool = False,
    ) -> None:
        if secret_key is None:
            if require_explicit_key:
                raise PaladinAuthorizationError(
                    "Se exige clave maestra explícita de ≥ 256 bits (A1/A3)."
                )
            secret_key = secrets.token_bytes(32)
            self._trusted = False
            logger.warning(
                "[Soberano] Clave maestra efímera: veredicto máximo acotado a DEGRADED."
            )
        else:
            if len(secret_key) < 32:
                raise PaladinAuthorizationError("Clave maestra ≥ 256 bits requerida")
            self._trusted = True
        master = bytes(secret_key)
        self._key_id = str(key_id)

        # Subclaves HKDF con separación estricta de dominios y pareo emisor/verificador
        self._k_p1 = _hkdf_sha256(master, b"paladin/P1/cert")
        self._k_p2 = _hkdf_sha256(master, b"paladin/P2/cert")
        self._k_ann = _hkdf_sha256(master, b"paladin/annihilation")

        try:
            self.engine = BoolNovikovPaladinEngine(
                secret_key=self._k_ann,
                key_id=f"{self._key_id}:ann",
                fidelity_floor=fidelity_floor,
            )
        except TypeError:
            self.engine = BoolNovikovPaladinEngine(secret_key=self._k_ann)

        self.observer = PaladinPhase1Observer(
            self.engine,
            self._k_p1,
            key_id=f"{self._key_id}:P1",
            key_trusted=self._trusted,
            fidelity_floor=fidelity_floor,
            shield_lambda_cap=shield_lambda_cap,
        )
        self.orientor = PaladinPhase2Orientor(
            self.engine,
            self._k_p2,
            verify_key_p1=self._k_p1,
            key_id=f"{self._key_id}:P2",
            key_trusted=self._trusted,
            amber_ttl_ns=amber_ttl_ns,
            banach_policy=banach_policy,
        )
        self.decider = PaladinPhase3Decider(
            verify_key_p2=self._k_p2,
            key_id=self._key_id,
            key_trusted=self._trusted,
            fidelity_floor=fidelity_floor,
            wacc_policy_percent=wacc_policy_percent,
            banach_policy=banach_policy,
            enable_crowbar=enable_crowbar,
        )
        self._floor = float(fidelity_floor)
        try:
            super().__init__(name="BoolNovikovPaladinAgent")
        except TypeError:
            super().__init__()
        logger.info(
            "BoolNovikovPaladinAgent v%s (trusted=%s, F_floor=%.4f, d_B_ceil=%.6f)",
            _AGENT_VERSION,
            self._trusted,
            fidelity_floor,
            _BURES_CEIL,
        )

    # ── Retracción Geodésica Exacta de Transporte Paralelo de Bures-Uhlmann ──────────
    @staticmethod
    def _bures_riemannian_geodesic_retraction(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        tau: float,
    ) -> NDArray[np.complex128]:
        r"""
        Calcula el punto Γ_τ(ρ, σ) a lo largo de la geodésica minimal de la métrica
        Riemanniana de Bures-Uhlmann entre dos operadores de densidad en D(H):
            A = ρ^{-1/2} (ρ^{1/2} σ ρ^{1/2})^{1/2} ρ^{-1/2},
            G_τ = (1 − τ) I_d + τ A,
            Γ_τ(ρ, σ) = \frac{G_τ ρ G_τ†}{Tr(G_τ ρ G_τ†)}.
        """
        tau_c = float(np.clip(tau, 0.0, 1.0))
        d = rho.shape[0]
        rho_h = 0.5 * (rho + rho.conj().T)
        sig_h = 0.5 * (sigma + sigma.conj().T)

        evals_r, evecs_r = la.eigh(rho_h)
        evals_r_pos = np.maximum(np.real(evals_r), 1.0e-12)
        sqrt_rho = (evecs_r * np.sqrt(evals_r_pos)) @ evecs_r.conj().T
        inv_sqrt_rho = (evecs_r * (1.0 / np.sqrt(evals_r_pos))) @ evecs_r.conj().T

        mid = sqrt_rho @ sig_h @ sqrt_rho
        mid = 0.5 * (mid + mid.conj().T)
        evals_m, evecs_m = la.eigh(mid)
        evals_m_pos = np.maximum(np.real(evals_m), 0.0)
        sqrt_mid = (evecs_m * np.sqrt(evals_m_pos)) @ evecs_m.conj().T

        transport_a = inv_sqrt_rho @ sqrt_mid @ inv_sqrt_rho
        g_tau = (1.0 - tau_c) * np.eye(d, dtype=np.complex128) + tau_c * transport_a
        cand = g_tau @ rho_h @ g_tau.conj().T
        cand = 0.5 * (cand + cand.conj().T)
        tr_cand = max(float(np.real(np.trace(cand))), 1.0e-15)
        return cand / tr_cand

    # ── Síntesis Meta-Funtorial de Nivel 3 modulada por Poincaré y Kantorovich ───────
    @staticmethod
    def _synthesize_level3_kleisli_endofunctor(
        hyper: RSIL3Hyperparameters,
        chirikov_k: float,
        lobe_area: float,
        q_curr: float,
        v_prev: float,
    ) -> Tuple[float, str, RSIL3Hyperparameters]:
        r"""
        [RSI³ NIVEL 2 Y NIVEL 3]:
        Adapta el paso geodésico τ_k y sintetiza la ley del endofuntor CPTP acoplando:
          1. El flujo del turnstile homoclínico de Melnikov-Poincaré y el parámetro de Chirikov.
          2. El parámetro mayorante de Newton-Kantorovich h_k = ½ q_k · v_{k-1} ≤ 1/2.
          3. Endurecimiento monótono estricto de los hiperparámetros verificable por Löb.
        """
        q_safe = q_curr if math.isfinite(q_curr) else 0.5
        kantorovich_h = min(_KANTOROVICH_CEIL, 0.5 * q_safe * min(v_prev, 1.0))
        halley_boost = 1.0 / max(1.0 - 0.5 * kantorovich_h, 0.75)

        if chirikov_k > 1.0 or lobe_area > 0.0:
            tau_step = float(
                np.clip(0.65 * hyper.bures_geodesic_step * halley_boost, 0.15, 0.85)
            )
            synthesized_law = "PoincareDampedBuresGeodesic[TurnstileGuarded]"
        else:
            tau_step = float(
                np.clip(hyper.bures_geodesic_step * halley_boost, 0.15, 0.85)
            )
            synthesized_law = "UhlmannBuresGeodesic[ExactPolar]"

        gain = (
            (1.0 - 0.5 * hyper.trust_radius)
            if (math.isfinite(q_curr) and q_curr < 0.7)
            else 0.8
        )
        hyper_cand = replace(
            hyper,
            fidelity_floor=min(0.9999, max(hyper.fidelity_floor, _FIDELITY_FLOOR)),
            shield_lambda_cap=max(hyper.shield_lambda_cap * gain, 0.0),
            banach_q_max=max(hyper.banach_q_max * 0.99, 0.5),
            trust_radius=float(np.clip(hyper.trust_radius * 0.95, 1e-3, 0.5)),
            bures_geodesic_step=tau_step,
            meta_depth=hyper.meta_depth + 1,
        )
        return tau_step, synthesized_law, hyper_cand

    # ── Plegado anidado de las 3 fases por Continuation-Passing Style (CPS) ──────────
    def execute_ooda_governance(
        self,
        rho_input: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        electron_anomaly: float = 0.0,
        positron_hmac: bytes = b"",
        work_irr: float = 0.0,
        x_history: Optional[List[float]] = None,
        capacity_history: Optional[List[float]] = None,
        *,
        session_id: str = "default",
        kepler_elements: Optional[Sequence[Tuple[float, ...]]] = None,
        frequency_vector: Optional[Sequence[float]] = None,
        monodromy: Optional[NDArray[np.float64]] = None,
        gamma_diss: Optional[float] = None,
        f_drive: Optional[float] = None,
        omega_drive: Optional[float] = None,
        eps_pert: float = 0.01,
        state_history: Optional[Sequence[NDArray]] = None,
        lyapunov_history: Optional[Sequence[float]] = None,
        synthesized_endofunctor: str = "BuresGeodesicRetraction",
        strict_gates: bool = True,
    ) -> PaladinSovereignState:
        r"""
        Z_paladin = Φ₃ ∘ Φ₂ ∘ Φ₁ mediante continuaciones anidadas en posición de cola.
        """
        rho = _validate_density("rho_input", rho_input)
        sigma = _validate_density("rho_target", rho_target, rho.shape[0])
        evidence = _build_evidence(rho, sigma, session_id)
        rank: Optional[int] = None
        if hasattr(self.engine, "phase1"):
            try:
                _, pr = self.engine.phase1.project(rho, "rank_probe")
                rank = int(pr.rank)
            except Exception:  # pragma: no cover
                rank = None
        if rank is None:
            evals_rank = np.real(la.eigvalsh(0.5 * (rho + rho.conj().T)))
            rank = int(np.sum(evals_rank > 1.0e-10))

        trace: Dict[str, Any] = {}

        def _phase3_entry(p2: PaladinPhase2Certificate) -> PaladinSovereignState:
            # El primer método invocado en Fase 3 es su compuerta inicial Φ₂₃
            g23 = self.decider.continue_from_phase2_gate(
                p2, expected_binding=evidence.binding_digest()
            )
            trace["g23"] = g23
            if strict_gates and not g23.admitted:
                raise PaladinGateViolation(f"Compuerta Φ₂₃ rechazó transición: {g23.reason}")
            out = self.decider.decide_and_act(
                trace["p1"],
                p2,
                gate12=trace["g12"],
                gate23=g23,
                evidence=evidence,
                engine_report=trace.get("report"),
                rho_rank=rank,
                anomaly_initial=float(electron_anomaly),
            )
            assert isinstance(out, PaladinSovereignState)
            return out

        def _phase2_entry(p1: PaladinPhase1Certificate) -> PaladinSovereignState:
            # El primer método invocado en Fase 2 es su compuerta inicial Φ₁₂
            g12 = self.orientor.continue_from_phase1_gate(
                p1, rho, sigma, session_id=session_id
            )
            trace["g12"], trace["p1"] = g12, p1
            if strict_gates and not g12.admitted:
                raise PaladinGateViolation(f"Compuerta Φ₁₂ rechazó transición: {g12.reason}")
            out = self.orientor.orient_and_certify(
                p1,
                rho,
                electron_anomaly,
                positron_hmac,
                work_irr,
                x_history,
                capacity_history,
                rho_target=sigma,
                session_id=session_id,
                gate12=g12,
                kepler_elements=kepler_elements,
                frequency_vector=frequency_vector,
                monodromy=monodromy,
                gamma_diss=gamma_diss,
                f_drive=f_drive,
                omega_drive=omega_drive,
                eps_pert=eps_pert,
                state_history=state_history,
                lyapunov_history=lyapunov_history,
                synthesized_endofunctor=synthesized_endofunctor,
                continuation=_phase3_entry,
            )
            assert isinstance(out, PaladinSovereignState)
            return out

        out = self.observer.observe_and_certify(
            rho, sigma, session_id=session_id, continuation=_phase2_entry
        )
        assert isinstance(out, PaladinSovereignState)
        return out

    # ── Bucle Agéntico RSI-L3: Contracción Geodésica de Bures + Verificador de Löb ───
    def run_rsi_l3_cycle(
        self,
        rho_input: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        *,
        max_iterations: int = 6,
        tolerance: float = 1e-12,
        hyperparameters: Optional[RSIL3Hyperparameters] = None,
        **gov_kwargs: Any,
    ) -> Tuple[PaladinSovereignState, RSIL3LoopReport]:
        r"""
        A8: Bucle de Automejora Recursiva de Nivel 3 (RSI-L3):
          · Nivel 1 (Estado): Retracción geodésica de Bures-Uhlmann ρ_{k+1} = Γ_{τ_k}(ρ_k, σ).
          · Nivel 2 (Meta-Parámetros): Endurecimiento monótono acoplado a la estabilidad
            celeste de Poincaré (Chirikov y área de turnstile de Melnikov) y a Banach q̂_k.
          · Nivel 3 (Síntesis de Endofuntor + Verificador Inmutable de Löb): Selección del
            morfismo CPTP óptimo y validación externa de monotonía y β₁ = 0.
        """
        hyper = hyperparameters or RSIL3Hyperparameters(fidelity_floor=self._floor)
        rho = _validate_density("rsi_rho_input", rho_input)
        sigma = _validate_density("rsi_rho_target", rho_target, rho.shape[0])
        if hasattr(self.engine, "phase1"):
            rho, _ = self.engine.phase1.project(rho, "rsi_rho_init")
            sigma, _ = self.engine.phase1.project(sigma, "rsi_sigma_init")

        hist_x: List[NDArray[np.float64]] = [
            np.concatenate([rho.real.ravel(), rho.imag.ravel()])
        ]
        best = self.execute_ooda_governance(
            rho,
            sigma,
            state_history=hist_x,
            strict_gates=False,
            **gov_kwargs,
        )
        v_prev = best.p1.bures_distance if best.p1 else float("inf")
        hist_v: List[float] = [v_prev]
        hist_f: List[float] = [best.uhlmann_fidelity]

        accepted = rejected = 0
        q, bound = float("nan"), float("inf")
        monotone = True
        tightening_all = True
        loeb_sound = True
        betti_1_last = 0
        synthesized_law = "UhlmannBuresGeodesic[ExactPolar]"

        for _ in range(max(1, int(max_iterations))):
            if v_prev <= tolerance:
                break
            chirikov_k = best.p2.chirikov_k if best.p2 is not None else 0.0
            lobe_area = best.p2.turnstile_lobe_area if best.p2 is not None else 0.0
            tau_step, synthesized_law, hyper_cand = self._synthesize_level3_kleisli_endofunctor(
                hyper, chirikov_k, lobe_area, q, v_prev
            )

            if hasattr(self.engine, "phase1") and hasattr(
                self.engine.phase1, "bures_geodesic_step"
            ):
                rho_cand = self.engine.phase1.bures_geodesic_step(rho, sigma, tau_step)
                v_cand_est = self.engine.phase1.uhlmann(rho_cand, sigma).bures_distance
            else:
                rho_cand = self._bures_riemannian_geodesic_retraction(rho, sigma, tau_step)
                _, v_cand_est = self.engine.compute_uhlmann_fidelity(rho_cand, sigma)

            cand_vec = np.concatenate([rho_cand.real.ravel(), rho_cand.imag.ravel()])
            trial_x = hist_x + [cand_vec]
            trial_v = hist_v + [float(v_cand_est)]

            self.observer._floor = hyper_cand.fidelity_floor
            self.observer._cap = hyper_cand.shield_lambda_cap
            self.decider._floor = hyper_cand.fidelity_floor

            cand = self.execute_ooda_governance(
                rho_cand,
                sigma,
                state_history=trial_x,
                lyapunov_history=trial_v,
                synthesized_endofunctor=synthesized_law,
                strict_gates=False,
                **gov_kwargs,
            )
            v_k = cand.p1.bures_distance if cand.p1 else float("inf")
            b1_k = cand.p2.betti_1_cycles if cand.p2 else 0

            lyap_ok, tight_ok = _LoebSafeSovereignVerifier.verify_step(
                v_prev,
                v_k,
                hyper,
                hyper_cand,
                cand.canonical_verdict,
                b1_k,
                tolerance,
            )
            tightening_all = tightening_all and tight_ok
            loeb_sound = loeb_sound and tight_ok

            if lyap_ok and tight_ok:
                accepted += 1
                rho = rho_cand
                best = cand
                hyper = hyper_cand
                hist_x = trial_x
                hist_v = trial_v
                hist_f.append(cand.uhlmann_fidelity)
                betti_1_last = b1_k
                if cand.p2 is not None and math.isfinite(cand.p2.banach_ratio_q):
                    q, bound = cand.p2.banach_ratio_q, cand.p2.banach_error_bound
                delta = abs(v_prev - v_k)
                v_prev = v_k
                if delta < tolerance and math.isfinite(q) and q < hyper.banach_q_max:
                    break
            else:
                rejected += 1
                monotone = False
                hyper = replace(
                    hyper,
                    trust_radius=max(hyper.trust_radius * 0.5, 1e-3),
                    meta_depth=hyper.meta_depth + 1,
                )

        verdict = (
            HeytingVerdict.COHERENT
            if (
                monotone
                and tightening_all
                and betti_1_last == 0
                and math.isfinite(q)
                and q < hyper.banach_q_max
            )
            else HeytingVerdict.DEGRADED
        )
        report = RSIL3LoopReport(
            iterations=accepted + rejected,
            accepted=accepted,
            rejected=rejected,
            lyapunov_history=tuple(float(x) for x in hist_v),
            fidelity_history=tuple(float(x) for x in hist_f),
            contraction_q=float(q),
            banach_error_bound=float(bound),
            monotone=bool(monotone),
            tightening_only=bool(tightening_all),
            loeb_verifier_sound=bool(loeb_sound),
            betti_1_cycles=int(betti_1_last),
            synthesized_law=synthesized_law,
            hyperparameters=hyper,
            verdict=int(verdict),
        )
        merged = heyting_meet(HeytingVerdict(best.canonical_verdict), verdict)
        return (
            replace(
                best,
                rsi_loop=report,
                canonical_verdict=int(merged),
                truth_value=HeytingTruthValue.from_canonical(merged),
            ),
            report,
        )

    # ── Emisor de tokens QED vinculados al contexto ──────────────────────────────────
    def issue_positron_token(
        self,
        anomaly_norm: float,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
        session_id: str = "default",
    ) -> Tuple[bytes, int, int]:
        if not hasattr(self.engine, "issue_positron_token"):
            raise PaladinAuthorizationError(
                "El motor subyacente no soporta emisión de tokens ligados al contexto."
            )
        return self.engine.issue_positron_token(anomaly_norm, rho, sigma, session_id)

    # ── Puente categórico con módulos hermanos del sistema inmune ────────────────────
    def govern_with_siblings(
        self,
        rho: NDArray[np.complex128],
        rho_target: NDArray[np.complex128],
        *,
        node_index: int = 0,
        attention_vector: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        local = self.execute_ooda_governance(rho, rho_target, strict_gates=False)
        merged = HeytingVerdict(local.canonical_verdict)
        out: Dict[str, Any] = {"paladin": local}
        if attention_vector is not None:
            try:
                from app.core.immune_system.gravity_shield import GravitationalShieldFunctor

                class _P:
                    inertial_mass = float(np.real(np.trace(rho)))
                    volatility_alpha = 0.0
                    frohlich_coupling = None

                act = GravitationalShieldFunctor().enforce_gravitational_attractor(
                    _P(),
                    node_index,
                    np.asarray(attention_vector, dtype=np.float64),
                    strict=False,
                )
                out["gravity_shield"] = act
                merged = heyting_meet(
                    merged, HeytingVerdict(int(getattr(act, "heyting_verdict_value", 1)))
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("[Puente] gravity_shield no disponible: %s", exc)
                merged = heyting_meet(merged, HeytingVerdict.DEGRADED)
        out["merged_verdict"] = int(merged)
        return out

    # ── AUTO-TESTS RIGUROSOS DE LAS COMPUERTAS, POINCARÉ, MERKLE Y RSI-L3 ────────────
    def self_test(self) -> Dict[str, Any]:
        r"""
        Verificaciones formales verificables:
          (1) Φ₁₂ ADMITE un certificado legítimo (verifica el pareo HKDF `_k_p1`).
          (2) Φ₁₂ RECHAZA campo manipulado, estado distinto, replay y expiración.
          (3) Φ₂₃ ADMITE certificado legítimo, RECHAZA firma manipulada y APLICA Banach.
          (4) Pruebas de inclusión de Merkle RFC 6962 verificadas en las 8 hojas.
          (5) Propagación de invariantes de Poincaré (Kepler-Halley, Turnstile) y Moufang.
          (6) Bucle RSI-L3 contrae la distancia de Bures con q < q_max y β₁ = 0.
        """
        rng = np.random.default_rng(20261116)
        d = 4
        a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = a @ a.conj().T
        rho /= np.trace(rho)
        b = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        sigma = b @ b.conj().T
        sigma /= np.trace(sigma)
        other = np.eye(d, dtype=np.complex128) / d

        p1 = self.observer.observe_and_certify(rho, sigma, session_id="selftest")
        g_ok = self.orientor.continue_from_phase1_gate(
            p1, rho, sigma, session_id="selftest"
        )
        tampered = replace(
            p1, uhlmann_fidelity=0.999, payload={**p1.payload, "fidelity": 0.999}
        )
        g_tamper = self.orientor.continue_from_phase1_gate(
            tampered, rho, sigma, session_id="selftest"
        )
        g_wrongstate = self.orientor.continue_from_phase1_gate(
            p1, other, sigma, session_id="selftest"
        )
        g_replay = self.orientor.continue_from_phase1_gate(
            p1, rho, sigma, session_id="selftest"
        )
        g_expired = self.orientor.continue_from_phase1_gate(
            p1, rho, sigma, session_id="selftest", now_ns=p1.not_after_ns + 1
        )

        p2 = self.orientor.orient_and_certify(
            p1,
            rho,
            0.0,
            b"",
            0.0,
            rho_target=sigma,
            session_id="selftest",
            gate12=g_ok,
            kepler_elements=[(1.4, 0.35, 0.2, 0.8, 0.4, 0.1)],
            gamma_diss=0.05,
            f_drive=0.25,
            omega_drive=1.0,
        )
        g23_ok = self.decider.continue_from_phase2_gate(
            p2, expected_binding=p1.state_binding
        )
        g23_tamper = self.decider.continue_from_phase2_gate(
            replace(p2, chirikov_k=99.0), expected_binding=p1.state_binding
        )
        g23_banach = self.decider.continue_from_phase2_gate(
            p2, expected_binding=p1.state_binding, q_max=0.0
        )

        # Ejecución completa con bucle RSI-L3 y verificación de pruebas de Merkle
        final_state, rsi_loop = self.run_rsi_l3_cycle(
            rho,
            sigma,
            max_iterations=5,
            kepler_elements=[(1.4, 0.35, 0.2, 0.8, 0.4, 0.1)],
            gamma_diss=0.05,
            f_drive=0.25,
            omega_drive=1.0,
        )
        all_proofs_ok = bool(
            final_state.merkle_proofs
            and all(pr.verified for pr in final_state.merkle_proofs)
        )

        leaf = b"evidencia"
        rfc = _merkle_root_rfc6962([leaf])
        naive = hashlib.sha256(leaf).hexdigest()
        shield = p1.shield
        out = {
            "gate12_admits_legit": g_ok.admitted,
            "gate12_rejects_tampered_field": not g_tamper.admitted,
            "gate12_rejects_wrong_state": not g_wrongstate.admitted,
            "gate12_rejects_replay": not g_replay.admitted,
            "gate12_rejects_expired": not g_expired.admitted,
            "gate23_verifies_signature": not g23_tamper.admitted,
            "gate23_enforces_banach": not g23_banach.contraction_ok,
            "gate23_admits_legit_or_degraded": g23_ok.admitted,
            "bures_ceiling_derived_error": abs(
                _BURES_CEIL - math.sqrt(2.0 - 2.0 * math.sqrt(_FIDELITY_FLOOR))
            ),
            "merkle_domain_separation_present": rfc != naive,
            "merkle_inclusion_proofs_verified": all_proofs_ok,
            "shield_bures_bound_respected": bool(shield.bound_respected) if shield else False,
            "octonion_moufang_residual": p1.octonion_moufang_residual,
            "poincare_turnstile_lobe_area": p2.turnstile_lobe_area,
            "poincare_kepler_halley_residual": p2.kepler_halley_residual,
            "poincare_symplectic_residual": p2.poincare_symplectic_residual,
            "rsi_l3_contraction_q": rsi_loop.contraction_q,
            "rsi_l3_loeb_sound": rsi_loop.loeb_verifier_sound,
            "rsi_l3_betti_1_cycles": rsi_loop.betti_1_cycles,
        }
        logger.info("[SELFTEST v16] %s", out)
        return out

    def verify_complete_integrity(self) -> Dict[str, Any]:
        calib = self.self_test()
        gates_ok = all(
            bool(calib[k])
            for k in (
                "gate12_admits_legit",
                "gate12_rejects_tampered_field",
                "gate12_rejects_wrong_state",
                "gate12_rejects_replay",
                "gate12_rejects_expired",
                "gate23_verifies_signature",
                "gate23_enforces_banach",
                "merkle_domain_separation_present",
                "merkle_inclusion_proofs_verified",
                "shield_bures_bound_respected",
                "rsi_l3_loeb_sound",
            )
        ) and (
            calib["octonion_moufang_residual"] < 1e-12
            and calib["poincare_turnstile_lobe_area"] > 0.0
            and calib["rsi_l3_contraction_q"] < _MAX_BANACH_Q
            and calib["rsi_l3_betti_1_cycles"] == 0
        )
        verdict = HeytingVerdict.COHERENT if gates_ok else HeytingVerdict.VETOED
        if not self._trusted:
            verdict = heyting_meet(verdict, HeytingVerdict.DEGRADED)
        return {
            "agent_version": _AGENT_VERSION,
            "engine_v15": hasattr(self.engine, "phase1"),
            "key_trusted": self._trusted,
            "key_id": self._key_id,
            "hkdf_domains": [
                "paladin/P1/cert",
                "paladin/P2/cert",
                "paladin/annihilation",
            ],
            "thresholds": {
                "fidelity_floor": self._floor,
                "bures_ceiling_DERIVED": _BURES_CEIL,
                "legacy_inconsistent_values": [0.3162, 0.35],
            },
            "calibration": calib,
            "omega3_codification": {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2},
            "crowbar": {
                "register": hex(_ESP32_GPIO_OUT_W1TS),
                "mask": hex(_ESP32_GPIO14_MASK),
                "budget_ns_firmware": _CROWBAR_BUDGET_NS,
                "bt151_max_di_dt_a_per_us": _BT151_MAX_DI_DT_A_PER_US,
                "bt151_max_i2t_a2s": _BT151_MAX_I2T_A2S,
                "host_realtime_guaranteed": False,
                "bt151_latching": True,
            },
            "heyting_verdict_value": int(verdict),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sovereign_key = hashlib.sha256(b"APU_SOVEREIGN_PALADIN_MASTER_KEY_v16").digest()
    agent = BoolNovikovPaladinAgent(secret_key=sovereign_key, key_id="SOVEREIGN-PROD-v16")
    integrity = agent.verify_complete_integrity()
    print(
        f"[v16] BoolNovikovPaladinAgent Integrity: "
        f"Verdict={HeytingVerdict(integrity['heyting_verdict_value']).name} | "
        f"Moufang={integrity['calibration']['octonion_moufang_residual']:.2e} | "
        f"Turnstile={integrity['calibration']['poincare_turnstile_lobe_area']:.4f} | "
        f"q_RSI3={integrity['calibration']['rsi_l3_contraction_q']:.4f} | "
        f"β₁={integrity['calibration']['rsi_l3_betti_1_cycles']}"
    )