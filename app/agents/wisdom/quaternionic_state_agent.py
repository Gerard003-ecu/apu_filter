# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quaternionic State Agent (Soberano Cuaterniónico de Lazo Cerrado)   ║
║ Ruta   : app/agents/wisdom/quaternionic_state_agent.py                       ║
║ Versión: 4.0.0-OODA-Heyting-Orbit-HMAC-Merkle-Crowbar-Secure                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Supervisor OODA del reactor `quaternionic_state_shifter`. Ejecuta el funtor

        Φ₃ ∘ Φ₂ ∘ Φ₁ :  R⁴ × R⁴ ──Φ₁──▶ H × H ──Φ₂──▶ D₂ ──Φ₃──▶ Certificado

con doble anidamiento: (i) el último método de la Fase k del agente invoca y
verifica al primer método de la Fase k+1; (ii) la Fase k del agente consume el
cierre de la Fase k del reactor (Banach/representaciones → informe algebraico
→ auditorías espectrales), con degradación elegante al API legado.

════════════════════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO (sólo enunciados demostrables)
════════════════════════════════════════════════════════════════════════════════

Def. 1 (Ω₃ de Gödel–Dummett). Ω₃ = {0 ≡ VETOED, ½ ≡ DEGRADED, 1 ≡ COHERENT} es
  un álgebra de Heyting lineal: a⊓b = min, a⊔b = max, (a→b) = 1 si a ≤ b, b si
  no; ¬a = a→0. Se cumple la residuación a⊓b ≤ c ⇔ a ≤ (b→c) y falla el
  tercio excluso en ½ (½ ⊔ ¬½ = ½). El combinador de seguridad es el MEET:
      ν_canales = ⊓_k ν_k      (cualquier canal vetado veta el conjunto).

Def. 2 (Clases de similitud y distancia de órbita). [q] = {s q s⁻¹} está
  determinada por (q₀, ‖v‖), i.e. por θ(q) = atan2(‖v‖, q₀) ∈ [0, π] si q ≠ 0
  (θ = d_{S³}(1, q̂)). Lema: para p̂, q̂ ∈ S³,
      min_{s∈S³} d_{S³}(p̂, s q̂ s*) = |θ_p − θ_q|,
  alcanzado por el rotor s que alinea v̂_q con v̂_p (Rodrigues); cota inferior
  por desigualdad triangular con el punto 1. La carta de Riemann de v̂ es una
  coordenada de GAUGE (no invariante de clase): se reporta, no se veta.

Def. 3 (Operador de densidad). ψ = (α, β) = (q₀+iq₁, q₂+iq₃); ρ = ψψ†/‖q‖².
  Tr ρ = 1, Tr ρ² = 1, ρ = ρ† ≥ 0.  Nota: Tr Φ_C(q) = 2q₀, NO ‖q‖².

Def. 4 (Rampa de confianza como refinamiento de celda). Con r = δ/Λ,
  α < β: c(r) = 1 (r ≤ α); ½ + ½(β−r)/(β−α) ∈ [½, 1) (α < r ≤ β); 0 (r > β).
  Canales algebraicos crisp: 1 si residual ≤ τ_hard, 0 si no.
  Lema: quantize(⊓ c_k) = ⊓ ν_k, con ν_k la cuantización crisp de cada canal.

Def. 5 (Gracia y override como implicación). σ = [ν_ch = ½], γ = [gracia
  vigente], ω = [override autenticado]:  ν = ν_ch ⊓ (σ → (γ ⊔ ω)).
  Si σ = 0 la implicación vale 1 (el veto duro no admite override).

════════════════════════════════════════════════════════════════════════════════
II. AXIOMÁTICA DE CONTROL
════════════════════════════════════════════════════════════════════════════════

Ax. I  (Hurwitz). ε_H = |‖pq‖ − ‖p‖‖q‖| / max(1, ‖p‖‖q‖) ≤ τ_hard.
Ax. II (Override autenticado). token = HMAC-SHA256(key, "QSA/OVERRIDE/v4" ‖
       session_sha256); comparación en tiempo constante; ligado a la sesión.
Ax. III(Crowbar). Presupuesto τ_IRAM = 400 ns; latencia nominal t₀ < τ con
       jitter N(0, σ²); P(t > τ) = ½·erfc((τ − t₀)/(σ√2)) se reporta; toda
       violación del presupuesto se certifica, nunca se recorta.
Inv. I (Almacenamiento). H = ½(δ² + ε_H² + ε_Tr²); disipación H_k − H_{k−1}
       auditada entre ciclos (no bloqueante).
Inv. II(Sellos). Sesión y certificado en binario canónico SHA-256; cadena de
       Merkle sobre (token_k, hash_k) de Φ₁→Φ₂→Φ₃.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from app.core.quaternionic_state_shifter import (  # type: ignore
        QuaternionicState,
        QuaternionicStateShifter,
        __version__ as _SHIFTER_VERSION,
    )
except ImportError:  # pragma: no cover - resolución local de desarrollo
    try:
        from quaternionic_state_shifter import (  # type: ignore
            QuaternionicState,
            QuaternionicStateShifter,
            __version__ as _SHIFTER_VERSION,
        )
    except ImportError:  # pragma: no cover
        from quaternionic_state_shifter import (  # type: ignore
            QuaternionicState,
            QuaternionicStateShifter,
        )
        _SHIFTER_VERSION = "unknown"

__all__ = [
    "QuaternionicStateAgent",
    "QuaternionicAgentCertificate",
    "Phase1AgentHandoff",
    "Phase2AgentHandoff",
    "Heyting3",
]

__version__: Final[str] = "4.0.0-OODA-Heyting-Orbit-HMAC-Merkle-Crowbar-Secure"

logger = logging.getLogger("APU.Agents.Wisdom.QuaternionicStateAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_BUDGET_NS: Final[float] = 400.0
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 360.0
_SOFT_VETO_LOWER: Final[float] = 0.3
_SOFT_VETO_UPPER: Final[float] = 0.5
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_SHA256_HEX_LEN: Final[int] = 64
_OVERRIDE_DOMAIN: Final[bytes] = b"QSA/OVERRIDE/v4"
_MERKLE_GENESIS: Final[bytes] = b"\x00" * 32
_NAN_BYTES: Final[bytes] = b"\x00\x00\x00\x00\x00\x00\xf8\x7f"


# ═════════════════════════════════════════════════════════════════════════════
# NÚCLEO NUMÉRICO, CANÓNICO Y DE VALIDACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _as_finite_vector4(S: Any, name: str) -> np.ndarray:
    """Vector real finito de forma exacta (4,), float64, cero canónico."""
    arr = np.asarray(S, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(
            f"La señal '{name}' debe ser estrictamente cuatridimensional. Obtenida: {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"La señal '{name}' contiene valores NaN o infinitos.")
    return _canonicalize_signed_zero(arr.reshape(4))


def _canonical_bytes(arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(_canonicalize_signed_zero(arr))
    header = f"{a.dtype.str}|{a.shape}".encode("utf-8")
    return len(header).to_bytes(8, "little") + header + a.tobytes()


def _pack_f64(value: Any) -> bytes:
    """float64 little-endian con NaN canónico (bool/None/no-numérico ⇒ NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        return _NAN_BYTES
    x = float(value)
    return _NAN_BYTES if math.isnan(x) else struct.pack("<d", x)


def _pack_flag(flag: bool) -> bytes:
    return b"\x01" if bool(flag) else b"\x00"


def _sha_update_str(hasher: "hashlib._Hash", text: str) -> None:
    payload = str(text).encode("utf-8")
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _sha_update_arr(hasher: "hashlib._Hash", arr: np.ndarray) -> None:
    payload = _canonical_bytes(arr)
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _merkle_chain(links: Sequence[Tuple[str, str]]) -> str:
    """h_k = SHA256(h_{k−1} ‖ len(tok) ‖ tok ‖ len(hash) ‖ hash); h₀ = 0³²."""
    digest = _MERKLE_GENESIS
    for token, link_hash in links:
        m = hashlib.sha256()
        m.update(digest)
        _sha_update_str(m, token)
        _sha_update_str(m, link_hash)
        digest = m.digest()
    return digest.hex()


def _finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _state_hash(state: Any) -> str:
    value = getattr(state, "sha256_hash", "")
    return value if isinstance(value, str) else ""


def _state_token(state: Any, fallback: str) -> str:
    value = getattr(state, "phase_token", None)
    return value if isinstance(value, str) and value else fallback


def _state_norm(state: Any) -> float:
    return _finite_or_nan(getattr(state, "norm", math.nan))


def _state_squared_norm(state: Any) -> float:
    sq = getattr(state, "squared_norm", None)
    if sq is not None:
        return _finite_or_nan(sq)
    n = _state_norm(state)
    return n * n if math.isfinite(n) else math.nan


def _vector_norm3(vector_part: Any) -> float:
    arr = np.asarray(vector_part, dtype=np.float64).ravel()
    return float(math.hypot(*[float(x) for x in arr])) if arr.size else 0.0


def _state_vector4(state: Any) -> np.ndarray:
    return np.asarray(getattr(state, "vector_rep"), dtype=np.float64).reshape(4)


def _hamilton(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Producto de Hamilton con acumulación de redondeo correcto."""
    p0, p1, p2, p3 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            math.fsum((p0 * q0, -p1 * q1, -p2 * q2, -p3 * q3)),
            math.fsum((p0 * q1, p1 * q0, p2 * q3, -p3 * q2)),
            math.fsum((p0 * q2, -p1 * q3, p2 * q0, p3 * q1)),
            math.fsum((p0 * q3, p1 * q2, -p2 * q1, p3 * q0)),
        ],
        dtype=np.float64,
    )


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _geodesic_angle(u: np.ndarray, v: np.ndarray) -> float:
    """d_{S³}(u, v) = 2·atan2(‖u − v‖, ‖u + v‖): error O(ε) uniforme en [0, π]."""
    return float(2.0 * math.atan2(math.hypot(*(u - v)), math.hypot(*(u + v))))


def _aligning_rotor(a: np.ndarray, b: np.ndarray, reg: float) -> np.ndarray:
    """
    Rotor unitario s ∈ S³ con s a s* = b para a, b ∈ S² ⊂ Im(H):
        s ∝ (1 + a·b, a × b)     (medio ángulo de Rodrigues).
    Caso antipodal: rotación de π alrededor de cualquier eje ⊥ a.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = 1.0 + float(np.dot(a, b))
    c = np.cross(a, b)
    if w < reg:
        e = np.zeros(3)
        e[int(np.argmin(np.abs(a)))] = 1.0
        n = np.cross(a, e)
        n /= math.hypot(*n)
        return np.array([0.0, n[0], n[1], n[2]], dtype=np.float64)
    s = np.array([w, c[0], c[1], c[2]], dtype=np.float64)
    return s / math.hypot(*s)


# ═════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE HEYTING TRIVALENTE Ω₃ (Gödel–Dummett)
# ═════════════════════════════════════════════════════════════════════════════

class Heyting3:
    """Retículo residuado lineal {0 < ½ < 1}; ver Def. 1."""

    VETOED: Final[float] = 0.0
    DEGRADED: Final[float] = 0.5
    COHERENT: Final[float] = 1.0
    CARRIER: Final[Tuple[float, float, float]] = (0.0, 0.5, 1.0)

    @staticmethod
    def meet(a: float, b: float) -> float:
        return float(min(a, b))

    @staticmethod
    def join(a: float, b: float) -> float:
        return float(max(a, b))

    @staticmethod
    def meet_all(values: Iterable[float]) -> float:
        out = Heyting3.COHERENT
        for v in values:
            out = Heyting3.meet(out, v)
        return out

    @staticmethod
    def implies(a: float, b: float) -> float:
        return Heyting3.COHERENT if a <= b else float(b)

    @staticmethod
    def neg(a: float) -> float:
        return Heyting3.implies(a, Heyting3.VETOED)

    @staticmethod
    def quantize(value: float) -> float:
        """Proyección {c = 0} ↦ 0, {0 < c < 1} ↦ ½, {c ≥ 1} ↦ 1; no finito ↦ 0."""
        if not math.isfinite(value) or value <= 0.0:
            return Heyting3.VETOED
        if value >= 1.0:
            return Heyting3.COHERENT
        return Heyting3.DEGRADED

    clamp = quantize  # compatibilidad: `clamp` era una reimplementación inerte.

    @staticmethod
    def from_verdict(verdict: str) -> float:
        return {"VETOED": 0.0, "DEGRADED": 0.5, "COHERENT": 1.0}.get(
            str(verdict).upper(), Heyting3.VETOED
        )

    @staticmethod
    def to_verdict(value: float) -> str:
        q = Heyting3.quantize(value)
        return "VETOED" if q == 0.0 else ("COHERENT" if q == 1.0 else "DEGRADED")

    @classmethod
    def verify_axioms(cls) -> bool:
        """Residuación, distributividad, no contradicción y no-booleanidad en Ω₃."""
        O = cls.CARRIER
        for a in O:
            for b in O:
                for c in O:
                    if (cls.meet(a, b) <= c) != (a <= cls.implies(b, c)):
                        return False
                    if cls.meet(a, cls.join(b, c)) != cls.join(cls.meet(a, b), cls.meet(a, c)):
                        return False
            if cls.meet(a, cls.neg(a)) != cls.VETOED:
                return False
        return cls.join(cls.DEGRADED, cls.neg(cls.DEGRADED)) == cls.DEGRADED


# ═════════════════════════════════════════════════════════════════════════════
# FRONTERAS DE FASE Y CERTIFICADO
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class Phase1AgentHandoff:
    """Φ₁→₂ : (p, q, σ₁, δ₁) ∈ H × H × Σ × D₁. Dominio de `phase2_from_phase1`."""

    p_state: QuaternionicState
    q_state: QuaternionicState
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1AgentHandoff):
            return NotImplemented
        return (self.session_sha256 == other.session_sha256
                and self.next_entrypoint == other.next_entrypoint
                and _state_hash(self.p_state) == _state_hash(other.p_state)
                and _state_hash(self.q_state) == _state_hash(other.q_state))


@dataclass(frozen=True, slots=True, eq=False)
class Phase2AgentHandoff:
    """Φ₂→₃ : (p, q, σ₁, δ, ε_H, ε_Tr, Λ, D₂). Dominio de `phase3_from_phase2`."""

    p_state: QuaternionicState
    q_state: QuaternionicState
    session_sha256: str
    similarity_mismatch: float
    hurwitz_drift: float
    trace_von_neumann_error: float
    elastic_bound: float
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def _metric_bytes(self) -> bytes:
        return b"".join(_pack_f64(x) for x in (
            self.similarity_mismatch, self.hurwitz_drift,
            self.trace_von_neumann_error, self.elastic_bound))

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint, self._metric_bytes()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2AgentHandoff):
            return NotImplemented
        return (self.session_sha256 == other.session_sha256
                and self.next_entrypoint == other.next_entrypoint
                and self._metric_bytes() == other._metric_bytes())


@dataclass(frozen=True, slots=True, eq=False)
class QuaternionicAgentCertificate:
    """Certificado inmutable. Campos originales conservados; los nuevos al final."""

    phase: str
    heyting_verdict: str
    cohomological_mismatch: float
    norm_drift_error: float
    trace_von_neumann_error: float
    is_surgery_active: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    session_sha256: str = ""
    phase_chain: Tuple[str, ...] = ()
    confidence: float = 1.0
    heyting_truth_value: float = 1.0
    elastic_bound: float = 0.0
    override_sha256: str = ""

    merkle_root: str = ""
    similarity_geodesic: float = math.nan
    similarity_gauge_defect: float = math.nan
    channel_truth_values: Tuple[float, ...] = ()
    override_rejected: bool = False
    latency_budget_ns: float = _CROWBAR_IRAM_BUDGET_NS
    latency_budget_violated: bool = False
    latency_exceedance_probability: float = math.nan
    lyapunov_storage: float = math.nan
    lyapunov_dissipation: float = math.nan
    reactor_version: str = ""

    def __hash__(self) -> int:
        return hash(self.digital_signature_sha256)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuaternionicAgentCertificate):
            return NotImplemented
        return self.digital_signature_sha256 == other.digital_signature_sha256

    def __repr__(self) -> str:
        return (f"QuaternionicAgentCertificate(verdict={self.heyting_verdict!r}, "
                f"confidence={self.confidence:.6g}, seal={self.digital_signature_sha256[:12]!r})")


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CALIBRE CUATERNIÓNICO — TRES FASES ANIDADAS (OODA)
# ═════════════════════════════════════════════════════════════════════════════

class QuaternionicStateAgent:
    """
    FASE 1  OBSERVE   : validación, Fase 1 del reactor, sello de sesión.
    FASE 2  ORIENT    : órbitas de similitud, Hurwitz, von Neumann, informe
                        algebraico y espectral del reactor, cota elástica.
    FASE 3  DECIDE/ACT: canales Ω₃, rampa, gracia/override HMAC, Lyapunov,
                        Crowbar con presupuesto, certificado + Merkle.
    """

    def __init__(
        self,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        rng_seed: Optional[int] = None,
        jitter_sigma: float = 2.8,
        soft_veto_lower: float = _SOFT_VETO_LOWER,
        soft_veto_upper: float = _SOFT_VETO_UPPER,
        override_hmac_key: Optional[bytes] = None,
        crowbar_nominal_latency_ns: float = _CROWBAR_NOMINAL_LATENCY_NS,
        latency_sampler: Optional[Callable[[], float]] = None,
        lyapunov_tolerance: float = 1e-12,
    ) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser finita y estrictamente positiva.")
        if not math.isfinite(grace_period_seconds) or grace_period_seconds < 0.0:
            raise ValueError("grace_period_seconds debe ser finita y no negativa.")
        if not math.isfinite(jitter_sigma) or jitter_sigma < 0.0:
            raise ValueError("jitter_sigma debe ser finita y no negativa.")
        if not (math.isfinite(soft_veto_lower) and math.isfinite(soft_veto_upper)):
            raise ValueError("Los umbrales de veto suave deben ser finitos.")
        if not (0.0 < soft_veto_lower < soft_veto_upper):
            raise ValueError("Se exige 0 < soft_veto_lower < soft_veto_upper.")
        if not math.isfinite(crowbar_nominal_latency_ns) or crowbar_nominal_latency_ns <= 0.0:
            raise ValueError("crowbar_nominal_latency_ns debe ser finita y positiva.")
        if crowbar_nominal_latency_ns >= _CROWBAR_IRAM_BUDGET_NS:
            raise ValueError(
                f"La latencia nominal ({crowbar_nominal_latency_ns} ns) debe ser menor que "
                f"el presupuesto τ_IRAM = {_CROWBAR_IRAM_BUDGET_NS} ns (Ax. III)."
            )
        if override_hmac_key is not None and (not isinstance(override_hmac_key, (bytes, bytearray))
                                              or len(override_hmac_key) < 16):
            raise ValueError("override_hmac_key debe ser bytes de al menos 16 octetos.")
        if not math.isfinite(lyapunov_tolerance) or lyapunov_tolerance < 0.0:
            raise ValueError("lyapunov_tolerance debe ser finita y no negativa.")
        if not Heyting3.verify_axioms():  # pragma: no cover - autoprueba algebraica
            raise RuntimeError("Ω₃ no satisface los axiomas de Heyting (fallo interno).")

        self._tol = float(tolerance)
        self._safety_margin = float(safety_margin)
        self._grace_limit = float(grace_period_seconds)
        self._regularization = max(1e-15, self._tol * 1e-3)
        self._jitter_sigma = float(jitter_sigma)
        self._soft_veto_lower = float(soft_veto_lower)
        self._soft_veto_upper = float(soft_veto_upper)
        self._override_key: Optional[bytes] = bytes(override_hmac_key) if override_hmac_key else None
        self._nominal_latency = float(crowbar_nominal_latency_ns)
        self._lyapunov_tol = float(lyapunov_tolerance)
        self._rng = np.random.default_rng(rng_seed)
        self._latency_sampler: Callable[[], float] = latency_sampler or (
            lambda: self._nominal_latency + float(self._rng.normal(0.0, self._jitter_sigma))
        )
        self._shifter = QuaternionicStateShifter(tolerance=tolerance)

    def _tolerance_of(self, scale: float = 1.0) -> float:
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    def _hard_algebraic_threshold(self) -> float:
        """Umbral crisp para identidades exactas en ℝ ya relativizadas."""
        return max(self._tol, 32.0 * _MACHINE_EPS)

    # ═════════════════════════════════════════════════════════════════════════
    # ADAPTADORES: GOBIERNO DE LAS FASES HOMÓLOGAS DEL REACTOR
    # ═════════════════════════════════════════════════════════════════════════

    def _optional_shifter_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._shifter, method_name, None)
        if not callable(method):
            return None
        try:
            return method(*args, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeError, TypeError) as exc:
            logger.warning("Reactor.%s falló: %s", method_name, exc)
            return None

    def _reactor_phase1(self, S: np.ndarray) -> Tuple[QuaternionicState, Dict[str, Any]]:
        """Fase 1 del reactor (cierre Φ₁→₂ con diagnósticos) o API legado."""
        closer = getattr(self._shifter, "phase1_close_and_open_phase2", None)
        if callable(closer):
            handoff = closer(S)
            return handoff.state, dict(handoff.diagnostics)
        ctor = getattr(self._shifter, "phase1_construct_canonical_state", None)
        return (ctor(S) if callable(ctor) else self._shifter.build_state(S)), {}

    def _reactor_phase2_report(self, state: QuaternionicState) -> Dict[str, Any]:
        """Informe algebraico de la Fase 2 del reactor (cierre Φ₂→₃), si existe."""
        handoff = self._optional_shifter_call("phase2_close_and_open_phase3", state)
        report = getattr(handoff, "algebra_report", None)
        return dict(report) if isinstance(report, Mapping) else {}

    def _reactor_phase3_audits(self, state: QuaternionicState) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, method in (("spectral", "phase3_spectral_audit"),
                            ("similarity", "phase3_similarity_class"),
                            ("hopf", "phase3_hopf_fibration"),
                            ("riemann", "phase3_riemann_sphere_chart"),
                            ("von_neumann", "phase1_von_neumann_trace_audit")):
            res = self._optional_shifter_call(method, state)
            out[key] = dict(res) if isinstance(res, Mapping) else {}
        return out

    def _raw_multiply(self, p: QuaternionicState, q: QuaternionicState) -> QuaternionicState:
        """Producto sin renormalización (deriva pura). Legado: puede renormalizar."""
        if hasattr(self._shifter, "phase2_hamilton_product"):
            return self._shifter.phase2_hamilton_product(p, q, verify=False)
        return self._shifter.quaternionic_multiply(p, q)

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — OBSERVE: VALIDACIÓN, CONSTRUCCIÓN Y SELLO DE SESIÓN
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_signal(self, S: np.ndarray, name: str) -> np.ndarray:
        """Fase 1.1 — S ∈ R⁴ finito, float64, −0.0 ↦ +0.0."""
        return _as_finite_vector4(S, name)

    def phase1_observe_pair(
        self, p_S: np.ndarray, q_S: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, QuaternionicState, QuaternionicState, Dict[str, Any]]:
        """
        Fase 1.2 — Pullback Φ × Φ : R⁴ × R⁴ → H × H gobernando la Fase 1 del
        reactor. Devuelve señales canónicas, estados y diagnósticos del reactor.
        """
        p_arr = self.phase1_validate_signal(p_S, "p_S")
        q_arr = self.phase1_validate_signal(q_S, "q_S")
        p_state, p_diag = self._reactor_phase1(p_arr)
        q_state, q_diag = self._reactor_phase1(q_arr)
        reactor_diag = {**{f"p_reactor_{k}": v for k, v in p_diag.items()},
                        **{f"q_reactor_{k}": v for k, v in q_diag.items()}}
        return p_arr, q_arr, p_state, q_state, reactor_diag

    def phase1_observation_diagnostics(
        self, p_state: QuaternionicState, q_state: QuaternionicState,
    ) -> Dict[str, Any]:
        """Fase 1.3 — Normas, unitariedad, ortogonalidad y ángulo polar del par."""
        def _pack(prefix: str, st: QuaternionicState) -> Dict[str, Any]:
            return {
                f"{prefix}_norm": _state_norm(st),
                f"{prefix}_squared_norm": _state_squared_norm(st),
                f"{prefix}_vector_norm": _finite_or_nan(getattr(st, "vector_norm", math.nan)),
                f"{prefix}_theta": self._polar_angle(st),
                f"{prefix}_is_unitary": bool(getattr(st, "is_unitary", False)),
                f"{prefix}_unitarity_residual": _finite_or_nan(getattr(st, "unitarity_residual", math.nan)),
                f"{prefix}_orthogonality_residual": _finite_or_nan(getattr(st, "orthogonality_residual", math.nan)),
                f"{prefix}_condition_estimate": _finite_or_nan(getattr(st, "condition_estimate", math.nan)),
                f"{prefix}_sha256_prefix": _state_hash(st)[:16],
            }
        return {**_pack("p", p_state), **_pack("q", q_state),
                "machine_epsilon": _MACHINE_EPS, "regularization": self._regularization,
                "tolerance": self._tol, "reactor_version": str(_SHIFTER_VERSION),
                "agent_version": __version__}

    def _phase1_session_hash(
        self, p_arr: np.ndarray, q_arr: np.ndarray,
        p_state: QuaternionicState, q_state: QuaternionicState,
    ) -> str:
        """
        Fase 1.4 — Sello de sesión SHA-256 (Inv. II): señales, hashes de estado,
        versiones y TODOS los parámetros de gobernanza (nunca la clave HMAC).
        """
        h = hashlib.sha256()
        h.update(b"QSA/SESSION/v4")
        _sha_update_arr(h, p_arr)
        _sha_update_arr(h, q_arr)
        _sha_update_str(h, _state_hash(p_state))
        _sha_update_str(h, _state_hash(q_state))
        _sha_update_str(h, str(_SHIFTER_VERSION))
        _sha_update_str(h, __version__)
        for v in (self._tol, self._safety_margin, self._grace_limit, self._soft_veto_lower,
                  self._soft_veto_upper, self._nominal_latency, self._jitter_sigma,
                  self._lyapunov_tol):
            h.update(_pack_f64(v))
        h.update(_pack_flag(self._override_key is not None))
        _sha_update_str(h, "PHASE1/OBSERVE")
        digest = h.hexdigest()
        if len(digest) != _SHA256_HEX_LEN:  # pragma: no cover
            raise RuntimeError("El sello de sesión no es un SHA-256 de 64 nibbles.")
        return digest

    def phase1_close_and_open_phase2(self, p_S: np.ndarray, q_S: np.ndarray) -> Phase1AgentHandoff:
        """
        Fase 1.5 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

            Φ₁→₂ : (p_S, q_S) ∈ R⁴ × R⁴ ↦ (p, q, σ₁, δ₁) ∈ H × H × Σ × D₁

        Último método de Fase 1. Su codominio es el dominio de `phase2_from_phase1`,
        que se invoca y verifica aquí (anidamiento como prefijo functorial).
        """
        p_arr, q_arr, p_state, q_state, reactor_diag = self.phase1_observe_pair(p_S, q_S)
        session_sha256 = self._phase1_session_hash(p_arr, q_arr, p_state, q_state)
        diagnostics = {**self.phase1_observation_diagnostics(p_state, q_state), **reactor_diag,
                       "session_sha256_prefix": session_sha256[:16]}
        handoff = Phase1AgentHandoff(p_state=p_state, q_state=q_state,
                                     session_sha256=session_sha256,
                                     diagnostics=diagnostics, next_entrypoint=_PHASE1_ENTRY)

        opened_p, opened_q = self.phase2_from_phase1(handoff)        # ← anidamiento Φ₁→₂
        if _state_hash(opened_p) != _state_hash(p_state) or _state_hash(opened_q) != _state_hash(q_state):
            raise RuntimeError("Invariante de anidamiento Φ₁→₂ violado.")
        logger.info("Fase Observe [QUATERNION_AGENT]: sesión sellada %s", session_sha256[:16])
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ORIENT: INVARIANTES DE ÓRBITA, ALGEBRAICOS Y ESPECTRALES
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1AgentHandoff) -> Tuple[QuaternionicState, QuaternionicState]:
        """Fase 2.0 — Entrada formal: admite Φ₁→₂ y devuelve (p, q) ∈ H × H."""
        if not isinstance(handoff, Phase1AgentHandoff):
            raise TypeError("Se esperaba Phase1AgentHandoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(f"Phase1AgentHandoff inválido: se esperaba {_PHASE1_ENTRY!r}.")
        if not isinstance(handoff.session_sha256, str) or len(handoff.session_sha256) != _SHA256_HEX_LEN:
            raise ValueError("El sello de sesión de Φ₁→₂ no es un SHA-256 válido.")
        for name, st in (("p", handoff.p_state), ("q", handoff.q_state)):
            if _state_vector4(st).shape != (4,):
                raise ValueError(f"El estado {name} de Φ₁→₂ no es cuatridimensional.")
        return handoff.p_state, handoff.q_state

    def _polar_angle(self, state: QuaternionicState) -> float:
        """θ(q) = atan2(‖v‖, q₀) ∈ [0, π]; NaN si q es nulo (clase sin representante unitario)."""
        n = _state_norm(state)
        if not math.isfinite(n) or n < self._regularization:
            return math.nan
        th = _finite_or_nan(getattr(state, "theta", math.nan))
        if math.isfinite(th):
            return th
        q0 = _finite_or_nan(getattr(state, "scalar_part", math.nan))
        vn = _finite_or_nan(getattr(state, "vector_norm", math.nan))
        if not math.isfinite(vn):
            vn = _vector_norm3(getattr(state, "vector_part", ()))
        return float(math.atan2(vn, q0)) if math.isfinite(q0) else math.nan

    def phase2_evaluate_similarity_sphere(self, state: QuaternionicState) -> float:
        """
        Fase 2.1 — Invariante de clase de un estado: δ(q) = θ(q)/π = d_{S³}(1, q̂)/π.

        Es constante sobre la órbita [q] (Def. 2) y libre de gimbal lock.
        Estado nulo ⇒ NaN (no certificable).
        """
        th = self._polar_angle(state)
        return float(th / math.pi) if math.isfinite(th) else math.nan

    def phase2_similarity_orbit_mismatch(
        self, p: QuaternionicState, q: QuaternionicState,
    ) -> Dict[str, float]:
        """
        Fase 2.2 — Distancia entre órbitas de similitud y defecto de gauge.

            δ_class = |θ_p − θ_q| / π            (invariante, Lema Def. 2)
            d_geo   = d_{S³}(p̂, q̂) / π          (incluye orientación)
            gauge   = d_geo − δ_class ≥ 0       (removible por conjugación)

        Se verifica el lema construyendo s con s v̂_q s* = v̂_p y midiendo
        d_{S³}(p̂, s q̂ s*) − |θ_p − θ_q| (debe ser ≈ 0).
        """
        th_p, th_q = self._polar_angle(p), self._polar_angle(q)
        nan_out = {"similarity_mismatch": math.nan, "similarity_geodesic": math.nan,
                   "similarity_gauge_defect": math.nan, "similarity_alignment_residual": math.nan,
                   "theta_p": th_p, "theta_q": th_q}
        if not (math.isfinite(th_p) and math.isfinite(th_q)):
            return nan_out
        pu = _state_vector4(p) / _state_norm(p)
        qu = _state_vector4(q) / _state_norm(q)
        class_distance = abs(th_p - th_q)
        geodesic = _geodesic_angle(pu, qu)
        gauge = geodesic - class_distance
        if gauge < -self._tolerance_of(1.0):
            logger.warning("Desigualdad triangular violada numéricamente: %.3e", gauge)

        vp, vq = _vector_norm3(pu[1:]), _vector_norm3(qu[1:])
        if vp >= self._regularization and vq >= self._regularization:
            s = _aligning_rotor(qu[1:] / vq, pu[1:] / vp, self._regularization)
            aligned = _hamilton(_hamilton(s, qu), _conj(s))
            alignment_residual = abs(_geodesic_angle(pu, aligned) - class_distance)
        else:
            alignment_residual = abs(gauge)   # p̂ o q̂ real: geodésica ≡ distancia de clase
        if alignment_residual > self._tolerance_of(1.0) * 1e3:
            logger.warning("Lema de órbita con residual %.3e", alignment_residual)
        return {
            "similarity_mismatch": float(class_distance / math.pi),
            "similarity_geodesic": float(geodesic / math.pi),
            "similarity_gauge_defect": float(max(0.0, gauge) / math.pi),
            "similarity_alignment_residual": float(alignment_residual),
            "theta_p": th_p, "theta_q": th_q,
        }

    def phase2_riemann_gauge_chart(self, state: QuaternionicState) -> Dict[str, Any]:
        """
        Fase 2.3 — Coordenada de gauge de v̂ ∈ S² (carta estereográfica del
        reactor). Diagnóstica: NO entra en el veredicto (Def. 2).
        """
        chart = self._optional_shifter_call("phase3_riemann_sphere_chart", state)
        if not isinstance(chart, Mapping):
            legacy = getattr(self._shifter, "project_to_riemann_sphere", None)
            if callable(legacy):
                try:
                    Z, z = legacy(state)
                    return {"gauge_chart": "legacy", "gauge_height": _finite_or_nan(z),
                            "gauge_coordinate_real": _finite_or_nan(complex(Z).real),
                            "gauge_coordinate_imag": _finite_or_nan(complex(Z).imag)}
                except (ValueError, ZeroDivisionError, TypeError):
                    pass
            return {"gauge_chart": "unavailable", "gauge_height": math.nan,
                    "gauge_coordinate_real": math.nan, "gauge_coordinate_imag": math.nan}
        return {"gauge_chart": str(chart.get("chart", "")),
                "gauge_height": _finite_or_nan(chart.get("height")),
                "gauge_coordinate_real": _finite_or_nan(chart.get("coordinate_real")),
                "gauge_coordinate_imag": _finite_or_nan(chart.get("coordinate_imag")),
                "gauge_transition_residual": _finite_or_nan(chart.get("transition_residual"))}

    def phase2_hopf_obstruction(self, state: QuaternionicState) -> Dict[str, float]:
        """Fase 2.4 — ‖π(û)‖ = 1 y Hopf ≡ Bloch, si el reactor los expone (NaN ⇒ neutro)."""
        hopf = self._optional_shifter_call("phase3_hopf_fibration", state)
        if not isinstance(hopf, Mapping):
            return {"hopf_base_norm": math.nan, "hopf_sphericity_residual": math.nan,
                    "hopf_bloch_identity_residual": math.nan}
        return {"hopf_base_norm": _finite_or_nan(hopf.get("base_norm")),
                "hopf_sphericity_residual": _finite_or_nan(hopf.get("sphericity_residual")),
                "hopf_bloch_identity_residual": _finite_or_nan(hopf.get("bloch_identity_residual"))}

    def phase2_audit_hurwitz_composition(self, p: QuaternionicState, q: QuaternionicState) -> float:
        """
        Fase 2.5 — ε_H = |‖pq‖ − ‖p‖‖q‖| / max(1, ‖p‖‖q‖) (Ax. I), cruzado con
        `phase2_hurwitz_residual` del reactor (se toma el máximo).
        """
        prod = self._raw_multiply(p, q)
        expected = _state_norm(p) * _state_norm(q)
        pn = _state_norm(prod)
        if not (math.isfinite(expected) and math.isfinite(pn)):
            return math.nan
        drift = abs(pn) if expected <= self._regularization else abs(pn - expected) / max(1.0, expected)
        alt = _finite_or_nan(self._optional_shifter_call("phase2_hurwitz_residual", p, q, prod))
        return float(max(drift, alt)) if math.isfinite(alt) else float(drift)

    def phase2_verify_von_neumann_trace(self, state: QuaternionicState) -> float:
        """
        Fase 2.6 — Def. 3 corregida: ρ = ψψ†/‖q‖², ψ = (q₀+iq₁, q₂+iq₃).

            ε_Tr = hypot(Re Tr ρ − 1, Im Tr ρ, Tr ρ² − 1, ‖ρ − ρ†‖_F)

        Estado nulo ⇒ NaN (no densificable). Se cruza con el reactor.
        """
        n2 = _state_squared_norm(state)
        if not math.isfinite(n2) or n2 < self._regularization:
            return math.nan
        v = _state_vector4(state)
        psi = np.array([complex(v[0], v[1]), complex(v[2], v[3])], dtype=np.complex128)
        rho = np.outer(psi, psi.conj()) / n2
        tr = complex(np.trace(rho))
        purity = float(np.real(np.trace(rho @ rho)))
        herm = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
        err = math.hypot(tr.real - 1.0, tr.imag, purity - 1.0, herm)
        if not math.isfinite(err):
            return math.nan
        alt = self._optional_shifter_call("phase1_von_neumann_trace_audit", state)
        if isinstance(alt, Mapping):
            for key in ("trace_residual", "purity_residual", "hermiticity_residual"):
                r = _finite_or_nan(alt.get(key))
                if math.isfinite(r):
                    err = max(err, r)
        return float(err)

    def phase2_reactor_algebra_report(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 2.7 — Identidades exactas de la Fase 2 del reactor: C*, polinomio
        mínimo, homomorfismos L/R/Φ_C. Canal opcional crisp (NaN ⇒ neutro ⊤).
        """
        rep = self._reactor_phase2_report(state)
        keys = ("cstar_residual", "minimal_polynomial_matrix_residual",
                "minimal_polynomial_algebraic_residual", "left_homomorphism_residual",
                "right_antihomomorphism_residual", "cayley_dickson_homomorphism_residual",
                "bimodule_commutation_residual", "exp_log_residual")
        out = {f"reactor_{k}": _finite_or_nan(rep.get(k)) for k in keys}
        finite = [v for k, v in out.items() if math.isfinite(v) and not k.endswith("exp_log_residual")]
        out["reactor_algebra_max_residual"] = max(finite) if finite else math.nan
        out["reactor_algebra_available"] = 1.0 if finite else 0.0
        return out

    def phase2_spectral_consistency(self, state: QuaternionicState) -> Dict[str, Any]:
        """Fase 2.8 — Espectro de L(q), clase de similitud del reactor y Cayley–Hamilton."""
        audits = self._reactor_phase3_audits(state)
        sp, sim = audits["spectral"], audits["similarity"]
        return {
            "spectral_radius": _finite_or_nan(sp.get("spectral_radius")),
            "spectral_residual": _finite_or_nan(sp.get("spectral_residual")),
            "eigenvalue_model_residual": _finite_or_nan(sp.get("eigenvalue_model_residual")),
            "determinant_residual": _finite_or_nan(sp.get("determinant_residual")),
            "cayley_hamilton_residual": _finite_or_nan(sp.get("cayley_hamilton_residual")),
            "normality_residual": _finite_or_nan(sp.get("normality_residual")),
            "similarity_center": _finite_or_nan(sim.get("center")),
            "similarity_radius": _finite_or_nan(sim.get("radius")),
            "similarity_class_type": str(sim.get("class_type", "unknown")),
            "similarity_cd_spectrum_residual": _finite_or_nan(sim.get("cayley_dickson_spectrum_residual")),
        }

    def phase2_elastic_bound(self, lipschitz_bound_Lmax: float) -> float:
        """Fase 2.9 — Λ = L_max · η (adimensional; escala el cociente r = δ/Λ)."""
        if not math.isfinite(lipschitz_bound_Lmax) or lipschitz_bound_Lmax <= 0.0:
            raise ValueError("lipschitz_bound_Lmax debe ser finita y estrictamente positiva.")
        bound = float(lipschitz_bound_Lmax) * self._safety_margin
        if not math.isfinite(bound) or bound <= 0.0:
            raise OverflowError("La cota elástica no es representable o es nula.")
        return bound

    def phase2_collect_invariants(
        self, p_state: QuaternionicState, q_state: QuaternionicState, lipschitz_bound_Lmax: float,
    ) -> Dict[str, Any]:
        """Fase 2.10 — Recolección conjunta: canales primarios + diagnósticos del reactor."""
        orbit = self.phase2_similarity_orbit_mismatch(p_state, q_state)
        return {
            **orbit,
            "similarity_mismatch_p_channel": self.phase2_evaluate_similarity_sphere(p_state),
            "similarity_mismatch_q_channel": self.phase2_evaluate_similarity_sphere(q_state),
            "hurwitz_drift": self.phase2_audit_hurwitz_composition(p_state, q_state),
            "trace_von_neumann_error": self.phase2_verify_von_neumann_trace(q_state),
            "elastic_bound": self.phase2_elastic_bound(lipschitz_bound_Lmax),
            **self.phase2_reactor_algebra_report(q_state),
            **self.phase2_spectral_consistency(q_state),
            **self.phase2_hopf_obstruction(q_state),
            **self.phase2_riemann_gauge_chart(q_state),
            "safety_margin": self._safety_margin, "tolerance": self._tol,
            "regularization": self._regularization,
            "soft_veto_lower": self._soft_veto_lower, "soft_veto_upper": self._soft_veto_upper,
        }

    def phase2_close_and_open_phase3(
        self, phase1_handoff: Phase1AgentHandoff, lipschitz_bound_Lmax: float,
    ) -> Phase2AgentHandoff:
        """
        Fase 2.11 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

            Φ₂→₃ : (p, q, σ₁) ↦ (p, q, σ₁, δ, ε_H, ε_Tr, Λ, D₂)

        Último método de Fase 2; su codominio es el dominio de `phase3_from_phase2`.
        """
        p_state, q_state = self.phase2_from_phase1(phase1_handoff)
        inv = self.phase2_collect_invariants(p_state, q_state, lipschitz_bound_Lmax)
        handoff = Phase2AgentHandoff(
            p_state=p_state, q_state=q_state, session_sha256=phase1_handoff.session_sha256,
            similarity_mismatch=float(inv["similarity_mismatch"]),
            hurwitz_drift=float(inv["hurwitz_drift"]),
            trace_von_neumann_error=float(inv["trace_von_neumann_error"]),
            elastic_bound=float(inv["elastic_bound"]),
            diagnostics=inv, next_entrypoint=_PHASE2_ENTRY,
        )
        opened = self.phase3_from_phase2(handoff)                     # ← anidamiento Φ₂→₃
        if opened.session_sha256 != handoff.session_sha256:
            raise RuntimeError("Invariante de anidamiento Φ₂→₃ violado.")
        logger.debug("Fase Orient [QUATERNION_AGENT]: δ=%.6e ε_H=%.6e ε_Tr=%.6e gauge=%.6e",
                     handoff.similarity_mismatch, handoff.hurwitz_drift,
                     handoff.trace_von_neumann_error, inv["similarity_gauge_defect"])
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — DECIDE / ACT: CANALES Ω₃, RAMPA, GRACIA, HARDWARE Y CERTIFICADO
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2AgentHandoff) -> Phase2AgentHandoff:
        """Fase 3.0 — Entrada formal: admite Φ₂→₃ y valida el dominio de las métricas."""
        if not isinstance(handoff, Phase2AgentHandoff):
            raise TypeError("Se esperaba Phase2AgentHandoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(f"Phase2AgentHandoff inválido: se esperaba {_PHASE2_ENTRY!r}.")
        if not isinstance(handoff.session_sha256, str) or len(handoff.session_sha256) != _SHA256_HEX_LEN:
            raise ValueError("El sello de sesión de Φ₂→₃ no es un SHA-256 válido.")
        d = handoff.similarity_mismatch
        if not (math.isnan(d) or 0.0 <= d <= 1.0 + 4.0 * _MACHINE_EPS):
            raise ValueError(f"similarity_mismatch fuera de [0, 1]: {d!r}")
        for name, val in (("hurwitz_drift", handoff.hurwitz_drift),
                          ("trace_von_neumann_error", handoff.trace_von_neumann_error)):
            if not (math.isnan(val) or val >= 0.0):
                raise ValueError(f"{name} debe ser ≥ 0 o NaN: {val!r}")
        if not math.isfinite(handoff.elastic_bound) or handoff.elastic_bound <= 0.0:
            raise ValueError("elastic_bound debe ser finita y estrictamente positiva.")
        return handoff

    def phase3_channel_truth_values(self, handoff: Phase2AgentHandoff) -> Dict[str, float]:
        """
        Fase 3.1 — Valores crisp ν_k ∈ Ω₃ por canal.

          similarity : r = δ/Λ;  r ≤ α ↦ 1;  α < r ≤ β ↦ ½;  r > β ↦ 0;  NaN ↦ 0
          hurwitz, von_neumann : residual ≤ τ_hard ↦ 1; si no ↦ 0; NaN ↦ 0
          reactor_algebra (opcional) : idem; NaN (no disponible) ↦ 1 (neutro)
        """
        hard = self._hard_algebraic_threshold()
        a, b = self._soft_veto_lower, self._soft_veto_upper

        def sphere(delta: float, lam: float) -> float:
            if not (math.isfinite(delta) and math.isfinite(lam)) or lam <= 0.0:
                return Heyting3.VETOED
            r = delta / lam
            return Heyting3.COHERENT if r <= a else (Heyting3.DEGRADED if r <= b else Heyting3.VETOED)

        def crisp(res: float, neutral_if_nan: bool = False) -> float:
            if not math.isfinite(res):
                return Heyting3.COHERENT if neutral_if_nan else Heyting3.VETOED
            return Heyting3.COHERENT if res <= hard else Heyting3.VETOED

        return {
            "similarity": sphere(handoff.similarity_mismatch, handoff.elastic_bound),
            "hurwitz": crisp(handoff.hurwitz_drift),
            "von_neumann": crisp(handoff.trace_von_neumann_error),
            "reactor_algebra": crisp(_finite_or_nan(handoff.diagnostics.get("reactor_algebra_max_residual")),
                                     neutral_if_nan=True),
        }

    def phase3_confidence_ramp(self, handoff: Phase2AgentHandoff) -> float:
        """
        Fase 3.2 — Confianza continua c = ⊓ c_k (Def. 4). El canal de similitud
        refina la celda DEGRADED en [½, 1); los algebraicos son crisp. Lema:
        quantize(c) = ⊓ ν_k (verificado en `phase3_decide_heyting`).
        """
        a, b = self._soft_veto_lower, self._soft_veto_upper
        d, lam = handoff.similarity_mismatch, handoff.elastic_bound
        if not (math.isfinite(d) and math.isfinite(lam)) or lam <= 0.0:
            c_sphere = Heyting3.VETOED
        else:
            r = d / lam
            if r <= a:
                c_sphere = Heyting3.COHERENT
            elif r <= b:
                c_sphere = 0.5 + 0.5 * (b - r) / (b - a)
            else:
                c_sphere = Heyting3.VETOED
        truths = self.phase3_channel_truth_values(handoff)
        c = Heyting3.meet_all([c_sphere, truths["hurwitz"], truths["von_neumann"], truths["reactor_algebra"]])
        return float(min(1.0, max(0.0, c)))

    def phase3_issue_override_token(self, session_sha256: str) -> str:
        """
        Fase 3.3a — Lado de la autoridad humana: token = HMAC-SHA256(key, dominio ‖ sesión).
        Requiere clave configurada (Ax. II).
        """
        if self._override_key is None:
            raise PermissionError("No hay override_hmac_key configurada: no se pueden emitir tokens.")
        if not isinstance(session_sha256, str) or len(session_sha256) != _SHA256_HEX_LEN:
            raise ValueError("session_sha256 inválido.")
        return hmac.new(self._override_key, _OVERRIDE_DOMAIN + session_sha256.encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def _phase3_verify_override(self, token: Optional[str], session_sha256: str) -> Tuple[bool, Optional[str], bool]:
        """
        Fase 3.3b — (aceptado, registro_sha256, rechazado). Con clave: comparación
        en tiempo constante. Sin clave: modo legado (cualquier token no vacío) con advertencia.
        """
        if token is None or not str(token).strip():
            return False, None, False
        tok = str(token).strip()
        record = hashlib.sha256(_OVERRIDE_DOMAIN + session_sha256.encode("utf-8")
                                + tok.encode("utf-8")).hexdigest()
        if self._override_key is None:
            logger.warning("Override en MODO LEGADO sin autenticación HMAC (Ax. II no aplicado).")
            return True, record, False
        expected = self.phase3_issue_override_token(session_sha256)
        if hmac.compare_digest(expected.lower(), tok.lower()):
            return True, record, False
        logger.error("Override RECHAZADO: HMAC inválido para la sesión %s.", session_sha256[:16])
        return False, record, True

    def phase3_decide_heyting(
        self,
        handoff: Phase2AgentHandoff,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fase 3.4 — Clasificador Ω₃:  ν = ν_ch ⊓ (σ → (γ ⊔ ω))   (Def. 5).

        ν_ch = ⊓ canales; σ = [ν_ch = ½]; γ = [gracia vigente]; ω = [override válido].
        El veto duro (ν_ch = 0) no admite override porque σ = 0 ⇒ (σ → ·) = 1.
        """
        truths = self.phase3_channel_truth_values(handoff)
        nu_ch = Heyting3.meet_all(truths.values())
        confidence = self.phase3_confidence_ramp(handoff)
        if Heyting3.quantize(confidence) != nu_ch:  # pragma: no cover - lema Def. 4
            raise RuntimeError("Lema de cuantización violado: rampa y canales discrepan.")

        if elapsed_grace_seconds is not None:
            e = float(elapsed_grace_seconds)
            if not math.isfinite(e) or e < 0.0:
                raise ValueError("elapsed_grace_seconds debe ser finito y no negativo.")
            gamma = Heyting3.COHERENT if e < self._grace_limit else Heyting3.VETOED
        else:
            e = math.nan
            gamma = Heyting3.VETOED if simulate_grace_expired else Heyting3.COHERENT

        sigma = Heyting3.COHERENT if nu_ch == Heyting3.DEGRADED else Heyting3.VETOED
        accepted, override_sha256, rejected = (False, None, False)
        if sigma == Heyting3.COHERENT:
            accepted, override_sha256, rejected = self._phase3_verify_override(
                override_token, handoff.session_sha256)
        omega = Heyting3.COHERENT if accepted else Heyting3.VETOED

        truth = Heyting3.meet(nu_ch, Heyting3.implies(sigma, Heyting3.join(gamma, omega)))
        verdict = Heyting3.to_verdict(truth)
        is_soft = sigma == Heyting3.COHERENT
        expired = bool(is_soft and gamma == Heyting3.VETOED and not accepted)

        if is_soft and accepted:
            logger.info("Override humano [e⁺] aceptado. Registro: %s", (override_sha256 or "")[:16])
        elif expired:
            logger.warning("Gracia de %.0f s expirada sin override válido: colapso a VETOED.", self._grace_limit)
        elif is_soft:
            logger.warning("VETO SUAVE (luz ámbar): r = %.4f ∈ (α, β]; cuenta atrás %.0f s.",
                           handoff.similarity_mismatch / handoff.elastic_bound, self._grace_limit)

        return {
            "heyting_verdict": verdict, "heyting_truth_value": float(truth),
            "channel_truth_values": dict(truths), "channels_meet": float(nu_ch),
            "confidence": float(confidence),
            "is_soft_veto": is_soft, "is_hard_veto": nu_ch == Heyting3.VETOED,
            "override_grace_period_expired": expired,
            "override_accepted": accepted, "override_rejected": rejected,
            "override_sha256": override_sha256,
            "sigma": sigma, "gamma": gamma, "omega": omega,
            "hard_threshold": self._hard_algebraic_threshold(),
            "elastic_bound": handoff.elastic_bound,
            "grace_period_seconds": self._grace_limit, "elapsed_grace_seconds": e,
        }

    def phase3_lyapunov_dissipation(
        self, handoff: Phase2AgentHandoff, previous_storage: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Fase 3.5 — Función de almacenamiento H = ½(δ² + ε_H² + ε_Tr²) (Inv. I) y
        disipación H_k − H_{k−1}; contractivo si ≤ τ_Lyapunov. No bloqueante.
        """
        terms = (handoff.similarity_mismatch, handoff.hurwitz_drift, handoff.trace_von_neumann_error)
        storage = math.inf if any(not math.isfinite(t) for t in terms) else 0.5 * math.fsum(t * t for t in terms)
        if previous_storage is None or not math.isfinite(previous_storage) or not math.isfinite(storage):
            dissipation, contractive = math.nan, math.nan
        else:
            dissipation = storage - float(previous_storage)
            contractive = 1.0 if dissipation <= self._lyapunov_tol else 0.0
        return {"lyapunov_storage": storage, "lyapunov_dissipation": dissipation,
                "lyapunov_contractive": contractive, "lyapunov_tolerance": self._lyapunov_tol}

    def phase3_actuate_hardware(self, heyting_verdict: str) -> Tuple[bool, float]:
        """
        Fase 3.6 — Crowbar BT151 (GPIO14 ↦ HIGH) sólo ante VETOED. La latencia
        procede del muestreador inyectado (telemetría real o modelo t₀ + N(0,σ²));
        NUNCA se recorta al presupuesto (Ax. III).
        """
        if str(heyting_verdict) != "VETOED":
            return False, 0.0
        latency = _finite_or_nan(self._latency_sampler())
        if math.isfinite(latency) and latency < 0.0:
            latency = math.nan
        logger.critical("Ω₃ colapsada a ⊥ en Soberano Cuaterniónico: GPIO14=HIGH vía IRAM en %.2f ns "
                        "(presupuesto %.0f ns). Crowbar BT151 gatillado.", latency, _CROWBAR_IRAM_BUDGET_NS)
        return True, latency

    def phase3_actuation_audit(self, fired: bool, latency_ns: float) -> Dict[str, Any]:
        """Fase 3.7 — Presupuesto τ_IRAM y probabilidad analítica de exceso ½·erfc((τ−t₀)/(σ√2))."""
        tau, t0, s = _CROWBAR_IRAM_BUDGET_NS, self._nominal_latency, self._jitter_sigma
        p_exceed = (0.5 * math.erfc((tau - t0) / (s * math.sqrt(2.0)))) if s > 0.0 else (0.0 if t0 <= tau else 1.0)
        violated = bool(fired and (not math.isfinite(latency_ns) or latency_ns > tau))
        if violated:
            logger.error("Presupuesto de actuación violado: %.2f ns > %.0f ns.", latency_ns, tau)
        return {"latency_budget_ns": tau, "latency_nominal_ns": t0, "jitter_sigma_ns": s,
                "latency_exceedance_probability": float(p_exceed),
                "latency_budget_violated": violated, "actuation_latency_ns": float(latency_ns)}

    def _phase3_certificate_seal(self, payload: Mapping[str, Any], chain: Sequence[str]) -> str:
        """Fase 3.8 — Sello binario canónico del certificado (Inv. II)."""
        h = hashlib.sha256()
        h.update(b"QSA/CERT/v4")
        for key in sorted(payload):
            _sha_update_str(h, key)
            val = payload[key]
            if isinstance(val, bool):
                h.update(b"b" + _pack_flag(val))
            elif isinstance(val, (int, float, np.floating, np.integer)):
                h.update(b"f" + _pack_f64(val))
            elif isinstance(val, (tuple, list)):
                h.update(b"t" + len(val).to_bytes(8, "little"))
                for x in val:
                    h.update(_pack_f64(x) if isinstance(x, (int, float)) else str(x).encode("utf-8"))
            else:
                h.update(b"s")
                _sha_update_str(h, "" if val is None else str(val))
        for tok in chain:
            _sha_update_str(h, tok)
        return h.hexdigest()

    def phase3_issue_certificate(
        self,
        handoff: Phase2AgentHandoff,
        decision: Mapping[str, Any],
        hardware_interlock_fired: bool,
        actuation_latency_ns: float,
        actuation_audit: Optional[Mapping[str, Any]] = None,
        lyapunov: Optional[Mapping[str, float]] = None,
    ) -> QuaternionicAgentCertificate:
        """Fase 3.9 — Certificado inmutable con sello SHA-256 y raíz de Merkle Φ₁→Φ₂→Φ₃."""
        act = dict(actuation_audit or self.phase3_actuation_audit(hardware_interlock_fired, actuation_latency_ns))
        lya = dict(lyapunov or self.phase3_lyapunov_dissipation(handoff))
        verdict = str(decision["heyting_verdict"])
        override_sha256 = str(decision.get("override_sha256") or "NO_OVERRIDE")
        truths = decision.get("channel_truth_values", {})
        truth_tuple = tuple(float(truths[k]) for k in ("similarity", "hurwitz", "von_neumann", "reactor_algebra")
                            if k in truths)
        p_token, q_token = _state_token(handoff.p_state, "LEGACY_P"), _state_token(handoff.q_state, "LEGACY_Q")
        phase_chain = ("PHASE1/OBSERVE", p_token, q_token, "PHASE2/ORIENT", "PHASE3/DECIDE/ACT")

        metrics_hash = hashlib.sha256(handoff._metric_bytes()).hexdigest()
        decision_hash = hashlib.sha256(b"".join((
            verdict.encode("utf-8"), _pack_f64(decision.get("heyting_truth_value")),
            _pack_f64(decision.get("confidence")), override_sha256.encode("utf-8")))).hexdigest()
        merkle_root = _merkle_chain((
            ("PHASE1/OBSERVE", handoff.session_sha256),
            (p_token, _state_hash(handoff.p_state)), (q_token, _state_hash(handoff.q_state)),
            ("PHASE2/ORIENT", metrics_hash), ("PHASE3/DECIDE/ACT", decision_hash)))

        payload: Dict[str, Any] = {
            "phase": "G_WISDOM_QUATERNION_SUTURATED", "verdict": verdict,
            "mismatch": handoff.similarity_mismatch, "hurwitz": handoff.hurwitz_drift,
            "trace": handoff.trace_von_neumann_error, "elastic_bound": handoff.elastic_bound,
            "geodesic": _finite_or_nan(handoff.diagnostics.get("similarity_geodesic")),
            "gauge_defect": _finite_or_nan(handoff.diagnostics.get("similarity_gauge_defect")),
            "confidence": float(decision.get("confidence", math.nan)),
            "truth": float(decision.get("heyting_truth_value", Heyting3.from_verdict(verdict))),
            "channel_truths": truth_tuple,
            "is_soft_veto": bool(decision["is_soft_veto"]),
            "expired": bool(decision["override_grace_period_expired"]),
            "override_rejected": bool(decision.get("override_rejected", False)),
            "hardware_fired": bool(hardware_interlock_fired),
            "latency_ns": float(actuation_latency_ns),
            "budget_violated": bool(act["latency_budget_violated"]),
            "p_exceed": float(act["latency_exceedance_probability"]),
            "lyapunov_storage": float(lya["lyapunov_storage"]),
            "lyapunov_dissipation": float(lya["lyapunov_dissipation"]),
            "session": handoff.session_sha256, "override": override_sha256,
            "merkle_root": merkle_root, "reactor_version": str(_SHIFTER_VERSION),
            "agent_version": __version__,
        }
        seal = self._phase3_certificate_seal(payload, phase_chain)

        return QuaternionicAgentCertificate(
            phase="G_WISDOM_QUATERNION_SUTURATED", heyting_verdict=verdict,
            cohomological_mismatch=handoff.similarity_mismatch,
            norm_drift_error=handoff.hurwitz_drift,
            trace_von_neumann_error=handoff.trace_von_neumann_error,
            is_surgery_active=(verdict == "DEGRADED"),
            is_soft_veto_active=bool(decision["is_soft_veto"]),
            override_grace_period_expired=bool(decision["override_grace_period_expired"]),
            hardware_interlock_fired=bool(hardware_interlock_fired),
            actuation_latency_ns=float(actuation_latency_ns),
            digital_signature_sha256=seal,
            session_sha256=handoff.session_sha256, phase_chain=phase_chain,
            confidence=payload["confidence"], heyting_truth_value=payload["truth"],
            elastic_bound=handoff.elastic_bound, override_sha256=override_sha256,
            merkle_root=merkle_root,
            similarity_geodesic=payload["geodesic"], similarity_gauge_defect=payload["gauge_defect"],
            channel_truth_values=truth_tuple,
            override_rejected=payload["override_rejected"],
            latency_budget_ns=float(act["latency_budget_ns"]),
            latency_budget_violated=payload["budget_violated"],
            latency_exceedance_probability=payload["p_exceed"],
            lyapunov_storage=payload["lyapunov_storage"],
            lyapunov_dissipation=payload["lyapunov_dissipation"],
            reactor_version=str(_SHIFTER_VERSION),
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2AgentHandoff,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
        previous_storage: Optional[float] = None,
    ) -> QuaternionicAgentCertificate:
        """Fase 3.10 — Orquestación de Fase 3: frontera → decisión → actuación → certificado."""
        validated = self.phase3_from_phase2(phase2_handoff)
        decision = self.phase3_decide_heyting(validated, override_token, simulate_grace_expired,
                                              elapsed_grace_seconds)
        fired, latency = self.phase3_actuate_hardware(decision["heyting_verdict"])
        act = self.phase3_actuation_audit(fired, latency)
        lya = self.phase3_lyapunov_dissipation(validated, previous_storage)
        cert = self.phase3_issue_certificate(validated, decision, fired, latency, act, lya)
        if cert.heyting_verdict == "VETOED":
            logger.critical("Soberano Cuaterniónico: VETO DURO. Sello %s", cert.digital_signature_sha256[:16])
        else:
            logger.info("Soberano Cuaterniónico regulado. Veredicto %s, confianza %.6g, sello %s",
                        cert.heyting_verdict, cert.confidence, cert.digital_signature_sha256[:16])
        return cert

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL OODA
    # ═════════════════════════════════════════════════════════════════════════

    def execute_quaternionic_control_cycle(
        self,
        p_S: np.ndarray,
        q_S: np.ndarray,
        lipschitz_bound_Lmax: float,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
        previous_storage: Optional[float] = None,
    ) -> QuaternionicAgentCertificate:
        """Φ₃ ∘ Φ₂ ∘ Φ₁ : OBSERVE → ORIENT → DECIDE/ACT."""
        h1 = self.phase1_close_and_open_phase2(p_S, q_S)
        h2 = self.phase2_close_and_open_phase3(h1, lipschitz_bound_Lmax)
        return self.phase3_close_loop(h2, override_token, simulate_grace_expired,
                                      elapsed_grace_seconds, previous_storage)

    # ═════════════════════════════════════════════════════════════════════════
    # MÉTODOS LEGADOS (COMPATIBILIDAD)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_similarity_sphere(self, state: QuaternionicState) -> float:
        """API legada — Fase 2.1 (ahora invariante de clase θ/π; nulo ⇒ NaN)."""
        return self.phase2_evaluate_similarity_sphere(state)

    def audit_hurwitz_composition(self, p: QuaternionicState, q: QuaternionicState) -> float:
        """API legada — Fase 2.5."""
        return self.phase2_audit_hurwitz_composition(p, q)

    def verify_von_neumann_trace(self, state: QuaternionicState) -> float:
        """API legada — Fase 2.6 (operador de densidad correcto)."""
        return self.phase2_verify_von_neumann_trace(state)