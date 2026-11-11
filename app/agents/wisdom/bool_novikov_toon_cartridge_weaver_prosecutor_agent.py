
# -*- coding: utf-8 -*-
r"""
Módulo   : BoolNovikovToonCartridgeWeaverProsecutorAgent v13.1.0
Ubicación: app/agents/wisdom/bool_novikov_toon_cartridge_weaver_prosecutor_agent.py
Versión  : 13.1.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PhD

Gobernanza OODA trifásica con compuertas fail-closed:
  Fase 1 OBSERVE  -> emit_phase1_certificate   (P1: hechos de ρ y de la entrada, firmados)
  Compuerta Φ₁₂  -> continue_to_phase2_gate   (integridad P1 y admisibilidad de ρ)
  Fase 2 ORIENT   -> execute_phase2 / emit_phase2_certificate (P2: informe del motor, firmado)
  Compuerta Φ₂₃  -> continue_to_phase3_gate   (integridad P2, binding P1→P2, invariantes)
  Fase 3 DECIDE   -> emit_sovereign_state      (estado soberano con árbol de Merkle)

Alcance: el agente NO calcula Rademacher–Walsh, Smith, Melnikov, Gromov, RSI³ ni Kleisli;
eso corresponde al motor. El agente certifica la cadena de custodia y aplica invariantes.
La adjunción de de Rham–Galois, el topos Ω₃ y el funtor semántico de la cabecera v13.0
son narrativa de gobernanza, no teoremas implementados aquí.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import logging
import math
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from app.wisdom.bool_novikov_toon_cartridge_weaver_prosecutor_engine import (
        BoolNovikovToonCartridgeWeaverProsecutorEngine,
        HeytingTruthValue,
        ToonCartridgeWeaverProsecutorSpectrumReport,
    )
except ImportError as exc_primary:
    try:
        from bool_novikov_toon_cartridge_weaver_prosecutor_engine import (
            BoolNovikovToonCartridgeWeaverProsecutorEngine,
            HeytingTruthValue,
            ToonCartridgeWeaverProsecutorSpectrumReport,
        )
    except ImportError as exc_fallback:
        raise ImportError(
            "No se pudo importar bool_novikov_toon_cartridge_weaver_prosecutor_engine "
            f"(ruta primaria: {exc_primary}; ruta alternativa: {exc_fallback})."
        ) from exc_fallback

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonCartridgeWeaverProsecutorAgent")

# ── Versiones y esquema ────────────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "13.1.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION
_SCHEMA_VERSION: Final[str] = "agent-state-v13.1"

# ── Clave HMAC (mismo esquema que el motor) ────────────────────────────────────
_HMAC_KEY_ENV: Final[str] = "APU_TOON_WEAVER_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_AGENT_DEV_ONLY_TOON_WEAVER_KEY_v13.1.0"

# ── Dominios de separación ─────────────────────────────────────────────────────
_CHAIN_DOMAIN_P1: Final[bytes] = b"CHAIN-P1-v13.1|"
_CHAIN_DOMAIN_P2: Final[bytes] = b"CHAIN-P2-v13.1|"
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-WEAVER-PROSECUTOR-AGENT-P1-OBSERVE-v13.1|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-WEAVER-PROSECUTOR-AGENT-P2-ORIENT-v13.1|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"

# ── Constantes de política y de verificación ───────────────────────────────────
_CROWBAR_BUDGET_NS: Final[float] = 392.15          # presupuesto de política, no medida
_RSI_MAX_DEPTH: Final[int] = 3
_KV_COMPRESSION_TOL: Final[float] = 1e-6           # exige que el motor no redondee el porcentaje
_SPOHN_TOL: Final[float] = -1e-12
_POLICY_WACC_PROTECTION_PCT: Final[float] = 15.0   # declarada, no derivada de cómputo
_POLICY_NET_LIQUIDITY_RELEASE_PCT: Final[float] = 11.5  # declarada, no derivada de cómputo


class ProsecutorGateError(RuntimeError):
    """Violación de una compuerta homomórfica (autenticidad, binding o invariante). Fail-closed."""


@dataclass(frozen=True)
class ProsecutorPolicy:
    """Umbrales de admisión. Son política declarada, no resultados de cómputo."""

    trace_tol: float = 1e-9
    hermitian_rel_tol: float = 1e-9
    psd_tol: float = -1e-12
    banach_lip_max: float = 1.0
    crowbar_budget_ns: float = _CROWBAR_BUDGET_NS
    require_selftest: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Certificados: campos firmados y no firmados separados
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ToonWeaverProsecutorSignedObservation:
    """Hechos de Fase 1 cubiertos por la cadena y la firma."""

    input_digest: str
    nonce_hex: str
    tr_rho_real: float
    tr_rho_imag: float
    herm_error: float
    min_eigenvalue: float
    state_admissible: bool


@dataclass(frozen=True)
class ToonWeaverProsecutorPhase1Certificate:
    """Certificado P1. `timestamp` es metadato no firmado."""

    timestamp: float
    signed: ToonWeaverProsecutorSignedObservation
    chain_hash: str
    hmac_signature: str


@dataclass(frozen=True)
class ToonWeaverProsecutorSignedSpectrum:
    """Todo lo que la Fase 3 consume, cubierto por la cadena P2 y su firma."""

    input_digest: str
    truth_value: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_budget_ns: float
    dirac_proof_sha256: str
    passport_hash: str
    positron_hmac_valid: Optional[bool]
    anomaly_norm: float
    rademacher_walsh_norm: float
    novikov_norm: float
    krull_valuation: float
    smith_torsion_free: bool
    lip_certified: float
    fixed_point: float
    a_posteriori_error: float
    rsi_accepted_depth: int
    selftest_passed: bool
    gromov_violated: bool
    melnikov_chaos_detected: bool
    delaunay_valid: bool
    spohn_entropy_prod: float
    capacity_jerk_d3C_dt3: float
    raw_token_count: int
    compressed_token_count: int
    kv_cache_compression_pct: float


@dataclass(frozen=True)
class ToonWeaverProsecutorPhase2Certificate:
    """Certificado P2. `p1_chain_hash` queda encadenado y autenticado por `chain_hash`."""

    timestamp: float
    p1_chain_hash: str
    signed: ToonWeaverProsecutorSignedSpectrum
    chain_hash: str
    hmac_signature: str


@dataclass(frozen=True)
class ToonWeaverProsecutorVisceralMetrics:
    """Métricas de presentación. Las marcadas `_policy` no son resultados de cómputo."""

    wacc_protection_pct_policy: float
    net_liquidity_release_pct_policy: float
    kv_cache_compression_pct: float
    tokens_per_cartridge: int
    capacity_jerk_d3C_dt3: float
    capacity_jerk_sign: int
    melnikov_chaos_detected: bool
    gromov_violated: bool
    delaunay_valid: bool


@dataclass(frozen=True)
class ToonWeaverProsecutorSovereignState:
    """Estado soberano final de Fase 3."""

    timestamp: float
    schema_version: str
    truth_value: HeytingTruthValue
    crowbar_triggered: bool
    crowbar_latency_budget_ns: float
    dirac_proof_sha256: str
    passport_hash: str
    input_digest: str
    p1_chain_hash: str
    p2_chain_hash: str
    merkle_root_hash: str
    rsi_accepted_depth: int
    metamorphic_selftest_passed: bool
    banach_lip_certified: float
    visceral_metrics: ToonWeaverProsecutorVisceralMetrics


# ══════════════════════════════════════════════════════════════════════════════
# Primitivas: serialización canónica, cadena, firma, entrada, Merkle
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(value: object) -> str:
    """Serialización canónica de un campo firmado. Determinista; rechaza valores ambiguos."""
    if value is None:
        return "none"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}:{int(value.value)}"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Campo firmado no finito; emisión abortada (fail-closed).")
        return f"{value:.17g}"
    if isinstance(value, str):
        if "|" in value:
            raise ValueError("Separador '|' prohibido en campo firmado.")
        return value
    raise TypeError(f"Tipo no canónico en campo firmado: {type(value).__name__}")


def _canonical_fields(obj: object) -> str:
    """Concatena `nombre=valor` en orden de declaración del dataclass."""
    return "|".join(f"{f.name}={_fmt(getattr(obj, f.name))}" for f in dataclasses.fields(obj))


def _chain_hash(domain: bytes, prev_hash: str, signed: object) -> str:
    """Hash de cadena: dominio, enlace previo y campos firmados."""
    h = hashlib.sha256()
    h.update(domain)
    h.update(prev_hash.encode("ascii"))
    h.update(b"#")
    h.update(_canonical_fields(signed).encode("utf-8"))
    return h.hexdigest()


def _sign(key: bytes, domain: bytes, chain_hash: str) -> str:
    return hmac.new(key, domain + chain_hash.encode("ascii"), hashlib.sha256).hexdigest()


def _input_digest(mat: NDArray[np.float64], exps: NDArray[np.float64]) -> str:
    """Digest de la entrada auditada (MIC y exponentes), con forma y bytes float64 contiguos."""
    h = hashlib.sha256()
    for name, arr in (("MIC", mat), ("EXP", exps)):
        h.update(f"{name}{arr.shape}".encode("ascii"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _merkle_root(leaves: Sequence[str]) -> str:
    """Árbol de Merkle SHA-256 con prefijos de dominio. El número de hojas es fijo por esquema."""
    if not leaves:
        raise ValueError("Árbol de Merkle vacío.")
    level = [hashlib.sha256(_MERKLE_LEAF + leaf.encode("utf-8")).digest() for leaf in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(hashlib.sha256(_MERKLE_NODE + level[i] + level[i + 1]).digest())
            else:
                nxt.append(level[i])  # nodo impar promovido sin rehash
        level = nxt
    return level[0].hex()


def _resolve_secret_key(explicit: Optional[bytes]) -> bytes:
    """Clave explícita, o variable de entorno, o clave de desarrollo con advertencia."""
    if explicit is not None:
        key = bytes(explicit)
    else:
        env_val = os.environ.get(_HMAC_KEY_ENV, "")
        if env_val:
            key = env_val.encode("utf-8")
        else:
            logger.warning("Sin %s: clave de DESARROLLO. No apta para producción.", _HMAC_KEY_ENV)
            key = _DEV_SECRET_KEY
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"Clave HMAC demasiado corta: {len(key)} < {_MIN_KEY_BYTES} bytes.")
    return key


def _as_density_matrix(rho: object) -> NDArray[np.complex128]:
    r = np.asarray(rho, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or r.shape[0] == 0:
        raise ValueError("ρ debe ser matriz cuadrada no vacía.")
    if not np.all(np.isfinite(r)):
        raise ValueError("ρ contiene valores no finitos.")
    return np.ascontiguousarray(r)


def _as_cost_matrix(m: object) -> NDArray[np.float64]:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("La matriz de coste debe ser cuadrada, finita y no vacía.")
    return np.ascontiguousarray(a)


def _as_exponents(e: object) -> NDArray[np.float64]:
    v = np.asarray(e, dtype=np.float64).reshape(-1)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("Los exponentes deben ser un vector finito no vacío.")
    return np.ascontiguousarray(v)


def _observe_state(
    rho: NDArray[np.complex128], policy: ProsecutorPolicy
) -> Tuple[float, float, float, float, bool]:
    """Mide Tr(ρ), hermiticidad y positividad. Devuelve (tr_re, tr_im, dev_herm, min_eig, admisible)."""
    tr = complex(np.trace(rho))
    dev = float(np.max(np.abs(rho - rho.conj().T)))
    scale = max(1.0, float(np.max(np.abs(rho))))
    herm_ok = dev <= policy.hermitian_rel_tol * scale
    hermitian_part = 0.5 * (rho + rho.conj().T)
    min_eig = float(np.linalg.eigvalsh(hermitian_part)[0])
    trace_ok = abs(tr.real - 1.0) <= policy.trace_tol and abs(tr.imag) <= policy.trace_tol
    psd_ok = min_eig >= policy.psd_tol
    return tr.real, tr.imag, dev, min_eig, bool(herm_ok and trace_ok and psd_ok)


def _positron_arguments(
    hmac_valid: Optional[bool], message: Optional[bytes], tag: Optional[str]
) -> Dict[str, Any]:
    """Valida la vía del positrón. Booleano y (mensaje, etiqueta) son excluyentes."""
    if hmac_valid is not None and (message is not None or tag is not None):
        raise ValueError("Ambigüedad: positron_hmac_valid y (message, tag) son excluyentes.")
    if (message is None) != (tag is None):
        raise ValueError("positron_message y positron_tag deben proporcionarse juntos.")
    args: Dict[str, Any] = {"positron_hmac_valid": hmac_valid}
    if message is not None:
        args["positron_message"] = message
        args["positron_tag"] = tag
    return args


def _numeric_arguments(
    orbital_elements: Optional[Sequence[float]],
    rsi_max_depth: int,
    kleisli_initial_lip: Optional[float],
) -> Dict[str, Any]:
    if isinstance(rsi_max_depth, bool) or not isinstance(rsi_max_depth, int) \
            or not 0 <= rsi_max_depth <= _RSI_MAX_DEPTH:
        raise ValueError(f"rsi_max_depth debe ser entero en [0, {_RSI_MAX_DEPTH}].")
    args: Dict[str, Any] = {"rsi_max_depth": rsi_max_depth}
    if orbital_elements is not None:
        oe = tuple(float(x) for x in orbital_elements)
        if len(oe) != 6 or not all(math.isfinite(x) for x in oe):
            raise ValueError("orbital_elements: 6 valores finitos en orden (a, e, i, M, ω, Ω).")
        args["orbital_elements"] = oe
    if kleisli_initial_lip is not None:
        lip0 = float(kleisli_initial_lip)
        if not (math.isfinite(lip0) and lip0 >= 0.0):
            raise ValueError("kleisli_initial_lip debe ser finito y no negativo.")
        args["kleisli_initial_lip"] = lip0
    return args


def _melnikov_arguments(
    delta: Optional[float], gamma: Optional[float], omega: Optional[float]
) -> Dict[str, float]:
    vals = (delta, gamma, omega)
    if all(v is None for v in vals):
        return {}
    if any(v is None for v in vals):
        raise ValueError("melnikov_delta, melnikov_gamma y melnikov_omega van juntos.")
    names = ("melnikov_delta", "melnikov_gamma", "melnikov_omega")
    out: Dict[str, float] = {}
    for name, v in zip(names, vals):
        fv = float(v)  # type: ignore[arg-type]
        if not math.isfinite(fv):
            raise ValueError(f"{name} debe ser finito.")
        out[name] = fv
    return out


def _signed_spectrum_from_report(
    rep: ToonCartridgeWeaverProsecutorSpectrumReport, input_digest: str
) -> ToonWeaverProsecutorSignedSpectrum:
    d = rep.dirac_report
    return ToonWeaverProsecutorSignedSpectrum(
        input_digest=input_digest,
        truth_value=rep.truth_value,
        crowbar_triggered=bool(d.crowbar_triggered),
        crowbar_latency_budget_ns=float(d.crowbar_latency_ns) if d.crowbar_triggered else 0.0,
        dirac_proof_sha256=str(d.sha256_proof),
        passport_hash=str(rep.passport_hash),
        positron_hmac_valid=None if d.positron_hmac_valid is None else bool(d.positron_hmac_valid),
        anomaly_norm=float(d.anomaly_norm),
        rademacher_walsh_norm=float(rep.rademacher_walsh_norm),
        novikov_norm=float(rep.novikov_norm),
        krull_valuation=float(rep.krull_valuation),
        smith_torsion_free=bool(rep.smith_torsion_free),
        lip_certified=float(rep.banach_contraction_lip),
        fixed_point=float(rep.banach_fixed_point),
        a_posteriori_error=float(rep.banach_a_posteriori_error),
        rsi_accepted_depth=int(rep.rsi_accepted_depth),
        selftest_passed=bool(rep.metamorphic_selftest_passed),
        gromov_violated=bool(rep.is_gromov_violated),
        melnikov_chaos_detected=bool(rep.melnikov_chaos_detected),
        delaunay_valid=bool(rep.delaunay_valid),
        spohn_entropy_prod=float(rep.spohn_entropy_prod),
        capacity_jerk_d3C_dt3=float(rep.capacity_jerk_d3C_dt3),
        raw_token_count=int(rep.raw_token_count),
        compressed_token_count=int(rep.compressed_token_count),
        kv_cache_compression_pct=float(rep.kv_cache_compression_pct),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — OBSERVE
# ══════════════════════════════════════════════════════════════════════════════

class ToonWeaverProsecutorPhase1Observer:
    """Fase 1 OBSERVE. Ingesta ρ y el par (MIC, exponentes); certifica hechos, sin veredicto."""

    def __init__(self, secret_key: bytes, policy: ProsecutorPolicy) -> None:
        self._secret_key = secret_key
        self._policy = policy

    def emit_phase1_certificate(
        self,
        rho_mac: object,
        apu_cost_matrix: object,
        exponents: object,
    ) -> ToonWeaverProsecutorPhase1Certificate:
        """Último método de la Fase 1: produce el certificado P1 firmado."""
        rho = _as_density_matrix(rho_mac)
        mat = _as_cost_matrix(apu_cost_matrix)
        exps = _as_exponents(exponents)
        tr_re, tr_im, dev, min_eig, admissible = _observe_state(rho, self._policy)
        signed = ToonWeaverProsecutorSignedObservation(
            input_digest=_input_digest(mat, exps),
            nonce_hex=secrets.token_hex(16),
            tr_rho_real=tr_re,
            tr_rho_imag=tr_im,
            herm_error=dev,
            min_eigenvalue=min_eig,
            state_admissible=admissible,
        )
        chain = _chain_hash(_CHAIN_DOMAIN_P1, "", signed)
        return ToonWeaverProsecutorPhase1Certificate(
            timestamp=time.time(),
            signed=signed,
            chain_hash=chain,
            hmac_signature=_sign(self._secret_key, _HMAC_DOMAIN_P1, chain),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Compuertas homomórficas Φ₁₂ y Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════

class ToonWeaverProsecutorGate:
    """Único punto de admisión entre fases. Fail-closed."""

    def __init__(self, secret_key: bytes, policy: ProsecutorPolicy) -> None:
        self._secret_key = secret_key
        self._policy = policy

    def _verify_chain_and_signature(
        self,
        label: str,
        chain_domain: bytes,
        hmac_domain: bytes,
        prev_hash: str,
        signed: object,
        chain_hash: str,
        signature: str,
    ) -> None:
        try:
            recomputed = _chain_hash(chain_domain, prev_hash, signed)
            chain_ok = hmac.compare_digest(recomputed, chain_hash)
            sig_ok = hmac.compare_digest(
                _sign(self._secret_key, hmac_domain, chain_hash), signature
            )
        except (TypeError, ValueError) as exc:
            raise ProsecutorGateError(f"{label}: certificado no canónico ({exc}).") from exc
        if not chain_ok:
            raise ProsecutorGateError(f"{label}: la cadena no reproduce sus campos (manipulación).")
        if not sig_ok:
            raise ProsecutorGateError(f"{label}: firma HMAC inválida.")

    def continue_to_phase2_gate(
        self, p1: ToonWeaverProsecutorPhase1Certificate
    ) -> ToonWeaverProsecutorPhase1Certificate:
        """Φ₁₂: verifica integridad de P1 y admisibilidad del estado ρ."""
        self._verify_chain_and_signature(
            "Φ₁₂", _CHAIN_DOMAIN_P1, _HMAC_DOMAIN_P1, "", p1.signed, p1.chain_hash, p1.hmac_signature
        )
        if not p1.signed.state_admissible:
            raise ProsecutorGateError(
                "Φ₁₂: estado ρ no admisible (hermiticidad, traza o positividad)."
            )
        return p1

    def _check_spectral_invariants(self, s: ToonWeaverProsecutorSignedSpectrum) -> None:
        pol = self._policy
        if pol.require_selftest and not s.selftest_passed:
            raise ProsecutorGateError("Φ₂₃: autoprueba metamórfica no superada; veredictos no fiables.")
        if not (math.isfinite(s.lip_certified) and 0.0 <= s.lip_certified < pol.banach_lip_max):
            raise ProsecutorGateError(
                f"Φ₂₃: cota de Banach fuera de [0, {pol.banach_lip_max}): {s.lip_certified}."
            )
        if not math.isfinite(s.fixed_point) or not (s.a_posteriori_error >= 0.0):
            raise ProsecutorGateError("Φ₂₃: punto fijo o error a posteriori no válidos.")
        if s.spohn_entropy_prod < _SPOHN_TOL:
            raise ProsecutorGateError(
                f"Φ₂₃: producción de entropía negativa ({s.spohn_entropy_prod}); error numérico o de motor."
            )
        if s.raw_token_count <= 0 or not (0 < s.compressed_token_count <= s.raw_token_count):
            raise ProsecutorGateError("Φ₂₃: conteo de tokens incoherente.")
        expected_pct = 100.0 * (1.0 - s.compressed_token_count / s.raw_token_count)
        if abs(s.kv_cache_compression_pct - expected_pct) > _KV_COMPRESSION_TOL:
            raise ProsecutorGateError("Φ₂₃: compresión KV no coincide con sus tokens.")
        if s.crowbar_triggered and s.truth_value is not HeytingTruthValue.VETOED:
            raise ProsecutorGateError("Φ₂₃: crowbar disparado sin veredicto VETOED.")
        if s.crowbar_triggered and s.crowbar_latency_budget_ns > pol.crowbar_budget_ns:
            raise ProsecutorGateError("Φ₂₃: presupuesto de crowbar excedido.")

    def continue_to_phase3_gate(
        self,
        p2: ToonWeaverProsecutorPhase2Certificate,
        admitted_p1: ToonWeaverProsecutorPhase1Certificate,
    ) -> ToonWeaverProsecutorPhase2Certificate:
        """Φ₂₃: integridad de P2, binding P1→P2, invariantes espectrales."""
        self._verify_chain_and_signature(
            "Φ₂₃", _CHAIN_DOMAIN_P2, _HMAC_DOMAIN_P2, p2.p1_chain_hash,
            p2.signed, p2.chain_hash, p2.hmac_signature,
        )
        if p2.p1_chain_hash != admitted_p1.chain_hash:
            raise ProsecutorGateError("Φ₂₃: P2 no encadena con el P1 admitido.")
        if p2.signed.input_digest != admitted_p1.signed.input_digest:
            raise ProsecutorGateError("Φ₂₃: la entrada auditada difiere de la observada en Fase 1.")
        self._check_spectral_invariants(p2.signed)
        return p2


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — ORIENT
# ══════════════════════════════════════════════════════════════════════════════

class ToonWeaverProsecutorPhase2Orientor:
    """Fase 2 ORIENT. Ejecuta el motor sobre la entrada observada y certifica el informe."""

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovToonCartridgeWeaverProsecutorEngine,
        gate: ToonWeaverProsecutorGate,
    ) -> None:
        self._secret_key = secret_key
        self._engine = engine
        self._gate = gate

    def execute_phase2(
        self,
        p1_cert: ToonWeaverProsecutorPhase1Certificate,
        apu_cost_matrix: object,
        exponents: object,
        *,
        raw_token_count: int = 412,
        positron_hmac_valid: Optional[bool] = None,
        positron_message: Optional[bytes] = None,
        positron_tag: Optional[str] = None,
        orbital_elements: Optional[Sequence[float]] = None,
        melnikov_delta: Optional[float] = None,
        melnikov_gamma: Optional[float] = None,
        melnikov_omega: Optional[float] = None,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        kleisli_initial_lip: Optional[float] = None,
    ) -> ToonWeaverProsecutorPhase2Certificate:
        admitted = self._gate.continue_to_phase2_gate(p1_cert)  # Φ₁₂ también aquí (defensa en profundidad)

        mat = _as_cost_matrix(apu_cost_matrix)
        exps = _as_exponents(exponents)
        if isinstance(raw_token_count, bool) or not isinstance(raw_token_count, int) or raw_token_count <= 0:
            raise ValueError("raw_token_count debe ser entero positivo.")
        if _input_digest(mat, exps) != admitted.signed.input_digest:
            raise ProsecutorGateError("Fase 2: la entrada no coincide con la observada en Fase 1.")

        kwargs: Dict[str, Any] = {}
        kwargs.update(_positron_arguments(positron_hmac_valid, positron_message, positron_tag))
        kwargs.update(_numeric_arguments(orbital_elements, rsi_max_depth, kleisli_initial_lip))
        kwargs.update(_melnikov_arguments(melnikov_delta, melnikov_gamma, melnikov_omega))

        report = self._engine.execute_toon_prosecutor_audit(
            apu_cost_matrix=mat,
            exponents=exps,
            raw_token_count=raw_token_count,
            **kwargs,
        )
        return self.emit_phase2_certificate(admitted, report)

    def emit_phase2_certificate(
        self,
        p1_cert: ToonWeaverProsecutorPhase1Certificate,
        report: ToonCartridgeWeaverProsecutorSpectrumReport,
    ) -> ToonWeaverProsecutorPhase2Certificate:
        """Último método de la Fase 2: produce el certificado P2 firmado y encadenado a P1."""
        signed = _signed_spectrum_from_report(report, p1_cert.signed.input_digest)
        chain = _chain_hash(_CHAIN_DOMAIN_P2, p1_cert.chain_hash, signed)
        return ToonWeaverProsecutorPhase2Certificate(
            timestamp=time.time(),
            p1_chain_hash=p1_cert.chain_hash,
            signed=signed,
            chain_hash=chain,
            hmac_signature=_sign(self._secret_key, _HMAC_DOMAIN_P2, chain),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — DECIDE & ACT
# ══════════════════════════════════════════════════════════════════════════════

class ToonWeaverProsecutorPhase3Decider:
    """Fase 3 DECIDE & ACT. Lee solo campos firmados y emite el estado soberano."""

    def __init__(self, gate: ToonWeaverProsecutorGate) -> None:
        self._gate = gate

    def emit_sovereign_state(
        self,
        p2_cert: ToonWeaverProsecutorPhase2Certificate,
        p1_cert: ToonWeaverProsecutorPhase1Certificate,
    ) -> ToonWeaverProsecutorSovereignState:
        """Último método de la Fase 3: estado soberano con raíz de Merkle de 7 hojas fijas."""
        p2 = self._gate.continue_to_phase3_gate(p2_cert, p1_cert)
        s = p2.signed

        leaves = (
            p1_cert.chain_hash,
            p2.chain_hash,
            s.passport_hash,
            s.truth_value.name,
            "1" if s.crowbar_triggered else "0",
            s.dirac_proof_sha256,
            s.input_digest,
        )
        visceral = ToonWeaverProsecutorVisceralMetrics(
            wacc_protection_pct_policy=_POLICY_WACC_PROTECTION_PCT,
            net_liquidity_release_pct_policy=_POLICY_NET_LIQUIDITY_RELEASE_PCT,
            kv_cache_compression_pct=s.kv_cache_compression_pct,
            tokens_per_cartridge=s.compressed_token_count,
            capacity_jerk_d3C_dt3=s.capacity_jerk_d3C_dt3,
            capacity_jerk_sign=int(np.sign(s.capacity_jerk_d3C_dt3)),
            melnikov_chaos_detected=s.melnikov_chaos_detected,
            gromov_violated=s.gromov_violated,
            delaunay_valid=s.delaunay_valid,
        )
        return ToonWeaverProsecutorSovereignState(
            timestamp=time.time(),
            schema_version=_SCHEMA_VERSION,
            truth_value=s.truth_value,
            crowbar_triggered=s.crowbar_triggered,
            crowbar_latency_budget_ns=s.crowbar_latency_budget_ns,
            dirac_proof_sha256=s.dirac_proof_sha256,
            passport_hash=s.passport_hash,
            input_digest=s.input_digest,
            p1_chain_hash=p1_cert.chain_hash,
            p2_chain_hash=p2.chain_hash,
            merkle_root_hash=_merkle_root(leaves),
            rsi_accepted_depth=s.rsi_accepted_depth,
            metamorphic_selftest_passed=s.selftest_passed,
            banach_lip_certified=s.lip_certified,
            visceral_metrics=visceral,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orquestador
# ══════════════════════════════════════════════════════════════════════════════

class BoolNovikovToonCartridgeWeaverProsecutorAgent:
    """Soberano Agéntico Fiscal Inquisidor Dual de Tejeduría Sináptica TOON (v13.1.0)."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        policy: Optional[ProsecutorPolicy] = None,
    ) -> None:
        key = _resolve_secret_key(secret_key)
        self._secret_key = key
        self._policy = policy if policy is not None else ProsecutorPolicy()
        self._engine = BoolNovikovToonCartridgeWeaverProsecutorEngine(secret_key=key)
        self._gate = ToonWeaverProsecutorGate(key, self._policy)
        self._observer = ToonWeaverProsecutorPhase1Observer(key, self._policy)
        self._orientor = ToonWeaverProsecutorPhase2Orientor(key, self._engine, self._gate)
        self._decider = ToonWeaverProsecutorPhase3Decider(self._gate)
        logger.info("%s v%s iniciado.", type(self).__name__, _AGENT_VERSION)

    def run_sovereign_governance_cycle(
        self,
        rho_mac: object,
        apu_cost_matrix: object,
        exponents: object,
        *,
        raw_token_count: int = 412,
        positron_hmac_valid: Optional[bool] = None,
        positron_message: Optional[bytes] = None,
        positron_tag: Optional[str] = None,
        orbital_elements: Optional[Sequence[float]] = None,
        melnikov_delta: Optional[float] = None,
        melnikov_gamma: Optional[float] = None,
        melnikov_omega: Optional[float] = None,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        kleisli_initial_lip: Optional[float] = None,
    ) -> ToonWeaverProsecutorSovereignState:
        """Ciclo OODA trifásico con compuertas fail-closed."""
        p1 = self._observer.emit_phase1_certificate(rho_mac, apu_cost_matrix, exponents)
        admitted_p1 = self._gate.continue_to_phase2_gate(p1)

        p2 = self._orientor.execute_phase2(
            admitted_p1,
            apu_cost_matrix,
            exponents,
            raw_token_count=raw_token_count,
            positron_hmac_valid=positron_hmac_valid,
            positron_message=positron_message,
            positron_tag=positron_tag,
            orbital_elements=orbital_elements,
            melnikov_delta=melnikov_delta,
            melnikov_gamma=melnikov_gamma,
            melnikov_omega=melnikov_omega,
            rsi_max_depth=rsi_max_depth,
            kleisli_initial_lip=kleisli_initial_lip,
        )

        state = self._decider.emit_sovereign_state(p2, admitted_p1)
        logger.info(
            "Ciclo OODA completado. Veredicto=%s, Crowbar=%s, RSI depth=%d.",
            state.truth_value.name, state.crowbar_triggered, state.rsi_accepted_depth,
        )
        return state


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo y prueba de manipulación
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import dataclasses as _dc

    logging.basicConfig(level=logging.INFO)
    agent = BoolNovikovToonCartridgeWeaverProsecutorAgent(secret_key=b"demo-key-0123456789abcdef")

    dim = 4
    rho_test = np.eye(dim, dtype=np.complex128) / float(dim)
    matrix_test = np.eye(dim, dtype=np.float64) * 1.5
    exps_test = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)

    state = agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test)
    print("=== TEST AUTÓNOMO: BoolNovikovToonCartridgeWeaverProsecutorAgent v13.1.0 ===")
    print(f"Veredicto Final Heyting Ω₃      : {state.truth_value.name} ({hex(state.truth_value.value)})")
    print(f"Crowbar Triggered               : {state.crowbar_triggered}")
    print(f"Presupuesto crowbar (ns, política): {state.crowbar_latency_budget_ns:.2f}")
    print(f"Autoprueba metamórfica          : {state.metamorphic_selftest_passed}")
    print(f"Cota Banach certificada         : {state.banach_lip_certified:.5f}")
    print(f"RSI profundidad aceptada        : {state.rsi_accepted_depth}")
    print(f"Raíz Merkle SHA-256             : {state.merkle_root_hash[:16]}...")
    vm = state.visceral_metrics
    print(f"Compresión KV-Cache (%)         : {vm.kv_cache_compression_pct:.4f} ({vm.tokens_per_cartridge} tokens)")
    print(f"Signo jerk medido               : {vm.capacity_jerk_sign:+d}")
    print("=== TEST COMPLETADO ===")

    # Prueba de manipulación: alterar un campo firmado de P1 debe ser rechazado.
    p1 = agent._observer.emit_phase1_certificate(rho_test, matrix_test, exps_test)
    forged = _dc.replace(p1, signed=_dc.replace(p1.signed, tr_rho_real=0.5))
    try:
        agent._gate.continue_to_phase2_gate(forged)
    except ProsecutorGateError as exc:
        print(f"Manipulación P1 rechazada: {exc}")
    else:
        raise SystemExit("FALLO DE INTEGRIDAD: manipulación de P1 aceptada.")