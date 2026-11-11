# -*- coding: utf-8 -*-
r"""
Módulo    : BoolNovikovToonCartridgeWeaverAgent v15.0.0-PhD
Ubicación : app/agents/wisdom/bool_novikov_toon_cartridge_weaver_agent.py
Versión   : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-AGENT-PhD
Motor      : BoolNovikovToonCartridgeWeaverEngine v15.0.0 (contrato de informe verificado en ejecución)
Gobernanza : OODA trifásica (Observe → Orient → Decide&Act) con compuertas Φ₁₂ y Φ₂₃.

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  F1 ToonWeaverPhase1Observer
     └ emit_phase1_certificate   ÚLTIMO de F1 → Φ₁₂ → execute_phase2 (PRIMERO de F2)
  F2 ToonWeaverPhase2Orientor
     └ emit_phase2_certificate   ÚLTIMO de F2 → Φ₂₃ → execute_phase3 (PRIMERO de F3)
  F3 ToonWeaverPhase3Decider
     └ emit_sovereign_state      ÚLTIMO de F3 (estado soberano firmado con hojas Merkle)

LÍMITES DECLARADOS:
  · El agente NO actúa hardware. El interlock es un presupuesto de control SIMULADO.
  · La admisibilidad de ρ es comprobación de invariantes sobre la matriz dada, no un teorema.
  · ρ NO alimenta el motor: el acoplamiento ρ↔motor es solo la conjunción de Heyting en F3.
  · Los objetivos WACC/liquidez son política declarada, no resultados derivados del espectro.
  · Una tercera diferencia finita es un indicador local; no prueba crecimiento super-exponencial.
  · El agente no implementa RSI³ ni mecánica de Poincaré: esas capacidades viven en el motor
    y aquí solo se consumen sus certificados.
  · Los nombres de campos del informe del motor son un CONTRATO: si difieren, el constructor
    falla cerrado en la autoprueba.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Importación del motor espectral (dos rutas de paquete) ─────────────────────
try:
    from bool_novikov_toon_cartridge_weaver_engine import (
        BoolNovikovToonCartridgeWeaverEngine,
        HeytingTruthValue,
        ToonCartridgeWeaverSpectrumReport,
    )
except ImportError:  # pragma: no cover - ruta de paquete alternativa
    try:
        from app.wisdom.bool_novikov_toon_cartridge_weaver_engine import (
            BoolNovikovToonCartridgeWeaverEngine,
            HeytingTruthValue,
            ToonCartridgeWeaverSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError("No se pudo importar bool_novikov_toon_cartridge_weaver_engine.") from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovToonCartridgeWeaverAgent")

# ── Versiones, dominios HMAC y Merkle ──────────────────────────────────────────
_AGENT_VERSION: Final[str] = "15.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-AGENT-PhD"
version: Final[str] = _AGENT_VERSION
_AGENT_SCHEMA: Final[str] = "agent-cert-v15.0"
_HMAC_KEY_ENV: Final[str] = "APU_TOON_WEAVER_AGENT_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_AGENT_DEV_ONLY_TOON_WEAVER_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-WEAVER-AGENT-P1-ISOMETRY-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-WEAVER-AGENT-P2-SPECTRUM-v15.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"TOON-WEAVER-AGENT-P3-SOVEREIGN-v15.0|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"

# ── Tolerancias de Fase 1 ──────────────────────────────────────────────────────
_TRACE_TOL: Final[float] = 1.0e-10
_HERMITICITY_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_PURITY_TOL: Final[float] = 1.0e-10

# ── Interlock crowbar: presupuesto de CONTROL determinista y SIMULADO ──────────
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_GPIO_WRITE_NS: Final[float] = 62.5     # 15 ciclos @ 240 MHz (política)
_CROWBAR_SIMD_EVAL_NS: Final[float] = 125.0     # 30 ciclos Xtensa LX7 (política)
_CROWBAR_GATE_DRIVE_NS: Final[float] = 200.0    # drive de compuerta (política)
_CROWBAR_BUDGET_NS: Final[float] = (
    _CROWBAR_GPIO_WRITE_NS + _CROWBAR_SIMD_EVAL_NS + _CROWBAR_GATE_DRIVE_NS
)

# ── Objetivos de política Φ_sem (declarados, no medidos) ───────────────────────
_WACC_PROTECTION_TARGET_PCT: Final[float] = 15.0
_NET_LIQUIDITY_TARGET_PCT: Final[float] = 11.5
_POLICY_PROVENANCE: Final[str] = "POLITICA_DECLARADA_v15.0 (objetivos, no mediciones)"

# ── Contrato del informe del motor (verificado en ejecución, fail-closed) ──────
_ENGINE_REPORT_CONTRACT: Final[Tuple[str, ...]] = (
    "input_digest",
    "passport_hash",
    "truth_value",
    "metamorphic_selftest_passed",
    "banach_contraction_lip",
    "banach_safety_lip",
    "rsi_certificate_chain",
    "kv_cache_compression_pct",
    "compressed_token_count",
    "capacity_third_difference",
)

# ── Orden canónico de campos firmados (define el hash de cada certificado) ─────
_P1_CHAIN_ORDER: Final[Tuple[str, ...]] = (
    "timestamp", "scale", "trace_tolerance", "tr_rho", "is_hermitian",
    "hermiticity_residual", "is_psd", "min_eigenvalue", "max_eigenvalue",
    "purity", "von_neumann_entropy", "is_admissible", "schema_version",
)
_P2_CHAIN_ORDER: Final[Tuple[str, ...]] = (
    "timestamp", "p1_chain_hash", "p1_hmac_signature", "p1_admissible",
    "engine_truth", "engine_selftest_passed", "input_digest", "passport_hash",
    "banach_contraction_lip", "banach_safety_lip", "rsi_certificate_chain",
    "kv_cache_compression_pct", "compressed_token_count",
    "capacity_third_difference", "schema_version",
)
_METRICS_ORDER: Final[Tuple[str, ...]] = (
    "wacc_protection_target_pct", "net_liquidity_release_target_pct",
    "kv_cache_compression_pct", "token_count", "capacity_third_difference",
    "capacity_jerk_status", "policy_provenance",
)
_STATE_CORE_ORDER: Final[Tuple[str, ...]] = (
    "timestamp", "truth_value", "engine_truth_value", "p1_admissible",
    "is_esp32_crowbar_triggered", "crowbar_actuation_latency_ns",
    "crowbar_latency_is_simulated", "p1_chain_hash", "p2_chain_hash",
    "merkle_root_hash", "schema_version",
)

# ── Índices de hojas Merkle del estado soberano ────────────────────────────────
_N_STATE_LEAVES: Final[int] = 8
_LEAF_P1: Final[int] = 0
_LEAF_P2: Final[int] = 1
_LEAF_PASSPORT: Final[int] = 2
_LEAF_INPUT: Final[int] = 3
_LEAF_VERDICT: Final[int] = 4
_LEAF_CROWBAR: Final[int] = 5
_LEAF_METRICS: Final[int] = 6
_LEAF_RSI: Final[int] = 7


# ══════════════════════════════════════════════════════════════════════════════
# Primitivas: clave, canonicalización, HMAC, Merkle y validadores
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_secret_key(explicit: Optional[bytes], allow_dev_key: bool) -> bytes:
    """Clave explícita, o entorno. Sin ninguna: ValueError, salvo allow_dev_key=True."""
    if explicit is not None:
        if not isinstance(explicit, (bytes, bytearray)):
            raise TypeError("secret_key debe ser bytes.")
        key = bytes(explicit)
    else:
        env_val = os.environ.get(_HMAC_KEY_ENV, "")
        if env_val:
            key = env_val.encode("utf-8")
        elif allow_dev_key:
            logger.warning("Sin %s: clave de DESARROLLO (no apta para producción).", _HMAC_KEY_ENV)
            key = _DEV_SECRET_KEY
        else:
            raise ValueError(
                f"Sin clave HMAC: defina {_HMAC_KEY_ENV} o pase secret_key "
                "(allow_dev_key=True solo para desarrollo)."
            )
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"Clave HMAC demasiado corta: {len(key)} < {_MIN_KEY_BYTES} bytes.")
    return key


def _fmt(value: object) -> str:
    """Serialización canónica de un campo. Rechaza no finitos y el separador '|'."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "none"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}:{int(value.value)}"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Campo canónico no finito (fail-closed).")
        return f"{value:.17g}"
    if isinstance(value, str):
        if "|" in value:
            raise ValueError("Separador '|' prohibido en campo canónico.")
        return value
    raise TypeError(f"Tipo no canónico: {type(value).__name__}")


def _canonical_digest(payload: Mapping[str, object], order: Tuple[str, ...]) -> str:
    """SHA-256 sobre la cabecera de nombres y los valores canónicos en orden fijo.
    Incluir los nombres en la cabecera detecta cualquier desalineación de esquema."""
    if any(key not in payload for key in order):
        raise KeyError("Campo canónico ausente en el payload.")
    header = ",".join(order)
    body = "|".join(_fmt(payload[key]) for key in order)
    return hashlib.sha256(f"{header}#{body}".encode("utf-8")).hexdigest()


def _payload(obj: object, order: Tuple[str, ...]) -> Dict[str, object]:
    """Extrae del objeto los campos firmados, en el orden canónico."""
    return {key: getattr(obj, key) for key in order}


def _sign(secret_key: bytes, domain: bytes, chain_hash: str) -> str:
    return hmac.new(secret_key, domain + chain_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def _const_eq(a: object, b: object) -> bool:
    """Comparación en tiempo constante; falso si alguno no es str."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _merkle_root(leaves: Sequence[str]) -> str:
    """Raíz Merkle binaria iterativa: H(0x00‖hoja), H(0x01‖izq‖der); nodo impar se promueve.
    Equivale al árbol de RFC-6962 (verificado en la autoprueba frente a la referencia recursiva)."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(_MERKLE_LEAF + s.encode("utf-8")).digest() for s in leaves]
    while len(level) > 1:
        nxt: List[bytes] = [
            hashlib.sha256(_MERKLE_NODE + level[i] + level[i + 1]).digest()
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def _merkle_root_reference(leaves: Sequence[str]) -> str:
    """MTH de RFC-6962 §2.1: división recursiva en la mayor potencia de 2 menor que n.
    Referencia de validación cruzada; no se usa en el camino de producción."""

    def mth(lo: int, hi: int) -> bytes:
        n = hi - lo
        if n == 1:
            return hashlib.sha256(_MERKLE_LEAF + leaves[lo].encode("utf-8")).digest()
        k = 1
        while k * 2 < n:
            k *= 2
        return hashlib.sha256(_MERKLE_NODE + mth(lo, lo + k) + mth(lo + k, hi)).digest()

    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    return mth(0, len(leaves)).hex()


def _require_int(name: str, value: object, lo: int, hi: Optional[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} debe ser entero.")
    v = int(value)
    if v < lo or (hi is not None and v > hi):
        raise ValueError(f"{name} fuera de rango [{lo}, {'∞' if hi is None else hi}].")
    return v


def _require_finite(name: str, value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} debe ser real.") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} debe ser finito.")
    return v


# ── Predicados físicos compartidos (emisión y re-derivación en Φ₁₂) ────────────
def _hermitian_ok(residual: float, scale: float) -> bool:
    return bool(residual <= _HERMITICITY_TOL * scale)


def _psd_ok(min_eigenvalue: float, scale: float) -> bool:
    return bool(min_eigenvalue >= -_PSD_TOL * scale)


def _derive_p1_admissibility(
    *, is_hermitian: bool, is_psd: bool, tr_rho: float, trace_tolerance: float, purity: float
) -> bool:
    """Admisibilidad = Hermítica ∧ PSD ∧ |Tr ρ − 1| ≤ tol ∧ Tr ρ² ≤ 1 + tol."""
    return bool(
        is_hermitian
        and is_psd
        and abs(tr_rho - 1.0) <= trace_tolerance
        and purity <= 1.0 + _PURITY_TOL
    )


def _final_verdict(engine_truth: HeytingTruthValue, p1_admissible: bool) -> HeytingTruthValue:
    """Veredicto final = meet(veredicto del motor, ⊤ si ρ admisible else ⊥)."""
    p1_verdict = HeytingTruthValue.COHERENT if p1_admissible else HeytingTruthValue.VETOED
    return engine_truth.meet(p1_verdict)


def _crowbar_leaf(triggered: bool, latency_ns: float) -> str:
    return f"crowbar:{int(bool(triggered))}:{_fmt(float(latency_ns))}"


def _metrics_digest(metrics: "ToonWeaverVisceralMetrics") -> str:
    return _canonical_digest(_payload(metrics, _METRICS_ORDER), _METRICS_ORDER)


def _require_report_contract(report: ToonCartridgeWeaverSpectrumReport) -> None:
    """Fail-closed: el informe del motor debe exponer el contrato completo."""
    missing = [name for name in _ENGINE_REPORT_CONTRACT if not hasattr(report, name)]
    if missing:
        raise RuntimeError(f"Contrato del informe del motor incumplido (fail-closed): faltan {missing}.")
    if not isinstance(report.truth_value, HeytingTruthValue):
        raise TypeError("truth_value del informe del motor no es HeytingTruthValue.")


def _engine_fields(report: ToonCartridgeWeaverSpectrumReport) -> Dict[str, object]:
    """Proyección canónica del informe del motor sobre los campos firmados de P2."""
    _require_report_contract(report)
    jerk = report.capacity_third_difference
    return {
        "engine_truth": report.truth_value,
        "engine_selftest_passed": bool(report.metamorphic_selftest_passed),
        "input_digest": str(report.input_digest),
        "passport_hash": str(report.passport_hash),
        "banach_contraction_lip": float(report.banach_contraction_lip),
        "banach_safety_lip": float(report.banach_safety_lip),
        "rsi_certificate_chain": str(report.rsi_certificate_chain),
        "kv_cache_compression_pct": float(report.kv_cache_compression_pct),
        "compressed_token_count": int(report.compressed_token_count),
        "capacity_third_difference": None if jerk is None else float(jerk),
    }


def _state_leaves(
    p2: "ToonWeaverPhase2Certificate",
    final_truth: HeytingTruthValue,
    crowbar_triggered: bool,
    crowbar_latency: float,
    metrics: "ToonWeaverVisceralMetrics",
) -> Tuple[str, ...]:
    """Hojas del estado en el orden de los índices _LEAF_*."""
    return (
        p2.p1_chain_hash,                                   # _LEAF_P1
        p2.chain_hash,                                      # _LEAF_P2
        p2.passport_hash,                                   # _LEAF_PASSPORT
        p2.input_digest,                                    # _LEAF_INPUT
        f"verdict:{final_truth.value}",                     # _LEAF_VERDICT
        _crowbar_leaf(crowbar_triggered, crowbar_latency),  # _LEAF_CROWBAR
        _metrics_digest(metrics),                           # _LEAF_METRICS
        p2.rsi_certificate_chain,                           # _LEAF_RSI
    )


def _verify_p1(cert: "ToonWeaverPhase1Certificate", secret_key: bytes) -> None:
    """Φ₁₂ (núcleo): contenido → hash → firma → re-derivación semántica. Lanza ValueError."""
    if not isinstance(cert, ToonWeaverPhase1Certificate):
        raise TypeError("Se exige un ToonWeaverPhase1Certificate.")
    if cert.schema_version != _AGENT_SCHEMA:
        raise ValueError("Φ₁₂: esquema de P1 no reconocido.")
    chain = _canonical_digest(_payload(cert, _P1_CHAIN_ORDER), _P1_CHAIN_ORDER)
    if not _const_eq(chain, cert.chain_hash):
        raise ValueError("Φ₁₂: el hash de cadena P1 no vincula el contenido (tamper).")
    if not _const_eq(_sign(secret_key, _HMAC_DOMAIN_P1, chain), cert.hmac_signature):
        raise ValueError("Φ₁₂: falla de autenticidad HMAC en P1.")
    if cert.is_hermitian != _hermitian_ok(cert.hermiticity_residual, cert.scale):
        raise ValueError("Φ₁₂: is_hermitian no derivable de la evidencia.")
    if cert.is_psd != _psd_ok(cert.min_eigenvalue, cert.scale):
        raise ValueError("Φ₁₂: is_psd no derivable de la evidencia.")
    derived = _derive_p1_admissibility(
        is_hermitian=cert.is_hermitian, is_psd=cert.is_psd, tr_rho=cert.tr_rho,
        trace_tolerance=cert.trace_tolerance, purity=cert.purity,
    )
    if cert.is_admissible != derived:
        raise ValueError("Φ₁₂: admisibilidad no derivable (certificado incoherente).")


def _verify_p2_integrity(cert: "ToonWeaverPhase2Certificate", secret_key: bytes) -> None:
    """Integridad de P2 (hash y firma). No verifica el enlace con P1."""
    if not isinstance(cert, ToonWeaverPhase2Certificate):
        raise TypeError("Se exige un ToonWeaverPhase2Certificate.")
    if cert.schema_version != _AGENT_SCHEMA:
        raise ValueError("Φ₂₃: esquema de P2 no reconocido.")
    chain = _canonical_digest(_payload(cert, _P2_CHAIN_ORDER), _P2_CHAIN_ORDER)
    if not _const_eq(chain, cert.chain_hash):
        raise ValueError("Φ₂₃: el hash de cadena P2 no vincula el contenido (tamper).")
    if not _const_eq(_sign(secret_key, _HMAC_DOMAIN_P2, chain), cert.hmac_signature):
        raise ValueError("Φ₂₃: falla de autenticidad HMAC en P2.")


def _verify_p2_binding(cert: "ToonWeaverPhase2Certificate", p1: "ToonWeaverPhase1Certificate") -> None:
    """P2 debe haberse construido sobre exactamente este P1 (hash, firma y admisibilidad)."""
    if not (
        _const_eq(cert.p1_chain_hash, p1.chain_hash)
        and _const_eq(cert.p1_hmac_signature, p1.hmac_signature)
        and cert.p1_admissible == p1.is_admissible
    ):
        raise ValueError("Φ₂₃: P2 no está vinculado al P1 de la cadena.")


def _resealed(obj: object, order: Tuple[str, ...], domain: bytes, key: bytes, **changes: object) -> object:
    """Re-sella un certificado con cambios (solo para la autoprueba de adversario con clave)."""
    payload = _payload(obj, order)
    payload.update(changes)
    chain = _canonical_digest(payload, order)
    return replace(obj, **changes, chain_hash=chain, hmac_signature=_sign(key, domain, chain))  # type: ignore[type-var]


def _ok(fn: Callable[[], object]) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False


def _raises_value_error(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Contratos: certificados, política, métricas y estado soberano
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ToonWeaverPhase1Certificate:
    """Certificado P1 (Observe): invariantes físicos firmados de ρ."""

    timestamp: float
    scale: float
    trace_tolerance: float
    tr_rho: float
    is_hermitian: bool
    hermiticity_residual: float
    is_psd: bool
    min_eigenvalue: float
    max_eigenvalue: float
    purity: float
    von_neumann_entropy: float
    is_admissible: bool
    schema_version: str
    chain_hash: str
    hmac_signature: str


@dataclass(frozen=True)
class ToonWeaverPhase2Certificate:
    """Certificado P2 (Orient): campos firmados del informe del motor, ligados a P1."""

    timestamp: float
    p1_chain_hash: str
    p1_hmac_signature: str
    p1_admissible: bool
    engine_truth: HeytingTruthValue
    engine_selftest_passed: bool
    input_digest: str
    passport_hash: str
    banach_contraction_lip: float
    banach_safety_lip: float
    rsi_certificate_chain: str
    kv_cache_compression_pct: float
    compressed_token_count: int
    capacity_third_difference: Optional[float]
    schema_version: str
    chain_hash: str
    hmac_signature: str
    spectrum_report: ToonCartridgeWeaverSpectrumReport = field(compare=False, repr=False)


@dataclass(frozen=True)
class ToonWeaverAuditPolicy:
    """Política de auditoría validada. Reenvía al motor todos los parámetros por palabra clave.
    orbital_elements = (a, e, i, M, ω, Ω); secular_system = ((m, a), ...)."""

    raw_token_count: int = 412
    positron_hmac_valid: Optional[bool] = None
    orbital_elements: Optional[Tuple[float, ...]] = None
    ball_r: float = 0.35
    cylinder_R: float = 0.50
    melnikov_delta: float = 0.01
    melnikov_gamma: float = 0.02
    melnikov_omega: float = 1.0
    pendulum_amplitude: float = 0.2
    poincare_periods: int = 256
    poincare_steps_per_period: int = 64
    secular_system: Optional[Tuple[Tuple[float, float], ...]] = None
    rsi_max_depth: int = 3
    initial_lip: float = 0.95
    capacity_series: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_token_count", _require_int("raw_token_count", self.raw_token_count, 56, None))
        object.__setattr__(self, "rsi_max_depth", _require_int("rsi_max_depth", self.rsi_max_depth, 0, 3))
        object.__setattr__(self, "poincare_periods", _require_int("poincare_periods", self.poincare_periods, 0, 20000))
        object.__setattr__(
            self, "poincare_steps_per_period",
            _require_int("poincare_steps_per_period", self.poincare_steps_per_period, 8, 4096),
        )
        lip = _require_finite("initial_lip", self.initial_lip)
        if not 0.0 <= lip < 1.0:
            raise ValueError("initial_lip ∈ [0,1).")
        object.__setattr__(self, "initial_lip", lip)
        for name in ("ball_r", "cylinder_R", "melnikov_delta", "melnikov_gamma",
                     "melnikov_omega", "pendulum_amplitude"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if not (0.0 < self.ball_r <= self.cylinder_R):
            raise ValueError("Política de diseño (Gromov): se exige 0 < ball_r ≤ cylinder_R.")
        if self.positron_hmac_valid is not None and not isinstance(self.positron_hmac_valid, bool):
            raise TypeError("positron_hmac_valid debe ser bool o None.")
        if self.orbital_elements is not None:
            orb = tuple(_require_finite("orbital_elements", x) for x in self.orbital_elements)
            if len(orb) != 6:
                raise ValueError("orbital_elements requiere 6 valores (a, e, i, M, ω, Ω).")
            if orb[0] <= 0.0 or not 0.0 <= orb[1] < 1.0:
                raise ValueError("orbital_elements: se exige a > 0 y e ∈ [0, 1).")
            object.__setattr__(self, "orbital_elements", orb)
        if self.secular_system is not None:
            planets: List[Tuple[float, float]] = []
            for pair in self.secular_system:
                if len(pair) != 2:
                    raise ValueError("secular_system: cada planeta es (m, a).")
                m = _require_finite("masa", pair[0])
                a = _require_finite("semieje", pair[1])
                if m <= 0.0 or a <= 0.0:
                    raise ValueError("secular_system: masa y semieje deben ser > 0.")
                planets.append((m, a))
            if not 1 <= len(planets) <= 8:
                raise ValueError("secular_system: entre 1 y 8 planetas.")
            object.__setattr__(self, "secular_system", tuple(planets))
        if self.capacity_series is not None:
            cap = tuple(_require_finite("capacity_series", x) for x in self.capacity_series)
            if len(cap) < 4:
                raise ValueError("capacity_series requiere ≥ 4 muestras (tercera diferencia).")
            object.__setattr__(self, "capacity_series", cap)

    def to_engine_kwargs(self) -> Dict[str, object]:
        """Palabras clave para el motor (nunca posicionales)."""
        return {
            "raw_token_count": self.raw_token_count,
            "positron_hmac_valid": self.positron_hmac_valid,
            "orbital_elements": self.orbital_elements,
            "ball_r": self.ball_r,
            "cylinder_R": self.cylinder_R,
            "melnikov_delta": self.melnikov_delta,
            "melnikov_gamma": self.melnikov_gamma,
            "melnikov_omega": self.melnikov_omega,
            "pendulum_amplitude": self.pendulum_amplitude,
            "poincare_periods": self.poincare_periods,
            "poincare_steps_per_period": self.poincare_steps_per_period,
            "secular_system": self.secular_system,
            "rsi_max_depth": self.rsi_max_depth,
            "initial_lip": self.initial_lip,
            "capacity_series": self.capacity_series,
        }


@dataclass(frozen=True)
class ToonWeaverVisceralMetrics:
    """Métricas Φ_sem: objetivos de política declarados y magnitudes medidas."""

    wacc_protection_target_pct: float
    net_liquidity_release_target_pct: float
    kv_cache_compression_pct: float
    token_count: int
    capacity_third_difference: Optional[float]
    capacity_jerk_status: str
    policy_provenance: str


@dataclass(frozen=True)
class ToonWeaverSovereignState:
    """Estado soberano (Decide & Act). Lleva sus hojas Merkle para re-verificación completa."""

    timestamp: float
    truth_value: HeytingTruthValue
    engine_truth_value: HeytingTruthValue
    p1_admissible: bool
    is_esp32_crowbar_triggered: bool
    crowbar_actuation_latency_ns: float
    crowbar_latency_is_simulated: bool
    p1_chain_hash: str
    p2_chain_hash: str
    merkle_leaves: Tuple[str, ...]
    merkle_root_hash: str
    visceral_metrics: ToonWeaverVisceralMetrics
    state_signature: str
    schema_version: str


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — OBSERVE: invariantes físicos de ρ
# ══════════════════════════════════════════════════════════════════════════════
class ToonWeaverPhase1Observer:
    """Fase 1 OODA. El espectro se calcula sobre la parte Hermítica de ρ."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128], trace_tolerance: float = _TRACE_TOL
    ) -> ToonWeaverPhase1Certificate:
        tol = _require_finite("trace_tolerance", trace_tolerance)
        if tol < 0.0:
            raise ValueError("trace_tolerance ≥ 0.")
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 1:
            raise ValueError("rho_mac debe ser cuadrada 2D no vacía.")
        if not np.all(np.isfinite(rho)):
            raise ValueError("rho_mac contiene valores no finitos.")
        scale = max(1.0, float(np.max(np.abs(rho))))
        residual = float(np.max(np.abs(rho - rho.conj().T)))
        rho_h = 0.5 * (rho + rho.conj().T)
        evals = np.linalg.eigvalsh(rho_h)
        pos = evals[evals > 0.0]
        return self.emit_phase1_certificate(
            timestamp=float(self._clock()),
            scale=scale,
            trace_tolerance=tol,
            tr_rho=float(np.real(np.trace(rho_h))),
            hermiticity_residual=residual,
            min_eigenvalue=float(evals[0]),
            max_eigenvalue=float(evals[-1]),
            purity=float(np.sum(evals ** 2)),
            von_neumann_entropy=float(-np.sum(pos * np.log(pos))) + 0.0,
        )

    # ── ÚLTIMO método de Fase 1 → Φ₁₂ → Fase 2 ─────────────────────────────────
    def emit_phase1_certificate(
        self,
        *,
        timestamp: float,
        scale: float,
        trace_tolerance: float,
        tr_rho: float,
        hermiticity_residual: float,
        min_eigenvalue: float,
        max_eigenvalue: float,
        purity: float,
        von_neumann_entropy: float,
    ) -> ToonWeaverPhase1Certificate:
        """ÚLTIMO método de F1. Deriva las banderas (no se aceptan del llamador) y firma P1."""
        scale_v = _require_finite("scale", scale)
        tol_v = _require_finite("trace_tolerance", trace_tolerance)
        if scale_v < 1.0 or tol_v < 0.0:
            raise ValueError("P1: se exige scale ≥ 1 y trace_tolerance ≥ 0.")
        is_herm = _hermitian_ok(hermiticity_residual, scale_v)
        is_psd = _psd_ok(min_eigenvalue, scale_v)
        is_adm = _derive_p1_admissibility(
            is_hermitian=is_herm, is_psd=is_psd, tr_rho=tr_rho,
            trace_tolerance=tol_v, purity=purity,
        )
        payload: Dict[str, object] = {
            "timestamp": float(timestamp),
            "scale": scale_v,
            "trace_tolerance": tol_v,
            "tr_rho": float(tr_rho),
            "is_hermitian": is_herm,
            "hermiticity_residual": float(hermiticity_residual),
            "is_psd": is_psd,
            "min_eigenvalue": float(min_eigenvalue),
            "max_eigenvalue": float(max_eigenvalue),
            "purity": float(purity),
            "von_neumann_entropy": float(von_neumann_entropy),
            "is_admissible": is_adm,
            "schema_version": _AGENT_SCHEMA,
        }
        chain = _canonical_digest(payload, _P1_CHAIN_ORDER)
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, chain)
        return ToonWeaverPhase1Certificate(chain_hash=chain, hmac_signature=sig, **payload)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — ORIENT: ejecución del motor y vinculación espectral
# ══════════════════════════════════════════════════════════════════════════════
class ToonWeaverPhase2Orientor:
    """Fase 2 OODA. Ejecuta el motor v15 y firma su informe ligado a P1."""

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovToonCartridgeWeaverEngine,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = secret_key
        self._engine = engine
        self._clock = clock

    # ── PRIMER método de Fase 2 (consume P1 vía Φ₁₂) ───────────────────────────
    def execute_phase2(
        self,
        p1_cert: ToonWeaverPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        policy: ToonWeaverAuditPolicy,
    ) -> ToonWeaverPhase2Certificate:
        _verify_p1(p1_cert, self._secret_key)  # defensa en profundidad
        if not isinstance(policy, ToonWeaverAuditPolicy):
            raise TypeError("Fase 2 exige un ToonWeaverAuditPolicy.")
        report = self._engine.execute_toon_cartridge_weaver_audit(
            apu_cost_matrix, exponents, **policy.to_engine_kwargs()
        )
        return self.emit_phase2_certificate(p1_cert, report, float(self._clock()))

    # ── ÚLTIMO método de Fase 2 → Φ₂₃ → Fase 3 ─────────────────────────────────
    def emit_phase2_certificate(
        self,
        p1_cert: ToonWeaverPhase1Certificate,
        report: ToonCartridgeWeaverSpectrumReport,
        timestamp: float,
    ) -> ToonWeaverPhase2Certificate:
        """ÚLTIMO método de F2. Firma P2 sobre los campos del motor y el enlace a P1."""
        _verify_p1(p1_cert, self._secret_key)
        payload: Dict[str, object] = {
            "timestamp": float(timestamp),
            "p1_chain_hash": p1_cert.chain_hash,
            "p1_hmac_signature": p1_cert.hmac_signature,
            "p1_admissible": bool(p1_cert.is_admissible),
            **_engine_fields(report),
            "schema_version": _AGENT_SCHEMA,
        }
        chain = _canonical_digest(payload, _P2_CHAIN_ORDER)
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, chain)
        return ToonWeaverPhase2Certificate(
            chain_hash=chain, hmac_signature=sig, spectrum_report=report, **payload  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — DECIDE & ACT: Heyting, interlock simulado, Φ_sem y estado firmado
# ══════════════════════════════════════════════════════════════════════════════
class ToonWeaverPhase3Decider:
    """Fase 3 OODA. Compone el veredicto, simula el interlock y firma el estado."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    # ── PRIMER método de Fase 3 (consume P2 vía Φ₂₃) ───────────────────────────
    def execute_phase3(self, p2_cert: ToonWeaverPhase2Certificate) -> ToonWeaverSovereignState:
        _verify_p2_integrity(p2_cert, self._secret_key)
        final_truth = _final_verdict(p2_cert.engine_truth, p2_cert.p1_admissible)
        triggered = final_truth == HeytingTruthValue.VETOED
        latency = _CROWBAR_BUDGET_NS if triggered else 0.0
        if triggered:
            logger.warning(
                "[INTERLOCK SIMULADO] Crowbar GPIO14/BT151 en %.2f ns (presupuesto, NO hardware).",
                latency,
            )
        return self.emit_sovereign_state(
            timestamp=float(self._clock()),
            p2_cert=p2_cert,
            final_truth=final_truth,
            crowbar_triggered=triggered,
            crowbar_latency=latency,
            metrics=self._visceral_metrics(p2_cert),
        )

    @staticmethod
    def _visceral_metrics(p2_cert: ToonWeaverPhase2Certificate) -> ToonWeaverVisceralMetrics:
        """Φ_sem: objetivos declarados + compresión, tokens y jerk medidos (firmados en P2)."""
        jerk = p2_cert.capacity_third_difference
        if jerk is None:
            status = "JERK_NO_SERIES (sin serie suministrada)"
        else:
            sign = "+" if jerk > 0.0 else ("-" if jerk < 0.0 else "0")
            status = (
                f"JERK_d3C_dt3={jerk:.6g}_sign={sign} "
                "(indicador local; no prueba crecimiento super-exponencial)"
            )
        return ToonWeaverVisceralMetrics(
            wacc_protection_target_pct=_WACC_PROTECTION_TARGET_PCT,
            net_liquidity_release_target_pct=_NET_LIQUIDITY_TARGET_PCT,
            kv_cache_compression_pct=float(p2_cert.kv_cache_compression_pct),
            token_count=int(p2_cert.compressed_token_count),
            capacity_third_difference=None if jerk is None else float(jerk),
            capacity_jerk_status=status,
            policy_provenance=_POLICY_PROVENANCE,
        )

    # ── ÚLTIMO método de Fase 3: estado soberano firmado ───────────────────────
    def emit_sovereign_state(
        self,
        *,
        timestamp: float,
        p2_cert: ToonWeaverPhase2Certificate,
        final_truth: HeytingTruthValue,
        crowbar_triggered: bool,
        crowbar_latency: float,
        metrics: ToonWeaverVisceralMetrics,
    ) -> ToonWeaverSovereignState:
        """ÚLTIMO método de F3. Re-deriva veredicto e interlock, construye la raíz Merkle y firma."""
        if not isinstance(p2_cert, ToonWeaverPhase2Certificate):
            raise TypeError("F3 exige un ToonWeaverPhase2Certificate.")
        if final_truth != _final_verdict(p2_cert.engine_truth, p2_cert.p1_admissible):
            raise ValueError("F3: veredicto final no derivable del motor y de P1.")
        expected_triggered = final_truth == HeytingTruthValue.VETOED
        if bool(crowbar_triggered) != expected_triggered:
            raise ValueError("F3: disparo de interlock inconsistente con el veredicto.")
        if float(crowbar_latency) != (_CROWBAR_BUDGET_NS if expected_triggered else 0.0):
            raise ValueError("F3: latencia de interlock inconsistente con el disparo.")

        leaves = _state_leaves(p2_cert, final_truth, expected_triggered, float(crowbar_latency), metrics)
        root = _merkle_root(leaves)
        core: Dict[str, object] = {
            "timestamp": float(timestamp),
            "truth_value": final_truth,
            "engine_truth_value": p2_cert.engine_truth,
            "p1_admissible": bool(p2_cert.p1_admissible),
            "is_esp32_crowbar_triggered": expected_triggered,
            "crowbar_actuation_latency_ns": float(crowbar_latency),
            "crowbar_latency_is_simulated": True,
            "p1_chain_hash": p2_cert.p1_chain_hash,
            "p2_chain_hash": p2_cert.chain_hash,
            "merkle_root_hash": root,
            "schema_version": _AGENT_SCHEMA,
        }
        state_sig = _sign(self._secret_key, _HMAC_DOMAIN_P3, _canonical_digest(core, _STATE_CORE_ORDER))
        return ToonWeaverSovereignState(
            **core,  # type: ignore[arg-type]
            merkle_leaves=leaves,
            visceral_metrics=metrics,
            state_signature=state_sig,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orquestador soberano: ciclo OODA con compuertas Φ₁₂ y Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonCartridgeWeaverAgent:
    """Soberano agéntico tejedor TOON (56 tokens). Composición de las tres fases OODA."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovToonCartridgeWeaverEngine] = None,
        clock: Callable[[], float] = time.time,
        allow_dev_key: bool = False,
    ) -> None:
        self._secret_key = _resolve_secret_key(secret_key, allow_dev_key)
        if engine is None:
            engine = BoolNovikovToonCartridgeWeaverEngine(secret_key=self._secret_key)
        if not callable(getattr(engine, "execute_toon_cartridge_weaver_audit", None)):
            raise TypeError("El motor debe exponer execute_toon_cartridge_weaver_audit.")
        self._engine = engine
        self._clock = clock
        self._observer = ToonWeaverPhase1Observer(self._secret_key, clock)
        self._orientor = ToonWeaverPhase2Orientor(self._secret_key, self._engine, clock)
        self._decider = ToonWeaverPhase3Decider(self._secret_key, clock)
        self._last_p2: Optional[ToonWeaverPhase2Certificate] = None
        if not self._metamorphic_selftest():
            raise RuntimeError("Autoprueba metamórfica del agente fallida (fail-closed).")
        logger.info("%s v%s iniciado correctamente.", self.__class__.__name__, _AGENT_VERSION)

    @property
    def last_phase2_certificate(self) -> Optional[ToonWeaverPhase2Certificate]:
        """Último certificado P2 emitido (para auditoría; incluye el informe del motor)."""
        return self._last_p2

    # ── Compuerta Φ₁₂: autentica P1 y re-deriva su admisibilidad ───────────────
    def continue_to_phase2_gate(self, p1: ToonWeaverPhase1Certificate) -> None:
        _verify_p1(p1, self._secret_key)

    # ── Compuerta Φ₂₃: autentica P2, lo vincula a P1 y exige estabilidad ───────
    def continue_to_phase3_gate(
        self, p1: ToonWeaverPhase1Certificate, p2: ToonWeaverPhase2Certificate
    ) -> None:
        _verify_p1(p1, self._secret_key)
        _verify_p2_integrity(p2, self._secret_key)
        _verify_p2_binding(p2, p1)
        if not p2.engine_selftest_passed:
            raise ValueError("Φ₂₃: autoprueba del motor no aprobada.")
        if p2.banach_contraction_lip > p2.banach_safety_lip:
            raise ValueError("Φ₂₃: cota exacta mayor que la cota de seguridad (inconsistente).")
        if not p2.banach_safety_lip < 1.0:
            raise ValueError("Φ₂₃: violación de estabilidad (Lip de seguridad ≥ 1).")

    def verify_state_signature(self, state: ToonWeaverSovereignState) -> bool:
        """Re-verificación interna: raíz Merkle, hojas, métricas, veredicto y firma."""
        try:
            if state.schema_version != _AGENT_SCHEMA:
                return False
            leaves = state.merkle_leaves
            if len(leaves) != _N_STATE_LEAVES or _merkle_root(leaves) != state.merkle_root_hash:
                return False
            if leaves[_LEAF_P1] != state.p1_chain_hash or leaves[_LEAF_P2] != state.p2_chain_hash:
                return False
            if leaves[_LEAF_VERDICT] != f"verdict:{state.truth_value.value}":
                return False
            if leaves[_LEAF_CROWBAR] != _crowbar_leaf(
                state.is_esp32_crowbar_triggered, state.crowbar_actuation_latency_ns
            ):
                return False
            if leaves[_LEAF_METRICS] != _metrics_digest(state.visceral_metrics):
                return False
            core = _payload(state, _STATE_CORE_ORDER)
            expected = _sign(self._secret_key, _HMAC_DOMAIN_P3, _canonical_digest(core, _STATE_CORE_ORDER))
            return _const_eq(expected, state.state_signature)
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def verify_state_binding(
        self,
        state: ToonWeaverSovereignState,
        p1: ToonWeaverPhase1Certificate,
        p2: ToonWeaverPhase2Certificate,
    ) -> bool:
        """Verificación externa: el estado proviene exactamente de este P1 y este P2."""
        try:
            leaves = state.merkle_leaves
            return bool(
                self.verify_state_signature(state)
                and _const_eq(state.p1_chain_hash, p1.chain_hash)
                and _const_eq(state.p2_chain_hash, p2.chain_hash)
                and state.p1_admissible == p1.is_admissible
                and state.engine_truth_value == p2.engine_truth
                and leaves[_LEAF_PASSPORT] == p2.passport_hash
                and leaves[_LEAF_INPUT] == p2.input_digest
                and leaves[_LEAF_RSI] == p2.rsi_certificate_chain
            )
        except (AttributeError, IndexError, TypeError):
            return False

    # ── Ciclo OODA completo ────────────────────────────────────────────────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        policy: Optional[ToonWeaverAuditPolicy] = None,
        *,
        positron_hmac_valid: Optional[bool] = None,
    ) -> ToonWeaverSovereignState:
        """Observe → Φ₁₂ → Orient → Φ₂₃ → Decide & Act sobre el fibrado Boole-Novikov."""
        base = policy if policy is not None else ToonWeaverAuditPolicy()
        if not isinstance(base, ToonWeaverAuditPolicy):
            raise TypeError("policy debe ser un ToonWeaverAuditPolicy.")
        eff = base if positron_hmac_valid is None else replace(base, positron_hmac_valid=positron_hmac_valid)
        p1 = self._observer.execute_phase1(rho_mac)
        self.continue_to_phase2_gate(p1)
        p2 = self._orientor.execute_phase2(p1, apu_cost_matrix, exponents, eff)
        self._last_p2 = p2
        self.continue_to_phase3_gate(p1, p2)
        state = self._decider.execute_phase3(p2)
        if not self.verify_state_binding(state, p1, p2):
            raise RuntimeError("Ciclo: el estado no está ligado a P1 y P2 (fail-closed).")
        logger.info("Ciclo OODA TOON Weaver completado. Veredicto: %s", state.truth_value.name)
        return state

    # ── Autoprueba metamórfica: adversario con y sin clave ─────────────────────
    def _metamorphic_selftest(self) -> bool:
        key = self._secret_key
        failures: List[str] = []

        def check(name: str, condition: bool) -> None:
            if not condition:
                failures.append(name)

        try:
            # 1. Presupuesto del interlock
            check("crowbar_budget_sum", abs(
                _CROWBAR_BUDGET_NS
                - (_CROWBAR_GPIO_WRITE_NS + _CROWBAR_SIMD_EVAL_NS + _CROWBAR_GATE_DRIVE_NS)
            ) < 1e-9)
            check("crowbar_budget_limit", _CROWBAR_BUDGET_NS <= _ESP32_CROWBAR_MAX_LATENCY_NS)

            # 2. Merkle: casos base y validación cruzada con RFC-6962 (n = 0..9)
            la = hashlib.sha256(_MERKLE_LEAF + b"aa").digest()
            lb = hashlib.sha256(_MERKLE_LEAF + b"bb").digest()
            check("merkle_empty", _merkle_root([]) == hashlib.sha256(b"").hexdigest())
            check("merkle_single", _merkle_root(["aa"]) == la.hex())
            check("merkle_pair", _merkle_root(["aa", "bb"])
                  == hashlib.sha256(_MERKLE_NODE + la + lb).hexdigest())
            check("merkle_rfc6962_crosscheck", all(
                _merkle_root(lv) == _merkle_root_reference(lv)
                for lv in ([f"h{i}" for i in range(n)] for n in range(10))
            ))

            # 3. Separación de dominios y sensibilidad a la clave
            s_p1 = _sign(key, _HMAC_DOMAIN_P1, "x")
            check("hmac_domain_separation", s_p1 != _sign(key, _HMAC_DOMAIN_P2, "x"))
            check("hmac_key_sensitivity", s_p1 != _sign(b"clave-alternativa-de-prueba", _HMAC_DOMAIN_P1, "x"))

            # 4. Canonicalización fail-closed
            check("canon_rejects_nan", _raises_value_error(lambda: _fmt(float("nan"))))
            check("canon_rejects_pipe", _raises_value_error(lambda: _fmt("a|b")))
            check("canon_np_bool", _fmt(np.bool_(True)) == "1")

            # 5. Heyting y regla de veredicto final
            check("heyting_veto_absorbs",
                  HeytingTruthValue.VETOED.meet(HeytingTruthValue.COHERENT) is HeytingTruthValue.VETOED)
            check("final_verdict_rules",
                  _final_verdict(HeytingTruthValue.COHERENT, False) is HeytingTruthValue.VETOED
                  and _final_verdict(HeytingTruthValue.DEGRADED, True) is HeytingTruthValue.DEGRADED)

            # 6. Fase 1 real: admisibilidad derivada
            rho_ok = np.eye(2, dtype=np.complex128) / 2.0
            rho_trace = np.eye(2, dtype=np.complex128) / 4.0
            rho_nonherm = np.array([[0.5, 0.1j], [0.0, 0.5]], dtype=np.complex128)
            p1_ok = self._observer.execute_phase1(rho_ok)
            p1_trace = self._observer.execute_phase1(rho_trace)
            p1_bad = self._observer.execute_phase1(rho_nonherm)
            check("p1_admissible_valid", p1_ok.is_admissible)
            check("p1_trace_rejected", not p1_trace.is_admissible)
            check("p1_nonhermitian_rejected", not p1_bad.is_admissible)

            # 7. Φ₁₂: contenido, re-derivación semántica re-sellada y dominio cruzado
            check("phi12_accepts_valid", _ok(lambda: _verify_p1(p1_ok, key)))
            check("phi12_rejects_content_tamper",
                  _raises_value_error(lambda: _verify_p1(replace(p1_ok, tr_rho=0.5), key)))
            forged = _resealed(p1_ok, _P1_CHAIN_ORDER, _HMAC_DOMAIN_P1, key,
                               is_admissible=not p1_ok.is_admissible)
            check("phi12_rejects_resealed_inconsistency",
                  _raises_value_error(lambda: _verify_p1(forged, key)))  # type: ignore[arg-type]
            cross = replace(p1_ok, hmac_signature=_sign(key, _HMAC_DOMAIN_P2, p1_ok.chain_hash))
            check("phi12_rejects_cross_domain", _raises_value_error(lambda: _verify_p1(cross, key)))

            # 8. Φ₂₃ con el motor real (ciclo rápido, sin Lyapunov)
            fast = ToonWeaverAuditPolicy(poincare_periods=0)
            mat = np.array([[0.01, 0.01], [0.0, 0.01]])
            exps = np.array([0.15, 0.30, 0.45])
            p2 = self._orientor.execute_phase2(p1_ok, mat, exps, fast)
            check("engine_selftest_reported", p2.engine_selftest_passed)
            check("phi23_accepts_valid", _ok(lambda: self.continue_to_phase3_gate(p1_ok, p2)))
            check("phi23_rejects_foreign_p1",
                  _raises_value_error(lambda: self.continue_to_phase3_gate(p1_bad, p2)))
            check("phi23_rejects_content_tamper", _raises_value_error(
                lambda: self.continue_to_phase3_gate(
                    p1_ok, replace(p2, p1_admissible=not p2.p1_admissible))))
            unsafe = _resealed(p2, _P2_CHAIN_ORDER, _HMAC_DOMAIN_P2, key, banach_safety_lip=1.0)
            check("phi23_rejects_unsafe_lip", _raises_value_error(
                lambda: self.continue_to_phase3_gate(p1_ok, unsafe)))  # type: ignore[arg-type]

            # 9. Estado soberano: vinculación, firma, métricas y veredicto
            state = self._decider.execute_phase3(p2)
            check("state_binding_valid", self.verify_state_binding(state, p1_ok, p2))
            check("state_binding_rejects_foreign_p1", not self.verify_state_binding(state, p1_bad, p2))
            check("state_rejects_metrics_tamper", not self.verify_state_signature(replace(
                state,
                visceral_metrics=replace(state.visceral_metrics, capacity_jerk_status="JERK_FALSO"),
            )))
            other = (HeytingTruthValue.COHERENT if state.truth_value == HeytingTruthValue.VETOED
                     else HeytingTruthValue.VETOED)
            check("state_rejects_verdict_tamper",
                  not self.verify_state_signature(replace(state, truth_value=other)))
            check("state_rejects_leaf_tamper", not self.verify_state_signature(replace(
                state, merkle_leaves=state.merkle_leaves[:-1] + ("alterada",))))
        except Exception:
            logger.exception("Autoprueba del agente: excepción (fail-closed).")
            failures.append("excepcion")

        if failures:
            logger.error("Autoprueba del agente fallida: %s", ", ".join(failures))
            return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"=== TEST AUTÓNOMO: BoolNovikovToonCartridgeWeaverAgent {_AGENT_VERSION} ===")
    if not os.environ.get(_HMAC_KEY_ENV):
        try:
            BoolNovikovToonCartridgeWeaverAgent()
        except ValueError as exc:
            print(f"Fail-closed sin clave (esperado): {exc}")

    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    matrix_a = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.9]])
    matrix_b = np.array([[0.01, 0.01], [0.0, 0.01]])
    exps_test = np.array([0.15, 0.30, 0.45])

    agent = BoolNovikovToonCartridgeWeaverAgent(secret_key=b"demo-key-0123456789abcdef")
    policy = ToonWeaverAuditPolicy(capacity_series=(1.0, 2.0, 4.0, 8.0))

    for label, mat in (("A (torsión)", matrix_a), ("B (unimodular)", matrix_b)):
        st = agent.run_sovereign_governance_cycle(rho_test, mat, exps_test, policy)
        print(f"\n=== Escenario {label}: {st.truth_value.name} ({hex(st.truth_value.value)}) ===")
        print(f"  Veredicto motor / ρ admisible : {st.engine_truth_value.name} / {st.p1_admissible}")
        print(f"  Crowbar (SIMULADO)            : {st.is_esp32_crowbar_triggered} @ {st.crowbar_actuation_latency_ns:.2f} ns")
        print(f"  Raíz Merkle / hojas           : {st.merkle_root_hash[:16]}... / {len(st.merkle_leaves)}")
        print(f"  Firma verificada              : {agent.verify_state_signature(st)}")
        print(f"  Compresión KV / Tokens        : {st.visceral_metrics.kv_cache_compression_pct:.1f}% / {st.visceral_metrics.token_count}")
        print(f"  Jerk                          : {st.visceral_metrics.capacity_jerk_status}")

    st_pos = agent.run_sovereign_governance_cycle(
        rho_test, matrix_b, exps_test, policy, positron_hmac_valid=True
    )
    report = agent.last_phase2_certificate.spectrum_report  # type: ignore[union-attr]
    kev = getattr(report, "dirac_annihilation_kev", None)  # nombre del campo a confirmar en el motor
    kev_txt = "no expuesto por el informe (ver contrato del motor)" if kev is None else f"{kev} keV"
    print(f"\nB con positrón: veredicto {st_pos.truth_value.name} | Dirac: {kev_txt}")
    print("=== TEST COMPLETADO ===")