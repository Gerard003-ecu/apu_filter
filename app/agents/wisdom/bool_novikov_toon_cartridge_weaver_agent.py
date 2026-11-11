# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovToonCartridgeWeaverAgent v14.0.0-PhD                                           ║
║ Ubicación: app/agents/wisdom/bool_novikov_toon_cartridge_weaver_agent.py                             ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-AGENT-POINCARE-RSI3-PhD               ║
║ Gobernanza: OODA Trifásica (①Observe ➔ ②Orient ➔ ③Decide&Act) con Compuertas Φ₁₂ y Φ₂₃              ║
║            sobre el motor BoolNovikovToonCartridgeWeaverEngine v14.0 (56 tokens / −86.4% KV-Cache).  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  F1 ToonWeaverPhase1Observer
     └ emit_phase1_certificate        ÚLTIMO de F1 → Φ₁₂ → execute_phase2 (PRIMERO de F2).
  F2 ToonWeaverPhase2Orientor
     └ emit_phase2_certificate        ÚLTIMO de F2 → Φ₂₃ → execute_phase3 (PRIMERO de F3).
  F3 ToonWeaverPhase3Decider
     └ emit_sovereign_state           ÚLTIMO de F3 (salida soberana firmada).

CAMBIOS RIGUROSOS v13.0 → v14.0 (granulares):
  · API: ToonWeaverAuditPolicy (frozen, validada) reenvía TODOS los parámetros del motor v14 por
    palabra clave. Corrige la llamada posicional ilegal y los campos inexistentes del informe.
  · Φ₁₂/Φ₂₃: las compuertas RE-DERIVAN el hash de cadena desde los campos del certificado y la
    firma HMAC (vinculación contenido→hash→firma). Alterar cualquier campo se detecta (fail-closed).
  · F1: certifica Hermiticidad ‖ρ−ρ†‖, Tr(ρ)=1, PSD, pureza Tr(ρ²)≤1 y entropía de von Neumann.
    El espectro se calcula sobre la parte Hermítica (eigvalsh sobre no-Hermítica es inválido).
    Un solo timestamp por certificado; reloj inyectable para reproducibilidad.
  · F2: vincula P1 (hash + admisibilidad), input_digest, pasaporte, esquema, veredicto, autoprueba,
    cota de Banach y cadena RSI³ del motor. Compatibilidad v13.1/v14 vía getattr declarado.
  · F3: veredicto final = motor.meet(⊤ si ρ admisible else ⊥) en la cadena de Heyting Ω₃.
    Raíz Merkle BINARIA real (RFC-6962: 0x00 hojas, 0x01 nodos). Firma HMAC-P3 del estado.
  · Crowbar: presupuesto de latencia DETERMINISTA y SIMULADO (GPIO+SIMD+drive ≤ 392.15 ns).
    NO escribe en hardware; no hay HAL. La conmutación del BT151 es externa y no se modela.
  · Φ_sem: WACC/liquidez son OBJETIVOS de política declarados (no medidos), con procedencia.
    Jerk = tercera diferencia MEDIDA (Optional); no se postula signo ni crecimiento super-exponencial.
  · Clave HMAC: entorno o explícita; desarrollo solo con advertencia. Serialización canónica fail-closed.

LÍMITES DECLARADOS:
  · El agente NO actúa hardware; el interlock es una simulación de presupuesto de control.
  · La admisibilidad de ρ es una comprobación de invariantes físicos, no un teorema independiente.
  · Los objetivos WACC/liquidez son política del autor, no resultados derivados del espectro.
  · Una tercera diferencia finita es un indicador local; no prueba crecimiento super-exponencial.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple

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

# ── Versiones y dominios HMAC ──────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-BOOL-NOVIKOV-TOON-CARTRIDGE-WEAVER-AGENT-POINCARE-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
_AGENT_SCHEMA: Final[str] = "agent-cert-v14.0"
_HMAC_KEY_ENV: Final[str] = "APU_TOON_WEAVER_AGENT_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_AGENT_DEV_ONLY_TOON_WEAVER_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"TOON-WEAVER-AGENT-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"TOON-WEAVER-AGENT-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"TOON-WEAVER-AGENT-P3-SOVEREIGN-v14.0|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"

# ── Tolerancias físicas de Fase 1 ──────────────────────────────────────────────
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

# ── Objetivos de política del Funtor Semántico Visceral (declarados, no medidos) ─
_WACC_PROTECTION_TARGET_PCT: Final[float] = 15.0
_NET_LIQUIDITY_TARGET_PCT: Final[float] = 11.5
_POLICY_PROVENANCE: Final[str] = "POLITICA_DECLARADA_v14.0 (objetivos, no mediciones)"


# ══════════════════════════════════════════════════════════════════════════════
# Primitivas canónicas y criptográficas
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_secret_key(explicit: Optional[bytes]) -> bytes:
    """Clave explícita, o entorno, o desarrollo con advertencia. Mínimo 16 bytes."""
    if explicit is not None:
        key = bytes(explicit)
    else:
        env_val = os.environ.get(_HMAC_KEY_ENV, "")
        if env_val:
            key = env_val.encode("utf-8")
        else:
            logger.warning("Sin %s: clave de DESARROLLO (no apta para producción).", _HMAC_KEY_ENV)
            key = _DEV_SECRET_KEY
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(f"Clave HMAC demasiado corta: {len(key)} < {_MIN_KEY_BYTES} bytes.")
    return key


def _fmt(value: object) -> str:
    """Serialización canónica de un campo. Rechaza no-finitos y el separador '|'."""
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


def _chain_hash(fields: Tuple[object, ...]) -> str:
    """SHA-256 de la serialización canónica de los campos (vinculación contenido→hash)."""
    payload = "|".join(_fmt(f) for f in fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sign(secret_key: bytes, domain: bytes, chain_hash: str) -> str:
    return hmac.new(secret_key, domain + chain_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def _merkle_root(leaves: Sequence[str]) -> str:
    """Raíz Merkle binaria estilo RFC-6962: H(0x00‖hoja), H(0x01‖izq‖der).
    Nodo impar se promueve sin duplicar. Vacío → SHA-256(b"")."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(_MERKLE_LEAF + s.encode("utf-8")).digest() for s in leaves]
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.sha256(_MERKLE_NODE + level[i] + level[i + 1]).digest())
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


# ══════════════════════════════════════════════════════════════════════════════
# Contratos: certificados, política, métricas y estado soberano
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ToonWeaverPhase1Certificate:
    """Certificado P1 (Fase 1 Observe): invariantes físicos de la matriz de densidad."""

    timestamp: float
    tr_rho: float
    is_hermitian: bool
    hermiticity_residual: float
    is_psd: bool
    min_eigenvalue: float
    max_eigenvalue: float
    purity: float
    von_neumann_entropy: float
    is_admissible: bool
    chain_hash: str
    hmac_signature: str
    schema_version: str


@dataclass(frozen=True)
class ToonWeaverPhase2Certificate:
    """Certificado P2 (Fase 2 Orient): informe del motor vinculado a P1."""

    timestamp: float
    spectrum_report: ToonCartridgeWeaverSpectrumReport
    p1_chain_hash: str
    p1_admissible: bool
    chain_hash: str
    hmac_signature: str
    schema_version: str


@dataclass(frozen=True)
class ToonWeaverAuditPolicy:
    """Política de auditoría: reenvío validado de TODOS los parámetros del motor v14."""

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
        if isinstance(self.raw_token_count, bool) or not isinstance(self.raw_token_count, int):
            raise ValueError("raw_token_count debe ser entero.")
        if self.raw_token_count < 56:
            raise ValueError("raw_token_count ≥ 56.")
        if not (0 <= self.rsi_max_depth <= 3):
            raise ValueError("rsi_max_depth ∈ [0,3].")
        if not (0.0 <= self.initial_lip < 1.0):
            raise ValueError("initial_lip ∈ [0,1).")
        if not (0 <= self.poincare_periods <= 20000):
            raise ValueError("poincare_periods ∈ [0,20000].")
        if not (8 <= self.poincare_steps_per_period <= 4096):
            raise ValueError("poincare_steps_per_period ∈ [8,4096].")
        for name in ("ball_r", "cylinder_R", "melnikov_delta", "melnikov_gamma",
                     "melnikov_omega", "pendulum_amplitude"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} debe ser finito.")
        if self.positron_hmac_valid is not None and not isinstance(self.positron_hmac_valid, bool):
            raise TypeError("positron_hmac_valid debe ser bool o None.")
        if self.capacity_series is not None:
            if len(self.capacity_series) < 4 or not all(math.isfinite(float(x)) for x in self.capacity_series):
                raise ValueError("capacity_series requiere ≥4 muestras finitas.")

    def to_engine_kwargs(self) -> Dict[str, object]:
        """Diccionario de palabras clave para el motor (nunca posicional)."""
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
    """Métricas del Funtor Semántico Visceral Φ_sem (objetivos declarados + medidas)."""

    wacc_protection_target_pct: float
    net_liquidity_release_target_pct: float
    kv_cache_compression_pct: float
    token_count: int
    capacity_third_difference: Optional[float]
    capacity_jerk_status: str
    policy_provenance: str


@dataclass(frozen=True)
class ToonWeaverSovereignState:
    """Estado soberano final (Fase 3 Decide & Act), firmado sobre raíz Merkle."""

    timestamp: float
    truth_value: HeytingTruthValue
    engine_truth_value: HeytingTruthValue
    p1_admissible: bool
    is_esp32_crowbar_triggered: bool
    crowbar_actuation_latency_ns: float
    crowbar_latency_is_simulated: bool
    merkle_root_hash: str
    state_signature: str
    visceral_metrics: ToonWeaverVisceralMetrics
    schema_version: str


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — OBSERVE: ingestión e invariantes físicos de ρ (MAC)
# ══════════════════════════════════════════════════════════════════════════════
class ToonWeaverPhase1Observer:
    """Fase 1 OODA. Certifica Hermiticidad, traza, PSD, pureza y entropía de ρ."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128], trace_tolerance: float = _TRACE_TOL
    ) -> ToonWeaverPhase1Certificate:
        """Inspeciona ρ y emite P1. El espectro se calcula sobre la parte Hermítica."""
        rho = np.asarray(rho_mac, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 1:
            raise ValueError("rho_mac debe ser cuadrada 2D no vacía.")
        if not np.all(np.isfinite(rho)):
            raise ValueError("rho_mac contiene valores no finitos.")
        scale = max(1.0, float(np.max(np.abs(rho))))
        herm_residual = float(np.max(np.abs(rho - rho.conj().T)))
        is_hermitian = bool(herm_residual <= _HERMITICITY_TOL * scale)
        rho_h = 0.5 * (rho + rho.conj().T)
        evals = np.linalg.eigvalsh(rho_h)
        tr_rho = float(np.real(np.trace(rho_h)))
        min_eig = float(evals[0])
        max_eig = float(evals[-1])
        is_psd = bool(min_eig >= -_PSD_TOL * scale)
        purity = float(np.sum(evals ** 2))
        pos = evals[evals > 0.0]
        entropy = float(-np.sum(pos * np.log(pos)))
        trace_ok = bool(abs(tr_rho - 1.0) <= trace_tolerance)
        purity_ok = bool(purity <= 1.0 + _PURITY_TOL)
        is_admissible = bool(is_hermitian and is_psd and trace_ok and purity_ok)
        return self.emit_phase1_certificate(
            timestamp=float(self._clock()),
            tr_rho=tr_rho,
            is_hermitian=is_hermitian,
            hermiticity_residual=herm_residual,
            is_psd=is_psd,
            min_eigenvalue=min_eig,
            max_eigenvalue=max_eig,
            purity=purity,
            von_neumann_entropy=entropy,
            is_admissible=is_admissible,
        )

    @staticmethod
    def _p1_chain_fields(cert: ToonWeaverPhase1Certificate) -> Tuple[object, ...]:
        """Orden canónico de campos que el hash vincula (idéntico en emisión y compuerta)."""
        return (
            cert.timestamp, cert.tr_rho, cert.is_hermitian, cert.hermiticity_residual,
            cert.is_psd, cert.min_eigenvalue, cert.max_eigenvalue, cert.purity,
            cert.von_neumann_entropy, cert.is_admissible, cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 1 → Φ₁₂ → Fase 2 ─────────────────────────────────
    def emit_phase1_certificate(self, **fields: object) -> ToonWeaverPhase1Certificate:
        """ÚLTIMO método de F1. Firma P1 y lo entrega a la Compuerta Homomórfica Φ₁₂."""
        base = dict(fields)
        base["schema_version"] = _AGENT_SCHEMA
        provisional = ToonWeaverPhase1Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p1_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, chain)
        return ToonWeaverPhase1Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — ORIENT: juzgamiento del motor y vinculación espectral
# ══════════════════════════════════════════════════════════════════════════════
class ToonWeaverPhase2Orientor:
    """Fase 2 OODA. Ejecuta el motor v14 y vincula P1 + pasaporte + cadena RSI³."""

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovToonCartridgeWeaverEngine,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = secret_key
        self._engine = engine
        self._clock = clock

    def execute_phase2(
        self,
        p1_cert: ToonWeaverPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        policy: ToonWeaverAuditPolicy,
    ) -> ToonWeaverPhase2Certificate:
        """PRIMER método de F2. Consume P1 (vía Φ₁₂) y la política; invoca el motor por palabra."""
        if not isinstance(p1_cert, ToonWeaverPhase1Certificate):
            raise TypeError("Fase 2 exige un ToonWeaverPhase1Certificate.")
        report = self._engine.execute_toon_cartridge_weaver_audit(
            apu_cost_matrix, exponents, **policy.to_engine_kwargs()
        )
        return self.emit_phase2_certificate(p1_cert, report, float(self._clock()))

    @staticmethod
    def _p2_chain_fields(cert: ToonWeaverPhase2Certificate) -> Tuple[object, ...]:
        r = cert.spectrum_report
        return (
            cert.timestamp, cert.p1_chain_hash, cert.p1_admissible,
            r.input_digest, r.passport_hash, r.schema_version,
            int(r.truth_value), bool(r.metamorphic_selftest_passed),
            float(r.banach_contraction_lip),
            getattr(r, "rsi_certificate_chain", ""),
            cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 2 → Φ₂₃ → Fase 3 ─────────────────────────────────
    def emit_phase2_certificate(
        self,
        p1_cert: ToonWeaverPhase1Certificate,
        report: ToonCartridgeWeaverSpectrumReport,
        timestamp: float,
    ) -> ToonWeaverPhase2Certificate:
        """ÚLTIMO método de F2. Firma P2 vinculando P1 y el informe del motor."""
        base = {
            "timestamp": timestamp,
            "spectrum_report": report,
            "p1_chain_hash": p1_cert.chain_hash,
            "p1_admissible": p1_cert.is_admissible,
            "schema_version": _AGENT_SCHEMA,
        }
        provisional = ToonWeaverPhase2Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p2_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, chain)
        return ToonWeaverPhase2Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — DECIDE & ACT: Heyting, interlock simulado, Φ_sem y raíz Merkle
# ══════════════════════════════════════════════════════════════════════════════
class ToonWeaverPhase3Decider:
    """Fase 3 OODA. Compone el veredicto, simula el interlock y firma el estado."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase3(self, p2_cert: ToonWeaverPhase2Certificate) -> ToonWeaverSovereignState:
        """PRIMER método de F3. Consume P2 (vía Φ₂₃) y adjudica sobre Heyting Ω₃."""
        if not isinstance(p2_cert, ToonWeaverPhase2Certificate):
            raise TypeError("Fase 3 exige un ToonWeaverPhase2Certificate.")
        report = p2_cert.spectrum_report
        engine_truth = report.truth_value
        p1_verdict = (
            HeytingTruthValue.COHERENT if p2_cert.p1_admissible else HeytingTruthValue.VETOED
        )
        final_truth = engine_truth.meet(p1_verdict)
        crowbar_triggered = bool(final_truth == HeytingTruthValue.VETOED)
        crowbar_latency = _CROWBAR_BUDGET_NS if crowbar_triggered else 0.0
        if crowbar_triggered:
            logger.warning(
                "[INTERLOCK SIMULADO] Crowbar GPIO14/BT151 en %.2f ns (presupuesto, NO hardware).",
                crowbar_latency,
            )
        metrics = self._visceral_metrics(report)
        return self.emit_sovereign_state(
            timestamp=float(self._clock()),
            final_truth=final_truth,
            engine_truth=engine_truth,
            p1_admissible=p2_cert.p1_admissible,
            crowbar_triggered=crowbar_triggered,
            crowbar_latency=crowbar_latency,
            p2_cert=p2_cert,
            metrics=metrics,
        )

    @staticmethod
    def _visceral_metrics(report: ToonCartridgeWeaverSpectrumReport) -> ToonWeaverVisceralMetrics:
        """Φ_sem: objetivos de política declarados + compresión/tokens/jerk MEDIDOS."""
        jerk = getattr(report, "capacity_third_difference", None)
        if jerk is None:
            jerk_status = "JERK_NO_SERIES (sin serie suministrada)"
        else:
            sign = "+" if jerk > 0.0 else ("-" if jerk < 0.0 else "0")
            jerk_status = (
                f"JERK_d3C_dt3={jerk:.6g}_sign={sign} "
                "(indicador local; no prueba crecimiento super-exponencial)"
            )
        return ToonWeaverVisceralMetrics(
            wacc_protection_target_pct=_WACC_PROTECTION_TARGET_PCT,
            net_liquidity_release_target_pct=_NET_LIQUIDITY_TARGET_PCT,
            kv_cache_compression_pct=float(report.kv_cache_compression_pct),
            token_count=int(report.compressed_token_count),
            capacity_third_difference=(None if jerk is None else float(jerk)),
            capacity_jerk_status=jerk_status,
            policy_provenance=_POLICY_PROVENANCE,
        )

    # ── ÚLTIMO método de Fase 3: estado soberano firmado ───────────────────────
    def emit_sovereign_state(
        self,
        *,
        timestamp: float,
        final_truth: HeytingTruthValue,
        engine_truth: HeytingTruthValue,
        p1_admissible: bool,
        crowbar_triggered: bool,
        crowbar_latency: float,
        p2_cert: ToonWeaverPhase2Certificate,
        metrics: ToonWeaverVisceralMetrics,
    ) -> ToonWeaverSovereignState:
        """ÚLTIMO método de F3. Raíz Merkle binaria + firma HMAC-P3 del estado."""
        report = p2_cert.spectrum_report
        leaves = [
            p2_cert.p1_chain_hash,
            p2_cert.chain_hash,
            report.passport_hash,
            report.input_digest,
            f"verdict:{int(final_truth)}",
            f"crowbar:{int(crowbar_triggered)}:{crowbar_latency:.6f}",
            _chain_hash((
                metrics.wacc_protection_target_pct,
                metrics.net_liquidity_release_target_pct,
                metrics.kv_cache_compression_pct,
                metrics.token_count,
                metrics.capacity_third_difference,
                metrics.capacity_jerk_status,
            )),
            getattr(report, "rsi_certificate_chain", ""),
        ]
        merkle_root = _merkle_root(leaves)
        core = (
            timestamp, int(final_truth), int(engine_truth), p1_admissible,
            crowbar_triggered, crowbar_latency, merkle_root, _AGENT_SCHEMA,
        )
        state_sig = _sign(
            self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core)
        )
        return ToonWeaverSovereignState(
            timestamp=timestamp,
            truth_value=final_truth,
            engine_truth_value=engine_truth,
            p1_admissible=p1_admissible,
            is_esp32_crowbar_triggered=crowbar_triggered,
            crowbar_actuation_latency_ns=crowbar_latency,
            crowbar_latency_is_simulated=True,
            merkle_root_hash=merkle_root,
            state_signature=state_sig,
            visceral_metrics=metrics,
            schema_version=_AGENT_SCHEMA,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orquestador soberano: ciclo OODA con compuertas homomórficas Φ₁₂ y Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovToonCartridgeWeaverAgent:
    """Soberano Agéntico Tejedor TOON (56 tokens). Composición de las tres fases OODA."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovToonCartridgeWeaverEngine] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = _resolve_secret_key(secret_key)
        self._engine = engine or BoolNovikovToonCartridgeWeaverEngine(secret_key=self._secret_key)
        self._clock = clock
        self._observer = ToonWeaverPhase1Observer(self._secret_key, clock)
        self._orientor = ToonWeaverPhase2Orientor(self._secret_key, self._engine, clock)
        self._decider = ToonWeaverPhase3Decider(self._secret_key, clock)
        if not self._metamorphic_selftest():
            raise RuntimeError("Autoprueba metamórfica del agente fallida (fail-closed).")
        logger.info("%s v%s iniciado correctamente.", self.__class__.__name__, _AGENT_VERSION)

    # ── Compuerta Homomórfica Φ₁₂ (re-deriva contenido→hash→firma) ─────────────
    def continue_to_phase2_gate(self, p1: ToonWeaverPhase1Certificate) -> None:
        """Φ₁₂: autentica P1 re-derivando hash y firma. ADMITE anomalías para auditoría
        (la admisibilidad física se compone en Fase 3, no se veta aquí)."""
        if p1.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₁₂: esquema de certificado P1 no reconocido.")
        expected_chain = _chain_hash(ToonWeaverPhase1Observer._p1_chain_fields(p1))
        if not hmac.compare_digest(expected_chain, p1.chain_hash):
            raise ValueError("Φ₁₂: el hash de cadena P1 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, expected_chain)
        if not hmac.compare_digest(expected_sig, p1.hmac_signature):
            raise ValueError("Φ₁₂: falla de autenticidad HMAC en P1.")

    # ── Compuerta Homomórfica Φ₂₃ (re-deriva + estabilidad de Banach) ──────────
    def continue_to_phase3_gate(self, p2: ToonWeaverPhase2Certificate) -> None:
        """Φ₂₃: autentica P2 y exige contracción de Banach Lip_cert < 1.0."""
        if p2.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₂₃: esquema de certificado P2 no reconocido.")
        expected_chain = _chain_hash(ToonWeaverPhase2Orientor._p2_chain_fields(p2))
        if not hmac.compare_digest(expected_chain, p2.chain_hash):
            raise ValueError("Φ₂₃: el hash de cadena P2 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, expected_chain)
        if not hmac.compare_digest(expected_sig, p2.hmac_signature):
            raise ValueError("Φ₂₃: falla de autenticidad HMAC en P2.")
        if not (p2.spectrum_report.banach_contraction_lip < 1.0):
            raise ValueError("Φ₂₃: violación de estabilidad (Lip_cert ≥ 1.0).")

    def verify_state_signature(self, state: ToonWeaverSovereignState) -> bool:
        """Verificación independiente de la firma del estado soberano."""
        core = (
            state.timestamp, int(state.truth_value), int(state.engine_truth_value),
            state.p1_admissible, state.is_esp32_crowbar_triggered,
            state.crowbar_actuation_latency_ns, state.merkle_root_hash, state.schema_version,
        )
        expected = _sign(self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core))
        return hmac.compare_digest(expected, state.state_signature)

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
        """Ejecuta Observe→Φ₁₂→Orient→Φ₂₃→Decide&Act sobre el fibrado dual Boole-Novikov."""
        eff_policy = policy if policy is not None else ToonWeaverAuditPolicy()
        if positron_hmac_valid is not None:
            eff_policy = replace(eff_policy, positron_hmac_valid=positron_hmac_valid)
        p1 = self._observer.execute_phase1(rho_mac)
        self.continue_to_phase2_gate(p1)                       # Φ₁₂
        p2 = self._orientor.execute_phase2(p1, apu_cost_matrix, exponents, eff_policy)
        self.continue_to_phase3_gate(p2)                       # Φ₂₃
        state = self._decider.execute_phase3(p2)
        logger.info("Ciclo OODA TOON Weaver completado. Veredicto: %s", state.truth_value.name)
        return state

    # ── Autoprueba metamórfica del agente (nivel meta-meta de gobernanza) ──────
    def _metamorphic_selftest(self) -> bool:
        """Verifica vinculación de compuertas, Merkle, composición Heyting y presupuesto."""
        try:
            if _CROWBAR_BUDGET_NS > _ESP32_CROWBAR_MAX_LATENCY_NS:
                return False
            # Merkle: casos vacío, hoja única y par (RFC-6962).
            if _merkle_root([]) != hashlib.sha256(b"").hexdigest():
                return False
            a, b = "aa", "bb"
            la = hashlib.sha256(_MERKLE_LEAF + a.encode()).digest()
            lb = hashlib.sha256(_MERKLE_LEAF + b.encode()).digest()
            if _merkle_root([a, b]) != hashlib.sha256(_MERKLE_NODE + la + lb).hexdigest():
                return False
            if _merkle_root([a]) != la.hex():
                return False
            # Composición Heyting: ⊥ absorbe en meet.
            if HeytingTruthValue.VETOED.meet(HeytingTruthValue.COHERENT) is not HeytingTruthValue.VETOED:
                return False
            # Canonicalización rechaza no-finitos.
            try:
                _fmt(float("nan"))
                return False
            except ValueError:
                pass
            # Ciclo rápido (sin Lyapunov) + detección de tamper en Φ₁₂ y Φ₂₃.
            fast = ToonWeaverAuditPolicy(poincare_periods=0)
            rho = np.eye(2, dtype=np.complex128) / 2.0
            mat = np.array([[0.01, 0.01], [0.0, 0.01]])
            exps = np.array([0.15, 0.30, 0.45])
            p1 = self._observer.execute_phase1(rho)
            self.continue_to_phase2_gate(p1)
            tampered_p1 = replace(p1, tr_rho=0.5)   # contenido alterado, hash intacto
            try:
                self.continue_to_phase2_gate(tampered_p1)
                return False
            except ValueError:
                pass
            p2 = self._orientor.execute_phase2(p1, mat, exps, fast)
            self.continue_to_phase3_gate(p2)
            tampered_p2 = replace(p2, p1_admissible=not p2.p1_admissible)
            try:
                self.continue_to_phase3_gate(tampered_p2)
                return False
            except ValueError:
                pass
            state = self._decider.execute_phase3(p2)
            if not self.verify_state_signature(state):
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica del agente: excepción (fail-closed).")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovToonCartridgeWeaverAgent v14.0 ===")
    dim = 4
    rho_test = np.eye(dim, dtype=np.complex128) / float(dim)
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
        print(f"  Raíz Merkle / Firma           : {st.merkle_root_hash[:16]}... / {st.state_signature[:16]}...")
        print(f"  Firma verificada              : {agent.verify_state_signature(st)}")
        print(f"  Compresión KV / Tokens        : {st.visceral_metrics.kv_cache_compression_pct:.1f}% / {st.visceral_metrics.token_count}")
        print(f"  Jerk                          : {st.visceral_metrics.capacity_jerk_status}")
    st_pos = agent.run_sovereign_governance_cycle(
        rho_test, matrix_b, exps_test, positron_hmac_valid=True
    )
    print(f"\nB con positrón: Dirac = {st_pos.visceral_metrics.kv_cache_compression_pct:.1f}% KV (ver informe del motor para keV)")
    print("=== TEST COMPLETADO ===")