# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPhantomCollusionMitigatorProsecutorAgent v14.0.0-PhD                           ║
║ Ubicación: app/wisdom/bool_novikov_phantom_collusion_mitigator_prosecutor_agent.py                  ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-PHANTOM-COLLUSION-PROSECUTOR-AGENT-POINCARE-RSI3-PhD        ║
║ Gobernanza: OODA Trifásica Fiscal (①Observe → ②Orient → ③Decide) con Compuertas Φ₁₂ y Φ₂₃            ║
║            sobre el Motor Fiscal v14.0. Actuación SIMULADA: sin HAL, sin escritura a hardware.       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  F1 PhantomCollusionProsecutorPhase1Observer
     └ emit_phase1_certificate        ÚLTIMO de F1 → Φ₁₂ → execute_phase2 (PRIMERO de F2).
  F2 PhantomCollusionProsecutorPhase2Orientor
     └ emit_phase2_certificate        ÚLTIMO de F2 → Φ₂₃ → execute_phase3 (PRIMERO de F3).
  F3 PhantomCollusionProsecutorPhase3Decider
     └ emit_sovereign_state           ÚLTIMO de F3 (estado soberano firmado).

CAMBIOS RIGUROSOS v13.0 → v14.0 (granulares):
  · Φ₂₃ deja de ser código muerto: la v13 construía `payload` y devolvía True sin verificar nada.
    Ahora RE-DERIVA el hash de cadena desde los campos canónicos y la firma HMAC (contenido→hash→
    firma) y exige el CONTRATO DE INTEGRIDAD DE ACTUACIÓN:
        crowbar_triggered ⇔ prosecution_warrant
        hardware_actuated = False, crowbar_latency_is_simulated = True
        latencia y presupuesto ≤ 392.15 ns; latencia > 0 ⇔ disparo
        security_veto ⇒ prosecution_warrant
        annihilation_energy_kev ∈ {0, 1021.9979}
  · Φ₁₂/Φ₂₃ usan hmac.compare_digest (la v13 usaba `!=`: canal lateral temporal).
  · Canonicalización .17g fail-closed (la v13 truncaba a .6f: colisiones de contenido).
  · F1: Hermiticidad ‖ρ−ρ†‖, Tr(ρ)=1 (tol 1e-10), PSD, pureza Tr(ρ²)≤1 y entropía de von Neumann.
    Espectro sobre la parte Hermítica. Un solo timestamp; reloj inyectable.
  · F2: PhantomProsecutorAuditPolicy (frozen, validada) reenvía TODOS los parámetros del motor v14.
    Vincula P1 (hash+admisibilidad), input_digest, pasaporte, esquema, veredicto, autoprueba, orden,
    Dirac/crowbar, asociadores y cadena RSI³.
  · F3: veredicto final = motor.meet(⊤ si ρ admisible else ⊥). Raíz Merkle BINARIA RFC-6962 y
    firma HMAC-P3 del estado. Estado de estado factual (sin retórica): la causa del degradado se
    deriva de las banderas certificadas, no se presupone.
  · Φ_sem: se ELIMINA kv_cache_compression_pct=86.4 (contaminación cruzada: el motor fiscal no
    comprime KV). Métricas genuinas: orden, sospecha, razones de asociadores, Dirac, torsión,
    Melnikov, profundidad RSI, jerk medido. WACC/liquidez son OBJETIVOS de política declarados.
  · Epistemia: la sospecha por asociadores es EVIDENCIA (degrada); solo la orden CERTIFICADA
    (prosecution_warrant) habilita el crowbar simulado. El agente verifica esa separación.
  · Clave HMAC por entorno; autoprueba metamórfica en init con forja de actuación no autorizada.

LÍMITES DECLARADOS:
  · El agente NO actúa hardware: no hay HAL ni escritura a registros. El crowbar es simulación
    de presupuesto de control; la conmutación del BT151 es externa y no se modela.
  · P1 certifica ρ TAL COMO SE PRESENTA; la normalización interna del motor no anula la
    inadmisibilidad de una ρ con Tr≠1 (se compone como veto en F3).
  · Los objetivos WACC/liquidez son política del autor, no resultados derivados del espectro.
  · La sospecha de colusión por asociadores es un heurístico del motor, no una prueba.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Importación del motor fiscal (dos rutas de paquete) ────────────────────────
try:
    from bool_novikov_phantom_collusion_mitigator_prosecutor_engine import (
        BoolNovikovPhantomCollusionMitigatorProsecutorEngine,
        HeytingTruthValue,
        PhantomCollusionProsecutorReport,
    )
except ImportError:  # pragma: no cover - ruta de paquete alternativa
    try:
        from app.wisdom.bool_novikov_phantom_collusion_mitigator_prosecutor_engine import (
            BoolNovikovPhantomCollusionMitigatorProsecutorEngine,
            HeytingTruthValue,
            PhantomCollusionProsecutorReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_phantom_collusion_mitigator_prosecutor_engine."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovPhantomCollusionMitigatorProsecutorAgent")

# ── Versiones y dominios HMAC ──────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-BOOL-NOVIKOV-PHANTOM-COLLUSION-PROSECUTOR-AGENT-POINCARE-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
_AGENT_SCHEMA: Final[str] = "prosecutor-agent-cert-v14.0"
_HMAC_KEY_ENV: Final[str] = "APU_PHANTOM_PROSECUTOR_AGENT_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_PROSECUTOR_AGENT_DEV_ONLY_PHANTOM_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"COLLUSION-PROSECUTOR-AGENT-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"COLLUSION-PROSECUTOR-AGENT-P2-PROSECUTION-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"COLLUSION-PROSECUTOR-AGENT-P3-SOVEREIGN-v14.0|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"

# ── Tolerancias físicas de Fase 1 ──────────────────────────────────────────────
_TRACE_TOL: Final[float] = 1.0e-10
_HERMITICITY_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_PURITY_TOL: Final[float] = 1.0e-10

# ── Contrato de actuación (debe coincidir con el motor v14) ────────────────────
_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_DIRAC_TOTAL_KEV: Final[float] = 1021.9979
_DIRAC_ENERGY_TOL: Final[float] = 1.0e-6
_CR_RESIDUAL_TOL: Final[float] = 1.0e-3
_RW_DEGRADED_TOL: Final[float] = 0.5

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
    """SHA-256 de la serialización canónica (vinculación contenido→hash)."""
    payload = "|".join(_fmt(f) for f in fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sign(secret_key: bytes, domain: bytes, chain_hash: str) -> str:
    return hmac.new(secret_key, domain + chain_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def _merkle_root(leaves: Sequence[str]) -> str:
    """Raíz Merkle binaria RFC-6962: H(0x00‖hoja), H(0x01‖izq‖der). Impar se promueve."""
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


def _actuation_integrity_ok(report: PhantomCollusionProsecutorReport) -> bool:
    """Contrato de integridad de actuación del fiscal (todo certificado, nada heurístico).
    orden ⇔ disparo; sin hardware; latencia simulada acotada; veto de seguridad ⇒ orden;
    energía de Dirac ∈ {0, 2·m_e·c²}."""
    dc = report.dirac_crowbar_report
    energy_ok = (
        abs(dc.annihilation_energy_kev) <= _DIRAC_ENERGY_TOL
        or abs(dc.annihilation_energy_kev - _DIRAC_TOTAL_KEV) <= _DIRAC_ENERGY_TOL
    )
    return bool(
        dc.crowbar_triggered == report.prosecution_warrant
        and dc.hardware_actuated is False
        and dc.crowbar_latency_is_simulated is True
        and dc.latency_budget_ns <= _CROWBAR_MAX_LATENCY_NS
        and dc.crowbar_latency_ns <= _CROWBAR_MAX_LATENCY_NS
        and (dc.crowbar_latency_ns > 0.0) == dc.crowbar_triggered
        and ((not dc.security_veto) or report.prosecution_warrant)
        and energy_ok
    )


# ══════════════════════════════════════════════════════════════════════════════
# Contratos: certificados, política, métricas y estado soberano
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PhantomCollusionProsecutorPhase1Certificate:
    """Certificado P1 (Observe): invariantes físicos de la matriz de densidad MAC."""

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
class PhantomCollusionProsecutorPhase2Certificate:
    """Certificado P2 (Orient): informe fiscal del motor vinculado a P1."""

    timestamp: float
    prosecution_report: PhantomCollusionProsecutorReport
    p1_chain_hash: str
    p1_admissible: bool
    chain_hash: str
    hmac_signature: str
    schema_version: str


@dataclass(frozen=True)
class PhantomProsecutorAuditPolicy:
    """Política de auditoría: reenvío validado de los parámetros del motor fiscal v14."""

    positron_hmac_valid: Optional[bool] = None
    ball_r: float = 0.35
    cylinder_R: float = 0.50
    melnikov_delta: float = 0.01
    melnikov_gamma: float = 0.02
    melnikov_omega: float = 1.0
    pendulum_amplitude: float = 0.2
    poincare_periods: int = 256
    poincare_steps: int = 64
    rsi_max_depth: int = 3
    initial_lip: float = 0.95
    capacity_series: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if isinstance(self.rsi_max_depth, bool) or not isinstance(self.rsi_max_depth, int):
            raise ValueError("rsi_max_depth debe ser entero.")
        if not (0 <= self.rsi_max_depth <= 3):
            raise ValueError("rsi_max_depth ∈ [0,3].")
        if not (0.0 <= self.initial_lip < 1.0):
            raise ValueError("initial_lip ∈ [0,1).")
        if not (0 <= self.poincare_periods <= 20000):
            raise ValueError("poincare_periods ∈ [0,20000].")
        if not (8 <= self.poincare_steps <= 4096):
            raise ValueError("poincare_steps ∈ [8,4096].")
        for name in ("ball_r", "cylinder_R", "melnikov_delta", "melnikov_gamma",
                     "melnikov_omega", "pendulum_amplitude"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} debe ser finito.")
        if self.positron_hmac_valid is not None and not isinstance(self.positron_hmac_valid, bool):
            raise TypeError("positron_hmac_valid debe ser bool o None.")
        if self.capacity_series is not None:
            if len(self.capacity_series) < 4 or not all(
                math.isfinite(float(x)) for x in self.capacity_series
            ):
                raise ValueError("capacity_series requiere ≥4 muestras finitas.")

    def to_engine_kwargs(self) -> Dict[str, object]:
        """Diccionario de palabras clave para el motor (nunca posicional)."""
        return {
            "positron_hmac_valid": self.positron_hmac_valid,
            "ball_r": self.ball_r,
            "cylinder_R": self.cylinder_R,
            "melnikov_delta": self.melnikov_delta,
            "melnikov_gamma": self.melnikov_gamma,
            "melnikov_omega": self.melnikov_omega,
            "pendulum_amplitude": self.pendulum_amplitude,
            "poincare_periods": self.poincare_periods,
            "poincare_steps": self.poincare_steps,
            "rsi_max_depth": self.rsi_max_depth,
            "initial_lip": self.initial_lip,
            "capacity_series": self.capacity_series,
        }


@dataclass(frozen=True)
class PhantomCollusionProsecutorVisceralMetrics:
    """Φ_sem: objetivos de política declarados + medidas genuinas del fiscal.
    Nota: NO incluye compresión KV (el motor fiscal no comprime; eso es del weaver)."""

    wacc_protection_target_pct: float
    net_liquidity_release_target_pct: float
    prosecution_warrant: bool
    phantom_collusion_suspected: bool
    associator_ratio_max: float
    dirac_energy_kev: float
    dirac_event_authorized: bool
    smith_torsion_free: bool
    melnikov_chaos_detected: bool
    rsi_accepted_depth: int
    capacity_third_difference: Optional[float]
    phantom_collusion_status: str
    policy_provenance: str


@dataclass(frozen=True)
class PhantomCollusionProsecutorSovereignState:
    """Estado soberano final (Decide), firmado sobre raíz Merkle. Actuación SIMULADA."""

    timestamp: float
    truth_value: HeytingTruthValue
    engine_truth_value: HeytingTruthValue
    p1_admissible: bool
    prosecution_warrant: bool
    crowbar_triggered: bool
    crowbar_latency_ns: float
    crowbar_latency_is_simulated: bool
    hardware_actuated: bool                 # SIEMPRE False
    warrant_actuation_consistent: bool
    merkle_root_hash: str
    state_signature: str
    visceral_metrics: PhantomCollusionProsecutorVisceralMetrics
    schema_version: str


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — OBSERVE: ingestión e invariantes físicos de ρ (MAC)
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionProsecutorPhase1Observer:
    """Fase 1 OODA. Certifica Hermiticidad, traza, PSD, pureza y entropía de ρ."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128], trace_tolerance: float = _TRACE_TOL
    ) -> PhantomCollusionProsecutorPhase1Certificate:
        """Inspeciona ρ TAL COMO SE PRESENTA (sin normalizar) y emite P1."""
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
    def _p1_chain_fields(cert: PhantomCollusionProsecutorPhase1Certificate) -> Tuple[object, ...]:
        """Orden canónico que el hash vincula (idéntico en emisión y compuerta)."""
        return (
            cert.timestamp, cert.tr_rho, cert.is_hermitian, cert.hermiticity_residual,
            cert.is_psd, cert.min_eigenvalue, cert.max_eigenvalue, cert.purity,
            cert.von_neumann_entropy, cert.is_admissible, cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 1 → Φ₁₂ → Fase 2 ─────────────────────────────────
    def emit_phase1_certificate(
        self, **fields: object
    ) -> PhantomCollusionProsecutorPhase1Certificate:
        """ÚLTIMO método de F1. Firma P1 y lo entrega a la Compuerta Homomórfica Φ₁₂."""
        base = dict(fields)
        base["schema_version"] = _AGENT_SCHEMA
        provisional = PhantomCollusionProsecutorPhase1Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p1_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, chain)
        return PhantomCollusionProsecutorPhase1Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — ORIENT: juicio del motor fiscal y vinculación espectral
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionProsecutorPhase2Orientor:
    """Fase 2 OODA. Ejecuta el motor fiscal v14 y vincula P1 + pasaporte + orden + RSI³."""

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovPhantomCollusionMitigatorProsecutorEngine,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = secret_key
        self._engine = engine
        self._clock = clock

    def execute_phase2(
        self,
        p1_cert: PhantomCollusionProsecutorPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        policy: PhantomProsecutorAuditPolicy,
    ) -> PhantomCollusionProsecutorPhase2Certificate:
        """PRIMER método de F2. Consume P1 (vía Φ₁₂) y la política; invoca el motor por palabra."""
        if not isinstance(p1_cert, PhantomCollusionProsecutorPhase1Certificate):
            raise TypeError("Fase 2 exige un PhantomCollusionProsecutorPhase1Certificate.")
        report = self._engine.execute_collusion_prosecutor_audit(
            apu_cost_matrix=apu_cost_matrix,
            exponents=np.asarray(exponents, dtype=np.float64),
            **policy.to_engine_kwargs(),
        )
        return self.emit_phase2_certificate(p1_cert, report, float(self._clock()))

    @staticmethod
    def _p2_chain_fields(cert: PhantomCollusionProsecutorPhase2Certificate) -> Tuple[object, ...]:
        r = cert.prosecution_report
        dc = r.dirac_crowbar_report
        return (
            cert.timestamp, cert.p1_chain_hash, cert.p1_admissible,
            r.input_digest, r.passport_hash, r.schema_version,
            int(r.truth_value), bool(r.metamorphic_selftest_passed),
            float(r.banach_contraction_lip), bool(r.prosecution_warrant),
            bool(dc.crowbar_triggered), float(dc.crowbar_latency_ns),
            bool(dc.hardware_actuated), bool(dc.security_veto),
            bool(dc.dirac_event_authorized), float(dc.annihilation_energy_kev),
            bool(r.associator_report.phantom_collusion_suspected),
            getattr(r, "rsi_certificate_chain", ""),
            cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 2 → Φ₂₃ → Fase 3 ─────────────────────────────────
    def emit_phase2_certificate(
        self,
        p1_cert: PhantomCollusionProsecutorPhase1Certificate,
        report: PhantomCollusionProsecutorReport,
        timestamp: float,
    ) -> PhantomCollusionProsecutorPhase2Certificate:
        """ÚLTIMO método de F2. Firma P2 vinculando P1 y el informe fiscal del motor."""
        base = {
            "timestamp": timestamp,
            "prosecution_report": report,
            "p1_chain_hash": p1_cert.chain_hash,
            "p1_admissible": p1_cert.is_admissible,
            "schema_version": _AGENT_SCHEMA,
        }
        provisional = PhantomCollusionProsecutorPhase2Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p2_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, chain)
        return PhantomCollusionProsecutorPhase2Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — DECIDE: Heyting, integridad de actuación, Φ_sem y raíz Merkle
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionProsecutorPhase3Decider:
    """Fase 3 OODA. Compone el veredicto, verifica la actuación y firma el estado."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase3(
        self, p2_cert: PhantomCollusionProsecutorPhase2Certificate
    ) -> PhantomCollusionProsecutorSovereignState:
        """PRIMER método de F3. Consume P2 (vía Φ₂₃) y adjudica sobre Heyting Ω₃."""
        if not isinstance(p2_cert, PhantomCollusionProsecutorPhase2Certificate):
            raise TypeError("Fase 3 exige un PhantomCollusionProsecutorPhase2Certificate.")
        report = p2_cert.prosecution_report
        engine_truth = report.truth_value
        p1_verdict = (
            HeytingTruthValue.COHERENT if p2_cert.p1_admissible else HeytingTruthValue.VETOED
        )
        final_truth = engine_truth.meet(p1_verdict)
        dc = report.dirac_crowbar_report
        consistent = _actuation_integrity_ok(report)
        if dc.crowbar_triggered:
            logger.warning(
                "[ORDEN CERTIFICADA] Crowbar SIMULADO %.2f ns (presupuesto, NO hardware).",
                dc.crowbar_latency_ns,
            )
        metrics = self._visceral_metrics(report, final_truth, p2_cert.p1_admissible)
        return self.emit_sovereign_state(
            timestamp=float(self._clock()),
            final_truth=final_truth,
            engine_truth=engine_truth,
            p1_admissible=p2_cert.p1_admissible,
            report=report,
            consistent=consistent,
            p2_cert=p2_cert,
            metrics=metrics,
        )

    @staticmethod
    def _visceral_metrics(
        report: PhantomCollusionProsecutorReport,
        final_truth: HeytingTruthValue,
        p1_admissible: bool,
    ) -> PhantomCollusionProsecutorVisceralMetrics:
        """Φ_sem: estado factual derivado de banderas certificadas (sin retórica)."""
        assoc = report.associator_report
        dc = report.dirac_crowbar_report
        ratio_max = max(
            assoc.pationion_ratio_32d, assoc.chingon_ratio_64d, assoc.routon_ratio_128d
        )
        if report.prosecution_warrant:
            status = "ORDEN_CERTIFICADA_DESMANTELAMIENTO (crowbar SIMULADO, sin hardware)"
        elif final_truth == HeytingTruthValue.VETOED:
            status = ("VETO_SIN_ORDEN: " + (
                "rho_inadmisible(P1)" if not p1_admissible else "verificador/autoprueba"
            ))
        elif final_truth == HeytingTruthValue.DEGRADED:
            causes: List[str] = []
            if assoc.phantom_collusion_suspected:
                causes.append("asociadores_sospechados(heuristico)")
            if report.rademacher_walsh_norm > _RW_DEGRADED_TOL:
                causes.append("influencia_RW_alta")
            if report.cauchy_riemann_residual > _CR_RESIDUAL_TOL:
                causes.append("armonicidad_residual")
            if not report.melnikov_eps_regime_valid:
                causes.append("melnikov_fuera_de_regimen")
            if report.poincare_melnikov_consistent is False:
                causes.append("tension_lyapunov_melnikov")
            status = "ALERTA_PREVENTIVA: " + (";".join(causes) if causes else "rsi_profundidad")
        else:
            status = "SIN_ORDEN_CERTIFICADA_ESPECTRO_COHERENTE"
        return PhantomCollusionProsecutorVisceralMetrics(
            wacc_protection_target_pct=_WACC_PROTECTION_TARGET_PCT,
            net_liquidity_release_target_pct=_NET_LIQUIDITY_TARGET_PCT,
            prosecution_warrant=bool(report.prosecution_warrant),
            phantom_collusion_suspected=bool(assoc.phantom_collusion_suspected),
            associator_ratio_max=float(ratio_max),
            dirac_energy_kev=float(dc.annihilation_energy_kev),
            dirac_event_authorized=bool(dc.dirac_event_authorized),
            smith_torsion_free=bool(report.smith_torsion_free),
            melnikov_chaos_detected=bool(report.melnikov_chaos_detected),
            rsi_accepted_depth=int(report.rsi_accepted_depth),
            capacity_third_difference=getattr(report, "capacity_third_difference", None),
            phantom_collusion_status=status,
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
        report: PhantomCollusionProsecutorReport,
        consistent: bool,
        p2_cert: PhantomCollusionProsecutorPhase2Certificate,
        metrics: PhantomCollusionProsecutorVisceralMetrics,
    ) -> PhantomCollusionProsecutorSovereignState:
        """ÚLTIMO método de F3. Raíz Merkle binaria + firma HMAC-P3. hardware_actuated=False."""
        dc = report.dirac_crowbar_report
        leaves = [
            p2_cert.p1_chain_hash,
            p2_cert.chain_hash,
            report.passport_hash,
            report.input_digest,
            f"verdict:{int(final_truth)}",
            f"warrant:{int(report.prosecution_warrant)}:trigger:{int(dc.crowbar_triggered)}:hardware:0",
            _chain_hash((
                metrics.wacc_protection_target_pct,
                metrics.net_liquidity_release_target_pct,
                metrics.prosecution_warrant,
                metrics.phantom_collusion_suspected,
                metrics.associator_ratio_max,
                metrics.dirac_energy_kev,
                metrics.dirac_event_authorized,
                metrics.smith_torsion_free,
                metrics.melnikov_chaos_detected,
                metrics.rsi_accepted_depth,
                metrics.capacity_third_difference,
                metrics.phantom_collusion_status,
            )),
            getattr(report, "rsi_certificate_chain", ""),
        ]
        merkle_root = _merkle_root(leaves)
        core = (
            timestamp, int(final_truth), int(engine_truth), p1_admissible,
            bool(report.prosecution_warrant), bool(dc.crowbar_triggered),
            float(dc.crowbar_latency_ns), True, False, consistent,
            merkle_root, _AGENT_SCHEMA,
        )
        state_sig = _sign(self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core))
        return PhantomCollusionProsecutorSovereignState(
            timestamp=timestamp,
            truth_value=final_truth,
            engine_truth_value=engine_truth,
            p1_admissible=p1_admissible,
            prosecution_warrant=bool(report.prosecution_warrant),
            crowbar_triggered=bool(dc.crowbar_triggered),
            crowbar_latency_ns=float(dc.crowbar_latency_ns),
            crowbar_latency_is_simulated=True,
            hardware_actuated=False,
            warrant_actuation_consistent=consistent,
            merkle_root_hash=merkle_root,
            state_signature=state_sig,
            visceral_metrics=metrics,
            schema_version=_AGENT_SCHEMA,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orquestador soberano: ciclo OODA con compuertas homomórficas Φ₁₂ y Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPhantomCollusionMitigatorProsecutorAgent:
    """Soberano Agéntico Fiscal Anti-Colusión (OODA trifásica). Actuación SIMULADA."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovPhantomCollusionMitigatorProsecutorEngine] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = _resolve_secret_key(secret_key)
        self._engine = engine or BoolNovikovPhantomCollusionMitigatorProsecutorEngine(
            secret_key=self._secret_key
        )
        self._clock = clock
        self._observer = PhantomCollusionProsecutorPhase1Observer(self._secret_key, clock)
        self._orientor = PhantomCollusionProsecutorPhase2Orientor(
            self._secret_key, self._engine, clock
        )
        self._decider = PhantomCollusionProsecutorPhase3Decider(self._secret_key, clock)
        if not self._metamorphic_selftest():
            raise RuntimeError("Autoprueba metamórfica del agente fiscal fallida (fail-closed).")
        logger.info("%s v%s iniciado correctamente.", self.__class__.__name__, _AGENT_VERSION)

    # ── Compuerta Homomórfica Φ₁₂ (re-deriva contenido→hash→firma) ─────────────
    def continue_to_phase2_gate(self, p1: PhantomCollusionProsecutorPhase1Certificate) -> None:
        """Φ₁₂: autentica P1 re-derivando hash y firma. ADMITE anomalías para su
        enjuiciamiento (la admisibilidad física se compone en Fase 3)."""
        if p1.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₁₂: esquema de certificado P1 no reconocido.")
        expected_chain = _chain_hash(
            PhantomCollusionProsecutorPhase1Observer._p1_chain_fields(p1)
        )
        if not hmac.compare_digest(expected_chain, p1.chain_hash):
            raise ValueError("Φ₁₂: el hash de cadena P1 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, expected_chain)
        if not hmac.compare_digest(expected_sig, p1.hmac_signature):
            raise ValueError("Φ₁₂: falla de autenticidad HMAC en P1.")

    # ── Compuerta Homomórfica Φ₂₃ (re-deriva + contrato de actuación) ──────────
    def continue_to_phase3_gate(self, p2: PhantomCollusionProsecutorPhase2Certificate) -> None:
        """Φ₂₃: autentica P2, exige Lip_cert<1 y el CONTRATO DE INTEGRIDAD DE ACTUACIÓN
        (orden⇔disparo, sin hardware, latencia simulada acotada, veto⇒orden, Dirac exacto)."""
        if p2.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₂₃: esquema de certificado P2 no reconocido.")
        expected_chain = _chain_hash(
            PhantomCollusionProsecutorPhase2Orientor._p2_chain_fields(p2)
        )
        if not hmac.compare_digest(expected_chain, p2.chain_hash):
            raise ValueError("Φ₂₃: el hash de cadena P2 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, expected_chain)
        if not hmac.compare_digest(expected_sig, p2.hmac_signature):
            raise ValueError("Φ₂₃: falla de autenticidad HMAC en P2.")
        rep = p2.prosecution_report
        if not (rep.banach_contraction_lip < 1.0):
            raise ValueError("Φ₂₃: violación de estabilidad (Lip_cert ≥ 1.0).")
        if not _actuation_integrity_ok(rep):
            raise ValueError("Φ₂₃: violación del contrato de integridad de actuación.")

    def verify_state_signature(self, state: PhantomCollusionProsecutorSovereignState) -> bool:
        """Verificación independiente de la firma del estado soberano."""
        core = (
            state.timestamp, int(state.truth_value), int(state.engine_truth_value),
            state.p1_admissible, state.prosecution_warrant, state.crowbar_triggered,
            state.crowbar_latency_ns, state.crowbar_latency_is_simulated,
            state.hardware_actuated, state.warrant_actuation_consistent,
            state.merkle_root_hash, state.schema_version,
        )
        expected = _sign(self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core))
        return hmac.compare_digest(expected, state.state_signature)

    # ── Ciclo OODA completo ────────────────────────────────────────────────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        policy: Optional[PhantomProsecutorAuditPolicy] = None,
        *,
        positron_hmac_valid: Optional[bool] = None,
    ) -> PhantomCollusionProsecutorSovereignState:
        """Ejecuta Observe→Φ₁₂→Orient→Φ₂₃→Decide sobre el fibrado dual Boole-Novikov."""
        eff_policy = policy if policy is not None else PhantomProsecutorAuditPolicy()
        if positron_hmac_valid is not None:
            eff_policy = replace(eff_policy, positron_hmac_valid=positron_hmac_valid)
        p1 = self._observer.execute_phase1(rho_mac)
        self.continue_to_phase2_gate(p1)                       # Φ₁₂
        p2 = self._orientor.execute_phase2(p1, apu_cost_matrix, exponents, eff_policy)
        self.continue_to_phase3_gate(p2)                       # Φ₂₃
        state = self._decider.execute_phase3(p2)
        logger.info(
            "Ciclo OODA Fiscal completado. Veredicto: %s | orden=%s | crowbar(sim)=%s",
            state.truth_value.name, state.prosecution_warrant, state.crowbar_triggered,
        )
        return state

    # ── Autoprueba metamórfica del agente (nivel meta-meta de gobernanza) ──────
    def _metamorphic_selftest(self) -> bool:
        """Verifica Merkle, Heyting, canonicalización, compuertas y forja de actuación."""
        try:
            # Merkle: vacío, hoja única y par (RFC-6962).
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
            # Ciclo rápido (sin Lyapunov) con ρ admisible y matriz unimodular.
            fast = PhantomProsecutorAuditPolicy(poincare_periods=0, positron_hmac_valid=True)
            rho = np.eye(2, dtype=np.complex128) / 2.0
            mat = np.array([[0.01, 0.01], [0.0, 0.01]])
            exps = np.array([0.15, 0.22, 0.45])
            p1 = self._observer.execute_phase1(rho)
            self.continue_to_phase2_gate(p1)
            # Tamper de contenido en P1 con hash intacto ⇒ Φ₁₂ debe rechazar.
            try:
                self.continue_to_phase2_gate(replace(p1, tr_rho=0.5))
                return False
            except ValueError:
                pass
            p2 = self._orientor.execute_phase2(p1, mat, exps, fast)
            self.continue_to_phase3_gate(p2)
            # Tamper de admisibilidad en P2 ⇒ Φ₂₃ debe rechazar.
            try:
                self.continue_to_phase3_gate(replace(p2, p1_admissible=not p2.p1_admissible))
                return False
            except ValueError:
                pass
            # FORJA DE ACTUACIÓN no autorizada: disparo sin orden ⇒ Φ₂₃ debe rechazar.
            rep = p2.prosecution_report
            forged_dc = replace(
                rep.dirac_crowbar_report,
                crowbar_triggered=not rep.dirac_crowbar_report.crowbar_triggered,
            )
            forged_p2 = replace(p2, prosecution_report=replace(rep, dirac_crowbar_report=forged_dc))
            try:
                self.continue_to_phase3_gate(forged_p2)
                return False
            except ValueError:
                pass
            state = self._decider.execute_phase3(p2)
            if not self.verify_state_signature(state):
                return False
            if state.hardware_actuated or not state.crowbar_latency_is_simulated:
                return False
            if not state.warrant_actuation_consistent:
                return False
            if state.crowbar_triggered != state.prosecution_warrant:
                return False
            if state.crowbar_latency_ns > _CROWBAR_MAX_LATENCY_NS:
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica del agente fiscal: excepción (fail-closed).")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPhantomCollusionMitigatorProsecutorAgent v14.0 ===")
    agent = BoolNovikovPhantomCollusionMitigatorProsecutorAgent(secret_key=b"demo-key-0123456789abcdef")
    exps_test = np.array([0.15, 0.22, 0.45], dtype=np.float64)
    policy = PhantomProsecutorAuditPolicy(
        positron_hmac_valid=True, capacity_series=(1.0, 2.0, 4.0, 8.0)
    )
    # A: cuantización a escala 100 ⇒ factores de torsión ⇒ orden certificada.
    mat_a = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.9]])
    # B: cuantiza a [[1,1],[0,1]] unimodular ⇒ sin orden certificada.
    mat_b = np.array([[0.01, 0.01], [0.0, 0.01]])
    for label, mat in (("A (torsión)", mat_a), ("B (unimodular)", mat_b)):
        rho = np.eye(mat.shape[0], dtype=np.complex128) / float(mat.shape[0])
        st = agent.run_sovereign_governance_cycle(rho, mat, exps_test, policy)
        m = st.visceral_metrics
        print(f"\n=== Escenario {label}: {st.truth_value.name} ({hex(st.truth_value.value)}) ===")
        print(f"  Veredicto motor / ρ admisible : {st.engine_truth_value.name} / {st.p1_admissible}")
        print(f"  Orden certificada (warrant)   : {st.prosecution_warrant}")
        print(f"  Crowbar SIMULADO / hardware   : {st.crowbar_triggered} @ {st.crowbar_latency_ns:.2f} ns / {st.hardware_actuated}")
        print(f"  Integridad orden⇔actuación    : {st.warrant_actuation_consistent}")
        print(f"  Dirac (keV) / evento          : {m.dirac_energy_kev:.4f} / {m.dirac_event_authorized}")
        print(f"  Colusión sospechada / ratio   : {m.phantom_collusion_suspected} / {m.associator_ratio_max:.4f}")
        print(f"  Torsión libre / RSI depth     : {m.smith_torsion_free} / {m.rsi_accepted_depth}")
        print(f"  Estado fiscal                 : {m.phantom_collusion_status}")
        print(f"  Raíz Merkle / Firma           : {st.merkle_root_hash[:16]}... / {st.state_signature[:16]}...")
        print(f"  Firma verificada              : {agent.verify_state_signature(st)}")
    print("\n=== TEST COMPLETADO ===")