# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPhantomCollusionMitigatorAgent v14.0.0-PhD                                     ║
║ Ubicación: app/wisdom/bool_novikov_phantom_collusion_mitigator_agent.py                             ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-PHANTOM-COLLUSION-MITIGATOR-AGENT-POINCARE-RSI3-PhD         ║
║ Gobernanza: OODA Trifásica (①Observe → ②Orient → ③Decide) con Compuertas Φ₁₂ y Φ₂₃ sobre el Motor    ║
║            raíz de Mitigación v14.0. SIN ACTUACIÓN: detecta, recomienda y escala al fiscal.          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  F1 PhantomCollusionPhase1Observer
     └ emit_phase1_certificate        ÚLTIMO de F1 → Φ₁₂ → execute_phase2 (PRIMERO de F2).
  F2 PhantomCollusionPhase2Orientor
     └ emit_phase2_certificate        ÚLTIMO de F2 → Φ₂₃ → execute_phase3 (PRIMERO de F3).
  F3 PhantomCollusionPhase3Decider
     └ emit_sovereign_state           ÚLTIMO de F3 (estado soberano firmado).

CAMBIOS RIGUROSOS v13.0 → v14.0 (granulares):
  · SE ELIMINA el disparo ESP32 Crowbar: el motor raíz declara que no modela hardware. El agente
    raíz DETECTA y RECOMIENDA; la actuación (simulada) corresponde al agente fiscal. El estado
    soberano expone `mitigation_recommended` (certificada por el motor) y `escalation_to_prosecutor`
    (veredicto final VETOED), con `hardware_actuated=False` invariante.
  · Φ₁₂/Φ₂₃ RE-DERIVAN el hash de cadena desde los campos canónicos y la firma HMAC
    (contenido→hash→firma). La v13 verificaba la firma sobre el hash recibido, pero no re-derivaba
    el hash: alterar tr_rho/is_psd con hash intacto pasaba.
  · Φ₂₃ exige además el CONTRATO DE INTEGRIDAD DE LA RECOMENDACIÓN, re-derivado del informe:
        smith_torsion_free ⇔ todo |dᵢ|=1 en la cadena SNF
        smith_torsion_order = ∏|dᵢ| (1 si la cadena es vacía)
        mitigation_recommended ⇔ (Gromov ∨ ¬torsión libre ∨ caos Melnikov ∨ v < 0.10)
        phantom_collusion_suspected = associator_report.phantom_collusion_suspected
        dirac_annihilation_kev ∈ {0, 1021.9979}
    y la contracción de Banach Lip_cert < 1.
  · F1: Hermiticidad ‖ρ−ρ†‖, Tr(ρ)=1 (tol 1e-10), PSD, pureza Tr(ρ²)≤1 y entropía de von Neumann.
    Espectro sobre la parte Hermítica. Un solo timestamp; reloj inyectable; canonicalización .17g.
  · F2: PhantomCollusionAuditPolicy (frozen, validada) reenvía TODOS los parámetros del motor v14.
    Vincula P1 (hash+admisibilidad), input_digest, pasaporte, esquema, veredicto, autoprueba,
    recomendación, sospecha, SNF, Gromov, Melnikov, Krull, Dirac y cadena RSI³.
  · F3: veredicto final = motor.meet(⊤ si ρ admisible else ⊥). Raíz Merkle BINARIA RFC-6962 y
    firma HMAC-P3 del estado. Estado factual derivado de banderas certificadas (sin retórica).
  · Φ_sem: se ELIMINA kv_cache_compression_pct=86.4 (contaminación cruzada: el motor mitigador no
    comprime KV). Métricas genuinas: recomendación, sospecha, razones de asociadores, SNF, Gromov,
    Melnikov, Dirac, profundidad RSI, jerk medido. WACC/liquidez son OBJETIVOS de política declarados.
  · Epistemia: la sospecha por asociadores es EVIDENCIA (degrada); solo los fallos CERTIFICADOS
    producen recomendación de mitigación, y solo el veredicto final VETOED produce escalada.
  · Clave HMAC por entorno; autoprueba metamórfica en init con forja de recomendación.

LÍMITES DECLARADOS:
  · El agente NO actúa hardware ni simula disparos: no hay HAL. La escalada es una señal de
    gobernanza hacia el agente fiscal, que sí modela (simuladamente) el interlock.
  · P1 certifica ρ TAL COMO SE PRESENTA; la normalización interna del motor no anula la
    inadmisibilidad de una ρ con Tr≠1 (se compone como veto en F3).
  · `_KRULL_FLOOR` debe coincidir con `_DEFAULT_KRULL_FLOOR` del motor (acoplamiento declarado).
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

# ── Importación del motor raíz de mitigación (dos rutas de paquete) ────────────
try:
    from bool_novikov_phantom_collusion_mitigator_engine import (
        BoolNovikovPhantomCollusionMitigatorEngine,
        HeytingTruthValue,
        PhantomCollusionSpectrumReport,
    )
except ImportError:  # pragma: no cover - ruta de paquete alternativa
    try:
        from app.wisdom.bool_novikov_phantom_collusion_mitigator_engine import (
            BoolNovikovPhantomCollusionMitigatorEngine,
            HeytingTruthValue,
            PhantomCollusionSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_phantom_collusion_mitigator_engine."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovPhantomCollusionMitigatorAgent")

# ── Versiones y dominios HMAC ──────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-BOOL-NOVIKOV-PHANTOM-COLLUSION-MITIGATOR-AGENT-POINCARE-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
_AGENT_SCHEMA: Final[str] = "mitigator-agent-cert-v14.0"
_HMAC_KEY_ENV: Final[str] = "APU_PHANTOM_MITIGATOR_AGENT_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_MITIGATOR_AGENT_DEV_ONLY_PHANTOM_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"PHANTOM-COLLUSION-AGENT-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"PHANTOM-COLLUSION-AGENT-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"PHANTOM-COLLUSION-AGENT-P3-SOVEREIGN-v14.0|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"

# ── Tolerancias físicas de Fase 1 ──────────────────────────────────────────────
_TRACE_TOL: Final[float] = 1.0e-10
_HERMITICITY_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_PURITY_TOL: Final[float] = 1.0e-10

# ── Acoplamientos declarados con el motor v14 ──────────────────────────────────
_KRULL_FLOOR: Final[float] = 0.10               # = motor._DEFAULT_KRULL_FLOOR
_DIRAC_TOTAL_KEV: Final[float] = 1021.9979
_DIRAC_ENERGY_TOL: Final[float] = 1.0e-6
_RW_DEGRADED_TOL: Final[float] = 0.5
_CR_RESIDUAL_TOL: Final[float] = 1.0e-3

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


def _recommendation_integrity_ok(report: PhantomCollusionSpectrumReport) -> bool:
    """Contrato de integridad: la recomendación y la semántica SNF se RE-DERIVAN del informe.
    Nada heurístico produce recomendación; nada certificado queda sin recomendar."""
    factors = report.snf_diagonal_factors
    torsion_free_ok = bool(report.smith_torsion_free == all(d == 1 for d in factors))
    order_ok = bool(report.smith_torsion_order == (math.prod(factors) if factors else 1))
    assoc_ok = bool(
        report.phantom_collusion_suspected
        == report.associator_report.phantom_collusion_suspected
    )
    certified = bool(
        report.is_gromov_violated
        or (not report.smith_torsion_free)
        or report.melnikov_chaos_detected
        or (report.krull_valuation < _KRULL_FLOOR)
    )
    rec_ok = bool(report.mitigation_recommended == certified)
    e = float(report.dirac_annihilation_kev)
    energy_ok = bool(abs(e) <= _DIRAC_ENERGY_TOL or abs(e - _DIRAC_TOTAL_KEV) <= _DIRAC_ENERGY_TOL)
    return bool(torsion_free_ok and order_ok and assoc_ok and rec_ok and energy_ok)


# ══════════════════════════════════════════════════════════════════════════════
# Contratos: certificados, política, métricas y estado soberano
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PhantomCollusionPhase1Certificate:
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
class PhantomCollusionPhase2Certificate:
    """Certificado P2 (Orient): informe del motor mitigador vinculado a P1."""

    timestamp: float
    spectrum_report: PhantomCollusionSpectrumReport
    p1_chain_hash: str
    p1_admissible: bool
    chain_hash: str
    hmac_signature: str
    schema_version: str


@dataclass(frozen=True)
class PhantomCollusionAuditPolicy:
    """Política de auditoría: reenvío validado de los parámetros del motor mitigador v14."""

    positron_hmac_valid: Optional[bool] = None
    ball_r: float = 0.50
    cylinder_R: float = 0.85
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
class PhantomCollusionVisceralMetrics:
    """Φ_sem: objetivos de política declarados + medidas genuinas del mitigador.
    Nota: NO incluye compresión KV (el motor mitigador no comprime; eso es del weaver)."""

    wacc_protection_target_pct: float
    net_liquidity_release_target_pct: float
    mitigation_recommended: bool
    phantom_collusion_suspected: bool
    associator_ratio_max: float
    smith_torsion_free: bool
    smith_torsion_order: int
    is_gromov_violated: bool
    melnikov_chaos_detected: bool
    dirac_energy_kev: float
    rsi_accepted_depth: int
    capacity_third_difference: Optional[float]
    anti_collusion_status: str
    policy_provenance: str


@dataclass(frozen=True)
class PhantomCollusionSovereignState:
    """Estado soberano final (Decide), firmado sobre raíz Merkle. SIN actuación de hardware."""

    timestamp: float
    truth_value: HeytingTruthValue
    engine_truth_value: HeytingTruthValue
    p1_admissible: bool
    mitigation_recommended: bool
    escalation_to_prosecutor: bool
    hardware_actuated: bool                 # SIEMPRE False: el mitigador raíz no actúa
    merkle_root_hash: str
    state_signature: str
    visceral_metrics: PhantomCollusionVisceralMetrics
    schema_version: str


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — OBSERVE: ingestión e invariantes físicos de ρ (MAC)
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionPhase1Observer:
    """Fase 1 OODA. Certifica Hermiticidad, traza, PSD, pureza y entropía de ρ."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128], trace_tolerance: float = _TRACE_TOL
    ) -> PhantomCollusionPhase1Certificate:
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
    def _p1_chain_fields(cert: PhantomCollusionPhase1Certificate) -> Tuple[object, ...]:
        """Orden canónico que el hash vincula (idéntico en emisión y compuerta)."""
        return (
            cert.timestamp, cert.tr_rho, cert.is_hermitian, cert.hermiticity_residual,
            cert.is_psd, cert.min_eigenvalue, cert.max_eigenvalue, cert.purity,
            cert.von_neumann_entropy, cert.is_admissible, cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 1 → Φ₁₂ → Fase 2 ─────────────────────────────────
    def emit_phase1_certificate(self, **fields: object) -> PhantomCollusionPhase1Certificate:
        """ÚLTIMO método de F1. Firma P1 y lo entrega a la Compuerta Homomórfica Φ₁₂."""
        base = dict(fields)
        base["schema_version"] = _AGENT_SCHEMA
        provisional = PhantomCollusionPhase1Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p1_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, chain)
        return PhantomCollusionPhase1Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — ORIENT: juicio del motor mitigador y vinculación espectral
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionPhase2Orientor:
    """Fase 2 OODA. Ejecuta el motor mitigador v14 y vincula P1 + pasaporte + SNF + RSI³."""

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovPhantomCollusionMitigatorEngine,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = secret_key
        self._engine = engine
        self._clock = clock

    def execute_phase2(
        self,
        p1_cert: PhantomCollusionPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        policy: PhantomCollusionAuditPolicy,
    ) -> PhantomCollusionPhase2Certificate:
        """PRIMER método de F2. Consume P1 (vía Φ₁₂) y la política; invoca el motor por palabra."""
        if not isinstance(p1_cert, PhantomCollusionPhase1Certificate):
            raise TypeError("Fase 2 exige un PhantomCollusionPhase1Certificate.")
        report = self._engine.execute_phantom_collusion_mitigation_audit(
            apu_cost_matrix=apu_cost_matrix,
            exponents=np.asarray(exponents, dtype=np.float64),
            **policy.to_engine_kwargs(),
        )
        return self.emit_phase2_certificate(p1_cert, report, float(self._clock()))

    @staticmethod
    def _p2_chain_fields(cert: PhantomCollusionPhase2Certificate) -> Tuple[object, ...]:
        r = cert.spectrum_report
        return (
            cert.timestamp, cert.p1_chain_hash, cert.p1_admissible,
            r.input_digest, r.passport_hash, r.schema_version,
            int(r.truth_value), bool(r.metamorphic_selftest_passed),
            float(r.banach_contraction_lip), bool(r.mitigation_recommended),
            bool(r.phantom_collusion_suspected), bool(r.smith_torsion_free),
            int(r.smith_torsion_order),
            ",".join(str(d) for d in r.snf_diagonal_factors),
            bool(r.is_gromov_violated), bool(r.melnikov_chaos_detected),
            float(r.krull_valuation), float(r.dirac_annihilation_kev),
            getattr(r, "rsi_certificate_chain", ""),
            cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 2 → Φ₂₃ → Fase 3 ─────────────────────────────────
    def emit_phase2_certificate(
        self,
        p1_cert: PhantomCollusionPhase1Certificate,
        report: PhantomCollusionSpectrumReport,
        timestamp: float,
    ) -> PhantomCollusionPhase2Certificate:
        """ÚLTIMO método de F2. Firma P2 vinculando P1 y el informe del motor mitigador."""
        base = {
            "timestamp": timestamp,
            "spectrum_report": report,
            "p1_chain_hash": p1_cert.chain_hash,
            "p1_admissible": p1_cert.is_admissible,
            "schema_version": _AGENT_SCHEMA,
        }
        provisional = PhantomCollusionPhase2Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p2_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, chain)
        return PhantomCollusionPhase2Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — DECIDE: Heyting, escalada al fiscal, Φ_sem y raíz Merkle
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionPhase3Decider:
    """Fase 3 OODA. Compone el veredicto, decide la escalada y firma el estado (sin hardware)."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase3(
        self, p2_cert: PhantomCollusionPhase2Certificate
    ) -> PhantomCollusionSovereignState:
        """PRIMER método de F3. Consume P2 (vía Φ₂₃) y adjudica sobre Heyting Ω₃."""
        if not isinstance(p2_cert, PhantomCollusionPhase2Certificate):
            raise TypeError("Fase 3 exige un PhantomCollusionPhase2Certificate.")
        report = p2_cert.spectrum_report
        engine_truth = report.truth_value
        p1_verdict = (
            HeytingTruthValue.COHERENT if p2_cert.p1_admissible else HeytingTruthValue.VETOED
        )
        final_truth = engine_truth.meet(p1_verdict)
        # Escalada: solo el veredicto final VETOED (fallos certificados, ρ inadmisible o
        # verificador). La sospecha heurística degrada y NUNCA escala por sí sola.
        escalation = bool(final_truth == HeytingTruthValue.VETOED)
        if escalation:
            logger.warning(
                "[ESCALADA] Veredicto VETOED: se recomienda juicio del agente fiscal "
                "(señal de gobernanza; SIN actuación de hardware)."
            )
        metrics = self._visceral_metrics(report, final_truth, p2_cert.p1_admissible)
        return self.emit_sovereign_state(
            timestamp=float(self._clock()),
            final_truth=final_truth,
            engine_truth=engine_truth,
            p1_admissible=p2_cert.p1_admissible,
            report=report,
            escalation=escalation,
            p2_cert=p2_cert,
            metrics=metrics,
        )

    @staticmethod
    def _visceral_metrics(
        report: PhantomCollusionSpectrumReport,
        final_truth: HeytingTruthValue,
        p1_admissible: bool,
    ) -> PhantomCollusionVisceralMetrics:
        """Φ_sem: estado factual derivado de banderas certificadas (sin retórica)."""
        assoc = report.associator_report
        ratio_max = max(assoc.pationionic_ratio, assoc.chingonic_ratio, assoc.routonic_ratio)
        if report.mitigation_recommended:
            status = "MITIGACION_CERTIFICADA_RECOMENDADA (escalada al fiscal)"
        elif final_truth == HeytingTruthValue.VETOED:
            status = "VETO_SIN_MITIGACION_ESPECTRAL: " + (
                "rho_inadmisible(P1)" if not p1_admissible else "verificador/autoprueba"
            )
        elif final_truth == HeytingTruthValue.DEGRADED:
            causes: List[str] = []
            if report.phantom_collusion_suspected:
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
            status = "SIN_MITIGACION_CERTIFICADA_ESPECTRO_COHERENTE"
        return PhantomCollusionVisceralMetrics(
            wacc_protection_target_pct=_WACC_PROTECTION_TARGET_PCT,
            net_liquidity_release_target_pct=_NET_LIQUIDITY_TARGET_PCT,
            mitigation_recommended=bool(report.mitigation_recommended),
            phantom_collusion_suspected=bool(report.phantom_collusion_suspected),
            associator_ratio_max=float(ratio_max),
            smith_torsion_free=bool(report.smith_torsion_free),
            smith_torsion_order=int(report.smith_torsion_order),
            is_gromov_violated=bool(report.is_gromov_violated),
            melnikov_chaos_detected=bool(report.melnikov_chaos_detected),
            dirac_energy_kev=float(report.dirac_annihilation_kev),
            rsi_accepted_depth=int(report.rsi_accepted_depth),
            capacity_third_difference=getattr(report, "capacity_third_difference", None),
            anti_collusion_status=status,
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
        report: PhantomCollusionSpectrumReport,
        escalation: bool,
        p2_cert: PhantomCollusionPhase2Certificate,
        metrics: PhantomCollusionVisceralMetrics,
    ) -> PhantomCollusionSovereignState:
        """ÚLTIMO método de F3. Raíz Merkle binaria + firma HMAC-P3. hardware_actuated=False."""
        leaves = [
            p2_cert.p1_chain_hash,
            p2_cert.chain_hash,
            report.passport_hash,
            report.input_digest,
            f"verdict:{int(final_truth)}",
            f"mitigation:{int(report.mitigation_recommended)}:escalation:{int(escalation)}:hardware:0",
            _chain_hash((
                metrics.wacc_protection_target_pct,
                metrics.net_liquidity_release_target_pct,
                metrics.mitigation_recommended,
                metrics.phantom_collusion_suspected,
                metrics.associator_ratio_max,
                metrics.smith_torsion_free,
                metrics.smith_torsion_order,
                metrics.is_gromov_violated,
                metrics.melnikov_chaos_detected,
                metrics.dirac_energy_kev,
                metrics.rsi_accepted_depth,
                metrics.capacity_third_difference,
                metrics.anti_collusion_status,
            )),
            getattr(report, "rsi_certificate_chain", ""),
        ]
        merkle_root = _merkle_root(leaves)
        core = (
            timestamp, int(final_truth), int(engine_truth), p1_admissible,
            bool(report.mitigation_recommended), escalation, False,
            merkle_root, _AGENT_SCHEMA,
        )
        state_sig = _sign(self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core))
        return PhantomCollusionSovereignState(
            timestamp=timestamp,
            truth_value=final_truth,
            engine_truth_value=engine_truth,
            p1_admissible=p1_admissible,
            mitigation_recommended=bool(report.mitigation_recommended),
            escalation_to_prosecutor=escalation,
            hardware_actuated=False,
            merkle_root_hash=merkle_root,
            state_signature=state_sig,
            visceral_metrics=metrics,
            schema_version=_AGENT_SCHEMA,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orquestador soberano: ciclo OODA con compuertas homomórficas Φ₁₂ y Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPhantomCollusionMitigatorAgent:
    """Soberano Agéntico Mitigador de Colusiones Fantasma (OODA trifásica, sin actuación)."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovPhantomCollusionMitigatorEngine] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = _resolve_secret_key(secret_key)
        self._engine = engine or BoolNovikovPhantomCollusionMitigatorEngine(
            secret_key=self._secret_key
        )
        self._clock = clock
        self._observer = PhantomCollusionPhase1Observer(self._secret_key, clock)
        self._orientor = PhantomCollusionPhase2Orientor(self._secret_key, self._engine, clock)
        self._decider = PhantomCollusionPhase3Decider(self._secret_key, clock)
        if not self._metamorphic_selftest():
            raise RuntimeError("Autoprueba metamórfica del agente mitigador fallida (fail-closed).")
        logger.info("%s v%s iniciado correctamente.", self.__class__.__name__, _AGENT_VERSION)

    # ── Compuerta Homomórfica Φ₁₂ (re-deriva contenido→hash→firma) ─────────────
    def continue_to_phase2_gate(self, p1: PhantomCollusionPhase1Certificate) -> None:
        """Φ₁₂: autentica P1 re-derivando hash y firma. ADMITE anomalías para su
        enjuiciamiento (la admisibilidad física se compone en Fase 3)."""
        if p1.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₁₂: esquema de certificado P1 no reconocido.")
        expected_chain = _chain_hash(PhantomCollusionPhase1Observer._p1_chain_fields(p1))
        if not hmac.compare_digest(expected_chain, p1.chain_hash):
            raise ValueError("Φ₁₂: el hash de cadena P1 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, expected_chain)
        if not hmac.compare_digest(expected_sig, p1.hmac_signature):
            raise ValueError("Φ₁₂: falla de autenticidad HMAC en P1.")

    # ── Compuerta Homomórfica Φ₂₃ (re-deriva + integridad de la recomendación) ──
    def continue_to_phase3_gate(self, p2: PhantomCollusionPhase2Certificate) -> None:
        """Φ₂₃: autentica P2, exige Lip_cert<1 y re-deriva la semántica SNF y la
        recomendación de mitigación desde los fallos certificados del informe."""
        if p2.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₂₃: esquema de certificado P2 no reconocido.")
        expected_chain = _chain_hash(PhantomCollusionPhase2Orientor._p2_chain_fields(p2))
        if not hmac.compare_digest(expected_chain, p2.chain_hash):
            raise ValueError("Φ₂₃: el hash de cadena P2 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, expected_chain)
        if not hmac.compare_digest(expected_sig, p2.hmac_signature):
            raise ValueError("Φ₂₃: falla de autenticidad HMAC en P2.")
        rep = p2.spectrum_report
        if not (rep.banach_contraction_lip < 1.0):
            raise ValueError("Φ₂₃: violación de estabilidad (Lip_cert ≥ 1.0).")
        if not _recommendation_integrity_ok(rep):
            raise ValueError("Φ₂₃: violación del contrato de integridad de la recomendación.")

    def verify_state_signature(self, state: PhantomCollusionSovereignState) -> bool:
        """Verificación independiente de la firma del estado soberano."""
        core = (
            state.timestamp, int(state.truth_value), int(state.engine_truth_value),
            state.p1_admissible, state.mitigation_recommended,
            state.escalation_to_prosecutor, state.hardware_actuated,
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
        policy: Optional[PhantomCollusionAuditPolicy] = None,
        *,
        positron_hmac_valid: Optional[bool] = None,
    ) -> PhantomCollusionSovereignState:
        """Ejecuta Observe→Φ₁₂→Orient→Φ₂₃→Decide sobre el fibrado dual Boole-Novikov."""
        eff_policy = policy if policy is not None else PhantomCollusionAuditPolicy()
        if positron_hmac_valid is not None:
            eff_policy = replace(eff_policy, positron_hmac_valid=positron_hmac_valid)
        p1 = self._observer.execute_phase1(rho_mac)
        self.continue_to_phase2_gate(p1)                       # Φ₁₂
        p2 = self._orientor.execute_phase2(p1, apu_cost_matrix, exponents, eff_policy)
        self.continue_to_phase3_gate(p2)                       # Φ₂₃
        state = self._decider.execute_phase3(p2)
        logger.info(
            "Ciclo OODA Anti-Colusión completado. Veredicto: %s | mitigación=%s | escalada=%s",
            state.truth_value.name, state.mitigation_recommended, state.escalation_to_prosecutor,
        )
        return state

    # ── Autoprueba metamórfica del agente (nivel meta-meta de gobernanza) ──────
    def _metamorphic_selftest(self) -> bool:
        """Verifica Merkle, Heyting, canonicalización, compuertas y forja de recomendación."""
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
            fast = PhantomCollusionAuditPolicy(poincare_periods=0, positron_hmac_valid=True)
            rho = np.eye(2, dtype=np.complex128) / 2.0
            mat = np.array([[0.01, 0.01], [0.0, 0.01]])
            exps = np.array([0.15, 0.25, 0.40])
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
            if not _recommendation_integrity_ok(p2.spectrum_report):
                return False
            # Tamper de admisibilidad en P2 ⇒ Φ₂₃ debe rechazar.
            try:
                self.continue_to_phase3_gate(replace(p2, p1_admissible=not p2.p1_admissible))
                return False
            except ValueError:
                pass
            # FORJA DE RECOMENDACIÓN: mitigación sin fallo certificado ⇒ Φ₂₃ debe rechazar.
            rep = p2.spectrum_report
            forged = replace(rep, mitigation_recommended=not rep.mitigation_recommended)
            try:
                self.continue_to_phase3_gate(replace(p2, spectrum_report=forged))
                return False
            except ValueError:
                pass
            state = self._decider.execute_phase3(p2)
            if not self.verify_state_signature(state):
                return False
            if state.hardware_actuated:            # el mitigador raíz NUNCA actúa hardware
                return False
            if state.escalation_to_prosecutor != (state.truth_value == HeytingTruthValue.VETOED):
                return False
            # ρ inadmisible (Tr≠1) ⇒ veto compuesto y escalada, aunque el motor sea coherente.
            bad_rho = np.eye(2, dtype=np.complex128)   # Tr = 2
            p1_bad = self._observer.execute_phase1(bad_rho)
            if p1_bad.is_admissible:
                return False
            self.continue_to_phase2_gate(p1_bad)     # Φ₁₂ admite la anomalía
            p2_bad = self._orientor.execute_phase2(p1_bad, mat, exps, fast)
            self.continue_to_phase3_gate(p2_bad)
            st_bad = self._decider.execute_phase3(p2_bad)
            if st_bad.truth_value is not HeytingTruthValue.VETOED:
                return False
            if not st_bad.escalation_to_prosecutor:
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica del agente mitigador: excepción (fail-closed).")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPhantomCollusionMitigatorAgent v14.0 ===")
    agent = BoolNovikovPhantomCollusionMitigatorAgent(secret_key=b"demo-key-0123456789abcdef")
    exps_test = np.array([0.15, 0.25, 0.40, 0.85], dtype=np.float64)
    policy = PhantomCollusionAuditPolicy(
        positron_hmac_valid=True, capacity_series=(1.0, 2.0, 4.0, 8.0)
    )
    # A: cuantiza a diag 105 / off 5 ⇒ gcd 5 ⇒ torsión ⇒ mitigación certificada.
    mat_a = np.eye(4) + 0.05 * np.ones((4, 4))
    # B: cuantiza a [[1,1],[0,1]] unimodular ⇒ sin torsión.
    mat_b = np.array([[0.01, 0.01], [0.0, 0.01]])
    for label, mat in (("A (interacción densa)", mat_a), ("B (unimodular)", mat_b)):
        rho = np.eye(mat.shape[0], dtype=np.complex128) / float(mat.shape[0])
        st = agent.run_sovereign_governance_cycle(rho, mat, exps_test, policy)
        m = st.visceral_metrics
        print(f"\n=== Escenario {label}: {st.truth_value.name} ({hex(st.truth_value.value)}) ===")
        print(f"  Veredicto motor / ρ admisible : {st.engine_truth_value.name} / {st.p1_admissible}")
        print(f"  Mitigación recomendada        : {st.mitigation_recommended} (certificada)")
        print(f"  Escalada al fiscal            : {st.escalation_to_prosecutor} (hardware={st.hardware_actuated})")
        print(f"  SNF torsión libre / |tors|    : {m.smith_torsion_free} / {m.smith_torsion_order}")
        print(f"  Colusión sospechada / ratio   : {m.phantom_collusion_suspected} / {m.associator_ratio_max:.4f}")
        print(f"  Gromov / Melnikov caos        : {m.is_gromov_violated} / {m.melnikov_chaos_detected}")
        print(f"  Dirac (keV) / RSI depth       : {m.dirac_energy_kev:.4f} / {m.rsi_accepted_depth}")
        print(f"  Estado anti-colusión          : {m.anti_collusion_status}")
        print(f"  Raíz Merkle / Firma           : {st.merkle_root_hash[:16]}... / {st.state_signature[:16]}...")
        print(f"  Firma verificada              : {agent.verify_state_signature(st)}")
    print("\n=== TEST COMPLETADO ===")