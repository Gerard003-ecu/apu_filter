# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPhantomCollusionMitigatorWitnessAgent v14.0.0-PhD                              ║
║ Ubicación: app/wisdom/bool_novikov_phantom_collusion_mitigator_witness_agent.py                     ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-PHANTOM-COLLUSION-WITNESS-AGENT-POINCARE-RSI3-PhD           ║
║ Gobernanza: OODA Trifásica Pasiva QND (①Observe → ②Orient → ③Decide) con Compuertas Φ₁₂ y Φ₂₃       ║
║            sobre el Testigo Anti-Colusiones Fantasma (PASSIVE_SILENT_QND, sin hardware).            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

FASES ANIDADAS Y RELEVO FORMAL (el último método de cada fase inicia la siguiente):
  F1 PhantomCollusionWitnessPhase1Observer
     └ emit_phase1_certificate        ÚLTIMO de F1 → Φ₁₂ → execute_phase2 (PRIMERO de F2).
  F2 PhantomCollusionWitnessPhase2Orientor
     └ emit_phase2_certificate        ÚLTIMO de F2 → Φ₂₃ → execute_phase3 (PRIMERO de F3).
  F3 PhantomCollusionWitnessPhase3Decider
     └ emit_sovereign_state           ÚLTIMO de F3 (estado soberano firmado, pasivo y silencioso).

CAMBIOS RIGUROSOS v13.0 → v14.0 (granulares):
  · Φ₁₂/Φ₂₃: RE-DERIVAN el hash de cadena desde los campos y la firma HMAC (vinculación
    contenido→hash→firma). Alterar cualquier campo se detecta (fail-closed). La v13 no
    verificaba HMAC en absoluto (secret_key sin usar): la firma era decorativa.
  · Φ₁₂ autentica pero ADMITE anomalías (la admisibilidad física de ρ se compone en F3).
    Φ₂₃ autentica y exige el contrato pasivo: Lip_cert<1 y conmutador relativo ≤1e-8.
  · F1: certifica Hermiticidad ‖ρ−ρ†‖, Tr(ρ)=1 (tol 1e-10, no 0.05), PSD, pureza Tr(ρ²)≤1
    y entropía de von Neumann. Espectro sobre la parte Hermítica (eigvalsh sobre no-Hermítica
    es inválido). Un solo timestamp; reloj inyectable; canonicalización .17g (no .6f lossy).
  · F2: PhantomWitnessAuditPolicy (frozen) reenvía TODOS los parámetros del motor v14 por
    palabra clave. Vincula P1 (hash+admisibilidad), input_digest, pasaporte, esquema, veredicto,
    autoprueba, QND, Bures, asociadores y cadena RSI³.
  · F3: veredicto final = motor.meet(⊤ si ρ admisible else ⊥) en Heyting Ω₃. Raíz Merkle
    BINARIA real (RFC-6962). Firma HMAC-P3 del estado. Contrato PASSIVE_SILENT_QND enforced:
    is_hardware_actuated=False SIEMPRE (el testigo no dispara hardware, a diferencia del weaver).
  · Φ_sem: se ELIMINA kv_cache_compression_pct=86.4 (contaminación cruzada: el motor testigo
    no comprime KV; eso pertenece al weaver). Métricas genuinas: fidelidad, Bures, asociadores,
    torsión, profundidad RSI, jerk medido. WACC/liquidez son OBJETIVOS de política declarados.
  · Clave HMAC por entorno; serialización canónica fail-closed; autoprueba metamórfica en init.

LÍMITES DECLARADOS:
  · El testigo es PASIVO y SILENCIOSO: no escribe en hardware ni modela disparo crowbar.
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

# ── Importación del motor espectral testigo (dos rutas de paquete) ─────────────
try:
    from bool_novikov_phantom_collusion_mitigator_witness_engine import (
        BoolNovikovPhantomCollusionMitigatorWitnessEngine,
        HeytingTruthValue,
        PhantomCollusionWitnessSpectrumReport,
    )
except ImportError:  # pragma: no cover - ruta de paquete alternativa
    try:
        from app.wisdom.bool_novikov_phantom_collusion_mitigator_witness_engine import (
            BoolNovikovPhantomCollusionMitigatorWitnessEngine,
            HeytingTruthValue,
            PhantomCollusionWitnessSpectrumReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_phantom_collusion_mitigator_witness_engine."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovPhantomCollusionMitigatorWitnessAgent")

# ── Versiones y dominios HMAC ──────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-BOOL-NOVIKOV-PHANTOM-COLLUSION-WITNESS-AGENT-POINCARE-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
_AGENT_SCHEMA: Final[str] = "witness-agent-cert-v14.0"
_HMAC_KEY_ENV: Final[str] = "APU_PHANTOM_WITNESS_AGENT_HMAC_KEY"
_MIN_KEY_BYTES: Final[int] = 16
_DEV_SECRET_KEY: Final[bytes] = b"APU_WITNESS_AGENT_DEV_ONLY_PHANTOM_KEY_v14.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"COLLUSION-WITNESS-AGENT-P1-ISOMETRY-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"COLLUSION-WITNESS-AGENT-P2-SPECTRUM-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"COLLUSION-WITNESS-AGENT-P3-SOVEREIGN-v14.0|"
_MERKLE_LEAF: Final[bytes] = b"\x00"
_MERKLE_NODE: Final[bytes] = b"\x01"

# ── Tolerancias físicas de Fase 1 ──────────────────────────────────────────────
_TRACE_TOL: Final[float] = 1.0e-10
_HERMITICITY_TOL: Final[float] = 1.0e-10
_PSD_TOL: Final[float] = 1.0e-12
_PURITY_TOL: Final[float] = 1.0e-10

# ── Contrato pasivo QND (debe coincidir con el motor v14) ──────────────────────
_QND_COMM_REL_TOL: Final[float] = 1.0e-8
_QND_BACK_ACTION_DB_MAX: Final[float] = 20.0 * math.log10(1.0 + _QND_COMM_REL_TOL)

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


# ══════════════════════════════════════════════════════════════════════════════
# Contratos: certificados, política, métricas y estado soberano
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PhantomCollusionWitnessPhase1Certificate:
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
class PhantomCollusionWitnessPhase2Certificate:
    """Certificado P2 (Orient): informe del motor testigo vinculado a P1."""

    timestamp: float
    spectrum_report: PhantomCollusionWitnessSpectrumReport
    p1_chain_hash: str
    p1_admissible: bool
    chain_hash: str
    hmac_signature: str
    schema_version: str


@dataclass(frozen=True, eq=False)
class PhantomWitnessAuditPolicy:
    """Política de auditoría: reenvío validado de los parámetros del motor testigo v14."""

    phi_i: Optional[NDArray[np.complex128]] = None
    phi_f: Optional[NDArray[np.complex128]] = None
    sigma_input: Optional[NDArray[np.complex128]] = None
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
        if self.capacity_series is not None:
            if len(self.capacity_series) < 4 or not all(
                math.isfinite(float(x)) for x in self.capacity_series
            ):
                raise ValueError("capacity_series requiere ≥4 muestras finitas.")

    def to_engine_kwargs(self) -> Dict[str, object]:
        """Palabras clave para el motor (nunca posicional; rho_mac_input va aparte)."""
        return {
            "phi_i": self.phi_i,
            "phi_f": self.phi_f,
            "sigma_input": self.sigma_input,
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
class PhantomCollusionWitnessVisceralMetrics:
    """Φ_sem: objetivos de política declarados + medidas genuinas del testigo.
    Nota: NO incluye compresión KV (el motor testigo no comprime; eso es del weaver)."""

    wacc_protection_target_pct: float
    net_liquidity_release_target_pct: float
    qnd_status: str
    fidelity_uhlmann: float
    bures_distance: float
    phantom_collusion_suspected: bool
    smith_torsion_free: bool
    rsi_accepted_depth: int
    capacity_third_difference: Optional[float]
    policy_provenance: str


@dataclass(frozen=True)
class PhantomCollusionWitnessSovereignState:
    """Estado soberano final (Decide), pasivo y silencioso, firmado sobre raíz Merkle."""

    timestamp: float
    truth_value: HeytingTruthValue
    engine_truth_value: HeytingTruthValue
    p1_admissible: bool
    is_passive_silent_qnd: bool
    is_hardware_actuated: bool          # SIEMPRE False: el testigo no dispara hardware
    back_action_db: float
    commutator_relative: float
    merkle_root_hash: str
    state_signature: str
    visceral_metrics: PhantomCollusionWitnessVisceralMetrics
    schema_version: str


# ══════════════════════════════════════════════════════════════════════════════
# Fase 1 — OBSERVE: ingestión e invariantes físicos de ρ (MAC)
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionWitnessPhase1Observer:
    """Fase 1 OODA. Certifica Hermiticidad, traza, PSD, pureza y entropía de ρ."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase1(
        self, rho_mac: NDArray[np.complex128], trace_tolerance: float = _TRACE_TOL
    ) -> PhantomCollusionWitnessPhase1Certificate:
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
    def _p1_chain_fields(cert: PhantomCollusionWitnessPhase1Certificate) -> Tuple[object, ...]:
        """Orden canónico que el hash vincula (idéntico en emisión y compuerta)."""
        return (
            cert.timestamp, cert.tr_rho, cert.is_hermitian, cert.hermiticity_residual,
            cert.is_psd, cert.min_eigenvalue, cert.max_eigenvalue, cert.purity,
            cert.von_neumann_entropy, cert.is_admissible, cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 1 → Φ₁₂ → Fase 2 ─────────────────────────────────
    def emit_phase1_certificate(self, **fields: object) -> PhantomCollusionWitnessPhase1Certificate:
        """ÚLTIMO método de F1. Firma P1 y lo entrega a la Compuerta Homomórfica Φ₁₂."""
        base = dict(fields)
        base["schema_version"] = _AGENT_SCHEMA
        provisional = PhantomCollusionWitnessPhase1Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p1_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, chain)
        return PhantomCollusionWitnessPhase1Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 2 — ORIENT: juzgamiento del motor testigo y vinculación espectral
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionWitnessPhase2Orientor:
    """Fase 2 OODA. Ejecuta el motor testigo v14 y vincula P1 + pasaporte + QND + RSI³."""

    def __init__(
        self,
        secret_key: bytes,
        engine: BoolNovikovPhantomCollusionMitigatorWitnessEngine,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = secret_key
        self._engine = engine
        self._clock = clock

    def execute_phase2(
        self,
        p1_cert: PhantomCollusionWitnessPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        rho_mac: NDArray[np.complex128],
        policy: PhantomWitnessAuditPolicy,
    ) -> PhantomCollusionWitnessPhase2Certificate:
        """PRIMER método de F2. Consume P1 (vía Φ₁₂) y la política; invoca el motor por palabra."""
        if not isinstance(p1_cert, PhantomCollusionWitnessPhase1Certificate):
            raise TypeError("Fase 2 exige un PhantomCollusionWitnessPhase1Certificate.")
        report = self._engine.execute_phantom_collusion_witness_audit(
            apu_cost_matrix, exponents, rho_mac_input=rho_mac, **policy.to_engine_kwargs()
        )
        return self.emit_phase2_certificate(p1_cert, report, float(self._clock()))

    @staticmethod
    def _p2_chain_fields(cert: PhantomCollusionWitnessPhase2Certificate) -> Tuple[object, ...]:
        r = cert.spectrum_report
        return (
            cert.timestamp, cert.p1_chain_hash, cert.p1_admissible,
            r.input_digest, r.passport_hash, r.schema_version,
            int(r.truth_value), bool(r.metamorphic_selftest_passed),
            float(r.banach_contraction_lip),
            bool(r.qnd_report.is_qnd_passive), float(r.qnd_report.commutator_relative),
            bool(r.bures_report.is_isometry_preserved),
            bool(r.associator_report.phantom_collusion_suspected),
            getattr(r, "rsi_certificate_chain", ""),
            cert.schema_version,
        )

    # ── ÚLTIMO método de Fase 2 → Φ₂₃ → Fase 3 ─────────────────────────────────
    def emit_phase2_certificate(
        self,
        p1_cert: PhantomCollusionWitnessPhase1Certificate,
        report: PhantomCollusionWitnessSpectrumReport,
        timestamp: float,
    ) -> PhantomCollusionWitnessPhase2Certificate:
        """ÚLTIMO método de F2. Firma P2 vinculando P1 y el informe del motor testigo."""
        base = {
            "timestamp": timestamp,
            "spectrum_report": report,
            "p1_chain_hash": p1_cert.chain_hash,
            "p1_admissible": p1_cert.is_admissible,
            "schema_version": _AGENT_SCHEMA,
        }
        provisional = PhantomCollusionWitnessPhase2Certificate(
            chain_hash="", hmac_signature="", **base  # type: ignore[arg-type]
        )
        chain = _chain_hash(self._p2_chain_fields(provisional))
        sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, chain)
        return PhantomCollusionWitnessPhase2Certificate(
            chain_hash=chain, hmac_signature=sig, **base  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fase 3 — DECIDE: Heyting, contrato pasivo, Φ_sem y raíz Merkle
# ══════════════════════════════════════════════════════════════════════════════
class PhantomCollusionWitnessPhase3Decider:
    """Fase 3 OODA. Compone el veredicto, verifica el contrato pasivo y firma el estado."""

    def __init__(self, secret_key: bytes, clock: Callable[[], float] = time.time) -> None:
        self._secret_key = secret_key
        self._clock = clock

    def execute_phase3(
        self, p2_cert: PhantomCollusionWitnessPhase2Certificate
    ) -> PhantomCollusionWitnessSovereignState:
        """PRIMER método de F3. Consume P2 (vía Φ₂₃) y adjudica sobre Heyting Ω₃."""
        if not isinstance(p2_cert, PhantomCollusionWitnessPhase2Certificate):
            raise TypeError("Fase 3 exige un PhantomCollusionWitnessPhase2Certificate.")
        report = p2_cert.spectrum_report
        engine_truth = report.truth_value
        p1_verdict = (
            HeytingTruthValue.COHERENT if p2_cert.p1_admissible else HeytingTruthValue.VETOED
        )
        final_truth = engine_truth.meet(p1_verdict)
        comm_rel = float(report.qnd_report.commutator_relative)
        # Contrato PASSIVE_SILENT_QND: sin hardware, conmutador relativo dentro de tolerancia.
        is_passive = bool(
            report.qnd_report.is_qnd_passive and comm_rel <= _QND_COMM_REL_TOL * (1.0 + 1e-6)
        )
        metrics = self._visceral_metrics(report, is_passive)
        return self.emit_sovereign_state(
            timestamp=float(self._clock()),
            final_truth=final_truth,
            engine_truth=engine_truth,
            p1_admissible=p2_cert.p1_admissible,
            is_passive=is_passive,
            back_action_db=float(report.qnd_report.back_action_db),
            comm_rel=comm_rel,
            p2_cert=p2_cert,
            metrics=metrics,
        )

    @staticmethod
    def _visceral_metrics(
        report: PhantomCollusionWitnessSpectrumReport, is_passive: bool
    ) -> PhantomCollusionWitnessVisceralMetrics:
        """Φ_sem: objetivos declarados + medidas genuinas del testigo (sin KV-cache)."""
        qnd_status = (
            f"PASSIVE_SILENT_QND_BACKACTION={report.qnd_report.back_action_db:.6e}dB"
            if is_passive else "QND_CONTRACT_BREACHED"
        )
        return PhantomCollusionWitnessVisceralMetrics(
            wacc_protection_target_pct=_WACC_PROTECTION_TARGET_PCT,
            net_liquidity_release_target_pct=_NET_LIQUIDITY_TARGET_PCT,
            qnd_status=qnd_status,
            fidelity_uhlmann=float(report.bures_report.fidelity_uhlmann),
            bures_distance=float(report.bures_report.bures_distance),
            phantom_collusion_suspected=bool(report.associator_report.phantom_collusion_suspected),
            smith_torsion_free=bool(report.smith_torsion_free),
            rsi_accepted_depth=int(report.rsi_accepted_depth),
            capacity_third_difference=getattr(report, "capacity_third_difference", None),
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
        is_passive: bool,
        back_action_db: float,
        comm_rel: float,
        p2_cert: PhantomCollusionWitnessPhase2Certificate,
        metrics: PhantomCollusionWitnessVisceralMetrics,
    ) -> PhantomCollusionWitnessSovereignState:
        """ÚLTIMO método de F3. Raíz Merkle binaria + firma HMAC-P3. is_hardware_actuated=False."""
        report = p2_cert.spectrum_report
        leaves = [
            p2_cert.p1_chain_hash,
            p2_cert.chain_hash,
            report.passport_hash,
            report.input_digest,
            f"verdict:{int(final_truth)}",
            f"passive:{int(is_passive)}:hardware:0",
            _chain_hash((
                metrics.wacc_protection_target_pct,
                metrics.net_liquidity_release_target_pct,
                metrics.fidelity_uhlmann,
                metrics.bures_distance,
                metrics.phantom_collusion_suspected,
                metrics.smith_torsion_free,
                metrics.rsi_accepted_depth,
                metrics.capacity_third_difference,
                metrics.qnd_status,
            )),
            getattr(report, "rsi_certificate_chain", ""),
        ]
        merkle_root = _merkle_root(leaves)
        core = (
            timestamp, int(final_truth), int(engine_truth), p1_admissible,
            is_passive, False, back_action_db, comm_rel, merkle_root, _AGENT_SCHEMA,
        )
        state_sig = _sign(self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core))
        return PhantomCollusionWitnessSovereignState(
            timestamp=timestamp,
            truth_value=final_truth,
            engine_truth_value=engine_truth,
            p1_admissible=p1_admissible,
            is_passive_silent_qnd=is_passive,
            is_hardware_actuated=False,
            back_action_db=back_action_db,
            commutator_relative=comm_rel,
            merkle_root_hash=merkle_root,
            state_signature=state_sig,
            visceral_metrics=metrics,
            schema_version=_AGENT_SCHEMA,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Orquestador soberano: ciclo OODA con compuertas homomórficas Φ₁₂ y Φ₂₃
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPhantomCollusionMitigatorWitnessAgent:
    """Soberano Agéntico Testigo de Mitigación de Colusiones Fantasma (pasivo, silencioso)."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        engine: Optional[BoolNovikovPhantomCollusionMitigatorWitnessEngine] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret_key = _resolve_secret_key(secret_key)
        self._engine = engine or BoolNovikovPhantomCollusionMitigatorWitnessEngine(
            secret_key=self._secret_key
        )
        self._clock = clock
        self._observer = PhantomCollusionWitnessPhase1Observer(self._secret_key, clock)
        self._orientor = PhantomCollusionWitnessPhase2Orientor(self._secret_key, self._engine, clock)
        self._decider = PhantomCollusionWitnessPhase3Decider(self._secret_key, clock)
        if not self._metamorphic_selftest():
            raise RuntimeError("Autoprueba metamórfica del agente testigo fallida (fail-closed).")
        logger.info("%s v%s iniciado correctamente.", self.__class__.__name__, _AGENT_VERSION)

    # ── Compuerta Homomórfica Φ₁₂ (re-deriva contenido→hash→firma) ─────────────
    def continue_to_phase2_gate(self, p1: PhantomCollusionWitnessPhase1Certificate) -> None:
        """Φ₁₂: autentica P1 re-derivando hash y firma. ADMITE anomalías para auditoría
        pasiva (la admisibilidad física se compone en Fase 3, no se veta aquí)."""
        if p1.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₁₂: esquema de certificado P1 no reconocido.")
        expected_chain = _chain_hash(PhantomCollusionWitnessPhase1Observer._p1_chain_fields(p1))
        if not hmac.compare_digest(expected_chain, p1.chain_hash):
            raise ValueError("Φ₁₂: el hash de cadena P1 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P1, expected_chain)
        if not hmac.compare_digest(expected_sig, p1.hmac_signature):
            raise ValueError("Φ₁₂: falla de autenticidad HMAC en P1.")

    # ── Compuerta Homomórfica Φ₂₃ (re-deriva + contrato pasivo de Banach/QND) ──
    def continue_to_phase3_gate(self, p2: PhantomCollusionWitnessPhase2Certificate) -> None:
        """Φ₂₃: autentica P2 y exige el contrato pasivo: Lip_cert<1 y conmutador relativo
        ≤1e-8. Los juicios espectrales (torsión, Bures, asociadores) se componen en F3."""
        if p2.schema_version != _AGENT_SCHEMA:
            raise ValueError("Φ₂₃: esquema de certificado P2 no reconocido.")
        expected_chain = _chain_hash(PhantomCollusionWitnessPhase2Orientor._p2_chain_fields(p2))
        if not hmac.compare_digest(expected_chain, p2.chain_hash):
            raise ValueError("Φ₂₃: el hash de cadena P2 no vincula el contenido (tamper).")
        expected_sig = _sign(self._secret_key, _HMAC_DOMAIN_P2, expected_chain)
        if not hmac.compare_digest(expected_sig, p2.hmac_signature):
            raise ValueError("Φ₂₃: falla de autenticidad HMAC en P2.")
        rep = p2.spectrum_report
        if not (rep.banach_contraction_lip < 1.0):
            raise ValueError("Φ₂₃: violación de estabilidad (Lip_cert ≥ 1.0).")
        if not (rep.qnd_report.is_qnd_passive
                and rep.qnd_report.commutator_relative <= _QND_COMM_REL_TOL * (1.0 + 1e-6)):
            raise ValueError("Φ₂₃: violación del contrato PASSIVE_SILENT_QND (back-action).")

    def verify_state_signature(self, state: PhantomCollusionWitnessSovereignState) -> bool:
        """Verificación independiente de la firma del estado soberano."""
        core = (
            state.timestamp, int(state.truth_value), int(state.engine_truth_value),
            state.p1_admissible, state.is_passive_silent_qnd, state.is_hardware_actuated,
            state.back_action_db, state.commutator_relative, state.merkle_root_hash,
            state.schema_version,
        )
        expected = _sign(self._secret_key, _HMAC_DOMAIN_P3, _chain_hash(core))
        return hmac.compare_digest(expected, state.state_signature)

    # ── Ciclo OODA completo ────────────────────────────────────────────────────
    def run_sovereign_governance_cycle(
        self,
        rho_mac: NDArray[np.complex128],
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        policy: Optional[PhantomWitnessAuditPolicy] = None,
    ) -> PhantomCollusionWitnessSovereignState:
        """Ejecuta Observe→Φ₁₂→Orient→Φ₂₃→Decide sobre el fibrado dual Boole-Novikov."""
        eff_policy = policy if policy is not None else PhantomWitnessAuditPolicy()
        p1 = self._observer.execute_phase1(rho_mac)
        self.continue_to_phase2_gate(p1)                       # Φ₁₂
        p2 = self._orientor.execute_phase2(p1, apu_cost_matrix, exponents, rho_mac, eff_policy)
        self.continue_to_phase3_gate(p2)                       # Φ₂₃
        state = self._decider.execute_phase3(p2)
        logger.info("Ciclo OODA Anti-Colusión Witness completado. Veredicto: %s", state.truth_value.name)
        return state

    # ── Autoprueba metamórfica del agente (nivel meta-meta de gobernanza) ──────
    def _metamorphic_selftest(self) -> bool:
        """Verifica vinculación de compuertas, Merkle, composición Heyting y contrato pasivo."""
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
            # Coherencia del umbral dB derivado del conmutador relativo.
            if not (_QND_BACK_ACTION_DB_MAX < 1.0e-6):
                return False
            # Ciclo rápido (sin Lyapunov) + detección de tamper en Φ₁₂ y Φ₂₃.
            fast = PhantomWitnessAuditPolicy(poincare_periods=0)
            rho = np.eye(2, dtype=np.complex128) / 2.0
            mat = np.array([[2.0, 0.1], [0.1, 2.5]])
            exps = np.array([0.15, 0.22, 0.40])
            p1 = self._observer.execute_phase1(rho)
            self.continue_to_phase2_gate(p1)
            tampered_p1 = replace(p1, tr_rho=0.5)   # contenido alterado, hash intacto
            try:
                self.continue_to_phase2_gate(tampered_p1)
                return False
            except ValueError:
                pass
            p2 = self._orientor.execute_phase2(p1, mat, exps, rho, fast)
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
            if state.is_hardware_actuated:        # el testigo NUNCA dispara hardware
                return False
            return True
        except Exception:
            logger.exception("Autoprueba metamórfica del agente testigo: excepción (fail-closed).")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Demo autónomo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPhantomCollusionMitigatorWitnessAgent v14.0 ===")
    rho_test = np.eye(2, dtype=np.complex128) / 2.0
    matrix_test = np.array([[2.0, 0.1], [0.1, 2.5]], dtype=np.float64)
    exps_test = np.array([0.15, 0.22, 0.40], dtype=np.float64)
    agent = BoolNovikovPhantomCollusionMitigatorWitnessAgent(secret_key=b"demo-key-0123456789abcdef")
    policy = PhantomWitnessAuditPolicy(capacity_series=(1.0, 2.0, 4.0, 8.0))
    st = agent.run_sovereign_governance_cycle(rho_test, matrix_test, exps_test, policy)
    print(f"Veredicto Final Heyting Ω₃  : {st.truth_value.name} ({hex(st.truth_value.value)})")
    print(f"Veredicto motor / ρ admisible: {st.engine_truth_value.name} / {st.p1_admissible}")
    print(f"PASSIVE_SILENT_QND          : {st.is_passive_silent_qnd} (hardware={st.is_hardware_actuated})")
    print(f"Back-Action QND (dB)        : {st.back_action_db:.6e} dB (conmutador rel={st.commutator_relative:.3e})")
    print(f"Fidelidad Uhlmann / Bures   : {st.visceral_metrics.fidelity_uhlmann:.6f} / {st.visceral_metrics.bures_distance:.6f}")
    print(f"Colusión fantasma sospechada: {st.visceral_metrics.phantom_collusion_suspected}")
    print(f"Torsión libre / RSI depth   : {st.visceral_metrics.smith_torsion_free} / {st.visceral_metrics.rsi_accepted_depth}")
    print(f"Raíz Merkle / Firma         : {st.merkle_root_hash[:16]}... / {st.state_signature[:16]}...")
    print(f"Firma verificada            : {agent.verify_state_signature(st)}")
    print(f"Objetivos política WACC/Liq : {st.visceral_metrics.wacc_protection_target_pct}% / "
          f"{st.visceral_metrics.net_liquidity_release_target_pct}% ({st.visceral_metrics.policy_provenance})")
    print("=== TEST COMPLETADO ===")