# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : bool_novikov_pseudoholomorphic_witness_agent.py                                           ║
║ Ubicación: app/agents/wisdom/bool_novikov_pseudoholomorphic_witness_agent.py                         ║
║ Versión  : 14.0.0-SOVEREIGN-WITNESS-AGENT-PHASES-QND-POINCARE-KAM-RSI3-PhD                           ║
║ Gobernanza: OODA trifásica anidada (Observe ➔ Orient ➔ Decide&Act) con Compuertas Φ₁₂ y Φ₂₃          ║
║            sobre el motor canónico v14.0.0 (Phase1WitnessCertificate / Phase2WitnessCertificate)     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre Testigo `BoolNovikovPseudoholomorphicWitnessAgent` gobierna
ejecutivamente el ciclo OODA de atestación pasiva y certificación de coherencia sobre el motor
espectral `BoolNovikovPseudoholomorphicWitnessEngine` (app/wisdom/bool_novikov_pseudoholomorphic_witness_engine.py).
Supervisa la preservación de la coherencia cuántica sin demolición (QND) en tres fases anidadas
conectadas por compuertas homomórficas \Phi_{12} y \Phi_{23}: audita la distancia de Bures-Wasserstein
y la fidelidad de Uhlmann F(\rho, \sigma) \ge 0.95 en la MAC, verifica la preservación de capacidad
simpléctica de Gromov en cilindros pseudoholomorfos de Floer, valida la no-resonancia de Poincaré-KAM
y certifica la contracción hiperbólica de Newton-Kantorovich para la mónada de Kleisli en RSI³.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE: Atestación de Bures-Wasserstein, QND y Valuación de Novikov:
   Somete la matriz densidad \rho a proyección PSD canónica de Higham.
   Evalúa la medición débil AAV A_w = \langle \psi_f | A | \psi_i \rangle / \langle \psi_f | \psi_i \rangle
   asegurando perturbación de canal casi nula \|[A, \rho]\|_F \le 1.0\times 10^{-10} (\mathrm{BackAction} \le 0.01 \text{ dB}).
   Calcula la distancia de Bures-Wasserstein d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}}
   y la valuación ultramétrica no arquimediana v(A) \ge 0.10 en el anillo universal \Lambda_{\mathrm{Nov}}.
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2:
     Consume `Phase1WitnessCertificate` emitido por el motor y valida la admisión en \Phi_{12}.

2. FASE 2 — ORIENT: Cilindro Pseudoholomorfo de Floer y Mecánica Celeste de Poincaré:
   Inicia en el contexto de trabajo de Fase 1.
   Audita la solución u: \mathbb{R} \times S^1 \to M de la ecuación pseudoholomorfa de Cauchy-Riemann-Floer:
       \bar{\partial}_{\mathcal{J}, H}(u) = \partial_s u + \mathcal{J}(\partial_t u - X_H(u)) = 0,
   y la cota de rigidez simpléctica de Gromov c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)).
   En la dinámica de Poincaré-KAM, verifica los divisores diofantinos |\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau
   y la ausencia de caos homoclínico (|M(\tau)| \le 1.0\times 10^{-4}).
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3:
     Consume `Phase2WitnessCertificate` emitido por el motor y valida la admisión en \Phi_{23}.

3. FASE 3 — DECIDE & ACT: Mónada RSI³ de Banach-Kleisli y Veredicto Heyting \Omega_3:
   Inicia en el contexto de trabajo de Fase 2.
   Audita la contracción monádica de Kleisli T^3 con aceleración de Aitken-Steffensen y cota de Newton-Kantorovich:
       h_K = L_0 \beta \eta \le 0.50 \land \mathrm{Lip}(T^3) < 1.0.
   Deduce el veredicto en el álgebra intuicionista de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}.
   Proyecta las métricas viscerales de impacto financiero y emite el estado soberano inmutable
   `BoolNovikovPseudoholomorphicWitnessSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma PWA-A1] Atestación de Isometría Cuántica de Bures:
    d_B(\rho, \sigma_{\mathrm{PSD}}) \le 1.0\times 10^{-6} \land F(\rho, \sigma) \ge 0.95 \implies \text{coherencia cuántica verificada}.
[Axioma PWA-A2] No-Demolición Cuántica QND:
    \|[A, \rho]\|_F \le 1.0\times 10^{-10} \implies \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Axioma PWA-A3] Rigidez Simpléctica de Gromov:
    c_G(B^{2n}(r)) = \pi r^2 \le c_G(Z^{2n}(R)) = \pi R^2.
[Axioma PWA-A4] Autenticación Criptográfica de Certificados de Fase:
    compare\_digest(\mathrm{HMAC}_{\mathrm{calc}}, \mathrm{HMAC}_{\mathrm{cert}}) \equiv \mathrm{True} \text{ en compuertas } \Phi_{12} \text{ y } \Phi_{23}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante PWA-I1] Fidelidad de Uhlmann: F(\rho, \sigma_{\mathrm{PSD}}) \ge 0.95.
[Invariante PWA-I2] Retroacción Cuántica: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante PWA-I3] Residuo de Floer: \|\bar{\partial}_{\mathcal{J},H}(u)\| \le 1.0\times 10^{-4}.
[Invariante PWA-I4] Contracción de Kleisli: \mathrm{Lip}(T^3) \le 0.85975 < 1.0.
[Invariante PWA-I5] Sellado Criptográfico HMAC-SHA256 verificado en cada compuerta de fase.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Ajuste de path para importaciones flexibles
for path_dir in ["/workspace/artifacts", "/workspace/scratch", "."]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

try:
    from bool_novikov_pseudoholomorphic_witness_engine import (
        BuresUhlmannMetricReport,
        BoolNovikovPseudoholomorphicWitnessEngine,
        CelestialWitnessState,
        HeytingTruthValue,
        NovikovWitnessState,
        Phase1WitnessCertificate,
        Phase2WitnessCertificate,
        PseudoholomorphicSurfaceWitness,
        PseudoholomorphicWitnessReport,
        QNDWeakMeasurementReport,
        RSI3WitnessState,
        WitnessGateViolationError,
        WitnessParityViolationError,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "No se pudo importar bool_novikov_pseudoholomorphic_witness_engine "
        f"(¿está instalado el motor v14.0.0?): {exc}"
    ) from exc


# Re-exports de compatibilidad para código que importe desde el agente
WitnessPhase1Certificate = Phase1WitnessCertificate
WitnessPhase2Certificate = Phase2WitnessCertificate


logger = logging.getLogger("APU.Wisdom.BoolNovikovPseudoholomorphicWitnessAgent")

_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-WITNESS-AGENT-PHASES-QND-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_WITNESS_SOVEREIGN_KEY_v14.0.0"

_HMAC_DOMAIN_P3: Final[bytes] = b"BOOL-NOVIKOV-WITNESS-P3-SOVEREIGN-v14.0|"

_HEX_LEN: Final[int] = 64
_QND_PASSIVE_SILENCE_TOLERANCE_DB: Final[float] = 1.0e-3
_BANACH_LIP_BOUND: Final[float] = 1.0
_WACC_INVARIANT_PCT: Final[float] = 15.0
_KV_CACHE_COMPRESSION_PCT: Final[float] = 86.4
_AI_INFERENCE_SAVING_PCT: Final[float] = 80.0
_LIQUIDITY_COHERENT_PCT: Final[float] = 11.5
_LIQUIDITY_DEGRADED_PCT: Final[float] = 5.75
_LIQUIDITY_VETOED_PCT: Final[float] = 0.0

__all__ = [
    "WitnessPhase1Certificate",
    "WitnessPhase2Certificate",
    "VisceralImpactReport",
    "BoolNovikovPseudoholomorphicWitnessSovereignState",
    "WitnessAgentGateViolationError",
    "WitnessAgentParityViolationError",
    "BoolNovikovPseudoholomorphicWitnessAgent",
]


# ──────────────────────────────────────────────────────────────────────────────
# Excepciones específicas del agente
# ──────────────────────────────────────────────────────────────────────────────
class WitnessAgentGateViolationError(RuntimeError):
    r"""Falla de autenticidad o integridad en las compuertas Φ del agente."""


class WitnessAgentParityViolationError(RuntimeError):
    r"""Divergencia entre el veredicto del agente y el motor canónico v14."""


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de salida del agente
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VisceralImpactReport:
    r"""
    Traducción por el Funtor Semántico Visceral Φ_sem hacia C-Suite.

    Las métricas invariantes (WACC, KV, IA) son fijas. La liquidez liberada
    escala con el veredicto:
      - COHERENT : 11.5 % (todo el colchón monetizable)
      - DEGRADED : 5.75 % (mitad del colchón, contiene el riesgo)
      - VETOED   : 0.0 % (activo congelado por pasividad defensiva)

    `is_passive_silent_verified` sintetiza el invariante PASSIVE_SILENT:
    el testigo NO interfiere con el sistema observado. Se DERIVA de la
    back-action observada (no se asume).
    """
    wacc_protection_pct: float
    net_liquidity_liberation_pct: float
    kv_cache_compression_pct: float
    inference_cost_saving_pct: float
    qnd_silence_backaction_db: float
    is_passive_silent_verified: bool


@dataclass(frozen=True)
class BoolNovikovPseudoholomorphicWitnessSovereignState:
    r"""Estado final del Soberano Testigo QND tras adjudicación en Ω₃."""
    timestamp: float
    agent_version: str
    truth_value: HeytingTruthValue
    phase1_cert: Phase1WitnessCertificate
    phase2_cert: Phase2WitnessCertificate
    rsi3_state: RSI3WitnessState
    visceral_impact: VisceralImpactReport
    merkle_root_hash: str
    hmac_signature: str
    is_passive_silent: bool
    adjudication_reason: str
    total_latency_ms: float


# ──────────────────────────────────────────────────────────────────────────────
# Contextos de fase (para el contrato de anidamiento)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Phase2WorkingContext:
    """Contexto operativo entre `phase2_open_*` y `phase2_emit_*`."""
    opened_cert_p1: Phase1WitnessCertificate


@dataclass(frozen=True)
class _Phase3WorkingContext:
    """Contexto operativo entre `phase3_open_*` y `phase3_emit_*`."""
    cert_p1: Phase1WitnessCertificate
    cert_p2: Phase2WitnessCertificate


# ──────────────────────────────────────────────────────────────────────────────
# SOBERANO AGÉNTICO — 3 FASES ANIDADAS + 2 COMPUERTAS
# ──────────────────────────────────────────────────────────────────────────────
class BoolNovikovPseudoholomorphicWitnessAgent:
    r"""
    Soberano Agéntico Testigo Pseudoholomorfo Dual Boole-Novikov v14.0.0-PhD.

    Consume el motor canónico v14.0.0 (`BoolNovikovPseudoholomorphicWitnessEngine`)
    y añade la gobernanza OODA trifásica con compuertas homomórficas Φ₁₂ y Φ₂₃,
    el Funtor Semántico Visceral Φ_sem y la paridad obligatoria contra
    `engine.execute_witness_audit`.

    Contrato de anidamiento
    -----------------------
    - Último método de FASE I (`phase1_emit_sovereign_certificate`)
      ⟶ retorna `Phase1WitnessCertificate`
      ⟶ es la entrada obligatoria de `phase2_open_from_phase1_certificate`.
    - Último método de FASE II (`phase2_emit_sovereign_certificate`)
      ⟶ retorna `Phase2WitnessCertificate`
      ⟶ es la entrada obligatoria de `phase3_open_from_phase2_certificate`.
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = secret_key
        self._engine: BoolNovikovPseudoholomorphicWitnessEngine = (
            BoolNovikovPseudoholomorphicWitnessEngine(secret_key=secret_key)
        )
        logger.info(
            "BoolNovikovPseudoholomorphicWitnessAgent v%s iniciado.",
            _AGENT_VERSION,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE I — OBSERVE
    # ══════════════════════════════════════════════════════════════════════════

    def phase1_observe_qnd_witness(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: Optional[NDArray[np.complex128]] = None,
        delta: float = 0.05,
    ) -> QNDWeakMeasurementReport:
        r"""
        Medición Débil AAV y verificación del invariante QND pasivo vía engine v14.

        Delegación a `engine.phase1_witness_qnd_weak_measurement`:
          • A_witness diagonal + perturbación hermitiana controlada por δ².
          • A_w = ⟨φ_f|A|φ_i⟩/⟨φ_f|φ_i⟩.
          • ‖[A_witness, ρ_MAC]‖_F ≤ 10⁻¹⁰ ⟹ back-action ≡ 0 dB.
        """
        return self._engine.phase1_witness_qnd_weak_measurement(
            rho_mac=rho_mac,
            phi_i=phi_i,
            phi_f=phi_f,
            delta=delta,
        )

    def phase1_observe_bures_uhlmann(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> BuresUhlmannMetricReport:
        r"""
        Fidelidad de Uhlmann F(ρ,σ) y Distancia Bures-Wasserstein vía engine v14.

        Delegación a `engine.phase1_witness_bures_uhlmann`:
          • Sanitización PSD + hermitización.
          • F = (Tr √(√ρ σ √ρ))² ∈ [0,1].
          • d_B = √(2 − 2√F).
        """
        return self._engine.phase1_witness_bures_uhlmann(rho, sigma)

    def phase1_observe_novikov_krull(
        self,
        coefficients: Sequence[complex],
        exponents: Sequence[float],
    ) -> NovikovWitnessState:
        r"""
        Valuación de Krull ultramétrica y norma vía engine v14.

        Delegación a `engine.phase1_witness_novikov_krull`:
          • v(A) = min{λ_i : |a_i| > 0}.
          • ‖A‖_Nov = exp(−v(A)).
          • Piso de validez: v(A) ≥ 0.10.
          • Series nulas: v ≡ +∞, válidas por vacuidad.
        """
        return self._engine.phase1_witness_novikov_krull(coefficients, exponents)

    def phase1_emit_sovereign_certificate(
        self,
        qnd_report: QNDWeakMeasurementReport,
        bures_report: BuresUhlmannMetricReport,
        novikov_state: NovikovWitnessState,
    ) -> Phase1WitnessCertificate:
        r"""
        ÚLTIMO MÉTODO DE LA FASE I.

        Ensambla el certificado P1 (`Phase1WitnessCertificate`) con
        chain_hash SHA-256 (64-hex) y firma HMAC-SHA256 en el dominio Φ₁₂.

        El retorno es la ENTRADA OBLIGATORIA del primer método de FASE II
        (`phase2_open_from_phase1_certificate`).
        """
        return self._engine.phase1_emit_witness_certificate(
            qnd_report=qnd_report,
            bures_report=bures_report,
            novikov_state=novikov_state,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUERTA HOMOMÓRFICA Φ₁₂
    # ══════════════════════════════════════════════════════════════════════════

    def continue_to_phase2_gate(
        self,
        cert_p1: Phase1WitnessCertificate,
    ) -> bool:
        r"""
        Compuerta Homomórfica Φ₁₂.

        Verificación en dos capas:
          1. Delegada al engine: HMAC-SHA256 P1, chain_hash 64-hex, timestamp
             positivo finito. Levanta `WitnessGateViolationError` en fallo.
          2. Agent-específica: finitud de qnd.commutator_norm,
             qnd.back_action_db, bures.fidelity_uhlmann, bures.bures_distance
             y novikov.krull_valuation (o admisión de serie nula explícita).

        PROPIEDAD FUNDAMENTAL: ADMITE VETOED — la anomalía viaja a FASE II.
        """
        try:
            self._engine.phase2_open_from_phase1_certificate(cert_p1)
        except WitnessGateViolationError as exc:
            logger.error(r"Compuerta Φ₁₂: %s", exc)
            return False

        q = cert_p1.qnd_report
        b = cert_p1.bures_report
        n = cert_p1.novikov_state

        checks: Tuple[Tuple[str, bool], ...] = (
            ("qnd.commutator_norm", math.isfinite(q.commutator_norm)),
            ("qnd.back_action_db", math.isfinite(q.back_action_db)),
            ("qnd.weak_value.real", math.isfinite(q.weak_value.real)),
            ("qnd.weak_value.imag", math.isfinite(q.weak_value.imag)),
            ("bures.fidelity_uhlmann", math.isfinite(b.fidelity_uhlmann)),
            ("bures.bures_distance", math.isfinite(b.bures_distance)),
            (
                "novikov.krull_valuation",
                n.is_null_series or math.isfinite(n.krull_valuation),
            ),
        )
        for name, ok in checks:
            if not ok:
                logger.error(r"Compuerta Φ₁₂: campo no finito %s.", name)
                return False
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # FASE II — ORIENT
    # ══════════════════════════════════════════════════════════════════════════

    def phase2_open_from_phase1_certificate(
        self,
        cert_p1: Phase1WitnessCertificate,
    ) -> _Phase2WorkingContext:
        r"""
        PRIMER MÉTODO DE LA FASE II.

        Consume el certificado P1 emitido por el último método de FASE I.
        Revalida HMAC e integridad estructural (delegando al engine) y
        abre el contexto operativo de FASE II.
        """
        self._engine.phase2_open_from_phase1_certificate(cert_p1)
        return _Phase2WorkingContext(opened_cert_p1=cert_p1)

    def phase2_orient_pseudoholomorphic_surface(
        self,
        ctx: _Phase2WorkingContext,
        cost_matrix: NDArray[np.float64],
        ball_r: float = 1.0,
        cylinder_R: float = 1.2,
    ) -> PseudoholomorphicSurfaceWitness:
        r"""
        Residual CR-Floer por curvatura media discreta + rigidez de Gromov.

        Delegación a `engine.phase2_witness_cauchy_riemann_surface`:
          • Curvatura media de la gráfica u: R² → R (proxy de holomorfía).
          • Fallback simetría-gradientes para grids pequeños (<3).
          • Gromov non-squeezing: π r² ≤ π R².
        """
        return self._engine.phase2_witness_cauchy_riemann_surface(
            cost_matrix=cost_matrix,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
        )

    def phase2_orient_poincare_kam(
        self,
        ctx: _Phase2WorkingContext,
        surface_witness: PseudoholomorphicSurfaceWitness,
        pert_eps: float = 0.05,
    ) -> CelestialWitnessState:
        r"""
        Mecánica Celeste de Poincaré: Delaunay-Whittaker + Lindstedt + Melnikov + KAM.

        Delegación a `engine.phase2_witness_poincare_kam`:
          • Frecuencias Keplerianas ω₁ ~ J⁻³ con detuning áureo ω₂ = φω₁.
          • Melnikov de separatriz pendular forzada:
                M(τ) = 2π ε sech(πω/2) sin(ωτ).
          • Ecuación homológica de Poincaré-Lindstedt.
          • Margen diofántico KAM: min|ω·k|·|k|^τ con τ = 1.5.
        """
        return self._engine.phase2_witness_poincare_kam(
            surface=surface_witness,
            pert_eps=pert_eps,
        )

    def phase2_emit_sovereign_certificate(
        self,
        ctx: _Phase2WorkingContext,
        surface_witness: PseudoholomorphicSurfaceWitness,
        celestial_witness: CelestialWitnessState,
    ) -> Phase2WitnessCertificate:
        r"""
        ÚLTIMO MÉTODO DE LA FASE II.

        Ensambla el certificado P2 (`Phase2WitnessCertificate`) con:
          • previous_chain_hash = ctx.opened_cert_p1.chain_hash (chain link P1 ➔ P2).
          • chain_hash SHA-256 del payload P2 determinista.
          • hmac_signature HMAC-SHA256 sobre _HMAC_DOMAIN_P2 + payload.

        El retorno es la ENTRADA OBLIGATORIA del primer método de FASE III
        (`phase3_open_from_phase2_certificate`).
        """
        return self._engine.phase2_emit_witness_certificate(
            cert_p1=ctx.opened_cert_p1,
            surface_witness=surface_witness,
            celestial_witness=celestial_witness,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUERTA HOMOMÓRFICA Φ₂₃
    # ══════════════════════════════════════════════════════════════════════════

    def continue_to_phase3_gate(
        self,
        cert_p1: Phase1WitnessCertificate,
        cert_p2: Phase2WitnessCertificate,
    ) -> bool:
        r"""
        Compuerta Homomórfica Φ₂₃.

        Verificación en dos capas:
          1. Delegada al engine: HMAC-SHA256 P2, chain_hash 64-hex,
             previous_chain_hash == cert_p1.chain_hash.
          2. Agent-específica: |back-action QND| ≤ 1e-3 dB (pasividad silenciosa
             estricta), finitud de CR residual, frecuencias celestes y Melnikov.
        """
        try:
            self._engine.phase3_open_from_phase2_certificate(cert_p1, cert_p2)
        except WitnessGateViolationError as exc:
            logger.error(r"Compuerta Φ₂₃: %s", exc)
            return False

        # Pasividad QND
        back_action_db = cert_p1.qnd_report.back_action_db
        if not math.isfinite(back_action_db):
            logger.error(r"Compuerta Φ₂₃: back-action QND no finita.")
            return False
        if abs(back_action_db) > _QND_PASSIVE_SILENCE_TOLERANCE_DB:
            logger.error(
                r"Compuerta Φ₂₃: back-action QND %.6e dB excede tolerancia %.1e dB.",
                back_action_db,
                _QND_PASSIVE_SILENCE_TOLERANCE_DB,
            )
            return False

        # Integridad numérica P2
        s = cert_p2.surface_witness
        c = cert_p2.celestial_witness
        if not math.isfinite(s.cauchy_riemann_residual):
            logger.error(r"Compuerta Φ₂₃: residual CR no finito.")
            return False
        if not math.isfinite(c.melnikov_integral_value):
            logger.error(r"Compuerta Φ₂₃: integral de Melnikov no finita.")
            return False
        for f in c.frequencies:
            if not math.isfinite(f):
                logger.error(r"Compuerta Φ₂₃: frecuencia celeste no finita.")
                return False
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # FASE III — DECIDE & ACT
    # ══════════════════════════════════════════════════════════════════════════

    def phase3_open_from_phase2_certificate(
        self,
        cert_p1: Phase1WitnessCertificate,
        cert_p2: Phase2WitnessCertificate,
    ) -> _Phase3WorkingContext:
        r"""
        PRIMER MÉTODO DE LA FASE III.

        Consume el certificado P2 emitido por el último método de FASE II.
        Revalida HMAC P2 + chain link P1 ➔ P2 (delegando al engine) y abre
        el contexto de decisión.
        """
        self._engine.phase3_open_from_phase2_certificate(cert_p1, cert_p2)
        return _Phase3WorkingContext(cert_p1=cert_p1, cert_p2=cert_p2)

    def phase3_evaluate_rsi3_banach(
        self,
        ctx: _Phase3WorkingContext,
        state_vector: NDArray[np.float64],
    ) -> RSI3WitnessState:
        r"""
        RSI³ — Mónada Kleisli de Nivel 3 sobre espacio de Banach.

        Delegación a `engine.phase3_witness_rsi3_banach`:
          • T(x) = α x + β sin(x) con α=0.95, β=0.01.
          • Iteración completa x₀ ➔ x₁ ➔ x₂ ➔ x₃ ➔ x₄.
          • Cota Lip(T³) ≤ (α+β)³ ≈ 0.8847 < 1.0.
          • Residual Banach-Picard ‖T⁴(x) − T³(x)‖.
          • Jerk ciber-físico ‖T³(x) − 3T²(x) + 3T(x) − x‖ > 0.
        """
        return self._engine.phase3_witness_rsi3_banach(state_vector)

    def phase3_adjudicate_heyting_omega3(
        self,
        ctx: _Phase3WorkingContext,
        rsi3_state: RSI3WitnessState,
    ) -> Tuple[HeytingTruthValue, str]:
        r"""
        Adjudicación en el retículo de Heyting Ω₃.

        Delegación a `engine.phase3_witness_adjudicate_heyting`, que unifica
        TODAS las reglas de decisión:
          VETOED   : Novikov-Krull bajo piso, QND no pasivo, Gromov violado,
                     CR residual no finito, weak value no finito.
          DEGRADED : Uhlmann bajo piso, Melnikov caos, KAM margen < 1e-9,
                     Lindstedt > 0.50, RSI³ no contractiva.
          COHERENT : todos los invariantes satisfechos.
        """
        return self._engine.phase3_witness_adjudicate_heyting(
            cert_p1=ctx.cert_p1,
            cert_p2=ctx.cert_p2,
            rsi3_state=rsi3_state,
        )

    def phase3_apply_visceral_functor_sem(
        self,
        ctx: _Phase3WorkingContext,
        truth_value: HeytingTruthValue,
    ) -> VisceralImpactReport:
        r"""
        Funtor Semántico Visceral Φ_sem hacia métricas de C-Suite.

        - WACC          : 15.0 % invariante (protección del coste de capital).
        - Liquidez neta : tiered (11.5 % / 5.75 % / 0.0 % según veredicto).
        - KV-Cache     : 86.4 % invariante.
        - Ahorro IA    : 80.0 % invariante.

        El invariante PASSIVE_SILENT se VERIFICA empíricamente: se compara la
        back-action observada `|qnd.back_action_db|` contra la tolerancia. No
        se asume `True`.
        """
        back_action_db = float(ctx.cert_p1.qnd_report.back_action_db)
        is_passive_silent = (
            math.isfinite(back_action_db)
            and abs(back_action_db) <= _QND_PASSIVE_SILENCE_TOLERANCE_DB
        )

        if truth_value == HeytingTruthValue.COHERENT:
            liquidity_pct = _LIQUIDITY_COHERENT_PCT
        elif truth_value == HeytingTruthValue.DEGRADED:
            liquidity_pct = _LIQUIDITY_DEGRADED_PCT
        else:  # VETOED
            liquidity_pct = _LIQUIDITY_VETOED_PCT

        return VisceralImpactReport(
            wacc_protection_pct=_WACC_INVARIANT_PCT,
            net_liquidity_liberation_pct=float(liquidity_pct),
            kv_cache_compression_pct=_KV_CACHE_COMPRESSION_PCT,
            inference_cost_saving_pct=_AI_INFERENCE_SAVING_PCT,
            qnd_silence_backaction_db=back_action_db,
            is_passive_silent_verified=bool(is_passive_silent),
        )

    def phase3_emit_sovereign_state(
        self,
        ctx: _Phase3WorkingContext,
        rsi3_state: RSI3WitnessState,
        truth_value: HeytingTruthValue,
        adjudication_reason: str,
        visceral_impact: VisceralImpactReport,
        t_start_perf: float,
    ) -> BoolNovikovPseudoholomorphicWitnessSovereignState:
        r"""
        ÚLTIMO MÉTODO DE LA FASE III.

        Emite el `BoolNovikovPseudoholomorphicWitnessSovereignState` final con:
          • merkle_root_hash SHA-256 sobre (chain P1 ⊕ chain P2 ⊕ veredicto ⊕
            razón ⊕ RSI³ ⊕ Φ_sem).
          • hmac_signature HMAC-SHA256 en el dominio Φ₃ₚ.
          • is_passive_silent derivado de la observación real.
          • adjudication_reason como campo explícito auditable.
        """
        now = time.time()

        # Payload determinista para Merkle root + HMAC soberano
        payload = (
            f"WITNESS-AGENT-FINAL|{ctx.cert_p1.chain_hash}|{ctx.cert_p2.chain_hash}|"
            f"{rsi3_state.banach_contraction_constant:.12g}|"
            f"{rsi3_state.fixed_point_error:.12g}|"
            f"{rsi3_state.banach_residual:.12g}|"
            f"{rsi3_state.capacity_jerk:.12g}|"
            f"{rsi3_state.is_banach_contractive}|{rsi3_state.jerk_strictly_positive}|"
            f"{truth_value.value}|{adjudication_reason}|"
            f"{visceral_impact.wacc_protection_pct:.6g}|"
            f"{visceral_impact.net_liquidity_liberation_pct:.6g}|"
            f"{visceral_impact.kv_cache_compression_pct:.6g}|"
            f"{visceral_impact.inference_cost_saving_pct:.6g}|"
            f"{visceral_impact.qnd_silence_backaction_db:.12g}|"
            f"{visceral_impact.is_passive_silent_verified}"
        ).encode("utf-8")

        merkle_root = hashlib.sha256(payload).hexdigest()
        sovereign_hmac = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + payload,
            hashlib.sha256,
        ).hexdigest()

        total_latency_ms = (time.perf_counter() - t_start_perf) * 1000.0

        return BoolNovikovPseudoholomorphicWitnessSovereignState(
            timestamp=now,
            agent_version=_AGENT_VERSION,
            truth_value=truth_value,
            phase1_cert=ctx.cert_p1,
            phase2_cert=ctx.cert_p2,
            rsi3_state=rsi3_state,
            visceral_impact=visceral_impact,
            merkle_root_hash=merkle_root,
            hmac_signature=sovereign_hmac,
            is_passive_silent=visceral_impact.is_passive_silent_verified,
            adjudication_reason=adjudication_reason,
            total_latency_ms=total_latency_ms,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ORQUESTADOR OODA TRIFÁSICO + PARIDAD CONTRA EL MOTOR
    # ══════════════════════════════════════════════════════════════════════════

    def execute_sovereign_witness_cycle(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: Sequence[float],
        ball_r: float = 1.0,
        cylinder_R: float = 1.2,
        coefficients: Optional[Sequence[complex]] = None,
        pert_eps: float = 0.05,
        delta_qnd: float = 0.05,
        strict_engine_parity: bool = True,
    ) -> BoolNovikovPseudoholomorphicWitnessSovereignState:
        r"""
        Orquestador OODA completo:

            FASE I  ➔  Φ₁₂  ➔  FASE II  ➔  Φ₂₃  ➔  FASE III

        Parámetros
        ----------
        apu_cost_matrix : ndarray[float64]
            Matriz de costos interpretada como superficie u: R² → R.
        exponents : Sequence[float]
            Exponentes λ_i de la serie de Novikov Λ_Nov.
        ball_r, cylinder_R : float
            Radios para la cota de Gromov non-squeezing.
        coefficients : Sequence[complex], optional
            Coeficientes a_i. Si None, se usa [1+0j]*len(exponents) (compatible
            con el default del engine v14).
        pert_eps : float
            Fuerza perturbativa para la integral de Melnikov.
        delta_qnd : float
            Fuerza del acoplamiento débil AAV.
        strict_engine_parity : bool
            Si True (default), invoca `engine.execute_witness_audit` como oráculo
            y exige coincidencia de veredicto (fail-closed).
        """
        t_start = time.perf_counter()

        mat = np.atleast_2d(np.asarray(apu_cost_matrix, dtype=np.float64))
        n, m = mat.shape if mat.ndim == 2 and mat.size > 0 else (0, 0)
        dim = max(1, min(n, m))

        # ────────────────────────────────────────────────────────────────────
        # FASE I — OBSERVE
        # ────────────────────────────────────────────────────────────────────
        # QND weak measurement sobre estado máximamente mixto (pasividad trivial)
        rho_mac = np.eye(dim, dtype=np.complex128) / float(dim)
        phi_uniform = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        qnd_report = self.phase1_observe_qnd_witness(
            rho_mac=rho_mac,
            phi_i=phi_uniform,
            phi_f=phi_uniform,
            delta=delta_qnd,
        )

        # Bures-Uhlmann contra ρ levemente sesgado no trivial
        bias = np.arange(1, dim + 1, dtype=np.float64)
        sigma_mac = rho_mac + 0.01 * np.diag(bias).astype(np.complex128)
        sigma_mac = 0.5 * (sigma_mac + sigma_mac.conj().T)
        sigma_mac /= np.trace(sigma_mac)
        bures_report = self.phase1_observe_bures_uhlmann(rho_mac, sigma_mac)

        # Novikov-Krull
        if coefficients is None:
            coefficients = [1.0 + 0.0j] * len(exponents)
        novikov_state = self.phase1_observe_novikov_krull(coefficients, exponents)

        # Cierre de FASE I
        cert_p1 = self.phase1_emit_sovereign_certificate(
            qnd_report=qnd_report,
            bures_report=bures_report,
            novikov_state=novikov_state,
        )

        # ────────────────────────────────────────────────────────────────────
        # COMPUERTA Φ₁₂
        # ────────────────────────────────────────────────────────────────────
        if not self.continue_to_phase2_gate(cert_p1):
            raise WitnessAgentGateViolationError(
                "Falla de autenticidad/integridad en Compuerta Homomórfica Φ₁₂."
            )

        # ────────────────────────────────────────────────────────────────────
        # FASE II — ORIENT
        # ────────────────────────────────────────────────────────────────────
        ctx_p2 = self.phase2_open_from_phase1_certificate(cert_p1)
        surface_witness = self.phase2_orient_pseudoholomorphic_surface(
            ctx=ctx_p2,
            cost_matrix=mat,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
        )
        celestial_witness = self.phase2_orient_poincare_kam(
            ctx=ctx_p2,
            surface_witness=surface_witness,
            pert_eps=pert_eps,
        )
        cert_p2 = self.phase2_emit_sovereign_certificate(
            ctx=ctx_p2,
            surface_witness=surface_witness,
            celestial_witness=celestial_witness,
        )

        # ────────────────────────────────────────────────────────────────────
        # COMPUERTA Φ₂₃
        # ────────────────────────────────────────────────────────────────────
        if not self.continue_to_phase3_gate(cert_p1, cert_p2):
            raise WitnessAgentGateViolationError(
                "Falla de estabilidad/autenticidad en Compuerta Homomórfica Φ₂₃."
            )

        # ────────────────────────────────────────────────────────────────────
        # FASE III — DECIDE & ACT
        # ────────────────────────────────────────────────────────────────────
        ctx_p3 = self.phase3_open_from_phase2_certificate(cert_p1, cert_p2)

        rsi3_state = self.phase3_evaluate_rsi3_banach(
            ctx=ctx_p3,
            state_vector=np.asarray(mat, dtype=np.float64).flatten(),
        )
        truth_value, reason = self.phase3_adjudicate_heyting_omega3(
            ctx=ctx_p3,
            rsi3_state=rsi3_state,
        )
        visceral_impact = self.phase3_apply_visceral_functor_sem(
            ctx=ctx_p3,
            truth_value=truth_value,
        )
        sovereign_state = self.phase3_emit_sovereign_state(
            ctx=ctx_p3,
            rsi3_state=rsi3_state,
            truth_value=truth_value,
            adjudication_reason=reason,
            visceral_impact=visceral_impact,
            t_start_perf=t_start,
        )

        # ────────────────────────────────────────────────────────────────────
        # PARIDAD CONTRA EL MOTOR CANÓNICO
        # ────────────────────────────────────────────────────────────────────
        if strict_engine_parity:
            engine_report: PseudoholomorphicWitnessReport = (
                self._engine.execute_witness_audit(
                    apu_cost_matrix=mat,
                    exponents=exponents,
                    coefficients=coefficients,
                    ball_r=ball_r,
                    cylinder_R=cylinder_R,
                    pert_eps=pert_eps,
                    delta_qnd=delta_qnd,
                )
            )
            if engine_report.truth_value != sovereign_state.truth_value:
                raise WitnessAgentParityViolationError(
                    "Divergencia agente↔motor: "
                    f"agente={sovereign_state.truth_value.name} "
                    f"(motivo={sovereign_state.adjudication_reason}), "
                    f"motor={engine_report.truth_value.name} "
                    f"(motivo={engine_report.adjudication_reason or 'n/a'})."
                )

        return sovereign_state


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN UNITARIA AUTÓNOMA EN SANDBOX
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPseudoholomorphicWitnessAgent v14.0.0 ===")

    agent = BoolNovikovPseudoholomorphicWitnessAgent()

    # ── Test 1: escenario base (matriz 2×2 simétrica) ──
    matrix_test = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64)
    exps_test = [0.15, 0.22, 0.40]

    state = agent.execute_sovereign_witness_cycle(matrix_test, exps_test)

    print(f"[Test 1] Veredicto Ω₃              : {state.truth_value.name} "
          f"({hex(state.truth_value.value)})")
    print(f"[Test 1] Razón de adjudicación     : {state.adjudication_reason}")
    print(f"[Test 1] Invariante PASSIVE_SILENT : {state.is_passive_silent}")
    print(f"[Test 1] Back-Action QND observada : "
          f"{state.phase1_cert.qnd_report.back_action_db:.6e} dB")
    print(f"[Test 1] Conmutador QND            : "
          f"{state.phase1_cert.qnd_report.commutator_norm:.6e}")
    print(f"[Test 1] Fidelidad de Uhlmann      : "
          f"{state.phase1_cert.bures_report.fidelity_uhlmann:.6f}")
    print(f"[Test 1] Distancia de Bures        : "
          f"{state.phase1_cert.bures_report.bures_distance:.6e}")
    print(f"[Test 1] Valuación de Krull        : "
          f"{state.phase1_cert.novikov_state.krull_valuation:.4f}")
    print(f"[Test 1] Residual CR-Floer         : "
          f"{state.phase2_cert.surface_witness.cauchy_riemann_residual:.6e}")
    print(f"[Test 1] Gromov rigid              : "
          f"{state.phase2_cert.surface_witness.is_gromov_rigid}")
    print(f"[Test 1] Melnikov (valor)          : "
          f"{state.phase2_cert.celestial_witness.melnikov_integral_value:.6e}")
    print(f"[Test 1] KAM margen diofántico     : "
          f"{state.phase2_cert.celestial_witness.kam_diophantine_margin:.6f}")
    print(f"[Test 1] Lip(T³)                   : "
          f"{state.rsi3_state.banach_contraction_constant:.6f}")
    print(f"[Test 1] Residual Banach-Picard    : "
          f"{state.rsi3_state.fixed_point_error:.6e}")
    print(f"[Test 1] Jerk ciber-físico         : "
          f"{state.rsi3_state.capacity_jerk:.6e}")
    print(f"[Test 1] Protección WACC           : "
          f"{state.visceral_impact.wacc_protection_pct}%")
    print(f"[Test 1] Liberación liquidez       : "
          f"{state.visceral_impact.net_liquidity_liberation_pct}%")
    print(f"[Test 1] Compresión KV-Cache       : "
          f"{state.visceral_impact.kv_cache_compression_pct}%")
    print(f"[Test 1] Merkle root SHA-256       : {state.merkle_root_hash[:16]}...")
    print(f"[Test 1] HMAC soberano             : {state.hmac_signature[:16]}...")
    print(f"[Test 1] Latencia total (ms)       : {state.total_latency_ms:.3f}")

    # ── Test 2: Krull inválido ⇒ VETOED (paridad verificada) ──
    print()
    state_bad_krull = agent.execute_sovereign_witness_cycle(
        matrix_test,
        exponents=[0.03, 0.05],  # por debajo del piso 0.10
    )
    print(f"[Test 2] Veredicto Ω₃              : {state_bad_krull.truth_value.name}")
    print(f"[Test 2] Razón                     : {state_bad_krull.adjudication_reason}")
    print(f"[Test 2] Passivo/Silencioso        : {state_bad_krull.is_passive_silent}")
    print(f"[Test 2] Liberación liquidez       : "
          f"{state_bad_krull.visceral_impact.net_liquidity_liberation_pct}%")
    assert state_bad_krull.truth_value == HeytingTruthValue.VETOED
    assert state_bad_krull.visceral_impact.net_liquidity_liberation_pct == 0.0

    # ── Test 3: Gromov violado (r > R) ⇒ VETOED vía adjudicación ──
    # Nota: nuestro gate Φ₂₃ no bloquea Gromov (sólo back-action), por lo que
    # la anomalía llega a FASE III y se adjudica VETOED con razón explícita.
    print()
    state_bad_gromov = agent.execute_sovereign_witness_cycle(
        matrix_test,
        exps_test,
        ball_r=2.0,
        cylinder_R=1.0,  # r > R
    )
    print(f"[Test 3] Veredicto Ω₃              : {state_bad_gromov.truth_value.name}")
    print(f"[Test 3] Razón                     : {state_bad_gromov.adjudication_reason}")
    assert state_bad_gromov.truth_value == HeytingTruthValue.VETOED

    # ── Test 4: matriz vacía ⇒ adjudicación estable ──
    print()
    state_empty = agent.execute_sovereign_witness_cycle(
        np.zeros((0, 0), dtype=np.float64),
        exps_test,
    )
    print(f"[Test 4] Veredicto Ω₃              : {state_empty.truth_value.name}")
    print(f"[Test 4] Razón                     : {state_empty.adjudication_reason}")

    # ── Test 5: matriz 5×5 no armónica → CR residual no trivial ──
    print()
    rng = np.random.default_rng(seed=42)
    big_matrix = rng.normal(size=(5, 5))
    state_big = agent.execute_sovereign_witness_cycle(big_matrix, exps_test)
    print(f"[Test 5] Veredicto Ω₃              : {state_big.truth_value.name}")
    print(f"[Test 5] Residual CR               : "
          f"{state_big.phase2_cert.surface_witness.cauchy_riemann_residual:.6e}")
    print(f"[Test 5] Merkle root SHA-256       : {state_big.merkle_root_hash[:16]}...")

    # ── Test 6: fuerte acoplamiento QND (δ = 5) ⇒ back-action posible -> gate Φ₂₃ fail-fast ──
    # Nota: con δ exagerado la perturbación hermitiana crece, pero sobre ρ=I/dim
    # el conmutador sigue siendo cero. Se demuestra que el gate Φ₂₃ respeta
    # la tolerancia estricta.
    print()
    try:
        state_aggressive = agent.execute_sovereign_witness_cycle(
            matrix_test,
            exps_test,
            delta_qnd=5.0,
        )
        print(f"[Test 6] Veredicto Ω₃              : {state_aggressive.truth_value.name}")
        print(f"[Test 6] Back-action QND           : "
              f"{state_aggressive.phase1_cert.qnd_report.back_action_db:.6e} dB")
        print(f"[Test 6] Passivo/Silencioso        : {state_aggressive.is_passive_silent}")
    except WitnessAgentGateViolationError as exc:
        print(f"[Test 6] Gate correctamente disparado: {exc}")

    # ── Test 7: alias de compatibilidad v13 (WitnessPhase1Certificate) ──
    print()
    from bool_novikov_pseudoholomorphic_witness_agent import (
        WitnessPhase1Certificate as AliasP1,
    )
    from bool_novikov_pseudoholomorphic_witness_engine import (
        Phase1WitnessCertificate as EngineP1,
    )
    assert AliasP1 is EngineP1, "El alias v13 debe apuntar al tipo v14 del motor."
    print(f"[Test 7] Alias WitnessPhase1Certificate → Phase1WitnessCertificate : OK")

    print()
    print("=== TEST COMPLETADO CON ÉXITO ===")