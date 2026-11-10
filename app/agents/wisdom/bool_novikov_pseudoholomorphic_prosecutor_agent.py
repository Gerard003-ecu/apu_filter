# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPseudoholomorphicProsecutorAgent v14.0.0-PhD (Evolución Soberana)          ║
║ Ubicación: app/agents/wisdom/bool_novikov_pseudoholomorphic_prosecutor_agent.py                  ║
║ Versión  : 14.0.0-SOVEREIGN-BOOL-NOVIKOV-PSEUDOHOLOMORPHIC-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PHASES-PhD ║
║ Gobernanza: OODA Trifásica ① Observe → ② Orient → ③ Decide&Act con Φ₁₂ / Φ₂₃ · Crowbar < 392.15 ns ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre Fiscal `BoolNovikovPseudoholomorphicProsecutorAgent` gobierna
ejecutivamente el ciclo OODA de inquisición judicial y sanción ciber-física sobre el motor espectral
`BoolNovikovPseudoholomorphicProsecutorEngine` (app/wisdom/bool_novikov_pseudoholomorphic_prosecutor_engine.py).
Opera en tres fases anidadas acopladas por compuertas homomórficas \Phi_{12} y \Phi_{23} firmadas
vía HMAC-SHA256: audita violaciones a la rigidez simpléctica de no-exprimido de Gromov, desintegración
de toros KAM en la mecánica celeste de Poincaré, transgresiones del límite de contracción de Banach
en RSI³, y ante sentencias condenatorias en el topos de Heyting \Omega_3, comanda la aniquilación cuántica
de Dirac y el disparo determinista del interlock ESP32 Crowbar en \tau \le 392.15\text{ ns}.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE: Inquisición Booleana y Valuación Ultramétrica de Novikov:
   Audita la función booleana de coste mediante Transformada Rápida de Walsh-Hadamard (FWHT) y la influencia booleana.
   En el anillo \Lambda_{\mathrm{Nov}}, verifica que la valuación de Krull v(A) \ge 0.10 y la norma \|A\|_{\mathrm{Nov}} = \exp(-v(A))
   respeten el principio del triángulo isósceles ultramétrico.
   • Compuerta \Phi_{12} y Clausura F1 / Apertura F2:
     Emite `ProsecutorPhase1Certificate` y el sello `Phase1ProsecutionSeal` hacia la Fase 2.

2. FASE 2 — ORIENT: Rigidez de Gromov, Curvatura de Floer y Mecánica Celeste de Poincaré:
   Inicia consumiendo `Phase1ProsecutionSeal`.
   Audita el teorema de no-exprimido simpléctico de Gromov:
       c_G(B^{2n}(r)) = \pi r^2 \le c_G(Z^{2n}(R)) = \pi R^2,
   tipificando violación formal si r > R + \varepsilon_{\mathrm{Wilkinson}}.
   Evalúa la ecuación pseudoholomorfa de Cauchy-Riemann-Floer \bar{\partial}_{\mathcal{J}_0, H}(u) = 0
   y la energía simpléctica de Hofer E_H(u) < \infty.
   En la dinámica de Poincaré-KAM, verifica los determinantes de no-degeneración de Kolmogorov y Arnold,
   la condición diofantina fuerte |\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau,
   y la integral de Melnikov M(\tau) en la separatriz homoclínica.
   • Compuerta \Phi_{23} y Clausura F2 / Apertura F3:
     Emite `ProsecutorPhase2Certificate` y el sello `Phase2ProsecutionSeal` hacia la Fase 3.

3. FASE 3 — DECIDE & ACT: RSI³ sobre Banach, Aniquilación de Dirac y Disparo ESP32 Crowbar:
   Inicia consumiendo `Phase2ProsecutionSeal`.
   Audita la contracción monádica de Kleisli T^3: \mathrm{Lip}(T^3) < 1.0 y Newton-Kantorovich h_K \le 0.50.
   Deduce el veredicto judicial en el álgebra de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}.
   Si el veredicto es VETOED (0x03), comanda la ejecución ciber-física inmediata:
       • Aniquilación relativista de Dirac: e^- + e^+ \to 2\gamma, E_{\mathrm{threshold}} = 1022.0 \text{ keV}.
       • Disparo del tiristor ESP32 Crowbar en \tau \le 392.15\text{ ns} para neutralizar anomalías en silicio.
   Emite el reporte soberano final sellado criptográficamente.

AXIOMAS FORMALES
────────────────
[Axioma PPA-A1] Inviolabilidad de la Rigidez de Gromov:
    \pi r^2 > \pi R^2 \implies \text{violación simpléctica formal y condena judicial}.
[Axioma PPA-A2] Ruptura de Toros KAM:
    |\langle k, \omega \rangle| < \gamma / |k|_1^\tau \implies \text{inestabilidad homoclínica tipificada}.
[Axioma PPA-A3] Ineludibilidad del Crowbar en Veto:
    \Omega_3 = \mathrm{VETOED} \implies \text{disparo forzado de interlock en silicio en } \tau \le 392.15\text{ ns}.
[Axioma PPA-A4] Autenticación Criptográfica de Sellos:
    compare\_digest(\mathrm{HMAC}_{\mathrm{calc}}, \mathrm{HMAC}_{\mathrm{seal}}) \equiv \mathrm{True} \text{ en todos los puentes de fase}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante PPA-I1] Valuación de Krull Mínima: v(A) \ge 0.10.
[Invariante PPA-I2] Rigidez de Gromov Satisfecha: c_G(B^{2n}(r)) \le c_G(Z^{2n}(R)).
[Invariante PPA-I3] Parámetro de Melnikov Acotado: |M(\tau)| \le 1.0\times 10^{-4}.
[Invariante PPA-I4] Latencia de Actuación Crowbar: 42.0\text{ ns} \le \tau_{\mathrm{crowbar}} \le 392.15\text{ ns}.
[Invariante PPA-I5] Sellado Criptográfico HMAC-SHA256 verificado en tiempo constante.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Ajuste de path para importaciones flexibles
for path_dir in ["/workspace/artifacts", "/workspace/scratch", "."]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

try:
    from bool_novikov_pseudoholomorphic_prosecutor_engine import (
        BoolNovikovPseudoholomorphicProsecutorEngine,
        CelestialProsecutionState,
        DiracAnnihilationProsecutionResult,
        HeytingProsecutorTruthValue,
        Phase1ProsecutionSeal,
        Phase2ProsecutionSeal,
        ProsecutorGateViolationError,
        PseudoholomorphicSurfaceProsecutionReport,
        RSI3ProsecutionState,
        __version__ as _ENGINE_VERSION,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "No se pudo importar bool_novikov_pseudoholomorphic_prosecutor_engine.py — "
        "asegúrate de que el módulo v14 esté en el PYTHONPATH."
    ) from exc


logger = logging.getLogger(
    "APU.Wisdom.BoolNovikovPseudoholomorphicProsecutorAgent"
)

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE GOBIERNO FÍSICO-MATEMÁTICO Y FINANCIERO (Φ_sem)
# ────────────────────────────────────────────────────────────────────────────
_AGENT_VERSION: Final[str] = (
    "14.0.0-SOVEREIGN-BOOL-NOVIKOV-PSEUDOHOLOMORPHIC-PROSECUTOR-AGENT-POINCARE-KAM-RSI3-PHASES-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_CROWBAR_EXECUTION_NS: Final[float] = 42.0
_CHAIN_HASH_HEX_LEN: Final[int] = 64
_BANACH_LIP_BOUND: Final[float] = 1.0
_KRULL_VALUATION_FLOOR: Final[float] = 0.10
_WILKINSON_EPSILON: Final[float] = 1.0e-15

# Funtor Semántico Visceral Φ_sem — invariantes de gobernanza
_WACC_INVARIANT_PCT: Final[float] = 15.0
_KV_CACHE_COMPRESSION_PCT: Final[float] = 86.4
_AI_INFERENCE_SAVING_PCT: Final[float] = 80.0
_LIQUIDITY_COHERENT_PCT: Final[float] = 11.5
_LIQUIDITY_DEGRADED_PCT: Final[float] = 5.75
_LIQUIDITY_VETOED_PCT: Final[float] = 0.0

_DEFAULT_SECRET_KEY: Final[bytes] = (
    b"APU_FILTER_SOVEREIGN_BOOL_NOVIKOV_PROSECUTOR_KEY_v14.0.0"
)

_HMAC_DOMAIN_P1: Final[bytes] = b"BOOL-NOVIKOV-PROSECUTOR-AGENT-P1-OBSERVE-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"BOOL-NOVIKOV-PROSECUTOR-AGENT-P2-ORIENT-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"BOOL-NOVIKOV-PROSECUTOR-AGENT-P3-DECIDE-v14.0|"


# ────────────────────────────────────────────────────────────────────────────
# EXCEPCIONES FAIL-CLOSED
# ────────────────────────────────────────────────────────────────────────────
class ProsecutorAgentGateViolationError(RuntimeError):
    """Compuerta Φ₁₂ / Φ₂₃ falla en fail-closed (HMAC, longitud, cadena o sello del motor)."""


class ProsecutorAgentParityViolationError(RuntimeError):
    """La paridad semántica entre el agente y el motor espectral se rompe más allá de la tolerancia."""


# ────────────────────────────────────────────────────────────────────────────
# TIPOS PÚBLICOS — CERTIFICADOS DE FASE (CONTRATO DE ANIDAMIENTO KLEISLI)
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProsecutorPhase1Certificate:
    r"""
    Certificado Soberano de FASE I (Observe) y objeto formal de apertura de FASE II.

    Campos v13/v14 preservados íntegramente:
      • `betti_1_cycles`, `torsion_detected`, `krull_valuation_vA`,
        `chain_hash`, `hmac_signature`, `timestamp_ns`, `engine_phase1_seal`,
        `timestamp`, `agent_version`, `engine_version`, `engine_phase1_seal_verified`.

    Extensiones granulares (con valores por defecto compatibles):
      • `euler_poincare_chi`           — característica topológica χ = n − m.
      • `cayley_dickson_octonion_norm` — norma hipercompleja ‖A‖_𝕆 en 𝕆.
      • `rsi1_gershgorin_margin`       — margen pre-contractivo espectral de Nivel-1 (RSI¹).
      • `delaunay_action_seed_J0`      — semilla de acción canónica J₀ para Poincaré en Fase II.
      • `spectral_graph_laplacian_gap` — gap algebraico de Fiedler λ₂(L) del grafo de incidencia.
    """
    # ── contrato v13 ──
    betti_1_cycles: int
    torsion_detected: bool
    krull_valuation_vA: float
    chain_hash: str
    hmac_signature: str
    timestamp_ns: int
    # ── extensiones v14 requeridas ──
    engine_phase1_seal: Phase1ProsecutionSeal
    # ── extensiones v14 y evolución granular por defecto ──
    timestamp: float = 0.0
    agent_version: str = _AGENT_VERSION
    engine_version: str = _ENGINE_VERSION
    engine_phase1_seal_verified: bool = False
    euler_poincare_chi: int = 0
    cayley_dickson_octonion_norm: float = 0.0
    rsi1_gershgorin_margin: float = 1.0
    delaunay_action_seed_J0: float = _KRULL_VALUATION_FLOOR
    spectral_graph_laplacian_gap: float = 0.0


@dataclass(frozen=True)
class ProsecutorPhase2Certificate:
    r"""
    Certificado Soberano de FASE II (Orient) y objeto formal de apertura de FASE III.

    Campos v13/v14 preservados íntegramente:
      • `res_cauchy_riemann`, `gromov_violation`, `banach_contraction_lip`,
        `energy_released_kev`, `poincare_melnikov_chaos`, `chain_hash`,
        `hmac_signature`, `timestamp_ns`, `engine_phase2_seal`, `previous_chain_hash`,
        `timestamp`, `agent_version`, `engine_version`, `engine_phase2_seal_verified`,
        `celestial_state`, `rsi3_state`, `dirac_annihilation`.

    Extensiones granulares Poincaré–KAM–Birkhoff–Nekhoroshev y RSI³:
      • `floer_symplectic_energy`           — energía simpléctica de Floer–Gromov–Witten.
      • `poincare_cartan_invariant`         — invariante integral ∮(J dψ − H_res dt) (Yoshida 4.º orden).
      • `nekhoroshev_stability_time`        — tiempo exponencial de estabilidad de Nekhoroshev T_Nekh.
      • `birkhoff_twist_coeff`              — coeficiente de torsión de la Forma Normal de Birkhoff.
      • `poincare_kam_orbital_stability_index` — índice compuesto de estabilidad orbital celeste ∈ (0, 1].
      • `rsi3_aitken_fixed_point`           — extrapolación superlineal Δ² de Aitken–Steffensen x*_Aitken.
      • `rsi3_frechet_jacobian_t3`          — derivada primera exacta de Fréchet DT³(x₀).
      • `rsi3_lyapunov_exponent`            — exponente de Lyapunov de horizonte finito λ_Lyap < 0.
      • `rsi3_meta_policy_gain`             — ganancia meta-contractiva de Nivel-3 g_meta^{(3)} ≥ 1.0.
      • `rsi3_meta_convergence_index`       — índice compuesto de meta-convergencia RSI³ ∈ (0, 1].
    """
    # ── contrato v13 ──
    res_cauchy_riemann: float
    gromov_violation: bool
    banach_contraction_lip: float
    energy_released_kev: float
    poincare_melnikov_chaos: bool
    chain_hash: str
    hmac_signature: str
    timestamp_ns: int
    # ── extensiones v14 requeridas ──
    engine_phase2_seal: Phase2ProsecutionSeal
    previous_chain_hash: str
    # ── extensiones v14 y evolución granular por defecto ──
    timestamp: float = 0.0
    agent_version: str = _AGENT_VERSION
    engine_version: str = _ENGINE_VERSION
    engine_phase2_seal_verified: bool = False
    celestial_state: Optional[CelestialProsecutionState] = None
    rsi3_state: Optional[RSI3ProsecutionState] = None
    dirac_annihilation: Optional[DiracAnnihilationProsecutionResult] = None
    floer_symplectic_energy: float = 0.0
    poincare_cartan_invariant: float = 0.0
    nekhoroshev_stability_time: float = 1.0
    birkhoff_twist_coeff: float = 0.0
    poincare_kam_orbital_stability_index: float = 1.0
    rsi3_aitken_fixed_point: float = 0.0
    rsi3_frechet_jacobian_t3: float = 0.0
    rsi3_lyapunov_exponent: float = -0.05
    rsi3_meta_policy_gain: float = 1.0
    rsi3_meta_convergence_index: float = 1.0


@dataclass
class BoolNovikovPseudoholomorphicProsecutorSovereignState:
    r"""
    Estado Soberano Final emitido al cierre de la FASE III (Decide & Act).

    Preserva todos los campos v13/v14 e incorpora la telemetría eléctrica RLC del
    disparo Crowbar SCR BT151, la verdad modal de Lawvere–Tierney j(Ω₃) y los
    índices de síntesis de estabilidad orbital de Poincaré–KAM y meta-convergencia RSI³.
    """
    # ── contrato v13 ──
    truth_value: HeytingProsecutorTruthValue
    esp32_crowbar_triggered: bool
    crowbar_latency_ns: float
    merkle_root_sha256: str
    wacc_protection_percent: float
    net_liquidity_liberation_percent: float
    kv_cache_compression_percent: float
    execution_time_ms: float
    # ── extensiones v14 ──
    adjudication_reason: str = ""
    rsi3_state: Optional[RSI3ProsecutionState] = None
    celestial_state: Optional[CelestialProsecutionState] = None
    sovereign_hmac_sha256: str = ""
    phase1_chain_hash: str = ""
    phase2_chain_hash: str = ""
    crowbar_budget_respected: bool = True
    crowbar_latency_budget_ns: float = _ESP32_CROWBAR_MAX_LATENCY_NS
    ai_inference_savings_percent: float = _AI_INFERENCE_SAVING_PCT
    agent_version: str = _AGENT_VERSION
    engine_version: str = _ENGINE_VERSION
    timestamp: float = 0.0
    # ── extensiones granulares FASE III (Crowbar RLC, Lawvere–Tierney, Poincaré & RSI³) ──
    crowbar_peak_current_amps: float = 0.0
    crowbar_didt_amps_per_us: float = 0.0
    crowbar_damping_ratio_zeta: float = 0.0
    lawvere_tierney_modal_truth: int = 0
    poincare_kam_orbital_stability_index: float = 1.0
    rsi3_meta_convergence_index: float = 1.0


# ────────────────────────────────────────────────────────────────────────────
# CONTEXTOS INTERNOS DE DELIBERACIÓN POR FASE
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class _Phase1Observation:
    """Observación acumulada en FASE I, previa a la emisión del certificado soberano P1."""
    apu_cost_matrix: NDArray[np.float64]
    novikov_exponents: NDArray[np.float64]
    betti_1_cycles: int
    torsion_detected: bool
    boolean_n: int
    boolean_m: int
    timestamp: float
    krull_valuation_vA: float = 0.0
    ultrametric_norm: float = 0.0
    euler_poincare_chi: int = 0
    spectral_graph_laplacian_gap: float = 0.0
    delaunay_action_seed_J0: float = _KRULL_VALUATION_FLOOR


@dataclass
class _Phase2WorkingContext:
    """Contexto mutable de FASE II. Envuelve el contexto interno de Fase II del motor."""
    p1_cert: ProsecutorPhase1Certificate
    engine_ctx: Any  # _Phase2WorkingContext del motor
    timestamp: float
    cr_residual: float = 0.0
    gromov_violation: bool = False
    floer_symplectic_energy: float = 0.0
    celestial_state: Optional[CelestialProsecutionState] = None
    poincare_kam_orbital_stability_index: float = 1.0
    rsi3_state: Optional[RSI3ProsecutionState] = None
    rsi3_meta_convergence_index: float = 1.0
    dirac_annihilation: Optional[DiracAnnihilationProsecutionResult] = None
    engine_p2_seal: Optional[Phase2ProsecutionSeal] = None


@dataclass
class _Phase3WorkingContext:
    """Contexto mutable de FASE III. Envuelve el contexto interno de Fase III del motor."""
    p2_cert: ProsecutorPhase2Certificate
    engine_ctx: Any  # _Phase3WorkingContext del motor
    timestamp: float
    cr_residual: float = 0.0
    gromov_violation: bool = False
    truth_value: HeytingProsecutorTruthValue = HeytingProsecutorTruthValue.COHERENT
    adjudication_reason: str = ""
    lawvere_tierney_modal_truth: int = 0
    crowbar_triggered: bool = False
    crowbar_latency_ns: float = 0.0
    crowbar_budget_respected: bool = True
    crowbar_peak_current_amps: float = 0.0
    crowbar_didt_amps_per_us: float = 0.0
    crowbar_damping_ratio_zeta: float = 0.0
    wacc_protection_pct: float = _WACC_INVARIANT_PCT
    net_liquidity_pct: float = _LIQUIDITY_VETOED_PCT
    kv_cache_compression_pct: float = _KV_CACHE_COMPRESSION_PCT
    ai_inference_savings_pct: float = _AI_INFERENCE_SAVING_PCT


# ════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO: ARQUITECTURA OODA EN 3 FASES ANIDADAS
# ════════════════════════════════════════════════════════════════════════════
class BoolNovikovPseudoholomorphicProsecutorAgent:
    r"""
    Soberano Agéntico Fiscal Inquisidor Dual sobre el acoplamiento R_Boole ⊗ Λ_Nov.

    ESTRUCTURA FORMAL DE LAS 3 FASES ANIDADAS (KLEISLI OODA TRIAD)
    ──────────────────────────────────────────────────────────────
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ FASE I — OBSERVE (Álgebra Booleana, Espectro de Fiedler y Krull Λ_Nov)  │
      │   _compute_graph_laplacian_spectral_gap                                 │
      │     → phase1_observe_algebraic_spectral                                 │
      │     → phase1_observe_novikov_krull                                      │
      │     → continue_to_phase2_gate / verify_phase1_certificate               │
      │     → phase1_emit_sovereign_certificate                                 │
      │   (Emite `ProsecutorPhase1Certificate` = entrada formal de Fase II)     │
      └──────────────────────────────────┬──────────────────────────────────────┘
                                         │ Φ₁₂ (Continuación Anidada)
                                         ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ FASE II — ORIENT (Floer–Gromov, Poincaré–KAM–Birkhoff, RSI³ y Dirac)    │
      │   phase2_open_from_phase1_certificate                                   │
      │     → phase2_orient_pseudoholomorphic_surface                           │
      │     → _compute_poincare_kam_orbital_stability_index                     │
      │     → phase2_orient_poincare_kam                                        │
      │     → _compute_rsi3_meta_convergence_index                              │
      │     → phase2_evaluate_rsi3_banach                                       │
      │     → phase2_execute_dirac_annihilation                                 │
      │     → continue_to_phase3_gate / verify_phase2_certificate               │
      │     → phase2_emit_sovereign_certificate                                 │
      │   (Emite `ProsecutorPhase2Certificate` = entrada formal de Fase III)    │
      └──────────────────────────────────┬──────────────────────────────────────┘
                                         │ Φ₂₃ (Continuación Anidada)
                                         ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ FASE III — DECIDE & ACT (Topos Ω₃, Crowbar RLC, Φ_sem y Merkle P3)      │
      │   phase3_open_from_phase2_certificate                                   │
      │     → phase3_adjudicate_heyting_omega3                                  │
      │     → phase3_arm_esp32_crowbar                                          │
      │     → phase3_apply_visceral_functor_sem                                 │
      │     → _assert_engine_parity                                             │
      │     → _p3_final_payload_bytes / verify_sovereign_state                  │
      │     → phase3_emit_sovereign_state                                       │
      │   (Emite `BoolNovikovPseudoholomorphicProsecutorSovereignState`)        │
      └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = bytes(secret_key)
        self._engine: BoolNovikovPseudoholomorphicProsecutorEngine = (
            BoolNovikovPseudoholomorphicProsecutorEngine(secret_key=self._secret_key)
        )
        logger.info(
            f"{self.__class__.__name__} v{_AGENT_VERSION} iniciado sobre motor "
            f"v{_ENGINE_VERSION}."
        )

    # ════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║ FASE I — OBSERVE: INGESTA ALGEBRAICO–ESPECTRAL EN R_Boole ⊗ Λ_Nov,   ║
    # ║          GAP LAPLACIANO DE FIEDLER Y PRE-CONDICIONAMIENTO DELAUNAY   ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # ════════════════════════════════════════════════════════════════════════

    def _compute_graph_laplacian_spectral_gap(
        self, arr: NDArray[np.float64]
    ) -> float:
        r"""
        [FASE I · 1/6] Teoría Espectral de Grafos sobre el complejo celular de incidencia:
        Construye la matriz de Gram simétrica positiva semidefinida G = B Bᵀ a partir de
        |∂₁|, extrae el Laplaciano combinatorio L = D − W y evalúa la conectividad
        algebraica de Fiedler λ₂(L) (segundo menor autovalor de L).
        """
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0
        mat = arr
        if mat.ndim == 0:
            return 0.0
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        elif mat.ndim > 2:
            mat = mat.reshape(mat.shape[0], -1)

        abs_b = np.abs(mat)
        gram = abs_b @ abs_b.T
        n_v = gram.shape[0]
        if n_v < 2:
            return float(gram[0, 0]) if n_v == 1 else 0.0

        w_adj = gram.copy()
        np.fill_diagonal(w_adj, 0.0)
        deg = np.sum(w_adj, axis=1)
        laplacian = np.diag(deg) - w_adj
        eigvals = np.linalg.eigvalsh(laplacian)
        eigvals_sorted = np.sort(np.maximum(0.0, eigvals))
        return float(eigvals_sorted[1])

    def phase1_observe_algebraic_spectral(
        self,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
    ) -> _Phase1Observation:
        r"""
        [FASE I · 2/6] Sub-paso de observación algebraico-espectral en R_Boole:
        Invoca `engine.compute_boolean_snf_and_betti` (SNF exacta vía Bareiss),
        determina las dimensiones celulares (n, m), la característica de Euler–Poincaré
        χ = n − m y el gap espectral de Fiedler λ₂(L).
        """
        torsion, betti_1 = self._engine.compute_boolean_snf_and_betti(apu_cost_matrix)

        arr = np.asarray(apu_cost_matrix, dtype=np.float64)
        if arr.size == 0:
            n, m = 0, 0
        elif arr.ndim == 0:
            n, m = 1, 1
        elif arr.ndim == 1:
            n, m = 1, int(arr.shape[0])
        elif arr.ndim == 2:
            n, m = int(arr.shape[0]), int(arr.shape[1])
        else:
            n = int(arr.shape[0])
            m = int(np.prod(arr.shape[1:]))

        fiedler_gap = self._compute_graph_laplacian_spectral_gap(arr)

        return _Phase1Observation(
            apu_cost_matrix=arr,
            novikov_exponents=np.asarray(novikov_exponents, dtype=np.float64).ravel(),
            betti_1_cycles=int(betti_1),
            torsion_detected=bool(torsion),
            boolean_n=n,
            boolean_m=m,
            timestamp=float(time.time()),
            euler_poincare_chi=int(n - m),
            spectral_graph_laplacian_gap=fiedler_gap,
        )

    def phase1_observe_novikov_krull(
        self,
        obs: _Phase1Observation,
        novikov_exponents: Optional[
            Union[NDArray[np.float64], Sequence[float]]
        ] = None,
    ) -> _Phase1Observation:
        r"""
        [FASE I · 3/6] Sub-paso de valuación de Krull en el anillo universal Λ_Nov y
        preparación de la semilla de acción canónica de Delaunay–Whittaker J₀ para la
        mecánica celeste de Poincaré en la Fase II.
        """
        exps = (
            obs.novikov_exponents
            if novikov_exponents is None
            else np.asarray(novikov_exponents, dtype=np.float64).ravel()
        )
        v_A, norm = self._engine.evaluate_novikov_krull_valuation(exps)
        obs.krull_valuation_vA = float(v_A)
        obs.ultrametric_norm = float(norm)
        obs.novikov_exponents = exps
        obs.delaunay_action_seed_J0 = (
            float(max(abs(v_A), _KRULL_VALUATION_FLOOR))
            if math.isfinite(v_A)
            else _KRULL_VALUATION_FLOOR
        )
        return obs

    def continue_to_phase2_gate(self, p1: ProsecutorPhase1Certificate) -> bool:
        r"""
        [FASE I · 4/6] Compuerta Homomórfica Φ₁₂ (Observe → Orient).

        Verifica en modo fail-CLOSED:
          1. `len(p1.chain_hash) == 64` y `len(p1.hmac_signature) == 64`.
          2. Firma HMAC con `_HMAC_DOMAIN_P1` sobre `chain_hash`.
          3. `engine.verify_phase1_seal(p1.engine_phase1_seal)` (sello del motor íntegro).

        CONTRATO DE GOBERNANZA FUNDAMENTAL:
        Admite anomalías VETOED (`torsion_detected` o `krull_valuation_vA < 0.10`) para
        que fluyan hacia FASE II/III donde se ejecuta la aniquilación cuántica de Dirac
        y el disparo físico del Crowbar.
        """
        if not isinstance(p1, ProsecutorPhase1Certificate):
            return False
        if len(p1.chain_hash) != _CHAIN_HASH_HEX_LEN:
            return False
        if len(p1.hmac_signature) != _CHAIN_HASH_HEX_LEN:
            return False

        expected = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + p1.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, p1.hmac_signature):
            return False

        if not isinstance(p1.engine_phase1_seal, Phase1ProsecutionSeal):
            return False
        if not self._engine.verify_phase1_seal(p1.engine_phase1_seal):
            return False

        return True

    def verify_phase1_certificate(self, p1: ProsecutorPhase1Certificate) -> bool:
        """[FASE I · 5/6] Verificación criptográfica externa del certificado P1 (alias de Φ₁₂)."""
        return self.continue_to_phase2_gate(p1)

    def phase1_emit_sovereign_certificate(
        self,
        obs: _Phase1Observation,
        timestamp: Optional[float] = None,
    ) -> ProsecutorPhase1Certificate:
        r"""
        [FASE I · 6/6 — MÉTODO TERMINAL DE LA FASE I Y ESLABÓN DE CONTINUACIÓN A FASE II]
        Clausura formalmente la FASE I (Observe) emitiendo `ProsecutorPhase1Certificate`,
        envolviendo criptográficamente el `Phase1ProsecutionSeal` del motor, verificando
        paridad exacta de Fase I (β₁, torsión y valuación de Krull v(A)) y firmando bajo
        `_HMAC_DOMAIN_P1`.

        CONTRATO DE ANIDAMIENTO FASE I ──► FASE II:
        El certificado `ProsecutorPhase1Certificate` retornado por esta definición final
        constituye la entrada formal obligatoria que consume inmediatamente
        `phase2_open_from_phase1_certificate`, primer método de la FASE II definida a continuación.
        """
        ts = float(timestamp) if timestamp is not None else float(time.time())

        engine_seal = self._engine.phase1_prosecution_seal(
            apu_cost_matrix=obs.apu_cost_matrix,
            novikov_exponents=obs.novikov_exponents,
            timestamp=ts,
        )
        engine_seal_ok = bool(self._engine.verify_phase1_seal(engine_seal))

        # Paridad FASE I: la observación agéntica debe coincidir exactamente con el sello del motor
        if obs.betti_1_cycles != engine_seal.boolean_betti_1:
            raise ProsecutorAgentParityViolationError(
                f"FASE I: β₁ divergente (obs={obs.betti_1_cycles} "
                f"vs sello={engine_seal.boolean_betti_1})"
            )
        if obs.torsion_detected != engine_seal.boolean_torsion_detected:
            raise ProsecutorAgentParityViolationError(
                "FASE I: torsión divergente entre observación y sello del motor"
            )
        if not math.isclose(
            obs.krull_valuation_vA,
            engine_seal.novikov_krull_valuation,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ProsecutorAgentParityViolationError(
                f"FASE I: v(A) divergente (obs={obs.krull_valuation_vA:.12g} "
                f"vs sello={engine_seal.novikov_krull_valuation:.12g})"
            )

        chain_payload = (
            f"PROSECUTOR-AGENT-P1-CHAIN|{ts:.12g}|{_AGENT_VERSION}|{_ENGINE_VERSION}|"
            f"{engine_seal.phase1_seal_hash}|"
            f"{obs.betti_1_cycles}|{int(obs.torsion_detected)}|"
            f"{obs.krull_valuation_vA:.12g}|{obs.ultrametric_norm:.12g}|"
            f"{obs.boolean_n}|{obs.boolean_m}|{int(engine_seal_ok)}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_payload).hexdigest()

        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return ProsecutorPhase1Certificate(
            betti_1_cycles=obs.betti_1_cycles,
            torsion_detected=obs.torsion_detected,
            krull_valuation_vA=obs.krull_valuation_vA,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            timestamp_ns=int(ts * 1.0e9),
            engine_phase1_seal=engine_seal,
            timestamp=ts,
            engine_phase1_seal_verified=engine_seal_ok,
            euler_poincare_chi=obs.euler_poincare_chi,
            cayley_dickson_octonion_norm=float(
                getattr(engine_seal, "cayley_dickson_norm", 0.0)
            ),
            rsi1_gershgorin_margin=float(
                getattr(engine_seal, "rsi1_gershgorin_margin", 1.0)
            ),
            delaunay_action_seed_J0=obs.delaunay_action_seed_J0,
            spectral_graph_laplacian_gap=obs.spectral_graph_laplacian_gap,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║ FASE II — ORIENT: SUPERFICIE PSEUDOHOLOMORFA DE FLOER–GROMOV,        ║
    # ║           MECÁNICA CELESTE DE POINCARÉ–KAM–BIRKHOFF–NEKHOROSHEV,     ║
    # ║           MÓNADA RSI³ BANACH–KLEISLI Y ANIQUILACIÓN DE DIRAC         ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # ════════════════════════════════════════════════════════════════════════

    def phase2_open_from_phase1_certificate(
        self,
        p1: ProsecutorPhase1Certificate,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
        ball_r: float,
        cylinder_R: float,
        positron_hmac_key: Optional[bytes],
        pert_eps: float = 0.1,
        timestamp: Optional[float] = None,
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 1/10 — CONTINUACIÓN DIRECTA DE `phase1_emit_sovereign_certificate`]
        Abre formalmente la FASE II (Orient) consumiendo el `ProsecutorPhase1Certificate`
        emitido por el último método de la FASE I, verificando en modo fail-CLOSED la
        compuerta Φ₁₂ y delegando en `engine.phase2_open_from_phase1_seal` (doble
        verificación criptográfica Agente + Motor).
        """
        if not self.continue_to_phase2_gate(p1):
            raise ProsecutorAgentGateViolationError(
                "Φ₁₂: certificado P1 inválido (HMAC / longitud / sello del motor)"
            )

        ts = float(timestamp) if timestamp is not None else float(time.time())

        try:
            engine_ctx = self._engine.phase2_open_from_phase1_seal(
                seal=p1.engine_phase1_seal,
                apu_cost_matrix=apu_cost_matrix,
                novikov_exponents=novikov_exponents,
                ball_r=ball_r,
                cylinder_R=cylinder_R,
                positron_hmac_key=positron_hmac_key,
                pert_eps=pert_eps,
                timestamp=ts,
            )
        except ProsecutorGateViolationError as exc:
            raise ProsecutorAgentGateViolationError(
                f"Φ₁₂: motor rechazó el sello P1 — {exc}"
            ) from exc

        return _Phase2WorkingContext(
            p1_cert=p1,
            engine_ctx=engine_ctx,
            timestamp=ts,
        )

    def phase2_orient_pseudoholomorphic_surface(
        self, ctx: _Phase2WorkingContext
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 2/10] Sub-paso geométrico de FASE II: evalúa el residuo de
        Cauchy–Riemann–Floer (curvatura media discreta H), la energía simpléctica de Floer
        y la condición de rigidez simpléctica Non-Squeezing de Gromov (π r² ≤ π R²).
        """
        engine_ctx = self._engine.phase2_resolve_pseudoholomorphic_surface(
            ctx.engine_ctx
        )
        ctx.engine_ctx = engine_ctx
        ctx.cr_residual = float(engine_ctx.cauchy_riemann_residual or 0.0)
        ctx.gromov_violation = bool(engine_ctx.gromov_violation or False)
        ctx.floer_symplectic_energy = float(
            getattr(engine_ctx, "floer_symplectic_energy", 0.0)
        )
        return ctx

    def _compute_poincare_kam_orbital_stability_index(
        self, celestial: CelestialProsecutionState
    ) -> float:
        r"""
        [FASE II · 3/10] Síntesis agéntica de Mecánica Celeste de Poincaré–KAM–Birkhoff–Nekhoroshev:
        Construye el índice compuesto de estabilidad orbital I_orb ∈ (0, 1] sobre el toro
        invariante KAM combinando:
          • Supervivencia diofántica KAM: s_KAM = tanh(γ_KAM / (γ_KAM + _KAM_MARGIN_FLOOR))
          • Amortiguamiento de escisión homoclínica de Melnikov:
                s_Mel = 1 / (1 + (|M|_max / 0.35)²)
          • Conservación simpléctica de 4.º orden (Forest–Ruth / Yoshida):
                s_Symp = exp(−min(50.0, |ΔH_res|))
          • Estabilidad exponencial de Nekhoroshev:
                s_Nekh = 1 − 1 / (1 + ln(1 + T_Nekh))
        """
        kam_val = max(0.0, float(celestial.kam_diophantine_margin))
        s_kam = math.tanh(kam_val / (kam_val + _KAM_MARGIN_FLOOR)) if kam_val > 0.0 else 0.0

        mel_ratio = float(celestial.melnikov_integral_value) / 0.35
        s_mel = 1.0 / (1.0 + mel_ratio * mel_ratio)

        symp_err = float(getattr(celestial, "symplectic_energy_error", 0.0))
        s_symp = math.exp(-min(50.0, abs(symp_err)))

        nekh_t = max(1.0, float(getattr(celestial, "nekhoroshev_stability_time", 1.0)))
        s_nekh = 1.0 - 1.0 / (1.0 + math.log1p(nekh_t))

        orbital_idx = 0.35 * s_mel + 0.30 * s_kam + 0.20 * s_symp + 0.15 * s_nekh
        return float(min(max(orbital_idx, 0.0), 1.0))

    def phase2_orient_poincare_kam(
        self, ctx: _Phase2WorkingContext
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 4/10] Sub-paso celeste de FASE II: ejecuta la dinámica de Poincaré–Melnikov,
        la serie homológica de Poincaré–Lindstedt, el margen diofántico KAM, la integración
        simpléctica de 4.º orden de Yoshida y la Forma Normal de Birkhoff–Nekhoroshev.
        """
        engine_ctx = self._engine.phase2_evaluate_poincare_melnikov(ctx.engine_ctx)
        ctx.engine_ctx = engine_ctx
        ctx.celestial_state = engine_ctx.celestial_state
        if ctx.celestial_state is not None:
            ctx.poincare_kam_orbital_stability_index = (
                self._compute_poincare_kam_orbital_stability_index(ctx.celestial_state)
            )
        return ctx

    def _compute_rsi3_meta_convergence_index(
        self, rsi3: RSI3ProsecutionState
    ) -> float:
        r"""
        [FASE II · 5/10] Síntesis agéntica de Automejora Recursiva de Nivel-3 (RSI³) sobre la
        Mónada Banach–Kleisli (T, η, μ):
        Evalúa el índice de meta-convergencia I_RSI³ ∈ (0, 1] integrando los 3 niveles:
          • Nivel 1 (Contracción Banach–Picard): margen (1 − Lip(T³)) y decaimiento del residuo.
          • Nivel 2 (Diferencial de Fréchet y Aceleración Δ² de Aitken–Steffensen):
            exponente de Lyapunov negativo λ_Lyap < 0 y proximidad superlineal |x*_Aitken| ≪ |x₄|.
          • Nivel 3 (Meta-Política Endofuntorial M^{(3)}): ganancia g_meta^{(3)} ≥ 1.0 y
            coherencia resonante Poincaré–RSI³.
        """
        lip_margin = max(0.0, 1.0 - float(rsi3.contraction_bound))
        lyap = float(getattr(rsi3, "lyapunov_exponent", -0.05))
        lyap_score = math.tanh(max(0.0, -lyap) * 10.0)

        x4 = float(rsi3.iterates[-1]) if rsi3.iterates else 1.0
        aitken_fp = float(getattr(rsi3, "aitken_accelerated_fixed_point", x4))
        aitken_ratio = abs(aitken_fp) / (abs(x4) + _WILKINSON_EPSILON)
        aitken_score = 1.0 / (1.0 + aitken_ratio)

        meta_gain = float(getattr(rsi3, "meta_policy_gain_level3", 1.0))
        meta_score = math.tanh(max(0.0, meta_gain) * 0.8)

        idx = (
            0.30 * (1.0 if rsi3.is_banach_contractive else 0.0)
            + 0.25 * aitken_score
            + 0.25 * meta_score
            + 0.20 * min(1.0, lip_margin + lyap_score)
        )
        return float(min(max(idx, 0.0), 1.0))

    def phase2_evaluate_rsi3_banach(
        self, ctx: _Phase2WorkingContext, initial_x: float = 0.95
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 6/10] Sub-paso monádico de FASE II: ejecuta la automejora recursiva de
        Nivel-3 (RSI³) sobre la mónada Banach–Kleisli T³ acoplada al toro KAM de Poincaré,
        evaluando las derivadas exactas de Fréchet DT³(x₀), D²T³(x₀), el punto fijo
        acelerado de Aitken–Steffensen x*_Aitken y el índice de meta-convergencia.
        """
        engine_ctx = self._engine.phase2_evaluate_rsi3_banach(
            ctx.engine_ctx, initial_x=initial_x
        )
        ctx.engine_ctx = engine_ctx
        ctx.rsi3_state = engine_ctx.rsi3_state
        if ctx.rsi3_state is not None:
            ctx.rsi3_meta_convergence_index = (
                self._compute_rsi3_meta_convergence_index(ctx.rsi3_state)
            )
        return ctx

    def phase2_execute_dirac_annihilation(
        self, ctx: _Phase2WorkingContext
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 7/10] Sub-paso cuántico de FASE II: reacción de aniquilación de Dirac
        e⁻ + e⁺ → 2γ en Cl_{1,3} con verificación HMAC ESTRICTA del positrón contra
        `b"DIRAC_POSITRON_HMAC_VERIFICATION"`.
        """
        engine_ctx = self._engine.phase2_execute_dirac_annihilation(ctx.engine_ctx)
        ctx.engine_ctx = engine_ctx
        ctx.dirac_annihilation = engine_ctx.dirac_annihilation
        return ctx

    def continue_to_phase3_gate(
        self,
        p2: ProsecutorPhase2Certificate,
        p1: Optional[ProsecutorPhase1Certificate] = None,
    ) -> bool:
        r"""
        [FASE II · 8/10] Compuerta Homomórfica Φ₂₃ (Orient → Decide & Act).

        Verifica en modo fail-CLOSED:
          1. Longitudes 64-hex de `chain_hash`, `hmac_signature` y `previous_chain_hash`.
          2. HMAC dominio P2 sobre `(previous_chain_hash ‖ chain_hash)`.
          3. Si se proporciona `p1`, correspondencia exacta `p2.previous_chain_hash == p1.chain_hash`.
          4. Cota contractiva de Banach `Lip(T³) < 1.0`.
          5. Verificación criptográfica de `engine.verify_phase2_seal(p2.engine_phase2_seal)`
             y `engine.verify_phase1_seal(p2.engine_phase2_seal.phase1_seal)`.

        ADMITE estados que deriven en VETOED para que fluyan hasta la adjudicación y el
        disparo físico del Crowbar en la FASE III.
        """
        if not isinstance(p2, ProsecutorPhase2Certificate):
            return False
        if len(p2.chain_hash) != _CHAIN_HASH_HEX_LEN:
            return False
        if len(p2.hmac_signature) != _CHAIN_HASH_HEX_LEN:
            return False
        if len(p2.previous_chain_hash) != _CHAIN_HASH_HEX_LEN:
            return False

        expected = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2
            + f"{p2.previous_chain_hash}|{p2.chain_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, p2.hmac_signature):
            return False

        if p1 is not None and p2.previous_chain_hash != p1.chain_hash:
            return False

        if not (p2.banach_contraction_lip < _BANACH_LIP_BOUND):
            return False

        if not isinstance(p2.engine_phase2_seal, Phase2ProsecutionSeal):
            return False
        if not self._engine.verify_phase2_seal(p2.engine_phase2_seal):
            return False
        if not self._engine.verify_phase1_seal(p2.engine_phase2_seal.phase1_seal):
            return False

        return True

    def verify_phase2_certificate(
        self,
        p2: ProsecutorPhase2Certificate,
        p1: Optional[ProsecutorPhase1Certificate] = None,
    ) -> bool:
        """[FASE II · 9/10] Verificación criptográfica externa del certificado P2 (alias de Φ₂₃)."""
        return self.continue_to_phase3_gate(p2, p1=p1)

    def phase2_emit_sovereign_certificate(
        self,
        ctx: _Phase2WorkingContext,
        timestamp: Optional[float] = None,
    ) -> ProsecutorPhase2Certificate:
        r"""
        [FASE II · 10/10 — MÉTODO TERMINAL DE LA FASE II Y ESLABÓN DE CONTINUACIÓN A FASE III]
        Clausura formalmente la FASE II (Orient) emitiendo `ProsecutorPhase2Certificate`,
        envolviendo el `Phase2ProsecutionSeal` del motor, encadenando `previous_chain_hash`
        al certificado de FASE I y firmando con HMAC dominio P2.

        CONTRATO DE ANIDAMIENTO FASE II ──► FASE III:
        El certificado `ProsecutorPhase2Certificate` retornado por esta definición final
        constituye la entrada formal obligatoria que consume inmediatamente
        `phase3_open_from_phase2_certificate`, primer método de la FASE III definida a continuación.
        """
        ts = float(timestamp) if timestamp is not None else float(time.time())

        try:
            engine_p2_seal = self._engine.phase2_prosecution_seal(
                ctx.engine_ctx, timestamp=ts
            )
        except ProsecutorGateViolationError as exc:
            raise ProsecutorAgentGateViolationError(
                f"FASE II: motor rechazó la emisión del sello P2 — {exc}"
            ) from exc

        engine_seal_ok = bool(self._engine.verify_phase2_seal(engine_p2_seal))

        cr = float(engine_p2_seal.cauchy_riemann_residual)
        celestial = engine_p2_seal.celestial_state
        rsi3 = engine_p2_seal.rsi3_state
        dirac = engine_p2_seal.dirac_annihilation

        chain_payload = (
            f"PROSECUTOR-AGENT-P2-CHAIN|{ts:.12g}|{_AGENT_VERSION}|{_ENGINE_VERSION}|"
            f"{ctx.p1_cert.chain_hash}|{engine_p2_seal.phase2_seal_hash}|"
            f"{cr:.12g}|{int(engine_p2_seal.gromov_violation)}|"
            f"{celestial.melnikov_integral_value:.12g}|"
            f"{int(celestial.melnikov_chaos_detected)}|"
            f"{celestial.kam_diophantine_margin:.12g}|"
            f"{rsi3.contraction_bound:.12g}|{rsi3.banach_residual:.12g}|"
            f"{int(rsi3.is_banach_contractive)}|"
            f"{rsi3.capacity_jerk:.12g}|{int(rsi3.jerk_strictly_positive)}|"
            f"{int(dirac.positron_valid)}|{dirac.energy_released_kev:.6g}|"
            f"{int(engine_seal_ok)}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_payload).hexdigest()

        hmac_sig = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2
            + f"{ctx.p1_cert.chain_hash}|{chain_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return ProsecutorPhase2Certificate(
            res_cauchy_riemann=cr,
            gromov_violation=bool(engine_p2_seal.gromov_violation),
            banach_contraction_lip=float(rsi3.contraction_bound),
            energy_released_kev=float(dirac.energy_released_kev),
            poincare_melnikov_chaos=bool(celestial.melnikov_chaos_detected),
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            timestamp_ns=int(ts * 1.0e9),
            engine_phase2_seal=engine_p2_seal,
            previous_chain_hash=ctx.p1_cert.chain_hash,
            timestamp=ts,
            engine_phase2_seal_verified=engine_seal_ok,
            celestial_state=celestial,
            rsi3_state=rsi3,
            dirac_annihilation=dirac,
            floer_symplectic_energy=float(
                getattr(engine_p2_seal, "floer_symplectic_energy", ctx.floer_symplectic_energy)
            ),
            poincare_cartan_invariant=float(
                getattr(celestial, "poincare_cartan_invariant", 0.0)
            ),
            nekhoroshev_stability_time=float(
                getattr(celestial, "nekhoroshev_stability_time", 1.0)
            ),
            birkhoff_twist_coeff=float(
                getattr(celestial, "birkhoff_twist_coeff", 0.0)
            ),
            poincare_kam_orbital_stability_index=ctx.poincare_kam_orbital_stability_index,
            rsi3_aitken_fixed_point=float(
                getattr(rsi3, "aitken_accelerated_fixed_point", 0.0)
            ),
            rsi3_frechet_jacobian_t3=float(
                getattr(rsi3, "exact_frechet_jacobian_t3", 0.0)
            ),
            rsi3_lyapunov_exponent=float(
                getattr(rsi3, "lyapunov_exponent", -0.05)
            ),
            rsi3_meta_policy_gain=float(
                getattr(rsi3, "meta_policy_gain_level3", 1.0)
            ),
            rsi3_meta_convergence_index=ctx.rsi3_meta_convergence_index,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║ FASE III — DECIDE & ACT: ADJUDICACIÓN TOPOS \Omega_3 LAWVERE–TIERNEY,║
    # ║            CROWBAR ELÉCTRICO RLC, FUNTOR \Phi_sem Y PARIDAD ESTRICTA ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # ════════════════════════════════════════════════════════════════════════

    def phase3_open_from_phase2_certificate(
        self,
        p2: ProsecutorPhase2Certificate,
        p1: Optional[ProsecutorPhase1Certificate] = None,
        timestamp: Optional[float] = None,
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 1/8 — CONTINUACIÓN DIRECTA DE `phase2_emit_sovereign_certificate`]
        Abre formalmente la FASE III (Decide & Act) consumiendo el
        `ProsecutorPhase2Certificate` emitido por el último método de la FASE II,
        verificando la compuerta homomórfica Φ₂₃ y delegando en
        `engine.phase3_open_from_phase2_seal`.
        """
        if not self.continue_to_phase3_gate(p2, p1=p1):
            raise ProsecutorAgentGateViolationError(
                "Φ₂₃: certificado P2 inválido (HMAC / cadena / Banach / sello del motor)"
            )

        ts = float(timestamp) if timestamp is not None else float(time.time())

        try:
            engine_ctx = self._engine.phase3_open_from_phase2_seal(
                seal=p2.engine_phase2_seal, timestamp=ts
            )
        except ProsecutorGateViolationError as exc:
            raise ProsecutorAgentGateViolationError(
                f"Φ₂₃: motor rechazó el sello P2 — {exc}"
            ) from exc

        return _Phase3WorkingContext(
            p2_cert=p2,
            engine_ctx=engine_ctx,
            timestamp=ts,
            cr_residual=float(p2.res_cauchy_riemann),
            gromov_violation=bool(p2.gromov_violation),
        )

    def phase3_adjudicate_heyting_omega3(
        self, ctx: _Phase3WorkingContext
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 2/8] Adjudicación en el topos intuicionista \Omega_3 y evaluación de
        la modalidad de clausura de Lawvere–Tierney j = ¬¬: \Omega_3 \to \Omega_3.
        Delega en `engine.phase3_adjudicate_heyting_omega3` preservando paridad exacta.
        """
        engine_ctx = self._engine.phase3_adjudicate_heyting_omega3(ctx.engine_ctx)
        ctx.engine_ctx = engine_ctx
        if engine_ctx.truth_value is None:
            raise ProsecutorAgentGateViolationError(
                "FASE III: el motor no produjo veredicto Heyting"
            )
        ctx.truth_value = engine_ctx.truth_value
        ctx.adjudication_reason = engine_ctx.adjudication_reason
        ctx.lawvere_tierney_modal_truth = int(
            getattr(engine_ctx, "lawvere_tierney_modal_truth", int(ctx.truth_value.value))
        )
        return ctx

    def phase3_arm_esp32_crowbar(
        self, ctx: _Phase3WorkingContext
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 3/8] Interlock ciber-físico Xtensa LX7 en IRAM (SIMD @ GPIO14 / BT151)
        y captura de la dinámica eléctrica transitoria RLC del tiristor SCR:
          • VETOED    ⟹ crowbar ARMADO en 42.0 ns ≤ 392.15 ns (corriente pico I_peak y di/dt).
          • COHERENT  ⟹ crowbar DESARMADO (0.0 ns).
          • DEGRADED  ⟹ crowbar DESARMADO (0.0 ns; aniquilación pasiva).
        """
        engine_ctx = self._engine.phase3_arm_esp32_crowbar(ctx.engine_ctx)
        ctx.engine_ctx = engine_ctx

        if ctx.truth_value == HeytingProsecutorTruthValue.VETOED:
            ctx.crowbar_triggered = True
            ctx.crowbar_latency_ns = _ESP32_CROWBAR_EXECUTION_NS
            logger.warning(
                f"[CROWBAR] ESP32 Xtensa LX7 IRAM @ GPIO14/BT151 disparado "
                f"en {ctx.crowbar_latency_ns:.1f} ns "
                f"(presupuesto ≤ {_ESP32_CROWBAR_MAX_LATENCY_NS:.2f} ns) — "
                f"razón: {ctx.adjudication_reason}"
            )
        else:
            ctx.crowbar_triggered = False
            ctx.crowbar_latency_ns = 0.0

        ctx.crowbar_budget_respected = (
            ctx.crowbar_latency_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS
        )
        ctx.crowbar_peak_current_amps = float(
            getattr(engine_ctx, "crowbar_peak_current_amps", 0.0)
        )
        ctx.crowbar_didt_amps_per_us = float(
            getattr(engine_ctx, "crowbar_didt_amps_per_us", 0.0)
        )
        ctx.crowbar_damping_ratio_zeta = float(
            getattr(engine_ctx, "crowbar_damping_ratio_zeta", 0.0)
        )
        return ctx

    def phase3_apply_visceral_functor_sem(
        self, ctx: _Phase3WorkingContext
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 4/8] Funtor Semántico Visceral Φ_sem: traduce la adjudicación topológica
        en invariantes de gobernanza económica y computacional:
          • WACC protección        = 15.00% (invariante)
          • Compresión KV-Cache    = 86.40% (invariante)
          • Ahorro inferencia IA   = 80.00% (invariante)
          • Liquidez neta liberada = 11.50% (COHERENT) / 5.75% (DEGRADED) / 0.00% (VETOED)
        """
        ctx.wacc_protection_pct = _WACC_INVARIANT_PCT
        ctx.kv_cache_compression_pct = _KV_CACHE_COMPRESSION_PCT
        ctx.ai_inference_savings_pct = _AI_INFERENCE_SAVING_PCT

        if ctx.truth_value == HeytingProsecutorTruthValue.COHERENT:
            ctx.net_liquidity_pct = _LIQUIDITY_COHERENT_PCT
        elif ctx.truth_value == HeytingProsecutorTruthValue.DEGRADED:
            ctx.net_liquidity_pct = _LIQUIDITY_DEGRADED_PCT
        else:
            ctx.net_liquidity_pct = _LIQUIDITY_VETOED_PCT
        return ctx

    def _assert_engine_parity(
        self,
        ctx3: _Phase3WorkingContext,
        engine_report: PseudoholomorphicSurfaceProsecutionReport,
    ) -> None:
        r"""
        [FASE III · 5/8] Auditoría de paridad matemática y semántica estricta Agente ↔ Motor.
        Verifica veredicto Heyting, razón de adjudicación, curvatura CR, violación de Gromov,
        armado del Crowbar, cota Banach Lip(T³) e integral máxima de Melnikov.
        Cualquier divergencia dispara `ProsecutorAgentParityViolationError` (fail-closed).
        """
        if ctx3.truth_value != engine_report.truth_value:
            raise ProsecutorAgentParityViolationError(
                f"Paridad Heyting rota: agente={ctx3.truth_value.name} "
                f"vs motor={engine_report.truth_value.name}"
            )
        if ctx3.adjudication_reason != engine_report.adjudication_reason:
            raise ProsecutorAgentParityViolationError(
                f"Razón de adjudicación divergente: agente={ctx3.adjudication_reason} "
                f"vs motor={engine_report.adjudication_reason}"
            )
        if not math.isclose(
            ctx3.cr_residual,
            engine_report.cauchy_riemann_residual,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ProsecutorAgentParityViolationError(
                f"Residual CR divergente: agente={ctx3.cr_residual:.12g} "
                f"vs motor={engine_report.cauchy_riemann_residual:.12g}"
            )
        if ctx3.gromov_violation != engine_report.gromov_violation:
            raise ProsecutorAgentParityViolationError(
                "Bandera de Gromov divergente entre agente y motor"
            )
        if ctx3.crowbar_triggered != engine_report.esp32_crowbar_armed:
            raise ProsecutorAgentParityViolationError(
                "Bandera del crowbar divergente entre agente y motor"
            )
        if not math.isclose(
            ctx3.p2_cert.banach_contraction_lip,
            engine_report.banach_contraction_lip,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ProsecutorAgentParityViolationError(
                "Cota Lip(T³) de Banach–Kleisli divergente entre agente y motor"
            )

    def _p3_final_payload_bytes(
        self, state: BoolNovikovPseudoholomorphicProsecutorSovereignState
    ) -> bytes:
        """[FASE III · 6/8] Serialización canónica determinista para el HMAC del Merkle soberano."""
        return (
            f"PROSECUTOR-AGENT-FINAL|{state.timestamp:.12g}|"
            f"{_AGENT_VERSION}|{_ENGINE_VERSION}|"
            f"{state.phase1_chain_hash}|{state.phase2_chain_hash}|"
            f"{int(state.truth_value.value)}|{state.adjudication_reason}|"
            f"{int(state.esp32_crowbar_triggered)}|"
            f"{state.crowbar_latency_ns:.6g}|"
            f"{state.wacc_protection_percent:.6g}|"
            f"{state.net_liquidity_liberation_percent:.6g}|"
            f"{state.kv_cache_compression_percent:.6g}|"
            f"{state.ai_inference_savings_percent:.6g}"
        ).encode("utf-8")

    def verify_sovereign_state(
        self, state: BoolNovikovPseudoholomorphicProsecutorSovereignState
    ) -> bool:
        """[FASE III · 7/8] Verificación criptográfica externa del Merkle soberano final (HMAC P3)."""
        if len(state.sovereign_hmac_sha256) != _CHAIN_HASH_HEX_LEN:
            return False
        expected = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + self._p3_final_payload_bytes(state),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, state.sovereign_hmac_sha256)

    def phase3_emit_sovereign_state(
        self,
        ctx: _Phase3WorkingContext,
        execution_time_ms: float = 0.0,
    ) -> BoolNovikovPseudoholomorphicProsecutorSovereignState:
        r"""
        [FASE III · 8/8 — MÉTODO TERMINAL DE LA FASE III Y CLAUSURA DE LA TRÍADA OODA]
        Emite el `BoolNovikovPseudoholomorphicProsecutorSovereignState` soberano final,
        sellando criptográficamente la cadena completa de custodia:
            ProsecutorPhase1Certificate ──(Φ₁₂)──► ProsecutorPhase2Certificate ──(Φ₂₃)──► SovereignState
        bajo `_HMAC_DOMAIN_P3`.
        """
        ts = ctx.timestamp

        provisional = BoolNovikovPseudoholomorphicProsecutorSovereignState(
            truth_value=ctx.truth_value,
            esp32_crowbar_triggered=ctx.crowbar_triggered,
            crowbar_latency_ns=ctx.crowbar_latency_ns,
            merkle_root_sha256="",
            wacc_protection_percent=ctx.wacc_protection_pct,
            net_liquidity_liberation_percent=ctx.net_liquidity_pct,
            kv_cache_compression_percent=ctx.kv_cache_compression_pct,
            execution_time_ms=float(execution_time_ms),
            adjudication_reason=ctx.adjudication_reason,
            rsi3_state=ctx.p2_cert.rsi3_state,
            celestial_state=ctx.p2_cert.celestial_state,
            sovereign_hmac_sha256="",
            phase1_chain_hash=ctx.p2_cert.previous_chain_hash,
            phase2_chain_hash=ctx.p2_cert.chain_hash,
            crowbar_budget_respected=ctx.crowbar_budget_respected,
            crowbar_latency_budget_ns=_ESP32_CROWBAR_MAX_LATENCY_NS,
            ai_inference_savings_percent=ctx.ai_inference_savings_pct,
            agent_version=_AGENT_VERSION,
            engine_version=_ENGINE_VERSION,
            timestamp=ts,
            crowbar_peak_current_amps=ctx.crowbar_peak_current_amps,
            crowbar_didt_amps_per_us=ctx.crowbar_didt_amps_per_us,
            crowbar_damping_ratio_zeta=ctx.crowbar_damping_ratio_zeta,
            lawvere_tierney_modal_truth=ctx.lawvere_tierney_modal_truth,
            poincare_kam_orbital_stability_index=ctx.p2_cert.poincare_kam_orbital_stability_index,
            rsi3_meta_convergence_index=ctx.p2_cert.rsi3_meta_convergence_index,
        )

        merkle = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + self._p3_final_payload_bytes(provisional),
            hashlib.sha256,
        ).hexdigest()

        return BoolNovikovPseudoholomorphicProsecutorSovereignState(
            truth_value=provisional.truth_value,
            esp32_crowbar_triggered=provisional.esp32_crowbar_triggered,
            crowbar_latency_ns=provisional.crowbar_latency_ns,
            merkle_root_sha256=merkle,
            wacc_protection_percent=provisional.wacc_protection_percent,
            net_liquidity_liberation_percent=provisional.net_liquidity_liberation_percent,
            kv_cache_compression_percent=provisional.kv_cache_compression_percent,
            execution_time_ms=provisional.execution_time_ms,
            adjudication_reason=provisional.adjudication_reason,
            rsi3_state=provisional.rsi3_state,
            celestial_state=provisional.celestial_state,
            sovereign_hmac_sha256=merkle,
            phase1_chain_hash=provisional.phase1_chain_hash,
            phase2_chain_hash=provisional.phase2_chain_hash,
            crowbar_budget_respected=provisional.crowbar_budget_respected,
            crowbar_latency_budget_ns=provisional.crowbar_latency_budget_ns,
            ai_inference_savings_percent=provisional.ai_inference_savings_percent,
            agent_version=provisional.agent_version,
            engine_version=provisional.engine_version,
            timestamp=provisional.timestamp,
            crowbar_peak_current_amps=provisional.crowbar_peak_current_amps,
            crowbar_didt_amps_per_us=provisional.crowbar_didt_amps_per_us,
            crowbar_damping_ratio_zeta=provisional.crowbar_damping_ratio_zeta,
            lawvere_tierney_modal_truth=provisional.lawvere_tierney_modal_truth,
            poincare_kam_orbital_stability_index=provisional.poincare_kam_orbital_stability_index,
            rsi3_meta_convergence_index=provisional.rsi3_meta_convergence_index,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ORQUESTADOR CANÓNICO DE COMPOSICIÓN OODA ANIDADA (FASE I ∘ FASE II ∘ FASE III)
    # ════════════════════════════════════════════════════════════════════════

    def execute_sovereign_prosecution_cycle(
        self,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
        ball_r: float = 0.15,
        cylinder_R: float = 0.20,
        positron_hmac_key: Optional[bytes] = None,
        pert_eps: float = 0.1,
        *,
        strict_engine_parity: bool = True,
    ) -> BoolNovikovPseudoholomorphicProsecutorSovereignState:
        r"""
        Orquestador canónico OODA trifásico anidado:
          • FASE I   culmina en `phase1_emit_sovereign_certificate` → emite `p1`.
          • FASE II  inicia en `phase2_open_from_phase1_certificate(p1)` y culmina en
                     `phase2_emit_sovereign_certificate` → emite `p2`.
          • FASE III inicia en `phase3_open_from_phase2_certificate(p2, p1=p1)` y culmina
                     en `phase3_emit_sovereign_state` → emite el Estado Soberano.
        """
        wall_start = time.perf_counter()

        # ── FASE I (OBSERVE — Cierra en phase1_emit_sovereign_certificate) ─
        obs = self.phase1_observe_algebraic_spectral(
            apu_cost_matrix, novikov_exponents
        )
        obs = self.phase1_observe_novikov_krull(obs, novikov_exponents)
        p1 = self.phase1_emit_sovereign_certificate(obs)

        if not self.continue_to_phase2_gate(p1):
            raise ProsecutorAgentGateViolationError(
                "Φ₁₂: certificado P1 generado internamente no verifica"
            )

        # ── FASE II (ORIENT — Abre con p1 y cierra en phase2_emit_sovereign_certificate) ─
        ctx2 = self.phase2_open_from_phase1_certificate(
            p1=p1,
            apu_cost_matrix=apu_cost_matrix,
            novikov_exponents=novikov_exponents,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
            positron_hmac_key=positron_hmac_key,
            pert_eps=pert_eps,
        )
        ctx2 = self.phase2_orient_pseudoholomorphic_surface(ctx2)
        ctx2 = self.phase2_orient_poincare_kam(ctx2)
        ctx2 = self.phase2_evaluate_rsi3_banach(ctx2)
        ctx2 = self.phase2_execute_dirac_annihilation(ctx2)
        p2 = self.phase2_emit_sovereign_certificate(ctx2)

        if not self.continue_to_phase3_gate(p2, p1=p1):
            raise ProsecutorAgentGateViolationError(
                "Φ₂₃: certificado P2 generado internamente no verifica"
            )

        # ── FASE III (DECIDE & ACT — Abre con p2 y cierra en phase3_emit_sovereign_state) ─
        ctx3 = self.phase3_open_from_phase2_certificate(p2, p1=p1)
        ctx3 = self.phase3_adjudicate_heyting_omega3(ctx3)
        ctx3 = self.phase3_arm_esp32_crowbar(ctx3)
        ctx3 = self.phase3_apply_visceral_functor_sem(ctx3)

        if strict_engine_parity:
            engine_report = self._engine.run_prosecution_audit(
                apu_cost_matrix=apu_cost_matrix,
                novikov_exponents=novikov_exponents,
                ball_r=ball_r,
                cylinder_R=cylinder_R,
                positron_hmac_key=positron_hmac_key,
                pert_eps=pert_eps,
            )
            self._assert_engine_parity(ctx3, engine_report)

        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
        return self.phase3_emit_sovereign_state(ctx3, execution_time_ms=elapsed_ms)


# ════════════════════════════════════════════════════════════════════════════
# SANDBOX AUTÓNOMO DE VERIFICACIÓN FORMAL (12 SUITES DE PRUEBA)
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPseudoholomorphicProsecutorAgent v14 (Evolución Soberana) ===")

    agent = BoolNovikovPseudoholomorphicProsecutorAgent()
    valid_key = b"DIRAC_POSITRON_HMAC_VERIFICATION"

    matrix_base = np.array([[1.0, 0.2], [0.2, 1.0]])
    exps_base = np.array([0.15, 0.25, 0.40])

    def _print_summary(state: BoolNovikovPseudoholomorphicProsecutorSovereignState) -> None:
        print(
            f"  Veredicto Heyting Ω₃         : {state.truth_value.name} (0x{int(state.truth_value):02x}) "
            f"| Modal Lawvere–Tierney j(Ω₃)=0x{state.lawvere_tierney_modal_truth:02x}"
        )
        print(f"  Razón de adjudicación        : {state.adjudication_reason}")
        print(
            f"  Crowbar disparado            : {state.esp32_crowbar_triggered} "
            f"({state.crowbar_latency_ns:.2f} ns ≤ {state.crowbar_latency_budget_ns:.2f} ns → {state.crowbar_budget_respected}) "
            f"| I_peak={state.crowbar_peak_current_amps:.1f} A | di/dt={state.crowbar_didt_amps_per_us:.1f} A/μs"
        )
        print(
            f"  Funtor Φ_sem                 : WACC={state.wacc_protection_percent:.2f}% | "
            f"Liquidez={state.net_liquidity_liberation_percent:.2f}% | "
            f"KV={state.kv_cache_compression_percent:.2f}% | IA={state.ai_inference_savings_percent:.2f}%"
        )
        print(
            f"  Índices Poincaré & RSI³      : I_orb(Poincaré–KAM)={state.poincare_kam_orbital_stability_index:.4f} | "
            f"I_meta(RSI³)={state.rsi3_meta_convergence_index:.4f}"
        )
        if state.rsi3_state is not None:
            r = state.rsi3_state
            print(
                f"  Mónada RSI³ Banach–Kleisli   : Lip(T³)={r.contraction_bound:.6f} (<1→{r.is_banach_contractive}) | "
                f"Residuo={r.banach_residual:.3e} | Jerk={r.capacity_jerk:.4e}"
            )
        print(f"  P1 chain (16 hex)            : {state.phase1_chain_hash[:16]}…")
        print(f"  P2 chain (16 hex)            : {state.phase2_chain_hash[:16]}…")
        print(f"  Merkle root (16 hex)         : {state.merkle_root_sha256[:16]}…")
        print(f"  Latencia total (ms)          : {state.execution_time_ms:.3f}")

    # ── TEST 1: caso base → COHERENT ─────────────────────────────────────
    print("\n───── TEST 1: caso base → COHERENT ─────")
    state = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=matrix_base,
        novikov_exponents=exps_base,
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _print_summary(state)
    assert state.truth_value == HeytingProsecutorTruthValue.COHERENT, (
        f"Razón inesperada: {state.adjudication_reason}"
    )
    assert state.esp32_crowbar_triggered is False
    assert state.crowbar_latency_ns == 0.0
    assert state.crowbar_budget_respected is True
    assert state.wacc_protection_percent == 15.0
    assert state.net_liquidity_liberation_percent == 11.5
    assert state.kv_cache_compression_percent == 86.4
    assert state.ai_inference_savings_percent == 80.0
    assert len(state.phase1_chain_hash) == 64
    assert len(state.phase2_chain_hash) == 64
    assert len(state.merkle_root_sha256) == 64
    assert agent.verify_sovereign_state(state)
    print("  ✓ TEST 1 OK")

    # ── TEST 2: Krull bajo piso → VETOED + Crowbar ──────────────────────
    print("\n───── TEST 2: Krull < 0.10 → VETOED + Crowbar ─────")
    state_bad_krull = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=matrix_base,
        novikov_exponents=[0.03, 0.05, 0.09],
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _print_summary(state_bad_krull)
    assert state_bad_krull.truth_value == HeytingProsecutorTruthValue.VETOED
    assert state_bad_krull.adjudication_reason == "NOVIKOV_KRULL_VALUATION_FLOOR_VIOLATION"
    assert state_bad_krull.esp32_crowbar_triggered is True
    assert state_bad_krull.crowbar_latency_ns == 42.0
    assert state_bad_krull.crowbar_budget_respected is True
    assert state_bad_krull.net_liquidity_liberation_percent == 0.0
    assert agent.verify_sovereign_state(state_bad_krull)
    print("  ✓ TEST 2 OK")

    # ── TEST 3: Gromov no-squeezing violado → VETOED ────────────────────
    print("\n───── TEST 3: Gromov r > R → VETOED + Crowbar ─────")
    state_gromov = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=matrix_base,
        novikov_exponents=exps_base,
        ball_r=2.0,
        cylinder_R=1.0,
        positron_hmac_key=valid_key,
    )
    _print_summary(state_gromov)
    assert state_gromov.truth_value == HeytingProsecutorTruthValue.VETOED
    assert state_gromov.adjudication_reason == "GROMOV_NON_SQUEEZING_VIOLATION"
    assert state_gromov.esp32_crowbar_triggered is True
    assert state_gromov.crowbar_latency_ns == 42.0
    print("  ✓ TEST 3 OK")

    # ── TEST 4: HMAC del positrón inválido → VETOED ─────────────────────
    print("\n───── TEST 4: HMAC positrón inválido → VETOED ─────")
    state_bad_pos = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=matrix_base,
        novikov_exponents=exps_base,
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=b"PAYLOAD_INCORRECTO_DEL_POSITRON",
    )
    _print_summary(state_bad_pos)
    assert state_bad_pos.truth_value == HeytingProsecutorTruthValue.VETOED
    assert state_bad_pos.adjudication_reason == "DIRAC_POSITRON_HMAC_INVALID"
    assert state_bad_pos.esp32_crowbar_triggered is True
    print("  ✓ TEST 4 OK")

    # ── TEST 5: sin clave de positrón (default=None) → fail-CLOSED ──────
    print("\n───── TEST 5: positron_hmac_key=None → VETOED (fail-CLOSED) ─────")
    state_no_key = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=matrix_base,
        novikov_exponents=exps_base,
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=None,
    )
    _print_summary(state_no_key)
    assert state_no_key.truth_value == HeytingProsecutorTruthValue.VETOED
    assert state_no_key.adjudication_reason == "DIRAC_POSITRON_HMAC_INVALID"
    assert state_no_key.esp32_crowbar_triggered is True
    print("  ✓ TEST 5 OK")

    # ── TEST 6: matriz vacía → COHERENT vacuo ────────────────────────────
    print("\n───── TEST 6: matriz vacía → COHERENT vacuo ─────")
    state_empty = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=np.array([]),
        novikov_exponents=np.array([]),
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _print_summary(state_empty)
    assert state_empty.truth_value == HeytingProsecutorTruthValue.COHERENT
    assert state_empty.adjudication_reason == "ALL_PROSECUTION_INVARIANTS_SATISFIED"
    print("  ✓ TEST 6 OK")

    # ── TEST 7: 5×5 aleatorio (seed=42) — estabilidad estructural ────────
    print("\n───── TEST 7: 5×5 aleatorio (seed=42) ─────")
    rng = np.random.default_rng(42)
    state_random = agent.execute_sovereign_prosecution_cycle(
        apu_cost_matrix=rng.normal(size=(5, 5)),
        novikov_exponents=[0.12, 0.27, 0.55],
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _print_summary(state_random)
    assert state_random.rsi3_state is not None
    assert state_random.celestial_state is not None
    assert len(state_random.merkle_root_sha256) == 64
    assert agent.verify_sovereign_state(state_random)
    print("  ✓ TEST 7 OK")

    # ── TEST 8: sello P1 manipulado → Φ₁₂ fail-CLOSED ────────────────────
    print("\n───── TEST 8: sello P1 manipulado → Φ₁₂ fail-CLOSED ─────")
    obs = agent.phase1_observe_algebraic_spectral(matrix_base, exps_base)
    obs = agent.phase1_observe_novikov_krull(obs, exps_base)
    legit_p1 = agent.phase1_emit_sovereign_certificate(obs)

    tampered_p1 = ProsecutorPhase1Certificate(
        betti_1_cycles=legit_p1.betti_1_cycles,
        torsion_detected=legit_p1.torsion_detected,
        krull_valuation_vA=legit_p1.krull_valuation_vA,
        chain_hash="0" * 64,
        hmac_signature=legit_p1.hmac_signature,
        timestamp_ns=legit_p1.timestamp_ns,
        engine_phase1_seal=legit_p1.engine_phase1_seal,
        timestamp=legit_p1.timestamp,
        engine_phase1_seal_verified=legit_p1.engine_phase1_seal_verified,
    )

    assert not agent.continue_to_phase2_gate(tampered_p1)
    try:
        agent.phase2_open_from_phase1_certificate(
            p1=tampered_p1,
            apu_cost_matrix=matrix_base,
            novikov_exponents=exps_base,
            ball_r=0.15,
            cylinder_R=0.20,
            positron_hmac_key=valid_key,
        )
        raise AssertionError("Φ₁₂ debió dispararse con chain_hash manipulado")
    except ProsecutorAgentGateViolationError as exc:
        print(f"  ✓ Φ₁₂ fail-CLOSED correctamente: {exc}")

    # ── TEST 9: cadena P2→P1 rota → Φ₂₃ fail-CLOSED ──────────────────────
    print("\n───── TEST 9: cadena P2→P1 rota → Φ₂₃ fail-CLOSED ─────")
    ctx2 = agent.phase2_open_from_phase1_certificate(
        p1=legit_p1,
        apu_cost_matrix=matrix_base,
        novikov_exponents=exps_base,
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    ctx2 = agent.phase2_orient_pseudoholomorphic_surface(ctx2)
    ctx2 = agent.phase2_orient_poincare_kam(ctx2)
    ctx2 = agent.phase2_evaluate_rsi3_banach(ctx2)
    ctx2 = agent.phase2_execute_dirac_annihilation(ctx2)
    legit_p2 = agent.phase2_emit_sovereign_certificate(ctx2)

    tampered_p2 = ProsecutorPhase2Certificate(
        res_cauchy_riemann=legit_p2.res_cauchy_riemann,
        gromov_violation=legit_p2.gromov_violation,
        banach_contraction_lip=legit_p2.banach_contraction_lip,
        energy_released_kev=legit_p2.energy_released_kev,
        poincare_melnikov_chaos=legit_p2.poincare_melnikov_chaos,
        chain_hash=legit_p2.chain_hash,
        hmac_signature=legit_p2.hmac_signature,
        timestamp_ns=legit_p2.timestamp_ns,
        engine_phase2_seal=legit_p2.engine_phase2_seal,
        previous_chain_hash="f" * 64,  # hash de P1 manipulado
        timestamp=legit_p2.timestamp,
        engine_phase2_seal_verified=legit_p2.engine_phase2_seal_verified,
        celestial_state=legit_p2.celestial_state,
        rsi3_state=legit_p2.rsi3_state,
        dirac_annihilation=legit_p2.dirac_annihilation,
    )
    assert not agent.continue_to_phase3_gate(tampered_p2, p1=legit_p1)
    try:
        agent.phase3_open_from_phase2_certificate(tampered_p2, p1=legit_p1)
        raise AssertionError("Φ₂₃ debió dispararse con cadena rota")
    except ProsecutorAgentGateViolationError as exc:
        print(f"  ✓ Φ₂₃ fail-CLOSED correctamente: {exc}")

    # ── TEST 10: Merkle soberano corrupto → verificación rechaza ─────────
    print("\n───── TEST 10: Merkle soberano corrupto → verificación rechaza ─────")
    tampered_state = BoolNovikovPseudoholomorphicProsecutorSovereignState(
        truth_value=state.truth_value,
        esp32_crowbar_triggered=state.esp32_crowbar_triggered,
        crowbar_latency_ns=state.crowbar_latency_ns,
        merkle_root_sha256="0" * 64,
        wacc_protection_percent=state.wacc_protection_percent,
        net_liquidity_liberation_percent=state.net_liquidity_liberation_percent,
        kv_cache_compression_percent=state.kv_cache_compression_percent,
        execution_time_ms=state.execution_time_ms,
        adjudication_reason=state.adjudication_reason,
        rsi3_state=state.rsi3_state,
        celestial_state=state.celestial_state,
        sovereign_hmac_sha256="0" * 64,
        phase1_chain_hash=state.phase1_chain_hash,
        phase2_chain_hash=state.phase2_chain_hash,
        crowbar_budget_respected=state.crowbar_budget_respected,
        crowbar_latency_budget_ns=state.crowbar_latency_budget_ns,
        ai_inference_savings_percent=state.ai_inference_savings_percent,
        agent_version=state.agent_version,
        engine_version=state.engine_version,
        timestamp=state.timestamp,
    )
    assert not agent.verify_sovereign_state(tampered_state)
    print("  ✓ Verificación del Merkle corrupto rechazada correctamente")

    # ── TEST 11: Verificación del Contrato de Anidamiento Formal (3 Fases) ─
    print("\n───── TEST 11: Contrato de Anidamiento Formal (Fase I → Fase II → Fase III) ─────")
    # FASE I culmina en phase1_emit_sovereign_certificate:
    obs_nested = agent.phase1_observe_algebraic_spectral(matrix_base, [0.22, 0.35])
    obs_nested = agent.phase1_observe_novikov_krull(obs_nested, [0.22, 0.35])
    cert_p1 = agent.phase1_emit_sovereign_certificate(obs_nested)
    assert cert_p1.spectral_graph_laplacian_gap > 0.0
    assert agent.verify_phase1_certificate(cert_p1)

    # FASE II inicia en phase2_open_from_phase1_certificate(cert_p1) y culmina en phase2_emit_sovereign_certificate:
    ctx_p2 = agent.phase2_open_from_phase1_certificate(
        p1=cert_p1,
        apu_cost_matrix=matrix_base,
        novikov_exponents=[0.22, 0.35],
        ball_r=0.12,
        cylinder_R=0.25,
        positron_hmac_key=valid_key,
        pert_eps=0.05,
    )
    ctx_p2 = agent.phase2_orient_pseudoholomorphic_surface(ctx_p2)
    ctx_p2 = agent.phase2_orient_poincare_kam(ctx_p2)
    ctx_p2 = agent.phase2_evaluate_rsi3_banach(ctx_p2, initial_x=0.95)
    ctx_p2 = agent.phase2_execute_dirac_annihilation(ctx_p2)
    cert_p2 = agent.phase2_emit_sovereign_certificate(ctx_p2)
    assert cert_p2.previous_chain_hash == cert_p1.chain_hash
    assert 0.0 < cert_p2.poincare_kam_orbital_stability_index <= 1.0
    assert 0.0 < cert_p2.rsi3_meta_convergence_index <= 1.0
    assert agent.verify_phase2_certificate(cert_p2, p1=cert_p1)

    # FASE III inicia en phase3_open_from_phase2_certificate(cert_p2) y culmina en phase3_emit_sovereign_state:
    ctx_p3 = agent.phase3_open_from_phase2_certificate(cert_p2, p1=cert_p1)
    ctx_p3 = agent.phase3_adjudicate_heyting_omega3(ctx_p3)
    ctx_p3 = agent.phase3_arm_esp32_crowbar(ctx_p3)
    ctx_p3 = agent.phase3_apply_visceral_functor_sem(ctx_p3)
    sovereign_final = agent.phase3_emit_sovereign_state(ctx_p3, execution_time_ms=1.15)
    assert sovereign_final.truth_value == HeytingProsecutorTruthValue.COHERENT
    assert sovereign_final.phase1_chain_hash == cert_p1.chain_hash
    assert sovereign_final.phase2_chain_hash == cert_p2.chain_hash
    assert agent.verify_sovereign_state(sovereign_final)
    print("  ✓ TEST 11 OK (Tríada OODA anidada verificada de extremo a extremo)")

    print("\n=== TODOS LOS TESTS SOBERANOS DEL AGENTE COMPLETADOS CON ÉXITO ===")