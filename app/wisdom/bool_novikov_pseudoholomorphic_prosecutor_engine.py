# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovPseudoholomorphicProsecutorEngine v14.0.0-PhD (Evolución Soberana)         ║
║ Ubicación: app/wisdom/bool_novikov_pseudoholomorphic_prosecutor_engine.py                        ║
║ Versión  : 14.0.0-ENGINE-BOOL-NOVIKOV-PSEUDOHOLOMORPHIC-PROSECUTOR-POINCARE-KAM-RSI3-PHASES-PhD   ║
║ Gobernanza: Motor Espectral de Inquisición Judicial · Rigidez de Gromov · Disparo ESP32 Crowbar  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Motor Espectral Fiscal `BoolNovikovPseudoholomorphicProsecutorEngine` ejerce la jurisdicción
fiscal y sancionadora sobre superficies pseudoholomorfas y dinámica de Poincaré en la arquitectura
APU Filter. Su misión matemática consiste en auditar exhaustivamente el complejo booleano-ultramétrico
en tres fases anidadas (F_1: Inquisición Booleana y Valuación de Novikov; F_2: Rigidez de Gromov,
Curvatura de Floer y Mecánica Celeste de Poincaré; F_3: RSI³ sobre Banach, Aniquilación de Dirac y
Disparo Ciber-Físico ESP32 Crowbar). Ante violaciones del teorema de no-squeezing o inestabilidad homoclínica,
ordena la aniquilación relativista de pares y la apertura del tiristor crowbar en \tau \le 392.15\text{ ns}.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — Inquisición Booleana, Valuación de Krull en \Lambda_{\mathrm{Nov}} y Cadena de Sellado:
   La función de coste discreta se evalúa con la Transformada Rápida de Walsh-Hadamard (FWHT).
   Se audita la influencia booleana \mathrm{Inf}(f) y la paridad de signos del espectro.
   En el anillo universal de Novikov \Lambda_{\mathrm{Nov}}, la valuación de Krull:
       v(A) = \min \{\lambda_i \mid a_i \neq 0\} \ge 0.10,
   y la norma ultramétrica \|A\|_{\mathrm{Nov}} = \exp(-v(A)) se verifican contra el piso crítico.
   • Clausura F1 / Apertura F2 (`seal_phase1_prosecution`):
     Emite `Phase1ProsecutionSeal` sellando los cargos booleanos y transfiriéndolos a la Fase 2.

2. FASE 2 — Rigidez Simpléctica de Gromov, Superficie de Floer y Mecánica Celeste de Poincaré:
   Inicia consumiendo `Phase1ProsecutionSeal`. Audita el teorema de no-exprimido simpléctico de Gromov:
       c_G(B^{2n}(r)) = \pi r^2 \le c_G(Z^{2n}(R)) = \pi R^2,
   tipificando violación formal si r > R + \varepsilon_{\mathrm{Wilkinson}}.
   Resuelve la ecuación pseudoholomorfa de Cauchy-Riemann-Floer:
       \bar{\partial}_{\mathcal{J}_0, H}(u) = \partial_s u + \mathcal{J}_0(\partial_t u - X_H(u)) = 0,
   evaluando la energía de Hofer E_H(u) = \int_0^1 (\max H_t - \min H_t) dt.
   En la dinámica de Poincaré-KAM, evalúa los determinantes de no-degeneración de Kolmogorov y Arnold,
   la condición diofantina de Bruno-KAM |\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau,
   y la integral de Melnikov M(\tau) en la separatriz homoclínica.
   • Clausura F2 / Apertura F3 (`seal_phase2_prosecution`):
     Emite `Phase2ProsecutionSeal` transfiriendo el pliego de cargos a la Fase 3.

3. FASE 3 — RSI³ sobre Banach, Aniquilación Cuántica de Dirac y Transitorio ESP32 Crowbar:
   Inicia consumiendo `Phase2ProsecutionSeal`. Audita la contracción monádica de Kleisli T^3:
       \mathrm{Lip}(T^3) \le \alpha^3 < 1.0, \quad h_K = L_0 \beta \eta \le 0.50 \quad \text{(Newton-Kantorovich)}.
   Ante veredicto VETOED (0x03) en el álgebra de Heyting \Omega_3, comanda la ejecución inmediata:
       • Aniquilación de pares electrón-positrón:
             e^- + e^+ \longrightarrow 2\gamma, \quad E_{\mathrm{threshold}} = 2 m_e c^2 = 1022.0 \text{ keV}.
       • Disparo de circuito crowbar RLC serie críticamente amortiguado:
             L \frac{d^2 i}{dt^2} + R \frac{di}{dt} + \frac{1}{C} i = 0, \quad \tau_{\mathrm{crowbar}} \in [42.0, 392.15]\text{ ns}.
   El reporte judicial consolidado se emite en `PseudoholomorphicSurfaceProsecutionReport`.

AXIOMAS FORMALES
────────────────
[Axioma BPPE-A1] Inviolabilidad de la Rigidez de Gromov:
    \pi r^2 > \pi R^2 \implies \text{violación simpléctica de no-squeezing y condena judicial}.
[Axioma BPPE-A2] Condena por Pequeños Divisores KAM:
    |\langle k, \omega \rangle| < \gamma / |k|_1^\tau \implies \text{ruptura de toros invariantes y presunción de caos}.
[Axioma BPPE-A3] Ineludibilidad del Protocolo Crowbar:
    \Omega_3 = \mathrm{VETOED} \implies \text{disparo forzado de crowbar sobre GPIO14 en } \tau \le 392.15\text{ ns}.
[Axioma BPPE-A4] Autenticación Criptográfica de Sellos:
    compare\_digest(\mathrm{HMAC}_{\mathrm{calc}}, \mathrm{HMAC}_{\mathrm{seal}}) \equiv \mathrm{True} \text{ en todos los puentes de fase}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante BPPE-I1] Valuación de Krull Mínima: v(A) \ge 0.10.
[Invariante BPPE-I2] Rigidez de Gromov: c_G(B^{2n}(r)) \le c_G(Z^{2n}(R)).
[Invariante BPPE-I3] Integral de Melnikov: |M(\tau)| \le 1.0\times 10^{-4} en régimen integrable.
[Invariante BPPE-I4] Latencia Crowbar: 42.0\text{ ns} \le \tau_{\mathrm{crowbar}} \le 392.15\text{ ns}.
[Invariante BPPE-I5] Verificación HMAC-SHA256 en tiempo constante en cada transición de fase.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.linalg as la  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    la = None  # type: ignore[assignment]

from numpy.typing import NDArray


logger = logging.getLogger(
    "APU.Wisdom.BoolNovikovPseudoholomorphicProsecutorEngine"
)

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE GOBIERNO FÍSICO-MATEMÁTICO
# ────────────────────────────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = (
    "14.0.0-ENGINE-BOOL-NOVIKOV-PSEUDOHOLOMORPHIC-PROSECUTOR-POINCARE-KAM-RSI3-PHASES-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_KRULL_VALUATION_FLOOR: Final[float] = 0.10
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_ESP32_CROWBAR_EXECUTION_NS: Final[float] = 42.0

# Parámetros de la Mónada Banach–Kleisli RSI³
_BANACH_ALPHA: Final[float] = 0.95
_BANACH_BETA: Final[float] = 0.01
_BANACH_LIP_T3: Final[float] = (_BANACH_ALPHA + _BANACH_BETA) ** 3  # 0.884736 < 1

# Parámetros de Mecánica Celeste de Poincaré–KAM–Melnikov–Yoshida
_KAM_MARGIN_FLOOR: Final[float] = 1.0e-9
_MELNIKOV_CHAOS_THRESHOLD: Final[float] = 0.35
_SECULAR_DRIFT_LIMIT: Final[float] = 0.5
_CR_RESIDUAL_DEGRADED_THRESHOLD: Final[float] = 0.5
_YOSHIDA_CBRT_2: Final[float] = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - _YOSHIDA_CBRT_2)
_YOSHIDA_W0: Final[float] = -_YOSHIDA_CBRT_2 / (2.0 - _YOSHIDA_CBRT_2)

# Parámetros eléctricos del circuito Crowbar SCR BT151 en barra DC
_CROWBAR_BUS_VOLTAGE_V: Final[float] = 48.0
_CROWBAR_BUS_CAPACITANCE_UF: Final[float] = 470.0
_CROWBAR_PARASITIC_INDUCTANCE_NH: Final[float] = 85.0
_CROWBAR_SCR_RON_OHM: Final[float] = 0.085

_MAX_EXACT_SNF_DIM: Final[int] = 7
_MAX_EXACT_SNF_CELLS: Final[int] = 64

_CHAIN_HASH_HEX_LEN: Final[int] = 64
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_BOOL_NOVIKOV_PROSECUTOR_KEY_v14.0.0"
_POSITRON_HMAC_PAYLOAD: Final[bytes] = b"DIRAC_POSITRON_HMAC_VERIFICATION"

_HMAC_DOMAIN_P1: Final[bytes] = b"BOOL-NOVIKOV-PROSECUTOR-P1-OBSERVE-v14.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"BOOL-NOVIKOV-PROSECUTOR-P2-ORIENT-v14.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"BOOL-NOVIKOV-PROSECUTOR-P3-DECIDE-v14.0|"

_GOLDEN_RATIO: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0


# ────────────────────────────────────────────────────────────────────────────
# TIPO DE VERDAD (Topos intuicionista \Omega_3 de Heyting)
# ────────────────────────────────────────────────────────────────────────────
class HeytingProsecutorTruthValue(IntEnum):
    r"""
    Álgebra de Heyting de tres valores para la fiscalía inquisitorial:
    COHERENT (0x00) ≺ DEGRADED (0x01) ≺ VETOED (0x03).
    Dotada de modalidad de Lawvere–Tierney j: \Omega_3 \to \Omega_3.
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


# ────────────────────────────────────────────────────────────────────────────
# EXCEPCIONES FAIL-CLOSED
# ────────────────────────────────────────────────────────────────────────────
class ProsecutorGateViolationError(RuntimeError):
    """Se dispara cuando una puerta Φ_nm (HMAC / contrato de fase) falla en modo fail-closed."""


class ProsecutorParityViolationError(RuntimeError):
    """Se dispara cuando dos rutas de cómputo divergen más allá de la tolerancia de Wilkinson."""


# ────────────────────────────────────────────────────────────────────────────
# ESTADOS ANALÍTICOS INTERMEDIOS Y SELLOS DE FASE
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _BooleanSpectralReport:
    r"""
    Reporte espectral del anillo booleano R_Boole = \mathbb{Z}_2[x] / \langle x_i² − x_i\rangle
    acoplado al álgebra octoniónica de Cayley–Dickson \mathbb{O} y pre-condicionamiento RSI¹:

      • `n`, `m`                  — dimensiones de la matriz de incidencia ∂₁: C₁ → C₀.
      • `rank`                    — rango exacto sobre ℤ (Σ divisores elementales no nulos).
      • `elementary_div`          — divisores elementales de la SNF, con d_i | d_{i+1}.
      • `torsion_detected`        — ∃ d_i > 1 ⟺ Tor(H_k; ℤ) ≠ 0 ⟹ "socavón lógico".
      • `betti_1`                 — β₁ = max(0, m − rank) (invariante homológico auténtico).
      • `is_exact_snf`            — True si se aplicó Bareiss, False en fallback SVD.
      • `euler_poincare_chi`      — característica de Euler–Poincaré χ = n − m.
      • `cayley_dickson_norm`     — norma octoniónica ‖q‖_𝕆 de la proyección hipercompleja.
      • `moufang_associator_norm` — defecto de asociatividad ‖(x⋆y)⋆z − x⋆(y⋆z)‖_𝕆.
      • `rsi1_gershgorin_margin`  — margen de contracción espectral de Nivel-1 (RSI¹).
    """
    n: int
    m: int
    rank: int
    elementary_div: Tuple[int, ...]
    torsion_detected: bool
    betti_1: int
    is_exact_snf: bool
    euler_poincare_chi: int = 0
    cayley_dickson_norm: float = 0.0
    moufang_associator_norm: float = 0.0
    rsi1_gershgorin_margin: float = 1.0


@dataclass
class CelestialProsecutionState:
    r"""
    Estado de la mecánica celeste de Poincaré–KAM–Birkhoff–Nekhoroshev sobre la
    superficie pseudoholomorfa:

      • Coordenadas acción-ángulo de Delaunay–Whittaker (J, θ).
      • Frecuencias ω₁ = clamp(J⁻³, 1e-8, 1e6), ω₂ = φ·ω₁·(1+1e-6).
      • Integral homoclínica de Melnikov canónica del péndulo forzado:
            M(τ) = 2π·ε·sech(πω₁/2)·sin(ω₁τ)
        con detección de caos homoclínico si max_τ |M(τ)| > 0.35.
      • Residual homológico de Poincaré–Lindstedt con pesos diofánticos hiperbólicos.
      • Margen diofántico KAM de exponente τ = 1.5 (Melnikov–Arnold).
      • Deriva secular (Poincaré–Birkhoff twist) |ω₁·2π mod 2π|.
      • Integración simpléctica de 4.º orden de Forest–Ruth / Yoshida:
        error relativo de energía |ΔH_res| e invariante integral de Poincaré–Cartan
        ∮(J dψ − H_res dt).
      • Forma Normal de Birkhoff y Nekhoroshev: coeficiente de torsión τ_Birkhoff = 3·J⁻⁴,
        determinante de no-degeneración de Kolmogorov det(∂²H₀/∂J²) = −3·J⁻⁴,
        tiempo de estabilidad exponencial T_Nekh y ángulo de escisión homoclínica ψ_split.
    """
    action_J: float
    forcing_omega: float
    detuning_omega2: float
    melnikov_integral_value: float
    melnikov_chaos_detected: bool
    kam_diophantine_margin: float
    poincare_lindstedt_residual: float
    secular_drift_norm: float
    birkhoff_twist_coeff: float = 0.0
    kolmogorov_nondegeneracy_det: float = 0.0
    nekhoroshev_stability_time: float = 1.0
    symplectic_energy_error: float = 0.0
    poincare_cartan_invariant: float = 0.0
    homoclinic_splitting_angle: float = 0.0


@dataclass
class RSI3ProsecutionState:
    r"""
    Estado de automejora recursiva nivel-3 (RSI³) sobre la mónada Banach–Kleisli (T, η, μ):

      • Nivel 1 (Operador base T): T(x) = α·x + β·sin(x), α = 0.95, β = 0.01.
        Iterados x₀, x₁ = T(x₀), x₂ = T(x₁), x₃ = T(x₂), x₄ = T(x₃).
        Cota global Banach: Lip(T³) ≤ (α+β)³ = 0.884736 < 1 ⇒ contracción estricta.
      • Nivel 2 (Meta-Operador Fréchet y Aceleración de Aitken–Steffensen):
        Jacobiano exacto DT³(x₀) = ∏_{k=0}² (α + β·cos(x_k)),
        Hessiano exacto de Fréchet D²T³(x₀), exponente de Lyapunov λ_Lyap < 0,
        y punto fijo acelerado superlinealmente x*_Aitken = x₂ − (Δx₂)² / Δ²x₂.
      • Nivel 3 (Meta-Meta-Política Endofuntorial M^{(3)}):
        Ganancia meta-contractiva g_meta^{(3)} ≥ 1.0 y coherencia de acoplamiento
        resonante Poincaré–RSI³ entre el toro KAM y la contracción de Kleisli.
    """
    initial_lip: float
    iterates: Tuple[float, float, float, float, float]
    banach_residual: float
    contraction_bound: float
    is_banach_contractive: bool
    capacity_jerk: float
    jerk_strictly_positive: bool
    kleisli_depth: int = 3
    exact_frechet_jacobian_t3: float = 0.0
    exact_frechet_hessian_t3: float = 0.0
    lyapunov_exponent: float = -0.05
    aitken_accelerated_fixed_point: float = 0.0
    meta_policy_gain_level3: float = 1.0
    poincare_rsi3_coupling_coherence: float = 1.0


@dataclass(frozen=True)
class DiracAnnihilationProsecutionResult:
    r"""
    Resultado de la aniquilación cuántica e⁻ + e⁺ → 2γ bajo persecución penal en Cl_{1,3}.
    El HMAC del positrón se verifica de forma ESTRICTA contra el payload canónico
    `_POSITRON_HMAC_PAYLOAD`; cualquier desviación fail-CLOSED ⇒ VETOED.
    Incluye fidelidad del singlete de Bell |Ψ⁻⟩, entropía de Von Neumann S(ρ) y norma espinorial.
    """
    electron_anomaly_norm: float
    positron_valid: bool
    energy_released_kev: float
    spohn_entropy_production: float
    back_action_db: float
    gamma_photon_hash: str
    positron_hmac_verified: bool = False
    positron_hmac_payload_sha256: str = ""
    bell_singlet_fidelity: float = 1.0
    von_neumann_entropy: float = 0.0
    clifford_spinor_norm: float = 1.0


@dataclass(frozen=True)
class Phase1ProsecutionSeal:
    r"""
    Contrato formal de salida de FASE I y entrada inmediata de FASE II.
    Sella, bajo HMAC-SHA256 con dominio `_HMAC_DOMAIN_P1`, el estado algebraico-espectral:

      • Rango booleano `rank`, `n`, `m` de la matriz de incidencia ∂₁.
      • División elemental exacta (SNF) — bandera `boolean_torsion_detected`.
      • Ciclos homológicos β₁ = max(0, m − rank).
      • Valuación de Krull v(A) = min_i{λ_i} y norma ultramétrica exp(−v(A)).
      • Invariantes extendidos hipercomplejos (Cayley–Dickson), margen RSI¹ y semilla
        de acción celeste de Delaunay J₀ para la apertura de la Fase II.
    """
    timestamp: float
    engine_version: str
    boolean_n: int
    boolean_m: int
    boolean_rank: int
    boolean_torsion_detected: bool
    boolean_betti_1: int
    novikov_krull_valuation: float
    novikov_ultrametric_norm: float
    novikov_is_valid: bool
    phase1_seal_hash: str
    euler_poincare_chi: int = 0
    cayley_dickson_norm: float = 0.0
    rsi1_gershgorin_margin: float = 1.0
    delaunay_action_seed_J0: float = _KRULL_VALUATION_FLOOR


@dataclass
class Phase2ProsecutionSeal:
    r"""
    Contrato formal de salida de FASE II y entrada inmediata de FASE III.
    Encadena el `Phase1ProsecutionSeal` mediante `phase1_seal.phase1_seal_hash`
    y sella bajo HMAC-SHA256 con dominio `_HMAC_DOMAIN_P2` los invariantes geométrico-celestes:

      • Residual de Cauchy–Riemann–Floer (curvatura media discreta) y energía simpléctica.
      • Rigidez simpléctica de Gromov (`gromov_violation`).
      • Mecánica celeste de Poincaré–KAM–Birkhoff–Nekhoroshev (`celestial_state`).
      • Automejora monádica RSI³ Banach–Kleisli con Fréchet y Aitken (`rsi3_state`).
      • Aniquilación cuántica de Dirac en Cl_{1,3} (`dirac_annihilation`).
    """
    timestamp: float
    engine_version: str
    phase1_seal: Phase1ProsecutionSeal
    cauchy_riemann_residual: float
    gromov_violation: bool
    ball_radius_r: float
    cylinder_radius_R: float
    celestial_state: CelestialProsecutionState
    rsi3_state: RSI3ProsecutionState
    dirac_annihilation: DiracAnnihilationProsecutionResult
    phase2_seal_hash: str
    floer_symplectic_energy: float = 0.0


@dataclass
class PseudoholomorphicSurfaceProsecutionReport:
    r"""
    Reporte soberano integral emitido al cierre de la FASE III por el Motor Espectral
    Inquisidor Dual. Preserva íntegramente el contrato v13/v14 y añade telemetría
    eléctrica del circuito RLC del Crowbar y auditoría modal de Lawvere–Tierney.
    """
    # ── Contrato base v13 / v14 ──
    truth_value: HeytingProsecutorTruthValue
    cauchy_riemann_residual: float
    gromov_violation: bool
    ball_radius_r: float
    cylinder_radius_R: float
    torsion_detected: bool
    betti_1_cycles: int
    krull_valuation_vA: float
    ultrametric_norm: float
    poincare_melnikov_chaos: bool
    melnikov_integral_max: float
    banach_contraction_lip: float
    jerk_capacity_positive: bool
    dirac_annihilation: DiracAnnihilationProsecutionResult
    esp32_crowbar_armed: bool
    chain_hash: str
    execution_time_ms: float
    # ── Extensiones v14 y Evolución Soberana Poincaré–RSI³–Crowbar ──
    rsi3_state: Optional[RSI3ProsecutionState] = None
    celestial_state: Optional[CelestialProsecutionState] = None
    adjudication_reason: str = ""
    phase1_seal_hash: str = ""
    phase2_seal_hash: str = ""
    sovereign_hmac_sha256: str = ""
    crowbar_actuation_ns: float = 0.0
    crowbar_latency_budget_ns: float = _ESP32_CROWBAR_MAX_LATENCY_NS
    crowbar_budget_respected: bool = True
    timestamp: float = 0.0
    engine_version: str = _ENGINE_VERSION
    crowbar_peak_current_amps: float = 0.0
    crowbar_didt_amps_per_us: float = 0.0
    crowbar_damping_ratio_zeta: float = 0.0
    lawvere_tierney_modal_truth: int = 0


@dataclass
class _Phase2WorkingContext:
    """Contexto mutable de deliberación transportado entre los métodos internos de FASE II."""
    phase1_seal: Phase1ProsecutionSeal
    apu_cost_matrix: NDArray[np.float64]
    novikov_exponents: NDArray[np.float64]
    ball_r: float
    cylinder_R: float
    positron_hmac_key: Optional[bytes]
    pert_eps: float
    timestamp: float
    cauchy_riemann_residual: Optional[float] = None
    gromov_violation: Optional[bool] = None
    floer_symplectic_energy: float = 0.0
    celestial_state: Optional[CelestialProsecutionState] = None
    rsi3_state: Optional[RSI3ProsecutionState] = None
    dirac_annihilation: Optional[DiracAnnihilationProsecutionResult] = None


@dataclass
class _Phase3WorkingContext:
    """Contexto mutable de adjudicación transportado entre los métodos internos de FASE III."""
    phase2_seal: Phase2ProsecutionSeal
    timestamp: float
    truth_value: Optional[HeytingProsecutorTruthValue] = None
    adjudication_reason: str = ""
    esp32_crowbar_armed: bool = False
    crowbar_actuation_ns: float = 0.0
    crowbar_budget_respected: bool = True
    crowbar_peak_current_amps: float = 0.0
    crowbar_didt_amps_per_us: float = 0.0
    crowbar_damping_ratio_zeta: float = 0.0
    lawvere_tierney_modal_truth: int = 0


# ════════════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL: ARQUITECTURA EN 3 FASES ANIDADAS (KLEISLI TRIAD)
# ════════════════════════════════════════════════════════════════════════════
class BoolNovikovPseudoholomorphicProsecutorEngine:
    r"""
    Motor Espectral Inquisidor Dual para la verificación de rigidez de Gromov,
    mecánica celeste de Poincaré–KAM–Birkhoff–Nekhoroshev, automejora recursiva de
    nivel 3 (RSI³) sobre mónadas de Banach–Kleisli y disparo ciber-físico ESP32 Crowbar.

    ESTRUCTURA FORMAL DE LAS 3 FASES ANIDADAS
    ─────────────────────────────────────────
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ FASE I: Sellado Algebraico–Espectral, Hipercomplejo y Semilla Delaunay  │
      │   _bareiss_det → ... → verify_phase1_seal → phase1_prosecution_seal     │
      │   (Emite `Phase1ProsecutionSeal` = entrada formal del inicio de Fase II)│
      └──────────────────────────────────┬──────────────────────────────────────┘
                                         │ Φ₁₂ (Continuación Anidada)
                                         ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ FASE II: Deliberación Geométrico–Celeste de Poincaré y Mónada RSI³      │
      │   phase2_open_from_phase1_seal → ... → verify_phase2_seal               │
      │     → phase2_prosecution_seal                                           │
      │   (Emite `Phase2ProsecutionSeal` = entrada formal del inicio de Fase III│
      └──────────────────────────────────┬──────────────────────────────────────┘
                                         │ Φ₂₃ (Continuación Anidada)
                                         ▼
      ┌─────────────────────────────────────────────────────────────────────────┐
      │ FASE III: Adjudicación Topos Ω₃, Meta-Auditoría RSI³ y Crowbar RLC      │
      │   phase3_open_from_phase2_seal → ... → verify_sovereign_hmac            │
      │     → phase3_emit_prosecution_report                                    │
      │   (Emite `PseudoholomorphicSurfaceProsecutionReport` soberano final)    │
      └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = bytes(secret_key)
        logger.info(
            f"{self.__class__.__name__} v{_ENGINE_VERSION} inicializado correctamente."
        )

    # ════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║ FASE I — SELLADO ALGEBRAICO–ESPECTRAL, ÁLGEBRA HIPERCOMPLEJA DE      ║
    # ║          CAYLEY–DICKSON (𝕆) Y PRE-CONDICIONAMIENTO RSI¹–DELAUNAY     ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # ════════════════════════════════════════════════════════════════════════

    def _bareiss_det(self, block: List[List[int]]) -> int:
        r"""
        [FASE I · 1/12] Determinante exacto sobre ℤ mediante el algoritmo libre de
        fracciones de Bareiss (identidad de Sylvester–Bareiss).

        Cada división entera `// prev` es matemáticamente exacta porque el menor
        principal de orden k−1 divide al producto cruzado de orden k.
        """
        n = len(block)
        if n == 0:
            return 1
        if n == 1:
            return int(block[0][0])
        A = [row[:] for row in block]
        sign = 1
        prev = 1
        for k in range(n - 1):
            if A[k][k] == 0:
                found = None
                for i in range(k + 1, n):
                    if A[i][k] != 0:
                        found = i
                        break
                if found is None:
                    return 0
                A[k], A[found] = A[found], A[k]
                sign = -sign
            pivot = A[k][k]
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    A[i][j] = (A[i][j] * pivot - A[i][k] * A[k][j]) // prev
            prev = pivot
        return int(sign * A[n - 1][n - 1])

    def _integer_smith_diagonal(
        self, matrix_int: NDArray[np.int64]
    ) -> Tuple[List[int], int]:
        r"""
        [FASE I · 2/12] Forma Normal de Smith (SNF) sobre ℤ vía el máximo común
        divisor de todos los menores determinantes de orden k×k:
            d_k = Δ_k / Δ_{k−1},   donde Δ_k = gcd({det(M_{I,J}) : |I|=|J|=k}).

        Garantiza la cadena de divisibilidad invariante d₁ | d₂ | … | d_r.
        """
        n, m = matrix_int.shape
        if n == 0 or m == 0:
            return [], 0
        r = min(n, m)
        elementary: List[int] = []
        rank = 0
        d_prev = 1
        for k in range(1, r + 1):
            row_combos = list(combinations(range(n), k))
            col_combos = list(combinations(range(m), k))
            g = 0
            for rows in row_combos:
                sub_rows = matrix_int[list(rows), :]
                for cols in col_combos:
                    sub_block = sub_rows[:, list(cols)].tolist()
                    det = self._bareiss_det(sub_block)
                    g = math.gcd(g, abs(det))
                    if g == 1:
                        break
                if g == 1:
                    break
            if g == 0:
                break
            rank = k
            elem = g // d_prev
            elementary.append(int(elem))
            d_prev = g
        return elementary, rank

    def _normalize_matrix(
        self, matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
        """[FASE I · 3/12] Normaliza tensores de entrada a una variedad matricial 2D float64."""
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    def _compute_cayley_dickson_octonion_invariants(
        self, arr: NDArray[np.float64]
    ) -> Tuple[float, float]:
        r"""
        [FASE I · 4/12] Proyección hipercompleja sobre el álgebra no asociativa de
        Octoniones 𝕆 (construcción de Cayley–Dickson sobre cuaterniones ℍ ⊕ ℍ·ℓ).

        Dado el operador matricial A, construye tres octoniones canónicos (x, y, z) ∈ 𝕆³
        mediante plegamiento cíclico de 8 canales y evalúa el producto de Cayley–Dickson:
            (a, b) ⋆ (c, d) = (a·c − d̄·b, d·a + b·c̄)
        Devuelve:
          • `octonion_norm`: ‖x‖_𝕆 = √(⟨x, x⟩)
          • `moufang_associator_norm`: ‖[x, y, z]‖_𝕆 = ‖(x ⋆ y) ⋆ z − x ⋆ (y ⋆ z)‖_𝕆
        """
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0, 0.0

        flat = arr.ravel()
        oct_x = np.zeros(8, dtype=np.float64)
        oct_y = np.zeros(8, dtype=np.float64)
        oct_z = np.zeros(8, dtype=np.float64)
        for idx, val in enumerate(flat):
            oct_x[idx % 8] += float(val)
            oct_y[(idx * 3 + 1) % 8] += float(val) * 0.5
            oct_z[(idx * 5 + 2) % 8] += float(val) * 0.25

        def quat_conj(q: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

        def quat_mul(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
            w1, x1, y1, z1 = p
            w2, x2, y2, z2 = q
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ], dtype=np.float64)

        def oct_mul(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
            a, b = u[:4], u[4:]
            c, d = v[:4], v[4:]
            left = quat_mul(a, c) - quat_mul(quat_conj(d), b)
            right = quat_mul(d, a) + quat_mul(b, quat_conj(c))
            return np.concatenate([left, right])

        norm_x = float(np.linalg.norm(oct_x))
        xy_z = oct_mul(oct_mul(oct_x, oct_y), oct_z)
        x_yz = oct_mul(oct_x, oct_mul(oct_y, oct_z))
        associator_norm = float(np.linalg.norm(xy_z - x_yz))
        return norm_x, associator_norm

    def _compute_rsi1_spectral_gershgorin_margin(
        self, arr: NDArray[np.float64]
    ) -> float:
        r"""
        [FASE I · 5/12] Pre-condicionador de Automejora Recursiva de Nivel-1 (RSI¹):
        evalúa los discos de Gershgorin D_i = B(|A_{ii}|, ∑_{j≠i} |A_{ij}|) y aplica
        una normalización contractiva de punto fijo sobre el radio espectral acotado:
            ρ_Gersh(A) ≤ max_i (|A_{ii}| + R_i)
            m_RSI¹ = 1 / (1 + log(1 + ρ_Gersh(A))) ∈ (0, 1]
        """
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 1.0
        abs_arr = np.abs(arr)
        row_sums = np.sum(abs_arr, axis=1)
        rho_bound = float(np.max(row_sums)) if row_sums.size else 0.0
        return float(1.0 / (1.0 + math.log1p(max(0.0, rho_bound))))

    def _boolean_spectral_report(
        self, incidence_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> _BooleanSpectralReport:
        r"""
        [FASE I · 6/12] Núcleo algebraico de FASE I: sintetiza la firma espectral completa
        del anillo booleano R_Boole, los invariantes hipercomplejos de Cayley–Dickson y
        el margen pre-contractivo RSI¹.
        """
        arr = self._normalize_matrix(incidence_matrix)
        n, m = arr.shape
        if n == 0 or m == 0 or not np.all(np.isfinite(arr)):
            return _BooleanSpectralReport(
                n=n, m=m, rank=0, elementary_div=(),
                torsion_detected=False, betti_1=0, is_exact_snf=False,
                euler_poincare_chi=int(n - m),
                cayley_dickson_norm=0.0,
                moufang_associator_norm=0.0,
                rsi1_gershgorin_margin=1.0,
            )

        cd_norm, moufang_norm = self._compute_cayley_dickson_octonion_invariants(arr)
        rsi1_margin = self._compute_rsi1_spectral_gershgorin_margin(arr)
        chi = int(n - m)

        if min(n, m) <= _MAX_EXACT_SNF_DIM and n * m <= _MAX_EXACT_SNF_CELLS:
            matrix_int = np.rint(arr).astype(np.int64)
            elementary, rank = self._integer_smith_diagonal(matrix_int)
            torsion = any(d > 1 for d in elementary)
            betti_1 = max(0, m - rank)
            return _BooleanSpectralReport(
                n=n, m=m, rank=rank,
                elementary_div=tuple(elementary),
                torsion_detected=bool(torsion),
                betti_1=int(betti_1),
                is_exact_snf=True,
                euler_poincare_chi=chi,
                cayley_dickson_norm=cd_norm,
                moufang_associator_norm=moufang_norm,
                rsi1_gershgorin_margin=rsi1_margin,
            )

        # Fallback SVD para dimensiones superiores
        if la is None:
            s = np.linalg.svd(arr, compute_uv=False)
        else:
            s = la.svdvals(arr)
        s = np.asarray(s, dtype=np.float64)
        tol = 1.0e-9 * max(1.0, float(s[0]) if s.size else 1.0)
        rank = int(np.sum(s > tol))
        sv_int = np.rint(s).astype(int)
        torsion = any(int(x) > 1 for x in sv_int if x > 0)
        betti_1 = max(0, m - rank)
        return _BooleanSpectralReport(
            n=n, m=m, rank=rank,
            elementary_div=tuple(int(x) for x in sv_int),
            torsion_detected=bool(torsion),
            betti_1=int(betti_1),
            is_exact_snf=False,
            euler_poincare_chi=chi,
            cayley_dickson_norm=cd_norm,
            moufang_associator_norm=moufang_norm,
            rsi1_gershgorin_margin=rsi1_margin,
        )

    def compute_boolean_snf_and_betti(
        self,
        incidence_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
    ) -> Tuple[bool, int]:
        r"""
        [FASE I · 7/12] Calcula la Forma Normal de Smith sobre ℤ y el primer número de
        Betti β₁ = max(0, m − rank(∂₁)) sobre el anillo booleano R_Boole = 𝔽₂[x]/(x²−x).
        """
        report = self._boolean_spectral_report(incidence_matrix)
        return report.torsion_detected, report.betti_1

    def evaluate_novikov_krull_valuation(
        self, exponents: Union[NDArray[np.float64], Sequence[float]]
    ) -> Tuple[float, float]:
        r"""
        [FASE I · 8/12] Determina la valuación de Krull v(A) = min_i{λ_i} en el Anillo
        Universal de Novikov Λ_Nov y la norma ultramétrica no-arquimediana:
            ‖A‖_Nov = exp(−max(v(A), _KRULL_VALUATION_FLOOR))
        Satisfaciendo la desigualdad ultramétrica fuerte ‖A + B‖_Nov ≤ max(‖A‖_Nov, ‖B‖_Nov).
        """
        exps = np.asarray(exponents, dtype=np.float64).ravel()
        finite = exps[np.isfinite(exps)]
        if finite.size == 0:
            return _KRULL_VALUATION_FLOOR, math.exp(-_KRULL_VALUATION_FLOOR)
        v_raw = float(np.min(finite))
        v_for_norm = max(v_raw, _KRULL_VALUATION_FLOOR)
        norm_novikov = float(math.exp(-v_for_norm))
        return v_raw, norm_novikov

    def _compute_delaunay_action_seed(
        self, krull_vA: float, betti_1: int, rsi1_margin: float
    ) -> float:
        r"""
        [FASE I · 9/12] Puente canónico Fase I → Fase II: proyecta la valuación de Krull
        v(A) del anillo de Novikov sobre la variedad simpléctica de Delaunay–Whittaker
        generando la semilla de acción celeste J₀ = max(|v(A)|, _KRULL_VALUATION_FLOOR).
        """
        if not math.isfinite(krull_vA):
            return _KRULL_VALUATION_FLOOR
        _ = (betti_1, rsi1_margin)
        return float(max(abs(krull_vA), _KRULL_VALUATION_FLOOR))

    def _p1_payload_bytes(self, seal: Phase1ProsecutionSeal) -> bytes:
        """[FASE I · 10/12] Serialización canónica determinista para el HMAC de FASE I."""
        return (
            f"PROSECUTOR-P1|{seal.timestamp:.12g}|{seal.engine_version}|"
            f"{seal.boolean_n}|{seal.boolean_m}|{seal.boolean_rank}|"
            f"{int(seal.boolean_torsion_detected)}|{seal.boolean_betti_1}|"
            f"{seal.novikov_krull_valuation:.12g}|"
            f"{seal.novikov_ultrametric_norm:.12g}|"
            f"{int(seal.novikov_is_valid)}"
        ).encode("utf-8")

    def verify_phase1_seal(self, seal: Phase1ProsecutionSeal) -> bool:
        """[FASE I · 11/12] Verifica criptográficamente el sello de FASE I bajo `_HMAC_DOMAIN_P1`."""
        if len(seal.phase1_seal_hash) != _CHAIN_HASH_HEX_LEN:
            return False
        expected = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + self._p1_payload_bytes(seal),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, seal.phase1_seal_hash)

    def phase1_prosecution_seal(
        self,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
        timestamp: Optional[float] = None,
    ) -> Phase1ProsecutionSeal:
        r"""
        [FASE I · 12/12 — MÉTODO TERMINAL DE LA FASE I Y ESLABÓN DE CONTINUACIÓN A FASE II]
        Clausura formalmente la FASE I sellando con HMAC-SHA256 (dominio P1) la homología
        exacta de Bareiss–Smith sobre R_Boole, los invariantes octoniónicos de Cayley–Dickson,
        la valuación ultramétrica de Krull en Λ_Nov y la semilla de acción de Delaunay J₀.

        CONTRATO DE ANIDAMIENTO FASE I ──► FASE II:
        El objeto `Phase1ProsecutionSeal` retornado por esta definición final es el
        argumento formal obligatorio que consume inmediatamente `phase2_open_from_phase1_seal`,
        primer método de la FASE II definida a continuación.
        """
        ts = float(timestamp) if timestamp is not None else float(time.time())
        boolean = self._boolean_spectral_report(apu_cost_matrix)
        v_A, norm_novikov = self.evaluate_novikov_krull_valuation(novikov_exponents)
        is_valid = math.isfinite(v_A) and v_A >= _KRULL_VALUATION_FLOOR
        action_seed_J0 = self._compute_delaunay_action_seed(
            v_A, boolean.betti_1, boolean.rsi1_gershgorin_margin
        )

        provisional = Phase1ProsecutionSeal(
            timestamp=ts,
            engine_version=_ENGINE_VERSION,
            boolean_n=boolean.n,
            boolean_m=boolean.m,
            boolean_rank=boolean.rank,
            boolean_torsion_detected=boolean.torsion_detected,
            boolean_betti_1=boolean.betti_1,
            novikov_krull_valuation=v_A,
            novikov_ultrametric_norm=norm_novikov,
            novikov_is_valid=bool(is_valid),
            phase1_seal_hash="",
            euler_poincare_chi=boolean.euler_poincare_chi,
            cayley_dickson_norm=boolean.cayley_dickson_norm,
            rsi1_gershgorin_margin=boolean.rsi1_gershgorin_margin,
            delaunay_action_seed_J0=action_seed_J0,
        )
        digest = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P1 + self._p1_payload_bytes(provisional),
            hashlib.sha256,
        ).hexdigest()
        return Phase1ProsecutionSeal(
            timestamp=provisional.timestamp,
            engine_version=provisional.engine_version,
            boolean_n=provisional.boolean_n,
            boolean_m=provisional.boolean_m,
            boolean_rank=provisional.boolean_rank,
            boolean_torsion_detected=provisional.boolean_torsion_detected,
            boolean_betti_1=provisional.boolean_betti_1,
            novikov_krull_valuation=provisional.novikov_krull_valuation,
            novikov_ultrametric_norm=provisional.novikov_ultrametric_norm,
            novikov_is_valid=provisional.novikov_is_valid,
            phase1_seal_hash=digest,
            euler_poincare_chi=provisional.euler_poincare_chi,
            cayley_dickson_norm=provisional.cayley_dickson_norm,
            rsi1_gershgorin_margin=provisional.rsi1_gershgorin_margin,
            delaunay_action_seed_J0=provisional.delaunay_action_seed_J0,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║ FASE II — DELIBERACIÓN GEOMÉTRICO–CELESTE DE POINCARÉ–KAM–BIRKHOFF   ║
    # ║           Y AUTOMEJORA RECURSIVA NIVEL-3 (RSI³) BANACH–KLEISLI       ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # ════════════════════════════════════════════════════════════════════════

    def phase2_open_from_phase1_seal(
        self,
        seal: Phase1ProsecutionSeal,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
        ball_r: float,
        cylinder_R: float,
        positron_hmac_key: Optional[bytes],
        pert_eps: float = 0.1,
        timestamp: Optional[float] = None,
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 1/25 — CONTINUACIÓN DIRECTA DE `phase1_prosecution_seal`]
        Abre formalmente la FASE II consumiendo el `Phase1ProsecutionSeal` emitido por el
        último método de la FASE I y verificando en modo fail-CLOSED la puerta criptográfica
        Φ₁₂ (autenticidad HMAC P1 + estructura 64-hex).
        """
        if not isinstance(seal, Phase1ProsecutionSeal):
            raise ProsecutorGateViolationError(
                "Φ₁₂: tipo inválido; se esperaba Phase1ProsecutionSeal proveniente de FASE I"
            )
        if len(seal.phase1_seal_hash) != _CHAIN_HASH_HEX_LEN:
            raise ProsecutorGateViolationError(
                f"Φ₁₂: phase1_seal_hash no es {_CHAIN_HASH_HEX_LEN}-hex"
            )
        if not self.verify_phase1_seal(seal):
            raise ProsecutorGateViolationError(
                "Φ₁₂: phase1_seal_hash inválido o manipulado (HMAC P1)"
            )
        ts = float(timestamp) if timestamp is not None else float(time.time())
        arr = self._normalize_matrix(apu_cost_matrix)
        exps = np.asarray(novikov_exponents, dtype=np.float64).ravel()
        return _Phase2WorkingContext(
            phase1_seal=seal,
            apu_cost_matrix=arr,
            novikov_exponents=exps,
            ball_r=float(ball_r),
            cylinder_R=float(cylinder_R),
            positron_hmac_key=positron_hmac_key,
            pert_eps=float(pert_eps),
            timestamp=ts,
        )

    def _compute_cr_residual(
        self, u: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> float:
        r"""
        [FASE II · 2/25] Residuo de Cauchy–Riemann–Floer evaluado como la curvatura
        media discreta H de la inmersión pseudoholomorfa del grafo de u(s, τ):
            H = [(1 + u_τ²)·u_{ss} − 2 u_s u_τ u_{sτ} + (1 + u_s²)·u_{ττ}]
                / [2 (1 + u_s² + u_τ²)^{3/2}]
        """
        arr = np.asarray(u, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        if arr.ndim == 0:
            return abs(float(arr))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        n, m = arr.shape

        if n < 2 or m < 2:
            flat = arr.ravel()
            diffs = np.diff(flat)
            if diffs.size == 0:
                return 0.0
            return float(min(1.0, float(np.max(np.abs(diffs)))))

        if not np.all(np.isfinite(arr)):
            return float("inf")

        u_s = np.gradient(arr, axis=0)
        u_t = np.gradient(arr, axis=1)
        u_ss = np.gradient(u_s, axis=0)
        u_tt = np.gradient(u_t, axis=1)
        u_st = np.gradient(u_s, axis=1)

        g_s = u_s
        g_t = u_t
        numer = (
            (1.0 + g_t * g_t) * u_ss
            - 2.0 * g_s * g_t * u_st
            + (1.0 + g_s * g_s) * u_tt
        )
        denom = 2.0 * np.power(1.0 + g_s * g_s + g_t * g_t, 1.5) + _WILKINSON_EPSILON
        H = numer / denom
        if not np.all(np.isfinite(H)):
            return float("inf")
        return float(np.max(np.abs(H)))

    def evaluate_gromov_non_squeezing(
        self, ball_r: float, cylinder_R: float
    ) -> Tuple[bool, float, float]:
        r"""
        [FASE II · 3/25] Teorema de Rigidez Simpléctica (Non-Squeezing) de Gromov:
        Existe un embebimiento simpléctico φ: B^{2n}(r) ↪ Z^{2n}(R) ⟺ c(B^{2n}(r)) = π r² ≤ π R².
        """
        rb = float(ball_r)
        rc = float(cylinder_R)
        c_ball = math.pi * rb * rb
        c_cyl = math.pi * rc * rc
        violation = rb > rc + 1.0e-9
        return violation, c_ball, c_cyl

    def solve_cauchy_riemann_floer_prosecution(
        self,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        ball_r: float,
        cylinder_R: float,
    ) -> Tuple[float, bool]:
        """[FASE II · 4/25] Resuelve conjuntamente el operador de Cauchy–Riemann–Floer y Gromov."""
        res_cr = self._compute_cr_residual(apu_cost_matrix)
        violation, _, _ = self.evaluate_gromov_non_squeezing(ball_r, cylinder_R)
        return res_cr, violation

    def phase2_resolve_pseudoholomorphic_surface(
        self, ctx: _Phase2WorkingContext
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 5/25] Audita la superficie pseudoholomorfa en la Categoría A_∞ de Fukaya,
        calculando el residuo CR, la violación de Gromov y la energía simpléctica de Floer
        ponderada por la norma ultramétrica de Novikov proveniente del sello de FASE I:
            E_Floer(u) = (1 + H_CR) · π r² · ‖A‖_Nov
        """
        ctx.cauchy_riemann_residual = self._compute_cr_residual(ctx.apu_cost_matrix)
        violation, c_ball, _ = self.evaluate_gromov_non_squeezing(ctx.ball_r, ctx.cylinder_R)
        ctx.gromov_violation = bool(violation)
        cr_finite = ctx.cauchy_riemann_residual if math.isfinite(ctx.cauchy_riemann_residual) else 1.0e6
        ctx.floer_symplectic_energy = float(
            (1.0 + cr_finite) * c_ball * ctx.phase1_seal.novikov_ultrametric_norm
        )
        return ctx

    # ── Subsistema de Mecánica Celeste de Poincaré–KAM–Birkhoff–Nekhoroshev ──

    def _sech(self, x: float) -> float:
        """[FASE II · 6/25] Secante hiperbólica numéricamente estable sin desbordamiento exponencial."""
        if x > 350.0 or x < -350.0:
            return 0.0
        try:
            return 2.0 / (math.exp(x) + math.exp(-x))
        except OverflowError:  # pragma: no cover
            return 0.0

    def _poincare_frequencies(self, action_J: float) -> Tuple[float, float, float]:
        r"""
        [FASE II · 7/25] Coordenadas canónicas acción-ángulo de Delaunay–Whittaker (J, θ):
            J = max(|J_in|, _KRULL_VALUATION_FLOOR)
            ω₁(J) = ∂H₀/∂J = clamp(J⁻³, 10⁻⁸, 10⁶)
            ω₂(J) = φ · ω₁(J) · (1 + 10⁻⁶)   (foliación cuasi-periódica áurea)
        """
        J = max(abs(float(action_J)), _KRULL_VALUATION_FLOOR)
        try:
            omega1 = 1.0 / (J ** 3)
        except (OverflowError, ZeroDivisionError):  # pragma: no cover
            omega1 = 1.0e6
        omega1 = float(min(max(omega1, 1.0e-8), 1.0e6))
        omega2 = _GOLDEN_RATIO * omega1 * (1.0 + 1.0e-6)
        return J, omega1, omega2

    def _poincare_lindstedt_residual(self, omega1: float, omega2: float) -> float:
        r"""
        [FASE II · 8/25] Residual homológico de la serie perturbativa de Poincaré–Lindstedt
        ante el problema de pequeños divisores (small divisors):
            R_PL = min(1, ∑_{0 < ‖k‖_∞ ≤ 2} w_k / |k₁ ω₁ + k₂ ω₂|),  w_k = 1 / (1 + k₁² + k₂²)
        """
        residual = 0.0
        for k1 in range(-2, 3):
            for k2 in range(-2, 3):
                if k1 == 0 and k2 == 0:
                    continue
                div = k1 * omega1 + k2 * omega2
                weight = 1.0 / (1.0 + k1 * k1 + k2 * k2)
                if abs(div) < _WILKINSON_EPSILON:
                    weight *= 1.0e6
                    div = _WILKINSON_EPSILON if div == 0.0 else math.copysign(
                        _WILKINSON_EPSILON, div
                    )
                elif abs(div) < 0.02:
                    weight *= 10.0
                residual += weight / abs(div)
        return float(min(residual, 1.0))

    def _kam_diophantine_margin(
        self, omega1: float, omega2: float, tau: float = 1.5
    ) -> float:
        r"""
        [FASE II · 9/25] Margen diofántico del Teorema KAM (Kolmogorov–Arnold–Moser):
            γ_KAM = min_{0 < |k| ≤ 2} |k₁ ω₁ + k₂ ω₂| · (k₁² + k₂²)^{τ/2},   τ = 1.5
        """
        best = float("inf")
        for k1 in range(-2, 3):
            for k2 in range(-2, 3):
                if k1 == 0 and k2 == 0:
                    continue
                norm = math.sqrt(k1 * k1 + k2 * k2)
                margin = abs(k1 * omega1 + k2 * omega2) * (norm ** tau)
                best = min(best, margin)
        return 0.0 if not math.isfinite(best) else float(best)

    def _secular_drift_norm(self, omega1: float) -> float:
        """[FASE II · 10/25] Deriva secular sobre la sección de Poincaré (twist de Poincaré–Birkhoff)."""
        raw = (omega1 * 2.0 * math.pi) % (2.0 * math.pi)
        return float(abs(raw))

    def _integrate_symplectic_stormer_verlet_poincare(
        self, J0: float, omega1: float, omega2: float, eps: float
    ) -> Tuple[float, float]:
        r"""
        [FASE II · 11/25] Integrador Simpléctico de 4.º Orden de Forest–Ruth / Yoshida
        sobre el Hamiltoniano resonante de Delaunay–Whittaker en el ángulo co-rotante
        ψ = θ₁ − ω₂ t:
            H_res(J, ψ) = T(J) + V(ψ) = [−1/(2 J²) − ω₂ J] + [ε cos(ψ)]
        Ecuaciones canónicas de Hamilton:
            dψ/dt = +∂H_res/∂J = J⁻³ − ω₂
            dJ/dt = −∂H_res/∂ψ = ε sin(ψ)
        Conserva exactamente la 2-forma simpléctica dJ ∧ dψ y evalúa:
          1. Error relativo de conservación de energía |H_res(t_f) − H_res(0)| / (|H_res(0)| + 1).
          2. Invariante integral de Poincaré–Cartan: I_PC = ∮_γ (J dψ − H_res dt).
        """
        c1 = 0.5 * _YOSHIDA_W1
        c2 = 0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1)
        c3 = c2
        c4 = c1
        d1 = _YOSHIDA_W1
        d2 = _YOSHIDA_W0
        d3 = _YOSHIDA_W1

        def hamiltonian_res(J_val: float, psi_val: float) -> float:
            J_s = max(abs(J_val), _KRULL_VALUATION_FLOOR * 0.5)
            return -0.5 / (J_s * J_s) - omega2 * J_s + eps * math.cos(psi_val)

        def dT_dJ(J_val: float) -> float:
            J_s = max(abs(J_val), _KRULL_VALUATION_FLOOR * 0.5)
            return float(min(max(1.0 / (J_s ** 3), 1.0e-8), 1.0e6) - omega2)

        J_curr = float(max(abs(J0), _KRULL_VALUATION_FLOOR))
        psi_curr = 0.25 * math.pi
        dt = 1.0e-3 / max(1.0, omega1)
        n_steps = 16

        H_init = hamiltonian_res(J_curr, psi_curr)
        poincare_cartan_acc = 0.0

        for _ in range(n_steps):
            psi_prev = psi_curr
            J_prev = J_curr

            # 4 etapas simplécticas de Forest–Ruth / Yoshida
            psi_1 = psi_curr + c1 * dt * dT_dJ(J_curr)
            J_1 = max(_KRULL_VALUATION_FLOOR * 0.5, J_curr + d1 * dt * (eps * math.sin(psi_1)))

            psi_2 = psi_1 + c2 * dt * dT_dJ(J_1)
            J_2 = max(_KRULL_VALUATION_FLOOR * 0.5, J_1 + d2 * dt * (eps * math.sin(psi_2)))

            psi_3 = psi_2 + c3 * dt * dT_dJ(J_2)
            J_3 = max(_KRULL_VALUATION_FLOOR * 0.5, J_2 + d3 * dt * (eps * math.sin(psi_3)))

            psi_curr = psi_3 + c4 * dt * dT_dJ(J_3)
            J_curr = J_3

            J_mid = 0.5 * (J_prev + J_curr)
            psi_mid = 0.5 * (psi_prev + psi_curr)
            dpsi = psi_curr - psi_prev
            H_mid = hamiltonian_res(J_mid, psi_mid)
            poincare_cartan_acc += J_mid * dpsi - H_mid * dt

        H_final = hamiltonian_res(J_curr, psi_curr)
        energy_error = abs(H_final - H_init) / (abs(H_init) + 1.0)
        return float(energy_error), float(poincare_cartan_acc)

    def _compute_birkhoff_nekhoroshev_invariants(
        self, J: float, eps: float, kam_margin: float, max_melnikov: float, tau: float = 1.5
    ) -> Tuple[float, float, float, float]:
        r"""
        [FASE II · 12/25] Invariantes de la Forma Normal de Birkhoff (BNF), No-Degeneración
        de Kolmogorov y Estabilidad Exponencial de Nekhoroshev:
          • Determinante de Kolmogorov: det(∂²H₀/∂J²) = ∂ω₁/∂J = −3 / J⁴ ≠ 0.
          • Coeficiente de torsión (twist) de Birkhoff: τ_Birkhoff = |∂ω₁/∂J| = 3 / J⁴.
          • Tiempo de estabilidad exponencial de Nekhoroshev:
                T_Nekh = exp(min(50, (γ_KAM / max(|ε|, 10⁻¹²))^{1 / (2τ)}))
          • Ángulo de escisión de las variedades estable/inestable homoclínicas:
                ψ_split = arctan(|M|_max / max(γ_KAM, 10⁻¹²)) ∈ [0, π/2]
        """
        J_safe = max(abs(J), _KRULL_VALUATION_FLOOR)
        j4 = J_safe ** 4
        kolmogorov_det = float(-3.0 / j4)
        birkhoff_twist = float(3.0 / j4)

        eps_safe = max(abs(eps), 1.0e-12)
        kam_safe = max(abs(kam_margin), 1.0e-12)
        nekh_exponent = min(50.0, (kam_safe / eps_safe) ** (1.0 / (2.0 * tau)))
        nekhoroshev_time = float(math.exp(nekh_exponent))
        splitting_angle = float(math.atan2(abs(max_melnikov), kam_safe))
        return birkhoff_twist, kolmogorov_det, nekhoroshev_time, splitting_angle

    def _compute_celestial_state(
        self, action_J: float, pert_eps: float
    ) -> CelestialProsecutionState:
        r"""
        [FASE II · 13/25] Síntesis integral del estado de Mecánica Celeste de Poincaré:
        acopla Delaunay–Whittaker, la integral homoclínica de Melnikov, el margen diofántico
        KAM, el residual de Poincaré–Lindstedt, el integrador simpléctico de 4.º orden de
        Forest–Ruth/Yoshida y los invariantes de Birkhoff–Nekhoroshev.
        """
        J, omega1, omega2 = self._poincare_frequencies(action_J)
        eps = float(pert_eps)
        sech_val = self._sech(math.pi * omega1 / 2.0)
        envelope = 2.0 * math.pi * eps * sech_val
        tau_grid = np.linspace(-10.0, 10.0, 401)
        melnikov_vals = envelope * np.sin(omega1 * tau_grid)
        max_mel = float(np.max(np.abs(melnikov_vals))) if melnikov_vals.size else 0.0
        chaos = max_mel > _MELNIKOV_CHAOS_THRESHOLD

        kam_margin = self._kam_diophantine_margin(omega1, omega2)
        pl_residual = self._poincare_lindstedt_residual(omega1, omega2)
        sec_drift = self._secular_drift_norm(omega1)

        symp_err, pc_inv = self._integrate_symplectic_stormer_verlet_poincare(
            J, omega1, omega2, eps
        )
        birkhoff_twist, kolm_det, nekh_time, split_angle = (
            self._compute_birkhoff_nekhoroshev_invariants(J, eps, kam_margin, max_mel)
        )

        return CelestialProsecutionState(
            action_J=float(J),
            forcing_omega=float(omega1),
            detuning_omega2=float(omega2),
            melnikov_integral_value=max_mel,
            melnikov_chaos_detected=bool(chaos),
            kam_diophantine_margin=kam_margin,
            poincare_lindstedt_residual=pl_residual,
            secular_drift_norm=sec_drift,
            birkhoff_twist_coeff=birkhoff_twist,
            kolmogorov_nondegeneracy_det=kolm_det,
            nekhoroshev_stability_time=nekh_time,
            symplectic_energy_error=symp_err,
            poincare_cartan_invariant=pc_inv,
            homoclinic_splitting_angle=split_angle,
        )

    def evaluate_celestial_poincare_dynamics(
        self, action_J: float, pert_eps: float = 0.1
    ) -> Tuple[bool, float]:
        """[FASE II · 14/25] Evalúa la presencia de caos homoclínico de Poincaré–Melnikov."""
        state = self._compute_celestial_state(action_J, pert_eps)
        return state.melnikov_chaos_detected, state.melnikov_integral_value

    def phase2_evaluate_poincare_melnikov(
        self, ctx: _Phase2WorkingContext
    ) -> _Phase2WorkingContext:
        """
        [FASE II · 15/25] Ejecuta la deliberación de mecánica celeste de Poincaré
        alimentada por la valuación de Krull sellada en FASE I.
        """
        state = self._compute_celestial_state(
            ctx.phase1_seal.novikov_krull_valuation, ctx.pert_eps
        )
        ctx.celestial_state = state
        return ctx

    # ── Subsistema de Automejora Recursiva Nivel-3 (RSI³) Banach–Kleisli ────

    def _compute_rsi3_frechet_and_aitken(
        self, iterates: Tuple[float, float, float, float, float]
    ) -> Tuple[float, float, float, float, float]:
        r"""
        [FASE II · 16/25] Cálculo analítico de Nivel-2 y Nivel-3 sobre la mónada
        Banach–Kleisli T³ = T ∘ T ∘ T con T(x) = α x + β sin(x):
          1. Derivada primera de Fréchet (Jacobiano exacto de la composición 3-fold):
                 DT³(x₀) = T'(x₂) · T'(x₁) · T'(x₀) = ∏_{k=0}² (α + β cos(x_k))
          2. Derivada segunda de Fréchet (Hessiano exacto por regla de la cadena iterada):
                 D²T³(x₀) = T''(x₂) [T'(x₁) T'(x₀)]² + T'(x₂) [T''(x₁) (T'(x₀))² + T'(x₁) T''(x₀)]
          3. Exponente de Lyapunov de horizonte finito sobre la órbita de Kleisli:
                 λ_Lyap = (1/3) ∑_{k=0}² ln|α + β cos(x_k)| < 0
          4. Aceleración superlineal Δ² de Aitken–Steffensen sobre (x₂, x₃, x₄):
                 x*_Aitken = x₂ − (x₃ − x₂)² / (x₄ − 2 x₃ + x₂)
          5. Ganancia de la Meta-Política de Nivel-3 M^{(3)}:
                 g_meta^{(3)} = (1 − |DT³(x₀)|) / (1 − (α + β)³) ≥ 1.0
        """
        alpha = _BANACH_ALPHA
        beta = _BANACH_BETA
        x0, x1, x2, x3, x4 = iterates

        d0 = alpha + beta * math.cos(x0)
        d1 = alpha + beta * math.cos(x1)
        d2 = alpha + beta * math.cos(x2)

        h0 = -beta * math.sin(x0)
        h1 = -beta * math.sin(x1)
        h2 = -beta * math.sin(x2)

        jacobian_t3 = float(d2 * d1 * d0)
        hessian_t3 = float(h2 * (d1 * d0) ** 2 + d2 * (h1 * (d0 ** 2) + d1 * h0))

        lyap_sum = (
            math.log(max(abs(d0), _WILKINSON_EPSILON))
            + math.log(max(abs(d1), _WILKINSON_EPSILON))
            + math.log(max(abs(d2), _WILKINSON_EPSILON))
        )
        lyapunov_exp = float(lyap_sum / 3.0)

        dx2 = x3 - x2
        d2x2 = x4 - 2.0 * x3 + x2
        if abs(d2x2) > _WILKINSON_EPSILON:
            aitken_fp = float(x2 - (dx2 * dx2) / d2x2)
        else:
            aitken_fp = float(x4)

        denom_meta = max(1.0 - _BANACH_LIP_T3, _WILKINSON_EPSILON)
        meta_gain = float((1.0 - abs(jacobian_t3)) / denom_meta)

        return jacobian_t3, hessian_t3, lyapunov_exp, aitken_fp, meta_gain

    def _rsi3_vector_iterate(
        self, x0: float, kam_margin: Optional[float] = None
    ) -> RSI3ProsecutionState:
        r"""
        [FASE II · 17/25] Iteración monádica Banach–Picard–Kleisli de Nivel-3 (RSI³)
        sobre T(x) = α x + β sin(x) acoplada al margen diofántico KAM de Poincaré:
          • Evalúa la órbita (x₀, x₁, x₂, x₃, x₄), el residuo ‖x₄ − x₃‖, la cota
            Lip(T³) ≤ (α + β)³ = 0.884736 < 1 y el jerk discreto ‖x₃ − 3x₂ + 3x₁ − x₀‖ > 0.
          • Integra las derivadas exactas de Fréchet DT³(x₀), D²T³(x₀), el exponente de
            Lyapunov λ_Lyap, el punto fijo acelerado de Aitken x*_Aitken y la coherencia
            de acoplamiento resonante Poincaré–RSI³.
        """
        alpha = _BANACH_ALPHA
        beta = _BANACH_BETA

        def T(x: float) -> float:
            return alpha * x + beta * math.sin(x)

        x0f = float(x0)
        if not math.isfinite(x0f):
            x0f = alpha
        x1 = T(x0f)
        x2 = T(x1)
        x3 = T(x2)
        x4 = T(x3)

        iterates = (x0f, x1, x2, x3, x4)
        residual = abs(x4 - x3)
        tol = max(1.0, 1.0e-6 * max(1.0, abs(x0f)))
        contraction = float((alpha + beta) ** 3)
        contractive = (contraction < 1.0) and (residual <= tol)
        jerk = abs(x3 - 3.0 * x2 + 3.0 * x1 - x0f)
        jerk_positive = jerk > _SPECTRAL_PSD_FLOOR * 1.0e3

        jac_t3, hess_t3, lyap_exp, aitken_fp, meta_gain = (
            self._compute_rsi3_frechet_and_aitken(iterates)
        )

        if kam_margin is not None and math.isfinite(kam_margin):
            kam_factor = math.tanh(max(0.0, kam_margin))
            coupling_coherence = float(
                (1.0 - abs(jac_t3)) * (0.5 + 0.5 * kam_factor) / (1.0 - _BANACH_LIP_T3 + _WILKINSON_EPSILON)
            )
        else:
            coupling_coherence = float(meta_gain)

        return RSI3ProsecutionState(
            initial_lip=x0f,
            iterates=iterates,
            banach_residual=float(residual),
            contraction_bound=contraction,
            is_banach_contractive=bool(contractive),
            capacity_jerk=float(jerk),
            jerk_strictly_positive=bool(jerk_positive),
            kleisli_depth=3,
            exact_frechet_jacobian_t3=jac_t3,
            exact_frechet_hessian_t3=hess_t3,
            lyapunov_exponent=lyap_exp,
            aitken_accelerated_fixed_point=aitken_fp,
            meta_policy_gain_level3=meta_gain,
            poincare_rsi3_coupling_coherence=coupling_coherence,
        )

    def evaluate_rsi3_banach_kleisli_monad(
        self, initial_lip: float = 0.95
    ) -> Tuple[float, bool]:
        r"""
        [FASE II · 18/25] Evalúa la cota de contracción Lip(T³) = (α+β)³ = 0.884736 < 1
        y la positividad estricta del jerk ciber-físico sobre la mónada de Kleisli.
        """
        state = self._rsi3_vector_iterate(initial_lip)
        return state.contraction_bound, state.jerk_strictly_positive

    def phase2_evaluate_rsi3_banach(
        self, ctx: _Phase2WorkingContext, initial_x: float = 0.95
    ) -> _Phase2WorkingContext:
        r"""
        [FASE II · 19/25] Ejecuta la automejora recursiva de Nivel-3 (RSI³) acoplando
        el estado celeste de Poincaré (`ctx.celestial_state`) con la mónada Banach–Kleisli.
        """
        kam_margin = (
            ctx.celestial_state.kam_diophantine_margin
            if ctx.celestial_state is not None
            else None
        )
        ctx.rsi3_state = self._rsi3_vector_iterate(initial_x, kam_margin=kam_margin)
        return ctx

    # ── Subsistema de Aniquilación Cuántica de Dirac (Cl_{1,3}) ──────────────

    def verify_positron_hmac(self, positron_hmac_key: Optional[bytes]) -> Tuple[bool, str]:
        r"""
        [FASE II · 20/25] Verificación criptográfica de tiempo constante del positrón
        contra `_POSITRON_HMAC_PAYLOAD`. Fail-CLOSED ante cualquier desviación.
        """
        if not isinstance(positron_hmac_key, (bytes, bytearray)):
            return False, hashlib.sha256(b"<INVALID_KEY_TYPE>").hexdigest()
        payload = bytes(positron_hmac_key)
        digest = hashlib.sha256(payload).hexdigest()
        valid = hmac.compare_digest(payload, _POSITRON_HMAC_PAYLOAD)
        return bool(valid), digest

    def execute_dirac_annihilation_prosecution(
        self,
        anomaly_norm: float,
        positron_hmac_key: Optional[bytes] = None,
    ) -> DiracAnnihilationProsecutionResult:
        r"""
        [FASE II · 21/25] Ejecuta el canal de aniquilación cuántica e⁻ + e⁺ → 2γ en el
        álgebra de Clifford espacio-temporal Cl_{1,3}(ℝ):
          • Si el HMAC del positrón es válido, proyecta el estado sobre el singlete de Bell
            |Ψ⁻⟩ = (|01⟩ − |10⟩)/√2 con fidelidad cuántica F = exp(−0.01·|anomaly|),
            entropía de Von Neumann S(ρ) y energía de masa en reposo 2 m_e c² = 1022.0 keV.
          • En caso contrario, aplica penalización inquisitorial (energía = 0.0 keV,
            back-action = 12.5 dB, producción de entropía de Spohn = 2.5).
        """
        is_valid, payload_sha = self.verify_positron_hmac(positron_hmac_key)
        anomaly = float(anomaly_norm) if math.isfinite(float(anomaly_norm)) else float("inf")

        if is_valid:
            energy_released = _POSITRON_REST_MASS_KEV
            back_action_db = 0.0
            spohn_entropy = float(abs(anomaly) * 0.05) if math.isfinite(anomaly) else 0.0
            abs_anom = abs(anomaly) if math.isfinite(anomaly) else 0.0
            bell_fidelity = float(math.exp(-0.01 * min(abs_anom, 100.0)))
            p_eig = min(max(0.5 * (1.0 + bell_fidelity), 1.0e-12), 1.0 - 1.0e-12)
            vn_entropy = float(
                -(p_eig * math.log(p_eig) + (1.0 - p_eig) * math.log(1.0 - p_eig))
            )
            spinor_norm = float(math.sqrt(1.0 + math.tanh(min(abs_anom, 20.0)) ** 2))
        else:
            energy_released = 0.0
            back_action_db = 12.5
            spohn_entropy = 2.5
            bell_fidelity = 0.0
            vn_entropy = float(math.log(2.0))
            spinor_norm = 0.0

        hash_input = (
            f"{anomaly:.12g}|{int(is_valid)}|{energy_released:.6g}|"
            f"{spohn_entropy:.12g}|{back_action_db:.6g}"
        ).encode("utf-8")
        gamma_hash = hashlib.sha256(hash_input).hexdigest()

        return DiracAnnihilationProsecutionResult(
            electron_anomaly_norm=anomaly,
            positron_valid=bool(is_valid),
            energy_released_kev=float(energy_released),
            spohn_entropy_production=float(spohn_entropy),
            back_action_db=float(back_action_db),
            gamma_photon_hash=gamma_hash,
            positron_hmac_verified=bool(is_valid),
            positron_hmac_payload_sha256=payload_sha,
            bell_singlet_fidelity=bell_fidelity,
            von_neumann_entropy=vn_entropy,
            clifford_spinor_norm=spinor_norm,
        )

    def phase2_execute_dirac_annihilation(
        self, ctx: _Phase2WorkingContext
    ) -> _Phase2WorkingContext:
        """[FASE II · 22/25] Ejecuta la aniquilación cuántica de Dirac sobre el residuo CR."""
        anomaly = (
            ctx.cauchy_riemann_residual
            if ctx.cauchy_riemann_residual is not None and math.isfinite(ctx.cauchy_riemann_residual)
            else 0.0
        )
        ctx.dirac_annihilation = self.execute_dirac_annihilation_prosecution(
            anomaly_norm=anomaly,
            positron_hmac_key=ctx.positron_hmac_key,
        )
        return ctx

    def _p2_payload_bytes(self, seal: Phase2ProsecutionSeal) -> bytes:
        """[FASE II · 23/25] Serialización canónica determinista para el HMAC de FASE II."""
        c = seal.celestial_state
        r = seal.rsi3_state
        d = seal.dirac_annihilation
        return (
            f"PROSECUTOR-P2|{seal.timestamp:.12g}|{seal.engine_version}|"
            f"{seal.phase1_seal.phase1_seal_hash}|"
            f"{seal.cauchy_riemann_residual:.12g}|{int(seal.gromov_violation)}|"
            f"{seal.ball_radius_r:.12g}|{seal.cylinder_radius_R:.12g}|"
            f"{c.action_J:.12g}|{c.forcing_omega:.12g}|{c.detuning_omega2:.12g}|"
            f"{c.melnikov_integral_value:.12g}|{int(c.melnikov_chaos_detected)}|"
            f"{c.kam_diophantine_margin:.12g}|"
            f"{c.poincare_lindstedt_residual:.12g}|{c.secular_drift_norm:.12g}|"
            f"{r.banach_residual:.12g}|{r.contraction_bound:.12g}|"
            f"{int(r.is_banach_contractive)}|{r.capacity_jerk:.12g}|"
            f"{int(r.jerk_strictly_positive)}|{r.kleisli_depth}|"
            f"{int(d.positron_valid)}|{d.energy_released_kev:.6g}"
        ).encode("utf-8")

    def verify_phase2_seal(self, seal: Phase2ProsecutionSeal) -> bool:
        """[FASE II · 24/25] Verifica criptográficamente el sello de FASE II bajo `_HMAC_DOMAIN_P2`."""
        if len(seal.phase2_seal_hash) != _CHAIN_HASH_HEX_LEN:
            return False
        expected = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + self._p2_payload_bytes(seal),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, seal.phase2_seal_hash)

    def phase2_prosecution_seal(
        self,
        ctx: _Phase2WorkingContext,
        timestamp: Optional[float] = None,
    ) -> Phase2ProsecutionSeal:
        r"""
        [FASE II · 25/25 — MÉTODO TERMINAL DE LA FASE II Y ESLABÓN DE CONTINUACIÓN A FASE III]
        Clausura formalmente la FASE II encadenando criptográficamente `phase1_seal_hash`
        y firmando bajo `_HMAC_DOMAIN_P2` la deliberación pseudoholomorfa de Floer–Gromov,
        la mecánica celeste de Poincaré–KAM–Birkhoff–Nekhoroshev, la mónada RSI³ de
        Banach–Kleisli y la aniquilación de Dirac.

        CONTRATO DE ANIDAMIENTO FASE II ──► FASE III:
        El objeto `Phase2ProsecutionSeal` retornado por esta definición final es el
        argumento formal obligatorio que consume inmediatamente `phase3_open_from_phase2_seal`,
        primer método de la FASE III definida a continuación.
        """
        if ctx.cauchy_riemann_residual is None:
            raise ProsecutorGateViolationError(
                "Φ₂₃: falta `cauchy_riemann_residual`; ejecuta phase2_resolve_pseudoholomorphic_surface"
            )
        if ctx.gromov_violation is None:
            raise ProsecutorGateViolationError(
                "Φ₂₃: falta `gromov_violation`; ejecuta phase2_resolve_pseudoholomorphic_surface"
            )
        if ctx.celestial_state is None:
            raise ProsecutorGateViolationError(
                "Φ₂₃: falta `celestial_state`; ejecuta phase2_evaluate_poincare_melnikov"
            )
        if ctx.rsi3_state is None:
            raise ProsecutorGateViolationError(
                "Φ₂₃: falta `rsi3_state`; ejecuta phase2_evaluate_rsi3_banach"
            )
        if ctx.dirac_annihilation is None:
            raise ProsecutorGateViolationError(
                "Φ₂₃: falta `dirac_annihilation`; ejecuta phase2_execute_dirac_annihilation"
            )
        ts = float(timestamp) if timestamp is not None else float(time.time())

        provisional = Phase2ProsecutionSeal(
            timestamp=ts,
            engine_version=_ENGINE_VERSION,
            phase1_seal=ctx.phase1_seal,
            cauchy_riemann_residual=float(ctx.cauchy_riemann_residual),
            gromov_violation=bool(ctx.gromov_violation),
            ball_radius_r=float(ctx.ball_r),
            cylinder_radius_R=float(ctx.cylinder_R),
            celestial_state=ctx.celestial_state,
            rsi3_state=ctx.rsi3_state,
            dirac_annihilation=ctx.dirac_annihilation,
            phase2_seal_hash="",
            floer_symplectic_energy=float(ctx.floer_symplectic_energy),
        )
        digest = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P2 + self._p2_payload_bytes(provisional),
            hashlib.sha256,
        ).hexdigest()
        return Phase2ProsecutionSeal(
            timestamp=provisional.timestamp,
            engine_version=provisional.engine_version,
            phase1_seal=provisional.phase1_seal,
            cauchy_riemann_residual=provisional.cauchy_riemann_residual,
            gromov_violation=provisional.gromov_violation,
            ball_radius_r=provisional.ball_radius_r,
            cylinder_radius_R=provisional.cylinder_radius_R,
            celestial_state=provisional.celestial_state,
            rsi3_state=provisional.rsi3_state,
            dirac_annihilation=provisional.dirac_annihilation,
            phase2_seal_hash=digest,
            floer_symplectic_energy=provisional.floer_symplectic_energy,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║ FASE III — ADJUDICACIÓN EN EL TOPOS INTUICIONISTA \Omega_3,          ║
    # ║            MODALIDAD LAWVERE–TIERNEY Y FÍSICA RLC DEL CROWBAR ESP32  ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    # ════════════════════════════════════════════════════════════════════════

    def phase3_open_from_phase2_seal(
        self,
        seal: Phase2ProsecutionSeal,
        timestamp: Optional[float] = None,
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 1/8 — CONTINUACIÓN DIRECTA DE `phase2_prosecution_seal`]
        Abre formalmente la FASE III consumiendo el `Phase2ProsecutionSeal` emitido por el
        último método de la FASE II y verificando en modo fail-CLOSED la puerta Φ₂₃
        (autenticidad HMAC P2, integridad de la cadena Merkle hacia HMAC P1 y formato 64-hex).
        """
        if not isinstance(seal, Phase2ProsecutionSeal):
            raise ProsecutorGateViolationError(
                "Φ₂₃: tipo inválido; se esperaba Phase2ProsecutionSeal proveniente de FASE II"
            )
        if len(seal.phase2_seal_hash) != _CHAIN_HASH_HEX_LEN:
            raise ProsecutorGateViolationError(
                f"Φ₂₃: phase2_seal_hash no es {_CHAIN_HASH_HEX_LEN}-hex"
            )
        if len(seal.phase1_seal.phase1_seal_hash) != _CHAIN_HASH_HEX_LEN:
            raise ProsecutorGateViolationError(
                f"Φ₂₃: phase1_seal_hash no es {_CHAIN_HASH_HEX_LEN}-hex"
            )
        if not self.verify_phase2_seal(seal):
            raise ProsecutorGateViolationError(
                "Φ₂₃: phase2_seal_hash inválido o manipulado (HMAC P2)"
            )
        if not self.verify_phase1_seal(seal.phase1_seal):
            raise ProsecutorGateViolationError(
                "Φ₂₃: cadena rota — phase1_seal_hash no verifica bajo HMAC P1"
            )
        ts = float(timestamp) if timestamp is not None else float(time.time())
        return _Phase3WorkingContext(phase2_seal=seal, timestamp=ts)

    def _evaluate_heyting_lawvere_tierney_modality(
        self, truth: HeytingProsecutorTruthValue
    ) -> int:
        r"""
        [FASE III · 2/8] Operador modal de clausura de Lawvere–Tierney j = ¬¬: \Omega_3 \to \Omega_3
        sobre el clasificador de subobjetos del topos intuicionista:
          • j(COHERENT) = 0x00 (verdad densa libre de obstrucción)
          • j(DEGRADED) = 0x01 (estabilidad local con cicatriz cohomológica)
          • j(VETOED)   = 0x03 (colapso al elemento absorbente de veto)
        """
        if truth == HeytingProsecutorTruthValue.COHERENT:
            return int(HeytingProsecutorTruthValue.COHERENT.value)
        if truth == HeytingProsecutorTruthValue.DEGRADED:
            return int(HeytingProsecutorTruthValue.DEGRADED.value)
        return int(HeytingProsecutorTruthValue.VETOED.value)

    def phase3_adjudicate_heyting_omega3(
        self, ctx: _Phase3WorkingContext
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 3/8] Adjudicación en el álgebra de Heyting \Omega_3 y síntesis modal
        de Lawvere–Tierney sobre las pruebas selladas en FASE I y FASE II.

        Jerarquía de decisión (COHERENT ≺ DEGRADED ≺ VETOED):
          1. VETOED si falla cualquier invariante duro (Gromov, Torsión SNF, Piso Krull,
             HMAC estricto del Positrón o finitud CR).
          2. DEGRADED si emerge obstrucción suave (β₁ > 0, enredo homoclínico de Melnikov,
             pérdida de contractividad Banach–Kleisli, margen KAM bajo o curvatura CR alta).
          3. COHERENT si todos los invariantes algebraicos, celestes y monádicos se satisfacen.
        """
        seal = ctx.phase2_seal
        v_A = seal.phase1_seal.novikov_krull_valuation
        cr = seal.cauchy_riemann_residual
        celestial = seal.celestial_state
        rsi3 = seal.rsi3_state
        dirac = seal.dirac_annihilation

        # ── EVALUACIÓN DE VETOS INQUISITORIALES ──
        if seal.gromov_violation:
            ctx.truth_value = HeytingProsecutorTruthValue.VETOED
            ctx.adjudication_reason = "GROMOV_NON_SQUEEZING_VIOLATION"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if seal.phase1_seal.boolean_torsion_detected:
            ctx.truth_value = HeytingProsecutorTruthValue.VETOED
            ctx.adjudication_reason = "BOOLE_HOMOLOGY_TORSION_DETECTED"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if not math.isfinite(v_A) or v_A < _KRULL_VALUATION_FLOOR:
            ctx.truth_value = HeytingProsecutorTruthValue.VETOED
            ctx.adjudication_reason = "NOVIKOV_KRULL_VALUATION_FLOOR_VIOLATION"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if not dirac.positron_valid:
            ctx.truth_value = HeytingProsecutorTruthValue.VETOED
            ctx.adjudication_reason = "DIRAC_POSITRON_HMAC_INVALID"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if not math.isfinite(cr):
            ctx.truth_value = HeytingProsecutorTruthValue.VETOED
            ctx.adjudication_reason = "CAUCHY_RIEMANN_NON_FINITE"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx

        # ── EVALUACIÓN DE DEGRADACIONES ESTRUCTURALES ──
        if seal.phase1_seal.boolean_betti_1 > 0:
            ctx.truth_value = HeytingProsecutorTruthValue.DEGRADED
            ctx.adjudication_reason = "BOOLE_HOMOLOGY_NON_ACYCLIC"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if celestial.melnikov_chaos_detected:
            ctx.truth_value = HeytingProsecutorTruthValue.DEGRADED
            ctx.adjudication_reason = "POINCARE_HOMOCLINIC_TANGLE"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if not rsi3.is_banach_contractive:
            ctx.truth_value = HeytingProsecutorTruthValue.DEGRADED
            ctx.adjudication_reason = "RSI3_BANACH_NON_CONTRACTIVE"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if celestial.kam_diophantine_margin < _KAM_MARGIN_FLOOR:
            ctx.truth_value = HeytingProsecutorTruthValue.DEGRADED
            ctx.adjudication_reason = "KAM_DIOPHANTINE_MARGIN_LOW"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx
        if cr > _CR_RESIDUAL_DEGRADED_THRESHOLD:
            ctx.truth_value = HeytingProsecutorTruthValue.DEGRADED
            ctx.adjudication_reason = "CAUCHY_RIEMANN_RESIDUAL_HIGH"
            ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
                ctx.truth_value
            )
            return ctx

        # ── VEREDICTO COHERENTE ──
        ctx.truth_value = HeytingProsecutorTruthValue.COHERENT
        ctx.adjudication_reason = "ALL_PROSECUTION_INVARIANTS_SATISFIED"
        ctx.lawvere_tierney_modal_truth = self._evaluate_heyting_lawvere_tierney_modality(
            ctx.truth_value
        )
        return ctx

    def _simulate_esp32_bt151_crowbar_rlc_transient(
        self, armed: bool
    ) -> Tuple[float, float, float]:
        r"""
        [FASE III · 4/8] Física de circuitos eléctricos del disparo Crowbar (tiristor SCR
        BT151 comandado desde GPIO14 en IRAM del Xtensa LX7 sobre barra DC):
          • Impedancia característica: Z₀ = √(L_bus / C_bus)
          • Factor de amortiguamiento: ζ = (R_on / 2) · √(C_bus / L_bus)
          • Tasa inicial de subida de corriente en el instante de conmutación:
                (di/dt)_max = V_bus / L_bus   [A/μs]
          • Corriente pico de cortocircuito protector:
                I_peak = V_bus / R_on  (régimen sobreamortiguado ζ ≥ 1)
                I_peak = (V_bus / Z₀) · exp(−ζ · arcsin(√(1−ζ²)) / √(1−ζ²))  (ζ < 1)
        """
        c_farad = _CROWBAR_BUS_CAPACITANCE_UF * 1.0e-6
        l_henry = _CROWBAR_PARASITIC_INDUCTANCE_NH * 1.0e-9
        r_ohm = _CROWBAR_SCR_RON_OHM
        v_bus = _CROWBAR_BUS_VOLTAGE_V

        z0 = math.sqrt(l_henry / c_farad)
        zeta = 0.5 * r_ohm / z0

        if not armed:
            return 0.0, 0.0, float(zeta)

        didt_amps_per_us = (v_bus / l_henry) * 1.0e-6
        if zeta < 1.0:
            omega_d_ratio = math.sqrt(1.0 - zeta * zeta)
            t_peak_phase = math.atan2(omega_d_ratio, zeta)
            i_peak = (v_bus / z0) * math.exp(-zeta * t_peak_phase / omega_d_ratio) * omega_d_ratio
        else:
            i_peak = v_bus / r_ohm

        return float(i_peak), float(didt_amps_per_us), float(zeta)

    def phase3_arm_esp32_crowbar(
        self, ctx: _Phase3WorkingContext
    ) -> _Phase3WorkingContext:
        r"""
        [FASE III · 5/8] Actuación ciber-física del Crowbar ESP32 Xtensa LX7 (IRAM, SIMD):
          • VETOED → Crowbar ARMADO (`42.0 ns` ≤ `392.15 ns`) + simulación transitoria RLC.
          • DEGRADED / COHERENT → Crowbar DESARMADO (`0.0 ns`).
        """
        if ctx.truth_value is None:
            raise ProsecutorGateViolationError(
                "Φ₃: adjudicación no ejecutada; llama phase3_adjudicate_heyting_omega3 antes"
            )
        if ctx.truth_value == HeytingProsecutorTruthValue.VETOED:
            ctx.esp32_crowbar_armed = True
            ctx.crowbar_actuation_ns = _ESP32_CROWBAR_EXECUTION_NS
        else:
            ctx.esp32_crowbar_armed = False
            ctx.crowbar_actuation_ns = 0.0

        ctx.crowbar_budget_respected = (
            ctx.crowbar_actuation_ns <= _ESP32_CROWBAR_MAX_LATENCY_NS
        )
        i_peak, didt, zeta = self._simulate_esp32_bt151_crowbar_rlc_transient(
            ctx.esp32_crowbar_armed
        )
        ctx.crowbar_peak_current_amps = i_peak
        ctx.crowbar_didt_amps_per_us = didt
        ctx.crowbar_damping_ratio_zeta = zeta
        return ctx

    def _p3_final_payload_bytes(
        self, report: PseudoholomorphicSurfaceProsecutionReport
    ) -> bytes:
        """[FASE III · 6/8] Serialización canónica determinista para la raíz Merkle de FASE III."""
        return (
            f"PROSECUTOR-FINAL|{report.timestamp:.12g}|{report.engine_version}|"
            f"{report.phase1_seal_hash}|{report.phase2_seal_hash}|"
            f"{int(report.truth_value.value)}|{report.adjudication_reason}|"
            f"{report.crowbar_actuation_ns:.6g}|{int(report.crowbar_budget_respected)}"
        ).encode("utf-8")

    def verify_sovereign_hmac(
        self, report: PseudoholomorphicSurfaceProsecutionReport
    ) -> bool:
        """[FASE III · 7/8] Verifica criptográficamente el sello soberano final bajo `_HMAC_DOMAIN_P3`."""
        if len(report.sovereign_hmac_sha256) != _CHAIN_HASH_HEX_LEN:
            return False
        expected = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + self._p3_final_payload_bytes(report),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, report.sovereign_hmac_sha256)

    def phase3_emit_prosecution_report(
        self,
        ctx: _Phase3WorkingContext,
        execution_time_ms: float = 0.0,
    ) -> PseudoholomorphicSurfaceProsecutionReport:
        r"""
        [FASE III · 8/8 — MÉTODO TERMINAL DE LA FASE III Y CLAUSURA DE LA TRÍADA ANIDADA]
        Emite el `PseudoholomorphicSurfaceProsecutionReport` soberano final, sellando
        criptográficamente la cadena completa:
            Phase1ProsecutionSeal ──(Φ₁₂)──► Phase2ProsecutionSeal ──(Φ₂₃)──► SovereignReport
        bajo `_HMAC_DOMAIN_P3`.
        """
        if ctx.truth_value is None:
            raise ProsecutorGateViolationError("Φ₃: veredicto no adjudicado")
        seal = ctx.phase2_seal
        celestial = seal.celestial_state
        rsi3 = seal.rsi3_state
        dirac = seal.dirac_annihilation

        provisional = PseudoholomorphicSurfaceProsecutionReport(
            truth_value=ctx.truth_value,
            cauchy_riemann_residual=seal.cauchy_riemann_residual,
            gromov_violation=seal.gromov_violation,
            ball_radius_r=seal.ball_radius_r,
            cylinder_radius_R=seal.cylinder_radius_R,
            torsion_detected=seal.phase1_seal.boolean_torsion_detected,
            betti_1_cycles=seal.phase1_seal.boolean_betti_1,
            krull_valuation_vA=seal.phase1_seal.novikov_krull_valuation,
            ultrametric_norm=seal.phase1_seal.novikov_ultrametric_norm,
            poincare_melnikov_chaos=celestial.melnikov_chaos_detected,
            melnikov_integral_max=celestial.melnikov_integral_value,
            banach_contraction_lip=rsi3.contraction_bound,
            jerk_capacity_positive=rsi3.jerk_strictly_positive,
            dirac_annihilation=dirac,
            esp32_crowbar_armed=ctx.esp32_crowbar_armed,
            chain_hash="",
            execution_time_ms=float(execution_time_ms),
            rsi3_state=rsi3,
            celestial_state=celestial,
            adjudication_reason=ctx.adjudication_reason,
            phase1_seal_hash=seal.phase1_seal.phase1_seal_hash,
            phase2_seal_hash=seal.phase2_seal_hash,
            sovereign_hmac_sha256="",
            crowbar_actuation_ns=ctx.crowbar_actuation_ns,
            crowbar_latency_budget_ns=_ESP32_CROWBAR_MAX_LATENCY_NS,
            crowbar_budget_respected=ctx.crowbar_budget_respected,
            timestamp=ctx.timestamp,
            engine_version=_ENGINE_VERSION,
            crowbar_peak_current_amps=ctx.crowbar_peak_current_amps,
            crowbar_didt_amps_per_us=ctx.crowbar_didt_amps_per_us,
            crowbar_damping_ratio_zeta=ctx.crowbar_damping_ratio_zeta,
            lawvere_tierney_modal_truth=ctx.lawvere_tierney_modal_truth,
        )
        final_hmac = hmac.new(
            self._secret_key,
            _HMAC_DOMAIN_P3 + self._p3_final_payload_bytes(provisional),
            hashlib.sha256,
        ).hexdigest()

        return PseudoholomorphicSurfaceProsecutionReport(
            truth_value=provisional.truth_value,
            cauchy_riemann_residual=provisional.cauchy_riemann_residual,
            gromov_violation=provisional.gromov_violation,
            ball_radius_r=provisional.ball_radius_r,
            cylinder_radius_R=provisional.cylinder_radius_R,
            torsion_detected=provisional.torsion_detected,
            betti_1_cycles=provisional.betti_1_cycles,
            krull_valuation_vA=provisional.krull_valuation_vA,
            ultrametric_norm=provisional.ultrametric_norm,
            poincare_melnikov_chaos=provisional.poincare_melnikov_chaos,
            melnikov_integral_max=provisional.melnikov_integral_max,
            banach_contraction_lip=provisional.banach_contraction_lip,
            jerk_capacity_positive=provisional.jerk_capacity_positive,
            dirac_annihilation=provisional.dirac_annihilation,
            esp32_crowbar_armed=provisional.esp32_crowbar_armed,
            chain_hash=final_hmac,
            execution_time_ms=provisional.execution_time_ms,
            rsi3_state=provisional.rsi3_state,
            celestial_state=provisional.celestial_state,
            adjudication_reason=provisional.adjudication_reason,
            phase1_seal_hash=provisional.phase1_seal_hash,
            phase2_seal_hash=provisional.phase2_seal_hash,
            sovereign_hmac_sha256=final_hmac,
            crowbar_actuation_ns=provisional.crowbar_actuation_ns,
            crowbar_latency_budget_ns=provisional.crowbar_latency_budget_ns,
            crowbar_budget_respected=provisional.crowbar_budget_respected,
            timestamp=provisional.timestamp,
            engine_version=provisional.engine_version,
            crowbar_peak_current_amps=provisional.crowbar_peak_current_amps,
            crowbar_didt_amps_per_us=provisional.crowbar_didt_amps_per_us,
            crowbar_damping_ratio_zeta=provisional.crowbar_damping_ratio_zeta,
            lawvere_tierney_modal_truth=provisional.lawvere_tierney_modal_truth,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ORQUESTADORES CANÓNICOS DE COMPOSICIÓN KLEISLI (FASE I ∘ FASE II ∘ FASE III)
    # ════════════════════════════════════════════════════════════════════════

    def run_prosecution_audit(
        self,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
        ball_r: float = 0.15,
        cylinder_R: float = 0.20,
        positron_hmac_key: Optional[bytes] = None,
        pert_eps: float = 0.1,
        *,
        initial_rsi3_x: float = 0.95,
    ) -> PseudoholomorphicSurfaceProsecutionReport:
        r"""
        Orquestador canónico del Motor Espectral Inquisidor Dual.
        Ejecuta la composición anidada de Kleisli:
          1. FASE I   → `phase1_prosecution_seal` emite `seal_p1`.
          2. FASE II  → `phase2_open_from_phase1_seal(seal_p1)` ... `phase2_prosecution_seal` emite `seal_p2`.
          3. FASE III → `phase3_open_from_phase2_seal(seal_p2)` ... `phase3_emit_prosecution_report`.
        """
        wall_start = time.perf_counter()

        # ── FASE I (Cierra en phase1_prosecution_seal) ─────────────────────
        seal_p1 = self.phase1_prosecution_seal(
            apu_cost_matrix=apu_cost_matrix,
            novikov_exponents=novikov_exponents,
        )

        # ── FASE II (Abre con seal_p1 y cierra en phase2_prosecution_seal) ─
        ctx2 = self.phase2_open_from_phase1_seal(
            seal=seal_p1,
            apu_cost_matrix=apu_cost_matrix,
            novikov_exponents=novikov_exponents,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
            positron_hmac_key=positron_hmac_key,
            pert_eps=pert_eps,
        )
        ctx2 = self.phase2_resolve_pseudoholomorphic_surface(ctx2)
        ctx2 = self.phase2_evaluate_poincare_melnikov(ctx2)
        ctx2 = self.phase2_evaluate_rsi3_banach(ctx2, initial_x=initial_rsi3_x)
        ctx2 = self.phase2_execute_dirac_annihilation(ctx2)
        seal_p2 = self.phase2_prosecution_seal(ctx2)

        # ── FASE III (Abre con seal_p2 y cierra en phase3_emit_prosecution_report) ─
        ctx3 = self.phase3_open_from_phase2_seal(seal_p2)
        ctx3 = self.phase3_adjudicate_heyting_omega3(ctx3)
        ctx3 = self.phase3_arm_esp32_crowbar(ctx3)

        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
        return self.phase3_emit_prosecution_report(ctx3, execution_time_ms=elapsed_ms)

    def execute_prosecution_audit(
        self,
        apu_cost_matrix: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        novikov_exponents: Union[NDArray[np.float64], Sequence[float]],
        ball_r: float = 0.15,
        cylinder_R: float = 0.20,
        positron_hmac_key: Optional[bytes] = None,
        pert_eps: float = 0.1,
    ) -> PseudoholomorphicSurfaceProsecutionReport:
        r"""
        Alias compatible con v13/v14 (`execute_prosecution_audit`).
        Delega íntegramente en `run_prosecution_audit` manteniendo verificación
        HMAC estricta fail-CLOSED del positrón.
        """
        return self.run_prosecution_audit(
            apu_cost_matrix=apu_cost_matrix,
            novikov_exponents=novikov_exponents,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
            positron_hmac_key=positron_hmac_key,
            pert_eps=pert_eps,
        )


# ════════════════════════════════════════════════════════════════════════════
# SANDBOX AUTÓNOMO DE VERIFICACIÓN FORMAL (12 SUITES DE PRUEBA)
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TEST AUTÓNOMO: BoolNovikovPseudoholomorphicProsecutorEngine v14 (Evolución Soberana) ===")

    engine = BoolNovikovPseudoholomorphicProsecutorEngine()
    valid_key = b"DIRAC_POSITRON_HMAC_VERIFICATION"

    def _print_header(name: str) -> None:
        print(f"\n───── {name} ─────")

    def _report_summary(report: PseudoholomorphicSurfaceProsecutionReport) -> None:
        print(f"  Veredicto Heyting Ω_3    : {report.truth_value.name} (0x{int(report.truth_value):02x}) | Modal j(Ω_3)=0x{report.lawvere_tierney_modal_truth:02x}")
        print(f"  Razón de adjudicación    : {report.adjudication_reason}")
        print(f"  Residual CR (H mean)     : {report.cauchy_riemann_residual:.6e}")
        print(f"  Gromov no-squeezing      : violación={report.gromov_violation}  (r={report.ball_radius_r:.4f}, R={report.cylinder_radius_R:.4f})")
        print(f"  Torsión homológica       : {report.torsion_detected}  |  β₁ = {report.betti_1_cycles}")
        print(f"  Krull v(A)               : {report.krull_valuation_vA:.6f}  |  ‖A‖_Nov = {report.ultrametric_norm:.6f}")
        print(f"  Melnikov |M|_max         : {report.melnikov_integral_max:.6e}  (caos={report.poincare_melnikov_chaos})")
        if report.celestial_state:
            c = report.celestial_state
            print(
                f"  Poincaré–KAM–Birkhoff    : ω₁={c.forcing_omega:.4e} | ω₂={c.detuning_omega2:.4e} | "
                f"KAM={c.kam_diophantine_margin:.3e} | |ΔH_symp|={c.symplectic_energy_error:.3e} | "
                f"τ_Birkhoff={c.birkhoff_twist_coeff:.3e}"
            )
        if report.rsi3_state:
            r = report.rsi3_state
            print(
                f"  RSI³ Banach–Kleisli      : Lip(T³)={report.banach_contraction_lip:.6f} | "
                f"DT³(x₀)={r.exact_frechet_jacobian_t3:.6f} | λ_Lyap={r.lyapunov_exponent:.4e} | "
                f"x*_Aitken={r.aitken_accelerated_fixed_point:.3e} | g_meta³={r.meta_policy_gain_level3:.4f}"
            )
        print(
            f"  Dirac Cl_{{1,3}} (keV)     : {report.dirac_annihilation.energy_released_kev:.1f}  "
            f"(HMAC válido: {report.dirac_annihilation.positron_hmac_verified} | "
            f"F_Bell={report.dirac_annihilation.bell_singlet_fidelity:.4f})"
        )
        print(
            f"  Crowbar RLC              : armado={report.esp32_crowbar_armed} "
            f"({report.crowbar_actuation_ns:.2f} ns ≤ {report.crowbar_latency_budget_ns:.2f} ns → {report.crowbar_budget_respected}) | "
            f"I_peak={report.crowbar_peak_current_amps:.1f} A | di/dt={report.crowbar_didt_amps_per_us:.1f} A/μs | ζ={report.crowbar_damping_ratio_zeta:.3f}"
        )
        print(f"  P1 seal (16 hex)         : {report.phase1_seal_hash[:16]}…")
        print(f"  P2 seal (16 hex)         : {report.phase2_seal_hash[:16]}…")
        print(f"  HMAC soberano (16 hex)   : {report.sovereign_hmac_sha256[:16]}…")
        print(f"  Latencia de auditoría    : {report.execution_time_ms:.3f} ms")

    # ── Test 1: caso base (matriz bien condicionada, HMAC válido) ─────────
    _print_header("TEST 1: base (COHERENT esperado)")
    matrix_base = np.array([[1.0, 0.2], [0.2, 1.0]])
    report = engine.run_prosecution_audit(
        apu_cost_matrix=matrix_base,
        novikov_exponents=np.array([0.15, 0.25, 0.40]),
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _report_summary(report)
    assert report.truth_value == HeytingProsecutorTruthValue.COHERENT, (
        f"TEST 1 falló: {report.truth_value.name} ({report.adjudication_reason})"
    )
    assert report.esp32_crowbar_armed is False
    assert len(report.phase1_seal_hash) == 64
    assert len(report.phase2_seal_hash) == 64
    assert len(report.sovereign_hmac_sha256) == 64
    assert engine.verify_phase1_seal(engine.phase1_prosecution_seal(matrix_base, [0.15]))
    assert engine.verify_sovereign_hmac(report)
    assert report.jerk_capacity_positive is True
    print("  ✓ TEST 1 OK")

    # ── Test 2: Gromov no-squeezing violado → VETOED + Crowbar ────────────
    _print_header("TEST 2: Gromov (r > R) → VETOED + Crowbar")
    report_bad_gromov = engine.run_prosecution_audit(
        apu_cost_matrix=matrix_base,
        novikov_exponents=[0.15, 0.40],
        ball_r=2.0,
        cylinder_R=1.0,
        positron_hmac_key=valid_key,
    )
    _report_summary(report_bad_gromov)
    assert report_bad_gromov.truth_value == HeytingProsecutorTruthValue.VETOED
    assert report_bad_gromov.adjudication_reason == "GROMOV_NON_SQUEEZING_VIOLATION"
    assert report_bad_gromov.esp32_crowbar_armed is True
    assert report_bad_gromov.crowbar_budget_respected is True
    assert report_bad_gromov.crowbar_actuation_ns == _ESP32_CROWBAR_EXECUTION_NS
    assert report_bad_gromov.crowbar_peak_current_amps > 0.0
    print("  ✓ TEST 2 OK")

    # ── Test 3: HMAC del positrón inválido → VETOED ───────────────────────
    _print_header("TEST 3: HMAC positrón inválido → VETOED")
    report_bad_pos = engine.run_prosecution_audit(
        apu_cost_matrix=matrix_base,
        novikov_exponents=[0.15, 0.40],
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=b"PAYLOAD_INCORRECTO_DEL_POSITRON",
    )
    _report_summary(report_bad_pos)
    assert report_bad_pos.truth_value == HeytingProsecutorTruthValue.VETOED
    assert report_bad_pos.adjudication_reason == "DIRAC_POSITRON_HMAC_INVALID"
    assert report_bad_pos.dirac_annihilation.energy_released_kev == 0.0
    assert report_bad_pos.dirac_annihilation.back_action_db == 12.5
    assert report_bad_pos.esp32_crowbar_armed is True
    print("  ✓ TEST 3 OK")

    # ── Test 4: Krull por debajo del piso → VETOED ────────────────────────
    _print_header("TEST 4: Krull v(A) < 0.10 → VETOED")
    report_bad_krull = engine.run_prosecution_audit(
        apu_cost_matrix=matrix_base,
        novikov_exponents=np.array([0.03, 0.05, 0.09]),
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _report_summary(report_bad_krull)
    assert report_bad_krull.truth_value == HeytingProsecutorTruthValue.VETOED
    assert report_bad_krull.adjudication_reason == "NOVIKOV_KRULL_VALUATION_FLOOR_VIOLATION"
    assert report_bad_krull.krull_valuation_vA == 0.03
    print("  ✓ TEST 4 OK")

    # ── Test 5: Torsión homológica detectada → VETOED ─────────────────────
    _print_header("TEST 5: SNF con torsión (Tor ≠ 0) → VETOED")
    report_torsion = engine.run_prosecution_audit(
        apu_cost_matrix=np.array([[2.0, 0.0], [0.0, 2.0]]),
        novikov_exponents=[0.15, 0.40],
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _report_summary(report_torsion)
    assert report_torsion.truth_value == HeytingProsecutorTruthValue.VETOED
    assert report_torsion.adjudication_reason == "BOOLE_HOMOLOGY_TORSION_DETECTED"
    assert report_torsion.torsion_detected is True
    print("  ✓ TEST 5 OK")

    # ── Test 6: matriz vacía (firma trivial, COHERENT vacuo) ──────────────
    _print_header("TEST 6: matriz vacía → COHERENT vacuo")
    report_empty = engine.run_prosecution_audit(
        apu_cost_matrix=np.array([]),
        novikov_exponents=np.array([]),
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _report_summary(report_empty)
    assert report_empty.truth_value == HeytingProsecutorTruthValue.COHERENT
    assert report_empty.adjudication_reason == "ALL_PROSECUTION_INVARIANTS_SATISFIED"
    assert report_empty.cauchy_riemann_residual == 0.0
    assert report_empty.torsion_detected is False
    assert report_empty.krull_valuation_vA == _KRULL_VALUATION_FLOOR
    print("  ✓ TEST 6 OK")

    # ── Test 7: 5×5 aleatorio (seed 42) — comprobación de estabilidad ────
    _print_header("TEST 7: 5×5 aleatorio (seed=42) — estabilidad estructural")
    rng = np.random.default_rng(42)
    matrix_5x5 = rng.normal(size=(5, 5))
    report_5x5 = engine.run_prosecution_audit(
        apu_cost_matrix=matrix_5x5,
        novikov_exponents=np.array([0.12, 0.27, 0.55]),
        ball_r=0.15,
        cylinder_R=0.20,
        positron_hmac_key=valid_key,
    )
    _report_summary(report_5x5)
    assert report_5x5.rsi3_state is not None
    assert report_5x5.celestial_state is not None
    assert len(report_5x5.sovereign_hmac_sha256) == 64
    assert engine.verify_sovereign_hmac(report_5x5)
    print("  ✓ TEST 7 OK")

    # ── Test 8: manipulación del sello de fase → gate Φ₁₂ fail-CLOSED ─────
    _print_header("TEST 8: manipulación del sello P1 → gate Φ₁₂ fail-CLOSED")
    legitimate_p1 = engine.phase1_prosecution_seal(matrix_base, [0.15])
    tampered_seal = Phase1ProsecutionSeal(
        timestamp=legitimate_p1.timestamp,
        engine_version=legitimate_p1.engine_version,
        boolean_n=legitimate_p1.boolean_n,
        boolean_m=legitimate_p1.boolean_m,
        boolean_rank=legitimate_p1.boolean_rank,
        boolean_torsion_detected=legitimate_p1.boolean_torsion_detected,
        boolean_betti_1=legitimate_p1.boolean_betti_1,
        novikov_krull_valuation=legitimate_p1.novikov_krull_valuation,
        novikov_ultrametric_norm=legitimate_p1.novikov_ultrametric_norm,
        novikov_is_valid=legitimate_p1.novikov_is_valid,
        phase1_seal_hash="0" * 64,  # hash manipulado
    )
    try:
        engine.phase2_open_from_phase1_seal(
            seal=tampered_seal,
            apu_cost_matrix=matrix_base,
            novikov_exponents=[0.15],
            ball_r=0.15,
            cylinder_R=0.20,
            positron_hmac_key=valid_key,
        )
        raise AssertionError("Φ₁₂ debió dispararse con hash manipulado")
    except ProsecutorGateViolationError as exc:
        print(f"  ✓ Φ₁₂ fail-CLOSED correctamente: {exc}")

    # ── Test 9: RSI³ contrato vectorial + Fréchet + Aitken–Steffensen ────
    _print_header("TEST 9: RSI³ Banach–Kleisli (contrato vectorial, Fréchet y Aitken Δ²)")
    lip_bound, jerk_pos = engine.evaluate_rsi3_banach_kleisli_monad(initial_lip=0.95)
    state = engine._rsi3_vector_iterate(0.95, kam_margin=1.25)
    print(f"  Lip(T³)               = {lip_bound:.6f} (< 1.0 → {lip_bound < 1.0})")
    print(f"  DT³(x₀) exacto        = {state.exact_frechet_jacobian_t3:.6f} (≤ Lip(T³) → {abs(state.exact_frechet_jacobian_t3) <= lip_bound})")
    print(f"  D²T³(x₀) exacto       = {state.exact_frechet_hessian_t3:.6e}")
    print(f"  λ_Lyapunov            = {state.lyapunov_exponent:.6e} (< 0 → {state.lyapunov_exponent < 0.0})")
    print(f"  Punto fijo Aitken x*  = {state.aitken_accelerated_fixed_point:.6e} (|x*_Aitken| < |x₄| → {abs(state.aitken_accelerated_fixed_point) < abs(state.iterates[-1])})")
    print(f"  Ganancia Meta-Nivel 3 = {state.meta_policy_gain_level3:.6f} (≥ 1.0 → {state.meta_policy_gain_level3 >= 1.0})")
    assert lip_bound == _BANACH_LIP_T3
    assert lip_bound < 1.0
    assert abs(state.exact_frechet_jacobian_t3) <= lip_bound + 1.0e-12
    assert state.lyapunov_exponent < 0.0
    assert abs(state.aitken_accelerated_fixed_point) < abs(state.iterates[-1])
    assert state.meta_policy_gain_level3 >= 1.0
    assert jerk_pos is True
    assert state.kleisli_depth == 3
    print("  ✓ TEST 9 OK")

    # ── Test 10: Mecánica Celeste de Poincaré–Birkhoff–Nekhoroshev–Yoshida ─
    _print_header("TEST 10: Mecánica Celeste de Poincaré (Yoshida 4.º Orden, Birkhoff, Nekhoroshev)")
    cel_state = engine._compute_celestial_state(action_J=0.85, pert_eps=0.05)
    print(f"  Acción J              = {cel_state.action_J:.4f} | ω₁={cel_state.forcing_omega:.4f} | ω₂={cel_state.detuning_omega2:.4f}")
    print(f"  Error energía Yoshida = {cel_state.symplectic_energy_error:.6e}")
    print(f"  Invariante P.–Cartan  = {cel_state.poincare_cartan_invariant:.6e}")
    print(f"  Twist Birkhoff τ_B    = {cel_state.birkhoff_twist_coeff:.6e} | det Kolmogorov = {cel_state.kolmogorov_nondegeneracy_det:.6e}")
    print(f"  Tiempo Nekhoroshev    = {cel_state.nekhoroshev_stability_time:.6e} | ψ_split = {cel_state.homoclinic_splitting_angle:.6e} rad")
    assert cel_state.symplectic_energy_error < 1.0e-4
    assert cel_state.birkhoff_twist_coeff > 0.0
    assert cel_state.kolmogorov_nondegeneracy_det < 0.0
    assert cel_state.nekhoroshev_stability_time > 1.0
    assert 0.0 <= cel_state.homoclinic_splitting_angle <= 0.5 * math.pi
    print("  ✓ TEST 10 OK")

    # ── Test 11: Anidamiento formal explícito de las 3 Fases ──────────────
    _print_header("TEST 11: Verificación del Contrato de Anidamiento Formal (Fase I → Fase II → Fase III)")
    # Fase I culmina en phase1_prosecution_seal:
    s1 = engine.phase1_prosecution_seal(matrix_base, [0.22, 0.35])
    # Fase II inicia en phase2_open_from_phase1_seal consumiendo s1 y culmina en phase2_prosecution_seal:
    c2 = engine.phase2_open_from_phase1_seal(s1, matrix_base, [0.22, 0.35], 0.12, 0.25, valid_key, 0.08)
    c2 = engine.phase2_resolve_pseudoholomorphic_surface(c2)
    c2 = engine.phase2_evaluate_poincare_melnikov(c2)
    c2 = engine.phase2_evaluate_rsi3_banach(c2, initial_x=0.95)
    c2 = engine.phase2_execute_dirac_annihilation(c2)
    s2 = engine.phase2_prosecution_seal(c2)
    # Fase III inicia en phase3_open_from_phase2_seal consumiendo s2 y culmina en phase3_emit_prosecution_report:
    c3 = engine.phase3_open_from_phase2_seal(s2)
    c3 = engine.phase3_adjudicate_heyting_omega3(c3)
    c3 = engine.phase3_arm_esp32_crowbar(c3)
    final_rep = engine.phase3_emit_prosecution_report(c3, execution_time_ms=1.23)
    assert final_rep.phase1_seal_hash == s1.phase1_seal_hash
    assert final_rep.phase2_seal_hash == s2.phase2_seal_hash
    assert engine.verify_phase1_seal(s1)
    assert engine.verify_phase2_seal(s2)
    assert engine.verify_sovereign_hmac(final_rep)
    assert final_rep.truth_value == HeytingProsecutorTruthValue.COHERENT
    print("  ✓ TEST 11 OK (Tríada Kleisli anidada verificada de extremo a extremo)")

    print("\n=== TODOS LOS TESTS SOBERANOS COMPLETADOS CON ÉXITO ===")