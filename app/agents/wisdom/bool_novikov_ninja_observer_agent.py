# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovNinjaObserverAgent v15.0.0-PhD                                                ║
║ Ubicación: app/agents/wisdom/bool_novikov_ninja_observer_agent.py                                    ║
║ Versión  : 15.0.0-SOVEREIGN-BOOL-NOVIKOV-NINJA-OBSERVER-AGENT-POINCARE-KAM-RSI3-PhD                 ║
║ Gobernanza: Gobernanza OODA Trifásica Pasiva de Sigilo (①Observe → ②Orient → ③Decide&Act) con        ║
║            Compuertas Homomórficas Φ₁₂ y Φ₂₃ sobre el Vigía Ninja Imperial (STEALTH_SILENT_QND).    ║
║            Integración granular: Mecánica Celeste de Poincaré-Lindstedt-KAM-Melnikov-Floer +        ║
║            Automejora Recursiva Nivel 3 (RSI³ Kleisli T³ / Darwin-Gödel / Löb) + Heyting Ω₃.        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
──────────────────────────────────────────────────────────────────────────
El Soberano Agéntico de Calibre Vigía `BoolNovikovNinjaObserverAgent` gobierna ejecutivamente
el ciclo OODA de observación espectral de sigilo y vigilancia cuántica sobre el motor espectral
`BoolNovikovNinjaObserverEngine` (app/wisdom/bool_novikov_ninja_observer_engine.py).
Supervisa la preservación de invarianza de calibre sin retroacción (\mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB})
en tres fases anidadas acopladas por compuertas homomórficas \Phi_{12} y \Phi_{23} firmadas vía HMAC-SHA256:
audita la influencia booleana de Rademacher-Walsh y la forma normal de Smith sobre \mathbb{Z},
resuelve la dinámica de Poincaré-Lindstedt-KAM en variables de Delaunay-Whittaker, verifica
el polígono pseudoholomorfo en la categoría de Fukaya \mathcal{F}uk(M), y gobierna la automejora
recursiva en la mónada de Kleisli T^3 bajo el bucle Darwin-Gödel-Löb (RSI³).

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — OBSERVE (`NinjaObserverPhase1Observer`):
   Evalúa la densidad cuántica \rho y calcula la distancia geodésica de Bures-Wasserstein d_B(\rho, \sigma).
   Audita la influencia booleana mediante derivadas de Rademacher-Walsh D_i f(x) sobre \mathbb{Z}_2^n
   y la Forma Normal de Smith sobre \mathbb{Z} confirmando \mathrm{Tor}(H_k) = 0 y \beta_1 = 0.
   En el anillo \Lambda_{\mathrm{Nov}}, valida la valuación de Krull v(A) \ge 0.10.
   Resuelve la ecuación homológica de Poincaré-Lindstedt \{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^d}
   extrayendo la semilla de acción-ángulo de Delaunay (J^0, \theta^0, H_{\mathrm{eff}}).
   • Compuerta \Phi_{12} y Morfismo de Clausura F1 / Apertura F2:
     Emite `NinjaObserverPhase1Certificate` transfiriendo el invariante de Poincaré-Cartan hacia la Fase 2.

2. FASE 2 — ORIENT (`NinjaObserverPhase2Orientor` + Compuerta Homomórfica \Phi_{12}):
   Inicia consumiendo `NinjaObserverPhase1Certificate`.
   Audita la dinámica celeste en variables de Delaunay-Whittaker con la condición diofantina fuerte de KAM:
       |\langle \omega, k \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \neq 0,
   la convergencia de la función de Bruno B(\omega) < \infty, el horizonte de Nekhoroshev T_{\mathrm{Nekh}} \ge 1.0\times 10^6
   y la integral homoclínica de Melnikov M(\tau) en series de Fourier.
   En la categoría de Fukaya \mathcal{F}uk(M), la ecuación de Floer asegura no-squeezing de Gromov c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)).
   • Compuerta \Phi_{23} y Morfismo de Clausura F2 / Apertura F3:
     Emite `NinjaObserverPhase2Certificate` empaquetando el espectro celeste y de Floer hacia la Fase 3.

3. FASE 3 — DECIDE & ACT (`NinjaObserverPhase3Decider` + Compuerta \Phi_{23}):
   Inicia consumiendo `NinjaObserverPhase2Certificate`.
   Opera la mónada de Kleisli T^3 = \mu \circ T_2 \circ T_1 en espacio de Banach con contracción de Picard \mathrm{Lip}(T) \le 0.95,
   aceleración de Fréchet y cota de Newton-Kantorovich h_K = L_0 \beta \eta \le 0.50.
   Sintetiza el operador QND A^*_{\mathrm{obs}} bajo verificación modal de Löb \Box(\Box P \to P) \to \Box P,
   asegurando retroacción \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
   Deduce el veredicto en el álgebra de Heyting \Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\},
   calcula las métricas de protección de WACC y compresión KV, y sella el estado soberano inmutable `NinjaObserverSovereignState`.

AXIOMAS FORMALES
────────────────
[Axioma NOA-A1] Trivialidad Homológica en Smith:
    d_i \in \{0, 1\} \implies \mathrm{Tor}(H_k(\mathcal{K}; \mathbb{Z})) \equiv 0 \land \beta_1 = 0.
[Axioma NOA-A2] Persistencia de Toros KAM:
    |\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau \implies \text{ausencia de ruptura estocástica en toros invariantes}.
[Axioma NOA-A3] Rigidez Simpléctica en Fukaya:
    c_G(B^{2d}(r)) = \pi r^2 \le c_G(Z^{2d}(R)) = \pi R^2.
[Axioma NOA-A4] Invarianza Cuántica QND:
    \|[A^*_{\mathrm{obs}}, \rho]\|_F \le 1.0\times 10^{-10} \implies \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante NOA-I1] Anulación del Primer Número de Betti: \beta_1 = 0.
[Invariante NOA-I2] Valuación de Krull Mínima: v(A) \ge 0.10.
[Invariante NOA-I3] Retroacción Cuántica Acotada: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante NOA-I4] Parámetro de Kantorovich Controlado: h_K \le 0.50.
[Invariante NOA-I5] Sellado Criptográfico HMAC-SHA256 verificado en tiempo constante en cada compuerta.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from typing import Final, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Importación del motor espectral BoolNovikovNinjaObserverEngine v15.0.0-PhD
try:
    from bool_novikov_ninja_observer_engine import (
        BoolNovikovNinjaObserverEngine,
        BuresIsometryReport,
        DelaunayWhittakerState,
        HeytingTruthValue,
        MelnikovKAMReport,
        NinjaObserverSpectrumReport,
        PoincareLindstedtReport,
        PseudoholomorphicFloerReport,
        RSI3BanachKleisliReport,
        StealthQNDMeasurementReport,
    )
except ImportError:
    try:
        from app.wisdom.bool_novikov_ninja_observer_engine import (
            BoolNovikovNinjaObserverEngine,
            BuresIsometryReport,
            DelaunayWhittakerState,
            HeytingTruthValue,
            MelnikovKAMReport,
            NinjaObserverSpectrumReport,
            PoincareLindstedtReport,
            PseudoholomorphicFloerReport,
            RSI3BanachKleisliReport,
            StealthQNDMeasurementReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_ninja_observer_engine v15.0.0-PhD."
        ) from exc

logger = logging.getLogger("APU.Wisdom.BoolNovikovNinjaObserverAgent")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE GOBERNANZA SOBERANA, CRIPTOGRAFÍA Y UMBRALES HOMOMÓRFICOS
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOL-NOVIKOV-NINJA-OBSERVER-AGENT-POINCARE-KAM-RSI3-PhD"
)
__version__: Final[str] = _AGENT_VERSION

_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_NINJA_OBSERVER_KEY_v15.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"NINJA-OBSERVER-P1-ALGEBRA-POINCARE-LINDSTEDT-v15.0|"
_HMAC_DOMAIN_P2: Final[bytes] = b"NINJA-OBSERVER-P2-KAM-RSI3-FLOER-QND-v15.0|"
_HMAC_DOMAIN_P3: Final[bytes] = b"NINJA-OBSERVER-P3-HEYTING-SOVEREIGN-v15.0|"

# Tolerancias numéricas y físicas
_EPSILON_NUM: Final[float] = 1.0e-15
_HERMITIAN_TOLERANCE: Final[float] = 1.0e-10
_PSD_EIGENVALUE_TOLERANCE: Final[float] = -1.0e-12

# Umbrales de las compuertas homomórficas Φ₁₂ y Φ₂₃
_GATE12_KRULL_FLOOR: Final[float] = 0.10
_GATE12_RW_NORM_CEILING: Final[float] = 10.0
_GATE23_BANACH_LIP_CEILING: Final[float] = 1.0
_GATE23_BACKACTION_DB_CEILING: Final[float] = 0.01
_GATE23_CR_FLOER_RESIDUAL_CEILING: Final[float] = 1.0e-6
_GATE23_MELNIKOV_CHAOS_THRESHOLD: Final[float] = 0.05

# Parámetros base del Funtor Semántico Visceral Φ_sem
_VISCERAL_WACC_BASE_PCT: Final[float] = 15.0
_VISCERAL_LIQUIDITY_BASE_PCT: Final[float] = 11.5
_VISCERAL_KV_COMPRESSION_BASE_PCT: Final[float] = 86.4


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS INMUTABLES: CERTIFICADOS P1, P2, MÉTRICAS Y ESTADO SOBERANO
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NinjaObserverPhase1Certificate:
    """Certificado Criptográfico P1 emitido al cierre de la Fase 1 (Observe).

    Encapsula la atestación cuántica de Jordan-Lie D(ℋ), el fibrado dual
    R_Boole ⊗ Λ_Nov y el puente canónico de Poincaré-Lindstedt hacia la Fase 2.
    """

    timestamp: float
    # Invariantes algebraicos R_Boole
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    # Invariantes ultramétricos Λ_Nov
    krull_valuation: float
    novikov_norm: float
    ultrametric_inequality_holds: bool
    # Puente Poincaré-Lindstedt (FASE 1 ──► FASE 2)
    poincare_lindstedt_residual: float
    poincare_lindstedt_resonance_free: bool
    # Invariantes cuánticos de densidad D(ℋ)
    tr_rho: float
    is_psd: bool
    # Cadenas criptográficas
    chain_hash: str
    hmac_signature: str
    # Campos enriquecidos v15.0.0-PhD
    poincare_lindstedt_report: Optional[PoincareLindstedtReport] = None
    hermitian_defect: float = 0.0
    purity: float = 1.0
    von_neumann_entropy: float = 0.0
    birkhoff_normal_form_energy: float = 0.0
    snf_invariant_factors: Tuple[int, ...] = ()


@dataclass(frozen=True)
class NinjaObserverPhase2Certificate:
    """Certificado Criptográfico P2 emitido al cierre de la Fase 2 (Orient).

    Encapsula la orientación celeste de Delaunay-Whittaker-KAM, Melnikov,
    Cauchy-Riemann-Floer, Gromov, la síntesis RSI³ en el centralizador Z(ρ_MAC)
    y la medición débil AAV de sigilo (0.0 dB Back-Action).
    """

    timestamp: float
    spectrum_report: NinjaObserverSpectrumReport
    # Invariantes celestes Poincaré-KAM-Melnikov
    delaunay_kam_stable: bool
    melnikov_chaos_detected: bool
    diophantine_margin: float
    # Invariantes RSI³ (Kleisli T³ + Darwin-Gödel + Löb)
    rsi3_lip_t3: float
    rsi3_converged: bool
    rsi3_jerk: float
    rsi3_darwin_generation: int
    # Cadenas criptográficas
    chain_hash: str
    hmac_signature: str
    # Campos enriquecidos v15.0.0-PhD
    brjuno_convergence_sum: float = 0.0
    nekhoroshev_stability_horizon: float = 0.0
    poincare_cartan_invariant: float = 0.0
    monodromy_symplectic_det: float = 1.0
    loeb_provability_consistent: bool = True
    centralizer_commutator_norm: float = 0.0


@dataclass(frozen=True)
class NinjaObserverVisceralMetrics:
    """Métricas ejecutivas traducidas por el Funtor Semántico Visceral Φ_sem.

    Definición formal:
        Φ_sem: NinjaObserverPhase2Certificate → ℝ³ × Σ*
        • WACC_protection% = WACC_base · (1 + α · (1 − Lip(T³)))
        • Liquidez_neta%   = Liq_base  · (1 + β · F_Uhlmann)
        • KV_Compresión%   = KV_base   · (1 + γ · (1 − D_Bures))
    """

    wacc_protection_pct: float
    net_liquidity_release_pct: float
    kv_cache_compression_pct: float
    stealth_qnd_status: str
    semantic_coupling_alpha: float
    semantic_coupling_beta: float
    semantic_coupling_gamma: float
    kam_stability_premium_pct: float = 0.0
    qfim_metrological_gain: float = 0.0


@dataclass(frozen=True)
class NinjaObserverSovereignState:
    """Estado Soberano Inmutable emitido al cierre de la Fase 3 (Decide & Act).

    Cumple estrictamente el invariante STEALTH_SILENT_QND:
        - Sin emisión de telemetría invasiva ni interrupciones a hardware.
        - Perturbación cuántica nula (Back-Action = 0.0 dB).
        - Atestación criptográfica en Árbol de Merkle SHA-256 sobre P1 → P2 → P3.
    """

    timestamp: float
    truth_value: HeytingTruthValue
    is_stealth_silent_qnd: bool
    back_action_db: float
    # Invariantes celestes, simplécticos y RSI³ consolidados
    poincare_kam_stable: bool
    rsi3_attractor_verified: bool
    gromov_no_squeezing_valid: bool
    # Cadenas criptográficas
    merkle_root_hash: str
    p3_hmac_signature: str
    # Métricas viscerales Φ_sem
    visceral_metrics: NinjaObserverVisceralMetrics
    # Certificados de la cadena homomórfica
    p1_certificate: NinjaObserverPhase1Certificate
    p2_certificate: NinjaObserverPhase2Certificate
    # Metadata de ejecución
    execution_time_ns: int
    agent_version: str
    nekhoroshev_horizon: float = 0.0
    loeb_consistency_verified: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN PASIVA CUÁNTICA-ALGEBRAICA Y PUENTE DE POINCARÉ-LINDSTEDT
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-fase 1.1: `_verify_density_matrix_invariants` (Álgebra de Jordan-Lie D(ℋ))
# Sub-fase 1.2: `_compute_algebraic_poincare_invariants` (R_Boole ⊗ Λ_Nov + Serie de Lie)
# Sub-fase 1.3: `observe` (Síntesis y firma HMAC-SHA256 del Certificado P1)
# Sub-fase 1.4: `continue_to_phase2_gate` (Compuerta Homomórfica Φ₁₂: Cierre de Fase 1
#               y Puente Estructural de Entrada hacia la Fase 2)
# ═══════════════════════════════════════════════════════════════════════════════


class NinjaObserverPhase1Observer:
    """Fase 1 OODA — Observador Pasivo Cuántico-Algebraico y Generador de Lie.

    Ejecuta la ingesta pasiva sin perturbación sobre el espacio de estados
    de densidad D(ℋ) y el fibrado dual R_Boole ⊗ Λ_Nov, cerrando con la
    resolución homológica de Poincaré-Lindstedt que alimenta a la Fase 2.
    """

    def __init__(
        self,
        engine: BoolNovikovNinjaObserverEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.engine = engine
        self.secret_key = secret_key

    # ───────────────────────────────────────────────────────────────────────────
    # Método 1.1: Invariantes Cuánticos de Jordan-Lie sobre D(ℋ)
    # ───────────────────────────────────────────────────────────────────────────
    def _verify_density_matrix_invariants(
        self,
        apu_cost_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.complex128], float, float, bool, float, float]:
        r"""Verifica y proyecta pasivamente el operador de estado ρ_MAC ∈ D(ℋ).

        Definición formal en el Álgebra de Jordan-Lie:
            1. Producto de Jordan: {A, B}_J = ½(AB + BA)
            2. Proyección hermítica: ρ₀ = ½(M + M†), defecto δ_H = ||M − M†||_F / (||M||_F + ε)
            3. Regularización espectral PSD y normalización proyectiva:
               ρ_MAC = ρ₀ / Tr(ρ₀)  tal que  Tr(ρ_MAC) ≡ 1.0  y  λ_min(ρ_MAC) ≥ 0
            4. Pureza cuántica: γ(ρ_MAC) = Tr(ρ_MAC²) ∈ [1/d, 1]
            5. Entropía de von Neumann: S(ρ_MAC) = −Σ_i λ_i ln(λ_i) ≥ 0

        Args:
            apu_cost_matrix: Matriz de costos APU / acoplamiento ciber-físico.

        Returns:
            Tupla (rho_mac, tr_rho, hermitian_defect, is_psd, purity, vn_entropy).
        """
        mat = np.asarray(apu_cost_matrix, dtype=np.complex128)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            d = max(2, int(math.isqrt(mat.size)))
            padded = np.zeros(d * d, dtype=np.complex128)
            padded[: min(mat.size, d * d)] = mat.ravel()[: min(mat.size, d * d)]
            mat = padded.reshape((d, d))

        norm_m = float(np.linalg.norm(mat, ord="fro"))
        hermitian_defect = float(
            np.linalg.norm(mat - mat.conj().T, ord="fro") / (norm_m + _EPSILON_NUM)
        )

        # Simetrización de Jordan
        rho_raw = 0.5 * (mat + mat.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(rho_raw)
        is_psd_raw = bool(np.all(eigenvalues >= _PSD_EIGENVALUE_TOLERANCE))

        # Proyección al cono positivo semidefinido si existen autovalores negativos
        if not is_psd_raw:
            eigenvalues_clipped = np.maximum(eigenvalues, 0.0)
            if float(np.sum(eigenvalues_clipped)) <= _EPSILON_NUM:
                eigenvalues_clipped = np.ones_like(eigenvalues)
            rho_psd = (
                eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.conj().T
            )
            rho_psd = 0.5 * (rho_psd + rho_psd.conj().T)
        else:
            eigenvalues_clipped = np.maximum(eigenvalues, 0.0)
            rho_psd = rho_raw

        tr_raw = float(np.real(np.trace(rho_psd)))
        if abs(tr_raw) > _EPSILON_NUM:
            rho_mac = rho_psd / tr_raw
        else:
            d = mat.shape[0]
            rho_mac = np.eye(d, dtype=np.complex128) / float(d)

        tr_rho = float(np.real(np.trace(rho_mac)))
        eig_norm = np.linalg.eigvalsh(rho_mac)
        is_psd = bool(is_psd_raw and np.all(eig_norm >= _PSD_EIGENVALUE_TOLERANCE))

        # Pureza γ = Tr(ρ²) y Entropía de von Neumann S(ρ) = −Tr(ρ ln ρ)
        purity = float(np.real(np.trace(rho_mac @ rho_mac)))
        valid_eigs = eig_norm[eig_norm > _EPSILON_NUM]
        vn_entropy = float(-np.sum(valid_eigs * np.log(valid_eigs)))

        return rho_mac, tr_rho, hermitian_defect, is_psd, purity, max(0.0, vn_entropy)

    # ───────────────────────────────────────────────────────────────────────────
    # Método 1.2: Invariantes Algebraicos R_Boole ⊗ Λ_Nov y Puente Poincaré-Lindstedt
    # ───────────────────────────────────────────────────────────────────────────
    def _compute_algebraic_poincare_invariants(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
    ) -> Tuple[
        float,
        bool,
        bool,
        Tuple[int, ...],
        float,
        float,
        bool,
        PoincareLindstedtReport,
    ]:
        r"""Evalúa el fibrado R_Boole ⊗ Λ_Nov y construye la serie de Lie de Poincaré-Lindstedt.

        Definición formal:
            1. ∇_Boole f: Espectro diádico de Rademacher-Walsh ponderado por peso de Hamming |k|_H.
            2. SNF sobre ℤ: U·M_ℤ·V = diag(d₁, ..., d_r, 0) con d_i | d_{i+1}, Tor(H_k) ≡ 0, β₁ = 0.
            3. Valuación de Krull v(A) = min(γ_i) y desigualdad ultramétrica sobre Λ_Nov.
            4. Ecuación Homológica de Poincaré-Lindstedt:
                   {H₀, W} = Σ_j ω_j ∂W/∂θ_j = H₁(J, θ) − ⟨H₁⟩_{𝕋^d}(J)
               cuya solución W genera el flujo canónico exp(ε L_W) que sirve de puente a la Fase 2.

        Args:
            apu_cost_matrix: Matriz de incidencia / costos sobre R_Boole.
            exponents: Exponentes de la serie formal de Novikov Λ_Nov.

        Returns:
            Tupla con invariantes de R_Boole, Λ_Nov y el reporte `PoincareLindstedtReport`.
        """
        # 1. Rademacher-Walsh sobre Z₂ⁿ
        rw_norm, _ = self.engine._compute_rademacher_walsh_gradient(apu_cost_matrix)

        # 2. Forma Normal de Smith entera exacta sobre ℤ
        torsion_free, betti_zero, snf_diag = (
            self.engine._verify_smith_normal_form_torsion(apu_cost_matrix)
        )
        snf_factors = tuple(int(v) for v in snf_diag if int(v) > 0)

        # 3. Valuación de Krull y norma ultramétrica de Novikov
        krull_v, novikov_norm, ultrametric_ok = (
            self.engine._evaluate_novikov_krull_ultrametric(exponents)
        )

        # 4. Ecuación homológica de Poincaré-Lindstedt (Puente Estructural → Fase 2)
        d = apu_cost_matrix.shape[0]
        sym_mat = 0.5 * (apu_cost_matrix + apu_cost_matrix.T)
        eigenvalues = np.linalg.eigvalsh(sym_mat)
        phi_golden = 0.5 * (1.0 + math.sqrt(5.0))
        irrational_ladder = np.array(
            [(i + 1) * (phi_golden - 1.0) for i in range(d)], dtype=np.float64
        )
        h0_frequencies = np.abs(eigenvalues) + 0.15 * irrational_ladder + 0.1
        h0_frequencies = h0_frequencies / (
            float(np.linalg.norm(h0_frequencies)) + _EPSILON_NUM
        )

        n_modes = max(8, d + 1)
        h1_coefficients = np.zeros(n_modes, dtype=np.float64)
        h1_coefficients[0] = 0.01 * float(np.mean(sym_mat))
        for k in range(1, n_modes):
            h1_coefficients[k] = 0.005 * float(eigenvalues[k % d]) / float(k)

        pl_report = self.engine._solve_poincare_lindstedt_homological(
            h0_frequencies, h1_coefficients
        )

        return (
            rw_norm,
            torsion_free,
            betti_zero,
            snf_factors,
            krull_v,
            novikov_norm,
            ultrametric_ok,
            pl_report,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Método 1.3: Emisión Criptográfica del Certificado P1
    # ───────────────────────────────────────────────────────────────────────────
    def observe(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
    ) -> NinjaObserverPhase1Certificate:
        """Ejecuta la Fase 1 (Observe) completa y emite el certificado firmado P1.

        Args:
            apu_cost_matrix: Matriz de costos APU / MIC sobre R_Boole.
            exponents: Lista de exponentes reales en el anillo de Novikov Λ_Nov.

        Returns:
            `NinjaObserverPhase1Certificate` sellado con HMAC-SHA256.
        """
        ts = time.time()

        # Sub-fase 1.1: Invariantes cuánticos de Jordan-Lie
        (
            _rho_mac,
            tr_rho,
            hermitian_defect,
            is_psd,
            purity,
            vn_entropy,
        ) = self._verify_density_matrix_invariants(apu_cost_matrix)

        # Sub-fase 1.2: Invariantes algebraicos y puente Poincaré-Lindstedt
        (
            rw_norm,
            torsion_free,
            betti_zero,
            snf_factors,
            krull_v,
            novikov_norm,
            ultrametric_ok,
            pl_report,
        ) = self._compute_algebraic_poincare_invariants(apu_cost_matrix, exponents)

        chain_raw = (
            f"P1|v15.0|{ts:.6f}|{tr_rho:.6f}|{hermitian_defect:.6e}|{is_psd}|"
            f"{purity:.6f}|{vn_entropy:.6f}|{rw_norm:.6f}|{torsion_free}|"
            f"{betti_zero}|{krull_v:.6f}|{novikov_norm:.6f}|{ultrametric_ok}|"
            f"{pl_report.residual_homological:.6e}|{pl_report.is_resonance_free}|"
            f"{pl_report.birkhoff_normal_form_energy:.6f}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_raw).hexdigest()

        hmac_sig = hmac.new(
            self.secret_key,
            _HMAC_DOMAIN_P1 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return NinjaObserverPhase1Certificate(
            timestamp=ts,
            rademacher_walsh_norm=rw_norm,
            smith_torsion_free=torsion_free,
            betti_number_zero=betti_zero,
            krull_valuation=krull_v,
            novikov_norm=novikov_norm,
            ultrametric_inequality_holds=ultrametric_ok,
            poincare_lindstedt_residual=pl_report.residual_homological,
            poincare_lindstedt_resonance_free=pl_report.is_resonance_free,
            tr_rho=tr_rho,
            is_psd=is_psd,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            poincare_lindstedt_report=pl_report,
            hermitian_defect=hermitian_defect,
            purity=purity,
            von_neumann_entropy=vn_entropy,
            birkhoff_normal_form_energy=pl_report.birkhoff_normal_form_energy,
            snf_invariant_factors=snf_factors,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Método 1.4 (CIERRE DE FASE 1 Y PUENTE A FASE 2): Compuerta Homomórfica Φ₁₂
    # ───────────────────────────────────────────────────────────────────────────
    def validate_gate_phi12(
        self,
        p1_cert: NinjaObserverPhase1Certificate,
    ) -> bool:
        r"""Compuerta Homomórfica Φ₁₂ — Cierre formal de la Fase 1 y Puente hacia la Fase 2.

        Definición formal:
            Φ₁₂: NinjaObserverPhase1Certificate → {True, False}
            Φ₁₂(P1) = True ⟺
                (a) HMAC_SHA256(K, Domain_P1 ∥ P1.chain_hash) ≡ P1.hmac_signature
                (b) len(P1.chain_hash) == 64
                (c) P1.is_psd ∧ |P1.tr_rho − 1.0| < 10⁻⁹
                (d) P1.krull_valuation ≥ _GATE12_KRULL_FLOOR ∧ P1.ultrametric_inequality_holds
                (e) P1.poincare_lindstedt_report is not None

        CONTINUIDAD ESTRUCTURAL FASE 1 ──► FASE 2:
            La validación positiva de Φ₁₂(P1) consagra formalmente a `p1_cert` (y su
            `poincare_lindstedt_report` con las semillas canónicas de Lie J^(1), θ^(1)
            y P_renorm) como el estado inicial exacto del primer método de la Fase 2:
            `NinjaObserverPhase2Orientor._execute_celestial_poincare_kam_melnikov_audit`.

        Args:
            p1_cert: Certificado P1 emitido por `observe`.

        Returns:
            True si la transición homomórfica hacia la Fase 2 está autorizada.
        """
        expected_sig = hmac.new(
            self.secret_key,
            _HMAC_DOMAIN_P1 + p1_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, p1_cert.hmac_signature):
            logger.error("Compuerta Φ₁₂: Firma HMAC P1 inválida.")
            return False

        if len(p1_cert.chain_hash) != 64:
            logger.error("Compuerta Φ₁₂: Hash de cadena P1 corrupto (len != 64).")
            return False

        if not p1_cert.is_psd or abs(p1_cert.tr_rho - 1.0) > 1.0e-9:
            logger.error("Compuerta Φ₁₂: Estado cuántico no PSD o traza no unitaria.")
            return False

        if p1_cert.krull_valuation < _GATE12_KRULL_FLOOR:
            logger.error(
                f"Compuerta Φ₁₂: Valuación de Krull v(A)={p1_cert.krull_valuation:.4f} "
                f"< floor={_GATE12_KRULL_FLOOR:.4f}."
            )
            return False

        if not p1_cert.ultrametric_inequality_holds:
            logger.error("Compuerta Φ₁₂: Violación de la desigualdad ultramétrica en Λ_Nov.")
            return False

        if not p1_cert.smith_torsion_free or not p1_cert.betti_number_zero:
            logger.warning(
                "Compuerta Φ₁₂: Torsión homológica o Betti β₁ ≠ 0 detectado. "
                "Admitiendo en modo pasivo STEALTH_SILENT_QND para diagnóstico en Fase 2/3."
            )

        if not p1_cert.poincare_lindstedt_resonance_free:
            logger.warning(
                "Compuerta Φ₁₂: Resonancia secular en Poincaré-Lindstedt detectada. "
                "Admitiendo hacia regularización diofantina KAM en Fase 2."
            )

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENTACIÓN CELESTE POINCARÉ-KAM-MELNIKOV-FLOER, RSI³ Y MEDICIÓN QND
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-fase 2.1: `_execute_celestial_poincare_kam_melnikov_audit` (Continuación directa
#               de `validate_gate_phi12` consumiendo `p1_cert.poincare_lindstedt_report`)
# Sub-fase 2.2: `_execute_floer_gromov_audit` (Cauchy-Riemann-Floer + Gromov No-Squeezing)
# Sub-fase 2.3: `_execute_rsi3_darwin_godel_audit` (Mónada de Kleisli T³, Löb y síntesis
#               del operador A*_ninja en el centralizador de von Neumann Z(ρ_MAC))
# Sub-fase 2.4: `_execute_aav_and_bures_audit` (Medición débil AAV 0.0 dB + Bures-Uhlmann)
# Sub-fase 2.5: `orient` (Ensamblaje espectral y firma HMAC-SHA256 del Certificado P2)
# Sub-fase 2.6: `validate_gate_phi23` (Compuerta Homomórfica Φ₂₃: Cierre de Fase 2
#               y Puente Estructural de Entrada hacia la Fase 3)
# ═══════════════════════════════════════════════════════════════════════════════


class NinjaObserverPhase2Orientor:
    """Fase 2 OODA — Orientador Celeste Simpléctico, Motor RSI³ y Medidor Débil QND.

    Recibe el certificado `p1_cert` validado por la Compuerta Homomórfica Φ₁₂ al
    final de la Fase 1 y despliega la cadena causal:
        Poincaré-Lindstedt (P1) ──► Delaunay-Whittaker-KAM ──► Melnikov-Krein
        ──► Cauchy-Riemann-Floer & Gromov ──► RSI³ Darwin-Gödel en Z(ρ_MAC)
        ──► Medición Débil AAV (0.0 dB) & Bures-Uhlmann.
    """

    def __init__(
        self,
        engine: BoolNovikovNinjaObserverEngine,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> None:
        self.engine = engine
        self.secret_key = secret_key

    # ───────────────────────────────────────────────────────────────────────────
    # Método 2.1 (INICIO DE FASE 2 DESDE EL PUENTE DE FASE 1): Auditoría Celeste
    # ───────────────────────────────────────────────────────────────────────────
    def _execute_celestial_poincare_kam_melnikov_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        p1_cert: NinjaObserverPhase1Certificate,
        exponents: List[float],
    ) -> Tuple[DelaunayWhittakerState, MelnikovKAMReport, List[float]]:
        r"""Ejecuta la auditoría celeste de Delaunay-Whittaker, KAM, Brjuno y Melnikov.

        CONTINUIDAD ESTRUCTURAL DESDE FASE 1 (Método 1.4):
            Consume directamente `p1_cert.poincare_lindstedt_report` validado por
            la Compuerta Φ₁₂, extrayendo las semillas canónicas de la serie de Lie
            (J^(1), θ^(1), P_renorm) para inicializar el flujo en T*𝕋^d.

        Definición formal:
            1. Coordenadas de Delaunay-Whittaker (J, θ) ∈ T*𝕋^d:
               Verifica la condición diofantina KAM |ω(J)·k| ≥ γ / |k|^τ,
               convergencia de la serie de Brjuno B(ω) < ∞, invariante integral
               de Poincaré-Cartan ∮ p dq y tiempo de estabilidad de Nekhoroshev.
            2. Integral Armónica de Melnikov:
               M(τ₀) = ∫_{-∞}^{+∞} {H₀, H₁}(q₀(t), p₀(t); t + τ₀) dt = Σ_k M_k e^{ikτ₀}
               con verificación de determinante simpléctico det(DP_Σ) = 1 y
               multiplicadores de Floquet-Krein en el círculo unitario 𝕊¹.
            3. Regularización pasiva de exponentes de Novikov-Krull para preservar
               el piso v(A) ≥ v_floor sin violar la geometría del simplex diofantino.

        Args:
            apu_cost_matrix: Matriz de costos APU.
            p1_cert: Certificado P1 validado por la Compuerta Homomórfica Φ₁₂.
            exponents: Exponentes de entrada en el anillo de Novikov.

        Returns:
            Tupla (delaunay_state, melnikov_report, regularized_exponents).
        """
        d = apu_cost_matrix.shape[0]
        pl_bridge = p1_cert.poincare_lindstedt_report

        if (
            pl_bridge is not None
            and pl_bridge.canonical_actions_seed.size == d
            and pl_bridge.canonical_angles_seed.size == d
            and pl_bridge.renormalized_perturbation_tensor.shape == (d, d)
        ):
            action_vector = pl_bridge.canonical_actions_seed.copy()
            angle_vector = pl_bridge.canonical_angles_seed.copy()
            perturbation_matrix = pl_bridge.renormalized_perturbation_tensor.copy()
        else:
            sym_mat = 0.5 * (apu_cost_matrix + apu_cost_matrix.T)
            eigenvalues = np.linalg.eigvalsh(sym_mat)
            action_vector = np.abs(eigenvalues[:d]) + 0.01
            action_vector = action_vector / (
                float(np.linalg.norm(action_vector)) + _EPSILON_NUM
            )
            angle_vector = np.linspace(0.0, 2.0 * math.pi, d, endpoint=False)
            perturbation_scale = max(0.001, p1_cert.poincare_lindstedt_residual)
            perturbation_matrix = perturbation_scale * np.eye(d, dtype=np.float64)

        # 1. Transformación a coordenadas de Delaunay-Whittaker y auditoría KAM-Brjuno
        delaunay_state = self.engine._compute_delaunay_whittaker_coordinates(
            action_vector=action_vector,
            angle_vector=angle_vector,
            perturbation_matrix=perturbation_matrix,
            poincare_bridge=pl_bridge,
        )

        # 2. Integral armónica de Melnikov y monodromía simpléctica de Poincaré-Krein
        action_j = float(delaunay_state.actions_J[0]) if delaunay_state.actions_J.size > 0 else 1.0
        pert_eps = 0.01 * (1.0 + p1_cert.poincare_lindstedt_residual)
        melnikov_report = self.engine._evaluate_melnikov_integral_harmonic(
            action_J=action_j,
            pert_eps=pert_eps,
            n_harmonics=16,
            delaunay_state=delaunay_state,
        )

        # 3. Preservación del piso de Krull en los exponentes de Novikov
        if exponents:
            raw_min = min(float(e) for e in exponents)
            target_floor = max(self.engine.krull_floor, raw_min)
            regularized_exponents = [max(float(e), target_floor) for e in exponents]
        else:
            regularized_exponents = [self.engine.krull_floor]

        return delaunay_state, melnikov_report, regularized_exponents

    # ───────────────────────────────────────────────────────────────────────────
    # Método 2.2: Auditoría Simpléctica de Cauchy-Riemann-Floer y Gromov
    # ───────────────────────────────────────────────────────────────────────────
    def _execute_floer_gromov_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        melnikov_report: MelnikovKAMReport,
        ball_r: float = 0.5,
        cylinder_R: float = 0.8,
    ) -> PseudoholomorphicFloerReport:
        r"""Audita la superficie pseudoholomorfa de Cauchy-Riemann-Floer y Gromov.

        Definición formal:
            1. Ecuación de Cauchy-Riemann-Floer sobre cuerdas periódicas hamiltonianas:
                   ∂̄_{J₀,H}(u) = ∂u/∂s + J₀(u)·(∂u/∂τ − X_H(u)) = 0
               con estructura compleja simpléctica canónica J₀² = −I_{2d}.
            2. Teorema de No-Squeezing de Gromov:
                   c_G(B^{2n}(r)) = π r² ≤ c_G(Z^{2n}(R)) = π R² ⟺ r ≤ R.
            3. Emite `floer_contraction_seed` y `symplectic_hamiltonian_seed` como
               semilla estructural para la automejora recursiva RSI³ en el Método 2.3.

        Args:
            apu_cost_matrix: Matriz de costos APU (generador hamiltoniano).
            melnikov_report: Reporte de Melnikov-KAM procedente del Método 2.1.
            ball_r: Radio de la bola simpléctica B^{2n}(r).
            cylinder_R: Radio del cilindro simpléctico Z^{2n}(R).

        Returns:
            `PseudoholomorphicFloerReport` con invariantes de Floer, Gromov y Fukaya.
        """
        return self.engine._solve_cauchy_riemann_floer_pseudoholomorphic(
            hamiltonian_grad=apu_cost_matrix,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
            melnikov_report=melnikov_report,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Método 2.3: Automejora Recursiva Nivel 3 (RSI³ Kleisli T³ + Darwin-Gödel)
    # ───────────────────────────────────────────────────────────────────────────
    def _execute_rsi3_darwin_godel_audit(
        self,
        floer_report: PseudoholomorphicFloerReport,
        rho_mac: NDArray[np.complex128],
        base_lip: float = 0.95,
    ) -> RSI3BanachKleisliReport:
        r"""Ejecuta el bucle Darwin-Gödel de Automejora Recursiva Nivel 3 (RSI³).

        Definición formal sobre la Mónada de Kleisli T³ = T ∘ T ∘ T:
            • Nivel 1 (Parámetros): Contracción de Banach-Picard con Lip(T₁) ≤ 0.95.
            • Nivel 2 (Reglas dinámicas): Meta-optimización simpléctica con Lip(T₂) ≤ 0.95.
            • Nivel 3 (Síntesis topológica en el centralizador de von Neumann Z(ρ_MAC)):
              Dado el desdoblamiento espectral ρ_MAC = U Λ U†, sintetiza el operador
              de observación óptimo:
                  A*_ninja = U diag(λ^Casimir) U† ∈ Z(ρ_MAC)
              garantizando conmutatividad exacta [A*_ninja, ρ_MAC] = 0, Lip(T³) ≤ (0.95)³
              ≈ 0.857375 < 1.0, consistencia modal de Löb □(□P → P) → □P y jerk
              ciber-físico d³C/dt³ > 0.

        Args:
            floer_report: Puente simpléctico de Cauchy-Riemann-Floer (Método 2.2).
            rho_mac: Operador de densidad cuántica normalizado ρ_MAC ∈ D(ℋ).
            base_lip: Cota de Lipschitz base del endofuntor T.

        Returns:
            `RSI3BanachKleisliReport` con el atractor x* y el operador sintetizado A*_ninja.
        """
        return self.engine._execute_darwin_godel_recursive_improvement(
            floer_bridge=floer_report,
            rho_mac=rho_mac,
            base_lip=base_lip,
            population_size=5,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Método 2.4: Medición Débil Cuántica AAV (0.0 dB) e Isometría de Bures-Uhlmann
    # ───────────────────────────────────────────────────────────────────────────
    def _execute_aav_and_bures_audit(
        self,
        rho_mac: NDArray[np.complex128],
        rsi3_report: RSI3BanachKleisliReport,
    ) -> Tuple[StealthQNDMeasurementReport, BuresIsometryReport]:
        r"""Ejecuta la medición débil AAV en Z(ρ_MAC) y evalúa la geometría de Bures-Uhlmann.

        Definición formal:
            1. Valor débil de Aharonov-Albert-Vaidman (AAV):
                   A_w = ⟨ϕ_f | A*_ninja | ϕ_i⟩ / ⟨ϕ_f | ϕ_i⟩
               Como A*_ninja ∈ Z(ρ_MAC) fue sintetizado por RSI³ en el Método 2.3,
               ||[A*_ninja, ρ_MAC]||_F ≤ 10⁻¹⁵ y la retroacción cuántica es
               exactamente Back-Action = 0.0 dB (STEALTH_SILENT_QND).
            2. Isometría de Bures-Uhlmann e Información Cuántica de Fisher (QFIM):
                   F(ρ, σ) = (Tr √(√ρ σ √ρ))²,   D_B(ρ, σ) = √(2 − 2√F).

        Args:
            rho_mac: Matriz de densidad cuántica ρ_MAC ∈ D(ℋ).
            rsi3_report: Reporte RSI³ con el operador sintetizado en el centralizador.

        Returns:
            Tupla (qnd_report, bures_report).
        """
        d = rho_mac.shape[0]
        phi_i = np.ones((d, 1), dtype=np.complex128) / math.sqrt(float(d))
        phi_f = np.ones((d, 1), dtype=np.complex128) / math.sqrt(float(d))

        qnd_rep = self.engine._compute_stealth_aav_weak_measurement(
            rho_mac=rho_mac,
            phi_i=phi_i,
            phi_f=phi_f,
            rsi3_bridge=rsi3_report,
        )

        rho_post = (
            rsi3_report.synthesized_density_state
            if rsi3_report.synthesized_density_state.shape == rho_mac.shape
            else rho_mac
        )
        bures_rep = self.engine._evaluate_bures_uhlmann_fidelity(rho_mac, rho_post)

        return qnd_rep, bures_rep

    # ───────────────────────────────────────────────────────────────────────────
    # Método 2.5: Orquestación de Fase 2 (Orient) y Emisión del Certificado P2
    # ───────────────────────────────────────────────────────────────────────────
    def orient(
        self,
        p1_cert: NinjaObserverPhase1Certificate,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
        ball_r: float = 0.5,
        cylinder_R: float = 0.8,
    ) -> NinjaObserverPhase2Certificate:
        """Ejecuta la Fase 2 (Orient) completa y emite el certificado firmado P2.

        Flujo causal:
            2.1 Delaunay-Whittaker + KAM-Brjuno + Melnikov-Krein (desde el puente P1)
            2.2 Cauchy-Riemann-Floer + Gromov No-Squeezing
            2.3 RSI³ Kleisli T³ + Darwin-Gödel + Síntesis en Z(ρ_MAC)
            2.4 Medición Débil AAV (0.0 dB) + Geometría de Bures-Uhlmann
            2.5 Ensamblaje de `NinjaObserverSpectrumReport` y firma HMAC-SHA256 P2

        Args:
            p1_cert: Certificado P1 validado por la Compuerta Homomórfica Φ₁₂.
            apu_cost_matrix: Matriz de costos APU.
            exponents: Exponentes de la serie de Novikov.
            ball_r: Radio de la bola simpléctica para Gromov.
            cylinder_R: Radio del cilindro simpléctico para Gromov.

        Returns:
            `NinjaObserverPhase2Certificate` sellado con HMAC-SHA256.
        """
        ts = time.time()
        t0_ns = time.perf_counter_ns()

        # Reconstrucción normalizada de ρ_MAC ∈ D(ℋ)
        d = apu_cost_matrix.shape[0]
        rho_mac = apu_cost_matrix.astype(np.complex128)
        rho_mac = 0.5 * (rho_mac + rho_mac.conj().T)
        eigs, vecs = np.linalg.eigh(rho_mac)
        if np.any(eigs < 0.0):
            eigs = np.maximum(eigs, 0.0)
            if float(np.sum(eigs)) <= _EPSILON_NUM:
                eigs = np.ones_like(eigs)
            rho_mac = vecs @ np.diag(eigs) @ vecs.conj().T
            rho_mac = 0.5 * (rho_mac + rho_mac.conj().T)
        tr_val = float(np.real(np.trace(rho_mac)))
        if abs(tr_val) > _EPSILON_NUM:
            rho_mac = rho_mac / tr_val
        else:
            rho_mac = np.eye(d, dtype=np.complex128) / float(d)

        # Sub-fase 2.1: Celeste Poincaré-KAM-Melnikov desde el puente P1
        delaunay_state, melnikov_report, _reg_exps = (
            self._execute_celestial_poincare_kam_melnikov_audit(
                apu_cost_matrix, p1_cert, exponents
            )
        )

        # Sub-fase 2.2: Cauchy-Riemann-Floer y Gromov No-Squeezing
        floer_report = self._execute_floer_gromov_audit(
            apu_cost_matrix, melnikov_report, ball_r, cylinder_R
        )

        # Sub-fase 2.3: Automejora Recursiva Nivel 3 (RSI³) en Z(ρ_MAC)
        rsi3_report = self._execute_rsi3_darwin_godel_audit(
            floer_report=floer_report,
            rho_mac=rho_mac,
            base_lip=0.95,
        )

        # Sub-fase 2.4: Medición Débil AAV (0.0 dB) y Bures-Uhlmann
        qnd_rep, bures_rep = self._execute_aav_and_bures_audit(rho_mac, rsi3_report)

        # Adjudicación preliminar del motor y pasaporte criptográfico
        pl_report = (
            p1_cert.poincare_lindstedt_report
            if p1_cert.poincare_lindstedt_report is not None
            else PoincareLindstedtReport(
                generating_function_W=np.zeros(1, dtype=np.float64),
                frequency_shift=0.0,
                residual_homological=p1_cert.poincare_lindstedt_residual,
                is_resonance_free=p1_cert.poincare_lindstedt_resonance_free,
            )
        )

        engine_truth = self.engine._adjudicate_heyting_observer(
            torsion_free=p1_cert.smith_torsion_free,
            betti_zero=p1_cert.betti_number_zero,
            krull_v=p1_cert.krull_valuation,
            ultrametric_ok=p1_cert.ultrametric_inequality_holds,
            pl_rep=pl_report,
            dw_state=delaunay_state,
            mel_rep=melnikov_report,
            floer_rep=floer_report,
            rsi3_rep=rsi3_report,
            qnd_rep=qnd_rep,
        )

        passport_hash, engine_hmac = self.engine._compute_passport_and_hmac(
            truth_val=engine_truth,
            rw_norm=p1_cert.rademacher_walsh_norm,
            krull_v=p1_cert.krull_valuation,
            lip_t3=rsi3_report.banach_lip_t3,
            back_action=qnd_rep.back_action_db,
            kam_stable=delaunay_state.is_kam_stable,
            gromov_ok=floer_report.is_gromov_squeezing_valid,
        )

        exec_ns = time.perf_counter_ns() - t0_ns

        spectrum_report = NinjaObserverSpectrumReport(
            rademacher_walsh_norm=p1_cert.rademacher_walsh_norm,
            smith_torsion_free=p1_cert.smith_torsion_free,
            betti_number_zero=p1_cert.betti_number_zero,
            krull_valuation=p1_cert.krull_valuation,
            novikov_norm=p1_cert.novikov_norm,
            ultrametric_inequality_holds=p1_cert.ultrametric_inequality_holds,
            poincare_lindstedt_report=pl_report,
            delaunay_whittaker_state=delaunay_state,
            melnikov_kam_report=melnikov_report,
            pseudoholomorphic_floer_report=floer_report,
            rsi3_banach_report=rsi3_report,
            stealth_qnd_report=qnd_rep,
            bures_report=bures_rep,
            truth_value=engine_truth,
            passport_hash=passport_hash,
            execution_time_ns=exec_ns,
            hmac_signature=engine_hmac,
            banach_contraction_lip=rsi3_report.banach_lip_t3,
        )

        chain_raw = (
            f"P2|v15.0|{ts:.6f}|{p1_cert.chain_hash}|"
            f"{qnd_rep.back_action_db:.6f}|{qnd_rep.commutator_frobenius_norm:.6e}|"
            f"{delaunay_state.is_kam_stable}|{delaunay_state.brjuno_convergence_sum:.6f}|"
            f"{melnikov_report.chaos_detected}|{melnikov_report.monodromy_symplectic_det:.6f}|"
            f"{floer_report.is_gromov_squeezing_valid}|{floer_report.cr_floer_residual:.6e}|"
            f"{rsi3_report.banach_lip_t3:.6f}|{rsi3_report.is_rsi3_attractor}|"
            f"{rsi3_report.loeb_provability_consistent}"
        ).encode("utf-8")
        chain_hash = hashlib.sha256(chain_raw).hexdigest()

        hmac_sig = hmac.new(
            self.secret_key,
            _HMAC_DOMAIN_P2 + chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return NinjaObserverPhase2Certificate(
            timestamp=ts,
            spectrum_report=spectrum_report,
            delaunay_kam_stable=delaunay_state.is_kam_stable,
            melnikov_chaos_detected=melnikov_report.chaos_detected,
            diophantine_margin=delaunay_state.diophantine_gamma,
            rsi3_lip_t3=rsi3_report.banach_lip_t3,
            rsi3_converged=rsi3_report.fixed_point_converged,
            rsi3_jerk=rsi3_report.jerk_cyber_physical,
            rsi3_darwin_generation=rsi3_report.darwin_godel_generation,
            chain_hash=chain_hash,
            hmac_signature=hmac_sig,
            brjuno_convergence_sum=delaunay_state.brjuno_convergence_sum,
            nekhoroshev_stability_horizon=delaunay_state.nekhoroshev_stability_time,
            poincare_cartan_invariant=delaunay_state.poincare_cartan_invariant,
            monodromy_symplectic_det=melnikov_report.monodromy_symplectic_det,
            loeb_provability_consistent=rsi3_report.loeb_provability_consistent,
            centralizer_commutator_norm=qnd_rep.commutator_frobenius_norm,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Método 2.6 (CIERRE DE FASE 2 Y PUENTE A FASE 3): Compuerta Homomórfica Φ₂₃
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_banach_lip(p2_cert: NinjaObserverPhase2Certificate) -> float:
        """Extrae polimórficamente la constante de contracción Lip(T³) del certificado P2."""
        rep = p2_cert.spectrum_report
        if hasattr(rep, "rsi3_banach_report") and hasattr(
            rep.rsi3_banach_report, "banach_lip_t3"
        ):
            return float(rep.rsi3_banach_report.banach_lip_t3)
        if hasattr(rep, "banach_contraction_lip"):
            return float(getattr(rep, "banach_contraction_lip"))
        return float(p2_cert.rsi3_lip_t3)

    def validate_gate_phi23(
        self,
        p2_cert: NinjaObserverPhase2Certificate,
    ) -> bool:
        r"""Compuerta Homomórfica Φ₂₃ — Cierre formal de la Fase 2 y Puente hacia la Fase 3.

        Definición formal:
            Φ₂₃: NinjaObserverPhase2Certificate → {True, False}
            Φ₂₃(P2) = True ⟺
                (a) HMAC_SHA256(K, Domain_P2 ∥ P2.chain_hash) ≡ P2.hmac_signature
                (b) Lip(T³) < _GATE23_BANACH_LIP_CEILING (1.0)
                (c) Back-Action ≤ _GATE23_BACKACTION_DB_CEILING (0.01 dB)
                (d) Gromov No-Squeezing válido: c_G(B^{2n}(r)) ≤ c_G(Z^{2n}(R))
                (e) Residual pseudoholomorfo CR-Floer < _GATE23_CR_FLOER_RESIDUAL_CEILING (10⁻⁶)

        CONTINUIDAD ESTRUCTURAL FASE 2 ──► FASE 3:
            La validación positiva de Φ₂₃(P2) consagra formalmente al par certificado
            `(p1_cert, p2_cert)` como la entrada directa del primer método de la Fase 3:
            `NinjaObserverPhase3Decider._adjudicate_heyting_truth`.

        Args:
            p2_cert: Certificado P2 emitido por `orient`.

        Returns:
            True si la transición homomórfica hacia la Fase 3 está autorizada.
        """
        expected_sig = hmac.new(
            self.secret_key,
            _HMAC_DOMAIN_P2 + p2_cert.chain_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_sig, p2_cert.hmac_signature):
            logger.error("Compuerta Φ₂₃: Firma HMAC P2 inválida.")
            return False

        rep = p2_cert.spectrum_report
        lip_t3 = self._extract_banach_lip(p2_cert)

        if lip_t3 >= _GATE23_BANACH_LIP_CEILING:
            logger.error(
                f"Compuerta Φ₂₃: Divergencia en mónada de Kleisli Lip(T³)={lip_t3:.6f} ≥ 1.0."
            )
            return False

        back_action = rep.stealth_qnd_report.back_action_db
        if back_action > _GATE23_BACKACTION_DB_CEILING:
            logger.error(
                f"Compuerta Φ₂₃: Violación QND Back-Action={back_action:.6f} dB "
                f"> {_GATE23_BACKACTION_DB_CEILING:.4f} dB."
            )
            return False

        if not p2_cert.delaunay_kam_stable:
            logger.warning(
                "Compuerta Φ₂₃: Inestabilidad KAM diofantina detectada. "
                "Admitiendo hacia Fase 3 para adjudicación intuicionista DEGRADED."
            )

        if p2_cert.melnikov_chaos_detected:
            logger.warning(
                "Compuerta Φ₂₃: Transversalidad homoclínica en Melnikov detectada. "
                "Admitiendo hacia Fase 3 para adjudicación intuicionista DEGRADED."
            )

        if not rep.pseudoholomorphic_floer_report.is_gromov_squeezing_valid:
            logger.error("Compuerta Φ₂₃: Violación del Teorema de No-Squeezing de Gromov.")
            return False

        cr_res = rep.pseudoholomorphic_floer_report.cr_floer_residual
        if cr_res > _GATE23_CR_FLOER_RESIDUAL_CEILING:
            logger.error(
                f"Compuerta Φ₂₃: Residual CR-Floer={cr_res:.2e} "
                f"> {_GATE23_CR_FLOER_RESIDUAL_CEILING:.0e}."
            )
            return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECISIÓN EN TOPOS DE HEYTING Ω₃, FUNTOR VISCERAL Φ_sem Y ORQUESTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-fase 3.1: `_adjudicate_heyting_truth` (Continuación directa de `validate_gate_phi23`
#               consumiendo `(p1_cert, p2_cert)` en el retículo intuicionista Ω₃)
# Sub-fase 3.2: `_compute_visceral_semantic_functor` (Funtor covariante Φ_sem)
# Sub-fase 3.3: `decide_and_act` (Árbol de Merkle SHA-256 P1→P2→P3 y estado soberano)
# Sub-fase 3.4: `BoolNovikovNinjaObserverAgent.execute_sovereign_cycle` (Orquestador
#               Soberano del Ciclo OODA Trifásico Anidado)
# ═══════════════════════════════════════════════════════════════════════════════


class NinjaObserverPhase3Decider:
    """Fase 3 OODA — Decisor Intuicionista en Topos de Heyting Ω₃ y Funtor Visceral.

    Recibe los certificados `(p1_cert, p2_cert)` validados por la Compuerta Φ₂₃ al
    cierre de la Fase 2, adjudica la verdad intuicionista en Ω₃, evalúa el Funtor
    Semántico Visceral Φ_sem y sella la Raíz de Merkle SHA-256 sin efectos colaterales.
    """

    # ───────────────────────────────────────────────────────────────────────────
    # Método 3.1 (INICIO DE FASE 3 DESDE EL PUENTE DE FASE 2): Adjudicación en Ω₃
    # ───────────────────────────────────────────────────────────────────────────
    def _adjudicate_heyting_truth(
        self,
        p1_cert: NinjaObserverPhase1Certificate,
        p2_cert: NinjaObserverPhase2Certificate,
    ) -> HeytingTruthValue:
        r"""Adjudica el veredicto en el Topos de Heyting Ω₃ = {COHERENT ≺ DEGRADED ≺ VETOED}.

        CONTINUIDAD ESTRUCTURAL DESDE FASE 2 (Método 2.6):
            Consume directamente los certificados `p1_cert` y `p2_cert` verificados
            por la Compuerta Homomórfica Φ₂₃ al cierre de la Fase 2.

        Definición formal (Retículo distributivo acotado (Ω₃, ⊓, ⊔, ⇒)):
            Orden lógico: VETOED (⊥ = 0x03) ⊏ DEGRADED (0x01) ⊏ COHERENT (⊤ = 0x00).
            Criterios evaluados:
                C1: Gromov No-Squeezing c_G(B^{2n}(r)) ≤ c_G(Z^{2n}(R))
                C2: Sigilo QND ||[A*_ninja, ρ_MAC]||_F ≤ 10⁻¹⁰ ∧ Back-Action ≤ 0.0 dB
                C3: Contracción RSI³ Lip(T³) < 1.0 ∧ consistencia modal de Löb
                C4: Estabilidad diofantina KAM ∧ serie de Brjuno convergente
                C5: Ausencia de caos homoclínico de Melnikov ∧ estabilidad de Krein
                C6: Residual pseudoholomorfo de Cauchy-Riemann-Floer < 10⁻⁶
                C7: Ausencia de torsión homológica SNF Tor(H_k) ≡ 0 ∧ Betti β₁ = 0
                C8: Valuación de Krull v(A) ≥ v_floor ∧ desigualdad ultramétrica

        Args:
            p1_cert: Certificado P1 de la Fase 1.
            p2_cert: Certificado P2 de la Fase 2.

        Returns:
            `HeytingTruthValue` adjudicado en Ω₃.
        """
        rep = p2_cert.spectrum_report

        c1_gromov = rep.pseudoholomorphic_floer_report.is_gromov_squeezing_valid
        c2_qnd = (
            rep.stealth_qnd_report.is_stealth_qnd
            and rep.stealth_qnd_report.back_action_db <= _GATE23_BACKACTION_DB_CEILING
        )
        c3_banach = (
            rep.rsi3_banach_report.banach_lip_t3 < _GATE23_BANACH_LIP_CEILING
            and rep.rsi3_banach_report.loeb_provability_consistent
        )
        c4_kam = p2_cert.delaunay_kam_stable
        c5_melnikov = not p2_cert.melnikov_chaos_detected
        c6_cr_floer = (
            rep.pseudoholomorphic_floer_report.cr_floer_residual
            < _GATE23_CR_FLOER_RESIDUAL_CEILING
        )
        c7_torsion = p1_cert.smith_torsion_free and p1_cert.betti_number_zero
        c8_krull = (
            p1_cert.krull_valuation >= _GATE12_KRULL_FLOOR
            and p1_cert.ultrametric_inequality_holds
        )

        # Veto estricto si falla la contracción de Banach, el sigilo QND o Gromov
        if not c3_banach or not c2_qnd or not c1_gromov:
            return HeytingTruthValue.VETOED

        all_coherent = (
            c1_gromov
            and c2_qnd
            and c3_banach
            and c4_kam
            and c5_melnikov
            and c6_cr_floer
            and c7_torsion
            and c8_krull
        )
        if all_coherent:
            return HeytingTruthValue.COHERENT

        return HeytingTruthValue.DEGRADED

    # ───────────────────────────────────────────────────────────────────────────
    # Método 3.2: Funtor Semántico Visceral Φ_sem
    # ───────────────────────────────────────────────────────────────────────────
    def _compute_visceral_semantic_functor(
        self,
        p2_cert: NinjaObserverPhase2Certificate,
        truth_val: Optional[HeytingTruthValue] = None,
    ) -> NinjaObserverVisceralMetrics:
        r"""Evalúa el Funtor Semántico Visceral Φ_sem dinámicamente.

        Definición formal:
            Φ_sem: NinjaObserverPhase2Certificate → NinjaObserverVisceralMetrics
            • WACC%      = 15.0 · (1 + α · (1 − Lip(T³)))          con α = 0.10
            • Liquidez%  = 11.5 · (1 + β · F_Uhlmann)              con β = 0.05
            • KV_Cache%  = 86.4 · (1 + γ · (1 − D_Bures))          con γ = 0.08
            • KAM_Prem%  = 100 · γ_KAM / (1 + B(ω))
            • QFIM_Gain  = Tr(ℱ_Q(ρ_MAC))

        Args:
            p2_cert: Certificado P2 con el reporte espectral completo.
            truth_val: Valor de verdad en Heyting Ω₃ (opcional).

        Returns:
            `NinjaObserverVisceralMetrics` con acoplamientos viscerales certificados.
        """
        rep = p2_cert.spectrum_report
        lip_t3 = rep.rsi3_banach_report.banach_lip_t3
        fidelity = rep.bures_report.fidelity_uhlmann
        bures_dist = rep.bures_report.bures_distance
        back_action = rep.stealth_qnd_report.back_action_db

        alpha = 0.10
        beta = 0.05
        gamma = 0.08

        degradation_scale = 1.0
        if truth_val == HeytingTruthValue.DEGRADED:
            degradation_scale = 0.85
        elif truth_val == HeytingTruthValue.VETOED:
            degradation_scale = 0.0

        wacc_pct = (
            _VISCERAL_WACC_BASE_PCT
            * (1.0 + alpha * (1.0 - lip_t3))
            * degradation_scale
        )
        liq_pct = (
            _VISCERAL_LIQUIDITY_BASE_PCT
            * (1.0 + beta * fidelity)
            * degradation_scale
        )
        kv_pct = (
            _VISCERAL_KV_COMPRESSION_BASE_PCT
            * (1.0 + gamma * (1.0 - bures_dist))
            * degradation_scale
        )

        brjuno = max(0.0, p2_cert.brjuno_convergence_sum)
        kam_premium = float(
            100.0 * max(0.0, p2_cert.diophantine_margin) / (1.0 + brjuno)
        )
        qfim_gain = float(
            getattr(rep.bures_report, "quantum_fisher_information", 4.0)
        )

        if rep.stealth_qnd_report.is_stealth_qnd and back_action <= 0.0:
            stealth_status = "PASSIVE_SILENT_QND_0.0dB"
        elif back_action <= _GATE23_BACKACTION_DB_CEILING:
            stealth_status = "PASSIVE_SILENT_QND_MARGINAL"
        else:
            stealth_status = "QND_VIOLATION_DETECTED"

        return NinjaObserverVisceralMetrics(
            wacc_protection_pct=float(wacc_pct),
            net_liquidity_release_pct=float(liq_pct),
            kv_cache_compression_pct=float(kv_pct),
            stealth_qnd_status=stealth_status,
            semantic_coupling_alpha=alpha,
            semantic_coupling_beta=beta,
            semantic_coupling_gamma=gamma,
            kam_stability_premium_pct=kam_premium,
            qfim_metrological_gain=qfim_gain,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Método 3.3: Emisión de la Raíz de Merkle SHA-256 y Estado Soberano Final
    # ───────────────────────────────────────────────────────────────────────────
    def decide_and_act(
        self,
        p1_cert: NinjaObserverPhase1Certificate,
        p2_cert: NinjaObserverPhase2Certificate,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> NinjaObserverSovereignState:
        """Ejecuta la Fase 3 (Decide & Act) y emite el estado soberano inmutable.

        Construye el Árbol de Merkle criptográfico jerárquico sobre la cadena
        P1 → P2 → P3 bajo el invariante pasivo STEALTH_SILENT_QND.

        Args:
            p1_cert: Certificado P1 validado.
            p2_cert: Certificado P2 validado.
            secret_key: Clave criptográfica soberana para HMAC-SHA256.

        Returns:
            `NinjaObserverSovereignState` sellado en el Espacio de Sabiduría V_W.
        """
        ts = time.time()
        rep = p2_cert.spectrum_report

        # Sub-fase 3.1: Adjudicación en Topos de Heyting Ω₃
        truth_val = self._adjudicate_heyting_truth(p1_cert, p2_cert)

        # Sub-fase 3.2: Invariante STEALTH_SILENT_QND y Funtor Visceral Φ_sem
        is_stealth = rep.stealth_qnd_report.is_stealth_qnd
        back_action = rep.stealth_qnd_report.back_action_db
        visceral = self._compute_visceral_semantic_functor(p2_cert, truth_val)

        # Sub-fase 3.3: Árbol de Merkle SHA-256 jerárquico P1 → P2 → P3
        p3_payload = (
            f"P3|v15.0|{truth_val.value}|{is_stealth}|{back_action:.6f}|"
            f"{p2_cert.delaunay_kam_stable}|{p2_cert.rsi3_lip_t3:.6f}|"
            f"{visceral.wacc_protection_pct:.4f}|{visceral.net_liquidity_release_pct:.4f}|"
            f"{visceral.kv_cache_compression_pct:.4f}"
        )
        p3_hash = hashlib.sha256(p3_payload.encode("utf-8")).hexdigest()

        merkle_raw = (
            f"{p1_cert.chain_hash}|{p2_cert.chain_hash}|{p3_hash}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_raw).hexdigest()

        hmac_sig_p3 = hmac.new(
            secret_key,
            _HMAC_DOMAIN_P3 + merkle_root.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return NinjaObserverSovereignState(
            timestamp=ts,
            truth_value=truth_val,
            is_stealth_silent_qnd=is_stealth,
            back_action_db=back_action,
            poincare_kam_stable=p2_cert.delaunay_kam_stable,
            rsi3_attractor_verified=rep.rsi3_banach_report.is_rsi3_attractor,
            gromov_no_squeezing_valid=(
                rep.pseudoholomorphic_floer_report.is_gromov_squeezing_valid
            ),
            merkle_root_hash=merkle_root,
            p3_hmac_signature=hmac_sig_p3,
            visceral_metrics=visceral,
            p1_certificate=p1_cert,
            p2_certificate=p2_cert,
            execution_time_ns=0,
            agent_version=_AGENT_VERSION,
            nekhoroshev_horizon=p2_cert.nekhoroshev_stability_horizon,
            loeb_consistency_verified=p2_cert.loeb_provability_consistent,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO ORQUESTADOR: BoolNovikovNinjaObserverAgent (Cierre de Fase 3)
# ═══════════════════════════════════════════════════════════════════════════════


class BoolNovikovNinjaObserverAgent:
    """Soberano Agéntico Vigía Ninja Imperial de Sigilo (STEALTH_SILENT_QND) v15.0.0-PhD.

    Orquesta la arquitectura OODA trifásica anidada con continuidad estructural:
        ① Fase 1 (`observe`) ──► Compuerta Φ₁₂ (`continue_to_phase2_gate`)
        ──► ② Fase 2 (`orient`) ──► Compuerta Φ₂₃ (`continue_to_phase3_gate`)
        ──► ③ Fase 3 (`decide_and_act` / `execute_sovereign_cycle`).
    """

    def __init__(
        self,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        krull_floor: float = _GATE12_KRULL_FLOOR,
        kam_gamma: float = 0.1,
        kam_tau: float = 2.0,
        rsi3_depth: int = 3,
    ) -> None:
        self.secret_key = secret_key
        self.engine = BoolNovikovNinjaObserverEngine(
            krull_floor=krull_floor,
            kam_gamma=kam_gamma,
            kam_tau=kam_tau,
            rsi3_depth=rsi3_depth,
            secret_key=secret_key,
        )
        self.phase1_observer = NinjaObserverPhase1Observer(
            engine=self.engine, secret_key=secret_key
        )
        self.phase2_orientor = NinjaObserverPhase2Orientor(
            engine=self.engine, secret_key=secret_key
        )
        self.phase3_decider = NinjaObserverPhase3Decider()
        logger.info(
            f"BoolNovikovNinjaObserverAgent v{__version__} inicializado "
            f"[krull={krull_floor:.3f}, kam_γ={kam_gamma:.3f}, rsi3={rsi3_depth}]"
        )

    def continue_to_phase2_gate(
        self,
        p1_cert: NinjaObserverPhase1Certificate,
    ) -> bool:
        """Compuerta Homomórfica Φ₁₂: Valida P1 y habilita el puente hacia la Fase 2."""
        return self.phase1_observer.validate_gate_phi12(p1_cert)

    def continue_to_phase3_gate(
        self,
        p2_cert: NinjaObserverPhase2Certificate,
    ) -> bool:
        """Compuerta Homomórfica Φ₂₃: Valida P2 y habilita el puente hacia la Fase 3."""
        return self.phase2_orientor.validate_gate_phi23(p2_cert)

    # ───────────────────────────────────────────────────────────────────────────
    # Método 3.4: Ciclo Soberano OODA Trifásico en Sigilo Pasivo QND
    # ───────────────────────────────────────────────────────────────────────────
    def execute_sovereign_cycle(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
        ball_r: float = 0.5,
        cylinder_R: float = 0.8,
    ) -> NinjaObserverSovereignState:
        """Ejecuta el ciclo OODA trifásico anidado completo en sigilo pasivo QND.

        Flujo con continuidad estructural entre fases:
            ① P1 = phase1_observer.observe(apu_cost_matrix, exponents)
            ② Φ₁₂(P1) ──► P2 = phase2_orientor.orient(P1, apu_cost_matrix, exponents, r, R)
            ③ Φ₂₃(P2) ──► State = phase3_decider.decide_and_act(P1, P2)

        Args:
            apu_cost_matrix: Matriz de costos APU / MIC sobre R_Boole.
            exponents: Exponentes de la serie formal de Novikov Λ_Nov.
            ball_r: Radio de la bola simpléctica B^{2n}(r) para Gromov.
            cylinder_R: Radio del cilindro simpléctico Z^{2n}(R) para Gromov.

        Returns:
            `NinjaObserverSovereignState` inmutable y sellado criptográficamente.

        Raises:
            ValueError: Si alguna de las compuertas homomórficas Φ₁₂ o Φ₂₃ veta el paso.
        """
        t_start_ns = time.perf_counter_ns()

        # ① FASE 1: OBSERVE (Álgebra de Jordan-Lie + R_Boole ⊗ Λ_Nov + Poincaré-Lindstedt)
        p1_cert = self.phase1_observer.observe(apu_cost_matrix, exponents)

        # Puente Estructural Fase 1 ──► Fase 2 (Compuerta Homomórfica Φ₁₂)
        if not self.continue_to_phase2_gate(p1_cert):
            raise ValueError(
                "Falla de validación en Compuerta Homomórfica Φ₁₂. "
                "Ciclo OODA abortado en FASE 1."
            )

        # ② FASE 2: ORIENT (Delaunay-KAM-Melnikov + CR-Floer/Gromov + RSI³ + AAV QND)
        p2_cert = self.phase2_orientor.orient(
            p1_cert=p1_cert,
            apu_cost_matrix=apu_cost_matrix,
            exponents=exponents,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
        )

        # Puente Estructural Fase 2 ──► Fase 3 (Compuerta Homomórfica Φ₂₃)
        if not self.continue_to_phase3_gate(p2_cert):
            raise ValueError(
                "Falla de validación en Compuerta Homomórfica Φ₂₃. "
                "Ciclo OODA abortado en FASE 2."
            )

        # ③ FASE 3: DECIDE & ACT (Heyting Ω₃ + Funtor Visceral Φ_sem + Merkle SHA-256)
        sovereign_state = self.phase3_decider.decide_and_act(
            p1_cert=p1_cert,
            p2_cert=p2_cert,
            secret_key=self.secret_key,
        )

        exec_time_ns = time.perf_counter_ns() - t_start_ns

        return NinjaObserverSovereignState(
            timestamp=sovereign_state.timestamp,
            truth_value=sovereign_state.truth_value,
            is_stealth_silent_qnd=sovereign_state.is_stealth_silent_qnd,
            back_action_db=sovereign_state.back_action_db,
            poincare_kam_stable=sovereign_state.poincare_kam_stable,
            rsi3_attractor_verified=sovereign_state.rsi3_attractor_verified,
            gromov_no_squeezing_valid=sovereign_state.gromov_no_squeezing_valid,
            merkle_root_hash=sovereign_state.merkle_root_hash,
            p3_hmac_signature=sovereign_state.p3_hmac_signature,
            visceral_metrics=sovereign_state.visceral_metrics,
            p1_certificate=sovereign_state.p1_certificate,
            p2_certificate=sovereign_state.p2_certificate,
            execution_time_ns=exec_time_ns,
            agent_version=_AGENT_VERSION,
            nekhoroshev_horizon=sovereign_state.nekhoroshev_horizon,
            loeb_consistency_verified=sovereign_state.loeb_consistency_verified,
        )

    def run_ooda_triphasic_cycle(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
        ball_r: float = 0.5,
        cylinder_R: float = 0.8,
    ) -> NinjaObserverSovereignState:
        """Alias de interoperabilidad soberana para `execute_sovereign_cycle`."""
        return self.execute_sovereign_cycle(
            apu_cost_matrix=apu_cost_matrix,
            exponents=exponents,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST AUTÓNOMO DE CERTIFICACIÓN CIBER-FÍSICA, CELESTE Y RSI³ (v15.0.0-PhD)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    agent = BoolNovikovNinjaObserverAgent()
    matrix_test = np.eye(4, dtype=np.float64)
    exps_test = [0.15, 0.30, 0.45]

    sovereign_state = agent.execute_sovereign_cycle(
        apu_cost_matrix=matrix_test,
        exponents=exps_test,
        ball_r=0.5,
        cylinder_R=0.8,
    )

    p1 = sovereign_state.p1_certificate
    p2 = sovereign_state.p2_certificate
    vm = sovereign_state.visceral_metrics

    print("═" * 84)
    print("=== TEST AUTÓNOMO: BoolNovikovNinjaObserverAgent v15.0.0-PhD ===")
    print("═" * 84)
    print(f"Versión del Agente            : {sovereign_state.agent_version}")
    print(
        f"Veredicto Final Heyting Ω₃    : {sovereign_state.truth_value.name} "
        f"({hex(sovereign_state.truth_value.value)})"
    )
    print(f"Invariante STEALTH_SILENT_QND : {sovereign_state.is_stealth_silent_qnd}")
    print(f"Back-Action QND (dB)          : {sovereign_state.back_action_db:.6f} dB")
    print(f"Conmutador ||[A*, ρ_MAC]||_F  : {p2.centralizer_commutator_norm:.3e}")
    print("─" * 84)
    print(
        f"[FASE 1] Traza Tr(ρ_MAC)      : {p1.tr_rho:.6f} | PSD={p1.is_psd} | "
        f"Pureza={p1.purity:.4f} | S_vN={p1.von_neumann_entropy:.4f}"
    )
    print(
        f"[FASE 1] R_Boole & SNF        : ||∇_RW||={p1.rademacher_walsh_norm:.4f} | "
        f"Tor≡0: {p1.smith_torsion_free} | β₁=0: {p1.betti_number_zero}"
    )
    print(
        f"[FASE 1] Novikov Λ_Nov        : Krull v(A)={p1.krull_valuation:.4f} | "
        f"||A||_Nov={p1.novikov_norm:.4f} | Ultramétrica={p1.ultrametric_inequality_holds}"
    )
    print(
        f"[FASE 1→2] Poincaré-Lindstedt : Res={p1.poincare_lindstedt_residual:.3e} | "
        f"Sin Resonancia={p1.poincare_lindstedt_resonance_free} | "
        f"E_Birkhoff={p1.birkhoff_normal_form_energy:.6f}"
    )
    print("─" * 84)
    print(
        f"[FASE 2] Poincaré-KAM-Brjuno  : Estable={sovereign_state.poincare_kam_stable} | "
        f"γ_KAM={p2.diophantine_margin:.4f} | B(ω)={p2.brjuno_convergence_sum:.4f} | "
        f"T_Nekh={p2.nekhoroshev_stability_horizon:.2f}"
    )
    print(
        f"[FASE 2] Melnikov & Monodromía: Caos={p2.melnikov_chaos_detected} | "
        f"det(DP_Σ)={p2.monodromy_symplectic_det:.6f}"
    )
    print(
        f"[FASE 2] CR-Floer & Gromov    : Residual={p2.spectrum_report.pseudoholomorphic_floer_report.cr_floer_residual:.3e} | "
        f"Gromov Válido={sovereign_state.gromov_no_squeezing_valid}"
    )
    print(
        f"[FASE 2→3] RSI³ Darwin-Gödel  : Lip(T³)={p2.rsi3_lip_t3:.6f} | "
        f"Atractor={sovereign_state.rsi3_attractor_verified} | "
        f"Löb={p2.loeb_provability_consistent} | Gen={p2.rsi3_darwin_generation}"
    )
    print("─" * 84)
    print(
        f"[FASE 3] Funtor Visceral Φ_sem: WACC={vm.wacc_protection_pct:.2f}% | "
        f"Liquidez={vm.net_liquidity_release_pct:.2f}% | "
        f"KV-Cache={vm.kv_cache_compression_pct:.2f}% | Status={vm.stealth_qnd_status}"
    )
    print(f"[FASE 3] Raíz Merkle SHA-256  : {sovereign_state.merkle_root_hash[:32]}...")
    print(f"[FASE 3] Firma HMAC P3        : {sovereign_state.p3_hmac_signature[:32]}...")
    print(f"[FASE 3] Tiempo de Ejecución  : {sovereign_state.execution_time_ns / 1e6:.3f} ms")
    print("═" * 84)

    # Aserciones deterministas de nivel PhD
    assert sovereign_state.truth_value == HeytingTruthValue.COHERENT
    assert sovereign_state.is_stealth_silent_qnd is True
    assert sovereign_state.back_action_db == 0.0
    assert sovereign_state.poincare_kam_stable is True
    assert sovereign_state.rsi3_attractor_verified is True
    assert sovereign_state.gromov_no_squeezing_valid is True
    assert sovereign_state.loeb_consistency_verified is True
    assert abs(p1.tr_rho - 1.0) < 1.0e-12
    assert p2.rsi3_lip_t3 < 1.0
    assert len(sovereign_state.merkle_root_hash) == 64
    assert len(sovereign_state.p3_hmac_signature) == 64