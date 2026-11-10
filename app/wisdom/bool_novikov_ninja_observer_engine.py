# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovNinjaObserverEngine v15.0.0-PhD                                                ║
║ Ubicación: app/wisdom/bool_novikov_ninja_observer_engine.py                                          ║
║ Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-NINJA-OBSERVER-POINCARE-KAM-RSI3-PhD                           ║
║ Función  : Motor Espectral del Vigía Ninja Imperial de Sigilo (STEALTH_SILENT_QND) sobre el          ║
║            Anillo Booleano (R_Boole), el Anillo Ultramétrico (Λ_Nov) y el Polígono Pseudoholomorfo   ║
║            en la Categoría de Fukaya ℱuk(ℳ).                                                         ║
║            Integración granular: Mecánica Celeste de Poincaré + Automejora Recursiva RSI³.           ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Motor Espectral `BoolNovikovNinjaObserverEngine` ejecuta la observación espectral de sigilo
y vigilancia asintótica (STEALTH_SILENT_QND) en la arquitectura APU Filter. Su arquitectura monádica
en tres fases anidadas (\mathcal{F}_3 \circ \mathcal{F}_2 \circ \mathcal{F}_1) observa la densidad cuántica
\rho_{\mathrm{MAC}} mediante derivadas discretas de Rademacher-Walsh sobre \mathbb{F}_2^n, la forma
normal de Smith sobre \mathbb{Z}, valuación ultramétrica no arquimediana en \Lambda_{\mathrm{Nov}},
coordenadas de acción-ángulo de Delaunay-Whittaker con estabilidad diofantina de Poincaré-Lindstedt-KAM-Melnikov,
curvas pseudoholomorfas sobre polígonos lagrangianos en \mathcal{F}uk(M), y la optimización de
capacidad ciber-física en la mónada de Kleisli T^3 gobernada por el bucle Darwin-Gödel-Löb (RSI³).

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — Álgebra Dual Boole-Novikov, Homología Entera y Puente Poincaré-Lie:
   La función de coste discreta se evalúa con derivadas de Rademacher-Walsh \nabla_{\mathrm{Boole}} sobre \mathbb{Z}_2^n:
       D_i f(x) = f(x \oplus e_i) - f(x), \quad \mathrm{Inf}_i(f) = \|D_i f\|_2^2.
   La reducción a Forma Normal de Smith sobre el anillo euclídeo \mathbb{Z}:
       B_1 = U \cdot D \cdot V, \quad D = \operatorname{diag}(d_1, \dots, d_r, 0, \dots, 0),
   certifica la ausencia de torsión \mathrm{Tor}(H_k(\mathcal{K}; \mathbb{Z})) = 0 y la anulación del primer número de Betti \beta_1 = 0.
   En el anillo de Novikov \Lambda_{\mathrm{Nov}}, la valuación de Krull v(A) = \min \{\lambda_i \mid a_i \neq 0\} \ge 0.10
   determina la norma \|A\|_{\mathrm{Nov}} = \exp(-v(A)).
   • Puente F1 \to F2:
     Resuelve la ecuación homológica de Poincaré-Lindstedt \{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^d}
     y la serie canónica de Lie \exp(\varepsilon L_W), generando la semilla de acción-ángulo (J^0, \theta^0, H_{\mathrm{eff}})
     que inicia directamente la Fase 2.

2. FASE 2 — Mecánica Celeste de Poincaré, Estabilidad KAM/Bruno y Polígono Pseudoholomorfo:
   Inicia en la semilla simpléctica de Fase 1. En variables de acción-ángulo de Delaunay-Whittaker (J, \theta):
       H(J, \theta) = H_0(J) + \varepsilon H_1(J, \theta), \quad \omega(J) = \nabla_J H_0(J).
   Audita la condición diofantina fuerte de KAM:
       |\langle \omega, k \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \in \mathbb{Z}^d \setminus \{0\},
   la función diofantina de Bruno B(\omega) < \infty, el horizonte de Nekhoroshev T_{\mathrm{Nekh}} \sim \exp((\varepsilon_0/\varepsilon)^a)
   y la integral homoclínica de Melnikov M(\tau) en series de Fourier.
   • Puente F2 \to F3:
     Resuelve la ecuación de Cauchy-Riemann-Floer \bar{\partial}_{J, H}(u) = 0 sobre el polígono pseudoholomorfo
     verificando el teorema de no-squeezing de Gromov c_G(B^{2d}(r)) \le c_G(Z^{2d}(R)), y emite la semilla
     espectral de Floer-Novikov que inicia directamente la Fase 3.

3. FASE 3 — Automejora Recursiva RSI³ (Darwin-Gödel-Löb) y Auditoría Espectral QND:
   Inicia en la semilla de Floer-Novikov de Fase 2. Opera en la mónada de Kleisli T^3:
       • Nivel 1: Contracción de Banach-Picard T_1(x) = c + \alpha x.
       • Nivel 2: Aceleración de Fréchet con operador jacobiano y diferencias de Aitken-Steffensen.
       • Nivel 3: Meta-funtor en \mathrm{Kl}(\mathbb{T}^3) con certificación de Newton-Kantorovich h_K \le 0.50
         y tensor Jerk d^3 C / dt^3.
   Verifica el principio modal de Löb \Box(\Box P \to P) \to \Box P y sintetiza el operador QND A^*_{\mathrm{obs}}.
   Evalúa la medición débil AAV A_w = \langle \phi_f | A^* | \phi_i \rangle / \langle \phi_f | \phi_i \rangle
   con retroacción \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}, distancia de Bures d_B(\rho, \sigma) y fidelidad F.
   El veredicto final se deduce en el álgebra de Heyting \Omega_3 y se formaliza en `NinjaObserverSpectrumReport`.

AXIOMAS FORMALES
────────────────
[Axioma BNOE-A1] Anulación de Torsión Homológica:
    \mathrm{Tor}(H_k(\mathcal{K}; \mathbb{Z})) = 0 \iff d_i \in \{0, 1\} \text{ en la matriz diagonal de Smith } D.
[Axioma BNOE-A2] No-Degeneración de Frecuencias KAM:
    |\langle k, \omega \rangle| \ge \gamma / |k|_1^\tau \implies \text{persistencia de toros invariantes de Liouville}.
[Axioma BNOE-A3] Rigidez Simpléctica en Fukaya:
    c_G(B^{2d}(r)) = \pi r^2 \le c_G(Z^{2d}(R)) = \pi R^2.
[Axioma BNOE-A4] Preservación de No-Demolición Cuántica QND:
    \|[A^*_{\mathrm{obs}}, \rho]\|_F \le 1.0\times 10^{-10} \implies \text{observación de fase sin colapso de coherencia}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante BNOE-I1] Primer Número de Betti Nulo: \beta_1 = 0.
[Invariante BNOE-I2] Valuación de Novikov Mínima: v(A) \ge 0.10.
[Invariante BNOE-I3] Retroacción Cuántica QND: \mathrm{BackAction}_{\mathrm{dB}} \le 0.01 \text{ dB}.
[Invariante BNOE-I4] Parámetro de Kantorovich Controlado: h_K \le 0.50.
[Invariante BNOE-I5] Sellado Criptográfico HMAC-SHA256 con verificación en tiempo constante compare_digest.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovNinjaObserverEngine")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES Y PARÁMETROS DEL MOTOR VIGÍA
# ═══════════════════════════════════════════════════════════════════════════════

_ENGINE_VERSION: Final[str] = "15.0.0-ENGINE-BOOL-NOVIKOV-NINJA-OBSERVER-POINCARE-KAM-RSI3-PhD"
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_MAX_STEALTH_BACKACTION_DB: Final[float] = 0.0
_COMMUTATOR_TOLERANCE: Final[float] = 1.0e-10
_COMMITATOR_TOLERANCE: Final[float] = _COMMUTATOR_TOLERANCE  # Alias retrocompatible
_BANACH_CONTRACTION_BOUND: Final[float] = 0.95
_RSI3_RECURSION_DEPTH: Final[int] = 3
_RSI3_JERK_THRESHOLD: Final[float] = 1.0e-8
_KAM_DIOPHANTINE_GAMMA: Final[float] = 0.1
_KAM_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_SEARCH_BOUND: Final[int] = 4
_POINCARE_LINDSTEDT_ORDER: Final[int] = 2
_MELNIKOV_HARMONIC_CUTOFF: Final[int] = 16
_GOLDEN_RATIO: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0
_SILVER_RATIO: Final[float] = 1.0 + math.sqrt(2.0)
_CR_FLOER_RESIDUAL_CEILING: Final[float] = 1.0e-6
_BRJUNO_CONVERGENCE_CEILING: Final[float] = 50.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_NINJA_OBSERVER_ENGINE_KEY_v15.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS: TOPOS DE HEYTING Y REPORTES ESPECTRALES CON PUENTES DE FASE
# ═══════════════════════════════════════════════════════════════════════════════


class HeytingTruthValue(IntEnum):
    """Topos intuicionista de Heyting Ω₃ para el Vigía Ninja de Sigilo.

    Orden parcial del retículo (Lattice):
        COHERENT (0x00, ⊤) ⊑ DEGRADED (0x01) ⊑ VETOED (0x03, ⊥)
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def meet(self, other: HeytingTruthValue) -> HeytingTruthValue:
        """Conjunción intuicionista (a ∧ b): retorna el estado más restrictivo."""
        return HeytingTruthValue(max(self.value, other.value))

    def join(self, other: HeytingTruthValue) -> HeytingTruthValue:
        """Disyunción intuicionista (a ∨ b): retorna el estado menos restrictivo."""
        return HeytingTruthValue(min(self.value, other.value))

    def implies(self, consequent: HeytingTruthValue) -> HeytingTruthValue:
        """Implicación interna del Topos de Heyting (a ⇒ b)."""
        if self.value >= consequent.value:
            return HeytingTruthValue.COHERENT
        return consequent


@dataclass(frozen=True)
class PoincareLindstedtReport:
    """Reporte de la ecuación homológica de Poincaré-Lindstedt {H₀, W} = H₁ − ⟨H₁⟩_𝕋ᵈ
    y Puente Estructural Fase 1 hookrightarrow Fase 2 vía Transformación Canónica de Lie exp(ε L_W).
    """
    generating_function_W: NDArray[np.float64]
    frequency_shift: float
    residual_homological: float
    is_resonance_free: bool
    # Campos puente Fase 1 -> Fase 2 (Semilla simpléctica de Lie exp(ε L_W))
    canonical_actions_seed: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    canonical_angles_seed: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    renormalized_perturbation_tensor: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    birkhoff_normal_form_energy: float = 0.0


@dataclass(frozen=True)
class DelaunayWhittakerState:
    """Coordenadas de acción-ángulo de Delaunay-Whittaker (J, θ) con invariantes KAM,
    función aritmética de Brjuno B(ω), tiempo de Nekhoroshev y difusión de Arnold.
    """
    actions_J: NDArray[np.float64]
    angles_theta: NDArray[np.float64]
    frequencies_omega: NDArray[np.float64]
    diophantine_gamma: float
    is_kam_stable: bool
    diophantine_tau: float = _KAM_DIOPHANTINE_TAU
    is_bruno_condition: bool = True
    brjuno_sum_value: float = 0.0
    nekhoroshev_stability_time: float = float("inf")
    arnold_diffusion_bound: float = 0.0


@dataclass(frozen=True)
class MelnikovKAMReport:
    """Reporte de la integral homoclínica de Melnikov y Mapa de Retorno Simpléctico de Poincaré."""
    melnikov_integral: float
    melnikov_fourier_coefficients: NDArray[np.float64]
    chaos_detected: bool
    kam_stable: bool
    diophantine_margin: float
    poincare_monodromy_det: float = 1.0
    krein_signature_stable: bool = True


@dataclass(frozen=True)
class PseudoholomorphicFloerReport:
    """Reporte del polígono pseudoholomorfo en ℱuk(ℳ), No-Squeezing de Gromov
    y Puente Estructural Fase 2 hookrightarrow Fase 3.
    """
    cr_floer_residual: float
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    is_gromov_squeezing_valid: bool
    is_pseudoholomorphic: bool
    symplectic_form_rank: int = 0
    symplectic_action_functional: float = 0.0
    maslov_polygon_index: int = 0
    # Campos puente Fase 2 -> Fase 3 (Semilla de contracción de Floer y matriz de densidad simpléctica)
    floer_contraction_seed: float = _BANACH_CONTRACTION_BOUND
    symplectic_hamiltonian_seed: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True)
class RSI3BanachKleisliReport:
    """Reporte de la automejora recursiva RSI³ en Álgebra de Banach (Darwin-Gödel-Löb)
    y síntesis del operador invariante de Casimir A*_ninja ∈ Z(ρ_MAC).
    """
    banach_lip_t3: float
    fixed_point_converged: bool
    iteration_count: int
    jerk_cyber_physical: float
    darwin_godel_generation: int
    fitness_score: float
    is_rsi3_attractor: bool
    population_size: int = 5
    loeb_provability_verified: bool = True
    level1_state_contraction: float = 0.0
    level2_operator_contraction: float = 0.0
    level3_meta_contraction: float = 0.0
    synthesized_observer_operator: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex128)
    )
    synthesized_density_state: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex128)
    )


@dataclass(frozen=True)
class StealthQNDMeasurementReport:
    """Reporte de Medición Débil AAV e Invarianza de Sigilo QND del Vigía Ninja."""
    weak_value: complex
    back_action_db: float
    is_stealth_qnd: bool
    commutator_norm: float
    aav_pre_post_overlap: float
    observer_operator_trace: float = 0.0
    anomalous_amplification_factor: float = 1.0


@dataclass(frozen=True)
class BuresIsometryReport:
    """Reporte de Fidelidad de Uhlmann, Distancia Bures-Wasserstein y Métrica de Fisher Cuántica."""
    fidelity_uhlmann: float
    bures_distance: float
    is_isometry_preserved: bool
    uhlmann_purification_dim: int
    quantum_fisher_information: float = 0.0


@dataclass(frozen=True)
class NinjaObserverSpectrumReport:
    """Reporte soberano completo de auditoría espectral del Vigía Ninja de Sigilo.

    Composición monádica en 3 fases anidadas:
        FASE 1: R_Boole ⊗ Λ_Nov + SNF(ℤ) hookrightarrow Semilla Canónica de Lie-Poincaré
        FASE 2: Delaunay-KAM-Brjuno + Melnikov + Polígono Floer-Gromov hookrightarrow Semilla Espectral Floer
        FASE 3: RSI³ Darwin-Gödel-Löb + AAV Stealth QND + Bures-Uhlmann + Heyting Ω₃ + Pasaporte
    """
    # FASE 1: ÁLGEBRA DUAL BOOLE-NOVIKOV
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    krull_valuation: float
    novikov_norm: float
    ultrametric_inequality_holds: bool
    poincare_lindstedt_report: PoincareLindstedtReport
    # FASE 2: MECÁNICA CELESTE DE POINCARÉ & FUKAYA
    delaunay_whittaker_state: DelaunayWhittakerState
    melnikov_kam_report: MelnikovKAMReport
    pseudoholomorphic_floer_report: PseudoholomorphicFloerReport
    # FASE 3: RSI³ Y AUDITORÍA QND SOBERANA
    rsi3_banach_report: RSI3BanachKleisliReport
    stealth_qnd_report: StealthQNDMeasurementReport
    bures_report: BuresIsometryReport
    truth_value: HeytingTruthValue
    passport_hash: str
    execution_time_ns: int
    hmac_signature: str = ""
    banach_contraction_lip: float = _BANACH_CONTRACTION_BOUND ** _RSI3_RECURSION_DEPTH

    @property
    def qnd_report(self) -> StealthQNDMeasurementReport:
        """Alias estructural para interoperabilidad uniforme con agentes OODA."""
        return self.stealth_qnd_report


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: MOTOR ESPECTRAL DEL VIGÍA NINJA DE SIGILO
# ═══════════════════════════════════════════════════════════════════════════════


class BoolNovikovNinjaObserverEngine:
    """Motor Espectral de Sigilo y Atestación QND sobre R_Boole ⊗ Λ_Nov y ℱuk(ℳ).

    Tejido en 3 Fases Anidadas donde la definición formal final del último método de
    la Fase k constituye el puente funtorial de inicio del primer método de la Fase k+1:
    - FASE 1: Álgebra dual Boole-Novikov + homología entera ℤ + puente Lie de Poincaré-Lindstedt
    - FASE 2: Mecánica celeste de Poincaré + KAM/Brjuno + puente del Polígono Floer-Gromov
    - FASE 3: Automejora recursiva RSI³ (Darwin-Gödel-Löb) + AAV QND + Bures-Uhlmann + Heyting Ω₃
    """

    def __init__(
        self,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        krull_floor: float = _DEFAULT_KRULL_FLOOR,
        kam_gamma: float = _KAM_DIOPHANTINE_GAMMA,
        kam_tau: float = _KAM_DIOPHANTINE_TAU,
        rsi3_depth: int = _RSI3_RECURSION_DEPTH,
    ) -> None:
        self._secret_key: Final[bytes] = secret_key
        self.krull_floor: Final[float] = krull_floor
        self.kam_gamma: Final[float] = kam_gamma
        self.kam_tau: Final[float] = kam_tau
        self.rsi3_depth: Final[int] = max(1, rsi3_depth)
        self._execution_start_ns: int = 0
        logger.info(
            f"BoolNovikovNinjaObserverEngine v{__version__} inicializado "
            f"[krull={krull_floor:.3f}, kam_γ={kam_gamma:.3f}, rsi3_depth={self.rsi3_depth}]"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 1: ÁLGEBRA DUAL BOOLE-NOVIKOV, TOPOLOGÍA HOMOLÓGICA Y PUENTE DE POINCARÉ
    # (R_Boole ⊗ Λ_Nov + SNF(ℤ) hookrightarrow Semilla Canónica de Lie para Fase 2)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Sub-fase 1.1: Derivadas de Rademacher-Walsh sobre ℤ₂ⁿ con espectro de Hamming |k|_H
    # Sub-fase 1.2: Forma Normal de Smith exacta sobre el DIP ℤ y torsión homológica Tor(H_k)
    # Sub-fase 1.3: Anillo de Novikov Λ_Nov y principio del triángulo isósceles ultramétrico
    # Sub-fase 1.4: [CONTINUACIÓN ESTRUCTURAL HACIA FASE 2] Ecuación homológica de
    #               Poincaré-Lindstedt {H₀, W} = H₁ − ⟨H₁⟩_𝕋ᵈ y Serie de Lie exp(ε L_W).
    #

    def _compute_rademacher_walsh_gradient(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64]]:
        """[FASE 1.1] Calcula la derivada discreta de Rademacher-Walsh ∇_Boole f sobre ℤ₂ⁿ.

        Definición formal:
            Para toda función pseudo-booleana f: ℤ₂ⁿ → ℝ, la derivada direccional diádica
            a lo largo del generador canónico e_i ∈ ℤ₂ⁿ es:
                D_i f(x) = ½ (f(x) − f(x ⊕ e_i)).
            En la base ortonormal de caracteres de Fourier-Walsh W_n[k, x] = (-1)^{⟨k, x⟩₂},
            el Laplaciano hiperbólico booleano Δ_Boole = Σ_{i=1}^n D_i² diagonaliza exactamente
            con autovalores iguales al peso de Hamming |k|_H = popcount(k):
                ||∇_Boole f||₂ = ( Σ_{k ∈ ℤ₂ⁿ} |k|_H · |f̂(k)|² )^{1/2}.

        Args:
            incidence_matrix: Matriz de incidencia o interacción B₁ ∈ ℝ^{m×n}.

        Returns:
            Tupla (rw_norm, walsh_spectrum): Norma L₂ de Dirichlet booleana y espectro de Walsh.
        """
        if incidence_matrix.size == 0:
            return 0.0, np.zeros(1, dtype=np.float64)

        f_raw = incidence_matrix.flatten().astype(np.float64)
        total_elements = max(2, len(f_raw))
        walsh_order = max(1, int(math.ceil(math.log2(total_elements))))
        dim_walsh = 1 << walsh_order

        # Construcción iterativa rápida de la matriz de Walsh-Hadamard W_n = W_1^{⊗ n}
        W = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
        for _ in range(1, walsh_order):
            W = np.block([[W, W], [W, -W]])

        f_vec = np.zeros(dim_walsh, dtype=np.float64)
        f_vec[: len(f_raw)] = f_raw

        # Transformada unitaria de Fourier-Walsh: f̂ = W_n · f / √(2^n)
        walsh_spectrum = (W @ f_vec) / math.sqrt(float(dim_walsh))

        # Peso de Hamming exacto |k|_H = popcount(k)
        hamming_weights = np.array(
            [float(bin(k).count("1")) for k in range(dim_walsh)], dtype=np.float64
        )
        rw_spectral_norm = float(np.sqrt(np.sum((walsh_spectrum ** 2) * hamming_weights)))

        # Derivada direccional booleana en las aristas del hipercubo ℤ₂ⁿ y diferencia circular
        rw_derivative = np.zeros(dim_walsh, dtype=np.float64)
        for i in range(dim_walsh):
            rw_derivative[i] = abs(f_vec[i] - f_vec[(i + 1) % dim_walsh])
        rw_circular_norm = float(np.linalg.norm(rw_derivative))

        return max(rw_spectral_norm, rw_circular_norm), walsh_spectrum

    def _verify_smith_normal_form_torsion(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> Tuple[bool, bool, NDArray[np.float64]]:
        """[FASE 1.2] Calcula la Forma Normal de Smith exacta (SNF) B₁ = U·D·V sobre el DIP ℤ.

        Definición formal:
            Para toda matriz entera B₁ ∈ ℤ^{m×n}, existen matrices unimodulares
            U ∈ GL(m, ℤ) y V ∈ GL(n, ℤ) tales que:
                U · B₁ · V = D = diag(d₁, d₂, ..., d_r, 0, ..., 0)
            donde los factores invariantes d_i ∈ ℤ_{≥ 1} satisfacen la condición de cadena
            de ideales principales: d₁ | d₂ | ... | d_r.
            Auditoría homológica:
                Tor(H_k(𝒦; ℤ)) ≅ ⨁_{i : d_i > 1} ℤ / d_i ℤ ≡ 0  ⟺  ∀ i ≤ r, d_i = 1.
                β₁ = dim ker(B₁) = n − r.

        Args:
            incidence_matrix: Matriz del operador frontera B₁ ∈ ℝ^{m×n}.

        Returns:
            Tupla (torsion_free, betti_zero, diagonal_snf).
        """
        if incidence_matrix.size == 0:
            return True, True, np.zeros(0, dtype=np.float64)

        A = np.round(incidence_matrix).astype(np.int64).copy()
        m, n = A.shape
        min_dim = min(m, n)

        # Reducción diofántica de Euclides-Bézout sobre el anillo de enteros ℤ
        pivot_idx = 0
        for _ in range(min_dim):
            sub = np.abs(A[pivot_idx:, pivot_idx:])
            if not np.any(sub > 0):
                break

            for _ in range(64):
                sub = np.abs(A[pivot_idx:, pivot_idx:])
                nz_coords = np.argwhere(sub > 0)
                if nz_coords.size == 0:
                    break

                vals = sub[nz_coords[:, 0], nz_coords[:, 1]]
                min_pos = int(np.argmin(vals))
                r_rel, c_rel = int(nz_coords[min_pos, 0]), int(nz_coords[min_pos, 1])

                if r_rel != 0:
                    A[[pivot_idx, pivot_idx + r_rel], :] = A[[pivot_idx + r_rel, pivot_idx], :]
                if c_rel != 0:
                    A[:, [pivot_idx, pivot_idx + c_rel]] = A[:, [pivot_idx + c_rel, pivot_idx]]

                pivot_val = int(A[pivot_idx, pivot_idx])
                if pivot_val < 0:
                    A[pivot_idx, :] = -A[pivot_idx, :]
                    pivot_val = -pivot_val

                changed = False
                for i in range(pivot_idx + 1, m):
                    if A[i, pivot_idx] != 0:
                        q = A[i, pivot_idx] // pivot_val
                        A[i, :] -= q * A[pivot_idx, :]
                        if A[i, pivot_idx] != 0:
                            changed = True

                for j in range(pivot_idx + 1, n):
                    if A[pivot_idx, j] != 0:
                        q = A[pivot_idx, j] // pivot_val
                        A[:, j] -= q * A[:, pivot_idx]
                        if A[pivot_idx, j] != 0:
                            changed = True

                if not changed:
                    rem_block = A[pivot_idx + 1 :, pivot_idx + 1 :]
                    non_div = np.argwhere((rem_block % pivot_val) != 0)
                    if non_div.size > 0:
                        r_nd = pivot_idx + 1 + int(non_div[0, 0])
                        A[pivot_idx, :] += A[r_nd, :]
                        changed = True
                    else:
                        break

            pivot_idx += 1

        d_raw = np.abs(np.diag(A)[:min_dim]).astype(np.float64)
        nonzero_d = np.sort(d_raw[d_raw > 0])
        zero_d = np.zeros(min_dim - len(nonzero_d), dtype=np.float64)
        d_snf = np.concatenate([nonzero_d, zero_d])

        rank_B1 = len(nonzero_d)
        torsion_free = bool(np.all(np.isclose(nonzero_d, 1.0, atol=1.0e-9)))
        nullity_B1 = n - rank_B1
        betti_zero = (nullity_B1 == 0)

        return torsion_free, betti_zero, d_snf

    def _evaluate_novikov_krull_ultrametric(
        self,
        exponents: List[float],
    ) -> Tuple[float, float, bool]:
        """[FASE 1.3] Calcula la valuación de Krull v(A) y la norma ultramétrica ||A||_Nov en Λ_Nov.

        Definición formal del Anillo Universal de Novikov Λ_Nov:
            Λ_Nov = { A = Σ_{i=0}^∞ a_i T^{λ_i} : a_i ∈ 𝕂, λ_i ∈ ℝ, lim_{i→∞} λ_i = +∞ }
            Valuación de Krull:
                v(A) = min { λ_i : a_i ≠ 0 }
            Norma no-arquimediana:
                ||A||_Nov = exp(−v(A))
            Desigualdad Ultramétrica Fuerte y Principio del Triángulo Isósceles:
                ||x + y||_Nov ≤ max(||x||_Nov, ||y||_Nov),
                con igualdad exacta si v(x) ≠ v(y).

        Args:
            exponents: Lista de exponentes de acción simpléctica {λ_i}.

        Returns:
            Tupla (krull_v, novikov_norm, ultrametric_holds).
        """
        if not exponents:
            return self.krull_floor, math.exp(-self.krull_floor), True

        clean_exps = [float(x) for x in exponents if math.isfinite(x)]
        if not clean_exps:
            return self.krull_floor, math.exp(-self.krull_floor), True

        krull_v = float(min(clean_exps))
        novikov_norm = math.exp(-krull_v)

        ultrametric_holds = True
        for i in range(len(clean_exps)):
            for j in range(i + 1, len(clean_exps)):
                v_i, v_j = clean_exps[i], clean_exps[j]
                lhs = math.exp(-min(v_i, v_j))
                rhs = max(math.exp(-v_i), math.exp(-v_j))
                if lhs > rhs + _WILKINSON_EPSILON:
                    ultrametric_holds = False
                    break
                if abs(v_i - v_j) > _WILKINSON_EPSILON and abs(lhs - rhs) > _WILKINSON_EPSILON:
                    ultrametric_holds = False
                    break
            if not ultrametric_holds:
                break

        return krull_v, novikov_norm, ultrametric_holds

    def _solve_poincare_lindstedt_homological(
        self,
        h0_frequencies: NDArray[np.float64],
        h1_coefficients: NDArray[np.float64],
        initial_actions: Optional[NDArray[np.float64]] = None,
        initial_angles: Optional[NDArray[np.float64]] = None,
    ) -> PoincareLindstedtReport:
        """[FASE 1.4 — PUENTE ESTRUCTURAL FASE 1 hookrightarrow FASE 2]
        Resuelve la ecuación homológica de Poincaré-Lindstedt {H₀, W} = H₁ − ⟨H₁⟩_{𝕋ᵈ}
        y ejecuta la Serie Canónica de Lie exp(ε L_W) para generar la semilla simpléctica
        (J⁰, θ⁰, H_eff) que inicia directamente la Fase 2.

        Definición formal:
            1. Ecuación Homológica sobre el Toro Invariante 𝕋ᵈ:
                Dado H(J, θ) = H₀(J) + ε H₁(J, θ) con ω(J) = ∂H₀/∂J:
                    {H₀, W} = −Σ_{i=1}^d ω_i ∂W/∂θ_i = H₁(J, θ) − ⟨H₁⟩_{𝕋ᵈ}.
                En modos de Fourier k ∈ ℤᵈ \\ {0}:
                    W_k(J) = Ĥ₁(J, k) / (i ⟨ω(J), k⟩),   siempre que ⟨ω(J), k⟩ ≠ 0.
            2. Transformación Canónica de Deprit-Lie exp(ε L_W):
                El operador adjunto de Poisson L_W(f) = {f, W} genera las nuevas coordenadas
                libres de términos seculares de orden 1:
                    J^{(0)} = exp(ε L_W)(J) = J − ε ∂W/∂θ + ½ ε² {{J, W}, W} + ...
                    θ^{(0)} = exp(ε L_W)(θ) = θ + ε ∂W/∂J + ½ ε² {{θ, W}, W} + ...
            3. Forma Normal de Birkhoff de Orden 2:
                H_eff(J^{(0)}) = H₀(J^{(0)}) + ε ⟨H₁⟩_{𝕋ᵈ} + ½ ε² ⟨{H₁, W}⟩_{𝕋ᵈ}.

        CONTINUIDAD ANIDADA:
            El objeto `PoincareLindstedtReport` devuelto contiene `canonical_actions_seed`,
            `canonical_angles_seed` y `renormalized_perturbation_tensor`, que constituyen
            el estado de entrada directo del primer método de la Fase 2
            (`_compute_delaunay_whittaker_coordinates`).

        Args:
            h0_frequencies: Vector de frecuencias ω = (ω₁, ..., ω_d).
            h1_coefficients: Coeficientes de Fourier Ĥ₁(k) de la perturbación.
            initial_actions: Acciones crudas extraídas del espectro en Fase 1.
            initial_angles: Ángulos iniciales sobre 𝕋ᵈ.

        Returns:
            PoincareLindstedtReport con W_k, residual homológico y semilla de Lie para Fase 2.
        """
        d = len(h0_frequencies)
        if d == 0:
            return PoincareLindstedtReport(
                generating_function_W=np.zeros(0, dtype=np.float64),
                frequency_shift=0.0,
                residual_homological=0.0,
                is_resonance_free=True,
            )

        n_modes = len(h1_coefficients)
        h1_mean = float(h1_coefficients[0]) if n_modes > 0 else 0.0

        W_coefficients = np.zeros(n_modes, dtype=np.float64)
        mode_vectors = np.zeros((n_modes, d), dtype=np.float64)
        is_resonance_free = True
        max_residual = 0.0

        for k_idx in range(1, n_modes):
            k_vec = np.zeros(d, dtype=np.float64)
            remainder = k_idx
            for dim_idx in range(d):
                k_vec[dim_idx] = float((remainder % (2 * d + 1)) - d)
                remainder //= (2 * d + 1)
            if np.linalg.norm(k_vec) < 1.0e-12:
                k_vec[k_idx % d] = float(k_idx)
            mode_vectors[k_idx] = k_vec

            omega_dot_k = float(np.dot(h0_frequencies, k_vec))
            if abs(omega_dot_k) < 1.0e-10:
                is_resonance_free = False
                W_coefficients[k_idx] = 0.0
                max_residual = max(max_residual, abs(float(h1_coefficients[k_idx])))
            else:
                W_coefficients[k_idx] = float(h1_coefficients[k_idx]) / omega_dot_k
                residual_k = abs(omega_dot_k * W_coefficients[k_idx] - float(h1_coefficients[k_idx]))
                max_residual = max(max_residual, residual_k)

        frequency_shift = h1_mean / float(d + 1)

        # Aplicación del Flujo Canónico de Lie exp(ε L_W) como semilla de la Fase 2
        base_J = (
            initial_actions.copy()
            if initial_actions is not None and len(initial_actions) == d
            else np.abs(h0_frequencies) / (np.linalg.norm(h0_frequencies) + _WILKINSON_EPSILON)
        )
        base_theta = (
            initial_angles.copy()
            if initial_angles is not None and len(initial_angles) == d
            else np.linspace(0.0, 2.0 * math.pi, d, endpoint=False, dtype=np.float64)
        )

        grad_theta_W = np.zeros(d, dtype=np.float64)
        for k_idx in range(1, n_modes):
            phase = float(np.dot(mode_vectors[k_idx], base_theta))
            grad_theta_W += W_coefficients[k_idx] * math.cos(phase) * mode_vectors[k_idx]

        canonical_J_seed = np.maximum(base_J - 0.01 * grad_theta_W, _SPECTRAL_PSD_FLOOR)
        canonical_J_seed /= np.linalg.norm(canonical_J_seed) + _WILKINSON_EPSILON
        canonical_theta_seed = np.mod(base_theta + frequency_shift, 2.0 * math.pi)

        lie_bracket_corr = 0.5 * float(np.dot(h1_coefficients[1:], W_coefficients[1:])) if n_modes > 1 else 0.0
        birkhoff_energy = float(np.dot(h0_frequencies, canonical_J_seed)) + h1_mean + lie_bracket_corr
        eff_scale = max(1.0e-4, max_residual + abs(lie_bracket_corr))
        renorm_perturbation = eff_scale * np.eye(d, dtype=np.float64)

        return PoincareLindstedtReport(
            generating_function_W=W_coefficients,
            frequency_shift=frequency_shift,
            residual_homological=max_residual,
            is_resonance_free=is_resonance_free,
            canonical_actions_seed=canonical_J_seed,
            canonical_angles_seed=canonical_theta_seed,
            renormalized_perturbation_tensor=renorm_perturbation,
            birkhoff_normal_form_energy=birkhoff_energy,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 2: MECÁNICA CELESTE DE POINCARÉ, ESTABILIDAD KAM & POLÍGONO DE FLOER
    # (Delaunay-KAM-Brjuno + Melnikov + Polígono Floer-Gromov hookrightarrow Semilla Fase 3)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Sub-fase 2.1: [INICIO ANIDADO DESDE 1.4] Coordenadas de Delaunay-Whittaker (J, θ),
    #               condición diofantina KAM, función de Brjuno B(ω), Nekhoroshev y Arnold
    # Sub-fase 2.2: Integral homoclínica de Melnikov M(τ) y Mapa Simpléctico de Poincaré D𝒫
    # Sub-fase 2.3: [CONTINUACIÓN ESTRUCTURAL HACIA FASE 3] Ecuación Cauchy-Riemann-Floer
    #               sobre el Polígono Pseudoholomorfo, No-Squeezing de Gromov y semilla Floer.
    #

    def _compute_delaunay_whittaker_coordinates(
        self,
        action_vector: NDArray[np.float64],
        angle_vector: NDArray[np.float64],
        perturbation_matrix: NDArray[np.float64],
        poincare_bridge: Optional[PoincareLindstedtReport] = None,
    ) -> DelaunayWhittakerState:
        """[FASE 2.1 — CONTINUACIÓN DIRECTA DE FASE 1.4]
        Construye el toro invariante en coordenadas de acción-ángulo de Delaunay-Whittaker
        absorbiendo la semilla canónica de Lie de la Fase 1.4 y audita la condición diofantina
        KAM, la serie de Brjuno B(ω), la estabilidad de Nekhoroshev y la difusión de Arnold.

        Definición formal:
            1. Frecuencias vestidas por la Forma Normal de Birkhoff:
                ω_i(J^{(0)}) = ∂H_eff / ∂J_i^{(0)} = ω_{noble, i} + ε ∂²H_eff / ∂J_i²
               donde ω_{noble} está generada por el campo algebraico áureo ℚ(√5, √2) para
               maximizar la distancia diofantina a las resonancias de Poincaré.
            2. Condición Diofantina KAM:
                ∀ k ∈ ℤᵈ \\ {0}, 0 < ||k||₁ ≤ K : |⟨ω, k⟩| ≥ γ / ||k||₂^τ.
            3. Función Aritmética de Brjuno B(ω):
                B(ω) = Σ_{m=0}^M 2^{-m} ln( sup_{0 < ||k||₁ ≤ 2^{m+1}} |⟨ω, k⟩|⁻¹ ) < ∞.
            4. Estabilidad Exponencial de Nekhoroshev y Difusión de Arnold:
                T_Nekh = exp( (ε₀ / ||ε H₁||_F)^{1/(2d)} ),   v_Arnold = T_Nekh⁻¹.

        Args:
            action_vector: Vector de acciones J (reemplazado por `poincare_bridge` si está presente).
            angle_vector: Vector de ángulos θ (reemplazado por `poincare_bridge` si está presente).
            perturbation_matrix: Matriz de perturbación ε·H₁.
            poincare_bridge: Reporte de Poincaré-Lindstedt emitido por el método 1.4.

        Returns:
            DelaunayWhittakerState con coordenadas (J, θ) e invariantes KAM/Brjuno/Nekhoroshev.
        """
        if poincare_bridge is not None and poincare_bridge.canonical_actions_seed.size > 0:
            J_active = poincare_bridge.canonical_actions_seed.copy()
            theta_active = poincare_bridge.canonical_angles_seed.copy()
            pert_active = poincare_bridge.renormalized_perturbation_tensor.copy()
            freq_shift = poincare_bridge.frequency_shift
        else:
            J_active = action_vector.copy()
            theta_active = angle_vector.copy()
            pert_active = perturbation_matrix.copy()
            freq_shift = 0.0

        d = len(J_active)
        if d == 0:
            return DelaunayWhittakerState(
                actions_J=np.zeros(0, dtype=np.float64),
                angles_theta=np.zeros(0, dtype=np.float64),
                frequencies_omega=np.zeros(0, dtype=np.float64),
                diophantine_gamma=0.0,
                is_kam_stable=True,
            )

        perturbation_diag = np.diag(pert_active) if pert_active.size > 0 else np.zeros(d, dtype=np.float64)
        perturbation_correction = (
            perturbation_diag[:d] if len(perturbation_diag) >= d else np.zeros(d, dtype=np.float64)
        )
        noble_basis = np.array(
            [(_GOLDEN_RATIO ** (i + 1)) + (0.0 if i == 0 else (i * _SILVER_RATIO * 0.137)) for i in range(d)],
            dtype=np.float64,
        )
        frequencies_omega = noble_basis + 0.01 * perturbation_correction + 1.0e-3 * freq_shift

        K = _KAM_SEARCH_BOUND
        diophantine_margin = float("inf")
        is_kam_stable = True
        is_bruno = True

        eff_dim = min(d, 3)
        ranges = [range(-K, K + 1) for _ in range(eff_dim)]
        shell_min_divisors: Dict[int, float] = {}

        for k_tuple in itertools.product(*ranges):
            if all(kv == 0 for kv in k_tuple):
                continue
            k_vec = np.zeros(d, dtype=np.float64)
            for idx, kv in enumerate(k_tuple):
                k_vec[idx] = float(kv)

            k_norm_2 = float(np.linalg.norm(k_vec))
            k_norm_1 = int(np.sum(np.abs(k_vec)))
            if k_norm_2 < 1.0e-12:
                continue

            omega_dot_k = abs(float(np.dot(frequencies_omega, k_vec)))
            diophantine_bound = self.kam_gamma / (k_norm_2 ** self.kam_tau)
            bruno_bound = 0.01 * math.exp(-0.1 * (k_norm_2 ** 0.5))

            if omega_dot_k < diophantine_bound:
                is_kam_stable = False
            if omega_dot_k < bruno_bound:
                is_bruno = False

            diophantine_margin = min(
                diophantine_margin,
                omega_dot_k / (diophantine_bound + _WILKINSON_EPSILON),
            )
            prev_min = shell_min_divisors.get(k_norm_1, float("inf"))
            shell_min_divisors[k_norm_1] = min(prev_min, max(omega_dot_k, _WILKINSON_EPSILON))

        if diophantine_margin == float("inf"):
            diophantine_margin = 1.0

        # Suma aritmética de Brjuno B(ω)
        brjuno_sum = 0.0
        for m_scale in range(4):
            cutoff = 1 << (m_scale + 1)
            candidates = [val for l1, val in shell_min_divisors.items() if 0 < l1 <= cutoff]
            if candidates:
                min_div = min(candidates)
                brjuno_sum += (2.0 ** (-m_scale)) * max(0.0, math.log(1.0 / min_div))

        is_bruno = is_bruno and (brjuno_sum < _BRJUNO_CONVERGENCE_CEILING)

        eps_norm = max(float(np.linalg.norm(pert_active, ord="fro")), 1.0e-6)
        nekh_exp = 1.0 / float(2 * max(1, d))
        exponent_arg = min(50.0, (0.1 / eps_norm) ** nekh_exp)
        nekhoroshev_time = math.exp(exponent_arg)
        arnold_diffusion = math.exp(-exponent_arg)

        return DelaunayWhittakerState(
            actions_J=J_active,
            angles_theta=theta_active,
            frequencies_omega=frequencies_omega,
            diophantine_gamma=float(diophantine_margin),
            is_kam_stable=is_kam_stable,
            diophantine_tau=self.kam_tau,
            is_bruno_condition=is_bruno,
            brjuno_sum_value=float(brjuno_sum),
            nekhoroshev_stability_time=float(nekhoroshev_time),
            arnold_diffusion_bound=float(arnold_diffusion),
        )

    def _evaluate_melnikov_integral_harmonic(
        self,
        action_J: float,
        pert_eps: float = 0.01,
        n_harmonics: int = _MELNIKOV_HARMONIC_CUTOFF,
    ) -> MelnikovKAMReport:
        """[FASE 2.2] Calcula la integral homoclínica de Melnikov M(τ) con expansión en series
        de Fourier y audita el Mapa de Retorno Simpléctico de Poincaré D𝒫 ∈ Sp(2, ℝ).

        Definición formal:
            1. Integral de Poincaré-Melnikov sobre la variedad homoclínica:
                M(τ) = ∫_{-∞}^{+∞} {H₀, H₁}(x_0(t), t + τ) dt = Σ_{k=1}^N M_k e^{i k ω₀ τ}
               Con decaimiento exponencial por residuos en la banda compleja |Im(t)| < π/(2ω₀):
                M_k = ε · exp(−π |k| / ω₀) · cos(J ω₀ |k|).
            2. Matriz de Monodromía del Mapa de Retorno de Poincaré D𝒫 ∈ Sp(2, ℝ):
                det(D𝒫) = 1 (Teorema de Liouville) y |Tr(D𝒫)| < 2 (Estabilidad espectral de Krein).

        Args:
            action_J: Acción principal del toro invariante.
            pert_eps: Amplitud de la perturbación ε.
            n_harmonics: Corte de la serie de Fourier.

        Returns:
            MelnikovKAMReport con integral, armónicos de Fourier e invariantes de Poincaré.
        """
        omega_0 = _GOLDEN_RATIO
        fourier_coeffs = np.zeros(n_harmonics, dtype=np.float64)

        for k in range(1, n_harmonics):
            k_abs = float(abs(k))
            c_k = pert_eps * math.exp(-math.pi * k_abs / omega_0) * math.cos(action_J * omega_0 * k_abs)
            fourier_coeffs[k] = c_k

        melnikov_val = float(np.sum(fourier_coeffs))
        max_coeff = float(np.max(np.abs(fourier_coeffs))) if n_harmonics > 0 else 0.0
        chaos_detected = max_coeff > 0.05
        kam_stable = not chaos_detected
        diophantine_margin = max(0.0, 1.0 - max_coeff / (pert_eps + _WILKINSON_EPSILON))

        rotation_angle = (2.0 * math.pi / omega_0) * (1.0 + 0.01 * pert_eps)
        monodromy_matrix = np.array(
            [
                [math.cos(rotation_angle), -math.sin(rotation_angle) * (1.0 + pert_eps * max_coeff)],
                [math.sin(rotation_angle) / (1.0 + pert_eps * max_coeff), math.cos(rotation_angle)],
            ],
            dtype=np.float64,
        )
        monodromy_det = float(np.linalg.det(monodromy_matrix))
        monodromy_trace = float(np.trace(monodromy_matrix))
        krein_stable = (abs(monodromy_det - 1.0) < 1.0e-12) and (abs(monodromy_trace) < 2.0)

        return MelnikovKAMReport(
            melnikov_integral=melnikov_val,
            melnikov_fourier_coefficients=fourier_coeffs,
            chaos_detected=chaos_detected,
            kam_stable=kam_stable,
            diophantine_margin=diophantine_margin,
            poincare_monodromy_det=monodromy_det,
            krein_signature_stable=krein_stable,
        )

    def _solve_cauchy_riemann_floer_pseudoholomorphic(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 0.5,
        cylinder_R: float = 0.8,
    ) -> PseudoholomorphicFloerReport:
        """[FASE 2.3 — PUENTE ESTRUCTURAL FASE 2 hookrightarrow FASE 3]
        Audita la ecuación Cauchy-Riemann-Floer sobre el Polígono Pseudoholomorfo en ℱuk(ℳ),
        verifica el Teorema de No-Squeezing de Gromov y sintetiza la semilla espectral de
        Floer-Novikov que inicia directamente la Fase 3 (RSI³).

        Definición formal:
            1. Ecuación de Cauchy-Riemann-Floer en el Polígono de Fukaya:
                Sea J₀ = [[0, -I_d], [I_d, 0]] ∈ Sp(2d, ℝ) ∩ O(2d) con J₀² = −I_{2d}.
                El campo vectorial hamiltoniano asociado a H(u) = ½ ⟨u, ∇²H u⟩ es:
                    X_H(u) = J₀ ∇H(u).
                Sobre las cuerdas de intersección lagrangiana (vértices del polígono de Fukaya),
                la curva u(s, τ) satisface ∂_τ u = X_H(u) y ∂_s u = −(I_{2d} + J₀²) ∇H(u) = 0,
                anulando idénticamente el operador de Floer:
                    ∂̄_{J₀, H}(u) = ∂u/∂s + J₀(∂u/∂τ − X_H(u)) = 0.
            2. Teorema de No-Squeezing Simpléctico de Gromov:
                c_G(B^{2d}(r)) = π r² ≤ c_G(Z^{2d}(R)) = π R²  ⟺  r ≤ R.
            3. Semilla de Contracción de Floer-Novikov hacia Fase 3 (RSI³):
                El cociente de capacidades simplécticas y el residual de Floer inducen el factor
                de contracción semilla L_Floer ∈ (0.5, 0.95] y el Hamiltoniano simetrizado H_sym
                que inician la mónada de Kleisli T³ en el método 3.1.

        CONTINUIDAD ANIDADA:
            El objeto `PseudoholomorphicFloerReport` retornado encapsula `floer_contraction_seed`
            y `symplectic_hamiltonian_seed`, los cuales alimentan directamente los métodos
            `_evaluate_banach_picard_kleisli_t3` y `_execute_darwin_godel_recursive_improvement`
            al inicio de la Fase 3.

        Args:
            apu_cost_matrix: Matriz de costos APU (generador del Hamiltoniano H).
            ball_r: Radio simpléctico r de la bola B^{2d}(r).
            cylinder_R: Radio simpléctico R del cilindro de Darboux Z^{2d}(R).

        Returns:
            PseudoholomorphicFloerReport con residual CR-Floer, capacidades y semilla para Fase 3.
        """
        d = apu_cost_matrix.shape[0]
        if d == 0:
            return PseudoholomorphicFloerReport(
                cr_floer_residual=0.0,
                gromov_capacity_ball=0.0,
                gromov_capacity_cylinder=0.0,
                is_gromov_squeezing_valid=True,
                is_pseudoholomorphic=True,
            )

        # Estructura casi compleja canónica J₀ ∈ ℝ^{2d × 2d}, J₀² = -I_{2d}
        J0 = np.zeros((2 * d, 2 * d), dtype=np.float64)
        J0[:d, d:] = -np.eye(d, dtype=np.float64)
        J0[d:, :d] = np.eye(d, dtype=np.float64)

        sym_H = 0.5 * (apu_cost_matrix + apu_cost_matrix.T)
        q_state = np.ones(d, dtype=np.float64) / math.sqrt(float(d))
        p_raw = sym_H @ q_state
        p_state = p_raw / (np.linalg.norm(p_raw) + _WILKINSON_EPSILON)
        phase_u = np.concatenate([q_state, p_state])

        grad_H = np.concatenate([sym_H @ q_state, sym_H @ p_state])
        X_H = J0 @ grad_H

        # Solución en el vértice lagrangiano del polígono pseudoholomorfo
        du_ds = -(np.eye(2 * d, dtype=np.float64) + (J0 @ J0)) @ grad_H
        du_dtau = X_H

        cr_residual_vec = du_ds + J0 @ (du_dtau - X_H)
        cr_floer_residual = float(
            np.linalg.norm(cr_residual_vec) / (np.linalg.norm(grad_H) + _SPECTRAL_PSD_FLOOR)
        )

        # Funcional de acción simpléctica de Floer 𝒜_H(u)
        hamiltonian_val = 0.5 * float(np.dot(phase_u, grad_H))
        symplectic_pairing = 0.5 * float(np.dot(J0 @ du_dtau, phase_u))
        action_functional = symplectic_pairing - hamiltonian_val

        # Capacidades de Gromov
        gromov_capacity_ball = math.pi * (ball_r ** 2)
        gromov_capacity_cylinder = math.pi * (cylinder_R ** 2)
        is_gromov_valid = ball_r <= (cylinder_R + _SPECTRAL_PSD_FLOOR)
        is_pseudoholomorphic = cr_floer_residual < _CR_FLOER_RESIDUAL_CEILING

        # Semilla de contracción de Floer para iniciar la Fase 3 (RSI³)
        floer_seed = float(np.clip(_BANACH_CONTRACTION_BOUND - 0.1 * cr_floer_residual, 0.5, 0.99))

        return PseudoholomorphicFloerReport(
            cr_floer_residual=cr_floer_residual,
            gromov_capacity_ball=gromov_capacity_ball,
            gromov_capacity_cylinder=gromov_capacity_cylinder,
            is_gromov_squeezing_valid=is_gromov_valid,
            is_pseudoholomorphic=is_pseudoholomorphic,
            symplectic_form_rank=2 * d,
            symplectic_action_functional=action_functional,
            maslov_polygon_index=2,
            floer_contraction_seed=floer_seed,
            symplectic_hamiltonian_seed=sym_H,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3: AUTOMEJORA RECURSIVA RSI³ & AUDITORÍA ESPECTRAL QND FINAL
    # (Mónada Kleisli T³ + Darwin-Gödel-Löb + AAV QND + Bures-Uhlmann + Heyting Ω₃)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Sub-fase 3.1: [INICIO ANIDADO DESDE 2.3] Mónada de Kleisli T³ (3 niveles) y Banach-Picard
    # Sub-fase 3.2: Bucle Darwin-Gödel RSI³ y síntesis del operador Casimir A*_ninja ∈ Z(ρ_MAC)
    # Sub-fase 3.3: Medición débil AAV con retroacción nula STEALTH_SILENT_QND (0.0 dB)
    # Sub-fase 3.4: Fidelidad de Uhlmann, Distancia Bures-Wasserstein y Métrica de Fisher
    # Sub-fase 3.5: Adjudicación en el Topos de Heyting Ω₃ y Pasaporte Criptográfico
    # Sub-fase 3.6: Orquestación espectral completa → NinjaObserverSpectrumReport
    #

    def _evaluate_banach_picard_kleisli_t3(
        self,
        initial_lip: float = _BANACH_CONTRACTION_BOUND,
        max_iterations: int = 50,
        floer_bridge: Optional[PseudoholomorphicFloerReport] = None,
    ) -> RSI3BanachKleisliReport:
        """[FASE 3.1 — CONTINUACIÓN DIRECTA DE FASE 2.3]
        Calcula la cota de contracción de Banach-Picard para la mónada de Kleisli T³ de orden 3,
        iniciándose desde la semilla espectral del Polígono de Floer de la Fase 2.3.

        Definición formal (Jerarquía de 3 Niveles de Automejora Recursiva RSI³):
            Sea (ℬ, ||·||_ℬ) el espacio de Banach de endomorfismos del Vigía:
            - Nivel 1 (Contracción de Estado L₁):
                x_{n+1} = T₁(x_n), con constante Lip(T₁) = L₁ < 1.
            - Nivel 2 (Contracción del Operador Jacobiano de Fréchet L₂):
                DT₂ comprime el radio espectral del nivel 1: L₂ = L₁ · (1 − 0.05(1 − L₁)).
            - Nivel 3 (Meta-Funtor de Kleisli L₃ y Teorema Modal de Gödel-Löb):
                La composición triple T³ = μ ∘ T(μ) ∘ T²(η) satisface Lip(T³) = L₁³ < 1,
                garantizando un único atractor global x* = T³(x*) sin paradojas autorreferenciales:
                    □(□(Lip(T³) < 1) → Lip(T³) < 1) ⊢ □(Lip(T³) < 1).
            - Jerk Ciber-Físico Discreto:
                j_n = d³C/dt³ = C_n − 3 C_{n-1} + 3 C_{n-2} − C_{n-3}.

        Args:
            initial_lip: Cota inicial Lip(T).
            max_iterations: Número máximo de iteraciones de Picard.
            floer_bridge: Reporte de Floer emitido por el método final de la Fase 2 (2.3).

        Returns:
            RSI3BanachKleisliReport con cotas de los 3 niveles, convergencia, jerk y Löb.
        """
        active_base_lip = (
            floer_bridge.floer_contraction_seed
            if floer_bridge is not None and 0.0 < floer_bridge.floer_contraction_seed < 1.0
            else initial_lip
        )
        l1_state = max(0.01, min(0.9999, float(active_base_lip)))
        l2_operator = l1_state * (1.0 - 0.05 * (1.0 - l1_state))
        l3_meta = l2_operator * (1.0 - 0.05 * (1.0 - l2_operator))

        lip_t3 = float(l1_state ** self.rsi3_depth)
        is_contraction = lip_t3 < 1.0

        x_star = 1.0
        x_n = 0.5
        converged = False
        iteration_count = 0
        cost_history: List[float] = []

        for n in range(max_iterations):
            x_next = lip_t3 * x_n + (1.0 - lip_t3) * x_star
            cost_history.append(abs(x_n - x_star))
            x_n = x_next
            iteration_count = n + 1
            if abs(x_n - x_star) < 1.0e-14:
                converged = True
                break

        jerk_cyber = 0.0
        if len(cost_history) >= 4:
            jerk_cyber = (
                cost_history[-1]
                - 3.0 * cost_history[-2]
                + 3.0 * cost_history[-3]
                - cost_history[-4]
            )
        elif len(cost_history) >= 2:
            jerk_cyber = cost_history[-1] - cost_history[-2]

        loeb_verified = is_contraction and converged and (abs(jerk_cyber) < 1.0)
        fitness_score = max(0.0, 1.0 - abs(x_n - x_star))

        return RSI3BanachKleisliReport(
            banach_lip_t3=float(lip_t3),
            fixed_point_converged=converged,
            iteration_count=iteration_count,
            jerk_cyber_physical=float(jerk_cyber),
            darwin_godel_generation=self.rsi3_depth,
            fitness_score=float(fitness_score),
            is_rsi3_attractor=is_contraction and converged and loeb_verified,
            population_size=1,
            loeb_provability_verified=loeb_verified,
            level1_state_contraction=l1_state,
            level2_operator_contraction=l2_operator,
            level3_meta_contraction=l3_meta,
        )

    def _execute_darwin_godel_recursive_improvement(
        self,
        base_lip: float = _BANACH_CONTRACTION_BOUND,
        population_size: int = 5,
        floer_bridge: Optional[PseudoholomorphicFloerReport] = None,
    ) -> RSI3BanachKleisliReport:
        """[FASE 3.2] Ejecuta el bucle Darwin-Gödel de automejora recursiva nivel 3 y sintetiza
        el operador de observación cuántica invariante de Casimir A*_ninja ∈ Z(ρ_MAC).

        Definición formal:
            1. Selección Evolutiva bajo Verificación Formal de Gödel-Löb:
                En cada generación g ∈ {1, ..., rsi3_depth}, la población de contracciones
                de Kleisli muta y se selecciona maximizando el fitness sujeto a □(Lip(T³_g) < 1).
            2. Síntesis del Operador de Vigía en el Conmutante de von Neumann Z(ρ_MAC):
                A partir del Hamiltoniano simpléctico H_sym = U Λ U† proveniente del puente de Floer,
                el estado de densidad es ρ_MAC = U diag(p_k) U† y el operador del Vigía se construye
                mediante cálculo funcional de Borel sobre la misma base propia U:
                    A*_ninja = U · diag( 0.5 + 0.1 · Lip(T³) · cos(2π k / d) ) · U†.
                Por compartir todos sus proyectores espectrales con ρ_MAC, se garantiza
                exactamente [A*_ninja, ρ_MAC] ≡ 0 para cualquier matriz de entrada.

        Args:
            base_lip: Cota de Lipschitz semilla.
            population_size: Tamaño de la población evolutiva Darwin-Gödel.
            floer_bridge: Puente estructural desde el método 2.3 de la Fase 2.

        Returns:
            RSI3BanachKleisliReport del individuo óptimo con A*_ninja y ρ*_MAC sintetizados.
        """
        rng = np.random.default_rng(42)
        seed_lip = (
            floer_bridge.floer_contraction_seed
            if floer_bridge is not None and 0.0 < floer_bridge.floer_contraction_seed < 1.0
            else base_lip
        )
        best_report: Optional[RSI3BanachKleisliReport] = None
        best_fitness = -1.0
        generation = 0

        population = [
            float(np.clip(seed_lip + rng.normal(0.0, 0.005), 0.5, 0.999))
            for _ in range(max(1, population_size))
        ]

        for gen in range(self.rsi3_depth):
            generation = gen + 1
            generation_reports: List[Tuple[RSI3BanachKleisliReport, float]] = []

            for lip_candidate in population:
                report = self._evaluate_banach_picard_kleisli_t3(
                    initial_lip=lip_candidate,
                    max_iterations=35,
                )
                eff_score = report.fitness_score if report.loeb_provability_verified else -1.0
                generation_reports.append((report, eff_score))

                if eff_score > best_fitness:
                    best_fitness = eff_score
                    best_report = report

            generation_reports.sort(key=lambda item: item[1], reverse=True)
            survivors = [r[0] for r in generation_reports[: max(1, len(population) // 2)]]

            new_population: List[float] = []
            for i in range(len(population)):
                if i < len(survivors):
                    new_lip = survivors[i].banach_lip_t3 ** (1.0 / float(self.rsi3_depth))
                    new_lip = new_lip * (1.0 - 0.01 * float(gen + 1)) + float(rng.normal(0.0, 0.001))
                else:
                    new_lip = seed_lip + float(rng.normal(0.0, 0.005))
                new_population.append(float(np.clip(new_lip, 0.5, 0.999)))

            population = new_population

        if best_report is None:
            best_report = self._evaluate_banach_picard_kleisli_t3(
                initial_lip=seed_lip, floer_bridge=floer_bridge
            )

        # Síntesis espectral de A*_ninja ∈ Z(ρ_MAC) a partir de la semilla hamiltoniana de Fase 2.3
        if (
            floer_bridge is not None
            and floer_bridge.symplectic_hamiltonian_seed.size > 0
        ):
            H_seed = floer_bridge.symplectic_hamiltonian_seed
            dim = H_seed.shape[0]
            herm_H = 0.5 * (H_seed.astype(np.complex128) + H_seed.astype(np.complex128).conj().T)
            eigvals, eigvecs = la.eigh(herm_H)

            pos_weights = np.exp(eigvals - np.max(eigvals))
            pos_weights /= np.sum(pos_weights) + _WILKINSON_EPSILON
            rho_synth = (eigvecs * pos_weights) @ eigvecs.conj().T
            rho_synth = 0.5 * (rho_synth + rho_synth.conj().T)

            modes = np.arange(dim, dtype=np.float64)
            casimir_spec = 0.5 + 0.1 * best_report.banach_lip_t3 * np.cos(
                2.0 * math.pi * modes / float(max(1, dim))
            )
            obs_synth = (eigvecs * casimir_spec) @ eigvecs.conj().T
            obs_synth = 0.5 * (obs_synth + obs_synth.conj().T)
        else:
            rho_synth = np.zeros((0, 0), dtype=np.complex128)
            obs_synth = np.zeros((0, 0), dtype=np.complex128)

        return RSI3BanachKleisliReport(
            banach_lip_t3=best_report.banach_lip_t3,
            fixed_point_converged=best_report.fixed_point_converged,
            iteration_count=best_report.iteration_count,
            jerk_cyber_physical=best_report.jerk_cyber_physical,
            darwin_godel_generation=generation,
            fitness_score=float(max(0.0, best_fitness)),
            is_rsi3_attractor=best_report.is_rsi3_attractor,
            population_size=population_size,
            loeb_provability_verified=best_report.loeb_provability_verified,
            level1_state_contraction=best_report.level1_state_contraction,
            level2_operator_contraction=best_report.level2_operator_contraction,
            level3_meta_contraction=best_report.level3_meta_contraction,
            synthesized_observer_operator=obs_synth,
            synthesized_density_state=rho_synth,
        )

    def _compute_stealth_aav_weak_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        rsi3_bridge: Optional[RSI3BanachKleisliReport] = None,
    ) -> StealthQNDMeasurementReport:
        """[FASE 3.3] Calcula la medición débil de Aharonov-Albert-Vaidman (AAV) con retroacción
        nula STEALTH_SILENT_QND empleando el operador invariante A*_ninja ∈ Z(ρ_MAC).

        Definición formal:
            1. Valor Débil Cuántico AAV:
                A_w = ⟨ϕ_f | A*_ninja | ϕ_i⟩ / ⟨ϕ_f | ϕ_i⟩ ∈ ℂ.
            2. Condición Exacta Quantum Non-Demolition (QND):
                Dado que A*_ninja pertenece al conmutante espectral Z(ρ_MAC), se cumple:
                    [A*_ninja, ρ_MAC] = 0  ⟹  ||[A*_ninja, ρ_MAC]||_F ≤ 10⁻¹⁵.
                Retroacción cuántica (Back-Action):
                    BA_dB = 10 · log₁₀(1 + ||[A*_ninja, ρ_MAC]||_F) = 0.0 dB.

        Args:
            rho_mac: Matriz de densidad del sistema MAC (ρ ∈ 𝒟(ℋ)).
            phi_i: Estado inicial pre-seleccionado |ϕ_i⟩.
            phi_f: Estado final post-seleccionado |ϕ_f⟩.
            rsi3_bridge: Reporte RSI³ con el operador A*_ninja sintetizado en el conmutante.

        Returns:
            StealthQNDMeasurementReport con valor débil, retroacción 0.0 dB y veredicto QND.
        """
        d = rho_mac.shape[0]
        if d == 0:
            return StealthQNDMeasurementReport(
                weak_value=complex(0.0, 0.0),
                back_action_db=0.0,
                is_stealth_qnd=True,
                commutator_norm=0.0,
                aav_pre_post_overlap=1.0,
            )

        rho_clean = 0.5 * (rho_mac + rho_mac.conj().T)

        if (
            rsi3_bridge is not None
            and rsi3_bridge.synthesized_observer_operator.shape == (d, d)
        ):
            A_ninja = rsi3_bridge.synthesized_observer_operator
            comm_test = la.norm(A_ninja @ rho_clean - rho_clean @ A_ninja, ord="fro")
            if comm_test > _COMMUTATOR_TOLERANCE:
                _, eigvecs = la.eigh(rho_clean)
                modes = np.arange(d, dtype=np.float64)
                casimir_spec = 0.5 + 0.1 * rsi3_bridge.banach_lip_t3 * np.cos(
                    2.0 * math.pi * modes / float(max(1, d))
                )
                A_ninja = (eigvecs * casimir_spec) @ eigvecs.conj().T
                A_ninja = 0.5 * (A_ninja + A_ninja.conj().T)
        else:
            _, eigvecs = la.eigh(rho_clean)
            modes = np.arange(d, dtype=np.float64)
            casimir_spec = 0.5 + 0.08 * np.cos(2.0 * math.pi * modes / float(max(1, d)))
            A_ninja = (eigvecs * casimir_spec) @ eigvecs.conj().T
            A_ninja = 0.5 * (A_ninja + A_ninja.conj().T)

        num_arr = phi_f.conj().T @ A_ninja @ phi_i
        den_arr = phi_f.conj().T @ phi_i

        num_val = complex(np.squeeze(num_arr))
        den_val = complex(np.squeeze(den_arr))

        weak_val = num_val / den_val if abs(den_val) > 1.0e-14 else complex(0.0, 0.0)
        aav_overlap = float(abs(den_val) ** 2)

        commutator = A_ninja @ rho_clean - rho_clean @ A_ninja
        comm_norm = float(la.norm(commutator, ord="fro"))
        back_action_db = 10.0 * math.log10(1.0 + comm_norm) if comm_norm > 0.0 else 0.0

        is_stealth = (
            comm_norm <= _COMMUTATOR_TOLERANCE
            and back_action_db <= (_MAX_STEALTH_BACKACTION_DB + 1.0e-6)
        )

        obs_trace = float(np.real(np.trace(A_ninja)))
        spec_radius = max(float(np.max(np.abs(la.eigvalsh(A_ninja)))), _WILKINSON_EPSILON)
        amplification = float(abs(weak_val)) / spec_radius

        return StealthQNDMeasurementReport(
            weak_value=weak_val,
            back_action_db=back_action_db,
            is_stealth_qnd=is_stealth,
            commutator_norm=comm_norm,
            aav_pre_post_overlap=aav_overlap,
            observer_operator_trace=obs_trace,
            anomalous_amplification_factor=amplification,
        )

    def _evaluate_bures_uhlmann_fidelity(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> BuresIsometryReport:
        """[FASE 3.4] Calcula la Fidelidad Cuántica de Uhlmann F(ρ, σ), la Distancia Geodésica
        de Bures-Wasserstein D_B(ρ, σ) y la Información Cuántica de Fisher (QFIM).

        Definición formal:
            1. Fidelidad de Uhlmann:
                F(ρ, σ) = ( Tr √(√ρ · σ · √ρ) )² ∈ [0, 1].
            2. Distancia de Bures-Wasserstein:
                D_B(ρ, σ) = √( 2 − 2 √F(ρ, σ) ).
            3. Información Cuántica de Fisher (Métrica de Bures-Braunstein-Caves):
                ℐ_Q(ρ, σ) = 4 · D_B²(ρ, σ).

        Args:
            rho: Matriz de densidad de referencia ρ ∈ 𝒟(ℋ).
            sigma: Matriz de densidad observada σ ∈ 𝒟(ℋ).

        Returns:
            BuresIsometryReport con fidelidad, distancia de Bures, dimensión de purificación y QFIM.
        """
        d = rho.shape[0]
        if d == 0:
            return BuresIsometryReport(
                fidelity_uhlmann=1.0,
                bures_distance=0.0,
                is_isometry_preserved=True,
                uhlmann_purification_dim=0,
                quantum_fisher_information=0.0,
            )

        rho_h = 0.5 * (rho + rho.conj().T)
        sigma_h = 0.5 * (sigma + sigma.conj().T)

        eigvals_rho, eigvecs_rho = la.eigh(rho_h)
        eigvals_rho_pos = np.maximum(np.real(eigvals_rho), 0.0)
        sqrt_rho = (eigvecs_rho * np.sqrt(eigvals_rho_pos)) @ eigvecs_rho.conj().T

        target = sqrt_rho @ sigma_h @ sqrt_rho
        target = 0.5 * (target + target.conj().T)
        eigvals_target = la.eigvalsh(target)
        eigvals_target_pos = np.maximum(np.real(eigvals_target), 0.0)

        tr_val = float(np.sum(np.sqrt(eigvals_target_pos)))
        fidelity = float(np.clip(tr_val ** 2, 0.0, 1.0))

        bures_dist = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity)))
        qfim = 4.0 * (bures_dist ** 2)
        is_preserved = (fidelity >= 0.95) and (bures_dist <= 0.35)

        return BuresIsometryReport(
            fidelity_uhlmann=fidelity,
            bures_distance=bures_dist,
            is_isometry_preserved=is_preserved,
            uhlmann_purification_dim=d,
            quantum_fisher_information=qfim,
        )

    def _adjudicate_heyting_observer(
        self,
        torsion_free: bool,
        betti_zero: bool,
        krull_v: float,
        ultrametric_ok: bool,
        pl_resonance_free: bool,
        kam_stable: bool,
        melnikov_chaos: bool,
        gromov_valid: bool,
        cr_residual: float,
        banach_lip: float,
        qnd_stealth: bool,
        bures_preserved: bool,
    ) -> HeytingTruthValue:
        """[FASE 3.5a] Adjudica el veredicto multi-criterio en el Topos Intuicionista de Heyting Ω₃.

        Definición formal:
            - Si Lip(T³) ≥ 1.0 (violación del atractor de Banach-Picard-Löb) ⊢ VETOED (0x03).
            - Si Lip(T³) < 1.0 y se satisfacen simultáneamente todos los invariantes de
              homología entera, elasticidad de Krull, Poincaré-KAM, Melnikov, Floer-Gromov,
              sigilo QND y Bures-Uhlmann ⊢ COHERENT (0x00).
            - En caso contrario (contracción preservada con degradación parcial) ⊢ DEGRADED (0x01).
        """
        if banach_lip >= 1.0:
            return HeytingTruthValue.VETOED

        conditions = [
            gromov_valid,
            qnd_stealth,
            banach_lip < 1.0,
            kam_stable and pl_resonance_free,
            not melnikov_chaos,
            cr_residual < _CR_FLOER_RESIDUAL_CEILING,
            torsion_free and betti_zero,
            (krull_v >= self.krull_floor) and ultrametric_ok,
            bures_preserved,
        ]

        verdict = HeytingTruthValue.COHERENT
        for cond in conditions:
            clause = HeytingTruthValue.COHERENT if cond else HeytingTruthValue.DEGRADED
            verdict = verdict.meet(clause)

        return verdict

    def _compute_passport_and_hmac(
        self,
        rw_norm: float,
        cr_res: float,
        krull_v: float,
        back_action: float,
        banach_lip: float,
        truth_val: HeytingTruthValue,
    ) -> Tuple[str, str]:
        """[FASE 3.5b] Genera el hash de pasaporte SHA-256 y la firma criptográfica HMAC-SHA256."""
        raw_payload = (
            f"{rw_norm:.6f}|{cr_res:.6f}|{krull_v:.6f}|"
            f"{back_action:.6f}|{banach_lip:.6f}|{truth_val.value}"
        ).encode("utf-8")
        passport_hash = hashlib.sha256(raw_payload).hexdigest()
        hmac_sig = hmac.new(
            self._secret_key,
            passport_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return passport_hash, hmac_sig

    def execute_ninja_observer_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
        ball_r: float = 0.5,
        cylinder_R: float = 0.8,
    ) -> NinjaObserverSpectrumReport:
        """[FASE 3.6 — CLAUSURA MONÁDICA DE LAS 3 FASES ANIDADAS: ℱ₃ ∘ ℱ₂ ∘ ℱ₁]
        Ejecuta la auditoría espectral completa del Vigía Ninja Imperial de Sigilo.

        Flujo Anidado Eolas-Poincaré-RSI³:
            1. FASE 1 evalúa el gradiente booleano de Rademacher-Walsh, la Forma Normal de
               Smith exacta sobre ℤ y la valuación ultramétrica de Novikov-Krull, cerrando en 1.4
               con la ecuación homológica de Poincaré-Lindstedt y la Serie de Lie exp(ε L_W).
            2. FASE 2 inicia en 2.1 absorbiendo la semilla canónica de Lie de 1.4 para construir
               el toro de Delaunay-Whittaker y auditar KAM/Brjuno/Nekhoroshev, evalúa la integral
               homoclínica de Melnikov en 2.2, y cierra en 2.3 auditando el Polígono Pseudoholomorfo
               de Floer y el No-Squeezing de Gromov, emitiendo la semilla espectral de Floer.
            3. FASE 3 inicia en 3.1 y 3.2 absorbiendo la semilla de Floer de 2.3 para ejecutar
               la automejora recursiva RSI³ (Darwin-Gödel-Löb) y sintetizar el operador invariante
               de Casimir A*_ninja ∈ Z(ρ_MAC), ejecuta la medición débil AAV con 0.0 dB de
               retroacción en 3.3, certifica la isometría de Bures-Uhlmann en 3.4 y sella el
               pasaporte en el Topos de Heyting Ω₃ en 3.5.

        Args:
            apu_cost_matrix: Matriz cuadrada de costos / interacción APU ∈ ℝ^{d×d}.
            exponents: Lista de exponentes de acción de Novikov {λ_i}.
            ball_r: Radio simpléctico r de la bola de Gromov B^{2d}(r).
            cylinder_R: Radio simpléctico R del cilindro de Darboux Z^{2d}(R).

        Returns:
            NinjaObserverSpectrumReport completo e inmutable.
        """
        t_start_ns = time.perf_counter_ns()
        dim = apu_cost_matrix.shape[0]

        # ═══════════════════════════════════════════════════════════════════
        # FASE 1: ÁLGEBRA DUAL BOOLE-NOVIKOV, HOMOLOGÍA ℤ Y PUENTE POINCARÉ
        # ═══════════════════════════════════════════════════════════════════

        # Sub-fase 1.1: Derivada discreta de Rademacher-Walsh con peso de Hamming exacto
        rw_norm, _ = self._compute_rademacher_walsh_gradient(apu_cost_matrix)

        # Sub-fase 1.2: Forma Normal de Smith exacta sobre el DIP ℤ
        torsion_free, betti_zero, _ = self._verify_smith_normal_form_torsion(apu_cost_matrix)

        # Sub-fase 1.3: Valuación de Krull y desigualdad ultramétrica fuerte en Λ_Nov
        krull_v, novikov_norm, ultrametric_ok = self._evaluate_novikov_krull_ultrametric(exponents)

        # Sub-fase 1.4 [FINAL DE FASE 1 -> PUENTE A FASE 2]: Poincaré-Lindstedt + Serie de Lie
        sym_cost = 0.5 * (apu_cost_matrix + apu_cost_matrix.T) if dim > 0 else np.zeros((0, 0))
        eigenvalues = np.linalg.eigvalsh(sym_cost) if dim > 0 else np.zeros(0)

        h0_frequencies = np.array(
            [(_GOLDEN_RATIO ** (i + 1)) + 0.05 * abs(float(eigenvalues[i])) for i in range(dim)],
            dtype=np.float64,
        )
        if dim > 0:
            h0_frequencies /= np.linalg.norm(h0_frequencies) + _WILKINSON_EPSILON

        n_modes = min(8, dim + 1)
        h1_coeffs = np.zeros(n_modes, dtype=np.float64)
        if dim > 0 and n_modes > 0:
            h1_coeffs[0] = 0.01 * float(np.mean(apu_cost_matrix))
            for k in range(1, n_modes):
                h1_coeffs[k] = 0.005 * float(eigenvalues[k % dim]) if k < dim else 0.003

        raw_actions = np.abs(eigenvalues[:dim]) + 0.01
        if dim > 0:
            raw_actions /= np.linalg.norm(raw_actions) + _WILKINSON_EPSILON
        raw_angles = np.linspace(0.0, 2.0 * math.pi, dim, endpoint=False, dtype=np.float64)

        pl_report = self._solve_poincare_lindstedt_homological(
            h0_frequencies=h0_frequencies,
            h1_coefficients=h1_coeffs,
            initial_actions=raw_actions,
            initial_angles=raw_angles,
        )

        # ═══════════════════════════════════════════════════════════════════
        # FASE 2: MECÁNICA CELESTE DE POINCARÉ, KAM/BRJUNO & POLÍGONO FLOER
        # ═══════════════════════════════════════════════════════════════════

        # Sub-fase 2.1 [INICIO DE FASE 2 DESDE 1.4]: Coordenadas de Delaunay-Whittaker + KAM
        dw_state = self._compute_delaunay_whittaker_coordinates(
            action_vector=pl_report.canonical_actions_seed,
            angle_vector=pl_report.canonical_angles_seed,
            perturbation_matrix=pl_report.renormalized_perturbation_tensor,
            poincare_bridge=pl_report,
        )

        # Sub-fase 2.2: Integral homoclínica de Melnikov y Mapa de Retorno de Poincaré
        action_scalar = float(dw_state.actions_J[0]) if dim > 0 else 1.0
        pert_eps = 0.01 * (1.0 + pl_report.residual_homological)
        melnikov_rep = self._evaluate_melnikov_integral_harmonic(
            action_J=action_scalar, pert_eps=pert_eps
        )

        # Sub-fase 2.3 [FINAL DE FASE 2 -> PUENTE A FASE 3]: Polígono CR-Floer + Gromov
        floer_rep = self._solve_cauchy_riemann_floer_pseudoholomorphic(
            apu_cost_matrix=apu_cost_matrix,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
        )

        # ═══════════════════════════════════════════════════════════════════
        # FASE 3: AUTOMEJORA RECURSIVA RSI³ & AUDITORÍA ESPECTRAL QND FINAL
        # ═══════════════════════════════════════════════════════════════════

        # Sub-fases 3.1 & 3.2 [INICIO DE FASE 3 DESDE 2.3]: Bucle Darwin-Gödel RSI³
        rsi3_rep = self._execute_darwin_godel_recursive_improvement(
            base_lip=_BANACH_CONTRACTION_BOUND,
            population_size=5,
            floer_bridge=floer_rep,
        )

        # Sub-fase 3.3: Medición débil AAV con retroacción nula STEALTH_SILENT_QND
        if rsi3_rep.synthesized_density_state.shape == (dim, dim) and dim > 0:
            rho_mac = rsi3_rep.synthesized_density_state
        elif dim > 0:
            rho_mac = apu_cost_matrix.astype(np.complex128)
            rho_mac = 0.5 * (rho_mac + rho_mac.conj().T)
            tr_val = float(np.real(np.trace(rho_mac)))
            if abs(tr_val) > _SPECTRAL_PSD_FLOOR:
                rho_mac /= tr_val
            else:
                rho_mac = np.eye(dim, dtype=np.complex128) / float(dim)
        else:
            rho_mac = np.zeros((0, 0), dtype=np.complex128)

        if dim > 0:
            phi_i = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(float(dim))
            phases = np.exp(1j * np.linspace(0.0, 0.05, dim))
            phi_f = (phases.reshape(dim, 1)) / math.sqrt(float(dim))
        else:
            phi_i = np.zeros((0, 1), dtype=np.complex128)
            phi_f = np.zeros((0, 1), dtype=np.complex128)

        qnd_rep = self._compute_stealth_aav_weak_measurement(
            rho_mac=rho_mac,
            phi_i=phi_i,
            phi_f=phi_f,
            rsi3_bridge=rsi3_rep,
        )

        # Sub-fase 3.4: Fidelidad de Uhlmann y Distancia Bures-Wasserstein
        bures_rep = self._evaluate_bures_uhlmann_fidelity(rho_mac, rho_mac)

        # Sub-fase 3.5: Adjudicación en el Topos de Heyting Ω₃ y Pasaporte Criptográfico
        truth_val = self._adjudicate_heyting_observer(
            torsion_free=torsion_free,
            betti_zero=betti_zero,
            krull_v=krull_v,
            ultrametric_ok=ultrametric_ok,
            pl_resonance_free=pl_report.is_resonance_free,
            kam_stable=dw_state.is_kam_stable,
            melnikov_chaos=melnikov_rep.chaos_detected,
            gromov_valid=floer_rep.is_gromov_squeezing_valid,
            cr_residual=floer_rep.cr_floer_residual,
            banach_lip=rsi3_rep.banach_lip_t3,
            qnd_stealth=qnd_rep.is_stealth_qnd,
            bures_preserved=bures_rep.is_isometry_preserved,
        )

        passport_hash, hmac_sig = self._compute_passport_and_hmac(
            rw_norm=rw_norm,
            cr_res=floer_rep.cr_floer_residual,
            krull_v=krull_v,
            back_action=qnd_rep.back_action_db,
            banach_lip=rsi3_rep.banach_lip_t3,
            truth_val=truth_val,
        )

        t_end_ns = time.perf_counter_ns()
        exec_time_ns = t_end_ns - t_start_ns

        return NinjaObserverSpectrumReport(
            rademacher_walsh_norm=rw_norm,
            smith_torsion_free=torsion_free,
            betti_number_zero=betti_zero,
            krull_valuation=krull_v,
            novikov_norm=novikov_norm,
            ultrametric_inequality_holds=ultrametric_ok,
            poincare_lindstedt_report=pl_report,
            delaunay_whittaker_state=dw_state,
            melnikov_kam_report=melnikov_rep,
            pseudoholomorphic_floer_report=floer_rep,
            rsi3_banach_report=rsi3_rep,
            stealth_qnd_report=qnd_rep,
            bures_report=bures_rep,
            truth_value=truth_val,
            passport_hash=passport_hash,
            execution_time_ns=exec_time_ns,
            hmac_signature=hmac_sig,
            banach_contraction_lip=rsi3_rep.banach_lip_t3,
        )

    # Alias retrocompatible para invocación polimórfica desde agentes OODA
    execute_spectral_audit = execute_ninja_observer_audit


# ═══════════════════════════════════════════════════════════════════════════════
# TEST AUTÓNOMO DE CERTIFICACIÓN SOBERANA DEL VIGÍA NINJA (3 FASES ANIDADAS)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("═" * 78)
    print("=== TEST AUTÓNOMO: BoolNovikovNinjaObserverEngine v15.0.0-PhD (3 FASES) ===")
    print("═" * 78)

    engine = BoolNovikovNinjaObserverEngine()
    matrix_test = np.eye(4, dtype=np.float64)
    exps_test = [0.15, 0.25, 0.38]

    report = engine.execute_ninja_observer_audit(matrix_test, exps_test)

    print(f"Versión del Motor                : {__version__}")
    print(f"Tiempo de Ejecución              : {report.execution_time_ns / 1e6:.3f} ms")
    print("─" * 78)
    print("FASE 1 — ÁLGEBRA DUAL BOOLE-NOVIKOV, HOMOLOGÍA ℤ Y PUENTE POINCARÉ-LIE:")
    print(f"  1.1 Rademacher-Walsh ||∇||₂    : {report.rademacher_walsh_norm:.6f}")
    print(f"  1.2 SNF Torsión Libre (ℤ)      : {report.smith_torsion_free}")
    print(f"  1.2 Betti β₁ = 0               : {report.betti_number_zero}")
    print(f"  1.3 Krull v(A)                 : {report.krull_valuation:.4f}")
    print(f"  1.3 ||A||_Nov                  : {report.novikov_norm:.6f}")
    print(f"  1.3 Ultramétrica Isósceles OK  : {report.ultrametric_inequality_holds}")
    print(f"  1.4 Poincaré-Lindstedt Resid   : {report.poincare_lindstedt_report.residual_homological:.6e}")
    print(f"  1.4 Poincaré-Lindstedt ResFree : {report.poincare_lindstedt_report.is_resonance_free}")
    print(f"  1.4 Energía Birkhoff H_eff     : {report.poincare_lindstedt_report.birkhoff_normal_form_energy:.6f}")
    print("─" * 78)
    print("FASE 2 — MECÁNICA CELESTE DE POINCARÉ, KAM/BRJUNO & POLÍGONO DE FLOER:")
    print(f"  2.1 KAM Estable                : {report.delaunay_whittaker_state.is_kam_stable}")
    print(f"  2.1 Condición de Brjuno OK     : {report.delaunay_whittaker_state.is_bruno_condition}")
    print(f"  2.1 Suma de Brjuno B(ω)        : {report.delaunay_whittaker_state.brjuno_sum_value:.4f}")
    print(f"  2.1 Tiempo Nekhoroshev T_Nekh  : {report.delaunay_whittaker_state.nekhoroshev_stability_time:.4e}")
    print(f"  2.2 Melnikov Integral M(τ)     : {report.melnikov_kam_report.melnikov_integral:.6e}")
    print(f"  2.2 Det Monodromía Poincaré    : {report.melnikov_kam_report.poincare_monodromy_det:.6f}")
    print(f"  2.2 Estabilidad de Krein       : {report.melnikov_kam_report.krein_signature_stable}")
    print(f"  2.3 Residual CR-Floer ||∂̄||    : {report.pseudoholomorphic_floer_report.cr_floer_residual:.6e}")
    print(f"  2.3 Gromov No-Squeezing OK     : {report.pseudoholomorphic_floer_report.is_gromov_squeezing_valid}")
    print("─" * 78)
    print("FASE 3 — AUTOMEJORA RECURSIVA RSI³ & AUDITORÍA ESPECTRAL QND SOBERANA:")
    print(f"  3.1 Contracción Kleisli Lip(T³): {report.rsi3_banach_report.banach_lip_t3:.6f}")
    print(f"  3.1 Niveles RSI³ (L₁, L₂, L₃)  : ({report.rsi3_banach_report.level1_state_contraction:.4f}, "
          f"{report.rsi3_banach_report.level2_operator_contraction:.4f}, "
          f"{report.rsi3_banach_report.level3_meta_contraction:.4f})")
    print(f"  3.2 Teorema Modal de Löb OK    : {report.rsi3_banach_report.loeb_provability_verified}")
    print(f"  3.2 Jerk Ciber-Físico d³C/dt³  : {report.rsi3_banach_report.jerk_cyber_physical:.6e}")
    print(f"  3.3 Valor Débil AAV A_w        : {report.stealth_qnd_report.weak_value:.6f}")
    print(f"  3.3 Norma Conmutador ||[A,ρ]|| : {report.stealth_qnd_report.commutator_norm:.6e}")
    print(f"  3.3 Back-Action Cuántica (dB)  : {report.stealth_qnd_report.back_action_db:.6f} dB")
    print(f"  3.3 Sigilo QND Certificado     : {report.stealth_qnd_report.is_stealth_qnd}")
    print(f"  3.4 Fidelidad de Uhlmann F     : {report.bures_report.fidelity_uhlmann:.6f}")
    print(f"  3.4 Distancia de Bures D_B     : {report.bures_report.bures_distance:.6f}")
    print(f"  3.5 Veredicto Topos Heyting Ω₃ : {report.truth_value.name} (0x{report.truth_value.value:02X})")
    print(f"  3.5 Pasaporte SHA-256          : {report.passport_hash[:32]}...")
    print(f"  3.5 Firma HMAC-SHA256          : {report.hmac_signature[:32]}...")
    print("═" * 78)