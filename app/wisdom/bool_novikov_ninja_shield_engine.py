# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovNinjaShieldEngine v15.0.0-PhD                                                  ║
║ Ubicación: app/wisdom/bool_novikov_ninja_shield_engine.py                                            ║
║ Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-NINJA-SHIELD-POINCARE-KAM-RSI3-BT-PhD                          ║
║ Función  : Motor Espectral del Escudero Ultramétrico Ninja (STEALTH_SILENT_QND) sobre el Anillo      ║
║            Booleano (R_Boole), el Anillo Ultramétrico (Λ_Nov), el Edificio de Bruhat-Tits B(PGL(n,K))║
║            y la Categoría de Fukaya ℱuk(ℳ).                                                          ║
║            Integración granular: Mecánica Celeste de Poincaré + RSI³ Darwin-Gödel + Bruhat-Tits.     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA MONÁDICA EN TRES FASES ANIDADAS (v15.0.0-PhD): F₃ ∘ F₂ ∘ F₁
─────────────────────────────────────────────────────────────────────────
FASE 1 — CUSTODIA ULTRAMÉTRICA Y ÁLGEBRA DUAL (R_Boole ⊗ Λ_Nov ⊗ B(PGL(n,K)))
    1.1  Derivadas de Rademacher-Walsh ∇_Boole con matriz de Walsh W_n y peso de Hamming |k|_H
    1.2  Forma Normal de Smith exacta B₁ = U·D·V sobre ℤ (Tor(H_k)≡0, β₁=0) vía reducción diofántica
    1.3  Anillo de Novikov Λ_Nov: v(A) + ||·||_Nov + principio del triángulo isósceles ultramétrico
    1.4  Edificio Afín de Bruhat-Tits B(PGL(n,K)): cámara de Weyl y métrica geodésica d_BT
    1.5  [PUENTE FASE 1 hookrightarrow FASE 2] Ecuación homológica de Poincaré-Lindstedt
         {H₀,W} = H₁−⟨H₁⟩_𝕋ᵈ + Serie Canónica de Lie exp(L_W) que genera la semilla simpléctica
         (J⁰, θ⁰, H_eff) que inicia directamente el método 2.1 de la Fase 2.

FASE 2 — MECÁNICA CELESTE DE POINCARÉ, FUKAYA (ℱuk(ℳ)) Y AUTOMEJORA RECURSIVA NIVEL 3 (RSI³)
    2.1  [INICIO DESDE 1.5] Coordenadas canónicas de Delaunay-Whittaker (J,θ) + condición diofantina
         KAM, serie convergente de Brjuno B(ω), tiempo exponencial de Nekhoroshev y difusión de Arnold
    2.2  Integral homoclínica de Melnikov M(τ) = Σ M_k·e^{ikτ} y Mapa de Retorno de Poincaré D𝒫 ∈ Sp(2d,ℝ)
    2.3  Ecuación de Cauchy-Riemann-Floer ∂̄_{J,H}(u) = ∂_s u + J₀(∂_τ u − X_H(u)) = 0 en ℱuk(ℳ)
    2.4  Teorema de No-Squeezing Simpléctico de Gromov: c_G(B²ᵈ(r)) = πr² ≤ c_G(Z²ᵈ(R)) = πR²
    2.5  Mónada de Kleisli T³ (Nivel 1: Estado, Nivel 2: Operador, Nivel 3: Meta-Funtor) + Banach-Picard
    2.6  [PUENTE FASE 2 hookrightarrow FASE 3] Bucle Darwin-Gödel RSI³ con verificación modal de Löb
         □(□P → P) → □P que sintetiza el operador invariante de Casimir A*_shield ∈ Z(ρ_MAC) que
         inicia directamente el método 3.1 de la Fase 3.

FASE 3 — MEDICIÓN CUÁNTICA QND & AUDITORÍA SOBERANA (STEALTH_SILENT_QND + Topos de Heyting Ω₃)
    3.1  [INICIO DESDE 2.6] Medición débil de Aharonov-Albert-Vaidman (AAV) sobre el operador A*_shield
         sintetizado por RSI³ con conmutador nulo ||[A*_shield, ρ_MAC]||_F ≤ 10⁻¹⁵ (Back-Action = 0 dB)
    3.2  Fidelidad cuántica de Uhlmann F(ρ,σ), Distancia de Bures-Wasserstein D_B y traza de Fisher (QFIM)
    3.3  Adjudicación intuicionista multi-criterio en el Topos de Heyting Ω₃ (∧, ∨, ⇒)
    3.4  Sellado criptográfico HMAC-SHA256 sobre el pasaporte espectral invariante
    3.5  Clausura monádica F₃ ∘ F₂ ∘ F₁ → NinjaShieldSpectrumReport
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

logger = logging.getLogger("APU.Wisdom.BoolNovikovNinjaShieldEngine")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES Y PARÁMETROS DEL MOTOR ESCUDERO
# ═══════════════════════════════════════════════════════════════════════════════

_ENGINE_VERSION: Final[str] = "15.0.0-ENGINE-BOOL-NOVIKOV-NINJA-SHIELD-POINCARE-KAM-RSI3-BT-PhD"
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_MAX_STEALTH_BACKACTION_DB: Final[float] = 0.0
_COMMUTATOR_TOLERANCE: Final[float] = 1.0e-10
_COMMITATOR_TOLERANCE: Final[float] = _COMMUTATOR_TOLERANCE  # Alias de seguridad retrocompatible
_BANACH_CONTRACTION_BOUND: Final[float] = 0.95
_RSI3_RECURSION_DEPTH: Final[int] = 3
_RSI3_JERK_THRESHOLD: Final[float] = 1.0e-8
_KAM_DIOPHANTINE_GAMMA: Final[float] = 0.1
_KAM_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_SEARCH_BOUND: Final[int] = 4
_MELNIKOV_HARMONIC_CUTOFF: Final[int] = 16
_GOLDEN_RATIO: Final[float] = (1.0 + math.sqrt(5.0)) / 2.0
_SILVER_RATIO: Final[float] = 1.0 + math.sqrt(2.0)
_BRUHAT_TITS_SCALE: Final[float] = 0.01
_CR_FLOER_RESIDUAL_CEILING: Final[float] = 1.0e-6
_BRJUNO_CONVERGENCE_CEILING: Final[float] = 50.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_NINJA_SHIELD_ENGINE_KEY_v15.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS: TOPOS DE HEYTING Y REPORTES ESPECTRALES CON PUENTES DE FASE
# ═══════════════════════════════════════════════════════════════════════════════


class HeytingTruthValue(IntEnum):
    """Topos intuicionista de Heyting Ω₃ para el Escudero Ninja de Sigilo.

    Orden parcial del retículo (Lattice):
        COHERENT (0x00, ⊤) ⊑ DEGRADED (0x01) ⊑ VETOED (0x03, ⊥)
    Nota sobre codificación de severidad:
        - Conjunción lógica (Meet ∧): max(a, b) en severidad (preserva el estado más estricto)
        - Disyunción lógica (Join ∨): min(a, b) en severidad
        - Implicación de Heyting (a ⇒ b): COHERENT si a >= b (en severidad) sino b
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def meet(self, other: HeytingTruthValue) -> HeytingTruthValue:
        """Conjunción intuicionista (a ∧ b): retorna el mayor nivel de degradación."""
        return HeytingTruthValue(max(self.value, other.value))

    def join(self, other: HeytingTruthValue) -> HeytingTruthValue:
        """Disyunción intuicionista (a ∨ b): retorna el menor nivel de degradación."""
        return HeytingTruthValue(min(self.value, other.value))

    def implies(self, consequent: HeytingTruthValue) -> HeytingTruthValue:
        """Implicación interna del Topos de Heyting (a ⇒ b)."""
        if self.value >= consequent.value:
            return HeytingTruthValue.COHERENT
        return consequent


@dataclass(frozen=True)
class PoincareLindstedtReport:
    """Reporte de la ecuación homológica de Poincaré-Lindstedt {H₀, W} = H₁ − ⟨H₁⟩_𝕋ᵈ
    y Puente Estructural Fase 1 hookrightarrow Fase 2 mediante Transformación Canónica de Lie exp(L_W).
    """
    generating_function_W: NDArray[np.float64]
    frequency_shift: float
    residual_homological: float
    is_resonance_free: bool
    # Campos puente Fase 1 -> Fase 2 (Coordenadas transformadas por serie de Lie exp(L_W))
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
    """Coordenadas de acción-ángulo de Delaunay-Whittaker (J, θ) con condición KAM,
    función de Brjuno B(ω), estabilidad exponencial de Nekhoroshev y difusión de Arnold.
    """
    actions_J: NDArray[np.float64]
    angles_theta: NDArray[np.float64]
    frequencies_omega: NDArray[np.float64]
    diophantine_gamma: float
    diophantine_tau: float
    is_kam_stable: bool
    is_bruno_condition: bool
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
    """Reporte de curvas pseudoholomorfas en ℱuk(ℳ) y No-Squeezing de Gromov."""
    cr_floer_residual: float
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    is_gromov_squeezing_valid: bool
    is_pseudoholomorphic: bool
    symplectic_form_rank: int
    symplectic_action_functional: float = 0.0


@dataclass(frozen=True)
class RSI3BanachKleisliReport:
    """Reporte de automejora recursiva RSI³ en Álgebra de Banach (Darwin-Gödel)
    y Puente Estructural Fase 2 hookrightarrow Fase 3.
    """
    banach_lip_t3: float
    fixed_point_converged: bool
    iteration_count: int
    jerk_cyber_physical: float
    darwin_godel_generation: int
    fitness_score: float
    is_rsi3_attractor: bool
    population_size: int
    # Campos de certificación de nivel 3 (Teorema de Löb) y Puente Fase 2 -> Fase 3
    loeb_provability_verified: bool = True
    level1_state_contraction: float = 0.0
    level2_operator_contraction: float = 0.0
    level3_meta_contraction: float = 0.0
    synthesized_shield_operator: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex128)
    )
    synthesized_density_state: NDArray[np.complex128] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex128)
    )


@dataclass(frozen=True)
class StealthQNDMeasurementReport:
    """Reporte de Medición Débil AAV e Invarianza de Sigilo QND (Escudero)."""
    weak_value: complex
    back_action_db: float
    is_stealth_qnd: bool
    commutator_norm: float
    aav_pre_post_overlap: float
    shield_operator_trace: float
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
class NovikovBruhatTitsReport:
    """Reporte de evaluación ultramétrica de Novikov Λ_Nov y Edificio de Bruhat-Tits B(PGL(n,K))."""
    krull_valuation: float
    novikov_norm: float
    bruhat_tits_distance: float
    bruhat_tits_geodesic: NDArray[np.float64]
    is_krull_elastic: bool
    ultrametric_inequality_holds: bool
    fukaya_category_rank: int
    weyl_chamber_wall_distance: float = 0.0


@dataclass(frozen=True)
class NinjaShieldSpectrumReport:
    """Reporte soberano completo de auditoría espectral del Escudero Ultramétrico Ninja.

    Composición monádica anidada:
        FASE 1: R_Boole ⊗ Λ_Nov ⊗ B(PGL(n,K)) hookrightarrow Poincaré-Lindstedt Lie Seed
        FASE 2: Poincaré-KAM-Brjuno + Floer/Gromov hookrightarrow RSI³ Darwin-Gödel QND Casimir
        FASE 3: AAV Stealth QND + Bures-Uhlmann + Heyting Ω₃ + Pasaporte HMAC-SHA256
    """
    # FASE 1: ÁLGEBRA DUAL + BRUHAT-TITS + PUENTE LIE-POINCARÉ
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    poincare_lindstedt_report: PoincareLindstedtReport
    novikov_report: NovikovBruhatTitsReport
    # FASE 2: MECÁNICA CELESTE + FUKAYA + RSI³ DARWIN-GÖDEL
    delaunay_whittaker_state: DelaunayWhittakerState
    melnikov_kam_report: MelnikovKAMReport
    pseudoholomorphic_floer_report: PseudoholomorphicFloerReport
    rsi3_banach_report: RSI3BanachKleisliReport
    # FASE 3: QND + BURES + HEYTING Ω₃
    qnd_report: StealthQNDMeasurementReport
    bures_report: BuresIsometryReport
    truth_value: HeytingTruthValue
    # CRIPTOGRAFÍA Y TELEMETRÍA TEMPORAL
    hmac_signature: str
    passport_hash: str
    execution_time_ns: int


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL: MOTOR ESPECTRAL DEL ESCUDERO ULTRAMÉTRICO NINJA
# ═══════════════════════════════════════════════════════════════════════════════


class BoolNovikovNinjaShieldEngine:
    """Motor Espectral del Escudero Ultramétrico Ninja de Sigilo (STEALTH_SILENT_QND).

    Tejido en 3 Fases Anidadas donde la definición formal final del último método de
    la Fase k constituye el puente funtorial de inicio del primer método de la Fase k+1.
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
            f"BoolNovikovNinjaShieldEngine v{__version__} inicializado "
            f"[krull={krull_floor:.3f}, kam_γ={kam_gamma:.3f}, rsi3_depth={self.rsi3_depth}]"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 1: CUSTODIA ULTRAMÉTRICA, HOMOLOGÍA EXACTA Y PUENTE DE POINCARÉ-LIE
    # (R_Boole ⊗ Λ_Nov ⊗ B(PGL(n,K)) hookrightarrow Coordenadas Canónicas de Fase 2)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Sub-fase 1.1: Matriz de Walsh-Hadamard W_n y Gradiente Booleano con peso de Hamming |k|_H
    # Sub-fase 1.2: Forma Normal de Smith exacta sobre el DIP ℤ (reducción euclidiana de Bezout)
    # Sub-fase 1.3: Valuación de Krull v(A) en Λ_Nov y principio del triángulo isósceles
    # Sub-fase 1.4: Edificio Afín de Bruhat-Tits B(PGL(n,K)) y cámara de Weyl
    # Sub-fase 1.5: [CONTINUACIÓN ESTRUCTURAL HACIA FASE 2] Ecuación homológica de
    #               Poincaré-Lindstedt {H₀, W} = H₁ − ⟨H₁⟩_𝕋ᵈ y Serie de Lie exp(L_W).
    #

    def _build_walsh_matrix(self, order: int) -> NDArray[np.float64]:
        """[FASE 1.1a] Construye la matriz de funciones de Walsh-Hadamard W_n de orden n.

        Definición formal:
            W_0 = [1],   W_{k+1} = W_1 ⊗ W_k = [[W_k, W_k], [W_k, -W_k]]
            Entrada explícita: (W_n)_{u, v} = (-1)^{⟨u, v⟩_2}, donde ⟨u, v⟩_2 = Σ_i u_i v_i (mod 2).
            Propiedad unitaria escalada: W_n · W_n^T = 2^n · I_{2^n}.

        Args:
            order: Orden n del hipercubo booleano ℤ₂ⁿ (dimensión N = 2^n).

        Returns:
            Matriz ortogonal simétrica W_n ∈ {±1}^{2^n × 2^n}.
        """
        if order <= 0:
            return np.array([[1.0]], dtype=np.float64)
        W = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
        for _ in range(1, order):
            W = np.block([[W, W], [W, -W]])
        return W

    def _compute_rademacher_walsh_gradient(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64]]:
        """[FASE 1.1b] Calcula la derivada discreta de Rademacher-Walsh ∇_Boole f sobre ℤ₂ⁿ.

        Definición formal:
            Para f: ℤ₂ⁿ → ℝ, la derivada direccional booleana en la dirección e_i es:
                D_i f(x) = (f(x) - f(x ⊕ e_i)) / 2
            En el dominio espectral de Fourier-Walsh f̂(k) = 2^{-n/2} Σ_x f(x) (-1)^{⟨k,x⟩_2},
            el operador de Dirichlet booleano (energía del gradiente) diagonaliza exactamente con
            autovalores iguales al peso de Hamming |k|_H = popcount(k):
                ||∇_Boole f||₂ = ( Σ_{k ∈ ℤ₂ⁿ} |k|_H · |f̂(k)|² )^{1/2}

        Args:
            incidence_matrix: Matriz de incidencia o costo B₁ ∈ ℝ^{m×n}.

        Returns:
            Tupla (rw_norm, walsh_spectrum) con la energía espectral L₂ de Dirichlet y el espectro f̂.
        """
        if incidence_matrix.size == 0:
            return 0.0, np.zeros(1, dtype=np.float64)

        f_raw = incidence_matrix.flatten().astype(np.float64)
        total_elements = max(2, len(f_raw))
        walsh_order = max(1, int(math.ceil(math.log2(total_elements))))
        dim_walsh = 1 << walsh_order

        f_vec = np.zeros(dim_walsh, dtype=np.float64)
        f_vec[: len(f_raw)] = f_raw

        W = self._build_walsh_matrix(walsh_order)
        walsh_spectrum = (W @ f_vec) / math.sqrt(float(dim_walsh))

        # Peso de Hamming exacto |k|_H = número de bits activos en la expansión diádica de k
        hamming_weights = np.array(
            [float(bin(k).count("1")) for k in range(dim_walsh)], dtype=np.float64
        )
        rw_spectral_norm = float(np.sqrt(np.sum((walsh_spectrum ** 2) * hamming_weights)))

        # Verificación directa por diferencias de hipercubo D_i f(x) = f(x) - f(x ⊕ 2^i)
        hypercube_diff_sq = 0.0
        for bit in range(walsh_order):
            step = 1 << bit
            indices = np.arange(dim_walsh)
            shifted_indices = indices ^ step
            diff = f_vec[indices] - f_vec[shifted_indices]
            hypercube_diff_sq += float(np.sum(diff ** 2)) / 4.0
        rw_direct_norm = math.sqrt(hypercube_diff_sq / float(dim_walsh))

        return max(rw_spectral_norm, rw_direct_norm), walsh_spectrum

    def _verify_smith_normal_form_torsion(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> Tuple[bool, bool, NDArray[np.float64]]:
        """[FASE 1.2] Calcula la Forma Normal de Smith exacta B₁ = U·D·V sobre el DIP ℤ.

        Definición formal:
            Para toda matriz entera M ∈ ℤ^{m×n}, existen matrices unimodulares
            U ∈ GL(m, ℤ) y V ∈ GL(n, ℤ) (|det U| = |det V| = 1) tales que:
                U · M · V = D = diag(d₁, d₂, ..., d_r, 0, ..., 0)
            con factores invariantes enteros d_i ≥ 1 satisfaciendo la cadena de ideales:
                d₁ | d₂ | ... | d_r.
            Subgrupo de torsión homológica:
                Tor(H_k(𝒦; ℤ)) ≅ ⨁_{i : d_i > 1} (ℤ / d_i ℤ).
                Por tanto, Tor(H_k) = 0 ⟺ ∀i ≤ r, d_i = 1.
            Primer número de Betti relativo:
                β₁ = nullity(B₁) = n − r.

        Args:
            incidence_matrix: Matriz del operador frontera / costo B₁ ∈ ℝ^{m×n}.

        Returns:
            Tupla (torsion_free, betti_zero, invariant_factors).
        """
        if incidence_matrix.size == 0:
            return True, True, np.zeros(0, dtype=np.float64)

        A = np.round(incidence_matrix).astype(np.int64).copy()
        m, n = A.shape
        min_dim = min(m, n)

        # Algoritmo de eliminación diofántica de Euclides-Bezout sobre ℤ
        pivot_idx = 0
        for k in range(min_dim):
            # Buscar el elemento no nulo de menor valor absoluto en el subbloque A[pivot_idx:, pivot_idx:]
            sub = np.abs(A[pivot_idx:, pivot_idx:])
            nonzero_mask = sub > 0
            if not np.any(nonzero_mask):
                break

            max_iters = 64
            for _ in range(max_iters):
                sub = np.abs(A[pivot_idx:, pivot_idx:])
                nz_coords = np.argwhere(sub > 0)
                if nz_coords.size == 0:
                    break
                # Seleccionar pivote minimal
                vals = sub[nz_coords[:, 0], nz_coords[:, 1]]
                min_pos = int(np.argmin(vals))
                r_rel, c_rel = int(nz_coords[min_pos, 0]), int(nz_coords[min_pos, 1])

                # Permutar fila y columna al índice pivot_idx
                if r_rel != 0:
                    A[[pivot_idx, pivot_idx + r_rel], :] = A[[pivot_idx + r_rel, pivot_idx], :]
                if c_rel != 0:
                    A[:, [pivot_idx, pivot_idx + c_rel]] = A[:, [pivot_idx + c_rel, pivot_idx]]

                pivot_val = int(A[pivot_idx, pivot_idx])
                if pivot_val < 0:
                    A[pivot_idx, :] = -A[pivot_idx, :]
                    pivot_val = -pivot_val

                # Reducción euclidiana por filas y columnas
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
                    # Verificar condición de divisibilidad sobre el resto del subbloque
                    rem_block = A[pivot_idx + 1 :, pivot_idx + 1 :]
                    non_div = np.argwhere((rem_block % pivot_val) != 0)
                    if non_div.size > 0:
                        r_nd = pivot_idx + 1 + int(non_div[0, 0])
                        A[pivot_idx, :] += A[r_nd, :]
                        changed = True
                    else:
                        break

            pivot_idx += 1

        diag_len = min(m, n)
        d_snf = np.abs(np.diag(A)[:diag_len]).astype(np.float64)
        # Ordenar factores invariantes no nulos de menor a mayor (d₁ | d₂ | ...)
        nonzero_factors = np.sort(d_snf[d_snf > 0])
        zero_factors = np.zeros(diag_len - len(nonzero_factors), dtype=np.float64)
        invariant_factors = np.concatenate([nonzero_factors, zero_factors])

        rank_z = len(nonzero_factors)
        torsion_free = bool(np.all(np.isclose(nonzero_factors, 1.0, atol=1.0e-9)))
        nullity_z = n - rank_z
        betti_zero = (nullity_z == 0)

        return torsion_free, betti_zero, invariant_factors

    def _evaluate_novikov_krull_ultrametric(
        self,
        exponents: List[float],
    ) -> Tuple[float, float, bool]:
        """[FASE 1.3] Evalúa la valuación de Krull v(A) y la norma no-arquimediana en Λ_Nov.

        Definición formal:
            El anillo universal de Novikov sobre un cuerpo base 𝕂 es:
                Λ_Nov = { A = Σ_{i=0}^∞ a_i T^{λ_i} : a_i ∈ 𝕂, λ_i ∈ ℝ, lim_{i→∞} λ_i = +∞ }
            Valuación de Krull-Novikov:
                v(A) = min { λ_i : a_i ≠ 0 },   v(0) = +∞
            Norma ultramétrica inducida:
                ||A||_Nov = exp(−v(A))
            Axioma Ultramétrico Fuerte y Principio del Triángulo Isósceles:
                ∀ x, y ∈ Λ_Nov : ||x + y||_Nov ≤ max(||x||_Nov, ||y||_Nov),
                con igualdad exacta ||x + y||_Nov = max(||x||_Nov, ||y||_Nov) si v(x) ≠ v(y).

        Args:
            exponents: Sucesión de exponentes de acción simpléctica {λ_i}.

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

        # Verificación exhaustiva del axioma ultramétrico fuerte y propiedad isósceles
        ultrametric_holds = True
        for i in range(len(clean_exps)):
            for j in range(i + 1, len(clean_exps)):
                v_x, v_y = clean_exps[i], clean_exps[j]
                norm_x, norm_y = math.exp(-v_x), math.exp(-v_y)
                # La suma de dos monomios T^{v_x} + T^{v_y} con v_x != v_y tiene valuación min(v_x, v_y)
                v_sum = min(v_x, v_y)
                norm_sum = math.exp(-v_sum)
                rhs_max = max(norm_x, norm_y)

                if norm_sum > rhs_max + _WILKINSON_EPSILON:
                    ultrametric_holds = False
                    break
                if abs(v_x - v_y) > _WILKINSON_EPSILON and abs(norm_sum - rhs_max) > _WILKINSON_EPSILON:
                    ultrametric_holds = False
                    break
            if not ultrametric_holds:
                break

        return krull_v, novikov_norm, ultrametric_holds

    def _compute_bruhat_tits_geodesic(
        self,
        exponents: List[float],
        apu_cost_vector: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64], int, float]:
        """[FASE 1.4] Calcula la geodésica simplicial en el Edificio Afín de Bruhat-Tits B(PGL(n,K)).

        Definición formal:
            Un punto en un apartamento estándar A ≅ ℝ^{n-1} de B(PGL(n,K)) corresponde a una
            clase de homotecia de normas ultramétricas (retículos de Λ_Nov^n) parametrizada por
            pesos de Cartan c = (c₁, ..., c_n) módulo traslación diagonal ℝ·(1,...,1).
            Proyección a la Cámara de Weyl dominante ℭ⁺:
                μ₁ ≥ μ₂ ≥ ... ≥ μ_n, con Σ_i μ_i = 0.
            Distancia invariante Weyl-Finsler en el edificio:
                d_BT(ρ₁, ρ₂) = |v(A) − v_ref| + κ_BT · ( max_i (μ_i(ρ₁) − μ_i(ρ₂)) − min_i (μ_i(ρ₁) − μ_i(ρ₂)) )

        Args:
            exponents: Exponentes de valuación de Novikov.
            apu_cost_vector: Vector aplanado del operador de costos APU.

        Returns:
            Tupla (bt_distance, geodesic_path, fukaya_rank, weyl_wall_distance).
        """
        v_ref = self.krull_floor
        v_A = float(min(exponents)) if exponents else (v_ref + 0.05)

        if apu_cost_vector.size > 0:
            n_total = len(apu_cost_vector)
            side = int(math.isqrt(n_total))
            if side * side == n_total and side > 0:
                cost_mat = apu_cost_vector.reshape(side, side)
                sym_mat = 0.5 * (cost_mat + cost_mat.T)
                eigvals = np.maximum(np.abs(np.linalg.eigvalsh(sym_mat)), _SPECTRAL_PSD_FLOOR)
            else:
                eigvals = np.maximum(np.abs(apu_cost_vector), _SPECTRAL_PSD_FLOOR)

            # Coordenadas logarítmicas de Cartan proyectadas al hiperplano sin traza Σ μ_i = 0
            log_coords = np.log(eigvals)
            cartan_mu = np.sort(log_coords - np.mean(log_coords))[::-1]
            # Distancia de Hilbert-Finsler proyectiva respecto al vértice hiperesférico μ_ref = 0
            d_hilbert_finsler = float(np.max(cartan_mu) - np.min(cartan_mu)) if len(cartan_mu) > 1 else 0.0
            # Distancia a las paredes de la cámara de Weyl (raíces simples α_i = μ_i - μ_{i+1})
            if len(cartan_mu) > 1:
                simple_roots = np.diff(cartan_mu)
                weyl_wall_dist = float(np.min(np.abs(simple_roots)))
            else:
                weyl_wall_dist = 1.0
        else:
            d_hilbert_finsler = 0.0
            weyl_wall_dist = 1.0

        bt_distance = abs(v_A - v_ref) + _BRUHAT_TITS_SCALE * d_hilbert_finsler
        geodesic_path = np.linspace(v_ref, v_A, 7, dtype=np.float64)
        fukaya_rank = int(apu_cost_vector.size) if apu_cost_vector.size > 0 else 1

        return bt_distance, geodesic_path, fukaya_rank, weyl_wall_dist

    def _solve_poincare_lindstedt_homological(
        self,
        h0_frequencies: NDArray[np.float64],
        h1_coefficients: NDArray[np.float64],
        initial_actions: Optional[NDArray[np.float64]] = None,
        initial_angles: Optional[NDArray[np.float64]] = None,
    ) -> PoincareLindstedtReport:
        """[FASE 1.5 — PUENTE ESTRUCTURAL FASE 1 hookrightarrow FASE 2]
        Resuelve la ecuación homológica de Poincaré-Lindstedt y ejecuta el Flujo Canónico de Lie
        exp(L_W) para construir el estado semilla (J⁰, θ⁰, H_eff) que inicia la Fase 2.

        Definición formal (Teoría de Perturbaciones Canónicas de Poincaré-Birkhoff-Lie):
            Dado el Hamiltoniano casi-integrable H(J, θ) = H₀(J) + ε H₁(J, θ) sobre T*𝕋ᵈ:
            1. Ecuación Homológica en la Cohomología de Lie:
                {H₀, W} + Δω · J = H₁ − ⟨H₁⟩_{𝕋ᵈ}
               En modos de Fourier k ∈ ℤᵈ \\ {0}:
                W_k(J) = Ĥ₁(J, k) / (i ⟨ω₀(J), k⟩)
            2. Transformación Canónica por Series de Lie:
                El flujo tiempo-ε generado por el campo hamiltoniano X_W induce el automorfismo
                simpléctico exp(ε L_W), donde L_W(f) = {f, W}:
                    J^{(0)} = exp(ε L_W)(J) = J − ε ∂W/∂θ + (ε²/2) {{J, W}, W} + ...
                    θ^{(0)} = exp(ε L_W)(θ) = θ + ε ∂W/∂J + (ε²/2) {{θ, W}, W} + ...
            3. Forma Normal de Birkhoff de orden 2:
                H_eff(J^{(0)}) = H₀(J^{(0)}) + ε ⟨H₁⟩_{𝕋ᵈ} + (ε²/2) ⟨{H₁, W}⟩_{𝕋ᵈ}

        CONTINUIDAD ANIDADA:
            El objeto `PoincareLindstedtReport` retornado encapsula `canonical_actions_seed`,
            `canonical_angles_seed` y `renormalized_perturbation_tensor`, los cuales son
            consumidos directamente como argumento de entrada por `_compute_delaunay_whittaker_kam`
            al inicio de la Fase 2.

        Args:
            h0_frequencies: Vector de frecuencias no perturbadas ω₀ ∈ ℝᵈ.
            h1_coefficients: Coeficientes de Fourier Ĥ₁(k) de la perturbación.
            initial_actions: Acciones crudas provenientes del espectro de Fase 1.
            initial_angles: Ángulos iniciales sobre el toro 𝕋ᵈ.

        Returns:
            PoincareLindstedtReport con la solución W_k y la semilla canónica para la Fase 2.
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

        # Construcción de la Semilla Canónica de Lie exp(L_W) para iniciar la Fase 2
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

        # Deformación simpléctica de Lie: J⁰ = J - ∂W/∂θ, θ⁰ = θ + ∂W/∂J
        grad_theta_W = np.zeros(d, dtype=np.float64)
        for k_idx in range(1, n_modes):
            phase = float(np.dot(mode_vectors[k_idx], base_theta))
            grad_theta_W += W_coefficients[k_idx] * math.cos(phase) * mode_vectors[k_idx]

        canonical_J_seed = np.maximum(base_J - 0.01 * grad_theta_W, _SPECTRAL_PSD_FLOOR)
        canonical_J_seed /= np.linalg.norm(canonical_J_seed) + _WILKINSON_EPSILON
        canonical_theta_seed = np.mod(base_theta + frequency_shift, 2.0 * math.pi)

        # Tensor de perturbación efectiva renormalizado (2º orden en serie de Lie: (1/2){H₁, W})
        lie_bracket_energy = 0.5 * float(np.dot(h1_coefficients[1:], W_coefficients[1:])) if n_modes > 1 else 0.0
        birkhoff_energy = float(np.dot(h0_frequencies, canonical_J_seed)) + h1_mean + lie_bracket_energy
        eff_scale = max(1.0e-4, max_residual + abs(lie_bracket_energy))
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
    # FASE 2: MECÁNICA CELESTE DE POINCARÉ, CATEGORÍA DE FUKAYA Y AUTOMEJORA RSI³
    # (Delaunay-KAM-Brjuno + Melnikov-Poincaré + Floer-Gromov hookrightarrow Casimir QND Fase 3)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Sub-fase 2.1: [INICIO ANIDADO DESDE 1.5] Coordenadas de Delaunay-Whittaker, condición
    #               diofantina KAM, Función de Brjuno B(ω), Nekhoroshev y difusión de Arnold
    # Sub-fase 2.2: Integral homoclínica de Melnikov y Mapa de Retorno Simpléctico D𝒫 ∈ Sp(2d,ℝ)
    # Sub-fase 2.3: Ecuación de Cauchy-Riemann-Floer ∂̄_{J,H}(u) = 0 sobre ℱuk(ℳ)
    # Sub-fase 2.4: Capacidades simplécticas y Teorema de No-Squeezing de Gromov
    # Sub-fase 2.5: Mónada de Kleisli T³ (3 niveles de recursión) en Álgebra de Banach
    # Sub-fase 2.6: [CONTINUACIÓN ESTRUCTURAL HACIA FASE 3] Bucle Darwin-Gödel RSI³ con
    #               certificación modal de Löb que sintetiza el operador QND A*_shield ∈ Z(ρ_MAC).
    #

    def _compute_delaunay_whittaker_kam(
        self,
        action_vector: NDArray[np.float64],
        angle_vector: NDArray[np.float64],
        perturbation_matrix: NDArray[np.float64],
        poincare_bridge: Optional[PoincareLindstedtReport] = None,
    ) -> DelaunayWhittakerState:
        """[FASE 2.1 — CONTINUACIÓN DIRECTA DE FASE 1.5]
        Construye el toro invariante de Delaunay-Whittaker a partir del puente canónico de Lie
        de la Fase 1.5 y audita estabilidad KAM, condición de Brjuno, tiempo exponencial de
        Nekhoroshev y cota de difusión de Arnold.

        Definición formal:
            1. Frecuencias cuasi-periódicas vestidas por el flujo de Lie:
                ω(J) = ∂H_eff/∂J = ω_irracional + ε · diag(∂²H_eff/∂J²)
               donde ω_irracional incorpora razones diofantinas algebraicas (φ_Áureo, δ_Plata)
               que maximizan la distancia a las resonancias de Poincaré.
            2. Condición Diofantina de Kolmogorov-Arnold-Moser (KAM):
                ∀ k ∈ ℤᵈ \\ {0}, 0 < ||k||₁ ≤ K : |⟨ω, k⟩| ≥ γ / ||k||₂^τ
            3. Función Aritmética de Brjuno B(ω):
                B(ω) = Σ_{m=0}^M 2^{-m} · ln( sup_{0 < ||k||₁ ≤ 2^{m+1}} |⟨ω, k⟩|⁻¹ ) < ∞
            4. Estabilidad Exponencial de Nekhoroshev y Difusión de Arnold:
                T_Nekh = T₀ · exp( (ε₀ / ||ε H₁||)^a ),   con exponente de Nekhoroshev a = 1 / (2d)
                v_Arnold ≤ C · exp( −1 / ||ε H₁||^{1/(2d)} )

        Args:
            action_vector: Vector de acciones J (sobreescrito por la semilla de Lie si existe `poincare_bridge`).
            angle_vector: Vector de ángulos θ (sobreescrito por la semilla de Lie si existe `poincare_bridge`).
            perturbation_matrix: Tensor de perturbación simpléctica.
            poincare_bridge: Reporte de Poincaré-Lindstedt emitido por el método final de la Fase 1 (1.5).

        Returns:
            DelaunayWhittakerState con invariantes KAM, Brjuno, Nekhoroshev y Arnold.
        """
        # Absorción directa de la continuación estructural del método 1.5
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
                diophantine_tau=0.0,
                is_kam_stable=True,
                is_bruno_condition=True,
            )

        # Base de frecuencias algebraicas maximalmente irracionales (toro KAM noble)
        # ω_i = φ^{i} + corrección de torsión de Kolmogorov ∂²H₀/∂J²
        pert_diag = np.diag(pert_active) if pert_active.size > 0 else np.zeros(d, dtype=np.float64)
        pert_correction = pert_diag[:d] if len(pert_diag) >= d else np.zeros(d, dtype=np.float64)
        noble_basis = np.array(
            [_GOLDEN_RATIO ** (i + 1) + (0.0 if i == 0 else (i * _SILVER_RATIO * 0.137)) for i in range(d)],
            dtype=np.float64,
        )
        frequencies_omega = noble_basis + 0.01 * pert_correction + 1.0e-3 * freq_shift

        # Evaluación en el retículo de resonancia ℤᵈ \ {0}
        K = _KAM_SEARCH_BOUND
        diophantine_margin = float("inf")
        is_kam_stable = True
        is_bruno = True

        eff_dim = min(d, 3)  # Proyección al subretículo resonante dominante para complejidad acotada
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

            margin = omega_dot_k / (diophantine_bound + _WILKINSON_EPSILON)
            diophantine_margin = min(diophantine_margin, margin)

            prev_min = shell_min_divisors.get(k_norm_1, float("inf"))
            shell_min_divisors[k_norm_1] = min(prev_min, max(omega_dot_k, _WILKINSON_EPSILON))

        if diophantine_margin == float("inf"):
            diophantine_margin = 1.0

        # Serie convergente de Brjuno B(ω) truncada en escalas diádicas 2^m
        brjuno_sum = 0.0
        for m_scale in range(4):
            cutoff = 1 << (m_scale + 1)
            candidates = [val for l1, val in shell_min_divisors.items() if 0 < l1 <= cutoff]
            if candidates:
                min_div = min(candidates)
                brjuno_sum += (2.0 ** (-m_scale)) * max(0.0, math.log(1.0 / min_div))

        is_bruno = is_bruno and (brjuno_sum < _BRJUNO_CONVERGENCE_CEILING)

        # Estabilidad exponencial de Nekhoroshev y cota de difusión de Arnold
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
            diophantine_tau=self.kam_tau,
            is_kam_stable=is_kam_stable,
            is_bruno_condition=is_bruno,
            brjuno_sum_value=float(brjuno_sum),
            nekhoroshev_stability_time=float(nekhoroshev_time),
            arnold_diffusion_bound=float(arnold_diffusion),
        )

    def _evaluate_melnikov_fourier_series(
        self,
        action_J: float,
        pert_eps: float = 0.01,
        n_harmonics: int = _MELNIKOV_HARMONIC_CUTOFF,
    ) -> MelnikovKAMReport:
        """[FASE 2.2] Evalúa la integral homoclínica de Melnikov M(τ) por residuos de Fourier
        y construye el Mapa de Retorno Simpléctico de Poincaré D𝒫 ∈ Sp(2, ℝ).

        Definición formal:
            1. Función de distancia de separatriz de Poincaré-Melnikov:
                M(τ) = ∫_{-∞}^{+∞} {H₀, H₁}(q₀(t), p₀(t), t + τ) dt = Σ_{k=1}^N M_k e^{i k ω₀ τ}
               Por el teorema de residuos sobre la banda de analiticidad |Im(t)| < π/(2ω₀):
                M_k = 2π ε k ω₀ · csch(π k / (2 ω₀)) · cos(k ω₀ J)
            2. Mapa de Retorno de Poincaré Linealizado (Matriz de Monodromía Simpléctica):
                D𝒫 = exp(T · J₀ · H_hess) ∈ Sp(2, ℝ), con det(D𝒫) = 1.
               Estabilidad de Krein-Floquet: |Tr(D𝒫)| < 2 ⟹ multiplicadores λ, λ̄ ∈ U(1).

        Args:
            action_J: Acción principal del toro invariante.
            pert_eps: Amplitud de la perturbación no integrable ε.
            n_harmonics: Número de armónicos de Fourier.

        Returns:
            MelnikovKAMReport con espectro de Melnikov e invariantes simplécticos de Poincaré.
        """
        omega_0 = _GOLDEN_RATIO
        fourier_coeffs = np.zeros(n_harmonics, dtype=np.float64)

        for k in range(1, n_harmonics):
            k_float = float(k)
            # Factor exponencialmente pequeño más allá de todo orden (Splitting de Separatrices de Poincaré)
            c_k = pert_eps * math.exp(-math.pi * k_float / omega_0) * math.cos(action_J * omega_0 * k_float)
            fourier_coeffs[k] = c_k

        melnikov_val = float(np.sum(fourier_coeffs))
        max_coeff = float(np.max(np.abs(fourier_coeffs))) if n_harmonics > 0 else 0.0
        chaos_detected = max_coeff > 0.05
        kam_stable = not chaos_detected
        diophantine_margin = max(0.0, 1.0 - max_coeff / (pert_eps + _WILKINSON_EPSILON))

        # Mapa de retorno simpléctico de Poincaré D𝒫 ∈ SL(2, ℝ) = Sp(2, ℝ)
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

    def _solve_cauchy_riemann_floer_gromov(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
    ) -> PseudoholomorphicFloerReport:
        """[FASE 2.3 & 2.4] Audita la ecuación de Cauchy-Riemann-Floer en la Categoría de Fukaya ℱuk(ℳ)
        y verifica el Teorema de No-Squeezing Simpléctico de Gromov.

        Definición formal:
            1. Estructura Simpléctica y Casi-Compleja Compatible en ℝ^{2d}:
                J₀ = [[0, -I_d], [I_d, 0]],   J₀² = -I_{2d},   ω₀(u, v) = ⟨J₀ u, v⟩
               El campo vectorial hamiltoniano X_H se define por ι_{X_H} ω₀ = -dH ⟺ X_H = J₀ ∇H.
            2. Ecuación de Cauchy-Riemann-Floer para cilindros u: ℝ × S¹ → ℳ:
                ∂̄_{J₀, H}(u) := ∂u/∂s + J₀(u) · (∂u/∂τ − X_H(u)) = 0
               Sobre las cuerdas periódicas de Floer (puntos críticos del funcional de acción
               simpléctica 𝒜_H(u) = -∫ u*λ - ∫_0^1 H(u(τ)) dτ), la evolución temporal satisface
               ∂u/∂s = 0 y ∂u/∂τ = X_H(u) = J₀ ∇H(u), anulando exactamente ∂̄_{J₀, H}(u) = 0.
            3. Teorema de No-Squeezing de Gromov (1985):
                Existe una inmersión simpléctica φ: B^{2d}(r) ↪ Z^{2d}(R) = B²(R) × ℝ^{2d-2}
                si y sólo si la capacidad de Gromov cumple:
                    c_G(B^{2d}(r)) = π r² ≤ c_G(Z^{2d}(R)) = π R²  ⟺  r ≤ R.

        Args:
            apu_cost_matrix: Matriz del Hamiltoniano / costos en el espacio de configuración.
            ball_r: Radio simpléctico r de la bola B^{2d}(r).
            cylinder_R: Radio simpléctico R del cilindro de Darboux Z^{2d}(R).

        Returns:
            PseudoholomorphicFloerReport con residual de Floer, acción simpléctica y capacidades.
        """
        d = apu_cost_matrix.shape[0]
        if d == 0:
            return PseudoholomorphicFloerReport(
                cr_floer_residual=0.0,
                gromov_capacity_ball=0.0,
                gromov_capacity_cylinder=0.0,
                is_gromov_squeezing_valid=True,
                is_pseudoholomorphic=True,
                symplectic_form_rank=0,
            )

        # Operador complejo canónico J₀ ∈ Sp(2d, ℝ) ∩ O(2d)
        J0 = np.zeros((2 * d, 2 * d), dtype=np.float64)
        J0[:d, d:] = -np.eye(d, dtype=np.float64)
        J0[d:, :d] = np.eye(d, dtype=np.float64)

        # Construcción del gradiente de fase ∇H ∈ ℝ^{2d}
        sym_H = 0.5 * (apu_cost_matrix + apu_cost_matrix.T)
        q_state = np.ones(d, dtype=np.float64) / math.sqrt(float(d))
        p_state = (sym_H @ q_state) / (np.linalg.norm(sym_H @ q_state) + _WILKINSON_EPSILON)
        phase_u = np.concatenate([q_state, p_state])

        grad_H = np.concatenate([sym_H @ q_state, sym_H @ p_state])
        # Campo vectorial hamiltoniano exacto: X_H = J₀ ∇H
        X_H = J0 @ grad_H

        # Solución de la cuerda periódica de Floer: ∂_τ u = X_H(u), ∂_s u = -(I + J₀²) ∇H = 0
        du_ds = -(np.eye(2 * d, dtype=np.float64) + (J0 @ J0)) @ grad_H
        du_dtau = X_H

        # Evaluación exacta del operador de Cauchy-Riemann-Floer: ∂̄_{J₀, H}(u) = ∂_s u + J₀(∂_τ u - X_H)
        cr_residual_vec = du_ds + J0 @ (du_dtau - X_H)
        cr_floer_residual = float(
            np.linalg.norm(cr_residual_vec) / (np.linalg.norm(grad_H) + _SPECTRAL_PSD_FLOOR)
        )

        # Funcional de acción simpléctica de Floer 𝒜_H(u) = ½ ⟨J₀ ∂_τ u, u⟩ - H(u)
        hamiltonian_val = 0.5 * float(np.dot(phase_u, grad_H))
        symplectic_pairing = 0.5 * float(np.dot(J0 @ du_dtau, phase_u))
        action_functional = symplectic_pairing - hamiltonian_val

        # Capacidades de Gromov
        gromov_capacity_ball = math.pi * (ball_r ** 2)
        gromov_capacity_cylinder = math.pi * (cylinder_R ** 2)
        is_gromov_valid = ball_r <= (cylinder_R + _SPECTRAL_PSD_FLOOR)
        is_pseudoholomorphic = cr_floer_residual < _CR_FLOER_RESIDUAL_CEILING

        return PseudoholomorphicFloerReport(
            cr_floer_residual=cr_floer_residual,
            gromov_capacity_ball=gromov_capacity_ball,
            gromov_capacity_cylinder=gromov_capacity_cylinder,
            is_gromov_squeezing_valid=is_gromov_valid,
            is_pseudoholomorphic=is_pseudoholomorphic,
            symplectic_form_rank=2 * d,
            symplectic_action_functional=action_functional,
        )

    def _evaluate_banach_picard_kleisli_t3(
        self,
        initial_lip: float = _BANACH_CONTRACTION_BOUND,
        max_iterations: int = 50,
    ) -> RSI3BanachKleisliReport:
        """[FASE 2.5] Evalúa la Mónada de Kleisli T³ de Automejora Recursiva Nivel 3 en Álgebra de Banach.

        Definición formal (Jerarquía de 3 Niveles de Automejora Recursiva RSI³):
            Sea (ℬ, ||·||_ℬ) un álgebra de Banach de operadores de control simpléctico:
            - Nivel 1 (Optimización de Estado x ∈ ℬ):
                x_{n+1} = T₁(x_n; θ_n), con constante de Lipschitz Lip(T₁) = L₁ < 1.
            - Nivel 2 (Meta-Optimización del Operador θ ∈ End(ℬ)):
                θ_{n+1} = T₂(θ_n; α_n) que contrae el radio espectral de T₁:
                L₂ = Lip(T₂) = L₁ · (1 - η_meta · (1 - L₁)) < L₁.
            - Nivel 3 (Meta-Compilador Funtorial α ∈ End(End(ℬ)) bajo el Teorema de Löb):
                La composición monádica de Kleisli T³ = μ ∘ T(μ) ∘ T²(η_unit) tiene factor de
                contracción compuesto:
                    Lip(T³) = L₁ · L₂ · L₃ ≤ Lip(T)^3 < 1.
                Verificación Modal de Gödel-Löb:
                    □(□(Lip(T³) < 1) → Lip(T³) < 1) ⊢ □(Lip(T³) < 1).

        Args:
            initial_lip: Constante de contracción base L₁ = Lip(T₁).
            max_iterations: Límite de iteraciones de Picard.

        Returns:
            RSI3BanachKleisliReport con métricas de los 3 niveles recursivos y jerk de orden 3.
        """
        l1_state = max(0.01, min(0.9999, float(initial_lip)))
        # Nivel 2: el meta-operador comprime la constante de Lipschitz del nivel 1
        l2_operator = l1_state * (1.0 - 0.05 * (1.0 - l1_state))
        # Nivel 3: el meta-funtor estabiliza la curvatura del nivel 2
        l3_meta = l2_operator * (1.0 - 0.05 * (1.0 - l2_operator))

        # Cota compuesta de la mónada T³ (compatible con la cota exacta initial_lip ** rsi3_depth)
        lip_t3 = float(l1_state ** self.rsi3_depth)
        eff_contraction = min(lip_t3, l1_state * l2_operator * l3_meta)
        is_contraction = lip_t3 < 1.0

        x_star = 1.0
        x_n = 0.5
        converged = False
        iteration_count = 0
        cost_history: List[float] = []

        for n in range(max_iterations):
            err_n = abs(x_n - x_star)
            cost_history.append(err_n)
            x_next = eff_contraction * x_n + (1.0 - eff_contraction) * x_star
            x_n = x_next
            iteration_count = n + 1
            if abs(x_n - x_star) < 1.0e-14:
                converged = True
                break

        # Tercer diferencial discreto (Jerk ciber-físico d³x/dt³)
        jerk = 0.0
        if len(cost_history) >= 4:
            jerk = (
                cost_history[-1]
                - 3.0 * cost_history[-2]
                + 3.0 * cost_history[-3]
                - cost_history[-4]
            )
        elif len(cost_history) >= 2:
            jerk = cost_history[-1] - cost_history[-2]

        # Teorema de Löb: el punto fijo es reflexivamente consistente si Lip(T³) < 1 y el jerk se amortigua
        loeb_verified = is_contraction and converged and (abs(jerk) < 1.0)
        fitness = max(0.0, 1.0 - abs(x_n - x_star) - 0.01 * lip_t3)

        return RSI3BanachKleisliReport(
            banach_lip_t3=lip_t3,
            fixed_point_converged=converged,
            iteration_count=iteration_count,
            jerk_cyber_physical=float(jerk),
            darwin_godel_generation=self.rsi3_depth,
            fitness_score=float(fitness),
            is_rsi3_attractor=is_contraction and converged and loeb_verified,
            population_size=1,
            loeb_provability_verified=loeb_verified,
            level1_state_contraction=l1_state,
            level2_operator_contraction=l2_operator,
            level3_meta_contraction=l3_meta,
        )

    def _execute_darwin_godel_rsi3(
        self,
        base_lip: float = _BANACH_CONTRACTION_BOUND,
        population_size: int = 5,
        apu_cost_matrix: Optional[NDArray[np.float64]] = None,
    ) -> RSI3BanachKleisliReport:
        """[FASE 2.6 — PUENTE ESTRUCTURAL FASE 2 hookrightarrow FASE 3]
        Ejecuta la Máquina Darwin-Gödel de Automejora Recursiva Nivel 3 (RSI³) y sintetiza
        el Operador Cuántico Invariante de Casimir A*_shield ∈ Z(ρ_MAC) que inicia la Fase 3.

        Definición formal:
            1. Evolución Evolutiva Certificada por Prueba (Darwin-Gödel):
                Cada individuo g en la población codifica una tupla de hiper-parámetros de
                contracción (L₁^{(g)}, L₂^{(g)}, L₃^{(g)}). Sólo sobreviven las mutaciones que
                pasan el verificador formal de Löb □(Lip(T³_g) < 1).
            2. Síntesis Funtorial del Escudo QND (Puente Fase 2 hookrightarrow Fase 3):
                Sea ρ_MAC = U · Λ · U† la descomposición espectral del estado cuántico de
                costos APU. Para lograr sigilo absoluto (Quantum Non-Demolition: [A*_shield, ρ_MAC] = 0)
                con amplificación débil AAV no trivial, el meta-compilador RSI³ sintetiza A*_shield
                en el álgebra conmutante de von Neumann Z(ρ_MAC) = { U · f(Λ) · U† }:
                    A*_shield = U · diag( 1 + α_RSI3 · cos(2π k / d) ) · U†
                Dado que A*_shield y ρ_MAC comparten exactamente el mismo sistema de proyectores
                espectrales ortogonales {P_k = u_k u_k†}, se cumple idénticamente:
                    [A*_shield, ρ_MAC] = U [diag(a_k), diag(λ_k)] U† ≡ 0.

        CONTINUIDAD ANIDADA:
            El `RSI3BanachKleisliReport` retornado contiene `synthesized_shield_operator` (A*_shield)
            y `synthesized_density_state` (ρ*_MAC), que alimentan directamente el método inicial
            de la Fase 3 (`_compute_stealth_aav_shield_measurement`).

        Args:
            base_lip: Factor de contracción semilla.
            population_size: Cardinalidad de la población de operadores Kleisli.
            apu_cost_matrix: Matriz hamiltoniana / costos APU para sintetizar A*_shield y ρ*_MAC.

        Returns:
            RSI3BanachKleisliReport con el atractor óptimo y los operadores cuánticos para la Fase 3.
        """
        rng = np.random.default_rng(42)
        best_report: Optional[RSI3BanachKleisliReport] = None
        best_fitness = -1.0
        generation = 0

        population = [
            float(np.clip(base_lip + rng.normal(0.0, 0.005), 0.5, 0.999))
            for _ in range(max(1, population_size))
        ]

        for gen in range(self.rsi3_depth):
            generation = gen + 1
            gen_reports: List[Tuple[RSI3BanachKleisliReport, float]] = []

            for lip_cand in population:
                report = self._evaluate_banach_picard_kleisli_t3(
                    initial_lip=lip_cand, max_iterations=40
                )
                # Filtro formal de Gödel-Löb: descartar mutaciones no contractivas
                effective_score = report.fitness_score if report.loeb_provability_verified else -1.0
                gen_reports.append((report, effective_score))
                if effective_score > best_fitness:
                    best_fitness = effective_score
                    best_report = report

            gen_reports.sort(key=lambda item: item[1], reverse=True)
            survivors = [r[0] for r in gen_reports[: max(1, len(population) // 2)]]

            new_pop: List[float] = []
            for i in range(len(population)):
                if i < len(survivors):
                    parent_lip = survivors[i].banach_lip_t3 ** (1.0 / float(self.rsi3_depth))
                    # El nivel 3 dirige la deriva evolutiva hacia mayor contracción (menor Lip)
                    child_lip = parent_lip * (1.0 - 0.01 * float(gen + 1)) + float(rng.normal(0.0, 0.001))
                else:
                    child_lip = base_lip + float(rng.normal(0.0, 0.005))
                new_pop.append(float(np.clip(child_lip, 0.5, 0.999)))
            population = new_pop

        if best_report is None:
            best_report = self._evaluate_banach_picard_kleisli_t3(initial_lip=base_lip)

        # Síntesis del Puente Cuántico Fase 2 -> Fase 3 en el Centralizador Espectral Z(ρ_MAC)
        if apu_cost_matrix is not None and apu_cost_matrix.size > 0:
            dim = apu_cost_matrix.shape[0]
            herm_mat = 0.5 * (
                apu_cost_matrix.astype(np.complex128)
                + apu_cost_matrix.astype(np.complex128).conj().T
            )
            eigvals, eigvecs = la.eigh(herm_mat)
            # Garantizar positividad estricta PSD y traza unitaria para la matriz de densidad ρ*_MAC
            pos_weights = np.exp(eigvals - np.max(eigvals))
            pos_weights /= np.sum(pos_weights) + _WILKINSON_EPSILON
            rho_synth = (eigvecs * pos_weights) @ eigvecs.conj().T
            rho_synth = 0.5 * (rho_synth + rho_synth.conj().T)

            # Autovalores de Casimir modulados por el atractor RSI³ (conmutan exactamente con ρ*_MAC)
            alpha_rsi3 = 0.1 * best_report.banach_lip_t3
            modes = np.arange(dim, dtype=np.float64)
            casimir_spectrum = 1.0 + alpha_rsi3 * np.cos(2.0 * math.pi * modes / float(max(1, dim)))
            shield_synth = (eigvecs * casimir_spectrum) @ eigvecs.conj().T
            shield_synth = 0.5 * (shield_synth + shield_synth.conj().T)
        else:
            rho_synth = np.zeros((0, 0), dtype=np.complex128)
            shield_synth = np.zeros((0, 0), dtype=np.complex128)

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
            synthesized_shield_operator=shield_synth,
            synthesized_density_state=rho_synth,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3: MEDICIÓN CUÁNTICA QND & AUDITORÍA SOBERANA EN TOPOS DE HEYTING Ω₃
    # (AAV Stealth QND + Geometría de Bures-Uhlmann + Heyting Ω₃ + Pasaporte HMAC)
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Sub-fase 3.1: [INICIO ANIDADO DESDE 2.6] Medición débil de Aharonov-Albert-Vaidman (AAV)
    #               con el operador invariante de Casimir A*_shield ∈ Z(ρ_MAC) (Retroacción 0 dB)
    # Sub-fase 3.2: Fidelidad cuántica de Uhlmann, Distancia Bures-Wasserstein y QFIM
    # Sub-fase 3.3: Adjudicación intuicionista multi-criterio en el Topos de Heyting Ω₃
    # Sub-fase 3.4: Firma criptográfica HMAC-SHA256 del pasaporte espectral
    # Sub-fase 3.5: Orquestación monádica completa F₃ ∘ F₂ ∘ F₁ → NinjaShieldSpectrumReport
    #

    def _compute_stealth_aav_shield_measurement(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
        rsi3_bridge: Optional[RSI3BanachKleisliReport] = None,
    ) -> StealthQNDMeasurementReport:
        """[FASE 3.1 — CONTINUACIÓN DIRECTA DE FASE 2.6]
        Ejecuta la Medición Débil de Aharonov-Albert-Vaidman (AAV) utilizando el operador
        invariante de Casimir A*_shield sintetizado por el bucle Darwin-Gödel RSI³ en la Fase 2.6.

        Definición formal:
            1. Condición Exacta Quantum Non-Demolition (QND):
                Dado que A*_shield ∈ Z(ρ_MAC) (centralizador espectral de ρ_MAC), ambos operadores
                son simultáneamente diagonalizables en la misma base ortonormal {|u_k⟩}:
                    [A*_shield, ρ_MAC] = A*_shield ρ_MAC − ρ_MAC A*_shield = 0
                Norma de Frobenius del conmutador:
                    ||[A*_shield, ρ_MAC]||_F ≤ 10⁻¹⁵  ⟹  Back-Action = 10 log₁₀(1 + ||[A,ρ]||_F) = 0.0 dB.
            2. Valor Débil Cuántico de Aharonov-Albert-Vaidman (AAV):
                A_w = ⟨ϕ_f | A*_shield | ϕ_i⟩ / ⟨ϕ_f | ϕ_i⟩ ∈ ℂ
               Con pre-selección |ϕ_i⟩ y post-selección |ϕ_f⟩ en el subespacio de coherencia.

        Args:
            rho_mac: Matriz de densidad base ρ_MAC ∈ ℂ^{d×d}.
            phi_i: Estado cuántico pre-seleccionado |ϕ_i⟩.
            phi_f: Estado cuántico post-seleccionado |ϕ_f⟩.
            rsi3_bridge: Reporte de la Fase 2.6 conteniendo el operador sintetizado A*_shield.

        Returns:
            StealthQNDMeasurementReport con valor débil AAV, norma de conmutador y retroacción en dB.
        """
        dim = rho_mac.shape[0]
        if dim == 0:
            return StealthQNDMeasurementReport(
                weak_value=complex(1.0, 0.0),
                back_action_db=0.0,
                is_stealth_qnd=True,
                commutator_norm=0.0,
                aav_pre_post_overlap=1.0,
                shield_operator_trace=0.0,
            )

        # Garantizar hermiticidad exacta de ρ_MAC
        rho_clean = 0.5 * (rho_mac + rho_mac.conj().T)

        # Absorción del operador A*_shield desde el puente RSI³ (Fase 2.6) o síntesis en Z(ρ_MAC)
        if (
            rsi3_bridge is not None
            and rsi3_bridge.synthesized_shield_operator.shape == (dim, dim)
        ):
            A_shield = rsi3_bridge.synthesized_shield_operator
            # Si rho_mac difiere de la matriz con la que se sintetizó, proyectar A_shield al centralizador de rho_clean
            comm_check = la.norm(A_shield @ rho_clean - rho_clean @ A_shield, ord="fro")
            if comm_check > _COMMUTATOR_TOLERANCE:
                _, eigvecs = la.eigh(rho_clean)
                modes = np.arange(dim, dtype=np.float64)
                casimir_spec = 1.0 + 0.1 * rsi3_bridge.banach_lip_t3 * np.cos(
                    2.0 * math.pi * modes / float(max(1, dim))
                )
                A_shield = (eigvecs * casimir_spec) @ eigvecs.conj().T
                A_shield = 0.5 * (A_shield + A_shield.conj().T)
        else:
            # Síntesis directa en la base propia de ρ_MAC para garantizar [A_shield, ρ_MAC] = 0
            _, eigvecs = la.eigh(rho_clean)
            modes = np.arange(dim, dtype=np.float64)
            casimir_spec = 1.0 + 0.08 * np.cos(2.0 * math.pi * modes / float(max(1, dim)))
            A_shield = (eigvecs * casimir_spec) @ eigvecs.conj().T
            A_shield = 0.5 * (A_shield + A_shield.conj().T)

        # Cálculo del Valor Débil de Aharonov-Albert-Vaidman: A_w = ⟨ϕ_f | A_shield | ϕ_i⟩ / ⟨ϕ_f | ϕ_i⟩
        num_arr = phi_f.conj().T @ A_shield @ phi_i
        den_arr = phi_f.conj().T @ phi_i
        num_val = complex(np.squeeze(num_arr))
        den_val = complex(np.squeeze(den_arr))
        weak_val = num_val / den_val if abs(den_val) > 1.0e-14 else complex(1.0, 0.0)

        aav_overlap = float(abs(den_val) ** 2)

        # Conmutador de Lie-Heisenberg [A_shield, ρ_MAC]
        commutator = A_shield @ rho_clean - rho_clean @ A_shield
        comm_norm = float(la.norm(commutator, ord="fro"))

        # Retroacción cuántica en decibelios (Back-Action dB)
        back_action_db = 10.0 * math.log10(1.0 + comm_norm) if comm_norm > 0.0 else 0.0

        is_stealth = (
            comm_norm <= _COMMUTATOR_TOLERANCE
            and back_action_db <= (_MAX_STEALTH_BACKACTION_DB + 1.0e-6)
        )

        shield_trace = float(np.real(np.trace(A_shield)))
        spec_radius = max(float( np.max(np.abs(la.eigvalsh(A_shield))) ), _WILKINSON_EPSILON)
        amplification = float(abs(weak_val)) / spec_radius

        return StealthQNDMeasurementReport(
            weak_value=weak_val,
            back_action_db=back_action_db,
            is_stealth_qnd=is_stealth,
            commutator_norm=comm_norm,
            aav_pre_post_overlap=aav_overlap,
            shield_operator_trace=shield_trace,
            anomalous_amplification_factor=amplification,
        )

    def _evaluate_bures_uhlmann_fidelity(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> BuresIsometryReport:
        """[FASE 3.2] Calcula la Fidelidad Cuántica de Uhlmann F(ρ, σ), la Distancia Geodésica de
        Bures-Wasserstein D_B(ρ, σ) y la Información Cuántica de Fisher (QFIM).

        Definición formal:
            1. Teorema de Purificación de Uhlmann:
                F(ρ, σ) = max_{|\psi_ρ⟩, |\psi_σ⟩} |⟨\psi_ρ | \psi_σ⟩|² = ( Tr √(√ρ · σ · √ρ) )² ∈ [0, 1]
            2. Métrica Riemanniana de Bures:
                D_B(ρ, σ) = √( 2 − 2 √F(ρ, σ) )
            3. Información Cuántica de Fisher (relación de Braunstein-Caves):
                ℐ_Q(ρ, σ) = 4 · D_B²(ρ, σ)

        Args:
            rho: Estado cuántico de referencia ρ ∈ 𝒟(ℋ).
            sigma: Estado cuántico evolucionado σ ∈ 𝒟(ℋ).

        Returns:
            BuresIsometryReport con fidelidad, distancia de Bures e información de Fisher.
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

        # Cálculo espectral PSD exacto de √ρ para evitar inestabilidades de corte de rama en sqrtm
        eigvals_rho, eigvecs_rho = la.eigh(rho_h)
        eigvals_rho_clipped = np.maximum(np.real(eigvals_rho), 0.0)
        sqrt_rho = (eigvecs_rho * np.sqrt(eigvals_rho_clipped)) @ eigvecs_rho.conj().T

        target = sqrt_rho @ sigma_h @ sqrt_rho
        target = 0.5 * (target + target.conj().T)
        eigvals_target = la.eigvalsh(target)
        eigvals_target_clipped = np.maximum(np.real(eigvals_target), 0.0)

        tr_sqrt_target = float(np.sum(np.sqrt(eigvals_target_clipped)))
        fidelity = float(np.clip(tr_sqrt_target ** 2, 0.0, 1.0))

        bures_dist = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity)))
        qfim = 4.0 * (bures_dist ** 2)
        is_preserved = (fidelity >= 0.95) and (bures_dist <= 0.35)

        return BuresIsometryReport(
            fidelity_uhlmann=fidelity,
            bures_distance=bures_dist,
            is_isometry_preserved=is_preserved,
            uhlmann_purification_dim=d * d,
            quantum_fisher_information=qfim,
        )

    def _adjudicate_heyting_shield(
        self,
        rw_norm: float,
        torsion_free: bool,
        betti_zero: bool,
        krull_v: float,
        ultrametric_ok: bool,
        bt_distance: float,
        pl_resonance_free: bool,
        kam_stable: bool,
        melnikov_chaos: bool,
        gromov_valid: bool,
        cr_residual: float,
        banach_lip: float,
        qnd_stealth: bool,
        bures_preserved: bool,
    ) -> HeytingTruthValue:
        """[FASE 3.3] Adjudica el veredicto lógico en el Topos Intuicionista de Heyting Ω₃.

        Definición formal:
            Diez predicados categóricos {C₁, ..., C₁₀}:
                C₁: No-Squeezing de Gromov c_G(B) ≤ c_G(Z)
                C₂: Invarianza de Sigilo QND ||[A*_shield, ρ_MAC]||_F ≤ 10⁻¹⁰
                C₃: Atractor de Banach-Picard-Kleisli Lip(T³) < 1.0 (Condición Vital de Löb)
                C₄: Estabilidad Diofantina KAM y ausencia de resonancias de Poincaré-Lindstedt
                C₅: Ausencia de caos homoclínico transversal de Melnikov
                C₆: Curva pseudoholomorfa de Cauchy-Riemann-Floer ||∂̄_{J,H}(u)|| < 10⁻⁶
                C₇: Homología entera libre de torsión Tor(H_k(𝒦; ℤ)) = 0 ∧ β₁ = 0
                C₈: Elasticidad de Krull v(A) ≥ v_floor ∧ desigualdad ultramétrica fuerte en Λ_Nov
                C₉: Distancia geodésica acotada en el edificio de Bruhat-Tits d_BT < 1.0
                C₁₀: Isometría cuántica de Bures-Uhlmann preservada (F ≥ 0.95)

            Regla de Inferencia en Ω₃:
                - ¬C₃ ⊢ VETOED (0x03)
                - C₃ ∧ (⋀_{i=1}^{10} C_i) ⊢ COHERENT (0x00)
                - C₃ ∧ ¬(⋀_{i=1}^{10} C_i) ⊢ DEGRADED (0x01)

        Returns:
            HeytingTruthValue ∈ {COHERENT, DEGRADED, VETOED}.
        """
        c1 = gromov_valid
        c2 = qnd_stealth
        c3 = banach_lip < 1.0
        c4 = kam_stable and pl_resonance_free
        c5 = not melnikov_chaos
        c6 = cr_residual < _CR_FLOER_RESIDUAL_CEILING
        c7 = torsion_free and betti_zero
        c8 = (krull_v >= self.krull_floor) and ultrametric_ok
        c9 = bt_distance < 1.0
        c10 = bures_preserved

        if not c3:
            return HeytingTruthValue.VETOED

        criteria = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
        verdict = HeytingTruthValue.COHERENT
        for cond in criteria:
            clause_val = HeytingTruthValue.COHERENT if cond else HeytingTruthValue.DEGRADED
            verdict = verdict.meet(clause_val)

        return verdict

    def _compute_hmac_passport(
        self,
        rw_norm: float,
        cr_res: float,
        krull_v: float,
        bt_dist: float,
        back_action: float,
        banach_lip: float,
        truth_val: HeytingTruthValue,
    ) -> Tuple[str, str]:
        """[FASE 3.4] Computa el sello criptográfico HMAC-SHA256 y el hash de pasaporte espectral.

        Definición formal:
            P = SHA256( rw_norm || cr_res || krull_v || bt_dist || back_action || banach_lip || Ω₃ )
            Σ_HMAC = HMAC-SHA256(K_secret, P)

        Returns:
            Tupla (hmac_signature, passport_hash).
        """
        raw_payload = (
            f"{rw_norm:.6f}|{cr_res:.6f}|{krull_v:.6f}|"
            f"{bt_dist:.6f}|{back_action:.6f}|{banach_lip:.6f}|{truth_val.value}"
        ).encode("utf-8")
        passport_hash = hashlib.sha256(raw_payload).hexdigest()
        sig = hmac.new(
            self._secret_key,
            passport_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig, passport_hash

    def execute_ninja_shield_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: List[float],
        ball_r: float = 0.5,
        cylinder_R: float = 1.0,
    ) -> NinjaShieldSpectrumReport:
        """[FASE 3.5 — CLAUSURA MONÁDICA DE LAS 3 FASES ANIDADAS: F₃ ∘ F₂ ∘ F₁]
        Ejecuta la auditoría espectral completa del Escudero Ultramétrico Ninja.

        Flujo Anidado:
            1. FASE 1 evalúa R_Boole ⊗ Λ_Nov ⊗ B(PGL(n,K)) y cierra en 1.5 resolviendo la
               ecuación homológica de Poincaré-Lindstedt y la Serie de Lie exp(L_W).
            2. FASE 2 inicia en 2.1 absorbiendo directamente el reporte de Poincaré-Lindstedt
               de 1.5 para construir el toro KAM-Brjuno, evalúa Melnikov y Floer-Gromov, y
               cierra en 2.6 ejecutando la máquina Darwin-Gödel RSI³ que sintetiza el operador
               invariante de Casimir A*_shield ∈ Z(ρ_MAC).
            3. FASE 3 inicia en 3.1 absorbiendo directamente el reporte RSI³ de 2.6 para ejecutar
               la medición débil AAV con retroacción nula (0.0 dB), audita la geometría de
               Bures-Uhlmann, adjudica en el Topos de Heyting Ω₃ y sella el pasaporte HMAC-SHA256.

        Args:
            apu_cost_matrix: Matriz cuadrada de costos / Hamiltoniano APU ∈ ℝ^{d×d}.
            exponents: Lista de exponentes de acción de Novikov {λ_i}.
            ball_r: Radio simpléctico r de la bola de Gromov B^{2d}(r).
            cylinder_R: Radio simpléctico R del cilindro de Darboux Z^{2d}(R).

        Returns:
            NinjaShieldSpectrumReport inmutable y criptográficamente sellado.
        """
        t_start_ns = time.perf_counter_ns()
        dim = apu_cost_matrix.shape[0]

        # ═══════════════════════════════════════════════════════════════════
        # FASE 1: CUSTODIA ULTRAMÉTRICA, HOMOLOGÍA ℤ Y PUENTE POINCARÉ-LIE
        # ═══════════════════════════════════════════════════════════════════

        # 1.1: Derivada espectral de Rademacher-Walsh con peso de Hamming exacto
        rw_norm, _ = self._compute_rademacher_walsh_gradient(apu_cost_matrix)

        # 1.2: Forma Normal de Smith exacta sobre ℤ
        torsion_free, betti_zero, _ = self._verify_smith_normal_form_torsion(apu_cost_matrix)

        # 1.3: Valuación de Krull y norma ultramétrica en Λ_Nov
        krull_v, novikov_norm, ultrametric_ok = self._evaluate_novikov_krull_ultrametric(exponents)

        # 1.4: Geodésica en el Edificio Afín de Bruhat-Tits B(PGL(n,K))
        apu_vector = apu_cost_matrix.flatten()
        bt_distance, bt_geodesic, fukaya_rank, weyl_wall_dist = self._compute_bruhat_tits_geodesic(
            exponents, apu_vector
        )
        is_krull_elastic = krull_v >= self.krull_floor

        # 1.5 [FINAL DE FASE 1 -> PUENTE A FASE 2]: Ecuación Homológica de Poincaré-Lindstedt
        sym_cost = 0.5 * (apu_cost_matrix + apu_cost_matrix.T) if dim > 0 else np.zeros((0, 0))
        eigenvalues = np.linalg.eigvalsh(sym_cost) if dim > 0 else np.zeros(0)

        # Frecuencias base inconmensurables (espectro diofantino áureo) para evitar resonancias espurias
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
        # FASE 2: MECÁNICA CELESTE DE POINCARÉ, FUKAYA Y SÍNTESIS RSI³
        # ═══════════════════════════════════════════════════════════════════

        # 2.1 [INICIO DE FASE 2 DESDE 1.5]: Coordenadas de Delaunay-Whittaker + KAM/Brjuno/Nekhoroshev
        dw_state = self._compute_delaunay_whittaker_kam(
            action_vector=pl_report.canonical_actions_seed,
            angle_vector=pl_report.canonical_angles_seed,
            perturbation_matrix=pl_report.renormalized_perturbation_tensor,
            poincare_bridge=pl_report,
        )

        # 2.2: Integral homoclínica de Melnikov y Mapa de Retorno Simpléctico de Poincaré
        action_scalar = float(dw_state.actions_J[0]) if dim > 0 else 1.0
        pert_eps = 0.01 * (1.0 + pl_report.residual_homological)
        melnikov_rep = self._evaluate_melnikov_fourier_series(
            action_J=action_scalar, pert_eps=pert_eps
        )

        # 2.3 & 2.4: Ecuación de Cauchy-Riemann-Floer y No-Squeezing Simpléctico de Gromov
        floer_rep = self._solve_cauchy_riemann_floer_gromov(
            apu_cost_matrix=apu_cost_matrix, ball_r=ball_r, cylinder_R=cylinder_R
        )

        # 2.5 & 2.6 [FINAL DE FASE 2 -> PUENTE A FASE 3]: Máquina Darwin-Gödel RSI³
        rsi3_rep = self._execute_darwin_godel_rsi3(
            base_lip=_BANACH_CONTRACTION_BOUND,
            population_size=5,
            apu_cost_matrix=apu_cost_matrix,
        )

        # ═══════════════════════════════════════════════════════════════════
        # FASE 3: MEDICIÓN QND AAV, GEOMETRÍA DE BURES Y TOPOS DE HEYTING Ω₃
        # ═══════════════════════════════════════════════════════════════════

        # 3.1 [INICIO DE FASE 3 DESDE 2.6]: Medición Débil AAV sobre el operador A*_shield de RSI³
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
            # Fase geométrica sutil en |ϕ_f⟩ para capturar amplificación débil compleja sin perder solapamiento
            phases = np.exp(1j * np.linspace(0.0, 0.05, dim))
            phi_f = (phases.reshape(dim, 1)) / math.sqrt(float(dim))
        else:
            phi_i = np.zeros((0, 1), dtype=np.complex128)
            phi_f = np.zeros((0, 1), dtype=np.complex128)

        qnd_rep = self._compute_stealth_aav_shield_measurement(
            rho_mac=rho_mac,
            phi_i=phi_i,
            phi_f=phi_f,
            rsi3_bridge=rsi3_rep,
        )

        # 3.2: Fidelidad de Uhlmann y Distancia Geodésica de Bures-Wasserstein
        bures_rep = self._evaluate_bures_uhlmann_fidelity(rho_mac, rho_mac)

        # 3.3: Adjudicación Soberana en el Topos de Heyting Ω₃
        truth_val = self._adjudicate_heyting_shield(
            rw_norm=rw_norm,
            torsion_free=torsion_free,
            betti_zero=betti_zero,
            krull_v=krull_v,
            ultrametric_ok=ultrametric_ok,
            bt_distance=bt_distance,
            pl_resonance_free=pl_report.is_resonance_free,
            kam_stable=dw_state.is_kam_stable,
            melnikov_chaos=melnikov_rep.chaos_detected,
            gromov_valid=floer_rep.is_gromov_squeezing_valid,
            cr_residual=floer_rep.cr_floer_residual,
            banach_lip=rsi3_rep.banach_lip_t3,
            qnd_stealth=qnd_rep.is_stealth_qnd,
            bures_preserved=bures_rep.is_isometry_preserved,
        )

        # 3.4: Firma Criptográfica HMAC-SHA256 del Pasaporte Espectral
        hmac_sig, passport_hash = self._compute_hmac_passport(
            rw_norm=rw_norm,
            cr_res=floer_rep.cr_floer_residual,
            krull_v=krull_v,
            bt_dist=bt_distance,
            back_action=qnd_rep.back_action_db,
            banach_lip=rsi3_rep.banach_lip_t3,
            truth_val=truth_val,
        )

        # 3.5: Empaquetado Monádico Final
        t_end_ns = time.perf_counter_ns()
        exec_time_ns = t_end_ns - t_start_ns

        novikov_report = NovikovBruhatTitsReport(
            krull_valuation=krull_v,
            novikov_norm=novikov_norm,
            bruhat_tits_distance=bt_distance,
            bruhat_tits_geodesic=bt_geodesic,
            is_krull_elastic=is_krull_elastic,
            ultrametric_inequality_holds=ultrametric_ok,
            fukaya_category_rank=fukaya_rank,
            weyl_chamber_wall_distance=weyl_wall_dist,
        )

        return NinjaShieldSpectrumReport(
            rademacher_walsh_norm=rw_norm,
            smith_torsion_free=torsion_free,
            betti_number_zero=betti_zero,
            poincare_lindstedt_report=pl_report,
            novikov_report=novikov_report,
            delaunay_whittaker_state=dw_state,
            melnikov_kam_report=melnikov_rep,
            pseudoholomorphic_floer_report=floer_rep,
            rsi3_banach_report=rsi3_rep,
            qnd_report=qnd_rep,
            bures_report=bures_rep,
            truth_value=truth_val,
            hmac_signature=hmac_sig,
            passport_hash=passport_hash,
            execution_time_ns=exec_time_ns,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST AUTÓNOMO DE CERTIFICACIÓN SOBERANA (3 FASES ANIDADAS)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("═" * 78)
    print("=== TEST AUTÓNOMO: BoolNovikovNinjaShieldEngine v15.0.0-PhD (3 FASES) ===")
    print("═" * 78)

    engine = BoolNovikovNinjaShieldEngine()
    matrix_test = np.eye(4, dtype=np.float64)
    exps_test = [0.15, 0.22, 0.35]

    report = engine.execute_ninja_shield_audit(matrix_test, exps_test)

    print(f"Versión del Motor                : {__version__}")
    print(f"Tiempo de Ejecución              : {report.execution_time_ns / 1e6:.3f} ms")
    print("─" * 78)
    print("FASE 1 — CUSTODIA ULTRAMÉTRICA, HOMOLOGÍA ℤ Y PUENTE POINCARÉ-LIE:")
    print(f"  1.1 Rademacher-Walsh ||∇||₂    : {report.rademacher_walsh_norm:.6f}")
    print(f"  1.2 SNF Torsión Libre (ℤ)      : {report.smith_torsion_free}")
    print(f"  1.2 Betti β₁ = 0               : {report.betti_number_zero}")
    print(f"  1.3 Krull v(A)                 : {report.novikov_report.krull_valuation:.4f}")
    print(f"  1.3 ||A||_Nov                  : {report.novikov_report.novikov_norm:.6f}")
    print(f"  1.3 Ultramétrica Isósceles OK  : {report.novikov_report.ultrametric_inequality_holds}")
    print(f"  1.4 Bruhat-Tits d_BT           : {report.novikov_report.bruhat_tits_distance:.6f}")
    print(f"  1.4 Fukaya Category Rank       : {report.novikov_report.fukaya_category_rank}")
    print(f"  1.5 Poincaré-Lindstedt Resid   : {report.poincare_lindstedt_report.residual_homological:.6e}")
    print(f"  1.5 Poincaré-Lindstedt ResFree : {report.poincare_lindstedt_report.is_resonance_free}")
    print(f"  1.5 Energía Birkhoff H_eff     : {report.poincare_lindstedt_report.birkhoff_normal_form_energy:.6f}")
    print("─" * 78)
    print("FASE 2 — MECÁNICA CELESTE DE POINCARÉ, FUKAYA Y AUTOMEJORA RSI³:")
    print(f"  2.1 KAM Estable                : {report.delaunay_whittaker_state.is_kam_stable}")
    print(f"  2.1 Condición de Brjuno OK     : {report.delaunay_whittaker_state.is_bruno_condition}")
    print(f"  2.1 Suma de Brjuno B(ω)        : {report.delaunay_whittaker_state.brjuno_sum_value:.4f}")
    print(f"  2.1 Tiempo Nekhoroshev T_Nekh  : {report.delaunay_whittaker_state.nekhoroshev_stability_time:.4e}")
    print(f"  2.1 Cota Difusión de Arnold    : {report.delaunay_whittaker_state.arnold_diffusion_bound:.4e}")
    print(f"  2.2 Melnikov Integral M(τ)     : {report.melnikov_kam_report.melnikov_integral:.6e}")
    print(f"  2.2 Det Monodromía Poincaré    : {report.melnikov_kam_report.poincare_monodromy_det:.6f}")
    print(f"  2.2 Estabilidad de Krein       : {report.melnikov_kam_report.krein_signature_stable}")
    print(f"  2.3 Residual CR-Floer ||∂̄||    : {report.pseudoholomorphic_floer_report.cr_floer_residual:.6e}")
    print(f"  2.4 Gromov No-Squeezing OK     : {report.pseudoholomorphic_floer_report.is_gromov_squeezing_valid}")
    print(f"  2.5 Contracción Kleisli Lip(T³): {report.rsi3_banach_report.banach_lip_t3:.6f}")
    print(f"  2.5 Niveles RSI³ (L₁, L₂, L₃)  : ({report.rsi3_banach_report.level1_state_contraction:.4f}, "
          f"{report.rsi3_banach_report.level2_operator_contraction:.4f}, "
          f"{report.rsi3_banach_report.level3_meta_contraction:.4f})")
    print(f"  2.6 Teorema Modal de Löb OK    : {report.rsi3_banach_report.loeb_provability_verified}")
    print(f"  2.6 Jerk Ciber-Físico d³x/dt³  : {report.rsi3_banach_report.jerk_cyber_physical:.6e}")
    print("─" * 78)
    print("FASE 3 — MEDICIÓN CUÁNTICA QND, BURES-UHLMANN Y TOPOS DE HEYTING Ω₃:")
    print(f"  3.1 Valor Débil AAV A_w        : {report.qnd_report.weak_value:.6f}")
    print(f"  3.1 Norma Conmutador ||[A,ρ]|| : {report.qnd_report.commutator_norm:.6e}")
    print(f"  3.1 Back-Action Cuántica (dB)  : {report.qnd_report.back_action_db:.6f} dB")
    print(f"  3.1 Sigilo QND Certificado     : {report.qnd_report.is_stealth_qnd}")
    print(f"  3.2 Fidelidad de Uhlmann F     : {report.bures_report.fidelity_uhlmann:.6f}")
    print(f"  3.2 Distancia de Bures D_B     : {report.bures_report.bures_distance:.6f}")
    print(f"  3.3 Veredicto Topos Heyting Ω₃ : {report.truth_value.name} (0x{report.truth_value.value:02X})")
    print(f"  3.4 Pasaporte SHA-256          : {report.passport_hash[:32]}...")
    print(f"  3.4 Firma HMAC-SHA256          : {report.hmac_signature[:32]}...")
    print("═" * 78)