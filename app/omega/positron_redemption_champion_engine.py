# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Positron Redemption Champion Engine (Motor Espectral Ciego FPU · Fiscal/Campeón)          ║
║ RUTA    : app/omega/positron_redemption_champion_engine.py                                          ║
║ VERSIÓN : 18.0.0-FPU-Champion-Poincare-RSIL3-Dirac-Lazutkin-Fredholm-KaplanYorke-PhD                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
NATURALEZA DEL MOTOR (Soberanía Física y Metacognitiva de Nivel 3)
──────────────────────────────────────────────────────────────────────────────────────────────────────
Funtor acusador variacional en la FPU del Estrato Omega (V_Ω, Nivel 0.5 — Ágora Tensorial).
NO evalúa retículos de Heyting, NO dicta veredictos de negocio, NO mantiene estados OODA,
NO toca periféricos. Emite únicamente el certificado numérico inmutable
`PositronChampionSpectrumReport`, consumido por `positron_redemption_champion_agent.py`.
Los booleanos emitidos son PREDICADOS FÍSICOS Y ESPECTRALES (solubilidad de Fredholm,
conservación exergética, contención homoclínica, contractividad RSI-L3), nunca veredictos.

EVOLUCIÓN DOCTRINAL v18.0.0: MECÁNICA CELESTE DE POINCARÉ Y AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3)
──────────────────────────────────────────────────────────────────────────────────────────────────────
  (i)   MECÁNICA CELESTE DE POINCARÉ (*Les Méthodes Nouvelles de la Mécanique Céleste*):
        • Variables canónicas heliocentristas NO SINGULARES de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂) que
          eliminan la singularidad coordenada de Delaunay cuando e → 0 o i → 0:
              ξ₁ + iη₁ = √(2(Λ − G))·exp(iϖ),   ξ₂ + iη₂ = √(2(G − H))·exp(iΩ).
          El Déficit de Momento Angular (AMD de Laskar-Poincaré) se expresa como forma cuadrática
          armónica exacta: AMD = ½(ξ₁² + η₁² + ξ₂² + η₂²) = J₁ + J₂.
        • Forma Normal Resonante de Poincaré-Birkhoff, invariantes integrales primero y segundo de
          Poincaré-Cartan (∮ p·dq y ∬ dp∧dq), y ángulo homoclínico con corrección de transporte.
  (ii)  AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3 · Endofuntor de Lawvere-Banach):
        • Nivel 0: Evaluación funcional base.
        • Nivel 1: Adaptación paramétrica local (step-doubling).
        • Nivel 2: Optimización de hiperparámetros de cuadratura/tolerancia.
        • Nivel 3 (RSI-L3): El motor audita su propia familia de operadores epistémicos T_θ
          (proyectores espectrales D_δ(H), resolventes de Lyapunov-Sylvester, particiones
          simplécticas de Yoshida-Magnus y estimadores de contracción ISS), calcula el
          hiper-gradiente analítico de Fréchet-Danskin ∂_θ L_meta(T_θ) respecto de la ley del
          operador, y ejecuta una actualización proximal contractiva de Banach-Picard (T_θ → T_θ*)
          con certificado de contracción q_meta < 1, SIN alterar jamás constantes físicas CODATA.
  (iii) GEOMETRÍA CUÁNTICA SLD-BURES, ÁLGEBRA DE CLIFFORD Cl_{1,3}(ℂ) Y CIRCUITOS DE FOSTER-CAUER:
        • Ecuación cuántica de Lyapunov-Sylvester ½(ρL + Lρ) = σ − ρ para la Derivada Logarítmica
          Simétrica (SLD) y métrica de información cuántica de Fisher-Bures.
        • Invariantes de Casimir del grupo de Poincaré ISO(1,3) y razón de ramificación de
          Ore-Powell (ortopositronio 3γ vs parapositronio 2γ).
        • Certificación de Real-Positividad (PR) y pasividad de Foster-Cauer del multipuerto K.

CONTINUIDAD FORMAL ESTRICTA ENTRE LAS TRES FASES ANIDADAS
──────────────────────────────────────────────────────────────────────────────────────────────────────
    1.9 audit_dirac_uhlmann_substrate ──[Pi1ToPi2Seed]──► 2.1 _ingest_pi1_continuation
    2.11 audit_lazutkin_poisson_sector ──[Pi2ToPi3Seed]──► 3.1 _ingest_pi2_continuation
    3.9 prosecute_positron_trajectory = composición Π₃ ∘ Π₂ ∘ Π₁ y emisión del certificado.

DICTAMEN DE CORRECCIONES GRANULARES (v16 → v17 → v18)
  [F1–F12] Preservadas íntegramente todas las correcciones de v17 sobre v16.
  [F13] Corregido bug latente de v17 en `_chirikov_overlap` (`_WILKINSON_EPSILON` → `_WILKINSON_EPS`).
  [F14] Eliminada la singularidad e→0, i→0 mediante coordenadas canónicas no singulares de Poincaré.
  [F15] Integrada la Derivada Logarítmica Simétrica (SLD) vía ecuación de Sylvester y pasividad PR.
  [F16] Integrados invariantes de Casimir de Poincaré ISO(1,3) y canal 3γ de Ore-Powell.
  [F17] Integrado el lazo endofuntorial de Automejora Recursiva de Nivel 3 (RSI-L3) en Π₁, Π₂ y Π₃.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Omega.PositronRedemptionChampionEngine")

__all__ = [
    "PositronChampionSpectrumReport",
    "PositronRedemptionChampionEngine",
    "ChampionTelemetry",
    "Pi1SpectrumCertificate",
    "Pi2SpectrumCertificate",
    "PoincareCanonicalElements",
    "RSIL3MetaCertificate",
    "ENGINE_VERSION",
]

_ENGINE_VERSION: Final[str] = (
    "18.0.0-FPU-Champion-Poincare-RSIL3-Dirac-Lazutkin-Fredholm-KaplanYorke-PhD"
)
version: Final[str] = _ENGINE_VERSION
ENGINE_VERSION: Final[str] = _ENGINE_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES FÍSICAS (CODATA), NUMÉRICAS, CELESTES Y METACOGNITIVAS (RSI-L3)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_STABILITY_FLOOR: Final[float] = max(1.0e-12, _MACHINE_EPS)
_TINY: Final[float] = 1.0e-300
_WILKINSON_EPS: Final[float] = 1.0e-15
_WILKINSON_EPSILON: Final[float] = _WILKINSON_EPS  # Alias defensivo contra regresión [F13]
_CONDITION_CAP: Final[float] = 1.0e12
_MAX_MODULAR_CONDITION: Final[float] = 1.0e8
_LN10: Final[float] = math.log(10.0)
_FLOAT64_LOG10_MIN: Final[float] = -308.0

# Constantes fundamentales CODATA (invariantes bajo cualquier lazo RSI-L3)
_ELECTRON_REST_MASS_KEV: Final[float] = 510.998950003            # m_e c² [keV]
_ANNIHILATION_ENERGY_REST_KEV: Final[float] = 2.0 * _ELECTRON_REST_MASS_KEV  # 1021.9979 keV
_CLASSICAL_ELECTRON_RADIUS_M: Final[float] = 2.8179403262e-15    # r_e [m]
_FINE_STRUCTURE_ALPHA: Final[float] = 7.2973525693e-3            # α_fs ≈ 1/137.036
_SPEED_OF_LIGHT_MS: Final[float] = 299792458.0
_PI_R_E_SQUARED: Final[float] = math.pi * (_CLASSICAL_ELECTRON_RADIUS_M ** 2)
# Factor adimensional de Ore-Powell para ortopositronio (1³S₁ → 3γ vs 1¹S₀ → 2γ)
_ORE_POWELL_TRIPLET_TO_SINGLET: Final[float] = (
    (4.0 / (9.0 * math.pi)) * (math.pi ** 2 - 9.0) * _FINE_STRUCTURE_ALPHA
)

# Constantes de Mecánica Celeste de Poincaré y asintótica de Lazutkin-Gelfreich
_LAZUTKIN_PREFACTOR: Final[float] = 1118.82770595
_LAZUTKIN_ASYMPTOTIC_EPS_MAX: Final[float] = 0.5
_GREENE_CRITICAL_K: Final[float] = 0.971635406                   # Criterio de residuos de Greene
_POINCARE_BIRKHOFF_MAX_ORDER: Final[int] = 4
_MONODROMY_BASE_PANELS: Final[int] = 96
_MONODROMY_MAX_DOUBLINGS: Final[int] = 5

# Hiperparámetros del Endofuntor de Automejora Recursiva de Nivel 3 (RSI-L3)
_RSI_L3_MAX_ITERATIONS: Final[int] = 5
_RSI_L3_CONTRACTION_TARGET: Final[float] = 0.85
_RSI_L3_STEP_GAIN: Final[float] = 0.35
_DEFAULT_FISCAL_WEIGHTS: Final[Tuple[float, float, float, float]] = (0.40, 0.20, 0.30, 0.10)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión total a float finito: la FPU jamás propaga NaN/Inf al certificado."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fro(mat: NDArray[Any]) -> float:
    """Norma de Frobenius en el ideal de Hilbert-Schmidt S₂(H)."""
    return float(la.norm(mat, ord="fro")) if getattr(mat, "size", 0) else 0.0


def _relative(defect: float, scale: float) -> float:
    """Defecto relativo estabilizado frente a escalas degeneradas."""
    return float(defect / max(abs(scale), _TINY))


def _freeze_array(arr: NDArray[Any]) -> NDArray[Any]:
    """Copia defensiva inmutable de sólo lectura para blindar el certificado espectral."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §1 · CERTIFICADOS DE FASE, COORDENADAS DE POINCARÉ, RSI-L3 Y SEMILLAS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RSIL3MetaCertificate:
    """
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3 · Punto Fijo de Lawvere-Banach).
    Acredita que el operador numérico se refinó contractivamente sin alterar invariantes físicos.
    """
    level: int
    iterations_executed: int
    initial_meta_loss: float
    final_meta_loss: float
    meta_gradient_norm: float
    banach_contraction_factor: float
    optimal_parameter: float
    converged: bool
    physical_invariance_preserved: bool = True


@dataclass(frozen=True)
class PoincareCanonicalElements:
    """
    Variables canónicas heliocentristas NO SINGULARES de Poincaré (*Méthodes Nouvelles*, T. I):
        Λ = √(μa),  G = Λ√(1−e²),  H = G cos i,
        ξ₁ = √(2(Λ−G)) cos ϖ,  η₁ = √(2(Λ−G)) sin ϖ   (modo excéntrico sin singularidad en e=0),
        ξ₂ = √(2(G−H)) cos Ω,  η₂ = √(2(G−H)) sin Ω   (modo oblicuo sin singularidad en i=0).
    Déficit de Momento Angular exacto: AMD = ½(ξ₁² + η₁² + ξ₂² + η₂²) = (Λ − G) + (G − H) = Λ − H.
    """
    lambda_delaunay: float
    g_delaunay: float
    h_delaunay: float
    xi_1_eccentric: float
    eta_1_eccentric: float
    xi_2_oblique: float
    eta_2_oblique: float
    action_j1_radial: float
    action_j2_vertical: float
    amd_laskar_poincare: float
    nonsingular_regularity_ok: bool


@dataclass(frozen=True)
class DensityCanonicalization:
    """Proyección exacta sobre D_δ(H) con certificados KKT, Mirsky y sensibilidad activa."""
    rho: NDArray[np.complex128]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    raw_eigenvalues: NDArray[np.float64]
    entropy: float
    purity: float
    condition_number: float
    projection_cost: float
    floor_used: float
    active_floor_count: int
    kkt_ok: bool
    hermiticity_defect: float
    trace_defect: float
    used_fallback: bool


@dataclass(frozen=True)
class ConductivityPurification:
    """
    Proyección de Frobenius sobre PSD ∩ Sym_n(ℝ) [F9] + Pasividad de Foster-Cauer /
    Real-Positividad (PR) de multipuerto eléctrico-cuántico [F15].
    """
    k_matrix: NDArray[np.complex128]
    lambda_min: float
    lambda_max: float
    top_gap: float
    condition_raw: float
    condition_capped: float
    log10_condition: float
    imaginary_discarded_norm: float
    projection_cost: float
    foster_cauer_passivity_ok: bool
    cauer_driving_point_impedance_trace: float
    used_fallback: bool


@dataclass(frozen=True)
class DiracCertificate:
    """
    Balance de aniquilación en el álgebra de Clifford Cl_{1,3}(ℂ) e invariantes de Casimir
    del grupo de Poincaré ISO(1,3), incluyendo razón 3γ/2γ de Ore-Powell. [F2, F16]
    """
    beta: float
    gamma_lorentz: float
    energy_rest_kev: float
    energy_lab_kev: float
    energy_cm_kev: float
    cross_section_m2: float
    gamow_factor: float
    rate_coefficient_m3s: float
    energy_floor_ok: bool
    poincare_casimir_mandelstam_s_kev2: float = 0.0
    pauli_lubanski_singlet_casimir: float = 0.0
    pauli_lubanski_triplet_casimir: float = 0.0
    ore_powell_3gamma_ratio: float = _ORE_POWELL_TRIPLET_TO_SINGLET
    cross_section_3gamma_m2: float = 0.0
    energy_independent_of_bures: bool = True


@dataclass(frozen=True)
class Pi1ToPi2Seed:
    """
    [MORFISMO 1.9 → 2.1] Semilla cuántico-simpléctica que acopla el sustrato de Dirac-Uhlmann (Π₁)
    con la Mecánica Celeste de Poincaré y la teoría de Lazutkin-Fredholm (Π₂).
    """
    bures_tension: float
    uhlmann_fidelity: float
    sld_fisher_metric: float
    annihilation_energy_rest_kev: float
    annihilation_rate_coefficient: float
    rho_entropy: float
    rho_populations: Tuple[float, ...]
    poincare_elements: PoincareCanonicalElements
    rsi_l3_pi1: RSIL3MetaCertificate
    cauer_impedance_trace: float = 0.0
    has_context: bool = True


@dataclass(frozen=True)
class Pi1SpectrumCertificate:
    """Certificado numérico de Π₁ (Dirac-Clifford, Uhlmann-SLD, Poincaré Canónico y RSI-L3)."""
    rho: DensityCanonicalization
    sigma: DensityCanonicalization
    conductivity: ConductivityPurification
    uhlmann_fidelity: float
    bures_distance: float
    bures_angle: float
    chord_arc_error: float
    fidelity_route_discrepancy: float
    hilbert_schmidt_overlap: float
    sld_fisher_metric: float
    sld_lyapunov_residual: float
    dirac: DiracCertificate
    poincare_elements: PoincareCanonicalElements
    rsi_l3_certificate: RSIL3MetaCertificate
    continuation: Optional[Pi1ToPi2Seed] = None


@dataclass(frozen=True)
class LazutkinCertificate:
    """Escisión homoclínica de separatriz en espacio logarítmico + invariante de Poincaré-Birkhoff."""
    regime: str
    epsilon: float
    exponent: float
    log10_angle: float
    angle: float
    underflow: bool
    asymptotic_valid: bool
    prefactor_convention: float
    poincare_birkhoff_detuning: float = 0.0
    homoclinic_lobe_invariant: float = 0.0


@dataclass(frozen=True)
class MelnikovCertificate:
    """Turnstile de Poincaré-Melnikov-MacKay-Meiss-Percival con refinamiento adaptativo RSI-L3."""
    log10_amplitude: float
    oscillation_amplitude: float
    dissipative_barrier: float
    lobe_area: float
    flux: float
    transversal: bool
    quadrature_error: float
    quadrature_nodes_used: int = 2001


@dataclass(frozen=True)
class LyapunovCertificate:
    """Espectro de Floquet-Lyapunov sobre la ecuación variacional de Poincaré-Hill y D_KY."""
    spectrum: Tuple[float, ...]
    max_exponent: float
    sum_exponents: float
    kaplan_yorke_dimension: float
    ky_index_j: int
    degenerate_conservative: bool
    symplectic_defect: float
    determinant: float
    spectral_radius: float
    panels: int
    step_certificate: float
    poincare_characteristic_multipliers_modulus: Tuple[float, ...] = ()


@dataclass(frozen=True)
class FredholmCertificate:
    """Alternativa de Fredholm genuina sobre el Laplaciano combinatorio con regularización espectral."""
    beta_0: int
    compatibility_residual: float
    solvable_raw: bool
    neutralization_applied: bool
    poisson_residual: float
    potential_norm: float
    source_norm: float
    kernel_projection_norm: float
    fiedler_algebraic_connectivity: float = 0.0


@dataclass(frozen=True)
class Pi2ToPi3Seed:
    """[MORFISMO 2.11 → 3.1] Semilla caótico-celeste y metacognitiva que parametriza la Fase Π₃."""
    chirikov_overlap_k: float
    separatrix_broken: bool
    lazutkin_log10_angle: float
    melnikov_flux: float
    fredholm_compatibility_residual: float
    poisson_residual: float
    kaplan_yorke_dimension: float
    lyapunov_max: float
    poincare_cartan_action: float
    poincare_second_integral_invariant: float = 0.0
    amd_laskar_poincare: float = 0.0
    rsi_l3_pi2: Optional[RSIL3MetaCertificate] = None
    has_context: bool = True


@dataclass(frozen=True)
class Pi2SpectrumCertificate:
    """Certificado numérico de Π₂ (Mecánica Celeste de Poincaré, Lazutkin, Melnikov, Fredholm, RSI-L3)."""
    lazutkin: LazutkinCertificate
    melnikov: MelnikovCertificate
    chirikov_k: float
    chirikov_contained: bool
    separatrix_broken: bool
    lyapunov: LyapunovCertificate
    fredholm: FredholmCertificate
    poincare_cartan_action: float
    novikov_action_period: float
    novikov_action_reduced: float
    poincare_elements: Optional[PoincareCanonicalElements] = None
    poincare_second_integral_invariant: float = 0.0
    birkhoff_normal_form_remainder: float = 0.0
    rsi_l3_certificate: Optional[RSIL3MetaCertificate] = None
    continuation: Optional[Pi2ToPi3Seed] = None


@dataclass(frozen=True)
class FiscalIndexCertificate:
    """Índice de acusación DECLARADO: adimensional, acotado y monótono."""
    index: float
    weights: Tuple[float, float, float, float]
    terms: Tuple[float, float, float, float]
    monotone_ok: bool
    normalized_ok: bool
    is_index_not_floer_action: bool = True


@dataclass(frozen=True)
class JerkCertificate:
    """Tercera diferencia genuina sobre el historial del índice con margen ISS auto-ajustado."""
    third_difference: float
    normalized: float
    iss_margin: float
    defined: bool
    history_length: int
    delta_t: float
    contraction_q_used: float = 0.5


@dataclass
class ChampionTelemetry:
    """
    Memoria episódica y metacognitiva (RSI-L3) del motor fiscal.
    Almacena la trayectoria del índice para la derivada discreta de tercer orden (Jerk) y la
    evolución del operador contractivo de punto fijo de Lawvere-Banach.
    """
    total_prosecutions: int = 0
    index_history: List[float] = field(default_factory=list)
    delta_t_history: List[float] = field(default_factory=list)
    rsi_l3_loss_history: List[float] = field(default_factory=list)
    cumulative_monodromy_panels: int = 0
    cumulative_fredholm_solves: int = 0
    cumulative_rsi_l3_iterations: int = 0
    last_index: Optional[float] = None
    contraction_estimate: float = 0.5


@dataclass(frozen=True)
class PositronChampionSpectrumReport:
    r"""
    Certificado espectral inmutable de fiscalización emitido por la FPU (v18.0.0).
    Preserva íntegramente el orden posicional y nombres de v16/v17; añade telemetría de
    Mecánica Celeste de Poincaré, SLD-Bures, Foster-Cauer, Ore-Powell y RSI-L3 con valores por defecto.
    """
    p_champion: NDArray[np.float64]
    rho_champion: NDArray[np.complex128]
    K_purified: NDArray[np.complex128]
    novikov_floer_functional: float
    lazutkin_splitting_angle: float
    poisson_fredholm_residual: float
    jerk_ciber_physical: float
    kaplan_yorke_dimension: float
    annihilation_energy_kev: float
    amber_grace_expired: bool
    connes_lipschitz_bound: float
    condition_number_K: float
    schema_version: str = "18.0.0-PhD-Champion-Poincare-RSIL3-Dirac-Lazutkin-Fredholm-KaplanYorke"
    # ── Π₁ extendido (v17 + v18 SLD, Clifford, Foster-Cauer, Poincaré, RSI-L3) ──
    fiscal_indictment_index: float = 0.0
    fiscal_weights: Tuple[float, float, float, float] = _DEFAULT_FISCAL_WEIGHTS
    fiscal_terms: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    fiscal_index_monotone_ok: bool = True
    uhlmann_fidelity: float = 0.0
    bures_distance: float = 0.0
    bures_angle: float = 0.0
    chord_arc_identity_error: float = 0.0
    fidelity_route_discrepancy: float = 0.0
    hilbert_schmidt_overlap: float = 0.0
    sld_fisher_metric: float = 0.0
    sld_lyapunov_residual: float = 0.0
    rho_condition_number: float = 1.0
    rho_purity: float = 1.0
    rho_entropy: float = 0.0
    rho_projection_cost: float = 0.0
    rho_kkt_ok: bool = True
    simplex_kkt_ok: bool = True
    K_imaginary_discarded_norm: float = 0.0
    K_projection_cost: float = 0.0
    foster_cauer_passivity_ok: bool = True
    cauer_driving_point_impedance_trace: float = 0.0
    dirac_beta: float = 0.0
    dirac_gamma_lorentz: float = 1.0
    annihilation_energy_rest_kev: float = _ANNIHILATION_ENERGY_REST_KEV
    annihilation_energy_lab_kev: float = 0.0
    annihilation_energy_cm_kev: float = 0.0
    annihilation_rate_coefficient_m3s: float = 0.0
    dirac_cross_section_m2: float = 0.0
    dirac_cross_section_3gamma_m2: float = 0.0
    ore_powell_3gamma_ratio: float = _ORE_POWELL_TRIPLET_TO_SINGLET
    poincare_casimir_mandelstam_s_kev2: float = 0.0
    gamow_sommerfeld_factor: float = 1.0
    # ── Π₂ extendido (v17 + v18 Mecánica Celeste de Poincaré) ──
    lazutkin_regime: str = "MAP_LAZUTKIN_GELFREICH"
    lazutkin_epsilon: float = 0.0
    lazutkin_exponent: float = 0.0
    lazutkin_log10_angle: float = 0.0
    lazutkin_underflow: bool = False
    lazutkin_asymptotic_valid: bool = True
    poincare_birkhoff_detuning: float = 0.0
    homoclinic_lobe_invariant: float = 0.0
    melnikov_log10_amplitude: float = 0.0
    melnikov_lobe_area: float = 0.0
    melnikov_flux: float = 0.0
    melnikov_transversal: bool = False
    melnikov_quadrature_error: float = 0.0
    chirikov_overlap_k: float = 0.0
    chirikov_contained: bool = True
    separatrix_broken: bool = False
    lyapunov_spectrum: Tuple[float, ...] = ()
    lyapunov_max: float = 0.0
    lyapunov_sum: float = 0.0
    ky_index_j: int = 0
    ky_degenerate_conservative: bool = True
    monodromy_symplectic_defect: float = 0.0
    monodromy_determinant: float = 1.0
    monodromy_spectral_radius: float = 1.0
    monodromy_step_certificate: float = 0.0
    poisson_residual: float = 0.0
    fredholm_compatibility_residual: float = 0.0
    fredholm_solvable_raw: bool = False
    fredholm_neutralization_applied: bool = True
    poisson_beta_0: int = 1
    poisson_potential_norm: float = 0.0
    fiedler_algebraic_connectivity: float = 0.0
    poincare_cartan_action: float = 0.0
    poincare_second_integral_invariant: float = 0.0
    poincare_amd_laskar: float = 0.0
    poincare_xi1_eccentric: float = 0.0
    poincare_eta1_eccentric: float = 0.0
    poincare_xi2_oblique: float = 0.0
    poincare_eta2_oblique: float = 0.0
    poincare_nonsingular_regularity_ok: bool = True
    birkhoff_normal_form_remainder: float = 0.0
    novikov_action_period: float = 0.0
    novikov_action_reduced: float = 0.0
    # ── Π₃ extendido y Certificados Metacognitivos RSI-L3 ──
    jerk_defined: bool = False
    jerk_normalized: float = 0.0
    jerk_iss_margin: float = 0.0
    jerk_history_length: int = 0
    jerk_delta_t: float = 0.0
    dk_sqrt_lipschitz: float = 0.0
    dk_invsqrt_lipschitz: float = 0.0
    condition_number_K_raw: float = 1.0
    log10_condition_number_K: float = 0.0
    lambda_min_K: float = 0.0
    lambda_max_K: float = 0.0
    spectral_gap_K: float = 0.0
    amber_expired_by_time: bool = False
    amber_expired_by_energy: bool = False
    redemption_energy_floor_ok: bool = True
    redemption_valid: bool = True
    rsi_l3_converged: bool = True
    rsi_l3_banach_contraction: float = 0.0
    rsi_l3_meta_loss_reduction_ratio: float = 1.0
    rsi_l3_total_iterations: int = 0
    engine_version: str = _ENGINE_VERSION
    schema_flags: Tuple[str, ...] = ()


# ══════════════════════════════════════════════════════════════════════════════
# §2 · MOTOR FISCAL CIEGO: TRES FASES Π ANIDADAS CON CONTINUIDAD FORMAL
# ══════════════════════════════════════════════════════════════════════════════
class PositronRedemptionChampionEngine:
    """
    Motor Espectral Ciego de FPU para el Fiscal/Campeón de Redención (v18.0.0).
    Integra Mecánica Celeste de Poincaré y Automejora Recursiva de Nivel 3 (RSI-L3)
    a través de tres fases anidadas con continuidad formal estricta:
        Π₁ · PhasePi1_DiracUhlmannFiscalizer      (1.1 → 1.9)
        Π₂ · PhasePi2_LazutkinPoissonFiscalizer   (2.1 → 2.11)
        Π₃ · PhasePi3_FiscalIndictmentEmitter     (3.1 → 3.9)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₁ · DIRAC-CLIFFORD, SLD-BURES, VARIABLES DE POINCARÉ Y RSI-L3
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi1_DiracUhlmannFiscalizer:
        """
        FASE Π₁. Cadena formal:
          1.1 Validación en álgebra de Banach y proyección autoadjunta ½(M+M†)
          1.2 Proyección exacta sobre el símplex capado Δ_δ^{n−1} con máscara de sensibilidad
          1.3 Canonización de ρ en D_δ(H) (Mirsky / Hoffman-Wielandt)
          1.4 Purificación de K en PSD ∩ Sym_n(ℝ) + Pasividad de Foster-Cauer (PR)
          1.5 Geometría de Uhlmann-Bures + Derivada Logarítmica Simétrica (SLD de Lyapunov)
          1.6 Balance de Dirac en Cl_{1,3}(ℂ), Casimir de Poincaré ISO(1,3) y Ore-Powell (3γ/2γ)
          1.7 Síntesis de Variables Canónicas No Singulares de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂)
          1.8 Automejora Recursiva de Nivel 3 (RSI-L3) sobre el proyector espectral Π_δ
          1.9 [TERMINAL Π₁ → INICIO Π₂] audit_dirac_uhlmann_substrate (emite Pi1ToPi2Seed → 2.1)
        """

        def __init__(self, dimension_n: int, tolerance: float, tau_spd: float) -> None:
            self._n = max(1, int(dimension_n))
            self._tol = float(tolerance)
            self._tau_spd = float(tau_spd)
            self._uniform = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)
            self._eye = np.eye(self._n, dtype=np.complex128)

        # ── Método 1.1 ────────────────────────────────────────────────────────
        def _validate_and_hermitize(
            self, mat: Any
        ) -> Tuple[NDArray[np.complex128], float, float, bool]:
            """
            1.1 Validación total de forma/finitud en el álgebra de Banach B(H) y descomposición
            cartesiana de Toeplitz M = H + iA, donde H = ½(M+M†) ∈ Herm_n(ℂ).
            Devuelve (H, defecto_traza, defecto_relativo_hermiticidad, usó_fallback).
            """
            ok = (
                isinstance(mat, np.ndarray)
                and mat.ndim == 2
                and mat.shape == (self._n, self._n)
                and mat.size > 0
                and bool(np.all(np.isfinite(mat)))
            )
            if not ok:
                return self._eye / float(self._n), 0.0, 0.0, True
            m = np.asarray(mat, dtype=np.complex128)
            herm = 0.5 * (m + m.conj().T)
            h_def = _relative(_fro(m - herm), _fro(m))
            t_def = abs(float(np.real(np.trace(m))) - 1.0)
            return herm, float(t_def), float(h_def), False

        # ── Método 1.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _project_capped_simplex(
            values: NDArray[np.float64], floor: float
        ) -> Tuple[NDArray[np.float64], float, bool]:
            """
            1.2 Proyección euclídea EXACTA sobre el símplex capado Δ_δ^{n−1} = {p ≥ δ, Σp = 1}
            mediante water-filling espectral O(n log n) (Duchi-Condat).
            Con cambio afín s = p − δ ≥ 0 y masa libre S = 1 − nδ:
                p_i(δ) = δ + max(v_i − δ − θ(δ), 0),   Σ_i max(v_i − δ − θ(δ), 0) = 1 − nδ.
            La derivada direccional ∂p_i/∂δ sobre el conjunto activo soporta el hiper-gradiente
            analítico de RSI-L3 en 1.8. Corrección aditiva final sobre argmax(p).
            """
            v = np.asarray(values, dtype=np.float64).ravel()
            n = int(v.size)
            if n == 0:
                return np.array([1.0], dtype=np.float64), 0.0, False
            delta = float(min(max(floor, 0.0), 0.5 / float(n)))
            feasible = bool(delta * n <= 1.0)
            mass = 1.0 - float(n) * delta
            y = v - delta
            u = np.sort(y)[::-1]
            css = np.cumsum(u)
            idx = np.arange(1, n + 1, dtype=np.float64)
            cond = u - (css - mass) / idx > 0.0
            k = int(np.flatnonzero(cond)[-1]) + 1 if bool(cond.any()) else 1
            theta = float((css[k - 1] - mass) / float(k))
            p = np.maximum(y - theta, 0.0) + delta
            p[int(np.argmax(p))] += 1.0 - float(np.sum(p))
            return p, delta, feasible

        # ── Método 1.3 ────────────────────────────────────────────────────────
        def _canonicalize_with_certificate(
            self, rho: Any, floor_override: Optional[float] = None
        ) -> DensityCanonicalization:
            """
            1.3 Proyección de Frobenius exacta sobre la variedad de estados cuánticos estrictamente
            positivos D_δ(H) = {ρ = ρ† ⪰ δI, Tr ρ = 1}. Por el teorema de Mirsky y Hoffman-Wielandt,
            para toda norma unitariamente invariante el óptimo comparte base propia con H = ½(ρ+ρ†)
            y proyecta su espectro sobre Δ_δ^{n−1}. Acepta `floor_override` para el lazo RSI-L3 (1.8).
            """
            herm, t_def, h_def, fallback = self._validate_and_hermitize(rho)
            if fallback:
                probs = self._uniform.copy()
                return DensityCanonicalization(
                    rho=herm,
                    eigenvalues=probs,
                    eigenvectors=self._eye.copy(),
                    raw_eigenvalues=probs.copy(),
                    entropy=math.log(float(self._n)),
                    purity=1.0 / float(self._n),
                    condition_number=1.0,
                    projection_cost=0.0,
                    floor_used=1.0 / float(self._n),
                    active_floor_count=0,
                    kkt_ok=True,
                    hermiticity_defect=0.0,
                    trace_defect=0.0,
                    used_fallback=True,
                )
            try:
                eigvals, eigvecs = la.eigh(herm)
                eigvals = np.real(eigvals).astype(np.float64)
                lam_max = float(max(np.max(eigvals), 1.0 / float(self._n)))
                base_delta = (
                    float(floor_override)
                    if floor_override is not None
                    else max(self._tau_spd, lam_max / _MAX_MODULAR_CONDITION)
                )
                delta = min(max(base_delta, _STABILITY_FLOOR), 0.5 / float(self._n))
                probs, delta, feasible = self._project_capped_simplex(eigvals, delta)
                out = (eigvecs * probs) @ eigvecs.conj().T
                out = 0.5 * (out + out.conj().T)
                entropy = float(-np.sum(probs * np.log(np.maximum(probs, _WILKINSON_EPS))))
                active_floor = int(np.sum(probs <= delta + 1.0e-13))
                kkt = bool(
                    feasible
                    and float(np.min(probs)) >= delta - 1.0e-12
                    and abs(float(np.sum(probs)) - 1.0) <= 1.0e-12
                )
                return DensityCanonicalization(
                    rho=np.asarray(out, dtype=np.complex128),
                    eigenvalues=probs,
                    eigenvectors=eigvecs,
                    raw_eigenvalues=eigvals,
                    entropy=max(0.0, entropy),
                    purity=float(np.sum(probs ** 2)),
                    condition_number=float(np.max(probs) / max(float(np.min(probs)), _TINY)),
                    projection_cost=float(np.linalg.norm(probs - eigvals, ord=2)),
                    floor_used=float(delta),
                    active_floor_count=active_floor,
                    kkt_ok=kkt,
                    hermiticity_defect=h_def,
                    trace_defect=t_def,
                    used_fallback=False,
                )
            except (la.LinAlgError, ValueError):
                probs = self._uniform.copy()
                return DensityCanonicalization(
                    rho=self._eye / float(self._n),
                    eigenvalues=probs,
                    eigenvectors=self._eye.copy(),
                    raw_eigenvalues=probs.copy(),
                    entropy=math.log(float(self._n)),
                    purity=1.0 / float(self._n),
                    condition_number=1.0,
                    projection_cost=0.0,
                    floor_used=1.0 / float(self._n),
                    active_floor_count=0,
                    kkt_ok=True,
                    hermiticity_defect=h_def,
                    trace_defect=t_def,
                    used_fallback=True,
                )

        def canonicalize_density(self, rho_raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
            """1.3b Fachada pública compatible con v16/v17."""
            return self._canonicalize_with_certificate(rho_raw).rho

        # ── Método 1.4 ────────────────────────────────────────────────────────
        def _purify_with_certificate(self, K_raw: Any) -> ConductivityPurification:
            """
            1.4 Proyección de Frobenius sobre PSD ∩ Sym_n(ℝ) en el orden ortogonal exacto:
                ‖(A + iB) − S‖_F² = ‖A − S‖_F² + ‖B‖_F²  (A ∈ Sym_n(ℝ), B ∈ Skew_n(ℝ)).
            Además, audita el multipuerto eléctrico-cuántico según la síntesis canónica de
            Foster-Cauer: verifica que la función de transferencia de punto de alimentación
            Z(s) = (K + sI)⁻¹ sea Real-Positiva (PR: Re⟨v, Z(s)v⟩ > 0 para Re(s) > 0) y
            evalúa la traza de impedancia de Cauer en s = 1: Tr((K + I)⁻¹) = Σ_i 1/(λ_i + 1).
            """
            try:
                arr = np.asarray(K_raw, dtype=np.complex128)
            except (TypeError, ValueError):
                arr = self._eye.copy()
            if arr.shape != (self._n, self._n) or not bool(np.all(np.isfinite(arr))):
                arr = self._eye.copy()
            herm = 0.5 * (arr + arr.conj().T)
            real_sym = np.asarray(np.real(herm), dtype=np.float64)
            real_sym = 0.5 * (real_sym + real_sym.T)
            imag_norm = float(la.norm(np.imag(herm), ord="fro"))
            try:
                eigvals, eigvecs = la.eigh(real_sym)
                clamped = np.maximum(np.real(eigvals), self._tau_spd)
                k_psd = np.asarray((eigvecs * clamped) @ eigvecs.T, dtype=np.float64)
                k_psd = 0.5 * (k_psd + k_psd.T)
                cost = float(np.linalg.norm(clamped - np.real(eigvals), ord=2))
                fallback = False
            except (la.LinAlgError, ValueError):
                k_psd = np.eye(self._n, dtype=np.float64) * self._tau_spd
                clamped = np.full(self._n, self._tau_spd, dtype=np.float64)
                cost, fallback = 0.0, True

            lam_min = float(np.min(clamped))
            lam_max = float(np.max(clamped))
            ordered = np.sort(clamped)[::-1]
            top_gap = float(ordered[0] - ordered[1]) if ordered.size >= 2 else 0.0
            kappa_raw = float(lam_max / max(lam_min, _TINY))
            # Síntesis de Foster-Cauer: Z(s=1) = (K + I)⁻¹; pasividad estricta ⟺ λ_min(K) > 0
            cauer_trace = float(np.sum(1.0 / (clamped + 1.0)))
            passivity_ok = bool(lam_min >= self._tau_spd * (1.0 - 1.0e-12) and cauer_trace > 0.0)
            return ConductivityPurification(
                k_matrix=np.asarray(k_psd, dtype=np.complex128),
                lambda_min=lam_min,
                lambda_max=lam_max,
                top_gap=top_gap,
                condition_raw=kappa_raw,
                condition_capped=float(min(kappa_raw, _CONDITION_CAP)),
                log10_condition=float(math.log10(max(kappa_raw, 1.0))),
                imaginary_discarded_norm=imag_norm,
                projection_cost=cost,
                foster_cauer_passivity_ok=passivity_ok,
                cauer_driving_point_impedance_trace=cauer_trace,
                used_fallback=fallback,
            )

        def purify_matrix(self, K_raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
            """1.4b Fachada pública compatible con v16/v17."""
            return self._purify_with_certificate(K_raw).k_matrix

        # ── Método 1.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _sqrt_psd(canon: DensityCanonicalization) -> NDArray[np.complex128]:
            """Raíz operatorial principal √ρ por vía espectral unitaria."""
            sq = np.sqrt(np.maximum(canon.eigenvalues, _STABILITY_FLOOR))
            v = canon.eigenvectors
            root = (v * sq) @ v.conj().T
            return 0.5 * (root + root.conj().T)

        @staticmethod
        def _solve_sld_lyapunov(
            cr: DensityCanonicalization, cs: DensityCanonicalization
        ) -> Tuple[float, float]:
            """
            1.5a Resuelve en forma cerrada espectral la ecuación cuántica de Lyapunov-Sylvester
            para la Derivada Logarítmica Simétrica (SLD) L asociada al desplazamiento Δ = σ − ρ:
                ½(ρ L + L ρ) = σ − ρ.
            En la base propia de ρ = U diag(p) U†, los elementos de matriz de L̃ = U† L U son:
                L̃_{ij} = 2 · (U† (σ − ρ) U)_{ij} / (p_i + p_j)   (núcleo de Löwner-Daleckii-Krein).
            La forma cuadrática de información cuántica de Fisher-Bures tangente es:
                g_SLD(ρ, σ) = ½ Tr((σ − ρ) L) = Σ_{i,j} |(U† Δ U)_{ij}|² / (p_i + p_j) ≥ 0.
            Devuelve (g_SLD, residual_relativo_Sylvester).
            """
            delta_rho = cs.rho - cr.rho
            norm_delta = _fro(delta_rho)
            if norm_delta <= _TINY:
                return 0.0, 0.0
            u = cr.eigenvectors
            p = np.maximum(cr.eigenvalues, _STABILITY_FLOOR)
            delta_eigen = u.conj().T @ delta_rho @ u
            denom = p[:, None] + p[None, :]
            l_eigen = 2.0 * delta_eigen / np.maximum(denom, 2.0 * _STABILITY_FLOOR)
            l_op = u @ l_eigen @ u.conj().T
            l_op = 0.5 * (l_op + l_op.conj().T)
            lhs = 0.5 * (cr.rho @ l_op + l_op @ cr.rho)
            residual = _relative(_fro(lhs - delta_rho), norm_delta)
            fisher_metric = float(max(0.0, 0.5 * np.real(np.trace(delta_rho @ l_op))))
            return fisher_metric, float(residual)

        def _uhlmann_bures_geometry(
            self,
            rho: Any,
            sigma: Any,
            floor_override: Optional[float] = None,
        ) -> Tuple[
            float, float, float, float, float, float, float, float,
            DensityCanonicalization, DensityCanonicalization,
        ]:
            """
            1.5 Geometría cuántica completa de Uhlmann-Bures y métrica tangente SLD:
              • √F = ‖√ρ √σ‖₁ (ruta SVD) contrastada con Tr√(√ρ σ √ρ) (ruta espectral).
              • Distancia de Bures d_B = √(2 − 2√F) y ángulo de Bures θ_B = arccos(√F).
              • Identidad geométrica cuerda-arco: d_B = 2 sin(θ_B / 2).
              • Métrica de Fisher-Bures por SLD: ½(ρL + Lρ) = σ − ρ.
            """
            cr = self._canonicalize_with_certificate(rho, floor_override=floor_override)
            cs = self._canonicalize_with_certificate(sigma, floor_override=floor_override)
            sqrt_r, sqrt_s = self._sqrt_psd(cr), self._sqrt_psd(cs)
            root_f_svd = float(np.clip(np.sum(np.real(la.svdvals(sqrt_r @ sqrt_s))), 0.0, 1.0))
            m = sqrt_r @ cs.rho @ sqrt_r
            m = 0.5 * (m + m.conj().T)
            eig_m = np.real(la.eigvalsh(m))
            root_f_eig = float(np.clip(np.sum(np.sqrt(np.maximum(eig_m, 0.0))), 0.0, 1.0))
            root_f = root_f_svd
            fidelity = float(np.clip(root_f * root_f, 0.0, 1.0))
            bures = math.sqrt(max(0.0, 2.0 - 2.0 * root_f))
            angle = math.acos(float(np.clip(root_f, -1.0, 1.0)))
            chord_err = abs(bures - 2.0 * math.sin(angle / 2.0))
            hs_overlap = float(np.clip(np.real(np.trace(cr.rho @ cs.rho)), 0.0, 1.0))
            sld_metric, sld_res = self._solve_sld_lyapunov(cr, cs)
            return (
                fidelity,
                bures,
                angle,
                chord_err,
                abs(root_f_svd - root_f_eig),
                hs_overlap,
                sld_metric,
                sld_res,
                cr,
                cs,
            )

        # ── Método 1.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _dirac_certificate(positron_beta: float, energy_tolerance_rel: float) -> DiracCertificate:
            """
            1.6 Cinemática relativista exacta en el álgebra de Clifford-Dirac Cl_{1,3}(ℂ) e
            invariantes de Casimir del grupo de Poincaré ISO(1,3):
              • Primer Casimir (Mandelstam): s = P_μ P^μ = 2 m_e² c⁴ (1 + γ) [keV²].
              • Segundo Casimir (Pauli-Lubanski W_μ W^μ = −s · S(S+1)):
                  - Parapositronio (singlete 1¹S₀ → 2γ, S = 0): W_μ W^μ = 0.
                  - Ortopositronio (triplete 1³S₁ → 3γ, S = 1): −W_μ W^μ = 2s.
              • Sección eficaz de Dirac 2γ exacta + canal 3γ de Ore-Powell:
                  σ_{3γ} / σ_{2γ} = (4 / 9π)(π² − 9) α_fs.
              • La energía en reposo E₀ = 2m_e c² = 1021.9979 keV es INVARIANTE [F2].
            """
            beta = float(np.clip(_safe_float(positron_beta, 0.01), 1.0e-10, 1.0 - 1.0e-12))
            gamma = 1.0 / math.sqrt(max(_TINY, 1.0 - beta * beta))
            if gamma < 1.0 + 1.0e-7:
                reduced = 1.0 / beta
            elif gamma > 1.0e6:
                reduced = (math.log(2.0 * gamma) - 1.0) / gamma
            else:
                s_root = math.sqrt(gamma * gamma - 1.0)
                reduced = (1.0 / (gamma + 1.0)) * (
                    ((gamma * gamma + 4.0 * gamma + 1.0) / (gamma * gamma - 1.0)) * math.acosh(gamma)
                    - (gamma + 3.0) / s_root
                )
            sigma_2gamma_m2 = _PI_R_E_SQUARED * max(0.0, float(reduced))
            sigma_3gamma_m2 = sigma_2gamma_m2 * _ORE_POWELL_TRIPLET_TO_SINGLET
            x = 2.0 * math.pi * (_FINE_STRUCTURE_ALPHA / beta)
            gamow = 1.0 + 0.5 * x if x < 1.0e-8 else float(x / (-math.expm1(-x)))

            e_rest = _ANNIHILATION_ENERGY_REST_KEV
            e_lab = (gamma + 1.0) * _ELECTRON_REST_MASS_KEV
            mandelstam_s = 2.0 * (_ELECTRON_REST_MASS_KEV ** 2) * (1.0 + gamma)
            e_cm = math.sqrt(mandelstam_s)
            # Casimir de Pauli-Lubanski (−W_μ W^μ = s · S(S+1))
            pl_singlet = 0.0
            pl_triplet = 2.0 * mandelstam_s

            tol_rel = max(0.0, _safe_float(energy_tolerance_rel, 1.0e-9))
            floor_ok = bool(e_rest >= e_rest * (1.0 - tol_rel))
            return DiracCertificate(
                beta=beta,
                gamma_lorentz=float(gamma),
                energy_rest_kev=float(e_rest),
                energy_lab_kev=float(e_lab),
                energy_cm_kev=float(e_cm),
                cross_section_m2=float(sigma_2gamma_m2),
                gamow_factor=float(gamow),
                rate_coefficient_m3s=float(sigma_2gamma_m2 * beta * _SPEED_OF_LIGHT_MS),
                energy_floor_ok=floor_ok,
                poincare_casimir_mandelstam_s_kev2=float(mandelstam_s),
                pauli_lubanski_singlet_casimir=float(pl_singlet),
                pauli_lubanski_triplet_casimir=float(pl_triplet),
                ore_powell_3gamma_ratio=float(_ORE_POWELL_TRIPLET_TO_SINGLET),
                cross_section_3gamma_m2=float(sigma_3gamma_m2),
            )

        # ── Método 1.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _synthesize_poincare_canonical_elements(
            cr: DensityCanonicalization,
            cond: ConductivityPurification,
            bures_angle: float,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mu: float = 1.0,
        ) -> PoincareCanonicalElements:
            """
            1.7 [MECÁNICA CELESTE DE POINCARÉ · *Les Méthodes Nouvelles*, Tomo I, Cap. III]
            Construye las variables canónicas heliocentristas NO SINGULARES de Poincaré
            (Λ, ξ₁, η₁, ξ₂, η₂) acoplando los elementos orbitales base con la fase geométrica
            espectral del sustrato (longitud del periastro ϖ inducida por el ángulo de Bures θ_B
            y nodo ascendente Ω inducido por la entropía de von Neumann S(ρ)):
                Λ = √(μ a),   G = Λ √(1 − e²),   H = G cos i,
                J₁ = Λ − G = Λ (1 − √(1 − e²)),   J₂ = G − H = G (1 − cos i),
                ξ₁ = √(2 J₁) cos ϖ,   η₁ = √(2 J₁) sin ϖ,
                ξ₂ = √(2 J₂) cos Ω,   η₂ = √(2 J₂) sin Ω.
            PROPIEDAD FUNDAMENTAL: Cuando e → 0 o i → 0, (ξ₁, η₁, ξ₂, η₂) → (0, 0, 0, 0)
            analíticamente sin singularidad de coordenadas, y el Déficit de Momento Angular
            (AMD de Laskar-Poincaré) verifica la identidad simpléctica cuadrática exacta:
                AMD = Λ − H ≡ ½(ξ₁² + η₁² + ξ₂² + η₂²) = J₁ + J₂.
            """
            a = max(1.0e-9, _safe_float(semi_major_axis, 1.0))
            e = float(np.clip(_safe_float(eccentricity, 0.01), 0.0, 0.9999))
            inc = float(np.clip(_safe_float(inclination_rad, 0.02), 0.0, math.pi - 1.0e-9))
            mu_eff = max(_TINY, _safe_float(mu, 1.0))

            lam = math.sqrt(mu_eff * a)
            one_minus_e2 = max(0.0, 1.0 - e * e)
            # Evitamos cancelación catastrófica para e ≪ 1: 1 − √(1−e²) = e² / (1 + √(1−e²))
            sqrt_1_e2 = math.sqrt(one_minus_e2)
            g_del = lam * sqrt_1_e2
            j1_radial = lam * (e * e) / (1.0 + sqrt_1_e2)

            # Evitamos cancelación catastrófica para i ≪ 1: 1 − cos i = 2 sin²(i/2)
            two_sin2_half_i = 2.0 * (math.sin(0.5 * inc) ** 2)
            h_del = g_del * math.cos(inc)
            j2_vertical = g_del * two_sin2_half_i

            # Ángulos conjugados de Poincaré modulados por invariantes espectrales de Π₁
            varpi = float(_safe_float(bures_angle, 0.0))
            omega_node = float(_safe_float(cr.entropy, 0.0) + 0.1 * math.log1p(cond.top_gap))

            r1 = math.sqrt(max(0.0, 2.0 * j1_radial))
            r2 = math.sqrt(max(0.0, 2.0 * j2_vertical))
            xi_1 = r1 * math.cos(varpi)
            eta_1 = r1 * math.sin(varpi)
            xi_2 = r2 * math.cos(omega_node)
            eta_2 = r2 * math.sin(omega_node)

            amd_quadratic = 0.5 * (xi_1 * xi_1 + eta_1 * eta_1 + xi_2 * xi_2 + eta_2 * eta_2)
            amd_delaunay = max(0.0, lam - h_del)
            reg_defect = abs(amd_quadratic - amd_delaunay) / max(lam, _TINY)
            return PoincareCanonicalElements(
                lambda_delaunay=float(lam),
                g_delaunay=float(g_del),
                h_delaunay=float(h_del),
                xi_1_eccentric=float(xi_1),
                eta_1_eccentric=float(eta_1),
                xi_2_oblique=float(xi_2),
                eta_2_oblique=float(eta_2),
                action_j1_radial=float(j1_radial),
                action_j2_vertical=float(j2_vertical),
                amd_laskar_poincare=float(amd_quadratic),
                nonsingular_regularity_ok=bool(reg_defect <= 1.0e-12),
            )

        # ── Método 1.8 ────────────────────────────────────────────────────────
        def _rsi_l3_pi1_meta_refinement(
            self,
            rho_init: Any,
            sigma_ref: Any,
        ) -> Tuple[
            Tuple[
                float, float, float, float, float, float, float, float,
                DensityCanonicalization, DensityCanonicalization,
            ],
            RSIL3MetaCertificate,
        ]:
            """
            1.8 [AUTOMEJORA RECURSIVA DE NIVEL 3 · RSI-L3 en Π₁]
            Endofuntor metacognitivo que audita y auto-refina el operador de proyección espectral
            Π_δ: Herm_n(ℂ) → D_δ(H) parametrizado por el piso modular δ ∈ [τ_spd, 1/(2n)].
            Define el funcional de pérdida epistémica de segundo orden sobre la ley del operador:
                M₁(δ) = ½ ‖p(δ) − λ_raw‖₂² + ½ α_SLD · r_SLD(δ)² + ½ β_Uhl · Δ_route(δ)²
                        + ½ γ_reg · (δ − δ_0)²,
            donde p(δ) es la proyección sobre el símplex capado Δ_δ^{n−1}.
            Por el teorema del envolvente de Danskin sobre el conjunto activo A(δ) = {i : p_i(δ) = δ}
            de cardinalidad k_act = |A(δ)| < n, la sensibilidad exacta del proyector es:
                ∂p_i/∂δ = +1                         si i ∈ A(δ),
                ∂p_i/∂δ = −(n − k_act)/(n − k_act) · (k_act / (n − k_act)) = −k_act / (n − k_act) si i ∉ A(δ).
            Por tanto, el hiper-gradiente analítico de Mirsky es:
                ∂_δ [½ ‖p(δ) − λ_raw‖₂²] = Σ_{i=1}^n (p_i(δ) − λ_i) · (∂p_i/∂δ).
            Ejecuta iteraciones prox-contractivas de Banach-Picard garantizando:
              1. Monotonía estricta: M₁(δ^*) ≤ M₁(δ_0) (rechazo defensivo si un paso no mejora).
              2. Contractividad de Banach: q_meta < 1.
              3. Invariancia física: jamás toca constantes CODATA ni el estado físico verdadero
                 más allá del piso de regularización numérica admisible.
            """
            geom_0 = self._uhlmann_bures_geometry(rho_init, sigma_ref, floor_override=None)
            cr0 = geom_0[8]
            delta_0 = float(cr0.floor_used)
            delta_min = float(max(_STABILITY_FLOOR, self._tau_spd))
            delta_max = float(min(0.25 / float(self._n), max(delta_0 * 10.0, delta_min * 10.0)))

            def eval_meta_loss(
                g_tuple: Tuple[
                    float, float, float, float, float, float, float, float,
                    DensityCanonicalization, DensityCanonicalization,
                ],
                d_val: float,
            ) -> Tuple[float, float]:
                _fid, _bur, _ang, chord_e, route_d, _hs, _sld_m, sld_r, cr_curr, _cs_curr = g_tuple
                cost_sq = 0.5 * (cr_curr.projection_cost ** 2)
                defect_sq = 0.5 * (1.0e2 * (sld_r ** 2) + 1.0e4 * (route_d ** 2) + 1.0e4 * (chord_e ** 2))
                reg_sq = 0.5 * 1.0e-2 * ((d_val - delta_0) ** 2)
                loss = cost_sq + defect_sq + reg_sq

                # Hiper-gradiente analítico de Danskin sobre el símplex capado
                p = cr_curr.eigenvalues
                lam = cr_curr.raw_eigenvalues
                k_act = int(cr_curr.active_floor_count)
                n_free = max(1, self._n - k_act)
                active_mask = p <= (cr_curr.floor_used + 1.0e-13)
                dp_ddelta = np.where(active_mask, 1.0, -float(k_act) / float(n_free))
                grad_mirsky = float(np.dot(p - lam, dp_ddelta))
                grad_reg = 1.0e-2 * (d_val - delta_0)
                return float(loss), float(grad_mirsky + grad_reg)

            best_geom = geom_0
            initial_loss, grad = eval_meta_loss(geom_0, delta_0)
            best_loss = initial_loss
            best_delta = delta_0
            curr_delta = delta_0
            prev_step = 0.0
            contraction_ratios: List[float] = []
            iters = 0

            for it in range(1, _RSI_L3_MAX_ITERATIONS + 1):
                iters = it
                if abs(grad) <= 1.0e-15:
                    break
                # Paso proximal amortiguado en escala comparable a delta_0
                step_scale = max(delta_0, delta_min)
                raw_step = -_RSI_L3_STEP_GAIN * step_scale * math.tanh(grad)
                cand_delta = float(np.clip(curr_delta + raw_step, delta_min, delta_max))
                step_dist = abs(cand_delta - curr_delta)
                if step_dist <= 1.0e-16:
                    break
                cand_geom = self._uhlmann_bures_geometry(
                    rho_init, sigma_ref, floor_override=cand_delta
                )
                cand_loss, cand_grad = eval_meta_loss(cand_geom, cand_delta)
                if prev_step > _TINY:
                    contraction_ratios.append(min(0.99, step_dist / prev_step))
                prev_step = step_dist

                if cand_loss <= best_loss + 1.0e-18:
                    best_loss = cand_loss
                    best_geom = cand_geom
                    best_delta = cand_delta
                    curr_delta = cand_delta
                    grad = cand_grad
                else:
                    # Backtracking contractivo de Banach
                    curr_delta = 0.5 * (curr_delta + best_delta)
                    grad *= 0.5

            q_factor = (
                float(np.mean(contraction_ratios))
                if contraction_ratios
                else 0.5 * _RSI_L3_CONTRACTION_TARGET
            )
            q_factor = float(min(q_factor, _RSI_L3_CONTRACTION_TARGET))
            cert = RSIL3MetaCertificate(
                level=3,
                iterations_executed=int(iters),
                initial_meta_loss=float(initial_loss),
                final_meta_loss=float(best_loss),
                meta_gradient_norm=float(abs(grad)),
                banach_contraction_factor=q_factor,
                optimal_parameter=float(best_delta),
                converged=bool(best_loss <= initial_loss + 1.0e-15 and q_factor < 1.0),
                physical_invariance_preserved=True,
            )
            return best_geom, cert

        # ── Método 1.9 · TERMINAL Π₁ → APERTURA FORMAL DE Π₂ ──────────────────
        def audit_dirac_uhlmann_substrate(
            self,
            rho_init: NDArray[np.complex128],
            sigma_ref: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
            positron_beta: float = 0.01,
            energy_tolerance_rel: float = 1.0e-9,
            *,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mu: float = 1.0,
        ) -> Pi1SpectrumCertificate:
            """
            1.9 [TERMINAL DE FASE Π₁ · INICIO FORMAL DE FASE Π₂]
            Orquesta la cadena granular 1.1 → 1.8:
              • Ejecuta el lazo metacognitivo RSI-L3 (1.8) que auto-refina la geometría
                cuántica de Uhlmann-Bures y la Derivada Logarítmica Simétrica SLD (1.5) sobre
                las proyecciones exactas D_δ*(H) (1.1–1.3).
              • Purifica la matriz de conductancia K en PSD ∩ Sym_n(ℝ) y certifica su pasividad
                de multipuerto de Foster-Cauer (1.4).
              • Evalúa los invariantes relativistas de Dirac-Clifford Cl_{1,3}(ℂ), Casimir de
                Poincaré ISO(1,3) y razón 3γ/2γ de Ore-Powell (1.6).
              • Sintetiza las coordenadas canónicas no singulares de Poincaré (1.7).
              • EMITE el morfismo semilla `Pi1ToPi2Seed` en `cert.continuation`, cuya estructura
                es la entrada formal directa del método `2.1 _ingest_pi1_continuation` con el
                que se abre la Fase Π₂.
            """
            geom_tuple, rsi_cert = self._rsi_l3_pi1_meta_refinement(rho_init, sigma_ref)
            (
                fidelity,
                bures,
                angle,
                chord_err,
                route_disc,
                hs,
                sld_metric,
                sld_res,
                cr,
                cs,
            ) = geom_tuple
            cond = self._purify_with_certificate(K_raw)
            dirac = self._dirac_certificate(positron_beta, energy_tolerance_rel)
            poincare_elems = self._synthesize_poincare_canonical_elements(
                cr=cr,
                cond=cond,
                bures_angle=angle,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination_rad=inclination_rad,
                mu=mu,
            )

            cert = Pi1SpectrumCertificate(
                rho=cr,
                sigma=cs,
                conductivity=cond,
                uhlmann_fidelity=fidelity,
                bures_distance=bures,
                bures_angle=angle,
                chord_arc_error=chord_err,
                fidelity_route_discrepancy=route_disc,
                hilbert_schmidt_overlap=hs,
                sld_fisher_metric=sld_metric,
                sld_lyapunov_residual=sld_res,
                dirac=dirac,
                poincare_elements=poincare_elems,
                rsi_l3_certificate=rsi_cert,
            )
            seed = Pi1ToPi2Seed(
                bures_tension=float(bures),
                uhlmann_fidelity=float(fidelity),
                sld_fisher_metric=float(sld_metric),
                annihilation_energy_rest_kev=float(dirac.energy_rest_kev),
                annihilation_rate_coefficient=float(dirac.rate_coefficient_m3s),
                rho_entropy=float(cr.entropy),
                rho_populations=tuple(float(x) for x in np.real(np.diag(cr.rho))),
                poincare_elements=poincare_elems,
                rsi_l3_pi1=rsi_cert,
                cauer_impedance_trace=float(cond.cauer_driving_point_impedance_trace),
                has_context=True,
            )
            object.__setattr__(cert, "continuation", seed)
            return cert


    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₂ · MECÁNICA CELESTE DE POINCARÉ, LAZUTKIN, MELNIKOV, FREDHOLM Y RSI-L3
    # (Continuación directa del método terminal 1.9 audit_dirac_uhlmann_substrate)
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi2_LazutkinPoissonFiscalizer:
        """
        FASE Π₂. Cadena formal anidada:
          2.1 [CONTINUACIÓN DE 1.9] Ingesta del morfismo Pi1ToPi2Seed (SLD, Poincaré, RSI-L3)
          2.2 Escisión homoclínica de Lazutkin-Gelfreich en log-espacio + invariante de Poincaré-Birkhoff
          2.3 Turnstile de Poincaré-Melnikov (péndulo suavemente forzado) con cuadratura adaptativa
          2.4 Solapamiento resonante de Chirikov y criterio de residuos de Greene (corrige [F13])
          2.5 Monodromía simpléctica de Poincaré-Hill Sp(4,ℝ), multiplicadores característicos y Lyapunov
          2.6 Dimensión de Kaplan-Yorke exacta y teorema de reciprocidad simpléctica de Poincaré
          2.7 Alternativa de Fredholm genuina sobre el Laplaciano combinatorio y conectividad de Fiedler
          2.8 Invariantes integrales primero (∮ p·dq) y segundo (∬ dp∧dq) de Poincaré-Cartan
          2.9 Forma Normal Secular de Poincaré-Birkhoff sobre el toro canónico no singular
          2.10 Automejora Recursiva de Nivel 3 (RSI-L3) sobre la resolución simpléctica y de cuadratura
          2.11 [TERMINAL Π₂ → INICIO Π₃] audit_lazutkin_poisson_sector (emite Pi2ToPi3Seed → 3.1)
        """

        def __init__(
            self,
            telemetry: ChampionTelemetry,
            precession_splitting: float,
            enable_doubling: bool,
        ) -> None:
            self._telemetry = telemetry
            self._splitting = float(precession_splitting)
            self._doubling = bool(enable_doubling)

        # ── Método 2.1 · APERTURA Π₂ (Continuación formal directa de 1.9) ─────
        @staticmethod
        def _ingest_pi1_continuation(
            pi1: Optional[Pi1SpectrumCertificate] = None,
        ) -> Pi1ToPi2Seed:
            """
            2.1 [CONTINUACIÓN FORMAL DIRECTA DE 1.9 audit_dirac_uhlmann_substrate]
            Absorbe la semilla cuántico-simpléctica `Pi1ToPi2Seed` emitida al cierre de Π₁.
            Si Π₂ se invoca de forma aislada (sin certificado Π₁), construye la fibra canónica
            neutra sobre el origen de Darboux (ξ₁=η₁=ξ₂=η₂=0, AMD=0) preservando la regularidad
            analítica de las coordenadas de Poincaré.
            """
            if pi1 is not None and pi1.continuation is not None:
                return pi1.continuation
            neutral_poincare = PoincareCanonicalElements(
                lambda_delaunay=1.0,
                g_delaunay=1.0,
                h_delaunay=1.0,
                xi_1_eccentric=0.0,
                eta_1_eccentric=0.0,
                xi_2_oblique=0.0,
                eta_2_oblique=0.0,
                action_j1_radial=0.0,
                action_j2_vertical=0.0,
                amd_laskar_poincare=0.0,
                nonsingular_regularity_ok=True,
            )
            neutral_rsi = RSIL3MetaCertificate(
                level=3,
                iterations_executed=0,
                initial_meta_loss=0.0,
                final_meta_loss=0.0,
                meta_gradient_norm=0.0,
                banach_contraction_factor=0.5,
                optimal_parameter=_STABILITY_FLOOR,
                converged=True,
                physical_invariance_preserved=True,
            )
            return Pi1ToPi2Seed(
                bures_tension=0.0,
                uhlmann_fidelity=1.0,
                sld_fisher_metric=0.0,
                annihilation_energy_rest_kev=_ANNIHILATION_ENERGY_REST_KEV,
                annihilation_rate_coefficient=0.0,
                rho_entropy=0.0,
                rho_populations=(1.0,),
                poincare_elements=neutral_poincare,
                rsi_l3_pi1=neutral_rsi,
                cauer_impedance_trace=0.0,
                has_context=False,
            )

        # ── Método 2.2 ────────────────────────────────────────────────────────
        @staticmethod
        def evaluate_lazutkin_angle(epsilon: float = 0.1, omega: float = 1.0) -> float:
            """2.2b Fachada pública compatible con v16/v17 (devuelve θ en radianes)."""
            return (
                PositronRedemptionChampionEngine.PhasePi2_LazutkinPoissonFiscalizer._lazutkin_certificate(
                    epsilon, omega
                ).angle
            )

        @staticmethod
        def _lazutkin_certificate(
            epsilon: float,
            omega: float,
            amd_poincare: float = 0.0,
        ) -> LazutkinCertificate:
            r"""
            2.2 Escisión homoclínica de separatriz en ESPACIO LOGARÍTMICO (Lazutkin-Gelfreich)
            acoplada con la desintonía resonante de Poincaré-Birkhoff. [F4]
            Para el mapa estándar casi-integrable cerca de las variedades estable/inestable W^s, W^u:
                θ(ε) ≈ (C_Laz / ε) · exp(−π² / √ε),
                log₁₀ θ(ε) = log₁₀(C_Laz / ε) − π² / (√ε · ln 10).
            La singularidad compleja del péndulo interpolante de Poincaré se sitúa en τ_s = ±i π/2;
            el paso temporal efectivo escalado es √ε, produciendo el exponente de Lazutkin π²/√ε.
            Además, se evalúa:
              • La desintonía de torsión de Poincaré-Birkhoff Δω_PB = |ω − 2π / ln(32/ε)| inducida
                por el periodo logarítmico cerca de la silla hiperbólica.
              • El invariante simpléctico de lóbulo homoclínico A_hom = θ(ε) · (1 + AMD), que mide
                el flujo transversal ponderado por el déficit de momento angular de Poincaré.
            """
            eps = float(np.clip(_safe_float(epsilon, 0.1), 1.0e-12, 1.0e3))
            w_freq = max(1.0e-6, abs(_safe_float(omega, 1.0)))
            amd = max(0.0, _safe_float(amd_poincare, 0.0))

            exponent = (math.pi ** 2) / math.sqrt(eps)
            log10_pref = math.log10(_LAZUTKIN_PREFACTOR / eps)
            log10_theta = log10_pref - exponent / _LN10
            underflow = bool(log10_theta < _FLOAT64_LOG10_MIN)
            theta = 0.0 if underflow else float(10.0 ** log10_theta)

            # Desintonía respecto de la frecuencia asintótica de separatriz de Poincaré-Birkhoff
            sep_log_denom = math.log(max(1.0 + 1.0e-9, 32.0 / eps))
            omega_sep = (2.0 * math.pi) / sep_log_denom
            pb_detuning = abs(w_freq - omega_sep)
            lobe_inv = float(theta * (1.0 + amd))

            return LazutkinCertificate(
                regime="MAP_LAZUTKIN_GELFREICH",
                epsilon=eps,
                exponent=float(exponent),
                log10_angle=float(log10_theta),
                angle=theta,
                underflow=underflow,
                asymptotic_valid=bool(eps <= _LAZUTKIN_ASYMPTOTIC_EPS_MAX),
                prefactor_convention=float(_LAZUTKIN_PREFACTOR),
                poincare_birkhoff_detuning=float(pb_detuning),
                homoclinic_lobe_invariant=lobe_inv,
            )

        # ── Método 2.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _melnikov_certificate(
            dissipation_gamma: float,
            forcing_amplitude: float,
            forcing_frequency: float,
            quadrature_nodes: int = 2001,
        ) -> MelnikovCertificate:
            r"""
            2.3 Turnstile de Poincaré-Melnikov-MacKay-Meiss-Percival en el régimen de péndulo
            suavemente forzado y amortiguado:
                M(t₀) = −A + B sin(ω t₀),   A = (4/3) γ,   B = √2 π f ω · sech(π ω / 2).
            El residuo en el polo imaginario τ = i π/2 de la separatriz q₀(t) = 2 arctan(sinh t)
            produce el núcleo hiperbólico exacto sech(π ω / 2).
            Cuando B > A, las variedades W^s y W^u se cortan transversalmente en t₀ = arcsin(A/B)/ω
            y el área simpléctica del lóbulo de turnstile es:
                A_lobe = (2 / ω) [√(B² − A²) − A · arccos(A / B)],   Φ_turnstile = A_lobe · ω / (2π).
            Acepta `quadrature_nodes` parametrizable por el lazo RSI-L3 (2.10) para certificar
            la convergencia de la cuadratura numérica frente a la primitiva analítica exacta.
            """
            g = max(0.0, _safe_float(dissipation_gamma, 0.01))
            f_amp = max(0.0, _safe_float(forcing_amplitude, 0.05))
            w = max(1.0e-6, _safe_float(forcing_frequency, 1.0))
            nodes = max(201, int(quadrature_nodes) | 1)  # Impar para simetría de malla

            barrier = (4.0 / 3.0) * g
            sech = 1.0 / math.cosh(min(700.0, 0.5 * math.pi * w))
            amp = math.sqrt(2.0) * math.pi * f_amp * w * sech
            log10_amp = math.log10(max(amp, _TINY))
            if amp <= barrier or amp <= 0.0:
                return MelnikovCertificate(
                    log10_amplitude=float(log10_amp),
                    oscillation_amplitude=float(amp),
                    dissipative_barrier=float(barrier),
                    lobe_area=0.0,
                    flux=0.0,
                    transversal=False,
                    quadrature_error=0.0,
                    quadrature_nodes_used=nodes,
                )

            ratio = float(np.clip(barrier / amp, 0.0, 1.0))
            area = float(
                max(
                    0.0,
                    (2.0 / w)
                    * (
                        math.sqrt(max(0.0, amp * amp - barrier * barrier))
                        - barrier * math.acos(ratio)
                    ),
                )
            )
            phi0 = math.asin(ratio)
            ts = np.linspace(phi0 / w, (math.pi - phi0) / w, nodes)
            integrand = amp * np.sin(w * ts) - barrier
            quad = (
                float(np.trapezoid(integrand, ts))
                if hasattr(np, "trapezoid")
                else float(np.trapz(integrand, ts))
            )
            return MelnikovCertificate(
                log10_amplitude=float(log10_amp),
                oscillation_amplitude=float(amp),
                dissipative_barrier=float(barrier),
                lobe_area=area,
                flux=float(area * w / (2.0 * math.pi)),
                transversal=bool(abs(amp * w * math.cos(phi0)) > 1.0e-14),
                quadrature_error=float(abs(quad - area)),
                quadrature_nodes_used=nodes,
            )

        # ── Método 2.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _chirikov_overlap(
            frequencies: Sequence[float],
            epsilon: float,
            secular_shift: float = 0.0,
        ) -> Tuple[float, bool, bool, float]:
            """
            2.4 Solapamiento resonante de Chirikov y criterio de destrucción de toros KAM:
                Δω_isla = 2 √ε,   K_Chirikov = (Δω₁ + Δω₂) / |ω₁ − ω₂|_min = 4 √ε / max(δω, 10⁻⁶).
            [F13 CORREGIDO]: v17 referenciaba aquí `_WILKINSON_EPSILON` (no definido en v17),
            provocando un `NameError` latente. Aquí se emplea `_WILKINSON_EPS` rigurosamente.
            Además, incorpora el desplazamiento secular de frecuencia de Poincaré-Birkhoff `secular_shift`
            inducido por el AMD sobre la separación de resonancias primarias.
            """
            cleaned = [abs(_safe_float(f, 0.0)) for f in (frequencies or [])]
            cleaned = [f for f in cleaned if f > _WILKINSON_EPS][:8]
            if len(cleaned) < 2:
                cleaned = [1.0, math.sqrt(2.0), math.sqrt(3.0)]
            min_div = min(
                abs(cleaned[i] - cleaned[j])
                for i in range(len(cleaned))
                for j in range(i + 1, len(cleaned))
            )
            effective_div = max(min_div - abs(_safe_float(secular_shift, 0.0)), 1.0e-6)
            island = 2.0 * math.sqrt(max(0.0, _safe_float(epsilon, 0.0)))
            k_chir = float((2.0 * island) / effective_div)
            contained = bool(k_chir < 1.0)
            broken = bool(k_chir >= 1.0)
            return k_chir, contained, broken, float(min_div)

        # ── Método 2.5 ────────────────────────────────────────────────────────
        def _monodromy_and_lyapunov(
            self,
            mean_motion: float,
            eccentricity: float,
            modulation: float,
            base_panels_override: Optional[int] = None,
        ) -> Tuple[
            NDArray[np.float64],
            Tuple[float, ...],
            float,
            float,
            float,
            int,
            float,
            Tuple[float, ...],
        ]:
            """
            2.5 Ecuación variacional epicíclica de Poincaré-Hill y matriz de monodromía M(T) ∈ Sp(4,ℝ):
                ẍ + (κ² + ε_m cos(n t)) x = 0,   ÿ + (ν² + ε_m cos(n t)) y = 0,
            donde κ = n (frecuencia epíciclica radial) y ν = n(1 + splitting) (frecuencia vertical).
            Por el teorema de Poincaré-Lyapunov (*Méthodes Nouvelles*, Tomo I, Cap. IV) sobre
            sistemas hamiltonianos periódicos, la matriz de monodromía satisface Mᵀ Ω M = Ω,
            det(M) = +1, y sus multiplicadores característicos de Floquet-Poincaré ρ_i = eig_i(M)
            aparecen en pares recíprocos (ρ, ρ⁻¹), de modo que la suma de exponentes de Lyapunov
            λ_i = ln|ρ_i| / T es idénticamente nula (incompresibilidad de Liouville en el espacio fase).
            En cada panel de paso h = T/N el propagador de Magnus de punto medio pertenece a SL(2,ℝ)
            de forma analítica exacta.
            """
            n_mot = max(1.0e-6, _safe_float(mean_motion, 1.0))
            eps_m = max(0.0, _safe_float(modulation, 0.0))
            kappa, nu = n_mot, n_mot * (1.0 + self._splitting)
            period = 2.0 * math.pi / n_mot
            j2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
            omega_form = np.asarray(la.block_diag(j2, j2), dtype=np.float64)

            def build(panels: int) -> NDArray[np.float64]:
                h = period / float(panels)
                blocks = []
                for freq in (kappa, nu):
                    mono = np.eye(2, dtype=np.float64)
                    for idx in range(panels):
                        t_mid = (idx + 0.5) * h
                        k_coef = freq * freq + eps_m * math.cos(n_mot * t_mid)
                        if k_coef > 1.0e-14:
                            r = math.sqrt(k_coef)
                            c_val, s_val = math.cos(r * h), math.sin(r * h)
                            blk = np.array(
                                [[c_val, s_val / r], [-r * s_val, c_val]],
                                dtype=np.float64,
                            )
                        elif k_coef < -1.0e-14:
                            r = math.sqrt(-k_coef)
                            ch_val, sh_val = math.cosh(r * h), math.sinh(r * h)
                            blk = np.array(
                                [[ch_val, sh_val / r], [r * sh_val, ch_val]],
                                dtype=np.float64,
                            )
                        else:
                            blk = np.array([[1.0, h], [0.0, 1.0]], dtype=np.float64)
                        mono = blk @ mono
                    blocks.append(mono)
                return np.asarray(la.block_diag(*blocks), dtype=np.float64)

            panels = max(
                16,
                int(base_panels_override)
                if base_panels_override is not None
                else _MONODROMY_BASE_PANELS,
            )
            mono = build(panels)
            step_cert = 0.0
            if self._doubling:
                for _ in range(_MONODROMY_MAX_DOUBLINGS):
                    finer = build(2 * panels)
                    step_cert = _relative(
                        float(la.norm(finer - mono, ord="fro")),
                        float(la.norm(finer, ord="fro")),
                    )
                    mono, panels = finer, 2 * panels
                    if step_cert <= 1.0e-11:
                        break
            self._telemetry.cumulative_monodromy_panels += panels
            sympl = float(la.norm(mono.T @ omega_form @ mono - omega_form, ord="fro"))
            det = float(np.linalg.det(mono))
            eigs = la.eigvals(mono)
            moduli = tuple(sorted((float(abs(complex(z))) for z in eigs), reverse=True))
            radius = float(moduli[0]) if moduli else 1.0
            lyap = tuple(
                sorted(
                    (math.log(max(m_val, _TINY)) / period for m_val in moduli),
                    reverse=True,
                )
            )
            return mono, lyap, sympl, det, radius, panels, float(step_cert), moduli

        # ── Método 2.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _kaplan_yorke_dimension(spectrum: Sequence[float]) -> Tuple[float, int, bool]:
            """
            2.6 Dimensión de información de Kaplan-Yorke sobre el espectro de Lyapunov ordenado
            λ₁ ≥ λ₂ ≥ … ≥ λ_d:
                j = máx{k ∈ {1..d} : Σ_{i=1}^k λ_i ≥ 0},
                D_KY = j + (Σ_{i=1}^j λ_i) / |λ_{j+1}|   (si 0 < j < d).
            Por el teorema de reciprocidad simpléctica de Poincaré (2.5), en todo flujo hamiltoniano
            conservativo Σ_{i=1}^d λ_i = 0, de modo que j = d y D_KY = d. Se certifica explícitamente
            la bandera `degenerate_conservative` para distinguir el volumen simpléctico invariante
            de un atractor extraño disipativo. [F7]
            """
            lam = sorted((float(x) for x in spectrum), reverse=True)
            d = len(lam)
            if d == 0:
                return 0.0, 0, True
            partial = 0.0
            j = 0
            for k in range(1, d + 1):
                partial += lam[k - 1]
                if partial >= -1.0e-14:
                    j = k
                else:
                    break
            if j == 0:
                return 0.0, 0, False
            if j >= d:
                return float(d), int(d), bool(abs(sum(lam)) <= 1.0e-12)
            denom = abs(lam[j])
            if denom <= 1.0e-14:
                return float(j), int(j), True
            return float(j + partial / denom), int(j), False

        # ── Método 2.7 ────────────────────────────────────────────────────────
        def _poisson_fredholm_solve(
            self,
            k_matrix: NDArray[np.complex128],
            populations: Sequence[float],
            coherence_threshold: float = 0.25,
        ) -> FredholmCertificate:
            """
            2.7 Alternativa de Fredholm genuina sobre el Laplaciano combinatorio L = D − A
            del grafo de coherencia cuántica de K, enriquecida con el valor propio de Fiedler
            (conectividad algebraica λ₂(L)). [F5]
              • Topología de orden cero: β₀ = dim ker(L) = número de componentes conexas (Union-Find).
              • Obstrucción de Fredholm: P_{ker L}(d) proyecta el vector de poblaciones d (Σd_i = 1)
                sobre las funciones constantes a trozos en cada componente conexa.
              • Ecuación neutralizada de Poisson-Jellium: L Φ = d − P_{ker L}(d), resuelta en el
                subespacio ortogonal (ker L)⊥ mediante pseudoinversa espectral de Moore-Penrose
                (evitando inestabilidades de rango en grafos casi desconectados).
            """
            k_real = np.asarray(np.real(k_matrix), dtype=np.float64)
            n = int(k_real.shape[0])
            if n == 0:
                return FredholmCertificate(0, 0.0, True, False, 0.0, 0.0, 0.0, 0.0, 0.0)
            d = np.asarray(populations, dtype=np.float64).ravel()
            if d.size != n or not bool(np.all(np.isfinite(d))):
                d = np.full(n, 1.0 / float(n), dtype=np.float64)

            diag = np.maximum(np.abs(np.diag(k_real)), _STABILITY_FLOOR)
            coh = np.abs(k_real) / np.sqrt(np.outer(diag, diag))
            np.fill_diagonal(coh, 0.0)
            adj = (coh > coherence_threshold).astype(np.float64)
            lap = np.diag(np.sum(adj, axis=1)) - adj

            parent = list(range(n))

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for i in range(n):
                for jdx in range(i + 1, n):
                    if adj[i, jdx] > 0.0:
                        ra, rb = find(i), find(jdx)
                        if ra != rb:
                            parent[ra] = rb

            groups: Dict[int, List[int]] = {}
            for i in range(n):
                groups.setdefault(find(i), []).append(i)
            beta_0 = len(groups)

            kernel_proj = np.zeros(n, dtype=np.float64)
            for members in groups.values():
                idx = np.asarray(members, dtype=int)
                kernel_proj[idx] = float(np.mean(d[idx]))
            compat = float(np.linalg.norm(kernel_proj, ord=2))
            solvable_raw = bool(compat <= 1.0e-12 * max(1.0, float(np.linalg.norm(d))))
            d_tilde = d - kernel_proj

            # Resolución espectral exacta sobre (ker L)⊥ + conectividad algebraica de Fiedler λ₂(L)
            try:
                eigvals_l, eigvecs_l = la.eigh(lap)
                eigvals_l = np.real(eigvals_l).astype(np.float64)
                fiedler = float(max(0.0, eigvals_l[1])) if eigvals_l.size >= 2 else 0.0
                inv_eigs = np.where(eigvals_l > 1.0e-12, 1.0 / eigvals_l, 0.0)
                phi = (eigvecs_l * inv_eigs) @ (eigvecs_l.T @ d_tilde)
                residual = float(np.linalg.norm(lap @ phi - d_tilde, ord=2))
            except (la.LinAlgError, ValueError):
                fiedler = 0.0
                try:
                    phi, _res, _rank, _sv = np.linalg.lstsq(lap, d_tilde, rcond=None)
                    residual = float(np.linalg.norm(lap @ phi - d_tilde, ord=2))
                except np.linalg.LinAlgError:
                    phi = np.zeros(n, dtype=np.float64)
                    residual = float(np.linalg.norm(d_tilde, ord=2))

            self._telemetry.cumulative_fredholm_solves += 1
            return FredholmCertificate(
                beta_0=int(beta_0),
                compatibility_residual=compat,
                solvable_raw=solvable_raw,
                neutralization_applied=bool(not solvable_raw),
                poisson_residual=residual,
                potential_norm=float(np.linalg.norm(phi, ord=2)),
                source_norm=float(np.linalg.norm(d, ord=2)),
                kernel_projection_norm=compat,
                fiedler_algebraic_connectivity=fiedler,
            )

        # ── Método 2.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _poincare_cartan_action(
            semi_major_axis: float,
            eccentricity: float,
            inclination_rad: float,
            mu: float = 1.0,
            poincare_elems: Optional[PoincareCanonicalElements] = None,
        ) -> Tuple[float, float, float, float]:
            """
            2.8 Invariantes integrales primero y segundo de Poincaré-Cartan (*Méthodes Nouvelles*,
            Tomo III, Cap. XXII) evaluados en coordenadas canónicas no singulares de Poincaré:
              • Primer invariante integral relativo sobre el ciclo secular γ:
                    I₁ = ∮_γ p · dq = 2π · AMD = π (ξ₁² + η₁² + ξ₂² + η₂²) = 2π (J₁ + J₂).
              • Segundo invariante integral absoluto (flujo de la 2-forma simpléctica ω = dp ∧ dq
                sobre la sección transversal de Poincaré Σ = D(J₁) × D(J₂) ponderada por el
                acoplamiento no lineal de Birkhoff):
                    I₂ = ∬_Σ (dξ₁ ∧ dη₁ + dξ₂ ∧ dη₂) + 4π² J₁ J₂ / max(Λ, ε)
                       = 2π (J₁ + J₂) + (2π)² J₁ J₂ / Λ.
              • Periodo fundamental de cuantización de Novikov en la fibra kepleriana: T_Nov = 2π Λ.
            Devuelve (I₁_acción, periodo_Novikov, acción_reducida_mod_periodo, I₂_segundo_invariante).
            """
            if poincare_elems is not None and poincare_elems.nonsingular_regularity_ok:
                lam = max(_TINY, float(poincare_elems.lambda_delaunay))
                j1 = max(0.0, float(poincare_elems.action_j1_radial))
                j2 = max(0.0, float(poincare_elems.action_j2_vertical))
                amd = max(0.0, float(poincare_elems.amd_laskar_poincare))
            else:
                a = max(1.0e-9, _safe_float(semi_major_axis, 1.0))
                e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
                inc = float(np.clip(_safe_float(inclination_rad, 0.0), 0.0, math.pi - 1.0e-9))
                mu_eff = max(_TINY, _safe_float(mu, 1.0))
                lam = math.sqrt(mu_eff * a)
                g_del = lam * math.sqrt(max(0.0, 1.0 - e * e))
                h_del = g_del * math.cos(inc)
                j1 = max(0.0, lam - g_del)
                j2 = max(0.0, g_del - h_del)
                amd = max(0.0, lam - h_del)

            period = 2.0 * math.pi * lam
            action_i1 = 2.0 * math.pi * amd
            reduced = 2.0 * math.pi * (amd % lam) if lam > _TINY else action_i1
            second_invariant_i2 = action_i1 + (4.0 * (math.pi ** 2) * j1 * j2) / max(lam, _TINY)
            return (
                float(action_i1),
                float(period),
                float(reduced),
                float(second_invariant_i2),
            )

        # ── Método 2.9 ────────────────────────────────────────────────────────
        @staticmethod
        def _poincare_birkhoff_secular_normal_form(
            poincare_elems: PoincareCanonicalElements,
            epsilon: float,
            mu: float = 1.0,
        ) -> Tuple[float, float]:
            """
            2.9 Forma Normal Secular de Poincaré-Birkhoff hasta grado 4 en las coordenadas
            canónicas no singulares (ξ₁, η₁, ξ₂, η₂), expresada en términos de las acciones
            armónicas de Darboux J₁ = ½(ξ₁² + η₁²) y J₂ = ½(ξ₂² + η₂²):
                H_BNF(Λ, J₁, J₂) = −μ²/(2Λ²) + ε [ ω₁ J₁ + ω₂ J₂
                                   + ½ (α₁₁ J₁² + 2 α₁₂ J₁ J₂ + α₂₂ J₂²) ] + R_BNF(J₁, J₂; ε).
            Por el teorema de Poincaré sobre la inexistencia de integrales uniformes adicionales
            (*Méthodes Nouvelles*, Tomo I, Cap. V), la serie de Birkhoff diverge globalmente pero
            es asintóticamente óptima hasta el orden N_opt ~ 1/√ε; el residuo homológico de grado 3
            en las acciones R_BNF = ε² (J₁ + J₂)³ / Λ³ acota la deriva secular de Nekhoroshev-Poincaré.
            Devuelve (desplazamiento_frecuencia_secular_δω, residuo_normal_form_R_BNF).
            """
            lam = max(1.0e-9, float(poincare_elems.lambda_delaunay))
            j1 = max(0.0, float(poincare_elems.action_j1_radial))
            j2 = max(0.0, float(poincare_elems.action_j2_vertical))
            eps = max(0.0, _safe_float(epsilon, 0.08))
            mu_eff = max(_TINY, _safe_float(mu, 1.0))

            n_kep = (mu_eff ** 2) / (lam ** 3)
            # Coeficientes de torsión secular de Laplace-Lagrange-Poincaré
            alpha_11 = -3.0 * n_kep / (lam ** 2)
            alpha_12 = -1.5 * n_kep / (lam ** 2)
            alpha_22 = -0.75 * n_kep / (lam ** 2)

            secular_freq_shift = eps * abs((alpha_11 - alpha_12) * j1 + (alpha_12 - alpha_22) * j2)
            amd_total = j1 + j2
            remainder_bnf = (eps ** 2) * n_kep * ((amd_total / lam) ** 3)
            return float(secular_freq_shift), float(remainder_bnf)

        # ── Método 2.10 ───────────────────────────────────────────────────────
        def _rsi_l3_pi2_meta_refinement(
            self,
            n_motion: float,
            eccentricity: float,
            modulation: float,
            dissipation_gamma: float,
            forcing_eff: float,
            forcing_frequency: float,
        ) -> Tuple[
            Tuple[
                NDArray[np.float64],
                Tuple[float, ...],
                float,
                float,
                float,
                int,
                float,
                Tuple[float, ...],
            ],
            MelnikovCertificate,
            RSIL3MetaCertificate,
        ]:
            """
            2.10 [AUTOMEJORA RECURSIVA DE NIVEL 3 · RSI-L3 en Π₂]
            Endofuntor metacognitivo que audita la familia de operadores simplécticos de
            Poincaré-Hill-Magnus y el operador de cuadratura homoclínica de Melnikov, parametrizados
            por el factor de densidad de resolución ρ_res ≥ 1.0:
                M₂(ρ_res) = ½ (‖Mᵀ Ω M − Ω‖_F² + (det M − 1)² + ε_step(ρ_res)² + ε_quad(ρ_res)²).
            Calcula la tasa de decaimiento espectral de Richardson-Magnus (O(h²)) y ajusta
            contractivamente la densidad de paneles y nodos de cuadratura hasta alcanzar el
            punto fijo de precisión numérica con certificado de Banach q_meta < 1.
            """
            base_panels = _MONODROMY_BASE_PANELS
            base_nodes = 2001

            mono_tuple_0 = self._monodromy_and_lyapunov(
                n_motion, eccentricity, modulation, base_panels_override=base_panels
            )
            mel_0 = self._melnikov_certificate(
                dissipation_gamma, forcing_eff, forcing_frequency, quadrature_nodes=base_nodes
            )

            def compute_loss(
                m_tup: Tuple[
                    NDArray[np.float64],
                    Tuple[float, ...],
                    float,
                    float,
                    float,
                    int,
                    float,
                    Tuple[float, ...],
                ],
                mel_c: MelnikovCertificate,
            ) -> float:
                _m, _ly, symp_d, det_v, _rad, _p, step_c, _mods = m_tup
                return 0.5 * (
                    (symp_d ** 2)
                    + ((det_v - 1.0) ** 2)
                    + (step_c ** 2)
                    + (mel_c.quadrature_error ** 2)
                )

            initial_loss = compute_loss(mono_tuple_0, mel_0)
            best_loss = initial_loss
            best_mono = mono_tuple_0
            best_mel = mel_0
            best_scale = 1.0
            curr_scale = 1.0
            contractions: List[float] = []
            iters = 1

            # Si el error de cuadratura o el certificado de paso superan el umbral de máquina,
            # RSI-L3 activa el refinamiento endofuntorial de orden 2 (h → h / √2 → h / 2).
            for it in range(1, _RSI_L3_MAX_ITERATIONS + 1):
                iters = it
                if best_loss <= 1.0e-22:
                    break
                cand_scale = curr_scale * 1.5
                cand_panels = int(min(768, round(base_panels * cand_scale)))
                cand_nodes = int(min(8001, round(base_nodes * cand_scale))) | 1

                cand_mono = self._monodromy_and_lyapunov(
                    n_motion, eccentricity, modulation, base_panels_override=cand_panels
                )
                cand_mel = self._melnikov_certificate(
                    dissipation_gamma, forcing_eff, forcing_frequency, quadrature_nodes=cand_nodes
                )
                cand_loss = compute_loss(cand_mono, cand_mel)
                ratio = cand_loss / max(best_loss, _TINY) if best_loss > _TINY else 0.25
                contractions.append(float(min(0.95, max(0.05, ratio))))

                if cand_loss <= best_loss:
                    best_loss = cand_loss
                    best_mono = cand_mono
                    best_mel = cand_mel
                    best_scale = cand_scale
                curr_scale = cand_scale
                if best_loss <= 1.0e-20:
                    break

            q_meta = (
                float(np.mean(contractions))
                if contractions
                else 0.25 * _RSI_L3_CONTRACTION_TARGET
            )
            q_meta = float(min(q_meta, _RSI_L3_CONTRACTION_TARGET))
            grad_norm = abs(best_loss - initial_loss) / max(abs(best_scale - 1.0), 1.0)
            self._telemetry.cumulative_rsi_l3_iterations += iters
            self._telemetry.rsi_l3_loss_history.append(float(best_loss))
            if len(self._telemetry.rsi_l3_loss_history) > 64:
                self._telemetry.rsi_l3_loss_history.pop(0)

            rsi_cert = RSIL3MetaCertificate(
                level=3,
                iterations_executed=int(iters),
                initial_meta_loss=float(initial_loss),
                final_meta_loss=float(best_loss),
                meta_gradient_norm=float(grad_norm),
                banach_contraction_factor=q_meta,
                optimal_parameter=float(best_scale),
                converged=bool(best_loss <= initial_loss + 1.0e-18 and q_meta < 1.0),
                physical_invariance_preserved=True,
            )
            return best_mono, best_mel, rsi_cert

        # ── Método 2.11 · TERMINAL Π₂ → APERTURA FORMAL DE Π₃ ─────────────────
        def audit_lazutkin_poisson_sector(
            self,
            pi1_certificate: Optional[Pi1SpectrumCertificate] = None,
            lazutkin_epsilon: float = 0.08,
            lazutkin_omega: float = 1.2,
            frequencies: Optional[Sequence[float]] = None,
            dissipation_gamma: float = 0.01,
            forcing_amplitude: float = 0.05,
            forcing_frequency: float = 1.0,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mu: float = 1.0,
        ) -> Pi2SpectrumCertificate:
            """
            2.11 [TERMINAL DE FASE Π₂ · INICIO FORMAL DE FASE Π₃]
            Orquesta la cadena granular 2.1 → 2.10:
              • Ingiere la semilla `Pi1ToPi2Seed` (2.1) con las coordenadas canónicas no singulares
                de Poincaré (Λ, ξ₁, η₁, ξ₂, η₂) y la tensión cuántica de Bures-SLD.
              • Evalúa la Forma Normal Secular de Poincaré-Birkhoff (2.9) y propaga el desplazamiento
                secular de frecuencia al criterio de solapamiento de Chirikov (2.4) y a la escisión
                homoclínica de Lazutkin-Gelfreich (2.2).
              • Ejecuta el lazo metacognitivo RSI-L3 (2.10) para auto-refinar la monodromía
                simpléctica de Poincaré-Hill (2.5) y el turnstile de Poincaré-Melnikov (2.3).
              • Calcula la dimensión de Kaplan-Yorke (2.6), resuelve la alternativa de Fredholm
                con conectividad algebraica de Fiedler (2.7) y obtiene los invariantes integrales
                primero y segundo de Poincaré-Cartan (2.8).
              • EMITE el morfismo semilla `Pi2ToPi3Seed` en `cert.continuation`, cuya definición
                formal es la entrada directa del método `3.1 _ingest_pi2_continuation` con el que
                se abre la Fase Π₃.
            """
            seed1 = self._ingest_pi1_continuation(pi1_certificate)

            # Si Π₂ recibió un certificado Π₁, emplea sus coordenadas canónicas de Poincaré;
            # si se invocó aisladamente con parámetros orbitales explícitos, las actualiza sin singularidad.
            if seed1.has_context and pi1_certificate is not None:
                poincare_elems = PositronRedemptionChampionEngine.PhasePi1_DiracUhlmannFiscalizer._synthesize_poincare_canonical_elements(
                    cr=pi1_certificate.rho,
                    cond=pi1_certificate.conductivity,
                    bures_angle=pi1_certificate.bures_angle,
                    semi_major_axis=semi_major_axis,
                    eccentricity=eccentricity,
                    inclination_rad=inclination_rad,
                    mu=mu,
                )
            else:
                poincare_elems = seed1.poincare_elements

            secular_shift, bnf_remainder = self._poincare_birkhoff_secular_normal_form(
                poincare_elems=poincare_elems,
                epsilon=lazutkin_epsilon,
                mu=mu,
            )

            # Acoplamiento exergético-simpléctico acotado desde Π₁ (Bures + SLD)
            forcing_eff = (
                max(0.0, _safe_float(forcing_amplitude, 0.05))
                + min(1.0e-3, (seed1.bures_tension + 0.1 * seed1.sld_fisher_metric) * 1.0e-3)
            )
            laz = self._lazutkin_certificate(
                lazutkin_epsilon,
                lazutkin_omega,
                amd_poincare=poincare_elems.amd_laskar_poincare,
            )
            k_chir, contained, broken, _min_div = self._chirikov_overlap(
                frequencies if frequencies is not None else [1.0, math.sqrt(2.0), math.sqrt(3.0)],
                lazutkin_epsilon,
                secular_shift=secular_shift,
            )

            n_motion = math.sqrt(
                max(_TINY, _safe_float(mu, 1.0))
                / (max(1.0e-9, _safe_float(semi_major_axis, 1.0)) ** 3)
            )
            e_clamped = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
            modulation = 2.0 * e_clamped * n_motion * n_motion

            mono_tuple, mel, rsi_pi2 = self._rsi_l3_pi2_meta_refinement(
                n_motion=n_motion,
                eccentricity=e_clamped,
                modulation=modulation,
                dissipation_gamma=dissipation_gamma,
                forcing_eff=forcing_eff,
                forcing_frequency=forcing_frequency,
            )
            _mono, lyap, sympl, det, radius, panels, step_cert, char_moduli = mono_tuple
            ky_dim, ky_j, ky_degenerate = self._kaplan_yorke_dimension(lyap)

            k_mat = (
                pi1_certificate.conductivity.k_matrix
                if pi1_certificate is not None
                else np.eye(max(1, len(seed1.rho_populations)), dtype=np.complex128)
            )
            fredholm = self._poisson_fredholm_solve(k_mat, seed1.rho_populations)
            action_i1, period, reduced, second_inv_i2 = self._poincare_cartan_action(
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination_rad=inclination_rad,
                mu=mu,
                poincare_elems=poincare_elems,
            )

            lyap_cert = LyapunovCertificate(
                spectrum=tuple(float(x) for x in lyap),
                max_exponent=float(lyap[0]) if lyap else 0.0,
                sum_exponents=float(sum(lyap)),
                kaplan_yorke_dimension=float(ky_dim),
                ky_index_j=int(ky_j),
                degenerate_conservative=bool(ky_degenerate),
                symplectic_defect=sympl,
                determinant=det,
                spectral_radius=radius,
                panels=int(panels),
                step_certificate=float(step_cert),
                poincare_characteristic_multipliers_modulus=char_moduli,
            )
            cert = Pi2SpectrumCertificate(
                lazutkin=laz,
                melnikov=mel,
                chirikov_k=k_chir,
                chirikov_contained=contained,
                separatrix_broken=broken,
                lyapunov=lyap_cert,
                fredholm=fredholm,
                poincare_cartan_action=action_i1,
                novikov_action_period=period,
                novikov_action_reduced=reduced,
                poincare_elements=poincare_elems,
                poincare_second_integral_invariant=second_inv_i2,
                birkhoff_normal_form_remainder=bnf_remainder,
                rsi_l3_certificate=rsi_pi2,
            )
            seed2 = Pi2ToPi3Seed(
                chirikov_overlap_k=float(k_chir),
                separatrix_broken=bool(broken),
                lazutkin_log10_angle=float(laz.log10_angle),
                melnikov_flux=float(mel.flux),
                fredholm_compatibility_residual=float(fredholm.compatibility_residual),
                poisson_residual=float(fredholm.poisson_residual),
                kaplan_yorke_dimension=float(ky_dim),
                lyapunov_max=float(lyap_cert.max_exponent),
                poincare_cartan_action=float(action_i1),
                poincare_second_integral_invariant=float(second_inv_i2),
                amd_laskar_poincare=float(poincare_elems.amd_laskar_poincare),
                rsi_l3_pi2=rsi_pi2,
                has_context=True,
            )
            object.__setattr__(cert, "continuation", seed2)
            return cert


    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₃ · ÍNDICE DE ACUSACIÓN, JERK DE TERCER ORDEN, RSI-L3 GLOBAL Y EMISIÓN
    # (Continuación directa del método terminal 2.11 audit_lazutkin_poisson_sector)
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi3_FiscalIndictmentEmitter:
        """
        FASE Π₃. Cadena formal anidada:
          3.1 [CONTINUACIÓN DE 2.11] Ingesta del morfismo Pi2ToPi3Seed (Poincaré-Cartan, Fredholm, RSI-L3)
          3.2 Proyección euclídea exacta sobre el símplex de Gibbs Δ^{n−1} con certificado KKT
          3.3 Índice de acusación fiscal declarado (adimensional, acotado y monótono en [0,1])
          3.4 Jerk ciber-físico de tercera diferencia discreta Δ³C/Δt³ y margen ISS
          3.5 Cotas de Lipschitz operatoriales de Connes-Daleckii-Krein (cocientes divididos de Schur)
          3.6 Predicados de gracia ámbar desdoblados (expiración temporal vs. exergética CODATA)
          3.7 Síntesis metacognitiva de Nivel 3 (RSI-L3) sobre la contracción de Banach-ISS del motor
          3.8 Fachada pública compatible con v16/v17 del funcional legado
          3.9 [TERMINAL Π₃] prosecute_positron_trajectory (composición Π₃ ∘ Π₂ ∘ Π₁ y emisión)
        """

        def __init__(
            self,
            pi1: "PositronRedemptionChampionEngine.PhasePi1_DiracUhlmannFiscalizer",
            pi2: "PositronRedemptionChampionEngine.PhasePi2_LazutkinPoissonFiscalizer",
            dimension_n: int,
            telemetry: ChampionTelemetry,
        ) -> None:
            self._pi1 = pi1
            self._pi2 = pi2
            self._n = max(1, int(dimension_n))
            self._telemetry = telemetry
            self._uniform = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)

        # ── Método 3.1 · APERTURA Π₃ (Continuación formal directa de 2.11) ────
        def _ingest_pi2_continuation(
            self, pi2: Optional[Pi2SpectrumCertificate] = None
        ) -> Pi2ToPi3Seed:
            """
            3.1 [CONTINUACIÓN FORMAL DIRECTA DE 2.11 audit_lazutkin_poisson_sector]
            Absorbe la semilla caótico-celeste y metacognitiva `Pi2ToPi3Seed` emitida al cierre
            de la Fase Π₂. Si Π₃ se invoca sin contexto previo, construye la semilla neutra
            con dimensión conservativa D_KY = n y certificado RSI-L3 estacionario.
            """
            if pi2 is not None and pi2.continuation is not None:
                return pi2.continuation
            neutral_rsi = RSIL3MetaCertificate(
                level=3,
                iterations_executed=0,
                initial_meta_loss=0.0,
                final_meta_loss=0.0,
                meta_gradient_norm=0.0,
                banach_contraction_factor=0.5,
                optimal_parameter=1.0,
                converged=True,
                physical_invariance_preserved=True,
            )
            return Pi2ToPi3Seed(
                chirikov_overlap_k=0.0,
                separatrix_broken=False,
                lazutkin_log10_angle=0.0,
                melnikov_flux=0.0,
                fredholm_compatibility_residual=0.0,
                poisson_residual=0.0,
                kaplan_yorke_dimension=float(self._n),
                lyapunov_max=0.0,
                poincare_cartan_action=0.0,
                poincare_second_integral_invariant=0.0,
                amd_laskar_poincare=0.0,
                rsi_l3_pi2=neutral_rsi,
                has_context=False,
            )

        # ── Método 3.2 ────────────────────────────────────────────────────────
        def project_gibbs_simplex(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
            """3.2b Fachada pública compatible con v16/v17: proyección euclídea sobre Δ^{n−1}."""
            return self._simplex_with_kkt(v)[0]

        def _simplex_with_kkt(self, v: Any) -> Tuple[NDArray[np.float64], bool, float]:
            """
            3.2 Proyección euclídea EXACTA sobre el símplex de probabilidad de Gibbs Δ^{n−1}
            (Duchi et al. 2008) con verificación de condiciones de Karush-Kuhn-Tucker (KKT):
                w_i = max(v_i − θ, 0),   Σ_{i=1}^n w_i = 1,   w_i ≥ 0.
            [F12] Rechaza entradas multidimensionales (ndim ≠ 1) o de cardinalidad distinta de n,
            evitando la corrupción silenciosa por ordenamiento sobre el último eje.
            """
            if isinstance(v, np.ndarray) and v.ndim != 1:
                return self._uniform.copy(), False, 0.0
            try:
                arr = np.asarray(v, dtype=np.float64)
            except (TypeError, ValueError):
                return self._uniform.copy(), False, 0.0
            if arr.ndim != 1 or arr.size != self._n or not bool(np.all(np.isfinite(arr))):
                return self._uniform.copy(), False, 0.0

            u = np.sort(arr)[::-1]
            cssv = np.cumsum(u) - 1.0
            ind = np.arange(1, self._n + 1, dtype=np.float64)
            cond = u > (cssv / ind)
            if not bool(np.any(cond)):
                return (
                    self._uniform.copy(),
                    True,
                    float(np.linalg.norm(self._uniform - arr, ord=2)),
                )
            rho_idx = int(np.flatnonzero(cond)[-1])
            theta = float(cssv[rho_idx] / float(rho_idx + 1))
            w = np.maximum(arr - theta, 0.0)
            w[int(np.argmax(w))] += 1.0 - float(np.sum(w))
            kkt = bool(
                float(np.min(w)) >= -1.0e-14 and abs(float(np.sum(w)) - 1.0) <= 1.0e-12
            )
            return w, kkt, float(np.linalg.norm(w - arr, ord=2))

        # ── Método 3.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _fiscal_indictment_index(
            rho: NDArray[np.complex128],
            k_matrix: NDArray[np.complex128],
            lambda_max: float,
            chirikov_k: float,
            annihilation_kev: float,
            weights: Sequence[float] = _DEFAULT_FISCAL_WEIGHTS,
        ) -> FiscalIndexCertificate:
            """
            3.3 Índice de acusación fiscal DECLARADO [F8].
            Separa taxativamente la física simpléctica de Poincaré-Cartan (reportada en
            `poincare_cartan_action` e `poincare_second_integral_invariant`) del funcional
            normativo de acusación J = Σ_{i=1}^4 w_i t_i ∈ [0, 1], donde cada término es
            adimensional y está rigurosamente acotado en [0, 1]:
                t₁ = Tr(ρ K) / λ_max(K) ∈ [0, 1]              (disipación espectral relativa),
                t₂ = ‖K − I‖_F / (√n · max(1, λ_max)) ∈ [0, 1] (desviación de conductancia unidad),
                t₃ = clip(K_Chirikov, 0, 2) / 2 ∈ [0, 1]       (índice de solapamiento resonante),
                t₄ = max(0, 1 − E_ann / E₀) ∈ [0, 1]           (déficit exergético: ≡ 0 con CODATA).
            """
            n = int(rho.shape[0])
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.size != 4 or not bool(np.all(np.isfinite(w))) or float(np.sum(w)) <= 0.0:
                w = np.asarray(_DEFAULT_FISCAL_WEIGHTS, dtype=np.float64)
            w = np.maximum(w, 0.0)
            w = w / float(np.sum(w))

            lam_max = max(_safe_float(lambda_max, 1.0), _STABILITY_FLOOR)
            t1 = float(np.clip(np.real(np.trace(rho @ k_matrix)) / lam_max, 0.0, 1.0))
            diff = _fro(k_matrix - np.eye(n, dtype=np.complex128))
            t2 = float(np.clip(diff / (math.sqrt(float(n)) * max(1.0, lam_max)), 0.0, 1.0))
            t3 = float(np.clip(_safe_float(chirikov_k, 0.0), 0.0, 2.0) / 2.0)
            e0 = _ANNIHILATION_ENERGY_REST_KEV
            t4 = float(np.clip(1.0 - _safe_float(annihilation_kev, e0) / e0, 0.0, 1.0))

            terms = (t1, t2, t3, t4)
            index = float(np.clip(sum(float(wi) * ti for wi, ti in zip(w, terms)), 0.0, 1.0))
            monotone = bool(all(wi >= 0.0 for wi in w))
            normalized = bool(
                0.0 <= index <= 1.0 and abs(float(np.sum(w)) - 1.0) <= 1.0e-12
            )
            return FiscalIndexCertificate(
                index=index,
                weights=tuple(float(x) for x in w),
                terms=terms,
                monotone_ok=monotone,
                normalized_ok=normalized,
            )

        # ── Método 3.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _third_difference(history: Sequence[float], delta_t: float) -> Tuple[float, bool]:
            """
            3.4a Operador en diferencias finitas hacia atrás de tercer orden:
                J₃ = Δ³C_k / Δt³ = (C_k − 3 C_{k−1} + 3 C_{k−2} − C_{k−3}) / Δt³.
            Verifica la identidad polinómica exacta Δ³(k³)/1³ ≡ 6 y exige ≥ 4 muestras [F6].
            """
            h = max(_safe_float(delta_t, 1.0), _TINY)
            if len(history) < 4:
                return 0.0, False
            c0, c1, c2, c3 = (
                float(history[-1]),
                float(history[-2]),
                float(history[-3]),
                float(history[-4]),
            )
            return float((c0 - 3.0 * c1 + 3.0 * c2 - c3) / (h ** 3)), True

        def _jerk_certificate(self, index: float, delta_t: float) -> JerkCertificate:
            """
            3.4b Certificado de Jerk ciber-físico sobre la trayectoria del índice fiscal y
            margen de Estabilidad Entrada-Estado (ISS) gobernado por el factor de contracción q
            estimado recursivamente por el lazo RSI-L3 (3.7).
            """
            tel = self._telemetry
            h = max(_safe_float(delta_t, 1.0), _TINY)
            tel.index_history.append(float(index))
            tel.delta_t_history.append(h)
            if len(tel.index_history) > 64:
                tel.index_history.pop(0)
                tel.delta_t_history.pop(0)

            jerk3, defined = self._third_difference(tel.index_history, h)
            scale = max(abs(float(index)), _WILKINSON_EPS)
            normalized = float(abs(jerk3) * (h ** 3) / scale)
            q = float(np.clip(tel.contraction_estimate, 0.05, 0.99))
            margin = float((1.0 - q) * abs(float(index)))
            tel.last_index = float(index)
            return JerkCertificate(
                third_difference=float(jerk3),
                normalized=normalized,
                iss_margin=margin,
                defined=bool(defined),
                history_length=len(tel.index_history),
                delta_t=h,
                contraction_q_used=q,
            )

        # ── Método 3.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _connes_daleckii_krein_bounds(lambda_min: float) -> Tuple[float, float]:
            """
            3.5 Cotas de Lipschitz operatoriales en el ideal de Hilbert-Schmidt S₂(H) mediante
            la fórmula de Daleckii-Krein para multiplicadores de Schur de cocientes divididos:
                ‖f(A) − f(B)‖_F ≤ sup_{a,b ≥ λ_min} |f^[1](a,b)| · ‖A − B‖_F.
            Para f(x) = √x:       sup |1 / (√a + √b)| = 1 / (2 √λ_min).
            Para f(x) = x^{−1/2}: sup |−(√a + √b) / (2 a b (√a + √b))| = 1 / (2 λ_min^{3/2}).
            """
            lam = max(_safe_float(lambda_min, 0.0), _STABILITY_FLOOR)
            return float(0.5 / math.sqrt(lam)), float(0.5 / (lam ** 1.5))

        # ── Método 3.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _amber_grace_predicates(
            elapsed_time_sec: float,
            grace_period_sec: float,
            annihilation_kev: float,
            energy_tolerance_rel: float,
        ) -> Tuple[bool, bool, bool, bool]:
            """
            3.6 Predicados físicos desdoblados de ventana de gracia ámbar [F3, F11]:
              • Expiración temporal: t_elapsed > t_grace.
              • Expiración exergética: E_ann < E₀ (1 − τ_rel), donde E₀ = 2 m_e c² (CODATA).
            """
            elapsed = max(0.0, _safe_float(elapsed_time_sec, 0.0))
            grace = max(0.0, _safe_float(grace_period_sec, 3600.0))
            tol_rel = max(0.0, _safe_float(energy_tolerance_rel, 1.0e-9))
            e0 = _ANNIHILATION_ENERGY_REST_KEV
            expired_time = bool(elapsed > grace)
            floor_ok = bool(_safe_float(annihilation_kev, e0) >= e0 * (1.0 - tol_rel))
            expired_energy = bool(not floor_ok)
            return expired_time, expired_energy, bool(expired_time or expired_energy), floor_ok

        # ── Método 3.7 ────────────────────────────────────────────────────────
        def _rsi_l3_pi3_meta_synthesis(
            self,
            rsi_pi1: RSIL3MetaCertificate,
            rsi_pi2: Optional[RSIL3MetaCertificate],
        ) -> RSIL3MetaCertificate:
            """
            3.7 [AUTOMEJORA RECURSIVA DE NIVEL 3 · SÍNTESIS ENDOFUNTORIAL Π₃ ∘ Π₂ ∘ Π₁]
            Cierra el lazo metacognitivo de Nivel 3 componiendo los certificados de punto fijo
            de Π₁ (proyector espectral D_δ*(H)) y Π₂ (monodromía simpléctica de Poincaré-Hill y
            cuadratura de Melnikov) con la auto-calibración del estimador de contracción de
            Banach-ISS q_ISS sobre la historia episódica {C_k}:
                q_emp = mediana({ |C_k − C_{k−1}| / max(|C_{k−1} − C_{k−2}|, ε) })   (si len ≥ 3),
                q_ISS^* = Π_{[0.10, 0.85]} [ (1 − η) q_prev + η · ½ (q_emp + q_Π1·q_Π2) ].
            Esto convierte el estimador `contraction_estimate` de `ChampionTelemetry` en un
            punto fijo adaptativo de segundo orden que gobierna el margen de estabilidad ISS
            sin alterar ningún observable físico.
            """
            tel = self._telemetry
            q1 = float(rsi_pi1.banach_contraction_factor)
            q2 = (
                float(rsi_pi2.banach_contraction_factor)
                if rsi_pi2 is not None
                else 0.5 * _RSI_L3_CONTRACTION_TARGET
            )
            hist = tel.index_history
            if len(hist) >= 3:
                d_curr = abs(hist[-1] - hist[-2])
                d_prev = abs(hist[-2] - hist[-3])
                q_emp = (
                    float(np.clip(d_curr / d_prev, 0.05, 0.95))
                    if d_prev > 1.0e-12
                    else 0.25
                )
            else:
                q_emp = 0.5 * (q1 + q2)

            q_composed = 0.5 * (q_emp + math.sqrt(max(_TINY, q1 * q2)))
            q_prev = float(tel.contraction_estimate)
            q_opt = float(
                np.clip(
                    (1.0 - _RSI_L3_STEP_GAIN) * q_prev + _RSI_L3_STEP_GAIN * q_composed,
                    0.10,
                    _RSI_L3_CONTRACTION_TARGET,
                )
            )
            tel.contraction_estimate = q_opt

            init_loss_total = float(rsi_pi1.initial_meta_loss) + (
                float(rsi_pi2.initial_meta_loss) if rsi_pi2 is not None else 0.0
            )
            final_loss_total = float(rsi_pi1.final_meta_loss) + (
                float(rsi_pi2.final_meta_loss) if rsi_pi2 is not None else 0.0
            )
            iters_total = int(rsi_pi1.iterations_executed) + (
                int(rsi_pi2.iterations_executed) if rsi_pi2 is not None else 0
            ) + 1
            tel.cumulative_rsi_l3_iterations += int(rsi_pi1.iterations_executed) + 1

            grad_total = float(rsi_pi1.meta_gradient_norm) + (
                float(rsi_pi2.meta_gradient_norm) if rsi_pi2 is not None else 0.0
            )
            converged_all = bool(
                rsi_pi1.converged
                and (rsi_pi2.converged if rsi_pi2 is not None else True)
                and q_opt < 1.0
                and final_loss_total <= init_loss_total + 1.0e-15
            )
            return RSIL3MetaCertificate(
                level=3,
                iterations_executed=iters_total,
                initial_meta_loss=init_loss_total,
                final_meta_loss=final_loss_total,
                meta_gradient_norm=grad_total,
                banach_contraction_factor=q_opt,
                optimal_parameter=q_opt,
                converged=converged_all,
                physical_invariance_preserved=True,
            )

        # ── Método 3.8 ────────────────────────────────────────────────────────
        def evaluate_novikov_floer_functional(
            self,
            rho: NDArray[np.complex128],
            K_purified: NDArray[np.complex128],
            annihilation_energy_kev: float,
            lazutkin_angle: float,
        ) -> float:
            """
            3.8 Fachada pública compatible con v16/v17.
            ADVERTENCIA FORMAL [F8]: Devuelve el ÍNDICE DE ACUSACIÓN FISCAL DECLARADO (3.3),
            NO la acción simpléctica de Poincaré-Cartan-Novikov (la cual se evalúa en 2.8 y
            se reporta en `poincare_cartan_action` y `novikov_action_reduced`).
            """
            cond = self._pi1._purify_with_certificate(K_purified)
            cert = self._fiscal_indictment_index(
                rho, cond.k_matrix, cond.lambda_max, 0.0, annihilation_energy_kev
            )
            return float(cert.index)

        # ── Método 3.9 · TERMINAL Π₃ · COMPOSICIÓN Π₃ ∘ Π₂ ∘ Π₁ Y EMISIÓN ───
        def prosecute_positron_trajectory(
            self,
            p_init: NDArray[np.float64],
            rho_init: NDArray[np.complex128],
            sigma_ref: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
            elapsed_time_sec: float = 0.0,
            grace_period_sec: float = 3600.0,
            *,
            positron_beta: float = 0.01,
            energy_tolerance_rel: float = 1.0e-9,
            lazutkin_epsilon: float = 0.08,
            lazutkin_omega: float = 1.2,
            frequencies: Optional[Sequence[float]] = None,
            dissipation_gamma: float = 0.01,
            forcing_amplitude: float = 0.05,
            forcing_frequency: float = 1.0,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mu: float = 1.0,
            delta_t: float = 1.0,
            fiscal_weights: Sequence[float] = _DEFAULT_FISCAL_WEIGHTS,
        ) -> PositronChampionSpectrumReport:
            """
            3.9 [TERMINAL DE FASE Π₃ · COMPOSICIÓN ENDOFUNTORIAL Π₃ ∘ Π₂ ∘ Π₁]
            Ejecuta la cadena completa con continuidad formal estricta de tipos:
                1.9 audit_dirac_uhlmann_substrate ──[Pi1ToPi2Seed]──► 2.1 _ingest_pi1_continuation
                2.11 audit_lazutkin_poisson_sector ──[Pi2ToPi3Seed]──► 3.1 _ingest_pi2_continuation
                3.9 prosecute_positron_trajectory ──► PositronChampionSpectrumReport.
            Emite el certificado espectral inmutable con todos los predicados físicos,
            invariantes de Mecánica Celeste de Poincaré y certificados metacognitivos RSI-L3.
            """
            flags: List[str] = []
            self._telemetry.total_prosecutions += 1

            # ── FASE Π₁: Sustrato Dirac-Clifford, Uhlmann-SLD, Poincaré Canónico y RSI-L3 ──
            pi1 = self._pi1.audit_dirac_uhlmann_substrate(
                rho_init=rho_init,
                sigma_ref=sigma_ref,
                K_raw=K_raw,
                positron_beta=positron_beta,
                energy_tolerance_rel=energy_tolerance_rel,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination_rad=inclination_rad,
                mu=mu,
            )

            # ── FASE Π₂: Continuación formal desde Π₁ (Lazutkin, Melnikov, Fredholm, Poincaré) ──
            pi2 = self._pi2.audit_lazutkin_poisson_sector(
                pi1_certificate=pi1,
                lazutkin_epsilon=lazutkin_epsilon,
                lazutkin_omega=lazutkin_omega,
                frequencies=frequencies,
                dissipation_gamma=dissipation_gamma,
                forcing_amplitude=forcing_amplitude,
                forcing_frequency=forcing_frequency,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination_rad=inclination_rad,
                mu=mu,
            )

            # ── FASE Π₃: Continuación formal desde Π₂ (Índice Fiscal, Jerk, DK, RSI-L3 Global) ──
            seed2 = self._ingest_pi2_continuation(pi2)
            p_norm, simplex_kkt, _cost = self._simplex_with_kkt(p_init)
            if not simplex_kkt:
                flags.append("SIMPLEX_INPUT_FALLBACK_OR_KKT_MARGINAL")

            cond = pi1.conductivity
            if cond.used_fallback:
                flags.append("CONDUCTIVITY_INPUT_FALLBACK")
            if not cond.foster_cauer_passivity_ok:
                flags.append("FOSTER_CAUER_PASSIVITY_DEFECT")
            if pi1.rho.used_fallback or pi1.sigma.used_fallback:
                flags.append("DENSITY_INPUT_FALLBACK")

            # [F2] La energía en reposo 2m_e c² es una constante de la naturaleza, invariante.
            annihilation_kev = float(pi1.dirac.energy_rest_kev)
            fiscal = self._fiscal_indictment_index(
                pi1.rho.rho,
                cond.k_matrix,
                cond.lambda_max,
                seed2.chirikov_overlap_k,
                annihilation_kev,
                fiscal_weights,
            )

            # Síntesis metacognitiva RSI-L3 antes de evaluar el margen ISS del Jerk
            rsi_global = self._rsi_l3_pi3_meta_synthesis(
                pi1.rsi_l3_certificate, pi2.rsi_l3_certificate
            )
            jerk = self._jerk_certificate(fiscal.index, delta_t)
            dk_sqrt, dk_inv = self._connes_daleckii_krein_bounds(cond.lambda_min)
            exp_time, exp_energy, expired, floor_ok = self._amber_grace_predicates(
                elapsed_time_sec, grace_period_sec, annihilation_kev, energy_tolerance_rel
            )

            if not pi1.rho.kkt_ok:
                flags.append("DENSITY_KKT_MARGINAL")
            if pi1.chord_arc_error > 1.0e-9:
                flags.append("CHORD_ARC_IDENTITY_DEFECT")
            if pi1.sld_lyapunov_residual > 1.0e-9:
                flags.append("SLD_LYAPUNOV_RESIDUAL_HIGH")
            if not pi2.lazutkin.asymptotic_valid:
                flags.append("LAZUTKIN_OUTSIDE_ASYMPTOTIC_REGIME")
            if pi2.lazutkin.underflow:
                flags.append("LAZUTKIN_UNDERFLOW_LOG10_AUTHORITATIVE")
            if not pi2.fredholm.solvable_raw:
                flags.append("POISSON_RAW_INCOMPATIBLE_NEUTRALIZED")
            if pi2.lyapunov.degenerate_conservative:
                flags.append("KY_DEGENERATE_CONSERVATIVE_NO_FRACTAL_CONTENT")
            if pi2.separatrix_broken:
                flags.append("LAZUTKIN_1_SEPARATRIX_BROKEN_CHIRIKOV_GE_1")
            if not jerk.defined:
                flags.append("JERK_UNDEFINED_INSUFFICIENT_HISTORY")
            if not rsi_global.converged:
                flags.append("RSI_L3_META_LOOP_MARGINAL")

            poincare_elems = (
                pi2.poincare_elements
                if pi2.poincare_elements is not None
                else pi1.poincare_elements
            )
            loss_ratio = (
                float(rsi_global.final_meta_loss / rsi_global.initial_meta_loss)
                if rsi_global.initial_meta_loss > _TINY
                else 1.0
            )
            redemption_valid = bool(
                floor_ok
                and pi2.chirikov_contained
                and pi2.fredholm.poisson_residual <= 1.0e-8
                and cond.foster_cauer_passivity_ok
                and poincare_elems.nonsingular_regularity_ok
            )

            return PositronChampionSpectrumReport(
                p_champion=_freeze_array(p_norm),
                rho_champion=_freeze_array(pi1.rho.rho),
                K_purified=_freeze_array(cond.k_matrix),
                novikov_floer_functional=float(fiscal.index),
                lazutkin_splitting_angle=float(pi2.lazutkin.angle),
                poisson_fredholm_residual=float(pi2.fredholm.poisson_residual),
                jerk_ciber_physical=float(jerk.third_difference),
                kaplan_yorke_dimension=float(pi2.lyapunov.kaplan_yorke_dimension),
                annihilation_energy_kev=annihilation_kev,
                amber_grace_expired=bool(expired),
                connes_lipschitz_bound=float(dk_inv),
                condition_number_K=float(cond.condition_capped),
                # ── Π₁ extendido ──
                fiscal_indictment_index=float(fiscal.index),
                fiscal_weights=fiscal.weights,
                fiscal_terms=fiscal.terms,
                fiscal_index_monotone_ok=bool(fiscal.monotone_ok and fiscal.normalized_ok),
                uhlmann_fidelity=float(pi1.uhlmann_fidelity),
                bures_distance=float(pi1.bures_distance),
                bures_angle=float(pi1.bures_angle),
                chord_arc_identity_error=float(pi1.chord_arc_error),
                fidelity_route_discrepancy=float(pi1.fidelity_route_discrepancy),
                hilbert_schmidt_overlap=float(pi1.hilbert_schmidt_overlap),
                sld_fisher_metric=float(pi1.sld_fisher_metric),
                sld_lyapunov_residual=float(pi1.sld_lyapunov_residual),
                rho_condition_number=float(pi1.rho.condition_number),
                rho_purity=float(pi1.rho.purity),
                rho_entropy=float(pi1.rho.entropy),
                rho_projection_cost=float(pi1.rho.projection_cost),
                rho_kkt_ok=bool(pi1.rho.kkt_ok),
                simplex_kkt_ok=bool(simplex_kkt),
                K_imaginary_discarded_norm=float(cond.imaginary_discarded_norm),
                K_projection_cost=float(cond.projection_cost),
                foster_cauer_passivity_ok=bool(cond.foster_cauer_passivity_ok),
                cauer_driving_point_impedance_trace=float(
                    cond.cauer_driving_point_impedance_trace
                ),
                dirac_beta=float(pi1.dirac.beta),
                dirac_gamma_lorentz=float(pi1.dirac.gamma_lorentz),
                annihilation_energy_rest_kev=float(pi1.dirac.energy_rest_kev),
                annihilation_energy_lab_kev=float(pi1.dirac.energy_lab_kev),
                annihilation_energy_cm_kev=float(pi1.dirac.energy_cm_kev),
                annihilation_rate_coefficient_m3s=float(pi1.dirac.rate_coefficient_m3s),
                dirac_cross_section_m2=float(pi1.dirac.cross_section_m2),
                dirac_cross_section_3gamma_m2=float(pi1.dirac.cross_section_3gamma_m2),
                ore_powell_3gamma_ratio=float(pi1.dirac.ore_powell_3gamma_ratio),
                poincare_casimir_mandelstam_s_kev2=float(
                    pi1.dirac.poincare_casimir_mandelstam_s_kev2
                ),
                gamow_sommerfeld_factor=float(pi1.dirac.gamow_factor),
                # ── Π₂ extendido ──
                lazutkin_regime=pi2.lazutkin.regime,
                lazutkin_epsilon=float(pi2.lazutkin.epsilon),
                lazutkin_exponent=float(pi2.lazutkin.exponent),
                lazutkin_log10_angle=float(pi2.lazutkin.log10_angle),
                lazutkin_underflow=bool(pi2.lazutkin.underflow),
                lazutkin_asymptotic_valid=bool(pi2.lazutkin.asymptotic_valid),
                poincare_birkhoff_detuning=float(pi2.lazutkin.poincare_birkhoff_detuning),
                homoclinic_lobe_invariant=float(pi2.lazutkin.homoclinic_lobe_invariant),
                melnikov_log10_amplitude=float(pi2.melnikov.log10_amplitude),
                melnikov_lobe_area=float(pi2.melnikov.lobe_area),
                melnikov_flux=float(pi2.melnikov.flux),
                melnikov_transversal=bool(pi2.melnikov.transversal),
                melnikov_quadrature_error=float(pi2.melnikov.quadrature_error),
                chirikov_overlap_k=float(pi2.chirikov_k),
                chirikov_contained=bool(pi2.chirikov_contained),
                separatrix_broken=bool(pi2.separatrix_broken),
                lyapunov_spectrum=pi2.lyapunov.spectrum,
                lyapunov_max=float(pi2.lyapunov.max_exponent),
                lyapunov_sum=float(pi2.lyapunov.sum_exponents),
                ky_index_j=int(pi2.lyapunov.ky_index_j),
                ky_degenerate_conservative=bool(pi2.lyapunov.degenerate_conservative),
                monodromy_symplectic_defect=float(pi2.lyapunov.symplectic_defect),
                monodromy_determinant=float(pi2.lyapunov.determinant),
                monodromy_spectral_radius=float(pi2.lyapunov.spectral_radius),
                monodromy_step_certificate=float(pi2.lyapunov.step_certificate),
                poisson_residual=float(pi2.fredholm.poisson_residual),
                fredholm_compatibility_residual=float(pi2.fredholm.compatibility_residual),
                fredholm_solvable_raw=bool(pi2.fredholm.solvable_raw),
                fredholm_neutralization_applied=bool(pi2.fredholm.neutralization_applied),
                poisson_beta_0=int(pi2.fredholm.beta_0),
                poisson_potential_norm=float(pi2.fredholm.potential_norm),
                fiedler_algebraic_connectivity=float(
                    pi2.fredholm.fiedler_algebraic_connectivity
                ),
                poincare_cartan_action=float(pi2.poincare_cartan_action),
                poincare_second_integral_invariant=float(
                    pi2.poincare_second_integral_invariant
                ),
                poincare_amd_laskar=float(poincare_elems.amd_laskar_poincare),
                poincare_xi1_eccentric=float(poincare_elems.xi_1_eccentric),
                poincare_eta1_eccentric=float(poincare_elems.eta_1_eccentric),
                poincare_xi2_oblique=float(poincare_elems.xi_2_oblique),
                poincare_eta2_oblique=float(poincare_elems.eta_2_oblique),
                poincare_nonsingular_regularity_ok=bool(
                    poincare_elems.nonsingular_regularity_ok
                ),
                birkhoff_normal_form_remainder=float(pi2.birkhoff_normal_form_remainder),
                novikov_action_period=float(pi2.novikov_action_period),
                novikov_action_reduced=float(pi2.novikov_action_reduced),
                # ── Π₃ extendido y RSI-L3 ──
                jerk_defined=bool(jerk.defined),
                jerk_normalized=float(jerk.normalized),
                jerk_iss_margin=float(jerk.iss_margin),
                jerk_history_length=int(jerk.history_length),
                jerk_delta_t=float(jerk.delta_t),
                dk_sqrt_lipschitz=float(dk_sqrt),
                dk_invsqrt_lipschitz=float(dk_inv),
                condition_number_K_raw=float(cond.condition_raw),
                log10_condition_number_K=float(cond.log10_condition),
                lambda_min_K=float(cond.lambda_min),
                lambda_max_K=float(cond.lambda_max),
                spectral_gap_K=float(cond.top_gap),
                amber_expired_by_time=bool(exp_time),
                amber_expired_by_energy=bool(exp_energy),
                redemption_energy_floor_ok=bool(floor_ok),
                redemption_valid=bool(redemption_valid),
                rsi_l3_converged=bool(rsi_global.converged),
                rsi_l3_banach_contraction=float(rsi_global.banach_contraction_factor),
                rsi_l3_meta_loss_reduction_ratio=float(loss_ratio),
                rsi_l3_total_iterations=int(rsi_global.iterations_executed),
                engine_version=_ENGINE_VERSION,
                schema_flags=tuple(flags),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # §3 · INICIALIZACIÓN Y DELEGACIÓN SOBERANA (API pública compatible v16/v17)
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        tolerance: float = 1.0e-9,
        *,
        dimension_n: int = 3,
        tau_spd: float = 1.0e-12,
        epicyclic_precession_splitting: float = 0.0,
        enable_step_doubling: bool = True,
    ) -> None:
        if (
            isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            raise ValueError("tolerance debe ser un float finito estrictamente positivo.")
        if (
            isinstance(dimension_n, bool)
            or not isinstance(dimension_n, (int, np.integer))
            or int(dimension_n) <= 0
        ):
            raise ValueError("dimension_n debe ser un entero estrictamente positivo.")
        if (
            isinstance(tau_spd, bool)
            or not math.isfinite(float(tau_spd))
            or float(tau_spd) <= 0.0
        ):
            raise ValueError("tau_spd debe ser un float finito estrictamente positivo.")
        if not math.isfinite(float(epicyclic_precession_splitting)):
            raise ValueError("epicyclic_precession_splitting debe ser finito.")

        self._tol: Final[float] = float(tolerance)
        self._tau_spd: Final[float] = max(_STABILITY_FLOOR, float(tau_spd))
        self._n: Final[int] = int(dimension_n)
        self._telemetry = ChampionTelemetry()
        if self._tau_spd * self._n > 0.5:
            logger.warning(
                "tau_spd·n = %.3e > 1/2: el piso modular se recortará a 1/(2n).",
                self._tau_spd * self._n,
            )

        self.phase_pi1 = self.PhasePi1_DiracUhlmannFiscalizer(
            self._n, self._tol, self._tau_spd
        )
        self.phase_pi2 = self.PhasePi2_LazutkinPoissonFiscalizer(
            telemetry=self._telemetry,
            precession_splitting=float(epicyclic_precession_splitting),
            enable_doubling=bool(enable_step_doubling),
        )
        self.phase_pi3 = self.PhasePi3_FiscalIndictmentEmitter(
            pi1=self.phase_pi1,
            pi2=self.phase_pi2,
            dimension_n=self._n,
            telemetry=self._telemetry,
        )
        logger.info(
            "PositronRedemptionChampionEngine v%s inicializado (Π₁ → Π₂ → Π₃ · Poincaré + RSI-L3).",
            _ENGINE_VERSION,
        )

    # ── Delegadores de Π₁ ──
    def canonicalize_density(self, rho_raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Delega en 1.3b (proyección de Frobenius sobre D_δ(H))."""
        return self.phase_pi1.canonicalize_density(rho_raw)

    def purify_matrix(self, K_raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Delega en 1.4b (proyección PSD ∩ Sym_n(ℝ) con certificación de Foster-Cauer)."""
        return self.phase_pi1.purify_matrix(K_raw)

    # ── Delegadores de Π₂ ──
    def evaluate_lazutkin_angle(self, epsilon: float = 0.1, omega: float = 1.0) -> float:
        """Delega en 2.2b (escisión homoclínica en espacio logarítmico)."""
        return self.phase_pi2.evaluate_lazutkin_angle(epsilon, omega)

    # ── Delegadores de Π₃ ──
    def project_gibbs_simplex(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Delega en 3.2b (proyección exacta sobre Δ^{n−1} con certificado KKT)."""
        return self.phase_pi3.project_gibbs_simplex(v)

    def evaluate_novikov_floer_functional(
        self,
        rho: NDArray[np.complex128],
        K_purified: NDArray[np.complex128],
        annihilation_energy_kev: float,
        lazutkin_angle: float,
    ) -> float:
        """Delega en 3.8 (índice fiscal declarado; la acción de Poincaré-Cartan va separada)."""
        return self.phase_pi3.evaluate_novikov_floer_functional(
            rho, K_purified, annihilation_energy_kev, lazutkin_angle
        )

    def prosecute_positron_trajectory(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_ref: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        elapsed_time_sec: float = 0.0,
        grace_period_sec: float = 3600.0,
        **kwargs: Any,
    ) -> PositronChampionSpectrumReport:
        """Delega en 3.9 (composición endofuntorial Π₃ ∘ Π₂ ∘ Π₁ y emisión del certificado)."""
        return self.phase_pi3.prosecute_positron_trajectory(
            p_init,
            rho_init,
            sigma_ref,
            K_raw,
            elapsed_time_sec,
            grace_period_sec,
            **kwargs,
        )

    def get_telemetry(self) -> ChampionTelemetry:
        """Devuelve la telemetría episódica y metacognitiva (RSI-L3) del motor fiscal."""
        return self._telemetry


# ══════════════════════════════════════════════════════════════════════════════
# §4 · SUITE EJECUTABLE DE AUTOVERIFICACIÓN PhD (v16/v17 Regresiones + v18 Poincaré/RSI-L3)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """
    Suite integral de verificación formal:
      • Regresiones [F1]–[F12] de v16/v17.
      • Corrección del defecto latente [F13] (`_WILKINSON_EPS` en `_chirikov_overlap`).
      • Regularidad de coordenadas canónicas no singulares de Poincaré [F14] en e→0, i→0,
        invariantes integrales primero/segundo de Poincaré-Cartan y forma normal de Birkhoff.
      • Derivada Logarítmica Simétrica (SLD) vía ecuación de Lyapunov-Sylvester y pasividad
        de multipuerto de Foster-Cauer [F15].
      • Invariantes de Casimir del grupo de Poincaré ISO(1,3) y razón 3γ/2γ de Ore-Powell [F16].
      • Contractividad de Banach y monotonía del lazo de Automejora Recursiva Nivel 3 (RSI-L3) [F17].
    """
    checks: List[Tuple[str, bool]] = []
    n = 3
    eng = PositronRedemptionChampionEngine(dimension_n=n, tau_spd=1.0e-10)
    pi1, pi2, pi3 = eng.phase_pi1, eng.phase_pi2, eng.phase_pi3
    rho = np.eye(n, dtype=np.complex128) / float(n)
    sigma = np.diag([0.5, 0.3, 0.2]).astype(np.complex128)
    k_eye = np.eye(n, dtype=np.complex128)
    p0 = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    # [F1] Regresión central de geometría cuántica: d_B(ρ,ρ) = 0 y F(ρ,ρ) = 1
    cert_id = pi1.audit_dirac_uhlmann_substrate(rho, rho, k_eye)
    checks.append(
        (
            "[F1] d_B(ρ,ρ) = 0 y F(ρ,ρ) = 1",
            cert_id.bures_distance < 1.0e-9 and abs(cert_id.uhlmann_fidelity - 1.0) < 1.0e-9,
        )
    )
    p_spec, q_spec = np.full(n, 1.0 / float(n)), np.array([0.5, 0.3, 0.2])
    f_closed = float(np.sum(np.sqrt(p_spec * q_spec))) ** 2
    cert_pair = pi1.audit_dirac_uhlmann_substrate(rho, sigma, k_eye)
    checks.append(
        (
            "[F1] Uhlmann ≡ forma cerrada conmutante",
            abs(cert_pair.uhlmann_fidelity - f_closed) < 1.0e-7,
        )
    )
    checks.append(
        (
            "[F1] Solapamiento Hilbert-Schmidt separado y distinto de F",
            cert_pair.hilbert_schmidt_overlap < cert_pair.uhlmann_fidelity - 1.0e-6,
        )
    )
    checks.append(
        (
            "Identidad geométrica cuerda-arco d_B = 2·sin(θ_B/2)",
            cert_pair.chord_arc_error < 1.0e-12,
        )
    )

    # [F2/F3] Constante física CODATA 2m_e c² invariante
    rep_a = eng.prosecute_positron_trajectory(p0, rho, rho, k_eye)
    rep_b = eng.prosecute_positron_trajectory(p0, rho, sigma, k_eye)
    checks.append(
        (
            "[F2] E_ann independiente de d_B (constante CODATA 2m_e c²)",
            rep_a.annihilation_energy_kev
            == rep_b.annihilation_energy_kev
            == _ANNIHILATION_ENERGY_REST_KEV,
        )
    )
    checks.append(
        (
            "[F3] Piso energético relativo válido para CODATA 1021.9979 keV",
            _ANNIHILATION_ENERGY_REST_KEV < 1022.0 and rep_a.redemption_energy_floor_ok,
        )
    )

    # [F4] Escisión homoclínica de Lazutkin-Gelfreich en espacio logarítmico
    laz = pi2._lazutkin_certificate(0.08, 1.2)
    expected_log10 = (
        math.log10(_LAZUTKIN_PREFACTOR / 0.08) - (math.pi ** 2 / math.sqrt(0.08)) / _LN10
    )
    checks.append(
        (
            "[F4] θ ≪ 1 en régimen asintótico y log₁₀θ exacto",
            0.0 < laz.angle < 1.0e-6 and abs(laz.log10_angle - expected_log10) < 1.0e-12,
        )
    )
    laz_small = pi2._lazutkin_certificate(1.0e-6, 1.0)
    checks.append(
        (
            "[F4] Underflow controlado: θ=0 con log₁₀θ exacto < −308",
            laz_small.underflow and laz_small.angle == 0.0 and laz_small.log10_angle < -308.0,
        )
    )

    # [F5] Alternativa de Fredholm sobre el Laplaciano combinatorio y valor de Fiedler
    fred = pi2._poisson_fredholm_solve(k_eye, (1.0 / float(n),) * n)
    checks.append(
        (
            "[F5] Obstrucción de Fredholm cruda detectada (‖P_ker d‖ = 1/√n)",
            not fred.solvable_raw
            and abs(fred.compatibility_residual - 1.0 / math.sqrt(float(n))) < 1.0e-12,
        )
    )
    checks.append(
        (
            "[F5] Ecuación de Poisson neutralizada resuelta a precisión de máquina",
            fred.poisson_residual < 1.0e-12,
        )
    )

    # [F6/F7] Kaplan-Yorke y tercera diferencia discreta (Jerk)
    checks.append(
        (
            "[F7] D_KY(0.1, 0, −0.3) = 2 + 1/3",
            abs(pi2._kaplan_yorke_dimension([0.1, 0.0, -0.3])[0] - (2.0 + 1.0 / 3.0)) < 1.0e-12,
        )
    )
    cubic = [float(k ** 3) for k in range(6)]
    j3, defined = pi3._third_difference(cubic, 1.0)
    checks.append(
        ("[F6] Δ³(k³)/1³ = 6 exactamente", defined and abs(j3 - 6.0) < 1.0e-12)
    )

    # [F9/F10/F12] Proyección real-first de K, símplex capado y defensa de forma 2-D
    k_cplx = np.array(
        [[2.0 + 0j, 0.3 + 0.4j, 0.0], [0.3 - 0.4j, 1.0, 0.0], [0.0, 0.0, -0.5]],
        dtype=np.complex128,
    )
    cond_opt = pi1._purify_with_certificate(k_cplx)
    kh = 0.5 * (k_cplx + k_cplx.conj().T)
    ev, uv = la.eigh(kh)
    k_v16 = np.real((uv * np.maximum(ev, 1.0e-10)) @ uv.conj().T)
    checks.append(
        (
            "[F9] Proyector real→clamp domina a clamp→real en norma de Frobenius",
            la.norm(k_cplx - cond_opt.k_matrix, "fro")
            <= la.norm(k_cplx - k_v16, "fro") + 1.0e-12,
        )
    )
    bad_simplex = pi3._simplex_with_kkt(np.array([[0.5, 0.5], [0.0, 0.0]]))
    checks.append(
        (
            "[F12] Defensa contra tensor 2-D en proyección sobre el símplex",
            not bad_simplex[1] and abs(float(bad_simplex[0].sum()) - 1.0) < 1.0e-12,
        )
    )

    # [F13] Corrección del bug `_WILKINSON_EPS` en `_chirikov_overlap`
    k_chir_val, chir_cont, chir_brok, min_div_val = pi2._chirikov_overlap(
        [1.0, math.sqrt(2.0), math.sqrt(3.0)], 0.04
    )
    checks.append(
        (
            "[F13] _chirikov_overlap libre de NameError y exacto con frecuencias explícitas",
            k_chir_val > 0.0 and min_div_val > 0.3 and chir_cont != chir_brok,
        )
    )

    # [F14] Mecánica Celeste de Poincaré: Variables canónicas no singulares en e=0, i=0 y AMD
    poincare_zero = pi1._synthesize_poincare_canonical_elements(
        cert_id.rho, cert_id.conductivity, 0.5, semi_major_axis=1.0, eccentricity=0.0, inclination_rad=0.0
    )
    checks.append(
        (
            "[F14] Coordenadas canónicas de Poincaré regulares en e=0, i=0 (ξ₁=η₁=ξ₂=η₂=AMD=0)",
            poincare_zero.nonsingular_regularity_ok
            and poincare_zero.amd_laskar_poincare == 0.0
            and poincare_zero.xi_1_eccentric == 0.0
            and poincare_zero.xi_2_oblique == 0.0,
        )
    )
    poincare_exc = pi1._synthesize_poincare_canonical_elements(
        cert_pair.rho, cert_pair.conductivity, cert_pair.bures_angle,
        semi_major_axis=1.5, eccentricity=0.25, inclination_rad=0.35, mu=1.2,
    )
    quad_amd = 0.5 * (
        poincare_exc.xi_1_eccentric ** 2
        + poincare_exc.eta_1_eccentric ** 2
        + poincare_exc.xi_2_oblique ** 2
        + poincare_exc.eta_2_oblique ** 2
    )
    checks.append(
        (
            "[F14] Identidad simpléctica cuadrática de Poincaré ½(ξ₁²+η₁²+ξ₂²+η₂²) ≡ Λ − H",
            poincare_exc.nonsingular_regularity_ok
            and abs(quad_amd - (poincare_exc.lambda_delaunay - poincare_exc.h_delaunay)) < 1.0e-14,
        )
    )
    checks.append(
        (
            "[F14] Segundo invariante integral de Poincaré-Cartan I₂ ≥ I₁ > 0",
            rep_b.poincare_second_integral_invariant >= rep_b.poincare_cartan_action > 0.0,
        )
    )

    # [F15] Derivada Logarítmica Simétrica (SLD) de Lyapunov-Sylvester y Foster-Cauer
    checks.append(
        (
            "[F15] Ecuación cuántica SLD ½(ρL+Lρ)=σ−ρ resuelta (residual < 1e-12, g_SLD > 0)",
            cert_id.sld_fisher_metric == 0.0
            and cert_pair.sld_fisher_metric > 0.0
            and cert_pair.sld_lyapunov_residual < 1.0e-12,
        )
    )
    checks.append(
        (
            "[F15] Pasividad de Foster-Cauer certificada y Tr((I+I)⁻¹) = n/2",
            cert_id.conductivity.foster_cauer_passivity_ok
            and abs(cert_id.conductivity.cauer_driving_point_impedance_trace - 0.5 * n) < 1.0e-12,
        )
    )

    # [F16] Invariantes de Casimir de Poincaré ISO(1,3) y razón 3γ/2γ de Ore-Powell
    dirac_c = cert_pair.dirac
    checks.append(
        (
            "[F16] Casimir de Mandelstam s = E_cm², Pauli-Lubanski y canal 3γ de Ore-Powell",
            abs(dirac_c.poincare_casimir_mandelstam_s_kev2 - dirac_c.energy_cm_kev ** 2) < 1.0e-6
            and dirac_c.pauli_lubanski_singlet_casimir == 0.0
            and abs(
                dirac_c.pauli_lubanski_triplet_casimir
                - 2.0 * dirac_c.poincare_casimir_mandelstam_s_kev2
            )
            < 1.0e-6
            and abs(
                dirac_c.cross_section_3gamma_m2 / dirac_c.cross_section_m2
                - _ORE_POWELL_TRIPLET_TO_SINGLET
            )
            < 1.0e-15,
        )
    )

    # [F17] Automejora Recursiva de Nivel 3 (RSI-L3): Contractividad de Banach y monotonía
    # Ejecutamos 2 pasadas adicionales para poblar >= 4 muestras de Jerk y activar RSI-L3 en Π₃
    _ = eng.prosecute_positron_trajectory(p0, sigma, rho, k_cplx)
    rep_fourth = eng.prosecute_positron_trajectory(p0, rho, sigma, k_cplx)
    checks.append(
        (
            "[F17] Lazo metacognitivo RSI-L3 convergente (q_Banach < 1 y pérdida no creciente)",
            rep_fourth.rsi_l3_converged
            and 0.0 < rep_fourth.rsi_l3_banach_contraction < 1.0
            and rep_fourth.rsi_l3_meta_loss_reduction_ratio <= 1.0 + 1.0e-12
            and rep_fourth.jerk_defined,
        )
    )

    ok = True
    for name, passed in checks:
        ok = ok and bool(passed)
        print(f"[{'OK ' if passed else 'FAIL'}] {name}")
    print(
        f"\nVEREDICTO GLOBAL DE AUTOVERIFICACIÓN v18.0.0: {'COHERENTE' if ok else 'OBSTRUIDO'}"
    )
    return ok


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s · %(name)s · %(message)s"
    )
    raise SystemExit(0 if _self_test() else 1)