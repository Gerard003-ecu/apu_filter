# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Positron Redemption Governor Engine (Motor Espectral Ciego de FPU)                         ║
║ RUTA    : app/omega/positron_redemption_governor_engine.py                                           ║
║ VERSIÓN : 18.0.0-FPU-Positron-Dirac-Uhlmann-Holonomy-Poincare-KAM-Yoshida-Krein-RSI3-PhD             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
NATURALEZA DEL MOTOR (inalterada respecto de v16/v17)
──────────────────────────────────────────────────────────────────────────────────────────────────────
Funtor covariante de transformación determinista en lazo abierto que habita la FPU del Estrato Omega
(V_Ω, Nivel 0.5 — Ágora Tensorial). NO evalúa retículos de Heyting, NO dicta veredictos de negocio,
NO mantiene estados agénticos OODA, NO toca periféricos. Emite únicamente el certificado numérico
inmutable `PositronRedemptionSpectrumReport` consumido por el soberano agéntico de calibre.
Toda la automejora recursiva de nivel 3 (RSI³) es ESTRICTAMENTE NUMÉRICA (optimización espectral del
piso regularizador de Mirsky-Daleckii-Krein, extrapolación homotópica de Richardson y refinamiento de
sub-paso simpléctico/isoespectral), nunca de política de negocio.

TRES FASES ESPECTRALES ANIDADAS CON CONTINUIDAD FORMAL ESTRICTA DE TIPOS
──────────────────────────────────────────────────────────────────────────────────────────────────────
    1.12 audit_dirac_uhlmann_substrate ──[Pi1ToPi2Seed]──► 2.1  _ingest_pi1_continuation
    2.10 audit_poincare_celestial_sector ──[Pi2ToPi3Seed]──► 3.1  _ingest_pi2_continuation
    3.11 execute_positron_redemption_trajectory = composición R₃∘R₂∘R₁ y emisión del certificado.

FASE Π₁ · ANIQUILACIÓN DE DIRAC-ORE-POWELL, HOLONOMÍA DE UHLMANN-BURES, TOMITA-TAKESAKI Y GKSL
  • [RSI³ Meta-Piso 1.3] Optimización cerrada del piso espectral δ*(λ) minimizando el funcional de
    riesgo compuesto de Mirsky-Daleckii-Krein: sesgo de Frobenius Σ(δ−λ_i)² vs. sensibilidad de raíz
    operatorial L_DK² ∝ ε_mach²/(4δ), con factibilidad garantizada δ ≤ 1/(2n).
  • Proyección EXACTA de Frobenius sobre D_δ(H)={ρ=ρ†⪰δI, Tr ρ=1} por water-filling espectral
    (Duchi-Condat + Mirsky/Hoffman-Wielandt) con corrección aditiva preservadora de KKT.
  • [Holonomía de Uhlmann y Cuaterniones 1.6] Descomposición polar √ρ√σ = |√ρ√σ|·V_Uhl mediante SVD
    √ρ√σ = U Σ W† ⟹ V_Uhl = U W† ∈ U(n), fase geométrica de Pancharatnam-Uhlmann Φ_Uhl = arg Tr(√ρ√σ)
    y proyección del sector principal 2×2 sobre la 3-esfera de cuaterniones unitarios S³ ⊂ ℍ en la
    base de Cayley-Dickson/Pauli {I₂, −iσ_x, −iσ_y, −iσ_z}.
  • Fidelidad de Uhlmann √F = ‖√ρ√σ‖₁ contrastada con la ruta espectral √(√ρσ√ρ), distancia de Bures
    d_B = √(2−2√F), ángulo estadístico θ_B = arccos√F, identidad cuerda-arco d_B = 2·sin(θ_B/2) y
    extrapolación homotópica de Richardson en δ → δ/2.
  • Entropía relativa de Umegaki con logs espectrales exactos (sin logm), condicionamiento del
    operador modular de Tomita-Takesaki Δ_{σ,ρ} = L_σ R_ρ^{−1}, cota cuántica de Pinsker D ≥ ½‖ρ−σ‖₁²
    y sándwich exacto de Fuchs-van de Graaf: 1 − √F(ρ,σ) ≤ D_tr(ρ,σ) ≤ √(1 − F(ρ,σ)).
  • Información oblicua cuántica de Wigner-Yanase-Dyson I_{1/2}(ρ, −ln σ) = −½ Tr([√ρ, ln σ]²) ≥ 0.
  • Sección eficaz de aniquilación de Dirac (e⁻+e⁺→2γ) libre de cancelación catastrófica mediante la
    parametrización relativista exacta √(γ²−1) = βγ y arcosh(γ) = ln(γ + βγ), factor de Coulomb de
    Gamow-Sommerfeld S(η)=2πη/(1−e^{−2πη}) y canal de 3 fotones de Ore-Powell (orto-positronio ³S₁→3γ):
    σ_{3γ}/σ_{2γ,NR} = [4(π²−9)/(9π)]·α_fs.
  • Canal GKSL de desfase (pinching) ρ ↦ (1−p)ρ + p·Δ(ρ), p = 1−e^{−γ_d t}, con certificado de
    positividad completa de Choi-Jamiołkowski y monotonía de entropía ΔS ≥ 0.

FASE Π₂ · MECÁNICA CELESTE DE POINCARÉ: INVARIANTE DE POINCARÉ-CARTAN, KEPLER, MELNIKOV, YOSHIDA-KREIN, KAM
  • Carta canónica regularizada de Poincaré en el espacio de fases extendido: verificación del
    invariante integral de Poincaré-Cartan Jᵀ Ω₄ J = Ω₄ sobre los dos pares canónicos (ξ₁,η₁,ξ₂,η₂),
    round-trip exacto sin polo 1/√Γ y condición diofántica KAM de no resonancia.
  • Ecuación de Kepler E − e·sin E = M por método de Halley de tercer orden con salvaguarda de
    bisección monótona en [M−e, M+e] y auditoría espectral sobre el toro de anomalías medias.
  • Teoría de Poincaré-Melnikov sobre la separatriz homoclínica de Duffing: área del lóbulo del
    turnstile de MacKay-Meiss-Percival, flujo de transporte en el espacio de fases, ángulo de
    intersección homoclínica transversal y verificación por cuadratura de Gauss-Trapecio.
  • Monodromía de Floquet epicíclica de Hill-Poincaré integrada mediante composición simpléctica de
    Yoshida de 4.º orden / exponencial exacta a trozos, signaturas simplécticas de Krein
    κ(ξ) = sgn(−i ξ†Ωξ), número de rotación simpléctico e índice de bifurcación de Hamilton-Hopf.
  • Serie perturbativa de Lindstedt-Poincaré libre de términos seculares contrastada con la
    resumación elíptica exacta mediante la Media Aritmético-Geométrica (AGM) de Gauss-Salamin.

FASE Π₃ · RSI³ = R₃∘R₂∘R₁ SOBRE Δ^{n−1} × D(H), SUB-STEPPING DE STRANG-BCH Y EMISIÓN
  • R₁ (Shahshahani): flujo exponencial del replicador estabilizado en dominio logarítmico.
  • R₃ (Brockett): flujo cuántico isoespectral de doble corchete ρ̇ = [ρ,[ρ,N(p)]], integrado
    unitariamente sin re-clampar el espectro dentro del lazo [P1].
  • [RSI³ Adaptación de Sub-paso] Estimador del conmutador de Baker-Campbell-Hausdorff entre los
    generadores de Shahshahani y Brockett para seleccionar el refinamiento m* ∈ {1,2,4} del
    splitting simétrico de Strang de orden 2, certificado empíricamente por extrapolación de Richardson.
  • Balance energético de Lyapunov descompuesto H(p,ρ) = −½pᵀKp + S(ρ), cotas de sensibilidad
    operatorial de Connes-Daleckii-Krein y congelación defensiva de buffers (`write=False`).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Omega.PositronRedemptionGovernorEngine")

__all__ = [
    "PositronRedemptionSpectrumReport",
    "PositronRedemptionGovernorEngine",
    "Pi1SpectrumCertificate",
    "Pi2SpectrumCertificate",
    "UhlmannHolonomyCertificate",
    "ENGINE_VERSION",
]

_ENGINE_VERSION: Final[str] = (
    "18.0.0-FPU-Positron-Dirac-Uhlmann-Holonomy-Poincare-KAM-Yoshida-Krein-RSI3-PhD"
)
version: Final[str] = _ENGINE_VERSION
ENGINE_VERSION: Final[str] = _ENGINE_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES FÍSICAS (CODATA), HIPERCOMPLEJAS, NUMÉRICAS Y DE MODELO
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_STABILITY_FLOOR: Final[float] = max(1.0e-12, _MACHINE_EPS)
_TINY: Final[float] = 1.0e-300
_WILKINSON_EPS: Final[float] = 1.0e-15
_CONDITION_CAP: Final[float] = 1.0e12
_MAX_MODULAR_CONDITION: Final[float] = 1.0e8

_ELECTRON_REST_MASS_KEV: Final[float] = 510.998950003          # m_e c² [keV]
_ANNIHILATION_ENERGY_KEV: Final[float] = 2.0 * _ELECTRON_REST_MASS_KEV
_CLASSICAL_ELECTRON_RADIUS_M: Final[float] = 2.8179403262e-15  # r_e [m]
_FINE_STRUCTURE_ALPHA: Final[float] = 7.2973525693e-3          # α_fs ≈ 1/137.035999
_SPEED_OF_LIGHT_MS: Final[float] = 299792458.0                 # c [m/s]
_PI_R_E_SQUARED: Final[float] = math.pi * _CLASSICAL_ELECTRON_RADIUS_M ** 2
# Razón radiativa de Ore-Powell (1949) para orto-positronio (³S₁ → 3γ vs ¹S₀ → 2γ):
_ORE_POWELL_3GAMMA_RATIO: Final[float] = (
    (4.0 * (math.pi ** 2 - 9.0)) / (9.0 * math.pi)
) * _FINE_STRUCTURE_ALPHA

# Coeficientes simplécticos de Yoshida de 4.º orden (1990):
_CBRT_2: Final[float] = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - _CBRT_2)
_YOSHIDA_W0: Final[float] = -_CBRT_2 / (2.0 - _CBRT_2)

_HALLEY_MAX_ITER: Final[int] = 40
_MONODROMY_BASE_PANELS: Final[int] = 96
_MONODROMY_MAX_DOUBLINGS: Final[int] = 6
_AGM_MAX_ITER: Final[int] = 60


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión total a float finito: la FPU nunca propaga NaN/Inf al certificado."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fro(mat: NDArray[Any]) -> float:
    """Norma de Hilbert-Schmidt / Frobenius ‖M‖_F = √Tr(M†M)."""
    return float(la.norm(mat, ord="fro")) if getattr(mat, "size", 0) else 0.0


def _relative(defect: float, scale: float) -> float:
    """Defecto relativo estabilizado frente a denominador nulo."""
    return float(defect / max(abs(scale), _TINY))


def _freeze_array(arr: NDArray[Any]) -> NDArray[Any]:
    """
    [P14] Copia defensiva con buffer de sólo lectura: garantiza inmutabilidad profunda
    de los tensores entregados en `PositronRedemptionSpectrumReport` sin mutar al llamador.
    """
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _arithmetic_geometric_mean(a0: float, b0: float, tol: float = 1.0e-16) -> Tuple[float, int]:
    """
    Media Aritmético-Geométrica (AGM) de Gauss-Brent-Salamin con convergencia cuadrática.
    Satisface K(k) = π / (2 · AGM(1, √(1 − k²))).
    """
    a, b = abs(float(a0)), abs(float(b0))
    if a <= _TINY and b <= _TINY:
        return 0.0, 0
    for k in range(1, _AGM_MAX_ITER + 1):
        if abs(a - b) <= tol * max(a, b):
            return 0.5 * (a + b), k
        a, b = 0.5 * (a + b), math.sqrt(max(0.0, a * b))
    return 0.5 * (a + b), _AGM_MAX_ITER


def _pauli_quaternion_projection(
    mat: NDArray[np.complex128],
) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Proyección del bloque principal 2×2 de un operador sobre el álgebra de cuaterniones ℍ
    vía el isomorfismo de Cayley-Dickson / Pauli SU(2) ≅ S³ ⊂ ℍ:
        q₀ = ½ Re Tr(M₂),   q₁ = −½ Im Tr(σ_x M₂),
        q₂ = −½ Im Tr(σ_y M₂),   q₃ = −½ Im Tr(σ_z M₂).
    Normaliza sobre S³ y devuelve ((q₀, q₁, q₂, q₃), defecto_de_unitariedad_S3).
    """
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        q0 = float(np.real(mat[0, 0])) if mat.size else 1.0
        norm = abs(q0) if abs(q0) > _TINY else 1.0
        return (q0 / norm, 0.0, 0.0, 0.0), abs(norm - 1.0)
    m2 = mat[:2, :2]
    q0 = 0.5 * float(np.real(m2[0, 0] + m2[1, 1]))
    q1 = -0.5 * float(np.imag(m2[0, 1] + m2[1, 0]))
    q2 = -0.5 * float(np.real(m2[0, 1] - m2[1, 0]))
    q3 = -0.5 * float(np.imag(m2[0, 0] - m2[1, 1]))
    norm = math.sqrt(max(_TINY, q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3))
    unit_q = (float(q0 / norm), float(q1 / norm), float(q2 / norm), float(q3 / norm))
    return unit_q, float(abs(norm - 1.0))


def _symplectic_omega(half_dim: int) -> NDArray[np.float64]:
    """Construye la matriz simpléctica estándar Ω_{2m} = diag(J₂, ..., J₂) en ℝ^{2m×2m}."""
    j2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    blocks = [j2 for _ in range(max(1, int(half_dim)))]
    return np.asarray(la.block_diag(*blocks), dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# §1 · CERTIFICADOS DE FASE, SEMILLAS DE CONTINUIDAD E INFORME ESPECTRAL
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DensityCanonicalization:
    """Proyección exacta sobre D_δ(H) con piso meta-optimizado RSI³ y certificados KKT/Mirsky."""
    rho: NDArray[np.complex128]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    reference_probs: NDArray[np.float64]
    entropy: float
    purity: float
    condition_number: float
    projection_cost: float
    floor_used: float
    kkt_ok: bool
    hermiticity_defect: float
    trace_defect: float
    used_fallback: bool
    rsi3_meta_risk: float = 0.0
    linear_entropy: float = 0.0


@dataclass(frozen=True)
class UhlmannHolonomyCertificate:
    """
    Certificado topológico-hipercomplejo del transporte paralelo de Uhlmann:
    √ρ√σ = |√ρ√σ| V_Uhl, fase de Pancharatnam-Uhlmann y proyección cuaterniónica S³ ⊂ ℍ.
    """
    holonomy_unitary: NDArray[np.complex128]
    pancharatnam_phase_rad: float
    determinant_phase_rad: float
    unitarity_defect: float
    quaternion_s3: Tuple[float, float, float, float]
    quaternion_norm_defect: float


@dataclass(frozen=True)
class DiracAnnihilationCertificate:
    """Sección eficaz de Dirac exacta (2γ), canal de Ore-Powell (3γ), factor de Gamow y cinemática."""
    beta: float
    gamma_lorentz: float
    cross_section_m2: float
    cross_section_reduced: float
    gamow_factor: float
    nr_gamow_cross_section_m2: float
    nr_to_exact_ratio: float
    rate_coefficient_m3s: float
    energy_rest_kev: float
    energy_lab_kev: float
    energy_cm_kev: float
    nr_regime_valid: bool
    ore_powell_3gamma_cross_section_m2: float = 0.0
    beta_gamma_momentum: float = 0.0


@dataclass(frozen=True)
class Pi1ToPi2Seed:
    """[MORFISMO 1.12 → 2.1] Semilla geométrico-exergética-cuántica que abre la Fase Π₂."""
    bures_tension: float
    umegaki_divergence: float
    annihilation_energy_kev: float
    annihilation_rate_coefficient: float
    rho_entropy: float
    rho_condition_number: float
    has_context: bool = True
    pancharatnam_phase_rad: float = 0.0
    wigner_yanase_skew_info: float = 0.0
    tomita_modular_condition: float = 1.0


@dataclass(frozen=True)
class Pi1SpectrumCertificate:
    """Certificado numérico completo de la Fase Π₁ (sin veredictos: la FPU es ciega)."""
    rho: NDArray[np.complex128]
    sigma: NDArray[np.complex128]
    rho_canon: DensityCanonicalization
    sigma_canon: DensityCanonicalization
    uhlmann_fidelity: float
    bures_distance: float
    bures_angle: float
    chord_arc_error: float
    fidelity_route_discrepancy: float
    umegaki_relative_entropy: float
    trace_distance: float
    pinsker_bound: float
    pinsker_residual: float
    pinsker_ok: bool
    dirac: DiracAnnihilationCertificate
    amber_grace_duration_sec: float
    continuation: Optional[Pi1ToPi2Seed] = None
    holonomy: Optional[UhlmannHolonomyCertificate] = None
    fuchs_lower_margin: float = 0.0
    fuchs_upper_margin: float = 0.0
    fuchs_sandwich_ok: bool = True
    tomita_modular_condition: float = 1.0
    wigner_yanase_skew_info: float = 0.0
    fidelity_homotopy_drift: float = 0.0


@dataclass(frozen=True)
class PoincareChartCertificate:
    """Carta regularizada de Poincaré con invariante integral de Poincaré-Cartan y condición KAM."""
    chart: NDArray[np.float64]
    mean_longitude: float
    algebraic_identity_residual: float
    roundtrip_error: float
    canonicity_defect: float
    chart_condition_number: float
    poincare_cartan_symplectic_defect_4d: float = 0.0
    kam_diophantine_margin: float = 0.0


@dataclass(frozen=True)
class KeplerCertificate:
    """Solución de Kepler con residuo a posteriori y orden empírico sobre el toro de anomalías."""
    eccentric_anomaly: float
    residual: float
    iterations: int
    worst_residual: float
    mean_iterations: float
    empirical_order: float
    bracket_safeguard_hits: int
    fourier_bessel_consistency_error: float = 0.0


@dataclass(frozen=True)
class MelnikovCertificate:
    """Turnstile de MacKay-Meiss-Percival: área, flujo, ángulo homoclínico y cuadratura."""
    lobe_area: float
    flux: float
    dissipative_barrier: float
    oscillation_amplitude: float
    amplitude_ratio: float
    transversal: bool
    critical_forcing: float
    quadrature_error: float
    homoclinic_splitting_angle_rad: float = 0.0


@dataclass(frozen=True)
class KreinMonodromyCertificate:
    """Monodromía de Floquet epicíclica (Yoshida 4.º orden + exponencial exacta) y teoría de Krein."""
    symplectic_defect: float
    determinant: float
    spectral_radius: float
    max_lyapunov_exponent: float
    krein_signatures: Tuple[float, ...]
    collision_detected: bool
    step_certificate: float
    panels: int
    radial_epicyclic_freq: float
    vertical_epicyclic_freq: float
    yoshida4_agreement_defect: float = 0.0
    poincare_rotation_number: float = 0.0
    hamilton_hopf_index: float = 0.0


@dataclass(frozen=True)
class Pi2ToPi3Seed:
    """[MORFISMO 2.10 → 3.1] Semilla celeste que parametriza la integración RSI³."""
    modulation_amplitude: float
    epicyclic_frequency: float
    melnikov_flux: float
    lindstedt_remainder: float
    kepler_residual: float
    has_context: bool = True
    poincare_rotation_number: float = 0.0
    kam_diophantine_margin: float = 0.0
    bures_tension_forwarded: float = 0.0


@dataclass(frozen=True)
class Pi2SpectrumCertificate:
    """Certificado numérico de la Fase Π₂ (mecánica celeste de Poincaré)."""
    chart: PoincareChartCertificate
    kepler: KeplerCertificate
    melnikov: MelnikovCertificate
    krein: KreinMonodromyCertificate
    lindstedt_omega_series: float
    lindstedt_omega_exact: float
    lindstedt_remainder_rel: float
    agm_iterations: int
    continuation: Optional[Pi2ToPi3Seed] = None


@dataclass(frozen=True)
class PositronRedemptionSpectrumReport:
    r"""
    Certificado espectral inmutable emitido por la FPU para el soberano agéntico.
    Orden posicional de v16 y v17 preservado íntegramente; los campos nuevos de v18
    se sitúan al final con valores por defecto. Los ndarray se entregan congelados [P14].
    """
    p_final: NDArray[np.float64]
    rho_final: NDArray[np.complex128]
    K_purified: NDArray[np.float64]
    uhlmann_fidelity: float
    bures_distance: float
    umegaki_relative_entropy: float
    pinsker_bound_residual: float
    annihilation_energy_kev: float
    amber_grace_duration_sec: float
    poincare_canonical_residual: float
    halley_kepler_residual: float
    melnikov_lobe_area: float
    krein_poincare_cartan_drift: float
    hamiltonian_energy: float
    analytic_energy_derivative: float
    numerical_energy_derivative: float
    shahshahani_speed: float
    brockett_alignment: float
    commutator_frobenius: float
    connes_lipschitz_bound: float
    condition_number_K: float
    spectral_gap_K: float
    schema_version: str = (
        "18.0.0-PhD-Positron-Dirac-Uhlmann-Holonomy-Poincare-KAM-Yoshida-Krein-RSI3"
    )
    # ── Π₁ extendido (v17) ──
    bures_angle: float = 0.0
    chord_arc_identity_error: float = 0.0
    fidelity_route_discrepancy: float = 0.0
    trace_distance: float = 0.0
    pinsker_bound: float = 0.0
    pinsker_ok: bool = True
    rho_condition_number: float = 1.0
    rho_purity: float = 1.0
    rho_projection_cost: float = 0.0
    rho_kkt_ok: bool = True
    von_neumann_entropy_final: float = 0.0
    dirac_beta: float = 0.0
    dirac_gamma_lorentz: float = 1.0
    dirac_cross_section_m2: float = 0.0
    dirac_cross_section_reduced: float = 0.0
    gamow_sommerfeld_factor: float = 1.0
    nr_gamow_to_exact_ratio: float = 1.0
    annihilation_rate_coefficient_m3s: float = 0.0
    annihilation_energy_lab_kev: float = 0.0
    annihilation_energy_cm_kev: float = 0.0
    # ── Π₂ extendido (v17) ──
    poincare_canonicity_defect: float = 0.0
    poincare_roundtrip_error: float = 0.0
    poincare_mean_longitude: float = 0.0
    kepler_worst_residual: float = 0.0
    kepler_mean_iterations: float = 0.0
    kepler_empirical_order: float = 0.0
    melnikov_flux: float = 0.0
    melnikov_transversal: bool = False
    melnikov_critical_forcing: float = 0.0
    melnikov_quadrature_error: float = 0.0
    monodromy_symplectic_defect: float = 0.0
    monodromy_determinant: float = 1.0
    monodromy_spectral_radius: float = 1.0
    krein_collision_detected: bool = False
    monodromy_step_certificate: float = 0.0
    lindstedt_omega_series: float = 0.0
    lindstedt_omega_exact: float = 0.0
    lindstedt_remainder_rel: float = 0.0
    agm_iterations: int = 0
    # ── Π₃ extendido (v17) ──
    strang_observed_order: float = 0.0
    strang_step_certificate: float = 0.0
    isospectrality_defect: float = 0.0
    simplex_kkt_ok: bool = True
    replicator_variance: float = 0.0
    lyapunov_replicator_term: float = 0.0
    gksl_entropy_production: float = 0.0
    gksl_trace_defect: float = 0.0
    gksl_cptp_ok: bool = True
    total_energy_derivative: float = 0.0
    lyapunov_passivity_ok: bool = True
    derivative_discrepancy_rel: float = 0.0
    dk_sqrt_lipschitz: float = 0.0
    dk_invsqrt_lipschitz: float = 0.0
    brockett_min_rate: float = 0.0
    condition_number_K_uncapped: float = 1.0
    log10_condition_number_K: float = 0.0
    integration_steps: int = 0
    integration_dt: float = 0.0
    engine_version: str = _ENGINE_VERSION
    schema_flags: Tuple[str, ...] = ()
    # ── Π₁ + Π₂ + Π₃ evolución granular v18.0.0 ──
    uhlmann_pancharatnam_phase_rad: float = 0.0
    uhlmann_holonomy_unitarity_defect: float = 0.0
    uhlmann_quaternion_s3: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fuchs_van_de_graaf_ok: bool = True
    fuchs_lower_margin: float = 0.0
    fuchs_upper_margin: float = 0.0
    tomita_modular_condition: float = 1.0
    wigner_yanase_skew_info: float = 0.0
    fidelity_homotopy_drift: float = 0.0
    ore_powell_3gamma_cross_section_m2: float = 0.0
    poincare_cartan_symplectic_defect_4d: float = 0.0
    kam_diophantine_margin: float = 0.0
    kepler_fourier_bessel_error: float = 0.0
    melnikov_splitting_angle_rad: float = 0.0
    yoshida4_monodromy_defect: float = 0.0
    poincare_rotation_number: float = 0.0
    hamilton_hopf_index: float = 0.0
    rsi3_substeps_per_step: int = 1
    bch_commutator_norm: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §2 · MOTOR ESPECTRAL CIEGO: TRES FASES Π ANIDADAS
# ══════════════════════════════════════════════════════════════════════════════
class PositronRedemptionGovernorEngine:
    """
    Motor Espectral Ciego de FPU para la Redención de Positrones y Mecánica Celeste.
    Tres fases anidadas con continuidad formal estricta:
        Π₁ · PhasePi1_DiracUhlmannAnnihilator     (1.1 → 1.12)
        Π₂ · PhasePi2_PoincareCelestialRegularizer (2.1 → 2.10)
        Π₃ · PhasePi3_RSI3TrajectoryIntegrator     (3.1 → 3.11)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₁ · ANIQUILACIÓN DE DIRAC, HOLONOMÍA DE UHLMANN-BURES, UMEGAKI Y GKSL
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi1_DiracUhlmannAnnihilator:
        """
        Π₁. Cadena formal:
        1.1 validación hermítica → 1.2 símplex capado → 1.3 meta-piso RSI³ de Mirsky-DK →
        1.4 canonización espectral → 1.5 raíces operatoriales → 1.6 holonomía de Uhlmann y
        cuaterniones S³ → 1.7 geometría de Bures-Uhlmann con extrapolación homotópica →
        1.8 Umegaki, Tomita-Takesaki y sándwich de Fuchs-van de Graaf → 1.9 información
        oblicua de Wigner-Yanase-Dyson → 1.10 Dirac 2γ + Ore-Powell 3γ + Gamow →
        1.11 GKSL y rampa Ámbar → 1.12 TERMINAL (emite Pi1ToPi2Seed hacia 2.1).
        """

        def __init__(
            self,
            n: int,
            stability_threshold: float,
            amber_base: float,
            amber_alpha: float,
            amber_reference_kev: float,
        ) -> None:
            self._n = int(n)
            self._floor = float(stability_threshold)
            self._amber_base = float(amber_base)
            self._amber_alpha = float(amber_alpha)
            self._amber_ref = float(amber_reference_kev)
            self._uniform = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)
            self._eye = np.eye(self._n, dtype=np.complex128)

        # ── Método 1.1 ────────────────────────────────────────────────────────
        def _validate_and_hermitize(
            self, mat: Any
        ) -> Tuple[NDArray[np.complex128], float, float, bool]:
            """
            1.1 Validación total de dimensión/finitud y proyección autoadjunta ½(M+M†).
            Devuelve (hermítico, defecto_de_traza, defecto_de_hermiticidad, fallback).
            [P8/P15] Garantiza clausura algebraica sobre el espacio de operadores hermíticos
            en ℂ^{n×n} antes de invocar rutinas espectrales LAPACK.
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
            h_defect = _relative(_fro(m - herm), _fro(m))
            t_defect = abs(float(np.real(np.trace(m))) - 1.0)
            return herm, float(t_defect), float(h_defect), False

        # ── Método 1.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _project_spectrum_capped_simplex(
            values: NDArray[np.float64], floor: float
        ) -> Tuple[NDArray[np.float64], float, bool]:
            """
            1.2 Proyección euclídea EXACTA sobre Δ_δ^{n−1} = {p ≥ δ, Σp = 1} (water-filling).
            Con s = p−δ ≥ 0 y masa libre S = 1−nδ: p = δ + max(v−δ−θ, 0), donde θ resuelve
            Σ max(v_i−δ−θ, 0) = S en tiempo O(n log n).
            Factibilidad ⟺ δ ≤ 1/n; se impone δ ≤ 1/(2n). La corrección final de redondeo
            de punto flotante es ADITIVA sobre argmax(p), preservando las condiciones KKT.
            """
            v = np.asarray(values, dtype=np.float64).ravel()
            n = int(v.size)
            delta = float(min(max(floor, 0.0), 0.5 / max(1, n)))
            feasible = bool(delta * n <= 1.0)
            mass = 1.0 - n * delta
            y = v - delta
            u = np.sort(y)[::-1]
            css = np.cumsum(u)
            idx = np.arange(1, n + 1, dtype=np.float64)
            cond = u - (css - mass) / idx > 0.0
            k = int(np.flatnonzero(cond)[-1]) + 1 if bool(cond.any()) else 1
            theta = float((css[k - 1] - mass) / k)
            p = np.maximum(y - theta, 0.0) + delta
            p[int(np.argmax(p))] += 1.0 - float(np.sum(p))
            return p, delta, feasible

        # ── Método 1.3 ────────────────────────────────────────────────────────
        def _rsi3_optimal_spectral_floor(
            self, eigvals: NDArray[np.float64]
        ) -> Tuple[float, float]:
            """
            1.3 [RSI³ Meta-Optimizador Espectral] Determina el piso óptimo δ*(λ) minimizando
            el funcional de riesgo compuesto de Mirsky-Daleckii-Krein sobre una malla
            logarítmica candidata dentro de [δ_base, min(1/(2n), max(δ_base, 100·δ_base))]:
                R_meta(δ) = ‖p_δ(λ) − λ‖₂² + (n · ε_mach)² / (4 · max(δ, ε_TINY)).
            El primer término mide la distorsión geométrica de Frobenius (teorema de Mirsky)
            y el segundo penaliza la amplificación de error de redondeo en la raíz cuadrada
            operatorial (cota de Daleckii-Krein L_sqrt = 1/(2√δ)). Devuelve (δ*, R_meta(δ*)).
            """
            lam_max = float(max(np.max(eigvals), 1.0 / self._n))
            delta_min = min(
                max(self._floor, lam_max / _MAX_MODULAR_CONDITION),
                0.5 / self._n,
            )
            delta_hi = min(0.5 / self._n, max(delta_min, 16.0 * delta_min))
            if delta_hi <= delta_min * (1.0 + 1e-12):
                p_proj, d_used, _ = self._project_spectrum_capped_simplex(eigvals, delta_min)
                risk = float(
                    np.sum((p_proj - eigvals) ** 2)
                    + (self._n * _MACHINE_EPS) ** 2 / max(4.0 * d_used, _TINY)
                )
                return d_used, risk

            candidates = np.geomspace(delta_min, delta_hi, num=7, dtype=np.float64)
            best_delta = delta_min
            best_risk = float("inf")
            dk_weight = (self._n * _MACHINE_EPS) ** 2
            for cand in candidates:
                p_proj, d_used, _ = self._project_spectrum_capped_simplex(eigvals, float(cand))
                fro_cost_sq = float(np.sum((p_proj - eigvals) ** 2))
                dk_penalty = dk_weight / max(4.0 * d_used, _TINY)
                total_risk = fro_cost_sq + dk_penalty
                if total_risk < best_risk:
                    best_risk = total_risk
                    best_delta = d_used
            return float(best_delta), float(best_risk)

        # ── Método 1.4 ────────────────────────────────────────────────────────
        def _canonicalize_with_certificate(
            self, rho: Any,floor_override: Optional[float] = None
        ) -> DensityCanonicalization:
            """
            1.4 Proyección exacta de Frobenius sobre D_δ(H) = {ρ=ρ†⪰δI, Tr ρ=1}.
            Por el teorema de Mirsky / Hoffman-Wielandt, la proyección de Frobenius sobre
            un convexo unitariamente invariante conserva la base propia de ρ y proyecta el
            espectro sobre el símplex capado Δ_δ^{n−1} [P8].
            """
            herm, t_def, h_def, fallback = self._validate_and_hermitize(rho)
            if fallback:
                probs = self._uniform.copy()
                purity = 1.0 / self._n
                return DensityCanonicalization(
                    rho=herm,
                    eigenvalues=probs,
                    eigenvectors=self._eye.copy(),
                    reference_probs=probs,
                    entropy=math.log(float(self._n)),
                    purity=purity,
                    condition_number=1.0,
                    projection_cost=0.0,
                    floor_used=1.0 / self._n,
                    kkt_ok=True,
                    hermiticity_defect=0.0,
                    trace_defect=0.0,
                    used_fallback=True,
                    rsi3_meta_risk=0.0,
                    linear_entropy=float(1.0 - purity),
                )
            try:
                eigvals, eigvecs = la.eigh(herm)
                eigvals = np.real(eigvals).astype(np.float64)
                if floor_override is not None:
                    delta_target = min(max(float(floor_override), _MACHINE_EPS), 0.5 / self._n)
                    meta_risk = 0.0
                else:
                    delta_target, meta_risk = self._rsi3_optimal_spectral_floor(eigvals)
                probs, delta, feasible = self._project_spectrum_capped_simplex(
                    eigvals, delta_target
                )
                raw = np.maximum(eigvals, 0.0)
                raw_sum = float(np.sum(raw))
                ref = raw / raw_sum if raw_sum > _WILKINSON_EPS else self._uniform.copy()
                out = (eigvecs * probs) @ eigvecs.conj().T
                out = 0.5 * (out + out.conj().T)
                entropy = float(-np.sum(probs * np.log(np.maximum(probs, _WILKINSON_EPS))))
                purity = float(np.sum(probs ** 2))
                kkt = bool(
                    feasible
                    and float(np.min(probs)) >= delta - 1e-12
                    and abs(float(np.sum(probs)) - 1.0) <= 1e-12
                )
                return DensityCanonicalization(
                    rho=np.asarray(out, dtype=np.complex128),
                    eigenvalues=probs,
                    eigenvectors=eigvecs,
                    reference_probs=ref,
                    entropy=max(0.0, entropy),
                    purity=purity,
                    condition_number=float(np.max(probs) / max(np.min(probs), _TINY)),
                    projection_cost=float(np.linalg.norm(probs - eigvals, ord=2)),
                    floor_used=float(delta),
                    kkt_ok=kkt,
                    hermiticity_defect=h_def,
                    trace_defect=t_def,
                    used_fallback=False,
                    rsi3_meta_risk=float(meta_risk),
                    linear_entropy=float(max(0.0, 1.0 - purity)),
                )
            except (la.LinAlgError, ValueError):
                probs = self._uniform.copy()
                purity = 1.0 / self._n
                return DensityCanonicalization(
                    rho=self._eye / float(self._n),
                    eigenvalues=probs,
                    eigenvectors=self._eye.copy(),
                    reference_probs=probs,
                    entropy=math.log(float(self._n)),
                    purity=purity,
                    condition_number=1.0,
                    projection_cost=0.0,
                    floor_used=1.0 / self._n,
                    kkt_ok=True,
                    hermiticity_defect=h_def,
                    trace_defect=t_def,
                    used_fallback=True,
                    rsi3_meta_risk=0.0,
                    linear_entropy=float(1.0 - purity),
                )

        def canonicalize_density_matrix(
            self, rho: NDArray[np.complex128]
        ) -> NDArray[np.complex128]:
            """1.4b Fachada pública compatible con v16/v17 (canonización C* sobre D(H))."""
            return self._canonicalize_with_certificate(rho).rho

        # ── Método 1.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _sqrt_and_invsqrt(
            canon: DensityCanonicalization,
        ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
            """
            1.5 Raíz operatorial principal ρ^{1/2} y raíz inversa ρ^{−1/2} por vía espectral.
            Reutiliza la descomposición unitaria de 1.4 con coste marginal O(n²).
            """
            sq = np.sqrt(np.maximum(canon.eigenvalues, _STABILITY_FLOOR))
            v = canon.eigenvectors
            root = (v * sq) @ v.conj().T
            inv = (v * (1.0 / sq)) @ v.conj().T
            return 0.5 * (root + root.conj().T), 0.5 * (inv + inv.conj().T)

        # ── Método 1.6 ────────────────────────────────────────────────────────
        def _uhlmann_holonomy_and_quaternion_sector(
            self,
            sqrt_r: NDArray[np.complex128],
            sqrt_s: NDArray[np.complex128],
        ) -> Tuple[UhlmannHolonomyCertificate, NDArray[np.float64]]:
            """
            1.6 [Fibrado Principal de Uhlmann y Proyección Cuaterniónica S³ ⊂ ℍ]
            Calcula la SVD del operador de transición de amplitudes M = √ρ√σ = U Σ W†.
            El factor polar unitario V_Uhl = U W† ∈ U(n) realiza el transporte paralelo de
            Uhlmann entre las purificaciones canónicas de ρ y σ:
                √F(ρ,σ) = Tr(|√ρ√σ|) = Tr(√ρ√σ · V_Uhl†) = Σ_i σ_i.
            Extrae:
              • Fase geométrica de Pancharatnam-Uhlmann Φ_Uhl = arg Tr(√ρ√σ),
              • Fase del determinante det(V_Uhl) = e^{i φ_det},
              • Proyección del bloque principal 2×2 de V_Uhl sobre la 3-esfera de cuaterniones
                unitarios S³ ⊂ ℍ vía el isomorfismo de Cayley-Dickson/Pauli.
            """
            prod = sqrt_r @ sqrt_s
            try:
                u_mat, sv, wh_mat = la.svd(prod, full_matrices=True)
                v_uhl = u_mat @ wh_mat
            except (la.LinAlgError, ValueError):
                sv = np.real(la.svdvals(prod))
                v_uhl = self._eye.copy()

            unit_def = _fro(v_uhl.conj().T @ v_uhl - self._eye)
            tr_prod = complex(np.trace(prod))
            pancharatnam = (
                float(math.atan2(tr_prod.imag, tr_prod.real))
                if abs(tr_prod) > _TINY
                else 0.0
            )
            det_v = complex(np.linalg.det(v_uhl))
            det_phase = (
                float(math.atan2(det_v.imag, det_v.real))
                if abs(det_v) > _TINY
                else 0.0
            )
            quat_s3, quat_def = _pauli_quaternion_projection(v_uhl)
            cert = UhlmannHolonomyCertificate(
                holonomy_unitary=_freeze_array(v_uhl),
                pancharatnam_phase_rad=pancharatnam,
                determinant_phase_rad=det_phase,
                unitarity_defect=float(unit_def),
                quaternion_s3=quat_s3,
                quaternion_norm_defect=float(quat_def),
            )
            return cert, np.real(sv).astype(np.float64)

        # ── Método 1.7 ────────────────────────────────────────────────────────
        def compute_uhlmann_fidelity(
            self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
        ) -> Tuple[float, float]:
            """1.7b Fachada pública compatible con v16/v17: devuelve (F, d_B)."""
            cert = self._uhlmann_bures_geometry(rho, sigma)
            return cert[0], cert[1]

        def _uhlmann_bures_geometry(
            self, rho: Any, sigma: Any
        ) -> Tuple[
            float,
            float,
            float,
            float,
            float,
            DensityCanonicalization,
            DensityCanonicalization,
            UhlmannHolonomyCertificate,
            float,
        ]:
            """
            1.7 Geometría de Bures-Uhlmann con certificado de holonomía y extrapolación
            homotópica de Richardson en δ → δ/2:
              • Ruta primaria (norma traza SVD): √F = ‖√ρ√σ‖₁ = Σ σ_i(√ρ√σ),
              • Ruta secundaria (espectral de √ρσ√ρ): controla el deterioro por κ²,
              • Identidad cuerda-arco en la esfera de purificaciones: d_B = 2·sin(θ_B/2),
              • Homotopía RSI³: compara F_δ con F_{δ/2} para acotar el sesgo de regularización.
            """
            cr = self._canonicalize_with_certificate(rho)
            cs = self._canonicalize_with_certificate(sigma)
            sqrt_r, _ = self._sqrt_and_invsqrt(cr)
            sqrt_s, _ = self._sqrt_and_invsqrt(cs)
            holonomy, sv = self._uhlmann_holonomy_and_quaternion_sector(sqrt_r, sqrt_s)
            root_f_svd = float(np.clip(np.sum(sv), 0.0, 1.0))

            m = sqrt_r @ cs.rho @ sqrt_r
            m = 0.5 * (m + m.conj().T)
            eig_m = np.real(la.eigvalsh(m))
            root_f_eig = float(np.clip(np.sum(np.sqrt(np.maximum(eig_m, 0.0))), 0.0, 1.0))

            root_f = root_f_svd
            fidelity = float(np.clip(root_f * root_f, 0.0, 1.0))
            bures = math.sqrt(max(0.0, 2.0 - 2.0 * root_f))
            angle = math.acos(float(np.clip(root_f, -1.0, 1.0)))
            chord_err = abs(bures - 2.0 * math.sin(angle / 2.0))

            # Extrapolación homotópica RSI³ (δ → δ/2) para certificar sensibilidad al piso:
            half_floor = max(_MACHINE_EPS, 0.5 * min(cr.floor_used, cs.floor_used))
            cr_half = self._canonicalize_with_certificate(rho, floor_override=half_floor)
            cs_half = self._canonicalize_with_certificate(sigma, floor_override=half_floor)
            sr_h, _ = self._sqrt_and_invsqrt(cr_half)
            ss_h, _ = self._sqrt_and_invsqrt(cs_half)
            sv_h = np.real(la.svdvals(sr_h @ ss_h))
            fid_half = float(np.clip(np.sum(sv_h), 0.0, 1.0)) ** 2
            homotopy_drift = float(abs(fidelity - fid_half))

            return (
                fidelity,
                bures,
                angle,
                chord_err,
                abs(root_f_svd - root_f_eig),
                cr,
                cs,
                holonomy,
                homotopy_drift,
            )

        # ── Método 1.8 ────────────────────────────────────────────────────────
        def compute_umegaki_relative_entropy(
            self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
        ) -> Tuple[float, float]:
            """1.8b Fachada pública compatible con v16/v17: (D(ρ‖σ), residuo_de_Pinsker)."""
            out = self._umegaki_pinsker_fuchs_sandwich(rho, sigma)
            return out[0], out[3]

        def _umegaki_pinsker_fuchs_sandwich(
            self,
            rho: Any,
            sigma: Any,
            fidelity_hint: Optional[float] = None,
            cr_hint: Optional[DensityCanonicalization] = None,
            cs_hint: Optional[DensityCanonicalization] = None,
        ) -> Tuple[float, float, float, float, bool, float, float, bool, float]:
            """
            1.8 Entropía relativa de Umegaki con logs espectrales exactos [P9], condicionamiento
            del operador modular de Tomita-Takesaki Δ_{σ,ρ} = L_σ R_ρ^{−1}, desigualdad
            cuántica de Pinsker D(ρ‖σ) ≥ ½‖ρ−σ‖₁² y sándwich de Fuchs-van de Graaf:
                1 − √F(ρ,σ) ≤ D_tr(ρ,σ) ≤ √(1 − F(ρ,σ)),   donde D_tr = ½‖ρ−σ‖₁.
            """
            cr = cr_hint if cr_hint is not None else self._canonicalize_with_certificate(rho)
            cs = cs_hint if cs_hint is not None else self._canonicalize_with_certificate(sigma)
            lam = np.maximum(cr.eigenvalues, _WILKINSON_EPS)
            s = np.maximum(cs.eigenvalues, _WILKINSON_EPS)
            tr_rho_ln_rho = float(np.sum(lam * np.log(lam)))
            weights = np.maximum(
                np.real(np.diag(cs.eigenvectors.conj().T @ cr.rho @ cs.eigenvectors)),
                0.0,
            )
            tr_rho_ln_sigma = float(np.sum(weights * np.log(s)))
            umegaki = tr_rho_ln_rho - tr_rho_ln_sigma
            finite = math.isfinite(umegaki)
            umegaki = float(umegaki) if finite else float("nan")

            diff = cr.rho - cs.rho
            diff = 0.5 * (diff + diff.conj().T)
            trace_norm = float(np.sum(np.abs(np.real(la.eigvalsh(diff)))))
            trace_dist = 0.5 * trace_norm
            pinsker_bound = 0.5 * trace_norm ** 2
            residual = (umegaki - pinsker_bound) if finite else float("nan")
            pinsker_ok = bool(finite and residual >= -1e-9)

            if fidelity_hint is None:
                fid, *_ = self._uhlmann_bures_geometry(cr.rho, cs.rho)
            else:
                fid = float(np.clip(fidelity_hint, 0.0, 1.0))
            root_f = math.sqrt(max(0.0, fid))
            fuchs_low = 1.0 - root_f
            fuchs_high = math.sqrt(max(0.0, 1.0 - fid))
            low_margin = float(trace_dist - fuchs_low)
            high_margin = float(fuchs_high - trace_dist)
            fuchs_ok = bool(low_margin >= -1e-9 and high_margin >= -1e-9)

            # Condicionamiento del operador modular de Tomita-Takesaki Δ_{σ,ρ}:
            mod_max = float(np.max(s) / max(float(np.min(lam)), _TINY))
            mod_min = float(np.min(s) / max(float(np.max(lam)), _TINY))
            tomita_cond = float(min(mod_max / max(mod_min, _TINY), _CONDITION_CAP))

            return (
                umegaki,
                trace_dist,
                pinsker_bound,
                float(residual),
                pinsker_ok,
                low_margin,
                high_margin,
                fuchs_ok,
                tomita_cond,
            )

        # ── Método 1.9 ────────────────────────────────────────────────────────
        def _wigner_yanase_skew_information(
            self,
            cr: DensityCanonicalization,
            cs: DensityCanonicalization,
        ) -> float:
            """
            1.9 Información oblicua cuántica de Wigner-Yanase-Dyson respecto del hamiltoniano
            modular K_σ = −ln σ:
                I_{1/2}(ρ, K_σ) = −½ Tr([√ρ, K_σ]²) = Tr(ρ K_σ²) − Tr(√ρ K_σ √ρ K_σ) ≥ 0.
            Mide la no-conmutatividad cuántica pura entre el estado ρ y el estado de referencia σ
            (se anula si y sólo si [ρ, σ] = 0).
            """
            sqrt_r, _ = self._sqrt_and_invsqrt(cr)
            s_eigs = np.maximum(cs.eigenvalues, _WILKINSON_EPS)
            v_s = cs.eigenvectors
            k_mod = (v_s * (-np.log(s_eigs))) @ v_s.conj().T
            k_mod = 0.5 * (k_mod + k_mod.conj().T)
            comm = sqrt_r @ k_mod - k_mod @ sqrt_r
            # Como √ρ y K_σ son hermíticos, [√ρ, K_σ] es antihermítico ⟹ −½ Tr(C²) = ½ ‖C‖_F² ≥ 0
            skew_info = 0.5 * (_fro(comm) ** 2)
            return float(max(0.0, skew_info))

        # ── Método 1.10 ───────────────────────────────────────────────────────
        @staticmethod
        def _dirac_annihilation_certificate(
            positron_beta: float,
        ) -> DiracAnnihilationCertificate:
            """
            1.10 Sección eficaz EXACTA de Dirac para e⁻+e⁺→2γ libre de cancelación catastrófica
            mediante la identidad cinemática √(γ²−1) = βγ y arcosh(γ) = ln(γ + βγ), junto con:
              • Factor atractivo de Coulomb de Gamow-Sommerfeld S(η) = 2πη/(1−e^{−2πη}), η=α_fs/β,
              • Canal de 3 fotones de Ore-Powell para orto-positronio (³S₁ → 3γ):
                    σ_{3γ} = [4(π²−9)/(9π)] · α_fs · σ_{2γ,NR},
              • Cinemática relativista de Mandelstam: E_rest=2m_ec², E_lab=(γ+1)m_ec²,
                √s = m_ec²√(2(1+γ)). [P10]
            """
            beta = float(np.clip(_safe_float(positron_beta, 0.01), 1.0e-10, 1.0 - 1.0e-12))
            one_minus_b2 = max(_TINY, (1.0 - beta) * (1.0 + beta))
            gamma = 1.0 / math.sqrt(one_minus_b2)
            bg = beta * gamma  # √(γ² − 1) exacto sin cancelación sustractiva

            if beta < 1.0e-5:
                # Expansión analítica de Taylor en potencias de β² cerca del umbral NR:
                reduced = (1.0 / beta) * (1.0 - (beta * beta) * (1.0 / 3.0))
            elif gamma > 1.0e6:
                reduced = (math.log(2.0 * gamma) - 1.0) / gamma
            else:
                acosh_g = math.log(gamma + bg)
                reduced = (1.0 / (gamma + 1.0)) * (
                    ((gamma * gamma + 4.0 * gamma + 1.0) / (bg * bg)) * acosh_g
                    - (gamma + 3.0) / bg
                )
            reduced = max(0.0, float(reduced))
            sigma_m2 = _PI_R_E_SQUARED * reduced

            eta = _FINE_STRUCTURE_ALPHA / beta
            x = 2.0 * math.pi * eta
            gamow = 1.0 + 0.5 * x if x < 1.0e-8 else float(x / (-math.expm1(-x)))
            sigma_nr = _PI_R_E_SQUARED * gamow / beta
            sigma_3gamma = _ORE_POWELL_3GAMMA_RATIO * sigma_nr
            ratio = _relative(sigma_nr, sigma_m2) if sigma_m2 > 0.0 else 0.0

            v_ms = beta * _SPEED_OF_LIGHT_MS
            e_rest = _ANNIHILATION_ENERGY_KEV
            e_lab = (gamma + 1.0) * _ELECTRON_REST_MASS_KEV
            e_cm = _ELECTRON_REST_MASS_KEV * math.sqrt(2.0 * (1.0 + gamma))
            return DiracAnnihilationCertificate(
                beta=beta,
                gamma_lorentz=float(gamma),
                cross_section_m2=float(sigma_m2),
                cross_section_reduced=float(reduced),
                gamow_factor=float(gamow),
                nr_gamow_cross_section_m2=float(sigma_nr),
                nr_to_exact_ratio=float(ratio),
                rate_coefficient_m3s=float(sigma_m2 * v_ms),
                energy_rest_kev=float(e_rest),
                energy_lab_kev=float(e_lab),
                energy_cm_kev=float(e_cm),
                nr_regime_valid=bool(gamma < 1.01 and 0.9 <= ratio <= 1.1),
                ore_powell_3gamma_cross_section_m2=float(sigma_3gamma),
                beta_gamma_momentum=float(bg),
            )

        # ── Método 1.11 ───────────────────────────────────────────────────────
        @staticmethod
        def _gksl_dephasing_step(
            rho: NDArray[np.complex128], rate: float, dt: float
        ) -> Tuple[NDArray[np.complex128], float, float, bool]:
            """
            1.11 Semigrupo cuántico GKSL de desfase puro (pinching de Schur-Hadamard):
                Φ_t(ρ) = (1−p)ρ + p·Δ(ρ),   p = 1 − e^{−γ_d dt} ∈ [0, 1].
            Es completamente positivo y preservador de traza (CPTP), unital (Φ_t(I)=I) y
            satisface la desigualdad de procesamiento de datos S(Φ_t(ρ)) ≥ S(ρ) por
            concavidad de la entropía de von Neumann y mayorización de Schur (diagonal ≺ espectro).
            """
            g = max(0.0, _safe_float(rate, 0.0))
            h = max(0.0, _safe_float(dt, 0.0))
            if g <= 0.0 or h <= 0.0:
                return rho, 0.0, 0.0, True
            p = float(-math.expm1(-g * h))
            diag_part = np.diag(np.diag(rho))
            out = (1.0 - p) * rho + p * diag_part
            out = 0.5 * (out + out.conj().T)
            tr_def = abs(float(np.real(np.trace(out))) - float(np.real(np.trace(rho))))

            def _entropy(mat: NDArray[np.complex128]) -> float:
                ev = np.maximum(np.real(la.eigvalsh(mat)), _WILKINSON_EPS)
                ev = ev / max(float(np.sum(ev)), _TINY)
                return float(-np.sum(ev * np.log(ev)))

            delta_s = _entropy(out) - _entropy(rho)
            psd_ok = bool(
                0.0 <= p <= 1.0 and np.min(np.real(la.eigvalsh(out))) >= -1e-10
            )
            return out, float(delta_s), float(tr_def), bool(psd_ok and tr_def <= 1e-10)

        def _amber_grace_duration(
            self, bures_distance: float, dirac: DiracAnnihilationCertificate
        ) -> float:
            """
            1.11b Rampa de Gracia Ámbar ponderada por la entalpía relativista en el centro de
            masas √s [P11]: τ_amber = τ₀ · exp(−α · d_B(ρ,σ)) · (√s / E_ref).
            """
            d_b = max(0.0, _safe_float(bures_distance, 0.0))
            ratio = _relative(dirac.energy_cm_kev, max(self._amber_ref, _WILKINSON_EPS))
            exponent = min(700.0, self._amber_alpha * d_b)
            return float(self._amber_base * math.exp(-exponent) * ratio)

        # ── Método 1.12 · TERMINAL Π₁ → APERTURA DIRECTA DE Π₂ ───────────────
        def audit_dirac_uhlmann_substrate(
            self,
            rho_init: NDArray[np.complex128],
            sigma_target: NDArray[np.complex128],
            positron_beta: float = 0.01,
        ) -> Pi1SpectrumCertificate:
            """
            1.12 [DEFINICIÓN TERMINAL DE LA FASE Π₁ ──► CONTINUACIÓN DIRECTA EN 2.1]
            Orquesta la cadena 1.1 → 1.11 sobre el fibrado de estados cuánticos D_δ(H) y
            construye el morfismo semilla `Pi1ToPi2Seed` (tensión de Bures, divergencia de
            Umegaki, fase geométrica de Pancharatnam-Uhlmann, información oblicua de
            Wigner-Yanase, condicionamiento modular de Tomita-Takesaki y cinemática de Dirac),
            cuya estructura de tipos es el argumento formal de apertura de
            `PhasePi2_PoincareCelestialRegularizer._ingest_pi1_continuation` (Método 2.1).
            """
            (
                fidelity,
                bures,
                angle,
                chord_err,
                route_disc,
                cr,
                cs,
                holonomy,
                homotopy_drift,
            ) = self._uhlmann_bures_geometry(rho_init, sigma_target)

            (
                umegaki,
                trace_dist,
                pinsker_b,
                pinsker_res,
                pinsker_ok,
                fuchs_low_m,
                fuchs_high_m,
                fuchs_ok,
                tomita_cond,
            ) = self._umegaki_pinsker_fuchs_sandwich(
                cr.rho, cs.rho, fidelity_hint=fidelity, cr_hint=cr, cs_hint=cs
            )

            wy_skew = self._wigner_yanase_skew_information(cr, cs)
            dirac = self._dirac_annihilation_certificate(positron_beta)
            amber = self._amber_grace_duration(bures, dirac)

            seed_to_pi2 = Pi1ToPi2Seed(
                bures_tension=float(bures),
                umegaki_divergence=float(umegaki if math.isfinite(umegaki) else 0.0),
                annihilation_energy_kev=float(dirac.energy_cm_kev),
                annihilation_rate_coefficient=float(dirac.rate_coefficient_m3s),
                rho_entropy=float(cr.entropy),
                rho_condition_number=float(cr.condition_number),
                has_context=True,
                pancharatnam_phase_rad=float(holonomy.pancharatnam_phase_rad),
                wigner_yanase_skew_info=float(wy_skew),
                tomita_modular_condition=float(tomita_cond),
            )

            return Pi1SpectrumCertificate(
                rho=cr.rho,
                sigma=cs.rho,
                rho_canon=cr,
                sigma_canon=cs,
                uhlmann_fidelity=fidelity,
                bures_distance=bures,
                bures_angle=angle,
                chord_arc_error=chord_err,
                fidelity_route_discrepancy=route_disc,
                umegaki_relative_entropy=umegaki,
                trace_distance=trace_dist,
                pinsker_bound=pinsker_b,
                pinsker_residual=pinsker_res,
                pinsker_ok=pinsker_ok,
                dirac=dirac,
                amber_grace_duration_sec=amber,
                continuation=seed_to_pi2,
                holonomy=holonomy,
                fuchs_lower_margin=fuchs_low_m,
                fuchs_upper_margin=fuchs_high_m,
                fuchs_sandwich_ok=fuchs_ok,
                tomita_modular_condition=tomita_cond,
                wigner_yanase_skew_info=wy_skew,
                fidelity_homotopy_drift=homotopy_drift,
            )


    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₂ · CARTA DE POINCARÉ-CARTAN, KEPLER, MELNIKOV, YOSHIDA-KREIN Y KAM
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi2_PoincareCelestialRegularizer:
        """
        Π₂. Cadena formal anidada (continuación directa de 1.12):
        2.1 INGESTA de Pi1ToPi2Seed → 2.2 margen diofántico KAM →
        2.3 carta canónica regularizada e invariante de Poincaré-Cartan 4D →
        2.4 Halley-Kepler con salvaguarda de bracket → 2.5 auditoría del toro de Kepler e
        invariante integral de Kapteyn-Fourier-Bessel → 2.6 turnstile de Melnikov y ángulo
        homoclínico → 2.7 integrador simpléctico de Yoshida de 4.º orden →
        2.8 monodromía epicíclica de Floquet-Krein y número de rotación de Poincaré →
        2.9 resonancia de Lindstedt-Poincaré vs. AGM elíptico exacto [P16] →
        2.10 TERMINAL (emite Pi2ToPi3Seed hacia 3.1).
        """

        def __init__(self, precession_splitting: float, enable_doubling: bool) -> None:
            self._splitting = float(precession_splitting)
            self._doubling = bool(enable_doubling)
            self._omega4 = _symplectic_omega(2)

        # ── Método 2.1 · APERTURA Π₂ (continuación directa de 1.12) ──────────
        @staticmethod
        def _ingest_pi1_continuation(
            pi1: Optional[Pi1SpectrumCertificate] = None,
        ) -> Pi1ToPi2Seed:
            """
            2.1 [CONTINUACIÓN FORMAL DIRECTA DE 1.12 audit_dirac_uhlmann_substrate]
            Absorbe la semilla geométrico-exergética-cuántica `Pi1ToPi2Seed` emitida por Π₁.
            Si Π₂ se invoca de forma aislada, devuelve el elemento neutro del acoplamiento.
            """
            if pi1 is not None and pi1.continuation is not None:
                return pi1.continuation
            return Pi1ToPi2Seed(
                bures_tension=0.0,
                umegaki_divergence=0.0,
                annihilation_energy_kev=_ANNIHILATION_ENERGY_KEV,
                annihilation_rate_coefficient=0.0,
                rho_entropy=0.0,
                rho_condition_number=1.0,
                has_context=False,
                pancharatnam_phase_rad=0.0,
                wigner_yanase_skew_info=0.0,
                tomita_modular_condition=1.0,
            )

        # ── Método 2.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _kam_diophantine_margin(
            omega_1: float,
            omega_2: float,
            max_order: int = 8,
            tau: float = 2.0,
        ) -> float:
            """
            2.2 Margen diofántico de Kolmogorov-Arnold-Moser (KAM) sobre el retículo de
            resonancias enteras (k₁, k₂) ∈ ℤ² \\ {(0,0)} con 1 ≤ |k₁| + |k₂| ≤ K_max:
                γ_KAM(ω₁, ω₂) = min_{0 < |k|₁ ≤ K_max} |k₁ ω₁ + k₂ ω₂| · |k|₁^τ / max(|ω₁|+|ω₂|, ε).
            Cuantifica la distancia aritmética a las resonancias seculares que destruyen los
            toros invariantes KAM.
            """
            w1 = _safe_float(omega_1, 1.0)
            w2 = _safe_float(omega_2, 1.0)
            scale = max(abs(w1) + abs(w2), _TINY)
            k_lim = max(1, int(max_order))
            best = float("inf")
            for k1 in range(-k_lim, k_lim + 1):
                for k2 in range(-k_lim, k_lim + 1):
                    l1 = abs(k1) + abs(k2)
                    if l1 == 0 or l1 > k_lim:
                        continue
                    # Evita duplicar (k1, k2) y (-k1, -k2)
                    if k1 < 0 or (k1 == 0 and k2 < 0):
                        continue
                    val = (abs(k1 * w1 + k2 * w2) / scale) * (float(l1) ** tau)
                    if val < best:
                        best = val
            return float(best if math.isfinite(best) else 0.0)

        # ── Método 2.3 ────────────────────────────────────────────────────────
        @staticmethod
        def evaluate_poincare_canonical_chart(
            semi_major_axis: float,
            eccentricity: float,
            inclination_rad: float,
            mean_anomaly: float,
            arg_pericenter_rad: float = 0.0,
            lon_node_rad: float = 0.0,
            mu: float = 1.0,
        ) -> Tuple[NDArray[np.float64], float]:
            """2.3b Fachada pública compatible con v16/v17: (carta_canónica, residuo_algebraico)."""
            cert = (
                PositronRedemptionGovernorEngine.PhasePi2_PoincareCelestialRegularizer._chart_with_certificate(
                    semi_major_axis,
                    eccentricity,
                    inclination_rad,
                    mean_anomaly,
                    arg_pericenter_rad,
                    lon_node_rad,
                    mu,
                )
            )
            return cert.chart, cert.algebraic_identity_residual

        @staticmethod
        def _chart_with_certificate(
            semi_major_axis: float,
            eccentricity: float,
            inclination_rad: float,
            mean_anomaly: float,
            arg_pericenter_rad: float = 0.0,
            lon_node_rad: float = 0.0,
            mu: float = 1.0,
            kam_margin_hint: float = 0.0,
        ) -> PoincareChartCertificate:
            """
            2.3 Carta canónica regularizada de Poincaré libre de singularidades en e→0, i→0 [P3]:
                Λ = √(μa),  G = Λ√(1−e²),  H = G cos i,  Γ = Λ − G,  Z = G − H,
                (ξ₁, η₁) = √(2Γ)(cos ϖ, −sin ϖ),   (ξ₂, η₂) = √(2Z)(cos Ω, −sin Ω).
            CERTIFICADOS:
              (a) Round-trip exacto sobre los dos sectores acción-ángulo.
              (b) Canonicidad planar: |det ∂(Γ,ϖ)/∂(ξ₁,η₁)| = 1.
              (c) Invariante integral de Poincaré-Cartan 4D: como dξ_k ∧ dη_k = dΓ_k ∧ d(−θ_k),
                  la aplicación Ψ: (ξ₁,η₁,ξ₂,η₂) ↦ (Γ, −ϖ, Z, −Ω) es simpléctica con orientación
                  positiva y su jacobiano 4×4 satisface ‖J₄ᵀ Ω₄ J₄ − Ω₄‖_F = 0.
            """
            a = max(1e-9, _safe_float(semi_major_axis, 1.0))
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
            inc = float(np.clip(_safe_float(inclination_rad, 0.0), 0.0, math.pi - 1e-9))
            m_anom = _safe_float(mean_anomaly, 0.0)
            varpi = _safe_float(arg_pericenter_rad, 0.0) + _safe_float(lon_node_rad, 0.0)
            omega_node = _safe_float(lon_node_rad, 0.0)
            mu_eff = max(_TINY, _safe_float(mu, 1.0))

            lam = math.sqrt(mu_eff * a)
            g_del = lam * math.sqrt(max(0.0, 1.0 - e * e))
            h_del = g_del * math.cos(inc)
            gamma_act = lam - g_del
            z_act = g_del - h_del

            def forward(action: float, angle: float) -> Tuple[float, float]:
                r = math.sqrt(max(0.0, 2.0 * action))
                return r * math.cos(angle), -r * math.sin(angle)

            def inverse(xi: float, eta: float) -> Tuple[float, float]:
                return 0.5 * (xi * xi + eta * eta), math.atan2(-eta, xi)

            xi1, eta1 = forward(gamma_act, varpi)
            xi2, eta2 = forward(z_act, omega_node)
            chart = np.array([lam, xi1, eta1, xi2, eta2], dtype=np.float64)

            h_recon = lam - 0.5 * (xi1 ** 2 + eta1 ** 2) - 0.5 * (xi2 ** 2 + eta2 ** 2)
            algebraic = abs(h_recon - h_del)

            g_back, varpi_back = inverse(xi1, eta1)
            z_back, omega_back = inverse(xi2, eta2)
            wrap = lambda d: math.atan2(math.sin(d), math.cos(d))  # noqa: E731
            err_1 = math.hypot(
                g_back - gamma_act,
                abs(wrap(varpi_back - varpi)) if gamma_act > 1e-12 else 0.0,
            )
            err_2 = math.hypot(
                z_back - z_act,
                abs(wrap(omega_back - omega_node)) if z_act > 1e-12 else 0.0,
            )
            roundtrip = math.hypot(err_1, err_2)

            # Jacobiano planar 2×2 sobre (ξ₁, η₁) -> (Γ, ϖ):
            h_fd = 1e-6 * max(1.0, math.hypot(xi1, eta1))
            gp_x, vp_x = inverse(xi1 + h_fd, eta1)
            gm_x, vm_x = inverse(xi1 - h_fd, eta1)
            gp_y, vp_y = inverse(xi1, eta1 + h_fd)
            gm_y, vm_y = inverse(xi1, eta1 - h_fd)
            j11 = (gp_x - gm_x) / (2.0 * h_fd)
            j12 = (gp_y - gm_y) / (2.0 * h_fd)
            j21 = wrap(vp_x - vm_x) / (2.0 * h_fd)
            j22 = wrap(vp_y - vm_y) / (2.0 * h_fd)
            det2 = j11 * j22 - j12 * j21
            canonicity = abs(abs(det2) - 1.0) if gamma_act > 1e-10 else 0.0

            # Invariante integral de Poincaré-Cartan 4D: Ψ(ξ₁,η₁,ξ₂,η₂) = (Γ, −ϖ, Z, −Ω).
            # Si Γ o Z están en el origen, se audita la carta en un punto desplazado regular
            # para certificar la estructura simpléctica intrínseca del difeomorfismo.
            x1_reg, y1_reg = (xi1, eta1) if gamma_act > 1e-6 else forward(1e-3, 0.35)
            x2_reg, y2_reg = (xi2, eta2) if z_act > 1e-6 else forward(1e-3, -0.55)
            base_pt = np.array([x1_reg, y1_reg, x2_reg, y2_reg], dtype=np.float64)

            def psi_symplectic(pt: NDArray[np.float64]) -> NDArray[np.float64]:
                act1, ang1 = inverse(float(pt[0]), float(pt[1]))
                act2, ang2 = inverse(float(pt[2]), float(pt[3]))
                return np.array([act1, -ang1, act2, -ang2], dtype=np.float64)

            j4 = np.zeros((4, 4), dtype=np.float64)
            step4 = 1e-6
            for col in range(4):
                dq = np.zeros(4, dtype=np.float64)
                dq[col] = step4
                fp = psi_symplectic(base_pt + dq)
                fm = psi_symplectic(base_pt - dq)
                j4[0, col] = (fp[0] - fm[0]) / (2.0 * step4)
                j4[1, col] = wrap(fp[1] - fm[1]) / (2.0 * step4)
                j4[2, col] = (fp[2] - fm[2]) / (2.0 * step4)
                j4[3, col] = wrap(fp[3] - fm[3]) / (2.0 * step4)

            omega4 = _symplectic_omega(2)
            symp_def_4d = _fro(j4.T @ omega4 @ j4 - omega4)
            kappa = float(max(1.0, 1.0 / max(2.0 * gamma_act, _TINY)))

            return PoincareChartCertificate(
                chart=chart,
                mean_longitude=float(m_anom + varpi),
                algebraic_identity_residual=float(algebraic),
                roundtrip_error=float(roundtrip),
                canonicity_defect=float(canonicity),
                chart_condition_number=min(kappa, _CONDITION_CAP),
                poincare_cartan_symplectic_defect_4d=float(symp_def_4d),
                kam_diophantine_margin=float(kam_margin_hint),
            )

        # ── Método 2.4 ────────────────────────────────────────────────────────
        @staticmethod
        def solve_halley_kepler(
            mean_anomaly: float, eccentricity: float, max_iter: int = _HALLEY_MAX_ITER
        ) -> Tuple[float, float]:
            """
            2.4 Solver de Halley de tercer orden para f(E) = E − e·sin E − M = 0 con
            salvaguarda de bisección monótona en [M−e, M+e] [P7].
            Como f'(E) = 1 − e·cos E ≥ 1 − e > 0, existe una única raíz real y el bracket
            garantiza convergencia global incondicional. Residuo evaluado a posteriori.
            """
            m = float(_safe_float(mean_anomaly, 0.0)) % (2.0 * math.pi)
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.999999))
            lo, hi = m - e - 1e-12, m + e + 1e-12
            ecc_anom = m + e * math.sin(m)
            for _ in range(max(1, int(max_iter))):
                f = ecc_anom - e * math.sin(ecc_anom) - m
                if abs(f) <= 4.0 * _WILKINSON_EPS:
                    break
                if f > 0.0:
                    hi = ecc_anom
                else:
                    lo = ecc_anom
                fp = 1.0 - e * math.cos(ecc_anom)
                fpp = e * math.sin(ecc_anom)
                den = 2.0 * fp * fp - f * fpp
                step: Optional[float] = None
                if abs(den) > 1e-14 * max(1.0, abs(fp)):
                    cand = ecc_anom - (2.0 * f * fp) / den
                    if lo < cand < hi:
                        step = cand - ecc_anom
                if step is None:
                    step = 0.5 * (lo + hi) - ecc_anom
                ecc_anom += step
            residual = abs(ecc_anom - e * math.sin(ecc_anom) - m)
            return float(ecc_anom), float(residual)

        # ── Método 2.5 ────────────────────────────────────────────────────────
        def _audit_kepler_torus(self, eccentricity: float, nodes: int = 16) -> KeplerCertificate:
            """
            2.5 Auditoría espectral sobre el toro de anomalías medias M ∈ (0, π):
              • Estima el orden empírico de convergencia p̂ = ln(e_{k+1}/e_k)/ln(e_k/e_{k−1}),
              • Certifica el invariante integral exacto de Kapteyn-Fourier-Bessel:
                    (2/π) ∫₀^π (E(M) − M) · sin(E(M)) dM = e.
                Demostración: como E − M = e·sin E y dM = (1 − e·cos E)dE, la integral se
                transforma en (2/π)∫₀^π e·sin²E·(1 − e·cos E)dE = e − (2e²/π)[sin³E/3]₀^π = e.
                Al ser el integrando impar-periódico y analítico, la cuadratura trapezoidal
                en el toro converge exponencialmente (teorema de Paley-Wiener).
            """
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.999999))
            n_pts = max(4, int(nodes))
            worst, total, orders, safeguard = 0.0, 0, [], 0
            ref_E, ref_res = 0.0, 0.0
            kapteyn_accum = 0.0

            for idx in range(1, n_pts + 1):
                m = math.pi * idx / (n_pts + 1)
                lo, hi = m - e - 1e-12, m + e + 1e-12
                E = m + e * math.sin(m)
                errs: List[float] = []
                its, hits = 0, 0
                for _ in range(_HALLEY_MAX_ITER):
                    its += 1
                    f = E - e * math.sin(E) - m
                    errs.append(abs(f))
                    if abs(f) <= 4.0 * _WILKINSON_EPS:
                        break
                    if f > 0.0:
                        hi = E
                    else:
                        lo = E
                    fp = 1.0 - e * math.cos(E)
                    fpp = e * math.sin(E)
                    den = 2.0 * fp * fp - f * fpp
                    step = None
                    if abs(den) > 1e-14 * max(1.0, abs(fp)):
                        cand = E - (2.0 * f * fp) / den
                        if lo < cand < hi:
                            step = cand - E
                    if step is None:
                        hits += 1
                        step = 0.5 * (lo + hi) - E
                    E += step
                res = abs(E - e * math.sin(E) - m)
                worst = max(worst, res)
                total += its
                safeguard += hits
                kapteyn_accum += (E - m) * math.sin(E)
                if idx == n_pts // 2:
                    ref_E, ref_res = E, res
                if len(errs) >= 3 and errs[-3] > 1e-12 and errs[-2] > 1e-14 and errs[-1] > 0.0:
                    num = math.log(max(errs[-1] / errs[-2], _TINY))
                    den_o = math.log(max(errs[-2] / errs[-3], _TINY))
                    if abs(den_o) > 1e-6:
                        orders.append(float(np.clip(num / den_o, 0.0, 6.0)))

            kapteyn_integral = (2.0 / float(n_pts + 1)) * kapteyn_accum
            fb_error = abs(kapteyn_integral - e)

            return KeplerCertificate(
                eccentric_anomaly=float(ref_E),
                residual=float(ref_res),
                iterations=int(total // n_pts),
                worst_residual=float(worst),
                mean_iterations=float(total / n_pts),
                empirical_order=float(np.mean(orders)) if orders else 0.0,
                bracket_safeguard_hits=int(safeguard),
                fourier_bessel_consistency_error=float(fb_error),
            )

        # ── Método 2.6 ────────────────────────────────────────────────────────
        @staticmethod
        def compute_melnikov_turnstile_lobe_area(
            dissipation_gamma: float = 0.01,
            forcing_amplitude: float = 0.05,
            forcing_frequency: float = 1.0,
        ) -> float:
            """2.6b Fachada pública compatible con v16/v17: área del lóbulo del turnstile."""
            cert = (
                PositronRedemptionGovernorEngine.PhasePi2_PoincareCelestialRegularizer._melnikov_certificate(
                    dissipation_gamma, forcing_amplitude, forcing_frequency
                )
            )
            return cert.lobe_area

        @staticmethod
        def _melnikov_certificate(
            dissipation_gamma: float, forcing_amplitude: float, forcing_frequency: float
        ) -> MelnikovCertificate:
            """
            2.6 Teoría de Poincaré-Melnikov y turnstile de MacKay-Meiss-Percival sobre la
            separatriz homoclínica de Duffing Γ₀(t) = (√2 sech t, −√2 sech t tanh t):
                M(t₀) = −A + B sin(ω t₀),   A = (4/3)γ_diss,   B = √2 π f ω · sech(πω/2).
            Cuando B > A existen ceros simples transversales en ωt₀* = arcsin(A/B) y π − ωt₀*:
              • Área exacta del lóbulo: A_lobe = (2/ω)[√(B² − A²) − A·arccos(A/B)],
              • Flujo de transporte de MacKay: Φ = A_lobe · ω / (2π),
              • Ángulo de escisión homoclínica transversal:
                    θ_split = arctan(|M'(t₀*)| / ‖Γ̇₀(0)‖²) = arctan(ω√(B² − A²) / 2).
            """
            g = max(0.0, _safe_float(dissipation_gamma, 0.0))
            f_amp = max(0.0, _safe_float(forcing_amplitude, 0.0))
            w = max(1e-6, _safe_float(forcing_frequency, 1.0))
            barrier = (4.0 / 3.0) * g
            sech = 1.0 / math.cosh(min(700.0, 0.5 * math.pi * w))
            amp = math.sqrt(2.0) * math.pi * f_amp * w * sech
            crit = barrier / max(math.sqrt(2.0) * math.pi * w * sech, _TINY)

            if amp <= barrier or amp <= 0.0:
                return MelnikovCertificate(
                    lobe_area=0.0,
                    flux=0.0,
                    dissipative_barrier=float(barrier),
                    oscillation_amplitude=float(amp),
                    amplitude_ratio=float(_relative(amp, barrier)) if barrier > 0 else 0.0,
                    transversal=False,
                    critical_forcing=float(min(crit, 1e12)),
                    quadrature_error=0.0,
                    homoclinic_splitting_angle_rad=0.0,
                )

            ratio = float(np.clip(barrier / amp, 0.0, 1.0))
            rad_diff = math.sqrt(max(0.0, amp * amp - barrier * barrier))
            area = (2.0 / w) * (rad_diff - barrier * math.acos(ratio))
            area = float(max(0.0, area))
            phi0 = math.asin(ratio)
            t1, t2 = phi0 / w, (math.pi - phi0) / w
            nodes = 2001
            ts = np.linspace(t1, t2, nodes)
            integrand = amp * np.sin(w * ts) - barrier
            quad = (
                float(np.trapezoid(integrand, ts))
                if hasattr(np, "trapezoid")
                else float(np.trapz(integrand, ts))
            )
            slope_at_zero = w * rad_diff
            splitting_angle = math.atan(0.5 * slope_at_zero)

            return MelnikovCertificate(
                lobe_area=area,
                flux=float(area * w / (2.0 * math.pi)),
                dissipative_barrier=float(barrier),
                oscillation_amplitude=float(amp),
                amplitude_ratio=float(amp / max(barrier, _TINY)),
                transversal=bool(slope_at_zero > 1e-14),
                critical_forcing=float(min(crit, 1e12)),
                quadrature_error=float(abs(quad - area)),
                homoclinic_splitting_angle_rad=float(splitting_angle),
            )

        # ── Método 2.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _yoshida4_hill_monodromy(
            period: float,
            panels: int,
            kappa: float,
            nu: float,
            eps_m: float,
            n_mot: float,
        ) -> NDArray[np.float64]:
            """
            2.7 Integrador simpléctico de Yoshida de 4.º orden (1990) para la monodromía
            de Floquet epicíclica 4×4. Cada sub-paso de Verlet S₂(τ, t) = D(τ/2)·K(τ, t+τ/2)·D(τ/2)
            es producto de cizallas simplécticas exactas con det = 1, y la composición simétrica
                S₄(h, t) = S₂(w₁ h, t + (w₁+w₀)h) · S₂(w₀ h, t + w₁ h) · S₂(w₁ h, t)
            con w₁ = 1/(2 − 2^{1/3}), w₀ = −2^{1/3}/(2 − 2^{1/3}) alcanza orden global O(h⁴)
            preservando det(M) = 1 y Mᵀ Ω₄ M = Ω₄ a precisión de máquina.
            """
            h = period / max(1, int(panels))

            def s2_verlet(tau: float, t_start: float, freq_sq: float) -> NDArray[np.float64]:
                t_mid = t_start + 0.5 * tau
                k_val = freq_sq + eps_m * math.cos(n_mot * t_mid)
                # D(tau/2) @ K(tau, t_mid) @ D(tau/2) en forma cerrada exacta (det = 1):
                half = 0.5 * tau
                a11 = 1.0 - half * tau * k_val
                a12 = tau - (half * half * tau) * k_val
                a21 = -tau * k_val
                a22 = 1.0 - half * tau * k_val
                return np.array([[a11, a12], [a21, a22]], dtype=np.float64)

            h1 = _YOSHIDA_W1 * h
            h0 = _YOSHIDA_W0 * h
            blocks: List[NDArray[np.float64]] = []
            for freq in (kappa, nu):
                f_sq = freq * freq
                mono2 = np.eye(2, dtype=np.float64)
                for idx in range(panels):
                    t0 = idx * h
                    step1 = s2_verlet(h1, t0, f_sq)
                    step2 = s2_verlet(h0, t0 + h1, f_sq)
                    step3 = s2_verlet(h1, t0 + h1 + h0, f_sq)
                    mono2 = (step3 @ step2 @ step1) @ mono2
                blocks.append(mono2)
            return np.asarray(la.block_diag(*blocks), dtype=np.float64)

        # ── Método 2.8 ────────────────────────────────────────────────────────
        def _epicyclic_krein_monodromy(
            self, mean_motion: float, eccentricity: float, modulation: float
        ) -> KreinMonodromyCertificate:
            """
            2.8 Monodromía de Floquet epicíclica REAL del sistema variacional de Hill [P2]:
                ẍ + (κ² + ε_m cos(nt))x = 0,   ÿ + (ν² + ε_m cos(nt))y = 0,
            con κ = n (radial) y ν = n(1 + splitting) (vertical).
            Integra por propagador exponencial exacto a trozos con duplicación adaptativa de
            paneles (Richardson) y audita su concordancia frente al integrador simpléctico de
            Yoshida de 4.º orden (Método 2.7). Extrae:
              • Defecto simpléctico ‖Mᵀ Ω₄ M − Ω₄‖_F y det(M) = +1,
              • Signaturas simplécticas de Krein κ(ξ) = sgn(−i ξ† Ω₄ ξ) y colisiones 1:1,
              • Número de rotación de Poincaré ν_rot = arccos(½ Tr M_rad)/(2π) ∈ [0, ½],
              • Índice de estabilidad de Hamilton-Hopf / Broucke Δ_HH = (Tr M_rad)² − 4 ≤ 0.
            """
            n_mot = max(1e-6, _safe_float(mean_motion, 1.0))
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
            eps_m = max(0.0, _safe_float(modulation, 2.0 * e * n_mot * n_mot))
            kappa = n_mot
            nu = n_mot * (1.0 + self._splitting)
            period = 2.0 * math.pi / n_mot
            omega = self._omega4

            def build_piecewise_exact(panels: int) -> NDArray[np.float64]:
                h = period / panels
                blocks = []
                for freq in (kappa, nu):
                    mono = np.eye(2, dtype=np.float64)
                    for idx in range(panels):
                        t_mid = (idx + 0.5) * h
                        k_coef = freq * freq + eps_m * math.cos(n_mot * t_mid)
                        if k_coef > 1e-14:
                            r = math.sqrt(k_coef)
                            blk = np.array(
                                [
                                    [math.cos(r * h), math.sin(r * h) / r],
                                    [-r * math.sin(r * h), math.cos(r * h)],
                                ],
                                dtype=np.float64,
                            )
                        elif k_coef < -1e-14:
                            r = math.sqrt(-k_coef)
                            blk = np.array(
                                [
                                    [math.cosh(r * h), math.sinh(r * h) / r],
                                    [r * math.sinh(r * h), math.cosh(r * h)],
                                ],
                                dtype=np.float64,
                            )
                        else:
                            blk = np.array([[1.0, h], [0.0, 1.0]], dtype=np.float64)
                        mono = blk @ mono
                    blocks.append(mono)
                return np.asarray(la.block_diag(*blocks), dtype=np.float64)

            panels = _MONODROMY_BASE_PANELS
            mono = build_piecewise_exact(panels)
            step_cert = 0.0
            if self._doubling:
                for _ in range(_MONODROMY_MAX_DOUBLINGS):
                    finer = build_piecewise_exact(2 * panels)
                    step_cert = _relative(_fro(finer - mono), _fro(finer))
                    mono, panels = finer, 2 * panels
                    if step_cert <= 1e-11:
                        break

            mono_yoshida = self._yoshida4_hill_monodromy(
                period, max(_MONODROMY_BASE_PANELS, panels // 2), kappa, nu, eps_m, n_mot
            )
            yoshida_defect = _relative(_fro(mono - mono_yoshida), _fro(mono))

            sympl = _fro(mono.T @ omega @ mono - omega)
            det = float(np.linalg.det(mono))
            eigvals, eigvecs = la.eig(mono)
            radius = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0

            signatures: List[Tuple[complex, float]] = []
            for idx in range(eigvals.size):
                lam = complex(eigvals[idx])
                if abs(abs(lam) - 1.0) <= 1e-6 and abs(lam.imag) > 1e-8:
                    xi = eigvecs[:, idx]
                    krein = float(np.real(-1j * np.vdot(xi, omega @ xi)))
                    signatures.append(
                        (lam, math.copysign(1.0, krein) if abs(krein) > 1e-12 else 0.0)
                    )
            collision = False
            for i in range(len(signatures)):
                for jdx in range(i + 1, len(signatures)):
                    li, ki = signatures[i]
                    lj, kj = signatures[jdx]
                    if ki * kj < 0.0 and abs(li - lj) < 1e-3:
                        collision = True

            tr_rad = float(mono[0, 0] + mono[1, 1])
            half_tr_clamped = float(np.clip(0.5 * tr_rad, -1.0, 1.0))
            rot_num = float(math.acos(half_tr_clamped) / (2.0 * math.pi))
            hh_index = float(tr_rad * tr_rad - 4.0)

            return KreinMonodromyCertificate(
                symplectic_defect=sympl,
                determinant=det,
                spectral_radius=radius,
                max_lyapunov_exponent=float(math.log(max(radius, _TINY)) / period),
                krein_signatures=tuple(float(s) for _, s in signatures),
                collision_detected=bool(collision),
                step_certificate=float(step_cert),
                panels=int(panels),
                radial_epicyclic_freq=float(kappa),
                vertical_epicyclic_freq=float(nu),
                yoshida4_agreement_defect=float(yoshida_defect),
                poincare_rotation_number=rot_num,
                hamilton_hopf_index=hh_index,
            )

        # ── Método 2.9 ────────────────────────────────────────────────────────
        @staticmethod
        def _lindstedt_agm_resonance(
            kappa: float, modulation: float, eccentricity: float
        ) -> Tuple[float, float, float, int]:
            """
            2.9 [Corrección Analítica P16] Serie de Lindstedt-Poincaré sin términos seculares
            para el oscilador anarmónico de Duffing ẍ + κ²x + ε_m x³ = 0 con amplitud a ≡ e.
            Con el parámetro adimensional de no-linealidad δ = ε_m e² / κ²:
              • Serie perturbativa de Lindstedt-Poincaré hasta O(δ³):
                    ω_LP(δ) = κ · (1 + 3δ/8 − 21δ²/256 + 81δ³/2048),
              • Frecuencia elíptica EXACTA de Jacobi-Gauss-Salamin:
                    ω_ex(δ) = κ √(1 + δ) · π / (2 K(k)) = κ √(1 + δ) · AGM(1, √(1 − k²)),
                donde el módulo elíptico exacto es k² = δ / (2(1 + δ)) [y NO δ/(1+2δ) como en v17].
            Con esta identidad exacta, |ω_LP − ω_ex|/ω_ex = (1095/32768)δ⁴ + O(δ⁵) (~1.3e−7
            para δ=0.045), satisfaciendo holgadamente el umbral de certificación < 1e−4.
            """
            k0 = max(1e-6, _safe_float(kappa, 1.0))
            eps = max(0.0, _safe_float(modulation, 0.0))
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.99))
            delta = float(np.clip(eps * e * e / (k0 * k0), 0.0, 1e3))

            series = k0 * (
                1.0
                + (3.0 / 8.0) * delta
                - (21.0 / 256.0) * (delta ** 2)
                + (81.0 / 2048.0) * (delta ** 3)
            )
            k_sq = delta / (2.0 * (1.0 + delta))
            agm, iters = _arithmetic_geometric_mean(1.0, math.sqrt(max(0.0, 1.0 - k_sq)))
            exact = k0 * math.sqrt(1.0 + delta) * agm
            remainder = _relative(abs(series - exact), max(exact, _TINY))
            return float(series), float(exact), float(remainder), int(iters)

        # ── Método 2.10 · TERMINAL Π₂ → APERTURA DIRECTA DE Π₃ ───────────────
        def audit_poincare_celestial_sector(
            self,
            pi1_certificate: Optional[Pi1SpectrumCertificate] = None,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mean_anomaly: float = 0.5,
            dissipation_gamma: float = 0.01,
            forcing_amplitude: float = 0.05,
            forcing_frequency: float = 1.0,
            mu: float = 1.0,
        ) -> Pi2SpectrumCertificate:
            """
            2.10 [DEFINICIÓN TERMINAL DE LA FASE Π₂ ──► CONTINUACIÓN DIRECTA EN 3.1]
            Orquesta la secuencia 2.1 → 2.9:
              1. Ingiere `Pi1ToPi2Seed` desde 1.12 (`_ingest_pi1_continuation`),
              2. Acopla la tensión de Bures de Π₁ al forzamiento homoclínico acotado,
              3. Evalúa la frecuencia no lineal exacta de Lindstedt-AGM y el margen KAM,
              4. Certifica la carta canónica de Poincaré-Cartan 4D, el toro de Kepler, el
                 turnstile de Melnikov y la monodromía de Floquet-Yoshida-Krein,
              5. Emite el morfismo semilla `Pi2ToPi3Seed` que abre formalmente la Fase Π₃ en
                 `PhasePi3_RSI3TrajectoryIntegrator._ingest_pi2_continuation` (Método 3.1).
            """
            seed1 = self._ingest_pi1_continuation(pi1_certificate)
            forcing_eff = max(0.0, _safe_float(forcing_amplitude, 0.05)) + min(
                1e-3, seed1.bures_tension * 1e-3
            )
            n_motion = math.sqrt(
                max(_TINY, _safe_float(mu, 1.0))
                / max(1e-9, _safe_float(semi_major_axis, 1.0)) ** 3
            )
            e_clamped = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
            modulation = 2.0 * e_clamped * n_motion * n_motion

            w_ser, w_ex, rem, agm_its = self._lindstedt_agm_resonance(
                n_motion, modulation, eccentricity
            )
            kam_margin = self._kam_diophantine_margin(
                w_ex, max(1e-6, _safe_float(forcing_frequency, 1.0))
            )
            chart = self._chart_with_certificate(
                semi_major_axis,
                eccentricity,
                inclination_rad,
                mean_anomaly,
                0.0,
                0.0,
                mu,
                kam_margin_hint=kam_margin,
            )
            kepler = self._audit_kepler_torus(eccentricity)
            melnikov = self._melnikov_certificate(
                dissipation_gamma, forcing_eff, forcing_frequency
            )
            krein = self._epicyclic_krein_monodromy(n_motion, eccentricity, modulation)

            seed_to_pi3 = Pi2ToPi3Seed(
                modulation_amplitude=float(modulation),
                epicyclic_frequency=float(n_motion),
                melnikov_flux=float(melnikov.flux),
                lindstedt_remainder=float(rem),
                kepler_residual=float(kepler.worst_residual),
                has_context=True,
                poincare_rotation_number=float(krein.poincare_rotation_number),
                kam_diophantine_margin=float(kam_margin),
                bures_tension_forwarded=float(seed1.bures_tension),
            )

            return Pi2SpectrumCertificate(
                chart=chart,
                kepler=kepler,
                melnikov=melnikov,
                krein=krein,
                lindstedt_omega_series=w_ser,
                lindstedt_omega_exact=w_ex,
                lindstedt_remainder_rel=rem,
                agm_iterations=agm_its,
                continuation=seed_to_pi3,
            )


# ──────────────────────────────────────────────────────────────────────────
    # FASE Π₃ · RSI³ = R₃∘R₂∘R₁ SOBRE Δ^{n−1} × D(H), STRANG-BCH Y EMISIÓN
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi3_RSI3TrajectoryIntegrator:
        """
        Π₃. Cadena formal anidada (continuación directa de 2.10):
        3.1 INGESTA de Pi2ToPi3Seed → 3.2 proyección sobre Δ^{n−1} con KKT →
        3.3 purificación de K en PSD ∩ Sym_n(ℝ) y gap espectral real →
        3.4 flujo replicador exponencial de Shahshahani (R₁) →
        3.5 flujo cuántico isoespectral de doble corchete de Brockett (R₃) →
        3.6 estimador BCH y selector RSI³ de sub-paso m* ∈ {1,2,4} →
        3.7 splitting simétrico de Strang de 2.º orden →
        3.8 auditoría de energía de Lyapunov e instantánea de Richardson →
        3.9 cotas operatoriales de Connes-Daleckii-Krein →
        3.10 tasa espectral mínima de Brockett →
        3.11 TERMINAL (composición monádica R₃∘R₂∘R₁ y emisión del certificado).
        """

        def __init__(
            self,
            pi1: "PositronRedemptionGovernorEngine.PhasePi1_DiracUhlmannAnnihilator",
            pi2: "PositronRedemptionGovernorEngine.PhasePi2_PoincareCelestialRegularizer",
            n: int,
            stability_threshold: float,
        ) -> None:
            self._pi1 = pi1
            self._pi2 = pi2
            self._n = int(n)
            self._floor = float(stability_threshold)
            self._uniform = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)

        # ── Método 3.1 · APERTURA Π₃ (continuación directa de 2.10) ──────────
        @staticmethod
        def _ingest_pi2_continuation(
            pi2_certificate: Optional[Pi2SpectrumCertificate] = None,
        ) -> Pi2ToPi3Seed:
            """
            3.1 [CONTINUACIÓN FORMAL DIRECTA DE 2.10 audit_poincare_celestial_sector]
            Absorbe la semilla celeste `Pi2ToPi3Seed` emitida al cierre de la Fase Π₂.
            """
            if pi2_certificate is not None and pi2_certificate.continuation is not None:
                return pi2_certificate.continuation
            return Pi2ToPi3Seed(
                modulation_amplitude=0.0,
                epicyclic_frequency=1.0,
                melnikov_flux=0.0,
                lindstedt_remainder=0.0,
                kepler_residual=0.0,
                has_context=False,
                poincare_rotation_number=0.0,
                kam_diophantine_margin=0.0,
                bures_tension_forwarded=0.0,
            )

        # ── Método 3.2 ────────────────────────────────────────────────────────
        def project_probability_simplex(
            self, v: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            """3.2b Fachada pública compatible con v16/v17: proyección euclídea sobre Δ^{n−1}."""
            return self._simplex_with_kkt(v)[0]

        def _simplex_with_kkt(
            self, v: Any
        ) -> Tuple[NDArray[np.float64], bool, float]:
            """
            3.2 Proyección euclídea EXACTA sobre el símplex de probabilidad Δ^{n−1}
            (Duchi et al. 2008) con verificación de condiciones KKT:
                w ≥ 0,  Σ_i w_i = 1,  w_i > 0 ⟹ v_i − θ = w_i (holgura complementaria).
            La restauración de suma unitaria frente a redondeo IEEE-754 es ADITIVA sobre
            argmax(w), preservando la exactitud del proyector ortogonal.
            """
            try:
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return self._uniform.copy(), False, 0.0
            if arr.size != self._n or not bool(np.all(np.isfinite(arr))):
                return self._uniform.copy(), False, 0.0
            u = np.sort(arr)[::-1]
            cssv = np.cumsum(u) - 1.0
            ind = np.arange(1, self._n + 1, dtype=np.float64)
            cond = u > (cssv / ind)
            if not bool(np.any(cond)):
                return (
                    self._uniform.copy(),
                    True,
                    float(np.linalg.norm(self._uniform - arr)),
                )
            rho_idx = int(np.flatnonzero(cond)[-1])
            theta = float(cssv[rho_idx] / float(rho_idx + 1))
            w = np.maximum(arr - theta, 0.0)
            w[int(np.argmax(w))] += 1.0 - float(np.sum(w))
            kkt = bool(
                float(np.min(w)) >= -1e-14 and abs(float(np.sum(w)) - 1.0) <= 1e-12
            )
            return w, kkt, float(np.linalg.norm(w - arr))

        # ── Método 3.3 ────────────────────────────────────────────────────────
        def purify_conductivity_matrix(
            self, K_raw: NDArray[np.complex128]
        ) -> Tuple[NDArray[np.float64], float, float]:
            """3.3b Fachada pública compatible con v16/v17: (K_psd_real, κ_capped, λ_min)."""
            out = self._purify_conductivity_with_gaps(K_raw)
            return out[0], out[1], out[2]

        def _purify_conductivity_with_gaps(
            self, K_raw: Any
        ) -> Tuple[NDArray[np.float64], float, float, float, float, float]:
            """
            3.3 Proyección ortogonal de Frobenius sobre el cono PSD ∩ Sym_n(ℝ) en el orden
            variacionalmente óptimo [P4]:
                ‖K − S‖_F² = ‖Re(K_H) − S‖_F² + ‖Im(K_H)‖_F²  ∀ S ∈ Sym_n(ℝ).
            Por tanto, se proyecta primero sobre Sym_n(ℝ) (parte real simétrica) y después
            se recorta el espectro por debajo en δ_floor (teorema de Hoffman-Wielandt).
            Devuelve también el gap espectral principal λ₁ − λ₂ ≥ 0 [P6].
            """
            try:
                arr = np.asarray(K_raw, dtype=np.complex128)
            except (TypeError, ValueError):
                arr = np.eye(self._n, dtype=np.complex128)
            if arr.shape != (self._n, self._n) or not bool(np.all(np.isfinite(arr))):
                arr = np.eye(self._n, dtype=np.complex128)
            herm = 0.5 * (arr + arr.conj().T)
            real_sym = np.asarray(np.real(herm), dtype=np.float64)
            real_sym = 0.5 * (real_sym + real_sym.T)
            eigvals, eigvecs = la.eigh(real_sym)
            clamped = np.maximum(np.real(eigvals), self._floor)
            k_psd = np.asarray((eigvecs * clamped) @ eigvecs.T, dtype=np.float64)
            k_psd = 0.5 * (k_psd + k_psd.T)
            lam_min = float(np.min(clamped))
            lam_max = float(np.max(clamped))
            ordered = np.sort(clamped)[::-1]
            top_gap = float(ordered[0] - ordered[1]) if ordered.size >= 2 else 0.0
            kappa_raw = float(lam_max / max(lam_min, _TINY))
            kappa = float(min(kappa_raw, _CONDITION_CAP))
            return (
                k_psd,
                kappa,
                lam_min,
                top_gap,
                kappa_raw,
                float(math.log10(max(kappa_raw, 1.0))),
            )

        # ── Método 3.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _shahshahani_replicator_step(
            p: NDArray[np.float64], k_mat: NDArray[np.float64], dt: float
        ) -> NDArray[np.float64]:
            """
            3.4 R₁ (Geometría de Información de Shahshahani): mapa exponencial del campo
            replicador con paisaje de adecuación congelado Kp:
                p_i(t + dt) = p_i(t) exp(dt · (Kp)_i) / Σ_j p_j(t) exp(dt · (Kp)_j).
            Estabilizado en dominio logarítmico mediante traslación invariante por el máximo.
            """
            h = max(0.0, _safe_float(dt, 0.0))
            if h <= 0.0:
                return p.copy()
            kp = k_mat @ p
            logits = np.log(np.maximum(p, _TINY)) + h * kp
            logits -= float(np.max(logits))
            w = np.exp(np.clip(logits, -700.0, 0.0))
            total = float(np.sum(w))
            if total <= _TINY or not math.isfinite(total):
                return p.copy()
            out = w / total
            out[int(np.argmax(out))] += 1.0 - float(np.sum(out))
            return np.maximum(out, 0.0)

        # ── Método 3.5 ────────────────────────────────────────────────────────
        @staticmethod
        def _brockett_double_bracket_step(
            rho: NDArray[np.complex128], p: NDArray[np.float64], dt: float
        ) -> Tuple[NDArray[np.complex128], float]:
            """
            3.5 R₃ (Flujo Isoespectral de Doble Corchete de Brockett):
                ρ̇ = [ρ, [ρ, N(p)]],   donde N(p) = diag(p).
            Como ρ y N(p) son hermíticos, el conmutador A = [ρ, N(p)] ∈ 𝔲(n) es antihermítico
            (A† = −A), de modo que U = exp(−dt · A) ∈ U(n) es estrictamente unitario y
            ρ ↦ U ρ U† preserva exactamente el espectro y la entropía de von Neumann [P1].
            NO se invoca `canonicalize_density_matrix` dentro del lazo para no inyectar
            entropía espuria de truncación.
            """
            h = max(0.0, _safe_float(dt, 0.0))
            if h <= 0.0:
                return rho.copy(), 0.0
            n_op = np.diag(np.asarray(p, dtype=np.float64)).astype(np.complex128)
            a_gen = rho @ n_op - n_op @ rho
            a_gen = 0.5 * (a_gen - a_gen.conj().T)
            u_mat = la.expm(-a_gen * h)
            out = u_mat @ rho @ u_mat.conj().T
            out = 0.5 * (out + out.conj().T)
            tr = float(np.real(np.trace(out)))
            if tr > _TINY:
                out = out + ((1.0 - tr) / out.shape[0]) * np.eye(
                    out.shape[0], dtype=np.complex128
                )
            spec_before = np.sort(np.real(la.eigvalsh(rho)))
            spec_after = np.sort(np.real(la.eigvalsh(out)))
            iso_defect = float(np.linalg.norm(spec_after - spec_before, ord=2))
            return out, iso_defect

        # ── Método 3.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _bch_commutator_and_rsi3_substeps(
            p: NDArray[np.float64],
            rho: NDArray[np.complex128],
            k_mat: NDArray[np.float64],
            dt: float,
            seed2: Pi2ToPi3Seed,
        ) -> Tuple[float, int]:
            """
            3.6 [RSI³ Adaptación Numérica de Sub-paso por Conmutador de Baker-Campbell-Hausdorff]
            Evalúa el conmutador de Lie entre el generador antihermítico de Brockett
            A(p) = [ρ, diag(p)] y su derivada direccional a lo largo del flujo de Shahshahani
            Ȧ(p) = [ρ, diag(ṗ)], con ṗ = p ⊙ (Kp − (pᵀKp)1):
                C_BCH = [A(p), Ȧ(p)] = A(p)Ȧ(p) − Ȧ(p)A(p).
            El residuo local de escisión de Strang escala como dt³ ‖C_BCH‖_F / 24.
            Combinando ‖C_BCH‖_F con el resto de Lindstedt-Poincaré y el residuo de Kepler de
            `Pi2ToPi3Seed`, selecciona determinísticamente el número de sub-pasos m* ∈ {1, 2, 4}.
            """
            kp = k_mat @ p
            phi = float(np.dot(p, kp))
            p_dot = p * (kp - phi)
            n_p = np.diag(p).astype(np.complex128)
            n_pdot = np.diag(p_dot).astype(np.complex128)
            a_op = rho @ n_p - n_p @ rho
            adot_op = rho @ n_pdot - n_pdot @ rho
            bch_comm = a_op @ adot_op - adot_op @ a_op
            bch_norm = _fro(bch_comm)

            h = max(0.0, _safe_float(dt, 0.05))
            local_bch_est = (h ** 3) * bch_norm / 24.0
            celestial_pressure = seed2.lindstedt_remainder + seed2.kepler_residual
            composite_indicator = local_bch_est + 0.01 * celestial_pressure

            if composite_indicator > 1.0e-4:
                substeps = 4
            elif composite_indicator > 1.0e-6:
                substeps = 2
            else:
                substeps = 1
            return float(bch_norm), int(substeps)

        # ── Método 3.7 ────────────────────────────────────────────────────────
        def _strang_split_step(
            self,
            p: NDArray[np.float64],
            rho: NDArray[np.complex128],
            k_mat: NDArray[np.float64],
            dt: float,
            lindblad_rate: float,
        ) -> Tuple[NDArray[np.float64], NDArray[np.complex128], float, float, float, bool]:
            """
            3.7 Operador simétrico de Strang de 2.º orden [P13]:
                S_Strang(h) = R₁(h/2) ∘ Φ_GKSL(h/2) ∘ R₃(h, p_{k+1/2}) ∘ Φ_GKSL(h/2) ∘ R₁(h/2).
            Aprovecha la estructura triangular del acoplamiento (ṗ = f(p), ρ̇ = g(ρ, p))
            evaluando el generador de Brockett exactamente en el punto medio temporal p_{k+1/2}.
            """
            h = max(0.0, _safe_float(dt, 0.0))
            p_half = self._shahshahani_replicator_step(p, k_mat, 0.5 * h)
            rho_a, ds_a, tr_a, ok_a = self._pi1._gksl_dephasing_step(
                rho, lindblad_rate, 0.5 * h
            )
            rho_mid, iso = self._brockett_double_bracket_step(rho_a, p_half, h)
            rho_b, ds_b, tr_b, ok_b = self._pi1._gksl_dephasing_step(
                rho_mid, lindblad_rate, 0.5 * h
            )
            p_full = self._shahshahani_replicator_step(p_half, k_mat, 0.5 * h)
            return (
                p_full,
                rho_b,
                iso,
                ds_a + ds_b,
                tr_a + tr_b,
                bool(ok_a and ok_b),
            )

        # ── Método 3.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _entropy_of(rho: NDArray[np.complex128]) -> float:
            """S(ρ) = −Tr(ρ ln ρ) = −Σ_i λ_i ln λ_i evaluada espectralmente sin `la.logm`."""
            ev = np.maximum(np.real(la.eigvalsh(rho)), _WILKINSON_EPS)
            return float(-np.sum(ev * np.log(ev)))

        def _energy_and_lyapunov_audit(
            self,
            p: NDArray[np.float64],
            rho: NDArray[np.complex128],
            k_mat: NDArray[np.float64],
            lindblad_rate: float,
        ) -> Dict[str, float]:
            """
            3.8 Auditoría descompuesta de la función de Lyapunov-Helmholtz [P12]:
                H(p, ρ) = −½ pᵀ K p + S(ρ).
              • Sector de Shahshahani (R₁): d(−½pᵀKp)/dt = −Var_p(Kp) ≤ 0 (Teorema Fundamental
                de Selección Natural de Fisher en la métrica de Shahshahani).
              • Sector de Brockett (R₃): dS(ρ)/dt|_R₃ = 0 exactamente por isoespectralidad.
              • Sector disipativo GKSL: dS(ρ)/dt|_GKSL ≥ 0 por mayorización de Schur.
              • Derivada numérica INSTANTÁNEA mediante extrapolación de Richardson O(h²) en el
                estado terminal, eliminando el sesgo de secante media de v16.
            """
            kp = k_mat @ p
            phi = float(np.dot(p, kp))
            var = float(np.dot(p, (kp - phi) ** 2))
            var = max(0.0, var)
            entropy = self._entropy_of(rho)
            h_energy = -0.5 * phi + entropy

            h_step = 1.0e-3
            p1, r1, _, ds1, _, _ = self._strang_split_step(
                p, rho, k_mat, h_step, lindblad_rate
            )
            e1 = -0.5 * float(np.dot(p1, k_mat @ p1)) + self._entropy_of(r1)
            p2, r2, _, ds2, _, _ = self._strang_split_step(
                p, rho, k_mat, 0.5 * h_step, lindblad_rate
            )
            e2 = -0.5 * float(np.dot(p2, k_mat @ p2)) + self._entropy_of(r2)
            d1 = (e1 - h_energy) / h_step
            d2 = (e2 - h_energy) / (0.5 * h_step)
            richardson = 2.0 * d2 - d1
            analytic_total = -var + (ds2 / max(0.5 * h_step, _TINY))
            discrepancy = _relative(
                abs(richardson - analytic_total), max(abs(analytic_total), 1e-12)
            )
            return {
                "energy": float(h_energy),
                "phi": phi,
                "variance": var,
                "replicator_term": float(-var),
                "entropy": entropy,
                "dissipative_term": float(ds2 / max(0.5 * h_step, _TINY)),
                "numerical_derivative": float(richardson),
                "total_analytic": float(analytic_total),
                "discrepancy_rel": float(min(discrepancy, 1e6)),
                "speed": math.sqrt(var),
            }

        # ── Método 3.9 ────────────────────────────────────────────────────────
        @staticmethod
        def _connes_daleckii_krein_bounds(lambda_min: float) -> Tuple[float, float]:
            """
            3.9 Cotas de sensibilidad operatorial de Daleckii-Krein en norma de Frobenius [P5]:
              • Para f(x) = x^{1/2}:  max_{a,b ≥ λ_min} |f^[1](a,b)| = 1 / (2 √λ_min),
              • Para f(x) = x^{−1/2}: max_{a,b ≥ λ_min} |f^[1](a,b)| = 1 / (2 λ_min^{3/2}).
            """
            lam = max(_safe_float(lambda_min, 0.0), _STABILITY_FLOOR)
            return float(0.5 / math.sqrt(lam)), float(0.5 / (lam ** 1.5))

        # ── Método 3.10 ───────────────────────────────────────────────────────
        @staticmethod
        def _brockett_min_rate(
            rho: NDArray[np.complex128], p: NDArray[np.float64]
        ) -> float:
            """
            3.10 Tasa espectral mínima de amortiguamiento transversal del flujo de Brockett
            alrededor de la variedad diagonal D = diag(d) [P6]:
                dE_{ij}/dt = −(p_i − p_j)(d_i − d_j) E_{ij}  ⟹  γ_Brockett = min_{(p_i−p_j)(d_i−d_j)>0} (p_i−p_j)(d_i−d_j).
            """
            d = np.real(np.diag(rho))
            rates: List[float] = []
            for i in range(d.size):
                for j in range(i + 1, d.size):
                    r = (p[i] - p[j]) * (d[i] - d[j])
                    if r > 1e-14:
                        rates.append(float(r))
            return float(min(rates)) if rates else 0.0

        # ── Método 3.11 · TERMINAL Π₃ · COMPOSICIÓN R₃∘R₂∘R₁ Y EMISIÓN ────────
        def execute_positron_redemption_trajectory(
            self,
            p_init: NDArray[np.float64],
            rho_init: NDArray[np.complex128],
            sigma_target: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
            dt: float = 0.05,
            steps: int = 4,
            *,
            positron_beta: float = 0.01,
            lindblad_rate: float = 0.0,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mean_anomaly: float = 0.5,
            dissipation_gamma: float = 0.01,
            forcing_amplitude: float = 0.05,
            forcing_frequency: float = 1.0,
            enable_order_certificate: bool = True,
        ) -> PositronRedemptionSpectrumReport:
            """
            3.11 [DEFINICIÓN TERMINAL DE LA FASE Π₃ · EMISIÓN DEL CERTIFICADO ESPECTRAL]
            Ejecuta la composición monádica completa con continuidad formal estricta:
                1.12 audit_dirac_uhlmann_substrate ──[Pi1ToPi2Seed]──►
                2.1  _ingest_pi1_continuation … 2.10 audit_poincare_celestial_sector ──[Pi2ToPi3Seed]──►
                3.1  _ingest_pi2_continuation … 3.11 execute_positron_redemption_trajectory.
            Integra la trayectoria acoplada en Δ^{n−1} × D_δ(H) con sub-stepping RSI³ guiado por
            el conmutador de Baker-Campbell-Hausdorff y emite `PositronRedemptionSpectrumReport`.
            """
            flags: List[str] = []
            h = _safe_float(dt, 0.05)
            if h <= 0.0:
                h = 0.05
                flags.append("DT_NON_POSITIVE_RESET")
            n_steps = (
                int(steps)
                if isinstance(steps, (int, np.integer)) and not isinstance(steps, bool)
                else 4
            )
            if n_steps < 0:
                n_steps = 0
                flags.append("STEPS_NEGATIVE_CLAMPED")
            rate = max(0.0, _safe_float(lindblad_rate, 0.0))
            if rate > 0.0:
                flags.append("GKSL_DISSIPATION_ACTIVE_LYAPUNOV_SIGN_INDEFINITE")

            # ── Ejecución de Fase Π₁ (1.1 → 1.12) ──
            pi1 = self._pi1.audit_dirac_uhlmann_substrate(
                rho_init, sigma_target, positron_beta
            )

            # ── Ejecución de Fase Π₂ (2.1 → 2.10, recibiendo Pi1ToPi2Seed) ──
            pi2 = self._pi2.audit_poincare_celestial_sector(
                pi1_certificate=pi1,
                semi_major_axis=semi_major_axis,
                eccentricity=eccentricity,
                inclination_rad=inclination_rad,
                mean_anomaly=mean_anomaly,
                dissipation_gamma=dissipation_gamma,
                forcing_amplitude=forcing_amplitude,
                forcing_frequency=forcing_frequency,
            )

            # ── Ejecución de Fase Π₃ (3.1 → 3.11, recibiendo Pi2ToPi3Seed) ──
            seed2 = self._ingest_pi2_continuation(pi2)
            p_curr, simplex_kkt, _cost = self._simplex_with_kkt(p_init)
            if not simplex_kkt:
                flags.append("SIMPLEX_INPUT_FALLBACK_OR_KKT_MARGINAL")

            k_mat, kappa, lam_min, top_gap, kappa_raw, log_kappa = (
                self._purify_conductivity_with_gaps(K_raw)
            )
            rho_curr = pi1.rho.copy()

            bch_norm, rsi3_substeps = self._bch_commutator_and_rsi3_substeps(
                p_curr, rho_curr, k_mat, h, seed2
            )
            sub_h = h / float(rsi3_substeps)

            iso_total, ds_total, tr_total, cptp_ok = 0.0, 0.0, 0.0, True
            for _ in range(n_steps * rsi3_substeps):
                p_curr, rho_curr, iso, ds, trd, ok = self._strang_split_step(
                    p_curr, rho_curr, k_mat, sub_h, rate
                )
                iso_total = max(iso_total, iso)
                ds_total += ds
                tr_total = max(tr_total, trd)
                cptp_ok = bool(cptp_ok and ok)

            if iso_total > 1e-9:
                flags.append("ISOSPECTRALITY_DEFECT_EXCEEDED")

            # Certificado de orden de convergencia de Strang por extrapolación de Richardson:
            order_obs, step_cert = 0.0, 0.0
            if enable_order_certificate and n_steps > 0:
                pa, ra, *_ = self._strang_split_step(p_curr, rho_curr, k_mat, h, rate)
                pb, rb, *_ = self._strang_split_step(
                    p_curr, rho_curr, k_mat, 0.5 * h, rate
                )
                pb, rb, *_ = self._strang_split_step(pb, rb, k_mat, 0.5 * h, rate)
                d_h = max(float(np.linalg.norm(pa - pb, ord=1)), _fro(ra - rb))
                pc, rc, *_ = self._strang_split_step(
                    p_curr, rho_curr, k_mat, 0.25 * h, rate
                )
                for _ in range(3):
                    pc, rc, *_ = self._strang_split_step(pc, rc, k_mat, 0.25 * h, rate)
                d_h2 = max(float(np.linalg.norm(pb - pc, ord=1)), _fro(rb - rc))
                step_cert = float(d_h)
                if d_h > 1e-15 and d_h2 > 1e-15:
                    order_obs = float(np.clip(math.log2(d_h / d_h2), 0.0, 6.0))

            audit = self._energy_and_lyapunov_audit(p_curr, rho_curr, k_mat, rate)
            n_op = np.diag(p_curr).astype(np.complex128)
            comm = rho_curr @ n_op - n_op @ rho_curr
            dk_sqrt, dk_inv = self._connes_daleckii_krein_bounds(lam_min)
            brockett_rate = self._brockett_min_rate(rho_curr, p_curr)
            passivity = bool(rate <= 0.0 and audit["replicator_term"] <= 1e-14)

            if not pi1.pinsker_ok:
                flags.append("PINSKER_RESIDUAL_NEGATIVE_OR_NON_FINITE")
            if not pi1.fuchs_sandwich_ok:
                flags.append("FUCHS_VAN_DE_GRAAF_SANDWICH_VIOLATION")
            if pi1.rho_canon.used_fallback or pi1.sigma_canon.used_fallback:
                flags.append("DENSITY_INPUT_FALLBACK")
            if pi2.krein.collision_detected:
                flags.append("KREIN_1TO1_EPICYCLIC_RESONANCE_DETECTED")
            if pi2.chart.canonicity_defect > 1e-5:
                flags.append("POINCARE_CHART_CANONICITY_DEFECT")

            hol = pi1.holonomy
            panch_phase = hol.pancharatnam_phase_rad if hol is not None else 0.0
            hol_unit_def = hol.unitarity_defect if hol is not None else 0.0
            quat_s3 = hol.quaternion_s3 if hol is not None else (1.0, 0.0, 0.0, 0.0)

            return PositronRedemptionSpectrumReport(
                p_final=_freeze_array(p_curr),
                rho_final=_freeze_array(rho_curr),
                K_purified=_freeze_array(k_mat),
                uhlmann_fidelity=float(pi1.uhlmann_fidelity),
                bures_distance=float(pi1.bures_distance),
                umegaki_relative_entropy=float(pi1.umegaki_relative_entropy),
                pinsker_bound_residual=float(pi1.pinsker_residual),
                annihilation_energy_kev=float(pi1.dirac.energy_rest_kev),
                amber_grace_duration_sec=float(pi1.amber_grace_duration_sec),
                poincare_canonical_residual=float(
                    pi2.chart.algebraic_identity_residual
                ),
                halley_kepler_residual=float(pi2.kepler.residual),
                melnikov_lobe_area=float(pi2.melnikov.lobe_area),
                krein_poincare_cartan_drift=float(pi2.krein.symplectic_defect),
                hamiltonian_energy=float(audit["energy"]),
                analytic_energy_derivative=float(audit["replicator_term"]),
                numerical_energy_derivative=float(audit["numerical_derivative"]),
                shahshahani_speed=float(audit["speed"]),
                brockett_alignment=float(np.real(np.trace(rho_curr @ n_op))),
                commutator_frobenius=_fro(comm),
                connes_lipschitz_bound=float(dk_inv),
                condition_number_K=float(kappa),
                spectral_gap_K=float(top_gap),
                bures_angle=float(pi1.bures_angle),
                chord_arc_identity_error=float(pi1.chord_arc_error),
                fidelity_route_discrepancy=float(pi1.fidelity_route_discrepancy),
                trace_distance=float(pi1.trace_distance),
                pinsker_bound=float(pi1.pinsker_bound),
                pinsker_ok=bool(pi1.pinsker_ok),
                rho_condition_number=float(pi1.rho_canon.condition_number),
                rho_purity=float(pi1.rho_canon.purity),
                rho_projection_cost=float(pi1.rho_canon.projection_cost),
                rho_kkt_ok=bool(pi1.rho_canon.kkt_ok),
                von_neumann_entropy_final=float(audit["entropy"]),
                dirac_beta=float(pi1.dirac.beta),
                dirac_gamma_lorentz=float(pi1.dirac.gamma_lorentz),
                dirac_cross_section_m2=float(pi1.dirac.cross_section_m2),
                dirac_cross_section_reduced=float(pi1.dirac.cross_section_reduced),
                gamow_sommerfeld_factor=float(pi1.dirac.gamow_factor),
                nr_gamow_to_exact_ratio=float(pi1.dirac.nr_to_exact_ratio),
                annihilation_rate_coefficient_m3s=float(pi1.dirac.rate_coefficient_m3s),
                annihilation_energy_lab_kev=float(pi1.dirac.energy_lab_kev),
                annihilation_energy_cm_kev=float(pi1.dirac.energy_cm_kev),
                poincare_canonicity_defect=float(pi2.chart.canonicity_defect),
                poincare_roundtrip_error=float(pi2.chart.roundtrip_error),
                poincare_mean_longitude=float(pi2.chart.mean_longitude),
                kepler_worst_residual=float(pi2.kepler.worst_residual),
                kepler_mean_iterations=float(pi2.kepler.mean_iterations),
                kepler_empirical_order=float(pi2.kepler.empirical_order),
                melnikov_flux=float(pi2.melnikov.flux),
                melnikov_transversal=bool(pi2.melnikov.transversal),
                melnikov_critical_forcing=float(pi2.melnikov.critical_forcing),
                melnikov_quadrature_error=float(pi2.melnikov.quadrature_error),
                monodromy_symplectic_defect=float(pi2.krein.symplectic_defect),
                monodromy_determinant=float(pi2.krein.determinant),
                monodromy_spectral_radius=float(pi2.krein.spectral_radius),
                krein_collision_detected=bool(pi2.krein.collision_detected),
                monodromy_step_certificate=float(pi2.krein.step_certificate),
                lindstedt_omega_series=float(pi2.lindstedt_omega_series),
                lindstedt_omega_exact=float(pi2.lindstedt_omega_exact),
                lindstedt_remainder_rel=float(pi2.lindstedt_remainder_rel),
                agm_iterations=int(pi2.agm_iterations),
                strang_observed_order=float(order_obs),
                strang_step_certificate=float(step_cert),
                isospectrality_defect=float(iso_total),
                simplex_kkt_ok=bool(simplex_kkt),
                replicator_variance=float(audit["variance"]),
                lyapunov_replicator_term=float(audit["replicator_term"]),
                gksl_entropy_production=float(ds_total),
                gksl_trace_defect=float(tr_total),
                gksl_cptp_ok=bool(cptp_ok),
                total_energy_derivative=float(audit["total_analytic"]),
                lyapunov_passivity_ok=passivity,
                derivative_discrepancy_rel=float(audit["discrepancy_rel"]),
                dk_sqrt_lipschitz=float(dk_sqrt),
                dk_invsqrt_lipschitz=float(dk_inv),
                brockett_min_rate=float(brockett_rate),
                condition_number_K_uncapped=float(kappa_raw),
                log10_condition_number_K=float(log_kappa),
                integration_steps=int(n_steps),
                integration_dt=float(h),
                engine_version=_ENGINE_VERSION,
                schema_flags=tuple(flags),
                uhlmann_pancharatnam_phase_rad=float(panch_phase),
                uhlmann_holonomy_unitarity_defect=float(hol_unit_def),
                uhlmann_quaternion_s3=quat_s3,
                fuchs_van_de_graaf_ok=bool(pi1.fuchs_sandwich_ok),
                fuchs_lower_margin=float(pi1.fuchs_lower_margin),
                fuchs_upper_margin=float(pi1.fuchs_upper_margin),
                tomita_modular_condition=float(pi1.tomita_modular_condition),
                wigner_yanase_skew_info=float(pi1.wigner_yanase_skew_info),
                fidelity_homotopy_drift=float(pi1.fidelity_homotopy_drift),
                ore_powell_3gamma_cross_section_m2=float(
                    pi1.dirac.ore_powell_3gamma_cross_section_m2
                ),
                poincare_cartan_symplectic_defect_4d=float(
                    pi2.chart.poincare_cartan_symplectic_defect_4d
                ),
                kam_diophantine_margin=float(pi2.chart.kam_diophantine_margin),
                kepler_fourier_bessel_error=float(
                    pi2.kepler.fourier_bessel_consistency_error
                ),
                melnikov_splitting_angle_rad=float(
                    pi2.melnikov.homoclinic_splitting_angle_rad
                ),
                yoshida4_monodromy_defect=float(pi2.krein.yoshida4_agreement_defect),
                poincare_rotation_number=float(pi2.krein.poincare_rotation_number),
                hamilton_hopf_index=float(pi2.krein.hamilton_hopf_index),
                rsi3_substeps_per_step=int(rsi3_substeps),
                bch_commutator_norm=float(bch_norm),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # §3 · INICIALIZACIÓN Y DELEGACIÓN PÚBLICA (API 100% compatible con v16/v17)
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        dimension_n: int,
        base_amber_duration_sec: float = 3600.0,
        decay_alpha: float = 1.5,
        *,
        stability_threshold: float = 1e-12,
        annihilation_reference_kev: float = _ANNIHILATION_ENERGY_KEV,
        epicyclic_precession_splitting: float = 0.0,
        enable_step_doubling: bool = True,
    ) -> None:
        if isinstance(dimension_n, bool) or not isinstance(
            dimension_n, (int, np.integer)
        ):
            raise ValueError("dimension_n debe ser un entero estrictamente positivo.")
        if int(dimension_n) <= 0:
            raise ValueError("La dimensión fundamental debe ser estrictamente positiva.")
        for name, val, allow_zero in (
            ("base_amber_duration_sec", base_amber_duration_sec, False),
            ("decay_alpha", decay_alpha, True),
            ("annihilation_reference_kev", annihilation_reference_kev, False),
        ):
            if (
                isinstance(val, bool)
                or not math.isfinite(float(val))
                or (float(val) < 0.0 if allow_zero else float(val) <= 0.0)
            ):
                raise ValueError(
                    f"{name} debe ser un float finito "
                    + ("no negativo." if allow_zero else "estrictamente positivo.")
                )
        if isinstance(stability_threshold, bool) or not math.isfinite(
            float(stability_threshold)
        ):
            raise ValueError("stability_threshold debe ser un float finito.")
        if not math.isfinite(float(epicyclic_precession_splitting)):
            raise ValueError("epicyclic_precession_splitting debe ser finito.")

        self._n: Final[int] = int(dimension_n)
        self._base_amber_sec: Final[float] = float(base_amber_duration_sec)
        self._decay_alpha: Final[float] = float(decay_alpha)
        self._stability_thresh: Final[float] = max(
            _STABILITY_FLOOR, float(stability_threshold)
        )
        if self._stability_thresh * self._n > 0.5:
            logger.warning(
                "stability_threshold·n = %.3e > 1/2: el piso se recortará a 1/(2n) para "
                "preservar la factibilidad de D_δ(H).",
                self._stability_thresh * self._n,
            )
        self._uniform: Final[NDArray[np.float64]] = np.full(
            self._n, 1.0 / float(self._n)
        )
        self._eye_c: Final[NDArray[np.complex128]] = np.eye(
            self._n, dtype=np.complex128
        )

        self.phase_pi1 = self.PhasePi1_DiracUhlmannAnnihilator(
            n=self._n,
            stability_threshold=self._stability_thresh,
            amber_base=self._base_amber_sec,
            amber_alpha=self._decay_alpha,
            amber_reference_kev=float(annihilation_reference_kev),
        )
        self.phase_pi2 = self.PhasePi2_PoincareCelestialRegularizer(
            precession_splitting=float(epicyclic_precession_splitting),
            enable_doubling=bool(enable_step_doubling),
        )
        self.phase_pi3 = self.PhasePi3_RSI3TrajectoryIntegrator(
            pi1=self.phase_pi1,
            pi2=self.phase_pi2,
            n=self._n,
            stability_threshold=self._stability_thresh,
        )
        logger.info(
            "PositronRedemptionGovernorEngine v%s inicializado (Π₁→Π₂→Π₃).",
            _ENGINE_VERSION,
        )

    # ── Delegadores de Π₁ ──
    def project_probability_simplex(
        self, v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Delega en 3.2b (proyección exacta sobre Δ^{n−1} con certificado KKT)."""
        return self.phase_pi3.project_probability_simplex(v)

    def canonicalize_density_matrix(
        self, rho: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Delega en 1.4b (proyección de Frobenius sobre D_δ(H) con meta-piso RSI³)."""
        return self.phase_pi1.canonicalize_density_matrix(rho)

    def compute_uhlmann_fidelity(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        """Delega en 1.7b (fidelidad de Uhlmann por norma traza + distancia de Bures)."""
        return self.phase_pi1.compute_uhlmann_fidelity(rho, sigma)

    def compute_umegaki_relative_entropy(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        """Delega en 1.8b (entropía relativa de Umegaki espectral + residuo de Pinsker)."""
        return self.phase_pi1.compute_umegaki_relative_entropy(rho, sigma)

    # ── Delegadores de Π₂ ──
    def solve_halley_kepler(
        self,
        mean_anomaly: float,
        eccentricity: float,
        max_iter: int = _HALLEY_MAX_ITER,
    ) -> Tuple[float, float]:
        """Delega en 2.4 (solver de Halley cúbico con salvaguarda de bisección)."""
        return self.phase_pi2.solve_halley_kepler(
            mean_anomaly, eccentricity, max_iter
        )

    def evaluate_poincare_canonical_chart(
        self,
        semi_major_axis: float,
        eccentricity: float,
        inclination_rad: float,
        mean_anomaly: float,
        arg_pericenter_rad: float = 0.0,
        lon_node_rad: float = 0.0,
        mu: float = 1.0,
    ) -> Tuple[NDArray[np.float64], float]:
        """Delega en 2.3b (carta regularizada de Poincaré)."""
        return self.phase_pi2.evaluate_poincare_canonical_chart(
            semi_major_axis,
            eccentricity,
            inclination_rad,
            mean_anomaly,
            arg_pericenter_rad,
            lon_node_rad,
            mu,
        )

    def compute_melnikov_turnstile_lobe_area(
        self,
        dissipation_gamma: float = 0.01,
        forcing_amplitude: float = 0.05,
        forcing_frequency: float = 1.0,
    ) -> float:
        """Delega en 2.6b (área del lóbulo del turnstile de MacKay-Meiss-Percival)."""
        return self.phase_pi2.compute_melnikov_turnstile_lobe_area(
            dissipation_gamma, forcing_amplitude, forcing_frequency
        )

    # ── Delegadores de Π₃ ──
    def purify_conductivity_matrix(
        self, K_raw: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.float64], float, float]:
        """Delega en 3.3b (proyección ortogonal sobre PSD ∩ Sym_n(ℝ))."""
        return self.phase_pi3.purify_conductivity_matrix(K_raw)

    def execute_positron_redemption_trajectory(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_target: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        dt: float = 0.05,
        steps: int = 4,
        **kwargs: Any,
    ) -> PositronRedemptionSpectrumReport:
        """Delega en 3.11 (composición monádica Π₁→Π₂→Π₃ y emisión del certificado)."""
        return self.phase_pi3.execute_positron_redemption_trajectory(
            p_init, rho_init, sigma_target, K_raw, dt, steps, **kwargs
        )


# ══════════════════════════════════════════════════════════════════════════════
# §4 · AUTOVERIFICACIÓN EJECUTABLE (identidades matemáticas y regresiones v16–v18)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """Suite de certificación numérica exhaustiva de las fases anidadas Π₁ → Π₂ → Π₃."""
    checks: List[Tuple[str, bool]] = []
    n = 4
    eng = PositronRedemptionGovernorEngine(n, stability_threshold=1e-8)
    pi1, pi2, pi3 = eng.phase_pi1, eng.phase_pi2, eng.phase_pi3

    # [P8] Proyección exacta sobre el símplex capado vs. clamp+renormalización ingenua.
    vals = np.array([-0.30, 0.10, 0.45, 0.75])
    exact, _d, feas = pi1._project_spectrum_capped_simplex(vals, 1e-8)
    clamped = np.maximum(vals, 1e-8)
    naive = clamped / clamped.sum()
    checks.append(
        (
            "[P8] Proyección exacta ≤ clamp+renorm en distancia de Frobenius",
            feas
            and float(np.linalg.norm(exact - vals))
            <= float(np.linalg.norm(naive - vals)) + 1e-12,
        )
    )
    checks.append(
        (
            "[P8] KKT del símplex capado (p≥δ, Σp=1)",
            abs(float(exact.sum()) - 1.0) < 1e-12 and float(exact.min()) >= 0.0,
        )
    )

    # [P4] Orden variacionalmente óptimo: real→clamp domina a clamp→real en Sym_n(ℝ)∩PSD.
    k_cplx = np.array(
        [
            [2.0 + 0j, 0.3 + 0.4j, 0, 0],
            [0.3 - 0.4j, 1.0, 0, 0],
            [0, 0, -0.5, 0.1],
            [0, 0, 0.1, 0.8],
        ],
        dtype=np.complex128,
    )
    k_opt, *_ = pi3._purify_conductivity_with_gaps(k_cplx)
    kh = 0.5 * (k_cplx + k_cplx.conj().T)
    ev, uv = la.eigh(kh)
    k_v16 = np.real((uv * np.maximum(ev, 1e-8)) @ uv.conj().T)
    checks.append(
        (
            "[P4] real→clamp domina a clamp→real (proyector de Frobenius)",
            la.norm(k_cplx - k_opt, "fro") <= la.norm(k_cplx - k_v16, "fro") + 1e-12,
        )
    )

    # [P1] Isoespectralidad exacta del doble corchete de Brockett.
    rho = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
    rho[0, 1], rho[1, 0] = 0.05j, -0.05j
    rho = pi1.canonicalize_density_matrix(rho)
    sigma = np.diag([0.25, 0.25, 0.25, 0.25]).astype(np.complex128)
    sigma = pi1.canonicalize_density_matrix(sigma)
    p_vec = np.array([0.4, 0.3, 0.2, 0.1])
    rho_b, iso = pi3._brockett_double_bracket_step(rho, p_vec, 0.05)
    checks.append(("[P1] Brockett isospectral ‖Δspec‖ < 1e-10", iso < 1e-10))
    checks.append(
        (
            "[P1] Entropía de von Neumann preservada por R₃",
            abs(pi3._entropy_of(rho_b) - pi3._entropy_of(rho)) < 1e-9,
        )
    )

    # Geometría de Bures-Uhlmann, Holonomía, Cuaterniones S³, Fuchs-van de Graaf y Wigner-Yanase.
    f_id, d_id = pi1.compute_uhlmann_fidelity(rho, rho)
    checks.append(
        ("Uhlmann F(ρ,ρ)=1 y d_B=0", abs(f_id - 1.0) < 1e-9 and d_id < 1e-9)
    )
    cert1 = pi1.audit_dirac_uhlmann_substrate(rho, sigma, positron_beta=0.01)
    checks.append(
        ("Identidad cuerda-arco d_B = 2·sin(θ_B/2)", cert1.chord_arc_error < 1e-12)
    )
    checks.append(("Pinsker D ≥ ½‖ρ−σ‖₁² (residuo ≥ 0)", cert1.pinsker_ok))
    checks.append(
        (
            "[v18] Sándwich de Fuchs-van de Graaf 1−√F ≤ D_tr ≤ √(1−F)",
            cert1.fuchs_sandwich_ok,
        )
    )
    checks.append(
        (
            "[v18] Holonomía de Uhlmann unitaria y cuaternión en S³ ⊂ ℍ",
            cert1.holonomy is not None
            and cert1.holonomy.unitarity_defect < 1e-12
            and cert1.holonomy.quaternion_norm_defect < 1e-12,
        )
    )
    checks.append(
        (
            "[v18] Información oblicua de Wigner-Yanase I_{1/2}(ρ, −ln σ) ≥ 0",
            cert1.wigner_yanase_skew_info >= 0.0,
        )
    )

    # [P10 + v18] Dirac 2γ, Ore-Powell 3γ y cinemática relativista.
    d_nr = pi1._dirac_annihilation_certificate(1e-4)
    d_ur = pi1._dirac_annihilation_certificate(0.999999)
    checks.append(
        (
            "[P10] Dirac NR: σ → π r_e²/β",
            abs(d_nr.cross_section_reduced * d_nr.beta - 1.0) < 1e-3,
        )
    )
    checks.append(
        (
            "[P10] Dirac UR: σ → (π r_e²/γ)(ln2γ−1)",
            abs(
                d_ur.cross_section_reduced
                - (math.log(2 * d_ur.gamma_lorentz) - 1) / d_ur.gamma_lorentz
            )
            < 1e-6,
        )
    )
    checks.append(
        (
            "[P10] √s(γ=1) = 2 m_e c²",
            abs(d_nr.energy_cm_kev - _ANNIHILATION_ENERGY_KEV) < 1e-3,
        )
    )
    checks.append(
        (
            "[v18] Canal orto-positronio 3γ de Ore-Powell positivo y consistente",
            d_nr.ore_powell_3gamma_cross_section_m2 > 0.0
            and abs(
                d_nr.ore_powell_3gamma_cross_section_m2
                / d_nr.nr_gamow_cross_section_m2
                - _ORE_POWELL_3GAMMA_RATIO
            )
            < 1e-12,
        )
    )

    # [P7 + v18] Kepler: residuo a posteriori e invariante integral de Kapteyn-Fourier-Bessel.
    _e_anom, res_k = pi2.solve_halley_kepler(0.05, 0.95)
    checks.append(("[P7] Kepler e=0.95: residuo a posteriori < 1e-12", res_k < 1e-12))
    kep_torus = pi2._audit_kepler_torus(0.25)
    checks.append(
        (
            "[v18] Invariante integral de Kapteyn-Fourier-Bessel en el toro < 1e-9",
            kep_torus.fourier_bessel_consistency_error < 1e-9,
        )
    )

    # [P3 + v18] Carta de Poincaré e invariante simpléctico 4D de Poincaré-Cartan.
    chart = pi2._chart_with_certificate(1.0, 0.2, 0.3, 0.7, 0.4, 0.2)
    checks.append(
        ("[P3] Round-trip de la carta < 1e-9", chart.roundtrip_error < 1e-9)
    )
    checks.append(
        ("[P3] |det ∂(Γ,ϖ)/∂(ξ,η)| = 1", chart.canonicity_defect < 1e-5)
    )
    checks.append(
        (
            "[v18] Invariante 4D de Poincaré-Cartan ‖J₄ᵀ Ω₄ J₄ − Ω₄‖_F < 1e-6",
            chart.poincare_cartan_symplectic_defect_4d < 1e-6,
        )
    )

    # Melnikov: forma cerrada, cuadratura y ángulo de escisión homoclínica.
    mel = pi2._melnikov_certificate(0.01, 0.05, 1.0)
    checks.append(
        (
            "Melnikov: área cerrada ≡ cuadratura del lóbulo y θ_split > 0",
            mel.lobe_area > 0.0
            and mel.quadrature_error < 1e-8
            and mel.homoclinic_splitting_angle_rad > 0.0,
        )
    )
    checks.append(
        (
            "Melnikov: B≤A ⟹ sin turnstile",
            pi2._melnikov_certificate(1.0, 0.001, 1.0).lobe_area == 0.0,
        )
    )

    # [P2 + v18] Monodromía simpléctica, Yoshida de 4.º orden y colisión de Krein 1:1.
    krein = pi2._epicyclic_krein_monodromy(1.0, 0.01, 0.02)
    checks.append(("[P2] ‖MᵀΩM−Ω‖_F < 1e-10", krein.symplectic_defect < 1e-10))
    checks.append(("[P2] det M = +1", abs(krein.determinant - 1.0) < 1e-9))
    checks.append(
        ("[P2] Kepler κ=ν ⟹ colisión de Krein 1:1 detectada", krein.collision_detected)
    )
    checks.append(
        (
            "[v18] Concordancia entre propagador exacto y Yoshida de 4.º orden < 1e-4",
            krein.yoshida4_agreement_defect < 1e-4,
        )
    )

    # [P16] Lindstedt-Poincaré vs. módulo elíptico exacto de Jacobi-AGM.
    w_s, w_x, rem, _it = pi2._lindstedt_agm_resonance(1.0, 0.5, 0.3)
    checks.append(
        (
            "[P16] Lindstedt ≡ AGM elíptico exacto (resto < 1e-4)",
            rem < 1e-4 and w_s > 0.0 and w_x > 0.0,
        )
    )

    # [P5] Cotas de Daleckii-Krein correctamente atribuidas.
    l_sq, l_inv = pi3._connes_daleckii_krein_bounds(1e-4)
    checks.append(
        (
            "[P5] L_sqrt = 1/(2√λ) y L_invsqrt = 1/(2λ^{3/2})",
            abs(l_sq - 0.5 / math.sqrt(1e-4)) < 1e-9
            and abs(l_inv - 0.5 / (1e-4 ** 1.5)) < 1e-3
            and l_inv > l_sq,
        )
    )

    # Trayectoria completa Π₁ → Π₂ → Π₃: orden 2 de Strang, pasividad y buffers congelados.
    k_raw = np.array(
        [
            [1.0, 0.2, 0, 0],
            [0.2, 0.9, 0.1, 0],
            [0, 0.1, 0.8, 0.05],
            [0, 0, 0.05, 0.7],
        ],
        dtype=np.complex128,
    )
    rep = eng.execute_positron_redemption_trajectory(
        p_init=np.array([0.25, 0.25, 0.25, 0.25]),
        rho_init=rho,
        sigma_target=sigma,
        K_raw=k_raw,
        dt=0.02,
        steps=6,
    )
    checks.append(
        ("[P13] Orden observado de Strang ≈ 2", 1.7 <= rep.strang_observed_order <= 2.4)
    )
    checks.append(
        ("[P1] Isoespectralidad acumulada < 1e-9", rep.isospectrality_defect < 1e-9)
    )
    checks.append(
        (
            "Axioma RSI3-1 en el sector unitario: dH/dt|_rep ≤ 0",
            rep.lyapunov_passivity_ok and rep.analytic_energy_derivative <= 0.0,
        )
    )
    checks.append(
        (
            "[P12] Derivada instantánea ≡ analítica (disp. < 5%)",
            rep.derivative_discrepancy_rel < 0.05,
        )
    )
    checks.append(("[P6] Gap espectral real λ_max−λ₂ ≥ 0", rep.spectral_gap_K >= 0.0))
    checks.append(
        (
            "[P14] Arrays del certificado son de sólo lectura",
            not rep.p_final.flags.writeable and not rep.rho_final.flags.writeable,
        )
    )
    checks.append(
        (
            "[v18] Sub-stepping RSI³ y conmutador BCH certificados",
            rep.rsi3_substeps_per_step in (1, 2, 4) and rep.bch_commutator_norm >= 0.0,
        )
    )

    rep_d = eng.execute_positron_redemption_trajectory(
        p_init=np.array([0.25, 0.25, 0.25, 0.25]),
        rho_init=rho,
        sigma_target=sigma,
        K_raw=k_raw,
        dt=0.02,
        steps=6,
        lindblad_rate=0.5,
    )
    checks.append(
        (
            "GKSL: CPTP, traza conservada y ΔS ≥ 0",
            rep_d.gksl_cptp_ok
            and rep_d.gksl_trace_defect < 1e-10
            and rep_d.gksl_entropy_production >= -1e-9,
        )
    )
    checks.append(
        (
            "GKSL activo ⟹ pasividad NO declarada (honestidad de signo)",
            not rep_d.lyapunov_passivity_ok,
        )
    )

    ok = True
    for name, passed in checks:
        ok = ok and bool(passed)
        print(f"[{'OK ' if passed else 'FAIL'}] {name}")
    print(
        f"\nVEREDICTO GLOBAL DE AUTOVERIFICACIÓN: {'COHERENTE' if ok else 'OBSTRUIDO'}"
    )
    return ok


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s · %(name)s · %(message)s"
    )
    raise SystemExit(0 if _self_test() else 1)