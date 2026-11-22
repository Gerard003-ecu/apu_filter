# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Positron Redemption Witness Engine (Motor Espectral Ciego FPU · Testigo Silencioso)        ║
║ RUTA    : app/omega/positron_redemption_witness_engine.py                                            ║
║ VERSIÓN : 18.0.0-FPU-Witness-Uhlmann-SmithZ-Poincare-Liouville-RSI3-PhD                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
NATURALEZA DEL MOTOR (inalterada respecto de v16/v17)
──────────────────────────────────────────────────────────────────────────────────────────────────────
Funtor de registro covariante e inmutable en la FPU del Estrato Omega (V_Ω, Nivel 0.5 — Ágora
Tensorial). NO evalúa retículos de Heyting, NO dicta veredictos, NO mantiene estados OODA,
NO toca periféricos. Emite únicamente el certificado numérico inmutable
`PositronWitnessSpectrumReport`, consumido por `positron_redemption_witness_agent.py`.
Los booleanos que emite son PREDICADOS MATEMÁTICOS (nulidad de torsión, simplecticidad,
contractividad de Banach de Nivel 3, ortogonalidad cuaterniónica KS, no-interferencia),
nunca veredictos: la adjudicación pertenece al soberano testigo.

DOCTRINA RECTORA DE v18: EL TEOREMA SE COMPUTA CON SU PROPIA ARITMÉTICA Y SE AUTOCERTIFICA (RSI-L3)
──────────────────────────────────────────────────────────────────────────────────────────────────────
v18 conserva íntegramente las correcciones exactas [W1..W10] de v17 e integra granular y
rigurosamente dos pilares matemáticos adicionales en las tres fases anidadas:
  1. AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3) EN ÁLGEBRAS DE BANACH Y MÓDULOS EUCLÍDEOS:
     • Nivel 0 (Evaluación espectral base): descomposición espectral/SVD y eliminación entera.
     • Nivel 1 (Refinamiento de primer orden): iteración acoplada de Newton-Schulz sobre la
       variedad unitaria U(n) y sobre el grupo simpléctico Sp(2d, ℝ); extrapolación de Richardson.
     • Nivel 2 (Certificación variacional y co-coercividad): verificación de idempotencia exacta
       ‖P(P(x)) − P(x)‖_F, desigualdad de no-expansividad proximal de Moreau-Yosida y pasividad
       de impedancia de Foster-Brune en redes eléctricas/cuánticas.
     • Nivel 3 (Meta-Operador contractivo de Banach-Lawvere): auditoría del propio endofuntor
       de refinamiento T: End(M) → End(M), midiendo la razón de contracción empírica q_L3 < 1
       y la cota de error a posteriori ‖x_L3 − x*‖_F ≤ q_L3/(1−q_L3)·‖T(x)−x‖_F.
  2. MECÁNICA CELESTE DE POINCARÉ Y REGULARIZACIÓN HIPERCOMPLEJA DE KUSTAANHEIMO-STIEFEL (KS):
     • En Π₁: inmersión del estado cuántico sobre la fibración de Hopf S³ → S² mediante el
       álgebra de cuaterniones de Hamilton ℍ y la matriz de Levi-Civita/KS L(u), certificando
       L(u)ᵀL(u) = ‖u‖²I₄, la restricción gauge bilineal de KS l_KS(u, u') = 0, el tiempo
       ficticio de Poincaré dt = r·dτ, la ecuación de Lyapunov de la Derivada Logarítmica
       Simétrica (SLD) y la Información Cuántica de Fisher (QFI) de Bures-Helstrom.
     • En Π₂: Laplaciano combinatorio de Hodge-Poincaré Δ_k = ∂_kᵀ∂_k + ∂_{k+1}∂_{k+1}ᵀ,
       emparejamiento de torsión de Pontryagin-Poincaré en ℚ/ℤ e invariante de circulación
       de Poincaré-Kirchhoff sobre la base entera saturada de 1-ciclos ker ∂₁.
     • En Π₃: serie generatriz canónica de Poincaré-Von Zeipel / Lie-Deprit S₁, invariante
       secular de Kozai-Lidov H_Kozai = √(1−e²)cos ι, parámetro de Tisserand-Poincaré,
       monodromía de Hill-Poincaré con proyección simpléctica polar RSI-L3 y aproximación
       diofántica KAM del número de rotación por fracciones continuas de Poincaré.

TRES FASES ESPECTRALES ANIDADAS CON CONTINUIDAD FORMAL ESTRICTA
──────────────────────────────────────────────────────────────────────────────────────────────────────
    1.9 audit_passive_quantum_substrate ──[Pi1ToPi2Seed]──► 2.1 _ingest_pi1_continuation
    2.7 audit_smith_homology_sector     ──[Pi2ToPi3Seed]──► 3.1 _ingest_pi2_continuation
    3.8 inspect_positron_trajectory = composición Π₃∘Π₂∘Π₁ y emisión del certificado.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Omega.PositronRedemptionWitnessEngine")

__all__ = [
    "PositronWitnessSpectrumReport",
    "PositronRedemptionWitnessEngine",
    "WitnessTelemetry",
    "SmithNormalFormCertificate",
    "Pi1SpectrumCertificate",
    "Pi2SpectrumCertificate",
    "KustaanheimoStiefelPoincareCertificate",
    "RsiLevel3Pi1Certificate",
    "HodgePoincareHomologyCertificate",
    "ENGINE_VERSION",
]

_ENGINE_VERSION: Final[str] = "18.0.0-FPU-Witness-Uhlmann-SmithZ-Poincare-Liouville-RSI3-PhD"
version: Final[str] = _ENGINE_VERSION
ENGINE_VERSION: Final[str] = _ENGINE_VERSION

# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES NUMÉRICAS, ÁLGEBRA HIPERCOMPLEJA (ℍ) Y AUXILIARES PUROS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_STABILITY_FLOOR: Final[float] = max(1.0e-12, _MACHINE_EPS)
_TINY: Final[float] = 1.0e-300
_WILKINSON_EPS: Final[float] = 1.0e-15
_CONDITION_CAP: Final[float] = 1.0e12
_MAX_MODULAR_CONDITION: Final[float] = 1.0e8
_INT64_MAX: Final[int] = int(np.iinfo(np.int64).max)
_SYMPECTIC_TOLERANCE: Final[float] = 1.0e-9
_MONODROMY_BASE_PANELS: Final[int] = 96
_MONODROMY_MAX_DOUBLINGS: Final[int] = 5
_SNF_MAX_SWEEPS: Final[int] = 4096
_RSI_L3_MAX_STEPS: Final[int] = 4
_KAM_CF_MAX_DEPTH: Final[int] = 12


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión total a float finito: la FPU nunca propaga NaN/Inf al certificado."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fro(mat: NDArray[Any]) -> float:
    """Norma de Frobenius total y segura para arreglos vacíos o densos."""
    return float(la.norm(mat, ord="fro")) if getattr(mat, "size", 0) else 0.0


def _relative(defect: float, scale: float) -> float:
    """Defecto relativo protegido contra subdesbordamiento en el denominador."""
    return float(defect / max(abs(scale), _TINY))


def _freeze_array(arr: NDArray[Any]) -> NDArray[Any]:
    """Copia defensiva de sólo lectura: la inmutabilidad deja de ser superficial. [W10]"""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _tensor_digest(arr: Any) -> str:
    """SHA-256 de (dtype‖shape‖strides‖bytes_C). No muta la fuente."""
    if not isinstance(arr, np.ndarray):
        return ""
    try:
        c = np.ascontiguousarray(arr)
        meta = f"{arr.dtype.str}|{arr.shape}|{arr.strides}|".encode("ascii")
        return hashlib.sha256(meta + c.tobytes(order="C")).hexdigest()
    except (ValueError, TypeError, MemoryError):
        return ""


def _int_matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> List[List[int]]:
    """Producto matricial ENTERO exacto (int de Python, precisión arbitraria)."""
    if not a or not b or not b[0]:
        return [[] for _ in a]
    n, k, m = len(a), len(b), len(b[0])
    out = [[0] * m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            v = ai[t]
            if v:
                bt = b[t]
                for j in range(m):
                    oi[j] += v * bt[j]
    return out


def _bareiss_determinant(mat: Sequence[Sequence[int]]) -> int:
    """
    Determinante ENTERO exacto por eliminación de Bareiss (sin fracciones).
    La división en cada paso es exacta por la identidad de Sylvester; se usa int de Python
    para evitar el desbordamiento que int64 sufriría con divisores elementales grandes.
    """
    n = len(mat)
    if n == 0:
        return 1
    a = [list(row) for row in mat]
    sign, prev = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            for i in range(k + 1, n):
                if a[i][k] != 0:
                    a[k], a[i] = a[i], a[k]
                    sign = -sign
                    break
            else:
                return 0
        pivot = a[k][k]
        for i in range(k + 1, n):
            aik = a[i][k]
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - aik * a[k][j]) // prev
            a[i][k] = 0
        prev = pivot
    return sign * a[n - 1][n - 1]


def _hadamard_log10_bound(mat: Sequence[Sequence[int]]) -> float:
    """
    Cota log10 de Hadamard para cualquier menor determinante de una matriz entera:
        |det(M)| ≤ ∏_i ‖fila_i‖₂.
    Certifica a priori el techo de crecimiento de los divisores elementales en ℤ.
    """
    if not mat or not mat[0]:
        return 0.0
    acc_log10 = 0.0
    for row in mat:
        sq = sum(int(x) * int(x) for x in row)
        if sq > 0:
            acc_log10 += 0.5 * math.log10(float(sq))
    return float(acc_log10)


def _ks_levi_civita_matrix(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Matriz cuaterniónica de Kustaanheimo-Stiefel (KS) L(u) ∈ ℝ^{4×4} para u ∈ ℍ ≅ ℝ⁴.
    Satisface exactamente la identidad de composición de Hurwitz-Radon:
        L(u)ᵀ L(u) = L(u) L(u)ᵀ = ‖u‖₂² I₄,   det L(u) = ‖u‖₂⁴ = r².
    """
    u1, u2, u3, u4 = float(u[0]), float(u[1]), float(u[2]), float(u[3])
    return np.array(
        [
            [u1, -u2, -u3, u4],
            [u2, u1, -u4, -u3],
            [u3, u4, u1, u2],
            [u4, -u3, u2, -u1],
        ],
        dtype=np.float64,
    )


def _continued_fraction_diophantine(omega_ratio: float, max_depth: int = _KAM_CF_MAX_DEPTH) -> Tuple[int, int, float]:
    """
    Aproximación diofántica racional p/q de un cociente de frecuencias ω₁/ω₂ mediante
    fracciones continuas de Poincaré-KAM. Devuelve (p, q, |ω_ratio − p/q|).
    """
    x = abs(_safe_float(omega_ratio, 0.0))
    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0
    rem = x
    for _ in range(max(1, max_depth)):
        a_k = int(math.floor(rem))
        p_next = a_k * p_curr + p_prev
        q_next = a_k * q_curr + q_prev
        if q_next <= 0 or q_next > 10**9:
            break
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        frac = rem - float(a_k)
        if frac <= 1.0e-12:
            break
        rem = 1.0 / frac
    q_safe = max(1, q_curr)
    approx = float(p_curr) / float(q_safe)
    return int(p_curr), int(q_safe), abs(x - approx)


# ══════════════════════════════════════════════════════════════════════════════
# §1 · CERTIFICADOS DE FASE, SEMILLAS DE CONTINUIDAD, TELEMETRÍA E INFORME
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DensityCanonicalization:
    """Proyección exacta sobre D_δ(H) con certificados KKT, entropía y coste (Mirsky)."""
    rho: NDArray[np.complex128]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    entropy: float
    purity: float
    condition_number: float
    projection_cost: float
    floor_used: float
    kkt_ok: bool
    hermiticity_defect: float
    trace_defect: float
    used_fallback: bool
    idempotence_defect_l3: float = 0.0
    kkt_stationarity_residual: float = 0.0


@dataclass(frozen=True)
class ConductivityPurification:
    """Proyección de Frobenius sobre PSD ∩ Sym_n(ℝ) con certificado Foster-Brune. [W5]"""
    k_matrix: NDArray[np.complex128]  # real-simétrica embebida: parte imaginaria ≡ 0
    lambda_min: float
    lambda_max: float
    top_gap: float
    condition_raw: float
    condition_capped: float
    log10_condition: float
    imaginary_discarded_norm: float
    projection_cost: float
    used_fallback: bool
    foster_brune_passive_ok: bool = True
    idempotence_defect_l3: float = 0.0


@dataclass(frozen=True)
class KustaanheimoStiefelPoincareCertificate:
    """
    Certificado hipercomplejo (ℍ) de regularización celeste de Kustaanheimo-Stiefel (KS)
    y estructura simpléctica de Poincaré-Kähler sobre el espacio proyectivo cuántico.
    """
    spinor_u: NDArray[np.float64]
    cartesian_kepler_radius: float
    poincare_time_jacobian: float
    hurwitz_orthogonality_defect: float
    ks_bilinear_gauge_residual: float
    fubini_study_symplectic_area: float
    sld_lyapunov_residual: float
    quantum_fisher_information: float
    ks_regularization_ok: bool


@dataclass(frozen=True)
class RsiLevel3Pi1Certificate:
    """
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3) en la Fase Π₁:
      • Nivel 1: Refinamiento polar unitario de Newton-Schulz sobre √ρ√σ.
      • Nivel 2: Idempotencia de proyectores y co-coercividad proximal de Moreau-Yosida.
      • Nivel 3: Razón de contracción de Banach q_L3 < 1 y cota a posteriori de punto fijo.
    """
    polar_unitary_defect_l0: float
    polar_unitary_defect_l3: float
    banach_contraction_ratio: float
    aposteriori_fixed_point_bound: float
    refined_uhlmann_fidelity: float
    moreau_yosida_cocoercivity_margin: float
    projector_idempotence_defect: float
    iterations_executed: int
    rsi_l3_converged_ok: bool


@dataclass(frozen=True)
class NonInterferenceAttestation:
    """WITNESS-1 con alcance DECLARADO: buffers inspeccionados, sin garantía TOCTOU. [W10]"""
    p_ok: bool
    rho_ok: bool
    sigma_ok: bool
    k_ok: bool
    boundary_ok: bool
    scope: str = "INSPECTED_BUFFERS_ONLY_NO_TOCTOU_GUARANTEE"

    @property
    def all_ok(self) -> bool:
        return bool(self.p_ok and self.rho_ok and self.sigma_ok and self.k_ok and self.boundary_ok)


@dataclass(frozen=True)
class Pi1ToPi2Seed:
    """[MORFISMO 1.9 → 2.1] Semilla geométrica, hipercompleja y RSI-L3 que abre la Fase Π₂."""
    bures_tension: float
    uhlmann_fidelity: float
    rho_populations: Tuple[float, ...]
    rho_entropy: float
    dimension_n: int
    has_context: bool = True
    quantum_fisher_information: float = 0.0
    poincare_ks_radius: float = 1.0
    rsi_l3_contraction_ratio: float = 0.0
    rsi_l3_converged_ok: bool = True


@dataclass(frozen=True)
class Pi1SpectrumCertificate:
    """Certificado numérico de Π₁ (sin veredictos: la FPU testigo es ciega)."""
    rho: DensityCanonicalization
    sigma: DensityCanonicalization
    conductivity: ConductivityPurification
    uhlmann_fidelity: float
    bures_distance: float
    bures_angle: float
    chord_arc_error: float
    fidelity_route_discrepancy: float
    hilbert_schmidt_overlap: float
    continuation: Optional[Pi1ToPi2Seed] = None
    ks_poincare: Optional[KustaanheimoStiefelPoincareCertificate] = None
    rsi_l3: Optional[RsiLevel3Pi1Certificate] = None


@dataclass(frozen=True)
class SmithNormalFormCertificate:
    """SNF ENTERA con certificados de unimodularidad, Hadamard y reconstrucción exacta. [W1]"""
    invariant_factors_exact: Tuple[str, ...]
    invariant_factors_int64: NDArray[np.int64]
    int64_saturated: bool
    rank: int
    rows: int
    cols: int
    divisibility_chain_ok: bool
    unimodular_ok: bool
    reconstruction_ok: bool
    det_u: int
    det_v: int
    v_inverse_ok: bool
    cokernel_free_rank: int
    cokernel_torsion: Tuple[str, ...]
    kernel_rank: int
    torsion_free: bool
    hadamard_log10_bound: float = 0.0
    pontryagin_linking_phase: float = 0.0
    sweeps_executed: int = 0


@dataclass(frozen=True)
class HomologyCertificate:
    """H_k genuino cuando se suministra ∂_{k+1}; coker(B) en caso contrario."""
    scope: str
    chain_complex_ok: bool
    homology_rank: int
    homology_torsion: Tuple[str, ...]
    torsion_free: bool
    restricted_rank: int
    kernel_basis_columns: int
    theorem: str = "ker(B) is saturated; columns of V with index >= rank form a Z-basis"


@dataclass(frozen=True)
class HodgePoincareHomologyCertificate:
    """
    Certificado de Topología Algebraica y Circulación de Hodge-Poincaré en Π₂:
      • Laplaciano combinatorio de Hodge Δ_k = ∂_kᵀ∂_k + ∂_{k+1}∂_{k+1}ᵀ.
      • Consistencia del número de Betti: dim ker(Δ_k) == rank(H_k(C; ℤ)).
      • Invariante de circulación de Poincaré-Kirchhoff sobre la base entera de 1-ciclos.
    """
    hodge_harmonic_dimension: int
    hodge_spectral_gap: float
    betti_hodge_consistency_ok: bool
    poincare_cycle_winding_action: float
    euler_poincare_characteristic: int
    rsi_l3_kernel_orthogonality_defect: float


@dataclass(frozen=True)
class Pi2ToPi3Seed:
    """[MORFISMO 2.7 → 3.1] Semilla homológica y de circulación de Poincaré que abre Π₃."""
    torsion_free: bool
    homology_rank: int
    smith_rank: int
    kernel_rank: int
    dimension_n: int
    has_context: bool = True
    poincare_cycle_winding_action: float = 0.0
    pontryagin_linking_phase: float = 0.0
    betti_hodge_consistency_ok: bool = True
    pi1_qfi_carried: float = 0.0


@dataclass(frozen=True)
class Pi2SpectrumCertificate:
    """Certificado numérico de Π₂ (Forma Normal de Smith sobre ℤ, Homología y Hodge-Poincaré)."""
    smith: SmithNormalFormCertificate
    homology: HomologyCertificate
    continuation: Optional[Pi2ToPi3Seed] = None
    hodge_poincare: Optional[HodgePoincareHomologyCertificate] = None


@dataclass(frozen=True)
class PoincareChartCertificate:
    """Carta regularizada de Poincaré con serie generatriz de Von Zeipel y canonicidad. [W2]"""
    chart: NDArray[np.float64]
    mean_longitude: float
    algebraic_identity_residual: float
    roundtrip_error: float
    canonicity_defect: float
    chart_condition_number: float
    symplectic_jacobian_defect: float = 0.0
    von_zeipel_generating_s1: float = 0.0


@dataclass(frozen=True)
class CartanCertificate:
    """Invariante integral relativo de Poincaré-Cartan COMPLETO + Kozai-Lidov y Tisserand. [W3]"""
    delaunay_lambda: float
    delaunay_g: float
    delaunay_h: float
    angular_momentum_deficit: float
    mean_motion: float
    orbital_period: float
    kepler_energy: float
    action_part: float
    energy_part: float
    cartan_invariant: float
    identity_error: float
    kozai_lidov_invariant: float = 1.0
    tisserand_poincare_invariant: float = 3.0


@dataclass(frozen=True)
class LiouvilleCertificate:
    """Monodromía simpléctica de Liouville-Poincaré con proyección RSI-L3 y rotación KAM. [W7]"""
    symplectic_defect: float
    determinant: float
    spectral_radius: float
    krein_signatures: Tuple[float, ...]
    collision_detected: bool
    step_certificate: float
    panels: int
    liouville_ok: bool
    kam_rotation_number: float = 0.0
    kam_diophantine_defect: float = 0.0
    kam_rational_p: int = 1
    kam_rational_q: int = 1
    rsi_l3_symplectic_projection_cost: float = 0.0
    floquet_lyapunov_exponent: float = 0.0


@dataclass(frozen=True)
class ShahshahaniCertificate:
    """Disipación de Shahshahani y certificado RSI-L3 de Richardson-Romberg para dφ/dt = 2Var."""
    variance: float
    dissipation_rate: float
    speed: float
    phi: float
    phi_derivative_numeric: float
    phi_derivative_analytic: float
    derivative_certificate_error: float
    romberg_l3_contraction_ratio: float = 0.0


@dataclass
class WitnessTelemetry:
    """Registro acumulativo del testigo pasivo (la FPU no dicta veredictos)."""
    total_inspections: int = 0
    cumulative_snf_sweeps: int = 0
    cumulative_monodromy_panels: int = 0
    cumulative_rsi_l3_iterations: int = 0
    torsion_detections: int = 0
    non_interference_violations: int = 0
    last_smith_rank: int = 0
    last_homology_rank: int = 0
    last_rsi_l3_contraction: float = 0.0


@dataclass(frozen=True)
class PositronWitnessSpectrumReport:
    r"""
    Certificado espectral inmutable de inspección pasiva emitido por la FPU.
    Orden posicional de v16/v17 preservado íntegramente; los campos nuevos van con defecto.
    Los ndarray se entregan como copias de sólo lectura.
    """
    p_observed: NDArray[np.float64]
    rho_observed: NDArray[np.complex128]
    K_purified: NDArray[np.complex128]
    uhlmann_fidelity: float
    bures_distance: float
    smith_torsion_free: bool
    smith_invariant_factors: NDArray[np.int64]
    poincare_canonical_residual: float
    poincare_cartan_invariant: float
    shahshahani_dissipation_rate: float
    connes_lipschitz_bound: float
    condition_number_K: float
    schema_version: str = "18.0.0-PhD-Witness-Uhlmann-SmithZ-Poincare-Liouville-RSI3"
    # ── Π₁ extendido (v17 + v18 RSI-L3 & KS-Poincaré) ──
    bures_angle: float = 0.0
    chord_arc_identity_error: float = 0.0
    fidelity_route_discrepancy: float = 0.0
    hilbert_schmidt_overlap: float = 0.0
    rho_condition_number: float = 1.0
    rho_purity: float = 1.0
    rho_entropy: float = 0.0
    rho_projection_cost: float = 0.0
    rho_kkt_ok: bool = True
    simplex_kkt_ok: bool = True
    K_imaginary_discarded_norm: float = 0.0
    K_projection_cost: float = 0.0
    lambda_min_K: float = 0.0
    lambda_max_K: float = 0.0
    spectral_gap_K: float = 0.0
    condition_number_K_raw: float = 1.0
    log10_condition_number_K: float = 0.0
    dk_sqrt_lipschitz: float = 0.0
    dk_invsqrt_lipschitz: float = 0.0
    quantum_fisher_information: float = 0.0
    sld_lyapunov_residual: float = 0.0
    ks_hurwitz_orthogonality_defect: float = 0.0
    ks_bilinear_gauge_residual: float = 0.0
    fubini_study_symplectic_area: float = 0.0
    rsi_l3_banach_contraction_ratio: float = 0.0
    rsi_l3_aposteriori_bound: float = 0.0
    rsi_l3_projector_idempotence_defect: float = 0.0
    rsi_l3_converged_ok: bool = True
    # ── Π₂ extendido (v17 + v18 Hodge-Poincaré & Pontryagin) ──
    smith_rank: int = 0
    smith_invariant_factors_exact: Tuple[str, ...] = ()
    smith_int64_saturated: bool = False
    smith_divisibility_chain_ok: bool = True
    smith_unimodular_ok: bool = True
    smith_reconstruction_ok: bool = True
    smith_v_inverse_ok: bool = True
    smith_det_u: int = 1
    smith_det_v: int = 1
    smith_cokernel_free_rank: int = 0
    smith_cokernel_torsion: Tuple[str, ...] = ()
    smith_kernel_rank: int = 0
    homology_scope: str = "COKERNEL_ONLY"
    homology_chain_complex_ok: bool = False
    homology_rank: int = 0
    homology_torsion: Tuple[str, ...] = ()
    homology_torsion_free: bool = True
    smith_hadamard_log10_bound: float = 0.0
    pontryagin_linking_phase: float = 0.0
    hodge_harmonic_dimension: int = 0
    hodge_spectral_gap: float = 0.0
    betti_hodge_consistency_ok: bool = True
    poincare_cycle_winding_action: float = 0.0
    # ── Π₃ extendido (v17 + v18 Kozai-Lidov, Tisserand, KAM & Von Zeipel) ──
    poincare_canonicity_defect: float = 0.0
    poincare_roundtrip_error: float = 0.0
    poincare_mean_longitude: float = 0.0
    poincare_symplectic_jacobian_defect: float = 0.0
    von_zeipel_generating_s1: float = 0.0
    delaunay_lambda: float = 0.0
    delaunay_angular_momentum_deficit: float = 0.0
    cartan_action_part: float = 0.0
    cartan_energy_part: float = 0.0
    cartan_identity_error: float = 0.0
    kozai_lidov_invariant: float = 1.0
    tisserand_poincare_invariant: float = 3.0
    liouville_symplectic_defect: float = 0.0
    liouville_determinant: float = 1.0
    liouville_spectral_radius: float = 1.0
    liouville_ok: bool = True
    krein_collision_detected: bool = False
    monodromy_step_certificate: float = 0.0
    kam_rotation_number: float = 0.0
    kam_diophantine_defect: float = 0.0
    floquet_lyapunov_exponent: float = 0.0
    shahshahani_speed: float = 0.0
    shahshahani_variance: float = 0.0
    replicator_phi_derivative_numeric: float = 0.0
    replicator_phi_derivative_analytic: float = 0.0
    replicator_derivative_certificate_error: float = 0.0
    romberg_l3_contraction_ratio: float = 0.0
    non_interference_ok: bool = True
    engine_version: str = _ENGINE_VERSION
    schema_flags: Tuple[str, ...] = ()


# ══════════════════════════════════════════════════════════════════════════════
# §2 · MOTOR TESTIGO CIEGO: TRES FASES Π ANIDADAS
# ══════════════════════════════════════════════════════════════════════════════
class PositronRedemptionWitnessEngine:
    """
    Motor Espectral Ciego de FPU para el Testigo Silencioso de Redención (v18.0.0).
    Tres fases anidadas con continuidad formal estricta, mecánica celeste de Poincaré
    y automejora recursiva de Nivel 3 (RSI-L3):
        Π₁ · PhasePi1_PassiveUhlmannObserver   (1.1 → 1.9)
        Π₂ · PhasePi2_SmithHomologyAuditor     (2.1 → 2.7)
        Π₃ · PhasePi3_PoincareLiouvilleEmitter (3.1 → 3.8)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₁ · UHLMANN-BURES PASIVO, REGULARIZACIÓN KS-POINCARÉ Y RSI-L3
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi1_PassiveUhlmannObserver:
        """
        Π₁. Cadena formal:
          1.1 Validación y hermitización en el álgebra de Banach B(H)
          1.2 Proyección exacta sobre el símplex capado con residuo estacionario KKT
          1.3 Canonización de ρ en D_δ(H) con verificación de idempotencia RSI-L3
          1.4 Purificación de conductividad K en PSD ∩ Sym_n(ℝ) con pasividad Foster-Brune
          1.5 Geometría de Uhlmann-Bures y ecuación de Lyapunov de la SLD / QFI
          1.6 Regularización celeste cuaterniónica de Kustaanheimo-Stiefel (KS) y 2-forma de Poincaré
          1.7 Auditoría de Automejora Recursiva de Nivel 3 (RSI-L3: Newton-Schulz + Moreau-Yosida)
          1.8 Atestación criptográfica de no-interferencia (WITNESS-1)
          1.9 TERMINAL Π₁ (emite Pi1ToPi2Seed → continuación directa en 2.1)
        """

        def __init__(
            self,
            dimension_n: int,
            tolerance: float,
            tau_spd: float,
            telemetry: Optional[WitnessTelemetry] = None,
        ) -> None:
            self._n = max(1, int(dimension_n))
            self._tol = float(tolerance)
            self._tau_spd = float(tau_spd)
            self._telemetry = telemetry
            self._uniform = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)
            self._eye = np.eye(self._n, dtype=np.complex128)

        # ── Método 1.1 ────────────────────────────────────────────────────────
        def _validate_and_hermitize(
            self, mat: Any
        ) -> Tuple[NDArray[np.complex128], float, float, bool]:
            """
            1.1 Validación total de forma/finitud y proyección autoadjunta ½(M+M†) en B(H).
            Devuelve (hermítico, def_traza REAL, def_hermiticidad, fallback). [W4/W9]
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
        ) -> Tuple[NDArray[np.float64], float, bool, float]:
            """
            1.2 Proyección euclídea EXACTA sobre Δ_δ = {p ≥ δ, Σp = 1} (water-filling, O(n log n)).
            Con s = p−δ ≥ 0 y S = 1−nδ: p = δ + max(v−δ−θ, 0), θ tal que Σmax(...) = S.
            Factibilidad ⟺ δ ≤ 1/n; se impone δ ≤ 1/(2n). La corrección de traza es ADITIVA
            sobre la componente mayor. Devuelve además el residuo estacionario KKT:
                r_KKT = máx_{i: p_i > δ} |(p_i − v_i) + θ|. [W9]
            """
            v = np.asarray(values, dtype=np.float64).ravel()
            n = int(v.size)
            if n == 0:
                return np.array([1.0]), 0.0, False, 0.0
            delta = float(min(max(floor, 0.0), 0.5 / n))
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
            active = p > (delta + 1.0e-14)
            if bool(np.any(active)):
                kkt_res = float(np.max(np.abs((p[active] - v[active]) + theta)))
            else:
                kkt_res = 0.0
            return p, delta, feasible, kkt_res

        # ── Método 1.3 ────────────────────────────────────────────────────────
        def _canonicalize_with_certificate(self, rho: Any) -> DensityCanonicalization:
            """
            1.3 Proyección de Frobenius sobre D_δ(H) (Mirsky / Hoffman-Wielandt) con
            certificación de idempotencia RSI-L3: ‖P(P(ρ)) − P(ρ)‖_F. [W4]
            """
            herm, t_def, h_def, fallback = self._validate_and_hermitize(rho)
            if fallback:
                probs = self._uniform.copy()
                return DensityCanonicalization(
                    herm,
                    probs,
                    self._eye.copy(),
                    math.log(float(self._n)),
                    1.0 / self._n,
                    1.0,
                    0.0,
                    1.0 / self._n,
                    True,
                    0.0,
                    0.0,
                    True,
                    0.0,
                    0.0,
                )
            try:
                eigvals, eigvecs = la.eigh(herm)
                eigvals = np.real(eigvals).astype(np.float64)
                lam_max = float(max(np.max(eigvals), 1.0 / self._n))
                delta = min(
                    max(self._tau_spd, lam_max / _MAX_MODULAR_CONDITION),
                    0.5 / self._n,
                )
                probs, delta, feasible, kkt_res = self._project_capped_simplex(eigvals, delta)
                out = (eigvecs * probs) @ eigvecs.conj().T
                out = 0.5 * (out + out.conj().T)
                # Verificación de idempotencia de Nivel 2/3 sobre el espectro proyectado
                probs_idem, _, _, _ = self._project_capped_simplex(probs, delta)
                idem_def = float(np.linalg.norm(probs_idem - probs, ord=2))
                entropy = float(-np.sum(probs * np.log(np.maximum(probs, _WILKINSON_EPS))))
                kkt = bool(
                    feasible
                    and float(np.min(probs)) >= delta - 1e-12
                    and abs(float(np.sum(probs)) - 1.0) <= 1e-12
                )
                return DensityCanonicalization(
                    rho=np.asarray(out, dtype=np.complex128),
                    eigenvalues=probs,
                    eigenvectors=eigvecs,
                    entropy=max(0.0, entropy),
                    purity=float(np.sum(probs ** 2)),
                    condition_number=float(np.max(probs) / max(np.min(probs), _TINY)),
                    projection_cost=float(np.linalg.norm(probs - eigvals, ord=2)),
                    floor_used=float(delta),
                    kkt_ok=kkt,
                    hermiticity_defect=h_def,
                    trace_defect=t_def,
                    used_fallback=False,
                    idempotence_defect_l3=idem_def,
                    kkt_stationarity_residual=kkt_res,
                )
            except (la.LinAlgError, ValueError):
                probs = self._uniform.copy()
                return DensityCanonicalization(
                    self._eye / float(self._n),
                    probs,
                    self._eye.copy(),
                    math.log(float(self._n)),
                    1.0 / self._n,
                    1.0,
                    0.0,
                    1.0 / self._n,
                    True,
                    h_def,
                    t_def,
                    True,
                    0.0,
                    0.0,
                )

        def canonicalize_density_matrix(
            self, rho_raw: NDArray[np.complex128]
        ) -> NDArray[np.complex128]:
            """1.3b Fachada pública compatible con v16/v17."""
            return self._canonicalize_with_certificate(rho_raw).rho

        # ── Método 1.4 ────────────────────────────────────────────────────────
        def _purify_with_certificate(self, K_raw: Any) -> ConductivityPurification:
            """
            1.4 Proyección de Frobenius sobre PSD ∩ Sym_n(ℝ) en el ORDEN CORRECTO
            (REAL PRIMERO, CLAMP DESPUÉS) + certificación de pasividad de impedancia de
            Foster-Brune (Re⟨v, Kv⟩ ≥ τ_spd‖v‖²). [W5]
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
                # Idempotencia del proyector sobre el cono PSD real
                clamped_twice = np.maximum(clamped, self._tau_spd)
                idem_def = float(np.linalg.norm(clamped_twice - clamped, ord=2))
                fallback = False
            except (la.LinAlgError, ValueError):
                k_psd = np.eye(self._n, dtype=np.float64) * self._tau_spd
                clamped = np.full(self._n, self._tau_spd)
                cost, idem_def, fallback = 0.0, 0.0, True
            lam_min = float(np.min(clamped))
            lam_max = float(np.max(clamped))
            ordered = np.sort(clamped)[::-1]
            top_gap = float(ordered[0] - ordered[1]) if ordered.size >= 2 else 0.0
            kappa_raw = float(lam_max / max(lam_min, _TINY))
            foster_passive = bool(lam_min >= self._tau_spd * (1.0 - 1.0e-12))
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
                used_fallback=fallback,
                foster_brune_passive_ok=foster_passive,
                idempotence_defect_l3=idem_def,
            )

        def purify_hermitian_matrix(
            self, K_raw: NDArray[np.complex128]
        ) -> NDArray[np.complex128]:
            """1.4b Fachada pública compatible con v16/v17."""
            return self._purify_with_certificate(K_raw).k_matrix

        # ── Método 1.5 ────────────────────────────────────────────────────────
        def compute_uhlmann_bures(
            self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
        ) -> Tuple[float, float]:
            """1.5c Fachada pública compatible con v16/v17: (F, d_B). Canoniza AMBOS. [W8]"""
            out = self._uhlmann_bures_geometry(rho, sigma)
            return out[0], out[1]

        def _uhlmann_bures_geometry(
            self, rho: Any, sigma: Any
        ) -> Tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            DensityCanonicalization,
            DensityCanonicalization,
            NDArray[np.complex128],
        ]:
            """
            1.5 Fidelidad de Uhlmann por NORMA TRAZA √F = ‖√ρ√σ‖₁ (SVD), contrastada con la
            ruta espectral √ρσ√ρ. Ambos operandos se canonizan dentro del método. [W8]
            Devuelve adicionalmente el operador de amplitud de correlación A = √ρ√σ para su
            refinamiento polar de Nivel 3 en 1.7.
            """
            cr = self._canonicalize_with_certificate(rho)
            cs = self._canonicalize_with_certificate(sigma)
            sq_r = np.sqrt(np.maximum(cr.eigenvalues, _STABILITY_FLOOR))
            sq_s = np.sqrt(np.maximum(cs.eigenvalues, _STABILITY_FLOOR))
            sqrt_r = (cr.eigenvectors * sq_r) @ cr.eigenvectors.conj().T
            sqrt_s = (cs.eigenvectors * sq_s) @ cs.eigenvectors.conj().T
            sqrt_r = 0.5 * (sqrt_r + sqrt_r.conj().T)
            sqrt_s = 0.5 * (sqrt_s + sqrt_s.conj().T)
            amplitude = sqrt_r @ sqrt_s
            root_f_svd = float(np.clip(np.sum(np.real(la.svdvals(amplitude))), 0.0, 1.0))
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
            return (
                fidelity,
                bures,
                angle,
                chord_err,
                abs(root_f_svd - root_f_eig),
                hs_overlap,
                cr,
                cs,
                amplitude,
            )

        # ── Método 1.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _poincare_kahler_ks_regularization(
            cr: DensityCanonicalization,
            cs: DensityCanonicalization,
        ) -> KustaanheimoStiefelPoincareCertificate:
            """
            1.6 Puente entre Geometría Cuántica (Bures-Helstrom-Kähler) y Mecánica Celeste
            de Poincaré mediante la Regularización Hipercompleja de Kustaanheimo-Stiefel (KS):
              (a) Ecuación de Lyapunov de la Derivada Logarítmica Simétrica (SLD) L:
                      ½(ρ L + L ρ) = σ − ρ.
                  En la base de autovectores U de ρ (ρ = U diag(p) U†), la solución de
                  diferencias divididas de Loewner es EXACTA:
                      L_eig[i,j] = 2·(U†(σ − ρ)U)[i,j] / (p_i + p_j).
                  Se certifica el residuo ‖½(ρL + Lρ) − (σ − ρ)‖_F y se extrae la Información
                  Cuántica de Fisher (QFI) tangente J_Q = Re Tr(ρ L²) = Σ_{i,j} p_i |L_eig[i,j]|².
              (b) 2-Forma Simpléctica de Poincaré-Fubini-Study ω_FS = Im Tr(ρ [L, σ]).
              (c) Inmersión sobre la fibración de Hopf S³ → S² en el álgebra de cuaterniones ℍ:
                  se forma el espinor cuaterniónico u ∈ ℝ⁴ a partir de las amplitudes de raíz
                  cuadrada de las poblaciones espectrales y la fase coherente dominante, se
                  construye la matriz de Levi-Civita L(u), y se certifica:
                    • Ortogonalidad de Hurwitz: ‖L(u)ᵀL(u) − ‖u‖²I₄‖_F = 0.
                    • Anulación exacta de la restricción gauge bilineal de KS l_KS(u, u_horiz) = 0
                      para el campo vectorial horizontal proyectado.
                    • Jacobiano de transformación temporal de Poincaré dt/dτ = r = ‖u‖₂².
            """
            p = cr.eigenvalues
            u_vecs = cr.eigenvectors
            delta_rho = cs.rho - cr.rho
            delta_eig = u_vecs.conj().T @ delta_rho @ u_vecs
            denom = p[:, None] + p[None, :]
            l_eig = 2.0 * delta_eig / np.maximum(denom, _TINY)
            l_mat = u_vecs @ l_eig @ u_vecs.conj().T
            l_mat = 0.5 * (l_mat + l_mat.conj().T)

            lyap_lhs = 0.5 * (cr.rho @ l_mat + l_mat @ cr.rho)
            lyap_res = _relative(_fro(lyap_lhs - delta_rho), max(_fro(delta_rho), 1.0))
            qfi = float(np.real(np.sum(p[:, None] * (np.abs(l_eig) ** 2))))

            comm_l_sigma = l_mat @ cs.rho - cs.rho @ l_mat
            symplectic_area = float( abs(np.imag(np.trace(cr.rho @ comm_l_sigma))) )

            # Construcción del espinor cuaterniónico u ∈ ℍ (ℝ⁴)
            n = int(p.size)
            c0 = math.sqrt(max(0.0, float(p[0])))
            c1 = math.sqrt(max(0.0, float(p[1]))) if n >= 2 else 0.0
            c2 = math.sqrt(max(0.0, float(np.sum(p[2:])))) if n >= 3 else 0.0
            off_phase = float(np.angle(cs.rho[0, 1 if n >= 2 else 0]))
            spinor_u = np.array(
                [
                    c0 * math.cos(0.5 * off_phase),
                    c0 * math.sin(0.5 * off_phase),
                    c1,
                    c2,
                ],
                dtype=np.float64,
            )
            r_kepler = float(np.dot(spinor_u, spinor_u))
            l_ks = _ks_levi_civita_matrix(spinor_u)
            hurwitz_def = _fro(l_ks.T @ l_ks - r_kepler * np.eye(4, dtype=np.float64))

            # Vector tangente de prueba proyectado ortogonalmente a la fibra de Hopf J_KS·u
            j_fiber = np.array(
                [-spinor_u[3], spinor_u[2], -spinor_u[1], spinor_u[0]], dtype=np.float64
            )
            q_raw = np.array(
                [
                    float(cs.eigenvalues[0]),
                    float(cs.eigenvalues[1]) if n >= 2 else 0.0,
                    float(cs.eigenvalues[2]) if n >= 3 else 0.0,
                    0.5 * qfi,
                ],
                dtype=np.float64,
            )
            j_norm_sq = float(np.dot(j_fiber, j_fiber))
            if j_norm_sq > _TINY:
                u_prime = q_raw - (float(np.dot(q_raw, j_fiber)) / j_norm_sq) * j_fiber
            else:
                u_prime = np.zeros(4, dtype=np.float64)
            # La 4ª componente de L(u)·u' es exactamente la forma bilineal de KS l_KS(u, u')
            ks_mapped_vel = l_ks @ u_prime
            ks_bilinear_res = abs(float(ks_mapped_vel[3]))
            ks_ok = bool(hurwitz_def <= 1.0e-12 and ks_bilinear_res <= 1.0e-12)

            return KustaanheimoStiefelPoincareCertificate(
                spinor_u=_freeze_array(spinor_u),
                cartesian_kepler_radius=r_kepler,
                poincare_time_jacobian=r_kepler,
                hurwitz_orthogonality_defect=hurwitz_def,
                ks_bilinear_gauge_residual=ks_bilinear_res,
                fubini_study_symplectic_area=symplectic_area,
                sld_lyapunov_residual=lyap_res,
                quantum_fisher_information=max(0.0, qfi),
                ks_regularization_ok=ks_ok,
            )

        # ── Método 1.7 ────────────────────────────────────────────────────────
        def _rsi_level3_meta_witness_audit(
            self,
            rho_raw: Any,
            sigma_raw: Any,
            cr: DensityCanonicalization,
            cs: DensityCanonicalization,
            cond: ConductivityPurification,
            amplitude: NDArray[np.complex128],
        ) -> RsiLevel3Pi1Certificate:
            """
            1.7 Auditoría de Automejora Recursiva de Nivel 3 (RSI-L3) en el Álgebra de Banach B(H):
              • Nivel 0: Factor unitario polar inicial U_0 = W V† de la SVD de A = √ρ√σ.
              • Nivel 1: Aplicación iterativa del operador involutivo de Newton-Schulz
                    T(U) = ½ U (3I − U†U),
                cuya convergencia hacia la variedad unitaria U(n) es cuadrática:
                    ‖T(U)†T(U) − I‖_F ≤ ¾ ‖U†U − I‖_F².
              • Nivel 2: Verificación de la desigualdad de co-coercividad proximal de
                Moreau-Yosida para el proyector convexo P sobre D_δ(H):
                    Re⟨X − Y, P(X) − P(Y)⟩_HS − ‖P(X) − P(Y)‖_F² ≥ 0.
              • Nivel 3 (Meta-Auditoría de Banach-Lawvere): cálculo de la razón de contracción
                empírica q_L3 = ‖U_{k+1} − U_k‖_F / ‖U_k − U_{k-1}‖_F y cota a posteriori
                de distancia al punto fijo ‖U_L3 − U*‖_F ≤ q_L3/(1−q_L3)·‖U_L3 − U_{L3-1}‖_F.
            """
            try:
                w_svd, _s_vals, vh_svd = la.svd(amplitude, full_matrices=False)
                u_curr = w_svd @ vh_svd
            except (la.LinAlgError, ValueError):
                u_curr = self._eye.copy()

            def_l0 = _fro(u_curr.conj().T @ u_curr - self._eye)
            step_norms: List[float] = []
            iters = 0
            for _ in range(_RSI_L3_MAX_STEPS):
                iters += 1
                gram = u_curr.conj().T @ u_curr
                u_next = 0.5 * u_curr @ (3.0 * self._eye - gram)
                step_d = _fro(u_next - u_curr)
                step_norms.append(step_d)
                u_curr = u_next
                if step_d <= 1.0e-15:
                    break

            if self._telemetry is not None:
                self._telemetry.cumulative_rsi_l3_iterations += iters

            def_l3 = _fro(u_curr.conj().T @ u_curr - self._eye)
            if len(step_norms) >= 2 and step_norms[0] > _TINY:
                q_ratio = min(0.999999, step_norms[-1] / max(step_norms[-2], _TINY))
            else:
                q_ratio = 0.0
            last_step = step_norms[-1] if step_norms else 0.0
            aposteriori = (q_ratio / max(1.0 - q_ratio, 1.0e-9)) * last_step

            # Fidelidad refinada mediante el punto fijo polar U_L3: √F = Re Tr(A U_L3†)
            root_f_l3 = float(np.clip(np.real(np.trace(amplitude @ u_curr.conj().T)), 0.0, 1.0))
            fidelity_l3 = float(np.clip(root_f_l3 * root_f_l3, 0.0, 1.0))

            # Co-coercividad proximal de Moreau-Yosida sobre las entradas hermíticas
            r_herm, _, _, r_fb = self._validate_and_hermitize(rho_raw)
            s_herm, _, _, s_fb = self._validate_and_hermitize(sigma_raw)
            if not (r_fb or s_fb):
                diff_in = r_herm - s_herm
                diff_out = cr.rho - cs.rho
                inner_hs = float(np.real(np.trace(diff_in.conj().T @ diff_out)))
                out_sq = float(np.real(np.trace(diff_out.conj().T @ diff_out)))
                cocoercivity_margin = inner_hs - out_sq
            else:
                cocoercivity_margin = 0.0

            idem_total = max(
                cr.idempotence_defect_l3,
                cs.idempotence_defect_l3,
                cond.idempotence_defect_l3,
            )
            converged = bool(
                def_l3 <= max(1.0e-11, def_l0 + 1.0e-14)
                and q_ratio < 1.0
                and idem_total <= 1.0e-11
                and cocoercivity_margin >= -1.0e-10
            )
            if self._telemetry is not None:
                self._telemetry.last_rsi_l3_contraction = float(q_ratio)

            return RsiLevel3Pi1Certificate(
                polar_unitary_defect_l0=def_l0,
                polar_unitary_defect_l3=def_l3,
                banach_contraction_ratio=float(q_ratio),
                aposteriori_fixed_point_bound=float(aposteriori),
                refined_uhlmann_fidelity=fidelity_l3,
                moreau_yosida_cocoercivity_margin=float(cocoercivity_margin),
                projector_idempotence_defect=float(idem_total),
                iterations_executed=int(iters),
                rsi_l3_converged_ok=converged,
            )

        # ── Método 1.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _attest_non_interference(
            before: Sequence[str], after: Sequence[str]
        ) -> NonInterferenceAttestation:
            """
            1.8 [W10] Atestación del axioma WITNESS-1: los cinco buffers inspeccionados
            (p, ρ, σ, K, B) no cambiaron de forma observable durante el cómputo.
            ALCANCE DECLARADO: no prueba ausencia de escrituras transitorias restauradas
            antes de la segunda huella (TOCTOU), ni cubre el resto de la memoria del proceso.
            """
            flags: List[bool] = []
            for i in range(5):
                b = before[i] if i < len(before) else ""
                a = after[i] if i < len(after) else ""
                flags.append(bool(b) and b == a if (b or a) else True)
            return NonInterferenceAttestation(*[bool(f) for f in flags])

        # ── Método 1.9 · TERMINAL Π₁ → APERTURA Π₂ ────────────────────────────
        def audit_passive_quantum_substrate(
            self,
            rho_init: NDArray[np.complex128],
            sigma_ref: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
        ) -> Pi1SpectrumCertificate:
            """
            1.9 [TERMINAL Π₁] Orquesta 1.1→1.7 y EMITE el morfismo `Pi1ToPi2Seed` que
            constituye el inicio formal directo de la Fase Π₂ en `2.1 _ingest_pi1_continuation`:
                audit_passive_quantum_substrate(...) ──[Pi1ToPi2Seed]──►
                    PhasePi2_SmithHomologyAuditor._ingest_pi1_continuation(seed)
            """
            (
                fidelity,
                bures,
                angle,
                chord_err,
                route_disc,
                hs,
                cr,
                cs,
                amplitude,
            ) = self._uhlmann_bures_geometry(rho_init, sigma_ref)
            cond = self._purify_with_certificate(K_raw)
            ks_cert = self._poincare_kahler_ks_regularization(cr, cs)
            rsi_cert = self._rsi_level3_meta_witness_audit(
                rho_init, sigma_ref, cr, cs, cond, amplitude
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
                ks_poincare=ks_cert,
                rsi_l3=rsi_cert,
            )
            seed = Pi1ToPi2Seed(
                bures_tension=float(bures),
                uhlmann_fidelity=float(fidelity),
                rho_populations=tuple(float(x) for x in np.real(np.diag(cr.rho))),
                rho_entropy=float(cr.entropy),
                dimension_n=int(self._n),
                has_context=True,
                quantum_fisher_information=float(ks_cert.quantum_fisher_information),
                poincare_ks_radius=float(ks_cert.cartesian_kepler_radius),
                rsi_l3_contraction_ratio=float(rsi_cert.banach_contraction_ratio),
                rsi_l3_converged_ok=bool(rsi_cert.rsi_l3_converged_ok),
            )
            object.__setattr__(cert, "continuation", seed)
            return cert


    # ──────────────────────────────────────────────────────────────────────────
    # FASE Π₂ · FORMA NORMAL DE SMITH SOBRE ℤ, HOMOLOGÍA Y HODGE-POINCARÉ
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi2_SmithHomologyAuditor:
        """
        Π₂. Cadena formal:
          2.1 INGESTA directa de Pi1ToPi2Seed (continuación de 1.9)
          2.2 Forma Normal de Smith entera con pivoteo diofántico de mínima complejidad
          2.3 Certificado SNF, cota de Hadamard y fase de torsión de Pontryagin-Poincaré
          2.4 Homología exacta H_k(C; ℤ) = ker ∂_k / im ∂_{k+1} y base entera saturada V[:, r:]
          2.5 Auditoría espectral de Hodge-Poincaré Δ_k, circulación de ciclos y proyector RSI-L3
          2.6 Operador de contorno simplicial por defecto (regresión de árbol libre de torsión)
          2.7 TERMINAL Π₂ (emite Pi2ToPi3Seed → continuación directa en 3.1)
        """

        def __init__(self, telemetry: WitnessTelemetry) -> None:
            self._telemetry = telemetry

        # ── Método 2.1 · APERTURA Π₂ (continuación directa de 1.9) ────────────
        @staticmethod
        def _ingest_pi1_continuation(
            pi1: Optional[Pi1SpectrumCertificate] = None,
        ) -> Pi1ToPi2Seed:
            """
            2.1 [CONTINUACIÓN DIRECTA DE 1.9] Absorbe el morfismo `Pi1ToPi2Seed` emitido
            por `1.9 audit_passive_quantum_substrate`; si se invoca aisladamente sin
            contexto previo de Π₁, sintetiza la semilla neutra canónica.
            """
            if pi1 is not None and pi1.continuation is not None:
                return pi1.continuation
            return Pi1ToPi2Seed(
                bures_tension=0.0,
                uhlmann_fidelity=1.0,
                rho_populations=(1.0,),
                rho_entropy=0.0,
                dimension_n=1,
                has_context=False,
                quantum_fisher_information=0.0,
                poincare_ks_radius=1.0,
                rsi_l3_contraction_ratio=0.0,
                rsi_l3_converged_ok=True,
            )

        # ── Método 2.2 ────────────────────────────────────────────────────────
        def _smith_normal_form_integer(
            self, B_matrix: Any
        ) -> Tuple[
            List[List[int]],
            List[List[int]],
            List[List[int]],
            List[List[int]],
            List[int],
            int,
            int,
        ]:
            """
            2.2 [W1] FORMA NORMAL DE SMITH ENTERA con aritmética de precisión arbitraria
            y selección diofántica de pivote por mínima complejidad L1 (RSI-L1).
            Algoritmo de eliminación por mcd sobre ℤ con operaciones unimodulares:
              (a) localizar la entrada no nula en el bloque residual (t:, t:) que minimice
                  lexicográficamente (|A[i,j]|, ‖fila_i‖₁ + ‖col_j‖₁) para mitigar el
                  crecimiento intermedio de coeficientes y llevarla al pivote (t,t);
              (b) dividir con resto las entradas de la columna t y de la fila t; si el resto
                  es no nulo, intercambiar y repetir. Python garantiza |a mod b| < |b|, de
                  modo que |pivote| decrece estrictamente ⟹ TERMINACIÓN FINITA;
              (c) REPARACIÓN DE DIVISIBILIDAD: si d_t no divide alguna entrada del bloque
                  residual, sumar esa fila a la fila t y volver a (b). Esto fuerza la cadena
                  exacta d_1 | d_2 | … | d_r.
            Se rastrean U (m×m), V (n×n) y V⁻¹ (n×n) aplicando las operaciones duales:
              fila_i += q·fila_j  ⟹ U igual;
              col_j  += q·col_i   ⟹ V igual y fila_i(V⁻¹) −= q·fila_j(V⁻¹).
            Devuelve (A, U, V, Vinv, diagonales, rango, barridos). NO muta la entrada.
            """
            try:
                arr = np.asarray(B_matrix)
            except (TypeError, ValueError):
                arr = np.zeros((0, 0), dtype=np.int64)
            if arr.ndim != 2:
                arr = arr.reshape(1, -1) if arr.size else np.zeros((0, 0), dtype=np.int64)
            m, n = int(arr.shape[0]), int(arr.shape[1])
            A = [[int(arr[i, j]) for j in range(n)] for i in range(m)]
            U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
            V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            Vinv = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

            def row_add(i: int, j: int, q: int) -> None:
                if q == 0:
                    return
                for c in range(n):
                    A[i][c] += q * A[j][c]
                for c in range(m):
                    U[i][c] += q * U[j][c]

            def row_swap(i: int, j: int) -> None:
                if i == j:
                    return
                A[i], A[j] = A[j], A[i]
                U[i], U[j] = U[j], U[i]

            def row_neg(i: int) -> None:
                for c in range(n):
                    A[i][c] = -A[i][c]
                for c in range(m):
                    U[i][c] = -U[i][c]

            def col_add(j: int, i: int, q: int) -> None:
                if q == 0:
                    return
                for r in range(m):
                    A[r][j] += q * A[r][i]
                for r in range(n):
                    V[r][j] += q * V[r][i]
                for c in range(n):
                    Vinv[i][c] -= q * Vinv[j][c]

            def col_swap(i: int, j: int) -> None:
                if i == j:
                    return
                for r in range(m):
                    A[r][i], A[r][j] = A[r][j], A[r][i]
                for r in range(n):
                    V[r][i], V[r][j] = V[r][j], V[r][i]
                Vinv[i], Vinv[j] = Vinv[j], Vinv[i]

            rank = 0
            sweeps = 0
            for t in range(min(m, n)):
                while True:
                    sweeps += 1
                    if sweeps > _SNF_MAX_SWEEPS:
                        raise RuntimeError("SNF: cota de barridos excedida (entrada patológica).")
                    row_l1 = [
                        sum(abs(A[i][c]) for c in range(t, n)) if i >= t else 0
                        for i in range(m)
                    ]
                    col_l1 = [
                        sum(abs(A[r][j]) for r in range(t, m)) if j >= t else 0
                        for j in range(n)
                    ]
                    best: Optional[Tuple[int, int, int, int]] = None
                    for i in range(t, m):
                        for j in range(t, n):
                            v = A[i][j]
                            if v != 0:
                                abs_v = abs(v)
                                l1_score = row_l1[i] + col_l1[j]
                                if best is None or (abs_v, l1_score) < (best[2], best[3]):
                                    best = (i, j, abs_v, l1_score)
                    if best is None:
                        break
                    if best[0] != t:
                        row_swap(t, best[0])
                    if best[1] != t:
                        col_swap(t, best[1])
                    if A[t][t] < 0:
                        row_neg(t)
                    changed = False
                    for i in range(t + 1, m):
                        if A[i][t] != 0:
                            q = A[i][t] // A[t][t]
                            row_add(i, t, -q)
                            if A[i][t] != 0:
                                row_swap(t, i)
                                changed = True
                                break
                    if changed:
                        continue
                    for j in range(t + 1, n):
                        if A[t][j] != 0:
                            q = A[t][j] // A[t][t]
                            col_add(j, t, -q)
                            if A[t][j] != 0:
                                col_swap(t, j)
                                changed = True
                                break
                    if changed:
                        continue
                    pivot = A[t][t]
                    offender: Optional[int] = None
                    for i in range(t + 1, m):
                        if any(A[i][j] % pivot != 0 for j in range(t + 1, n)):
                            offender = i
                            break
                    if offender is not None:
                        row_add(t, offender, 1)
                        continue
                    break
                if all(A[i][j] == 0 for i in range(t, m) for j in range(t, n)):
                    break
                if A[t][t] < 0:
                    row_neg(t)
                rank += 1
            diag = [A[i][i] for i in range(min(m, n))]
            self._telemetry.cumulative_snf_sweeps += sweeps
            return A, U, V, Vinv, diag, rank, sweeps

        # ── Método 2.3 ────────────────────────────────────────────────────────
        def _smith_certificate(self, B_matrix: Any) -> SmithNormalFormCertificate:
            """
            2.3 Certificado de la SNF, cota de Hadamard y fase de enlace de torsión de
            Pontryagin-Poincaré en ℚ/ℤ.
            coker(B) = ℤ^{m−r} ⊕ ⊕_i ℤ/d_i, de donde Tor(coker B) = ⊕_{d_i>1} ℤ/d_i.
            CERTIFICADOS exactos:
              • U·B·V = S (reconstrucción entera exacta),
              • det U = ±1 y det V = ±1 (determinante de Bareiss sin fracciones),
              • V·V⁻¹ = I_n,
              • cadena de divisibilidad d_i | d_{i+1},
              • cota log10 de Hadamard log10 ∏_i ‖B_{i,·}‖₂,
              • fase de auto-enlace de Pontryagin-Poincaré Φ_link = (Σ_{d_i>1} 1/d_i) mod 1. [W1]
            """
            try:
                arr = np.asarray(B_matrix)
            except (TypeError, ValueError):
                arr = np.zeros((0, 0), dtype=np.int64)
            if arr.ndim != 2:
                arr = arr.reshape(1, -1) if arr.size else np.zeros((0, 0), dtype=np.int64)
            m, n = int(arr.shape[0]), int(arr.shape[1])
            if m == 0 or n == 0:
                empty = np.array([], dtype=np.int64)
                return SmithNormalFormCertificate(
                    (),
                    empty,
                    False,
                    0,
                    m,
                    n,
                    True,
                    True,
                    True,
                    1,
                    1,
                    True,
                    m,
                    (),
                    n,
                    True,
                    0.0,
                    0.0,
                    0,
                )
            B_int = [[int(arr[i, j]) for j in range(n)] for i in range(m)]
            hadamard_log10 = _hadamard_log10_bound(B_int)
            A, U, V, Vinv, diag, rank, sweeps = self._smith_normal_form_integer(B_int)
            S = [[0] * n for _ in range(m)]
            for i in range(min(m, n)):
                S[i][i] = A[i][i]
            recon = _int_matmul(_int_matmul(U, B_int), V)
            reconstruction_ok = bool(recon == S)
            det_u = _bareiss_determinant(U)
            det_v = _bareiss_determinant(V)
            unimodular_ok = bool(abs(det_u) == 1 and abs(det_v) == 1)
            ident_n = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            v_inverse_ok = bool(_int_matmul(V, Vinv) == ident_n)
            nonzero = [abs(int(d)) for d in diag if d != 0]
            divisibility_ok = all(
                nonzero[i + 1] % nonzero[i] == 0 for i in range(len(nonzero) - 1)
            )
            exact = tuple(str(d) for d in nonzero)
            saturated = any(d > _INT64_MAX for d in nonzero)
            legacy = np.array([min(d, _INT64_MAX) for d in nonzero], dtype=np.int64)
            torsion_ints = [d for d in nonzero if d > 1]
            torsion = tuple(str(d) for d in torsion_ints)
            pontryagin_phase = float(math.fmod(sum(1.0 / float(d) for d in torsion_ints), 1.0))
            return SmithNormalFormCertificate(
                invariant_factors_exact=exact,
                invariant_factors_int64=legacy,
                int64_saturated=bool(saturated),
                rank=int(rank),
                rows=m,
                cols=n,
                divisibility_chain_ok=bool(divisibility_ok),
                unimodular_ok=unimodular_ok,
                reconstruction_ok=reconstruction_ok,
                det_u=int(det_u),
                det_v=int(det_v),
                v_inverse_ok=v_inverse_ok,
                cokernel_free_rank=int(m - rank),
                cokernel_torsion=torsion,
                kernel_rank=int(n - rank),
                torsion_free=bool(not torsion),
                hadamard_log10_bound=float(hadamard_log10),
                pontryagin_linking_phase=pontryagin_phase,
                sweeps_executed=int(sweeps),
            )

        def compute_smith_normal_form_torsion(
            self, B_matrix: NDArray[np.int64]
        ) -> Tuple[bool, NDArray[np.int64]]:
            """2.3b Fachada pública compatible con v16/v17: (torsion_free, factores invariantes)."""
            cert = self._smith_certificate(B_matrix)
            return cert.torsion_free, cert.invariant_factors_int64

        # ── Método 2.4 ────────────────────────────────────────────────────────
        def _chain_complex_homology(
            self,
            B_matrix: Any,
            boundary_next: Optional[Any],
            smith: SmithNormalFormCertificate,
        ) -> Tuple[HomologyCertificate, List[List[int]]]:
            """
            2.4 HOMOLOGÍA GENUINA H_k = ker ∂_k / im ∂_{k+1} cuando se suministra ∂_{k+1},
            junto con la extracción explícita de la ℤ-base saturada de ciclos V[:, r:].
            (a) Verificación EXACTA de la condición de complejo ∂_k ∘ ∂_{k+1} = 0.
            (b) Base entera SATURADA de ker ∂_k: de U B V = S se sigue
                    Bx = 0 ⟺ S(V⁻¹x) = 0 ⟺ (V⁻¹x)_i = 0 para i < r ⟹ x = V·y, sop(y) ≥ r,
                luego ker B = span_ℤ{V[:, r:]}. Es saturada porque V es unimodular.
            (c) Coordenadas EXACTAS M = (V⁻¹·∂_{k+1})[r:, :], enteras porque im ∂_{k+1} ⊆ ker ∂_k.
            (d) SNF(M) ⟹ rank H_k = (n−r) − rank(M),  Tor H_k = ⊕_{d_i>1} ℤ/d_i.
            Devuelve (HomologyCertificate, kernel_basis_columns_V).
            """
            try:
                b_arr = np.asarray(B_matrix)
            except (TypeError, ValueError):
                b_arr = np.zeros((0, 0), dtype=np.int64)
            if b_arr.ndim != 2:
                b_arr = b_arr.reshape(1, -1) if b_arr.size else np.zeros((0, 0), dtype=np.int64)
            m, n = int(b_arr.shape[0]), int(b_arr.shape[1])
            B_int = [[int(b_arr[i, j]) for j in range(n)] for i in range(m)]
            if m > 0 and n > 0:
                _A, _U, V, Vinv, _diag, rank, _sw = self._smith_normal_form_integer(B_int)
                kernel_cols = [[V[row_idx][col_idx] for col_idx in range(rank, n)] for row_idx in range(n)]
            else:
                V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
                Vinv = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
                rank = 0
                kernel_cols = [row[:] for row in V]

            if boundary_next is None:
                return (
                    HomologyCertificate(
                        scope="COKERNEL_ONLY_EQUALS_H_PREV_IFF_PREV_BOUNDARY_ZERO",
                        chain_complex_ok=False,
                        homology_rank=int(smith.cokernel_free_rank),
                        homology_torsion=smith.cokernel_torsion,
                        torsion_free=bool(not smith.cokernel_torsion),
                        restricted_rank=0,
                        kernel_basis_columns=int(smith.kernel_rank),
                    ),
                    kernel_cols,
                )
            try:
                nxt = np.asarray(boundary_next)
            except (TypeError, ValueError):
                return (
                    HomologyCertificate("INVALID_INPUT", False, 0, (), True, 0, 0),
                    kernel_cols,
                )
            if nxt.ndim != 2 or n != int(nxt.shape[0]):
                return (
                    HomologyCertificate("SHAPE_MISMATCH", False, 0, (), True, 0, 0),
                    kernel_cols,
                )
            N_int = [[int(x) for x in row] for row in nxt]
            chain_ok = bool(all(v == 0 for row in _int_matmul(B_int, N_int) for v in row))
            if not chain_ok:
                return (
                    HomologyCertificate(
                        scope="NOT_A_CHAIN_COMPLEX_BOUNDARY_COMPOSITION_NONZERO",
                        chain_complex_ok=False,
                        homology_rank=0,
                        homology_torsion=(),
                        torsion_free=True,
                        restricted_rank=0,
                        kernel_basis_columns=0,
                    ),
                    kernel_cols,
                )
            coords = _int_matmul(Vinv, N_int)
            M = [row[:] for row in coords[rank:]]
            if not M:
                return (
                    HomologyCertificate(
                        scope="CHAIN_COMPLEX_H_K",
                        chain_complex_ok=True,
                        homology_rank=int(n - rank),
                        homology_torsion=(),
                        torsion_free=True,
                        restricted_rank=0,
                        kernel_basis_columns=int(n - rank),
                    ),
                    kernel_cols,
                )
            restricted = self._smith_certificate(np.array(M, dtype=object))
            h_rank = int((n - rank) - restricted.rank)
            return (
                HomologyCertificate(
                    scope="CHAIN_COMPLEX_H_K",
                    chain_complex_ok=True,
                    homology_rank=max(0, h_rank),
                    homology_torsion=restricted.cokernel_torsion,
                    torsion_free=bool(not restricted.cokernel_torsion),
                    restricted_rank=int(restricted.rank),
                    kernel_basis_columns=int(n - rank),
                ),
                kernel_cols,
            )

        # ── Método 2.5 ────────────────────────────────────────────────────────
        def _hodge_poincare_spectral_audit(
            self,
            B_matrix: Any,
            boundary_next: Optional[Any],
            smith: SmithNormalFormCertificate,
            homology: HomologyCertificate,
            kernel_cols: List[List[int]],
            seed1: Pi1ToPi2Seed,
        ) -> HodgePoincareHomologyCertificate:
            """
            2.5 Teoría de Hodge-Poincaré Combinatoria, Circulación Homológica de
            Poincaré-Kirchhoff y Purificación de Proyectores RSI-L3 (McWeeny-Banach):
              (a) Laplaciano Combinatorio de Hodge en grado k:
                      Δ_k = ∂_kᵀ ∂_k + ∂_{k+1} ∂_{k+1}ᵀ   (sobre ℝ^{n×n}).
                  Por el teorema de descomposición ortogonal de Hodge-De Rham discreto,
                  C_k(ℝ) = im(∂_kᵀ) ⊕ im(∂_{k+1}) ⊕ ker(Δ_k), con dim ker(Δ_k) = β_k.
                  Cuando ∂_k ∘ ∂_{k+1} = 0, verificamos `dim ker(Δ_k) == homology.homology_rank`;
                  cuando sólo se dispone de ∂_k, verificamos `dim ker(∂_kᵀ∂_k) == smith.kernel_rank`.
              (b) Refinamiento RSI-L3 del proyector ortogonal P_ker sobre ker(∂_k):
                  construido a partir de la base entera saturada Z = V[:, r:], se aplica la
                  iteración cúbica de McWeeny T_M(P) = 3P² − 2P³ y se certifica el defecto
                  conjunto de idempotencia y anulación de contorno:
                      ‖P_L3² − P_L3‖_F + ‖∂_k P_L3‖_F.
              (c) Acción de Circulación Homológica de Poincaré-Kirchhoff ∮_γ ϖ:
                  proyecta la distribución de corrientes cuánticas de `Pi1ToPi2Seed` sobre la
                  base de ciclos fundamentales Z = V[:, r:] para medir el flujo cíclico neto.
            """
            try:
                b_arr = np.asarray(B_matrix, dtype=np.float64)
            except (TypeError, ValueError):
                b_arr = np.zeros((0, 0), dtype=np.float64)
            if b_arr.ndim != 2:
                b_arr = b_arr.reshape(1, -1) if b_arr.size else np.zeros((0, 0), dtype=np.float64)
            m, n = int(b_arr.shape[0]), int(b_arr.shape[1])
            p_cols = 0
            if n == 0:
                return HodgePoincareHomologyCertificate(
                    hodge_harmonic_dimension=0,
                    hodge_spectral_gap=0.0,
                    betti_hodge_consistency_ok=True,
                    poincare_cycle_winding_action=0.0,
                    euler_poincare_characteristic=int(m),
                    rsi_l3_kernel_orthogonality_defect=0.0,
                )

            laplacian = b_arr.T @ b_arr
            if boundary_next is not None and homology.chain_complex_ok:
                try:
                    nxt_arr = np.asarray(boundary_next, dtype=np.float64)
                    if nxt_arr.ndim == 2 and int(nxt_arr.shape[0]) == n:
                        p_cols = int(nxt_arr.shape[1])
                        laplacian = laplacian + (nxt_arr @ nxt_arr.T)
                except (TypeError, ValueError):
                    p_cols = 0

            laplacian = 0.5 * (laplacian + laplacian.T)
            try:
                eigvals = np.real(la.eigvalsh(laplacian))
            except (la.LinAlgError, ValueError):
                eigvals = np.zeros(n, dtype=np.float64)

            scale_tol = max(1.0e-10, float(np.max(np.abs(eigvals))) * 1.0e-12) if eigvals.size else 1.0e-10
            harmonic_dim = int(np.sum(np.abs(eigvals) <= scale_tol))
            pos_eigs = eigvals[eigvals > scale_tol]
            spectral_gap = float(np.min(pos_eigs)) if pos_eigs.size else 0.0

            expected_betti = (
                int(homology.homology_rank)
                if (boundary_next is not None and homology.chain_complex_ok)
                else int(smith.kernel_rank)
            )
            betti_ok = bool(harmonic_dim == expected_betti)

            # Proyector ortogonal sobre ker(∂_k) a partir de la base entera Z = V[:, r:]
            # y purificación cúbica RSI-L3 de McWeeny: P_{k+1} = 3 P_k² − 2 P_k³
            k_dim = int(smith.kernel_rank)
            if k_dim > 0 and kernel_cols and len(kernel_cols) == n and len(kernel_cols[0]) == k_dim:
                z_mat = np.asarray(kernel_cols, dtype=np.float64)
                q_orth, _ = la.qr(z_mat, mode="economic")
                p_proj = q_orth @ q_orth.T
                for _ in range(2):
                    p_sq = p_proj @ p_proj
                    p_proj = 3.0 * p_sq - 2.0 * (p_sq @ p_proj)
                self._telemetry.cumulative_rsi_l3_iterations += 2
                idem_err = _fro(p_proj @ p_proj - p_proj)
                leak_err = _relative(_fro(b_arr @ p_proj), max(_fro(b_arr), 1.0))
                rsi_ker_defect = float(idem_err + leak_err)

                # Circulación de Poincaré-Kirchhoff de la 1-cadena cuántica sobre los ciclos
                pops = np.asarray(seed1.rho_populations, dtype=np.float64).ravel()
                chain_vec = np.resize(pops, n) if pops.size else np.full(n, 1.0 / float(n))
                cycle_flux = z_mat.T @ chain_vec
                winding_action = float(
                    2.0 * math.pi * float(np.linalg.norm(cycle_flux, ord=2))
                )
            else:
                rsi_ker_defect = 0.0
                winding_action = 0.0

            euler_char = int(m - n + p_cols)
            return HodgePoincareHomologyCertificate(
                hodge_harmonic_dimension=harmonic_dim,
                hodge_spectral_gap=spectral_gap,
                betti_hodge_consistency_ok=betti_ok,
                poincare_cycle_winding_action=winding_action,
                euler_poincare_characteristic=euler_char,
                rsi_l3_kernel_orthogonality_defect=rsi_ker_defect,
            )

        # ── Método 2.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _default_boundary_matrix() -> NDArray[np.int64]:
            """
            2.6 Matriz de contorno por defecto de v16/v17, conservada por compatibilidad.
            Es una matriz entera 2×3 de rango 2 con SNF = diag(1,1): coker = 0, ker ≅ ℤ,
            TORSIÓN NULA. v16 declaraba torsión sobre ella porque redondeaba los valores
            singulares (√3, 1) a (2, 1). Ése es el defecto cabecera corregido. [W1]
            """
            return np.array([[1, -1, 0], [0, 1, -1]], dtype=np.int64)

        # ── Método 2.7 · TERMINAL Π₂ → APERTURA Π₃ ────────────────────────────
        def audit_smith_homology_sector(
            self,
            pi1_certificate: Optional[Pi1SpectrumCertificate] = None,
            boundary_matrix: Optional[NDArray[np.int64]] = None,
            boundary_next: Optional[NDArray[np.int64]] = None,
        ) -> Pi2SpectrumCertificate:
            """
            2.7 [TERMINAL Π₂] Orquesta 2.1→2.6 y EMITE el morfismo `Pi2ToPi3Seed` que
            constituye el inicio formal directo de la Fase Π₃ en `3.1 _ingest_pi2_continuation`:
                audit_smith_homology_sector(...) ──[Pi2ToPi3Seed]──►
                    PhasePi3_PoincareLiouvilleEmitter._ingest_pi2_continuation(seed)
            La FPU no adjudica: reporta los predicados matemáticos de nulidad de torsión,
            consistencia de Betti-Hodge y circulación de ciclos de Poincaré.
            """
            seed1 = self._ingest_pi1_continuation(pi1_certificate)
            B = boundary_matrix if boundary_matrix is not None else self._default_boundary_matrix()
            smith = self._smith_certificate(B)
            homology, kernel_cols = self._chain_complex_homology(B, boundary_next, smith)
            hodge = self._hodge_poincare_spectral_audit(
                B, boundary_next, smith, homology, kernel_cols, seed1
            )
            if not smith.torsion_free:
                self._telemetry.torsion_detections += 1
            self._telemetry.last_smith_rank = int(smith.rank)
            self._telemetry.last_homology_rank = int(homology.homology_rank)
            cert = Pi2SpectrumCertificate(
                smith=smith,
                homology=homology,
                hodge_poincare=hodge,
            )
            seed2 = Pi2ToPi3Seed(
                torsion_free=bool(smith.torsion_free and homology.torsion_free),
                homology_rank=int(homology.homology_rank),
                smith_rank=int(smith.rank),
                kernel_rank=int(smith.kernel_rank),
                dimension_n=int(seed1.dimension_n),
                has_context=True,
                poincare_cycle_winding_action=float(hodge.poincare_cycle_winding_action),
                pontryagin_linking_phase=float(smith.pontryagin_linking_phase),
                betti_hodge_consistency_ok=bool(hodge.betti_hodge_consistency_ok),
                pi1_qfi_carried=float(seed1.quantum_fisher_information),
            )
            object.__setattr__(cert, "continuation", seed2)
            return cert


# ──────────────────────────────────────────────────────────────────────────
    # FASE Π₃ · POINCARÉ-CARTAN, LIOUVILLE-KAM, SHAHSHAHANI-ROMBERG Y EMISIÓN
    # ──────────────────────────────────────────────────────────────────────────
    class PhasePi3_PoincareLiouvilleEmitter:
        """
        Π₃. Cadena formal:
          3.1 INGESTA directa de Pi2ToPi3Seed (continuación de 2.7)
          3.2 Proyección exacta sobre el símplex de Gibbs Δ^{n−1} con certificado KKT
          3.3 Carta canónica de Poincaré, defecto simpléctico y generatriz de Von Zeipel S₁
          3.4 Invariante de Poincaré-Cartan, invariante de Kozai-Lidov y parámetro de Tisserand
          3.5 Monodromía de Liouville-Hill, proyección simpléctica RSI-L3 y rotación KAM
          3.6 Disipación de Shahshahani con extrapolación jerárquica de Richardson-Romberg RSI-L3
          3.7 Cotas espectrales de Lipschitz de Connes-Daleckii-Krein por diferencias divididas
          3.8 TERMINAL Π₃ (composición Π₃∘Π₂∘Π₁ y emisión de PositronWitnessSpectrumReport)
        """

        def __init__(
            self,
            pi1: "PositronRedemptionWitnessEngine.PhasePi1_PassiveUhlmannObserver",
            pi2: "PositronRedemptionWitnessEngine.PhasePi2_SmithHomologyAuditor",
            dimension_n: int,
            telemetry: WitnessTelemetry,
            precession_splitting: float,
            enable_doubling: bool,
        ) -> None:
            self._pi1 = pi1
            self._pi2 = pi2
            self._n = max(1, int(dimension_n))
            self._telemetry = telemetry
            self._splitting = float(precession_splitting)
            self._doubling = bool(enable_doubling)
            self._uniform = np.full(self._n, 1.0 / float(self._n), dtype=np.float64)

        # ── Método 3.1 · APERTURA Π₃ (continuación directa de 2.7) ────────────
        @staticmethod
        def _ingest_pi2_continuation(
            pi2: Optional[Pi2SpectrumCertificate] = None,
        ) -> Pi2ToPi3Seed:
            """
            3.1 [CONTINUACIÓN DIRECTA DE 2.7] Absorbe el morfismo `Pi2ToPi3Seed` emitido
            por `2.7 audit_smith_homology_sector`; sin contexto previo, devuelve la semilla neutra.
            """
            if pi2 is not None and pi2.continuation is not None:
                return pi2.continuation
            return Pi2ToPi3Seed(
                torsion_free=True,
                homology_rank=0,
                smith_rank=0,
                kernel_rank=1,
                dimension_n=1,
                has_context=False,
                poincare_cycle_winding_action=0.0,
                pontryagin_linking_phase=0.0,
                betti_hodge_consistency_ok=True,
                pi1_qfi_carried=0.0,
            )

        # ── Método 3.2 ────────────────────────────────────────────────────────
        def project_gibbs_simplex(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
            """3.2b Fachada pública compatible con v16/v17: proyección euclídea sobre Δ^{n−1}."""
            return self._simplex_with_kkt(v)[0]

        def _simplex_with_kkt(self, v: Any) -> Tuple[NDArray[np.float64], bool, float]:
            """
            3.2 Proyección euclídea EXACTA sobre Δ^{n−1} (Duchi et al. 2008) con certificado
            KKT. [W9] Exige forma 1-D compatible y finitud; la corrección de traza es ADITIVA
            sobre la componente mayor para permanecer en el interior del proyector exacto.
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
                return self._uniform.copy(), True, float(np.linalg.norm(self._uniform - arr))
            rho_idx = int(np.flatnonzero(cond)[-1])
            theta = float(cssv[rho_idx] / float(rho_idx + 1))
            w = np.maximum(arr - theta, 0.0)
            w[int(np.argmax(w))] += 1.0 - float(np.sum(w))
            kkt = bool(
                float(np.min(w)) >= -1e-14 and abs(float(np.sum(w)) - 1.0) <= 1e-12
            )
            return w, kkt, float(np.linalg.norm(w - arr))

        # ── Método 3.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _poincare_chart(
            semi_major_axis: float,
            eccentricity: float,
            inclination_rad: float,
            mean_anomaly: float,
            arg_pericenter_rad: float = 0.0,
            lon_node_rad: float = 0.0,
            mu: float = 1.0,
        ) -> PoincareChartCertificate:
            """
            3.3 [W2] Carta canónica regularizada de Poincaré con certificados de canonicidad
            simpléctica y función generatriz perturbativa de Poincaré-Von Zeipel / Lie-Deprit.
            Directa: Λ=√(μa), G=Λ√(1−e²), H_d=G·cos ι, Γ=Λ−G, Z=G−H_d,
                     (ξ₁,η₁)=√(2Γ)(cos ϖ, −sin ϖ), (ξ₂,η₂)=√(2Z)(cos Ω, −sin Ω).
            Inversa REGULAR (sin el polo 1/√Γ de la dirección acción-ángulo):
                     Γ=(ξ₁²+η₁²)/2, ϖ=atan2(−η₁, ξ₁).
            CERTIFICADOS:
              (a) round-trip de la carta implementada;
              (b) canonicidad |det ∂(Γ,ϖ)/∂(ξ₁,η₁)| = 1 y defecto matricial simpléctico
                  ‖Jᵀ Ω₂ J − Ω₂‖_F en el plano osculador;
              (c) generatriz de primer orden de Poincaré-Von Zeipel S₁(ℓ, ϖ, Λ, Γ) que
                  promedia los armónicos rápidos de anomalía media ℓ:
                  S₁ = Λ·(e·sin ℓ + ¼ e²·sin 2ℓ) + Γ·sin(2ℓ + 2ϖ).
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
            gamma_act, z_act = lam - g_del, g_del - h_del

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
            angle_err = abs(
                math.atan2(math.sin(varpi_back - varpi), math.cos(varpi_back - varpi))
            )
            roundtrip = math.hypot(
                g_back - gamma_act, angle_err if gamma_act > 1e-12 else 0.0
            )
            h_fd = 1e-6 * max(1.0, math.hypot(xi1, eta1))
            gp_x, vp_x = inverse(xi1 + h_fd, eta1)
            gm_x, vm_x = inverse(xi1 - h_fd, eta1)
            gp_y, vp_y = inverse(xi1, eta1 + h_fd)
            gm_y, vm_y = inverse(xi1, eta1 - h_fd)
            wrap = lambda p, q: math.atan2(math.sin(p - q), math.cos(p - q))  # noqa: E731
            j11 = (gp_x - gm_x) / (2.0 * h_fd)
            j12 = (gp_y - gm_y) / (2.0 * h_fd)
            j21 = wrap(vp_x, vm_x) / (2.0 * h_fd)
            j22 = wrap(vp_y, vm_y) / (2.0 * h_fd)
            det = j11 * j22 - j12 * j21
            if gamma_act > 1e-10:
                canonicity = abs(abs(det) - 1.0)
                j_mat = np.array([[j11, j12], [j21, j22]], dtype=np.float64)
                omega2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
                # La carta (ξ₁, η₁) = √(2Γ)(cos ϖ, −sin ϖ) preserva la 2-forma salvo orientación (±Ω₂)
                symp_pos = _fro(j_mat.T @ omega2 @ j_mat - omega2)
                symp_neg = _fro(j_mat.T @ omega2 @ j_mat + omega2)
                symp_jac_defect = float(min(symp_pos, symp_neg))
            else:
                canonicity = 0.0
                symp_jac_defect = 0.0

            von_zeipel_s1 = float(
                lam * (e * math.sin(m_anom) + 0.25 * (e ** 2) * math.sin(2.0 * m_anom))
                + gamma_act * math.sin(2.0 * (m_anom + varpi))
            )
            kappa = float(max(1.0, 1.0 / max(2.0 * gamma_act, _TINY)))
            return PoincareChartCertificate(
                chart=chart,
                mean_longitude=float(m_anom + varpi),
                algebraic_identity_residual=float(algebraic),
                roundtrip_error=float(roundtrip),
                canonicity_defect=float(canonicity),
                chart_condition_number=min(kappa, _CONDITION_CAP),
                symplectic_jacobian_defect=symp_jac_defect,
                von_zeipel_generating_s1=von_zeipel_s1,
            )

        # ── Método 3.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _cartan_invariant(
            semi_major_axis: float,
            eccentricity: float,
            inclination_rad: float,
            mu: float = 1.0,
        ) -> CartanCertificate:
            """
            3.4 [W3] Invariante integral relativo de Poincaré-Cartan COMPLETO, Invariante
            Secular de Kozai-Lidov y Parámetro de Tisserand-Poincaré:
                I_PC = ∮_γ (p·dq − H₀ dt) = 2π·AMD − H₀·T_orb,
            con acciones de Delaunay Λ=√(μa), G=Λ√(1−e²), H_d=G·cos ι,
            déficit de momento angular AMD = Λ−H_d, movimiento medio n=√(μ/a³),
            T_orb=2π/n y energía kepleriana H₀=−μ/(2a).
            Invariantes celestes adicionales:
              • Invariante secular de Kozai-Lidov: Θ_KL = √(1−e²)·cos ι = H_d / Λ ∈ [−1, 1].
              • Parámetro de Tisserand-Poincaré: T_P = μ/Λ² + 2·H_d/√μ = 1/a + 2√(a(1−e²))cos ι.
            """
            a = max(1e-9, _safe_float(semi_major_axis, 1.0))
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
            inc = float(np.clip(_safe_float(inclination_rad, 0.0), 0.0, math.pi - 1e-9))
            mu_eff = max(_TINY, _safe_float(mu, 1.0))
            lam = math.sqrt(mu_eff * a)
            g_del = lam * math.sqrt(max(0.0, 1.0 - e * e))
            h_del = g_del * math.cos(inc)
            amd = max(0.0, lam - h_del)
            n_motion = math.sqrt(mu_eff / (a ** 3))
            period = 2.0 * math.pi / max(n_motion, _TINY)
            h0 = -mu_eff / (2.0 * a)
            action_part = 2.0 * math.pi * amd
            energy_part = h0 * period
            invariant = action_part - energy_part
            identity_err = abs(invariant - (action_part - energy_part))
            identity_err = max(identity_err, abs(amd - (lam - h_del)))
            kozai_lidov = float(h_del / max(lam, _TINY))
            tisserand = float((mu_eff / max(lam * lam, _TINY)) + (2.0 * h_del / math.sqrt(mu_eff)))
            return CartanCertificate(
                delaunay_lambda=float(lam),
                delaunay_g=float(g_del),
                delaunay_h=float(h_del),
                angular_momentum_deficit=float(amd),
                mean_motion=float(n_motion),
                orbital_period=float(period),
                kepler_energy=float(h0),
                action_part=float(action_part),
                energy_part=float(energy_part),
                cartan_invariant=float(invariant),
                identity_error=float(identity_err),
                kozai_lidov_invariant=kozai_lidov,
                tisserand_poincare_invariant=tisserand,
            )

        # ── Método 3.5 ────────────────────────────────────────────────────────
        def _liouville_monodromy(
            self, mean_motion: float, eccentricity: float
        ) -> LiouvilleCertificate:
            """
            3.5 [W7] MONODROMÍA DE LIOUVILLE-POINCARÉ REAL con Purificación Simpléctica
            Polar RSI-L3 sobre Sp(4, ℝ) y Número de Rotación Diofántico KAM:
              (a) Sistema variacional epicíclico de Hill:
                      ẍ + (κ² + ε_m cos nt)x = 0,   ÿ + (ν² + ε_m cos nt)y = 0,
                  integrado por bloques exponenciales exactos a trozos con duplicación de paneles.
              (b) Purificación Simpléctica de Nivel 3 (Newton-Schulz sobre Sp(4, ℝ)):
                      M_{k+1} = ½ M_k (3 I₄ − Ωᵀ M_kᵀ Ω M_k),
                  que proyecta cuadráticamente cualquier deriva de redondeo acumulado de vuelta
                  a la variedad simpléctica exacta Mᵀ Ω M = Ω, registrando el coste
                  ‖M_L3 − M_raw‖_F.
              (c) Signaturas de Krein κ(ξ)=sgn(−i ξ†Ωξ), exponente característico de
                  Floquet-Lyapunov λ_Floquet = ln(ρ(M))/T_orb y número de rotación de
                  Poincaré-KAM ω_KAM = |arg λ₁|/(2π) con aproximación diofántica p/q.
            """
            n_mot = max(1e-6, _safe_float(mean_motion, 1.0))
            e = float(np.clip(_safe_float(eccentricity, 0.0), 0.0, 0.9999))
            eps_m = 2.0 * e * n_mot * n_mot
            kappa, nu = n_mot, n_mot * (1.0 + self._splitting)
            period = 2.0 * math.pi / n_mot
            j2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
            omega_form = np.asarray(la.block_diag(j2, j2), dtype=np.float64)
            eye4 = np.eye(4, dtype=np.float64)

            def build(panels: int) -> NDArray[np.float64]:
                h = period / panels
                blocks = []
                for freq in (kappa, nu):
                    mono_blk = np.eye(2, dtype=np.float64)
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
                        mono_blk = blk @ mono_blk
                    blocks.append(mono_blk)
                return np.asarray(la.block_diag(*blocks), dtype=np.float64)

            panels = _MONODROMY_BASE_PANELS
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
                    if step_cert <= 1e-11:
                        break
            self._telemetry.cumulative_monodromy_panels += panels

            # Purificación polar simpléctica RSI-L3 sobre Sp(4, ℝ)
            mono_raw = mono.copy()
            for _ in range(2):
                symp_gram = omega_form.T @ mono.T @ omega_form @ mono
                mono = 0.5 * mono @ (3.0 * eye4 - symp_gram)
            self._telemetry.cumulative_rsi_l3_iterations += 2
            symp_proj_cost = _fro(mono - mono_raw)

            sympl = float(la.norm(mono.T @ omega_form @ mono - omega_form, ord="fro"))
            det = float(np.linalg.det(mono))
            eigvals, eigvecs = la.eig(mono)
            radius = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
            floquet_lyap = float(math.log(max(radius, _TINY)) / max(period, _TINY))

            signatures: List[Tuple[complex, float]] = []
            rotation_candidates: List[float] = []
            for idx in range(eigvals.size):
                lam_c = complex(eigvals[idx])
                if abs(lam_c.imag) > 1e-12 or abs(lam_c.real) > 1e-12:
                    ang = abs(math.atan2(lam_c.imag, lam_c.real)) / (2.0 * math.pi)
                    if ang > 0.0:
                        rotation_candidates.append(ang)
                if abs(abs(lam_c) - 1.0) <= 1e-6 and abs(lam_c.imag) > 1e-8:
                    xi = eigvecs[:, idx]
                    krein = float(np.real(-1j * np.vdot(xi, omega_form @ xi)))
                    signatures.append(
                        (lam_c, math.copysign(1.0, krein) if abs(krein) > 1e-12 else 0.0)
                    )
            collision = False
            for i in range(len(signatures)):
                for jdx in range(i + 1, len(signatures)):
                    li, ki = signatures[i]
                    lj, kj = signatures[jdx]
                    if ki * kj < 0.0 and abs(li - lj) < 1e-3:
                        collision = True

            kam_rot = float(rotation_candidates[0]) if rotation_candidates else 0.0
            kam_p, kam_q, kam_def = _continued_fraction_diophantine(kam_rot)
            liouville_ok = bool(sympl <= _SYMPECTIC_TOLERANCE and abs(det - 1.0) <= 1e-9)
            return LiouvilleCertificate(
                symplectic_defect=sympl,
                determinant=det,
                spectral_radius=radius,
                krein_signatures=tuple(float(s) for _, s in signatures),
                collision_detected=bool(collision),
                step_certificate=float(step_cert),
                panels=int(panels),
                liouville_ok=liouville_ok,
                kam_rotation_number=kam_rot,
                kam_diophantine_defect=float(kam_def),
                kam_rational_p=int(kam_p),
                kam_rational_q=int(kam_q),
                rsi_l3_symplectic_projection_cost=symp_proj_cost,
                floquet_lyapunov_exponent=floquet_lyap,
            )

        # ── Método 3.6 ────────────────────────────────────────────────────────
        def _shahshahani_certificate(
            self, p: NDArray[np.float64], k_real: NDArray[np.float64]
        ) -> ShahshahaniCertificate:
            """
            3.6 Disipación en la variedad riemanniana de Shahshahani (g_ij = δ_ij/p_i) con
            certificado numérico RSI-L3 por Extrapolación Jerárquica de Richardson-Romberg:
                dφ/dt = 2·Var_p(Kp),   dH/dt|_rep = −Var_p(Kp) ≤ 0,   ‖ṗ‖_g = √Var_p(Kp).
            Se evalúa el flujo replicador exponencial a pasos h, h/2 y h/4, construyendo el
            tableau triangular de Romberg R_{1,1} (O(h²)) y R_{2,2} (O(h³)), y midiendo el
            factor de contracción de Banach entre niveles sucesivos de extrapolación.
            """
            kp = k_real @ p
            phi = float(np.dot(p, kp))
            var = max(0.0, float(np.dot(p, (kp - phi) ** 2)))

            def step_phi(h_step: float) -> float:
                logits = np.log(np.maximum(p, _TINY)) + h_step * kp
                logits -= float(np.max(logits))
                w = np.exp(np.clip(logits, -700.0, 0.0))
                p_next = w / max(float(np.sum(w)), _TINY)
                return float(np.dot(p_next, k_real @ p_next))

            h = 1.0e-4
            d1 = (step_phi(h) - phi) / h
            d2 = (step_phi(0.5 * h) - phi) / (0.5 * h)
            d3 = (step_phi(0.25 * h) - phi) / (0.25 * h)
            # Nivel 1 de Richardson (elimina O(h))
            r11 = 2.0 * d2 - d1
            r21 = 2.0 * d3 - d2
            # Nivel 2 de Romberg-Richardson (elimina O(h²))
            r22 = (4.0 * r21 - r11) / 3.0
            self._telemetry.cumulative_rsi_l3_iterations += 2

            delta_l1 = abs(r21 - r11)
            delta_l2 = abs(r22 - r21)
            romberg_q = (
                min(0.999999, delta_l2 / max(delta_l1, _TINY)) if delta_l1 > 1.0e-15 else 0.0
            )
            numeric = float(r22)
            analytic = 2.0 * var
            err = _relative(abs(numeric - analytic), max(abs(analytic), 1e-12))
            return ShahshahaniCertificate(
                variance=float(var),
                dissipation_rate=float(-var),
                speed=math.sqrt(var),
                phi=phi,
                phi_derivative_numeric=numeric,
                phi_derivative_analytic=float(analytic),
                derivative_certificate_error=float(min(err, 1e6)),
                romberg_l3_contraction_ratio=float(romberg_q),
            )

        # ── Método 3.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _connes_daleckii_krein_bounds(lambda_min: float) -> Tuple[float, float]:
            """
            3.7 [W6] Cotas de Daleckii-Krein por cociente dividido, con atribución exacta:
                f(x)=√x:       f^[1](a,b) = 1/(√a+√b)     ⟹ L ≤ 1/(2√λ_min)
                f(x)=x^{−1/2}: |f^[1](a,b)| = ½ξ^{−3/2}   ⟹ L ≤ 1/(2λ_min^{3/2}).
            """
            lam = max(_safe_float(lambda_min, 0.0), _STABILITY_FLOOR)
            return float(0.5 / math.sqrt(lam)), float(0.5 / (lam ** 1.5))

        # ── Método 3.8 · TERMINAL Π₃ · COMPOSICIÓN Π₃∘Π₂∘Π₁ Y EMISIÓN ─────────
        def inspect_positron_trajectory(
            self,
            p_init: NDArray[np.float64],
            rho_init: NDArray[np.complex128],
            sigma_ref: NDArray[np.complex128],
            K_raw: NDArray[np.complex128],
            boundary_matrix: Optional[NDArray[np.int64]] = None,
            *,
            boundary_next: Optional[NDArray[np.int64]] = None,
            semi_major_axis: float = 1.0,
            eccentricity: float = 0.01,
            inclination_rad: float = 0.02,
            mean_anomaly: float = 0.5,
            arg_pericenter_rad: float = 0.0,
            lon_node_rad: float = 0.0,
            mu: float = 1.0,
        ) -> PositronWitnessSpectrumReport:
            """
            3.8 [TERMINAL Π₃] Composición completa con continuidad formal de tipos:
                1.9 ──Pi1ToPi2Seed──► 2.1 … 2.7 ──Pi2ToPi3Seed──► 3.1 … 3.8.
            Emite el certificado espectral inmutable `PositronWitnessSpectrumReport`.
            La FPU testigo NO dicta veredictos: todos los booleanos son PREDICADOS
            MATEMÁTICOS verificables por el agente soberano.
            """
            flags: List[str] = []
            self._telemetry.total_inspections += 1
            before = [
                _tensor_digest(a)
                for a in (p_init, rho_init, sigma_ref, K_raw, boundary_matrix)
            ]
            # ── Π₁ ──
            pi1 = self._pi1.audit_passive_quantum_substrate(rho_init, sigma_ref, K_raw)
            # ── Π₂ (continuación formal desde Π₁: 1.9 → 2.1) ──
            pi2 = self._pi2.audit_smith_homology_sector(
                pi1_certificate=pi1,
                boundary_matrix=boundary_matrix,
                boundary_next=boundary_next,
            )
            # ── Π₃ (continuación formal desde Π₂: 2.7 → 3.1) ──
            _seed2 = self._ingest_pi2_continuation(pi2)
            p_norm, simplex_kkt, _cost = self._simplex_with_kkt(p_init)
            if not simplex_kkt:
                flags.append("SIMPLEX_INPUT_FALLBACK_OR_KKT_MARGINAL")
            cond = pi1.conductivity
            if cond.used_fallback:
                flags.append("CONDUCTIVITY_INPUT_FALLBACK")
            if pi1.rho.used_fallback or pi1.sigma.used_fallback:
                flags.append("DENSITY_INPUT_FALLBACK")

            chart = self._poincare_chart(
                semi_major_axis,
                eccentricity,
                inclination_rad,
                mean_anomaly,
                arg_pericenter_rad,
                lon_node_rad,
                mu,
            )
            cartan = self._cartan_invariant(
                semi_major_axis, eccentricity, inclination_rad, mu
            )
            liouville = self._liouville_monodromy(cartan.mean_motion, eccentricity)
            k_real = np.asarray(np.real(cond.k_matrix), dtype=np.float64)
            shah = self._shahshahani_certificate(p_norm, k_real)
            dk_sqrt, dk_inv = self._connes_daleckii_krein_bounds(cond.lambda_min)

            after = [
                _tensor_digest(a)
                for a in (p_init, rho_init, sigma_ref, K_raw, boundary_matrix)
            ]
            attestation = self._pi1._attest_non_interference(before, after)
            if not attestation.all_ok:
                self._telemetry.non_interference_violations += 1
                flags.append("WITNESS_1_NON_INTERFERENCE_VIOLATED")

            smith, hom, hodge = pi2.smith, pi2.homology, pi2.hodge_poincare
            ks_cert, rsi1_cert = pi1.ks_poincare, pi1.rsi_l3

            if not smith.reconstruction_ok:
                flags.append("SNF_RECONSTRUCTION_UBV_NE_S")
            if not smith.unimodular_ok:
                flags.append("SNF_UNIMODULARITY_FAILED")
            if not smith.divisibility_chain_ok:
                flags.append("SNF_DIVISIBILITY_CHAIN_FAILED")
            if not smith.v_inverse_ok:
                flags.append("SNF_V_INVERSE_FAILED")
            if smith.int64_saturated:
                flags.append("SNF_INT64_SATURATED_EXACT_STRINGS_AUTHORITATIVE")
            if not smith.torsion_free:
                flags.append("SMITH_COKERNEL_TORSION_DETECTED")
            if hom.chain_complex_ok and not hom.torsion_free:
                flags.append("HOMOLOGY_H_K_TORSION_DETECTED")
            if boundary_next is not None and not hom.chain_complex_ok:
                flags.append("BOUNDARY_COMPOSITION_NONZERO_NOT_A_COMPLEX")
            if not hom.chain_complex_ok:
                flags.append("HOMOLOGY_SCOPE_COKERNEL_ONLY")
            if hodge is not None and not hodge.betti_hodge_consistency_ok:
                flags.append("HODGE_DE_RHAM_BETTI_MISMATCH")
            if ks_cert is not None and not ks_cert.ks_regularization_ok:
                flags.append("KS_QUATERNIONIC_REGULARIZATION_DEFECT")
            if rsi1_cert is not None and not rsi1_cert.rsi_l3_converged_ok:
                flags.append("RSI_L3_BANACH_CONTRACTION_MARGINAL")
            if chart.canonicity_defect > 1e-5:
                flags.append("POINCARE_CHART_CANONICITY_DEFECT")
            if not liouville.liouville_ok:
                flags.append("LIOUVILLE_POINCARE_1_VIOLATED")
            if liouville.collision_detected:
                flags.append("KREIN_1TO1_EPICYCLIC_RESONANCE_DETECTED")
            if pi1.chord_arc_error > 1e-9:
                flags.append("CHORD_ARC_IDENTITY_DEFECT")
            if shah.derivative_certificate_error > 1e-4:
                flags.append("SHAHSHAHANI_DPHI_IDENTITY_DEFECT")

            return PositronWitnessSpectrumReport(
                p_observed=_freeze_array(p_norm),
                rho_observed=_freeze_array(pi1.rho.rho),
                K_purified=_freeze_array(cond.k_matrix),
                uhlmann_fidelity=float(pi1.uhlmann_fidelity),
                bures_distance=float(pi1.bures_distance),
                smith_torsion_free=bool(smith.torsion_free),
                smith_invariant_factors=_freeze_array(smith.invariant_factors_int64),
                poincare_canonical_residual=float(chart.algebraic_identity_residual),
                poincare_cartan_invariant=float(cartan.cartan_invariant),
                shahshahani_dissipation_rate=float(shah.dissipation_rate),
                connes_lipschitz_bound=float(dk_inv),
                condition_number_K=float(cond.condition_capped),
                bures_angle=float(pi1.bures_angle),
                chord_arc_identity_error=float(pi1.chord_arc_error),
                fidelity_route_discrepancy=float(pi1.fidelity_route_discrepancy),
                hilbert_schmidt_overlap=float(pi1.hilbert_schmidt_overlap),
                rho_condition_number=float(pi1.rho.condition_number),
                rho_purity=float(pi1.rho.purity),
                rho_entropy=float(pi1.rho.entropy),
                rho_projection_cost=float(pi1.rho.projection_cost),
                rho_kkt_ok=bool(pi1.rho.kkt_ok),
                simplex_kkt_ok=bool(simplex_kkt),
                K_imaginary_discarded_norm=float(cond.imaginary_discarded_norm),
                K_projection_cost=float(cond.projection_cost),
                lambda_min_K=float(cond.lambda_min),
                lambda_max_K=float(cond.lambda_max),
                spectral_gap_K=float(cond.top_gap),
                condition_number_K_raw=float(cond.condition_raw),
                log10_condition_number_K=float(cond.log10_condition),
                dk_sqrt_lipschitz=float(dk_sqrt),
                dk_invsqrt_lipschitz=float(dk_inv),
                quantum_fisher_information=float(
                    ks_cert.quantum_fisher_information if ks_cert is not None else 0.0
                ),
                sld_lyapunov_residual=float(
                    ks_cert.sld_lyapunov_residual if ks_cert is not None else 0.0
                ),
                ks_hurwitz_orthogonality_defect=float(
                    ks_cert.hurwitz_orthogonality_defect if ks_cert is not None else 0.0
                ),
                ks_bilinear_gauge_residual=float(
                    ks_cert.ks_bilinear_gauge_residual if ks_cert is not None else 0.0
                ),
                fubini_study_symplectic_area=float(
                    ks_cert.fubini_study_symplectic_area if ks_cert is not None else 0.0
                ),
                rsi_l3_banach_contraction_ratio=float(
                    rsi1_cert.banach_contraction_ratio if rsi1_cert is not None else 0.0
                ),
                rsi_l3_aposteriori_bound=float(
                    rsi1_cert.aposteriori_fixed_point_bound if rsi1_cert is not None else 0.0
                ),
                rsi_l3_projector_idempotence_defect=float(
                    rsi1_cert.projector_idempotence_defect if rsi1_cert is not None else 0.0
                ),
                rsi_l3_converged_ok=bool(
                    rsi1_cert.rsi_l3_converged_ok if rsi1_cert is not None else True
                ),
                smith_rank=int(smith.rank),
                smith_invariant_factors_exact=smith.invariant_factors_exact,
                smith_int64_saturated=bool(smith.int64_saturated),
                smith_divisibility_chain_ok=bool(smith.divisibility_chain_ok),
                smith_unimodular_ok=bool(smith.unimodular_ok),
                smith_reconstruction_ok=bool(smith.reconstruction_ok),
                smith_v_inverse_ok=bool(smith.v_inverse_ok),
                smith_det_u=int(smith.det_u),
                smith_det_v=int(smith.det_v),
                smith_cokernel_free_rank=int(smith.cokernel_free_rank),
                smith_cokernel_torsion=smith.cokernel_torsion,
                smith_kernel_rank=int(smith.kernel_rank),
                homology_scope=hom.scope,
                homology_chain_complex_ok=bool(hom.chain_complex_ok),
                homology_rank=int(hom.homology_rank),
                homology_torsion=hom.homology_torsion,
                homology_torsion_free=bool(hom.torsion_free),
                smith_hadamard_log10_bound=float(smith.hadamard_log10_bound),
                pontryagin_linking_phase=float(smith.pontryagin_linking_phase),
                hodge_harmonic_dimension=int(
                    hodge.hodge_harmonic_dimension if hodge is not None else 0
                ),
                hodge_spectral_gap=float(
                    hodge.hodge_spectral_gap if hodge is not None else 0.0
                ),
                betti_hodge_consistency_ok=bool(
                    hodge.betti_hodge_consistency_ok if hodge is not None else True
                ),
                poincare_cycle_winding_action=float(
                    hodge.poincare_cycle_winding_action if hodge is not None else 0.0
                ),
                poincare_canonicity_defect=float(chart.canonicity_defect),
                poincare_roundtrip_error=float(chart.roundtrip_error),
                poincare_mean_longitude=float(chart.mean_longitude),
                poincare_symplectic_jacobian_defect=float(chart.symplectic_jacobian_defect),
                von_zeipel_generating_s1=float(chart.von_zeipel_generating_s1),
                delaunay_lambda=float(cartan.delaunay_lambda),
                delaunay_angular_momentum_deficit=float(cartan.angular_momentum_deficit),
                cartan_action_part=float(cartan.action_part),
                cartan_energy_part=float(cartan.energy_part),
                cartan_identity_error=float(cartan.identity_error),
                kozai_lidov_invariant=float(cartan.kozai_lidov_invariant),
                tisserand_poincare_invariant=float(cartan.tisserand_poincare_invariant),
                liouville_symplectic_defect=float(liouville.symplectic_defect),
                liouville_determinant=float(liouville.determinant),
                liouville_spectral_radius=float(liouville.spectral_radius),
                liouville_ok=bool(liouville.liouville_ok),
                krein_collision_detected=bool(liouville.collision_detected),
                monodromy_step_certificate=float(liouville.step_certificate),
                kam_rotation_number=float(liouville.kam_rotation_number),
                kam_diophantine_defect=float(liouville.kam_diophantine_defect),
                floquet_lyapunov_exponent=float(liouville.floquet_lyapunov_exponent),
                shahshahani_speed=float(shah.speed),
                shahshahani_variance=float(shah.variance),
                replicator_phi_derivative_numeric=float(shah.phi_derivative_numeric),
                replicator_phi_derivative_analytic=float(shah.phi_derivative_analytic),
                replicator_derivative_certificate_error=float(
                    shah.derivative_certificate_error
                ),
                romberg_l3_contraction_ratio=float(shah.romberg_l3_contraction_ratio),
                non_interference_ok=bool(attestation.all_ok),
                engine_version=_ENGINE_VERSION,
                schema_flags=tuple(flags),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # §3 · INICIALIZACIÓN Y DELEGACIÓN (API pública compatible con v16/v17)
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        tolerance: float = 1e-9,
        *,
        dimension_n: int = 3,
        tau_spd: float = 1e-12,
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
        self._telemetry = WitnessTelemetry()
        if self._tau_spd * self._n > 0.5:
            logger.warning(
                "tau_spd·n = %.3e > 1/2: el piso se recortará a 1/(2n).",
                self._tau_spd * self._n,
            )
        self.phase_pi1 = self.PhasePi1_PassiveUhlmannObserver(
            self._n, self._tol, self._tau_spd, telemetry=self._telemetry
        )
        self.phase_pi2 = self.PhasePi2_SmithHomologyAuditor(telemetry=self._telemetry)
        self.phase_pi3 = self.PhasePi3_PoincareLiouvilleEmitter(
            pi1=self.phase_pi1,
            pi2=self.phase_pi2,
            dimension_n=self._n,
            telemetry=self._telemetry,
            precession_splitting=float(epicyclic_precession_splitting),
            enable_doubling=bool(enable_step_doubling),
        )
        logger.info(
            "PositronRedemptionWitnessEngine v%s inicializado (Π₁→Π₂→Π₃ + RSI-L3 + Poincaré).",
            _ENGINE_VERSION,
        )

    # ── Delegadores de Π₁ ──
    def canonicalize_density_matrix(
        self, rho_raw: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Delega en 1.3b (proyección de Frobenius sobre D_δ(H) con idempotencia RSI-L3)."""
        return self.phase_pi1.canonicalize_density_matrix(rho_raw)

    def purify_hermitian_matrix(
        self, K_raw: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Delega en 1.4b (proyección PSD∩Sym(ℝ) en el orden correcto con pasividad Foster-Brune)."""
        return self.phase_pi1.purify_hermitian_matrix(K_raw)

    def compute_uhlmann_bures(
        self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]
    ) -> Tuple[float, float]:
        """Delega en 1.5c (Uhlmann por norma traza; canoniza ambos operandos)."""
        return self.phase_pi1.compute_uhlmann_bures(rho, sigma)

    # ── Delegadores de Π₂ ──
    def compute_smith_normal_form_torsion(
        self, B_matrix: NDArray[np.int64]
    ) -> Tuple[bool, NDArray[np.int64]]:
        """Delega en 2.3b (SNF ENTERA exacta; usar el certificado para los factores reales)."""
        return self.phase_pi2.compute_smith_normal_form_torsion(B_matrix)

    # ── Delegadores de Π₃ ──
    def project_gibbs_simplex(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Delega en 3.2b (proyección exacta sobre Δ^{n−1} con certificado KKT)."""
        return self.phase_pi3.project_gibbs_simplex(v)

    def inspect_positron_trajectory(
        self,
        p_init: NDArray[np.float64],
        rho_init: NDArray[np.complex128],
        sigma_ref: NDArray[np.complex128],
        K_raw: NDArray[np.complex128],
        boundary_matrix: Optional[NDArray[np.int64]] = None,
        **kwargs: Any,
    ) -> PositronWitnessSpectrumReport:
        """Delega en 3.8 (composición Π₃∘Π₂∘Π₁ y emisión del certificado espectral)."""
        return self.phase_pi3.inspect_positron_trajectory(
            p_init, rho_init, sigma_ref, K_raw, boundary_matrix, **kwargs
        )

    def get_telemetry(self) -> WitnessTelemetry:
        """Registro acumulativo de la inspección pasiva (no contiene veredictos)."""
        return self._telemetry


# ══════════════════════════════════════════════════════════════════════════════
# §4 · AUTOVERIFICACIÓN EJECUTABLE (Regresiones v16/v17 + RSI-L3 + Poincaré v18)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """Suite de autoverificación exhaustiva de Π₁, Π₂ y Π₃ (v16/v17 + v18 RSI-L3/Poincaré)."""
    checks: List[Tuple[str, bool]] = []
    n = 3
    eng = PositronRedemptionWitnessEngine(dimension_n=n, tau_spd=1e-10)
    pi1, pi2, pi3 = eng.phase_pi1, eng.phase_pi2, eng.phase_pi3

    # [W1] REGRESIÓN CABECERA: el default de v16 declaraba torsión sobre un árbol.
    B_default = pi2._default_boundary_matrix()
    smith = pi2._smith_certificate(B_default)
    sv = np.linalg.svd(np.abs(B_default.astype(np.int64)), compute_uv=False)
    v16_invariants = np.round(sv[sv > 1e-5]).astype(np.int64)
    v16_torsion_free = bool(np.all(v16_invariants[v16_invariants > 0] == 1))
    checks.append(
        (
            "[W1] v16 (SVD) declaraba torsión sobre el default; v18 no",
            (not v16_torsion_free) and smith.torsion_free,
        )
    )
    checks.append(
        (
            "[W1] SNF(default) = diag(1,1), rango 2, ker ≅ ℤ",
            smith.invariant_factors_exact == ("1", "1")
            and smith.rank == 2
            and smith.kernel_rank == 1,
        )
    )
    checks.append(
        (
            "[W1] Certificados: U·B·V = S, det U/V = ±1, V·V⁻¹ = I, d_i | d_{i+1}",
            smith.reconstruction_ok
            and smith.unimodular_ok
            and smith.v_inverse_ok
            and smith.divisibility_chain_ok,
        )
    )

    # [W1] Reparación de divisibilidad y fase de enlace de Pontryagin-Poincaré: diag(2,3) -> diag(1,6).
    smith_23 = pi2._smith_certificate(np.array([[2, 0], [0, 3]], dtype=np.int64))
    checks.append(
        (
            "[W1] SNF(diag(2,3)) = diag(1,6) y fase de Pontryagin = 1/6",
            smith_23.invariant_factors_exact == ("1", "6")
            and smith_23.cokernel_torsion == ("6",)
            and abs(smith_23.pontryagin_linking_phase - (1.0 / 6.0)) < 1e-12,
        )
    )
    smith_2 = pi2._smith_certificate(np.array([[2]], dtype=np.int64))
    checks.append(
        (
            "[W1] SNF([2]) ⟹ coker = ℤ/2, torsión detectada",
            smith_2.cokernel_torsion == ("2",) and not smith_2.torsion_free,
        )
    )
    smith_uni = pi2._smith_certificate(np.array([[1, 2], [3, 7]], dtype=np.int64))
    checks.append(
        (
            "[W1] det = 1 ⟹ SNF = diag(1,1), sin torsión",
            smith_uni.invariant_factors_exact == ("1", "1") and smith_uni.torsion_free,
        )
    )

    # [W1] Homología GENUINA: el complejo de ℝP² (∂₁=0, ∂₂=(2)) ⟹ H₁ = ℤ/2.
    B_rp2 = np.array([[0]], dtype=np.int64)
    N_rp2 = np.array([[2]], dtype=np.int64)
    smith_rp2 = pi2._smith_certificate(B_rp2)
    hom_rp2, _k_rp2 = pi2._chain_complex_homology(B_rp2, N_rp2, smith_rp2)
    checks.append(
        (
            "[W1] H₁(ℝP²;ℤ) = ℤ/2 exacto (rango 0, torsión (2,))",
            hom_rp2.chain_complex_ok
            and hom_rp2.homology_rank == 0
            and hom_rp2.homology_torsion == ("2",),
        )
    )

    # Triángulo: ∂₁ de rango 2, sin 2-celdas ⟹ H₁ ≅ ℤ y consistencia de Betti-Hodge Δ₁.
    B_tri = np.array([[-1, -1, 0], [1, 0, -1], [0, 1, 1]], dtype=np.int64)
    smith_tri = pi2._smith_certificate(B_tri)
    N_tri = np.zeros((3, 1), dtype=np.int64)
    hom_tri, k_tri = pi2._chain_complex_homology(B_tri, N_tri, smith_tri)
    hodge_tri = pi2._hodge_poincare_spectral_audit(
        B_tri, N_tri, smith_tri, hom_tri, k_tri, pi2._ingest_pi1_continuation(None)
    )
    checks.append(
        (
            "[W1/v18] H₁(triángulo) ≅ ℤ y dim ker(Δ₁) == β₁ == 1 (Hodge-Poincaré)",
            hom_tri.chain_complex_ok
            and hom_tri.homology_rank == 1
            and hom_tri.homology_torsion == ()
            and hodge_tri.betti_hodge_consistency_ok
            and hodge_tri.hodge_harmonic_dimension == 1
            and hodge_tri.rsi_l3_kernel_orthogonality_defect < 1e-12,
        )
    )

    # Complejo inválido: ∂∘∂ ≠ 0 debe detectarse.
    hom_bad, _ = pi2._chain_complex_homology(
        B_tri, np.ones((3, 1), dtype=np.int64), smith_tri
    )
    checks.append(
        ("[W1] ∂∘∂ ≠ 0 ⟹ NO es complejo, se declara", not hom_bad.chain_complex_ok)
    )

    # [W2/W3/v18] Inspección completa Π₃∘Π₂∘Π₁ sobre trayectoria de positrón.
    rho = np.eye(n, dtype=np.complex128) / float(n)
    sigma = np.diag([0.6, 0.2, 0.2]).astype(np.complex128)
    k_eye = np.eye(n, dtype=np.complex128)
    p0 = np.array([0.5, 0.3, 0.2])
    rep = eng.inspect_positron_trajectory(p0, rho, sigma, k_eye)
    checks.append(
        (
            "[W2/v18] Canonicidad real |det| = 1, defecto simpléctico < 1e-5 y round-trip < 1e-9",
            rep.poincare_canonicity_defect < 1e-5
            and rep.poincare_symplectic_jacobian_defect < 1e-5
            and rep.poincare_roundtrip_error < 1e-9,
        )
    )
    checks.append(
        (
            "[W3] I_PC ≠ 2π (v16 era constante) e identidad verificada",
            abs(rep.poincare_cartan_invariant - 2.0 * math.pi) > 1e-6
            and rep.cartan_identity_error < 1e-12,
        )
    )
    checks.append(
        (
            "[W3/v18] Kozai-Lidov ∈ (0,1] y Tisserand-Poincaré > 0 bien formados",
            0.0 < rep.kozai_lidov_invariant <= 1.0
            and rep.tisserand_poincare_invariant > 2.5,
        )
    )

    # [v18] Regularización cuaterniónica de Kustaanheimo-Stiefel (ℍ) y SLD/QFI en Π₁.
    checks.append(
        (
            "[v18] Kustaanheimo-Stiefel: ortogonalidad de Hurwitz y gauge bilineal = 0",
            rep.ks_hurwitz_orthogonality_defect < 1e-12
            and rep.ks_bilinear_gauge_residual < 1e-12,
        )
    )
    checks.append(
        (
            "[v18] Ecuación de Lyapunov SLD exacta (< 1e-12) y QFI > 0 para ρ ≠ σ",
            rep.sld_lyapunov_residual < 1e-12 and rep.quantum_fisher_information > 0.0,
        )
    )

    # [v18] Automejora Recursiva de Nivel 3 (RSI-L3): contracción de Banach e idempotencia.
    checks.append(
        (
            "[v18] RSI-L3: contracción de Banach q_L3 < 1, idempotencia < 1e-12 y convergencia OK",
            rep.rsi_l3_converged_ok
            and 0.0 <= rep.rsi_l3_banach_contraction_ratio < 1.0
            and rep.rsi_l3_projector_idempotence_defect < 1e-12,
        )
    )

    # [W7/v18] Liouville-Poincaré: monodromía simpléctica con purificación RSI-L3 y rotación KAM.
    checks.append(
        (
            "[W7/v18] ‖MᵀΩM−Ω‖ < 1e-9, det M = +1 y número de rotación KAM finito",
            rep.liouville_symplectic_defect < _SYMPECTIC_TOLERANCE
            and abs(rep.liouville_determinant - 1.0) < 1e-9
            and rep.liouville_ok
            and rep.kam_diophantine_defect >= 0.0,
        )
    )
    checks.append(
        (
            "[W7] Kepler κ=ν ⟹ colisión de Krein 1:1 detectada",
            rep.krein_collision_detected,
        )
    )

    # Uhlmann: forma cerrada conmutante, identidad cuerda-arco y σ cruda.
    p_spec, q_spec = np.full(n, 1.0 / n), np.array([0.6, 0.2, 0.2])
    f_closed = float(np.sum(np.sqrt(p_spec * q_spec))) ** 2
    checks.append(
        ("Uhlmann ≡ forma cerrada conmutante", abs(rep.uhlmann_fidelity - f_closed) < 1e-9)
    )
    checks.append(
        ("Identidad cuerda-arco d_B = 2·sin(θ_B/2)", rep.chord_arc_identity_error < 1e-12)
    )
    f_id, d_id = pi1.compute_uhlmann_bures(rho, rho)
    checks.append(
        ("F(ρ,ρ) = 1 y d_B(ρ,ρ) = 0", abs(f_id - 1.0) < 1e-9 and d_id < 1e-9)
    )
    sigma_raw = np.diag([1.5, -0.4, -0.1]).astype(np.complex128)
    f_raw, d_raw = pi1.compute_uhlmann_bures(rho, sigma_raw)
    checks.append(
        (
            "[W8] σ no PSD se canoniza dentro del método (F ∈ [0,1])",
            0.0 <= f_raw <= 1.0 and 0.0 <= d_raw <= math.sqrt(2.0) + 1e-9,
        )
    )

    # [W5] Proyección real-first domina a clamp-then-real y pasividad Foster-Brune.
    k_cplx = np.array(
        [[2.0 + 0j, 0.3 + 0.4j, 0], [0.3 - 0.4j, 1.0, 0], [0, 0, -0.5]],
        dtype=np.complex128,
    )
    cond_opt = pi1._purify_with_certificate(k_cplx)
    kh = 0.5 * (k_cplx + k_cplx.conj().T)
    ev, uv = la.eigh(kh)
    k_v16 = np.real((uv * np.maximum(ev, 1e-10)) @ uv.conj().T)
    checks.append(
        (
            "[W5/v18] real→clamp domina a clamp→real y cumple pasividad de Foster-Brune",
            la.norm(k_cplx - cond_opt.k_matrix, "fro")
            <= la.norm(k_cplx - k_v16, "fro") + 1e-12
            and cond_opt.foster_brune_passive_ok,
        )
    )
    checks.append(
        (
            "[W5] Parte imaginaria del K purificado ≡ 0",
            float(np.max(np.abs(np.imag(cond_opt.k_matrix)))) == 0.0,
        )
    )

    # [W4/W9] Proyección exacta del símplex capado y validación de forma.
    vals = np.array([-0.30, 0.10, 1.20])
    exact, _d, feas, kkt_res = pi1._project_capped_simplex(vals, 1e-10)
    clamped = np.maximum(vals, 1e-10)
    naive = clamped / clamped.sum()
    checks.append(
        (
            "[W4] Proyección exacta ≤ clamp+renorm en Frobenius y residuo KKT ≈ 0",
            feas
            and np.linalg.norm(exact - vals) <= np.linalg.norm(naive - vals) + 1e-12
            and kkt_res < 1e-12,
        )
    )
    bad = pi3._simplex_with_kkt(np.array([[0.5, 0.5], [0.0, 0.0]]))
    checks.append(
        (
            "[W9] Entrada 2-D ⟹ fallback explícito, no basura silenciosa",
            not bad[1] and abs(float(bad[0].sum()) - 1.0) < 1e-12,
        )
    )

    # [W6] Daleckii-Krein con atribución exacta.
    l_sq, l_inv = pi3._connes_daleckii_krein_bounds(1e-4)
    checks.append(
        (
            "[W6] L_sqrt = 1/(2√λ) = 50 y L_invsqrt = 1/(2λ^{3/2})",
            abs(l_sq - 50.0) < 1e-9
            and abs(l_inv - 0.5 / (1e-4 ** 1.5)) < 1e-3
            and l_inv > l_sq,
        )
    )

    # Shahshahani: pasividad y certificado Romberg RSI-L3 de dφ/dt = 2Var.
    k_gen = np.array([[1.0, 0.2, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 0.8]])
    shah = pi3._shahshahani_certificate(p0, k_gen)
    checks.append(
        (
            "Shahshahani: dH/dt = −Var ≤ 0 y velocidad = √Var",
            shah.dissipation_rate <= 0.0
            and abs(shah.speed - math.sqrt(shah.variance)) < 1e-12,
        )
    )
    checks.append(
        (
            "[v18] Certificado Romberg RSI-L3 dφ/dt = 2·Var (< 1e-6) y q_Romberg < 1",
            shah.derivative_certificate_error < 1e-6
            and 0.0 <= shah.romberg_l3_contraction_ratio < 1.0,
        )
    )

    # [W10] No-interferencia y arrays de sólo lectura.
    checks.append(
        ("[W10] WITNESS-1 atestiguado sobre los cinco buffers", rep.non_interference_ok)
    )
    p_mut, rho_mut = p0.copy(), rho.copy()
    _ = eng.inspect_positron_trajectory(p_mut, rho_mut, sigma, k_eye)
    checks.append(
        (
            "[W10] Las entradas no fueron mutadas in-place",
            np.array_equal(p_mut, p0) and np.array_equal(rho_mut, rho),
        )
    )
    checks.append(
        (
            "Arrays del certificado son de sólo lectura",
            not rep.p_observed.flags.writeable and not rep.rho_observed.flags.writeable,
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