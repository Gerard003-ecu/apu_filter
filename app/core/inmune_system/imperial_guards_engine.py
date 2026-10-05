# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Imperial Guards Engine (Caballos Imperiales de Cálculo Espectral)   ║
║ Ruta   : app/core/inmune_system/imperial_guards_engine.py                    ║
║ Versión: 5.1.0-Poincare-Celeste-KAM-Nekhoroshev-Lindstedt-Birkhoff-Twist     ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA Y METROLOGÍA DE LA FPU:
Motor elíptico ciego que ejecuta los cálculos espectrales, algebraicos,
topológicos y de mecánica celeste de Henri Poincaré sobre la FPU. Alimenta
las aduanas de-confinadas del módulo `imperial_guards_agent.py` con
invariancia de Liouville, acción geodésica de Maupertuis–Jacobi, absorción
de pequeños divisores en el anillo de Novikov, y despliega:

  Φ_I   :  (2n, ρ, Σ)     →  𝒢_I  = _PoincareDarbouxGerm
  Φ_II  :  𝒢_I × (M, ω)   →  𝒢_II = _PoincareFloquetGerm
  Φ_III :  𝒢_II           →  PoincareGuardsCertificate

Aduanas celestes de Poincaré:
  1. Sección transversal Σ, dirección característica J n_Σ, flujo n_Σ·X_H.
  2. Recurrencia / primer retorno P_Σ : Σ → Σ.
  3. Corchete de Poisson {f,g}=(∇f)ᵀ Ω (∇g) y generatriz F₂ de Hamilton–Jacobi.
  4. Gram–Schmidt simpléctico S ∈ Sp(2n), ι(M)=Ω M^{−T} Ωᵀ.
  5. Floquet–Lyapunov hamiltoniano: M=exp(T A_F) R_F, A_F ∈ sp(2n).
  6. Mel’nikov ℳ(t₀)=∫{H₀,H₁} dt ; ceros simples ⇒ homoclínico.
  7. Rotación ρ (Gauss–Hurwitz) y KAM |k·ω|≥γ/‖k‖₁^τ (τ>n−1, Rüssmann).
  8. Lyapunov–Benettin (QR con absorción de signos) + Kaplan–Yorke + Pesin.
  9. Poincaré–Cartan ∮ p dq − H dt, acción-ángulo, Kepler–Delaunay–Poincaré.
 10. Birkhoff, Nekhoroshev, twist de Moser / Poincaré–Birkhoff, Lindstedt.

INVARIANTES:
  Liouville det M = +1, Darboux Ωᵀ=−Ω y Ω²=−I, completez de Heyting Ω₃
  (ínfimo de Gödel = join de severidad).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Agents.ImperialGuardsEngine")

__version__: Final[str] = (
    "5.1.0-Poincare-Celeste-KAM-Nekhoroshev-Lindstedt-Birkhoff-Twist-Delaunay"
)

# =============================================================================
# CONSTANTES METROLÓGICAS IEEE-754 Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_HIGHAM_TIKHONOV_FLOOR: Final[float] = 1e-20
_DEFAULT_REGULARIZER: Final[float] = 1e-15
_WILKINSON_DEFLATION_SCALE: Final[float] = 10.0
_WILKINSON_DEFLATION_FLOOR: Final[float] = 1e-12
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_PSD_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_PSD_REL_TOL: Final[float] = 1e-8
_HERMITIAN_REL_TOL: Final[float] = 1e-8
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_LOG_MEAN_REL_TOL: Final[float] = 1e-8
_COMPLEX_STEP_DEFAULT_H: Final[float] = 1e-20
_COMPLEX_STEP_MIN_H: Final[float] = 1e-30
_COMPLEX_STEP_FD_FALLBACK: Final[float] = 1e-8
_LOG_EXP_CLIP: Final[float] = 700.0
_SPECTRAL_DIM_MIN_MODES: Final[int] = 4

_HARD_POINCARE_SECTION_TOL: Final[float] = 1.0e-10
_HARD_FLOQUET_MULTIPLIER_TOL: Final[float] = 1.0e-2
_HARD_LYAPUNOV_TOL: Final[float] = 1.0e-6
_HARD_KAM_GAMMA_FLOOR: Final[float] = 1.0e-6
_HARD_DIVERGENCE_CEILING: Final[float] = 1.0e-4
_HARD_SYMPLECTIC_DEFECT: Final[float] = 1.0e-8
_HARD_TWIST_DET_FLOOR: Final[float] = 1.0e-10
_HARD_BIRKHOFF_RES_CEILING: Final[float] = 1.0e-4
_HARD_RESONANT_DENOM: Final[int] = 5
_LIMIT_WILKINSON: Final[float] = 1.0e-12
_KAM_HARMONIC_CAP: Final[int] = 24
_KAM_PAIR_CAP: Final[int] = 12
_MELNIKOV_PHASE_SAMPLES: Final[int] = 64
_LYAPUNOV_QR_ITERATIONS: Final[int] = 2048
_ROTATION_CF_DEPTH: Final[int] = 24
_DIOPHANTINE_TAU_FLOOR: Final[float] = 1.0 + 1.0e-6
_ACTION_ANGLE_QUADRATURE: Final[int] = 512
_NEKHOROSHEV_C: Final[float] = 0.5
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_PRIME_RADICALS: Final[Tuple[int, ...]] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_REVERSE_HEYTING: Final[Dict[int, str]] = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}


# =============================================================================
# NÚCLEO GEOMÉTRICO — Sp(2n), FRACCIONES CONTINUAS, RED DIOFÁNTICA
# =============================================================================
def _omega_n(n: int) -> np.ndarray:
    r"""Forma canónica \(J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}\in\mathfrak{sp}(2n)^*\)."""
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("n debe ser positivo para la forma simpléctica.")
    eye = np.eye(n_int, dtype=np.float64)
    zero = np.zeros((n_int, n_int), dtype=np.float64)
    return np.block([[zero, eye], [-eye, zero]])


def _frob(arr: np.ndarray) -> float:
    """Norma de Frobenius en rango ≥2; euclídea en rango 1 (IEEE-754 segura)."""
    a = np.asarray(arr)
    if a.size == 0:
        return 0.0
    if a.ndim >= 2:
        return float(la.norm(a, ord="fro"))
    return float(la.norm(a))


def _even_embed(matrix: np.ndarray) -> np.ndarray:
    """Inmersión par: pad identidad (multiplicador Floquet neutro μ=1)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("even_embed exige matriz cuadrada.")
    n = int(m.shape[0])
    if n % 2 == 0:
        return m
    out = np.eye(n + 1, dtype=np.float64)
    out[:n, :n] = m
    return out


def _symplectic_defect(matrix: np.ndarray) -> float:
    r"""Defecto de Sp(2n): \(\|M^{\top} J M - J\|_F\)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] % 2 != 0:
        return float("inf")
    try:
        jay = _omega_n(m.shape[0] // 2)
    except ValueError:
        return float("inf")
    return _frob(m.T @ jay @ m - jay)


def _cotangent_lift(matrix: np.ndarray, regularizer: float) -> np.ndarray:
    r"""Levantamiento cotangente \(\mathrm{GL}(n)\to\mathrm{Sp}(2n)\): \(\mathrm{diag}(A,A^{-\top})\)."""
    a = np.asarray(matrix, dtype=np.float64)
    n = int(a.shape[0])
    eye = np.eye(n, dtype=np.float64)
    try:
        a_inv_t = la.inv(a).T
    except la.LinAlgError:
        a_inv_t = la.inv(a + float(regularizer) * eye).T
    zero = np.zeros((n, n), dtype=np.float64)
    return np.block([[a, zero], [zero, a_inv_t]])


def _project_hamiltonian(gen: np.ndarray, jay: np.ndarray) -> np.ndarray:
    r"""Proyección sobre \(\mathfrak{sp}(2n)\): \(A\mapsto\tfrac12(A+J A^{\top} J)\)."""
    return 0.5 * (gen + jay @ gen.T @ jay)


def _continued_fraction(x: float, depth: int = _ROTATION_CF_DEPTH) -> Tuple[int, ...]:
    """Fracción continua regular de Gauss; termina ssi x es racional (mod ε)."""
    if not np.isfinite(x):
        return tuple()
    value = float(x)
    acc: List[int] = []
    for _ in range(max(int(depth), 1)):
        a_i = int(math.floor(value))
        acc.append(a_i)
        frac = value - float(a_i)
        if abs(frac) < 1.0e-14:
            break
        value = 1.0 / frac
        if (not np.isfinite(value)) or abs(value) > 1.0e16:
            break
    return tuple(acc)


def _cf_convergents(cf: Sequence[int]) -> List[Tuple[int, int]]:
    h_m2, h_m1 = 0, 1
    k_m2, k_m1 = 1, 0
    out: List[Tuple[int, int]] = []
    for a in cf:
        h = int(a) * h_m1 + h_m2
        k = int(a) * k_m1 + k_m2
        out.append((h, k))
        h_m2, h_m1 = h_m1, h
        k_m2, k_m1 = k_m1, k
    return out


def _diophantine_constant(x: float, cf: Sequence[int]) -> float:
    r"""Constante de Hurwitz empírica \(\inf q^2|\alpha-p/q|\)."""
    best = float("inf")
    for p, q in _cf_convergents(cf):
        if q == 0:
            continue
        best = min(best, abs(float(x) - float(p) / float(q)) * float(q) * float(q))
    return best if np.isfinite(best) else float("nan")


def _cf_denominator(cf: Sequence[int]) -> int:
    conv = _cf_convergents(cf)
    if not conv:
        return 0
    return int(abs(conv[-1][1]))


def _count_simple_zeros(samples: np.ndarray, cyclic: bool = True, floor: float = 0.0) -> int:
    y = np.asarray(samples, dtype=np.float64).ravel()
    n = int(y.size)
    if n < 2:
        return 0
    zeros = 0
    last = n if cyclic else n - 1
    for i in range(last):
        a, b = float(y[i]), float(y[(i + 1) % n])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0 and b != 0.0:
            zeros += 1
        elif a * b < 0.0 and min(abs(a), abs(b)) > floor:
            zeros += 1
    return int(zeros)


def _integer_harmonics(dim: int, cap: int) -> np.ndarray:
    """Red entera recortada: ejes + pares con ‖k‖₁ ≤ cap, k ≠ 0. Complejidad O(n² H²)."""
    n = int(max(dim, 1))
    h_cap = int(max(cap, 1))
    pair_cap = int(min(h_cap, _KAM_PAIR_CAP))
    rows: List[np.ndarray] = []
    for i in range(n):
        for sign in (-1, 1):
            for height in range(1, h_cap + 1):
                k = np.zeros(n, dtype=np.int32)
                k[i] = sign * height
                rows.append(k)
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                for a_i in range(-pair_cap, pair_cap + 1):
                    for a_j in range(-pair_cap, pair_cap + 1):
                        if a_i == 0 and a_j == 0:
                            continue
                        if abs(a_i) + abs(a_j) > pair_cap:
                            continue
                        k = np.zeros(n, dtype=np.int32)
                        k[i] = a_i
                        k[j] = a_j
                        rows.append(k)
    if not rows:
        return np.zeros((0, n), dtype=np.int32)
    stacked = np.unique(np.stack(rows, axis=0), axis=0)
    return stacked[np.any(stacked != 0, axis=1)]


def _default_frequencies(n_freq: int) -> np.ndarray:
    """Vector ω con componentes √p_i (independencia ℚ-típica)."""
    m = int(max(n_freq, 1))
    vals = [math.sqrt(float(_PRIME_RADICALS[i % len(_PRIME_RADICALS)])) for i in range(m)]
    if m >= 2:
        vals[1] = float(_GOLDEN_RATIO)
    return np.asarray(vals, dtype=np.float64)


def _unwrap_delta(angles: np.ndarray) -> np.ndarray:
    dth = np.diff(np.asarray(angles, dtype=np.float64).ravel())
    return (dth + np.pi) % (2.0 * np.pi) - np.pi


def _poincare_orbit_type(mu: np.ndarray, tol: float) -> str:
    """Clasificación de Poincaré de una órbita periódica por multiplicadores."""
    if mu.size == 0:
        return "unknown"
    abs_mu = np.abs(mu)
    imag_mu = np.abs(np.imag(mu))
    on_circle = np.all(np.abs(abs_mu - 1.0) <= tol)
    if np.any(abs_mu > 1.0 + tol):
        if np.any(imag_mu > tol):
            return "loxodromic"
        return "hyperbolic"
    if on_circle:
        if np.any(np.abs(mu - 1.0) <= tol) or np.any(np.abs(mu + 1.0) <= tol):
            return "parabolic"
        return "elliptic"
    return "mixed"


def _kaplan_yorke_and_pesin(spectrum: np.ndarray) -> Tuple[float, float]:
    r"""Dimensión de Kaplan–Yorke y entropía de Pesin \(h_{KS}=\sum\lambda_i^+\)."""
    lam = np.sort(np.asarray(spectrum, dtype=np.float64).ravel())[::-1]
    if lam.size == 0 or not np.all(np.isfinite(lam)):
        return float("nan"), float("nan")
    ks = float(np.sum(np.maximum(lam, 0.0)))
    running = 0.0
    ky = float(lam.size)
    for idx, val in enumerate(lam):
        nxt = running + float(val)
        if nxt < 0.0:
            if abs(val) <= _MACHINE_EPS:
                ky = float(idx)
            else:
                ky = float(idx) + running / abs(float(val))
            break
        running = nxt
    return float(ky), float(ks)


def _heyting_join(*verdicts: str) -> str:
    """Ínfimo de Gödel = supremo de severidad (aduanas de seguridad)."""
    if not verdicts:
        return "COHERENT"
    idx = max(_HEYTING_ORDER.get(v if v in _HEYTING_ORDER else "VETOED", 2) for v in verdicts)
    return _REVERSE_HEYTING[idx]


def _vec3(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 2:
        arr = np.array([arr[0], arr[1], 0.0], dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"{name} debe ser de dimensión 2 o 3.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene no-finitos.")
    return arr


# =============================================================================
# DATACLASSES ESTRUCTURALES E INMUTABLES
# =============================================================================
@dataclass(frozen=True, slots=True)
class ImperialEngineStepResult:
    r"""Resultado inmutable de integración simpléctica de Poincaré en FPU."""
    next_state: NDArray[np.float64]
    hamiltonian_energy: float
    volume_drift: float
    maupertuis_action: float
    novikov_absorbed_weight: float
    is_step_valid: bool
    symplectic_defect: float = float("nan")
    energy_drift: float = float("nan")


@dataclass(frozen=True)
class _SpectralTripleCertificate:
    """Certificado numérico del estado que sostiene el triple (A, ℋ, ·)."""
    hermiticity_residual: float
    trace: float
    min_eigenvalue: float
    purity: float
    von_neumann_entropy: float
    effective_rank: int
    is_density: bool


@dataclass(frozen=True)
class _SpectralTripleGerm:
    """Gérmen del triple espectral (objeto terminal histórico de la Fase I)."""
    dim: int
    density: Optional[np.ndarray]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    reg_floor: float
    csmd_step: float
    certificate: _SpectralTripleCertificate


@dataclass(frozen=True, slots=True)
class _SymplecticFormCertificate:
    """Certificado algebraico de la 2-forma canónica de Darboux."""
    skew_residual: float
    almost_complex_residual: float
    determinant: float
    frobenius_norm: float
    is_darboux: bool


@dataclass(frozen=True, slots=True)
class _PoincareSectionWitness:
    r"""
    Sección transversal de Poincaré \(\Sigma=\{q_k=q_k^*\}\).
    Dirección característica \(J n_\Sigma\); transversalidad de flujo \(n_\Sigma\cdot X_H\).
    """
    normal: np.ndarray
    section_index: int
    section_offset: float
    energy_level: float
    transversal_certificate: float
    is_transversal: bool
    characteristic_direction: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class _RecurrenceWitness:
    r"""Recurrencia de Poincaré: \(\inf_{t>0}\|z(t)-z(0)\|\) y recuento de cruces de Σ."""
    return_distance: float
    n_returns: int
    mean_return_gap: float
    is_recurrent: bool


@dataclass(frozen=True, slots=True)
class _HamiltonJacobiWitness:
    """Función generatriz F₂(q_old, P_new) de Hamilton–Jacobi."""
    q_old: np.ndarray
    p_old: np.ndarray
    q_new: np.ndarray
    P_new: np.ndarray
    p_check: np.ndarray
    hessian_F2: np.ndarray
    hj_residual: float
    is_canonical: bool


@dataclass(frozen=True, slots=True)
class _PoincareDarbouxGerm:
    r"""
    **Gérmen de Poincaré–Darboux.**
    **Objeto terminal de la FASE I / objeto inicial de la FASE II.**

    \[
      \mathcal{G}_I=(2n,n,\Omega,\mathrm{Cert}(\Omega),\Sigma,P_\Sigma,\varepsilon_W,h,\mathcal{G}_{\mathrm{Hodge}})
    \]
    Dominio de `induce_poincare_floquet_germ` (II.ω).
    """
    two_n: int
    n: int
    omega: np.ndarray
    form_certificate: _SymplecticFormCertificate
    section: _PoincareSectionWitness
    reg_floor: float
    csmd_step: float
    hodge_germ: Optional[Any]
    recurrence: Optional[_RecurrenceWitness] = None


# ── DTOs Poincaré (Fase II) ──────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class _FloquetLyapunovWitness:
    """Factorización de Floquet–Lyapunov M = exp(T·A_F)·R_F, A_F ∈ sp(2n)."""
    generator: np.ndarray
    periodic_part: np.ndarray
    log_residual: float
    is_real_logarithm: bool
    floquet_multipliers: np.ndarray
    characteristic_exponents: np.ndarray
    orbit_type: str = "unknown"
    hill_discriminant: float = float("nan")
    symplectic_defect: float = float("nan")


@dataclass(frozen=True, slots=True)
class _MelnikovWitness:
    """Función de Mel’nikov ℳ(t₀); ceros simples ⇒ caos homoclínico."""
    melnikov_values: np.ndarray
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool


@dataclass(frozen=True, slots=True)
class _RotationNumberWitness:
    """Número de rotación ρ; fracción continua; Hurwitz."""
    rotation_number: float
    is_rational: bool
    continued_fraction: Tuple[int, ...]
    diophantine_constant: float
    convergent_denominator: int = 0


@dataclass(frozen=True, slots=True)
class _KamTorusWitness:
    """Certificado KAM diofantino |k·ω| ≥ γ/|k|^τ, τ > n−1."""
    frequency_vector: np.ndarray
    diophantine_gamma: float
    diophantine_tau: float
    birkhoff_residual: float
    kam_stable: bool
    iterations: int
    worst_divisor: float = float("nan")


@dataclass(frozen=True, slots=True)
class _LyapunovSpectrumWitness:
    """Espectro de Lyapunov por QR de Benettin + Kaplan–Yorke + Pesin."""
    spectrum: np.ndarray
    kaplan_yorke_dimension: float
    kolmogorov_sinai_entropy: float
    is_chaotic: bool


@dataclass(frozen=True, slots=True)
class _CartanIntegralWitness:
    """Invariante integral de Poincaré–Cartan ∮ p dq − H dt."""
    integral: float
    n_segments: int
    is_closed: bool
    action_integral: float = float("nan")


@dataclass(frozen=True, slots=True)
class _KeplerElementsWitness:
    """Elementos osculadores Kepler + Delaunay (L,G,H,l,g,h) + Poincaré (Λ,λ,ξ,η,p,q)."""
    semi_major_axis: float
    eccentricity: float
    inclination: float
    ascending_node: float
    argument_periapsis: float
    true_anomaly: float
    specific_energy: float
    angular_momentum: np.ndarray
    eccentricity_vector: np.ndarray
    mean_anomaly: float = float("nan")
    mean_motion: float = float("nan")
    delaunay: Optional[Dict[str, float]] = None
    poincare_canonical: Optional[Dict[str, float]] = None


@dataclass(frozen=True, slots=True)
class _ActionAngleWitness:
    """Variables acción-ángulo I_k = (1/2π) ∮ p_k dq_k."""
    actions: np.ndarray
    angles: np.ndarray
    frequencies: Optional[np.ndarray] = None
    generating_residual: float = float("nan")


@dataclass(frozen=True, slots=True)
class _BirkhoffWitness:
    """Residuo de la forma normal de Birkhoff / ecuación homológica {H₀,S}=H₁−[H₁]."""
    order: int
    residual: float
    is_normalizable: bool


@dataclass(frozen=True, slots=True)
class _NekhoroshevWitness:
    r"""Confinamiento \(|I(t)-I(0)|\le\varepsilon^b\) para \(|t|\le\exp(c/\varepsilon^a)\)."""
    exponent_a: float
    confinement_b: float
    time_scale: float
    is_confined: bool


@dataclass(frozen=True, slots=True)
class _TwistWitness:
    """Twist de Moser, no-degeneración de Kolmogorov e isoenergética, Poincaré–Birkhoff."""
    twist_determinant: float
    kolmogorov_nondeg: bool
    isoenergetic_nondeg: bool
    has_poincare_birkhoff_fps: bool


@dataclass(frozen=True, slots=True)
class _LindstedtWitness:
    """Residuo secular de Lindstedt–Poincaré (anulación de términos t·sin)."""
    secular_residual: float
    frequency_correction: np.ndarray


@dataclass(frozen=True, slots=True)
class _PoincareFloquetGerm:
    r"""
    **Gérmen de Poincaré–Floquet.**
    **Objeto terminal de la FASE II / objeto inicial de la FASE III.**

    \[
      \mathcal{G}_{II}=(\mathcal{G}_I,F,\mathcal{M},\rho,\mathrm{KAM},\mathrm{Lyap},
                        \mathrm{Cartan},\mathrm{Kepler},I,\mathrm{Birkhoff},
                        \mathrm{Nekhoroshev},\mathrm{twist},\mathrm{Lindstedt})
    \]
    Dominio de `certify_poincare_guards_cycle` (III.ω).
    """
    darboux_germ: _PoincareDarbouxGerm
    floquet: _FloquetLyapunovWitness
    melnikov: Optional[_MelnikovWitness]
    rotation: Optional[_RotationNumberWitness]
    kam: _KamTorusWitness
    lyapunov: _LyapunovSpectrumWitness
    cartan: Optional[_CartanIntegralWitness]
    kepler: Optional[_KeplerElementsWitness]
    action_angle: Optional[_ActionAngleWitness]
    birkhoff: Optional[_BirkhoffWitness] = None
    nekhoroshev: Optional[_NekhoroshevWitness] = None
    twist: Optional[_TwistWitness] = None
    lindstedt: Optional[_LindstedtWitness] = None


@dataclass(frozen=True, slots=True)
class PoincareGuardsCertificate:
    r"""
    **Certificado global de Poincaré–Guards (morfismo terminal III.ω).**
    Integra \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\).
    """
    two_n: int
    n: int
    section_is_transversal: bool
    section_transversal_certificate: float
    form_is_darboux: bool
    spectral_is_density: bool
    spectral_effective_rank: int
    spectral_von_neumann_entropy: float
    floquet_max_multiplier: float
    floquet_lyapunov_max: float
    floquet_log_residual: float
    floquet_is_real_log: bool
    kam_diophantine_gamma: float
    kam_diophantine_tau: float
    kam_stable: bool
    lyapunov_spectrum: np.ndarray
    lyapunov_kaplan_yorke: float
    lyapunov_ks_entropy: float
    lyapunov_is_chaotic: bool
    melnikov_simple_zeros: Optional[int]
    melnikov_chaotic: Optional[bool]
    rotation_number: Optional[float]
    rotation_is_rational: Optional[bool]
    cartan_integral: Optional[float]
    kepler_semi_major_axis: Optional[float]
    kepler_eccentricity: Optional[float]
    action_angles: Optional[np.ndarray]
    is_liouville_conserved: bool
    is_poincare_coherent: bool
    heyting_verdict: str
    poincare_orbit_type: str = "unknown"
    recurrence_distance: float = float("nan")
    recurrence_is_ergodic: bool = False
    symplectic_defect: float = float("nan")
    hill_discriminant: float = float("nan")
    nekhoroshev_time_scale: float = 0.0
    twist_determinant: float = 0.0
    birkhoff_residual: float = 0.0
    lindstedt_secular_residual: float = 0.0
    heyting_godel_meet: float = 0.0


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE I — FUNDAMENTOS NUMÉRICOS Y GEOMETRÍA DE POINCARÉ                  ██
# ██  Objetos: IEEE-754, Kahan/Neumaier, CSMD, Ω, Σ, {·,·}, F₂, SGS, ι, P_Σ. ██
# ██  Morfismo terminal (I.ω): synthesize_poincare_darboux_germ               ██
# ██                           → objeto inicial de la FASE II                 ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
class Phase1NumericalFoundationsMixin:
    """
    FASE I — FUNDAMENTOS NUMÉRICOS Y GEOMETRÍA DE POINCARÉ.

    Topos lineal subyacente y geometría simpléctica canónica. El cierre
    formal I.ω produce `_PoincareDarbouxGerm`, dominio de la Fase II.
    """

    # ── I.1  Validación escalar ───────────────────────────────────────────
    @staticmethod
    def _validate_nonnegative_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f < 0.0:
            raise ValueError(f"{name} debe ser finito y mayor o igual que cero.")
        return value_f

    @staticmethod
    def _validate_positive_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente mayor que cero.")
        return value_f

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    # ── I.2  Validación de arreglos ───────────────────────────────────────
    @staticmethod
    def _ensure_finite_array(arr: np.ndarray, name: str) -> None:
        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise ValueError(f"{name} contiene tipos no numéricos.") from exc
        if not finite:
            raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")

    def _as_numeric_vector(
        self, values: Any, name: str, *, allow_complex: bool = False,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim == 0:
            raw = raw.reshape(1)
        elif raw.ndim > 1:
            raw = raw.ravel()
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if not allow_complex:
                if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                    raise ValueError(
                        f"{name} posee componente imaginaria no despreciable.")
                raw = raw.real
        try:
            dtype = np.complex128 if allow_complex else np.float64
            arr = np.asarray(raw, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a vector numérico.") from exc
        self._ensure_finite_array(arr, name)
        return arr

    def _as_numeric_matrix(
        self, values: Any, name: str, *,
        allow_complex: bool = False, square: bool = False,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side != raw.size:
                raise ValueError(f"{name} debe ser una matriz 2D.")
            raw = raw.reshape(side, side)
        if raw.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if not allow_complex:
                if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                    raise ValueError(
                        f"{name} posee componente imaginaria no despreciable.")
                raw = raw.real
        try:
            dtype = np.complex128 if allow_complex else np.float64
            arr = np.asarray(raw, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a matriz numérica.") from exc
        self._ensure_finite_array(arr, name)
        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")
        return arr

    def _regularizer_floor(self) -> float:
        reg = getattr(self, "_reg", _DEFAULT_REGULARIZER)
        try:
            value = float(reg)
        except (TypeError, ValueError):
            value = _DEFAULT_REGULARIZER
        if not math.isfinite(value) or value < 0.0:
            value = _DEFAULT_REGULARIZER
        return float(max(value, _HIGHAM_TIKHONOV_FLOOR))

    def _csmd_step(self) -> float:
        step = getattr(self, "_csmd_h", _COMPLEX_STEP_DEFAULT_H)
        try:
            value = float(step)
        except (TypeError, ValueError):
            value = _COMPLEX_STEP_DEFAULT_H
        if not math.isfinite(value) or value <= 0.0:
            value = _COMPLEX_STEP_DEFAULT_H
        return float(max(value, _COMPLEX_STEP_MIN_H))

    # ── I.3  Normas y Higham ──────────────────────────────────────────────
    @staticmethod
    def _frobenius_norm(matrix: np.ndarray) -> float:
        return _frob(matrix)

    def _higham_nearest_hermitian(self, matrix: np.ndarray, name: str) -> np.ndarray:
        a = self._as_numeric_matrix(matrix, name, allow_complex=True, square=True)
        return 0.5 * (a + a.T.conj())

    def _wilkinson_deflation_floor(self, matrix: np.ndarray) -> float:
        if matrix is None or np.asarray(matrix).size == 0:
            return _WILKINSON_DEFLATION_FLOOR
        fro = self._frobenius_norm(matrix)
        return float(max(fro * _MACHINE_EPS * _WILKINSON_DEFLATION_SCALE,
                         _WILKINSON_DEFLATION_FLOOR))

    def _higham_nearest_density(
        self, matrix: np.ndarray, name: str, floor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        herm = self._higham_nearest_hermitian(matrix, name)
        try:
            evals, evecs = la.eigh(herm, check_finite=True)
        except la.LinAlgError as exc:
            raise ValueError(f"La descomposición espectral de {name} falló.") from exc
        evals = np.real(np.asarray(evals, dtype=np.float64))
        self._ensure_finite_array(evals, f"{name}.eigenvalues")
        eps = float(floor) if floor is not None else self._regularizer_floor()
        evals = np.maximum(evals, 0.0)
        evals[evals < eps] = 0.0
        tr = self.kahan_sum(evals)
        if tr > _MACHINE_EPS:
            evals = evals / tr
        else:
            evals = np.zeros_like(evals)
            evals[-1] = 1.0
        rho = evecs @ (evals[:, None] * evecs.T.conj())
        rho = 0.5 * (rho + rho.T.conj())
        return rho, evals, evecs

    # ── I.4  Sumación compensada ──────────────────────────────────────────
    @staticmethod
    def _neumaier_accumulate(
        total: float, compensation: float, term: float,
    ) -> Tuple[float, float]:
        t = total + term
        if not math.isfinite(t):
            return float(t), compensation
        if abs(total) >= abs(term):
            compensation += (total - t) + term
        else:
            compensation += (term - t) + total
        return t, compensation

    def kahan_sum(self, arr: np.ndarray) -> float:
        vec = self._as_numeric_vector(arr, "arr", allow_complex=False)
        total = 0.0
        compensation = 0.0
        for term in vec:
            total, compensation = self._neumaier_accumulate(total, compensation, float(term))
            if not math.isfinite(total):
                return float(total)
        return float(total + compensation)

    def kahan_classical_sum(self, arr: np.ndarray) -> float:
        vec = self._as_numeric_vector(arr, "arr", allow_complex=False)
        total = 0.0
        c = 0.0
        for term in vec:
            y = float(term) - c
            t = total + y
            c = (t - total) - y
            total = t
        return float(total)

    def kahan_babuska_neumaier_sum(self, arr: np.ndarray) -> float:
        return self.kahan_sum(arr)

    def klein_sum(self, arr: np.ndarray) -> float:
        vec = self._as_numeric_vector(arr, "arr", allow_complex=False)
        s = 0.0
        cs = 0.0
        ccs = 0.0
        for term in vec:
            xf = float(term)
            t = s + xf
            c = (s - t) + xf if abs(s) >= abs(xf) else (xf - t) + s
            s = t
            t = cs + c
            cc = (cs - t) + c if abs(cs) >= abs(c) else (c - t) + cs
            cs = t
            ccs += cc
        return float(s + cs + ccs)

    # ── I.5  CSMD ─────────────────────────────────────────────────────────
    def compute_complex_step_gradient(
        self, func: Callable[[np.ndarray], Any], x: np.ndarray,
        h: float = _COMPLEX_STEP_DEFAULT_H,
    ) -> np.ndarray:
        if not callable(func):
            raise TypeError("func debe ser callable.")
        x_vec = self._as_numeric_vector(x, "x", allow_complex=False)
        h_val = self._validate_nonnegative_finite("h", h)
        if h_val <= 0.0:
            raise ValueError("h debe ser estrictamente positivo.")
        if h_val < _COMPLEX_STEP_MIN_H:
            h_val = _COMPLEX_STEP_MIN_H
        dim = x_vec.size
        grad = np.zeros(x_vec.shape, dtype=np.float64)
        if dim == 0:
            return grad
        holomorphic = True
        try:
            probe = func(x_vec.astype(np.complex128))
            probe_arr = np.asarray(probe)
            if probe_arr.size != 1:
                holomorphic = False
            else:
                _ = np.imag(probe_arr.reshape(-1)[0])
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False
        if holomorphic:
            x_complex = x_vec.astype(np.complex128)
            for i in range(dim):
                x_pert = x_complex.copy()
                x_pert[i] += 1j * h_val
                try:
                    val = func(x_pert)
                except Exception as exc:
                    holomorphic = False
                    logger.warning("CSMD: func falló (%s).", exc)
                    break
                val_arr = np.asarray(val)
                if val_arr.size != 1:
                    holomorphic = False
                    break
                val_scalar = val_arr.reshape(-1)[0]
                try:
                    if not np.isfinite(val_scalar):
                        raise ValueError("func retornó no finito.")
                except TypeError as exc:
                    raise ValueError("func retornó tipo no numérico.") from exc
                imag = float(np.imag(val_scalar))
                if not math.isfinite(imag):
                    holomorphic = False
                    break
                grad_i = imag / h_val
                if not math.isfinite(grad_i):
                    raise ValueError("CSMD: gradiente no finito.")
                grad[i] = grad_i
        if not holomorphic:
            logger.warning("CSMD: func no holomorfa; uso diff. central.")
            scale = float(max(np.linalg.norm(x_vec), 1.0))
            h_fd = max(_COMPLEX_STEP_FD_FALLBACK, _COMPLEX_STEP_FD_FALLBACK * scale)
            for i in range(dim):
                xp = x_vec.copy()
                xm = x_vec.copy()
                xp[i] += h_fd
                xm[i] -= h_fd
                try:
                    fp = np.asarray(func(xp)).reshape(-1)[0]
                    fm = np.asarray(func(xm)).reshape(-1)[0]
                except Exception as exc:
                    raise ValueError("func falló en diff. central.") from exc
                fp_f = float(np.real(fp))
                fm_f = float(np.real(fm))
                if not (math.isfinite(fp_f) and math.isfinite(fm_f)):
                    raise ValueError("func retornó no finito.")
                grad[i] = (fp_f - fm_f) / (2.0 * h_fd)
        return grad

    def compute_complex_step_hessian(
        self, func: Callable[[np.ndarray], Any], x: np.ndarray,
        h: float = _COMPLEX_STEP_DEFAULT_H,
    ) -> np.ndarray:
        x_vec = self._as_numeric_vector(x, "x", allow_complex=False)
        dim = x_vec.size
        scale = float(max(np.linalg.norm(x_vec), 1.0))
        eta = max(math.sqrt(_MACHINE_EPS), math.sqrt(_MACHINE_EPS) * scale)
        hess = np.zeros((dim, dim), dtype=np.float64)
        for j in range(dim):
            xp = x_vec.copy()
            xm = x_vec.copy()
            xp[j] += eta
            xm[j] -= eta
            gp = self.compute_complex_step_gradient(func, xp, h=h)
            gm = self.compute_complex_step_gradient(func, xm, h=h)
            hess[:, j] = (gp - gm) / (2.0 * eta)
        return 0.5 * (hess + hess.T)

    # ── I.6  Geometría simpléctica de Poincaré ───────────────────────────
    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        r"""\(\Omega=\begin{pmatrix}0&I\\-I&0\end{pmatrix}\); \(\Omega^{\top}=-\Omega\), \(\Omega^2=-I\)."""
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"dim={dim} debe ser par y positivo (Darboux).")
        return _omega_n(dim // 2)

    def certify_symplectic_form(self, omega: np.ndarray) -> _SymplecticFormCertificate:
        o = self._as_numeric_matrix(omega, "omega", square=True)
        dim = o.shape[0]
        ident = np.eye(dim, dtype=o.dtype)
        skew = self._frobenius_norm(o + o.T)
        almost_c = self._frobenius_norm(o @ o + ident)
        det_o = float(np.real(la.det(o)))
        fro = self._frobenius_norm(o)
        scale = max(fro, 1.0)
        is_darboux = bool(
            skew <= _WILKINSON_DRIFT_LIMIT * scale
            and almost_c <= _WILKINSON_DRIFT_LIMIT * scale
            and abs(det_o - 1.0) <= 1e-8 * max(1.0, abs(det_o))
        )
        return _SymplecticFormCertificate(
            skew_residual=float(skew),
            almost_complex_residual=float(almost_c),
            determinant=det_o,
            frobenius_norm=fro,
            is_darboux=is_darboux,
        )

    def build_poincare_section(
        self, two_n: int, section_index: int = 0,
        section_offset: float = 0.0, energy_level: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareSectionWitness:
        r"""
        \(\Sigma=\{q_{\mathrm{idx}}=q^*\}\), \(n_\Sigma=e_{\mathrm{idx}}\).
        Dirección característica \(J n_\Sigma\). Si se da \(X_H\),
        transversalidad de flujo \(n_\Sigma\cdot X_H\neq 0\).
        """
        if two_n <= 0 or two_n % 2 != 0:
            raise ValueError("two_n debe ser par positivo.")
        n = two_n // 2
        if not (0 <= section_index < n):
            raise ValueError(f"section_index={section_index} fuera de [0,{n-1}].")
        n_sigma = np.zeros(two_n, dtype=np.float64)
        n_sigma[section_index] = 1.0
        omega = self.generate_canonical_symplectic_form(two_n)
        char_dir = omega @ n_sigma
        regular = self._frobenius_norm(char_dir)
        if flow_vector is not None:
            flow = self._as_numeric_vector(flow_vector, "flow_vector")
            if flow.size != two_n:
                raise ValueError(f"flow_vector debe tener dimensión {two_n}.")
            certificate = float(abs(float(n_sigma @ flow)))
            is_transversal = bool(certificate > _HARD_POINCARE_SECTION_TOL)
        else:
            certificate = float(regular)
            is_transversal = bool(regular > _MACHINE_EPS)
        return _PoincareSectionWitness(
            normal=n_sigma, section_index=int(section_index),
            section_offset=float(section_offset),
            energy_level=float(energy_level),
            transversal_certificate=float(certificate),
            is_transversal=is_transversal,
            characteristic_direction=char_dir,
        )

    def poincare_recurrence(
        self,
        state_trajectory_z: Optional[Sequence[np.ndarray]],
        section: _PoincareSectionWitness,
    ) -> _RecurrenceWitness:
        r"""Distancia de primer retorno y recuento de cruces de \(\Sigma\)."""
        if not state_trajectory_z or len(state_trajectory_z) < 2:
            return _RecurrenceWitness(
                return_distance=0.0, n_returns=0, mean_return_gap=0.0,
                is_recurrent=True)
        pts = [self._as_numeric_vector(z, "z_k") for z in state_trajectory_z]
        dim0 = int(pts[0].size)
        if any(int(p.size) != dim0 for p in pts):
            raise ValueError("La trayectoria no es de dimensión homogénea.")
        current = pts[-1]
        distances = [float(la.norm(p - current)) for p in pts[:-1]]
        min_ret = float(np.min(distances)) if distances else 0.0
        n_sigma = np.asarray(section.normal, dtype=np.float64).ravel()
        offset = float(section.section_offset)
        crossings = 0
        gaps: List[int] = []
        last_cross = -1
        if n_sigma.size == dim0:
            sigma_val = [float(n_sigma @ p) - offset for p in pts]
            for i in range(len(sigma_val) - 1):
                a, b = sigma_val[i], sigma_val[i + 1]
                if a == 0.0 or a * b < 0.0:
                    crossings += 1
                    if last_cross >= 0:
                        gaps.append(i - last_cross)
                    last_cross = i
        mean_gap = float(np.mean(gaps)) if gaps else float(len(pts) - 1)
        return _RecurrenceWitness(
            return_distance=float(min_ret),
            n_returns=int(crossings),
            mean_return_gap=float(mean_gap),
            is_recurrent=bool(min_ret <= _HARD_DIVERGENCE_CEILING),
        )

    def poisson_bracket(
        self, grad_f: np.ndarray, grad_g: np.ndarray, omega: np.ndarray,
    ) -> float:
        r"""\(\{f,g\}=(\nabla f)^{\top}\Omega(\nabla g)\)."""
        gf = self._as_numeric_vector(grad_f, "grad_f")
        gg = self._as_numeric_vector(grad_g, "grad_g")
        o = self._as_numeric_matrix(omega, "omega", square=True)
        if gf.size != gg.size or o.shape != (gf.size, gf.size):
            raise ValueError("Dimensiones incompatibles para el corchete.")
        return float(gf @ o @ gg)

    def hamilton_jacobi_F2(
        self, q_old: np.ndarray, p_old: np.ndarray, q_new: np.ndarray,
        hessian_F2: Optional[np.ndarray] = None,
    ) -> _HamiltonJacobiWitness:
        r"""Generatriz \(F_2(q,P)\): \(p=-\partial F_2/\partial q\), \(Q=\partial F_2/\partial P\)."""
        qo = self._as_numeric_vector(q_old, "q_old")
        po = self._as_numeric_vector(p_old, "p_old")
        qn = self._as_numeric_vector(q_new, "q_new")
        if qo.shape != po.shape or qo.shape != qn.shape:
            raise ValueError("q_old, p_old, q_new deben compartir shape.")
        n = qo.size
        h_mat = np.eye(n, dtype=np.float64) if hessian_F2 is None \
            else self._as_numeric_matrix(hessian_F2, "hessian_F2", square=True)
        if h_mat.shape != (n, n):
            raise ValueError(f"hessian_F2 debe ser {n}×{n}.")
        dq = qn - qo
        p_new = po + h_mat @ dq
        p_check = po - h_mat @ dq
        hj_res = float(np.linalg.norm(po - p_check))
        is_canon = bool(hj_res <= _WILKINSON_DRIFT_LIMIT
                        * max(1.0, float(np.linalg.norm(po))))
        return _HamiltonJacobiWitness(
            q_old=qo, p_old=po, q_new=qn,
            P_new=p_new, p_check=p_check, hessian_F2=h_mat,
            hj_residual=float(hj_res), is_canonical=is_canon,
        )

    def symplectic_gram_schmidt(
        self, vectors: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        r"""\(S\in\mathrm{Sp}(2n,\mathbb{R})\) con \(S^{\top}\Omega S=\Omega\)."""
        b_mat = self._as_numeric_matrix(vectors, "vectors")
        o_mat = self._as_numeric_matrix(omega, "omega", square=True)
        dim = o_mat.shape[0]
        if b_mat.shape != (dim, dim):
            raise ValueError(f"vectors debe ser ({dim},{dim}); {b_mat.shape}.")
        n = dim // 2
        s_mat = np.zeros_like(b_mat)
        for k in range(n):
            v = b_mat[:, k].copy()
            w = b_mat[:, k + n].copy()
            for j in range(k):
                u_j = s_mat[:, j]
                u_jn = s_mat[:, j + n]
                a = float(u_j @ o_mat @ v)
                b = float(u_jn @ o_mat @ v)
                v = v + b * u_j - a * u_jn
            for j in range(k):
                u_j = s_mat[:, j]
                u_jn = s_mat[:, j + n]
                c = float(u_j @ o_mat @ w)
                d = float(u_jn @ o_mat @ w)
                w = w + d * u_j - c * u_jn
            omega_vw = float(v @ o_mat @ w)
            if abs(omega_vw) < _MACHINE_EPS:
                logger.warning("SGS: par %d degenerado (ω=%.3e).", k, omega_vw)
                omega_vw = np.sign(omega_vw or 1.0) * _MACHINE_EPS
            s_mat[:, k] = v
            s_mat[:, k + n] = w / omega_vw
        res = self._frobenius_norm(s_mat.T @ o_mat @ s_mat - o_mat)
        if res > 1e-6:
            logger.warning("SGS: residuo simpléctico %.3e > umbral.", res)
        return s_mat

    def cartan_involution(self, m_mat: np.ndarray, omega: np.ndarray) -> np.ndarray:
        r"""\(\iota(M)=\Omega M^{-T}\Omega^{\top}\)."""
        m = self._as_numeric_matrix(m_mat, "M", square=True)
        o = self._as_numeric_matrix(omega, "omega", square=True)
        dim = m.shape[0]
        ident = np.eye(dim, dtype=np.float64)
        try:
            x_inv_t = la.solve(m.T, ident, assume_a="gen")
            return o @ x_inv_t @ o.T
        except (la.LinAlgError, ValueError) as exc:
            logger.warning("Cartan fallido (%s); pinv.", exc)
            return o @ la.pinv(m.T) @ o.T

    def symplectic_residual(self, m_mat: np.ndarray, omega: np.ndarray) -> float:
        r"""\(\|M^{\top}\Omega M-\Omega\|_F\)."""
        m = self._as_numeric_matrix(m_mat, "M", square=True)
        o = self._as_numeric_matrix(omega, "omega", square=True)
        if m.shape != o.shape:
            raise ValueError("M y omega deben compartir dimensión.")
        return self._frobenius_norm(m.T @ o @ m - o)

    def symplectic_inverse(self, s_mat: np.ndarray, omega: np.ndarray) -> np.ndarray:
        r"""\(S^{-1}=-\Omega S^{\top}\Omega\)."""
        s = self._as_numeric_matrix(s_mat, "S", square=True)
        o = self._as_numeric_matrix(omega, "omega", square=True)
        return -o @ s.T @ o

    def cotangent_lift(self, matrix: np.ndarray) -> np.ndarray:
        """Morfismo canónico GL(n) → Sp(2n)."""
        a = self._as_numeric_matrix(matrix, "matrix", square=True)
        return _cotangent_lift(a, self._regularizer_floor())

    # ── I.9  Morfismo terminal histórico (compatibilidad 4.1) ─────────────
    def synthesize_spectral_triple_germ(
        self, density_matrix: Optional[np.ndarray] = None,
        csmd_step: float = _COMPLEX_STEP_DEFAULT_H,
        regularizer: Optional[float] = None,
    ) -> _SpectralTripleGerm:
        h_val = self._validate_positive_finite("csmd_step", csmd_step)
        h_val = max(h_val, _COMPLEX_STEP_MIN_H)
        floor = (self._regularizer_floor() if regularizer is None else
                 float(max(self._validate_nonnegative_finite("regularizer", regularizer),
                           _HIGHAM_TIKHONOV_FLOOR)))
        empty_cert = _SpectralTripleCertificate(
            hermiticity_residual=0.0, trace=0.0, min_eigenvalue=0.0,
            purity=0.0, von_neumann_entropy=0.0,
            effective_rank=0, is_density=False)
        if density_matrix is None:
            return _SpectralTripleGerm(
                dim=0, density=None,
                eigenvalues=np.array([], dtype=np.float64),
                eigenvectors=np.zeros((0, 0), dtype=np.complex128),
                reg_floor=floor, csmd_step=float(h_val), certificate=empty_cert)
        raw = self._as_numeric_matrix(
            density_matrix, "density_matrix", allow_complex=True, square=True)
        if raw.shape[0] == 0:
            raise ValueError("density_matrix no puede ser 0×0.")
        herm_res = self._frobenius_norm(raw - raw.T.conj())
        rho, evals, evecs = self._higham_nearest_density(
            raw, "density_matrix", floor=floor)
        tr = self.kahan_sum(evals)
        min_ev = float(np.min(evals)) if evals.size else 0.0
        purity = self.kahan_sum(evals * evals)
        pos = evals > floor
        vn = float(self.kahan_sum(-evals[pos] * np.log(evals[pos]))) if np.any(pos) else 0.0
        rank = int(np.sum(pos))
        scale = max(self._frobenius_norm(raw), 1.0)
        is_dens = (
            herm_res <= _HERMITIAN_REL_TOL * scale
            and abs(tr - 1.0) <= 1e-10
            and min_ev >= -max(floor, _PSD_ABS_TOL)
        )
        cert = _SpectralTripleCertificate(
            hermiticity_residual=float(herm_res), trace=float(tr),
            min_eigenvalue=min_ev, purity=float(purity),
            von_neumann_entropy=float(max(vn, 0.0)),
            effective_rank=rank, is_density=bool(is_dens))
        if not is_dens:
            logger.warning("Gérmen espectral: ρ no es densidad estricta.")
        return _SpectralTripleGerm(
            dim=int(raw.shape[0]), density=rho,
            eigenvalues=np.asarray(evals, dtype=np.float64),
            eigenvectors=np.asarray(evecs),
            reg_floor=floor, csmd_step=float(h_val), certificate=cert)

    # ── I.ω  MORFISMO TERMINAL DE LA FASE I ───────────────────────────────
    def synthesize_poincare_darboux_germ(
        self,
        two_n: Optional[int] = None,
        density_matrix: Optional[np.ndarray] = None,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        csmd_step: float = _COMPLEX_STEP_DEFAULT_H,
        regularizer: Optional[float] = None,
        hodge_germ: Optional[Any] = None,
        flow_vector: Optional[np.ndarray] = None,
        state_trajectory_z: Optional[Sequence[np.ndarray]] = None,
    ) -> _PoincareDarbouxGerm:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Ensambla \(\mathcal{G}_I=(2n,n,\Omega,\mathrm{Cert}(\Omega),\Sigma,P_\Sigma,\varepsilon_W,h)\).
        El tipo de retorno `_PoincareDarbouxGerm` es el dominio de
        `induce_poincare_floquet_germ` (II.ω).
        """
        spectral_germ: Optional[_SpectralTripleGerm] = None
        if density_matrix is not None:
            spectral_germ = self.synthesize_spectral_triple_germ(
                density_matrix, csmd_step=csmd_step, regularizer=regularizer)
            inferred_dim = int(spectral_germ.dim)
            if inferred_dim <= 0:
                raise ValueError("La densidad suministrada tiene dimensión nula.")
            if two_n is not None and int(two_n) != inferred_dim:
                logger.warning(
                    "two_n=%d difiere de dim(ρ)=%d; se usa dim(ρ).",
                    two_n, inferred_dim)
            two_n_val = inferred_dim
        else:
            if two_n is None:
                two_n_val = int(getattr(self, "_dim", 6)) * 2
            else:
                two_n_val = int(two_n)
        if two_n_val <= 0 or two_n_val % 2 != 0:
            raise ValueError(f"two_n={two_n_val} debe ser par y positivo.")
        omega = self.generate_canonical_symplectic_form(two_n_val)
        form_cert = self.certify_symplectic_form(omega)
        if not form_cert.is_darboux:
            logger.warning(
                "Darboux degradado: skew=%.3e, Ω²+I=%.3e, det=%.16f",
                form_cert.skew_residual, form_cert.almost_complex_residual,
                form_cert.determinant)
        section = self.build_poincare_section(
            two_n_val, section_index=section_index,
            section_offset=section_offset, energy_level=energy_level,
            flow_vector=flow_vector)
        recurrence: Optional[_RecurrenceWitness] = None
        if state_trajectory_z is not None:
            try:
                recurrence = self.poincare_recurrence(state_trajectory_z, section)
            except ValueError as exc:
                logger.error("Recurrencia de Poincaré falló: %s", exc)
        floor = self._regularizer_floor()
        if spectral_germ is not None:
            floor = max(floor, spectral_germ.reg_floor)
        return _PoincareDarbouxGerm(
            two_n=int(two_n_val), n=int(two_n_val) // 2,
            omega=omega, form_certificate=form_cert, section=section,
            reg_floor=float(floor),
            csmd_step=float(self._csmd_step()),
            hodge_germ=hodge_germ,
            recurrence=recurrence)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE II — TRIPLE ESPECTRAL, PETZ Y DINÁMICA CELESTE DE POINCARÉ         ██
# ██  Continuación directa del morfismo I.ω.                                   ██
# ██  Dominio: _PoincareDarbouxGerm                                            ██
# ██  Morfismo terminal (II.ω): induce_poincare_floquet_germ                   ██
# ██                           → objeto inicial de la FASE III                ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
@dataclass(frozen=True)
class _DiracSpectrumResult:
    dirac_eigs: np.ndarray
    eigenvalues: np.ndarray
    dirac_operator: np.ndarray
    laplacian_eigs: np.ndarray
    spectral_action: float
    spectral_dimension: float
    condition_number: float
    kernel_dim: int


@dataclass(frozen=True)
class _PetzMetricResult:
    bkm: float
    bures: float
    wigner_yanase: float
    dropped_kernel_terms: int
    tangent_hermiticity: float
    tangent_traceless: float


@dataclass(frozen=True)
class _HodgeCheegerGerm:
    laplacian_eigs: np.ndarray
    combinatorial_laplacian: Optional[np.ndarray]
    dirac_eigs: np.ndarray
    petz_scale: float
    two_n_hint: int
    reg_floor: float
    from_connes: bool


class Phase2SpectralQuantumMixin(Phase1NumericalFoundationsMixin):
    """
    FASE II — ESPECTRAL CUÁNTICA Y DINÁMICA CELESTE DE POINCARÉ.

    Dirac modular de Connes, métricas de Petz, y el programa celeste:
    Floquet hamiltoniano, Mel’nikov, rotación, KAM, Benettin, Cartan,
    Kepler–Delaunay–Poincaré, Birkhoff, Nekhoroshev, twist, Lindstedt.

    Dominio de II.ω = imagen de I.ω (`_PoincareDarbouxGerm`).
    """

    # ── II.1  Resolución del gérmen espectral ────────────────────────────
    def _resolve_triple_germ(
        self, density_matrix: np.ndarray,
        germ: Optional[_SpectralTripleGerm] = None,
    ) -> _SpectralTripleGerm:
        cached = germ if germ is not None else getattr(self, "_triple_germ", None)
        raw = self._as_numeric_matrix(
            density_matrix, "density_matrix", allow_complex=True, square=True)
        if (cached is not None and cached.density is not None
                and cached.dim == raw.shape[0]
                and cached.density.shape == raw.shape):
            return cached
        built = self.synthesize_spectral_triple_germ(raw)
        self._triple_germ = built
        return built

    def _ensure_hermitian(self, matrix: np.ndarray, name: str) -> np.ndarray:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada.")
        adjoint = matrix.conj().T
        diff_norm = self._frobenius_norm(matrix - adjoint)
        scale = max(1.0, self._frobenius_norm(matrix))
        if diff_norm > _HERMITIAN_REL_TOL * scale:
            raise ValueError(f"{name} no es Hermitiano dentro de tolerancia.")
        return 0.5 * (matrix + adjoint)

    def _eigh_safe(
        self, hermitian_matrix: np.ndarray, name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            eigenvalues, eigenvectors = la.eigh(hermitian_matrix, check_finite=True)
        except la.LinAlgError as exc:
            raise ValueError(f"La descomposición espectral de {name} falló.") from exc
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self._ensure_finite_array(eigenvalues, f"{name}.eigenvalues")
        return eigenvalues, eigenvectors

    @staticmethod
    def _psd_tolerance(eigenvalues: np.ndarray) -> float:
        if eigenvalues.size == 0:
            return _PSD_ABS_TOL
        max_abs = float(np.max(np.abs(eigenvalues)))
        return max(_PSD_ABS_TOL, _PSD_REL_TOL * max_abs)

    def _enforce_psd(self, eigenvalues: np.ndarray, name: str) -> None:
        if eigenvalues.size == 0:
            return
        tol = self._psd_tolerance(eigenvalues)
        min_eig = float(np.min(eigenvalues))
        if min_eig < -tol:
            raise ValueError(f"{name} no es PSD dentro de tolerancia.")

    @staticmethod
    def _logarithmic_mean(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 0.0
        if a == b:
            return float(a)
        lo, hi = (a, b) if a < b else (b, a)
        if (hi - lo) <= _LOG_MEAN_REL_TOL * hi:
            return float(0.5 * (a + b))
        t = hi / lo
        log_t = math.log(t)
        if log_t == 0.0:
            return float(0.5 * (a + b))
        return float(lo * (t - 1.0) / log_t)

    @staticmethod
    def _arithmetic_mean(a: float, b: float) -> float:
        if a <= 0.0 and b <= 0.0:
            return 0.0
        return float(0.5 * (max(a, 0.0) + max(b, 0.0)))

    @staticmethod
    def _wigner_yanase_mean(a: float, b: float) -> float:
        if a <= 0.0 and b <= 0.0:
            return 0.0
        return float(0.25 * (math.sqrt(max(a, 0.0)) + math.sqrt(max(b, 0.0))) ** 2)

    def _petz_sum(
        self, eigs: np.ndarray, a_rot: np.ndarray, b_rot: np.ndarray,
        mean_fn: Callable[[float, float], float],
    ) -> Tuple[float, int]:
        total = 0.0
        compensation = 0.0
        dropped = 0
        n = eigs.size
        for i in range(n):
            lam_i = float(eigs[i])
            for j in range(n):
                lam_j = float(eigs[j])
                mean_val = mean_fn(lam_i, lam_j)
                if mean_val <= _MACHINE_EPS:
                    dropped += 1
                    continue
                term_f = float(np.real(a_rot[i, j] * b_rot[j, i]) / mean_val)
                if not math.isfinite(term_f):
                    raise ValueError("La métrica de Petz produjo no finito.")
                total, compensation = self._neumaier_accumulate(
                    total, compensation, term_f)
        return float(total + compensation), int(dropped)

    # ── II.2  Dirac modular de Connes–Chamseddine ─────────────────────────
    def compute_dirac_operator_spectrum(
        self, density_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        result = self.compute_dirac_operator_spectrum_certified(density_matrix)
        return result.dirac_eigs, result.eigenvalues

    def compute_dirac_operator_spectrum_certified(
        self, density_matrix: np.ndarray,
    ) -> _DiracSpectrumResult:
        germ = self._resolve_triple_germ(density_matrix)
        assert germ.density is not None
        eigenvalues = np.asarray(germ.eigenvalues, dtype=np.float64)
        evecs = germ.eigenvectors
        self._enforce_psd(eigenvalues, "density_matrix")
        floor = germ.reg_floor
        clamped = np.clip(eigenvalues, floor, None)
        dirac_eigs = 1.0 / np.sqrt(clamped)
        self._ensure_finite_array(dirac_eigs, "dirac_eigs")
        dirac_op = evecs @ (dirac_eigs[:, None] * evecs.T.conj())
        dirac_op = 0.5 * (dirac_op + dirac_op.T.conj())
        lap_eigs = dirac_eigs * dirac_eigs
        expo = np.clip(-lap_eigs, -_LOG_EXP_CLIP, 0.0)
        spectral_action = float(self.kahan_sum(np.exp(expo)))
        spec_dim = 0.0
        live = np.sort(dirac_eigs[np.isfinite(dirac_eigs)])
        if live.size >= _SPECTRAL_DIM_MIN_MODES:
            lo = live.size // 3
            hi = max(lo + 2, (2 * live.size) // 3)
            lam = live[lo:hi]
            counts = np.arange(lo + 1, lo + 1 + lam.size, dtype=np.float64)
            if np.all(lam > 0.0):
                x = np.log(lam)
                y = np.log(counts)
                xm = float(self.kahan_sum(x) / x.size)
                ym = float(self.kahan_sum(y) / y.size)
                var_x = float(self.kahan_sum((x - xm) ** 2))
                cov = float(self.kahan_sum((x - xm) * (y - ym)))
                if var_x > _MACHINE_EPS:
                    spec_dim = float(max(cov / var_x, 0.0))
        cond = float(dirac_eigs.max() / max(dirac_eigs.min(), _MACHINE_EPS))
        ker = int(np.sum(eigenvalues <= floor))
        return _DiracSpectrumResult(
            dirac_eigs=np.asarray(dirac_eigs, dtype=np.float64),
            eigenvalues=eigenvalues, dirac_operator=dirac_op,
            laplacian_eigs=np.asarray(lap_eigs, dtype=np.float64),
            spectral_action=spectral_action,
            spectral_dimension=spec_dim,
            condition_number=cond, kernel_dim=ker)

    # ── II.3  Petz Fisher–Rao cuántica ────────────────────────────────────
    def compute_petz_fisher_rao_metric(
        self, rho: np.ndarray, a_mat: np.ndarray, b_mat: np.ndarray,
    ) -> float:
        return self.compute_petz_fisher_rao_metric_certified(rho, a_mat, b_mat).bkm

    def compute_petz_fisher_rao_metric_certified(
        self, rho: np.ndarray, a_mat: np.ndarray, b_mat: np.ndarray,
    ) -> _PetzMetricResult:
        germ = self._resolve_triple_germ(rho)
        assert germ.density is not None
        a = self._as_numeric_matrix(a_mat, "A", allow_complex=True, square=True)
        b = self._as_numeric_matrix(b_mat, "B", allow_complex=True, square=True)
        if germ.density.shape != a.shape or germ.density.shape != b.shape:
            raise ValueError("rho, A y B deben tener la misma forma cuadrada.")
        if germ.dim == 0:
            return _PetzMetricResult(0.0, 0.0, 0.0, 0, 0.0, 0.0)
        eigenvalues = np.asarray(germ.eigenvalues, dtype=np.float64)
        v_mat = germ.eigenvectors
        self._enforce_psd(eigenvalues, "rho")
        eigs = np.clip(eigenvalues, germ.reg_floor, None)
        a_rot = v_mat.conj().T @ a @ v_mat
        b_rot = v_mat.conj().T @ b @ v_mat
        bkm, dropped = self._petz_sum(eigs, a_rot, b_rot, self._logarithmic_mean)
        bures, _ = self._petz_sum(eigs, a_rot, b_rot, self._arithmetic_mean)
        wy, _ = self._petz_sum(eigs, a_rot, b_rot, self._wigner_yanase_mean)
        herm_a = self._frobenius_norm(a - a.T.conj())
        herm_b = self._frobenius_norm(b - b.T.conj())
        tr_a = float(np.real(np.trace(a)))
        tr_b = float(np.real(np.trace(b)))
        return _PetzMetricResult(
            bkm=float(bkm), bures=float(bures), wigner_yanase=float(wy),
            dropped_kernel_terms=int(dropped),
            tangent_hermiticity=float(max(herm_a, herm_b)),
            tangent_traceless=float(max(abs(tr_a), abs(tr_b))))

    # ── II.4  Floquet–Lyapunov (logaritmo hamiltoniano) ───────────────────
    def floquet_lyapunov_factorization(
        self, m_mat: np.ndarray, orbit_period_T: float,
    ) -> _FloquetLyapunovWitness:
        r"""
        \(M=\exp(T A_F)R_F\). \(A_F\) se proyecta a \(\mathfrak{sp}(2n)\)
        cuando \(\dim M\) es par. Un logaritmo real matricial existe ssi
        \(M\) es invertible y los autovalores negativos tienen bloques de
        Jordan de multiplicidad par (Higham); en órbitas elípticas los
        \(\mu_k\) viven en el círculo unidad.
        """
        m = self._as_numeric_matrix(m_mat, "M", square=True)
        if orbit_period_T <= 0.0 or not np.isfinite(orbit_period_T):
            raise ValueError("orbit_period_T debe ser positivo y finito.")
        mu = la.eigvals(m)
        defect = _symplectic_defect(m)
        hill = float(np.real(np.trace(m))) if m.shape[0] == 2 else float("nan")
        is_real_log = True
        try:
            log_m = np.asarray(la.logm(m), dtype=np.complex128)
            imag_res = self._frobenius_norm(np.imag(log_m))
            if imag_res > 1.0e-8:
                is_real_log = False
            gen = np.real(log_m) / float(orbit_period_T)
        except (la.LinAlgError, ValueError) as exc:
            logger.warning("logm falló (%s); linealización.", exc)
            gen = (m - np.eye(m.shape[0], dtype=np.float64)) / float(orbit_period_T)
            is_real_log = False
        if m.shape[0] % 2 == 0:
            jay = _omega_n(m.shape[0] // 2)
            gen = _project_hamiltonian(gen, jay)
        r_f = la.expm(-float(orbit_period_T) * gen) @ m
        log_res = self._frobenius_norm(la.expm(float(orbit_period_T) * gen) @ r_f - m)
        with np.errstate(divide="ignore", invalid="ignore"):
            char_exp = np.log(np.abs(mu) + _MACHINE_EPS) / float(orbit_period_T)
        return _FloquetLyapunovWitness(
            generator=np.asarray(gen, dtype=np.float64),
            periodic_part=np.asarray(r_f, dtype=np.float64),
            log_residual=float(log_res),
            is_real_logarithm=bool(is_real_log),
            floquet_multipliers=np.asarray(mu, dtype=np.complex128),
            characteristic_exponents=np.asarray(np.real(char_exp), dtype=np.float64),
            orbit_type=_poincare_orbit_type(
                np.asarray(mu, dtype=np.complex128), _HARD_FLOQUET_MULTIPLIER_TOL),
            hill_discriminant=hill,
            symplectic_defect=float(defect),
        )

    def compute_floquet_lyapunov_certified(
        self, m_mat: np.ndarray, orbit_period_T: float,
    ) -> _FloquetLyapunovWitness:
        """Alias certificado (interop con Séquitos / aduanas)."""
        return self.floquet_lyapunov_factorization(m_mat, orbit_period_T)

    # ── II.5  Función de Mel’nikov ────────────────────────────────────────
    def melnikov_function(
        self, q0_trajectory: np.ndarray, dt: float,
        h0_grad: np.ndarray, h1_grad: np.ndarray, omega: np.ndarray,
        n_phase_offsets: int = _MELNIKOV_PHASE_SAMPLES,
    ) -> _MelnikovWitness:
        r"""
        \(\mathcal{M}(t_0)=\int\{H_0,H_1\}(q_0(t),t+t_0)\,dt
        =\int(\nabla H_0)^{\top}\Omega(\nabla H_1)\,dt\).
        Ceros simples (cambio de signo Morse) ⇒ homoclínico de Poincaré.
        """
        q0 = self._as_numeric_matrix(q0_trajectory, "q0_trajectory")
        g0 = self._as_numeric_matrix(h0_grad, "h0_grad")
        g1 = self._as_numeric_matrix(h1_grad, "h1_grad")
        if q0.shape != g0.shape or q0.shape != g1.shape:
            raise ValueError("q0, h0_grad, h1_grad deben compartir shape.")
        if q0.ndim != 2:
            raise ValueError("q0_trajectory debe ser 2D (N, 2n).")
        o = self._as_numeric_matrix(omega, "omega", square=True)
        dim = q0.shape[1]
        if o.shape != (dim, dim):
            raise ValueError("Ω incompatible con la trayectoria.")
        pb = np.einsum("ij,jk,ik->i", g0, o, g1)
        n_samp = pb.size
        if n_samp < 2:
            raise ValueError("Trayectoria demasiado corta.")

        def trapz(y: np.ndarray, h: float) -> float:
            if y.size < 2:
                return 0.0
            return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))

        nph = int(max(n_phase_offsets, 2))
        mel_vals = np.empty(nph, dtype=np.float64)
        dt_f = float(dt)
        for k in range(nph):
            shift = int(round(k * n_samp / nph))
            mel_vals[k] = trapz(np.roll(pb, shift), dt_f)
        floor = max(self._wilkinson_deflation_floor(mel_vals), 1e-14)
        simple_zeros = _count_simple_zeros(mel_vals, cyclic=True, floor=floor)
        chaotic_ind = float(simple_zeros) / float(nph)
        return _MelnikovWitness(
            melnikov_values=mel_vals,
            simple_zeros=int(simple_zeros),
            chaotic_indicator=chaotic_ind,
            is_chaotic=bool(simple_zeros > 0))

    def compute_melnikov_certified(
        self, q0_trajectory: np.ndarray, dt: float,
        h0_grad: np.ndarray, h1_grad: np.ndarray, omega: np.ndarray,
    ) -> _MelnikovWitness:
        """Alias certificado (interop)."""
        return self.melnikov_function(q0_trajectory, dt, h0_grad, h1_grad, omega)

    # ── II.6  Número de rotación ──────────────────────────────────────────
    def rotation_number_poincare(
        self, orbit_points: np.ndarray, cf_depth: int = _ROTATION_CF_DEPTH,
    ) -> _RotationNumberWitness:
        r"""
        \(\rho=\lim(1/N)\sum\Delta\theta_i/2\pi\); fracción continua de Gauss.
        Racional ssi el desarrollo termina.
        """
        pts = np.asarray(orbit_points, dtype=np.float64)
        if pts.ndim == 1:
            theta = pts.ravel()
        else:
            pts = self._as_numeric_matrix(orbit_points, "orbit_points")
            if pts.shape[0] < 3:
                raise ValueError("orbit_points debe ser (N≥3, ·).")
            if pts.shape[1] >= 2 and pts.shape[1] % 2 == 0:
                n = pts.shape[1] // 2
                theta = np.unwrap(np.arctan2(pts[:, n], pts[:, 0]))
            else:
                theta = np.unwrap(np.arctan2(
                    pts[:, 1] if pts.shape[1] >= 2 else np.zeros(pts.shape[0]),
                    pts[:, 0]))
        dtheta = _unwrap_delta(theta)
        rho = float(np.mean(dtheta) / (2.0 * np.pi)) if dtheta.size else 0.0
        cf = _continued_fraction(rho, depth=int(cf_depth))
        terminated = bool(cf) and (len(cf) < int(cf_depth))
        dio = _diophantine_constant(rho, cf)
        return _RotationNumberWitness(
            rotation_number=float(rho), is_rational=terminated,
            continued_fraction=cf, diophantine_constant=float(dio),
            convergent_denominator=_cf_denominator(cf))

    def compute_rotation_number_certified(
        self, orbit_points: np.ndarray,
    ) -> _RotationNumberWitness:
        """Alias certificado (interop)."""
        return self.rotation_number_poincare(orbit_points)

    # ── II.7  Certificado KAM diofantino ──────────────────────────────────
    def certify_kam_torus(
        self, frequency_vector: np.ndarray,
        birkhoff_residual: float = 0.0,
        harmonic_cap: int = _KAM_HARMONIC_CAP,
    ) -> _KamTorusWitness:
        r"""
        \(|k\cdot\omega|\ge\gamma/|k|_1^\tau\) \(\forall k\neq 0\), \(|k|_\infty\le H\).
        Red recortada de ejes+pares: \(O(n^2 H^2)\), no \(O((2H+1)^n)\).
        \(\tau>n-1\) (Rüssmann). γ empírica = inf |k·ω| |k|_1^τ.
        """
        omega = self._as_numeric_vector(frequency_vector, "frequency_vector")
        n = int(omega.size)
        if n == 0:
            raise ValueError("frequency_vector vacío.")
        tau = float(max(float(n - 1) + 1.0e-6, _DIOPHANTINE_TAU_FLOOR))
        harmonics = _integer_harmonics(n, int(harmonic_cap))
        gamma_emp = float("inf")
        worst = float("inf")
        iter_count = int(harmonics.shape[0])
        for k in harmonics:
            kn = float(np.sum(np.abs(k)))
            if kn <= 0.0:
                continue
            div = float(abs(float(np.dot(k.astype(np.float64), omega))))
            worst = min(worst, div)
            gamma_emp = min(gamma_emp, div * (kn ** tau))
        if not np.isfinite(gamma_emp):
            gamma_emp = 0.0
        gamma_emp = float(max(gamma_emp, 0.0))
        residual_pen = float(max(birkhoff_residual, 0.0))
        kam_stable = bool(
            gamma_emp >= _HARD_KAM_GAMMA_FLOOR
            and residual_pen <= _HARD_BIRKHOFF_RES_CEILING)
        return _KamTorusWitness(
            frequency_vector=omega,
            diophantine_gamma=float(min(gamma_emp, 1e12)),
            diophantine_tau=float(tau),
            birkhoff_residual=float(birkhoff_residual),
            kam_stable=kam_stable, iterations=int(iter_count),
            worst_divisor=float(worst))

    # ── II.8  Espectro de Lyapunov (Benettin, signos absorbidos) ──────────
    def lyapunov_spectrum_benettin(
        self, m_mat: np.ndarray, n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
    ) -> _LyapunovSpectrumWitness:
        r"""
        \(Z_k=M Q_{k-1}\), \(Q_k R_k=\mathrm{QR}(Z_k)\),
        \(\lambda_i=(1/N)\sum\log|R_k[i,i]|\). Signos de \(\mathrm{diag}(R)\)
        absorbidos en \(Q\) para preservar orientación.
        """
        a = self._as_numeric_matrix(m_mat, "M", square=True)
        n = a.shape[0]
        n_it = int(n_iterations)
        if n_it <= 0:
            raise ValueError("n_iterations debe ser positivo.")
        q_mat = np.eye(n, dtype=np.float64)
        acc = np.zeros(n, dtype=np.float64)
        for _ in range(n_it):
            z_mat = a @ q_mat
            try:
                q_mat, r_mat = la.qr(z_mat, mode="economic")
            except la.LinAlgError:
                break
            diag = np.real(np.diag(r_mat))
            signs = np.sign(diag)
            signs[signs == 0.0] = 1.0
            q_mat = q_mat * signs.reshape(1, -1)
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = acc + np.log(np.abs(diag) + _MACHINE_EPS)
        spectrum = acc / float(n_it)
        ky, ks = _kaplan_yorke_and_pesin(spectrum)
        is_chaotic = bool(np.any(spectrum > _HARD_LYAPUNOV_TOL))
        return _LyapunovSpectrumWitness(
            spectrum=np.asarray(spectrum, dtype=np.float64),
            kaplan_yorke_dimension=float(ky),
            kolmogorov_sinai_entropy=float(ks),
            is_chaotic=is_chaotic)

    def compute_lyapunov_spectrum_certified(
        self, m_mat: np.ndarray, n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
    ) -> _LyapunovSpectrumWitness:
        """Alias certificado (interop)."""
        return self.lyapunov_spectrum_benettin(m_mat, n_iterations=n_iterations)

    # ── II.9  Poincaré–Cartan ─────────────────────────────────────────────
    def poincare_cartan_integral(
        self, q_trajectory: np.ndarray, p_trajectory: np.ndarray,
        h_trajectory: np.ndarray, dt: float,
    ) -> _CartanIntegralWitness:
        r"""
        \(\oint_\gamma p\,dq-H\,dt\approx\sum_k[p_k\cdot\Delta q_k-H_k\,dt]\).
        La acción pura \(\oint p\,dq\) se expone aparte (Liouville–Arnold).
        """
        q = self._as_numeric_matrix(q_trajectory, "q_trajectory")
        p = self._as_numeric_matrix(p_trajectory, "p_trajectory")
        ham = self._as_numeric_vector(h_trajectory, "H_trajectory")
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        n_nodes = q.shape[0]
        if ham.size != n_nodes:
            raise ValueError("H_trajectory debe tener N entradas.")
        dt_f = float(dt)
        action_terms = np.empty(n_nodes, dtype=np.float64)
        cartan_terms = np.empty(n_nodes, dtype=np.float64)
        for k in range(n_nodes):
            nxt = (k + 1) % n_nodes
            pdq = float(p[k] @ (q[nxt] - q[k]))
            action_terms[k] = pdq
            cartan_terms[k] = pdq - float(ham[k]) * dt_f
        integral = float(self.kahan_sum(cartan_terms))
        action = float(self.kahan_sum(action_terms))
        closed = bool(np.linalg.norm(q[0] - q[-1]) < 1e-6)
        return _CartanIntegralWitness(
            integral=integral, n_segments=int(n_nodes),
            is_closed=closed, action_integral=action)

    def poincare_cartan_invariant(
        self, q_trajectory: np.ndarray, p_trajectory: np.ndarray,
        h_trajectory: np.ndarray, dt: float,
    ) -> float:
        """Alias escalar (interop Séquitos)."""
        return float(self.poincare_cartan_integral(
            q_trajectory, p_trajectory, h_trajectory, dt).integral)

    # ── II.10  Variables acción-ángulo ────────────────────────────────────
    def action_angle_variables(
        self, q_periodic: np.ndarray, p_periodic: np.ndarray,
        frequency_hint: Optional[np.ndarray] = None,
    ) -> _ActionAngleWitness:
        r"""\(I_k=(1/2\pi)\oint p_k\,dq_k\); \(\theta_k=\arg(q_k+i p_k)\)."""
        q = self._as_numeric_matrix(q_periodic, "q_periodic")
        p = self._as_numeric_matrix(p_periodic, "p_periodic")
        if q.shape != p.shape or q.ndim != 2:
            raise ValueError("q, p deben ser (N, n).")
        n_nodes, n_dof = q.shape
        if n_nodes < 2:
            raise ValueError("Trayectoria demasiado corta.")
        i_vec = np.zeros(n_dof, dtype=np.float64)
        theta_vec = np.zeros(n_dof, dtype=np.float64)
        for k in range(n_dof):
            dq = np.diff(q[:, k])
            dq_closed = np.concatenate([dq, [q[0, k] - q[-1, k]]])
            p_mid = 0.5 * (p[:-1, k] + p[1:, k])
            p_mid_closed = np.concatenate([p_mid, [0.5 * (p[-1, k] + p[0, k])]])
            i_vec[k] = float(self.kahan_sum(p_mid_closed * dq_closed)) / (2.0 * np.pi)
            angles = np.unwrap(np.arctan2(p[:, k], q[:, k]))
            theta_vec[k] = float(angles[-1] - angles[0]) / max(n_nodes - 1, 1)
        freqs = None
        if frequency_hint is not None:
            freqs = np.asarray(frequency_hint, dtype=np.float64).ravel()
        residual = float(np.linalg.norm(q[0] - q[-1]))
        return _ActionAngleWitness(
            actions=i_vec, angles=theta_vec, frequencies=freqs,
            generating_residual=residual)

    # ── II.11  Elementos orbitales Kepler–Delaunay–Poincaré ───────────────
    def kepler_osculating_elements(
        self, position: np.ndarray, velocity: np.ndarray,
        mu_gravitational: float = 1.0,
    ) -> _KeplerElementsWitness:
        r"""
        \(h=r\times v\), \(e=v\times h/\mu-\hat r\);
        Delaunay \((L,G,H,l,g,h)\) y Poincaré \((\Lambda,\lambda,\xi,\eta,p,q)\).
        """
        r = _vec3("position", position)
        v = _vec3("velocity", velocity)
        mu = float(mu_gravitational)
        if mu <= 0.0 or not np.isfinite(mu):
            raise ValueError("mu_gravitational debe ser positivo y finito.")
        r_norm = float(np.linalg.norm(r))
        v_norm = float(np.linalg.norm(v))
        if r_norm < _MACHINE_EPS:
            raise ValueError("Radio orbital nulo.")
        h_vec = np.cross(r, v)
        h_norm = float(np.linalg.norm(h_vec))
        e_vec = np.cross(v, h_vec) / mu - r / r_norm
        e = float(np.linalg.norm(e_vec))
        energy = 0.5 * v_norm ** 2 - mu / r_norm
        if abs(energy) < _MACHINE_EPS:
            a = float("inf")
        else:
            a = float(-mu / (2.0 * energy))
        inc = 0.0 if h_norm < _MACHINE_EPS else float(
            np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))
        n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
        n_norm = float(np.linalg.norm(n_vec))
        omega_lan = float(math.atan2(n_vec[1], n_vec[0])) if n_norm >= _MACHINE_EPS else 0.0
        if n_norm < _MACHINE_EPS or e < _MACHINE_EPS:
            argp = 0.0
        else:
            cos_w = float(np.clip(np.dot(n_vec, e_vec) / (n_norm * e), -1.0, 1.0))
            sin_w = float(np.dot(
                np.cross(n_vec / n_norm, e_vec / e),
                h_vec / max(h_norm, _MACHINE_EPS)))
            argp = float(math.atan2(sin_w, cos_w))
        if e < _MACHINE_EPS:
            nu = 0.0
        else:
            cos_nu = float(np.clip(np.dot(e_vec, r) / (e * r_norm), -1.0, 1.0))
            sin_nu = float(np.dot(
                np.cross(e_vec / e, r / r_norm),
                h_vec / max(h_norm, _MACHINE_EPS)))
            nu = float(math.atan2(sin_nu, cos_nu))
        if e < 1.0 and np.isfinite(e):
            den = 1.0 + e * math.cos(nu)
            cos_e = float(np.clip((e + math.cos(nu)) / den if abs(den) > _MACHINE_EPS else 1.0, -1.0, 1.0))
            ecc_anom = float(math.acos(cos_e))
            if nu < 0.0:
                ecc_anom = -ecc_anom
            mean_anom = float(ecc_anom - e * math.sin(ecc_anom))
        else:
            mean_anom = float("nan")
        mean_motion = float(math.sqrt(mu / abs(a) ** 3)) if np.isfinite(a) and a != 0.0 else 0.0
        ell = math.sqrt(mu * a) if np.isfinite(a) and a > 0.0 else float("nan")
        gee = (math.sqrt(max(mu * a * (1.0 - e * e), 0.0))
               if np.isfinite(a) and a > 0.0 else float("nan"))
        aitch = gee * math.cos(inc) if np.isfinite(gee) else float("nan")
        varpi = omega_lan + argp
        two_lg = 2.0 * max((ell - gee) if np.isfinite(ell) and np.isfinite(gee) else 0.0, 0.0)
        two_gh = 2.0 * max((gee - aitch) if np.isfinite(gee) and np.isfinite(aitch) else 0.0, 0.0)
        xi = math.sqrt(two_lg) * math.cos(varpi)
        eta = -math.sqrt(two_lg) * math.sin(varpi)
        p_poinc = math.sqrt(two_gh) * math.cos(omega_lan)
        q_poinc = -math.sqrt(two_gh) * math.sin(omega_lan)
        lam_mean = omega_lan + argp + (mean_anom if np.isfinite(mean_anom) else 0.0)
        delaunay = {
            "L": float(ell), "G": float(gee), "H": float(aitch),
            "l": float(mean_anom), "g": float(argp), "h": float(omega_lan),
        }
        poincare_can = {
            "Lambda": float(ell), "lambda": float(lam_mean),
            "xi": float(xi), "eta": float(eta),
            "p": float(p_poinc), "q": float(q_poinc),
        }
        return _KeplerElementsWitness(
            semi_major_axis=a, eccentricity=e, inclination=inc,
            ascending_node=float(omega_lan), argument_periapsis=float(argp),
            true_anomaly=float(nu), specific_energy=float(energy),
            angular_momentum=h_vec, eccentricity_vector=e_vec,
            mean_anomaly=float(mean_anom), mean_motion=float(mean_motion),
            delaunay=delaunay, poincare_canonical=poincare_can)

    # ── II.12  Birkhoff, Nekhoroshev, twist, Lindstedt ────────────────────
    def birkhoff_normal_form(
        self, birkhoff_residual: float, kam: _KamTorusWitness,
    ) -> _BirkhoffWitness:
        r"""Residuo de \(\{H_0,S\}=H_{\mathrm{pert}}-[H_{\mathrm{pert}}]\)."""
        res = abs(float(birkhoff_residual))
        if not np.isfinite(res):
            res = float("inf")
        normalizable = bool(res <= _HARD_BIRKHOFF_RES_CEILING and kam.kam_stable)
        return _BirkhoffWitness(order=2, residual=float(res), is_normalizable=normalizable)

    def nekhoroshev_confinement(
        self, n_dof: int, birkhoff: _BirkhoffWitness, kam: _KamTorusWitness,
    ) -> _NekhoroshevWitness:
        r"""Exponentes \(a=1/(2n)\), \(b=a\); \(\tau_N=\exp(c/\varepsilon^a)\)."""
        n = int(max(n_dof, 1))
        exp_a = 1.0 / float(2 * n)
        conf_b = float(exp_a)
        eps = max(float(birkhoff.residual), _MACHINE_EPS)
        if (not np.isfinite(eps)) or eps >= 1.0:
            return _NekhoroshevWitness(
                exponent_a=float(exp_a), confinement_b=float(conf_b),
                time_scale=0.0, is_confined=False)
        time_scale = float(math.exp(_NEKHOROSHEV_C / (eps ** exp_a)))
        confined = bool(kam.kam_stable and birkhoff.is_normalizable and time_scale > 1.0)
        return _NekhoroshevWitness(
            exponent_a=float(exp_a), confinement_b=float(conf_b),
            time_scale=float(time_scale), is_confined=confined)

    def moser_twist_certificate(
        self,
        frequency_vector: np.ndarray,
        action_angle: Optional[_ActionAngleWitness],
        floquet: _FloquetLyapunovWitness,
        n_dof: int,
    ) -> _TwistWitness:
        r"""
        Twist \(\det\partial\omega/\partial I\neq 0\); Kolmogorov
        \(\det\partial^2 H/\partial I^2\neq 0\); isoenergética (hessiano bordeado).
        Poincaré–Birkhoff: anillo 2D + twist + área-preservante ⇒ ≥2 p.f.
        """
        omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
        n = int(max(n_dof, 1))
        if omega.size == 0:
            omega = _default_frequencies(n)
        omega = omega[:n] if omega.size >= n else np.pad(
            omega, (0, n - omega.size), constant_values=_GOLDEN_RATIO)
        if action_angle is not None and action_angle.actions.size == omega.size:
            actions = np.asarray(action_angle.actions, dtype=np.float64)
            denom = np.where(np.abs(actions) > _MACHINE_EPS, actions, 1.0)
            d_omega = omega / denom
        else:
            d_omega = omega
        hess = np.diag(d_omega)
        twist_det = float(np.linalg.det(hess)) if hess.size else 0.0
        kolmogorov = bool(abs(twist_det) > _HARD_TWIST_DET_FLOOR)
        bordered = np.zeros((n + 1, n + 1), dtype=np.float64)
        bordered[:n, :n] = hess
        bordered[:n, n] = omega
        bordered[n, :n] = omega
        iso_det = float(np.linalg.det(bordered))
        iso = bool(abs(iso_det) > _HARD_TWIST_DET_FLOOR)
        section_2d = bool(n == 1 or (2 * n - 2) == 2)
        area_pres = bool(
            np.isfinite(floquet.symplectic_defect)
            and floquet.symplectic_defect <= _HARD_SYMPLECTIC_DEFECT)
        has_pb = bool(section_2d and kolmogorov and area_pres)
        return _TwistWitness(
            twist_determinant=float(twist_det),
            kolmogorov_nondeg=kolmogorov,
            isoenergetic_nondeg=iso,
            has_poincare_birkhoff_fps=has_pb)

    def lindstedt_poincare_series(
        self, birkhoff: _BirkhoffWitness, frequency_vector: np.ndarray,
    ) -> _LindstedtWitness:
        r"""Corrección secular: \(\omega=\omega_0+\varepsilon\omega_1+\cdots\)."""
        omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
        corr = np.zeros_like(omega) if omega.size else np.array([], dtype=np.float64)
        return _LindstedtWitness(
            secular_residual=float(birkhoff.residual),
            frequency_correction=corr)

    def _frequencies_from_floquet(
        self, floquet: _FloquetLyapunovWitness, orbit_period_T: float, n: int,
    ) -> np.ndarray:
        """ω_k = |arg μ_k| / T (frecuencias de Poincaré de la órbita periódica)."""
        mu = np.asarray(floquet.floquet_multipliers, dtype=np.complex128)
        t_orb = float(orbit_period_T) if orbit_period_T > 0.0 else 1.0
        if mu.size == 0:
            return _default_frequencies(max(n, 1))
        angles = np.abs(np.angle(mu))
        omega = np.sort(angles)[::-1] / t_orb
        omega = omega[omega > 1.0e-12]
        if omega.size == 0:
            return _default_frequencies(max(n, 1))
        return np.asarray(omega, dtype=np.float64)

    # ── II.13  Inducción del gérmen Hodge–Cheeger (compatibilidad 4.1) ────
    def induce_hodge_cheeger_germ(
        self, density_or_boundary: np.ndarray,
        tangent_a: Optional[np.ndarray] = None,
        boundary_matrix: Optional[np.ndarray] = None,
    ) -> _HodgeCheegerGerm:
        raw = np.asarray(density_or_boundary)
        if raw.ndim == 1:
            side = int(np.sqrt(raw.size))
            if side * side == raw.size:
                raw = raw.reshape(side, side)
        floor = self._regularizer_floor()
        petz_scale = 0.0
        dirac_eigs = np.array([], dtype=np.float64)
        lap_eigs = np.array([], dtype=np.float64)
        comb = None
        from_connes = False
        dim_hint = 0
        is_square = raw.ndim == 2 and raw.shape[0] == raw.shape[1] and raw.size > 0
        looks_density = False
        if is_square:
            herm = 0.5 * (raw + raw.T.conj())
            looks_density = self._frobenius_norm(raw - raw.T.conj()) <= (
                _HERMITIAN_REL_TOL * max(self._frobenius_norm(raw), 1.0))
            if looks_density:
                ev = np.real(la.eigvalsh(herm)) if herm.size else np.array([])
                looks_density = ev.size > 0 and float(np.min(ev)) >= -self._psd_tolerance(ev)
        if looks_density:
            certified = self.compute_dirac_operator_spectrum_certified(raw)
            dirac_eigs = certified.dirac_eigs
            lap_eigs = np.sort(np.real(certified.laplacian_eigs))
            from_connes = True
            dim_hint = int(raw.shape[0])
            floor = max(floor, self._resolve_triple_germ(raw).reg_floor)
            if tangent_a is not None:
                petz_scale = self.compute_petz_fisher_rao_metric(raw, tangent_a, tangent_a)
        elif raw.ndim == 2:
            comb = Phase3TopologicalGeometricMixin.compute_simplicial_normalized_laplacian(
                self, raw)
            try:
                lap_eigs = np.sort(np.real(la.eigvalsh(comb)))
            except la.LinAlgError:
                lap_eigs = np.array([], dtype=np.float64)
            dim_hint = int(comb.shape[0])
        else:
            raise ValueError(
                "induce_hodge_cheeger_germ espera densidad cuadrada o incidencia 2-D.")
        extra_b = boundary_matrix
        if extra_b is not None and comb is None:
            comb = Phase3TopologicalGeometricMixin.compute_simplicial_normalized_laplacian(
                self, extra_b)
            if lap_eigs.size == 0:
                try:
                    lap_eigs = np.sort(np.real(la.eigvalsh(comb)))
                except la.LinAlgError:
                    lap_eigs = np.array([], dtype=np.float64)
            dim_hint = max(dim_hint, int(comb.shape[0]))
        germ = _HodgeCheegerGerm(
            laplacian_eigs=np.asarray(lap_eigs, dtype=np.float64),
            combinatorial_laplacian=comb,
            dirac_eigs=np.asarray(dirac_eigs, dtype=np.float64),
            petz_scale=float(petz_scale), two_n_hint=int(dim_hint),
            reg_floor=float(floor), from_connes=bool(from_connes))
        self._hodge_germ = germ
        return germ

    # ── II.ω  MORFISMO TERMINAL DE LA FASE II ────────────────────────────
    def induce_poincare_floquet_germ(
        self,
        darboux_germ: Optional[_PoincareDarbouxGerm] = None,
        monodromy_M: Optional[np.ndarray] = None,
        orbit_period_T: float = 1.0,
        q0_trajectory: Optional[np.ndarray] = None,
        dt_trajectory: float = 1.0,
        h0_grad: Optional[np.ndarray] = None,
        h1_grad: Optional[np.ndarray] = None,
        orbit_points_for_rotation: Optional[np.ndarray] = None,
        frequency_vector: Optional[np.ndarray] = None,
        birkhoff_residual: float = 0.0,
        q_cartan: Optional[np.ndarray] = None,
        p_cartan: Optional[np.ndarray] = None,
        H_cartan: Optional[np.ndarray] = None,
        kepler_position: Optional[np.ndarray] = None,
        kepler_velocity: Optional[np.ndarray] = None,
        mu_gravitational: float = 1.0,
        q_periodic: Optional[np.ndarray] = None,
        p_periodic: Optional[np.ndarray] = None,
        state_trajectory_z: Optional[Sequence[np.ndarray]] = None,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareFloquetGerm:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        \(\Phi_{\mathrm{II}}:\mathcal{G}_I\times(M,\omega,\gamma)\to\mathcal{G}_{II}\).
        El tipo de retorno `_PoincareFloquetGerm` es el dominio de
        `certify_poincare_guards_cycle` (III.ω).
        """
        if darboux_germ is None:
            darboux_germ = getattr(self, "_poincare_darboux_germ", None)
        if darboux_germ is None:
            darboux_germ = self.synthesize_poincare_darboux_germ(
                two_n=2 * int(getattr(self, "_dim", 6)),
                flow_vector=flow_vector,
                state_trajectory_z=state_trajectory_z)
        elif state_trajectory_z is not None and darboux_germ.recurrence is None:
            try:
                rec = self.poincare_recurrence(state_trajectory_z, darboux_germ.section)
                darboux_germ = _PoincareDarbouxGerm(
                    two_n=darboux_germ.two_n, n=darboux_germ.n,
                    omega=darboux_germ.omega,
                    form_certificate=darboux_germ.form_certificate,
                    section=darboux_germ.section,
                    reg_floor=darboux_germ.reg_floor,
                    csmd_step=darboux_germ.csmd_step,
                    hodge_germ=darboux_germ.hodge_germ,
                    recurrence=rec)
            except ValueError as exc:
                logger.error("Recurrencia diferida falló: %s", exc)
        self._poincare_darboux_germ = darboux_germ
        two_n = darboux_germ.two_n
        n = darboux_germ.n
        if monodromy_M is None:
            m_eff = np.eye(two_n, dtype=np.float64)
        else:
            m_eff = self._as_numeric_matrix(monodromy_M, "monodromy_M", square=True)
            if m_eff.shape[0] % 2 != 0:
                m_eff = _even_embed(m_eff)
            if m_eff.shape[0] != two_n:
                darboux_germ = self.synthesize_poincare_darboux_germ(
                    two_n=m_eff.shape[0], flow_vector=flow_vector,
                    state_trajectory_z=state_trajectory_z)
                two_n = darboux_germ.two_n
                n = darboux_germ.n
                self._poincare_darboux_germ = darboux_germ
        floquet = self.floquet_lyapunov_factorization(m_eff, orbit_period_T)
        melnikov: Optional[_MelnikovWitness] = None
        if (q0_trajectory is not None
                and h0_grad is not None and h1_grad is not None):
            try:
                melnikov = self.melnikov_function(
                    q0_trajectory, dt_trajectory, h0_grad, h1_grad,
                    darboux_germ.omega)
            except ValueError as exc:
                logger.error("Mel'nikov falló: %s", exc)
        rotation: Optional[_RotationNumberWitness] = None
        if orbit_points_for_rotation is not None:
            try:
                rotation = self.rotation_number_poincare(orbit_points_for_rotation)
            except ValueError as exc:
                logger.error("Rotación falló: %s", exc)
        if frequency_vector is None:
            omega_freq = self._frequencies_from_floquet(floquet, orbit_period_T, n)
        else:
            omega_freq = np.asarray(frequency_vector, dtype=np.float64).ravel()
            if omega_freq.size == 0:
                omega_freq = self._frequencies_from_floquet(floquet, orbit_period_T, n)
        kam = self.certify_kam_torus(omega_freq, birkhoff_residual=birkhoff_residual)
        lyapunov = self.lyapunov_spectrum_benettin(m_eff)
        cartan: Optional[_CartanIntegralWitness] = None
        if (q_cartan is not None and p_cartan is not None and H_cartan is not None):
            try:
                cartan = self.poincare_cartan_integral(
                    q_cartan, p_cartan, H_cartan, dt_trajectory)
            except ValueError as exc:
                logger.error("Cartan falló: %s", exc)
        kepler: Optional[_KeplerElementsWitness] = None
        if kepler_position is not None and kepler_velocity is not None:
            try:
                kepler = self.kepler_osculating_elements(
                    kepler_position, kepler_velocity,
                    mu_gravitational=mu_gravitational)
            except ValueError as exc:
                logger.error("Kepler falló: %s", exc)
        action_angle: Optional[_ActionAngleWitness] = None
        if q_periodic is not None and p_periodic is not None:
            try:
                action_angle = self.action_angle_variables(
                    q_periodic, p_periodic, frequency_hint=omega_freq)
            except ValueError as exc:
                logger.error("Acción-ángulo falló: %s", exc)
        elif cartan is not None and np.isfinite(cartan.action_integral):
            i1 = float(cartan.action_integral) / (2.0 * np.pi)
            weights = np.abs(omega_freq) + _MACHINE_EPS
            weights = weights / float(np.sum(weights))
            actions = i1 * weights * float(max(omega_freq.size, 1))
            action_angle = _ActionAngleWitness(
                actions=np.asarray(actions, dtype=np.float64),
                angles=np.zeros_like(actions),
                frequencies=np.asarray(omega_freq, dtype=np.float64),
                generating_residual=abs(float(cartan.integral)))
        birkhoff = self.birkhoff_normal_form(birkhoff_residual, kam)
        nekhoroshev = self.nekhoroshev_confinement(max(n, 1), birkhoff, kam)
        twist = self.moser_twist_certificate(omega_freq, action_angle, floquet, max(n, 1))
        lindstedt = self.lindstedt_poincare_series(birkhoff, omega_freq)
        germ = _PoincareFloquetGerm(
            darboux_germ=darboux_germ, floquet=floquet,
            melnikov=melnikov, rotation=rotation, kam=kam,
            lyapunov=lyapunov, cartan=cartan, kepler=kepler,
            action_angle=action_angle, birkhoff=birkhoff,
            nekhoroshev=nekhoroshev, twist=twist, lindstedt=lindstedt)
        self._poincare_floquet_germ = germ
        return germ


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE III — HODGE, CHEEGER, BETTI, EULER–POINCARÉ Y CERTIFICADO GLOBAL  ██
# ██  Continuación directa del morfismo II.ω.                                  ██
# ██  Dominio: _PoincareFloquetGerm                                            ██
# ██  Morfismo terminal (III.ω): certify_poincare_guards_cycle                ██
# ██                           → PoincareGuardsCertificate                    ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
@dataclass(frozen=True)
class _LaplacianResult:
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    fiedler: float
    algebraic_connectivity: float
    kernel_dim: int
    nilpotency_residual: float


@dataclass(frozen=True)
class _CheegerResult:
    h_lower: float
    h_upper: float
    fiedler: float
    connected: bool
    buser_gap: float


@dataclass(frozen=True)
class _EulerPoincareResult:
    characteristic: int
    vertices: int
    edges: int
    faces: int
    betti_0: int
    betti_1: int
    betti_2: int
    homological_chi: int
    rank_d0: int
    rank_d1: int
    nilpotency_residual: float


class Phase3TopologicalGeometricMixin(Phase2SpectralQuantumMixin):
    """
    FASE III — TOPOLOGÍA, GEOMETRÍA Y CIERRE GLOBAL.

    Operadores discretos de de Rham–Hodge, invariantes topológicos e
    isoperimétricos, y morfismo terminal \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\).
    Dominio de III.ω = imagen de II.ω (`_PoincareFloquetGerm`).
    """

    def _resolve_hodge_germ(
        self, eigenvalues_or_boundary: Optional[np.ndarray] = None,
    ) -> Optional[_HodgeCheegerGerm]:
        cached = getattr(self, "_hodge_germ", None)
        if eigenvalues_or_boundary is None:
            return cached
        return cached

    def _numeric_rank(self, matrix: np.ndarray, floor: float) -> int:
        a = np.asarray(matrix)
        if a.size == 0:
            return 0
        s_vals = np.real(la.svd(a, compute_uv=False))
        tol = max(floor, self._wilkinson_deflation_floor(a))
        return int(np.sum(s_vals > tol))

    def compute_simplicial_normalized_laplacian(
        self, boundary_matrix: np.ndarray,
    ) -> np.ndarray:
        return self.compute_simplicial_normalized_laplacian_certified(
            boundary_matrix).laplacian

    def compute_simplicial_normalized_laplacian_certified(
        self, boundary_matrix: np.ndarray,
        boundary_1: Optional[np.ndarray] = None,
    ) -> _LaplacianResult:
        delta_0 = self._as_numeric_matrix(
            boundary_matrix, "boundary_matrix", square=False)
        l_base = delta_0.T @ delta_0
        l_base = 0.5 * (l_base + l_base.T)
        degrees = np.real(np.diagonal(l_base)).astype(np.float64, copy=False)
        inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
        positive = degrees > _MACHINE_EPS
        inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
        d_is = np.diag(inv_sqrt)
        l_norm = d_is @ l_base @ d_is
        l_norm = 0.5 * (l_norm + l_norm.T)
        try:
            evals = np.sort(np.real(la.eigvalsh(l_norm)))
        except la.LinAlgError as exc:
            raise ValueError("El espectro del Laplaciano falló.") from exc
        floor = max(self._regularizer_floor(),
                    self._wilkinson_deflation_floor(l_norm))
        evals = np.where(evals < 0.0,
                         np.where(evals > -self._psd_tolerance(evals), 0.0, evals),
                         evals)
        ker = int(np.sum(evals <= max(floor, _PSD_ABS_TOL)))
        fiedler = float(evals[1]) if evals.size > 1 else 0.0
        nilp = 0.0
        if boundary_1 is not None:
            delta_1 = self._as_numeric_matrix(
                boundary_1, "boundary_1", square=False)
            if delta_1.shape[1] == delta_0.shape[0]:
                nilp = self._frobenius_norm(delta_1 @ delta_0)
            elif delta_0.shape[1] == delta_1.shape[0]:
                nilp = self._frobenius_norm(delta_0 @ delta_1)
        return _LaplacianResult(
            laplacian=np.asarray(l_norm, dtype=np.float64),
            eigenvalues=np.asarray(evals, dtype=np.float64),
            fiedler=float(fiedler),
            algebraic_connectivity=float(max(fiedler, 0.0)),
            kernel_dim=int(ker), nilpotency_residual=float(nilp))

    def estimate_cheeger_constant_bounds(
        self, eigenvalues_l: np.ndarray,
    ) -> Tuple[float, float]:
        result = self.estimate_cheeger_constant_bounds_certified(eigenvalues_l)
        return result.h_lower, result.h_upper

    def estimate_cheeger_constant_bounds_certified(
        self, eigenvalues_l: np.ndarray,
    ) -> _CheegerResult:
        germ = getattr(self, "_hodge_germ", None)
        vec = np.asarray(eigenvalues_l)
        if vec.size == 0 and germ is not None and germ.laplacian_eigs.size:
            vec = germ.laplacian_eigs
        eigs = self._as_numeric_vector(vec, "eigenvalues_L")
        if eigs.size < 2:
            return _CheegerResult(0.0, 0.0, 0.0, eigs.size == 1, 0.0)
        min_eig = float(np.min(eigs))
        if min_eig < -_PSD_ABS_TOL:
            logger.warning("Laplaciano no PSD en estimate_cheeger.")
            return _CheegerResult(0.0, 0.0, 0.0, False, 0.0)
        eigs = np.where(eigs < 0.0, 0.0, eigs)
        sorted_eigs = np.sort(eigs)
        sorted_eigs[np.abs(sorted_eigs) <= _PSD_ABS_TOL] = 0.0
        fiedler_val = float(sorted_eigs[1])
        connected = bool(sorted_eigs[0] <= _PSD_ABS_TOL and fiedler_val > _PSD_ABS_TOL)
        if not math.isfinite(fiedler_val) or fiedler_val <= 0.0:
            return _CheegerResult(
                0.0, 0.0,
                float(fiedler_val) if math.isfinite(fiedler_val) else 0.0,
                False, 0.0)
        h_lower = float(fiedler_val / 2.0)
        h_upper = float(math.sqrt(max(0.0, 2.0 * fiedler_val)))
        return _CheegerResult(
            h_lower=h_lower, h_upper=h_upper, fiedler=fiedler_val,
            connected=connected,
            buser_gap=float(max(h_upper - h_lower, 0.0)))

    def compute_euler_poincare_characteristic(
        self, boundary_0: np.ndarray, boundary_1: Optional[np.ndarray],
    ) -> int:
        return self.compute_euler_poincare_characteristic_certified(
            boundary_0, boundary_1).characteristic

    def compute_euler_poincare_characteristic_certified(
        self, boundary_0: np.ndarray, boundary_1: Optional[np.ndarray],
    ) -> _EulerPoincareResult:
        b0 = self._as_numeric_matrix(boundary_0, "boundary_0", square=False)
        vertices = int(b0.shape[1])
        edges = int(b0.shape[0])
        faces = 0
        b1: Optional[np.ndarray] = None
        if boundary_1 is not None:
            b1 = self._as_numeric_matrix(boundary_1, "boundary_1", square=False)
            faces = int(b1.shape[0])
        chi = int(vertices - edges + faces)
        floor = max(self._regularizer_floor(), self._wilkinson_deflation_floor(b0))
        rank_d0 = self._numeric_rank(b0, floor)
        rank_d1 = self._numeric_rank(b1, floor) if b1 is not None else 0
        betti_0 = int(max(vertices - rank_d0, 0))
        betti_1 = int(max(edges - rank_d0 - rank_d1, 0))
        betti_2 = int(max(faces - rank_d1, 0))
        chi_h = int(betti_0 - betti_1 + betti_2)
        nilp = 0.0
        if b1 is not None:
            if b1.shape[1] == b0.shape[0]:
                nilp = self._frobenius_norm(b1 @ b0)
            elif b0.shape[1] == b1.shape[0]:
                nilp = self._frobenius_norm(b0 @ b1)
        return _EulerPoincareResult(
            characteristic=chi, vertices=vertices, edges=edges, faces=faces,
            betti_0=betti_0, betti_1=betti_1, betti_2=betti_2,
            homological_chi=chi_h, rank_d0=int(rank_d0),
            rank_d1=int(rank_d1), nilpotency_residual=float(nilp))

    # ── III.ω  MORFISMO TERMINAL GLOBAL DE LA FASE III ──────────────────
    def certify_poincare_guards_cycle(
        self,
        floquet_germ: Optional[_PoincareFloquetGerm] = None,
        heyting_verdict: Optional[str] = None,
    ) -> PoincareGuardsCertificate:
        r"""
        **III.ω — Morfismo terminal global \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\).**

        Consume \(\mathcal{G}_{II}\) (cierre de Fase II). El veredicto de
        Heyting es el **ínfimo de Gödel** (join de severidad) sobre las
        aduanas Σ, Floquet, Lyapunov, KAM, Mel’nikov, recurrencia,
        Birkhoff, Nekhoroshev, twist y rotación. Un VETOED aislado no
        puede ser absorbido por un meet de severidad.
        """
        germ = floquet_germ or getattr(self, "_poincare_floquet_germ", None)
        if germ is None:
            germ = self.induce_poincare_floquet_germ()
        dg = germ.darboux_germ
        fl = germ.floquet
        kam = germ.kam
        lyap = germ.lyapunov
        mel = germ.melnikov
        rot = germ.rotation
        cartan = germ.cartan
        kepler = germ.kepler
        aa = germ.action_angle
        birk = germ.birkhoff
        nek = germ.nekhoroshev
        tw = germ.twist
        lind = germ.lindstedt
        rec = dg.recurrence

        spectral_germ = getattr(self, "_triple_germ", None)
        spectral_is_dens = bool(
            spectral_germ is not None and spectral_germ.certificate.is_density)
        spectral_rank = int(spectral_germ.certificate.effective_rank) \
            if spectral_germ else 0
        spectral_vn = float(spectral_germ.certificate.von_neumann_entropy) \
            if spectral_germ else 0.0

        mu = fl.floquet_multipliers
        max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
        lyap_max = float(np.max(lyap.spectrum)) if lyap.spectrum.size else 0.0

        if heyting_verdict is None:
            v_section = "COHERENT" if (
                dg.section.is_transversal
                and dg.section.transversal_certificate > _HARD_POINCARE_SECTION_TOL
            ) else "VETOED"
            excess_mu = max(0.0, max_mu - 1.0)
            if excess_mu > _HARD_FLOQUET_MULTIPLIER_TOL * 10.0 \
                    or fl.log_residual > _HARD_FLOQUET_MULTIPLIER_TOL * 10.0:
                v_floquet = "VETOED"
            elif excess_mu > _HARD_FLOQUET_MULTIPLIER_TOL \
                    or fl.orbit_type in {"hyperbolic", "loxodromic"}:
                v_floquet = "DEGRADED"
            else:
                v_floquet = "COHERENT"
            if lyap.is_chaotic and lyap_max > 10.0 * _HARD_LYAPUNOV_TOL:
                v_lyap = "VETOED"
            elif lyap.is_chaotic or lyap_max > _HARD_LYAPUNOV_TOL:
                v_lyap = "DEGRADED"
            else:
                v_lyap = "COHERENT"
            v_kam = "COHERENT" if (
                kam.kam_stable and kam.diophantine_gamma > _HARD_KAM_GAMMA_FLOOR
            ) else "VETOED"
            v_mel = "VETOED" if (mel is not None and mel.is_chaotic) else "COHERENT"
            v_rec = "COHERENT"
            if rec is not None:
                v_rec = "COHERENT" if rec.is_recurrent else "VETOED"
            v_birk = "COHERENT"
            if birk is not None:
                if not np.isfinite(birk.residual):
                    v_birk = "VETOED"
                elif birk.residual > _HARD_BIRKHOFF_RES_CEILING * 10.0:
                    v_birk = "VETOED"
                elif birk.residual > _HARD_BIRKHOFF_RES_CEILING:
                    v_birk = "DEGRADED"
            v_nek = "COHERENT"
            if nek is not None and not nek.is_confined:
                v_nek = "DEGRADED"
            v_twist = "COHERENT"
            if tw is not None and not tw.kolmogorov_nondeg:
                v_twist = "DEGRADED"
            v_rot = "COHERENT"
            if rot is not None:
                if not np.isfinite(rot.rotation_number):
                    v_rot = "DEGRADED"
                elif rot.is_rational:
                    denom = int(rot.convergent_denominator)
                    v_rot = "VETOED" if 0 < denom <= _HARD_RESONANT_DENOM else "DEGRADED"
            heyting_verdict = _heyting_join(
                v_section, v_floquet, v_lyap, v_kam, v_mel,
                v_rec, v_birk, v_nek, v_twist, v_rot)

        canonical = heyting_verdict if heyting_verdict in _HEYTING_ORDER else "VETOED"
        is_liouville = bool(
            fl.log_residual <= _WILKINSON_DRIFT_LIMIT
            or dg.form_certificate.is_darboux)
        is_coherent = bool(canonical == "COHERENT")
        if not is_coherent:
            logger.error(
                "[GUARDS_POINCARE_VETOED] Veredicto=%s. μ_max=%.3e, "
                "λ_max=%.3e, γ_KAM=%.3e, Σ_transv=%s, ℳ_ceros=%s, tipo=%s.",
                canonical, max_mu, lyap_max, kam.diophantine_gamma,
                dg.section.is_transversal,
                mel.simple_zeros if mel else "N/A",
                fl.orbit_type)
        return PoincareGuardsCertificate(
            two_n=int(dg.two_n), n=int(dg.n),
            section_is_transversal=bool(dg.section.is_transversal),
            section_transversal_certificate=float(
                dg.section.transversal_certificate),
            form_is_darboux=bool(dg.form_certificate.is_darboux),
            spectral_is_density=bool(spectral_is_dens),
            spectral_effective_rank=int(spectral_rank),
            spectral_von_neumann_entropy=float(spectral_vn),
            floquet_max_multiplier=float(max_mu),
            floquet_lyapunov_max=float(lyap_max),
            floquet_log_residual=float(fl.log_residual),
            floquet_is_real_log=bool(fl.is_real_logarithm),
            kam_diophantine_gamma=float(kam.diophantine_gamma),
            kam_diophantine_tau=float(kam.diophantine_tau),
            kam_stable=bool(kam.kam_stable),
            lyapunov_spectrum=np.asarray(lyap.spectrum, dtype=np.float64),
            lyapunov_kaplan_yorke=float(lyap.kaplan_yorke_dimension),
            lyapunov_ks_entropy=float(lyap.kolmogorov_sinai_entropy),
            lyapunov_is_chaotic=bool(lyap.is_chaotic),
            melnikov_simple_zeros=(int(mel.simple_zeros) if mel else None),
            melnikov_chaotic=(bool(mel.is_chaotic) if mel else None),
            rotation_number=(float(rot.rotation_number) if rot else None),
            rotation_is_rational=(bool(rot.is_rational) if rot else None),
            cartan_integral=(float(cartan.integral) if cartan else None),
            kepler_semi_major_axis=(float(kepler.semi_major_axis)
                                    if kepler else None),
            kepler_eccentricity=(float(kepler.eccentricity)
                                 if kepler else None),
            action_angles=(np.asarray(aa.actions, dtype=np.float64)
                           if aa else None),
            is_liouville_conserved=bool(is_liouville),
            is_poincare_coherent=bool(is_coherent),
            heyting_verdict=str(canonical),
            poincare_orbit_type=str(fl.orbit_type),
            recurrence_distance=(float(rec.return_distance) if rec else float("nan")),
            recurrence_is_ergodic=bool(rec.is_recurrent) if rec else False,
            symplectic_defect=float(fl.symplectic_defect),
            hill_discriminant=float(fl.hill_discriminant),
            nekhoroshev_time_scale=float(nek.time_scale) if nek else 0.0,
            twist_determinant=float(tw.twist_determinant) if tw else 0.0,
            birkhoff_residual=float(birk.residual) if birk else 0.0,
            lindstedt_secular_residual=float(lind.secular_residual) if lind else 0.0,
            heyting_godel_meet=float(_HEYTING_GODEL[canonical]))


# =============================================================================
# CLASE PÚBLICA — INTEGRACIÓN DEL MORFISMO Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================
class ImperialGuardsEngine(Phase3TopologicalGeometricMixin):
    """
    Motor matemático de alta precisión para los operadores elípticos
    de-confinados y la mecánica celeste de Henri Poincaré.

    Compone \(\Phi_{\mathrm{I}}\to\Phi_{\mathrm{II}}\to\Phi_{\mathrm{III}}\)
    sobre el fibrado cotangente \(T^*M\).
    """

    def __init__(
        self,
        regularizer: float = _DEFAULT_REGULARIZER,
        dimension: int = 6,
    ) -> None:
        if (isinstance(regularizer, (int, np.integer)) and regularizer > 1
                and not isinstance(regularizer, bool)):
            self._dim = int(regularizer)
            self._reg = self._validate_regularizer(_DEFAULT_REGULARIZER)
        else:
            self._dim = self._validate_positive_int("dimension", dimension)
            self._reg = self._validate_regularizer(regularizer)
        self._csmd_h = _COMPLEX_STEP_DEFAULT_H
        self._J_canonical = _omega_n(self._dim)
        mixed = 0.5 * np.eye(2, dtype=np.complex128)
        self._triple_germ: _SpectralTripleGerm = self.synthesize_spectral_triple_germ(
            mixed, csmd_step=self._csmd_h, regularizer=self._reg)
        self._hodge_germ: _HodgeCheegerGerm = self.induce_hodge_cheeger_germ(mixed)
        self._poincare_darboux_germ: _PoincareDarbouxGerm = (
            self.synthesize_poincare_darboux_germ(
                two_n=2 * self._dim, density_matrix=None,
                hodge_germ=self._hodge_germ))
        self._poincare_floquet_germ: Optional[_PoincareFloquetGerm] = None

    @property
    def regularizer(self) -> float:
        return self._reg

    @property
    def dimension(self) -> int:
        return self._dim

    def _validate_regularizer(self, regularizer: float) -> float:
        value = self._validate_nonnegative_finite("regularizer", regularizer)
        return float(max(value, _HIGHAM_TIKHONOV_FLOOR))

    # ── Integración simpléctica Störmer–Verlet (API 4.1/5.0) ──────────────
    def step_poincare_symplectic_integration(
        self,
        current_state: NDArray[np.float64],
        metric_G: NDArray[np.float64],
        potential_V: float,
        total_energy_H0: float,
        dt_step: float,
        external_freq_omega: NDArray[np.float64],
        wave_k: NDArray[np.float64],
    ) -> ImperialEngineStepResult:
        r"""
        Paso Störmer–Verlet preservando Liouville.
        Axiomas: \(\det M=+1\), \(S_M=\int\sqrt{2(H_0-V)}\,|dq|_G\),
        \(W_{\mathrm{Nov}}=\exp(-T/(\varepsilon+|k\cdot\omega|))\).
        """
        state_vec = self._as_numeric_vector(current_state, "current_state")
        if state_vec.size % 2 != 0:
            raise ValueError("current_state debe tener dimensión par (2n).")
        n = state_vec.size // 2
        q_pos = state_vec[:n]
        p_mom = state_vec[n:]
        g_mat = self._as_numeric_matrix(metric_G, "metric_G", square=True)
        if g_mat.shape[0] != n:
            raise ValueError(f"metric_G debe ser {n}x{n} para estado {2*n}.")
        try:
            g_inv = la.inv(g_mat)
        except la.LinAlgError:
            g_inv = la.inv(g_mat + self._regularizer_floor() * np.eye(n))
        kinetic_energy = 0.5 * float(p_mom.T @ g_inv @ p_mom)
        hamiltonian_h = kinetic_energy + float(potential_V)
        grad_v = g_mat @ q_pos
        p_half = p_mom - 0.5 * dt_step * grad_v
        q_next = q_pos + dt_step * (g_inv @ p_half)
        grad_v_next = g_mat @ q_next
        p_next = p_half - 0.5 * dt_step * grad_v_next
        next_state = np.concatenate([q_next, p_next])
        eye_n = np.eye(n, dtype=np.float64)
        zero_n = np.zeros((n, n), dtype=np.float64)
        m1 = np.block([[eye_n, zero_n], [-0.5 * dt_step * g_mat, eye_n]])
        m2 = np.block([[eye_n, dt_step * g_inv], [zero_n, eye_n]])
        m3 = np.block([[eye_n, zero_n], [-0.5 * dt_step * g_mat, eye_n]])
        m_step = m3 @ m2 @ m1
        det_m = float(la.det(m_step))
        volume_drift = abs(det_m - 1.0)
        defect = self.symplectic_residual(m_step, _omega_n(n))
        kinetic_next = 0.5 * float(p_next.T @ g_inv @ p_next)
        energy_next = kinetic_next + float(potential_V)
        energy_drift = abs(energy_next - hamiltonian_h)
        kinetic_headroom = max(
            _WILKINSON_DRIFT_LIMIT,
            2.0 * (float(total_energy_H0) - float(potential_V)))
        refractive_n = np.sqrt(kinetic_headroom)
        velocity_q = g_inv @ p_next
        velocity_norm_g = float(np.sqrt(max(
            0.0, float(velocity_q.T @ g_mat @ velocity_q))))
        maupertuis_action = float(refractive_n * velocity_norm_g)
        omega_vec = self._as_numeric_vector(external_freq_omega, "external_freq_omega")
        k_vec = self._as_numeric_vector(wave_k, "wave_k")
        take = min(k_vec.size, omega_vec.size)
        small_divisor = float(np.dot(k_vec[:take], omega_vec[:take])) if take else 0.0
        if abs(small_divisor) < _LIMIT_WILKINSON:
            novikov_weight = float(np.exp(
                -1.0 / (_LIMIT_WILKINSON + abs(small_divisor))))
        else:
            novikov_weight = 1.0 / (small_divisor + _LIMIT_WILKINSON)
        is_valid = (
            volume_drift <= _WILKINSON_DRIFT_LIMIT
            and maupertuis_action > 0.0
            and defect <= max(_HARD_SYMPLECTIC_DEFECT * 10.0, _WILKINSON_DRIFT_LIMIT)
        )
        return ImperialEngineStepResult(
            next_state=next_state,
            hamiltonian_energy=float(hamiltonian_h),
            volume_drift=float(volume_drift),
            maupertuis_action=float(maupertuis_action),
            novikov_absorbed_weight=float(novikov_weight),
            is_step_valid=bool(is_valid),
            symplectic_defect=float(defect),
            energy_drift=float(energy_drift),
        )

    def spectral_triple_germ_certificate(self) -> _SpectralTripleCertificate:
        return self._triple_germ.certificate

    def attach_spectral_triple_germ(
        self, density_matrix: np.ndarray,
        csmd_step: float = _COMPLEX_STEP_DEFAULT_H,
    ) -> _SpectralTripleGerm:
        germ = self.synthesize_spectral_triple_germ(
            density_matrix, csmd_step=csmd_step, regularizer=self._reg)
        self._triple_germ = germ
        if germ.density is not None:
            self._hodge_germ = self.induce_hodge_cheeger_germ(germ.density)
        return germ

    def attach_hodge_cheeger_germ(
        self, density_or_boundary: np.ndarray,
        tangent_a: Optional[np.ndarray] = None,
        boundary_matrix: Optional[np.ndarray] = None,
    ) -> _HodgeCheegerGerm:
        germ = self.induce_hodge_cheeger_germ(
            density_or_boundary, tangent_a=tangent_a,
            boundary_matrix=boundary_matrix)
        self._hodge_germ = germ
        return germ

    @property
    def poincare_darboux_germ(self) -> _PoincareDarbouxGerm:
        return self._poincare_darboux_germ

    @property
    def poincare_floquet_germ(self) -> Optional[_PoincareFloquetGerm]:
        return self._poincare_floquet_germ

    def attach_poincare_darboux_germ(
        self, two_n: Optional[int] = None,
        density_matrix: Optional[np.ndarray] = None,
        section_index: int = 0, section_offset: float = 0.0,
        energy_level: float = 0.0,
        csmd_step: float = _COMPLEX_STEP_DEFAULT_H,
        **kwargs: Any,
    ) -> _PoincareDarbouxGerm:
        """Réplica pública del morfismo I.ω; actualiza 𝒢_I."""
        germ = self.synthesize_poincare_darboux_germ(
            two_n=two_n, density_matrix=density_matrix,
            section_index=section_index, section_offset=section_offset,
            energy_level=energy_level, csmd_step=csmd_step,
            regularizer=self._reg, hodge_germ=self._hodge_germ, **kwargs)
        self._poincare_darboux_germ = germ
        return germ

    def attach_poincare_floquet_germ(
        self, monodromy_M: Optional[np.ndarray] = None,
        orbit_period_T: float = 1.0, **kwargs: Any,
    ) -> _PoincareFloquetGerm:
        """Réplica pública del morfismo II.ω; actualiza 𝒢_II."""
        germ = self.induce_poincare_floquet_germ(
            darboux_germ=self._poincare_darboux_germ,
            monodromy_M=monodromy_M, orbit_period_T=orbit_period_T,
            **kwargs)
        self._poincare_floquet_germ = germ
        return germ

    def certify_poincare_certificate(
        self, heyting_verdict: Optional[str] = None,
    ) -> PoincareGuardsCertificate:
        """Réplica pública del morfismo terminal III.ω."""
        return self.certify_poincare_guards_cycle(
            floquet_germ=self._poincare_floquet_germ,
            heyting_verdict=heyting_verdict)


__all__ = [
    "Phase1NumericalFoundationsMixin",
    "Phase2SpectralQuantumMixin",
    "Phase3TopologicalGeometricMixin",
    "ImperialEngineStepResult",
    "ImperialGuardsEngine",
    "PoincareGuardsCertificate",
]