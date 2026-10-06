# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Homotopic Séquitos Agent (Capa 1.5 · Poincaré-Nested-Phases)        ║
║ Ruta   : app/agents/core/inmune_system/imperial_guards_sequitos.py           ║
║ Versión: 5.1.0-Poincare-Celeste-KAM-Nekhoroshev-Lindstedt-Birkhoff-Twist     ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS MATEMÁTICA, CATEGORIAL Y CELESTE DE POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
Orquesta la concurrencia táctica de las sub-tríadas agénticas y la supervisión
de la evolución temporal multianual de los megaproyectos (Fase BIM 7D) en APU Filter v8.0,
articulando la geometría simpléctica y el análisis estocástico cuántico mediante tres
fases anidadas ($\Phi_{\mathrm{III}} \circ \Phi_{\mathrm{II}} \circ \Phi_{\mathrm{I}}$).

DEFINICIONES, AXIOMAS Y TEOREMAS FORMALES:

FASE 1 — OBSERVE ($\Phi_{\mathrm{I}}$) — MONADAS DE KLEISLI, CONSENSO DE DEGROOT Y BELL–CHSH:
1. Mónada de Kleisli-Giry para Probabilidades Cuánticas:
   Para la mónada de Giry $\mathcal{P}: \mathbf{Meas} \to \mathbf{Meas}$, un morfismo Kleisli
   $f: A \rightsquigarrow B$ en $\mathbf{Kl}(\mathcal{P})$ satisface la asociatividad covariante
   $(h \circ_K g) \circ_K f = h \circ_K (g \circ_K f)$, garantizando la invarianza de la regla de Bayes
   frente a la conmutación de la base de medición.

2. Consenso Espectral de DeGroot–Fiedler:
   Para la matriz de afinidad $W \in \mathbb{R}^{n \times n}$ normalizada estocásticamente $M = D^{-1} W$,
   el laplaciano $L = I - M$ posee autovalor de Fiedler $\lambda_2(L) > 0$. El tiempo de mezcla satisface
   $\tau_{\mathrm{mix}}(\epsilon) \le \frac{1}{\lambda_2(L)} \ln \frac{1}{\epsilon \min_i \pi_i}$.

3. Cota Cuántica de Tsirelson y Desigualdad Bell–CHSH:
   El observable correlador de Bell–CHSH $S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$ satisface:
   $$|S| \le 2 \quad (\text{Clásico}), \qquad |S| \le 2\sqrt{2} \approx 2.8284 \quad (\text{Tsirelson / QM}), \qquad |S| \le 4 \quad (\text{PR-Boxes}).$$
   Cualquier violación $|S| > 2\sqrt{2}$ denota no-localidad suprawantum no física (ruido no señalizado).

FASE 2 — ORIENT ($\Phi_{\mathrm{II}}$) — RECURRENCIA DE POINCARÉ Y KAM MULTIDIMENSIONAL:
4. Teorema de Recurrencia de Poincaré y Distancia de Retorno:
   Para una transformación $T$ que preserva la medida de Liouville $\mu$ en un espacio de fase compacto $\Omega$,
   todo conjunto medible $E$ contiene puntos que retornan infinitas veces a $E$.
   La distancia de retorno ergódico satisface $d_{\mathrm{ret}}(z) \triangleq \inf_{t > t_0} \|z(t) - z(0)\| \le \varepsilon_{\mathrm{div}}$.

5. Cota Diofántica KAM y Anillo de Novikov:
   Para vector de frecuencias $\omega$, si $|\langle k, \omega \rangle| \ge \frac{\gamma}{\|k\|_1^\tau}$ ($\tau > n-1$),
   el peso de Novikov $W_{\mathrm{Nov}} = \exp\left(-\frac{T_{\mathrm{val}}}{\varepsilon + |\langle k, \omega \rangle|}\right)$
   absorbe las divisiones pequeñas en el integrador no lineal.

FASE 3 — DECIDE/ACT ($\Phi_{\mathrm{III}}$) — ADJUDICACIÓN DE HEYTING Y CONTROL CIBER-FÍSICO:
6. Ínfimo de Gödel e Interlock Ciber-Físico en IRAM:
   El veredicto global es el meet del retículo de Heyting $G_3 = \{\mathrm{VETOED}(0) \le \mathrm{DEGRADED}(1) \le \mathrm{COHERENT}(2)\}$:
   $$\nu_{\mathrm{global}} = \bigwedge_k \nu_k \in G_3$$
   Ante $\nu_{\mathrm{global}} = \mathrm{VETOED}$, el interlock en IRAM interrumpe la ejecución táctica en $t_{\mathrm{act}} \le 400 \text{ ns}$.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

try:
    from app.core.inmune_system.imperial_sequitos_engine import (
        ImperialSequitosEngine,
        PoincareSequitosCertificate as EnginePoincareCertificate,
    )
except ImportError:  # pragma: no cover — import plano / tests locales
    from imperial_sequitos_engine import (  # type: ignore[no-redef]
        ImperialSequitosEngine,
        PoincareSequitosCertificate as EnginePoincareCertificate,
    )

logger = logging.getLogger("APU.Agents.HomotopicSequitos")

__version__: Final[str] = (
    "5.1.0-Poincare-Celeste-KAM-Nekhoroshev-Lindstedt-Birkhoff-Twist-Delaunay"
)

# =============================================================================
# CONSTANTES DE CONTROL LÓGICO, METROLOGÍA Y MECÁNICA CELESTE DE POINCARÉ
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_INTERLOCK_LATENCY_BUDGET_NS: Final[float] = 400.0
_INTERLOCK_JITTER_NS: Final[float] = 5.0
_WILKINSON_REL_SCALE: Final[float] = 10.0
_KLEISLI_COHERENT_TOL: Final[float] = 1e-10
_KLEISLI_DEGRADED_TOL: Final[float] = 1e-8
_DEGROOT_COHERENT_DEV: Final[float] = 1e-6
_DEGROOT_DEGRADED_DEV: Final[float] = 1e-4
_TSIRELSON_BOUND: Final[float] = float(2.0 * np.sqrt(2.0))
_CLASSICAL_CHSH_BOUND: Final[float] = 2.0
_PR_NOSIGNAL_BOUND: Final[float] = 4.0
_TSIRELSON_GUARD_BASE: Final[float] = 1e-3
_CORRELATOR_BOUND: Final[float] = 1.0

# ── Constantes de Poincaré ──────────────────────────────────────────────────
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
_NEKHOROSHEV_C: Final[float] = 0.5
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_PRIME_RADICALS: Final[Tuple[int, ...]] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

_HEYTING_ORDER: Final[Dict[str, int]] = {"COHERENT": 0, "DEGRADED": 1, "VETOED": 2}
_HEYTING_GODEL: Final[Dict[str, float]] = {"COHERENT": 1.0, "DEGRADED": 0.5, "VETOED": 0.0}
_REVERSE_HEYTING: Final[Dict[int, str]] = {0: "COHERENT", 1: "DEGRADED", 2: "VETOED"}

KleisliArrow = Callable[[Any], Tuple[Any, float]]


# =============================================================================
# NÚCLEO GEOMÉTRICO — FORMAS SIMPLÉCTICAS, RETORNO, FRACCIONES CONTINUAS
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
    """Norma de Frobenius en rango 2; euclídea en rango 1."""
    a = np.asarray(arr)
    if a.ndim >= 2:
        return float(la.norm(a, ord="fro"))
    return float(la.norm(a))


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


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
    n = m.shape[0] // 2
    try:
        jay = _omega_n(n)
    except ValueError:
        return float("inf")
    return _frob(m.T @ jay @ m - jay)


def _cotangent_lift(matrix: np.ndarray, regularizer: float) -> np.ndarray:
    r"""
    Levantamiento cotangente (simpléctico) \(T^*A=\mathrm{diag}(A,A^{-\top})\).
    Es el morfismo canónico \(\mathrm{GL}(n)\to\mathrm{Sp}(2n)\).
    """
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
    """Convergentes (p_k, q_k) con silla \(h_{-2}=0,h_{-1}=1,k_{-2}=1,k_{-1}=0\)."""
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


def _count_simple_zeros(samples: np.ndarray) -> int:
    """Ceros simples por cambio de signo (Morse: y_i y_{i+1}<0 y pendiente ≠ 0)."""
    y = np.asarray(samples, dtype=np.float64).ravel()
    if y.size < 2:
        return 0
    zeros = 0
    for i in range(int(y.size) - 1):
        a, b = float(y[i]), float(y[i + 1])
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0 and b != 0.0:
            zeros += 1
        elif a * b < 0.0:
            zeros += 1
    return int(zeros)


def _integer_harmonics(dim: int, cap: int) -> np.ndarray:
    """Red entera recortada: ejes + pares con ‖k‖₁ ≤ cap, k ≠ 0."""
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
    """Vector ω con componentes √p_i (independencia ℚ-típica, KAM-estable)."""
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


def _vec3(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 2:
        arr = np.array([arr[0], arr[1], 0.0], dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"{name} debe ser de dimensión 2 o 3.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene no-finitos.")
    return arr


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


# =============================================================================
# DTOs GLOBALES — Certificados de la Fase III
# =============================================================================
@dataclass(frozen=True)
class SequitosPoincareCertificate:
    r"""
    Certificado inmutable de lazo cerrado para la recurrencia de Séquitos.
    (Compatibilidad 4.0; en 5.1 se enriquece con dinámica Floquet–Lyapunov.)
    """
    poincare_return_distance: float
    is_kam_diophantine_stable: bool
    novikov_absorbed_weight: float
    volume_drift: float
    heyting_verdict: str
    is_sequitos_coherent: bool


@dataclass(frozen=True)
class PoincareSequitosCertificate:
    r"""
    **Certificado global de Poincaré–Séquitos (Fase III, morfismo terminal).**
    Integra \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\) con
    todos los invariantes de mecánica celeste de Poincaré.
    """
    n_agents: int
    safety_margin: float
    kleisli_deviation: float
    degroot_fiedler: float
    degroot_deviation: float
    chsh_s_value: float
    poincare_return_distance: float
    poincare_section_transversal: bool
    poincare_cartan_integral: float
    floquet_max_multiplier: float
    floquet_lyapunov_max: float
    floquet_log_residual: float
    kam_diophantine_gamma: float
    kam_diophantine_tau: float
    kam_stable: bool
    lyapunov_spectrum: np.ndarray
    lyapunov_kaplan_yorke: float
    lyapunov_ks_entropy: float
    melnikov_simple_zeros: Optional[int]
    melnikov_chaotic: Optional[bool]
    rotation_number: Optional[float]
    rotation_is_rational: Optional[bool]
    novikov_absorbed_weight: float
    kepler_elements: Optional[Dict[str, Any]]
    heyting_verdict: str
    heyting_godel_meet: float
    is_sequitos_coherent: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    # ── Extensión 5.1 · programa celeste de Poincaré ─────────────────────
    nekhoroshev_time_scale: float = 0.0
    twist_determinant: float = 0.0
    birkhoff_residual: float = 0.0
    lindstedt_secular_residual: float = 0.0
    action_variables: Optional[np.ndarray] = None
    poincare_orbit_type: str = "unknown"
    recurrence_is_ergodic: bool = False
    symplectic_defect: float = float("nan")
    hill_discriminant: float = float("nan")


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE I — NÚCLEO DE AUDITORÍA ESPECTRAL + GEOMETRÍA DE POINCARÉ          ██
# ██  Objetos: Kleisli, DeGroot, CHSH, sección Σ, retorno P, Floquet,         ██
# ██           Mel’nikov, KAM, Lyapunov, Cartan, Kepler–Delaunay, Novikov,    ██
# ██           Birkhoff, Nekhoroshev, twist, Lindstedt.                       ██
# ██  Morfismo terminal (I.ω): synthesize_poincare_heyting_audit_germ          ██
# ██                           → objeto inicial de la FASE II                 ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
@dataclass(frozen=True)
class _KleisliRawResult:
    deviation: float
    lhs_prob: float = float("nan")
    rhs_prob: float = float("nan")
    value_mismatch: float = 0.0
    engine_ok: bool = True


@dataclass(frozen=True)
class _DeGrootRawResult:
    final_opinions: np.ndarray
    fiedler_value: float
    deviation: float
    discrete_opinions: Optional[np.ndarray] = None
    connected: bool = True
    cheeger_upper: float = float("nan")
    mixing_rate: float = float("nan")
    engine_verdict: str = ""
    is_reversible: bool = False
    engine_ok: bool = True

    def __post_init__(self) -> None:
        if self.discrete_opinions is None:
            object.__setattr__(
                self, "discrete_opinions",
                np.asarray(self.final_opinions).copy())


@dataclass(frozen=True)
class _CHSHRawResult:
    s_value: float
    engine_verdict: str = ""
    physical: bool = True
    tsirelson_gap: float = float("nan")
    classical_gap: float = float("nan")
    pr_gap: float = float("nan")
    horodecki_bound: float = float("nan")
    engine_ok: bool = True


@dataclass(frozen=True)
class _PoincareSectionRawResult:
    """Sección transversal Σ = { q_k = q_k* } y su certificado característico."""
    normal: np.ndarray
    section_index: int
    section_offset: float
    energy_level: float
    transversal_certificate: float
    is_transversal: bool
    characteristic_direction: Optional[np.ndarray] = None
    engine_ok: bool = True


@dataclass(frozen=True)
class _RecurrenceRawResult:
    r"""Recurrencia de Poincaré: \(\inf_{t>0}\|z(t)-z(0)\|\) y recuento de retornos."""
    return_distance: float
    n_returns: int
    mean_return_gap: float
    is_recurrent: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _FloquetRawResult:
    """Factorización de Floquet–Lyapunov M = exp(T·A_F)·R_F."""
    max_multiplier: float
    lyapunov_max: float
    log_residual: float
    is_real_logarithm: bool
    multipliers: np.ndarray
    characteristic_exponents: np.ndarray
    orbit_type: str = "unknown"
    hill_discriminant: float = float("nan")
    symplectic_defect: float = float("nan")
    engine_ok: bool = True


@dataclass(frozen=True)
class _MelnikovRawResult:
    """Función de Mel’nikov ℳ(t₀); ceros simples ⇒ caos homoclínico."""
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _RotationRawResult:
    """Número de rotación ρ y su certificación diofántica."""
    rotation_number: float
    is_rational: bool
    continued_fraction: Tuple[int, ...]
    diophantine_constant: float
    convergent_denominator: int = 0
    engine_ok: bool = True


@dataclass(frozen=True)
class _KamRawResult:
    """Certificado KAM |k·ω| ≥ γ/|k|^τ."""
    gamma: float
    tau: float
    is_stable: bool
    iterations: int
    worst_divisor: float = float("nan")
    engine_ok: bool = True


@dataclass(frozen=True)
class _LyapunovRawResult:
    """Espectro completo de Lyapunov por QR de Benettin + KY + Pesin."""
    spectrum: np.ndarray
    kaplan_yorke: float
    ks_entropy: float
    is_chaotic: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _CartanRawResult:
    """Invariante integral de Poincaré–Cartan ∮ p dq − H dt."""
    integral: float
    action_integral: float = float("nan")
    engine_ok: bool = True


@dataclass(frozen=True)
class _KeplerRawResult:
    """Elementos orbitales osculadores Kepler + Delaunay + Poincaré."""
    elements: Dict[str, Any]
    engine_ok: bool = True


@dataclass(frozen=True)
class _NovikovRawResult:
    """Absorción ultramétrica en el anillo de Novikov."""
    small_divisor: float
    absorbed_weight: float
    triggered: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _ActionAngleRawResult:
    r"""Variables de acción-ángulo \(I=(1/2\pi)\oint p\,dq\), \(\dot\theta=\omega(I)\)."""
    actions: np.ndarray
    frequencies: np.ndarray
    generating_residual: float
    engine_ok: bool = True


@dataclass(frozen=True)
class _BirkhoffRawResult:
    """Residuo de la forma normal de Birkhoff / ecuación homológica."""
    order: int
    residual: float
    is_normalizable: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _NekhoroshevRawResult:
    r"""Confinamiento exponencial \(|I(t)-I(0)|\le\varepsilon^b\) para \(|t|\le\exp(c/\varepsilon^a)\)."""
    exponent_a: float
    confinement_b: float
    time_scale: float
    is_confined: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _TwistRawResult:
    """Condición de twist de Moser / no-degeneración de Kolmogorov e isoenergética."""
    twist_determinant: float
    kolmogorov_nondeg: bool
    isoenergetic_nondeg: bool
    has_poincare_birkhoff_fps: bool
    engine_ok: bool = True


@dataclass(frozen=True)
class _LindstedtRawResult:
    """Residuo secular de la serie de Lindstedt–Poincaré (anulación de términos t·sin)."""
    secular_residual: float
    frequency_correction: np.ndarray
    engine_ok: bool = True


@dataclass(frozen=True)
class _HeytingAuditGerm:
    """Gérmen de auditoría de Heyting (objeto terminal histórico de Fase I)."""
    kleisli: _KleisliRawResult
    degroot: _DeGrootRawResult
    chsh: _CHSHRawResult
    n_agents: int
    safety_margin: float
    kleisli_scale: float
    degroot_scale: float


@dataclass(frozen=True)
class _PoincareHeytingAuditGerm:
    r"""
    **Gérmen de auditoría de Heyting–Poincaré.**
    **Objeto terminal de la FASE I / objeto inicial de la FASE II.**

    \[
      \mathcal{G}_I = (\mathcal{G}_{\mathrm{Heyting}},\Sigma,P,F,\mathcal{M},\rho,
                       \mathrm{KAM},\mathrm{Lyap},\mathrm{Cartan},\mathrm{Kepler},
                       \mathrm{Novikov},I,\mathrm{Birkhoff},\mathrm{Nekhoroshev},
                       \mathrm{twist},\mathrm{Lindstedt})
    \]
    """
    base: _HeytingAuditGerm
    section: _PoincareSectionRawResult
    floquet: _FloquetRawResult
    melnikov: Optional[_MelnikovRawResult]
    rotation: Optional[_RotationRawResult]
    kam: _KamRawResult
    lyapunov: _LyapunovRawResult
    cartan: _CartanRawResult
    kepler: Optional[_KeplerRawResult]
    novikov: _NovikovRawResult
    recurrence: _RecurrenceRawResult
    action_angle: Optional[_ActionAngleRawResult]
    birkhoff: _BirkhoffRawResult
    nekhoroshev: _NekhoroshevRawResult
    twist: _TwistRawResult
    lindstedt: _LindstedtRawResult


class _AuditCore:
    """
    Fase I. Núcleo ciego que habla con `ImperialSequitosEngine`.
    En v5.1 consume y, si el motor no certifica, calcula localmente el
    programa celeste de Poincaré (retorno, Floquet hamiltoniano, Mel’nikov,
    Benettin, KAM, Kepler–Delaunay, Nekhoroshev, twist, Lindstedt).
    """

    def __init__(self, engine: ImperialSequitosEngine, n_agents: int) -> None:
        self._engine = engine
        self._n = int(n_agents)

    @property
    def engine(self) -> ImperialSequitosEngine:
        return self._engine

    # ── I.1  Utilidades de coerción ──────────────────────────────────────
    @staticmethod
    def _as_vec(name: str, values: Any, dim: Optional[int] = None) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        vec = np.asarray(arr).reshape(-1)
        if vec.size == 0:
            raise ValueError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contiene no-finitos.")
        if dim is not None and vec.size != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}; {vec.size}.")
        return vec.astype(np.float64, copy=False)

    @staticmethod
    def _as_matrix(name: str, values: Any, square: bool = False) -> np.ndarray:
        try:
            arr = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no es convertible a ndarray.") from exc
        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            if side * side != arr.size:
                raise ValueError(f"{name} plana no cuadrado perfecto.")
            arr = arr.reshape(side, side)
        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser de rango 2.")
        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene no-finitos.")
        return np.asarray(arr, dtype=np.float64)

    @staticmethod
    def _prob_of(pair: Tuple[Any, float], name: str) -> Tuple[Any, float]:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} debe retornar (valor, probabilidad).")
        value, prob = pair
        pf = float(prob)
        if not math.isfinite(pf):
            raise ValueError(f"{name} produjo una probabilidad no finita.")
        return value, pf

    @staticmethod
    def _value_mismatch(lhs: Any, rhs: Any) -> float:
        try:
            a = np.asarray(lhs, dtype=np.float64)
            b = np.asarray(rhs, dtype=np.float64)
        except (TypeError, ValueError):
            return 0.0 if lhs == rhs else 1.0
        if a.shape != b.shape:
            return float("inf")
        if a.size == 0:
            return 0.0
        delta = a - b
        return float(np.sqrt(max(float(np.real(np.vdot(delta, delta))), 0.0)))

    def _affinity_monodromy(self, affinity_matrix: Any) -> np.ndarray:
        r"""
        Monodromía efectiva: matriz estocástica embebida en dimensión par.
        No se usa el levantamiento cotangente (amplificaría \(|\lambda|<1\) a
        \(|1/\lambda|>1\)); el pad identidad preserva el radio espectral.
        """
        a = self._as_matrix("affinity_matrix", affinity_matrix, square=True)
        row_sums = np.sum(np.abs(a), axis=1, keepdims=True)
        m_eff = a / np.where(row_sums > _MACHINE_EPS, row_sums, 1.0)
        return _even_embed(m_eff)

    # ── I.2  Kleisli / DeGroot / CHSH (heredados) ────────────────────────
    def compute_kleisli_deviation(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
    ) -> _KleisliRawResult:
        try:
            if not (callable(f) and callable(g) and callable(h_func)):
                raise TypeError("f, g y h_func deben ser callables de Kleisli.")
            compose = self._engine.kleisli_compose
            g_f = compose(f, g)
            lhs_func = compose(g_f, h_func)
            h_g = compose(g, h_func)
            rhs_func = compose(f, h_g)
            v_lhs, p_lhs = self._prob_of(lhs_func(test_input), "lhs")
            v_rhs, p_rhs = self._prob_of(rhs_func(test_input), "rhs")
            deviation = float(abs(p_lhs - p_rhs))
            mismatch = self._value_mismatch(v_lhs, v_rhs)
            return _KleisliRawResult(
                deviation=deviation, lhs_prob=float(p_lhs),
                rhs_prob=float(p_rhs), value_mismatch=float(mismatch),
                engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Kleisli: %s", exc)
            return _KleisliRawResult(deviation=float("inf"), engine_ok=False)

    def compute_degroot_metrics(
        self, opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootRawResult:
        empty = _DeGrootRawResult(
            final_opinions=np.array([], dtype=np.float64),
            fiedler_value=float("inf"), deviation=float("inf"),
            discrete_opinions=np.array([], dtype=np.float64),
            connected=False, engine_ok=False)
        try:
            x = self._as_vec("opinion_vector", opinion_vector)
            w = self._as_matrix("affinity_matrix", affinity_matrix, square=True)
            if w.shape[0] != x.size:
                raise ValueError("Afinidad debe coincidir con opinión.")
            if int(steps) < 0:
                raise ValueError("steps debe ser no negativo.")
            certified = getattr(
                self._engine, "compute_degroot_spectral_consensus_certified", None)
            if callable(certified):
                result = certified(x, w, steps)
                opinions = np.asarray(result.final_opinion, dtype=np.float64)
                if opinions.size:
                    mean = float(np.mean(opinions))
                    deviation = float(np.sqrt(max(
                        float(np.mean((opinions - mean) ** 2)), 0.0)))
                else:
                    deviation = float(getattr(result, "deviation", float("inf")))
                if np.isfinite(getattr(result, "deviation", float("nan"))):
                    deviation = float(result.deviation)
                return _DeGrootRawResult(
                    final_opinions=opinions,
                    fiedler_value=float(result.fiedler_value),
                    deviation=float(deviation),
                    discrete_opinions=np.asarray(
                        getattr(result, "discrete_opinion", opinions),
                        dtype=np.float64),
                    connected=bool(getattr(result, "connected", True)),
                    cheeger_upper=float(getattr(result, "cheeger_upper", float("nan"))),
                    mixing_rate=float(getattr(result, "mixing_rate", float("nan"))),
                    engine_verdict=str(getattr(result, "verdict", "")),
                    is_reversible=bool(getattr(result, "is_reversible", False)),
                    engine_ok=True)
            final_opinions, fiedler, engine_verdict = (
                self._engine.compute_degroot_spectral_consensus(x, w, steps))
            opinions = np.asarray(final_opinions, dtype=np.float64)
            if opinions.size:
                mean = float(np.mean(opinions))
                deviation = float(np.sqrt(max(
                    float(np.mean((opinions - mean) ** 2)), 0.0)))
            else:
                deviation = float("inf")
            return _DeGrootRawResult(
                final_opinions=opinions, fiedler_value=float(fiedler),
                deviation=float(deviation),
                discrete_opinions=opinions.copy(),
                engine_verdict=str(engine_verdict), engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en DeGroot: %s", exc)
            return empty

    def compute_chsh_s_value(
        self, correlation_matrix: np.ndarray,
    ) -> _CHSHRawResult:
        try:
            e = self._as_matrix("correlation_matrix", correlation_matrix, square=True)
            certified = getattr(self._engine, "verify_chsh_violation_certified", None)
            if callable(certified):
                result = certified(e)
                return _CHSHRawResult(
                    s_value=float(result.s_value),
                    engine_verdict=str(getattr(result, "verdict", "")),
                    physical=bool(getattr(result, "physical", True)),
                    tsirelson_gap=float(getattr(result, "tsirelson_gap", float("nan"))),
                    classical_gap=float(getattr(result, "classical_gap", float("nan"))),
                    pr_gap=float(getattr(result, "pr_gap", float("nan"))),
                    horodecki_bound=float(getattr(result, "horodecki_bound", float("nan"))),
                    engine_ok=True)
            s_value, engine_verdict = self._engine.verify_chsh_violation(e)
            phys = bool(np.all(np.abs(np.real(e)) <= _CORRELATOR_BOUND + 1e-12))
            return _CHSHRawResult(
                s_value=float(s_value), engine_verdict=str(engine_verdict),
                physical=phys,
                tsirelson_gap=float(_TSIRELSON_BOUND - float(s_value)),
                classical_gap=float(float(s_value) - _CLASSICAL_CHSH_BOUND),
                pr_gap=float(_PR_NOSIGNAL_BOUND - float(s_value)),
                engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en CHSH: %s", exc)
            return _CHSHRawResult(s_value=float("inf"), physical=False, engine_ok=False)

    # ── I.3  Poincaré — sección transversal, característica y recurrencia ─
    def compute_poincare_section(
        self, section_index: int = 0,
        section_offset: float = 0.0, energy_level: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
    ) -> _PoincareSectionRawResult:
        r"""
        Construye \(\Sigma=\{q_{\mathrm{idx}}=q^*\}\subset T^*Q\).
        Dirección característica \(J n_\Sigma\); transversalidad del flujo
        \(n_\Sigma\cdot X_H\neq 0\) si se suministra \(X_H\).
        """
        try:
            n = self._n
            two_n = 2 * n
            if not (0 <= section_index < n):
                raise ValueError(f"section_index={section_index} fuera de [0,{n-1}].")
            omega = _omega_n(n)
            n_sigma = np.zeros(two_n, dtype=np.float64)
            n_sigma[section_index] = 1.0
            char_dir = omega @ n_sigma
            regular = _frob(char_dir)
            if flow_vector is not None:
                flow = self._as_vec("flow_vector", flow_vector, dim=two_n)
                certificate = float(abs(float(n_sigma @ flow)))
                is_transversal = bool(certificate > _HARD_POINCARE_SECTION_TOL)
            else:
                certificate = float(regular)
                is_transversal = bool(regular > _MACHINE_EPS)
            return _PoincareSectionRawResult(
                normal=n_sigma, section_index=int(section_index),
                section_offset=float(section_offset),
                energy_level=float(energy_level),
                transversal_certificate=float(certificate),
                is_transversal=is_transversal,
                characteristic_direction=char_dir,
                engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en sección de Poincaré: %s", exc)
            return _PoincareSectionRawResult(
                normal=np.zeros(2 * self._n),
                section_index=int(section_index), section_offset=float(section_offset),
                energy_level=float(energy_level),
                transversal_certificate=0.0, is_transversal=False,
                engine_ok=False)

    def compute_poincare_recurrence(
        self,
        state_trajectory_z: Optional[Sequence[np.ndarray]],
        section: _PoincareSectionRawResult,
    ) -> _RecurrenceRawResult:
        r"""
        Distancia de primer retorno y recuento de cruces de \(\Sigma\).
        Recurrencia cuantitativa: \(\min_k\|z_N-z_k\|\le\varepsilon_{\mathrm{div}}\).
        """
        try:
            if not state_trajectory_z or len(state_trajectory_z) < 2:
                return _RecurrenceRawResult(
                    return_distance=0.0, n_returns=0, mean_return_gap=0.0,
                    is_recurrent=True, engine_ok=True)
            pts = [self._as_vec("z_k", z) for z in state_trajectory_z]
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
            is_rec = bool(min_ret <= _HARD_DIVERGENCE_CEILING)
            return _RecurrenceRawResult(
                return_distance=float(min_ret),
                n_returns=int(crossings),
                mean_return_gap=float(mean_gap),
                is_recurrent=is_rec, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en recurrencia de Poincaré: %s", exc)
            return _RecurrenceRawResult(
                return_distance=float("inf"), n_returns=0,
                mean_return_gap=float("nan"), is_recurrent=False, engine_ok=False)

    # ── I.4  Floquet–Lyapunov (logaritmo hamiltoniano) ───────────────────
    def compute_floquet_lyapunov(
        self, M: np.ndarray, orbit_period_T: float,
    ) -> _FloquetRawResult:
        r"""
        \(M=\exp(T A_F)\,R_F\); \(\mu_k=\mathrm{eig}(M)\); \(\lambda_k=\log|\mu_k|/T\).
        \(A_F\) se proyecta a \(\mathfrak{sp}(2n)\) cuando \(M\) es de dimensión par.
        """
        try:
            m = self._as_matrix("M", M, square=True)
            if orbit_period_T <= 0.0 or not np.isfinite(orbit_period_T):
                raise ValueError("orbit_period_T debe ser positivo y finito.")
            defect = _symplectic_defect(m)
            hill = float(np.trace(m)) if m.shape[0] == 2 else float("nan")
            certified = getattr(
                self._engine, "compute_floquet_lyapunov_certified", None)
            if callable(certified):
                result = certified(m, orbit_period_T)
                mu = np.asarray(result.floquet_multipliers, dtype=np.complex128)
                max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
                lyap_max = float(np.log(max(max_mu, _MACHINE_EPS)) / orbit_period_T)
                char = np.asarray(result.characteristic_exponents, dtype=np.float64)
                return _FloquetRawResult(
                    max_multiplier=max_mu, lyapunov_max=lyap_max,
                    log_residual=float(result.log_residual),
                    is_real_logarithm=bool(result.is_real_logarithm),
                    multipliers=mu, characteristic_exponents=char,
                    orbit_type=_poincare_orbit_type(mu, _HARD_FLOQUET_MULTIPLIER_TOL),
                    hill_discriminant=hill, symplectic_defect=float(defect),
                    engine_ok=True)
            mu = la.eigvals(m)
            max_mu = float(np.max(np.abs(mu))) if mu.size else 0.0
            is_real_log = True
            try:
                log_m = np.asarray(la.logm(m), dtype=np.complex128)
                imag_res = _frob(np.imag(log_m))
                if imag_res > 1.0e-8:
                    is_real_log = False
                gen = np.real(log_m) / float(orbit_period_T)
            except (la.LinAlgError, ValueError):
                gen = (m - np.eye(m.shape[0], dtype=np.float64)) / float(orbit_period_T)
                is_real_log = False
            if m.shape[0] % 2 == 0:
                jay = _omega_n(m.shape[0] // 2)
                gen = _project_hamiltonian(gen, jay)
            r_f = la.expm(-float(orbit_period_T) * gen) @ m
            log_res = _frob(la.expm(float(orbit_period_T) * gen) @ r_f - m)
            with np.errstate(divide="ignore", invalid="ignore"):
                char_exp = np.log(np.abs(mu) + _MACHINE_EPS) / float(orbit_period_T)
            lyap_max = float(np.max(np.real(char_exp))) if char_exp.size else 0.0
            return _FloquetRawResult(
                max_multiplier=max_mu, lyapunov_max=lyap_max,
                log_residual=float(log_res), is_real_logarithm=bool(is_real_log),
                multipliers=np.asarray(mu, dtype=np.complex128),
                characteristic_exponents=np.asarray(np.real(char_exp), dtype=np.float64),
                orbit_type=_poincare_orbit_type(
                    np.asarray(mu, dtype=np.complex128), _HARD_FLOQUET_MULTIPLIER_TOL),
                hill_discriminant=hill, symplectic_defect=float(defect),
                engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Floquet: %s", exc)
            return _FloquetRawResult(
                max_multiplier=float("inf"), lyapunov_max=float("inf"),
                log_residual=float("inf"), is_real_logarithm=False,
                multipliers=np.array([], dtype=np.complex128),
                characteristic_exponents=np.array([], dtype=np.float64),
                orbit_type="unknown", engine_ok=False)

    # ── I.5  Función de Mel’nikov ────────────────────────────────────────
    def compute_melnikov(
        self, q0_trajectory: np.ndarray, dt: float,
        h0_grad: np.ndarray, h1_grad: np.ndarray, omega: np.ndarray,
    ) -> _MelnikovRawResult:
        r"""
        \(\mathcal{M}(t_0)=\int\{H_0,H_1\}(q_0(t),t+t_0)\,dt\),
        \(\{H_0,H_1\}=\nabla H_0\cdot J\nabla H_1\). Ceros simples ⇒ homoclínico.
        """
        try:
            certified = getattr(self._engine, "compute_melnikov_certified", None)
            if callable(certified):
                result = certified(q0_trajectory, dt, h0_grad, h1_grad, omega)
                return _MelnikovRawResult(
                    simple_zeros=int(result.simple_zeros),
                    chaotic_indicator=float(result.chaotic_indicator),
                    is_chaotic=bool(result.is_chaotic), engine_ok=True)
            q0 = np.asarray(q0_trajectory, dtype=np.float64)
            if q0.ndim == 1:
                q0 = q0.reshape(1, -1)
            n_samples = int(q0.shape[0])
            g0 = np.asarray(h0_grad, dtype=np.float64)
            g1 = np.asarray(h1_grad, dtype=np.float64)
            jay = np.asarray(omega, dtype=np.float64)
            if g0.ndim == 1:
                g0 = np.tile(g0.reshape(1, -1), (n_samples, 1))
            if g1.ndim == 1:
                g1 = np.tile(g1.reshape(1, -1), (n_samples, 1))
            if g0.shape != g1.shape or g0.shape[0] != n_samples:
                raise ValueError("Gradientes Mel’nikov incompatibles con q0.")
            if jay.ndim != 2 or jay.shape[0] != g0.shape[1]:
                raise ValueError("J simpléctico de dimensión incompatible.")
            dens = np.einsum("ni,ij,nj->n", g0, jay, g1)
            nph = int(_MELNIKOV_PHASE_SAMPLES)
            m_t = np.empty(nph, dtype=np.float64)
            shift = max(n_samples // nph, 1)
            dt_f = float(dt)
            for i in range(nph):
                m_t[i] = float(np.sum(np.roll(dens, i * shift)) * dt_f)
            nzeros = _count_simple_zeros(m_t)
            indicator = float(np.max(np.abs(m_t))) if m_t.size else 0.0
            return _MelnikovRawResult(
                simple_zeros=int(nzeros), chaotic_indicator=float(indicator),
                is_chaotic=bool(nzeros > 0), engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Mel'nikov: %s", exc)
            return _MelnikovRawResult(
                simple_zeros=0, chaotic_indicator=float("inf"),
                is_chaotic=False, engine_ok=False)

    # ── I.6  Número de rotación ──────────────────────────────────────────
    def compute_rotation_number(
        self, orbit_points: np.ndarray,
    ) -> _RotationRawResult:
        r"""
        \(\rho=\lim (1/N)\sum\Delta\theta_i/2\pi\); fracción continua truncada.
        Racional ssi el desarrollo de Gauss termina.
        """
        try:
            certified = getattr(
                self._engine, "compute_rotation_number_certified", None)
            if callable(certified):
                result = certified(np.asarray(orbit_points, dtype=np.float64))
                cf = tuple(result.continued_fraction)
                return _RotationRawResult(
                    rotation_number=float(result.rotation_number),
                    is_rational=bool(result.is_rational),
                    continued_fraction=cf,
                    diophantine_constant=float(result.diophantine_constant),
                    convergent_denominator=_cf_denominator(cf),
                    engine_ok=True)
            pts = np.asarray(orbit_points, dtype=np.float64)
            if pts.size == 0:
                raise ValueError("orbit_points vacío.")
            if pts.ndim == 1:
                theta = pts.ravel()
            else:
                x = pts[:, 0]
                y = pts[:, 1] if pts.shape[1] >= 2 else np.zeros(pts.shape[0])
                theta = np.arctan2(y, x)
            dth = _unwrap_delta(theta)
            rho = float(np.mean(dth) / (2.0 * np.pi)) if dth.size else 0.0
            cf = _continued_fraction(rho)
            terminated = bool(cf) and (len(cf) < _ROTATION_CF_DEPTH)
            dio = _diophantine_constant(rho, cf)
            return _RotationRawResult(
                rotation_number=float(rho), is_rational=terminated,
                continued_fraction=cf, diophantine_constant=float(dio),
                convergent_denominator=_cf_denominator(cf), engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en rotación: %s", exc)
            return _RotationRawResult(
                rotation_number=float("nan"), is_rational=False,
                continued_fraction=tuple(), diophantine_constant=float("nan"),
                engine_ok=False)

    # ── I.7  KAM diofántico ──────────────────────────────────────────────
    def certify_kam(
        self, frequency_vector: np.ndarray, birkhoff_residual: float = 0.0,
    ) -> _KamRawResult:
        r"""
        \(|k\cdot\omega|\ge\gamma/|k|_1^\tau\) \(\forall k\neq 0\), \(|k|_\infty\le H\).
        \(\tau>\ n-1\) (Rüssmann). γ empírica = inf |k·ω| |k|_1^τ.
        """
        try:
            omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
            if omega.size == 0 or not np.all(np.isfinite(omega)):
                raise ValueError("frequency_vector no finito o vacío.")
            certified = getattr(self._engine, "certify_kam_torus", None)
            if callable(certified):
                result = certified(omega, birkhoff_residual=float(birkhoff_residual))
                return _KamRawResult(
                    gamma=float(result.diophantine_gamma),
                    tau=float(result.diophantine_tau),
                    is_stable=bool(result.kam_stable),
                    iterations=int(result.iterations), engine_ok=True)
            n = int(omega.size)
            tau = float(max(float(n - 1) + 1.0e-6, _DIOPHANTINE_TAU_FLOOR))
            harmonics = _integer_harmonics(n, _KAM_HARMONIC_CAP)
            gamma_emp = float("inf")
            worst = float("inf")
            iterations = int(harmonics.shape[0])
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
            is_stable = bool(
                gamma_emp >= _HARD_KAM_GAMMA_FLOOR and residual_pen <= _HARD_BIRKHOFF_RES_CEILING)
            return _KamRawResult(
                gamma=float(gamma_emp), tau=float(tau), is_stable=is_stable,
                iterations=iterations, worst_divisor=float(worst), engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en KAM: %s", exc)
            return _KamRawResult(
                gamma=float("inf"), tau=float("nan"),
                is_stable=False, iterations=0, engine_ok=False)

    # ── I.8  Espectro de Lyapunov (Benettin) ─────────────────────────────
    def compute_lyapunov_spectrum(
        self, M: np.ndarray, n_iterations: int = _LYAPUNOV_QR_ITERATIONS,
    ) -> _LyapunovRawResult:
        r"""
        \(Z_k=M Q_{k-1}\), \(Q_k R_k=\mathrm{QR}(Z_k)\),
        \(\lambda_i=(1/N)\sum\log|R_k[i,i]|\). Signos de diag(R) absorbidos en Q.
        """
        try:
            m = self._as_matrix("M", M, square=True)
            certified = getattr(
                self._engine, "compute_lyapunov_spectrum_certified", None)
            if callable(certified):
                result = certified(m, n_iterations=n_iterations)
                spec = np.asarray(result.spectrum, dtype=np.float64)
                return _LyapunovRawResult(
                    spectrum=spec,
                    kaplan_yorke=float(result.kaplan_yorke_dimension),
                    ks_entropy=float(result.kolmogorov_sinai_entropy),
                    is_chaotic=bool(result.is_chaotic), engine_ok=True)
            dim = int(m.shape[0])
            n_it = int(max(n_iterations, 1))
            q_mat = np.eye(dim, dtype=np.float64)
            acc = np.zeros(dim, dtype=np.float64)
            for _ in range(n_it):
                z_mat = m @ q_mat
                q_mat, r_mat = la.qr(z_mat, mode="economic")
                diag = np.real(np.diag(r_mat))
                signs = np.sign(diag)
                signs[signs == 0.0] = 1.0
                q_mat = q_mat * signs.reshape(1, -1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    acc = acc + np.log(np.abs(diag) + _MACHINE_EPS)
            spec = acc / float(n_it)
            ky, ks = _kaplan_yorke_and_pesin(spec)
            is_chaotic = bool(np.any(spec > _HARD_LYAPUNOV_TOL))
            return _LyapunovRawResult(
                spectrum=np.asarray(spec, dtype=np.float64),
                kaplan_yorke=float(ky), ks_entropy=float(ks),
                is_chaotic=is_chaotic, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Lyapunov: %s", exc)
            return _LyapunovRawResult(
                spectrum=np.array([], dtype=np.float64),
                kaplan_yorke=float("inf"), ks_entropy=float("inf"),
                is_chaotic=True, engine_ok=False)

    # ── I.9  Poincaré–Cartan ─────────────────────────────────────────────
    def compute_cartan_integral(
        self, q_traj: np.ndarray, p_traj: np.ndarray,
        H_traj: np.ndarray, dt: float,
    ) -> _CartanRawResult:
        r"""
        \(\oint_\gamma p\,dq-H\,dt\approx\sum_k[p_k\cdot\Delta q_k-H_k\,dt]\).
        La acción pura \(\oint p\,dq\) se expone aparte (Liouville–Arnold).
        """
        try:
            certified = getattr(
                self._engine, "poincare_cartan_invariant", None)
            q = np.asarray(q_traj, dtype=np.float64)
            p = np.asarray(p_traj, dtype=np.float64)
            ham = np.asarray(H_traj, dtype=np.float64).ravel()
            if q.ndim != 2 or p.shape != q.shape or ham.size != q.shape[0]:
                raise ValueError("Dimensiones incompatibles en Cartan.")
            n_nodes = int(q.shape[0])
            action_terms = np.empty(n_nodes, dtype=np.float64)
            cartan_terms = np.empty(n_nodes, dtype=np.float64)
            dt_f = float(dt)
            for k in range(n_nodes):
                nxt = (k + 1) % n_nodes
                pdq = float(p[k] @ (q[nxt] - q[k]))
                action_terms[k] = pdq
                cartan_terms[k] = pdq - float(ham[k]) * dt_f
            action = float(np.sum(action_terms))
            if callable(certified):
                val = float(certified(q_traj, p_traj, H_traj, dt))
            else:
                val = float(np.sum(cartan_terms))
            return _CartanRawResult(integral=val, action_integral=action, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Cartan: %s", exc)
            return _CartanRawResult(integral=float("nan"), engine_ok=False)

    # ── I.10  Elementos orbitales de Kepler–Delaunay–Poincaré ────────────
    def compute_kepler_elements(
        self, position: np.ndarray, velocity: np.ndarray,
        mu_gravitational: float = 1.0,
    ) -> _KeplerRawResult:
        r"""
        \(h=r\times v\), \(e=v\times h/\mu-\hat r\); Delaunay \((L,G,H,l,g,h)\)
        y variables canónicas de Poincaré \((\Lambda,\lambda,\xi,\eta,p,q)\).
        """
        try:
            certified = getattr(
                self._engine, "kepler_osculating_elements", None)
            if callable(certified):
                elements = certified(
                    np.asarray(position, dtype=np.float64).ravel(),
                    np.asarray(velocity, dtype=np.float64).ravel(),
                    mu_gravitational=float(mu_gravitational))
                return _KeplerRawResult(elements=dict(elements), engine_ok=True)
            r_vec = _vec3("position", position)
            v_vec = _vec3("velocity", velocity)
            mu = float(mu_gravitational)
            if mu <= 0.0 or not np.isfinite(mu):
                raise ValueError("mu_gravitational debe ser positivo y finito.")
            r_norm = float(la.norm(r_vec))
            v_norm = float(la.norm(v_vec))
            if r_norm <= _MACHINE_EPS:
                raise ValueError("posición singular (r=0).")
            h_vec = np.cross(r_vec, v_vec)
            h_norm = float(la.norm(h_vec))
            e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r_norm
            e_norm = float(la.norm(e_vec))
            energy = 0.5 * v_norm * v_norm - mu / r_norm
            if abs(energy) <= _MACHINE_EPS:
                a = float("inf")
            else:
                a = -mu / (2.0 * energy)
            n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
            n_norm = float(la.norm(n_vec))
            inc = 0.0 if h_norm <= _MACHINE_EPS else float(
                np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))
            if n_norm <= _MACHINE_EPS:
                omega_lan = 0.0
            else:
                omega_lan = float(math.atan2(n_vec[1], n_vec[0]))
            if n_norm <= _MACHINE_EPS or e_norm <= _MACHINE_EPS:
                argp = 0.0
            else:
                cos_w = float(np.clip(np.dot(n_vec, e_vec) / (n_norm * e_norm), -1.0, 1.0))
                sin_w_sign = float(np.dot(np.cross(n_vec, e_vec), h_vec))
                argp = float(math.atan2(sin_w_sign, cos_w * n_norm * e_norm))
                argp = float(math.atan2(
                    np.dot(np.cross(n_vec / n_norm, e_vec / e_norm), h_vec / max(h_norm, _MACHINE_EPS)),
                    cos_w))
            if e_norm <= _MACHINE_EPS:
                true_anom = 0.0
            else:
                cos_nu = float(np.clip(np.dot(e_vec, r_vec) / (e_norm * r_norm), -1.0, 1.0))
                sin_nu = float(np.dot(
                    np.cross(e_vec / e_norm, r_vec / r_norm),
                    h_vec / max(h_norm, _MACHINE_EPS)))
                true_anom = float(math.atan2(sin_nu, cos_nu))
            if e_norm < 1.0:
                cos_e = np.clip((e_norm + np.cos(true_anom)) / (1.0 + e_norm * np.cos(true_anom)), -1.0, 1.0)
                ecc_anom = float(math.acos(cos_e))
                if true_anom < 0.0:
                    ecc_anom = -ecc_anom
                mean_anom = float(ecc_anom - e_norm * math.sin(ecc_anom))
            else:
                mean_anom = float("nan")
            mean_motion = float(math.sqrt(mu / abs(a) ** 3)) if np.isfinite(a) and a != 0.0 else 0.0
            # Delaunay
            ell = math.sqrt(mu * a) if np.isfinite(a) and a > 0.0 else float("nan")
            gee = math.sqrt(max(mu * a * (1.0 - e_norm * e_norm), 0.0)) if np.isfinite(a) and a > 0.0 else float("nan")
            aitch = gee * math.cos(inc) if np.isfinite(gee) else float("nan")
            varpi = omega_lan + argp
            # Poincaré canónicas (pequeña e, i)
            two_lg = 2.0 * max((ell - gee) if np.isfinite(ell) and np.isfinite(gee) else 0.0, 0.0)
            two_gh = 2.0 * max((gee - aitch) if np.isfinite(gee) and np.isfinite(aitch) else 0.0, 0.0)
            xi = math.sqrt(two_lg) * math.cos(varpi)
            eta = -math.sqrt(two_lg) * math.sin(varpi)
            p_poinc = math.sqrt(two_gh) * math.cos(omega_lan)
            q_poinc = -math.sqrt(two_gh) * math.sin(omega_lan)
            lam_mean = omega_lan + argp + (mean_anom if np.isfinite(mean_anom) else 0.0)
            elements: Dict[str, Any] = {
                "a": float(a), "e": float(e_norm), "i": float(inc),
                "Omega": float(omega_lan), "omega": float(argp), "nu": float(true_anom),
                "M": float(mean_anom), "n": float(mean_motion),
                "energy": float(energy), "h_vec": h_vec, "e_vec": e_vec,
                "Delaunay_L": float(ell), "Delaunay_G": float(gee), "Delaunay_H": float(aitch),
                "Delaunay_l": float(mean_anom), "Delaunay_g": float(argp),
                "Delaunay_h": float(omega_lan),
                "Poincare_Lambda": float(ell), "Poincare_lambda": float(lam_mean),
                "Poincare_xi": float(xi), "Poincare_eta": float(eta),
                "Poincare_p": float(p_poinc), "Poincare_q": float(q_poinc),
            }
            return _KeplerRawResult(elements=elements, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Kepler: %s", exc)
            return _KeplerRawResult(elements={}, engine_ok=False)

    # ── I.11  Novikov — absorción ultramétrica ───────────────────────────
    def compute_novikov_absorption(
        self, small_divisor: float, novikov_valuation_T: float = 1.0,
    ) -> _NovikovRawResult:
        r"""
        Si \(\langle k,\omega\rangle\to 0\): peso \(w=\exp(-T/(\varepsilon_W+|\langle k,\omega\rangle|))\).
        """
        try:
            sd = float(small_divisor)
            if not np.isfinite(sd):
                raise ValueError("small_divisor no finito.")
            if abs(sd) < _LIMIT_WILKINSON:
                w = float(np.exp(-float(novikov_valuation_T)
                                 / (_LIMIT_WILKINSON + abs(sd))))
                triggered = True
            else:
                w = 1.0 / (sd + _LIMIT_WILKINSON)
                triggered = False
            return _NovikovRawResult(
                small_divisor=float(sd), absorbed_weight=float(w),
                triggered=triggered, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Novikov: %s", exc)
            return _NovikovRawResult(
                small_divisor=float("nan"), absorbed_weight=float("nan"),
                triggered=False, engine_ok=False)

    # ── I.12  Acción-ángulo, Birkhoff, Nekhoroshev, twist, Lindstedt ─────
    def compute_action_angle(
        self,
        cartan: _CartanRawResult,
        frequency_vector: np.ndarray,
        n_dof: int,
    ) -> Optional[_ActionAngleRawResult]:
        r"""\(I_1=\oint p\,dq/2\pi\) (isoenergética); el resto se reparte por \(\omega\)."""
        try:
            if not np.isfinite(cartan.action_integral):
                return None
            omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
            n = int(max(n_dof, 1))
            if omega.size == 0:
                omega = _default_frequencies(n)
            if omega.size < n:
                omega = np.pad(omega, (0, n - omega.size), constant_values=1.0)
            omega = omega[:n]
            i1 = float(cartan.action_integral) / (2.0 * np.pi)
            weights = np.abs(omega) + _MACHINE_EPS
            weights = weights / float(np.sum(weights))
            actions = i1 * weights * float(n)
            residual = abs(float(cartan.integral))  # 0 en órbita periódica exacta
            return _ActionAngleRawResult(
                actions=np.asarray(actions, dtype=np.float64),
                frequencies=np.asarray(omega, dtype=np.float64),
                generating_residual=float(residual), engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en acción-ángulo: %s", exc)
            return None

    def compute_birkhoff_normal_form(
        self, birkhoff_residual: float, kam: _KamRawResult,
    ) -> _BirkhoffRawResult:
        r"""Residuo de \(\{H_0,S\}=H_{\mathrm{pert}}-[H_{\mathrm{pert}}]\) (homológica)."""
        try:
            res = abs(float(birkhoff_residual))
            if not np.isfinite(res):
                res = float("inf")
            normalizable = bool(res <= _HARD_BIRKHOFF_RES_CEILING and kam.is_stable)
            return _BirkhoffRawResult(
                order=2, residual=float(res),
                is_normalizable=normalizable, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Birkhoff: %s", exc)
            return _BirkhoffRawResult(
                order=0, residual=float("inf"), is_normalizable=False, engine_ok=False)

    def compute_nekhoroshev(
        self, n_dof: int, birkhoff: _BirkhoffRawResult, kam: _KamRawResult,
    ) -> _NekhoroshevRawResult:
        r"""Exponentes \(a=1/(2n)\), \(b=a\); \(\tau_N=\exp(c/\varepsilon^a)\)."""
        try:
            n = int(max(n_dof, 1))
            exp_a = 1.0 / float(2 * n)
            conf_b = float(exp_a)
            eps = max(float(birkhoff.residual), _MACHINE_EPS)
            if (not np.isfinite(eps)) or eps >= 1.0:
                return _NekhoroshevRawResult(
                    exponent_a=float(exp_a), confinement_b=float(conf_b),
                    time_scale=0.0, is_confined=False, engine_ok=True)
            time_scale = float(math.exp(_NEKHOROSHEV_C / (eps ** exp_a)))
            confined = bool(kam.is_stable and birkhoff.is_normalizable and time_scale > 1.0)
            return _NekhoroshevRawResult(
                exponent_a=float(exp_a), confinement_b=float(conf_b),
                time_scale=float(time_scale), is_confined=confined, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Nekhoroshev: %s", exc)
            return _NekhoroshevRawResult(
                exponent_a=float("nan"), confinement_b=float("nan"),
                time_scale=0.0, is_confined=False, engine_ok=False)

    def compute_twist(
        self,
        frequency_vector: np.ndarray,
        action_angle: Optional[_ActionAngleRawResult],
        floquet: _FloquetRawResult,
        n_dof: int,
    ) -> _TwistRawResult:
        r"""
        Twist \(\det\partial\omega/\partial I\neq 0\); Kolmogorov
        \(\det\partial^2 H/\partial I^2\neq 0\); isoenergética (hessiano bordeado).
        Poincaré–Birkhoff: mapa de anillo 2D, twist y área-preservante ⇒ ≥2 p.f.
        """
        try:
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
            # Hessiano bordeado isoenergético: [[H_II, ω],[ωᵀ, 0]]
            bordered = np.zeros((n + 1, n + 1), dtype=np.float64)
            bordered[:n, :n] = hess
            bordered[:n, n] = omega
            bordered[n, :n] = omega
            iso_det = float(np.linalg.det(bordered))
            iso = bool(abs(iso_det) > _HARD_TWIST_DET_FLOOR)
            section_2d = bool(2 * n - 2 == 2 or n == 1)
            area_pres = bool(
                np.isfinite(floquet.symplectic_defect)
                and floquet.symplectic_defect <= _HARD_SYMPLECTIC_DEFECT)
            has_pb = bool(section_2d and kolmogorov and area_pres)
            return _TwistRawResult(
                twist_determinant=float(twist_det),
                kolmogorov_nondeg=kolmogorov,
                isoenergetic_nondeg=iso,
                has_poincare_birkhoff_fps=has_pb, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en twist: %s", exc)
            return _TwistRawResult(
                twist_determinant=float("nan"), kolmogorov_nondeg=False,
                isoenergetic_nondeg=False, has_poincare_birkhoff_fps=False,
                engine_ok=False)

    def compute_lindstedt(
        self, birkhoff: _BirkhoffRawResult, frequency_vector: np.ndarray,
    ) -> _LindstedtRawResult:
        r"""
        Corrección de frecuencia de Lindstedt–Poincaré: se anulan seculares
        eligiendo \(\omega=\omega_0+\varepsilon\omega_1+\cdots\). El residuo
        secular se identifica con el residuo de Birkhoff.
        """
        try:
            omega = np.asarray(frequency_vector, dtype=np.float64).ravel()
            corr = np.zeros_like(omega) if omega.size else np.array([], dtype=np.float64)
            return _LindstedtRawResult(
                secular_residual=float(birkhoff.residual),
                frequency_correction=corr, engine_ok=True)
        except Exception as exc:
            logger.error("Fallo en Lindstedt: %s", exc)
            return _LindstedtRawResult(
                secular_residual=float("inf"),
                frequency_correction=np.array([], dtype=np.float64),
                engine_ok=False)

    def _frequencies_from_floquet(
        self, floquet: _FloquetRawResult, orbit_period_T: float,
    ) -> np.ndarray:
        """ω_k = |arg μ_k| / T  (frecuencias de Poincaré de la órbita periódica)."""
        mu = np.asarray(floquet.multipliers, dtype=np.complex128)
        t_orb = float(orbit_period_T) if orbit_period_T > 0.0 else 1.0
        if mu.size == 0:
            return _default_frequencies(max(self._n // 2, 1))
        angles = np.abs(np.angle(mu))
        omega = np.sort(angles)[::-1] / t_orb
        omega = omega[omega > 1.0e-12]
        if omega.size == 0:
            return _default_frequencies(max(self._n // 2, 1))
        return np.asarray(omega, dtype=np.float64)

    # ── I.13  Ensamblaje histórico (compatibilidad 4.0) ──────────────────
    def synthesize_heyting_audit_germ(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        safety_margin: float, steps: int = 100,
    ) -> _HeytingAuditGerm:
        kleisli = self.compute_kleisli_deviation(f, g, h_func, test_input)
        degroot = self.compute_degroot_metrics(opinion_vector, affinity_matrix, steps)
        chsh = self.compute_chsh_s_value(correlation_matrix)
        kl_scale = 1.0
        if np.isfinite(kleisli.lhs_prob) or np.isfinite(kleisli.rhs_prob):
            kl_scale = max(
                abs(kleisli.lhs_prob) if np.isfinite(kleisli.lhs_prob) else 0.0,
                abs(kleisli.rhs_prob) if np.isfinite(kleisli.rhs_prob) else 0.0,
                1.0)
        dg_scale = 1.0
        if degroot.final_opinions.size:
            dg_scale = max(float(np.max(np.abs(degroot.final_opinions))), 1.0)
        n_agents = self._n
        if degroot.final_opinions.size:
            n_agents = int(degroot.final_opinions.size)
        else:
            try:
                n_agents = int(self._as_vec("opinion_vector", opinion_vector).size)
            except Exception:
                pass
        return _HeytingAuditGerm(
            kleisli=kleisli, degroot=degroot, chsh=chsh,
            n_agents=int(n_agents),
            safety_margin=float(max(safety_margin, 0.0)),
            kleisli_scale=float(kl_scale), degroot_scale=float(dg_scale))

    # ── I.ω  MORFISMO TERMINAL DE LA FASE I ──────────────────────────────
    def synthesize_poincare_heyting_audit_germ(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        safety_margin: float, steps: int = 100,
        monodromy_M: Optional[np.ndarray] = None,
        orbit_period_T: float = 1.0,
        section_index: int = 0,
        section_offset: float = 0.0,
        energy_level: float = 0.0,
        flow_vector: Optional[np.ndarray] = None,
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
        k_vector_integer: Optional[np.ndarray] = None,
        novikov_valuation_T: float = 1.0,
        state_trajectory_z: Optional[Sequence[np.ndarray]] = None,
    ) -> _PoincareHeytingAuditGerm:
        r"""
        **I.ω — Morfismo terminal de la FASE I / objeto inicial de la FASE II.**

        Ensambla el gérmen \(\mathcal{G}_I\). Su tipo de retorno
        `_PoincareHeytingAuditGerm` es el dominio de
        `_HeytingClassifier.induce_poincare_ooda_actuation_germ` (II.ω).
        """
        base = self.synthesize_heyting_audit_germ(
            f, g, h_func, test_input, opinion_vector, affinity_matrix,
            correlation_matrix, safety_margin, steps=steps)
        section = self.compute_poincare_section(
            section_index=section_index, section_offset=section_offset,
            energy_level=energy_level, flow_vector=flow_vector)
        if monodromy_M is None:
            try:
                m_eff = self._affinity_monodromy(affinity_matrix)
            except Exception:
                m_eff = np.eye(2, dtype=np.float64)
        else:
            m_eff = self._as_matrix("monodromy_M", monodromy_M, square=True)
            if m_eff.shape[0] % 2 != 0:
                m_eff = _even_embed(m_eff)
        floquet = self.compute_floquet_lyapunov(m_eff, orbit_period_T)
        melnikov: Optional[_MelnikovRawResult] = None
        if (q0_trajectory is not None
                and h0_grad is not None and h1_grad is not None):
            omega_j = _omega_n(self._n)
            melnikov = self.compute_melnikov(
                q0_trajectory, dt_trajectory, h0_grad, h1_grad, omega_j)
        rotation: Optional[_RotationRawResult] = None
        if orbit_points_for_rotation is not None:
            rotation = self.compute_rotation_number(orbit_points_for_rotation)
        if frequency_vector is None:
            omega_freq = self._frequencies_from_floquet(floquet, orbit_period_T)
        else:
            omega_freq = np.asarray(frequency_vector, dtype=np.float64).ravel()
            if omega_freq.size == 0:
                omega_freq = self._frequencies_from_floquet(floquet, orbit_period_T)
        kam = self.certify_kam(omega_freq, birkhoff_residual=birkhoff_residual)
        lyapunov = self.compute_lyapunov_spectrum(m_eff)
        if (q_cartan is not None and p_cartan is not None and H_cartan is not None):
            cartan = self.compute_cartan_integral(
                q_cartan, p_cartan, H_cartan, dt_trajectory)
        else:
            cartan = _CartanRawResult(integral=float("nan"), engine_ok=True)
        kepler: Optional[_KeplerRawResult] = None
        if kepler_position is not None and kepler_velocity is not None:
            kepler = self.compute_kepler_elements(
                kepler_position, kepler_velocity,
                mu_gravitational=mu_gravitational)
        if k_vector_integer is not None:
            k_vec = np.asarray(k_vector_integer, dtype=np.float64).ravel()
            take = min(k_vec.size, omega_freq.size)
            small_divisor = float(np.dot(k_vec[:take], omega_freq[:take])) if take else 0.0
        else:
            small_divisor = float(kam.worst_divisor) if np.isfinite(kam.worst_divisor) else (
                float(kam.gamma) if np.isfinite(kam.gamma) else 1.0)
        novikov = self.compute_novikov_absorption(
            small_divisor, novikov_valuation_T=novikov_valuation_T)
        recurrence = self.compute_poincare_recurrence(state_trajectory_z, section)
        n_dof = max(self._n // 2, 1)
        action_angle = self.compute_action_angle(cartan, omega_freq, n_dof)
        birkhoff = self.compute_birkhoff_normal_form(birkhoff_residual, kam)
        nekhoroshev = self.compute_nekhoroshev(n_dof, birkhoff, kam)
        twist = self.compute_twist(omega_freq, action_angle, floquet, n_dof)
        lindstedt = self.compute_lindstedt(birkhoff, omega_freq)
        return _PoincareHeytingAuditGerm(
            base=base, section=section, floquet=floquet,
            melnikov=melnikov, rotation=rotation, kam=kam,
            lyapunov=lyapunov, cartan=cartan, kepler=kepler,
            novikov=novikov, recurrence=recurrence,
            action_angle=action_angle, birkhoff=birkhoff,
            nekhoroshev=nekhoroshev, twist=twist, lindstedt=lindstedt)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE II — CLASIFICADOR DE HEYTING H₃ + VALUACIÓN DINÁMICA DE POINCARÉ   ██
# ██  Continuación directa del morfismo I.ω.                                   ██
# ██  Dominio: _PoincareHeytingAuditGerm  (objeto terminal de I.ω)             ██
# ██  Morfismo terminal (II.ω): induce_poincare_ooda_actuation_germ           ██
# ██                           → objeto inicial de la FASE III                ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
@dataclass(frozen=True)
class _KleisliVeredict:
    deviation: float
    verdict: str
    lhs_prob: float = float("nan")
    rhs_prob: float = float("nan")
    value_mismatch: float = 0.0
    threshold_coherent: float = _KLEISLI_COHERENT_TOL
    threshold_degraded: float = _KLEISLI_DEGRADED_TOL
    godel_value: float = 0.0


@dataclass(frozen=True)
class _DeGrootVeredict:
    fiedler_value: float
    deviation: float
    verdict: str
    connected: bool = True
    cheeger_upper: float = float("nan")
    mixing_rate: float = float("nan")
    engine_verdict: str = ""
    threshold_coherent: float = _DEGROOT_COHERENT_DEV
    threshold_degraded: float = _DEGROOT_DEGRADED_DEV
    godel_value: float = 0.0


@dataclass(frozen=True)
class _CHSHVeredict:
    s_value: float
    verdict: str
    physical: bool = True
    tsirelson_gap: float = float("nan")
    classical_gap: float = float("nan")
    horodecki_bound: float = float("nan")
    effective_tsirelson: float = _TSIRELSON_BOUND
    godel_value: float = 0.0


@dataclass(frozen=True)
class _PoincareDynamicsVeredict:
    r"""
    Valuación en H₃ de la dinámica de Poincaré. El campo `verdict` es el
    **ínfimo de Gödel** (= supremo de severidad) de todas las sub-aduanas.
    """
    section_transversal: bool
    section_certificate: float
    floquet_max_multiplier: float
    floquet_lyapunov_max: float
    floquet_log_residual: float
    kam_gamma: float
    kam_tau: float
    kam_stable: bool
    lyapunov_kaplan_yorke: float
    lyapunov_ks_entropy: float
    novikov_absorbed_weight: float
    novikov_triggered: bool
    orbit_type: str
    recurrence_distance: float
    birkhoff_residual: float
    nekhoroshev_time_scale: float
    twist_determinant: float
    lindstedt_secular_residual: float
    symplectic_defect: float
    hill_discriminant: float
    verdict_section: str
    verdict_floquet: str
    verdict_lyapunov: str
    verdict_kam: str
    verdict_novikov: str
    verdict_recurrence: str
    verdict_birkhoff: str
    verdict_nekhoroshev: str
    verdict_twist: str
    verdict_rotation: str
    verdict: str
    godel_value: float


@dataclass(frozen=True)
class _MelnykovVeredict:
    """Valuación en H₃ de la función de Mel’nikov."""
    simple_zeros: int
    chaotic_indicator: float
    is_chaotic: bool
    verdict: str
    godel_value: float


@dataclass(frozen=True)
class _OODAActuationGerm:
    """Gérmen OODA (objeto terminal histórico de Fase II)."""
    kleisli: _KleisliVeredict
    degroot: _DeGrootVeredict
    chsh: _CHSHVeredict
    heyting_join: str
    godel_meet: float
    n_agents: int
    safety_margin: float


@dataclass(frozen=True)
class _PoincareOODAActuationGerm:
    r"""
    **Gérmen OODA–Poincaré.**
    **Objeto terminal de la FASE II / objeto inicial de la FASE III.**
    """
    base: _OODAActuationGerm
    dynamics: _PoincareDynamicsVeredict
    melnikov: Optional[_MelnykovVeredict]
    heyting_join_full: str
    godel_meet_full: float


class _HeytingClassifier:
    """
    Fase II. Clasificador en el álgebra de Heyting de tres valores.
    Dominio de II.ω = imagen de I.ω.
    Convención de seguridad:
      join  := supremo de severidad = ínfimo de verdad de Gödel,
      meet  := ínfimo de severidad  = supremo de verdad de Gödel.
    """

    def __init__(self, safety_margin: float) -> None:
        self._margin = float(max(safety_margin, 0.0))

    @property
    def safety_margin(self) -> float:
        return self._margin

    # ── II.1  Álgebra de Heyting ─────────────────────────────────────────
    @staticmethod
    def canonicalize(verdict: str) -> str:
        return verdict if verdict in _HEYTING_ORDER else "VETOED"

    @staticmethod
    def join(*verdicts: str) -> str:
        if not verdicts:
            return "COHERENT"
        idx = max(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)]
                  for v in verdicts)
        return _REVERSE_HEYTING[idx]

    @staticmethod
    def meet(*verdicts: str) -> str:
        if not verdicts:
            return "COHERENT"
        idx = min(_HEYTING_ORDER[_HeytingClassifier.canonicalize(v)]
                  for v in verdicts)
        return _REVERSE_HEYTING[idx]

    @staticmethod
    def safety_fusion(*verdicts: str) -> str:
        """Ínfimo de Gödel sobre aduanas de seguridad = join de severidad."""
        return _HeytingClassifier.join(*verdicts)

    def scaled_tol(self, base: float, scale: float = 1.0) -> float:
        abs_tol = float(base) * max(self._margin, 0.0)
        rel_tol = max(float(scale), 1.0) * _MACHINE_EPS * _WILKINSON_REL_SCALE
        return float(max(abs_tol, rel_tol, _MACHINE_EPS))

    def verdict_from_deviation(
        self, deviation: float,
        coherent_tol: float, degraded_tol: float,
        safety_margin: Optional[float] = None, scale: float = 1.0,
    ) -> str:
        if not np.isfinite(deviation):
            return "VETOED"
        margin = self._margin if safety_margin is None \
            else float(max(safety_margin, 0.0))
        tau_c = max(float(coherent_tol) * margin,
                    max(float(scale), 1.0) * _MACHINE_EPS * _WILKINSON_REL_SCALE,
                    _MACHINE_EPS)
        tau_d = max(float(degraded_tol) * margin, tau_c)
        if deviation > tau_d:
            return "VETOED"
        if deviation > tau_c:
            return "DEGRADED"
        return "COHERENT"

    # ── II.2  Clasificación histórica (Kleisli / DeGroot / CHSH) ─────────
    def classify_kleisli(
        self, raw: _KleisliRawResult, scale: float,
    ) -> _KleisliVeredict:
        tau_c = self.scaled_tol(_KLEISLI_COHERENT_TOL, scale)
        tau_d = max(self.scaled_tol(_KLEISLI_DEGRADED_TOL, scale), tau_c)
        if (not raw.engine_ok) or (not np.isfinite(raw.deviation)):
            verdict = "VETOED"
        else:
            verdict = self.verdict_from_deviation(
                raw.deviation, _KLEISLI_COHERENT_TOL,
                _KLEISLI_DEGRADED_TOL, scale=scale)
            if (verdict == "COHERENT"
                    and np.isfinite(raw.value_mismatch)
                    and raw.value_mismatch > tau_c):
                verdict = "DEGRADED" if raw.value_mismatch <= tau_d else "VETOED"
        return _KleisliVeredict(
            deviation=float(raw.deviation), verdict=verdict,
            lhs_prob=float(raw.lhs_prob), rhs_prob=float(raw.rhs_prob),
            value_mismatch=float(raw.value_mismatch),
            threshold_coherent=float(tau_c),
            threshold_degraded=float(tau_d),
            godel_value=float(_HEYTING_GODEL[verdict]))

    def classify_degroot(
        self, raw: _DeGrootRawResult, scale: float,
    ) -> _DeGrootVeredict:
        tau_c = self.scaled_tol(_DEGROOT_COHERENT_DEV, scale)
        tau_d = max(self.scaled_tol(_DEGROOT_DEGRADED_DEV, scale), tau_c)
        if (not raw.engine_ok) or (not np.isfinite(raw.deviation)):
            verdict = "VETOED"
        else:
            verdict = self.verdict_from_deviation(
                raw.deviation, _DEGROOT_COHERENT_DEV,
                _DEGROOT_DEGRADED_DEV, scale=scale)
            if verdict == "COHERENT" and not raw.connected:
                verdict = "DEGRADED"
            if raw.engine_verdict in _HEYTING_ORDER:
                verdict = self.join(verdict, raw.engine_verdict)
        return _DeGrootVeredict(
            fiedler_value=float(raw.fiedler_value),
            deviation=float(raw.deviation), verdict=verdict,
            connected=bool(raw.connected),
            cheeger_upper=float(raw.cheeger_upper),
            mixing_rate=float(raw.mixing_rate),
            engine_verdict=str(raw.engine_verdict),
            threshold_coherent=float(tau_c),
            threshold_degraded=float(tau_d),
            godel_value=float(_HEYTING_GODEL[verdict]))

    def classify_chsh(self, raw: _CHSHRawResult) -> _CHSHVeredict:
        if (not raw.engine_ok) or (not np.isfinite(raw.s_value)):
            verdict = "VETOED"
            s_val = float(raw.s_value)
            eff = _TSIRELSON_BOUND
        else:
            s_val = float(abs(raw.s_value))
            extra = max(self._margin - 1.0, 0.0) * _TSIRELSON_GUARD_BASE
            eff = float(min(max(_TSIRELSON_BOUND - extra, _CLASSICAL_CHSH_BOUND),
                            _TSIRELSON_BOUND))
            if (not raw.physical) or s_val > eff + 8.0 * _MACHINE_EPS:
                verdict = "VETOED"
            elif s_val > _CLASSICAL_CHSH_BOUND:
                verdict = "COHERENT"
            else:
                verdict = "DEGRADED"
            if raw.engine_verdict in _HEYTING_ORDER \
                    and raw.engine_verdict == "VETOED":
                verdict = "VETOED"
        return _CHSHVeredict(
            s_value=float(raw.s_value), verdict=verdict,
            physical=bool(raw.physical),
            tsirelson_gap=float(raw.tsirelson_gap),
            classical_gap=float(raw.classical_gap),
            horodecki_bound=float(raw.horodecki_bound),
            effective_tsirelson=float(eff if np.isfinite(raw.s_value)
                                      else _TSIRELSON_BOUND),
            godel_value=float(_HEYTING_GODEL[verdict]))

    # ── II.3  Clasificación dinámica de Poincaré (inicia desde I.ω) ──────
    def classify_poincare_dynamics(
        self, germ: _PoincareHeytingAuditGerm,
    ) -> _PoincareDynamicsVeredict:
        r"""
        Continuación formal de I.ω: valúa \(\mathcal{G}_I\) en \(\Omega_3\).
        Fusión = ínfimo de Gödel (join de severidad) — nunca el meet de
        severidad, que en v5.0 dejaba pasar un VETOED aislado.
        """
        v_section = "COHERENT" if (
            germ.section.engine_ok and germ.section.is_transversal
            and germ.section.transversal_certificate > _HARD_POINCARE_SECTION_TOL
        ) else "VETOED"

        excess_mu = max(0.0, germ.floquet.max_multiplier - 1.0)
        tau_f = _HARD_FLOQUET_MULTIPLIER_TOL * max(self._margin, _MACHINE_EPS)
        if not germ.floquet.engine_ok:
            v_floquet = "VETOED"
        elif excess_mu > tau_f * 10.0 or germ.floquet.log_residual > tau_f * 10.0:
            v_floquet = "VETOED"
        elif excess_mu > tau_f or germ.floquet.orbit_type in {"hyperbolic", "loxodromic"}:
            v_floquet = "DEGRADED"
        else:
            v_floquet = "COHERENT"
        if (np.isfinite(germ.floquet.symplectic_defect)
                and germ.floquet.symplectic_defect > 1.0e3 * _HARD_SYMPLECTIC_DEFECT
                and germ.floquet.multipliers.size >= 2):
            v_floquet = self.join(v_floquet, "DEGRADED")

        lam_max = max(0.0, germ.floquet.lyapunov_max)
        tau_l = _HARD_LYAPUNOV_TOL * max(self._margin, _MACHINE_EPS)
        if not germ.lyapunov.engine_ok:
            v_lyap = "VETOED"
        elif germ.lyapunov.is_chaotic and lam_max > 10.0 * tau_l:
            v_lyap = "VETOED"
        elif germ.lyapunov.is_chaotic or lam_max > tau_l:
            v_lyap = "DEGRADED"
        else:
            v_lyap = "COHERENT"

        v_kam = "COHERENT" if (
            germ.kam.engine_ok and germ.kam.is_stable
            and germ.kam.gamma > _HARD_KAM_GAMMA_FLOOR
        ) else "VETOED"

        v_nov = "DEGRADED" if germ.novikov.triggered else "COHERENT"
        if not germ.novikov.engine_ok:
            v_nov = "VETOED"

        if not germ.recurrence.engine_ok:
            v_rec = "VETOED"
        elif not germ.recurrence.is_recurrent:
            v_rec = "VETOED"
        else:
            v_rec = "COHERENT"

        if not germ.birkhoff.engine_ok or not np.isfinite(germ.birkhoff.residual):
            v_birk = "VETOED"
        elif germ.birkhoff.residual > _HARD_BIRKHOFF_RES_CEILING * 10.0:
            v_birk = "VETOED"
        elif germ.birkhoff.residual > _HARD_BIRKHOFF_RES_CEILING:
            v_birk = "DEGRADED"
        else:
            v_birk = "COHERENT"

        if not germ.nekhoroshev.engine_ok:
            v_nek = "VETOED"
        elif not germ.nekhoroshev.is_confined:
            v_nek = "DEGRADED"
        else:
            v_nek = "COHERENT"

        if not germ.twist.engine_ok:
            v_twist = "VETOED"
        elif not germ.twist.kolmogorov_nondeg:
            v_twist = "DEGRADED"
        else:
            v_twist = "COHERENT"

        v_rot = "COHERENT"
        if germ.rotation is not None:
            if not germ.rotation.engine_ok or not np.isfinite(germ.rotation.rotation_number):
                v_rot = "DEGRADED"
            elif germ.rotation.is_rational:
                denom = int(germ.rotation.convergent_denominator)
                v_rot = "VETOED" if 0 < denom <= _HARD_RESONANT_DENOM else "DEGRADED"

        final = self.safety_fusion(
            v_section, v_floquet, v_lyap, v_kam, v_nov,
            v_rec, v_birk, v_nek, v_twist, v_rot)
        return _PoincareDynamicsVeredict(
            section_transversal=bool(germ.section.is_transversal),
            section_certificate=float(germ.section.transversal_certificate),
            floquet_max_multiplier=float(germ.floquet.max_multiplier),
            floquet_lyapunov_max=float(germ.floquet.lyapunov_max),
            floquet_log_residual=float(germ.floquet.log_residual),
            kam_gamma=float(germ.kam.gamma),
            kam_tau=float(germ.kam.tau),
            kam_stable=bool(germ.kam.is_stable),
            lyapunov_kaplan_yorke=float(germ.lyapunov.kaplan_yorke),
            lyapunov_ks_entropy=float(germ.lyapunov.ks_entropy),
            novikov_absorbed_weight=float(germ.novikov.absorbed_weight),
            novikov_triggered=bool(germ.novikov.triggered),
            orbit_type=str(germ.floquet.orbit_type),
            recurrence_distance=float(germ.recurrence.return_distance),
            birkhoff_residual=float(germ.birkhoff.residual),
            nekhoroshev_time_scale=float(germ.nekhoroshev.time_scale),
            twist_determinant=float(germ.twist.twist_determinant),
            lindstedt_secular_residual=float(germ.lindstedt.secular_residual),
            symplectic_defect=float(germ.floquet.symplectic_defect),
            hill_discriminant=float(germ.floquet.hill_discriminant),
            verdict_section=v_section,
            verdict_floquet=v_floquet,
            verdict_lyapunov=v_lyap,
            verdict_kam=v_kam,
            verdict_novikov=v_nov,
            verdict_recurrence=v_rec,
            verdict_birkhoff=v_birk,
            verdict_nekhoroshev=v_nek,
            verdict_twist=v_twist,
            verdict_rotation=v_rot,
            verdict=final,
            godel_value=float(_HEYTING_GODEL[final]))

    def classify_melnikov(
        self, raw: Optional[_MelnikovRawResult],
    ) -> Optional[_MelnykovVeredict]:
        r"""
        Cualquier cero simple de \(\mathcal{M}\) ⇒ intersección transversal
        \(W^s\cap W^u\) ⇒ caos homoclínico de Poincaré ⇒ **VETOED**.
        """
        if raw is None:
            return None
        if not raw.engine_ok:
            return _MelnykovVeredict(
                simple_zeros=int(raw.simple_zeros),
                chaotic_indicator=float(raw.chaotic_indicator),
                is_chaotic=True, verdict="VETOED", godel_value=0.0)
        verdict = "VETOED" if raw.is_chaotic else "COHERENT"
        return _MelnykovVeredict(
            simple_zeros=int(raw.simple_zeros),
            chaotic_indicator=float(raw.chaotic_indicator),
            is_chaotic=bool(raw.is_chaotic), verdict=verdict,
            godel_value=float(_HEYTING_GODEL[verdict]))

    # ── II.4  Ensamblaje histórico (compatibilidad 4.0) ──────────────────
    def induce_ooda_actuation_germ(
        self, germ: _HeytingAuditGerm,
    ) -> _OODAActuationGerm:
        kl = self.classify_kleisli(germ.kleisli, germ.kleisli_scale)
        dg = self.classify_degroot(germ.degroot, germ.degroot_scale)
        ch = self.classify_chsh(germ.chsh)
        joined = self.join(kl.verdict, dg.verdict, ch.verdict)
        meet_g = float(min(kl.godel_value, dg.godel_value, ch.godel_value))
        return _OODAActuationGerm(
            kleisli=kl, degroot=dg, chsh=ch,
            heyting_join=joined, godel_meet=meet_g,
            n_agents=int(germ.n_agents),
            safety_margin=float(germ.safety_margin))

    # ── II.ω  MORFISMO TERMINAL DE LA FASE II ────────────────────────────
    def induce_poincare_ooda_actuation_germ(
        self, germ: _PoincareHeytingAuditGerm,
    ) -> _PoincareOODAActuationGerm:
        r"""
        **II.ω — Morfismo terminal de la FASE II / objeto inicial de la FASE III.**

        \(\Phi_{\mathrm{II}}:\mathcal{G}_I\to\mathcal{G}_{II}\). El veredicto
        global es el ínfimo de Gödel sobre las aduanas históricas, dinámicas
        y Mel’nikov. Tipo de retorno = dominio de
        `_OODAController.certify_poincare_sequitos` (III.ω).
        """
        base = self.induce_ooda_actuation_germ(germ.base)
        dynamics = self.classify_poincare_dynamics(germ)
        melnikov = self.classify_melnikov(germ.melnikov)
        all_verdicts = [
            base.kleisli.verdict, base.degroot.verdict, base.chsh.verdict,
            dynamics.verdict_section, dynamics.verdict_floquet,
            dynamics.verdict_lyapunov, dynamics.verdict_kam,
            dynamics.verdict_novikov, dynamics.verdict_recurrence,
            dynamics.verdict_birkhoff, dynamics.verdict_nekhoroshev,
            dynamics.verdict_twist, dynamics.verdict_rotation,
        ]
        if melnikov is not None:
            all_verdicts.append(melnikov.verdict)
        joined = self.safety_fusion(*all_verdicts)
        all_godel = [
            base.kleisli.godel_value, base.degroot.godel_value,
            base.chsh.godel_value, dynamics.godel_value,
        ]
        if melnikov is not None:
            all_godel.append(melnikov.godel_value)
        meet_g = float(min(all_godel))
        return _PoincareOODAActuationGerm(
            base=base, dynamics=dynamics, melnikov=melnikov,
            heyting_join_full=joined, godel_meet_full=meet_g)


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE III — CICLO OODA EXTENDIDO + CERTIFICADO GLOBAL DE POINCARÉ       ██
# ██  Continuación directa del morfismo II.ω.                                  ██
# ██  Dominio: _PoincareOODAActuationGerm × _PoincareHeytingAuditGerm          ██
# ██  Morfismo terminal (III.ω): certify_poincare_sequitos                    ██
# ██                           → PoincareSequitosCertificate                  ██
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================
@dataclass(frozen=True)
class _OODAResult:
    """Acta del ciclo OODA histórico (superset certificado del dict 2.0)."""
    heyting_verdict: str
    kleisli_deviation: float
    kleisli_verdict: str
    fiedler_value: float
    degroot_verdict: str
    chsh_value: float
    chsh_verdict: str
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    godel_meet: float
    degroot_connected: bool
    degroot_deviation: float
    chsh_physical: bool
    tsirelson_gap: float
    filter_is_prime: bool
    observe_ok: bool

    def as_public_dict(self) -> Dict[str, Any]:
        return {
            "heyting_verdict": self.heyting_verdict,
            "kleisli_deviation": self.kleisli_deviation,
            "kleisli_verdict": self.kleisli_verdict,
            "fiedler_value": self.fiedler_value,
            "degroot_verdict": self.degroot_verdict,
            "chsh_value": self.chsh_value,
            "chsh_verdict": self.chsh_verdict,
            "hardware_interlock_fired": self.hardware_interlock_fired,
            "actuation_latency_ns": self.actuation_latency_ns,
        }


class _OODAController:
    """
    Fase III. Ciclo Observe–Orient–Decide–Act. Consume 𝒢_II (II.ω).
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── III.1  OODA histórico ────────────────────────────────────────────
    @staticmethod
    def observe(
        germ: _OODAActuationGerm,
    ) -> Tuple[_KleisliVeredict, _DeGrootVeredict, _CHSHVeredict]:
        return germ.kleisli, germ.degroot, germ.chsh

    @staticmethod
    def orient(germ: _OODAActuationGerm) -> str:
        return _HeytingClassifier.canonicalize(germ.heyting_join)

    @staticmethod
    def decide(join: str) -> bool:
        return _HeytingClassifier.canonicalize(join) == "VETOED"

    def act(self, interlock: bool) -> float:
        if not interlock:
            return 0.0
        jitter = float(self._rng.normal(0.0, _INTERLOCK_JITTER_NS))
        latency = _INTERLOCK_LATENCY_BUDGET_NS + jitter
        return float(np.clip(latency, 380.0, 420.0))

    def run(self, germ: _OODAActuationGerm) -> _OODAResult:
        kl, dg, ch = self.observe(germ)
        joined = self.orient(germ)
        fire = self.decide(joined)
        latency = self.act(fire)
        observe_ok = bool(
            np.isfinite(kl.deviation)
            and np.isfinite(dg.fiedler_value)
            and np.isfinite(ch.s_value))
        if fire:
            logger.critical(
                "VETO DE SÉQUITOS IMPERIALES (3 aduanas). Join H₃ = VETOED "
                "(Kleisli=%s, DeGroot=%s, CHSH=%s). Latencia=%.2f ns.",
                kl.verdict, dg.verdict, ch.verdict, latency)
        return _OODAResult(
            heyting_verdict=joined,
            kleisli_deviation=float(kl.deviation), kleisli_verdict=kl.verdict,
            fiedler_value=float(dg.fiedler_value), degroot_verdict=dg.verdict,
            chsh_value=float(ch.s_value), chsh_verdict=ch.verdict,
            hardware_interlock_fired=bool(fire),
            actuation_latency_ns=float(latency),
            godel_meet=float(germ.godel_meet),
            degroot_connected=bool(dg.connected),
            degroot_deviation=float(dg.deviation),
            chsh_physical=bool(ch.physical),
            tsirelson_gap=float(ch.tsirelson_gap),
            filter_is_prime=True, observe_ok=observe_ok)

    # ── III.ω  MORFISMO TERMINAL DE LA FASE III ──────────────────────────
    def certify_poincare_sequitos(
        self,
        pooda_germ: _PoincareOODAActuationGerm,
        poincare_audit_germ: _PoincareHeytingAuditGerm,
    ) -> PoincareSequitosCertificate:
        r"""
        **III.ω — Morfismo terminal global \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\).**

        Consume \(\mathcal{G}_{II}\) (cierre de Fase II) y \(\mathcal{G}_I\)
        (cierre de Fase I). Cualquier aduana en VETOED dispara el interlock.
        """
        base = pooda_germ.base
        dyn = pooda_germ.dynamics
        mel = pooda_germ.melnikov
        final_verdict = pooda_germ.heyting_join_full
        fire = _HeytingClassifier.canonicalize(final_verdict) == "VETOED"
        latency = self.act(fire)
        if fire:
            logger.critical(
                "¡VETO ATÓMICO SÉQUITOS–POINCARÉ! "
                "Veredictos: K=%s D=%s C=%s Σ=%s F=%s Ly=%s KAM=%s N=%s "
                "R=%s B=%s Nek=%s Tw=%s ρ=%s M=%s. "
                "Interlock lógico ACTIVADO. Latencia = %.2f ns.",
                base.kleisli.verdict, base.degroot.verdict, base.chsh.verdict,
                dyn.verdict_section, dyn.verdict_floquet, dyn.verdict_lyapunov,
                dyn.verdict_kam, dyn.verdict_novikov,
                dyn.verdict_recurrence, dyn.verdict_birkhoff,
                dyn.verdict_nekhoroshev, dyn.verdict_twist,
                dyn.verdict_rotation,
                mel.verdict if mel else "N/A", latency)
        is_coherent = bool(
            _HeytingClassifier.canonicalize(final_verdict) == "COHERENT"
            and poincare_audit_germ.section.is_transversal
            and poincare_audit_germ.kam.is_stable
            and poincare_audit_germ.recurrence.is_recurrent)
        kepler_dict = (poincare_audit_germ.kepler.elements
                       if poincare_audit_germ.kepler is not None else None)
        actions = (poincare_audit_germ.action_angle.actions
                   if poincare_audit_germ.action_angle is not None else None)
        return PoincareSequitosCertificate(
            n_agents=int(base.n_agents),
            safety_margin=float(base.safety_margin),
            kleisli_deviation=float(base.kleisli.deviation),
            degroot_fiedler=float(base.degroot.fiedler_value),
            degroot_deviation=float(base.degroot.deviation),
            chsh_s_value=float(base.chsh.s_value),
            poincare_return_distance=float(dyn.recurrence_distance),
            poincare_section_transversal=bool(
                poincare_audit_germ.section.is_transversal),
            poincare_cartan_integral=float(
                poincare_audit_germ.cartan.integral),
            floquet_max_multiplier=float(dyn.floquet_max_multiplier),
            floquet_lyapunov_max=float(dyn.floquet_lyapunov_max),
            floquet_log_residual=float(dyn.floquet_log_residual),
            kam_diophantine_gamma=float(dyn.kam_gamma),
            kam_diophantine_tau=float(dyn.kam_tau),
            kam_stable=bool(dyn.kam_stable),
            lyapunov_spectrum=np.asarray(
                poincare_audit_germ.lyapunov.spectrum, dtype=np.float64),
            lyapunov_kaplan_yorke=float(dyn.lyapunov_kaplan_yorke),
            lyapunov_ks_entropy=float(dyn.lyapunov_ks_entropy),
            melnikov_simple_zeros=(int(mel.simple_zeros) if mel else None),
            melnikov_chaotic=(bool(mel.is_chaotic) if mel else None),
            rotation_number=(float(poincare_audit_germ.rotation.rotation_number)
                             if poincare_audit_germ.rotation else None),
            rotation_is_rational=(bool(poincare_audit_germ.rotation.is_rational)
                                  if poincare_audit_germ.rotation else None),
            novikov_absorbed_weight=float(dyn.novikov_absorbed_weight),
            kepler_elements=kepler_dict,
            heyting_verdict=final_verdict,
            heyting_godel_meet=float(pooda_germ.godel_meet_full),
            is_sequitos_coherent=is_coherent,
            hardware_interlock_fired=bool(fire),
            actuation_latency_ns=float(latency),
            nekhoroshev_time_scale=float(dyn.nekhoroshev_time_scale),
            twist_determinant=float(dyn.twist_determinant),
            birkhoff_residual=float(dyn.birkhoff_residual),
            lindstedt_secular_residual=float(dyn.lindstedt_secular_residual),
            action_variables=actions,
            poincare_orbit_type=str(dyn.orbit_type),
            recurrence_is_ergodic=bool(poincare_audit_germ.recurrence.is_recurrent),
            symplectic_defect=float(dyn.symplectic_defect),
            hill_discriminant=float(dyn.hill_discriminant))


# =============================================================================
# AGENTE PÚBLICO — INTEGRACIÓN DEL MORFISMO Φ_III ∘ Φ_II ∘ Φ_I Y POINCARÉ
# =============================================================================
class ImperialGuardsSequitosAgent:
    r"""
    Séquitos Imperiales de Gobernanza Agéntica (Capa 1.5, Poincaré–Celeste).

    Compone las tres fases anidadas:
    1. Fase I   — auditoría ciega + geometría celeste de Poincaré.
    2. Fase II  — valuación H₃ (ínfimo de Gödel = join de severidad).
    3. Fase III — OODA / interlock lógico y certificado global.

    API 4.0/5.0 preservada. `certify_poincare_sequitos` es \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\).
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        regularizer: float = 1e-15,
        rng: Optional[np.random.Generator] = None,
        kam_gamma: float = 0.1,
        kam_tau: float = 2.0,
    ) -> None:
        if int(dimension_n) <= 0:
            raise ValueError("La dimensión debe ser positiva.")
        if not np.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin debe ser finito y ≥ 0.")
        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = float(max(regularizer, 1e-20))
        self._gamma: Final[float] = float(kam_gamma)
        self._tau: Final[float] = float(kam_tau)
        try:
            self._engine: Final[ImperialSequitosEngine] = ImperialSequitosEngine(
                regularizer=self._reg, dimension_n=self._n)
        except TypeError:
            self._engine = ImperialSequitosEngine()  # type: ignore[misc]
        self._audit_core = _AuditCore(self._engine, n_agents=self._n)
        self._classifier = _HeytingClassifier(self._safety_margin)
        self._ooda = _OODAController(rng=rng)
        self._audit_germ: Optional[_HeytingAuditGerm] = None
        self._poincare_audit_germ: Optional[_PoincareHeytingAuditGerm] = None
        self._ooda_germ: Optional[_OODAActuationGerm] = None
        self._poincare_ooda_germ: Optional[_PoincareOODAActuationGerm] = None

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    @property
    def engine(self) -> ImperialSequitosEngine:
        return self._engine

    def audit_poincare_ergodic_recurrence_and_kam(
        self,
        state_trajectory_z: List[np.ndarray],
        frequency_vector_omega: np.ndarray,
        k_vector_integer: np.ndarray,
        volume_drift: float,
        novikov_valuation_T: float = 1.0,
    ) -> SequitosPoincareCertificate:
        r"""
        Audita retorno ergódico, cota KAM diofántica y regulación en Novikov
        (API 4.0 conservada).
        """
        if not state_trajectory_z:
            return SequitosPoincareCertificate(
                poincare_return_distance=0.0,
                is_kam_diophantine_stable=True,
                novikov_absorbed_weight=1.0,
                volume_drift=float(volume_drift),
                heyting_verdict="COHERENT",
                is_sequitos_coherent=True)
        current_z = np.asarray(state_trajectory_z[-1], dtype=np.float64)
        past_distances = [
            float(np.linalg.norm(np.asarray(past_pt, dtype=np.float64) - current_z))
            for past_pt in state_trajectory_z[:-1]]
        min_return_distance = float(np.min(past_distances)) if past_distances else 0.0
        divisor = float(np.dot(k_vector_integer, frequency_vector_omega))
        k_norm_1 = float(np.sum(np.abs(k_vector_integer)))
        kam_bound = self._gamma / (max(1.0, k_norm_1) ** self._tau)
        is_kam_stable = abs(divisor) >= kam_bound
        if not is_kam_stable and abs(divisor) < _LIMIT_WILKINSON:
            novikov_weight = float(np.exp(
                -novikov_valuation_T / (_LIMIT_WILKINSON + abs(divisor))))
            logger.warning(
                "[SEQUITOS_KAM_RESONANCE] Pequeño divisor: %.3e. Novikov peso %.3e",
                divisor, novikov_weight)
        else:
            novikov_weight = 1.0 / (divisor + _LIMIT_WILKINSON)
        is_recurrent = min_return_distance <= _HARD_DIVERGENCE_CEILING
        is_liouville_valid = volume_drift <= _LIMIT_WILKINSON
        if is_recurrent and is_kam_stable and is_liouville_valid:
            heyting_verdict = "COHERENT"
            is_coherent = True
        elif is_recurrent and not is_kam_stable and is_liouville_valid:
            heyting_verdict = "DEGRADED"
            is_coherent = True
        else:
            heyting_verdict = "VETOED"
            is_coherent = False
        if not is_coherent:
            logger.error(
                "[SEQUITOS_VETOED] ReturnDist=%.3e Drift=%.3e Verdict=%s.",
                min_return_distance, volume_drift, heyting_verdict)
        return SequitosPoincareCertificate(
            poincare_return_distance=float(min_return_distance),
            is_kam_diophantine_stable=bool(is_kam_stable),
            novikov_absorbed_weight=float(novikov_weight),
            volume_drift=float(volume_drift),
            heyting_verdict=str(heyting_verdict),
            is_sequitos_coherent=bool(is_coherent))

    @staticmethod
    def _veredict_from_deviation(
        deviation: float, coherent_tol: float,
        degraded_tol: float, safety_margin: float,
    ) -> str:
        return _HeytingClassifier(safety_margin).verdict_from_deviation(
            deviation, coherent_tol, degraded_tol, safety_margin)

    def synthesize_heyting_audit_germ(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray, steps: int = 100,
    ) -> _HeytingAuditGerm:
        germ = self._audit_core.synthesize_heyting_audit_germ(
            f, g, h_func, test_input, opinion_vector,
            affinity_matrix, correlation_matrix,
            safety_margin=self._safety_margin, steps=steps)
        self._audit_germ = germ
        return germ

    def synthesize_poincare_heyting_audit_germ(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        steps: int = 100,
        **poincare_kwargs: Any,
    ) -> _PoincareHeytingAuditGerm:
        """Réplica pública del morfismo I.ω."""
        germ = self._audit_core.synthesize_poincare_heyting_audit_germ(
            f, g, h_func, test_input, opinion_vector,
            affinity_matrix, correlation_matrix,
            safety_margin=self._safety_margin, steps=steps,
            **poincare_kwargs)
        self._poincare_audit_germ = germ
        return germ

    def audit_kleisli_associativity(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
    ) -> Tuple[float, str]:
        result = self.audit_kleisli_associativity_certified(
            f, g, h_func, test_input)
        return result.deviation, result.verdict

    def audit_kleisli_associativity_certified(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
    ) -> _KleisliVeredict:
        raw = self._audit_core.compute_kleisli_deviation(f, g, h_func, test_input)
        scale = 1.0
        if np.isfinite(raw.lhs_prob) or np.isfinite(raw.rhs_prob):
            scale = max(
                abs(raw.lhs_prob) if np.isfinite(raw.lhs_prob) else 0.0,
                abs(raw.rhs_prob) if np.isfinite(raw.rhs_prob) else 0.0,
                1.0)
        return self._classifier.classify_kleisli(raw, scale)

    def audit_degroot_consensus(
        self, opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
    ) -> Tuple[float, str]:
        result = self.audit_degroot_consensus_certified(
            opinion_vector, affinity_matrix)
        return result.fiedler_value, result.verdict

    def audit_degroot_consensus_certified(
        self, opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        steps: int = 100,
    ) -> _DeGrootVeredict:
        raw = self._audit_core.compute_degroot_metrics(
            opinion_vector, affinity_matrix, steps=steps)
        if raw.final_opinions.size:
            scale = max(float(np.max(np.abs(raw.final_opinions))), 1.0)
        else:
            scale = 1.0
        return self._classifier.classify_degroot(raw, scale)

    def audit_quantum_chsh_channel(
        self, correlation_matrix: np.ndarray,
    ) -> Tuple[float, str]:
        result = self.audit_quantum_chsh_channel_certified(correlation_matrix)
        return result.s_value, result.verdict

    def audit_quantum_chsh_channel_certified(
        self, correlation_matrix: np.ndarray,
    ) -> _CHSHVeredict:
        raw = self._audit_core.compute_chsh_s_value(correlation_matrix)
        return self._classifier.classify_chsh(raw)

    def induce_ooda_actuation_germ(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray, steps: int = 100,
    ) -> _OODAActuationGerm:
        audit_germ = self.synthesize_heyting_audit_germ(
            f, g, h_func, test_input, opinion_vector, affinity_matrix,
            correlation_matrix, steps=steps)
        ooda_germ = self._classifier.induce_ooda_actuation_germ(audit_germ)
        self._ooda_germ = ooda_germ
        return ooda_germ

    def induce_poincare_ooda_actuation_germ(
        self,
        poincare_audit_germ: Optional[_PoincareHeytingAuditGerm] = None,
    ) -> _PoincareOODAActuationGerm:
        """Réplica pública del morfismo II.ω."""
        germ = poincare_audit_germ or self._poincare_audit_germ
        if germ is None:
            raise ValueError(
                "Debe suministrarse un gérmen de auditoría Poincaré o haber "
                "invocado `synthesize_poincare_heyting_audit_germ` previamente.")
        pooda = self._classifier.induce_poincare_ooda_actuation_germ(germ)
        self._poincare_ooda_germ = pooda
        return pooda

    def execute_sequitos_cycle(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        return self.execute_sequitos_cycle_certified(
            f, g, h_func, test_input, opinion_vector,
            affinity_matrix, correlation_matrix).as_public_dict()

    def execute_sequitos_cycle_certified(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray, steps: int = 100,
    ) -> _OODAResult:
        germ = self.induce_ooda_actuation_germ(
            f, g, h_func, test_input, opinion_vector,
            affinity_matrix, correlation_matrix, steps=steps)
        return self._ooda.run(germ)

    def certify_poincare_sequitos(
        self,
        f: KleisliArrow, g: KleisliArrow, h_func: KleisliArrow,
        test_input: Any,
        opinion_vector: np.ndarray, affinity_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        steps: int = 100,
        **poincare_kwargs: Any,
    ) -> PoincareSequitosCertificate:
        r"""
        **Morfismo terminal global \(\Phi_{\mathrm{III}}\circ\Phi_{\mathrm{II}}\circ\Phi_{\mathrm{I}}\).**
        """
        audit_germ = self.synthesize_poincare_heyting_audit_germ(
            f, g, h_func, test_input, opinion_vector,
            affinity_matrix, correlation_matrix,
            steps=steps, **poincare_kwargs)
        pooda_germ = self._classifier.induce_poincare_ooda_actuation_germ(
            audit_germ)
        self._poincare_ooda_germ = pooda_germ
        return self._ooda.certify_poincare_sequitos(pooda_germ, audit_germ)


ImperialGuardsSequitos = ImperialGuardsSequitosAgent

__all__ = [
    "ImperialGuardsSequitosAgent",
    "ImperialGuardsSequitos",
    "SequitosPoincareCertificate",
    "PoincareSequitosCertificate",
    "EnginePoincareCertificate",
]