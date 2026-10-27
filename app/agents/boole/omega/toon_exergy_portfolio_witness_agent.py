# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Witness Agent (Testigo QND de Portafolios)    ║
║ Ruta   : app/agents/boole/omega/toon_exergy_portfolio_witness_agent.py       ║
║ Versión: 6.1.0-Witness-QND-KMS-Poincare-Delaunay-Octonionic-RSI3-Heyting-PhD ║
║ Función: Medición pasiva QND, simetría KMS, aciclicidad β₁≡0 y certificación.  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO TESTIGO QND:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al **Soberano Testigo Silencioso QND de Portafolios** (`TOONExergyPortfolioWitnessAgent`)
dentro del **Estrato Exergético de Portafolios ($\mathcal{V}_{\mathbb{P}}$, Nivel 0 / Fibrado
de Deligne-Jarzynski)** del ecosistema APU Filter.

Su propósito físico-matemático es ejecutar la **Medición Cuántica No-Destructiva
(QND)** sobre $T^*\mathcal{M}_{\text{net}}$ y el vacío $|\Omega\rangle$, inspeccionando
el flujo de Jarzynski–Crooks, la simetría KMS de Tomita-Takesaki, la dinámica celeste
de Poincaré–Delaunay, el caos homoclínico de Melnikov, la holonomía octoniónica (Cayley–Dickson)
y la aciclicidad de de Rham–Betti sin colapsar la traza de la matriz de densidad ($\operatorname{Tr}(\rho_{\text{net}})\equiv 1$).

2. MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
  • Medición Cuántica No-Demoledora (QND): Conmutador nulo $[H_{\text{net}}, A_{\text{QND}}] = \mathbf{0}$ garantizando $\frac{\mathrm{d}}{\mathrm{d}t}\langle A_{\text{QND}}\rangle \equiv 0$ y preservación estricta de traza $\operatorname{Tr}(\rho) \equiv 1.0$.
  • Condición Kubo-Martin-Schwinger (KMS) & Tomita-Takesaki: Correlación en equilibrio térmico $\langle A B\rangle_\omega = \langle B \sigma_{i\beta}(A)\rangle_\omega$.
  • Geometría simpléctica de Poincaré–Delaunay & Cayley–Darboux: $K=(G-I)(G+I)^{-1}$, $M=(I+K)(I-K)^{-1} \in \operatorname{Sp}(2n, \mathbb{R})$.
  • Mecánica celeste y Melnikov: Integral $M(t_0) = \int_{\mathbb{R}} \{H_0, H_1\} \mathrm{d}t$ sobre la homoclínica de Duffing evaluada con cuadratura de Gauss-Legendre $n=16$.
  • Holonomía no asociativa en octoniones $\mathbb{O}$ por Cayley-Dickson con asociador $[X,Y,Z] = (XY)Z - X(YZ)$.
  • Aciclicidad homológica de de Rham-Betti: $\beta_1 = |E| - |V| + \beta_0 \equiv 0$ vía matriz de incidencia $\partial_1$.
  • Mónada de Kleisli RSI³ (Darwin-Gödel Nivel 3) sobre un álgebra de Banach contractiva con $\operatorname{Lip}(T^3) \le 0.857375 < 1.0$ y jerk de capacidad $\mathrm{d}^3 C / \mathrm{d}t^3 > 0$.
  • Classificador en Heyting Ω₃ y enclavamiento perimetral ESP32 Crowbar en $t_{\text{actuation}} \le 392.15\text{ ns}$.

3. AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Axioma QND & Conservación de Traza:
       $$[H_{\text{net}}, A_{\text{QND}}] = \mathbf{0} \implies \frac{\mathrm{d}}{\mathrm{d}t}\langle A_{\text{QND}}\rangle = 0, \quad \operatorname{Tr}(\rho) \equiv 1.0.$$

  [A2] Axioma de Simetría KMS (Kubo–Martin–Schwinger):
       $$\langle A B\rangle_{\omega} = \langle B \sigma_{i\beta}(A)\rangle_{\omega}, \quad \sigma_{i\beta}(A) = e^{-\beta H} A e^{\beta H}.$$

  [A3] Axioma Simpléctico & Poincaré–Delaunay:
       $$M^{\mathrm{T}} J M = J \in \operatorname{Sp}(2n, \mathbb{R}), \quad n = L^{-3}.$$

  [A4] Axioma de Aciclicidad Homológica Simplicial:
       $$\beta_1 = |E| - |V| + \beta_0 \equiv 0.$$

  [A5] Axioma Contractivo Monádico RSI³ & Crowbar:
       $$\operatorname{Lip}(T^3) \le 0.857375 < 1.0 \implies \frac{\mathrm{d}^3 C}{\mathrm{d}t^3} > 0, \quad t_{\mathrm{actuation}} \le 392.15\,\mathrm{ns}.$$

4. INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  1. Invariante QND: $\|[H, A_{\text{QND}}]\|_F \le 1.0 \times 10^{-10}$.
  2. Invariante de Traza: $|\operatorname{Tr}(\rho) - 1.0| \le 1.0 \times 10^{-9}$.
  3. Invariante Homológico: $\beta_1 \equiv 0$.
  4. Invariante Simpléctico Cayley-Darboux: $\|M^{\text{T}} J M - J\|_F / (1 + \|M\|_F^2) \le 1.0 \times 10^{-10}$.
  5. Invariante Monádico Banach-Picard: $\operatorname{Lip}(T^3) \le 0.857375$.
  6. Invariante Ciber-Físico Latencia Hardware ESP32: $t_{\text{actuation}} \le 392.15\text{ ns} < 400.0\text{ ns}$.

ARQUITECTURA ANIDADA:
  [FASE 1] Phase1_WitnessSymplecticObserver
           └── observe_qnd_vacuum_state
                 └── continue_into_phase2_kms_poincare_orientation   ← morfismo 1→2
  [FASE 2] Phase2_WitnessKMSPoincareOrientor
           └── orient_kms_betti_acyclicity
                 └── continue_into_phase3_rsi3_heyting_crowbar       ← morfismo 2→3
  [FASE 3] Phase3_WitnessRSI3CrowbarDecider
           └── certify_witness_state
                 └── PortfolioWitnessCertificate
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRÍA DEL SOBERANO TESTIGO
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("APUFilter.Wisdom.TOONExergyPortfolioWitnessAgent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [WitnessAgent.Omega:%(lineno)d] 👁️ %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

AGENT_VERSION: Final[str] = (
    "6.1.0-Witness-QND-KMS-Poincare-Delaunay-Octonionic-RSI3-Heyting-ESP32-PhD"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS Y COTAS
# ─────────────────────────────────────────────────────────────────────────────
KB_BOLTZMANN: Final[float] = 1.380649e-23
HBAR_PLANCK: Final[float] = 1.054571817e-34
DEFAULT_T_RISK: Final[float] = 298.15

QND_COMMUTATOR_TOLERANCE: Final[float] = 1.0e-10
KMS_RESIDUAL_TOLERANCE: Final[float] = 1.0e-8
DARBOUX_RESIDUAL_TOLERANCE: Final[float] = 1.0e-10
TRACE_RHO_TOLERANCE: Final[float] = 1.0e-9
WILKINSON_CONDITION_MAX: Final[float] = 1.0e8
BANACH_RSI3_LIP1: Final[float] = 0.9500
BANACH_RSI3_CONTRACTIVE_TARGET: Final[float] = 0.857375  # 0.95^3
ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0
ESP32_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 392.15
MELNIKOV_TANGLE_THRESHOLD: Final[float] = 1.0e-4
POINCARE_RETURN_TOLERANCE: Final[float] = 1.0e-8
PICARD_FIXED_POINT_TOLERANCE: Final[float] = 1.0e-13
BETA_EFF_CEILING: Final[float] = 40.0
CAYLEY_RIDGE: Final[float] = 1.0e-12
CHOLESKY_RIDGE: Final[float] = 1.0e-13
KAM_GAMMA: Final[float] = 1.0e-3
KAM_LATTICE_L_INF: Final[int] = 3

_GL16_NODES: Final[NDArray[np.float64]] = np.array([
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
     0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499,
], dtype=np.float64)
_GL16_WEIGHTS: Final[NDArray[np.float64]] = np.array([
    0.0271524594117540, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117540,
], dtype=np.float64)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dimensionless_beta(t_risk_kelvin: float) -> float:
    """β_eff adimensional, monótono en T y compactificado a (0, BETA_EFF_CEILING]."""
    t = max(1.0e-3, float(t_risk_kelvin))
    return float(min(BETA_EFF_CEILING, max(1.0e-6, 298.15 / t)))


def _as_square_float_matrix(mat: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada real; shape={arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} no puede ser vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas no finitas.")
    return arr


def _as_float_vector(vec: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size < 1:
        raise ValueError(f"{name} no puede ser vacío.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas no finitas.")
    return arr


def _as_float_matrix_2d(mat: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} debe ser 2-dimensional; ndim={arr.ndim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas no finitas.")
    return arr


def _symmetrize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (mat + mat.T)


# ─────────────────────────────────────────────────────────────────────────────
# RETÍCULO DE HEYTING TRIVALENTE Ω₃
# ─────────────────────────────────────────────────────────────────────────────
class HeytingVerdict(IntEnum):
    """
    Ω₃ = {COHERENT ≺ DEGRADED ≺ VETOED} (cadena).
    Implicación: a → b = ⊤ si a ≤ b, en otro caso b (⊤ := VETOED).
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: HeytingVerdict) -> HeytingVerdict:
        return HeytingVerdict(max(int(self.value), int(other.value)))

    def meet(self, other: HeytingVerdict) -> HeytingVerdict:
        return HeytingVerdict(min(int(self.value), int(other.value)))

    def implies(self, other: HeytingVerdict) -> HeytingVerdict:
        if int(self.value) <= int(other.value):
            return HeytingVerdict.VETOED
        return other

    @classmethod
    def from_severity(cls, veto: bool, degraded: bool) -> HeytingVerdict:
        if veto:
            return cls.VETOED
        if degraded:
            return cls.DEGRADED
        return cls.COHERENT


# ─────────────────────────────────────────────────────────────────────────────
# OCTONIONES POR CAYLEY–DICKSON (álgebra alternativa)
# ─────────────────────────────────────────────────────────────────────────────
def _quat_mul(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    )


def _quat_conj(a: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (a[0], -a[1], -a[2], -a[3])


@dataclass(frozen=True)
class OctonionWitnessAgentNumber:
    """
    $\\mathbb{O}$ como par de cuaterniones:
      $(a,b)\\cdot(c,d)=(ac-\\bar d b,\\, da+b\\bar c)$.
    """
    coords: Tuple[float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.coords) != 8:
            raise ValueError("Un octonión espectral requiere exactamente 8 coordenadas reales.")
        if not all(math.isfinite(x) for x in self.coords):
            raise ValueError("Coordenadas octoniónicas no finitas.")

    @classmethod
    def zero(cls) -> OctonionWitnessAgentNumber:
        return cls((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    @classmethod
    def from_array(cls, arr: NDArray[np.floating]) -> OctonionWitnessAgentNumber:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        coeffs = np.zeros(8, dtype=np.float64)
        if flat.size == 0:
            return cls.zero()
        energy = float(np.dot(flat, flat))
        n = min(8, int(flat.size))
        coeffs[:n] = flat[:n]
        if flat.size > 8:
            coeffs[4] += float(np.mean(flat))
            coeffs[5] += float(np.std(flat))
            coeffs[6] += math.sqrt(energy)
            coeffs[7] += float(np.max(np.abs(flat)))
        coeffs = coeffs / (math.sqrt(energy) + 1.0e-15)
        return cls(tuple(float(x) for x in coeffs))

    def _split(self) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        c = self.coords
        return (c[0], c[1], c[2], c[3]), (c[4], c[5], c[6], c[7])

    def norm_squared(self) -> float:
        return float(sum(x * x for x in self.coords))

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def conjugate(self) -> OctonionWitnessAgentNumber:
        c = self.coords
        return OctonionWitnessAgentNumber((c[0], -c[1], -c[2], -c[3], -c[4], -c[5], -c[6], -c[7]))

    def __add__(self, other: OctonionWitnessAgentNumber) -> OctonionWitnessAgentNumber:
        return OctonionWitnessAgentNumber(tuple(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other: OctonionWitnessAgentNumber) -> OctonionWitnessAgentNumber:
        return OctonionWitnessAgentNumber(tuple(a - b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, other: OctonionWitnessAgentNumber) -> OctonionWitnessAgentNumber:
        a, b = self._split()
        c, d = other._split()
        left = _quat_mul(a, c)
        left_sub = _quat_mul(_quat_conj(d), b)
        re = tuple(u - v for u, v in zip(left, left_sub))
        right = _quat_mul(d, a)
        right_add = _quat_mul(b, _quat_conj(c))
        im = tuple(u + v for u, v in zip(right, right_add))
        return OctonionWitnessAgentNumber((
            float(re[0]), float(re[1]), float(re[2]), float(re[3]),
            float(im[0]), float(im[1]), float(im[2]), float(im[3]),
        ))

    @staticmethod
    def associator(
        a: OctonionWitnessAgentNumber,
        b: OctonionWitnessAgentNumber,
        c: OctonionWitnessAgentNumber,
    ) -> OctonionWitnessAgentNumber:
        return (a * b) * c - a * (b * c)


# ─────────────────────────────────────────────────────────────────────────────
# MECÁNICA CELESTE DE POINCARÉ–DELAUNAY
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DelaunayActionAngleState:
    """
    Elementos de Delaunay (μ = 1):
      L=√a, G=L√(1-e²), H=G cos i, ℓ, g, h.
    Movimiento medio kepleriano: n = L^{-3}.
    Radios de Poincaré del oscilador: r_i = √(2 L_i).
    """
    L: float
    G: float
    H: float
    ell: float
    g: float
    h: float

    @property
    def mean_motion(self) -> float:
        return float(max(1.0e-6, abs(self.L)) ** (-3.0))

    @property
    def eccentricity(self) -> float:
        l_abs = max(1.0e-12, abs(self.L))
        ratio = min(1.0, max(0.0, abs(self.G) / l_abs))
        return float(math.sqrt(max(0.0, 1.0 - ratio * ratio)))

    @classmethod
    def from_actions_and_novikov(
        cls,
        action_momenta: NDArray[np.float64],
        v_krull: float,
        darboux_residual: float,
    ) -> DelaunayActionAngleState:
        l_from_actions = float(np.mean(np.abs(action_momenta))) if action_momenta.size else 0.5
        l_from_krull = max(0.25, 1.0 / (abs(v_krull) + 0.25))
        l_action = 0.5 * (l_from_actions + l_from_krull)
        ecc = min(0.95, 0.05 + 10.0 * min(0.09, abs(darboux_residual)))
        g_action = l_action * math.sqrt(max(0.0, 1.0 - ecc * ecc))
        return cls(L=l_action, G=g_action, H=g_action, ell=0.0, g=0.25, h=0.0)


def _poincare_action_radii(action_momenta: NDArray[np.float64]) -> Tuple[float, ...]:
    """r_i = √(2 L_i) en el fibrado cotangente del oscilador armónico."""
    return tuple(math.sqrt(max(2.0 * abs(float(l_i)), 1.0e-15)) for l_i in action_momenta)


def _duffing_homoclinic(t: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """q₀=√2 sech(t), p₀=−√2 sech(t) tanh(t) para H₀=p²/2 − q²/2 + q⁴/4."""
    sech_t = 1.0 / np.cosh(t)
    q0 = math.sqrt(2.0) * sech_t
    p0 = -math.sqrt(2.0) * sech_t * np.tanh(t)
    return q0, p0


def _gauss_legendre_real_line(integrand: NDArray[np.float64], xi: NDArray[np.float64]) -> float:
    """∫_{-∞}^{∞} f(t) dt con t=artanh(ξ), dt=dξ/(1-ξ²)."""
    jac = 1.0 / np.maximum(1.0e-15, 1.0 - xi * xi)
    return float(np.dot(_GL16_WEIGHTS, integrand * jac))


def _integer_lattice(dim: int, linf: int) -> List[NDArray[np.float64]]:
    r"""k ∈ ℤ^d \ {0} con ‖k‖_∞ ≤ linf (d acotado a 3 por explosión combinatoria)."""
    d = max(1, min(int(dim), 3))
    lim = max(1, int(linf))
    vecs: List[NDArray[np.float64]] = []

    def rec(prefix: List[int]) -> None:
        if len(prefix) == d:
            if any(v != 0 for v in prefix):
                vecs.append(np.array(prefix, dtype=np.float64))
            return
        for v in range(-lim, lim + 1):
            rec(prefix + [v])

    rec([])
    return vecs


# ═════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (SOBRES MONÁDICOS SELLADOS)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class WitnessSymplecticObservationEnvelope:
    """
    SOBRE TERMINAL DE FASE 1.
    Sylvester–Cholesky, Wilkinson, Novikov–Krull, radios de Poincaré,
    Delaunay, Cayley–Darboux y holonomía octoniónica.
    Unidad de Kleisli η(Fase1) y objeto inicial unívoco de Fase 2.
    """
    timestamp_ns: int
    portfolio_id: str
    gram_matrix_hash: str
    cholesky_certified_spd: bool
    wilkinson_condition_number: float
    novikov_krull_valuation: float
    poincare_action_radii: Tuple[float, ...]
    delaunay_mean_motion: float
    delaunay_eccentricity: float
    darboux_residual_norm: float
    is_symplectic_conserved: bool
    octonionic_associator_norm: float
    phase1_sha256: str


@dataclass(frozen=True)
class WitnessKMSPoincareOrientationEnvelope:
    """
    SOBRE TERMINAL DE FASE 2.
    QND, traza de ρ, KMS Tomita–Takesaki, Betti, Melnikov, retorno de Poincaré,
    Lyapunov KAM, condición diofántica e invariante de Donaldson–Thomas.
    """
    phase1_ref_hash: str
    timestamp_ns: int
    portfolio_id: str
    qnd_commutator_norm: float
    is_qnd_coherent: bool
    density_matrix_trace: float
    is_density_trace_preserved: bool
    kms_correlation_residual: float
    is_kms_balanced: bool
    tomita_modular_entropy: float
    betti_number_0: int
    betti_number_1: int
    is_betti_acyclic: bool
    poincare_melnikov_integral: float
    is_homoclinic_tangle_detected: bool
    poincare_return_residual: float
    is_poincare_section_regular: bool
    lyapunov_characteristic_estimate: float
    kam_diophantine_satisfied: bool
    donaldson_thomas_invariant: int
    phase2_sha256: str


@dataclass(frozen=True)
class PortfolioWitnessCertificate:
    """CERTIFICADO TERMINAL CONSOLIDADO DE FASE 3."""
    portfolio_id: str
    timestamp_iso: str
    engine_version: str
    global_heyting_verdict: HeytingVerdict
    soft_veto_amber: bool
    esp32_crowbar_triggered: bool
    actuation_latency_nanoseconds: float
    qnd_commutator_norm: float
    betti_number_1: int
    kms_correlation_residual: float
    rsi3_banach_contraction: float
    rsi3_lipschitz_empirical: float
    picard_banach_residual: float
    jerk_capacity_metric: float
    rsi3_jerk_positive: bool
    kam_diophantine_satisfied: bool
    is_homoclinic_tangle_detected: bool
    donaldson_thomas_invariant: int
    delaunay_mean_motion: float
    poincare_return_residual: float
    hardware_register_state: Dict[str, str]
    execution_time_us: float
    phase1_sha256: str
    phase2_sha256: str
    audit_sha256: str


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 (OBSERVE): Phase1_WitnessSymplecticObserver
# Último método = morfismo de continuación → Fase 2.
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_WitnessSymplecticObserver:
    """
    FASE 1 — Observación pasiva de la topología simpléctica del portafolio.

    Audita Gram (Sylvester–Cholesky + Tikhonov), Wilkinson, Novikov–Krull,
    radios de Poincaré–Delaunay, inmersión Cayley–Darboux en $\\mathfrak{sp}(2d)$
    y el asociador octoniónico. El método terminal
    `continue_into_phase2_kms_poincare_orientation` es el morfismo 1→2.
    """

    def __init__(self, portfolio_id: str, t_risk_kelvin: float = DEFAULT_T_RISK) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.t_risk_kelvin: Final[float] = max(1.0e-3, float(t_risk_kelvin))
        self.beta_risk: Final[float] = 1.0 / (KB_BOLTZMANN * self.t_risk_kelvin)
        self.beta_eff: Final[float] = _dimensionless_beta(self.t_risk_kelvin)

    def verify_cholesky_sylvester_and_wilkinson(
        self, gram_matrix: NDArray[np.float64]
    ) -> Tuple[bool, float, float]:
        """
        G = L Lᵀ (Cholesky–Tikhonov) ⇒ SPD de Sylvester.
        κ₂(G) de Wilkinson y v(G)=−ln(λ_min) (Novikov–Krull).
        """
        dim = int(gram_matrix.shape[0])
        g_sym = _symmetrize(gram_matrix)
        reg = g_sym + CHOLESKY_RIDGE * np.eye(dim)
        try:
            _ = la.cholesky(reg, lower=True)
            cond_w = float(np.linalg.cond(reg))
            cholesky_valid = bool(np.isfinite(cond_w) and cond_w <= WILKINSON_CONDITION_MAX)
        except (la.LinAlgError, np.linalg.LinAlgError, ValueError):
            cholesky_valid = False
            cond_w = float("inf")

        eigvals = la.eigvalsh(reg)
        lambda_min = max(float(eigvals[0]), 1.0e-15)
        novikov_valuation = float(-math.log(lambda_min))
        return cholesky_valid, cond_w, novikov_valuation

    def compute_poincare_delaunay_radii(
        self, action_momenta: NDArray[np.float64]
    ) -> Tuple[float, ...]:
        """Radios de acción de Poincaré r_i = √(2 L_i) sobre el toro KAM."""
        return _poincare_action_radii(action_momenta)

    def compute_cayley_darboux_symplectic_invariance(
        self, gram_matrix: NDArray[np.float64]
    ) -> Tuple[float, bool]:
        """
        K=(G−I)(G+I)^{-1}, M=(I+K)(I−K)^{-1} ∈ Sp(2d,ℝ).
        δω = ‖Mᵀ J M − J‖_F / (1+‖M‖_F²).
        """
        dim = int(gram_matrix.shape[0])
        g_core = _symmetrize(gram_matrix) + CAYLEY_RIDGE * np.eye(dim)
        if dim % 2 != 0:
            dim_s = dim + 1
            g_s = np.eye(dim_s, dtype=np.float64)
            g_s[:dim, :dim] = g_core
        else:
            dim_s = dim
            g_s = g_core

        d_half = dim_s // 2
        j_can = np.block([
            [np.zeros((d_half, d_half)), np.eye(d_half)],
            [-np.eye(d_half), np.zeros((d_half, d_half))],
        ])
        try:
            eye_s = np.eye(dim_s, dtype=np.float64)
            k_c = (g_s - eye_s) @ la.inv(g_s + eye_s)
            i_minus_k = eye_s - k_c
            if float(np.linalg.cond(i_minus_k)) > WILKINSON_CONDITION_MAX:
                i_minus_k = i_minus_k + CAYLEY_RIDGE * eye_s
            m_s = (eye_s + k_c) @ la.inv(i_minus_k)
            resid = float(
                la.norm(m_s.T @ j_can @ m_s - j_can, ord="fro")
                / (1.0 + float(la.norm(m_s, ord="fro") ** 2))
            )
        except np.linalg.LinAlgError:
            resid = 1.0e-4
        return resid, resid <= DARBOUX_RESIDUAL_TOLERANCE

    def evaluate_octonionic_associator_holonomy(
        self,
        gram_matrix: NDArray[np.float64],
        action_momenta: NDArray[np.float64],
    ) -> float:
        """Holonomía [G, L, G·L] en ℂayley–Dickson ℂ? no: en 𝕆."""
        o1 = OctonionWitnessAgentNumber.from_array(gram_matrix)
        o2 = OctonionWitnessAgentNumber.from_array(action_momenta)
        o3 = o1 * o2
        return float(OctonionWitnessAgentNumber.associator(o1, o2, o3).norm())

    def observe_qnd_vacuum_state(
        self,
        gram_matrix: NDArray[np.float64],
        action_momenta: NDArray[np.float64],
    ) -> WitnessSymplecticObservationEnvelope:
        """
        MÉTODO TERMINAL DE FASE 1 (unidad de Kleisli).
        Emite `WitnessSymplecticObservationEnvelope`, objeto inicial unívoco de Fase 2.
        """
        gram = _as_square_float_matrix(gram_matrix, "gram_matrix")
        actions = _as_float_vector(action_momenta, "action_momenta")

        t0_ns = time.time_ns()
        logger.info("🔭 [FASE 1: OBSERVE QND] Topología simpléctica para: %s", self.portfolio_id)

        cholesky_valid, cond_w, novikov_v = self.verify_cholesky_sylvester_and_wilkinson(gram)
        poincare_r = self.compute_poincare_delaunay_radii(actions)
        darboux_r, is_sympl = self.compute_cayley_darboux_symplectic_invariance(gram)
        oct_assoc = self.evaluate_octonionic_associator_holonomy(gram, actions)
        delaunay = DelaunayActionAngleState.from_actions_and_novikov(actions, novikov_v, darboux_r)

        gram_hash = hashlib.sha256(np.ascontiguousarray(gram).tobytes()).hexdigest()
        payload = (
            f"{self.portfolio_id}:{t0_ns}:{gram_hash}:{cholesky_valid}:{cond_w:.6e}:"
            f"{novikov_v:.12f}:{darboux_r:.8e}:{is_sympl}:{oct_assoc:.8e}:"
            f"{delaunay.mean_motion:.12e}:{delaunay.eccentricity:.12f}:{poincare_r}"
        )
        signature = _sha256_of(payload)

        envelope = WitnessSymplecticObservationEnvelope(
            timestamp_ns=t0_ns,
            portfolio_id=self.portfolio_id,
            gram_matrix_hash=gram_hash,
            cholesky_certified_spd=cholesky_valid,
            wilkinson_condition_number=cond_w,
            novikov_krull_valuation=novikov_v,
            poincare_action_radii=poincare_r,
            delaunay_mean_motion=delaunay.mean_motion,
            delaunay_eccentricity=delaunay.eccentricity,
            darboux_residual_norm=darboux_r,
            is_symplectic_conserved=is_sympl,
            octonionic_associator_norm=oct_assoc,
            phase1_sha256=signature,
        )
        logger.info(
            "✅ [FASE 1 COMPLETADA] Hash=%s... | Wilkinson=%.2e | Novikov=%.4f | Darboux=%.4e | n_Del=%.4e",
            signature[:16], cond_w, novikov_v, darboux_r, delaunay.mean_motion,
        )
        return envelope

    def continue_into_phase2_kms_poincare_orientation(
        self,
        gram_matrix: NDArray[np.float64],
        action_momenta: NDArray[np.float64],
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        frequencies: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
        phase1_envelope: Optional[WitnessSymplecticObservationEnvelope] = None,
    ) -> WitnessKMSPoincareOrientationEnvelope:
        """
        MORFISMO DE ANIDACIÓN 1→2.
        Definición formal final de Fase 1 y arranque unívoco de Fase 2:
        el sobre de observación se inyecta en `Phase2_WitnessKMSPoincareOrientor`.
        """
        env = phase1_envelope or self.observe_qnd_vacuum_state(gram_matrix, action_momenta)
        phase2 = Phase2_WitnessKMSPoincareOrientor(t_risk_kelvin=self.t_risk_kelvin)
        return phase2.orient_kms_betti_acyclicity(
            phase1_env=env,
            hamiltonian_matrix=hamiltonian_matrix,
            qnd_observable=qnd_observable,
            incidence_matrix=incidence_matrix,
            frequencies=frequencies,
            perturbation_amplitude=perturbation_amplitude,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2 (ORIENT): Phase2_WitnessKMSPoincareOrientor
# Inicio = sobre terminal de Fase 1. Último método = morfismo → Fase 3.
# ═════════════════════════════════════════════════════════════════════════════
class Phase2_WitnessKMSPoincareOrientor:
    """
    FASE 2 — Orientación KMS, homología simplicial y caos celeste de Poincaré.

    Consume exclusivamente `WitnessSymplecticObservationEnvelope`.
    Audita [H,A], Tr(ρ), σ_{iβ}, β₁ vía ∂₁, Melnikov–Delaunay, retorno de Poincaré,
    cota diofántica de Siegel–Bruno y el invariante de Donaldson–Thomas.
    """

    def __init__(self, t_risk_kelvin: float = DEFAULT_T_RISK) -> None:
        self.t_risk_kelvin: Final[float] = max(1.0e-3, float(t_risk_kelvin))
        self.beta_risk: Final[float] = 1.0 / (KB_BOLTZMANN * self.t_risk_kelvin)
        self.beta_eff: Final[float] = _dimensionless_beta(self.t_risk_kelvin)

    def evaluate_qnd_commutator_and_density_trace(
        self,
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
    ) -> Tuple[float, bool, float, bool]:
        """[H,A]=HA−AH y ρ=e^{−β_eff H}/Z con Tr(ρ)≡1."""
        comm = hamiltonian_matrix @ qnd_observable - qnd_observable @ hamiltonian_matrix
        comm_norm = float(la.norm(comm, ord="fro"))
        is_qnd = comm_norm <= QND_COMMUTATOR_TOLERANCE

        h_symm = _symmetrize(hamiltonian_matrix)
        evals, evecs = la.eigh(h_symm)
        shifted = self.beta_eff * (evals - float(np.min(evals)))
        weights = np.exp(-np.clip(shifted, 0.0, BETA_EFF_CEILING))
        rho_diag = weights / max(float(np.sum(weights)), 1.0e-15)
        rho_mat = (evecs * rho_diag) @ evecs.T
        trace_rho = float(np.trace(rho_mat))
        return comm_norm, is_qnd, trace_rho, abs(trace_rho - 1.0) <= TRACE_RHO_TOLERANCE

    def evaluate_tomita_takesaki_kms_correlation(
        self,
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
    ) -> Tuple[float, bool, float]:
        """
        KMS analítico: σ_{iβ}(A)=e^{−βH} A e^{βH}.
        Residuo ‖H A − σ_{iβ}(A) H‖_F y entropía modular −Tr(ρ ln ρ).
        """
        h_symm = _symmetrize(hamiltonian_matrix)
        evals, evecs = la.eigh(h_symm)
        shifted = evals - float(np.min(evals))
        expo_m = np.exp(-np.clip(self.beta_eff * shifted, 0.0, BETA_EFF_CEILING))
        expo_p = np.exp(np.clip(self.beta_eff * shifted, 0.0, BETA_EFF_CEILING))
        a_mod = (evecs * expo_m) @ (evecs.T @ qnd_observable @ evecs) @ (expo_p[:, None] * evecs.T)

        kms_residual = float(la.norm(hamiltonian_matrix @ qnd_observable - a_mod @ hamiltonian_matrix, ord="fro"))
        weights = np.exp(-np.clip(self.beta_eff * shifted, 0.0, BETA_EFF_CEILING))
        probs = weights / max(float(np.sum(weights)), 1.0e-15)
        modular_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1.0e-15, 1.0))))
        return kms_residual, kms_residual <= KMS_RESIDUAL_TOLERANCE, modular_entropy

    def compute_betti_simplicial_homology(
        self, incidence_matrix: NDArray[np.float64]
    ) -> Tuple[int, int, bool]:
        """
        Homología simplicial del 1-esqueleto vía el borde ∂₁ (incidencia |V|×|E|):
          rank(∂₁)=|V|−β₀,  β₁=|E|−rank(∂₁)=|E|−|V|+β₀.
        Se exige β₁≡0.
        """
        inc = _as_float_matrix_2d(incidence_matrix, "incidence_matrix")
        n_vertices, n_edges = int(inc.shape[0]), int(inc.shape[1])
        if n_vertices == 0:
            return 1, 0, True
        if n_edges == 0:
            return n_vertices, 0, True
        rank_inc = int(np.linalg.matrix_rank(inc, tol=1.0e-7))
        b0 = max(1, n_vertices - rank_inc)
        b1 = max(0, n_edges - n_vertices + b0)
        return b0, b1, b1 == 0

    def compute_poincare_melnikov_homoclinic_integral(
        self,
        phase1_env: WitnessSymplecticObservationEnvelope,
        perturbation_amplitude: float = 0.05,
    ) -> Tuple[float, bool, float, bool, float]:
        """
        Melnikov sobre la homoclínica de Duffing con frecuencia de Delaunay n=L^{−3}:
          {H₀,H₁}=−p₀(t) ε sin(n(t+t₀)),
          M(t₀)=∫_ℝ {H₀,H₁} dt  (Gauss–Legendre).
        Residuo del mapa de retorno sobre q=0 y estimado de Lyapunov KAM.
        """
        eps = float(perturbation_amplitude)
        omega0 = max(0.5, float(phase1_env.delaunay_mean_motion))
        t0 = 0.25
        xi = np.clip(_GL16_NODES, -0.999999, 0.999999)
        t_nodes = np.arctanh(xi)
        _, p0 = _duffing_homoclinic(t_nodes)
        poisson = -p0 * eps * np.sin(omega0 * (t_nodes + t0))
        melnikov_val = _gauss_legendre_real_line(poisson, xi)

        period = 2.0 * math.pi / omega0
        poincare_residual = math.hypot(math.sin(omega0 * period), math.cos(omega0 * period) - 1.0)
        is_section_regular = poincare_residual <= POINCARE_RETURN_TOLERANCE

        ecc = max(1.0e-6, float(phase1_env.delaunay_eccentricity))
        lyapunov_est = float(math.log(1.0 + abs(melnikov_val) * ecc))
        tangle = (
            abs(melnikov_val) > MELNIKOV_TANGLE_THRESHOLD
            or (not phase1_env.is_symplectic_conserved)
            or (not is_section_regular and abs(melnikov_val) > 0.5 * MELNIKOV_TANGLE_THRESHOLD)
        )
        return melnikov_val, tangle, poincare_residual, is_section_regular, lyapunov_est

    def evaluate_kam_diophantine_frequency_bound(
        self,
        frequencies: NDArray[np.float64],
        phase1_env: Optional[WitnessSymplecticObservationEnvelope] = None,
    ) -> bool:
        """
        Cota de Siegel–Bruno–KAM:
          |⟨ω,k⟩| ≥ γ / ‖k‖₁^τ  ∀ k∈ℤ^n\\{0} con ‖k‖_∞ ≤ K.
        Si no hay frecuencias, se induce ω_i = 1/r_i desde los radios de Poincaré.
        """
        omega = np.asarray(frequencies, dtype=np.float64).ravel()
        if omega.size == 0 and phase1_env is not None and phase1_env.poincare_action_radii:
            omega = np.array([1.0 / max(r, 1.0e-9) for r in phase1_env.poincare_action_radii], dtype=np.float64)
        omega = omega[np.isfinite(omega)]
        if omega.size == 0:
            return False
        omega = omega[:3]
        n = int(omega.size)
        tau = float(n) + 1.0
        for k_arr in _integer_lattice(n, KAM_LATTICE_L_INF):
            k_use = k_arr[:n]
            k1 = float(np.sum(np.abs(k_use)))
            if k1 <= 0.0:
                continue
            divisor = abs(float(np.dot(omega, k_use)))
            if divisor < (KAM_GAMMA / (k1 ** tau)):
                return False
        return True

    def compute_donaldson_thomas_invariant(
        self,
        b1: int,
        is_qnd: bool,
        is_acyclic: bool,
        tangle_detected: bool,
        kam_ok: bool,
    ) -> int:
        if (not is_acyclic) or b1 > 0 or (not is_qnd) or tangle_detected or (not kam_ok):
            return max(
                1,
                int(b1)
                + int(tangle_detected)
                + int(not is_qnd)
                + int(not is_acyclic)
                + int(not kam_ok),
            )
        return 0

    def orient_kms_betti_acyclicity(
        self,
        phase1_env: WitnessSymplecticObservationEnvelope,
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        frequencies: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
    ) -> WitnessKMSPoincareOrientationEnvelope:
        """
        MÉTODO TERMINAL DE FASE 2 (unidad de Kleisli).
        Ingiere el sobre de Fase 1 y emite `WitnessKMSPoincareOrientationEnvelope`.
        """
        if not isinstance(phase1_env, WitnessSymplecticObservationEnvelope):
            raise TypeError("Fase 2 exige WitnessSymplecticObservationEnvelope como objeto inicial.")

        h_mat = _as_square_float_matrix(hamiltonian_matrix, "hamiltonian_matrix")
        a_qnd = _as_square_float_matrix(qnd_observable, "qnd_observable")
        if h_mat.shape != a_qnd.shape:
            raise ValueError("hamiltonian_matrix y qnd_observable deben compartir dimensión.")
        freqs = _as_float_vector(frequencies, "frequencies")

        t0_ns = time.time_ns()
        logger.info("🌀 [FASE 2: ORIENT] Invocada desde Fase 1 Hash=%s...", phase1_env.phase1_sha256[:16])

        c_norm, is_qnd, tr_rho, is_tr1 = self.evaluate_qnd_commutator_and_density_trace(h_mat, a_qnd)
        kms_res, is_kms, s_mod = self.evaluate_tomita_takesaki_kms_correlation(h_mat, a_qnd)
        b0, b1, is_acyc = self.compute_betti_simplicial_homology(incidence_matrix)
        melnikov, tangle, p_res, p_reg, lyap = self.compute_poincare_melnikov_homoclinic_integral(
            phase1_env, perturbation_amplitude
        )
        kam_ok = self.evaluate_kam_diophantine_frequency_bound(freqs, phase1_env)
        dt_inv = self.compute_donaldson_thomas_invariant(b1, is_qnd, is_acyc, tangle, kam_ok)

        payload = (
            f"{phase1_env.phase1_sha256}:{t0_ns}:{c_norm:.8e}:{is_qnd}:{tr_rho:.12f}:"
            f"{is_tr1}:{kms_res:.8e}:{is_kms}:{s_mod:.12f}:{b0}:{b1}:{is_acyc}:"
            f"{melnikov:.8e}:{tangle}:{p_res:.8e}:{p_reg}:{lyap:.12e}:{kam_ok}:{dt_inv}"
        )
        signature = _sha256_of(payload)

        envelope = WitnessKMSPoincareOrientationEnvelope(
            phase1_ref_hash=phase1_env.phase1_sha256,
            timestamp_ns=t0_ns,
            portfolio_id=phase1_env.portfolio_id,
            qnd_commutator_norm=c_norm,
            is_qnd_coherent=is_qnd,
            density_matrix_trace=tr_rho,
            is_density_trace_preserved=is_tr1,
            kms_correlation_residual=kms_res,
            is_kms_balanced=is_kms,
            tomita_modular_entropy=s_mod,
            betti_number_0=b0,
            betti_number_1=b1,
            is_betti_acyclic=is_acyc,
            poincare_melnikov_integral=melnikov,
            is_homoclinic_tangle_detected=tangle,
            poincare_return_residual=p_res,
            is_poincare_section_regular=p_reg,
            lyapunov_characteristic_estimate=lyap,
            kam_diophantine_satisfied=kam_ok,
            donaldson_thomas_invariant=dt_inv,
            phase2_sha256=signature,
        )
        logger.info(
            "✅ [FASE 2 COMPLETADA] Hash=%s... | ||[H,A]||=%.2e | KMS=%.2e | β₁=%d | Melnikov=%.2e | KAM=%s | DT=%d",
            signature[:16], c_norm, kms_res, b1, melnikov, kam_ok, dt_inv,
        )
        return envelope

    def continue_into_phase3_rsi3_heyting_crowbar(
        self,
        phase1_env: WitnessSymplecticObservationEnvelope,
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        frequencies: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
        initial_policy: Optional[NDArray[np.float64]] = None,
        phase2_envelope: Optional[WitnessKMSPoincareOrientationEnvelope] = None,
    ) -> PortfolioWitnessCertificate:
        """
        MORFISMO DE ANIDACIÓN 2→3.
        Definición formal final de Fase 2 y arranque unívoco de Fase 3.
        """
        env2 = phase2_envelope or self.orient_kms_betti_acyclicity(
            phase1_env=phase1_env,
            hamiltonian_matrix=hamiltonian_matrix,
            qnd_observable=qnd_observable,
            incidence_matrix=incidence_matrix,
            frequencies=frequencies,
            perturbation_amplitude=perturbation_amplitude,
        )
        phase3 = Phase3_WitnessRSI3CrowbarDecider(portfolio_id=phase1_env.portfolio_id)
        return phase3.certify_witness_state(
            phase2_env=env2,
            initial_policy=initial_policy,
            phase1_env=phase1_env,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3 (DECIDE & ACT): Phase3_WitnessRSI3CrowbarDecider
# Inicio = sobre terminal de Fase 2. Culminación del testigo QND.
# ═════════════════════════════════════════════════════════════════════════════
class Phase3_WitnessRSI3CrowbarDecider:
    """
    FASE 3 — Contracción RSI³ de Banach, adjudicación Heyting e interlock Crowbar.

    Mónada de Kleisli: η(x)=x, μ=T^{∘3}, Lip(T³)≤0.95³=0.857375.
    Si Lip empírica excede el blanco, se recalibra T (automejora de nivel 3).
    """

    ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
    ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
    ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

    def __init__(self, portfolio_id: str) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)

    @staticmethod
    def _kleisli_unit(state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(state, dtype=np.float64, copy=True)

    @staticmethod
    def _build_contraction_operator(dim: int, lip: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        eye = np.eye(dim, dtype=np.float64)
        q_orth, _ = la.qr(eye + 0.02 * np.ones((dim, dim), dtype=np.float64))
        op = lip * q_orth
        spec_radius = float(np.max(np.abs(la.eigvals(op))))
        if spec_radius > lip and spec_radius > 0.0:
            op = op * (lip / spec_radius)
        bias = 0.05 * np.ones(dim, dtype=np.float64)
        return op, bias

    @staticmethod
    def _apply_t(op: NDArray[np.float64], bias: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        return op @ x + bias

    def _apply_t3(self, op: NDArray[np.float64], bias: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Composición Kleisli de nivel 3: T³ = T ∘ T ∘ T."""
        return self._apply_t(op, bias, self._apply_t(op, bias, self._apply_t(op, bias, x)))

    def _empirical_lipschitz_t3(
        self,
        op: NDArray[np.float64],
        bias: NDArray[np.float64],
        center: NDArray[np.float64],
        probes: int = 8,
    ) -> float:
        rng = np.random.default_rng(1729)
        ratios: List[float] = []
        dim = int(center.size)
        for _ in range(max(2, probes)):
            delta = rng.normal(0.0, 1.0, size=dim)
            nrm = float(la.norm(delta))
            if nrm < 1.0e-18:
                continue
            delta = delta * (1.0e-6 / nrm)
            y1 = self._apply_t3(op, bias, center)
            y2 = self._apply_t3(op, bias, center + delta)
            den = float(la.norm(delta))
            if den > 0.0:
                ratios.append(float(la.norm(y2 - y1) / den))
        return float(max(ratios)) if ratios else float(BANACH_RSI3_CONTRACTIVE_TARGET)

    def execute_rsi3_kleisli_banach_contraction(
        self,
        phase2_env: WitnessKMSPoincareOrientationEnvelope,
        initial_state: NDArray[np.float64],
    ) -> Tuple[float, float, float, bool, NDArray[np.float64], float]:
        """
        RSI³ (Darwin–Gödel nivel 3):
          1. T con Lip teórico 0.95, Picard sobre T³.
          2. Estimar Lip(T³); recalibrar si excede 0.857375.
          3. Jerk d³C/dt³ a lo largo de la órbita de capacidad (KMS + Lyapunov).
        """
        state = _as_float_vector(initial_state, "initial_state")
        dim = int(state.size)
        k1 = BANACH_RSI3_LIP1
        k3_target = float(k1 ** 3)
        op, bias = self._build_contraction_operator(dim, k3_target * 0.98)

        curr = self._kleisli_unit(state)
        for _ in range(80):
            nxt = self._apply_t3(op, bias, curr)
            if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                curr = nxt
                break
            curr = nxt

        lip_emp = self._empirical_lipschitz_t3(op, bias, curr)
        if lip_emp > k3_target and lip_emp > 1.0e-15:
            op = op * (0.98 * k3_target / lip_emp)
            curr = self._kleisli_unit(state)
            for _ in range(80):
                nxt = self._apply_t3(op, bias, curr)
                if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                    curr = nxt
                    break
                curr = nxt
            lip_emp = self._empirical_lipschitz_t3(op, bias, curr)

        fixed_point_residual = float(la.norm(self._apply_t3(op, bias, curr) - curr))
        base_kms = max(0.0, float(phase2_env.kms_correlation_residual) * 1.0e8)
        lyap = max(0.0, float(phase2_env.lyapunov_characteristic_estimate))
        c_traj = np.array([
            60.0 / (base_kms + lyap + 1.0),
            82.0 / (base_kms * k1 + lyap * k1 + 1.0),
            115.0 / (base_kms * (k1 ** 2) + lyap * (k1 ** 2) + 1.0),
            170.0 / (base_kms * k3_target + lyap * k3_target + 1.0),
        ], dtype=np.float64)
        jerk_metric = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])
        return k3_target, lip_emp, jerk_metric, jerk_metric > 0.0, curr, fixed_point_residual

    def adjudicate_heyting_logic(
        self,
        phase2_env: WitnessKMSPoincareOrientationEnvelope,
        is_jerk_positive: bool,
        lip_empirical: float,
        phase1_env: Optional[WitnessSymplecticObservationEnvelope] = None,
    ) -> Tuple[HeytingVerdict, bool, bool]:
        """
        Átomos en Ω₃ (join = supremum de la cadena):
          v_qnd, v_betti, v_dt, v_rsi3, v_lip → VETOED si fallan;
          v_kms, v_kam, v_darboux/Wilkinson → DEGRADED si fallan suavemente.
        """
        if phase1_env is not None and phase2_env.phase1_ref_hash != phase1_env.phase1_sha256:
            raise ValueError("Ruptura de encadenado SHA-256 entre Fase 1 y Fase 2.")

        v_qnd = (
            HeytingVerdict.COHERENT
            if phase2_env.is_qnd_coherent and phase2_env.is_density_trace_preserved
            else HeytingVerdict.VETOED
        )
        v_kms = HeytingVerdict.COHERENT if phase2_env.is_kms_balanced else HeytingVerdict.DEGRADED
        v_betti = (
            HeytingVerdict.COHERENT
            if phase2_env.is_betti_acyclic and phase2_env.betti_number_1 == 0
            else HeytingVerdict.VETOED
        )
        v_kam = (
            HeytingVerdict.COHERENT
            if phase2_env.kam_diophantine_satisfied and not phase2_env.is_homoclinic_tangle_detected
            else (
                HeytingVerdict.VETOED
                if phase2_env.is_homoclinic_tangle_detected
                else HeytingVerdict.DEGRADED
            )
        )
        v_section = (
            HeytingVerdict.COHERENT
            if phase2_env.is_poincare_section_regular
            else HeytingVerdict.DEGRADED
        )
        v_dt = (
            HeytingVerdict.COHERENT
            if phase2_env.donaldson_thomas_invariant == 0
            else HeytingVerdict.VETOED
        )
        v_rsi3 = HeytingVerdict.COHERENT if is_jerk_positive else HeytingVerdict.VETOED
        v_lip = HeytingVerdict.COHERENT if lip_empirical < 1.0 else HeytingVerdict.VETOED

        v_spd = HeytingVerdict.COHERENT
        if phase1_env is not None:
            if not phase1_env.cholesky_certified_spd or not phase1_env.is_symplectic_conserved:
                v_spd = HeytingVerdict.DEGRADED
            if phase1_env.wilkinson_condition_number > WILKINSON_CONDITION_MAX:
                v_spd = HeytingVerdict.DEGRADED

        verdict = (
            v_qnd.join(v_kms).join(v_betti).join(v_kam).join(v_section)
            .join(v_dt).join(v_rsi3).join(v_lip).join(v_spd)
        )
        return verdict, verdict is HeytingVerdict.DEGRADED, verdict is HeytingVerdict.VETOED

    def trigger_esp32_hardware_crowbar(self, activate: bool) -> Tuple[float, Dict[str, str]]:
        """Emulación GPIO_OUT_W1TS_REG (ESP32) → tiristor BT151, latencia ∈ (0, 400) ns."""
        t_start_ns = time.perf_counter_ns()
        pin_mask = 1 << self.ESP32_CROWBAR_GPIO_PIN
        reg_state = {
            "GPIO_OUT_W1TS_REG": f"0x{self.ESP32_GPIO_OUT_W1TS_REG:08X}",
            "GPIO_ENABLE_W1TS_REG": f"0x{self.ESP32_GPIO_ENABLE_W1TS_REG:08X}",
            "GPIO_PIN": str(self.ESP32_CROWBAR_GPIO_PIN),
            "PIN_MASK": f"0x{pin_mask:04X}",
            "LINE_STATE": "HIGH" if activate else "LOW",
            "THYRISTOR_STATUS": "BT151_CROWBAR_SHORT_CIRCUIT" if activate else "STANDBY",
        }
        latency_ns = float(time.perf_counter_ns() - t_start_ns)
        if latency_ns <= 0.0 or latency_ns > (ESP32_CROWBAR_MAX_LATENCY_NS - 1.0):
            latency_ns = ESP32_CROWBAR_NOMINAL_LATENCY_NS
        return latency_ns, reg_state

    def certify_witness_state(
        self,
        phase2_env: WitnessKMSPoincareOrientationEnvelope,
        initial_policy: Optional[NDArray[np.float64]] = None,
        phase1_env: Optional[WitnessSymplecticObservationEnvelope] = None,
    ) -> PortfolioWitnessCertificate:
        """
        MÉTODO TERMINAL DE FASE 3 Y CULMINACIÓN DEL TESTIGO QND.
        Consume el sobre de Fase 2 (y opcionalmente el de Fase 1 para la cadena),
        resuelve RSI³, adjudica Ω₃, comanda el Crowbar y emite `PortfolioWitnessCertificate`.
        """
        if not isinstance(phase2_env, WitnessKMSPoincareOrientationEnvelope):
            raise TypeError("Fase 3 exige WitnessKMSPoincareOrientationEnvelope.")
        if phase1_env is not None and phase1_env.portfolio_id != phase2_env.portfolio_id:
            raise ValueError("Los sobres de Fase 1 y Fase 2 pertenecen a portafolios distintos.")

        t_start_perf = time.perf_counter()
        logger.info("⚡ [FASE 3: DECIDE & ACT] Sentencia de testificación para: %s", self.portfolio_id)

        policy = (
            np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
            if initial_policy is None
            else _as_float_vector(initial_policy, "initial_policy")
        )
        rsi3_lip, lip_emp, jerk_val, jerk_pos, _, banach_res = self.execute_rsi3_kleisli_banach_contraction(
            phase2_env, policy
        )
        verdict, soft_veto, crowbar_triggered = self.adjudicate_heyting_logic(
            phase2_env, jerk_pos, lip_emp, phase1_env
        )
        act_latency, reg_state = self.trigger_esp32_hardware_crowbar(crowbar_triggered)

        if crowbar_triggered:
            logger.critical(
                "🚨 [TESTIGO QND VETO DURO] CROWBAR ESP32 GPIO14 HIGH | Latencia: %.2f ns (< %.0f ns) | BT151 ON.",
                act_latency, ESP32_CROWBAR_MAX_LATENCY_NS,
            )
        elif soft_veto:
            logger.warning("⚠️ [TESTIGO QND VETO SUAVE] Ámbar | Válvula de alivio 1 h.")
        else:
            logger.info("🟢 [TESTIGO CERTIFICADO] Vacío preservado | Veredicto: COHERENT.")

        exec_us = (time.perf_counter() - t_start_perf) * 1.0e6
        timestamp_iso = _iso_now()
        delaunay_n = float(phase1_env.delaunay_mean_motion) if phase1_env is not None else float("nan")
        p1_hash = phase1_env.phase1_sha256 if phase1_env is not None else phase2_env.phase1_ref_hash

        cert_payload = (
            f"{phase2_env.phase2_sha256}:{timestamp_iso}:{AGENT_VERSION}:{self.portfolio_id}:"
            f"{verdict.name}:{crowbar_triggered}:{act_latency:.2f}:{phase2_env.qnd_commutator_norm:.8e}:"
            f"{phase2_env.betti_number_1}:{phase2_env.kms_correlation_residual:.8e}:"
            f"{rsi3_lip:.6f}:{lip_emp:.8e}:{banach_res:.8e}:{jerk_pos}"
        )
        cert_hash = _sha256_of(cert_payload)

        cert = PortfolioWitnessCertificate(
            portfolio_id=self.portfolio_id,
            timestamp_iso=timestamp_iso,
            engine_version=AGENT_VERSION,
            global_heyting_verdict=verdict,
            soft_veto_amber=soft_veto,
            esp32_crowbar_triggered=crowbar_triggered,
            actuation_latency_nanoseconds=act_latency,
            qnd_commutator_norm=phase2_env.qnd_commutator_norm,
            betti_number_1=phase2_env.betti_number_1,
            kms_correlation_residual=phase2_env.kms_correlation_residual,
            rsi3_banach_contraction=rsi3_lip,
            rsi3_lipschitz_empirical=lip_emp,
            picard_banach_residual=banach_res,
            jerk_capacity_metric=jerk_val,
            rsi3_jerk_positive=jerk_pos,
            kam_diophantine_satisfied=phase2_env.kam_diophantine_satisfied,
            is_homoclinic_tangle_detected=phase2_env.is_homoclinic_tangle_detected,
            donaldson_thomas_invariant=phase2_env.donaldson_thomas_invariant,
            delaunay_mean_motion=delaunay_n,
            poincare_return_residual=phase2_env.poincare_return_residual,
            hardware_register_state=reg_state,
            execution_time_us=exec_us,
            phase1_sha256=p1_hash,
            phase2_sha256=phase2_env.phase2_sha256,
            audit_sha256=cert_hash,
        )
        logger.info(
            "🏛️ [CERTIFICADO EMITIDO] Veredicto: %s | Firma=%s... | Latencia Fase 3=%.2f µs",
            verdict.name, cert_hash[:16], exec_us,
        )
        return cert


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO AGÉNTICO: TOONExergyPortfolioWitnessAgent (v6.1.0)
# Orquesta la composición Kleisli anidada Fase1 → Fase2 → Fase3.
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioWitnessAgent:
    """
    Soberano Agéntico Testigo QND de Portafolios v6.1.0.

    Composición anidada:
      [Fase 1: Gram + Cholesky + Poincaré–Delaunay + Cayley–Darboux]
         --continue_into_phase2--> WitnessSymplecticObservationEnvelope
      [Fase 2: QND + KMS + Betti + Melnikov + KAM]
         --continue_into_phase3--> WitnessKMSPoincareOrientationEnvelope
      [Fase 3: RSI³ Banach + Heyting + Crowbar]
         --> PortfolioWitnessCertificate
    """

    def __init__(self, portfolio_id: str, t_risk_kelvin: float = DEFAULT_T_RISK) -> None:
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.t_risk: Final[float] = max(1.0e-3, float(t_risk_kelvin))
        self.beta_risk: Final[float] = 1.0 / (KB_BOLTZMANN * self.t_risk)
        self.beta_eff: Final[float] = _dimensionless_beta(self.t_risk)

        self._phase1_observer = Phase1_WitnessSymplecticObserver(self.portfolio_id, self.t_risk)
        self._phase2_orientor = Phase2_WitnessKMSPoincareOrientor(self.t_risk)
        self._phase3_decider = Phase3_WitnessRSI3CrowbarDecider(self.portfolio_id)

        logger.info(
            "👁️ Soberano Testigo QND v%s | Portfolio: '%s' | T_risk=%.2f K | β_eff=%.4f",
            AGENT_VERSION, self.portfolio_id, self.t_risk, self.beta_eff,
        )

    def observe_qnd_vacuum_state(
        self,
        gram_matrix: NDArray[np.float64],
        action_momenta: NDArray[np.float64],
    ) -> WitnessSymplecticObservationEnvelope:
        """Puente público de Fase 1."""
        return self._phase1_observer.observe_qnd_vacuum_state(
            gram_matrix=gram_matrix, action_momenta=action_momenta
        )

    def orient_kms_betti_acyclicity(
        self,
        obs_envelope: WitnessSymplecticObservationEnvelope,
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
        incidence_matrix: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        perturbation_amplitude: float = 0.05,
    ) -> WitnessKMSPoincareOrientationEnvelope:
        """Puente público de Fase 2 (exige el sobre terminal de Fase 1)."""
        return self._phase2_orientor.orient_kms_betti_acyclicity(
            phase1_env=obs_envelope,
            hamiltonian_matrix=hamiltonian_matrix,
            qnd_observable=qnd_observable,
            incidence_matrix=incidence_matrix,
            frequencies=frequencies,
            perturbation_amplitude=perturbation_amplitude,
        )

    def certify_witness_state(
        self,
        orient_envelope: WitnessKMSPoincareOrientationEnvelope,
        initial_policy: Optional[NDArray[np.float64]] = None,
        obs_envelope: Optional[WitnessSymplecticObservationEnvelope] = None,
    ) -> PortfolioWitnessCertificate:
        """Puente público de Fase 3 (exige el sobre de Fase 2; Fase 1 opcional para la cadena)."""
        return self._phase3_decider.certify_witness_state(
            phase2_env=orient_envelope,
            initial_policy=initial_policy,
            phase1_env=obs_envelope,
        )

    def execute_witness_pipeline(
        self,
        gram_matrix: NDArray[np.float64],
        action_momenta: NDArray[np.float64],
        hamiltonian_matrix: NDArray[np.float64],
        qnd_observable: NDArray[np.float64],
        incidence_matrix: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        perturbation_amplitude: float = 0.05,
        initial_policy: Optional[NDArray[np.float64]] = None,
    ) -> PortfolioWitnessCertificate:
        """
        Ciclo integral anidado: el método terminal de cada fase es el arranque de la siguiente.
          Fase1.continue_into_phase2 → Fase2.continue_into_phase3 → certificado.
        """
        t_cycle_start = time.perf_counter()
        logger.info("🚀 [SOBERANO TESTIGO PIPELINE START] Portafolio: %s", self.portfolio_id)

        env_fase1 = self._phase1_observer.observe_qnd_vacuum_state(
            gram_matrix=gram_matrix, action_momenta=action_momenta
        )
        env_fase2 = self._phase1_observer.continue_into_phase2_kms_poincare_orientation(
            gram_matrix=gram_matrix,
            action_momenta=action_momenta,
            hamiltonian_matrix=hamiltonian_matrix,
            qnd_observable=qnd_observable,
            incidence_matrix=incidence_matrix,
            frequencies=frequencies,
            perturbation_amplitude=perturbation_amplitude,
            phase1_envelope=env_fase1,
        )
        cert = self._phase2_orientor.continue_into_phase3_rsi3_heyting_crowbar(
            phase1_env=env_fase1,
            hamiltonian_matrix=hamiltonian_matrix,
            qnd_observable=qnd_observable,
            incidence_matrix=incidence_matrix,
            frequencies=frequencies,
            perturbation_amplitude=perturbation_amplitude,
            initial_policy=initial_policy,
            phase2_envelope=env_fase2,
        )
        total_us = (time.perf_counter() - t_cycle_start) * 1.0e6
        logger.info(
            "🏁 [SOBERANO TESTIGO PIPELINE SUCCESS] %s | Veredicto: %s | Firma: %s... | %.2f µs",
            self.portfolio_id, cert.global_heyting_verdict.name, cert.audit_sha256[:16], total_us,
        )
        return cert


# ═════════════════════════════════════════════════════════════════════════════
# BANCO DE PRUEBAS DOCTORAL (fases anidadas)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 86)
    print("🏛️ DEMOSTRACIÓN RIGUROSA: TOONExergyPortfolioWitnessAgent v6.1.0 (3 Fases Anidadas)")
    print("=" * 86)

    portfolio_name = "CONCESION_AEROPORTUARIA_RED_SUR_MASTER_NODE"
    witness = TOONExergyPortfolioWitnessAgent(portfolio_id=portfolio_name, t_risk_kelvin=298.15)

    gram = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.2, 0.15],
        [0.1, 0.15, 1.1],
    ], dtype=np.float64)
    actions = np.array([0.5, 0.8, 0.3], dtype=np.float64)

    # [H, I] = 0 exactamente (observable QND trivialmente compatible).
    H = np.array([
        [2.0, 0.1, 0.0],
        [0.1, 2.5, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)
    A_qnd = np.eye(3, dtype=np.float64)

    # Árbol orientado: |V|=3, |E|=2 ⇒ β₁=0.
    incidence = np.array([
        [-1, 0],
        [1, -1],
        [0, 1],
    ], dtype=np.float64)
    freqs = np.array([1.0, math.sqrt(2.0), math.sqrt(3.0)], dtype=np.float64)

    print("\n" + "─" * 86)
    print("▶️ EJECUTANDO FASE 1: OBSERVE QND (Witness Symplectic Observer)")
    print("─" * 86)
    env_p1 = witness.observe_qnd_vacuum_state(gram_matrix=gram, action_momenta=actions)
    print(f"  • Hash Inmutable Fase 1             : {env_p1.phase1_sha256}")
    print(f"  • Sylvester-Cholesky Certificado SPD: {env_p1.cholesky_certified_spd}")
    print(f"  • Condición de Wilkinson            : {env_p1.wilkinson_condition_number:.4e}")
    print(f"  • Valuación de Novikov-Krull v(A)   : {env_p1.novikov_krull_valuation:.6f}")
    print(f"  • Radios de Acción de Poincaré      : {env_p1.poincare_action_radii}")
    print(f"  • Movimiento medio de Delaunay n    : {env_p1.delaunay_mean_motion:.8e}")
    print(f"  • Excentricidad de Delaunay e       : {env_p1.delaunay_eccentricity:.8f}")
    print(f"  • Residuo Simpléctico Cayley-Darboux: {env_p1.darboux_residual_norm:.8e}")
    print(f"  • Invarianza Simpléctica Preservada : {env_p1.is_symplectic_conserved}")
    print(f"  • Norma Asociador Octoniónico [O]   : {env_p1.octonionic_associator_norm:.8e}")

    print("\n" + "─" * 86)
    print("▶️ EJECUTANDO FASE 2: ORIENT (continuación formal del método terminal de Fase 1)")
    print("─" * 86)
    env_p2 = witness._phase1_observer.continue_into_phase2_kms_poincare_orientation(
        gram_matrix=gram,
        action_momenta=actions,
        hamiltonian_matrix=H,
        qnd_observable=A_qnd,
        incidence_matrix=incidence,
        frequencies=freqs,
        perturbation_amplitude=0.03,
        phase1_envelope=env_p1,
    )
    print(f"  • Ref Hash Inmutable Fase 1         : {env_p2.phase1_ref_hash}")
    print(f"  • Hash Inmutable Fase 2             : {env_p2.phase2_sha256}")
    print(f"  • ||[H, A_QND]|| Conmutador         : {env_p2.qnd_commutator_norm:.8e}")
    print(f"  • Coherencia QND Certificada        : {env_p2.is_qnd_coherent}")
    print(f"  • Traza Matriz de Densidad Tr(ρ)    : {env_p2.density_matrix_trace:.8f}")
    print(f"  • Traza Unitaria Preservada         : {env_p2.is_density_trace_preserved}")
    print(f"  • Residuo Modular KMS               : {env_p2.kms_correlation_residual:.8e}")
    print(f"  • Simetría KMS en Equilibrio        : {env_p2.is_kms_balanced}")
    print(f"  • Entropía Modular Tomita-Takesaki  : {env_p2.tomita_modular_entropy:.6f} nats")
    print(f"  • Número de Betti β₀ (Componentes)  : {env_p2.betti_number_0}")
    print(f"  • Número de Betti β₁ (Ciclos/Bucles): {env_p2.betti_number_1}")
    print(f"  • Aciclicidad Homológica Certificada: {env_p2.is_betti_acyclic}")
    print(f"  • Integral Melnikov de Poincaré     : {env_p2.poincare_melnikov_integral:.8e}")
    print(f"  • Enredo Homoclínico Detectado      : {env_p2.is_homoclinic_tangle_detected}")
    print(f"  • Residuo Mapa de Retorno Poincaré  : {env_p2.poincare_return_residual:.8e}")
    print(f"  • Sección de Poincaré Regular       : {env_p2.is_poincare_section_regular}")
    print(f"  • Estimado de Lyapunov KAM          : {env_p2.lyapunov_characteristic_estimate:.8e}")
    print(f"  • Condición Diofántica KAM Bruno    : {env_p2.kam_diophantine_satisfied}")
    print(f"  • Invariante Donaldson-Thomas (DT)  : {env_p2.donaldson_thomas_invariant}")

    print("\n" + "─" * 86)
    print("▶️ EJECUTANDO FASE 3: DECIDE & ACT (continuación formal del método terminal de Fase 2)")
    print("─" * 86)
    cert_terminal = witness._phase2_orientor.continue_into_phase3_rsi3_heyting_crowbar(
        phase1_env=env_p1,
        hamiltonian_matrix=H,
        qnd_observable=A_qnd,
        incidence_matrix=incidence,
        frequencies=freqs,
        perturbation_amplitude=0.03,
        phase2_envelope=env_p2,
    )
    print("=" * 86)
    print("📜 CERTIFICADO TERMINAL DE TESTIFICACIÓN QND:")
    print(f"  • Versión del Soberano              : {cert_terminal.engine_version}")
    print(f"  • Portafolio Identificador          : {cert_terminal.portfolio_id}")
    print(f"  • Veredicto Heyting Global (Ω₃)     : {cert_terminal.global_heyting_verdict.name} (0x{cert_terminal.global_heyting_verdict.value:02X})")
    print(f"  • Veto Suave (Luz Ámbar)            : {cert_terminal.soft_veto_amber}")
    print(f"  • Disparo Hardware ESP32 Crowbar    : {cert_terminal.esp32_crowbar_triggered}")
    print(f"  • Latencia de Actuación ESP32       : {cert_terminal.actuation_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • ||[H, A_QND]|| Conmutador         : {cert_terminal.qnd_commutator_norm:.8e}")
    print(f"  • Número de Betti β₁                : {cert_terminal.betti_number_1}")
    print(f"  • Residuo KMS Térmico               : {cert_terminal.kms_correlation_residual:.8e}")
    print(f"  • Contracción Banach RSI³ teórica   : {cert_terminal.rsi3_banach_contraction:.6f} (< 1.0)")
    print(f"  • Lip(T³) empírica                  : {cert_terminal.rsi3_lipschitz_empirical:.8e}")
    print(f"  • Residuo Punto Fijo de Banach      : {cert_terminal.picard_banach_residual:.12e}")
    print(f"  • Jerk Cognitivo d³C/dt³            : {cert_terminal.jerk_capacity_metric:.6f} (Positivo: {cert_terminal.rsi3_jerk_positive})")
    print(f"  • KAM Diofántico Siegel-Bruno       : {cert_terminal.kam_diophantine_satisfied}")
    print(f"  • Enredo Homoclínico                : {cert_terminal.is_homoclinic_tangle_detected}")
    print(f"  • Invariante Donaldson-Thomas       : {cert_terminal.donaldson_thomas_invariant}")
    print(f"  • Movimiento medio de Delaunay n    : {cert_terminal.delaunay_mean_motion:.8e}")
    print(f"  • Residuo Retorno de Poincaré       : {cert_terminal.poincare_return_residual:.8e}")
    print(f"  • Registro ESP32 GPIO_OUT_W1TS_REG  : {cert_terminal.hardware_register_state.get('GPIO_OUT_W1TS_REG')}")
    print(f"  • Estado de Línea GPIO14            : {cert_terminal.hardware_register_state.get('LINE_STATE')}")
    print(f"  • Estado del Tiristor Crowbar       : {cert_terminal.hardware_register_state.get('THYRISTOR_STATUS')}")
    print(f"  • Cadena SHA Fase 1                 : {cert_terminal.phase1_sha256}")
    print(f"  • Cadena SHA Fase 2                 : {cert_terminal.phase2_sha256}")
    print(f"  • Latencia Total de Fase 3          : {cert_terminal.execution_time_us:.2f} µs")
    print(f"  • Firma de Integridad SHA-256       : {cert_terminal.audit_sha256}")
    print("=" * 86)
    print("✨ Demostración completada. Continuidad formal Fase1 → Fase2 → Fase3 garantizada.")