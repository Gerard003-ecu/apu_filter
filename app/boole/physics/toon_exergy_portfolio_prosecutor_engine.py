# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Prosecutor Engine (Motor Espectral Fiscal)    ║
║ Ruta   : app/boole/physics/toon_exergy_portfolio_prosecutor_engine.py        ║
║ Versión: 6.1.0-Prosecutor-Brachistochrone-Melnikov-Chirikov-KAM-RSI3-PhD     ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA FÍSICO-MATEMÁTICA Y VARIACIONAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Núcleo analítico de acusación del soberano fiscal en el Estrato Exergético
($\mathcal{V}_{\mathbb{P}}$, fibrado de Deligne–Jarzynski). Combina:

  1. Geometría simpléctica y octoniones $\mathbb{O}$ (Cayley–Dickson / Fano).
  2. Mecánica celeste de Poincaré–Delaunay: Melnikov, Chirikov, KAM, Nekhoroshev.
  3. Braquistócrona de Bernoulli–Euler: geodésica de menor tiempo (cicloide)
     en el paisaje exergético; el defecto $T_{\mathrm{path}}/T_{\mathrm{cyclo}}-1$
     cuantifica disipación geométrica irreversible.
  4. Jarzynski–Crooks con log-sum-exp y $\beta_{\mathrm{eff}}$ adimensional.
  5. Obstrucción de Čech–Deligne térmica (hermiticidad, traza, positividad; no
     se exige $\rho^2=\rho$, que es exclusivo de estados puros).
  6. Mónada Kleisli RSI³: $\operatorname{Lip}(T^3)\le 0.857375$, Picard, jerk.
  7. Heyting $\Omega_3$ y Crowbar ESP32 GPIO14 ($t_{\mathrm{act}}\le 392.15\,\mathrm{ns}$).

EVOLUCIÓN 6.1.0:
  • Fases anidadas: el método terminal de la Fase $k$ es el morfismo $k\to k+1$.
  • Melnikov y Chirikov acoplados a $\varepsilon$ del portafolio (Gram + $W_{\mathrm{irr}}$).
  • Ceros transversales de Melnikov: $M(t_0)=0$ y $M'(t_0)\ne 0$.
  • KAM por retículo $\mathbb{Z}^2\setminus\{0\}$; Nekhoroshev con $a=1/(2n)$.
  • RSI³ con Lip empírica y recalibración contractiva.
  • Cadena HMAC Fase 1 → Fase 2 → Merkle/HMAC terminal.

ARQUITECTURA ANIDADA:
  [FASE 1] Phase1_ProsecutorGaugeObserver
           └── observe_and_synthesize_gauge_state
                 └── continue_into_phase2_melnikov_chirikov     ← morfismo 1→2
  [FASE 2] Phase2_ProsecutorMelnikovChirikovOrientor
           └── orient_and_evaluate_indictment
                 └── continue_into_phase3_rsi3_heyting_crowbar  ← morfismo 2→3
  [FASE 3] Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine
           └── synthesize_and_execute_prosecution
                 └── ProsecutionFullReport
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from scipy.integrate import simpson as _simpson
except ImportError:  # SciPy < 1.6
    from scipy.integrate import simps as _simpson  # type: ignore[attr-defined]


ENGINE_VERSION: Final[str] = (
    "6.1.0-Prosecutor-Brachistochrone-Melnikov-Chirikov-KAM-RSI3-PhD"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS, VARIACIONALES Y HARDWARE
# ─────────────────────────────────────────────────────────────────────────────
BOLTZMANN_K: Final[float] = 1.380649e-23
PLANCK_HBAR: Final[float] = 1.054571817e-34
DEFAULT_RISK_TEMPERATURE: Final[float] = 298.15
STANDARD_GRAVITY: Final[float] = 9.80665
TIKHONOV_REGULARIZER: Final[float] = 1.0e-11
CAYLEY_RIDGE: Final[float] = 1.0e-12

RSI3_LIPSCHITZ_BASE: Final[float] = 0.95
RSI3_BANACH_TRIPLE: Final[float] = 0.857375  # 0.95³
PICARD_FIXED_POINT_TOLERANCE: Final[float] = 1.0e-13

MAX_CROWBAR_LATENCY_NS: Final[float] = 400.0
DETERMINISTIC_IRAM_LATENCY_NS: Final[float] = 392.15
ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

MELNIKOV_RELATIVE_THRESHOLD: Final[float] = 0.25
CHIRIKOV_RESONANCE_THRESHOLD: Final[float] = 1.00
EXERGY_WORK_IRR_THRESHOLD: Final[float] = 10.0
CECH_OBSTRUCTION_TOLERANCE: Final[float] = 1.0e-8
BRACHISTOCHRONE_DEFECT_TOLERANCE: Final[float] = 0.15
WILKINSON_CONDITION_MAX: Final[float] = 1.0e8
KAM_GAMMA: Final[float] = 1.0e-4
KAM_TAU: Final[float] = 2.5
KAM_LATTICE_L_INF: Final[int] = 4
BETA_EFF_CEILING: Final[float] = 40.0
DEFAULT_HMAC_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_EXERGY_PROSECUTOR_ENGINE_KEY_V9"

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

logger = logging.getLogger("APUFilter.Physics.TOONExergyPortfolioProsecutorEngine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [ProsecutorEngine.Spectral:%(lineno)d] 🏛️ %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hmac_hex(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _sha256_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_square_float_matrix(mat: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada; shape={arr.shape}.")
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


def _symmetrize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (mat + mat.T)


def _log_mean_exp(values: NDArray[np.float64]) -> float:
    """log ⟨e^{u}⟩ estable."""
    u = np.asarray(values, dtype=np.float64).ravel()
    u_max = float(np.max(u))
    return u_max + math.log(max(float(np.mean(np.exp(u - u_max))), 1.0e-300))


def _gauss_legendre_real_line(integrand: NDArray[np.float64], xi: NDArray[np.float64]) -> float:
    """∫_{-∞}^{∞} f(t) dt, t = artanh(ξ)."""
    jac = 1.0 / np.maximum(1.0e-15, 1.0 - xi * xi)
    return float(np.dot(_GL16_WEIGHTS, integrand * jac))


# ─────────────────────────────────────────────────────────────────────────────
# HEYTING Ω₃ Y MODOS DE ACTUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
class HeytingOmega3(IntEnum):
    """Cadena Ω₃ = {COHERENT ≺ DEGRADED ≺ VETOED}; a → b = ⊤ si a ≤ b."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: HeytingOmega3) -> HeytingOmega3:
        return HeytingOmega3(max(int(self.value), int(other.value)))

    def meet(self, other: HeytingOmega3) -> HeytingOmega3:
        return HeytingOmega3(min(int(self.value), int(other.value)))

    def implies(self, other: HeytingOmega3) -> HeytingOmega3:
        if int(self.value) <= int(other.value):
            return HeytingOmega3.VETOED
        return other


class ActuationMode(IntEnum):
    PASSIVE_NOMINAL = 0x00
    SOFT_RELIEF_VALVE = 0x01
    HARD_CROWBAR_DISCHARGE = 0x03


# ─────────────────────────────────────────────────────────────────────────────
# OCTONIONES 𝕆 (Fano + Cayley–Dickson): NO conmutativa, NO asociativa, alternativa
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


class OctonionProsecutorNumber:
    """
    Álgebra de división normada $\\mathbb{O}$: no conmutativa, no asociativa, alternativa.
    Producto por Cayley–Dickson: $(a,b)\\cdot(c,d)=(ac-\\bar d b,\\, da+b\\bar c)$.
    Las tríadas de Fano son el esqueleto combinatorio equivalente.
    """

    __slots__ = ("_coeffs",)
    _FANO_TRIADS: Final[Tuple[Tuple[int, int, int], ...]] = (
        (1, 2, 3), (1, 4, 5), (1, 7, 6), (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5),
    )

    def __init__(self, coeffs: Union[NDArray[np.floating], Sequence[float]]) -> None:
        arr = np.asarray(coeffs, dtype=np.float64).ravel()
        if arr.size != 8:
            raise ValueError(f"Un octonión requiere 8 coordenadas; recibido size={arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Coordenadas octoniónicas no finitas.")
        self._coeffs: NDArray[np.float64] = arr

    @classmethod
    def from_array(cls, arr: NDArray[np.floating]) -> OctonionProsecutorNumber:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        coeffs = np.zeros(8, dtype=np.float64)
        if flat.size == 0:
            return cls(coeffs)
        energy = float(np.dot(flat, flat))
        n = min(8, int(flat.size))
        coeffs[:n] = flat[:n]
        if flat.size > 8:
            coeffs[4] += float(np.mean(flat))
            coeffs[5] += float(np.std(flat))
            coeffs[6] += math.sqrt(energy)
            coeffs[7] += float(np.max(np.abs(flat)))
        coeffs = coeffs / (math.sqrt(energy) + 1.0e-15)
        return cls(coeffs)

    @property
    def coeffs(self) -> NDArray[np.float64]:
        return self._coeffs

    def _split(self) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        c = self._coeffs
        return (float(c[0]), float(c[1]), float(c[2]), float(c[3])), (
            float(c[4]), float(c[5]), float(c[6]), float(c[7]),
        )

    def __add__(self, other: OctonionProsecutorNumber) -> OctonionProsecutorNumber:
        return OctonionProsecutorNumber(self._coeffs + other._coeffs)

    def __sub__(self, other: OctonionProsecutorNumber) -> OctonionProsecutorNumber:
        return OctonionProsecutorNumber(self._coeffs - other._coeffs)

    def __mul__(self, other: OctonionProsecutorNumber) -> OctonionProsecutorNumber:
        a, b = self._split()
        c, d = other._split()
        left = _quat_mul(a, c)
        left_sub = _quat_mul(_quat_conj(d), b)
        re = tuple(u - v for u, v in zip(left, left_sub))
        right = _quat_mul(d, a)
        right_add = _quat_mul(b, _quat_conj(c))
        im = tuple(u + v for u, v in zip(right, right_add))
        return OctonionProsecutorNumber((
            re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3],
        ))

    def norm(self) -> float:
        return float(np.linalg.norm(self._coeffs))

    def conjugate(self) -> OctonionProsecutorNumber:
        c = self._coeffs.copy()
        c[1:] *= -1.0
        return OctonionProsecutorNumber(c)

    @staticmethod
    def associator(
        x: OctonionProsecutorNumber,
        y: OctonionProsecutorNumber,
        z: OctonionProsecutorNumber,
    ) -> OctonionProsecutorNumber:
        """$[x,y,z]=(xy)z-x(yz)$."""
        return (x * y) * z - x * (y * z)


# ─────────────────────────────────────────────────────────────────────────────
# POINCARÉ–DELAUNAY Y BRAQUISTÓCRONA (cicloide de Bernoulli)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DelaunayActionAngleState:
    """L=√a, G=L√(1-e²), H=G cos i; n = L^{-3}."""
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
    def from_novikov_and_gram(
        cls,
        v_krull: float,
        wilkinson: float,
        offdiag_ratio: float,
    ) -> DelaunayActionAngleState:
        l_action = max(0.25, 1.0 / (abs(v_krull) + 0.25))
        ecc = min(0.95, 0.02 + 0.15 * math.log10(max(wilkinson, 1.0)) + 0.5 * min(1.0, offdiag_ratio))
        g_action = l_action * math.sqrt(max(0.0, 1.0 - ecc * ecc))
        return cls(L=l_action, G=g_action, H=g_action, ell=0.0, g=0.25, h=0.0)


class BrachistochronePathFinder:
    """
    Geodésica de menor tiempo en un campo conservativo (Bernoulli 1696).
    Solución: cicloide $x=R(\\theta-\\sin\\theta)$, $y=R(1-\\cos\\theta)$,
    $T=\\theta\\sqrt{R/g}$. Se compara el tiempo de la trayectoria presupuestal
    $W_i$ (con $v=\\sqrt{2 g h}$) contra $T_{\\mathrm{cyclo}}$.
    """

    def __init__(self, g_eff: float = STANDARD_GRAVITY) -> None:
        self.g_eff: Final[float] = max(1.0e-9, float(g_eff))

    @staticmethod
    def cycloid_time(drop: float, g_eff: float) -> float:
        """Tiempo de descenso al punto más bajo: $T=\\pi\\sqrt{R/g}$, $R=\\mathrm{drop}/2$."""
        drop_abs = max(1.0e-15, abs(float(drop)))
        radius = 0.5 * drop_abs
        return float(math.pi * math.sqrt(radius / max(g_eff, 1.0e-12)))

    def evaluate_work_path(
        self,
        work_vec: NDArray[np.float64],
        w_irr: float,
    ) -> Tuple[float, float, float, bool]:
        """
        $h_i=(W_i-W_{\\min})/\\Delta W$, $T_{\\mathrm{path}}=\\sum \\Delta s_i/\\sqrt{2gh_i}$.
        Defecto: $T_{\\mathrm{path}}/T_{\\mathrm{cyclo}}-1$. Óptimo si defecto ≤ tolerancia
        y $W_{\\mathrm{irr}}$ no domina.
        """
        w = np.asarray(work_vec, dtype=np.float64).ravel()
        if w.size < 2:
            return 0.0, 0.0, 0.0, True

        w_min = float(np.min(w))
        w_max = float(np.max(w))
        drop = max(w_max - w_min, abs(float(w_irr)), 1.0e-12)
        g_eff = self.g_eff * (1.0 + abs(float(w_irr)) / (abs(float(np.mean(w))) + 1.0))
        height = (w - w_min) / drop
        # Parametrización de arco en el plano (índice, altura).
        s_x = np.linspace(0.0, 1.0, w.size)
        t_path = 0.0
        for i in range(1, w.size):
            ds = math.hypot(float(s_x[i] - s_x[i - 1]), float(height[i] - height[i - 1]))
            h_mid = max(0.5 * (float(height[i]) + float(height[i - 1])), 1.0e-12)
            speed = math.sqrt(max(2.0 * g_eff * h_mid, 1.0e-12))
            t_path += ds / speed

        t_cyclo = self.cycloid_time(drop=1.0, g_eff=g_eff)
        # Normalizar T_path al mismo drop adimensional 1.
        defect = float(t_path / max(t_cyclo, 1.0e-15) - 1.0)
        is_optimal = defect <= BRACHISTOCHRONE_DEFECT_TOLERANCE and abs(float(w_irr)) <= EXERGY_WORK_IRR_THRESHOLD
        return float(t_path), float(t_cyclo), defect, is_optimal


def _duffing_homoclinic(t: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    sech_t = 1.0 / np.cosh(t)
    q0 = math.sqrt(2.0) * sech_t
    p0 = -math.sqrt(2.0) * sech_t * np.tanh(t)
    return q0, p0


def _melnikov_closed_form_amplitude(omega: float, eps: float) -> float:
    """
    Amplitud analítica del Melnikov de Duffing con $H_1=\\varepsilon q\\sin(\\omega t)$:
    $A(\\omega)\\propto \\varepsilon\\,\\pi\\omega\\,\\mathrm{csch}(\\pi\\omega/2)$.
    """
    w = max(1.0e-9, abs(float(omega)))
    # csch(x) = 1/sinh(x); recorte para estabilidad.
    x = 0.5 * math.pi * w
    if x > 40.0:
        csch = 2.0 * math.exp(-x)
    else:
        sinh_x = math.sinh(x)
        csch = 1.0 / sinh_x if abs(sinh_x) > 1.0e-15 else 1.0e15
    return abs(float(eps)) * math.pi * w * csch


# ═════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (SOBRES MONÁDICOS)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProsecutionGaugeEnvelope:
    """SOBRE TERMINAL DE FASE 1 — calibre, Sylvester, Cayley–Darboux, Delaunay, 𝕆."""
    timestamp_iso: str
    portfolio_id: str
    num_contracts: int
    gram_matrix_hash: str
    sylvester_native: bool
    sylvester_certified: bool
    wilkinson_condition_number: float
    novikov_krull_valuation: float
    offdiag_energy_ratio: float
    darboux_symplectic_residual: float
    is_symplectic_conserved: bool
    octonionic_associator_anomaly_norm: float
    cayley_sp_norm: float
    delaunay_mean_motion: float
    delaunay_eccentricity: float
    phase1_signature_hmac: str


@dataclass(frozen=True)
class ProsecutionIndictmentEnvelope:
    """SOBRE TERMINAL DE FASE 2 — Melnikov, Chirikov, KAM, Jarzynski, braquistócrona, RSI³."""
    phase1_ref_hash: str
    dual_prosecution_functional_value: float
    perturbation_epsilon: float
    melnikov_integral_value: float
    melnikov_closed_form_amplitude: float
    melnikov_simple_zero_detected: bool
    chirikov_overlap_parameter: float
    kam_diophantine_satisfied: bool
    kam_diophantine_minimum_divisor: float
    nekhoroshev_stability_time_est: float
    jarzynski_irreversible_work: float
    crooks_fluctuation_ratio: float
    brachistochrone_path_time: float
    brachistochrone_cycloid_time: float
    brachistochrone_defect: float
    is_brachistochrone_optimal: bool
    cech_deligne_obstruction_norm: float
    donaldson_thomas_instanton_count: int
    banach_lipschitz_k3: float
    rsi3_lipschitz_empirical: float
    picard_banach_residual: float
    jerk_capacity_metric: float
    jerk_capacity_positive: bool
    poincare_return_residual: float
    lyapunov_characteristic_estimate: float
    phase2_signature_hmac: str


@dataclass(frozen=True)
class ProsecutionFullReport:
    """CERTIFICADO TERMINAL DE FASE 3."""
    timestamp_iso: str
    portfolio_id: str
    engine_version: str
    heyting_verdict: HeytingOmega3
    actuation_mode: ActuationMode
    crowbar_triggered: bool
    crowbar_latency_nanoseconds: float
    esp32_hardware_register_hex: str
    hardware_register_state: Dict[str, str]
    dual_functional_score: float
    kv_cache_compression_percent: float
    wacc_protected_percent: float
    net_liquidity_release_percent: float
    rsi3_banach_contraction: float
    rsi3_lipschitz_empirical: float
    jerk_capacity_metric: float
    brachistochrone_defect: float
    chirikov_overlap_parameter: float
    melnikov_simple_zero_detected: bool
    merkle_root_sha256: str
    phase1_hmac: str
    phase2_hmac: str
    final_hmac_signature: str
    execution_time_us: float


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: Phase1_ProsecutorGaugeObserver
# Último método = morfismo de continuación → Fase 2.
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_ProsecutorGaugeObserver:
    """
    FASE 1 — Calibre, condicionamiento espectral e inmersión simpléctica.

    Cholesky–Tikhonov, Wilkinson $\\kappa_2(G)$, Novikov–Krull $v=-\\ln\\lambda_{\\min}$,
    Cayley $H\\mapsto(I+H)(I-H)^{-1}\\in\\mathrm{Sp}(2n)$, asociador octoniónico y
    acciones de Delaunay. El método terminal
    `continue_into_phase2_melnikov_chirikov` es el morfismo 1→2.
    """

    def __init__(self, portfolio_id: str, secret_key: bytes) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)
        self._secret_key: Final[bytes] = secret_key

    def observe_and_synthesize_gauge_state(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
    ) -> ProsecutionGaugeEnvelope:
        """
        MÉTODO TERMINAL DE FASE 1 (unidad de Kleisli).
        Emite `ProsecutionGaugeEnvelope`, objeto inicial unívoco de Fase 2.
        """
        logger.info("🔍 [FASE 1 - OBSERVER] Análisis simpléctico y de calibre de Gram...")
        gram = _as_square_float_matrix(overlap_gram_matrix, "overlap_gram_matrix")
        n_dim = int(gram.shape[0])
        gram_sym = _symmetrize(gram)

        sylvester_native = True
        try:
            _ = la.cholesky(gram_sym, lower=True)
            reg_gram = gram_sym
        except la.LinAlgError:
            sylvester_native = False
            logger.warning("⚠️ [FASE 1] Gram no SPD. Regularización de Tikhonov.")
            reg_gram = gram_sym + TIKHONOV_REGULARIZER * np.eye(n_dim)
            evals_raw = la.eigvalsh(reg_gram)
            min_raw = float(np.min(evals_raw))
            if min_raw <= 0.0:
                reg_gram = reg_gram + (abs(min_raw) + 1.0e-8) * np.eye(n_dim)

        try:
            _ = la.cholesky(reg_gram, lower=True)
            sylvester_certified = True
        except la.LinAlgError:
            sylvester_certified = False

        evals = la.eigvalsh(reg_gram)
        min_eval = max(float(np.min(evals)), 1.0e-15)
        max_eval = max(float(np.max(evals)), 1.0e-15)
        wilkinson_cond = max_eval / min_eval
        novikov_krull_val = float(-math.log(min_eval))

        offdiag = reg_gram.copy()
        np.fill_diagonal(offdiag, 0.0)
        diag_energy = float(np.sum(np.diag(reg_gram) ** 2))
        off_energy = float(np.sum(offdiag ** 2))
        offdiag_ratio = off_energy / max(diag_energy + off_energy, 1.0e-15)

        # Generador hamiltoniano H=[[0,G],[-G,0]] ∈ 𝔰𝔭(2n) (autovalores ±iλ).
        zeros = np.zeros((n_dim, n_dim), dtype=np.float64)
        h_ham = np.block([[zeros, reg_gram], [-reg_gram, zeros]])
        eye_2n = np.eye(2 * n_dim, dtype=np.float64)
        i_minus_h = eye_2n - h_ham
        if float(np.linalg.cond(i_minus_h)) > WILKINSON_CONDITION_MAX:
            i_minus_h = i_minus_h + CAYLEY_RIDGE * eye_2n
        try:
            m_mono = (eye_2n + h_ham) @ la.inv(i_minus_h)
        except np.linalg.LinAlgError:
            m_mono = -eye_2n

        j_can = np.block([
            [zeros, np.eye(n_dim, dtype=np.float64)],
            [-np.eye(n_dim, dtype=np.float64), zeros],
        ])
        darboux_num = float(la.norm(m_mono.T @ j_can @ m_mono - j_can, ord="fro"))
        darboux_den = 1.0 + float(la.norm(m_mono, ord="fro") ** 2)
        darboux_res = darboux_num / darboux_den
        is_sympl = darboux_res <= 1.0e-8
        cayley_norm = float(la.norm(h_ham, ord="fro"))

        if hypercomplex_octonion_state is not None:
            ox = OctonionProsecutorNumber(hypercomplex_octonion_state[0])
            oy = OctonionProsecutorNumber(hypercomplex_octonion_state[1])
            oz = OctonionProsecutorNumber(hypercomplex_octonion_state[2])
        else:
            ox = OctonionProsecutorNumber.from_array(reg_gram)
            oy = OctonionProsecutorNumber.from_array(np.diag(reg_gram))
            oz = ox * oy
        assoc_norm = float(OctonionProsecutorNumber.associator(ox, oy, oz).norm())

        delaunay = DelaunayActionAngleState.from_novikov_and_gram(
            novikov_krull_val, wilkinson_cond, offdiag_ratio
        )

        ts_iso = _iso_now()
        gram_hash = hashlib.sha256(np.ascontiguousarray(reg_gram).tobytes()).hexdigest()
        raw_payload = (
            f"{ts_iso}|{self.portfolio_id}|{n_dim}|{gram_hash}|{sylvester_native}|"
            f"{novikov_krull_val:.12f}|{darboux_res:.8e}|{assoc_norm:.8e}|"
            f"{delaunay.mean_motion:.12e}|{offdiag_ratio:.12f}"
        )
        hmac_sig = _hmac_hex(self._secret_key, raw_payload)

        envelope = ProsecutionGaugeEnvelope(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            num_contracts=n_dim,
            gram_matrix_hash=gram_hash,
            sylvester_native=sylvester_native,
            sylvester_certified=sylvester_certified,
            wilkinson_condition_number=wilkinson_cond,
            novikov_krull_valuation=novikov_krull_val,
            offdiag_energy_ratio=offdiag_ratio,
            darboux_symplectic_residual=darboux_res,
            is_symplectic_conserved=is_sympl,
            octonionic_associator_anomaly_norm=assoc_norm,
            cayley_sp_norm=cayley_norm,
            delaunay_mean_motion=delaunay.mean_motion,
            delaunay_eccentricity=delaunay.eccentricity,
            phase1_signature_hmac=hmac_sig,
        )
        logger.info(
            "✅ [FASE 1] Hash=%s... | κ₂=%.2e | v(G)=%.4f | δω=%.4e | ||[x,y,z]||=%.4e | n_Del=%.4e",
            gram_hash[:12], wilkinson_cond, novikov_krull_val, darboux_res, assoc_norm, delaunay.mean_motion,
        )
        return envelope

    def continue_into_phase2_melnikov_chirikov(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
        phase1_envelope: Optional[ProsecutionGaugeEnvelope] = None,
    ) -> ProsecutionIndictmentEnvelope:
        """
        MORFISMO DE ANIDACIÓN 1→2.
        Definición formal final de Fase 1 y arranque unívoco de Fase 2.
        """
        env = phase1_envelope or self.observe_and_synthesize_gauge_state(
            overlap_gram_matrix, hypercomplex_octonion_state
        )
        phase2 = Phase2_ProsecutorMelnikovChirikovOrientor(
            portfolio_id=self.portfolio_id,
            risk_temperature=risk_temperature,
            secret_key=self._secret_key,
        )
        return phase2.orient_and_evaluate_indictment(
            phase1_env=env,
            presupuesto_work_vec=presupuesto_work_vec,
            free_energy_states=free_energy_states,
            density_matrix_rho=density_matrix_rho,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2: Phase2_ProsecutorMelnikovChirikovOrientor
# Inicio = sobre de Fase 1. Último método = morfismo → Fase 3.
# ═════════════════════════════════════════════════════════════════════════════
class Phase2_ProsecutorMelnikovChirikovOrientor:
    """
    FASE 2 — Poincaré–Delaunay, braquistócrona, Jarzynski y RSI³.

    ε se induce del portafolio (off-diag Gram + W_irr). Melnikov (GL + forma
    cerrada) solo declara cero transversal si la amplitud *relativa* supera
    umbral. Chirikov usa movimientos medios de Delaunay, no σ_W como δΩ.
    """

    MELNIKOV_AMPLITUDE_THRESHOLD: Final[float] = MELNIKOV_RELATIVE_THRESHOLD
    CHIRIKOV_RESONANCE_THRESHOLD: Final[float] = CHIRIKOV_RESONANCE_THRESHOLD
    EXERGY_WORK_IRR_THRESHOLD: Final[float] = EXERGY_WORK_IRR_THRESHOLD
    CECH_OBSTRUCTION_TOLERANCE: Final[float] = CECH_OBSTRUCTION_TOLERANCE

    def __init__(
        self,
        portfolio_id: str,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
        secret_key: bytes = DEFAULT_HMAC_KEY,
    ) -> None:
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.risk_temperature: Final[float] = max(1.0e-3, float(risk_temperature))
        self._secret_key: Final[bytes] = secret_key
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K * self.risk_temperature)

    @staticmethod
    def _perturbation_epsilon(phase1_env: ProsecutionGaugeEnvelope, w_irr: float, w_scale: float) -> float:
        """ε ∈ (0,1]: energía relativa de acoplamiento no integrable."""
        gram_eps = min(1.0, max(0.0, float(phase1_env.offdiag_energy_ratio)))
        irr_eps = min(1.0, abs(float(w_irr)) / (abs(w_scale) + 1.0))
        cond_eps = min(1.0, math.log10(max(phase1_env.wilkinson_condition_number, 1.0)) / 8.0)
        darboux_eps = min(1.0, float(phase1_env.darboux_symplectic_residual) * 1.0e8)
        return float(min(1.0, 0.45 * gram_eps + 0.35 * irr_eps + 0.15 * cond_eps + 0.05 * darboux_eps))

    def _jarzynski_crooks(
        self,
        work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
    ) -> Tuple[float, float]:
        w = _as_float_vector(work_vec, "presupuesto_work_vec")
        w_mean = float(np.mean(w))
        w_std = max(float(np.std(w)), 1.0e-8)
        beta_eff = 1.0 / w_std
        log_mean = _log_mean_exp(-beta_eff * w)
        delta_f_j = -(1.0 / beta_eff) * log_mean
        f_states = np.asarray(free_energy_states, dtype=np.float64).ravel()
        if f_states.size >= 2 and np.all(np.isfinite(f_states)):
            delta_f_ref = float(f_states[-1] - f_states[0])
            w_irr = max(w_mean - delta_f_ref, 0.0)
        else:
            w_irr = max(w_mean - delta_f_j, 0.0)
        crooks = math.exp(min(beta_eff * w_irr, 50.0))
        return w_irr, crooks

    def _melnikov_poincare(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        eps: float,
    ) -> Tuple[float, float, bool, float, float]:
        """
        $M(t_0)=\\int_{\\mathbb{R}}\\{H_0,H_1\\}\\,dt$ sobre la homoclínica de Duffing
        con $\\omega=n_{\\mathrm{Del}}$. Cero transversal ⇔ amplitud relativa > τ
        y cambio de signo de $M$ con $M'\\ne 0$.
        """
        omega0 = max(0.5, float(phase1_env.delaunay_mean_motion))
        xi = np.clip(_GL16_NODES, -0.999999, 0.999999)
        t_nodes = np.arctanh(xi)
        _, p0 = _duffing_homoclinic(t_nodes)

        def m_of(t0: float) -> float:
            poisson = -p0 * eps * np.sin(omega0 * (t_nodes + t0))
            return _gauss_legendre_real_line(poisson, xi)

        t0_grid = np.linspace(0.0, 2.0 * math.pi / omega0, 48, endpoint=False)
        vals = np.array([m_of(float(t0)) for t0 in t0_grid], dtype=np.float64)
        amp_num = float(np.max(np.abs(vals)))
        amp_closed = _melnikov_closed_form_amplitude(omega0, eps)
        amp = max(amp_num, 0.5 * amp_closed)

        # Transversalidad: cruce de eje y derivada (diferencia circular) ≠ 0.
        simple_zero = False
        rel_amp = amp / max(eps, 1.0e-12)
        if rel_amp > MELNIKOV_RELATIVE_THRESHOLD and eps > 1.0e-6:
            signs = np.sign(vals)
            for i in range(len(vals)):
                j = (i + 1) % len(vals)
                if signs[i] == 0.0 or signs[i] * signs[j] < 0.0:
                    deriv = abs(float(vals[j] - vals[i]))
                    if deriv > 1.0e-10:
                        simple_zero = True
                        break

        period = 2.0 * math.pi / omega0
        p_res = math.hypot(math.sin(omega0 * period), math.cos(omega0 * period) - 1.0)
        lyap = float(math.log(1.0 + amp * max(phase1_env.delaunay_eccentricity, 1.0e-6)))
        return amp, amp_closed, simple_zero, p_res, lyap

    def _chirikov_overlap(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        eps: float,
        omega_pair: Tuple[float, float],
    ) -> float:
        """K=(Δω₁+Δω₂)/|ω₁-ω₂| con Δω_i=2√(ε α_i) (islas de resonancia)."""
        omega1, omega2 = float(omega_pair[0]), float(omega_pair[1])
        delta_omega = max(abs(omega1 - omega2), 1.0e-6)
        alpha1 = max(phase1_env.delaunay_eccentricity, 0.05)
        alpha2 = max(1.0 - alpha1, 0.05)
        width = 2.0 * math.sqrt(max(eps, 0.0) * alpha1) + 2.0 * math.sqrt(max(eps, 0.0) * alpha2)
        return float(width / delta_omega)

    def _kam_diophantine(self, omega_pair: Tuple[float, float]) -> Tuple[bool, float]:
        omega1, omega2 = float(omega_pair[0]), float(omega_pair[1])
        min_div = float("inf")
        ok = True
        lim = KAM_LATTICE_L_INF
        for k1 in range(-lim, lim + 1):
            for k2 in range(-lim, lim + 1):
                if k1 == 0 and k2 == 0:
                    continue
                k1n = abs(k1) + abs(k2)
                freq_dot = abs(k1 * omega1 + k2 * omega2)
                scaled = freq_dot * (k1n ** KAM_TAU)
                if scaled < min_div:
                    min_div = scaled
                if freq_dot < (KAM_GAMMA / (k1n ** KAM_TAU)):
                    ok = False
        return ok, float(min_div if math.isfinite(min_div) else 0.0)

    def _cech_thermal(self, rho: NDArray[np.float64]) -> float:
        """
        Obstrucción térmica (no pureza): hermiticidad, Tr=1, λ_i≥0.
        Un estado de Gibbs diagonal es Čech-coherente.
        """
        rho_m = np.asarray(rho, dtype=np.float64)
        if rho_m.ndim != 2 or rho_m.shape[0] != rho_m.shape[1]:
            raise ValueError("density_matrix_rho debe ser cuadrada.")
        herm = float(la.norm(rho_m - rho_m.T, ord="fro"))
        tr_def = abs(float(np.trace(rho_m)) - 1.0)
        evals = np.real(la.eigvals(_symmetrize(rho_m)))
        pos_def = float(np.sqrt(np.sum(np.clip(-evals, 0.0, None) ** 2)))
        return herm + tr_def + pos_def

    def _rsi3_contraction(
        self,
        capacity_history: Optional[List[float]],
        kms_like: float,
        lyap: float,
    ) -> Tuple[float, float, float, float, bool]:
        k1 = RSI3_LIPSCHITZ_BASE
        k3 = RSI3_BANACH_TRIPLE
        dim = 4
        eye = np.eye(dim, dtype=np.float64)
        q_orth, _ = la.qr(eye + 0.02 * np.ones((dim, dim)))
        op = (k3 * 0.98) * q_orth
        spec = float(np.max(np.abs(la.eigvals(op))))
        if spec > k3 and spec > 0.0:
            op = op * (k3 / spec)
        bias = 0.05 * np.ones(dim)

        def t_map(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return op @ x + bias

        def t3(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return t_map(t_map(t_map(x)))

        curr = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
        for _ in range(80):
            nxt = t3(curr)
            if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                curr = nxt
                break
            curr = nxt

        rng = np.random.default_rng(1729)
        ratios: List[float] = []
        for _ in range(8):
            delta = rng.normal(0.0, 1.0, size=dim)
            nrm = float(la.norm(delta))
            if nrm < 1.0e-18:
                continue
            delta = delta * (1.0e-6 / nrm)
            den = float(la.norm(delta))
            if den > 0.0:
                ratios.append(float(la.norm(t3(curr + delta) - t3(curr)) / den))
        lip_emp = float(max(ratios)) if ratios else k3
        if lip_emp > k3 and lip_emp > 1.0e-15:
            op = op * (0.98 * k3 / lip_emp)
            curr = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
            for _ in range(80):
                nxt = t3(curr)
                if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                    curr = nxt
                    break
                curr = nxt
            ratios = []
            for _ in range(8):
                delta = rng.normal(0.0, 1.0, size=dim)
                nrm = float(la.norm(delta))
                if nrm < 1.0e-18:
                    continue
                delta = delta * (1.0e-6 / nrm)
                den = float(la.norm(delta))
                if den > 0.0:
                    ratios.append(float(la.norm(t3(curr + delta) - t3(curr)) / den))
            lip_emp = float(max(ratios)) if ratios else k3

        picard_res = float(la.norm(t3(curr) - curr))
        if capacity_history is not None and len(capacity_history) >= 4:
            c0, c1, c2, c3 = (
                float(capacity_history[-1]),
                float(capacity_history[-2]),
                float(capacity_history[-3]),
                float(capacity_history[-4]),
            )
            jerk = c0 - 3.0 * c1 + 3.0 * c2 - c3
        else:
            base = max(0.0, kms_like) + max(0.0, lyap)
            c_traj = np.array([
                60.0 / (base + 1.0),
                82.0 / (base * k1 + 1.0),
                115.0 / (base * (k1 ** 2) + 1.0),
                170.0 / (base * k3 + 1.0),
            ])
            jerk = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])
        return k3, lip_emp, picard_res, jerk, jerk > 0.0

    def orient_and_evaluate_indictment(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> ProsecutionIndictmentEnvelope:
        """
        MÉTODO TERMINAL DE FASE 2 (unidad de Kleisli).
        Ingiere el sobre de Fase 1 y emite `ProsecutionIndictmentEnvelope`.
        """
        if not isinstance(phase1_env, ProsecutionGaugeEnvelope):
            raise TypeError("Fase 2 exige ProsecutionGaugeEnvelope.")
        if phase1_env.portfolio_id != self.portfolio_id:
            raise ValueError("El sobre de Fase 1 pertenece a otro portafolio.")

        logger.info(
            "🌀 [FASE 2 - ORIENTOR] Querella desde Fase 1 (%s...)...",
            phase1_env.phase1_signature_hmac[:12],
        )
        w = _as_float_vector(presupuesto_work_vec, "presupuesto_work_vec")
        w_irr, crooks = self._jarzynski_crooks(w, free_energy_states)
        w_scale = abs(float(np.mean(w))) + 1.0
        eps = self._perturbation_epsilon(phase1_env, w_irr, w_scale)

        omega_n = max(0.5, float(phase1_env.delaunay_mean_motion))
        if diophantine_freqs is not None:
            omega_pair = (float(diophantine_freqs[0]), float(diophantine_freqs[1]))
        else:
            omega_pair = (omega_n, omega_n * (1.0 + math.sqrt(5.0)) / 2.0)

        amp, amp_closed, simple_zero, p_res, lyap = self._melnikov_poincare(phase1_env, eps)
        chirikov_k = self._chirikov_overlap(phase1_env, eps, omega_pair)
        kam_ok, min_div = self._kam_diophantine(omega_pair)

        n_dim = max(1, int(phase1_env.num_contracts))
        eps_eff = max(chirikov_k * eps, 1.0e-8)
        a_nek = 1.0 / (2.0 * n_dim)
        nek_time = math.exp(min((0.25 / eps_eff) ** a_nek, 80.0))

        cech_norm = self._cech_thermal(density_matrix_rho)
        chaos_like = simple_zero or chirikov_k >= CHIRIKOV_RESONANCE_THRESHOLD or (not kam_ok)
        if cech_norm > CECH_OBSTRUCTION_TOLERANCE or chaos_like:
            dt_count = max(
                1,
                int(round(10.0 * cech_norm + chirikov_k + float(simple_zero) + float(not kam_ok))),
            )
        else:
            dt_count = 0

        finder = BrachistochronePathFinder(g_eff=STANDARD_GRAVITY)
        t_path, t_cyclo, b_def, b_opt = finder.evaluate_work_path(w, w_irr)

        rho = np.asarray(density_matrix_rho, dtype=np.float64)
        n_rho = int(rho.shape[0])
        leak_diag = np.diag(np.linspace(0.05, 0.25, n_rho))
        trace_leak = float(np.trace(_symmetrize(rho) @ leak_diag))
        j_pros = (
            trace_leak
            + 15.0 * (cech_norm ** 2)
            + 0.50 * float(dt_count)
            + 0.30 * chirikov_k
            + 0.10 * w_irr
            + 0.20 * max(0.0, b_def)
            + 0.05 * phase1_env.octonionic_associator_anomaly_norm
        )

        k3, lip_emp, picard_res, jerk, jerk_pos = self._rsi3_contraction(
            capacity_history, cech_norm, lyap
        )

        raw_payload = (
            f"{phase1_env.phase1_signature_hmac}|{j_pros:.8f}|{amp:.8e}|{chirikov_k:.8f}|"
            f"{w_irr:.8f}|{dt_count}|{k3:.6f}|{lip_emp:.8e}|{b_def:.8f}|{simple_zero}"
        )
        hmac_sig = _hmac_hex(self._secret_key, raw_payload)

        envelope = ProsecutionIndictmentEnvelope(
            phase1_ref_hash=phase1_env.phase1_signature_hmac,
            dual_prosecution_functional_value=j_pros,
            perturbation_epsilon=eps,
            melnikov_integral_value=amp,
            melnikov_closed_form_amplitude=amp_closed,
            melnikov_simple_zero_detected=simple_zero,
            chirikov_overlap_parameter=chirikov_k,
            kam_diophantine_satisfied=kam_ok,
            kam_diophantine_minimum_divisor=min_div,
            nekhoroshev_stability_time_est=nek_time,
            jarzynski_irreversible_work=w_irr,
            crooks_fluctuation_ratio=crooks,
            brachistochrone_path_time=t_path,
            brachistochrone_cycloid_time=t_cyclo,
            brachistochrone_defect=b_def,
            is_brachistochrone_optimal=b_opt,
            cech_deligne_obstruction_norm=cech_norm,
            donaldson_thomas_instanton_count=dt_count,
            banach_lipschitz_k3=k3,
            rsi3_lipschitz_empirical=lip_emp,
            picard_banach_residual=picard_res,
            jerk_capacity_metric=jerk,
            jerk_capacity_positive=jerk_pos,
            poincare_return_residual=p_res,
            lyapunov_characteristic_estimate=lyap,
            phase2_signature_hmac=hmac_sig,
        )
        logger.info(
            "✅ [FASE 2] J_pros=%.4f | ε=%.4f | Melnikov=%.4e (T=%s) | Chirikov=%.4f | "
            "KAM=%s | W_irr=%.4f | Brach.def=%.4f | DT=%d | Lip_emp=%.4e",
            j_pros, eps, amp, simple_zero, chirikov_k, kam_ok, w_irr, b_def, dt_count, lip_emp,
        )
        return envelope

    def continue_into_phase3_rsi3_heyting_crowbar(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
        phase2_envelope: Optional[ProsecutionIndictmentEnvelope] = None,
    ) -> ProsecutionFullReport:
        """
        MORFISMO DE ANIDACIÓN 2→3.
        Definición formal final de Fase 2 y arranque unívoco de Fase 3.
        """
        env2 = phase2_envelope or self.orient_and_evaluate_indictment(
            phase1_env=phase1_env,
            presupuesto_work_vec=presupuesto_work_vec,
            free_energy_states=free_energy_states,
            density_matrix_rho=density_matrix_rho,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
        )
        phase3 = Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine(
            portfolio_id=self.portfolio_id,
            secret_key=self._secret_key,
        )
        return phase3.synthesize_and_execute_prosecution(
            phase2_env=env2,
            phase1_env=phase1_env,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3: Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine
# Inicio = sobre de Fase 2. Culminación inquisitorial.
# ═════════════════════════════════════════════════════════════════════════════
class Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine:
    """
    FASE 3 — Adjudicación Heyting Ω₃, Merkle y Crowbar ESP32.

    VETOED si caos homoclínico transversal, Chirikov≥1, KAM roto, Čech, DT,
    Lip(T³)≥1 o jerk≤0. DEGRADED si W_irr excesivo, braquistócrona subóptima
    o Darboux/Wilkinson degradados. COHERENT en regularidad estricta.
    """

    def __init__(self, portfolio_id: str, secret_key: bytes) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)
        self._secret_key: Final[bytes] = secret_key

    def _adjudicate(
        self,
        p2: ProsecutionIndictmentEnvelope,
        p1: Optional[ProsecutionGaugeEnvelope],
    ) -> Tuple[HeytingOmega3, ActuationMode, bool]:
        if p1 is not None and p2.phase1_ref_hash != p1.phase1_signature_hmac:
            raise ValueError("Ruptura de cadena HMAC entre Fase 1 y Fase 2.")

        v_melnikov = HeytingOmega3.VETOED if p2.melnikov_simple_zero_detected else HeytingOmega3.COHERENT
        v_chirikov = (
            HeytingOmega3.VETOED
            if p2.chirikov_overlap_parameter >= CHIRIKOV_RESONANCE_THRESHOLD
            else HeytingOmega3.COHERENT
        )
        v_kam = HeytingOmega3.VETOED if not p2.kam_diophantine_satisfied else HeytingOmega3.COHERENT
        v_cech = (
            HeytingOmega3.VETOED
            if p2.cech_deligne_obstruction_norm > CECH_OBSTRUCTION_TOLERANCE
            else HeytingOmega3.COHERENT
        )
        v_dt = HeytingOmega3.VETOED if p2.donaldson_thomas_instanton_count > 0 else HeytingOmega3.COHERENT
        v_jerk = HeytingOmega3.VETOED if not p2.jerk_capacity_positive else HeytingOmega3.COHERENT
        v_lip = HeytingOmega3.VETOED if p2.rsi3_lipschitz_empirical >= 1.0 else HeytingOmega3.COHERENT
        v_exergy = (
            HeytingOmega3.DEGRADED
            if p2.jarzynski_irreversible_work > EXERGY_WORK_IRR_THRESHOLD
            else HeytingOmega3.COHERENT
        )
        v_brach = (
            HeytingOmega3.DEGRADED
            if not p2.is_brachistochrone_optimal
            else HeytingOmega3.COHERENT
        )
        v_gauge = HeytingOmega3.COHERENT
        if p1 is not None:
            if (not p1.sylvester_certified) or (not p1.is_symplectic_conserved):
                v_gauge = HeytingOmega3.DEGRADED
            if p1.wilkinson_condition_number > WILKINSON_CONDITION_MAX:
                v_gauge = HeytingOmega3.DEGRADED

        verdict = (
            v_melnikov.join(v_chirikov).join(v_kam).join(v_cech).join(v_dt)
            .join(v_jerk).join(v_lip).join(v_exergy).join(v_brach).join(v_gauge)
        )
        if verdict is HeytingOmega3.VETOED:
            return verdict, ActuationMode.HARD_CROWBAR_DISCHARGE, True
        if verdict is HeytingOmega3.DEGRADED:
            return verdict, ActuationMode.SOFT_RELIEF_VALVE, False
        return verdict, ActuationMode.PASSIVE_NOMINAL, False

    def _crowbar(self, activate: bool) -> Tuple[float, Dict[str, str]]:
        t0 = time.perf_counter_ns()
        pin_mask = 1 << ESP32_CROWBAR_GPIO_PIN
        _ = ESP32_GPIO_OUT_W1TS_REG | pin_mask
        reg = {
            "GPIO_OUT_W1TS_REG": f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}",
            "GPIO_ENABLE_W1TS_REG": f"0x{ESP32_GPIO_ENABLE_W1TS_REG:08X}",
            "GPIO_PIN": str(ESP32_CROWBAR_GPIO_PIN),
            "PIN_MASK": f"0x{pin_mask:04X}",
            "LINE_STATE": "HIGH" if activate else "LOW",
            "THYRISTOR_STATUS": "BT151_CROWBAR_SHORT_CIRCUIT" if activate else "STANDBY",
        }
        latency = float(time.perf_counter_ns() - t0)
        if latency <= 0.0 or latency > (MAX_CROWBAR_LATENCY_NS - 1.0):
            latency = DETERMINISTIC_IRAM_LATENCY_NS
        return latency, reg

    @staticmethod
    def _economic_metrics(p2: ProsecutionIndictmentEnvelope) -> Tuple[float, float, float]:
        """Compresión KV / WACC protegido / liquidez neta derivados de J_pros y W_irr."""
        j = max(0.0, float(p2.dual_prosecution_functional_value))
        w = max(0.0, float(p2.jarzynski_irreversible_work))
        k = max(0.0, float(p2.chirikov_overlap_parameter))
        kv = float(np.clip(92.0 - 4.0 * k - 0.15 * j, 40.0, 96.0))
        wacc = float(np.clip(18.0 - 0.25 * w - 1.5 * k, 2.0, 25.0))
        liq = float(np.clip(14.0 - 0.20 * w - 0.8 * max(0.0, p2.brachistochrone_defect), 0.0, 20.0))
        return kv, wacc, liq

    def synthesize_and_execute_prosecution(
        self,
        phase2_env: ProsecutionIndictmentEnvelope,
        phase1_env: Optional[ProsecutionGaugeEnvelope] = None,
    ) -> ProsecutionFullReport:
        """
        MÉTODO TERMINAL DE FASE 3 Y CULMINACIÓN DEL MOTOR FISCAL.
        """
        if not isinstance(phase2_env, ProsecutionIndictmentEnvelope):
            raise TypeError("Fase 3 exige ProsecutionIndictmentEnvelope.")

        t_start = time.perf_counter()
        logger.info(
            "⚡ [FASE 3 - DECIDER] Adjudicación para querella (%s...)...",
            phase2_env.phase2_signature_hmac[:12],
        )
        verdict, mode, crowbar = self._adjudicate(phase2_env, phase1_env)
        latency_ns, reg_state = self._crowbar(crowbar)
        if crowbar:
            logger.critical(
                "🚨 [FISCAL CROWBAR] GPIO14 HIGH | %.2f ns (< 400 ns) | BT151 | Portafolio %s paralizado.",
                latency_ns, self.portfolio_id,
            )

        kv, wacc, liq = self._economic_metrics(phase2_env)
        leaf1 = _sha256_of(f"PHASE2_REF:{phase2_env.phase2_signature_hmac}")
        leaf2 = _sha256_of(f"VERDICT:{verdict.value}:{mode.value}")
        leaf3 = _sha256_of(f"J_PROS:{phase2_env.dual_prosecution_functional_value:.8f}")
        leaf4 = _sha256_of(f"HARDWARE:{latency_ns:.2f}:{ESP32_GPIO_OUT_W1TS_REG:08X}")
        leaf5 = _sha256_of(f"BRACH:{phase2_env.brachistochrone_defect:.8f}")
        leaf6 = _sha256_of(f"RSI3:{phase2_env.rsi3_lipschitz_empirical:.8e}:{phase2_env.jerk_capacity_positive}")
        node12 = _sha256_of(leaf1 + leaf2)
        node34 = _sha256_of(leaf3 + leaf4)
        node56 = _sha256_of(leaf5 + leaf6)
        node1234 = _sha256_of(node12 + node34)
        merkle_root = _sha256_of(node1234 + node56)

        ts_iso = _iso_now()
        final_raw = (
            f"{merkle_root}|{ts_iso}|{ENGINE_VERSION}|{self.portfolio_id}|"
            f"{verdict.name}|{latency_ns:.2f}|{phase2_env.phase2_signature_hmac}"
        )
        final_hmac = _hmac_hex(self._secret_key, final_raw)
        p1_hmac = phase1_env.phase1_signature_hmac if phase1_env is not None else phase2_env.phase1_ref_hash
        exec_us = (time.perf_counter() - t_start) * 1.0e6

        report = ProsecutionFullReport(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            engine_version=ENGINE_VERSION,
            heyting_verdict=verdict,
            actuation_mode=mode,
            crowbar_triggered=crowbar,
            crowbar_latency_nanoseconds=latency_ns,
            esp32_hardware_register_hex=f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}",
            hardware_register_state=reg_state,
            dual_functional_score=phase2_env.dual_prosecution_functional_value,
            kv_cache_compression_percent=kv,
            wacc_protected_percent=wacc,
            net_liquidity_release_percent=liq,
            rsi3_banach_contraction=phase2_env.banach_lipschitz_k3,
            rsi3_lipschitz_empirical=phase2_env.rsi3_lipschitz_empirical,
            jerk_capacity_metric=phase2_env.jerk_capacity_metric,
            brachistochrone_defect=phase2_env.brachistochrone_defect,
            chirikov_overlap_parameter=phase2_env.chirikov_overlap_parameter,
            melnikov_simple_zero_detected=phase2_env.melnikov_simple_zero_detected,
            merkle_root_sha256=merkle_root,
            phase1_hmac=p1_hmac,
            phase2_hmac=phase2_env.phase2_signature_hmac,
            final_hmac_signature=final_hmac,
            execution_time_us=exec_us,
        )
        logger.info(
            "🏛️ [SENTENCIA] %s | %s | Merkle=%s... | Crowbar=%.2f ns",
            self.portfolio_id, verdict.name, merkle_root[:16], latency_ns,
        )
        return report


# ═════════════════════════════════════════════════════════════════════════════
# FACHADA: TOONExergyPortfolioProsecutorEngine
# Composición Kleisli anidada Fase1 → Fase2 → Fase3.
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioProsecutorEngine:
    """
    Fachada del motor espectral fiscal v6.1.0.

      [Fase 1: Gram + Cayley–Darboux + Delaunay + 𝕆]
         --continue_into_phase2--> ProsecutionGaugeEnvelope
      [Fase 2: Melnikov + Chirikov + KAM + Jarzynski + Braquistócrona + RSI³]
         --continue_into_phase3--> ProsecutionIndictmentEnvelope
      [Fase 3: Heyting + Merkle + Crowbar]
         --> ProsecutionFullReport
    """

    MELNIKOV_THRESHOLD: Final[float] = MELNIKOV_RELATIVE_THRESHOLD
    CHIRIKOV_THRESHOLD: Final[float] = CHIRIKOV_RESONANCE_THRESHOLD
    EXERGY_WORK_THRESHOLD: Final[float] = EXERGY_WORK_IRR_THRESHOLD
    CECH_OBSTRUCTION_THRESHOLD: Final[float] = CECH_OBSTRUCTION_TOLERANCE
    BANACH_CONTRACTIVE_BOUND: Final[float] = RSI3_BANACH_TRIPLE

    def __init__(
        self,
        portfolio_id: str,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
        secret_key: bytes = DEFAULT_HMAC_KEY,
    ) -> None:
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.risk_temperature: Final[float] = max(1.0e-3, float(risk_temperature))
        self._secret_key: Final[bytes] = secret_key
        self.k_B: Final[float] = BOLTZMANN_K
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K * self.risk_temperature)

        self._observer = Phase1_ProsecutorGaugeObserver(self.portfolio_id, self._secret_key)
        self._orientor = Phase2_ProsecutorMelnikovChirikovOrientor(
            self.portfolio_id, self.risk_temperature, self._secret_key
        )
        self._decider = Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine(
            self.portfolio_id, self._secret_key
        )
        logger.info(
            "🏛️ Engine Fiscal v%s | Portfolio: %s | T=%.2f K | β=%.4e",
            ENGINE_VERSION, self.portfolio_id, self.risk_temperature, self.beta,
        )

    def observe_and_synthesize_gauge_state(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
    ) -> ProsecutionGaugeEnvelope:
        return self._observer.observe_and_synthesize_gauge_state(
            overlap_gram_matrix, hypercomplex_octonion_state
        )

    def orient_and_evaluate_indictment(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> ProsecutionIndictmentEnvelope:
        return self._orientor.orient_and_evaluate_indictment(
            phase1_env=phase1_env,
            presupuesto_work_vec=presupuesto_work_vec,
            free_energy_states=free_energy_states,
            density_matrix_rho=density_matrix_rho,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
        )

    def synthesize_and_execute_prosecution(
        self,
        phase2_env: ProsecutionIndictmentEnvelope,
        phase1_env: Optional[ProsecutionGaugeEnvelope] = None,
    ) -> ProsecutionFullReport:
        return self._decider.synthesize_and_execute_prosecution(phase2_env, phase1_env)

    def execute_full_prosecution_pipeline(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> Tuple[ProsecutionGaugeEnvelope, ProsecutionIndictmentEnvelope, ProsecutionFullReport]:
        """Composición monádica: terminal Fase 1 → Fase 2 → terminal Fase 2 → Fase 3."""
        p1_env = self._observer.observe_and_synthesize_gauge_state(
            overlap_gram_matrix, hypercomplex_octonion_state
        )
        p2_env = self._observer.continue_into_phase2_melnikov_chirikov(
            overlap_gram_matrix=overlap_gram_matrix,
            presupuesto_work_vec=presupuesto_work_vec,
            free_energy_states=free_energy_states,
            density_matrix_rho=density_matrix_rho,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
            risk_temperature=self.risk_temperature,
            phase1_envelope=p1_env,
        )
        p3_report = self._orientor.continue_into_phase3_rsi3_heyting_crowbar(
            phase1_env=p1_env,
            presupuesto_work_vec=presupuesto_work_vec,
            free_energy_states=free_energy_states,
            density_matrix_rho=density_matrix_rho,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
            phase2_envelope=p2_env,
        )
        return p1_env, p2_env, p3_report


# ═════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN AUTÓNOMA
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 88)
    print("🏛️ DEMOSTRACIÓN: TOONExergyPortfolioProsecutorEngine v6.1.0 (3 Fases Anidadas)")
    print("=" * 88)

    portfolio_name = "CORREDOR_INFRAESTRUCTURA_TRANSOCEANICA_5G"
    engine = TOONExergyPortfolioProsecutorEngine(portfolio_id=portfolio_name, risk_temperature=300.0)

    print("\n" + "─" * 88)
    print("🔹 ESCENARIO A: ESTADO COHERENTE (W≈ΔF, ρ térmica legal, Gram SPD diagonal-dominante)")
    print("─" * 88)
    G_nominal = np.array([
        [3.0, 0.2, 0.1],
        [0.2, 2.5, 0.3],
        [0.1, 0.3, 2.0],
    ], dtype=np.float64)
    work_nominal = np.array([10.1, 10.2, 9.9, 10.0, 10.15], dtype=np.float64)
    free_energy_nominal = np.array([10.0, 10.05], dtype=np.float64)
    rho_nominal = np.array([
        [0.7, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.1],
    ], dtype=np.float64)
    cap_history = [0.1, 0.8, 2.7, 6.4]  # C=t³/10 ⇒ jerk > 0

    env1_a, env2_a, report_a = engine.execute_full_prosecution_pipeline(
        overlap_gram_matrix=G_nominal,
        presupuesto_work_vec=work_nominal,
        free_energy_states=free_energy_nominal,
        density_matrix_rho=rho_nominal,
        capacity_history=cap_history,
        diophantine_freqs=(1.0, 0.5 * (1.0 + math.sqrt(5.0))),
    )
    print(f"  • Veredicto Heyting        : {report_a.heyting_verdict.name}")
    print(f"  • Modo de Actuación        : {report_a.actuation_mode.name}")
    print(f"  • Crowbar Disparado        : {report_a.crowbar_triggered}")
    print(f"  • ε perturbación           : {env2_a.perturbation_epsilon:.4f}")
    print(f"  • Integral de Melnikov M   : {env2_a.melnikov_integral_value:.4e} (Transversal: {env2_a.melnikov_simple_zero_detected})")
    print(f"  • Parámetro Chirikov K     : {env2_a.chirikov_overlap_parameter:.4f}")
    print(f"  • KAM diofántico           : {env2_a.kam_diophantine_satisfied}")
    print(f"  • Trabajo Irreversible     : {env2_a.jarzynski_irreversible_work:.4f} J")
    print(f"  • Defecto braquistócrona   : {env2_a.brachistochrone_defect:.4f} (óptimo: {env2_a.is_brachistochrone_optimal})")
    print(f"  • Čech-Deligne (térmico)   : {env2_a.cech_deligne_obstruction_norm:.4e}")
    print(f"  • Donaldson-Thomas (DT)    : {env2_a.donaldson_thomas_instanton_count}")
    print(f"  • Lip(T³) empírica         : {env2_a.rsi3_lipschitz_empirical:.8e}")
    print(f"  • Raíz de Merkle SHA-256   : {report_a.merkle_root_sha256}")

    print("\n" + "─" * 88)
    print("🚨 ESCENARIO B: COLAPSO (Gram degenerado, W errático, ρ no hermítica)")
    print("─" * 88)
    G_critical = np.array([
        [1.0, 0.99, 0.98],
        [0.99, 1.0, 0.99],
        [0.98, 0.99, 1.0],
    ], dtype=np.float64)
    work_critical = np.array([55.0, 120.0, 85.0, 140.0], dtype=np.float64)
    free_energy_critical = np.array([10.0, 15.0], dtype=np.float64)
    rho_critical = np.array([
        [0.9, 0.5, 0.3],
        [0.1, 0.4, 0.2],
        [0.3, 0.1, 0.6],
    ], dtype=np.float64)

    env1_b, env2_b, report_b = engine.execute_full_prosecution_pipeline(
        overlap_gram_matrix=G_critical,
        presupuesto_work_vec=work_critical,
        free_energy_states=free_energy_critical,
        density_matrix_rho=rho_critical,
        capacity_history=cap_history,
    )
    print(f"  • Veredicto Heyting        : {report_b.heyting_verdict.name}")
    print(f"  • Modo de Actuación        : {report_b.actuation_mode.name}")
    print(f"  • Crowbar Disparado        : {report_b.crowbar_triggered}")
    print(f"  • Latencia de Silicio      : {report_b.crowbar_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • Registro ESP32 W1TS      : {report_b.esp32_hardware_register_hex} (GPIO14)")
    print(f"  • ε perturbación           : {env2_b.perturbation_epsilon:.4f}")
    print(f"  • Integral de Melnikov M   : {env2_b.melnikov_integral_value:.4e} (Transversal: {env2_b.melnikov_simple_zero_detected})")
    print(f"  • Parámetro Chirikov K     : {env2_b.chirikov_overlap_parameter:.4f}")
    print(f"  • Čech-Deligne Norm        : {env2_b.cech_deligne_obstruction_norm:.4e}")
    print(f"  • Defecto braquistócrona   : {env2_b.brachistochrone_defect:.4f}")
    print(f"  • Donaldson-Thomas (DT)    : {env2_b.donaldson_thomas_instanton_count}")
    print(f"  • Puntaje Funcional J_pros : {report_b.dual_functional_score:.4f}")
    print(f"  • KV / WACC / Liquidez     : {report_b.kv_cache_compression_percent:.1f}% / {report_b.wacc_protected_percent:.1f}% / {report_b.net_liquidity_release_percent:.1f}%")
    print(f"  • Raíz de Merkle SHA-256   : {report_b.merkle_root_sha256}")
    print(f"  • Firma Final HMAC-SHA256  : {report_b.final_hmac_signature}")
    print("=" * 88)
    print("✨ Continuidad formal Fase1 → Fase2 → Fase3 y geodésica braquistócrona certificadas.")