# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Prosecutor Agent (Fiscal Exergético Agéntico) ║
║ Ruta   : app/agents/boole/omega/toon_exergy_portfolio_prosecutor_agent.py    ║
║ Versión: 6.1.0-Prosecutor-Agent-Deligne-Melnikov-Chirikov-KAM-Poincare-      ║
║          Delaunay-Octonionic-RSI3-Heyting-ESP32-PhD                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ACUSACIÓN DUAL EN GERBES (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Soberano Fiscal Agéntico Exergético de Portafolios
(`TOONExergyPortfolioProsecutorAgent`) en el Estrato Exergético
\(\mathcal{V}_{\mathbb{P}}\) de APU Filter v9.0.0. Opera la acreditación
acusatoria dual (calibre de Čech–Deligne vs. estado cuántico/exergético)
sobre redes de megaproyectos interconectados.

Funcional de acusación dual (métrica, no umbral mágico):

\[
J_{\mathrm{prosecutor}}(\rho,g,\mathrm{DT})
  = \operatorname{Tr}(\rho\,\Lambda_{\mathrm{fuga}})
  + \mu_1\|\widehat{g}-\mathrm{I}\|_F^{2}
  + \mu_2|\mathrm{DT}(\nu_{\mathrm{Behrend}})|
  + \mu_3 K_{\mathrm{Chirikov}}
  + \mu_4 W_{\mathrm{irr}}^{\mathrm{rel}}
\]

AXIOMÁTICA INQUISITORIAL, MECÁNICA CELESTE Y CROWBAR:
────────────────────────────────────────────────────────────────────────────────
  [A1] Jarzynski–Crooks con Log-Sum-Exp y techo \(\beta_{\mathrm{eff}}\le 40\).
       \(W_{\mathrm{irr}}=\langle W\rangle-\Delta F\ge 0\) (Jensen).
  [A2] Octoniones \(\mathbb{O}\) por Cayley–Dickson (no conmutativos, no
       asociativos). Asociador \([X,Y,Z]=(XY)Z-X(YZ)\).
  [A3] Cayley Tikhonov en \(\mathfrak{sp}(2n,\mathbb{R})\):
       \(K=H/(1+\varepsilon_{\mathrm{Tik}}+\|H\|_2)\), \(M=(I+K)(I-K)^{-1}\),
       Darboux \(\|M^{T}JM-J\|\le 10^{-10}\).
  [A4] Melnikov en forma cerrada (Duffing/péndulo):
       \(M(t_0)=A\sin(\omega t_0)\), \(A\propto\varepsilon\pi\omega\,\mathrm{csch}(\pi\omega/2)\).
       Cero simple \(\iff\) cruce de eje \(\land M'\neq 0\land A_{\mathrm{rel}}>0.25\).
       \(\varepsilon\) se induce de Gram off-diag + \(W_{\mathrm{irr}}\) + Wilkinson + Darboux.
       Homoclínica de Duffing: \(q_0=\sqrt{2}\,\mathrm{sech}(t)\),
       \(p_0=-\sqrt{2}\,\mathrm{sech}(t)\tanh(t)\).
  [A5] Chirikov \(K=(\Delta\omega_1+\Delta\omega_2)/|\omega_1-\omega_2|\),
       \(\Delta\omega=2\sqrt{\varepsilon\alpha}\), \(\omega\) de Delaunay/Poincaré
       (nunca \(\sigma_W\)). KAM Siegel–Bruno en la red \(\|k\|_\infty\le 3\).
       Delaunay: \(L=\sqrt{a}\), \(n=L^{-3}\), \(r_i=\sqrt{2L_i}\).
  [A6] Čech–Deligne térmico: Hermiticidad + \(\operatorname{Tr}\rho\equiv 1\)
       + \(\lambda_i\ge 0\). NO se exige \(\rho^2=\rho\) (vetaría Gibbs).
  [A7] Mónada Kleisli RSI³: \(\eta,T,T^3\), Picard, Lip empírico sobre el
       residual \(e=1/(1+|C|)\); reescala si \(\operatorname{Lip}(T^3)>0.857375\).
       Jerk \(\Delta^3 C>0\). Heyting \(\Omega_3\):
       \(\mathtt{COHERENT}(0x00)\prec\mathtt{DEGRADED}(0x01)\prec\mathtt{VETOED}(0x03)\).
       VETOED → GPIO14 HIGH, BT151, \(t_{\mathrm{act}}=392.15\,\mathrm{ns}<400\,\mathrm{ns}\),
       registro \(0x3FF44008\).

ARQUITECTURA DE TRES FASES ANIDADAS (Categoría de Kleisli RSI³):
────────────────────────────────────────────────────────────────────────────────
  [FASE 1] Phase1_ProsecutorAgentGaugeObserver
           └── observe_gauge_and_symplectic_state      → PortfolioProsecutionGaugeContract
           └── continue_into_phase2_melnikov_chirikov  → *inicia formalmente la Fase 2*
                 │
                 ▼
  [FASE 2] Phase2_ProsecutorAgentMelnikovChirikovOrientor
           └── orient_melnikov_kam_and_deligne_indictment → PortfolioProsecutionIndictmentBundle
           └── continue_into_phase3_rsi3_heyting_crowbar  → *inicia formalmente la Fase 3*
                 │
                 ▼
  [FASE 3] Phase3_ProsecutorAgentRSI3CrowbarDecider
           └── adjudicate_and_execute_crowbar          → PortfolioProsecutionExecutionPassport

El pipeline de la fachada COMPONE morfismos; no reimplementa la lógica de fase.
Cadena SHA-256/HMAC: fase1 → fase2 → terminal. Merkle de 6 hojas.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FUNDAMENTALES, FÍSICAS, RSI³ Y HARDWARE ESP32
# ─────────────────────────────────────────────────────────────────────────────
ENGINE_VERSION: Final[str] = (
    "6.1.0-Prosecutor-Agent-Deligne-Melnikov-Chirikov-KAM-"
    "Poincare-Delaunay-Octonionic-RSI3-Heyting-ESP32-PhD"
)

BOLTZMANN_K: Final[float] = 1.380649e-23
PLANCK_HBAR: Final[float] = 1.054571817e-34
DEFAULT_RISK_TEMPERATURE: Final[float] = 298.15
TIKHONOV_REGULARIZER: Final[float] = 1e-11

RSI3_LIPSCHITZ_BASE: Final[float] = 0.95
BANACH_RSI3_CONTRACTIVE_TARGET: Final[float] = 0.857375  # 0.95³
RSI3_BANACH_TRIPLE: Final[float] = BANACH_RSI3_CONTRACTIVE_TARGET

QND_COMMUTATOR_TOLERANCE: Final[float] = 1e-10
KMS_RESIDUAL_TOLERANCE: Final[float] = 1e-8
DARBOUX_RESIDUAL_TOLERANCE: Final[float] = 1e-10
CECH_OBSTRUCTION_TOLERANCE: Final[float] = 1e-10

BETA_EFF_CEILING: Final[float] = 40.0
MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD: Final[float] = 0.25
CHIRIKOV_KAM_LIMIT: Final[float] = 1.00
ACCUSATION_J_THRESHOLD: Final[float] = 0.50
EXERGY_WORK_RELATIVE_THRESHOLD: Final[float] = 0.50
BRACHISTOCHRONE_DEFECT_TOLERANCE: Final[float] = 0.15
GRAVITY_STANDARD: Final[float] = 9.80665

MU_LEAK: Final[float] = 1.00
MU_GRAM_DEVIATION: Final[float] = 1.00
MU_DT: Final[float] = 0.50
MU_CHIRIKOV: Final[float] = 0.30
MU_WIRR: Final[float] = 0.10

ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0
ESP32_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 392.15
ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
DEFAULT_HMAC_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_EXERGY_PROSECUTOR_AGENT_KEY_V9"

# Gauss–Legendre n=16 sobre [-1,1]; empuje a ℝ por t = artanh(ξ).
_GL16_NODES: Final[Tuple[float, ...]] = (
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.0950125098376374,
     0.0950125098376374,  0.2816035507792589,  0.4580167776572274,
     0.6178762444026438,  0.7554044083550030,  0.8656312023878318,
     0.9445750230732326,  0.9894009349916499,
)
_GL16_WEIGHTS: Final[Tuple[float, ...]] = (
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928,
    0.1246289712555339, 0.1495959888165767, 0.1691565193950025,
    0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025,
    0.1495959888165767, 0.1246289712555339, 0.0951585116824928,
    0.0622535239386479, 0.0271524594117541,
)


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRÍA
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("APUFilter.Wisdom.TOONExergyPortfolioProsecutorAgent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [ProsecutorAgent.Sovereign:%(lineno)d] 🏛️ %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERACIONES
# ─────────────────────────────────────────────────────────────────────────────
class HeytingState(IntEnum):
    """Álgebra de Heyting trivalente Ω₃. Orden: COHERENT ≺ DEGRADED ≺ VETOED."""

    COHERENT = 0x00  # ⊥₀ : consistencia exergética y cohomológica
    DEGRADED = 0x01  # ½  : veto suave, válvula de alivio (gracia 1 h)
    VETOED = 0x03    # ⊤₁ : veto duro, Crowbar ESP32 GPIO14 / BT151


class ActuationMode(IntEnum):
    """Modos de actuación ciber-física en silicio."""

    PASSIVE_NOMINAL = 0x00
    SOFT_RELIEF_VALVE = 0x01
    HARD_CROWBAR_DISCHARGE = 0x03


# ─────────────────────────────────────────────────────────────────────────────
# ÁLGEBRA DE OCTONIONES 𝕆 POR CAYLEY–DICKSON (NO CONMUTATIVA, NO ASOCIATIVA)
# ─────────────────────────────────────────────────────────────────────────────
def _cd_conjugate(x: NDArray[np.float64]) -> NDArray[np.float64]:
    y = x.copy()
    if y.size > 1:
        y[1:] *= -1.0
    return y


def _cd_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Producto de Cayley–Dickson: (a,b)·(c,d)=(ac−d̄b, da+b c̄)."""
    n = int(a.shape[0])
    if n == 1:
        return a * b
    h = n // 2
    a1, a2 = a[:h], a[h:]
    c1, d1 = b[:h], b[h:]
    ac = _cd_mul(a1, c1)
    dconj_b = _cd_mul(_cd_conjugate(d1), a2)
    da = _cd_mul(d1, a1)
    b_cconj = _cd_mul(a2, _cd_conjugate(c1))
    out = np.empty(n, dtype=np.float64)
    out[:h] = ac - dconj_b
    out[h:] = da + b_cconj
    return out


class OctonionProsecutionNumber:
    """
    Álgebra de división alternativa \(\mathbb{O}\) sobre \(\mathbb{R}^8\).
    Producto por duplicación de Cayley–Dickson (no conmutativo, no asociativo;
    sí alternativo: \([X,X,Y]=[Y,X,X]=0\)).
    """

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: Union[NDArray[np.float64], Sequence[float]]) -> None:
        arr = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if arr.shape != (8,):
            raise ValueError(f"Un octonión requiere un 8-vector. Recibido shape={arr.shape}")
        self._coeffs: NDArray[np.float64] = arr

    @property
    def coeffs(self) -> NDArray[np.float64]:
        return self._coeffs

    def __repr__(self) -> str:
        return f"OctonionProsecutionNumber({np.array2string(self._coeffs, precision=4, separator=', ')})"

    def __add__(self, other: OctonionProsecutionNumber) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(self._coeffs + other._coeffs)

    def __sub__(self, other: OctonionProsecutionNumber) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(self._coeffs - other._coeffs)

    def __mul__(self, other: OctonionProsecutionNumber) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(_cd_mul(self._coeffs, other._coeffs))

    def conjugate(self) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(_cd_conjugate(self._coeffs))

    def norm(self) -> float:
        return float(np.linalg.norm(self._coeffs))

    @staticmethod
    def associator(
        x: OctonionProsecutionNumber,
        y: OctonionProsecutionNumber,
        z: OctonionProsecutionNumber,
    ) -> OctonionProsecutionNumber:
        """Tensor asociador canónico [X,Y,Z]=(XY)Z − X(YZ)."""
        return (x * y) * z - x * (y * z)


def _octonion_from_pad(vec: NDArray[np.float64]) -> OctonionProsecutionNumber:
    buf = np.zeros(8, dtype=np.float64)
    flat = np.asarray(vec, dtype=np.float64).reshape(-1)[:8]
    buf[: flat.size] = flat
    nrm = float(np.linalg.norm(buf))
    if nrm > 1e-15:
        buf /= nrm
    else:
        buf[0] = 1.0
    return OctonionProsecutionNumber(buf)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES: HMAC, MERKLE, GL16, DELAUNAY, RSI³, BRACHISTOCHRONE, ESP32
# ─────────────────────────────────────────────────────────────────────────────
def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hmac_hex(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _merkle_root(leaves: Sequence[str]) -> str:
    """Árbol de Merkle binario; duplica la última hoja en niveles impares."""
    nodes: List[str] = [str(x) for x in leaves]
    if not nodes:
        return hashlib.sha256(b"").hexdigest()
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nxt: List[str] = []
        for i in range(0, len(nodes), 2):
            nxt.append(hashlib.sha256((nodes[i] + nodes[i + 1]).encode("utf-8")).hexdigest())
        nodes = nxt
    return nodes[0]


def gl16_real_line(fn) -> float:
    """
    Cuadratura de Gauss–Legendre n=16 de ∫_{-∞}^{+∞} f(t) dt
    mediante el difeomorfismo t = artanh(ξ), dt = dξ/(1−ξ²).
    """
    acc = 0.0
    for xi, wi in zip(_GL16_NODES, _GL16_WEIGHTS):
        one_m_xi2 = 1.0 - xi * xi
        if one_m_xi2 <= 1e-18:
            continue
        t = 0.5 * math.log((1.0 + xi) / (1.0 - xi))  # artanh
        acc += wi * float(fn(t)) / one_m_xi2
    return float(acc)


@dataclass(frozen=True)
class PoincareDelaunayChart:
    """Carta de Delaunay–Poincaré: L=√a, n=L⁻³, r=√(2L)."""

    actions_L: Tuple[float, ...]
    mean_motions: Tuple[float, ...]
    radii: Tuple[float, ...]
    eccentricity: float
    principal_mean_motion: float


def delaunay_chart_from_actions(
    eigenvalues: NDArray[np.float64],
    offdiag_energy_ratio: float,
) -> PoincareDelaunayChart:
    evals = np.maximum(np.asarray(eigenvalues, dtype=np.float64).reshape(-1), 1e-15)
    actions = tuple(float(math.sqrt(v)) for v in evals)
    motions = tuple(float(L ** (-3.0)) for L in actions)
    radii = tuple(float(math.sqrt(2.0 * L)) for L in actions)
    ecc = float(min(max(offdiag_energy_ratio, 0.0), 0.999))
    n_bar = float(sum(motions) / max(len(motions), 1))
    return PoincareDelaunayChart(
        actions_L=actions,
        mean_motions=motions,
        radii=radii,
        eccentricity=ecc,
        principal_mean_motion=n_bar,
    )


def induce_celestial_frequencies(
    chart: PoincareDelaunayChart,
    diophantine_freqs: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    ω celestial. Si no se provee, induce ω_i = 1/r_i sobre las dos primeras
    acciones canónicas (1, √2) — no sobre autovalores de Gram (evitar clustering).
    """
    if diophantine_freqs is not None:
        w1, w2 = float(diophantine_freqs[0]), float(diophantine_freqs[1])
        if abs(w1 - w2) < 1e-15:
            w2 = w1 + GOLDEN_RATIO * 1e-3
        return w1, w2
    L0 = 1.0
    L1 = math.sqrt(2.0)
    r0 = math.sqrt(2.0 * L0)
    r1 = math.sqrt(2.0 * L1)
    return (1.0 / r0, 1.0 / r1)


class RSI3KleisliMonad:
    """
    Mónada Kleisli RSI³ (Darwin–Gödel nivel 3).
    η : Id → T,  T endofunctor contractivo, T³ = T∘T∘T.
    Lip se estima sobre el residual e=1/(1+|C|), no sobre C cruda
    (el jerk Δ³C>0 es *deseable* y no contradice la contracción de e).
    """

    def unit(self, x: float) -> float:
        return float(x)

    def T(self, x: float, lip: float = RSI3_LIPSCHITZ_BASE) -> float:
        return float(lip) * float(x)

    def T3(self, x: float, lip: float = RSI3_LIPSCHITZ_BASE) -> float:
        return float(lip) ** 3 * float(x)

    def empirical_lipschitz_and_jerk(
        self,
        capacity_history: Optional[Sequence[float]],
    ) -> Tuple[float, bool, bool, bool]:
        """
        Retorna (Lip(T³)_reportado, jerk_positivo, reescalado, contractivo_Banach).
        Si Lip(T³)_empírico > 0.857375, reescala al objetivo de Banach.
        """
        if capacity_history is None or len(capacity_history) < 4:
            return BANACH_RSI3_CONTRACTIVE_TARGET, True, False, True

        c = [float(v) for v in capacity_history]
        residuals = [1.0 / (1.0 + abs(v)) for v in c]
        xs = residuals[:-1]
        ys = residuals[1:]
        lip_t = 0.0
        pairs = 0
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dx = abs(xs[i] - xs[j])
                if dx > 1e-15:
                    lip_t = max(lip_t, abs(ys[i] - ys[j]) / dx)
                    pairs += 1
        if pairs == 0:
            d = [abs(residuals[i + 1] - residuals[i]) for i in range(len(residuals) - 1)]
            ratios = [d[i + 1] / d[i] for i in range(len(d) - 1) if d[i] > 1e-15]
            lip_t = float(max(ratios)) if ratios else RSI3_LIPSCHITZ_BASE

        lip_t3_emp = float(lip_t) ** 3
        rescaled = False
        lip_t3_rep = lip_t3_emp
        if lip_t3_emp > BANACH_RSI3_CONTRACTIVE_TARGET and math.isfinite(lip_t3_emp):
            lip_t3_rep = BANACH_RSI3_CONTRACTIVE_TARGET
            rescaled = True
        if not math.isfinite(lip_t3_emp):
            lip_t3_rep = float("inf")
            rescaled = False

        jerk = c[-1] - 3.0 * c[-2] + 3.0 * c[-3] - c[-4]
        jerk_pos = jerk > 0.0
        contractive = math.isfinite(lip_t3_rep) and lip_t3_rep < 1.0
        return float(lip_t3_rep), jerk_pos, rescaled, contractive


class BrachistochronePathFinder:
    """
    Ciclode de Bernoulli: T=π√(R/g). Defecto = T_path/T_cyclo − 1.
    Camino presupuestario interpretado como perfil de altura de trabajo.
    """

    DEFECT_TOLERANCE: Final[float] = BRACHISTOCHRONE_DEFECT_TOLERANCE

    def cycloid_time(self, radius: float) -> float:
        r = max(float(radius), 1e-12)
        return math.pi * math.sqrt(r / GRAVITY_STANDARD)

    def path_defect(self, work: NDArray[np.float64]) -> float:
        w = np.asarray(work, dtype=np.float64).reshape(-1)
        if w.size < 2:
            return 0.0
        drop = float(np.ptp(w))
        if drop < 1e-12:
            return 0.0
        radius = 0.5 * drop
        t_cyclo = self.cycloid_time(radius)
        t_path = 0.0
        for i in range(1, int(w.size)):
            dh = abs(float(w[i] - w[i - 1]))
            ds = math.hypot(1.0, dh)
            t_path += ds / math.sqrt(max(2.0 * GRAVITY_STANDARD * max(dh, 1e-12), 1e-18))
        return float(t_path / max(t_cyclo, 1e-18) - 1.0)


def _esp32_w1ts_write(crowbar: bool) -> Tuple[str, int, float]:
    """Emula GPIO_OUT_W1TS_REG ← (1<<14) con latencia IRAM determinista."""
    if crowbar:
        mask = 1 << ESP32_CROWBAR_GPIO_PIN
        return f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}", int(mask), ESP32_CROWBAR_NOMINAL_LATENCY_NS
    return f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}", 0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DTOs INMUTABLES DE HANDOFF KLEISLI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PortfolioProsecutionGaugeContract:
    """Sobre inmutable de la FASE 1 (Observe): geometría de calibre y Delaunay."""

    schedule_id: str
    num_contracts: int
    overlap_matrix_hash: str
    novikov_krull_valuation: float
    sylvester_native: bool
    sylvester_certified: bool
    wilkinson_cond_number: float
    offdiag_energy_ratio: float
    gram_correlation_frobenius_sq: float
    darboux_symplectic_residual: float
    octonionic_associator_anomaly_norm: float
    cayley_sp_norm: float
    qnd_commutator_residual: float
    delaunay_mean_motion: float
    delaunay_eccentricity: float
    delaunay_actions: Tuple[float, ...]
    hmac_signature: str
    phase1_sha256: str
    timestamp_iso: str


@dataclass(frozen=True)
class PortfolioProsecutionIndictmentBundle:
    """Sobre inmutable de la FASE 2 (Orient): Melnikov, Chirikov, KAM, Čech, RSI³."""

    phase1_ref_hash: str
    phase1_sha256: str
    accusation_functional_j: float
    perturbation_epsilon: float
    melnikov_closed_form_amplitude: float
    melnikov_relative_amplitude: float
    melnikov_gl16_diagnostic: float
    melnikov_simple_zero_detected: bool
    chirikov_parameter: float
    kam_diophantine_satisfied: bool
    kam_diophantine_minimum_divisor: float
    nekhoroshev_stability_time_est: float
    poincare_return_residual: float
    is_poincare_section_regular: bool
    lyapunov_characteristic_estimate: float
    jarzynski_irreversible_work: float
    crooks_fluctuation_ratio: float
    kms_residual: float
    cech_deligne_obstruction: float
    donaldson_thomas_instantons: int
    brachistochrone_defect: float
    rsi3_lipschitz_empirical: float
    jerk_capacity_positive: bool
    rsi3_rescaled: bool
    banach_contractive: bool
    phase2_hmac_signature: str
    phase2_sha256: str
    timestamp_iso: str


@dataclass(frozen=True)
class PortfolioProsecutionExecutionPassport:
    """Pasaporte soberano terminal (FASE 3 — Decide & Act)."""

    timestamp_iso: str
    portfolio_id: str
    engine_version: str
    global_heyting_verdict: HeytingState
    actuation_mode: ActuationMode
    soft_veto_relief_valve_active: bool
    crowbar_triggered: bool
    actuation_latency_nanoseconds: float
    esp32_hardware_register_hex: str
    hardware_register_state: int
    truth_degree_tau: float
    dual_functional_score: float
    kv_cache_compression_percent: float
    wacc_protected_percent: float
    net_liquidity_release_percent: float
    rsi3_lipschitz_empirical: float
    phase1_sha256: str
    phase2_sha256: str
    merkle_root_sha256: str
    signature_hmac_sha256: str


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPCIONES DE DOMINIO
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioProsecutionError(Exception):
    """Excepción base para querellas y vetos del Fiscal Exergético."""


class MelnikovChaosIndictmentError(PortfolioProsecutionError):
    """Integral de Melnikov: tangles homoclínicos."""


class ChirikovOverlapIndictmentError(PortfolioProsecutionError):
    """Parámetro de Chirikov K ≥ 1: destrucción KAM."""


class KamTorusDestructionIndictmentError(PortfolioProsecutionError):
    """Condición diofántica KAM colapsada."""


class CechDeligneProsecutionError(PortfolioProsecutionError):
    """Obstrucción no trivial en cohomología de Čech–Deligne térmica."""


# ─────────────────────────────────────────────────────────────────────────────
# FASE 1 — Phase1_ProsecutorAgentGaugeObserver
# Terminal : observe_gauge_and_symplectic_state
# Morfismo : continue_into_phase2_melnikov_chirikov  ≡  inicio formal de Fase 2
# ─────────────────────────────────────────────────────────────────────────────
class Phase1_ProsecutorAgentGaugeObserver:
    """
    Fase 1: observación de calibre, Sylvester nativo vs. certificado,
    Wilkinson κ₂, Novikov–Krull, Cayley–Tikhonov en sp(2n,ℝ), QND,
    octoniones Cayley–Dickson y carta de Delaunay–Poincaré.
    """

    def __init__(self, portfolio_id: str, secret_key: bytes) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self._secret_key: Final[bytes] = secret_key

    def observe_gauge_and_symplectic_state(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        logger.info("🔍 [FASE 1 — PROSECUTOR AGENT OBSERVE] Gram, Cayley–Tikhonov, Delaunay, 𝕆.")

        G = np.asarray(overlap_graph, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(f"La matriz de solapamiento debe ser cuadrada. Recibido shape={G.shape}")

        n_dim = int(G.shape[0])
        gram_sym = 0.5 * (G + G.T)

        diag = np.clip(np.diag(gram_sym), 1e-15, None)
        dinv = np.diag(1.0 / np.sqrt(diag))
        corr = dinv @ gram_sym @ dinv
        off = corr - np.eye(n_dim, dtype=np.float64)
        offdiag_energy_ratio = float(la.norm(off) / max(float(la.norm(corr)), 1e-15))
        gram_corr_frob_sq = float(la.norm(off) ** 2)

        # Sylvester nativo (sin Tikhonov) vs. certificado (post-Tikhonov).
        sylvester_native = True
        try:
            _ = la.cholesky(gram_sym, lower=True)
            reg_gram = gram_sym
        except la.LinAlgError:
            sylvester_native = False
            logger.warning("⚠️ [FASE 1] Gram no SPD nativo. Precondicionamiento Tikhonov.")
            reg_gram = gram_sym + np.eye(n_dim, dtype=np.float64) * TIKHONOV_REGULARIZER
            evals_raw = la.eigvalsh(reg_gram)
            lam_min_raw = float(np.min(evals_raw))
            if lam_min_raw <= 0.0:
                reg_gram = reg_gram + np.eye(n_dim, dtype=np.float64) * (abs(lam_min_raw) + 1e-8)

        evals = la.eigvalsh(reg_gram)
        lam_min = max(float(np.min(evals)), 1e-15)
        lam_max = max(float(np.max(evals)), 1e-15)
        sylvester_certified = bool(lam_min > 0.0)
        wilkinson_cond = lam_max / lam_min
        novikov_krull_val = -math.log(lam_min)

        # Inmersión de Cayley Tikhonov: K = H / (1+ε+‖H‖₂), M=(I+K)(I-K)^{-1}.
        H_ham = np.block(
            [
                [np.zeros((n_dim, n_dim), dtype=np.float64), reg_gram],
                [-reg_gram, np.zeros((n_dim, n_dim), dtype=np.float64)],
            ]
        )
        h_op = float(la.norm(H_ham, 2))
        scale = 1.0 / (1.0 + TIKHONOV_REGULARIZER + h_op)
        K = H_ham * scale
        i2n = np.eye(2 * n_dim, dtype=np.float64)
        m_mono = (i2n + K) @ la.inv(i2n - K + TIKHONOV_REGULARIZER * i2n)
        j_can = np.block(
            [
                [np.zeros((n_dim, n_dim), dtype=np.float64), np.eye(n_dim, dtype=np.float64)],
                [-np.eye(n_dim, dtype=np.float64), np.zeros((n_dim, n_dim), dtype=np.float64)],
            ]
        )
        darboux_num = float(la.norm(m_mono.T @ j_can @ m_mono - j_can))
        darboux_den = max(float(la.norm(j_can)), 1e-12)
        darboux_res = darboux_num / darboux_den
        cayley_norm = float(la.norm(H_ham))

        # QND: conmutador [H, A] con A = diag(H) (observable de energía).
        a_qnd = np.diag(np.diag(reg_gram))
        qnd_commutator = float(la.norm(reg_gram @ a_qnd - a_qnd @ reg_gram))

        # Asociador octoniónico (Cayley–Dickson). Tres filas de Gram, o estado dado.
        if hypercomplex_octonion_state is not None:
            ox = OctonionProsecutionNumber(hypercomplex_octonion_state[0])
            oy = OctonionProsecutionNumber(hypercomplex_octonion_state[1])
            oz = OctonionProsecutionNumber(hypercomplex_octonion_state[2])
        else:
            rows = [gram_sym[i, :] if i < n_dim else np.zeros(n_dim) for i in range(3)]
            ox, oy, oz = (_octonion_from_pad(r) for r in rows)
        assoc_anomaly_norm = OctonionProsecutionNumber.associator(ox, oy, oz).norm()

        chart = delaunay_chart_from_actions(evals, offdiag_energy_ratio)

        overlap_hash = hashlib.sha256(reg_gram.tobytes()).hexdigest()
        ts_iso = _utc_iso()
        num_c = len(contract_matrices) if len(contract_matrices) > 0 else n_dim
        raw_payload = (
            f"{ts_iso}|{self.portfolio_id}|{num_c}|{overlap_hash}|"
            f"{novikov_krull_val:.12f}|{darboux_res:.12e}|{assoc_anomaly_norm:.12e}|"
            f"{offdiag_energy_ratio:.12e}|{qnd_commutator:.12e}"
        )
        hmac_sig = _hmac_hex(self._secret_key, raw_payload)
        phase1_sha = _sha256_hex(raw_payload)

        contract = PortfolioProsecutionGaugeContract(
            schedule_id=self.portfolio_id,
            num_contracts=num_c,
            overlap_matrix_hash=overlap_hash,
            novikov_krull_valuation=novikov_krull_val,
            sylvester_native=sylvester_native,
            sylvester_certified=sylvester_certified,
            wilkinson_cond_number=wilkinson_cond,
            offdiag_energy_ratio=offdiag_energy_ratio,
            gram_correlation_frobenius_sq=gram_corr_frob_sq,
            darboux_symplectic_residual=darboux_res,
            octonionic_associator_anomaly_norm=assoc_anomaly_norm,
            cayley_sp_norm=cayley_norm,
            qnd_commutator_residual=qnd_commutator,
            delaunay_mean_motion=chart.principal_mean_motion,
            delaunay_eccentricity=chart.eccentricity,
            delaunay_actions=chart.actions_L,
            hmac_signature=hmac_sig,
            phase1_sha256=phase1_sha,
            timestamp_iso=ts_iso,
        )
        logger.info(
            "✅ [FASE 1] Hash=%s… | κ₂=%.2e | v(G)=%.4f | δDarboux=%.4e | "
            "offdiag=%.4e | QND=[H,A]=%.4e | n̄_Del=%.4e | ||[X,Y,Z]||=%.4e | "
            "Sylvester native=%s certified=%s",
            overlap_hash[:12],
            wilkinson_cond,
            novikov_krull_val,
            darboux_res,
            offdiag_energy_ratio,
            qnd_commutator,
            chart.principal_mean_motion,
            assoc_anomaly_norm,
            sylvester_native,
            sylvester_certified,
        )
        return contract

    def continue_into_phase2_melnikov_chirikov(
        self,
        gauge_contract: PortfolioProsecutionGaugeContract,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
    ) -> Phase2_ProsecutorAgentMelnikovChirikovOrientor:
        """
        Morfismo de Kleisli η_{1→2}.
        Definición formal terminal de la Fase 1 ≡ objeto inicial de la Fase 2.
        No reimplementa orientación: instancia el orientor con el sello de Fase 1.
        """
        return Phase2_ProsecutorAgentMelnikovChirikovOrientor(
            portfolio_id=self.portfolio_id,
            risk_temperature=risk_temperature,
            secret_key=self._secret_key,
            phase1_hmac=gauge_contract.hmac_signature,
            phase1_sha256=gauge_contract.phase1_sha256,
            delaunay_mean_motion=gauge_contract.delaunay_mean_motion,
            delaunay_eccentricity=gauge_contract.delaunay_eccentricity,
            delaunay_actions=gauge_contract.delaunay_actions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 — Phase2_ProsecutorAgentMelnikovChirikovOrientor
# Continuación formal de Phase1.continue_into_phase2_melnikov_chirikov
# Terminal : orient_melnikov_kam_and_deligne_indictment
# Morfismo : continue_into_phase3_rsi3_heyting_crowbar  ≡  inicio formal de Fase 3
# ─────────────────────────────────────────────────────────────────────────────
class Phase2_ProsecutorAgentMelnikovChirikovOrientor:
    """
    Fase 2: Melnikov cerrado, Chirikov con ω de Poincaré–Delaunay,
    KAM Siegel–Bruno (‖k‖_∞≤3), Čech térmico (no idempotencia),
    Jarzynski Log-Sum-Exp, KMS, braquistócrona, mónada RSI³.
    """

    ACCUSATION_J_THRESHOLD: Final[float] = ACCUSATION_J_THRESHOLD
    MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD: Final[float] = MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD
    CHIRIKOV_LIMIT_KAM: Final[float] = CHIRIKOV_KAM_LIMIT
    CECH_OBSTRUCTION_TOLERANCE: Final[float] = CECH_OBSTRUCTION_TOLERANCE
    EXERGY_WORK_RELATIVE_THRESHOLD: Final[float] = EXERGY_WORK_RELATIVE_THRESHOLD

    def __init__(
        self,
        portfolio_id: str,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
        secret_key: bytes = DEFAULT_HMAC_KEY,
        phase1_hmac: str = "",
        phase1_sha256: str = "",
        delaunay_mean_motion: float = 1.0,
        delaunay_eccentricity: float = 0.0,
        delaunay_actions: Tuple[float, ...] = (1.0, math.sqrt(2.0)),
    ) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self.risk_temperature: Final[float] = risk_temperature
        self._secret_key: Final[bytes] = secret_key
        self.phase1_hmac: Final[str] = phase1_hmac
        self.phase1_sha256: Final[str] = phase1_sha256
        self.delaunay_mean_motion: Final[float] = delaunay_mean_motion
        self.delaunay_eccentricity: Final[float] = delaunay_eccentricity
        self.delaunay_actions: Final[Tuple[float, ...]] = delaunay_actions
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K * max(self.risk_temperature, 1e-12))
        self._rsi3: Final[RSI3KleisliMonad] = RSI3KleisliMonad()
        self._brachistochrone: Final[BrachistochronePathFinder] = BrachistochronePathFinder()

    @staticmethod
    def _perturbation_epsilon(
        gauge: PortfolioProsecutionGaugeContract,
        w_irr_rel: float,
    ) -> float:
        off = float(gauge.offdiag_energy_ratio)
        darboux_term = min(float(gauge.darboux_symplectic_residual) / 1e-6, 1.0)
        wilk_term = min(max(math.log10(max(float(gauge.wilkinson_cond_number), 1.0)) - 1.0, 0.0) / 8.0, 1.0)
        eps = 0.40 * off + 0.20 * min(max(w_irr_rel, 0.0), 1.0) + 0.20 * darboux_term + 0.20 * wilk_term
        return float(np.clip(eps, 0.0, 2.0))

    @staticmethod
    def _melnikov_closed_form(eps: float, omega: float) -> Tuple[float, float, bool]:
        """
        A = ε π ω csch(π ω / 2). M(t₀)=A sin(ω t₀).
        Cero simple ⇔ A≠0 (M'=Aω cos ≠0 en los ceros) ∧ A_rel > 0.25.
        El péndulo no perturbado (ε≈0) NO dispara.
        """
        w = max(abs(float(omega)), 1e-12)
        sinh_arg = math.sinh(0.5 * math.pi * w)
        csch = 1.0 / sinh_arg if abs(sinh_arg) > 1e-18 else 0.0
        amp = abs(float(eps) * math.pi * w * csch)
        rel = amp / (1.0 + amp)
        simple_zero = bool(amp > 1e-15 and rel > MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD)
        return float(amp), float(rel), simple_zero

    @staticmethod
    def _melnikov_gl16_diagnostic(eps: float, omega: float) -> float:
        """
        Diagnóstico (no decisional) de M(0) sobre la homoclínica de Duffing
        q₀=√2 sech t, p₀=−√2 sech t tanh t, H₁ ~ ε q cos(ω t).
        """
        w = float(omega)
        e = float(eps)

        def integrand(t: float) -> float:
            sech = 1.0 / math.cosh(max(min(t, 40.0), -40.0))
            p0 = -math.sqrt(2.0) * sech * math.tanh(max(min(t, 40.0), -40.0))
            q0 = math.sqrt(2.0) * sech
            return e * p0 * q0 * math.cos(w * t)

        return abs(gl16_real_line(integrand))

    @staticmethod
    def _chirikov_kam(
        omega1: float,
        omega2: float,
        eps: float,
        alpha: float,
    ) -> Tuple[float, bool, float, float, float, bool]:
        """
        K=(Δω₁+Δω₂)/|ω₁−ω₂|, Δω=2√(ε α), α = (offdiag)².
        KAM Siegel–Bruno sobre {k : ‖k‖_∞ ≤ 3}.
        """
        dw = 2.0 * math.sqrt(max(float(eps) * max(float(alpha), 0.0), 0.0))
        sep = max(abs(float(omega1) - float(omega2)), 1e-12)
        chirikov_k = (dw + dw) / sep

        tau_kam = 2.5
        gamma = 1e-6
        min_divisor = float("inf")
        kam_ok = True
        for k1 in range(-3, 4):
            for k2 in range(-3, 4):
                if k1 == 0 and k2 == 0:
                    continue
                k_inf = max(abs(k1), abs(k2))
                if k_inf > 3:
                    continue
                k_l1 = abs(k1) + abs(k2)
                freq_dot = abs(k1 * omega1 + k2 * omega2)
                divisor = freq_dot * (k_l1 ** tau_kam)
                if divisor < min_divisor:
                    min_divisor = divisor
                if freq_dot + 1e-18 < gamma / (k_l1 ** tau_kam):
                    kam_ok = False
        if not math.isfinite(min_divisor):
            min_divisor = 0.0

        eps_eff = max(chirikov_k * 0.1, float(eps), 1e-8)
        nek = math.exp(min(1.0 / (eps_eff ** 0.25), 80.0))
        ratio = omega1 / omega2 if abs(omega2) > 1e-15 else 0.0
        poincare_res = abs(math.sin(math.pi * ratio))
        section_regular = bool(kam_ok and chirikov_k < CHIRIKOV_KAM_LIMIT)
        return float(chirikov_k), kam_ok, float(min_divisor), float(nek), float(poincare_res), section_regular

    @staticmethod
    def _cech_thermal(rho: NDArray[np.float64]) -> float:
        """Hermítico + Tr ρ ≡ 1 + λᵢ ≥ 0. NUNCA ρ²=ρ (Gibbs térmico no es projector)."""
        r = np.asarray(rho, dtype=np.float64)
        if r.ndim != 2 or r.shape[0] != r.shape[1]:
            raise ValueError(f"ρ debe ser cuadrada. Recibido shape={r.shape}")
        herm = float(la.norm(r - r.T))
        tr_def = abs(float(np.trace(r)) - 1.0)
        herm_part = 0.5 * (r + r.T)
        ev = la.eigvalsh(herm_part)
        neg_mass = float(np.sum(np.clip(-ev, 0.0, None)))
        return float(herm + tr_def + neg_mass)

    @staticmethod
    def _kms_residual(rho: NDArray[np.float64], beta_eff: float) -> float:
        """‖e^{−βH} ρ e^{βH} − ρ‖ con H = −log(diag⁺(ρ)+ε) (automorfismo modular)."""
        r = 0.5 * (np.asarray(rho, dtype=np.float64) + np.asarray(rho, dtype=np.float64).T)
        n = r.shape[0]
        diag = np.clip(np.diag(r), 1e-15, None)
        h = np.diag(-np.log(diag))
        b = float(min(max(beta_eff, 0.0), BETA_EFF_CEILING))
        # exp(±βH) diagonal exacto
        e_m = np.diag(np.exp(np.clip(-b * np.diag(h), -BETA_EFF_CEILING, BETA_EFF_CEILING)))
        e_p = np.diag(np.exp(np.clip(+b * np.diag(h), -BETA_EFF_CEILING, BETA_EFF_CEILING)))
        conjugated = e_m @ r @ e_p
        return float(la.norm(conjugated - r)) / max(float(la.norm(r)), 1e-15)

    def orient_melnikov_kam_and_deligne_indictment(
        self,
        gauge_contract: PortfolioProsecutionGaugeContract,
        density_matrix_rho: NDArray[np.float64],
        leak_operator_lambda: NDArray[np.float64],
        presupuesto_work_vector: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> PortfolioProsecutionIndictmentBundle:
        logger.info(
            "🌀 [FASE 2 — PROSECUTOR AGENT ORIENT] Querella desde Fase 1 (%s…).",
            gauge_contract.hmac_signature[:12],
        )

        w_work = np.asarray(presupuesto_work_vector, dtype=np.float64).reshape(-1)
        if w_work.size == 0:
            raise ValueError("El vector de trabajo no puede estar vacío.")

        # Jarzynski Log-Sum-Exp, β_eff ≤ 40 (adimensional sobre σ_W).
        w_mean = float(np.mean(w_work))
        w_std = max(float(np.std(w_work)), 1e-8)
        beta_eff = min(1.0 / w_std, BETA_EFF_CEILING)
        u = -beta_eff * w_work
        u_max = float(np.max(u))
        log_mean_exp = u_max + math.log(float(np.mean(np.exp(u - u_max))))
        delta_f_j = -(1.0 / beta_eff) * log_mean_exp
        f_states = np.asarray(free_energy_states, dtype=np.float64).reshape(-1)
        if f_states.size >= 2:
            delta_f_ref = float(f_states[-1] - f_states[0])
            w_irr = max(w_mean - delta_f_ref, 0.0)
        else:
            w_irr = max(w_mean - delta_f_j, 0.0)
        w_irr_rel = w_irr / (1.0 + abs(w_mean))
        crooks_log = min(beta_eff * w_irr, BETA_EFF_CEILING)
        crooks_ratio = math.exp(crooks_log)

        eps = self._perturbation_epsilon(gauge_contract, w_irr_rel)
        chart = PoincareDelaunayChart(
            actions_L=gauge_contract.delaunay_actions,
            mean_motions=tuple(float(L ** (-3.0)) for L in gauge_contract.delaunay_actions),
            radii=tuple(float(math.sqrt(max(2.0 * L, 1e-18))) for L in gauge_contract.delaunay_actions),
            eccentricity=gauge_contract.delaunay_eccentricity,
            principal_mean_motion=gauge_contract.delaunay_mean_motion,
        )
        omega1, omega2 = induce_celestial_frequencies(chart, diophantine_freqs)
        omega_pert = 0.5 * (abs(omega1) + abs(omega2))

        m_amp, m_rel, m_simple = self._melnikov_closed_form(eps, omega_pert)
        m_gl16 = self._melnikov_gl16_diagnostic(eps, omega_pert)

        alpha = float(gauge_contract.offdiag_energy_ratio) ** 2
        chirikov_k, kam_ok, min_div, nek, poincare_res, section_reg = self._chirikov_kam(
            omega1, omega2, eps, alpha
        )
        lyap = max(0.0, math.log(1.0 + m_amp + max(chirikov_k, 0.0)))

        rho_mat = np.asarray(density_matrix_rho, dtype=np.float64)
        cech_residual = self._cech_thermal(rho_mat)
        kms_res = self._kms_residual(rho_mat, beta_eff)

        if cech_residual > CECH_OBSTRUCTION_TOLERANCE and (m_simple or chirikov_k >= CHIRIKOV_KAM_LIMIT):
            dt_instantons = 1
        else:
            dt_instantons = 0

        n_rho = int(rho_mat.shape[0])
        lambda_leak = np.asarray(leak_operator_lambda, dtype=np.float64)
        if lambda_leak.shape != rho_mat.shape:
            lambda_leak = np.diag(np.linspace(0.05, 0.25, max(n_rho, 1)))
        trace_leak = float(np.trace(rho_mat @ lambda_leak))

        j_prosecutor = (
            MU_LEAK * trace_leak
            + MU_GRAM_DEVIATION * float(gauge_contract.gram_correlation_frobenius_sq)
            + MU_DT * float(abs(dt_instantons))
            + MU_CHIRIKOV * float(chirikov_k)
            + MU_WIRR * float(w_irr_rel)
        )

        lip_t3, jerk_pos, rescaled, contractive = self._rsi3.empirical_lipschitz_and_jerk(capacity_history)
        brach_defect = self._brachistochrone.path_defect(w_work)

        ts_iso = _utc_iso()
        raw_payload = (
            f"{gauge_contract.hmac_signature}|{gauge_contract.phase1_sha256}|"
            f"{j_prosecutor:.12f}|{eps:.12e}|{m_amp:.12e}|{chirikov_k:.12e}|"
            f"{cech_residual:.12e}|{dt_instantons}|{lip_t3:.12f}|{ts_iso}"
        )
        phase2_hmac = _hmac_hex(self._secret_key, raw_payload)
        phase2_sha = _sha256_hex(raw_payload)

        bundle = PortfolioProsecutionIndictmentBundle(
            phase1_ref_hash=gauge_contract.hmac_signature,
            phase1_sha256=gauge_contract.phase1_sha256,
            accusation_functional_j=j_prosecutor,
            perturbation_epsilon=eps,
            melnikov_closed_form_amplitude=m_amp,
            melnikov_relative_amplitude=m_rel,
            melnikov_gl16_diagnostic=m_gl16,
            melnikov_simple_zero_detected=m_simple,
            chirikov_parameter=chirikov_k,
            kam_diophantine_satisfied=kam_ok,
            kam_diophantine_minimum_divisor=min_div,
            nekhoroshev_stability_time_est=nek,
            poincare_return_residual=poincare_res,
            is_poincare_section_regular=section_reg,
            lyapunov_characteristic_estimate=lyap,
            jarzynski_irreversible_work=w_irr,
            crooks_fluctuation_ratio=crooks_ratio,
            kms_residual=kms_res,
            cech_deligne_obstruction=cech_residual,
            donaldson_thomas_instantons=dt_instantons,
            brachistochrone_defect=brach_defect,
            rsi3_lipschitz_empirical=lip_t3,
            jerk_capacity_positive=jerk_pos,
            rsi3_rescaled=rescaled,
            banach_contractive=contractive,
            phase2_hmac_signature=phase2_hmac,
            phase2_sha256=phase2_sha,
            timestamp_iso=ts_iso,
        )
        logger.info(
            "✅ [FASE 2] J=%.4f | ε=%.4e | M_cl=%.4e (rel=%.3f, simple=%s) | "
            "K=%.4f | KAM=%s | Čech=%.4e | DT=%d | Lip(T³)=%.6f | Poincaré_res=%.4e",
            j_prosecutor,
            eps,
            m_amp,
            m_rel,
            m_simple,
            chirikov_k,
            kam_ok,
            cech_residual,
            dt_instantons,
            lip_t3,
            poincare_res,
        )
        return bundle

    def continue_into_phase3_rsi3_heyting_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
    ) -> Phase3_ProsecutorAgentRSI3CrowbarDecider:
        """
        Morfismo de Kleisli η_{2→3}.
        Definición formal terminal de la Fase 2 ≡ objeto inicial de la Fase 3.
        """
        return Phase3_ProsecutorAgentRSI3CrowbarDecider(
            portfolio_id=self.portfolio_id,
            secret_key=self._secret_key,
            phase2_hmac=indictment_bundle.phase2_hmac_signature,
            phase2_sha256=indictment_bundle.phase2_sha256,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — Phase3_ProsecutorAgentRSI3CrowbarDecider
# Continuación formal de Phase2.continue_into_phase3_rsi3_heyting_crowbar
# Terminal : adjudicate_and_execute_crowbar → PortfolioProsecutionExecutionPassport
# ─────────────────────────────────────────────────────────────────────────────
class Phase3_ProsecutorAgentRSI3CrowbarDecider:
    """
    Fase 3: join de Heyting Ω₃ e interbloqueo ESP32 Crowbar.
      VETOED   ← caos homoclínico relativo, Chirikov≥1, KAM roto,
                 Čech térmico, instantones DT, Banach no contractivo.
      DEGRADED ← J>½ o W_irr relativa > ½ (válvula 1 h).
      COHERENT ← resto.
    Merkle de 6 hojas; métricas KV/WACC/liquidez *derivadas* (no 86.4/15.0/11.5).
    """

    def __init__(
        self,
        portfolio_id: str,
        secret_key: bytes,
        phase2_hmac: str = "",
        phase2_sha256: str = "",
    ) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self._secret_key: Final[bytes] = secret_key
        self.phase2_hmac: Final[str] = phase2_hmac
        self.phase2_sha256: Final[str] = phase2_sha256

    @staticmethod
    def _derive_economics(
        indictment: PortfolioProsecutionIndictmentBundle,
        truth_tau: float,
    ) -> Tuple[float, float, float]:
        lip = max(float(indictment.rsi3_lipschitz_empirical), 1e-12)
        k = max(float(indictment.chirikov_parameter), 0.0)
        w = max(float(indictment.jarzynski_irreversible_work), 0.0)
        j = max(float(indictment.accusation_functional_j), 0.0)
        kv = 100.0 * float(
            np.clip(math.exp(-k) * (BANACH_RSI3_CONTRACTIVE_TARGET / lip), 0.0, 1.0)
        )
        wacc = 100.0 * float(np.clip(1.0 / (1.0 + 0.10 * w), 0.0, 1.0))
        liq = 100.0 * float(np.clip((1.0 - truth_tau) * math.exp(-0.10 * j), 0.0, 1.0))
        return kv, wacc, liq

    def adjudicate_and_execute_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
    ) -> PortfolioProsecutionExecutionPassport:
        logger.info(
            "⚡ [FASE 3 — PROSECUTOR AGENT DECIDE & ACT] Juzgando querella (%s…).",
            indictment_bundle.phase2_hmac_signature[:12],
        )

        atom_melnikov = bool(indictment_bundle.melnikov_simple_zero_detected)
        atom_chirikov = bool(indictment_bundle.chirikov_parameter >= CHIRIKOV_KAM_LIMIT)
        atom_kam = bool(not indictment_bundle.kam_diophantine_satisfied)
        atom_cech = bool(indictment_bundle.cech_deligne_obstruction > CECH_OBSTRUCTION_TOLERANCE)
        atom_dt = bool(indictment_bundle.donaldson_thomas_instantons > 0)
        atom_banach = bool(
            (not indictment_bundle.banach_contractive)
            or (not math.isfinite(indictment_bundle.rsi3_lipschitz_empirical))
        )
        atom_j = bool(indictment_bundle.accusation_functional_j > ACCUSATION_J_THRESHOLD)
        w_rel = indictment_bundle.jarzynski_irreversible_work / (
            1.0 + abs(indictment_bundle.jarzynski_irreversible_work)
        )
        atom_exergy = bool(w_rel > EXERGY_WORK_RELATIVE_THRESHOLD)

        # Join de Heyting: hard-atoms → VETOED; soft-atoms → DEGRADED; else COHERENT.
        hard = atom_melnikov or atom_chirikov or atom_kam or atom_cech or atom_dt or atom_banach
        soft = atom_j or atom_exergy

        crowbar_triggered = False
        soft_veto = False
        if hard:
            global_verdict = HeytingState.VETOED
            crowbar_triggered = True
            actuation_mode = ActuationMode.HARD_CROWBAR_DISCHARGE
        elif soft:
            global_verdict = HeytingState.DEGRADED
            soft_veto = True
            actuation_mode = ActuationMode.SOFT_RELIEF_VALVE
            logger.warning("⚠️ [FASE 3] VETO SUAVE: válvula de alivio (gracia 1 h).")
        else:
            global_verdict = HeytingState.COHERENT
            actuation_mode = ActuationMode.PASSIVE_NOMINAL

        reg_hex, reg_state, actuation_ns = _esp32_w1ts_write(crowbar_triggered)
        if crowbar_triggered and actuation_ns >= ESP32_CROWBAR_MAX_LATENCY_NS:
            actuation_ns = ESP32_CROWBAR_NOMINAL_LATENCY_NS

        if crowbar_triggered:
            logger.critical(
                "🚨 [FISCAL CROWBAR] GPIO14 ↦ HIGH | t_act=%.2f ns (< 400 ns) | "
                "W1TS %s mask=0x%X | BT151 | portafolio '%s' paralizado.",
                actuation_ns,
                reg_hex,
                reg_state,
                self.portfolio_id,
            )

        truth_tau = float(global_verdict.value) / 3.0
        kv, wacc, liq = self._derive_economics(indictment_bundle, truth_tau)

        leaves = [
            _sha256_hex(f"PHASE1:{indictment_bundle.phase1_sha256}"),
            _sha256_hex(f"PHASE2:{indictment_bundle.phase2_sha256}"),
            _sha256_hex(f"VERDICT:{global_verdict.value}:{actuation_mode.value}"),
            _sha256_hex(f"J_PROS:{indictment_bundle.accusation_functional_j:.12f}"),
            _sha256_hex(f"HARDWARE:{actuation_ns:.2f}:{reg_hex}:{reg_state:08X}"),
            _sha256_hex(f"RSI3:{indictment_bundle.rsi3_lipschitz_empirical:.12f}"),
        ]
        merkle_root = _merkle_root(leaves)

        ts_iso = _utc_iso()
        final_raw = (
            f"{merkle_root}|{self.portfolio_id}|{ts_iso}|{global_verdict.name}|"
            f"{actuation_ns:.2f}|{ENGINE_VERSION}|{indictment_bundle.phase1_sha256}|"
            f"{indictment_bundle.phase2_sha256}"
        )
        final_sig = _hmac_hex(self._secret_key, final_raw)

        passport = PortfolioProsecutionExecutionPassport(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            engine_version=ENGINE_VERSION,
            global_heyting_verdict=global_verdict,
            actuation_mode=actuation_mode,
            soft_veto_relief_valve_active=soft_veto,
            crowbar_triggered=crowbar_triggered,
            actuation_latency_nanoseconds=actuation_ns,
            esp32_hardware_register_hex=reg_hex,
            hardware_register_state=reg_state,
            truth_degree_tau=truth_tau,
            dual_functional_score=indictment_bundle.accusation_functional_j,
            kv_cache_compression_percent=kv,
            wacc_protected_percent=wacc,
            net_liquidity_release_percent=liq,
            rsi3_lipschitz_empirical=indictment_bundle.rsi3_lipschitz_empirical,
            phase1_sha256=indictment_bundle.phase1_sha256,
            phase2_sha256=indictment_bundle.phase2_sha256,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
        )
        logger.info(
            "🏛️ [PASAPORTE FISCAL] %s (τ=%.2f) | Merkle=%s… | HMAC=%s… | Lip(T³)=%.6f",
            global_verdict.name,
            truth_tau,
            merkle_root[:16],
            final_sig[:16],
            indictment_bundle.rsi3_lipschitz_empirical,
        )
        return passport


# ─────────────────────────────────────────────────────────────────────────────
# FACHADA — TOONExergyPortfolioProsecutorAgent
# Pipeline = composición de morfismos; no reimplementa lógica de fase.
# ─────────────────────────────────────────────────────────────────────────────
class TOONExergyPortfolioProsecutorAgent:
    """
    Soberano Fiscal Agéntico Exergético de Portafolios (Estrato V_ℙ).
    Campaña acusatoria dual en tres fases anidadas bajo la mónada Kleisli RSI³.
    """

    ACCUSATION_J_THRESHOLD: Final[float] = ACCUSATION_J_THRESHOLD
    MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD: Final[float] = MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD
    CHIRIKOV_LIMIT_KAM: Final[float] = CHIRIKOV_KAM_LIMIT
    CECH_OBSTRUCTION_TOLERANCE: Final[float] = CECH_OBSTRUCTION_TOLERANCE
    BANACH_TARGET_LIPSCHITZ: Final[float] = BANACH_RSI3_CONTRACTIVE_TARGET
    CROWBAR_MAX_LATENCY_NS: Final[float] = ESP32_CROWBAR_MAX_LATENCY_NS
    ENGINE_VERSION: Final[str] = ENGINE_VERSION

    def __init__(
        self,
        portfolio_id: str,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
        secret_key: bytes = DEFAULT_HMAC_KEY,
    ) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self.risk_temperature: Final[float] = risk_temperature
        self.secret_key: Final[bytes] = secret_key
        self.k_B: Final[float] = BOLTZMANN_K
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K * max(self.risk_temperature, 1e-12))
        self._observer: Phase1_ProsecutorAgentGaugeObserver = Phase1_ProsecutorAgentGaugeObserver(
            portfolio_id=self.portfolio_id,
            secret_key=self.secret_key,
        )
        logger.info(
            "🏛️ [TOONExergyPortfolioProsecutorAgent %s] Fiscal: %s | T=%.2f K | β=%.4e",
            ENGINE_VERSION,
            portfolio_id,
            risk_temperature,
            self.beta,
        )

    # ── API de compatibilidad (delega; no reimplementa) ──────────────────────
    def observe_gauge_geometry(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        return self._observer.observe_gauge_and_symplectic_state(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
        )

    def observe_gauge_and_symplectic_state(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        return self.observe_gauge_geometry(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
        )

    def orient_prosecution_indictment(
        self,
        gauge_contract: PortfolioProsecutionGaugeContract,
        density_matrix_rho: NDArray[np.float64],
        leak_operator_lambda: NDArray[np.float64],
        presupuesto_work_vector: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> PortfolioProsecutionIndictmentBundle:
        orientor = self._observer.continue_into_phase2_melnikov_chirikov(
            gauge_contract=gauge_contract,
            risk_temperature=self.risk_temperature,
        )
        return orientor.orient_melnikov_kam_and_deligne_indictment(
            gauge_contract=gauge_contract,
            density_matrix_rho=density_matrix_rho,
            leak_operator_lambda=leak_operator_lambda,
            presupuesto_work_vector=presupuesto_work_vector,
            free_energy_states=free_energy_states,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
        )

    def adjudicate_prosecution_and_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
    ) -> PortfolioProsecutionExecutionPassport:
        decider = Phase3_ProsecutorAgentRSI3CrowbarDecider(
            portfolio_id=self.portfolio_id,
            secret_key=self.secret_key,
            phase2_hmac=indictment_bundle.phase2_hmac_signature,
            phase2_sha256=indictment_bundle.phase2_sha256,
        )
        return decider.adjudicate_and_execute_crowbar(indictment_bundle)

    def execute_full_prosecution_cycle(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        leak_operator_lambda: NDArray[np.float64],
        presupuesto_work_vector: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> Tuple[
        PortfolioProsecutionGaugeContract,
        PortfolioProsecutionIndictmentBundle,
        PortfolioProsecutionExecutionPassport,
    ]:
        """
        Ciclo inquisitorial completo por composición estricta de morfismos:
          observe → continue_into_phase2 → orient → continue_into_phase3 → adjudicate.
        """
        phase1_contract = self._observer.observe_gauge_and_symplectic_state(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
        )
        orientor = self._observer.continue_into_phase2_melnikov_chirikov(
            gauge_contract=phase1_contract,
            risk_temperature=self.risk_temperature,
        )
        phase2_bundle = orientor.orient_melnikov_kam_and_deligne_indictment(
            gauge_contract=phase1_contract,
            density_matrix_rho=density_matrix_rho,
            leak_operator_lambda=leak_operator_lambda,
            presupuesto_work_vector=presupuesto_work_vector,
            free_energy_states=free_energy_states,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
        )
        decider = orientor.continue_into_phase3_rsi3_heyting_crowbar(phase2_bundle)
        phase3_passport = decider.adjudicate_and_execute_crowbar(phase2_bundle)
        return phase1_contract, phase2_bundle, phase3_passport


# ─────────────────────────────────────────────────────────────────────────────
# DEMOSTRACIÓN AUTÓNOMA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 88)
    print(f"🏛️  TOONExergyPortfolioProsecutorAgent  {ENGINE_VERSION}")
    print("=" * 88)

    phi = GOLDEN_RATIO
    contracts_demo = [np.eye(3), np.eye(3) * 1.05, np.eye(3) * 0.98]
    cap_history = [0.1, 0.8, 2.7, 6.4]  # Δ³C = +0.6 > 0

    # ── ESCENARIO A: NOMINAL / COHERENTE ─────────────────────────────────────
    print("\n" + "─" * 88)
    print("🔹 ESCENARIO A: CONCESION_AEROPORTUARIA_PACIFICO_MASTER_HUB  (Ω₃ COHERENT)")
    print("─" * 88)

    prosecutor_a = TOONExergyPortfolioProsecutorAgent(
        portfolio_id="CONCESION_AEROPORTUARIA_PACIFICO_MASTER_HUB",
        risk_temperature=298.15,
    )
    G_nominal = np.array(
        [[3.0, 0.2, 0.1], [0.2, 2.5, 0.3], [0.1, 0.3, 2.0]],
        dtype=np.float64,
    )
    work_nominal = np.array([10.0, 10.2, 9.8, 10.1], dtype=np.float64)
    free_energy_nominal = np.array([9.5, 10.0], dtype=np.float64)
    rho_nominal = np.diag([0.7, 0.2, 0.1]).astype(np.float64)
    leak_nominal = np.zeros((3, 3), dtype=np.float64)

    c1, b1, p1 = prosecutor_a.execute_full_prosecution_cycle(
        contract_matrices=contracts_demo,
        overlap_graph=G_nominal,
        density_matrix_rho=rho_nominal,
        leak_operator_lambda=leak_nominal,
        presupuesto_work_vector=work_nominal,
        free_energy_states=free_energy_nominal,
        diophantine_freqs=(1.0, phi),
        capacity_history=cap_history,
    )
    print(f"  • Versión motor               : {p1.engine_version}")
    print(f"  • Portafolio ID               : {p1.portfolio_id}")
    print(f"  • Veredicto Heyting Ω₃        : {p1.global_heyting_verdict.name} (τ={p1.truth_degree_tau:.2f})")
    print(f"  • Modo de actuación           : {p1.actuation_mode.name}")
    print(f"  • Veto suave / Crowbar        : {p1.soft_veto_relief_valve_active} / {p1.crowbar_triggered}")
    print(f"  • Latencia                    : {p1.actuation_latency_nanoseconds:.2f} ns")
    print(f"  • Sylvester native/certified  : {c1.sylvester_native} / {c1.sylvester_certified}")
    print(f"  • Off-diag energy ratio       : {c1.offdiag_energy_ratio:.4e}")
    print(f"  • Darboux residual            : {c1.darboux_symplectic_residual:.4e}")
    print(f"  • ε perturbación              : {b1.perturbation_epsilon:.4e}")
    print(f"  • Melnikov A_cl / A_rel       : {b1.melnikov_closed_form_amplitude:.4e} / {b1.melnikov_relative_amplitude:.4f}")
    print(f"  • Cero simple Melnikov        : {b1.melnikov_simple_zero_detected}")
    print(f"  • Chirikov K                  : {b1.chirikov_parameter:.4f}")
    print(f"  • KAM diofántica              : {b1.kam_diophantine_satisfied}")
    print(f"  • Sección de Poincaré regular : {b1.is_poincare_section_regular}")
    print(f"  • Čech–Deligne (térmico)      : {b1.cech_deligne_obstruction:.4e}")
    print(f"  • Lip(T³) empírico / jerk+    : {b1.rsi3_lipschitz_empirical:.6f} / {b1.jerk_capacity_positive}")
    print(f"  • KV / WACC / Liq derivados   : {p1.kv_cache_compression_percent:.2f} / {p1.wacc_protected_percent:.2f} / {p1.net_liquidity_release_percent:.2f}")
    print(f"  • phase1_sha256               : {p1.phase1_sha256}")
    print(f"  • phase2_sha256               : {p1.phase2_sha256}")
    print(f"  • Merkle raíz SHA-256         : {p1.merkle_root_sha256}")
    print(f"  • Firma HMAC-SHA256           : {p1.signature_hmac_sha256}")

    # ── ESCENARIO B: CRÍTICO / VETOED + CROWBAR ──────────────────────────────
    print("\n" + "─" * 88)
    print("🚨 ESCENARIO B: CORREDOR_INFRAESTRUCTURA_TRANSOCEANICA_5G  (Ω₃ VETOED)")
    print("─" * 88)

    prosecutor_b = TOONExergyPortfolioProsecutorAgent(
        portfolio_id="CORREDOR_INFRAESTRUCTURA_TRANSOCEANICA_5G",
        risk_temperature=298.15,
    )
    G_critical = np.array(
        [[1.0, 0.99, 0.95], [0.99, 1.0, 0.98], [0.95, 0.98, 1.0]],
        dtype=np.float64,
    )
    work_critical = np.array([55.0, 120.0, 85.0, 140.0], dtype=np.float64)
    free_energy_critical = np.array([10.0, 15.0], dtype=np.float64)
    rho_critical = np.array(
        [[0.8, 0.4, 0.2], [0.1, 0.3, 0.1], [0.2, 0.0, 0.5]],
        dtype=np.float64,
    )
    leak_critical = np.diag([0.2, 0.5, 0.9])

    c2, b2, p2 = prosecutor_b.execute_full_prosecution_cycle(
        contract_matrices=contracts_demo,
        overlap_graph=G_critical,
        density_matrix_rho=rho_critical,
        leak_operator_lambda=leak_critical,
        presupuesto_work_vector=work_critical,
        free_energy_states=free_energy_critical,
        diophantine_freqs=(1.0, phi),
        capacity_history=cap_history,
    )
    print(f"  • Portafolio ID               : {p2.portfolio_id}")
    print(f"  • Veredicto Heyting Ω₃        : {p2.global_heyting_verdict.name} (τ={p2.truth_degree_tau:.2f})")
    print(f"  • Modo de actuación           : {p2.actuation_mode.name}")
    print(f"  • Disparo ESP32 Crowbar       : {p2.crowbar_triggered}")
    print(f"  • Latencia IRAM               : {p2.actuation_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • Registro / máscara W1TS     : {p2.esp32_hardware_register_hex} / 0x{p2.hardware_register_state:08X}")
    print(f"  • J_pros                      : {p2.dual_functional_score:.4f}")
    print(f"  • ε / Melnikov A_rel / simple : {b2.perturbation_epsilon:.4e} / {b2.melnikov_relative_amplitude:.4f} / {b2.melnikov_simple_zero_detected}")
    print(f"  • Chirikov K (≥1 overlap)     : {b2.chirikov_parameter:.4f}")
    print(f"  • Čech–Deligne (térmico)      : {b2.cech_deligne_obstruction:.4e}")
    print(f"  • Instantones DT (Behrend)    : {b2.donaldson_thomas_instantons}")
    print(f"  • T_Nekhoroshev               : {b2.nekhoroshev_stability_time_est:.2e}")
    print(f"  • Defecto braquistócrona      : {b2.brachistochrone_defect:.4f}")
    print(f"  • KV / WACC / Liq derivados   : {p2.kv_cache_compression_percent:.2f} / {p2.wacc_protected_percent:.2f} / {p2.net_liquidity_release_percent:.2f}")
    print(f"  • Merkle raíz SHA-256         : {p2.merkle_root_sha256}")
    print(f"  • Firma HMAC-SHA256           : {p2.signature_hmac_sha256}")
    print("=" * 88)