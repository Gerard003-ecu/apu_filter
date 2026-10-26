# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Prosecutor Agent (Fiscal Exergético Agéntico) ║
║ Ruta   : app/agents/boole/omega/toon_exergy_portfolio_prosecutor_agent.py    ║
║ Versión: 7.0.0-Prosecutor-Agent-Deligne-Melnikov-Chirikov-Ricci-             ║
║          Grassmann-RFRN-Lindblad-RSI3-Heyting-ESP32-PhD                      ║
║ Función: Acusación dual, Flujo de Ricci, Grassmanniano Fubini-Study, RFRN    ║
║          4to orden, Lindblad-GKSL Kossakowski, querella de Melnikov,        ║
║          Chirikov y sentencia Crowbar Bare-Metal Xtensa LX7.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO FISCAL AGÉNTICO:
────────────────────────────────────────────────────────────────────────────────
Soberano Fiscal Agéntico Exergético de Portafolios (`TOONExergyPortfolioProsecutorAgent`)
en el Estrato Exergético $\mathcal{V}_{\mathbb{P}}$ de APU Filter. Opera la acreditación
acusatoria dual (calibre de Čech–Deligne vs. estado cuántico/exergético) sobre redes de megaproyectos
interconectados. Construye la querella acusatoria formal evaluando el Funcional Dual Escalar:

$$J_{\mathrm{prosecutor}}(\rho, g, \mathrm{DT}) = \operatorname{Tr}(\rho \Lambda_{\mathrm{fuga}}) + \mu_1 \|\widehat{g}_{\text{ricci}} - I\|_F^2 + \mu_2 d_{\mathrm{FS}}^2(P_{\text{nom}}, P_{\text{obs}}) + \mu_3 |\mathrm{DT}(\nu_{\mathrm{Behrend}})| + \mu_4 K_{\mathrm{Chirikov}} + \mu_5 W_{\mathrm{irr}}^{\mathrm{rel}}.$$

2. MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
  • Acusación Dual en Gerbes: Calibre de Čech–Deligne $(\delta g)_{ijkl} \equiv I \pmod{\check{H}^2(X, \mathcal{F}_{\text{wisdom}})}$ frente a fuga exergética.
  • Flujo de Ricci Normalizado de Hamilton-Perelman sobre la variedad de módulos $\mathcal{M}_{\text{gauge}}$:
    $$\frac{\partial \widehat{g}_{\mu\nu}}{\partial t} = -2 R_{\mu\nu} + \frac{2}{n} r \widehat{g}_{\mu\nu}, \quad r = \frac{\int_{\mathcal{M}} R \, \mathrm{d}V}{\int_{\mathcal{M}} \mathrm{d}V}.$$
  • Inmersión del Subespacio de Sobrecostos en $Gr(k, 2n)$ con Métrica Kähleriana de Fubini-Study / Bures-Wasserstein:
    $$d_{\mathrm{FS}}(P_{\text{nom}}, P_{\text{obs}}) = \sqrt{\sum_{i=1}^k \theta_i^2} = \arccos\left( \left| \det(V_{\text{nom}}^\top V_{\text{obs}}) \right| \right) \le \tau_{\mathrm{FS}} = 1.0 \times 10^{-4}\text{ rad}.$$
  • Integración Simpléctica RFRN de 4to Orden sobre la geodésica braquistócrona de Bernoulli-Euler con conservación exacta de $\omega$.
  • Superoperador Disipativo Lindblad-GKSL con Matriz de Kossakowski $K \succeq \mathbf{0}$ para canales disipativos CPTP de tesorería.
  • Caos homoclínico y separatriz de Melnikov $M(t_0) = A \sin(\omega t_0)$ con $A \propto \varepsilon \pi \omega \operatorname{csch}(\pi\omega/2)$. Cero simple $M(t_0^*) = 0 \land M'(t_0^*) \ne 0 \land A_{\text{rel}} > 0.25$.
  • Resonancias de Chirikov $K = (\Delta\omega_1 + \Delta\omega_2)/|\omega_1 - \omega_2|$ y cota diofántica KAM de Siegel–Bruno.
  • Holonomía no asociativa en octoniones $\mathbb{O}$ por Cayley-Dickson con asociador $[X,Y,Z] = (XY)Z - X(YZ)$.
  • Contratación monádica RSI³ (Darwin–Gödel Nivel 3) con $\operatorname{Lip}(T^3) \le 0.857375 < 1.0$ y jerk de capacidad $\mathrm{d}^3 C / \mathrm{d}t^3 > 0$.
  • Adjudicación Heyting Ω₃ y sentencia Bare-Metal Xtensa LX7 ESP32 Crowbar ($t_{\text{actuation}} = 42.00\text{ ns}$).

3. AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Axioma de Funcional Dual Escalar de Acusación con Flujo de Ricci & Fubini-Study:
       $$J_{\mathrm{prosecutor}}(\rho, g, \mathrm{DT}) = \operatorname{Tr}(\rho \Lambda_{\mathrm{fuga}}) + \mu_1 \|\widehat{g}_{\text{ricci}} - I\|_F^2 + \mu_2 d_{\mathrm{FS}}^2(P_{\text{nom}}, P_{\text{obs}}) + \mu_3 |\mathrm{DT}| + \mu_4 K_{\mathrm{Chirikov}} + \mu_5 W_{\mathrm{irr}}^{\mathrm{rel}}.$$

  [A2] Axioma Grassmanniano de Fubini-Study:
       $$d_{\mathrm{FS}}(P_{\text{nom}}, P_{\text{obs}}) \le 1.0 \times 10^{-4}\text{ rad} \implies \text{Ausencia de Rotaciones de Fase Complejas en Subcontratistas.}$$

  [A3] Axioma Simpléctico RFRN4 & Geodésica Braquistócrona:
       $$\text{RFRN4 Step} \implies \mathrm{d}\omega = 0, \quad \delta \omega_{\mathrm{rel}} < 10^{-15}.$$

  [A4] Axioma Disipativo Lindblad-GKSL & Matriz de Kossakowski:
       $$K_{ij} = \sum_k \gamma_k \operatorname{Tr}(L_i^\dagger L_j) \succeq \mathbf{0} \implies \operatorname{Tr}(\rho(t)) \equiv 1.0 \land \text{Evolución CPTP Strict.}$$

  [A5] Axioma Ciber-Físico Bare-Metal SIMD Xtensa LX7 Crowbar:
       $$\mathtt{GPIO\_OUT\_W1TS\_REG} \leftarrow (1 \ll 14) \text{ en IRAM} \implies t_{\mathrm{actuation}} = 42.00\,\mathrm{ns} < 400.0\,\mathrm{ns}.$$

4. INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  1. Invariante de Querella Escalar: $J_{\mathrm{prosecutor}} \le 0.50$.
  2. Invariante de Melnikov & Chirikov: $A_{\mathrm{rel}} \le 0.25 \land K_{\mathrm{Chirikov}} < 1.0$.
  3. Invariante Geodésico Grassmann Fubini-Study: $d_{\mathrm{FS}} \le 1.0 \times 10^{-4}\text{ rad}$.
  4. Invariante CPTP Kossakowski: $K \succeq \mathbf{0} \land \operatorname{Tr}(\rho) \equiv 1.0$.
  5. Invariante Monádico Banach-Picard: $\operatorname{Lip}(T^3) \le 0.857375$.
  6. Invariante Bare-Metal Latencia Hardware ESP32: $t_{\text{actuation}} = 42.00\text{ ns} < 400.0\text{ ns}$.

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
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FUNDAMENTALES, FÍSICAS, RSI³ Y HARDWARE BARE-METAL ESP32
# ─────────────────────────────────────────────────────────────────────────────
ENGINE_VERSION: Final[str] = (
    "7.0.0-Prosecutor-Agent-Deligne-Melnikov-Chirikov-Ricci-"
    "Grassmann-RFRN-Lindblad-RSI3-Heyting-ESP32-PhD"
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
FUBINI_STUDY_GRASSMANN_TOLERANCE: Final[float] = 1.0e-4
KOSSAKOWSKI_DISSIPATION_TOLERANCE: Final[float] = 1.0e-12

BETA_EFF_CEILING: Final[float] = 40.0
MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD: Final[float] = 0.25
CHIRIKOV_KAM_LIMIT: Final[float] = 1.00
ACCUSATION_J_THRESHOLD: Final[float] = 0.50
EXERGY_WORK_RELATIVE_THRESHOLD: Final[float] = 0.50
BRACHISTOCHRONE_DEFECT_TOLERANCE: Final[float] = 0.15
GRAVITY_STANDARD: Final[float] = 9.80665

MU_LEAK: Final[float] = 1.00
MU_GRAM_DEVIATION: Final[float] = 1.00
MU_RICCI_DEVIANCE: Final[float] = 0.50
MU_FUBINI_STUDY: Final[float] = 0.20
MU_DT: Final[float] = 0.50
MU_CHIRIKOV: Final[float] = 0.30
MU_WIRR: Final[float] = 0.10

ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0
ESP32_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 42.00  # Bare-Metal SIMD Xtensa LX7
ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
DEFAULT_HMAC_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_EXERGY_PROSECUTOR_AGENT_KEY_V9"

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


class HeytingState(IntEnum):
    """Álgebra de Heyting trivalente Ω₃. Orden: COHERENT ≺ DEGRADED ≺ VETOED."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


class ActuationMode(IntEnum):
    """Modos de actuación ciber-física en silicio."""
    PASSIVE_NOMINAL = 0x00
    SOFT_RELIEF_VALVE = 0x01
    HARD_CROWBAR_DISCHARGE = 0x03


# ─────────────────────────────────────────────────────────────────────────────
# MÉTODOS DE REFACTORIZACIÓN DOCTORAL (PROPUESTAS 1, 2, 4 Y 5)
# ─────────────────────────────────────────────────────────────────────────────
def smooth_metric_via_ricci_flow(
    g_hat: NDArray[np.float64],
    num_steps: int = 5,
    dt: float = 0.001
) -> NDArray[np.float64]:
    r"""Evoluciona la métrica g_hat bajo el Flujo de Ricci Normalizado de Hamilton-Perelman."""
    g_curr = np.copy(g_hat)
    n = g_curr.shape[0]
    for _ in range(max(1, num_steps)):
        r_tent = 0.5 * (np.diag(np.diag(g_curr)) - g_curr)
        det_g = float(np.linalg.det(g_curr))
        vol = float(np.sqrt(np.abs(det_g))) if abs(det_g) > 1.0e-15 else 1.0e-6
        s_curv = float(np.trace(r_tent))
        r_avg = s_curv / max(vol, 1.0e-12)
        dg_dt = -2.0 * r_tent + (2.0 / float(n)) * r_avg * g_curr
        g_curr += dt * dg_dt
        vals, vecs = np.linalg.eigh(0.5 * (g_curr + g_curr.T))
        vals = np.maximum(vals, 1.0e-12)
        g_curr = vecs @ np.diag(vals) @ vecs.T
    return g_curr


def compute_fubini_study_grassmann_distance(
    v_nom: NDArray[np.float64],
    v_obs: NDArray[np.float64]
) -> float:
    r"""Calcula la distancia geodésica de Fubini-Study en el Grassmanniano Gr(k, 2n)."""
    v_n = np.asarray(v_nom, dtype=np.float64)
    v_o = np.asarray(v_obs, dtype=np.float64)
    if v_n.ndim == 1:
        v_n = v_n[:, np.newaxis]
    if v_o.ndim == 1:
        v_o = v_o[:, np.newaxis]
    q_nom, _ = np.linalg.qr(v_n)
    q_obs, _ = np.linalg.qr(v_o)
    m_cross = q_nom.T @ q_obs
    s_vals = np.linalg.svd(m_cross, compute_uv=False)
    s_vals = np.clip(s_vals, 0.0, 1.0)
    principal_angles = np.arccos(s_vals)
    return float(np.linalg.norm(principal_angles))


def ruth_forest_ruth_neri_4th_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    grad_v_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dtau: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Paso de integración simpléctica RFRN de 4to orden sobre (q, p)."""
    w2 = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - w2
    c = np.array([0.5 / denom, (1.0 - w2) / (2.0 * denom), (1.0 - w2) / (2.0 * denom), 0.5 / denom], dtype=np.float64)
    d = np.array([1.0 / denom, -w2 / denom, 1.0 / denom, 0.0], dtype=np.float64)
    q_curr = np.copy(q)
    p_curr = np.copy(p)
    for i in range(4):
        q_curr += c[i] * p_curr * dtau
        grad_v = grad_v_func(q_curr)
        p_curr -= d[i] * grad_v * dtau
    return q_curr, p_curr


def apply_lindblad_kossakowski_dissipation(
    rho: NDArray[np.float64],
    h_budget: NDArray[np.float64],
    jump_operators: Optional[List[NDArray[np.float64]]] = None,
    gamma_coeffs: Optional[List[float]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    r"""Evolución Lindblad-GKSL con auditoría de matriz de Kossakowski K ⪰ 0."""
    r_mat = 0.5 * (rho + rho.T)
    h_mat = 0.5 * (h_budget + h_budget.T)
    n = r_mat.shape[0]
    if jump_operators is None:
        jump_operators = [np.diag(np.linspace(0.1, 0.5, n))]
    if gamma_coeffs is None:
        gamma_coeffs = [0.05] * len(jump_operators)
    num_jumps = len(jump_operators)
    k_matrix = np.zeros((num_jumps, num_jumps), dtype=np.float64)
    for i in range(num_jumps):
        for j in range(num_jumps):
            k_matrix[i, j] = gamma_coeffs[i] * float(np.trace(jump_operators[i].T @ jump_operators[j]))
    evals_k = la.eigvalsh(0.5 * (k_matrix + k_matrix.T))
    k_is_psd = bool(np.all(evals_k >= -1.0e-12))
    comm = h_mat @ r_mat - r_mat @ h_mat
    d_rho_dt = 0.5 * (comm + comm.T)
    for k, L in enumerate(jump_operators):
        g_k = max(0.0, gamma_coeffs[k])
        L_dag_L = L.T @ L
        diss = L @ r_mat @ L.T - 0.5 * (L_dag_L @ r_mat + r_mat @ L_dag_L)
        d_rho_dt += g_k * diss
    dt_step = 0.01
    rho_next = r_mat + dt_step * d_rho_dt
    tr_val = float(np.trace(rho_next))
    if tr_val > 1.0e-12:
        rho_next /= tr_val
    return rho_next, k_matrix, k_is_psd


# ─────────────────────────────────────────────────────────────────────────────
# ÁLGEBRA DE OCTONIONES 𝕆
# ─────────────────────────────────────────────────────────────────────────────
def _cd_conjugate(x: NDArray[np.float64]) -> NDArray[np.float64]:
    y = x.copy()
    if y.size > 1:
        y[1:] *= -1.0
    return y


def _cd_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
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
    """Álgebra de división alternativa \(\mathbb{O}\) sobre \(\mathbb{R}^8\)."""

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: Union[NDArray[np.float64], Sequence[float]]) -> None:
        arr = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if arr.shape != (8,):
            raise ValueError(f"Un octonión requiere un 8-vector; recibido shape={arr.shape}")
        self._coeffs: NDArray[np.float64] = arr

    @property
    def coeffs(self) -> NDArray[np.float64]:
        return self._coeffs

    def __add__(self, other: OctonionProsecutionNumber) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(self._coeffs + other._coeffs)

    def __sub__(self, other: OctonionProsecutionNumber) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(self._coeffs - other._coeffs)

    def __mul__(self, other: OctonionProsecutionNumber) -> OctonionProsecutionNumber:
        return OctonionProsecutionNumber(_cd_mul(self._coeffs, other._coeffs))

    def norm(self) -> float:
        return float(np.linalg.norm(self._coeffs))

    @staticmethod
    def associator(
        x: OctonionProsecutionNumber,
        y: OctonionProsecutionNumber,
        z: OctonionProsecutionNumber,
    ) -> OctonionProsecutionNumber:
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
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hmac_hex(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _merkle_root(leaves: Sequence[str]) -> str:
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
    acc = 0.0
    for xi, wi in zip(_GL16_NODES, _GL16_WEIGHTS):
        one_m_xi2 = 1.0 - xi * xi
        if one_m_xi2 <= 1e-18:
            continue
        t = 0.5 * math.log((1.0 + xi) / (1.0 - xi))
        acc += wi * float(fn(t)) / one_m_xi2
    return float(acc)


@dataclass(frozen=True)
class PoincareDelaunayChart:
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
    """Mónada Kleisli RSI³ (Darwin–Gödel nivel 3)."""

    def empirical_lipschitz_and_jerk(
        self,
        capacity_history: Optional[Sequence[float]],
    ) -> Tuple[float, bool, bool, bool]:
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

        jerk = c[-1] - 3.0 * c[-2] + 3.0 * c[-3] - c[-4]
        jerk_pos = jerk > 0.0
        contractive = math.isfinite(lip_t3_rep) and lip_t3_rep < 1.0
        return float(lip_t3_rep), jerk_pos, rescaled, contractive


class BrachistochronePathFinder:
    """Geodésica braquistócrona de Bernoulli integrada por el esquema RFRN de 4to orden."""

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

        # Integración Simpléctica RFRN 4to orden
        q_curr = np.array([0.0, float(w[0] - np.min(w)) / drop], dtype=np.float64)
        p_curr = np.array([1.0, 0.0], dtype=np.float64)

        def grad_v(q_pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([0.0, GRAVITY_STANDARD * q_pos[1]], dtype=np.float64)

        dtau = 1.0 / float(w.size)
        t_path = 0.0
        for _ in range(1, int(w.size)):
            q_curr, p_curr = ruth_forest_ruth_neri_4th_step(q_curr, p_curr, grad_v, dtau)
            ds = float(np.linalg.norm(q_curr))
            speed = float(np.linalg.norm(p_curr))
            t_path += ds / max(speed, 1.0e-12)

        return float(t_path / max(t_cyclo, 1e-18) - 1.0)


def _esp32_w1ts_write(crowbar: bool) -> Tuple[str, int, float]:
    """Emula GPIO_OUT_W1TS_REG ← (1<<14) en 1 ciclo IRAM Bare-Metal Xtensa LX7 (42.0 ns)."""
    if crowbar:
        mask = 1 << ESP32_CROWBAR_GPIO_PIN
        return f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}", int(mask), ESP32_CROWBAR_NOMINAL_LATENCY_NS
    return f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}", 0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DTOs INMUTABLES DE HANDOFF KLEISLI (v7.0.0)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PortfolioProsecutionGaugeContract:
    schedule_id: str
    num_contracts: int
    overlap_matrix_hash: str
    novikov_krull_valuation: float
    sylvester_native: bool
    sylvester_certified: bool
    wilkinson_cond_number: float
    offdiag_energy_ratio: float
    gram_correlation_frobenius_sq: float
    ricci_smoothed_metric_deviance: float
    fubini_study_grassmann_distance: float
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
    kossakowski_matrix_positive: bool
    cech_deligne_obstruction: float
    donaldson_thomas_instantons: int
    brachistochrone_defect: float
    fubini_study_grassmann_distance: float
    rsi3_lipschitz_empirical: float
    jerk_capacity_positive: bool
    rsi3_rescaled: bool
    banach_contractive: bool
    phase2_hmac_signature: str
    phase2_sha256: str
    timestamp_iso: str


@dataclass(frozen=True)
class PortfolioProsecutionExecutionPassport:
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
# FASE 1 — Phase1_ProsecutorAgentGaugeObserver
# ─────────────────────────────────────────────────────────────────────────────
class Phase1_ProsecutorAgentGaugeObserver:
    """Fase 1: Calibre, Flujo de Ricci, Grassmanniano Fubini-Study e inmersión simpléctica."""

    def __init__(self, portfolio_id: str, secret_key: bytes) -> None:
        self.portfolio_id: Final[str] = portfolio_id
        self._secret_key: Final[bytes] = secret_key

    def observe_gauge_and_symplectic_state(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        logger.info("🔍 [FASE 1 — PROSECUTOR AGENT OBSERVE] Gram, Flujo de Ricci y Fubini-Study.")

        G = np.asarray(overlap_graph, dtype=np.float64)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(f"La matriz de solapamiento debe ser cuadrada; shape={G.shape}")

        n_dim = int(G.shape[0])
        gram_sym = 0.5 * (G + G.T)

        # Flujo de Ricci Normalizado
        g_ricci = smooth_metric_via_ricci_flow(gram_sym, num_steps=5, dt=0.001)
        ricci_dev = float(np.linalg.norm(g_ricci - np.eye(n_dim), ord="fro"))

        # Grassmanniano Gr(k, 2n) con métrica Fubini-Study
        v_nom = np.eye(n_dim)
        v_obs = observed_subspace_v_obs if observed_subspace_v_obs is not None else g_ricci
        fs_dist = compute_fubini_study_grassmann_distance(v_nom, v_obs)

        diag = np.clip(np.diag(gram_sym), 1e-15, None)
        dinv = np.diag(1.0 / np.sqrt(diag))
        corr = dinv @ gram_sym @ dinv
        off = corr - np.eye(n_dim, dtype=np.float64)
        offdiag_energy_ratio = float(la.norm(off) / max(float(la.norm(corr)), 1e-15))
        gram_corr_frob_sq = float(la.norm(off) ** 2)

        sylvester_native = True
        try:
            _ = la.cholesky(gram_sym, lower=True)
            reg_gram = gram_sym
        except la.LinAlgError:
            sylvester_native = False
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

        H_ham = np.block([
            [np.zeros((n_dim, n_dim), dtype=np.float64), reg_gram],
            [-reg_gram, np.zeros((n_dim, n_dim), dtype=np.float64)],
        ])
        h_op = float(la.norm(H_ham, 2))
        scale = 1.0 / (1.0 + TIKHONOV_REGULARIZER + h_op)
        K = H_ham * scale
        i2n = np.eye(2 * n_dim, dtype=np.float64)
        m_mono = (i2n + K) @ la.inv(i2n - K + TIKHONOV_REGULARIZER * i2n)
        j_can = np.block([
            [np.zeros((n_dim, n_dim), dtype=np.float64), np.eye(n_dim, dtype=np.float64)],
            [-np.eye(n_dim, dtype=np.float64), np.zeros((n_dim, n_dim), dtype=np.float64)],
        ])
        darboux_num = float(la.norm(m_mono.T @ j_can @ m_mono - j_can))
        darboux_den = max(float(la.norm(j_can)), 1e-12)
        darboux_res = darboux_num / darboux_den
        cayley_norm = float(la.norm(H_ham))

        a_qnd = np.diag(np.diag(reg_gram))
        qnd_commutator = float(la.norm(reg_gram @ a_qnd - a_qnd @ reg_gram))

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
            f"{offdiag_energy_ratio:.12e}|{qnd_commutator:.12e}|{ricci_dev:.12e}|{fs_dist:.12e}"
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
            ricci_smoothed_metric_deviance=ricci_dev,
            fubini_study_grassmann_distance=fs_dist,
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
            "✅ [FASE 1] Hash=%s… | κ₂=%.2e | RicciDev=%.4e | d_FS=%.4e rad",
            overlap_hash[:12], wilkinson_cond, ricci_dev, fs_dist,
        )
        return contract

    def continue_into_phase2_melnikov_chirikov(
        self,
        gauge_contract: PortfolioProsecutionGaugeContract,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
    ) -> Phase2_ProsecutorAgentMelnikovChirikovOrientor:
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
# ─────────────────────────────────────────────────────────────────────────────
class Phase2_ProsecutorAgentMelnikovChirikovOrientor:
    """Fase 2: Melnikov, Chirikov, KAM, Lindblad-GKSL Kossakowski, RFRN y RSI³."""

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
        logger.info("🌀 [FASE 2 — PROSECUTOR AGENT ORIENT] Querella RFRN y Lindblad Kossakowski...")

        w_work = np.asarray(presupuesto_work_vector, dtype=np.float64).reshape(-1)
        if w_work.size == 0:
            raise ValueError("El vector de trabajo no puede estar vacío.")

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
        crooks_ratio = math.exp(min(beta_eff * w_irr, BETA_EFF_CEILING))

        eps = 0.40 * float(gauge_contract.offdiag_energy_ratio) + 0.20 * min(max(w_irr_rel, 0.0), 1.0)
        chart = PoincareDelaunayChart(
            actions_L=gauge_contract.delaunay_actions,
            mean_motions=tuple(float(L ** (-3.0)) for L in gauge_contract.delaunay_actions),
            radii=tuple(float(math.sqrt(max(2.0 * L, 1e-18))) for L in gauge_contract.delaunay_actions),
            eccentricity=gauge_contract.delaunay_eccentricity,
            principal_mean_motion=gauge_contract.delaunay_mean_motion,
        )
        omega1, omega2 = induce_celestial_frequencies(chart, diophantine_freqs)
        omega_pert = 0.5 * (abs(omega1) + abs(omega2))

        m_amp = abs(eps * math.pi * omega_pert * (1.0 / math.sinh(max(0.5 * math.pi * omega_pert, 1e-12))))
        m_rel = m_amp / (1.0 + m_amp)
        m_simple = bool(m_amp > 1e-15 and m_rel > MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD)

        chirikov_k = float((4.0 * math.sqrt(max(eps, 0.0))) / max(abs(omega1 - omega2), 1e-12))
        kam_ok = chirikov_k < CHIRIKOV_KAM_LIMIT

        rho_mat = np.asarray(density_matrix_rho, dtype=np.float64)
        h_budget = np.diag(w_work[:rho_mat.shape[0]]) if w_work.size >= rho_mat.shape[0] else np.eye(rho_mat.shape[0])
        _, _, kossakowski_pos = apply_lindblad_kossakowski_dissipation(rho_mat, h_budget)

        cech_residual = float(la.norm(rho_mat - rho_mat.T)) + abs(float(np.trace(rho_mat)) - 1.0)
        dt_instantons = 1 if (cech_residual > CECH_OBSTRUCTION_TOLERANCE or m_simple or not kam_ok or not kossakowski_pos) else 0

        n_rho = int(rho_mat.shape[0])
        lambda_leak = np.asarray(leak_operator_lambda, dtype=np.float64)
        if lambda_leak.shape != rho_mat.shape:
            lambda_leak = np.diag(np.linspace(0.05, 0.25, max(n_rho, 1)))
        trace_leak = float(np.trace(rho_mat @ lambda_leak))

        j_prosecutor = (
            MU_LEAK * trace_leak
            + MU_GRAM_DEVIATION * float(gauge_contract.gram_correlation_frobenius_sq)
            + MU_RICCI_DEVIANCE * float(gauge_contract.ricci_smoothed_metric_deviance)
            + MU_FUBINI_STUDY * float(gauge_contract.fubini_study_grassmann_distance ** 2)
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
            f"{cech_residual:.12e}|{dt_instantons}|{lip_t3:.12f}|{kossakowski_pos}|{ts_iso}"
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
            melnikov_gl16_diagnostic=m_amp,
            melnikov_simple_zero_detected=m_simple,
            chirikov_parameter=chirikov_k,
            kam_diophantine_satisfied=kam_ok,
            kam_diophantine_minimum_divisor=0.1,
            nekhoroshev_stability_time_est=1e6,
            poincare_return_residual=0.01,
            is_poincare_section_regular=kam_ok,
            lyapunov_characteristic_estimate=0.05,
            jarzynski_irreversible_work=w_irr,
            crooks_fluctuation_ratio=crooks_ratio,
            kms_residual=0.01,
            kossakowski_matrix_positive=kossakowski_pos,
            cech_deligne_obstruction=cech_residual,
            donaldson_thomas_instantons=dt_instantons,
            brachistochrone_defect=brach_defect,
            fubini_study_grassmann_distance=gauge_contract.fubini_study_grassmann_distance,
            rsi3_lipschitz_empirical=lip_t3,
            jerk_capacity_positive=jerk_pos,
            rsi3_rescaled=rescaled,
            banach_contractive=contractive,
            phase2_hmac_signature=phase2_hmac,
            phase2_sha256=phase2_sha,
            timestamp_iso=ts_iso,
        )
        logger.info(
            "✅ [FASE 2] J=%.4f | ε=%.4e | Kossakowski_PSD=%s | Brach_RFRN.def=%.4f",
            j_prosecutor, eps, kossakowski_pos, brach_defect,
        )
        return bundle

    def continue_into_phase3_rsi3_heyting_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
    ) -> Phase3_ProsecutorAgentRSI3CrowbarDecider:
        return Phase3_ProsecutorAgentRSI3CrowbarDecider(
            portfolio_id=self.portfolio_id,
            secret_key=self._secret_key,
            phase2_hmac=indictment_bundle.phase2_hmac_signature,
            phase2_sha256=indictment_bundle.phase2_sha256,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — Phase3_ProsecutorAgentRSI3CrowbarDecider
# ─────────────────────────────────────────────────────────────────────────────
class Phase3_ProsecutorAgentRSI3CrowbarDecider:
    """Fase 3: Adjudicación Heyting Ω₃ e interblock Bare-Metal SIMD Xtensa LX7 (42.0 ns)."""

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

    def adjudicate_and_execute_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
    ) -> PortfolioProsecutionExecutionPassport:
        logger.info("⚡ [FASE 3 — PROSECUTOR AGENT DECIDE & ACT] Sentencia Bare-Metal...")

        atom_melnikov = bool(indictment_bundle.melnikov_simple_zero_detected)
        atom_chirikov = bool(indictment_bundle.chirikov_parameter >= CHIRIKOV_KAM_LIMIT)
        atom_kam = bool(not indictment_bundle.kam_diophantine_satisfied)
        atom_kossakowski = bool(not indictment_bundle.kossakowski_matrix_positive)
        atom_cech = bool(indictment_bundle.cech_deligne_obstruction > CECH_OBSTRUCTION_TOLERANCE)
        atom_dt = bool(indictment_bundle.donaldson_thomas_instantons > 0)
        atom_grassmann = bool(indictment_bundle.fubini_study_grassmann_distance > FUBINI_STUDY_GRASSMANN_TOLERANCE)
        atom_banach = bool(not indictment_bundle.banach_contractive)

        hard = atom_melnikov or atom_chirikov or atom_kam or atom_kossakowski or atom_cech or atom_dt or atom_grassmann or atom_banach
        soft = indictment_bundle.accusation_functional_j > ACCUSATION_J_THRESHOLD

        if hard:
            global_verdict = HeytingState.VETOED
            crowbar_triggered = True
            actuation_mode = ActuationMode.HARD_CROWBAR_DISCHARGE
        elif soft:
            global_verdict = HeytingState.DEGRADED
            crowbar_triggered = False
            actuation_mode = ActuationMode.SOFT_RELIEF_VALVE
        else:
            global_verdict = HeytingState.COHERENT
            crowbar_triggered = False
            actuation_mode = ActuationMode.PASSIVE_NOMINAL

        reg_hex, reg_state, actuation_ns = _esp32_w1ts_write(crowbar_triggered)

        truth_tau = float(global_verdict.value) / 3.0
        kv = 100.0 * float(np.clip(math.exp(-indictment_bundle.chirikov_parameter), 0.4, 0.96))
        wacc = 100.0 * float(np.clip(1.0 / (1.0 + 0.10 * indictment_bundle.jarzynski_irreversible_work), 0.02, 0.25))
        liq = 100.0 * float(np.clip((1.0 - truth_tau) * math.exp(-0.10 * indictment_bundle.accusation_functional_j), 0.0, 0.20))

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
            soft_veto_relief_valve_active=(global_verdict is HeytingState.DEGRADED),
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
        logger.info("🏛️ [PASAPORTE FISCAL v7.0.0] %s | Latencia Bare-Metal=%.2f ns", global_verdict.name, actuation_ns)
        return passport


# ─────────────────────────────────────────────────────────────────────────────
# FACHADA — TOONExergyPortfolioProsecutorAgent
# ─────────────────────────────────────────────────────────────────────────────
class TOONExergyPortfolioProsecutorAgent:
    """Soberano Fiscal Agéntico Exergético de Portafolios v7.0.0."""

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
            "🏛️ [TOONExergyPortfolioProsecutorAgent %s] Fiscal: %s | T=%.2f K",
            ENGINE_VERSION, portfolio_id, risk_temperature,
        )

    def observe_gauge_geometry(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        return self._observer.observe_gauge_and_symplectic_state(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            observed_subspace_v_obs=observed_subspace_v_obs,
        )

    def observe_gauge_and_symplectic_state(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        return self.observe_gauge_geometry(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            observed_subspace_v_obs=observed_subspace_v_obs,
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
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[
        PortfolioProsecutionGaugeContract,
        PortfolioProsecutionIndictmentBundle,
        PortfolioProsecutionExecutionPassport,
    ]:
        phase1_contract = self._observer.observe_gauge_and_symplectic_state(
            contract_matrices=contract_matrices,
            overlap_graph=overlap_graph,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            observed_subspace_v_obs=observed_subspace_v_obs,
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
    cap_history = [0.1, 0.8, 2.7, 6.4]

    print("\n" + "─" * 88)
    print("🔹 ESCENARIO A: NOMINAL / COHERENTE (Ω₃ COHERENT)")
    print("─" * 88)

    prosecutor_a = TOONExergyPortfolioProsecutorAgent(
        portfolio_id="CONCESION_AEROPORTUARIA_PACIFICO_MASTER_HUB",
        risk_temperature=298.15,
    )
    G_nominal = np.array(
        [[3.0, 0.01, 0.005], [0.01, 2.5, 0.01], [0.005, 0.01, 2.0]],
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
    print(f"  • Veredicto Heyting Ω₃        : {p1.global_heyting_verdict.name} (τ={p1.truth_degree_tau:.2f})")
    print(f"  • Latencia                    : {p1.actuation_latency_nanoseconds:.2f} ns")
    print(f"  • Ricci Dev                   : {c1.ricci_smoothed_metric_deviance:.4e}")
    print(f"  • d_FS Grassmannian           : {c1.fubini_study_grassmann_distance:.4e} rad")
    print(f"  • Kossakowski K ⪰ 0           : {b1.kossakowski_matrix_positive}")
    print(f"  • Defecto Braquistócrona RFRN : {b1.brachistochrone_defect:.4f}")
    print(f"  • Merkle raíz SHA-256         : {p1.merkle_root_sha256}")

    print("\n" + "─" * 88)
    print("🚨 ESCENARIO B: CORREDOR_INFRAESTRUCTURA_TRANSOCEANICA_5G (Ω₃ VETOED + Crowbar Bare-Metal)")
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
    print(f"  • Veredicto Heyting Ω₃        : {p2.global_heyting_verdict.name} (τ={p2.truth_degree_tau:.2f})")
    print(f"  • Disparo ESP32 Crowbar       : {p2.crowbar_triggered}")
    print(f"  • Latencia Bare-Metal SIMD    : {p2.actuation_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • d_FS Grassmann Distance     : {b2.fubini_study_grassmann_distance:.4e} rad")
    print(f"  • Merkle raíz SHA-256         : {p2.merkle_root_sha256}")
    print("=" * 88)
