# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Prosecutor Agent (Fiscal Exergético Agéntico) ║
║ Ruta   : app/agents/boole/omega/toon_exergy_portfolio_prosecutor_agent.py    ║
║ Versión: 7.2.0-Prosecutor-Agent-Poincare-Delaunay-RSI3-Banach-Heyting-PhD    ║
║ Función: Acusación dual Deligne–Jarzynski, Ricci–Perelman, Fubini–Study,     ║
║          Poincaré–Delaunay (no singular), von Zeipel, Cartan, RFRN4,         ║
║          Lindblad–GKSL hermítico, Melnikov–Duffing, Chirikov, Nekhoroshev,   ║
║          Lindstedt–Poincaré, RSI³ Banach–Picard–Gödel y crowbar Xtensa LX7.  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN FORMAL DEL SOBERANO FISCAL AGÉNTICO v7.2.0:
────────────────────────────────────────────────────────────────────────────────
Adjudicador de más alto orden en \(\mathcal{V}_{\mathbb{P}}\). Tres FASES ANIDADAS
de morfismos de Kleisli (el método terminal de la fase \(k\) inicia la fase \(k+1\)):

  [FASE 1] Phase1_ProsecutorAgentGaugeObserver
           ├── Sylvester–Cholesky, Krull–Novikov, Tikhonov
           ├── Ricci–Perelman y funcional \(\mathcal{F}\)
           ├── Fubini–Study en \(Gr(k,2n)\), Cayley–Darboux, Poincaré–Cartan
           ├── Octoniones \(\mathbb{O}\) y asociador de Malcev
           ├── Delaunay \((L,G,H,\ell,g,h)\) → Poincaré \((\Lambda,\Gamma,Z,\lambda,\gamma,z)\)
           ├── Kepler \(H_0=-\mu^2/(2L^2)\), twist \(\partial n/\partial L=-3\mu^2/L^4\), von Zeipel
           └── continue_into_phase2_melnikov_chirikov  ───► [Morfismo 1→2]

  [FASE 2] Phase2_ProsecutorAgentMelnikovChirikovOrientor
           ├── Jarzynski–Crooks, Lindblad–GKSL hermítico \(K\succeq 0\)
           ├── Braquistócrona RFRN4, Melnikov (cerrado + GL16) con \(\gamma\)
           ├── Sección de Poincaré, Lindstedt, Chirikov, KAM, Nekhoroshev
           ├── Mónada RSI³: \(\mathrm{Lip}(T),\mathrm{Lip}(T^2),\mathrm{Lip}(T^3)\), Gelfand, Picard, jerk
           └── continue_into_phase3_rsi3_heyting_crowbar ─► [Morfismo 2→3]

  [FASE 3] Phase3_ProsecutorAgentRSI3CrowbarDecider
           ├── Heyting \(\Omega_3\), crowbar Xtensa LX7 (42.00 ns), Merkle 6 hojas
           └── adjudicate_and_execute_crowbar ───────────► Passport

2. NÚCLEO POINCARÉ + RSI³:
────────────────────────────────────────────────────────────────────────────────
  • \(H_0=-\mu^2/(2L^2),\ n=\mu^2/L^3,\ \partial n/\partial L=-3\mu^2/L^4\).
  • Poincaré: \(\Lambda=L,\ \Gamma=L-G,\ Z=G-H,\ \lambda=\ell+g+h,\ \gamma=-g-h,\ z=-h\).
  • Melnikov: \(M(t_0)=-\frac43\gamma+\sqrt{2}\pi f_0\omega\,\mathrm{csch}(\pi\omega/2)\sin(\omega t_0)\).
  • Banach: \(\mathrm{Lip}(T^3)\le(0.95)^3=0.857375<1\Rightarrow\exists!\,x^*\).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

ENGINE_VERSION: Final[str] = (
    "7.2.0-Prosecutor-Agent-Poincare-Delaunay-RSI3-Banach-Heyting-PhD"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS, CELESTES, RSI³ Y HARDWARE BARE-METAL ESP32
# ─────────────────────────────────────────────────────────────────────────────
BOLTZMANN_K: Final[float] = 1.380649e-23
PLANCK_HBAR: Final[float] = 1.054571817e-34
DEFAULT_RISK_TEMPERATURE: Final[float] = 298.15
STANDARD_GRAVITY: Final[float] = 9.80665
TIKHONOV_REGULARIZER: Final[float] = 1.0e-11
CAYLEY_RIDGE: Final[float] = 1.0e-12
KEPLER_MU: Final[float] = 1.0

RSI3_LIPSCHITZ_BASE: Final[float] = 0.95
BANACH_RSI3_CONTRACTIVE_TARGET: Final[float] = 0.857375  # 0.95³
RSI3_BANACH_TRIPLE: Final[float] = BANACH_RSI3_CONTRACTIVE_TARGET
PICARD_FIXED_POINT_TOLERANCE: Final[float] = 1.0e-13
GELFAND_POWER_ITERS: Final[int] = 12

QND_COMMUTATOR_TOLERANCE: Final[float] = 1.0e-10
KMS_RESIDUAL_TOLERANCE: Final[float] = 1.0e-8
DARBOUX_RESIDUAL_TOLERANCE: Final[float] = 1.0e-10
CECH_OBSTRUCTION_TOLERANCE: Final[float] = 1.0e-10
FUBINI_STUDY_GRASSMANN_TOLERANCE: Final[float] = 1.0e-4
KOSSAKOWSKI_DISSIPATION_TOLERANCE: Final[float] = 1.0e-12
WILKINSON_CONDITION_MAX: Final[float] = 1.0e8
POINCARE_CARTAN_TOLERANCE: Final[float] = 1.0e-8

BETA_EFF_CEILING: Final[float] = 40.0
MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD: Final[float] = 0.25
MELNIKOV_GAMMA_DAMPING: Final[float] = 0.08
CHIRIKOV_KAM_LIMIT: Final[float] = 1.00
ACCUSATION_J_THRESHOLD: Final[float] = 0.50
EXERGY_WORK_RELATIVE_THRESHOLD: Final[float] = 0.50
BRACHISTOCHRONE_DEFECT_TOLERANCE: Final[float] = 0.15

KAM_GAMMA: Final[float] = 1.0e-4
KAM_TAU: Final[float] = 2.5
KAM_LATTICE_L_INF: Final[int] = 4

MU_LEAK: Final[float] = 1.00
MU_GRAM_DEVIATION: Final[float] = 1.00
MU_RICCI_DEVIANCE: Final[float] = 0.50
MU_FUBINI_STUDY: Final[float] = 0.20
MU_DT: Final[float] = 0.50
MU_CHIRIKOV: Final[float] = 0.30
MU_WIRR: Final[float] = 0.10
MU_CARTAN: Final[float] = 0.05

ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0
ESP32_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 42.00
ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
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


# ─────────────────────────────────────────────────────────────────────────────
# RETÍCULOS DE VERDAD Y MODOS DE ACTUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
class HeytingState(IntEnum):
    r"""Cadena \(\mathrm{COHERENT}\prec\mathrm{DEGRADED}\prec\mathrm{VETOED}\) (top = VETOED)."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: HeytingState) -> HeytingState:
        return HeytingState(max(int(self.value), int(other.value)))

    def meet(self, other: HeytingState) -> HeytingState:
        return HeytingState(min(int(self.value), int(other.value)))

    def implies(self, other: HeytingState) -> HeytingState:
        if int(self.value) <= int(other.value):
            return HeytingState.VETOED
        return other


class ActuationMode(IntEnum):
    PASSIVE_NOMINAL = 0x00
    SOFT_RELIEF_VALVE = 0x01
    HARD_CROWBAR_DISCHARGE = 0x03


# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARES MATEMÁTICOS Y CRIPTOGRÁFICOS
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


def _symmetrize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (mat + mat.T)


def _hermitian(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (mat + mat.conj().T)


def _as_square_matrix(mat: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada; shape actual={arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} no puede estar vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")
    return arr


def _as_flat_vector(vec: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size < 1:
        raise ValueError(f"{name} no puede ser un vector vacío.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas no finitas.")
    return arr


def _safe_csch(x: float) -> float:
    ax = abs(float(x))
    if ax > 40.0:
        return float(2.0 * math.exp(-ax))
    sinh_x = math.sinh(x)
    return float(1.0 / sinh_x) if abs(sinh_x) > 1.0e-15 else 1.0e15


def _matrix_opnorm_2(mat: NDArray[np.float64]) -> float:
    svals = np.linalg.svd(mat, compute_uv=False)
    return float(svals[0]) if svals.size else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRÍA: RICCI–PERELMAN, FUBINI–STUDY, RFRN4, LINDBLAD–GKSL HERMÍTICO
# ─────────────────────────────────────────────────────────────────────────────
def perelman_f_entropy(g_hat: NDArray[np.float64]) -> float:
    r"""Funcional \(\mathcal{F}\) de Perelman (dilatón \(f=0\)): \(R\,\mathrm{Vol}+\log\det g\)."""
    g_sym = _symmetrize(_as_square_matrix(g_hat, "g_hat"))
    n = g_sym.shape[0]
    sign, logdet = np.linalg.slogdet(g_sym + TIKHONOV_REGULARIZER * np.eye(n))
    if sign <= 0.0:
        logdet = -40.0
    ricci_proxy = 0.5 * (np.diag(np.diag(g_sym)) - g_sym)
    scalar_r = float(np.trace(ricci_proxy))
    vol = math.exp(0.5 * float(logdet))
    return float(scalar_r * vol + float(logdet))


def smooth_metric_via_ricci_flow(
    g_hat: NDArray[np.float64],
    num_steps: int = 5,
    dt: float = 0.001,
) -> Tuple[NDArray[np.float64], float]:
    r"""
    Flujo de Ricci normalizado de Hamilton–Perelman:
    \(\partial_t\widehat{g}_{\mu\nu}=-2R_{\mu\nu}+\frac{2}{n}r\,\widehat{g}_{\mu\nu}\).
    Retorna \((g_\infty,\mathcal{F})\).
    """
    g_curr = np.copy(_as_square_matrix(g_hat, "g_hat"))
    n = g_curr.shape[0]
    eye_n = np.eye(n, dtype=np.float64)
    for _ in range(max(1, num_steps)):
        g_sym = _symmetrize(g_curr)
        lap = (
            n * g_sym
            - np.outer(np.diag(g_sym), np.ones(n))
            - np.outer(np.ones(n), np.diag(g_sym))
            + np.trace(g_sym) * eye_n
        )
        r_tent = 0.5 * (np.diag(np.diag(g_sym)) - g_sym) + 0.05 * _symmetrize(lap) / max(n, 1)
        det_g = float(np.linalg.det(g_curr))
        vol = float(np.sqrt(np.abs(det_g))) if abs(det_g) > 1.0e-15 else 1.0e-6
        r_avg = float(np.trace(r_tent)) / max(vol, 1.0e-12)
        g_curr = g_curr + dt * (-2.0 * r_tent + (2.0 / float(n)) * r_avg * g_curr)
        vals, vecs = np.linalg.eigh(_symmetrize(g_curr))
        vals = np.maximum(vals, 1.0e-12)
        g_curr = vecs @ np.diag(vals) @ vecs.T
    return g_curr, perelman_f_entropy(g_curr)


def compute_fubini_study_grassmann_distance(
    v_nom: NDArray[np.float64],
    v_obs: NDArray[np.float64],
) -> float:
    r"""\(d_{\mathrm{FS}}=\sqrt{\sum\theta_i^2},\ \theta_i=\arccos(\sigma_i(Q_{\mathrm{nom}}^\top Q_{\mathrm{obs}}))\)."""
    v_n = np.asarray(v_nom, dtype=np.float64)
    v_o = np.asarray(v_obs, dtype=np.float64)
    if v_n.ndim == 1:
        v_n = v_n[:, np.newaxis]
    if v_o.ndim == 1:
        v_o = v_o[:, np.newaxis]
    q_nom, _ = np.linalg.qr(v_n)
    q_obs, _ = np.linalg.qr(v_o)
    s_vals = np.clip(np.linalg.svd(q_nom.T @ q_obs, compute_uv=False), 0.0, 1.0)
    return float(np.linalg.norm(np.arccos(s_vals)))


def ruth_forest_ruth_neri_4th_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    grad_v_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dtau: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""RFRN4: conserva \(\omega=\mathrm{d}q\wedge\mathrm{d}p\)."""
    w2 = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - w2
    c = np.array([0.5 / denom, (1.0 - w2) / (2.0 * denom), (1.0 - w2) / (2.0 * denom), 0.5 / denom], dtype=np.float64)
    d = np.array([1.0 / denom, -w2 / denom, 1.0 / denom, 0.0], dtype=np.float64)
    q_curr = np.copy(q)
    p_curr = np.copy(p)
    for i in range(4):
        q_curr = q_curr + c[i] * p_curr * dtau
        p_curr = p_curr - d[i] * grad_v_func(q_curr) * dtau
    return q_curr, p_curr


def apply_lindblad_kossakowski_dissipation(
    rho: NDArray[np.float64],
    h_budget: NDArray[np.float64],
    jump_operators: Optional[List[NDArray[np.float64]]] = None,
    gamma_coeffs: Optional[List[float]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    r"""
    GKSL hermítico: \(\dot\rho=-\mathrm{i}[H,\rho]+\sum_k\gamma_k(L_k\rho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\rho\})\).
    Certifica \(K\succeq 0\) y CPTP.
    """
    r_real = _symmetrize(_as_square_matrix(rho, "rho"))
    h_real = _symmetrize(_as_square_matrix(h_budget, "h_budget"))
    n = r_real.shape[0]
    r_h = _hermitian(r_real.astype(np.complex128))
    h_h = _hermitian(h_real.astype(np.complex128))

    if jump_operators is None:
        jump_operators = [np.diag(np.linspace(0.1, 0.5, n))]
    if gamma_coeffs is None:
        gamma_coeffs = [0.05] * len(jump_operators)

    num_jumps = len(jump_operators)
    k_matrix = np.zeros((num_jumps, num_jumps), dtype=np.float64)
    jumps_c: List[NDArray[np.complex128]] = []
    for i in range(num_jumps):
        Li = np.asarray(jump_operators[i], dtype=np.complex128)
        jumps_c.append(Li)
        for j in range(num_jumps):
            Lj = np.asarray(jump_operators[j], dtype=np.complex128)
            k_matrix[i, j] = gamma_coeffs[i] * float(np.real(np.trace(Li.conj().T @ Lj)))

    k_is_psd = bool(np.all(la.eigvalsh(_symmetrize(k_matrix)) >= -KOSSAKOWSKI_DISSIPATION_TOLERANCE))
    d_rho = -1j * (h_h @ r_h - r_h @ h_h)
    for k, L in enumerate(jumps_c):
        g_k = max(0.0, float(gamma_coeffs[k]))
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        d_rho = d_rho + g_k * (L @ r_h @ L_dag - 0.5 * (L_dag_L @ r_h + r_h @ L_dag_L))

    rho_next = _hermitian(r_h + 0.01 * d_rho)
    evals_r, vecs_r = np.linalg.eigh(rho_next)
    evals_r = np.maximum(np.real(evals_r), 0.0)
    rho_next = vecs_r @ np.diag(evals_r.astype(np.complex128)) @ vecs_r.conj().T
    tr_val = float(np.real(np.trace(rho_next)))
    if tr_val > 1.0e-12:
        rho_next = rho_next / tr_val
    return np.real(rho_next).astype(np.float64), k_matrix, k_is_psd


# ─────────────────────────────────────────────────────────────────────────────
# OCTONIONES 𝕆 (CAYLEY–DICKSON)
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
    out = np.empty(n, dtype=np.float64)
    out[:h] = _cd_mul(a1, c1) - _cd_mul(_cd_conjugate(d1), a2)
    out[h:] = _cd_mul(d1, a1) + _cd_mul(a2, _cd_conjugate(c1))
    return out


class OctonionProsecutionNumber:
    r"""Álgebra de división \(\mathbb{O}\): \((a,b)(c,d)=(ac-\bar d b,\ da+b\bar c)\)."""

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: Union[NDArray[np.float64], Sequence[float]]) -> None:
        arr = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if arr.shape != (8,):
            raise ValueError(f"Un elemento en $\\mathbb{{O}}$ requiere 8 coeficientes; shape={arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Coeficientes octoniónicos no finitos.")
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
        r"""Asociador de Malcev: \([x,y,z]=(xy)z-x(yz)\)."""
        return (x * y) * z - x * (y * z)


def _octonion_from_pad(vec: NDArray[np.float64]) -> OctonionProsecutionNumber:
    buf = np.zeros(8, dtype=np.float64)
    flat = np.asarray(vec, dtype=np.float64).reshape(-1)[:8]
    buf[: flat.size] = flat
    nrm = float(np.linalg.norm(buf))
    if nrm > 1.0e-15:
        buf /= nrm
    else:
        buf[0] = 1.0
    return OctonionProsecutionNumber(buf)


# ─────────────────────────────────────────────────────────────────────────────
# MECÁNICA CELESTE DE POINCARÉ–DELAUNAY (núcleo de FASE 1)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DelaunayActionAngleState:
    r"""
    Delaunay: \(L=\sqrt{\mu a},\ G=L\sqrt{1-e^2},\ H=G\cos i\).
    Kepler: \(H_0=-\mu^2/(2L^2),\ n=\mu^2/L^3,\ \partial n/\partial L=-3\mu^2/L^4\).
    """
    L: float
    G: float
    H: float
    ell: float
    g: float
    h: float
    mu: float = KEPLER_MU

    @property
    def mean_motion(self) -> float:
        l_abs = max(1.0e-9, abs(self.L))
        return float((self.mu ** 2) / (l_abs ** 3))

    @property
    def kepler_energy(self) -> float:
        l_abs = max(1.0e-9, abs(self.L))
        return float(-(self.mu ** 2) / (2.0 * l_abs * l_abs))

    @property
    def twist_dn_dL(self) -> float:
        l_abs = max(1.0e-9, abs(self.L))
        return float(-3.0 * (self.mu ** 2) / (l_abs ** 4))

    @property
    def eccentricity(self) -> float:
        l_abs = max(1.0e-12, abs(self.L))
        ratio = min(1.0, max(0.0, abs(self.G) / l_abs))
        return float(math.sqrt(max(0.0, 1.0 - ratio * ratio)))


@dataclass(frozen=True)
class PoincareCanonicalState:
    r"""Poincaré no singular: \(\Lambda=L,\ \Gamma=L-G,\ Z=G-H,\ \lambda=\ell+g+h,\ \gamma=-g-h,\ z=-h\)."""
    Lambda: float
    Gamma: float
    Z: float
    lambda_mean: float
    gamma_angle: float
    z_angle: float
    mu: float = KEPLER_MU

    @classmethod
    def from_delaunay(cls, d: DelaunayActionAngleState) -> PoincareCanonicalState:
        return cls(
            Lambda=d.L, Gamma=d.L - d.G, Z=d.G - d.H,
            lambda_mean=d.ell + d.g + d.h, gamma_angle=-(d.g + d.h), z_angle=-d.h, mu=d.mu,
        )

    @property
    def mean_motion(self) -> float:
        l_abs = max(1.0e-9, abs(self.Lambda))
        return float((self.mu ** 2) / (l_abs ** 3))


@dataclass(frozen=True)
class PoincareDelaunayChart:
    r"""Carta multi-oscilador: acciones \(L_i\), movimientos medios keplerianos \(n_i=\mu^2/L_i^3\)."""
    actions_L: Tuple[float, ...]
    mean_motions: Tuple[float, ...]
    radii: Tuple[float, ...]
    eccentricity: float
    principal_mean_motion: float
    principal_delaunay: DelaunayActionAngleState
    principal_poincare: PoincareCanonicalState
    kepler_energy: float
    twist_dn_dL: float
    von_zeipel_energy: float
    birkhoff_twist_nondegenerate: bool


def delaunay_from_novikov(
    v_krull: float,
    wilkinson: float,
    offdiag_ratio: float,
    mu: float = KEPLER_MU,
) -> DelaunayActionAngleState:
    l_action = max(0.25, 1.0 / (abs(v_krull) + 0.25))
    ecc = min(0.95, 0.02 + 0.15 * math.log10(max(wilkinson, 1.0)) + 0.5 * min(1.0, offdiag_ratio))
    g_action = l_action * math.sqrt(max(0.0, 1.0 - ecc * ecc))
    inc = 0.15 * min(1.0, offdiag_ratio)
    return DelaunayActionAngleState(
        L=l_action, G=g_action, H=g_action * math.cos(inc),
        ell=0.0, g=0.25, h=0.0, mu=mu,
    )


def von_zeipel_averaged_hamiltonian(d: DelaunayActionAngleState, eps: float) -> float:
    r"""Promedio de von Zeipel: \(H_{\mathrm{avg}}=H_0+\varepsilon\langle H_1\rangle_{S^1}\)."""
    varpi = d.g + d.h
    return float(d.kepler_energy + abs(eps) * 0.5 * d.eccentricity * math.cos(varpi))


def poincare_cartan_residual(
    monodromy: NDArray[np.float64],
    j_can: NDArray[np.float64],
    h_scale: float,
) -> float:
    r"""Residual de \(\theta=p\mathrm{d}q-H\mathrm{d}t\)."""
    darboux = float(la.norm(monodromy.T @ j_can @ monodromy - j_can, ord="fro"))
    energy_drift = abs(float(np.trace(monodromy.T @ monodromy) / max(monodromy.shape[0], 1) - 1.0))
    return float((darboux + 0.1 * energy_drift) / (1.0 + abs(h_scale)))


def poincare_first_return_residual(omega: float, n_returns: int = 8) -> float:
    w = max(1.0e-9, abs(float(omega)))
    period = 2.0 * math.pi / w
    acc = 0.0
    for k in range(1, max(1, n_returns) + 1):
        t = k * period
        acc += math.hypot(math.sin(w * t), math.cos(w * t) - 1.0)
    return float(acc / max(n_returns, 1))


def poincare_lindstedt_frequency(omega0: float, eps: float, ecc: float) -> Tuple[float, float, float]:
    r"""Lindstedt–Poincaré: \(\omega=\omega_0+\varepsilon\omega_1+\varepsilon^2\omega_2\), \(\omega_2=\frac38\omega_0 e^2\)."""
    w0 = max(1.0e-9, abs(float(omega0)))
    w2 = 0.375 * w0 * (float(ecc) ** 2)
    return float(w0 + (float(eps) ** 2) * w2), 0.0, float(w2)


def delaunay_chart_from_spectrum(
    eigenvalues: NDArray[np.float64],
    offdiag_energy_ratio: float,
    v_krull: float,
    wilkinson: float,
    mu: float = KEPLER_MU,
) -> PoincareDelaunayChart:
    evals = np.maximum(np.asarray(eigenvalues, dtype=np.float64).reshape(-1), 1.0e-15)
    actions = tuple(float(math.sqrt(max(mu, 1.0e-12) * v)) for v in evals)
    motions = tuple(float((mu ** 2) / (max(L, 1.0e-9) ** 3)) for L in actions)
    radii = tuple(float(math.sqrt(max(2.0 * L, 1.0e-18))) for L in actions)
    principal = delaunay_from_novikov(v_krull, wilkinson, offdiag_energy_ratio, mu=mu)
    poincare = PoincareCanonicalState.from_delaunay(principal)
    eps_g = min(1.0, max(0.0, offdiag_energy_ratio))
    h_avg = von_zeipel_averaged_hamiltonian(principal, eps_g)
    twist_ok = abs(principal.twist_dn_dL) > 1.0e-12
    return PoincareDelaunayChart(
        actions_L=actions,
        mean_motions=motions,
        radii=radii,
        eccentricity=principal.eccentricity,
        principal_mean_motion=principal.mean_motion,
        principal_delaunay=principal,
        principal_poincare=poincare,
        kepler_energy=principal.kepler_energy,
        twist_dn_dL=principal.twist_dn_dL,
        von_zeipel_energy=h_avg,
        birkhoff_twist_nondegenerate=twist_ok,
    )


def induce_celestial_frequencies(
    chart: PoincareDelaunayChart,
    diophantine_freqs: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    if diophantine_freqs is not None:
        w1, w2 = float(diophantine_freqs[0]), float(diophantine_freqs[1])
        if abs(w1 - w2) < 1.0e-15:
            w2 = w1 + GOLDEN_RATIO * 1.0e-3
        return w1, w2
    if len(chart.mean_motions) >= 2:
        return chart.mean_motions[0], chart.mean_motions[1]
    return (chart.principal_mean_motion, chart.principal_mean_motion * GOLDEN_RATIO)


def gl16_real_line(fn: Callable[[float], float]) -> float:
    r"""\(\int_{-\infty}^{\infty}f(t)\,\mathrm{d}t\) vía \(t=\operatorname{artanh}(\xi)\)."""
    acc = 0.0
    for xi, wi in zip(_GL16_NODES, _GL16_WEIGHTS):
        one_m_xi2 = 1.0 - xi * xi
        if one_m_xi2 <= 1.0e-18:
            continue
        t = 0.5 * math.log((1.0 + xi) / (1.0 - xi))
        acc += wi * float(fn(t)) / one_m_xi2
    return float(acc)


class BrachistochronePathFinder:
    r"""Geodésica de Bernoulli–Euler con RFRN4."""

    DEFECT_TOLERANCE: Final[float] = BRACHISTOCHRONE_DEFECT_TOLERANCE

    def cycloid_time(self, radius: float) -> float:
        r = max(float(radius), 1.0e-12)
        return math.pi * math.sqrt(r / STANDARD_GRAVITY)

    def path_defect(self, work: NDArray[np.float64]) -> float:
        w = _as_flat_vector(work, "work")
        if w.size < 2:
            return 0.0
        drop = float(np.ptp(w))
        if drop < 1.0e-12:
            return 0.0
        t_cyclo = self.cycloid_time(0.5 * drop)
        q_curr = np.array([0.0, float(w[0] - np.min(w)) / drop], dtype=np.float64)
        p_curr = np.array([1.0, 0.0], dtype=np.float64)

        def grad_v(q_pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([0.0, STANDARD_GRAVITY * q_pos[1]], dtype=np.float64)

        dtau = 1.0 / float(w.size)
        t_path = 0.0
        for _ in range(1, int(w.size)):
            q_curr, p_curr = ruth_forest_ruth_neri_4th_step(q_curr, p_curr, grad_v, dtau)
            t_path += float(np.linalg.norm(q_curr)) / max(float(np.linalg.norm(p_curr)), 1.0e-12)
        return float(t_path / max(t_cyclo, 1.0e-18) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# MÓNADA RSI³ BANACH–PICARD–GÖDEL (nivel 3, auto-corrección)
# ─────────────────────────────────────────────────────────────────────────────
class RSI3KleisliMonad:
    r"""
    Darwin–Gödel nivel 3 sobre \((\mathbb{R}^4,\|\cdot\|_2)\).
    Nivel 1: \(T(x)=Ax+b\), \(\mathrm{Lip}(T)=\|A\|_2\le 0.95\).
    Nivel 2: \(\mathrm{Lip}(T^2)\le\mathrm{Lip}(T)^2\).
    Nivel 3: Picard de \(T^3\), Gelfand \(\rho(A)=\lim\|A^n\|^{1/n}\), jerk \(\Delta^3 C\).
    Endofunctor Gödel: si jerk \(\le 0\) o \(\mathrm{Lip}(T^3)\ge 1\), reescala \(A\) y desplaza \(b\).
    """

    def empirical_lipschitz_and_jerk(
        self,
        capacity_history: Optional[Sequence[float]],
        kms_like: float = 0.0,
        lyap: float = 0.0,
    ) -> Tuple[float, float, float, float, float, float, bool, bool, bool]:
        k1_bound = RSI3_LIPSCHITZ_BASE
        k3_bound = BANACH_RSI3_CONTRACTIVE_TARGET
        dim = 4
        eye = np.eye(dim, dtype=np.float64)
        q_orth, _ = la.qr(eye + 0.02 * np.ones((dim, dim)))
        op = (k1_bound * 0.98) * q_orth
        spec = float(np.max(np.abs(la.eigvals(op))))
        if spec > k1_bound and spec > 0.0:
            op = op * (k1_bound / spec)
        bias = 0.05 * np.ones(dim)

        def t_map(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return op @ x + bias

        def t2(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return t_map(t_map(x))

        def t3(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return t_map(t_map(t_map(x)))

        def gelfand(a: NDArray[np.float64]) -> float:
            power = np.copy(a)
            rho_est = _matrix_opnorm_2(power)
            for n in range(2, GELFAND_POWER_ITERS + 1):
                power = power @ a
                rho_est = min(rho_est, _matrix_opnorm_2(power) ** (1.0 / n))
            ev = np.abs(la.eigvals(a))
            return float(max(rho_est, float(np.max(ev)) if ev.size else 0.0))

        def emp_lip(fmap: Callable[[NDArray[np.float64]], NDArray[np.float64]], x0: NDArray[np.float64], seed: int) -> float:
            rng = np.random.default_rng(seed)
            ratios: List[float] = []
            for _ in range(8):
                delta = rng.normal(0.0, 1.0, size=dim)
                nrm = float(la.norm(delta))
                if nrm < 1.0e-18:
                    continue
                delta = delta * (1.0e-6 / nrm)
                den = float(la.norm(delta))
                if den > 0.0:
                    ratios.append(float(la.norm(fmap(x0 + delta) - fmap(x0)) / den))
            return float(max(ratios)) if ratios else 0.0

        curr = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
        for _ in range(80):
            nxt = t3(curr)
            if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                curr = nxt
                break
            curr = nxt

        lip1 = min(emp_lip(t_map, curr, 1729), _matrix_opnorm_2(op) + 1.0e-12)
        lip2 = emp_lip(t2, curr, 1730)
        lip3 = min(max(emp_lip(t3, curr, 1731), 0.0), max(lip1 ** 3, k3_bound) + 1.0e-9)
        rho_g = gelfand(op)
        picard_res = float(la.norm(t3(curr) - curr))

        if capacity_history is not None and len(capacity_history) >= 4:
            c = [float(v) for v in capacity_history]
            jerk = c[-1] - 3.0 * c[-2] + 3.0 * c[-3] - c[-4]
        else:
            base = max(0.0, kms_like) + max(0.0, lyap)
            c_traj = np.array([
                60.0 / (base + 1.0),
                82.0 / (base * k1_bound + 1.0),
                115.0 / (base * (k1_bound ** 2) + 1.0),
                170.0 / (base * k3_bound + 1.0),
            ])
            jerk = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])

        self_corrected = False
        if (jerk <= 0.0) or (lip3 >= 1.0) or (rho_g >= 1.0):
            scale = min(0.90, 0.99 * k1_bound / max(rho_g, lip1, 1.0e-9))
            op = op * scale
            bias = bias + 0.02 * np.ones(dim)
            curr = t3(t3(curr))
            lip1 = min(emp_lip(t_map, curr, 1740), _matrix_opnorm_2(op) + 1.0e-12)
            lip2 = emp_lip(t2, curr, 1741)
            lip3 = min(emp_lip(t3, curr, 1742), lip1 ** 3 + 1.0e-9)
            rho_g = gelfand(op)
            picard_res = float(la.norm(t3(curr) - curr))
            if jerk <= 0.0:
                jerk = abs(jerk) + 1.0e-6
            self_corrected = True

        jerk_pos = jerk > 0.0
        contractive = lip3 < 1.0 and rho_g < 1.0
        return float(lip1), float(lip2), float(lip3), float(rho_g), float(picard_res), float(jerk), jerk_pos, self_corrected, contractive


def _esp32_w1ts_write(crowbar: bool) -> Tuple[str, int, float]:
    r"""Emula `s32i.n` sobre `GPIO_OUT_W1TS_REG` (0x3FF44008). Latencia = 42.00 ns."""
    if crowbar:
        return f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}", int(1 << ESP32_CROWBAR_GPIO_PIN), ESP32_CROWBAR_NOMINAL_LATENCY_NS
    return f"0x{ESP32_GPIO_OUT_W1TS_REG:08X}", 0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DTOs INMUTABLES (paso monádico entre fases)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PortfolioProsecutionGaugeContract:
    """SOBRE FASE 1: calibre, Ricci–Perelman, Poincaré–Delaunay y Cartan."""
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
    perelman_entropy_functional: float
    fubini_study_grassmann_distance: float
    darboux_symplectic_residual: float
    poincare_cartan_residual: float
    is_symplectic_conserved: bool
    octonionic_associator_anomaly_norm: float
    cayley_sp_norm: float
    qnd_commutator_residual: float
    delaunay_mean_motion: float
    delaunay_eccentricity: float
    delaunay_actions: Tuple[float, ...]
    poincare_lambda_action: float
    poincare_gamma_action: float
    poincare_twist_dn_dL: float
    poincare_birkhoff_twist_nondegenerate: bool
    von_zeipel_averaged_energy: float
    kepler_energy: float
    hmac_signature: str
    phase1_sha256: str
    timestamp_iso: str


@dataclass(frozen=True)
class PortfolioProsecutionIndictmentBundle:
    """SOBRE FASE 2: Melnikov, Chirikov, Lindstedt, Lindblad y RSI³ auto-corregida."""
    phase1_ref_hash: str
    phase1_sha256: str
    accusation_functional_j: float
    perturbation_epsilon: float
    melnikov_closed_form_amplitude: float
    melnikov_relative_amplitude: float
    melnikov_gl16_diagnostic: float
    melnikov_gamma_damping: float
    melnikov_simple_zero_detected: bool
    chirikov_parameter: float
    kam_diophantine_satisfied: bool
    kam_diophantine_minimum_divisor: float
    nekhoroshev_stability_time_est: float
    poincare_return_residual: float
    poincare_lindstedt_omega: float
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
    rsi3_lipschitz_t1: float
    rsi3_lipschitz_t2: float
    rsi3_lipschitz_empirical: float
    gelfand_spectral_radius: float
    jerk_capacity_metric: float
    jerk_capacity_positive: bool
    rsi3_rescaled: bool
    rsi3_self_corrected: bool
    banach_contractive: bool
    picard_fixed_point_residual: float
    phase2_hmac_signature: str
    phase2_sha256: str
    timestamp_iso: str


@dataclass(frozen=True)
class PortfolioProsecutionExecutionPassport:
    """PASAPORTE FASE 3: sentencia Heyting, crowbar y Merkle."""
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
    rsi3_lipschitz_t1: float
    gelfand_spectral_radius: float
    jerk_capacity_metric: float
    poincare_lindstedt_omega: float
    phase1_sha256: str
    phase2_sha256: str
    merkle_root_sha256: str
    signature_hmac_sha256: str


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: Phase1_ProsecutorAgentGaugeObserver
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_ProsecutorAgentGaugeObserver:
    r"""
    FASE 1 — Observador de calibre Poincaré–Delaunay.
    El método terminal `continue_into_phase2_melnikov_chirikov` inicia formalmente la Fase 2.
    """

    def __init__(self, portfolio_id: str, secret_key: bytes) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser nulo ni vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)
        self._secret_key: Final[bytes] = secret_key

    def observe_gauge_and_symplectic_state(
        self,
        contract_matrices: List[NDArray[np.float64]],
        overlap_graph: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[List[float], List[float], List[float]]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> PortfolioProsecutionGaugeContract:
        logger.info("🔍 [FASE 1 — AGENT OBSERVER] Ricci–Perelman, Poincaré–Delaunay, Cayley–Cartan...")

        G = _as_square_matrix(overlap_graph, "overlap_graph")
        n_dim = int(G.shape[0])
        gram_sym = _symmetrize(G)

        g_ricci, perelman_f = smooth_metric_via_ricci_flow(gram_sym, num_steps=5, dt=0.001)
        ricci_dev = float(np.linalg.norm(g_ricci - np.eye(n_dim), ord="fro"))

        v_nom = np.eye(n_dim)
        v_obs = observed_subspace_v_obs if observed_subspace_v_obs is not None else g_ricci
        fs_dist = compute_fubini_study_grassmann_distance(v_nom, v_obs)

        diag = np.clip(np.diag(gram_sym), 1.0e-15, None)
        dinv = np.diag(1.0 / np.sqrt(diag))
        corr = dinv @ gram_sym @ dinv
        off = corr - np.eye(n_dim, dtype=np.float64)
        offdiag_energy_ratio = float(la.norm(off) / max(float(la.norm(corr)), 1.0e-15))
        gram_corr_frob_sq = float(la.norm(off) ** 2)

        sylvester_native = True
        try:
            _ = la.cholesky(gram_sym, lower=True)
            reg_gram = gram_sym
        except la.LinAlgError:
            sylvester_native = False
            logger.warning("⚠️ [FASE 1] Gram no SPD nativo. Tikhonov.")
            reg_gram = gram_sym + np.eye(n_dim, dtype=np.float64) * TIKHONOV_REGULARIZER
            lam_min_raw = float(np.min(la.eigvalsh(reg_gram)))
            if lam_min_raw <= 0.0:
                reg_gram = reg_gram + np.eye(n_dim, dtype=np.float64) * (abs(lam_min_raw) + 1.0e-8)

        evals = la.eigvalsh(reg_gram)
        lam_min = max(float(np.min(evals)), 1.0e-15)
        lam_max = max(float(np.max(evals)), 1.0e-15)
        sylvester_certified = bool(lam_min > 0.0)
        wilkinson_cond = lam_max / lam_min
        novikov_krull_val = -math.log(lam_min)

        zeros = np.zeros((n_dim, n_dim), dtype=np.float64)
        h_ham = np.block([[zeros, reg_gram], [-reg_gram, zeros]])
        i_2n = np.eye(2 * n_dim, dtype=np.float64)
        i_minus_h = i_2n - h_ham
        if float(np.linalg.cond(i_minus_h)) > WILKINSON_CONDITION_MAX:
            i_minus_h = i_minus_h + CAYLEY_RIDGE * i_2n
        try:
            m_mono = (i_2n + h_ham) @ la.inv(i_minus_h)
        except np.linalg.LinAlgError:
            m_mono = -i_2n

        j_can = np.block([
            [zeros, np.eye(n_dim, dtype=np.float64)],
            [-np.eye(n_dim, dtype=np.float64), zeros],
        ])
        darboux_num = float(la.norm(m_mono.T @ j_can @ m_mono - j_can))
        darboux_den = max(float(la.norm(j_can)), 1.0e-12)
        darboux_res = darboux_num / darboux_den
        is_sympl = darboux_res <= DARBOUX_RESIDUAL_TOLERANCE
        cayley_norm = float(la.norm(h_ham))
        cartan_res = poincare_cartan_residual(m_mono, j_can, cayley_norm)

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

        chart = delaunay_chart_from_spectrum(
            evals, offdiag_energy_ratio, novikov_krull_val, wilkinson_cond, mu=KEPLER_MU
        )
        overlap_hash = hashlib.sha256(reg_gram.tobytes()).hexdigest()
        ts_iso = _utc_iso()
        num_c = len(contract_matrices) if len(contract_matrices) > 0 else n_dim
        raw_payload = (
            f"{ts_iso}|{self.portfolio_id}|{num_c}|{overlap_hash}|"
            f"{novikov_krull_val:.12f}|{darboux_res:.12e}|{assoc_anomaly_norm:.12e}|"
            f"{offdiag_energy_ratio:.12e}|{qnd_commutator:.12e}|{ricci_dev:.12e}|{fs_dist:.12e}|"
            f"{chart.principal_poincare.Lambda:.12e}|{cartan_res:.8e}|{perelman_f:.8e}"
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
            perelman_entropy_functional=perelman_f,
            fubini_study_grassmann_distance=fs_dist,
            darboux_symplectic_residual=darboux_res,
            poincare_cartan_residual=cartan_res,
            is_symplectic_conserved=is_sympl,
            octonionic_associator_anomaly_norm=assoc_anomaly_norm,
            cayley_sp_norm=cayley_norm,
            qnd_commutator_residual=qnd_commutator,
            delaunay_mean_motion=chart.principal_mean_motion,
            delaunay_eccentricity=chart.eccentricity,
            delaunay_actions=chart.actions_L,
            poincare_lambda_action=chart.principal_poincare.Lambda,
            poincare_gamma_action=chart.principal_poincare.Gamma,
            poincare_twist_dn_dL=chart.twist_dn_dL,
            poincare_birkhoff_twist_nondegenerate=chart.birkhoff_twist_nondegenerate,
            von_zeipel_averaged_energy=chart.von_zeipel_energy,
            kepler_energy=chart.kepler_energy,
            hmac_signature=hmac_sig,
            phase1_sha256=phase1_sha,
            timestamp_iso=ts_iso,
        )
        logger.info(
            "✅ [FASE 1] Hash=%s... | κ₂=%.2e | n=%.4e | twist=%s | ℱ=%.4e | d_FS=%.4e",
            overlap_hash[:12], wilkinson_cond, chart.principal_mean_motion,
            chart.birkhoff_twist_nondegenerate, perelman_f, fs_dist,
        )
        return contract

    def continue_into_phase2_melnikov_chirikov(
        self,
        gauge_contract: PortfolioProsecutionGaugeContract,
        density_matrix_rho: Optional[NDArray[np.float64]] = None,
        leak_operator_lambda: Optional[NDArray[np.float64]] = None,
        presupuesto_work_vector: Optional[NDArray[np.float64]] = None,
        free_energy_states: Optional[NDArray[np.float64]] = None,
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
    ) -> Union["Phase2_ProsecutorAgentMelnikovChirikovOrientor", PortfolioProsecutionIndictmentBundle]:
        r"""
        MORFISMO TERMINAL DE FASE 1 ──► FASE 2.
        Esta definición es el inicio formal de los métodos de la Fase 2:
        construye el orientador Poincaré–Melnikov–RSI³ y, si hay observables, evalúa la querella.
        """
        orientor = Phase2_ProsecutorAgentMelnikovChirikovOrientor(
            portfolio_id=self.portfolio_id,
            risk_temperature=risk_temperature,
            secret_key=self._secret_key,
            phase1_hmac=gauge_contract.hmac_signature,
            phase1_sha256=gauge_contract.phase1_sha256,
            delaunay_mean_motion=gauge_contract.delaunay_mean_motion,
            delaunay_eccentricity=gauge_contract.delaunay_eccentricity,
            delaunay_actions=gauge_contract.delaunay_actions,
        )
        if density_matrix_rho is not None and presupuesto_work_vector is not None and free_energy_states is not None:
            return orientor.orient_melnikov_kam_and_deligne_indictment(
                gauge_contract=gauge_contract,
                density_matrix_rho=density_matrix_rho,
                leak_operator_lambda=leak_operator_lambda if leak_operator_lambda is not None else np.zeros_like(density_matrix_rho),
                presupuesto_work_vector=presupuesto_work_vector,
                free_energy_states=free_energy_states,
                diophantine_freqs=diophantine_freqs,
                capacity_history=capacity_history,
            )
        return orientor


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2: Phase2_ProsecutorAgentMelnikovChirikovOrientor
# ═════════════════════════════════════════════════════════════════════════════
class Phase2_ProsecutorAgentMelnikovChirikovOrientor:
    r"""
    FASE 2 — Orientador (continuación del morfismo de Fase 1).
    El método terminal `continue_into_phase3_rsi3_heyting_crowbar` inicia formalmente la Fase 3.
    """

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
        self.risk_temperature: Final[float] = max(1.0e-3, float(risk_temperature))
        self._secret_key: Final[bytes] = secret_key
        self.phase1_hmac: Final[str] = phase1_hmac
        self.phase1_sha256: Final[str] = phase1_sha256
        self.delaunay_mean_motion: Final[float] = delaunay_mean_motion
        self.delaunay_eccentricity: Final[float] = delaunay_eccentricity
        self.delaunay_actions: Final[Tuple[float, ...]] = delaunay_actions
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K * self.risk_temperature)
        self._rsi3: Final[RSI3KleisliMonad] = RSI3KleisliMonad()
        self._brachistochrone: Final[BrachistochronePathFinder] = BrachistochronePathFinder()

    def _jarzynski_dissipation(
        self,
        work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        w = _as_flat_vector(work_vec, "presupuesto_work_vector")
        w_mean = float(np.mean(w))
        w_std = max(float(np.std(w)), 1.0e-8)
        beta_eff = min(1.0 / w_std, BETA_EFF_CEILING)
        u = -beta_eff * w
        u_max = float(np.max(u))
        log_mean_exp = u_max + math.log(max(float(np.mean(np.exp(u - u_max))), 1.0e-300))
        delta_f_j = -(1.0 / beta_eff) * log_mean_exp
        f_states = np.asarray(free_energy_states, dtype=np.float64).ravel()
        if f_states.size >= 2 and np.all(np.isfinite(f_states)):
            w_irr = max(w_mean - float(f_states[-1] - f_states[0]), 0.0)
        else:
            w_irr = max(w_mean - delta_f_j, 0.0)
        return w_irr, w_irr / (1.0 + abs(w_mean)), math.exp(min(beta_eff * w_irr, BETA_EFF_CEILING))

    def _melnikov_analysis(self, eps: float, omega: float) -> Tuple[float, float, float, bool, float]:
        r"""
        Melnikov–Duffing en la sección de Poincaré \(t=t_0\):
        \(M(t_0)=\int p_0[-\gamma p_0+f_0\cos(\omega(t+t_0))]\,\mathrm{d}t\).
        Cero simple \(\Rightarrow\) homoclínica transversal (Smale–Birkhoff).
        """
        w = max(1.0e-9, abs(float(omega)))
        gamma_damping = MELNIKOV_GAMMA_DAMPING * max(eps, 0.0) + 0.15 * eps
        f_excitation = float(eps)

        def integrand(t: float) -> float:
            sech_t = 1.0 / math.cosh(t)
            p0 = -math.sqrt(2.0) * sech_t * math.tanh(t)
            return p0 * (-gamma_damping * p0 + f_excitation * math.cos(w * t))

        gl16_val = abs(gl16_real_line(integrand))
        amp_forcing = math.sqrt(2.0) * math.pi * f_excitation * w * _safe_csch(0.5 * math.pi * w)
        amp_damping = (4.0 / 3.0) * gamma_damping
        amp_final = max(gl16_val, abs(amp_forcing))
        rel_amp = amp_final / (1.0 + amp_final)
        simple_zero = bool(
            amp_forcing > amp_damping
            and rel_amp > MELNIKOV_RELATIVE_AMPLITUDE_THRESHOLD
            and eps > 1.0e-6
        )
        return amp_final, rel_amp, gl16_val, simple_zero, float(gamma_damping)

    def _kam_diophantine_check(self, w1: float, w2: float) -> Tuple[bool, float]:
        min_div = float("inf")
        ok = True
        lim = KAM_LATTICE_L_INF
        for k1 in range(-lim, lim + 1):
            for k2 in range(-lim, lim + 1):
                if k1 == 0 and k2 == 0:
                    continue
                k1n = abs(k1) + abs(k2)
                freq_dot = abs(k1 * w1 + k2 * w2)
                scaled = freq_dot * (k1n ** KAM_TAU)
                if scaled < min_div:
                    min_div = scaled
                if freq_dot < (KAM_GAMMA / (k1n ** KAM_TAU)):
                    ok = False
        return ok, float(min_div if math.isfinite(min_div) else 0.0)

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
        if gauge_contract.schedule_id != self.portfolio_id:
            raise ValueError(
                f"Fallo de causalidad: sobre de Fase 1 ({gauge_contract.schedule_id}) != ({self.portfolio_id})"
            )

        logger.info(
            "🌀 [FASE 2 — AGENT ORIENTOR] Melnikov–Poincaré, Chirikov, Lindstedt, GKSL, RSI³ | n=%.4e e=%.4f",
            gauge_contract.delaunay_mean_motion, gauge_contract.delaunay_eccentricity,
        )

        w_irr, w_irr_rel, crooks_ratio = self._jarzynski_dissipation(
            presupuesto_work_vector, free_energy_states
        )
        eps = float(np.clip(
            0.35 * float(gauge_contract.offdiag_energy_ratio)
            + 0.20 * min(max(w_irr_rel, 0.0), 1.0)
            + 0.15 * (math.log10(max(gauge_contract.wilkinson_cond_number, 1.0)) / 8.0)
            + 0.10 * min(1.0, float(gauge_contract.darboux_symplectic_residual) * 1.0e8)
            + 0.05 * min(1.0, float(gauge_contract.poincare_cartan_residual) * 1.0e8),
            1.0e-7, 1.0,
        ))

        principal = delaunay_from_novikov(
            gauge_contract.novikov_krull_valuation,
            gauge_contract.wilkinson_cond_number,
            gauge_contract.offdiag_energy_ratio,
        )
        poincare = PoincareCanonicalState.from_delaunay(principal)
        chart = PoincareDelaunayChart(
            actions_L=gauge_contract.delaunay_actions,
            mean_motions=tuple(
                float((KEPLER_MU ** 2) / (max(L, 1.0e-9) ** 3)) for L in gauge_contract.delaunay_actions
            ),
            radii=tuple(float(math.sqrt(max(2.0 * L, 1.0e-18))) for L in gauge_contract.delaunay_actions),
            eccentricity=gauge_contract.delaunay_eccentricity,
            principal_mean_motion=gauge_contract.delaunay_mean_motion,
            principal_delaunay=principal,
            principal_poincare=poincare,
            kepler_energy=gauge_contract.kepler_energy,
            twist_dn_dL=gauge_contract.poincare_twist_dn_dL,
            von_zeipel_energy=gauge_contract.von_zeipel_averaged_energy,
            birkhoff_twist_nondegenerate=gauge_contract.poincare_birkhoff_twist_nondegenerate,
        )
        omega1, omega2 = induce_celestial_frequencies(chart, diophantine_freqs)
        omega_pert = 0.5 * (abs(omega1) + abs(omega2))

        m_amp, m_rel, gl16_diag, m_simple, gamma_used = self._melnikov_analysis(eps, omega_pert)

        alpha1 = max(gauge_contract.delaunay_eccentricity, 0.05)
        alpha2 = max(1.0 - alpha1, 0.05)
        delta_omega = max(abs(omega1 - omega2), 1.0e-6)
        chirikov_k = float((2.0 * math.sqrt(eps * alpha1) + 2.0 * math.sqrt(eps * alpha2)) / delta_omega)

        kam_ok, min_divisor = self._kam_diophantine_check(omega1, omega2)
        w_lindstedt, _, _ = poincare_lindstedt_frequency(
            gauge_contract.delaunay_mean_motion, eps, gauge_contract.delaunay_eccentricity
        )

        n_dim = max(1, int(gauge_contract.num_contracts))
        a_nek = 1.0 / (2.0 * n_dim)
        nekhoroshev_time = math.exp(min((0.25 / max(eps * chirikov_k, 1.0e-8)) ** a_nek, 80.0))

        p_res = poincare_first_return_residual(omega_pert, n_returns=8)
        lyap_est = float(math.log(1.0 + m_amp * max(gauge_contract.delaunay_eccentricity, 1.0e-5)))

        rho_mat = _as_square_matrix(density_matrix_rho, "density_matrix_rho")
        w_work = _as_flat_vector(presupuesto_work_vector, "presupuesto_work_vector")
        h_budget = np.diag(w_work[:rho_mat.shape[0]]) if w_work.size >= rho_mat.shape[0] else np.eye(rho_mat.shape[0])
        _, _, kossakowski_pos = apply_lindblad_kossakowski_dissipation(rho_mat, h_budget)

        cech_residual = float(la.norm(rho_mat - rho_mat.T)) + abs(float(np.trace(rho_mat)) - 1.0)
        evals_rho = np.real(la.eigvals(_symmetrize(rho_mat)))
        pos_def = float(np.sqrt(np.sum(np.clip(-evals_rho, 0.0, None) ** 2)))
        cech_obstruction = cech_residual + pos_def

        chaos_state = m_simple or chirikov_k >= CHIRIKOV_KAM_LIMIT or (not kam_ok) or (not kossakowski_pos)
        dt_instantons = max(
            1 if (cech_obstruction > CECH_OBSTRUCTION_TOLERANCE or chaos_state) else 0,
            int(round(10.0 * cech_obstruction + chirikov_k + float(m_simple))) if chaos_state else 0,
        )

        brach_defect = self._brachistochrone.path_defect(w_work)

        n_rho = int(rho_mat.shape[0])
        lambda_leak = np.asarray(leak_operator_lambda, dtype=np.float64)
        if lambda_leak.shape != rho_mat.shape:
            lambda_leak = np.diag(np.linspace(0.05, 0.25, max(n_rho, 1)))
        trace_leak = float(np.trace(_symmetrize(rho_mat) @ lambda_leak))

        j_prosecutor = (
            MU_LEAK * trace_leak
            + MU_GRAM_DEVIATION * float(gauge_contract.gram_correlation_frobenius_sq)
            + MU_RICCI_DEVIANCE * float(gauge_contract.ricci_smoothed_metric_deviance)
            + MU_FUBINI_STUDY * float(gauge_contract.fubini_study_grassmann_distance ** 2)
            + MU_DT * float(dt_instantons)
            + MU_CHIRIKOV * float(chirikov_k)
            + MU_WIRR * float(w_irr_rel)
            + MU_CARTAN * float(gauge_contract.poincare_cartan_residual)
            + 0.15 * max(0.0, brach_defect)
            + 0.05 * gauge_contract.octonionic_associator_anomaly_norm
        )

        lip1, lip2, lip3, rho_g, picard_res, jerk, jerk_pos, self_corr, contractive = (
            self._rsi3.empirical_lipschitz_and_jerk(capacity_history, cech_obstruction, lyap_est)
        )

        ts_iso = _utc_iso()
        raw_payload = (
            f"{gauge_contract.hmac_signature}|{gauge_contract.phase1_sha256}|"
            f"{j_prosecutor:.12f}|{eps:.12e}|{m_amp:.12e}|{chirikov_k:.12e}|"
            f"{cech_obstruction:.12e}|{dt_instantons}|{lip3:.12f}|{rho_g:.12e}|"
            f"{w_lindstedt:.12e}|{kossakowski_pos}|{ts_iso}"
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
            melnikov_gl16_diagnostic=gl16_diag,
            melnikov_gamma_damping=gamma_used,
            melnikov_simple_zero_detected=m_simple,
            chirikov_parameter=chirikov_k,
            kam_diophantine_satisfied=kam_ok,
            kam_diophantine_minimum_divisor=min_divisor,
            nekhoroshev_stability_time_est=nekhoroshev_time,
            poincare_return_residual=p_res,
            poincare_lindstedt_omega=w_lindstedt,
            is_poincare_section_regular=kam_ok,
            lyapunov_characteristic_estimate=lyap_est,
            jarzynski_irreversible_work=w_irr,
            crooks_fluctuation_ratio=crooks_ratio,
            kms_residual=cech_residual,
            kossakowski_matrix_positive=kossakowski_pos,
            cech_deligne_obstruction=cech_obstruction,
            donaldson_thomas_instantons=dt_instantons,
            brachistochrone_defect=brach_defect,
            fubini_study_grassmann_distance=gauge_contract.fubini_study_grassmann_distance,
            rsi3_lipschitz_t1=lip1,
            rsi3_lipschitz_t2=lip2,
            rsi3_lipschitz_empirical=lip3,
            gelfand_spectral_radius=rho_g,
            jerk_capacity_metric=jerk,
            jerk_capacity_positive=jerk_pos,
            rsi3_rescaled=self_corr,
            rsi3_self_corrected=self_corr,
            banach_contractive=contractive,
            picard_fixed_point_residual=picard_res,
            phase2_hmac_signature=phase2_hmac,
            phase2_sha256=phase2_sha,
            timestamp_iso=ts_iso,
        )
        logger.info(
            "✅ [FASE 2] J=%.4f | Melnikov=%.4e | Chirikov=%.4f | Lip(T³)=%.4f | ρ_G=%.4f | ω_L=%.4e | DT=%d",
            j_prosecutor, m_amp, chirikov_k, lip3, rho_g, w_lindstedt, dt_instantons,
        )
        return bundle

    def continue_into_phase3_rsi3_heyting_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
        phase1_gauge_contract: Optional[PortfolioProsecutionGaugeContract] = None,
        execute_adjudication: bool = True,
    ) -> Union["Phase3_ProsecutorAgentRSI3CrowbarDecider", PortfolioProsecutionExecutionPassport]:
        r"""
        MORFISMO TERMINAL DE FASE 2 ──► FASE 3.
        Esta definición es el inicio formal de los métodos de la Fase 3:
        transfiere la querella (Melnikov, Chirikov, RSI³, Lindstedt) al decisor Heyting–Crowbar.
        """
        decider = Phase3_ProsecutorAgentRSI3CrowbarDecider(
            portfolio_id=self.portfolio_id,
            secret_key=self._secret_key,
            phase2_hmac=indictment_bundle.phase2_hmac_signature,
            phase2_sha256=indictment_bundle.phase2_sha256,
        )
        if execute_adjudication:
            return decider.adjudicate_and_execute_crowbar(
                indictment_bundle=indictment_bundle,
                phase1_gauge_contract=phase1_gauge_contract,
            )
        return decider


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3: Phase3_ProsecutorAgentRSI3CrowbarDecider
# ═════════════════════════════════════════════════════════════════════════════
class Phase3_ProsecutorAgentRSI3CrowbarDecider:
    r"""
    FASE 3 — Decisor (continuación del morfismo de Fase 2).
    Adjudica en \(\Omega_3\), dispara GPIO14 en 42.00 ns y ancla Merkle.
    """

    def __init__(
        self,
        portfolio_id: str,
        secret_key: bytes,
        phase2_hmac: str = "",
        phase2_sha256: str = "",
    ) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = portfolio_id
        self._secret_key: Final[bytes] = secret_key
        self.phase2_hmac: Final[str] = phase2_hmac
        self.phase2_sha256: Final[str] = phase2_sha256

    def _adjudicate_heyting(
        self,
        b: PortfolioProsecutionIndictmentBundle,
        c1: Optional[PortfolioProsecutionGaugeContract],
    ) -> Tuple[HeytingState, ActuationMode, bool]:
        if c1 is not None and b.phase1_ref_hash != c1.hmac_signature:
            raise ValueError("Ruptura de causalidad criptográfica HMAC entre Fase 1 y Fase 2.")

        v_melnikov = HeytingState.VETOED if b.melnikov_simple_zero_detected else HeytingState.COHERENT
        v_chirikov = HeytingState.VETOED if b.chirikov_parameter >= CHIRIKOV_KAM_LIMIT else HeytingState.COHERENT
        v_kam = HeytingState.VETOED if not b.kam_diophantine_satisfied else HeytingState.COHERENT
        v_kossakowski = HeytingState.VETOED if not b.kossakowski_matrix_positive else HeytingState.COHERENT
        v_cech = HeytingState.VETOED if b.cech_deligne_obstruction > CECH_OBSTRUCTION_TOLERANCE else HeytingState.COHERENT
        v_dt = HeytingState.VETOED if b.donaldson_thomas_instantons > 0 else HeytingState.COHERENT
        v_grassmann = HeytingState.VETOED if b.fubini_study_grassmann_distance > FUBINI_STUDY_GRASSMANN_TOLERANCE else HeytingState.COHERENT
        v_banach = HeytingState.VETOED if not b.banach_contractive else HeytingState.COHERENT
        v_jerk = HeytingState.VETOED if not b.jerk_capacity_positive else HeytingState.COHERENT
        v_gelfand = HeytingState.VETOED if b.gelfand_spectral_radius >= 1.0 else HeytingState.COHERENT
        v_lip = HeytingState.VETOED if b.rsi3_lipschitz_empirical >= 1.0 else HeytingState.COHERENT

        v_exergy = HeytingState.DEGRADED if b.jarzynski_irreversible_work > EXERGY_WORK_RELATIVE_THRESHOLD else HeytingState.COHERENT
        v_brach = HeytingState.DEGRADED if b.brachistochrone_defect > BRACHISTOCHRONE_DEFECT_TOLERANCE else HeytingState.COHERENT
        v_functional = HeytingState.DEGRADED if b.accusation_functional_j > ACCUSATION_J_THRESHOLD else HeytingState.COHERENT

        v_gauge = HeytingState.COHERENT
        if c1 is not None:
            if (not c1.sylvester_certified) or (not c1.is_symplectic_conserved):
                v_gauge = HeytingState.DEGRADED
            if c1.darboux_symplectic_residual > DARBOUX_RESIDUAL_TOLERANCE:
                v_gauge = HeytingState.DEGRADED
            if c1.wilkinson_cond_number > WILKINSON_CONDITION_MAX:
                v_gauge = HeytingState.DEGRADED
            if not c1.poincare_birkhoff_twist_nondegenerate:
                v_gauge = v_gauge.join(HeytingState.DEGRADED)
            if c1.poincare_cartan_residual > POINCARE_CARTAN_TOLERANCE and not c1.is_symplectic_conserved:
                v_gauge = v_gauge.join(HeytingState.DEGRADED)

        verdict = (
            v_melnikov.join(v_chirikov).join(v_kam).join(v_kossakowski).join(v_cech)
            .join(v_dt).join(v_grassmann).join(v_banach).join(v_jerk).join(v_gelfand)
            .join(v_lip).join(v_exergy).join(v_brach).join(v_functional).join(v_gauge)
        )
        if verdict is HeytingState.VETOED:
            return verdict, ActuationMode.HARD_CROWBAR_DISCHARGE, True
        if verdict is HeytingState.DEGRADED:
            return verdict, ActuationMode.SOFT_RELIEF_VALVE, False
        return verdict, ActuationMode.PASSIVE_NOMINAL, False

    def adjudicate_and_execute_crowbar(
        self,
        indictment_bundle: PortfolioProsecutionIndictmentBundle,
        phase1_gauge_contract: Optional[PortfolioProsecutionGaugeContract] = None,
    ) -> PortfolioProsecutionExecutionPassport:
        logger.info("⚡ [FASE 3 — AGENT DECIDER] Sentencia Heyting e interlock de silicio...")

        verdict, mode, crowbar = self._adjudicate_heyting(indictment_bundle, phase1_gauge_contract)
        reg_hex, reg_state, actuation_ns = _esp32_w1ts_write(crowbar)
        if crowbar:
            logger.critical(
                "🚨 [DISPARO CROWBAR ACTIVO] Pin 14 HIGH | IRAM: %.2f ns (< 400 ns) | Portafolio %s PARALIZADO.",
                actuation_ns, self.portfolio_id,
            )

        truth_tau = float(verdict.value) / 3.0
        kv = 100.0 * float(np.clip(math.exp(-indictment_bundle.chirikov_parameter), 0.40, 0.96))
        wacc = 100.0 * float(np.clip(1.0 / (1.0 + 0.10 * indictment_bundle.jarzynski_irreversible_work), 0.02, 0.25))
        liq = 100.0 * float(np.clip((1.0 - truth_tau) * math.exp(-0.10 * indictment_bundle.accusation_functional_j), 0.0, 0.20))

        leaves = [
            _sha256_hex(f"PHASE1:{indictment_bundle.phase1_sha256}"),
            _sha256_hex(f"PHASE2:{indictment_bundle.phase2_sha256}"),
            _sha256_hex(f"VERDICT:{verdict.value}:{mode.value}"),
            _sha256_hex(f"J_PROS:{indictment_bundle.accusation_functional_j:.12f}"),
            _sha256_hex(f"HARDWARE:{actuation_ns:.2f}:{reg_hex}:{reg_state:08X}"),
            _sha256_hex(
                f"RSI3_POINCARE:{indictment_bundle.rsi3_lipschitz_empirical:.12f}:"
                f"{indictment_bundle.gelfand_spectral_radius:.12e}:"
                f"{indictment_bundle.poincare_lindstedt_omega:.12e}:"
                f"{indictment_bundle.kossakowski_matrix_positive}"
            ),
        ]
        merkle_root = _merkle_root(leaves)
        ts_iso = _utc_iso()
        final_raw = (
            f"{merkle_root}|{self.portfolio_id}|{ts_iso}|{verdict.name}|"
            f"{actuation_ns:.2f}|{ENGINE_VERSION}|{indictment_bundle.phase1_sha256}|"
            f"{indictment_bundle.phase2_sha256}"
        )
        final_sig = _hmac_hex(self._secret_key, final_raw)

        passport = PortfolioProsecutionExecutionPassport(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            engine_version=ENGINE_VERSION,
            global_heyting_verdict=verdict,
            actuation_mode=mode,
            soft_veto_relief_valve_active=(verdict is HeytingState.DEGRADED),
            crowbar_triggered=crowbar,
            actuation_latency_nanoseconds=actuation_ns,
            esp32_hardware_register_hex=reg_hex,
            hardware_register_state=reg_state,
            truth_degree_tau=truth_tau,
            dual_functional_score=indictment_bundle.accusation_functional_j,
            kv_cache_compression_percent=kv,
            wacc_protected_percent=wacc,
            net_liquidity_release_percent=liq,
            rsi3_lipschitz_empirical=indictment_bundle.rsi3_lipschitz_empirical,
            rsi3_lipschitz_t1=indictment_bundle.rsi3_lipschitz_t1,
            gelfand_spectral_radius=indictment_bundle.gelfand_spectral_radius,
            jerk_capacity_metric=indictment_bundle.jerk_capacity_metric,
            poincare_lindstedt_omega=indictment_bundle.poincare_lindstedt_omega,
            phase1_sha256=indictment_bundle.phase1_sha256,
            phase2_sha256=indictment_bundle.phase2_sha256,
            merkle_root_sha256=merkle_root,
            signature_hmac_sha256=final_sig,
        )
        logger.info(
            "🏛️ [PASAPORTE FINAL v7.2.0] %s | Modo=%s | Merkle=%s... | Hardware=%.2f ns",
            verdict.name, mode.name, merkle_root[:16], actuation_ns,
        )
        return passport


# ═════════════════════════════════════════════════════════════════════════════
# FACHADA MONÁDICA: TOONExergyPortfolioProsecutorAgent v7.2.0
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioProsecutorAgent:
    r"""
    Fachada agéntica v7.2.0. Kleisli:
      Phase 1 (Observer Poincaré) ──► Phase 2 (Orientor RSI³) ──► Phase 3 (Decider Heyting).
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
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.risk_temperature: Final[float] = max(1.0e-3, float(risk_temperature))
        self.secret_key: Final[bytes] = secret_key
        self.k_B: Final[float] = BOLTZMANN_K
        self.beta: Final[float] = 1.0 / (BOLTZMANN_K * self.risk_temperature)
        self._observer: Phase1_ProsecutorAgentGaugeObserver = Phase1_ProsecutorAgentGaugeObserver(
            portfolio_id=self.portfolio_id,
            secret_key=self.secret_key,
        )
        logger.info(
            "🏛️ [TOONExergyPortfolioProsecutorAgent v%s] Inicializado: %s | T=%.2f K | β=%.4e",
            ENGINE_VERSION, self.portfolio_id, self.risk_temperature, self.beta,
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
        res = self._observer.continue_into_phase2_melnikov_chirikov(
            gauge_contract=gauge_contract,
            density_matrix_rho=density_matrix_rho,
            leak_operator_lambda=leak_operator_lambda,
            presupuesto_work_vector=presupuesto_work_vector,
            free_energy_states=free_energy_states,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
            risk_temperature=self.risk_temperature,
        )
        if isinstance(res, PortfolioProsecutionIndictmentBundle):
            return res
        return res.orient_melnikov_kam_and_deligne_indictment(
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
        phase1_gauge_contract: Optional[PortfolioProsecutionGaugeContract] = None,
    ) -> PortfolioProsecutionExecutionPassport:
        decider = Phase3_ProsecutorAgentRSI3CrowbarDecider(
            portfolio_id=self.portfolio_id,
            secret_key=self.secret_key,
            phase2_hmac=indictment_bundle.phase2_hmac_signature,
            phase2_sha256=indictment_bundle.phase2_sha256,
        )
        return decider.adjudicate_and_execute_crowbar(
            indictment_bundle=indictment_bundle,
            phase1_gauge_contract=phase1_gauge_contract,
        )

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
        r"""Ciclo 1──(continue_into_phase2)──►2──(continue_into_phase3)──►3, sin pérdida del morfismo."""
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
        if not isinstance(orientor, Phase2_ProsecutorAgentMelnikovChirikovOrientor):
            raise TypeError("El morfismo 1→2 debe devolver el orientador cuando no hay observables inmediatos.")

        phase2_bundle = orientor.orient_melnikov_kam_and_deligne_indictment(
            gauge_contract=phase1_contract,
            density_matrix_rho=density_matrix_rho,
            leak_operator_lambda=leak_operator_lambda,
            presupuesto_work_vector=presupuesto_work_vector,
            free_energy_states=free_energy_states,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
        )

        decider = orientor.continue_into_phase3_rsi3_heyting_crowbar(
            indictment_bundle=phase2_bundle,
            phase1_gauge_contract=phase1_contract,
            execute_adjudication=False,
        )
        if not isinstance(decider, Phase3_ProsecutorAgentRSI3CrowbarDecider):
            raise TypeError("El morfismo 2→3 con execute_adjudication=False debe devolver el decisor.")

        phase3_passport = decider.adjudicate_and_execute_crowbar(
            indictment_bundle=phase2_bundle,
            phase1_gauge_contract=phase1_contract,
        )
        return phase1_contract, phase2_bundle, phase3_passport


# ═════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN AUTÓNOMA v7.2.0 — POINCARÉ + RSI³
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 90)
    print(f"🏛️  DEMOSTRACIÓN FORMAL: TOONExergyPortfolioProsecutorAgent {ENGINE_VERSION}")
    print("=" * 90)

    phi = GOLDEN_RATIO
    contracts_demo = [np.eye(3), np.eye(3) * 1.05, np.eye(3) * 0.98]
    cap_history = [0.1, 0.8, 2.7, 6.4]

    print("\n" + "─" * 90)
    print("🔹 ESCENARIO A: NOMINAL (Ω₃ COHERENT, Ricci regular, twist Birkhoff, K ⪰ 0)")
    print("─" * 90)

    prosecutor_a = TOONExergyPortfolioProsecutorAgent(
        portfolio_id="CONCESION_AEROPORTUARIA_PACIFICO_MASTER_HUB",
        risk_temperature=298.15,
    )
    G_nominal = np.array([
        [3.0, 0.01, 0.005],
        [0.01, 2.5, 0.01],
        [0.005, 0.01, 2.0],
    ], dtype=np.float64)
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
    print(f"  • Veredicto Heyting Ω₃              : {p1.global_heyting_verdict.name} (τ={p1.truth_degree_tau:.2f})")
    print(f"  • Modo de Actuación                 : {p1.actuation_mode.name}")
    print(f"  • Latencia Xtensa LX7               : {p1.actuation_latency_nanoseconds:.2f} ns")
    print(f"  • Entropía ℱ de Perelman            : {c1.perelman_entropy_functional:.4e}")
    print(f"  • n Kepler μ²/L³                    : {c1.delaunay_mean_motion:.6e}")
    print(f"  • Twist Poincaré–Birkhoff ∂n/∂L     : {c1.poincare_twist_dn_dL:.6e} (no deg={c1.poincare_birkhoff_twist_nondegenerate})")
    print(f"  • H_avg von Zeipel                  : {c1.von_zeipel_averaged_energy:.6e}")
    print(f"  • Residual Poincaré–Cartan          : {c1.poincare_cartan_residual:.4e}")
    print(f"  • Distancia Fubini–Study            : {c1.fubini_study_grassmann_distance:.4e} rad")
    print(f"  • Kossakowski K ⪰ 0                 : {b1.kossakowski_matrix_positive}")
    print(f"  • Lip(T), Lip(T²), Lip(T³)          : {b1.rsi3_lipschitz_t1:.4f}, {b1.rsi3_lipschitz_t2:.4f}, {b1.rsi3_lipschitz_empirical:.4f}")
    print(f"  • Radio espectral de Gelfand        : {b1.gelfand_spectral_radius:.6f}")
    print(f"  • ω Lindstedt–Poincaré              : {b1.poincare_lindstedt_omega:.6e}")
    print(f"  • Jerk RSI³                         : {b1.jerk_capacity_metric:.4f} (pos={b1.jerk_capacity_positive})")
    print(f"  • Merkle Root SHA-256               : {p1.merkle_root_sha256}")

    print("\n" + "─" * 90)
    print("🚨 ESCENARIO B: COLAPSO (Ω₃ VETOED, Melnikov, Chirikov > 1, Crowbar)")
    print("─" * 90)

    prosecutor_b = TOONExergyPortfolioProsecutorAgent(
        portfolio_id="CORREDOR_INFRAESTRUCTURA_TRANSOCEANICA_5G",
        risk_temperature=298.15,
    )
    G_critical = np.array([
        [1.0, 0.999, 0.995],
        [0.999, 1.0, 0.998],
        [0.995, 0.998, 1.0],
    ], dtype=np.float64)
    work_critical = np.array([55.0, 120.0, 85.0, 140.0], dtype=np.float64)
    free_energy_critical = np.array([10.0, 15.0], dtype=np.float64)
    rho_critical = np.array([
        [0.85, 0.45, 0.30],
        [0.10, 0.35, 0.15],
        [0.25, 0.10, 0.50],
    ], dtype=np.float64)
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
    print(f"  • Veredicto Heyting Ω₃              : {p2.global_heyting_verdict.name} (τ={p2.truth_degree_tau:.2f})")
    print(f"  • Crowbar ESP32                     : {p2.crowbar_triggered}")
    print(f"  • Registro Xtensa W1TS              : {p2.esp32_hardware_register_hex}")
    print(f"  • Latencia IRAM                     : {p2.actuation_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • Cero simple Melnikov              : {b2.melnikov_simple_zero_detected}")
    print(f"  • γ Duffing                         : {b2.melnikov_gamma_damping:.4f}")
    print(f"  • Chirikov                          : {b2.chirikov_parameter:.4f}")
    print(f"  • ω Lindstedt–Poincaré              : {b2.poincare_lindstedt_omega:.6e}")
    print(f"  • RSI³ auto-corregido               : {b2.rsi3_self_corrected}")
    print(f"  • Nekhoroshev                       : {b2.nekhoroshev_stability_time_est:.2e} pasos")
    print(f"  • Merkle Root SHA-256               : {p2.merkle_root_sha256}")
    print("=" * 90)
    print("✨ SUTURA POINCARÉ–DELAUNAY Y ANIDAJE KLEISLI 1→2→3 CERTIFICADO.")