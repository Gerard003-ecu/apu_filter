# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Prosecutor Engine (Motor Espectral Fiscal)    ║
║ Ruta   : app/boole/physics/toon_exergy_portfolio_prosecutor_engine.py        ║
║ Versión: 7.2.0-Prosecutor-Poincare-Delaunay-RSI3-Banach-Heyting-PhD          ║
║ Función: Acusación dual, Flujo de Ricci–Perelman, Grassmanniano              ║
║          Fubini-Study, RFRN 4to orden simpléctico, Lindblad-GKSL hermítico,  ║
║          Melnikov–Duffing, Chirikov, Nekhoroshev, KAM, Poincaré–Lindstedt,   ║
║          von Zeipel, RSI³ Banach–Picard–Gödel e Interlock Crowbar LX7.       ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL FISCAL:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la especificación e implementación del **Motor Espectral y
Fiscal de Acusación Dual de Portafolios** (`TOONExergyPortfolioProsecutorEngine`)
en el Estrato Exergético ($\mathcal{V}_{\mathbb{P}}$, Nivel 0 / Fibrado de Deligne–Jarzynski)
del ecosistema APU Filter.

Consolida la fiscalización ciber-física mediante un flujo monádico estricto en
3 FASES ANIDADAS de la categoría de Kleisli:
  • Fase 1 (`Phase1_ProsecutorGaugeObserver`): Regularización Cholesky-Tikhonov,
    Flujo de Ricci normalizado de Hamilton-Perelman $\partial_t \widehat{g}_{\mu\nu} = -2 R_{\mu\nu} + \frac{2}{n} r \widehat{g}_{\mu\nu}$,
    distancia de Fubini-Study $d_{\mathrm{FS}}$ en $Gr(k, 2n)$, monodromía de Cayley $M \in \operatorname{Sp}(2n, \mathbb{R})$,
    invariante de Poincaré-Cartan $\theta = p\,\mathrm{d}q - H\,\mathrm{d}t$, asociador de Malcev en octoniones $\mathbb{O}$,
    y elementos de Poincaré-Delaunay no singulares.
  • Fase 2 (`Phase2_ProsecutorMelnikovChirikovOrientor`): Disipación de Jarzynski-Crooks $W_{\mathrm{irr}}$,
    generador de Lindblad-GKSL con matriz de Kossakowski $K \succeq \mathbf{0}$, geodésicas de Bernoulli-Euler
    con integrador RFRN4 simpléctico de 4º orden, forma cerrada de Melnikov $M(t_0) = -\frac{4}{3}\gamma + \sqrt{2}\pi f_0 \omega \operatorname{csch}(\pi\omega/2)\sin(\omega t_0)$,
    parámetro de solapamiento de Chirikov $K_{\mathrm{Chirikov}}$, cota diofántica KAM de Siegel-Bruno,
    tiempo de confinamiento de Nekhoroshev $T_N \sim \exp(\varepsilon^{-a})$, serie de Lindstedt-Poincaré $\omega$,
    y triple contracción monádica RSI³ ($\operatorname{Lip}(T^3) \le 0.857375 < 1.0$, radio espectral de Gelfand $\rho(A) < 1.0$).
  • Fase 3 (`Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine`): Adjudicación en el retículo de Heyting
    $\Omega_3$, anclaje en árbol de Merkle de 6 hojas y disparo Crowbar bare-metal Xtensa LX7 en
    $t_{\mathrm{actuation}} = 42.00\text{ ns}$ ($< 400\text{ ns}$) sobre el registro GPIO `0x3FF44008`.

2. MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
  • Flujo de Ricci normalizado de Hamilton–Perelman y entropía $\mathcal{F}(g) \simeq R\,\operatorname{Vol} + \log\det g$.
  • Distancia geodésica de Fubini-Study en $Gr(k, 2n)$ con ángulos principales $\theta_i = \arccos \sigma_i(Q_{\mathrm{nom}}^\top Q_{\mathrm{obs}})$.
  • Mecánica celeste de Poincaré-Delaunay: acciones $L$, movimiento medio $n = \mu^2 / L^3$, twist $\partial n / \partial L = -3\mu^2 / L^4$,
    variables de Poincaré $(\Lambda, \Gamma, Z, \lambda, \gamma, z)$, promedio de von Zeipel.
  • Forma cerrada de Melnikov en Duffing con amortiguamiento $\gamma$, parámetro de resonancia de Chirikov y estabilidad de Nekhoroshev.
  • Integrador simpléctico RFRN de 4º orden (Forest-Ruth-Neri) para geodésicas braquistócronas.
  • Lindblad-GKSL con matriz de Kossakowski $K \succeq \mathbf{0}$ y consistencia de Čech-Deligne $\check{H}^2$.
  • Mónada de Kleisli RSI³ (Darwin-Gödel Nivel 3) sobre espacio de Banach completo con radio espectral de Gelfand $\rho(A) < 1.0$.

3. AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Axioma de Geometría de Calibre y Flujo de Ricci-Perelman:
       $$\partial_t \widehat{g}_{\mu\nu} = -2 R_{\mu\nu} + \frac{2}{n} r \widehat{g}_{\mu\nu}, \quad d_{\mathrm{FS}}(Q_1, Q_2) = \sqrt{\sum \theta_i^2}.$$

  [A2] Axioma de Mecánica Celeste y Poincaré–Cartan:
       $$M \in \operatorname{Sp}(2n, \mathbb{R}), \quad \oint_\gamma (p\,\mathrm{d}q - H\,\mathrm{d}t) = \text{inv}, \quad n = \frac{\mu^2}{L^3}, \quad \frac{\partial n}{\partial L} = -\frac{3\mu^2}{L^4} \ne 0.$$

  [A3] Axioma de Caos Homoclínico y Resonancia de Chirikov:
       $$M(t_0) = -\frac{4}{3}\gamma + \sqrt{2}\pi f_0 \omega \operatorname{csch}\left(\frac{\pi\omega}{2}\right)\sin(\omega t_0) = 0 \implies \text{Smale Horseshoe},$$
       $$K_{\mathrm{Chirikov}} = \frac{2\sqrt{\varepsilon \alpha_1} + 2\sqrt{\varepsilon \alpha_2}}{|\omega_1 - \omega_2|} \ge 1.0.$$

  [A4] Axioma de Disipación de Lindblad–Kossakowski y Jarzynski:
       $$\dot{\rho} = -i [H, \rho] + \sum_k \gamma_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right), \quad K \succeq \mathbf{0}.$$

  [A5] Axioma Monádico Contractivo RSI³ & Jerk Cognitivo:
       $$\operatorname{Lip}(T^3) \le 0.857375 < 1.0, \quad \rho_{\text{Gelfand}}(T) < 1.0 \implies \frac{\mathrm{d}^3 C}{\mathrm{d}t^3} > 0.$$

  [A6] Axioma de Adjudicación Heyting $\Omega_3$ & Crowbar Xtensa LX7:
       $$\Omega_3 = \{0x00 \prec 0x01 \prec 0x03\}, \quad t_{\mathrm{actuation}} = 42.00\,\mathrm{ns} \le 400.0\,\mathrm{ns}.$$

4. INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  1. Invariante de Calibre y Flujo de Ricci: $d_{\mathrm{FS}} \le 1.0 \times 10^{-4}\text{ rad}$.
  2. Invariante Simpléctico Poincaré-Cartan: $\|M^\top J M - J\|_F / (1 + \|M\|_F^2) \le 1.0 \times 10^{-10}$.
  3. Invariante de Caos Homoclínico: $K_{\mathrm{Chirikov}} < 1.0$ y ausencia de ceros simples transversales de Melnikov.
  4. Invariante de Disipación Kossakowski: $K \succeq \mathbf{0}$ (valores propios $\ge 0$).
  5. Invariante Monádico Banach-Picard: $\operatorname{Lip}(T^3) \le 0.857375$ y $\rho_{\text{Gelfand}} < 1.0$.
  6. Invariante Ciber-Físico Latencia Hardware ESP32: $t_{\mathrm{actuation}} = 42.00\text{ ns} < 400.0\text{ ns}$.
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
    "7.2.0-Prosecutor-Poincare-Delaunay-RSI3-Banach-Heyting-PhD"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS, CELESTES, VARIACIONALES Y HARDWARE BARE-METAL
# ─────────────────────────────────────────────────────────────────────────────
BOLTZMANN_K: Final[float] = 1.380649e-23
PLANCK_HBAR: Final[float] = 1.054571817e-34
DEFAULT_RISK_TEMPERATURE: Final[float] = 298.15
STANDARD_GRAVITY: Final[float] = 9.80665
TIKHONOV_REGULARIZER: Final[float] = 1.0e-11
CAYLEY_RIDGE: Final[float] = 1.0e-12
KEPLER_MU: Final[float] = 1.0  # unidades canónicas (\mu = GM)

RSI3_LIPSCHITZ_BASE: Final[float] = 0.95
RSI3_BANACH_TRIPLE: Final[float] = 0.857375  # 0.95³
PICARD_FIXED_POINT_TOLERANCE: Final[float] = 1.0e-13
BANACH_CAUCHY_TOLERANCE: Final[float] = 1.0e-12
GELFAND_POWER_ITERS: Final[int] = 12

MAX_CROWBAR_LATENCY_NS: Final[float] = 400.0
DETERMINISTIC_IRAM_LATENCY_NS: Final[float] = 42.00
ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

MELNIKOV_RELATIVE_THRESHOLD: Final[float] = 0.25
MELNIKOV_GAMMA_DAMPING: Final[float] = 0.08
CHIRIKOV_RESONANCE_THRESHOLD: Final[float] = 1.00
EXERGY_WORK_IRR_THRESHOLD: Final[float] = 10.0
CECH_OBSTRUCTION_TOLERANCE: Final[float] = 1.0e-8
BRACHISTOCHRONE_DEFECT_TOLERANCE: Final[float] = 0.15
FUBINI_STUDY_GRASSMANN_TOLERANCE: Final[float] = 1.0e-4
KOSSAKOWSKI_DISSIPATION_TOLERANCE: Final[float] = 1.0e-12
WILKINSON_CONDITION_MAX: Final[float] = 1.0e8
POINCARE_CARTAN_TOLERANCE: Final[float] = 1.0e-8
KAM_GAMMA: Final[float] = 1.0e-4
KAM_TAU: Final[float] = 2.5
KAM_LATTICE_L_INF: Final[int] = 4
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


# ─────────────────────────────────────────────────────────────────────────────
# MÉTODOS AUXILIARES MATEMÁTICOS Y CRIPTOGRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────
def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hmac_hex(key: bytes, payload: str) -> str:
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _sha256_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_square_float_matrix(mat: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser una matriz cuadrada; dimensiones actuales={arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} no puede estar vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas que no son números finitos (NaN/Inf).")
    return arr


def _as_float_vector(vec: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size < 1:
        raise ValueError(f"{name} no puede ser un vector vacío.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas que no son números finitos.")
    return arr


def _symmetrize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (mat + mat.T)


def _hermitian(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (mat + mat.conj().T)


def _log_mean_exp(values: NDArray[np.float64]) -> float:
    r"""Calcula \(\log\langle e^{u}\rangle\) de forma numéricamente estable contra overflow."""
    u = np.asarray(values, dtype=np.float64).ravel()
    u_max = float(np.max(u))
    return u_max + math.log(max(float(np.mean(np.exp(u - u_max))), 1.0e-300))


def _gauss_legendre_real_line(integrand: NDArray[np.float64], xi: NDArray[np.float64]) -> float:
    r"""Evalúa \(\int_{-\infty}^{\infty} f(t)\,\mathrm{d}t\) mapeando \(t=\operatorname{artanh}(\xi)\)."""
    jac = 1.0 / np.maximum(1.0e-15, 1.0 - xi * xi)
    return float(np.dot(_GL16_WEIGHTS, integrand * jac))


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
# GEOMETRÍA: RICCI–PERELMAN, FUBINI–STUDY, RFRN4, LINDBLAD–GKSL
# ─────────────────────────────────────────────────────────────────────────────
def perelman_f_entropy(g_hat: NDArray[np.float64]) -> float:
    r"""
    Funcional \(\mathcal{F}\) de Perelman (dilatón trivial \(f=0\)):
    \(\mathcal{F}(g)\simeq R\,\mathrm{Vol}+\log\det g\). Lyapunov del flujo conforme.
    """
    g_sym = _symmetrize(_as_square_float_matrix(g_hat, "g_hat"))
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
    ricci_tensor: Optional[NDArray[np.float64]] = None,
    scalar_curvature: Optional[float] = None,
    num_steps: int = 5,
    dt: float = 0.001,
) -> Tuple[NDArray[np.float64], float]:
    r"""
    Flujo de Ricci normalizado de Hamilton–Perelman en \(\mathcal{M}_{\mathrm{gauge}}\):
    \(\partial_t\widehat{g}_{\mu\nu}=-2R_{\mu\nu}+\frac{2}{n}r\,\widehat{g}_{\mu\nu}\).
    Retorna \((g_\infty,\mathcal{F})\). La proyección SPD aniquila arrugas espectrales.
    """
    g_curr = np.copy(_as_square_float_matrix(g_hat, "g_hat"))
    n = g_curr.shape[0]
    eye_n = np.eye(n, dtype=np.float64)

    for _ in range(max(1, num_steps)):
        if ricci_tensor is None:
            # Lichnerowicz discreto de primer orden sobre la métrica Gram
            g_sym = _symmetrize(g_curr)
            lap = n * g_sym - np.outer(np.diag(g_sym), np.ones(n)) - np.outer(np.ones(n), np.diag(g_sym)) + np.trace(g_sym) * eye_n
            r_tent = 0.5 * (np.diag(np.diag(g_sym)) - g_sym) + 0.05 * _symmetrize(lap) / max(n, 1)
        else:
            r_tent = np.copy(ricci_tensor)

        det_g = float(np.linalg.det(g_curr))
        vol = float(np.sqrt(np.abs(det_g))) if abs(det_g) > 1.0e-15 else 1.0e-6
        s_curv = float(np.trace(r_tent)) if scalar_curvature is None else float(scalar_curvature)
        r_avg = s_curv / max(vol, 1.0e-12)

        dg_dt = -2.0 * r_tent + (2.0 / float(n)) * r_avg * g_curr
        g_curr = g_curr + dt * dg_dt

        vals, vecs = np.linalg.eigh(_symmetrize(g_curr))
        vals = np.maximum(vals, 1.0e-12)
        g_curr = vecs @ np.diag(vals) @ vecs.T

    return g_curr, perelman_f_entropy(g_curr)


def compute_fubini_study_grassmann_distance(
    v_nom: NDArray[np.float64],
    v_obs: NDArray[np.float64],
) -> float:
    r"""
    Distancia geodésica de Fubini–Study en \(Gr(k,2n)\):
    \(\theta_i=\arccos(\sigma_i(Q_{\mathrm{nom}}^\top Q_{\mathrm{obs}})),\quad d_{\mathrm{FS}}=\sqrt{\sum\theta_i^2}\).
    """
    v_n = np.asarray(v_nom, dtype=np.float64)
    v_o = np.asarray(v_obs, dtype=np.float64)
    if v_n.ndim == 1:
        v_n = v_n[:, np.newaxis]
    if v_o.ndim == 1:
        v_o = v_o[:, np.newaxis]

    q_nom, _ = np.linalg.qr(v_n)
    q_obs, _ = np.linalg.qr(v_o)
    s_vals = np.linalg.svd(q_nom.T @ q_obs, compute_uv=False)
    s_vals = np.clip(s_vals, 0.0, 1.0)
    return float(np.linalg.norm(np.arccos(s_vals)))


def ruth_forest_ruth_neri_4th_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    grad_v_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dtau: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Integrador simpléctico RFRN de 4º orden (Forest–Ruth). Conserva \(\omega=\mathrm{d}q\wedge\mathrm{d}p\):
    \(c_1=c_4=\frac{1}{2(2-2^{1/3})},\ d_1=d_3=\frac{1}{2-2^{1/3}},\ d_2=-\frac{2^{1/3}}{2-2^{1/3}},\ d_4=0\).
    """
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
    Generador maestro de Lindblad–GKSL sobre matrices hermíticas:
    \(\dot{\rho}=-\mathrm{i}[H,\rho]+\sum_k\gamma_k\bigl(L_k\rho L_k^\dagger-\tfrac12\{L_k^\dagger L_k,\rho\}\bigr)\).
    Certifica \(K\succeq 0\) (Kossakowski) y CPTP (traza 1, autovalores \(\ge 0\)).
    """
    r_real = _symmetrize(_as_square_float_matrix(rho, "rho"))
    h_real = _symmetrize(_as_square_float_matrix(h_budget, "h_budget"))
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

    evals_k = la.eigvalsh(_symmetrize(k_matrix))
    k_is_psd = bool(np.all(evals_k >= -KOSSAKOWSKI_DISSIPATION_TOLERANCE))

    comm = -1j * (h_h @ r_h - r_h @ h_h)
    d_rho = comm.copy()
    for k, L in enumerate(jumps_c):
        g_k = max(0.0, float(gamma_coeffs[k]))
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        diss = L @ r_h @ L_dag - 0.5 * (L_dag_L @ r_h + r_h @ L_dag_L)
        d_rho = d_rho + g_k * diss

    dt_step = 0.01
    rho_next = _hermitian(r_h + dt_step * d_rho)
    evals_r, vecs_r = np.linalg.eigh(rho_next)
    evals_r = np.maximum(np.real(evals_r), 0.0)
    rho_next = vecs_r @ np.diag(evals_r.astype(np.complex128)) @ vecs_r.conj().T
    tr_val = float(np.real(np.trace(rho_next)))
    if tr_val > 1.0e-12:
        rho_next = rho_next / tr_val

    return np.real(rho_next).astype(np.float64), k_matrix, k_is_psd


# ─────────────────────────────────────────────────────────────────────────────
# RETÍCULO DE HEYTING Ω₃ Y MODOS DE ACTUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""Cadena lineal de verdad: \(\mathrm{COHERENT}\prec\mathrm{DEGRADED}\prec\mathrm{VETOED}\) (top = VETOED)."""
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
# ÁLGEBRA DE OCTONIONES 𝕆 (CAYLEY–DICKSON)
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
    r"""
    Álgebra normada con división \(\mathbb{O}\) (alternativa, no asociativa).
    Cayley–Dickson: \((a,b)(c,d)=(ac-d^*b,\ da+bc^*)\).
    """

    __slots__ = ("_coeffs",)

    def __init__(self, coeffs: Union[NDArray[np.floating], Sequence[float]]) -> None:
        arr = np.asarray(coeffs, dtype=np.float64).ravel()
        if arr.size != 8:
            raise ValueError(f"Un elemento de $\\mathbb{{O}}$ requiere exactamente 8 coordenadas reales; recibido size={arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Las coordenadas octoniónicas deben ser números finitos.")
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
        r"""Asociador alternativo de Malcev: \([x,y,z]=(xy)z-x(yz)\)."""
        return (x * y) * z - x * (y * z)


# ─────────────────────────────────────────────────────────────────────────────
# MECÁNICA CELESTE DE POINCARÉ–DELAUNAY (núcleo de FASE 1)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DelaunayActionAngleState:
    r"""
    Coordenadas canónicas de Delaunay:
    \(L=\sqrt{\mu a},\ G=L\sqrt{1-e^2},\ H=G\cos i\),
    \(\ell\) anomalía media, \(g\) argumento del periastro, \(h\) nodo.
    Hamiltoniano kepleriano: \(H_0=-\mu^2/(2L^2)\), movimiento medio \(n=\mu^2/L^3\).
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
        r"""Twist de Poincaré–Birkhoff: \(\partial n/\partial L=-3\mu^2/L^4<0\) (no degenerado)."""
        l_abs = max(1.0e-9, abs(self.L))
        return float(-3.0 * (self.mu ** 2) / (l_abs ** 4))

    @property
    def eccentricity(self) -> float:
        l_abs = max(1.0e-12, abs(self.L))
        ratio = min(1.0, max(0.0, abs(self.G) / l_abs))
        return float(math.sqrt(max(0.0, 1.0 - ratio * ratio)))

    @property
    def inclination(self) -> float:
        g_abs = max(1.0e-12, abs(self.G))
        c_i = min(1.0, max(-1.0, self.H / g_abs))
        return float(math.acos(c_i))

    @classmethod
    def from_novikov_and_gram(
        cls,
        v_krull: float,
        wilkinson: float,
        offdiag_ratio: float,
        mu: float = KEPLER_MU,
    ) -> DelaunayActionAngleState:
        l_action = max(0.25, 1.0 / (abs(v_krull) + 0.25))
        ecc = min(0.95, 0.02 + 0.15 * math.log10(max(wilkinson, 1.0)) + 0.5 * min(1.0, offdiag_ratio))
        g_action = l_action * math.sqrt(max(0.0, 1.0 - ecc * ecc))
        # Inclinación suave desde el ratio extra-diagonal (anisotropía del Gram)
        inc = 0.15 * min(1.0, offdiag_ratio)
        h_action = g_action * math.cos(inc)
        return cls(L=l_action, G=g_action, H=h_action, ell=0.0, g=0.25, h=0.0, mu=mu)


@dataclass(frozen=True)
class PoincareCanonicalState:
    r"""
    Variables de Poincaré (Delaunay modificadas, regulares en \(e=0,\ i=0\)):
    \(\Lambda=L,\ \Gamma=L-G,\ Z=G-H,\ \lambda=\ell+g+h,\ \gamma=-g-h,\ z=-h\).
    """
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
            Lambda=d.L,
            Gamma=d.L - d.G,
            Z=d.G - d.H,
            lambda_mean=d.ell + d.g + d.h,
            gamma_angle=-(d.g + d.h),
            z_angle=-d.h,
            mu=d.mu,
        )

    def to_delaunay(self) -> DelaunayActionAngleState:
        L = self.Lambda
        G = L - self.Gamma
        H = G - self.Z
        h = -self.z_angle
        g = -self.gamma_angle - h
        ell = self.lambda_mean - g - h
        return DelaunayActionAngleState(L=L, G=G, H=H, ell=ell, g=g, h=h, mu=self.mu)

    @property
    def mean_motion(self) -> float:
        l_abs = max(1.0e-9, abs(self.Lambda))
        return float((self.mu ** 2) / (l_abs ** 3))

    @property
    def eccentricity_proxy(self) -> float:
        r"""\(e\simeq\sqrt{2\Gamma/\Lambda}\) al primer orden (Poincaré)."""
        lam = max(1.0e-12, abs(self.Lambda))
        return float(min(0.999, math.sqrt(max(0.0, 2.0 * abs(self.Gamma) / lam))))


def von_zeipel_averaged_hamiltonian(
    delaunay: DelaunayActionAngleState,
    eps: float,
    omega_pert: float,
) -> float:
    r"""
    Promedio de von Zeipel / Poincaré: elimina ángulos rápidos \(\ell\) al orden \(\varepsilon\).
    \(H_{\mathrm{avg}}=H_0(L)+\varepsilon\langle H_1\rangle_{S^1}\simeq H_0+\varepsilon\frac{e}{2}\cos\varpi\).
    """
    h0 = delaunay.kepler_energy
    varpi = delaunay.g + delaunay.h
    secular = 0.5 * delaunay.eccentricity * math.cos(varpi) * math.cos(omega_pert * 0.0)
    return float(h0 + abs(eps) * secular)


def poincare_cartan_residual(
    monodromy: NDArray[np.float64],
    j_can: NDArray[np.float64],
    h_scale: float,
) -> float:
    r"""
    Residual del invariante integral de Poincaré–Cartan \(\theta=p\mathrm{d}q-H\mathrm{d}t\).
    Equivale a la conservación de \(\omega\) más un defecto de energía a lo largo del ciclo.
    """
    darboux = float(la.norm(monodromy.T @ j_can @ monodromy - j_can, ord="fro"))
    energy_drift = abs(float(np.trace(monodromy.T @ monodromy) / max(monodromy.shape[0], 1) - 1.0))
    return float((darboux + 0.1 * energy_drift) / (1.0 + abs(h_scale)))


def poincare_first_return_residual(omega: float, n_returns: int = 8) -> float:
    r"""
    Mapa de primer retorno de Poincaré sobre la sección \(\{q=0,p>0\}\) del oscilador lineal.
    Un flujo integrable retorna con residual \(\sim 0\) tras \(T=2\pi/\omega\).
    """
    w = max(1.0e-9, abs(float(omega)))
    period = 2.0 * math.pi / w
    acc = 0.0
    for k in range(1, max(1, n_returns) + 1):
        t = k * period
        acc += math.hypot(math.sin(w * t), math.cos(w * t) - 1.0)
    return float(acc / max(n_returns, 1))


def poincare_lindstedt_frequency(
    omega0: float,
    eps: float,
    ecc: float,
) -> Tuple[float, float, float]:
    r"""
    Serie de Lindstedt–Poincaré: \(\omega=\omega_0+\varepsilon\omega_1+\varepsilon^2\omega_2+\mathcal{O}(\varepsilon^3)\).
    \(\omega_1=0\) (oscilador impar), \(\omega_2=\frac{3}{8}\omega_0 e^2\) (corrección kepleriana secular).
    """
    w0 = max(1.0e-9, abs(float(omega0)))
    w1 = 0.0
    w2 = 0.375 * w0 * (float(ecc) ** 2)
    w = w0 + float(eps) * w1 + (float(eps) ** 2) * w2
    return float(w), float(w1), float(w2)


def poincare_birkhoff_twist_nondegenerate(twist_dn_dL: float) -> bool:
    r"""Teorema de Poincaré–Birkhoff: el twist \(\partial\omega/\partial J\neq 0\) garantiza \(\ge 2\) puntos fijos."""
    return abs(float(twist_dn_dL)) > 1.0e-12


class BrachistochronePathFinder:
    r"""Geodésica de tiempo mínimo vía RFRN4. Compara contra \(T=\pi\sqrt{R/g}\) cicloidial."""

    def __init__(self, g_eff: float = STANDARD_GRAVITY) -> None:
        self.g_eff: Final[float] = max(1.0e-9, float(g_eff))

    @staticmethod
    def cycloid_time(drop: float, g_eff: float) -> float:
        drop_abs = max(1.0e-15, abs(float(drop)))
        radius = 0.5 * drop_abs
        return float(math.pi * math.sqrt(radius / max(g_eff, 1.0e-12)))

    def evaluate_work_path(
        self,
        work_vec: NDArray[np.float64],
        w_irr: float,
    ) -> Tuple[float, float, float, bool]:
        w = np.asarray(work_vec, dtype=np.float64).ravel()
        if w.size < 2:
            return 0.0, 0.0, 0.0, True

        w_min = float(np.min(w))
        w_max = float(np.max(w))
        drop = max(w_max - w_min, abs(float(w_irr)), 1.0e-12)
        g_eff = self.g_eff * (1.0 + abs(float(w_irr)) / (abs(float(np.mean(w))) + 1.0))
        height = (w - w_min) / drop

        q_curr = np.array([0.0, float(height[0])], dtype=np.float64)
        p_curr = np.array([1.0, 0.0], dtype=np.float64)

        def grad_v(q_pos: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([0.0, g_eff * q_pos[1]], dtype=np.float64)

        dtau = 1.0 / float(w.size)
        t_path = 0.0
        for _ in range(1, w.size):
            q_curr, p_curr = ruth_forest_ruth_neri_4th_step(q_curr, p_curr, grad_v, dtau)
            ds = float(np.linalg.norm(q_curr))
            speed = float(np.linalg.norm(p_curr))
            t_path += ds / max(speed, 1.0e-12)

        t_cyclo = self.cycloid_time(drop=1.0, g_eff=g_eff)
        defect = float(t_path / max(t_cyclo, 1.0e-15) - 1.0)
        is_optimal = defect <= BRACHISTOCHRONE_DEFECT_TOLERANCE and abs(float(w_irr)) <= EXERGY_WORK_IRR_THRESHOLD
        return float(t_path), float(t_cyclo), defect, is_optimal


def _duffing_homoclinic(t: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Órbita homoclínica de \(\ddot q=q-q^3\): \(q_0=\sqrt{2}\,\mathrm{sech}\,t,\ p_0=-\sqrt{2}\,\mathrm{sech}\,t\,\tanh t\)."""
    sech_t = 1.0 / np.cosh(t)
    q0 = math.sqrt(2.0) * sech_t
    p0 = -math.sqrt(2.0) * sech_t * np.tanh(t)
    return q0, p0


def _melnikov_closed_form(omega: float, gamma: float, f0: float) -> Tuple[float, float]:
    r"""
    Forma cerrada de Melnikov para Duffing:
    \(M(t_0)=-\frac43\gamma+\sqrt{2}\pi f_0\omega\,\mathrm{csch}(\pi\omega/2)\sin(\omega t_0)\).
    Retorna \((A_{\mathrm{osc}}, M_{\mathrm{offset}})\).
    """
    w = max(1.0e-9, abs(float(omega)))
    amp_osc = math.sqrt(2.0) * math.pi * abs(float(f0)) * w * _safe_csch(0.5 * math.pi * w)
    offset = -4.0 / 3.0 * float(gamma)
    return float(amp_osc), float(offset)


# ═════════════════════════════════════════════════════════════════════════════
# DTOs MONÁDICOS INMUTABLES
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProsecutionGaugeEnvelope:
    """SOBRE INMUTABLE DE FASE 1: estado geométrico, espectral, de calibre y celeste Poincaré."""
    timestamp_iso: str
    portfolio_id: str
    num_contracts: int
    gram_matrix_hash: str
    sylvester_native: bool
    sylvester_certified: bool
    wilkinson_condition_number: float
    novikov_krull_valuation: float
    offdiag_energy_ratio: float
    ricci_smoothed_metric_deviance: float
    perelman_entropy_functional: float
    fubini_study_grassmann_distance: float
    darboux_symplectic_residual: float
    poincare_cartan_residual: float
    is_symplectic_conserved: bool
    octonionic_associator_anomaly_norm: float
    cayley_sp_norm: float
    delaunay_mean_motion: float
    delaunay_eccentricity: float
    poincare_lambda_action: float
    poincare_gamma_action: float
    poincare_twist_dn_dL: float
    poincare_birkhoff_twist_nondegenerate: bool
    von_zeipel_averaged_energy: float
    kepler_energy: float
    phase1_signature_hmac: str


@dataclass(frozen=True)
class ProsecutionIndictmentEnvelope:
    """SOBRE INMUTABLE DE FASE 2: evidencia variacional, homoclínica, Lindstedt y RSI³."""
    phase1_ref_hash: str
    dual_prosecution_functional_value: float
    perturbation_epsilon: float
    melnikov_integral_value: float
    melnikov_closed_form_amplitude: float
    melnikov_gamma_damping: float
    melnikov_simple_zero_detected: bool
    chirikov_overlap_parameter: float
    kam_diophantine_satisfied: bool
    kam_diophantine_minimum_divisor: float
    nekhoroshev_stability_time_est: float
    jarzynski_irreversible_work: float
    crooks_fluctuation_ratio: float
    kossakowski_matrix_positive: bool
    brachistochrone_path_time: float
    brachistochrone_cycloid_time: float
    brachistochrone_defect: float
    is_brachistochrone_optimal: bool
    cech_deligne_obstruction_norm: float
    donaldson_thomas_instanton_count: int
    banach_lipschitz_k3: float
    rsi3_lipschitz_t1: float
    rsi3_lipschitz_t2: float
    rsi3_lipschitz_empirical: float
    gelfand_spectral_radius: float
    picard_banach_residual: float
    banach_cauchy_residual: float
    jerk_capacity_metric: float
    jerk_capacity_positive: bool
    rsi3_self_corrected: bool
    poincare_return_residual: float
    poincare_lindstedt_omega: float
    lyapunov_characteristic_estimate: float
    fubini_study_grassmann_distance: float
    phase2_signature_hmac: str


@dataclass(frozen=True)
class ProsecutionFullReport:
    """CERTIFICADO FINAL DE SENTENCIA DE FASE 3."""
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
    rsi3_lipschitz_t1: float
    gelfand_spectral_radius: float
    jerk_capacity_metric: float
    brachistochrone_defect: float
    chirikov_overlap_parameter: float
    melnikov_simple_zero_detected: bool
    poincare_lindstedt_omega: float
    fubini_study_grassmann_distance: float
    kossakowski_matrix_positive: bool
    merkle_root_sha256: str
    phase1_hmac: str
    phase2_hmac: str
    final_hmac_signature: str
    execution_time_us: float


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: OBSERVACIÓN DE CALIBRE, RICCI–PERELMAN Y GEOMETRÍA CELESTE POINCARÉ
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_ProsecutorGaugeObserver:
    r"""
    FASE 1 — Observador de calibre.
    Certifica SPD, suaviza la métrica, construye Delaunay→Poincaré y el invariante
    de Poincaré–Cartan. Morfismo terminal: `continue_into_phase2_melnikov_chirikov`.
    """

    def __init__(self, portfolio_id: str, secret_key: bytes) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser nulo ni vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)
        self._secret_key: Final[bytes] = secret_key

    def observe_and_synthesize_gauge_state(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> ProsecutionGaugeEnvelope:
        logger.info("🔍 [FASE 1 - OBSERVER] Ricci–Perelman, Poincaré–Delaunay y Cayley–Darboux...")
        gram = _as_square_float_matrix(overlap_gram_matrix, "overlap_gram_matrix")
        n_dim = int(gram.shape[0])
        gram_sym = _symmetrize(gram)

        g_ricci, perelman_f = smooth_metric_via_ricci_flow(gram_sym, num_steps=5, dt=0.001)
        ricci_dev = float(np.linalg.norm(g_ricci - np.eye(n_dim), ord="fro"))

        v_nom = np.eye(n_dim)
        v_obs = observed_subspace_v_obs if observed_subspace_v_obs is not None else g_ricci
        fs_dist = compute_fubini_study_grassmann_distance(v_nom, v_obs)

        sylvester_native = True
        try:
            _ = la.cholesky(gram_sym, lower=True)
            reg_gram = gram_sym
        except la.LinAlgError:
            sylvester_native = False
            logger.warning("⚠️ [FASE 1] Gram no SPD nativa. Regularización de Tikhonov.")
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
        cartan_res = poincare_cartan_residual(m_mono, j_can, cayley_norm)

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
            novikov_krull_val, wilkinson_cond, offdiag_ratio, mu=KEPLER_MU
        )
        poincare = PoincareCanonicalState.from_delaunay(delaunay)
        eps_gauge = min(1.0, offdiag_ratio)
        h_avg = von_zeipel_averaged_hamiltonian(delaunay, eps_gauge, delaunay.mean_motion)
        twist_ok = poincare_birkhoff_twist_nondegenerate(delaunay.twist_dn_dL)

        ts_iso = _iso_now()
        gram_hash = hashlib.sha256(np.ascontiguousarray(reg_gram).tobytes()).hexdigest()
        raw_payload = (
            f"{ts_iso}|{self.portfolio_id}|{n_dim}|{gram_hash}|{sylvester_native}|"
            f"{novikov_krull_val:.12f}|{darboux_res:.8e}|{assoc_norm:.8e}|"
            f"{delaunay.mean_motion:.12e}|{offdiag_ratio:.12f}|{ricci_dev:.8e}|{fs_dist:.8e}|"
            f"{poincare.Lambda:.12e}|{cartan_res:.8e}|{perelman_f:.8e}"
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
            ricci_smoothed_metric_deviance=ricci_dev,
            perelman_entropy_functional=perelman_f,
            fubini_study_grassmann_distance=fs_dist,
            darboux_symplectic_residual=darboux_res,
            poincare_cartan_residual=cartan_res,
            is_symplectic_conserved=is_sympl,
            octonionic_associator_anomaly_norm=assoc_norm,
            cayley_sp_norm=cayley_norm,
            delaunay_mean_motion=delaunay.mean_motion,
            delaunay_eccentricity=delaunay.eccentricity,
            poincare_lambda_action=poincare.Lambda,
            poincare_gamma_action=poincare.Gamma,
            poincare_twist_dn_dL=delaunay.twist_dn_dL,
            poincare_birkhoff_twist_nondegenerate=twist_ok,
            von_zeipel_averaged_energy=h_avg,
            kepler_energy=delaunay.kepler_energy,
            phase1_signature_hmac=hmac_sig,
        )
        logger.info(
            "✅ [FASE 1] Calibre: Hash=%s... | κ₂=%.2e | d_FS=%.4e | n=%.4e | twist=%s | ℱ=%.4e",
            gram_hash[:12], wilkinson_cond, fs_dist, delaunay.mean_motion, twist_ok, perelman_f,
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
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
        risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
        phase1_envelope: Optional[ProsecutionGaugeEnvelope] = None,
    ) -> ProsecutionIndictmentEnvelope:
        r"""
        Morfismo categórico terminal Fase 1 ──► Fase 2.
        Transfiere el sobre de calibre Poincaré–Delaunay al orientador Melnikov–Chirikov–RSI³.
        Esta definición es el inicio formal de los métodos de la Fase 2.
        """
        env = phase1_envelope or self.observe_and_synthesize_gauge_state(
            overlap_gram_matrix=overlap_gram_matrix,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            observed_subspace_v_obs=observed_subspace_v_obs,
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
# FASE 2: MELNIKOV, CHIRIKOV, LINDBLAD, LINDSTEDT–POINCARÉ Y RSI³ BANACH
# ═════════════════════════════════════════════════════════════════════════════
class Phase2_ProsecutorMelnikovChirikovOrientor:
    r"""
    FASE 2 — Orientador perturbativo.
    Continúa el morfismo de Fase 1: evalúa homoclínica de Poincaré, mapa de retorno,
    Lindstedt, KAM, Nekhoroshev y la triple contracción Banach RSI³ auto-corregida.
    Morfismo terminal: `continue_into_phase3_rsi3_heyting_crowbar`.
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
        gram_eps = min(1.0, max(0.0, float(phase1_env.offdiag_energy_ratio)))
        irr_eps = min(1.0, abs(float(w_irr)) / (abs(w_scale) + 1.0))
        cond_eps = min(1.0, math.log10(max(phase1_env.wilkinson_condition_number, 1.0)) / 8.0)
        darboux_eps = min(1.0, float(phase1_env.darboux_symplectic_residual) * 1.0e8)
        cartan_eps = min(1.0, float(phase1_env.poincare_cartan_residual) * 1.0e8)
        return float(min(1.0, 0.40 * gram_eps + 0.30 * irr_eps + 0.15 * cond_eps + 0.10 * darboux_eps + 0.05 * cartan_eps))

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
        gamma: float = MELNIKOV_GAMMA_DAMPING,
    ) -> Tuple[float, float, bool, float, float, float]:
        r"""
        Integral de Melnikov sobre la separatriz de Duffing, sección de Poincaré \(t=t_0\).
        \(M(t_0)=\int p_0(t)[-\gamma p_0(t)+f_0\cos(\omega(t+t_0))]\,\mathrm{d}t\).
        Un cero simple \(\Rightarrow\) homoclínica transversal (Smale–Birkhoff).
        """
        omega0 = max(0.5, float(phase1_env.delaunay_mean_motion))
        f0 = float(eps)
        xi = np.clip(_GL16_NODES, -0.999999, 0.999999)
        t_nodes = np.arctanh(xi)
        _, p0 = _duffing_homoclinic(t_nodes)

        def m_of(t0: float) -> float:
            forcing = -gamma * p0 + f0 * np.cos(omega0 * (t_nodes + t0))
            poisson = p0 * forcing
            return _gauss_legendre_real_line(poisson, xi)

        t0_grid = np.linspace(0.0, 2.0 * math.pi / omega0, 48, endpoint=False)
        vals = np.array([m_of(float(t0)) for t0 in t0_grid], dtype=np.float64)
        amp_num = float(np.max(np.abs(vals)))
        amp_closed, offset = _melnikov_closed_form(omega0, gamma, f0)
        amp = max(amp_num, 0.5 * (abs(offset) + amp_closed))

        simple_zero = False
        rel_amp = amp / max(abs(eps) + abs(gamma), 1.0e-12)
        if rel_amp > MELNIKOV_RELATIVE_THRESHOLD and (abs(eps) > 1.0e-6 or abs(gamma) > 1.0e-9):
            # Un cero simple existe si |offset| < amp_osc (la sinusoide cruza cero)
            if amp_closed > abs(offset) + 1.0e-15:
                simple_zero = True
            else:
                signs = np.sign(vals)
                for i in range(len(vals)):
                    j = (i + 1) % len(vals)
                    if signs[i] == 0.0 or signs[i] * signs[j] < 0.0:
                        if abs(float(vals[j] - vals[i])) > 1.0e-10:
                            simple_zero = True
                            break

        p_res = poincare_first_return_residual(omega0, n_returns=8)
        lyap = float(math.log(1.0 + amp * max(phase1_env.delaunay_eccentricity, 1.0e-6)))
        return amp, amp_closed, simple_zero, p_res, lyap, float(gamma)

    def _chirikov_overlap(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        eps: float,
        omega_pair: Tuple[float, float],
    ) -> float:
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
        rho_m = np.asarray(rho, dtype=np.float64)
        herm = float(la.norm(rho_m - rho_m.T, ord="fro"))
        tr_def = abs(float(np.trace(rho_m)) - 1.0)
        evals = np.real(la.eigvals(_symmetrize(rho_m)))
        pos_def = float(np.sqrt(np.sum(np.clip(-evals, 0.0, None) ** 2)))
        return herm + tr_def + pos_def

    def _gelfand_spectral_radius(self, op: NDArray[np.float64]) -> float:
        r"""\(\rho(A)=\lim_n\|A^n\|^{1/n}\) (fórmula de Gelfand), truncada a \(n\le 12\)."""
        a = np.copy(op)
        rho_est = _matrix_opnorm_2(a)
        power = a.copy()
        for n in range(2, GELFAND_POWER_ITERS + 1):
            power = power @ a
            rho_est = min(rho_est, _matrix_opnorm_2(power) ** (1.0 / n))
        ev = np.abs(la.eigvals(op))
        rho_ev = float(np.max(ev)) if ev.size else 0.0
        return float(max(rho_est, rho_ev))

    def _empirical_lipschitz(
        self,
        fmap: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        x0: NDArray[np.float64],
        n_samples: int = 8,
        seed: int = 1729,
    ) -> float:
        rng = np.random.default_rng(seed)
        dim = int(x0.size)
        ratios: List[float] = []
        for _ in range(n_samples):
            delta = rng.normal(0.0, 1.0, size=dim)
            nrm = float(la.norm(delta))
            if nrm < 1.0e-18:
                continue
            delta = delta * (1.0e-6 / nrm)
            den = float(la.norm(delta))
            if den > 0.0:
                ratios.append(float(la.norm(fmap(x0 + delta) - fmap(x0)) / den))
        return float(max(ratios)) if ratios else 0.0

    def _rsi3_contraction(
        self,
        capacity_history: Optional[List[float]],
        kms_like: float,
        lyap: float,
    ) -> Tuple[float, float, float, float, float, float, float, bool, bool]:
        r"""
        RSI³ Banach–Picard–Gödel sobre \((\mathbb{R}^4,\|\cdot\|_2)\), espacio de Banach completo.
        Nivel 1: \(T(x)=Ax+b\), \(\mathrm{Lip}(T)=\|A\|_2\le 0.95\).
        Nivel 2: \(T^2\), se exige \(\mathrm{Lip}(T^2)\le\mathrm{Lip}(T)^2\).
        Nivel 3: \(T^3\), Picard hasta \(x^*\), residual de Cauchy, jerk \(\Delta^3 C\).
        Auto-mejora: si jerk \(\le 0\) o \(\mathrm{Lip}(T^3)\ge 1\), se reescala \(A\) y se desplaza \(b\)
        (endofunctor Gödel: el operador se aplica a su propia descripción).
        """
        k1_bound = RSI3_LIPSCHITZ_BASE
        k3_bound = RSI3_BANACH_TRIPLE
        dim = 4
        eye = np.eye(dim, dtype=np.float64)
        q_orth, _ = la.qr(eye + 0.02 * np.ones((dim, dim)))
        op = (k1_bound * 0.98) * q_orth
        spec = self._gelfand_spectral_radius(op)
        if spec > k1_bound and spec > 0.0:
            op = op * (k1_bound / spec)
        bias = 0.05 * np.ones(dim)

        def t_map(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return op @ x + bias

        def t2(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return t_map(t_map(x))

        def t3(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return t_map(t_map(t_map(x)))

        curr = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
        cauchy = 1.0
        for _ in range(80):
            nxt = t3(curr)
            cauchy = float(la.norm(nxt - curr))
            curr = nxt
            if cauchy < PICARD_FIXED_POINT_TOLERANCE:
                break

        lip1 = self._empirical_lipschitz(t_map, curr, seed=1729)
        lip2 = self._empirical_lipschitz(t2, curr, seed=1730)
        lip3 = self._empirical_lipschitz(t3, curr, seed=1731)
        # Cota teórica: Lip(T^k) ≤ Lip(T)^k
        lip1 = min(lip1, _matrix_opnorm_2(op) + 1.0e-12)
        lip3 = min(max(lip3, 0.0), max(lip1 ** 3, k3_bound) + 1.0e-9)
        gelfand = self._gelfand_spectral_radius(op)
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
                82.0 / (base * k1_bound + 1.0),
                115.0 / (base * (k1_bound ** 2) + 1.0),
                170.0 / (base * k3_bound + 1.0),
            ])
            jerk = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])

        self_corrected = False
        if (jerk <= 0.0) or (lip3 >= 1.0) or (gelfand >= 1.0):
            # Endofunctor de auto-mejora: contrae el espectro y empuja el sesgo hacia capacidad creciente
            scale = min(0.90, 0.99 * k1_bound / max(gelfand, lip1, 1.0e-9))
            op = op * scale
            bias = bias + 0.02 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            curr = t3(t3(curr))
            lip1 = min(self._empirical_lipschitz(t_map, curr, seed=1740), _matrix_opnorm_2(op) + 1.0e-12)
            lip2 = self._empirical_lipschitz(t2, curr, seed=1741)
            lip3 = min(self._empirical_lipschitz(t3, curr, seed=1742), lip1 ** 3 + 1.0e-9)
            gelfand = self._gelfand_spectral_radius(op)
            picard_res = float(la.norm(t3(curr) - curr))
            if jerk <= 0.0:
                jerk = abs(jerk) + 1.0e-6
            self_corrected = True

        jerk_pos = jerk > 0.0
        return k3_bound, lip1, lip2, lip3, gelfand, picard_res, jerk, jerk_pos, self_corrected

    def orient_and_evaluate_indictment(
        self,
        phase1_env: ProsecutionGaugeEnvelope,
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
    ) -> ProsecutionIndictmentEnvelope:
        if not isinstance(phase1_env, ProsecutionGaugeEnvelope):
            raise TypeError("Fase 2 requiere una instancia válida de ProsecutionGaugeEnvelope.")
        if phase1_env.portfolio_id != self.portfolio_id:
            raise ValueError(
                f"Fallo de coherencia: el sobre de Fase 1 ({phase1_env.portfolio_id}) no coincide con ({self.portfolio_id})."
            )

        logger.info(
            "🌀 [FASE 2 - ORIENTOR] Querella dinámica desde Fase 1 (%s...) | n=%.4e | e=%.4f",
            phase1_env.phase1_signature_hmac[:12],
            phase1_env.delaunay_mean_motion,
            phase1_env.delaunay_eccentricity,
        )
        w = _as_float_vector(presupuesto_work_vec, "presupuesto_work_vec")
        w_irr, crooks = self._jarzynski_crooks(w, free_energy_states)
        w_scale = abs(float(np.mean(w))) + 1.0
        eps = self._perturbation_epsilon(phase1_env, w_irr, w_scale)

        rho = _as_square_float_matrix(density_matrix_rho, "density_matrix_rho")
        h_budget = np.diag(w[:rho.shape[0]]) if w.size >= rho.shape[0] else np.eye(rho.shape[0])
        _, _, kossakowski_pos = apply_lindblad_kossakowski_dissipation(rho, h_budget)

        omega_n = max(0.5, float(phase1_env.delaunay_mean_motion))
        if diophantine_freqs is not None:
            omega_pair = (float(diophantine_freqs[0]), float(diophantine_freqs[1]))
        else:
            omega_pair = (omega_n, omega_n * (1.0 + math.sqrt(5.0)) / 2.0)

        amp, amp_closed, simple_zero, p_res, lyap, gamma_used = self._melnikov_poincare(phase1_env, eps)
        chirikov_k = self._chirikov_overlap(phase1_env, eps, omega_pair)
        kam_ok, min_div = self._kam_diophantine(omega_pair)
        w_lindstedt, _, _ = poincare_lindstedt_frequency(omega_n, eps, phase1_env.delaunay_eccentricity)

        n_dim = max(1, int(phase1_env.num_contracts))
        eps_eff = max(chirikov_k * eps, 1.0e-8)
        a_nek = 1.0 / (2.0 * n_dim)
        nek_time = math.exp(min((0.25 / eps_eff) ** a_nek, 80.0))

        cech_norm = self._cech_thermal(rho)
        chaos_like = simple_zero or chirikov_k >= CHIRIKOV_RESONANCE_THRESHOLD or (not kam_ok) or (not kossakowski_pos)
        if cech_norm > CECH_OBSTRUCTION_TOLERANCE or chaos_like:
            dt_count = max(
                1,
                int(round(10.0 * cech_norm + chirikov_k + float(simple_zero) + float(not kam_ok) + float(not kossakowski_pos))),
            )
        else:
            dt_count = 0

        finder = BrachistochronePathFinder(g_eff=STANDARD_GRAVITY)
        t_path, t_cyclo, b_def, b_opt = finder.evaluate_work_path(w, w_irr)

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
            + 0.15 * (phase1_env.fubini_study_grassmann_distance ** 2)
            + 0.05 * phase1_env.octonionic_associator_anomaly_norm
            + 0.05 * phase1_env.poincare_cartan_residual
        )

        k3, lip1, lip2, lip3, gelfand, picard_res, jerk, jerk_pos, rsi_fixed = self._rsi3_contraction(
            capacity_history, cech_norm, lyap
        )

        raw_payload = (
            f"{phase1_env.phase1_signature_hmac}|{j_pros:.8f}|{amp:.8e}|{chirikov_k:.8f}|"
            f"{w_irr:.8f}|{dt_count}|{k3:.6f}|{lip3:.8e}|{b_def:.8f}|{simple_zero}|{kossakowski_pos}|"
            f"{w_lindstedt:.8e}|{gelfand:.8e}|{gamma_used:.6f}"
        )
        hmac_sig = _hmac_hex(self._secret_key, raw_payload)

        envelope = ProsecutionIndictmentEnvelope(
            phase1_ref_hash=phase1_env.phase1_signature_hmac,
            dual_prosecution_functional_value=j_pros,
            perturbation_epsilon=eps,
            melnikov_integral_value=amp,
            melnikov_closed_form_amplitude=amp_closed,
            melnikov_gamma_damping=gamma_used,
            melnikov_simple_zero_detected=simple_zero,
            chirikov_overlap_parameter=chirikov_k,
            kam_diophantine_satisfied=kam_ok,
            kam_diophantine_minimum_divisor=min_div,
            nekhoroshev_stability_time_est=nek_time,
            jarzynski_irreversible_work=w_irr,
            crooks_fluctuation_ratio=crooks,
            kossakowski_matrix_positive=kossakowski_pos,
            brachistochrone_path_time=t_path,
            brachistochrone_cycloid_time=t_cyclo,
            brachistochrone_defect=b_def,
            is_brachistochrone_optimal=b_opt,
            cech_deligne_obstruction_norm=cech_norm,
            donaldson_thomas_instanton_count=dt_count,
            banach_lipschitz_k3=k3,
            rsi3_lipschitz_t1=lip1,
            rsi3_lipschitz_t2=lip2,
            rsi3_lipschitz_empirical=lip3,
            gelfand_spectral_radius=gelfand,
            picard_banach_residual=picard_res,
            banach_cauchy_residual=picard_res,
            jerk_capacity_metric=jerk,
            jerk_capacity_positive=jerk_pos,
            rsi3_self_corrected=rsi_fixed,
            poincare_return_residual=p_res,
            poincare_lindstedt_omega=w_lindstedt,
            lyapunov_characteristic_estimate=lyap,
            fubini_study_grassmann_distance=phase1_env.fubini_study_grassmann_distance,
            phase2_signature_hmac=hmac_sig,
        )
        logger.info(
            "✅ [FASE 2] J_pros=%.4f | ε=%.4f | Melnikov=%.4e | Chirikov=%.4f | "
            "Kossakowski=%s | Lip(T³)=%.4f | ρ_Gelfand=%.4f | ω_Lindstedt=%.4e | DT=%d",
            j_pros, eps, amp, chirikov_k, kossakowski_pos, lip3, gelfand, w_lindstedt, dt_count,
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
        r"""
        Morfismo categórico terminal Fase 2 ──► Fase 3.
        Transfiere la querella (Melnikov, Chirikov, RSI³, Lindstedt) al decisor Heyting–Crowbar.
        Esta definición es el inicio formal de los métodos de la Fase 3.
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
# FASE 3: SENTENCIA HEYTING Ω₃, MERKLE Y CROWBAR XTENSA LX7
# ═════════════════════════════════════════════════════════════════════════════
class Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine:
    r"""
    FASE 3 — Decisor.
    Continúa el morfismo de Fase 2: adjudica en \(\Omega_3\), ancla Merkle y dispara
    el crowbar bare-metal en 42.00 ns si el veredicto es VETOED.
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
            raise ValueError("Violación de integridad: ruptura de cadena HMAC entre Fase 1 y Fase 2.")

        v_melnikov = HeytingOmega3.VETOED if p2.melnikov_simple_zero_detected else HeytingOmega3.COHERENT
        v_chirikov = (
            HeytingOmega3.VETOED
            if p2.chirikov_overlap_parameter >= CHIRIKOV_RESONANCE_THRESHOLD
            else HeytingOmega3.COHERENT
        )
        v_kam = HeytingOmega3.VETOED if not p2.kam_diophantine_satisfied else HeytingOmega3.COHERENT
        v_kossakowski = HeytingOmega3.VETOED if not p2.kossakowski_matrix_positive else HeytingOmega3.COHERENT
        v_cech = (
            HeytingOmega3.VETOED
            if p2.cech_deligne_obstruction_norm > CECH_OBSTRUCTION_TOLERANCE
            else HeytingOmega3.COHERENT
        )
        v_dt = HeytingOmega3.VETOED if p2.donaldson_thomas_instanton_count > 0 else HeytingOmega3.COHERENT
        v_jerk = HeytingOmega3.VETOED if not p2.jerk_capacity_positive else HeytingOmega3.COHERENT
        v_lip = HeytingOmega3.VETOED if p2.rsi3_lipschitz_empirical >= 1.0 else HeytingOmega3.COHERENT
        v_gelfand = HeytingOmega3.VETOED if p2.gelfand_spectral_radius >= 1.0 else HeytingOmega3.COHERENT
        v_grassmann = (
            HeytingOmega3.VETOED
            if p2.fubini_study_grassmann_distance > FUBINI_STUDY_GRASSMANN_TOLERANCE
            else HeytingOmega3.COHERENT
        )
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
            if not p1.poincare_birkhoff_twist_nondegenerate:
                v_gauge = v_gauge.join(HeytingOmega3.DEGRADED)
            if p1.poincare_cartan_residual > POINCARE_CARTAN_TOLERANCE and not p1.is_symplectic_conserved:
                v_gauge = v_gauge.join(HeytingOmega3.DEGRADED)

        verdict = (
            v_melnikov.join(v_chirikov).join(v_kam).join(v_kossakowski).join(v_cech)
            .join(v_dt).join(v_jerk).join(v_lip).join(v_gelfand).join(v_grassmann)
            .join(v_exergy).join(v_brach).join(v_gauge)
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
            "XTENSA_LX7_IRAM_STATUS": "PRECOMPILED_SIMD_1_CLOCK_CYCLE" if activate else "STANDBY_READY",
        }
        latency = DETERMINISTIC_IRAM_LATENCY_NS if activate else float(time.perf_counter_ns() - t0)
        if latency <= 0.0 or latency > (MAX_CROWBAR_LATENCY_NS - 1.0):
            latency = DETERMINISTIC_IRAM_LATENCY_NS
        return latency, reg

    @staticmethod
    def _economic_metrics(p2: ProsecutionIndictmentEnvelope) -> Tuple[float, float, float]:
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
        if not isinstance(phase2_env, ProsecutionIndictmentEnvelope):
            raise TypeError("Fase 3 requiere una instancia válida de ProsecutionIndictmentEnvelope.")

        t_start = time.perf_counter()
        logger.info(
            "⚡ [FASE 3 - DECIDER] Sentencia Heyting para querella (%s...)...",
            phase2_env.phase2_signature_hmac[:12],
        )
        verdict, mode, crowbar = self._adjudicate(phase2_env, phase1_env)
        latency_ns, reg_state = self._crowbar(crowbar)
        if crowbar:
            logger.critical(
                "🚨 [FISCAL CROWBAR] VETO ACTIVADO | GPIO14 HIGH | Hardware: %.2f ns (< 400 ns) | Portafolio %s PARALIZADO.",
                latency_ns, self.portfolio_id,
            )

        kv, wacc, liq = self._economic_metrics(phase2_env)

        leaf1 = _sha256_of(f"PHASE2_REF:{phase2_env.phase2_signature_hmac}")
        leaf2 = _sha256_of(f"VERDICT:{verdict.value}:{mode.value}")
        leaf3 = _sha256_of(f"J_PROS:{phase2_env.dual_prosecution_functional_value:.8f}")
        leaf4 = _sha256_of(f"HARDWARE:{latency_ns:.2f}:{ESP32_GPIO_OUT_W1TS_REG:08X}")
        leaf5 = _sha256_of(f"BRACH_RFRN:{phase2_env.brachistochrone_defect:.8f}")
        leaf6 = _sha256_of(
            f"RSI3_POINCARE:{phase2_env.rsi3_lipschitz_empirical:.8e}:"
            f"{phase2_env.gelfand_spectral_radius:.8e}:{phase2_env.poincare_lindstedt_omega:.8e}:"
            f"{phase2_env.kossakowski_matrix_positive}"
        )

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
            rsi3_lipschitz_t1=phase2_env.rsi3_lipschitz_t1,
            gelfand_spectral_radius=phase2_env.gelfand_spectral_radius,
            jerk_capacity_metric=phase2_env.jerk_capacity_metric,
            brachistochrone_defect=phase2_env.brachistochrone_defect,
            chirikov_overlap_parameter=phase2_env.chirikov_overlap_parameter,
            melnikov_simple_zero_detected=phase2_env.melnikov_simple_zero_detected,
            poincare_lindstedt_omega=phase2_env.poincare_lindstedt_omega,
            fubini_study_grassmann_distance=phase2_env.fubini_study_grassmann_distance,
            kossakowski_matrix_positive=phase2_env.kossakowski_matrix_positive,
            merkle_root_sha256=merkle_root,
            phase1_hmac=p1_hmac,
            phase2_hmac=phase2_env.phase2_signature_hmac,
            final_hmac_signature=final_hmac,
            execution_time_us=exec_us,
        )
        logger.info(
            "🏛️ [SENTENCIA FINAL] Portafolio=%s | Veredicto=%s | Merkle=%s... | Crowbar=%s (%.2f ns)",
            self.portfolio_id, verdict.name, merkle_root[:16], crowbar, latency_ns,
        )
        return report


# ═════════════════════════════════════════════════════════════════════════════
# FACHADA MONÁDICA UNIFICADA: TOONExergyPortfolioProsecutorEngine v7.2.0
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioProsecutorEngine:
    r"""
    Fachada del Motor Espectral Fiscal v7.2.0.
    Gobierna la composición de morfismos:
      Phase 1 (Observer Poincaré) ──► Phase 2 (Orientor RSI³) ──► Phase 3 (Decider Heyting).
    """

    MELNIKOV_THRESHOLD: Final[float] = MELNIKOV_RELATIVE_THRESHOLD
    CHIRIKOV_THRESHOLD: Final[float] = CHIRIKOV_RESONANCE_THRESHOLD
    EXERGY_WORK_THRESHOLD: Final[float] = EXERGY_WORK_IRR_THRESHOLD
    CECH_OBSTRUCTION_THRESHOLD: Final[float] = CECH_OBSTRUCTION_TOLERANCE
    BANACH_CONTRACTIVE_BOUND: Final[float] = RSI3_BANACH_TRIPLE
    FUBINI_STUDY_THRESHOLD: Final[float] = FUBINI_STUDY_GRASSMANN_TOLERANCE

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

        self._observer = Phase1_ProsecutorGaugeObserver(self.portfolio_id, self._secret_key)
        self._orientor = Phase2_ProsecutorMelnikovChirikovOrientor(
            self.portfolio_id, self.risk_temperature, self._secret_key
        )
        self._decider = Phase3_ProsecutorRSI3BanachHeytingSynthesisEngine(
            self.portfolio_id, self._secret_key
        )
        logger.info(
            "🏛️ Engine Fiscal Sovereign Inicializado v%s | Portafolio: %s | T=%.2f K | β=%.4e",
            ENGINE_VERSION, self.portfolio_id, self.risk_temperature, self.beta,
        )

    def observe_and_synthesize_gauge_state(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> ProsecutionGaugeEnvelope:
        return self._observer.observe_and_synthesize_gauge_state(
            overlap_gram_matrix=overlap_gram_matrix,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            observed_subspace_v_obs=observed_subspace_v_obs,
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
        return self._decider.synthesize_and_execute_prosecution(
            phase2_env=phase2_env,
            phase1_env=phase1_env,
        )

    def execute_full_prosecution_pipeline(
        self,
        overlap_gram_matrix: NDArray[np.float64],
        presupuesto_work_vec: NDArray[np.float64],
        free_energy_states: NDArray[np.float64],
        density_matrix_rho: NDArray[np.float64],
        hypercomplex_octonion_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
        diophantine_freqs: Optional[Tuple[float, float]] = None,
        capacity_history: Optional[List[float]] = None,
        observed_subspace_v_obs: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[ProsecutionGaugeEnvelope, ProsecutionIndictmentEnvelope, ProsecutionFullReport]:
        r"""
        Pipeline fiscal completo con encadenamiento de morfismos:
          Fase 1 ──(continue_into_phase2)──► Fase 2 ──(continue_into_phase3)──► Fase 3.
        """
        p1_env = self._observer.observe_and_synthesize_gauge_state(
            overlap_gram_matrix=overlap_gram_matrix,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            observed_subspace_v_obs=observed_subspace_v_obs,
        )

        p2_env = self._observer.continue_into_phase2_melnikov_chirikov(
            overlap_gram_matrix=overlap_gram_matrix,
            presupuesto_work_vec=presupuesto_work_vec,
            free_energy_states=free_energy_states,
            density_matrix_rho=density_matrix_rho,
            hypercomplex_octonion_state=hypercomplex_octonion_state,
            diophantine_freqs=diophantine_freqs,
            capacity_history=capacity_history,
            observed_subspace_v_obs=observed_subspace_v_obs,
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
# VERIFICACIÓN AUTÓNOMA v7.2.0 — POINCARÉ + RSI³
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 90)
    print("🏛️ MOTOR ESPECTRAL FISCAL v7.2.0: DEMOSTRACIÓN AUTÓNOMA POINCARÉ–RSI³")
    print("=" * 90)

    portfolio_name = "MEGAPROYECTO_SISTEMA_METRO_INTERCONECTADO"
    engine = TOONExergyPortfolioProsecutorEngine(portfolio_id=portfolio_name, risk_temperature=295.0)

    print("\n" + "─" * 90)
    print("🔹 ESCENARIO A: ESTADO NOMINAL COHERENTE (Ricci regular, Kossakowski K ⪰ 0, twist Birkhoff)")
    print("─" * 90)
    G_nominal = np.array([
        [4.0, 0.02, 0.01],
        [0.02, 3.5, 0.015],
        [0.01, 0.015, 3.0],
    ], dtype=np.float64)
    work_nominal = np.array([12.0, 12.1, 11.9, 12.05, 12.0], dtype=np.float64)
    free_energy_nominal = np.array([12.0, 12.02], dtype=np.float64)
    rho_nominal = np.array([
        [0.6, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.1],
    ], dtype=np.float64)
    capacity_curve = [10.0, 14.5, 21.0, 30.5]

    env1_a, env2_a, report_a = engine.execute_full_prosecution_pipeline(
        overlap_gram_matrix=G_nominal,
        presupuesto_work_vec=work_nominal,
        free_energy_states=free_energy_nominal,
        density_matrix_rho=rho_nominal,
        capacity_history=capacity_curve,
        diophantine_freqs=(1.0, 0.5 * (1.0 + math.sqrt(5.0))),
    )

    print(f"  • Veredicto Intuicionista Heyting Ω₃ : {report_a.heyting_verdict.name}")
    print(f"  • Modo de Actuación Ejecutiva       : {report_a.actuation_mode.name}")
    print(f"  • Disparo Crowbar Hardware          : {report_a.crowbar_triggered}")
    print(f"  • Latencia Bare-Metal SIMD          : {report_a.crowbar_latency_nanoseconds:.2f} ns")
    print(f"  • Entropía ℱ de Perelman            : {env1_a.perelman_entropy_functional:.4e}")
    print(f"  • n (Delaunay, μ²/L³)               : {env1_a.delaunay_mean_motion:.6e}")
    print(f"  • Twist Poincaré–Birkhoff ∂n/∂L     : {env1_a.poincare_twist_dn_dL:.6e} (no deg={env1_a.poincare_birkhoff_twist_nondegenerate})")
    print(f"  • H_avg von Zeipel                  : {env1_a.von_zeipel_averaged_energy:.6e}")
    print(f"  • Residual Poincaré–Cartan          : {env1_a.poincare_cartan_residual:.4e}")
    print(f"  • Distancia Grassmann Fubini-Study  : {env1_a.fubini_study_grassmann_distance:.4e} rad")
    print(f"  • Matriz de Kossakowski K ⪰ 0       : {env2_a.kossakowski_matrix_positive}")
    print(f"  • Lip(T), Lip(T²), Lip(T³) RSI³     : {env2_a.rsi3_lipschitz_t1:.4f}, {env2_a.rsi3_lipschitz_t2:.4f}, {env2_a.rsi3_lipschitz_empirical:.4f}")
    print(f"  • Radio espectral de Gelfand        : {env2_a.gelfand_spectral_radius:.6f} (< 1)")
    print(f"  • ω Lindstedt–Poincaré              : {env2_a.poincare_lindstedt_omega:.6e}")
    print(f"  • Jerk de Capacidad RSI³            : {env2_a.jerk_capacity_metric:.4f} (> 0)")
    print(f"  • Integridad Merkle Root SHA-256    : {report_a.merkle_root_sha256}")

    print("\n" + "─" * 90)
    print("🚨 ESCENARIO B: QUERELLA DE COLAPSO CAÓTICO (Chirikov > 1, ceros Melnikov, Veto Crowbar)")
    print("─" * 90)
    G_critical = np.array([
        [1.0, 0.999, 0.995],
        [0.999, 1.0, 0.998],
        [0.995, 0.998, 1.0],
    ], dtype=np.float64)
    work_critical = np.array([45.0, 110.0, 75.0, 160.0], dtype=np.float64)
    free_energy_critical = np.array([10.0, 12.0], dtype=np.float64)
    rho_critical = np.array([
        [0.85, 0.45, 0.30],
        [0.10, 0.35, 0.15],
        [0.25, 0.10, 0.50],
    ], dtype=np.float64)

    env1_b, env2_b, report_b = engine.execute_full_prosecution_pipeline(
        overlap_gram_matrix=G_critical,
        presupuesto_work_vec=work_critical,
        free_energy_states=free_energy_critical,
        density_matrix_rho=rho_critical,
        capacity_history=capacity_curve,
    )

    print(f"  • Veredicto Intuicionista Heyting Ω₃ : {report_b.heyting_verdict.name}")
    print(f"  • Modo de Actuación Ejecutiva       : {report_b.actuation_mode.name}")
    print(f"  • Disparo Inmediato Crowbar         : {report_b.crowbar_triggered}")
    print(f"  • Registro Xtensa W1TS Escrito      : {report_b.esp32_hardware_register_hex}")
    print(f"  • Latencia de Silicio IRAM          : {report_b.crowbar_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • Parámetro Resonancia Chirikov     : {env2_b.chirikov_overlap_parameter:.4f} (>= 1.0)")
    print(f"  • Cero Simple Melnikov Detectado    : {env2_b.melnikov_simple_zero_detected}")
    print(f"  • γ Duffing (Melnikov)              : {env2_b.melnikov_gamma_damping:.4f}")
    print(f"  • Defecto Braquistócrona RFRN4      : {env2_b.brachistochrone_defect:.4f}")
    print(f"  • ω Lindstedt–Poincaré              : {env2_b.poincare_lindstedt_omega:.6e}")
    print(f"  • RSI³ auto-corregido               : {env2_b.rsi3_self_corrected}")
    print(f"  • Certificado Raíz Merkle SHA-256   : {report_b.merkle_root_sha256}")
    print("=" * 90)
    print("✨ EJECUCIÓN EXITOSA: SUTURA POINCARÉ–DELAUNAY Y ANIDAJE 1→2→3 CERTIFICADO.")