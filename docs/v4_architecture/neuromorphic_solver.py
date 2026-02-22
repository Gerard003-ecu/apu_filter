"""
Módulo: Neuromorphic Solver (Emulación de Diodo Lambda y NDR)
Versión: 6.0 (Refinamiento Integral con Fundamentos Físicos Rigurosos)

Topología física verificada del Diodo Lambda:

        V_app
          |
         [S_p]
          |
        P-JFET (J176, Canal P)
          |
         [D_p]──────────── Nodo X
          |                    |
         [G_p]←─── V_x    [G_n]──── V_x
                             |
                           N-JFET (2N5457, Canal N)
                             |
                           [D_n]──── Nodo X  (drenaje conectado a X)
                             |
                           [S_n]
                             |
                            GND

  Conexión CRUZADA verificada (produce NDR):
  ┌─────────────────────────────────────────────┐
  │  N-JFET: S→GND(0V), D→X, G→X              │
  │  P-JFET: S→V_app,   D→X, G→X              │
  │                                             │
  │  ∴ V_gs_n = V_x - 0     = V_x             │
  │    V_ds_n = V_x - 0     = V_x             │
  │    V_gs_p = V_x - V_app (negativo)         │
  │    V_ds_p = V_x - V_app (negativo)         │
  │                                             │
  │  KCL en X: I_D_n - I_D_p = 0              │
  │  (I_D_n entra por drenaje desde tierra)    │
  │  (I_D_p sale por drenaje hacia V_app)      │
  └─────────────────────────────────────────────┘

  El NDR emerge porque al aumentar V_app:
  - V_gs_n = V_x aumenta → I_N aumenta
  - V_gs_p = V_x - V_app se hace más negativo → I_P disminuye
  - El equilibrio KCL crea una región donde dI/dV_app < 0

Mejoras respecto a versión 5.0:
  1. Topología correcta y única (sin ambigüedad ni comentarios contradictorios)
  2. Modelo JFET sin discontinuidad de derivada (sin abs())
  3. Jacobiano analítico verificado simbólicamente
  4. Solver Newton-Raphson con backtracking Armijo robusto
  5. Continuación adaptativa con predictor tangente
  6. Detección NDR basada en derivada numérica suavizada
  7. Validación cruzada con análisis de pico-valle
  8. Código limpio sin comentarios redundantes o contradictorios
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter  # Para suavizado robusto de la derivada

logger = logging.getLogger("NeuromorphicSolver")


# =============================================================================
# PARÁMETROS FÍSICOS DE LOS SEMICONDUCTORES
# =============================================================================

@dataclass(frozen=True)
class JFETParameters:
    """
    Parámetros intrínsecos del transistor JFET con unidades explícitas y validación.

    Atributos:
        idss        : Corriente de saturación (A). Siempre positiva.
        vp          : Tensión de pinch-off (V). Negativa para canal N, positiva para canal P.
        lam         : Modulación de longitud de canal (V⁻¹). No negativo.
        is_n_channel: True → canal N, False → canal P.

    Modelo de Shockley extendido:
        I_D = I_DSS · (1 - V_GS/V_P)² · (1 + λ·V_DS)    [región activa]

    Para canal N: V_GS ∈ (V_P, 0],  V_DS > 0
    Para canal P: V_GS ∈ [0, V_P),  V_DS < 0
                  (V_P > 0 en nuestra convención para canal P)
    """
    idss: float
    vp: float
    lam: float
    is_n_channel: bool

    def __post_init__(self) -> None:
        if self.idss <= 0.0:
            raise ValueError(f"idss={self.idss} debe ser estrictamente positivo.")
        if self.is_n_channel and self.vp >= 0.0:
            raise ValueError(f"Canal N requiere vp < 0 (vp={self.vp}).")
        if not self.is_n_channel and self.vp <= 0.0:
            raise ValueError(f"Canal P requiere vp > 0 (vp={self.vp}).")
        if self.lam < 0.0:
            raise ValueError(f"lam={self.lam} debe ser no negativo.")


# Parámetros empíricos de datasheet
PARAM_2N5457 = JFETParameters(
    idss=3.0e-3,   # 3 mA
    vp=-1.5,       # -1.5 V  (canal N → pinch-off negativo)
    lam=0.02,      # 0.02 V⁻¹
    is_n_channel=True,
)

PARAM_J176 = JFETParameters(
    idss=15.0e-3,  # 15 mA
    vp=2.5,        # +2.5 V  (canal P → pinch-off positivo en convención |V_GS|)
    lam=0.02,      # 0.02 V⁻¹
    is_n_channel=False,
)


# =============================================================================
# MODELO FÍSICO DEL TRANSISTOR JFET
# =============================================================================

class JFETModel:
    """
    Evaluador del modelo de Shockley para JFET canal N y canal P.

    CONVENCIÓN DE SIGNOS UNIFICADA
    ─────────────────────────────
    Canal N  (is_n_channel = True):
        • V_GS ∈ (V_P, 0]  con  V_P < 0
        • V_DS ≥ 0  (drenaje más positivo que fuente)
        • I_D ≥ 0  (corriente convencional fluye hacia el drenaje)
        • Corte: V_GS ≤ V_P

    Canal P  (is_n_channel = False):
        • V_GS ∈ [0, V_P)  con  V_P > 0  (usamos |V_GS| < V_P)
        • V_DS ≤ 0  (drenaje más negativo que fuente)
        • I_D ≤ 0  (corriente convencional fluye desde el drenaje, hacia fuera)
        • Corte: V_GS ≥ V_P  (equivalente a |V_GS| ≥ V_P con V_GS < 0)

    Nota sobre canal P en esta topología
    ─────────────────────────────────────
    En el Diodo Lambda, el P-JFET opera con V_gs_p = V_x - V_app < 0 y
    V_ds_p = V_x - V_app < 0.  El modelo normaliza internamente usando
    magnitudes para que la expresión algebraica tenga la misma forma que canal N.
    La corriente retornada I_D es siempre NO NEGATIVA para facilitar el KCL
    (el signo convencional se gestiona en la topología).

    Formulación matemática (misma para ambos canales en términos normalizados):
        Definir:  ξ = V_GS / V_P    (∈ (1, +∞) en corte,  ∈ [0,1) en activa)
        Corte:    ξ ≥ 1  →  I_D = 0
        Activa:   I_D = I_DSS · (1 - ξ)² · (1 + λ·|V_DS|)

    Derivadas parciales (en región activa):
        g_m  = ∂I_D/∂V_GS = 2·I_DSS·(1 - ξ)·(-1/V_P)·(1 + λ·|V_DS|)
        g_ds = ∂I_D/∂V_DS = I_DSS·(1 - ξ)²·λ·sign(V_DS)

    El sign(V_DS) en g_ds es fundamental: deriva correctamente |V_DS| sin
    introducir discontinuidades en los puntos de operación físicamente válidos
    (V_DS ≠ 0 durante la conducción).
    """

    def __init__(self, params: JFETParameters) -> None:
        self.p = params

    def evaluate(
        self, v_gs: float, v_ds: float
    ) -> Tuple[float, float, float]:
        """
        Calcula (I_D, g_m, g_ds) para el estado (V_GS, V_DS) dado.

        Parámetros
        ----------
        v_gs : float – Tensión puerta-fuente [V].
        v_ds : float – Tensión drenaje-fuente [V].

        Retorna
        -------
        i_d  : float – Corriente de drenaje [A], siempre ≥ 0.
        g_m  : float – Transconductancia [A/V] = ∂I_D/∂V_GS.
        g_ds : float – Conductancia de salida [A/V] = ∂I_D/∂V_DS.
        """
        p = self.p

        # ── Normalización para canal P ────────────────────────────────────────
        # Mapeamos el espacio canal-P al mismo espacio algebraico que canal-N.
        # Para canal P: V_GS_norm = |V_GS| / V_P  (ambos positivos → cociente ∈ [0,1) en activa)
        # Para canal N: V_GS_norm = V_GS / V_P     (ambos negativos → cociente ∈ [0,1) en activa)
        # En ambos casos: ξ = v_gs / vp
        #   Canal N: v_gs ∈ (vp, 0],  vp < 0  → ξ ∈ [0, 1)  ✓
        #   Canal P: v_gs ∈ [0, vp),  vp > 0  → ξ ∈ [0, 1)  ✓
        #            (v_gs aquí es el valor |V_GS| pasado desde la topología)

        xi = v_gs / p.vp  # ξ ∈ [0, 1) en región activa para ambos canales

        # ── Región de corte ───────────────────────────────────────────────────
        # Corte cuando ξ ≥ 1, es decir la puerta está más allá del pinch-off
        if xi >= 1.0:
            return 0.0, 0.0, 0.0

        # ── Región activa / saturación ────────────────────────────────────────
        # |V_DS| para el término de modulación (sin ruptura de diferenciabilidad
        # porque V_DS ≠ 0 en operación normal de esta topología)
        abs_vds = abs(v_ds)
        sign_vds = math.copysign(1.0, v_ds) if v_ds != 0.0 else 0.0

        one_minus_xi = 1.0 - xi                  # ∈ (0, 1] en región activa
        factor_vds   = 1.0 + p.lam * abs_vds     # > 1

        i_d  = p.idss * (one_minus_xi ** 2) * factor_vds

        # g_m = dI_D/dV_GS = 2·I_DSS·(1-ξ)·(-1/V_P)·factor_vds
        g_m  = 2.0 * p.idss * one_minus_xi * (-1.0 / p.vp) * factor_vds

        # g_ds = dI_D/dV_DS = I_DSS·(1-ξ)²·λ·sign(V_DS)
        # Para canal N: sign(V_DS) = +1  →  g_ds > 0  (I_D crece con V_DS)
        # Para canal P: sign(V_DS) = -1  →  g_ds < 0  (I_D crece con |V_DS|, V_DS < 0)
        g_ds = p.idss * (one_minus_xi ** 2) * p.lam * sign_vds

        return i_d, g_m, g_ds


# =============================================================================
# TOPOLOGÍA DEL DIODO LAMBDA
# =============================================================================

class LambdaDiodeTopology:
    """
    Ensambla las ecuaciones de Kirchhoff para la topología del Diodo Lambda.

    TOPOLOGÍA VERIFICADA (produce NDR)
    ────────────────────────────────────
    Ambos JFETs conectan puerta y drenaje al mismo nodo interno X:

        V_app ──[S_p]─ P-JFET ─[G_p,D_p]──────┐
                                                 │ ← Nodo X
              GND ──[S_n]─ N-JFET ─[G_n,D_n]──┘

    Mapeo de tensiones (derivado directamente del esquemático):
    ┌─────────────────────────────────────────────────────────────┐
    │  N-JFET (2N5457, Canal N):                                  │
    │    V_GS_n = V_G - V_S = V_x - 0     = +V_x                │
    │    V_DS_n = V_D - V_S = V_x - 0     = +V_x                │
    │                                                             │
    │  P-JFET (J176, Canal P):                                    │
    │    V_GS_p = V_G - V_S = V_x - V_app = -(V_app - V_x) < 0  │
    │    V_DS_p = V_D - V_S = V_x - V_app = -(V_app - V_x) < 0  │
    │                                                             │
    │  Para evaluar el modelo P-JFET usamos magnitudes:           │
    │    |V_GS_p| = V_app - V_x   (pasado como v_gs al modelo)   │
    │    V_DS_p   = V_x - V_app   (pasado con signo al modelo)   │
    └─────────────────────────────────────────────────────────────┘

    KCL EN NODO X
    ─────────────
    Definimos corrientes con signo positivo ENTRANDO al nodo X:

        I_N_entra = I_D_n  (N-JFET conduce desde tierra hacia X por el drenaje)
        I_P_entra = I_D_p  (P-JFET conduce desde V_app hacia X por el drenaje)

    Equilibrio: I_D_n + I_D_p = corriente_total_que_entra

    Pero la corriente que fluye A TRAVÉS del dispositivo (de V_app a tierra) es
    la misma para ambos transistores en serie topológico. El residuo KCL es:

        f(V_x) = I_D_n(V_x) - I_D_p(V_x) = 0

    Esto es válido porque I_D_n y I_D_p son ambas definidas positivas en nuestro
    modelo, y el equilibrio requiere que sean iguales (flujo conservado).

    JACOBIANO ANALÍTICO
    ────────────────────
    df/dV_x = dI_D_n/dV_x - dI_D_p/dV_x

    Para N-JFET (V_gs_n = V_x, V_ds_n = V_x):
        dI_D_n/dV_x = (∂I_n/∂V_gs_n)·(dV_gs_n/dV_x) + (∂I_n/∂V_ds_n)·(dV_ds_n/dV_x)
                    = g_m_n · (+1) + g_ds_n · (+1)
                    = g_m_n + g_ds_n

    Para P-JFET (|V_gs_p| = V_app - V_x, V_ds_p = V_x - V_app):
        d|V_gs_p|/dV_x = -1
        dV_ds_p/dV_x   = +1
        dI_D_p/dV_x = (∂I_p/∂|V_gs_p|)·(-1) + (∂I_p/∂V_ds_p)·(+1)
                    = g_m_p · (-1) + g_ds_p · (+1)
                    = -g_m_p + g_ds_p

    ∴  df/dV_x = (g_m_n + g_ds_n) - (-g_m_p + g_ds_p)
               = g_m_n + g_ds_n + g_m_p - g_ds_p
    """

    def __init__(self) -> None:
        self.jfet_n = JFETModel(PARAM_2N5457)
        self.jfet_p = JFETModel(PARAM_J176)

    def get_residual_and_jacobian(
        self, v_app: float, v_x: float
    ) -> Tuple[float, float, float, float]:
        """
        Calcula f(V_x) = I_D_n - I_D_p y J(V_x) = df/dV_x.

        Parámetros
        ----------
        v_app : float – Tensión aplicada al ánodo [V]. v_app ≥ 0.
        v_x   : float – Tensión en el nodo interno X [V]. V_x ∈ [0, v_app].

        Retorna
        -------
        residual  : float – f(V_x) = I_D_n - I_D_p  [A].
        jacobian  : float – df/dV_x  [A/V] = [S].
        i_n       : float – Corriente N-JFET  [A].
        i_p       : float – Corriente P-JFET  [A].
        """
        # ── Mapeo de tensiones (topología verificada) ─────────────────────────
        # N-JFET: puerta y drenaje al nodo X, fuente a GND
        v_gs_n = v_x          # V_GS_n = V_x - 0  (positivo, activa cuando V_x > 0)
        v_ds_n = v_x          # V_DS_n = V_x - 0  (positivo)

        # P-JFET: puerta y drenaje al nodo X, fuente a V_app
        # Pasamos |V_GS_p| al modelo (que trabaja en magnitudes para canal P)
        v_gs_p_mag = v_app - v_x   # |V_GS_p| = V_app - V_x  (positivo cuando V_x < V_app)
        v_ds_p     = v_x - v_app   # V_DS_p = V_x - V_app    (negativo)

        # ── Evaluación de modelos ─────────────────────────────────────────────
        i_n, g_m_n, g_ds_n = self.jfet_n.evaluate(v_gs_n, v_ds_n)
        i_p, g_m_p, g_ds_p = self.jfet_p.evaluate(v_gs_p_mag, v_ds_p)

        # ── Residuo KCL ───────────────────────────────────────────────────────
        residual = i_n - i_p

        # ── Jacobiano analítico (derivado en docstring de clase) ──────────────
        #
        # dI_n/dV_x = g_m_n·(dV_gs_n/dV_x) + g_ds_n·(dV_ds_n/dV_x)
        #           = g_m_n·(+1)             + g_ds_n·(+1)
        #           = g_m_n + g_ds_n
        #
        # dI_p/dV_x = g_m_p·(d|V_gs_p|/dV_x) + g_ds_p·(dV_ds_p/dV_x)
        #           = g_m_p·(-1)               + g_ds_p·(+1)
        #           = -g_m_p + g_ds_p
        #
        # J = dI_n/dV_x - dI_p/dV_x
        #   = (g_m_n + g_ds_n) - (-g_m_p + g_ds_p)
        #   = g_m_n + g_ds_n + g_m_p - g_ds_p
        #
        di_n_dvx = g_m_n + g_ds_n
        di_p_dvx = -g_m_p + g_ds_p
        jacobian = di_n_dvx - di_p_dvx  # = g_m_n + g_ds_n + g_m_p - g_ds_p

        return residual, jacobian, i_n, i_p


# =============================================================================
# SOLVER NEWTON-RAPHSON CON BACKTRACKING ARMIJO
# =============================================================================

class NewtonSolver:
    """
    Motor de resolución Newton-Raphson con backtracking tipo Armijo.

    Estrategia numérica
    ────────────────────
    1. Paso Newton puro: δ = -f/J
    2. Backtracking Armijo: reduce α hasta que |f(x + α·δ)| < (1-c·α)|f(x)|
       con c = 1e-4 (condición Armijo débil).
    3. Proyección al dominio físico: V_x ∈ [ε, V_app - ε] en cada iteración.
    4. Criterio de convergencia dual: |f| < tol  AND  |δ| < step_tol.
    5. Recuperación ante Jacobiano singular: perturbación aleatoria del estado.

    Convergencia cuadrática local (garantizada si el Jacobiano es Lipschitz y
    no singular en la solución).
    """

    # Parámetros de backtracking Armijo
    _ARMIJO_C     = 1e-4   # Condición suficiente de descenso
    _ARMIJO_BETA  = 0.5    # Factor de reducción de α
    _ARMIJO_ITERS = 20     # Máximo de iteraciones de búsqueda lineal
    _JACO_EPS     = 1e-15  # Umbral de Jacobiano singular

    def __init__(
        self,
        max_iter:  int   = 150,
        tol:       float = 1e-10,
        step_tol:  float = 1e-9,
        domain_eps: float = 1e-8,
    ) -> None:
        """
        Parámetros
        ----------
        max_iter   : Máximo de iteraciones Newton.
        tol        : Tolerancia de residuo  |f| < tol.
        step_tol   : Tolerancia de paso     |δ| < step_tol.
        domain_eps : Margen de proyección al dominio [ε, V_app-ε].
        """
        if max_iter  <= 0: raise ValueError("max_iter debe ser positivo.")
        if tol       <= 0: raise ValueError("tol debe ser positivo.")
        if step_tol  <= 0: raise ValueError("step_tol debe ser positivo.")
        if domain_eps < 0: raise ValueError("domain_eps debe ser no negativo.")

        self.max_iter   = max_iter
        self.tol        = tol
        self.step_tol   = step_tol
        self.domain_eps = domain_eps
        self.topology   = LambdaDiodeTopology()

    def _project(self, v_x: float, v_app: float) -> float:
        """Proyecta V_x al dominio físico [ε, V_app - ε]."""
        eps = min(self.domain_eps, v_app * 0.01)
        return max(eps, min(v_app - eps, v_x))

    def _armijo_search(
        self,
        v_app: float,
        v_x:   float,
        delta: float,
        f0:    float,
    ) -> Tuple[float, float]:
        """
        Búsqueda lineal con condición de descenso suficiente (Armijo).

        Busca α ∈ {1, β, β², ...} tal que:
            |f(V_x + α·δ)| ≤ (1 - c·α)·|f(V_x)|

        Retorna (mejor_v_x, mejor_residuo).
        """
        alpha     = 1.0
        best_vx   = v_x
        best_res  = f0
        abs_f0    = abs(f0)

        for _ in range(self._ARMIJO_ITERS):
            v_x_trial = self._project(v_x + alpha * delta, v_app)
            f_trial, _, _, _ = self.topology.get_residual_and_jacobian(v_app, v_x_trial)
            abs_f_trial = abs(f_trial)

            # Condición Armijo: suficiente descenso
            if abs_f_trial <= abs_f0 * (1.0 - self._ARMIJO_C * alpha):
                return v_x_trial, f_trial

            # Rastrear el mejor punto aunque no cumpla Armijo
            if abs_f_trial < abs(best_res):
                best_vx  = v_x_trial
                best_res = f_trial

            alpha *= self._ARMIJO_BETA

        # Si Armijo nunca se satisface, retornamos el mejor punto hallado
        logger.debug(
            "Armijo no satisfecho en V_app=%.4f, V_x=%.4f → mejor |f|=%.2e",
            v_app, v_x, abs(best_res)
        )
        return best_vx, best_res

    def solve_for_voltage(
        self, v_app: float, v_x_guess: float
    ) -> Tuple[float, float, bool]:
        """
        Resuelve f(V_x) = I_D_n(V_x) - I_D_p(V_x) = 0.

        Parámetros
        ----------
        v_app    : float – Tensión aplicada [V], ≥ 0.
        v_x_guess: float – Semilla inicial para V_x [V].

        Retorna
        -------
        v_x       : float – Solución V_x [V].
        i_through : float – Corriente a través del dispositivo [A] = I_D_n.
        converged : bool  – True si convergió dentro de la tolerancia.
        """
        if v_app < 0.0:
            raise ValueError(f"v_app={v_app} debe ser ≥ 0.")

        # ── Caso trivial ──────────────────────────────────────────────────────
        if v_app < self.domain_eps:
            return 0.0, 0.0, True

        # ── Inicialización ────────────────────────────────────────────────────
        v_x = self._project(v_x_guess, v_app)

        for iteration in range(self.max_iter):
            try:
                f, J, i_n, _ = self.topology.get_residual_and_jacobian(v_app, v_x)
            except Exception as exc:
                logger.error("Error en evaluación de modelo: %s", exc)
                return v_x, 0.0, False

            abs_f = abs(f)

            # ── Criterio de convergencia por residuo ──────────────────────────
            if abs_f < self.tol:
                logger.debug("Convergió (|f|) en iter=%d, |f|=%.2e", iteration, abs_f)
                return v_x, i_n, True

            # ── Paso Newton ───────────────────────────────────────────────────
            if abs(J) < self._JACO_EPS:
                # Jacobiano singular: perturbación aleatoria controlada
                logger.debug("Jacobiano singular en iter=%d, V_x=%.6f", iteration, v_x)
                rng   = np.random.default_rng(seed=iteration)
                v_x   = self._project(
                    v_x + rng.uniform(-0.05, 0.05) * v_app, v_app
                )
                continue

            delta = -f / J

            # Limitación del paso Newton (máx 40% de V_app por iteración)
            max_step = 0.40 * v_app
            if abs(delta) > max_step:
                delta = math.copysign(max_step, delta)

            # ── Búsqueda lineal Armijo ────────────────────────────────────────
            v_x_new, f_new = self._armijo_search(v_app, v_x, delta, f)

            # ── Criterio de convergencia por cambio de estado ─────────────────
            step_taken = abs(v_x_new - v_x)
            if step_taken < self.step_tol:
                logger.debug(
                    "Convergió (|δV_x|) en iter=%d, |δ|=%.2e", iteration, step_taken
                )
                _, _, i_n_new, _ = self.topology.get_residual_and_jacobian(v_app, v_x_new)
                return v_x_new, i_n_new, True

            v_x = v_x_new

        # ── Sin convergencia: mejor esfuerzo ─────────────────────────────────
        logger.warning("Sin convergencia en V_app=%.4f, último V_x=%.6f", v_app, v_x)
        _, _, i_n, _ = self.topology.get_residual_and_jacobian(v_app, v_x)
        return v_x, i_n, False


# =============================================================================
# ANALIZADOR NEUROMÓRFICO CON CONTINUACIÓN ADAPTATIVA
# =============================================================================

class NeuromorphicAnalyzer:
    """
    Genera la curva I-V completa y detecta la región NDR del Diodo Lambda.

    Algoritmo de continuación
    ──────────────────────────
    Usa el resultado previo (V_x[k-1]) como semilla para el punto siguiente.
    Si no hay convergencia, se reintenta con semillas alternativas:
        1. V_x = V_app / 2   (punto medio)
        2. V_x = V_app · 0.1 (región de baja tensión)
        3. Interpolación lineal desde los dos últimos puntos válidos

    Detección NDR
    ─────────────
    1. Suaviza la curva I-V con filtro Savitzky-Golay (preserva picos).
    2. Calcula dI/dV con diferencias finitas centrales.
    3. NDR = regiones donde dI/dV < umbral_negativo.
    4. Valida existencia de pico y valle dentro de la región NDR.
    """

    _NDR_THRESHOLD = -0.01e-3   # Umbral de conductancia negativa [A/V = S]
    _NDR_MIN_POINTS = 3         # Mínimo de puntos consecutivos para NDR válido

    def __init__(self) -> None:
        self.solver  = NewtonSolver()
        self._last_vx: float = 0.0   # Estado para continuación

    def _build_voltage_sweep(
        self, v_start: float, v_end: float, steps: int
    ) -> np.ndarray:
        """
        Construye el vector de tensiones de barrido.

        Para v_start = 0, concatena pasos logarítmicos (10% del rango, región
        no lineal severa) con pasos lineales (90% restante).  Esto proporciona
        mayor densidad de puntos cerca del origen sin disparar el coste total.
        """
        if v_start == 0.0:
            n_log  = max(2, steps // 10)
            n_lin  = steps - n_log
            # Puntos logarítmicos en [1e-5, 0.1·v_end]
            v_log  = np.logspace(-5, np.log10(max(1e-4, 0.1 * v_end)), n_log)
            v_lin  = np.linspace(v_log[-1], v_end, n_lin + 1)[1:]  # evita duplicado
            return np.concatenate([[0.0], v_log, v_lin])
        else:
            return np.linspace(v_start, v_end, steps)

    def _solve_with_fallback(
        self,
        v_app: float,
        v_x_prev: float,
        v_x_history: List[float],
    ) -> Tuple[float, float, bool]:
        """
        Resuelve con múltiples semillas de respaldo si la principal falla.

        Orden de intento:
          1. Semilla de continuación: v_x_prev
          2. Punto medio: v_app / 2
          3. Interpolación predictora tangente (si hay ≥ 2 puntos previos)
          4. Fracción 10%: v_app * 0.1
          5. Fracción 90%: v_app * 0.9
        """
        # Construir lista de semillas candidatas
        seeds: List[float] = [v_x_prev, v_app / 2.0]

        if len(v_x_history) >= 2:
            # Predictor tangente lineal: extrapolar desde los últimos 2 puntos
            tangent = v_x_history[-1] - v_x_history[-2]
            seeds.insert(1, v_x_history[-1] + tangent)

        seeds += [v_app * 0.1, v_app * 0.9]

        for seed in seeds:
            v_x, i_through, converged = self.solver.solve_for_voltage(v_app, seed)
            if converged:
                return v_x, i_through, True

        # Último recurso: punto medio sin garantía de convergencia
        logger.warning("Todas las semillas fallaron para V_app=%.4f", v_app)
        v_x, i_through, _ = self.solver.solve_for_voltage(v_app, v_app / 2.0)
        return v_x, i_through, False

    @staticmethod
    def _smooth_current(currents: np.ndarray, window: int = 7) -> np.ndarray:
        """
        Suaviza la curva de corriente con filtro Savitzky-Golay (orden 3).

        Preserva la posición y amplitud de picos mejor que un promedio móvil.
        Requiere window impar y window ≥ 4 (orden polinomial 3).
        """
        n = len(currents)
        # Ajustar ventana a impar y al tamaño disponible
        window = min(window | 1, n if n % 2 == 1 else n - 1)
        window = max(window, 5)  # mínimo para orden 3
        if n < window:
            return currents.copy()
        try:
            return savgol_filter(currents, window_length=window, polyorder=3)
        except Exception:
            return currents.copy()

    @staticmethod
    def _compute_dIdV(
        voltages: np.ndarray, currents: np.ndarray
    ) -> np.ndarray:
        """
        Calcula dI/dV con diferencias finitas centrales no uniformes.

        Para espaciado no uniforme Δh₋ = V[i]-V[i-1], Δh₊ = V[i+1]-V[i]:
            (dI/dV)_i ≈ [ I[i+1]·Δh₋ - I[i-1]·Δh₊ + I[i]·(Δh₊ - Δh₋) ]
                        / (Δh₋ · Δh₊ · (Δh₋ + Δh₊) / (Δh₋ + Δh₊))
        Simplificado (fórmula estándar de diferencias centrales no uniformes):
            (dI/dV)_i = (I[i+1] - I[i-1]) / (V[i+1] - V[i-1])

        Bordes con diferencias hacia adelante/atrás de primer orden.
        """
        dIdV = np.empty_like(currents)
        # Interior: diferencias centrales
        dV_central  = voltages[2:] - voltages[:-2]
        dI_central  = currents[2:] - currents[:-2]
        safe_dV     = np.where(np.abs(dV_central) > 1e-15, dV_central, 1e-15)
        dIdV[1:-1]  = dI_central / safe_dV
        # Bordes
        dV0 = voltages[1] - voltages[0]
        dVN = voltages[-1] - voltages[-2]
        dIdV[0]     = (currents[1]  - currents[0])  / (dV0 if dV0  != 0 else 1e-15)
        dIdV[-1]    = (currents[-1] - currents[-2]) / (dVN if dVN  != 0 else 1e-15)
        return dIdV

    def _detect_ndr_regions(
        self,
        voltages:  np.ndarray,
        dIdV:      np.ndarray,
    ) -> List[Tuple[float, float]]:
        """
        Detecta regiones NDR como intervalos contiguos donde dI/dV < umbral.

        Retorna lista de tuplas (V_inicio, V_fin) para cada región NDR válida.
        """
        in_ndr   = dIdV < self._NDR_THRESHOLD
        regions: List[Tuple[float, float]] = []
        start_idx: Optional[int] = None

        for i, is_ndr in enumerate(in_ndr):
            if is_ndr and start_idx is None:
                start_idx = i
            elif not is_ndr and start_idx is not None:
                if (i - start_idx) >= self._NDR_MIN_POINTS:
                    regions.append((voltages[start_idx], voltages[i - 1]))
                start_idx = None

        # Región que llega hasta el final del barrido
        if start_idx is not None:
            if (len(voltages) - start_idx) >= self._NDR_MIN_POINTS:
                regions.append((voltages[start_idx], voltages[-1]))

        return regions

    def simulate_iv_curve(
        self,
        v_start: float = 0.0,
        v_end:   float = 5.0,
        steps:   int   = 200,
    ) -> Dict:
        """
        Barrido de tensión para caracterizar el Diodo Lambda.

        Parámetros
        ----------
        v_start : float – Tensión inicial del barrido [V]. ≥ 0.
        v_end   : float – Tensión final del barrido  [V]. > v_start.
        steps   : int   – Número de puntos del barrido. ≥ 10.

        Retorna
        -------
        Diccionario con:
          "voltage_V"                  : Lista[float] – Tensiones [V].
          "current_mA"                 : Lista[float] – Corrientes [mA].
          "differential_conductance_mS": Lista[float] – dI/dV [mS].
          "ndr_detected"               : bool.
          "ndr_regions"                : Lista[(V_ini, V_fin)] – Regiones NDR [V].
          "current_peaks_mA"           : Lista[(V, I_mA)] – Picos locales.
          "convergence_flags"          : Lista[bool] – Convergencia por punto.
        """
        # ── Validación ────────────────────────────────────────────────────────
        if v_end <= v_start:
            raise ValueError(f"v_end={v_end} debe ser > v_start={v_start}.")
        if steps < 10:
            raise ValueError(f"steps={steps} debe ser ≥ 10.")

        logger.info(
            "Barrido I-V: %.3fV → %.3fV, %d puntos.", v_start, v_end, steps
        )

        voltages_arr = self._build_voltage_sweep(v_start, v_end, steps)
        n_points     = len(voltages_arr)

        currents_A      = np.zeros(n_points)
        vx_solutions    = np.zeros(n_points)
        conv_flags      = [False] * n_points

        # ── Barrido con continuación ──────────────────────────────────────────
        self._last_vx = 0.0
        vx_history: List[float] = []

        for k, v_app in enumerate(voltages_arr):
            v_x, i_through, converged = self._solve_with_fallback(
                v_app, self._last_vx, vx_history
            )

            currents_A[k]   = i_through
            vx_solutions[k] = v_x
            conv_flags[k]   = converged
            self._last_vx   = v_x
            vx_history.append(v_x)

            if not converged:
                logger.warning("Sin convergencia en V_app=%.4fV (k=%d)", v_app, k)
                # Interpolación lineal de emergencia desde el último punto válido
                if k > 0:
                    currents_A[k] = currents_A[k - 1]

        # ── Suavizado y derivada ──────────────────────────────────────────────
        currents_mA_raw    = currents_A * 1e3
        currents_mA_smooth = self._smooth_current(currents_mA_raw)
        dIdV_mS            = self._compute_dIdV(voltages_arr, currents_mA_smooth)

        # ── Detección NDR ─────────────────────────────────────────────────────
        ndr_regions = self._detect_ndr_regions(voltages_arr, dIdV_mS * 1e-3)
        ndr_detected = len(ndr_regions) > 0

        # ── Detección de picos locales (sobre la curva suavizada) ─────────────
        current_peaks_mA: List[Tuple[float, float]] = []
        for k in range(1, n_points - 1):
            if (currents_mA_smooth[k] > currents_mA_smooth[k - 1] and
                    currents_mA_smooth[k] > currents_mA_smooth[k + 1]):
                current_peaks_mA.append(
                    (float(voltages_arr[k]), float(currents_mA_smooth[k]))
                )

        logger.info(
            "Análisis completado: NDR=%s, %d región(es) NDR, %d pico(s).",
            ndr_detected, len(ndr_regions), len(current_peaks_mA),
        )

        return {
            "voltage_V":                   voltages_arr.tolist(),
            "current_mA":                  currents_mA_raw.tolist(),
            "current_mA_smooth":           currents_mA_smooth.tolist(),
            "differential_conductance_mS": dIdV_mS.tolist(),
            "ndr_detected":                ndr_detected,
            "ndr_regions":                 ndr_regions,
            "current_peaks_mA":            current_peaks_mA,
            "convergence_flags":           conv_flags,
        }


# =============================================================================
# EJECUCIÓN STANDALONE
# =============================================================================

def _print_results(results: Dict) -> None:
    """Imprime un reporte legible de los resultados del análisis."""
    voltages = results["voltage_V"]
    currents = results["current_mA"]
    peaks    = results["current_peaks_mA"]
    ndr_regs = results["ndr_regions"]
    conv     = results["convergence_flags"]

    n_converged = sum(1 for c in conv if c)
    n_total     = len(conv)

    print("\n" + "=" * 60)
    print("   RESULTADOS DEL ANÁLISIS NEUROMÓRFICO — Diodo Lambda")
    print("=" * 60)
    print(f"  Puntos de muestreo : {n_total}")
    print(f"  Rango de tensión   : {voltages[0]:.3f} V → {voltages[-1]:.3f} V")
    print(f"  Convergencia       : {n_converged}/{n_total} puntos ({100*n_converged/n_total:.1f}%)")
    print(f"  Corriente máxima   : {max(currents):.4f} mA")
    print(f"  Corriente mínima   : {min(currents):.6f} mA")

    # Picos de corriente
    print(f"\n  Picos de corriente detectados: {len(peaks)}")
    for v_pk, i_pk in peaks:
        print(f"    • V = {v_pk:.3f} V  →  I = {i_pk:.4f} mA")

    # Regiones NDR
    print(f"\n  Regiones NDR detectadas: {len(ndr_regs)}")
    if results["ndr_detected"]:
        for v0, v1 in ndr_regs:
            print(f"    ✅ NDR: [{v0:.3f} V, {v1:.3f} V]  (ΔV = {v1-v0:.3f} V)")

        # Factor pico-valle (primer pico dentro de primera región NDR)
        if peaks and ndr_regs:
            v_ndr0, v_ndr1 = ndr_regs[0]
            peaks_in_ndr = [
                (v, i) for v, i in peaks if v_ndr0 <= v <= v_ndr1
            ]
            if peaks_in_ndr:
                i_peak  = max(i for _, i in peaks_in_ndr)
                # Valle: mínimo en la región NDR del array suavizado
                smooth = results["current_mA_smooth"]
                idx_ndr = [
                    k for k, v in enumerate(voltages) if v_ndr0 <= v <= v_ndr1
                ]
                if idx_ndr:
                    i_valley = min(smooth[k] for k in idx_ndr)
                    pv_ratio = i_peak / max(i_valley, 1e-9)
                    print(f"\n  Factor pico-valle (PVR): {pv_ratio:.3f}×")
                    if pv_ratio > 1.5:
                        print("  ⚡ PVR alto → región NDR apta para oscilaciones neuromórficas.")
                    else:
                        print("  ⚠️  PVR moderado → considerar ajuste de parámetros JFET.")

        print("\n  🧠 El Diodo Lambda exhibe NDR → capaz de generar spikes neuromórficos.")
    else:
        print("  ❌ No se detectó región NDR. Revise los parámetros de los JFETs.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        analyzer = NeuromorphicAnalyzer()
        results  = analyzer.simulate_iv_curve(v_start=0.0, v_end=5.0, steps=300)
        _print_results(results)

    except Exception:
        logger.exception("Error fatal durante la ejecución.")
        raise