"""
Módulo: Neuromorphic Solver (Emulación de Diodo Lambda y NDR)
Versión: 7.0 (Correcciones físicas, matemáticas y de robustez sobre v6.1)

═══════════════════════════════════════════════════════════════════════════════
CORRECCIONES RESPECTO A v6.1
═══════════════════════════════════════════════════════════════════════════════

[C1] JFETModel.evaluate() — Régimen de acumulación (V_GS > 0 en canal N):
     En v6.1, ξ = v_gs/vp con vp<0 y v_gs>0 produce ξ<0 → (1-ξ)²>1 → I_D>I_DSS.
     Físicamente imposible. El modelo de Shockley se satura en I_DSS cuando
     ξ ≤ 0 (canal completamente abierto). Corrección: clamping ξ = max(ξ, 0.0).

[C2] JFETModel.evaluate() — g_ds canal P con signo incorrecto:
     v_ds_p = V_x - V_app < 0 → sign(V_DS)=-1 → g_ds_p<0.
     En el Jacobiano de LambdaDiodeTopology:
       df/dV_x = (g_m_n+g_ds_n) - (-g_m_p + g_ds_p)
               = g_m_n + g_ds_n + g_m_p - g_ds_p
     Con g_ds_p<0: -g_ds_p>0, inflando el Jacobiano. El modelo normalizado
     para canal P debe retornar g_ds siempre ≥ 0 (conductancia de magnitud),
     porque ya gestionamos el signo en la topología vía la derivada de |V_DS|.
     Corrección: g_ds = p.idss * (1-ξ)² * p.lam * abs(sign_vds).

[C3] _compute_dIdV() + _detect_ndr_regions() — Unidades inconsistentes:
     En v6.1 dIdV se calcula sobre currents_mA_smooth (unidad: mA) dando
     dIdV en [mA/V]. Se nombra "dIdV_mS" pero no es mS (=mA/V sí es mS ✓).
     Sin embargo en _detect_ndr_regions() se pasa dIdV_mS * 1e-3, convirtiendo
     a [A/V], y el umbral ndr_threshold está en [A/V] (−1e-4). El error está
     en que dIdV_mS ya es [mA/V]=[mS], y multiplicar por 1e-3 da [A/V]=[S],
     mientras el umbral −1e-4 A/V = −0.1 mS. La conversión es correcta pero
     confusa y propensa a errores futuros. Corrección: trabajar en unidades SI
     (A y V) desde el inicio; suavizar currents_A directamente.

[C4] _solve_with_fallback() — Predictor tangente sin correlación V_app:
     En v6.1 se extrapola V_x usando solo la historia de V_x convergentes,
     sin considerar el incremento ΔV_app. Esto puede generar semillas
     V_x > V_app (fuera del dominio). Corrección: predictor proporcional
     V_x_pred = V_x[-1] * (V_app / V_app_prev), que mantiene la fracción
     relativa de V_x respecto a V_app.

[C5] _detect_ndr_regions() — Cierre de región en borde del array:
     En v6.1 el cierre "if start_idx is not None" al final del bucle podría
     usar voltages[-1] con índice incorrecto si el array termina en NDR.
     Corrección: tratamiento explícito con índice final.

[C6] _smooth_current() — Garantía de ventana impar válida:
     En v6.1: window | 1 da impar solo si window es par (OR con 1).
     Pero si window=7 (impar), 7|1=7 ✓. Si window=6, 6|1=7 ✓.
     El bug real: min(window, n-1) puede dar par si n es par.
     Corrección: secuencia explícita ceil_odd → clamp → recheck odd.

[C7] _armijo_search() — División/comparación cuando f0=0:
     Si abs_f0=0, la condición Armijo f_trial ≤ 0*(1-c*α) = 0 nunca
     se satisface para f_trial≠0. Corrección: retorno inmediato si f0=0.

[C8] pv_ratios — El pico no está necesariamente en idx_start:
     En v6.1 se asume que el primer punto de la región NDR es el pico.
     El pico real es el máximo dentro de una ventana antes del inicio
     de la región NDR (o dentro si la región empieza justo tras el pico).
     Corrección: buscar máximo en ventana [max(0, idx_start-5), idx_start+1]
     y mínimo (valle) en [idx_end, min(n, idx_end+5)].

[C9] _project() — Colapso del dominio para V_app muy pequeño:
     eps = min(domain_eps, v_app*0.01). Para v_app=0.001 y domain_eps=1e-8,
     eps=1e-8. Para v_app=1e-9, eps=min(1e-8, 1e-11)=1e-11 y
     v_app-eps ≈ v_app > eps, OK. Pero para v_app=2e-8:
     eps = min(1e-8, 2e-10) = 2e-10; upper = 2e-8 - 2e-10 ≈ 1.98e-8 > eps ✓.
     El colapso ocurre si v_app < 2*domain_eps: eps=v_app*0.01 y
     upper=v_app*0.99 > v_app*0.01 ✓. En realidad el caso cubierto por
     "v_app < domain_eps" en solve_for_voltage. Refactorizar para claridad.

═══════════════════════════════════════════════════════════════════════════════
TOPOLOGÍA FÍSICA VERIFICADA DEL DIODO LAMBDA
═══════════════════════════════════════════════════════════════════════════════

        V_app
          │
        [S_p]  ← Fuente del P-JFET (J176)
          │
        P-JFET (J176, Canal P)
          │
        [D_p]──────────── Nodo X
          │                    │
        [G_p]←─── V_x    [G_n]──── V_x
                             │
                           N-JFET (2N5457, Canal N)
                             │
                           [D_n]──── Nodo X
                             │
                           [S_n]
                             │
                            GND

  Mapeo de tensiones:
  ┌─────────────────────────────────────────────────────────────┐
  │  N-JFET (Canal N):                                          │
  │    V_GS_n = V_x − 0     = +V_x   (≥ 0 en operación)       │
  │    V_DS_n = V_x − 0     = +V_x   (≥ 0)                    │
  │                                                             │
  │  P-JFET (Canal P):                                          │
  │    V_GS_p = V_x − V_app          (< 0)                     │
  │    V_DS_p = V_x − V_app          (< 0)                     │
  │    |V_GS_p| = V_app − V_x  (magnitud pasada al modelo)     │
  │                                                             │
  │  KCL en X: I_D_n − I_D_p = 0                              │
  └─────────────────────────────────────────────────────────────┘

  NDR emerge porque al aumentar V_app:
    − V_gs_n = V_x crece  → I_N sube
    − |V_gs_p| = V_app−V_x crece más rápido que V_x → I_P baja
    − El balance KCL tiene dI/dV_app < 0 en cierto rango
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger("NeuromorphicSolver")


# =============================================================================
# PARÁMETROS FÍSICOS DE LOS SEMICONDUCTORES
# =============================================================================

@dataclass(frozen=True)
class JFETParameters:
    """
    Parámetros intrínsecos del transistor JFET con unidades explícitas.

    Atributos
    ---------
    idss         : Corriente de saturación [A]. Siempre > 0.
    vp           : Tensión de pinch-off [V].
                   Canal N → vp < 0.
                   Canal P → vp > 0 (convención de magnitud).
    lam          : Coeficiente de modulación de longitud de canal [V⁻¹]. ≥ 0.
    is_n_channel : True → Canal N.  False → Canal P.

    Modelo de Shockley extendido (región activa):
    ─────────────────────────────────────────────
        ξ   = clamp(v_gs / vp, 0, 1)          # [C1] clamp inferior a 0
        I_D = I_DSS · (1 − ξ)² · (1 + λ·|V_DS|)

    El clamp inferior (ξ ≥ 0) implementa la saturación física en I_DSS cuando
    V_GS empuja el canal a acumulación (ξ < 0 en el cociente crudo).
    """
    idss: float
    vp: float
    lam: float
    is_n_channel: bool

    def __post_init__(self) -> None:
        if self.idss <= 0.0:
            raise ValueError(
                f"idss={self.idss:.3e} A debe ser estrictamente positivo."
            )
        if self.is_n_channel and self.vp >= 0.0:
            raise ValueError(
                f"Canal N requiere vp < 0 (recibido vp={self.vp} V)."
            )
        if not self.is_n_channel and self.vp <= 0.0:
            raise ValueError(
                f"Canal P requiere vp > 0 (recibido vp={self.vp} V)."
            )
        if self.lam < 0.0:
            raise ValueError(
                f"lam={self.lam:.3e} V⁻¹ debe ser no negativo."
            )


# Parámetros empíricos de datasheet
PARAM_2N5457 = JFETParameters(
    idss=3.0e-3,    # 3 mA típico datasheet
    vp=-1.5,        # −1.5 V  (canal N → pinch-off negativo)
    lam=0.02,       # 0.02 V⁻¹ modulación de canal
    is_n_channel=True,
)

PARAM_J176 = JFETParameters(
    idss=15.0e-3,   # 15 mA típico datasheet
    vp=2.5,         # +2.5 V  (canal P → magnitud de pinch-off)
    lam=0.02,       # 0.02 V⁻¹
    is_n_channel=False,
)


# =============================================================================
# MODELO FÍSICO DEL TRANSISTOR JFET
# =============================================================================

class JFETModel:
    """
    Evaluador del modelo de Shockley para JFET canal N y canal P.

    CONVENCIÓN DE SIGNOS UNIFICADA
    ──────────────────────────────
    El modelo recibe siempre magnitudes normalizadas:

      ─ Canal N: v_gs ∈ (vp, 0] con vp < 0
                 ξ = v_gs/vp ∈ [0, 1) en activa
                 En acumulación (v_gs > 0): ξ = v_gs/vp < 0 → clamp a 0 [C1]

      ─ Canal P: el llamador pasa |V_GS_p| y V_DS_p con su signo real.
                 ξ = |v_gs|/vp ∈ [0, 1) en activa  (vp > 0, |v_gs| < vp)
                 Corte: |v_gs| ≥ vp → ξ ≥ 1

    g_ds SE RETORNA COMO MAGNITUD (≥ 0) [C2]
    ──────────────────────────────────────────
    La dependencia de I_D con V_DS es vía |V_DS| en el término de modulación.
    La conductancia de salida física es siempre positiva:
        g_ds = ∂I_D / ∂|V_DS| = I_DSS · (1−ξ)² · λ ≥ 0

    El signo de la contribución al Jacobiano del circuito se gestiona en
    LambdaDiodeTopology.get_residual_and_jacobian(), donde se conoce
    la topología completa y la derivada de la cadena correcta.

    DERIVADAS ANALÍTICAS
    ─────────────────────
        ∂I_D/∂v_gs  = −2·I_DSS·(1−ξ)/vp · (1+λ·|V_DS|)   = g_m  ≥ 0
        ∂I_D/∂|v_ds| = I_DSS·(1−ξ)²·λ                      = g_ds ≥ 0
    """

    def __init__(self, params: JFETParameters) -> None:
        self.p = params

    def evaluate(
        self, v_gs: float, v_ds: float
    ) -> Tuple[float, float, float]:
        """
        Calcula (I_D, g_m, g_ds) para el estado (V_GS, V_DS).

        Parámetros
        ----------
        v_gs : Tensión puerta-fuente [V].
               Canal N → valor con signo (normalmente ≤ 0 en activa).
               Canal P → magnitud |V_GS_p| (normalmente > 0 en activa).
        v_ds : Tensión drenaje-fuente [V] con su signo real.

        Retorna
        -------
        i_d  : Corriente de drenaje [A], siempre ≥ 0.
        g_m  : Transconductancia [A/V] = ∂I_D/∂v_gs ≥ 0.
        g_ds : Conductancia de salida [A/V] = ∂I_D/∂|v_ds| ≥ 0.  [C2]
        """
        p = self.p

        # ── ξ = v_gs / vp ─────────────────────────────────────────────────────
        # Canal N: vp < 0, v_gs ≤ 0 en activa → ξ ∈ [0,1)
        #          v_gs > 0 (acumulación) → ξ < 0 → clamp a 0 [C1]
        # Canal P: vp > 0, v_gs = |V_GS_p| ∈ [0, vp) → ξ ∈ [0,1)
        xi = v_gs / p.vp
        xi = max(xi, 0.0)   # [C1] Clamp: I_D no puede superar I_DSS

        # ── Región de corte (ξ ≥ 1) ───────────────────────────────────────────
        if xi >= 1.0:
            return 0.0, 0.0, 0.0

        # ── Región activa ─────────────────────────────────────────────────────
        abs_vds      = abs(v_ds)
        one_minus_xi = 1.0 - xi                   # ∈ (0, 1]
        factor_vds   = 1.0 + p.lam * abs_vds      # ≥ 1

        i_d = p.idss * (one_minus_xi ** 2) * factor_vds

        # g_m = −2·I_DSS·(1−ξ)/vp · factor_vds
        # Para canal N: vp < 0 → −1/vp > 0 → g_m > 0  ✓
        # Para canal P: vp > 0 → −1/vp < 0 → g_m < 0?
        #   No: el llamador pasa |V_GS_p| y vp>0, así que ξ = |V_GS_p|/vp.
        #   La derivada respecto a |V_GS_p| es:
        #   ∂I_D/∂|V_GS_p| = −2·I_DSS·(1−ξ)/vp · factor_vds  (negativa para vp>0)
        #   → Correcto: al aumentar |V_GS_p|, I_D disminuye.
        #   El llamador maneja el signo vía la cadena d|V_GS_p|/dV_x = −1.
        g_m = 2.0 * p.idss * one_minus_xi * (-1.0 / p.vp) * factor_vds

        # g_ds = I_DSS·(1−ξ)²·λ  (MAGNITUD, siempre ≥ 0)  [C2]
        g_ds = p.idss * (one_minus_xi ** 2) * p.lam

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

        V_app ──[S_p]─ P-JFET(J176) ─[G_p,D_p]──┐
                                                   │ ← Nodo X
              GND ──[S_n]─ N-JFET(2N5457) ─[G_n,D_n]──┘

    MAPEO DE TENSIONES
    ──────────────────
    N-JFET (Canal N, 2N5457):
        V_GS_n = V_x − 0     = +V_x         (positivo; acumulación cuando V_x>0)
        V_DS_n = V_x − 0     = +V_x
        → Modelo recibe: v_gs=V_x (con signo), v_ds=V_x

    P-JFET (Canal P, J176):
        V_GS_p = V_x − V_app              (negativo)
        V_DS_p = V_x − V_app              (negativo)
        → Modelo recibe: v_gs=|V_GS_p|=V_app−V_x, v_ds=V_x−V_app

    KCL EN NODO X
    ─────────────
    Corriente que entra al nodo X desde arriba (P-JFET) = I_D_p  (magnitud)
    Corriente que sale del nodo X hacia abajo (N-JFET) = I_D_n  (magnitud)
    Equilibrio: f(V_x) = I_D_n − I_D_p = 0

    JACOBIANO ANALÍTICO CORREGIDO [C2]
    ───────────────────────────────────
    Usando la regla de la cadena con g_ds como magnitud:

    N-JFET: V_GS_n = V_x, V_DS_n = V_x
        dI_D_n/dV_x = g_m_n · (dV_GS_n/dV_x) + g_ds_n · (d|V_DS_n|/dV_x)
                    = g_m_n · (+1)              + g_ds_n · sign(V_x)·(+1)
                    = g_m_n + g_ds_n            (V_x > 0 en operación)

    P-JFET: v_gs_arg = |V_GS_p| = V_app − V_x,  v_ds_arg = V_x − V_app
        dI_D_p/dV_x = g_m_p · (d|V_GS_p|/dV_x) + g_ds_p · (d|V_DS_p|/dV_x)
                                                             [g_ds ya es magnitud]
        d|V_GS_p|/dV_x = d(V_app−V_x)/dV_x = −1
        d|V_DS_p|/dV_x = d|V_x−V_app|/dV_x = −sign(V_x−V_app) = +1  (V_x<V_app)
        dI_D_p/dV_x = g_m_p·(−1) + g_ds_p·(+1)
                    = −g_m_p + g_ds_p

    Jacobiano total:
        df/dV_x = dI_D_n/dV_x − dI_D_p/dV_x
                = (g_m_n + g_ds_n) − (−g_m_p + g_ds_p)
                = g_m_n + g_ds_n + g_m_p − g_ds_p

    ⚠️  Con g_ds_p ≥ 0 (magnitud), el signo de −g_ds_p es siempre negativo,
    reduciendo el Jacobiano de forma física: alta conductancia de salida del
    P-JFET tiende a estabilizar el punto de operación.
    """

    def __init__(self) -> None:
        self.jfet_n = JFETModel(PARAM_2N5457)
        self.jfet_p = JFETModel(PARAM_J176)

    def get_residual_and_jacobian(
        self, v_app: float, v_x: float
    ) -> Tuple[float, float, float, float]:
        """
        Calcula f(V_x) = I_D_n − I_D_p y J = df/dV_x.

        Parámetros
        ----------
        v_app : Tensión aplicada al ánodo [V]. Debe ser ≥ 0.
        v_x   : Tensión en el nodo interno X [V]. Físicamente ∈ [0, v_app].

        Retorna
        -------
        residual : f(V_x) = I_D_n − I_D_p  [A].
        jacobian : df/dV_x  [A/V = S].
        i_n      : Corriente N-JFET  [A] (magnitud).
        i_p      : Corriente P-JFET  [A] (magnitud).
        """
        # ── Mapeo de tensiones ────────────────────────────────────────────────
        # N-JFET: v_gs con signo (puede ser positivo → acumulación)
        v_gs_n = v_x           # V_GS_n = V_x (positivo)
        v_ds_n = v_x           # V_DS_n = V_x (positivo)

        # P-JFET: magnitud de V_GS_p, V_DS_p con signo real
        v_gs_p_mag = max(v_app - v_x, 0.0)   # |V_GS_p| ≥ 0 (protección numérica)
        v_ds_p     = v_x - v_app              # V_DS_p ≤ 0

        # ── Evaluación de modelos ─────────────────────────────────────────────
        i_n, g_m_n, g_ds_n = self.jfet_n.evaluate(v_gs_n, v_ds_n)
        i_p, g_m_p, g_ds_p = self.jfet_p.evaluate(v_gs_p_mag, v_ds_p)

        # ── Residuo KCL ───────────────────────────────────────────────────────
        residual = i_n - i_p

        # ── Jacobiano analítico (ver docstring de clase) ──────────────────────
        # g_m_n ≥ 0, g_ds_n ≥ 0 (magnitudes por [C2])
        # g_m_p: ∂I_D_p/∂|V_GS_p|. Con vp>0: g_m_p = 2·I_DSS·(1−ξ)·(-1/vp) < 0
        #        La cadena d|V_GS_p|/dV_x = -1 invierte el signo en la contrib.
        # g_ds_p ≥ 0 (magnitud por [C2])
        di_n_dvx = g_m_n + g_ds_n              # > 0 en operación
        di_p_dvx = g_m_p * (-1.0) + g_ds_p    # contribución P-JFET al Jacobiano
        jacobian = di_n_dvx - di_p_dvx         # df/dV_x

        return residual, jacobian, i_n, i_p


# =============================================================================
# SOLVER NEWTON-RAPHSON CON BACKTRACKING ARMIJO
# =============================================================================

class NewtonSolver:
    """
    Motor Newton-Raphson con backtracking tipo Armijo para sistemas escalares.

    Estrategia numérica
    ────────────────────
    1. Paso Newton puro: δ = −f / J.
    2. Backtracking Armijo: reduce α hasta que |f(x+α·δ)| < (1−c·α)·|f(x)|.
       Condición Armijo clásica con c = 1e-4, β = 0.5.  [C7] Retorno inmediato
       si f0 = 0 (ya convergido).
    3. Proyección al dominio físico: V_x ∈ [ε, V_app−ε].  [C9]
    4. Criterio de convergencia dual: |f| < tol  AND  |δ| < step_tol.
    5. Perturbación aleatoria ante Jacobiano singular (semilla fija por iteración).
    """

    _ARMIJO_C     = 1e-4    # Condición de Armijo (descenso suficiente)
    _ARMIJO_BETA  = 0.5     # Factor de reducción de paso
    _ARMIJO_ITERS = 30      # Aumentado de 20→30 para mayor robustez
    _JACO_EPS     = 1e-15   # Umbral de Jacobiano singular

    def __init__(
        self,
        max_iter:   int   = 150,
        tol:        float = 1e-10,
        step_tol:   float = 1e-9,
        domain_eps: float = 1e-8,
    ) -> None:
        if max_iter   <= 0: raise ValueError("max_iter debe ser positivo.")
        if tol        <= 0: raise ValueError("tol debe ser positivo.")
        if step_tol   <= 0: raise ValueError("step_tol debe ser positivo.")
        if domain_eps <  0: raise ValueError("domain_eps no puede ser negativo.")

        self.max_iter   = max_iter
        self.tol        = tol
        self.step_tol   = step_tol
        self.domain_eps = domain_eps
        self.topology   = LambdaDiodeTopology()

    def _project(self, v_x: float, v_app: float) -> float:
        """
        Proyecta V_x al dominio físico [ε, V_app − ε].

        [C9] La fracción del 1% evita colapso del dominio para V_app pequeño.
        Se garantiza ε < V_app/2 tomando ε = min(domain_eps, v_app * 0.01).
        Para V_app muy pequeño (< 2·domain_eps), la cobertura del solve_for_voltage
        ya retorna trivialmente (0, 0, True).
        """
        eps   = min(self.domain_eps, v_app * 0.01)
        lower = eps
        upper = v_app - eps
        if lower >= upper:          # V_app extremadamente pequeño (no debería ocurrir)
            return v_app * 0.5
        return max(lower, min(upper, v_x))

    def _armijo_search(
        self,
        v_app: float,
        v_x:   float,
        delta: float,
        f0:    float,
    ) -> Tuple[float, float]:
        """
        Búsqueda lineal con condición de Armijo.

        [C7] Si f0 = 0, el punto ya es solución: retorno inmediato.

        Busca α ∈ {1, β, β², …} tal que:
            |f(v_x + α·δ)| ≤ (1 − c·α)·|f(v_x)|

        Retorna (mejor_v_x, mejor_residuo).
        """
        # [C7] Caso trivial: ya convergido
        if f0 == 0.0:
            return v_x, 0.0

        abs_f0   = abs(f0)
        alpha    = 1.0
        best_vx  = v_x
        best_res = f0

        for _ in range(self._ARMIJO_ITERS):
            v_trial = self._project(v_x + alpha * delta, v_app)
            f_trial, _, _, _ = self.topology.get_residual_and_jacobian(v_app, v_trial)
            abs_f_trial = abs(f_trial)

            # Condición de descenso suficiente (Armijo)
            if abs_f_trial <= abs_f0 * (1.0 - self._ARMIJO_C * alpha):
                return v_trial, f_trial

            # Rastrear el mejor punto hallado (por si Armijo nunca se satisface)
            if abs_f_trial < abs(best_res):
                best_vx  = v_trial
                best_res = f_trial

            alpha *= self._ARMIJO_BETA

        logger.debug(
            "Armijo no satisfecho: V_app=%.4f V_x=%.4f → mejor |f|=%.2e",
            v_app, v_x, abs(best_res),
        )
        return best_vx, best_res

    def solve_for_voltage(
        self, v_app: float, v_x_guess: float
    ) -> Tuple[float, float, bool]:
        """
        Resuelve f(V_x) = I_D_n(V_x) − I_D_p(V_x) = 0 por Newton-Raphson.

        Parámetros
        ----------
        v_app     : Tensión aplicada [V]. Debe ser ≥ 0.
        v_x_guess : Semilla inicial para V_x [V].

        Retorna
        -------
        v_x       : Solución V_x [V].
        i_through : Corriente a través del dispositivo [A] = I_D_n.
        converged : True si el criterio de convergencia se satisfizo.
        """
        if v_app < 0.0:
            raise ValueError(f"v_app={v_app:.6f} V debe ser ≥ 0.")

        # ── Caso trivial ──────────────────────────────────────────────────────
        if v_app < self.domain_eps:
            return 0.0, 0.0, True

        # ── Inicialización ────────────────────────────────────────────────────
        v_x = self._project(v_x_guess, v_app)

        for iteration in range(self.max_iter):
            try:
                f, J, i_n, _ = self.topology.get_residual_and_jacobian(v_app, v_x)
            except Exception as exc:
                logger.error("Error en evaluación del modelo [iter=%d]: %s", iteration, exc)
                return v_x, 0.0, False

            abs_f = abs(f)

            # ── Criterio 1: convergencia por residuo ──────────────────────────
            if abs_f < self.tol:
                logger.debug(
                    "Convergencia por |f| en iter=%d: |f|=%.2e", iteration, abs_f
                )
                return v_x, i_n, True

            # ── Recuperación ante Jacobiano singular ──────────────────────────
            if abs(J) < self._JACO_EPS:
                logger.debug(
                    "Jacobiano singular en iter=%d, V_x=%.6f V", iteration, v_x
                )
                rng = np.random.default_rng(seed=iteration)
                v_x = self._project(
                    v_x + rng.uniform(-0.05, 0.05) * v_app, v_app
                )
                continue

            # ── Paso Newton con limitación ────────────────────────────────────
            delta = -f / J
            max_step = 0.40 * v_app
            if abs(delta) > max_step:
                delta = math.copysign(max_step, delta)

            # ── Búsqueda lineal Armijo ────────────────────────────────────────
            v_x_new, f_new = self._armijo_search(v_app, v_x, delta, f)

            # ── Criterio 2: convergencia por tamaño de paso ───────────────────
            step_taken = abs(v_x_new - v_x)
            if step_taken < self.step_tol:
                _, _, i_n_new, _ = self.topology.get_residual_and_jacobian(
                    v_app, v_x_new
                )
                logger.debug(
                    "Convergencia por |δV_x| en iter=%d: |δ|=%.2e", iteration, step_taken
                )
                return v_x_new, i_n_new, True

            v_x = v_x_new

        # ── Sin convergencia ──────────────────────────────────────────────────
        logger.warning(
            "Sin convergencia en V_app=%.4f V tras %d iteraciones. Último V_x=%.6f V",
            v_app, self.max_iter, v_x,
        )
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
    Usa V_x[k-1] como semilla para el punto k. Si falla:
      1. Predictor proporcional: V_x_pred = V_x_prev · (V_app / V_app_prev) [C4]
      2. Punto medio: V_app / 2
      3. Fracción 10%: V_app · 0.1
      4. Fracción 90%: V_app · 0.9

    Detección NDR [C3]
    ──────────────────
    1. Se trabaja en unidades SI (A, V) desde el inicio.
    2. Suavizado con Savitzky-Golay sobre currents_A.
    3. dI/dV en [A/V = S] con diferencias centrales no uniformes.
    4. NDR = dI/dV < ndr_threshold (umbral en A/V).
    """

    def __init__(
        self,
        ndr_threshold:  float = -1e-4,    # Umbral de conductancia diferencial [A/V]
        ndr_min_points: int   = 3,         # Mínimo puntos consecutivos para NDR válido
        smooth_window:  int   = 7,         # Ventana Savitzky-Golay (impar, ≥ 5)
    ) -> None:
        """
        Parámetros
        ----------
        ndr_threshold  : Umbral de conductancia diferencial [A/V].
                         Valor negativo (p.ej. −1e-4 A/V = −0.1 mS).
        ndr_min_points : Puntos consecutivos mínimos con dI/dV < umbral.
        smooth_window  : Ventana del filtro Savitzky-Golay. Se ajusta a impar. [C6]
        """
        if ndr_threshold >= 0.0:
            raise ValueError("ndr_threshold debe ser negativo para detectar NDR.")
        if ndr_min_points < 1:
            raise ValueError("ndr_min_points debe ser ≥ 1.")
        if smooth_window < 5:
            raise ValueError("smooth_window debe ser ≥ 5.")

        self.ndr_threshold  = ndr_threshold
        self.ndr_min_points = ndr_min_points
        self.smooth_window  = smooth_window
        self.solver         = NewtonSolver()

    def _build_voltage_sweep(
        self, v_start: float, v_end: float, steps: int
    ) -> np.ndarray:
        """
        Construye el vector de tensiones de barrido.

        Para v_start = 0: combina una zona logarítmica densa (10% del rango,
        ∼15% de los puntos) con una zona lineal uniforme (90% restante).
        Esto aumenta la resolución en la región de curvatura alta sin coste
        adicional significativo.
        """
        if v_start == 0.0:
            n_log = max(2, steps // 10)
            n_lin = steps - n_log
            v_log = np.logspace(
                -5, np.log10(max(1e-4, 0.1 * v_end)), n_log
            )
            # linspace desde el último punto log hasta v_end, sin duplicar
            v_lin = np.linspace(v_log[-1], v_end, n_lin + 1)[1:]
            return np.concatenate([[0.0], v_log, v_lin])
        return np.linspace(v_start, v_end, steps)

    def _solve_with_fallback(
        self,
        v_app:      float,
        v_app_prev: float,          # V_app del punto anterior (para predictor) [C4]
        v_x_prev:   float,          # V_x del punto anterior (puede no converger)
        vx_converged_history: List[Tuple[float, float]],  # [(v_app_k, v_x_k)] convergentes
    ) -> Tuple[float, float, bool]:
        """
        Resuelve el punto (v_app) con estrategia de semillas múltiples.

        Orden de intento:
          1. Continuación directa: v_x_prev.
          2. Predictor proporcional: v_x_prev · (v_app / v_app_prev).  [C4]
          3. Punto medio: v_app / 2.
          4. Fracción baja:  v_app · 0.1.
          5. Fracción alta:  v_app · 0.9.

        [C4] El predictor proporcional mantiene la relación V_x/V_app, que es
        más estable que una extrapolación lineal de V_x solo.
        """
        seeds: List[float] = [v_x_prev]

        # Predictor proporcional [C4]
        if v_app_prev > 0.0 and v_x_prev > 0.0:
            v_x_proportional = v_x_prev * (v_app / v_app_prev)
            seeds.append(v_x_proportional)

        # Semillas de respaldo
        seeds += [v_app / 2.0, v_app * 0.1, v_app * 0.9]

        # Eliminar duplicados aproximados y restringir al dominio (0, v_app)
        seen: List[float] = []
        for s in seeds:
            s_clamped = max(1e-9, min(v_app - 1e-9, s))
            if not any(abs(s_clamped - prev) < 1e-10 for prev in seen):
                seen.append(s_clamped)

        for seed in seen:
            v_x, i_through, converged = self.solver.solve_for_voltage(v_app, seed)
            if converged:
                return v_x, i_through, True

        # Último recurso: punto medio sin garantía
        logger.warning("Todas las semillas fallaron para V_app=%.4f V", v_app)
        v_x, i_through, _ = self.solver.solve_for_voltage(v_app, v_app / 2.0)
        return v_x, i_through, False

    @staticmethod
    def _make_odd_window(window: int, n_points: int) -> int:
        """
        Retorna una ventana impar válida para Savitzky-Golay. [C6]

        Garantías:
          • w ≥ 5 (mínimo para polyorder=3).
          • w ≤ n_points − (1 si n_points es par, else 0).
          • w es impar.
        """
        # Asegurar impar
        w = window if window % 2 == 1 else window + 1
        # Límite superior: el mayor impar ≤ n_points
        max_w = n_points if n_points % 2 == 1 else n_points - 1
        w = min(w, max_w)
        # Mínimo
        w = max(w, 5)
        # Recheck impar tras clamp
        if w % 2 == 0:
            w -= 1
        return max(w, 5)

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Suaviza `signal` con filtro Savitzky-Golay de orden 3.

        Trabaja en las unidades que recibe (no asume escala). [C3]
        Preserva la posición y amplitud de picos mejor que un MA.
        """
        n = len(signal)
        window = self._make_odd_window(self.smooth_window, n)
        if n < window:
            return signal.copy()
        try:
            return savgol_filter(signal, window_length=window, polyorder=3)
        except Exception as exc:
            logger.warning("Savitzky-Golay falló (%s); sin suavizado.", exc)
            return signal.copy()

    @staticmethod
    def _compute_dIdV(
        voltages: np.ndarray, currents: np.ndarray
    ) -> np.ndarray:
        """
        Calcula dI/dV en [unidad_corriente / V] con diferencias centrales. [C3]

        Para espaciado no uniforme (interior):
            (dI/dV)_i = (I[i+1] − I[i−1]) / (V[i+1] − V[i−1])

        Bordes: diferencias de primer orden hacia adelante/atrás.
        División por zero protegida con umbral 1e-15.
        """
        n    = len(currents)
        dIdV = np.empty(n)

        # Interior: diferencias centrales no uniformes
        dV = voltages[2:] - voltages[:-2]
        dI = currents[2:] - currents[:-2]
        safe_dV    = np.where(np.abs(dV) > 1e-15, dV, 1e-15)
        dIdV[1:-1] = dI / safe_dV

        # Bordes
        dV0 = voltages[1]  - voltages[0]
        dVN = voltages[-1] - voltages[-2]
        dIdV[0]  = (currents[1]  - currents[0])  / (dV0 if abs(dV0) > 1e-15 else 1e-15)
        dIdV[-1] = (currents[-1] - currents[-2]) / (dVN if abs(dVN) > 1e-15 else 1e-15)

        return dIdV

    def _detect_ndr_regions(
        self,
        voltages: np.ndarray,
        dIdV:     np.ndarray,    # [C3] En unidades SI [A/V]
    ) -> List[Tuple[float, float, int, int]]:
        """
        Detecta regiones NDR contiguas donde dI/dV < ndr_threshold.

        [C5] Cierre de región al final del array con índice explícito.

        Retorna
        -------
        Lista de (V_inicio, V_fin, idx_ini, idx_fin) para cada región válida.
        """
        in_ndr: np.ndarray = dIdV < self.ndr_threshold
        regions: List[Tuple[float, float, int, int]] = []
        start_idx: Optional[int] = None
        n = len(voltages)

        for i in range(n):
            if in_ndr[i] and start_idx is None:
                start_idx = i
            elif not in_ndr[i] and start_idx is not None:
                # Fin de región NDR en índice i-1
                end_idx = i - 1
                if (end_idx - start_idx + 1) >= self.ndr_min_points:
                    regions.append(
                        (voltages[start_idx], voltages[end_idx], start_idx, end_idx)
                    )
                start_idx = None

        # [C5] Región que llega al final del array
        if start_idx is not None:
            end_idx = n - 1
            if (end_idx - start_idx + 1) >= self.ndr_min_points:
                regions.append(
                    (voltages[start_idx], voltages[end_idx], start_idx, end_idx)
                )

        return regions

    @staticmethod
    def _compute_pvr(
        currents_smooth: np.ndarray,
        idx_start:       int,
        idx_end:         int,
        n_points:        int,
    ) -> float:
        """
        Calcula el factor pico-valle (PVR) para una región NDR dada. [C8]

        El pico se busca en una ventana justo ANTES del inicio de la región NDR
        (donde la corriente es máxima antes de caer). El valle se busca
        en una ventana justo DESPUÉS del fin de la región NDR.

        Ventanas de búsqueda:
          Pico  : [max(0, idx_start−5), idx_start+2]  (incluye el borde del inicio NDR)
          Valle : [idx_end−1, min(n_points, idx_end+6)]

        PVR = I_pico / I_valle  (si I_valle > 0, else ∞)
        """
        # Ventana para el pico (antes o al inicio de la región NDR)
        peak_lo = max(0, idx_start - 5)
        peak_hi = min(n_points, idx_start + 2)
        i_peak  = float(np.max(currents_smooth[peak_lo:peak_hi]))

        # Ventana para el valle (después o al final de la región NDR)
        valley_lo = max(0, idx_end - 1)
        valley_hi = min(n_points, idx_end + 6)
        i_valley  = float(np.min(currents_smooth[valley_lo:valley_hi]))

        if i_valley > 0.0:
            return i_peak / i_valley
        return float("inf")

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
        v_start : Tensión inicial del barrido [V]. ≥ 0.
        v_end   : Tensión final del barrido  [V]. > v_start.
        steps   : Número de puntos del barrido. ≥ 10.

        Retorna
        -------
        Diccionario con:
          "voltage_V"                  : List[float] – Tensiones [V].
          "current_A"                  : List[float] – Corrientes [A] (sin suavizar).
          "current_A_smooth"           : List[float] – Corrientes suavizadas [A].
          "current_mA"                 : List[float] – Corrientes [mA] (sin suavizar).
          "current_mA_smooth"          : List[float] – Corrientes suavizadas [mA].
          "differential_conductance_S" : List[float] – dI/dV [A/V = S].
          "differential_conductance_mS": List[float] – dI/dV [mA/V = mS].
          "ndr_detected"               : bool.
          "ndr_regions"                : List[(V_ini, V_fin)] – Regiones NDR [V].
          "current_peaks_mA"           : List[(V, I_mA)] – Picos locales suavizados.
          "convergence_flags"          : List[bool].
          "pv_ratios"                  : List[float] – Factor pico-valle por región.
        """
        # ── Validación ────────────────────────────────────────────────────────
        if v_end <= v_start:
            raise ValueError(f"v_end={v_end} debe ser > v_start={v_start}.")
        if steps < 10:
            raise ValueError(f"steps={steps} debe ser ≥ 10.")

        logger.info(
            "Iniciando barrido I-V: %.3f V → %.3f V, %d puntos.", v_start, v_end, steps
        )

        voltages_arr = self._build_voltage_sweep(v_start, v_end, steps)
        n_points     = len(voltages_arr)

        currents_A   = np.zeros(n_points)
        conv_flags   = [False] * n_points

        # ── Barrido con continuación ──────────────────────────────────────────
        v_x_prev:    float = 0.0
        v_app_prev:  float = 0.0
        vx_converged_history: List[Tuple[float, float]] = []  # [(v_app, v_x)]

        for k, v_app in enumerate(voltages_arr):
            v_x, i_through, converged = self._solve_with_fallback(
                v_app,
                v_app_prev,
                v_x_prev,
                vx_converged_history,
            )

            currents_A[k] = i_through
            conv_flags[k] = converged

            if converged:
                vx_converged_history.append((v_app, v_x))
                v_x_prev   = v_x
                v_app_prev = v_app
            else:
                logger.warning(
                    "Sin convergencia: V_app=%.4f V (punto %d/%d)", v_app, k + 1, n_points
                )
                # Mejor estimación de emergencia: mantener la corriente anterior
                if k > 0:
                    currents_A[k] = currents_A[k - 1]
                # NO actualizar v_x_prev ni v_app_prev para no propagar la semilla mala

        # ── Suavizado [C3]: operar directamente en [A] ────────────────────────
        currents_A_smooth = self._smooth_signal(currents_A)

        # ── Derivada dI/dV en [A/V = S] [C3] ─────────────────────────────────
        dIdV_S = self._compute_dIdV(voltages_arr, currents_A_smooth)

        # ── Detección NDR [C3][C5]: umbral en [A/V] ───────────────────────────
        ndr_regions_full = self._detect_ndr_regions(voltages_arr, dIdV_S)
        ndr_detected     = len(ndr_regions_full) > 0

        # Extraer solo (V_ini, V_fin) para la salida pública
        ndr_regions_public: List[Tuple[float, float]] = [
            (v0, v1) for v0, v1, _, _ in ndr_regions_full
        ]

        # ── Picos locales (sobre corriente suavizada en A) ────────────────────
        current_peaks_mA: List[Tuple[float, float]] = []
        for k in range(1, n_points - 1):
            if (currents_A_smooth[k] > currents_A_smooth[k - 1] and
                    currents_A_smooth[k] > currents_A_smooth[k + 1]):
                current_peaks_mA.append(
                    (float(voltages_arr[k]), float(currents_A_smooth[k] * 1e3))
                )

        # ── Factores pico-valle [C8] ──────────────────────────────────────────
        pv_ratios: List[float] = []
        for _, _, idx_s, idx_e in ndr_regions_full:
            pvr = self._compute_pvr(currents_A_smooth, idx_s, idx_e, n_points)
            pv_ratios.append(pvr)

        logger.info(
            "Barrido completado: NDR=%s, %d región(es), %d pico(s), "
            "convergencia %d/%d (%.1f%%).",
            ndr_detected,
            len(ndr_regions_full),
            len(current_peaks_mA),
            sum(conv_flags),
            n_points,
            100.0 * sum(conv_flags) / n_points,
        )

        return {
            "voltage_V":                   voltages_arr.tolist(),
            "current_A":                   currents_A.tolist(),
            "current_A_smooth":            currents_A_smooth.tolist(),
            "current_mA":                  (currents_A * 1e3).tolist(),
            "current_mA_smooth":           (currents_A_smooth * 1e3).tolist(),
            "differential_conductance_S":  dIdV_S.tolist(),
            "differential_conductance_mS": (dIdV_S * 1e3).tolist(),
            "ndr_detected":                ndr_detected,
            "ndr_regions":                 ndr_regions_public,
            "current_peaks_mA":            current_peaks_mA,
            "convergence_flags":           conv_flags,
            "pv_ratios":                   pv_ratios,
        }


# =============================================================================
# EJECUCIÓN STANDALONE
# =============================================================================

def _print_results(results: Dict) -> None:
    """Imprime un reporte legible y estructurado de los resultados."""
    voltages   = results["voltage_V"]
    currents   = results["current_mA"]
    peaks      = results["current_peaks_mA"]
    ndr_regs   = results["ndr_regions"]
    conv       = results["convergence_flags"]
    pv_ratios  = results.get("pv_ratios", [])

    n_total     = len(conv)
    n_converged = sum(1 for c in conv if c)
    conv_pct    = 100.0 * n_converged / n_total if n_total > 0 else 0.0

    # Conductancia diferencial mínima (máxima NDR)
    dIdV_mS = results.get("differential_conductance_mS", [])
    g_min   = min(dIdV_mS) if dIdV_mS else float("nan")

    sep = "=" * 65

    print(f"\n{sep}")
    print("   RESULTADOS DEL ANÁLISIS NEUROMÓRFICO — Diodo Lambda")
    print(sep)
    print(f"  Puntos de muestreo      : {n_total}")
    print(f"  Rango de tensión        : {voltages[0]:.4f} V → {voltages[-1]:.4f} V")
    print(f"  Convergencia            : {n_converged}/{n_total}  ({conv_pct:.1f}%)")
    print(f"  Corriente máxima        : {max(currents):.4f} mA")
    print(f"  Corriente mínima (>0)   : {min(c for c in currents if c > 0):.6f} mA"
          if any(c > 0 for c in currents) else "  Corriente mínima        : N/A")
    print(f"  Conductancia mín (dI/dV): {g_min:.4f} mS")

    # Picos
    print(f"\n  Picos de corriente detectados : {len(peaks)}")
    for v_pk, i_pk in peaks:
        print(f"    •  V = {v_pk:.4f} V   →   I = {i_pk:.4f} mA")

    # Regiones NDR
    print(f"\n  Regiones NDR detectadas : {len(ndr_regs)}")
    if results["ndr_detected"]:
        for idx, (v0, v1) in enumerate(ndr_regs):
            width = v1 - v0
            line  = f"    ✅  NDR {idx + 1}: [{v0:.4f} V,  {v1:.4f} V]  (ΔV = {width:.4f} V)"
            if idx < len(pv_ratios):
                pvr = pv_ratios[idx]
                pvr_str = f"{pvr:.3f}×" if math.isfinite(pvr) else "∞ (valle→0)"
                line += f"   PVR = {pvr_str}"
            print(line)

        # Resumen PVR
        if pv_ratios:
            pvr0 = pv_ratios[0]
            print(f"\n  Factor pico-valle (PVR) principal : ", end="")
            if math.isfinite(pvr0):
                print(f"{pvr0:.3f}×")
                if pvr0 > 1.5:
                    print("  ⚡ PVR alto → región NDR apta para oscilaciones neuromórficas.")
                else:
                    print("  ⚠️  PVR moderado → ajustar parámetros JFET.")
            else:
                print("∞  (corriente de valle ≈ 0 A)")

        print("\n  🧠 NDR confirmada → capacidad de generación de spikes neuromórficos.")
    else:
        print("  ❌ No se detectó NDR. Revise parámetros JFET o rango de tensión.")

    print(f"{sep}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    )

    try:
        # ndr_threshold en A/V: −1e-4 A/V = −0.1 mS
        analyzer = NeuromorphicAnalyzer(
            ndr_threshold=-1e-4,
            ndr_min_points=3,
            smooth_window=7,
        )
        results = analyzer.simulate_iv_curve(v_start=0.0, v_end=5.0, steps=300)
        _print_results(results)

    except Exception:
        logger.exception("Error fatal durante la ejecución.")
        raise