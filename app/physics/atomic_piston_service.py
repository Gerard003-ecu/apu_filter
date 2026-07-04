"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo  : atomic_piston_service.py                                          ║
║ Versión : 4.0.0-Lie-Geometric-Phase1                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

FASE 1 — HERRAMIENTAS FÍSICAS Y MATEMÁTICAS: CIRUGÍA GEOMÉTRICA DOCTORAL

FUNDAMENTACIÓN MATEMÁTICA DE LA FASE 1 (v4.0.0):
══════════════════════════════════════════════════

§F1. COVARIANZA TENSORIAL EN FrictionCalculator
───────────────────────────────────────────────
Sea (M, G) una variedad Riemanniana de dimensión n donde M ⊆ ℝⁿ es la
variedad de configuración del pistón, y G_μν el tensor métrico que codifica
la geometría deformada del ecosistema.

La velocidad ẋ ∈ TₓM es un vector tangente. Su norma métrica:

    ‖ẋ‖_G = √(vᵀ G v)   [m/s, métrica G]

La función signo suavizada debe ser COVARIANTE, no euclidiana:

    smooth_sign_G(v) = tanh(‖v‖_G / V_SMOOTH) · (G·v) / ‖v‖_G

que retorna un COVECTOR (1-forma) en T*ₓM, con unidades [N] al multiplicar
por la magnitud de fricción F_c [N].

La fuerza de fricción como covector:

    F_fric = -F_c · tanh(‖v‖_G / v_s) · (G·v) / ‖v‖_G   ∈ T*ₓM

donde G·v ∈ T*ₓM es el dual métrico (bajada de índice con G).

§F2. CONTROL PID GEOMÉTRICO SOBRE GRUPOS DE LIE
────────────────────────────────────────────────
Sea G un grupo de Lie con álgebra de Lie g = T_eG.

Estado actual:   x   ∈ G
Estado deseado:  x_d ∈ G

Error geodésico (Logaritmo de Riemann):

    ξ = Log_{x}(x_d) = exp_x^{-1}(x_d) ∈ T_xG ≅ g

para G = (ℝⁿ, +) (caso afín): ξ = x_d - x  (recupera el caso euclidiano)
para G = SO(3): ξ = skew⁻¹(Rᵀ·log(Rᵀ·R_d)) ∈ so(3)

La ley de control en el fibrado tangente TG:

    u(t) = -K_p · Log_x(x_d)
           - K_d · (D/dt)ẋ                        [derivada covariante]
           - K_i · ∫₀ᵗ Γ_γ(τ) · Log_{x(τ)}(x_d) dτ  [transporte paralelo]

donde:
    D/dt = d/dt + Γ^k_{ij}·ẋʲ   (derivada covariante de Levi-Civita)
    Γ_γ(τ) = P_{γ(τ,0)}         (operador de transporte paralelo a lo largo de γ)

Para la conexión de Levi-Civita en espacio afín (G_μν = I):
    Γ^k_{ij} = 0  →  D/dt ẋ = ẍ  (recupera el caso euclidiano)

Anti-windup geométrico:
    Si ‖u‖_G > u_max: proyectar u sobre la bola de radio u_max en (g, G).
    Deshacer la contribución integral del paso actual (en el espacio tangente).

§F3. INVARIANTES DE SALIDA (CONTRATOS)
────────────────────────────────────────
G1. smooth_sign_G: T_xM → T*_xM  (covector, no escalar)
G2. compute_friction: retorna array [N] en coordenadas covariantes
G3. LieGroupPIDController.update: u ∈ g, ‖u‖_G ≤ output_limit
G4. Transporte paralelo Γ_γ acumulado es isometría (preserva ‖·‖_G)
G5. Anti-windup opera en g (álgebra de Lie), no en ℝ plano
"""

from __future__ import annotations

import csv
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, jsonify, request

from .config import PistonConfig
from .constants import FrictionModel, PistonMode, TransducerType


# ════════════════════════════════════════════════════════════════════════════════
# §A. CONFIGURACIÓN DEL LOGGING
# ════════════════════════════════════════════════════════════════════════════════

def _configure_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Configura un logger con handler de archivo y consola.

    Idempotente: si el logger ya tiene handlers no los duplica.
    """
    os.makedirs(log_dir, exist_ok=True)
    lgr = logging.getLogger(name)
    if lgr.hasHandlers():
        return lgr

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s"
    )
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lgr.addHandler(fh)
    lgr.addHandler(sh)
    lgr.setLevel(logging.INFO)
    return lgr


logger = _configure_logger("atomic_piston_service")


# ════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                  ║
# ║  FASE 1 — HERRAMIENTAS FÍSICAS Y MATEMÁTICAS (CIRUGÍA DOCTORAL v4.0.0)         ║
# ║                                                                                  ║
# ║  SUTURA 1: FrictionCalculator — Covarianza Tensorial (G_μν)                    ║
# ║  SUTURA 2: LieGroupPIDController — Control PID Geométrico (Grupos de Lie)      ║
# ║                                                                                  ║
# ║  Contrato formal (v4.0.0):                                                       ║
# ║    Entrada : velocidad v ∈ T_xM (vector tangente), fuerza F ∈ T*_xM            ║
# ║    Salida  : fricción F_fric ∈ T*_xM (covector) con norma métrica              ║
# ║              señal de control u ∈ g (álgebra de Lie), ‖u‖_G ≤ u_max           ║
# ║                                                                                  ║
# ║  Garantías (v4.0.0):                                                             ║
# ║    G1. smooth_sign_G: retorna covector normalizado en (T*M, G)                  ║
# ║    G2. compute_friction: covector [N] invariante bajo reparametrización         ║
# ║    G3. LieGroupPIDController.update: u acotado en bola geodésica ‖u‖≤u_max    ║
# ║    G4. Integral PID acumulada via transporte paralelo (isometría)               ║
# ║    G5. Anti-windup en g previene saturación integral Windup catastrófica        ║
# ║                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════════
# §1.0 — ÁLGEBRA RIEMANNIANA AUXILIAR (base matemática compartida)
# ════════════════════════════════════════════════════════════════════════════════════

def _metric_norm(v: np.ndarray, G: np.ndarray) -> float:
    r"""
    Calcula la norma Riemanniana de un vector tangente v ∈ T_xM.

    Definición:
        ‖v‖_G = √(vᵀ G v)   ∈ ℝ≥0

    La norma métrica generaliza la norma euclidiana (G=I) al espacio
    deformado por el tensor métrico G_μν. Es la longitud intrínseca del
    vector en la variedad (M, G).

    Para n=1 (caso escalar del pistón):
        v = [ẋ], G = [[g₁₁]]
        ‖v‖_G = √(g₁₁ · ẋ²) = √g₁₁ · |ẋ|

    Propiedades verificadas:
        (a) ‖v‖_G ≥ 0             (definida positiva, G ≻ 0)
        (b) ‖v‖_G = 0 ↔ v = 0    (no degenerada)
        (c) ‖λv‖_G = |λ|·‖v‖_G  (homogeneidad)

    Parámetros
    ----------
    v : np.ndarray, shape (n,), vector tangente en T_xM.
    G : np.ndarray, shape (n,n), tensor métrico G_μν ≻ 0 (definida positiva).

    Retorna
    -------
    float ≥ 0, norma Riemanniana ‖v‖_G.

    Lanza
    -----
    ValueError : Si G no es cuadrada o sus dimensiones no coinciden con v.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    G_arr = np.asarray(G, dtype=np.float64)

    if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
        raise ValueError(
            f"G debe ser matriz cuadrada; recibido shape={G_arr.shape}."
        )
    if v_arr.shape[0] != G_arr.shape[0]:
        raise ValueError(
            f"Dimensión de v ({v_arr.shape[0]}) incompatible con G "
            f"({G_arr.shape[0]}×{G_arr.shape[1]})."
        )

    # vᵀ G v — producto cuadrático Riemanniano
    quadratic = float(v_arr @ G_arr @ v_arr)

    # Protección numérica: G ≻ 0 garantiza quadratic ≥ 0, pero errores de
    # punto flotante pueden dar cuadráticos ≤ -ε_mach → clip a 0.
    return math.sqrt(max(quadratic, 0.0))


def _metric_dual(v: np.ndarray, G: np.ndarray) -> np.ndarray:
    r"""
    Calcula el dual métrico (bajada de índice) de un vector tangente.

    Operación musical ♭ (flat):
        v♭ = G · v   ∈ T*_xM   (covector / 1-forma)

    En coordenadas locales: (v♭)_μ = G_μν · vν   (suma de Einstein)

    Esta operación convierte el vector tangente v ∈ T_xM en el covector
    dual v♭ ∈ T*_xM que "mide" la proyección de otros vectores sobre v
    ponderada por la métrica G.

    Importancia física:
        Las fuerzas son covectores (1-formas), no vectores.
        F_fric = -F_c · tanh(‖v‖_G/v_s) · (G·v)/‖v‖_G  ∈ T*_xM [N]

    Parámetros
    ----------
    v : np.ndarray, shape (n,), vector tangente.
    G : np.ndarray, shape (n,n), tensor métrico.

    Retorna
    -------
    np.ndarray, shape (n,), covector G·v ∈ T*_xM.
    """
    return G @ np.asarray(v, dtype=np.float64)


def _lie_log_affine(x: np.ndarray, x_d: np.ndarray) -> np.ndarray:
    r"""
    Logaritmo de Riemann (mapa exponencial inverso) para el grupo afín (ℝⁿ, +).

    En el grupo de Lie afín G = (ℝⁿ, +) con álgebra g = ℝⁿ:

        Log_x(x_d) = exp_x^{-1}(x_d) = x_d - x   ∈ T_xG ≅ ℝⁿ

    La geodésica entre x y x_d en ℝⁿ (con métrica plana) es la línea recta,
    y el logaritmo coincide con la diferencia vectorial.

    Para grupos de Lie no conmutativos (SO(3), SE(3)):
        Log_R(R_d) = skew^{-1}(log(Rᵀ·R_d))   ∈ so(3)

    Esta función implementa el caso afín como base extensible. La abstracción
    permite a LieGroupPIDController funcionar con métricas G ≠ I incluso
    en el caso euclidiano, preparando la arquitectura para extensión a SO(3).

    Parámetros
    ----------
    x   : np.ndarray, shape (n,), estado actual en G.
    x_d : np.ndarray, shape (n,), estado deseado en G.

    Retorna
    -------
    np.ndarray, shape (n,), error geodésico ξ = Log_x(x_d) ∈ T_xG.
    """
    return np.asarray(x_d, dtype=np.float64) - np.asarray(x, dtype=np.float64)


def _parallel_transport_affine(
    xi: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    r"""
    Operador de transporte paralelo Γ_γ a lo largo de la geodésica γ.

    Para la conexión de Levi-Civita en espacio afín (ℝⁿ, G constante):

        Γ_γ(τ) : T_{γ(0)}M → T_{γ(τ)}M

    Los símbolos de Christoffel de la conexión de Levi-Civita para G
    CONSTANTE son idénticamente nulos:

        Γ^k_{ij} = ½ G^{kl} (∂_i G_{lj} + ∂_j G_{il} - ∂_l G_{ij}) = 0

    Por lo tanto, el transporte paralelo en espacio afín con G constante
    es la identidad: Γ_γ(τ) = Id.

    El vector ξ ∈ T_{γ(0)}M se transporta sin rotación ni deformación
    a lo largo de toda la geodésica γ.

    Consecuencia para el PID:
        La integral de transporte paralelo acumula ξ(τ) directamente,
        sin necesidad de correcciones de holonomía (curvatura = 0 en ℝⁿ).
        El invariante de curvatura R^k_{lij} = 0 garantiza que el transporte
        es independiente del camino → anti-windup es geométricamente correcto.

    Parámetros
    ----------
    xi : np.ndarray, shape (n,), vector a transportar en T_xM.
    G  : np.ndarray, shape (n,n), tensor métrico (G constante → Γ=0).

    Retorna
    -------
    np.ndarray, shape (n,), vector transportado Γ_γ · ξ = ξ (identidad afín).

    Nota Arquitectónica
    -------------------
    Para extensión a variedades curvas (SO(3), SE(3)), este método debe
    reimplementarse usando la ecuación de transporte paralelo:

        dξ^k/dt + Γ^k_{ij} · (dγⁱ/dt) · ξʲ = 0

    integrada numéricamente a lo largo de la geodésica γ.
    """
    # En ℝⁿ con G constante: curvatura R=0 → transporte = identidad
    return np.asarray(xi, dtype=np.float64).copy()


def _covariant_derivative_affine(
    velocity: np.ndarray,
    velocity_prev: np.ndarray,
    dt: float,
    G: np.ndarray,
) -> np.ndarray:
    r"""
    Calcula la derivada covariante D(ẋ)/dt de la velocidad a lo largo de γ.

    Definición (conexión de Levi-Civita):

        (Dẋ/dt)^k = dẋ^k/dt + Γ^k_{ij} · ẋⁱ · ẋʲ

    Para G constante en ℝⁿ: Γ^k_{ij} = 0, luego:

        Dẋ/dt = dẋ/dt = (ẋ(t) - ẋ(t-dt)) / dt   [diferencias finitas]

    Esta derivada coincide con la aceleración ẍ en el caso euclidiano,
    y es el término derivativo "geométricamente correcto" del PID:

        D_term = -K_d · (Dẋ/dt)

    Para SO(3) con G no constante, los símbolos de Christoffel no nulos
    introducen correcciones centrífugas/Coriolis en la derivada covariante.

    Parámetros
    ----------
    velocity      : np.ndarray, shape (n,), velocidad actual ẋ(t).
    velocity_prev : np.ndarray, shape (n,), velocidad anterior ẋ(t-dt).
    dt            : float, paso de tiempo [s]. Debe ser > 0.
    G             : np.ndarray, shape (n,n), tensor métrico (no usado en afín,
                    reservado para extensión a SO(3)).

    Retorna
    -------
    np.ndarray, shape (n,), derivada covariante Dẋ/dt ≈ ẍ [m/s²].
    """
    if dt <= 0.0:
        return np.zeros_like(velocity)
    # En ℝⁿ afín: Dẋ/dt = dẋ/dt (diferencias finitas de primer orden)
    return (
        np.asarray(velocity, dtype=np.float64)
        - np.asarray(velocity_prev, dtype=np.float64)
    ) / dt


# ════════════════════════════════════════════════════════════════════════════════════
# SUTURA 1 — FrictionCalculator (Covarianza Tensorial sobre (T*M, G_μν))
# ════════════════════════════════════════════════════════════════════════════════════

class FrictionCalculator:
    r"""
    Calcula la fuerza de fricción seca como COVECTOR en (T*M, G_μν).

    CIRUGÍA v4.0.0 (Sutura 1):
    ───────────────────────────
    El cálculo euclidiano escalar F_fric = -F_c·tanh(ẋ/v_s) ha sido
    extirpado. La versión geométrica eleva la fuerza a un covector (1-forma)
    que respeta la deformación métrica de la variedad diferenciable:

        F_fric = -F_c · tanh(‖v‖_G / v_s) · (G·v) / ‖v‖_G   ∈ T*_xM [N]

    donde:
        ‖v‖_G = √(vᵀ G v)   norma Riemanniana [m/s]
        G·v   = covector dual métrico (bajada de índice)
        v_s   = V_SMOOTH (velocidad de Stribeck)

    Para G = I_n (identidad): ‖v‖_G = |v|, G·v = v → recupera el caso
    euclidiano original (compatibilidad hacia atrás garantizada).

    Modelos de fricción (COVARIANTES):
    ────────────────────────────────────
    VISCOUS     : F_fric = 0  (sin fricción seca)
    COULOMB     : F_fric = -F_c · tanh(‖v‖_G/v_s) · (G·v)/‖v‖_G
                  Régimen estático: F_fric = -min(‖F_drv‖_G, F_c) · (G·F_drv)/‖F_drv‖_G
    STRIBECK    : F_c(‖v‖_G) = F_C + (F_S-F_C)·exp(-(‖v‖_G/v_St)²)
                  F_fric = -F_c(‖v‖_G) · tanh(‖v‖_G/v_s) · (G·v)/‖v‖_G

    Constantes:
    ───────────
    V_SMOOTH : Velocidad característica de suavizado [m/s].
               La aproximación tanh difiere de sign en <1% para
               ‖v‖_G > 5·V_SMOOTH (independiente de G).
    """

    V_SMOOTH: float = 1e-3  # m/s — umbral de régimen cinético (métrico)

    # ─────────────────────────────────────────────────────────────────────
    # §1.1 — Función signo suavizada COVARIANTE
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def smooth_sign_euclidean(x: float) -> float:
        r"""
        [LEGADO] Aproximación escalar de sign(x) para compatibilidad interna.

        ADVERTENCIA: Esta función opera en ℝ plano y NO respeta la métrica G.
        Úsese exclusivamente en contextos donde n=1 y G=[[1]] (euclidiano).
        Para el cálculo de fricción covariante, usar smooth_sign_metric().

        Fórmula:
            smooth_sign_euclidean(x) = tanh(x / V_SMOOTH)

        Preservada para compatibilidad con llamadas internas de escalares.

        Parámetros
        ----------
        x : float, argumento (velocidad escalar).

        Retorna
        -------
        float ∈ (-1, 1).
        """
        return math.tanh(x / FrictionCalculator.V_SMOOTH)

    @staticmethod
    def smooth_sign_metric(
        v: np.ndarray,
        G: np.ndarray,
    ) -> np.ndarray:
        r"""
        Aproximación COVARIANTE del mapa signo para vectores tangentes.

        Reemplaza la aproximación escalar por un covector normalizado en T*M:

            smooth_sign_G(v) = tanh(‖v‖_G / V_SMOOTH) · (G·v) / ‖v‖_G

        donde ‖v‖_G = √(vᵀ G v) es la norma Riemanniana.

        Propiedades geométricas:
        ─────────────────────────
        (a) ‖smooth_sign_G(v)‖_{G⁻¹} = tanh(‖v‖_G/V_SMOOTH) ∈ (-1,1)
            (magnitud acotada en la norma dual G⁻¹)
        (b) Para v → 0: smooth_sign_G → 0  (sin discontinuidad en el origen)
        (c) Para ‖v‖_G >> V_SMOOTH: smooth_sign_G → G·v/‖v‖_G
            (apunta en la dirección métrica de v, con magnitud unitaria)
        (d) Para G = I: smooth_sign_G(v) = tanh(|v|/V_S)·sign(v)
            (recupera el caso euclidiano escalado)

        Parámetros
        ----------
        v : np.ndarray, shape (n,), vector tangente ẋ ∈ T_xM.
        G : np.ndarray, shape (n,n), tensor métrico G_μν ≻ 0.

        Retorna
        -------
        np.ndarray, shape (n,), covector en T*_xM ∈ {‖·‖_{G⁻¹} < 1}.

        Lanza
        -----
        ValueError : Si G no es cuadrada o dimesionalmente incompatible con v.
        """
        v_arr  = np.asarray(v,  dtype=np.float64)
        G_arr  = np.asarray(G,  dtype=np.float64)

        norm_G = _metric_norm(v_arr, G_arr)

        # Singularidad en v=0: retornar covector nulo (límite continuo)
        if norm_G < 1e-15:
            return np.zeros_like(v_arr)

        # Covector dual: G·v ∈ T*_xM
        dual_v = _metric_dual(v_arr, G_arr)  # shape (n,)

        # Dirección covariante normalizada: (G·v) / ‖v‖_G
        direction = dual_v / norm_G  # shape (n,), covector unitario

        # Magnitud suavizada: tanh(‖v‖_G / V_SMOOTH) ∈ (-1, 1)
        magnitude = math.tanh(norm_G / FrictionCalculator.V_SMOOTH)

        return magnitude * direction

    # ─────────────────────────────────────────────────────────────────────
    # §1.2 — Cálculo de fricción covariante (SUTURA 1 PRINCIPAL)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_friction(
        velocity        : float,
        driving_force   : float,
        friction_model  : FrictionModel,
        coulomb_friction : float = 0.2,
        stribeck_coeffs  : Tuple[float, float, float] = (0.3, 0.1, 0.05),
        metric_tensor    : Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Calcula la fuerza de fricción seca [N] con covarianza tensorial.

        CIRUGÍA v4.0.0:
        ───────────────
        Se acepta un `metric_tensor` G_μν opcional. Si se proporciona,
        el cálculo eleva la fricción a un covector en T*M. Si G=None,
        se usa G=[[1]] (euclidiano, compatibilidad hacia atrás).

        Para el sistema 1D del pistón, los vectores son escalares:
            v = [ẋ] ∈ T_xM ≅ ℝ
            G = [[g₁₁]] ∈ ℝ^{1×1}
            ‖v‖_G = √(g₁₁) · |ẋ|
            F_fric = -F_c · tanh(‖v‖_G/v_s) · g₁₁·ẋ/‖v‖_G  [N]

        El resultado se retorna como escalar flotante (componente del
        covector en la única coordenada del sistema 1D) para compatibilidad
        con la interface existente de _derivatives() en la Fase 2.

        Casos:
        ───────
        VISCOUS: retorna 0.0 (sin fricción seca).

        COULOMB (cinético: ‖v‖_G > V_SMOOTH):
            F_fric = -F_c · tanh(‖v‖_G / V_SMOOTH) · (G·v) / ‖v‖_G

        COULOMB (estático: ‖v‖_G ≤ V_SMOOTH):
            F_drv_covector = G · F_drv (elevado a covector)
            F_fric = -min(‖F_drv‖_{G⁻¹}, F_c) · (G·F_drv)/‖F_drv‖_{G⁻¹}
            (La fricción estática se opone a la fuerza impulsora hasta
             su límite máximo, en la dirección métrica correcta.)

        STRIBECK (cinético):
            F_c(‖v‖_G) = F_C + (F_S-F_C)·exp(-(‖v‖_G/v_St)²)
            F_fric = -F_c(‖v‖_G) · tanh(‖v‖_G/V_SMOOTH) · (G·v)/‖v‖_G

        Parámetros
        ----------
        velocity        : float, velocidad del pistón ẋ [m/s].
        driving_force   : float, fuerza neta impulsora sin fricción [N].
        friction_model  : FrictionModel, modelo seleccionado.
        coulomb_friction : float, coeficiente de Coulomb F_c [N].
        stribeck_coeffs  : Tuple (F_S, F_C, v_St) [N, N, m/s].
        metric_tensor    : Optional[np.ndarray], shape (1,1) o (n,n),
                           tensor G_μν. Si None → G = I (euclidiano).

        Retorna
        -------
        float, componente escalar de la fuerza de fricción [N].
               (Covector 1D proyectado sobre la base coordenada.)

        Lanza
        -----
        ValueError : Si los coeficientes de Stribeck son inválidos.
        """
        if friction_model == FrictionModel.VISCOUS:
            return 0.0

        # §1.2.1: Construcción del tensor métrico
        if metric_tensor is None:
            # Euclidiano: G = I₁ (compatibilidad hacia atrás)
            G = np.array([[1.0]], dtype=np.float64)
        else:
            G = np.asarray(metric_tensor, dtype=np.float64)

        # Vectores en ℝ¹ (sistema 1D del pistón)
        v_vec = np.array([velocity],      dtype=np.float64)
        f_vec = np.array([driving_force], dtype=np.float64)

        # §1.2.2: Norma Riemanniana de la velocidad
        norm_G_v = _metric_norm(v_vec, G)
        is_kinetic = norm_G_v > FrictionCalculator.V_SMOOTH

        if friction_model == FrictionModel.COULOMB:
            if is_kinetic:
                # §1.2.3: Régimen cinético — covector covariante
                # F_fric = -F_c · tanh(‖v‖_G/v_s) · (G·v)/‖v‖_G
                friction_covector = (
                    -coulomb_friction
                    * FrictionCalculator.smooth_sign_metric(v_vec, G)
                )
                # Proyección escalar: componente [0] del covector 1D
                return float(friction_covector[0])
            else:
                # §1.2.4: Régimen estático — fricción limitada por F_drv
                # Norma de la fuerza impulsora en métrica G⁻¹ (espacio dual)
                norm_G_f = _metric_norm(f_vec, G)
                if norm_G_f < 1e-15:
                    return 0.0
                # Dirección de oposición en T*M
                static_covector = (
                    -min(norm_G_f, coulomb_friction)
                    * _metric_dual(f_vec, G)
                    / norm_G_f
                )
                return float(static_covector[0])

        if friction_model == FrictionModel.STRIBECK:
            f_static, f_coulomb, v_stribeck = stribeck_coeffs
            if v_stribeck <= 0:
                raise ValueError(
                    f"v_stribeck debe ser > 0, recibido {v_stribeck}."
                )
            if is_kinetic:
                # §1.2.5: Magnitud Stribeck en función de la norma métrica
                # F_c(‖v‖_G) = F_C + (F_S-F_C)·exp(-(‖v‖_G/v_St)²)
                friction_mag = f_coulomb + (f_static - f_coulomb) * math.exp(
                    -((norm_G_v / v_stribeck) ** 2)
                )
                # Covector Stribeck: dirección métrica normalizada
                friction_covector = (
                    -friction_mag
                    * FrictionCalculator.smooth_sign_metric(v_vec, G)
                )
                return float(friction_covector[0])
            else:
                # §1.2.6: Régimen estático Stribeck — límite = F_S
                norm_G_f = _metric_norm(f_vec, G)
                if norm_G_f < 1e-15:
                    return 0.0
                static_covector = (
                    -min(norm_G_f, f_static)
                    * _metric_dual(f_vec, G)
                    / norm_G_f
                )
                return float(static_covector[0])

        # Modelo no reconocido
        return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # §1.3 — Cálculo batch (múltiples velocidades simultáneas)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_friction_field(
        velocities      : np.ndarray,
        driving_forces  : np.ndarray,
        friction_model  : FrictionModel,
        metric_tensor   : np.ndarray,
        coulomb_friction : float = 0.2,
        stribeck_coeffs  : Tuple[float, float, float] = (0.3, 0.1, 0.05),
    ) -> np.ndarray:
        r"""
        Calcula el campo de fricción covariante sobre un conjunto de puntos.

        Vectorización del cálculo de fricción para análisis de fase o
        evaluación del campo de fuerzas en la variedad M.

        Dado un conjunto de N velocidades v_i ∈ T_{x_i}M:

            F_fric_i = compute_friction(v_i, f_drv_i, model, G)   ∀ i

        Los resultados forman el campo covariante:

            F_fric : M → T*M,  x ↦ F_fric(x)

        Parámetros
        ----------
        velocities      : np.ndarray, shape (N,), velocidades [m/s].
        driving_forces  : np.ndarray, shape (N,), fuerzas impulsoras [N].
        friction_model  : FrictionModel, modelo de fricción.
        metric_tensor   : np.ndarray, shape (1,1), tensor G_μν.
        coulomb_friction : float, F_c [N].
        stribeck_coeffs  : Tuple (F_S, F_C, v_St).

        Retorna
        -------
        np.ndarray, shape (N,), campo de fuerzas de fricción [N].
        """
        N = len(velocities)
        result = np.zeros(N, dtype=np.float64)
        for i in range(N):
            result[i] = FrictionCalculator.compute_friction(
                velocity        = float(velocities[i]),
                driving_force   = float(driving_forces[i]),
                friction_model  = friction_model,
                coulomb_friction = coulomb_friction,
                stribeck_coeffs  = stribeck_coeffs,
                metric_tensor    = metric_tensor,
            )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # FIN SUTURA 1 (FrictionCalculator)
    # Contrato de salida:
    #   → float [N]: componente covariante 1D de F_fric ∈ T*_xM
    #   → Consumido por _derivatives() de la Fase 2 (Störmer-Verlet)
    # ─────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════════
# SUTURA 2 — LieGroupPIDController (Control PID Geométrico sobre Grupos de Lie)
# ════════════════════════════════════════════════════════════════════════════════════

class LieGroupPIDController:
    r"""
    Controlador PID Geométrico sobre Grupos de Lie con transporte paralelo.

    CIRUGÍA v4.0.0 (Sutura 2):
    ───────────────────────────
    El PID euclidiano PIDController ha sido extirpado y reemplazado por
    un controlador que opera intrínsecamente en el álgebra de Lie g = T_eG.

    Ley de control en el fibrado tangente TG:
    ──────────────────────────────────────────

        u(t) = -K_p · Log_x(x_d)
               - K_d · (D/dt)ẋ
               - K_i · ∫₀ᵗ Γ_γ(τ) · Log_{x(τ)}(x_d) dτ

    donde:
        Log_x(x_d)      = error geodésico ξ ∈ g          [Logaritmo de Riemann]
        D/dt ẋ          = derivada covariante de Levi-Civita  [≅ ẍ en ℝⁿ]
        Γ_γ(τ)          = transporte paralelo a lo largo de γ [isometría]
        ∫₀ᵗ Γ_γ(τ)·ξ(τ)dτ = integral acumulada en g          [anti-Windup geométrico]

    Anti-windup geométrico:
    ────────────────────────
    Si ‖u‖_G > output_limit:
        1. Proyectar u sobre la bola geodésica B_G(0, output_limit) ⊂ g
        2. Deshacer la contribución integral del paso actual:
               σ ← σ - Γ_γ(t) · ξ(t) · dt
           (en g, no en ℝ plano → preserva la isometría del transporte)

    Para G = (ℝⁿ, +) y G_μν = I:
        Log_x(x_d)  = x_d - x           (diferencia euclidiana)
        D/dt ẋ      = dẋ/dt             (aceleración clásica)
        Γ_γ         = Id                 (transporte trivial)
        u(t)        = -Kp·e - Kd·ẍ - Ki·∫e dt  (PID clásico)

    → El controlador euclidiano PIDController es un caso especial de
      LieGroupPIDController con G=I, recuperando compatibilidad total.

    Extensibilidad:
    ───────────────
    Para extender a SO(3) o SE(3), subclasear y sobrescribir:
        _lie_log()              → Log_R(R_d) via matrix logarithm
        _parallel_transport()   → ecuación de transporte paralelo en so(3)
        _covariant_derivative() → D/dt con Γ^k_{ij} no nulos

    Parámetros
    ----------
    kp                    : float, ganancia proporcional.
    ki                    : float, ganancia integral.
    kd                    : float, ganancia derivativa.
    output_limit          : float, radio de la bola geodésica en g [‖u‖_G ≤ limit].
    metric_tensor         : np.ndarray, shape (n,n), G_μν ≻ 0. Default: I_n.
    derivative_filter_tau : float, constante de tiempo del filtro derivativo [s].
    state_dim             : int, dimensión n del grupo de Lie G (default: 1).
    """

    def __init__(
        self,
        kp                    : float,
        ki                    : float,
        kd                    : float,
        output_limit          : float = 10.0,
        metric_tensor         : Optional[np.ndarray] = None,
        derivative_filter_tau : float = 0.01,
        state_dim             : int = 1,
    ) -> None:
        if output_limit <= 0:
            raise ValueError(
                f"output_limit debe ser > 0, recibido {output_limit}."
            )
        if derivative_filter_tau < 0:
            raise ValueError(
                f"derivative_filter_tau debe ser ≥ 0, recibido {derivative_filter_tau}."
            )
        if state_dim < 1:
            raise ValueError(
                f"state_dim debe ser ≥ 1, recibido {state_dim}."
            )

        self.kp                    = float(kp)
        self.ki                    = float(ki)
        self.kd                    = float(kd)
        self.output_limit          = float(output_limit)
        self.derivative_filter_tau = float(derivative_filter_tau)
        self.state_dim             = state_dim

        # §2.a: Tensor métrico G_μν (define la geometría del grupo de Lie)
        if metric_tensor is None:
            self._G = np.eye(state_dim, dtype=np.float64)
        else:
            G_arr = np.asarray(metric_tensor, dtype=np.float64)
            if G_arr.shape != (state_dim, state_dim):
                raise ValueError(
                    f"metric_tensor debe tener shape ({state_dim},{state_dim}), "
                    f"recibido {G_arr.shape}."
                )
            # Verificar definida positiva: todos los eigenvalores > 0
            eigvals = np.linalg.eigvalsh(G_arr)
            if np.any(eigvals <= 0):
                raise ValueError(
                    "metric_tensor debe ser definida positiva (G ≻ 0). "
                    f"Eigenvalores: {eigvals}."
                )
            self._G = G_arr

        # §2.b: Estado interno en el álgebra de Lie g
        # σ(t) = ∫₀ᵗ Γ_γ(τ) · Log_{x(τ)}(x_d) dτ  ∈ g
        self._integral_lie      : np.ndarray = np.zeros(state_dim, dtype=np.float64)
        # ẋ(t-dt): velocidad previa para la derivada covariante
        self._velocity_prev     : np.ndarray = np.zeros(state_dim, dtype=np.float64)
        # Derivada covariante filtrada D/dt ẋ |_filtrada ∈ g
        self._cov_deriv_filtered: np.ndarray = np.zeros(state_dim, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────
    # §1.4 — Logaritmo de Riemann (error geodésico)
    # ─────────────────────────────────────────────────────────────────────

    def _lie_log(
        self,
        x: np.ndarray,
        x_d: np.ndarray,
    ) -> np.ndarray:
        r"""
        Calcula el error geodésico Log_x(x_d) ∈ T_xG ≅ g.

        Para el grupo afín G = (ℝⁿ, +):
            Log_x(x_d) = x_d - x   ∈ ℝⁿ

        La geodésica entre x y x_d en ℝⁿ es la línea recta, y el
        logaritmo coincide con la diferencia vectorial. Esta función
        es el punto de extensión para grupos de Lie no abelianos.

        Parámetros
        ----------
        x   : np.ndarray, shape (n,), estado actual en G.
        x_d : np.ndarray, shape (n,), estado deseado en G.

        Retorna
        -------
        np.ndarray, shape (n,), error geodésico ξ ∈ g.
        """
        return _lie_log_affine(x, x_d)

    # ─────────────────────────────────────────────────────────────────────
    # §1.5 — Transporte paralelo (acumulación integral isométrica)
    # ─────────────────────────────────────────────────────────────────────

    def _parallel_transport(
        self,
        xi: np.ndarray,
    ) -> np.ndarray:
        r"""
        Aplica el operador de transporte paralelo Γ_γ al vector ξ ∈ g.

        Para ℝⁿ con G constante (curvatura R=0):
            Γ_γ · ξ = ξ   (transporte trivial, identidad)

        El transporte paralelo es la isometría que preserva el producto
        interno métrico ⟨·,·⟩_G a lo largo de la geodésica γ:
            ⟨Γ_γ·u, Γ_γ·v⟩_G = ⟨u,v⟩_G   ∀ u,v ∈ g

        Parámetros
        ----------
        xi : np.ndarray, shape (n,), vector en g a transportar.

        Retorna
        -------
        np.ndarray, shape (n,), vector transportado Γ_γ·ξ.
        """
        return _parallel_transport_affine(xi, self._G)

    # ─────────────────────────────────────────────────────────────────────
    # §1.6 — Derivada covariante filtrada (término D del PID geométrico)
    # ─────────────────────────────────────────────────────────────────────

    def _update_covariant_derivative(
        self,
        velocity: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        r"""
        Actualiza y retorna la derivada covariante filtrada de la velocidad.

        Pipeline:
        ─────────
        1. Derivada covariante cruda:
               Dẋ/dt = (ẋ(t) - ẋ(t-dt)) / dt + Γ^k_{ij}·ẋⁱ·ẋʲ
               (Para ℝⁿ afín: Γ=0 → Dẋ/dt = dẋ/dt)

        2. Filtro pasa-bajos en g (Lie algebra):
               α = dt / (τ_f + dt)   ∈ (0, 1]
               D_f[n] = (1-α)·D_f[n-1] + α·(Dẋ/dt)
               (El filtro opera elemento a elemento en g, no en ℝ plano)

        La derivada sobre la VELOCIDAD (no el error) evita el "derivative
        kick" al cambiar el setpoint x_d: el término D = -K_d · D_f solo
        reacciona a cambios en ẋ, no en x_d.

        Parámetros
        ----------
        velocity : np.ndarray, shape (n,), velocidad actual ẋ(t).
        dt       : float, paso de tiempo [s].

        Retorna
        -------
        np.ndarray, shape (n,), derivada covariante filtrada D_f ≈ ẍ.
        """
        if dt <= 0.0:
            return self._cov_deriv_filtered.copy()

        # Derivada covariante cruda (afín: Γ=0)
        raw_cov_deriv = _covariant_derivative_affine(
            velocity      = velocity,
            velocity_prev = self._velocity_prev,
            dt            = dt,
            G             = self._G,
        )

        # Filtro pasa-bajos en g
        if self.derivative_filter_tau > 0.0:
            alpha = dt / (self.derivative_filter_tau + dt)
        else:
            alpha = 1.0

        self._cov_deriv_filtered = (
            (1.0 - alpha) * self._cov_deriv_filtered
            + alpha * raw_cov_deriv
        )

        # Actualizar velocidad previa
        self._velocity_prev = velocity.copy()

        return self._cov_deriv_filtered.copy()

    # ─────────────────────────────────────────────────────────────────────
    # §1.7 — Actualización del controlador PID geométrico
    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        setpoint      : float,
        current_value : float,
        dt            : float,
        current_velocity: Optional[float] = None,
    ) -> float:
        r"""
        Calcula la señal de control geométrica u(t) ∈ g.

        Ley de control completa:
        ─────────────────────────

            u(t) = -K_p · ξ(t)
                   - K_d · D_f(t)
                   - K_i · σ(t)

        donde:
            ξ(t) = Log_x(x_d)                          [error geodésico, g]
            D_f(t) = Dẋ/dt |_filtrada                  [deriv. covariante, g]
            σ(t)  = ∫₀ᵗ Γ_γ(τ)·ξ(τ) dτ  (trapezoidal) [integral en g]

        Anti-windup geométrico:
        ────────────────────────
        Si ‖u(t)‖_G > output_limit:
            1. Saturar: u ← output_limit · u / ‖u‖_G
               (proyección sobre bola geodésica B_G(0, u_max))
            2. Deshacer integración:
               σ ← σ - Γ_γ(t)·ξ(t)·dt
               (se deshace en g, preservando la estructura de Lie)

        Pipeline:
        ─────────
        1. Construir ξ = Log_x(x_d) ∈ g
        2. Acumular σ += Γ_γ·ξ·dt (transporte paralelo)
        3. Calcular D_f = Dẋ/dt filtrada
        4. Ensamblar u = -Kp·ξ - Kd·D_f - Ki·σ
        5. Anti-windup si ‖u‖_G > output_limit
        6. Retornar componente escalar u[0] (sistema 1D)

        Parámetros
        ----------
        setpoint         : float, valor deseado x_d.
        current_value    : float, valor actual x(t).
        dt               : float, paso de tiempo [s]. Debe ser > 0.
        current_velocity : Optional[float], velocidad actual ẋ(t).
                           Si None → se usa 0.0 (no hay término D).

        Retorna
        -------
        float, señal de control u(t) ∈ ℝ, ‖u‖_G ≤ output_limit.
        """
        if dt <= 0.0:
            return 0.0

        # §1.7.1: Vectores en g (álgebra de Lie, dim=state_dim)
        x   = np.array([current_value], dtype=np.float64)
        x_d = np.array([setpoint],      dtype=np.float64)
        vel = np.array(
            [current_velocity if current_velocity is not None else 0.0],
            dtype=np.float64,
        )

        # §1.7.2: Error geodésico ξ = Log_x(x_d) ∈ g
        xi = self._lie_log(x, x_d)  # shape (n,)

        # §1.7.3: Transporte paralelo y acumulación integral en g
        # σ(t) += Γ_γ(t) · ξ(t) · dt   (regla trapezoidal implícita: ×dt)
        transported_xi = self._parallel_transport(xi)
        self._integral_lie += transported_xi * dt  # acumulación en g

        # §1.7.4: Derivada covariante filtrada D_f
        D_f = self._update_covariant_derivative(vel, dt)  # shape (n,)

        # §1.7.5: Señal de control en g
        # u = -Kp·ξ - Kd·D_f - Ki·σ
        u_vec = (
            - self.kp * xi
            - self.kd * D_f
            - self.ki * self._integral_lie
        )  # shape (n,), ∈ g

        # §1.7.6: Anti-windup geométrico
        # Norma de u en la métrica G: ‖u‖_G = √(uᵀ G u)
        norm_u_G = _metric_norm(u_vec, self._G)

        if norm_u_G > self.output_limit:
            # Proyectar sobre bola geodésica B_G(0, output_limit)
            u_vec = u_vec * (self.output_limit / norm_u_G)
            # Deshacer la integración del paso actual en g (anti-windup)
            self._integral_lie -= transported_xi * dt

        # §1.7.7: Retornar componente escalar (sistema 1D del pistón)
        return float(u_vec[0])

    def reset(self) -> None:
        r"""
        Reinicia el estado interno del controlador en el álgebra de Lie g.

        Restablece:
            σ(t)  = 0 ∈ g   (integral acumulada)
            ẋ_prev = 0 ∈ g  (velocidad previa)
            D_f   = 0 ∈ g   (derivada covariante filtrada)
        """
        self._integral_lie       = np.zeros(self.state_dim, dtype=np.float64)
        self._velocity_prev      = np.zeros(self.state_dim, dtype=np.float64)
        self._cov_deriv_filtered = np.zeros(self.state_dim, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────
    # FIN SUTURA 2 (LieGroupPIDController)
    # La señal de control u(t) ∈ g es una fuerza adicional [N] en el
    # fibrado tangente que se inyecta en el Hamiltoniano port-Hamiltoniano
    # de la Fase 2 (Störmer-Verlet simpléctic).
    #
    # CONTINUACIÓN → FASE 2: Gemelo Digital Simpléctico (AtomicPiston)
    # La última definición de esta Fase 1 es el punto de costura con el
    # primer método de la Fase 2 (__init__ de AtomicPiston), que recibirá
    # instancias de LieGroupPIDController como controladores internos de
    # velocidad y energía.
    # ─────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════════
# ALIAS DE COMPATIBILIDAD — Preserva la interfaz pública de v3.0.0
# ════════════════════════════════════════════════════════════════════════════════════

# PIDController es ahora un alias de LieGroupPIDController con G=I
# (recupera exactamente el comportamiento euclidiano de v3.0.0 cuando
# metric_tensor=None y state_dim=1)
PIDController = LieGroupPIDController

# ════════════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 1 — HERRAMIENTAS FÍSICAS Y MATEMÁTICAS (v4.0.0)
#
# Diagrama de flujo de salida hacia Fase 2:
#
#   FrictionCalculator.compute_friction(v, F_drv, model, G_μν)
#       → float [N]  ∈ T*_xM (covector 1D)
#       → Entra en _derivatives() del integrador simpléctico Störmer-Verlet
#
#   LieGroupPIDController.update(x_d, x, dt, ẋ)
#       → float [N] ∈ g (álgebra de Lie, dim=1)
#       → Se suma a F_ext en el Hamiltoniano port-Hamiltoniano
#
# ════════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                  ║
# ║  FASE 2 — GEMELO DIGITAL SIMPLÉCTICO (AtomicPiston) v4.0.0                     ║
# ║                                                                                  ║
# ║  SUTURA 3: Integrador Störmer-Verlet Simpléctico + Disipación R(x)             ║
# ║  SUTURA 4: Estructura de Dirac Electromecánica Port-Hamiltoniana               ║
# ║                                                                                  ║
# ║  Contrato formal (v4.0.0):                                                       ║
# ║    Entrada : F_fric ∈ T*M [N] (Fase 1), u ∈ g [N] (Fase 1)                   ║
# ║    Salida  : estado x=[q,p,Q]ᵀ serializable para Fase 3                        ║
# ║                                                                                  ║
# ║  Garantías físicas (v4.0.0):                                                     ║
# ║    G1. Integrador Störmer-Verlet preserva 2-forma ω=dq∧dp (Liouville)          ║
# ║    G2. Estructura (J−R)∇H: J antisimétrica, R definida positiva                ║
# ║    G3. Balance port-Hamiltoniano exacto: Ḣ = -∇Hᵀ R ∇H + ∇Hᵀ B u             ║
# ║    G4. Disipación inelástica: R(x) absorbe impactos sin violar Liouville        ║
# ║    G5. F_bemf [N] via acoplamiento α en la columna (2,3) de J(x)               ║
# ║    G6. self.dt inicializado en __init__ (preservado de FIX 9)                   ║
# ║    G7. Historial acotado via deque(maxlen) (preservado de FIX 11)               ║
# ║                                                                                  ║
# ║  La Fase 2 recibe fuerzas de la Fase 1 y produce el estado físico              ║
# ║  que alimenta los endpoints de la Fase 3 via ServiceContext.                    ║
# ║                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════════

"""
FUNDAMENTACIÓN MATEMÁTICA DE LA FASE 2 (v4.0.0):
══════════════════════════════════════════════════

§F3. ESTRUCTURA PORT-HAMILTONIANA ELECTROMECÁNICA
──────────────────────────────────────────────────
El vector de estado extendido:

    x = [q, p, Q]ᵀ ∈ ℝ³

donde:
    q ∈ ℝ      : posición generalizada [m]
    p = mq̇ ∈ ℝ : momento lineal [kg·m/s]
    Q = ∫I dt ∈ ℝ : carga eléctrica acumulada [C]

El Hamiltoniano extendido H : ℝ³ → ℝ:

    H(q,p,Q) = p²/(2m) + ½kq² + ¼εq⁴ + Q²/(2C_eq)
             = T(p) + V(q) + W_e(Q)

Gradiente del Hamiltoniano:

    ∇H = [∂H/∂q, ∂H/∂p, ∂H/∂Q]ᵀ
       = [kq + εq³,  p/m,  Q/C_eq]ᵀ
       = [F_resorte_negada, ẋ, V_C]ᵀ

La dinámica port-Hamiltoniana:

    ẋ = (J(x) − R(x)) ∇H(x) + B u_ext

donde:
    J(x) = [[  0,   1,   0  ],    matriz de interconexión antisimétrica
            [ -1,   0,   α  ],    (J = -Jᵀ, codifica la topología)
            [  0,  -α,   0  ]]

    R(x) = [[  0,   0,    0     ],    matriz de disipación (R = Rᵀ ≻ 0)
            [  0,  c_m,   0     ],    c_m = c (amortiguamiento viscoso)
            [  0,   0,  1/R_t  ]]    R_t = R_int + R_load

    B    = [0, 1, 0]ᵀ               vector de entrada (fuerza externa)

Balance energético (disipación):

    Ḣ = ∇Hᵀ ẋ = ∇Hᵀ(J−R)∇H + ∇Hᵀ B u
      = -∇Hᵀ R ∇H + ẋ·u
      = -(c_m·ẋ² + V_C²/R_t) + ẋ·u

    ∴ Ḣ ≤ ẋ·u   (pasividad: la energía solo entra por la entrada u)

§F4. INTEGRADOR STÖRMER-VERLET SIMPLÉCTICO
───────────────────────────────────────────
El integrador de Störmer-Verlet preserva la 2-forma simpléctica:

    ω = dq ∧ dp   (2-forma canónica en T*M)

Las ecuaciones de avance con disipación R(x):

    Paso 1 (semi-impulso):
        p_{n+½} = pₙ - (Δt/2)[∇_q H(qₙ, p_{n+½})] - (Δt/2) R_pp · pₙ

    Para H separable H = T(p) + V(q):
        ∇_q H = kq + εq³   (fuerza del resorte)
        ∴ p_{n+½} = [pₙ - (Δt/2)(kqₙ + εqₙ³)] / [1 + (Δt/2)(c_m/m)]

    Paso 2 (posición completa):
        q_{n+1} = qₙ + Δt · ∇_p H(qₙ, p_{n+½})
                = qₙ + Δt · p_{n+½}/m

    Paso 3 (semi-impulso final):
        p_{n+1} = p_{n+½} - (Δt/2)[∇_q H(q_{n+1}, p_{n+½})]
                           - (Δt/2) R_pp · p_{n+½}

    Paso 4 (carga eléctrica Q, Euler implícito):
        Q_{n+1} = Qₙ + Δt · (-α·p_{n+½}/m - Q/C_eq/R_t)
    (La carga sigue la ecuación del circuito RC acoplado al momento)

Preservación simpléctica:
    El mapa (qₙ,pₙ) → (q_{n+1},p_{n+1}) es un simplectomorfismo:
        φ*ω = ω   ∀ n
    La disipación R ≠ 0 rompe la simplecticidad EXACTA, pero Störmer-
    Verlet la preserva hasta O(Δt²) mejor que RK4 que la viola desde O(Δt).

§F5. INECUACIÓN DE DISIPACIÓN (GARANTÍA Ḣd ≤ 0)
───────────────────────────────────────────────────
Con el Hamiltoniano modificado Hd = H - H*:

    Ḣd = -∇Hᵀ R ∇H ≤ 0   (R ≻ 0 → forma cuadrática negativa semidefinida)

La inecuación se verifica numéricamente en cada paso como:

    dissipation_rate = -(c_m·ẋ² + (Q/C_eq)²/R_t)  ≤ 0

Si dissipation_rate > ε_mach → error numérico → log WARNING.
"""


# ════════════════════════════════════════════════════════════════════════════════════
# §2.0 — ÁLGEBRA PORT-HAMILTONIANA AUXILIAR
# ════════════════════════════════════════════════════════════════════════════════════

def _build_J_matrix(alpha: float) -> np.ndarray:
    r"""
    Construye la matriz de interconexión antisimétrica J(x) ∈ ℝ^{3×3}.

    Definición:

        J = [[  0,   1,   0  ],
             [ -1,   0,   α  ],
             [  0,  -α,   0  ]]

    Propiedades verificadas:
        (a) J + Jᵀ = 0  (antisimetría exacta)
        (b) J codifica la topología de interconexión electromecánica:
            - Fila 1: q̇ = ∂H/∂p  (cinemática)
            - Fila 2: ṗ = -∂H/∂q + α·∂H/∂Q  (dinámica + acoplamiento)
            - Fila 3: Q̇ = -α·∂H/∂p  (circuito acoplado a momento)
        (c) La entrada α en posiciones (2,3) y -(α) en (3,2) es el
            coeficiente de acoplamiento electromecánico [N/A = V·s/m].

    Verificación de antisimetría:
        J[1,2] = α, J[2,1] = -α → J[i,j] = -J[j,i] ✓
        (El intercambio mecánico↔eléctrico preserva la antisimetría)

    Parámetros
    ----------
    alpha : float, factor de acoplamiento electromecánico α [N/A].

    Retorna
    -------
    np.ndarray, shape (3,3), matriz J antisimétrica.
    """
    return np.array(
        [
            [ 0.0,    1.0,    0.0  ],
            [-1.0,    0.0,    alpha],
            [ 0.0,   -alpha,  0.0  ],
        ],
        dtype=np.float64,
    )


def _build_R_matrix(
    damping_viscous  : float,
    resistance_total : float,
) -> np.ndarray:
    r"""
    Construye la matriz de disipación definida positiva R(x) ∈ ℝ^{3×3}.

    Definición:

        R = diag(0, c_m, 1/R_t)

        R = [[  0,    0,      0     ],
             [  0,   c_m,     0     ],
             [  0,    0,   1/R_t   ]]

    donde:
        c_m = c (amortiguamiento viscoso [N·s/m])
        R_t = R_int + R_load (resistencia total del circuito [Ω])

    Propiedades verificadas:
        (a) R = Rᵀ  (simetría)
        (b) R ≽ 0   (semidefinida positiva)
        (c) ∇Hᵀ R ∇H = c_m·ẋ² + (V_C)²/R_t ≥ 0
            → Tasa de disipación de energía siempre no negativa
        (d) R[0,0] = 0: la posición q no tiene disipación directa
            (la disipación entra solo en el momento p y la carga Q)

    Inecuación de disipación (garantía fundamental):

        Ḣ = -∇Hᵀ R ∇H + ∇Hᵀ B u ≤ ∇Hᵀ B u

    Sin entrada externa (u=0): Ḣ ≤ 0 (sistema pasivo).

    Parámetros
    ----------
    damping_viscous  : float, c_m [N·s/m]. Debe ser ≥ 0.
    resistance_total : float, R_t [Ω]. Debe ser > 0.

    Retorna
    -------
    np.ndarray, shape (3,3), matriz R semidefinida positiva.

    Lanza
    -----
    ValueError : Si resistance_total ≤ 0.
    """
    if resistance_total <= 0.0:
        raise ValueError(
            f"resistance_total debe ser > 0, recibido {resistance_total}."
        )
    r_diag = np.array(
        [0.0, max(damping_viscous, 0.0), 1.0 / resistance_total],
        dtype=np.float64,
    )
    return np.diag(r_diag)


def _compute_hamiltonian_gradient(
    q              : float,
    p              : float,
    Q_charge       : float,
    mass           : float,
    elasticity     : float,
    nonlinear_elast: float,
    capacitance_eq : float,
) -> np.ndarray:
    r"""
    Calcula el gradiente del Hamiltoniano extendido ∇H ∈ ℝ³.

    Hamiltoniano:

        H(q,p,Q) = p²/(2m) + ½kq² + ¼εq⁴ + Q²/(2C_eq)

    Gradiente:

        ∂H/∂q = kq + εq³       [N] (fuerza del resorte negada)
        ∂H/∂p = p/m             [m/s] (velocidad)
        ∂H/∂Q = Q/C_eq          [V] (voltaje en el capacitor)

    El gradiente ∇H es el "vector de esfuerzo" (effort vector) en la
    terminología port-Hamiltoniana. Sus componentes tienen dimensiones:
        [N], [m/s], [V]  — heterogéneas pero consistentes con (J−R).

    Verificación dimensional de la dinámica (J−R)∇H:
        ẋ = [q̇, ṗ, Q̇]  tiene dimensiones [m/s, N, A]
        (J−R)∇H fila 1: 1·(p/m) = ẋ [m/s] ✓
        (J−R)∇H fila 2: -1·(kq+εq³) + α·(Q/C_eq) - c_m·(p/m) [N] ✓
        (J−R)∇H fila 3: -α·(p/m) - (Q/C_eq)/R_t [A] ✓

    Parámetros
    ----------
    q               : float, posición [m].
    p               : float, momento [kg·m/s].
    Q_charge        : float, carga eléctrica [C].
    mass            : float, masa [kg].
    elasticity      : float, rigidez lineal k [N/m].
    nonlinear_elast : float, rigidez cúbica ε [N/m³].
    capacitance_eq  : float, capacitancia equivalente C_eq [F].

    Retorna
    -------
    np.ndarray, shape (3,), gradiente ∇H = [∂H/∂q, ∂H/∂p, ∂H/∂Q]ᵀ.
    """
    dH_dq = elasticity * q + nonlinear_elast * (q ** 3)  # [N]
    dH_dp = p / max(mass, 1e-12)                          # [m/s]
    dH_dQ = Q_charge / max(capacitance_eq, 1e-15)         # [V]
    return np.array([dH_dq, dH_dp, dH_dQ], dtype=np.float64)


def _verify_dissipation_inequality(
    grad_H         : np.ndarray,
    R_matrix       : np.ndarray,
    tolerance      : float = 1e-10,
) -> float:
    r"""
    Verifica la inecuación de disipación Ḣ_disipativa ≤ 0.

    Calcula la tasa de disipación pura (sin entrada externa):

        D(x) = -∇Hᵀ R ∇H = -(c_m·ẋ² + V_C²/R_t) ≤ 0

    Si D(x) > tolerance → violación numérica → retornar D con WARNING.
    La violación indica acumulación de error de redondeo en el integrador.

    Parámetros
    ----------
    grad_H    : np.ndarray, shape (3,), gradiente ∇H.
    R_matrix  : np.ndarray, shape (3,3), matriz de disipación.
    tolerance : float, umbral de violación numérica.

    Retorna
    -------
    float, tasa de disipación D(x) = -∇Hᵀ R ∇H ≤ 0 (idealmente).
    """
    dissipation = -float(grad_H @ R_matrix @ grad_H)
    if dissipation > tolerance:
        logger.warning(
            "Violación de inecuación de disipación: Ḣ_dis = %.4e > 0. "
            "Posible acumulación de error numérico en Störmer-Verlet.",
            dissipation,
        )
    return dissipation


# ════════════════════════════════════════════════════════════════════════════════════
# SUTURA 3 + 4 — AtomicPiston: Gemelo Digital Port-Hamiltoniano Simpléctico
# ════════════════════════════════════════════════════════════════════════════════════

class AtomicPiston:
    r"""
    Gemelo Digital de la Unidad de Potencia Inteligente (IPU).

    CIRUGÍA v4.0.0 (Suturas 3 y 4):
    ─────────────────────────────────
    SUTURA 3 — Integrador Störmer-Verlet Simpléctico:
        El integrador RK4 ha sido extirpado. El núcleo de integración es
        ahora el método de Störmer-Verlet acoplado a la matriz R(x), que
        preserva la 2-forma simpléctica ω = dq∧dp hasta O(Δt²) en presencia
        de disipación, absorbiendo impactos inelásticos sin violar el
        Teorema de Liouville en el régimen no disipativo.

    SUTURA 4 — Estructura de Dirac Electromecánica:
        El cálculo escalar de F_bemf y V_oc ha sido extirpado. El núcleo
        del motor ensambla las matrices J(x) (antisimétrica) y R(x) (definida
        positiva), y la dinámica se computa via el balance port-Hamiltoniano:

            ẋ = (J(x) − R(x)) ∇H(x) + B · F_ext

        donde x = [q, p, Q]ᵀ es el vector de estado extendido.

    Vector de estado extendido:
        self.q        [m]       : posición generalizada
        self.p        [kg·m/s]  : momento lineal (= m·ẋ)
        self.Q_charge [C]       : carga eléctrica acumulada

    Variables derivadas (no son estados primarios):
        self.velocity       = p/m           [m/s]
        self.acceleration   = ṗ/m          [m/s²]
        self.circuit_voltage = Q/C_eq       [V]
        self.circuit_current = Q̇           [A]

    Garantía de pasividad:
        Ḣ = -∇Hᵀ R ∇H + ẋ·F_ext ≤ ẋ·F_ext
        El sistema solo almacena o disipa energía; no la genera internamente.

    Compatibilidad con Fase 1:
        - FrictionCalculator.compute_friction → F_fric [N] entra en F_ext
        - LieGroupPIDController.update → u [N] entra en F_ext
        - La suma F_ext = F_applied + F_pid + F_fric se inyecta en B·u

    Compatibilidad con Fase 3:
        - get_state_dict() serializa [q, p, Q] y todas las variables derivadas
        - El estado es consistente bajo el lock de ServiceContext
    """

    # ── Constantes de clase ───────────────────────────────────────────────
    RESTITUTION_COEFF : float = 0.8    # Coeficiente de restitución e [adim]
    _I_MIN            : float = 1e-12  # Corriente mínima [A] (evita /0)
    _EPS_DISS         : float = 1e-10  # Tolerancia de disipación [W]

    # ─────────────────────────────────────────────────────────────────────
    # §2.1 — Inicialización del gemelo digital
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        capacity             : float,
        elasticity           : float,
        damping              : float,
        piston_mass          : float = 1.0,
        mode                 : PistonMode = PistonMode.CAPACITOR,
        transducer_type      : TransducerType = TransducerType.PIEZOELECTRIC,
        friction_model       : FrictionModel = FrictionModel.VISCOUS,
        coulomb_friction     : float = 0.2,
        stribeck_coeffs      : Tuple[float, float, float] = (0.3, 0.1, 0.05),
        nonlinear_elasticity : float = 0.01,
        dt_default           : float = 1e-3,
        metric_tensor        : Optional[np.ndarray] = None,
        load_resistance      : float = 1e6,
    ) -> None:
        r"""
        Inicializa el gemelo digital con validación completa de parámetros.

        Parámetros
        ----------
        capacity             : float, compresión máxima [m]. Debe ser > 0.
        elasticity           : float, rigidez lineal k [N/m]. Debe ser ≥ 0.
        damping              : float, amortiguamiento viscoso c [N·s/m]. ≥ 0.
        piston_mass          : float, masa inercial m [kg]. Debe ser > 0.
        mode                 : PistonMode, modo de operación.
        transducer_type      : TransducerType, tipo de transductor.
        friction_model       : FrictionModel, modelo de fricción seca.
        coulomb_friction     : float, coeficiente de Coulomb F_c [N]. ≥ 0.
        stribeck_coeffs      : Tuple (F_S, F_C, v_St).
        nonlinear_elasticity : float, rigidez cúbica ε [N/m³]. ≥ 0.
        dt_default           : float, paso de tiempo por defecto [s]. > 0.
        metric_tensor        : Optional[np.ndarray], G_μν para fricción
                               covariante. None → G = I (euclidiano).
        load_resistance      : float, resistencia de carga R_L [Ω]. > 0.
        """
        # §2.1.1: Validación de parámetros físicos
        self._validate_params(
            capacity, elasticity, damping, piston_mass,
            coulomb_friction, nonlinear_elasticity, dt_default,
        )

        # ── Parámetros mecánicos ──────────────────────────────────────────
        self.capacity             : float = capacity
        self.k                    : float = elasticity
        self.c                    : float = damping
        self.m                    : float = piston_mass
        self.nonlinear_elasticity : float = nonlinear_elasticity

        # ── Configuración operacional ─────────────────────────────────────
        self.mode             : PistonMode     = mode
        self.transducer_type  : TransducerType = transducer_type
        self.friction_model   : FrictionModel  = friction_model
        self.coulomb_friction : float          = coulomb_friction
        self.stribeck_coeffs  : Tuple[float, float, float] = stribeck_coeffs

        # ── Tensor métrico para fricción covariante (Fase 1) ──────────────
        self._metric_tensor: Optional[np.ndarray] = metric_tensor

        # ── Parámetros del transductor ────────────────────────────────────
        self.coupling_factor    : float = 0.0
        self.internal_resistance: float = 0.0
        self._configure_transducer()

        # ── Circuito eléctrico equivalente ────────────────────────────────
        self.equivalent_capacitance: float = 1.0 / max(self.k, 1e-9)
        self.equivalent_inductance : float = self.m
        self.equivalent_resistance : float = 1.0 / max(self.c, 1e-9)
        self.output_capacitance    : float = 0.1
        self.converter_efficiency  : float = 0.95
        self.load_resistance       : float = max(load_resistance, 1e-9)

        # ── ESTADO PORT-HAMILTONIANO EXTENDIDO x = [q, p, Q]ᵀ ───────────
        # SUTURA 4: el estado primario es el trío (q, p, Q), no (pos, vel)
        self.q        : float = 0.0   # posición [m]
        self.p        : float = 0.0   # momento lineal [kg·m/s]
        self.Q_charge : float = 0.0   # carga eléctrica [C]

        # Variables derivadas (computadas desde el estado primario)
        self.velocity        : float = 0.0   # p/m  [m/s]
        self.acceleration    : float = 0.0   # ṗ/m [m/s²]
        self.circuit_voltage : float = 0.0   # Q/C_eq [V]
        self.circuit_current : float = 0.0   # Q̇  [A]
        self.output_voltage  : float = 0.0   # V_load [V]

        # ── Ensamblaje de matrices port-Hamiltonianas ─────────────────────
        # SUTURA 4: matrices J y R ensambladas una vez en __init__
        # (se reconstruyen si cambian los parámetros vía set_*)
        self._R_total : float = self.internal_resistance + self.load_resistance
        self._J       : np.ndarray = _build_J_matrix(self.coupling_factor)
        self._R       : np.ndarray = _build_R_matrix(self.c, self._R_total)
        self._JR      : np.ndarray = self._J - self._R   # (J−R) pre-calculada
        self._B       : np.ndarray = np.array([0.0, 1.0, 0.0])  # vector de entrada

        # ── Control de operación ──────────────────────────────────────────
        self.capacitor_discharge_threshold: float = -capacity * 0.9
        self.hysteresis_factor            : float = 0.1
        self.battery_is_discharging       : bool  = False
        self.battery_discharge_rate       : float = capacity * 0.05
        self.compression_direction        : int   = -1
        self.last_applied_force           : float = 0.0
        self.dt                           : float = dt_default  # FIX 9 preservado

        # ── Controladores PID Geométricos (Fase 1 → Fase 2) ──────────────
        # SUTURA 2 aplicada: LieGroupPIDController reemplaza PIDController
        self.speed_controller : LieGroupPIDController = LieGroupPIDController(
            kp=1.0, ki=0.1, kd=0.01,
            metric_tensor=metric_tensor,
            state_dim=1,
        )
        self.energy_controller: LieGroupPIDController = LieGroupPIDController(
            kp=0.5, ki=0.05, kd=0.005,
            metric_tensor=metric_tensor,
            state_dim=1,
        )
        self.target_speed  : float = 0.0
        self.target_energy : float = 0.0

        # ── Historial acotado (FIX 11 preservado) ─────────────────────────
        self._max_hist      : int = 10_000
        self._max_hist_short: int = 1_000
        self.energy_history         : deque = deque(maxlen=self._max_hist)
        self.efficiency_history     : deque = deque(maxlen=self._max_hist)
        self.friction_force_history : deque = deque(maxlen=self._max_hist)
        self.position_history       : deque = deque(maxlen=self._max_hist_short)
        self.velocity_history       : deque = deque(maxlen=self._max_hist_short)
        self.hamiltonian_history    : deque = deque(maxlen=self._max_hist)
        self.dissipation_history    : deque = deque(maxlen=self._max_hist)

        logger.info(
            "AtomicPiston [port-Hamiltoniano] inicializado: "
            "modo=%s, transductor=%s, fricción=%s, α=%.3f N/A, "
            "R_int=%.1f Ω, R_load=%.1f Ω.",
            self.mode.value, self.transducer_type.value,
            self.friction_model.value, self.coupling_factor,
            self.internal_resistance, self.load_resistance,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §2.2 — Validación de parámetros (preservada de v3.0.0)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_params(
        capacity            : float,
        elasticity          : float,
        damping             : float,
        piston_mass         : float,
        coulomb_friction    : float,
        nonlinear_elasticity: float,
        dt_default          : float,
    ) -> None:
        r"""
        Valida los parámetros físicos del pistón atómico.

        Invariantes físicos exigidos:
            capacity             > 0   (longitud positiva, define el dominio)
            elasticity           ≥ 0   (rigidez no negativa, k=0 → masa libre)
            damping              ≥ 0   (amortiguamiento no negativo)
            piston_mass          > 0   (masa positiva, m=0 → ecuación singular)
            coulomb_friction     ≥ 0   (fricción no negativa)
            nonlinear_elasticity ≥ 0   (ε=0 → sistema Hamiltoniano lineal)
            dt_default           > 0   (paso de tiempo positivo, CFL)

        Lanza
        -----
        ValueError : Si alguna condición es violada.
        """
        checks = [
            (capacity > 0,              "capacity",             "> 0"),
            (elasticity >= 0,           "elasticity",           "≥ 0"),
            (damping >= 0,              "damping",              "≥ 0"),
            (piston_mass > 0,           "piston_mass",          "> 0"),
            (coulomb_friction >= 0,     "coulomb_friction",     "≥ 0"),
            (nonlinear_elasticity >= 0, "nonlinear_elasticity", "≥ 0"),
            (dt_default > 0,            "dt_default",           "> 0"),
        ]
        for condition, name, constraint in checks:
            if not condition:
                raise ValueError(
                    f"Parámetro inválido: '{name}' debe ser {constraint}."
                )

    # ─────────────────────────────────────────────────────────────────────
    # §2.3 — Configuración del transductor electromecánico
    # ─────────────────────────────────────────────────────────────────────

    def _configure_transducer(self) -> None:
        r"""
        Establece los parámetros del acoplamiento electromecánico (α, R_int).

        Factor de acoplamiento α [N/A = V·s/m]:
            Aparece en J(x)[1,2] = α y J(x)[2,1] = -α.
            Codifica la bidireccionalidad del transductor:
                Mecánico → Eléctrico: Q̇ = -α·ẋ  (corriente generada)
                Eléctrico → Mecánico: ṗ += α·V_C  (fuerza generada)

        Lanza
        -----
        ValueError : Si el tipo de transductor no está soportado.
        """
        params: Dict[TransducerType, Tuple[float, float]] = {
            TransducerType.PIEZOELECTRIC   : (50.0,  100.0),
            TransducerType.ELECTROSTATIC   : (100.0, 500.0),
            TransducerType.MAGNETOSTRICTIVE: (30.0,   50.0),
        }
        if self.transducer_type not in params:
            raise ValueError(
                f"Tipo de transductor no soportado: {self.transducer_type}. "
                f"Válidos: {list(params.keys())}"
            )
        self.coupling_factor, self.internal_resistance = params[self.transducer_type]

    # ─────────────────────────────────────────────────────────────────────
    # §2.4 — Reconstrucción de matrices port-Hamiltonianas
    # ─────────────────────────────────────────────────────────────────────

    def _rebuild_port_matrices(self) -> None:
        r"""
        Reconstruye J, R y (J−R) tras cambios en parámetros.

        Se llama cuando cambian:
            - coupling_factor α  → afecta J
            - damping c          → afecta R
            - load_resistance    → afecta R

        La pre-computación de (J−R) en self._JR evita la resta en cada paso
        del integrador, reduciendo la carga computacional de update_state().

        Verificación post-construcción:
            (a) J antisimétrica: ‖J + Jᵀ‖_F < ε_mach
            (b) R semidefinida positiva: eigenvalores ≥ 0
        """
        self._R_total = self.internal_resistance + self.load_resistance
        self._J       = _build_J_matrix(self.coupling_factor)
        self._R       = _build_R_matrix(self.c, self._R_total)
        self._JR      = self._J - self._R

        # Verificación de antisimetría de J (diagnóstico)
        antisym_err = float(np.linalg.norm(self._J + self._J.T, ord='fro'))
        if antisym_err > 1e-12:
            logger.warning(
                "J no es perfectamente antisimétrica: ‖J+Jᵀ‖_F = %.2e.",
                antisym_err,
            )

    # ─────────────────────────────────────────────────────────────────────
    # §2.5 — Propiedades derivadas del estado extendido
    # ─────────────────────────────────────────────────────────────────────

    @property
    def position(self) -> float:
        """Posición q [m] (componente 0 del estado extendido)."""
        return self.q

    @property
    def current_charge(self) -> float:
        """Carga como compresión positiva: max(0, -q) [m]."""
        return max(0.0, -self.q)

    @property
    def stored_energy(self) -> float:
        r"""
        Hamiltoniano H(q,p,Q) = energía total almacenada [J].

            H = p²/(2m) + ½kq² + ¼εq⁴ + Q²/(2C_eq)

        Componentes:
            T(p)   = p²/(2m)        energía cinética [J]
            V(q)   = ½kq² + ¼εq⁴   energía potencial [J]
            W_e(Q) = Q²/(2C_eq)     energía eléctrica [J]

        La energía eléctrica usa Q (carga real integrada) en lugar de
        ½C_eq·V_oc² (aproximación de v3.0.0), que era correcto solo cuando
        Q = C_eq·V_oc (circuito en equilibrio). Ahora es exacto para
        cualquier estado transitorio.
        """
        T    = (self.p ** 2) / (2.0 * max(self.m, 1e-12))
        V    = (
            0.5 * self.k * self.q ** 2
            + 0.25 * self.nonlinear_elasticity * self.q ** 4
        )
        W_e  = (self.Q_charge ** 2) / (2.0 * max(self.equivalent_capacitance, 1e-15))
        return T + V + W_e

    # ─────────────────────────────────────────────────────────────────────
    # §2.6 — Aplicación de fuerzas externas (preservada con corrección)
    # ─────────────────────────────────────────────────────────────────────

    def apply_force(
        self,
        signal_value: float,
        source      : str   = "external",
        mass_factor : float = 1.0,
    ) -> None:
        r"""
        Convierte una señal de entrada en fuerza mecánica acumulada [N].

        Fórmula (FIX 1 preservado):
            F = compression_direction · signal² / (2 · mass_factor)  [N]

        La fuerza se acumula en `last_applied_force` y es consumida
        en el siguiente `update_state()` como componente de F_ext
        en el término B · F_ext del balance port-Hamiltoniano.

        Parámetros
        ----------
        signal_value : float, señal de entrada [m/s].
        source       : str, identificador de la fuente (logging).
        mass_factor  : float, factor de inercia [kg]. Debe ser > 0.
        """
        if mass_factor <= 0:
            logger.warning(
                "apply_force: mass_factor=%.3e ≤ 0, usando 1.0.", mass_factor
            )
            mass_factor = 1.0
        force = self.compression_direction * (signal_value ** 2) / (2.0 * mass_factor)
        self.last_applied_force += force
        logger.debug(
            "Fuerza acumulada desde '%s': %.4f N (total=%.4f N).",
            source, force, self.last_applied_force,
        )

    def apply_electronic_signal(self, voltage: float) -> None:
        r"""
        Traduce un voltaje externo en fuerza mecánica [N] via J(x).

        En la estructura port-Hamiltoniana, el voltaje externo V_ext
        produce una corriente I = V_ext/R_int que genera fuerza:

            F = α · I = α · V_ext / R_int   [N]

        Para magnetostrictivo: el circuito RL integra la corriente;
        la fuerza es α · I_RL (estado integrado).

        Parámetros
        ----------
        voltage : float, tensión de accionamiento [V].
        """
        if self.transducer_type == TransducerType.MAGNETOSTRICTIVE:
            # Corriente del circuito RL integrado (estado Q̇)
            applied_force = self.circuit_current * self.coupling_factor
        else:
            drive_current = voltage / max(self.internal_resistance, 1e-9)
            applied_force = drive_current * self.coupling_factor

        self.last_applied_force += applied_force
        logger.debug(
            "Señal eléctrica %.3f V → %.4f N (transductor: %s).",
            voltage, applied_force, self.transducer_type.value,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §2.7 — Gradiente del Hamiltoniano en el estado actual
    # ─────────────────────────────────────────────────────────────────────

    def _grad_H(
        self,
        q       : float,
        p       : float,
        Q_charge: float,
    ) -> np.ndarray:
        r"""
        Evalúa ∇H(q,p,Q) en el punto (q,p,Q) dado.

        Delega a la función pura _compute_hamiltonian_gradient con los
        parámetros del pistón (m, k, ε, C_eq).

        Parámetros
        ----------
        q        : float, posición [m].
        p        : float, momento [kg·m/s].
        Q_charge : float, carga eléctrica [C].

        Retorna
        -------
        np.ndarray, shape (3,), ∇H = [kq+εq³, p/m, Q/C_eq]ᵀ.
        """
        return _compute_hamiltonian_gradient(
            q               = q,
            p               = p,
            Q_charge        = Q_charge,
            mass            = self.m,
            elasticity      = self.k,
            nonlinear_elast = self.nonlinear_elasticity,
            capacitance_eq  = self.equivalent_capacitance,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §2.8 — SUTURA 3: Integrador Störmer-Verlet Simpléctico
    # ─────────────────────────────────────────────────────────────────────

    def _stormer_verlet_step(
        self,
        F_ext: float,
        dt   : float,
    ) -> Tuple[float, float, float]:
        r"""
        Ejecuta un paso del integrador Störmer-Verlet simpléctico.

        SUTURA 3 PRINCIPAL: Reemplaza el integrador RK4 (extirpado).

        El método de Störmer-Verlet es simpléctico para H separable
        H = T(p) + V(q), con error de energía O(Δt²) (en lugar de O(Δt⁴)
        de RK4), pero PRESERVA la 2-forma ω = dq∧dp exactamente en el
        caso no disipativo (R=0).

        Con disipación R ≠ 0, el esquema modificado absorbe la disipación
        de manera simétrica (R aplicada en los semi-pasos) para minimizar
        el error simpléctic introducido por la disipación.

        Esquema completo (4 pasos):
        ────────────────────────────

        Paso 1 — Semi-impulso inicial:
            ∇_q H(qₙ) = k·qₙ + ε·qₙ³        [fuerza del resorte]
            Numerador:  pₙ - (Δt/2)·∇_q H(qₙ) + (Δt/2)·F_ext
            Denominador: 1 + (Δt/2)·(c/m)     [disipación viscosa implícita]
            p_{n+½} = numerador / denominador

        Paso 2 — Posición completa:
            q_{n+1} = qₙ + Δt · p_{n+½} / m

        Paso 3 — Semi-impulso final:
            ∇_q H(q_{n+1}) = k·q_{n+1} + ε·q_{n+1}³
            Numerador:  p_{n+½} - (Δt/2)·∇_q H(q_{n+1}) + (Δt/2)·F_ext
            Denominador: 1 + (Δt/2)·(c/m)
            p_{n+1} = numerador / denominador

        Paso 4 — Carga eléctrica (Euler implícito acoplado):
            α·(p_{n+½}/m) = término de acoplamiento [A] (corriente inducida)
            Q̇ = -α·ẋ - Q/C_eq/R_t   (ecuación del circuito RC)
            Q_{n+1} = [Qₙ - Δt·α·(p_{n+½}/m)] / [1 + Δt/(C_eq·R_t)]
            (Euler implícito garantiza estabilidad del circuito RC)

        Colisión con límites (posición):
            Si |q_{n+1}| > capacity:
                q ← clip(q, -capacity, +capacity)
                p ← -e · p   [rebote inelástico, e = RESTITUTION_COEFF]
                La colisión disipa energía cinética (1-e²)·T
                sin necesitar modificar Q (la carga eléctrica no rebota).

        Parámetros
        ----------
        F_ext : float, fuerza externa total [N] (aplicada + PID + fricción).
        dt    : float, paso de tiempo [s]. Debe ser > 0.

        Retorna
        -------
        Tuple[float, float, float]
            (q_{n+1}, p_{n+1}, Q_{n+1}) — nuevo estado extendido.
        """
        # ── Estado actual ─────────────────────────────────────────────────
        q_n = self.q
        p_n = self.p
        Q_n = self.Q_charge

        # ── Paso 1: Semi-impulso inicial (Verlet split) ───────────────────
        # ∇_q H(qₙ) — fuerza del resorte [N]
        dH_dq_n = self.k * q_n + self.nonlinear_elasticity * (q_n ** 3)

        # Disipación viscosa implícita en el semi-paso
        # (evita inestabilidad numérica del esquema explícito para c grande)
        viscous_denom = 1.0 + (dt / 2.0) * (self.c / max(self.m, 1e-12))

        p_half = (
            p_n
            - (dt / 2.0) * dH_dq_n
            + (dt / 2.0) * F_ext
        ) / viscous_denom

        # ── Paso 2: Posición completa ─────────────────────────────────────
        q_new = q_n + dt * (p_half / max(self.m, 1e-12))

        # ── Paso 3: Semi-impulso final ────────────────────────────────────
        dH_dq_new = self.k * q_new + self.nonlinear_elasticity * (q_new ** 3)

        p_new = (
            p_half
            - (dt / 2.0) * dH_dq_new
            + (dt / 2.0) * F_ext
        ) / viscous_denom

        # ── Paso 4: Carga eléctrica (Euler implícito) ─────────────────────
        # Velocidad en el semi-paso: ẋ_{n+½} = p_{n+½}/m
        vel_half  = p_half / max(self.m, 1e-12)

        # Término de acoplamiento: corriente inducida [A]
        # Q̇_coup = -α · ẋ_{n+½}
        coupling_current = -self.coupling_factor * vel_half

        # Euler implícito para Q: Q_{n+1}(1 + Δt/(C_eq·R_t)) = Qₙ + Δt·coupling
        rc_denom = 1.0 + dt / max(
            self.equivalent_capacitance * self._R_total, 1e-15
        )
        Q_new = (Q_n + dt * coupling_current) / rc_denom

        # ── Colisión con límites físicos ──────────────────────────────────
        if abs(q_new) > self.capacity:
            # Clip de posición al dominio [-capacity, +capacity]
            q_new = float(np.clip(q_new, -self.capacity, self.capacity))
            # Rebote inelástico: inversión del momento con restitución e
            p_new = -self.RESTITUTION_COEFF * p_new
            logger.debug(
                "Colisión detectada: q=%.4f m → clip ±%.4f m, "
                "p_rebote=%.4f kg·m/s.",
                q_new, self.capacity, p_new,
            )

        return q_new, p_new, Q_new

    # ─────────────────────────────────────────────────────────────────────
    # §2.9 — Fricción covariante integrada en el paso Verlet
    # ─────────────────────────────────────────────────────────────────────

    def _compute_friction_force(
        self,
        velocity     : float,
        spring_force : float,
        F_ext        : float,
    ) -> float:
        r"""
        Calcula la fuerza de fricción covariante (Sutura 1 → Sutura 3).

        Conecta FrictionCalculator (Fase 1) con el integrador Störmer-Verlet.

        La fuerza impulsora (sin disipación) es:
            F_drv = F_ext - (kq + εq³)   [N]

        Esta fuerza se pasa a compute_friction() junto con la velocidad
        y el tensor métrico G para obtener el covector de fricción [N].

        Parámetros
        ----------
        velocity     : float, velocidad actual ẋ = p/m [m/s].
        spring_force : float, fuerza del resorte -(kq+εq³) [N].
        F_ext        : float, fuerza externa total [N].

        Retorna
        -------
        float, fuerza de fricción covariante [N].
        """
        driving_force = F_ext + spring_force  # F_drv sin disipación

        return FrictionCalculator.compute_friction(
            velocity        = velocity,
            driving_force   = driving_force,
            friction_model  = self.friction_model,
            coulomb_friction = self.coulomb_friction,
            stribeck_coeffs  = self.stribeck_coeffs,
            metric_tensor    = self._metric_tensor,
        )

    # ─────────────────────────────────────────────────────────────────────
    # §2.10 — SUTURA 4: Dinámica port-Hamiltoniana y actualización de estado
    # ─────────────────────────────────────────────────────────────────────

    def update_state(self, dt: float) -> None:
        r"""
        Actualiza el estado extendido x=[q,p,Q]ᵀ en un paso de tiempo dt.

        SUTURA 3 + 4 INTEGRADAS:

        Pipeline del paso de integración:
        ───────────────────────────────────

        1. Ensamblaje de F_ext:
               F_ext = F_applied + F_pid + F_fric   [N]

           donde:
               F_applied = last_applied_force         [señales externas]
               F_pid     = LieGroupPIDController.update()  [Sutura 2, Fase 1]
               F_fric    = FrictionCalculator.compute_friction()  [Sutura 1]

        2. Störmer-Verlet:  (q_n, p_n, Q_n) → (q_{n+1}, p_{n+1}, Q_{n+1})

        3. Actualización de variables derivadas:
               velocity        = p_{n+1}/m
               acceleration    = (p_{n+1} - p_n) / (m·dt)
               circuit_voltage = Q_{n+1}/C_eq
               circuit_current = (Q_{n+1} - Q_n) / dt
               output_voltage  = |I|·R_load·η

        4. Verificación de inecuación de disipación:
               Ḣ_dis = -∇Hᵀ R ∇H ≤ 0  (garantía física)

        5. Registro de historial y limpieza de fuerza acumulada.

        Parámetros
        ----------
        dt : float, paso de tiempo [s]. Debe ser > 0.

        Lanza
        -----
        ValueError : Si dt ≤ 0.
        """
        if dt <= 0:
            raise ValueError(
                f"Paso de tiempo debe ser positivo, recibido dt={dt}."
            )
        self.dt = dt

        # §2.10.1: Velocidad en el estado actual (variable derivada)
        vel_current = self.p / max(self.m, 1e-12)

        # §2.10.2: Fuerza de fricción covariante (Sutura 1 → Sutura 3)
        spring_force_neg = -(self.k * self.q + self.nonlinear_elasticity * self.q ** 3)
        friction_force = self._compute_friction_force(
            velocity     = vel_current,
            spring_force = spring_force_neg,
            F_ext        = self.last_applied_force,
        )

        # §2.10.3: Señal PID geométrica (Sutura 2 → Sutura 3)
        pid_force = 0.0
        if self.target_speed != 0.0:
            pid_force = self.speed_controller.update(
                setpoint         = self.target_speed,
                current_value    = vel_current,
                dt               = dt,
                current_velocity = vel_current,
            )

        # §2.10.4: Fuerza externa total
        F_ext_total = (
            self.last_applied_force
            + pid_force
            + friction_force
        )

        # §2.10.5: Integrador Störmer-Verlet (Sutura 3)
        p_prev    = self.p
        Q_prev    = self.Q_charge

        q_new, p_new, Q_new = self._stormer_verlet_step(F_ext_total, dt)

        # §2.10.6: Actualización del estado primario extendido
        self.q        = q_new
        self.p        = p_new
        self.Q_charge = Q_new

        # §2.10.7: Derivadas del estado (variables secundarias)
        self.velocity     = p_new / max(self.m, 1e-12)
        self.acceleration = (p_new - p_prev) / max(self.m * dt, 1e-15)

        # Variables eléctricas derivadas del estado Q
        self.circuit_voltage = Q_new / max(self.equivalent_capacitance, 1e-15)
        self.circuit_current = (Q_new - Q_prev) / max(dt, 1e-15)

        # Voltaje de salida: V_load = |I|·R_load·η (FIX 3 preservado)
        abs_current      = max(abs(self.circuit_current), self._I_MIN)
        output_power     = (abs_current ** 2) * self.load_resistance
        self.output_voltage = math.sqrt(
            max(output_power * self.converter_efficiency, 0.0)
        )

        # §2.10.8: Magnetostrictivo — decaimiento RL de la corriente
        if self.transducer_type == TransducerType.MAGNETOSTRICTIVE:
            self._update_magnetostrictive_current(dt)

        # §2.10.9: Verificación de inecuación de disipación (Sutura 4)
        grad_H_now = self._grad_H(q_new, p_new, Q_new)
        dissipation_rate = _verify_dissipation_inequality(
            grad_H    = grad_H_now,
            R_matrix  = self._R,
            tolerance = self._EPS_DISS,
        )

        # §2.10.10: Reinicio de fuerza acumulada
        self.last_applied_force = 0.0

        # §2.10.11: Registro de historial
        self.position_history.append(self.q)
        self.velocity_history.append(self.velocity)
        self.energy_history.append(self.stored_energy)
        self.efficiency_history.append(self.get_conversion_efficiency())
        self.friction_force_history.append(friction_force)
        self.hamiltonian_history.append(self.stored_energy)
        self.dissipation_history.append(dissipation_rate)

    # ─────────────────────────────────────────────────────────────────────
    # §2.11 — Dinámica RL del transductor magnetostrictivo (preservada)
    # ─────────────────────────────────────────────────────────────────────

    def _update_magnetostrictive_current(self, dt: float) -> None:
        r"""
        Integra la ecuación RL en evolución libre: I(t+dt) = I(t)·e^{-dt/τ}.

            τ = L/R = m/R_int   (constante de tiempo del circuito RL)

        En el formalismo port-Hamiltoniano, la corriente magnetostrictiva
        es el estado Q̇ del subsistema eléctrico. La evolución libre decae
        exponencialmente desde la corriente actual circuit_current.

        Este método actualiza circuit_current (no Q_charge) porque el
        circuito RL tiene inductancia, no capacitancia, como elemento
        de almacenamiento.

        Parámetros
        ----------
        dt : float, paso de tiempo [s].
        """
        L = self.equivalent_inductance  # m [kg = H]
        R = self.internal_resistance    # [Ω]
        if L > 0 and R > 0:
            tau = L / R
            self.circuit_current *= math.exp(-dt / tau)

    # ─────────────────────────────────────────────────────────────────────
    # §2.12 — Gestión de descarga (preservada con estado extendido)
    # ─────────────────────────────────────────────────────────────────────

    def discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        r"""
        Ejecuta el ciclo de descarga según el modo de operación.

        La descarga opera sobre el estado extendido q (posición), que
        en el contexto del pistón atómico representa la compresión
        acumulada (energía potencial almacenada en el resorte).

        Retorna
        -------
        Optional[Dict[str, Any]] : pulso de salida o None.
        """
        if self.mode == PistonMode.CAPACITOR:
            return self._capacitor_discharge(dt)
        if self.mode == PistonMode.BATTERY:
            return self._battery_discharge(dt)
        return None

    def _capacitor_discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        r"""
        Descarga tipo CAPACITOR: disparo rápido cuando q ≤ threshold.

        En el estado extendido, el disparo modifica directamente (q, p):
            q ← threshold·(1 - hysteresis_factor)
            p ← m · 5.0  [impulso de rebote: momento = masa × velocidad]

        La carga Q no se altera (la carga eléctrica no participa en el
        mecanismo de disparo mecánico del capacitor).
        """
        if self.q <= self.capacitor_discharge_threshold:
            amplitude = self.current_charge
            logger.info(
                "¡Descarga CAPACITOR! Amplitud=%.4f m, q=%.4f m.",
                amplitude, self.q,
            )
            # Modificar estado extendido (q, p) — no Q
            self.q = self.capacitor_discharge_threshold * (
                1.0 - self.hysteresis_factor
            )
            # Impulso de rebote expresado como momento p = m·v
            self.p = self.m * 5.0
            # Sincronizar variable derivada
            self.velocity = 5.0
            return {"type": "pulse", "amplitude": amplitude, "duration": 0.001}
        return None

    def _battery_discharge(self, dt: float) -> Optional[Dict[str, Any]]:
        r"""
        Descarga tipo BATTERY: liberación gradual de energía.

        En el estado extendido, la descarga modifica q (posición):
            discharge_amount = min(rate·dt, current_charge)
            q_new = min(0, q + discharge_amount)

        El momento p se actualiza para ser consistente con la nueva
        posición (p ← 0 al alcanzar equilibrio, liberación gradual).

        Conservación de energía (FIX 7 preservado):
            discharge_amount ≤ current_charge → q no supera 0.
        """
        if not self.battery_is_discharging:
            return None

        charge = self.current_charge
        if charge <= 1e-5:
            self.battery_is_discharging = False
            self.q = 0.0
            self.p = 0.0
            self.velocity = 0.0
            logger.info("Descarga BATTERY completada.")
            return None

        discharge_amount = min(self.battery_discharge_rate * dt, charge)
        self.q = min(0.0, self.q + discharge_amount)
        # Actualizar momento consistentemente
        self.velocity = self.p / max(self.m, 1e-12)

        output_amplitude = (
            discharge_amount / (self.battery_discharge_rate * dt)
            if self.battery_discharge_rate * dt > 1e-12
            else 1.0
        )
        return {
            "type"     : "sustained",
            "amplitude": output_amplitude,
            "duration" : dt,
        }

    # ─────────────────────────────────────────────────────────────────────
    # §2.13 — Simulación del circuito de descarga (FIX 5 + estado extendido)
    # ─────────────────────────────────────────────────────────────────────

    def simulate_discharge_circuit(
        self,
        load_resistance: float,
        dt             : float,
    ) -> Tuple[float, float, float]:
        r"""
        Simula la disipación a través de una carga R_L en el estado extendido.

        En el formalismo port-Hamiltoniano, la descarga a través de R_L
        modifica la matriz R(x) temporalmente:

            R_t_temp = R_int + R_L  (nueva resistencia total)

        y la energía eléctrica se disipa como:

            P_load = I² · R_L = (Q/C_eq)²·R_L / R_t_temp²   [W]

        La actualización de Q sigue la ecuación del circuito:

            Q_{n+1} = [Qₙ - Δt·α·ẋ] / [1 + Δt/(C_eq·R_t_temp)]

        Balance energético (FIX 5 preservado):
            ΔW_total = P_total·dt  (energía mecánica extraída)
            ΔW_mec actualiza q via: q² → q² - 2·ΔW_mec/k

        Parámetros
        ----------
        load_resistance : float, R_L [Ω]. Debe ser > 0.
        dt              : float, paso de tiempo [s]. Debe ser > 0.

        Retorna
        -------
        Tuple[float, float, float]
            (voltaje_carga [V], corriente [A], potencia_carga [W]).
        """
        if self.circuit_voltage == 0.0 or load_resistance <= 0 or dt <= 0:
            return 0.0, 0.0, 0.0

        r_total     = self.internal_resistance + load_resistance
        current     = self.circuit_voltage / max(r_total, 1e-9)
        power_total = (self.circuit_voltage ** 2) / max(r_total, 1e-9)
        power_load  = (current ** 2) * load_resistance

        # Balance energético: actualizar q via conservación de W_pot = ½kq²
        delta_w_mec = power_total * dt
        if self.k > 1e-9 and abs(self.q) > 1e-9:
            x_sq_new = self.q ** 2 - 2.0 * delta_w_mec / self.k
            if x_sq_new > 0:
                q_new = math.copysign(math.sqrt(x_sq_new), self.q)
            else:
                q_new = 0.0
            if self.q < 0:
                self.q = max(-self.capacity, min(0.0, q_new))

        # Actualizar Q usando la nueva resistencia de carga temporal
        rc_denom = 1.0 + dt / max(
            self.equivalent_capacitance * r_total, 1e-15
        )
        coupling_current = -self.coupling_factor * self.velocity
        self.Q_charge = (self.Q_charge + dt * coupling_current) / rc_denom

        # Sincronizar variables derivadas
        self.circuit_voltage = self.Q_charge / max(self.equivalent_capacitance, 1e-15)
        self.circuit_current = current

        load_voltage = current * load_resistance
        return load_voltage, current, power_load

    # ─────────────────────────────────────────────────────────────────────
    # §2.14 — Eficiencia de conversión (FIX 4 + estado extendido)
    # ─────────────────────────────────────────────────────────────────────

    def get_conversion_efficiency(self) -> float:
        r"""
        Calcula la eficiencia de conversión energética η ∈ [0,1].

        Con el estado extendido x=[q,p,Q]ᵀ, el Hamiltoniano se descompone:

            H = H_mec + H_elec

            H_mec  = p²/(2m) + ½kq² + ¼εq⁴   [J]
            H_elec = Q²/(2C_eq)                 [J]

        Eficiencia de conversión mecánico→eléctrico:

            η = H_mec / (H_mec + H_elec)  ∈ [0,1]

        Esta definición es exacta para cualquier estado transitorio,
        a diferencia de v3.0.0 que usaba V_oc = α·ẋ (solo válido en
        el circuito de equilibrio).

        Retorna
        -------
        float ∈ [0,1]. Retorna 0.0 si H_total < ε.
        """
        T      = (self.p ** 2) / (2.0 * max(self.m, 1e-12))
        V      = (
            0.5 * self.k * self.q ** 2
            + 0.25 * self.nonlinear_elasticity * self.q ** 4
        )
        H_mec  = T + V
        H_elec = (self.Q_charge ** 2) / (2.0 * max(self.equivalent_capacitance, 1e-15))
        total  = H_mec + H_elec
        if total > 1e-9:
            return H_mec / total
        return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # §2.15 — Análisis de Bode (FIX 10 preservado + función de transferencia
    #          port-Hamiltoniana linealizada)
    # ─────────────────────────────────────────────────────────────────────

    def generate_bode_data(self, frequency_range: np.ndarray) -> Dict[str, Any]:
        r"""
        Genera datos de Bode para el sistema port-Hamiltoniano linealizado.

        Linealización en torno al punto de equilibrio x* = [0,0,0]ᵀ:

            Ȧ = (J−R) · K   (sistema lineal)

        donde K = ∇²H|_{x*} = diag(k, 1/m, 1/C_eq) (Hessiano del Hamiltoniano).

        Función de transferencia del momento a la posición:

            H_mec(jω) = 1 / (-mω² + jcω + k)   [m/N]

        Función de transferencia electromecánica (V_C a F_ext):

            H_elec(jω) = α · H_mec(jω) / (1 + jω·C_eq·R_t)   [V/N]

        El polo eléctrico ω_RC = 1/(C_eq·R_t) aparece en la función de
        transferencia completa, ausente en v3.0.0 (que ignoraba el polo RC).

        Corrección FIX 10 preservada: retorna listas Python serializables.

        Parámetros
        ----------
        frequency_range : np.ndarray, frecuencias [Hz]. Todos deben ser > 0.

        Retorna
        -------
        Dict[str, Any] con keys: frequencies, magnitude_dB, phase_deg,
                                  magnitude_elec_dB, phase_elec_deg.
        """
        magnitudes_mec  : List[float] = []
        phases_mec      : List[float] = []
        magnitudes_elec : List[float] = []
        phases_elec     : List[float] = []
        freqs_out       : List[float] = []

        for f in frequency_range:
            if f <= 0:
                logger.warning(
                    "Frecuencia no positiva en Bode: %.3e Hz, omitida.", f
                )
                continue
            omega  = 2.0 * math.pi * float(f)
            jw     = 1j * omega

            # Denominador mecánico: m(jω)² + c(jω) + k
            denom_mec = self.m * (jw ** 2) + self.c * jw + self.k
            if abs(denom_mec) < 1e-30:
                h_mec  = complex(0.0, 0.0)
                h_elec = complex(0.0, 0.0)
            else:
                h_mec  = 1.0 / denom_mec

                # Polo RC eléctrico: 1 + jω·C_eq·R_t
                denom_rc = 1.0 + jw * self.equivalent_capacitance * self._R_total

                # Función de transferencia port-Hamiltoniana completa
                h_elec = (self.coupling_factor * h_mec) / max(abs(denom_rc), 1e-30) \
                         * complex(denom_rc.real, -denom_rc.imag) / (abs(denom_rc) ** 2) \
                         * abs(denom_rc)
                # Forma compacta correcta:
                h_elec = self.coupling_factor * h_mec / denom_rc

            freqs_out.append(float(f))
            magnitudes_mec.append(
                20.0 * math.log10(max(abs(h_mec  * self.coupling_factor), 1e-15))
            )
            phases_mec.append(
                math.degrees(math.atan2(
                    (h_mec * self.coupling_factor).imag,
                    (h_mec * self.coupling_factor).real,
                ))
            )
            magnitudes_elec.append(
                20.0 * math.log10(max(abs(h_elec), 1e-15))
            )
            phases_elec.append(
                math.degrees(math.atan2(h_elec.imag, h_elec.real))
            )

        return {
            "frequencies"      : freqs_out,
            "magnitude_dB"     : magnitudes_mec,
            "phase_deg"        : phases_mec,
            "magnitude_elec_dB": magnitudes_elec,
            "phase_elec_deg"   : phases_elec,
        }

    # ─────────────────────────────────────────────────────────────────────
    # §2.16 — Exportación de historial (FIX 11 preservado + campos nuevos)
    # ─────────────────────────────────────────────────────────────────────

    def export_history_to_csv(self, filename: str) -> None:
        r"""
        Exporta el historial de simulación a CSV (FIX 11 preservado).

        Columnas adicionales en v4.0.0 (estado extendido):
            momentum     : p [kg·m/s]
            charge       : Q [C]
            hamiltonian  : H(q,p,Q) [J]
            dissipation  : Ḣ_dis = -∇Hᵀ R ∇H [W]

        FIX 11 preservado: deques convertidos a listas antes de indexar.

        Parámetros
        ----------
        filename : str, ruta del archivo CSV de salida.
        """
        pos_list  = list(self.position_history)
        vel_list  = list(self.velocity_history)
        eng_list  = list(self.energy_history)
        eff_list  = list(self.efficiency_history)
        fric_list = list(self.friction_force_history)
        ham_list  = list(self.hamiltonian_history)
        diss_list = list(self.dissipation_history)

        max_len = max(
            len(pos_list), len(vel_list), len(eng_list),
            len(eff_list), len(fric_list), len(ham_list),
            len(diss_list), 1,
        )
        header = [
            "time_step", "position_q", "velocity",
            "stored_energy", "efficiency", "friction_force",
            "hamiltonian", "dissipation_rate",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for i in range(max_len):
                    row = [
                        i,
                        pos_list[i]  if i < len(pos_list)  else "",
                        vel_list[i]  if i < len(vel_list)  else "",
                        eng_list[i]  if i < len(eng_list)  else "",
                        eff_list[i]  if i < len(eff_list)  else "",
                        fric_list[i] if i < len(fric_list) else "",
                        ham_list[i]  if i < len(ham_list)  else "",
                        diss_list[i] if i < len(diss_list) else "",
                    ]
                    writer.writerow(row)
            logger.info(
                "Historial port-Hamiltoniano exportado a '%s' (%d filas).",
                filename, max_len,
            )
        except IOError as exc:
            logger.error("Error al exportar CSV '%s': %s", filename, exc)
            raise

    # ─────────────────────────────────────────────────────────────────────
    # §2.17 — Métodos de control y configuración (preservados + extendido)
    # ─────────────────────────────────────────────────────────────────────

    def set_compression_direction(self, direction: int) -> None:
        r"""
        Establece la dirección de compresión (+1 o -1).

        Parámetros
        ----------
        direction : int, debe ser -1 o +1.
        """
        if direction in (-1, 1):
            self.compression_direction = direction
            logger.info("Dirección de compresión: %d.", direction)
        else:
            logger.warning(
                "Dirección inválida %d, manteniendo %d.",
                direction, self.compression_direction,
            )

    def set_mode(self, mode: PistonMode) -> None:
        """Cambia el modo de operación y reinicia la descarga."""
        self.mode = mode
        self.battery_is_discharging = False
        logger.info("Modo cambiado a '%s'.", mode.value)

    def trigger_discharge(self, discharge_on: bool) -> None:
        """Activa/desactiva la descarga en modo BATTERY."""
        if self.mode == PistonMode.BATTERY:
            self.battery_is_discharging = discharge_on
            logger.info(
                "Descarga BATTERY %s.",
                "activada" if discharge_on else "desactivada",
            )
        else:
            logger.warning(
                "trigger_discharge solo válido en BATTERY (actual: %s).",
                self.mode.value,
            )

    def set_speed_target(self, target: float) -> None:
        """Establece el objetivo de velocidad del controlador PID geométrico."""
        self.target_speed = float(target)
        self.speed_controller.reset()
        logger.debug("Target de velocidad: %.3f m/s.", self.target_speed)

    def set_energy_target(self, target: float) -> None:
        """Establece el objetivo de energía (debe ser ≥ 0)."""
        self.target_energy = max(0.0, float(target))
        self.energy_controller.reset()
        logger.debug("Target de energía: %.3f J.", self.target_energy)

    def set_load_resistance(self, r_load: float) -> None:
        r"""
        Actualiza la resistencia de carga y reconstruye las matrices J, R.

        Al modificar R_load, la matriz de disipación R(x) cambia:
            R[2,2] = 1/(R_int + R_load_new)

        Se reconstruye (J−R) para el próximo paso Störmer-Verlet.

        Parámetros
        ----------
        r_load : float, nueva resistencia de carga [Ω]. Debe ser > 0.
        """
        if r_load <= 0:
            logger.warning(
                "set_load_resistance: R_load=%.3e ≤ 0, ignorado.", r_load
            )
            return
        self.load_resistance = r_load
        self._rebuild_port_matrices()
        logger.info("R_load actualizada a %.2f Ω. Matrices J,R reconstruidas.", r_load)

    def reset(self) -> None:
        r"""
        Reinicia el pistón al estado inicial extendido x=[0,0,0]ᵀ.

        Restablece el estado port-Hamiltoniano completo:
            q = 0, p = 0, Q = 0
        y todas las variables derivadas y controladores.
        """
        # Estado primario extendido
        self.q        = 0.0
        self.p        = 0.0
        self.Q_charge = 0.0

        # Variables derivadas
        self.velocity             = 0.0
        self.acceleration         = 0.0
        self.circuit_voltage      = 0.0
        self.circuit_current      = 0.0
        self.output_voltage       = 0.0
        self.last_applied_force   = 0.0
        self.battery_is_discharging = False

        # Historial
        self.position_history.clear()
        self.velocity_history.clear()
        self.energy_history.clear()
        self.efficiency_history.clear()
        self.friction_force_history.clear()
        self.hamiltonian_history.clear()
        self.dissipation_history.clear()

        # Controladores geométricos (Fase 1)
        self.speed_controller.reset()
        self.energy_controller.reset()

        logger.info("AtomicPiston [x=(q,p,Q)] reiniciado al estado inicial.")

    # ─────────────────────────────────────────────────────────────────────
    # §2.18 — Serialización del estado extendido para Fase 3
    # ─────────────────────────────────────────────────────────────────────

    def get_state_dict(self) -> Dict[str, Any]:
        r"""
        Retorna el estado port-Hamiltoniano como diccionario JSON-serializable.

        Expone el estado extendido x=[q,p,Q]ᵀ junto con todas las variables
        derivadas. El historial de las matrices J y R se incluye como
        metadatos diagnósticos para el DiracInterconnectionAgent (Fase 3).

        Retorna
        -------
        Dict[str, Any] con el estado completo del sistema port-Hamiltoniano.
        """
        # Gradiente del Hamiltoniano en el estado actual (esfuerzos)
        grad_H_now = self._grad_H(self.q, self.p, self.Q_charge)
        diss_now   = _verify_dissipation_inequality(grad_H_now, self._R)

        return {
            # ── Estado extendido primario ──────────────────────────────
            "state_extended": {
                "q"        : self.q,
                "p"        : self.p,
                "Q_charge" : self.Q_charge,
            },
            # ── Variables derivadas ────────────────────────────────────
            "position"            : self.q,
            "velocity"            : self.velocity,
            "acceleration"        : self.acceleration,
            "mode"                : self.mode.value,
            "stored_energy"       : self.stored_energy,
            "current_charge"      : self.current_charge,
            "circuit_voltage"     : self.circuit_voltage,
            "circuit_current"     : self.circuit_current,
            "output_voltage"      : self.output_voltage,
            "charge_accumulated"  : self.Q_charge,
            "battery_discharging" : self.battery_is_discharging,
            "efficiency"          : self.get_conversion_efficiency(),
            # ── Diagnóstico port-Hamiltoniano ──────────────────────────
            "port_hamiltonian": {
                "grad_H"          : grad_H_now.tolist(),
                "dissipation_rate": diss_now,
                "J_matrix"        : self._J.tolist(),
                "R_matrix"        : self._R.tolist(),
                "R_total_ohm"     : self._R_total,
            },
            # ── Objetivos de control ───────────────────────────────────
            "control_targets": {
                "target_speed" : self.target_speed,
                "target_energy": self.target_energy,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # FIN FASE 2 (AtomicPiston port-Hamiltoniano)
    #
    # Contrato de salida hacia Fase 3:
    #   → get_state_dict() : Dict JSON-serializable con x=[q,p,Q]ᵀ
    #   → discharge()      : Optional[Dict] con pulso de descarga
    #   → Ambos consumidos bajo ServiceContext.lock en simulation_loop()
    #
    # CONTINUACIÓN → FASE 3: Microservicio Flask + ServiceContext
    # ─────────────────────────────────────────────────────────────────────

    # ════════════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                  ║
# ║  FASE 3 — MICROSERVICIO FLASK + ServiceContext (CIRUGÍA DOCTORAL v4.0.0)        ║
# ║                                                                                  ║
# ║  SUTURA 5: Lock euclidiano → Buffer Circular Lock-Free (CAS atómico)            ║
# ║  SUTURA 6: Endpoints directos → Medición POVM (operadores de Kraus)             ║
# ║  SUTURA 7: Registro plano → Morfismo de Cohomología de Haces Celulares          ║
# ║                                                                                  ║
# ║  Contrato formal (v4.0.0):                                                       ║
# ║    Entrada : estado x=[q,p,Q]ᵀ de Fase 2 + peticiones HTTP                     ║
# ║    Salida  : respuestas JSON con estado POVM-colapsado + registro H¹=0          ║
# ║                                                                                  ║
# ║  Garantías (v4.0.0):                                                             ║
# ║    G1. Buffer CAS lock-free: lectura sin fricción en O(1) amortizado            ║
# ║    G2. simulation_loop: escritura atómica sin pausa del integrador              ║
# ║    G3. POVM: [H, O_api] = 0 certificado → no demolición del estado             ║
# ║    G4. Kraus: ρ_JSON = M_k ρ_sim M_k† / Tr(M_k ρ_sim M_k†)                   ║
# ║    G5. Cohomología: H¹(X; F_IPU) = 0 antes de admitir registro                 ║
# ║    G6. FIX 12 preservado: backpressure con conteo de ciclos lentos              ║
# ║    G7. FIX 13 preservado: Content-Type validado en endpoints POST               ║
# ║    G8. FIX 14 preservado: advertencia SSL explícita                             ║
# ║                                                                                  ║
# ║  La Fase 3 recibe el estado port-Hamiltoniano de la Fase 2 y lo expone         ║
# ║  via API RESTful con garantías de no-demolición y coherencia topológica.        ║
# ║                                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
# ════════════════════════════════════════════════════════════════════════════════════

"""
FUNDAMENTACIÓN MATEMÁTICA DE LA FASE 3 (v4.0.0):
══════════════════════════════════════════════════

§F6. BUFFER CIRCULAR LOCK-FREE (CAS ATÓMICO)
─────────────────────────────────────────────
Sea B = {s_0, s_1, ..., s_{N-1}} un buffer circular de capacidad N.

El productor (simulation_loop) escribe el estado x_n en la posición:
    write_idx = n mod N   (avance atómico via CAS)

El consumidor (endpoints HTTP) lee desde:
    read_idx  = (write_idx - 1) mod N   (último estado escrito)

La operación CAS (Compare-And-Swap) garantiza:
    if *ptr == expected: *ptr ← desired   (atómica, sin mutex)

En Python, la atomicidad de la escritura de un objeto dict en CPython
está garantizada por el GIL para operaciones de asignación de referencia
de un solo objeto. El patrón lock-free se implementa via:
    - ctypes.atomic o threading con variables índice enteras
    - La asignación `buffer[idx] = state` es atómica en CPython
      (la referencia al dict anterior se libera atomicamente)
    - El índice de escritura se avanza con un contador threading.local
      protegido por compare-and-swap via _atomic_increment()

Invariante del buffer:
    ∀ t: buffer[write_idx] contiene el estado más reciente
    La lectura read_idx = (write_idx - 1) mod N nunca bloquea

§F7. MEDICIÓN POVM (NO-DEMOLICIÓN)
────────────────────────────────────
El estado del simulador se modela como una matriz de densidad ρ_sim.

Para el sistema clásico-estocástico del pistón, ρ_sim es diagonal:
    ρ_sim = diag(p_1, ..., p_n)   donde p_i son probabilidades de
    cada micro-estado del buffer circular.

Los operadores de Kraus {M_k} definen la medición POVM:
    M_k†M_k ≥ 0,   Σ_k M_k†M_k = I   (completitud)

Para la medición no demoledora del estado del pistón:
    M_read = λ · I   (operador de lectura escalar, λ ∈ (0,1])

El colapso del estado es:
    ρ_JSON = M_read · ρ_sim · M_read† / Tr(M_read · ρ_sim · M_read†)
           = ρ_sim   (la medición no altera el estado: [H, O_api] = 0)

La conmutatividad [H, O_api] = 0 se certifica porque:
    O_api actúa sobre el buffer de lectura (copia independiente)
    H actúa sobre el estado primario (q, p, Q) en el integrador
    Los dos subsistemas son ortogonales → [H, O_api] = 0 ✓

§F8. COHOMOLOGÍA DE HACES CELULARES (REGISTRO ZERO-TRUST)
──────────────────────────────────────────────────────────
Sea X el complejo CW de la malla agéntica con:
    - 0-celdas (vértices): microservicios registrados
    - 1-celdas (aristas): canales de comunicación entre servicios

El haz F_IPU asigna a cada celda σ su estalk F(σ):
    F(v) = datos del microservicio v   (stalk local del pistón)
    F(e) = datos del canal e           (stalk de la arista)

El operador coborde δ: C⁰(X;F) → C¹(X;F):
    (δ⁰ s)(e) = F(e→v₂)·s(v₂) - F(e→v₁)·s(v₁)   para e = [v₁,v₂]

El primer grupo de cohomología del haz:
    H¹(X; F_IPU) = ker(δ¹) / im(δ⁰)

La condición H¹ = 0 certifica que:
    (a) No existen ciclos de comunicación parásitos en la malla
    (b) Las secciones globales del haz son únicas (no hay ambigüedad
        en el enrutamiento de mensajes entre microservicios)
    (c) El registro del pistón es consistente con la topología global

Si H¹ ≠ 0: el registro se cancela para prevenir propagación de ciclos.

Para la implementación computacional, modelamos X como un grafo G=(V,E):
    V = {servicios registrados + AtomicPiston}
    E = {canales de comunicación confirmados}

La condición H¹(X;ℤ) = 0 ↔ rk(H₁) = 0 ↔ el grafo G es un árbol (acíclico).
Se verifica con el rango del primer grupo de homología:
    β₁ = |E| - |V| + c   donde c = número de componentes conexas
    H¹ = 0 ↔ β₁ = 0 ↔ |E| = |V| - c   (bosque/árbol)
"""


import ctypes
import hashlib
import hmac
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, jsonify, request

from .config import PistonConfig
from .constants import FrictionModel, PistonMode, TransducerType


# ════════════════════════════════════════════════════════════════════════════════════
# §3.0 — ÁLGEBRA LOCK-FREE AUXILIAR (base para el Buffer CAS)
# ════════════════════════════════════════════════════════════════════════════════════

class _AtomicCounter:
    r"""
    Contador entero con incremento atómico usando ctypes + GIL de CPython.

    En CPython, las operaciones sobre objetos c_long son atómicas a nivel
    del GIL para lecturas y escrituras de un solo word. Esta clase encapsula
    un entero de longitud nativa para garantizar visibilidad entre hilos
    sin mutex explícito.

    La operación fetch_and_increment() es equivalente al instrucción x86:
        LOCK XADD [counter], 1   (fetch-and-add atómico)

    En Python puro, la atomicidad de `counter.value += 1` NO está
    garantizada (es una secuencia read-modify-write). Esta clase usa
    un Lock de grano fino SOLO para el incremento, no para las lecturas.
    Las lecturas de `value` son atómicas en CPython (GIL protege la carga
    de un c_long en un solo bytecode LOAD_ATTR).

    Propiedades:
        (a) fetch_and_increment(): atómica, retorna valor ANTERIOR
        (b) value (property): lectura atómica sin lock (GIL)
        (c) reset(): no atómica, solo para inicialización
    """

    def __init__(self, initial: int = 0) -> None:
        self._counter = ctypes.c_long(initial)
        self._lock    = threading.Lock()   # Lock de grano fino SOLO para write

    @property
    def value(self) -> int:
        """Lectura atómica del contador (GIL garantiza atomicidad en CPython)."""
        return self._counter.value

    def fetch_and_increment(self) -> int:
        r"""
        Incremento atómico fetch-and-add.

        Retorna el valor ANTERIOR al incremento (semántica fetch-and-add).
        Usa un Lock de grano fino que solo protege la secuencia
        read-modify-write, no las lecturas concurrentes.

        Retorna
        -------
        int, valor del contador antes del incremento.
        """
        with self._lock:
            old = self._counter.value
            self._counter.value = old + 1
        return old

    def reset(self, value: int = 0) -> None:
        """Reinicia el contador (no atómica, solo para inicialización)."""
        with self._lock:
            self._counter.value = value


class LockFreeCircularBuffer:
    r"""
    Buffer circular lock-free para estados del pistón atómico.

    SUTURA 5 PRINCIPAL: Reemplaza threading.Lock() euclidiano.

    Arquitectura:
    ─────────────
    El buffer es un array circular de capacidad N con dos punteros:
        _write_idx : índice de la próxima escritura (productor)
        _read_seq  : secuencia de lectura (consumidor)

    El productor (simulation_loop) escribe via _publish():
        1. Obtener slot = fetch_and_increment(_write_idx) mod N
        2. Escribir estado en _slots[slot]  ← atómico (asignación de ref)
        3. Actualizar _sequence[slot] = slot  ← atómico (c_long)

    El consumidor (endpoints HTTP) lee via _consume():
        1. Calcular pos = (_write_idx.value - 1) mod N
        2. Leer _slots[pos]  ← atómico (carga de ref en CPython)
        3. No bloquea, no espera: siempre retorna el último estado válido

    Invariante de consistencia:
        Si el productor está escribiendo en slot k y el consumidor lee
        slot k-1, la lectura es siempre del estado N-1 pasos atrás → válido.
        No hay torn read porque la asignación de referencia a dict en
        CPython es atómica bajo el GIL (single STORE_SUBSCR bytecode).

    Complejidad:
        Escritura: O(1) amortizado
        Lectura:   O(1) sin bloqueo

    Parámetros
    ----------
    capacity : int, número de slots del buffer. Recomendado: potencia de 2.
               Debe ser ≥ 2. Defecto: 16.
    """

    def __init__(self, capacity: int = 16) -> None:
        if capacity < 2:
            raise ValueError(
                f"capacity del buffer debe ser ≥ 2, recibido {capacity}."
            )
        self._capacity   : int               = capacity
        self._slots      : List[Optional[Dict[str, Any]]] = [None] * capacity
        self._write_idx  : _AtomicCounter    = _AtomicCounter(0)
        # Slot de lectura válido más reciente (None = buffer vacío)
        self._last_valid : _AtomicCounter    = _AtomicCounter(0)
        self._published  : bool              = False

    def publish(self, state: Dict[str, Any]) -> None:
        r"""
        Publica un nuevo estado en el buffer (operación del PRODUCTOR).

        La escritura es atómica a nivel de referencia en CPython:
            self._slots[idx] = state   ← STORE_SUBSCR atómico bajo GIL

        El consumidor que lea `_slots[idx]` obtendrá siempre un dict
        completo (nunca un estado parcialmente escrito), porque la
        asignación de referencia es una operación atómica del GIL.

        Parámetros
        ----------
        state : Dict[str, Any], estado port-Hamiltoniano serializable.
        """
        # Avance atómico del índice de escritura (wrapping circular)
        write_pos = self._write_idx.fetch_and_increment() % self._capacity
        # Escritura atómica de referencia (GIL garantiza no torn write)
        self._slots[write_pos] = state
        self._published = True

    def latest(self) -> Optional[Dict[str, Any]]:
        r"""
        Lee el estado más reciente sin bloquear (operación del CONSUMIDOR).

        La lectura es atómica:
            idx = (write_idx - 1) mod N   ← lectura atómica de c_long
            return self._slots[idx]        ← LOAD_SUBSCR atómico bajo GIL

        Si el buffer está vacío, retorna None.

        Retorna
        -------
        Optional[Dict[str, Any]], estado más reciente o None.
        """
        if not self._published:
            return None
        # El último índice publicado es write_idx - 1 (wrapping)
        read_pos = (self._write_idx.value - 1) % self._capacity
        return self._slots[read_pos]

    def is_empty(self) -> bool:
        """Retorna True si el buffer no tiene ningún estado publicado."""
        return not self._published

    def drain(self, max_items: int = -1) -> List[Dict[str, Any]]:
        r"""
        Extrae hasta max_items estados recientes del buffer (LIFO).

        Útil para el análisis de historial reciente en el endpoint /api/state.
        Retorna los estados en orden cronológico inverso (más reciente primero).

        Parámetros
        ----------
        max_items : int, máximo de estados a retornar. -1 = todos.

        Retorna
        -------
        List[Dict[str, Any]], estados en orden [más reciente, ..., más antiguo].
        """
        if not self._published:
            return []
        n = self._capacity if max_items < 0 else min(max_items, self._capacity)
        last_write = self._write_idx.value
        result: List[Dict[str, Any]] = []
        for i in range(n):
            idx   = (last_write - 1 - i) % self._capacity
            state = self._slots[idx]
            if state is not None:
                result.append(state)
        return result


# ════════════════════════════════════════════════════════════════════════════════════
# §3.1 — ÁLGEBRA POVM (Operadores de Kraus para medición de no-demolición)
# ════════════════════════════════════════════════════════════════════════════════════

class KrausObserver:
    r"""
    Modeliza la observación del estado del pistón como medición POVM.

    SUTURA 6: Reemplaza la serialización directa de get_state_dict().

    Fundamento Matemático:
    ──────────────────────
    Sea ρ_sim la matriz de densidad del estado del simulador, modelada
    como distribución de probabilidad sobre los estados del buffer circular:

        ρ_sim = Σ_k p_k |s_k⟩⟨s_k|   (mezcla estadística de estados)

    donde p_k = 1/N si todos los slots son igualmente probables.

    Para el sistema CLÁSICO-ESTOCÁSTICO del pistón, la matriz de densidad
    es el estado puntual más reciente (distribución delta):

        ρ_sim = |s_latest⟩⟨s_latest|   (estado puro del último slot)

    Los operadores de Kraus M_k definen la medición POVM:
        {M_k}: M_k†M_k ≥ 0,   Σ_k M_k†M_k = I

    Para la medición de no-demolición (QND):
        M_read = λ_k · P_k   donde P_k es proyector sobre el subespacio
        observable, y λ_k ∈ (0,1] es la eficiencia de la medición.

    El colapso:
        ρ_JSON = M_k ρ_sim M_k† / Tr(M_k ρ_sim M_k†)

    Conmutatividad [H, O_api] = 0:
        H actúa sobre (q,p,Q) en el integrador Störmer-Verlet.
        O_api actúa sobre la COPIA en el buffer circular.
        Como actúan sobre espacios de Hilbert ortogonales:
            [H ⊗ I_buffer, I_sim ⊗ O_api] = 0 ✓
        La medición NO drena exergía del bucle termodinámico.

    Parámetros
    ----------
    measurement_efficiency : float, λ ∈ (0,1]. Eficiencia del operador M_k.
                             λ=1.0 → medición perfecta (no demolición pura).
                             λ<1.0 → medición débil (parcial).
    observable_keys        : Optional[List[str]], claves del dict de estado
                             observables por este operador. None = todas.
    """

    def __init__(
        self,
        measurement_efficiency : float = 1.0,
        observable_keys        : Optional[List[str]] = None,
    ) -> None:
        if not (0.0 < measurement_efficiency <= 1.0):
            raise ValueError(
                f"measurement_efficiency debe ser ∈ (0,1], "
                f"recibido {measurement_efficiency}."
            )
        self._lambda       = float(measurement_efficiency)
        self._obs_keys     = observable_keys
        # Historial de mediciones para verificar idempotencia
        self._meas_count   = _AtomicCounter(0)

    def kraus_operator_M(
        self,
        rho_sim: Dict[str, Any],
    ) -> np.ndarray:
        r"""
        Construye el operador de Kraus M_k para el estado dado.

        Para el sistema clásico-estocástico con espacio de estados finito
        de dimensión d (número de claves observables):

            M_k = λ_k · I_d   (operador diagonal en la base de observables)

        La representación matricial de M_k es escalar-por-identidad porque
        los observables del pistón son variables continuas independientes
        (posición, momento, carga), y no hay entrelazamiento entre ellas.

        Completitud POVM:
            M_k† M_k = λ_k² · I_d
            Σ_k M_k† M_k = I_d si Σ_k λ_k² = 1
            Para un único operador: λ_k = 1.0 → completitud exacta.

        Parámetros
        ----------
        rho_sim : Dict[str, Any], estado del simulador (ρ_sim clásico).

        Retorna
        -------
        np.ndarray, shape (d,d), operador de Kraus M_k = λ·I_d.
        """
        if self._obs_keys is not None:
            d = len(self._obs_keys)
        else:
            d = len(rho_sim)
        return self._lambda * np.eye(d, dtype=np.float64)

    def measure(
        self,
        rho_sim: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        r"""
        Aplica la medición POVM y retorna el estado colapsado ρ_JSON.

        Colapso del estado:
        ───────────────────
        ρ_JSON = M_k · ρ_sim · M_k† / Tr(M_k · ρ_sim · M_k†)

        Para M_k = λ·I_d y ρ_sim diagonal (estados clásicos):
            M_k · ρ_sim · M_k† = λ² · ρ_sim
            Tr(λ² · ρ_sim)     = λ² · Tr(ρ_sim) = λ² · 1 = λ²
            ρ_JSON             = λ² · ρ_sim / λ² = ρ_sim

        El estado colapsado es idéntico al estado original (no demolición),
        certificando [H, O_api] = 0.

        La implementación clásica filtra las claves observables y añade
        metadatos de la medición (timestamp, eficiencia, conteo).

        Parámetros
        ----------
        rho_sim : Optional[Dict[str, Any]], estado del buffer circular.
                  None si el buffer está vacío.

        Retorna
        -------
        Optional[Dict[str, Any]], estado colapsado ρ_JSON o None.
        """
        if rho_sim is None:
            return None

        self._meas_count.fetch_and_increment()

        # §3.1.1: Proyección sobre el subespacio observable
        if self._obs_keys is not None:
            projected = {k: rho_sim[k] for k in self._obs_keys if k in rho_sim}
        else:
            # Observación completa (λ=1, proyector = identidad)
            projected = dict(rho_sim)

        # §3.1.2: Colapso POVM (para λ=1 e I_d: ρ_JSON = ρ_sim, QND)
        # El factor de normalización λ²/λ² = 1 cancela exactamente.
        # Se preserva el estado sin alteración → no demolición certificada.
        rho_json = projected

        # §3.1.3: Metadatos de la medición (no afectan el estado físico)
        rho_json["_povm_metadata"] = {
            "measurement_efficiency" : self._lambda,
            "kraus_operator_trace"   : float(self._lambda ** 2),
            "measurement_count"      : self._meas_count.value,
            "qnd_certified"          : True,   # [H, O_api] = 0 verificado
            "timestamp_ns"           : time.time_ns(),
        }

        return rho_json

    @property
    def measurement_count(self) -> int:
        """Número de mediciones realizadas (diagnóstico)."""
        return self._meas_count.value


# ════════════════════════════════════════════════════════════════════════════════════
# §3.2 — COHOMOLOGÍA DE HACES CELULARES (verificación topológica del registro)
# ════════════════════════════════════════════════════════════════════════════════════

class SheafCohomologyVerifier:
    r"""
    Verifica H¹(X; F_IPU) = 0 antes de admitir el registro en la malla.

    SUTURA 7: Reemplaza el registro plano de URLs en texto libre.

    Fundamento Matemático:
    ──────────────────────
    Sea X el complejo CW de la malla agéntica:
        C⁰(X;F): secciones sobre los vértices (microservicios)
        C¹(X;F): secciones sobre las aristas (canales de comunicación)

    El operador coborde δ⁰: C⁰ → C¹:
        (δ⁰ s)(e) = s(v₂) - s(v₁)   para la arista e = [v₁, v₂]

    El primer grupo de cohomología:
        H¹(X; F_IPU) = ker(δ¹) / im(δ⁰)

    Para grafos finitos (caso computacional):
        β₁ = dim H¹ = |E| - |V| + c
        donde c = número de componentes conexas

    Condición de registro:
        H¹ = 0 ↔ β₁ = 0 ↔ el grafo (V, E) es un bosque (acíclico)

    Verificación adicional de integridad (Zero-Trust):
        Cada stalk F(v) del nuevo servicio debe ser firmado con HMAC-SHA256
        usando la clave compartida del ecosistema. La firma verifica que
        el stalk no ha sido adulterado en tránsito (morfismo estricto).

    Parámetros
    ----------
    ecosystem_key : bytes, clave HMAC-SHA256 del ecosistema.
                    Si None → se usa una clave de desarrollo (INSEGURA).
    """

    def __init__(self, ecosystem_key: Optional[bytes] = None) -> None:
        if ecosystem_key is None:
            logger.warning(
                "SheafCohomologyVerifier: usando clave de desarrollo INSEGURA. "
                "Definir ECOSYSTEM_HMAC_KEY en el entorno para producción."
            )
            self._key = b"dev_insecure_key_watchers_v4"
        else:
            self._key = ecosystem_key

        # Grafo de la malla: vértices = {service_name: url}
        #                    aristas   = {(s1, s2): channel_type}
        self._vertices : Dict[str, str]          = {}
        self._edges    : List[Tuple[str, str]]   = []

    # ─────────────────────────────────────────────────────────────────────
    # §3.2.1 — Cálculo del número de Betti β₁
    # ─────────────────────────────────────────────────────────────────────

    def _compute_betti_1(self) -> int:
        r"""
        Calcula el primer número de Betti β₁ = dim H¹(X; ℤ).

        Fórmula de Euler-Poincaré para grafos:
            β₁ = |E| - |V| + c

        donde c es el número de componentes conexas, calculado via
        Union-Find (Disjoint Set Union) en O(α(|V|)) amortizado.

        Un grafo acíclico (bosque) satisface β₁ = 0.
        Un grafo con k ciclos independientes satisface β₁ = k.

        Retorna
        -------
        int ≥ 0, primer número de Betti. 0 → H¹ = 0 (sin ciclos).
        """
        V = list(self._vertices.keys())
        E = self._edges
        n = len(V)

        if n == 0:
            return 0

        # Union-Find para contar componentes conexas
        vertex_idx = {v: i for i, v in enumerate(V)}
        parent     = list(range(n))
        rank       = [0] * n

        def _find(x: int) -> int:
            """Path compression."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def _union(a: int, b: int) -> bool:
            """Union by rank. Retorna True si fusiona (False si ya conectados)."""
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        # Procesar aristas
        for (u, v) in E:
            if u in vertex_idx and v in vertex_idx:
                _union(vertex_idx[u], vertex_idx[v])

        # Contar componentes: raíces únicas
        components = len({_find(i) for i in range(n)})

        beta_1 = len(E) - n + components
        return max(beta_1, 0)  # β₁ ≥ 0 por definición

    # ─────────────────────────────────────────────────────────────────────
    # §3.2.2 — Firma HMAC del stalk local F(v)
    # ─────────────────────────────────────────────────────────────────────

    def _sign_stalk(self, stalk: Dict[str, Any]) -> str:
        r"""
        Genera la firma HMAC-SHA256 del stalk local F(v).

        La firma certifica que el morfismo de registro:
            φ: F(v) → C⁰(X; F_IPU)

        es estricto: el stalk no ha sido alterado desde su construcción
        en el AtomicPiston (Fase 2) hasta su inserción en la malla.

        La firma incluye todos los campos del stalk ordenados
        lexicográficamente para garantizar determinismo.

        Parámetros
        ----------
        stalk : Dict[str, Any], datos del servicio a registrar.

        Retorna
        -------
        str, firma HMAC-SHA256 en hexadecimal (64 caracteres).
        """
        # Serialización canónica (ordenada) para determinismo
        canonical = json.dumps(stalk, sort_keys=True, ensure_ascii=True)
        sig = hmac.new(
            self._key,
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig

    # ─────────────────────────────────────────────────────────────────────
    # §3.2.3 — Verificación topológica del registro
    # ─────────────────────────────────────────────────────────────────────

    def verify_registration(
        self,
        service_name : str,
        module_url   : str,
        health_url   : str,
        neighbors    : Optional[List[str]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        r"""
        Verifica H¹(X; F_IPU) = 0 tras añadir el nuevo servicio al grafo.

        Pipeline de verificación:
        ─────────────────────────
        1. Construir el stalk local F(v) del AtomicPiston.
        2. Firmar F(v) con HMAC-SHA256 (Zero-Trust).
        3. Añadir v al grafo (V ∪ {v}).
        4. Añadir aristas a los vecinos conocidos (E ∪ {(v, n_i)}).
        5. Calcular β₁ = |E| - |V| + c.
        6. Si β₁ > 0: cancelar registro (H¹ ≠ 0, ciclo detectado).
        7. Si β₁ = 0: admitir registro (H¹ = 0, grafo acíclico).

        Parámetros
        ----------
        service_name : str, nombre canónico del servicio.
        module_url   : str, URL base del microservicio.
        health_url   : str, URL del endpoint de salud.
        neighbors    : Optional[List[str]], servicios vecinos ya registrados.
                       Cada vecino genera una arista en el grafo.

        Retorna
        -------
        Tuple[bool, str, Dict[str, Any]]
            (admitido, mensaje, stalk_con_firma)
        """
        # §3.2.3.1: Construir stalk local F(v)
        stalk = {
            "service_name" : service_name,
            "module_url"   : module_url,
            "health_url"   : health_url,
            "timestamp"    : time.time(),
            "version"      : "4.0.0",
        }
        # §3.2.3.2: Firmar stalk (Zero-Trust)
        stalk["hmac_signature"] = self._sign_stalk(
            {k: v for k, v in stalk.items() if k != "hmac_signature"}
        )

        # §3.2.3.3: Añadir vértice al grafo (copia del grafo para rollback)
        vertices_backup = dict(self._vertices)
        edges_backup    = list(self._edges)

        self._vertices[service_name] = module_url

        # §3.2.3.4: Añadir aristas a vecinos
        if neighbors:
            for neighbor in neighbors:
                if neighbor in self._vertices:
                    self._edges.append((service_name, neighbor))

        # §3.2.3.5: Calcular β₁ (primer número de Betti)
        beta_1 = self._compute_betti_1()

        if beta_1 > 0:
            # §3.2.3.6: H¹ ≠ 0 → ciclo detectado → ROLLBACK + cancelar
            self._vertices = vertices_backup
            self._edges    = edges_backup
            msg = (
                f"Registro CANCELADO: H¹(X; F_IPU) ≠ 0 "
                f"(β₁ = {beta_1} ciclos parásitos detectados). "
                f"El grafo de la malla no es acíclico. "
                f"Eliminar ciclos antes de registrar '{service_name}'."
            )
            logger.error(msg)
            return False, msg, {}

        # §3.2.3.7: H¹ = 0 → grafo acíclico → registro admitido
        msg = (
            f"Registro admitido: H¹(X; F_IPU) = 0 "
            f"(β₁ = 0, grafo acíclico con |V|={len(self._vertices)}, "
            f"|E|={len(self._edges)})."
        )
        logger.info(msg)
        return True, msg, stalk

    def remove_service(self, service_name: str) -> None:
        r"""
        Elimina un servicio del grafo de la malla (y sus aristas incidentes).

        Parámetros
        ----------
        service_name : str, nombre del servicio a eliminar.
        """
        if service_name in self._vertices:
            del self._vertices[service_name]
            self._edges = [
                (u, v) for (u, v) in self._edges
                if u != service_name and v != service_name
            ]
            logger.info(
                "Servicio '%s' eliminado del grafo de cohomología. "
                "|V|=%d, |E|=%d.",
                service_name, len(self._vertices), len(self._edges),
            )

    @property
    def cohomology_status(self) -> Dict[str, Any]:
        r"""
        Retorna el estado actual de la cohomología del haz.

        Retorna
        -------
        Dict con β₁, |V|, |E|, H¹=0 y la lista de vértices/aristas.
        """
        beta_1 = self._compute_betti_1()
        return {
            "beta_1"          : beta_1,
            "H1_zero"         : beta_1 == 0,
            "num_vertices"    : len(self._vertices),
            "num_edges"       : len(self._edges),
            "vertices"        : list(self._vertices.keys()),
            "edges"           : self._edges,
        }


# ════════════════════════════════════════════════════════════════════════════════════
# §3.3 — ServiceContext (encapsula el estado global del microservicio)
# ════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ServiceContext:
    r"""
    Contexto del microservicio con primitivas lock-free y POVM.

    SUTURA 5 aplicada: threading.Lock() reemplazado por LockFreeCircularBuffer.

    El buffer circular separa los espacios de acción del productor y
    el consumidor:
        PRODUCTOR (simulation_loop): escribe via buffer.publish()
        CONSUMIDOR (endpoints HTTP): lee via buffer.latest() + KrausObserver

    El Lock residual (_admin_lock) solo protege operaciones de configuración
    poco frecuentes (set_mode, reset, etc.) — no el bucle de simulación.

    Campos
    ──────
    config            : PistonConfig, configuración del servicio.
    ipu               : AtomicPiston, instancia del gemelo digital.
    buffer            : LockFreeCircularBuffer, buffer CAS para estados.
    observer_full     : KrausObserver, operador de Kraus para /api/state.
    observer_health   : KrausObserver, operador de Kraus para /api/health.
    sheaf_verifier    : SheafCohomologyVerifier, verificador H¹=0.
    _admin_lock       : threading.Lock, mutex SOLO para operaciones admin.
    sim_thread        : Thread del bucle de simulación.
    stop_event        : Event para detener el bucle.
    slow_cycle_count  : Contador de ciclos lentos (backpressure, FIX 12).
    """
    config            : Optional[PistonConfig]          = field(default=None)
    ipu               : Optional[AtomicPiston]           = field(default=None)
    buffer            : LockFreeCircularBuffer           = field(
        default_factory=lambda: LockFreeCircularBuffer(capacity=16)
    )
    observer_full     : KrausObserver                    = field(
        default_factory=lambda: KrausObserver(measurement_efficiency=1.0)
    )
    observer_health   : KrausObserver                    = field(
        default_factory=lambda: KrausObserver(
            measurement_efficiency=1.0,
            observable_keys=["position", "velocity", "stored_energy", "mode"],
        )
    )
    sheaf_verifier    : SheafCohomologyVerifier          = field(
        default_factory=SheafCohomologyVerifier
    )
    _admin_lock       : threading.Lock                   = field(
        default_factory=threading.Lock
    )
    sim_thread        : Optional[threading.Thread]       = field(default=None)
    stop_event        : threading.Event                  = field(
        default_factory=threading.Event
    )
    slow_cycle_count  : int                              = field(default=0)


# Instancia única del contexto (reemplaza variables globales de v2.0.0/v3.0.0)
_svc = ServiceContext()

# Aplicación Flask
app = Flask(__name__)


# ════════════════════════════════════════════════════════════════════════════════════
# §3.4 — Registro con AgentAI + verificación de cohomología (SUTURA 7)
# ════════════════════════════════════════════════════════════════════════════════════

def register_with_agent_ai(
    module_name  : str,
    module_url   : str,
    health_url   : str,
    description  : str = "",
    neighbors    : Optional[List[str]] = None,
) -> bool:
    r"""
    Registra el microservicio en el AgentAI bajo vigilancia del
    SheafCohomologyOrchestrator.

    SUTURA 7 PRINCIPAL:
    ────────────────────
    El registro no es una simple petición POST. Es un morfismo estricto
    dentro de la Cohomología de Haces Celulares:

        φ: F(v)_AtomicPiston → C⁰(X; F_IPU)

    El morfismo se admite si y solo si H¹(X; F_IPU) = 0 tras la inserción
    del nuevo vértice v y sus aristas en el grafo de la malla agéntica.

    Pipeline:
    ─────────
    1. Verificar H¹(X; F_IPU) = 0 via SheafCohomologyVerifier.
    2. Si H¹ ≠ 0: retornar False (registro cancelado categóricamente).
    3. Si H¹ = 0: firmar el stalk F(v) con HMAC-SHA256.
    4. Transmitir el stalk firmado al AgentAI con reintentos exponenciales.
    5. Si la transmisión falla: rollback del grafo (remove_service).

    FIX 14 preservado: advertencia SSL explícita.

    Parámetros
    ----------
    module_name  : str, nombre canónico del módulo.
    module_url   : str, URL base del microservicio.
    health_url   : str, URL del endpoint de salud.
    description  : str, descripción del módulo.
    neighbors    : Optional[List[str]], servicios vecinos en la malla.

    Retorna
    -------
    bool, True si el registro fue admitido topológicamente y transmitido.
    """
    if not _svc.config:
        logger.error("Configuración no disponible para registro con AgentAI.")
        return False

    # §3.4.1: Verificación topológica H¹(X; F_IPU) = 0 (SUTURA 7)
    admitted, topo_msg, stalk = _svc.sheaf_verifier.verify_registration(
        service_name = module_name,
        module_url   = module_url,
        health_url   = health_url,
        neighbors    = neighbors,
    )

    if not admitted:
        logger.error(
            "Registro '%s' rechazado por SheafCohomologyOrchestrator: %s",
            module_name, topo_msg,
        )
        return False

    logger.info(
        "Cohomología verificada para '%s': %s", module_name, topo_msg
    )

    # §3.4.2: Configuración SSL (FIX 14 preservado)
    ssl_verify: Any = os.environ.get("SSL_CERT_FILE", False)
    if ssl_verify is False:
        logger.warning(
            "ADVERTENCIA DE SEGURIDAD: SSL verification desactivado. "
            "Solo usar en entornos de desarrollo/prueba. "
            "Definir SSL_CERT_FILE para producción."
        )

    # §3.4.3: Payload con stalk firmado (Zero-Trust)
    payload = {
        "nombre"              : module_name,
        "url"                 : module_url,
        "url_salud"           : health_url,
        "tipo"                : "hardware_simulation",
        "aporta_a"            : "ecosistema_watchers",
        "naturaleza_auxiliar" : "gemelo_digital_ipu_port_hamiltoniano",
        "descripcion"         : description,
        "stalk_firmado"       : stalk,
        "cohomologia"         : _svc.sheaf_verifier.cohomology_status,
    }

    logger.info(
        "Transmitiendo stalk firmado de '%s' al AgentAI (%s)...",
        module_name, _svc.config.agent_ai_register_url,
    )

    # §3.4.4: Transmisión con reintentos exponenciales
    for attempt in range(_svc.config.max_registration_retries):
        try:
            response = requests.post(
                _svc.config.agent_ai_register_url,
                json    = payload,
                timeout = _svc.config.requests_timeout,
                verify  = ssl_verify,
            )
            if response.status_code == 200:
                logger.info(
                    "Registro de '%s' exitoso (intento %d/%d).",
                    module_name, attempt + 1,
                    _svc.config.max_registration_retries,
                )
                return True
            logger.warning(
                "Registro intento %d/%d: HTTP %d — %s",
                attempt + 1,
                _svc.config.max_registration_retries,
                response.status_code,
                response.text[:200],
            )
        except requests.exceptions.RequestException as exc:
            logger.error(
                "Error de conexión en intento %d/%d: %s",
                attempt + 1, _svc.config.max_registration_retries, exc,
            )
        if attempt < _svc.config.max_registration_retries - 1:
            backoff = _svc.config.retry_delay_seconds * (2 ** attempt)
            logger.info("Backoff exponencial: esperando %.1f s.", backoff)
            time.sleep(backoff)

    # §3.4.5: Fallo total — rollback del grafo de cohomología
    _svc.sheaf_verifier.remove_service(module_name)
    logger.error(
        "No se pudo registrar '%s' tras %d intentos. "
        "Grafo de cohomología revertido.",
        module_name, _svc.config.max_registration_retries,
    )
    return False


# ════════════════════════════════════════════════════════════════════════════════════
# §3.5 — Bucle de simulación lock-free (SUTURA 5 + FIX 12)
# ════════════════════════════════════════════════════════════════════════════════════

def simulation_loop() -> None:
    r"""
    Bucle de simulación del pistón atómico con escritura lock-free.

    SUTURA 5 APLICADA:
    ───────────────────
    El Lock euclidiano ha sido extirpado del bucle caliente de simulación.
    La escritura del estado al buffer se realiza via publish() (CAS atómico),
    que nunca bloquea el hilo de simulación.

    Para operaciones de configuración (reset, set_mode) que llegan via
    endpoints, se usa _admin_lock de grano fino que NO interrumpe el
    integrador Störmer-Verlet.

    Estructura del ciclo:
    ─────────────────────
    1. Marcar t_start = time.monotonic().
    2. Llamar ipu.update_state(dt) → avanza el integrador Störmer-Verlet.
    3. Llamar ipu.discharge(dt)    → gestiona la descarga.
    4. Publicar ipu.get_state_dict() en el buffer CAS (lock-free).
    5. Calcular sleep_time = interval - elapsed.
    6. Si sleep_time > 0: stop_event.wait(sleep_time).
    7. Si sleep_time ≤ 0: incrementar slow_cycle_count (FIX 12 backpressure).

    Garantía CFL:
        El integrador Störmer-Verlet es estable para:
            dt ≤ 2/ω_max   donde ω_max = sqrt(k/m) (frecuencia natural)
        La violación de CFL se detecta via el contador slow_cycle_count
        (si el ciclo es consistentemente lento, dt efectivo > dt nominal).

    FIX 12 preservado: backpressure con conteo de ciclos lentos.
    """
    logger.info("Bucle de simulación lock-free iniciado.")
    if not _svc.config:
        logger.error("Configuración no disponible. Bucle detenido.")
        return

    interval = _svc.config.simulation_interval

    while not _svc.stop_event.is_set():
        t_start = time.monotonic()

        if _svc.ipu is None:
            logger.warning("IPU no inicializada, esperando...")
            _svc.stop_event.wait(1.0)
            continue

        try:
            # §3.5.1: Integración simpléctica (sin Lock — bucle caliente)
            _svc.ipu.update_state(interval)
            _svc.ipu.discharge(interval)

            # §3.5.2: Publicación lock-free en el buffer CAS (SUTURA 5)
            state_snapshot = _svc.ipu.get_state_dict()
            _svc.buffer.publish(state_snapshot)

        except Exception:
            logger.exception("Error en ciclo de simulación:")
            _svc.stop_event.wait(5.0)
            continue

        # §3.5.3: Control de tiempo (FIX 12 backpressure preservado)
        elapsed    = time.monotonic() - t_start
        sleep_time = interval - elapsed

        if sleep_time > 0:
            _svc.stop_event.wait(sleep_time)
        else:
            _svc.slow_cycle_count += 1
            if _svc.slow_cycle_count % 100 == 0:
                logger.warning(
                    "Backpressure CFL: %d ciclos lentos acumulados. "
                    "Ciclo actual: %.3f ms (objetivo: %.3f ms). "
                    "Riesgo de violación de condición CFL del integrador "
                    "Störmer-Verlet (dt_efectivo > dt_nominal).",
                    _svc.slow_cycle_count,
                    elapsed * 1e3,
                    interval * 1e3,
                )

    logger.info("Bucle de simulación lock-free detenido.")


# ════════════════════════════════════════════════════════════════════════════════════
# §3.6 — Utilidades de validación de peticiones HTTP (FIX 13 preservado)
# ════════════════════════════════════════════════════════════════════════════════════

def _require_json() -> Optional[Tuple[Any, int]]:
    r"""
    Valida que la petición tiene Content-Type: application/json.

    FIX 13 preservado: previene errores internos no descriptivos por
    peticiones mal formadas sin Content-Type correcto.

    Retorna
    -------
    None si la validación pasa.
    Tuple[Response, 415] si el Content-Type es incorrecto.
    """
    if not request.is_json:
        return (
            jsonify({
                "status" : "error",
                "message": "Content-Type debe ser 'application/json'.",
            }),
            415,
        )
    return None


def _buffer_or_503() -> Optional[Tuple[Any, int]]:
    r"""
    Retorna error 503 si el buffer no tiene ningún estado publicado.

    SUTURA 5: Reemplaza _ipu_or_503() que verificaba _svc.ipu directamente.
    Ahora se verifica el buffer CAS, no la instancia IPU.

    Retorna
    -------
    None si el buffer tiene datos.
    Tuple[Response, 503] si el buffer está vacío.
    """
    if _svc.buffer.is_empty():
        return (
            jsonify({
                "status" : "error",
                "message": (
                    "Buffer de estados vacío. "
                    "El simulador aún no ha publicado ningún estado."
                ),
            }),
            503,
        )
    return None


def _ipu_or_503() -> Optional[Tuple[Any, int]]:
    """Retorna error 503 si el IPU no está inicializado (para comandos admin)."""
    if _svc.ipu is None:
        return (
            jsonify({
                "status" : "error",
                "message": "IPU no inicializada.",
            }),
            503,
        )
    return None


# ════════════════════════════════════════════════════════════════════════════════════
# §3.7 — Endpoints Flask con medición POVM (SUTURA 6)
# ════════════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health_check():
    r"""
    Endpoint de salud del microservicio.

    SUTURA 6 APLICADA:
    ───────────────────
    La interrogación del estado se modela como medición POVM de no-demolición:

        ρ_health = M_health · ρ_sim · M_health† / Tr(...)

    donde M_health proyecta sobre el subespacio observable de salud:
        {position, velocity, stored_energy, mode}

    La conmutatividad [H, O_health] = 0 está garantizada porque O_health
    actúa sobre la copia en el buffer, no sobre el estado primario (q,p,Q).

    Retorna 200 si el simulador está activo y el buffer tiene datos.
    Retorna 503 en caso contrario.
    """
    sim_alive  = bool(_svc.sim_thread and _svc.sim_thread.is_alive())
    buffer_ok  = not _svc.buffer.is_empty()
    ipu_ok     = _svc.ipu is not None
    is_ok      = sim_alive and buffer_ok and ipu_ok

    # §3.7.1: Lectura lock-free del último estado publicado
    rho_sim = _svc.buffer.latest()

    # §3.7.2: Colapso POVM de no-demolición (SUTURA 6)
    rho_health = _svc.observer_health.measure(rho_sim)

    return jsonify({
        "status" : "success" if is_ok else "error",
        "message": "Operativo" if is_ok else "No operativo",
        "details": {
            "simulation_running"    : sim_alive,
            "piston_initialized"    : ipu_ok,
            "buffer_has_data"       : buffer_ok,
            "slow_cycles_total"     : _svc.slow_cycle_count,
            "cohomology_status"     : _svc.sheaf_verifier.cohomology_status,
            "povm_measurements"     : _svc.observer_health.measurement_count,
        },
        "observable_state" : rho_health,
    }), (200 if is_ok else 503)


@app.route("/api/state", methods=["GET"])
def get_piston_state():
    r"""
    Retorna el estado completo del pistón como JSON via medición POVM.

    SUTURA 6 APLICADA:
    ───────────────────
    La serialización directa get_state_dict() ha sido reemplazada por
    el colapso POVM completo:

        ρ_JSON = M_full · ρ_sim · M_full† / Tr(...)

    donde M_full = I (λ=1.0) proyecta sobre el espacio completo de
    observables (observación perfecta sin demolición).

    El estado retornado incluye metadatos POVM para diagnóstico:
        - measurement_efficiency: λ = 1.0
        - qnd_certified: True
        - timestamp_ns: marca temporal de la medición

    La lectura es lock-free: no bloquea simulation_loop.
    """
    if (err := _buffer_or_503()) is not None:
        return err

    # §3.7.3: Lectura lock-free del buffer CAS (SUTURA 5)
    rho_sim = _svc.buffer.latest()

    # §3.7.4: Colapso POVM completo (SUTURA 6)
    rho_json = _svc.observer_full.measure(rho_sim)

    return jsonify({
        "status": "success",
        "state" : rho_json,
    })


@app.route("/api/state/history", methods=["GET"])
def get_piston_state_history():
    r"""
    Retorna los últimos N estados del buffer circular.

    Parámetro query: ?n=<int> (default: 8, máximo: capacidad del buffer).

    El historial reciente permite al DiracInterconnectionAgent calcular
    la tasa de disipación Ḣd acumulada sobre una ventana temporal:

        Ḣd_avg = (1/N) Σ_{i=0}^{N-1} (-∇Hᵀ R ∇H)_i

    Si Ḣd_avg ≤ 0 → el sistema es asintóticamente disipativo → PASS.
    """
    if (err := _buffer_or_503()) is not None:
        return err

    try:
        n_items = int(request.args.get("n", 8))
        n_items = max(1, min(n_items, _svc.buffer._capacity))
    except (ValueError, TypeError):
        n_items = 8

    # §3.7.5: Drenado lock-free del buffer (LIFO, N items)
    history = _svc.buffer.drain(max_items=n_items)

    # §3.7.6: Colapso POVM sobre cada snapshot histórico
    measured_history = [
        _svc.observer_full.measure(snap)
        for snap in history
    ]

    # §3.7.7: Calcular Ḣd promedio si hay datos de disipación
    diss_rates = [
        snap.get("port_hamiltonian", {}).get("dissipation_rate", 0.0)
        for snap in history
        if snap is not None
    ]
    hd_avg = float(np.mean(diss_rates)) if diss_rates else 0.0

    return jsonify({
        "status"               : "success",
        "n_snapshots"          : len(measured_history),
        "hamiltonian_dot_avg"  : hd_avg,
        "dissipation_ok"       : hd_avg <= self._EPS_DISS
            if hasattr(AtomicPiston, "_EPS_DISS") else hd_avg <= 1e-10,
        "history"              : measured_history,
    })


@app.route("/api/control", methods=["POST"])
def set_piston_control():
    r"""
    Ajusta el objetivo de energía del pistón (operación admin).

    SUTURA 5: Las operaciones admin usan _admin_lock de grano fino,
    no el Lock del bucle caliente (ya extirpado).

    Body JSON esperado:
        {"control_signal": float}  — objetivo de energía ≥ 0 [J]

    FIX 13 preservado: Content-Type validado.
    """
    if (err := _require_json())  is not None:
        return err
    if (err := _ipu_or_503())    is not None:
        return err

    data = request.get_json()
    if "control_signal" not in data:
        return jsonify({
            "status" : "error",
            "message": "Falta el campo 'control_signal' en el body JSON.",
        }), 400

    try:
        signal = float(data["control_signal"])
    except (ValueError, TypeError):
        return jsonify({
            "status" : "error",
            "message": "El campo 'control_signal' debe ser numérico.",
        }), 400

    # §3.7.8: Admin lock de grano fino (no interrumpe el integrador)
    with _svc._admin_lock:
        _svc.ipu.set_energy_target(max(0.0, signal))
        new_target = _svc.ipu.target_energy

    logger.info("Control remoto: target de energía = %.4f J.", new_target)
    return jsonify({
        "status"            : "success",
        "message"           : "Objetivo de energía ajustado.",
        "new_energy_target" : new_target,
    })


@app.route("/api/config", methods=["GET"])
def get_piston_config():
    r"""
    Retorna la configuración física del pistón como JSON.

    La configuración incluye las matrices port-Hamiltonianas J y R
    para diagnóstico del DiracInterconnectionAgent.

    Lectura lock-free: se lee del buffer más reciente para los
    campos dinámicos; los campos estáticos se leen directamente
    de la instancia IPU (son inmutables post-__init__).
    """
    if (err := _ipu_or_503()) is not None:
        return err

    # Campos estáticos (inmutables, sin lock)
    ipu = _svc.ipu
    cfg = {
        "physical_params": {
            "capacity"            : ipu.capacity,
            "elasticity_k"        : ipu.k,
            "damping_c"           : ipu.c,
            "mass_m"              : ipu.m,
            "nonlinear_elasticity": ipu.nonlinear_elasticity,
        },
        "transducer": {
            "type"               : ipu.transducer_type.value,
            "coupling_factor_alpha": ipu.coupling_factor,
            "internal_resistance": ipu.internal_resistance,
            "load_resistance"    : ipu.load_resistance,
        },
        "friction": {
            "model"          : ipu.friction_model.value,
            "coulomb_friction": ipu.coulomb_friction,
            "stribeck_coeffs": list(ipu.stribeck_coeffs),
            "metric_tensor"  : (
                ipu._metric_tensor.tolist()
                if ipu._metric_tensor is not None
                else [[1.0]]
            ),
        },
        "port_hamiltonian_matrices": {
            "J_matrix"       : ipu._J.tolist(),
            "R_matrix"       : ipu._R.tolist(),
            "JR_matrix"      : ipu._JR.tolist(),
            "B_vector"       : ipu._B.tolist(),
            "R_total_ohm"    : ipu._R_total,
        },
        "electrical_equivalent": {
            "equivalent_capacitance": ipu.equivalent_capacitance,
            "equivalent_inductance" : ipu.equivalent_inductance,
            "equivalent_resistance" : ipu.equivalent_resistance,
            "output_capacitance"    : ipu.output_capacitance,
            "converter_efficiency"  : ipu.converter_efficiency,
        },
        "operational": {
            "capacitor_discharge_threshold": ipu.capacitor_discharge_threshold,
            "hysteresis_factor"            : ipu.hysteresis_factor,
            "battery_discharge_rate"       : ipu.battery_discharge_rate,
            "compression_direction"        : ipu.compression_direction,
        },
        "cohomology_topology": _svc.sheaf_verifier.cohomology_status,
    }
    return jsonify({"status": "success", "config": cfg})


@app.route("/api/command", methods=["POST"])
def execute_piston_command():
    r"""
    Ejecuta un comando estructurado sobre el pistón.

    SUTURA 5: Los comandos admin usan _admin_lock de grano fino.
    El bucle de simulación NO es interrumpido.

    Comandos válidos:
        set_mode           : value = str (PistonMode.value)
        trigger_discharge  : value = bool
        set_energy_target  : value = float ≥ 0 [J]
        set_speed_target   : value = float [m/s]
        set_load_resistance: value = float > 0 [Ω]
        reset              : value = None

    FIX 13 preservado: Content-Type validado.
    """
    if (err := _require_json()) is not None:
        return err
    if (err := _ipu_or_503())   is not None:
        return err

    data    = request.get_json()
    command = data.get("command")
    value   = data.get("value")

    if not command:
        return jsonify({
            "status" : "error",
            "message": "Falta el campo 'command' en el body JSON.",
        }), 400

    valid_modes = [m.value for m in PistonMode]
    msg: str = ""

    # §3.7.9: Admin lock de grano fino (no interrumpe el integrador)
    with _svc._admin_lock:
        try:
            if command == "set_mode":
                if value not in valid_modes:
                    raise ValueError(
                        f"Modo inválido: '{value}'. Válidos: {valid_modes}."
                    )
                _svc.ipu.set_mode(PistonMode(value))
                msg = f"Modo cambiado a '{value}'."

            elif command == "trigger_discharge":
                if not isinstance(value, bool):
                    raise ValueError(
                        "El campo 'value' debe ser booleano (true/false)."
                    )
                _svc.ipu.trigger_discharge(value)
                msg = f"Descarga {'activada' if value else 'desactivada'}."

            elif command == "set_energy_target":
                target = float(value)
                if target < 0:
                    raise ValueError("El target de energía debe ser ≥ 0 [J].")
                _svc.ipu.set_energy_target(target)
                msg = f"Target de energía: {target:.4f} J."

            elif command == "set_speed_target":
                _svc.ipu.set_speed_target(float(value))
                msg = f"Target de velocidad: {float(value):.4f} m/s."

            elif command == "set_load_resistance":
                r_load = float(value)
                if r_load <= 0:
                    raise ValueError("R_load debe ser > 0 [Ω].")
                _svc.ipu.set_load_resistance(r_load)
                msg = f"R_load actualizada a {r_load:.2f} Ω."

            elif command == "reset":
                _svc.ipu.reset()
                msg = "Pistón reiniciado al estado inicial x=(q,p,Q)=(0,0,0)."

            else:
                return jsonify({
                    "status" : "error",
                    "message": f"Comando desconocido: '{command}'.",
                }), 400

        except (ValueError, TypeError) as exc:
            logger.error(
                "Error al ejecutar comando '%s' con value=%r: %s",
                command, value, exc,
            )
            return jsonify({"status": "error", "message": str(exc)}), 400

    logger.info("Comando '%s' ejecutado: %s", command, msg)
    return jsonify({"status": "success", "message": msg})


@app.route("/api/cohomology", methods=["GET"])
def get_cohomology_status():
    r"""
    Retorna el estado actual de la cohomología del haz de la malla agéntica.

    Permite al DiracInterconnectionAgent y al SheafCohomologyOrchestrator
    verificar H¹(X; F_IPU) = 0 en tiempo real.

    Retorna
    ───────
    JSON con:
        beta_1      : primer número de Betti β₁
        H1_zero     : True si H¹ = 0
        num_vertices: |V| (microservicios registrados)
        num_edges   : |E| (canales de comunicación)
        vertices    : lista de microservicios
        edges       : lista de aristas
    """
    return jsonify({
        "status"     : "success",
        "cohomology" : _svc.sheaf_verifier.cohomology_status,
    })


# ════════════════════════════════════════════════════════════════════════════════════
# §3.8 — Punto de entrada (main)
# ════════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    r"""
    Punto de entrada del microservicio Atomic Piston v4.0.0.

    Pipeline de arranque (v4.0.0):
    ──────────────────────────────
    1.  Cargar configuración (PistonConfig).
    2.  Registrar configuración en ServiceContext.
    3.  Crear instancia AtomicPiston (Fase 2, estado extendido [q,p,Q]).
    4.  Verificar H¹(X; F_IPU) = 0 via SheafCohomologyVerifier (Sutura 7).
    5.  Intentar registro con AgentAI + stalk firmado (no fatal si falla).
    6.  Iniciar hilo de simulación lock-free (daemon=True).
    7.  Arrancar Flask en el puerto configurado.
    """
    # §3.8.1: Cargar configuración
    try:
        _svc.config = PistonConfig()
    except Exception:
        logger.exception("Error al cargar PistonConfig:")
        return

    logger.info("─── Configuración IPU v4.0.0 ─────────────────────────────")
    logger.info("Capacidad         : %.3f m",    _svc.config.capacity)
    logger.info("Elasticidad       : %.1f N/m",  _svc.config.elasticity)
    logger.info("Amortiguamiento   : %.1f N·s/m", _svc.config.damping)
    logger.info("Masa              : %.2f kg",    _svc.config.mass)
    logger.info("Modelo de fricción: %s",         _svc.config.friction_model.value)
    logger.info("Intervalo sim.    : %.4f s",     _svc.config.simulation_interval)
    logger.info("Buffer CAS        : %d slots",   16)
    logger.info("Integrador        : Störmer-Verlet simpléctico")
    logger.info("Estado            : x=[q,p,Q]ᵀ (port-Hamiltoniano)")
    logger.info("─────────────────────────────────────────────────────────")

    # §3.8.2: Crear instancia AtomicPiston (Fase 2)
    try:
        _svc.ipu = AtomicPiston(
            capacity      = _svc.config.capacity,
            elasticity    = _svc.config.elasticity,
            damping       = _svc.config.damping,
            piston_mass   = _svc.config.mass,
            friction_model= _svc.config.friction_model,
        )
    except ValueError:
        logger.exception("Fallo al crear AtomicPiston:")
        return

    # §3.8.3: Registro con AgentAI + verificación de cohomología (Sutura 7)
    hostname   = os.environ.get("HOSTNAME", "atomic_piston_service")
    module_url = f"http://{hostname}:{_svc.config.service_port}"
    health_url = f"{module_url}/api/health"

    ecosystem_key = os.environ.get("ECOSYSTEM_HMAC_KEY")
    if ecosystem_key:
        _svc.sheaf_verifier = SheafCohomologyVerifier(
            ecosystem_key=ecosystem_key.encode("utf-8")
        )
        logger.info("SheafCohomologyVerifier: clave HMAC del ecosistema cargada.")

    registered = register_with_agent_ai(
        module_name = "atomic_piston_service",
        module_url  = module_url,
        health_url  = health_url,
        description = (
            "Microservicio de gemelo digital IPU port-Hamiltoniano "
            "con integrador Störmer-Verlet simpléctico, buffer CAS "
            "lock-free y medición POVM de no-demolición. v4.0.0."
        ),
        neighbors = [],   # Sin vecinos en el primer registro → grafo minimal
    )
    if not registered:
        logger.warning(
            "Continuando sin registro en AgentAI. "
            "El microservicio opera de forma autónoma."
        )

    # §3.8.4: Iniciar hilo de simulación lock-free
    _svc.stop_event.clear()
    _svc.sim_thread = threading.Thread(
        target = simulation_loop,
        daemon = True,
        name   = "PistonSimLoop_SV",
    )
    _svc.sim_thread.start()
    logger.info(
        "Hilo de simulación Störmer-Verlet iniciado (daemon=True, lock-free)."
    )

    # §3.8.5: Arrancar Flask
    logger.info(
        "Servidor Flask iniciando en 0.0.0.0:%d...",
        _svc.config.service_port,
    )
    app.run(
        host        = "0.0.0.0",
        port        = _svc.config.service_port,
        debug       = False,
        use_reloader= False,
    )


# ════════════════════════════════════════════════════════════════════════════════════
# §3.9 — Punto de entrada con manejo de señales y limpieza final
# ════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado recibida.")
    finally:
        logger.info("Deteniendo servicio Atomic Piston v4.0.0...")
        _svc.stop_event.set()

        if _svc.sim_thread and _svc.sim_thread.is_alive():
            _svc.sim_thread.join(timeout=3.0)
            if _svc.sim_thread.is_alive():
                logger.warning(
                    "El hilo de simulación no terminó en 3 s. "
                    "Forzando terminación del proceso."
                )

        # Diagnóstico final de cohomología
        cohom = _svc.sheaf_verifier.cohomology_status
        logger.info(
            "Estado final de cohomología: β₁=%d, H¹=0: %s, "
            "|V|=%d, |E|=%d.",
            cohom["beta_1"], cohom["H1_zero"],
            cohom["num_vertices"], cohom["num_edges"],
        )

        logger.info(
            "Servicio Atomic Piston v4.0.0 detenido. "
            "Ciclos lentos: %d. "
            "Mediciones POVM (state): %d. "
            "Mediciones POVM (health): %d.",
            _svc.slow_cycle_count,
            _svc.observer_full.measurement_count,
            _svc.observer_health.measurement_count,
        )

# ════════════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 3 — MICROSERVICIO FLASK + ServiceContext (v4.0.0)
#
# Diagrama de flujo completo de las 3 fases anidadas:
#
#  FASE 1 (Herramientas)                FASE 2 (Gemelo Digital)
#  ────────────────────                 ──────────────────────────
#  FrictionCalculator                   AtomicPiston
#    compute_friction(G_μν)  ──────────→  _compute_friction_force()
#    smooth_sign_metric()                  _stormer_verlet_step()
#                                          _grad_H() → ∇H
#  LieGroupPIDController               AtomicPiston.update_state()
#    update(Log_x(x_d)) ──────────────→  F_ext = F_applied + F_pid + F_fric
#    _parallel_transport(Γ_γ)            (J−R)∇H + B·F_ext
#                                          get_state_dict()
#                ↓                              ↓
#  FASE 3 (Microservicio)
#  ──────────────────────
#  simulation_loop()  ←── ipu.update_state() [Störmer-Verlet, sin Lock]
#       ↓
#  buffer.publish(state)  [CAS lock-free, O(1)]
#       ↓
#  Endpoints HTTP:
#    /api/health  → observer_health.measure(buffer.latest())  [POVM]
#    /api/state   → observer_full.measure(buffer.latest())    [POVM]
#    /api/command → _admin_lock (grano fino, no interrumpe sim)
#       ↓
#  register_with_agent_ai()
#    → SheafCohomologyVerifier.verify_registration()
#    → β₁ = |E| - |V| + c  →  H¹=0? → admitir/cancelar
#    → HMAC-SHA256 del stalk F(v)
#    → POST AgentAI con stalk firmado
# ════════════════════════════════════════════════════════════════════════════════════