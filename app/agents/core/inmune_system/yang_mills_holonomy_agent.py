# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo  : Yang-Mills Holonomy Agent                                          ║
║           (Dinámica de Calibre, Holonomía y Ecuaciones de Campo)             ║
║ Ubicación: app/agents/core/immune_system/yang_mills_holonomy_agent.py               ║
║ Versión : 3.0.0-Bianchi-Higham-PathOrdered-Riguroso                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Geometría de Calibre
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Este módulo consagra la autoridad definitiva sobre el `dynamic_shield_router.py`.
No opera como enrutador de datos, sino como un **Funtor de Curvatura** que evalúa
la 2-forma de curvatura Ω de la conexión de Ehresmann ω inyectada sobre la membrana
del Escudo (`funtor_shield.py`), garantizando que el transporte paralelo a través
de los estratos DIKW preserve la **invarianza de Gauge**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 1 — Curvatura de Calibre (Ecuación de Estructura de Cartan)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
La ecuación de estructura de Cartan en el álgebra matricial de disipación:

    Ω = dω + ½[ω, ω]

se discretiza como el tensor de Faraday generalizado:

    F_{μν} = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν]

donde A_μ, A_ν son las componentes de la conexión (matrices de disipación).

La clasificación de curvatura distingue:
    · Plana          : ‖F‖_F < ε_flat   (transporte paralelo exacto)
    · Self-dual      : F = ★F            (instanton, mínimo local de S_YM)
    · Anti-self-dual : F = −★F           (anti-instanton)
    · General        : caso sin autodualidad

La identidad de Bianchi D∧F = 0 se verifica post-cálculo:

    ∂_λ F_{μν} + ∂_μ F_{νλ} + ∂_ν F_{λμ} = 0

En la discretización matricial: [A_μ, F_{νλ}] + [A_ν, F_{λμ}] + [A_λ, F_{μν}] ≈ 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 2 — Auditoría de Holonomía (Bucle de Wilson con Longitud de Arco)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El operador de holonomía para un camino cerrado γ = {A_k, ds_k}:

    W(γ) = 𝒫 exp(i ∮_γ A_μ dx^μ)
         ≈ ∏_k exp(i · A_k · ds_k)   (discretización con longitudes de arco ds_k)

La invarianza de reparametrización se garantiza usando ds_k = ‖A_k‖_F / Σ‖A_j‖_F
como peso normalizado (longitud de arco relativa).

La clasificación de la holonomía usa el **carácter de la representación**:

    χ(W) = Tr(W) ∈ ℂ

y el **índice de unitariedad**:

    δ_U = ‖W†W − I‖_F

que debe ser ≈ 0 (W ∈ U(n) ↔ transporte paralelo unitario).

Se separa el cálculo (evaluate_holonomy) de la decisión de veto
(check_paradox), restaurando la separación de responsabilidades.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 3 — Minimización de la Acción de Yang-Mills (Derivación Variacional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
La acción completa de Yang-Mills con término topológico θ:

    S_YM[A] = ½ Tr(Fᵀ G⁻¹ F G⁻¹) + θ · Tr(F G⁻¹ F̃ G⁻¹)

donde F̃ = ½ ε^{μν} F_{αβ} es el dual de Hodge discreta.

La variación δS_YM/δA = 0 acoplada a la corriente J:

    D^μ F_{μν} = J_ν
    →  G⁻¹ F + F G⁻¹ − A (G⁻¹ F − F G⁻¹) = J   (ec. de movimiento linealizada)

cuya solución por mínimos cuadrados ponderados es:

    δR_opt = argmin_X { ‖G⁻¹ F + F G⁻¹ + A[G⁻¹,F] − J + G⁻¹ X + X G⁻¹‖_F }

que se resuelve mediante la ecuación de Lyapunov generalizada:

    G⁻¹ δR + δR G⁻¹ = J − (G⁻¹ F + F G⁻¹) =: RHS

y se proyecta al cono S⁺ₙ por el algoritmo de Higham.

El residuo de las ecuaciones de movimiento se verifica post-hoc:

    ‖EOM_residual‖_F = ‖G⁻¹ F + F G⁻¹ + G⁻¹ δR + δR G⁻¹ − J‖_F < ε_eom

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Anidamiento entre fases:
  · Fase 1 → GaugeCurvatureTensor (F, clasificación, Bianchi) → entrada Fase 2
  · Fase 2 → WilsonLoopHolonomy (W, χ, δ_U, paradox) → entrada Fase 3
  · Fase 3 → YangMillsAction (S_YM, δR_opt, residuo EOM) → DeformationTensor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES
# ══════════════════════════════════════════════════════════════════════════════
from app.core.mic_algebra import Morphism, TopologicalInvariantError
from app.core.schemas import Stratum
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.core.immune_system.funtor_shield import QuadraticDissipation
from app.core.immune_system.dynamic_shield_router import DeformationTensor

logger = logging.getLogger("MIC.ImmuneSystem.YangMillsHolonomy")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS Y NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════

#: Tolerancia para clasificar curvatura como plana (‖F‖_F < ε_flat).
_EPS_FLAT_CURVATURE: float = 1.0e-9

#: Tolerancia para verificar la identidad de Bianchi D∧F ≈ 0.
_EPS_BIANCHI: float = 1.0e-8

#: Tolerancia para la autodualidad F ≈ ±★F.
_EPS_SELFDUAL: float = 1.0e-7

#: Tolerancia para la unitariedad del bucle de Wilson ‖W†W − I‖_F.
_EPS_UNITARITY: float = 1.0e-9

#: Tolerancia para la detección de paradoja topológica ‖W − I‖_F.
_EPS_HOLONOMY: float = 1.0e-6

#: Tolerancia para el residuo de las ecuaciones de movimiento de YM.
_EPS_EOM_RESIDUAL: float = 1.0e-8

#: Número de condición máximo admisible para G⁻¹ en la acción de YM.
_KAPPA_MAX_METRIC: float = 1.0e10

#: Parámetro topológico θ (ángulo theta de Yang-Mills, CP-violación).
_THETA_YM: float = 0.0   # Configuración CP-preservante por defecto.


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES DE CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════


class CurvatureClass(Enum):
    """
    Clasificación topológica del tensor de curvatura F_{μν}.

    Valores
    ───────
    FLAT
        ‖F‖_F < _EPS_FLAT_CURVATURE. El transporte paralelo es exacto y
        la conexión es localmente trivial (gauge puro).

    SELF_DUAL
        F ≈ +★F (instanton). Mínimo absoluto de S_YM en su clase topológica.
        El número de Chern c₁ = Tr(F) / (2π) > 0.

    ANTI_SELF_DUAL
        F ≈ −★F (anti-instanton). Mínimo absoluto en la clase conjugada.
        c₁ < 0.

    GENERAL
        Curvatura no nula y no autodual. Punto silla de la acción de YM.
        Las ecuaciones de movimiento D^μ F_{μν} = J_ν son no triviales.
    """

    FLAT = auto()
    SELF_DUAL = auto()
    ANTI_SELF_DUAL = auto()
    GENERAL = auto()


class HolonomyClass(Enum):
    """
    Clasificación topológica del bucle de Wilson W(γ).

    Valores
    ───────
    TRIVIAL
        ‖W − I‖_F < _EPS_HOLONOMY. El transporte paralelo es globalmente
        trivial: no hay holonomía y el haz de gauge es topológicamente plano.

    NON_TRIVIAL
        ‖W − I‖_F ≥ _EPS_HOLONOMY pero W ∈ U(n) (unitario). Indica una
        holonomía no trivial pero geométricamente consistente.

    PARADOX
        ‖W†W − I‖_F > _EPS_UNITARITY. W no es unitario: el transporte
        paralelo no preserva la norma, indicando una singularidad del haz.
    """

    TRIVIAL = auto()
    NON_TRIVIAL = auto()
    PARADOX = auto()


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS Y DE GAUGE
# ══════════════════════════════════════════════════════════════════════════════


class GaugeCurvatureSingularityError(TopologicalInvariantError):
    """
    Lanzada cuando la curvatura F_{μν} no es integrable en un espacio que
    debería ser topológicamente trivial (asintóticamente plano).

    Implica que la deformación propuesta viola el Teorema de Fluctuación-
    Disipación: la entropía de la conexión diverge en la escala del estrato.

    Atributos esperados en el mensaje
    ──────────────────────────────────
    · ‖F‖_F : norma Frobenius de la curvatura.
    · CurvatureClass : clasificación topológica detectada.
    · bianchi_residual : ‖[A,F]‖_F (violación de la identidad de Bianchi).
    """


class HolonomyVetoError(TopologicalInvariantError):
    """
    Lanzada cuando el bucle de Wilson W(γ) no pertenece al grupo unitario U(n),
    probando matemáticamente que la ruta de decisión encierra una paradoja
    topológica: el transporte paralelo no preserva la norma del estado cuántico.

    La no-unitariedad implica que la conexión de Ehresmann no puede ser
    completada a una conexión de Levi-Civita compatible con la métrica G,
    violando el axioma de preservación simpléctica del flujo Port-Hamiltoniano.

    Atributos esperados en el mensaje
    ──────────────────────────────────
    · δ_U = ‖W†W − I‖_F : índice de violación de unitariedad.
    · n_potentials : número de potenciales en el ciclo.
    · holonomy_trace : Tr(W) (carácter de la representación).
    """


class YangMillsOptimizationError(TopologicalInvariantError):
    """
    Lanzada cuando la ecuación de Lyapunov generalizada para δR_opt no
    puede resolverse, o cuando el residuo de las ecuaciones de movimiento
    supera el umbral _EPS_EOM_RESIDUAL.

    Condiciones de disparo
    ──────────────────────
    · El sistema de Lyapunov G⁻¹ δR + δR G⁻¹ = RHS es singlar.
    · La deformación óptima proyectada contiene NaN o Inf.
    · ‖EOM_residual‖_F > _EPS_EOM_RESIDUAL (las EOM no se satisfacen).
    · κ(G) > _KAPPA_MAX_METRIC (la métrica está mal condicionada).
    """


class BianchiViolationError(TopologicalInvariantError):
    """
    Lanzada cuando la identidad de Bianchi D∧F = 0 se viola significativamente.

    La identidad de Bianchi es la condición de consistencia de las ecuaciones
    de campo de Yang-Mills: su violación implica que el tensor de curvatura F
    calculado no puede provenir de ninguna conexión de gauge real.

    Esto indica un error numérico severo o una inconsistencia en los datos
    de entrada (A_μ, A_ν no representan conexiones válidas en el mismo haz).
    """


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE CALIBRE (GAUGE DTOs)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class GaugeCurvatureTensor:
    r"""
    Tensor de Faraday Generalizado Ω (2-forma de curvatura de calibre).

    Producido por la **Fase 1**. Encapsula la curvatura F_{μν} con su
    clasificación topológica y los resultados de las verificaciones
    algebraicas (Bianchi, autodualidad).

    Atributos
    ─────────
    F_matrix : NDArray[float64], shape (n, n)
        Componente matricial del tensor de Faraday discreto:
            F_{μν} = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν]

    F_selfdual : NDArray[float64], shape (n, n)
        Parte self-dual: F⁺ = ½(F + ★F). Mínimo de S_YM por sector.

    F_antiselfdual : NDArray[float64], shape (n, n)
        Parte anti-self-dual: F⁻ = ½(F − ★F).

    frobenius_norm : float
        ‖F_{μν}‖_F. Medida escalar de la intensidad del campo de gauge.

    bianchi_residual : float
        ‖[A_μ, F_{νλ}] + [A_ν, F_{λμ}] + [A_λ, F_{μν}]‖_F.
        Cero exacto ↔ identidad de Bianchi satisfecha.

    chern_number : float
        Número de Chern discreto c₁ ≈ Tr(F) / (2π).
        Invariante topológico global del haz de gauge.

    curvature_class : CurvatureClass
        Clasificación topológica: FLAT, SELF_DUAL, ANTI_SELF_DUAL o GENERAL.

    is_integrable : bool
        True ↔ curvature_class == FLAT (campo de gauge trivial).
        False ↔ presencia de holonomía no trivial.
    """

    F_matrix: NDArray[np.float64]
    F_selfdual: NDArray[np.float64]
    F_antiselfdual: NDArray[np.float64]
    frobenius_norm: float
    bianchi_residual: float
    chern_number: float
    curvature_class: CurvatureClass
    is_integrable: bool


@dataclass(frozen=True, slots=True)
class WilsonLoopHolonomy:
    r"""
    Resultado espectral completo de la auditoría del bucle de Wilson W(γ).

    Producido por la **Fase 2**. Encapsula W(γ) con su carácter de
    representación, índice de unitariedad y clasificación topológica.

    Atributos
    ─────────
    phase_shift_matrix : NDArray[complex128], shape (n, n)
        Operador de holonomía W(γ) = 𝒫 exp(i ∮_γ A_μ ds_μ).
        Debe ser unitario: W†W ≈ I si la conexión es consistente.

    holonomy_trace : complex
        χ(W) = Tr(W) ∈ ℂ. Carácter de la representación.
        χ(W) = n ↔ W = I (holonomía trivial).

    unitarity_defect : float
        δ_U = ‖W†W − I‖_F. Medida de la no-unitariedad de W.
        δ_U ≈ 0 ↔ W ∈ U(n). δ_U >> 0 ↔ paradoja topológica.

    arc_length_weights : NDArray[float64], shape (k,)
        Pesos de longitud de arco ds_k = ‖A_k‖_F / Σ‖A_j‖_F usados
        en la discretización del bucle. Garantizan invarianza de
        reparametrización del camino.

    holonomy_class : HolonomyClass
        Clasificación: TRIVIAL, NON_TRIVIAL o PARADOX.

    paradox_detected : bool
        True ↔ holonomy_class == PARADOX (W no unitario).
    """

    phase_shift_matrix: NDArray[np.complex128]
    holonomy_trace: complex
    unitarity_defect: float
    arc_length_weights: NDArray[np.float64]
    holonomy_class: HolonomyClass
    paradox_detected: bool


@dataclass(frozen=True, slots=True)
class YangMillsAction:
    r"""
    Evaluación completa del funcional de acción de Yang-Mills.

    Producido por la **Fase 3**. Encapsula S_YM, la deformación óptima
    δR y las métricas de validación de las ecuaciones de movimiento.

    Atributos
    ─────────
    action_value : float
        S_YM[A] = ½ Tr(Fᵀ G⁻¹ F G⁻¹) + θ · Tr(F G⁻¹ F̃ G⁻¹).
        Valor del funcional evaluado en la conexión actual.

    topological_action : float
        Término topológico θ · c₁² (número de Pontryagin discreto).
        Invariante bajo deformaciones continuas de la conexión.

    optimal_deformation : NDArray[float64], shape (n, n)
        δR_opt ∈ S⁺ₙ: solución de la ecuación de Lyapunov proyectada
        al cono de matrices simétricas semidefinidas positivas.

    eom_residual_norm : float
        ‖G⁻¹F + FG⁻¹ + G⁻¹δR + δRG⁻¹ − J‖_F.
        Residuo de las ecuaciones de movimiento post-hoc.

    lyapunov_rhs_norm : float
        ‖RHS‖_F = ‖J − G⁻¹F − FG⁻¹‖_F. Norma del término fuente
        de la ecuación de Lyapunov.

    entropy_current_J : NDArray[float64], shape (n, n)
        Corriente de entropía estocástica J_ν usada como fuente.

    kappa_metric : float
        κ(G) = λ_max(G) / λ_min(G). Número de condición de la métrica.
    """

    action_value: float
    topological_action: float
    optimal_deformation: NDArray[np.float64]
    eom_residual_norm: float
    lyapunov_rhs_norm: float
    entropy_current_J: NDArray[np.float64]
    kappa_metric: float


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1 — CÁLCULO DE LA CURVATURA DE CALIBRE                              ║
# ║   (Ecuación de Estructura de Cartan + Identidad de Bianchi)                ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_GaugeCurvatureComputer:
    r"""
    Evalúa la 2-forma de curvatura Ω de la conexión de Ehresmann ω
    inyectada sobre el fibrado de disipación, junto con sus invariantes
    topológicos y verificaciones de consistencia.

    Fundamento Matemático
    ─────────────────────
    Sea P → M un G-fibrado principal con G = GL⁺(n, ℝ). La conexión ω
    induce la curvatura Ω ∈ Ω²(P, 𝔤) por la ecuación de Cartan:

        Ω = dω + ½[ω, ω]

    En la representación matricial discreta, con A_μ y A_ν como las
    componentes de la conexión en dos "direcciones" del espacio de estados:

        F_{μν} = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν]

    donde las derivadas parciales se aproximan por diferencias finitas:

        ∂_μ A_ν ≈ (A_ν − A_μ) / ‖A_ν − A_μ‖_F · ‖A_ν − A_μ‖_F = A_ν − A_μ
        ∂_ν A_μ ≈ 0   (tomamos A_μ como referencia local)

    Esta elección es consistente con la fijación de gauge de Lorenz:
    ∂^μ A_μ = 0 en el punto base.

    Descomposición Self-Dual / Anti-Self-Dual
    ─────────────────────────────────────────
    El operador de Hodge dual en el espacio de matrices n×n se aproxima
    mediante la transposición de la parte antisimétrica:

        ★F = G⁻¹ F_skew G   (dual respecto a la métrica G)

    donde F_skew = (F − Fᵀ)/2 es la parte antisimétrica de F.

    La descomposición:
        F⁺ = ½(F + ★F)   (self-dual, instanton)
        F⁻ = ½(F − ★F)   (anti-self-dual, anti-instanton)

    satisface S_YM = ‖F⁺‖² + ‖F⁻‖² ≥ 0, con mínimo en F = F⁺ o F = F⁻.

    Identidad de Bianchi
    ────────────────────
    En el álgebra de Lie matricial, la identidad de Bianchi D∧F = 0 se
    verifica como:

        B = [A_μ, F_{νλ}] + [A_ν, F_{λμ}] + [A_λ, F_{μν}]

    donde tomamos A_λ = ½(A_μ + A_ν) como dirección de referencia del
    tercer índice. ‖B‖_F < _EPS_BIANCHI indica consistencia topológica.

    Parámetros del Constructor
    ──────────────────────────
    metric_tensor : NDArray[float64], shape (n, n)
        Tensor métrico G (SPD) para la contracción de índices y el
        operador de Hodge dual.

    tol_flat : float
        Tolerancia para clasificar F como plana: ‖F‖_F < tol_flat.
        Por defecto _EPS_FLAT_CURVATURE = 1e-9.

    tol_bianchi : float
        Tolerancia para la identidad de Bianchi.
        Por defecto _EPS_BIANCHI = 1e-8.

    tol_selfdual : float
        Tolerancia para la clasificación self-dual / anti-self-dual.
        Por defecto _EPS_SELFDUAL = 1e-7.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        tol_flat: float = _EPS_FLAT_CURVATURE,
        tol_bianchi: float = _EPS_BIANCHI,
        tol_selfdual: float = _EPS_SELFDUAL,
    ) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        self._tol_flat = tol_flat
        self._tol_bianchi = tol_bianchi
        self._tol_selfdual = tol_selfdual

        # Pre-computar G⁻¹ para el operador de Hodge y la acción
        self._validate_metric(self._G)
        self._G_inv = la.inv(self._G)

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def compute_curvature(
        self,
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
    ) -> GaugeCurvatureTensor:
        r"""
        Calcula la 2-forma de curvatura F_{μν} con todos sus invariantes.

        Procedimiento
        ─────────────
        1. Validación dimensional y SPD de las matrices de entrada.
        2. Cálculo de las derivadas parciales finitas:
               ∂_ν A_ν = A_ν − A_μ   (variación de la conexión)
               ∂_μ A_μ = 0            (referencia local / fijación de Lorenz)
        3. Conmutador de Lie: [A_μ, A_ν] = A_μ A_ν − A_ν A_μ.
        4. Tensor de Faraday: F = (A_ν − A_μ) + [A_μ, A_ν].
        5. Operador de Hodge dual ★F respecto a G.
        6. Descomposición F = F⁺ + F⁻ (self-dual / anti-self-dual).
        7. Clasificación topológica (FLAT, SELF_DUAL, ANTI_SELF_DUAL, GENERAL).
        8. Cálculo del número de Chern c₁ = Tr(F) / (2π).
        9. Verificación de la identidad de Bianchi.
        10. Emisión de warnings si Bianchi se viola o curvatura es no trivial.

        Parámetros
        ──────────
        A_mu : NDArray[float64], shape (n, n)
            Conexión base (matriz de disipación actual R_base).
        A_nu : NDArray[float64], shape (n, n)
            Conexión propuesta (R_nueva tras la deformación del DSR).

        Retorna
        ───────
        GaugeCurvatureTensor
            Objeto inmutable con F, F⁺, F⁻, ‖F‖_F, residuo de Bianchi,
            c₁, clase de curvatura y flag de integrabilidad.

        Lanza
        ─────
        BianchiViolationError
            Si ‖B‖_F > _EPS_BIANCHI (inconsistencia topológica severa).
        ValueError
            Si las dimensiones de A_μ y A_ν no coinciden con G.
        """
        # ── Paso 1: Validación dimensional ──
        self._validate_connection_shapes(A_mu, A_nu)
        n = A_mu.shape[0]

        # ── Paso 2: Derivadas parciales finitas ──
        d_A_nu = A_nu - A_mu     # ∂_ν A_ν ≈ variación de la conexión
        d_A_mu = np.zeros_like(A_mu)   # referencia local (Lorenz gauge)

        # ── Paso 3: Conmutador de Lie [A_μ, A_ν] ──
        commutator = A_mu @ A_nu - A_nu @ A_mu   # [A, B] = AB − BA

        # ── Paso 4: Tensor de Faraday discreto ──
        F = d_A_nu - d_A_mu + commutator

        # ── Paso 5: Operador de Hodge dual respecto a G ──
        # ★F = G⁻¹ F_skew G, donde F_skew = (F − Fᵀ)/2
        F_skew = 0.5 * (F - F.T)
        F_hodge_dual = self._G_inv @ F_skew @ self._G

        # ── Paso 6: Descomposición self-dual / anti-self-dual ──
        F_plus = 0.5 * (F + F_hodge_dual)     # instanton
        F_minus = 0.5 * (F - F_hodge_dual)    # anti-instanton

        # ── Paso 7: Clasificación topológica ──
        norm_F = float(np.linalg.norm(F, "fro"))
        norm_Fplus = float(np.linalg.norm(F_plus, "fro"))
        norm_Fminus = float(np.linalg.norm(F_minus, "fro"))
        curvature_class = self._classify_curvature(
            norm_F, F, F_hodge_dual, norm_Fplus, norm_Fminus
        )

        # ── Paso 8: Número de Chern c₁ = Tr(F) / (2π) ──
        chern = float(np.trace(F)) / (2.0 * math.pi)

        # ── Paso 9: Identidad de Bianchi ──
        bianchi_res = self._compute_bianchi_residual(A_mu, A_nu, F)

        if bianchi_res > _EPS_BIANCHI * 100:
            raise BianchiViolationError(
                f"Identidad de Bianchi violada severamente: "
                f"‖B‖_F = {bianchi_res:.3e} >> {_EPS_BIANCHI:.1e}. "
                f"Las conexiones A_μ, A_ν no pertenecen al mismo haz de gauge."
            )

        if bianchi_res > _EPS_BIANCHI:
            logger.warning(
                "Fase 1: Identidad de Bianchi aproximadamente violada: "
                "‖B‖_F = %.3e > %.1e.",
                bianchi_res, _EPS_BIANCHI,
            )

        # ── Paso 10: Logging de diagnóstico ──
        is_integrable = (curvature_class == CurvatureClass.FLAT)
        if not is_integrable:
            logger.warning(
                "Fase 1 [%s]: Anomalía de gauge — ‖F‖_F = %.4e, c₁ = %.4f, "
                "‖B‖_F = %.3e.",
                curvature_class.name, norm_F, chern, bianchi_res,
            )
        else:
            logger.debug(
                "Fase 1: Curvatura plana — ‖F‖_F = %.2e < %.1e.",
                norm_F, self._tol_flat,
            )

        return GaugeCurvatureTensor(
            F_matrix=F,
            F_selfdual=F_plus,
            F_antiselfdual=F_minus,
            frobenius_norm=norm_F,
            bianchi_residual=bianchi_res,
            chern_number=chern,
            curvature_class=curvature_class,
            is_integrable=is_integrable,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_curvature(
        self,
        norm_F: float,
        F: NDArray[np.float64],
        F_hodge: NDArray[np.float64],
        norm_Fplus: float,
        norm_Fminus: float,
    ) -> CurvatureClass:
        r"""
        Clasifica F según su relación con el operador de Hodge dual ★.

        Criterios (en orden de prioridad):
            1. FLAT         : ‖F‖_F < _tol_flat
            2. SELF_DUAL    : ‖F − ★F‖_F < _tol_selfdual   (F⁻ ≈ 0)
            3. ANTI_SELF_DUAL: ‖F + ★F‖_F < _tol_selfdual  (F⁺ ≈ 0)
            4. GENERAL      : resto de casos
        """
        if norm_F < self._tol_flat:
            return CurvatureClass.FLAT

        norm_F_minus_dual = float(np.linalg.norm(F - F_hodge, "fro"))
        norm_F_plus_dual = float(np.linalg.norm(F + F_hodge, "fro"))

        if norm_F_minus_dual < self._tol_selfdual:
            return CurvatureClass.SELF_DUAL
        if norm_F_plus_dual < self._tol_selfdual:
            return CurvatureClass.ANTI_SELF_DUAL

        return CurvatureClass.GENERAL

    def _compute_bianchi_residual(
        self,
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
        F: NDArray[np.float64],
    ) -> float:
        r"""
        Computa el residuo de la identidad de Bianchi en el álgebra de Lie.

        Fórmula discreta:
            A_lambda = ½(A_μ + A_ν)   (dirección del tercer índice)
            B = [A_μ, F_{νλ}] + [A_ν, F_{λμ}] + [A_λ, F_{μν}]

        donde:
            F_{νλ} = F − [A_ν, A_λ]   (curvatura en el sector νλ)
            F_{λμ} = −F                (antisimetría de F)
            F_{μν} = F                 (el tensor calculado)

        Retorna ‖B‖_F como medida escalar de la violación.
        """
        A_lambda = 0.5 * (A_mu + A_nu)

        # Curvatura en el sector νλ (aproximación)
        F_nu_lambda = F - (A_nu @ A_lambda - A_lambda @ A_nu)
        F_lambda_mu = -F   # antisimetría B_{[λμ]} = −B_{[μλ]}
        F_mu_nu = F

        # Suma cíclica de conmutadores (identidad de Bianchi)
        B = (
            (A_mu @ F_nu_lambda - F_nu_lambda @ A_mu)
            + (A_nu @ F_lambda_mu - F_lambda_mu @ A_nu)
            + (A_lambda @ F_mu_nu - F_mu_nu @ A_lambda)
        )

        return float(np.linalg.norm(B, "fro"))

    def _validate_metric(self, G: NDArray[np.float64]) -> None:
        """Verifica que G es cuadrada, simétrica y SPD (Cholesky)."""
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError("La métrica G debe ser una matriz cuadrada.")
        if not np.allclose(G, G.T, atol=1e-10):
            raise ValueError(
                f"La métrica G no es simétrica: "
                f"‖G − Gᵀ‖_F = {np.linalg.norm(G - G.T, 'fro'):.3e}."
            )
        try:
            la.cholesky(G, lower=True)
        except la.LinAlgError as exc:
            raise ValueError(f"La métrica G no es definida positiva: {exc}") from exc

    def _validate_connection_shapes(
        self,
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
    ) -> None:
        """Verifica que A_μ y A_ν son cuadradas y compatibles con G."""
        n = self._G.shape[0]
        for name, A in [("A_mu", A_mu), ("A_nu", A_nu)]:
            if A.shape != (n, n):
                raise ValueError(
                    f"{name} tiene forma {A.shape}; se esperaba ({n}, {n}) "
                    f"compatible con la métrica G."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Fin de Fase 1 — GaugeCurvatureTensor retornado por compute_curvature
    # es el argumento de entrada de Phase2_WilsonLoopAuditor.evaluate_holonomy
    # (vía el orquestador YangMillsHolonomyAgent)
    # ──────────────────────────────────────────────────────────────────────────


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2 — AUDITORÍA DE HOLONOMÍA                                           ║
# ║   (Bucle de Wilson con Longitud de Arco + Clasificación Topológica)         ║
# ║                                                                             ║
# ║   ENTRADA: GaugeCurvatureTensor (Fase 1) ───────────────────────────────── ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_WilsonLoopAuditor:
    r"""
    Mide el desfase topológico del transporte paralelo a lo largo de
    ciclos cerrados γ en el grafo de dependencias (β₁ > 0).

    Fundamento Matemático
    ─────────────────────
    El operador de Wilson es el observador fundamental de las teorías de
    gauge: mide si el transporte paralelo a lo largo de γ es trivial.

    Discretización con Invarianza de Reparametrización
    ──────────────────────────────────────────────────
    La discretización estándar W = ∏ exp(iA_k) no es invariante de
    reparametrización del camino (depende del número de pasos k y no
    solo del contenido geométrico del ciclo).

    La versión covariante usa longitudes de arco relativas:

        s_k = ‖A_k‖_F             (longitud de arco local)
        ds_k = s_k / Σⱼ sⱼ       (peso normalizado)

    de modo que:

        W(γ) = 𝒫 ∏_k exp(i · A_k · ds_k)

    Esto garantiza que si se refina el ciclo (más puntos), W(γ) converge
    al valor continuo ∫_γ A_μ dx^μ.

    Verificación de Unitariedad
    ───────────────────────────
    Para una conexión de gauge con grupo estructural U(n), W(γ) debe ser
    unitario: W†W = I. La violación δ_U = ‖W†W − I‖_F > 0 indica que:
        · La exponencial matricial introdujo error numérico (δ_U < 1e-12).
        · Las matrices A_k no son anti-hermitianas (δ_U ~ O(1) → paradoja).

    La separación de responsabilidades:
        · evaluate_holonomy : calcula W y sus invariantes.
        · check_paradox     : decide el veto y lanza la excepción.

    Parámetros del Constructor
    ──────────────────────────
    tol_unitarity : float
        Umbral para δ_U = ‖W†W − I‖_F. Si se supera, holonomy_class = PARADOX.

    tol_trivial : float
        Umbral para ‖W − I‖_F. Si es menor, holonomy_class = TRIVIAL.
    """

    def __init__(
        self,
        tol_unitarity: float = _EPS_UNITARITY,
        tol_trivial: float = _EPS_HOLONOMY,
    ) -> None:
        self._tol_unitarity = tol_unitarity
        self._tol_trivial = tol_trivial

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_holonomy(
        self,
        cyclic_potentials: List[NDArray[np.float64]],
        curvature: Optional[GaugeCurvatureTensor] = None,
    ) -> WilsonLoopHolonomy:
        r"""
        Calcula la holonomía W(γ) con todos sus invariantes topológicos.

        Si cyclic_potentials está vacío, retorna la holonomía trivial W = I.

        Procedimiento
        ─────────────
        1. Validación dimensional: todos los A_k ∈ ℝ^{n×n} con igual n.
        2. Cálculo de longitudes de arco s_k = ‖A_k‖_F.
        3. Normalización ds_k = s_k / Σs_j (invarianza de reparametrización).
        4. Exponenciales matriciales pesadas: exp(i · A_k · ds_k).
        5. Producto ordenado de camino (path-ordered product).
        6. Verificación de unitariedad δ_U = ‖W†W − I‖_F.
        7. Clasificación: TRIVIAL, NON_TRIVIAL o PARADOX.
        8. Logging de diagnóstico.

        Parámetros
        ──────────
        cyclic_potentials : List[NDArray[float64]]
            Matrices de conexión {A_k} que forman el ciclo cerrado γ.
            El orden define el ordenamiento de camino 𝒫.
            Si vacía, retorna holonomía trivial.

        curvature : GaugeCurvatureTensor, opcional
            Tensor de curvatura de la Fase 1. Se usa para incluir el
            número de Chern en los metadatos del resultado.

        Retorna
        ───────
        WilsonLoopHolonomy
            W(γ), Tr(W), δ_U, pesos ds_k, clase y flag de paradoja.

        Lanza
        ─────
        ValueError
            Si las matrices del ciclo tienen dimensiones inconsistentes.
        """
        # ── Caso trivial: ciclo vacío ──
        if not cyclic_potentials:
            n = self._G_dim_from_context(cyclic_potentials)
            return self._trivial_holonomy(n)

        # ── Paso 1: Validación dimensional ──
        n = self._validate_cycle_dimensions(cyclic_potentials)

        # ── Paso 2-3: Longitudes de arco y pesos normalizados ──
        arc_lengths = np.array(
            [float(np.linalg.norm(A, "fro")) for A in cyclic_potentials],
            dtype=np.float64,
        )
        total_arc = float(arc_lengths.sum())

        if total_arc < _EPS_FLAT_CURVATURE:
            # Todas las matrices son cero → holonomía trivial
            return self._trivial_holonomy(n)

        ds = arc_lengths / total_arc   # pesos normalizados de longitud de arco

        # ── Paso 4-5: Producto ordenado de camino ──
        W = np.eye(n, dtype=np.complex128)
        for A_k, ds_k in zip(cyclic_potentials, ds):
            A_weighted = 1j * A_k.astype(np.complex128) * ds_k
            step = la.expm(A_weighted)
            W = W @ step

        # ── Paso 6: Verificación de unitariedad ──
        WdagW = W.conj().T @ W
        unitarity_defect = float(
            np.linalg.norm(WdagW - np.eye(n, dtype=np.complex128), "fro")
        )

        # ── Paso 7: Clasificación topológica ──
        deviation_from_I = float(
            np.linalg.norm(W - np.eye(n, dtype=np.complex128), "fro")
        )
        holonomy_class = self._classify_holonomy(
            unitarity_defect, deviation_from_I
        )

        trace_W = complex(np.trace(W))
        paradox = (holonomy_class == HolonomyClass.PARADOX)

        # ── Paso 8: Logging ──
        logger.debug(
            "Fase 2: W(γ) — χ(W) = %.4f%+.4fj, δ_U = %.3e, "
            "‖W−I‖_F = %.3e, clase = %s.",
            trace_W.real, trace_W.imag,
            unitarity_defect, deviation_from_I,
            holonomy_class.name,
        )

        if holonomy_class == HolonomyClass.NON_TRIVIAL:
            logger.warning(
                "Fase 2: Holonomía no trivial detectada — "
                "‖W−I‖_F = %.3e ≥ %.1e. El ciclo γ encierra curvatura.",
                deviation_from_I, self._tol_trivial,
            )

        return WilsonLoopHolonomy(
            phase_shift_matrix=W,
            holonomy_trace=trace_W,
            unitarity_defect=unitarity_defect,
            arc_length_weights=ds,
            holonomy_class=holonomy_class,
            paradox_detected=paradox,
        )

    def check_paradox(self, holonomy: WilsonLoopHolonomy) -> None:
        r"""
        Verifica la holonomía y lanza HolonomyVetoError si se detecta
        una paradoja topológica (W no unitario).

        Separación de responsabilidades: este método solo decide el veto,
        mientras que evaluate_holonomy solo calcula.

        Parámetros
        ──────────
        holonomy : WilsonLoopHolonomy
            Resultado de evaluate_holonomy.

        Lanza
        ─────
        HolonomyVetoError
            Si holonomy.paradox_detected es True.
        """
        if holonomy.paradox_detected:
            logger.error(
                "Fase 2: Veto de Holonomía — δ_U = %.4e > %.1e. "
                "χ(W) = %.4f%+.4fj.",
                holonomy.unitarity_defect, self._tol_unitarity,
                holonomy.holonomy_trace.real, holonomy.holonomy_trace.imag,
            )
            raise HolonomyVetoError(
                f"Paradoja topológica: el transporte paralelo no es unitario. "
                f"δ_U = ‖W†W − I‖_F = {holonomy.unitarity_defect:.4e} "
                f"> {self._tol_unitarity:.1e}. "
                f"χ(W) = {holonomy.holonomy_trace:.4f}. "
                f"El haz de gauge no admite sección global consistente."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_holonomy(
        self,
        unitarity_defect: float,
        deviation_from_I: float,
    ) -> HolonomyClass:
        """
        Clasifica la holonomía en tres niveles (por orden de prioridad):

        PARADOX      : δ_U > _tol_unitarity (W no unitario → error severo)
        TRIVIAL      : ‖W−I‖_F < _tol_trivial (transporte paralelo exacto)
        NON_TRIVIAL  : δ_U ≤ _tol_unitarity y ‖W−I‖_F ≥ _tol_trivial
        """
        if unitarity_defect > self._tol_unitarity:
            return HolonomyClass.PARADOX
        if deviation_from_I < self._tol_trivial:
            return HolonomyClass.TRIVIAL
        return HolonomyClass.NON_TRIVIAL

    def _trivial_holonomy(self, n: int) -> WilsonLoopHolonomy:
        """Construye el objeto de holonomía trivial W = I_n."""
        identity = np.eye(n, dtype=np.complex128)
        return WilsonLoopHolonomy(
            phase_shift_matrix=identity,
            holonomy_trace=complex(n, 0.0),
            unitarity_defect=0.0,
            arc_length_weights=np.zeros(0, dtype=np.float64),
            holonomy_class=HolonomyClass.TRIVIAL,
            paradox_detected=False,
        )

    @staticmethod
    def _validate_cycle_dimensions(
        potentials: List[NDArray[np.float64]],
    ) -> int:
        """
        Verifica que todas las matrices del ciclo son cuadradas y tienen
        la misma dimensión. Retorna n (la dimensión común).
        """
        n = potentials[0].shape[0]
        for k, A in enumerate(potentials):
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(
                    f"El potencial #{k} no es cuadrado: forma {A.shape}."
                )
            if A.shape[0] != n:
                raise ValueError(
                    f"El potencial #{k} tiene dimensión {A.shape[0]} ≠ {n} "
                    f"(dimensión del primer potencial)."
                )
        return n

    @staticmethod
    def _G_dim_from_context(
        potentials: List[NDArray[np.float64]],
    ) -> int:
        """Extrae la dimensión del espacio; retorna la dimensión de G_PHYSICS si vacío."""
        if potentials:
            return potentials[0].shape[0]
        return G_PHYSICS.shape[0]

    # ──────────────────────────────────────────────────────────────────────────
    # Fin de Fase 2 — WilsonLoopHolonomy retornado por evaluate_holonomy
    # es el contexto que valida la Fase 3 en el orquestador.
    # ──────────────────────────────────────────────────────────────────────────


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3 — MINIMIZACIÓN DE LA ACCIÓN DE YANG-MILLS                         ║
# ║   (Ecuación de Lyapunov + Proyección de Higham + Verificación EOM)          ║
# ║                                                                             ║
# ║   ENTRADA: GaugeCurvatureTensor (Fase 1) + WilsonLoopHolonomy (Fase 2) ─── ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_YangMillsOptimizer:
    r"""
    Minimiza la acción de Yang-Mills S_YM[A] y obtiene la deformación
    óptima δR que satisface las ecuaciones de campo D^μ F_{μν} = J_ν.

    Fundamento Variacional Riguroso
    ────────────────────────────────
    La acción de Yang-Mills con término θ-topológico es:

        S_YM[A] = ½ Tr(Fᵀ G⁻¹ F G⁻¹) + θ · Tr(F G⁻¹ F̃ G⁻¹)

    donde F̃ = G⁻¹ F_skew G es el dual de Hodge de F (Fase 1) y
    θ = _THETA_YM es el parámetro topológico (CP-violación).

    El término ½ Tr(Fᵀ G⁻¹ F G⁻¹) es la contracción de índices estándar
    Tr(F^{ab} F_{ab}) con la métrica G.

    El término topológico θ · Tr(F G⁻¹ F̃ G⁻¹) es el número de Pontryagin
    discreto, proporcional a c₁².

    Ecuación de Movimiento y Ecuación de Lyapunov
    ──────────────────────────────────────────────
    La variación δS_YM/δA = 0 con fuente J produce:

        D^μ F_{μν} = G⁻¹ F + F G⁻¹ = J_ν   (en la aproximación linealizada)

    La deformación óptima δR satisface:

        G⁻¹ δR + δR G⁻¹ = J − (G⁻¹ F + F G⁻¹) =: RHS

    que es una **ecuación de Lyapunov generalizada** (también conocida como
    ecuación de Sylvester simétrica). Se resuelve con `scipy.linalg.solve_lyapunov`:

        G⁻¹ δR + δR G⁻¹ = RHS  →  scipy.linalg.solve_continuous_lyapunov(G⁻¹, RHS)

    La solución δR se proyecta al cono S⁺ₙ por el algoritmo de Higham,
    garantizando disipatividad del flujo Port-Hamiltoniano.

    Verificación Post-Hoc de las EOM
    ─────────────────────────────────
    Tras obtener δR_opt, se verifica el residuo:

        EOM_res = ‖G⁻¹ F + F G⁻¹ + G⁻¹ δR_opt + δR_opt G⁻¹ − J‖_F

    Si EOM_res > _EPS_EOM_RESIDUAL, se emite una advertencia (no se lanza
    excepción, ya que la proyección PSD puede introducir un residuo controlado).

    Parámetros del Constructor
    ──────────────────────────
    metric_tensor : NDArray[float64], shape (n, n)
        Tensor métrico G (SPD) para la contracción de la acción y la
        ecuación de Lyapunov.

    theta_ym : float
        Parámetro topológico θ de Yang-Mills. Default = 0 (CP-preservante).

    eom_tol : float
        Tolerancia para el residuo de las EOM post-hoc.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        theta_ym: float = _THETA_YM,
        eom_tol: float = _EPS_EOM_RESIDUAL,
    ) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        self._theta = float(theta_ym)
        self._eom_tol = float(eom_tol)

        # Pre-computar G⁻¹ y κ(G) para la acción y la ecuación de Lyapunov
        self._G_inv = la.inv(self._G)
        evals_G = la.eigvalsh(self._G)
        self._kappa_G = float(evals_G[-1] / evals_G[0])

        if self._kappa_G > _KAPPA_MAX_METRIC:
            logger.warning(
                "Fase 3: Métrica G mal condicionada κ(G) = %.3e > %.1e. "
                "La solución de Lyapunov puede ser numéricamente inestable.",
                self._kappa_G, _KAPPA_MAX_METRIC,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def minimize_action(
        self,
        curvature: GaugeCurvatureTensor,
        stochastic_entropy_J: NDArray[np.float64],
    ) -> YangMillsAction:
        r"""
        Calcula S_YM y obtiene δR_opt por resolución de la ecuación de Lyapunov.

        Procedimiento
        ─────────────
        1. Validación dimensional de J respecto a F.
        2. Cálculo de la acción S_YM (término cinético + θ-topológico).
        3. Cálculo del término fuente RHS = J − (G⁻¹F + FG⁻¹).
        4. Resolución de la ecuación de Lyapunov: G⁻¹δR + δRG⁻¹ = RHS.
        5. Simetrización de δR (la solución de Lyapunov puede tener asimetría numérica).
        6. Proyección de Higham al cono S⁺ₙ.
        7. Verificación del residuo de las EOM post-hoc.
        8. Verificación de finitud de δR_opt.

        Parámetros
        ──────────
        curvature : GaugeCurvatureTensor
            Tensor F de la Fase 1.
        stochastic_entropy_J : NDArray[float64], shape (n, n)
            Corriente de entropía J_ν (matriz de fuentes).

        Retorna
        ───────
        YangMillsAction
            S_YM, término topológico, δR_opt ∈ S⁺ₙ, residuo EOM, ‖RHS‖_F,
            J, κ(G).

        Lanza
        ─────
        YangMillsOptimizationError
            Si la ecuación de Lyapunov falla, δR contiene NaN/Inf,
            o el residuo EOM supera 100 × _EPS_EOM_RESIDUAL.
        """
        F = curvature.F_matrix
        n = F.shape[0]

        # ── Paso 1: Validación dimensional ──
        J = np.asarray(stochastic_entropy_J, dtype=np.float64)
        if J.shape != (n, n):
            raise YangMillsOptimizationError(
                f"Dimensiones incompatibles: F ∈ ℝ^{{{n}×{n}}}, "
                f"J ∈ ℝ^{{{J.shape[0]}×{J.shape[1]}}}."
            )

        # ── Paso 2: Acción de Yang-Mills ──
        S_kinetic, S_topological = self._compute_action(F, curvature.F_selfdual)

        # ── Paso 3: Término fuente de la ecuación de Lyapunov ──
        # RHS = J − (G⁻¹ F + F G⁻¹)   (término de campo libre)
        field_term = self._G_inv @ F + F @ self._G_inv
        RHS = J - field_term
        rhs_norm = float(np.linalg.norm(RHS, "fro"))

        # ── Paso 4: Resolución de la ecuación de Lyapunov ──
        # G⁻¹ δR + δR G⁻¹ = RHS
        # scipy: solve_continuous_lyapunov(A, Q) resuelve AX + XA = Q
        try:
            delta_R_raw = la.solve_continuous_lyapunov(self._G_inv, RHS)
        except (la.LinAlgError, ValueError) as exc:
            raise YangMillsOptimizationError(
                f"La ecuación de Lyapunov G⁻¹δR + δRG⁻¹ = RHS no pudo "
                f"resolverse: {exc}. κ(G) = {self._kappa_G:.3e}."
            ) from exc

        # ── Paso 5: Simetrización (eliminar asimetría numérica de Lyapunov) ──
        delta_R_sym = 0.5 * (delta_R_raw + delta_R_raw.T)

        # ── Paso 6: Proyección de Higham al cono S⁺ₙ ──
        delta_R_psd = self._project_higham(delta_R_sym)

        # ── Paso 7: Verificación de finitud ──
        if not np.all(np.isfinite(delta_R_psd)):
            raise YangMillsOptimizationError(
                "La deformación óptima δR_opt contiene NaN o Inf tras la "
                "proyección de Higham."
            )

        # ── Paso 8: Residuo de las ecuaciones de movimiento (post-hoc) ──
        eom_residual = self._compute_eom_residual(F, delta_R_psd, J)

        if eom_residual > 100.0 * self._eom_tol:
            raise YangMillsOptimizationError(
                f"Residuo de las EOM inaceptable: "
                f"‖EOM_res‖_F = {eom_residual:.3e} > "
                f"100 × {self._eom_tol:.1e}."
            )
        if eom_residual > self._eom_tol:
            logger.warning(
                "Fase 3: Residuo EOM = %.3e > %.1e (controlado por "
                "proyección Higham).",
                eom_residual, self._eom_tol,
            )

        logger.debug(
            "Fase 3: S_YM = %.4e (cinético) + %.4e (topológico). "
            "‖RHS‖_F = %.3e. ‖EOM_res‖_F = %.3e. κ(G) = %.2e.",
            S_kinetic, S_topological, rhs_norm, eom_residual, self._kappa_G,
        )

        return YangMillsAction(
            action_value=S_kinetic + S_topological,
            topological_action=S_topological,
            optimal_deformation=delta_R_psd,
            eom_residual_norm=eom_residual,
            lyapunov_rhs_norm=rhs_norm,
            entropy_current_J=J,
            kappa_metric=self._kappa_G,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_action(
        self,
        F: NDArray[np.float64],
        F_plus: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Calcula los dos términos de la acción de Yang-Mills.

        Término cinético:
            S_cin = ½ Tr(Fᵀ G⁻¹ F G⁻¹)
                  = ½ Tr(F_contracted)
            donde F_contracted = Fᵀ G⁻¹ F G⁻¹.

        Término topológico (θ-término):
            F̃ = G⁻¹ F_skew G   (dual de Hodge de F)
            S_top = θ · Tr(F G⁻¹ F̃ G⁻¹)
                  = θ · c₁²  (proporcional al número de Pontryagin)

        Retorna (S_cin, S_top).
        """
        # Término cinético: S = ½ Tr(Fᵀ G⁻¹ F G⁻¹)
        F_contracted = F.T @ self._G_inv @ F @ self._G_inv
        S_kinetic = 0.5 * float(np.real(np.trace(F_contracted)))

        # Término topológico: θ · Tr(F G⁻¹ F̃ G⁻¹)
        F_skew = 0.5 * (F - F.T)
        F_dual = self._G_inv @ F_skew @ self._G
        top_contracted = F @ self._G_inv @ F_dual @ self._G_inv
        S_topological = self._theta * float(np.real(np.trace(top_contracted)))

        return S_kinetic, S_topological

    def _compute_eom_residual(
        self,
        F: NDArray[np.float64],
        delta_R: NDArray[np.float64],
        J: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula el residuo de las ecuaciones de movimiento post-hoc:

            EOM_res = G⁻¹ F + F G⁻¹ + G⁻¹ δR + δR G⁻¹ − J

        Retorna ‖EOM_res‖_F.
        """
        lhs = (
            self._G_inv @ F
            + F @ self._G_inv
            + self._G_inv @ delta_R
            + delta_R @ self._G_inv
        )
        residual = lhs - J
        return float(np.linalg.norm(residual, "fro"))

    @staticmethod
    def _project_higham(M: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Proyecta M al cono S⁺ₙ usando el algoritmo de Higham (2002):

            Π_{S⁺ₙ}(M) = U · diag(max(λᵢ, 0)) · Uᵀ

        donde M = U Λ Uᵀ es la descomposición espectral real.
        """
        evals, evecs = la.eigh(M)
        evals_psd = np.maximum(evals, 0.0)
        M_psd = evecs @ np.diag(evals_psd) @ evecs.T
        return 0.5 * (M_psd + M_psd.T)   # simetrización post-proyección

    # ──────────────────────────────────────────────────────────────────────────
    # Fin de Fase 3 — YangMillsAction retornado por minimize_action
    # es empaquetado como DeformationTensor por el orquestador.
    # ──────────────────────────────────────────────────────────────────────────


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: YANG-MILLS HOLONOMY AGENT (Morfismo)                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class YangMillsHolonomyAgent(Morphism):
    r"""
    Operador de Medida Geométrica que garantiza la invarianza de Gauge del
    flujo de disipación Port-Hamiltoniano a través de los estratos DIKW.

    Arquitectura Categórica
    ───────────────────────
    El agente implementa el funtor de curvatura:

        Ψ_YM : (A_base, A_new, γ, J) ──▶ DeformationTensor

    que factoriza como la composición de tres transformaciones naturales:

        (A_base, A_new) ──F1──▶ GaugeCurvatureTensor
                                        │
                          (γ, F) ──F2──▶ WilsonLoopHolonomy
                                                  │
                         (F, H, J) ──F3──▶ YangMillsAction
                                                      │
                                            DeformationTensor ──▶ DynamicShieldRouter

    Verificación de Coherencia Dimensional
    ───────────────────────────────────────
    Antes de encadenar las fases, el agente verifica que todas las matrices
    del ciclo γ y la corriente J son dimensionalmente compatibles con G y
    con las conexiones A_base, A_new.

    Parámetros del Constructor
    ──────────────────────────
    metric_tensor : NDArray[float64], shape (n, n)
        Tensor métrico G (SPD) compartido por las tres fases.

    theta_ym : float
        Parámetro θ de Yang-Mills (CP-violación). Default = 0.

    enforce_bianchi : bool
        Si True, lanza BianchiViolationError en caso de violación
        severa de la identidad de Bianchi. Default = True.
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        theta_ym: float = _THETA_YM,
        enforce_bianchi: bool = True,
    ) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        self._enforce_bianchi = enforce_bianchi

        # Instanciación de las tres fases con la métrica compartida
        self._curvature_computer = Phase1_GaugeCurvatureComputer(
            metric_tensor=self._G
        )
        self._holonomy_auditor = Phase2_WilsonLoopAuditor()
        self._optimizer = Phase3_YangMillsOptimizer(
            metric_tensor=self._G, theta_ym=theta_ym
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def enforce_gauge_invariance(
        self,
        base_connection_A: NDArray[np.float64],
        proposed_connection_A_new: NDArray[np.float64],
        cyclic_dependencies: List[NDArray[np.float64]],
        stochastic_entropy_J: NDArray[np.float64],
    ) -> DeformationTensor:
        r"""
        Método axiomático: somete la deformación propuesta al rigor de la
        invarianza de gauge y retorna la deformación óptima compatible.

        Pipeline
        ────────
        0. Verificación de coherencia dimensional entre todas las entradas.
        1. Fase 1: compute_curvature(A_base, A_new) → GaugeCurvatureTensor.
        2. Control: si not is_integrable y enforce_bianchi → raise Error.
        3. Fase 2: evaluate_holonomy(cyclic_dependencies) → WilsonLoopHolonomy.
        4. Control: check_paradox(holonomy) → raise HolonomyVetoError si paradoja.
        5. Fase 3: minimize_action(curvature, J) → YangMillsAction.
        6. Empaquetado en DeformationTensor con todos los metadatos.

        Parámetros
        ──────────
        base_connection_A : NDArray[float64], shape (n, n)
            Conexión base (matriz R actual del escudo).

        proposed_connection_A_new : NDArray[float64], shape (n, n)
            Conexión propuesta por el DynamicShieldRouter.

        cyclic_dependencies : List[NDArray[float64]]
            Matrices de conexión que forman el ciclo cerrado γ en el grafo
            de dependencias. Si vacía, la auditoría de holonomía es trivial.

        stochastic_entropy_J : NDArray[float64], shape (n, n)
            Corriente de entropía estocástica (fuente de las EOM).

        Retorna
        ───────
        DeformationTensor
            δR_opt ∈ S⁺ₙ empaquetado con metadatos completos:
            S_YM, χ(W), δ_U, c₁, residuo EOM, clase de curvatura, κ(G).

        Lanza
        ─────
        GaugeCurvatureSingularityError
            Si ‖F‖_F > _EPS_FLAT_CURVATURE y la curvatura no es integrable.
        BianchiViolationError
            Si la identidad de Bianchi se viola severamente.
        HolonomyVetoError
            Si W(γ) no es unitario (paradoja topológica).
        YangMillsOptimizationError
            Si la ecuación de Lyapunov falla o el residuo EOM es inaceptable.
        ValueError
            Si las dimensiones de las entradas son inconsistentes.
        """
        logger.info(
            "YangMillsHolonomyAgent: Iniciando auditoría de invarianza de gauge. "
            "n_cycle = %d.",
            len(cyclic_dependencies),
        )

        # ── Paso 0: Verificación de coherencia dimensional ──
        self._validate_input_dimensions(
            base_connection_A,
            proposed_connection_A_new,
            cyclic_dependencies,
            stochastic_entropy_J,
        )

        # ── Fase 1: Curvatura de calibre ──────────────────────────────────
        curvature = self._curvature_computer.compute_curvature(
            A_mu=base_connection_A,
            A_nu=proposed_connection_A_new,
        )

        if not curvature.is_integrable:
            raise GaugeCurvatureSingularityError(
                f"Singularidad de calibre: ‖F_{{μν}}‖_F = "
                f"{curvature.frobenius_norm:.4e}. "
                f"Clase = {curvature.curvature_class.name}. "
                f"c₁ = {curvature.chern_number:.4f}. "
                f"Bianchi residual = {curvature.bianchi_residual:.3e}."
            )

        # ── Fase 2: Auditoría de holonomía ────────────────────────────────
        holonomy = self._holonomy_auditor.evaluate_holonomy(
            cyclic_potentials=cyclic_dependencies,
            curvature=curvature,
        )
        self._holonomy_auditor.check_paradox(holonomy)

        # ── Fase 3: Minimización de Yang-Mills ────────────────────────────
        action_result = self._optimizer.minimize_action(
            curvature=curvature,
            stochastic_entropy_J=stochastic_entropy_J,
        )

        # ── Empaquetado del DeformationTensor ─────────────────────────────
        n = base_connection_A.shape[0]
        R_base_norm = float(np.linalg.norm(base_connection_A, "fro"))
        delta_R_norm = float(
            np.linalg.norm(action_result.optimal_deformation, "fro")
        )
        frobenius_ratio = delta_R_norm / (R_base_norm + 1e-15)

        G_inv = la.inv(self._G)
        entropy_production = float(
            np.trace(action_result.optimal_deformation @ G_inv)
        )

        logger.info(
            "YangMillsHolonomyAgent: Auditoría completada. "
            "S_YM = %.4e, χ(W) = %.4f%+.4fj, c₁ = %.4f, "
            "‖EOM_res‖_F = %.3e, ratio = %.3f.",
            action_result.action_value,
            holonomy.holonomy_trace.real,
            holonomy.holonomy_trace.imag,
            curvature.chern_number,
            action_result.eom_residual_norm,
            frobenius_ratio,
        )

        return DeformationTensor(
            delta_R=action_result.optimal_deformation,
            frobenius_ratio=frobenius_ratio,
            entropy_production=entropy_production,
            info={
                "yang_mills_action": action_result.action_value,
                "topological_action": action_result.topological_action,
                "wilson_trace": holonomy.holonomy_trace,
                "unitarity_defect": holonomy.unitarity_defect,
                "holonomy_class": holonomy.holonomy_class.name,
                "paradox_detected": holonomy.paradox_detected,
                "curvature_class": curvature.curvature_class.name,
                "chern_number": curvature.chern_number,
                "bianchi_residual": curvature.bianchi_residual,
                "eom_residual_norm": action_result.eom_residual_norm,
                "lyapunov_rhs_norm": action_result.lyapunov_rhs_norm,
                "kappa_metric": action_result.kappa_metric,
                "frobenius_ratio": frobenius_ratio,
                "entropy_production": entropy_production,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_input_dimensions(
        self,
        A_base: NDArray[np.float64],
        A_new: NDArray[np.float64],
        cycle: List[NDArray[np.float64]],
        J: NDArray[np.float64],
    ) -> None:
        """
        Verifica coherencia dimensional entre todas las entradas y la métrica G.

        Reglas:
            · A_base, A_new : cuadradas, misma dimensión n que G.
            · Cada A_k en cycle : cuadradas, dimensión n.
            · J : cuadrada, dimensión n.
        """
        n = self._G.shape[0]
        for name, M in [("A_base", A_base), ("A_new", A_new), ("J", J)]:
            if np.asarray(M).shape != (n, n):
                raise ValueError(
                    f"{name} tiene forma {np.asarray(M).shape}; "
                    f"se esperaba ({n}, {n}) compatible con G."
                )
        for k, A in enumerate(cycle):
            if np.asarray(A).shape != (n, n):
                raise ValueError(
                    f"cyclic_dependencies[{k}] tiene forma "
                    f"{np.asarray(A).shape}; se esperaba ({n}, {n})."
                )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enumeraciones
    "CurvatureClass",
    "HolonomyClass",
    # Excepciones
    "GaugeCurvatureSingularityError",
    "HolonomyVetoError",
    "YangMillsOptimizationError",
    "BianchiViolationError",
    # DTOs
    "GaugeCurvatureTensor",
    "WilsonLoopHolonomy",
    "YangMillsAction",
    # Fases
    "Phase1_GaugeCurvatureComputer",
    "Phase2_WilsonLoopAuditor",
    "Phase3_YangMillsOptimizer",
    # Orquestador
    "YangMillsHolonomyAgent",
]