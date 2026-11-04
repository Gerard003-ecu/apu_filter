# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo  : Yang-Mills Holonomy Agent                                          ║
║           (Dinámica de Calibre, Holonomía, Mecánica de Poincaré y RSI-L3)    ║
║ Ubicación: app/agents/core/immune_system/yang_mills_holonomy_agent.py        ║
║ Versión : 4.0.0-Poincare-Floquet-Magnus-BPS-RSI-L3-Bianchi-Higham-PhD        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DE CALIBRE (Rigor Doctoral): ──────────────
Este módulo consagra la autoridad definitiva sobre el `dynamic_shield_router.py`.
No opera como un mero enrutador o filtro procedimental de datos, sino
como un **Funtor de Curvatura No Abeliana**, **Auditor de Monodromía de Poincaré**
y **Motor de Automejora Recursiva de Nivel 3 (RSI-L3)** que evalúa la 2-forma de
curvatura $$\Omega$$ de la conexión de Ehresmann $$\omega$$ inyectada sobre la membrana
de disipación del Escudo (`funtor_shield.py`). Su mandato primordial es garantizar
que el transporte paralelo de los tensores de estado a través de la filtración de
la jerarquía DIKW preserve estrictamente la **invarianza de Gauge**, la **estabilidad
diofántica KAM** y la **unitariedad de Floquet-Poincaré**.

A través de la cuantización de las obstrucciones en el espacio de fase, la
verificación de las identidades de Bianchi-Jacobi, la expansión homotópica de
Magnus y la minimización del funcional de acción de Yang-Mills bajo la cota
topológica de Bogomol'nyi-Prasad-Sommerfield (BPS), el módulo actúa como el censor
supremo del pipeline, confinando toda la seguridad operacional a veredictos puros
sobre retículos intuicionistas de Heyting $$\Omega_3$$ en software.

ARQUITECTURA DE TRES FASES ANIDADAS (Continuación Formal Monoidal): ─────────────
La evolución de la coherencia geométrica de calibre se compone estrictamente de
tres morfismos encadenados donde el último método de cada fase constituye el
inicio formal de la siguiente:

  Fase 1 (`Phase1_GaugeCurvatureComputer`)
    ├─► CÁLCULO DE LA CURVATURA DE CALIBRE (Cartan, Bianchi-Jacobi y Vorticidad KAM)
    │   Evalúa la 2-forma de curvatura $$\Omega$$ mediante la ecuación de estructura
    │   de Cartan, la descomposición de Hodge autodual/anti-autodual y los invariantes
    │   de vorticidad simpléctica de Poincaré-Cartan:
    │   $$D_A F = dF + [A \wedge F] \equiv 0 \quad\big[765, 814\big]$$
    └─► Último método: `handoff_phase1_to_phase2`
          │ (Continuación formal directa — entrega `GaugeCurvatureTensor`)
          ▼
  Fase 2 (`Phase2_WilsonLoopAuditor`)
    ├─► Primer método: `ingest_from_phase1_handoff`
    ├─► AUDITORÍA DE HOLONOMÍA, MAPA DE RETORNO DE POINCARÉ Y SERIE DE MAGNUS
    │   Calcula el transporte paralelo ordenado por camino (Path-Ordered), extrae
    │   los multiplicadores característicos de Floquet-Poincaré, el número de
    │   rotación toroidal y el defecto de curvatura no abeliana de Stokes-Magnus:
    │   $$W(\gamma) = \mathcal{P} \exp\left( i \oint_\gamma \omega \right) \in U(n) \quad\big[813\big]$$
    └─► Último método: `handoff_phase2_to_phase3`
          │ (Continuación formal directa — entrega `WilsonLoopHolonomy`)
          ▼
  Fase 3 (`Phase3_YangMillsOptimizer`)
    ├─► Primer método: `ingest_from_phase2_handoff`
    ├─► MINIMIZACIÓN VARIACIONAL DE YANG-MILLS (Lyapunov, BPS, Higham y RSI-L3)
    │   Resuelve las ecuaciones de campo clásicas para la curvatura extrema bajo
    │   corrientes externas, verificando la cota de Bogomol'nyi-Prasad-Sommerfield
    │   y proyectando la deformación al cono semidefinido positivo $$S^+_n$$:
    │   $$D^\mu F_{\mu\nu} = J_\nu \quad\big[816\big]$$
    └─► Último método: `finalize_nested_pipeline`
          │ (Cierre autopoiético RSI-L3 — entrega `YangMillsAction` / `DeformationTensor`)
          ▼
  Orquestador (`YangMillsHolonomyAgent.enforce_gauge_invariance`)

INVARIANTES MATEMÁTICOS, CELESTES (POINCARÉ) Y METACOGNITIVOS (RSI-L3): ───────
  [I1] Invariancia de Calibre Local, Identidad de Bianchi y Asociador de Jacobi:
       La curvatura $$F = d\omega + \frac{1}{2}[\omega \wedge \omega]$$ satisface la
       Identidad de Bianchi y la identidad cíclica de Jacobi con precisión de máquina:
       $$\|D_A F\|_F \le \varepsilon_{\mathrm{Bianchi}} \quad \land \quad \|\mathcal{J}_{\mathrm{Jacobi}}(A_\mu, A_\nu, F)\|_F \le \varepsilon_{\mathrm{Bianchi}}$$

  [I2] Unitariedad del Bucle de Wilson y Estabilidad Elíptica de Floquet-Poincaré:
       Para todo ciclo cerrado $$\gamma$$, el operador de primer retorno $$W(\gamma)$$
       pertenece estrictamente a $$U(n)$$, con multiplicadores de Floquet sobre $$S^1$$:
       $$\|W^\dagger W - I\|_F \le \varepsilon_{\mathrm{unitarity}} \quad \land \quad \lambda_{\mathrm{Floquet}}(\gamma) = \max_a |\ln|\mu_a|| \approx 0$$

  [I3] Conservación de la Acción de Yang-Mills y Cota Topológica BPS:
       El funcional de acción con término topológico $$\theta$$ satisface la desigualdad
       de Bogomol'nyi-Prasad-Sommerfield respecto a la carga de Chern-Pontryagin:
       $$S_{\mathrm{YM}}[A] = \frac{1}{2} \int_M \operatorname{Tr}(F \wedge \star F) + \theta \int_M \operatorname{Tr}(F \wedge F) \ge S_{\mathrm{BPS}} \ge 0$$

  [I4] Automejora Recursiva de Nivel 3 (Endofuntor de Punto Fijo de Lawvere):
       El motor $$\mathfrak{R}_{\mathrm{YM}}^{(3)}$$ evalúa el meta-potencial variacional
       $$\mathcal{V}_{\mathrm{YM}}(\Theta_g)$$ y reescribe recursivamente los operadores
       de regularización bajo un certificado de contracción de Banach $$q_g \in (0, 1)$$.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

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

try:
    from app.core.immune_system.dynamic_shield_router import HeytingVerdict
except ImportError:
    class HeytingVerdict(Enum):  # type: ignore[no-redef]
        """Retículo distributivo acotado de Heyting Ω₃ = {VETOED, DEGRADED, COHERENT}."""

        VETOED = 0
        DEGRADED = 1
        COHERENT = 2

        def meet(self, other: HeytingVerdict) -> HeytingVerdict:
            return HeytingVerdict(min(self.value, other.value))

        def join(self, other: HeytingVerdict) -> HeytingVerdict:
            return HeytingVerdict(max(self.value, other.value))

logger = logging.getLogger("MIC.ImmuneSystem.YangMillsHolonomy")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, ESPECTRALES, CELESTES (POINCARÉ) Y METACOGNITIVAS (RSI-L3)
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

#: Constante diofántica basal γ_KAM para frecuencias del álgebra de Lie de calibre.
_KAM_GAUGE_GAMMA: float = 1.0e-3

#: Exponente diofántico τ de Poincaré-KAM para resonancias en la fibra de calibre.
_KAM_GAUGE_TAU: float = 2.0

#: Factor objetivo de contracción de Banach-Caccioppoli para Automejora RSI Nivel 3.
_RSI_L3_YM_CONTRACTION_TARGET: float = 0.6180339887498949


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
    """


class HolonomyVetoError(TopologicalInvariantError):
    """
    Lanzada cuando el bucle de Wilson W(γ) no pertenece al grupo unitario U(n),
    probando matemáticamente que la ruta de decisión encierra una paradoja
    topológica: el transporte paralelo no preserva la norma del estado cuántico.
    """


class YangMillsOptimizationError(TopologicalInvariantError):
    """
    Lanzada cuando la ecuación de Lyapunov generalizada para δR_opt no
    puede resolverse, o cuando el residuo de las ecuaciones de movimiento
    supera el umbral _EPS_EOM_RESIDUAL.
    """


class BianchiViolationError(TopologicalInvariantError):
    """
    Lanzada cuando la identidad de Bianchi D∧F = 0 se viola significativamente.
    """


# ══════════════════════════════════════════════════════════════════════════════
# INVARIANTES CELESTES DE POINCARÉ Y MOTOR METACOGNITIVO RSI DE NIVEL 3
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PoincareGaugeInvariants:
    r"""
    Invariantes de la Mecánica Celeste de Poincaré proyectados sobre el
    fibrado de calibre de Yang-Mills.

    Atributos
    ─────────
    poincare_cartan_vorticity : float
        Flujo de vorticidad simpléctica $$\omega_{\mathrm{PC}} = \operatorname{Tr}(F G^{-1})$$.
    kam_diophantine_margin : float
        Margen diofántico de no-resonancia $$\gamma_{\mathrm{KAM}}^{\mathrm{YM}}$$ entre
        los espectros de la subálgebra de Cartan de $$A_\mu$$ y $$A_\nu$$.
    floquet_rotation_number : float
        Número de rotación medio de Poincaré $$\rho_{\mathrm{Poincaré}}(\gamma) \in [-0.5, 0.5]$$
        extraído de los argumentos espectrales del mapa de retorno $$W(\gamma)$$.
    floquet_lyapunov_exponent : float
        Exponente de inestabilidad hiperbólica $$\lambda_{\mathrm{Floquet}} = \max_a |\ln|\mu_a||$$.
    magnus_commutator_defect : float
        Defecto no abeliano de Stokes-Magnus $$\|W(\gamma) - \exp(\Omega_{\mathrm{Magnus}})\|_F$$.
    bps_topological_floor : float
        Cota inferior topológica de Bogomol'nyi-Prasad-Sommerfield $$S_{\mathrm{BPS}}$$.
    """

    poincare_cartan_vorticity: float = 0.0
    kam_diophantine_margin: float = 1.0
    floquet_rotation_number: float = 0.0
    floquet_lyapunov_exponent: float = 0.0
    magnus_commutator_defect: float = 0.0
    bps_topological_floor: float = 0.0


@dataclass(frozen=True, slots=True)
class YangMillsRSILevel3Telemetry:
    r"""
    Certificado metacognitivo emitido por el Motor de Automejora Recursiva de
    Nivel 3 (`YangMillsRSILevel3Engine` — Punto Fijo de Lawvere).

    Atributos
    ─────────
    generation : int
        Generación evolutiva $$g \in \mathbb{N}$$ del endofuntor $$\mathfrak{R}_{\mathrm{YM}}^{(3)}$$.
    banach_contraction_factor : float
        Factor de contractividad de Banach-Caccioppoli $$q_g \in (0, 1)$$.
    meta_lyapunov_potential : float
        Potencial variacional compuesto $$\mathcal{V}_{\mathrm{YM}}(\Theta_g)$$.
    evolved_bianchi_gain : float
        Ganancia de sensibilidad para el asociador de Jacobi-Bianchi.
    evolved_magnus_homotopy_gain : float
        Ganancia de peso de segundo orden en la serie de Magnus.
    evolved_lyapunov_shift : float
        Piso de regularización espectral adaptativo para el operador de Lyapunov.
    verdict : HeytingVerdict
        Veredicto intuicionista en $$\Omega_3$$.
    """

    generation: int = 0
    banach_contraction_factor: float = _RSI_L3_YM_CONTRACTION_TARGET
    meta_lyapunov_potential: float = 0.0
    evolved_bianchi_gain: float = 1.0
    evolved_magnus_homotopy_gain: float = 1.0
    evolved_lyapunov_shift: float = 0.0
    verdict: HeytingVerdict = HeytingVerdict.COHERENT


class YangMillsRSILevel3Engine:
    r"""
    Motor de Automejora Recursiva de Nivel 3 (Endofuntor de Punto Fijo de Lawvere)
    para el Agente de Holonomía de Yang-Mills.

    Fundamento Matemático (Nivel 3)
    ───────────────────────────────
    Define un endofuntor de orden superior $$\mathfrak{R}_{\mathrm{YM}}^{(3)}: \operatorname{End}(\mathcal{A}) \to \operatorname{End}(\mathcal{A})$$
    sobre el espacio de Banach de operadores variacionales de calibre. En cada
    ejecución del pipeline de 3 fases, evalúa el funcional de Lyapunov-Poincaré:

    $$\mathcal{V}_{\mathrm{YM}}(\Theta_g) = S_{\mathrm{YM}}[A] + \delta_U(\gamma) + \lambda_{\mathrm{Floquet}}(\gamma) + \frac{\|\mathrm{EOM}_{\mathrm{res}}\|_F}{1 + \|J\|_F}$$

    y actualiza la ley generadora $$\Theta_{g+1} = \Psi_g(\Theta_g)$$ garantizando la
    contracción de Banach-Caccioppoli $$q_g \in (0, 1)$$.
    """

    _state: YangMillsRSILevel3Telemetry = YangMillsRSILevel3Telemetry()

    @classmethod
    def get_meta_state(cls) -> YangMillsRSILevel3Telemetry:
        """Retorna el estado metacognitivo actual del endofuntor RSI-L3."""
        return cls._state

    @classmethod
    def commit_level3_fixed_point(
        cls,
        action_value: float,
        unitarity_defect: float,
        floquet_exponent: float,
        eom_residual: float,
        norm_J: float,
        verdict: HeytingVerdict,
    ) -> YangMillsRSILevel3Telemetry:
        r"""
        Ejecuta la recursión autopoiética de Nivel 3 al cierre de la Fase 3,
        sintetizando la nueva ley variacional bajo contracción de Banach.
        """
        prev = cls._state
        gen = prev.generation + 1

        normalized_eom = eom_residual / (norm_J + 1.0)
        V_ym = (
            float(abs(action_value))
            + float(unitarity_defect)
            + float(floquet_exponent)
            + float(normalized_eom)
        )

        # Certificado de contracción de Banach-Caccioppoli q_g ∈ (0, 1)
        q_g = float(
            _RSI_L3_YM_CONTRACTION_TARGET
            * np.tanh(1.0 / (1.0 + V_ym + 1.0e-12))
        )
        q_g = float(np.clip(q_g, 0.05, 0.99))

        target_bianchi_gain = float(np.clip(1.0 / (1.0 + 0.1 * V_ym), 0.5, 1.5))
        evolved_bianchi = (
            q_g * prev.evolved_bianchi_gain
            + (1.0 - q_g) * target_bianchi_gain
        )

        target_magnus_gain = float(
            np.clip(1.0 / (1.0 + floquet_exponent), 0.5, 1.5)
        )
        evolved_magnus = (
            q_g * prev.evolved_magnus_homotopy_gain
            + (1.0 - q_g) * target_magnus_gain
        )

        target_lyap_shift = float(
            np.clip(1.0e-14 * normalized_eom, 0.0, 1.0e-10)
        )
        evolved_shift = (
            q_g * prev.evolved_lyapunov_shift
            + (1.0 - q_g) * target_lyap_shift
        )

        cls._state = YangMillsRSILevel3Telemetry(
            generation=gen,
            banach_contraction_factor=q_g,
            meta_lyapunov_potential=V_ym,
            evolved_bianchi_gain=float(evolved_bianchi),
            evolved_magnus_homotopy_gain=float(evolved_magnus),
            evolved_lyapunov_shift=float(evolved_shift),
            verdict=verdict,
        )
        return cls._state


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE CALIBRE (GAUGE DTOs)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class GaugeCurvatureTensor:
    r"""
    Tensor de Faraday Generalizado Ω (2-forma de curvatura de calibre).

    Producido por la **Fase 1**. Encapsula la curvatura F_{μν} con su
    clasificación topológica, invariantes de vorticidad de Poincaré-KAM
    y resultados de las verificaciones algebraicas (Bianchi, autodualidad).

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
    chern_number : float
        Número de Chern discreto c₁ ≈ Tr(F) / (2π).
    curvature_class : CurvatureClass
        Clasificación topológica: FLAT, SELF_DUAL, ANTI_SELF_DUAL o GENERAL.
    is_integrable : bool
        True ↔ curvature_class == FLAT (campo de gauge trivial).
    jacobi_associator_norm : float
        Norma del asociador cíclico de Jacobi ‖[A_μ,[A_ν,F]] + cícl.‖_F.
    poincare_vorticity : float
        Flujo de vorticidad simpléctica de Poincaré-Cartan Tr(F G⁻¹).
    kam_diophantine_margin : float
        Margen diofántico KAM entre modos espectrales de A_μ y A_ν.
    """

    F_matrix: NDArray[np.float64]
    F_selfdual: NDArray[np.float64]
    F_antiselfdual: NDArray[np.float64]
    frobenius_norm: float
    bianchi_residual: float
    chern_number: float
    curvature_class: CurvatureClass
    is_integrable: bool
    jacobi_associator_norm: float = 0.0
    poincare_vorticity: float = 0.0
    kam_diophantine_margin: float = 1.0


@dataclass(frozen=True, slots=True)
class WilsonLoopHolonomy:
    r"""
    Resultado espectral completo de la auditoría del bucle de Wilson W(γ)
    y del Mapa de Primer Retorno de Poincaré-Floquet.

    Producido por la **Fase 2**. Encapsula W(γ) con su carácter de
    representación, índice de unitariedad, multiplicadores de Floquet,
    defecto de Stokes-Magnus y clasificación topológica.

    Atributos
    ─────────
    phase_shift_matrix : NDArray[complex128], shape (n, n)
        Operador de holonomía W(γ) = 𝒫 exp(i ∮_γ A_μ ds_μ).
    holonomy_trace : complex
        χ(W) = Tr(W) ∈ ℂ. Carácter de la representación.
    unitarity_defect : float
        δ_U = ‖W†W − I‖_F. Medida de la no-unitariedad de W.
    arc_length_weights : NDArray[float64], shape (k,)
        Pesos de longitud de arco ds_k = ‖A_k‖_F / Σ‖A_j‖_F.
    holonomy_class : HolonomyClass
        Clasificación: TRIVIAL, NON_TRIVIAL o PARADOX.
    paradox_detected : bool
        True ↔ holonomy_class == PARADOX (W no unitario).
    floquet_rotation_number : float
        Número de rotación toroidal de Poincaré ρ(γ) ∈ [-0.5, 0.5].
    floquet_lyapunov_exponent : float
        Exponente de inestabilidad hiperbólica de Floquet max_a |ln|μ_a||.
    magnus_commutator_defect : float
        Defecto de truncamiento de segundo orden de la serie de Magnus.
    """

    phase_shift_matrix: NDArray[np.complex128]
    holonomy_trace: complex
    unitarity_defect: float
    arc_length_weights: NDArray[np.float64]
    holonomy_class: HolonomyClass
    paradox_detected: bool
    floquet_rotation_number: float = 0.0
    floquet_lyapunov_exponent: float = 0.0
    magnus_commutator_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class YangMillsAction:
    r"""
    Evaluación completa del funcional de acción de Yang-Mills, cota BPS y
    tensor de energía-momento de Poincaré-Cartan.

    Producido por la **Fase 3**. Encapsula S_YM, la deformación óptima
    δR y las métricas de validación de las ecuaciones de movimiento.

    Atributos
    ─────────
    action_value : float
        S_YM[A] = ½ Tr(Fᵀ G⁻¹ F G⁻¹) + θ · Tr(F G⁻¹ F̃ G⁻¹).
    topological_action : float
        Término topológico θ · c₁² (número de Pontryagin discreto).
    optimal_deformation : NDArray[float64], shape (n, n)
        δR_opt ∈ S⁺ₙ: solución de la ecuación de Lyapunov proyectada al cono PSD.
    eom_residual_norm : float
        ‖G⁻¹F + FG⁻¹ + G⁻¹δR + δRG⁻¹ − J‖_F.
    lyapunov_rhs_norm : float
        ‖RHS‖_F = ‖J − G⁻¹F − FG⁻¹‖_F.
    entropy_current_J : NDArray[float64], shape (n, n)
        Corriente de entropía estocástica J_ν usada como fuente.
    kappa_metric : float
        κ(G) = λ_max(G) / λ_min(G). Número de condición de la métrica.
    bps_bound_value : float
        Cota topológica inferior de Bogomol'nyi-Prasad-Sommerfield S_BPS.
    stress_energy_trace : float
        Traza del tensor de energía-momento de Yang-Mills Tr(T^{YM} G⁻¹).
    rsi_telemetry : YangMillsRSILevel3Telemetry
        Certificado metacognitivo de Automejora Recursiva de Nivel 3.
    """

    action_value: float
    topological_action: float
    optimal_deformation: NDArray[np.float64]
    eom_residual_norm: float
    lyapunov_rhs_norm: float
    entropy_current_J: NDArray[np.float64]
    kappa_metric: float
    bps_bound_value: float = 0.0
    stress_energy_trace: float = 0.0
    rsi_telemetry: YangMillsRSILevel3Telemetry = field(
        default_factory=YangMillsRSILevel3Telemetry
    )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE ANIDADA 1 — CÁLCULO DE LA CURVATURA DE CALIBRE                       ║
# ║   (Ecuación de Cartan, Identidad de Bianchi-Jacobi y Vorticidad KAM)        ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_GaugeCurvatureComputer:
    r"""
    Evalúa la 2-forma de curvatura $$\Omega$$ de la conexión de Ehresmann $$\omega$$
    inyectada sobre el fibrado de disipación, junto con sus invariantes
    topológicos de Chern, el asociador de Jacobi-Bianchi y la vorticidad
    simpléctica de Poincaré-Cartan.

    Fundamento Matemático
    ─────────────────────
    Sea $$P \to M$$ un G-fibrado principal con $$G = \mathrm{GL}^+(n, \mathbb{R})$$.
    La conexión $$\omega$$ induce la curvatura $$\Omega \in \Omega^2(P, \mathfrak{g})$$
    por la ecuación estructural de Cartan:

    $$\Omega = d\omega + \frac{1}{2}[\omega \wedge \omega]$$

    En la representación matricial discreta sobre el 1-símplice $$(A_\mu, A_\nu)$$:

    $$F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu], \quad \partial_\mu A_\nu \approx A_\nu - A_\mu$$

    Descomposición de Hodge Self-Dual / Anti-Self-Dual
    ──────────────────────────────────────────────────
    El operador estrella de Hodge $$\star_G$$ inducido por la métrica $$G$$ actúa
    sobre la parte antisimétrica $$F_{\mathrm{skew}} = \frac{1}{2}(F - F^\top)$$:

    $$\star_G F = G^{-1} F_{\mathrm{skew}} G, \qquad F^\pm = \frac{1}{2}(F \pm \star_G F)$$

    Identidad de Bianchi y Asociador de Jacobi-Cartan
    ─────────────────────────────────────────────────
    Se verifica tanto el residuo covariante simplicial:
    $$B = [A_\mu, F_{\nu\lambda}] + [A_\nu, F_{\lambda\mu}] + [A_\lambda, F_{\mu\nu}]$$
    como la identidad cíclica exacta de Jacobi en el álgebra de Lie:
    $$\mathcal{J}_{\mathrm{Jacobi}}(A_\mu, A_\nu, F) = [A_\mu, [A_\nu, F]] + [A_\nu, [F, A_\mu]] + [F, [A_\mu, A_\nu]] \equiv 0.$$
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        tol_flat: float = _EPS_FLAT_CURVATURE,
        tol_bianchi: float = _EPS_BIANCHI,
        tol_selfdual: float = _EPS_SELFDUAL,
    ) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        self._tol_flat = float(tol_flat)
        self._tol_bianchi = float(tol_bianchi)
        self._tol_selfdual = float(tol_selfdual)

        self._validate_metric(self._G)
        self._G_inv = la.inv(self._G)

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DE LA FASE 1
    # ──────────────────────────────────────────────────────────────────────────

    def compute_curvature(
        self,
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
    ) -> GaugeCurvatureTensor:
        r"""
        Calcula la 2-forma de curvatura $$F_{\mu\nu}$$ con todos sus invariantes
        topológicos, algebraicos (Bianchi-Jacobi) y celestes (Poincaré-KAM).
        """
        # ── Paso 1: Validación dimensional ──
        self._validate_connection_shapes(A_mu, A_nu)

        # ── Paso 2: Derivadas parciales finitas (fijación de calibre de Lorenz) ──
        d_A_nu = A_nu - A_mu
        d_A_mu = np.zeros_like(A_mu)

        # ── Paso 3: Conmutador de Lie [A_μ, A_ν] ──
        commutator = A_mu @ A_nu - A_nu @ A_mu

        # ── Paso 4: Tensor de Faraday discreto ──
        F = d_A_nu - d_A_mu + commutator

        # ── Paso 5: Operador de Hodge dual respecto a G ──
        F_skew = 0.5 * (F - F.T)
        F_hodge_dual = self._G_inv @ F_skew @ self._G

        # ── Paso 6: Descomposición self-dual / anti-self-dual ──
        F_plus = 0.5 * (F + F_hodge_dual)
        F_minus = 0.5 * (F - F_hodge_dual)

        # ── Paso 7: Clasificación topológica ──
        norm_F = float(np.linalg.norm(F, "fro"))
        norm_Fplus = float(np.linalg.norm(F_plus, "fro"))
        norm_Fminus = float(np.linalg.norm(F_minus, "fro"))
        curvature_class = self._classify_curvature(
            norm_F, F, F_hodge_dual, norm_Fplus, norm_Fminus
        )

        # ── Paso 8: Número de Chern c₁ = Tr(F) / (2π) ──
        chern = float(np.trace(F)) / (2.0 * math.pi)

        # ── Paso 9: Identidad de Bianchi y Asociador de Jacobi ──
        bianchi_res = self._compute_bianchi_residual(A_mu, A_nu, F)
        jacobi_norm = self._compute_jacobi_associator(A_mu, A_nu, F)

        if bianchi_res > self._tol_bianchi * 100.0:
            raise BianchiViolationError(
                f"Identidad de Bianchi violada severamente: "
                f"‖B‖_F = {bianchi_res:.3e} >> {self._tol_bianchi:.1e}. "
                f"Las conexiones A_μ, A_ν no pertenecen al mismo haz de gauge."
            )

        if bianchi_res > self._tol_bianchi:
            logger.warning(
                "Fase 1: Identidad de Bianchi aproximadamente violada: "
                "‖B‖_F = %.3e > %.1e.",
                bianchi_res,
                self._tol_bianchi,
            )

        # ── Paso 10: Vorticidad de Poincaré-Cartan y Margen Diofántico KAM ──
        poincare_vorticity, kam_margin = self._compute_poincare_gauge_invariants(
            A_mu, A_nu, F
        )

        # ── Paso 11: Logging de diagnóstico ──
        is_integrable = curvature_class == CurvatureClass.FLAT
        if not is_integrable:
            logger.warning(
                "Fase 1 [%s]: Anomalía de gauge — ‖F‖_F = %.4e, c₁ = %.4f, "
                "‖B‖_F = %.3e, ω_PC = %.4e, γ_KAM = %.3e.",
                curvature_class.name,
                norm_F,
                chern,
                bianchi_res,
                poincare_vorticity,
                kam_margin,
            )
        else:
            logger.debug(
                "Fase 1: Curvatura plana — ‖F‖_F = %.2e < %.1e, γ_KAM = %.3e.",
                norm_F,
                self._tol_flat,
                kam_margin,
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
            jacobi_associator_norm=jacobi_norm,
            poincare_vorticity=poincare_vorticity,
            kam_diophantine_margin=kam_margin,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE LA FASE 1
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
        Clasifica $$F$$ según su relación con el operador de Hodge dual $$\star_G$$.
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
        Computa el residuo de la identidad de Bianchi en el álgebra de Lie:
            $$A_\lambda = \frac{1}{2}(A_\mu + A_\nu)$$
            $$B = [A_\mu, F_{\nu\lambda}] + [A_\nu, F_{\lambda\mu}] + [A_\lambda, F_{\mu\nu}]$$
        """
        A_lambda = 0.5 * (A_mu + A_nu)

        F_nu_lambda = F - (A_nu @ A_lambda - A_lambda @ A_nu)
        F_lambda_mu = -F
        F_mu_nu = F

        B = (
            (A_mu @ F_nu_lambda - F_nu_lambda @ A_mu)
            + (A_nu @ F_lambda_mu - F_lambda_mu @ A_nu)
            + (A_lambda @ F_mu_nu - F_mu_nu @ A_lambda)
        )

        return float(np.linalg.norm(B, "fro"))

    @staticmethod
    def _compute_jacobi_associator(
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
        F: NDArray[np.float64],
    ) -> float:
        r"""
        Evalúa la identidad cíclica exacta de Jacobi en el álgebra de Lie:
            $$\mathcal{J}_{\mathrm{Jacobi}}(A_\mu, A_\nu, F) = [A_\mu, [A_\nu, F]] + [A_\nu, [F, A_\mu]] + [F, [A_\mu, A_\nu]]$$
        cuantificando la pérdida de asociatividad numérica de punto flotante IEEE-754.
        """
        comm_nu_F = A_nu @ F - F @ A_nu
        comm_F_mu = F @ A_mu - A_mu @ F
        comm_mu_nu = A_mu @ A_nu - A_nu @ A_mu

        jacobi = (
            (A_mu @ comm_nu_F - comm_nu_F @ A_mu)
            + (A_nu @ comm_F_mu - comm_F_mu @ A_nu)
            + (F @ comm_mu_nu - comm_mu_nu @ F)
        )
        return float(np.linalg.norm(jacobi, "fro"))

    def _compute_poincare_gauge_invariants(
        self,
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
        F: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Calcula la vorticidad simpléctica de Poincaré-Cartan $$\omega_{\mathrm{PC}} = \operatorname{Tr}(F G^{-1})$$
        y el margen diofántico KAM entre las proyecciones simétricas de las conexiones
        $$A_\mu$$ y $$A_\nu$$ en la subálgebra de Cartan.
        """
        poincare_vorticity = float(np.trace(F @ self._G_inv))

        sym_mu = 0.5 * (A_mu + A_mu.T)
        sym_nu = 0.5 * (A_nu + A_nu.T)
        evals_mu = la.eigvalsh(sym_mu)
        evals_nu = la.eigvalsh(sym_nu)

        scale = float(np.linalg.norm(evals_mu) + np.linalg.norm(evals_nu) + 1.0)
        omega_mu = evals_mu / scale
        omega_nu = evals_nu / scale

        n = A_mu.shape[0]
        min_div = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs((omega_mu[i] - omega_mu[j]) - (omega_nu[i] - omega_nu[j]))
                weighted = diff * ((i + j + 2) ** _KAM_GAUGE_TAU)
                if weighted > 0.0 and weighted < min_div:
                    min_div = float(weighted)

        kam_margin = max(min_div, _KAM_GAUGE_GAMMA)
        return poincare_vorticity, kam_margin

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

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE ANIDAMIENTO (FASE 1 ──► FASE 2)
    # Último método de la Fase 1: su definición formal es el inicio directo
    # del primer método de la Fase 2 (`ingest_from_phase1_handoff`).
    # ══════════════════════════════════════════════════════════════════════════

    def handoff_phase1_to_phase2(
        self,
        A_mu: NDArray[np.float64],
        A_nu: NDArray[np.float64],
        enforce_integrability: bool = True,
        phase2_auditor: Optional[Phase2_WilsonLoopAuditor] = None,
        cyclic_potentials: Optional[List[NDArray[np.float64]]] = None,
        enforce_paradox_check: bool = True,
    ) -> Union[GaugeCurvatureTensor, Tuple[GaugeCurvatureTensor, WilsonLoopHolonomy]]:
        r"""
        Morfismo terminal de la **Fase 1** (`handoff_phase1_to_phase2`) e inicio
        formal anidado de la **Fase 2** (`ingest_from_phase1_handoff`).

        Calcula el `GaugeCurvatureTensor`, verifica la condición de integrabilidad
        asintótica cuando `enforce_integrability=True` y, si se provee el auditor
        de la Fase 2 junto con `cyclic_potentials`, transfiere síncronamente el
        control al método inicial de la Fase 2:
            $$\Phi_2.\mathrm{ingest\_from\_phase1\_handoff}(\mathrm{curvature}, \mathrm{cyclic\_potentials}, \dots)$$
        """
        curvature = self.compute_curvature(A_mu=A_mu, A_nu=A_nu)

        if enforce_integrability and not curvature.is_integrable:
            raise GaugeCurvatureSingularityError(
                f"Singularidad de calibre: ‖F_{{μν}}‖_F = "
                f"{curvature.frobenius_norm:.4e}. "
                f"Clase = {curvature.curvature_class.name}. "
                f"c₁ = {curvature.chern_number:.4f}. "
                f"Bianchi residual = {curvature.bianchi_residual:.3e}."
            )

        if phase2_auditor is not None and cyclic_potentials is not None:
            holonomy = phase2_auditor.ingest_from_phase1_handoff(
                curvature=curvature,
                cyclic_potentials=cyclic_potentials,
                enforce_paradox_check=enforce_paradox_check,
            )
            return curvature, holonomy

        return curvature


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE ANIDADA 2 — AUDITORÍA DE HOLONOMÍA Y MAPA DE RETORNO DE POINCARÉ     ║
# ║   (Bucle de Wilson, Multiplicadores de Floquet y Expansión de Magnus)       ║
# ║                                                                             ║
# ║   CONTINUACIÓN FORMAL DIRECTA DE:                                           ║
# ║   `Phase1_GaugeCurvatureComputer.handoff_phase1_to_phase2`                  ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_WilsonLoopAuditor:
    r"""
    Mide el desfase topológico del transporte paralelo a lo largo de ciclos
    cerrados $$\gamma$$ en el grafo de dependencias ($$\beta_1 > 0$$), evaluando
    el operador de Wilson como un **Mapa de Primer Retorno de Poincaré** sobre la
    fibra de calibre.

    Fundamento Matemático y Mecánica Celeste de Poincaré-Floquet
    ────────────────────────────────────────────────────────────
    La discretización invariante bajo reparametrización pondera cada generador
    por su longitud de arco relativa de Frobenius $$ds_k = \|A_k\|_F / \sum_j \|A_j\|_F$$:

    $$W(\gamma) = \mathcal{P} \prod_{k=1}^m \exp(i A_k ds_k)$$

    1. **Multiplicadores Característicos de Floquet y Número de Rotación de Poincaré**:
       Del espectro $$\{\mu_a\}_{a=1}^n = \operatorname{spec}(W(\gamma))$$, se extraen:
       $$\rho_{\mathrm{Poincaré}}(\gamma) = \frac{1}{2\pi n} \sum_{a=1}^n \arg(\mu_a), \qquad
         \lambda_{\mathrm{Floquet}}(\gamma) = \max_{1 \le a \le n} |\ln|\mu_a||.$$
       Si $$W(\gamma) \in U(n)$$, todos los multiplicadores residen sobre el toro
       unitario $$|\mu_a| = 1$$ ($$\lambda_{\mathrm{Floquet}} = 0$$, estabilidad elíptica KAM).
       Si algún $$|\mu_a| \neq 1$$, emerge una bifurcación hiperbólica (paradoja topológica).

    2. **Expansión Homotópica No Abeliana de Stokes-Magnus**:
       Se compara $$W(\gamma)$$ frente al generador de Magnus de segundo orden:
       $$\Omega_{\mathrm{Magnus}} = i \sum_k A_k ds_k - \frac{1}{2} \sum_{j < k} [A_j, A_k] ds_j ds_k$$
       para cuantificar el área no abeliana encerrada en la sección de Poincaré.
    """

    def __init__(
        self,
        tol_unitarity: float = _EPS_UNITARITY,
        tol_trivial: float = _EPS_HOLONOMY,
    ) -> None:
        self._tol_unitarity = float(tol_unitarity)
        self._tol_trivial = float(tol_trivial)

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE RECEPCIÓN (CONTINUACIÓN DE `handoff_phase1_to_phase2`)
    # Primer método de la Fase 2: recibe directamente la salida de la Fase 1.
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_from_phase1_handoff(
        self,
        curvature: GaugeCurvatureTensor,
        cyclic_potentials: List[NDArray[np.float64]],
        enforce_paradox_check: bool = True,
    ) -> WilsonLoopHolonomy:
        r"""
        Morfismo inicial de la **Fase 2**, continuación formal directa del último
        método de la Fase 1 (`Phase1_GaugeCurvatureComputer.handoff_phase1_to_phase2`).

        Consume el `GaugeCurvatureTensor` certificado en la Fase 1, evalúa el
        bucle de Wilson y la monodromía de Poincaré-Floquet sobre `cyclic_potentials`,
        y ejecuta `check_paradox` si `enforce_paradox_check=True`.
        """
        holonomy = self.evaluate_holonomy(
            cyclic_potentials=cyclic_potentials,
            curvature=curvature,
        )
        if enforce_paradox_check:
            self.check_paradox(holonomy)
        return holonomy

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DE LA FASE 2
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_holonomy(
        self,
        cyclic_potentials: List[NDArray[np.float64]],
        curvature: Optional[GaugeCurvatureTensor] = None,
    ) -> WilsonLoopHolonomy:
        r"""
        Calcula la holonomía $$W(\gamma)$$ con todos sus invariantes topológicos,
        multiplicadores de Floquet-Poincaré y defecto no abeliano de Magnus.
        """
        # ── Caso trivial: ciclo vacío ──
        if not cyclic_potentials:
            n = self._G_dim_from_context(cyclic_potentials, curvature)
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
            return self._trivial_holonomy(n)

        ds = arc_lengths / total_arc

        # ── Paso 4-5: Producto ordenado de camino (Path-Ordered Product) ──
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
        paradox = holonomy_class == HolonomyClass.PARADOX

        # ── Paso 8: Monodromía de Poincaré-Floquet y Expansión de Magnus ──
        rot_num, floquet_exp, magnus_defect = (
            self._compute_poincare_floquet_and_magnus(W, cyclic_potentials, ds)
        )

        # ── Paso 9: Logging de diagnóstico ──
        logger.debug(
            "Fase 2: W(γ) — χ(W) = %.4f%+.4fj, δ_U = %.3e, "
            "‖W−I‖_F = %.3e, ρ_Poincaré = %.4f, λ_Floquet = %.3e, clase = %s.",
            trace_W.real,
            trace_W.imag,
            unitarity_defect,
            deviation_from_I,
            rot_num,
            floquet_exp,
            holonomy_class.name,
        )

        if holonomy_class == HolonomyClass.NON_TRIVIAL:
            logger.warning(
                "Fase 2: Holonomía no trivial detectada — "
                "‖W−I‖_F = %.3e ≥ %.1e, ρ_Poincaré = %.4f. El ciclo γ encierra curvatura.",
                deviation_from_I,
                self._tol_trivial,
                rot_num,
            )

        return WilsonLoopHolonomy(
            phase_shift_matrix=W,
            holonomy_trace=trace_W,
            unitarity_defect=unitarity_defect,
            arc_length_weights=ds,
            holonomy_class=holonomy_class,
            paradox_detected=paradox,
            floquet_rotation_number=rot_num,
            floquet_lyapunov_exponent=floquet_exp,
            magnus_commutator_defect=magnus_defect,
        )

    def check_paradox(self, holonomy: WilsonLoopHolonomy) -> None:
        r"""
        Verifica la holonomía y lanza `HolonomyVetoError` si se detecta
        una paradoja topológica ($$W \notin U(n)$$).
        """
        if holonomy.paradox_detected:
            logger.error(
                "Fase 2: Veto de Holonomía — δ_U = %.4e > %.1e. "
                "χ(W) = %.4f%+.4fj, λ_Floquet = %.4e.",
                holonomy.unitarity_defect,
                self._tol_unitarity,
                holonomy.holonomy_trace.real,
                holonomy.holonomy_trace.imag,
                holonomy.floquet_lyapunov_exponent,
            )
            raise HolonomyVetoError(
                f"Paradoja topológica: el transporte paralelo no es unitario. "
                f"δ_U = ‖W†W − I‖_F = {holonomy.unitarity_defect:.4e} "
                f"> {self._tol_unitarity:.1e}. "
                f"χ(W) = {holonomy.holonomy_trace:.4f}. "
                f"El haz de gauge no admite sección global consistente."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE LA FASE 2
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_holonomy(
        self,
        unitarity_defect: float,
        deviation_from_I: float,
    ) -> HolonomyClass:
        """
        Clasifica la holonomía en tres niveles (por orden de prioridad):
        PARADOX, TRIVIAL o NON_TRIVIAL.
        """
        if unitarity_defect > self._tol_unitarity:
            return HolonomyClass.PARADOX
        if deviation_from_I < self._tol_trivial:
            return HolonomyClass.TRIVIAL
        return HolonomyClass.NON_TRIVIAL

    @staticmethod
    def _compute_poincare_floquet_and_magnus(
        W: NDArray[np.complex128],
        potentials: List[NDArray[np.float64]],
        ds: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        Extrae el número de rotación toroidal de Poincaré, el exponente de
        Floquet-Lyapunov del espectro de $$W(\gamma)$$ y el residuo de truncamiento
        de segundo orden de la expansión continua de Magnus.
        """
        n = W.shape[0]
        evals_W = la.eigvals(W)
        abs_evals = np.maximum(np.abs(evals_W), 1.0e-15)
        angles = np.angle(evals_W)

        rotation_number = float(np.sum(angles) / (2.0 * math.pi * float(n)))
        floquet_exponent = float(np.max(np.abs(np.log(abs_evals))))

        # Serie de Magnus hasta 2º orden:
        # Ω₁ = i Σ_k A_k ds_k
        # Ω₂ = -½ Σ_{j < k} [i A_j ds_j, i A_k ds_k] = ½ Σ_{j < k} [A_j, A_k] ds_j ds_k
        # (en convención multiplicativa por la derecha W = exp(Ω₁_0) exp(Ω₁_1)...
        #  BCH: log(exp(X)exp(Y)) ≈ X + Y + ½[X, Y])
        omega_1 = np.zeros((n, n), dtype=np.complex128)
        omega_2 = np.zeros((n, n), dtype=np.complex128)
        scaled_ops = [
            1j * A_k.astype(np.complex128) * float(ds_k)
            for A_k, ds_k in zip(potentials, ds)
        ]
        for j, X_j in enumerate(scaled_ops):
            omega_1 += X_j
            for k in range(j + 1, len(scaled_ops)):
                X_k = scaled_ops[k]
                omega_2 += 0.5 * (X_j @ X_k - X_k @ X_j)

        rsi_gain = YangMillsRSILevel3Engine.get_meta_state().evolved_magnus_homotopy_gain
        W_magnus = la.expm(omega_1 + rsi_gain * omega_2)
        magnus_defect = float(np.linalg.norm(W - W_magnus, "fro"))

        return rotation_number, floquet_exponent, magnus_defect

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
            floquet_rotation_number=0.0,
            floquet_lyapunov_exponent=0.0,
            magnus_commutator_defect=0.0,
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
        curvature: Optional[GaugeCurvatureTensor] = None,
    ) -> int:
        """Extrae la dimensión del espacio del contexto o de G_PHYSICS."""
        if potentials:
            return potentials[0].shape[0]
        if curvature is not None:
            return curvature.F_matrix.shape[0]
        return G_PHYSICS.shape[0]

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE ANIDAMIENTO (FASE 2 ──► FASE 3)
    # Último método de la Fase 2: su definición formal es el inicio directo
    # del primer método de la Fase 3 (`ingest_from_phase2_handoff`).
    # ══════════════════════════════════════════════════════════════════════════

    def handoff_phase2_to_phase3(
        self,
        curvature: GaugeCurvatureTensor,
        cyclic_potentials: List[NDArray[np.float64]],
        enforce_paradox_check: bool = True,
        phase3_optimizer: Optional[Phase3_YangMillsOptimizer] = None,
        stochastic_entropy_J: Optional[NDArray[np.float64]] = None,
        base_connection_A: Optional[NDArray[np.float64]] = None,
    ) -> Union[WilsonLoopHolonomy, YangMillsAction, DeformationTensor]:
        r"""
        Morfismo terminal de la **Fase 2** (`handoff_phase2_to_phase3`) e inicio
        formal anidado de la **Fase 3** (`ingest_from_phase2_handoff`).

        Ejecuta la recepción y auditoría de holonomía de la Fase 2
        (`ingest_from_phase1_handoff`) y, si recibe el optimizador de la Fase 3
        junto con la corriente de entropía $$J_\nu$$, transfiere síncronamente el
        control al método inicial de la Fase 3:
            $$\Phi_3.\mathrm{ingest\_from\_phase2\_handoff}(\mathrm{curvature}, \mathrm{holonomy}, J_\nu, \dots)$$
        """
        holonomy = self.ingest_from_phase1_handoff(
            curvature=curvature,
            cyclic_potentials=cyclic_potentials,
            enforce_paradox_check=enforce_paradox_check,
        )
        if phase3_optimizer is not None and stochastic_entropy_J is not None:
            return phase3_optimizer.ingest_from_phase2_handoff(
                curvature=curvature,
                holonomy=holonomy,
                stochastic_entropy_J=stochastic_entropy_J,
                base_connection_A=base_connection_A,
            )
        return holonomy


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE ANIDADA 3 — MINIMIZACIÓN DE LA ACCIÓN DE YANG-MILLS Y CIERRE RSI-L3  ║
# ║   (Ecuación de Lyapunov, Cota BPS, Proyección de Higham y Punto Fijo)       ║
# ║                                                                             ║
# ║   CONTINUACIÓN FORMAL DIRECTA DE:                                           ║
# ║   `Phase2_WilsonLoopAuditor.handoff_phase2_to_phase3`                       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_YangMillsOptimizer:
    r"""
    Minimiza la acción variacional de Yang-Mills $$S_{\mathrm{YM}}[A]$$, verifica la
    cota inferior topológica de Bogomol'nyi-Prasad-Sommerfield (BPS), resuelve
    la ecuación continua de Lyapunov para la deformación óptima $$\delta R_{\mathrm{opt}} \in S^+_n$$
    y cierra el bucle de Automejora Recursiva de Nivel 3 (`YangMillsRSILevel3Engine`).

    Fundamento Variacional Riguroso
    ────────────────────────────────
    La acción de Yang-Mills con término $$\theta$$-topológico es:

    $$S_{\mathrm{YM}}[A] = \frac{1}{2} \operatorname{Tr}(F^\top G^{-1} F G^{-1}) + \theta \cdot \operatorname{Tr}(F G^{-1} \tilde{F} G^{-1})$$

    Por la identidad algebraica $$F = F^+ + F^-$$, la acción cinética satisface la
    **Desigualdad Topológica de Bogomol'nyi-Prasad-Sommerfield (BPS)**:

    $$S_{\mathrm{kin}}[A] \ge S_{\mathrm{BPS}} := \frac{1}{2} \left|\operatorname{Tr}(F^\top G^{-1} \tilde{F} G^{-1})\right| \ge 0$$

    donde la igualdad se alcanza si y sólo si la curvatura es autodual ($$F = F^+$$,
    instanton) o anti-autodual ($$F = F^-$$, anti-instanton).

    La variación $$\delta S_{\mathrm{YM}} / \delta A = 0$$ acoplada a la corriente de
    entropía estocástica $$J_\nu$$ induce la ecuación de Lyapunov generalizada:

    $$G^{-1} \delta R + \delta R G^{-1} = J - (G^{-1} F + F G^{-1}) =: \mathrm{RHS}$$

    cuya solución simetrizada se proyecta ortogonalmente en norma de Frobenius al
    cono semidefinido positivo $$S^+_n$$ mediante el Teorema de Higham (2002).
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

        self._G_inv = la.inv(self._G)
        evals_G = la.eigvalsh(self._G)
        self._kappa_G = float(evals_G[-1] / evals_G[0])

        if self._kappa_G > _KAPPA_MAX_METRIC:
            logger.warning(
                "Fase 3: Métrica G mal condicionada κ(G) = %.3e > %.1e. "
                "La solución de Lyapunov puede ser numéricamente inestable.",
                self._kappa_G,
                _KAPPA_MAX_METRIC,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN FORMAL DE RECEPCIÓN (CONTINUACIÓN DE `handoff_phase2_to_phase3`)
    # Primer método de la Fase 3: recibe directamente la salida de la Fase 2.
    # ══════════════════════════════════════════════════════════════════════════

    def ingest_from_phase2_handoff(
        self,
        curvature: GaugeCurvatureTensor,
        holonomy: WilsonLoopHolonomy,
        stochastic_entropy_J: NDArray[np.float64],
        base_connection_A: Optional[NDArray[np.float64]] = None,
    ) -> Union[YangMillsAction, DeformationTensor]:
        r"""
        Morfismo inicial de la **Fase 3**, continuación formal directa del último
        método de la Fase 2 (`Phase2_WilsonLoopAuditor.handoff_phase2_to_phase3`).

        Consume el `GaugeCurvatureTensor` (Fase 1) y el `WilsonLoopHolonomy`
        (Fase 2), ejecuta la optimización variacional `minimize_action` y culmina
        en el cierre autopoiético de las 3 fases anidadas `finalize_nested_pipeline`.
        """
        action_result = self.minimize_action(
            curvature=curvature,
            stochastic_entropy_J=stochastic_entropy_J,
        )
        return self.finalize_nested_pipeline(
            curvature=curvature,
            holonomy=holonomy,
            action_result=action_result,
            base_connection_A=base_connection_A,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DE LA FASE 3
    # ──────────────────────────────────────────────────────────────────────────

    def minimize_action(
        self,
        curvature: GaugeCurvatureTensor,
        stochastic_entropy_J: NDArray[np.float64],
    ) -> YangMillsAction:
        r"""
        Calcula $$S_{\mathrm{YM}}$$, la cota BPS, el tensor de energía-momento de
        Poincaré-Cartan y obtiene $$\delta R_{\mathrm{opt}} \in S^+_n$$ por resolución
        de la ecuación continua de Lyapunov y proyección de Higham.
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

        # ── Paso 2: Acción de Yang-Mills, Cota BPS y Tensor de Energía-Momento ──
        S_kinetic, S_topological = self._compute_action(F, curvature.F_selfdual)
        bps_bound, stress_trace = self._compute_bps_bound_and_stress_energy(F)

        # ── Paso 3: Término fuente de la ecuación de Lyapunov ──
        field_term = self._G_inv @ F + F @ self._G_inv
        RHS = J - field_term
        rhs_norm = float(np.linalg.norm(RHS, "fro"))

        # ── Paso 4: Resolución de la ecuación continua de Lyapunov ──
        rsi_shift = YangMillsRSILevel3Engine.get_meta_state().evolved_lyapunov_shift
        G_inv_reg = (
            self._G_inv + rsi_shift * np.eye(n, dtype=np.float64)
            if rsi_shift > 0.0
            else self._G_inv
        )
        try:
            delta_R_raw = la.solve_continuous_lyapunov(G_inv_reg, RHS)
        except (la.LinAlgError, ValueError) as exc:
            raise YangMillsOptimizationError(
                f"La ecuación de Lyapunov G⁻¹δR + δRG⁻¹ = RHS no pudo "
                f"resolverse: {exc}. κ(G) = {self._kappa_G:.3e}."
            ) from exc

        # ── Paso 5: Simetrización ──
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
                eom_residual,
                self._eom_tol,
            )

        logger.debug(
            "Fase 3: S_YM = %.4e (cinético) + %.4e (topológico), S_BPS = %.4e. "
            "‖RHS‖_F = %.3e. ‖EOM_res‖_F = %.3e. κ(G) = %.2e.",
            S_kinetic,
            S_topological,
            bps_bound,
            rhs_norm,
            eom_residual,
            self._kappa_G,
        )

        return YangMillsAction(
            action_value=S_kinetic + S_topological,
            topological_action=S_topological,
            optimal_deformation=delta_R_psd,
            eom_residual_norm=eom_residual,
            lyapunov_rhs_norm=rhs_norm,
            entropy_current_J=J,
            kappa_metric=self._kappa_G,
            bps_bound_value=bps_bound,
            stress_energy_trace=stress_trace,
            rsi_telemetry=YangMillsRSILevel3Engine.get_meta_state(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE LA FASE 3
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_action(
        self,
        F: NDArray[np.float64],
        F_plus: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Calcula los dos términos de la acción de Yang-Mills:
            $$S_{\mathrm{cin}} = \frac{1}{2}\operatorname{Tr}(F^\top G^{-1} F G^{-1}), \qquad
              S_{\mathrm{top}} = \theta \cdot \operatorname{Tr}(F G^{-1} \tilde{F} G^{-1})$$
        """
        F_contracted = F.T @ self._G_inv @ F @ self._G_inv
        S_kinetic = 0.5 * float(np.real(np.trace(F_contracted)))

        F_skew = 0.5 * (F - F.T)
        F_dual = self._G_inv @ F_skew @ self._G
        top_contracted = F @ self._G_inv @ F_dual @ self._G_inv
        S_topological = self._theta * float(np.real(np.trace(top_contracted)))

        return S_kinetic, S_topological

    def _compute_bps_bound_and_stress_energy(
        self,
        F: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Calcula la cota inferior topológica de Bogomol'nyi-Prasad-Sommerfield (BPS):
            $$S_{\mathrm{BPS}} = \frac{1}{2} \left|\operatorname{Tr}(F^\top G^{-1} \tilde{F} G^{-1})\right|$$
        y la traza contraída del tensor de energía-momento de Poincaré-Cartan:
            $$T_{\mathrm{YM}} = F^\top G^{-1} F - \frac{1}{4}\operatorname{Tr}(F^\top G^{-1} F G^{-1}) G.$$
        """
        F_skew = 0.5 * (F - F.T)
        F_dual = self._G_inv @ F_skew @ self._G
        bps_bound = 0.5 * abs(
            float(np.real(np.trace(F.T @ self._G_inv @ F_dual @ self._G_inv)))
        )

        lagrangian_density = 0.25 * float(
            np.real(np.trace(F.T @ self._G_inv @ F @ self._G_inv))
        )
        T_ym = F.T @ self._G_inv @ F - lagrangian_density * self._G
        stress_trace = float(np.trace(T_ym @ self._G_inv))

        return bps_bound, stress_trace

    def _compute_eom_residual(
        self,
        F: NDArray[np.float64],
        delta_R: NDArray[np.float64],
        J: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula el residuo de las ecuaciones de movimiento post-hoc:
            $$\mathrm{EOM}_{\mathrm{res}} = G^{-1} F + F G^{-1} + G^{-1} \delta R + \delta R G^{-1} - J$$
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
        Proyecta $$M$$ al cono $$S^+_n$$ usando el algoritmo de Higham (2002):
            $$\Pi_{S^+_n}(M) = U \cdot \operatorname{diag}(\max(\lambda_i, 0)) \cdot U^\top$$
        """
        evals, evecs = la.eigh(M)
        evals_psd = np.maximum(evals, 0.0)
        M_psd = evecs @ np.diag(evals_psd) @ evecs.T
        return 0.5 * (M_psd + M_psd.T)

    # ══════════════════════════════════════════════════════════════════════════
    # ESLABÓN TERMINAL DE LAS 3 FASES ANIDADAS (CIERRE AUTOPOIÉTICO RSI-L3)
    # Último método de la Fase 3: cierra el ciclo recursivo de Nivel 3 y entrega
    # el DeformationTensor inmutable al Orquestador Categórico.
    # ══════════════════════════════════════════════════════════════════════════

    def finalize_nested_pipeline(
        self,
        curvature: GaugeCurvatureTensor,
        holonomy: WilsonLoopHolonomy,
        action_result: YangMillsAction,
        base_connection_A: Optional[NDArray[np.float64]] = None,
    ) -> Union[YangMillsAction, DeformationTensor]:
        r"""
        Morfismo terminal de la **Fase 3** (`finalize_nested_pipeline`) y cierre
        del acoplamiento monoidal de las tres fases anidadas:
            $$\Psi_{\mathrm{YM}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$

        Evalúa el veredicto en el retículo intuicionista de Heyting $$\Omega_3$$,
        actualiza el punto fijo de Lawvere en el motor de Automejora Recursiva de
        Nivel 3 (`YangMillsRSILevel3Engine`) y empaqueta la solución variacional
        en un `DeformationTensor` inmutable cuando se provee `base_connection_A`.
        """
        verdict = HeytingVerdict.COHERENT
        if holonomy.paradox_detected:
            verdict = HeytingVerdict.VETOED
        elif (
            action_result.eom_residual_norm > self._eom_tol
            or holonomy.holonomy_class == HolonomyClass.NON_TRIVIAL
            or not curvature.is_integrable
        ):
            verdict = verdict.meet(HeytingVerdict.DEGRADED)

        norm_J = float(np.linalg.norm(action_result.entropy_current_J, "fro"))
        rsi_cert = YangMillsRSILevel3Engine.commit_level3_fixed_point(
            action_value=action_result.action_value,
            unitarity_defect=holonomy.unitarity_defect,
            floquet_exponent=holonomy.floquet_lyapunov_exponent,
            eom_residual=action_result.eom_residual_norm,
            norm_J=norm_J,
            verdict=verdict,
        )

        if base_connection_A is None:
            return action_result

        R_base_norm = float(np.linalg.norm(base_connection_A, "fro"))
        delta_R_norm = float(
            np.linalg.norm(action_result.optimal_deformation, "fro")
        )
        frobenius_ratio = delta_R_norm / (R_base_norm + 1e-15)

        entropy_production = float(
            np.trace(action_result.optimal_deformation @ self._G_inv)
        )

        return DeformationTensor(
            delta_R=action_result.optimal_deformation,
            frobenius_ratio=frobenius_ratio,
            entropy_production=entropy_production,
            info={
                "yang_mills_action": action_result.action_value,
                "topological_action": action_result.topological_action,
                "bps_bound_value": action_result.bps_bound_value,
                "stress_energy_trace": action_result.stress_energy_trace,
                "wilson_trace": holonomy.holonomy_trace,
                "unitarity_defect": holonomy.unitarity_defect,
                "holonomy_class": holonomy.holonomy_class.name,
                "paradox_detected": holonomy.paradox_detected,
                "floquet_rotation_number": holonomy.floquet_rotation_number,
                "floquet_lyapunov_exponent": holonomy.floquet_lyapunov_exponent,
                "magnus_commutator_defect": holonomy.magnus_commutator_defect,
                "curvature_class": curvature.curvature_class.name,
                "chern_number": curvature.chern_number,
                "bianchi_residual": curvature.bianchi_residual,
                "jacobi_associator_norm": curvature.jacobi_associator_norm,
                "poincare_vorticity": curvature.poincare_vorticity,
                "kam_diophantine_margin": curvature.kam_diophantine_margin,
                "eom_residual_norm": action_result.eom_residual_norm,
                "lyapunov_rhs_norm": action_result.lyapunov_rhs_norm,
                "kappa_metric": action_result.kappa_metric,
                "frobenius_ratio": frobenius_ratio,
                "entropy_production": entropy_production,
                "rsi_generation": rsi_cert.generation,
                "rsi_banach_contraction": rsi_cert.banach_contraction_factor,
                "heyting_verdict": rsi_cert.verdict.name,
            },
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: YANG-MILLS HOLONOMY AGENT (Morfismo Anidado)         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class YangMillsHolonomyAgent(Morphism):
    r"""
    Operador de Medida Geométrica que garantiza la invarianza de Gauge del
    flujo de disipación Port-Hamiltoniano a través de los estratos DIKW,
    encadenando las tres fases anidadas mediante sus morfismos formales de
    continuación:

        Phase1.handoff_phase1_to_phase2
          └──▶ Phase2.ingest_from_phase1_handoff ... Phase2.handoff_phase2_to_phase3
                 └──▶ Phase3.ingest_from_phase2_handoff ... Phase3.finalize_nested_pipeline
                        └──▶ DeformationTensor inmutable
    """

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        theta_ym: float = _THETA_YM,
        enforce_bianchi: bool = True,
    ) -> None:
        self._G = np.asarray(metric_tensor, dtype=np.float64)
        self._enforce_bianchi = enforce_bianchi

        # Instanciación de las tres fases anidadas con la métrica compartida
        self._curvature_computer = Phase1_GaugeCurvatureComputer(
            metric_tensor=self._G
        )
        self._holonomy_auditor = Phase2_WilsonLoopAuditor()
        self._optimizer = Phase3_YangMillsOptimizer(
            metric_tensor=self._G, theta_ym=theta_ym
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA DEL AGENTE DE HOLONOMÍA
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
        invarianza de gauge, la monodromía de Poincaré-Floquet y la optimización
        variacional de Yang-Mills encadenando las 3 fases anidadas.
        """
        logger.info(
            "YangMillsHolonomyAgent: Iniciando auditoría anidada de invarianza de gauge. "
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

        # ── Ejecución Anidada Continua: Fase 1 ──► Fase 2 ──► Fase 3 ─────────
        # 1. El último método de la Fase 1 calcula y certifica la curvatura
        curvature = self._curvature_computer.handoff_phase1_to_phase2(
            A_mu=base_connection_A,
            A_nu=proposed_connection_A_new,
            enforce_integrability=True,
        )
        assert isinstance(curvature, GaugeCurvatureTensor)

        # 2. El último método de la Fase 2 consume la salida de la Fase 1
        #    (`ingest_from_phase1_handoff`) y transfiere directamente al primer
        #    método de la Fase 3 (`ingest_from_phase2_handoff`), el cual culmina
        #    en `finalize_nested_pipeline` emitiendo el DeformationTensor final.
        deformation = self._holonomy_auditor.handoff_phase2_to_phase3(
            curvature=curvature,
            cyclic_potentials=cyclic_dependencies,
            enforce_paradox_check=True,
            phase3_optimizer=self._optimizer,
            stochastic_entropy_J=stochastic_entropy_J,
            base_connection_A=base_connection_A,
        )
        assert isinstance(deformation, DeformationTensor)

        logger.info(
            "YangMillsHolonomyAgent: Auditoría completada. "
            "S_YM = %.4e, χ(W) = %.4f%+.4fj, c₁ = %.4f, "
            "‖EOM_res‖_F = %.3e, ratio = %.3f, RSI_gen = %d, Ω₃ = %s.",
            deformation.info["yang_mills_action"],
            deformation.info["wilson_trace"].real,
            deformation.info["wilson_trace"].imag,
            curvature.chern_number,
            deformation.info["eom_residual_norm"],
            deformation.frobenius_ratio,
            deformation.info["rsi_generation"],
            deformation.info["heyting_verdict"],
        )

        return deformation

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
    "HeytingVerdict",
    # Excepciones
    "GaugeCurvatureSingularityError",
    "HolonomyVetoError",
    "YangMillsOptimizationError",
    "BianchiViolationError",
    # Invariantes de Poincaré y Motor RSI-L3
    "PoincareGaugeInvariants",
    "YangMillsRSILevel3Telemetry",
    "YangMillsRSILevel3Engine",
    # DTOs
    "GaugeCurvatureTensor",
    "WilsonLoopHolonomy",
    "YangMillsAction",
    # Fases Anidadas
    "Phase1_GaugeCurvatureComputer",
    "Phase2_WilsonLoopAuditor",
    "Phase3_YangMillsOptimizer",
    # Orquestador
    "YangMillsHolonomyAgent",
]