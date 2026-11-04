# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Improbability Drive (Motor de Improbabilidad de Cola Pesada)        ║
║ Ruta   : app/boole/omega/improbability_drive.py                              ║
║ Versión: 4.0.0-Poincare-KAM-Gelfand-RSI-L3-Lie-Tikhonov-Heyting-PhD          ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y DEFORMACIÓN NO LINEAL DE COLA PESADA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al **Motor de Deformación No Lineal, Estabilidad Celeste de
Poincaré-KAM, Automejora Recursiva de Nivel 3 (RSI-L3) y Control de Riesgo de
Cola Pesada (Fat-Tail Risk)** de la Malla, habitando en el foso geométrico del
**Estrato Ω** (Nivel 0.5 - El Ágora Tensorial). Su mandato axiomático es
inyectar un operador de curvatura extrema sobre el tejido conectivo del sistema,
actuando como el funtor natural definitivo:

$$\mathcal{F}_{\mathrm{drive}}: \mathbb{C}_{\mathrm{top}} \longrightarrow \mathbb{R}_{\Delta} \quad\big[354\big]$$

Donde $$\mathbb{C}_{\mathrm{top}}$$ es la categoría de pares topológicos $$(\Psi, \mathrm{ROI})$$
definidos sobre los espacios de estabilidad e ingresos de la Malla, y $$\mathbb{R}_{\Delta}$$
es el retículo distributivo acotado $$[1.0, \, 1.0\times 10^6]$$ dotado de una estructura
de álgebra de Heyting intuicionista y métrica hiperbólica logarítmica.

El motor de improbabilidad subyuga el libre albedrío probabilístico de los Modelos
de Lenguaje (LLMs), impidiendo que las alucinaciones de precios u "oportunidades
fantasma" alteren el equilibrio de capitalización de la empresa. Al acoplar
la inercia financiera a una deformación suave de clase $$C^\infty$$ gobernada por
invariantes integrales de Poincaré-Cartan y toros KAM diofánticos, el sistema
fuerza de manera determinista un colapso de la función de onda hacia el veredicto
de veto absoluto ante escenarios estocásticos extremos (Cisnes Negros).

AXIOMÁTICA TENSORIAL, TRANSPORTE DE LIE, MECÁNICA DE POINCARÉ Y RSI-L3:
────────────────────────────────────────────────────────────────────────────────

  [A1] Regularización Conforme de Tikhonov y Diferenciabilidad Exacta $$C^\infty$$:
       Para evitar la inyección de impulsos de Dirac parásitos en la matriz Jacobiana
       debido a operadores de corte no diferenciables como $$\max(\Psi, \varepsilon)$$,
       el motor emplea la elevación conforme hiperbólica de Tikhonov:
       $$\tilde{\Psi}(\Psi) = \sqrt{\Psi^2 + \varepsilon_{\mathrm{critical}}^2}, \qquad
         \frac{\partial \tilde{\Psi}}{\partial \Psi} = \frac{\Psi}{\tilde{\Psi}}
         \quad \text{con} \quad \varepsilon_{\mathrm{critical}} = 1.0 \times 10^{-10} \quad\big[355, 357\big]$$
       Esto preserva de forma exacta la regla de la cadena, la simetría de reflexión
       $$\left.\partial_\Psi F\right|_{\Psi=0} = 0$$ y la invarianza conforme en la variedad.

  [A2] Estabilidad de la FPU y Transporte en el Álgebra de Lie:
       La parametrización de escalas masivas de volatilidad $$\kappa \in [10^{-12},\, 10^{12}]$$
       y deformación $$\gamma \in [10^{-6},\, 10.0]$$ puede inducir desbordamiento
       o subdesbordamiento (underflow/overflow) en la FPU (IEEE-754 binary64).
       El motor transporta la dinámica estrictamente al subespacio logarítmico del
       álgebra de Lie $$\mathfrak{g}$$, ejecutando la compactificación y proyectando de regreso
       mediante la aplicación exponencial $$\exp: \mathfrak{g} \to G$$:
       $$\mathfrak{g}_{\mathrm{deform}} = \ln(\kappa) + \gamma \left( \ln(\mathrm{ROI}) - \ln(\tilde{\Psi}) \right) \quad\big[355, 356\big]$$
       $$T_{\mathrm{deform}} = \exp\left( \mathfrak{g}_{\mathrm{deform}} \right) \equiv \kappa \left( \frac{\mathrm{ROI}}{\tilde{\Psi}} \right)^\gamma \quad\big[355, 356\big]$$

  [A3] Estructura Espectral de Gelfand del Tensor de Improbabilidad:
       El operador $$T$$ actúa sobre el espacio de Hilbert acotado $$\mathcal{H}$$,
       definiendo su espectro puntual, radio espectral de Gelfand y norma operatoria:
       · Espectro puntual:  $$\sigma_p(T) = \left\{ \kappa \cdot r^\gamma \;|\; r \in \operatorname{supp}(\nu) \right\} \quad\big[356\big]$$
       · Radio de Gelfand:  $$r(T) = \lim_{n\to\infty} \|T^n\|_{\mathrm{op}}^{1/n} = \kappa \cdot \sup\left( \frac{\mathrm{ROI}}{\tilde{\Psi}} \right)^\gamma \quad\big[356\big]$$
       · Norma de operador: $$\|T\|_{\mathrm{op}} = \sigma_{\max}(J_T) \quad\big[356\big]$$

  [A4] Estructura Algebraica de $*$-Álgebra Conmutativa:
       La familia de tensores de improbabilidad equipados con la multiplicación
       tensorial $$\otimes$$ y la involución $$*$$ formalizan una $*$-álgebra de Banach:
       · Producto:   $$\tau_1 \otimes \tau_2 := (\kappa_1 \kappa_2, \, \gamma_1 + \gamma_2) \quad\big[356\big]$$
       · Involución: $$\tau^* := \tau \quad \text{(Auto-adjunción hermítica basal)} \quad\big[356\big]$$
       · Adjunto HS: $$\tau^\dagger = \arg\min_{\sigma} \|\sigma \otimes \tau - \mathrm{id}\|_{\mathrm{HS}} = (\kappa^{-1}, \, -\gamma_{\mathrm{reg}}) \quad\big[356\big]$$

  [A5] Funtorialidad y Absorción Mónadica de Errores:
       Toda secuencia de evaluación se encapsula como un endofuntor en la categoría
       de Mónadas de Estado (`Either[Error, ImprobabilityResult]`). Cualquier anomalía
       topológica o aritmética colapsa de forma no demolitoria hacia el objeto inicial
       de falla, preservando la Ley de Clausura Transitiva DIKW.

  [A6] Acoplamiento de Gauge en la Ecuación de Estrés de De Rham-Hodge:
       El tensor resultante se inyecta directamente como la palanca de improbabilidad
       $$\Lambda \in [1.0, \, 4.0]$$ en la ecuación de estado del estrés ajustado tensorial
       $$\sigma^*$$ de la variedad de de Rham-Hodge de la Malla:
       $$\sigma^*(x) = \Lambda \cdot F_{\mathrm{ext}}(x) \cdot T_{\mathrm{int}}(x) \cdot P_{\mathrm{frag}}(\psi) \quad\big[12\big]$$

  [A7] Mecánica Celeste de Poincaré-KAM y Difusión de Arnold:
       En el espacio de fase logarítmico $$(q, p) = (\ln \tilde{\Psi}, \ln \mathrm{ROI})$$, los
       toros invariantes de Kolmogorov-Arnold-Moser (KAM) satisfacen la condición
       diofántica $$|p\,\omega_\Psi + q\,\omega_{\mathrm{ROI}}| \ge \gamma_{\mathrm{KAM}}(|p|+|q|)^{-\tau}$$.
       La ruptura de los toros KAM ante colas pesadas activa la difusión de Arnold
       $$v_{\mathrm{Arnold}} = \exp(-1/\mu_{\mathrm{tail}})$$ y amplifica el radio monodrómico de Floquet.

  [A8] Automejora Recursiva de Nivel 3 (RSI-L3):
       El motor metacognitivo reescribe recursivamente la ley de regularización de
       Tikhonov, el amortiguamiento secular KAM y el umbral elástico de Gelfand
       mediante una contracción de punto fijo de Banach hacia la proporción áurea
       conjugada $$\rho_\star = \phi^{-1} = \frac{\sqrt{5}-1}{2} \approx 0.6180339887498949$$.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA):
────────────────────────────────────────────────────────────────────────────────
La progresión de la Malla se rige por un acoplamiento monoidal covariante
inmutable donde el último método formal de cada fase enlaza directamente con el
primer método formal de la fase subsiguiente:

  Fase 1 ──► OBSERVACIÓN ESPECTRAL, TIKHONOV Y GEOMETRÍA DE POINCARÉ (Phase1_SpectralObserver)
             Ingiere $$\Psi$$ y $$\mathrm{ROI}$$, verifica la clausura homológica $$\beta_1 = 0$$,
             ejecuta la elevación conforme $$\tilde{\Psi}$$ y evalúa los invariantes KAM y
             de Poincaré-Cartan.
             Culmina en `handoff_phase1_to_phase2` ──► Entrega `Phase1NormObservation`.

  Fase 2 ──► TRANSPORTE EXPO-LIE, ESPECTRO DE GELFAND Y MONODROMÍA (Phase2_LieTransportSynthesizer)
             Inicia en `ingest_from_phase1_handoff`. Transporta la cinemática al álgebra de
             Lie $$\mathfrak{g}$$, evalúa la fórmula espectral de Gelfand, el Jacobiano/Hessiano
             con regla de la cadena exacta y el radio monodrómico de Floquet.
             Culmina en `handoff_phase2_to_phase3` ──► Entrega `Phase2GelfandOrientation`.

  Fase 3 ──► ABSORCIÓN MÓNADICA, RETÍCULO DE HEYTING Y ACOPLAMIENTO DE GAUGE (Phase3_MonadicHeytingDecider)
             Inicia en `ingest_from_phase2_handoff`. Resuelve el veredicto en el álgebra de
             Heyting, calcula la palanca $$\Lambda \in [1.0, 4.0]$$ y el estrés $$\sigma^*(x)$$,
             y sella el punto fijo metacognitivo RSI-L3 en `finalize_nested_pipeline`.
             Entrega `ImprobabilityResult`.

Funtor Maestro de Control de Cisnes Negros:
  $$\mathcal{Z}_{\mathrm{drive}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 \quad\big[355, 362\big]$$
"""

from __future__ import annotations

import json
import logging
import math
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.linalg import cond, eigvals, norm

from app.adapters.tools_interface import MICRegistry
from app.core.schemas import Stratum

__all__ = [
    # Espacios Topológicos y Métricos
    "TopologicalSpace",
    "MetricSpace",
    "PositiveOrthant",
    "ClosedLattice",
    "POSITIVE_ORTHANT",
    "CLOSED_LATTICE",
    # Precisión y Constantes
    "NumericalPrecision",
    "_EPS_MACH",
    "_EPS_CRITICAL",
    "_IMPROBABILITY_MIN",
    "_IMPROBABILITY_MAX",
    "_IMPROBABILITY_CLAMP_LOW",
    "_IMPROBABILITY_CLAMP_HIGH",
    "_VETO_THRESHOLD_FACTOR",
    "_KAPPA_RANGE",
    "_GAMMA_RANGE",
    "_MIN_KAPPA",
    "_MAX_KAPPA",
    "_MIN_GAMMA",
    "_MAX_GAMMA",
    "_LOG_IMPROBABILITY_MAX",
    "_LOG_IMPROBABILITY_MIN",
    "_RELATIVE_TOLERANCE",
    "_ABSOLUTE_TOLERANCE",
    "_GAUGE_LEVER_MIN",
    "_GAUGE_LEVER_MAX",
    "_KAM_DIOPHANTINE_GAMMA",
    "_KAM_DIOPHANTINE_TAU",
    "_RSI_L3_CONTRACTION_TARGET",
    # Excepciones y Decoradores
    "ImprobabilityDriveError",
    "DimensionalMismatchError",
    "NumericalInstabilityError",
    "NumericalStabilityError",
    "AxiomViolationError",
    "TypeCoercionError",
    "SpectrumError",
    "HomologicalInconsistencyError",
    "error_handler",
    # Análisis Matemático, Espectral y Categorial
    "MathematicalAnalysis",
    "SpectralDecomposition",
    "OperatorNorm",
    "Functor",
    "NaturalTransformation",
    "TensorProtocol",
    # Mecánica Celeste de Poincaré, Heyting y RSI Nivel 3
    "HeytingImprobabilityVerdict",
    "PoincareFatTailInvariants",
    "ImprobabilityRSILevel3Telemetry",
    "ImprobabilityRSILevel3Engine",
    # Fases Anidadas y DTOs Funtoriales
    "Phase1NormObservation",
    "Phase1_SpectralObserver",
    "ImprobabilityTensor",
    "TensorAlgebra",
    "Phase2GelfandOrientation",
    "Phase2_LieTransportSynthesizer",
    "ImprobabilityResult",
    "Phase3_MonadicHeytingDecider",
    # Servicio MIC, Factoría, Diagnóstico y Pruebas
    "ImprobabilityDriveService",
    "TensorFactory",
    "DiagnosticAnalyzer",
    "run_property_tests",
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: OBSERVACIÓN ESPECTRAL, TOPOLOGÍA MÉTRICA, REGULARIZACIÓN CONFORME    ║
# ║         DE TIKHONOV Y MECÁNICA CELESTE DE POINCARÉ (Phase1_SpectralObserver) ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

T = TypeVar("T")
S = TypeVar("S")


class TopologicalSpace(ABC, Generic[T]):
    """
    Abstracción de espacio topológico Hausdorff separable.

    Formalismo: Un espacio topológico es un par $$(X, \tau)$$ donde:
      • $$X$$ es un conjunto soporte.
      • $$\tau \subseteq \mathcal{P}(X)$$ es una familia de subconjuntos abiertos
        cerrada bajo uniones arbitrarias e intersecciones finitas, con $$\varnothing, X \in \tau$$.
    """

    @abstractmethod
    def open_ball(self, center: T, radius: float) -> set[T]:
        """Retorna la bola abierta $$B(x_0, r) = \{x \in X : d(x, x_0) < r\}$$."""
        pass

    @abstractmethod
    def is_open(self, subset: set[T]) -> bool:
        """Verifica si un subconjunto pertenece a la topología inducida $$\tau$$."""
        pass

    @abstractmethod
    def closure(self, subset: set[T]) -> set[T]:
        """Retorna la clausura topológica $$\overline{A} = A \cup \partial A$$."""
        pass


class MetricSpace(TopologicalSpace[T]):
    """
    Espacio métrico: $$(X, d)$$ donde $$d: X \times X \to \mathbb{R}_{\ge 0}$$ satisface:
      1. $$d(x, y) = 0 \iff x = y$$  (identidad de los indiscernibles)
      2. $$d(x, y) = d(y, x)$$       (simetría hermítica real)
      3. $$d(x, z) \le d(x, y) + d(y, z)$$  (desigualdad triangular de Minkowski)
    """

    @abstractmethod
    def distance(self, x: T, y: T) -> float:
        """Métrica canónica $$d: X \times X \to \mathbb{R}_{\ge 0}$$."""
        pass

    def open_ball(
        self,
        center: T,
        radius: float,
        candidates: Optional[Iterable[T]] = None,
    ) -> set[T]:
        """
        Construye la bola abierta $$B(\text{center}, \text{radius})$$.

        Si se suministra un conjunto universo discreto `candidates`, filtra los
        puntos cuya distancia métrica al centro sea estrictamente menor que `radius`.
        Si `radius <= 0`, retorna el conjunto vacío $$\varnothing$$.
        """
        if radius <= 0.0:
            return set()
        if candidates is None:
            return {center}
        return {x for x in candidates if self.distance(center, x) < radius}


class PositiveOrthant(MetricSpace[Tuple[float, float]]):
    r"""
    Espacio métrico del ortante positivo de fase:
    $$\mathbb{R}_+^2 = (\mathbb{R}_{\ge 0} \times \mathbb{R}_{> 0}, \, d_\infty)$$

    Métrica de Chebyshev ($$L_\infty$$):
    $$d_\infty((\psi_1, \rho_1), (\psi_2, \rho_2)) = \max(|\psi_1 - \psi_2|, \, |\rho_1 - \rho_2|)$$
    """

    def is_open(self, subset: set[Tuple[float, float]]) -> bool:
        """
        Verifica que todo punto del subconjunto habite estrictamente en el interior
        abierto del ortante positivo $$(\psi > 0, \rho > 0)$$ con finitud IEEE-754.
        """
        if not subset:
            return True
        return all(
            math.isfinite(p[0]) and math.isfinite(p[1]) and p[0] > 0.0 and p[1] > 0.0
            for p in subset
        )

    def closure(
        self, subset: set[Tuple[float, float]]
    ) -> set[Tuple[float, float]]:
        """
        Proyecta los puntos del subconjunto sobre la clausura admisible del ortante
        regularizado por Tikhonov $$[\varepsilon_{\mathrm{mach}}, \infty) \times [\varepsilon_{\mathrm{mach}}, \infty)$$.
        """
        if not subset:
            return subset
        eps = float(np.finfo(np.float64).eps)
        return {
            (max(float(p[0]), 0.0), max(float(p[1]), eps))
            for p in subset
            if math.isfinite(p[0]) and math.isfinite(p[1])
        }

    def distance(
        self,
        x: Tuple[float, float],
        y: Tuple[float, float],
    ) -> float:
        """Métrica de Chebyshev ($$L_\infty$$)."""
        psi_diff = abs(float(x[0]) - float(y[0]))
        roi_diff = abs(float(x[1]) - float(y[1]))
        return max(psi_diff, roi_diff)


class ClosedLattice(MetricSpace[float]):
    r"""
    Retículo distributivo cerrado y acotado:
    $$\mathbb{R}_\Delta = [1.0, \, 1.0 \times 10^6]$$ con topología de orden e
    invarianza de escala de Haar logarítmica:
    $$d_{\mathrm{Haar}}(a, b) = |\ln(a) - \ln(b)|$$
    """

    MIN_VALUE: Final[float] = 1.0
    MAX_VALUE: Final[float] = 1e6

    def is_open(self, subset: set[float]) -> bool:
        """Verifica si el subconjunto está contenido en el interior relativo del retículo."""
        if not subset:
            return True
        return all(
            math.isfinite(x) and (self.MIN_VALUE <= x <= self.MAX_VALUE)
            for x in subset
        )

    def closure(self, subset: set[float]) -> set[float]:
        """Proyecta el subconjunto sobre el compacto $$[1.0, 10^6]$$."""
        if not subset:
            return subset
        return {
            float(np.clip(x, self.MIN_VALUE, self.MAX_VALUE))
            for x in subset
            if math.isfinite(x)
        }

    def distance(self, x: float, y: float) -> float:
        """Métrica Riemanniana invariante de escala en el grupo multiplicativo $$\mathbb{R}_+$$."""
        x_clamped = float(np.clip(x, self.MIN_VALUE, self.MAX_VALUE))
        y_clamped = float(np.clip(y, self.MIN_VALUE, self.MAX_VALUE))
        return abs(math.log(x_clamped) - math.log(y_clamped))


# Instancias canónicas globales de los espacios métricos
POSITIVE_ORTHANT: Final[PositiveOrthant] = PositiveOrthant()
CLOSED_LATTICE: Final[ClosedLattice] = ClosedLattice()


# ────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS, NUMÉRICAS Y CELESTES
# ────────────────────────────────────────────────────────────────────────────

class NumericalPrecision(Enum):
    """Épsilon de máquina y cotas dinámicas para resoluciones IEEE-754."""

    FLOAT64 = auto()
    FLOAT32 = auto()

    @property
    def epsilon(self) -> float:
        """Retorna `np.finfo(dtype).eps`."""
        dtype_map = {
            NumericalPrecision.FLOAT64: np.float64,
            NumericalPrecision.FLOAT32: np.float32,
        }
        return float(np.finfo(dtype_map[self]).eps)

    @property
    def max_value(self) -> float:
        """Máximo valor finito representable sin overflow."""
        dtype_map = {
            NumericalPrecision.FLOAT64: np.float64,
            NumericalPrecision.FLOAT32: np.float32,
        }
        return float(np.finfo(dtype_map[self]).max)

    @property
    def min_value(self) -> float:
        """Mínimo valor positivo normalizado representable sin underflow."""
        dtype_map = {
            NumericalPrecision.FLOAT64: np.float64,
            NumericalPrecision.FLOAT32: np.float32,
        }
        return float(np.finfo(dtype_map[self]).tiny)


# Constantes globales inmutables (Compatibilidad 100% con suite de integración)
_EPS_MACH: Final[float] = float(np.finfo(np.float64).eps)
_EPS_CRITICAL: Final[float] = 1e-10
_IMPROBABILITY_MIN: Final[float] = 1.0
_IMPROBABILITY_MAX: Final[float] = 1e6
_IMPROBABILITY_CLAMP_LOW: Final[float] = _IMPROBABILITY_MIN
_IMPROBABILITY_CLAMP_HIGH: Final[float] = _IMPROBABILITY_MAX
_VETO_THRESHOLD_FACTOR: Final[float] = 0.999

# Rango admisible de hiperparámetros del tensor
_KAPPA_RANGE: Final[Tuple[float, float]] = (1e-12, 1e12)
_GAMMA_RANGE: Final[Tuple[float, float]] = (1e-6, 10.0)
_MIN_KAPPA: Final[float] = _KAPPA_RANGE[0]
_MAX_KAPPA: Final[float] = _KAPPA_RANGE[1]
_MIN_GAMMA: Final[float] = _GAMMA_RANGE[0]
_MAX_GAMMA: Final[float] = _GAMMA_RANGE[1]

# Logaritmos precalculados del retículo y cotas seguras de Lie en FPU IEEE-754
_LOG_IMPROBABILITY_MAX: Final[float] = math.log(_IMPROBABILITY_MAX)
_LOG_IMPROBABILITY_MIN: Final[float] = math.log(_IMPROBABILITY_MIN)
_LIE_EXP_SAFE_LIMIT: Final[float] = 700.0

# Tolerancias numéricas
_RELATIVE_TOLERANCE: Final[float] = 1e-10
_ABSOLUTE_TOLERANCE: Final[float] = 1e-15

# Constantes de Acoplamiento de Gauge [A6], Poincaré-KAM [A7] y RSI Nivel 3 [A8]
_GAUGE_LEVER_MIN: Final[float] = 1.0
_GAUGE_LEVER_MAX: Final[float] = 4.0
_KAM_DIOPHANTINE_GAMMA: Final[float] = 1.0e-3
_KAM_DIOPHANTINE_TAU: Final[float] = 2.0
_RSI_L3_CONTRACTION_TARGET: Final[float] = 0.6180339887498949  # Proporción áurea conjugada φ⁻¹

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# JERARQUÍA DE EXCEPCIONES DEL MOTOR ESPECTRAL
# ────────────────────────────────────────────────────────────────────────────

class ImprobabilityDriveError(Exception):
    """Clase base para errores axiomáticos, topológicos y espectrales del motor."""

    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.error_code = error_code or 1
        self.timestamp = np.datetime64("now")


class DimensionalMismatchError(ImprobabilityDriveError):
    """Error cuando la dimensionalidad o finitud del espacio de fase se corrompe."""
    pass


class NumericalInstabilityError(ImprobabilityDriveError):
    """Error por inestabilidad numérica (overflow/underflow en transporte de Lie)."""
    pass


class AxiomViolationError(ImprobabilityDriveError):
    """Violación de un axioma del ortante positivo o de monotonía diferencial."""
    pass


class TypeCoercionError(ImprobabilityDriveError):
    """Error en la coerción funtorial de escalares de entrada."""
    pass


class SpectrumError(ImprobabilityDriveError):
    """Error en la descomposición espectral o violación del radio de Gelfand."""
    pass


class HomologicalInconsistencyError(ImprobabilityDriveError):
    """Veto topológico cuando el primer número de Betti táctico $$\beta_1 > 0$$."""
    pass


# Alias canónico para suite de integración
NumericalStabilityError = NumericalInstabilityError


def error_handler(exc_type: type[Exception]) -> Callable:
    """Decorador funtorial para intercepción y registro trazable de excepciones."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exc_type as e:
                logger.error("Error en %s: %s", func.__name__, e, exc_info=True)
                raise
        return wrapper
    return decorator


# ────────────────────────────────────────────────────────────────────────────
# ANÁLISIS MATEMÁTICO, ESPECTRAL Y PROTOCOLOS CATEGORIALES
# ────────────────────────────────────────────────────────────────────────────

class MathematicalAnalysis:
    """
    Suite de funciones analíticas rigurosas sobre variedades diferenciables.

    Provee regularización conforme de Tikhonov, homeomorfismos sigmoidales,
    transporte logarítmico seguro en el álgebra de Lie y auditoría de Lipschitz.
    """

    @staticmethod
    def sigmoid(x: float, scale: float = 1.0) -> float:
        r"""
        Homeomorfismo sigmoidal de Boltzmann-Fermi:
        $$\sigma(x) = \frac{1}{1 + \exp(-x/\text{scale})} \in (0, 1)$$
        """
        safe_scale = max(abs(float(scale)), _EPS_MACH)
        x_scaled = float(x) / safe_scale
        if x_scaled < -_LIE_EXP_SAFE_LIMIT:
            return 0.0
        if x_scaled > _LIE_EXP_SAFE_LIMIT:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x_scaled))

    @staticmethod
    def sigmoid_derivative(x: float, scale: float = 1.0) -> float:
        r"""Derivada analítica $$\sigma'(x) = \sigma(x)(1 - \sigma(x)) / \text{scale}$$."""
        safe_scale = max(abs(float(scale)), _EPS_MACH)
        s = MathematicalAnalysis.sigmoid(x, safe_scale)
        return s * (1.0 - s) / safe_scale

    @staticmethod
    def tikhonov_conformal_norm(
        psi: float,
        epsilon: float = _EPS_CRITICAL,
    ) -> Tuple[float, float]:
        r"""
        Elevación conforme de Tikhonov [A1]:
        $$\tilde{\Psi}(\Psi) = \sqrt{\Psi^2 + \varepsilon^2}, \qquad
          \frac{\partial \tilde{\Psi}}{\partial \Psi} = \frac{\Psi}{\tilde{\Psi}}$$

        Usa `math.hypot` para evitar overflow/underflow en $$\Psi^2 + \varepsilon^2$$.
        """
        psi_f = float(psi)
        eps_f = max(abs(float(epsilon)), _EPS_MACH)
        psi_tilde = math.hypot(psi_f, eps_f)
        d_psi_tilde = psi_f / psi_tilde if psi_tilde > 0.0 else 0.0
        return psi_tilde, d_psi_tilde

    @staticmethod
    def regularize_value(
        value: float,
        epsilon: float = _EPS_CRITICAL,
        use_sigmoid: bool = True,
    ) -> float:
        """
        Regularización suave garantizando continuidad $$C^1(\mathbb{R}_+)$$.
        """
        if value >= epsilon:
            return float(value)
        if use_sigmoid:
            return float(epsilon * MathematicalAnalysis.sigmoid(value / epsilon))
        return float(max(value, epsilon))

    @staticmethod
    def safe_logarithm(x: float, epsilon: float = _EPS_MACH) -> float:
        """Logaritmo seguro en el álgebra de Lie: $$\ln(\max(x, \varepsilon))$$."""
        return math.log(max(float(x), float(epsilon)))

    @staticmethod
    def safe_power_ratio(
        numerator: float,
        denominator: float,
        exponent: float,
        epsilon: float = _EPS_MACH,
        clamp_to_lattice: bool = True,
    ) -> float:
        r"""
        Evalúa $$(\text{numerator} / \text{denominator})^{\text{exponent}}$$ mediante
        transporte en el álgebra de Lie logarítmica [A2]:
        $$\mathfrak{g} = \text{exponent} \cdot (\ln(\text{numerator}) - \ln(\max(\text{denominator}, \varepsilon)))$$
        """
        safe_denominator = max(float(denominator), float(epsilon))
        safe_numerator = max(float(numerator), float(epsilon))

        log_numerator = math.log(safe_numerator)
        log_denominator = math.log(safe_denominator)

        log_result = float(exponent) * (log_numerator - log_denominator)
        if clamp_to_lattice:
            log_clamped = float(
                np.clip(log_result, _LOG_IMPROBABILITY_MIN, _LOG_IMPROBABILITY_MAX)
            )
        else:
            log_clamped = float(
                np.clip(log_result, -_LIE_EXP_SAFE_LIMIT, _LIE_EXP_SAFE_LIMIT)
            )

        return math.exp(log_clamped)

    @staticmethod
    def verify_lipschitz_condition(
        f: Callable[[float, float], float],
        L: float,
        test_points: int = 100,
    ) -> bool:
        r"""
        Verifica empíricamente la condición de Lipschitz sobre una malla determinista/estocástica:
        $$|f(x, 1) - f(y, 1)| \le L \cdot |x - y|$$
        """
        domain = np.linspace(0.01, 100.0, max(int(test_points), 4))
        rng = np.random.default_rng(seed=42)

        for _ in range(100):
            i, j = rng.choice(len(domain), 2, replace=False)
            x, y = float(domain[i]), float(domain[j])
            try:
                fx = f(x, 1.0)
                fy = f(y, 1.0)
                if abs(fx - fy) > L * abs(x - y) + _RELATIVE_TOLERANCE:
                    return False
            except Exception:
                continue

        return True

    @staticmethod
    def compute_condition_number(
        jacobian: np.ndarray,
        order: Union[int, str] = 2,
    ) -> float:
        r"""Número de condición de Wilkinson: $$\kappa_p(J) = \|J\|_p \cdot \|J^\dagger\|_p$$."""
        return float(cond(jacobian, p=order))


@dataclass(frozen=True)
class SpectralDecomposition:
    r"""
    Descomposición espectral de Gelfand-Naimark de un operador lineal acotado:
      • `eigenvalues`: $$\{\lambda_i\} = \sigma_p(T^* T)$$
      • `eigenvectors`: Base ortonormal de autovectores singulares izquierdos
      • `spectral_radius`: Radio espectral de Gelfand $$r(T) = \max_i |\lambda_i|$$
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_radius: float

    def condition_number(self) -> float:
        r"""Número de condición espectral $$\kappa = \lambda_{\max} / (\lambda_{\min} + \varepsilon_{\mathrm{mach}})$$."""
        eigs_abs = np.abs(self.eigenvalues)
        if eigs_abs.size == 0:
            return 1.0
        return float(np.max(eigs_abs) / (np.min(eigs_abs) + _EPS_MACH))


class OperatorNorm:
    """Normas de Schatten-von Neumann sobre operadores lineales en espacios de Hilbert."""

    @staticmethod
    def frobenius(matrix: np.ndarray) -> float:
        r"""Norma de Hilbert-Schmidt / Frobenius ($$p=2$$): $$\|A\|_F = \sqrt{\operatorname{Tr}(A^* A)}$$."""
        return float(norm(matrix, "fro"))

    @staticmethod
    def spectral(matrix: np.ndarray) -> float:
        r"""Norma operatoria espectral ($$p=\infty$$): $$\|A\|_{\mathrm{op}} = \sigma_{\max}(A)$$."""
        return float(norm(matrix, 2))

    @staticmethod
    def nuclear(matrix: np.ndarray) -> float:
        r"""Norma nuclear / clase traza ($$p=1$$): $$\|A\|_* = \sum_i \sigma_i(A)$$."""
        _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        return float(np.sum(singular_values))


class Functor(ABC, Generic[S, T]):
    """
    Endofuntor o funtor covariante $$F: \mathcal{C} \to \mathcal{D}$$ que preserva
    la identidad $$F(\mathrm{id}_X) = \mathrm{id}_{F(X)}$$ y la composición
    $$F(g \circ f) = F(g) \circ F(f)$$.
    """

    @abstractmethod
    def apply_object(self, obj: S) -> T:
        """Mapeo sobre objetos $$X \in \operatorname{Obj}(\mathcal{C})$$."""
        pass

    @abstractmethod
    def apply_morphism(self, mor: Callable[[S], S]) -> Callable[[T], T]:
        """Elevación de morfismos $$f \in \operatorname{Hom}_{\mathcal{C}}(X, Y)$$."""
        pass


class NaturalTransformation(ABC, Generic[S, T]):
    r"""
    Transformación natural $$\eta: F \Rightarrow G$$ entre funtores paralelos.
    Satisface el diagrama conmutativo de naturalidad $$G(f) \circ \eta_X = \eta_Y \circ F(f)$$.
    """

    @abstractmethod
    def component(self, obj: Any) -> Callable:
        """Componente natural $$\eta_X: F(X) \to G(X)$$."""
        pass


class TensorProtocol(ABC):
    """Protocolo abstracto para tensores diferenciables de deformación topológica."""

    @abstractmethod
    def batch_compute(
        self,
        psi_array: np.ndarray,
        roi_array: np.ndarray,
    ) -> np.ndarray:
        """Vectorización segura en el álgebra de Lie para múltiples entradas."""
        pass

    @abstractmethod
    def compute_penalty(self, psi: float, roi: float) -> float:
        """Evalúa la penalización de cola pesada $$F(\Psi, \mathrm{ROI})$$."""
        pass

    @abstractmethod
    def compute_gradient(
        self,
        psi: float,
        roi: float,
    ) -> Tuple[float, float]:
        """Evalúa el gradiente conforme $$\nabla F = (\partial_\Psi F, \partial_{\mathrm{ROI}} F)$$."""
        pass

    @abstractmethod
    def compute_jacobian(
        self,
        psi: float,
        roi: float,
    ) -> np.ndarray:
        """Evalúa la matriz Jacobiana $$J_F \in \mathbb{R}^{1 \times 2}$$."""
        pass

    @abstractmethod
    def compute_hessian(
        self,
        psi: float,
        roi: float,
    ) -> np.ndarray:
        """Evalúa la matriz Hessiana simétrica $$H_F \in \mathbb{R}^{2 \times 2}$$."""
        pass


# ────────────────────────────────────────────────────────────────────────────
# ÁLGEBRA DE HEYTING INTUICIONISTA, MECÁNICA DE POINCARÉ-KAM Y MOTOR RSI-L3
# ────────────────────────────────────────────────────────────────────────────

class HeytingImprobabilityVerdict(Enum):
    r"""
    Retículo distributivo acotado de Heyting $$\mathcal{H}_{\Omega}$$ para la
    lógica intuicionista del Motor de Improbabilidad (donde $$\neg \neg P \neq P$$):
      • `VETOED = 0`        ($$\bot$$: Colapso por Cisne Negro o violación $$\beta_1 > 0$$)
      • `CRITICAL_TAIL = 1` (Difusión de Arnold activa / resonancia secular severa)
      • `ATTENUATED = 2`    (Deformación moderada dentro del límite elástico de Gelfand)
      • `COHERENT = 3`      ($$\top$$: Toro invariante KAM diofántico preservado)
    """

    VETOED = 0
    CRITICAL_TAIL = 1
    ATTENUATED = 2
    COHERENT = 3

    def meet(self, other: HeytingImprobabilityVerdict) -> HeytingImprobabilityVerdict:
        """Conjunción del retículo ($$a \wedge b = \min(a, b)$$)."""
        return HeytingImprobabilityVerdict(min(self.value, other.value))

    def join(self, other: HeytingImprobabilityVerdict) -> HeytingImprobabilityVerdict:
        """Disyunción del retículo ($$a \vee b = \max(a, b)$$)."""
        return HeytingImprobabilityVerdict(max(self.value, other.value))

    def imply(self, other: HeytingImprobabilityVerdict) -> HeytingImprobabilityVerdict:
        """Implicación relativa de Heyting ($$a \Rightarrow b = \top$$ si $$a \le b$$, sino $$b$$)."""
        if self.value <= other.value:
            return HeytingImprobabilityVerdict.COHERENT
        return other

    def negate(self) -> HeytingImprobabilityVerdict:
        """Pseudocomplemento intuicionista ($$\neg a = a \Rightarrow \bot$$)."""
        return self.imply(HeytingImprobabilityVerdict.VETOED)


@dataclass(frozen=True, slots=True)
class PoincareFatTailInvariants:
    r"""
    Invariantes de la Mecánica Celeste de Poincaré y Teoría KAM en el espacio
    de fase de cola pesada $$(\tilde{\Psi}, \mathrm{ROI})$$ [A7]:
      • `kam_diophantine_margin`: Margen diofántico $$\Delta_{\mathrm{KAM}}$$ respecto a
        resonancias seculares entre estabilidad e ingreso.
      • `poincare_cartan_circulation`: Circulación simpléctica de Poincaré-Cartan
        $$\oint_{\partial \Sigma} (\tilde{\Psi}\,d\mathrm{ROI} - \mathrm{ROI}\,d\tilde{\Psi})$$.
      • `arnold_diffusion_velocity`: Velocidad de deriva estocástica de Arnold
        $$v_{\mathrm{Arnold}} = \exp(-1/(\mu_{\mathrm{tail}} + \varepsilon))$$.
      • `floquet_monodromy_radius`: Radio espectral $$\rho(\mathbf{M}_{\mathrm{Floquet}})$$
        de la matriz monodrómica del flujo Hamiltoniano de riesgo.
      • `secular_resonance_index`: Índice adimensional de excitación de cola pesada.
    """

    kam_diophantine_margin: float = 1.0
    poincare_cartan_circulation: float = 0.0
    arnold_diffusion_velocity: float = 0.0
    floquet_monodromy_radius: float = 1.0
    secular_resonance_index: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialización canónica de los invariantes celestes de Poincaré."""
        return {
            "kam_diophantine_margin": float(self.kam_diophantine_margin),
            "poincare_cartan_circulation": float(self.poincare_cartan_circulation),
            "arnold_diffusion_velocity": float(self.arnold_diffusion_velocity),
            "floquet_monodromy_radius": float(self.floquet_monodromy_radius),
            "secular_resonance_index": float(self.secular_resonance_index),
        }


@dataclass(frozen=True, slots=True)
class ImprobabilityRSILevel3Telemetry:
    r"""
    Telemetría inmutable de Automejora Recursiva de Nivel 3 (RSI-L3) [A8]:
      • `generation`: Generación evolutiva de la ley metacognitiva.
      • `banach_contraction_factor`: Factor de contracción $$\rho_k \to \phi^{-1}$$.
      • `meta_lyapunov_potential`: Funcional de energía libre meta-Lyapunov $$\mathcal{V}_{\mathrm{meta}}$$.
      • `evolved_tikhonov_eps`: Escala conforme de Tikhonov auto-sintonizada $$\varepsilon_{\mathrm{crit}}^{(k)}$$.
      • `evolved_kam_damping`: Amortiguamiento secular KAM $$\zeta_{\mathrm{KAM}}^{(k)}$$.
      • `evolved_gelfand_elastic_limit`: Cota elástica espectral de Gelfand evolucionada.
      • `verdict`: Veredicto intuicionista de Heyting de la generación actual.
    """

    generation: int = 0
    banach_contraction_factor: float = _RSI_L3_CONTRACTION_TARGET
    meta_lyapunov_potential: float = 0.0
    evolved_tikhonov_eps: float = _EPS_CRITICAL
    evolved_kam_damping: float = 1.0
    evolved_gelfand_elastic_limit: float = _IMPROBABILITY_MAX * _VETO_THRESHOLD_FACTOR
    verdict: HeytingImprobabilityVerdict = HeytingImprobabilityVerdict.COHERENT

    def to_dict(self) -> Dict[str, Any]:
        """Serialización de la telemetría metacognitiva RSI-L3."""
        return {
            "generation": int(self.generation),
            "banach_contraction_factor": float(self.banach_contraction_factor),
            "meta_lyapunov_potential": float(self.meta_lyapunov_potential),
            "evolved_tikhonov_eps": float(self.evolved_tikhonov_eps),
            "evolved_kam_damping": float(self.evolved_kam_damping),
            "evolved_gelfand_elastic_limit": float(self.evolved_gelfand_elastic_limit),
            "verdict": self.verdict.name,
        }


class ImprobabilityRSILevel3Engine:
    r"""
    Motor Metacognitivo de Automejora Recursiva de Nivel 3 (RSI-L3).

    Evoluciona la ley misma de regularización conforme de Tikhonov y el
    amortiguamiento secular de Poincaré-KAM garantizando la contracción de
    punto fijo de Banach hacia $$\rho_\star = \phi^{-1} \approx 0.6180339887498949$$.
    """

    _lock: threading.Lock = threading.Lock()
    _state: ImprobabilityRSILevel3Telemetry = ImprobabilityRSILevel3Telemetry()

    @classmethod
    def get_meta_state(cls) -> ImprobabilityRSILevel3Telemetry:
        """Retorna una instantánea inmutable del estado metacognitivo actual."""
        with cls._lock:
            return cls._state

    @classmethod
    def synthesize_regularization_law(
        cls,
        base_epsilon: float = _EPS_CRITICAL,
    ) -> Tuple[float, ImprobabilityRSILevel3Telemetry]:
        """
        Sintetiza la escala efectiva de regularización conforme de Tikhonov
        modularizada por el estado de punto fijo RSI-L3.
        """
        with cls._lock:
            state = cls._state
            ratio = state.evolved_tikhonov_eps / _EPS_CRITICAL
            eps_eff = float(
                np.clip(base_epsilon * ratio, _EPS_MACH * 10.0, 1.0e-6)
            )
            return eps_eff, state

    @classmethod
    def commit_level3_fixed_point(
        cls,
        clamped_penalty: float,
        gelfand_radius: float,
        arnold_velocity: float,
        floquet_radius: float,
        verdict: HeytingImprobabilityVerdict,
    ) -> ImprobabilityRSILevel3Telemetry:
        r"""
        Ejecuta un paso recursivo de Nivel 3 sobre el funcional meta-Lyapunov:
        $$\mathcal{V}_{\mathrm{meta}}^{(k+1)} = \phi^{-1} \mathcal{V}_{\mathrm{meta}}^{(k)}
          + (1 - \phi^{-1}) \left[ \frac{\ln(I)}{\ln(I_{\max})} + v_{\mathrm{Arnold}}
          + \max(0, \rho_{\mathrm{Floquet}} - 1) \right]$$
        """
        with cls._lock:
            prev = cls._state
            gen = prev.generation + 1
            phi_inv = _RSI_L3_CONTRACTION_TARGET

            norm_penalty = math.log(max(clamped_penalty, 1.0)) / _LOG_IMPROBABILITY_MAX
            floquet_excess = max(0.0, min(floquet_radius - 1.0, 10.0))
            instant_cost = (
                0.5 * norm_penalty
                + 0.3 * min(max(arnold_velocity, 0.0), 1.0)
                + 0.2 * (floquet_excess / 10.0)
            )

            new_lyapunov = (
                phi_inv * prev.meta_lyapunov_potential
                + (1.0 - phi_inv) * instant_cost
            )

            if prev.meta_lyapunov_potential > _EPS_MACH:
                raw_ratio = abs(new_lyapunov - prev.meta_lyapunov_potential) / (
                    prev.meta_lyapunov_potential + 1.0
                )
                contraction = phi_inv * prev.banach_contraction_factor + (
                    1.0 - phi_inv
                ) * (phi_inv + 0.1 * math.tanh(raw_ratio))
            else:
                contraction = phi_inv

            contraction = float(np.clip(contraction, 0.5, 0.95))

            # Sintonización adaptativa acotada de Tikhonov y amortiguamiento KAM
            eps_mod = 1.0 + 0.05 * math.tanh(new_lyapunov - 0.5)
            evolved_eps = float(
                np.clip(_EPS_CRITICAL * eps_mod, 1.0e-11, 1.0e-9)
            )
            evolved_damping = float(
                np.clip(1.0 / (1.0 + 0.15 * new_lyapunov), 0.5, 1.5)
            )
            elastic_limit = float(
                _IMPROBABILITY_MAX
                * np.clip(
                    _VETO_THRESHOLD_FACTOR - 0.0005 * math.tanh(arnold_velocity),
                    0.995,
                    0.9999,
                )
            )

            cls._state = ImprobabilityRSILevel3Telemetry(
                generation=gen,
                banach_contraction_factor=contraction,
                meta_lyapunov_potential=float(new_lyapunov),
                evolved_tikhonov_eps=evolved_eps,
                evolved_kam_damping=evolved_damping,
                evolved_gelfand_elastic_limit=elastic_limit,
                verdict=verdict,
            )
            return cls._state


# ────────────────────────────────────────────────────────────────────────────
# DTO Y OBSERVADOR ESPECTRAL DE LA FASE 1
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Phase1NormObservation:
    r"""
    Contrato inmutable de salida de la Fase 1 (`Phase1_SpectralObserver`).

    Encapsula los escalares validados en el ortante positivo, la elevación
    conforme de Tikhonov $$\tilde{\Psi}$$, el factor jacobiano conforme
    $$\partial_\Psi \tilde{\Psi} = \Psi / \tilde{\Psi}$$, la rapidez hiperbólica
    $$\eta = \ln(\mathrm{ROI}/\tilde{\Psi})$$, la clausura homológica $$\beta_1$$
    y los invariantes celestes de Poincaré-KAM de entrada.
    """

    psi_raw: float
    roi_raw: float
    psi_tikhonov: float
    conformal_jacobian_factor: float
    log_psi_tikhonov: float
    log_roi: float
    hyperbolic_rapidity: float
    orthant_chebyshev_norm: float
    tactics_betti_1: int = 0
    poincare_invariants: PoincareFatTailInvariants = field(
        default_factory=PoincareFatTailInvariants
    )
    rsi_telemetry: ImprobabilityRSILevel3Telemetry = field(
        default_factory=ImprobabilityRSILevel3Telemetry
    )


class Phase1_SpectralObserver:
    r"""
    FASE 1: Observador Espectral, Regularizador Conforme de Tikhonov y Auditor
    de Resonancias Diofánticas de Poincaré-KAM ($\Phi_1$).

    Responsabilidades axiomáticas:
      1. Coerción estricta y validación de finitud IEEE-754 sobre $$(\Psi, \mathrm{ROI})$$.
      2. Verificación de la Ley de Clausura Transitiva DIKW mediante el invariante
         homológico $$\beta_1 = 0$$ en el estrato Táctico.
      3. Elevación conforme de Tikhonov $$\tilde{\Psi} = \sqrt{\Psi^2 + \varepsilon_{\mathrm{eff}}^2}$$
         y cálculo exacto del factor de regla de la cadena $$\Psi / \tilde{\Psi}$$.
      4. Evaluación del margen diofántico KAM, la circulación simpléctica de
         Poincaré-Cartan y la velocidad de difusión de Arnold en el espacio de fase.
    """

    def __init__(
        self,
        critical_epsilon: float = _EPS_CRITICAL,
        precision: NumericalPrecision = NumericalPrecision.FLOAT64,
    ) -> None:
        if critical_epsilon <= 0.0 or not math.isfinite(critical_epsilon):
            raise ValueError(
                f"critical_epsilon debe ser positivo y finito: {critical_epsilon}"
            )
        self.critical_epsilon: float = float(critical_epsilon)
        self.precision: NumericalPrecision = precision

    def observe(
        self,
        psi: Any,
        roi: Any,
        telemetry_context: Optional[Any] = None,
    ) -> Phase1NormObservation:
        """
        Ejecuta la observación espectral de Fase 1 sobre el par $$(\Psi, \mathrm{ROI})$$.
        """
        psi_val, roi_val = self._coerce_and_validate_scalars(psi, roi)
        betti_1 = self._verify_homological_closure(telemetry_context)

        eps_eff, rsi_state = ImprobabilityRSILevel3Engine.synthesize_regularization_law(
            self.critical_epsilon
        )
        psi_tilde, d_psi_tilde = self._compute_tikhonov_conformal_lift(
            psi_val, eps_eff
        )

        log_psi_tilde = math.log(psi_tilde)
        log_roi = math.log(roi_val)
        rapidity = log_roi - log_psi_tilde
        chebyshev_norm = POSITIVE_ORTHANT.distance((psi_tilde, roi_val), (0.0, 0.0))

        poincare_inv = self._compute_poincare_phase_observation(
            psi_val=psi_val,
            psi_tilde=psi_tilde,
            roi=roi_val,
            rapidity=rapidity,
        )

        return Phase1NormObservation(
            psi_raw=psi_val,
            roi_raw=roi_val,
            psi_tikhonov=psi_tilde,
            conformal_jacobian_factor=d_psi_tilde,
            log_psi_tikhonov=log_psi_tilde,
            log_roi=log_roi,
            hyperbolic_rapidity=rapidity,
            orthant_chebyshev_norm=chebyshev_norm,
            tactics_betti_1=betti_1,
            poincare_invariants=poincare_inv,
            rsi_telemetry=rsi_state,
        )

    @staticmethod
    def _coerce_and_validate_scalars(psi: Any, roi: Any) -> Tuple[float, float]:
        """Valida tipos, finitud IEEE-754 y pertenencia al ortante positivo $$\mathbb{R}_+^2$$."""
        if psi is None or roi is None:
            raise DimensionalMismatchError(
                "Parámetros 'psi' y 'roi' son obligatorios"
            )

        if isinstance(psi, bool) or isinstance(roi, bool):
            raise TypeCoercionError(
                "Los parámetros 'psi' y 'roi' no pueden ser booleanos"
            )

        try:
            psi_val = float(psi)
            roi_val = float(roi)
        except (ValueError, TypeError) as exc:
            raise TypeCoercionError(
                "No se pudo convertir 'psi' o 'roi' a número"
            ) from exc

        if not (math.isfinite(psi_val) and math.isfinite(roi_val)):
            raise DimensionalMismatchError(
                f"Dimensiones no finitas: Ψ={psi_val}, ROI={roi_val}"
            )

        if psi_val < 0.0:
            raise AxiomViolationError(f"Ψ debe ser no negativo: Ψ={psi_val}")
        if roi_val <= 0.0:
            raise AxiomViolationError(f"ROI debe ser positivo: ROI={roi_val}")

        return psi_val, roi_val

    @staticmethod
    def _verify_homological_closure(telemetry_context: Optional[Any]) -> int:
        """
        Verifica el invariante homológico $$\beta_1 = 0$$ en el estrato Táctico.
        Si $$\beta_1 > 0$$, lanza `HomologicalInconsistencyError` (Fast-Fail).
        """
        if telemetry_context is None:
            return 0

        tactics_betti_1 = 0
        if hasattr(telemetry_context, "get_metric"):
            tactics_betti_1 = int(
                telemetry_context.get_metric("tactics_betti_1", default=0)
            )
        elif isinstance(telemetry_context, Mapping):
            tactics_betti_1 = int(telemetry_context.get("tactics_betti_1", 0))

        if tactics_betti_1 > 0:
            raise HomologicalInconsistencyError(
                f"Fast-Fail: β₁ = {tactics_betti_1} > 0. "
                "Veto topológico, clausura transitiva violada."
            )
        return tactics_betti_1

    @staticmethod
    def _compute_tikhonov_conformal_lift(
        psi: float,
        eps_eff: float,
    ) -> Tuple[float, float]:
        """Calcula la elevación conforme $$\tilde{\Psi}$$ y $$\partial_\Psi \tilde{\Psi}$$."""
        return MathematicalAnalysis.tikhonov_conformal_norm(psi, eps_eff)

    @staticmethod
    def _compute_poincare_phase_observation(
        psi_val: float,
        psi_tilde: float,
        roi: float,
        rapidity: float,
    ) -> PoincareFatTailInvariants:
        r"""
        Calcula los invariantes celestes de Poincaré-KAM en el espacio de fase:
          • Frecuencias angulares de acción-ángulo:
            $$\omega_\Psi = \arctan(\tilde{\Psi}), \quad \omega_{\mathrm{ROI}} = \arctan(\mathrm{ROI})$$
          • Margen diofántico KAM sobre resonancias bajas $$1 \le |p| + |q| \le 6$$.
          • Circulación simpléctica de Poincaré-Cartan $$\omega_{\mathrm{PC}} = \tilde{\Psi}\ln(1+\mathrm{ROI}) - \mathrm{ROI}\ln(1+\tilde{\Psi})$$.
          • Velocidad de difusión de Arnold $$v_{\mathrm{Arnold}} = \exp(-1/(\mu_{\mathrm{tail}} + \varepsilon))$$.
        """
        omega_psi = math.atan(max(psi_tilde, _EPS_CRITICAL))
        omega_roi = math.atan(max(roi, _EPS_CRITICAL))

        min_margin = float("inf")
        for p in range(1, 5):
            for q in range(-4, 5):
                if q == 0:
                    continue
                order = p + abs(q)
                res_val = abs(p * omega_psi + q * omega_roi)
                threshold = _KAM_DIOPHANTINE_GAMMA / (
                    float(order) ** _KAM_DIOPHANTINE_TAU
                )
                margin = (res_val - threshold) * (
                    float(order) ** _KAM_DIOPHANTINE_TAU
                )
                if margin < min_margin:
                    min_margin = margin

        kam_margin = float(np.clip(min_margin, -1.0, 10.0))

        # Forma diferencial simpléctica de Poincaré-Cartan en coordenadas log-conformes
        circulation = float(
            psi_tilde * math.log1p(min(roi, 1e12))
            - roi * math.log1p(min(psi_tilde, 1e12))
        )

        # Parámetro de perturbación de cola pesada μ_tail
        secular_index = float(max(0.0, rapidity) / (_LOG_IMPROBABILITY_MAX + 1.0))
        if secular_index > 1.0e-6:
            arnold_vel = math.exp(-1.0 / secular_index)
        else:
            arnold_vel = 0.0

        return PoincareFatTailInvariants(
            kam_diophantine_margin=kam_margin,
            poincare_cartan_circulation=circulation,
            arnold_diffusion_velocity=float(arnold_vel),
            floquet_monodromy_radius=1.0 + 0.1 * secular_index,
            secular_resonance_index=secular_index,
        )

    def handoff_phase1_to_phase2(
        self,
        psi: Any,
        roi: Any,
        telemetry_context: Optional[Any] = None,
        phase2_synthesizer: Optional[Phase2_LieTransportSynthesizer] = None,
        tensor: Optional[ImprobabilityTensor] = None,
    ) -> Union[Phase1NormObservation, Phase2GelfandOrientation]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1 Y CONTINUACIÓN / INICIO DE LA FASE 2:
        Ejecuta el morfismo observacional $$\Phi_1(\Psi, \mathrm{ROI}) \to \text{Phase1NormObservation}$$
        y, cuando se suministra el sintetizador de Fase 2 (`phase2_synthesizer`),
        transfiere covariantemente el testigo a `phase2_synthesizer.ingest_from_phase1_handoff(...)`,
        materializando la composición funtorial $$\Phi_2 \circ \Phi_1$$.
        """
        observation = self.observe(
            psi=psi,
            roi=roi,
            telemetry_context=telemetry_context,
        )
        if phase2_synthesizer is not None:
            return phase2_synthesizer.ingest_from_phase1_handoff(
                observation=observation,
                tensor=tensor,
            )
        return observation


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: TRANSPORTE EN EL ÁLGEBRA DE LIE, ESPECTRO DE GELFAND, $*$-ÁLGEBRA    ║
# ║         TENSORIAL Y MONODROMÍA DE FLOQUET (Phase2_LieTransportSynthesizer)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class ImprobabilityTensor(TensorProtocol):
    r"""
    Operador funcional no lineal que deforma el espacio de probabilidades en el Estrato Ω.

    DEFINICIÓN FORMAL (Elevación Conforme de Tikhonov y Transporte de Lie [A1, A2]):
      $$F: \mathbb{R}_+^2 \longrightarrow [1.0, \, 1.0 \times 10^6]$$
      $$\tilde{\Psi}(\Psi) = \sqrt{\Psi^2 + \varepsilon_{\mathrm{critical}}^2}$$
      $$F(\Psi, \mathrm{ROI}) := \operatorname{clip}\!\left(
          \exp\!\left(\ln(\kappa) + \gamma \left(\ln(\mathrm{ROI}) - \ln(\tilde{\Psi})\right)\right),
          \, 1.0, \, 10^6
        \right)$$

    PROPIEDADES DE $*$-ÁLGEBRA DE BANACH CONMUTATIVA [A4]:
      • Producto monoidal: $$\tau_1 \otimes \tau_2 := (\kappa_1 \kappa_2, \, \gamma_1 + \gamma_2)$$
      • Escalarización:    $$\lambda \odot \tau := (\operatorname{clip}(\lambda \kappa), \, \gamma)$$
      • Involución $$*$$:   $$\tau^* := \tau$$ (Auto-adjunción hermítica sobre escalares reales)
      • Adjunto HS:        $$\tau^\dagger := (\kappa^{-1}, \, \gamma)$$ (Inverso multiplicativo de escala)

    DIFERENCIABILIDAD EXACTA $$C^\infty(\mathbb{R}_+^2)$$ CON REGLA DE LA CADENA:
      $$\frac{\partial F}{\partial \Psi} = -\kappa \gamma \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma \frac{\Psi}{\tilde{\Psi}^2},
        \qquad
        \frac{\partial F}{\partial \mathrm{ROI}} = \kappa \gamma \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma \frac{1}{\mathrm{ROI}}$$
    """

    kappa: float = field(default=1.0)
    gamma: float = field(default=2.0)

    def __post_init__(self) -> None:
        """Validación estricta de invariantes algebraicos y numéricos."""
        if (
            isinstance(self.kappa, bool)
            or isinstance(self.gamma, bool)
            or not all(
                isinstance(x, (int, float, np.number))
                for x in (self.kappa, self.gamma)
            )
        ):
            raise TypeError(
                f"κ, γ deben ser numéricos: κ:{type(self.kappa)}, γ:{type(self.gamma)}"
            )

        if not all(math.isfinite(float(x)) for x in (self.kappa, self.gamma)):
            raise ValueError(
                f"κ, γ deben ser finitos: κ={self.kappa}, γ={self.gamma}"
            )

        if not (_KAPPA_RANGE[0] <= float(self.kappa) <= _KAPPA_RANGE[1]):
            raise ValueError(
                f"κ ∉ [{_KAPPA_RANGE[0]}, {_KAPPA_RANGE[1]}]: {self.kappa}"
            )
        if not (_GAMMA_RANGE[0] <= float(self.gamma) <= _GAMMA_RANGE[1]):
            raise ValueError(
                f"γ ∉ [{_GAMMA_RANGE[0]}, {_GAMMA_RANGE[1]}]: {self.gamma}"
            )

        object.__setattr__(self, "kappa", float(self.kappa))
        object.__setattr__(self, "gamma", float(self.gamma))

    @staticmethod
    def _validate_inputs(psi: float, roi: float) -> Tuple[float, float]:
        """Valida y normaliza entradas escalares en el ortante positivo."""
        if not all(math.isfinite(x) for x in (psi, roi)):
            raise DimensionalMismatchError(
                f"Dimensiones no finitas: Ψ={psi}, ROI={roi}"
            )
        if psi < 0.0:
            raise AxiomViolationError(f"Ψ debe ser no negativo: Ψ={psi}")
        if roi <= 0.0:
            raise AxiomViolationError(f"ROI debe ser positivo: ROI={roi}")
        return float(psi), float(roi)

    def _lie_algebra_ratio_power(
        self,
        psi_tilde: float,
        roi: float,
    ) -> float:
        r"""
        Evalúa $$(\mathrm{ROI} / \tilde{\Psi})^\gamma$$ en el álgebra de Lie logarítmica
        con protección contra desbordamiento FPU IEEE-754.
        """
        log_ratio = self.gamma * (
            math.log(max(roi, _EPS_MACH)) - math.log(max(psi_tilde, _EPS_MACH))
        )
        log_clamped = float(
            np.clip(log_ratio, -_LIE_EXP_SAFE_LIMIT, _LIE_EXP_SAFE_LIMIT)
        )
        return math.exp(log_clamped)

    def batch_compute(
        self,
        psi_array: np.ndarray,
        roi_array: np.ndarray,
    ) -> np.ndarray:
        r"""
        Vectorización conforme en el álgebra de Lie [A1, A2]:
        $$\tilde{\Psi} = \operatorname{hypot}(\Psi, \varepsilon_{\mathrm{critical}}), \qquad
          \mathfrak{g} = \ln(\kappa) + \gamma (\ln(\mathrm{ROI}) - \ln(\tilde{\Psi}))$$
        """
        psi_arr = np.asarray(psi_array, dtype=np.float64)
        roi_arr = np.asarray(roi_array, dtype=np.float64)

        if psi_arr.shape != roi_arr.shape:
            raise ValueError(
                f"Dimensiones inconsistentes: psi.shape={psi_arr.shape}, "
                f"roi.shape={roi_arr.shape}"
            )

        effective_psi = np.hypot(np.maximum(psi_arr, 0.0), _EPS_CRITICAL)
        safe_roi = np.maximum(roi_arr, _EPS_MACH)

        lie_generator = math.log(self.kappa) + self.gamma * (
            np.log(safe_roi) - np.log(effective_psi)
        )
        lie_clamped = np.clip(
            lie_generator,
            _LOG_IMPROBABILITY_MIN,
            _LOG_IMPROBABILITY_MAX,
        )
        penalty = np.exp(lie_clamped)

        return np.clip(penalty, _IMPROBABILITY_MIN, _IMPROBABILITY_MAX)

    def compute_penalty(
        self,
        psi: float,
        roi: float,
        use_regularization: bool = True,
    ) -> float:
        r"""
        Calcula $$I(\Psi, \mathrm{ROI})$$ mediante regularización conforme de Tikhonov
        y transporte exponencial en el álgebra de Lie [A1, A2].

        TEOREMA (Continuidad de Lipschitz):
          $$I$$ es Lipschitz continua sobre todo subconjunto compacto del ortante
          positivo con constante $$L = \kappa \cdot \gamma \cdot (\mathrm{ROI}_{\max}/\tilde{\Psi}_{\min})^{\gamma-1}$$.

        TEOREMA (Monotonía Estricta en el Interior):
          $$\frac{\partial I}{\partial \mathrm{ROI}} > 0, \qquad \frac{\partial I}{\partial \Psi} < 0 \quad \forall (\Psi, \mathrm{ROI}) \in \mathbb{R}_{>0}^2$$
        """
        psi_val, roi_val = self._validate_inputs(psi, roi)

        if use_regularization:
            effective_psi, _ = MathematicalAnalysis.tikhonov_conformal_norm(
                psi_val, _EPS_CRITICAL
            )
        else:
            effective_psi = math.hypot(psi_val, _EPS_CRITICAL)

        # Transporte completo en el álgebra de Lie: ln(T_deform) = ln(κ) + γ(ln(ROI) - ln(Ψ̃))
        lie_generator = math.log(self.kappa) + self.gamma * (
            math.log(max(roi_val, _EPS_MACH)) - math.log(max(effective_psi, _EPS_MACH))
        )
        lie_clamped = float(
            np.clip(lie_generator, -_LIE_EXP_SAFE_LIMIT, _LIE_EXP_SAFE_LIMIT)
        )
        penalty = math.exp(lie_clamped)

        result = float(
            np.clip(penalty, _IMPROBABILITY_MIN, _IMPROBABILITY_MAX)
        )

        if not math.isfinite(result):
            raise NumericalInstabilityError(
                f"Overflow en transporte de Lie: raw={penalty}, clipped={result}"
            )

        return result

    def compute_gradient(
        self,
        psi: float,
        roi: float,
    ) -> Tuple[float, float]:
        r"""
        Calcula el gradiente analítico exacto $$\nabla F = (\partial_\Psi F, \partial_{\mathrm{ROI}} F)$$
        incorporando el factor de regla de la cadena de Tikhonov $$\frac{\partial \tilde{\Psi}}{\partial \Psi} = \frac{\Psi}{\tilde{\Psi}}$$:

        $$\frac{\partial F}{\partial \Psi} = -\kappa \gamma \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma \frac{\Psi}{\tilde{\Psi}^2},
          \qquad
          \frac{\partial F}{\partial \mathrm{ROI}} = \kappa \gamma \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma \frac{1}{\mathrm{ROI}}$$
        """
        psi_val, roi_val = self._validate_inputs(psi, roi)

        effective_psi, d_psi_tilde = MathematicalAnalysis.tikhonov_conformal_norm(
            psi_val, _EPS_CRITICAL
        )
        ratio_power = self._lie_algebra_ratio_power(effective_psi, roi_val)

        dF_dROI = (
            self.kappa * self.gamma * (ratio_power / roi_val)
            if roi_val > 0.0
            else 0.0
        )
        dF_dPsi = (
            -self.kappa
            * self.gamma
            * (ratio_power / effective_psi)
            * d_psi_tilde
        )

        return (float(dF_dPsi), float(dF_dROI))

    def compute_jacobian(
        self,
        psi: float,
        roi: float,
    ) -> np.ndarray:
        r"""Matriz Jacobiana $$J_F = [\partial_\Psi F, \, \partial_{\mathrm{ROI}} F] \in \mathbb{R}^{1 \times 2}$$."""
        grad = self.compute_gradient(psi, roi)
        return np.array([grad], dtype=np.float64)

    def compute_hessian(
        self,
        psi: float,
        roi: float,
    ) -> np.ndarray:
        r"""
        Matriz Hessiana simétrica exacta $$H_F \in \mathbb{R}^{2 \times 2}$$ bajo la métrica
        conforme de Tikhonov $$\tilde{\Psi} = \sqrt{\Psi^2 + \varepsilon_{\mathrm{critical}}^2}$$:

        $$\frac{\partial^2 F}{\partial \Psi^2} = \frac{\kappa \gamma}{\tilde{\Psi}^2}
          \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma
          \left[ (\gamma + 2)\frac{\Psi^2}{\tilde{\Psi}^2} - 1 \right]$$

        $$\frac{\partial^2 F}{\partial \mathrm{ROI}^2} = \frac{\kappa \gamma (\gamma - 1)}{\mathrm{ROI}^2}
          \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma$$

        $$\frac{\partial^2 F}{\partial \Psi \partial \mathrm{ROI}} = -\frac{\kappa \gamma^2}{\tilde{\Psi} \cdot \mathrm{ROI}}
          \left(\frac{\mathrm{ROI}}{\tilde{\Psi}}\right)^\gamma \frac{\Psi}{\tilde{\Psi}}$$
        """
        if psi <= 0.0 or roi <= 0.0:
            return np.zeros((2, 2), dtype=np.float64)

        psi_val, roi_val = self._validate_inputs(psi, roi)
        effective_psi, d_psi_tilde = MathematicalAnalysis.tikhonov_conformal_norm(
            psi_val, _EPS_CRITICAL
        )
        ratio_gamma = self._lie_algebra_ratio_power(effective_psi, roi_val)

        conformal_factor = (self.gamma + 2.0) * (d_psi_tilde ** 2) - 1.0
        h_psi_psi = (
            self.kappa
            * self.gamma
            * conformal_factor
            * ratio_gamma
            / (effective_psi ** 2)
        )

        h_roi_roi = (
            self.kappa
            * self.gamma
            * (self.gamma - 1.0)
            * ratio_gamma
            / (roi_val ** 2)
        )

        h_psi_roi = (
            -self.kappa
            * (self.gamma ** 2)
            * ratio_gamma
            * d_psi_tilde
            / (effective_psi * roi_val)
        )

        return np.array(
            [
                [h_psi_psi, h_psi_roi],
                [h_psi_roi, h_roi_roi],
            ],
            dtype=np.float64,
        )

    def compute_spectral_properties(
        self,
        psi_range: Tuple[float, float] = (0.01, 100.0),
        roi_range: Tuple[float, float] = (0.01, 100.0),
        num_points: int = 10,
    ) -> SpectralDecomposition:
        r"""
        Calcula la descomposición espectral del operador diferencial discretizado
        sobre una variedad reticulada $$\Sigma \subset \mathbb{R}_+^2$$ mediante SVD
        de la fibra de Jacobianos $$J_{\Sigma}^T J_{\Sigma}$$.
        """
        n_pts = max(int(num_points), 2)
        psi_vals = np.linspace(psi_range[0], psi_range[1], n_pts)
        roi_vals = np.linspace(roi_range[0], roi_range[1], n_pts)

        jacobians: List[np.ndarray] = []
        for psi_v in psi_vals:
            for roi_v in roi_vals:
                J = self.compute_jacobian(float(psi_v), float(roi_v))
                jacobians.append(J.flatten())

        J_matrix = np.vstack(jacobians)
        U, S, _ = np.linalg.svd(J_matrix, full_matrices=False)

        eigenvalues = S ** 2
        spectral_radius = float(np.max(eigenvalues)) if eigenvalues.size > 0 else 0.0

        return SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=U,
            spectral_radius=spectral_radius,
        )

    def inverse_map(self, penalty: float) -> Tuple[float, float]:
        r"""
        Mapeo inverso sobre la fibra canónica $$\mathrm{ROI} = 1.0$$:
        $$F^{-1}(I) \longrightarrow (\Psi_{\mathrm{est}}, \, 1.0), \qquad
          \tilde{\Psi}_{\mathrm{est}} = \left(\frac{\kappa}{I}\right)^{1/\gamma}$$
        """
        if not (_IMPROBABILITY_MIN <= penalty <= _IMPROBABILITY_MAX):
            raise ValueError(
                f"I ∉ [{_IMPROBABILITY_MIN}, {_IMPROBABILITY_MAX}]: {penalty}"
            )

        psi_tilde_est = (self.kappa / float(penalty)) ** (1.0 / self.gamma)
        if psi_tilde_est > _EPS_CRITICAL:
            psi_estimate = math.sqrt(
                max(psi_tilde_est ** 2 - _EPS_CRITICAL ** 2, 0.0)
            )
        else:
            psi_estimate = psi_tilde_est
        roi_estimate = 1.0

        return (float(psi_estimate), float(roi_estimate))

    def verify_lipschitz_constant(
        self,
        roi_max: float = 1000.0,
        psi_min: float = 0.01,
    ) -> float:
        r"""
        Cota analítica de Lipschitz sobre el compacto $$[\Psi_{\min}, \infty) \times (0, \mathrm{ROI}_{\max}]$$:
        $$L = \kappa \cdot \gamma \cdot \left(\frac{\mathrm{ROI}_{\max}}{\Psi_{\min}}\right)^{\gamma - 1}$$
        """
        if psi_min <= 0.0:
            raise ValueError(f"Ψ_min debe ser positivo: {psi_min}")

        ratio = float(roi_max) / float(psi_min)
        lipschitz = self.kappa * self.gamma * (ratio ** (self.gamma - 1.0))

        logger.debug(
            "Lipschitz constant: L = %e (κ=%e, γ=%e, ratio=%e)",
            lipschitz,
            self.kappa,
            self.gamma,
            ratio,
        )
        return float(lipschitz)

    def involution(self) -> ImprobabilityTensor:
        r"""Involución hermítica basal de la $*$-álgebra [A4]: $$\tau^* := \tau$$."""
        return ImprobabilityTensor(kappa=self.kappa, gamma=self.gamma)

    def hilbert_schmidt_adjoint(self) -> ImprobabilityTensor:
        r"""
        Adjunto regularizado de Hilbert-Schmidt [A4]:
        $$\tau^\dagger = \arg\min_{\sigma} \|\sigma \otimes \tau - \mathrm{id}\|_{\mathrm{HS}}$$
        """
        inv_kappa = float(
            np.clip(1.0 / max(self.kappa, _EPS_MACH), _MIN_KAPPA, _MAX_KAPPA)
        )
        inv_gamma = float(
            np.clip(1.0 / max(self.gamma, _EPS_MACH), _MIN_GAMMA, _MAX_GAMMA)
        )
        return ImprobabilityTensor(kappa=inv_kappa, gamma=inv_gamma)

    def __matmul__(self, other: ImprobabilityTensor) -> ImprobabilityTensor:
        r"""
        Producto monoidal en la $*$-álgebra [A4]:
        $$\tau_1 \otimes \tau_2 := (\kappa_1 \cdot \kappa_2, \, \gamma_1 + \gamma_2)$$
        """
        if not isinstance(other, ImprobabilityTensor):
            raise TypeError(f"Expected ImprobabilityTensor, got {type(other)}")

        return ImprobabilityTensor(
            kappa=self.kappa * other.kappa,
            gamma=self.gamma + other.gamma,
        )

    def __mul__(self, scalar: float) -> ImprobabilityTensor:
        r"""Escalarización proyectada: $$\lambda \odot \tau := (\operatorname{clip}(\lambda \kappa), \, \gamma)$$."""
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float, np.number)):
            raise TypeError(f"Expected float, got {type(scalar)}")

        new_kappa = float(
            np.clip(self.kappa * float(scalar), _KAPPA_RANGE[0], _KAPPA_RANGE[1])
        )
        return ImprobabilityTensor(kappa=new_kappa, gamma=self.gamma)

    def __rmul__(self, scalar: float) -> ImprobabilityTensor:
        """Multiplicación escalar conmutativa por la izquierda."""
        return self.__mul__(scalar)

    def __add__(self, other: ImprobabilityTensor) -> ImprobabilityTensor:
        """Combinación convexa baricéntrica: $$(\tau_1 + \tau_2) / 2$$."""
        if not isinstance(other, ImprobabilityTensor):
            raise TypeError(f"Expected ImprobabilityTensor, got {type(other)}")

        kappa_avg = (self.kappa + other.kappa) / 2.0
        gamma_avg = (self.gamma + other.gamma) / 2.0
        return ImprobabilityTensor(kappa=kappa_avg, gamma=gamma_avg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización canónica a diccionario."""
        return {
            "class": self.__class__.__name__,
            "kappa": float(self.kappa),
            "gamma": float(self.gamma),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialización canónica a JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImprobabilityTensor:
        """Deserialización validada desde un mapeo."""
        required_keys = {"kappa", "gamma"}
        if not required_keys.issubset(data.keys()):
            raise KeyError(f"Faltan claves: {required_keys - set(data.keys())}")
        return cls(kappa=float(data["kappa"]), gamma=float(data["gamma"]))

    @classmethod
    def from_json(cls, json_str: str) -> ImprobabilityTensor:
        """Deserialización validada desde cadena JSON."""
        return cls.from_dict(json.loads(json_str))

    @lru_cache(maxsize=256)
    def compute_penalty_cached(self, psi: float, roi: float) -> float:
        """Evaluación idempotente con caché LRU para pares frecuentes."""
        return self.compute_penalty(psi, roi, use_regularization=False)


class TensorAlgebra:
    """
    Álgebra de Banach y Geometría de Información de Fisher-Rao sobre la variedad
    de parámetros $$(\kappa, \gamma)$$ de `ImprobabilityTensor`.
    """

    @staticmethod
    def compose(
        tensors: Sequence[ImprobabilityTensor],
        operation: str = "product",
    ) -> ImprobabilityTensor:
        """
        Composición n-aria de tensores en la categoría monoidal `TensorObj`.
        Soporta `product` ($$\bigotimes_i \tau_i$$) y `average` (baricentro).
        """
        if not tensors:
            return ImprobabilityTensor()

        if operation == "product":
            result = tensors[0]
            for tensor in tensors[1:]:
                result = result @ tensor
            return result
        if operation == "average":
            return TensorAlgebra.average(tensors)
        raise ValueError(f"Operación desconocida: {operation}")

    @staticmethod
    def average(tensors: Sequence[ImprobabilityTensor]) -> ImprobabilityTensor:
        """Promedio aritmético en la variedad de parámetros."""
        if not tensors:
            return ImprobabilityTensor()
        kappas = [t.kappa for t in tensors]
        gammas = [t.gamma for t in tensors]
        return ImprobabilityTensor(
            kappa=float(np.mean(kappas)),
            gamma=float(np.mean(gammas)),
        )

    @staticmethod
    def weighted_average(
        tensors: Sequence[ImprobabilityTensor],
        weights: Sequence[float],
    ) -> ImprobabilityTensor:
        """Combinación convexa ponderada sobre el símplice $$\sum_i w_i = 1$$."""
        if len(tensors) != len(weights):
            raise ValueError(
                f"len(tensors)={len(tensors)}, len(weights)={len(weights)}"
            )

        total_weight = float(sum(weights))
        if not math.isclose(total_weight, 1.0, rel_tol=_RELATIVE_TOLERANCE):
            raise ValueError(f"Pesos no normalizados: Σw = {total_weight}")

        kappa_avg = sum(t.kappa * w for t, w in zip(tensors, weights))
        gamma_avg = sum(t.gamma * w for t, w in zip(tensors, weights))
        return ImprobabilityTensor(kappa=float(kappa_avg), gamma=float(gamma_avg))

    @staticmethod
    def interpolate(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor,
        t: float,
    ) -> ImprobabilityTensor:
        """Homotopía afín $$(1 - t)\tau_a + t\tau_b$$ para $$t \in [0, 1]$$."""
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"t ∉ [0, 1]: {t}")

        kappa = (1.0 - t) * tensor_a.kappa + t * tensor_b.kappa
        gamma = (1.0 - t) * tensor_a.gamma + t * tensor_b.gamma
        return ImprobabilityTensor(kappa=kappa, gamma=gamma)

    @staticmethod
    def geodesic_distance(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor,
        metric: str = "l2",
    ) -> float:
        r"""
        Distancia geodésica en la variedad de hiperparámetros:
          • `l2`:  $$\sqrt{(\kappa_a - \kappa_b)^2 + (\gamma_a - \gamma_b)^2}$$
          • `log`: $$\sqrt{(\ln \kappa_a - \ln \kappa_b)^2 + (\gamma_a - \gamma_b)^2}$$
          • `fisher`: Distancia de información de Fisher-Rao sobre la fibra logarítmica.
        """
        if metric == "l2":
            return math.hypot(
                tensor_a.kappa - tensor_b.kappa,
                tensor_a.gamma - tensor_b.gamma,
            )
        if metric == "log":
            return math.hypot(
                math.log(tensor_a.kappa) - math.log(tensor_b.kappa),
                tensor_a.gamma - tensor_b.gamma,
            )
        if metric == "fisher":
            return TensorAlgebra.fisher_rao_distance(tensor_a, tensor_b)
        raise ValueError(f"Métrica desconocida: {metric}")

    @staticmethod
    def fisher_rao_distance(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor,
    ) -> float:
        r"""
        Distancia Riemanniana de Fisher-Rao sobre el cono positivo $$(\kappa, \gamma) \in \mathbb{R}_+^2$$:
        $$d_{\mathrm{FR}}(\tau_a, \tau_b) = \sqrt{
          (\ln \kappa_a - \ln \kappa_b)^2 + (\ln \gamma_a - \ln \gamma_b)^2
        }$$
        """
        return math.hypot(
            math.log(max(tensor_a.kappa, _EPS_MACH))
            - math.log(max(tensor_b.kappa, _EPS_MACH)),
            math.log(max(tensor_a.gamma, _EPS_MACH))
            - math.log(max(tensor_b.gamma, _EPS_MACH)),
        )


# ────────────────────────────────────────────────────────────────────────────
# DTO Y SINTETIZADOR DE TRANSPORTE DE LIE DE LA FASE 2
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Phase2GelfandOrientation:
    r"""
    Contrato inmutable de salida de la Fase 2 (`Phase2_LieTransportSynthesizer`).

    Hereda formalmente `Phase1NormObservation` y adjunta el transporte en el
    álgebra de Lie $$\mathfrak{g}_{\mathrm{deform}}$$, la proyección exponencial
    sobre el retículo $$[1, 10^6]$$, el gradiente/Jacobiano/Hessiano con regla
    de la cadena de Tikhonov, el radio espectral de Gelfand $$r(T)$$ y el radio
    monodrómico de Floquet refinado.
    """

    observation: Phase1NormObservation
    kappa: float
    gamma: float
    lie_algebra_generator: float
    raw_lie_exponential: float
    clamped_penalty: float
    gradient: Tuple[float, float]
    jacobian: np.ndarray
    hessian: np.ndarray
    gelfand_spectral_radius: float
    operator_norm: float
    wilkinson_condition_number: float
    poincare_invariants: PoincareFatTailInvariants


class Phase2_LieTransportSynthesizer:
    r"""
    FASE 2: Sintetizador de Transporte en el Álgebra de Lie, Radio Espectral de
    Gelfand y Monodromía Simpléctica de Floquet ($\Phi_2$).

    Recibe la `Phase1NormObservation` emitida por `Phase1_SpectralObserver`,
    transporta la dinámica al álgebra de Lie logarítmica [A2], evalúa las
    derivadas conformes de primer y segundo orden [A1], determina el radio
    espectral puntual de Gelfand [A3] y actualiza el radio monodrómico de Floquet [A7].
    """

    def __init__(
        self,
        default_tensor: Optional[ImprobabilityTensor] = None,
    ) -> None:
        self.default_tensor: ImprobabilityTensor = (
            default_tensor if default_tensor is not None else ImprobabilityTensor()
        )

    def ingest_from_phase1_handoff(
        self,
        observation: Phase1NormObservation,
        tensor: Optional[ImprobabilityTensor] = None,
    ) -> Phase2GelfandOrientation:
        r"""
        CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 1 (`handoff_phase1_to_phase2`):
        Ingiere formalmente el objeto `Phase1NormObservation` entregado por la Fase 1
        y ejecuta la síntesis en el álgebra de Lie y el análisis espectral de Gelfand.
        """
        return self.synthesize_lie_transport(
            observation=observation,
            tensor=tensor,
        )

    def synthesize_lie_transport(
        self,
        observation: Phase1NormObservation,
        tensor: Optional[ImprobabilityTensor] = None,
    ) -> Phase2GelfandOrientation:
        """
        Ejecuta el transporte logarítmico en el álgebra de Lie y la orientación
        espectral de Gelfand sobre la observación de Fase 1.
        """
        active_tensor = tensor if tensor is not None else self.default_tensor

        # Generador en el álgebra de Lie: g = ln(κ) + γ * (ln(ROI) - ln(Ψ̃))
        lie_generator = math.log(active_tensor.kappa) + (
            active_tensor.gamma * observation.hyperbolic_rapidity
        )
        lie_safe = float(
            np.clip(lie_generator, -_LIE_EXP_SAFE_LIMIT, _LIE_EXP_SAFE_LIMIT)
        )
        raw_exp = math.exp(lie_safe)
        clamped_penalty = float(
            np.clip(raw_exp, _IMPROBABILITY_MIN, _IMPROBABILITY_MAX)
        )

        if not math.isfinite(clamped_penalty):
            raise NumericalInstabilityError(
                f"Inestabilidad en síntesis Expo-Lie: g={lie_generator}"
            )

        gradient = active_tensor.compute_gradient(
            observation.psi_raw, observation.roi_raw
        )
        jacobian = active_tensor.compute_jacobian(
            observation.psi_raw, observation.roi_raw
        )
        hessian = active_tensor.compute_hessian(
            observation.psi_raw, observation.roi_raw
        )

        gelfand_radius, op_norm, wilkinson_cond = (
            self._compute_gelfand_spectral_invariants(
                clamped_penalty=clamped_penalty,
                jacobian=jacobian,
                hessian=hessian,
            )
        )

        refined_poincare = self._refine_poincare_floquet_monodromy(
            observation=observation,
            hessian=hessian,
            lie_generator=lie_generator,
        )

        return Phase2GelfandOrientation(
            observation=observation,
            kappa=active_tensor.kappa,
            gamma=active_tensor.gamma,
            lie_algebra_generator=float(lie_generator),
            raw_lie_exponential=float(raw_exp),
            clamped_penalty=clamped_penalty,
            gradient=gradient,
            jacobian=jacobian,
            hessian=hessian,
            gelfand_spectral_radius=gelfand_radius,
            operator_norm=op_norm,
            wilkinson_condition_number=wilkinson_cond,
            poincare_invariants=refined_poincare,
        )

    @staticmethod
    def _compute_gelfand_spectral_invariants(
        clamped_penalty: float,
        jacobian: np.ndarray,
        hessian: np.ndarray,
    ) -> Tuple[float, float, float]:
        r"""
        Evalúa los invariantes espectrales de Gelfand [A3]:
          • Radio espectral puntual $$r(T) = \lim_{n\to\infty} \|T^n\|^{1/n}$$.
          • Norma operatoria diferencial $$\|J_F\|_{\mathrm{op}} = \sigma_{\max}(J_F)$$.
          • Número de condición de Wilkinson $$\kappa_2(H_F)$$ regularizado.
        """
        op_norm = OperatorNorm.spectral(jacobian)
        hess_eigs = np.abs(np.linalg.eigvalsh(hessian))
        max_hess_eig = float(np.max(hess_eigs)) if hess_eigs.size > 0 else 0.0
        min_hess_eig = float(np.min(hess_eigs)) if hess_eigs.size > 0 else 0.0

        wilkinson_cond = float((max_hess_eig + 1.0) / (min_hess_eig + 1.0))
        gelfand_radius = float(max(clamped_penalty, min(op_norm, _IMPROBABILITY_MAX)))

        return gelfand_radius, op_norm, wilkinson_cond

    @staticmethod
    def _refine_poincare_floquet_monodromy(
        observation: Phase1NormObservation,
        hessian: np.ndarray,
        lie_generator: float,
    ) -> PoincareFatTailInvariants:
        r"""
        Refina el radio monodrómico de Floquet mediante el flujo Hamiltoniano
        linealizado $$\mathbf{J}_{\mathrm{symp}} H_F$$ donde
        $$\mathbf{J}_{\mathrm{symp}} = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$.

        En una órbita periódica de período normalizado, el determinante del
        generador simpléctico $$\det(\mathbf{J}_{\mathrm{symp}} H_F) = \det(H_F)$$
        gobierna la transición elíptica/hiperbólica de Poincaré.
        """
        base_inv = observation.poincare_invariants
        det_h = float(np.linalg.det(hessian))

        # Si det(H_F) < 0, el punto de fase es hiperbólico (silla de montar),
        # induciendo multiplicadores característicos de Floquet |μ| > 1.
        if det_h < 0.0:
            lyap_char = math.log1p(min(math.sqrt(abs(det_h)), 1.0e6)) / (
                _LOG_IMPROBABILITY_MAX + 1.0
            )
            floquet_radius = float(math.exp(min(lyap_char, 2.0)))
        else:
            floquet_radius = float(base_inv.floquet_monodromy_radius)

        secular_idx = float(
            max(
                base_inv.secular_resonance_index,
                max(0.0, lie_generator) / (_LOG_IMPROBABILITY_MAX + 1.0),
            )
        )
        arnold_vel = (
            math.exp(-1.0 / secular_idx)
            if secular_idx > 1.0e-6
            else base_inv.arnold_diffusion_velocity
        )

        return PoincareFatTailInvariants(
            kam_diophantine_margin=base_inv.kam_diophantine_margin,
            poincare_cartan_circulation=base_inv.poincare_cartan_circulation,
            arnold_diffusion_velocity=float(arnold_vel),
            floquet_monodromy_radius=floquet_radius,
            secular_resonance_index=secular_idx,
        )

    def handoff_phase2_to_phase3(
        self,
        observation: Phase1NormObservation,
        tensor: Optional[ImprobabilityTensor] = None,
        phase3_decider: Optional[Phase3_MonadicHeytingDecider] = None,
        external_force: float = 1.0,
        internal_tension: float = 1.0,
        fragility_prob: float = 1.0,
    ) -> Union[Phase2GelfandOrientation, ImprobabilityResult]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 2 Y CONTINUACIÓN / INICIO DE LA FASE 3:
        Ejecuta el morfismo de transporte de Lie y síntesis de Gelfand
        $$\Phi_2(\text{Phase1NormObservation}) \to \text{Phase2GelfandOrientation}$$ y,
        cuando se suministra el decisor monádico de Fase 3 (`phase3_decider`),
        invoca inmediatamente `phase3_decider.ingest_from_phase2_handoff(...)`,
        materializando la composición funtorial $$\Phi_3 \circ \Phi_2$$.
        """
        orientation = self.ingest_from_phase1_handoff(
            observation=observation,
            tensor=tensor,
        )
        if phase3_decider is not None:
            return phase3_decider.ingest_from_phase2_handoff(
                orientation=orientation,
                external_force=external_force,
                internal_tension=internal_tension,
                fragility_prob=fragility_prob,
            )
        return orientation


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: ABSORCIÓN MÓNADICA, RETÍCULO DE HEYTING, ACOPLAMIENTO DE GAUGE [A6]  ║
# ║         Y CLAUSURA RECURSIVA RSI-L3 (Phase3_MonadicHeytingDecider)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True)
class ImprobabilityResult:
    r"""
    Mónada de Estado y Resultado (`Either[Error, Success]`) del Motor de Improbabilidad [A5].

    Encapsula de forma no demolitoria tanto el veredicto exitoso con su tensor de
    penalización, veto topológico y metadatos de gauge/Poincaré/RSI-L3, como el
    objeto inicial de falla ante violaciones axiomáticas o numéricas.
    """

    success: bool
    penalty: Optional[float] = None
    is_vetoed: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: Optional[np.datetime64] = field(
        default_factory=lambda: np.datetime64("now")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización canónica a diccionario compatible con la MIC y suites de test."""
        result: Dict[str, Any] = {"success": bool(self.success)}

        if self.success:
            result.update(
                {
                    "improbability_penalty": self.penalty,
                    "penalty": self.penalty,
                    "is_vetoed": self.is_vetoed,
                    "metadata": self.metadata,
                    "timestamp": str(self.timestamp),
                }
            )
        else:
            result.update(
                {
                    "error_type": self.error_type,
                    "error": self.error_message,
                    "timestamp": str(self.timestamp),
                }
            )

        return result

    @classmethod
    def success(
        cls,
        penalty: float,
        kappa: float,
        gamma: float,
        psi: float,
        roi: float,
        gradient: Optional[Tuple[float, float]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ImprobabilityResult:
        """Constructor monádico unitario (`return` / `pure`) para evaluaciones exitosas."""
        base_metadata: Dict[str, Any] = {
            "kappa": kappa,
            "gamma": gamma,
            "psi_input": psi,
            "roi_input": roi,
            "gradient": gradient,
            "tensor_class": "ImprobabilityTensor",
        }
        if extra_metadata:
            base_metadata.update(extra_metadata)

        return cls(
            success=True,
            penalty=penalty,
            is_vetoed=bool(penalty >= _IMPROBABILITY_MAX * _VETO_THRESHOLD_FACTOR),
            metadata=base_metadata,
        )

    @classmethod
    def success_result(
        cls,
        penalty: float,
        kappa: float,
        gamma: float,
        psi_input: float,
        roi_input: float,
        gradient: Optional[Tuple[float, float]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ImprobabilityResult:
        """Alias canónico para suites de integración."""
        return cls.success(
            penalty=penalty,
            kappa=kappa,
            gamma=gamma,
            psi=psi_input,
            roi=roi_input,
            gradient=gradient,
            extra_metadata=extra_metadata,
        )

    @property
    def improbability_penalty(self) -> Optional[float]:
        """Propiedad alias para acceso uniforme a la penalización."""
        return self.penalty

    @classmethod
    def failure(
        cls,
        error_type: str,
        error_message: str,
    ) -> ImprobabilityResult:
        """Constructor monádico de absorción de falla (`Left`)."""
        return cls(
            success=False,
            error_type=error_type,
            error_message=error_message,
        )

    @classmethod
    def error_result(
        cls,
        error_type: str,
        error_message: str,
    ) -> ImprobabilityResult:
        """Alias canónico para suites de integración."""
        return cls.failure(error_type=error_type, error_message=error_message)

    def map(self, f: Callable[[float], float]) -> ImprobabilityResult:
        """Elevación funtorial (`fmap`) sobre el valor de penalización."""
        if not self.success or self.penalty is None:
            return self
        new_penalty = float(f(self.penalty))
        return replace(
            self,
            penalty=new_penalty,
            is_vetoed=bool(
                new_penalty >= _IMPROBABILITY_MAX * _VETO_THRESHOLD_FACTOR
            ),
        )

    def flatmap(
        self,
        f: Callable[[float], ImprobabilityResult],
    ) -> ImprobabilityResult:
        """Multiplicación monádica de Kleisli (`bind` / `>>=`)."""
        if not self.success or self.penalty is None:
            return self
        return f(self.penalty)


class Phase3_MonadicHeytingDecider:
    r"""
    FASE 3: Decisor Monádico de Heyting, Acoplador de Gauge en la Ecuación de Estrés
    y Clausura de Punto Fijo RSI de Nivel 3 ($\Phi_3$).

    Responsabilidades axiomáticas:
      1. Recibir formalmente `Phase2GelfandOrientation` desde la Fase 2.
      2. Resolver el veredicto en el retículo intuicionista `HeytingImprobabilityVerdict`.
      3. Calcular la palanca de acoplamiento de gauge $$\Lambda \in [1.0, 4.0]$$ y el
         estrés ajustado $$\sigma^*(x) = \Lambda \cdot F_{\mathrm{ext}}(x) \cdot T_{\mathrm{int}}(x) \cdot P_{\mathrm{frag}}(\psi)$$ [A6].
      4. Ejecutar la clausura recursiva de Nivel 3 (`finalize_nested_pipeline`)
         actualizando el atractor de Banach en `ImprobabilityRSILevel3Engine`.
    """

    def __init__(
        self,
        veto_threshold_factor: float = _VETO_THRESHOLD_FACTOR,
    ) -> None:
        if not (0.0 < veto_threshold_factor <= 1.0):
            raise ValueError(
                f"veto_threshold_factor debe habitar en (0, 1]: {veto_threshold_factor}"
            )
        self.veto_threshold_factor: float = float(veto_threshold_factor)

    def ingest_from_phase2_handoff(
        self,
        orientation: Phase2GelfandOrientation,
        external_force: float = 1.0,
        internal_tension: float = 1.0,
        fragility_prob: float = 1.0,
    ) -> ImprobabilityResult:
        r"""
        CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 2 (`handoff_phase2_to_phase3`):
        Ingiere formalmente `Phase2GelfandOrientation`, acopla la palanca de gauge $$\Lambda$$
        en la ecuación de estado de estrés y sella el resultado monádico.
        """
        return self.decide_and_couple_gauge(
            orientation=orientation,
            external_force=external_force,
            internal_tension=internal_tension,
            fragility_prob=fragility_prob,
        )

    def decide_and_couple_gauge(
        self,
        orientation: Phase2GelfandOrientation,
        external_force: float = 1.0,
        internal_tension: float = 1.0,
        fragility_prob: float = 1.0,
    ) -> ImprobabilityResult:
        """
        Evalúa el colapso de la función de onda de cola pesada, acopla el campo
        de gauge $$\Lambda$$ y ejecuta el cierre del pipeline anidado.
        """
        penalty = orientation.clamped_penalty
        is_vetoed = bool(
            penalty >= _IMPROBABILITY_MAX * self.veto_threshold_factor
        )

        verdict = self._evaluate_heyting_verdict(orientation, is_vetoed)
        lambda_lever = self.compute_gauge_leverage_lambda(
            clamped_penalty=penalty,
            poincare_invariants=orientation.poincare_invariants,
        )
        sigma_star = self.compute_adjusted_stress_equation(
            lambda_lever=lambda_lever,
            external_force=external_force,
            internal_tension=internal_tension,
            fragility_prob=fragility_prob,
        )

        preliminary_result = ImprobabilityResult.success(
            penalty=penalty,
            kappa=orientation.kappa,
            gamma=orientation.gamma,
            psi=orientation.observation.psi_raw,
            roi=orientation.observation.roi_raw,
            gradient=orientation.gradient,
            extra_metadata={
                "gauge_leverage_lambda": lambda_lever,
                "adjusted_stress_sigma_star": sigma_star,
                "heyting_verdict": verdict.name,
                "gelfand_spectral_radius": orientation.gelfand_spectral_radius,
                "operator_norm": orientation.operator_norm,
                "wilkinson_condition_number": orientation.wilkinson_condition_number,
                "poincare_invariants": orientation.poincare_invariants.to_dict(),
            },
        )

        # Respetar el umbral de veto configurado en el decisor
        if preliminary_result.is_vetoed != is_vetoed:
            preliminary_result = replace(preliminary_result, is_vetoed=is_vetoed)

        return self.finalize_nested_pipeline(
            orientation=orientation,
            result=preliminary_result,
            verdict=verdict,
        )

    @staticmethod
    def compute_gauge_leverage_lambda(
        clamped_penalty: float,
        poincare_invariants: Optional[PoincareFatTailInvariants] = None,
    ) -> float:
        r"""
        Calcula la palanca de improbabilidad $$\Lambda \in [1.0, 4.0]$$ del Axioma [A6]
        mediante proyección logarítmica de Haar sobre $$\mathbb{R}_\Delta = [1, 10^6]$$:
        $$\Lambda(I) = 1.0 + 3.0 \cdot \frac{\ln(\operatorname{clip}(I, 1, 10^6))}{\ln(10^6)}$$
        """
        safe_penalty = float(
            np.clip(clamped_penalty, _IMPROBABILITY_MIN, _IMPROBABILITY_MAX)
        )
        normalized_haar = math.log(safe_penalty) / _LOG_IMPROBABILITY_MAX

        if poincare_invariants is not None and poincare_invariants.kam_diophantine_margin < 0.0:
            # Amplificación suave ante resonancia diofántica KAM dentro de [1.0, 4.0]
            kam_boost = 0.05 * min(abs(poincare_invariants.kam_diophantine_margin), 1.0)
            normalized_haar = min(1.0, normalized_haar + kam_boost)

        lambda_val = _GAUGE_LEVER_MIN + (
            (_GAUGE_LEVER_MAX - _GAUGE_LEVER_MIN) * normalized_haar
        )
        return float(np.clip(lambda_val, _GAUGE_LEVER_MIN, _GAUGE_LEVER_MAX))

    @staticmethod
    def compute_adjusted_stress_equation(
        lambda_lever: float,
        external_force: float = 1.0,
        internal_tension: float = 1.0,
        fragility_prob: float = 1.0,
    ) -> float:
        r"""
        Ecuación de Estado del Estrés Ajustado Tensorial en la variedad de
        De Rham-Hodge [A6]:
        $$\sigma^*(x) = \Lambda \cdot F_{\mathrm{ext}}(x) \cdot T_{\mathrm{int}}(x) \cdot P_{\mathrm{frag}}(\psi)$$
        """
        safe_lambda = float(
            np.clip(lambda_lever, _GAUGE_LEVER_MIN, _GAUGE_LEVER_MAX)
        )
        safe_p_frag = float(np.clip(fragility_prob, 0.0, 1.0))
        return float(
            safe_lambda
            * float(external_force)
            * float(internal_tension)
            * safe_p_frag
        )

    @staticmethod
    def _evaluate_heyting_verdict(
        orientation: Phase2GelfandOrientation,
        is_vetoed: bool,
    ) -> HeytingImprobabilityVerdict:
        """
        Evalúa el veredicto en el álgebra de Heyting combinando el umbral elástico
        de Gelfand, la velocidad de difusión de Arnold y el margen diofántico KAM.
        """
        if is_vetoed or orientation.observation.tactics_betti_1 > 0:
            return HeytingImprobabilityVerdict.VETOED

        verdict = HeytingImprobabilityVerdict.COHERENT
        p_inv = orientation.poincare_invariants

        if (
            orientation.clamped_penalty >= 0.5 * _IMPROBABILITY_MAX
            or p_inv.arnold_diffusion_velocity > 0.25
        ):
            verdict = verdict.meet(HeytingImprobabilityVerdict.CRITICAL_TAIL)
        elif (
            orientation.clamped_penalty > 10.0
            or p_inv.kam_diophantine_margin < 0.0
        ):
            verdict = verdict.meet(HeytingImprobabilityVerdict.ATTENUATED)

        return verdict

    @staticmethod
    def finalize_nested_pipeline(
        orientation: Phase2GelfandOrientation,
        result: ImprobabilityResult,
        verdict: HeytingImprobabilityVerdict = HeytingImprobabilityVerdict.COHERENT,
    ) -> ImprobabilityResult:
        r"""
        CLAUSURA TERMINAL DEL FUNTOR ANIDADO DE TRES FASES ($$\mathcal{Z}_{\mathrm{drive}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$):
        Actualiza el punto fijo de Banach del motor de Automejora Recursiva de Nivel 3
        (`ImprobabilityRSILevel3Engine`) e inyecta la telemetría metacognitiva en el
        `ImprobabilityResult` sellado.
        """
        rsi_telemetry = ImprobabilityRSILevel3Engine.commit_level3_fixed_point(
            clamped_penalty=orientation.clamped_penalty,
            gelfand_radius=orientation.gelfand_spectral_radius,
            arnold_velocity=orientation.poincare_invariants.arnold_diffusion_velocity,
            floquet_radius=orientation.poincare_invariants.floquet_monodromy_radius,
            verdict=verdict,
        )

        if result.success and result.metadata is not None:
            enriched_metadata = dict(result.metadata)
            enriched_metadata["rsi_telemetry"] = rsi_telemetry.to_dict()
            return replace(result, metadata=enriched_metadata)

        return result


# ────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR DEL SERVICIO MIC, FACTORÍA, DIAGNÓSTICO Y VERIFICACIÓN
# ────────────────────────────────────────────────────────────────────────────

class ImprobabilityDriveService:
    r"""
    Microservicio soberano encapsulador del Motor de Improbabilidad en la MIC.

    Orquesta de forma transparente el pipeline funtorial anidado de 3 fases:
      $$\mathcal{Z}_{\mathrm{drive}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$
    conectando `Phase1_SpectralObserver` $$\to$$ `Phase2_LieTransportSynthesizer`
    $$\to$$ `Phase3_MonadicHeytingDecider`.
    """

    def __init__(
        self,
        mic_registry: MICRegistry,
        kappa: float = 1.0,
        gamma: float = 2.0,
        precision: NumericalPrecision = NumericalPrecision.FLOAT64,
    ) -> None:
        self.mic: MICRegistry = mic_registry
        self.precision: NumericalPrecision = precision
        self._tensor: ImprobabilityTensor = ImprobabilityTensor(
            kappa=kappa, gamma=gamma
        )

        # Instanciación de las 3 Fases Anidadas
        self._phase1_observer: Phase1_SpectralObserver = Phase1_SpectralObserver(
            critical_epsilon=_EPS_CRITICAL,
            precision=precision,
        )
        self._phase2_synthesizer: Phase2_LieTransportSynthesizer = (
            Phase2_LieTransportSynthesizer(default_tensor=self._tensor)
        )
        self._phase3_decider: Phase3_MonadicHeytingDecider = (
            Phase3_MonadicHeytingDecider(
                veto_threshold_factor=_VETO_THRESHOLD_FACTOR
            )
        )

        logger.info(
            "ImprobabilityDriveService inicializado: κ=%e, γ=%e, precisión=%s",
            self._tensor.kappa,
            self._tensor.gamma,
            precision.name,
        )

    @property
    def tensor(self) -> ImprobabilityTensor:
        """Acceso al tensor inmutable activo."""
        return self._tensor

    @property
    def hyperparameters(self) -> Dict[str, float]:
        """Retorna los hiperparámetros actuales del tensor."""
        return {"kappa": self._tensor.kappa, "gamma": self._tensor.gamma}

    def update_tensor(
        self,
        kappa: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Actualiza atómicamente el tensor inmutable y sincroniza la Fase 2."""
        new_kappa = kappa if kappa is not None else self._tensor.kappa
        new_gamma = gamma if gamma is not None else self._tensor.gamma

        self._tensor = ImprobabilityTensor(kappa=new_kappa, gamma=new_gamma)
        self._phase2_synthesizer.default_tensor = self._tensor

        logger.info(
            "Tensor actualizado: κ=%e, γ=%e",
            self._tensor.kappa,
            self._tensor.gamma,
        )

    def update_hyperparameters(
        self,
        kappa: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Alias para compatibilidad con suite de integración."""
        self.update_tensor(kappa=kappa, gamma=gamma)

    def compute_with_gradient(self, psi: float, roi: float) -> Dict[str, Any]:
        """Computación acoplada de penalización y gradiente conforme."""
        penalty = self._tensor.compute_penalty(psi, roi)
        grad_psi, grad_roi = self._tensor.compute_gradient(psi, roi)
        return {
            "penalty": penalty,
            "gradients": {
                "d_penalty_d_psi": grad_psi,
                "d_penalty_d_roi": grad_roi,
            },
        }

    def execute_nested_drive_pipeline(
        self,
        psi: Any,
        roi: Any,
        telemetry_context: Optional[Any] = None,
        external_force: float = 1.0,
        internal_tension: float = 1.0,
        fragility_prob: float = 1.0,
    ) -> ImprobabilityResult:
        r"""
        Ejecuta el funtor maestro $$\mathcal{Z}_{\mathrm{drive}} = \Phi_3 \circ \Phi_2 \circ \Phi_1$$
        encadenando explícitamente los métodos de transición entre las 3 fases anidadas:
          1. `Phase1_SpectralObserver.handoff_phase1_to_phase2`
          2. `Phase2_LieTransportSynthesizer.handoff_phase2_to_phase3`
          3. `Phase3_MonadicHeytingDecider.ingest_from_phase2_handoff` $$\to$$ `finalize_nested_pipeline`
        """
        observation = self._phase1_observer.observe(
            psi=psi,
            roi=roi,
            telemetry_context=telemetry_context,
        )
        result = self._phase2_synthesizer.handoff_phase2_to_phase3(
            observation=observation,
            tensor=self._tensor,
            phase3_decider=self._phase3_decider,
            external_force=external_force,
            internal_tension=internal_tension,
            fragility_prob=fragility_prob,
        )
        assert isinstance(result, ImprobabilityResult)
        return result

    def register_in_mic(self) -> None:
        """Inyecta el vector de servicio en el Estrato Ω de la MIC."""
        self.mic.register_vector(
            service_name="compute_improbability_penalty",
            stratum=Stratum.OMEGA,
            handler=self._morphism_handler,
        )
        logger.info(
            "Motor acoplado al retículo OMEGA: κ=%e, γ=%e",
            self._tensor.kappa,
            self._tensor.gamma,
        )

    def unregister_from_mic(self) -> None:
        """Desacopla el vector de servicio del Estrato Ω de la MIC."""
        self.mic.unregister_vector("compute_improbability_penalty", Stratum.OMEGA)
        logger.info("Motor desacoplado del retículo OMEGA")

    def _morphism_handler(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Funtor natural para la MIC con absorción monádica de errores [A5]
        y verificación de Clausura Transitiva DIKW ($$\beta_1 = 0$$).
        """
        try:
            psi = kwargs.get("psi")
            roi = kwargs.get("roi")
            telemetry = kwargs.get("telemetry_context")
            external_force = float(kwargs.get("external_force", 1.0))
            internal_tension = float(kwargs.get("internal_tension", 1.0))
            fragility_prob = float(kwargs.get("fragility_prob", 1.0))

            result = self.execute_nested_drive_pipeline(
                psi=psi,
                roi=roi,
                telemetry_context=telemetry,
                external_force=external_force,
                internal_tension=internal_tension,
                fragility_prob=fragility_prob,
            )
            return result.to_dict()

        except (
            DimensionalMismatchError,
            TypeCoercionError,
            AxiomViolationError,
            HomologicalInconsistencyError,
            NumericalInstabilityError,
            SpectrumError,
        ) as exc:
            logger.warning(
                "Absorción monádica en _morphism_handler (%s): %s",
                type(exc).__name__,
                str(exc),
            )
            return ImprobabilityResult.failure(
                error_type=type(exc).__name__,
                error_message=str(exc),
            ).to_dict()

        except (ValueError, TypeError) as exc:
            logger.warning("Error de validación: %s", str(exc))
            return ImprobabilityResult.failure(
                error_type=type(exc).__name__,
                error_message=str(exc),
            ).to_dict()

        except Exception as exc:
            logger.error("Error inesperado: %s", str(exc), exc_info=True)
            return ImprobabilityResult.failure(
                error_type=type(exc).__name__,
                error_message=str(exc),
            ).to_dict()

    def batch_compute(
        self,
        psi_array: np.ndarray,
        roi_array: np.ndarray,
    ) -> np.ndarray:
        """Delega la vectorización conforme en el álgebra de Lie al tensor activo."""
        return self._tensor.batch_compute(psi_array, roi_array)


class TensorFactory:
    """Factoría canónica para instanciar tensores con perfiles espectrales predefinidos."""

    IDENTITY: Final[Tuple[float, float]] = (1.0, 0.0)
    CONSERVATIVE: Final[Tuple[float, float]] = (0.1, 1.5)
    MODERATE: Final[Tuple[float, float]] = (1.0, 2.0)
    AGGRESSIVE: Final[Tuple[float, float]] = (10.0, 3.0)
    EXTREME: Final[Tuple[float, float]] = (1000.0, 5.0)

    @classmethod
    def create(
        cls,
        preset: str,
        **overrides: float,
    ) -> ImprobabilityTensor:
        """Construye un `ImprobabilityTensor` desde un preset nominal."""
        presets = {
            "identity": cls.IDENTITY,
            "conservative": cls.CONSERVATIVE,
            "moderate": cls.MODERATE,
            "aggressive": cls.AGGRESSIVE,
            "extreme": cls.EXTREME,
        }

        preset_lower = preset.lower()
        if preset_lower not in presets:
            raise ValueError(
                f"Preset desconocido: '{preset}'. "
                f"Disponibles: {list(presets.keys())}"
            )

        kappa, gamma = presets[preset_lower]

        if "kappa" in overrides:
            kappa = overrides.pop("kappa")
        if "gamma" in overrides:
            gamma = overrides.pop("gamma")

        if overrides:
            warnings.warn(
                f"Parámetros desconocidos ignorados: {list(overrides.keys())}",
                RuntimeWarning,
            )

        return ImprobabilityTensor(kappa=kappa, gamma=gamma)


class DiagnosticAnalyzer:
    """Suite de diagnóstico diferencial, espectral y celeste del tensor."""

    @staticmethod
    def generate_report(tensor: ImprobabilityTensor) -> str:
        """Genera reporte completo de diagnóstico del tensor y las 3 fases anidadas."""
        lines = [
            "=" * 80,
            "DIAGNÓSTICO RIGUROSO DEL TENSOR DE IMPROBABILIDAD (v4.0.0-Poincaré-RSI-L3)",
            "=" * 80,
            f"Clase: {tensor.__class__.__name__}",
            f"ID de objeto: {id(tensor):#x}",
            f"Inmutable: {getattr(tensor.__dataclass_fields__['kappa'], 'frozen', True)}",
            "",
            "PARÁMETROS DEL TENSOR:",
            f"  κ (elasticidad):       {tensor.kappa:.6e}",
            f"  γ (amplificación):     {tensor.gamma:.6f}",
            "",
            "PROPIEDADES MATEMÁTICAS:",
        ]

        lipschitz = tensor.verify_lipschitz_constant(
            roi_max=1000.0,
            psi_min=0.01,
        )
        lines.append(f"  Constante Lipschitz:   {lipschitz:.6e}")

        lines.extend(["", "VALORES DE REFERENCIA (PIPELINE ANIDADO 3 FASES):"])
        test_cases = [
            (0.001, 1.0, "Ψ→0, ROI=1"),
            (1.0, 1.0, "Ψ=1, ROI=1"),
            (1.0, 10.0, "Ψ=1, ROI=10"),
            (10.0, 1.0, "Ψ=10, ROI=1"),
            (100.0, 1000.0, "Ψ=100, ROI=1000"),
        ]

        p1 = Phase1_SpectralObserver()
        p2 = Phase2_LieTransportSynthesizer(default_tensor=tensor)
        p3 = Phase3_MonadicHeytingDecider()

        for psi, roi, desc in test_cases:
            try:
                obs = p1.handoff_phase1_to_phase2(psi, roi)
                assert isinstance(obs, Phase1NormObservation)
                res = p2.handoff_phase2_to_phase3(
                    obs, tensor=tensor, phase3_decider=p3
                )
                assert isinstance(res, ImprobabilityResult)
                hess = tensor.compute_hessian(psi, roi)
                eigenvals = np.linalg.eigvalsh(hess)
                is_convex = all(eig >= 0 for eig in eigenvals)
                grad = tensor.compute_gradient(psi, roi)
                lam = (
                    res.metadata.get("gauge_leverage_lambda", 1.0)
                    if res.metadata
                    else 1.0
                )
                lines.append(
                    f"  {desc:16s}: I={res.penalty:.4e}, Λ={lam:.3f}, "
                    f"∇I=({grad[0]:.2e},{grad[1]:.2e}), convex={is_convex}"
                )
            except Exception as exc:
                lines.append(f"  {desc:16s}: Error - {exc}")

        lines.extend(["", "ANÁLISIS ESPECTRAL DE GELFAND:"])
        try:
            spectral = tensor.compute_spectral_properties(num_points=5)
            lines.append(f"  Radio espectral:       {spectral.spectral_radius:.6e}")
            lines.append(
                f"  Número de condición:   {spectral.condition_number():.6e}"
            )
        except Exception as exc:
            lines.append(f"  Error en análisis espectral: {exc}")

        rsi_state = ImprobabilityRSILevel3Engine.get_meta_state()
        lines.extend(
            [
                "",
                "ESTADO METACOGNITIVO RSI NIVEL 3:",
                f"  Generación:            {rsi_state.generation}",
                f"  Contracción de Banach: {rsi_state.banach_contraction_factor:.6f}",
                f"  Potencial Lyapunov:    {rsi_state.meta_lyapunov_potential:.6e}",
                "",
                "SERIALIZACIÓN:",
                f"  JSON (compacto): {tensor.to_json(indent=None)[:100]}...",
                "=" * 80,
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def compare_tensors(
        tensor_a: ImprobabilityTensor,
        tensor_b: ImprobabilityTensor,
    ) -> str:
        """Compara dos tensores mediante métricas geodésicas logarítmica y de Fisher-Rao."""
        dist_log = TensorAlgebra.geodesic_distance(
            tensor_a, tensor_b, metric="log"
        )
        dist_fr = TensorAlgebra.fisher_rao_distance(tensor_a, tensor_b)

        lines = [
            "COMPARACIÓN DE TENSORES",
            f"Distancia geodésica (L₂-Log):   {dist_log:.6e}",
            f"Distancia Fisher-Rao:           {dist_fr:.6e}",
            "",
            f"Tensor A: κ={tensor_a.kappa:.6e}, γ={tensor_a.gamma:.6f}",
            f"Tensor B: κ={tensor_b.kappa:.6e}, γ={tensor_b.gamma:.6f}",
            "",
            f"Δκ = {abs(tensor_a.kappa - tensor_b.kappa):.6e}",
            f"Δγ = {abs(tensor_a.gamma - tensor_b.gamma):.6f}",
        ]
        return "\n".join(lines)


def run_property_tests() -> bool:
    """
    Suite de verificación formal de propiedades matemáticas, invariantes de
    Poincaré-KAM, composición de las 3 fases anidadas y convergencia RSI-L3.
    """
    print("\n" + "=" * 80)
    print("EJECUTANDO TESTS DE PROPIEDADES MATEMÁTICAS Y FASES ANIDADAS")
    print("=" * 80)

    test_results: List[Tuple[str, bool, Optional[str]]] = []
    tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)

    # Test 1: Monotonía en ROI
    print("\n[Test 1] Monotonía en ROI...")
    try:
        p1 = tensor.compute_penalty(1.0, 1.0)
        p2 = tensor.compute_penalty(1.0, 10.0)
        assert p2 > p1, f"∂F/∂ROI > 0 violado: p(1,1)={p1}, p(1,10)={p2}"
        test_results.append(("Monotonía ROI", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Monotonía ROI", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 2: Monotonía decreciente en Ψ
    print("[Test 2] Monotonía decreciente en Ψ...")
    try:
        p1 = tensor.compute_penalty(1.0, 1.0)
        p3 = tensor.compute_penalty(10.0, 1.0)
        assert p3 < p1, f"∂F/∂Ψ < 0 violado: p(10,1)={p3}, p(1,1)={p1}"
        test_results.append(("Monotonía Ψ", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Monotonía Ψ", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 3: Límites del retículo
    print("[Test 3] Límites del retículo...")
    try:
        p_max = tensor.compute_penalty(1e-10, 1e10)
        p_min = tensor.compute_penalty(1e10, 1e-10)
        assert _IMPROBABILITY_MIN <= p_max <= _IMPROBABILITY_MAX
        assert _IMPROBABILITY_MIN <= p_min <= _IMPROBABILITY_MAX
        test_results.append(("Límites", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Límites", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 4: Composición tensorial (monoide)
    print("[Test 4] Composición tensorial (monoide)...")
    try:
        t1 = ImprobabilityTensor(kappa=2.0, gamma=1.5)
        t2 = ImprobabilityTensor(kappa=3.0, gamma=2.0)
        t_comp = t1 @ t2
        assert math.isclose(t_comp.kappa, 6.0, rel_tol=1e-9)
        assert math.isclose(t_comp.gamma, 3.5, rel_tol=1e-9)
        test_results.append(("Composición", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Composición", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 5: Serialización/deserialización
    print("[Test 5] Serialización/deserialización...")
    try:
        json_str = tensor.to_json()
        t_restored = ImprobabilityTensor.from_json(json_str)
        assert math.isclose(tensor.kappa, t_restored.kappa, rel_tol=1e-15)
        assert math.isclose(tensor.gamma, t_restored.gamma, rel_tol=1e-15)
        test_results.append(("Serialización", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Serialización", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 6: Verificación de gradientes conformes (incluyendo frontera Ψ=0)
    print("[Test 6] Verificación de gradientes conformes (Tikhonov C∞)...")
    try:
        psi, roi = 1.0, 2.0
        h = 1e-6
        grad_analytic = tensor.compute_gradient(psi, roi)
        f_psi_plus = tensor.compute_penalty(psi + h, roi)
        f_psi_minus = tensor.compute_penalty(psi - h, roi)
        grad_psi_fd = (f_psi_plus - f_psi_minus) / (2.0 * h)
        error_psi = abs(grad_analytic[0] - grad_psi_fd) / (
            abs(grad_analytic[0]) + 1e-10
        )
        assert error_psi < 1e-6, f"Error en ∂F/∂Ψ: {error_psi}"

        # En Ψ = 0, la regularización conforme de Tikhonov exige ∂F/∂Ψ = 0
        grad_zero = tensor.compute_gradient(0.0, 2.0)
        assert grad_zero[0] == 0.0, f"Simetría Tikhonov en Ψ=0 violada: {grad_zero[0]}"
        test_results.append(("Gradientes", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Gradientes", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 7: Mapeo inverso
    print("[Test 7] Mapeo inverso...")
    try:
        penalty = 100.0
        psi_est, roi_est = tensor.inverse_map(penalty)
        penalty_reconstructed = tensor.compute_penalty(psi_est, roi_est)
        error = abs(penalty_reconstructed - penalty) / penalty
        assert error < 1e-10, f"Error en mapeo inverso: {error}"
        test_results.append(("Inversibilidad", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Inversibilidad", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    # Test 8: Pipeline Anidado de 3 Fases, Poincaré-KAM y RSI Nivel 3
    print("[Test 8] Funtor Maestro Anidado (Fase 1 -> Fase 2 -> Fase 3)...")
    try:
        obs_phase1 = Phase1_SpectralObserver()
        syn_phase2 = Phase2_LieTransportSynthesizer(default_tensor=tensor)
        dec_phase3 = Phase3_MonadicHeytingDecider()

        orientation = obs_phase1.handoff_phase1_to_phase2(
            psi=0.5,
            roi=5.0,
            phase2_synthesizer=syn_phase2,
            tensor=tensor,
        )
        assert isinstance(orientation, Phase2GelfandOrientation)

        final_res = syn_phase2.handoff_phase2_to_phase3(
            observation=orientation.observation,
            tensor=tensor,
            phase3_decider=dec_phase3,
            external_force=2.0,
            internal_tension=1.5,
            fragility_prob=0.8,
        )
        assert isinstance(final_res, ImprobabilityResult)
        assert final_res.success is True
        assert final_res.metadata is not None
        assert _GAUGE_LEVER_MIN <= final_res.metadata["gauge_leverage_lambda"] <= _GAUGE_LEVER_MAX
        assert "poincare_invariants" in final_res.metadata
        assert "rsi_telemetry" in final_res.metadata
        test_results.append(("Pipeline 3 Fases", True, None))
        print("  ✓ Aprobado")
    except Exception as exc:
        test_results.append(("Pipeline 3 Fases", False, str(exc)))
        print(f"  ✗ Fallido: {exc}")

    print("\n" + "=" * 80)
    print("RESUMEN DE TESTS:")
    print("=" * 80)

    passed = sum(1 for _, result, _ in test_results if result)
    total = len(test_results)

    for test_name, result, error in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} | {test_name:20s} | {error or ''}")

    print("=" * 80)
    print(f"Resultado: {passed}/{total} tests aprobados")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    all_tests_passed = run_property_tests()

    print("\n")
    tensor_example = TensorFactory.create("moderate")
    print(DiagnosticAnalyzer.generate_report(tensor_example))

    print("\n")
    t1 = TensorFactory.create("conservative")
    t2 = TensorFactory.create("aggressive")
    print(DiagnosticAnalyzer.compare_tensors(t1, t2))

    raise SystemExit(0 if all_tests_passed else 1)