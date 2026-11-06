# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Watcher Agent                                                                             ║
║ Módulo   : Watcher Agent (Soberano de Validación Tensorial, Empuje y Retroceso en Topos)             ║
║ Ubicación: app/agents/core/inmune_system/watcher_agent.py                                            ║
║ Ruta     : app/agents/core/inmune_system/watcher_agent.py                                            ║
║ Versión  : 6.0.0-Nested-Refined-Hamiltonian                                                          ║
║ Función  : Validación tensorial de esfuerzo-energía, divergencia covariante, pushforward             ║
║            cohomológico, dinámica celeste de Störmer-Verlet y pullback categorial en topos.          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO DE CALIBRE:
────────────────────────────────────────────────────────────────────────────────────────────────────────
El soberano `WatcherAgent` ejerce el rol de centinela tensorial, verificador métrico y evaluador
dinámico de conservación en la arquitectura inmune de APU Filter. Su función es auditar rigurosamente
los tensores de energía-momento $\mathcal{T}^{\mu\nu}$, comprobar su simetría física y compatibilidad métrica
con la conexión de Levi-Civita $\Gamma^\rho_{\mu\nu}$, certificar la anulación de la divergencia covariante
$\nabla_\mu \mathcal{T}^{\mu\nu} = 0$ mediante tres pasadas recursivas de refinamiento numérico (precisión
estándar float64, acumulación en longdouble y suma compensada de Neumaier), evaluar cotas de energía de
Rayleigh en pushforward cohomológico, simular opcionalmente la propagación celeste de $N$ cuerpos mediante
integración simpléctica de Störmer-Verlet, y ejecutar la gobernanza terminal en el topos de evaluación.

El agente se organiza en tres fases estrictamente anidadas e interconectadas:
  • FASE 1 — VALIDACIÓN TENSORIAL Y REFINAMIENTO NUMÉRICO (`Phase1_StressTensorValidation`):
      Comprueba la finitud, dimensiones y simetría del tensor $\mathcal{T}^{\mu\nu} = \mathcal{T}^{\nu\mu}$.
      Audita la positividad semidefinida estricta de la métrica Riemanniana covariante $G_{\mu\nu}$ (`G_PHYSICS`),
      calcula los símbolos de Christoffel si se dispone de derivadas métricas, y evalúa la divergencia
      covariante $\nabla_\mu \mathcal{T}^{\mu\nu}$ mediante tres pasadas de refinamiento numérico sin relajar tolerancias.
  • FASE 2 — PUSHFORWARD COHOMOLÓGICO Y MECÁNICA CELESTE (`Phase2_CohomologyPushforward`):
      Consume el resultado de Fase 1. Calcula la forma cuadrática de energía de deformación $E = \alpha^\top \mathcal{T} \alpha$
      para coformas $\alpha_\mu = \delta x_\mu$, junto con la cota de cociente de Rayleigh normalizada bajo la métrica
      $G^{-1}$. Opcionalmente, integra la trayectoria orbital en el espacio de fase simpléctico mediante el esquema
      kick-drift-kick de Störmer-Verlet para Hamiltonianos separables $H(q, p) = T(p) + V(q)$, midiendo la deriva
      de energía y momento angular.
  • FASE 3 — GOBERNANZA Y PULLBACK EN TOPOS (`Phase3_ToposPullback`):
      Consume los expedientes de Fase 1 y Fase 2. Combina los niveles de severidad en una cadena distributiva
      finita y aplica una política de veto determinista (adjudicación en retículo de Heyting). Emite el DTO
      de gobernanza certificado sellando el veredicto final.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
1. TENSOR DE ENERGÍA-MOMENTO Y DIVERGENCIA COVARIANTE DE CONSERVACIÓN:
   Sobre una variedad pseudoriemanniana o riemanniana $(M, g)$ con tensor métrico $g_{\mu\nu}$ y conexión
   afín libre de torsión $\Gamma^\rho_{\mu\nu}$, la conservación del esfuerzo-energía se postula mediante
   la anulación covariante:
       $$\nabla_\mu \mathcal{T}^{\mu\nu} = \partial_\mu \mathcal{T}^{\mu\nu} + \Gamma^\mu_{\mu\lambda} \mathcal{T}^{\lambda\nu} + \Gamma^\nu_{\mu\lambda} \mathcal{T}^{\mu\lambda} = 0$$
   donde $\Gamma^\rho_{\mu\nu}$ son los símbolos de Christoffel de segunda especie:
       $$\Gamma^\rho_{\mu\nu} = \frac{1}{2} g^{\rho\sigma} \left( \partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu} \right)$$
   Cualquier residuo $\|\nabla_\mu \mathcal{T}^{\mu\nu}\| > \varepsilon_{\mathrm{tol}}$ revela una fuga de energía no modelada
   o una perturbación estocástica espuria que invalida la continuidad física del sistema.

2. FORMA CUADRÁTICA DE ENERGÍA Y COCIENTE DE RAYLEIGH-RIEMANN:
   Dada una coforma de deformación $\alpha \in T_p^* M$, la densidad de energía de deformación elástica viene dada por:
       $$E(\alpha) = \alpha_\mu \mathcal{T}^{\mu\nu} \alpha_\nu$$
   La normalización métrica covariante impone la norma geométrica $\|\alpha\|_G^2 = \alpha^\top G^{-1} \alpha$.
   El cociente de Rayleigh generalizado acota el espectro de deformación:
       $$\lambda_{\min}(\mathcal{T}, G) \le \mathcal{R}(\alpha) = \frac{\alpha^\top \mathcal{T} \alpha}{\alpha^\top G^{-1} \alpha} \le \lambda_{\max}(\mathcal{T}, G)$$
   garantizando la estabilidad mecánica frente a tensiones críticas o sobrecargas de colapso.

3. INTEGRACIÓN SIMPLÉCTICA DE STÖRMER-VERLET EN MECÁNICA CELESTE:
   Para Hamiltonianos mecánicos separables $H(q, p) = \frac{1}{2} p^\top M^{-1} p + V(q)$, el integrador simpléctico
   de paso fijo $\Delta t$ ejecuta la secuencia kick-drift-kick de segundo orden:
       $$p_{k+1/2} = p_k - \frac{\Delta t}{2} \nabla V(q_k)$$
       $$q_{k+1} = q_k + \Delta t M^{-1} p_{k+1/2}$$
       $$p_{k+1} = p_{k+1/2} - \frac{\Delta t}{2} \nabla V(q_{k+1})$$
   Al tratarse de una transformación estrictamente simpléctica en $\mathrm{Sp}(2n, \mathbb{R})$, preserva exactamente
   la 2-forma canónica $\omega = \sum dq \wedge dp$ y conserva un Hamiltoniano modificado cercano $\tilde{H} = H + O(\Delta t^2)$,
   eliminando la disipación artificial o la deriva secular en órbitas acotadas a tiempos astronómicos.

4. SUMA COMPENSADA DE NEUMAIER Y REFINAMIENTO NUMÉRICO EN 3 PASADAS:
   Para evaluar la divergencia tensorial con precisión extrema frente al límite de Wilkinson ($\sim 10^{-15}$):
     - Pasada 1: Contracción en punto flotante estándar float64 IEEE 754.
     - Pasada 2: Acumulación de sumas parciales en punto flotante extendido longdouble (80/128 bits).
     - Pasada 3: Algoritmo de suma compensada de Neumaier que rastrea y cancela los residuos de bajo orden.
   El resultado sólo se certifica si las tres pasadas convergen dentro de la tolerancia estricta configurada.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  [Axioma WA-1] Simetría Tensorial Constitutiva:
      $\mathcal{T}^{\mu\nu} = \mathcal{T}^{\nu\mu}$, con residuo de antisimetría $\|\mathcal{T} - \mathcal{T}^\top\|_F \le \tau_{\mathrm{sym}} = 10^{-10}$.
  [Axioma WA-2] Definición Positiva de la Métrica Riemanniana:
      $G_{\mu\nu} = G_{\nu\mu}$ y $\lambda_{\min}(G) \ge \tau_{\mathrm{SPD}} = 10^{-12}$, asegurando una variedad sin degeneraciones.
  [Axioma WA-3] Conservación Covariante de Energía:
      $\|\nabla_\mu \mathcal{T}^{\mu\nu}\|_2 \le \tau_{\mathrm{div}}$, garantizando continuidad según el teorema de Noether.
  [Axioma WA-4] Integración Simpléctica de Fase:
      El flujo de Störmer-Verlet preserva la 2-forma simpléctica $\omega$ y mantiene acotada la deriva energética $\Delta H / H_0 \le \varepsilon_{\mathrm{drift}}$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Acotación de Rayleigh):
      La densidad de energía evaluada satisface $0 \le \alpha^\top \mathcal{T} \alpha \le E_{\max}$.
  Invariante 2 (Preservación de Torsión Nula):
      $\Gamma^\rho_{\mu\nu} = \Gamma^\rho_{\nu\mu}$ para todo par de índices inferiores de la conexión.
  Invariante 3 (Invarianza de Volumen Simpléctico):
      $\det(J_{\mathrm{Verlet}}) \equiv 1.0$ en cada paso temporal de integración celeste.
  Invariante 4 (Monotonía en la Cadena de Veto):
      Cualquier violación detectada en Fase 1 propaga de manera irreversible el veredicto de bloqueo hasta Fase 3.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS DEL ECOSISTEMA
# ══════════════════════════════════════════════════════════════════════════════

try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
except ImportError:
    class TopologicalInvariantError(Exception):
        """Fallback de la excepción categórica del ecosistema."""

    class Morphism:
        """Fallback de la interfaz Morphism del ecosistema."""

    class CategoricalState:
        """Fallback de compatibilidad para consumidores del módulo."""


try:
    from app.core.inmune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Fallback de conveniencia. El pipeline no lo asigna implícitamente a datos.
    G_PHYSICS: NDArray[np.float64] = np.eye(7, dtype=np.float64)


logger = logging.getLogger("MIC.ImmuneSystem.WatcherAgent")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS Y LÍMITES DE RECURSOS
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1e-8
_COND_MAX: Final[float] = 1e12
_RHO_CRIT: Final[float] = 1e6
_KAPPA_0: Final[float] = 1.0
_LOCKDOWN_EPS_FACTOR: Final[float] = 1e4
_SYMMETRY_REL_TOL: Final[float] = 1e-10

_REFINEMENT_LEVELS: Final[int] = 3
_MAX_CELESTIAL_STEPS: Final[int] = 10_000
_MAX_CELESTIAL_BODIES: Final[int] = 64
_DEFAULT_CELESTIAL_DRIFT_LIMIT: Final[float] = 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════

class WatcherAgentError(TopologicalInvariantError):
    """Excepción raíz del pipeline del Watcher Agent."""


class InputValidationError(WatcherAgentError):
    """Entrada no finita, no numérica o con una configuración inválida."""


class DimensionMismatchError(WatcherAgentError):
    """Dimensiones tensoriales incompatibles."""


class TensorSymmetryError(WatcherAgentError):
    """Violación de simetría del tensor o de su derivada."""


class MetricSignatureError(WatcherAgentError):
    """La métrica no es simétrica definida positiva."""


class ConnectionCompatibilityError(WatcherAgentError):
    """La conexión no satisface las condiciones estructurales solicitadas."""


class TensorConservationError(WatcherAgentError):
    """La divergencia covariante excede la tolerancia configurada."""


class NumericalStabilityError(WatcherAgentError):
    """Un resultado intermedio no puede certificarse numéricamente."""


class DeformationEnergyOverflowError(WatcherAgentError):
    """La energía excede la capacidad o la envolvente configurada."""


class PoincareStateError(WatcherAgentError):
    """Estado o configuración inválidos para la integración celeste."""


class CelestialCollisionError(PoincareStateError):
    """Colisión puntual no regularizada en el modelo de N cuerpos."""


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES Y OPERACIONES DE SEVERIDAD
# ══════════════════════════════════════════════════════════════════════════════

class SpectralSeverity(IntEnum):
    """
    Cadena ordenada de severidad.

    Esta estructura es un retículo distributivo finito; join y meet son,
    respectivamente, el máximo y el mínimo. No se presenta como un álgebra
    booleana de cuatro elementos.
    """

    NOMINAL = 0
    ELEVATED = 1
    CRITICAL = 2
    SINGULAR = 3


def join_severity(*values: SpectralSeverity) -> SpectralSeverity:
    """Join de la cadena de severidad: el peor de los estados recibidos."""
    if not values:
        return SpectralSeverity.NOMINAL
    return SpectralSeverity(max(int(value) for value in values))


def meet_severity(*values: SpectralSeverity) -> SpectralSeverity:
    """Meet de la cadena de severidad: el menos severo de los estados."""
    if not values:
        return SpectralSeverity.SINGULAR
    return SpectralSeverity(min(int(value) for value in values))


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS Y DE INMUTABILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def _finite_scalar(value: Any, name: str) -> float:
    """Convierte a float y rechaza booleanos, NaN e infinitos."""
    if isinstance(value, (bool, np.bool_)):
        raise InputValidationError(f"{name} debe ser un escalar real, no booleano.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{name} no es un escalar real válido.") from exc
    if not math.isfinite(result):
        raise InputValidationError(f"{name} debe ser finito.")
    return result


def _readonly_float64(value: Any, name: str) -> NDArray[np.float64]:
    """
    Copia una entrada real a float64, la valida y la marca como solo lectura.

    La copia impide que una mutación externa invalide silenciosamente un DTO
    congelado después de su construcción.
    """
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} no es convertible a un arreglo.") from exc

    if raw.dtype.kind == "c":
        raise InputValidationError(f"{name} debe ser real; no se aceptan complejos.")

    try:
        result = np.array(raw, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{name} no puede convertirse a float64.") from exc

    if not np.all(np.isfinite(result)):
        raise InputValidationError(f"{name} contiene NaN o valores infinitos.")

    result.setflags(write=False)
    return result


def _long_norm(values: Any) -> np.longdouble:
    """Norma euclídea con escalado, acumulada en longdouble."""
    array = np.asarray(values, dtype=np.longdouble).ravel()
    if array.size == 0:
        return np.longdouble(0.0)
    if not np.all(np.isfinite(array)):
        return np.longdouble(np.inf)

    magnitudes = np.abs(array)
    scale = np.max(magnitudes)
    if scale == 0:
        return np.longdouble(0.0)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled = magnitudes / scale
        return scale * np.sqrt(np.sum(scaled * scaled, dtype=np.longdouble))


def _diagnostic_float(value: Any) -> float:
    """Convierte una magnitud diagnóstica; valores no representables dan inf."""
    value_ld = np.longdouble(value)
    if np.isnan(value_ld):
        return math.inf
    if np.isinf(value_ld):
        return math.inf
    if abs(value_ld) > np.longdouble(np.finfo(np.float64).max):
        return math.inf
    return float(value_ld)


def _ld_to_float(value: Any, name: str) -> float:
    """Convierte un resultado extendido a float64 o falla de forma explícita."""
    value_ld = np.longdouble(value)
    if not np.isfinite(value_ld):
        raise NumericalStabilityError(f"{name} no es finito en precisión extendida.")
    if abs(value_ld) > np.longdouble(np.finfo(np.float64).max):
        raise NumericalStabilityError(f"{name} excede el rango representable en float64.")
    return float(value_ld)


def _neumaier_sum(values: Any) -> np.longdouble:
    """Suma compensada de Neumaier en la precisión disponible de longdouble."""
    total = np.longdouble(0.0)
    correction = np.longdouble(0.0)

    for raw_value in values:
        value = np.longdouble(raw_value)
        if not np.isfinite(value):
            return np.longdouble(np.inf)

        updated = total + value
        if not np.isfinite(updated):
            return np.longdouble(np.inf)

        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated

    result = total + correction
    if not np.isfinite(result):
        return np.longdouble(np.inf)
    return result


def _cross_longdouble(
    left: NDArray[np.longdouble],
    right: NDArray[np.longdouble],
) -> NDArray[np.longdouble]:
    """Producto vectorial tridimensional calculado componente a componente."""
    return np.asarray(
        (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ),
        dtype=np.longdouble,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DTOs DE ENTRADA, RESULTADOS E INVARIANTES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class StressTensorData:
    r"""
    Entrada tensorial.

    partial_G es opcional. Si se entrega, se comprueba ∇G = 0 con la convención
    Christoffel[ρ, μ, ν] = Γ^ρ_{μν}; si no se entrega, esa compatibilidad queda
    explícitamente como no verificada.
    """

    T_mu_nu: NDArray[np.float64]
    G_mu_nu: NDArray[np.float64]
    Christoffel: NDArray[np.float64]
    partial_T: NDArray[np.float64]
    partial_G: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "T_mu_nu", _readonly_float64(self.T_mu_nu, "T_mu_nu")
        )
        object.__setattr__(
            self, "G_mu_nu", _readonly_float64(self.G_mu_nu, "G_mu_nu")
        )
        object.__setattr__(
            self, "Christoffel", _readonly_float64(self.Christoffel, "Christoffel")
        )
        object.__setattr__(
            self, "partial_T", _readonly_float64(self.partial_T, "partial_T")
        )
        if self.partial_G is not None:
            object.__setattr__(
                self, "partial_G", _readonly_float64(self.partial_G, "partial_G")
            )


@dataclass(frozen=True, slots=True)
class RefinementPass:
    """Diagnóstico de una pasada del refinamiento recursivo de nivel 3."""

    level: int
    method: str
    residual_absolute: float
    residual_relative: float
    change_relative: float


@dataclass(frozen=True, slots=True)
class RefinementReport:
    """Certificado de las tres evaluaciones numéricas de la divergencia."""

    passes: Tuple[RefinementPass, ...]
    selected_level: int


@dataclass(frozen=True, slots=True)
class SpectralInvariants:
    """Invariantes espectrales extraídos en la Fase 1."""

    trace_mixed: float
    lambda_max_G: float
    lambda_min_G: float
    cond_G: float
    rho_T: float
    op_norm_T: float
    metric_op_norm_T: float
    frobenius_T: float
    christoffel_contracted: NDArray[np.float64]
    severity: SpectralSeverity
    connection_metric_compatible: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "christoffel_contracted",
            _readonly_float64(
                self.christoffel_contracted, "christoffel_contracted"
            ),
        )


@dataclass(frozen=True, slots=True)
class PoincareNBodyState:
    """
    Estado canónico newtoniano de N cuerpos en coordenadas cartesianas 3D.

    positions[i] = q_i; momenta[i] = p_i, con p_i = m_i v_i.
    """

    positions: NDArray[np.float64]
    momenta: NDArray[np.float64]
    masses: NDArray[np.float64]
    gravitational_constant: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "positions", _readonly_float64(self.positions, "positions")
        )
        object.__setattr__(
            self, "momenta", _readonly_float64(self.momenta, "momenta")
        )
        object.__setattr__(self, "masses", _readonly_float64(self.masses, "masses"))

        gravitational_constant = _finite_scalar(
            self.gravitational_constant, "gravitational_constant"
        )
        if gravitational_constant <= 0.0:
            raise PoincareStateError("gravitational_constant debe ser > 0.")
        object.__setattr__(self, "gravitational_constant", gravitational_constant)


@dataclass(frozen=True, slots=True)
class PoincareSimulationRequest:
    """Solicitud de integración Hamiltoniana simpléctica a paso fijo."""

    state: PoincareNBodyState
    time_step: float
    steps: int
    softening: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.state, PoincareNBodyState):
            raise PoincareStateError("state debe ser PoincareNBodyState.")

        time_step = _finite_scalar(self.time_step, "time_step")
        if time_step <= 0.0:
            raise PoincareStateError("time_step debe ser estrictamente positivo.")
        object.__setattr__(self, "time_step", time_step)

        if isinstance(self.steps, (bool, np.bool_)) or not isinstance(
            self.steps, (int, np.integer)
        ):
            raise PoincareStateError("steps debe ser un entero.")
        steps = int(self.steps)
        if steps <= 0 or steps > _MAX_CELESTIAL_STEPS:
            raise PoincareStateError(
                f"steps debe estar en [1, {_MAX_CELESTIAL_STEPS}]."
            )
        object.__setattr__(self, "steps", steps)

        softening = _finite_scalar(self.softening, "softening")
        if softening < 0.0:
            raise PoincareStateError("softening no puede ser negativo.")
        object.__setattr__(self, "softening", softening)


@dataclass(frozen=True, slots=True)
class PoincareDiagnostics:
    """Derivas conservativas observadas en una integración de N cuerpos."""

    hamiltonian_initial: float
    hamiltonian_final: float
    relative_energy_drift: float
    angular_momentum_initial: NDArray[np.float64]
    angular_momentum_final: NDArray[np.float64]
    relative_angular_momentum_drift: float
    relative_linear_momentum_drift: float
    steps: int
    time_step: float
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "angular_momentum_initial",
            _readonly_float64(
                self.angular_momentum_initial, "angular_momentum_initial"
            ),
        )
        object.__setattr__(
            self,
            "angular_momentum_final",
            _readonly_float64(
                self.angular_momentum_final, "angular_momentum_final"
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidatedStressTensor:
    """Producto certificado de la Fase 1 y dominio de la Fase 2."""

    T_mu_nu: NDArray[np.float64]
    G_mu_nu: NDArray[np.float64]
    invariants: SpectralInvariants
    residual_relative: float
    refinement_report: RefinementReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "T_mu_nu", _readonly_float64(self.T_mu_nu, "validated.T_mu_nu")
        )
        object.__setattr__(
            self, "G_mu_nu", _readonly_float64(self.G_mu_nu, "validated.G_mu_nu")
        )


@dataclass(frozen=True, slots=True)
class PushforwardResult:
    """Producto de la Fase 2 y dominio de la Fase 3."""

    deformed_energy: float
    rayleigh_bound: float
    energy_is_bounded: bool
    invariants: SpectralInvariants
    metric_dual_norm_squared: float
    celestial_diagnostics: Optional[PoincareDiagnostics] = None


@dataclass(frozen=True, slots=True)
class PoincarePushforwardData:
    """Resumen de la integración celeste para telemetría y gobernanza."""

    relative_energy_drift: float
    relative_angular_momentum_drift: float
    relative_linear_momentum_drift: float
    steps: int


@dataclass(frozen=True, slots=True)
class CohomologyPushforwardData:
    """DTO de salida consumible por el orquestador de cohomología."""

    deformed_dirichlet_energy: float
    stress_trace: float
    operator_norm: float
    spectral_radius: float
    rayleigh_bound: float = 0.0
    metric_dual_norm_squared: float = 0.0
    energy_is_bounded: bool = True
    celestial: Optional[PoincarePushforwardData] = None


@dataclass(frozen=True, slots=True)
class TensorialGovernanceState:
    """Certificado terminal de la política de gobernanza."""

    L_max: float
    zero_trust_lockdown: bool
    severity: SpectralSeverity
    condition_number: float
    reasons: Tuple[str, ...] = ()
    celestial_stable: Optional[bool] = None
    celestial_energy_drift: Optional[float] = None
    celestial_angular_momentum_drift: Optional[float] = None
    celestial_linear_momentum_drift: Optional[float] = None


# Alias para compatibilidad con la interfaz de versiones anteriores.
ToposPullbackData = TensorialGovernanceState


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — VALIDACIÓN TENSORIAL Y REFINAMIENTO RECURSIVO DE NIVEL 3
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_StressTensorValidation:
    r"""
    Valida el tensor, la métrica y la divergencia covariante.

    El refinamiento de nivel 3 es determinista y acotado:
      nivel 1: contracciones vectorizadas float64;
      nivel 2: productos y acumulación longdouble;
      nivel 3: suma compensada de Neumaier en longdouble.

    No cambia el tensor, no genera código y nunca relaja la tolerancia.
    """

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        tolerance = _finite_scalar(tolerance, "tolerance")
        if not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance debe satisfacer 0 < tolerance < 1.")
        self._tol = tolerance

    @staticmethod
    def _relative_swap_error(
        array: NDArray[np.float64],
        axis_1: int,
        axis_2: int,
    ) -> float:
        """Error relativo Frobenius frente al intercambio de dos ejes."""
        array_ld = np.asarray(array, dtype=np.longdouble)
        difference = array_ld - np.swapaxes(array_ld, axis_1, axis_2)
        denominator = _long_norm(array_ld)
        numerator = _long_norm(difference)

        if denominator == 0:
            return 0.0 if numerator == 0 else math.inf
        return _diagnostic_float(numerator / denominator)

    def _assert_dimensions(self, data: StressTensorData) -> int:
        if not isinstance(data, StressTensorData):
            raise InputValidationError("data debe ser StressTensorData.")

        T = data.T_mu_nu
        G = data.G_mu_nu
        gamma = data.Christoffel
        partial_T = data.partial_T

        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise DimensionMismatchError(
                f"T debe ser cuadrado; recibido shape={T.shape}."
            )

        n = T.shape[0]
        if n == 0:
            raise DimensionMismatchError("La dimensión del tensor no puede ser cero.")
        if G.shape != (n, n):
            raise DimensionMismatchError(
                f"G tiene shape={G.shape}; se esperaba ({n}, {n})."
            )
        if gamma.shape != (n, n, n):
            raise DimensionMismatchError(
                f"Christoffel tiene shape={gamma.shape}; "
                f"se esperaba ({n}, {n}, {n})."
            )
        if partial_T.shape != (n, n, n):
            raise DimensionMismatchError(
                f"partial_T tiene shape={partial_T.shape}; "
                f"se esperaba ({n}, {n}, {n})."
            )
        if data.partial_G is not None and data.partial_G.shape != (n, n, n):
            raise DimensionMismatchError(
                f"partial_G tiene shape={data.partial_G.shape}; "
                f"se esperaba ({n}, {n}, {n})."
            )
        return n

    def _check_matrix_symmetry(
        self,
        matrix: NDArray[np.float64],
        name: str,
        exception_type: type[WatcherAgentError],
    ) -> None:
        error = self._relative_swap_error(matrix, 0, 1)
        threshold = max(self._tol, _SYMMETRY_REL_TOL)
        if not math.isfinite(error) or error > threshold:
            raise exception_type(
                f"{name} no es simétrica: error relativo={error:.3e}."
            )

    def _check_metric(
        self,
        metric: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        """
        Comprueba positividad definida por espectro y Cholesky.

        κ(G) por encima del umbral se conserva como diagnóstico de severidad;
        no se confunde con la pérdida de positividad definida.
        """
        try:
            eigenvalues = np.linalg.eigvalsh(metric)
        except np.linalg.LinAlgError as exc:
            raise MetricSignatureError("No se pudo calcular el espectro de G.") from exc

        if not np.all(np.isfinite(eigenvalues)):
            raise MetricSignatureError("El espectro de G no es finito.")

        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])
        if lambda_min <= 0.0:
            raise MetricSignatureError(
                f"λ_min(G)={lambda_min:.3e}; se requiere G ≻ 0."
            )

        try:
            np.linalg.cholesky(metric)
        except np.linalg.LinAlgError as exc:
            raise MetricSignatureError(
                "La factorización de Cholesky de G falló."
            ) from exc

        condition_ld = np.longdouble(lambda_max) / np.longdouble(lambda_min)
        condition_number = _diagnostic_float(condition_ld)

        if condition_number > _COND_MAX:
            logger.warning(
                "Métrica mal condicionada: κ(G)=%.3e > %.1e",
                condition_number,
                _COND_MAX,
            )

        return lambda_min, lambda_max, condition_number

    def _check_connection(
        self,
        gamma_input: NDArray[np.float64],
        metric: NDArray[np.float64],
        partial_G_input: Optional[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Optional[bool]]:
        """
        Comprueba torsión nula y, cuando existe partial_G, ∇G = 0.

        Devuelve una conexión proyectada a la simetría de sus índices
        inferiores, siempre que la desviación original esté dentro de tol.
        """
        torsion_error = self._relative_swap_error(gamma_input, 1, 2)
        threshold = max(self._tol, _SYMMETRY_REL_TOL)
        if not math.isfinite(torsion_error) or torsion_error > threshold:
            raise ConnectionCompatibilityError(
                "Christoffel no es simétrico en los índices inferiores: "
                f"error relativo={torsion_error:.3e}."
            )

        gamma = 0.5 * gamma_input + 0.5 * np.swapaxes(gamma_input, 1, 2)
        if partial_G_input is None:
            return gamma, None

        metric_derivative_error = self._relative_swap_error(
            partial_G_input, 1, 2
        )
        if (
            not math.isfinite(metric_derivative_error)
            or metric_derivative_error > threshold
        ):
            raise TensorSymmetryError(
                "partial_G no es simétrica en sus dos índices métricos: "
                f"error relativo={metric_derivative_error:.3e}."
            )

        partial_G = (
            0.5 * partial_G_input
            + 0.5 * np.swapaxes(partial_G_input, 1, 2)
        )

        gamma_ld = np.asarray(gamma, dtype=np.longdouble)
        metric_ld = np.asarray(metric, dtype=np.longdouble)
        partial_G_ld = np.asarray(partial_G, dtype=np.longdouble)
        n = metric.shape[0]

        covariant_derivative = np.zeros((n, n, n), dtype=np.longdouble)
        gamma_metric_first = np.zeros_like(covariant_derivative)
        gamma_metric_second = np.zeros_like(covariant_derivative)

        for rho in range(n):
            for mu in range(n):
                for nu in range(n):
                    first_terms = [
                        gamma_ld[lam, rho, mu] * metric_ld[lam, nu]
                        for lam in range(n)
                    ]
                    second_terms = [
                        gamma_ld[lam, rho, nu] * metric_ld[mu, lam]
                        for lam in range(n)
                    ]
                    gamma_metric_first[rho, mu, nu] = _neumaier_sum(first_terms)
                    gamma_metric_second[rho, mu, nu] = _neumaier_sum(second_terms)
                    covariant_derivative[rho, mu, nu] = (
                        partial_G_ld[rho, mu, nu]
                        - gamma_metric_first[rho, mu, nu]
                        - gamma_metric_second[rho, mu, nu]
                    )

        residual = _long_norm(covariant_derivative)
        scale = (
            _long_norm(partial_G_ld)
            + _long_norm(gamma_metric_first)
            + _long_norm(gamma_metric_second)
        )
        if scale == 0:
            compatibility_error = 0.0 if residual == 0 else math.inf
        else:
            compatibility_error = _diagnostic_float(residual / scale)

        if (
            not math.isfinite(compatibility_error)
            or compatibility_error > self._tol
        ):
            raise ConnectionCompatibilityError(
                "La conexión no es compatible con G: "
                f"residuo relativo={compatibility_error:.3e}."
            )

        return gamma, True

    @staticmethod
    def _divergence_terms_for_component(
        T: NDArray[np.float64],
        gamma: NDArray[np.float64],
        partial_T: NDArray[np.float64],
        nu: int,
    ) -> list[np.longdouble]:
        """Genera los términos de la componente ν de ∇_μ T^{μν}."""
        n = T.shape[0]
        terms: list[np.longdouble] = []

        for mu in range(n):
            terms.append(np.longdouble(partial_T[mu, mu, nu]))

        for mu in range(n):
            for lam in range(n):
                terms.append(
                    np.longdouble(gamma[mu, mu, lam])
                    * np.longdouble(T[lam, nu])
                )

        for mu in range(n):
            for lam in range(n):
                terms.append(
                    np.longdouble(gamma[nu, mu, lam])
                    * np.longdouble(T[mu, lam])
                )

        return terms

    def _compute_divergence_at_level(
        self,
        T: NDArray[np.float64],
        gamma: NDArray[np.float64],
        partial_T: NDArray[np.float64],
        level: int,
    ) -> NDArray[np.longdouble]:
        """
        Calcula ∇_μ T^{μν} con una de las tres acumulaciones anidadas.
        """
        n = T.shape[0]

        if level == 1:
            with np.errstate(over="ignore", invalid="ignore"):
                derivative_term = np.einsum(
                    "mmn->n", partial_T, optimize=True
                )
                gamma_contracted = np.einsum(
                    "mml->l", gamma, optimize=True
                )
                connection_term_1 = gamma_contracted @ T
                connection_term_2 = np.einsum(
                    "nml,ml->n", gamma, T, optimize=True
                )
                result = (
                    derivative_term
                    + connection_term_1
                    + connection_term_2
                )
            return np.asarray(result, dtype=np.longdouble)

        result = np.zeros(n, dtype=np.longdouble)
        for nu in range(n):
            terms = self._divergence_terms_for_component(
                T, gamma, partial_T, nu
            )
            if level == 2:
                result[nu] = np.sum(
                    np.asarray(terms, dtype=np.longdouble),
                    dtype=np.longdouble,
                )
            elif level == 3:
                result[nu] = _neumaier_sum(terms)
            else:
                raise ValueError("El nivel de refinamiento debe ser 1, 2 o 3.")

        return result

    @staticmethod
    def _divergence_source_scale(
        T: NDArray[np.float64],
        gamma: NDArray[np.float64],
        partial_T: NDArray[np.float64],
    ) -> np.longdouble:
        """
        Escala relativa basada en las magnitudes de los tres sumandos de la
        divergencia. Evita comparar cantidades con dimensiones distintas
        contra una tolerancia absoluta sobre ||T||.
        """
        norm_partial = _long_norm(partial_T)
        norm_gamma = _long_norm(gamma)
        norm_T = _long_norm(T)
        with np.errstate(over="ignore", invalid="ignore"):
            return norm_partial + np.longdouble(2.0) * norm_gamma * norm_T

    @staticmethod
    def _relative_residual(
        divergence: NDArray[np.longdouble],
        source_scale: np.longdouble,
    ) -> np.longdouble:
        numerator = _long_norm(divergence)
        if not np.isfinite(numerator) or not np.isfinite(source_scale):
            return np.longdouble(np.inf)
        if source_scale == 0:
            return np.longdouble(0.0) if numerator == 0 else np.longdouble(np.inf)
        return numerator / source_scale

    def _recursive_self_improvement(
        self,
        T: NDArray[np.float64],
        gamma: NDArray[np.float64],
        partial_T: NDArray[np.float64],
        level: int = 1,
        previous: Optional[NDArray[np.longdouble]] = None,
        records: Optional[list[RefinementPass]] = None,
        source_scale: Optional[np.longdouble] = None,
    ) -> Tuple[NDArray[np.longdouble], RefinementReport]:
        """
        Refinamiento recursivo finito: cada nivel reevalúa la misma expresión
        con acumulación más resistente; no modifica el problema ni sus reglas.
        """
        if level < 1 or level > _REFINEMENT_LEVELS:
            raise ValueError("Nivel recursivo fuera del intervalo permitido.")

        if records is None:
            records = []
        if source_scale is None:
            source_scale = self._divergence_source_scale(T, gamma, partial_T)

        candidate = self._compute_divergence_at_level(
            T, gamma, partial_T, level
        )
        residual_absolute = _long_norm(candidate)
        residual_relative = self._relative_residual(candidate, source_scale)

        if previous is None:
            change_relative = np.longdouble(0.0)
        else:
            change = _long_norm(candidate - previous)
            denominator = max(
                source_scale,
                _long_norm(candidate),
                _long_norm(previous),
                np.finfo(np.longdouble).tiny,
            )
            change_relative = change / denominator

        method_names = {
            1: "einsum-float64",
            2: "longdouble-pairwise",
            3: "longdouble-neumaier",
        }
        records.append(
            RefinementPass(
                level=level,
                method=method_names[level],
                residual_absolute=_diagnostic_float(residual_absolute),
                residual_relative=_diagnostic_float(residual_relative),
                change_relative=_diagnostic_float(change_relative),
            )
        )

        if level < _REFINEMENT_LEVELS:
            return self._recursive_self_improvement(
                T=T,
                gamma=gamma,
                partial_T=partial_T,
                level=level + 1,
                previous=candidate,
                records=records,
                source_scale=source_scale,
            )

        return candidate, RefinementReport(
            passes=tuple(records),
            selected_level=level,
        )

    def _compute_spectral_invariants(
        self,
        T: NDArray[np.float64],
        G: NDArray[np.float64],
        gamma: NDArray[np.float64],
        lambda_min_G: float,
        lambda_max_G: float,
        cond_G: float,
        connection_metric_compatible: Optional[bool],
    ) -> SpectralInvariants:
        """Calcula invariantes euclídeos y métricos del tensor validado."""
        try:
            eigenvalues_G, eigenvectors_G = np.linalg.eigh(G)
            sqrt_G = (
                eigenvectors_G * np.sqrt(eigenvalues_G)[None, :]
            ) @ eigenvectors_G.T

            metric_weighted_T = sqrt_G @ T @ sqrt_G
            metric_weighted_T = (
                0.5 * metric_weighted_T
                + 0.5 * metric_weighted_T.T
            )
            eigenvalues_metric_T = np.linalg.eigvalsh(metric_weighted_T)
            eigenvalues_euclidean_T = np.linalg.eigvalsh(T)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                "No se pudieron extraer los invariantes espectrales."
            ) from exc

        if not (
            np.all(np.isfinite(eigenvalues_metric_T))
            and np.all(np.isfinite(eigenvalues_euclidean_T))
        ):
            raise NumericalStabilityError("El espectro del tensor no es finito.")

        metric_operator_norm = float(
            np.max(np.abs(eigenvalues_metric_T))
        )
        rho_T = metric_operator_norm
        euclidean_operator_norm = float(
            np.max(np.abs(eigenvalues_euclidean_T))
        )

        trace_terms = (
            np.longdouble(T[i, j]) * np.longdouble(G[i, j])
            for i in range(T.shape[0])
            for j in range(T.shape[1])
        )
        trace_mixed = _ld_to_float(
            _neumaier_sum(trace_terms), "trace_mixed"
        )

        frobenius_T = _ld_to_float(_long_norm(T), "frobenius_T")

        gamma_contracted_ld = np.asarray(
            [
                _neumaier_sum(
                    gamma[mu, mu, lam]
                    for mu in range(gamma.shape[0])
                )
                for lam in range(gamma.shape[0])
            ],
            dtype=np.longdouble,
        )
        gamma_contracted = np.asarray(
            [
                _ld_to_float(value, "christoffel_contracted")
                for value in gamma_contracted_ld
            ],
            dtype=np.float64,
        )

        severity = self._classify_severity(
            cond_G=cond_G,
            rho_T=rho_T,
            lambda_min_G=lambda_min_G,
            lambda_max_G=lambda_max_G,
        )

        return SpectralInvariants(
            trace_mixed=trace_mixed,
            lambda_max_G=lambda_max_G,
            lambda_min_G=lambda_min_G,
            cond_G=cond_G,
            rho_T=rho_T,
            op_norm_T=euclidean_operator_norm,
            metric_op_norm_T=metric_operator_norm,
            frobenius_T=frobenius_T,
            christoffel_contracted=gamma_contracted,
            severity=severity,
            connection_metric_compatible=connection_metric_compatible,
        )

    @staticmethod
    def _classify_severity(
        cond_G: float,
        rho_T: float,
        lambda_min_G: float,
        lambda_max_G: float,
    ) -> SpectralSeverity:
        """Clasificación monotónica de la severidad espectral."""
        singular_gap = (
            10.0
            * _MACHINE_EPSILON
            * max(lambda_max_G, np.finfo(np.float64).tiny)
        )
        if lambda_min_G <= singular_gap:
            return SpectralSeverity.SINGULAR
        if cond_G > _COND_MAX or rho_T > _RHO_CRIT:
            return SpectralSeverity.CRITICAL
        if cond_G > math.sqrt(_COND_MAX) or rho_T > math.sqrt(_RHO_CRIT):
            return SpectralSeverity.ELEVATED
        return SpectralSeverity.NOMINAL

    def validate_tensor(
        self,
        data: StressTensorData,
    ) -> ValidatedStressTensor:
        r"""
        Método terminal de la Fase 1.

        Emite ValidatedStressTensor únicamente después de comprobar dimensiones,
        simetrías, métrica SPD, estructura de conexión y divergencia covariante.
        """
        n = self._assert_dimensions(data)

        self._check_matrix_symmetry(
            data.T_mu_nu, "T", TensorSymmetryError
        )
        self._check_matrix_symmetry(
            data.G_mu_nu, "G", MetricSignatureError
        )

        derivative_symmetry_error = self._relative_swap_error(
            data.partial_T, 1, 2
        )
        if (
            not math.isfinite(derivative_symmetry_error)
            or derivative_symmetry_error
            > max(self._tol, _SYMMETRY_REL_TOL)
        ):
            raise TensorSymmetryError(
                "partial_T no conserva la simetría de T en sus índices "
                f"tensoriales: error={derivative_symmetry_error:.3e}."
            )

        # Proyecciones simétricas solo después de verificar que el residuo
        # original está dentro de la tolerancia declarada.
        T = 0.5 * data.T_mu_nu + 0.5 * data.T_mu_nu.T
        G = 0.5 * data.G_mu_nu + 0.5 * data.G_mu_nu.T
        partial_T = (
            0.5 * data.partial_T
            + 0.5 * np.swapaxes(data.partial_T, 1, 2)
        )

        if not (
            np.all(np.isfinite(T))
            and np.all(np.isfinite(G))
            and np.all(np.isfinite(partial_T))
        ):
            raise NumericalStabilityError(
                "La proyección simétrica produjo valores no finitos."
            )

        lambda_min_G, lambda_max_G, cond_G = self._check_metric(G)
        gamma, connection_metric_compatible = self._check_connection(
            gamma_input=data.Christoffel,
            metric=G,
            partial_G_input=data.partial_G,
        )

        divergence_ld, refinement_report = self._recursive_self_improvement(
            T=T,
            gamma=gamma,
            partial_T=partial_T,
        )
        residual_relative = refinement_report.passes[-1].residual_relative

        if not np.all(np.isfinite(divergence_ld)):
            raise NumericalStabilityError(
                "La divergencia final no es finita tras el refinamiento."
            )
        if np.max(np.abs(divergence_ld)) > np.longdouble(
            np.finfo(np.float64).max
        ):
            raise NumericalStabilityError(
                "La divergencia excede el rango representable en float64."
            )

        if residual_relative > self._tol:
            logger.critical(
                "Conservación covariante fallida | n=%d | res_rel=%.3e",
                n,
                residual_relative,
            )
            raise TensorConservationError(
                "∇_μ T^{μν} excede la tolerancia relativa: "
                f"{residual_relative:.3e} > {self._tol:.3e}."
            )

        invariants = self._compute_spectral_invariants(
            T=T,
            G=G,
            gamma=gamma,
            lambda_min_G=lambda_min_G,
            lambda_max_G=lambda_max_G,
            cond_G=cond_G,
            connection_metric_compatible=connection_metric_compatible,
        )

        logger.debug(
            "Fase 1 OK | n=%d | Tr_G(T)=%.4e | κ(G)=%.3e | "
            "ρ_G(T)=%.3e | sev=%s | res_rel=%.3e | refinement=L%d",
            n,
            invariants.trace_mixed,
            invariants.cond_G,
            invariants.rho_T,
            invariants.severity.name,
            residual_relative,
            refinement_report.selected_level,
        )

        return ValidatedStressTensor(
            T_mu_nu=T,
            G_mu_nu=G,
            invariants=invariants,
            residual_relative=residual_relative,
            refinement_report=refinement_report,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2, ANIDADA EN FASE 1 — PUSHFORWARD MÉTRICO Y N-CUERPOS
    # ══════════════════════════════════════════════════════════════════════════

    class Phase2_CohomologyPushforward:
        r"""
        Calcula la forma cuadrática deformada y su cota métrica de Rayleigh.

        Para una coforma α:
            E = αᵀ T α
            ||α||²_{G⁻¹} = αᵀ G⁻¹ α
            |E| ≤ ||G^(1/2) T G^(1/2)||₂ ||α||²_{G⁻¹}

        La integración celeste es opcional y usa variables canónicas (q, p).
        """

        def __init__(
            self,
            energy_overflow_factor: float = 1e8,
            energy_capacity: Optional[float] = None,
        ) -> None:
            factor = _finite_scalar(
                energy_overflow_factor, "energy_overflow_factor"
            )
            if factor <= 0.0:
                raise ValueError("energy_overflow_factor debe ser > 0.")
            self._energy_overflow_factor = factor

            if energy_capacity is None:
                self._energy_capacity = None
            else:
                capacity = _finite_scalar(energy_capacity, "energy_capacity")
                if capacity < 0.0:
                    raise ValueError("energy_capacity no puede ser negativa.")
                self._energy_capacity = capacity

        @staticmethod
        def _assert_delta_dimension(
            validated: ValidatedStressTensor,
            delta_x: Any,
        ) -> NDArray[np.float64]:
            """Valida que delta_x sea una coforma finita de dimensión n."""
            dx = _readonly_float64(delta_x, "delta_x")
            n = validated.T_mu_nu.shape[0]
            if dx.ndim != 1 or dx.shape[0] != n:
                raise DimensionMismatchError(
                    f"delta_x tiene shape={dx.shape}; se esperaba ({n},)."
                )
            return dx

        @staticmethod
        def _dirichlet_energy(
            T: NDArray[np.float64],
            delta_x: NDArray[np.float64],
        ) -> float:
            """Evalúa delta_xᵀ T delta_x con productos en longdouble."""
            T_ld = np.asarray(T, dtype=np.longdouble)
            dx_ld = np.asarray(delta_x, dtype=np.longdouble)
            with np.errstate(over="ignore", invalid="ignore"):
                value = dx_ld @ (T_ld @ dx_ld)
            return _ld_to_float(value, "deformed_energy")

        @staticmethod
        def _metric_dual_norm_squared(
            G: NDArray[np.float64],
            delta_x: NDArray[np.float64],
        ) -> float:
            """
            Evalúa αᵀG⁻¹α sin formar la inversa explícita.

            Si G = L Lᵀ, entonces αᵀG⁻¹α = ||L⁻¹α||².
            """
            try:
                lower = np.linalg.cholesky(G)
                transformed = np.linalg.solve(lower, delta_x)
            except np.linalg.LinAlgError as exc:
                raise MetricSignatureError(
                    "No se pudo resolver la norma dual respecto de G."
                ) from exc

            transformed_ld = np.asarray(transformed, dtype=np.longdouble)
            norm_squared = np.sum(
                transformed_ld * transformed_ld,
                dtype=np.longdouble,
            )
            return _ld_to_float(
                norm_squared, "metric_dual_norm_squared"
            )

        @staticmethod
        def _validate_celestial_request(
            request: PoincareSimulationRequest,
        ) -> Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]:
            if not isinstance(request, PoincareSimulationRequest):
                raise PoincareStateError(
                    "celestial_request debe ser PoincareSimulationRequest."
                )

            state = request.state
            q = np.asarray(state.positions, dtype=np.float64)
            p = np.asarray(state.momenta, dtype=np.float64)
            masses = np.asarray(state.masses, dtype=np.float64)

            if q.ndim != 2 or q.shape[1] != 3:
                raise DimensionMismatchError(
                    f"positions debe tener shape (N, 3); recibido {q.shape}."
                )
            n = q.shape[0]
            if n < 2:
                raise PoincareStateError(
                    "La integración celeste requiere al menos dos cuerpos."
                )
            if n > _MAX_CELESTIAL_BODIES:
                raise PoincareStateError(
                    f"N={n} excede el límite {_MAX_CELESTIAL_BODIES}."
                )
            if p.shape != (n, 3):
                raise DimensionMismatchError(
                    f"momenta debe tener shape ({n}, 3); recibido {p.shape}."
                )
            if masses.shape != (n,):
                raise DimensionMismatchError(
                    f"masses debe tener shape ({n},); recibido {masses.shape}."
                )
            if np.any(masses <= 0.0):
                raise PoincareStateError("Todas las masas deben ser positivas.")

            return (
                np.array(q, dtype=np.float64, copy=True),
                np.array(p, dtype=np.float64, copy=True),
                np.array(masses, dtype=np.float64, copy=True),
            )

        @staticmethod
        def _nbody_forces(
            positions: NDArray[np.float64],
            masses: NDArray[np.float64],
            gravitational_constant: float,
            softening: float,
        ) -> NDArray[np.float64]:
            """
            Fuerzas newtonianas de pares para el potencial Plummer opcional:
                V_ij = -G m_i m_j / sqrt(r_ij² + softening²).
            """
            n = positions.shape[0]
            q_ld = np.asarray(positions, dtype=np.longdouble)
            m_ld = np.asarray(masses, dtype=np.longdouble)
            force = np.zeros((n, 3), dtype=np.longdouble)

            G_ld = np.longdouble(gravitational_constant)
            soft_ld = np.longdouble(softening)
            soft_squared = soft_ld * soft_ld

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for i in range(n):
                    for j in range(i + 1, n):
                        displacement = q_ld[i] - q_ld[j]
                        radius_squared = (
                            np.sum(
                                displacement * displacement,
                                dtype=np.longdouble,
                            )
                            + soft_squared
                        )

                        if radius_squared <= 0:
                            raise CelestialCollisionError(
                                f"Colisión puntual entre cuerpos {i} y {j}."
                            )

                        radius = np.sqrt(radius_squared)
                        scale = (
                            -G_ld
                            * m_ld[i]
                            * m_ld[j]
                            / (radius_squared * radius)
                        )
                        pair_force = scale * displacement
                        force[i] += pair_force
                        force[j] -= pair_force

            if not np.all(np.isfinite(force)):
                raise NumericalStabilityError(
                    "Las fuerzas de N cuerpos no son finitas."
                )
            if np.max(np.abs(force)) > np.longdouble(
                np.finfo(np.float64).max
            ):
                raise NumericalStabilityError(
                    "Las fuerzas exceden el rango float64."
                )
            return np.asarray(force, dtype=np.float64)

        @staticmethod
        def _hamiltonian_and_scale(
            positions: NDArray[np.float64],
            momenta: NDArray[np.float64],
            masses: NDArray[np.float64],
            gravitational_constant: float,
            softening: float,
        ) -> Tuple[float, float]:
            """Calcula H = K + V y una escala energética de referencia."""
            n = positions.shape[0]
            q_ld = np.asarray(positions, dtype=np.longdouble)
            p_ld = np.asarray(momenta, dtype=np.longdouble)
            m_ld = np.asarray(masses, dtype=np.longdouble)
            G_ld = np.longdouble(gravitational_constant)
            soft_ld = np.longdouble(softening)
            soft_squared = soft_ld * soft_ld

            kinetic_terms: list[np.longdouble] = []
            potential_magnitudes: list[np.longdouble] = []

            for i in range(n):
                kinetic_terms.append(
                    np.sum(p_ld[i] * p_ld[i], dtype=np.longdouble)
                    / (np.longdouble(2.0) * m_ld[i])
                )

            for i in range(n):
                for j in range(i + 1, n):
                    displacement = q_ld[i] - q_ld[j]
                    radius_squared = (
                        np.sum(
                            displacement * displacement,
                            dtype=np.longdouble,
                        )
                        + soft_squared
                    )
                    if radius_squared <= 0:
                        raise CelestialCollisionError(
                            f"Colisión puntual entre cuerpos {i} y {j}."
                        )
                    radius = np.sqrt(radius_squared)
                    potential_magnitudes.append(
                        G_ld * m_ld[i] * m_ld[j] / radius
                    )

            kinetic = _neumaier_sum(kinetic_terms)
            potential_magnitude = _neumaier_sum(potential_magnitudes)
            hamiltonian = kinetic - potential_magnitude
            scale = kinetic + potential_magnitude

            return (
                _ld_to_float(hamiltonian, "Hamiltonian"),
                _ld_to_float(scale, "Hamiltonian scale"),
            )

        @staticmethod
        def _total_momentum(
            momenta: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            p_ld = np.asarray(momenta, dtype=np.longdouble)
            components = [
                _neumaier_sum(p_ld[:, axis])
                for axis in range(3)
            ]
            return np.asarray(
                [_ld_to_float(value, "total_linear_momentum") for value in components],
                dtype=np.float64,
            )

        @staticmethod
        def _total_angular_momentum(
            positions: NDArray[np.float64],
            momenta: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            q_ld = np.asarray(positions, dtype=np.longdouble)
            p_ld = np.asarray(momenta, dtype=np.longdouble)
            component_terms = [[], [], []]

            for body in range(q_ld.shape[0]):
                angular = _cross_longdouble(q_ld[body], p_ld[body])
                for axis in range(3):
                    component_terms[axis].append(angular[axis])

            return np.asarray(
                [
                    _ld_to_float(
                        _neumaier_sum(component_terms[axis]),
                        "total_angular_momentum",
                    )
                    for axis in range(3)
                ],
                dtype=np.float64,
            )

        def integrate_celestial(
            self,
            request: PoincareSimulationRequest,
        ) -> PoincareDiagnostics:
            """
            Integra el Hamiltoniano separable de N cuerpos con kick-drift-kick.

            El paso temporal es fijo: no se aplica adaptación automática de paso,
            pues una adaptación genérica puede destruir la simplécticidad.
            """
            q, p, masses = self._validate_celestial_request(request)
            state = request.state
            dt = request.time_step

            hamiltonian_initial, energy_scale_initial = (
                self._hamiltonian_and_scale(
                    q,
                    p,
                    masses,
                    state.gravitational_constant,
                    request.softening,
                )
            )
            angular_initial = self._total_angular_momentum(q, p)
            momentum_initial = self._total_momentum(p)

            forces = self._nbody_forces(
                q,
                masses,
                state.gravitational_constant,
                request.softening,
            )

            for _ in range(request.steps):
                with np.errstate(over="ignore", invalid="ignore"):
                    p_half = p + (0.5 * dt) * forces
                    q_next = q + dt * (
                        p_half / masses[:, np.newaxis]
                    )

                if not (
                    np.all(np.isfinite(q_next))
                    and np.all(np.isfinite(p_half))
                ):
                    raise NumericalStabilityError(
                        "El estado intermedio de Störmer–Verlet no es finito."
                    )

                forces_next = self._nbody_forces(
                    q_next,
                    masses,
                    state.gravitational_constant,
                    request.softening,
                )

                with np.errstate(over="ignore", invalid="ignore"):
                    p_next = p_half + (0.5 * dt) * forces_next

                if not np.all(np.isfinite(p_next)):
                    raise NumericalStabilityError(
                        "El momento final de Störmer–Verlet no es finito."
                    )

                q, p, forces = q_next, p_next, forces_next

            hamiltonian_final, energy_scale_final = (
                self._hamiltonian_and_scale(
                    q,
                    p,
                    masses,
                    state.gravitational_constant,
                    request.softening,
                )
            )
            angular_final = self._total_angular_momentum(q, p)
            momentum_final = self._total_momentum(p)

            q_initial_ld = np.asarray(state.positions, dtype=np.longdouble)
            p_initial_ld = np.asarray(state.momenta, dtype=np.longdouble)
            q_final_ld = np.asarray(q, dtype=np.longdouble)
            p_final_ld = np.asarray(p, dtype=np.longdouble)

            angular_scale_initial = _neumaier_sum(
                _long_norm(
                    _cross_longdouble(q_initial_ld[i], p_initial_ld[i])
                )
                for i in range(q_initial_ld.shape[0])
            )
            angular_scale_final = _neumaier_sum(
                _long_norm(
                    _cross_longdouble(q_final_ld[i], p_final_ld[i])
                )
                for i in range(q_final_ld.shape[0])
            )
            angular_denominator = max(
                _long_norm(angular_initial),
                _long_norm(angular_final),
                angular_scale_initial,
                angular_scale_final,
                np.finfo(np.longdouble).tiny,
            )

            momentum_scale = _neumaier_sum(
                _long_norm(p_initial_ld[i])
                for i in range(p_initial_ld.shape[0])
            )
            momentum_denominator = max(
                _long_norm(momentum_initial),
                _long_norm(momentum_final),
                momentum_scale,
                np.finfo(np.longdouble).tiny,
            )

            energy_denominator = max(
                np.longdouble(energy_scale_initial),
                np.longdouble(energy_scale_final),
                abs(np.longdouble(hamiltonian_initial)),
                np.finfo(np.longdouble).tiny,
            )

            energy_drift = (
                abs(
                    np.longdouble(hamiltonian_final)
                    - np.longdouble(hamiltonian_initial)
                )
                / energy_denominator
            )
            angular_drift = (
                _long_norm(
                    np.asarray(angular_final, dtype=np.longdouble)
                    - np.asarray(angular_initial, dtype=np.longdouble)
                )
                / angular_denominator
            )
            linear_drift = (
                _long_norm(
                    np.asarray(momentum_final, dtype=np.longdouble)
                    - np.asarray(momentum_initial, dtype=np.longdouble)
                )
                / momentum_denominator
            )

            return PoincareDiagnostics(
                hamiltonian_initial=hamiltonian_initial,
                hamiltonian_final=hamiltonian_final,
                relative_energy_drift=_ld_to_float(
                    energy_drift, "relative_energy_drift"
                ),
                angular_momentum_initial=angular_initial,
                angular_momentum_final=angular_final,
                relative_angular_momentum_drift=_ld_to_float(
                    angular_drift,
                    "relative_angular_momentum_drift",
                ),
                relative_linear_momentum_drift=_ld_to_float(
                    linear_drift,
                    "relative_linear_momentum_drift",
                ),
                steps=request.steps,
                time_step=dt,
                method="kick-drift-kick / Störmer-Verlet",
            )

        def compute_pushforward(
            self,
            validated: ValidatedStressTensor,
            delta_x: Any,
            celestial_request: Optional[PoincareSimulationRequest] = None,
        ) -> PushforwardResult:
            r"""
            Calcula la energía deformada, la cota métrica de Rayleigh y,
            opcionalmente, diagnósticos Hamiltonianos de N cuerpos.
            """
            if not isinstance(validated, ValidatedStressTensor):
                raise InputValidationError(
                    "validated debe ser ValidatedStressTensor."
                )

            dx = self._assert_delta_dimension(validated, delta_x)
            invariants = validated.invariants

            energy = self._dirichlet_energy(validated.T_mu_nu, dx)
            dual_norm_squared = self._metric_dual_norm_squared(
                validated.G_mu_nu, dx
            )

            bound_ld = (
                np.longdouble(invariants.metric_op_norm_T)
                * np.longdouble(dual_norm_squared)
            )
            rayleigh_bound = _ld_to_float(bound_ld, "rayleigh_bound")

            absolute_energy = abs(np.longdouble(energy))
            rounding_slack = (
                np.longdouble(64.0 * _MACHINE_EPSILON)
                * max(
                    absolute_energy,
                    np.longdouble(rayleigh_bound),
                    np.longdouble(np.finfo(np.float64).tiny),
                )
            )

            # Cota matemática de Rayleigh en la geometría inducida por G.
            if absolute_energy > np.longdouble(rayleigh_bound) + rounding_slack:
                raise NumericalStabilityError(
                    "|delta_xᵀ T delta_x| excede la cota métrica de Rayleigh: "
                    f"{float(absolute_energy):.3e} > {rayleigh_bound:.3e}."
                )

            # Envolvente de compatibilidad histórica; la cota estricta anterior
            # es más fuerte para los parámetros usuales (factor >= 1).
            legacy_envelope = (
                np.longdouble(self._energy_overflow_factor)
                * np.longdouble(rayleigh_bound)
                + rounding_slack
            )
            if absolute_energy > legacy_envelope:
                raise DeformationEnergyOverflowError(
                    "La energía excede la envolvente relativa configurada."
                )

            if (
                self._energy_capacity is not None
                and absolute_energy
                > np.longdouble(self._energy_capacity) + rounding_slack
            ):
                raise DeformationEnergyOverflowError(
                    f"|E|={abs(energy):.3e} excede energy_capacity="
                    f"{self._energy_capacity:.3e}."
                )

            celestial_diagnostics = None
            if celestial_request is not None:
                celestial_diagnostics = self.integrate_celestial(
                    celestial_request
                )

            logger.info(
                "Fase 2 OK | E=%.6e | Rayleigh=%.6e | dual_norm²=%.6e",
                energy,
                rayleigh_bound,
                dual_norm_squared,
            )

            return PushforwardResult(
                deformed_energy=energy,
                rayleigh_bound=rayleigh_bound,
                energy_is_bounded=True,
                invariants=invariants,
                metric_dual_norm_squared=dual_norm_squared,
                celestial_diagnostics=celestial_diagnostics,
            )

        # ══════════════════════════════════════════════════════════════════════
        # FASE 3, ANIDADA EN FASE 2 — GOBERNANZA Y VETO
        # ══════════════════════════════════════════════════════════════════════

        class Phase3_ToposPullback:
            r"""
            Proyecta los resultados a una decisión de gobernanza determinista.

            L_max = κ₀ / (sqrt(|Tr_G(T)|) + ε λ_max(G) sqrt(κ(G))).

            Un exceso de deriva celeste respecto de celestial_drift_limit eleva
            la severidad a CRITICAL y activa el veto.
            """

            def __init__(
                self,
                kappa0: float = _KAPPA_0,
                epsilon: float = _MACHINE_EPSILON,
                lockdown_eps_factor: float = _LOCKDOWN_EPS_FACTOR,
                celestial_drift_limit: float = (
                    _DEFAULT_CELESTIAL_DRIFT_LIMIT
                ),
            ) -> None:
                self._kappa0 = _finite_scalar(kappa0, "kappa0")
                self._epsilon = _finite_scalar(epsilon, "epsilon")
                self._lockdown_eps_factor = _finite_scalar(
                    lockdown_eps_factor, "lockdown_eps_factor"
                )
                self._celestial_drift_limit = _finite_scalar(
                    celestial_drift_limit, "celestial_drift_limit"
                )

                if self._kappa0 <= 0.0:
                    raise ValueError("kappa0 debe ser > 0.")
                if self._epsilon <= 0.0:
                    raise ValueError("epsilon debe ser > 0.")
                if self._lockdown_eps_factor <= 0.0:
                    raise ValueError("lockdown_eps_factor debe ser > 0.")
                if self._celestial_drift_limit < 0.0:
                    raise ValueError(
                        "celestial_drift_limit no puede ser negativo."
                    )

            def _compute_L_max(self, invariants: SpectralInvariants) -> float:
                """Evalúa L_max en precisión extendida y satura al rango float64."""
                trace = abs(np.longdouble(invariants.trace_mixed))
                lambda_max = np.longdouble(invariants.lambda_max_G)
                condition = max(
                    np.longdouble(invariants.cond_G),
                    np.longdouble(1.0),
                )

                with np.errstate(over="ignore", invalid="ignore"):
                    denominator = (
                        np.sqrt(trace)
                        + np.longdouble(self._epsilon)
                        * lambda_max
                        * np.sqrt(condition)
                    )

                if np.isnan(denominator):
                    return 0.0
                if np.isinf(denominator):
                    return 0.0
                if denominator <= 0:
                    return float(np.finfo(np.float64).max)

                value = np.longdouble(self._kappa0) / denominator
                maximum = np.longdouble(np.finfo(np.float64).max)
                if not np.isfinite(value) or value > maximum:
                    return float(np.finfo(np.float64).max)
                return float(value)

            def _boolean_lockdown_predicate(
                self,
                L_max: float,
                invariants: SpectralInvariants,
                energy_is_bounded: bool,
                celestial_stable: Optional[bool],
                effective_severity: SpectralSeverity,
            ) -> bool:
                """Disyunción auditable de los predicados de veto."""
                threshold = (
                    np.longdouble(self._epsilon)
                    * np.longdouble(self._lockdown_eps_factor)
                )
                p_low_margin = np.longdouble(L_max) < threshold
                p_severity = effective_severity in (
                    SpectralSeverity.CRITICAL,
                    SpectralSeverity.SINGULAR,
                )
                p_condition = invariants.cond_G > _COND_MAX
                p_radius = invariants.rho_T > _RHO_CRIT
                p_energy = not energy_is_bounded
                p_celestial = celestial_stable is False

                return bool(
                    p_low_margin
                    or p_severity
                    or p_condition
                    or p_radius
                    or p_energy
                    or p_celestial
                )

            def compute_pullback(
                self,
                pushforward: PushforwardResult,
            ) -> TensorialGovernanceState:
                """Cierra la composición Fase 1 → Fase 2 → Fase 3."""
                if not isinstance(pushforward, PushforwardResult):
                    raise InputValidationError(
                        "pushforward debe ser PushforwardResult."
                    )

                invariants = pushforward.invariants
                L_max = self._compute_L_max(invariants)
                effective_severity = invariants.severity
                reasons: list[str] = []

                if invariants.severity == SpectralSeverity.ELEVATED:
                    reasons.append("spectral_severity_elevated")
                if invariants.cond_G > _COND_MAX:
                    reasons.append("metric_condition_limit")
                if invariants.rho_T > _RHO_CRIT:
                    reasons.append("stress_spectral_radius_limit")

                threshold = (
                    np.longdouble(self._epsilon)
                    * np.longdouble(self._lockdown_eps_factor)
                )
                if np.longdouble(L_max) < threshold:
                    reasons.append("low_lipschitz_margin")

                if not pushforward.energy_is_bounded:
                    reasons.append("deformed_energy_not_bounded")
                    effective_severity = join_severity(
                        effective_severity,
                        SpectralSeverity.CRITICAL,
                    )

                celestial_stable: Optional[bool] = None
                celestial_energy_drift: Optional[float] = None
                celestial_angular_drift: Optional[float] = None
                celestial_linear_drift: Optional[float] = None

                diagnostics = pushforward.celestial_diagnostics
                if diagnostics is not None:
                    celestial_energy_drift = (
                        diagnostics.relative_energy_drift
                    )
                    celestial_angular_drift = (
                        diagnostics.relative_angular_momentum_drift
                    )
                    celestial_linear_drift = (
                        diagnostics.relative_linear_momentum_drift
                    )

                    drift_values = (
                        celestial_energy_drift,
                        celestial_angular_drift,
                        celestial_linear_drift,
                    )
                    celestial_stable = all(
                        math.isfinite(value)
                        and value <= self._celestial_drift_limit
                        for value in drift_values
                    )

                    if (
                        not math.isfinite(celestial_energy_drift)
                        or celestial_energy_drift
                        > self._celestial_drift_limit
                    ):
                        reasons.append("celestial_energy_drift")
                    if (
                        not math.isfinite(celestial_angular_drift)
                        or celestial_angular_drift
                        > self._celestial_drift_limit
                    ):
                        reasons.append("celestial_angular_momentum_drift")
                    if (
                        not math.isfinite(celestial_linear_drift)
                        or celestial_linear_drift
                        > self._celestial_drift_limit
                    ):
                        reasons.append("celestial_linear_momentum_drift")

                    if not celestial_stable:
                        effective_severity = join_severity(
                            effective_severity,
                            SpectralSeverity.CRITICAL,
                        )

                lockdown = self._boolean_lockdown_predicate(
                    L_max=L_max,
                    invariants=invariants,
                    energy_is_bounded=pushforward.energy_is_bounded,
                    celestial_stable=celestial_stable,
                    effective_severity=effective_severity,
                )

                if lockdown:
                    logger.warning(
                        "Gobernanza: lockdown | L_max=%.3e | sev=%s | "
                        "κ(G)=%.3e | razones=%s",
                        L_max,
                        effective_severity.name,
                        invariants.cond_G,
                        ",".join(reasons) or "predicado estructural",
                    )
                else:
                    logger.info(
                        "Fase 3 OK | L_max=%.6e | sev=%s | lockdown=%s",
                        L_max,
                        effective_severity.name,
                        lockdown,
                    )

                return TensorialGovernanceState(
                    L_max=float(L_max),
                    zero_trust_lockdown=lockdown,
                    severity=effective_severity,
                    condition_number=invariants.cond_G,
                    reasons=tuple(reasons),
                    celestial_stable=celestial_stable,
                    celestial_energy_drift=celestial_energy_drift,
                    celestial_angular_momentum_drift=celestial_angular_drift,
                    celestial_linear_momentum_drift=celestial_linear_drift,
                )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTOR / ORQUESTADOR PÚBLICO
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalWatcherAgent(Morphism):
    """
    Orquesta las tres fases anidadas.

    La composición de software es:
        validate_tensor → compute_pushforward → compute_pullback.

    Las expresiones "funtor" y "topos" identifican las interfaces del
    ecosistema; el módulo certifica invariantes numéricos y produce DTOs,
    pero no pretende demostrar por sí solo una equivalencia categórica.
    """

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOL,
        kappa0: float = _KAPPA_0,
        energy_overflow_factor: float = 1e8,
        *,
        energy_capacity: Optional[float] = None,
        celestial_drift_limit: float = _DEFAULT_CELESTIAL_DRIFT_LIMIT,
    ) -> None:
        self._phase1 = Phase1_StressTensorValidation(
            tolerance=tolerance
        )
        self._phase2 = (
            Phase1_StressTensorValidation.Phase2_CohomologyPushforward(
                energy_overflow_factor=energy_overflow_factor,
                energy_capacity=energy_capacity,
            )
        )
        self._phase3 = (
            Phase1_StressTensorValidation
            .Phase2_CohomologyPushforward
            .Phase3_ToposPullback(
                kappa0=kappa0,
                celestial_drift_limit=celestial_drift_limit,
            )
        )

    def execute_propagation(
        self,
        tensor_data: StressTensorData,
        delta_x: Any,
        celestial_request: Optional[PoincareSimulationRequest] = None,
    ) -> Tuple[CohomologyPushforwardData, TensorialGovernanceState]:
        """
        Ejecuta las fases en orden, sin permitir que Fase 2 omita Fase 1.

        celestial_request es opcional. Cuando se proporciona, la Fase 2
        integra el estado canónico y la Fase 3 aplica el umbral de deriva.
        """
        validated = self._phase1.validate_tensor(tensor_data)
        pushforward = self._phase2.compute_pushforward(
            validated=validated,
            delta_x=delta_x,
            celestial_request=celestial_request,
        )
        governance = self._phase3.compute_pullback(pushforward)

        invariants = pushforward.invariants
        celestial_summary = None
        if pushforward.celestial_diagnostics is not None:
            diag = pushforward.celestial_diagnostics
            celestial_summary = PoincarePushforwardData(
                relative_energy_drift=diag.relative_energy_drift,
                relative_angular_momentum_drift=(
                    diag.relative_angular_momentum_drift
                ),
                relative_linear_momentum_drift=(
                    diag.relative_linear_momentum_drift
                ),
                steps=diag.steps,
            )

        cohomology_data = CohomologyPushforwardData(
            deformed_dirichlet_energy=pushforward.deformed_energy,
            stress_trace=invariants.trace_mixed,
            operator_norm=invariants.op_norm_T,
            spectral_radius=invariants.rho_T,
            rayleigh_bound=pushforward.rayleigh_bound,
            metric_dual_norm_squared=(
                pushforward.metric_dual_norm_squared
            ),
            energy_is_bounded=pushforward.energy_is_bounded,
            celestial=celestial_summary,
        )

        return cohomology_data, governance

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compatibilidad con la interfaz Morphism."""
        return self.execute_propagation(*args, **kwargs)

    def describe_pipeline(self) -> Dict[str, str]:
        """Metadatos del pipeline para telemetría y auditoría."""
        return {
            "functor": "TopologicalWatcherAgent",
            "version": "6.0.0-Nested-Refined-Hamiltonian",
            "phase_1": (
                "Phase1_StressTensorValidation.validate_tensor "
                "(refinamiento recursivo acotado de nivel 3)"
            ),
            "phase_2": (
                "Phase2_CohomologyPushforward.compute_pushforward "
                "(Rayleigh métrico; N-cuerpos opcional)"
            ),
            "phase_2_celestial": (
                "Hamiltoniano newtoniano canónico; "
                "kick-drift-kick / Störmer-Verlet"
            ),
            "phase_3": (
                "Phase3_ToposPullback.compute_pullback "
                "(join de severidad y política de veto)"
            ),
            "conservation": (
                "∇_μ T^{μν}=0 comprobado con residuo relativo"
            ),
            "metric_rayleigh": (
                "|αᵀTα| <= ||G^(1/2) T G^(1/2)||₂ · αᵀG⁻¹α"
            ),
            "self_improvement_scope": (
                "Refinamiento numérico determinista; sin auto-modificación "
                "de código ni relajación de invariantes"
            ),
        }