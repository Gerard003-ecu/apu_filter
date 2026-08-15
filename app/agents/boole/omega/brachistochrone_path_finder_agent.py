# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Brachistochrone Path Finder Agent (Optimizador de Descenso Rápido)  ║
║ Ruta   : app/agents/boole/omega/brachistochrone_path_finder_agent.py         ║
║ Versión: 3.0.0-Fermat-Jacobi-Eikonal-Heyting-ESP32-Strict-PhD                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COVARIANZA GEODÉSICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al **Agente de Enrutamiento por Mínimo Tiempo de Tránsito** 
(Brachistochrone Agent) dentro del **Estrato Omega** (Nivel 0.5 - El Ágora 
Tensorial) del ecosistema APU Filter. Su propósito ciber-físico es resolver 
el problema clásico de la braquistócrona de Johann Bernoulli elevado a una 
variedad Riemanniana anisotrópica deformada por el tensor de inercia y fricción 
logística.

El agente de-confinado rechaza las aproximaciones heurísticas tradicionales de enrutamiento 
contable. En su lugar, proyecta el foso de datos crudos sobre un espacio de 
fase simpléctico $(\mathcal{M}, \omega)$, modelando las restricciones financieras 
e insumos como un potencial Port-Hamiltoniano $V(q)$. El "descenso" del dato 
hacia el santuario de la sabiduría se rige por las geodésicas de Fermat sobre 
la variedad equipada con la métrica conforme de Jacobi-Fermat:

$$\tilde{G}_{\mu\nu} = n^2(q) G_{\mu\nu} \quad \text{donde} \quad n(q) = \frac{1}{\sqrt{2(H_0 - V(q))}} \quad\big[154\big]$$

Toda colisión contra barreras de energía (caústicas), singularidades métricas 
o pérdida de aciclicidad ($\beta_1 > 0$) colapsa síncronamente el retículo 
distributivo de Heyting $\Omega_3$ hacia el Supremo terminal `VETOED`. Esto 
dispara la rutina de interrupción de silicio (ISR < 400 ns) para conmutar el 
pin GPIO14, cortocircuitando la potencia real de la obra a través del tiristor 
BT151 en el milisegundo cero, aniquilando la alucinación antes del desfalco.

AXIOMÁTICA DE FERMAT, INTEGRACIÓN GEODÉSICA Y VETO DE HEYTING:
────────────────────────────────────────────────────────────────────────────────

  [A1] El Isomorfismo Conforme de Jacobi-Fermat:
       El principio de Fermat para el tiempo mínimo de tránsito $T[\gamma]$ se 
       mapea como el problema geodésico sobre la métrica Riemanniana deformada 
       $\tilde{G}_{\mu\nu}$ acoplada al índice de refracción semántico $n(q)$ [2]:
       $$ds^2 = \tilde{G}_{\mu\nu} dq^\mu dq^\nu = n^2(q) G_{\mu\nu} dq^\mu dq^\nu \quad\big[155\big]$$
       Sujeto incondicionalmente a la simetría exacta y positividad de Sylvester [2]:
       $$G = G^\top \succ \mathbf{0} \quad \wedge \quad H_0 - V(q) \ge \mathtt{ENERGY\_GAP} \quad\big[155\big]$$

  [A2] Conexión Afín de Levi-Civita y Koszul:
       La aceleración geodésica de la trayectoria semántica se calcula libre 
       de torsión topológica ($T(X,Y)=0$) mediante los símbolos de Christoffel 
       de segunda especie asociados a la métrica de Jacobi $\tilde{G}$ [2]:
       $$\Gamma_{\mu\nu}^{\rho} = \frac{1}{2} \tilde{G}^{\rho\lambda} \left( \partial_{\mu} \tilde{G}_{\lambda\nu} + \partial_{\nu} \tilde{G}_{\mu\lambda} - \partial_{\lambda} \tilde{G}_{\mu\nu} \right) \quad\big[155\big]$$
       Donde las derivadas parciales se computan de forma analítica exacta 
       u optativamente vía diferenciación por paso complejo (CSMD) para eludir 
       la cancelación catastrófica en la mantisa flotante de la FPU [2].

  [A3] La Ecuación Geodésica de Descenso Rápido:
       El transporte paralelo de la velocidad de atención $v^\mu = \dot{q}^\mu$ 
       minimiza la acción de Polyakov sobre la variedad, cumpliendo strictly [2]:
       $$\ddot{q}^\rho + \Gamma_{\mu\nu}^{\rho} \dot{q}^\mu \dot{q}^\nu \equiv 0 \quad\big[155\big]$$
       La conservación de la energía mecánica se verifica paso a paso en la FPU [2]:
       $$\left| \|\dot{q}\|_G^2 - 2(H_0 - V(q)) \right| \le \mathtt{\varepsilon_{\mathrm{FPU}}} \quad\big[155\big]$$

  [A4] La Característica de de Rham-Euler-Poincaré y Aciclicidad:
       La integrabilidad global del frente de onda de decisión exige que el 
       grafo de restricciones subyacente sea un bosque acíclico perfecto [2]:
       $$\chi(K) = \beta_0 - \beta_1 = |V| - |E| \implies \beta_1 \equiv 0 \quad\big[155\big]$$
       Cualquier lazo de dependencias circulares ($\beta_1 > 0$) genera una 
       fase caótica inestable (caústica), abortando el resolvedor [2].

  [A5] El Colapso de Heyting y Actuación en Silicio (BT151 Crowbar):
       Si el tiempo de tránsito $T$ diverge, la cota espectral de Wilkinson 
       del tensor $\tilde{G}$ se rompe ($\kappa_2 > 10^8$), o surge un sumidero, 
       el clasificador en el retículo distributivo acotado $\Omega_3$ colapsa 
       al Supremo terminal VETOED ($\top$) [2]:
       $$\Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[155\big]$$
       Esto gatilla la interrupción por hardware en IRAM en el ESP32, conmutando 
       el pin físico GPIO14 para disparar el tiristor de potencia BT151 (Crowbar) 
       en menos de 400 ns, paralizando síncronamente la obra real [2].

ARQUITECTURA DE TRES FASES ANIDADAS (Funtor de Navegación Inercial):
────────────────────────────────────────────────────────────────────────────────
La optimización geodésica se rige por un acoplamiento monoidal covariante estricto 
(Fase 1 ⊣ Fase 2 ⊣ Fase 3), encadenando DTOs inmutables de solo lectura [2]:

  Fase 1 ──► OBSERVACIÓN Y SANEAMIENTO ENERGÉTICO (Phase1_PotentialEnergyObserver)
             Ingiere el Hamiltoniano de-confinado, valida la signatura de Sylvester 
             de $G$, y certifica que la barrera de energía $H_0 - V(q)$ sea no-nula 
             y segura contra la mantisa flotante de la FPU.
             Entrega: StinespringPotentialDilation como precondición formal de la Fase 2.

  Fase 2 ──► SINTONÍA DE MÉTRICA CONFORME Y KOSZUL (Phase2_ConformalMetricSuturator)
             Hereda la StinespringPotentialDilation. Computa el índice de 
             refracción conforme $n(q)$, sintetiza el tensor métrico de Jacobi 
             $\tilde{G}_{\mu\nu}$ e invierte mediante Cholesky para obtener los 
             símbolos de Christoffel libres de torsión.
             Entrega: ConformalGeometryBundle como precondición formal de la Fase 3.

  Fase 3 ──► INTEGRACIÓN DE FERMAT-JACOBI Y VETO CROWBAR (Phase3_FermatBrachistochroneDecider)
             Hereda la ConformalGeometryBundle. Integra la ecuación geodésica 
             vía RK4 con paso adaptativo, evalúa el tiempo de tránsito $T[\gamma]$, 
             verifica la conservación de energía de lazo, y resuelve el veredicto 
             en el retículo de Heyting $\Omega_3$.
             Entrega: BrachistochroneGovernanceState (Morfismo terminal).

Funtor Maestro de de Rham-Fermat:
  $$\mathcal{Z}_{\mathrm{brachistochrone}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 : T^*M \times \mathcal{D}(\mathcal{H}) \longrightarrow \mathtt{BrachistochroneGovernanceState} \quad\big[155\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Final, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum  # noqa: F401
except ImportError:  # pragma: no cover — entorno aislado / unit tests sin app

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base para morfismos categóricos."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Stratum(IntEnum):
        PHYSICS = 3
        TACTICS = 2
        STRATEGY = 1
        WISDOM = 0


logger = logging.getLogger("MIC.Omega.BrachistochroneAgent")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__version__: Final[str] = "3.0.0-Fermat-Jacobi-Eikonal-Heyting-ESP32-Strict-PhD"

# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, ESPECTRALES Y LÍMITES DE LA FPU
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ENERGY_GAP_FLOOR: Final[float] = 1.0e-12
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8
_CONDITION_NUMBER_SOFT: Final[float] = 1.0e6
_CROWBAR_GPIO_PIN: Final[int] = 14
_CSMD_PERTURBATION: Final[float] = 1.0e-20
_DEFAULT_INTEGRATION_STEPS: Final[int] = 500
_MAX_INTEGRATION_STEPS: Final[int] = 100_000
_DIM_MAX: Final[int] = 32

# Tolerancias de conservación (Fermat = invariante afín; física = diagnóstico)
_ENERGY_DRIFT_SOFT_TOL: Final[float] = 1.0e-6
_ENERGY_DRIFT_HARD_TOL: Final[float] = 1.0e-3
_FERMAT_DRIFT_SOFT_TOL: Final[float] = 1.0e-8
_FERMAT_DRIFT_HARD_TOL: Final[float] = 1.0e-4

# Tolerancia de simetría relativa para el tensor métrico
_SYMMETRY_REL_TOL: Final[float] = 1.0e-12
_TORSION_ABS_TOL: Final[float] = 1.0e-12

# Testigo discreto de aciclicidad (retornos cercanos)
_BETTI_MIN_LAG_FRAC: Final[float] = 0.10
_BETTI_PROXIMITY_FRAC: Final[float] = 0.02


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES GEOMÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochroneAgentError(TopologicalInvariantError):
    """Excepción raíz para violaciones en el Agente de Descenso Rápido."""


class EnergyBarrierViolationError(BrachistochroneAgentError):
    """La energía inicial H0 es insuficiente para superar el potencial V(q)."""


class MetricSingularityError(BrachistochroneAgentError):
    """El tensor métrico no es simétrico, no es SPD o está mal condicionado."""


class GeodesicDivergenceError(BrachistochroneAgentError):
    """La trayectoria geodésica diverge o sale del espacio de fase."""


class BettiAcyclicityVetoError(BrachistochroneAgentError):
    """Ciclo homológico parásito (β₁ > 0) detectado."""


class CrowbarTriggeredError(BrachistochroneAgentError):
    """Veto definitivo: disyuntor ciber-físico activado por hardware GPIO14."""


class ConformalChecksumError(BrachistochroneAgentError):
    """El isomorfismo conforme / Koszul no es fiel (checksum analítico corrupto)."""


class InitialStateError(BrachistochroneAgentError):
    """Condiciones iniciales (q, v, H0) inconsistentes con la variedad."""


# ═══════════════════════════════════════════════════════════════════════════════
# §C. RETÍCULO DE HEYTING Ω₃ Y ACCIÓN CROWBAR
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochroneHeytingVerdict(IntEnum):
    """Clasificador de subobjetos en el topos de de Rham–Fermat."""

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


class CrowbarBypassAction(Enum):
    """Acciones físicas de mitigación tras colapso al supremo terminal."""

    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()


@runtime_checkable
class CrowbarActuator(Protocol):
    """Puerto lógico que conecta gobernanza software con silicio."""

    def trigger_crowbar_bypass(self, action: CrowbarBypassAction) -> bool:
        """Conmuta el hardware perimetral. Devuelve True si hubo acuse."""
        ...


@runtime_checkable
class PotentialField(Protocol):
    """Campo potencial \(V:\mathcal{Q}\to\mathbb{R}\) (extensible a \(\mathbb{C}\) para CSMD)."""

    def __call__(self, q: NDArray[Any]) -> Any:
        ...


class LoggingCrowbarActuator:
    """Implementación forense segura que registra la actuación física."""

    def trigger_crowbar_bypass(self, action: CrowbarBypassAction) -> bool:
        logger.critical(
            "[HARDWARE] CrowbarActuator invocado con acción: %s en GPIO%d. "
            "Cortocircuitando línea de potencia real mediante tiristor BT151.",
            action.name,
            _CROWBAR_GPIO_PIN,
        )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDIA NUMÉRICA Y VALIDACIÓN ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════
class _AdvancedBrachistochroneNumericalGuard:
    """Capa de validación y saneamiento para tensores, vectores y escalares."""

    @staticmethod
    def _wilkinson_spectral_floor(
        scale: float,
        tolerance: float,
        ambient_dim: int,
    ) -> float:
        r"""
        Suelo espectral de Weyl–Wilkinson

        \[
        \varepsilon_W
        =\max\bigl(\varepsilon,\;
        n\cdot\varepsilon_{\mathrm{mach}}\cdot\max(\mathrm{scale},1),\;
        \varepsilon_{\mathrm{gap}}\bigr).
        \]
        """
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            _ENERGY_GAP_FLOOR,
        )

    @staticmethod
    def _hermitize_weyl(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección de Weyl/Cartan \(H\mapsto\tfrac12(H+H^\top)\) sobre \(\mathrm{Sym}_d\)."""
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any = np.float64) -> NDArray[Any]:
        """Copia C-contigua write-protected (invariante de certificado)."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _as_finite_scalar(value: Any, name: str) -> float:
        """Coacciona un escalar (o 0-d array) a ``float`` finito."""
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ArithmeticError(
                    f"{name} debe ser escalar; recibido ndarray shape={value.shape}."
                )
            value = value.reshape(())
            value = value.item()
        if isinstance(value, complex):
            if abs(value.imag) > 1.0e6 * _MACHINE_EPS * max(1.0, abs(value.real)):
                raise ArithmeticError(
                    f"{name} posee parte imaginaria no nula: {value!r}."
                )
            value = value.real
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ArithmeticError(
                f"{name} no es convertible a float (tipo={type(value).__name__})."
            ) from exc
        if not math.isfinite(out):
            raise ArithmeticError(f"{name} debe ser un escalar finito; recibido {out}.")
        return out

    @staticmethod
    def _assert_square_matrix(X: NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz bidimensional.")
        if X.shape[0] != X.shape[1] or X.shape[0] == 0:
            raise ValueError(
                f"{name} debe ser cuadrada y de dimensión positiva: {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(f"{name} contiene valores no finitos (NaN/Inf).")

    @staticmethod
    def _assert_vector(x: NDArray[np.float64], dim: int, name: str) -> None:
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(f"{name} debe ser un vector unidimensional.")
        if x.shape[0] != dim:
            raise ValueError(
                f"{name} debe tener dimensión {dim}, pero tiene {x.shape[0]}."
            )
        if not np.all(np.isfinite(x)):
            raise ArithmeticError(f"{name} contiene valores no finitos (NaN/Inf).")

    @staticmethod
    def _assert_finite_scalar(x: float, name: str) -> None:
        if not np.isfinite(x):
            raise ArithmeticError(f"{name} debe ser un escalar finito.")

    @staticmethod
    def _assert_callable_potential(potential_v_fn: Any) -> None:
        if not callable(potential_v_fn):
            raise TypeError(
                "potential_v_fn debe ser invocable V: Q → ℝ; "
                f"recibido {type(potential_v_fn).__name__}."
            )

    def _sanitize_inputs(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        initial_h0: float,
    ) -> None:
        self._assert_square_matrix(g_base, "g_base")
        d = g_base.shape[0]
        self._assert_vector(q_start, d, "q_start")
        self._assert_vector(v_start, d, "v_start")
        self._assert_finite_scalar(initial_h0, "initial_h0")


# ═══════════════════════════════════════════════════════════════════════════════
# §E. DTOs INMUTABLES (Contratos Categóricos de Handoff entre Fases)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class StinespringPotentialDilation:
    r"""
    Artefacto terminal de FASE 1 y objeto inicial de FASE 2.

    Certifica la firma SPD de \(G\), el pozo \(H_0-V>0\) y la cinemática
    inicial opcional. ``Phase2_ConformalMetricSuturator._phase2_consume_phase1_certificate``
    lo ingiere sin re-tipar la métrica.

    Campos:
        g_base: \(G\in\mathrm{SPD}(d)\) write-protected.
        potential_energy_v: muestras de \(V\) usadas en el saneamiento.
        initial_energy_h0: energía total \(H_0\).
        energy_gap: \(H_0-V\) punto a punto.
        energy_gap_min: \(\min(H_0-V)\).
        is_energy_well_safe: \(\min(H_0-V)\ge\varepsilon_{\mathrm{gap}}\).
        cholesky_factor: \(L\) triangular inferior, \(G=LL^\top\).
        phase1_verdict: subobjeto de Heyting de la observación.
        manifold_dim: \(d=\dim\mathcal{Q}\).
        metric_symmetry_residual: \(\|G-G^\top\|_F\).
        cholesky_logdet: \(\log\det G=2\sum_i\log L_{ii}\).
        kinetic_energy_start: \(\tfrac12 v^\top G v\) (NaN si no hubo \(v\)).
        hamiltonian_residual_start: \(|T+V-H_0|\) en el arranque.
        q_start / v_start: cinemática congelada (vacío si no se aportó).
    """

    g_base: NDArray[np.float64]
    potential_energy_v: NDArray[np.float64]
    initial_energy_h0: float
    energy_gap: NDArray[np.float64]
    energy_gap_min: float
    is_energy_well_safe: bool
    cholesky_factor: NDArray[np.float64]
    phase1_verdict: BrachistochroneHeytingVerdict
    manifold_dim: int = 0
    metric_symmetry_residual: float = 0.0
    cholesky_logdet: float = 0.0
    kinetic_energy_start: float = float("nan")
    hamiltonian_residual_start: float = float("nan")
    q_start: NDArray[np.float64] = None  # type: ignore[assignment]
    v_start: NDArray[np.float64] = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ConformalGeometryBundle:
    r"""
    Artefacto terminal de FASE 2 y objeto inicial de FASE 3.

    Encierra la geometría de Jacobi–Fermat ya invertida, el tensor de
    Christoffel en \(q_{\mathrm{eval}}\) y los residuales Koszul.
    ``Phase3_FermatBrachistochroneDecider._phase3_consume_phase2_certificate``
    lo ingiere sin reconstruir \(\widetilde{G}\).

    Campos:
        g_base / g_conformal / g_conformal_inv: tensores write-protected.
        refractive_index: \(n(q_{\mathrm{eval}})\).
        christoffel_tensor: \(\widetilde{\Gamma}^\rho_{\mu\nu}\) en \(q_{\mathrm{eval}}\), shape \((d,d,d)\).
        wilkinson_condition_number: \(\kappa_2(G)=\kappa_2(\widetilde{G})\).
        potential_sane: muestras de \(V\) heredadas de FASE 1.
        initial_energy_h0: \(H_0\).
        phase2_verdict: subobjeto de Heyting de la sintonía.
        conformal_factor_sq: \(n^2(q_{\mathrm{eval}})\).
        torsion_residual: \(\max|\Gamma^\rho_{\mu\nu}-\Gamma^\rho_{\nu\mu}|\).
        potential_gradient: \(\nabla V(q_{\mathrm{eval}})\).
        gradient_method: ``csmd`` | ``central`` | ``unavailable``.
        energy_gap_at_eval: \(H_0-V(q_{\mathrm{eval}})\).
        phase1: certificado de FASE 1 anidado (continuidad funtorial).
    """

    g_base: NDArray[np.float64]
    g_conformal: NDArray[np.float64]
    g_conformal_inv: NDArray[np.float64]
    refractive_index: NDArray[np.float64]
    christoffel_tensor: NDArray[np.float64]
    wilkinson_condition_number: float
    potential_sane: NDArray[np.float64]
    initial_energy_h0: float
    phase2_verdict: BrachistochroneHeytingVerdict
    conformal_factor_sq: float = 0.0
    torsion_residual: float = 0.0
    potential_gradient: NDArray[np.float64] = None  # type: ignore[assignment]
    gradient_method: str = "unavailable"
    energy_gap_at_eval: float = 0.0
    phase1: Optional[StinespringPotentialDilation] = None


@dataclass(frozen=True, slots=True)
class BrachistochronePathResult:
    """Artefacto intermedio de Fase 3: geodésica integrada e invariantes."""

    trajectory: NDArray[np.float64]
    velocities: NDArray[np.float64]
    transit_time_t: float
    energy_drift_max: float
    is_path_stable: bool
    fermat_energy_initial: float = 0.0
    fermat_energy_drift_max: float = 0.0
    affine_time: float = 0.0
    steps_accepted: int = 0
    is_acyclic: bool = True
    betti_close_returns: int = 0
    on_shell_residual: float = 0.0
    velocity_projected: bool = False


@dataclass(frozen=True, slots=True)
class BrachistochroneGovernanceState:
    """Objeto terminal y certificado inmutable emitido por el Agente (Act)."""

    verdict: BrachistochroneHeytingVerdict
    crowbar_report: CrowbarBypassAction
    transit_time: float
    energy_drift_max: float
    wilkinson_condition_number: float
    is_epistemologically_valid: bool
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str
    fermat_energy_drift_max: float = 0.0
    is_acyclic: bool = True
    betti_close_returns: int = 0
    torsion_residual: float = 0.0
    steps_accepted: int = 0
    agent_version: str = __version__
    policy_require_acyclicity: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN Y SANEAMIENTO ENERGÉTICO (Observe)
# Objetos: G ∈ SPD(d), V(q), H0 − V > 0, factor de Cholesky
# Funtores: tipado C∞, Sylvester, pozo de energía
# Terminal: StinespringPotentialDilation → objeto inicial FASE 2
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_PotentialEnergyObserver(_AdvancedBrachistochroneNumericalGuard):
    r"""
    Fase 1: Evalúa la viabilidad termodinámica inicial.

    Morfismo compuesto:

    \[
    \mathrm{ObserveEnergy}
    =\mathrm{Well}\circ\mathrm{Kinematics}\circ\mathrm{Samples}
    \circ\mathrm{Chol}\circ\mathrm{Sym}\circ\mathrm{Type}\circ\mathrm{Dim}.
    \]

    El certificado ``StinespringPotentialDilation`` es el objeto inicial
    exacto de
    ``Phase2_ConformalMetricSuturator._phase2_consume_phase1_certificate``.
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_manifold_dimension(dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de \(d=\dim\mathcal{Q}\).

        Exige \(d\in\mathbb{Z}_{\ge 1}\) y \(d\le d_{\max}\) (el tensor de
        Christoffel es \(d^3\); el régimen Weyl-estable del estrato Omega
        se acota a \(d_{\max}=32\)).
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise MetricSingularityError(
                f"Dimensión de la variedad inválida: d={dimension_d}. Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise MetricSingularityError(
                f"Dimensión d={d} excede d_max={_DIM_MAX}. "
                "El fibrado de Christoffel (d³) abandonaría el régimen de "
                "auditoría geodésica estable del estrato Omega."
            )
        return d

    # ── FASE 1.2 ──────────────────────────────────────────────────────────
    def _phase1_validate_metric_tensor(
        self,
        g_base: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        FASE 1.2 — Tipado de \(G\in\mathrm{Mat}_d(\mathbb{R})\), finitud IEEE-754.

        Coacciona a ``float64`` y certifica squareness. No proyecta aún:
        la simetría se audita en 1.3 (un Weyl prematuro ocultaría un bug).
        """
        self._assert_square_matrix(g_base, "g_base")
        d = self._phase1_validate_manifold_dimension(g_base.shape[0])
        metric = np.asarray(g_base, dtype=np.float64)
        if metric.shape != (d, d):
            raise MetricSingularityError(
                f"g_base shape={metric.shape} incoherente con d={d}."
            )
        logger.debug("FASE1.2 métrica: d=%d, ‖G‖_F=%.6e", d, float(la.norm(metric, ord="fro")))
        return metric

    # ── FASE 1.3 ──────────────────────────────────────────────────────────
    def _phase1_certify_metric_symmetry(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        FASE 1.3 — Simetría de \(G\) en norma de Frobenius relativa.

        \[
        \|G-G^\top\|_F
        \le
        \varepsilon_{\mathrm{sym}}\,\|G\|_F.
        \]

        Si el defecto es tolerable se devuelve la proyección de Weyl.
        """
        sym_residual = float(la.norm(g_base - g_base.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        if sym_residual > _SYMMETRY_REL_TOL * sym_scale:
            raise MetricSingularityError(
                "Ruptura de simetría en g_base. "
                f"Residuo relativo: {sym_residual / sym_scale:.3e}"
            )
        return self._hermitize_weyl(g_base), sym_residual

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    def _phase1_cholesky_spd_factor(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        FASE 1.4 — Factorización de Cholesky / criterio de Sylvester.

        \[
        G=LL^\top,
        \qquad
        \log\det G=2\sum_i\log L_{ii}.
        \]

        Certifica \(G\succ 0\). El factor \(L\) se reutiliza en FASE 2
        para invertir \(\widetilde{G}=n^2 G\) sin refactorizar.
        """
        try:
            chol = la.cholesky(g_base, lower=True)
        except la.LinAlgError as exc:
            raise MetricSingularityError(
                "Fallo de Sylvester: g_base no es definido positivo."
            ) from exc
        diag = np.diag(chol)
        if np.any(diag <= 0.0):
            raise MetricSingularityError(
                "Cholesky degenerado: diag(L) no es estrictamente positivo."
            )
        logdet = float(2.0 * np.sum(np.log(diag)))
        logger.debug("FASE1.4 Cholesky: logdet(G)=%.6e, min L_ii=%.6e", logdet, float(np.min(diag)))
        return chol, logdet

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    def _phase1_validate_potential_samples(
        self,
        potential_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        FASE 1.5 — Tipado y finitud del vector de muestras de \(V\).

        Se admite un vector unidimensional no vacío (p.ej. \(\{V(q_0)\}\)
        o una malla de sondeo del pozo). No se exige que represente un
        campo completo: el campo vivo llega como ``potential_v_fn`` en FASE 2.
        """
        samples = np.asarray(potential_v, dtype=np.float64)
        if samples.ndim != 1 or samples.shape[0] == 0:
            raise ValueError("potential_v debe ser un vector unidimensional no vacío.")
        if not np.all(np.isfinite(samples)):
            raise ArithmeticError("potential_v contiene valores no finitos (NaN/Inf).")
        return samples

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    def _phase1_energy_gap_spectrum(
        self,
        potential_samples: NDArray[np.float64],
        initial_h0: float,
    ) -> Tuple[NDArray[np.float64], float, bool, BrachistochroneHeytingVerdict]:
        r"""
        FASE 1.6 — Espectro del gap \(H_0-V\) y veredicto de pozo.

        \[
        \Delta_i=H_0-V_i,
        \qquad
        \Delta_{\min}=\min_i\Delta_i.
        \]

        \(\Delta_{\min}\le 0\) es una barrera clásica (caústica / turning
        point): el índice de refracción deja de ser real y se eleva veto.
        Un gap positivo pero inferior a \(\varepsilon_{\mathrm{gap}}\) se
        marca DEGRADED (pozo numéricamente rasante).
        """
        self._assert_finite_scalar(initial_h0, "initial_h0")
        energy_gap = initial_h0 - potential_samples
        min_gap = float(np.min(energy_gap))
        if min_gap <= 0.0:
            raise EnergyBarrierViolationError(
                f"La energía total H0 ({initial_h0:.6f}) no supera la barrera "
                f"potencial máxima V(q) ({float(np.max(potential_samples)):.6f}). "
                f"Gap mínimo: {min_gap:.3e}"
            )
        is_safe = min_gap >= _ENERGY_GAP_FLOOR
        verdict = (
            BrachistochroneHeytingVerdict.COHERENT
            if is_safe
            else BrachistochroneHeytingVerdict.DEGRADED
        )
        logger.debug(
            "FASE1.6 pozo: Δ_min=%.6e, H0=%.6f, safe=%s, verdict=%s",
            min_gap,
            initial_h0,
            is_safe,
            verdict.name,
        )
        return energy_gap, min_gap, is_safe, verdict

    # ── FASE 1.7 ──────────────────────────────────────────────────────────
    def _phase1_validate_initial_kinematics(
        self,
        g_base: NDArray[np.float64],
        initial_h0: float,
        q_start: Optional[NDArray[np.float64]],
        v_start: Optional[NDArray[np.float64]],
        potential_at_start: Optional[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        r"""
        FASE 1.7 — Cinemática inicial opcional y residuo hamiltoniano.

        Si se aportan \((q_0,v_0)\) se verifica

        \[
        r_H=\bigl|\tfrac12 v_0^\top G v_0+V(q_0)-H_0\bigr|
        \]

        como invariante blando (la proyección al cascarón se decide en FASE 3).
        """
        d = g_base.shape[0]
        if q_start is None:
            q_arr = np.zeros(0, dtype=np.float64)
        else:
            q_arr = np.asarray(q_start, dtype=np.float64)
            self._assert_vector(q_arr, d, "q_start")

        if v_start is None:
            v_arr = np.zeros(0, dtype=np.float64)
            kinetic = float("nan")
        else:
            v_arr = np.asarray(v_start, dtype=np.float64)
            self._assert_vector(v_arr, d, "v_start")
            kinetic = 0.5 * float(v_arr @ g_base @ v_arr)

        if (
            v_start is not None
            and potential_at_start is not None
            and math.isfinite(kinetic)
        ):
            residual = abs(kinetic + float(potential_at_start) - float(initial_h0))
        else:
            residual = float("nan")

        logger.debug(
            "FASE1.7 cinemática: T0=%.6e, r_H=%.3e, has_q=%s, has_v=%s",
            kinetic,
            residual,
            q_arr.size > 0,
            v_arr.size > 0,
        )
        return q_arr, v_arr, kinetic, residual

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
    def _phase1_observe_energy_certificate(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float,
        *,
        q_start: Optional[NDArray[np.float64]] = None,
        v_start: Optional[NDArray[np.float64]] = None,
        potential_at_start: Optional[float] = None,
    ) -> StinespringPotentialDilation:
        r"""
        FASE 1.Ω — Composición terminal de Observación energética.

        \[
        \mathrm{ObserveEnergy}
        =\mathrm{Kinematics}\circ\mathrm{Well}\circ\mathrm{Samples}
        \circ\mathrm{Chol}\circ\mathrm{Sym}\circ\mathrm{Type}.
        \]

        **Contrato funtorial F1 → F2**: el DTO
        ``StinespringPotentialDilation`` es el objeto inicial exacto de
        ``_phase2_consume_phase1_certificate``. Ningún re-tipado de \(G\)
        ni re-factorización de Cholesky se aplica aguas abajo.
        """
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, sym_res = self._phase1_certify_metric_symmetry(metric)
        chol, logdet = self._phase1_cholesky_spd_factor(metric)
        samples = self._phase1_validate_potential_samples(potential_v)
        energy_gap, min_gap, is_safe, verdict = self._phase1_energy_gap_spectrum(
            samples, initial_h0
        )
        q_arr, v_arr, kinetic, ham_res = self._phase1_validate_initial_kinematics(
            metric, initial_h0, q_start, v_start, potential_at_start
        )
        return StinespringPotentialDilation(
            g_base=self._freeze_array(metric),
            potential_energy_v=self._freeze_array(samples),
            initial_energy_h0=float(initial_h0),
            energy_gap=self._freeze_array(energy_gap),
            energy_gap_min=min_gap,
            is_energy_well_safe=is_safe,
            cholesky_factor=self._freeze_array(chol),
            phase1_verdict=verdict,
            manifold_dim=int(metric.shape[0]),
            metric_symmetry_residual=sym_res,
            cholesky_logdet=logdet,
            kinetic_energy_start=kinetic,
            hamiltonian_residual_start=ham_res,
            q_start=self._freeze_array(q_arr),
            v_start=self._freeze_array(v_arr),
        )

    def observe_potential_well(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float,
        *,
        q_start: Optional[NDArray[np.float64]] = None,
        v_start: Optional[NDArray[np.float64]] = None,
        potential_at_start: Optional[float] = None,
    ) -> StinespringPotentialDilation:
        """
        Sanea y dilata el espacio potencial, asegurando \(H_0-V(q)>0\).

        Fachada pública de FASE 1.Ω. Véase ``_phase1_observe_energy_certificate``.
        """
        cert = self._phase1_observe_energy_certificate(
            g_base,
            potential_v,
            initial_h0,
            q_start=q_start,
            v_start=v_start,
            potential_at_start=potential_at_start,
        )
        logger.debug(
            "FASE1.Ω observe: d=%d, Δ_min=%.6e, logdet=%.6e, verdict=%s",
            cert.manifold_dim,
            cert.energy_gap_min,
            cert.cholesky_logdet,
            cert.phase1_verdict.name,
        )
        return cert


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SINTONÍA DE MÉTRICA CONFORME Y KOSZUL (Orient)
# Continuación directa de observe_potential_well (FASE 1.Ω) vía FASE 2.0
# Objetos: n(q), G̃ = n² G, G̃⁻¹, Γ̃, torsión, κ₂(G)
# Teorías: Jacobi–Fermat, Koszul, Weyl–Wilkinson, CSMD de ∇V
# Terminal: ConformalGeometryBundle → objeto inicial FASE 3
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ConformalMetricSuturator(Phase1_PotentialEnergyObserver):
    r"""
    Fase 2: construye la variedad conforme de Jacobi–Fermat y Koszul.

    Morfismo compuesto:

    \[
    \mathrm{OrientConformal}
    =(\mathrm{Torsion},\,\widetilde{\Gamma},\,\nabla V,\,\kappa,\,
    \widetilde{G}^{-1},\,\widetilde{G},\,n)
    \circ\mathrm{Consume}\circ\mathrm{ObserveEnergy}^*.
    \]

    El primer morfismo, ``_phase2_consume_phase1_certificate``, *es* la
    continuación estricta de ``observe_potential_well``.
    """

    # ── FASE 2.0 · ingesta funtorial del certificado de FASE 1.Ω ──────────
    def _phase2_consume_phase1_certificate(
        self,
        dilation: StinespringPotentialDilation,
        q_eval: NDArray[np.float64],
    ) -> Tuple[StinespringPotentialDilation, NDArray[np.float64], int]:
        r"""
        FASE 2.0 — Ingesta funtorial del certificado de FASE 1.Ω.

        **Continuación estricta de**
        ``Phase1_PotentialEnergyObserver.observe_potential_well``.
        Verifica la coherencia \((G,L,d)\) y entrega el objeto de trabajo
        de Orient *sin re-tipado ni re-Cholesky*.

        Raises:
            TypeError, MetricSingularityError, InitialStateError.
        """
        if not isinstance(dilation, StinespringPotentialDilation):
            raise TypeError("dilation debe ser StinespringPotentialDilation")
        d = int(dilation.manifold_dim or dilation.g_base.shape[0])
        if dilation.g_base.shape != (d, d):
            raise MetricSingularityError(
                f"FASE2.0: g_base shape={dilation.g_base.shape} ≠ {(d, d)}."
            )
        if dilation.cholesky_factor.shape != (d, d):
            raise MetricSingularityError(
                "FASE2.0: cholesky_factor incoherente con g_base."
            )
        q = np.asarray(q_eval, dtype=np.float64)
        self._assert_vector(q, d, "q_eval")
        logger.debug(
            "FASE2.0 consume F1: d=%d, H0=%.6f, Δ_min=%.6e, verdict=%s",
            d,
            dilation.initial_energy_h0,
            dilation.energy_gap_min,
            dilation.phase1_verdict.name,
        )
        return dilation, q, d

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    def _phase2_conformal_factor_at_q(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        clamp: bool = False,
    ) -> Tuple[float, float, float]:
        r"""
        FASE 2.1 — Factor conforme / índice de refracción en \(q\).

        \[
        \Delta=H_0-V(q),
        \qquad
        n=\frac{1}{\sqrt{2\Delta}},
        \qquad
        n^2=\frac{1}{2\Delta}.
        \]

        Si ``clamp`` y \(\Delta\le\varepsilon_{\mathrm{gap}}\) se regulariza
        el turning point (uso interno del integrador). En sintonía (F2.Ω)
        ``clamp=False`` y la barrera se eleva como veto.
        """
        self._assert_callable_potential(potential_v_fn)
        val_v = self._as_finite_scalar(potential_v_fn(q), "potential_v_fn(q)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            if not clamp:
                raise EnergyBarrierViolationError(
                    "Colapso de barrera de energía en el punto de evaluación "
                    f"geodésico: H0-V(q)={gap:.3e}."
                )
            gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        n_factor = math.sqrt(n_sq)
        if not math.isfinite(n_factor) or n_sq <= 0.0:
            raise GeodesicDivergenceError(
                f"Índice de refracción no físico: n²={n_sq}, n={n_factor}."
            )
        return n_factor, n_sq, gap

    # ── FASE 2.2 ──────────────────────────────────────────────────────────
    def _phase2_assemble_jacobi_metric(
        self,
        g_base: NDArray[np.float64],
        n_sq: float,
    ) -> NDArray[np.float64]:
        r"""
        FASE 2.2 — Ensamblaje de la métrica de Jacobi–Fermat.

        \[
        \widetilde{G}=n^2\,G.
        \]
        """
        conformal = n_sq * np.asarray(g_base, dtype=np.float64)
        if not np.all(np.isfinite(conformal)):
            raise MetricSingularityError(
                "Métrica conforme no finita (overflow de n²·G)."
            )
        return conformal

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    def _phase2_invert_conformal_cholesky(
        self,
        dilation: StinespringPotentialDilation,
        n_sq: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        FASE 2.3 — Inversión estable \(\widetilde{G}^{-1}=n^{-2}(LL^\top)^{-1}\).

        Reutiliza el factor de Cholesky de FASE 1.4 (sin refactorizar):

        \[
        G^{-1}=(L^{-1})^\top L^{-1},
        \qquad
        \widetilde{G}^{-1}=n^{-2}G^{-1}.
        \]
        """
        d = dilation.g_base.shape[0]
        chol = np.asarray(dilation.cholesky_factor, dtype=np.float64)
        try:
            eye = np.eye(d, dtype=np.float64)
            l_inv = la.solve_triangular(chol, eye, lower=True)
            g_inv = l_inv.T @ l_inv
        except la.LinAlgError as exc:
            raise MetricSingularityError(
                "Fallo de inversión de Cholesky de-confinada."
            ) from exc
        g_conformal_inv = (1.0 / n_sq) * g_inv
        return g_inv, g_conformal_inv

    # ── FASE 2.4 ──────────────────────────────────────────────────────────
    def _phase2_wilkinson_condition(
        self,
        g_base: NDArray[np.float64],
    ) -> float:
        r"""
        FASE 2.4 — Número de condición de Wilkinson.

        Como \(n^2\) es escalar, \(\kappa_2(\widetilde{G})=\kappa_2(G)\).
        Se evalúa sobre el espectro simétrico de \(G\):

        \[
        \kappa_2(G)=\lambda_{\max}(G)/\lambda_{\min}(G).
        \]
        """
        evals = la.eigvalsh(g_base)
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        if lam_min <= 0.0:
            raise MetricSingularityError(
                f"Espectro de G no SPD en FASE 2.4: λ_min={lam_min:.3e}."
            )
        cond = lam_max / lam_min
        if cond > _CONDITION_NUMBER_MAX:
            raise MetricSingularityError(
                f"Mal-condicionamiento de Wilkinson: cond(G) = {cond:.3e}"
            )
        logger.debug("FASE2.4 Wilkinson: κ₂=%.6e, λ∈[%.3e, %.3e]", cond, lam_min, lam_max)
        return float(cond)

    # ── FASE 2.5 ──────────────────────────────────────────────────────────
    def _phase2_estimate_potential_gradient(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        FASE 2.5 — Gradiente \(\nabla V(q)\) por CSMD con fallback central.

        * Si \(V\) admite extensión holomorfa, el paso complejo
          \(q_k\leftarrow q_k+ih\) extrae \(\partial_k V=\Im V(q+ihe_k)/h\)
          con error \(\mathcal{O}(h^2)\) sin cancelación sustractiva.
        * Si \(V\) no es holomorfa / no acepta \(\mathbb{C}\), se cae a
          diferencias centrales reales con \(h\sim\sqrt{\varepsilon_{\mathrm{mach}}}\).
        """
        d = int(q.shape[0])
        grad = np.zeros(d, dtype=np.float64)

        try:
            for k in range(d):
                q_c = q.astype(np.complex128, copy=True)
                q_c[k] += 1.0j * _CSMD_PERTURBATION
                val = potential_v_fn(q_c)
                if isinstance(val, np.ndarray):
                    val = val.reshape(()).item()
                imag = float(np.imag(val))
                grad[k] = imag / _CSMD_PERTURBATION
            if np.all(np.isfinite(grad)):
                logger.debug("FASE2.5 ∇V: método=csmd, ‖∇V‖=%.6e", float(la.norm(grad)))
                return grad, "csmd"
        except Exception as exc:  # noqa: BLE001 — fallback deliberado a diferencias reales
            logger.debug("FASE2.5 CSMD no aplicable (%s); se usa diferencias centrales.", exc)

        scale = max(1.0, float(la.norm(q)))
        step = math.sqrt(_MACHINE_EPS) * scale
        for k in range(d):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[k] += step
            q_minus[k] -= step
            f_plus = self._as_finite_scalar(potential_v_fn(q_plus), "V(q+he_k)")
            f_minus = self._as_finite_scalar(potential_v_fn(q_minus), "V(q-he_k)")
            grad[k] = (f_plus - f_minus) / (2.0 * step)
        if not np.all(np.isfinite(grad)):
            raise GeodesicDivergenceError("∇V no finito en el punto de evaluación.")
        logger.debug("FASE2.5 ∇V: método=central, ‖∇V‖=%.6e", float(la.norm(grad)))
        return grad, "central"

    # ── FASE 2.6 ──────────────────────────────────────────────────────────
    def _phase2_conformal_christoffel(
        self,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
    ) -> NDArray[np.float64]:
        r"""
        FASE 2.6 — Christoffel conforme en forma cerrada (\(G\) constante).

        Con \(\phi=\ln n=-\tfrac12\ln\bigl(2(H_0-V)\bigr)\) se tiene
        \(\nabla\phi=\nabla V\big/\,2\Delta\) y

        \[
        \widetilde{\Gamma}^i_{jk}
        =\delta^i_j\partial_k\phi+\delta^i_k\partial_j\phi
         -G_{jk}\,G^{il}\partial_l\phi.
        \]

        Complejidad \(\mathcal{O}(d^3)\) en el tensor (se materializa para
        el certificado); la aceleración de FASE 3 usa la contracción
        \(\mathcal{O}(d^2)\) equivalente.
        """
        d = g_base.shape[0]
        if gap <= 0.0:
            raise EnergyBarrierViolationError(
                "Gap no positivo al formar Christoffel conforme."
            )
        dphi = grad_v / (2.0 * gap)
        g_inv_dphi = g_inv @ dphi
        gamma = np.zeros((d, d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    gamma[i, j, k] = (
                        (dphi[k] if i == j else 0.0)
                        + (dphi[j] if i == k else 0.0)
                        - g_base[j, k] * g_inv_dphi[i]
                    )
        return gamma

    # ── FASE 2.7 ──────────────────────────────────────────────────────────
    def _phase2_koszul_torsion_freeness(
        self,
        gamma: NDArray[np.float64],
    ) -> float:
        r"""
        FASE 2.7 — Testigo de torsión nula (Levi-Civita).

        \[
        \tau=\max_{i,j,k}\bigl|\Gamma^i_{jk}-\Gamma^i_{kj}\bigr|.
        \]

        La forma cerrada es idénticamente libre de torsión; un \(\tau\)
        por encima de tolerancia delata corrupción de índices / redondeo
        patológico.
        """
        torsion = gamma - np.swapaxes(gamma, 1, 2)
        residual = float(np.max(np.abs(torsion)))
        if residual > max(_TORSION_ABS_TOL, 1.0e3 * _MACHINE_EPS):
            raise ConformalChecksumError(
                f"Conexión con torsión no nula: τ={residual:.3e}."
            )
        logger.debug("FASE2.7 Koszul: τ=%.3e (libre de torsión)", residual)
        return residual

    # ── FASE 2.8 ──────────────────────────────────────────────────────────
    def _phase2_christoffel_acceleration_identity(
        self,
        gamma: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
        probe: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""
        FASE 2.8 — Checksum \(\Gamma(v,v)\) ↔ aceleración contraída \(\mathcal{O}(d^2)\).

        \[
        \bigl(\Gamma(v,v)\bigr)^i
        =2(\nabla\phi\cdot v)\,v^i-\|v\|_G^2\,(G^{-1}\nabla\phi)^i.
        \]
        """
        d = g_base.shape[0]
        vec = (
            np.ones(d, dtype=np.float64) / math.sqrt(d)
            if probe is None
            else np.asarray(probe, dtype=np.float64)
        )
        dphi = grad_v / (2.0 * gap)
        contracted = np.einsum("ijk,j,k->i", gamma, vec, vec)
        closed = 2.0 * float(dphi @ vec) * vec - float(vec @ g_base @ vec) * (g_inv @ dphi)
        defect = float(la.norm(contracted - closed))
        scale = max(1.0, float(la.norm(closed)))
        if defect > 1.0e-9 * scale:
            raise ConformalChecksumError(
                f"Checksum Γ(v,v)↔a_cerrada corrupto: ‖Δ‖={defect:.3e}."
            )
        return defect

    def _compute_christoffel_at_q(
        self,
        q: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: Any,
        initial_h0: float,
        *,
        g_inv: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        r"""
        Fachada de compatibilidad: Christoffel conforme en \(q\) (forma cerrada).

        Sustituye el CSMD matricial de la v2 (incorrecto: evaluaba \(V\)
        en la parte real y no era holomorfo sobre \(G\)). El gradiente de
        \(V\) sí se estima por CSMD/central (FASE 2.5).
        """
        d = g_base.shape[0]
        self._assert_vector(q, d, "q")
        self._assert_square_matrix(g_base, "g_base")
        _n, _n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=True
        )
        if g_inv is None:
            g_inv = la.inv(np.asarray(g_base, dtype=np.float64))
        grad_v, _method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        return self._phase2_conformal_christoffel(g_base, g_inv, grad_v, gap)

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    def sintonizar_metrica_conforme(
        self,
        dilation: StinespringPotentialDilation,
        q_eval: NDArray[np.float64],
        potential_v_fn: Any,
    ) -> ConformalGeometryBundle:
        r"""
        FASE 2.Ω — Composición terminal Orient (métrica + Koszul + κ).

        **Continuación funtorial de FASE 1.Ω**: consume
        ``StinespringPotentialDilation`` vía FASE 2.0.

        **Contrato funtorial F2 → F3**: el DTO
        ``ConformalGeometryBundle`` es el objeto inicial exacto de
        ``_phase3_consume_phase2_certificate``.
        """
        dilation, q, _d = self._phase2_consume_phase1_certificate(dilation, q_eval)
        self._assert_callable_potential(potential_v_fn)

        g_base = np.asarray(dilation.g_base, dtype=np.float64)
        initial_h0 = float(dilation.initial_energy_h0)

        n_factor, n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False
        )
        g_conformal = self._phase2_assemble_jacobi_metric(g_base, n_sq)
        g_inv, g_conformal_inv = self._phase2_invert_conformal_cholesky(dilation, n_sq)
        cond = self._phase2_wilkinson_condition(g_base)
        grad_v, grad_method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        gamma = self._phase2_conformal_christoffel(g_base, g_inv, grad_v, gap)
        torsion = self._phase2_koszul_torsion_freeness(gamma)
        self._phase2_christoffel_acceleration_identity(
            gamma, g_base, g_inv, grad_v, gap
        )

        if cond > _CONDITION_NUMBER_SOFT:
            phase2_verdict = BrachistochroneHeytingVerdict.DEGRADED
        else:
            phase2_verdict = BrachistochroneHeytingVerdict.COHERENT

        logger.debug(
            "FASE2.Ω conforme: n=%.6e, Δ=%.6e, κ=%.3e, τ=%.3e, ∇V=%s, verdict=%s",
            n_factor,
            gap,
            cond,
            torsion,
            grad_method,
            phase2_verdict.name,
        )
        return ConformalGeometryBundle(
            g_base=self._freeze_array(g_base),
            g_conformal=self._freeze_array(g_conformal),
            g_conformal_inv=self._freeze_array(g_conformal_inv),
            refractive_index=self._freeze_array(np.array([n_factor], dtype=np.float64)),
            christoffel_tensor=self._freeze_array(gamma),
            wilkinson_condition_number=cond,
            potential_sane=self._freeze_array(dilation.potential_energy_v),
            initial_energy_h0=initial_h0,
            phase2_verdict=phase2_verdict,
            conformal_factor_sq=n_sq,
            torsion_residual=torsion,
            potential_gradient=self._freeze_array(grad_v),
            gradient_method=grad_method,
            energy_gap_at_eval=gap,
            phase1=dilation,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTEGRACIÓN DE FERMAT–JACOBI Y VETO CROWBAR (Decide & Act)
# Continuación directa de sintonizar_metrica_conforme (FASE 2.Ω) vía FASE 3.0
# Objetos: cascarón H0, geodésica RK4, E_F, β₁ discreto, Ω₃, Crowbar
# Teorías: geodésicas afines, conservación Fermat, Betti, Heyting
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_FermatBrachistochroneDecider(Phase2_ConformalMetricSuturator):
    r"""
    Fase 3: integra la geodésica de Fermat, confronta invariantes y decide.

    Morfismo compuesto:

    \[
    \mathrm{DecideFermat}
    =(\mathrm{Betti},\,E_F,\,\mathrm{RK4},\,a_{\widetilde{\Gamma}},\,\mathrm{Shell})
    \circ\mathrm{Consume}\circ\mathrm{OrientConformal}^*.
    \]

    El primer morfismo, ``_phase3_consume_phase2_certificate``, *es* la
    continuación estricta de ``sintonizar_metrica_conforme``.
    """

    # ── FASE 3.0 · ingesta funtorial del certificado de FASE 2.Ω ──────────
    def _phase3_consume_phase2_certificate(
        self,
        bundle: ConformalGeometryBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
    ) -> Tuple[ConformalGeometryBundle, NDArray[np.float64], NDArray[np.float64], int]:
        r"""
        FASE 3.0 — Ingesta funtorial del certificado de FASE 2.Ω.

        **Continuación estricta de**
        ``Phase2_ConformalMetricSuturator.sintonizar_metrica_conforme``.
        Verifica la coherencia geométrica de \(\widetilde{G}\) y
        \(\widetilde{\Gamma}\) (shape \(d^3\)) y entrega el objeto de
        trabajo de Decide *sin reconstruir la métrica conforme*.
        """
        if not isinstance(bundle, ConformalGeometryBundle):
            raise TypeError("bundle debe ser ConformalGeometryBundle")
        d = int(bundle.g_base.shape[0])
        if bundle.g_conformal.shape != (d, d) or bundle.g_conformal_inv.shape != (d, d):
            raise MetricSingularityError(
                "FASE3.0: tensores conformes incoherentes con g_base."
            )
        if bundle.christoffel_tensor.shape != (d, d, d):
            raise MetricSingularityError(
                f"FASE3.0: Christoffel shape={bundle.christoffel_tensor.shape} ≠ {(d, d, d)}."
            )
        q = np.asarray(q_start, dtype=np.float64)
        v = np.asarray(v_start, dtype=np.float64)
        self._assert_vector(q, d, "q_start")
        self._assert_vector(v, d, "v_start")
        logger.debug(
            "FASE3.0 consume F2: d=%d, κ=%.3e, n=%.6e, τ=%.3e, verdict=%s",
            d,
            bundle.wilkinson_condition_number,
            float(bundle.refractive_index.reshape(-1)[0]) if bundle.refractive_index.size else float("nan"),
            bundle.torsion_residual,
            bundle.phase2_verdict.name,
        )
        return bundle, q, v, d

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    def _phase3_project_energy_shell(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        project_on_shell: bool,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        FASE 3.1 — Proyección al cascarón de Maupertuis \(T=H_0-V\).

        Una braquistócrona de energía \(H_0\) vive en

        \[
        \tfrac12 v^\top G v=H_0-V(q).
        \]

        Si \(v=0\), se toma la dirección de descenso \(-G^{-1}\nabla V\)
        (o \(e_1\) si el gradiente es nulo). El residuo pre-proyección se
        reporta como diagnóstico.
        """
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q_start)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            raise EnergyBarrierViolationError(
                f"Cascarón vacío en el arranque: H0-V(q0)={gap:.3e}."
            )
        kinetic = 0.5 * float(v @ g_base @ v)
        residual = abs(kinetic - gap)
        if not project_on_shell:
            return np.asarray(v, dtype=np.float64), residual, False

        target_quad = 2.0 * gap  # vᵀ G v
        current_quad = float(v @ g_base @ v)
        if current_quad > _ENERGY_GAP_FLOOR:
            scaled = v * math.sqrt(target_quad / current_quad)
            return np.asarray(scaled, dtype=np.float64), residual, True

        grad_v, _method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        try:
            direction = -la.solve(g_base, grad_v, assume_a="pos")
        except la.LinAlgError:
            direction = -grad_v
        quad = float(direction @ g_base @ direction)
        if quad <= _ENERGY_GAP_FLOOR:
            direction = np.zeros_like(v)
            direction[0] = 1.0
            quad = float(direction @ g_base @ direction)
        scaled = direction * math.sqrt(target_quad / quad)
        logger.debug(
            "FASE3.1 cascarón: r_H=%.3e, proyectado=%s, ‖v‖_G²→%.6e",
            residual,
            True,
            target_quad,
        )
        return np.asarray(scaled, dtype=np.float64), residual, True

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    def _phase3_conformal_acceleration(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        r"""
        FASE 3.2 — Aceleración geodésica conforme en forma cerrada \(\mathcal{O}(d^2)\).

        \[
        a
        =-2(\nabla\phi\cdot v)\,v
         +\|v\|_G^2\,G^{-1}\nabla\phi,
        \qquad
        \nabla\phi=\frac{\nabla V}{2(H_0-V)}.
        \]

        Equivale a \(a^\rho=-\widetilde{\Gamma}^\rho_{\mu\nu}v^\mu v^\nu\)
        sin materializar el tensor \(d^3\).
        """
        _n, _n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False
        )
        grad_v, _method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        dphi = grad_v / (2.0 * gap)
        v_quad = float(v @ g_base @ v)
        accel = -2.0 * float(dphi @ v) * v + v_quad * (g_inv @ dphi)
        if not np.all(np.isfinite(accel)):
            raise GeodesicDivergenceError("Aceleración geodésica no finita.")
        return accel

    def _compute_acceleration(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: Any,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        """Fachada de compatibilidad: delega en la aceleración cerrada (FASE 3.2)."""
        d = g_base.shape[0]
        self._assert_vector(q, d, "q")
        self._assert_vector(v, d, "v")
        g_inv = la.inv(np.asarray(g_base, dtype=np.float64))
        return self._phase3_conformal_acceleration(
            q, v, g_base, g_inv, potential_v_fn, initial_h0
        )

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    def _phase3_fermat_energy(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> float:
        r"""
        FASE 3.3 — Invariante Fermat \(E_F=\tfrac12 n^2\|v\|_G^2\).

        Es la energía cinética de la métrica conforme y se conserva a
        lo largo de geodésicas afines. En el cascarón \(E_F=\tfrac12\).
        """
        n_factor, n_sq, _gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=True
        )
        del n_factor
        return 0.5 * n_sq * float(v @ g_base @ v)

    def _phase3_physical_hamiltonian(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[float, float]:
        r"""
        Hamiltoniano físico \(H=T+V\) (diagnóstico; no es el invariante afín).
        """
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q)")
        kinetic = 0.5 * float(v @ g_base @ v)
        return kinetic + val_v, val_v

    # ── FASE 3.4 ──────────────────────────────────────────────────────────
    def _rk4_step(
        self,
        state: NDArray[np.float64],
        dt: float,
        g_base: NDArray[np.float64],
        potential_v_fn: Any,
        initial_h0: float,
        *,
        g_inv: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """FASE 3.4 — Un paso de Runge–Kutta 4 para \(\dot q=v\), \(\dot v=a(q,v)\)."""
        d = g_base.shape[0]
        if g_inv is None:
            g_inv = la.inv(np.asarray(g_base, dtype=np.float64))

        def rhs(s: NDArray[np.float64]) -> NDArray[np.float64]:
            qq = s[:d]
            vv = s[d:]
            acc = self._phase3_conformal_acceleration(
                qq, vv, g_base, g_inv, potential_v_fn, initial_h0
            )
            return np.concatenate([vv, acc])

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(nxt)):
            raise GeodesicDivergenceError("Estado RK4 no finito.")
        return nxt

    # ── FASE 3.5 ──────────────────────────────────────────────────────────
    def _phase3_betti_acyclicity_witness(
        self,
        trajectory: NDArray[np.float64],
        *,
        min_lag_frac: float = _BETTI_MIN_LAG_FRAC,
        proximity_frac: float = _BETTI_PROXIMITY_FRAC,
    ) -> Tuple[bool, int]:
        r"""
        FASE 3.5 — Testigo discreto de aciclicidad (\(\beta_1\) proxy).

        Un retorno cercano a un punto con retraso mínimo
        \(\lfloor \lambda N\rfloor\) y umbral \(\varepsilon\cdot\mathrm{diam}\)
        se contabiliza como generador potencial de \(H_1\). No distingue
        oscilaciones 1-D de lazos espaciales: es un invariante *blando*
        salvo política ``require_acyclicity``.
        """
        n_pts = int(trajectory.shape[0])
        if n_pts < 8:
            return True, 0
        centered = trajectory - np.mean(trajectory, axis=0, keepdims=True)
        diameter = float(np.max(la.norm(centered, axis=1)))
        if diameter <= _ENERGY_GAP_FLOOR:
            return True, 0
        threshold = proximity_frac * diameter
        min_lag = max(3, int(min_lag_frac * n_pts))
        close_returns = 0
        # Muestreo sublineal: cada k-ésimo punto contra el prefijo retrasado.
        stride = max(1, n_pts // 256)
        for i in range(min_lag, n_pts, stride):
            pref = trajectory[: i - min_lag + 1 : stride]
            dist = np.min(la.norm(pref - trajectory[i], axis=1))
            if dist < threshold:
                close_returns += 1
        is_acyclic = close_returns == 0
        logger.debug(
            "FASE3.5 Betti: close_returns=%d, diam=%.6e, acyclic=%s",
            close_returns,
            diameter,
            is_acyclic,
        )
        return is_acyclic, int(close_returns)

    # ── FASE 3.Ω · composición terminal Decide (integración) ──────────────
    def integrar_braquistocrona(
        self,
        bundle: ConformalGeometryBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Any,
        t_max: float = 10.0,
        dt: float = 0.01,
        *,
        project_on_shell: bool = True,
        require_acyclicity: bool = False,
    ) -> BrachistochronePathResult:
        r"""
        FASE 3.Ω — Integra las geodésicas de Fermat–Jacobi mediante RK4.

        **Continuación funtorial de FASE 2.Ω**: consume
        ``ConformalGeometryBundle`` vía FASE 3.0.

        El parámetro de integración es el parámetro *afín* \(\lambda\) de
        \(\widetilde{G}\). El tiempo de tránsito físico/óptico es la
        longitud de Fermat

        \[
        T=\int n(q)\,\|dq\|_G
         =\int\sqrt{\widetilde{G}(\dot q,\dot q)}\,d\lambda.
        \]

        El invariante de conservación es \(E_F\), no \(H=T+V\) (este último
        se reporta como diagnóstico de cascarón).

        **Contrato funtorial F3 → Seal**: ``BrachistochronePathResult``
        alimenta el retículo de Heyting del agente soberano.
        """
        bundle, q0, v0, d = self._phase3_consume_phase2_certificate(
            bundle, q_start, v_start
        )
        self._assert_callable_potential(potential_v_fn)
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("dt debe ser positivo y finito.")
        if t_max <= 0.0 or not math.isfinite(t_max):
            raise ValueError("t_max debe ser positivo y finito.")

        steps = int(math.floor(t_max / dt + 1.0e-12))
        if steps < 1:
            steps = 1
        if steps > _MAX_INTEGRATION_STEPS:
            raise GeodesicDivergenceError(
                f"Horizonte de integración {steps} excede "
                f"max_steps={_MAX_INTEGRATION_STEPS}."
            )

        g_base = np.asarray(bundle.g_base, dtype=np.float64)
        initial_h0 = float(bundle.initial_energy_h0)
        try:
            g_inv = la.inv(g_base)
        except la.LinAlgError as exc:
            raise MetricSingularityError("Inversión de g_base fallida en FASE 3.Ω.") from exc

        v0, on_shell_res, projected = self._phase3_project_energy_shell(
            q0,
            v0,
            g_base,
            potential_v_fn,
            initial_h0,
            project_on_shell=project_on_shell,
        )

        trajectory = np.zeros((steps, d), dtype=np.float64)
        velocities = np.zeros((steps, d), dtype=np.float64)
        state = np.concatenate([q0, v0])
        energy_drift_max = 0.0
        fermat_drift_max = 0.0
        transit_time_t = 0.0
        is_path_stable = True
        last_index = 0

        e_f_0 = self._phase3_fermat_energy(q0, v0, g_base, potential_v_fn, initial_h0)

        for step in range(steps):
            q_curr = state[:d]
            v_curr = state[d:]
            trajectory[step] = q_curr
            velocities[step] = v_curr
            last_index = step

            try:
                ham, val_v = self._phase3_physical_hamiltonian(
                    q_curr, v_curr, g_base, potential_v_fn
                )
                e_f = self._phase3_fermat_energy(
                    q_curr, v_curr, g_base, potential_v_fn, initial_h0
                )
            except (GeodesicDivergenceError, EnergyBarrierViolationError, ArithmeticError) as exc:
                logger.warning("Singularidad de observación en paso %d: %s", step, exc)
                is_path_stable = False
                break

            energy_drift_max = max(energy_drift_max, abs(ham - initial_h0))
            fermat_drift_max = max(fermat_drift_max, abs(e_f - e_f_0))

            v_quad = float(v_curr @ g_base @ v_curr)
            v_norm_g = math.sqrt(max(0.0, v_quad))
            gap = initial_h0 - val_v
            if gap <= _ENERGY_GAP_FLOOR:
                logger.warning("Barrera de energía alcanzada en paso %d (Δ=%.3e).", step, gap)
                is_path_stable = False
                break
            n_factor = 1.0 / math.sqrt(2.0 * gap)
            transit_time_t += n_factor * v_norm_g * dt

            try:
                state = self._rk4_step(
                    state, dt, g_base, potential_v_fn, initial_h0, g_inv=g_inv
                )
            except (GeodesicDivergenceError, MetricSingularityError, EnergyBarrierViolationError) as exc:
                logger.warning("Singularidad geodésica en paso %d: %s", step, exc)
                is_path_stable = False
                break

        used = last_index + 1
        traj = trajectory[:used]
        vel = velocities[:used]
        is_acyclic, close_returns = self._phase3_betti_acyclicity_witness(traj)
        if require_acyclicity and not is_acyclic:
            raise BettiAcyclicityVetoError(
                f"Ciclo homológico parásito: close_returns={close_returns}."
            )

        logger.debug(
            "FASE3.Ω integrar: steps=%d/%d, T=%.6e, δH=%.3e, δE_F=%.3e, "
            "acyclic=%s, stable=%s",
            used,
            steps,
            transit_time_t,
            energy_drift_max,
            fermat_drift_max,
            is_acyclic,
            is_path_stable,
        )
        return BrachistochronePathResult(
            trajectory=self._freeze_array(traj),
            velocities=self._freeze_array(vel),
            transit_time_t=float(transit_time_t),
            energy_drift_max=float(energy_drift_max),
            is_path_stable=bool(is_path_stable),
            fermat_energy_initial=float(e_f_0),
            fermat_energy_drift_max=float(fermat_drift_max),
            affine_time=float(used * dt),
            steps_accepted=int(used),
            is_acyclic=bool(is_acyclic),
            betti_close_returns=int(close_returns),
            on_shell_residual=float(on_shell_res),
            velocity_projected=bool(projected),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CONTROL GEODÉSICO DE SUTURA
# Observe (F1) ⟶ Orient (F2) ⟶ Decide (F3) ⟶ Seal / Crowbar
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochronePathFinderAgent(Morphism, Phase3_FermatBrachistochroneDecider):
    r"""
    Agente Soberano del Optimizador de Descenso Rápido.

    Endofuntor de gobernanza:

    \[
    \mathcal{OODA}_{\mathrm{Fermat}}
    :
    (G,q_0,v_0,V,H_0)
    \longrightarrow
    \mathbf{GovState}(\Omega_3)
    \]

    compuesto como

    \[
    \mathrm{Seal}\circ\mathrm{Integrate}\circ\mathrm{Conformal}\circ\mathrm{Observe}.
    \]
    """

    def __init__(
        self,
        crowbar_actuator: Optional[CrowbarActuator] = None,
        raise_on_veto: bool = False,
    ) -> None:
        super().__init__()
        self._crowbar = crowbar_actuator or LoggingCrowbarActuator()
        self._raise_on_veto = raise_on_veto

    # ── FASE Ω.1 · conjunción de hard-gates en Ω₃ ─────────────────────────
    @staticmethod
    def _phase_omega_heyting_join(
        *verdicts: BrachistochroneHeytingVerdict,
    ) -> BrachistochroneHeytingVerdict:
        """Supremo del retículo \(\{\mathrm{COHERENT}\prec\mathrm{DEGRADED}\prec\mathrm{VETOED}\)."""
        if not verdicts:
            return BrachistochroneHeytingVerdict.COHERENT
        return BrachistochroneHeytingVerdict(max(v.value for v in verdicts))

    def _phase_omega_decide_heyting_lattice(
        self,
        dilation: StinespringPotentialDilation,
        bundle: ConformalGeometryBundle,
        path_result: BrachistochronePathResult,
        *,
        require_acyclicity: bool,
    ) -> BrachistochroneHeytingVerdict:
        r"""
        FASE Ω.1 — Clasificador \(\chi\in\Omega_3\) a partir de los tres certificados.

        Hard-gates: inestabilidad de trayectoria, deriva Fermat dura,
        deriva física dura, \(\kappa>\kappa_{\max}\), aciclicidad (si política).
        Soft-gates: deriva / κ en zona gris → DEGRADED.
        """
        components = [dilation.phase1_verdict, bundle.phase2_verdict]
        if not path_result.is_path_stable:
            components.append(BrachistochroneHeytingVerdict.VETOED)
        if path_result.fermat_energy_drift_max > _FERMAT_DRIFT_HARD_TOL:
            components.append(BrachistochroneHeytingVerdict.VETOED)
        elif path_result.fermat_energy_drift_max > _FERMAT_DRIFT_SOFT_TOL:
            components.append(BrachistochroneHeytingVerdict.DEGRADED)
        if path_result.energy_drift_max > _ENERGY_DRIFT_HARD_TOL:
            components.append(BrachistochroneHeytingVerdict.VETOED)
        elif path_result.energy_drift_max > _ENERGY_DRIFT_SOFT_TOL:
            components.append(BrachistochroneHeytingVerdict.DEGRADED)
        if bundle.wilkinson_condition_number > _CONDITION_NUMBER_MAX:
            components.append(BrachistochroneHeytingVerdict.VETOED)
        elif bundle.wilkinson_condition_number > _CONDITION_NUMBER_SOFT:
            components.append(BrachistochroneHeytingVerdict.DEGRADED)
        if require_acyclicity and not path_result.is_acyclic:
            components.append(BrachistochroneHeytingVerdict.VETOED)
        elif not path_result.is_acyclic:
            components.append(BrachistochroneHeytingVerdict.DEGRADED)
        return self._phase_omega_heyting_join(*components)

    # ── FASE Ω.2 · sellado ────────────────────────────────────────────────
    def _phase_omega_seal_governance_state(
        self,
        verdict: BrachistochroneHeytingVerdict,
        action_report: CrowbarBypassAction,
        path_result: Optional[BrachistochronePathResult],
        cond_num: float,
        energy_drift: float,
        time_t: float,
        diagnostic_msg: str,
        latency_ms: float,
        *,
        require_acyclicity: bool,
        torsion_residual: float,
    ) -> BrachistochroneGovernanceState:
        """FASE Ω.2 — Sellado del objeto terminal frozen del funtor OODA."""
        timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fermat_drift = (
            path_result.fermat_energy_drift_max if path_result is not None else 0.0
        )
        prov_payload = (
            f"{verdict.name}-{time_t:.6e}-{energy_drift:.6e}-"
            f"{fermat_drift:.6e}-{cond_num:.6e}-{latency_ms:.3f}-{__version__}"
        )
        provenance_hash = hashlib.sha256(prov_payload.encode("utf-8")).hexdigest()
        return BrachistochroneGovernanceState(
            verdict=verdict,
            crowbar_report=action_report,
            transit_time=time_t,
            energy_drift_max=energy_drift,
            wilkinson_condition_number=cond_num,
            is_epistemologically_valid=(verdict != BrachistochroneHeytingVerdict.VETOED),
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"{diagnostic_msg} Latencia de lazo: {latency_ms:.3f} ms.",
            fermat_energy_drift_max=fermat_drift,
            is_acyclic=True if path_result is None else path_result.is_acyclic,
            betti_close_returns=0 if path_result is None else path_result.betti_close_returns,
            torsion_residual=torsion_residual,
            steps_accepted=0 if path_result is None else path_result.steps_accepted,
            agent_version=__version__,
            policy_require_acyclicity=require_acyclicity,
        )

    # ── FASE Ω.Veto ───────────────────────────────────────────────────────
    def _phase_omega_apply_crowbar(
        self,
        verdict: BrachistochroneHeytingVerdict,
    ) -> CrowbarBypassAction:
        """Dispara el actuador según el subobjeto de Heyting (una sola vez)."""
        if verdict == BrachistochroneHeytingVerdict.VETOED:
            action = CrowbarBypassAction.HARD_SHORT
            self._crowbar.trigger_crowbar_bypass(action)
            return action
        if verdict == BrachistochroneHeytingVerdict.DEGRADED:
            action = CrowbarBypassAction.WATCHDOG_PULSE
            self._crowbar.trigger_crowbar_bypass(action)
            return action
        return CrowbarBypassAction.NONE

    # ── Compositor público OODA ───────────────────────────────────────────
    def execute_brachistochrone_governance(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Any,
        initial_h0: float,
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
        require_acyclicity: bool = False,
    ) -> BrachistochroneGovernanceState:
        r"""
        Orquesta el ciclo OODA completo sobre la braquistócrona de-confinada.

        El objeto terminal de cada fase es el objeto inicial de la siguiente:

        .. code-block:: text

            ┌────────────────────────────────────────────────────────────┐
            │ FASE 1  Observe / Energy                                   │
            │   1.1 validate_manifold_dimension                          │
            │   1.2 validate_metric_tensor                               │
            │   1.3 certify_metric_symmetry                              │
            │   1.4 cholesky_spd_factor                                  │
            │   1.5 validate_potential_samples                           │
            │   1.6 energy_gap_spectrum                                  │
            │   1.7 validate_initial_kinematics                          │
            │   1.Ω observe_potential_well  ──► StinespringDil. ──┐      │
            ├─────────────────────────────────────────────────────┼──────┤
            │ FASE 2  Orient / Conformal  ◄───────────────────────┘      │
            │   2.0 consume_phase1_certificate                           │
            │   2.1 conformal_factor_at_q                                │
            │   2.2 assemble_jacobi_metric                               │
            │   2.3 invert_conformal_cholesky                            │
            │   2.4 wilkinson_condition                                  │
            │   2.5 estimate_potential_gradient                          │
            │   2.6 conformal_christoffel  (forma cerrada)               │
            │   2.7 koszul_torsion_freeness                              │
            │   2.8 christoffel_acceleration_identity                    │
            │   2.Ω sintonizar_metrica_conforme  ──► Bundle ──┐          │
            ├─────────────────────────────────────────────────┼──────────┤
            │ FASE 3  Decide / Integrate  ◄───────────────────┘          │
            │   3.0 consume_phase2_certificate                           │
            │   3.1 project_energy_shell                                 │
            │   3.2 conformal_acceleration                               │
            │   3.3 fermat_energy                                        │
            │   3.4 rk4_step                                             │
            │   3.5 betti_acyclicity_witness                             │
            │   3.Ω integrar_braquistocrona  ──► PathResult ──┐          │
            ├─────────────────────────────────────────────────┼──────────┤
            │ SEAL / CROWBAR  ◄───────────────────────────────┘          │
            │   Ω.1 decide_heyting_lattice                               │
            │   Ω.2 seal_governance_state                                │
            │   Ω.Veto apply_crowbar                                     │
            └────────────────────────────────────────────────────────────┘

        Args:
            g_base: tensor métrico SPD \((d,d)\), constante.
            q_start, v_start: condiciones iniciales en \(T\mathcal{Q}\).
            potential_v_fn: campo \(V(q)\to\mathbb{R}\) (holomorfo ⇒ CSMD).
            initial_h0: energía total de Maupertuis.
            t_max, dt: horizonte y paso del parámetro afín.
            project_on_shell: reescala \(v_0\) al cascarón \(T=H_0-V\).
            require_acyclicity: eleva ``BettiAcyclicityVetoError`` / VETOED
                si el testigo de retornos es positivo.

        Returns:
            BrachistochroneGovernanceState: certificado maestro con veredicto.

        Raises:
            CrowbarTriggeredError: si ``raise_on_veto`` y el retículo colapsa.
        """
        t_start_nano = time.perf_counter_ns()
        diagnostic_msg = "Gobernanza Geodésica iniciada exitosamente."
        verdict = BrachistochroneHeytingVerdict.COHERENT
        action_report = CrowbarBypassAction.NONE
        cond_num = 1.0
        energy_drift = 0.0
        time_t = 0.0
        torsion = 0.0
        path_result: Optional[BrachistochronePathResult] = None
        pending_raise: Optional[CrowbarTriggeredError] = None

        try:
            self._sanitize_inputs(g_base, q_start, v_start, initial_h0)
            self._assert_callable_potential(potential_v_fn)
            val_start = self._as_finite_scalar(potential_v_fn(q_start), "V(q_start)")

            # ── FASE 1 · Observe ──────────────────────────────────────────
            dilation = self.observe_potential_well(
                g_base=np.asarray(g_base, dtype=np.float64),
                potential_v=np.array([val_start], dtype=np.float64),
                initial_h0=float(initial_h0),
                q_start=np.asarray(q_start, dtype=np.float64),
                v_start=np.asarray(v_start, dtype=np.float64),
                potential_at_start=val_start,
            )

            # ── FASE 2 · Orient  (continúa certificado F1.Ω) ──────────────
            bundle = self.sintonizar_metrica_conforme(
                dilation=dilation,
                q_eval=np.asarray(q_start, dtype=np.float64),
                potential_v_fn=potential_v_fn,
            )
            cond_num = bundle.wilkinson_condition_number
            torsion = bundle.torsion_residual

            # ── FASE 3 · Decide  (continúa certificado F2.Ω) ──────────────
            path_result = self.integrar_braquistocrona(
                bundle=bundle,
                q_start=np.asarray(q_start, dtype=np.float64),
                v_start=np.asarray(v_start, dtype=np.float64),
                potential_v_fn=potential_v_fn,
                t_max=t_max,
                dt=dt,
                project_on_shell=project_on_shell,
                require_acyclicity=require_acyclicity,
            )
            energy_drift = path_result.energy_drift_max
            time_t = path_result.transit_time_t

            verdict = self._phase_omega_decide_heyting_lattice(
                dilation,
                bundle,
                path_result,
                require_acyclicity=require_acyclicity,
            )
            if verdict == BrachistochroneHeytingVerdict.VETOED:
                diagnostic_msg = (
                    f"VETO TOPOLÓGICO: estable={path_result.is_path_stable}, "
                    f"δH={energy_drift:.3e}, δE_F={path_result.fermat_energy_drift_max:.3e}, "
                    f"κ={cond_num:.3e}, acyclic={path_result.is_acyclic}."
                )
            elif verdict == BrachistochroneHeytingVerdict.DEGRADED:
                diagnostic_msg = (
                    f"Degradación espectral: δH={energy_drift:.3e}, "
                    f"δE_F={path_result.fermat_energy_drift_max:.3e}, κ={cond_num:.3e}."
                )

        except CrowbarTriggeredError:
            raise
        except (BrachistochroneAgentError, TopologicalInvariantError) as exc:
            verdict = BrachistochroneHeytingVerdict.VETOED
            diagnostic_msg = f"VETO TOPOLÓGICO: {exc}"
            cond_num = _CONDITION_NUMBER_MAX * 10.0
            energy_drift = 1.0
            time_t = math.inf
            if self._raise_on_veto:
                pending_raise = CrowbarTriggeredError(diagnostic_msg)
                pending_raise.__cause__ = exc
        except Exception as exc:  # noqa: BLE001 — fail-secure del lazo de gobernanza
            verdict = BrachistochroneHeytingVerdict.VETOED
            diagnostic_msg = f"FALLO CRÍTICO IMPREVISTO: {exc}"
            cond_num = _CONDITION_NUMBER_MAX * 10.0
            energy_drift = 1.0
            time_t = math.inf
            if self._raise_on_veto:
                pending_raise = CrowbarTriggeredError(diagnostic_msg)
                pending_raise.__cause__ = exc

        action_report = self._phase_omega_apply_crowbar(verdict)
        t_end_nano = time.perf_counter_ns()
        latency_ms = (t_end_nano - t_start_nano) / 1.0e6

        state = self._phase_omega_seal_governance_state(
            verdict,
            action_report,
            path_result,
            cond_num,
            energy_drift,
            time_t,
            diagnostic_msg,
            latency_ms,
            require_acyclicity=require_acyclicity,
            torsion_residual=torsion,
        )
        if pending_raise is not None:
            raise pending_raise
        if self._raise_on_veto and verdict == BrachistochroneHeytingVerdict.VETOED:
            raise CrowbarTriggeredError(diagnostic_msg)
        return state

    # ─────────────────────────────────────────────────────────────────────
    # Fábricas de referencia (calibración / tests del agente)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def euclidean_metric(dimension_d: int) -> NDArray[np.float64]:
        r"""Métrica euclídea \(G=I_d\)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        return np.eye(int(dimension_d), dtype=np.float64)

    @staticmethod
    def harmonic_potential(
        omega: float = 1.0,
        center: Optional[NDArray[np.float64]] = None,
    ) -> Callable[[NDArray[Any]], Any]:
        r"""
        Pozo armónico \(V(q)=\tfrac12\omega^2\|q-q_\star\|^2\).

        Holomorfo en \(q\) (usa \(q\cdot q\), no \(|q|^2\)), por lo que
        admite CSMD.
        """
        if omega < 0.0 or not math.isfinite(omega):
            raise ValueError(f"omega debe ser ≥ 0 y finito; recibido {omega}")

        def _v(q: NDArray[Any]) -> Any:
            qq = np.asarray(q)
            if center is None:
                delta = qq
            else:
                delta = qq - np.asarray(center)
            return 0.5 * (omega ** 2) * np.dot(delta, delta)

        return _v

    @staticmethod
    def linear_gravity_potential(
        g_acc: float = 1.0,
        axis: int = -1,
    ) -> Callable[[NDArray[Any]], Any]:
        r"""
        Gravedad lineal \(V(q)=-g\,q_{\mathrm{axis}}\) (braquistócrona clásica
        de Bernoulli cuando \(G=I_2\) y el eje es el vertical).
        """
        if not math.isfinite(g_acc):
            raise ValueError(f"g_acc debe ser finito; recibido {g_acc}")

        def _v(q: NDArray[Any]) -> Any:
            qq = np.asarray(q)
            return -g_acc * qq[axis]

        return _v

    @staticmethod
    def constant_potential(value: float = 0.0) -> Callable[[NDArray[Any]], Any]:
        """Potencial constante (geodésicas = rectas de \(G\))."""
        if not math.isfinite(value):
            raise ValueError(f"value debe ser finito; recibido {value}")

        def _v(q: NDArray[Any]) -> float:
            del q
            return float(value)

        return _v


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "BrachistochroneHeytingVerdict",
    "CrowbarBypassAction",
    "CrowbarActuator",
    "PotentialField",
    "LoggingCrowbarActuator",
    "BrachistochroneAgentError",
    "EnergyBarrierViolationError",
    "MetricSingularityError",
    "GeodesicDivergenceError",
    "BettiAcyclicityVetoError",
    "CrowbarTriggeredError",
    "ConformalChecksumError",
    "InitialStateError",
    "StinespringPotentialDilation",
    "ConformalGeometryBundle",
    "BrachistochronePathResult",
    "BrachistochroneGovernanceState",
    "Phase1_PotentialEnergyObserver",
    "Phase2_ConformalMetricSuturator",
    "Phase3_FermatBrachistochroneDecider",
    "BrachistochronePathFinderAgent",
    "__version__",
]