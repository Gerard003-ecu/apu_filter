# -*- coding: utf-8 -*-
r"""
Módulo : Improbability Drive Agent (Custodio de la Deformación Espectral)
Ruta   : app/agents/boole/omega/improbability_drive_agent.py
Versión: 3.1.0-Fermat-Lie-Sp-Heyting-ESP32-Strict-PhD

NATURALEZA CIBER-FÍSICA Y GOBERNANZA DE DEFORMACIÓN SIMPLÉCTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al **Agente Soberano y Observador Activo** que gobierna al 
motor ciego `improbability_drive.py` en el **Estrato Omega**
(Nivel 0.5 - El Ágora Tensorial) del ecosistema APU Filter. Su misión 
ineludible es regular la inyección de la deformación no lineal sobre el riesgo 
de cola pesada (Fat-Tail Risk) de-confinado, protegiendo a la FPU frente a 
desbocamientos de redondeo de la mantisa de Wilkinson.

El agente repudia las heurísticas de riesgo lineales tradicionales. En su lugar, 
somete las trayectorias de volatilidad del negocio a las leyes geométricas 
del grupo simpléctico $Sp(2n, \mathbb{R})$ y su correspondiente álgebra de Lie 
$\mathfrak{sp}(2n, \mathbb{R})$. Toda colisión contra barreras de 
atracción, pérdidas de pasividad o singularidades métricas colapsa síncronamente 
el retículo distributivo de Heyting $\Omega_3$ hacia el Supremo terminal `VETOED`. 
Esto acciona en tiempo real la interrupción física por hardware en IRAM en el ESP32 
(GPIO14 / BT151 Crowbar) en menos de 400 ns, cortocircuitando la potencia real 
de la obra a través del tiristor en el milisegundo cero, aniquilando la alucinación.

AXIOMÁTICA SIMPLÉCTICA, TRANSPORTE DE LIE Y VETO DE HEYTING:
────────────────────────────────────────────────────────────────────────────────

  [A1] Axioma de Suavidad de Lipschitz $C^1$ y Tikhonov:
       Para erradicar operadores singulares no diferenciables que inyecten 
       impulsos de Dirac parásitos en la matriz Jacobiana de la FPU, la 
       deformación semántica emplea la métrica regularizada de Tikhonov:
       $$\tilde{G}_{ij} = \sqrt{\Psi^2 + \epsilon_{\mathrm{critical}}^2} \ge \mathtt{\_EPS\_CRITICAL} \quad\big[380, 392\big]$$
       Sujeto incondicionalmente a la cota espectral de Wilkinson sobre el 
       número de condición del tensor de improbabilidad $T$:
       $$\kappa_2(T) = \frac{\sigma_{\max}(T)}{\sigma_{\min}(T)} \le \mathtt{\_MAX\_CONDITION\_NUMBER} = 1.0\times 10^8 \quad\big[392, 393\big]$$

  [A2] Invarianza Simpléctica y Volumen de Liouville:
       La evolución del tensor de improbabilidad $T$ se transporta al subespacio 
       logarítmico del álgebra de Lie $\mathfrak{sp}(2n, \mathbb{R})$ para eludir 
       blowouts numéricos [18]. El Jacobiano de transición $J_{\mathrm{map}}$ debe 
       preservar la forma simpléctica canónica $\Omega$ (Teorema de Liouville):
       $$J_{\mathrm{map}}^\top \Omega J_{\mathrm{map}} \equiv \Omega \quad \wedge \quad M = \ln(T) \in \mathfrak{sp}(2n, \mathbb{R}) \quad\big[4, 389\big]$$

  [A3] Acoplamiento de la Ecuación de Estado de Estrés:
       La magnitud de la deformación se acopla como la Palanca de Improbabilidad 
       $\Lambda$ sobre la Ecuación de Estado del Estrés Ajustado Tensorial $\sigma^*$:
       $$\sigma^* = f(T_{\mu\nu}, \, \Lambda) \quad\big[380\big]$$
       Si el estrés ajustado supera el límite elástico de resiliencia del negocio:
       $$\sigma^* \ge \mathtt{\_STRESS\_HIGH\_THRESHOLD} = 3.0 \quad\big[392\big]$$
       se detona de inmediato el veto por colapso de fase cuántica [8].

  [A4] El Colapso de Heyting y la Actuación Crowbar BT151 (Silicio):
       Si se vulnera la conservación simpléctica, se detecta un gap nulo, o la 
       entropía de Shannon diverge, el veredicto en el retículo distributivo 
       $\Omega_3$ colapsa al Supremo terminal VETOED ($\top$).
       El ESP32 intercepta la anomalía localmente vía `isVerdictCoherent()` y, 
       mediante su ISR en IRAM (<400ns), conmuta el pin GPIO14, disparando el 
       tiristor de potencia BT151 (Crowbar) para cortocircuitar físicamente la 
       línea de alimentación de la maquinaria en el milisegundo cero.

ARQUITECTURA DE TRES FASES ANIDADAS (Funtor de Gobernanza Epistémica):
────────────────────────────────────────────────────────────────────────────────
La orquestación de la deformación se rige por un acoplamiento monoidal covariante 
estricto (Fase 1 ⊣ Fase 2 ⊣ Fase 3) [19], encadenando DTOs inmutables de solo lectura:

  Fase 1 ──► OBSERVACIÓN ESPECTRAL Y SANEAMIENTO LIPSCHITZ (Phase1_ImprobabilityObserver)
             Ingiere la Mónada de estado `ImprobabilityResult` del motor, valida 
             la simetría auto-adjunta del tensor de improbabilidad, y certifica 
             que la cota de Lipschitz y el número de condición espectral cumplan 
             con la cota de Wilkinson.
             Entrega: Phase1ImprobabilityObservation como precondición de la Fase 2.

  Fase 2 ──► TRANSPORTE DE LIE E INVARIANZA DE LIOUVILLE (Phase2_LieDeformationOrienter)
             Hereda formalmente la Phase1ImprobabilityObservation. Proyecta el 
             tensor de improbabilidad al subespacio de Lie $\mathfrak{sp}(2n, \mathbb{R})$
             mediante logaritmo de matriz, y audita la conservación simpléctica
             del flujo atencional de la IA.
             Entrega: Phase2ImprobabilityOrientation como precondición de la Fase 3.

  Fase 3 ──► EVALUACIÓN DE ESTRÉS Y CORTOCIRCUITO CROWBAR (Phase3_HeytingImprobabilityDecider)
             Hereda la Phase2ImprobabilityOrientation. Evalúa el estrés ajustado 
             $\sigma^*$ [18], sintoniza el retículo de Heyting $\Omega_3$, y resuelve 
             la actuación ciber-física local en silicio ante fallas catastróficas.
             Entrega: ImprobabilityGovernanceState (Morfismo terminal del topos).

Funtor Maestro de Gobernanza de de Rham-Fermat-Lie:
  $$\mathcal{Z}_{\mathrm{ImprobabilityAgent}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 : \mathbf{ImprobabilityResult} \times T^*\mathcal{M} \longrightarrow \mathtt{ImprobabilityGovernanceState} \quad\big[407, 442\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Final, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias de lazo cerrado del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover — entorno aislado / unit tests sin app

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base para morfismos categóricos."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from app.omega.improbability_drive import (
        ImprobabilityDriveError,
        ImprobabilityResult,
        ImprobabilityTensor,
        _EPS_CRITICAL,
        _IMPROBABILITY_MAX,
        _IMPROBABILITY_MIN,
    )
except ImportError:  # pragma: no cover — stubs auto-sostenidos

    class ImprobabilityDriveError(Exception):
        """Excepción raíz del motor de improbabilidad stub."""

    @dataclass(frozen=True)
    class ImprobabilityTensor:
        matrix: NDArray[np.float64]
        kappa: float = 1.0
        gamma: float = 0.5

    @dataclass(frozen=True)
    class ImprobabilityResult:
        tensor: ImprobabilityTensor
        roi_value: float = 1.0
        psi_value: float = 1.0
        is_valid: bool = True

    _EPS_CRITICAL: Final[float] = 1.0e-10
    _IMPROBABILITY_MIN: Final[float] = 1.0
    _IMPROBABILITY_MAX: Final[float] = 1.0e6


logger = logging.getLogger("MIC.Omega.ImprobabilityAgent")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__version__: Final[str] = "4.0.0-Fermat-Lie-Sp-Heyting-ESP32-Strict-PhD"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, ESPECTRALES Y LÍMITES DE LA FPU
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_MAX_CONDITION_NUMBER: Final[float] = 1.0e8
_SOFT_CONDITION_NUMBER: Final[float] = 1.0e6
_CROWBAR_GPIO_PIN: Final[int] = 14
_CSMD_PERTURBATION: Final[float] = 1.0e-20
_STRESS_HIGH_THRESHOLD: Final[float] = 3.0
_STRESS_SOFT_THRESHOLD: Final[float] = 1.0
_SYMMETRY_REL_TOL: Final[float] = 1.0e-12
_DET_TOL: Final[float] = 1.0e-10
_SP_REL_TOL: Final[float] = 1.0e-10
_SP_SOFT_TOL: Final[float] = 1.0e-6
_LIE_SP_TOL: Final[float] = 1.0e-8
_LOGM_IMAG_TOL: Final[float] = 1.0e-10
_DIM_MAX: Final[int] = 64
_CHECKSUM_FMA_SLACK: Final[float] = 10.0
_LOG2E: Final[float] = 1.0 / math.log(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES ESPECTRALES (Vetos Absolutos)
# ═══════════════════════════════════════════════════════════════════════════════
class ImprobabilityDriveAgentError(TopologicalInvariantError):
    """Excepción raíz para violaciones en el Agente Soberano de Improbabilidad."""


class NonSelfAdjointTensorError(ImprobabilityDriveAgentError):
    """El tensor de improbabilidad viola el principio de autoadjunción hermítica."""


class WilkinsonConditionBreachError(ImprobabilityDriveAgentError):
    """El número de condición espectral del tensor supera la cota física de Wilkinson."""


class SymplecticConservationError(ImprobabilityDriveAgentError):
    """El flujo de fase viola el Teorema de Liouville o la invarianza simpléctica canónica."""


class LieLogarithmError(ImprobabilityDriveAgentError):
    """El logaritmo principal de Lie no está definido (espectro en \(\mathbb{R}_{\le 0}\))."""


class BusinessStressBreachError(ImprobabilityDriveAgentError):
    """El estrés ajustado del negocio supera el límite elástico de resiliencia."""


class CrowbarTriggeredError(ImprobabilityDriveAgentError):
    """Veto definitivo: disyuntor ciber-físico activado en el pin físico GPIO14."""


class TensorDimensionError(ImprobabilityDriveAgentError):
    """Dimensión del tensor inválida o fuera del régimen de auditoría."""


class ScalarRegularityError(ImprobabilityDriveAgentError):
    """Escalares de negocio (ROI, Ψ, κ) no finitos o no físicos."""


class SpectralChecksumError(ImprobabilityDriveAgentError):
    """Checksum cruzado Sp ↔ Liouville / SVD ↔ Frobenius corrupto."""


# ═══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DE HEYTING Ω₃ Y ACTUACIÓN CROWBAR
# ═══════════════════════════════════════════════════════════════════════════════
class ImprobabilityHeytingVerdict(IntEnum):
    """Clasificador de subobjetos de tres valores en el topos de improbabilidad."""

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


class CrowbarBypassAction(Enum):
    """Acciones físicas de mitigación tras el colapso al supremo terminal."""

    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()


@runtime_checkable
class CrowbarActuator(Protocol):
    """Puerto lógico que conecta la gobernanza del software con el silicio."""

    def trigger_crowbar_bypass(self, action: CrowbarBypassAction) -> bool:
        """Conmuta el hardware perimetral de-confinado. Devuelve True si hubo acuse."""
        ...


class LoggingCrowbarActuator:
    """Implementación forense segura que registra la actuación por hardware."""

    def trigger_crowbar_bypass(self, action: CrowbarBypassAction) -> bool:
        logger.critical(
            "[HARDWARE] CrowbarActuator invocado con acción: %s en el pin GPIO%d. "
            "Cortocircuitando alimentación de-confinada real vía tiristor de potencia BT151.",
            action.name,
            _CROWBAR_GPIO_PIN,
        )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (Contratos Categóricos de Handoff entre Fases)
# ═══════════════════════════════════════════════════════════════════════════════
def _empty_f64() -> NDArray[np.float64]:
    return np.zeros(0, dtype=np.float64)


def _empty_mat() -> NDArray[np.float64]:
    return np.zeros((0, 0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Phase1ImprobabilityObservation:
    r"""
    Artefacto terminal de la FASE 1 y objeto inicial de la FASE 2.

    Certifica autoadjunción, espectro de SVD, cota de Daleckii–Krein y
    escalares de negocio. ``Phase2_LieDeformationOrienter._phase2_consume_phase1_certificate``
    lo ingiere sin re-tipar \(T\).

    Campos:
        tensor_matrix: \(T\) simetrizado write-protected, shape \((d,d)\).
        purity_factor: \(\|T\|_F^2=\sum_i\sigma_i^2\).
        is_hermitian: residuo de simetría ≤ ε.
        condition_number: \(\kappa_2(T)=\sigma_{\max}/\sigma_{\min}\).
        lipschitz_bound: cota de Daleckii–Krein \(L_{\mathrm{DK}}\).
        initial_roi / initial_psi: escalares de negocio saneados.
        symmetry_residual_relative: \(\|T-T^\top\|_F/\|T\|_F\) pre-Weyl.
        phase1_verdict: subobjeto de Heyting de la observación.
        manifold_dim: \(d\).
        sigma_max / sigma_min: extremos del espectro singular.
        nuclear_norm: \(\|T\|_*=\sum_i\sigma_i\).
        frobenius_norm: \(\|T\|_F\).
        coupling_kappa / coupling_gamma: acoplamientos del tensor.
        is_invertible: \(\sigma_{\min}>\varepsilon_{\mathrm{crit}}\).
        result_declared_valid: bandera ``is_valid`` del motor ciego.
    """

    tensor_matrix: NDArray[np.float64]
    purity_factor: float
    is_hermitian: bool
    condition_number: float
    lipschitz_bound: float
    initial_roi: float
    initial_psi: float
    symmetry_residual_relative: float
    phase1_verdict: ImprobabilityHeytingVerdict
    manifold_dim: int = 0
    sigma_max: float = 0.0
    sigma_min: float = 0.0
    nuclear_norm: float = 0.0
    frobenius_norm: float = 0.0
    coupling_kappa: float = 1.0
    coupling_gamma: float = 0.5
    is_invertible: bool = True
    result_declared_valid: bool = True


@dataclass(frozen=True, slots=True)
class Phase2ImprobabilityOrientation:
    r"""
    Artefacto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Encierra el logaritmo de Lie, los residuos Sp/Liouville y la entropía
    espectral. ``Phase3_HeytingImprobabilityDecider._phase3_consume_phase2_certificate``
    lo ingiere sin recomputar \(\mathrm{Log}\,T\).

    Campos:
        tensor_matrix: \(T\) heredado (write-protected).
        lie_log_matrix: \(\mathrm{Re}\,\mathrm{Log}\,T\) write-protected.
        symplectic_deviation: \(\|T^\top\Omega T-\Omega\|_F\) (NaN si \(d\) impar).
        liouville_determinant_deviation: \(|\det T-1|\).
        is_symplectic: residuo Sp y Liouville bajo tolerancia dura.
        entropy_bits: entropía de Shannon del espectro singular normalizado.
        initial_roi / initial_psi: escalares transportados.
        phase2_verdict: subobjeto de Heyting de la orientación.
        is_even_dimensional: \(d=2n\) (Sp nativo).
        lie_imaginary_residual: \(\|\mathrm{Im}\,\mathrm{Log}\,T\|_F\).
        lie_sp_residual: \(\|\Omega X+X^\top\Omega\|_F\) (NaN si \(d\) impar).
        determinant: \(\det T\) (real).
        condition_number: \(\kappa_2\) heredado de FASE 1.
        logm_branch_stable: logaritmo principal finito y casi real.
        phase1: certificado de FASE 1 anidado.
    """

    tensor_matrix: NDArray[np.float64]
    lie_log_matrix: NDArray[np.float64]
    symplectic_deviation: float
    liouville_determinant_deviation: float
    is_symplectic: bool
    entropy_bits: float
    initial_roi: float
    initial_psi: float
    phase2_verdict: ImprobabilityHeytingVerdict
    is_even_dimensional: bool = False
    lie_imaginary_residual: float = 0.0
    lie_sp_residual: float = float("nan")
    determinant: float = float("nan")
    condition_number: float = 1.0
    logm_branch_stable: bool = True
    phase1: Optional[Phase1ImprobabilityObservation] = None


@dataclass(frozen=True, slots=True)
class Phase3ImprobabilityDecision:
    r"""
    Artefacto terminal de la FASE 3 y objeto inicial del Seal (Ω).

    Certifica el estrés ajustado y la clasificación en \(\Omega_3\) *antes*
    de disparar el Crowbar (una sola actuación, en el agente).

    Campos:
        adjusted_stress: \(\sigma^\star=\|T\|_2\cdot\mathrm{ROI}/\Psi\).
        operator_norm: \(\|T\|_2=\sigma_{\max}\).
        psi_regularized: \(\max(\Psi,\varepsilon_{\mathrm{mach}})\).
        phase3_verdict: subobjeto local de estrés.
        lattice_verdict: supremo \(\bigvee(\chi_1,\chi_2,\chi_3)\).
        orientation: certificado de FASE 2 anidado.
    """

    adjusted_stress: float
    operator_norm: float
    psi_regularized: float
    phase3_verdict: ImprobabilityHeytingVerdict
    lattice_verdict: ImprobabilityHeytingVerdict
    orientation: Phase2ImprobabilityOrientation


@dataclass(frozen=True, slots=True)
class ImprobabilityGovernanceState:
    """Objeto terminal y certificado inmutable emitido por el Agente (Act)."""

    verdict: ImprobabilityHeytingVerdict
    crowbar_report: CrowbarBypassAction
    adjusted_stress: float
    wilkinson_condition: float
    symplectic_deviation: float
    is_epistemologically_valid: bool
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str
    liouville_deviation: float = 0.0
    entropy_bits: float = 0.0
    lipschitz_bound: float = 0.0
    is_symplectic: bool = False
    agent_version: str = __version__
    policy_require_symplectic: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDIA NUMÉRICA Y VALIDACIÓN ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════
class _AdvancedImprobabilityNumericalGuard:
    """Capa de saneamiento y validación para tensores y escalares."""

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
        \varepsilon_{\mathrm{crit}}\bigr).
        """
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            float(_EPS_CRITICAL),
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
            value = value.reshape(()).item()
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
    def _assert_finite_scalar(x: float, name: str) -> None:
        if not np.isfinite(x):
            raise ArithmeticError(f"{name} debe ser un escalar finito.")

    @staticmethod
    def _assert_positive_scalar(x: float, name: str) -> None:
        _AdvancedImprobabilityNumericalGuard._assert_finite_scalar(x, name)
        if x <= 0.0:
            raise ValueError(f"{name} debe ser positivo.")

    @staticmethod
    def _heyting_join(
        *verdicts: ImprobabilityHeytingVerdict,
    ) -> ImprobabilityHeytingVerdict:
        """Supremo del retículo \(\{\mathrm{COHERENT}\prec\mathrm{DEGRADED}\prec\mathrm{VETOED}\)."""
        if not verdicts:
            return ImprobabilityHeytingVerdict.COHERENT
        return ImprobabilityHeytingVerdict(max(int(v) for v in verdicts))


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN ESPECTRAL Y SANEAMIENTO LIPSCHITZ (Observe)
# Objetos: T ∈ Mat_d(ℝ), T=Tᵀ, σ(T), κ₂, L_DK, ROI, Ψ
# Funtores: tipado, Weyl, SVD, Daleckii–Krein
# Terminal: Phase1ImprobabilityObservation → objeto inicial FASE 2
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_ImprobabilityObserver(_AdvancedImprobabilityNumericalGuard):
    r"""
    Fase 1: sanea y audita la regularidad del tensor de improbabilidad.

    Morfismo compuesto:

    \[
    \mathrm{ObserveImprob}
    =\mathrm{ROI/}\Psi\circ L_{\mathrm{DK}}\circ\mathrm{SVD}
    \circ\mathrm{Weyl}\circ\mathrm{Type}\circ\mathrm{Dim}.
    \]

    El certificado ``Phase1ImprobabilityObservation`` es el objeto inicial
    exacto de
    ``Phase2_LieDeformationOrienter._phase2_consume_phase1_certificate``.
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_tensor_dimension(dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de \(d=\dim T\).

        Exige \(d\in\mathbb{Z}_{\ge 1}\) y \(d\le d_{\max}\) (``logm`` es
        \(\mathcal{O}(d^3)\); el régimen Weyl-estable del estrato Omega
        se acota a \(d_{\max}=64\)).
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise TensorDimensionError(
                f"Dimensión del tensor inválida: d={dimension_d}. Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise TensorDimensionError(
                f"Dimensión d={d} excede d_max={_DIM_MAX}. "
                "logm / SVD abandonarían el régimen de auditoría espectral "
                "estable del estrato Omega."
            )
        return d

    # ── FASE 1.2 ──────────────────────────────────────────────────────────
    def _phase1_validate_tensor_typing(
        self,
        t_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        FASE 1.2 — Tipado de \(T\in\mathrm{Mat}_d(\mathbb{R})\), finitud IEEE-754.

        Coacciona a ``float64`` y certifica squareness. No proyecta aún:
        la simetría se audita en 1.3 (un Weyl prematuro ocultaría un bug).
        """
        self._assert_square_matrix(t_matrix, "tensor.matrix")
        d = self._phase1_validate_tensor_dimension(t_matrix.shape[0])
        matrix = np.asarray(t_matrix, dtype=np.float64)
        if matrix.shape != (d, d):
            raise TensorDimensionError(
                f"tensor.matrix shape={matrix.shape} incoherente con d={d}."
            )
        logger.debug(
            "FASE1.2 tensor: d=%d, ‖T‖_F=%.6e",
            d,
            float(la.norm(matrix, ord="fro")),
        )
        return matrix

    # ── FASE 1.3 ──────────────────────────────────────────────────────────
    def _phase1_certify_self_adjointness(
        self,
        t_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        FASE 1.3 — Autoadjunción de \(T\) en norma de Frobenius relativa.

        \[
        \|T-T^\top\|_F
        \le
        \varepsilon_{\mathrm{sym}}\,\|T\|_F.
        \]

        Si el defecto es tolerable se devuelve la proyección de Weyl.
        El residuo *pre-proyección* se reporta.
        """
        sym_residual = float(la.norm(t_matrix - t_matrix.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(t_matrix, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        is_herm = symmetry_rel <= _SYMMETRY_REL_TOL
        if not is_herm:
            raise NonSelfAdjointTensorError(
                f"Ruptura de simetría hermítica. Residuo relativo: {symmetry_rel:.3e}"
            )
        return self._hermitize_weyl(t_matrix), symmetry_rel, is_herm

    def _verify_tensor_symmetry(self, t_matrix: NDArray[np.float64]) -> float:
        """Fachada de compatibilidad: residuo relativo de simetría (FASE 1.3)."""
        _herm, symmetry_rel, _ok = self._phase1_certify_self_adjointness(
            self._phase1_validate_tensor_typing(t_matrix)
        )
        del _herm, _ok
        return symmetry_rel

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    def _phase1_svd_wilkinson_spectrum(
        self,
        t_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, float, float, bool]:
        r"""
        FASE 1.4 — Espectro singular y número de condición de Wilkinson.

        \[
        T=U\Sigma V^\top,
        \qquad
        \kappa_2(T)=\sigma_{\max}/\sigma_{\min}.
        \]

        No se pisa \(\sigma_{\min}\) con \(\varepsilon_{\mathrm{mach}}\)
        *antes* de decidir invertibilidad: un suelo silencioso ocultaría
        singularidades. Si \(\sigma_{\min}\le\varepsilon_{\mathrm{crit}}\)
        el tensor no es invertible y \(\mathrm{Log}\,T\) no existe.
        \(\kappa_2\) se reporta como \(+\infty\) en ese caso y se eleva
        veto de Wilkinson (un tensor singular no es un drive físico).
        """
        try:
            s_vals = np.asarray(la.svdvals(t_matrix), dtype=np.float64)
        except la.LinAlgError as exc:
            raise WilkinsonConditionBreachError(
                "Fallo del análisis espectral SVD en la FPU."
            ) from exc
        if s_vals.size == 0:
            raise WilkinsonConditionBreachError("Espectro singular vacío.")
        sigma_max = float(s_vals[0])
        sigma_min = float(s_vals[-1])
        invertible = sigma_min > float(_EPS_CRITICAL)
        if not invertible:
            raise WilkinsonConditionBreachError(
                f"Tensor singular o cuasi-singular: σ_min={sigma_min:.3e} "
                f"≤ ε_crit={float(_EPS_CRITICAL):.1e}. Log T no está definido."
            )
        cond_number = sigma_max / sigma_min
        if cond_number > _MAX_CONDITION_NUMBER:
            raise WilkinsonConditionBreachError(
                f"Mal-condicionamiento de Wilkinson: cond(T) = {cond_number:.3e} "
                f"> {_MAX_CONDITION_NUMBER:.1e}"
            )
        logger.debug(
            "FASE1.4 SVD: σ_max=%.6e, σ_min=%.6e, κ₂=%.6e, invert=%s",
            sigma_max,
            sigma_min,
            cond_number,
            invertible,
        )
        return s_vals, sigma_max, sigma_min, float(cond_number), invertible

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    def _phase1_daleckii_krein_lipschitz(
        self,
        sigma_min: float,
        coupling_kappa: float,
        ambient_dim: int,
    ) -> float:
        r"""
        FASE 1.5 — Cota de Lipschitz espectral (Daleckii–Krein).

        Para una función escalar \(f\in C^1\) y un operador autoadjunto,
        el teorema de Daleckii–Krein acota el Lipschitz operatorial por
        el máximo de las diferencias divididas de \(f\) sobre
        \(\sigma(T)\times\sigma(T)\). En el drive, el acoplamiento \(\kappa\)
        escala una resolución espectral cuyo peor caso es

        \[
        L_{\mathrm{DK}}
        =\frac{\kappa}{\sigma_{\min}+\varepsilon_W},
        \]

        análogo a \(\|(T)^{-1}\|_2=\sigma_{\min}^{-1}\) (resolvente en 0).
        """
        self._assert_finite_scalar(coupling_kappa, "tensor.kappa")
        if coupling_kappa < 0.0:
            raise ScalarRegularityError(
                f"tensor.kappa debe ser ≥ 0; recibido {coupling_kappa}."
            )
        floor = self._wilkinson_spectral_floor(sigma_min, float(_EPS_CRITICAL), ambient_dim)
        bound = float(coupling_kappa) / (float(sigma_min) + floor)
        if not math.isfinite(bound):
            raise WilkinsonConditionBreachError(
                f"Cota de Daleckii–Krein no finita: L={bound}."
            )
        logger.debug("FASE1.5 L_DK=%.6e (κ=%.6e, σ_min=%.6e)", bound, coupling_kappa, sigma_min)
        return bound

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    def _phase1_spectral_masses(
        self,
        s_vals: NDArray[np.float64],
        t_matrix: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        FASE 1.6 — Masas espectrales y checksum Frobenius.

        \[
        \|T\|_*=\sum_i\sigma_i,
        \qquad
        \|T\|_F=\bigl(\sum_i\sigma_i^2\bigr)^{1/2},
        \qquad
        P=\|T\|_F^2.
        \]

        Se confronta \(\|T\|_F\) vía SVD contra la norma de Frobenius
        directa (checksum de isometría de Hilbert–Schmidt).
        """
        nuclear = float(np.sum(s_vals))
        fro_svd = float(np.sqrt(np.sum(s_vals * s_vals)))
        fro_dir = float(la.norm(t_matrix, ord="fro"))
        scale = max(1.0, fro_dir)
        if abs(fro_svd - fro_dir) > _CHECKSUM_FMA_SLACK * _MACHINE_EPS * scale * max(t_matrix.shape[0], 1):
            raise SpectralChecksumError(
                f"Checksum ‖T‖_F SVD↔directo corrupto: "
                f"|{fro_svd:.6e}−{fro_dir:.6e}|."
            )
        purity = fro_svd * fro_svd
        return nuclear, fro_svd, purity

    # ── FASE 1.7 ──────────────────────────────────────────────────────────
    def _phase1_validate_business_scalars(
        self,
        result: ImprobabilityResult,
    ) -> Tuple[float, float, float, float, bool]:
        r"""
        FASE 1.7 — Saneamiento de escalares de negocio \((\mathrm{ROI},\Psi,\kappa,\gamma)\).

        ROI y Ψ deben ser finitos; Ψ se exige no negativo (denominador de
        estrés). κ, γ se exigen finitos y κ ≥ 0.
        """
        if not isinstance(result, ImprobabilityResult):
            raise TypeError("result debe ser ImprobabilityResult")
        roi = self._as_finite_scalar(result.roi_value, "roi_value")
        psi = self._as_finite_scalar(result.psi_value, "psi_value")
        if psi < 0.0:
            raise ScalarRegularityError(f"psi_value debe ser ≥ 0; recibido {psi}.")
        kappa = self._as_finite_scalar(result.tensor.kappa, "tensor.kappa")
        gamma = self._as_finite_scalar(result.tensor.gamma, "tensor.gamma")
        if kappa < 0.0:
            raise ScalarRegularityError(f"tensor.kappa debe ser ≥ 0; recibido {kappa}.")
        declared = bool(getattr(result, "is_valid", True))
        logger.debug(
            "FASE1.7 escalares: ROI=%.6e, Ψ=%.6e, κ=%.6e, γ=%.6e, valid=%s",
            roi,
            psi,
            kappa,
            gamma,
            declared,
        )
        return roi, psi, kappa, gamma, declared

    # ── FASE 1.8 ──────────────────────────────────────────────────────────
    def _phase1_local_heyting_verdict(
        self,
        condition_number: float,
        result_declared_valid: bool,
    ) -> ImprobabilityHeytingVerdict:
        r"""
        FASE 1.8 — Subobjeto local de observación.

        Hard-gates de κ ya se elevaron como excepción. Aquí sólo queda
        la zona gris \(\kappa_{\mathrm{soft}}<\kappa_2\le\kappa_{\max}\)
        y la bandera ``is_valid`` del motor ciego.
        """
        if not result_declared_valid:
            return ImprobabilityHeytingVerdict.DEGRADED
        if condition_number > _SOFT_CONDITION_NUMBER:
            return ImprobabilityHeytingVerdict.DEGRADED
        return ImprobabilityHeytingVerdict.COHERENT

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
    def observe_improbability_tensor(
        self,
        result: ImprobabilityResult,
    ) -> Phase1ImprobabilityObservation:
        r"""
        FASE 1.Ω — Composición terminal de Observación espectral.

        \[
        \mathrm{ObserveImprob}
        =\chi_1\circ\mathrm{Mass}\circ L_{\mathrm{DK}}\circ\mathrm{SVD}
        \circ\mathrm{Weyl}\circ\mathrm{Type}\circ\mathrm{Scalars}.
        \]

        **Contrato funtorial F1 → F2**: el DTO
        ``Phase1ImprobabilityObservation`` es el objeto inicial exacto de
        ``_phase2_consume_phase1_certificate``. Ningún re-tipado de \(T\)
        ni re-SVD se aplica aguas abajo.

        Raises:
            TypeError, TensorDimensionError, NonSelfAdjointTensorError,
            WilkinsonConditionBreachError, ScalarRegularityError,
            SpectralChecksumError.
        """
        roi, psi, kappa, gamma, declared = self._phase1_validate_business_scalars(result)
        raw = self._phase1_validate_tensor_typing(result.tensor.matrix)
        herm, symmetry_rel, is_herm = self._phase1_certify_self_adjointness(raw)
        s_vals, s_max, s_min, cond, invertible = self._phase1_svd_wilkinson_spectrum(herm)
        lipschitz = self._phase1_daleckii_krein_lipschitz(s_min, kappa, herm.shape[0])
        nuclear, fro, purity = self._phase1_spectral_masses(s_vals, herm)
        verdict = self._phase1_local_heyting_verdict(cond, declared)

        logger.debug(
            "FASE1.Ω observe: d=%d, κ₂=%.3e, L_DK=%.3e, P=%.6e, verdict=%s",
            herm.shape[0],
            cond,
            lipschitz,
            purity,
            verdict.name,
        )
        return Phase1ImprobabilityObservation(
            tensor_matrix=self._freeze_array(herm),
            purity_factor=purity,
            is_hermitian=is_herm,
            condition_number=cond,
            lipschitz_bound=lipschitz,
            initial_roi=roi,
            initial_psi=psi,
            symmetry_residual_relative=symmetry_rel,
            phase1_verdict=verdict,
            manifold_dim=int(herm.shape[0]),
            sigma_max=s_max,
            sigma_min=s_min,
            nuclear_norm=nuclear,
            frobenius_norm=fro,
            coupling_kappa=kappa,
            coupling_gamma=gamma,
            is_invertible=invertible,
            result_declared_valid=declared,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — TRANSPORTE DE LIE E INVARIANZA DE LIOUVILLE (Orient)
# Continuación directa de observe_improbability_tensor (FASE 1.Ω) vía FASE 2.0
# Objetos: Log T, Ω, TᵀΩT, det T, H_Shannon, sp(2n)
# Teorías: GL(d), Sp(2n), Liouville, álgebra de Lie, Shannon
# Terminal: Phase2ImprobabilityOrientation → objeto inicial FASE 3
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_LieDeformationOrienter(Phase1_ImprobabilityObserver):
    r"""
    Fase 2: proyecta la deformación al álgebra de Lie y diagnostica Sp/Liouville.

    Morfismo compuesto:

    \[
    \mathrm{OrientLie}
    =(\mathrm{Entropy},\,\mathfrak{sp},\,\mathrm{Liouville},\,\mathrm{Sp},\,
    \mathrm{Log},\,\Omega)
    \circ\mathrm{Consume}\circ\mathrm{ObserveImprob}^*.
    \]

    El primer morfismo, ``_phase2_consume_phase1_certificate``, *es* la
    continuación estricta de ``observe_improbability_tensor``.
    """

    # ── FASE 2.0 · ingesta funtorial del certificado de FASE 1.Ω ──────────
    def _phase2_consume_phase1_certificate(
        self,
        observation: Phase1ImprobabilityObservation,
    ) -> Tuple[Phase1ImprobabilityObservation, NDArray[np.float64], int]:
        r"""
        FASE 2.0 — Ingesta funtorial del certificado de FASE 1.Ω.

        **Continuación estricta de**
        ``Phase1_ImprobabilityObserver.observe_improbability_tensor``.
        Verifica la coherencia \((T,d,\kappa_2)\) y entrega el objeto de
        trabajo de Orient *sin re-tipado ni re-SVD*.
        """
        if not isinstance(observation, Phase1ImprobabilityObservation):
            raise TypeError("observation debe ser Phase1ImprobabilityObservation")
        t_matrix = np.asarray(observation.tensor_matrix, dtype=np.float64)
        d = int(observation.manifold_dim or t_matrix.shape[0])
        if t_matrix.shape != (d, d):
            raise TensorDimensionError(
                f"FASE2.0: tensor shape={t_matrix.shape} ≠ {(d, d)}."
            )
        if not observation.is_hermitian:
            raise NonSelfAdjointTensorError(
                "FASE2.0: el certificado de FASE 1 no es autoadjunto."
            )
        if not observation.is_invertible:
            raise WilkinsonConditionBreachError(
                "FASE2.0: el certificado de FASE 1 no es invertible; Log T no existe."
            )
        logger.debug(
            "FASE2.0 consume F1: d=%d, κ₂=%.3e, L_DK=%.3e, verdict=%s",
            d,
            observation.condition_number,
            observation.lipschitz_bound,
            observation.phase1_verdict.name,
        )
        return observation, t_matrix, d

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    def _phase2_construct_canonical_omega(
        self,
        dimension_d: int,
    ) -> Tuple[Optional[NDArray[np.float64]], bool]:
        r"""
        FASE 2.1 — Forma simpléctica canónica \(\Omega\) (sólo si \(d=2n\)).

        \[
        \Omega
        =\begin{pmatrix}0&I_n\\-I_n&0\end{pmatrix},
        \qquad
        \Omega^\top=-\Omega,
        \qquad
        \Omega^2=-I_{2n}.
        \]

        Si \(d\) es impar, Sp no es nativo: se devuelve ``(None, False)``
        y **no** se inmerge en dimensión par (eso falsearía el diagnóstico
        sobre el tensor original).
        """
        d = int(dimension_d)
        if d < 2 or d % 2 != 0:
            return None, False
        n = d // 2
        eye_n = np.eye(n, dtype=np.float64)
        zero_n = np.zeros((n, n), dtype=np.float64)
        omega = np.block([[zero_n, eye_n], [-eye_n, zero_n]])
        return omega, True

    def _construct_symplectic_form(self, d: int) -> NDArray[np.float64]:
        r"""
        Fachada de compatibilidad.

        Si \(d\) es impar se eleva ``TensorDimensionError`` en lugar de
        paddear silenciosamente (el pad de v3.1 no es una inmersión
        simpléctica del tensor original).
        """
        omega, even = self._phase2_construct_canonical_omega(d)
        if not even or omega is None:
            raise TensorDimensionError(
                f"Sp(2n) exige dimensión par; recibido d={d}. "
                "No se realiza inmersión silenciosa."
            )
        return omega

    # ── FASE 2.2 ──────────────────────────────────────────────────────────
    def _phase2_principal_lie_logarithm(
        self,
        t_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        FASE 2.2 — Logaritmo principal de Lie \(\mathrm{Log}\,T\).

        Para \(T\in\mathrm{GL}(d)\) sin espectro en \(\mathbb{R}_{\le 0}\),
        existe un único logaritmo principal. Si \(T\succ 0\) (típico del
        motor), \(\mathrm{Log}\,T\) es real y simétrico.

        Se mide \(\|\mathrm{Im}\,\mathrm{Log}\,T\|_F\); un residuo por
        encima de tolerancia delata un corte de rama (autovalores ≤ 0).
        """
        try:
            log_c = np.asarray(la.logm(t_matrix), dtype=np.complex128)
        except (la.LinAlgError, ValueError) as exc:
            raise LieLogarithmError(
                f"Fallo del transporte al álgebra de Lie: {exc}"
            ) from exc
        if not np.all(np.isfinite(log_c)):
            raise LieLogarithmError("Log T contiene NaN/Inf (rama no definida).")
        imag_res = float(la.norm(np.imag(log_c), ord="fro"))
        scale = max(1.0, float(la.norm(np.real(log_c), ord="fro")))
        branch_stable = imag_res <= max(_LOGM_IMAG_TOL, _LOGM_IMAG_TOL * scale)
        if not branch_stable:
            raise LieLogarithmError(
                f"Log T no es real: ‖Im Log T‖_F={imag_res:.3e}. "
                "El espectro corta el semieje real no positivo."
            )
        lie_log = np.real(log_c).astype(np.float64, copy=False)
        logger.debug(
            "FASE2.2 Log: ‖Im‖_F=%.3e, stable=%s, ‖X‖_F=%.6e",
            imag_res,
            branch_stable,
            float(la.norm(lie_log, ord="fro")),
        )
        return lie_log, imag_res, branch_stable

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    def _phase2_symplectic_pullback_residual(
        self,
        t_matrix: NDArray[np.float64],
        omega: Optional[NDArray[np.float64]],
    ) -> float:
        r"""
        FASE 2.3 — Residuo de pullback simpléctico.

        \[
        \delta_{\mathrm{Sp}}
        =\bigl\|T^\top\Omega T-\Omega\bigr\|_F.
        \]

        Devuelve ``NaN`` si Sp no es nativo (\(d\) impar).
        """
        if omega is None:
            return float("nan")
        pulled = t_matrix.T @ omega @ t_matrix
        return float(la.norm(pulled - omega, ord="fro"))

    # ── FASE 2.4 ──────────────────────────────────────────────────────────
    def _phase2_liouville_determinant(
        self,
        t_matrix: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        FASE 2.4 — Residuo de Liouville \(|\det T-1|\).

        Todo \(T\in\mathrm{Sp}(2n)\) cumple \(\det T=+1\) automáticamente.
        Fuera de Sp, \(\det T\neq 0\) (ya garantizado por FASE 1.4) pero
        no necesariamente 1: el residuo es diagnóstico de volumen.
        """
        try:
            det = float(np.real(la.det(t_matrix)))
        except la.LinAlgError:
            return float("nan"), 1.0
        if not math.isfinite(det):
            return det, 1.0
        return det, abs(det - 1.0)

    # ── FASE 2.5 ──────────────────────────────────────────────────────────
    def _phase2_lie_algebra_sp_residual(
        self,
        lie_log: NDArray[np.float64],
        omega: Optional[NDArray[np.float64]],
    ) -> float:
        r"""
        FASE 2.5 — Residuo de álgebra de Lie \(\mathfrak{sp}(2n)\).

        \[
        X\in\mathfrak{sp}(2n)
        \;\Longleftrightarrow\;
        \Omega X+X^\top\Omega=0.
        \]

        Si \(T\in\mathrm{Sp}(2n)\) y \(\mathrm{Log}\) es el principal,
        \(X=\mathrm{Log}\,T\) yace en el álgebra. Devuelve ``NaN`` si
        Sp no es nativo.
        """
        if omega is None:
            return float("nan")
        residual = omega @ lie_log + lie_log.T @ omega
        return float(la.norm(residual, ord="fro"))

    # ── FASE 2.6 ──────────────────────────────────────────────────────────
    def _phase2_shannon_spectral_entropy(
        self,
        t_matrix: NDArray[np.float64],
        s_vals_hint: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""
        FASE 2.6 — Entropía de Shannon del espectro singular normalizado.

        \[
        p_i=\sigma_i\big/\sum_j\sigma_j,
        \qquad
        H=-\sum_i p_i\log_2 p_i
        \in\bigl[0,\log_2 d\bigr].
        \]
        """
        if s_vals_hint is None:
            s_vals = np.asarray(la.svdvals(t_matrix), dtype=np.float64)
        else:
            s_vals = np.asarray(s_vals_hint, dtype=np.float64)
        total = float(np.sum(s_vals))
        if total <= _MACHINE_EPS:
            return 0.0
        probs = s_vals / total
        # −Σ p log2 p = −(log 2)⁻¹ Σ p log p, evitando log2(0).
        entropy = -float(np.sum(probs * np.log(np.clip(probs, _MACHINE_EPS, 1.0)))) * _LOG2E
        return max(0.0, entropy)

    # ── FASE 2.7 ──────────────────────────────────────────────────────────
    def _phase2_crosscheck_sp_liouville(
        self,
        is_even: bool,
        symplectic_deviation: float,
        liouville_deviation: float,
        lie_sp_residual: float,
    ) -> None:
        r"""
        FASE 2.7 — Checksum Sp ⇒ Liouville / \(\mathfrak{sp}\).

        Si \(\delta_{\mathrm{Sp}}\) es infinitesimal, \(\det T\) debe ser
        \(+1\) y \(\mathrm{Log}\,T\in\mathfrak{sp}(2n)\). Un desacuerdo
        delata corrupción de \(\Omega\) o del logaritmo.
        """
        if not is_even or not math.isfinite(symplectic_deviation):
            return
        if symplectic_deviation > _SP_REL_TOL:
            return
        if math.isfinite(liouville_deviation) and liouville_deviation > max(_DET_TOL, 1.0e-8):
            raise SpectralChecksumError(
                "Checksum Sp⇒Liouville corrupto: "
                f"δ_Sp={symplectic_deviation:.3e} pero |det−1|={liouville_deviation:.3e}."
            )
        if math.isfinite(lie_sp_residual) and lie_sp_residual > max(_LIE_SP_TOL, 1.0e-6):
            raise SpectralChecksumError(
                "Checksum Sp⇒sp(2n) corrupto: "
                f"δ_Sp={symplectic_deviation:.3e} pero ‖ΩX+XᵀΩ‖_F={lie_sp_residual:.3e}."
            )

    # ── FASE 2.8 ──────────────────────────────────────────────────────────
    def _phase2_local_heyting_verdict(
        self,
        observation: Phase1ImprobabilityObservation,
        is_even: bool,
        symplectic_deviation: float,
        liouville_deviation: float,
        *,
        require_symplectic: bool,
    ) -> Tuple[bool, ImprobabilityHeytingVerdict]:
        r"""
        FASE 2.8 — Clasificación Sp/Liouville y subobjeto local.

        * \(d\) impar: Sp no aplica; no degrada ni veta por sí solo.
        * \(d\) par, residuos duros: ``is_symplectic=True``.
        * Política ``require_symplectic``: residuos fuera de tolerancia
          dura → VETOED; zona gris → DEGRADED.
        * Sin política: residuos grandes → DEGRADED (nunca VETOED sólo
          por no-Sp: un tensor SPD casi nunca es simpléctico).
        """
        if not is_even:
            is_symp = False
            local = observation.phase1_verdict
            return is_symp, local

        is_symp = (
            math.isfinite(symplectic_deviation)
            and symplectic_deviation <= _SP_REL_TOL
            and math.isfinite(liouville_deviation)
            and liouville_deviation <= _DET_TOL
        )
        components = [observation.phase1_verdict]
        if require_symplectic and not is_symp:
            hard = (
                (math.isfinite(symplectic_deviation) and symplectic_deviation > _SP_SOFT_TOL)
                or (math.isfinite(liouville_deviation) and liouville_deviation > 1.0e-6)
            )
            components.append(
                ImprobabilityHeytingVerdict.VETOED
                if hard
                else ImprobabilityHeytingVerdict.DEGRADED
            )
        elif not is_symp:
            if (
                math.isfinite(symplectic_deviation) and symplectic_deviation > _SP_SOFT_TOL
            ) or (
                math.isfinite(liouville_deviation) and liouville_deviation > 1.0e-6
            ):
                components.append(ImprobabilityHeytingVerdict.DEGRADED)
        return is_symp, self._heyting_join(*components)

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    def orient_lie_deformation(
        self,
        observation: Phase1ImprobabilityObservation,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Phase2ImprobabilityOrientation:
        r"""
        FASE 2.Ω — Composición terminal Orient (Log + Sp + Liouville + H).

        **Continuación funtorial de FASE 1.Ω**: consume
        ``Phase1ImprobabilityObservation`` vía FASE 2.0.

        **Contrato funtorial F2 → F3**: el DTO
        ``Phase2ImprobabilityOrientation`` es el objeto inicial exacto de
        ``_phase3_consume_phase2_certificate``.

        Raises:
            LieLogarithmError: rama de Log no definida (siempre, es de entrada).
            SymplecticConservationError: si ``require_symplectic`` y ``strict``.
            SpectralChecksumError: dualidad Sp ↔ Liouville corrupta.
        """
        observation, t_matrix, d = self._phase2_consume_phase1_certificate(observation)

        omega, is_even = self._phase2_construct_canonical_omega(d)
        lie_log, imag_res, branch_ok = self._phase2_principal_lie_logarithm(t_matrix)
        sp_dev = self._phase2_symplectic_pullback_residual(t_matrix, omega)
        det, liouville = self._phase2_liouville_determinant(t_matrix)
        lie_sp = self._phase2_lie_algebra_sp_residual(lie_log, omega)
        self._phase2_crosscheck_sp_liouville(is_even, sp_dev, liouville, lie_sp)

        s_hint = None
        if observation.sigma_max > 0.0 and observation.nuclear_norm > 0.0:
            # No re-SVD: la entropía puede recalcularse; preferimos SVD fresco
            # sólo si el certificado no transporta el vector singular.
            pass
        entropy = self._phase2_shannon_spectral_entropy(t_matrix, s_hint)

        is_symp, verdict = self._phase2_local_heyting_verdict(
            observation,
            is_even,
            sp_dev,
            liouville,
            require_symplectic=require_symplectic,
        )

        if require_symplectic and not is_symp and strict:
            raise SymplecticConservationError(
                "Política Sp violada: "
                f"δ_Sp={sp_dev:.3e}, |det−1|={liouville:.3e}, d={d}, even={is_even}."
            )

        logger.debug(
            "FASE2.Ω Orient: even=%s, δ_Sp=%.3e, |det−1|=%.3e, H=%.4f, "
            "sp_res=%.3e, verdict=%s",
            is_even,
            sp_dev,
            liouville,
            entropy,
            lie_sp,
            verdict.name,
        )
        return Phase2ImprobabilityOrientation(
            tensor_matrix=self._freeze_array(t_matrix),
            lie_log_matrix=self._freeze_array(lie_log),
            symplectic_deviation=float(sp_dev),
            liouville_determinant_deviation=float(liouville),
            is_symplectic=bool(is_symp),
            entropy_bits=float(entropy),
            initial_roi=float(observation.initial_roi),
            initial_psi=float(observation.initial_psi),
            phase2_verdict=verdict,
            is_even_dimensional=bool(is_even),
            lie_imaginary_residual=float(imag_res),
            lie_sp_residual=float(lie_sp),
            determinant=float(det),
            condition_number=float(observation.condition_number),
            logm_branch_stable=bool(branch_ok),
            phase1=observation,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — EVALUACIÓN DE ESTRÉS Y DECISIÓN HEYTING (Decide)
# Continuación directa de orient_lie_deformation (FASE 2.Ω) vía FASE 3.0
# Objetos: ‖T‖₂, σ* = ‖T‖₂·ROI/Ψ, Ω₃
# Teorías: norma operatorial, regularización de Tikhonov, retículo de Heyting
# Terminal: Phase3ImprobabilityDecision → objeto inicial Seal / Crowbar
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingImprobabilityDecider(Phase2_LieDeformationOrienter):
    r"""
    Fase 3: evalúa el estrés ajustado y clasifica en \(\Omega_3\).

    Morfismo compuesto:

    \[
    \mathrm{DecideStress}
    =(\chi_{\Omega_3},\,\sigma^\star,\,\Psi_{\varepsilon},\,\|T\|_2)
    \circ\mathrm{Consume}\circ\mathrm{OrientLie}^*.
    \]

    El primer morfismo, ``_phase3_consume_phase2_certificate``, *es* la
    continuación estricta de ``orient_lie_deformation``. **No** dispara
    el Crowbar: esa actuación es única y vive en el Seal del agente.
    """

    # ── FASE 3.0 · ingesta funtorial del certificado de FASE 2.Ω ──────────
    def _phase3_consume_phase2_certificate(
        self,
        orientation: Phase2ImprobabilityOrientation,
    ) -> Tuple[Phase2ImprobabilityOrientation, NDArray[np.float64], int]:
        r"""
        FASE 3.0 — Ingesta funtorial del certificado de FASE 2.Ω.

        **Continuación estricta de**
        ``Phase2_LieDeformationOrienter.orient_lie_deformation``.
        Verifica la coherencia de \(T\) y \(\mathrm{Log}\,T\) y entrega
        el objeto de trabajo de Decide *sin recomputar el logaritmo*.
        """
        if not isinstance(orientation, Phase2ImprobabilityOrientation):
            raise TypeError("orientation debe ser Phase2ImprobabilityOrientation")
        t_matrix = np.asarray(orientation.tensor_matrix, dtype=np.float64)
        d = int(t_matrix.shape[0])
        if t_matrix.shape != (d, d):
            raise TensorDimensionError(
                f"FASE3.0: tensor shape={t_matrix.shape} no es cuadrada."
            )
        if orientation.lie_log_matrix.shape != (d, d):
            raise LieLogarithmError(
                "FASE3.0: lie_log_matrix incoherente con tensor_matrix."
            )
        if not orientation.logm_branch_stable:
            raise LieLogarithmError("FASE3.0: rama de Log marcada inestable.")
        logger.debug(
            "FASE3.0 consume F2: d=%d, κ₂=%.3e, δ_Sp=%.3e, verdict=%s",
            d,
            orientation.condition_number,
            orientation.symplectic_deviation,
            orientation.phase2_verdict.name,
        )
        return orientation, t_matrix, d

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    def _phase3_operator_norm(
        self,
        orientation: Phase2ImprobabilityOrientation,
        t_matrix: NDArray[np.float64],
    ) -> float:
        r"""
        FASE 3.1 — Norma operatorial \(\|T\|_2=\sigma_{\max}\).

        Reutiliza \(\sigma_{\max}\) del certificado de FASE 1 si está
        anidado; si no, un SVD fresco (caller que construyó el DTO a mano).
        """
        if orientation.phase1 is not None and orientation.phase1.sigma_max > 0.0:
            return float(orientation.phase1.sigma_max)
        try:
            return float(la.svdvals(t_matrix)[0])
        except la.LinAlgError:
            return float(la.norm(t_matrix, 2))

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    def _phase3_regularize_psi(self, psi: float) -> float:
        r"""
        FASE 3.2 — Regularización de Tikhonov del denominador \(\Psi\).

        \[
        \Psi_{\varepsilon}=\max(\Psi,\varepsilon_{\mathrm{mach}}).
        \]
        """
        self._assert_finite_scalar(psi, "psi")
        if psi < 0.0:
            raise ScalarRegularityError(f"psi debe ser ≥ 0; recibido {psi}.")
        return float(psi) if float(psi) >= _MACHINE_EPS else _MACHINE_EPS

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    def _phase3_adjusted_business_stress(
        self,
        operator_norm: float,
        roi: float,
        psi_regularized: float,
    ) -> float:
        r"""
        FASE 3.3 — Ecuación de estado del estrés ajustado.

        \[
        \sigma^\star
        =\|T\|_2\cdot\frac{\mathrm{ROI}}{\Psi_{\varepsilon}}.
        \]

        Interpreta \(\|T\|_2\) como ganancia de deformación, ROI como
        exposición y \(\Psi\) como colchón de resiliencia. Un \(\Psi\)
        nulo se regularizó en 3.2 para evitar Dirac en el origen.
        """
        self._assert_finite_scalar(roi, "roi")
        if psi_regularized <= 0.0:
            raise ScalarRegularityError("Ψ regularizado no positivo.")
        stress = float(operator_norm) * (float(roi) / float(psi_regularized))
        if not math.isfinite(stress):
            raise BusinessStressBreachError(
                f"Estrés ajustado no finito: ‖T‖₂={operator_norm}, ROI={roi}, Ψ={psi_regularized}."
            )
        logger.debug(
            "FASE3.3 estrés: σ*=%.6e (‖T‖₂=%.6e, ROI=%.6e, Ψ_ε=%.6e)",
            stress,
            operator_norm,
            roi,
            psi_regularized,
        )
        return stress

    # ── FASE 3.4 ──────────────────────────────────────────────────────────
    def _phase3_stress_heyting_verdict(
        self,
        adjusted_stress: float,
    ) -> ImprobabilityHeytingVerdict:
        r"""
        FASE 3.4 — Subobjeto local de estrés.

        \[
        \chi_{\sigma}
        =
        \begin{cases}
        \mathrm{VETOED}   & \sigma^\star\ge\sigma_{\mathrm{high}},\\
        \mathrm{DEGRADED} & \sigma^\star\ge\sigma_{\mathrm{soft}},\\
        \mathrm{COHERENT} & \text{en otro caso}.
        \end{cases}
        \]
        """
        if adjusted_stress >= _STRESS_HIGH_THRESHOLD:
            return ImprobabilityHeytingVerdict.VETOED
        if adjusted_stress >= _STRESS_SOFT_THRESHOLD:
            return ImprobabilityHeytingVerdict.DEGRADED
        return ImprobabilityHeytingVerdict.COHERENT

    # ── FASE 3.5 ──────────────────────────────────────────────────────────
    def _phase3_lattice_conjunction(
        self,
        orientation: Phase2ImprobabilityOrientation,
        stress_verdict: ImprobabilityHeytingVerdict,
        *,
        require_symplectic: bool,
    ) -> ImprobabilityHeytingVerdict:
        r"""
        FASE 3.5 — Supremo del retículo \(\Omega_3\).

        \[
        \chi
        =\chi_1\vee\chi_2\vee\chi_{\sigma}
        \;\bigl(\vee\chi_{\mathrm{Sp}}\bigr)_{\mathrm{opt}}.
        \]

        Wilkinson duro ya se elevó en FASE 1; aquí se re-evalúa la zona
        gris de \(\kappa_2\) por si un caller construyó el DTO a mano.
        """
        components = [orientation.phase2_verdict, stress_verdict]
        if orientation.phase1 is not None:
            components.append(orientation.phase1.phase1_verdict)
        if orientation.condition_number > _MAX_CONDITION_NUMBER:
            components.append(ImprobabilityHeytingVerdict.VETOED)
        elif orientation.condition_number > _SOFT_CONDITION_NUMBER:
            components.append(ImprobabilityHeytingVerdict.DEGRADED)
        if require_symplectic and not orientation.is_symplectic:
            components.append(ImprobabilityHeytingVerdict.VETOED)
        return self._heyting_join(*components)

    # ── FASE 3.Ω · composición terminal Decide ────────────────────────────
    def decidir_gobernanza_improbabilidad(
        self,
        orientation: Phase2ImprobabilityOrientation,
        crowbar: Optional[CrowbarActuator] = None,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Phase3ImprobabilityDecision:
        r"""
        FASE 3.Ω — Composición terminal Decide (estrés + \(\Omega_3\)).

        **Continuación funtorial de FASE 2.Ω**: consume
        ``Phase2ImprobabilityOrientation`` vía FASE 3.0.

        **Contrato funtorial F3 → Seal**: el DTO
        ``Phase3ImprobabilityDecision`` alimenta el sellado de gobernanza
        en ``ImprobabilityDriveAgent.execute_improbability_governance``.

        El argumento ``crowbar`` se acepta por compatibilidad con v3.1
        pero **se ignora**: el Crowbar se dispara una sola vez en el
        Seal del agente (evita doble actuación HARD_SHORT).

        Raises:
            BusinessStressBreachError: si ``strict`` y \(\sigma^\star\) es
                no finito o (bajo política implícita) el lattice es VETOED
                *sólo* cuando el caller pidió ``strict`` y el estrés es
                no finito. El veto de lattice se sella, no se eleva aquí,
                para permitir soft-audit.
        """
        del crowbar  # actuación única en el agente
        orientation, t_matrix, _d = self._phase3_consume_phase2_certificate(orientation)

        op_norm = self._phase3_operator_norm(orientation, t_matrix)
        psi_eps = self._phase3_regularize_psi(orientation.initial_psi)
        stress = self._phase3_adjusted_business_stress(
            op_norm, orientation.initial_roi, psi_eps
        )
        stress_verdict = self._phase3_stress_heyting_verdict(stress)
        lattice = self._phase3_lattice_conjunction(
            orientation, stress_verdict, require_symplectic=require_symplectic
        )

        if strict and lattice == ImprobabilityHeytingVerdict.VETOED:
            if stress_verdict == ImprobabilityHeytingVerdict.VETOED:
                raise BusinessStressBreachError(
                    f"Estrés ajustado σ*={stress:.6e} ≥ {_STRESS_HIGH_THRESHOLD}."
                )

        logger.debug(
            "FASE3.Ω Decide: σ*=%.6e, χ_σ=%s, χ_Ω=%s",
            stress,
            stress_verdict.name,
            lattice.name,
        )
        return Phase3ImprobabilityDecision(
            adjusted_stress=float(stress),
            operator_norm=float(op_norm),
            psi_regularized=float(psi_eps),
            phase3_verdict=stress_verdict,
            lattice_verdict=lattice,
            orientation=orientation,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CONTROL ESPECTRAL DE IMPROBABILIDAD
# Observe (F1) ⟶ Orient (F2) ⟶ Decide (F3) ⟶ Seal / Crowbar
# ═══════════════════════════════════════════════════════════════════════════════
class ImprobabilityDriveAgent(Morphism, Phase3_HeytingImprobabilityDecider):
    r"""
    Agente Soberano que unifica el motor de improbabilidad con \(\Omega_3\).

    Endofuntor de gobernanza:

    \[
    \mathcal{OODA}_{\mathrm{Improb}}
    :
    \mathbf{ImprobabilityResult}
    \longrightarrow
    \mathbf{GovState}(\Omega_3)
    \]

    compuesto como

    \[
    \mathrm{Seal}\circ\mathrm{Decide}\circ\mathrm{Orient}\circ\mathrm{Observe}.
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

    # ── FASE Ω.1 · Crowbar único ──────────────────────────────────────────
    def _phase_omega_apply_crowbar(
        self,
        verdict: ImprobabilityHeytingVerdict,
    ) -> CrowbarBypassAction:
        """Dispara el actuador según el subobjeto de Heyting (una sola vez)."""
        if verdict == ImprobabilityHeytingVerdict.VETOED:
            action = CrowbarBypassAction.HARD_SHORT
            self._crowbar.trigger_crowbar_bypass(action)
            return action
        if verdict == ImprobabilityHeytingVerdict.DEGRADED:
            action = CrowbarBypassAction.WATCHDOG_PULSE
            self._crowbar.trigger_crowbar_bypass(action)
            return action
        return CrowbarBypassAction.NONE

    # ── FASE Ω.2 · sellado ────────────────────────────────────────────────
    def _phase_omega_seal_governance_state(
        self,
        decision: Optional[Phase3ImprobabilityDecision],
        verdict: ImprobabilityHeytingVerdict,
        action_report: CrowbarBypassAction,
        diagnostic_msg: str,
        latency_ms: float,
        *,
        require_symplectic: bool,
        fallback_stress: float = 0.0,
        fallback_cond: float = 1.0,
        fallback_sp: float = 0.0,
    ) -> ImprobabilityGovernanceState:
        """FASE Ω.2 — Sellado del objeto terminal frozen del funtor OODA."""
        timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if decision is not None:
            ori = decision.orientation
            stress = decision.adjusted_stress
            cond = ori.condition_number
            sp_dev = ori.symplectic_deviation
            liouville = ori.liouville_determinant_deviation
            entropy = ori.entropy_bits
            is_symp = ori.is_symplectic
            lipschitz = ori.phase1.lipschitz_bound if ori.phase1 is not None else 0.0
        else:
            stress = fallback_stress
            cond = fallback_cond
            sp_dev = fallback_sp
            liouville = float("inf")
            entropy = 0.0
            is_symp = False
            lipschitz = 0.0

        prov_payload = (
            f"{verdict.name}-{stress:.6e}-{cond:.6e}-"
            f"{sp_dev:.6e}-{latency_ms:.3f}-{__version__}"
        )
        provenance_hash = hashlib.sha256(prov_payload.encode("utf-8")).hexdigest()
        return ImprobabilityGovernanceState(
            verdict=verdict,
            crowbar_report=action_report,
            adjusted_stress=float(stress),
            wilkinson_condition=float(cond),
            symplectic_deviation=float(sp_dev) if math.isfinite(sp_dev) else float("inf"),
            is_epistemologically_valid=(verdict != ImprobabilityHeytingVerdict.VETOED),
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"{diagnostic_msg} Latencia de lazo: {latency_ms:.3f} ms.",
            liouville_deviation=float(liouville) if math.isfinite(liouville) else float("inf"),
            entropy_bits=float(entropy),
            lipschitz_bound=float(lipschitz),
            is_symplectic=bool(is_symp),
            agent_version=__version__,
            policy_require_symplectic=require_symplectic,
        )

    def _build_vetoed_state(
        self,
        note: str,
        t_start_nano: int,
        *,
        require_symplectic: bool = False,
    ) -> ImprobabilityGovernanceState:
        """Estado de veto de emergencia con trazabilidad (fail-secure)."""
        latency_ms = (time.perf_counter_ns() - t_start_nano) / 1.0e6
        action = CrowbarBypassAction.HARD_SHORT
        return self._phase_omega_seal_governance_state(
            None,
            ImprobabilityHeytingVerdict.VETOED,
            action,
            note,
            latency_ms,
            require_symplectic=require_symplectic,
            fallback_stress=math.inf,
            fallback_cond=_MAX_CONDITION_NUMBER * 10.0,
            fallback_sp=math.inf,
        )

    # ── Compositor público OODA ───────────────────────────────────────────
    def execute_improbability_governance(
        self,
        result: ImprobabilityResult,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> ImprobabilityGovernanceState:
        r"""
        Orquesta el ciclo OODA completo sobre el motor de improbabilidad ciego.

        El objeto terminal de cada fase es el objeto inicial de la siguiente:

        .. code-block:: text

            ┌────────────────────────────────────────────────────────────┐
            │ FASE 1  Observe / Spectral                                 │
            │   1.1 validate_tensor_dimension                            │
            │   1.2 validate_tensor_typing                               │
            │   1.3 certify_self_adjointness                             │
            │   1.4 svd_wilkinson_spectrum                               │
            │   1.5 daleckii_krein_lipschitz                             │
            │   1.6 spectral_masses                                      │
            │   1.7 validate_business_scalars                            │
            │   1.8 local_heyting_verdict                                │
            │   1.Ω observe_improbability_tensor  ──► Phase1 ──┐         │
            ├──────────────────────────────────────────────────┼─────────┤
            │ FASE 2  Orient / Lie  ◄──────────────────────────┘         │
            │   2.0 consume_phase1_certificate                           │
            │   2.1 construct_canonical_omega                            │
            │   2.2 principal_lie_logarithm                              │
            │   2.3 symplectic_pullback_residual                         │
            │   2.4 liouville_determinant                                │
            │   2.5 lie_algebra_sp_residual                              │
            │   2.6 shannon_spectral_entropy                             │
            │   2.7 crosscheck_sp_liouville                              │
            │   2.8 local_heyting_verdict                                │
            │   2.Ω orient_lie_deformation  ──► Phase2 ──┐               │
            ├────────────────────────────────────────────┼───────────────┤
            │ FASE 3  Decide / Stress  ◄─────────────────┘               │
            │   3.0 consume_phase2_certificate                           │
            │   3.1 operator_norm                                        │
            │   3.2 regularize_psi                                       │
            │   3.3 adjusted_business_stress                             │
            │   3.4 stress_heyting_verdict                               │
            │   3.5 lattice_conjunction                                  │
            │   3.Ω decidir_gobernanza  ──► Phase3 ──┐                   │
            ├────────────────────────────────────────┼───────────────────┤
            │ SEAL / CROWBAR  ◄──────────────────────┘                   │
            │   Ω.1 apply_crowbar                                        │
            │   Ω.2 seal_governance_state                                │
            └────────────────────────────────────────────────────────────┘

        Args:
            result: ``ImprobabilityResult`` emitido por el motor físico.
            require_symplectic: hard-gate de Sp/Liouville (default False:
                un tensor SPD casi nunca es simpléctico).
            strict: si es verdadero, violaciones Sp (con política) y
                estrés duro se elevan como excepción además de sellarse.

        Returns:
            ImprobabilityGovernanceState: certificado maestro con veredicto.

        Raises:
            CrowbarTriggeredError: si ``raise_on_veto`` y el retículo colapsa.
        """
        t_start_nano = time.perf_counter_ns()
        diagnostic_msg = "Gobernanza Espectral de Improbabilidad Estable."
        decision: Optional[Phase3ImprobabilityDecision] = None
        verdict = ImprobabilityHeytingVerdict.COHERENT
        pending_raise: Optional[BaseException] = None
        crowbar_already = False

        try:
            # ── FASE 1 · Observe ──────────────────────────────────────────
            observation = self.observe_improbability_tensor(result)

            # ── FASE 2 · Orient  (continúa certificado F1.Ω) ──────────────
            orientation = self.orient_lie_deformation(
                observation,
                require_symplectic=require_symplectic,
                strict=strict,
            )

            # ── FASE 3 · Decide  (continúa certificado F2.Ω) ──────────────
            decision = self.decidir_gobernanza_improbabilidad(
                orientation,
                require_symplectic=require_symplectic,
                strict=strict,
            )
            verdict = decision.lattice_verdict
            if verdict == ImprobabilityHeytingVerdict.VETOED:
                diagnostic_msg = (
                    f"VETO POR EXCESO DE ESTRÉS / LATTICE: σ*={decision.adjusted_stress:.6e}, "
                    f"κ₂={orientation.condition_number:.3e}, "
                    f"δ_Sp={orientation.symplectic_deviation:.3e}."
                )
            elif verdict == ImprobabilityHeytingVerdict.DEGRADED:
                diagnostic_msg = (
                    f"Degradación espectral moderada: σ*={decision.adjusted_stress:.6e}, "
                    f"κ₂={orientation.condition_number:.3e}."
                )

        except CrowbarTriggeredError:
            raise
        except (ImprobabilityDriveAgentError, TopologicalInvariantError) as exc:
            verdict = ImprobabilityHeytingVerdict.VETOED
            diagnostic_msg = f"VETO TOPOLÓGICO DE CENSURA: {exc}"
            self._crowbar.trigger_crowbar_bypass(CrowbarBypassAction.HARD_SHORT)
            crowbar_already = True
            if self._raise_on_veto:
                pending_raise = CrowbarTriggeredError(diagnostic_msg)
                pending_raise.__cause__ = exc
        except Exception as exc:  # noqa: BLE001 — fail-secure del lazo de gobernanza
            verdict = ImprobabilityHeytingVerdict.VETOED
            diagnostic_msg = f"ANILQUILACIÓN CUÁNTICA POR INCOHERENCIA DE ENTRADA: {exc}"
            self._crowbar.trigger_crowbar_bypass(CrowbarBypassAction.HARD_SHORT)
            crowbar_already = True
            if self._raise_on_veto:
                pending_raise = CrowbarTriggeredError(diagnostic_msg)
                pending_raise.__cause__ = exc

        if crowbar_already:
            action_report = CrowbarBypassAction.HARD_SHORT
        else:
            action_report = self._phase_omega_apply_crowbar(verdict)

        latency_ms = (time.perf_counter_ns() - t_start_nano) / 1.0e6
        if decision is None and verdict == ImprobabilityHeytingVerdict.VETOED:
            state = self._build_vetoed_state(
                diagnostic_msg,
                t_start_nano,
                require_symplectic=require_symplectic,
            )
        else:
            state = self._phase_omega_seal_governance_state(
                decision,
                verdict,
                action_report,
                diagnostic_msg,
                latency_ms,
                require_symplectic=require_symplectic,
            )

        if pending_raise is not None:
            raise pending_raise
        if self._raise_on_veto and verdict == ImprobabilityHeytingVerdict.VETOED:
            raise CrowbarTriggeredError(diagnostic_msg)
        return state

    # ─────────────────────────────────────────────────────────────────────
    # Fábricas de referencia (calibración / tests del agente)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def identity_result(
        dimension_d: int,
        *,
        roi: float = 1.0,
        psi: float = 1.0,
        kappa: float = 1.0,
        gamma: float = 0.5,
    ) -> ImprobabilityResult:
        r"""Tensor identidad \(T=I_d\) (κ₂=1, Sp nativo si \(d\) par y \(T\in\mathrm{Sp}\) no)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        tensor = ImprobabilityTensor(
            matrix=np.eye(int(dimension_d), dtype=np.float64),
            kappa=float(kappa),
            gamma=float(gamma),
        )
        return ImprobabilityResult(
            tensor=tensor, roi_value=float(roi), psi_value=float(psi), is_valid=True
        )

    @staticmethod
    def spd_gaussian_result(
        dimension_d: int,
        *,
        seed: int = 0,
        roi: float = 1.0,
        psi: float = 1.0,
        kappa: float = 1.0,
    ) -> ImprobabilityResult:
        r"""Tensor SPD aleatorio \(T=AA^\top+\varepsilon I\) (drive típico, no Sp)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(int(dimension_d), int(dimension_d)))
        spd = raw @ raw.T + 1.0e-3 * np.eye(int(dimension_d))
        tensor = ImprobabilityTensor(matrix=spd.astype(np.float64), kappa=float(kappa), gamma=0.5)
        return ImprobabilityResult(
            tensor=tensor, roi_value=float(roi), psi_value=float(psi), is_valid=True
        )

    @staticmethod
    def symplectic_rotation_result(
        n_pairs: int = 1,
        theta: float = 0.3,
        *,
        roi: float = 1.0,
        psi: float = 1.0,
        kappa: float = 1.0,
    ) -> ImprobabilityResult:
        r"""
        Rotación simpléctica canónica en \(\mathrm{Sp}(2n)\):

        \[
        T=\bigoplus_{i=1}^{n}
        \begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
        \quad\text{(en pares }(q_i,p_i)\text{, layout }(q,p)\text{)}.
        \]

        En el layout \(\Omega=\bigl(\begin{smallmatrix}0&I\\-I&0\end{smallmatrix}\bigr)\)
        esto corresponde a un bloque de rotación mixto \(q\leftrightarrow p\).
        Se construye \(T=\exp(\theta J)\) con \(J=\Omega^{-1}\) sobre cada plano.
        """
        if n_pairs < 1:
            raise ValueError(f"n_pairs debe ser ≥ 1; recibido {n_pairs}")
        n = int(n_pairs)
        d = 2 * n
        # Rotación hamiltoniana: q' = q cos θ + p sin θ, p' = −q sin θ + p cos θ
        # es decir T = [[c I, s I], [−s I, c I]] ∈ Sp(2n).
        c = math.cos(theta)
        s = math.sin(theta)
        eye = np.eye(n, dtype=np.float64)
        t_matrix = np.block([[c * eye, s * eye], [-s * eye, c * eye]])
        tensor = ImprobabilityTensor(matrix=t_matrix, kappa=float(kappa), gamma=0.5)
        return ImprobabilityResult(
            tensor=tensor, roi_value=float(roi), psi_value=float(psi), is_valid=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "ImprobabilityHeytingVerdict",
    "CrowbarBypassAction",
    "CrowbarActuator",
    "LoggingCrowbarActuator",
    "ImprobabilityDriveAgentError",
    "NonSelfAdjointTensorError",
    "WilkinsonConditionBreachError",
    "SymplecticConservationError",
    "LieLogarithmError",
    "BusinessStressBreachError",
    "CrowbarTriggeredError",
    "TensorDimensionError",
    "ScalarRegularityError",
    "SpectralChecksumError",
    "Phase1ImprobabilityObservation",
    "Phase2ImprobabilityOrientation",
    "Phase3ImprobabilityDecision",
    "ImprobabilityGovernanceState",
    "Phase1_ImprobabilityObserver",
    "Phase2_LieDeformationOrienter",
    "Phase3_HeytingImprobabilityDecider",
    "ImprobabilityDriveAgent",
    "ImprobabilityDriveError",
    "ImprobabilityTensor",
    "ImprobabilityResult",
    "__version__",
]