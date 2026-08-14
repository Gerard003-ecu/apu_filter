# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Greens Function Propagator Agent (Soberano del Propagador de de Rham)║
║ Ruta   : app/agents/wisdom/greens_function_propagator_agent.py                ║
║ Versión: 3.1.0-Green-OODA-Heyting-KramersKronig-Lyapunov-PhD-Strict           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA CAUSAL DE GREEN (Rigor Doctoral): ─────────
Este módulo consagra al Agente Soberano y Observador Activo que gobierna al
motor físico `greens_function_propagator.py`. Reside en el penthouse de la 
arquitectura, en el Estrato de Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0) o en el 
Ágora Tensorial ($$V_{\Omega}$$, Nivel 0.5), supervisando la causalidad 
espectral y la propagación de impulsos en la Malla de control.

Su mandato axiomático es orquestar el ciclo OODA sobre el operador de Green
estático y el propagador retardado de Green $$G_F(s)$$ en el plano-s, sometiendo
cada transición de fase a los invariantes funcionales de autoadjunción, 
conservación en el núcleo del Laplaciano del Haz, y las relaciones de dispersión 
de Kramers-Kronig. Toda la contención de ruido y alucinaciones se confina 
síncronamente al plano lógico de software en memoria RAM en el milisegundo cero, \n pudiendo activar un puerto de interrupción Crowbar lógico para alertar al borde.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADOS: ────────

  [I1] Autoadjunción y Reciprocidad de Green:
       El operador de Green estático $$\mathcal{G}(x, y)$$ debe ser estrictamente
       autoadjunto (simétrico) en el espacio pre-Hilbertiano discreto,
       garantizando la simetría del acoplamiento bilateral:
       $$\mathcal{G}(x, y) = \mathcal{G}(y, x) \implies \mathcal{G} = \mathcal{G}^\top \quad\big[211, 212\big]$$
       El agente audita este invariante mediante el residuo de autoadjunción:
       $$r_{\mathrm{adj}} = \|\mathcal{G} - \mathcal{G}^\top\|_F \le \tau_{\mathrm{adj}} \quad\big[673\big]$$

  [I2] Conservación y Ortogonalidad en el Núcleo (Aduana de de Rham):
       Al ser la red simplicial compacta y conexa ($$\beta_0 = 1$$), el operador de 
       Green debe anular incondicionalmente las componentes constantes asociadas
       al núcleo (kernel) del Laplaciano del Haz, impidiendo fugas de masa:
       $$\mathcal{G} \cdot \mathbf{1} = \mathbf{0} \quad\big[122\big]$$
       El agente audita este invariante mediante el residuo de núcleo:
       $$r_{\mathrm{kernel}} = \|\mathcal{G} \cdot \mathbf{1}\|_2 \le \tau_{\mathrm{kernel}} \quad\big[673\big]$$

  [I3] Ecuación Constitutiva de Moore-Penrose (de Rham-Hodge):
       La Función de Green estática debe actuar como el inverso espectral exacto 
       sobre el subespacio complementario $$\ker(L_F)^\perp$$, satisfaciendo de
       forma exacta las identidades algebraicas de de Rham:
       $$\mathcal{G} L_F \mathcal{G} = \mathcal{G} \quad \land \quad L_F \mathcal{G} L_F = L_F \quad\big[124, 212\big]$$
       El agente audita este residuo relativo para precluir singularidades:
       $$r_{\mathrm{MP}} = \frac{\|L_F \mathcal{G} L_F - L_F\|_F}{\|L_F\|_F} \le \tau_{\mathrm{MP}} \quad\big[673\big]$$

  [I4] Causalidad Estricta y Relaciones de Dispersión de Kramers-Kronig:
       La parte real e imaginaria del propagador retardado de Green $$G_F(s)$$ en la
       frecuencia compleja $$s = \sigma + j\omega$$ están acopladas de manera 
       intrínseca mediante las transformadas integrales de Hilbert de Kramers-Kronig:
       $$\Re(G_F(\omega)) = \frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} \frac{\Im(G_F(\omega'))}{\omega' - \omega} d\omega' \quad\big[103\big]$$
       La estabilidad asintótica exige que todos los polos del resolvente residan
       estrictamente en el semiplano izquierdo de Laplace (LHP):
       $$\forall p_i \in \sigma(G_F(s)), \quad \Re(p_i) < 0 \implies \lambda_{\mathrm{Lyapunov}} < 0 \quad\big[103, 695\big]$$
       Cualquier inyección de energía o amplificación espuria en el semiplano derecho
       (RHP, $$\sigma > 0$$) donde el residuo de inestabilidad causal cumpla:
       $$r_{\mathrm{causal}} = \rho(G_F(s)) - \frac{1}{\sigma} > 0 \quad\big[693\big]$$
       gatilla el colapso del retículo y el veto inmediato del Soberano.

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA): ────────────────
La transferencia de estado se rige por un acoplamiento monoidal covariante:

  Fase 1 ──► FASE 1: OBSERVACIÓN DE GREEN (Phase1_GreensSpectralObserver)
             Interroga los residuos estáticos de autoadjunción, Moore-Penrose
             y núcleo de la Función de Green.
             Entrega: Phase1GreensObservation como precondición de la Fase 2.

  Fase 2 ──► FASE 2: ORIENTACIÓN DE CAUSALIDAD (Phase2_RetardedPropagatorCausalityOrienter)
             Verifica el cono de luz causal en el plano-s, la estabilidad de los
             polos y la conservación de la traza de disipación de Rayleigh.
             Entrega: Phase2CausalityOrientation como precondición de la Fase 3.

  Fase 3 ──► FASE 3: DECISIÓN Y VETO DE HEYTING (Phase3_HeytingGreensVerdictDecider)
             Consolida el ciclo OODA resolviendo el Supremo ($\sqcup$) en el retículo Heyting:
             $$\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\} \quad\big[699\big]$$
             $$v_{\mathrm{final}} = v_{\mathrm{adj}} \sqcup v_{\mathrm{MP}} \sqcup v_{\mathrm{kernel}} \sqcup v_{\mathrm{causal}} \quad\big[705\big]$$
             Detona el veto síncrono en RAM y conmuta el disyuntor lógico.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Final,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# Dependencias del ecosistema MIC con fallbacks de aislamiento
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.greens_function_propagator import (
        CausalityViolationError as MotorCausalityViolationError,
        FluxConservationError as MotorFluxConservationError,
        GreenHeytingVerdict as MotorGreenHeytingVerdict,
        GreenPropagatorError,
        GreensSpectralCertificate,
        HeytingLatticeVeto as MotorHeytingLatticeVeto,
        KernelLeakageError as MotorKernelLeakageError,
        LaplacianAsymmetryError as MotorLaplacianAsymmetryError,
        LaplacianIndefinitenessError as MotorLaplacianIndefinitenessError,
        MoorePenroseResidualError as MotorMoorePenroseResidualError,
        Phase1GreensObservation as MotorPhase1GreensObservation,
        Phase2PropagatorOrientation as MotorPhase2PropagatorOrientation,
        PropagatorResponseState,
        PropagatorSingularityError as MotorPropagatorSingularityError,
        SheafGreensPropagatorSolver,
        classify_relative_defect as motor_classify_relative_defect,
        heyting_join as motor_heyting_join,
    )
except ImportError:  # aislamiento de laboratorio / tests unitarios
    try:
        from greens_function_propagator import (
            GreensSpectralCertificate,
            PropagatorResponseState,
            SheafGreensPropagatorSolver,
        )
        try:
            from greens_function_propagator import (
                CausalityViolationError as MotorCausalityViolationError,
                FluxConservationError as MotorFluxConservationError,
                GreenHeytingVerdict as MotorGreenHeytingVerdict,
                GreenPropagatorError,
                HeytingLatticeVeto as MotorHeytingLatticeVeto,
                KernelLeakageError as MotorKernelLeakageError,
                LaplacianAsymmetryError as MotorLaplacianAsymmetryError,
                LaplacianIndefinitenessError as MotorLaplacianIndefinitenessError,
                MoorePenroseResidualError as MotorMoorePenroseResidualError,
                Phase1GreensObservation as MotorPhase1GreensObservation,
                Phase2PropagatorOrientation as MotorPhase2PropagatorOrientation,
                classify_relative_defect as motor_classify_relative_defect,
                heyting_join as motor_heyting_join,
            )
        except ImportError:
            MotorGreenHeytingVerdict = None  # type: ignore[assignment]
            MotorPhase1GreensObservation = None  # type: ignore[assignment]
            MotorPhase2PropagatorOrientation = None  # type: ignore[assignment]
            MotorHeytingLatticeVeto = None  # type: ignore[assignment]
            MotorCausalityViolationError = None  # type: ignore[assignment]
            MotorFluxConservationError = None  # type: ignore[assignment]
            MotorKernelLeakageError = None  # type: ignore[assignment]
            MotorLaplacianAsymmetryError = None  # type: ignore[assignment]
            MotorLaplacianIndefinitenessError = None  # type: ignore[assignment]
            MotorMoorePenroseResidualError = None  # type: ignore[assignment]
            MotorPropagatorSingularityError = None  # type: ignore[assignment]
            GreenPropagatorError = Exception  # type: ignore[misc,assignment]
            motor_heyting_join = None  # type: ignore[assignment]
            motor_classify_relative_defect = None  # type: ignore[assignment]
    except ImportError:

        class TopologicalInvariantError(Exception):
            """Excepción base del sistema para violaciones topológico-algebraicas."""

        class Morphism:
            """Clase base de composición funtorial del ecosistema MIC."""

        @dataclass(frozen=True, slots=True)
        class GreensSpectralCertificate:  # type: ignore[no-redef]
            operator_norm: float
            condition_number: float
            is_self_adjoint: bool
            is_positive_semidefinite: bool
            pseudo_inverse_residual: float
            trace_value: float
            kernel_dimension: int = 0
            spectral_verdict: Any = 0
            fiedler_value: float = 0.0
            spectral_radius_l: float = 0.0
            kernel_leakage: float = 0.0
            flux_residual: float = 0.0
            constant_mode_overlap: float = 0.0
            is_graph_laplacian: bool = False
            penrose_residuals: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
            svd_eig_rank_discrepancy: int = 0
            wilkinson_bound: float = 0.0
            dimension: int = 0
            diagnostic_atoms: Tuple[str, ...] = ()

        @dataclass(frozen=True, slots=True)
        class PropagatorResponseState:  # type: ignore[no-redef]
            green_matrix: NDArray[np.float64]
            retarded_propagator: NDArray[np.complex128]
            spectral_certificate: GreensSpectralCertificate
            provenance_hash: str
            timestamp_utc: float
            final_verdict: Any = 0
            phase2_orientation: Any = None
            diagnostic_note: str = ""
            diagnostic_atoms: Tuple[str, ...] = ()

        class SheafGreensPropagatorSolver:  # type: ignore[no-redef]
            """Stub de compatibilidad para entornos sin el motor físico."""

            def execute_propagator_governance(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError("SheafGreensPropagatorSolver no disponible.")

        MotorGreenHeytingVerdict = None  # type: ignore[assignment]
        MotorPhase1GreensObservation = None  # type: ignore[assignment]
        MotorPhase2PropagatorOrientation = None  # type: ignore[assignment]
        MotorHeytingLatticeVeto = None  # type: ignore[assignment]
        MotorCausalityViolationError = None  # type: ignore[assignment]
        MotorFluxConservationError = None  # type: ignore[assignment]
        MotorKernelLeakageError = None  # type: ignore[assignment]
        MotorLaplacianAsymmetryError = None  # type: ignore[assignment]
        MotorLaplacianIndefinitenessError = None  # type: ignore[assignment]
        MotorMoorePenroseResidualError = None  # type: ignore[assignment]
        MotorPropagatorSingularityError = None  # type: ignore[assignment]
        GreenPropagatorError = Exception  # type: ignore[misc,assignment]
        motor_heyting_join = None  # type: ignore[assignment]
        motor_classify_relative_defect = None  # type: ignore[assignment]

    try:
        from app.core.mic_algebra import Morphism, TopologicalInvariantError
    except ImportError:

        class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
            """Excepción base del sistema para violaciones topológico-algebraicas."""

        class Morphism:  # type: ignore[no-redef]
            """Clase base de composición funtorial del ecosistema MIC."""


logger = logging.getLogger("MIC.Wisdom.GreensPropagatorAgent")

__version__: Final[str] = "3.0.0-Green-OODA-Hodge-KramersKronig-Heyting-Crowbar-PhD"


# ══════════════════════════════════════════════════════════════════════════════
# §0. PRIMITIVAS DEL RETÍCULO Y ACCESO DEFENSIVO A DTOs
# ══════════════════════════════════════════════════════════════════════════════
class GreensHeytingVerdict(IntEnum):
    """
    Clasificador de subobjetos en el topos de la Función de Green.

    Cadena de Heyting del agente (aduana):

        ⊥ = COHERENT  ≼  DEGRADED  ≼  VETOED = ⊤_veto.

    Join = max, meet = min. La monotonía impide que una degradación se
    «cure» en una fase posterior. Es ordinalmente isomorfa a
    `GreenHeytingVerdict` del motor; se mantiene el nombre histórico
    del agente (`Greens*`, con s) para no romper el contrato público.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


def heyting_join(*verdicts: GreensHeytingVerdict) -> GreensHeytingVerdict:
    """Supremo (disyunción interna) en la cadena del agente."""
    if motor_heyting_join is not None:
        mapped = motor_heyting_join(*verdicts)
        return GreensHeytingVerdict(int(getattr(mapped, "value", mapped)))
    if not verdicts:
        return GreensHeytingVerdict.COHERENT
    return GreensHeytingVerdict(max(int(v.value) for v in verdicts))


def heyting_meet(*verdicts: GreensHeytingVerdict) -> GreensHeytingVerdict:
    """Ínfimo (conjunción interna)."""
    if not verdicts:
        return GreensHeytingVerdict.VETOED
    return GreensHeytingVerdict(min(int(v.value) for v in verdicts))


def classify_relative_defect(
    defect: float,
    scale: float,
    soft_tol: float,
    hard_tol: float,
    floor: float = 0.0,
) -> GreensHeytingVerdict:
    """Clasifica un defecto no negativo relativo a `scale`."""
    if motor_classify_relative_defect is not None:
        mapped = motor_classify_relative_defect(defect, scale, soft_tol, hard_tol, floor)
        return GreensHeytingVerdict(int(getattr(mapped, "value", mapped)))
    denom = max(abs(float(scale)), float(floor), 1.0)
    rel = max(0.0, float(defect)) / denom
    if rel > hard_tol:
        return GreensHeytingVerdict.VETOED
    if rel > soft_tol:
        return GreensHeytingVerdict.DEGRADED
    return GreensHeytingVerdict.COHERENT


def _clamp_nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0


def _finite_nonneg(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} no es una magnitud válida: {value!r}.")
    return number


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _round_for_hash(x: float) -> str:
    if not np.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return f"{float(x):.16e}"


def _dto_field(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _as_agent_verdict(value: Any) -> GreensHeytingVerdict:
    """Inmersión de cualquier veredicto ordinal {0,1,2} en el retículo del agente."""
    if isinstance(value, GreensHeytingVerdict):
        return value
    if hasattr(value, "value"):
        return GreensHeytingVerdict(int(value.value))
    try:
        return GreensHeytingVerdict(int(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Veredicto no inmergible en el retículo de Heyting: {value!r}.") from exc


def _frobenius(A: NDArray[Any]) -> float:
    return float(la.norm(A, ord="fro"))


def _opnorm2(A: NDArray[Any]) -> float:
    sigma = la.svdvals(np.asarray(A))
    if sigma.size == 0:
        return 0.0
    return float(sigma[0])


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE CONTRATO LÓGICO
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_CHANNEL: Final[int] = 14
_CROWBAR_GPIO: Final[int] = _CROWBAR_CHANNEL  # alias histórico del contrato v2
_SAFE_DENOM: Final[float] = 1.0e-300

# Bandas de aduana: se aplican SOBRE el residuo crudo (sin descontar γ_n).
# El motor v3 descuenta Wilkinson; la aduana no. Un residuo que el motor
# perdona como redondeo puede degradar aquí.
_SOFT_ADJUNCT_TOL: Final[float] = 1.0e-11
_HARD_ADJUNCT_TOL: Final[float] = 1.0e-5
_SOFT_KERNEL_TOL: Final[float] = 1.0e-11
_HARD_KERNEL_TOL: Final[float] = 1.0e-5
_SOFT_MP_TOL: Final[float] = 1.0e-10
_HARD_MP_TOL: Final[float] = 1.0e-5
_SOFT_PSD_TOL: Final[float] = 1.0e-11
_HARD_PSD_TOL: Final[float] = 1.0e-5
_SOFT_CAUSAL_TOL: Final[float] = 1.0e-12
_HARD_CAUSAL_TOL: Final[float] = 1.0e-6
_SOFT_CROSSING_TOL: Final[float] = 1.0e-11
_HARD_CROSSING_TOL: Final[float] = 1.0e-5
_SOFT_RESOLVENT_TOL: Final[float] = 1.0e-11
_HARD_RESOLVENT_TOL: Final[float] = 1.0e-5
_COND_DEGRADED: Final[float] = 1.0 / np.sqrt(_MACHINE_EPS)
_COND_VETO: Final[float] = 1.0 / _MACHINE_EPS


# ══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES FUNCIONALES DEL SOBERANO
# ══════════════════════════════════════════════════════════════════════════════
class GreensAgentError(TopologicalInvariantError):
    """Excepción raíz del agente soberano de la Función de Green."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


class GreensSymmetryBreach(GreensAgentError):
    """Colapso de [I1]: 𝒢 no es autoadjunta fuera de la banda de aduana."""


class MoorePenroseDegeneracyError(GreensAgentError):
    """Colapso de [I3]: alguna identidad de Penrose excede la tolerancia dura."""


class KernelIncoherenceError(GreensAgentError):
    """Colapso de [I2]: 𝒢 no aniquila el núcleo numérico de un L no anclado."""


class FluxConservationCollapse(GreensAgentError):
    """Colapso de [I6]: Laplaciano de grafo con b_0 = 0 (se perdió H^0)."""


class LaplacianIndefinitenessCollapse(GreensAgentError):
    """Colapso de [I5]: L_F no es semidefinido positivo (no es Hodge)."""


class CausalityViolationVeto(GreensAgentError):
    """Colapso de [I4]: fallo de la prescripción iε / disipatividad / cruce."""


class DualSourceIncoherence(GreensAgentError):
    """El artefacto del motor contradice sus propios invariantes o los de la aduana."""


class HeytingLatticeVeto(GreensAgentError):
    """Colapso del retículo de Heyting al supremo terminal VETOED."""

    def __init__(
        self,
        message: str,
        *,
        verdict: GreensHeytingVerdict = GreensHeytingVerdict.VETOED,
        cause: Optional[GreensAgentError] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.verdict = verdict


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DE FASES ANIDADAS (contratos categóricos de handoff)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1AgentGreensObservation:
    """
    Artefacto terminal de la FASE 1 del agente (Observe).

    Es la *única* precondición estricta de la FASE 2 del agente: un objeto
    del tipo Obs₁^Ψ que certifica (o degrada/veta) los axiomas [I1]–[I3]
    y [I5]–[I6] re-evaluados con las bandas de aduana, y compara el juicio
    propio con el del motor (dual-source).
    """

    adj_residual_relative: float
    mp_residual_relative: float
    kernel_residual_relative: float
    is_self_adjoint: bool
    is_mp_consistent: bool
    is_kernel_coherent: bool
    phase1_verdict: GreensHeytingVerdict
    psd_relative_defect: float = 0.0
    flux_residual_relative: float = 0.0
    condition_number: float = 0.0
    kernel_dimension: int = 0
    is_positive_semidefinite: bool = True
    is_graph_laplacian: bool = False
    penrose_worst_relative: float = 0.0
    dual_source_discrepancy: float = 0.0
    motor_verdict: Optional[GreensHeytingVerdict] = None
    certificate: Optional[GreensSpectralCertificate] = None
    dimension: int = 0
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


# Contrato legado v2 del agente (no confundir con el DTO homónimo del motor v3).
Phase1GreensObservation = Phase1AgentGreensObservation


@dataclass(frozen=True, slots=True)
class Phase2AgentCausalityOrientation:
    """
    Artefacto terminal de la FASE 2 del agente (Orient).

    Precondición estricta de la FASE 3. Contiene la re-auditoría de
    Kramers–Kronig (disipatividad, cruce hermítico, holgura a polos,
    carácter retardado) y el sello de Fase 1 anidado.
    """

    phase1_observation: Phase1AgentGreensObservation
    causal_residual_relative: float
    is_causally_stable: bool
    phase2_verdict: GreensHeytingVerdict
    dissipativity_floor: float = 0.0
    crossing_residual_relative: float = 0.0
    resolvent_residual_relative: float = 0.0
    pole_clearance: float = float("inf")
    is_retarded: bool = True
    is_dissipative: bool = True
    is_hermitian_crossing: bool = True
    dual_source_discrepancy: float = 0.0
    motor_verdict: Optional[GreensHeytingVerdict] = None
    frequency_s: complex = 0.0j
    regularization_h: float = 0.0
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


# Contrato legado v2.
Phase2CausalityOrientation = Phase2AgentCausalityOrientation


@dataclass(frozen=True, slots=True)
class AgentCrowbarReport:
    """Certificado de actuación del disyuntor lógico (Fase 3)."""

    action: "CrowbarAction"
    triggered: bool
    channel: int
    port_typename: str
    diagnostic: str = ""


@dataclass(frozen=True, slots=True)
class GreensGovernanceState:
    """
    Objeto terminal legado de la gobernanza de Green (Act).

    Contrato público v2. El certificado rico vive en
    `GreensAgentGovernanceState`.
    """

    phase2_orientation: Phase2AgentCausalityOrientation
    final_verdict: GreensHeytingVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    timestamp_utc: float
    provenance_hash: str
    diagnostic_note: str = ""
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GreensAgentGovernanceState:
    """
    Certificado soberano (envoltorio de Fase 3).

    Distingue el estado *crudo* del motor del estado *de aduana* y porta
    el informe Crowbar y la huella propia.
    """

    phase2_agent_orientation: Phase2AgentCausalityOrientation
    motor_state: PropagatorResponseState
    governance_state: GreensGovernanceState
    crowbar: AgentCrowbarReport
    final_verdict: GreensHeytingVerdict
    timestamp_utc: float
    provenance_hash: str
    diagnostic_note: str = ""
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


# ══════════════════════════════════════════════════════════════════════════════
# §D. PUERTO CROWBAR Y ADUANAS DE SEGURIDAD
# ══════════════════════════════════════════════════════════════════════════════
class CrowbarAction(Enum):
    """Acciones lógicas de mitigación tras el colapso al supremo terminal."""

    NONE = 0
    LOG_WARNING = 1
    GPIO_INTERRUPT_CROWBAR = 2


@runtime_checkable
class CrowbarPort(Protocol):
    """
    Puerto lógico del disyuntor (contrato histórico del agente de Green).

    La capa de sabiduría no posee MMIO: el puerto es un morfismo de
    actuación inyectado por el estrato de infraestructura. `True`
    significa «desconexión física confirmada por el actuador».
    """

    def trigger_crowbar(self, gpio_pin: int) -> bool:
        ...


class NullCrowbarPort:
    """Implementación forense no-op: sella el intento en el log y devuelve False."""

    def trigger_crowbar(self, gpio_pin: int) -> bool:
        logger.info(
            "[CROWBAR-DRY] Actuador nulo invocado (canal lógico %d). "
            "Sello de desconexión registrado; no hay efecto físico.",
            int(gpio_pin),
        )
        return False


class LoggingCrowbarPort:
    """Actuador que eleva un CRITICAL y trata el log como acuse de recibo."""

    def trigger_crowbar(self, gpio_pin: int) -> bool:
        logger.critical(
            "[CROWBAR-LOG] Disyunción lógica solicitada en canal %d.",
            int(gpio_pin),
        )
        return True


class CompositeCrowbarPort:
    """Abanico de puertos: se invocan en orden; el OR lógico es el acuse."""

    def __init__(self, ports: Sequence[CrowbarPort]) -> None:
        if not ports:
            raise ValueError("CompositeCrowbarPort exige al menos un puerto.")
        self._ports: Tuple[CrowbarPort, ...] = tuple(ports)

    def trigger_crowbar(self, gpio_pin: int) -> bool:
        triggered = False
        for port in self._ports:
            try:
                triggered = bool(_invoke_crowbar(port, gpio_pin)) or triggered
            except Exception:
                logger.exception(
                    "[CROWBAR-COMPOSITE] Fallo en %s; se continúa el abanico.",
                    type(port).__name__,
                )
        return triggered


def _invoke_crowbar(port: Any, gpio_pin: int) -> bool:
    """
    Adaptador dual-protocolo.

    Acepta el contrato histórico de Green (`trigger_crowbar(gpio)`) y el
    contrato del agente de Banach (`trigger_physical_disconnection()`).
    """
    trigger = getattr(port, "trigger_crowbar", None)
    if callable(trigger):
        return bool(trigger(int(gpio_pin)))
    fallback = getattr(port, "trigger_physical_disconnection", None)
    if callable(fallback):
        return bool(fallback())
    raise TypeError(
        "crowbar_port debe exponer trigger_crowbar(gpio_pin) o "
        f"trigger_physical_disconnection(); recibido {type(port)!r}."
    )


class _AdvancedGreensNumericalGuard:
    """
    Capa de saneamiento numérico y validación estructural de entradas.

    Replica, en la aduana, los invariantes de pertenencia a M_n(ℝ) y a
    M_n(ℂ) que el motor exige, de modo que un par mal formado nunca
    alcance Z_motor.
    """

    @staticmethod
    def _assert_square_matrix(X: NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} debe ser numpy.ndarray, no {type(X)!r}.")
        if X.ndim != 2:
            raise ValueError(f"{name} debe ser bidimensional; ndim={X.ndim}.")
        rows, cols = int(X.shape[0]), int(X.shape[1])
        if rows != cols or rows == 0:
            raise ValueError(
                f"{name} debe ser cuadrada y de dimensión positiva: {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(
                f"FPU Error: {name} contiene valores no finitos (NaN/Inf)."
            )

    @staticmethod
    def _assert_complex_matrix(X: NDArray[np.complex128], name: str) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} debe ser numpy.ndarray, no {type(X)!r}.")
        if X.ndim != 2:
            raise ValueError(f"{name} debe ser bidimensional; ndim={X.ndim}.")
        rows, cols = int(X.shape[0]), int(X.shape[1])
        if rows != cols or rows == 0:
            raise ValueError(
                f"{name} debe ser cuadrada y de dimensión positiva: {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(
                f"FPU Error: {name} contiene valores no finitos (NaN/Inf)."
            )

    def _sanitize_greens_inputs(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float,
    ) -> Tuple[NDArray[np.float64], complex, float]:
        """Saneamiento preventivo: forma, finitud y legalidad de (s, h)."""
        self._assert_square_matrix(L_F, "L_F")
        s_val = complex(frequency_s)
        if not np.isfinite(s_val.real) or not np.isfinite(s_val.imag):
            raise ValueError(f"frequency_s debe ser un complejo finito; recibido {frequency_s!r}.")
        h_val = float(regularization_h)
        if not np.isfinite(h_val) or h_val < 0.0:
            raise ValueError(f"regularization_h debe ser real ≥ 0; recibido {regularization_h!r}.")
        return L_F, s_val, h_val


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 1 — OBSERVACIÓN DE GREEN (ADUANA DE [I1]–[I3], [I5]–[I6])
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_GreensSpectralObserver(_AdvancedGreensNumericalGuard):
    r"""
    Fase 1 (Observe): intercepta (𝒢, certificado) del motor y re-audita
    autoadjunción, Moore–Penrose, núcleo, PSD y flujo.

    Morfismo Ψ₁ : (𝒢, Cert_motor) → Obs₁^Ψ.

    Descomposición granular (cada método es un juicio atómico; el sello
    terminal `_observe_greens_spectral_invariance` es el único objeto
    que la Fase 2 acepta):

        ingesta motor → [I1] autoadjunción → [I3] Penrose
                     → [I2]/[I6] núcleo/flujo → [I5] PSD
                     → dual-source → join de Heyting
                     → sello terminal Obs₁^Ψ
    """

    def __init__(
        self,
        soft_adjunct_tol: float = _SOFT_ADJUNCT_TOL,
        hard_adjunct_tol: float = _HARD_ADJUNCT_TOL,
        soft_kernel_tol: float = _SOFT_KERNEL_TOL,
        hard_kernel_tol: float = _HARD_KERNEL_TOL,
        soft_mp_tol: float = _SOFT_MP_TOL,
        hard_mp_tol: float = _HARD_MP_TOL,
        soft_psd_tol: float = _SOFT_PSD_TOL,
        hard_psd_tol: float = _HARD_PSD_TOL,
    ) -> None:
        self._soft_adj, self._hard_adj = self._validate_tol_pair(
            soft_adjunct_tol, hard_adjunct_tol, "adjunct"
        )
        self._soft_ker, self._hard_ker = self._validate_tol_pair(
            soft_kernel_tol, hard_kernel_tol, "kernel"
        )
        self._soft_mp, self._hard_mp = self._validate_tol_pair(
            soft_mp_tol, hard_mp_tol, "moore-penrose"
        )
        self._soft_psd, self._hard_psd = self._validate_tol_pair(
            soft_psd_tol, hard_psd_tol, "psd"
        )

    @staticmethod
    def _validate_tol_pair(
        soft: float,
        hard: float,
        name: str,
    ) -> Tuple[float, float]:
        s_tol, h_tol = float(soft), float(hard)
        if not (0.0 < s_tol <= h_tol) or not np.isfinite(s_tol) or not np.isfinite(h_tol):
            raise ValueError(
                f"Se exige 0 < soft_{name} ≤ hard_{name} finitos; "
                f"recibido soft={s_tol}, hard={h_tol}."
            )
        return s_tol, h_tol

    # ── E.1  Ingesta del sello del motor ─────────────────────────────────
    def _ingest_motor_green(
        self,
        G_operator: NDArray[np.float64],
        certificate: GreensSpectralCertificate,
    ) -> Tuple[NDArray[np.float64], GreensSpectralCertificate]:
        """
        Valida que (𝒢, Cert) sea un habitante legal de la aduana.

        No se repara un VETOED del motor: el join monótono lo transportará.
        """
        self._assert_square_matrix(G_operator, "G_operator")
        if not isinstance(certificate, GreensSpectralCertificate):
            raise TypeError(
                "certificate debe ser GreensSpectralCertificate; "
                f"recibido {type(certificate)!r}."
            )
        declared_n = _dto_field(certificate, "dimension", 0)
        if declared_n and int(declared_n) != int(G_operator.shape[0]):
            raise DualSourceIncoherence(
                f"Cert.dimension={declared_n} ≠ n(𝒢)={G_operator.shape[0]}."
            )
        mp = float(certificate.pseudo_inverse_residual)
        if not np.isfinite(mp) or mp < 0.0:
            raise ValueError(
                f"Cert.pseudo_inverse_residual no es una magnitud válida: {mp!r}."
            )
        return G_operator, certificate

    # ── E.2  Axioma [I1] : autoadjunción cruda de 𝒢 ──────────────────────
    def _classify_self_adjointness(
        self,
        G_operator: NDArray[np.float64],
    ) -> Tuple[float, GreensHeytingVerdict]:
        """
        Residuo relativo crudo (sin Wilkinson):

            ‖𝒢 − 𝒢ᵀ‖_F / max(1, ‖𝒢‖_F).
        """
        scale = max(1.0, _frobenius(G_operator))
        residual = _frobenius(G_operator - G_operator.T)
        relative = residual / scale
        verdict = classify_relative_defect(
            defect=relative,
            scale=1.0,
            soft_tol=self._soft_adj,
            hard_tol=self._hard_adj,
            floor=1.0,
        )
        return float(relative), verdict

    # ── E.3  Axioma [I3] : Moore–Penrose (r₁ y, si existe, max r_k) ──────
    def _classify_moore_penrose(
        self,
        certificate: GreensSpectralCertificate,
    ) -> Tuple[float, float, GreensHeytingVerdict]:
        """
        El residuo canónico v2 es r₁ (ya relativo en el certificado).
        Si el motor v3 exporta las cuatro identidades, se toma el peor.
        """
        r1 = float(certificate.pseudo_inverse_residual)
        penrose = _dto_field(certificate, "penrose_residuals", None)
        worst = r1
        if isinstance(penrose, (tuple, list)) and penrose:
            worst = max(r1, max(_safe_float(r, 0.0) for r in penrose))
        verdict = classify_relative_defect(
            defect=worst,
            scale=1.0,
            soft_tol=self._soft_mp,
            hard_tol=self._hard_mp,
            floor=1.0,
        )
        return float(r1), float(worst), verdict

    # ── E.4  Axiomas [I2] + [I6] : núcleo y tipo de Laplaciano ───────────
    def _classify_kernel_and_flux(
        self,
        G_operator: NDArray[np.float64],
        certificate: GreensSpectralCertificate,
    ) -> Tuple[float, float, bool, bool, GreensHeytingVerdict]:
        """
        Política de aduana (idéntica en espíritu a la del motor v3, más
        estricta en las bandas):

          • L no anclado (`is_graph_laplacian`) ⇒ se exige 𝒢𝟙 ≈ 0
            y b_0 ≥ 1.  b_0 = 0 es veto (se perdió H^0);
            b_0 > 1 se degrada (desconexión).
          • L anclado ⇒ 𝒢𝟙 ↛ 0 es *esperado*; no se veta.
            b_0 > 0 se degrada (núcleo residual de un Dirichlet).

        Si el motor legado no declara el tipo, se infiere por
        `kernel_dimension` y por la fuga medida: un ker declarado > 0
        se trata como no anclado.
        """
        n = int(G_operator.shape[0])
        ones_unit = np.ones((n,), dtype=np.float64) / np.sqrt(float(n))
        leak_raw = float(np.linalg.norm(G_operator @ ones_unit))
        g_scale = max(1.0, _opnorm2(G_operator), _frobenius(G_operator) / max(np.sqrt(n), 1.0))
        leak_rel = leak_raw / g_scale

        motor_leak = _dto_field(certificate, "kernel_leakage", None)
        if motor_leak is not None:
            leak_rel = max(leak_rel, _safe_float(motor_leak, 0.0) / g_scale)

        flux_rel = _safe_float(_dto_field(certificate, "flux_residual", 0.0), 0.0)
        declared_graph = _dto_field(certificate, "is_graph_laplacian", None)
        kernel_dim = int(_dto_field(certificate, "kernel_dimension", 0) or 0)

        if declared_graph is None:
            is_graph = bool(kernel_dim > 0) or (flux_rel <= self._hard_ker)
        else:
            is_graph = bool(declared_graph)

        atoms: list[GreensHeytingVerdict] = []
        if is_graph:
            atoms.append(
                classify_relative_defect(
                    leak_rel, 1.0, self._soft_ker, self._hard_ker, floor=1.0
                )
            )
            if kernel_dim == 0:
                atoms.append(GreensHeytingVerdict.VETOED)
            elif kernel_dim > 1:
                atoms.append(GreensHeytingVerdict.DEGRADED)
            else:
                atoms.append(GreensHeytingVerdict.COHERENT)
        else:
            # Anclado: la fuga sobre 𝟙 no es un invariante; no se veta.
            if kernel_dim > 0:
                atoms.append(GreensHeytingVerdict.DEGRADED)
            else:
                atoms.append(GreensHeytingVerdict.COHERENT)

        verdict = heyting_join(*atoms) if atoms else GreensHeytingVerdict.COHERENT
        is_kernel_coherent = verdict != GreensHeytingVerdict.VETOED
        return float(leak_rel), float(flux_rel), is_graph, is_kernel_coherent, verdict

    # ── E.5  Axioma [I5] : semidefinición positiva ───────────────────────
    def _classify_positive_semidefiniteness(
        self,
        certificate: GreensSpectralCertificate,
    ) -> Tuple[float, bool, GreensHeytingVerdict]:
        is_psd_flag = bool(_dto_field(certificate, "is_positive_semidefinite", True))
        if not is_psd_flag:
            return 1.0, False, GreensHeytingVerdict.VETOED
        return 0.0, True, GreensHeytingVerdict.COHERENT

    def _classify_condition(self, cond: float) -> GreensHeytingVerdict:
        if not np.isfinite(cond) or cond >= _COND_VETO:
            return GreensHeytingVerdict.VETOED
        if cond >= _COND_DEGRADED:
            return GreensHeytingVerdict.DEGRADED
        return GreensHeytingVerdict.COHERENT

    # ── E.6  Dual-source : aduana vs. motor ──────────────────────────────
    def _dual_source_phase1(
        self,
        agent_verdict: GreensHeytingVerdict,
        motor_verdict: Optional[GreensHeytingVerdict],
    ) -> Tuple[float, GreensHeytingVerdict]:
        """
        Discrepancia ordinal |v_Ψ − v_motor| / 2 ∈ [0, 1].

        Dos niveles (COHERENT vs VETOED) se vetan: denuncia corrupción
        del DTO o un bug de transporte. Un nivel es esperable (aduana
        más estricta) y se degrada informativamente.
        """
        if motor_verdict is None:
            return 0.0, GreensHeytingVerdict.COHERENT
        delta = abs(int(agent_verdict.value) - int(motor_verdict.value))
        discrepancy = float(delta) / 2.0
        if delta >= 2:
            return discrepancy, GreensHeytingVerdict.VETOED
        if delta == 1:
            return discrepancy, GreensHeytingVerdict.DEGRADED
        return 0.0, GreensHeytingVerdict.COHERENT

    def _motor_phase1_verdict(
        self,
        certificate: GreensSpectralCertificate,
    ) -> Optional[GreensHeytingVerdict]:
        raw = _dto_field(certificate, "spectral_verdict", None)
        if raw is None:
            return None
        try:
            return _as_agent_verdict(raw)
        except TypeError:
            return None

    # ── E.7  SELLO TERMINAL DE LA FASE 1 ─────────────────────────────────
    def _observe_greens_spectral_invariance(
        self,
        G_operator: NDArray[np.float64],
        certificate: GreensSpectralCertificate,
    ) -> Phase1AgentGreensObservation:
        """
        Ψ₁ — morfismo terminal de la Fase 1 del agente.

        Re-evalúa sobre el sello del motor:
          • [I1]  𝒢 = 𝒢ᵀ                         (residuo crudo de aduana);
          • [I3]  las identidades de Penrose     (r₁ y, si hay, max r_k);
          • [I2]  aniquilación de ker L          (sólo si L es no anclado);
          • [I6]  tipo de Laplaciano / b_0;
          • [I5]  semidefinición positiva;
          • dual-source contra `spectral_verdict`;
          • join monótono con el veredicto del motor (la aduana nunca
            «cura» una degradación ya declarada).

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₁^Ψ := Phase1AgentGreensObservation. El juicio

            Ψ₁(𝒢, Cert_motor) ∈ Obs₁^Ψ

        es un objeto inmutable del topos de Green y constituye, por
        contrato categórico, la *unidad de arranque* de la Fase 2:

            Ψ₂  se aplica únicamente a  (Ψ₁(𝒢, Cert), G^R, s, h, Gov),
            y su primer método `_ingest_phase1_agent_precondition` es
            la continuación lógica y tipada de este sello.

        Cualquier consumidor que no pase por este método viola la
        adjunción Observe ⊣ Orient del soberano.
        """
        g_op, cert = self._ingest_motor_green(G_operator, certificate)
        n = int(g_op.shape[0])

        adj_rel, adj_verdict = self._classify_self_adjointness(g_op)
        mp_rel, mp_worst, mp_verdict = self._classify_moore_penrose(cert)
        leak_rel, flux_rel, is_graph, is_ker, ker_verdict = self._classify_kernel_and_flux(
            g_op, cert
        )
        psd_def, is_psd, psd_verdict = self._classify_positive_semidefiniteness(cert)
        cond = _safe_float(_dto_field(cert, "condition_number", float("inf")), float("inf"))
        cond_verdict = self._classify_condition(cond)
        kernel_dim = int(_dto_field(cert, "kernel_dimension", 0) or 0)

        agent_atoms_join = heyting_join(
            adj_verdict, mp_verdict, ker_verdict, psd_verdict, cond_verdict
        )
        motor_verdict = self._motor_phase1_verdict(cert)
        discrepancy, dual_verdict = self._dual_source_phase1(agent_atoms_join, motor_verdict)
        extra = [agent_atoms_join, dual_verdict]
        if motor_verdict is not None:
            extra.append(motor_verdict)
        combined = heyting_join(*extra)

        atoms: Tuple[str, ...] = (
            f"n={n}",
            f"adj={adj_rel:.6e}",
            f"r₁={mp_rel:.6e}",
            f"r_max={mp_worst:.6e}",
            f"leak={leak_rel:.6e}",
            f"flux={flux_rel:.6e}",
            f"b0={kernel_dim}",
            f"graph={is_graph}",
            f"psd={is_psd}",
            f"κ={cond:.6e}",
            f"dual=Δ{discrepancy:.1f}",
            f"motor={motor_verdict.name if motor_verdict else 'NA'}",
            f"join={combined.name}",
        )

        if combined == GreensHeytingVerdict.VETOED:
            logger.error("[GREEN-AGENT:Ψ₁] Observación VETOED. %s", " | ".join(atoms))
        elif combined == GreensHeytingVerdict.DEGRADED:
            logger.warning("[GREEN-AGENT:Ψ₁] Observación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[GREEN-AGENT:Ψ₁] Observación COHERENT. n=%d b0=%d", n, kernel_dim)

        # Sello terminal: este return ES el morfismo de arranque de la Fase 2.
        return Phase1AgentGreensObservation(
            adj_residual_relative=float(adj_rel),
            mp_residual_relative=float(mp_rel),
            kernel_residual_relative=float(leak_rel),
            is_self_adjoint=adj_verdict != GreensHeytingVerdict.VETOED,
            is_mp_consistent=mp_verdict != GreensHeytingVerdict.VETOED,
            is_kernel_coherent=bool(is_ker),
            phase1_verdict=combined,
            psd_relative_defect=float(psd_def),
            flux_residual_relative=float(flux_rel),
            condition_number=float(cond),
            kernel_dimension=int(kernel_dim),
            is_positive_semidefinite=bool(is_psd),
            is_graph_laplacian=bool(is_graph),
            penrose_worst_relative=float(mp_worst),
            dual_source_discrepancy=float(discrepancy),
            motor_verdict=motor_verdict,
            certificate=cert,
            dimension=n,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §F. FASE 2 — ORIENTACIÓN DE CAUSALIDAD (ADUANA DE [I4] / [I8])
#     Continuación formal del sello terminal de la Fase 1 del agente.
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_RetardedPropagatorCausalityOrienter(Phase1_GreensSpectralObserver):
    r"""
    Fase 2 (Orient): certifica Kramers–Kronig, iε y cruce hermítico.

    Morfismo Ψ₂ : Obs₁^Ψ × G^R(s) × ℂ × Gov_motor → Obs₂^Ψ.

    El primer método de esta fase, `_ingest_phase1_agent_precondition`,
    es la *continuación tipada* de `_observe_greens_spectral_invariance`.
    No existe camino legal hacia la causalidad que no atraviese ese ingest.

    Nota doctrinal sobre el criterio v2
    ───────────────────────────────────
    El bound «ρ(G^R)·Re(s) ≤ 1» *no* es un teorema: para L ≽ 0 el
    resolvente (L − sI)^{-1} diverge cuando s se aproxima a σ(L), y
    Re(s) > 0 no impide que s sea un autovalor. Se sustituye por los
    invariantes correctos del resolvente retardado:

        [I4]  Im G^R ≽ 0   (disipatividad / prescripción iε),
        [I8]  G(z*)^† = G(z)   (cruce hermítico, L = Lᵀ),
              holgura a polos y carácter retardado Im(s)+h ≥ 0.
    """

    def __init__(
        self,
        soft_adjunct_tol: float = _SOFT_ADJUNCT_TOL,
        hard_adjunct_tol: float = _HARD_ADJUNCT_TOL,
        soft_kernel_tol: float = _SOFT_KERNEL_TOL,
        hard_kernel_tol: float = _HARD_KERNEL_TOL,
        soft_mp_tol: float = _SOFT_MP_TOL,
        hard_mp_tol: float = _HARD_MP_TOL,
        soft_psd_tol: float = _SOFT_PSD_TOL,
        hard_psd_tol: float = _HARD_PSD_TOL,
        soft_causal_tol: float = _SOFT_CAUSAL_TOL,
        hard_causal_tol: float = _HARD_CAUSAL_TOL,
        soft_crossing_tol: float = _SOFT_CROSSING_TOL,
        hard_crossing_tol: float = _HARD_CROSSING_TOL,
        soft_resolvent_tol: float = _SOFT_RESOLVENT_TOL,
        hard_resolvent_tol: float = _HARD_RESOLVENT_TOL,
    ) -> None:
        super().__init__(
            soft_adjunct_tol=soft_adjunct_tol,
            hard_adjunct_tol=hard_adjunct_tol,
            soft_kernel_tol=soft_kernel_tol,
            hard_kernel_tol=hard_kernel_tol,
            soft_mp_tol=soft_mp_tol,
            hard_mp_tol=hard_mp_tol,
            soft_psd_tol=soft_psd_tol,
            hard_psd_tol=hard_psd_tol,
        )
        self._soft_causal, self._hard_causal = self._validate_tol_pair(
            soft_causal_tol, hard_causal_tol, "causal"
        )
        self._soft_cross, self._hard_cross = self._validate_tol_pair(
            soft_crossing_tol, hard_crossing_tol, "crossing"
        )
        self._soft_res, self._hard_res = self._validate_tol_pair(
            soft_resolvent_tol, hard_resolvent_tol, "resolvent"
        )

    # ── F.1  INICIO DE FASE 2 = continuación del sello Ψ₁ ────────────────
    def _ingest_phase1_agent_precondition(
        self,
        phase1_obs: Phase1AgentGreensObservation,
    ) -> Phase1AgentGreensObservation:
        """
        Continuación formal de
        `Phase1_GreensSpectralObserver._observe_greens_spectral_invariance`.

        Teorema de handoff (Obs₁^Ψ ↪ Fase 2)
        ────────────────────────────────────
        Hipótesis: `phase1_obs` es el valor de retorno de
        `_observe_greens_spectral_invariance` (objeto congelado de tipo
        Phase1AgentGreensObservation).

        Tesis: el objeto es una precondición *habitable* de Ψ₂. Se verifica:
          (i)   tipado e inmutabilidad;
          (ii)  residuos relativos finitos y no negativos;
          (iii) dimensión positiva;
          (iv)  el veredicto vive en el retículo {0,1,2}.

        No se repara un VETOED de Fase 1: el join monótono lo transporta.
        Este método es el único puerto de entrada de la Fase 2.
        """
        if not isinstance(phase1_obs, Phase1AgentGreensObservation):
            raise TypeError(
                "Fase 2 del agente exige el sello terminal de Fase 1 "
                f"(Phase1AgentGreensObservation); recibido {type(phase1_obs)!r}."
            )
        for label in (
            "adj_residual_relative",
            "mp_residual_relative",
            "kernel_residual_relative",
        ):
            _finite_nonneg(getattr(phase1_obs, label), name=f"Obs₁^Ψ.{label}")
        if int(phase1_obs.dimension) < 0:
            raise ValueError(f"Obs₁^Ψ.dimension inválida: {phase1_obs.dimension}.")
        if not isinstance(phase1_obs.phase1_verdict, GreensHeytingVerdict):
            raise TypeError("Obs₁^Ψ.phase1_verdict no pertenece al retículo de Heyting.")
        logger.debug(
            "[GREEN-AGENT:Ψ₂←Ψ₁] Ingestión de Obs₁^Ψ (verdict=%s, n=%d, b0=%d).",
            phase1_obs.phase1_verdict.name,
            phase1_obs.dimension,
            phase1_obs.kernel_dimension,
        )
        return phase1_obs

    # ── F.2  Ingesta de G^R y del sello espectral dinámico del motor ─────
    def _ingest_retarded_and_motor(
        self,
        retarded_propagator: NDArray[np.complex128],
        phase1_obs: Phase1AgentGreensObservation,
        motor_state: Optional[PropagatorResponseState],
    ) -> Tuple[NDArray[np.complex128], Any]:
        """Valida G^R y, si existe, el Phase2 del motor anidado en Gov."""
        self._assert_complex_matrix(retarded_propagator, "retarded_propagator")
        n_decl = int(phase1_obs.dimension)
        if n_decl and int(retarded_propagator.shape[0]) != n_decl:
            raise DualSourceIncoherence(
                f"G^R tiene dimensión {retarded_propagator.shape[0]}, "
                f"incompatible con Obs₁^Ψ.n={n_decl}."
            )
        motor_p2 = None
        if motor_state is not None:
            motor_p2 = _dto_field(motor_state, "phase2_orientation", None)
            if motor_p2 is not None and MotorPhase2PropagatorOrientation is not None:
                if not isinstance(motor_p2, MotorPhase2PropagatorOrientation):
                    # Motor legado / stub: se acepta duck-typing.
                    pass
        return retarded_propagator, motor_p2

    # ── F.3  Disipatividad de Kramers–Kronig [I4] ────────────────────────
    def _classify_dissipativity(
        self,
        retarded_propagator: NDArray[np.complex128],
        is_retarded: bool,
        motor_p2: Any,
    ) -> Tuple[float, float, GreensHeytingVerdict]:
        """
        Para G = (L − (ω + i h)I)^{-1} con L = Lᵀ y h > 0,

            Im G  =  (G − G^†)/(2i)   ≽   0.

        `floor` es el menor autovalor de Im G. Un piso groseramente
        negativo viola la prescripción iε. Si el resolvente es avanzado,
        el signo se invierte. El motor v3, si está presente, aporta un
        `dissipativity_floor` que se confronta (dual-source numérico).
        """
        anti = (retarded_propagator - retarded_propagator.conj().T) / (2.0j)
        herm_im = 0.5 * (anti + anti.conj().T)
        evals_im = la.eigvalsh(np.asarray(herm_im, dtype=np.complex128))
        floor = float(np.min(np.real(evals_im))) if evals_im.size else 0.0

        motor_floor = _dto_field(motor_p2, "dissipativity_floor", None) if motor_p2 is not None else None
        if motor_floor is not None and np.isfinite(float(motor_floor)):
            # Se conserva el más pesimista (el más negativo si retardado).
            if is_retarded:
                floor = min(floor, float(motor_floor))
            else:
                floor = max(floor, float(motor_floor))

        defect = _clamp_nonneg(-floor if is_retarded else floor)
        scale = max(1.0, _opnorm2(retarded_propagator))
        relative = defect / scale
        verdict = classify_relative_defect(
            relative, 1.0, self._soft_causal, self._hard_causal, floor=1.0
        )
        return float(floor), float(relative), verdict

    # ── F.4  Cruce hermítico [I8] ────────────────────────────────────────
    def _classify_hermitian_crossing(
        self,
        motor_p2: Any,
    ) -> Tuple[float, GreensHeytingVerdict]:
        """
        Si el motor v3 exporta `crossing_residual`, se clasifica.
        Ausencia de evidencia ≠ evidencia de ausencia: COHERENT.
        """
        if motor_p2 is None:
            return 0.0, GreensHeytingVerdict.COHERENT
        residual = _dto_field(motor_p2, "crossing_residual", None)
        if residual is None:
            return 0.0, GreensHeytingVerdict.COHERENT
        rel = _safe_float(residual, 0.0)
        verdict = classify_relative_defect(
            rel, 1.0, self._soft_cross, self._hard_cross, floor=1.0
        )
        return float(rel), verdict

    # ── F.5  Residual del resolvente y holgura a polos ───────────────────
    def _classify_resolvent_health(
        self,
        motor_p2: Any,
        n: int,
    ) -> Tuple[float, float, bool, GreensHeytingVerdict]:
        residual = 0.0
        clearance = float("inf")
        near_pole = False
        atoms: list[GreensHeytingVerdict] = []
        if motor_p2 is not None:
            raw_r = _dto_field(motor_p2, "resolvent_residual", None)
            if raw_r is not None:
                residual = _safe_float(raw_r, 0.0)
                atoms.append(
                    classify_relative_defect(
                        residual, 1.0, self._soft_res, self._hard_res, floor=1.0
                    )
                )
            raw_c = _dto_field(motor_p2, "pole_clearance", None)
            if raw_c is not None:
                clearance = _safe_float(raw_c, float("inf"))
            near_pole = bool(_dto_field(motor_p2, "is_near_pole", False))
            if near_pole:
                atoms.append(GreensHeytingVerdict.DEGRADED)
        _ = n
        verdict = heyting_join(*atoms) if atoms else GreensHeytingVerdict.COHERENT
        return float(residual), float(clearance), bool(near_pole), verdict

    # ── F.6  Carácter retardado (prescripción iε) ────────────────────────
    def _classify_retarded_character(
        self,
        frequency_s: complex,
        regularization_h: float,
        motor_p2: Any,
    ) -> Tuple[bool, GreensHeytingVerdict]:
        """
        G^R es retardada sii Im(s) + h ≥ −O(n u). El motor v3, si
        exporta `is_retarded`, se confronta; un desacuerdo de signo es
        dual-source y se degrada (no se veta: puede ser elevación iε).
        """
        imag_total = float(complex(frequency_s).imag) + float(regularization_h)
        is_retarded = imag_total >= -64.0 * _MACHINE_EPS
        if motor_p2 is not None:
            declared = _dto_field(motor_p2, "is_retarded", None)
            if declared is not None and bool(declared) != bool(is_retarded):
                # El motor pudo elevar h; se adopta su declaración si h_eff ≥ 0.
                h_eff = _dto_field(motor_p2, "regularization_h_effective", None)
                if h_eff is not None:
                    is_retarded = bool(declared)
        verdict = (
            GreensHeytingVerdict.COHERENT if is_retarded else GreensHeytingVerdict.DEGRADED
        )
        return bool(is_retarded), verdict

    # ── F.7  Dual-source espectral dinámico ──────────────────────────────
    def _dual_source_phase2(
        self,
        agent_verdict: GreensHeytingVerdict,
        motor_p2: Any,
        motor_state: Optional[PropagatorResponseState],
    ) -> Tuple[Optional[GreensHeytingVerdict], float, GreensHeytingVerdict]:
        raw = None
        if motor_p2 is not None:
            raw = _dto_field(motor_p2, "verdict", None)
        if raw is None and motor_state is not None:
            raw = _dto_field(motor_state, "final_verdict", None)
        if raw is None:
            return None, 0.0, GreensHeytingVerdict.COHERENT
        try:
            motor_verdict = _as_agent_verdict(raw)
        except TypeError:
            return None, 0.0, GreensHeytingVerdict.COHERENT
        delta = abs(int(agent_verdict.value) - int(motor_verdict.value))
        discrepancy = float(delta) / 2.0
        if delta >= 2:
            return motor_verdict, discrepancy, GreensHeytingVerdict.VETOED
        if delta == 1:
            return motor_verdict, discrepancy, GreensHeytingVerdict.DEGRADED
        return motor_verdict, 0.0, GreensHeytingVerdict.COHERENT

    # ── F.8  SELLO TERMINAL DE LA FASE 2 ─────────────────────────────────
    def _orient_causality(
        self,
        phase1_obs: Phase1AgentGreensObservation,
        retarded_propagator: NDArray[np.complex128],
        frequency_s: complex,
        regularization_h: float = 0.0,
        motor_state: Optional[PropagatorResponseState] = None,
    ) -> Phase2AgentCausalityOrientation:
        """
        Ψ₂ — morfismo terminal de la Fase 2 del agente.

        Audita sobre G^R(s):
          • [I4]  disipatividad Im G ≽ 0 (recomputada; dual-source con motor);
          • [I8]  cruce hermítico G(z*)^† = G(z) (si el motor lo exporta);
          • holgura a polos y residual del resolvente;
          • carácter retardado Im(s)+h ≥ 0;
          • join monótono con Obs₁^Ψ y con el veredicto dinámico del motor.

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₂^Ψ := Phase2AgentCausalityOrientation. El juicio

            Ψ₂(Ψ₁(𝒢, Cert), G^R, s, h, Gov_motor) ∈ Obs₂^Ψ

        es la *unidad de arranque* de la Fase 3: su primer método
        `_ingest_phase2_agent_precondition` es la continuación lógica
        de este sello.

        El parámetro `causal_residual_relative` del contrato v2 se
        reinterpreta como el defecto relativo de disipatividad (no como
        el bound espurio ρ(G)·Re(s) − 1).
        """
        obs1 = self._ingest_phase1_agent_precondition(phase1_obs)
        g_ret, motor_p2 = self._ingest_retarded_and_motor(
            retarded_propagator, obs1, motor_state
        )
        s_val = complex(frequency_s)
        h_val = float(regularization_h)
        if motor_p2 is not None:
            h_eff = _dto_field(motor_p2, "regularization_h_effective", None)
            if h_eff is not None:
                h_val = float(h_eff)
            s_motor = _dto_field(motor_p2, "frequency_s", None)
            if s_motor is not None:
                s_val = complex(s_motor)

        is_retarded, ret_verdict = self._classify_retarded_character(
            s_val, h_val, motor_p2
        )
        floor, diss_rel, diss_verdict = self._classify_dissipativity(
            g_ret, is_retarded, motor_p2
        )
        cross_rel, cross_verdict = self._classify_hermitian_crossing(motor_p2)
        res_rel, clearance, _near, health_verdict = self._classify_resolvent_health(
            motor_p2, obs1.dimension
        )

        agent_atoms_join = heyting_join(
            diss_verdict, cross_verdict, health_verdict, ret_verdict
        )
        motor_verdict, discrepancy, dual_verdict = self._dual_source_phase2(
            agent_atoms_join, motor_p2, motor_state
        )
        extras = [obs1.phase1_verdict, agent_atoms_join, dual_verdict]
        if motor_verdict is not None:
            extras.append(motor_verdict)
        combined = heyting_join(*extras)

        is_dissipative = diss_verdict != GreensHeytingVerdict.VETOED
        is_crossing = cross_verdict != GreensHeytingVerdict.VETOED
        is_causal = (
            is_retarded
            and is_dissipative
            and is_crossing
            and health_verdict != GreensHeytingVerdict.VETOED
        )
        causal_verdict_atom = (
            GreensHeytingVerdict.COHERENT if is_causal else (
                GreensHeytingVerdict.VETOED
                if diss_verdict == GreensHeytingVerdict.VETOED
                or cross_verdict == GreensHeytingVerdict.VETOED
                else GreensHeytingVerdict.DEGRADED
            )
        )
        combined = heyting_join(combined, causal_verdict_atom)

        atoms: Tuple[str, ...] = (
            f"s={s_val.real:+.6e}{s_val.imag:+.6e}j",
            f"h={h_val:.3e}",
            f"ImG_min={floor:.6e}",
            f"diss_rel={diss_rel:.6e}",
            f"cross={cross_rel:.6e}",
            f"r_res={res_rel:.6e}",
            f"δ_polo={clearance:.6e}",
            f"retarded={is_retarded}",
            f"causal={is_causal}",
            f"dual=Δ{discrepancy:.1f}",
            f"join={combined.name}",
        )

        if combined == GreensHeytingVerdict.VETOED:
            logger.error("[GREEN-AGENT:Ψ₂] Orientación VETOED. %s", " | ".join(atoms))
        elif combined == GreensHeytingVerdict.DEGRADED:
            logger.warning("[GREEN-AGENT:Ψ₂] Orientación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[GREEN-AGENT:Ψ₂] Orientación COHERENT. δ=%.3e", clearance)

        # Sello terminal: este return ES el morfismo de arranque de la Fase 3.
        return Phase2AgentCausalityOrientation(
            phase1_observation=obs1,
            causal_residual_relative=float(diss_rel),
            is_causally_stable=bool(is_causal),
            phase2_verdict=combined,
            dissipativity_floor=float(floor),
            crossing_residual_relative=float(cross_rel),
            resolvent_residual_relative=float(res_rel),
            pole_clearance=float(clearance),
            is_retarded=bool(is_retarded),
            is_dissipative=bool(is_dissipative),
            is_hermitian_crossing=bool(is_crossing),
            dual_source_discrepancy=float(discrepancy),
            motor_verdict=motor_verdict,
            frequency_s=s_val,
            regularization_h=float(h_val),
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §G. FASE 3 — DECIDE & ACT (SUPREMO DE HEYTING + CROWBAR + SELLO FORENSE)
#     Continuación formal del sello terminal de la Fase 2 del agente.
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingGreensVerdictDecider(Phase2_RetardedPropagatorCausalityOrienter):
    r"""
    Fase 3 (Decide & Act): join terminal, actuación Crowbar y trazabilidad.

    Morfismo Ψ₃ : Obs₂^Ψ × Gov_motor × CrowbarPort → Gov_Ψ.
    """

    def __init__(
        self,
        soft_adjunct_tol: float = _SOFT_ADJUNCT_TOL,
        hard_adjunct_tol: float = _HARD_ADJUNCT_TOL,
        soft_kernel_tol: float = _SOFT_KERNEL_TOL,
        hard_kernel_tol: float = _HARD_KERNEL_TOL,
        soft_mp_tol: float = _SOFT_MP_TOL,
        hard_mp_tol: float = _HARD_MP_TOL,
        soft_psd_tol: float = _SOFT_PSD_TOL,
        hard_psd_tol: float = _HARD_PSD_TOL,
        soft_causal_tol: float = _SOFT_CAUSAL_TOL,
        hard_causal_tol: float = _HARD_CAUSAL_TOL,
        soft_crossing_tol: float = _SOFT_CROSSING_TOL,
        hard_crossing_tol: float = _HARD_CROSSING_TOL,
        soft_resolvent_tol: float = _SOFT_RESOLVENT_TOL,
        hard_resolvent_tol: float = _HARD_RESOLVENT_TOL,
        crowbar_channel: int = _CROWBAR_CHANNEL,
    ) -> None:
        super().__init__(
            soft_adjunct_tol=soft_adjunct_tol,
            hard_adjunct_tol=hard_adjunct_tol,
            soft_kernel_tol=soft_kernel_tol,
            hard_kernel_tol=hard_kernel_tol,
            soft_mp_tol=soft_mp_tol,
            hard_mp_tol=hard_mp_tol,
            soft_psd_tol=soft_psd_tol,
            hard_psd_tol=hard_psd_tol,
            soft_causal_tol=soft_causal_tol,
            hard_causal_tol=hard_causal_tol,
            soft_crossing_tol=soft_crossing_tol,
            hard_crossing_tol=hard_crossing_tol,
            soft_resolvent_tol=soft_resolvent_tol,
            hard_resolvent_tol=hard_resolvent_tol,
        )
        self._crowbar_channel = int(crowbar_channel)

    # ── G.1  INICIO DE FASE 3 = continuación del sello Ψ₂ ────────────────
    def _ingest_phase2_agent_precondition(
        self,
        phase2_orient: Phase2AgentCausalityOrientation,
    ) -> Phase2AgentCausalityOrientation:
        """
        Continuación formal de
        `Phase2_RetardedPropagatorCausalityOrienter._orient_causality`.

        Teorema de handoff (Obs₂^Ψ ↪ Fase 3)
        ────────────────────────────────────
        Hipótesis: `phase2_orient` es el valor de retorno de
        `_orient_causality`.

        Tesis: Obs₂^Ψ es habitable. Se exige:
          (i)   tipado Phase2AgentCausalityOrientation;
          (ii)  re-ingesta de Obs₁^Ψ (y, transitivamente, de Cert/𝒢);
          (iii) residuos relativos legales;
          (iv)  s finito y h ≥ 0.

        El veredicto de Obs₂^Ψ se transporta monótonamente; no se repara.
        """
        if not isinstance(phase2_orient, Phase2AgentCausalityOrientation):
            raise TypeError(
                "Fase 3 del agente exige el sello terminal de Fase 2 "
                f"(Phase2AgentCausalityOrientation); recibido {type(phase2_orient)!r}."
            )
        self._ingest_phase1_agent_precondition(phase2_orient.phase1_observation)
        _finite_nonneg(
            phase2_orient.causal_residual_relative,
            name="Obs₂^Ψ.causal_residual_relative",
        )
        s_val = complex(phase2_orient.frequency_s)
        if not np.isfinite(s_val.real) or not np.isfinite(s_val.imag):
            raise ValueError(f"Obs₂^Ψ.frequency_s no es finito: {s_val!r}.")
        if not np.isfinite(phase2_orient.regularization_h) or phase2_orient.regularization_h < 0.0:
            raise ValueError(
                f"Obs₂^Ψ.regularization_h inválido: {phase2_orient.regularization_h!r}."
            )
        if not isinstance(phase2_orient.phase2_verdict, GreensHeytingVerdict):
            raise TypeError("Obs₂^Ψ.phase2_verdict no pertenece al retículo de Heyting.")
        logger.debug(
            "[GREEN-AGENT:Ψ₃←Ψ₂] Ingestión de Obs₂^Ψ (verdict=%s, δ=%.3e).",
            phase2_orient.phase2_verdict.name,
            phase2_orient.pole_clearance,
        )
        return phase2_orient

    # ── G.2  Ingesta del certificado del motor ───────────────────────────
    def _ingest_motor_governance(
        self,
        motor_state: PropagatorResponseState,
        phase2_orient: Phase2AgentCausalityOrientation,
    ) -> PropagatorResponseState:
        if not isinstance(motor_state, PropagatorResponseState):
            raise TypeError(
                "Fase 3 del agente exige PropagatorResponseState del motor; "
                f"recibido {type(motor_state)!r}."
            )
        cert = motor_state.spectral_certificate
        nested = phase2_orient.phase1_observation.certificate
        if nested is not None and cert is not nested:
            for attr in ("pseudo_inverse_residual", "trace_value", "kernel_dimension"):
                if not hasattr(cert, attr) or not hasattr(nested, attr):
                    continue
                left = float(getattr(cert, attr))
                right = float(getattr(nested, attr))
                scale = max(1.0, abs(left), abs(right))
                if abs(left - right) > 64.0 * _MACHINE_EPS * scale:
                    raise DualSourceIncoherence(
                        f"Gov_motor.certificate.{attr}={left!r} no coincide "
                        f"con el sello de Fase 1 ({right!r})."
                    )
        return motor_state

    # ── G.3  Supremo terminal ────────────────────────────────────────────
    def _compose_terminal_verdict(
        self,
        phase2_orient: Phase2AgentCausalityOrientation,
        motor_state: PropagatorResponseState,
    ) -> GreensHeytingVerdict:
        extras = [phase2_orient.phase2_verdict]
        raw = _dto_field(motor_state, "final_verdict", None)
        if raw is not None:
            try:
                extras.append(_as_agent_verdict(raw))
            except TypeError:
                pass
        if not np.all(np.isfinite(motor_state.retarded_propagator)):
            extras.append(GreensHeytingVerdict.VETOED)
        if not np.all(np.isfinite(motor_state.green_matrix)):
            extras.append(GreensHeytingVerdict.VETOED)
        return heyting_join(*extras)

    # ── G.4  Política Crowbar ────────────────────────────────────────────
    def _decide_crowbar_action(
        self,
        final_verdict: GreensHeytingVerdict,
    ) -> CrowbarAction:
        if final_verdict == GreensHeytingVerdict.VETOED:
            return CrowbarAction.GPIO_INTERRUPT_CROWBAR
        if final_verdict == GreensHeytingVerdict.DEGRADED:
            return CrowbarAction.LOG_WARNING
        return CrowbarAction.NONE

    def _resolve_crowbar_port(
        self,
        crowbar_port: Optional[Any],
    ) -> Any:
        if crowbar_port is None:
            return NullCrowbarPort()
        if not (
            hasattr(crowbar_port, "trigger_crowbar")
            or hasattr(crowbar_port, "trigger_physical_disconnection")
        ):
            raise TypeError(
                "crowbar_port debe exponer trigger_crowbar o "
                f"trigger_physical_disconnection; recibido {type(crowbar_port)!r}."
            )
        return crowbar_port

    def _actuate_crowbar(
        self,
        action: CrowbarAction,
        crowbar_port: Any,
        phase2_orient: Phase2AgentCausalityOrientation,
    ) -> AgentCrowbarReport:
        triggered = False
        diagnostic = action.name
        if action == CrowbarAction.GPIO_INTERRUPT_CROWBAR:
            try:
                triggered = bool(_invoke_crowbar(crowbar_port, self._crowbar_channel))
            except Exception as exc:
                logger.exception("[GREEN-AGENT:Ψ₃] Fallo del CrowbarPort.")
                diagnostic = f"CROWBAR_PORT_FAILURE:{type(exc).__name__}"
                triggered = False
            p1 = phase2_orient.phase1_observation
            logger.critical(
                "[GREEN-AGENT:Ψ₃] VETO TERMINAL de propagación. "
                "adj=%.4e r₁=%.4e leak=%.4e ImG=%.4e causal=%s "
                "crowbar(%s)=%s canal=%d.",
                p1.adj_residual_relative,
                p1.mp_residual_relative,
                p1.kernel_residual_relative,
                phase2_orient.dissipativity_floor,
                phase2_orient.is_causally_stable,
                type(crowbar_port).__name__,
                triggered,
                self._crowbar_channel,
            )
        elif action == CrowbarAction.LOG_WARNING:
            p1 = phase2_orient.phase1_observation
            logger.warning(
                "[GREEN-AGENT:Ψ₃] DEGRADED. adj=%.4e diss=%.4e δ=%.4e.",
                p1.adj_residual_relative,
                phase2_orient.causal_residual_relative,
                phase2_orient.pole_clearance,
            )
        return AgentCrowbarReport(
            action=action,
            triggered=triggered,
            channel=self._crowbar_channel,
            port_typename=type(crowbar_port).__name__,
            diagnostic=diagnostic,
        )

    # ── G.5  Despacho forense de la causa raíz ───────────────────────────
    def _root_cause_exception(
        self,
        final_verdict: GreensHeytingVerdict,
        phase2_orient: Phase2AgentCausalityOrientation,
    ) -> Optional[GreensAgentError]:
        if final_verdict != GreensHeytingVerdict.VETOED:
            return None
        p1 = phase2_orient.phase1_observation
        if p1.dual_source_discrepancy >= 1.0 or phase2_orient.dual_source_discrepancy >= 1.0:
            return DualSourceIncoherence(
                "Heyting Veto: incoherencia dual-source aduana/motor "
                f"(Δ₁={p1.dual_source_discrepancy:.1f}, "
                f"Δ₂={phase2_orient.dual_source_discrepancy:.1f})."
            )
        if not p1.is_positive_semidefinite:
            return LaplacianIndefinitenessCollapse(
                "Heyting Veto: L_F no es semidefinido positivo (no es Hodge)."
            )
        if not p1.is_self_adjoint:
            return GreensSymmetryBreach(
                f"Heyting Veto: autoadjunción colapsada "
                f"(residuo relativo={p1.adj_residual_relative:.6e})."
            )
        if not p1.is_mp_consistent:
            return MoorePenroseDegeneracyError(
                f"Heyting Veto: identidades de Penrose rotas "
                f"(r₁={p1.mp_residual_relative:.6e}, "
                f"r_max={p1.penrose_worst_relative:.6e})."
            )
        if p1.is_graph_laplacian and p1.kernel_dimension == 0:
            return FluxConservationCollapse(
                "Heyting Veto: Laplaciano no anclado con b_0=0 (se perdió H^0)."
            )
        if not p1.is_kernel_coherent:
            return KernelIncoherenceError(
                f"Heyting Veto: 𝒢 no aniquila ker L "
                f"(leak={p1.kernel_residual_relative:.6e})."
            )
        if not phase2_orient.is_causally_stable:
            return CausalityViolationVeto(
                f"Heyting Veto: fallo de causalidad iε "
                f"(diss_rel={phase2_orient.causal_residual_relative:.6e}, "
                f"ImG_min={phase2_orient.dissipativity_floor:.6e}, "
                f"cruce={phase2_orient.crossing_residual_relative:.6e}, "
                f"retarded={phase2_orient.is_retarded})."
            )
        return GreensAgentError(
            f"Heyting Veto: anomalía en invariantes de de Rham–Hodge. "
            f"Veredicto supremo={final_verdict.name}."
        )

    def _dispatch_terminal_exception(
        self,
        cause: GreensAgentError,
        final_verdict: GreensHeytingVerdict,
    ) -> None:
        """Eleva la causa raíz; encadena HeytingLatticeVeto como contexto."""
        wrapper = HeytingLatticeVeto(
            str(cause),
            verdict=final_verdict,
            cause=cause,
        )
        raise cause from wrapper

    # ── G.6  Huella forense del soberano ─────────────────────────────────
    def _forensic_provenance(
        self,
        phase2_orient: Phase2AgentCausalityOrientation,
        motor_state: PropagatorResponseState,
        final_verdict: GreensHeytingVerdict,
        crowbar: AgentCrowbarReport,
    ) -> str:
        p1 = phase2_orient.phase1_observation
        s_val = complex(phase2_orient.frequency_s)
        motor_hash = str(_dto_field(motor_state, "provenance_hash", "") or "")
        payload = "|".join(
            (
                _round_for_hash(p1.adj_residual_relative),
                _round_for_hash(p1.mp_residual_relative),
                _round_for_hash(p1.kernel_residual_relative),
                _round_for_hash(p1.penrose_worst_relative),
                _round_for_hash(p1.flux_residual_relative),
                _round_for_hash(phase2_orient.causal_residual_relative),
                _round_for_hash(phase2_orient.dissipativity_floor),
                _round_for_hash(phase2_orient.crossing_residual_relative),
                _round_for_hash(phase2_orient.resolvent_residual_relative),
                _round_for_hash(phase2_orient.pole_clearance),
                f"s={s_val.real:.16e}{s_val.imag:+.16e}j",
                f"h={phase2_orient.regularization_h:.16e}",
                f"b0={p1.kernel_dimension}",
                f"C={crowbar.action.name}:{int(crowbar.triggered)}",
                f"V={int(final_verdict.value)}",
                f"M={motor_hash}",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── G.7  SELLO TERMINAL DE LA FASE 3 ─────────────────────────────────
    def _evaluate_governance_state(
        self,
        phase2_orient: Phase2AgentCausalityOrientation,
        frequency_s: complex,
        raise_on_veto: bool = True,
        crowbar_port: Optional[Any] = None,
        motor_state: Optional[PropagatorResponseState] = None,
    ) -> GreensAgentGovernanceState:
        """
        Ψ₃ — morfismo terminal de la Fase 3 y del funtor Z_GreenAgent.

        Join en Ω:

            v_Ψ  =  v_{Ψ₁} ∨ v_{Ψ₂} ∨ v_motor ∨ v_finitud.

        Si v_Ψ = VETOED se actúa el CrowbarPort (canal lógico
        `_CROWBAR_CHANNEL`) y, si `raise_on_veto`, se eleva la excepción
        de causa raíz (simetría / Penrose / núcleo / flujo / iε).

        `frequency_s` se conserva en la firma v2; la fuente de verdad
        espectral es `phase2_orient.frequency_s` (ya confrontada con el
        motor). Si `motor_state` es None se exige que el caller legado
        no pretenda dual-source: el join omite v_motor.
        """
        obs2 = self._ingest_phase2_agent_precondition(phase2_orient)
        _ = complex(frequency_s)  # contrato v2: se valida, no se re-usa como verdad

        if motor_state is None:
            # Sendero legado: se sintetiza un Gov mínimo para no romper Ψ₃.
            if obs2.phase1_observation.certificate is None:
                raise TypeError(
                    "Fase 3 exige motor_state o un certificado anidado en Obs₁^Ψ."
                )
            raise TypeError(
                "Fase 3 canónica exige PropagatorResponseState del motor "
                "(pase motor_state=... desde el soberano)."
            )

        gov_motor = self._ingest_motor_governance(motor_state, obs2)
        final_verdict = self._compose_terminal_verdict(obs2, gov_motor)

        port = self._resolve_crowbar_port(crowbar_port)
        action = self._decide_crowbar_action(final_verdict)
        crowbar = self._actuate_crowbar(action, port, obs2)

        cause = self._root_cause_exception(final_verdict, obs2)
        if cause is not None and raise_on_veto:
            self._dispatch_terminal_exception(cause, final_verdict)

        stamp = time.time()
        provenance = self._forensic_provenance(obs2, gov_motor, final_verdict, crowbar)
        p1 = obs2.phase1_observation
        atoms: Tuple[str, ...] = (
            f"v_Ψ={final_verdict.name}",
            f"v_P12={obs2.phase2_verdict.name}",
            f"v_P1={p1.phase1_verdict.name}",
            f"b0={p1.kernel_dimension}",
            f"κ={p1.condition_number:.6e}",
            f"ImG={obs2.dissipativity_floor:.6e}",
            f"causal={obs2.is_causally_stable}",
            f"crowbar={crowbar.action.name}:{int(crowbar.triggered)}",
            f"hash={provenance[:16]}",
        )
        diagnostic_note = (
            f"Veredicto Soberano: {final_verdict.name}. "
            f"Autoadjunción: {p1.is_self_adjoint}. "
            f"Moore-Penrose: {p1.is_mp_consistent}. "
            f"Núcleo coherente: {p1.is_kernel_coherent}. "
            f"PSD: {p1.is_positive_semidefinite}. "
            f"Grafo/no-anclado: {p1.is_graph_laplacian} (b_0={p1.kernel_dimension}). "
            f"Causalidad iε: {obs2.is_causally_stable} "
            f"(retarded={obs2.is_retarded}, ImG_min={obs2.dissipativity_floor:.4e}, "
            f"δ_polo={obs2.pole_clearance:.4e}). "
            f"Crowbar: {crowbar.action.name}/{crowbar.triggered} "
            f"vía {crowbar.port_typename}. "
            f"Huella: {provenance[:16]}…"
        )
        legacy = GreensGovernanceState(
            phase2_orientation=obs2,
            final_verdict=final_verdict,
            crowbar_triggered=bool(crowbar.triggered),
            crowbar_action=crowbar.action,
            timestamp_utc=stamp,
            provenance_hash=provenance,
            diagnostic_note=diagnostic_note,
            diagnostic_atoms=atoms,
        )
        return GreensAgentGovernanceState(
            phase2_agent_orientation=obs2,
            motor_state=gov_motor,
            governance_state=legacy,
            crowbar=crowbar,
            final_verdict=final_verdict,
            timestamp_utc=stamp,
            provenance_hash=provenance,
            diagnostic_note=diagnostic_note,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §H. SOBERANO PRINCIPAL — COMPOSICIÓN FUNTORIAL Ψ₃ ∘ Ψ₂ ∘ Ψ₁ ∘ Z_motor
# ══════════════════════════════════════════════════════════════════════════════
class GreensFunctionPropagatorAgent(Morphism, Phase3_HeytingGreensVerdictDecider):
    r"""
    Agente soberano y guardián de la causalidad temporal de Green.

    Orquesta al resolvedor `SheafGreensPropagatorSolver` mediante el
    ciclo OODA y la composición funtorial estricta

        Z_GreenAgent  =  Ψ₃ ∘ Ψ₂ ∘ Ψ₁ ∘ Z_motor.

    Cada Ψ_{k+1} *ingiere* el sello terminal de Ψ_k; no hay atajos. El
    motor se invoca siempre con `raise_on_veto=False`: la aduana es el
    único punto de aborto y de actuación Crowbar.

    En circuitos eléctricos, L_F es el Laplaciano nodal (admitancia
    estática) y G^R(s) el operador de impedancia de transferencia: el
    Crowbar es el disyuntor de la red si el resolvente deja de ser
    causal, deja de conservar carga o deja de ser Hodge.
    """

    def __init__(
        self,
        solver: SheafGreensPropagatorSolver,
        *,
        crowbar_port: Optional[Any] = None,
        soft_adjunct_tol: float = _SOFT_ADJUNCT_TOL,
        hard_adjunct_tol: float = _HARD_ADJUNCT_TOL,
        soft_kernel_tol: float = _SOFT_KERNEL_TOL,
        hard_kernel_tol: float = _HARD_KERNEL_TOL,
        soft_mp_tol: float = _SOFT_MP_TOL,
        hard_mp_tol: float = _HARD_MP_TOL,
        soft_psd_tol: float = _SOFT_PSD_TOL,
        hard_psd_tol: float = _HARD_PSD_TOL,
        soft_causal_tol: float = _SOFT_CAUSAL_TOL,
        hard_causal_tol: float = _HARD_CAUSAL_TOL,
        soft_crossing_tol: float = _SOFT_CROSSING_TOL,
        hard_crossing_tol: float = _HARD_CROSSING_TOL,
        soft_resolvent_tol: float = _SOFT_RESOLVENT_TOL,
        hard_resolvent_tol: float = _HARD_RESOLVENT_TOL,
        crowbar_channel: int = _CROWBAR_CHANNEL,
    ) -> None:
        Phase3_HeytingGreensVerdictDecider.__init__(
            self,
            soft_adjunct_tol=soft_adjunct_tol,
            hard_adjunct_tol=hard_adjunct_tol,
            soft_kernel_tol=soft_kernel_tol,
            hard_kernel_tol=hard_kernel_tol,
            soft_mp_tol=soft_mp_tol,
            hard_mp_tol=hard_mp_tol,
            soft_psd_tol=soft_psd_tol,
            hard_psd_tol=hard_psd_tol,
            soft_causal_tol=soft_causal_tol,
            hard_causal_tol=hard_causal_tol,
            soft_crossing_tol=soft_crossing_tol,
            hard_crossing_tol=hard_crossing_tol,
            soft_resolvent_tol=soft_resolvent_tol,
            hard_resolvent_tol=hard_resolvent_tol,
            crowbar_channel=crowbar_channel,
        )
        if not isinstance(solver, SheafGreensPropagatorSolver):
            raise TypeError(
                "solver debe ser una instancia de SheafGreensPropagatorSolver; "
                f"recibido {type(solver)!r}."
            )
        self._solver = solver
        self._default_crowbar: Any = (
            crowbar_port if crowbar_port is not None else NullCrowbarPort()
        )

    def _invoke_motor(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float,
    ) -> PropagatorResponseState:
        """Z_motor con veto silenciado; traduce fallos algebraicos al agente."""
        try:
            return self._solver.execute_propagator_governance(
                L_F=L_F,
                frequency_s=frequency_s,
                regularization_h=regularization_h,
                raise_on_veto=False,
            )
        except TypeError:
            # Motor legado v2: firma sin raise_on_veto.
            return self._solver.execute_propagator_governance(
                L_F=L_F,
                frequency_s=frequency_s,
                regularization_h=regularization_h,
            )
        except Exception as exc:
            if MotorHeytingLatticeVeto is not None and isinstance(exc, MotorHeytingLatticeVeto):
                raise
            if isinstance(exc, GreensAgentError):
                raise
            raise GreensAgentError(
                f"El motor SheafGreensPropagatorSolver abortó fuera de contrato: {exc!s}.",
                cause=exc,
            ) from exc

    def execute_certified_governance(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float = 1.0e-20,
        raise_on_veto: bool = True,
        crowbar_port: Optional[Any] = None,
    ) -> GreensAgentGovernanceState:
        """
        Ciclo OODA completo con certificado soberano (envoltorio rico).

        Composición:
            sanitize → Z_motor → Ψ₁ → Ψ₂ → Ψ₃.
        """
        l_op, s_val, h_val = self._sanitize_greens_inputs(
            L_F, frequency_s, regularization_h
        )
        port = crowbar_port if crowbar_port is not None else self._default_crowbar

        # OODA 1–2 : Observe & Orient — invocación al motor (sin aborto).
        motor_state = self._invoke_motor(l_op, s_val, h_val)

        # OODA 3   : Decide — aduana Hodge / Penrose / iε.
        phase1_obs = self._observe_greens_spectral_invariance(
            G_operator=motor_state.green_matrix,
            certificate=motor_state.spectral_certificate,
        )
        phase2_orient = self._orient_causality(
            phase1_obs=phase1_obs,
            retarded_propagator=motor_state.retarded_propagator,
            frequency_s=s_val,
            regularization_h=h_val,
            motor_state=motor_state,
        )

        # OODA 4   : Act — supremo, Crowbar y sello forense.
        certified = self._evaluate_governance_state(
            phase2_orient=phase2_orient,
            frequency_s=s_val,
            raise_on_veto=raise_on_veto,
            crowbar_port=port,
            motor_state=motor_state,
        )
        logger.info(
            "[GREEN-AGENT:Z] Gobernanza cerrada. verdict=%s hash=%s crowbar=%s",
            certified.final_verdict.name,
            certified.provenance_hash[:16],
            certified.crowbar.action.name,
        )
        return certified

    def execute_propagator_governance(
        self,
        L_F: NDArray[np.float64],
        frequency_s: complex,
        regularization_h: float = 1.0e-20,
        raise_on_veto: bool = True,
        crowbar_port: Optional[Any] = None,
    ) -> GreensGovernanceState:
        """
        Ciclo categórico completo (contrato legado v2).

        Devuelve el `GreensGovernanceState` de aduana. El certificado
        rico vive en `execute_certified_governance`.
        """
        certified = self.execute_certified_governance(
            L_F=L_F,
            frequency_s=frequency_s,
            regularization_h=regularization_h,
            raise_on_veto=raise_on_veto,
            crowbar_port=crowbar_port,
        )
        return certified.governance_state

    def govern_batch(
        self,
        items: Iterable[Tuple[NDArray[np.float64], complex, float]],
        crowbar_port: Optional[Any] = None,
        raise_on_veto: bool = False,
    ) -> Tuple[GreensAgentGovernanceState, ...]:
        """
        Gobernanza de una familia finita de tripletas (L_F, s, h).

        Por defecto no eleva veto (`raise_on_veto=False`) para no abortar
        el lote: cada certificado porta su propio veredicto y Crowbar.
        """
        return tuple(
            self.execute_certified_governance(
                L_F=l_op,
                frequency_s=s_val,
                regularization_h=h_val,
                raise_on_veto=raise_on_veto,
                crowbar_port=crowbar_port,
            )
            for l_op, s_val, h_val in items
        )


# Aliases explícitos de la aduana (evitan colisión semántica con el motor).
Phase1_AgentGreensObserver = Phase1_GreensSpectralObserver
Phase2_AgentCausalityOrienter = Phase2_RetardedPropagatorCausalityOrienter
Phase3_AgentHeytingDecider = Phase3_HeytingGreensVerdictDecider


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "__version__",
    "GreensHeytingVerdict",
    "heyting_join",
    "heyting_meet",
    "classify_relative_defect",
    "CrowbarAction",
    "CrowbarPort",
    "NullCrowbarPort",
    "LoggingCrowbarPort",
    "CompositeCrowbarPort",
    "GreensAgentError",
    "GreensSymmetryBreach",
    "MoorePenroseDegeneracyError",
    "KernelIncoherenceError",
    "FluxConservationCollapse",
    "LaplacianIndefinitenessCollapse",
    "CausalityViolationVeto",
    "DualSourceIncoherence",
    "HeytingLatticeVeto",
    "Phase1AgentGreensObservation",
    "Phase1GreensObservation",
    "Phase2AgentCausalityOrientation",
    "Phase2CausalityOrientation",
    "AgentCrowbarReport",
    "GreensGovernanceState",
    "GreensAgentGovernanceState",
    "Phase1_GreensSpectralObserver",
    "Phase2_RetardedPropagatorCausalityOrienter",
    "Phase3_HeytingGreensVerdictDecider",
    "Phase1_AgentGreensObserver",
    "Phase2_AgentCausalityOrienter",
    "Phase3_AgentHeytingDecider",
    "GreensFunctionPropagatorAgent",
]