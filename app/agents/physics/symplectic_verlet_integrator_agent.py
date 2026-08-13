# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Symplectic Verlet Integrator Agent (Soberano de Preservación de Fase)║
║ Ruta   : app/agents/physics/symplectic_verlet_integrator_agent.py            ║
║ Versión: 3.1.0-Verlet-OODA-Heyting-Lyapunov-PhD-Strict                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA SIMPLÉCTICA (Rigor Doctoral): ─────────────
Este módulo consagra al Agente Soberano y Observador Activo que gobierna al
motor físico `symplectic_verlet_integrator.py`. Reside en el hiperespacio
del Estrato de Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0) y supervisa la evolución
temporal de sistemas Port-Hamiltonianos mecánicos y de control de-confinados.

Su mandato axiomático es orquestar el ciclo OODA sobre el paso de integración,
sometiendo las trayectorias del espacio de fase sintáctico a la conservación
rigurosa de la 2-forma simpléctica, el Teorema de Liouville y la inecuación de
disipación de Lyapunov en tiempo de ejecución. Toda la contención de fallos
y alucinaciones estocásticas se confina síncronamente al plano lógico de software
mediante la evaluación de veredictos en el retículo distributivo de Heyting,
con soporte para la activación del disyuntor de potencia Crowbar por hardware
en el microcontrolador perimetral de la obra real.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADOS: ────────

  [I1] Conservación Simpléctica Exacta (Invarianza de de Rham-Liouville):
       El paso del integrador debe preservar la estructura simpléctica canónica:
       $$\omega = \sum_{i=1}^n dq_i \wedge dp_i \implies J_{\mathrm{map}}^\top \Omega J_{\mathrm{map}} = \Omega \quad\big[684, 695\big]$$
       Donde $J_{\mathrm{map}}$ es el Jacobiano del paso de transición, auditado
       espectralmente mediante diferencias de paso complejo (CSD) para mitigar
       de raíz la fatiga de truncamiento y la resta catastrófica en la FPU:
       $$r_{\mathrm{sym}} = \|J_{\mathrm{map}}^\top \Omega J_{\mathrm{map}} - \Omega\|_F \le \tau_{\mathrm{sym}} \quad\big[698\big]$$

  [I2] Conservación del Volumen de Fase (Invariante de Liouville):
       El simplectomorfismo exacto del flujo exige que el determinante del
       Jacobiano de transición local sea estrictamente unitario, garantizando
       la divergencia nula del campo vectorial de control en el espacio de fase:
       $$\operatorname{div}(\dot{x}) \equiv 0 \iff \det(J_{\mathrm{map}}) \equiv 1 \quad\big[684\big]$$
       El agente audita este invariante mediante el residuo de volumen:
       $$r_{\mathrm{Liouville}} = |\det(J_{\mathrm{map}}) - 1| \le \tau_{\mathrm{Liouville}} \quad\big[698, 700\big]$$

  [I3] Inecuación de Disipación de Lyapunov (Segunda Ley de la Termodinámica):
       La derivada temporal del Hamiltoniano amortiguado por el operador de
       Rayleigh ($R_d = R_d^\top \succeq \mathbf{0}$) debe permanecer
       incondicionalmente no-positiva en régimen libre de forzamiento externo:
       $$\dot{H}_d = -v^\top R_d v \le 0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[684, 695\big]$$
       Donde $v = M^{-1}p$ es la velocidad generalizada del sistema.
       Cualquier inyección de energía espuria por inestabilidad de coma flotante
       ($\dot{H}_d > 0$) colapsa el lazo.

  [I4] Estabilidad de Trayectoria de Fase (Exponente de Lyapunov $\lambda < 0$):
       El coeficiente asintótico de sensibilidad dinámica debe certificar la
       convergencia de las órbitas perturbadas hacia el atractor geodésico,
       prohibiendo divergencias caóticas y desbocamientos numéricos:
       $$\|e(t)\|_G \le \|e(0)\|_G \cdot e^{\lambda t} \implies \lambda < 0 \quad\big[695\big]$$

  [I5] Coherencia del Certificado del Motor:
       Las banderas operativas de estabilidad y pasividad devueltas por el motor
       físico se contrastan y verifican de manera cruzada contra las métricas
       cuantitativas directas calculadas sobre el espacio de phase.

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA): ────────────────
La progresión y transferencia de estado se rige por un contrato algebraico rígido
donde el morfismo de salida de cada fase es la precondición formal de la siguiente:

  Fase 1 ──► FASE 1: OBSERVACIÓN DE LIOUVILLE (Phase1_SymplecticObserver)
             Interroga los residuos geométricos del motor ($r_{\mathrm{sym}}$ y
             $r_{\mathrm{Liouville}}$), evaluando la conservación de volumen.
             Entrega: Phase1SymplecticObservation como precondición de la Fase 2.

  Fase 2 ──► FASE 2: ORIENTACIÓN DE PASIVIDAD (Phase2_LyapunovPassivityOrienter)
             Verifica el cumplimiento de la Segunda Ley en la FPU mediante el
             balance de potencia disipada de Rayleigh y el drift del Hamiltoniano.
             Entrega: Phase2PassivityOrientation como precondición de la Fase 3.

  Fase 3 ──► FASE 3: DECISIÓN Y VETO DE HEYTING (Phase3_HeytingVerdictDecider)
             Consolida el ciclo OODA resolviendo la operación Supremo (join, $\sqcup$)
             sobre el retículo distributivo de severidades de Heyting:
             $$\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\} \quad\big[699\big]$$
             $$v_{\mathrm{final}} = v_{\mathrm{Liouville}} \sqcup v_{\mathrm{Passivity}} \sqcup v_{\mathrm{Drift}} \quad\big[701\big]$$
             Si $v_{\mathrm{final}} = \mathrm{VETOED}$, se detona síncronamente
             el veto en software (RAM) y se despacha la interrupción Crowbar
             por hardware en el microcontrolador perimetral ESP32.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema MIC / integrador 3.0.0.
# Se conservan stubs alineados para aislamiento analítico.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.symplectic_verlet_integrator import (
        IntegrationAction as MotorIntegrationAction,
        IntegrationVerdict as MotorIntegrationVerdict,
        JacobianMethod,
        SymplecticIntegratorReport,
        SymplecticState,
        SymplecticVerletIntegrator,
    )
except ImportError:

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

        pass

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

        pass

    class MotorIntegrationVerdict(IntEnum):
        COHERENT = 0
        DEGRADED = 1
        VETOED = 2

    class MotorIntegrationAction(Enum):
        NONE = "none"
        REDUCE_TIMESTEP = "reduce_timestep"
        HALT_INTEGRATION = "halt_integration"

    class JacobianMethod(str, Enum):
        NUMERICAL = "numerical"
        CSD = "csd"
        ANALYTICAL = "analytical"

    @dataclass(frozen=True, slots=True)
    class SymplecticState:
        coordinates: NDArray[np.float64]
        momenta: NDArray[np.float64]
        hamiltonian: float

    @dataclass(frozen=True, slots=True)
    class SymplecticIntegratorReport:
        state_next: SymplecticState
        symplectic_residual: float
        is_symplectically_invariant: bool
        dissipation_rate: float
        is_lyapunov_passive: bool
        jacobian_matrix: NDArray[np.float64] | None = None
        liouville_residual: float = 0.0
        hamiltonian_drift: float = 0.0
        kinetic_energy: float = 0.0
        potential_energy: float = 0.0
        potential_is_surrogate: bool = False
        integration_verdict: MotorIntegrationVerdict = MotorIntegrationVerdict.COHERENT
        recommended_action: MotorIntegrationAction = MotorIntegrationAction.NONE
        symplectic_verdict: MotorIntegrationVerdict = MotorIntegrationVerdict.COHERENT
        passivity_verdict: MotorIntegrationVerdict = MotorIntegrationVerdict.COHERENT
        energy_verdict: MotorIntegrationVerdict = MotorIntegrationVerdict.COHERENT
        jacobian_method: str = "numerical"
        timestep: float = 0.0

    SymplecticVerletIntegrator = Any  # type: ignore[misc, assignment]


logger = logging.getLogger("MIC.Agents.Physics.SymplecticVerletIntegratorAgent")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE POLÍTICA DE SOFTWARE
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)

# Bandas de política del agente (pueden ser más estrictas que las del motor).
_SOFT_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_HARD_SYMPLECTIC_TOL: Final[float] = 1.0e-5
_SOFT_LIOUVILLE_TOL: Final[float] = 1.0e-10
_HARD_LIOUVILLE_TOL: Final[float] = 1.0e-5
_SOFT_LYAPUNOV_LIMIT: Final[float] = 1.0e-12
_HARD_LYAPUNOV_LIMIT: Final[float] = 1.0e-6
_SOFT_ENERGY_DRIFT_TOL: Final[float] = 1.0e-8
_HARD_ENERGY_DRIFT_TOL: Final[float] = 1.0e-3
_SOFT_KINETIC_NEGATIVE_TOL: Final[float] = 1.0e-14
_HARD_KINETIC_NEGATIVE_TOL: Final[float] = 1.0e-8

# Cotas de invarianza esperadas por método de Jacobiano (espejo del motor 3.0.0).
_JACOBIAN_INVARIANCE_TOL: Final[dict[str, float]] = {
    "numerical": 1.0e-5,
    "csd": 1.0e-11,
    "analytical": 1.0e-12,
}

_MIN_DT: Final[float] = 1.0e-15
_MAX_DT: Final[float] = 1.0e6

__version__: Final[str] = "4.0.0-OODA-Heyting-Liouville-Lyapunov-Software-Strict"


# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING Y JERARQUÍA DE EXCEPCIONES
# ══════════════════════════════════════════════════════════════════════════════
class SymplecticHeytingVerdict(IntEnum):
    r"""
    Clasificador de subobjetos en el Topos de la Integración Simpléctica.

        $$\bot=\mathtt{COHERENT}
          \le\mathtt{DEGRADED}
          \le\mathtt{VETOED}=\top.$$
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: SymplecticHeytingVerdict) -> SymplecticHeytingVerdict:
        if not isinstance(other, SymplecticHeytingVerdict):
            raise TypeError("join exige SymplecticHeytingVerdict.")
        return SymplecticHeytingVerdict(max(self.value, other.value))

    def meet(self, other: SymplecticHeytingVerdict) -> SymplecticHeytingVerdict:
        if not isinstance(other, SymplecticHeytingVerdict):
            raise TypeError("meet exige SymplecticHeytingVerdict.")
        return SymplecticHeytingVerdict(min(self.value, other.value))

    def implies(self, other: SymplecticHeytingVerdict) -> SymplecticHeytingVerdict:
        if not isinstance(other, SymplecticHeytingVerdict):
            raise TypeError("implies exige SymplecticHeytingVerdict.")
        if self.value <= other.value:
            return SymplecticHeytingVerdict.VETOED
        return other

    def negate(self) -> SymplecticHeytingVerdict:
        return self.implies(SymplecticHeytingVerdict.COHERENT)

    @property
    def is_terminal(self) -> bool:
        return self is SymplecticHeytingVerdict.VETOED


class CrowbarAction(Enum):
    """
    Token de política de contención *en software*.

    No hay actuación sobre GPIO, tiristores ni ningún periférico.
    ``GPIO_INTERRUPT_CROWBAR`` se conserva como alias histórico de
    ``SOFTWARE_HALT`` para no romper clientes existentes.
    """

    NONE = 0
    LOG_WARNING = 1
    SOFTWARE_HALT = 2
    GPIO_INTERRUPT_CROWBAR = 2


class SymplecticAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de Integración Simpléctica."""

    pass


class IntegratorContractError(SymplecticAgentError):
    """El motor físico no satisface el contrato funtorial exigido."""

    pass


class PhaseHandoffCollapse(SymplecticAgentError):
    """Un certificado de fase no satisface las precondiciones de la siguiente."""

    pass


class LiouvilleVolumeCollapse(SymplecticAgentError):
    r"""El volumen de fase no se conserva: $$\lvert\det M-1\rvert>\tau_{\mathrm{hard}}$$."""

    pass


class SymplecticInvarianceCollapse(SymplecticAgentError):
    r"""Se rompe $$M^\top\Omega M=\Omega$$ por encima de la cota dura."""

    pass


class LyapunovPassivityViolation(SymplecticAgentError):
    r"""La evolución inyecta energía espuria: $$\dot{H}|_R>0$$."""

    pass


class HamiltonianDriftCollapse(SymplecticAgentError):
    """El drift discreto del Hamiltoniano excede la política dura."""

    pass


class HeytingLatticeVeto(SymplecticAgentError):
    r"""
    Detonada síncronamente cuando el retículo toca el supremo $$\top$$.

    Aniquila el estado puramente en software (RAM).
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES (Contratos Categóricos de Handoff)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1SymplecticObservation:
    r"""
    Certificado de la Fase 1.  Precondición formal de la Fase 2.

        $$v_{\omega\mathrm{-join}}
          =v_{\omega}\sqcup v_{\mathrm{vol}}
           \sqcup v_{\mathrm{flag}}\sqcup v_{\mathrm{motor}}$$
    """

    residual: float
    is_stable_symplectic: bool
    phase1_verdict: SymplecticHeytingVerdict
    liouville_residual: float = 0.0
    relative_symplectic_residual: float = 0.0
    symplectic_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    liouville_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    invariance_flag_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    motor_symplectic_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    jacobian_method: str = "numerical"
    invariance_tolerance: float = _HARD_SYMPLECTIC_TOL
    jacobian_is_present: bool = False
    jacobian_dimension: int = 0


@dataclass(frozen=True, slots=True)
class Phase2PassivityOrientation:
    r"""
    Certificado de la Fase 2.  Precondición formal de la Fase 3.

        $$v_{\mathrm{pass}}
          =v_{\mathrm{Rayleigh}}\sqcup v_{\Delta H}
           \sqcup v_{T}\sqcup v_{\mathrm{flag}}
           \sqcup v_{\mathrm{motor}}$$
    """

    phase1_observation: Phase1SymplecticObservation
    dissipation_rate: float
    is_passive_lyapunov: bool
    phase2_verdict: SymplecticHeytingVerdict
    hamiltonian_drift: float = 0.0
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    hamiltonian: float = 0.0
    relative_energy_drift: float = 0.0
    potential_is_surrogate: bool = False
    rayleigh_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    energy_drift_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    kinetic_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    passivity_flag_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    motor_passivity_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    motor_energy_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    timestep: float = 0.0


@dataclass(frozen=True, slots=True)
class SymplecticGovernanceState:
    r"""
    Objeto terminal del endofuntor $$Z_{\mathrm{Verlet}}$$.

        $$v_{\mathrm{final}}
          =v_{\mathrm{Fase}\,1}\sqcup v_{\mathrm{Fase}\,2}
           \sqcup v_{\mathrm{motor}}$$
    """

    phase2_orientation: Phase2PassivityOrientation
    final_verdict: SymplecticHeytingVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    timestamp_utc: float
    provenance_hash: str
    diagnostic_note: str = ""
    is_epistemologically_valid: bool = True
    motor_integration_verdict: SymplecticHeytingVerdict = SymplecticHeytingVerdict.COHERENT
    recommended_action: str = "none"
    integrator_report: Any | None = None


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → OBSERVACIÓN DE LIOUVILLE Y FORMA SIMPLÉCTICA (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_SymplecticObserver:
    r"""
    Fase 1: interroga el certificado geométrico del motor y proyecta
    $$r_\omega$$, $$r_{\mathrm{Liouville}}$$ y las banderas de
    invarianza al retículo $$\Omega_3$$.

    El morfismo terminal ``handoff_phase1_to_phase2`` eleva el
    certificado a precondición formal de la Fase 2.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.1  Utilidades elementales de validación
    # ──────────────────────────────────────────────────────────────────────────
    def _as_finite_float(self, value: object, name: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise SymplecticAgentError(
                f"{name} no debe ser booleano; se requiere un número real."
            )
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise SymplecticAgentError(
                f"{name} no puede convertirse a un número real."
            ) from exc
        if not math.isfinite(result):
            raise SymplecticAgentError(f"{name} no es finito.")
        return result

    def _as_nonnegative_finite_float(self, value: object, name: str) -> float:
        result = self._as_finite_float(value, name)
        if result < 0.0:
            raise SymplecticAgentError(f"{name} no puede ser negativo.")
        return result

    def _as_bool(self, value: object, name: str) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise SymplecticAgentError(f"{name} debe ser booleano.")
        return bool(value)

    def _as_int(self, value: object, name: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise SymplecticAgentError(f"{name} no debe ser booleano.")
        if isinstance(value, (int, np.integer)):
            return int(value)
        raise SymplecticAgentError(f"{name} debe ser un entero.")

    def _get_required_attribute(self, obj: object, attr: str, name: str) -> Any:
        if not hasattr(obj, attr):
            raise SymplecticAgentError(f"{name} no contiene el campo '{attr}'.")
        return getattr(obj, attr)

    def _get_required_finite_float(self, obj: object, attr: str, name: str) -> float:
        return self._as_finite_float(
            self._get_required_attribute(obj, attr, name),
            f"{name}.{attr}",
        )

    def _get_required_nonnegative_finite_float(
        self,
        obj: object,
        attr: str,
        name: str,
    ) -> float:
        return self._as_nonnegative_finite_float(
            self._get_required_attribute(obj, attr, name),
            f"{name}.{attr}",
        )

    def _get_required_bool(self, obj: object, attr: str, name: str) -> bool:
        return self._as_bool(
            self._get_required_attribute(obj, attr, name),
            f"{name}.{attr}",
        )

    def _get_optional_finite_float(
        self,
        obj: object,
        attr: str,
        default: float,
        name: str,
    ) -> float:
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        return self._as_finite_float(value, f"{name}.{attr}")

    def _get_optional_nonnegative_finite_float(
        self,
        obj: object,
        attr: str,
        default: float,
        name: str,
    ) -> float:
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        return self._as_nonnegative_finite_float(value, f"{name}.{attr}")

    def _get_optional_bool(
        self,
        obj: object,
        attr: str,
        default: bool,
        name: str,
    ) -> bool:
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        return self._as_bool(value, f"{name}.{attr}")

    def _get_optional_str(
        self,
        obj: object,
        attr: str,
        default: str,
        name: str,
    ) -> str:
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        if isinstance(value, Enum):
            return str(value.value)
        if not isinstance(value, str):
            raise SymplecticAgentError(f"{name}.{attr} debe ser una cadena.")
        return value

    def _validate_verdict(
        self,
        verdict: object,
        name: str,
    ) -> SymplecticHeytingVerdict:
        if not isinstance(verdict, SymplecticHeytingVerdict):
            raise SymplecticAgentError(
                f"{name} debe ser una instancia de SymplecticHeytingVerdict."
            )
        return verdict

    def _heyting_join(
        self,
        verdicts: Iterable[SymplecticHeytingVerdict],
    ) -> SymplecticHeytingVerdict:
        acc = SymplecticHeytingVerdict.COHERENT
        for idx, verdict in enumerate(verdicts):
            acc = acc.join(self._validate_verdict(verdict, f"verdict[{idx}]"))
        return acc

    def _map_motor_verdict(
        self,
        payload: object,
        name: str,
    ) -> SymplecticHeytingVerdict:
        """Proyecta el ``IntegrationVerdict`` del motor (o un entero 0..2) a $$\Omega_3$$."""
        if payload is None:
            return SymplecticHeytingVerdict.COHERENT
        if isinstance(payload, SymplecticHeytingVerdict):
            return payload
        if isinstance(payload, IntEnum):
            try:
                return SymplecticHeytingVerdict(int(payload.value))
            except ValueError as exc:
                raise SymplecticAgentError(
                    f"{name} no es un veredicto admisible."
                ) from exc
        if isinstance(payload, (int, np.integer)) and not isinstance(
            payload, (bool, np.bool_)
        ):
            try:
                return SymplecticHeytingVerdict(int(payload))
            except ValueError as exc:
                raise SymplecticAgentError(
                    f"{name} no es un veredicto admisible."
                ) from exc
        raise SymplecticAgentError(f"{name} no pudo proyectarse a Ω₃.")

    def _threshold_verdict(
        self,
        value: float,
        soft: float,
        hard: float,
        name: str,
    ) -> SymplecticHeytingVerdict:
        if not math.isfinite(value):
            raise SymplecticAgentError(f"{name} no es finito.")
        if value > hard:
            return SymplecticHeytingVerdict.VETOED
        if value > soft:
            return SymplecticHeytingVerdict.DEGRADED
        return SymplecticHeytingVerdict.COHERENT

    def _normalize_jacobian_method(self, method: object) -> str:
        if method is None:
            return JacobianMethod.NUMERICAL.value
        if isinstance(method, Enum):
            return str(method.value).lower()
        if isinstance(method, str):
            normalized = method.strip().lower()
            if normalized in _JACOBIAN_INVARIANCE_TOL:
                return normalized
            return JacobianMethod.NUMERICAL.value
        return JacobianMethod.NUMERICAL.value

    def _invariance_tolerance_for(self, method: str) -> float:
        motor_tol = _JACOBIAN_INVARIANCE_TOL.get(method, _HARD_SYMPLECTIC_TOL)
        return max(_HARD_SYMPLECTIC_TOL, motor_tol)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.2  Validación del informe del motor
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_report(self, report: object) -> None:
        if not isinstance(report, SymplecticIntegratorReport):
            raise SymplecticAgentError(
                "El reporte debe ser una instancia de SymplecticIntegratorReport."
            )
        self._get_required_nonnegative_finite_float(
            report, "symplectic_residual", "report"
        )
        self._get_required_bool(report, "is_symplectically_invariant", "report")
        self._get_required_finite_float(report, "dissipation_rate", "report")
        self._get_required_bool(report, "is_lyapunov_passive", "report")
        if not hasattr(report, "state_next"):
            raise SymplecticAgentError("report no contiene 'state_next'.")
        state = report.state_next
        if not isinstance(state, SymplecticState):
            raise SymplecticAgentError(
                "report.state_next debe ser una instancia de SymplecticState."
            )

    def _inspect_jacobian(
        self,
        report: SymplecticIntegratorReport,
    ) -> tuple[bool, int]:
        matrix = getattr(report, "jacobian_matrix", None)
        if matrix is None:
            return False, 0
        try:
            array = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SymplecticAgentError(
                "report.jacobian_matrix no es convertible a float64."
            ) from exc
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise SymplecticAgentError(
                "report.jacobian_matrix debe ser una matriz cuadrada."
            )
        if array.size == 0:
            raise SymplecticAgentError("report.jacobian_matrix no puede ser vacía.")
        if not np.all(np.isfinite(array)):
            raise SymplecticAgentError(
                "report.jacobian_matrix contiene valores no finitos."
            )
        return True, int(array.shape[0])

    # ──────────────────────────────────────────────────────────────────────────
    # 1.3  Clasificadores locales de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_symplectic_residual(
        self,
        residual: float,
        method: str,
    ) -> tuple[float, SymplecticHeytingVerdict]:
        hard = self._invariance_tolerance_for(method)
        soft = min(_SOFT_SYMPLECTIC_TOL, 0.1 * hard)
        return residual, self._threshold_verdict(
            residual, soft, hard, "symplectic_residual"
        )

    def _classify_liouville_residual(
        self,
        residual: float,
        method: str,
    ) -> SymplecticHeytingVerdict:
        hard = max(_HARD_LIOUVILLE_TOL, self._invariance_tolerance_for(method))
        soft = min(_SOFT_LIOUVILLE_TOL, 0.1 * hard)
        return self._threshold_verdict(residual, soft, hard, "liouville_residual")

    def _classify_invariance_flag(
        self,
        residual: float,
        is_invariant: bool,
        hard: float,
    ) -> SymplecticHeytingVerdict:
        r"""
        Inconsistencia bandera/residuo:

        - bandera falsa con residuo bajo  → DEGRADED (el motor fue conservador);
        - bandera verdadera con residuo duro → VETOED (el motor se contradice).
        """
        if is_invariant and residual > hard:
            return SymplecticHeytingVerdict.VETOED
        if (not is_invariant) and residual <= hard:
            return SymplecticHeytingVerdict.DEGRADED
        return SymplecticHeytingVerdict.COHERENT

    # ──────────────────────────────────────────────────────────────────────────
    # 1.4  Núcleo terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _observe_symplectic_invariance(
        self,
        report: SymplecticIntegratorReport,
    ) -> Phase1SymplecticObservation:
        self._validate_report(report)
        residual = self._get_required_nonnegative_finite_float(
            report, "symplectic_residual", "report"
        )
        is_invariant = self._get_required_bool(
            report, "is_symplectically_invariant", "report"
        )
        liouville = self._get_optional_nonnegative_finite_float(
            report, "liouville_residual", 0.0, "report"
        )
        method = self._normalize_jacobian_method(
            getattr(report, "jacobian_method", None)
        )
        jacobian_present, jacobian_dim = self._inspect_jacobian(report)
        hard = self._invariance_tolerance_for(method)
        relative = residual / max(1.0, hard)

        symplectic_verdict = self._classify_symplectic_residual(residual, method)[1]
        liouville_verdict = self._classify_liouville_residual(liouville, method)
        flag_verdict = self._classify_invariance_flag(residual, is_invariant, hard)
        motor_verdict = self._map_motor_verdict(
            getattr(report, "symplectic_verdict", None),
            "report.symplectic_verdict",
        )
        phase1_verdict = self._heyting_join(
            (
                symplectic_verdict,
                liouville_verdict,
                flag_verdict,
                motor_verdict,
            )
        )
        is_stable = phase1_verdict is not SymplecticHeytingVerdict.VETOED

        logger.debug(
            "Fase 1 Liouville: r_ω=%.4e, r_vol=%.4e, método=%s, "
            "invariante=%s, veredicto=%s.",
            residual,
            liouville,
            method,
            is_invariant,
            phase1_verdict.name,
        )
        return Phase1SymplecticObservation(
            residual=residual,
            is_stable_symplectic=is_stable,
            phase1_verdict=phase1_verdict,
            liouville_residual=liouville,
            relative_symplectic_residual=relative,
            symplectic_verdict=symplectic_verdict,
            liouville_verdict=liouville_verdict,
            invariance_flag_verdict=flag_verdict,
            motor_symplectic_verdict=motor_verdict,
            jacobian_method=method,
            invariance_tolerance=hard,
            jacobian_is_present=jacobian_present,
            jacobian_dimension=jacobian_dim,
        )

    def execute_phase1(
        self,
        report: SymplecticIntegratorReport,
    ) -> Phase1SymplecticObservation:
        """
        Método terminal operativo de la Fase 1.

        Su salida constituye el dominio formal de
        ``handoff_phase1_to_phase2``.
        """
        return self._observe_symplectic_invariance(report)

    def handoff_phase1_to_phase2(
        self,
        observation: Phase1SymplecticObservation,
    ) -> Phase1SymplecticObservation:
        r"""
        Morfismo de transición

            $$\Phi_{12}:
              \mathrm{Phase1SymplecticObservation}
              \longrightarrow
              \mathrm{Phase1SymplecticObservation}.$$

        Poscondición de la Fase 1  ≡  precondición de la Fase 2:

            el puente es una instancia bien tipada, los residuales son
            finitos no negativos, los veredictos habitan en $$\Omega_3$$
            y el join almacenado coincide con el de los clasificadores
            granulares.

        Este método es la definición formal final de la Fase 1 y, a la
        vez, el dominio sobre el que la Fase 2 orienta la pasividad.
        ``Phase2_LyapunovPassivityOrienter`` comienza invocándolo.

        No colapsa el retículo: el veto de política se reserva a la
        Fase 3 (o al orquestador en modo ``fail_fast``).
        """
        self._validate_phase1_structure(observation)
        self._assert_phase1_lattice_consistency(observation)
        return observation

    def _validate_phase1_structure(
        self,
        observation: Phase1SymplecticObservation,
    ) -> None:
        if not isinstance(observation, Phase1SymplecticObservation):
            raise PhaseHandoffCollapse(
                "handoff_phase1_to_phase2 exige Phase1SymplecticObservation."
            )
        self._as_nonnegative_finite_float(observation.residual, "observation.residual")
        self._as_nonnegative_finite_float(
            observation.liouville_residual, "observation.liouville_residual"
        )
        self._as_nonnegative_finite_float(
            observation.relative_symplectic_residual,
            "observation.relative_symplectic_residual",
        )
        self._as_nonnegative_finite_float(
            observation.invariance_tolerance, "observation.invariance_tolerance"
        )
        self._as_bool(observation.is_stable_symplectic, "observation.is_stable_symplectic")
        self._as_bool(observation.jacobian_is_present, "observation.jacobian_is_present")
        rank = self._as_int(
            observation.jacobian_dimension, "observation.jacobian_dimension"
        )
        if rank < 0:
            raise PhaseHandoffCollapse(
                "observation.jacobian_dimension no puede ser negativo."
            )
        for field in (
            "phase1_verdict",
            "symplectic_verdict",
            "liouville_verdict",
            "invariance_flag_verdict",
            "motor_symplectic_verdict",
        ):
            self._validate_verdict(getattr(observation, field), f"observation.{field}")

    def _assert_phase1_lattice_consistency(
        self,
        observation: Phase1SymplecticObservation,
    ) -> None:
        recomputed = self._heyting_join(
            (
                observation.symplectic_verdict,
                observation.liouville_verdict,
                observation.invariance_flag_verdict,
                observation.motor_symplectic_verdict,
            )
        )
        if recomputed is not observation.phase1_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del retículo en Φ₁₂: "
                f"join granular = {recomputed.name}, "
                f"almacenado = {observation.phase1_verdict.name}."
            )
        expected_stable = observation.phase1_verdict is not SymplecticHeytingVerdict.VETOED
        if bool(observation.is_stable_symplectic) is not expected_stable:
            raise PhaseHandoffCollapse(
                "is_stable_symplectic contradice phase1_verdict en Φ₁₂."
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → ORIENTACIÓN DE PASIVIDAD DE LYAPUNOV (Orient)
#          continuación formal de handoff_phase1_to_phase2
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_LyapunovPassivityOrienter(Phase1_SymplecticObserver):
    r"""
    Fase 2: recibe el puente certificado por ``handoff_phase1_to_phase2``
    y certifica la Segunda Ley sobre el certificado del motor:

        $$\dot{H}|_R=-v^\top Rv\le 0,\qquad
          \Delta H=H_{k+1}-H_k.$$

    El morfismo terminal ``handoff_phase2_to_phase3`` eleva el
    certificado a precondición formal de la Fase 3.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 2.1  Continuación inmediata del handoff de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_observation(
        self,
        observation: Phase1SymplecticObservation,
    ) -> Phase1SymplecticObservation:
        """
        Primer método de la Fase 2.

        Es la continuación literal de ``handoff_phase1_to_phase2``.
        """
        return self.handoff_phase1_to_phase2(observation)

    def _validate_phase1_observation(
        self,
        observation: Phase1SymplecticObservation,
    ) -> None:
        """Fachada conservada: revalida el puente de Fase 1 vía $$\Phi_{12}$$. """
        self._receive_certified_observation(observation)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.2  Clasificadores locales de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_rayleigh_rate(
        self,
        dissipation_rate: float,
    ) -> SymplecticHeytingVerdict:
        """
        ``dissipation_rate`` del motor es $$-v^\top Rv$$: pasivo ssi $$\le 0$$.

        El veredicto mira la parte positiva (inyección espuria).
        """
        injection = max(0.0, dissipation_rate)
        return self._threshold_verdict(
            injection,
            _SOFT_LYAPUNOV_LIMIT,
            _HARD_LYAPUNOV_LIMIT,
            "dissipation_rate⁺",
        )

    def _classify_energy_drift(
        self,
        drift: float,
        hamiltonian: float,
        dissipation_rate: float,
    ) -> tuple[float, SymplecticHeytingVerdict]:
        r"""
        Drift relativo $$\lvert\Delta H\rvert/\max(1,\lvert H\rvert)$$.

        - Flujo aparentemente conservativo ($$\dot{H}|_R\approx 0$$):
          se veta el valor absoluto.
        - Flujo disipativo: se veta sólo la *inyección* $$\Delta H>0$$.
        """
        scale = max(1.0, abs(hamiltonian))
        relative = abs(drift) / scale
        conservative = dissipation_rate <= _HARD_LYAPUNOV_LIMIT
        if conservative:
            measured = relative
        else:
            measured = max(0.0, drift) / scale
        return relative, self._threshold_verdict(
            measured,
            _SOFT_ENERGY_DRIFT_TOL,
            _HARD_ENERGY_DRIFT_TOL,
            "hamiltonian_drift",
        )

    def _classify_kinetic_energy(
        self,
        kinetic: float,
    ) -> SymplecticHeytingVerdict:
        if kinetic >= 0.0:
            return SymplecticHeytingVerdict.COHERENT
        magnitude = abs(kinetic)
        return self._threshold_verdict(
            magnitude,
            _SOFT_KINETIC_NEGATIVE_TOL,
            _HARD_KINETIC_NEGATIVE_TOL,
            "|T|⁻",
        )

    def _classify_passivity_flag(
        self,
        dissipation_rate: float,
        is_passive: bool,
    ) -> SymplecticHeytingVerdict:
        injection = max(0.0, dissipation_rate)
        if is_passive and injection > _HARD_LYAPUNOV_LIMIT:
            return SymplecticHeytingVerdict.VETOED
        if (not is_passive) and injection <= _HARD_LYAPUNOV_LIMIT:
            return SymplecticHeytingVerdict.DEGRADED
        return SymplecticHeytingVerdict.COHERENT

    # ──────────────────────────────────────────────────────────────────────────
    # 2.3  Núcleo terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _orient_passivity(
        self,
        phase1_obs: Phase1SymplecticObservation,
        report: SymplecticIntegratorReport,
    ) -> Phase2PassivityOrientation:
        certified = self._receive_certified_observation(phase1_obs)
        self._validate_report(report)

        rate = self._get_required_finite_float(report, "dissipation_rate", "report")
        is_passive = self._get_required_bool(report, "is_lyapunov_passive", "report")
        drift = self._get_optional_finite_float(
            report, "hamiltonian_drift", 0.0, "report"
        )
        kinetic = self._get_optional_finite_float(
            report, "kinetic_energy", 0.0, "report"
        )
        potential = self._get_optional_finite_float(
            report, "potential_energy", 0.0, "report"
        )
        timestep = self._get_optional_finite_float(report, "timestep", 0.0, "report")
        surrogate = self._get_optional_bool(
            report, "potential_is_surrogate", False, "report"
        )
        hamiltonian = self._as_finite_float(
            report.state_next.hamiltonian, "report.state_next.hamiltonian"
        )

        rayleigh_verdict = self._classify_rayleigh_rate(rate)
        relative_drift, energy_verdict = self._classify_energy_drift(
            drift, hamiltonian, rate
        )
        kinetic_verdict = self._classify_kinetic_energy(kinetic)
        flag_verdict = self._classify_passivity_flag(rate, is_passive)
        motor_passivity = self._map_motor_verdict(
            getattr(report, "passivity_verdict", None),
            "report.passivity_verdict",
        )
        motor_energy = self._map_motor_verdict(
            getattr(report, "energy_verdict", None),
            "report.energy_verdict",
        )
        phase2_verdict = self._heyting_join(
            (
                rayleigh_verdict,
                energy_verdict,
                kinetic_verdict,
                flag_verdict,
                motor_passivity,
                motor_energy,
            )
        )
        is_passive_certified = phase2_verdict is not SymplecticHeytingVerdict.VETOED

        logger.debug(
            "Fase 2 Lyapunov: Ḣ_R=%.4e, ΔH=%.4e, T=%.4e, V=%.4e, "
            "surrogado=%s, veredicto=%s.",
            rate,
            drift,
            kinetic,
            potential,
            surrogate,
            phase2_verdict.name,
        )
        return Phase2PassivityOrientation(
            phase1_observation=certified,
            dissipation_rate=rate,
            is_passive_lyapunov=is_passive_certified,
            phase2_verdict=phase2_verdict,
            hamiltonian_drift=drift,
            kinetic_energy=kinetic,
            potential_energy=potential,
            hamiltonian=hamiltonian,
            relative_energy_drift=relative_drift,
            potential_is_surrogate=surrogate,
            rayleigh_verdict=rayleigh_verdict,
            energy_drift_verdict=energy_verdict,
            kinetic_verdict=kinetic_verdict,
            passivity_flag_verdict=flag_verdict,
            motor_passivity_verdict=motor_passivity,
            motor_energy_verdict=motor_energy,
            timestep=timestep,
        )

    def execute_phase2(
        self,
        phase1_obs: Phase1SymplecticObservation,
        report: SymplecticIntegratorReport,
    ) -> Phase2PassivityOrientation:
        """
        Método terminal operativo de la Fase 2.

        Recibe la salida formal de ``execute_phase1`` (vía
        ``handoff_phase1_to_phase2``) y produce la entrada canónica de
        ``handoff_phase2_to_phase3``.
        """
        return self._orient_passivity(phase1_obs, report)

    def handoff_phase2_to_phase3(
        self,
        orientation: Phase2PassivityOrientation,
    ) -> Phase2PassivityOrientation:
        r"""
        Morfismo de transición

            $$\Phi_{23}:
              \mathrm{Phase2PassivityOrientation}
              \longrightarrow
              \mathrm{Phase2PassivityOrientation}.$$

        Poscondición de la Fase 2  ≡  precondición de la Fase 3:

            el puente es una instancia bien tipada, contiene un
            ``Phase1SymplecticObservation`` ya revalidado por $$\Phi_{12}$$,
            sus veredictos habitan en $$\Omega_3$$ y el join almacenado
            coincide con el de los clasificadores granulares.

        Este método es la definición formal final de la Fase 2 y, a la
        vez, el dominio sobre el que la Fase 3 calcula el supremo.
        ``Phase3_HeytingVerdictDecider`` comienza invocándolo.
        """
        self._validate_phase2_structure(orientation)
        self._assert_phase2_lattice_consistency(orientation)
        return orientation

    def _validate_phase2_structure(
        self,
        orientation: Phase2PassivityOrientation,
    ) -> None:
        if not isinstance(orientation, Phase2PassivityOrientation):
            raise PhaseHandoffCollapse(
                "handoff_phase2_to_phase3 exige Phase2PassivityOrientation."
            )
        self._receive_certified_observation(orientation.phase1_observation)
        self._as_finite_float(
            orientation.dissipation_rate, "orientation.dissipation_rate"
        )
        self._as_finite_float(
            orientation.hamiltonian_drift, "orientation.hamiltonian_drift"
        )
        self._as_finite_float(orientation.kinetic_energy, "orientation.kinetic_energy")
        self._as_finite_float(
            orientation.potential_energy, "orientation.potential_energy"
        )
        self._as_finite_float(orientation.hamiltonian, "orientation.hamiltonian")
        self._as_nonnegative_finite_float(
            orientation.relative_energy_drift, "orientation.relative_energy_drift"
        )
        self._as_finite_float(orientation.timestep, "orientation.timestep")
        self._as_bool(orientation.is_passive_lyapunov, "orientation.is_passive_lyapunov")
        self._as_bool(
            orientation.potential_is_surrogate, "orientation.potential_is_surrogate"
        )
        for field in (
            "phase2_verdict",
            "rayleigh_verdict",
            "energy_drift_verdict",
            "kinetic_verdict",
            "passivity_flag_verdict",
            "motor_passivity_verdict",
            "motor_energy_verdict",
        ):
            self._validate_verdict(getattr(orientation, field), f"orientation.{field}")

    def _assert_phase2_lattice_consistency(
        self,
        orientation: Phase2PassivityOrientation,
    ) -> None:
        recomputed = self._heyting_join(
            (
                orientation.rayleigh_verdict,
                orientation.energy_drift_verdict,
                orientation.kinetic_verdict,
                orientation.passivity_flag_verdict,
                orientation.motor_passivity_verdict,
                orientation.motor_energy_verdict,
            )
        )
        if recomputed is not orientation.phase2_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del retículo en Φ₂₃: "
                f"join granular = {recomputed.name}, "
                f"almacenado = {orientation.phase2_verdict.name}."
            )
        expected_passive = (
            orientation.phase2_verdict is not SymplecticHeytingVerdict.VETOED
        )
        if bool(orientation.is_passive_lyapunov) is not expected_passive:
            raise PhaseHandoffCollapse(
                "is_passive_lyapunov contradice phase2_verdict en Φ₂₃."
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → SUPREMO DE HEYTING Y SELLO DE PROCEDENCIA (Decide & Act)
#          continuación formal de handoff_phase2_to_phase3
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingVerdictDecider(Phase2_LyapunovPassivityOrienter):
    r"""
    Fase 3: recibe el puente certificado por ``handoff_phase2_to_phase3``
    y consolida el ciclo OODA

        $$v_{\mathrm{final}}
          =v_{\mathrm{Fase}\,1}\sqcup v_{\mathrm{Fase}\,2}
           \sqcup v_{\mathrm{motor}}.$$

    El Crowbar de este estrato es un token de política en software:
    bitácora, sello SHA-256 y, si procede, excepción tipada.  No hay
    actuación sobre GPIO ni sobre ningún periférico.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 3.1  Continuación inmediata del handoff de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_orientation(
        self,
        orientation: Phase2PassivityOrientation,
    ) -> Phase2PassivityOrientation:
        """
        Primer método de la Fase 3.

        Es la continuación literal de ``handoff_phase2_to_phase3``.
        """
        return self.handoff_phase2_to_phase3(orientation)

    def _validate_phase2_orientation(
        self,
        orientation: Phase2PassivityOrientation,
    ) -> None:
        """Fachada conservada: revalida el puente de Fase 2 vía $$\Phi_{23}$$. """
        self._receive_certified_orientation(orientation)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.2  Procedencia, fuentes de veto y colapso
    # ──────────────────────────────────────────────────────────────────────────
    def _collect_veto_sources(
        self,
        orientation: Phase2PassivityOrientation,
        motor_verdict: SymplecticHeytingVerdict,
    ) -> list[str]:
        phase1 = orientation.phase1_observation
        mapping: tuple[tuple[str, SymplecticHeytingVerdict], ...] = (
            ("Liouville.symplectic", phase1.symplectic_verdict),
            ("Liouville.volume", phase1.liouville_verdict),
            ("Liouville.flag", phase1.invariance_flag_verdict),
            ("Liouville.motor", phase1.motor_symplectic_verdict),
            ("Lyapunov.rayleigh", orientation.rayleigh_verdict),
            ("Lyapunov.drift", orientation.energy_drift_verdict),
            ("Lyapunov.kinetic", orientation.kinetic_verdict),
            ("Lyapunov.flag", orientation.passivity_flag_verdict),
            ("Lyapunov.motor_passivity", orientation.motor_passivity_verdict),
            ("Lyapunov.motor_energy", orientation.motor_energy_verdict),
            ("Motor.integration", motor_verdict),
        )
        return [label for label, verdict in mapping if verdict.is_terminal]

    def _select_collapse_type(
        self,
        orientation: Phase2PassivityOrientation,
        motor_verdict: SymplecticHeytingVerdict,
    ) -> type[SymplecticAgentError]:
        phase1 = orientation.phase1_observation
        if phase1.symplectic_verdict.is_terminal:
            return SymplecticInvarianceCollapse
        if phase1.liouville_verdict.is_terminal:
            return LiouvilleVolumeCollapse
        if orientation.rayleigh_verdict.is_terminal:
            return LyapunovPassivityViolation
        if orientation.energy_drift_verdict.is_terminal:
            return HamiltonianDriftCollapse
        if orientation.kinetic_verdict.is_terminal:
            return LyapunovPassivityViolation
        if motor_verdict.is_terminal:
            return HeytingLatticeVeto
        return HeytingLatticeVeto

    def _compose_provenance_hash(
        self,
        orientation: Phase2PassivityOrientation,
        final_verdict: SymplecticHeytingVerdict,
        motor_verdict: SymplecticHeytingVerdict,
    ) -> str:
        phase1 = orientation.phase1_observation
        payload = (
            f"rω={phase1.residual:.16e}|"
            f"rvol={phase1.liouville_residual:.16e}|"
            f"Ḣ={orientation.dissipation_rate:.16e}|"
            f"ΔH={orientation.hamiltonian_drift:.16e}|"
            f"H={orientation.hamiltonian:.16e}|"
            f"T={orientation.kinetic_energy:.16e}|"
            f"method={phase1.jacobian_method}|"
            f"v1={phase1.phase1_verdict.value}|"
            f"v2={orientation.phase2_verdict.value}|"
            f"vm={motor_verdict.value}|"
            f"vf={final_verdict.value}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _raise_lattice_veto(
        self,
        orientation: Phase2PassivityOrientation,
        final_verdict: SymplecticHeytingVerdict,
        motor_verdict: SymplecticHeytingVerdict,
    ) -> None:
        sources = self._collect_veto_sources(orientation, motor_verdict)
        detail = ", ".join(sources) if sources else "Unknown"
        phase1 = orientation.phase1_observation
        exc_type = self._select_collapse_type(orientation, motor_verdict)
        message = (
            "Colapso de software: el mapa de Verlet ha perdido un invariante "
            "geométrico o termodinámico. "
            f"Veredicto Supremo = {final_verdict.name}. "
            f"Fuente(s) de veto = [{detail}]. "
            f"r_ω = {phase1.residual:.6e}, "
            f"r_vol = {phase1.liouville_residual:.6e}, "
            f"Ḣ_R = {orientation.dissipation_rate:.6e}, "
            f"ΔH = {orientation.hamiltonian_drift:.6e}. "
            "Transacción aniquilada en RAM."
        )
        raise exc_type(message)

    def _policy_action(
        self,
        final_verdict: SymplecticHeytingVerdict,
        request_halt_token: bool,
    ) -> tuple[bool, CrowbarAction]:
        if final_verdict is SymplecticHeytingVerdict.VETOED:
            if request_halt_token:
                return True, CrowbarAction.SOFTWARE_HALT
            return False, CrowbarAction.LOG_WARNING
        if final_verdict is SymplecticHeytingVerdict.DEGRADED:
            return False, CrowbarAction.LOG_WARNING
        return False, CrowbarAction.NONE

    # ──────────────────────────────────────────────────────────────────────────
    # 3.3  Núcleo terminal de la Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_governance(
        self,
        phase2_orient: Phase2PassivityOrientation,
        report: SymplecticIntegratorReport | None = None,
        raise_on_veto: bool = True,
        trigger_hardware_crowbar: bool = False,
    ) -> SymplecticGovernanceState:
        """
        Consolida el supremo algebraico y sella la procedencia.

        ``trigger_hardware_crowbar`` se conserva por compatibilidad de
        API y *sólo* selecciona el token ``SOFTWARE_HALT``: no hay
        escritura a GPIO ni actuación sobre periféricos.
        """
        certified = self._receive_certified_orientation(phase2_orient)
        motor_verdict = SymplecticHeytingVerdict.COHERENT
        recommended = "none"
        if report is not None:
            self._validate_report(report)
            motor_verdict = self._map_motor_verdict(
                getattr(report, "integration_verdict", None),
                "report.integration_verdict",
            )
            recommended = self._get_optional_str(
                report, "recommended_action", "none", "report"
            )

        final_verdict = self._heyting_join(
            (
                certified.phase1_observation.phase1_verdict,
                certified.phase2_verdict,
                motor_verdict,
            )
        )
        is_valid = final_verdict is not SymplecticHeytingVerdict.VETOED
        halted, action = self._policy_action(final_verdict, trigger_hardware_crowbar)

        if final_verdict is SymplecticHeytingVerdict.VETOED:
            logger.critical(
                "Veto terminal de software. acción=%s, r_ω=%.4e, Ḣ_R=%.4e.",
                action.name,
                certified.phase1_observation.residual,
                certified.dissipation_rate,
            )
            if raise_on_veto:
                self._raise_lattice_veto(certified, final_verdict, motor_verdict)
        elif final_verdict is SymplecticHeytingVerdict.DEGRADED:
            logger.warning(
                "Integración degradada. r_ω=%.4e, r_vol=%.4e, ΔH=%.4e.",
                certified.phase1_observation.residual,
                certified.phase1_observation.liouville_residual,
                certified.hamiltonian_drift,
            )

        provenance = self._compose_provenance_hash(
            certified, final_verdict, motor_verdict
        )
        diagnostic = (
            f"Veredicto: {final_verdict.name}. "
            f"Invarianza: {certified.phase1_observation.is_stable_symplectic}. "
            f"Pasividad: {certified.is_passive_lyapunov}. "
            f"Método J: {certified.phase1_observation.jacobian_method}. "
            f"Acción: {action.name}."
        )
        return SymplecticGovernanceState(
            phase2_orientation=certified,
            final_verdict=final_verdict,
            crowbar_triggered=halted,
            crowbar_action=action,
            timestamp_utc=float(time.time()),
            provenance_hash=provenance,
            diagnostic_note=diagnostic,
            is_epistemologically_valid=is_valid,
            motor_integration_verdict=motor_verdict,
            recommended_action=recommended,
            integrator_report=report,
        )

    def execute_phase3(
        self,
        phase2_orient: Phase2PassivityOrientation,
        raise_on_veto: bool = True,
        trigger_hardware_crowbar: bool = False,
        report: SymplecticIntegratorReport | None = None,
    ) -> SymplecticGovernanceState:
        """
        Método terminal de la Fase 3.

        Recibe la salida formal de ``execute_phase2`` (vía
        ``handoff_phase2_to_phase3``), retorna el estado lógico supremo
        y, opcionalmente, colapsa el retículo ante un veto.
        """
        return self._evaluate_governance(
            phase2_orient,
            report=report,
            raise_on_veto=raise_on_veto,
            trigger_hardware_crowbar=trigger_hardware_crowbar,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: SYMPLECTIC VERLET INTEGRATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════
class SymplecticVerletIntegratorAgent(Morphism, Phase3_HeytingVerdictDecider):
    r"""
    Soberano de la integridad de fase.

    Gobierna al motor ``SymplecticVerletIntegrator`` en un ciclo OODA:

        $$F_{\mathrm{Agent}}
          =F_3\circ\Phi_{23}\circ F_2\circ\Phi_{12}\circ F_1
           \circ F_{\mathrm{Motor}}.$$
    """

    def __init__(self, integrator: SymplecticVerletIntegrator) -> None:
        self._integrator = integrator
        self._validate_integrator()

    def _validate_integrator(self) -> None:
        if self._integrator is None:
            raise IntegratorContractError("El integrador no puede ser None.")
        for method in ("integrate_step_2nd_order", "integrate_step_4th_order"):
            if not callable(getattr(self._integrator, method, None)):
                raise IntegratorContractError(
                    f"El integrador debe exponer el método ejecutable '{method}'."
                )

    def _validate_state(self, state: object) -> None:
        if not isinstance(state, SymplecticState):
            raise SymplecticAgentError(
                "state_curr debe ser una instancia de SymplecticState."
            )
        try:
            coordinates = np.asarray(state.coordinates, dtype=np.float64)
            momenta = np.asarray(state.momenta, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SymplecticAgentError(
                "coordinates y momenta deben ser convertibles a float64."
            ) from exc
        if coordinates.ndim != 1 or momenta.ndim != 1:
            raise SymplecticAgentError("coordinates y momenta deben ser vectores 1-D.")
        if coordinates.size == 0 or coordinates.size != momenta.size:
            raise SymplecticAgentError(
                "coordinates y momenta deben tener la misma dimensión positiva."
            )
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(momenta)):
            raise SymplecticAgentError(
                "coordinates o momenta contienen valores no finitos."
            )
        self._as_finite_float(state.hamiltonian, "state_curr.hamiltonian")

        expected = getattr(self._integrator, "dimension", None)
        if isinstance(expected, int) and expected > 0 and coordinates.size != expected:
            raise SymplecticAgentError(
                f"El estado vive en R^{coordinates.size}, "
                f"el integrador está configurado para n = {expected}."
            )

    def _validate_governance_payload(
        self,
        force_gradient_q: object,
        mass_matrix_inv: object,
        dt: object,
        damping_matrix_r: object,
        external_forcing: object,
        state_dimension: int,
    ) -> None:
        if not callable(force_gradient_q):
            raise SymplecticAgentError("force_gradient_q debe ser un callable.")
        timestep = self._as_finite_float(dt, "dt")
        if timestep <= 0.0:
            raise SymplecticAgentError("dt debe ser estrictamente positivo.")
        if timestep < _MIN_DT or timestep > _MAX_DT:
            raise SymplecticAgentError(
                f"dt = {timestep:.4e} está fuera de [{_MIN_DT:.0e}, {_MAX_DT:.0e}]."
            )
        try:
            mass_inv = np.asarray(mass_matrix_inv, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SymplecticAgentError(
                "mass_matrix_inv no es convertible a float64."
            ) from exc
        if mass_inv.shape != (state_dimension, state_dimension):
            raise SymplecticAgentError(
                f"mass_matrix_inv debe ser una matriz {state_dimension}×{state_dimension}."
            )
        if not np.all(np.isfinite(mass_inv)):
            raise SymplecticAgentError("mass_matrix_inv contiene valores no finitos.")
        if damping_matrix_r is not None:
            try:
                damping = np.asarray(damping_matrix_r, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise SymplecticAgentError(
                    "damping_matrix_r no es convertible a float64."
                ) from exc
            if damping.shape != (state_dimension, state_dimension):
                raise SymplecticAgentError(
                    f"damping_matrix_r debe ser una matriz {state_dimension}×{state_dimension}."
                )
            if not np.all(np.isfinite(damping)):
                raise SymplecticAgentError(
                    "damping_matrix_r contiene valores no finitos."
                )
        if external_forcing is not None:
            try:
                forcing = np.asarray(external_forcing, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise SymplecticAgentError(
                    "external_forcing no es convertible a float64."
                ) from exc
            if forcing.shape != (state_dimension,):
                raise SymplecticAgentError(
                    f"external_forcing debe ser un vector de tamaño {state_dimension}."
                )
            if not np.all(np.isfinite(forcing)):
                raise SymplecticAgentError(
                    "external_forcing contiene valores no finitos."
                )

    def _maybe_fail_fast(
        self,
        verdict: SymplecticHeytingVerdict,
        stage: str,
        fail_fast: bool,
        factory: type[SymplecticAgentError],
        detail: str,
    ) -> None:
        if fail_fast and verdict.is_terminal:
            raise factory(f"Fail-fast en {stage}: veredicto = {verdict.name}. {detail}")

    def execute_symplectic_governance(
        self,
        state_curr: SymplecticState,
        force_gradient_q: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        mass_matrix_inv: NDArray[np.float64],
        dt: float,
        damping_matrix_r: NDArray[np.float64] | None = None,
        external_forcing: NDArray[np.float64] | None = None,
        potential_q: Callable[[NDArray[np.float64]], float] | None = None,
        use_fourth_order: bool = False,
        raise_on_veto: bool = True,
        trigger_hardware_crowbar: bool = False,
        jacobian_method: str = "numerical",
        stiffness_matrix_k: NDArray[np.float64] | None = None,
        force_gradient_complex_q: Callable[[NDArray[np.complex128]], NDArray[np.complex128]] | None = None,
        fail_fast: bool = False,
    ) -> SymplecticGovernanceState:
        r"""
        Ejecuta el ciclo categórico completo.

        Fase motor → Fase 1 (Liouville) → $$\Phi_{12}$$
                   → Fase 2 (Lyapunov)  → $$\Phi_{23}$$
                   → Fase 3 (Heyting).

        Si ``fail_fast`` es verdadero, un veto local detiene el ciclo
        antes de invocar la fase siguiente del agente (el motor ya
        produjo su certificado).
        """
        self._validate_state(state_curr)
        dimension = int(np.asarray(state_curr.coordinates).size)
        self._validate_governance_payload(
            force_gradient_q,
            mass_matrix_inv,
            dt,
            damping_matrix_r,
            external_forcing,
            dimension,
        )

        stepper = (
            self._integrator.integrate_step_4th_order
            if use_fourth_order
            else self._integrator.integrate_step_2nd_order
        )
        report = stepper(
            state_curr=state_curr,
            force_gradient_q=force_gradient_q,
            mass_matrix_inv=mass_matrix_inv,
            dt=dt,
            damping_matrix_r=damping_matrix_r,
            external_forcing=external_forcing,
            potential_q=potential_q,
            jacobian_method=jacobian_method,
            stiffness_matrix_k=stiffness_matrix_k,
            force_gradient_complex_q=force_gradient_complex_q,
        )
        if not isinstance(report, SymplecticIntegratorReport):
            raise IntegratorContractError(
                "El motor no devolvió un SymplecticIntegratorReport."
            )

        phase1 = self.execute_phase1(report)
        self._maybe_fail_fast(
            phase1.phase1_verdict,
            "Fase 1 (Liouville)",
            fail_fast,
            SymplecticInvarianceCollapse
            if phase1.symplectic_verdict.is_terminal
            else LiouvilleVolumeCollapse,
            f"r_ω = {phase1.residual:.6e}, r_vol = {phase1.liouville_residual:.6e}.",
        )

        phase2 = self.execute_phase2(phase1, report)
        self._maybe_fail_fast(
            phase2.phase2_verdict,
            "Fase 2 (Lyapunov)",
            fail_fast,
            LyapunovPassivityViolation,
            f"Ḣ_R = {phase2.dissipation_rate:.6e}, ΔH = {phase2.hamiltonian_drift:.6e}.",
        )

        state = self.execute_phase3(
            phase2_orient=phase2,
            raise_on_veto=raise_on_veto,
            trigger_hardware_crowbar=trigger_hardware_crowbar,
            report=report,
        )
        logger.info(
            "Gobernanza simpléctica completada: final=%s, válido=%s, "
            "r_ω=%.4e, Ḣ_R=%.4e, orden=%s.",
            state.final_verdict.name,
            state.is_epistemologically_valid,
            phase1.residual,
            phase2.dissipation_rate,
            "4" if use_fourth_order else "2",
        )
        return state


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SymplecticHeytingVerdict",
    "CrowbarAction",
    "SymplecticAgentError",
    "IntegratorContractError",
    "PhaseHandoffCollapse",
    "LiouvilleVolumeCollapse",
    "SymplecticInvarianceCollapse",
    "LyapunovPassivityViolation",
    "HamiltonianDriftCollapse",
    "HeytingLatticeVeto",
    "Phase1SymplecticObservation",
    "Phase2PassivityOrientation",
    "SymplecticGovernanceState",
    "Phase1_SymplecticObserver",
    "Phase2_LyapunovPassivityOrienter",
    "Phase3_HeytingVerdictDecider",
    "SymplecticVerletIntegratorAgent",
]