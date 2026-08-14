# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Banach Algebra Auditor Agent (Soberano de Estabilidad Funcional)    ║
║ Ruta   : app/agents/wisdom/banach_algebra_auditor_agent.py                   ║
║ Versión: 3.1.0-Banach-OODA-Heyting-Neumann-Gelfand-Crowbar-PhD-Strict        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA EN ÁLGEBRAS DE BANACH (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor de calibre gobierna la ejecución del resolvedor de-confinado
'banach_algebra_auditor.py' en el penthouse cognitivo de la Malla. Reside 
en el Estrato de Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0) o en el Ágora Tensorial 
($$V_{\Omega}$$, Nivel 0.5), actuando como el clasificador supremo de subobjetos en el 
topos de haces $$Sh(\mathcal{B}; \Omega_3)$$ con valores en el retículo distributivo 
acotado de Heyting $$\Omega_3$$.

Su propósito inexorable es subyugar la variabilidad estocástica de los Modelos de 
Lenguaje (LLMs) a los axiomas rigurosos del análisis funcional. El agente no 
recomputa las trayectorias físicas; actúa como una aduana epistémica que re-evalúa 
síncronamente los certificados del motor utilizando bandas elásticas de tolerancia 
propias y estrictamente más conservadoras, aislando los efectos de borde al plano lógico 
del software en la memoria RAM.

AXIOMAS DE ESTABILIDAD FUNCIONAL Y TEOREMAS DE OPERADORES PRESERVADOS:
────────────────────────────────────────────────────────────────────────────────
  [I1] Completitud y Acotación en el Espacio de Banach Basal:
       La Matriz Atómica de Conocimiento (MAC) y la Matriz de Interacción Central (MIC) 
       se modelan como operadores lineales acotados sobre el espacio de Hilbert 
       discreto-continuo $$\mathcal{H}$$, donde toda sucesión de Cauchy de decisiones 
       converge de forma exacta dentro de la clausura del espacio de Banach $$\mathcal{A} \cong \mathcal{B}(\mathcal{H})$$:
       $$\lim_{n, m \to \infty} \|T_n - T_m\|_{\mathrm{op}} = 0 \implies \exists T^* \in \mathcal{A} \quad \text{tal que} \quad \lim_{n \to \infty} \|T_n - T^*\|_{\mathrm{op}} = 0$$

  [I2] Submultiplicatividad Estricta (Aduana de Coherencia de Rayleigh):
       La norma del operador de la composición de dos transiciones de fase está 
       estrictamente acotada por el producto de sus normas individuales en la FPU:
       $$\|XY\|_2 \le \|X\|_2 \cdot \|Y\|_2 \quad \forall X, Y \in \mathcal{A} \quad\big[120\big]$$
       Cualquier deriva que vulnere la cota elástica de Wilkinson activa de forma síncrona:
       $$r_{\mathrm{sub}} = \|XY\|_2 - \|X\|_2 \|Y\|_2 \le \tau_{\mathrm{agent\_soft\_sub}} \quad\big[127, 128\big]$$

  [I3] Univalencia e Invarianza de la Identidad Semántica:
       Si el álgebra contiene un elemento de identidad unitario $$e = I_n$$ que representa 
       la interfaz de despacho neutro, su norma de operador en doble precisión ($$\text{IEEE-754}$$) 
       debe ser numéricamente pura e unitaria [4]:
       $$\|e\|_2 \equiv 1.0 \quad \land \quad X e = e X = X \quad \forall X \in \mathcal{A}$$

  [I4] Isometría Cuántica de la C*-Identidad:
       La inyección de cartuchos semánticos TOON (vitaminas cognitivas) sobre la MAC exige 
       la conservación estricta de la estructura algebraica de C*-álgebra sobre el operador normal:
       $$\|A^\top A\|_2 \equiv \|A\|_2^2 \quad\big[129\big]$$

  [I5] Isometría Espectral y Fórmula Asintótica de Gelfand:
       Para un operador normal de transición $$T$$, la isometría de Gelfand-Naimark exige 
       que su norma espectral coincida de forma biyectiva con su radio espectral $$\rho(T)$$, 
       el cual actúa como el ínfimo de las normas de las potencias consecutivas de Banach:
       $$\|T\|_2 = \rho(T) \equiv \max_{\lambda \in \sigma(T)} |\lambda| = \lim_{k \to \infty} \|T^k\|_2^{1/k} \quad\big[124\big]$$
       El agente re-audita de forma independiente el residuo asintótico de Gelfand para $$k = 5$$:
       $$r_{\mathrm{Gelfand}} = \|T^k\|_2^{1/k} - \rho(T) \ge 0 \quad\big[124, 129\big]$$

  [I6] Convergencia Uniforme de la Serie de Neumann bajo Perturbaciones:
       Dada una transición $$T$$ expuesta a una fluctuación de-normalizada del entorno $$\delta T$$, 
       el operador perturbado $$T + \delta T$$ es incondicionalmente invertible dentro de la variedad 
       si y solo si la perturbación se confina estrictamente dentro del radio de convergencia de la serie:
       $$(T + \delta T)^{-1} = \sum_{k=0}^{\infty} (-1)^k \left( T^{-1} \delta T \right)^k T^{-1}$$
       Lo cual se verifica síncronamente en la FPU mediante el radio de Neumann:
       $$\rho\left( T^{-1} \delta T \right) < 1.0 \quad\big[129\big]$$

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA):
────────────────────────────────────────────────────────────────────────────────
La progresión y el tránsito del Pasaporte de Telemetría se rige por un acoplamiento 
monoidal covariante (Observe ⊣ Orient ⊣ Decide):

  Fase 1 ──► OBSERVE: ADUANA DE NORMAS (Phase1_BanachNormObserver)
             Ingiere el reporte de Phase1NormObservation del motor. Re-audita la 
             submultiplicatividad, la univalencia de la identidad y la C*-identidad 
             bajo tolerancias duras de calibre del agente.
             Entrega: Phase1AgentObservation como precondición de Fase 2.

  Fase 2 ──► ORIENT: ADUANA DE GELFAND-NAIMARK (Phase2_GelfandSpectralOrienter)
             Hereda formalmente la Phase1AgentObservation. Certifica que la 
             isometría de Gelfand y el gap asintótico se mantengan estables.
             Entrega: Phase2AgentOrientation como precondición de Fase.

  Fase 3 ──► DECIDE & ACT: ESTABILIZACIÓN DE NEUMANN (Phase3_HeytingBanachDecider)
             Hereda formalmente la Phase2AgentOrientation. Evalúa la convergencia 
             de la Serie de Neumann. Consolida las severidades parciales mediante la 
             operación Supremo (join, $$\sqcup$$) sobre el retículo de Heyting $$\Omega_3$$:
             $$v_{\mathrm{final}} = v_{\mathrm{sub}} \sqcup v_{\mathrm{Gelfand}} \sqcup v_{\mathrm{Neumann}} \in \Omega_3 = \{\mathrm{COHERENT}, \, \mathrm{DEGRADED}, \, \mathrm{VETOED}\} \quad\big[126\big]$$
             Si $$v_{\mathrm{final}} = \mathrm{VETOED}$$ ($$\top$$), detona síncronamente la excepción 
             'HeytingLatticeVeto' [4, 7]. Purga la memoria RAM en el milisegundo cero y 
             señaliza de manera virtual el puerto lógico de interrupción de hardware 'CrowbarPort' 
             para anular de forma determinista la transacción de la CPU.

  Funtor Completo de Calibre:
             $$\mathcal{Z}_{\mathrm{BanachAgent}} = \Psi_3 \circ \Psi_2 \circ \Psi_1 \circ \mathcal{Z}_{\mathrm{motor}} \quad\big[123, 134\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field, fields, replace, MISSING
from enum import Enum
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
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# Dependencias del ecosistema MIC con fallbacks de aislamiento
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.banach_algebra_auditor import (
        BanachHeytingVerdict,
        BanachAlgebraError,
        SubmultiplicativityViolation,
        IdentityDegeneracyError,
        NeumannSeriesDivergence,
        HeytingLatticeVeto,
        Phase1NormObservation,
        Phase2GelfandOrientation,
        BanachGovernanceState,
        BanachAlgebraAuditor,
        heyting_join,
        classify_relative_defect,
    )
except ImportError:  # aislamiento de laboratorio / tests unitarios
    try:
        from banach_algebra_auditor import (
            BanachHeytingVerdict,
            BanachAlgebraError,
            SubmultiplicativityViolation,
            IdentityDegeneracyError,
            NeumannSeriesDivergence,
            HeytingLatticeVeto,
            Phase1NormObservation,
            Phase2GelfandOrientation,
            BanachGovernanceState,
            BanachAlgebraAuditor,
        )
        try:
            from banach_algebra_auditor import heyting_join, classify_relative_defect
        except ImportError:
            heyting_join = None  # type: ignore[assignment]
            classify_relative_defect = None  # type: ignore[assignment]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "BanachAlgebraAuditorAgent exige el motor banach_algebra_auditor."
        ) from exc

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""


logger = logging.getLogger("MIC.Wisdom.BanachAlgebraAuditorAgent")

__version__: Final[str] = "3.0.0-Banach-OODA-Heyting-Neumann-Gelfand-Crowbar-PhD"


# ══════════════════════════════════════════════════════════════════════════════
# §0. PRIMITIVAS DEL RETÍCULO (fallbacks si el motor legado no las exporta)
# ══════════════════════════════════════════════════════════════════════════════
def _heyting_join(*verdicts: BanachHeytingVerdict) -> BanachHeytingVerdict:
    """Supremo (disyunción interna) en la cadena COHERENT ≼ DEGRADED ≼ VETOED."""
    if heyting_join is not None:
        return heyting_join(*verdicts)
    if not verdicts:
        return BanachHeytingVerdict.COHERENT
    return BanachHeytingVerdict(max(int(v.value) for v in verdicts))


def _classify_relative_defect(
    defect: float,
    scale: float,
    soft_tol: float,
    hard_tol: float,
    floor: float = 0.0,
) -> BanachHeytingVerdict:
    """Clasifica un defecto no negativo relativo a `scale`."""
    if classify_relative_defect is not None:
        return classify_relative_defect(defect, scale, soft_tol, hard_tol, floor)
    denom = max(abs(float(scale)), float(floor), 1.0)
    rel = max(0.0, float(defect)) / denom
    if rel > hard_tol:
        return BanachHeytingVerdict.VETOED
    if rel > soft_tol:
        return BanachHeytingVerdict.DEGRADED
    return BanachHeytingVerdict.COHERENT


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


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE CONTRATO LÓGICO
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_CHANNEL: Final[int] = 14  # identificador lógico del canal de disyunción
_HASH_ROUND_DIGITS: Final[int] = 16

# Bandas de aduana: iguales en orden de magnitud al motor, aplicadas SOBRE
# el residuo crudo (sin descontar Wilkinson). El agente es estrictamente
# más conservador que el motor v3, que sí descuenta γ_n + O(n u) SVD.
_SOFT_SUBMULTIPLICATIVITY_TOL: Final[float] = 1.0e-11
_HARD_SUBMULTIPLICATIVITY_TOL: Final[float] = 1.0e-5
_SOFT_GELFAND_DRIFT_TOL: Final[float] = 1.0e-11
_HARD_GELFAND_DRIFT_TOL: Final[float] = 1.0e-5
_SOFT_IDENTITY_TOL: Final[float] = 1.0e-11
_HARD_IDENTITY_TOL: Final[float] = 1.0e-5
_SOFT_CSTAR_TOL: Final[float] = 1.0e-11
_HARD_CSTAR_TOL: Final[float] = 1.0e-5
_AGENT_SPECTRAL_MARGIN: Final[float] = 1.0e-12
_NEUMANN_PROXIMITY_FACTOR: Final[float] = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# §B. EXCEPCIONES FUNCIONALES DEL SOBERANO
# ══════════════════════════════════════════════════════════════════════════════
class BanachAgentError(Exception):
    """Excepción raíz para violaciones detectadas por el agente soberano."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.cause = cause


class SubmultiplicativityCollapse(BanachAgentError):
    """Colapso de [I2]: ‖XY‖₂ > ‖X‖₂·‖Y‖₂ fuera de la banda de aduana."""


class IdentityUnivalenceCollapse(BanachAgentError):
    """Colapso de [I3]: ‖I_n‖₂ se desvía de 1 más allá de la tolerancia dura."""


class CStarIdentityCollapse(BanachAgentError):
    """Colapso de [I4]: ‖AᵀA‖₂ ≠ ‖A‖₂² fuera de tolerancia."""


class GelfandIsometryDrift(BanachAgentError):
    """Colapso de [I6]: ‖Tᵏ‖₂^{1/k} < ρ(T) de forma numéricamente imposible."""


class NeumannSeriesDivergenceAgent(BanachAgentError):
    """Colapso de [I7]: ρ(T⁻¹ dT) ≥ 1 − margen, o T no es invertible."""


class DualSourceIncoherence(BanachAgentError):
    """El artefacto del motor contradice sus propios invariantes declarados."""


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DE FASES ANIDADAS (contratos categóricos de handoff)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1AgentObservation:
    """
    Artefacto terminal de la FASE 1 del agente (Observe).

    Es la *única* precondición estricta de la FASE 2 del agente: un objeto
    del tipo Obs₁^Ψ que certifica (o degrada/veta) los axiomas [I2]–[I4]
    re-evaluados con las bandas de aduana, y compara el juicio propio con
    el del motor (dual-source).
    """

    observation: Phase1NormObservation
    verdict: BanachHeytingVerdict
    relative_violation: float
    identity_relative_defect: float = 0.0
    cstar_relative_defect: float = 0.0
    dual_source_discrepancy: float = 0.0
    is_submultiplicative: bool = True
    is_identity_coherent: bool = True
    is_cstar_coherent: bool = True
    motor_verdict: Optional[BanachHeytingVerdict] = None
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Phase2AgentOrientation:
    """
    Artefacto terminal de la FASE 2 del agente (Orient).

    Precondición estricta de la FASE 3. Contiene la re-auditoría de la
    isometría de Gelfand, el gap espectral y la alineación de normalidad,
    más el sello de Fase 1 anidado.
    """

    phase1_agent_observation: Phase1AgentObservation
    orientation: Phase2GelfandOrientation
    verdict: BanachHeytingVerdict
    relative_drift: float
    spectral_gap_defect: float = 0.0
    normality_alignment_defect: float = 0.0
    dual_source_discrepancy: float = 0.0
    is_gelfand_isometric: bool = True
    is_gap_coherent: bool = True
    motor_verdict: Optional[BanachHeytingVerdict] = None
    diagnostic_atoms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AgentCrowbarReport:
    """Certificado de actuación del disyuntor lógico (Fase 3)."""

    action: "CrowbarAction"
    triggered: bool
    channel: int
    port_typename: str
    diagnostic: str = ""


@dataclass(frozen=True, slots=True)
class BanachAgentGovernanceState:
    """
    Certificado soberano (envoltorio de Fase 3).

    Distingue el estado *crudo* del motor del estado *reconstruido* con
    el veredicto de aduana, y porta el informe Crowbar y la huella propia.
    """

    phase2_agent_orientation: Phase2AgentOrientation
    motor_state: BanachGovernanceState
    governance_state: BanachGovernanceState
    crowbar: AgentCrowbarReport
    final_verdict: BanachHeytingVerdict
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
    Puerto lógico del disyuntor.

    La capa de sabiduría no posee MMIO: el puerto es un morfismo de
    actuación inyectado por el estrato de infraestructura. `True` significa
    «desconexión física confirmada por el actuador».
    """

    def trigger_physical_disconnection(self) -> bool:
        ...


class NullCrowbarPort:
    """Implementación forense no-op: sella el intento en el log y devuelve False."""

    def trigger_physical_disconnection(self) -> bool:
        logger.info(
            "[CROWBAR-DRY] Actuador nulo invocado (canal lógico %d). "
            "Sello de desconexión registrado; no hay efecto físico.",
            _CROWBAR_CHANNEL,
        )
        return False


class LoggingCrowbarPort:
    """Actuador que eleva un CRITICAL y trata el log como acuse de recibo."""

    def trigger_physical_disconnection(self) -> bool:
        logger.critical(
            "[CROWBAR-LOG] Disyunción lógica solicitada en canal %d.",
            _CROWBAR_CHANNEL,
        )
        return True


class CompositeCrowbarPort:
    """Abanico de puertos: se invocan en orden; el OR lógico es el acuse."""

    def __init__(self, ports: Sequence[CrowbarPort]) -> None:
        if not ports:
            raise ValueError("CompositeCrowbarPort exige al menos un puerto.")
        self._ports: Tuple[CrowbarPort, ...] = tuple(ports)

    def trigger_physical_disconnection(self) -> bool:
        triggered = False
        for port in self._ports:
            try:
                triggered = bool(port.trigger_physical_disconnection()) or triggered
            except Exception:  # el fallo de un actuador no silencia a los demás
                logger.exception(
                    "[CROWBAR-COMPOSITE] Fallo en %s; se continúa el abanico.",
                    type(port).__name__,
                )
        return triggered


class _AdvancedNumericalGuard:
    """
    Capa de saneamiento numérico y validación estructural de entradas.

    Replica, en la aduana, los invariantes de pertenencia a M_n(ℝ) que el
    motor exige, de modo que un par mal formado nunca alcance Z_motor.
    """

    @staticmethod
    def _assert_square_operator(X: NDArray[np.float64], name: str) -> None:
        """Valida matriz cuadrada, bidimensional, de dimensión positiva y finita."""
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"El operador {name} debe ser numpy.ndarray, no {type(X)!r}."
            )
        if X.ndim != 2:
            raise ValueError(
                f"El operador {name} debe ser bidimensional; ndim={X.ndim}."
            )
        rows, cols = int(X.shape[0]), int(X.shape[1])
        if rows != cols or rows == 0:
            raise ValueError(
                f"El operador {name} debe ser cuadrado y de dimensión positiva: "
                f"{X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(
                f"FPU Error: el operador {name} contiene valores no finitos (NaN/Inf)."
            )

    def _sanitize_input(
        self,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Saneamiento preventivo: forma, finitud y conformidad dimensional.

        No materializa copia de trabajo (el motor v3 ya inmerge en float64
        C-contiguo); sólo certifica el contrato de entrada del soberano.
        """
        self._assert_square_operator(T, "T")
        self._assert_square_operator(dT, "dT")
        if T.shape != dT.shape:
            raise ValueError(
                f"Los operadores T y dT deben compartir dimensión: "
                f"{T.shape} != {dT.shape}."
            )
        return T, dT


# ══════════════════════════════════════════════════════════════════════════════
# §E. FASE 1 — OBSERVACIÓN DE NORMAS (ADUANA DE [I2]–[I4])
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_BanachNormObserver(_AdvancedNumericalGuard):
    r"""
    Fase 1 (Observe): intercepta `Phase1NormObservation` del motor y
    re-audita submultiplicatividad, univalencia de I_n e identidad C*.

    Morfismo Ψ₁ : Obs₁^motor → Obs₁^Ψ.

    Descomposición granular (cada método es un juicio atómico; el sello
    terminal `_observe_norm_coherence` es el único objeto que la Fase 2
    acepta):

        ingesta motor → [I2] crudo → [I3] identidad → [I4] C*
                     → dual-source → join de Heyting
                     → sello terminal Obs₁^Ψ
    """

    def __init__(
        self,
        soft_submultiplicativity_tol: float = _SOFT_SUBMULTIPLICATIVITY_TOL,
        hard_submultiplicativity_tol: float = _HARD_SUBMULTIPLICATIVITY_TOL,
        soft_identity_tol: float = _SOFT_IDENTITY_TOL,
        hard_identity_tol: float = _HARD_IDENTITY_TOL,
        soft_cstar_tol: float = _SOFT_CSTAR_TOL,
        hard_cstar_tol: float = _HARD_CSTAR_TOL,
    ) -> None:
        self._soft_sub = self._validate_tol_pair(
            soft_submultiplicativity_tol, hard_submultiplicativity_tol, "submult"
        )[0]
        self._hard_sub = float(hard_submultiplicativity_tol)
        self._soft_id, self._hard_id = self._validate_tol_pair(
            soft_identity_tol, hard_identity_tol, "identity"
        )
        self._soft_cstar, self._hard_cstar = self._validate_tol_pair(
            soft_cstar_tol, hard_cstar_tol, "cstar"
        )

    @staticmethod
    def _validate_tol_pair(
        soft: float,
        hard: float,
        name: str,
    ) -> Tuple[float, float]:
        s, h = float(soft), float(hard)
        if not (0.0 < s <= h):
            raise ValueError(
                f"Se exige 0 < soft_{name} ≤ hard_{name}; recibido soft={s}, hard={h}."
            )
        return s, h

    # ── E.1  Ingesta del sello del motor ─────────────────────────────────
    def _ingest_motor_phase1(
        self,
        observation: Phase1NormObservation,
    ) -> Phase1NormObservation:
        """
        Valida que el artefacto del motor sea un habitante legal de Obs₁.

        No se repara un VETOED del motor: el join monótono lo transportará.
        """
        if not isinstance(observation, Phase1NormObservation):
            raise TypeError(
                "Fase 1 del agente exige Phase1NormObservation del motor; "
                f"recibido {type(observation)!r}."
            )
        for label in ("norm_x", "norm_y", "norm_composed", "identity_norm"):
            _finite_nonneg(getattr(observation, label), name=f"Obs₁.{label}")
        residual = float(observation.submultiplicativity_residual)
        if not np.isfinite(residual):
            raise ValueError(
                f"Obs₁.submultiplicativity_residual no es finito: {residual!r}."
            )
        if not isinstance(observation.verdict, BanachHeytingVerdict):
            raise TypeError("Obs₁.verdict no pertenece al retículo de Heyting.")
        dimension = _dto_field(observation, "dimension", None)
        if dimension is not None and int(dimension) <= 0:
            raise ValueError(f"Obs₁.dimension no positiva: {dimension}.")
        return observation

    # ── E.2  Axioma [I2] : residuo crudo de submultiplicatividad ─────────
    def _relative_submultiplicative_violation(
        self,
        observation: Phase1NormObservation,
    ) -> float:
        """
        Violación relativa cruda (sin descontar Wilkinson):

            r₊ / max(1, ‖X‖₂‖Y‖₂, ‖XY‖₂),
            r₊ := max(0, ‖XY‖₂ − ‖X‖₂‖Y‖₂).

        El motor v3 descuenta γ_n + O(n u) antes de clasificar; la aduana
        no lo hace. Un residuo que el motor perdona como redondeo puede
        degradar aquí.
        """
        residual = float(observation.submultiplicativity_residual)
        scale = max(
            1.0,
            float(observation.norm_x) * float(observation.norm_y),
            float(observation.norm_composed),
        )
        return _clamp_nonneg(residual) / scale

    def _classify_submultiplicativity(
        self,
        relative_violation: float,
    ) -> BanachHeytingVerdict:
        return _classify_relative_defect(
            defect=relative_violation,
            scale=1.0,
            soft_tol=self._soft_sub,
            hard_tol=self._hard_sub,
            floor=1.0,
        )

    # ── E.3  Axioma [I3] : univalencia de I_n ────────────────────────────
    def _classify_identity_univalence(
        self,
        observation: Phase1NormObservation,
    ) -> Tuple[float, BanachHeytingVerdict]:
        residual = abs(float(observation.identity_residual))
        verdict = _classify_relative_defect(
            defect=residual,
            scale=1.0,
            soft_tol=self._soft_id,
            hard_tol=self._hard_id,
            floor=1.0,
        )
        return residual, verdict

    # ── E.4  Axioma [I4] : identidad C* ──────────────────────────────────
    def _classify_cstar_coherence(
        self,
        observation: Phase1NormObservation,
    ) -> Tuple[float, BanachHeytingVerdict]:
        """
        Si el motor no exporta residuos C*, el átomo se declara COHERENT
        (ausencia de evidencia ≠ evidencia de ausencia) y se anota.
        """
        rx = _safe_float(_dto_field(observation, "star_identity_residual_x", 0.0))
        ry = _safe_float(_dto_field(observation, "star_identity_residual_y", 0.0))
        if (
            not hasattr(observation, "star_identity_residual_x")
            and not hasattr(observation, "star_identity_residual_y")
        ):
            return 0.0, BanachHeytingVerdict.COHERENT
        scale = max(
            1.0,
            float(observation.norm_x) ** 2,
            float(observation.norm_y) ** 2,
        )
        defect = max(rx, ry)
        verdict = _classify_relative_defect(
            defect=defect,
            scale=scale,
            soft_tol=self._soft_cstar,
            hard_tol=self._hard_cstar,
            floor=1.0,
        )
        return defect / max(scale, 1.0), verdict

    # ── E.5  Dual-source : aduana vs. motor ──────────────────────────────
    def _dual_source_phase1(
        self,
        agent_verdict: BanachHeytingVerdict,
        motor_verdict: BanachHeytingVerdict,
    ) -> Tuple[float, BanachHeytingVerdict]:
        """
        Discrepancia ordinal |v_Ψ − v_motor| / 2 ∈ [0, 1].

        Una discrepancia de dos niveles (COHERENT vs VETOED) se veta:
        denuncia o bien un bug de transporte o una corrupción del DTO.
        Un nivel de diferencia es esperable (aduana más estricta) y se
        degrada informativamente, sin imponer un veto adicional.
        """
        delta = abs(int(agent_verdict.value) - int(motor_verdict.value))
        discrepancy = float(delta) / 2.0
        if delta >= 2:
            return discrepancy, BanachHeytingVerdict.VETOED
        if delta == 1:
            return discrepancy, BanachHeytingVerdict.DEGRADED
        return 0.0, BanachHeytingVerdict.COHERENT

    # ── E.6  SELLO TERMINAL DE LA FASE 1 ─────────────────────────────────
    def _observe_norm_coherence(
        self,
        observation: Phase1NormObservation,
    ) -> Phase1AgentObservation:
        """
        Ψ₁ — morfismo terminal de la Fase 1 del agente.

        Re-evalúa sobre el sello del motor:
          • [I2]  ‖XY‖₂ ≤ ‖X‖₂·‖Y‖₂     (residuo crudo de aduana);
          • [I3]  ‖I_n‖₂ = 1;
          • [I4]  ‖AᵀA‖₂ = ‖A‖₂²         (si el motor lo exporta);
          • dual-source contra `observation.verdict`;
          • join monótono con el veredicto del motor (la aduana nunca
            «cura» una degradación ya declarada).

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₁^Ψ := Phase1AgentObservation. El juicio

            Ψ₁(Obs₁^motor) ∈ Obs₁^Ψ

        es un objeto inmutable del topos de Banach y constituye, por
        contrato categórico, la *unidad de arranque* de la Fase 2:

            Ψ₂  se aplica únicamente a  (Ψ₁(Obs₁^motor), Obs₂^motor),
            y su primer método `_ingest_phase1_agent_precondition` es
            la continuación lógica y tipada de este sello.

        Cualquier consumidor que no pase por este método viola la
        adjunción Observe ⊣ Orient del soberano.
        """
        obs = self._ingest_motor_phase1(observation)

        relative_violation = self._relative_submultiplicative_violation(obs)
        sub_verdict = self._classify_submultiplicativity(relative_violation)
        id_defect, id_verdict = self._classify_identity_univalence(obs)
        cstar_defect, cstar_verdict = self._classify_cstar_coherence(obs)

        agent_atoms_join = _heyting_join(sub_verdict, id_verdict, cstar_verdict)
        discrepancy, dual_verdict = self._dual_source_phase1(agent_atoms_join, obs.verdict)
        combined = _heyting_join(agent_atoms_join, dual_verdict, obs.verdict)

        atoms: Tuple[str, ...] = (
            f"r_sub_rel={relative_violation:.6e}",
            f"id_rel={id_defect:.6e}",
            f"C*_rel={cstar_defect:.6e}",
            f"dual=Δ{discrepancy:.1f}",
            f"motor={obs.verdict.name}",
            f"join={combined.name}",
        )

        if combined == BanachHeytingVerdict.VETOED:
            logger.error("[BANACH-AGENT:Ψ₁] Observación VETOED. %s", " | ".join(atoms))
        elif combined == BanachHeytingVerdict.DEGRADED:
            logger.warning("[BANACH-AGENT:Ψ₁] Observación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[BANACH-AGENT:Ψ₁] Observación COHERENT.")

        # Sello terminal: este return ES el morfismo de arranque de la Fase 2.
        return Phase1AgentObservation(
            observation=obs,
            verdict=combined,
            relative_violation=relative_violation,
            identity_relative_defect=id_defect,
            cstar_relative_defect=cstar_defect,
            dual_source_discrepancy=discrepancy,
            is_submultiplicative=sub_verdict != BanachHeytingVerdict.VETOED,
            is_identity_coherent=id_verdict != BanachHeytingVerdict.VETOED,
            is_cstar_coherent=cstar_verdict != BanachHeytingVerdict.VETOED,
            motor_verdict=obs.verdict,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §F. FASE 2 — ORIENTACIÓN ESPECTRAL DE GELFAND (ADUANA DE [I5]–[I6])
#     Continuación formal del sello terminal de la Fase 1 del agente.
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_GelfandSpectralOrienter(Phase1_BanachNormObserver):
    r"""
    Fase 2 (Orient): certifica la isometría de Gelfand–Naimark y el gap.

    Morfismo Ψ₂ : Obs₁^Ψ × Obs₂^motor → Obs₂^Ψ.

    El primer método de esta fase, `_ingest_phase1_agent_precondition`,
    es la *continuación tipada* de `_observe_norm_coherence`. No existe
    camino legal hacia Gelfand que no atraviese ese ingest.
    """

    def __init__(
        self,
        soft_submultiplicativity_tol: float = _SOFT_SUBMULTIPLICATIVITY_TOL,
        hard_submultiplicativity_tol: float = _HARD_SUBMULTIPLICATIVITY_TOL,
        soft_identity_tol: float = _SOFT_IDENTITY_TOL,
        hard_identity_tol: float = _HARD_IDENTITY_TOL,
        soft_cstar_tol: float = _SOFT_CSTAR_TOL,
        hard_cstar_tol: float = _HARD_CSTAR_TOL,
        soft_gelfand_tol: float = _SOFT_GELFAND_DRIFT_TOL,
        hard_gelfand_tol: float = _HARD_GELFAND_DRIFT_TOL,
    ) -> None:
        super().__init__(
            soft_submultiplicativity_tol=soft_submultiplicativity_tol,
            hard_submultiplicativity_tol=hard_submultiplicativity_tol,
            soft_identity_tol=soft_identity_tol,
            hard_identity_tol=hard_identity_tol,
            soft_cstar_tol=soft_cstar_tol,
            hard_cstar_tol=hard_cstar_tol,
        )
        self._soft_gelfand, self._hard_gelfand = self._validate_tol_pair(
            soft_gelfand_tol, hard_gelfand_tol, "gelfand"
        )

    # ── F.1  INICIO DE FASE 2 = continuación del sello Ψ₁ ────────────────
    def _ingest_phase1_agent_precondition(
        self,
        phase1_agent_obs: Phase1AgentObservation,
    ) -> Phase1AgentObservation:
        """
        Continuación formal de `Phase1_BanachNormObserver._observe_norm_coherence`.

        Teorema de handoff (Obs₁^Ψ ↪ Fase 2)
        ────────────────────────────────────
        Hipótesis: `phase1_agent_obs` es el valor de retorno de
        `_observe_norm_coherence` (objeto congelado de tipo
        Phase1AgentObservation).

        Tesis: el objeto es una precondición *habitable* de Ψ₂. Se verifica:
          (i)   tipado e inmutabilidad;
          (ii)  re-ingesta del Phase1NormObservation anidado;
          (iii) magnitudes relativas finitas y no negativas;
          (iv)  el veredicto vive en el retículo {0,1,2}.

        No se repara un VETOED de Fase 1: el join monótono lo transporta.
        Este método es el único puerto de entrada de la Fase 2.
        """
        if not isinstance(phase1_agent_obs, Phase1AgentObservation):
            raise TypeError(
                "Fase 2 del agente exige el sello terminal de Fase 1 "
                f"(Phase1AgentObservation); recibido {type(phase1_agent_obs)!r}."
            )
        self._ingest_motor_phase1(phase1_agent_obs.observation)
        _finite_nonneg(
            phase1_agent_obs.relative_violation,
            name="Obs₁^Ψ.relative_violation",
        )
        if not isinstance(phase1_agent_obs.verdict, BanachHeytingVerdict):
            raise TypeError("Obs₁^Ψ.verdict no pertenece al retículo de Heyting.")
        logger.debug(
            "[BANACH-AGENT:Ψ₂←Ψ₁] Ingestión de Obs₁^Ψ (verdict=%s).",
            phase1_agent_obs.verdict.name,
        )
        return phase1_agent_obs

    # ── F.2  Ingesta del sello espectral del motor ───────────────────────
    def _ingest_motor_phase2(
        self,
        orientation: Phase2GelfandOrientation,
        phase1_agent_obs: Phase1AgentObservation,
    ) -> Phase2GelfandOrientation:
        """Valida Obs₂^motor y su enlace con el Obs₁ ya auditado."""
        if not isinstance(orientation, Phase2GelfandOrientation):
            raise TypeError(
                "Fase 2 del agente exige Phase2GelfandOrientation del motor; "
                f"recibido {type(orientation)!r}."
            )
        rho = _finite_nonneg(orientation.spectral_radius, name="Obs₂.spectral_radius")
        if not orientation.gelfand_bounds:
            raise ValueError("Obs₂.gelfand_bounds no puede ser vacío.")
        last = float(orientation.gelfand_bounds[-1])
        if not np.isfinite(last) or last < 0.0:
            raise ValueError(f"Obs₂.gelfand_bounds[-1] inválido: {last!r}.")
        if not isinstance(orientation.verdict, BanachHeytingVerdict):
            raise TypeError("Obs₂.verdict no pertenece al retículo de Heyting.")

        nested = orientation.phase1_observation
        ingested = phase1_agent_obs.observation
        if nested is not ingested:
            # Igualdad estructural mínima: residuos y normas deben coincidir.
            for attr in (
                "submultiplicativity_residual",
                "identity_residual",
                "norm_x",
                "norm_y",
                "norm_composed",
            ):
                left = float(getattr(nested, attr))
                right = float(getattr(ingested, attr))
                if not np.isfinite(left) or abs(left - right) > 0.0 and abs(left - right) > 8.0 * _MACHINE_EPS * max(1.0, abs(left), abs(right)):
                    # tolerancia holgada: 8u relativo, o mismatch grosero
                    scale = max(1.0, abs(left), abs(right))
                    if abs(left - right) > 64.0 * _MACHINE_EPS * scale:
                        raise DualSourceIncoherence(
                            f"Obs₂.phase1_observation.{attr}={left!r} no coincide "
                            f"con el sello de Fase 1 ({right!r})."
                        )
        _ = rho
        return orientation

    # ── F.3  Isometría de Gelfand (dirección prohibida) ──────────────────
    def _relative_gelfand_drift(
        self,
        orientation: Phase2GelfandOrientation,
    ) -> float:
        """
        Drift relativo de la dirección prohibida:

            max(0, ρ − ‖T^K‖₂^{1/K}) / max(1, ρ, g_K).

        En un álgebra de Banach, ρ(T) ≤ ‖Tᵏ‖^{1/k} para todo k. Un drift
        negativo (g_K < ρ) sólo puede ser redondeo o corrupción.
        """
        drift = float(orientation.gelfand_residual)
        rho = float(orientation.spectral_radius)
        last_bound = float(orientation.gelfand_bounds[-1])
        scale = max(1.0, rho, last_bound)
        return _clamp_nonneg(-drift) / scale

    def _classify_gelfand_isometry(
        self,
        relative_drift: float,
    ) -> BanachHeytingVerdict:
        return _classify_relative_defect(
            defect=relative_drift,
            scale=1.0,
            soft_tol=self._soft_gelfand,
            hard_tol=self._hard_gelfand,
            floor=1.0,
        )

    # ── F.4  Gap espectral [I5] ──────────────────────────────────────────
    def _classify_spectral_gap(
        self,
        orientation: Phase2GelfandOrientation,
    ) -> Tuple[float, BanachHeytingVerdict]:
        """
        gap := ‖T‖₂ − ρ(T) ≥ 0. Un gap negativo es numéricamente imposible.
        Si el motor legado no exporta el campo, el átomo es COHERENT.
        """
        gap = _dto_field(orientation, "spectral_norm_gap", None)
        op_norm = _dto_field(orientation, "operator_norm_t", None)
        if gap is None:
            return 0.0, BanachHeytingVerdict.COHERENT
        gap_f = float(gap)
        scale = max(1.0, float(op_norm) if op_norm is not None else 0.0, float(orientation.spectral_radius))
        impossible = _clamp_nonneg(-gap_f)
        verdict = _classify_relative_defect(
            defect=impossible,
            scale=scale,
            soft_tol=self._soft_gelfand,
            hard_tol=self._hard_gelfand,
            floor=1.0,
        )
        return impossible / max(scale, 1.0), verdict

    # ── F.5  Alineación normalidad ↔ gap ─────────────────────────────────
    def _classify_normality_alignment(
        self,
        orientation: Phase2GelfandOrientation,
    ) -> Tuple[float, BanachHeytingVerdict]:
        """
        Si T es numéricamente normal, ‖T‖₂ ≃ ρ(T). Un gap grande junto a
        `is_numerically_normal=True` es incoherencia de diagnóstico
        (no un veto algebraico duro: se degrada).
        """
        is_normal = bool(_dto_field(orientation, "is_numerically_normal", False))
        gap = _dto_field(orientation, "spectral_norm_gap", None)
        op_norm = _dto_field(orientation, "operator_norm_t", None)
        if not is_normal or gap is None:
            return 0.0, BanachHeytingVerdict.COHERENT
        scale = max(1.0, float(op_norm) if op_norm is not None else 0.0)
        rel_gap = _clamp_nonneg(float(gap)) / scale
        # Umbral holgado: la normalidad numérica del motor usa hard_tol·‖T‖_F²,
        # el gap puede ser O(u κ). Sólo se degrada, nunca se veta aquí.
        if rel_gap > self._hard_gelfand:
            return rel_gap, BanachHeytingVerdict.DEGRADED
        return rel_gap, BanachHeytingVerdict.COHERENT

    # ── F.6  Dual-source espectral ───────────────────────────────────────
    def _dual_source_phase2(
        self,
        agent_verdict: BanachHeytingVerdict,
        motor_verdict: BanachHeytingVerdict,
    ) -> Tuple[float, BanachHeytingVerdict]:
        delta = abs(int(agent_verdict.value) - int(motor_verdict.value))
        discrepancy = float(delta) / 2.0
        if delta >= 2:
            return discrepancy, BanachHeytingVerdict.VETOED
        if delta == 1:
            return discrepancy, BanachHeytingVerdict.DEGRADED
        return 0.0, BanachHeytingVerdict.COHERENT

    # ── F.7  SELLO TERMINAL DE LA FASE 2 ─────────────────────────────────
    def _orient_gelfand_isometry(
        self,
        phase1_agent_obs: Phase1AgentObservation,
        orientation: Phase2GelfandOrientation,
    ) -> Phase2AgentOrientation:
        """
        Ψ₂ — morfismo terminal de la Fase 2 del agente.

        Verifica
            ρ(T)  ≤  ‖T^K‖₂^{1/K}     (dirección de Gelfand),
            ρ(T)  ≤  ‖T‖₂             (cota espectral),
        y la alineación «normal ⇒ gap ≃ 0». El join con Obs₁^Ψ y con
        Obs₂^motor transporta cualquier degradación previa.

        Definición formal del artefacto emitido
        ───────────────────────────────────────
        Sea Obs₂^Ψ := Phase2AgentOrientation. El juicio

            Ψ₂(Ψ₁(Obs₁^motor), Obs₂^motor) ∈ Obs₂^Ψ

        es la *unidad de arranque* de la Fase 3: su primer método
        `_ingest_phase2_agent_precondition` es la continuación lógica
        de este sello.
        """
        obs1 = self._ingest_phase1_agent_precondition(phase1_agent_obs)
        obs2 = self._ingest_motor_phase2(orientation, obs1)

        relative_drift = self._relative_gelfand_drift(obs2)
        gelfand_verdict = self._classify_gelfand_isometry(relative_drift)
        gap_defect, gap_verdict = self._classify_spectral_gap(obs2)
        align_defect, align_verdict = self._classify_normality_alignment(obs2)

        agent_atoms_join = _heyting_join(gelfand_verdict, gap_verdict, align_verdict)
        discrepancy, dual_verdict = self._dual_source_phase2(agent_atoms_join, obs2.verdict)
        combined = _heyting_join(
            obs1.verdict,
            agent_atoms_join,
            dual_verdict,
            obs2.verdict,
        )

        atoms: Tuple[str, ...] = (
            f"drift_rel={relative_drift:.6e}",
            f"gap_rel={gap_defect:.6e}",
            f"align={align_defect:.6e}",
            f"dual=Δ{discrepancy:.1f}",
            f"motor={obs2.verdict.name}",
            f"join={combined.name}",
        )

        if combined == BanachHeytingVerdict.VETOED:
            logger.error("[BANACH-AGENT:Ψ₂] Orientación VETOED. %s", " | ".join(atoms))
        elif combined == BanachHeytingVerdict.DEGRADED:
            logger.warning("[BANACH-AGENT:Ψ₂] Orientación DEGRADED. %s", " | ".join(atoms))
        else:
            logger.info("[BANACH-AGENT:Ψ₂] Orientación COHERENT.")

        # Sello terminal: este return ES el morfismo de arranque de la Fase 3.
        return Phase2AgentOrientation(
            phase1_agent_observation=obs1,
            orientation=obs2,
            verdict=combined,
            relative_drift=relative_drift,
            spectral_gap_defect=gap_defect,
            normality_alignment_defect=align_defect,
            dual_source_discrepancy=discrepancy,
            is_gelfand_isometric=gelfand_verdict != BanachHeytingVerdict.VETOED,
            is_gap_coherent=gap_verdict != BanachHeytingVerdict.VETOED,
            motor_verdict=obs2.verdict,
            diagnostic_atoms=atoms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §G. FASE 3 — DECIDE & ACT (SUPREMO DE HEYTING + CROWBAR + SELLO FORENSE)
#     Continuación formal del sello terminal de la Fase 2 del agente.
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingBanachDecider(Phase2_GelfandSpectralOrienter):
    r"""
    Fase 3 (Decide & Act): join terminal, actuación Crowbar y trazabilidad.

    Morfismo Ψ₃ : Obs₂^Ψ × Gov_motor × CrowbarPort → Gov_Ψ.

    Consume la orientación de Fase 2 y el certificado del motor, decide
    el supremo en Ω y, si el retículo colapsa, dispara el disyuntor.
    """

    def __init__(
        self,
        soft_submultiplicativity_tol: float = _SOFT_SUBMULTIPLICATIVITY_TOL,
        hard_submultiplicativity_tol: float = _HARD_SUBMULTIPLICATIVITY_TOL,
        soft_identity_tol: float = _SOFT_IDENTITY_TOL,
        hard_identity_tol: float = _HARD_IDENTITY_TOL,
        soft_cstar_tol: float = _SOFT_CSTAR_TOL,
        hard_cstar_tol: float = _HARD_CSTAR_TOL,
        soft_gelfand_tol: float = _SOFT_GELFAND_DRIFT_TOL,
        hard_gelfand_tol: float = _HARD_GELFAND_DRIFT_TOL,
        spectral_margin: float = _AGENT_SPECTRAL_MARGIN,
    ) -> None:
        super().__init__(
            soft_submultiplicativity_tol=soft_submultiplicativity_tol,
            hard_submultiplicativity_tol=hard_submultiplicativity_tol,
            soft_identity_tol=soft_identity_tol,
            hard_identity_tol=hard_identity_tol,
            soft_cstar_tol=soft_cstar_tol,
            hard_cstar_tol=hard_cstar_tol,
            soft_gelfand_tol=soft_gelfand_tol,
            hard_gelfand_tol=hard_gelfand_tol,
        )
        margin = float(spectral_margin)
        if not (0.0 <= margin < 1.0):
            raise ValueError(f"spectral_margin debe vivir en [0, 1); recibido {margin}.")
        self._spectral_margin = margin

    # ── G.1  INICIO DE FASE 3 = continuación del sello Ψ₂ ────────────────
    def _ingest_phase2_agent_precondition(
        self,
        phase2_agent_orientation: Phase2AgentOrientation,
    ) -> Phase2AgentOrientation:
        """
        Continuación formal de
        `Phase2_GelfandSpectralOrienter._orient_gelfand_isometry`.

        Teorema de handoff (Obs₂^Ψ ↪ Fase 3)
        ────────────────────────────────────
        Hipótesis: `phase2_agent_orientation` es el valor de retorno de
        `_orient_gelfand_isometry`.

        Tesis: Obs₂^Ψ es habitable. Se exige:
          (i)   tipado Phase2AgentOrientation;
          (ii)  re-ingesta de Obs₁^Ψ (y, transitivamente, de Obs₁^motor);
          (iii) re-ingesta de Obs₂^motor;
          (iv)  drift relativo finito y no negativo.

        El veredicto de Obs₂^Ψ se transporta monótonamente; no se repara.
        """
        if not isinstance(phase2_agent_orientation, Phase2AgentOrientation):
            raise TypeError(
                "Fase 3 del agente exige el sello terminal de Fase 2 "
                f"(Phase2AgentOrientation); recibido {type(phase2_agent_orientation)!r}."
            )
        obs1 = self._ingest_phase1_agent_precondition(
            phase2_agent_orientation.phase1_agent_observation
        )
        self._ingest_motor_phase2(phase2_agent_orientation.orientation, obs1)
        _finite_nonneg(
            phase2_agent_orientation.relative_drift,
            name="Obs₂^Ψ.relative_drift",
        )
        if not isinstance(phase2_agent_orientation.verdict, BanachHeytingVerdict):
            raise TypeError("Obs₂^Ψ.verdict no pertenece al retículo de Heyting.")
        logger.debug(
            "[BANACH-AGENT:Ψ₃←Ψ₂] Ingestión de Obs₂^Ψ (verdict=%s).",
            phase2_agent_orientation.verdict.name,
        )
        return phase2_agent_orientation

    # ── G.2  Ingesta del certificado del motor ───────────────────────────
    def _ingest_motor_governance(
        self,
        motor_state: BanachGovernanceState,
        phase2_agent_orientation: Phase2AgentOrientation,
    ) -> BanachGovernanceState:
        """Valida Gov_motor y su enlace con la orientación ya auditada."""
        if not isinstance(motor_state, BanachGovernanceState):
            raise TypeError(
                "Fase 3 del agente exige BanachGovernanceState del motor; "
                f"recibido {type(motor_state)!r}."
            )
        if not isinstance(motor_state.final_verdict, BanachHeytingVerdict):
            raise TypeError("Gov_motor.final_verdict no pertenece al retículo.")
        nested = motor_state.phase2_orientation
        oriented = phase2_agent_orientation.orientation
        if nested is not oriented:
            for attr in ("spectral_radius", "gelfand_residual"):
                left = float(getattr(nested, attr))
                right = float(getattr(oriented, attr))
                scale = max(1.0, abs(left), abs(right))
                if abs(left - right) > 64.0 * _MACHINE_EPS * scale:
                    raise DualSourceIncoherence(
                        f"Gov_motor.phase2_orientation.{attr}={left!r} no coincide "
                        f"con el sello de Fase 2 ({right!r})."
                    )
        return motor_state

    # ── G.3  Re-clasificación de Neumann en la aduana ────────────────────
    def _classify_neumann_agent(
        self,
        motor_state: BanachGovernanceState,
    ) -> BanachHeytingVerdict:
        """
        Relee ρ(W) y la singularidad con el margen de aduana.

        El motor ya clasificó; aquí se aplica el mismo criterio con el
        margen del agente y se degrada si ρ está en la corona
        (1 − λ·margen, 1 − margen], λ = _NEUMANN_PROXIMITY_FACTOR.
        """
        singular = bool(motor_state.singular_base)
        rho = float(motor_state.neumann_radius)
        stable_flag = bool(motor_state.is_neumann_stable)
        threshold = 1.0 - self._spectral_margin
        finite_rho = bool(np.isfinite(rho))
        is_stable = (not singular) and finite_rho and (rho < threshold)

        if singular or not is_stable:
            return BanachHeytingVerdict.VETOED
        if not stable_flag and is_stable:
            # El motor vetó y la aduana no: dual-source, se respeta el veto.
            return BanachHeytingVerdict.VETOED
        proximity = 1.0 - rho
        if proximity < _NEUMANN_PROXIMITY_FACTOR * self._spectral_margin:
            return BanachHeytingVerdict.DEGRADED
        return BanachHeytingVerdict.COHERENT

    # ── G.4  Supremo terminal ────────────────────────────────────────────
    def _compose_terminal_verdict(
        self,
        phase2_agent_orientation: Phase2AgentOrientation,
        motor_state: BanachGovernanceState,
        neumann_verdict: BanachHeytingVerdict,
    ) -> BanachHeytingVerdict:
        """Join monótono: aduana P1–P2 + relectura Neumann + veredicto motor."""
        return _heyting_join(
            phase2_agent_orientation.verdict,
            neumann_verdict,
            motor_state.final_verdict,
        )

    # ── G.5  Política Crowbar ────────────────────────────────────────────
    def _decide_crowbar_action(
        self,
        final_verdict: BanachHeytingVerdict,
    ) -> CrowbarAction:
        if final_verdict == BanachHeytingVerdict.VETOED:
            return CrowbarAction.GPIO_INTERRUPT_CROWBAR
        if final_verdict == BanachHeytingVerdict.DEGRADED:
            return CrowbarAction.LOG_WARNING
        return CrowbarAction.NONE

    def _resolve_crowbar_port(
        self,
        crowbar_port: Optional[CrowbarPort],
    ) -> CrowbarPort:
        if crowbar_port is None:
            return NullCrowbarPort()
        if not isinstance(crowbar_port, CrowbarPort):
            raise TypeError(
                "crowbar_port debe satisfacer CrowbarPort "
                f"(método trigger_physical_disconnection); recibido {type(crowbar_port)!r}."
            )
        return crowbar_port

    def _actuate_crowbar(
        self,
        action: CrowbarAction,
        crowbar_port: CrowbarPort,
        motor_state: BanachGovernanceState,
        phase2_agent_orientation: Phase2AgentOrientation,
    ) -> AgentCrowbarReport:
        """Ejecuta la política. Nunca toca hardware directo: sólo el puerto."""
        triggered = False
        diagnostic = action.name
        if action == CrowbarAction.GPIO_INTERRUPT_CROWBAR:
            try:
                triggered = bool(crowbar_port.trigger_physical_disconnection())
            except Exception as exc:
                logger.exception("[BANACH-AGENT:Ψ₃] Fallo del CrowbarPort.")
                diagnostic = f"CROWBAR_PORT_FAILURE:{type(exc).__name__}"
                triggered = False
            phase1 = motor_state.phase2_orientation.phase1_observation
            logger.critical(
                "[BANACH-AGENT:Ψ₃] VETO TERMINAL en B(H_n). "
                "Residuo submultiplicativo=%.4e, drift Gelfand=%.4e, "
                "ρ(W)=%.4e, Neumann=%s, crowbar(%s)=%s, canal lógico=%d.",
                float(phase1.submultiplicativity_residual),
                float(motor_state.phase2_orientation.gelfand_residual),
                float(motor_state.neumann_radius),
                motor_state.is_neumann_stable,
                type(crowbar_port).__name__,
                triggered,
                _CROWBAR_CHANNEL,
            )
            _ = phase2_agent_orientation  # enlace causal ya logueado vía residuos
        elif action == CrowbarAction.LOG_WARNING:
            logger.warning(
                "[BANACH-AGENT:Ψ₃] DEGRADED. Neumann=%s ρ(W)=%.4e.",
                motor_state.is_neumann_stable,
                float(motor_state.neumann_radius),
            )
        return AgentCrowbarReport(
            action=action,
            triggered=triggered,
            channel=_CROWBAR_CHANNEL,
            port_typename=type(crowbar_port).__name__,
            diagnostic=diagnostic,
        )

    # ── G.6  Despacho forense de la causa raíz ───────────────────────────
    def _root_cause_exception(
        self,
        final_verdict: BanachHeytingVerdict,
        phase2_agent_orientation: Phase2AgentOrientation,
        motor_state: BanachGovernanceState,
    ) -> Optional[BanachAgentError]:
        """Clasifica la causa raíz; `None` si no hay colapso."""
        if final_verdict != BanachHeytingVerdict.VETOED:
            return None

        phase1 = phase2_agent_orientation.phase1_agent_observation
        if bool(motor_state.singular_base) or not bool(motor_state.is_neumann_stable):
            return NeumannSeriesDivergenceAgent(
                f"Heyting Veto: serie de Neumann no certificable "
                f"(singular={motor_state.singular_base}, "
                f"ρ(W)={float(motor_state.neumann_radius):.6e}). "
                f"Veredicto supremo={final_verdict.name}."
            )
        if not phase1.is_identity_coherent:
            return IdentityUnivalenceCollapse(
                f"Heyting Veto: univalencia de I_n colapsada "
                f"(defect={phase1.identity_relative_defect:.6e})."
            )
        if not phase1.is_cstar_coherent:
            return CStarIdentityCollapse(
                f"Heyting Veto: identidad C* colapsada "
                f"(defect={phase1.cstar_relative_defect:.6e})."
            )
        if not phase1.is_submultiplicative or phase1.verdict == BanachHeytingVerdict.VETOED:
            return SubmultiplicativityCollapse(
                "Heyting Veto: ruptura de submultiplicatividad espectral "
                f"rel={phase1.relative_violation:.6e}, "
                f"residual={float(phase1.observation.submultiplicativity_residual):.6e}."
            )
        if not phase2_agent_orientation.is_gelfand_isometric:
            return GelfandIsometryDrift(
                f"Heyting Veto: drift de Gelfand inaceptable "
                f"(rel={phase2_agent_orientation.relative_drift:.6e}, "
                f"residual={float(phase2_agent_orientation.orientation.gelfand_residual):.6e})."
            )
        return BanachAgentError(
            f"Heyting Veto: anomalía en invariantes del espacio de Banach. "
            f"Veredicto supremo={final_verdict.name}."
        )

    def _dispatch_terminal_exception(
        self,
        cause: BanachAgentError,
        final_verdict: BanachHeytingVerdict,
    ) -> None:
        """Eleva la causa raíz y, si el contrato lo pide, un HeytingLatticeVeto."""
        # Preferimos la excepción específica del agente (más informativa).
        # HeytingLatticeVeto se encadena como contexto si el motor lo define
        # con la firma v3 (verdict, cause).
        try:
            wrapper = HeytingLatticeVeto(
                str(cause),
                verdict=final_verdict,
                cause=cause,
            )
        except TypeError:
            wrapper = None  # motor legado: firma de un solo argumento
        if wrapper is not None:
            raise cause from wrapper
        raise cause

    # ── G.7  Huella forense del soberano ─────────────────────────────────
    def _forensic_provenance(
        self,
        phase2_agent_orientation: Phase2AgentOrientation,
        motor_state: BanachGovernanceState,
        final_verdict: BanachHeytingVerdict,
        crowbar: AgentCrowbarReport,
    ) -> str:
        """SHA-256 de residuos de las tres fases + actuación Crowbar."""
        phase1 = phase2_agent_orientation.phase1_agent_observation
        motor_hash = str(_dto_field(motor_state, "provenance_hash", ""))
        payload = "|".join(
            (
                _round_for_hash(phase1.relative_violation),
                _round_for_hash(phase1.identity_relative_defect),
                _round_for_hash(phase1.cstar_relative_defect),
                _round_for_hash(phase2_agent_orientation.relative_drift),
                _round_for_hash(phase2_agent_orientation.spectral_gap_defect),
                _round_for_hash(float(motor_state.neumann_radius)),
                _round_for_hash(_safe_float(_dto_field(motor_state, "inversion_residual", 0.0))),
                f"N={int(bool(motor_state.is_neumann_stable))}",
                f"S={int(bool(motor_state.singular_base))}",
                f"C={crowbar.action.name}:{int(crowbar.triggered)}",
                f"V={int(final_verdict.value)}",
                f"M={motor_hash}",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── G.8  Reconstrucción compatible del DTO del motor ─────────────────
    def _rebuild_governance_state(
        self,
        motor_state: BanachGovernanceState,
        final_verdict: BanachHeytingVerdict,
        provenance_hash: str,
        diagnostic_note: str,
        diagnostic_atoms: Tuple[str, ...],
        timestamp_utc: float,
    ) -> BanachGovernanceState:
        """
        `dataclasses.replace` sobre el estado del motor, de modo que los
        campos v3 (Kato, rcond, diagnostic_atoms, …) se preserven y los
        campos ausentes en un motor legado no se exijan.
        """
        available = {f.name for f in fields(type(motor_state))}
        overrides: dict[str, Any] = {
            "final_verdict": final_verdict,
            "timestamp_utc": timestamp_utc,
            "provenance_hash": provenance_hash,
            "diagnostic_note": diagnostic_note,
        }
        if "diagnostic_atoms" in available:
            motor_atoms = tuple(_dto_field(motor_state, "diagnostic_atoms", ()) or ())
            overrides["diagnostic_atoms"] = motor_atoms + diagnostic_atoms
        return replace(motor_state, **overrides)

    # ── G.9  SELLO TERMINAL DE LA FASE 3 ─────────────────────────────────
    def _evaluate_banach_governance(
        self,
        phase2_agent_orientation: Phase2AgentOrientation,
        motor_state: BanachGovernanceState,
        crowbar_port: Optional[CrowbarPort] = None,
        raise_on_veto: bool = True,
    ) -> BanachAgentGovernanceState:
        """
        Ψ₃ — morfismo terminal de la Fase 3 y del funtor Z_BanachAgent.

        Join en Ω:

            v_Ψ  =  v_{Ψ₁} ∨ v_{Ψ₂} ∨ v_Neumann^Ψ ∨ v_motor.

        Si v_Ψ = VETOED se actúa el CrowbarPort (canal lógico
        `_CROWBAR_CHANNEL`) y, si `raise_on_veto`, se eleva la excepción
        de causa raíz (Neumann / [I2] / [I3] / [I4] / Gelfand).
        """
        obs2 = self._ingest_phase2_agent_precondition(phase2_agent_orientation)
        gov_motor = self._ingest_motor_governance(motor_state, obs2)

        neumann_verdict = self._classify_neumann_agent(gov_motor)
        final_verdict = self._compose_terminal_verdict(obs2, gov_motor, neumann_verdict)

        port = self._resolve_crowbar_port(crowbar_port)
        action = self._decide_crowbar_action(final_verdict)
        crowbar = self._actuate_crowbar(action, port, gov_motor, obs2)

        cause = self._root_cause_exception(final_verdict, obs2, gov_motor)
        if cause is not None and raise_on_veto:
            self._dispatch_terminal_exception(cause, final_verdict)

        stamp = time.time()
        provenance = self._forensic_provenance(obs2, gov_motor, final_verdict, crowbar)

        phase1 = obs2.phase1_agent_observation
        atoms: Tuple[str, ...] = (
            f"v_Ψ={final_verdict.name}",
            f"v_motor={gov_motor.final_verdict.name}",
            f"v_P12={obs2.verdict.name}",
            f"v_N={neumann_verdict.name}",
            f"ρ(W)={float(gov_motor.neumann_radius):.6e}",
            f"crowbar={crowbar.action.name}:{int(crowbar.triggered)}",
            f"hash={provenance[:16]}",
        )
        diagnostic_note = (
            f"Veredicto Soberano: {final_verdict.name}. "
            f"Submultiplicatividad: {phase1.is_submultiplicative}. "
            f"Identidad coherente: {phase1.is_identity_coherent}. "
            f"C* coherente: {phase1.is_cstar_coherent}. "
            f"Isometría Gelfand: {obs2.is_gelfand_isometric}. "
            f"Gap coherente: {obs2.is_gap_coherent}. "
            f"Neumann estable: {gov_motor.is_neumann_stable} "
            f"(ρ={float(gov_motor.neumann_radius):.4e}). "
            f"Crowbar: {crowbar.action.name}/{crowbar.triggered} "
            f"vía {crowbar.port_typename}. "
            f"Huella: {provenance[:16]}…"
        )

        rebuilt = self._rebuild_governance_state(
            motor_state=gov_motor,
            final_verdict=final_verdict,
            provenance_hash=provenance,
            diagnostic_note=diagnostic_note,
            diagnostic_atoms=atoms,
            timestamp_utc=stamp,
        )

        return BanachAgentGovernanceState(
            phase2_agent_orientation=obs2,
            motor_state=gov_motor,
            governance_state=rebuilt,
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
class BanachAlgebraAuditorAgent(Morphism, Phase3_HeytingBanachDecider):
    r"""
    Soberano y guardián del espacio de operadores en el álgebra de Banach.

    Orquesta al motor `BanachAlgebraAuditor` mediante el ciclo OODA y la
    composición funtorial estricta

        Z_BanachAgent  =  Ψ₃ ∘ Ψ₂ ∘ Ψ₁ ∘ Z_motor.

    Cada Ψ_{k+1} *ingiere* el sello terminal de Ψ_k; no hay atajos. El
    motor se invoca siempre con `raise_on_veto=False`: la aduana es el
    único punto de aborto y de actuación Crowbar.

    En circuitos eléctricos, T modela el operador de malla/impedancia y
    dT una perturbación de componentes: el Crowbar es el disyuntor de la
    red si el resolvente perturbado deja de ser certificado.
    """

    def __init__(
        self,
        auditor: BanachAlgebraAuditor,
        *,
        crowbar_port: Optional[CrowbarPort] = None,
        soft_submultiplicativity_tol: float = _SOFT_SUBMULTIPLICATIVITY_TOL,
        hard_submultiplicativity_tol: float = _HARD_SUBMULTIPLICATIVITY_TOL,
        soft_identity_tol: float = _SOFT_IDENTITY_TOL,
        hard_identity_tol: float = _HARD_IDENTITY_TOL,
        soft_cstar_tol: float = _SOFT_CSTAR_TOL,
        hard_cstar_tol: float = _HARD_CSTAR_TOL,
        soft_gelfand_tol: float = _SOFT_GELFAND_DRIFT_TOL,
        hard_gelfand_tol: float = _HARD_GELFAND_DRIFT_TOL,
        spectral_margin: float = _AGENT_SPECTRAL_MARGIN,
    ) -> None:
        Phase3_HeytingBanachDecider.__init__(
            self,
            soft_submultiplicativity_tol=soft_submultiplicativity_tol,
            hard_submultiplicativity_tol=hard_submultiplicativity_tol,
            soft_identity_tol=soft_identity_tol,
            hard_identity_tol=hard_identity_tol,
            soft_cstar_tol=soft_cstar_tol,
            hard_cstar_tol=hard_cstar_tol,
            soft_gelfand_tol=soft_gelfand_tol,
            hard_gelfand_tol=hard_gelfand_tol,
            spectral_margin=spectral_margin,
        )
        if not isinstance(auditor, BanachAlgebraAuditor):
            raise TypeError(
                "auditor debe ser una instancia de BanachAlgebraAuditor; "
                f"recibido {type(auditor)!r}."
            )
        self._auditor = auditor
        self._default_crowbar: CrowbarPort = (
            crowbar_port if crowbar_port is not None else NullCrowbarPort()
        )

    def _invoke_motor(
        self,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
    ) -> BanachGovernanceState:
        """Z_motor con veto silenciado; traduce fallos algebraicos al agente."""
        try:
            return self._auditor.execute_banach_audit(T, dT, raise_on_veto=False)
        except HeytingLatticeVeto:
            # El motor no debería elevar con raise_on_veto=False; si lo hace,
            # es una violación de contrato y se propaga intacta.
            raise
        except BanachAlgebraError as exc:
            raise BanachAgentError(
                f"El motor BanachAlgebraAuditor abortó fuera de contrato: {exc!s}.",
                cause=exc,
            ) from exc

    def execute_certified_governance(
        self,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
        crowbar_port: Optional[CrowbarPort] = None,
        raise_on_veto: bool = True,
    ) -> BanachAgentGovernanceState:
        """
        Ciclo OODA completo con certificado soberano (envoltorio rico).

        Composición:
            sanitize → Z_motor → Ψ₁ → Ψ₂ → Ψ₃.
        """
        t_op, dt_op = self._sanitize_input(T, dT)
        port = crowbar_port if crowbar_port is not None else self._default_crowbar

        # OODA 1–2 : Observe & Orient — invocación al motor (sin aborto).
        motor_state = self._invoke_motor(t_op, dt_op)

        # OODA 3   : Decide — aduana de normas e isometría.
        phase1_agent_obs = self._observe_norm_coherence(
            motor_state.phase2_orientation.phase1_observation
        )
        phase2_agent_orientation = self._orient_gelfand_isometry(
            phase1_agent_obs,
            motor_state.phase2_orientation,
        )

        # OODA 4   : Act — supremo, Crowbar y sello forense.
        certified = self._evaluate_banach_governance(
            phase2_agent_orientation=phase2_agent_orientation,
            motor_state=motor_state,
            crowbar_port=port,
            raise_on_veto=raise_on_veto,
        )
        logger.info(
            "[BANACH-AGENT:Z] Gobernanza cerrada. verdict=%s hash=%s crowbar=%s",
            certified.final_verdict.name,
            certified.provenance_hash[:16],
            certified.crowbar.action.name,
        )
        return certified

    def execute_banach_governance(
        self,
        T: NDArray[np.float64],
        dT: NDArray[np.float64],
        crowbar_port: Optional[CrowbarPort] = None,
        raise_on_veto: bool = True,
    ) -> BanachGovernanceState:
        """
        Ciclo categórico completo (contrato legado).

        Devuelve el `BanachGovernanceState` reconstruido con el veredicto
        soberano. El certificado rico vive en `execute_certified_governance`.
        """
        certified = self.execute_certified_governance(
            T=T,
            dT=dT,
            crowbar_port=crowbar_port,
            raise_on_veto=raise_on_veto,
        )
        return certified.governance_state

    def govern_batch(
        self,
        pairs: Iterable[Tuple[NDArray[np.float64], NDArray[np.float64]]],
        crowbar_port: Optional[CrowbarPort] = None,
        raise_on_veto: bool = False,
    ) -> Tuple[BanachAgentGovernanceState, ...]:
        """
        Gobernanza de una familia finita de pares (T, dT).

        Por defecto no eleva veto (`raise_on_veto=False`) para no abortar
        el lote: cada certificado porta su propio veredicto y Crowbar.
        """
        return tuple(
            self.execute_certified_governance(
                T=t_op,
                dT=dt_op,
                crowbar_port=crowbar_port,
                raise_on_veto=raise_on_veto,
            )
            for t_op, dt_op in pairs
        )


# Aliases explícitos de la aduana (evitan colisión semántica con el motor).
Phase1_AgentNormObserver = Phase1_BanachNormObserver
Phase2_AgentGelfandOrienter = Phase2_GelfandSpectralOrienter
Phase3_AgentHeytingDecider = Phase3_HeytingBanachDecider


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "__version__",
    "BanachHeytingVerdict",
    "CrowbarAction",
    "CrowbarPort",
    "NullCrowbarPort",
    "LoggingCrowbarPort",
    "CompositeCrowbarPort",
    "BanachAgentError",
    "SubmultiplicativityCollapse",
    "IdentityUnivalenceCollapse",
    "CStarIdentityCollapse",
    "GelfandIsometryDrift",
    "NeumannSeriesDivergenceAgent",
    "DualSourceIncoherence",
    "HeytingLatticeVeto",
    "Phase1NormObservation",
    "Phase2GelfandOrientation",
    "BanachGovernanceState",
    "Phase1AgentObservation",
    "Phase2AgentOrientation",
    "AgentCrowbarReport",
    "BanachAgentGovernanceState",
    "Phase1_BanachNormObserver",
    "Phase2_GelfandSpectralOrienter",
    "Phase3_HeytingBanachDecider",
    "Phase1_AgentNormObserver",
    "Phase2_AgentGelfandOrienter",
    "Phase3_AgentHeytingDecider",
    "BanachAlgebraAuditorAgent",
]