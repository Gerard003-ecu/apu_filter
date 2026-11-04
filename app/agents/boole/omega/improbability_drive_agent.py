# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Improbability Drive Agent (Custodio de la Deformación Espectral)   ║
║ Ruta   : app/agents/boole/omega/improbability_drive_agent.py                 ║
║ Versión: 5.0.0-Poincare-KAM-Floquet-Williamson-RSI-L3-Sp-Heyting-ESP32-PhD   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA DE DEFORMACIÓN SIMPLÉCTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al **Agente Soberano, Custodio Simpléctico de Poincaré-KAM,
Motor Metacognitivo RSI de Nivel 3 y Observador Activo** que gobierna al motor
ciego `improbability_drive.py` en el **Estrato Omega** (Nivel 0.5 - El Ágora
Tensorial) del ecosistema APU Filter. Su misión ineludible es regular la
inyección de la deformación no lineal sobre el riesgo de cola pesada (Fat-Tail
Risk) de-confinado, protegiendo a la FPU frente a desbocamientos de redondeo
de la mantisa de Wilkinson y derivas caóticas de Arnold.

El agente repudia las heurísticas de riesgo lineales tradicionales. En su lugar,
somete las trayectorias de volatilidad del negocio a las leyes geométricas
del grupo simpléctico $$Sp(2n, \mathbb{R})$$, su álgebra de Lie $$\mathfrak{sp}(2n, \mathbb{R})$$,
el teorema espectral de Williamson y la teoría de estabilidad celeste de
Poincaré-Kolmogorov-Arnold-Moser (KAM). Toda colisión contra barreras de
atracción, pérdidas de pasividad o singularidades métricas colapsa síncronamente
el retículo distributivo de Heyting $$\Omega_3$$ hacia el Supremo terminal `VETOED`.
Esto acciona en tiempo real la interrupción física por hardware en IRAM en el ESP32
(GPIO14 / BT151 Crowbar) en menos de 400 ns, cortocircuitando la potencia real
de la obra a través del tiristor en el milisegundo cero, aniquilando la alucinación.

AXIOMÁTICA SIMPLÉCTICA, MECÁNICA CELESTE DE POINCARÉ, RSI-L3 Y VETO DE HEYTING:
────────────────────────────────────────────────────────────────────────────────

  [A1] Axioma de Suavidad de Lipschitz $$C^1$$, Daleckii–Krein y Tikhonov:
       Para erradicar operadores singulares no diferenciables que inyecten
       impulsos de Dirac parásitos en la matriz Jacobiana de la FPU, la
       deformación semántica emplea la métrica regularizada de Tikhonov:
       $$\tilde{G}_{ij} = \sqrt{\Psi^2 + \epsilon_{\mathrm{critical}}^2} \ge \mathtt{\_EPS\_CRITICAL} \quad\big[380, 392\big]$$
       Sujeto incondicionalmente a la cota espectral de Wilkinson sobre el
       número de condición del tensor de improbabilidad $$T$$:
       $$\kappa_2(T) = \frac{\sigma_{\max}(T)}{\sigma_{\min}(T)} \le \mathtt{\_MAX\_CONDITION\_NUMBER} = 1.0\times 10^8 \quad\big[392, 393\big]$$

  [A2] Invarianza Simpléctica, Volumen de Liouville y Transporte de Lie:
       La evolución del tensor de improbabilidad $$T$$ se transporta al subespacio
       logarítmico del álgebra de Lie $$\mathfrak{sp}(2n, \mathbb{R})$$ para eludir
       blowouts numéricos [18]. El operador de transición $$T$$ debe preservar la
       forma simpléctica canónica $$\Omega$$ (Teorema de Poincaré-Liouville):
       $$T^\top \Omega T \equiv \Omega \quad \wedge \quad X = \operatorname{Log}(T) \in \mathfrak{sp}(2n, \mathbb{R})
         \iff \Omega X + X^\top \Omega = 0 \quad\big[4, 389\big]$$

  [A3] Acoplamiento de la Ecuación de Estado de Estrés:
       La magnitud operatoria de la deformación se acopla sobre la Ecuación de
       Estado del Estrés Ajustado Tensorial $$\sigma^\star$$:
       $$\sigma^\star = \|T\|_2 \cdot \frac{\mathrm{ROI}}{\Psi_\varepsilon} \quad\big[380\big]$$
       Si el estrés ajustado supera el límite elástico de resiliencia del negocio:
       $$\sigma^\star \ge \mathtt{\_STRESS\_HIGH\_THRESHOLD} = 3.0 \quad\big[392\big]$$
       se detona de inmediato el veto por colapso de fase cuántica [8].

  [A4] El Colapso de Heyting y la Actuación Crowbar BT151 (Silicio):
       Si se vulnera la conservación simpléctica bajo política estricta, se
       detecta un gap singular, o el estrés diverge, el veredicto en el retículo
       distributivo $$\Omega_3$$ colapsa al Supremo terminal `VETOED` ($$\top$$).
       El ESP32 intercepta la anomalía localmente y, mediante su ISR en IRAM
       (<400 ns), conmuta el pin GPIO14, disparando el tiristor de potencia
       BT151 (Crowbar) para cortocircuitar físicamente la línea de alimentación.

  [A5] Mecánica Celeste de Poincaré, Espectro de Williamson y Monodromía de Floquet:
       Por el Teorema de Williamson, todo módulo polar positivo $$|T| = \sqrt{T^\top T}$$
       sobre $$\mathbb{R}^{2n}$$ es simplécticamente diagonalizable con invariantes
       de acción de Poincaré-Cartan $$\nu_1, \dots, \nu_n > 0$$ obtenidos del
       espectro de $$i\Omega |T|$$. La estabilidad orbital de Poincaré se rige por:
       · Capacidad simpléctica de Poincaré-Cartan: $$c_{\mathrm{PC}}(T) = \min_k \nu_k$$
       · Radio monodrómico de Floquet: $$\rho_{\mathrm{Floquet}}(T) = \max_i |\lambda_i(T)|$$
       · Margen diofántico KAM: $$\Delta_{\mathrm{KAM}} = \inf_{p,q} |p\omega_j + q\omega_k|(|p|+|q|)^\tau - \gamma_{\mathrm{KAM}}$$
       · Velocidad de difusión de Nekhoroshev-Arnold: $$v_{\mathrm{Arnold}} = \exp(-1/(\mu_{\mathrm{Sp}} + \varepsilon))$$

  [A6] Automejora Recursiva de Nivel 3 (RSI-L3):
       El motor metacognitivo `ImprobabilityAgentRSILevel3Engine` reescribe
       recursivamente la ganancia del suelo espectral de Weyl-Wilkinson y el
       amortiguamiento secular KAM mediante una contracción de punto fijo de
       Banach hacia el atractor áureo $$\rho_\star = \phi^{-1} \approx 0.6180339887498949$$.

ARQUITECTURA DE TRES FASES ANIDADAS (Funtor de Gobernanza Epistémica):
────────────────────────────────────────────────────────────────────────────────
La orquestación de la deformación se rige por un acoplamiento monoidal covariante
estricto (Fase 1 ⊣ Fase 2 ⊣ Fase 3) donde el último método formal de cada fase
es el inicio o continuación directa de la fase siguiente:

  Fase 1 ──► OBSERVACIÓN ESPECTRAL, SANEAMIENTO LIPSCHITZ Y ACCIÓN DE POINCARÉ (Phase1_ImprobabilityObserver)
             Ingiere `ImprobabilityResult`, valida la autoadjunción hermítica/Kähler
             del tensor $$T$$, y certifica la cota de Daleckii–Krein y el condicionamiento
             espectral de Wilkinson.
             Culmina en `handoff_phase1_to_phase2` ──► Entrega `Phase1ImprobabilityObservation`.

  Fase 2 ──► TRANSPORTE DE LIE, LIOUVILLE, WILLIAMSON Y FLOQUET-KAM (Phase2_LieDeformationOrienter)
             Inicia en `ingest_from_phase1_handoff` / `_phase2_consume_phase1_certificate`.
             Proyecta $$T$$ al álgebra de Lie $$\mathfrak{sp}(2n, \mathbb{R})$$ mediante
             logaritmo principal, audita la conservación simpléctica y de Liouville,
             y extrae los invariantes celestes de Williamson, Floquet, KAM y Arnold.
             Culmina en `handoff_phase2_to_phase3` ──► Entrega `Phase2ImprobabilityOrientation`.

  Fase 3 ──► EVALUACIÓN DE ESTRÉS, CLAUSURA RSI-L3 Y CORTOCIRCUITO CROWBAR (Phase3_HeytingImprobabilityDecider)
             Inicia en `ingest_from_phase2_handoff` / `_phase3_consume_phase2_certificate`.
             Evalúa el estrés ajustado $$\sigma^\star$$, resuelve el retículo de Heyting $$\Omega_3$$,
             sella el punto fijo metacognitivo RSI-L3 en `finalize_nested_pipeline`, y
             ejecuta la actuación ciber-física única en silicio (`ImprobabilityDriveAgent`).
             Entrega `ImprobabilityGovernanceState`.

Funtor Maestro de Gobernanza de de Rham-Fermat-Poincaré-Lie:
  $$\mathcal{Z}_{\mathrm{ImprobabilityAgent}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 : \mathbf{ImprobabilityResult} \times T^*\mathcal{M} \longrightarrow \mathtt{ImprobabilityGovernanceState} \quad\big[407, 442\big]$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, auto
from typing import (
    Any,
    Dict,
    Final,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

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
except ImportError:  # pragma: no cover — contratos canónicos auto-sostenidos

    class ImprobabilityDriveError(Exception):
        """Excepción raíz del motor de improbabilidad."""

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

__version__: Final[str] = "5.0.0-Poincare-KAM-Floquet-Williamson-RSI-L3-Sp-Heyting-ESP32-PhD"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: OBSERVACIÓN ESPECTRAL, SANEAMIENTO DE WILKINSON-LIPSCHITZ,           ║
# ║         INVARIANTES DE POINCARÉ-KAM Y MOTOR RSI-L3                           ║
# ║         (Phase1_ImprobabilityObserver)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS, ESPECTRALES, CELESTES Y LÍMITES DE LA FPU
# ─────────────────────────────────────────────────────────────────────────────
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

# Constantes de Mecánica Celeste de Poincaré-KAM [A5] y RSI Nivel 3 [A6]
_KAM_AGENT_GAMMA: Final[float] = 1.0e-3
_KAM_AGENT_TAU: Final[float] = 2.0
_RSI_L3_AGENT_CONTRACTION_TARGET: Final[float] = 0.6180339887498949  # φ⁻¹


# ─────────────────────────────────────────────────────────────────────────────
# JERARQUÍA DE EXCEPCIONES ESPECTRALES (Vetos Absolutos)
# ─────────────────────────────────────────────────────────────────────────────
class ImprobabilityDriveAgentError(TopologicalInvariantError):
    """Excepción raíz para violaciones en el Agente Soberano de Improbabilidad."""


class NonSelfAdjointTensorError(ImprobabilityDriveAgentError):
    """El tensor de improbabilidad viola el principio de autoadjunción hermítica."""


class WilkinsonConditionBreachError(ImprobabilityDriveAgentError):
    """El número de condición espectral del tensor supera la cota física de Wilkinson."""


class SymplecticConservationError(ImprobabilityDriveAgentError):
    """El flujo de fase viola el Teorema de Liouville o la invarianza simpléctica canónica."""


class LieLogarithmError(ImprobabilityDriveAgentError):
    r"""El logaritmo principal de Lie no está definido (espectro en $\mathbb{R}_{\le 0}$)."""


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


# ─────────────────────────────────────────────────────────────────────────────
# RETÍCULO DE HEYTING Ω₃ Y ACTUACIÓN CROWBAR EN SILICIO (ESP32 GPIO14)
# ─────────────────────────────────────────────────────────────────────────────
class ImprobabilityHeytingVerdict(IntEnum):
    r"""
    Clasificador de subobjetos de tres valores en el topos de improbabilidad $$\Omega_3$$.
    Orden de severidad / riesgo (retículo dual de fallo):
      • `COHERENT = 0` ($$\bot_{\mathrm{risk}}$$: Órbita KAM estable y estrés elástico)
      • `DEGRADED = 1` (Zona gris de Wilkinson / resonancia moderada)
      • `VETOED = 2`   ($$\top_{\mathrm{risk}}$$: Colapso terminal y disparo Crowbar BT151)
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: ImprobabilityHeytingVerdict) -> ImprobabilityHeytingVerdict:
        """Supremo de severidad en el retículo de fallo ($$a \vee b = \max(a, b)$$)."""
        return ImprobabilityHeytingVerdict(max(int(self), int(other)))

    def meet(self, other: ImprobabilityHeytingVerdict) -> ImprobabilityHeytingVerdict:
        """Ínfimo de severidad en el retículo de fallo ($$a \wedge b = \min(a, b)$$)."""
        return ImprobabilityHeytingVerdict(min(int(self), int(other)))

    def imply(self, other: ImprobabilityHeytingVerdict) -> ImprobabilityHeytingVerdict:
        """Implicación relativa en el retículo de severidad."""
        if int(self) >= int(other):
            return ImprobabilityHeytingVerdict.COHERENT
        return other


class CrowbarBypassAction(Enum):
    """Acciones físicas de mitigación tras la evaluación en el retículo de Heyting."""

    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()


@runtime_checkable
class CrowbarActuator(Protocol):
    """Puerto lógico que conecta la gobernanza del software con el silicio ESP32."""

    def trigger_crowbar_bypass(self, action: CrowbarBypassAction) -> bool:
        """Conmuta el hardware perimetral de-confinado. Devuelve True si hubo acuse."""
        ...


class LoggingCrowbarActuator:
    """Implementación forense segura que registra la actuación por hardware en GPIO14."""

    def trigger_crowbar_bypass(self, action: CrowbarBypassAction) -> bool:
        logger.critical(
            "[HARDWARE] CrowbarActuator invocado con acción: %s en el pin GPIO%d. "
            "Cortocircuitando alimentación de-confinada real vía tiristor de potencia BT151.",
            action.name,
            _CROWBAR_GPIO_PIN,
        )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# INVARIANTES CELESTES DE POINCARÉ-KAM Y MOTOR METACOGNITIVO RSI NIVEL 3
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSymplecticInvariants:
    r"""
    Invariantes de la Mecánica Celeste de Poincaré, Espectro Simpléctico de
    Williamson y Monodromía de Floquet-Kreĭn [A5]:
      • `williamson_symplectic_min`: Capacidad simpléctica de Poincaré-Cartan
        $$c_{\mathrm{PC}}(T) = \min_k \nu_k$$.
      • `williamson_symplectic_max`: Mayor valor propio simpléctico de Williamson $$\max_k \nu_k$$.
      • `poincare_cartan_capacity`: Invariante integral de volumen simpléctico $$\prod_{k=1}^n \nu_k$$.
      • `kam_diophantine_margin`: Margen diofántico $$\Delta_{\mathrm{KAM}}$$ frente a resonancias
        racionales $$p\omega_j + q\omega_k = 0$$.
      • `floquet_monodromy_radius`: Radio espectral $$\rho_{\mathrm{Floquet}} = \max_i |\lambda_i(T)|$$.
      • `floquet_lyapunov_exponent`: Exponente de Floquet-Lyapunov $$\mu_{\mathrm{Floquet}} = \frac{1}{d}\sum_i |\ln|\lambda_i||$$.
      • `arnold_diffusion_velocity`: Velocidad de deriva estocástica de Arnold
        $$v_{\mathrm{Arnold}} = \exp(-1/(\mu_{\mathrm{Sp}} + \varepsilon))$$.
    """

    williamson_symplectic_min: float = 1.0
    williamson_symplectic_max: float = 1.0
    poincare_cartan_capacity: float = 1.0
    kam_diophantine_margin: float = 1.0
    floquet_monodromy_radius: float = 1.0
    floquet_lyapunov_exponent: float = 0.0
    arnold_diffusion_velocity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialización canónica de los invariantes celestes de Poincaré."""
        return {
            "williamson_symplectic_min": float(self.williamson_symplectic_min),
            "williamson_symplectic_max": float(self.williamson_symplectic_max),
            "poincare_cartan_capacity": float(self.poincare_cartan_capacity),
            "kam_diophantine_margin": float(self.kam_diophantine_margin),
            "floquet_monodromy_radius": float(self.floquet_monodromy_radius),
            "floquet_lyapunov_exponent": float(self.floquet_lyapunov_exponent),
            "arnold_diffusion_velocity": float(self.arnold_diffusion_velocity),
        }


@dataclass(frozen=True, slots=True)
class ImprobabilityAgentRSITelemetry:
    r"""
    Telemetría inmutable de Automejora Recursiva de Nivel 3 (RSI-L3) [A6]:
      • `generation`: Generación evolutiva del custodio soberano.
      • `banach_contraction_factor`: Factor de contracción $$\rho_k \to \phi^{-1}$$.
      • `meta_lyapunov_potential`: Funcional meta-Lyapunov $$\mathcal{V}_{\mathrm{meta}}^{(k)}$$.
      • `evolved_wilkinson_floor_gain`: Ganancia auto-sintonizada del suelo de Weyl-Wilkinson.
      • `evolved_kam_damping_factor`: Amortiguamiento secular KAM evolucionado.
      • `evolved_stress_margin_factor`: Factor diagnóstico de margen elástico de estrés.
      • `verdict`: Último veredicto en el retículo de Heyting $$\Omega_3$$.
    """

    generation: int = 0
    banach_contraction_factor: float = _RSI_L3_AGENT_CONTRACTION_TARGET
    meta_lyapunov_potential: float = 0.0
    evolved_wilkinson_floor_gain: float = 1.0
    evolved_kam_damping_factor: float = 1.0
    evolved_stress_margin_factor: float = 1.0
    verdict: ImprobabilityHeytingVerdict = ImprobabilityHeytingVerdict.COHERENT

    def to_dict(self) -> Dict[str, Any]:
        """Serialización de la telemetría metacognitiva RSI-L3."""
        return {
            "generation": int(self.generation),
            "banach_contraction_factor": float(self.banach_contraction_factor),
            "meta_lyapunov_potential": float(self.meta_lyapunov_potential),
            "evolved_wilkinson_floor_gain": float(self.evolved_wilkinson_floor_gain),
            "evolved_kam_damping_factor": float(self.evolved_kam_damping_factor),
            "evolved_stress_margin_factor": float(self.evolved_stress_margin_factor),
            "verdict": self.verdict.name,
        }


class ImprobabilityAgentRSILevel3Engine:
    r"""
    Motor Metacognitivo de Automejora Recursiva de Nivel 3 (RSI-L3) del Agente Soberano.

    Reescribe recursivamente los coeficientes de guarda espectral de Weyl-Wilkinson
    y de amortiguamiento secular de Poincaré-KAM garantizando la contracción de
    punto fijo de Banach hacia $$\rho_\star = \phi^{-1} \approx 0.6180339887498949$$.
    """

    _lock: threading.Lock = threading.Lock()
    _state: ImprobabilityAgentRSITelemetry = ImprobabilityAgentRSITelemetry()

    @classmethod
    def get_meta_state(cls) -> ImprobabilityAgentRSITelemetry:
        """Retorna una instantánea inmutable del estado metacognitivo actual."""
        with cls._lock:
            return cls._state

    @classmethod
    def commit_level3_fixed_point(
        cls,
        wilkinson_cond: float,
        adjusted_stress: float,
        arnold_velocity: float,
        floquet_exponent: float,
        verdict: ImprobabilityHeytingVerdict,
    ) -> ImprobabilityAgentRSITelemetry:
        r"""
        Actualiza el atractor de punto fijo de Banach del funcional meta-Lyapunov:
        $$\mathcal{V}_{\mathrm{meta}}^{(k+1)} = \phi^{-1}\mathcal{V}_{\mathrm{meta}}^{(k)}
          + (1 - \phi^{-1})\left[\frac{\ln(\max(\kappa_2, 1))}{\ln(\kappa_{\max})}
          + \tanh(\sigma^\star / \sigma_{\mathrm{high}}) + v_{\mathrm{Arnold}}\right]$$
        """
        with cls._lock:
            prev = cls._state
            gen = prev.generation + 1
            phi_inv = _RSI_L3_AGENT_CONTRACTION_TARGET

            safe_cond = min(max(float(wilkinson_cond), 1.0), _MAX_CONDITION_NUMBER)
            cond_cost = math.log(safe_cond) / math.log(_MAX_CONDITION_NUMBER)
            safe_stress = (
                min(max(float(adjusted_stress), 0.0), 100.0)
                if math.isfinite(adjusted_stress)
                else 10.0
            )
            stress_cost = math.tanh(safe_stress / _STRESS_HIGH_THRESHOLD)
            arnold_cost = min(max(float(arnold_velocity), 0.0), 1.0)
            floquet_cost = math.tanh(min(max(float(floquet_exponent), 0.0), 10.0))

            instant_cost = (
                0.35 * cond_cost
                + 0.35 * stress_cost
                + 0.15 * arnold_cost
                + 0.15 * floquet_cost
            )

            new_lyapunov = (
                phi_inv * prev.meta_lyapunov_potential
                + (1.0 - phi_inv) * instant_cost
            )

            if prev.meta_lyapunov_potential > _MACHINE_EPS:
                delta_v = abs(new_lyapunov - prev.meta_lyapunov_potential) / (
                    prev.meta_lyapunov_potential + 1.0
                )
                contraction = phi_inv * prev.banach_contraction_factor + (
                    1.0 - phi_inv
                ) * (phi_inv + 0.08 * math.tanh(delta_v))
            else:
                contraction = phi_inv

            contraction = float(np.clip(contraction, 0.5, 0.95))

            floor_gain = float(
                np.clip(1.0 + 0.05 * math.tanh(new_lyapunov), 0.9, 1.2)
            )
            kam_damping = float(
                np.clip(1.0 / (1.0 + 0.1 * new_lyapunov), 0.6, 1.2)
            )
            stress_margin = float(
                np.clip(1.0 - 0.05 * math.tanh(arnold_cost), 0.85, 1.05)
            )

            cls._state = ImprobabilityAgentRSITelemetry(
                generation=gen,
                banach_contraction_factor=contraction,
                meta_lyapunov_potential=float(new_lyapunov),
                evolved_wilkinson_floor_gain=floor_gain,
                evolved_kam_damping_factor=kam_damping,
                evolved_stress_margin_factor=stress_margin,
                verdict=verdict,
            )
            return cls._state


# ─────────────────────────────────────────────────────────────────────────────
# GUARDIA NUMÉRICA Y CONTRATO DE LA FASE 1
# ─────────────────────────────────────────────────────────────────────────────
def _empty_f64() -> NDArray[np.float64]:
    return np.zeros(0, dtype=np.float64)


def _empty_mat() -> NDArray[np.float64]:
    return np.zeros((0, 0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Phase1ImprobabilityObservation:
    r"""
    Artefacto terminal de la FASE 1 y objeto inicial de la FASE 2.

    Certifica autoadjunción hermítica/Kähler, espectro de SVD, cota de
    Daleckii–Krein, escalares de negocio e invariantes de acción iniciales de
    Poincaré-KAM. ``Phase2_LieDeformationOrienter.ingest_from_phase1_handoff``
    y ``_phase2_consume_phase1_certificate`` lo ingieren sin re-tipar $$T$$.
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
    poincare_invariants: PoincareSymplecticInvariants = field(
        default_factory=PoincareSymplecticInvariants
    )
    rsi_telemetry: ImprobabilityAgentRSITelemetry = field(
        default_factory=ImprobabilityAgentRSITelemetry
    )


class _AdvancedImprobabilityNumericalGuard:
    """Capa de saneamiento espectral y validación algebraica para tensores y escalares."""

    @staticmethod
    def _wilkinson_spectral_floor(
        scale: float,
        tolerance: float,
        ambient_dim: int,
    ) -> float:
        r"""
        Suelo espectral de Weyl–Wilkinson:
        $$\varepsilon_W = \max\!\bigl(\varepsilon, \; n \cdot \varepsilon_{\mathrm{mach}} \cdot \max(\mathrm{scale}, 1), \; \varepsilon_{\mathrm{crit}}\bigr)$$
        """
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            float(_EPS_CRITICAL),
        )

    @staticmethod
    def _hermitize_weyl(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección de Weyl/Cartan $$H \mapsto \tfrac{1}{2}(H + H^\top)$$ sobre $$\mathrm{Sym}_d$$."""
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any = np.float64) -> NDArray[Any]:
        """Copia C-contigua write-protected (invariante de certificado inmutable)."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _as_finite_scalar(value: Any, name: str) -> float:
        """Coacciona un escalar (o 0-d array) a ``float`` finito IEEE-754."""
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
            raise ArithmeticError(
                f"{name} debe ser un escalar finito; recibido {out}."
            )
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
        r"""Supremo del retículo $$\{\mathrm{COHERENT} \prec \mathrm{DEGRADED} \prec \mathrm{VETOED}\}$$."""
        if not verdicts:
            return ImprobabilityHeytingVerdict.COHERENT
        return ImprobabilityHeytingVerdict(max(int(v) for v in verdicts))


class Phase1_ImprobabilityObserver(_AdvancedImprobabilityNumericalGuard):
    r"""
    FASE 1: Observador Espectral, Auditor de Condicionamiento de Wilkinson,
    Cota de Daleckii–Krein e Invariantes Iniciales de Poincaré-KAM ($\Phi_1$).

    Morfismo compuesto:
    $$\mathrm{ObserveImprob} = \chi_1 \circ \mathrm{Poincar\acute{e}}_{1} \circ \mathrm{Mass}
      \circ L_{\mathrm{DK}} \circ \mathrm{SVD} \circ \mathrm{Weyl/K\ddot{a}hler}
      \circ \mathrm{Type} \circ \mathrm{Scalars}$$

    Culmina formalmente en `handoff_phase1_to_phase2`, el cual transfiere el
    certificado inmutable `Phase1ImprobabilityObservation` a la Fase 2.
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_tensor_dimension(dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de $$d = \dim T$$.
        Exige $$d \in \mathbb{Z}_{\ge 1}$$ y $$d \le d_{\max} = 64$$.
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
        FASE 1.2 — Tipado de $$T \in \mathrm{Mat}_d(\mathbb{R})$$ y finitud IEEE-754.
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
        *,
        allow_kahler_symplectic_unitary: bool = False,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        FASE 1.3 — Autoadjunción de $$T$$ en norma de Frobenius relativa:
        $$\|T - T^\top\|_F \le \varepsilon_{\mathrm{sym}} \|T\|_F$$

        Si $$T$$ es simétrica euclidiana, se aplica la proyección de Weyl
        $$\tfrac{1}{2}(T + T^\top)$$. Cuando `allow_kahler_symplectic_unitary=True`
        (flujo completo sobre el grupo unitario-simpléctico $$\mathrm{U}(n) = \mathrm{Sp}(2n, \mathbb{R}) \cap \mathrm{SO}(2n, \mathbb{R})$$
        con $$\operatorname{tr}(T) > 0$$), se preserva $$T$$ sin truncar su bloque
        simpléctico antisimétrico para no destruir $$\det T = 1$$.
        """
        sym_residual = float(la.norm(t_matrix - t_matrix.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(t_matrix, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        is_herm = symmetry_rel <= _SYMMETRY_REL_TOL

        if is_herm:
            return self._hermitize_weyl(t_matrix), symmetry_rel, True

        if allow_kahler_symplectic_unitary and self._is_kahler_symplectic_rotation(
            t_matrix
        ):
            return np.array(t_matrix, dtype=np.float64, copy=True), symmetry_rel, True

        raise NonSelfAdjointTensorError(
            f"Ruptura de simetría hermítica. Residuo relativo: {symmetry_rel:.3e}"
        )

    @staticmethod
    def _is_kahler_symplectic_rotation(t_matrix: NDArray[np.float64]) -> bool:
        r"""
        Verifica si $$T \in \mathrm{Sp}(2n, \mathbb{R}) \cap \mathrm{SO}(2n, \mathbb{R}) \cong \mathrm{U}(n)$$
        con parte simétrica positiva no degenerada ($$\operatorname{tr}(T) > \varepsilon_{\mathrm{crit}}$$):
          1. $$d = 2n \ge 2$$
          2. $$\|T^\top T - I_{2n}\|_F \le \varepsilon_{\mathrm{sym}} \sqrt{2n}$$ (ortogonalidad)
          3. $$\|T^\top \Omega T - \Omega\|_F \le \varepsilon_{\mathrm{sym}} \sqrt{2n}$$ (simplecticidad)
          4. $$\operatorname{tr}(T) > \varepsilon_{\mathrm{crit}}$$ (excluye operadores puramente antisimétricos como $$\Omega$$)
        """
        d = t_matrix.shape[0]
        if d < 2 or d % 2 != 0:
            return False
        if float(np.trace(t_matrix)) <= _EPS_CRITICAL:
            return False

        scale = math.sqrt(float(d))
        eye_d = np.eye(d, dtype=np.float64)
        ortho_err = float(la.norm(t_matrix.T @ t_matrix - eye_d, ord="fro")) / scale
        if ortho_err > _SP_REL_TOL:
            return False

        n = d // 2
        eye_n = np.eye(n, dtype=np.float64)
        zero_n = np.zeros((n, n), dtype=np.float64)
        omega = np.block([[zero_n, eye_n], [-eye_n, zero_n]])
        sp_err = float(la.norm(t_matrix.T @ omega @ t_matrix - omega, ord="fro")) / scale
        return sp_err <= _SP_REL_TOL

    def _verify_tensor_symmetry(self, t_matrix: NDArray[np.float64]) -> float:
        """Fachada de compatibilidad: residuo relativo de simetría estricta (FASE 1.3)."""
        _herm, symmetry_rel, _ok = self._phase1_certify_self_adjointness(
            self._phase1_validate_tensor_typing(t_matrix),
            allow_kahler_symplectic_unitary=False,
        )
        del _herm, _ok
        return symmetry_rel

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    def _phase1_svd_wilkinson_spectrum(
        self,
        t_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, float, float, bool]:
        r"""
        FASE 1.4 — Espectro singular y número de condición de Wilkinson:
        $$T = U \Sigma V^\top, \qquad \kappa_2(T) = \frac{\sigma_{\max}}{\sigma_{\min}}$$
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
        FASE 1.5 — Cota de Lipschitz espectral (Daleckii–Krein):
        $$L_{\mathrm{DK}} = \frac{\kappa}{\sigma_{\min} + \varepsilon_W}$$
        """
        self._assert_finite_scalar(coupling_kappa, "tensor.kappa")
        if coupling_kappa < 0.0:
            raise ScalarRegularityError(
                f"tensor.kappa debe ser ≥ 0; recibido {coupling_kappa}."
            )
        floor = self._wilkinson_spectral_floor(
            sigma_min, float(_EPS_CRITICAL), ambient_dim
        )
        bound = float(coupling_kappa) / (float(sigma_min) + floor)
        if not math.isfinite(bound):
            raise WilkinsonConditionBreachError(
                f"Cota de Daleckii–Krein no finita: L={bound}."
            )
        logger.debug(
            "FASE1.5 L_DK=%.6e (κ=%.6e, σ_min=%.6e)",
            bound,
            coupling_kappa,
            sigma_min,
        )
        return bound

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    def _phase1_spectral_masses(
        self,
        s_vals: NDArray[np.float64],
        t_matrix: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        FASE 1.6 — Masas espectrales y checksum de isometría de Frobenius:
        $$\|T\|_* = \sum_i \sigma_i, \qquad \|T\|_F = \Bigl(\sum_i \sigma_i^2\Bigr)^{1/2}, \qquad P = \|T\|_F^2$$
        """
        nuclear = float(np.sum(s_vals))
        fro_svd = float(np.sqrt(np.sum(s_vals * s_vals)))
        fro_dir = float(la.norm(t_matrix, ord="fro"))
        scale = max(1.0, fro_dir)
        if abs(fro_svd - fro_dir) > (
            _CHECKSUM_FMA_SLACK * _MACHINE_EPS * scale * max(t_matrix.shape[0], 1)
        ):
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
        FASE 1.7 — Saneamiento de escalares de negocio $$(\mathrm{ROI}, \Psi, \kappa, \gamma)$$.
        Soporta tanto el contrato matricial de `ImprobabilityResult` como el
        contrato monádico con `metadata` de `improbability_drive.py`.
        """
        if not isinstance(result, ImprobabilityResult) and not (
            hasattr(result, "tensor") or hasattr(result, "penalty")
        ):
            raise TypeError("result debe ser ImprobabilityResult")

        if hasattr(result, "tensor") and getattr(result, "tensor") is not None:
            roi = self._as_finite_scalar(result.roi_value, "roi_value")
            psi = self._as_finite_scalar(result.psi_value, "psi_value")
            if psi < 0.0:
                raise ScalarRegularityError(
                    f"psi_value debe ser ≥ 0; recibido {psi}."
                )
            kappa = self._as_finite_scalar(result.tensor.kappa, "tensor.kappa")
            gamma = self._as_finite_scalar(result.tensor.gamma, "tensor.gamma")
            if kappa < 0.0:
                raise ScalarRegularityError(
                    f"tensor.kappa debe ser ≥ 0; recibido {kappa}."
                )
            declared = bool(getattr(result, "is_valid", True))
        else:
            meta = getattr(result, "metadata", None) or {}
            roi = self._as_finite_scalar(meta.get("roi_input", 1.0), "roi_value")
            psi = self._as_finite_scalar(meta.get("psi_input", 1.0), "psi_value")
            if psi < 0.0:
                raise ScalarRegularityError(
                    f"psi_value debe ser ≥ 0; recibido {psi}."
                )
            kappa = self._as_finite_scalar(meta.get("kappa", 1.0), "tensor.kappa")
            gamma = self._as_finite_scalar(meta.get("gamma", 0.5), "tensor.gamma")
            if kappa < 0.0:
                raise ScalarRegularityError(
                    f"tensor.kappa debe ser ≥ 0; recibido {kappa}."
                )
            declared = bool(getattr(result, "success", True))

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
        FASE 1.8 — Subobjeto local de observación en $$\Omega_3$$.
        """
        if not result_declared_valid:
            return ImprobabilityHeytingVerdict.DEGRADED
        if condition_number > _SOFT_CONDITION_NUMBER:
            return ImprobabilityHeytingVerdict.DEGRADED
        return ImprobabilityHeytingVerdict.COHERENT

    # ── FASE 1.9 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_initial_poincare_kam_observation(
        s_vals: NDArray[np.float64],
    ) -> PoincareSymplecticInvariants:
        r"""
        FASE 1.9 — Evaluación inicial del margen diofántico KAM sobre las
        frecuencias singulares logarítmicas $$\omega_i = \ln(\max(\sigma_i, \varepsilon_{\mathrm{crit}}))$$:
        $$|p\,\omega_j + q\,\omega_k| \ge \gamma_{\mathrm{KAM}}(|p| + |q|)^{-\tau}$$
        """
        log_freqs = np.log(np.maximum(s_vals, _EPS_CRITICAL))
        if log_freqs.size < 2:
            kam_margin = 1.0
        else:
            w0 = float(log_freqs[0])
            w1 = float(log_freqs[-1])
            if abs(w0) < _EPS_CRITICAL and abs(w1) < _EPS_CRITICAL:
                kam_margin = 1.0
            else:
                min_margin = float("inf")
                for p in range(1, 5):
                    for q in range(-4, 5):
                        if q == 0:
                            continue
                        order = p + abs(q)
                        res = abs(p * w0 + q * w1)
                        thresh = _KAM_AGENT_GAMMA / (
                            float(order) ** _KAM_AGENT_TAU
                        )
                        margin = (res - thresh) * (
                            float(order) ** _KAM_AGENT_TAU
                        )
                        if margin < min_margin:
                            min_margin = margin
                kam_margin = float(np.clip(min_margin, -1.0, 10.0))

        s_min = float(s_vals[-1]) if s_vals.size > 0 else 1.0
        s_max = float(s_vals[0]) if s_vals.size > 0 else 1.0
        return PoincareSymplecticInvariants(
            williamson_symplectic_min=s_min,
            williamson_symplectic_max=s_max,
            poincare_cartan_capacity=s_min,
            kam_diophantine_margin=kam_margin,
            floquet_monodromy_radius=s_max,
            floquet_lyapunov_exponent=float(np.mean(np.abs(log_freqs))),
            arnold_diffusion_velocity=0.0,
        )

    # ── FASE 1.Ω · Composición de Observación y Handoff a Fase 2 ──────────
    def observe_improbability_tensor(
        self,
        result: ImprobabilityResult,
    ) -> Phase1ImprobabilityObservation:
        r"""
        FASE 1.Ω — Composición de Observación espectral.

        $$\mathrm{ObserveImprob} = \chi_1 \circ \mathrm{Poincar\acute{e}}_1 \circ \mathrm{Mass}
          \circ L_{\mathrm{DK}} \circ \mathrm{SVD} \circ \mathrm{Weyl/K\ddot{a}hler}
          \circ \mathrm{Type} \circ \mathrm{Scalars}$$
        """
        roi, psi, kappa, gamma, declared = self._phase1_validate_business_scalars(
            result
        )
        if hasattr(result, "tensor") and getattr(result, "tensor") is not None:
            raw_input = result.tensor.matrix
        else:
            pen = max(float(getattr(result, "penalty", 1.0) or 1.0), 1.0)
            s_pen = math.sqrt(pen)
            raw_input = np.diag([s_pen, 1.0 / s_pen]).astype(np.float64)

        raw = self._phase1_validate_tensor_typing(raw_input)
        herm, symmetry_rel, is_herm = self._phase1_certify_self_adjointness(
            raw, allow_kahler_symplectic_unitary=True
        )
        s_vals, s_max, s_min, cond, invertible = (
            self._phase1_svd_wilkinson_spectrum(herm)
        )
        lipschitz = self._phase1_daleckii_krein_lipschitz(
            s_min, kappa, herm.shape[0]
        )
        nuclear, fro, purity = self._phase1_spectral_masses(s_vals, herm)
        verdict = self._phase1_local_heyting_verdict(cond, declared)
        poincare_obs = self._phase1_initial_poincare_kam_observation(s_vals)
        rsi_state = ImprobabilityAgentRSILevel3Engine.get_meta_state()

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
            poincare_invariants=poincare_obs,
            rsi_telemetry=rsi_state,
        )

    def handoff_phase1_to_phase2(
        self,
        result: ImprobabilityResult,
        phase2_orienter: Optional[Phase2_LieDeformationOrienter] = None,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Union[Phase1ImprobabilityObservation, Phase2ImprobabilityOrientation]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1 Y CONTINUACIÓN / INICIO DE LA FASE 2:
        Ejecuta el morfismo observacional $$\Phi_1(\mathrm{ImprobabilityResult}) \to \mathrm{Phase1ImprobabilityObservation}$$
        y, cuando se suministra el orientador de Fase 2 (`phase2_orienter`),
        transfiere covariantemente el certificado a `phase2_orienter.ingest_from_phase1_handoff(...)`,
        materializando la composición funtorial $$\Phi_2 \circ \Phi_1$$.
        """
        observation = self.observe_improbability_tensor(result)
        if phase2_orienter is not None:
            return phase2_orienter.ingest_from_phase1_handoff(
                observation,
                require_symplectic=require_symplectic,
                strict=strict,
            )
        return observation


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: TRANSPORTE DE LIE, INVARIANZA DE LIOUVILLE, ESPECTRO SIMPLÉCTICO     ║
# ║         DE WILLIAMSON Y MONODROMÍA DE FLOQUET-KAM                            ║
# ║         (Phase2_LieDeformationOrienter)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class Phase2ImprobabilityOrientation:
    r"""
    Artefacto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Encierra el logaritmo principal de Lie $$\operatorname{Re}\operatorname{Log} T$$,
    los residuos de pullback simpléctico $$\|T^\top\Omega T - \Omega\|_F$$ y de
    Liouville $$|\det T - 1|$$, la entropía espectral de Shannon y los invariantes
    celestes de Williamson, Floquet-Kreĭn, KAM y difusión de Arnold.
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
    poincare_invariants: PoincareSymplecticInvariants = field(
        default_factory=PoincareSymplecticInvariants
    )
    rsi_telemetry: ImprobabilityAgentRSITelemetry = field(
        default_factory=ImprobabilityAgentRSITelemetry
    )


class Phase2_LieDeformationOrienter(Phase1_ImprobabilityObserver):
    r"""
    FASE 2: Orientador de Deformación en el Álgebra de Lie $$\mathfrak{sp}(2n, \mathbb{R})$$,
    Volumen de Liouville, Espectro Simpléctico de Williamson y Monodromía de Floquet ($\Phi_2$).

    Inicia formalmente en `ingest_from_phase1_handoff` y `_phase2_consume_phase1_certificate`
    como continuación directa del último método de la Fase 1 (`handoff_phase1_to_phase2`),
    y culmina en `handoff_phase2_to_phase3`.
    """

    # ── FASE 2.0 · Continuación Directa de la Fase 1 ──────────────────────
    def ingest_from_phase1_handoff(
        self,
        observation: Phase1ImprobabilityObservation,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Phase2ImprobabilityOrientation:
        r"""
        CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 1 (`handoff_phase1_to_phase2`):
        Ingiere el certificado `Phase1ImprobabilityObservation` y ejecuta la orientación
        en el álgebra de Lie $$\mathfrak{sp}(2n, \mathbb{R})$$ y el análisis celeste de Poincaré.
        """
        return self.orient_lie_deformation(
            observation,
            require_symplectic=require_symplectic,
            strict=strict,
        )

    def _phase2_consume_phase1_certificate(
        self,
        observation: Phase1ImprobabilityObservation,
    ) -> Tuple[Phase1ImprobabilityObservation, NDArray[np.float64], int]:
        r"""
        FASE 2.0 — Ingesta funtorial del certificado de FASE 1.Ω.
        Verifica la coherencia $$(T, d, \kappa_2)$$ y entrega el objeto de
        trabajo de Orient *sin re-tipado ni re-SVD*.
        """
        if not isinstance(observation, Phase1ImprobabilityObservation):
            raise TypeError(
                "observation debe ser Phase1ImprobabilityObservation"
            )
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
        FASE 2.1 — Forma simpléctica canónica $$\Omega$$ (sólo si $$d = 2n$$):
        $$\Omega = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix},
          \qquad \Omega^\top = -\Omega, \qquad \Omega^2 = -I_{2n}$$
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
        Fachada de compatibilidad para $$\Omega \in \mathfrak{so}(2n) \cap \mathrm{Sp}(2n)$$.
        Si $$d$$ es impar eleva `TensorDimensionError` sin inmersión silenciosa.
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
        FASE 2.2 — Logaritmo principal de Lie $$\operatorname{Log} T$$.
        Para $$T \in \mathrm{GL}(d, \mathbb{R})$$ sin espectro en $$\mathbb{R}_{\le 0}$$,
        existe un único logaritmo real principal.
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
        FASE 2.3 — Residuo de pullback simpléctico:
        $$\delta_{\mathrm{Sp}} = \|T^\top \Omega T - \Omega\|_F$$
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
        FASE 2.4 — Residuo de volumen de Poincaré-Liouville $$|\det T - 1|$$.
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
        FASE 2.5 — Residuo de pertenencia al álgebra de Lie $$\mathfrak{sp}(2n, \mathbb{R})$$:
        $$X \in \mathfrak{sp}(2n, \mathbb{R}) \iff \Omega X + X^\top \Omega = 0$$
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
        FASE 2.6 — Entropía espectral de von Neumann / Shannon:
        $$p_i = \frac{\sigma_i}{\sum_j \sigma_j}, \qquad H = -\sum_i p_i \log_2 p_i \in [0, \log_2 d]$$
        """
        if s_vals_hint is None:
            s_vals = np.asarray(la.svdvals(t_matrix), dtype=np.float64)
        else:
            s_vals = np.asarray(s_vals_hint, dtype=np.float64)
        total = float(np.sum(s_vals))
        if total <= _MACHINE_EPS:
            return 0.0
        probs = s_vals / total
        entropy = (
            -float(np.sum(probs * np.log(np.clip(probs, _MACHINE_EPS, 1.0))))
            * _LOG2E
        )
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
        FASE 2.7 — Checksum de dualidad $$\mathrm{Sp}(2n) \implies \det T = +1 \;\wedge\; \operatorname{Log} T \in \mathfrak{sp}(2n)$$.
        """
        if not is_even or not math.isfinite(symplectic_deviation):
            return
        if symplectic_deviation > _SP_REL_TOL:
            return
        if math.isfinite(liouville_deviation) and liouville_deviation > max(
            _DET_TOL, 1.0e-8
        ):
            raise SpectralChecksumError(
                "Checksum Sp⇒Liouville corrupto: "
                f"δ_Sp={symplectic_deviation:.3e} pero |det−1|={liouville_deviation:.3e}."
            )
        if math.isfinite(lie_sp_residual) and lie_sp_residual > max(
            _LIE_SP_TOL, 1.0e-6
        ):
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
        FASE 2.8 — Clasificación Sp/Liouville y subobjeto local en $$\Omega_3$$.
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
                math.isfinite(symplectic_deviation)
                and symplectic_deviation > _SP_SOFT_TOL
            ) or (
                math.isfinite(liouville_deviation)
                and liouville_deviation > 1.0e-6
            )
            components.append(
                ImprobabilityHeytingVerdict.VETOED
                if hard
                else ImprobabilityHeytingVerdict.DEGRADED
            )
        elif not is_symp:
            if (
                math.isfinite(symplectic_deviation)
                and symplectic_deviation > _SP_SOFT_TOL
            ) or (
                math.isfinite(liouville_deviation)
                and liouville_deviation > 1.0e-6
            ):
                components.append(ImprobabilityHeytingVerdict.DEGRADED)
        return is_symp, self._heyting_join(*components)

    # ── FASE 2.9 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_poincare_williamson_floquet_invariants(
        observation: Phase1ImprobabilityObservation,
        t_matrix: NDArray[np.float64],
        omega: Optional[NDArray[np.float64]],
        symplectic_deviation: float,
    ) -> PoincareSymplecticInvariants:
        r"""
        FASE 2.9 — Teorema de Williamson, Monodromía de Floquet-Kreĭn y
        Velocidad de Difusión de Nekhoroshev-Arnold [A5]:
          1. Multiplicadores de Floquet $$\lambda_i \in \sigma(T)$$:
             $$\rho_{\mathrm{Floquet}} = \max_i |\lambda_i|, \qquad
               \mu_{\mathrm{Floquet}} = \frac{1}{d}\sum_i |\ln|\lambda_i||$$
          2. Espectro simpléctico de Williamson (para $$d = 2n$$):
             Sea $$|T| = (T^\top T)^{1/2}$$ el factor polar simétrico positivo. Los
             autovalores de $$i\Omega |T|$$ vienen en pares $$\pm \nu_k$$ con $$\nu_k > 0$$.
          3. Velocidad de difusión de Arnold:
             $$v_{\mathrm{Arnold}} = \exp\!\left(-\frac{1}{\mu_{\mathrm{Sp}} + \varepsilon_{\mathrm{crit}}}\right),
               \qquad \mu_{\mathrm{Sp}} = \frac{\delta_{\mathrm{Sp}}}{1 + \|T\|_F}$$
        """
        base_inv = observation.poincare_invariants
        eigs_t = la.eigvals(t_matrix)
        abs_eigs = np.maximum(np.abs(eigs_t), _EPS_CRITICAL)
        floquet_radius = float(np.max(abs_eigs))
        floquet_lyap = float(np.mean(np.abs(np.log(abs_eigs))))

        if omega is not None:
            # Módulo polar positivo |T| = U Σ U^T a partir de T^T T
            u_polar, s_polar, _ = la.svd(t_matrix, full_matrices=False)
            abs_t = (u_polar * s_polar) @ u_polar.T
            williamson_eigs = np.abs(la.eigvals(omega @ abs_t))
            w_min = float(np.min(williamson_eigs)) if williamson_eigs.size > 0 else 1.0
            w_max = float(np.max(williamson_eigs)) if williamson_eigs.size > 0 else 1.0
            # Capacidad de Poincaré-Cartan
            pc_capacity = w_min
        else:
            w_min = base_inv.williamson_symplectic_min
            w_max = base_inv.williamson_symplectic_max
            pc_capacity = base_inv.poincare_cartan_capacity

        if math.isfinite(symplectic_deviation) and symplectic_deviation > _SP_REL_TOL:
            mu_pert = symplectic_deviation / (1.0 + observation.frobenius_norm)
            arnold_vel = float(math.exp(-1.0 / max(mu_pert, 1.0e-6)))
        else:
            arnold_vel = 0.0

        return PoincareSymplecticInvariants(
            williamson_symplectic_min=w_min,
            williamson_symplectic_max=w_max,
            poincare_cartan_capacity=pc_capacity,
            kam_diophantine_margin=base_inv.kam_diophantine_margin,
            floquet_monodromy_radius=floquet_radius,
            floquet_lyapunov_exponent=floquet_lyap,
            arnold_diffusion_velocity=arnold_vel,
        )

    # ── FASE 2.Ω · Composición de Orientación y Handoff a Fase 3 ──────────
    def orient_lie_deformation(
        self,
        observation: Phase1ImprobabilityObservation,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Phase2ImprobabilityOrientation:
        r"""
        FASE 2.Ω — Composición terminal Orient (Log + Sp + Liouville + Williamson/Floquet + H).

        **Continuación funtorial de FASE 1.Ω**: consume
        ``Phase1ImprobabilityObservation`` vía FASE 2.0.
        """
        observation, t_matrix, d = self._phase2_consume_phase1_certificate(
            observation
        )

        omega, is_even = self._phase2_construct_canonical_omega(d)
        lie_log, imag_res, branch_ok = self._phase2_principal_lie_logarithm(
            t_matrix
        )
        sp_dev = self._phase2_symplectic_pullback_residual(t_matrix, omega)
        det, liouville = self._phase2_liouville_determinant(t_matrix)
        lie_sp = self._phase2_lie_algebra_sp_residual(lie_log, omega)
        self._phase2_crosscheck_sp_liouville(is_even, sp_dev, liouville, lie_sp)

        entropy = self._phase2_shannon_spectral_entropy(t_matrix, None)
        poincare_inv = self._phase2_poincare_williamson_floquet_invariants(
            observation=observation,
            t_matrix=t_matrix,
            omega=omega,
            symplectic_deviation=sp_dev,
        )

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
            poincare_invariants=poincare_inv,
            rsi_telemetry=observation.rsi_telemetry,
        )

    def handoff_phase2_to_phase3(
        self,
        observation: Phase1ImprobabilityObservation,
        phase3_decider: Optional[Phase3_HeytingImprobabilityDecider] = None,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Union[Phase2ImprobabilityOrientation, Phase3ImprobabilityDecision]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 2 Y CONTINUACIÓN / INICIO DE LA FASE 3:
        Ejecuta el morfismo de orientación $$\Phi_2(\mathrm{Phase1ImprobabilityObservation}) \to \mathrm{Phase2ImprobabilityOrientation}$$
        y, cuando se suministra el decisor de Fase 3 (`phase3_decider`),
        transfiere covariantemente el certificado a `phase3_decider.ingest_from_phase2_handoff(...)`,
        materializando la composición funtorial $$\Phi_3 \circ \Phi_2$$.
        """
        orientation = self.ingest_from_phase1_handoff(
            observation,
            require_symplectic=require_symplectic,
            strict=strict,
        )
        if phase3_decider is not None:
            return phase3_decider.ingest_from_phase2_handoff(
                orientation,
                require_symplectic=require_symplectic,
                strict=strict,
            )
        return orientation


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: EVALUACIÓN DE ESTRÉS, RETÍCULO DE HEYTING Ω₃, CLAUSURA RSI-L3        ║
# ║         Y ACTUACIÓN CIBER-FÍSICA CROWBAR BT151 EN SILICIO (GPIO14)           ║
# ║         (Phase3_HeytingImprobabilityDecider & ImprobabilityDriveAgent)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class Phase3ImprobabilityDecision:
    r"""
    Artefacto terminal de la FASE 3 y objeto inicial del Seal (Ω).

    Certifica el estrés ajustado $$\sigma^\star = \|T\|_2 \cdot \mathrm{ROI}/\Psi_\varepsilon$$,
    la clasificación en $$\Omega_3$$ y la clausura del punto fijo RSI-L3 *antes*
    de disparar el Crowbar (una sola actuación ciber-física en el agente).
    """

    adjusted_stress: float
    operator_norm: float
    psi_regularized: float
    phase3_verdict: ImprobabilityHeytingVerdict
    lattice_verdict: ImprobabilityHeytingVerdict
    orientation: Phase2ImprobabilityOrientation
    poincare_invariants: PoincareSymplecticInvariants = field(
        default_factory=PoincareSymplecticInvariants
    )
    rsi_telemetry: ImprobabilityAgentRSITelemetry = field(
        default_factory=ImprobabilityAgentRSITelemetry
    )


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
    poincare_invariants: PoincareSymplecticInvariants = field(
        default_factory=PoincareSymplecticInvariants
    )
    rsi_telemetry: ImprobabilityAgentRSITelemetry = field(
        default_factory=ImprobabilityAgentRSITelemetry
    )


class Phase3_HeytingImprobabilityDecider(Phase2_LieDeformationOrienter):
    r"""
    FASE 3: Decisor en el Retículo de Heyting $$\Omega_3$$, Evaluador de Estrés
    Ajustado y Clausura de Punto Fijo RSI de Nivel 3 ($\Phi_3$).

    Inicia formalmente en `ingest_from_phase2_handoff` y `_phase3_consume_phase2_certificate`
    como continuación directa del último método de la Fase 2 (`handoff_phase2_to_phase3`),
    y culmina en `decidir_gobernanza_improbabilidad` y `finalize_nested_pipeline`.
    """

    # ── FASE 3.0 · Continuación Directa de la Fase 2 ──────────────────────
    def ingest_from_phase2_handoff(
        self,
        orientation: Phase2ImprobabilityOrientation,
        crowbar: Optional[CrowbarActuator] = None,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Phase3ImprobabilityDecision:
        r"""
        CONTINUACIÓN DIRECTA DEL ÚLTIMO MÉTODO DE LA FASE 2 (`handoff_phase2_to_phase3`):
        Ingiere formalmente `Phase2ImprobabilityOrientation`, evalúa la ecuación de
        estado de estrés ajustado, resuelve el retículo de Heyting $$\Omega_3$$ y
        ejecuta la clausura recursiva RSI-L3.
        """
        return self.decidir_gobernanza_improbabilidad(
            orientation,
            crowbar=crowbar,
            require_symplectic=require_symplectic,
            strict=strict,
        )

    def _phase3_consume_phase2_certificate(
        self,
        orientation: Phase2ImprobabilityOrientation,
    ) -> Tuple[Phase2ImprobabilityOrientation, NDArray[np.float64], int]:
        r"""
        FASE 3.0 — Ingesta funtorial del certificado de FASE 2.Ω.
        Verifica la coherencia de $$T$$ y $$\operatorname{Log} T$$ y entrega
        el objeto de trabajo de Decide *sin recomputar el logaritmo*.
        """
        if not isinstance(orientation, Phase2ImprobabilityOrientation):
            raise TypeError(
                "orientation debe ser Phase2ImprobabilityOrientation"
            )
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
        FASE 3.1 — Norma operatorial $$\|T\|_2 = \sigma_{\max}$$.
        Reutiliza $$\sigma_{\max}$$ del certificado de FASE 1 si está anidado.
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
        FASE 3.2 — Regularización de Tikhonov del denominador $$\Psi$$:
        $$\Psi_\varepsilon = \max(\Psi, \varepsilon_{\mathrm{mach}})$$
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
        FASE 3.3 — Ecuación de estado del estrés ajustado [A3]:
        $$\sigma^\star = \|T\|_2 \cdot \frac{\mathrm{ROI}}{\Psi_\varepsilon}$$
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
        FASE 3.4 — Subobjeto local de estrés en $$\Omega_3$$:
        $$\chi_\sigma = \begin{cases}
          \mathrm{VETOED}   & \sigma^\star \ge \sigma_{\mathrm{high}} \\
          \mathrm{DEGRADED} & \sigma^\star \ge \sigma_{\mathrm{soft}} \\
          \mathrm{COHERENT} & \text{en otro caso}
        \end{cases}$$
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
        FASE 3.5 — Supremo del retículo $$\Omega_3$$:
        $$\chi = \chi_1 \vee \chi_2 \vee \chi_\sigma \;\bigl(\vee \chi_{\mathrm{Sp}}\bigr)_{\mathrm{opt}}$$
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

    # ── FASE 3.Ω · Composición Decide y Clausura RSI Nivel 3 ──────────────
    def decidir_gobernanza_improbabilidad(
        self,
        orientation: Phase2ImprobabilityOrientation,
        crowbar: Optional[CrowbarActuator] = None,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> Phase3ImprobabilityDecision:
        r"""
        FASE 3.Ω — Composición terminal Decide (estrés + $$\Omega_3$$ + RSI-L3).

        **Continuación funtorial de FASE 2.Ω**: consume
        ``Phase2ImprobabilityOrientation`` vía FASE 3.0.
        """
        del crowbar  # Actuación única en el Seal del agente
        orientation, t_matrix, _d = self._phase3_consume_phase2_certificate(
            orientation
        )

        op_norm = self._phase3_operator_norm(orientation, t_matrix)
        psi_eps = self._phase3_regularize_psi(orientation.initial_psi)
        stress = self._phase3_adjusted_business_stress(
            op_norm, orientation.initial_roi, psi_eps
        )
        stress_verdict = self._phase3_stress_heyting_verdict(stress)
        lattice = self._phase3_lattice_conjunction(
            orientation, stress_verdict, require_symplectic=require_symplectic
        )

        preliminary_decision = Phase3ImprobabilityDecision(
            adjusted_stress=float(stress),
            operator_norm=float(op_norm),
            psi_regularized=float(psi_eps),
            phase3_verdict=stress_verdict,
            lattice_verdict=lattice,
            orientation=orientation,
            poincare_invariants=orientation.poincare_invariants,
            rsi_telemetry=orientation.rsi_telemetry,
        )

        sealed_decision = self.finalize_nested_pipeline(preliminary_decision)

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
        return sealed_decision

    @staticmethod
    def finalize_nested_pipeline(
        decision: Phase3ImprobabilityDecision,
    ) -> Phase3ImprobabilityDecision:
        r"""
        CLAUSURA TERMINAL DE LAS 3 FASES ANIDADAS ($$\Phi_3 \circ \Phi_2 \circ \Phi_1$$):
        Ejecuta el paso de punto fijo de Banach del motor de Automejora Recursiva
        de Nivel 3 (`ImprobabilityAgentRSILevel3Engine`) y adjunta la telemetría
        metacognitiva al certificado de decisión.
        """
        p_inv = decision.poincare_invariants
        rsi_telemetry = ImprobabilityAgentRSILevel3Engine.commit_level3_fixed_point(
            wilkinson_cond=decision.orientation.condition_number,
            adjusted_stress=decision.adjusted_stress,
            arnold_velocity=p_inv.arnold_diffusion_velocity,
            floquet_exponent=p_inv.floquet_lyapunov_exponent,
            verdict=decision.lattice_verdict,
        )
        return replace(decision, rsi_telemetry=rsi_telemetry)


# ─────────────────────────────────────────────────────────────────────────────
# SOBERANO DE CONTROL ESPECTRAL DE IMPROBABILIDAD (Orquestador OODA + Silicio)
# ─────────────────────────────────────────────────────────────────────────────
class ImprobabilityDriveAgent(Morphism, Phase3_HeytingImprobabilityDecider):
    r"""
    Agente Soberano que unifica el motor de improbabilidad con el retículo $$\Omega_3$$,
    la Mecánica Celeste de Poincaré-KAM, el motor RSI-L3 y el tiristor Crowbar BT151.

    Endofuntor de gobernanza:
    $$\mathcal{OODA}_{\mathrm{Improb}} = \mathrm{Seal}_{\mathrm{GPIO14}} \circ \Phi_3 \circ \Phi_2 \circ \Phi_1 :
      \mathbf{ImprobabilityResult} \longrightarrow \mathtt{ImprobabilityGovernanceState}$$
    """

    def __init__(
        self,
        crowbar_actuator: Optional[CrowbarActuator] = None,
        raise_on_veto: bool = False,
    ) -> None:
        super().__init__()
        self._crowbar = crowbar_actuator or LoggingCrowbarActuator()
        self._raise_on_veto = raise_on_veto

    # ── FASE Ω.1 · Actuación Ciber-Física Crowbar Única (GPIO14) ──────────
    def _phase_omega_apply_crowbar(
        self,
        verdict: ImprobabilityHeytingVerdict,
    ) -> CrowbarBypassAction:
        """Dispara el actuador de silicio según el subobjeto de Heyting (una sola vez)."""
        if verdict == ImprobabilityHeytingVerdict.VETOED:
            action = CrowbarBypassAction.HARD_SHORT
            self._crowbar.trigger_crowbar_bypass(action)
            return action
        if verdict == ImprobabilityHeytingVerdict.DEGRADED:
            action = CrowbarBypassAction.WATCHDOG_PULSE
            self._crowbar.trigger_crowbar_bypass(action)
            return action
        return CrowbarBypassAction.NONE

    # ── FASE Ω.2 · Sellado Criptográfico SHA-256 ──────────────────────────
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
        """FASE Ω.2 — Sellado del objeto terminal inmutable del funtor OODA."""
        timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if decision is not None:
            ori = decision.orientation
            stress = decision.adjusted_stress
            cond = ori.condition_number
            sp_dev = ori.symplectic_deviation
            liouville = ori.liouville_determinant_deviation
            entropy = ori.entropy_bits
            is_symp = ori.is_symplectic
            lipschitz = (
                ori.phase1.lipschitz_bound if ori.phase1 is not None else 0.0
            )
            poincare_inv = decision.poincare_invariants
            rsi_tel = decision.rsi_telemetry
        else:
            stress = fallback_stress
            cond = fallback_cond
            sp_dev = fallback_sp
            liouville = float("inf")
            entropy = 0.0
            is_symp = False
            lipschitz = 0.0
            poincare_inv = PoincareSymplecticInvariants()
            rsi_tel = ImprobabilityAgentRSILevel3Engine.commit_level3_fixed_point(
                wilkinson_cond=fallback_cond,
                adjusted_stress=fallback_stress,
                arnold_velocity=1.0,
                floquet_exponent=1.0,
                verdict=verdict,
            )

        prov_payload = (
            f"{verdict.name}-{stress:.6e}-{cond:.6e}-"
            f"{sp_dev:.6e}-{latency_ms:.3f}-{__version__}"
        )
        provenance_hash = hashlib.sha256(
            prov_payload.encode("utf-8")
        ).hexdigest()

        return ImprobabilityGovernanceState(
            verdict=verdict,
            crowbar_report=action_report,
            adjusted_stress=float(stress),
            wilkinson_condition=float(cond),
            symplectic_deviation=(
                float(sp_dev) if math.isfinite(sp_dev) else float("inf")
            ),
            is_epistemologically_valid=(
                verdict != ImprobabilityHeytingVerdict.VETOED
            ),
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"{diagnostic_msg} Latencia de lazo: {latency_ms:.3f} ms.",
            liouville_deviation=(
                float(liouville) if math.isfinite(liouville) else float("inf")
            ),
            entropy_bits=float(entropy),
            lipschitz_bound=float(lipschitz),
            is_symplectic=bool(is_symp),
            agent_version=__version__,
            policy_require_symplectic=require_symplectic,
            poincare_invariants=poincare_inv,
            rsi_telemetry=rsi_tel,
        )

    def _build_vetoed_state(
        self,
        note: str,
        t_start_nano: int,
        *,
        require_symplectic: bool = False,
    ) -> ImprobabilityGovernanceState:
        """Estado de veto de emergencia con trazabilidad forense (fail-secure)."""
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

    # ── Compositor Público OODA de las 3 Fases Anidadas ───────────────────
    def execute_improbability_governance(
        self,
        result: ImprobabilityResult,
        *,
        require_symplectic: bool = False,
        strict: bool = True,
    ) -> ImprobabilityGovernanceState:
        r"""
        Orquesta el ciclo OODA completo sobre el motor de improbabilidad ciego
        encadenando explícitamente las 3 fases anidadas:
          1. `handoff_phase1_to_phase2` (Fase 1 $$\to$$ Fase 2)
          2. `handoff_phase2_to_phase3` (Fase 2 $$\to$$ Fase 3)
          3. `finalize_nested_pipeline` $$\to$$ `_phase_omega_apply_crowbar` $$\to$$ `_phase_omega_seal_governance_state`
        """
        t_start_nano = time.perf_counter_ns()
        diagnostic_msg = "Gobernanza Espectral de Improbabilidad Estable."
        decision: Optional[Phase3ImprobabilityDecision] = None
        verdict = ImprobabilityHeytingVerdict.COHERENT
        pending_raise: Optional[BaseException] = None
        crowbar_already = False

        try:
            # ── FASE 1 ──► FASE 2 ──► FASE 3 (Handoffs Anidados) ──────────
            observation = self.handoff_phase1_to_phase2(result)
            assert isinstance(observation, Phase1ImprobabilityObservation)

            decision_or_ori = self.handoff_phase2_to_phase3(
                observation,
                phase3_decider=self,
                require_symplectic=require_symplectic,
                strict=strict,
            )
            assert isinstance(decision_or_ori, Phase3ImprobabilityDecision)
            decision = decision_or_ori
            orientation = decision.orientation

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
            diagnostic_msg = (
                f"ANILQUILACIÓN CUÁNTICA POR INCOHERENCIA DE ENTRADA: {exc}"
            )
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
    # Fábricas Canónicas de Referencia (Calibración y Verificación)
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
        r"""Tensor identidad $$T = I_d$$ ($$\kappa_2 = 1$$, Sp nativo si $$d$$ es par)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        tensor = ImprobabilityTensor(
            matrix=np.eye(int(dimension_d), dtype=np.float64),
            kappa=float(kappa),
            gamma=float(gamma),
        )
        return ImprobabilityResult(
            tensor=tensor,
            roi_value=float(roi),
            psi_value=float(psi),
            is_valid=True,
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
        r"""Tensor SPD aleatorio $$T = A A^\top + \varepsilon I$$ (drive típico, no Sp)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(int(dimension_d), int(dimension_d)))
        spd = raw @ raw.T + 1.0e-3 * np.eye(int(dimension_d))
        tensor = ImprobabilityTensor(
            matrix=spd.astype(np.float64), kappa=float(kappa), gamma=0.5
        )
        return ImprobabilityResult(
            tensor=tensor,
            roi_value=float(roi),
            psi_value=float(psi),
            is_valid=True,
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
        Rotación unitario-simpléctica canónica en $$\mathrm{Sp}(2n, \mathbb{R}) \cap \mathrm{SO}(2n, \mathbb{R}) \cong \mathrm{U}(n)$$:
        $$T = \begin{pmatrix} \cos\theta \, I_n & \sin\theta \, I_n \\ -\sin\theta \, I_n & \cos\theta \, I_n \end{pmatrix}$$
        """
        if n_pairs < 1:
            raise ValueError(f"n_pairs debe ser ≥ 1; recibido {n_pairs}")
        n = int(n_pairs)
        c = math.cos(theta)
        s = math.sin(theta)
        eye = np.eye(n, dtype=np.float64)
        t_matrix = np.block([[c * eye, s * eye], [-s * eye, c * eye]])
        tensor = ImprobabilityTensor(
            matrix=t_matrix, kappa=float(kappa), gamma=0.5
        )
        return ImprobabilityResult(
            tensor=tensor,
            roi_value=float(roi),
            psi_value=float(psi),
            is_valid=True,
        )

    @staticmethod
    def symplectic_boost_result(
        n_pairs: int = 1,
        eta: float = 0.25,
        *,
        roi: float = 0.5,
        psi: float = 1.0,
        kappa: float = 1.0,
    ) -> ImprobabilityResult:
        r"""
        Transformación hiperbólica de Bogoliubov (Squeeze Simpléctico Simétrico)
        en $$\mathrm{Sp}(2n, \mathbb{R}) \cap \mathrm{Sym}^+(2n, \mathbb{R})$$:
        $$T = \exp\!\left(\eta \begin{pmatrix} 0 & I_n \\ I_n & 0 \end{pmatrix}\right)
            = \begin{pmatrix} \cosh\eta \, I_n & \sinh\eta \, I_n \\ \sinh\eta \, I_n & \cosh\eta \, I_n \end{pmatrix}$$
        Cumple simultáneamente $$T = T^\top$$, $$T \succ 0$$, $$T^\top \Omega T = \Omega$$ y $$\det T = 1$$.
        """
        if n_pairs < 1:
            raise ValueError(f"n_pairs debe ser ≥ 1; recibido {n_pairs}")
        n = int(n_pairs)
        ch = math.cosh(eta)
        sh = math.sinh(eta)
        eye = np.eye(n, dtype=np.float64)
        t_matrix = np.block([[ch * eye, sh * eye], [sh * eye, ch * eye]])
        tensor = ImprobabilityTensor(
            matrix=t_matrix, kappa=float(kappa), gamma=0.5
        )
        return ImprobabilityResult(
            tensor=tensor,
            roi_value=float(roi),
            psi_value=float(psi),
            is_valid=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "ImprobabilityHeytingVerdict",
    "CrowbarBypassAction",
    "CrowbarActuator",
    "LoggingCrowbarActuator",
    "PoincareSymplecticInvariants",
    "ImprobabilityAgentRSITelemetry",
    "ImprobabilityAgentRSILevel3Engine",
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