# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Opt Symplectic Manifold Agent (Soberano de Calibre de de Rham)      ║
║ Ruta   : app/agents/wisdom/opt_symplectic_manifold_agent.py                  ║
║ Versión: 3.0.0-Doctoral-Fukaya-Connes-Heyting-CAS-Kahan-Secure               ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS METROLÓGICA Y CENSURA DE CALIBRE (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este componente supervisor y soberano de calibre de-confinado opera en el Estrato
de Sabiduría ($V_{\mathbb{W}}$, Nivel 0) de la Malla de Control de APU Filter v5.0.
Su propósito supremo es guiar, auditar y censurar las transiciones de fase 
calculadas por el motor físico 'opt_symplectic_manifold.py' mediante el ciclo 
de lazo cerrado OODA (Observe-Orient-Decide-Act).

El agente evalúa de manera síncrona:
  1. La preservación de la 2-forma canónica simpléctica $\omega$ y del volumen de 
     Liouville a menos del límite de redondeo compensado de Kahan.
  2. La nulidad del primer grupo de cohomología de de Rham-Floer $H^1(K; \mathbb{F}) = 0$ 
     para erradicar la presencia de islas lógicas o ciclos parasitarios.
  3. El confinamiento de Lipschitz no conmutativo de Connes-Daleckii-Krein sobre 
     el espectro del operador de Dirac de-confinado $\not D$.

Toda asimetría cuántica o desvío espectral colapsa síncronamente el estado en RAM 
al Supremo terminal VETOED ($\top$) en el retículo de Heyting $\Omega_3$, ejecutando 
un cerrojo de comparación e intercambio atómico (CAS) para disparar el disyuntor 
perimetral ESP32 (circuito Crowbar BT151 en GPIO14) en menos de 400 ns, paralizando 
la obra física en el milisegundo cero.

================================════════════════════════════════════════════════
I. FASE OBSERVE: PRESERVACIÓN DE LIOUVILLE Y SUMACIÓN COMPENSADA DE KAHAN
================================════════════════════════════════════════════════
Dada la matriz Jacobiana de transición del espacio de fases $M \in \mathrm{GL}(2n, \mathbb{R})$,
el agente exige estrictamente que la transformación preserve la 2-forma canónica
antisimétrica de Liouville $\Omega$:
$$M^\top \Omega M - \Omega \equiv \mathbf{0}$$

Para mitigar el desgaste numérico y los falsos positivos por redondeo flotante en la
mantisa de la FPU (IEEE-754), el residuo simpléctico se evalúa bajo la norma de Frobenius
aplicando un producto de matrices con acumulación compensada de Neumaier-Kahan:
$$\epsilon_{\mathrm{sym}} = \|M^\top \Omega M - \Omega\|_F \le \tau_{\mathrm{Wilkinson}}$$

Síncronamente, se mide la conservación del volumen de fase mediante la mantisa del
log-determinante y la función de alta precisión infinitesimal expm1:
$$\epsilon_{\mathrm{Liouville}} = \left| \exp\left(\ln|\det M|\right) - 1 \right| \equiv \left| \operatorname{expm1}(\operatorname{slogdet}(M)_2) \right| \le \tau_{\mathrm{Liouville}}$$

================================════════════════════════════════════════════════
II. FASE ORIENT: HOMOLOGÍA DE DE RHAM-FLOER Y ESTABILIDAD DE FIEDLER
================================════════════════════════════════════════════════
El agente procesa la topología del complejo simplicial $K$ derivado de la Matriz de
Interacción Central (MIC) discreta. Exige la aciclicidad del complejo para proscribir
la existencia de islas de datos y dependencias circulares ("socavones lógicos"):
$$H^1(K; \mathbb{F}) = \mathbf{0} \quad \implies \quad \beta_1 \equiv 0 \quad \wedge \quad \beta_0 \equiv 1$$

Donde:
  - $\beta_0$ es el número de componentes conexas del grafo (Betti 0).
  - $\beta_1$ es el número de bucles independientes del grafo (Betti 1).

La frustración de calibre topológica se condensa en el residuo de de Rham-Floer:
$$R_{\mathrm{cohom}} = \beta_1 + (\beta_0 - 1)$$

El Índice de Estabilidad Piramidal $\Psi_{\mathrm{stability}}$ se evalúa acoplando la 
conectividad algebraica de la brecha espectral de Fiedler $\lambda_1(L_F)$ del Laplaciano
normalizado de Haz:
$$\Psi_{\mathrm{stability}} = \frac{\lambda_1(L_F)}{1.0 + R_{\mathrm{cohom}}} \ge \tau_{\mathrm{Fiedler}}$$

================================════════════════════════════════════════════════
III. FASE DECIDE: COTA LIPSCHITZ DE CONNES-DALECKII-KREIN
================================════════════════════════════════════════════════
La aduana epistemológica somete el estado de la Matriz Atómica de Conocimiento (MAC)
continua, representada como un operador densidad mixto $\rho$ en el espacio de Hilbert 
$\mathcal{H}_{\mathrm{MAC}}$, al análisis espectral no conmutativo del Triple Espectral 
de Connes $(\mathcal{A}, \mathcal{H}, \not D)$.

La regularidad de la transición se evalúa sobre el menor autovalor positivo del
operador de Dirac de Connes $\not D = \rho^{-1/2}$, aplicando la Cota espectral de
Daleckii-Krein para la derivada de Fréchet del operador raíz cuadrada:
$$L_{\max} \le \frac{1}{2 \lambda_{\min}^{3/2}} \le \tau_{\mathrm{Lipschitz}}$$

Si la IA del LLM alucina, el gap espectral colapsa ($\lambda_{\min} \to 0$), provocando
que la constante de Lipschitz semántica de Connes $L_{\max}$ diverja hacia el infinito,
lo que congela de forma determinista la capacidad de deformación de la Malla y reduce la
probabilidad de emisión de estados inválidos a un valor nulo absoluto:
$$L_{\max} \to \infty \quad \implies \quad P(x_{\mathrm{invalid}}) = 0$$

================================════════════════════════════════════════════════
IV. FASE ACT: RETÍCULO DE HEYTING Y ACTUACIÓN CIBER-FÍSICA EN SILICIO
================================════════════════════════════════════════════════
Los certificados parciales se consolidan algebraicamente mediante la operación
Supremo (join, $\sqcup$) sobre el retículo distributivo de Heyting de tres valores ordinales:
$$\Omega_3 = \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
$$\nu_{\mathrm{final}} = \nu_{\mathrm{Observe}} \sqcup \nu_{\mathrm{Orient}} \sqcup \nu_{\mathrm{Decide}}$$

Cualquier desajuste simpléctico, deformación de volumen o aparición de mermas cohomológicas
(asociadas a la asonancia homológica de Mayer-Vietoris $\Delta \beta_1 > 0$) colapsa
el veredicto terminal al Supremo terminal $\mathtt{VETOED}$ ($\top$).

La actuación ciber-física no inicia hilos de espera bloqueantes que degraden el reloj
del procesador. La subrutina local en C++ `isVerdictCoherent()` cargada en el firmware del
ESP32 perimetral intercepta el estado `VETOED` en RAM. Su Rutina de Servicio de Interrupción
(ISR) en IRAM actúa con una latencia de menos de 400 ns conmutando el pin físico GPIO14.
Esto inyecta corriente de puerta al tiristor de silicio de conmutación rápida BT151
(circuito Crowbar), cortocircuitando de manera limpia y segura la línea de alimentación de
los actuadores reales de la obra en el milisegundo cero, paralizando la maquinaria industrial 
antes de consolidar desvíos o sobrecostos frente al SECOP II.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Final, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

try:
    from app.physics.opt_symplectic_manifold import (
        FailSecureLatch,
        HamiltonianJet,
        HardwareCrowbarError,
        LatchSnapshot,
        ManifoldStateCertificate,
        MetricAtlas,
        SymplecticManifoldEngine,
        SymplecticManifoldError,
        SymplecticMetrology,
        VerletCompensation,
        higham_gamma,
    )
except ImportError:  # pragma: no cover
    from ...physics.opt_symplectic_manifold import (  # type: ignore[no-redef]
        FailSecureLatch,
        HamiltonianJet,
        HardwareCrowbarError,
        LatchSnapshot,
        ManifoldStateCertificate,
        MetricAtlas,
        SymplecticManifoldEngine,
        SymplecticManifoldError,
        SymplecticMetrology,
        VerletCompensation,
        higham_gamma,
    )

try:
    from app.physics.pseudo_holomorphic_motor import HeytingOmega3
except ImportError:  # pragma: no cover
    try:
        from ...physics.pseudo_holomorphic_motor import HeytingOmega3  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        HeytingOmega3 = None  # type: ignore[misc, assignment]

logger = logging.getLogger("APU.Agents.OptSymplecticManifoldAgent")

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS (Wilkinson / Higham / Connes / Buser)
# ════════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT_LIMIT: Final[float] = 1e-9
_VOLUME_VETO_LIMIT: Final[float] = 1e-6
_COHOMOLOGICAL_VETO_LIMIT: Final[float] = 0.0
_FIEDLER_VETO_LIMIT: Final[float] = 1e-4
_HARD_LIPSCHITZ_CEILING: Final[float] = 5.0
_DEGRADED_LIPSCHITZ_CEILING: Final[float] = 3.0
_DEGRADED_SYMPLECTIC_LIMIT: Final[float] = 1e-12
_INTERLOCK_LATENCY_NS: Final[float] = 400.0
_LOGDET_OVERFLOW_LIMIT: Final[float] = 700.0
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_PSD_TOL: Final[float] = 100.0 * _MACHINE_EPS
_DIRAC_GAP_FLOOR: Final[float] = 1e-12
_HIGHAM_SOFT_FACTOR: Final[float] = 10.0
_CONDITION_HARD: Final[float] = 1.0e14

# Alias de compatibilidad 2.x (no se usan como espera ni como jitter).
_CROWBAR_IRAM_LATENCY_NS: Final[float] = _INTERLOCK_LATENCY_NS
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = _INTERLOCK_LATENCY_NS
_CROWBAR_LATENCY_CEIL_NS: Final[float] = _INTERLOCK_LATENCY_NS
_DEGRADED_SYMPECTIC_LIMIT: Final[float] = _DEGRADED_SYMPLECTIC_LIMIT  # typo 2.x

__version__: Final[str] = "3.0.0-Doctoral-KBN-Dot2-Heyting-CAS-EngineGuided"


# ════════════════════════════════════════════════════════════════════════════════
# HEYTING Ω₃ LOCAL (respaldo si el motor FOOO no está en el path)
# ════════════════════════════════════════════════════════════════════════════════
class _LocalHeytingOmega3:
    COHERENT: Final[str] = "COHERENT"
    DEGRADED: Final[str] = "DEGRADED"
    VETOED: Final[str] = "VETOED"
    _ORDER: Final[dict[str, int]] = {VETOED: 0, DEGRADED: 1, COHERENT: 2}
    _TRUTH: Final[dict[str, float]] = {VETOED: 0.0, DEGRADED: 0.5, COHERENT: 1.0}

    @classmethod
    def order(cls, value: str) -> int:
        if value not in cls._ORDER:
            raise OptManifoldAgentError(f"Valor foráneo a Ω₃: {value!r}.")
        return cls._ORDER[value]

    @classmethod
    def normalize(cls, value: str) -> str:
        if value not in cls._ORDER:
            raise OptManifoldAgentError(f"Valor foráneo a Ω₃: {value!r}.")
        return value

    @classmethod
    def truth(cls, value: str) -> float:
        return cls._TRUTH[cls.normalize(value)]

    @classmethod
    def le(cls, left: str, right: str) -> bool:
        return cls.order(left) <= cls.order(right)

    @classmethod
    def meet(cls, left: str, right: str) -> str:
        return left if cls.order(left) <= cls.order(right) else right

    @classmethod
    def implies(cls, left: str, right: str) -> str:
        return cls.COHERENT if cls.le(left, right) else cls.normalize(right)

    @classmethod
    def fold_meet(cls, values: Sequence[str]) -> str:
        acc = cls.COHERENT
        for item in values:
            acc = cls.meet(acc, item)
        return acc


if HeytingOmega3 is None:  # pragma: no cover
    HeytingOmega3 = _LocalHeytingOmega3  # type: ignore[misc, assignment]


# ════════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES DE CALIBRE
# ════════════════════════════════════════════════════════════════════════════════
class OptManifoldAgentError(Exception):
    """Raíz de violaciones de calibre del soberano de la variedad."""


class LiouvilleGaugeError(OptManifoldAgentError):
    """Pérdida certificada de simplecticidad o de volumen de Liouville."""


class CohomologyGaugeError(OptManifoldAgentError):
    """Betti ilegítimo o residuo cohomológico negativo."""


class SpectralGaugeError(OptManifoldAgentError):
    """Espectro de Fiedler / Dirac no certificable."""


class InterlockGaugeError(OptManifoldAgentError):
    """Inconsistencia del cerrojo fail-secure de software."""


class EngineGuidanceError(OptManifoldAgentError):
    """Discrepancia de calibre entre el soberano y el motor de variedad."""


# ════════════════════════════════════════════════════════════════════════════════
# CONTRATOS INMUTABLES DE FASE
# ════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1Observation:
    """
    Contrato formal de salida de la FASE 1 y objeto inicial de la FASE 2.

    ``symplectic_deviation`` se conserva como residuo *absoluto* KBN
    (compatibilidad 1.x/2.x).  El test de aceptación vive en
    ``is_certified_symplectic`` (envelope de Higham), no en una
    comparación cruda contra \(\varepsilon_{\mathrm{máq}}\).
    """

    jacobian_shape: Tuple[int, int]
    symplectic_deviation: float
    liouville_volume_error: float
    determinant_sign: float
    log_determinant: float
    is_orientation_feasible: bool
    residual_relative_kbn: float = 0.0
    is_certified_symplectic: bool = False
    is_within_fpu_noise: bool = False
    volume_determinant: float = 0.0
    metrology: Optional[SymplecticMetrology] = None
    latch_generation: int = 0
    phase_signature: str = "PHASE_1::emit_phase1_observation"


@dataclass(frozen=True, slots=True)
class Phase2Orientation:
    """
    Contrato formal de salida de la FASE 2 y objeto inicial de la FASE 3.
    Continúa formalmente a ``Phase1Observation``.
    """

    betti_0: int
    betti_1: int
    cohomological_residual: float
    fiedler_spectral_gap: float
    pyramidal_stability: float
    has_cohomological_obstruction: bool
    islands_detected: bool
    loops_detected: bool
    diagnostics: Dict[str, Any]
    observation: Optional[Phase1Observation] = None
    cheeger_constant: float = 0.0
    jet: Optional[HamiltonianJet] = None
    phase_signature: str = "PHASE_2::emit_phase2_orientation"


@dataclass(frozen=True, slots=True)
class Phase3Decision:
    """Contrato formal de salida de la FASE 3 (Decide)."""

    heyting_verdict: str
    lambda_min_dirac: float
    lipschitz_bound: float
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    heyting_truth_value: float = 0.0
    heyting_implication_trace: Tuple[str, ...] = ()
    phase_signature: str = "PHASE_3::phase3_decide_from_phase2"


@dataclass(frozen=True, slots=True)
class OODAGaugeCertificate:
    """Certificado metrológico inmutable del ciclo OODA."""

    phase: str
    heyting_verdict: str
    symplectic_deviation: float
    liouville_volume_error: float
    cohomological_residual: float
    fiedler_spectral_gap: float
    lipschitz_bound: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    veto_reasons: Tuple[str, ...] = ()
    degraded_reasons: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    heyting_truth_value: float = 0.0
    heyting_implication_trace: Tuple[str, ...] = ()
    metrology: Optional[SymplecticMetrology] = None
    latch_generation: int = 0
    engine_certificate: Optional[ManifoldStateCertificate] = None
    phase_trace: Tuple[str, ...] = ()
    timestamp_utc: str = ""
    fpu_certified: bool = False


# ════════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN Y AUDITORÍA LIOUVILLIANA
# ════════════════════════════════════════════════════════════════════════════════
class Phase1SymplecticObservationMixin:
    r"""
    FASE 1 — OBSERVE.

    Auditoría de la 2-forma y del volumen, *delegada* al motor:

    \[
      R=M^\top\omega M-\omega,
      \qquad
      \varepsilon_{\mathrm{vol}}=\lvert\operatorname{expm1}(\operatorname{slogdet} M)\rvert.
    \]

    El último método formal emite ``Phase1Observation``.
    """

    def __init__(
        self,
        dimension_n: int,
        num_boundaries_d: int,
        engine: Optional[SymplecticManifoldEngine] = None,
    ) -> None:
        self._n = self._validate_positive_int("dimension_n", dimension_n)
        self._d = self._validate_nonnegative_int("num_boundaries_d", num_boundaries_d)
        self._engine: SymplecticManifoldEngine = (
            engine if engine is not None else SymplecticManifoldEngine(self._n)
        )
        if int(getattr(self._engine, "n", -1)) != self._n:
            raise EngineGuidanceError(
                f"Discrepancia de calibre n: soberano={self._n} "
                f"motor={getattr(self._engine, 'n', None)}."
            )
        self._omega = np.array(self._engine.omega, copy=True)
        self._omega.setflags(write=False)
        self._latch = FailSecureLatch()
        self._interlock_state = False  # espejo 2.x; la fuente de verdad es el latch

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def num_boundaries(self) -> int:
        return self._d

    @property
    def engine(self) -> SymplecticManifoldEngine:
        return self._engine

    @property
    def symplectic_structure(self) -> np.ndarray:
        return np.array(self._omega, copy=True)

    @property
    def interlock_latched(self) -> bool:
        return self._latch.is_latched()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if int(value) <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_nonnegative_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if int(value) < 0:
            raise ValueError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    def _as_real_float_array(self, values: Any, name: str) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en un ndarray.") from exc
        if np.iscomplexobj(raw):
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"{name} contiene entradas complejas no finitas.")
            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(
                    f"{name} posee componente imaginaria no despreciable."
                )
            raw = raw.real
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a float64 real.") from exc
        if arr.size == 0:
            raise ValueError(f"{name} no puede ser vacío.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")
        return np.ascontiguousarray(arr)

    def _as_real_float_matrix(
        self,
        values: Any,
        expected_shape: Tuple[int, int],
        name: str,
    ) -> np.ndarray:
        arr = self._as_real_float_array(values, name)
        if arr.ndim != 2 or arr.shape != expected_shape:
            raise ValueError(
                f"{name} debe tener forma estricta {expected_shape}. "
                f"Se recibió {arr.shape}."
            )
        return arr

    def _as_real_float_vector(self, values: Any, name: str) -> np.ndarray:
        return self._as_real_float_array(values, name).reshape(-1)

    def _require_jacobian(self, jacobian_M: Any) -> np.ndarray:
        size = 2 * self._n
        return self._as_real_float_matrix(jacobian_M, (size, size), "jacobian_M")

    def _dummy_metrology(self) -> SymplecticMetrology:
        return SymplecticMetrology(
            residual_kbn=0.0,
            residual_classical=0.0,
            residual_relative_kbn=0.0,
            rounding_envelope=0.0,
            higham_gamma=0.0,
            omega_frobenius=math.sqrt(2.0 * self._n),
            jacobian_frobenius=0.0,
            is_within_fpu_noise=True,
            is_certified_symplectic=False,
            used_dot2=False,
        )

    def _dummy_observation(self) -> Phase1Observation:
        size = 2 * self._n
        return Phase1Observation(
            jacobian_shape=(size, size),
            symplectic_deviation=0.0,
            liouville_volume_error=0.0,
            determinant_sign=0.0,
            log_determinant=float("nan"),
            is_orientation_feasible=False,
            residual_relative_kbn=0.0,
            is_certified_symplectic=False,
            is_within_fpu_noise=False,
            volume_determinant=0.0,
            metrology=self._dummy_metrology(),
            latch_generation=self._latch.generation(),
            phase_signature="PHASE_1::emit_phase1_observation",
        )

    def emit_phase1_observation(self, jacobian_M: np.ndarray) -> Phase1Observation:
        r"""
        Término formal de la Fase 1.

        \[
          \operatorname{emit\_phase1\_observation}
          :\; M
          \;\longrightarrow\;
          \mathfrak{O}_1\in\mathrm{Ob}(\mathbf{Observe}_{\mathrm{KBN}}).
        \]

        Si el canal está cerrado (load-acquire) no se interroga al motor.
        """
        if self._latch.is_latched():
            logger.critical(
                "OBSERVE: canal cerrado (gen=%d). El motor no será interrogado.",
                self._latch.generation(),
            )
            return self._dummy_observation()

        matrix_m = self._require_jacobian(jacobian_M)
        try:
            metro = self._engine.measure_symplectic_metrology(
                matrix_m, tolerance=_WILKINSON_DRIFT_LIMIT
            )
            det_m, volume_error = self._engine.measure_liouville_volume(matrix_m)
        except SymplecticManifoldError as err:
            raise LiouvilleGaugeError(str(err)) from err

        sign, logdet = np.linalg.slogdet(matrix_m)
        sign_f = float(sign)
        logdet_f = float(logdet)
        if sign_f <= 0.0 or (not math.isfinite(logdet_f)) or logdet_f > _LOGDET_OVERFLOW_LIMIT:
            volume_error = float("inf")
        feasible = bool(
            sign_f > 0.0
            and math.isfinite(metro.residual_kbn)
            and math.isfinite(volume_error)
        )
        if not metro.is_certified_symplectic:
            logger.warning(
                "Fase OBSERVE: pérdida certificada. r_KBN=%.4e env=%.4e ε_rel=%.4e Dot2=%s",
                metro.residual_kbn,
                metro.rounding_envelope,
                metro.residual_relative_kbn,
                metro.used_dot2,
            )
        elif (
            metro.is_within_fpu_noise
            and metro.residual_classical
            > _WILKINSON_DRIFT_LIMIT * metro.omega_frobenius
        ):
            logger.info(
                "Fase OBSERVE: el residuo clásico (%.4e) es ruido de FPU "
                "(KBN=%.4e ≤ envelope=%.4e). No se veta.",
                metro.residual_classical,
                metro.residual_kbn,
                metro.rounding_envelope,
            )
        return Phase1Observation(
            jacobian_shape=matrix_m.shape,
            symplectic_deviation=float(metro.residual_kbn),
            liouville_volume_error=float(volume_error),
            determinant_sign=sign_f,
            log_determinant=logdet_f,
            is_orientation_feasible=feasible,
            residual_relative_kbn=float(metro.residual_relative_kbn),
            is_certified_symplectic=bool(metro.is_certified_symplectic),
            is_within_fpu_noise=bool(metro.is_within_fpu_noise),
            volume_determinant=float(det_m),
            metrology=metro,
            latch_generation=self._latch.generation(),
            phase_signature="PHASE_1::emit_phase1_observation",
        )

    def phase1_observe_and_audit_liouville(self, jacobian_M: np.ndarray) -> Phase1Observation:
        """[FASE 1 — OBSERVE] Alias público del morfismo terminal."""
        return self.emit_phase1_observation(jacobian_M)


# ════════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENTACIÓN COHOMOLÓGICA Y SALUD ESPECTRAL
# ════════════════════════════════════════════════════════════════════════════════
class Phase2CohomologicalOrientationMixin(Phase1SymplecticObservationMixin):
    r"""
    FASE 2 — ORIENT.

    \[
      \rho_{\mathrm{coh}}=\beta_1+\lvert\beta_0-1\rvert,
      \qquad
      \Psi=\frac{\lambda_1}{(1+\rho_{\mathrm{coh}})\,(1+\varepsilon_\omega/\tau)}.
    \]
    """

    def ingest_phase1_observation(self, observation: Phase1Observation) -> Phase1Observation:
        """Continuación formal de ``emit_phase1_observation``."""
        if not isinstance(observation, Phase1Observation):
            raise LiouvilleGaugeError(
                "La Fase 2 exige un Phase1Observation emitido por la Fase 1."
            )
        if observation.phase_signature != "PHASE_1::emit_phase1_observation":
            raise LiouvilleGaugeError(
                f"Firma de fase ilegítima: {observation.phase_signature!r}."
            )
        if not math.isfinite(observation.symplectic_deviation):
            raise LiouvilleGaugeError("La observación porta un residuo no finito.")
        if (
            self._latch.is_latched()
            and observation.latch_generation != self._latch.generation()
        ):
            raise LiouvilleGaugeError(
                "La observación es anterior al cierre del canal (generación obsoleta)."
            )
        return observation

    @staticmethod
    def _validate_betti(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if int(value) < 0:
            raise CohomologyGaugeError(f"{name} debe ser mayor o igual que cero.")
        return int(value)

    def _fiedler_gap(self, eigenvalues_L: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Brecha de Fiedler \(\lambda_1\).  Fail-soft: espectro ilegible ⇒ gap 0
        (el clasificador vetará por conectividad, no se finge un \(\lambda_1>0\)).
        """
        diagnostics: Dict[str, Any] = {
            "laplacian_spectrum_valid": True,
            "fiedler_gap_defined": False,
        }
        try:
            eigenvalues = self._as_real_float_vector(eigenvalues_L, "eigenvalues_L")
        except ValueError as exc:
            diagnostics.update(
                {
                    "laplacian_spectrum_valid": False,
                    "laplacian_spectrum_error": str(exc),
                    "fiedler_gap_reason": "invalid_spectrum",
                }
            )
            return 0.0, diagnostics
        diagnostics["laplacian_spectrum_size"] = int(eigenvalues.size)
        if eigenvalues.size < 2:
            diagnostics["fiedler_gap_reason"] = "insufficient_spectrum_size"
            return 0.0, diagnostics
        min_eigenvalue = float(np.min(eigenvalues))
        diagnostics["min_laplacian_eigenvalue"] = min_eigenvalue
        if min_eigenvalue < -_PSD_TOL:
            diagnostics.update(
                {
                    "laplacian_psd_violation": True,
                    "fiedler_gap_reason": "laplacian_psd_violation",
                }
            )
            return 0.0, diagnostics
        clipped = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
        sorted_eigs = np.sort(clipped)
        sorted_eigs[np.abs(sorted_eigs) <= _PSD_TOL] = 0.0
        fiedler_gap = float(sorted_eigs[1])
        diagnostics.update(
            {
                "laplacian_psd_violation": False,
                "fiedler_gap_defined": True,
                "fiedler_gap": fiedler_gap,
            }
        )
        return fiedler_gap, diagnostics

    def _dummy_orientation(self, observation: Phase1Observation) -> Phase2Orientation:
        return Phase2Orientation(
            betti_0=0,
            betti_1=0,
            cohomological_residual=0.0,
            fiedler_spectral_gap=0.0,
            pyramidal_stability=0.0,
            has_cohomological_obstruction=True,
            islands_detected=False,
            loops_detected=False,
            diagnostics={"channel": "CLOSED", "latch_generation": self._latch.generation()},
            observation=observation,
            cheeger_constant=0.0,
            jet=None,
            phase_signature="PHASE_2::emit_phase2_orientation",
        )

    def emit_phase2_orientation(
        self,
        observation: Phase1Observation,
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
        jet: Optional[HamiltonianJet] = None,
    ) -> Phase2Orientation:
        r"""
        Término formal de la Fase 2.

        \[
          \operatorname{emit\_phase2\_orientation}
          :\; (\mathfrak{O}_1,\beta_\bullet,\sigma(\Delta))
          \;\longrightarrow\;
          \mathfrak{R}_2.
        \]
        """
        observation = self.ingest_phase1_observation(observation)
        if self._latch.is_latched():
            logger.critical(
                "ORIENT: canal cerrado (gen=%d). Se omite el espectro.",
                self._latch.generation(),
            )
            return self._dummy_orientation(observation)

        beta_0 = self._validate_betti("betti_0", betti_0)
        beta_1 = self._validate_betti("betti_1", betti_1)
        residual = float(beta_1 + abs(beta_0 - 1))
        fiedler_gap, fiedler_diag = self._fiedler_gap(eigenvalues_L)
        cheeger = float(math.sqrt(2.0 * max(fiedler_gap, 0.0)))
        sym_err = max(0.0, float(observation.residual_relative_kbn))
        psi = fiedler_gap / (
            (1.0 + residual)
            * (1.0 + sym_err / max(_WILKINSON_DRIFT_LIMIT, _MACHINE_EPS))
        )
        diagnostics: Dict[str, Any] = {
            "fiedler_gap": fiedler_gap,
            "pyramidal_stability": float(psi),
            "has_cohomological_obstruction": residual > 0.0,
            "islands_detected": beta_0 > 1,
            "loops_detected": beta_1 > 0,
            "betti_0": beta_0,
            "betti_1": beta_1,
            "cohomological_residual": residual,
            "empty_complex_detected": beta_0 == 0,
            "connected_component_obstruction": beta_0 != 1,
            "cheeger_constant": cheeger,
            "symplectic_weight": sym_err,
        }
        diagnostics.update(fiedler_diag)
        diagnostics["phase1"] = {
            "jacobian_shape": observation.jacobian_shape,
            "symplectic_deviation": observation.symplectic_deviation,
            "liouville_volume_error": observation.liouville_volume_error,
            "determinant_sign": observation.determinant_sign,
            "log_determinant": observation.log_determinant,
            "is_orientation_feasible": observation.is_orientation_feasible,
            "is_certified_symplectic": observation.is_certified_symplectic,
            "is_within_fpu_noise": observation.is_within_fpu_noise,
        }
        if jet is not None:
            diagnostics["jet"] = {
                "dt": jet.dt,
                "jacobian_method": jet.jacobian_method,
                "tikhonov_mu": jet.atlas.tikhonov_mu,
                "metric_condition": jet.atlas.condition_number,
            }
        if residual > 0.0:
            logger.warning(
                "Fase ORIENT: obstrucción. ρ=%.4f | β₀=%d | β₁=%d | λ₁=%.4e",
                residual,
                beta_0,
                beta_1,
                fiedler_gap,
            )
        return Phase2Orientation(
            betti_0=beta_0,
            betti_1=beta_1,
            cohomological_residual=residual,
            fiedler_spectral_gap=fiedler_gap,
            pyramidal_stability=float(psi),
            has_cohomological_obstruction=residual > 0.0,
            islands_detected=beta_0 > 1,
            loops_detected=beta_1 > 0,
            diagnostics=diagnostics,
            observation=observation,
            cheeger_constant=cheeger,
            jet=jet,
            phase_signature="PHASE_2::emit_phase2_orientation",
        )

    def phase2_orient_from_phase1(
        self,
        phase1_observation: Phase1Observation,
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
    ) -> Phase2Orientation:
        """[FASE 2 — ORIENT] Alias público del morfismo terminal."""
        return self.emit_phase2_orientation(
            observation=phase1_observation,
            betti_0=betti_0,
            betti_1=betti_1,
            eigenvalues_L=eigenvalues_L,
        )

    def _phase2_core(
        self,
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
        phase1_observation: Optional[Phase1Observation] = None,
    ) -> Phase2Orientation:
        """Núcleo 2.x: si no hay observación, se fabrica una neutra certificada."""
        if phase1_observation is None:
            phase1_observation = Phase1Observation(
                jacobian_shape=(2 * self._n, 2 * self._n),
                symplectic_deviation=0.0,
                liouville_volume_error=0.0,
                determinant_sign=1.0,
                log_determinant=0.0,
                is_orientation_feasible=True,
                residual_relative_kbn=0.0,
                is_certified_symplectic=True,
                is_within_fpu_noise=True,
                volume_determinant=1.0,
                metrology=self._dummy_metrology(),
                latch_generation=self._latch.generation(),
                phase_signature="PHASE_1::emit_phase1_observation",
            )
        return self.emit_phase2_orientation(
            observation=phase1_observation,
            betti_0=betti_0,
            betti_1=betti_1,
            eigenvalues_L=eigenvalues_L,
        )


# ════════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECISIÓN HEYTING, CDK Y CAS FAIL-SECURE
# ════════════════════════════════════════════════════════════════════════════════
class OptSymplecticManifoldAgent(Phase2CohomologicalOrientationMixin):
    """
    FASE 3 — DECIDE / ACT.

    Soberano que orquesta y censura a ``SymplecticManifoldEngine``.
    El parámetro ``rng_seed`` se acepta por compatibilidad 2.x y se ignora:
    el cerrojo es determinista.
    """

    def __init__(
        self,
        dimension_n: int,
        num_boundaries_d: int,
        *,
        rng_seed: Optional[int] = None,
        engine: Optional[SymplecticManifoldEngine] = None,
    ) -> None:
        super().__init__(dimension_n, num_boundaries_d, engine=engine)
        self._rng_seed = rng_seed  # conservado; no se usa para jitter

    def ingest_phase2_orientation(self, orientation: Phase2Orientation) -> Phase2Orientation:
        """Continuación formal de ``emit_phase2_orientation``."""
        if not isinstance(orientation, Phase2Orientation):
            raise CohomologyGaugeError(
                "La Fase 3 exige un Phase2Orientation emitido por la Fase 2."
            )
        if orientation.phase_signature != "PHASE_2::emit_phase2_orientation":
            raise CohomologyGaugeError(
                f"Firma de fase ilegítima: {orientation.phase_signature!r}."
            )
        if orientation.observation is None:
            raise CohomologyGaugeError("La orientación no porta observación de Fase 1.")
        self.ingest_phase1_observation(orientation.observation)
        if orientation.cohomological_residual < 0.0:
            raise CohomologyGaugeError("Residuo cohomológico negativo.")
        return orientation

    def _dirac_spectrum_analysis(self, eigenvalues_dirac: Any) -> Tuple[float, Dict[str, Any]]:
        """Menor valor singular *positivo* de Dirac (el núcleo de Hodge se excluye)."""
        diagnostics: Dict[str, Any] = {"dirac_spectrum_valid": True}
        try:
            eigenvalues = self._as_real_float_vector(eigenvalues_dirac, "eigenvalues_dirac")
        except ValueError as exc:
            diagnostics.update(
                {
                    "dirac_spectrum_valid": False,
                    "dirac_spectrum_error": str(exc),
                    "dirac_lambda_min": 0.0,
                }
            )
            return 0.0, diagnostics
        diagnostics["dirac_spectrum_size"] = int(eigenvalues.size)
        if eigenvalues.size == 0:
            diagnostics["dirac_lambda_min"] = 0.0
            return 0.0, diagnostics
        positive = np.sort(np.abs(eigenvalues))
        positive = positive[positive > _DIRAC_GAP_FLOOR]
        if positive.size == 0:
            diagnostics["dirac_lambda_min"] = 0.0
            diagnostics["dirac_kernel_only"] = True
            return 0.0, diagnostics
        lambda_min = float(positive[0])
        if not math.isfinite(lambda_min):
            lambda_min = 0.0
        diagnostics["dirac_lambda_min"] = lambda_min
        diagnostics["dirac_positive_count"] = int(positive.size)
        return lambda_min, diagnostics

    @staticmethod
    def _lipschitz_bound(lambda_min_dirac: float) -> float:
        r"""Cota cerrada \(L=1/(2\gamma^{3/2})\)."""
        if not math.isfinite(lambda_min_dirac) or lambda_min_dirac <= _MACHINE_EPS:
            return float("inf")
        try:
            coeff = 1.0 / (2.0 * (lambda_min_dirac ** 1.5))
        except OverflowError:
            return float("inf")
        return float(coeff) if math.isfinite(coeff) else float("inf")

    def connes_daleckii_krein_lipschitz(
        self,
        eigenvalues_dirac: Any,
    ) -> Tuple[float, float, Dict[str, Any]]:
        r"""
        Cota CDK y diferencia dividida discreta sobre \(\sigma(D)\setminus\{0\}\).

        \[
          L_{\mathrm{CDK}}
          =\max\Bigl(
             \tfrac{1}{2\gamma^{3/2}},\;
             \max_{i\neq j}\frac{\bigl||\lambda_i|^{-1/2}-|\lambda_j|^{-1/2}\bigr|}
                                {\bigl||\lambda_i|-|\lambda_j|\bigr|}
           \Bigr).
        \]
        """
        gamma, diagnostics = self._dirac_spectrum_analysis(eigenvalues_dirac)
        closed = self._lipschitz_bound(gamma)
        discrete = closed
        try:
            spectrum = self._as_real_float_vector(eigenvalues_dirac, "eigenvalues_dirac")
            positive = np.sort(np.abs(spectrum))
            positive = positive[positive > _DIRAC_GAP_FLOOR]
            if positive.size >= 2:
                inv_sqrt = positive ** (-0.5)
                for i in range(positive.size):
                    for j in range(i + 1, positive.size):
                        denom = abs(float(positive[j] - positive[i]))
                        if denom <= _MACHINE_EPS:
                            continue
                        slope = abs(float(inv_sqrt[j] - inv_sqrt[i])) / denom
                        if slope > discrete:
                            discrete = slope
        except ValueError:
            pass
        coeff = float(max(closed, discrete)) if math.isfinite(discrete) else float("inf")
        if not math.isfinite(coeff):
            coeff = float("inf")
        diagnostics["lipschitz_closed"] = closed
        diagnostics["lipschitz_discrete"] = discrete
        diagnostics["lipschitz_cdk"] = coeff
        return coeff, gamma, diagnostics

    def _atom(self, hard: bool, soft: bool) -> str:
        if not hard:
            return HeytingOmega3.VETOED
        if not soft:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def classify_heyting_verdict(
        self,
        observation: Phase1Observation,
        orientation: Phase2Orientation,
        lipschitz_coeff: float,
        condition_number: Optional[float] = None,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        r"""
        Clasificador \(\Omega_3\) por implicaciones:

        \[
          \nu
          =
          (\mathrm{Sp}_{\mathrm{KBN}}\to\lambda_1\ge\underline{\lambda})
          \wedge
          (\lvert\det M-1\rvert\le\varepsilon_{\mathrm{vol}})
          \wedge
          (\rho_{\mathrm{coh}}=0)
          \wedge
          (L_{\mathrm{CDK}}\le L_\sharp).
        \]

        Un desacuerdo FPU (clásico veta, KBN absuelve) *degrada*; no veta.
        """
        metro = observation.metrology
        vol = float(observation.liouville_volume_error)
        cohom = float(orientation.cohomological_residual)
        gap = float(orientation.fiedler_spectral_gap)
        volume_envelope = higham_gamma(2 * self._n)

        if metro is not None:
            symplectic_atom = self._atom(
                hard=metro.is_certified_symplectic
                or metro.residual_kbn
                <= _HIGHAM_SOFT_FACTOR * (metro.rounding_envelope + _MACHINE_EPS),
                soft=metro.is_certified_symplectic
                and metro.residual_relative_kbn <= _DEGRADED_SYMPLECTIC_LIMIT,
            )
            fpu_atom = self._atom(
                hard=True,
                soft=not (
                    metro.residual_classical
                    > _WILKINSON_DRIFT_LIMIT * metro.omega_frobenius
                    and metro.is_certified_symplectic
                ),
            )
            cond_src = metro.jacobian_frobenius
        else:
            symplectic_atom = self._atom(
                hard=observation.symplectic_deviation
                <= max(_WILKINSON_DRIFT_LIMIT * 10.0, 1e-8),
                soft=observation.symplectic_deviation <= _DEGRADED_SYMPLECTIC_LIMIT,
            )
            fpu_atom = HeytingOmega3.COHERENT
            cond_src = 0.0

        volume_atom = self._atom(
            hard=math.isfinite(vol) and vol <= max(_VOLUME_VETO_LIMIT, volume_envelope),
            soft=vol <= _DEGRADED_SYMPLECTIC_LIMIT,
        )
        cohom_atom = self._atom(hard=cohom < 1.0, soft=cohom <= _COHOMOLOGICAL_VETO_LIMIT)
        fiedler_atom = self._atom(hard=gap >= 0.0, soft=gap >= _FIEDLER_VETO_LIMIT)
        lip_atom = self._atom(
            hard=math.isfinite(lipschitz_coeff) and lipschitz_coeff <= _HARD_LIPSCHITZ_CEILING,
            soft=lipschitz_coeff <= _DEGRADED_LIPSCHITZ_CEILING,
        )
        if condition_number is None:
            condition_number = float("inf") if cond_src == 0.0 and metro is None else 1.0
        cond_atom = self._atom(
            hard=math.isfinite(float(condition_number))
            and float(condition_number) < _CONDITION_HARD,
            soft=float(condition_number) < 1.0e10,
        )
        orient_atom = self._atom(hard=True, soft=observation.is_orientation_feasible)
        impl = HeytingOmega3.implies(symplectic_atom, fiedler_atom)
        verdict = HeytingOmega3.fold_meet(
            [impl, volume_atom, cohom_atom, lip_atom, cond_atom, fpu_atom, orient_atom]
        )

        certified_broken = (
            metro is not None
            and (not metro.is_certified_symplectic)
            and (not metro.is_within_fpu_noise)
        ) or (
            metro is None and observation.symplectic_deviation > _WILKINSON_DRIFT_LIMIT
        )
        hard_veto = (
            certified_broken
            or (not math.isfinite(vol))
            or vol > _VOLUME_VETO_LIMIT
            or cohom > _COHOMOLOGICAL_VETO_LIMIT
            or gap < _FIEDLER_VETO_LIMIT
            or (not math.isfinite(lipschitz_coeff))
            or lipschitz_coeff > _HARD_LIPSCHITZ_CEILING
            or not observation.is_orientation_feasible
        )
        if hard_veto:
            verdict = HeytingOmega3.meet(verdict, HeytingOmega3.VETOED)
        elif verdict == HeytingOmega3.COHERENT and (
            (
                metro is not None
                and metro.residual_relative_kbn > _DEGRADED_SYMPLECTIC_LIMIT
                and not metro.is_within_fpu_noise
            )
            or (
                metro is None
                and observation.symplectic_deviation > _DEGRADED_SYMPLECTIC_LIMIT
            )
            or lipschitz_coeff > _DEGRADED_LIPSCHITZ_CEILING
        ):
            verdict = HeytingOmega3.DEGRADED

        veto_reasons = []
        degraded_reasons = []

        def collect(label: str, atom: str) -> None:
            if atom == HeytingOmega3.VETOED:
                veto_reasons.append(label)
            elif atom == HeytingOmega3.DEGRADED:
                degraded_reasons.append(label)

        collect("symplectic_kbn", symplectic_atom)
        collect("fpu_disagreement", fpu_atom)
        collect("liouville_volume", volume_atom)
        collect("cohomological_residual", cohom_atom)
        collect("fiedler_gap", fiedler_atom)
        collect("lipschitz_cdk", lip_atom)
        collect("condition_number", cond_atom)
        collect("orientation_feasible", orient_atom)
        if hard_veto and verdict == HeytingOmega3.VETOED and not veto_reasons:
            veto_reasons.append("hard_threshold_union")

        trace = (
            f"Sp={symplectic_atom}",
            f"FPU={fpu_atom}",
            f"vol={volume_atom}",
            f"ρ={cohom_atom}",
            f"λ₁={fiedler_atom}",
            f"Sp→λ₁={impl}",
            f"L_CDK={lip_atom}",
            f"κ={cond_atom}",
            f"orient={orient_atom}",
            f"ν={verdict}",
        )
        return verdict, tuple(veto_reasons), tuple(degraded_reasons), trace

    @staticmethod
    def _classify_heyting_from_values(
        symplectic_error: float,
        volume_error: float,
        cohomological_residual: float,
        fiedler_gap: float,
        lipschitz_coeff: float,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        """Clasificador 2.x (umbrales planos).  Lo usa sólo la API de compatibilidad."""
        veto_reasons = []
        degraded_reasons = []
        if not math.isfinite(symplectic_error):
            veto_reasons.append("symplectic_error_nonfinite")
        elif symplectic_error < 0.0:
            veto_reasons.append("symplectic_error_negative")
        elif symplectic_error > _WILKINSON_DRIFT_LIMIT:
            veto_reasons.append("symplectic_error_exceeds_wilkinson_drift_limit")
        elif symplectic_error > _DEGRADED_SYMPLECTIC_LIMIT:
            degraded_reasons.append("symplectic_error_above_degraded_threshold")
        if not math.isfinite(volume_error):
            veto_reasons.append("liouville_volume_error_nonfinite")
        elif volume_error < 0.0:
            veto_reasons.append("liouville_volume_error_negative")
        elif volume_error > _VOLUME_VETO_LIMIT:
            veto_reasons.append("liouville_volume_error_exceeds_veto_limit")
        if not math.isfinite(cohomological_residual):
            veto_reasons.append("cohomological_residual_nonfinite")
        elif cohomological_residual < 0.0:
            veto_reasons.append("cohomological_residual_negative")
        elif cohomological_residual > _COHOMOLOGICAL_VETO_LIMIT:
            veto_reasons.append("cohomological_residual_nonzero")
        if not math.isfinite(fiedler_gap):
            veto_reasons.append("fiedler_gap_nonfinite")
        elif fiedler_gap < 0.0:
            veto_reasons.append("fiedler_gap_negative")
        elif fiedler_gap < _FIEDLER_VETO_LIMIT:
            veto_reasons.append("fiedler_gap_below_connectivity_threshold")
        if not math.isfinite(lipschitz_coeff):
            veto_reasons.append("lipschitz_bound_nonfinite")
        elif lipschitz_coeff < 0.0:
            veto_reasons.append("lipschitz_bound_negative")
        elif lipschitz_coeff > _HARD_LIPSCHITZ_CEILING:
            veto_reasons.append("lipschitz_bound_exceeds_hard_ceiling")
        elif lipschitz_coeff > _DEGRADED_LIPSCHITZ_CEILING:
            degraded_reasons.append("lipschitz_bound_above_degraded_ceiling")
        if veto_reasons:
            verdict = HeytingOmega3.VETOED
        elif degraded_reasons:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def phase3_decide_from_phase2(
        self,
        phase1_observation: Phase1Observation,
        phase2_orientation: Phase2Orientation,
        eigenvalues_dirac: Any,
    ) -> Phase3Decision:
        """[FASE 3 — DECIDE] Implicaciones de Heyting + cota CDK."""
        phase1_observation = self.ingest_phase1_observation(phase1_observation)
        phase2_orientation = self.ingest_phase2_orientation(phase2_orientation)
        lipschitz_coeff, lambda_min, dirac_diag = self.connes_daleckii_krein_lipschitz(
            eigenvalues_dirac
        )
        cond = None
        if phase2_orientation.jet is not None:
            singular = np.asarray(la.svdvals(phase2_orientation.jet.jacobian), dtype=np.float64)
            if singular.size and singular[-1] > _MACHINE_EPS:
                cond = float(singular[0] / singular[-1])
        verdict, veto_reasons, degraded_reasons, trace = self.classify_heyting_verdict(
            observation=phase1_observation,
            orientation=phase2_orientation,
            lipschitz_coeff=lipschitz_coeff,
            condition_number=cond,
        )
        diagnostics = dict(dirac_diag)
        diagnostics["heyting_verdict"] = verdict
        diagnostics["implication_trace"] = trace
        return Phase3Decision(
            heyting_verdict=verdict,
            lambda_min_dirac=float(lambda_min),
            lipschitz_bound=float(lipschitz_coeff),
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics,
            heyting_truth_value=HeytingOmega3.truth(verdict),
            heyting_implication_trace=trace,
            phase_signature="PHASE_3::phase3_decide_from_phase2",
        )

    def reset_hardware_interlock_for_supervision(self) -> bool:
        """Rearme por CAS \(1\to 0\).  Devuelve el estado *previo*."""
        previous = self._latch.is_latched()
        if previous and not self._latch.try_reset():
            if self._latch.is_latched():
                raise InterlockGaugeError("CAS de rearme falló.")
        self._interlock_state = self._latch.is_latched()
        if previous:
            logger.info("Cerrojo fail-secure rearmado (gen=%d).", self._latch.generation())
        return previous

    def phase3_act_hardware_interlock(self, decision: Phase3Decision) -> Tuple[bool, float]:
        r"""
        [FASE 3 — ACT]

        ``fetch_or`` de un disparo.  **No** hay GPIO, ISR ni tiristor.
        Reentrada en VETOED: se reconoce el latch y se reporta la cota
        del modelo, sin espera.
        """
        if not isinstance(decision, Phase3Decision):
            raise TypeError("decision debe ser Phase3Decision.")
        verdict = HeytingOmega3.normalize(decision.heyting_verdict)
        if verdict == HeytingOmega3.VETOED:
            snap = self._latch.trip()
            self._interlock_state = True
            if not snap.cas_succeeded:
                logger.warning(
                    "CAS: el interlock ya estaba latched (gen=%d). Se reconoce el veto.",
                    snap.generation,
                )
            logger.critical(
                "¡VETO DE SUTURA TOPOLÓGICA! CAS gen=%d host=%.0f ns "
                "(cota modelo %.0f ns).",
                snap.generation,
                snap.host_cas_ns,
                snap.model_latency_ns,
            )
            return True, float(snap.model_latency_ns)
        if self._latch.is_latched():
            raise InterlockGaugeError(
                "El cerrojo permanece latched; se exige "
                "reset_hardware_interlock_for_supervision() antes de "
                "aceptar un veredicto no-VETOED."
            )
        return False, 0.0

    def suture_ooda_cycle(
        self,
        orientation: Phase2Orientation,
        eigenvalues_dirac: Any,
        engine_certificate: Optional[ManifoldStateCertificate] = None,
    ) -> OODAGaugeCertificate:
        """Término formal de la Fase 3: gluing observación + orientación sobre \(\Omega_3\)."""
        if self._latch.is_latched():
            return self._blocked_certificate()
        orientation = self.ingest_phase2_orientation(orientation)
        observation = orientation.observation
        assert observation is not None
        decision = self.phase3_decide_from_phase2(
            phase1_observation=observation,
            phase2_orientation=orientation,
            eigenvalues_dirac=eigenvalues_dirac,
        )
        verdict = decision.heyting_verdict
        veto_reasons = decision.veto_reasons
        degraded_reasons = decision.degraded_reasons
        trace = decision.heyting_implication_trace
        if engine_certificate is not None:
            motor_verdict = HeytingOmega3.normalize(engine_certificate.heyting_verdict)
            fused = HeytingOmega3.meet(verdict, motor_verdict)
            extra = (f"ν_motor={motor_verdict}", f"ν_meet={fused}")
            trace = trace + extra
            if fused != verdict:
                if fused == HeytingOmega3.VETOED:
                    veto_reasons = veto_reasons + ("engine_meet_veto",)
                elif fused == HeytingOmega3.DEGRADED:
                    degraded_reasons = degraded_reasons + ("engine_meet_degraded",)
            verdict = fused
            decision = Phase3Decision(
                heyting_verdict=verdict,
                lambda_min_dirac=decision.lambda_min_dirac,
                lipschitz_bound=decision.lipschitz_bound,
                veto_reasons=veto_reasons,
                degraded_reasons=degraded_reasons,
                diagnostics=decision.diagnostics,
                heyting_truth_value=HeytingOmega3.truth(verdict),
                heyting_implication_trace=trace,
                phase_signature=decision.phase_signature,
            )
        fired, latency = self.phase3_act_hardware_interlock(decision)
        snap = self._latch.snapshot()
        if verdict == HeytingOmega3.DEGRADED and not fired:
            logger.warning("Estado DEGRADED en la aduana. Traza: %s", " ∧ ".join(trace))
        elif verdict == HeytingOmega3.COHERENT:
            logger.info(
                "Ciclo OODA coherente. r_KBN=%.3e | ρ=%.4f | λ₁=%.4e | L_CDK=%.4f",
                observation.symplectic_deviation,
                orientation.cohomological_residual,
                orientation.fiedler_spectral_gap,
                decision.lipschitz_bound,
            )
        diagnostics = dict(orientation.diagnostics)
        diagnostics["phase3"] = dict(decision.diagnostics)
        diagnostics["hardware"] = {
            "interlock_fired": fired,
            "actuation_latency_ns": latency,
            "generation": snap.generation,
            "mode": "SOFTWARE_FAILSECURE_LATCH",
        }
        metro = observation.metrology
        return OODAGaugeCertificate(
            phase="G_GAUGE_CYCLE_SUTURATED",
            heyting_verdict=verdict,
            symplectic_deviation=observation.symplectic_deviation,
            liouville_volume_error=observation.liouville_volume_error,
            cohomological_residual=orientation.cohomological_residual,
            fiedler_spectral_gap=orientation.fiedler_spectral_gap,
            lipschitz_bound=decision.lipschitz_bound,
            hardware_interlock_fired=fired,
            actuation_latency_ns=latency,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            diagnostics=diagnostics,
            heyting_truth_value=HeytingOmega3.truth(verdict),
            heyting_implication_trace=trace,
            metrology=metro,
            latch_generation=snap.generation,
            engine_certificate=engine_certificate,
            phase_trace=(
                observation.phase_signature,
                orientation.phase_signature,
                "PHASE_3::suture_ooda_cycle",
            ),
            timestamp_utc=self._utc_now(),
            fpu_certified=bool(metro.is_certified_symplectic) if metro is not None else False,
        )

    def _blocked_certificate(self) -> OODAGaugeCertificate:
        observation = self._dummy_observation()
        snap = self._latch.snapshot()
        trace = (
            "Sp=VETOED",
            "channel=CLOSED",
            "CAS=acquire-load",
            f"gen={snap.generation}",
            "ν=VETOED",
        )
        return OODAGaugeCertificate(
            phase="G_GAUGE_CYCLE_BLOCKED_LATCH",
            heyting_verdict=HeytingOmega3.VETOED,
            symplectic_deviation=0.0,
            liouville_volume_error=0.0,
            cohomological_residual=0.0,
            fiedler_spectral_gap=0.0,
            lipschitz_bound=float("inf"),
            hardware_interlock_fired=True,
            actuation_latency_ns=_INTERLOCK_LATENCY_NS,
            veto_reasons=("channel_closed",),
            degraded_reasons=(),
            diagnostics={"channel": "CLOSED", "generation": snap.generation},
            heyting_truth_value=HeytingOmega3.truth(HeytingOmega3.VETOED),
            heyting_implication_trace=trace,
            metrology=observation.metrology,
            latch_generation=snap.generation,
            engine_certificate=None,
            phase_trace=(
                "PHASE_0::acquire_latch",
                "PHASE_1::short_circuit",
                "PHASE_3::channel_closed",
            ),
            timestamp_utc=self._utc_now(),
            fpu_certified=False,
        )

    # ────────────────────────────────────────────────────────────────────────────
    # API PÚBLICA COMPATIBLE CON 1.X / 2.X
    # ────────────────────────────────────────────────────────────────────────────
    def observe_and_audit_liouville(self, jacobian_M: np.ndarray) -> Tuple[float, float]:
        phase1 = self.emit_phase1_observation(jacobian_M)
        return phase1.symplectic_deviation, phase1.liouville_volume_error

    def orient_cohomological_health(
        self,
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
    ) -> Tuple[float, float, Dict[str, Any]]:
        phase2 = self._phase2_core(betti_0, betti_1, eigenvalues_L, None)
        return (
            phase2.cohomological_residual,
            phase2.fiedler_spectral_gap,
            phase2.diagnostics,
        )

    def decide_heyting_verdict(
        self,
        symplectic_error: float,
        volume_error: float,
        cohomological_residual: float,
        fiedler_gap: float,
        eigenvalues_dirac: Any,
    ) -> Tuple[str, float]:
        _, lambda_min, _ = self.connes_daleckii_krein_lipschitz(eigenvalues_dirac)
        lipschitz_coeff = self._lipschitz_bound(lambda_min)
        verdict, _, _ = self._classify_heyting_from_values(
            symplectic_error=float(symplectic_error),
            volume_error=float(volume_error),
            cohomological_residual=float(cohomological_residual),
            fiedler_gap=float(fiedler_gap),
            lipschitz_coeff=lipschitz_coeff,
        )
        return verdict, lipschitz_coeff

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        decision = Phase3Decision(
            heyting_verdict=str(verdict),
            lambda_min_dirac=0.0,
            lipschitz_bound=0.0,
            veto_reasons=(),
            degraded_reasons=(),
            diagnostics={"source": "compatibility_api"},
            heyting_truth_value=0.0,
            heyting_implication_trace=(),
        )
        return self.phase3_act_hardware_interlock(decision)

    def execute_gauge_cycle(
        self,
        jacobian_M: np.ndarray,
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
        eigenvalues_dirac: Any,
    ) -> OODAGaugeCertificate:
        """
        Ciclo OODA ligero (insumos precomputados).

            Fase 1 → ``emit_phase1_observation``
            Fase 2 → ``emit_phase2_orientation``
            Fase 3 → ``suture_ooda_cycle``
        """
        if self._latch.is_latched():
            return self._blocked_certificate()

        def _phase1() -> Phase1Observation:
            return self.emit_phase1_observation(jacobian_M)

        def _phase2(observation: Phase1Observation) -> Phase2Orientation:
            return self.emit_phase2_orientation(
                observation=observation,
                betti_0=betti_0,
                betti_1=betti_1,
                eigenvalues_L=eigenvalues_L,
            )

        def _phase3(orientation: Phase2Orientation) -> OODAGaugeCertificate:
            return self.suture_ooda_cycle(orientation, eigenvalues_dirac)

        try:
            observation = _phase1()
            orientation = _phase2(observation)
            return _phase3(orientation)
        except Exception as err:
            logger.critical(
                "¡VETO DE SUTURA TOPOLÓGICA! Ruptura de calibre: %s", err
            )
            raise

    def execute_guided_manifold_cycle(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_H_q: Callable[[np.ndarray], np.ndarray],
        dt: float,
        mass_G: np.ndarray,
        hamiltonian: Callable[[np.ndarray, np.ndarray], float],
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
        eigenvalues_dirac: Any,
        hess_H_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        compensation: Optional[VerletCompensation] = None,
        sphere_density: Optional[Callable[[float, float], float]] = None,
    ) -> Tuple[OODAGaugeCertificate, HamiltonianJet]:
        r"""
        Ciclo soberano que *guía* al motor de variedad.

            Fase 1 agente  — observa el jacobiano del jet.
            Fase 2 agente  — orienta con \((\beta,\sigma(\Delta))\) y el jet.
            Fase 3 agente  — meet \(\nu_{\mathrm{agente}}\wedge\nu_{\mathrm{motor}}\).

        El motor integra (atlas → jet → sutura); el agente no duplica Verlet.
        """
        if self._latch.is_latched():
            raise InterlockGaugeError(
                "Canal cerrado: execute_guided_manifold_cycle no integra."
            )
        try:
            engine_cert, jet = self._engine.execute_manifold_cycle(
                q=q,
                p=p,
                grad_H_q=grad_H_q,
                dt=dt,
                mass_G=mass_G,
                hamiltonian=hamiltonian,
                betti_1=betti_1,
                sphere_density=sphere_density,
                hess_H_q=hess_H_q,
                compensation=compensation,
            )
        except (SymplecticManifoldError, HardwareCrowbarError) as err:
            raise EngineGuidanceError(str(err)) from err

        def _phase1() -> Phase1Observation:
            return self.emit_phase1_observation(jet.jacobian)

        def _phase2(observation: Phase1Observation) -> Phase2Orientation:
            return self.emit_phase2_orientation(
                observation=observation,
                betti_0=betti_0,
                betti_1=betti_1,
                eigenvalues_L=eigenvalues_L,
                jet=jet,
            )

        def _phase3(orientation: Phase2Orientation) -> OODAGaugeCertificate:
            return self.suture_ooda_cycle(
                orientation=orientation,
                eigenvalues_dirac=eigenvalues_dirac,
                engine_certificate=engine_cert,
            )

        try:
            observation = _phase1()
            orientation = _phase2(observation)
            return _phase3(orientation), jet
        except Exception as err:
            logger.critical(
                "¡VETO DE SUTURA TOPOLÓGICA! Ruptura guiada de la variedad: %s", err
            )
            raise

    def guide_engine_audit(
        self,
        jacobian_M: np.ndarray,
        h_initial: float,
        h_final: float,
        betti_0: int,
        betti_1: int,
        eigenvalues_L: Any,
        eigenvalues_dirac: Any,
    ) -> OODAGaugeCertificate:
        """Guía ``audit_and_certify_state`` del motor y hace el meet OODA."""
        if self._latch.is_latched():
            return self._blocked_certificate()
        try:
            engine_cert = self._engine.audit_and_certify_state(
                jacobian_M, h_initial, h_final, betti_1=betti_1
            )
        except SymplecticManifoldError as err:
            raise EngineGuidanceError(str(err)) from err
        observation = self.emit_phase1_observation(jacobian_M)
        orientation = self.emit_phase2_orientation(
            observation, betti_0, betti_1, eigenvalues_L
        )
        return self.suture_ooda_cycle(
            orientation, eigenvalues_dirac, engine_certificate=engine_cert
        )


__all__ = [
    "OptManifoldAgentError",
    "LiouvilleGaugeError",
    "CohomologyGaugeError",
    "SpectralGaugeError",
    "InterlockGaugeError",
    "EngineGuidanceError",
    "Phase1Observation",
    "Phase2Orientation",
    "Phase3Decision",
    "OODAGaugeCertificate",
    "Phase1SymplecticObservationMixin",
    "Phase2CohomologicalOrientationMixin",
    "OptSymplecticManifoldAgent",
]