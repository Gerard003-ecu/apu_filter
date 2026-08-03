# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Discrete Hodge Star Agent (Soberano de la Métrica Constitutiva)    ║
║  Ruta   : app/agents/physics/discrete_hodge_star_agent.py                    ║
║  Versión: 2.0.0-Hodge-Weyl-Wilkinson-Helmholtz-OODA-Topos-Strict             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):           ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Agente Soberano y Observador Activo que gobierna el operador estrella de    ║
║  Hodge discreto (★_k) mediante un ciclo OODA covariante de tres fases        ║
║  anidadas. Delega el álgebra espectral, los Laplacianos ponderados y la      ║
║  descomposición de Helmholtz-Hodge al módulo certificado                     ║
║  `app.physics.discrete_hodge_star` (v2.0), y añade la capa de gobernanza:    ║
║                                                                              ║
║    FASE 1  Observe  → auditoría SPD + morfismo de dualidad + Tikhonov        ║
║    FASE 2  Orient   → Laplaciano Port-Hamiltoniano + pasividad + Betti       ║
║    FASE 3  Decide/Act → Helmholtz-Hodge + retículo Ω₃ + Crowbar GPIO         ║
║                                                                              ║
║  Continuidad formal obligatoria:                                             ║
║    Phase1HodgeObservation  → entrada de FASE 2                               ║
║    Phase2HodgeOrientation  → entrada de FASE 3                               ║
║    Phase3HodgeDecision     → entrada del Soberano (Crowbar / State)          ║
║                                                                              ║
║  Leyes preservadas:                                                          ║
║    ★_k = ★_kᵀ ≻ 0                                                            ║
║    L₀★ = ∂₁ ★₁ ∂₁ᵀ                                                           ║
║    I  = I_exact + I_coexact + I_harmonic                                     ║
║    Ḣ  ≤ 0   (pasividad Port-Hamiltoniana)                                    ║
║                                                                              ║
║  Contrato Crowbar (Bypass ESP32 / GPIO14):                                   ║
║    Si vorticidad parasitaria > umbral  ∨  β₁ > β₁_max  ∨  métrica no SPD,    ║
║    el clasificador Ω₃ colapsa a VETOED y se activa el disyuntor físico.      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum, auto
from typing import (
    Final,
    Optional,
    Tuple,
    List,
    Dict,
    Any,
    Protocol,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias del núcleo MIC (fallbacks seguros para entorno aislado)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Excepción base para violaciones de invariantes topológico-algebraicos."""

    class Morphism:
        """Clase base de morfismos en el Topos de la Malla."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Stratum(IntEnum):
        """Estratos de la pirámide de información DIKW."""
        PHYSICS = 1
        TACTICS = 2
        STRATEGY = 3
        WISDOM = 4

# ─────────────────────────────────────────────────────────────────────────────
# Motor geométrico certificado (módulo physics v2.0 — las 3 fases algebraicas)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.physics.discrete_hodge_star import (
        DiscreteHodgeStar,
        HodgeDegree,
        HodgeDualityMorphism,
        HodgeMetricState,
        SpectralCertificate,
        WeightedHodgeLaplacian,
        LaplacianSpectrum,
        DeRhamComplexCertificate,
        DiscreteDeRhamComplex,
        HelmholtzHodgeDecomposition,
        PortHamiltonianState,
        build_certified_de_rham_complex,
        _MACHINE_EPS as _PHYS_EPS,
        _SPECTRAL_TOL as _PHYS_SPECTRAL_TOL,
        _CONDITION_SAFE as _PHYS_CONDITION_SAFE,
    )
    _HAS_PHYSICS_ENGINE = True
except ImportError:  # pragma: no cover
    _HAS_PHYSICS_ENGINE = False
    _PHYS_EPS = float(np.finfo(np.float64).eps)
    _PHYS_SPECTRAL_TOL = 1e-12
    _PHYS_CONDITION_SAFE = 1e12

logger = logging.getLogger("MIC.Agents.Physics.DiscreteHodgeStarAgent")

RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]

# Constantes de Wilkinson / silicio
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1e-13
_CONDITION_NUMBER_MAX: Final[float] = 1e8
_RECONSTRUCTION_TOL: Final[float] = 1e-10
_KCL_TOL: Final[float] = 1e-8
_TIKHONOV_SCALE: Final[float] = 1e4
_CROWBAR_GPIO: Final[int] = 14  # GPIO14 — contrato hardware ESP32


# =============================================================================
# JERARQUÍA DE EXCEPCIONES (VETOS DUROS)
# =============================================================================

class HodgeStarAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de la Estrella de Hodge."""


class NonFiniteWeightError(HodgeStarAgentError):
    """Pesos con NaN/Inf (fuga de FPU)."""


class MetricDegeneracyError(HodgeStarAgentError):
    """Pérdida de definición positiva o mal condicionamiento severo.
    Nota: se corrige el typo histórico MetricDefeneracyError → MetricDegeneracyError.
    """


# Alias de compatibilidad con v3.x
MetricDefeneracyError = MetricDegeneracyError


class LaplacianPassivityError(HodgeStarAgentError):
    """Violación de pasividad Port-Hamiltoniana (Ḣ > 0)."""


class HelmholtzDecompositionError(HodgeStarAgentError):
    """Fallo de consistencia en la descomposición ortogonal de Hodge."""


class ParasiticVorticityVeto(HodgeStarAgentError):
    """Veto de aduana geométrica: ciclos lógicos superan la cota elástica."""


class BettiNumberVeto(HodgeStarAgentError):
    """Veto topológico: número de Betti β₁ excede el máximo admisible."""


class CrowbarActivationError(HodgeStarAgentError):
    """Fallo al señalizar el disyuntor físico (GPIO)."""


# =============================================================================
# CLASIFICADOR DE SUBOBJETOS Ω₃ (retículo de Heyting acotado)
# =============================================================================

class HodgeSovereignVerdict(IntEnum):
    r"""
    Veredicto en el retículo de verdad de tres valores (topos de Hodge).

    Orden de Heyting:  COHERENT ⊑ DEGRADED ⊑ VETOED.

    COHERENT (0)
        Estado plano-laminar asintóticamente estable; vorticidad ≈ 0.
    DEGRADED (1)
        Ciclos tolerables o ineficiencias amortiguadas; vigilancia activa.
    VETOED (2)
        Vorticidad desbocada, pérdida de pasividad o β₁ crítico → Crowbar.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __or__(self, other: "HodgeSovereignVerdict") -> "HodgeSovereignVerdict":
        """Join del retículo (peor veredicto gana)."""
        return self.__class__(max(int(self), int(other)))

    def __and__(self, other: "HodgeSovereignVerdict") -> "HodgeSovereignVerdict":
        """Meet del retículo."""
        return self.__class__(min(int(self), int(other)))


class CrowbarAction(Enum):
    """Acción de actuador físico tras el veredicto Ω₃."""
    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()  # GPIO14 → Crowbar SCR/MOSFET


# =============================================================================
# PROTOCOLO DE ACTUADOR HARDWARE (inyectable — testability)
# =============================================================================

@runtime_checkable
class CrowbarActuator(Protocol):
    """Puerto de salida hacia el disyuntor físico (ESP32 GPIO u homólogo)."""

    def assert_crowbar(self, gpio: int, reason: str) -> bool:
        """Activa el Crowbar. Retorna True si el hardware ACK-eó."""
        ...

    def deassert_crowbar(self, gpio: int) -> bool:
        """Libera el Crowbar (reset controlado)."""
        ...


class NullCrowbarActuator:
    """Actuador nulo (dry-run / CI). Solo registra."""

    def __init__(self) -> None:
        self.activations: List[Tuple[int, str]] = []
        self.deactivations: List[int] = []

    def assert_crowbar(self, gpio: int, reason: str) -> bool:
        self.activations.append((gpio, reason))
        logger.warning("NullCrowbar ACTIVATE gpio=%d reason=%s", gpio, reason)
        return True

    def deassert_crowbar(self, gpio: int) -> bool:
        self.deactivations.append(gpio)
        logger.info("NullCrowbar DEASSERT gpio=%d", gpio)
        return True


class LoggingCrowbarActuator:
    """Actuador que emite la señal por log estructurado (puente a firmware)."""

    def assert_crowbar(self, gpio: int, reason: str) -> bool:
        logger.critical(
            "CROWBAR_ASSERT gpio=%d reason=%s protocol=ESP32_GPIO",
            gpio, reason,
        )
        return True

    def deassert_crowbar(self, gpio: int) -> bool:
        logger.info("CROWBAR_DEASSERT gpio=%d", gpio)
        return True


# =============================================================================
# DTOs INMUTABLES — CONTRATOS DE CONTINUIDAD FUNTORIAL ENTRE FASES
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase1HodgeObservation:
    r"""
    Artefacto de la FASE 1 (Observe).

    Certificado de estabilidad espectral de ★_k tras saneamiento Wilkinson /
    Tikhonov. Es el *único* objeto legítimo de entrada a la FASE 2.

    Campos
    ------
    dimension : int
        dim C^k (número de aristas / 1-símplices).
    condition_number : float
        κ₂(★_k) = λ_max / λ_min.
    min_eigenvalue : float
        Brecha espectral basal λ_min.
    max_eigenvalue : float
        λ_max.
    is_strictly_spd : bool
        Criterio de Sylvester estricto bajo tolerancia.
    regularized_applied : bool
        True si se inyectó desplazamiento de Tikhonov.
    tikhonov_shift : float
        Magnitud del shift (0 si no hubo regularización).
    weights_sanitized : RealVector
        Pesos post-saneamiento (diagonal de ★_k).
    hodge_matrix : RealMatrix
        ★_k diagonal saneada.
    duality_morphism_payload : Optional[Any]
        HodgeDualityMorphism del motor physics (si disponible).
    metric_state_payload : Optional[Any]
        HodgeMetricState del motor physics (si disponible).
    spectral_trace : float
        tr(★_k).
    log_determinant : float
        log det(★_k).
    """
    dimension: int
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    is_strictly_spd: bool
    regularized_applied: bool
    tikhonov_shift: float
    weights_sanitized: RealVector
    hodge_matrix: RealMatrix
    duality_morphism_payload: Optional[Any] = None
    metric_state_payload: Optional[Any] = None
    spectral_trace: float = 0.0
    log_determinant: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase2HodgeOrientation:
    r"""
    Artefacto de la FASE 2 (Orient).

    Certificado de consistencia Port-Hamiltoniana y topología del 1-esqueleto.
    Es el *único* objeto legítimo de entrada a la FASE 3.

    Campos
    ------
    observation_stamp : Phase1HodgeObservation
        Certificado FASE 1 heredado (continuidad funtorial).
    dirichlet_energy : float
        E_★(φ) = ½ φᵀ L₀★ φ  (potencia de Joule).
    is_passive : bool
        Ḣ = −2 E_★ ≤ 0.
    laplacian_condition : float
        κ₂(L₀★) sobre el subespacio de rango completo (λ_max/λ₂).
    laplacian_0 : RealMatrix
        L₀★ = ∂₁ ★₁ ∂₁ᵀ.
    algebraic_connectivity : float
        λ₂ (Fiedler) — conectividad algebraica.
    betti_0 : int
        Número de componentes conexas (dim ker Δ₀).
    betti_1 : int
        Ciclos independientes (dim ker Δ₁) — vorticidad topológica.
    is_self_adjoint : bool
        Certificado de autoadjunción numérica de L₀★.
    max_eigenresidual : float
        ‖L v − λ v‖_∞ máximo.
    spectral_gap_laplacian : float
        λ_min no nulo.
    energy_scale : float
        tr(★₀) + tr(★₁) (escala energética de referencia).
    de_rham_certificate_payload : Optional[Any]
        DeRhamComplexCertificate del motor physics (si disponible).
    """
    observation_stamp: Phase1HodgeObservation
    dirichlet_energy: float
    is_passive: bool
    laplacian_condition: float
    laplacian_0: RealMatrix
    algebraic_connectivity: float
    betti_0: int
    betti_1: int
    is_self_adjoint: bool
    max_eigenresidual: float
    spectral_gap_laplacian: float
    energy_scale: float
    de_rham_certificate_payload: Optional[Any] = None


@dataclass(frozen=True, slots=True)
class Phase3HodgeDecision:
    r"""
    Artefacto de la FASE 3 (Decide).

    Reporte de descomposición ortogonal de Helmholtz-Hodge + veredicto Ω₃.
    Es el *único* objeto legítimo de entrada al actuador Crowbar del Soberano.

    Campos
    ------
    orientation_stamp : Phase2HodgeOrientation
        Certificado FASE 2 heredado.
    exact_norm : float
        ‖I_exact‖₂  (componente laminar / gradiente).
    coexact_norm : float
        ‖I_coexact‖₂  (componente solenoidal / ciclos).
    harmonic_norm : float
        ‖I_harmonic‖₂  (cohomología H¹).
    parasitic_vorticity : float
        Ω_vort = √(I_coexactᵀ ★₁⁻¹ I_coexact)  en métrica de Joule.
    harmonic_vorticity : float
        Ω_harm = √(I_harmᵀ ★₁⁻¹ I_harm).
    total_vorticity : float
        Ω_vort + Ω_harm  (presupuesto topológico total).
    kcl_residual : float
        ‖∂₁ I_coexact‖₂  (debe ≈ 0).
    reconstruction_error : float
        ‖I − (exact+coexact+harmonic)‖₂.
    joule_exact : float
        Energía de la parte exacta en métrica ★₁⁻¹.
    joule_coexact : float
        Energía de la parte coexacta.
    verdict : HodgeSovereignVerdict
        Clasificación en el retículo Ω₃.
    crowbar_action : CrowbarAction
        Acción de actuador derivada del veredicto.
    veto_reasons : Tuple[str, ...]
        Lista de causas de degradación / veto (vacía si COHERENT).
    """
    orientation_stamp: Phase2HodgeOrientation
    exact_norm: float
    coexact_norm: float
    harmonic_norm: float
    parasitic_vorticity: float
    harmonic_vorticity: float
    total_vorticity: float
    kcl_residual: float
    reconstruction_error: float
    joule_exact: float
    joule_coexact: float
    verdict: HodgeSovereignVerdict
    crowbar_action: CrowbarAction
    veto_reasons: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class HodgeSovereignState:
    """
    Certificado inmutable del estado global de gobernanza de Hodge (Act).

    Cierra el ciclo OODA y es el artefacto que consumen los estratos
    superiores (TACTICS / STRATEGY) y el firmware de protección.
    """
    decision_stamp: Phase3HodgeDecision
    is_secure: bool
    is_crowbar_active: bool
    crowbar_gpio: int
    timestamp_utc: str
    stratum: str = "PHYSICS"
    agent_version: str = "4.0.0"
    ooda_latency_hints: Tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
# FASE 1 — OBSERVACIÓN ESPECTRAL Y SANEAMIENTO DE LA MÉTRICA (Observe)
#           Último método: emit_phase1_observation → Phase1HodgeObservation
#           que es el contrato de arranque obligatorio de la FASE 2.
# =============================================================================

class Phase1_SpectralMetricObserver:
    r"""
    Fase 1 del endofuntor OODA: sanea y valida la estabilidad espectral
    del vector de pesos constitutivos que define ★_k.

    Pipeline interno
    ----------------
    1. Finitud (NaN/Inf).
    2. Positividad física (w_i > 0).
    3. Análisis espectral de Wilkinson (κ, λ_min, λ_max).
    4. Regularización de Tikhonov adaptativa si κ > κ_max o λ_min < ε.
    5. (Si motor physics disponible) construcción de DiscreteHodgeStar +
       HodgeDualityMorphism certificado.
    6. Emisión del DTO Phase1HodgeObservation.
    """

    def __init__(
        self,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
    ) -> None:
        self._condition_max = float(condition_max)
        self._spectral_tol = float(spectral_tol)

    # ── validaciones atómicas ────────────────────────────────────────────────

    def _assert_finite(self, weights: RealVector) -> RealVector:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size == 0:
            raise NonFiniteWeightError(
                "Axioma de No-Trivialidad: vector de pesos vacío."
            )
        if not np.all(np.isfinite(w)):
            raise NonFiniteWeightError(
                "Fuga de FPU: el vector de pesos de Hodge contiene NaN o Inf."
            )
        return w

    def _assert_positive(self, weights: RealVector) -> None:
        if np.any(weights <= 0.0):
            n_bad = int(np.sum(weights <= 0.0))
            raise MetricDegeneracyError(
                f"Violación de Pasividad: {n_bad} peso(s) ≤ 0 detectado(s) "
                "en el medio constitutivo."
            )

    def _spectral_invariants(
        self, weights: RealVector
    ) -> Tuple[float, float, float, float, float]:
        """Retorna (λ_min, λ_max, κ, tr, log det)."""
        lam_min = float(np.min(weights))
        lam_max = float(np.max(weights))
        kappa = lam_max / (lam_min + _MACHINE_EPS)
        trace = float(np.sum(weights))
        log_det = float(np.sum(np.log(np.maximum(weights, _MACHINE_EPS))))
        return lam_min, lam_max, kappa, trace, log_det

    def _tikhonov_regularize(
        self,
        weights: RealVector,
        lam_min: float,
        lam_max: float,
        kappa: float,
    ) -> Tuple[RealVector, float, bool]:
        r"""
        Desplazamiento de Tikhonov adaptativo:
        $$\tilde{w}_i = w_i + \delta, \quad
          \delta = \lambda_{\max}\cdot\varepsilon_{\mathrm{mach}}\cdot 10^4.$$

        Se aplica solo si λ_min < tol  ∨  κ > κ_max.
        """
        needs = (lam_min < self._spectral_tol) or (kappa > self._condition_max)
        if not needs:
            return weights, 0.0, False

        shift = max(lam_max * _MACHINE_EPS * _TIKHONOV_SCALE, self._spectral_tol)
        logger.warning(
            "FASE1 Tikhonov: κ=%.4e λ_min=%.4e → shift=%.4e",
            kappa, lam_min, shift,
        )
        w_reg = weights + shift
        return w_reg, float(shift), True

    def _try_physics_duality(
        self, weights: RealVector
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Intenta construir HodgeDualityMorphism vía motor physics v2."""
        if not _HAS_PHYSICS_ENGINE:
            return None, None
        try:
            star = DiscreteHodgeStar(
                weights, degree=HodgeDegree.ONE, validate_strict=True
            )
            morph = star.induce_duality_morphism()
            state = star.audit_metric(tolerance=self._spectral_tol)
            return morph, state
        except Exception as exc:  # pragma: no cover
            logger.debug("Motor physics no disponible en FASE1: %s", exc)
            return None, None

    # ── método público principal de FASE 1 ───────────────────────────────────

    def audit_metric_weights(
        self,
        weights: RealVector,
        tolerance: Optional[float] = None,
    ) -> Phase1HodgeObservation:
        r"""
        Valida y sanea el vector de pesos del medio constitutivo.

        Mapea $w\mapsto\star_1=\operatorname{diag}(w)$ y garantiza SPD
        bajo el criterio de Sylvester, aplicando Tikhonov si es necesario:

        $$\tilde{\star}_1 = \star_1 + \delta I_m.$$

        Parameters
        ----------
        weights :
            Vector de pesos (conductancias / permeabilidades) $w\in\mathbb{R}^m_{>0}$.
        tolerance :
            Épsilon de Wilkinson para el gap espectral mínimo.
            Si None, usa el configurado en el constructor.

        Returns
        -------
        Phase1HodgeObservation
            DTO inmutable — contrato de continuidad hacia FASE 2.

        Raises
        ------
        NonFiniteWeightError
        MetricDegeneracyError
        """
        tol = self._spectral_tol if tolerance is None else float(tolerance)

        # 1–2. Finitud + positividad
        w = self._assert_finite(weights)
        self._assert_positive(w)

        # 3. Espectro crudo
        lam_min, lam_max, kappa, trace, log_det = self._spectral_invariants(w)

        # 4. Tikhonov adaptativo
        w_san, shift, reg = self._tikhonov_regularize(w, lam_min, lam_max, kappa)
        if reg:
            lam_min, lam_max, kappa, trace, log_det = self._spectral_invariants(w_san)

        is_spd = (lam_min >= tol) and (kappa <= self._condition_max)
        if not is_spd:
            raise MetricDegeneracyError(
                f"★_k numéricamente singular tras Tikhonov: "
                f"κ={kappa:.4e} > {self._condition_max:.4e}, "
                f"λ_min={lam_min:.4e}."
            )

        hodge = np.diag(w_san)

        # 5. Motor physics (duality morphism certificado)
        morph, mstate = self._try_physics_duality(w_san)

        return Phase1HodgeObservation(
            dimension=int(w_san.size),
            condition_number=float(kappa),
            min_eigenvalue=float(lam_min),
            max_eigenvalue=float(lam_max),
            is_strictly_spd=bool(is_spd),
            regularized_applied=bool(reg),
            tikhonov_shift=float(shift),
            weights_sanitized=w_san.copy(),
            hodge_matrix=hodge,
            duality_morphism_payload=morph,
            metric_state_payload=mstate,
            spectral_trace=float(trace),
            log_determinant=float(log_det),
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 1 ────────────────────────────────────
    # Su valor de retorno (Phase1HodgeObservation) es el objeto de arranque
    # obligatorio de la FASE 2 (continuidad de la definición formal).

    def emit_phase1_observation(
        self,
        weights: RealVector,
        tolerance: Optional[float] = None,
    ) -> Phase1HodgeObservation:
        r"""
        Punto de emisión canónico de la FASE 1.

        Envuelve ``audit_metric_weights`` y garantiza que el DTO saliente
        satisface los invariantes de continuidad funtorial exigidos por
        ``Phase2_HodgeLaplacianOrienter.orient_from_observation``.

        Invariantes post-condición
        --------------------------
        * ``observation.is_strictly_spd is True``
        * ``observation.dimension == len(weights)`` (o len post-Tikhonov)
        * ``observation.hodge_matrix.shape == (m, m)``
        * ``observation.condition_number <= condition_max``

        Returns
        -------
        Phase1HodgeObservation
            Semilla formal de la FASE 2.
        """
        obs = self.audit_metric_weights(weights, tolerance=tolerance)
        # Post-condiciones de continuidad
        if not obs.is_strictly_spd:
            raise MetricDegeneracyError(
                "Post-condición FASE1 violada: is_strictly_spd=False."
            )
        if obs.hodge_matrix.shape != (obs.dimension, obs.dimension):
            raise MetricDegeneracyError(
                "Post-condición FASE1 violada: shape(★) ≠ (m,m)."
            )
        if obs.condition_number > self._condition_max + 1.0:
            # +1 holgura numérica
            raise MetricDegeneracyError(
                "Post-condición FASE1 violada: κ excede cota tras emisión."
            )
        logger.info(
            "FASE1 EMIT dim=%d κ=%.3e λ_min=%.3e reg=%s",
            obs.dimension, obs.condition_number, obs.min_eigenvalue,
            obs.regularized_applied,
        )
        return obs


# =============================================================================
# FASE 2 — LAPLACIANO DE HODGE Y PASIVIDAD PORT-HAMILTONIANA (Orient)
#           Continúa desde Phase1HodgeObservation (emit_phase1_observation).
#           Último método: emit_phase2_orientation → Phase2HodgeOrientation
#           que es el contrato de arranque obligatorio de la FASE 3.
# =============================================================================

class Phase2_HodgeLaplacianOrienter(Phase1_SpectralMetricObserver):
    r"""
    Fase 2 del endofuntor OODA: ensambla el Laplaciano de Hodge ponderado
    de grado 0, audita la disipación de Dirichlet y extrae invariantes
    topológicos (Betti, Fiedler).

    Ecuaciones gobernantes
    ---------------------
    .. math::

        L_0^\star &= \partial_1\,\star_1\,\partial_1^\top,\\
        E_\star(\phi) &= \tfrac12\phi^\top L_0^\star\phi,\\
        \dot H &= -2 E_\star(\phi) \le 0 \quad\text{(pasividad)}.

    Si el motor physics v2 está presente se reutiliza
    ``WeightedHodgeLaplacian`` + ``certify_de_rham_complex`` para obtener
    β₀, β₁ y el espectro con residuos certificados.
    """

    def __init__(
        self,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
        max_betti_1: int = 0,
    ) -> None:
        super().__init__(condition_max=condition_max, spectral_tol=spectral_tol)
        self._max_betti_1 = int(max_betti_1)

    # ── construcción del Laplaciano ──────────────────────────────────────────

    def _assemble_laplacian_0(
        self,
        star_1: RealMatrix,
        boundary_1: RealMatrix,
    ) -> RealMatrix:
        r"""L₀★ = B₁ ★₁ B₁ᵀ  con proyección hermítica de seguridad."""
        L = boundary_1 @ star_1 @ boundary_1.T
        # Proyección al subespacio de matrices simétricas (error de redondeo)
        skew = float(np.max(np.abs(L - L.T)))
        if skew > 100.0 * _MACHINE_EPS * max(L.shape):
            logger.warning(
                "FASE2: desvío de simetría en L₀ ||A-Aᵀ||_∞=%.3e — proyectando.",
                skew,
            )
        return 0.5 * (L + L.T)

    def _laplacian_spectrum(
        self, L: RealMatrix, tol: float
    ) -> Tuple[RealVector, float, float, float, int, bool, float]:
        """
        Retorna
        (eigenvalues, κ_rango, λ_Fiedler, λ_gap, β₀, is_self_adjoint, max_residual).
        """
        # Autoadjunción ya garantizada por proyección; re-chequeo barato
        is_sa = float(np.max(np.abs(L - L.T))) < 100.0 * _MACHINE_EPS * max(L.shape[0], 1)

        ev = la.eigvalsh(L)
        # Orden no-decreciente garantizado por eigvalsh
        max_eig = float(ev[-1]) if ev.size else 0.0
        tol_ker = tol * max(1.0, abs(max_eig))
        betti0 = int(np.sum(np.abs(ev) < tol_ker))
        nonzero = ev[ev >= tol_ker]
        if nonzero.size == 0:
            lam_fiedler = 0.0
            lam_gap = 0.0
            kappa = 1.0
        else:
            lam_gap = float(nonzero[0])
            lam_fiedler = lam_gap  # λ_{β₀+1}
            kappa = max_eig / (lam_gap + _MACHINE_EPS)

        # Residuo de un par propio representativo (extremal)
        max_res = 0.0
        if L.shape[0] > 0 and ev.size > 0:
            # Usar eigh completo solo si n pequeño; si no, residuo del mayor
            n = L.shape[0]
            if n <= 256:
                ev_f, V = la.eigh(L)
                for i in range(n):
                    r = L @ V[:, i] - ev_f[i] * V[:, i]
                    max_res = max(max_res, float(np.max(np.abs(r))))
            else:
                # Estimación barata con potencia
                v = np.ones(n) / math.sqrt(n)
                Lv = L @ v
                lam_est = float(v @ Lv)
                max_res = float(np.max(np.abs(Lv - lam_est * v)))

        return ev, float(kappa), float(lam_fiedler), float(lam_gap), betti0, is_sa, max_res

    def _dirichlet_energy(self, L: RealMatrix, phi: RealVector) -> float:
        r"""E_★(φ) = ½ φᵀ L φ."""
        phi = np.asarray(phi, dtype=np.float64).ravel()
        if phi.shape[0] != L.shape[0]:
            raise MetricDegeneracyError(
                f"Dimensión de potencial ({phi.shape[0]}) ≠ nodos ({L.shape[0]})."
            )
        return float(0.5 * phi @ (L @ phi))

    def _try_physics_laplacian(
        self,
        observation: Phase1HodgeObservation,
        boundary_1: RealMatrix,
        weights_0: Optional[RealVector],
    ) -> Tuple[Optional[Any], Optional[int], Optional[int], Optional[float]]:
        """
        Intenta WeightedHodgeLaplacian + certify_de_rham_complex.
        Retorna (cert, β₀, β₁, energy_scale) o (None,…).
        """
        if not _HAS_PHYSICS_ENGINE:
            return None, None, None, None
        try:
            n_nodes = boundary_1.shape[0]
            w0 = (
                np.asarray(weights_0, dtype=np.float64).ravel()
                if weights_0 is not None
                else np.ones(n_nodes, dtype=np.float64)
            )
            w1 = observation.weights_sanitized
            star0 = DiscreteHodgeStar(w0, degree=HodgeDegree.ZERO)
            star1 = DiscreteHodgeStar(w1, degree=HodgeDegree.ONE)
            dual0 = star0.induce_duality_morphism()
            dual1 = (
                observation.duality_morphism_payload
                if observation.duality_morphism_payload is not None
                else star1.induce_duality_morphism()
            )
            lap = WeightedHodgeLaplacian(boundary_1, dual0, dual1)
            cert = lap.certify_de_rham_complex()
            return cert, cert.betti_0, cert.betti_1, cert.energy_scale
        except Exception as exc:  # pragma: no cover
            logger.debug("Motor physics FASE2 fallback: %s", exc)
            return None, None, None, None

    def _estimate_betti_1_kernel(
        self,
        boundary_1: RealMatrix,
        star_1: RealMatrix,
        tol: float,
    ) -> int:
        r"""
        Estimación de β₁ sin motor physics:
        dim ker(★₁ ∂₁ᵀ ★₀⁻¹ ∂₁) ≈ dim ker(B₁) cuando ★₀=I
        (fórmula del ciclo: m − n + β₀ para grafos conexos).
        """
        # Rango numérico de B₁
        s = la.svdvals(boundary_1)
        rank = int(np.sum(s > tol * max(1.0, float(s[0]) if s.size else 1.0)))
        n_nodes, n_edges = boundary_1.shape
        # β₀ se estima aparte; aquí usamos rank-nullity crudo
        # dim ker B₁ = m − rank(B₁)
        return max(n_edges - rank, 0)

    # ── método público principal de FASE 2 ───────────────────────────────────

    def orient_laplacian_dynamics(
        self,
        observation: Phase1HodgeObservation,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        tolerance: Optional[float] = None,
    ) -> Phase2HodgeOrientation:
        r"""
        Ensambla L₀★, calcula energía de Dirichlet e invariantes topológicos.

        Parameters
        ----------
        observation :
            Certificado FASE 1 (obligatorio — continuidad formal).
        boundary_matrix :
            ∂₁ de tamaño (n, m).
        potential_vector :
            Potenciales nodales φ ∈ ℝⁿ.
        weights_0 :
            Pesos de ★₀ (nodos). Si None, se asume 𝟙.
        tolerance :
            Tolerancia espectral.

        Returns
        -------
        Phase2HodgeOrientation
        """
        if not isinstance(observation, Phase1HodgeObservation):
            raise TypeError(
                "FASE2 exige Phase1HodgeObservation emitido por FASE1."
            )
        if not observation.is_strictly_spd:
            raise MetricDegeneracyError(
                "FASE2 rechaza observación no-SPD (invariante de continuidad)."
            )

        tol = self._spectral_tol if tolerance is None else float(tolerance)
        B = np.asarray(boundary_matrix, dtype=np.float64)
        if B.ndim != 2:
            raise MetricDegeneracyError("∂₁ debe ser matriz 2-D.")
        n_nodes, n_edges = B.shape
        if n_edges != observation.dimension:
            raise MetricDegeneracyError(
                f"Incompatibilidad ∂₁/★₁: aristas={n_edges} ≠ "
                f"dim(★₁)={observation.dimension}."
            )

        star_1 = observation.hodge_matrix
        L0 = self._assemble_laplacian_0(star_1, B)

        # Espectro + Betti₀ + Fiedler
        ev, kappa_L, lam_fiedler, lam_gap, betti0, is_sa, max_res = (
            self._laplacian_spectrum(L0, tol)
        )

        # Energía de Dirichlet + pasividad
        energy = self._dirichlet_energy(L0, potential_vector)
        is_passive = energy >= _SPECTRAL_PSD_FLOOR
        if not is_passive:
            logger.error(
                "FASE2: violación de pasividad E_★=%.4e < floor", energy
            )

        # Intento con motor physics (β₀, β₁ certificados)
        cert, b0_p, b1_p, e_scale = self._try_physics_laplacian(
            observation, B, weights_0
        )
        if b0_p is not None:
            betti0 = b0_p
        betti1 = b1_p if b1_p is not None else self._estimate_betti_1_kernel(B, star_1, tol)
        energy_scale = (
            float(e_scale) if e_scale is not None
            else float(observation.spectral_trace + n_nodes)
        )

        return Phase2HodgeOrientation(
            observation_stamp=observation,
            dirichlet_energy=energy,
            is_passive=bool(is_passive),
            laplacian_condition=float(kappa_L),
            laplacian_0=L0,
            algebraic_connectivity=float(lam_fiedler),
            betti_0=int(betti0),
            betti_1=int(betti1),
            is_self_adjoint=bool(is_sa),
            max_eigenresidual=float(max_res),
            spectral_gap_laplacian=float(lam_gap),
            energy_scale=energy_scale,
            de_rham_certificate_payload=cert,
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 2 ────────────────────────────────────
    # Su valor de retorno (Phase2HodgeOrientation) es el objeto de arranque
    # obligatorio de la FASE 3.

    def emit_phase2_orientation(
        self,
        observation: Phase1HodgeObservation,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        tolerance: Optional[float] = None,
        enforce_passivity: bool = True,
        enforce_betti_1: bool = False,
    ) -> Phase2HodgeOrientation:
        r"""
        Punto de emisión canónico de la FASE 2.

        Envuelve ``orient_laplacian_dynamics`` y aplica post-condiciones
        de continuidad funtorial exigidas por
        ``Phase3_HelmholtzDecisionMaker.decide_from_orientation``.

        Parameters
        ----------
        enforce_passivity :
            Si True, lanza LaplacianPassivityError cuando E_★ < 0.
        enforce_betti_1 :
            Si True, lanza BettiNumberVeto cuando β₁ > max_betti_1.

        Returns
        -------
        Phase2HodgeOrientation
            Semilla formal de la FASE 3.
        """
        orient = self.orient_laplacian_dynamics(
            observation,
            boundary_matrix,
            potential_vector,
            weights_0=weights_0,
            tolerance=tolerance,
        )

        # Post-condiciones
        if not orient.is_self_adjoint:
            logger.warning("FASE2 post: L₀ no autoadjunto numéricamente.")
        if enforce_passivity and not orient.is_passive:
            raise LaplacianPassivityError(
                f"Post-condición FASE2: E_★={orient.dirichlet_energy:.4e} < 0 "
                "(pasividad Port-Hamiltoniana rota)."
            )
        if enforce_betti_1 and orient.betti_1 > self._max_betti_1:
            raise BettiNumberVeto(
                f"Post-condición FASE2: β₁={orient.betti_1} > "
                f"max_admisible={self._max_betti_1}."
            )
        if orient.observation_stamp is not observation:
            raise MetricDegeneracyError(
                "Post-condición FASE2: observation_stamp no preserva identidad."
            )

        logger.info(
            "FASE2 EMIT β₀=%d β₁=%d E_★=%.3e κ_L=%.3e Fiedler=%.3e passive=%s",
            orient.betti_0, orient.betti_1, orient.dirichlet_energy,
            orient.laplacian_condition, orient.algebraic_connectivity,
            orient.is_passive,
        )
        return orient


# =============================================================================
# FASE 3 — HELMHOLTZ-HODGE + RETÍCULO Ω₃ + DECISIÓN CROWBAR (Decide / Act-prep)
#           Continúa desde Phase2HodgeOrientation (emit_phase2_orientation).
#           Último método: emit_phase3_decision → Phase3HodgeDecision
#           que es el contrato de arranque del Soberano (Crowbar / State).
# =============================================================================

class Phase3_HelmholtzDecisionMaker(Phase2_HodgeLaplacianOrienter):
    r"""
    Fase 3 del endofuntor OODA: proyecta el espacio de corrientes sobre la
    descomposición ortogonal de Helmholtz-Hodge y clasifica el estado en el
    retículo distributivo Ω₃.

    .. math::

        I = I_{\mathrm{exact}} + I_{\mathrm{coexact}} + I_{\mathrm{harmonic}}

    con ortogonalidad en la métrica de Joule $\star_1^{-1}$:

    .. math::

        \langle I_a, I_b\rangle_{\star_1^{-1}} = 0
        \quad(a\neq b\in\{\mathrm{ex},\mathrm{co},\mathrm{har}\}).

    La vorticidad parasitaria

    .. math::

        \Omega_{\mathrm{vort}}
        = \sqrt{I_{\mathrm{coexact}}^\top \star_1^{-1} I_{\mathrm{coexact}}}

    se somete al clasificador de Heyting acotado para emitir el veredicto.
    """

    def __init__(
        self,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
        max_betti_1: int = 0,
        vorticity_threshold: float = 1.0,
        coherent_fraction: float = 0.1,
    ) -> None:
        super().__init__(
            condition_max=condition_max,
            spectral_tol=spectral_tol,
            max_betti_1=max_betti_1,
        )
        self._vorticity_threshold = float(vorticity_threshold)
        self._coherent_fraction = float(coherent_fraction)

    # ── Helmholtz-Hodge (nativo + delegación al motor physics) ───────────────

    def _helmholtz_native(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_1: RealMatrix,
        current: RealVector,
        tol: float,
    ) -> Tuple[RealVector, RealVector, RealVector, float, float]:
        r"""
        Descomposición nativa (sin motor physics):

        1. Resolver L₀ φ = ∂₁ I   (Poisson, pseudoinversa).
        2. I_exact = ★₁ ∂₁ᵀ φ.
        3. I_coexact = I − I_exact  (incluye armónicos si β₁>0).
        4. I_harmonic = 0  (refinado aparte si β₁>0 vía SVD del ciclo).

        Returns
        -------
        exact, coexact, harmonic, kcl_residual, reconstruction_error
        """
        B = boundary_1
        star_1 = orientation.observation_stamp.hodge_matrix
        L0 = orientation.laplacian_0
        I = np.asarray(current, dtype=np.float64).ravel()

        if I.shape[0] != orientation.observation_stamp.dimension:
            raise HelmholtzDecompositionError(
                f"dim(I)={I.shape[0]} ≠ m={orientation.observation_stamp.dimension}."
            )

        # Poisson: L₀ φ = B I
        source = B @ I
        try:
            phi = la.lstsq(L0, source, cond=tol)[0]
        except Exception as exc:
            raise HelmholtzDecompositionError(
                f"Fallo al resolver Poisson L₀φ = BI: {exc}"
            ) from exc

        exact = star_1 @ (B.T @ phi)
        residual_current = I - exact

        # Separar armónicos si β₁ > 0: ker(B) ∩ residual
        harmonic = np.zeros_like(I)
        if orientation.betti_1 > 0:
            # Base de ker(B) vía SVD
            # B = U S Vh  →  ker = últimas m-rank columnas de Vh.T = filas de Vh
            try:
                U, S, Vh = la.svd(B, full_matrices=True)
                rank = int(np.sum(S > tol * max(1.0, float(S[0]) if S.size else 1.0)))
                ker_dim = B.shape[1] - rank
                if ker_dim > 0:
                    ker_basis = Vh[rank:].T  # (m, ker_dim)
                    # Proyectar residual_current sobre ker (ℓ²; luego se mide en ★⁻¹)
                    coeffs = ker_basis.T @ residual_current
                    harmonic = ker_basis @ coeffs
            except Exception as exc:  # pragma: no cover
                logger.debug("SVD ker(B) falló: %s — harmonic=0", exc)

        coexact = residual_current - harmonic

        kcl = float(la.norm(B @ (coexact + harmonic), ord=2))
        recon = exact + coexact + harmonic
        recon_err = float(la.norm(I - recon, ord=2))

        return exact, coexact, harmonic, kcl, recon_err

    def _helmholtz_physics(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_1: RealMatrix,
        current: RealVector,
        weights_0: Optional[RealVector],
    ) -> Optional[Tuple[RealVector, RealVector, RealVector, float, float]]:
        """Delega en DiscreteDeRhamComplex.helmholtz_hodge_decomposition."""
        if not _HAS_PHYSICS_ENGINE:
            return None
        cert = orientation.de_rham_certificate_payload
        try:
            if cert is None:
                # Construir al vuelo
                n_nodes = boundary_1.shape[0]
                w0 = (
                    np.asarray(weights_0, dtype=np.float64).ravel()
                    if weights_0 is not None
                    else np.ones(n_nodes, dtype=np.float64)
                )
                cplx = build_certified_de_rham_complex(
                    weights_0=w0,
                    weights_1=orientation.observation_stamp.weights_sanitized,
                    boundary_1=boundary_1,
                )
            else:
                cplx = DiscreteDeRhamComplex(cert)

            dec: HelmholtzHodgeDecomposition = cplx.helmholtz_hodge_decomposition(
                np.asarray(current, dtype=np.float64).ravel()
            )
            kcl = float(
                la.norm(boundary_1 @ (dec.coexact_part + dec.harmonic_part), ord=2)
            )
            return (
                dec.exact_part,
                dec.coexact_part,
                dec.harmonic_part,
                kcl,
                float(dec.residual_norm),
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Physics Helmholtz fallback → native: %s", exc)
            return None

    def _metric_norm_star_inv(
        self, vec: RealVector, star_1: RealMatrix
    ) -> float:
        r"""‖v‖_{★⁻¹} = √(vᵀ ★⁻¹ v)."""
        w = np.diag(star_1)
        # ★ diagonal ⇒ ★⁻¹ = diag(1/w)
        return float(math.sqrt(max(np.sum(vec * vec / w), 0.0)))

    def _metric_energy_star_inv(
        self, vec: RealVector, star_1: RealMatrix
    ) -> float:
        r"""Energía de Joule ½ vᵀ ★⁻¹ v."""
        w = np.diag(star_1)
        return float(0.5 * np.sum(vec * vec / w))

    def _classify_omega3(
        self,
        parasitic: float,
        harmonic_v: float,
        total_v: float,
        kcl: float,
        recon_err: float,
        orientation: Phase2HodgeOrientation,
        threshold: float,
    ) -> Tuple[HodgeSovereignVerdict, CrowbarAction, Tuple[str, ...]]:
        """Clasificador de Heyting acotado → (veredicto, acción, razones)."""
        reasons: List[str] = []
        verdict = HodgeSovereignVerdict.COHERENT

        # --- criterios de VETO duro ---
        if not orientation.is_passive:
            verdict = HodgeSovereignVerdict.VETOED
            reasons.append("passivity_violated")
        if total_v > threshold:
            verdict = HodgeSovereignVerdict.VETOED
            reasons.append(
                f"total_vorticity={total_v:.4e}>{threshold:.4e}"
            )
        if orientation.betti_1 > self._max_betti_1 and total_v > threshold * 0.5:
            verdict = HodgeSovereignVerdict.VETOED
            reasons.append(
                f"betti_1={orientation.betti_1}>max={self._max_betti_1}"
            )
        if recon_err > _RECONSTRUCTION_TOL * 100:
            verdict = HodgeSovereignVerdict.VETOED
            reasons.append(f"reconstruction_error={recon_err:.4e}")

        # --- criterios de DEGRADED ---
        if verdict == HodgeSovereignVerdict.COHERENT:
            coh_cut = threshold * self._coherent_fraction
            if total_v > coh_cut:
                verdict = HodgeSovereignVerdict.DEGRADED
                reasons.append(
                    f"vorticity_degraded={total_v:.4e}>{coh_cut:.4e}"
                )
            if kcl > _KCL_TOL:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append(f"kcl_residual={kcl:.4e}")
            if orientation.laplacian_condition > self._condition_max * 0.1:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append(
                    f"laplacian_kappa={orientation.laplacian_condition:.4e}"
                )
            if orientation.observation_stamp.regularized_applied:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append("tikhonov_regularized")

        # --- acción Crowbar ---
        if verdict == HodgeSovereignVerdict.VETOED:
            action = CrowbarAction.HARD_SHORT
        elif verdict == HodgeSovereignVerdict.DEGRADED:
            action = CrowbarAction.WATCHDOG_PULSE
        else:
            action = CrowbarAction.NONE

        return verdict, action, tuple(reasons)

    # ── método público principal de FASE 3 ───────────────────────────────────

    def resolve_helmholtz_decomposition(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_matrix: RealMatrix,
        current_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        vorticity_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> Phase3HodgeDecision:
        r"""
        Ejecuta Helmholtz-Hodge + clasificación Ω₃.

        Parameters
        ----------
        orientation :
            Certificado FASE 2 (obligatorio — continuidad formal).
        boundary_matrix :
            ∂₁ (n, m).
        current_vector :
            Corrientes de arista I ∈ ℝᵐ.
        weights_0 :
            Pesos nodales (para motor physics).
        vorticity_threshold :
            Umbral crítico de vorticidad.
        tolerance :
            Tolerancia de Poisson / SVD.

        Returns
        -------
        Phase3HodgeDecision
        """
        if not isinstance(orientation, Phase2HodgeOrientation):
            raise TypeError(
                "FASE3 exige Phase2HodgeOrientation emitido por FASE2."
            )

        tol = self._spectral_tol if tolerance is None else float(tolerance)
        thr = (
            self._vorticity_threshold
            if vorticity_threshold is None
            else float(vorticity_threshold)
        )
        B = np.asarray(boundary_matrix, dtype=np.float64)
        I = np.asarray(current_vector, dtype=np.float64).ravel()
        star_1 = orientation.observation_stamp.hodge_matrix

        # Preferir motor physics; fallback nativo
        phys = self._helmholtz_physics(orientation, B, I, weights_0)
        if phys is not None:
            exact, coexact, harmonic, kcl, recon_err = phys
        else:
            exact, coexact, harmonic, kcl, recon_err = self._helmholtz_native(
                orientation, B, I, tol
            )

        if recon_err > _RECONSTRUCTION_TOL:
            # Tolerancia dura de isomorfismo
            if recon_err > 1e-6:
                raise HelmholtzDecompositionError(
                    f"Error en isomorfismo de de Rham: "
                    f"desviación ortogonal={recon_err:.4e}."
                )
            logger.warning(
                "FASE3: reconstruction_error=%.3e (dentro de holgura).", recon_err
            )

        # Normas ℓ² y métricas ★⁻¹
        exact_n = float(la.norm(exact, ord=2))
        coexact_n = float(la.norm(coexact, ord=2))
        harmonic_n = float(la.norm(harmonic, ord=2))

        parasitic = self._metric_norm_star_inv(coexact, star_1)
        harm_v = self._metric_norm_star_inv(harmonic, star_1)
        total_v = parasitic + harm_v

        joule_ex = self._metric_energy_star_inv(exact, star_1)
        joule_co = self._metric_energy_star_inv(coexact, star_1)

        verdict, action, reasons = self._classify_omega3(
            parasitic, harm_v, total_v, kcl, recon_err, orientation, thr
        )

        return Phase3HodgeDecision(
            orientation_stamp=orientation,
            exact_norm=exact_n,
            coexact_norm=coexact_n,
            harmonic_norm=harmonic_n,
            parasitic_vorticity=float(parasitic),
            harmonic_vorticity=float(harm_v),
            total_vorticity=float(total_v),
            kcl_residual=float(kcl),
            reconstruction_error=float(recon_err),
            joule_exact=float(joule_ex),
            joule_coexact=float(joule_co),
            verdict=verdict,
            crowbar_action=action,
            veto_reasons=reasons,
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 3 ────────────────────────────────────
    # Su valor de retorno (Phase3HodgeDecision) es el objeto de arranque
    # obligatorio del Soberano DiscreteHodgeStarAgent (Crowbar / State).

    def emit_phase3_decision(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_matrix: RealMatrix,
        current_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        vorticity_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
        raise_on_veto: bool = False,
    ) -> Phase3HodgeDecision:
        r"""
        Punto de emisión canónico de la FASE 3.

        Envuelve ``resolve_helmholtz_decomposition`` y garantiza
        post-condiciones de continuidad hacia el Soberano.

        Parameters
        ----------
        raise_on_veto :
            Si True, materializa ParasiticVorticityVeto cuando
            verdict == VETOED (modo estricto de aduana).

        Returns
        -------
        Phase3HodgeDecision
            Semilla formal del Actuador Crowbar / HodgeSovereignState.
        """
        decision = self.resolve_helmholtz_decomposition(
            orientation,
            boundary_matrix,
            current_vector,
            weights_0=weights_0,
            vorticity_threshold=vorticity_threshold,
            tolerance=tolerance,
        )

        # Post-condiciones
        if decision.orientation_stamp is not orientation:
            raise HelmholtzDecompositionError(
                "Post-condición FASE3: orientation_stamp no preserva identidad."
            )
        if decision.reconstruction_error < 0.0:
            raise HelmholtzDecompositionError(
                "Post-condición FASE3: reconstruction_error negativo."
            )
        if raise_on_veto and decision.verdict == HodgeSovereignVerdict.VETOED:
            raise ParasiticVorticityVeto(
                "Aduana geométrica: veredicto VETOED. Razones: "
                + "; ".join(decision.veto_reasons)
            )

        logger.info(
            "FASE3 EMIT verdict=%s Ω_vort=%.3e Ω_harm=%.3e KCL=%.3e action=%s",
            decision.verdict.name,
            decision.parasitic_vorticity,
            decision.harmonic_vorticity,
            decision.kcl_residual,
            decision.crowbar_action.name,
        )
        return decision


# =============================================================================
# SOBERANO — DISCRETE HODGE STAR AGENT (orquestador OODA completo)
#             Continúa desde Phase3HodgeDecision (emit_phase3_decision).
# =============================================================================

class DiscreteHodgeStarAgent(Morphism, Phase3_HelmholtzDecisionMaker):
    r"""
    Agente soberano de gobernanza y control geométrico de la Estrella de Hodge.

    Orquesta las tres fases anidadas del ciclo OODA covariante:

    .. code-block:: text

        weights ──► FASE1.emit_phase1_observation ──► Phase1HodgeObservation
                          │
                          ▼
        (B,φ) ────► FASE2.emit_phase2_orientation ──► Phase2HodgeOrientation
                          │
                          ▼
        I ────────► FASE3.emit_phase3_decision    ──► Phase3HodgeDecision
                          │
                          ▼
                    Act (Crowbar GPIO)             ──► HodgeSovereignState

    El actuador Crowbar es inyectable (Protocol) para garantizar testabilidad
    sin hardware; por defecto se usa ``LoggingCrowbarActuator``.
    """

    def __init__(
        self,
        target_stratum: Stratum = Stratum.PHYSICS,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
        max_betti_1: int = 0,
        vorticity_threshold: float = 1.0,
        coherent_fraction: float = 0.1,
        crowbar_actuator: Optional[CrowbarActuator] = None,
        crowbar_gpio: int = _CROWBAR_GPIO,
        raise_on_veto: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        target_stratum :
            Estrato DIKW de publicación del certificado.
        condition_max :
            Cota superior de κ(★).
        spectral_tol :
            Tolerancia de Wilkinson.
        max_betti_1 :
            Máximo β₁ admisible antes de veto topológico.
        vorticity_threshold :
            Umbral crítico Ω_vort + Ω_harm.
        coherent_fraction :
            Fracción del umbral bajo la cual el estado es COHERENT.
        crowbar_actuator :
            Implementación del puerto GPIO (inyectable).
        crowbar_gpio :
            Número de pin del disyuntor (default 14).
        raise_on_veto :
            Propagar ParasiticVorticityVeto en modo aduana estricta.
        """
        Morphism.__init__(self)
        Phase3_HelmholtzDecisionMaker.__init__(
            self,
            condition_max=condition_max,
            spectral_tol=spectral_tol,
            max_betti_1=max_betti_1,
            vorticity_threshold=vorticity_threshold,
            coherent_fraction=coherent_fraction,
        )
        self._target_stratum = target_stratum
        self._actuator: CrowbarActuator = (
            crowbar_actuator if crowbar_actuator is not None
            else LoggingCrowbarActuator()
        )
        self._crowbar_gpio = int(crowbar_gpio)
        self._raise_on_veto = bool(raise_on_veto)
        self._last_state: Optional[HodgeSovereignState] = None

    # ── propiedades de inspección ────────────────────────────────────────────

    @property
    def last_state(self) -> Optional[HodgeSovereignState]:
        """Último certificado emitido (None si aún no hay ciclo)."""
        return self._last_state

    @property
    def target_stratum(self) -> Stratum:
        return self._target_stratum

    # ── Act: Crowbar ─────────────────────────────────────────────────────────

    def _act_crowbar(self, decision: Phase3HodgeDecision) -> bool:
        """
        Ejecuta la acción de actuador derivada del veredicto.
        Retorna is_crowbar_active.
        """
        if decision.crowbar_action == CrowbarAction.HARD_SHORT:
            reason = (
                f"VETOED Ω_total={decision.total_vorticity:.4e} "
                f"reasons={decision.veto_reasons}"
            )
            ok = self._actuator.assert_crowbar(self._crowbar_gpio, reason)
            if not ok:
                raise CrowbarActivationError(
                    f"Hardware no ACK-eó Crowbar en GPIO{self._crowbar_gpio}."
                )
            logger.error(
                "¡VETO GEOMÉTRICO! Crowbar ACTIVO gpio=%d | %s",
                self._crowbar_gpio, reason,
            )
            return True

        if decision.crowbar_action == CrowbarAction.WATCHDOG_PULSE:
            logger.warning(
                "DEGRADED: watchdog pulse (sin hard-short). reasons=%s",
                decision.veto_reasons,
            )
            # Política: no hard-short en DEGRADED; solo telemetría.
            return False

        # COHERENT — asegurar liberación
        self._actuator.deassert_crowbar(self._crowbar_gpio)
        logger.info(
            "Gobernanza de Hodge COHERENT. Veredicto=%s Ω_total=%.4e",
            decision.verdict.name, decision.total_vorticity,
        )
        return False

    def _emergency_veto_state(
        self,
        weights: RealVector,
        boundary_matrix: RealMatrix,
        err: Exception,
    ) -> HodgeSovereignState:
        """Construye un HodgeSovereignState de emergencia (fail-safe VETOED)."""
        w = np.asarray(weights, dtype=np.float64).ravel()
        n = int(boundary_matrix.shape[0]) if boundary_matrix is not None else 1
        m = int(w.size) if w.size > 0 else 1
        w_safe = w if w.size > 0 else np.ones(1)

        dummy_obs = Phase1HodgeObservation(
            dimension=m,
            condition_number=float("inf"),
            min_eigenvalue=0.0,
            max_eigenvalue=float("inf"),
            is_strictly_spd=False,
            regularized_applied=False,
            tikhonov_shift=0.0,
            weights_sanitized=w_safe,
            hodge_matrix=np.diag(w_safe) if w_safe.size else np.eye(1),
            spectral_trace=0.0,
            log_determinant=float("-inf"),
        )
        dummy_orient = Phase2HodgeOrientation(
            observation_stamp=dummy_obs,
            dirichlet_energy=0.0,
            is_passive=False,
            laplacian_condition=float("inf"),
            laplacian_0=np.zeros((n, n)),
            algebraic_connectivity=0.0,
            betti_0=0,
            betti_1=0,
            is_self_adjoint=False,
            max_eigenresidual=float("inf"),
            spectral_gap_laplacian=0.0,
            energy_scale=0.0,
        )
        dummy_decision = Phase3HodgeDecision(
            orientation_stamp=dummy_orient,
            exact_norm=0.0,
            coexact_norm=0.0,
            harmonic_norm=0.0,
            parasitic_vorticity=float("inf"),
            harmonic_vorticity=float("inf"),
            total_vorticity=float("inf"),
            kcl_residual=float("inf"),
            reconstruction_error=float("inf"),
            joule_exact=0.0,
            joule_coexact=0.0,
            verdict=HodgeSovereignVerdict.VETOED,
            crowbar_action=CrowbarAction.HARD_SHORT,
            veto_reasons=(f"emergency:{type(err).__name__}:{err}",),
        )

        # Intentar activar Crowbar incluso en emergencia
        try:
            self._actuator.assert_crowbar(
                self._crowbar_gpio,
                f"EMERGENCY {type(err).__name__}: {err}",
            )
            crowbar_on = True
        except Exception as act_exc:  # pragma: no cover
            logger.critical("Crowbar de emergencia falló: %s", act_exc)
            crowbar_on = True  # fail-safe: asumir activo

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return HodgeSovereignState(
            decision_stamp=dummy_decision,
            is_secure=False,
            is_crowbar_active=crowbar_on,
            crowbar_gpio=self._crowbar_gpio,
            timestamp_utc=ts,
            stratum=self._target_stratum.name,
            ooda_latency_hints=(f"emergency_path:{type(err).__name__}",),
        )

    # ── ciclo OODA completo ──────────────────────────────────────────────────

    def execute_sovereign_governance(
        self,
        weights: RealVector,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        current_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        vorticity_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> HodgeSovereignState:
        r"""
        Ejecuta el ciclo OODA completo sobre la métrica y los flujos.

        Encadena formalmente:

        1. ``emit_phase1_observation(weights)``
        2. ``emit_phase2_orientation(obs, B, φ)``
        3. ``emit_phase3_decision(orient, B, I)``
        4. ``_act_crowbar(decision)`` → ``HodgeSovereignState``

        Parameters
        ----------
        weights :
            Pesos de ★₁ (m,).
        boundary_matrix :
            ∂₁ (n, m).
        potential_vector :
            Potenciales nodales φ (n,).
        current_vector :
            Corrientes de arista I (m,).
        weights_0 :
            Pesos de ★₀ (n,); default 𝟙.
        vorticity_threshold :
            Override del umbral de vorticidad.
        tolerance :
            Override de la tolerancia espectral.

        Returns
        -------
        HodgeSovereignState
            Certificado inmutable de gobernanza (siempre; fail-safe en error).
        """
        hints: List[str] = []
        try:
            # ── FASE 1: Observe ──────────────────────────────────────────────
            obs = self.emit_phase1_observation(weights, tolerance=tolerance)
            hints.append("phase1:ok")

            # ── FASE 2: Orient ───────────────────────────────────────────────
            orient = self.emit_phase2_orientation(
                obs,
                boundary_matrix,
                potential_vector,
                weights_0=weights_0,
                tolerance=tolerance,
                enforce_passivity=False,  # se refleja en Ω₃, no se aborta
                enforce_betti_1=False,
            )
            hints.append("phase2:ok")

            # ── FASE 3: Decide ───────────────────────────────────────────────
            decision = self.emit_phase3_decision(
                orient,
                boundary_matrix,
                current_vector,
                weights_0=weights_0,
                vorticity_threshold=vorticity_threshold,
                tolerance=tolerance,
                raise_on_veto=self._raise_on_veto,
            )
            hints.append("phase3:ok")

            # ── ACT: Crowbar ─────────────────────────────────────────────────
            crowbar_on = self._act_crowbar(decision)
            hints.append(f"act:crowbar={'ON' if crowbar_on else 'OFF'}")

            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            state = HodgeSovereignState(
                decision_stamp=decision,
                is_secure=(decision.verdict != HodgeSovereignVerdict.VETOED),
                is_crowbar_active=crowbar_on,
                crowbar_gpio=self._crowbar_gpio,
                timestamp_utc=ts,
                stratum=self._target_stratum.name,
                ooda_latency_hints=tuple(hints),
            )
            self._last_state = state
            return state

        except HodgeStarAgentError as err:
            logger.critical(
                "Colapso de invariante en gobernanza de Hodge: %s. "
                "Fail-safe Crowbar + VETOED.",
                err,
            )
            state = self._emergency_veto_state(weights, boundary_matrix, err)
            self._last_state = state
            return state

        except Exception as err:  # pragma: no cover
            logger.critical(
                "Excepción no clasificada en gobernanza de Hodge: %s", err
            )
            state = self._emergency_veto_state(weights, boundary_matrix, err)
            self._last_state = state
            return state

    # ── API de conveniencia: ciclo parcial (solo observación) ────────────────

    def observe_only(self, weights: RealVector, **kw: Any) -> Phase1HodgeObservation:
        """Ejecuta únicamente FASE 1 (diagnóstico métrico)."""
        return self.emit_phase1_observation(weights, **kw)

    def observe_and_orient(
        self,
        weights: RealVector,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        **kw: Any,
    ) -> Phase2HodgeOrientation:
        """Ejecuta FASE 1 → FASE 2 (sin decisión ni Crowbar)."""
        obs = self.emit_phase1_observation(weights, tolerance=kw.get("tolerance"))
        return self.emit_phase2_orientation(
            obs, boundary_matrix, potential_vector,
            weights_0=kw.get("weights_0"),
            tolerance=kw.get("tolerance"),
        )

    def summary(self) -> str:
        """Resumen del último estado de gobernanza."""
        s = self._last_state
        if s is None:
            return "DiscreteHodgeStarAgent: sin ciclo OODA ejecutado."
        d = s.decision_stamp
        o = d.orientation_stamp
        lines = [
            "=" * 72,
            "HODGE SOVEREIGN STATE — CERTIFICADO OODA (v4.0.0)",
            "=" * 72,
            f"  Timestamp UTC     : {s.timestamp_utc}",
            f"  Stratum           : {s.stratum}",
            f"  Secure            : {s.is_secure}",
            f"  Crowbar           : {'ACTIVE gpio='+str(s.crowbar_gpio) if s.is_crowbar_active else 'idle'}",
            f"  Verdict Ω₃        : {d.verdict.name}",
            f"  Crowbar action    : {d.crowbar_action.name}",
            f"  Veto reasons      : {d.veto_reasons or '—'}",
            f"  dim(★₁)           : {o.observation_stamp.dimension}",
            f"  κ(★₁)             : {o.observation_stamp.condition_number:.4e}",
            f"  λ_min(★₁)         : {o.observation_stamp.min_eigenvalue:.4e}",
            f"  Tikhonov          : {o.observation_stamp.regularized_applied}",
            f"  E_★ (Dirichlet)   : {o.dirichlet_energy:.4e}",
            f"  Passiveive           : {o.is_passive}",
            f"  β₀ / β₁           : {o.betti_0} / {o.betti_1}",
            f"  Fiedler λ₂        : {o.algebraic_connectivity:.4e}",
            f"  κ(L₀)             : {o.laplacian_condition:.4e}",
            f"  ‖I_exact‖₂        : {d.exact_norm:.4e}",
            f"  ‖I_coexact‖₂      : {d.coexact_norm:.4e}",
            f"  ‖I_harmonic‖₂     : {d.harmonic_norm:.4e}",
            f"  Ω_vort (parasitic): {d.parasitic_vorticity:.4e}",
            f"  Ω_harm            : {d.harmonic_vorticity:.4e}",
            f"  Ω_total           : {d.total_vorticity:.4e}",
            f"  KCL residual      : {d.kcl_residual:.4e}",
            f"  Recon error       : {d.reconstruction_error:.4e}",
            f"  OODA hints        : {s.ooda_latency_hints}",
            "=" * 72,
        ]
        return "\n".join(lines)


# =============================================================================
# FACTORÍA Y EXPORTACIÓN CANÓNICA
# =============================================================================

def build_hodge_sovereign_agent(
    *,
    vorticity_threshold: float = 1.0,
    max_betti_1: int = 0,
    condition_max: float = _CONDITION_NUMBER_MAX,
    crowbar_actuator: Optional[CrowbarActuator] = None,
    dry_run: bool = False,
    raise_on_veto: bool = False,
) -> DiscreteHodgeStarAgent:
    """
    Factoría de alto nivel del Agente Soberano.

    Parameters
    ----------
    dry_run :
        Si True, inyecta ``NullCrowbarActuator`` (CI / simulación).
    """
    actuator: CrowbarActuator
    if crowbar_actuator is not None:
        actuator = crowbar_actuator
    elif dry_run:
        actuator = NullCrowbarActuator()
    else:
        actuator = LoggingCrowbarActuator()

    return DiscreteHodgeStarAgent(
        vorticity_threshold=vorticity_threshold,
        max_betti_1=max_betti_1,
        condition_max=condition_max,
        crowbar_actuator=actuator,
        raise_on_veto=raise_on_veto,
    )


__all__ = [
    # Excepciones
    "HodgeStarAgentError",
    "NonFiniteWeightError",
    "MetricDegeneracyError",
    "MetricDefeneracyError",  # alias v3
    "LaplacianPassivityError",
    "HelmholtzDecompositionError",
    "ParasiticVorticityVeto",
    "BettiNumberVeto",
    "CrowbarActivationError",
    # Ω₃ y Crowbar
    "HodgeSovereignVerdict",
    "CrowbarAction",
    "CrowbarActuator",
    "NullCrowbarActuator",
    "LoggingCrowbarActuator",
    # DTOs de continuidad funtorial
    "Phase1HodgeObservation",
    "Phase2HodgeOrientation",
    "Phase3HodgeDecision",
    "HodgeSovereignState",
    # Fases anidadas
    "Phase1_SpectralMetricObserver",
    "Phase2_HodgeLaplacianOrienter",
    "Phase3_HelmholtzDecisionMaker",
    # Soberano
    "DiscreteHodgeStarAgent",
    "build_hodge_sovereign_agent",
]


# =============================================================================
# Cierre formal de las tres fases anidadas del Agente:
#
#   Phase1_SpectralMetricObserver.emit_phase1_observation
#       → Phase1HodgeObservation
#           → Phase2_HodgeLaplacianOrienter.emit_phase2_orientation
#               → Phase2HodgeOrientation
#                   → Phase3_HelmholtzDecisionMaker.emit_phase3_decision
#                       → Phase3HodgeDecision
#                           → DiscreteHodgeStarAgent._act_crowbar
#                               → HodgeSovereignState
# =============================================================================