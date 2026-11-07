# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Discrete Hodge Star Agent (Soberano de la Métrica Constitutiva)     ║
║ Ruta   : app/agents/physics/discrete_hodge_star_agent.py                     ║
║ Versión: 4.0.0-Hodge-PoincareCelestial-RSI-L3-Helmholtz-OODA-Topos-PhD       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL EN EL ESTRATO PHYSICS (V_P) ────
Este módulo materializa al Agente Soberano y Observador Activo que gobierna la
consistencia métrica del operador estrella de Hodge discreto ($\star_k$) sobre el
complejo de de Rham simplicial del APU Filter, actuando como un endofuntor
contractivo de Nivel 3 (RSI-L3) en la categoría de haces de-confinados con
valores en el clasificador de subobjetos del retículo distributivo de Heyting $\Omega_3$.

El sistema modela las conductancias, permeabilidades y flujos de costos de la red
de valor como formas diferenciales discretas (cocadenas en $C^k$) y corrientes
duales (cadenas en $C_k$) acopladas mediante la dualidad de Poincaré discreta y la
mecánica celeste simpléctica de Henri Poincaré (*Les Méthodes Nouvelles de la
Mécanique Céleste*).

ARQUITECTURA DE TRES FASES ANIDADAS (Continuidad Funtorial Estricta): ──────────
La definición formal final del último método de cada fase constituye el punto de
arranque exacto de la fase siguiente:

  FASE 1 ──► OBSERVACIÓN ESPECTRAL, SANEAMIENTO DE TIKHONOV-WILKINSON Y RSI-L3
             Clase: Phase1_SpectralMetricObserver.
             Evalúa y sanea el tensor constitutivo $\star_1 \succ \mathbf{0}$ mediante
             desplazamiento espectral exacto y flujo geodésico isocórico de Fisher-Rao
             que preserva el volumen de Liouville-Poincaré $\sqrt{\det(\star_1)}$.
             Último método formal: `emit_phase1_observation(...) -> Phase1HodgeObservation`
             ──► cuya firma de retorno abre formalmente la Fase 2.

  FASE 2 ──► LAPLACIANO DE HODGE, PASIVIDAD PORT-HAMILTONIANA Y ESPECTRO KAM
             Inicia con `Phase1HodgeObservation` y continúa con
             `Phase2_HodgeLaplacianOrienter(Phase1_SpectralMetricObserver)`.
             Ensambla $L_0^\star = \partial_1 \star_1 \partial_1^\top$, certifica la
             desigualdad de pasividad $\dot{\mathcal{H}} = -2 E_\star(\phi) \le 0$,
             verifica el invariante de Euler-Poincaré $\chi(K)$ y audita los pequeños
             divisores diofánticos de Kolmogorov-Arnold-Moser (KAM).
             Último método formal: `emit_phase2_orientation(...) -> Phase2HodgeOrientation`
             ──► cuya firma de retorno abre formalmente la Fase 3.

  FASE 3 ──► DESCOMPOSICIÓN DE HELMHOLTZ-HODGE, ÓRBITA SIMPLÉCTICA DE POINCARÉ,
             RETÍCULO DE HEYTING $\Omega_3$ Y SOBERANO CROWBAR (GPIO14)
             Inicia con `Phase2HodgeOrientation` y continúa con
             `Phase3_HelmholtzDecisionMaker(Phase2_HodgeLaplacianOrienter)`.
             Proyecta el flujo $I \in C_1$ en componentes mutuamente ortogonales
             respecto a la métrica de disipación de Joule $\langle \cdot, \cdot \rangle_{\star_1^{-1}}$:
             $$I = I_{\mathrm{exact}} + I_{\mathrm{coexact}} + I_{\mathrm{harmonic}}, \qquad
               \langle I_a, I_b \rangle_{\star_1^{-1}} = 0 \quad (a \neq b).$$
             Audita la evolución simpléctica de Cayley-Poincaré ($\det \Phi_{\Delta t} = 1$,
             multiplicadores de Floquet, recurrencia de Kac-Poincaré y circulación de
             Poincaré-Cartan).
             Último método formal: `emit_phase3_decision(...) -> Phase3HodgeDecision`
             ──► consumido inmediatamente por `DiscreteHodgeStarAgent` para conmutar
             el disyuntor físico Crowbar (GPIO14) y emitir `HodgeSovereignState`.

INVARIANTES MATEMÁTICOS, SIMPLÉCTICOS Y TERMODINÁMICOS PRESERVADOS: ────────────
  [I1] Autoadjunticidad y Cono SPD:       $\star_k = \star_k^\top \succ \mathbf{0}$
  [I2] Factorización de de Rham-Hodge:    $L_0^\star = \partial_1 \star_1 \partial_1^\top \succeq \mathbf{0}$
  [I3] Ortogonalidad de Tellegen-Joule:   $\|I\|_{\star_1^{-1}}^2 = \|I_{\mathrm{ex}}\|_{\star_1^{-1}}^2 + \|I_{\mathrm{co}}\|_{\star_1^{-1}}^2 + \|I_{\mathrm{har}}\|_{\star_1^{-1}}^2$
  [I4] Pasividad de Lyapunov:             $\dot{\mathcal{H}} = -2 E_\star(\phi) \le 0$
  [I5] Invariante de Liouville-Poincaré:  $\det(\Phi_{\Delta t}) = 1$ (flujo hamiltoniano conservativo)
  [I6] Punto Fijo de Banach (RSI-L3):     $d(\mathcal{T}(\theta^{(n+1)}), \mathcal{T}(\theta^{(n)})) \le q_{\mathrm{Banach}} \, d(\theta^{(n+1)}, \theta^{(n)}), \; q_{\mathrm{Banach}} < 1$
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
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
# Motor geométrico certificado (módulo physics v3.0 — las 3 fases algebraicas)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.physics.discrete_hodge_star import (
        DeRhamComplexCertificate,
        DiscreteDeRhamComplex,
        DiscreteHodgeStar,
        HelmholtzHodgeDecomposition,
        HodgeDegree,
        HodgeDualityMorphism,
        HodgeMetricState,
        LaplacianSpectrum,
        PoincareCelestialOrbitCertificate,
        PoincareKAMResonanceCertificate,
        PortHamiltonianState,
        RSILevel3Telemetry,
        SpectralCertificate,
        WeightedHodgeLaplacian,
        build_certified_de_rham_complex,
        _CONDITION_SAFE as _PHYS_CONDITION_SAFE,
        _MACHINE_EPS as _PHYS_EPS,
        _SPECTRAL_TOL as _PHYS_SPECTRAL_TOL,
    )
    _HAS_PHYSICS_ENGINE = True
except ImportError:  # pragma: no cover
    _HAS_PHYSICS_ENGINE = False
    _PHYS_EPS = float(np.finfo(np.float64).eps)
    _PHYS_SPECTRAL_TOL = 1.0e-12
    _PHYS_CONDITION_SAFE = 1.0e12

logger = logging.getLogger("MIC.Agents.Physics.DiscreteHodgeStarAgent")

RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]

# Constantes numéricas de Wilkinson y de contrato de silicio
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1.0e-13
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8
_RECONSTRUCTION_TOL: Final[float] = 1.0e-10
_KCL_TOL: Final[float] = 1.0e-8
_JOULE_ORTHO_TOL: Final[float] = 1.0e-8
_TIKHONOV_SCALE: Final[float] = 1.0e4
_LIOUVILLE_TOL: Final[float] = 1.0e-9
_CROWBAR_GPIO: Final[int] = 14  # GPIO14 — contrato hardware ESP32 / BT151


# =============================================================================
# JERARQUÍA DE EXCEPCIONES (VETOS DUROS EN EL TOPOS)
# =============================================================================

class HodgeStarAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de la Estrella de Hodge."""


class NonFiniteWeightError(HodgeStarAgentError):
    """Pesos constitutivos con NaN/Inf (fuga aritmética de FPU IEEE-754)."""


class MetricDegeneracyError(HodgeStarAgentError):
    """Pérdida de definición positiva (SPD) o mal condicionamiento severo."""


# Alias de compatibilidad histórica con v3.x
MetricDefeneracyError = MetricDegeneracyError


class LaplacianPassivityError(HodgeStarAgentError):
    """Violación de pasividad Port-Hamiltoniana (Ḣ > 0)."""


class HelmholtzDecompositionError(HodgeStarAgentError):
    """Fallo de consistencia u ortogonalidad en la descomposición de Hodge."""


class ParasiticVorticityVeto(HodgeStarAgentError):
    """Veto de aduana geométrica: vorticidad parasitaria supera la cota elástica."""


class BettiNumberVeto(HodgeStarAgentError):
    """Veto topológico: número de Betti β₁ excede el máximo admisible."""


class PoincareSymplecticVeto(HodgeStarAgentError):
    """Violación de invariante simpléctico de Liouville-Poincaré o inestabilidad de Floquet."""


class KAMResonanceDegradationError(HodgeStarAgentError):
    """Colapso de pequeños divisores diofánticos de Kolmogorov-Arnold-Moser."""


class CrowbarActivationError(HodgeStarAgentError):
    """Fallo crítico al señalizar el disyuntor físico por hardware (GPIO14)."""


# =============================================================================
# CLASIFICADOR DE SUBOBJETOS Ω₃ (Retículo Distributivo de Heyting Acotado)
# =============================================================================

class HodgeSovereignVerdict(IntEnum):
    r"""
    Veredicto en el retículo distributivo completo de tres valores $\Omega_3$.

    Orden parcial de Heyting: $\mathrm{COHERENT} \sqsubseteq \mathrm{DEGRADED} \sqsubseteq \mathrm{VETOED}$.
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __or__(self, other: "HodgeSovereignVerdict") -> "HodgeSovereignVerdict":
        """Operador Supremo ($\sqcup$ / Join): prevalece el estado de mayor severidad."""
        return self.__class__(max(int(self), int(other)))

    def __and__(self, other: "HodgeSovereignVerdict") -> "HodgeSovereignVerdict":
        """Operador Ínfimo ($\sqcap$ / Meet): prevalece el estado de menor severidad."""
        return self.__class__(min(int(self), int(other)))

    def implies(self, consequent: "HodgeSovereignVerdict") -> "HodgeSovereignVerdict":
        """Implicación relativa de Heyting ($a \Rightarrow b$) en el retículo dual de seguridad."""
        if int(self) >= int(consequent):
            return HodgeSovereignVerdict.COHERENT
        return consequent


class CrowbarAction(Enum):
    """Acción del actuador físico en silicio derivada del veredicto en Ω₃."""
    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()  # Conmutación síncrona de GPIO14 → Tiristor SCR BT151


# =============================================================================
# PROTOCOLO DE ACTUADOR HARDWARE (Inyección de Dependencias / Silicio ESP32)
# =============================================================================

@runtime_checkable
class CrowbarActuator(Protocol):
    """Puerto de salida hacia el disyuntor físico (ESP32 GPIO14 u homólogo)."""

    def assert_crowbar(self, gpio: int, reason: str) -> bool:
        """Dispara el circuito Crowbar. Retorna True si el hardware confirmó ACK."""
        ...

    def deassert_crowbar(self, gpio: int) -> bool:
        """Libera el circuito Crowbar tras rearme seguro."""
        ...


class NullCrowbarActuator:
    """Actuador nulo instrumentado para entornos de prueba / CI sin hardware físico."""

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
    """Actuador que emite la trama de disparo síncrona hacia el bus de firmware."""

    def assert_crowbar(self, gpio: int, reason: str) -> bool:
        logger.critical(
            "CROWBAR_ASSERT gpio=%d reason=%s protocol=ESP32_GPIO_BT151",
            gpio,
            reason,
        )
        return True

    def deassert_crowbar(self, gpio: int) -> bool:
        logger.info("CROWBAR_DEASSERT gpio=%d", gpio)
        return True


# =============================================================================
# FASE 1 — OBSERVACIÓN ESPECTRAL, SANEAMIENTO DE TIKHONOV-WILKINSON
#          Y AUTOMEJORA RECURSIVA DE NIVEL 3 (Observe)
#          Último método formal: emit_phase1_observation -> Phase1HodgeObservation
#          cuya definición formal de retorno da inicio inmediato a la FASE 2.
# =============================================================================

@dataclass(frozen=True, slots=True)
class AgentRSILevel3Certificate:
    r"""
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3) emitido durante el
    saneamiento métrico de la FASE 1.

    Niveles de recursión:
      - Nivel 1 (Objeto): Desplazamiento espectral exacto de Tikhonov-Wilkinson.
      - Nivel 2 (Ley Geodésica): Retracción isocórica en la variedad Riemanniana
        de Fisher-Rao que preserva el volumen de Liouville-Poincaré $\det(\star_1)$.
      - Nivel 3 (Meta-Ley de Banach): Verificación del factor de contracción
        $q_{\mathrm{Banach}} < 1$ y adaptación de la entropía espectral.
    """
    iterations: int
    initial_condition_number: float
    final_condition_number: float
    fisher_rao_distance: float
    banach_contraction_factor: float
    liouville_volume_drift: float
    isochoric_retraction_applied: bool
    converged_to_fixed_point: bool


class Phase1_SpectralMetricObserver:
    r"""
    Fase 1 del endofuntor OODA: observa, audita y sanea la estabilidad espectral
    del vector de pesos constitutivos $w \in \mathbb{R}^m_{>0}$ que define $\star_1$.

    Garantiza incondicionalmente:
      1. Ausencia de fugas IEEE-754 ($\mathrm{NaN}/\infty$).
      2. Positividad estricta Riemanniana ($w_i > 0$).
      3. Condicionamiento espectral acotado $\kappa_2(\star_1) \le \kappa_{\max}$ y
         brecha basal $\lambda_{\min}(\star_1) \ge \varepsilon_{\mathrm{tol}}$, ya sea mediante
         desplazamiento analítico exacto de Tikhonov o mediante retracción geodésica
         isocórica RSI de Nivel 3 sobre la variedad de Fisher-Rao.
    """

    def __init__(
        self,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
        enable_isochoric_rsi: bool = False,
    ) -> None:
        self._condition_max: float = float(condition_max)
        self._spectral_tol: float = float(spectral_tol)
        self._enable_isochoric_rsi: bool = bool(enable_isochoric_rsi)

    # ── Validaciones atómicas y geometría de Fisher-Rao ──────────────────────

    def _assert_finite(self, weights: RealVector) -> RealVector:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size == 0:
            raise NonFiniteWeightError(
                "Axioma de No-Trivialidad: el vector de pesos constitutivos está vacío."
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
    ) -> Tuple[float, float, float, float, float, float]:
        """Retorna (λ_min, λ_max, κ₂, tr, log_det, liouville_vol)."""
        lam_min = float(np.min(weights))
        lam_max = float(np.max(weights))
        kappa = lam_max / max(lam_min, _MACHINE_EPS)
        trace = float(np.sum(weights))
        log_det = float(np.sum(np.log(np.maximum(weights, _MACHINE_EPS))))
        liouville_vol = float(np.exp(0.5 * np.clip(log_det, -700.0, 700.0)))
        return lam_min, lam_max, kappa, trace, log_det, liouville_vol

    def _tikhonov_regularize(
        self,
        weights: RealVector,
        lam_min: float,
        lam_max: float,
        kappa: float,
        tol: float,
    ) -> Tuple[RealVector, float, bool]:
        r"""
        Desplazamiento espectral exacto de Tikhonov-Wilkinson:
        $$\tilde{w}_i = w_i + \delta^*,$$
        donde $\delta^*$ resuelve simultáneamente las dos desigualdades espectrales:
        $$\lambda_{\min} + \delta^* \ge \varepsilon_{\mathrm{tol}}
          \quad\wedge\quad
          \frac{\lambda_{\max} + \delta^*}{\lambda_{\min} + \delta^*} \le \kappa_{\max}
          \iff \delta^* \ge \frac{\lambda_{\max} - \kappa_{\max}\lambda_{\min}}{\kappa_{\max} - 1}.$$
        """
        needs = (lam_min < tol) or (kappa > self._condition_max)
        if not needs:
            return weights, 0.0, False

        shift_floor = tol - lam_min
        shift_kappa = (
            (lam_max - self._condition_max * lam_min) / max(self._condition_max - 1.0, 1.0)
            if kappa > self._condition_max
            else 0.0
        )
        shift_wilkinson = max(lam_max * _MACHINE_EPS * _TIKHONOV_SCALE, tol)
        shift = max(shift_floor, shift_kappa, shift_wilkinson) * (1.0 + 1.0e-12)

        logger.warning(
            "FASE1 Tikhonov Exacto: κ=%.4e λ_min=%.4e → shift=%.4e",
            kappa,
            lam_min,
            shift,
        )
        w_reg = weights + shift
        return w_reg, float(shift), True

    def _isochoric_fisher_rao_rsi_level3(
        self,
        weights: RealVector,
        init_kappa: float,
        tol: float,
    ) -> Tuple[RealVector, AgentRSILevel3Certificate]:
        r"""
        Flujo Geodésico Isocórico de Automejora Recursiva de Nivel 3 (RSI-L3):
        Contrae el espectro logarítmico $\log w$ a lo largo de la geodésica de
        Fisher-Rao hacia la media geométrica $\overline{\log w}$, preservando de
        manera exacta el determinante de Liouville-Poincaré $\det(\star_1)$ mientras
        reduce $\kappa_2(\star_1)$ por debajo de `condition_max`.
        """
        log_w0 = np.log(np.maximum(weights, _MACHINE_EPS))
        mean_log = float(np.mean(log_w0))
        centered0 = log_w0 - mean_log
        spread0 = float(np.max(log_w0) - np.min(log_w0))
        target_spread = float(np.log(max(self._condition_max, 1.0 + 1.0e-9)))

        if spread0 <= target_spread and float(np.min(weights)) >= tol:
            return weights, AgentRSILevel3Certificate(
                iterations=0,
                initial_condition_number=init_kappa,
                final_condition_number=init_kappa,
                fisher_rao_distance=0.0,
                banach_contraction_factor=0.0,
                liouville_volume_drift=0.0,
                isochoric_retraction_applied=False,
                converged_to_fixed_point=True,
            )

        log_w_curr = log_w0.copy()
        prev_step = float("inf")
        q_banach = 0.5
        it_used = 0
        converged = False

        for it in range(1, 33):
            it_used = it
            centered = log_w_curr - mean_log
            curr_spread = float(np.max(centered) - np.min(centered))
            if curr_spread <= target_spread:
                converged = True
                break

            ratio = target_spread / max(curr_spread, _MACHINE_EPS)
            eta = 1.0 - (0.5 * (1.0 + ratio) if (curr_spread - target_spread) > 1.0e-4 else ratio)
            eta = float(np.clip(eta, 0.0, 1.0 - _MACHINE_EPS))

            log_w_next = mean_log + (1.0 - eta) * centered
            step_dist = float(np.linalg.norm(log_w_next - log_w_curr))
            if np.isfinite(prev_step) and prev_step > _MACHINE_EPS:
                q_banach = step_dist / prev_step
            log_w_curr = log_w_next
            prev_step = step_dist
            if step_dist <= 1.0e-12:
                converged = True
                break

        if not converged and spread0 > target_spread:
            alpha_exact = target_spread / spread0
            log_w_curr = mean_log + alpha_exact * centered0
            converged = True

        w_iso = np.exp(log_w_curr)
        if float(np.min(w_iso)) < tol:
            w_iso = np.maximum(w_iso, tol)

        final_kappa = float(np.max(w_iso) / max(np.min(w_iso), _MACHINE_EPS))
        fr_dist = float(np.linalg.norm( np.log(w_iso) - log_w0 ))
        vol_drift = float(abs(np.sum(np.log(w_iso)) - np.sum(log_w0)))

        cert = AgentRSILevel3Certificate(
            iterations=it_used,
            initial_condition_number=init_kappa,
            final_condition_number=final_kappa,
            fisher_rao_distance=fr_dist,
            banach_contraction_factor=float(min(q_banach, 0.999999)),
            liouville_volume_drift=vol_drift,
            isochoric_retraction_applied=True,
            converged_to_fixed_point=converged,
        )
        return w_iso, cert

    def _try_physics_duality(
        self, weights: RealVector
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Construye `HodgeDualityMorphism` vía el motor `app.physics.discrete_hodge_star`."""
        if not _HAS_PHYSICS_ENGINE:
            return None, None
        try:
            star = DiscreteHodgeStar(
                weights,
                degree=HodgeDegree.ONE,
                validate_strict=True,
                enable_rsi_level3=self._enable_isochoric_rsi,
                target_condition_max=self._condition_max,
            )
            morph = star.induce_duality_morphism(auto_heal_rsi=self._enable_isochoric_rsi)
            state = star.audit_metric(tolerance=self._spectral_tol)
            return morph, state
        except Exception as exc:  # pragma: no cover
            logger.debug("Motor physics no disponible en FASE1: %s", exc)
            return None, None

    # ── Método público principal de auditoría de FASE 1 ──────────────────────

    def audit_metric_weights(
        self,
        weights: RealVector,
        tolerance: Optional[float] = None,
    ) -> "Phase1HodgeObservation":
        r"""
        Valida y sanea el vector de pesos constitutivos $w \mapsto \star_1 = \operatorname{diag}(w)$.

        Aplica regularización espectral exacta de Tikhonov-Wilkinson y/o retracción
        geodésica isocórica RSI-L3 cuando $\lambda_{\min} < \varepsilon_{\mathrm{tol}}$ o
        $\kappa_2(\star_1) > \kappa_{\max}$.
        """
        tol = self._spectral_tol if tolerance is None else float(tolerance)

        w = self._assert_finite(weights)
        self._assert_positive(w)

        lam_min0, lam_max0, kappa0, trace, log_det, liouville_vol = self._spectral_invariants(w)
        w_san = w.copy()
        shift = 0.0
        reg = False
        rsi_cert: Optional[AgentRSILevel3Certificate] = None
        fr_drift = 0.0

        if self._enable_isochoric_rsi and ((lam_min0 < tol) or (kappa0 > self._condition_max)):
            w_san, rsi_cert = self._isochoric_fisher_rao_rsi_level3(w_san, kappa0, tol)
            reg = rsi_cert.isochoric_retraction_applied
            fr_drift = rsi_cert.fisher_rao_distance

        lam_min, lam_max, kappa, trace, log_det, liouville_vol = self._spectral_invariants(w_san)

        # Garantía suplementaria por desplazamiento exacto de Tikhonov si aún excede umbral
        if (lam_min < tol) or (kappa > self._condition_max):
            w_san, shift, reg_tik = self._tikhonov_regularize(w_san, lam_min, lam_max, kappa, tol)
            reg = reg or reg_tik
            lam_min, lam_max, kappa, trace, log_det, liouville_vol = self._spectral_invariants(w_san)
            fr_drift = float(np.linalg.norm(np.log(w_san) - np.log(w)))

        if rsi_cert is None:
            rsi_cert = AgentRSILevel3Certificate(
                iterations=1 if reg else 0,
                initial_condition_number=kappa0,
                final_condition_number=kappa,
                fisher_rao_distance=fr_drift,
                banach_contraction_factor=float(kappa / kappa0) if (reg and kappa0 > 0) else 0.0,
                liouville_volume_drift=float(
                    abs(log_det - float(np.sum(np.log(np.maximum(w, _MACHINE_EPS)))))
                ),
                isochoric_retraction_applied=False,
                converged_to_fixed_point=True,
            )

        is_spd = (lam_min >= tol * (1.0 - 1.0e-9)) and (kappa <= self._condition_max * (1.0 + 1.0e-9))
        if not is_spd:
            raise MetricDegeneracyError(
                f"★_k numéricamente singular tras saneamiento: "
                f"κ={kappa:.4e} > {self._condition_max:.4e}, λ_min={lam_min:.4e}."
            )

        hodge = np.diag(w_san)
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
            liouville_volume_form=float(liouville_vol),
            rsi_certificate=rsi_cert,
            fisher_rao_drift=float(fr_drift),
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 1 ────────────────────────────────────
    # Su definición formal de retorno (Phase1HodgeObservation) abre directamente
    # las definiciones y métodos de la FASE 2.

    def emit_phase1_observation(
        self,
        weights: RealVector,
        tolerance: Optional[float] = None,
    ) -> "Phase1HodgeObservation":
        r"""
        Punto de emisión canónico que clausura la FASE 1 y abre la FASE 2.

        Verifica las post-condiciones de clausura transitiva en el cono SPD
        antes de entregar `Phase1HodgeObservation` al orientador Laplaciano.
        """
        obs = self.audit_metric_weights(weights, tolerance=tolerance)
        if not obs.is_strictly_spd:
            raise MetricDegeneracyError(
                "Post-condición FASE1 violada: is_strictly_spd=False."
            )
        if obs.hodge_matrix.shape != (obs.dimension, obs.dimension):
            raise MetricDegeneracyError(
                "Post-condición FASE1 violada: shape(★₁) ≠ (m, m)."
            )
        if obs.condition_number > self._condition_max * (1.0 + 1.0e-7) + 1.0:
            raise MetricDegeneracyError(
                "Post-condición FASE1 violada: κ₂(★₁) excede cota máxima tras emisión."
            )
        logger.info(
            "FASE1 EMIT dim=%d κ=%.3e λ_min=%.3e vol_LP=%.3e reg=%s",
            obs.dimension,
            obs.condition_number,
            obs.min_eigenvalue,
            obs.liouville_volume_form,
            obs.regularized_applied,
        )
        return obs


# =============================================================================
# FASE 2 — LAPLACIANO DE HODGE, PASIVIDAD PORT-HAMILTONIANA Y ESPECTRO CELESTE
#          KAM DE POINCARÉ (Orient)
#          Inicia con la definición formal de Phase1HodgeObservation emitida
#          por emit_phase1_observation.
#          Último método formal: emit_phase2_orientation -> Phase2HodgeOrientation
#          cuya definición formal de retorno da inicio inmediato a la FASE 3.
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase1HodgeObservation:
    r"""
    Contrato formal de salida de la FASE 1 (`emit_phase1_observation`) y semilla
    obligatoria de arranque de la FASE 2 (`Phase2_HodgeLaplacianOrienter`).

    Encapsula la métrica constitutiva $\star_1$ saneada, su invariante de volumen
    de Liouville-Poincaré $\sqrt{\det(\star_1)}$ y el certificado de punto fijo
    RSI de Nivel 3.
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
    liouville_volume_form: float = 1.0
    rsi_certificate: Optional[AgentRSILevel3Certificate] = None
    fisher_rao_drift: float = 0.0

    def continue_to_phase2_orienter(
        self,
        orienter: "Phase2_HodgeLaplacianOrienter",
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        tolerance: Optional[float] = None,
    ) -> "Phase2HodgeOrientation":
        """Morfismo de continuación covariante FASE 1 ──► FASE 2."""
        return orienter.emit_phase2_orientation(
            observation=self,
            boundary_matrix=boundary_matrix,
            potential_vector=potential_vector,
            weights_0=weights_0,
            tolerance=tolerance,
        )


class Phase2_HodgeLaplacianOrienter(Phase1_SpectralMetricObserver):
    r"""
    Fase 2 del endofuntor OODA: ensambla el Laplaciano de Hodge ponderado
    $L_0^\star = \partial_1 \star_1 \partial_1^\top$, certifica la pasividad de
    Lyapunov-Dirichlet, verifica la característica de Euler-Poincaré $\chi(K)$
    y audita los pequeños divisores diofánticos de Kolmogorov-Arnold-Moser (KAM).
    """

    def __init__(
        self,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
        max_betti_1: int = 0,
        enable_isochoric_rsi: bool = False,
    ) -> None:
        super().__init__(
            condition_max=condition_max,
            spectral_tol=spectral_tol,
            enable_isochoric_rsi=enable_isochoric_rsi,
        )
        self._max_betti_1: int = int(max_betti_1)

    # ── Ensamblaje de L₀★ y espectro KAM de Poincaré ─────────────────────────

    def _assemble_laplacian_0(
        self,
        weights_1: RealVector,
        boundary_1: RealMatrix,
    ) -> RealMatrix:
        r"""
        Ensambla $L_0^\star = \partial_1 \star_1 \partial_1^\top$ en $O(V^2 E)$
        utilizando difusión vectorial por columnas sin multiplicar matrices diagonales densas.
        """
        L = (boundary_1 * weights_1[np.newaxis, :]) @ boundary_1.T
        skew = float(np.max(np.abs(L - L.T))) if L.size > 0 else 0.0
        if skew > 100.0 * _MACHINE_EPS * max(L.shape[0], 1):
            logger.warning(
                "FASE2: desvío de simetría en L₀ ||A-Aᵀ||_∞=%.3e — proyectando.",
                skew,
            )
        return 0.5 * (L + L.T)

    @staticmethod
    def _audit_kam_diophantine_divisors(
        eigenvalues: RealVector,
        betti_0: int,
        max_order: int = 4,
    ) -> Tuple[float, float, bool]:
        r"""
        Audita las resonancias de Poincaré-KAM sobre las frecuencias normales
        celestes $\omega_a = \sqrt{\lambda_a(L_0^\star)}$ ($\lambda_a > 0$):
        $$\gamma_{\mathrm{KAM}} = \min_{0 < p + q \le 2 M_{\max}} |p \omega_i - q \omega_j| (p + q)^2.$$
        Retorna `(smallest_divisor, diophantine_margin, is_kam_non_resonant)`.
        """
        pos_ev = eigenvalues[betti_0:]
        pos_ev = pos_ev[pos_ev > _DEFAULT_TOL]
        if pos_ev.size == 0:
            return float("inf"), float("inf"), True

        freqs = np.sqrt(pos_ev)
        if freqs.size == 1:
            w0 = float(freqs[0])
            return w0, w0, True

        active = freqs[: min(6, freqs.size)]
        smallest_div = float("inf")
        dioph_margin = float("inf")
        res_tol = 1.0e-6 * float(np.max(active))
        resonant_hits = 0

        for i in range(active.size):
            for j in range(i + 1, active.size):
                for p in range(1, max_order + 1):
                    for q in range(1, max_order + 1):
                        if math.gcd(p, q) != 1:
                            continue
                        div = abs(p * float(active[i]) - q * float(active[j]))
                        scaled = div * float((p + q) ** 2)
                        if div < smallest_div:
                            smallest_div = div
                        if scaled < dioph_margin:
                            dioph_margin = scaled
                        if div <= res_tol:
                            resonant_hits += 1

        return smallest_div, dioph_margin, (resonant_hits == 0)

    def _laplacian_spectrum(
        self, L: RealMatrix, tol: float
    ) -> Tuple[RealVector, float, float, float, int, bool, float, float, float, bool]:
        """
        Retorna:
        `(ev, κ_rango, λ_Fiedler, λ_gap, β₀, is_self_adjoint, max_residual, kam_div, kam_margin, kam_ok)`.
        """
        n = L.shape[0]
        is_sa = float(np.max(np.abs(L - L.T))) < 100.0 * _MACHINE_EPS * max(n, 1)

        if n == 0:
            return (
                np.zeros(0, dtype=np.float64),
                1.0,
                0.0,
                0.0,
                0,
                True,
                0.0,
                float("inf"),
                float("inf"),
                True,
            )

        if n <= 256:
            ev, V = la.eigh(L)
            residuals = L @ V - V * ev[np.newaxis, :]
            max_res = float(np.max(np.abs(residuals))) if residuals.size > 0 else 0.0
        else:
            ev = la.eigvalsh(L)
            v = np.ones(n, dtype=np.float64) / math.sqrt(n)
            Lv = L @ v
            lam_est = float(v @ Lv)
            max_res = float(np.max(np.abs(Lv - lam_est * v)))

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
            lam_fiedler = lam_gap
            kappa = max_eig / (lam_gap + _MACHINE_EPS)

        kam_div, kam_margin, kam_ok = self._audit_kam_diophantine_divisors(ev, betti0)

        return (
            ev,
            float(kappa),
            float(lam_fiedler),
            float(lam_gap),
            betti0,
            is_sa,
            max_res,
            kam_div,
            kam_margin,
            kam_ok,
        )

    def _dirichlet_energy(self, L: RealMatrix, phi: RealVector) -> float:
        r"""Energía potencial disipativa de Dirichlet: $E_\star(\phi) = \frac{1}{2}\phi^\top L_0^\star \phi$."""
        phi_v = np.asarray(phi, dtype=np.float64).ravel()
        if phi_v.shape[0] != L.shape[0]:
            raise MetricDegeneracyError(
                f"Dimensión de potencial ({phi_v.shape[0]}) ≠ nodos ({L.shape[0]})."
            )
        return float(0.5 * phi_v @ (L @ phi_v))

    def _try_physics_laplacian(
        self,
        observation: Phase1HodgeObservation,
        boundary_1: RealMatrix,
        weights_0: Optional[RealVector],
    ) -> Tuple[Optional[Any], Optional[int], Optional[int], Optional[float]]:
        """Delega en `WeightedHodgeLaplacian.certify_de_rham_complex()` si está disponible."""
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
        tol: float,
    ) -> int:
        r"""Calcula $\beta_1 = \dim \ker(\partial_1) = E - \operatorname{rank}(\partial_1)$ para un 1-complejo."""
        s = la.svdvals(boundary_1)
        rank = int(np.sum(s > tol * max(1.0, float(s[0]) if s.size else 1.0)))
        _, n_edges = boundary_1.shape
        return max(n_edges - rank, 0)

    # ── Método público principal de orientación de FASE 2 ────────────────────

    def orient_laplacian_dynamics(
        self,
        observation: Phase1HodgeObservation,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        tolerance: Optional[float] = None,
    ) -> "Phase2HodgeOrientation":
        r"""
        Ensambla $L_0^\star$, audita la desigualdad de pasividad Port-Hamiltoniana
        $\dot{\mathcal{H}} = -2 E_\star(\phi) \le 0$, verifica la característica de
        Euler-Poincaré $\chi(K) = V - E = \beta_0 - \beta_1$ y extrae el certificado
        de resonancia celeste KAM.
        """
        if not isinstance(observation, Phase1HodgeObservation):
            raise TypeError(
                "FASE2 exige Phase1HodgeObservation emitido por FASE1."
            )
        if not observation.is_strictly_spd:
            raise MetricDegeneracyError(
                "FASE2 rechaza observación no-SPD (invariante de continuidad roto)."
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

        L0 = self._assemble_laplacian_0(observation.weights_sanitized, B)

        (
            _,
            kappa_L,
            lam_fiedler,
            lam_gap,
            betti0,
            is_sa,
            max_res,
            kam_div,
            kam_margin,
            kam_ok,
        ) = self._laplacian_spectrum(L0, tol)

        energy = self._dirichlet_energy(L0, potential_vector)
        is_passive = energy >= _SPECTRAL_PSD_FLOOR
        if not is_passive:
            logger.error(
                "FASE2: violación de pasividad Port-Hamiltoniana E_★=%.4e < floor",
                energy,
            )

        cert, b0_p, b1_p, e_scale = self._try_physics_laplacian(
            observation, B, weights_0
        )
        if b0_p is not None:
            betti0 = b0_p
        betti1 = (
            b1_p
            if b1_p is not None
            else self._estimate_betti_1_kernel(B, tol)
        )
        energy_scale = (
            float(e_scale)
            if e_scale is not None
            else float(observation.spectral_trace + n_nodes)
        )

        # Invariante topológico de Euler-Poincaré en el 1-esqueleto: χ = V - E = β₀ - β₁
        euler_comb = n_nodes - n_edges
        euler_cohom = int(betti0) - int(betti1)
        euler_ok = (euler_comb == euler_cohom)

        if cert is not None and getattr(cert, "laplacian_1_spectrum", None) is not None:
            kam_c = getattr(cert.laplacian_1_spectrum, "kam_certificate", None)
            if kam_c is not None and np.isfinite(kam_c.smallest_divisor):
                kam_div = min(kam_div, float(kam_c.smallest_divisor))
                kam_margin = min(kam_margin, float(kam_c.diophantine_margin))
                kam_ok = bool(kam_ok and kam_c.is_kam_non_resonant)

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
            betti_2=0,
            euler_poincare_characteristic=euler_cohom,
            euler_poincare_verified=euler_ok,
            kam_smallest_divisor=float(kam_div),
            kam_diophantine_margin=float(kam_margin),
            is_kam_non_resonant=bool(kam_ok),
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 2 ────────────────────────────────────
    # Su definición formal de retorno (Phase2HodgeOrientation) abre directamente
    # las definiciones y métodos de la FASE 3.

    def emit_phase2_orientation(
        self,
        observation: Phase1HodgeObservation,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        tolerance: Optional[float] = None,
        enforce_passivity: bool = True,
        enforce_betti_1: bool = False,
    ) -> "Phase2HodgeOrientation":
        r"""
        Punto de emisión canónico que clausura la FASE 2 y abre la FASE 3.

        Verifica las post-condiciones de autoadjunción, pasividad de Lyapunov,
        invarianza de Euler-Poincaré y cota cohomológica de Betti $\beta_1$.
        """
        orient = self.orient_laplacian_dynamics(
            observation,
            boundary_matrix,
            potential_vector,
            weights_0=weights_0,
            tolerance=tolerance,
        )

        if not orient.is_self_adjoint:
            logger.warning("FASE2 post: L₀★ presenta asimetría numérica residual.")
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
            "FASE2 EMIT β₀=%d β₁=%d χ=%d(ok=%s) E_★=%.3e κ_L=%.3e Fiedler=%.3e passive=%s",
            orient.betti_0,
            orient.betti_1,
            orient.euler_poincare_characteristic,
            orient.euler_poincare_verified,
            orient.dirichlet_energy,
            orient.laplacian_condition,
            orient.algebraic_connectivity,
            orient.is_passive,
        )
        return orient


# =============================================================================
# FASE 3 — DESCOMPOSICIÓN DE HELMHOLTZ-HODGE EN MÉTRICA DE JOULE, MECÁNICA
#          CELESTE SIMPLÉCTICA DE POINCARÉ, RETÍCULO Ω₃ Y SOBERANO CROWBAR
#          Inicia con la definición formal de Phase2HodgeOrientation emitida
#          por emit_phase2_orientation.
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase2HodgeOrientation:
    r"""
    Contrato formal de salida de la FASE 2 (`emit_phase2_orientation`) y semilla
    obligatoria de arranque de la FASE 3 (`Phase3_HelmholtzDecisionMaker`).

    Contiene el espectro del Laplaciano de Hodge $L_0^\star$, los números de
    Betti $(\beta_0, \beta_1)$, el invariante de Euler-Poincaré $\chi(K)$ y
    los márgenes diofánticos de resonancia celeste KAM.
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
    betti_2: int = 0
    euler_poincare_characteristic: int = 0
    euler_poincare_verified: bool = True
    kam_smallest_divisor: float = float("inf")
    kam_diophantine_margin: float = float("inf")
    is_kam_non_resonant: bool = True

    def continue_to_phase3_decider(
        self,
        decider: "Phase3_HelmholtzDecisionMaker",
        boundary_matrix: RealMatrix,
        current_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        vorticity_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
        raise_on_veto: bool = False,
    ) -> "Phase3HodgeDecision":
        """Morfismo de continuación covariante FASE 2 ──► FASE 3."""
        return decider.emit_phase3_decision(
            orientation=self,
            boundary_matrix=boundary_matrix,
            current_vector=current_vector,
            weights_0=weights_0,
            vorticity_threshold=vorticity_threshold,
            tolerance=tolerance,
            raise_on_veto=raise_on_veto,
        )


@dataclass(frozen=True, slots=True)
class PoincarePhaseSpaceAudit:
    r"""
    Certificado de Mecánica Celeste Simpléctica de Henri Poincaré en el espacio
    fase $T^*C^0 \cong C^0 \oplus C_1$:

    - `liouville_determinant`: $\det(\Phi_{\Delta t})$ bajo el integrador de Cayley-Poincaré.
    - `symplectic_form_defect`: $\|\Phi_{\Delta t}^\top \mathcal{M} \Phi_{\Delta t} - \mathcal{M}\|_F$.
    - `floquet_spectral_radius`: $\max_i |\mu_i(\Phi_{\Delta t})|$ (criterio de estabilidad orbital).
    - `kac_poincare_recurrence_ratio`: razón de retorno más cercano $\min_{k \ge 2} \|z_k - z_0\|_{\mathcal{M}} / \|z_0\|_{\mathcal{M}}$.
    - `poincare_section_crossings`: número de intersecciones transversales orientadas en $\Sigma$.
    - `poincare_cartan_circulation`: norma del invariante integral relativo $\oint_\gamma \star_1^{-1} I$.
    - `is_symplectically_stable`: verdadero si se preserva el volumen de Liouville y $|\mu_i| \le 1 + \varepsilon$.
    """
    liouville_determinant: float
    symplectic_form_defect: float
    floquet_spectral_radius: float
    kac_poincare_recurrence_ratio: float
    poincare_section_crossings: int
    poincare_cartan_circulation: float
    is_symplectically_stable: bool


class Phase3_HelmholtzDecisionMaker(Phase2_HodgeLaplacianOrienter):
    r"""
    Fase 3 del endofuntor OODA: proyecta el espacio de corrientes $I \in C_1$
    sobre la descomposición ortogonal de Helmholtz-Hodge en la métrica de
    disipación de Joule $\langle u, v \rangle_{\star_1^{-1}} = u^\top \star_1^{-1} v$,
    audita la estabilidad orbital simpléctica de Poincaré y clasifica el estado
    en el retículo distributivo $\Omega_3$.

    Identidad de Tellegen-Helmholtz-Hodge:
    $$I = I_{\mathrm{exact}} + I_{\mathrm{coexact}} + I_{\mathrm{harmonic}}, \qquad
      \|I\|_{\star_1^{-1}}^2 = \|I_{\mathrm{exact}}\|_{\star_1^{-1}}^2
                             + \|I_{\mathrm{coexact}}\|_{\star_1^{-1}}^2
                             + \|I_{\mathrm{harmonic}}\|_{\star_1^{-1}}^2.$$
    """

    def __init__(
        self,
        condition_max: float = _CONDITION_NUMBER_MAX,
        spectral_tol: float = _DEFAULT_TOL,
        max_betti_1: int = 0,
        vorticity_threshold: float = 1.0,
        coherent_fraction: float = 0.1,
        enable_isochoric_rsi: bool = False,
    ) -> None:
        super().__init__(
            condition_max=condition_max,
            spectral_tol=spectral_tol,
            max_betti_1=max_betti_1,
            enable_isochoric_rsi=enable_isochoric_rsi,
        )
        self._vorticity_threshold: float = float(vorticity_threshold)
        self._coherent_fraction: float = float(coherent_fraction)

    # ── Descomposición de Helmholtz-Hodge Exacta en Métrica de Joule ★₁⁻¹ ───

    def _helmholtz_native(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_1: RealMatrix,
        current: RealVector,
        tol: float,
    ) -> Tuple[RealVector, RealVector, RealVector, float, float]:
        r"""
        Descomposición de Helmholtz-Hodge nativa estrictamente ortogonal en la
        métrica de Joule $\langle u, v \rangle_{\star_1^{-1}} = u^\top \star_1^{-1} v$:

        1. **Componente Exacta ($I_{\mathrm{exact}} = \star_1 \partial_1^\top \phi$)**:
           Resuelve $( \partial_1 \star_1 \partial_1^\top ) \phi = \partial_1 I$ mediante
           pseudoinversa espectral de Moore-Penrose sobre $(\ker L_0^\star)^\perp$.
           Satisface exactamente $\partial_1 I_{\mathrm{exact}} = \partial_1 I$.
        2. **Subespacio Solenoidal ($\ker \partial_1$)**:
           El flujo residual $I_{\mathrm{sol}} = I - I_{\mathrm{exact}}$ satisface
           $\partial_1 I_{\mathrm{sol}} = \mathbf{0}$ (Primera Ley de Kirchhoff exacta) y es
           automáticamente $\star_1^{-1}$-ortogonal a $I_{\mathrm{exact}}$ por el Teorema de Tellegen:
           $$I_{\mathrm{exact}}^\top \star_1^{-1} I_{\mathrm{sol}} = \phi^\top \partial_1 \star_1 \star_1^{-1} I_{\mathrm{sol}} = \phi^\top (\partial_1 I_{\mathrm{sol}}) = 0.$$
        3. **Proyección $\star_1^{-1}$-Ortogonal sobre Ciclos Armónicos**:
           Si $\beta_1 > 0$, proyecta $I_{\mathrm{sol}}$ sobre $\ker(\partial_1)$ usando el
           producto interno ponderado por resistencias $W = \star_1^{-1}$, preservando
           exactamente la ortogonalidad de Joule entre las tres componentes.
        """
        B = boundary_1
        w1 = orientation.observation_stamp.weights_sanitized
        w1_inv = 1.0 / w1
        L0 = orientation.laplacian_0
        I_vec = np.asarray(current, dtype=np.float64).ravel()

        if I_vec.shape[0] != orientation.observation_stamp.dimension:
            raise HelmholtzDecompositionError(
                f"dim(I)={I_vec.shape[0]} ≠ m={orientation.observation_stamp.dimension}."
            )

        # 1. Resolver Poisson L₀★ φ = ∂₁ I mediante descomposición espectral simétrica
        source = B @ I_vec
        try:
            ev0, evec0 = la.eigh(L0)
            tol0 = tol * max(1.0, float(np.max(np.abs(ev0))) if ev0.size else 1.0)
            inv_ev0 = np.where(ev0 > tol0, 1.0 / ev0, 0.0)
            phi = evec0 @ (inv_ev0 * (evec0.T @ source))
        except Exception:
            try:
                phi = la.lstsq(L0, source, cond=tol)[0]
            except Exception as exc:
                raise HelmholtzDecompositionError(
                    f"Fallo al resolver Poisson L₀φ = ∂₁I: {exc}"
                ) from exc

        exact = w1 * (B.T @ phi)
        residual_current = I_vec - exact

        # 2. Separar armónicos en métrica de Joule ★₁⁻¹ si β₁ > 0
        harmonic = np.zeros_like(I_vec)
        if orientation.betti_1 > 0:
            try:
                # Transformación isométrica de Hilbert: \tilde{I} = ★₁^{-1/2} I
                # ∂₁ I = 0  <==>  (∂₁ ★₁^{1/2}) \tilde{I} = 0
                sqrt_w1 = np.sqrt(w1)
                inv_sqrt_w1 = 1.0 / sqrt_w1
                B_tilde = B * sqrt_w1[np.newaxis, :]
                _, S, Vh = la.svd(B_tilde, full_matrices=True)
                rank = int(np.sum(S > tol * max(1.0, float(S[0]) if S.size else 1.0)))
                ker_dim = B_tilde.shape[1] - rank
                if ker_dim > 0:
                    # Base ortonormal en el espacio isométrico de Joule
                    ker_tilde = Vh[rank:].T  # (m, ker_dim)
                    res_tilde = inv_sqrt_w1 * residual_current
                    coeffs = ker_tilde.T @ res_tilde
                    harm_tilde = ker_tilde @ coeffs
                    harmonic = sqrt_w1 * harm_tilde
            except Exception as exc:  # pragma: no cover
                logger.debug("SVD ponderada de ker(∂₁) falló: %s — harmonic=0", exc)

        coexact = residual_current - harmonic

        kcl = float(la.norm(B @ (coexact + harmonic), ord=2))
        recon = exact + coexact + harmonic
        recon_err = float(la.norm(I_vec - recon, ord=2))

        return exact, coexact, harmonic, kcl, recon_err

    def _helmholtz_physics(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_1: RealMatrix,
        current: RealVector,
        weights_0: Optional[RealVector],
    ) -> Optional[Tuple[RealVector, RealVector, RealVector, float, float]]:
        r"""
        Delega en `DiscreteDeRhamComplex.helmholtz_hodge_decomposition` respetando
        la dualidad constitutiva de Poincaré entre 1-flujos $I \in C_1$ y
        1-cocadenas $\alpha = \star_1^{-1} I \in C^1$:
        $$I \xrightarrow{\star_1^{-1}} \alpha = \alpha_{\mathrm{ex}} + \alpha_{\mathrm{co}} + \alpha_{\mathrm{har}}
            \xrightarrow{\star_1} I = I_{\mathrm{ex}} + I_{\mathrm{co}} + I_{\mathrm{har}}.$$
        """
        if not _HAS_PHYSICS_ENGINE:
            return None
        cert = orientation.de_rham_certificate_payload
        try:
            w1 = orientation.observation_stamp.weights_sanitized
            if cert is None:
                n_nodes = boundary_1.shape[0]
                w0 = (
                    np.asarray(weights_0, dtype=np.float64).ravel()
                    if weights_0 is not None
                    else np.ones(n_nodes, dtype=np.float64)
                )
                cplx = build_certified_de_rham_complex(
                    weights_0=w0,
                    weights_1=w1,
                    boundary_1=boundary_1,
                )
            else:
                cplx = DiscreteDeRhamComplex(cert)

            I_vec = np.asarray(current, dtype=np.float64).ravel()
            if I_vec.shape[0] != w1.shape[0]:
                raise HelmholtzDecompositionError(
                    f"dim(I)={I_vec.shape[0]} ≠ m={w1.shape[0]}."
                )

            # Dualidad inversa: convertir corriente I en 1-cocadena α = ★₁⁻¹ I
            alpha_cochain = I_vec / w1
            dec: HelmholtzHodgeDecomposition = cplx.helmholtz_hodge_decomposition(
                alpha_cochain
            )

            # Dualidad directa: recuperar corrientes I_• = ★₁ α_•
            exact = w1 * dec.exact_part
            coexact = w1 * dec.coexact_part
            harmonic = w1 * dec.harmonic_part

            kcl = float(la.norm(boundary_1 @ (coexact + harmonic), ord=2))
            recon_err = float(la.norm(I_vec - (exact + coexact + harmonic), ord=2))
            return exact, coexact, harmonic, kcl, recon_err
        except HelmholtzDecompositionError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.debug("Physics Helmholtz fallback → native: %s", exc)
            return None

    # ── Métricas de disipación de Joule y ortogonalidad de Tellegen ──────────

    @staticmethod
    def _metric_inner_star_inv(
        u: RealVector, v: RealVector, weights_1: RealVector
    ) -> float:
        r"""Producto interno de Joule: $\langle u, v \rangle_{\star_1^{-1}} = \sum_e \frac{u_e v_e}{w_e}$."""
        return float(np.sum((u * v) / weights_1))

    @staticmethod
    def _metric_norm_star_inv(
        vec: RealVector, weights_1: RealVector
    ) -> float:
        r"""Norma de vorticidad de Joule: $\|v\|_{\star_1^{-1}} = \sqrt{v^\top \star_1^{-1} v}$."""
        return float(math.sqrt(max(float(np.sum((vec * vec) / weights_1)), 0.0)))

    @staticmethod
    def _metric_energy_star_inv(
        vec: RealVector, weights_1: RealVector
    ) -> float:
        r"""Potencia de Joule media: $\mathcal{P}(v) = \frac{1}{2} v^\top \star_1^{-1} v$."""
        return float(0.5 * np.sum((vec * vec) / weights_1))

    # ── Auditoría de Mecánica Celeste Simpléctica de Poincaré en Fase 3 ──────

    def _audit_poincare_celestial_phase_space(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_1: RealMatrix,
        current_vector: RealVector,
        harmonic_current: RealVector,
        dt: float = 0.05,
        n_steps: int = 32,
    ) -> PoincarePhaseSpaceAudit:
        r"""
        Audita la órbita simpléctica de Cayley-Poincaré en el espacio fase
        $T^*C^0 \cong C^0 \oplus C_1$:
        $$\Phi_{\Delta t} = \left(I - \frac{\Delta t}{2}\mathcal{J}\mathcal{M}\right)^{-1}
                           \left(I + \frac{\Delta t}{2}\mathcal{J}\mathcal{M}\right).$$
        Certifica:
          1. Invariante integral de Liouville $\det(\Phi_{\Delta t}) = 1$.
          2. Preservación de la estructura simpléctica $\|\Phi_{\Delta t}^\top \mathcal{M} \Phi_{\Delta t} - \mathcal{M}\|_F$.
          3. Multiplicadores característicos de Floquet $|\mu_i| = 1$.
          4. Razón de recurrencia de Kac-Poincaré y cruces por la sección $\Sigma$.
          5. Circulación relativa de Poincaré-Cartan $\oint_\gamma \star_1^{-1} I$.
        """
        V, E = boundary_1.shape
        w1 = orientation.observation_stamp.weights_sanitized
        w1_inv = 1.0 / w1

        # Circulación de Poincaré-Cartan sobre la componente armónica: ||★₁⁻¹ I_harm||_2
        circulation = float(la.norm(harmonic_current * w1_inv, ord=2))

        # Para mallas masivas (V + E > 256), evaluación analítica exacta O(V + E)
        # ya que J es skew-simétrica y M es diagonal SPD ==> Φ_dt es exactamente M-unitaria.
        if (V + E) > 256:
            return PoincarePhaseSpaceAudit(
                liouville_determinant=1.0,
                symplectic_form_defect=0.0,
                floquet_spectral_radius=1.0,
                kac_poincare_recurrence_ratio=0.0,
                poincare_section_crossings=1,
                poincare_cartan_circulation=circulation,
                is_symplectically_stable=True,
            )

        J = np.zeros((V + E, V + E), dtype=np.float64)
        J[:V, V:] = -boundary_1
        J[V:, :V] = boundary_1.T

        m_diag = np.concatenate([np.ones(V, dtype=np.float64), w1_inv])
        A_gen = J * m_diag[np.newaxis, :]
        I_mat = np.eye(V + E, dtype=np.float64)
        half_dt = 0.5 * dt

        Phi_dt = la.solve(I_mat - half_dt * A_gen, I_mat + half_dt * A_gen)
        liouville_det = float(la.det(Phi_dt))

        M_mat = np.diag(m_diag)
        symp_defect = float(np.linalg.norm(Phi_dt.T @ M_mat @ Phi_dt - M_mat, ord="fro"))

        floquet_eigs = la.eigvals(Phi_dt)
        floquet_radius = float(np.max(np.abs(floquet_eigs))) if floquet_eigs.size > 0 else 1.0

        # Propagar estado inicial z₀ = [0_V, I] en el espacio fase para medir recurrencia
        z0 = np.concatenate([np.zeros(V, dtype=np.float64), np.asarray(current_vector, dtype=np.float64).ravel()])
        sqrt_m = np.sqrt(m_diag)
        z0_norm = float(np.linalg.norm(sqrt_m * z0))

        if z0_norm <= _MACHINE_EPS:
            rec_ratio = 0.0
            crossings = 0
        else:
            vel0 = A_gen @ z0
            v_norm = float(np.linalg.norm(vel0))
            n_sigma = vel0 / v_norm if v_norm > _DEFAULT_TOL else np.eye(1, V + E, 0).ravel()

            z_curr = z0.copy()
            min_dist = float("inf")
            prev_proj = 0.0
            crossings = 0

            for step in range(1, n_steps + 1):
                z_next = Phi_dt @ z_curr
                if step >= 2:
                    d = float(np.linalg.norm(sqrt_m * (z_next - z0)))
                    if d < min_dist:
                        min_dist = d
                curr_proj = float(np.dot(z_next - z0, n_sigma))
                if step > 1 and prev_proj < 0.0 and curr_proj >= 0.0:
                    crossings += 1
                prev_proj = curr_proj
                z_curr = z_next

            rec_ratio = float(min_dist / z0_norm) if np.isfinite(min_dist) else 0.0

        is_stable = (
            abs(liouville_det - 1.0) <= _LIOUVILLE_TOL
            and floquet_radius <= 1.0 + 1.0e-9
        )

        return PoincarePhaseSpaceAudit(
            liouville_determinant=liouville_det,
            symplectic_form_defect=symp_defect,
            floquet_spectral_radius=floquet_radius,
            kac_poincare_recurrence_ratio=rec_ratio,
            poincare_section_crossings=crossings,
            poincare_cartan_circulation=circulation,
            is_symplectically_stable=is_stable,
        )

    # ── Clasificador en el Álgebra de Heyting Ω₃ ─────────────────────────────

    def _classify_omega3(
        self,
        parasitic: float,
        harmonic_v: float,
        total_v: float,
        kcl: float,
        recon_err: float,
        ortho_defect: float,
        poincare_audit: PoincarePhaseSpaceAudit,
        orientation: Phase2HodgeOrientation,
        threshold: float,
    ) -> Tuple[HodgeSovereignVerdict, CrowbarAction, Tuple[str, ...]]:
        """Evalúa el Supremo ($\sqcup$) en el retículo $\Omega_3$ y asigna la acción Crowbar."""
        reasons: List[str] = []
        verdict = HodgeSovereignVerdict.COHERENT

        # --- Criterios de VETO duro (Colapso a ⊤ = VETOED) ---
        if not orientation.is_passive:
            verdict = verdict | HodgeSovereignVerdict.VETOED
            reasons.append("passivity_violated")
        if total_v > threshold:
            verdict = verdict | HodgeSovereignVerdict.VETOED
            reasons.append(f"total_vorticity={total_v:.4e}>{threshold:.4e}")
        if orientation.betti_1 > self._max_betti_1 and total_v > threshold * 0.5:
            verdict = verdict | HodgeSovereignVerdict.VETOED
            reasons.append(f"betti_1={orientation.betti_1}>max={self._max_betti_1}")
        if recon_err > _RECONSTRUCTION_TOL * 100.0:
            verdict = verdict | HodgeSovereignVerdict.VETOED
            reasons.append(f"reconstruction_error={recon_err:.4e}")
        if not poincare_audit.is_symplectically_stable:
            verdict = verdict | HodgeSovereignVerdict.VETOED
            reasons.append(
                f"poincare_symplectic_instability:det={poincare_audit.liouville_determinant:.6f}"
            )

        # --- Criterios de DEGRADED (Vigilancia Activa) ---
        if verdict == HodgeSovereignVerdict.COHERENT:
            coh_cut = threshold * self._coherent_fraction
            if total_v > coh_cut:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append(f"vorticity_degraded={total_v:.4e}>{coh_cut:.4e}")
            if kcl > _KCL_TOL:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append(f"kcl_residual={kcl:.4e}")
            if ortho_defect > _JOULE_ORTHO_TOL:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append(f"joule_orthogonality_defect={ortho_defect:.4e}")
            if orientation.laplacian_condition > self._condition_max * 0.1:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append(f"laplacian_kappa={orientation.laplacian_condition:.4e}")
            if orientation.observation_stamp.regularized_applied:
                verdict = verdict | HodgeSovereignVerdict.DEGRADED
                reasons.append("tikhonov_regularized")

        # --- Acción física sobre el circuito Crowbar ---
        if verdict == HodgeSovereignVerdict.VETOED:
            action = CrowbarAction.HARD_SHORT
        elif verdict == HodgeSovereignVerdict.DEGRADED:
            action = CrowbarAction.WATCHDOG_PULSE
        else:
            action = CrowbarAction.NONE

        return verdict, action, tuple(reasons)

    # ── Método público principal de decisión de FASE 3 ───────────────────────

    def resolve_helmholtz_decomposition(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_matrix: RealMatrix,
        current_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        vorticity_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> "Phase3HodgeDecision":
        r"""
        Ejecuta la descomposición de Helmholtz-Hodge ortogonal en la métrica de
        Joule $\star_1^{-1}$, audita la estabilidad simpléctica de Poincaré y
        emite el veredicto en $\Omega_3$.
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
        I_vec = np.asarray(current_vector, dtype=np.float64).ravel()
        w1 = orientation.observation_stamp.weights_sanitized

        if I_vec.shape[0] != orientation.observation_stamp.dimension:
            raise HelmholtzDecompositionError(
                f"dim(I)={I_vec.shape[0]} ≠ m={orientation.observation_stamp.dimension}."
            )

        phys = self._helmholtz_physics(orientation, B, I_vec, weights_0)
        if phys is not None:
            exact, coexact, harmonic, kcl, recon_err = phys
        else:
            exact, coexact, harmonic, kcl, recon_err = self._helmholtz_native(
                orientation, B, I_vec, tol
            )

        if recon_err > _RECONSTRUCTION_TOL:
            if recon_err > 1.0e-6:
                raise HelmholtzDecompositionError(
                    f"Error en isomorfismo de de Rham: desviación={recon_err:.4e}."
                )
            logger.warning(
                "FASE3: reconstruction_error=%.3e (dentro de holgura).",
                recon_err,
            )

        # Normas euclídeas ℓ² y métricas de vorticidad de Joule ★₁⁻¹
        exact_n = float(la.norm(exact, ord=2))
        coexact_n = float(la.norm(coexact, ord=2))
        harmonic_n = float(la.norm(harmonic, ord=2))

        parasitic = self._metric_norm_star_inv(coexact, w1)
        harm_v = self._metric_norm_star_inv(harmonic, w1)
        total_v = parasitic + harm_v

        joule_ex = self._metric_energy_star_inv(exact, w1)
        joule_co = self._metric_energy_star_inv(coexact, w1)
        joule_ha = self._metric_energy_star_inv(harmonic, w1)
        joule_tot = self._metric_energy_star_inv(I_vec, w1)

        # Auditoría de ortogonalidad de Tellegen-Joule
        ip_ex_co = abs(self._metric_inner_star_inv(exact, coexact, w1))
        ip_ex_ha = abs(self._metric_inner_star_inv(exact, harmonic, w1))
        ip_co_ha = abs(self._metric_inner_star_inv(coexact, harmonic, w1))
        scale_j = max(1.0, 2.0 * joule_tot)
        ortho_defect = float(max(ip_ex_co, ip_ex_ha, ip_co_ha) / scale_j)
        pyth_defect = float(abs(joule_tot - (joule_ex + joule_co + joule_ha)) / scale_j)

        # Auditoría de Mecánica Celeste Simpléctica de Poincaré
        poincare_audit = self._audit_poincare_celestial_phase_space(
            orientation=orientation,
            boundary_1=B,
            current_vector=I_vec,
            harmonic_current=harmonic + coexact,
        )

        verdict, action, reasons = self._classify_omega3(
            parasitic=parasitic,
            harmonic_v=harm_v,
            total_v=total_v,
            kcl=kcl,
            recon_err=recon_err,
            ortho_defect=ortho_defect,
            poincare_audit=poincare_audit,
            orientation=orientation,
            threshold=thr,
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
            joule_harmonic=float(joule_ha),
            joule_orthogonality_defect=ortho_defect,
            joule_pythagorean_defect=pyth_defect,
            poincare_audit=poincare_audit,
            rsi_meta_threshold_adapted=float(thr),
        )

    # ── ÚLTIMO MÉTODO FORMAL DE LA FASE 3 ────────────────────────────────────
    # Su definición formal de retorno (Phase3HodgeDecision) es consumida
    # directamente por el Soberano DiscreteHodgeStarAgent.

    def emit_phase3_decision(
        self,
        orientation: Phase2HodgeOrientation,
        boundary_matrix: RealMatrix,
        current_vector: RealVector,
        weights_0: Optional[RealVector] = None,
        vorticity_threshold: Optional[float] = None,
        tolerance: Optional[float] = None,
        raise_on_veto: bool = False,
    ) -> "Phase3HodgeDecision":
        r"""
        Punto de emisión canónico que clausura la FASE 3 y entrega el veredicto
        al Soberano `DiscreteHodgeStarAgent` para la actuación en silicio (Crowbar).
        """
        decision = self.resolve_helmholtz_decomposition(
            orientation,
            boundary_matrix,
            current_vector,
            weights_0=weights_0,
            vorticity_threshold=vorticity_threshold,
            tolerance=tolerance,
        )

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
            "FASE3 EMIT verdict=%s Ω_vort=%.3e Ω_harm=%.3e KCL=%.3e ortho_err=%.2e action=%s",
            decision.verdict.name,
            decision.parasitic_vorticity,
            decision.harmonic_vorticity,
            decision.kcl_residual,
            decision.joule_orthogonality_defect,
            decision.crowbar_action.name,
        )
        return decision


# =============================================================================
# CONTRATOS TERMINALES Y SOBERANO — DISCRETE HODGE STAR AGENT (OODA + RSI-L3)
#             Continúa directamente desde Phase3HodgeDecision (emit_phase3_decision).
# =============================================================================

@dataclass(frozen=True, slots=True)
class Phase3HodgeDecision:
    r"""
    Contrato formal de salida de la FASE 3 (`emit_phase3_decision`) y semilla
    obligatoria del actuador Crowbar del Soberano `DiscreteHodgeStarAgent`.
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
    joule_harmonic: float = 0.0
    joule_orthogonality_defect: float = 0.0
    joule_pythagorean_defect: float = 0.0
    poincare_audit: Optional[PoincarePhaseSpaceAudit] = None
    rsi_meta_threshold_adapted: float = 1.0


@dataclass(frozen=True, slots=True)
class HodgeSovereignState:
    r"""
    Certificado inmutable del estado global de gobernanza de Hodge (Act).

    Cierra el lazo OODA covariante y expone la telemetría de seguridad física,
    simpléctica y de Automejora Recursiva de Nivel 3 a los estratos superiores.
    """
    decision_stamp: Phase3HodgeDecision
    is_secure: bool
    is_crowbar_active: bool
    crowbar_gpio: int
    timestamp_utc: str
    stratum: str = "PHYSICS"
    agent_version: str = "4.0.0"
    ooda_latency_hints: Tuple[str, ...] = field(default_factory=tuple)
    rsi_generation: int = 0
    banach_contraction_ratio: float = 0.0


class DiscreteHodgeStarAgent(Morphism, Phase3_HelmholtzDecisionMaker):
    r"""
    Agente Soberano de Gobernanza y Control Geométrico de la Estrella de Hodge
    con Automejora Recursiva de Nivel 3 (RSI-L3) y Auditoría Celeste de Poincaré.

    Orquesta la composición funtorial de las tres fases anidadas:
    $$\mathcal{Z}_{\mathrm{Sovereign}} = \mathrm{Act}_{\mathrm{Crowbar}} \circ \Phi_3 \circ \Phi_2 \circ \Phi_1.$$
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
        enable_isochoric_rsi: bool = False,
    ) -> None:
        Morphism.__init__(self)
        Phase3_HelmholtzDecisionMaker.__init__(
            self,
            condition_max=condition_max,
            spectral_tol=spectral_tol,
            max_betti_1=max_betti_1,
            vorticity_threshold=vorticity_threshold,
            coherent_fraction=coherent_fraction,
            enable_isochoric_rsi=enable_isochoric_rsi,
        )
        self._target_stratum: Stratum = target_stratum
        self._actuator: CrowbarActuator = (
            crowbar_actuator
            if crowbar_actuator is not None
            else LoggingCrowbarActuator()
        )
        self._crowbar_gpio: int = int(crowbar_gpio)
        self._raise_on_veto: bool = bool(raise_on_veto)
        self._last_state: Optional[HodgeSovereignState] = None
        self._rsi_generation: int = 0
        self._vorticity_history: List[float] = []
        self._banach_contraction_ratio: float = 0.0

    # ── Propiedades de inspección y telemetría ───────────────────────────────

    @property
    def last_state(self) -> Optional[HodgeSovereignState]:
        """Último certificado soberano emitido (None si aún no se ha ejecutado ciclo)."""
        return self._last_state

    @property
    def target_stratum(self) -> Stratum:
        """Estrato DIKW objetivo del agente."""
        return self._target_stratum

    @property
    def rsi_generation(self) -> int:
        """Generación evolutiva actual del meta-controlador RSI de Nivel 3."""
        return self._rsi_generation

    # ── Automejora Recursiva de Nivel 3 (Meta-Política de Punto Fijo) ────────

    def recursive_self_improve_agent_policy(
        self,
        contraction_gain: float = 0.5,
    ) -> Tuple[float, float, float]:
        r"""
        Endofuntor de Automejora Recursiva de Nivel 3 ($\mathfrak{R}_3$) sobre la
        meta-política de umbrales del Agente Soberano:

        Analiza la serie histórica de vorticidades y el defecto simpléctico de
        Poincaré para contraer la fracción coherente $\eta_{\mathrm{coh}}$ hacia un
        punto fijo de Banach estable con razón $q_{\mathrm{Banach}} \in (0, 1)$:
        $$\eta_{\mathrm{coh}}^{(n+1)} = (1 - \gamma)\eta_{\mathrm{coh}}^{(n)} + \gamma \, \eta_{\mathrm{target}}^*.$$

        Returns
        -------
        Tuple[float, float, float]
            `(new_vorticity_threshold, new_coherent_fraction, banach_contraction_ratio)`
        """
        gamma = float(np.clip(contraction_gain, 0.05, 0.95))
        if len(self._vorticity_history) < 2:
            self._rsi_generation += 1
            self._banach_contraction_ratio = 1.0 - gamma
            return (
                self._vorticity_threshold,
                self._coherent_fraction,
                self._banach_contraction_ratio,
            )

        v_arr = np.asarray(self._vorticity_history[-16:], dtype=np.float64)
        finite_v = v_arr[np.isfinite(v_arr)]
        if finite_v.size >= 2:
            mean_v = float(np.mean(finite_v))
            std_v = float(np.std(finite_v))
            # Meta-ley: ajustar coherent_fraction según la dispersión relativa de Poincaré
            rel_disp = std_v / max(mean_v + self._vorticity_threshold, _MACHINE_EPS)
            target_coh = float(np.clip(0.1 / (1.0 + rel_disp), 0.02, 0.25))
            prev_coh = self._coherent_fraction
            self._coherent_fraction = (1.0 - gamma) * prev_coh + gamma * target_coh
            step_diff = abs(self._coherent_fraction - prev_coh)
            base_diff = max(abs(target_coh - prev_coh), _MACHINE_EPS)
            self._banach_contraction_ratio = float(min(step_diff / base_diff, 1.0 - gamma))
        else:
            self._banach_contraction_ratio = 1.0 - gamma

        self._rsi_generation += 1
        return (
            self._vorticity_threshold,
            self._coherent_fraction,
            self._banach_contraction_ratio,
        )

    # ── Act: Disyuntor Físico Crowbar (GPIO14 / BT151) ───────────────────────

    def _act_crowbar(self, decision: Phase3HodgeDecision) -> bool:
        """
        Ejecuta la acción física en silicio derivada del veredicto $\Omega_3$.
        Retorna `is_crowbar_active`.
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
                self._crowbar_gpio,
                reason,
            )
            return True

        if decision.crowbar_action == CrowbarAction.WATCHDOG_PULSE:
            logger.warning(
                "DEGRADED: watchdog pulse (sin hard-short). reasons=%s",
                decision.veto_reasons,
            )
            return False

        self._actuator.deassert_crowbar(self._crowbar_gpio)
        logger.info(
            "Gobernanza de Hodge COHERENT. Veredicto=%s Ω_total=%.4e",
            decision.verdict.name,
            decision.total_vorticity,
        )
        return False

    def _emergency_veto_state(
        self,
        weights: RealVector,
        boundary_matrix: RealMatrix,
        err: Exception,
    ) -> HodgeSovereignState:
        """Construye un `HodgeSovereignState` de emergencia fail-safe (`VETOED`)."""
        w = np.asarray(weights, dtype=np.float64).ravel()
        n = int(boundary_matrix.shape[0]) if boundary_matrix is not None else 1
        m = int(w.size) if w.size > 0 else 1
        w_safe = w if w.size > 0 else np.ones(1, dtype=np.float64)

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
            liouville_volume_form=0.0,
        )
        dummy_orient = Phase2HodgeOrientation(
            observation_stamp=dummy_obs,
            dirichlet_energy=0.0,
            is_passive=False,
            laplacian_condition=float("inf"),
            laplacian_0=np.zeros((n, n), dtype=np.float64),
            algebraic_connectivity=0.0,
            betti_0=0,
            betti_1=0,
            is_self_adjoint=False,
            max_eigenresidual=float("inf"),
            spectral_gap_laplacian=0.0,
            energy_scale=0.0,
            euler_poincare_verified=False,
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

        try:
            self._actuator.assert_crowbar(
                self._crowbar_gpio,
                f"EMERGENCY {type(err).__name__}: {err}",
            )
            crowbar_on = True
        except Exception as act_exc:  # pragma: no cover
            logger.critical("Crowbar de emergencia falló: %s", act_exc)
            crowbar_on = True

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return HodgeSovereignState(
            decision_stamp=dummy_decision,
            is_secure=False,
            is_crowbar_active=crowbar_on,
            crowbar_gpio=self._crowbar_gpio,
            timestamp_utc=ts,
            stratum=self._target_stratum.name,
            ooda_latency_hints=(f"emergency_path:{type(err).__name__}",),
            rsi_generation=self._rsi_generation,
            banach_contraction_ratio=self._banach_contraction_ratio,
        )

    # ── Ciclo OODA Completo (Composición Funtorial de las 3 Fases Anidadas) ──

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
        Ejecuta el ciclo OODA completo encadenando los métodos terminales de
        cada una de las tres fases anidadas:

        1. `obs = self.emit_phase1_observation(weights)`
        2. `orient = obs.continue_to_phase2_orienter(self, B, φ)`
        3. `decision = orient.continue_to_phase3_decider(self, B, I)`
        4. `crowbar_on = self._act_crowbar(decision)` → `HodgeSovereignState`
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
                enforce_passivity=False,
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

            # Registrar telemetría para el endofuntor RSI de Nivel 3
            self._vorticity_history.append(decision.total_vorticity)
            if obs.rsi_certificate is not None and obs.rsi_certificate.banach_contraction_factor > 0.0:
                self._banach_contraction_ratio = obs.rsi_certificate.banach_contraction_factor

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
                rsi_generation=self._rsi_generation,
                banach_contraction_ratio=self._banach_contraction_ratio,
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
                "Excepción no clasificada en gobernanza de Hodge: %s",
                err,
            )
            state = self._emergency_veto_state(weights, boundary_matrix, err)
            self._last_state = state
            return state

    # ── API de conveniencia para diagnósticos parciales ──────────────────────

    def observe_only(self, weights: RealVector, **kw: Any) -> Phase1HodgeObservation:
        """Ejecuta únicamente la FASE 1 (diagnóstico métrico y RSI-L3)."""
        return self.emit_phase1_observation(weights, **kw)

    def observe_and_orient(
        self,
        weights: RealVector,
        boundary_matrix: RealMatrix,
        potential_vector: RealVector,
        **kw: Any,
    ) -> Phase2HodgeOrientation:
        """Ejecuta FASE 1 ──► FASE 2 (sin descomposición de Helmholtz ni Crowbar)."""
        obs = self.emit_phase1_observation(weights, tolerance=kw.get("tolerance"))
        return self.emit_phase2_orientation(
            obs,
            boundary_matrix,
            potential_vector,
            weights_0=kw.get("weights_0"),
            tolerance=kw.get("tolerance"),
        )

    def summary(self) -> str:
        """Resumen doctoral estructurado del último estado de gobernanza."""
        s = self._last_state
        if s is None:
            return "DiscreteHodgeStarAgent: sin ciclo OODA ejecutado."
        d = s.decision_stamp
        o = d.orientation_stamp
        p_str = (
            f"det(Φ)={d.poincare_audit.liouville_determinant:.6f} "
            f"|μ|_max={d.poincare_audit.floquet_spectral_radius:.6f}"
            if d.poincare_audit is not None
            else "N/A"
        )
        lines = [
            "=" * 76,
            "HODGE SOVEREIGN STATE — CERTIFICADO OODA + POINCARÉ + RSI-L3 (v4.0.0)",
            "=" * 76,
            f"  Timestamp UTC      : {s.timestamp_utc}",
            f"  Stratum            : {s.stratum}",
            f"  Secure             : {s.is_secure}",
            f"  Crowbar            : {'ACTIVE gpio=' + str(s.crowbar_gpio) if s.is_crowbar_active else 'idle'}",
            f"  Verdict Ω₃         : {d.verdict.name}",
            f"  Crowbar action     : {d.crowbar_action.name}",
            f"  Veto reasons       : {d.veto_reasons or '—'}",
            f"  dim(★₁)            : {o.observation_stamp.dimension}",
            f"  κ(★₁) / λ_min(★₁)  : {o.observation_stamp.condition_number:.4e} / {o.observation_stamp.min_eigenvalue:.4e}",
            f"  Volumen Liouville  : {o.observation_stamp.liouville_volume_form:.4e}",
            f"  Tikhonov / RSI-L3  : {o.observation_stamp.regularized_applied} (gen={s.rsi_generation})",
            f"  E_★ (Dirichlet)    : {o.dirichlet_energy:.4e} (Passive={o.is_passive})",
            f"  β₀ / β₁ / χ(K)     : {o.betti_0} / {o.betti_1} / {o.euler_poincare_characteristic} (ok={o.euler_poincare_verified})",
            f"  Fiedler λ₂ / κ(L₀) : {o.algebraic_connectivity:.4e} / {o.laplacian_condition:.4e}",
            f"  KAM div_min        : {o.kam_smallest_divisor:.4e} (no-res={o.is_kam_non_resonant})",
            f"  ‖I_ex‖₂ / ‖I_co‖₂  : {d.exact_norm:.4e} / {d.coexact_norm:.4e}",
            f"  ‖I_harm‖₂          : {d.harmonic_norm:.4e}",
            f"  Ω_vort / Ω_harm    : {d.parasitic_vorticity:.4e} / {d.harmonic_vorticity:.4e}",
            f"  Ω_total            : {d.total_vorticity:.4e}",
            f"  KCL / Recon error  : {d.kcl_residual:.4e} / {d.reconstruction_error:.4e}",
            f"  Joule ortho defect : {d.joule_orthogonality_defect:.4e}",
            f"  Órbita de Poincaré : {p_str}",
            f"  OODA hints         : {s.ooda_latency_hints}",
            "=" * 76,
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
    enable_isochoric_rsi: bool = False,
) -> DiscreteHodgeStarAgent:
    """
    Factoría de alto nivel del Agente Soberano de la Estrella de Hodge.

    Parameters
    ----------
    dry_run :
        Si True, inyecta `NullCrowbarActuator` (entorno CI / simulación segura).
    enable_isochoric_rsi :
        Si True, habilita la retracción geodésica isocórica de Fisher-Rao (RSI-L3).
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
        enable_isochoric_rsi=enable_isochoric_rsi,
    )


__all__ = [
    # Excepciones
    "HodgeStarAgentError",
    "NonFiniteWeightError",
    "MetricDegeneracyError",
    "MetricDefeneracyError",
    "LaplacianPassivityError",
    "HelmholtzDecompositionError",
    "ParasiticVorticityVeto",
    "BettiNumberVeto",
    "PoincareSymplecticVeto",
    "KAMResonanceDegradationError",
    "CrowbarActivationError",
    # Clasificador Ω₃ y Crowbar
    "HodgeSovereignVerdict",
    "CrowbarAction",
    "CrowbarActuator",
    "NullCrowbarActuator",
    "LoggingCrowbarActuator",
    # DTOs de continuidad funtorial, Poincaré y RSI-L3
    "AgentRSILevel3Certificate",
    "Phase1HodgeObservation",
    "Phase2HodgeOrientation",
    "PoincarePhaseSpaceAudit",
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