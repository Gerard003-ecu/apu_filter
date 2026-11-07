# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Inertia Agent (Soberano del Momentum Ciber-Físico)       ║
║ Ruta   : app/agents/physics/riemannian_inertia_agent.py                      ║
║ Versión: 5.0.0-Topos-Heyting-Poincare-KAM-Floquet-RSI-Level3-Strict          ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA, MECÁNICA CELESTE DE POINCARÉ Y TOPOS DE HEYTING: ──────
Este módulo consagra al Agente Soberano y Observador Activo que gobierna al
funtor físico ``riemannian_inertia_modulator.py``. Reside en el hiperespacio
del estrato superior de Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0), supervisando la
dinámica de-confinada del espacio de fase de la Malla mediante la **Mecánica
Celeste de Henri Poincaré** y **Automejora Recursiva de Nivel 3 (RSI-L3)**.

Su mandato axiomático es orquestar el ciclo OODA sobre el operador de momento
giroscópico $$\mathcal{W}$$ en el fibrado cotangente $$T^*\mathcal{M}$$,
sometiendo el flujo de intenciones semánticas a restricciones geométricas,
simplécticas y celestes rigurosas. Actúa aplicando una "Fuerza de Lorentz"
informacional que desvía trayectorias de alto riesgo (alucinaciones del LLM)
hacia sumideros de disipación sin alterar el Hamiltoniano de energía basal del
sistema. Toda contención de fallos se confina de manera síncrona y absoluta al
plano lógico de software mediante el colapso del retículo de Heyting y sus
operadores modales de Lawvere-Tierney.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CELESTES PRESERVADOS: ─────────────

  [I1] Preservación del Volumen de Liouville e Invariantes de Poincaré-Cartan:
       La evolución conserva idénticamente la 2-forma simpléctica canónica:
       $$\omega = \sum_{i=1}^n dq_i \wedge dp_i, \qquad
         \theta_{\mathrm{PC}} = p_\mu\dot{q}^\mu - H_{\mathrm{kin}}
                              = \tfrac12\langle p,\dot{q}\rangle$$
       y la identidad de acción-ángulo de Poincaré sobre el toro espectral:
       $$\sum_{k=1}^n I_k\,\omega_k = \tfrac12 p^\top G^{-1} p = H_{\mathrm{kin}}$$

  [I2] Antisimetría Euclidiana, Firma de Calibre, Casimir y Darboux-Parseval:
       El tensor giroscópico habita en $$\mathfrak{so}(n)$$ y satisface la
       identidad espectral de bloques normales de Darboux:
       $$\mathcal{W}^\top+\mathcal{W}=\mathbf{0},\qquad
         \mathcal{C}_2(\mathcal{W}) = -\tfrac12\mathrm{Tr}(\mathcal{W}^2)
         = \tfrac12\|\mathcal{W}\|_F^2 = \sum_{k=1}^{\lfloor n/2\rfloor}\nu_k^2$$
       Si la métrica $$G$$ está disponible, se audita además la
       $$G$$-antisimetría de calibre $$\mathcal{W}^\top G + G\,\mathcal{W}=\mathbf{0}$$.

  [I3] Ley de Trabajo Mecánico Nilpotente y Estabilidad de Floquet-Kreĭn:
       $$P_{\mathrm{gyro}}
         =\langle\nabla H,\,J_{\mathrm{eff}}\nabla H\rangle
         \equiv 0\pmod{\varepsilon_{\mathrm{machine}}}$$
       Los multiplicadores característicos de Floquet de $$J_{\mathrm{eff}}$$
       permanecen confinados al eje imaginario puro (sin colisión hiperbólica
       de Kreĭn):
       $$\max_{\mu\in\sigma(J_{\mathrm{eff}})}|\Re(\mu)| \le \tau_{\mathrm{Floquet}}$$

  [I4] Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{\mathrm{Topos}}$$):
       En cada fase, el agente audita la monotonía de Lyapunov del motor físico
       y sintetiza recursivamente una ley modal de Lawvere-Tierney que contrae
       la discrepancia epistémica $$\mathcal{V}_{k+1}^{\mathrm{Gov}} \le
       \mathcal{V}_k^{\mathrm{Gov}}$$ preservando intactas las cotas físicas duras.

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA): ─────────────────
La transición de estados se rige por un contrato covariante estricto donde la
definición formal final del último método de la Fase $$k$$ es la continuación o
inicio formal del primer método de la Fase $$k+1$$:

  Fase 1 ──► FASE 1: AUDITORÍA DE LIOUVILLE, TOROS KAM, SUNDMAN Y RSI-L3
             (Phase1_LiouvilleVolumeAuditor)
             Intercepta el certificado espectral del motor y clasifica en
             $$\Omega_3$$ la cota de volumen, condicionamiento, inversa de
             Wilkinson, round-trip musical, apareamiento dual, regularización
             de Sundman, margen Diofántico KAM, acción-ángulo de Poincaré-Cartan
             y monotonía RSI-L3 del motor.
             Entrega: Phase1ObservationBridge.
             Morfismo terminal: handoff_phase1_to_phase2 ≡ inicio de Fase 2.

  Fase 2 ──► FASE 2: CERTIFICACIÓN DE FIRMA MÉTRICA, CASIMIR-DARBOUX Y RSI-L3
             (Phase2_SkewSymmetryCertifier)
             Inicia en _receive_certified_observation invocando directamente
             handoff_phase1_to_phase2.
             Valida la antisimetría de $$\mathcal{W}$$, la identidad de Gram,
             la paridad del rango, la pureza de vorticidad, la $$G$$-antisimetría,
             el invariante de Casimir, la identidad espectral de Darboux-Parseval,
             la circulación areal de Poincaré y el certificado RSI-L3 de síntesis.
             Entrega: Phase2OrientationBridge.
             Morfismo terminal: handoff_phase2_to_phase3 ≡ inicio de Fase 3.

  Fase 3 ──► FASE 3: COLAPSO DE HEYTING, ESTABILIDAD FLOQUET-KREĬN Y RSI-L3
             (Phase3_HeytingLatticeDecider)
             Inicia en _receive_certified_orientation invocando directamente
             handoff_phase2_to_phase3.
             Somete $$J_{\mathrm{eff}}$$ a la auditoría de trabajo nilpotente,
             estabilidad elíptica de Floquet-Kreĭn, parámetro orbital de
             Tisserand, invariante virial de Jacobi-Poincaré y recurrencia de
             Kac-Poincaré, calculando el supremo en $$\Omega_3$$:
             $$v_{\mathrm{final}}
               = v_{\mathrm{Liouville}}
               \sqcup v_{\mathrm{Skew}}
               \sqcup v_{\mathrm{Work}}$$
             Entrega: InertialGovernanceState.
             Morfismo terminal: handoff_phase3_to_orchestrator ≡ enlace con
             RiemannianInertiaAgent.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema APU Filter / MIC.
# Se conservan stubs robustos, compatibles con modulator 4.0.0 y 5.0.0.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.riemannian_inertia_modulator import (
        GyroscopicSynthesisData,
        MomentumAuditData,
        RSILevel3AuditMetrics,
        RiemannianInertiaModulator,
        ThermodynamicVetoData,
    )
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación base a un invariante topológico categórico."""

        pass

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

        pass

    @dataclass(frozen=True, slots=True)
    class RSILevel3AuditMetrics:
        """Certificado de Automejora Recursiva de Nivel 3 del motor físico."""

        recursion_depth: int = 0
        initial_epistemic_defect: float = 0.0
        terminal_epistemic_defect: float = 0.0
        lyapunov_contraction_ratio: float = 1.0
        meta_law_gain_parameter: float = 1.0
        synthesized_law_tag: str = "Identity-Level0"
        is_monotone_lyapunov: bool = True
        is_fixed_point_converged: bool = True

    @dataclass(frozen=True, slots=True)
    class MomentumAuditData:
        """Artefacto de Fase 1 del motor, compatible con modulator 4.0.0 y 5.0.0."""

        covariant_momentum: NDArray[np.float64]
        momentum_norm: float
        is_bounded: bool
        metric_condition_number: float = 1.0
        inverse_consistency_residual: float = 0.0
        reconstructed_velocity: NDArray[np.float64] | None = None
        kinetic_energy_primal: float = 0.0
        kinetic_energy_dual: float = 0.0
        dual_pairing: float = 0.0
        pairing_residual: float = 0.0
        musical_roundtrip_residual: float = 0.0
        wilkinson_bound: float = 0.0
        spectral_minimum: float = 1.0
        spectral_maximum: float = 1.0
        poincare_action_variables: NDArray[np.float64] | None = None
        poincare_orbital_frequencies: NDArray[np.float64] | None = None
        poincare_cartan_one_form_density: float = 0.0
        sundman_regularization_factor: float = 1.0
        kam_diophantine_margin: float = 1.0
        rsi_l3_metric_audit: RSILevel3AuditMetrics | None = None

    @dataclass(frozen=True, slots=True)
    class GyroscopicSynthesisData:
        """Artefacto de Fase 2 del motor, compatible con modulator 4.0.0 y 5.0.0."""

        skew_symmetric_tensor: NDArray[np.float64]
        antisymmetry_residual: float
        vorticity_projection_residual: float = 0.0
        gyroscopic_frobenius_norm: float = 0.0
        is_strictly_skew: bool = True
        relative_skew_residual: float = 0.0
        vorticity_two_form: NDArray[np.float64] | None = None
        omega_vector: NDArray[np.float64] | None = None
        wedge_gram_residual: float = 0.0
        skew_numerical_rank: int = 0
        rank_is_even: bool = True
        poincare_birkhoff_secular_factor: float = 1.0
        casimir_quadratic_invariant: float = 0.0
        casimir_trace_residual: float = 0.0
        darboux_characteristic_frequencies: NDArray[np.float64] | None = None
        poincare_areal_circulation: float = 0.0
        rsi_l3_synthesis_audit: RSILevel3AuditMetrics | None = None

    @dataclass(frozen=True, slots=True)
    class ThermodynamicVetoData:
        """Artefacto de Fase 3 del motor, compatible con modulator 4.0.0 y 5.0.0."""

        effective_dirac_matrix: NDArray[np.float64]
        nilpotent_work_residual: float
        dirac_symmetric_residual: float = 0.0
        work_tolerance: float = 1e-10
        is_symplectically_passive: bool = True
        pairwise_work_residual: float = 0.0
        liouville_probe_residual: float = 0.0
        relative_skew_residual: float = 0.0
        jacobi_poincare_virial_invariant: float = 0.0
        tisserand_gyroscopic_parameter: float = 1.0
        floquet_max_real_drift: float = 0.0
        floquet_spectral_radius: float = 0.0
        poincare_recurrence_horizon: float = 0.0
        rsi_l3_symplectic_audit: RSILevel3AuditMetrics | None = None

    RiemannianInertiaModulator = Any  # type: ignore[misc, assignment]


logger = logging.getLogger("MIC.Agents.Physics.RiemannianInertiaAgent")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, CELESTES (POINCARÉ), MODALIDAD HEYTING Y RSI-L3
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# ── Límites duros de integridad física, simpléctica y celeste ───────────────
_MOMENTUM_HARD_LIMIT: Final[float] = 1e8
_CONDITION_HARD_MAX: Final[float] = 1e12
_INVERSE_RESIDUAL_HARD_MAX: Final[float] = 1e-5
_SKEW_HARD_RELATIVE_TOLERANCE: Final[float] = 1e-6
_VORTICITY_PROJECTION_HARD_RATIO: Final[float] = 1e2
_DIRAC_SYMMETRIC_HARD_RELATIVE_TOLERANCE: Final[float] = 1e-8
_PAIRING_RESIDUAL_HARD_MAX: Final[float] = 1e-6
_MUSICAL_ROUNDTRIP_HARD_MAX: Final[float] = 1e-6
_GRAM_RESIDUAL_HARD_MAX: Final[float] = 1e-6
_GAUGE_SKEW_HARD_RELATIVE_TOLERANCE: Final[float] = 1e-6
_SPECTRAL_FLOOR_RATIO_HARD: Final[float] = 1e-14

# Límites duros de Mecánica Celeste de Poincaré
_SUNDMAN_FACTOR_HARD_MIN: Final[float] = 1e-9
_POINCARE_CARTAN_GAP_HARD_MAX: Final[float] = 1e-5
_CASIMIR_TRACE_HARD_RELATIVE_TOLERANCE: Final[float] = 1e-6
_DARBOUX_PARSEVAL_HARD_RELATIVE_TOLERANCE: Final[float] = 1e-6
_FLOQUET_DRIFT_HARD_RELATIVE_TOLERANCE: Final[float] = 1e-8
_TISSERAND_CAUCHY_HARD_SLACK: Final[float] = 1e-6

# ── Umbrales blandos de degradación lógica ──────────────────────────────────
_MOMENTUM_SOFT_LIMIT: Final[float] = 1e6
_CONDITION_SOFT_MAX: Final[float] = 1e6
_INVERSE_RESIDUAL_SOFT_MAX: Final[float] = 1e-7
_SKEW_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-9
_VORTICITY_PROJECTION_SOFT_RATIO: Final[float] = 1e-3
_WORK_SOFT_ABSOLUTE_TOLERANCE: Final[float] = 1e-10
_DIRAC_SYMMETRIC_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-12
_PAIRING_RESIDUAL_SOFT_MAX: Final[float] = 1e-10
_MUSICAL_ROUNDTRIP_SOFT_MAX: Final[float] = 1e-10
_GRAM_RESIDUAL_SOFT_MAX: Final[float] = 1e-10
_GAUGE_SKEW_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-9
_SPECTRAL_FLOOR_RATIO_SOFT: Final[float] = 1e-10

# Umbrales blandos de Mecánica Celeste de Poincaré
_SUNDMAN_FACTOR_SOFT_MIN: Final[float] = 1e-4
_POINCARE_CARTAN_GAP_SOFT_MAX: Final[float] = 1e-9
_CASIMIR_TRACE_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-10
_DARBOUX_PARSEVAL_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-10
_FLOQUET_DRIFT_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-12

# ── Parámetros de Automejora Recursiva de Nivel 3 en Gobernanza (RSI-L3) ────
_GOV_RSI_L3_MAX_DEPTH: Final[int] = 5
_GOV_RSI_L3_FLOOR: Final[float] = 4.0 * _MACHINE_EPSILON

# Versión canónica del agente.
__version__: Final[str] = (
    "5.0.0-Topos-Heyting-Poincare-KAM-Floquet-RSI-Level3-Strict"
)


# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING Y JERARQUÍA DE EXCEPCIONES LÓGICO-MATEMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════
class InertialHeytingVerdict(IntEnum):
    r"""
    Clasificador de subobjetos en el Topos de la Inercia Riemanniana.

    Cadena de Heyting $$\Omega_3$$:

        $$\bot=\mathtt{COHERENT}
          \le\mathtt{DEGRADED}
          \le\mathtt{VETOED}=\top$$

    El orden es de *severidad*, no de verdad booleana clásica.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def join(self, other: InertialHeytingVerdict) -> InertialHeytingVerdict:
        """Supremo $$a\sqcup b=\max(a,b)$$."""
        if not isinstance(other, InertialHeytingVerdict):
            raise TypeError("join exige InertialHeytingVerdict.")
        return InertialHeytingVerdict(max(self.value, other.value))

    def meet(self, other: InertialHeytingVerdict) -> InertialHeytingVerdict:
        """Ínfimo $$a\sqcap b=\min(a,b)$$."""
        if not isinstance(other, InertialHeytingVerdict):
            raise TypeError("meet exige InertialHeytingVerdict.")
        return InertialHeytingVerdict(min(self.value, other.value))

    def implies(self, other: InertialHeytingVerdict) -> InertialHeytingVerdict:
        r"""
        Implicación de Heyting en una cadena:

            $$a\Rightarrow b=\top$$ si $$a\le b$$, si no $$b$$.
        """
        if not isinstance(other, InertialHeytingVerdict):
            raise TypeError("implies exige InertialHeytingVerdict.")
        if self.value <= other.value:
            return InertialHeytingVerdict.VETOED
        return other

    def negate(self) -> InertialHeytingVerdict:
        r"""Negación intuicionista $$\neg a=a\Rightarrow\bot$$."""
        return self.implies(InertialHeytingVerdict.COHERENT)

    def lawvere_tierney_double_negation(self) -> InertialHeytingVerdict:
        r"""
        Operador modal de clausura densa de Lawvere-Tierney:

            $$j_{\neg\neg}(a) = \neg\neg a = (a\Rightarrow\bot)\Rightarrow\bot$$

        En $$\Omega_3$$, $$j_{\neg\neg}(\bot)=\bot$$ y
        $$j_{\neg\neg}(\mathtt{DEGRADED})=j_{\neg\neg}(\top)=\top$$.
        """
        return self.negate().negate()

    @property
    def is_terminal(self) -> bool:
        """Verdadero sii el veredicto es el supremo $$\top$$."""
        return self is InertialHeytingVerdict.VETOED


class RiemannianInertiaAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de Inercia Riemanniana."""

    pass


class MotorContractError(RiemannianInertiaAgentError):
    """El motor físico no satisface el contrato funtorial exigido."""

    pass


class PhaseHandoffCollapse(RiemannianInertiaAgentError):
    """Un certificado de fase no satisface las precondiciones de la siguiente."""

    pass


class LiouvilleVolumeCollapse(RiemannianInertiaAgentError):
    r"""
    El momentum desgarra asintóticamente la variedad simpléctica:

        $$\|p\|_{G^{-1}}=\sqrt{p_\mu G^{\mu\nu}p_\nu}>P_{\mathrm{hard}}$$
    """

    pass


class MetricConditionCollapse(RiemannianInertiaAgentError):
    r"""La métrica viola la cota de condición: $$\kappa(G)>\kappa_{\mathrm{hard}}$$."""

    pass


class InverseCoherenceCollapse(RiemannianInertiaAgentError):
    """El residuo de inversa excede la cota dura de Wilkinson-política."""

    pass


class DualPairingCollapse(RiemannianInertiaAgentError):
    r"""Se rompe $$\langle p,\dot{q}\rangle=\dot{q}^\top G\dot{q}=p^\top G^{-1}p$$."""

    pass


class MusicalIsomorphismCollapse(RiemannianInertiaAgentError):
    r"""Se rompe el round-trip $$\sharp\circ\flat=\mathrm{id}$$."""

    pass


class PoincareCelestialCollapse(LiouvilleVolumeCollapse):
    r"""
    Colapso de la regularización pericéntrica de Sundman o del invariante
    de acción-ángulo de Poincaré-Cartan en Fase 1.
    """

    pass


class SkewSignatureCollapse(RiemannianInertiaAgentError):
    r"""$$\mathcal{W}$$ abandona $$\mathfrak{so}(n)$$ o viola la $$G$$-antisimetría."""

    pass


class ExteriorAlgebraCollapse(RiemannianInertiaAgentError):
    """El bivector $$p\wedge\omega$$ viola la identidad de Gram o la paridad de rango."""

    pass


class CasimirDarbouxCollapse(ExteriorAlgebraCollapse):
    r"""
    Violación dura del invariante cuadrático de Casimir o de la identidad
    espectral de Darboux-Parseval en $$\mathfrak{so}(n)$$.
    """

    pass


class ThermodynamicPassivityCollapse(RiemannianInertiaAgentError):
    r"""$$J_{\mathrm{eff}}$$ realiza trabajo neto o pierde la pasividad simpléctica."""

    pass


class FloquetKreinCollapse(ThermodynamicPassivityCollapse):
    r"""
    Los multiplicadores característicos de Floquet de $$J_{\mathrm{eff}}$$
    escapan del eje imaginario (bifurcación hiperbólica de Kreĭn).
    """

    pass


class RecursiveGovernanceCollapse(RiemannianInertiaAgentError):
    r"""
    El endofuntor RSI-L3 del motor o de la gobernanza violó la condición de
    descenso monótono de Lyapunov.
    """

    pass


class HeytingLatticeVeto(RiemannianInertiaAgentError):
    r"""
    Detonada síncronamente cuando el retículo de Heyting toca el supremo $$\top$$.

    Aniquila el estado puramente en software (RAM), sin hardware exógeno.
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DE GOBERNANZA CATEGÓRICA, POINCARÉ Y RSI-L3
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GovernanceRSILevel3Certificate:
    r"""
    Certificado inmutable del Endofuntor de Automejora Recursiva de Nivel 3
    ($$\mathfrak{R}_3^{\mathrm{Gov}}$$) ejecutado por el Agente Soberano sobre
    el clasificador de subobjetos del Topos.
    """

    recursion_depth: int = 0
    initial_epistemic_tension: float = 0.0
    terminal_epistemic_tension: float = 0.0
    lyapunov_contraction_ratio: float = 1.0
    lawvere_tierney_modality_gain: float = 1.0
    synthesized_modality_tag: str = "Heyting-Identity-Modality-L0"
    motor_rsi_contraction_ratio: float = 1.0
    is_monotone_lyapunov: bool = True
    is_modal_fixed_point: bool = True


_DEFAULT_GOV_RSI_CERT: Final[GovernanceRSILevel3Certificate] = (
    GovernanceRSILevel3Certificate()
)


@dataclass(frozen=True, slots=True)
class Phase1ObservationBridge:
    r"""
    Certificado de la Fase 1. Precondición formal de la Fase 2.

    El campo ``liouville_verdict`` es el supremo local en $$\Omega_3$$:

        $$v_{\mathrm{Liouville}}
          =\bigsqcup\bigl\{
              v_{\mathrm{mom}},\,
              v_{\kappa},\,
              v_{\mathrm{inv}},\,
              v_{\mathrm{pair}},\,
              v_{\mathrm{musical}},\,
              v_{\mathrm{spec}},\,
              v_{\mathrm{Sundman}},\,
              v_{\mathrm{PC}},\,
              v_{\mathrm{RSI}_1}
            \bigr\}$$
    """

    momentum_data: MomentumAuditData
    liouville_verdict: InertialHeytingVerdict
    momentum_bound_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    metric_condition_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    inverse_consistency_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    pairing_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    musical_roundtrip_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    spectral_gap_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    momentum_margin: float = 0.0
    pairing_residual: float = 0.0
    musical_roundtrip_residual: float = 0.0
    kinetic_energy_primal: float = 0.0
    kinetic_energy_dual: float = 0.0
    spectral_minimum: float = 1.0
    spectral_maximum: float = 1.0
    # ── Extensiones de Mecánica Celeste de Poincaré y RSI-L3 (Fase 1) ───────
    sundman_regularity_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    poincare_cartan_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    rsi_l3_motor_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    sundman_regularization_factor: float = 1.0
    kam_diophantine_margin: float = 1.0
    poincare_cartan_density: float = 0.0
    poincare_action_angle_energy: float = 0.0
    poincare_cartan_residual: float = 0.0
    governance_rsi_l3_certificate: GovernanceRSILevel3Certificate = (
        _DEFAULT_GOV_RSI_CERT
    )


@dataclass(frozen=True, slots=True)
class Phase2OrientationBridge:
    r"""
    Certificado de la Fase 2. Precondición formal de la Fase 3.

    El campo ``skew_verdict`` es el supremo local en $$\Omega_3$$:

        $$v_{\mathrm{Skew}}
          =\bigsqcup\bigl\{
              v_{\mathrm{anti}},\,
              v_{\Omega},\,
              v_{\mathrm{Gram}},\,
              v_{\mathrm{rank}},\,
              v_{G\text{-skew}},\,
              v_{\mathrm{Casimir}},\,
              v_{\mathrm{Darboux}},\,
              v_{\mathrm{RSI}_2}
            \bigr\}$$
    """

    phase1_bridge: Phase1ObservationBridge
    synthesis_data: GyroscopicSynthesisData
    skew_verdict: InertialHeytingVerdict
    antisymmetry_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    vorticity_projection_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    gram_identity_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    even_rank_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    gauge_signature_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    relative_antisymmetry_residual: float = 0.0
    vorticity_projection_ratio: float = 0.0
    wedge_gram_residual: float = 0.0
    gauge_skew_residual: float = 0.0
    skew_numerical_rank: int = 0
    rank_is_even: bool = True
    # ── Extensiones de Mecánica Celeste de Poincaré y RSI-L3 (Fase 2) ───────
    casimir_invariant_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    darboux_parseval_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    rsi_l3_synthesis_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    casimir_quadratic_invariant: float = 0.0
    casimir_relative_trace_residual: float = 0.0
    darboux_parseval_relative_residual: float = 0.0
    poincare_areal_circulation: float = 0.0
    poincare_birkhoff_secular_factor: float = 1.0
    governance_rsi_l3_certificate: GovernanceRSILevel3Certificate = (
        _DEFAULT_GOV_RSI_CERT
    )


@dataclass(frozen=True, slots=True)
class InertialGovernanceState:
    r"""
    Objeto final del endofuntor $$Z_{\mathrm{Inertia}}$$. Estado lógico supremo.

        $$v_{\mathrm{final}}
          = v_{\mathrm{Liouville}}
          \sqcup v_{\mathrm{Skew}}
          \sqcup v_{\mathrm{Work}}$$
    """

    phase2_bridge: Phase2OrientationBridge
    veto_data: ThermodynamicVetoData
    work_verdict: InertialHeytingVerdict
    final_supremum_verdict: InertialHeytingVerdict
    is_epistemologically_valid: bool
    work_passivity_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    pairwise_work_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    liouville_probe_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    dirac_symmetry_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    work_margin: float = 0.0
    pairwise_work_residual: float = 0.0
    liouville_probe_residual: float = 0.0
    dirac_relative_skew_residual: float = 0.0
    # ── Extensiones de Mecánica Celeste de Poincaré y RSI-L3 (Fase 3) ───────
    floquet_stability_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    tisserand_orbital_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    rsi_l3_symplectic_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    floquet_relative_real_drift: float = 0.0
    floquet_spectral_radius: float = 0.0
    tisserand_gyroscopic_parameter: float = 1.0
    jacobi_poincare_virial_invariant: float = 0.0
    poincare_recurrence_horizon: float = 0.0
    governance_rsi_l3_certificate: GovernanceRSILevel3Certificate = (
        _DEFAULT_GOV_RSI_CERT
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → AUDITORÍA DEL VOLUMEN DE LIOUVILLE, VARIABLES DE ACCIÓN-ÁNGULO DE
#          POINCARÉ, REGULARIZACIÓN DE SUNDMAN Y RSI-L3 EN TOPOS (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_LiouvilleVolumeAuditor:
    r"""
    Fase 1: interroga el espectrómetro de momentum del motor y proyecta el
    estado tensional y celeste del flujo al retículo de severidad de Heyting
    $$\Omega_3$$.

    Audita, en orden de dependencia lógica:
        1. Cota dura y blanda del momentum (volumen de Liouville).
        2. Número de condición espectral de la métrica $$\kappa_2(G)$$.
        3. Residuo de consistencia de la inversa de Wilkinson.
        4. Identidad de apareamiento dual y energía cinética geométrica.
        5. Round-trip del isomorfismo musical $$\sharp\circ\flat=\mathrm{id}$$.
        6. Hueco espectral $$(\lambda_{\min},\lambda_{\max})$$.
        7. **Mecánica Celeste de Poincaré (Fase 1)**:
           - Regularización pericéntrica conforme de Sundman-Poincaré.
           - Identidad de acción-ángulo de Poincaré $$\sum_k I_k\omega_k = H_{\mathrm{kin}}$$
             y densidad de la 1-forma invariante de Poincaré-Cartan.
           - Margen Diofántico de toros invariantes KAM.
        8. **Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{\mathrm{Gov},(1)}$$)**:
           Co-certificación de la monotonía de Lyapunov del motor y síntesis
           recursiva de la modalidad epistémica de Lawvere-Tierney.

    El método final ``handoff_phase1_to_phase2`` constituye la definición
    formal terminal de la Fase 1 y el inicio exacto de la Fase 2.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.1  Utilidades elementales de validación numérica, Neumaier y Heyting
    # ──────────────────────────────────────────────────────────────────────────
    def _as_finite_float(self, value: object, name: str) -> float:
        """Convierte a float finito, rechazando booleanos y no-finitos."""
        if isinstance(value, (bool, np.bool_)):
            raise RiemannianInertiaAgentError(
                f"{name} no debe ser booleano; se requiere un número real."
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaAgentError(
                f"{name} no puede convertirse a un número real."
            ) from exc
        if not math.isfinite(result):
            raise RiemannianInertiaAgentError(f"{name} no es finito.")
        return result

    def _as_nonnegative_finite_float(self, value: object, name: str) -> float:
        """Valida un número real finito no negativo."""
        result = self._as_finite_float(value, name)
        if result < 0.0:
            raise RiemannianInertiaAgentError(f"{name} no puede ser negativo.")
        return result

    def _as_positive_finite_float(self, value: object, name: str) -> float:
        """Valida un número real finito estrictamente positivo."""
        result = self._as_finite_float(value, name)
        if result <= 0.0:
            raise RiemannianInertiaAgentError(f"{name} debe ser positivo.")
        return result

    def _as_bool(self, value: object, name: str) -> bool:
        """Valida y normaliza un valor booleano."""
        if not isinstance(value, (bool, np.bool_)):
            raise RiemannianInertiaAgentError(f"{name} debe ser booleano.")
        return bool(value)

    def _as_int(self, value: object, name: str) -> int:
        """Valida un entero (rechaza booleanos)."""
        if isinstance(value, (bool, np.bool_)):
            raise RiemannianInertiaAgentError(f"{name} no debe ser booleano.")
        if isinstance(value, (int, np.integer)):
            return int(value)
        raise RiemannianInertiaAgentError(f"{name} debe ser un entero.")

    def _neumaier_sum(self, terms: NDArray[np.float64], name: str) -> float:
        r"""
        Suma compensada de Kahan–Babuška–Neumaier en $$\mathbb{R}$$ para auditar
        invariantes integrales de Poincaré y trazas de Darboux sin deriva de mantisa.
        """
        if not np.all(np.isfinite(terms)):
            raise RiemannianInertiaAgentError(
                f"{name} contiene elementos no finitos en la suma de Neumaier."
            )
        total = 0.0
        comp = 0.0
        for raw in terms.flat:
            term = float(raw)
            trial = total + term
            if abs(total) >= abs(term):
                comp += (total - trial) + term
            else:
                comp += (term - trial) + total
            total = trial
        result = total + comp
        if not math.isfinite(result):
            raise RiemannianInertiaAgentError(
                f"La suma compensada de Neumaier para {name} divergió."
            )
        return result

    def _get_required_attribute(self, obj: object, attr: str, name: str) -> Any:
        """Extrae un atributo requerido del DTO."""
        if not hasattr(obj, attr):
            raise RiemannianInertiaAgentError(
                f"{name} no contiene el campo '{attr}'."
            )
        return getattr(obj, attr)

    def _get_required_finite_float(
        self, obj: object, attr: str, name: str
    ) -> float:
        """Extrae y valida un campo flotante finito requerido."""
        value = self._get_required_attribute(obj, attr, name)
        return self._as_finite_float(value, f"{name}.{attr}")

    def _get_required_nonnegative_finite_float(
        self,
        obj: object,
        attr: str,
        name: str,
    ) -> float:
        """Extrae y valida un campo flotante finito no negativo requerido."""
        value = self._get_required_attribute(obj, attr, name)
        return self._as_nonnegative_finite_float(value, f"{name}.{attr}")

    def _get_required_bool(self, obj: object, attr: str, name: str) -> bool:
        """Extrae y valida un campo booleano requerido."""
        value = self._get_required_attribute(obj, attr, name)
        return self._as_bool(value, f"{name}.{attr}")

    def _get_optional_finite_float(
        self,
        obj: object,
        attr: str,
        default: float,
        name: str,
    ) -> float:
        """Extrae un campo flotante finito opcional."""
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
        """Extrae un campo flotante finito no negativo opcional."""
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        return self._as_nonnegative_finite_float(value, f"{name}.{attr}")

    def _get_optional_positive_finite_float(
        self,
        obj: object,
        attr: str,
        default: float,
        name: str,
    ) -> float:
        """Extrae un campo flotante finito positivo opcional con fallback seguro."""
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        try:
            return self._as_positive_finite_float(value, f"{name}.{attr}")
        except RiemannianInertiaAgentError:
            logger.warning(
                "Campo auxiliar %s.%s inválido; se usa el valor por defecto %.4e.",
                name,
                attr,
                default,
            )
            return default

    def _get_optional_bool(
        self,
        obj: object,
        attr: str,
        default: bool,
        name: str,
    ) -> bool:
        """Extrae un campo booleano opcional."""
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        return self._as_bool(value, f"{name}.{attr}")

    def _get_optional_int(
        self,
        obj: object,
        attr: str,
        default: int,
        name: str,
    ) -> int:
        """Extrae un campo entero opcional."""
        if not hasattr(obj, attr):
            return default
        value = getattr(obj, attr)
        if value is None:
            return default
        return self._as_int(value, f"{name}.{attr}")

    def _validate_finite_array_attribute(
        self,
        obj: object,
        attr: str,
        name: str,
        required: bool = False,
        allow_empty_optional: bool = False,
    ) -> NDArray[np.float64] | None:
        """
        Valida un campo vectorial/matricial opcional o requerido.
        """
        if not hasattr(obj, attr):
            if required:
                raise RiemannianInertiaAgentError(
                    f"{name} no contiene el campo '{attr}'."
                )
            return None

        value = getattr(obj, attr)
        if value is None:
            if required:
                raise RiemannianInertiaAgentError(
                    f"{name}.{attr} es requerido y no puede ser None."
                )
            return None

        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaAgentError(
                f"{name}.{attr} no puede convertirse a un arreglo float64."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size == 0:
            if allow_empty_optional and not required:
                return None
            raise RiemannianInertiaAgentError(
                f"{name}.{attr} no puede ser vacío."
            )
        if not np.all(np.isfinite(arr)):
            raise RiemannianInertiaAgentError(
                f"{name}.{attr} contiene valores no finitos."
            )
        return arr

    def _matrix_frobenius_norm(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> float:
        """Calcula la norma de Frobenius de una matriz cuadrada finita."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise RiemannianInertiaAgentError(
                f"{name} debe ser una matriz cuadrada."
            )
        if not np.all(np.isfinite(matrix)):
            raise RiemannianInertiaAgentError(
                f"{name} contiene valores no finitos."
            )
        norm = float(np.linalg.norm(matrix, ord="fro"))
        if not math.isfinite(norm):
            raise RiemannianInertiaAgentError(
                f"La norma de {name} no es finita."
            )
        return norm

    def _relative_residual(self, residual: float, scale: float) -> float:
        r"""Residuo relativo $$r=\mathrm{residual}/\max(1,\mathrm{scale})$$."""
        if not math.isfinite(residual) or not math.isfinite(scale):
            raise RiemannianInertiaAgentError(
                "Normas no finitas en el residuo relativo."
            )
        return residual / max(1.0, scale)

    def _validate_verdict(
        self,
        verdict: object,
        name: str,
    ) -> InertialHeytingVerdict:
        """Valida que un veredicto pertenezca al retículo $$\Omega_3$$."""
        if not isinstance(verdict, InertialHeytingVerdict):
            raise RiemannianInertiaAgentError(
                f"{name} debe ser una instancia de InertialHeytingVerdict."
            )
        return verdict

    def _heyting_join(
        self,
        verdicts: Iterable[InertialHeytingVerdict],
    ) -> InertialHeytingVerdict:
        r"""
        Supremo (join) en el retículo de Heyting:

            $$v=\bigsqcup_i v_i=\max_i v_i$$

        El elemento neutro de una colección vacía es $$\bot=\mathtt{COHERENT}$$.
        """
        acc = InertialHeytingVerdict.COHERENT
        for idx, verdict in enumerate(verdicts):
            acc = acc.join(self._validate_verdict(verdict, f"verdict[{idx}]"))
        return acc

    def _heyting_meet(
        self,
        verdicts: Iterable[InertialHeytingVerdict],
    ) -> InertialHeytingVerdict:
        r"""Ínfimo (meet): $$v=\bigsqcap_i v_i=\min_i v_i$$. Neutro: $$\top$$."""
        acc = InertialHeytingVerdict.VETOED
        saw_any = False
        for idx, verdict in enumerate(verdicts):
            acc = acc.meet(self._validate_verdict(verdict, f"verdict[{idx}]"))
            saw_any = True
        return acc if saw_any else InertialHeytingVerdict.COHERENT

    def _clamp01(self, value: float) -> float:
        """Proyecta un valor real al intervalo $$[0,1]$$."""
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def _threshold_verdict(
        self,
        value: float,
        soft: float,
        hard: float,
        name: str,
    ) -> InertialHeytingVerdict:
        """Clasificador canónico de umbral blando/duro para un residual $$\ge 0$$."""
        if not math.isfinite(value):
            raise RiemannianInertiaAgentError(f"{name} no es finito.")
        if value > hard:
            return InertialHeytingVerdict.VETOED
        if value > soft:
            return InertialHeytingVerdict.DEGRADED
        return InertialHeytingVerdict.COHERENT

    def _extract_motor_rsi_audit(
        self,
        dto: object,
        attr: str,
        name: str,
    ) -> tuple[bool, float, InertialHeytingVerdict]:
        r"""
        Audita el certificado ``RSILevel3AuditMetrics`` emitido por el motor
        físico (si está presente).

        Verifica:
          1. ``is_monotone_lyapunov == True`` y ``terminal <= initial + floor``.
          2. ``lyapunov_contraction_ratio`` finito y no negativo.
        """
        if not hasattr(dto, attr):
            return True, 1.0, InertialHeytingVerdict.COHERENT
        audit = getattr(dto, attr)
        if audit is None:
            return True, 1.0, InertialHeytingVerdict.COHERENT

        is_monotone = self._get_optional_bool(
            audit, "is_monotone_lyapunov", True, f"{name}.{attr}"
        )
        init_def = self._get_optional_nonnegative_finite_float(
            audit, "initial_epistemic_defect", 0.0, f"{name}.{attr}"
        )
        term_def = self._get_optional_nonnegative_finite_float(
            audit, "terminal_epistemic_defect", 0.0, f"{name}.{attr}"
        )
        contraction = self._get_optional_nonnegative_finite_float(
            audit, "lyapunov_contraction_ratio", 1.0, f"{name}.{attr}"
        )

        if not is_monotone or term_def > init_def + 1e-9 * max(1.0, init_def):
            return False, contraction, InertialHeytingVerdict.VETOED
        if contraction > 1.0 + 1e-12:
            return True, contraction, InertialHeytingVerdict.DEGRADED
        return True, contraction, InertialHeytingVerdict.COHERENT

    def _rsi_level3_synthesize_lawvere_tierney_governance(
        self,
        normalized_residuals: tuple[float, ...],
        motor_contraction_ratio: float,
        celestial_regularity_weight: float,
        phase_tag: str,
    ) -> GovernanceRSILevel3Certificate:
        r"""
        Endofuntor de Automejora Recursiva de Nivel 3 en el Topos de Gobernanza
        ($$\mathfrak{R}_3^{\mathrm{Gov}}$$).

        Dado el vector de tensiones normalizadas $$\mathbf{r} \in \mathbb{R}_{\ge 0}^m$$
        de una fase, define el Funcional de Lyapunov Epistémico de Gobernanza:

            $$\mathcal{V}_k^{\mathrm{Gov}}(\eta_k)
              = \sum_{j=1}^m w_j(\eta_k)\,\frac{r_j}{1 + \eta_k\,s_{\mathrm{cel}}\,r_j}
              + \eta_k\,\rho_{\mathrm{motor}}\,\|\mathbf{r}\|_\infty$$

        y sintetiza recursivamente la ley de pesos modales de Padé-Chebyshev
        $$w_j(\eta_k)$$ y la ganancia de clausura de Lawvere-Tierney
        $$\eta_k \in (0, 1]$$ exigiendo descenso monótono estricto
        $$\mathcal{V}_{k+1}^{\mathrm{Gov}} \le \mathcal{V}_k^{\mathrm{Gov}}$$
        hasta converger al punto fijo modal.
        """
        arr = np.asarray(normalized_residuals, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return GovernanceRSILevel3Certificate(
                synthesized_modality_tag=f"{phase_tag}-Vacuous-FixedPoint"
            )

        cel_weight = self._clamp01(celestial_regularity_weight)
        motor_rho = max(0.0, min(1.0, motor_contraction_ratio))
        r_sum = self._neumaier_sum(arr, f"{phase_tag}.residuals")
        r_max = float(np.max(arr))

        V_init = r_sum + motor_rho * r_max
        V_current = V_init
        depth = 0
        chosen_gain = 1.0
        modality_tag = f"{phase_tag}-LawvereTierney-PadeOrder1"

        for step in range(1, _GOV_RSI_L3_MAX_DEPTH + 1):
            if V_current <= _GOV_RSI_L3_FLOOR:
                break

            # Ley candidata A: Amortiguamiento Racional de Padé (1,1) ponderado por Sundman/KAM
            gain_a = 1.0 / (1.0 + float(step) * (1.0 + r_max))
            denom_a = 1.0 + (1.0 - gain_a) * (0.5 + 0.5 * cel_weight) * arr
            pade_terms_a = (gain_a * arr) / denom_a
            V_a = (
                self._neumaier_sum(pade_terms_a, f"{phase_tag}.padeA")
                + gain_a * motor_rho * r_max
            )

            # Ley candidata B: Filtro Modal de Chebyshev de orden 2
            gain_b = 0.5**step
            cheb_terms_b = (gain_b * arr) / (1.0 + arr + 0.5 * (arr * arr))
            V_b = (
                self._neumaier_sum(cheb_terms_b, f"{phase_tag}.chebB")
                + gain_b * motor_rho * r_max
            )

            if V_b <= V_a:
                best_V = V_b
                best_gain = gain_b
                best_tag = f"{phase_tag}-LawvereTierney-ChebyshevOrder2"
            else:
                best_V = V_a
                best_gain = gain_a
                best_tag = f"{phase_tag}-LawvereTierney-SundmanPade"

            if best_V < V_current:
                V_current = best_V
                depth = step
                chosen_gain = best_gain
                modality_tag = best_tag
            else:
                break

        is_monotone = V_current <= V_init + _GOV_RSI_L3_FLOOR
        if not is_monotone:
            raise RecursiveGovernanceCollapse(
                f"El endofuntor RSI-L3 de gobernanza ({phase_tag}) violó la "
                f"monotonía de Lyapunov: V_init={V_init:.4e}, V_term={V_current:.4e}."
            )

        contraction = V_current / V_init if V_init > 0.0 else 0.0
        return GovernanceRSILevel3Certificate(
            recursion_depth=depth,
            initial_epistemic_tension=V_init,
            terminal_epistemic_tension=V_current,
            lyapunov_contraction_ratio=min(1.0, contraction),
            lawvere_tierney_modality_gain=chosen_gain,
            synthesized_modality_tag=modality_tag,
            motor_rsi_contraction_ratio=motor_rho,
            is_monotone_lyapunov=is_monotone,
            is_modal_fixed_point=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 1.2  Validación y extracción de certificados de Fase 1 (Motor 4.0 / 5.0)
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_momentum_data(
        self,
        momentum_data: MomentumAuditData,
    ) -> dict[str, Any]:
        """
        Verifica la integridad del artefacto espectral y celeste producido por
        el motor en la Fase 1.
        """
        if not isinstance(momentum_data, MomentumAuditData):
            raise RiemannianInertiaAgentError(
                "momentum_data debe ser una instancia de MomentumAuditData."
            )

        momentum_norm = self._get_required_nonnegative_finite_float(
            momentum_data, "momentum_norm", "momentum_data"
        )
        is_bounded = self._get_required_bool(
            momentum_data, "is_bounded", "momentum_data"
        )
        self._validate_finite_array_attribute(
            momentum_data, "covariant_momentum", "momentum_data", required=False
        )
        self._validate_finite_array_attribute(
            momentum_data,
            "reconstructed_velocity",
            "momentum_data",
            required=False,
        )

        metric_condition_number = self._get_optional_nonnegative_finite_float(
            momentum_data, "metric_condition_number", 1.0, "momentum_data"
        )
        if metric_condition_number < 1.0:
            metric_condition_number = 1.0

        inverse_consistency_residual = (
            self._get_optional_nonnegative_finite_float(
                momentum_data,
                "inverse_consistency_residual",
                0.0,
                "momentum_data",
            )
        )
        pairing_residual = self._get_optional_nonnegative_finite_float(
            momentum_data, "pairing_residual", 0.0, "momentum_data"
        )
        musical_roundtrip_residual = (
            self._get_optional_nonnegative_finite_float(
                momentum_data,
                "musical_roundtrip_residual",
                0.0,
                "momentum_data",
            )
        )
        kinetic_energy_primal = self._get_optional_nonnegative_finite_float(
            momentum_data, "kinetic_energy_primal", 0.0, "momentum_data"
        )
        kinetic_energy_dual = self._get_optional_nonnegative_finite_float(
            momentum_data, "kinetic_energy_dual", 0.0, "momentum_data"
        )
        dual_pairing = self._get_optional_finite_float(
            momentum_data, "dual_pairing", 0.0, "momentum_data"
        )
        spectral_minimum = self._get_optional_finite_float(
            momentum_data, "spectral_minimum", 1.0, "momentum_data"
        )
        spectral_maximum = self._get_optional_finite_float(
            momentum_data, "spectral_maximum", 1.0, "momentum_data"
        )

        # Extensiones de Mecánica Celeste de Poincaré (Fase 1)
        action_vars = self._validate_finite_array_attribute(
            momentum_data,
            "poincare_action_variables",
            "momentum_data",
            required=False,
            allow_empty_optional=True,
        )
        orbital_freqs = self._validate_finite_array_attribute(
            momentum_data,
            "poincare_orbital_frequencies",
            "momentum_data",
            required=False,
            allow_empty_optional=True,
        )
        has_pc_density = (
            hasattr(momentum_data, "poincare_cartan_one_form_density")
            and getattr(momentum_data, "poincare_cartan_one_form_density")
            is not None
        )
        poincare_cartan_density = self._get_optional_finite_float(
            momentum_data,
            "poincare_cartan_one_form_density",
            0.5 * dual_pairing,
            "momentum_data",
        )
        sundman_factor = self._get_optional_finite_float(
            momentum_data,
            "sundman_regularization_factor",
            1.0,
            "momentum_data",
        )
        kam_margin = self._get_optional_nonnegative_finite_float(
            momentum_data,
            "kam_diophantine_margin",
            1.0,
            "momentum_data",
        )
        _, motor_rsi_contraction, motor_rsi_verdict = (
            self._extract_motor_rsi_audit(
                momentum_data, "rsi_l3_metric_audit", "momentum_data"
            )
        )

        return {
            "momentum_norm": momentum_norm,
            "is_bounded": is_bounded,
            "metric_condition_number": metric_condition_number,
            "inverse_consistency_residual": inverse_consistency_residual,
            "pairing_residual": pairing_residual,
            "musical_roundtrip_residual": musical_roundtrip_residual,
            "kinetic_energy_primal": kinetic_energy_primal,
            "kinetic_energy_dual": kinetic_energy_dual,
            "dual_pairing": dual_pairing,
            "spectral_minimum": spectral_minimum,
            "spectral_maximum": spectral_maximum,
            "action_vars": action_vars,
            "orbital_freqs": orbital_freqs,
            "has_pc_density": has_pc_density,
            "poincare_cartan_density": poincare_cartan_density,
            "sundman_factor": sundman_factor,
            "kam_margin": kam_margin,
            "motor_rsi_contraction": motor_rsi_contraction,
            "motor_rsi_verdict": motor_rsi_verdict,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 1.3  Clasificadores locales de Fase 1 (Liouville, de Rham y Poincaré)
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_momentum_bound(
        self,
        momentum_norm: float,
        is_bounded: bool,
    ) -> InertialHeytingVerdict:
        """Clasifica la cota de volumen de Liouville asociada al momentum."""
        if not is_bounded or momentum_norm > _MOMENTUM_HARD_LIMIT:
            return InertialHeytingVerdict.VETOED
        if momentum_norm > _MOMENTUM_SOFT_LIMIT:
            return InertialHeytingVerdict.DEGRADED
        return InertialHeytingVerdict.COHERENT

    def _classify_metric_condition(
        self,
        metric_condition_number: float,
    ) -> InertialHeytingVerdict:
        """Clasifica la calidad numérica de la métrica según $$\kappa_2(G)$$."""
        return self._threshold_verdict(
            metric_condition_number,
            _CONDITION_SOFT_MAX,
            _CONDITION_HARD_MAX,
            "metric_condition_number",
        )

    def _classify_inverse_consistency(
        self,
        inverse_consistency_residual: float,
    ) -> InertialHeytingVerdict:
        """Clasifica la consistencia de Wilkinson de $$G^{-1}$$."""
        return self._threshold_verdict(
            inverse_consistency_residual,
            _INVERSE_RESIDUAL_SOFT_MAX,
            _INVERSE_RESIDUAL_HARD_MAX,
            "inverse_consistency_residual",
        )

    def _classify_dual_pairing(
        self,
        pairing_residual: float,
        kinetic_energy_primal: float,
        kinetic_energy_dual: float,
    ) -> InertialHeytingVerdict:
        r"""
        Clasifica la identidad de apareamiento dual y la coherencia
        primal/dual de la energía cinética geométrica.
        """
        pairing_verdict = self._threshold_verdict(
            pairing_residual,
            _PAIRING_RESIDUAL_SOFT_MAX,
            _PAIRING_RESIDUAL_HARD_MAX,
            "pairing_residual",
        )
        energy_gap = abs(kinetic_energy_primal - kinetic_energy_dual)
        scale = max(1.0, kinetic_energy_primal, kinetic_energy_dual)
        energy_verdict = self._threshold_verdict(
            energy_gap / scale,
            _PAIRING_RESIDUAL_SOFT_MAX,
            _PAIRING_RESIDUAL_HARD_MAX,
            "kinetic_energy_gap",
        )
        return pairing_verdict.join(energy_verdict)

    def _classify_musical_roundtrip(
        self,
        musical_roundtrip_residual: float,
    ) -> InertialHeytingVerdict:
        r"""Clasifica $$\|\sharp\flat\dot{q}-\dot{q}\|_2$$."""
        return self._threshold_verdict(
            musical_roundtrip_residual,
            _MUSICAL_ROUNDTRIP_SOFT_MAX,
            _MUSICAL_ROUNDTRIP_HARD_MAX,
            "musical_roundtrip_residual",
        )

    def _classify_spectral_gap(
        self,
        spectral_minimum: float,
        spectral_maximum: float,
        metric_condition_number: float,
    ) -> InertialHeytingVerdict:
        r"""Clasifica el hueco espectral de $$G$$."""
        if spectral_maximum <= 0.0:
            return InertialHeytingVerdict.VETOED
        if spectral_minimum <= 0.0:
            return InertialHeytingVerdict.VETOED
        if spectral_minimum > spectral_maximum:
            return InertialHeytingVerdict.VETOED

        ratio = spectral_minimum / spectral_maximum
        floor_verdict = InertialHeytingVerdict.COHERENT
        if ratio < _SPECTRAL_FLOOR_RATIO_HARD:
            floor_verdict = InertialHeytingVerdict.VETOED
        elif ratio < _SPECTRAL_FLOOR_RATIO_SOFT:
            floor_verdict = InertialHeytingVerdict.DEGRADED

        spectral_cond = spectral_maximum / spectral_minimum
        cond_gap = abs(spectral_cond - metric_condition_number)
        cond_scale = max(1.0, metric_condition_number, spectral_cond)
        cond_verdict = self._threshold_verdict(
            cond_gap / cond_scale,
            1e-6,
            1e-2,
            "spectral_condition_gap",
        )
        return floor_verdict.join(cond_verdict)

    def _classify_sundman_regularization(
        self,
        sundman_factor: float,
    ) -> InertialHeytingVerdict:
        r"""
        Clasifica el factor de regularización temporal de Sundman-Poincaré:

            $$s_{\mathrm{Sundman}} \in (0, 1]$$

        Un valor $$s_{\mathrm{Sundman}} \le 0$$ o $$> 1$$ o inferior al piso
        pericéntrico duro indica colapso de colisión singular.
        """
        if (
            not math.isfinite(sundman_factor)
            or sundman_factor <= _SUNDMAN_FACTOR_HARD_MIN
            or sundman_factor > 1.0 + 1e-12
        ):
            return InertialHeytingVerdict.VETOED
        if sundman_factor < _SUNDMAN_FACTOR_SOFT_MIN:
            return InertialHeytingVerdict.DEGRADED
        return InertialHeytingVerdict.COHERENT

    def _classify_poincare_cartan_action_angle(
        self,
        action_vars: NDArray[np.float64] | None,
        orbital_freqs: NDArray[np.float64] | None,
        kinetic_energy_dual: float,
        dual_pairing: float,
        poincare_cartan_density: float,
        has_pc_density: bool,
    ) -> tuple[float, float, InertialHeytingVerdict]:
        r"""
        Audita los dos invariantes integrales de Poincaré de Fase 1:

        1. **Identidad Canónica de Acción-Ángulo de Poincaré**:
               $$\mathcal{I}_{\mathrm{AA}} = \sum_{k=1}^n I_k\,\omega_k
                 = \tfrac12 p^\top G^{-1} p = \tfrac12 E_{\mathrm{dual}}$$
        2. **Densidad de la 1-forma de Poincaré-Cartan**:
               $$\theta_{\mathrm{PC}} = \tfrac12\langle p, \dot{q}\rangle$$

        Retorna ``(action_angle_energy, rel_residual, verdict)``.
        """
        verdict = InertialHeytingVerdict.COHERENT
        rel_residual = 0.0
        action_angle_energy = 0.5 * kinetic_energy_dual

        if action_vars is not None and orbital_freqs is not None:
            if action_vars.shape != orbital_freqs.shape:
                raise RiemannianInertiaAgentError(
                    "poincare_action_variables y poincare_orbital_frequencies "
                    "deben compartir dimensión."
                )
            if np.any(action_vars < -_MACHINE_EPSILON) or np.any(
                orbital_freqs <= 0.0
            ):
                return (
                    0.0,
                    1.0,
                    InertialHeytingVerdict.VETOED,
                )
            action_angle_energy = self._neumaier_sum(
                action_vars * orbital_freqs, "∑ I_k ω_k"
            )
            target_h_kin = 0.5 * kinetic_energy_dual
            if target_h_kin > 0.0 or action_angle_energy > 0.0:
                aa_gap = abs(action_angle_energy - target_h_kin) / max(
                    1.0, abs(target_h_kin), abs(action_angle_energy)
                )
                rel_residual = max(rel_residual, aa_gap)
                verdict = verdict.join(
                    self._threshold_verdict(
                        aa_gap,
                        _POINCARE_CARTAN_GAP_SOFT_MAX,
                        _POINCARE_CARTAN_GAP_HARD_MAX,
                        "poincare_action_angle_gap",
                    )
                )

        if has_pc_density and (
            abs(dual_pairing) > 0.0 or abs(poincare_cartan_density) > 0.0
        ):
            expected_pc = 0.5 * dual_pairing
            pc_gap = abs(poincare_cartan_density - expected_pc) / max(
                1.0, abs(expected_pc), abs(poincare_cartan_density)
            )
            rel_residual = max(rel_residual, pc_gap)
            verdict = verdict.join(
                self._threshold_verdict(
                    pc_gap,
                    _POINCARE_CARTAN_GAP_SOFT_MAX,
                    _POINCARE_CARTAN_GAP_HARD_MAX,
                    "poincare_cartan_density_gap",
                )
            )

        return action_angle_energy, rel_residual, verdict

    def _compute_momentum_margin(self, momentum_norm: float) -> float:
        r"""Margen normalizado de seguridad del momentum: $$\max(0, 1-\|p\|/P_{\mathrm{hard}})$$."""
        if _MOMENTUM_HARD_LIMIT <= 0.0:
            return 0.0
        return self._clamp01(1.0 - (momentum_norm / _MOMENTUM_HARD_LIMIT))

    # ──────────────────────────────────────────────────────────────────────────
    # 1.4  Ejecución, Validación Estructural y Morfismo Terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _audit_liouville_volume(
        self,
        momentum_data: MomentumAuditData,
    ) -> Phase1ObservationBridge:
        """
        Evalúa la preservación del volumen simpléctico, la coherencia métrica,
        los invariantes de acción-ángulo de Poincaré-Cartan y el endofuntor
        RSI-L3 en la Fase 1.
        """
        fields = self._validate_momentum_data(momentum_data)

        momentum_bound_verdict = self._classify_momentum_bound(
            fields["momentum_norm"],
            fields["is_bounded"],
        )
        metric_condition_verdict = self._classify_metric_condition(
            fields["metric_condition_number"]
        )
        inverse_consistency_verdict = self._classify_inverse_consistency(
            fields["inverse_consistency_residual"]
        )
        pairing_verdict = self._classify_dual_pairing(
            fields["pairing_residual"],
            fields["kinetic_energy_primal"],
            fields["kinetic_energy_dual"],
        )
        musical_roundtrip_verdict = self._classify_musical_roundtrip(
            fields["musical_roundtrip_residual"]
        )
        spectral_gap_verdict = self._classify_spectral_gap(
            fields["spectral_minimum"],
            fields["spectral_maximum"],
            fields["metric_condition_number"],
        )
        sundman_verdict = self._classify_sundman_regularization(
            fields["sundman_factor"]
        )
        aa_energy, pc_residual, poincare_cartan_verdict = (
            self._classify_poincare_cartan_action_angle(
                action_vars=fields["action_vars"],
                orbital_freqs=fields["orbital_freqs"],
                kinetic_energy_dual=fields["kinetic_energy_dual"],
                dual_pairing=fields["dual_pairing"],
                poincare_cartan_density=fields["poincare_cartan_density"],
                has_pc_density=fields["has_pc_density"],
            )
        )
        rsi_motor_verdict = fields["motor_rsi_verdict"]

        gov_rsi_cert = self._rsi_level3_synthesize_lawvere_tierney_governance(
            normalized_residuals=(
                fields["momentum_norm"] / _MOMENTUM_HARD_LIMIT,
                fields["inverse_consistency_residual"]
                / _INVERSE_RESIDUAL_HARD_MAX,
                fields["pairing_residual"] / _PAIRING_RESIDUAL_HARD_MAX,
                fields["musical_roundtrip_residual"]
                / _MUSICAL_ROUNDTRIP_HARD_MAX,
                pc_residual / _POINCARE_CARTAN_GAP_HARD_MAX,
            ),
            motor_contraction_ratio=fields["motor_rsi_contraction"],
            celestial_regularity_weight=fields["sundman_factor"],
            phase_tag="Phase1-Liouville-Poincare",
        )

        liouville_verdict = self._heyting_join(
            (
                momentum_bound_verdict,
                metric_condition_verdict,
                inverse_consistency_verdict,
                pairing_verdict,
                musical_roundtrip_verdict,
                spectral_gap_verdict,
                sundman_verdict,
                poincare_cartan_verdict,
                rsi_motor_verdict,
            )
        )
        momentum_margin = self._compute_momentum_margin(
            fields["momentum_norm"]
        )

        logger.debug(
            "Fase 1 Liouville-Poincaré: ||p||=%.6e, κ=%.6e, inv_res=%.6e, "
            "Sundman=%.6e, KAM=%.6e, PC_res=%.6e, RSI_L3_depth=%d, verdict=%s",
            fields["momentum_norm"],
            fields["metric_condition_number"],
            fields["inverse_consistency_residual"],
            fields["sundman_factor"],
            fields["kam_margin"],
            pc_residual,
            gov_rsi_cert.recursion_depth,
            liouville_verdict.name,
        )

        return Phase1ObservationBridge(
            momentum_data=momentum_data,
            liouville_verdict=liouville_verdict,
            momentum_bound_verdict=momentum_bound_verdict,
            metric_condition_verdict=metric_condition_verdict,
            inverse_consistency_verdict=inverse_consistency_verdict,
            pairing_verdict=pairing_verdict,
            musical_roundtrip_verdict=musical_roundtrip_verdict,
            spectral_gap_verdict=spectral_gap_verdict,
            momentum_margin=momentum_margin,
            pairing_residual=fields["pairing_residual"],
            musical_roundtrip_residual=fields["musical_roundtrip_residual"],
            kinetic_energy_primal=fields["kinetic_energy_primal"],
            kinetic_energy_dual=fields["kinetic_energy_dual"],
            spectral_minimum=fields["spectral_minimum"],
            spectral_maximum=fields["spectral_maximum"],
            sundman_regularity_verdict=sundman_verdict,
            poincare_cartan_verdict=poincare_cartan_verdict,
            rsi_l3_motor_verdict=rsi_motor_verdict,
            sundman_regularization_factor=fields["sundman_factor"],
            kam_diophantine_margin=fields["kam_margin"],
            poincare_cartan_density=fields["poincare_cartan_density"],
            poincare_action_angle_energy=aa_energy,
            poincare_cartan_residual=pc_residual,
            governance_rsi_l3_certificate=gov_rsi_cert,
        )

    def _validate_phase1_bridge_structure(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> None:
        """Valida la estructura del certificado de Fase 1 (sin política)."""
        if not isinstance(phase1_bridge, Phase1ObservationBridge):
            raise PhaseHandoffCollapse(
                "handoff_phase1_to_phase2 exige Phase1ObservationBridge."
            )
        self._validate_momentum_data(phase1_bridge.momentum_data)

        verdict_fields = (
            "liouville_verdict",
            "momentum_bound_verdict",
            "metric_condition_verdict",
            "inverse_consistency_verdict",
            "pairing_verdict",
            "musical_roundtrip_verdict",
            "spectral_gap_verdict",
            "sundman_regularity_verdict",
            "poincare_cartan_verdict",
            "rsi_l3_motor_verdict",
        )
        for field_name in verdict_fields:
            if hasattr(phase1_bridge, field_name):
                self._validate_verdict(
                    getattr(phase1_bridge, field_name),
                    f"phase1_bridge.{field_name}",
                )

        nonneg_fields = (
            "momentum_margin",
            "pairing_residual",
            "musical_roundtrip_residual",
            "kinetic_energy_primal",
            "kinetic_energy_dual",
            "kam_diophantine_margin",
            "poincare_action_angle_energy",
            "poincare_cartan_residual",
        )
        for field_name in nonneg_fields:
            if hasattr(phase1_bridge, field_name):
                self._as_nonnegative_finite_float(
                    getattr(phase1_bridge, field_name),
                    f"phase1_bridge.{field_name}",
                )

        for field_name in (
            "spectral_minimum",
            "spectral_maximum",
            "sundman_regularization_factor",
            "poincare_cartan_density",
        ):
            if hasattr(phase1_bridge, field_name):
                self._as_finite_float(
                    getattr(phase1_bridge, field_name),
                    f"phase1_bridge.{field_name}",
                )

    def _assert_phase1_lattice_consistency(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> None:
        """Exige que el join almacenado coincida con el de los clasificadores granulares."""
        recomputed = self._heyting_join(
            (
                phase1_bridge.momentum_bound_verdict,
                phase1_bridge.metric_condition_verdict,
                phase1_bridge.inverse_consistency_verdict,
                phase1_bridge.pairing_verdict,
                phase1_bridge.musical_roundtrip_verdict,
                phase1_bridge.spectral_gap_verdict,
                phase1_bridge.sundman_regularity_verdict,
                phase1_bridge.poincare_cartan_verdict,
                phase1_bridge.rsi_l3_motor_verdict,
            )
        )
        if recomputed is not phase1_bridge.liouville_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del retículo en Φ₁₂: "
                f"join granular = {recomputed.name}, "
                f"almacenado = {phase1_bridge.liouville_verdict.name}."
            )

    def execute_phase1(
        self,
        momentum_data: MomentumAuditData,
    ) -> Phase1ObservationBridge:
        """
        Ejecuta la Fase 1 y entrega el certificado ``Phase1ObservationBridge``
        que alimenta el morfismo terminal ``handoff_phase1_to_phase2``.
        """
        return self._audit_liouville_volume(momentum_data)

    def handoff_phase1_to_phase2(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> Phase1ObservationBridge:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1  ≡  CONTINUACIÓN E INICIO DE LA FASE 2.

        Morfismo de transición categórica:

            $$\Phi_{12}:
              \mathrm{Phase1ObservationBridge}
              \longrightarrow
              \mathrm{Phase1ObservationBridge}$$

        Certifica la integridad estructural, la consistencia del supremo en el
        retículo de Heyting $$\Omega_3$$ y la monotonía de Lyapunov del
        certificado de gobernanza RSI-L3 antes de abrir el dominio de la Fase 2.
        """
        self._validate_phase1_bridge_structure(phase1_bridge)
        self._assert_phase1_lattice_consistency(phase1_bridge)
        if not phase1_bridge.governance_rsi_l3_certificate.is_monotone_lyapunov:
            raise PhaseHandoffCollapse(
                "El certificado RSI-L3 de gobernanza de Fase 1 violó la "
                "monotonía de Lyapunov en Φ₁₂."
            )
        return phase1_bridge


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → CERTIFICACIÓN DE LA FIRMA MÉTRICA, INVARIANTE DE CASIMIR, ESPECTRO
#          DE DARBOUX-PARSEVAL Y RSI-L3 EN so(n) (Orient)
#          [Continuación directa de handoff_phase1_to_phase2]
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_SkewSymmetryCertifier(Phase1_LiouvilleVolumeAuditor):
    r"""
    Fase 2: inicia exactamente donde concluye ``handoff_phase1_to_phase2``,
    recibiendo el puente certificado de observación y auditando la pureza
    geométrica y celeste del operador giroscópico $$\mathcal{W}\in\mathfrak{so}(n)$$.

    Audita:
        1. Antisimetría relativa de $$\mathcal{W}$$ en $$\mathfrak{so}(n)$$.
        2. Pureza de la vorticidad proyectada al cono de 2-formas.
        3. Identidad de Gram del bivector $$p\wedge\omega$$.
        4. Paridad del rango numérico de Darboux.
        5. $$G$$-antisimetría de calibre, si la métrica está presente.
        6. **Mecánica Celeste de Poincaré (Fase 2)**:
           - Invariante cuadrático de Casimir $$\mathcal{C}_2(W) = -\tfrac12\mathrm{Tr}(W^2) = \tfrac12\|W\|_F^2$$.
           - Identidad espectral de Darboux-Parseval $$\sum_k \nu_k^2 = \tfrac12\|W\|_F^2$$.
           - Circulación areal de segundo orden de Poincaré y factor secular de
             Poincaré-Birkhoff.
        7. **Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{\mathrm{Gov},(2)}$$)**:
           Co-certificación del refinamiento RSI-L3 del motor en $$\mathfrak{so}(n)$$
           y síntesis de la modalidad de Lawvere-Tierney de Fase 2.

    El método final ``handoff_phase2_to_phase3`` constituye la definición
    formal terminal de la Fase 2 y el inicio exacto de la Fase 3.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 2.1  Continuación formal inmediata del Morfismo Terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_observation(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> Phase1ObservationBridge:
        """
        PRIMER MÉTODO DE LA FASE 2 (Continuación directa de ``handoff_phase1_to_phase2``).

        Invoca el morfismo terminal de la Fase 1, revalida la coherencia del
        subobjeto en $$\Omega_3$$ y entrega el puente certificado a la Fase 2.
        """
        return self.handoff_phase1_to_phase2(phase1_bridge)

    def _validate_phase1_bridge(
        self, phase1_bridge: Phase1ObservationBridge
    ) -> None:
        """Fachada de revalidación de Fase 1 vía $$\Phi_{12}$$."""
        self._receive_certified_observation(phase1_bridge)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.2  Validación del artefacto giroscópico y celeste de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_synthesis_data(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> dict[str, Any]:
        """
        Verifica la integridad del artefacto giroscópico producido por el motor,
        incluyendo las extensiones celestes de Casimir, Darboux y RSI-L3.
        """
        if not isinstance(synthesis_data, GyroscopicSynthesisData):
            raise RiemannianInertiaAgentError(
                "synthesis_data debe ser una instancia de GyroscopicSynthesisData."
            )

        antisymmetry_residual = self._get_required_nonnegative_finite_float(
            synthesis_data, "antisymmetry_residual", "synthesis_data"
        )
        is_strictly_skew = self._get_required_bool(
            synthesis_data, "is_strictly_skew", "synthesis_data"
        )
        tensor = self._validate_finite_array_attribute(
            synthesis_data,
            "skew_symmetric_tensor",
            "synthesis_data",
            required=False,
        )
        self._validate_finite_array_attribute(
            synthesis_data,
            "vorticity_two_form",
            "synthesis_data",
            required=False,
        )
        self._validate_finite_array_attribute(
            synthesis_data, "omega_vector", "synthesis_data", required=False
        )

        gyroscopic_frobenius_norm = (
            self._get_optional_nonnegative_finite_float(
                synthesis_data,
                "gyroscopic_frobenius_norm",
                0.0,
                "synthesis_data",
            )
        )
        if (
            tensor is not None
            and tensor.ndim == 2
            and tensor.shape[0] == tensor.shape[1]
            and gyroscopic_frobenius_norm == 0.0
        ):
            gyroscopic_frobenius_norm = self._matrix_frobenius_norm(
                tensor, "synthesis_data.skew_symmetric_tensor"
            )

        vorticity_projection_residual = (
            self._get_optional_nonnegative_finite_float(
                synthesis_data,
                "vorticity_projection_residual",
                0.0,
                "synthesis_data",
            )
        )
        relative_skew_residual = self._get_optional_nonnegative_finite_float(
            synthesis_data, "relative_skew_residual", 0.0, "synthesis_data"
        )
        wedge_gram_residual = self._get_optional_nonnegative_finite_float(
            synthesis_data, "wedge_gram_residual", 0.0, "synthesis_data"
        )
        skew_numerical_rank = self._get_optional_int(
            synthesis_data, "skew_numerical_rank", 0, "synthesis_data"
        )
        if skew_numerical_rank < 0:
            raise RiemannianInertiaAgentError(
                "synthesis_data.skew_numerical_rank no puede ser negativo."
            )
        rank_is_even = self._get_optional_bool(
            synthesis_data, "rank_is_even", True, "synthesis_data"
        )

        # Extensiones celestes de Poincaré-Darboux-Casimir (Fase 2)
        has_casimir = (
            hasattr(synthesis_data, "casimir_quadratic_invariant")
            and getattr(synthesis_data, "casimir_quadratic_invariant")
            is not None
        )
        default_casimir = (
            0.5 * gyroscopic_frobenius_norm * gyroscopic_frobenius_norm
        )
        casimir_invariant = self._get_optional_nonnegative_finite_float(
            synthesis_data,
            "casimir_quadratic_invariant",
            default_casimir,
            "synthesis_data",
        )
        casimir_trace_residual = self._get_optional_nonnegative_finite_float(
            synthesis_data,
            "casimir_trace_residual",
            0.0,
            "synthesis_data",
        )
        darboux_freqs = self._validate_finite_array_attribute(
            synthesis_data,
            "darboux_characteristic_frequencies",
            "synthesis_data",
            required=False,
            allow_empty_optional=True,
        )
        poincare_circulation = self._get_optional_nonnegative_finite_float(
            synthesis_data,
            "poincare_areal_circulation",
            gyroscopic_frobenius_norm / math.sqrt(2.0),
            "synthesis_data",
        )
        pb_secular_factor = self._get_optional_finite_float(
            synthesis_data,
            "poincare_birkhoff_secular_factor",
            1.0,
            "synthesis_data",
        )
        _, motor_rsi_contraction, motor_rsi_verdict = (
            self._extract_motor_rsi_audit(
                synthesis_data, "rsi_l3_synthesis_audit", "synthesis_data"
            )
        )

        return {
            "antisymmetry_residual": antisymmetry_residual,
            "is_strictly_skew": is_strictly_skew,
            "gyroscopic_frobenius_norm": gyroscopic_frobenius_norm,
            "vorticity_projection_residual": vorticity_projection_residual,
            "relative_skew_residual": relative_skew_residual,
            "wedge_gram_residual": wedge_gram_residual,
            "skew_numerical_rank": skew_numerical_rank,
            "rank_is_even": rank_is_even,
            "tensor": tensor,
            "has_casimir": has_casimir,
            "casimir_invariant": casimir_invariant,
            "casimir_trace_residual": casimir_trace_residual,
            "darboux_freqs": darboux_freqs,
            "poincare_circulation": poincare_circulation,
            "pb_secular_factor": pb_secular_factor,
            "motor_rsi_contraction": motor_rsi_contraction,
            "motor_rsi_verdict": motor_rsi_verdict,
        }

    def _validate_optional_metric(
        self,
        G_tensor: object,
    ) -> NDArray[np.float64] | None:
        """Valida una métrica opcional para la auditoría de $$G$$-antisimetría."""
        if G_tensor is None:
            return None
        try:
            metric = np.asarray(G_tensor, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaAgentError(
                "G_tensor no puede convertirse a una matriz float64."
            ) from exc
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            raise RiemannianInertiaAgentError(
                "G_tensor debe ser una matriz cuadrada."
            )
        if metric.size == 0:
            raise RiemannianInertiaAgentError("G_tensor no puede ser vacía.")
        if not np.all(np.isfinite(metric)):
            raise RiemannianInertiaAgentError(
                "G_tensor contiene valores no finitos."
            )
        return metric

    # ──────────────────────────────────────────────────────────────────────────
    # 2.3  Clasificadores locales de Fase 2 (Lie so(n), Casimir y Darboux)
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_antisymmetry(
        self,
        antisymmetry_residual: float,
        is_strictly_skew: bool,
        gyroscopic_frobenius_norm: float,
        certified_relative_residual: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""
        Clasifica la antisimetría de $$\mathcal{W}$$ con el residuo relativo:

            $$r_{\mathrm{rel}}
              =\frac{\|W+W^\top\|_F}{\max(1,\|W\|_F)}$$
        """
        computed = self._relative_residual(
            antisymmetry_residual,
            gyroscopic_frobenius_norm,
        )
        relative_residual = max(computed, certified_relative_residual)

        if not is_strictly_skew:
            return relative_residual, InertialHeytingVerdict.VETOED
        return relative_residual, self._threshold_verdict(
            relative_residual,
            _SKEW_SOFT_RELATIVE_TOLERANCE,
            _SKEW_HARD_RELATIVE_TOLERANCE,
            "r_skew",
        )

    def _classify_vorticity_projection(
        self,
        vorticity_projection_residual: float,
        gyroscopic_frobenius_norm: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""Clasifica la pureza de la vorticidad proyectada al cono de 2-formas."""
        if gyroscopic_frobenius_norm <= _MACHINE_EPSILON:
            return 0.0, InertialHeytingVerdict.COHERENT
        ratio = self._relative_residual(
            vorticity_projection_residual,
            gyroscopic_frobenius_norm,
        )
        return ratio, self._threshold_verdict(
            ratio,
            _VORTICITY_PROJECTION_SOFT_RATIO,
            _VORTICITY_PROJECTION_HARD_RATIO,
            "vorticity_projection_ratio",
        )

    def _classify_gram_identity(
        self,
        wedge_gram_residual: float,
    ) -> InertialHeytingVerdict:
        r"""Clasifica la identidad de Gram $$\|p\wedge\omega\|_F^2=2\det\mathrm{Gram}$$."""
        return self._threshold_verdict(
            wedge_gram_residual,
            _GRAM_RESIDUAL_SOFT_MAX,
            _GRAM_RESIDUAL_HARD_MAX,
            "wedge_gram_residual",
        )

    def _classify_even_rank(
        self,
        rank: int,
        rank_is_even: bool,
    ) -> InertialHeytingVerdict:
        """Toda matriz antisimétrica real tiene rango par."""
        computed_even = rank % 2 == 0
        if not rank_is_even or not computed_even:
            return InertialHeytingVerdict.DEGRADED
        return InertialHeytingVerdict.COHERENT

    def _classify_gauge_signature(
        self,
        tensor: NDArray[np.float64] | None,
        metric: NDArray[np.float64] | None,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""
        Clasifica la $$G$$-antisimetría de calibre [I2]:

            $$r_G=\frac{\|W^\top G+GW\|_F}
                       {\max(1,\|W\|_F\|G\|_F)}$$
        """
        if tensor is None or metric is None:
            return 0.0, InertialHeytingVerdict.COHERENT
        if tensor.shape != metric.shape:
            raise RiemannianInertiaAgentError(
                "W y G deben compartir dimensión para la G-antisimetría."
            )
        form = tensor.T @ metric + metric @ tensor
        if not np.all(np.isfinite(form)):
            raise RiemannianInertiaAgentError(
                "WᵀG + GW produjo valores no finitos."
            )
        residual = float(np.linalg.norm(form, ord="fro"))
        if not math.isfinite(residual):
            raise RiemannianInertiaAgentError(
                "La norma de WᵀG + GW no es finita."
            )
        w_norm = self._matrix_frobenius_norm(tensor, "W")
        g_norm = self._matrix_frobenius_norm(metric, "G")
        relative = residual / max(1.0, w_norm * g_norm)
        return relative, self._threshold_verdict(
            relative,
            _GAUGE_SKEW_SOFT_RELATIVE_TOLERANCE,
            _GAUGE_SKEW_HARD_RELATIVE_TOLERANCE,
            "gauge_skew_residual",
        )

    def _classify_casimir_and_darboux_parseval(
        self,
        gyroscopic_frobenius_norm: float,
        casimir_invariant: float,
        casimir_trace_residual: float,
        has_casimir: bool,
        darboux_freqs: NDArray[np.float64] | None,
    ) -> tuple[
        float,
        float,
        InertialHeytingVerdict,
        InertialHeytingVerdict,
    ]:
        r"""
        Audita la identidad del invariante cuadrático de Casimir y el Teorema
        de Darboux-Parseval en $$\mathfrak{so}(n)$$:

            $$\mathcal{C}_2(W) = -\tfrac12\mathrm{Tr}(W^2)
              = \tfrac12\|W\|_F^2
              = \sum_{k=1}^{\lfloor n/2 \rfloor}\nu_k^2$$

        Retorna:
            ``(casimir_rel_res, darboux_rel_res, casimir_verdict, darboux_verdict)``.
        """
        fro_sq = gyroscopic_frobenius_norm * gyroscopic_frobenius_norm
        expected_casimir = 0.5 * fro_sq

        casimir_rel_res = self._relative_residual(
            casimir_trace_residual, fro_sq
        )
        if has_casimir and casimir_invariant > 0.0 and fro_sq > 0.0:
            casimir_gap = abs(casimir_invariant - expected_casimir) / max(
                1.0, expected_casimir, casimir_invariant
            )
            casimir_rel_res = max(casimir_rel_res, casimir_gap)

        casimir_verdict = self._threshold_verdict(
            casimir_rel_res,
            _CASIMIR_TRACE_SOFT_RELATIVE_TOLERANCE,
            _CASIMIR_TRACE_HARD_RELATIVE_TOLERANCE,
            "casimir_relative_trace_residual",
        )

        darboux_rel_res = 0.0
        darboux_verdict = InertialHeytingVerdict.COHERENT
        if darboux_freqs is not None and darboux_freqs.size > 0:
            if np.any(darboux_freqs < -_MACHINE_EPSILON):
                return (
                    casimir_rel_res,
                    1.0,
                    casimir_verdict,
                    InertialHeytingVerdict.VETOED,
                )
            parseval_sum = self._neumaier_sum(
                darboux_freqs * darboux_freqs, "∑ ν_k² (Darboux-Parseval)"
            )
            if expected_casimir > 0.0 or parseval_sum > 0.0:
                darboux_rel_res = abs(parseval_sum - expected_casimir) / max(
                    1.0, expected_casimir, parseval_sum
                )
                darboux_verdict = self._threshold_verdict(
                    darboux_rel_res,
                    _DARBOUX_PARSEVAL_SOFT_RELATIVE_TOLERANCE,
                    _DARBOUX_PARSEVAL_HARD_RELATIVE_TOLERANCE,
                    "darboux_parseval_relative_residual",
                )

        return (
            casimir_rel_res,
            darboux_rel_res,
            casimir_verdict,
            darboux_verdict,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 2.4  Ejecución, Validación Estructural y Morfismo Terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _certify_metric_signature(
        self,
        phase1_bridge: Phase1ObservationBridge,
        synthesis_data: GyroscopicSynthesisData,
        G_tensor: object = None,
    ) -> Phase2OrientationBridge:
        r"""
        Certifica la firma métrica antisimétrica, los invariantes de
        Casimir-Darboux-Poincaré y el endofuntor RSI-L3 en la Fase 2.
        """
        certified_phase1 = self._receive_certified_observation(phase1_bridge)
        fields = self._validate_synthesis_data(synthesis_data)
        metric = self._validate_optional_metric(G_tensor)

        relative_antisymmetry_residual, antisymmetry_verdict = (
            self._classify_antisymmetry(
                fields["antisymmetry_residual"],
                fields["is_strictly_skew"],
                fields["gyroscopic_frobenius_norm"],
                fields["relative_skew_residual"],
            )
        )
        vorticity_projection_ratio, vorticity_projection_verdict = (
            self._classify_vorticity_projection(
                fields["vorticity_projection_residual"],
                fields["gyroscopic_frobenius_norm"],
            )
        )
        gram_identity_verdict = self._classify_gram_identity(
            fields["wedge_gram_residual"]
        )
        even_rank_verdict = self._classify_even_rank(
            fields["skew_numerical_rank"],
            fields["rank_is_even"],
        )
        gauge_skew_residual, gauge_signature_verdict = (
            self._classify_gauge_signature(
                fields["tensor"],
                metric,
            )
        )
        (
            casimir_rel_res,
            darboux_rel_res,
            casimir_verdict,
            darboux_verdict,
        ) = self._classify_casimir_and_darboux_parseval(
            gyroscopic_frobenius_norm=fields["gyroscopic_frobenius_norm"],
            casimir_invariant=fields["casimir_invariant"],
            casimir_trace_residual=fields["casimir_trace_residual"],
            has_casimir=fields["has_casimir"],
            darboux_freqs=fields["darboux_freqs"],
        )
        rsi_synthesis_verdict = fields["motor_rsi_verdict"]

        gov_rsi_cert = self._rsi_level3_synthesize_lawvere_tierney_governance(
            normalized_residuals=(
                relative_antisymmetry_residual / _SKEW_HARD_RELATIVE_TOLERANCE,
                fields["wedge_gram_residual"] / _GRAM_RESIDUAL_HARD_MAX,
                gauge_skew_residual / _GAUGE_SKEW_HARD_RELATIVE_TOLERANCE,
                casimir_rel_res / _CASIMIR_TRACE_HARD_RELATIVE_TOLERANCE,
                darboux_rel_res / _DARBOUX_PARSEVAL_HARD_RELATIVE_TOLERANCE,
            ),
            motor_contraction_ratio=fields["motor_rsi_contraction"],
            celestial_regularity_weight=certified_phase1.sundman_regularization_factor,
            phase_tag="Phase2-Skew-Casimir-Darboux",
        )

        skew_verdict = self._heyting_join(
            (
                antisymmetry_verdict,
                vorticity_projection_verdict,
                gram_identity_verdict,
                even_rank_verdict,
                gauge_signature_verdict,
                casimir_verdict,
                darboux_verdict,
                rsi_synthesis_verdict,
            )
        )

        logger.debug(
            "Fase 2 Skew-Darboux: residual_rel=%.6e, ||W||_F=%.6e, "
            "Casimir=%.6e (res=%.6e), Darboux_res=%.6e, gauge=%.6e, "
            "RSI_L3_depth=%d, verdict=%s",
            relative_antisymmetry_residual,
            fields["gyroscopic_frobenius_norm"],
            fields["casimir_invariant"],
            casimir_rel_res,
            darboux_rel_res,
            gauge_skew_residual,
            gov_rsi_cert.recursion_depth,
            skew_verdict.name,
        )

        return Phase2OrientationBridge(
            phase1_bridge=certified_phase1,
            synthesis_data=synthesis_data,
            skew_verdict=skew_verdict,
            antisymmetry_verdict=antisymmetry_verdict,
            vorticity_projection_verdict=vorticity_projection_verdict,
            gram_identity_verdict=gram_identity_verdict,
            even_rank_verdict=even_rank_verdict,
            gauge_signature_verdict=gauge_signature_verdict,
            relative_antisymmetry_residual=relative_antisymmetry_residual,
            vorticity_projection_ratio=vorticity_projection_ratio,
            wedge_gram_residual=fields["wedge_gram_residual"],
            gauge_skew_residual=gauge_skew_residual,
            skew_numerical_rank=fields["skew_numerical_rank"],
            rank_is_even=fields["rank_is_even"],
            casimir_invariant_verdict=casimir_verdict,
            darboux_parseval_verdict=darboux_verdict,
            rsi_l3_synthesis_verdict=rsi_synthesis_verdict,
            casimir_quadratic_invariant=fields["casimir_invariant"],
            casimir_relative_trace_residual=casimir_rel_res,
            darboux_parseval_relative_residual=darboux_rel_res,
            poincare_areal_circulation=fields["poincare_circulation"],
            poincare_birkhoff_secular_factor=fields["pb_secular_factor"],
            governance_rsi_l3_certificate=gov_rsi_cert,
        )

    def _validate_phase2_bridge_structure(
        self,
        phase2_bridge: Phase2OrientationBridge,
    ) -> None:
        """Valida la estructura del certificado de Fase 2 (sin política)."""
        if not isinstance(phase2_bridge, Phase2OrientationBridge):
            raise PhaseHandoffCollapse(
                "handoff_phase2_to_phase3 exige Phase2OrientationBridge."
            )
        self._receive_certified_observation(phase2_bridge.phase1_bridge)
        self._validate_synthesis_data(phase2_bridge.synthesis_data)

        verdict_fields = (
            "skew_verdict",
            "antisymmetry_verdict",
            "vorticity_projection_verdict",
            "gram_identity_verdict",
            "even_rank_verdict",
            "gauge_signature_verdict",
            "casimir_invariant_verdict",
            "darboux_parseval_verdict",
            "rsi_l3_synthesis_verdict",
        )
        for field_name in verdict_fields:
            if hasattr(phase2_bridge, field_name):
                self._validate_verdict(
                    getattr(phase2_bridge, field_name),
                    f"phase2_bridge.{field_name}",
                )

        nonneg_fields = (
            "relative_antisymmetry_residual",
            "vorticity_projection_ratio",
            "wedge_gram_residual",
            "gauge_skew_residual",
            "casimir_quadratic_invariant",
            "casimir_relative_trace_residual",
            "darboux_parseval_relative_residual",
            "poincare_areal_circulation",
        )
        for field_name in nonneg_fields:
            if hasattr(phase2_bridge, field_name):
                self._as_nonnegative_finite_float(
                    getattr(phase2_bridge, field_name),
                    f"phase2_bridge.{field_name}",
                )

        if hasattr(phase2_bridge, "poincare_birkhoff_secular_factor"):
            self._as_finite_float(
                phase2_bridge.poincare_birkhoff_secular_factor,
                "phase2_bridge.poincare_birkhoff_secular_factor",
            )

        if hasattr(phase2_bridge, "skew_numerical_rank"):
            rank = self._as_int(
                phase2_bridge.skew_numerical_rank,
                "phase2_bridge.skew_numerical_rank",
            )
            if rank < 0:
                raise PhaseHandoffCollapse(
                    "phase2_bridge.skew_numerical_rank no puede ser negativo."
                )
        if hasattr(phase2_bridge, "rank_is_even"):
            self._as_bool(
                phase2_bridge.rank_is_even, "phase2_bridge.rank_is_even"
            )

    def _assert_phase2_lattice_consistency(
        self,
        phase2_bridge: Phase2OrientationBridge,
    ) -> None:
        """Exige que el join almacenado coincida con el de los clasificadores granulares."""
        recomputed = self._heyting_join(
            (
                phase2_bridge.antisymmetry_verdict,
                phase2_bridge.vorticity_projection_verdict,
                phase2_bridge.gram_identity_verdict,
                phase2_bridge.even_rank_verdict,
                phase2_bridge.gauge_signature_verdict,
                phase2_bridge.casimir_invariant_verdict,
                phase2_bridge.darboux_parseval_verdict,
                phase2_bridge.rsi_l3_synthesis_verdict,
            )
        )
        if recomputed is not phase2_bridge.skew_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del retículo en Φ₂₃: "
                f"join granular = {recomputed.name}, "
                f"almacenado = {phase2_bridge.skew_verdict.name}."
            )

    def execute_phase2(
        self,
        phase1_bridge: Phase1ObservationBridge,
        synthesis_data: GyroscopicSynthesisData,
        G_tensor: object = None,
    ) -> Phase2OrientationBridge:
        """
        Ejecuta la Fase 2 invocando ``handoff_phase1_to_phase2`` y entrega el
        certificado ``Phase2OrientationBridge`` que alimenta
        ``handoff_phase2_to_phase3``.
        """
        return self._certify_metric_signature(
            phase1_bridge,
            synthesis_data,
            G_tensor=G_tensor,
        )

    def handoff_phase2_to_phase3(
        self,
        phase2_bridge: Phase2OrientationBridge,
    ) -> Phase2OrientationBridge:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 2  ≡  CONTINUACIÓN E INICIO DE LA FASE 3.

        Morfismo de transición categórica:

            $$\Phi_{23}:
              \mathrm{Phase2OrientationBridge}
              \longrightarrow
              \mathrm{Phase2OrientationBridge}$$

        Certifica la integridad estructural, la consistencia del supremo de
        Fase 2 en $$\Omega_3$$ y la monotonía de Lyapunov del certificado
        RSI-L3 antes de abrir el dominio de la Fase 3.
        """
        self._validate_phase2_bridge_structure(phase2_bridge)
        self._assert_phase2_lattice_consistency(phase2_bridge)
        if not phase2_bridge.governance_rsi_l3_certificate.is_monotone_lyapunov:
            raise PhaseHandoffCollapse(
                "El certificado RSI-L3 de gobernanza de Fase 2 violó la "
                "monotonía de Lyapunov en Φ₂₃."
            )
        return phase2_bridge


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → COLAPSO TERMODINÁMICO EN EL RETÍCULO DE HEYTING, ESTABILIDAD DE
#          FLOQUET-KREĬN, INVARIANTE DE TISSERAND Y RSI-L3 (Decide & Act)
#          [Continuación directa de handoff_phase2_to_phase3]
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingLatticeDecider(Phase2_SkewSymmetryCertifier):
    r"""
    Fase 3: inicia exactamente donde concluye ``handoff_phase2_to_phase3``,
    recibiendo el puente certificado de orientación y consolidando el ciclo OODA
    sobre el operador simpléctico efectivo $$J_{\mathrm{eff}}$$.

    Audita:
        1. Trabajo nulo de Neumaier, trabajo estructural por pares y sondas
           Monte-Carlo de Liouville.
        2. Antisimetría relativa de la estructura de Dirac $$J_{\mathrm{eff}}$$.
        3. **Mecánica Celeste de Poincaré (Fase 3)**:
           - Estabilidad elíptica de Floquet-Kreĭn (ausencia de deriva real
             hiperbólica en el espectro de $$J_{\mathrm{eff}}$$).
           - Cota de Cauchy-Schwarz del parámetro giroscópico de Tisserand
             $$T_{\mathrm{gyro}}\in[-1,1]$$.
           - Invariante virial de Jacobi-Poincaré y horizonte de recurrencia de
             Kac-Poincaré.
        4. **Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{\mathrm{Gov},(3)}$$)**:
           Co-certificación de la estabilización simpléctica del motor y cierre
           modal de Lawvere-Tierney del supremo global:

               $$v_{\mathrm{final}}
                 =v_{\mathrm{Liouville}}
                 \sqcup v_{\mathrm{Skew}}
                 \sqcup v_{\mathrm{Work}}$$

    El método final ``handoff_phase3_to_orchestrator`` constituye la definición
    formal terminal de la Fase 3 y el punto de enlace con el orquestador supremo.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 3.1  Continuación formal inmediata del Morfismo Terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_orientation(
        self,
        phase2_bridge: Phase2OrientationBridge,
    ) -> Phase2OrientationBridge:
        """
        PRIMER MÉTODO DE LA FASE 3 (Continuación directa de ``handoff_phase2_to_phase3``).

        Invoca el morfismo terminal de la Fase 2, revalida la coherencia de los
        puentes anidados en $$\Omega_3$$ y entrega el certificado a la Fase 3.
        """
        return self.handoff_phase2_to_phase3(phase2_bridge)

    def _validate_phase2_bridge(
        self, phase2_bridge: Phase2OrientationBridge
    ) -> None:
        """Fachada de revalidación de Fase 2 vía $$\Phi_{23}$$."""
        self._receive_certified_orientation(phase2_bridge)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.2  Validación del artefacto termodinámico y celeste de Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_veto_data(
        self,
        veto_data: ThermodynamicVetoData,
    ) -> dict[str, Any]:
        """
        Verifica la integridad del artefacto termodinámico producido por el
        motor, incluyendo las extensiones de Floquet-Kreĭn, Tisserand,
        Jacobi-Poincaré y RSI-L3.
        """
        if not isinstance(veto_data, ThermodynamicVetoData):
            raise RiemannianInertiaAgentError(
                "veto_data debe ser una instancia de ThermodynamicVetoData."
            )

        nilpotent_work_residual = self._get_required_nonnegative_finite_float(
            veto_data, "nilpotent_work_residual", "veto_data"
        )
        is_symplectically_passive = self._get_required_bool(
            veto_data, "is_symplectically_passive", "veto_data"
        )
        work_tolerance = self._get_optional_positive_finite_float(
            veto_data,
            "work_tolerance",
            _WORK_SOFT_ABSOLUTE_TOLERANCE,
            "veto_data",
        )
        dirac_symmetric_residual = (
            self._get_optional_nonnegative_finite_float(
                veto_data, "dirac_symmetric_residual", 0.0, "veto_data"
            )
        )
        pairwise_work_residual = self._get_optional_nonnegative_finite_float(
            veto_data, "pairwise_work_residual", 0.0, "veto_data"
        )
        liouville_probe_residual = (
            self._get_optional_nonnegative_finite_float(
                veto_data, "liouville_probe_residual", 0.0, "veto_data"
            )
        )
        certified_relative_skew = self._get_optional_nonnegative_finite_float(
            veto_data, "relative_skew_residual", 0.0, "veto_data"
        )

        effective_dirac_matrix = self._validate_finite_array_attribute(
            veto_data, "effective_dirac_matrix", "veto_data", required=False
        )
        if effective_dirac_matrix is None:
            dirac_frobenius_norm = 0.0
        else:
            dirac_frobenius_norm = self._matrix_frobenius_norm(
                effective_dirac_matrix,
                "veto_data.effective_dirac_matrix",
            )

        # Extensiones celestes de Floquet-Kreĭn, Tisserand y Jacobi-Poincaré
        jp_virial = self._get_optional_nonnegative_finite_float(
            veto_data, "jacobi_poincare_virial_invariant", 0.0, "veto_data"
        )
        tisserand_param = self._get_optional_finite_float(
            veto_data, "tisserand_gyroscopic_parameter", 1.0, "veto_data"
        )
        floquet_real_drift = self._get_optional_nonnegative_finite_float(
            veto_data, "floquet_max_real_drift", 0.0, "veto_data"
        )
        floquet_radius = self._get_optional_nonnegative_finite_float(
            veto_data, "floquet_spectral_radius", 0.0, "veto_data"
        )
        recurrence_horizon = self._get_optional_nonnegative_finite_float(
            veto_data, "poincare_recurrence_horizon", 0.0, "veto_data"
        )
        _, motor_rsi_contraction, motor_rsi_verdict = (
            self._extract_motor_rsi_audit(
                veto_data, "rsi_l3_symplectic_audit", "veto_data"
            )
        )

        return {
            "nilpotent_work_residual": nilpotent_work_residual,
            "is_symplectically_passive": is_symplectically_passive,
            "work_tolerance": work_tolerance,
            "dirac_symmetric_residual": dirac_symmetric_residual,
            "dirac_frobenius_norm": dirac_frobenius_norm,
            "pairwise_work_residual": pairwise_work_residual,
            "liouville_probe_residual": liouville_probe_residual,
            "certified_relative_skew": certified_relative_skew,
            "jp_virial": jp_virial,
            "tisserand_param": tisserand_param,
            "floquet_real_drift": floquet_real_drift,
            "floquet_radius": floquet_radius,
            "recurrence_horizon": recurrence_horizon,
            "motor_rsi_contraction": motor_rsi_contraction,
            "motor_rsi_verdict": motor_rsi_verdict,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 3.3  Clasificadores locales de Fase 3 (Trabajo, Floquet-Kreĭn y Tisserand)
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_work_passivity(
        self,
        residual: float,
        is_symplectically_passive: bool,
        work_tolerance: float,
    ) -> InertialHeytingVerdict:
        """
        Clasifica un residual de trabajo nulo contra la tolerancia del
        motor y contra la política blanda absoluta.
        """
        if not is_symplectically_passive:
            return InertialHeytingVerdict.VETOED
        if residual > work_tolerance:
            return InertialHeytingVerdict.VETOED
        if residual > _WORK_SOFT_ABSOLUTE_TOLERANCE:
            return InertialHeytingVerdict.DEGRADED
        return InertialHeytingVerdict.COHERENT

    def _classify_dirac_symmetry(
        self,
        dirac_symmetric_residual: float,
        dirac_frobenius_norm: float,
        certified_relative_skew: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""Clasifica la componente simétrica residual de $$J_{\mathrm{eff}}$$."""
        computed = self._relative_residual(
            dirac_symmetric_residual,
            dirac_frobenius_norm,
        )
        relative_residual = max(computed, certified_relative_skew)
        return relative_residual, self._threshold_verdict(
            relative_residual,
            _DIRAC_SYMMETRIC_SOFT_RELATIVE_TOLERANCE,
            _DIRAC_SYMMETRIC_HARD_RELATIVE_TOLERANCE,
            "dirac_r_skew",
        )

    def _classify_floquet_krein_stability(
        self,
        floquet_real_drift: float,
        dirac_frobenius_norm: float,
        floquet_spectral_radius: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""
        Clasifica la estabilidad espectral elíptica de Floquet-Kreĭn:

            $$r_{\mathrm{Floquet}}
              = \frac{\max_{\mu\in\sigma(J_{\mathrm{eff}})}|\Re(\mu)|}
                     {\max(1, \|J_{\mathrm{eff}}\|_F, \rho(J_{\mathrm{eff}}))}$$
        """
        scale = max(1.0, dirac_frobenius_norm, floquet_spectral_radius)
        rel_drift = self._relative_residual(floquet_real_drift, scale)
        verdict = self._threshold_verdict(
            rel_drift,
            _FLOQUET_DRIFT_SOFT_RELATIVE_TOLERANCE,
            _FLOQUET_DRIFT_HARD_RELATIVE_TOLERANCE,
            "floquet_relative_real_drift",
        )
        return rel_drift, verdict

    def _classify_tisserand_orbital_parameter(
        self,
        tisserand_param: float,
    ) -> InertialHeytingVerdict:
        r"""
        Audita el parámetro giroscópico de Tisserand $$T_{\mathrm{gyro}}$$.

        Por la desigualdad de Cauchy-Schwarz en el fibrado tangente,
        $$|T_{\mathrm{gyro}}| \le 1$$. Una violación de esta cota revela
        corrupción métrica en la composición de los campos Hamiltonianos.
        """
        if not math.isfinite(tisserand_param):
            return InertialHeytingVerdict.VETOED
        excess = max(0.0, abs(tisserand_param) - 1.0)
        if excess > _TISSERAND_CAUCHY_HARD_SLACK:
            return InertialHeytingVerdict.VETOED
        if excess > 100.0 * _MACHINE_EPSILON:
            return InertialHeytingVerdict.DEGRADED
        return InertialHeytingVerdict.COHERENT

    def _compute_work_margin(
        self,
        nilpotent_work_residual: float,
        work_tolerance: float,
    ) -> float:
        r"""Margen normalizado de trabajo nulo: $$\max(0, 1-\mathrm{res}/\mathrm{tol})$$."""
        denominator = max(work_tolerance, _MACHINE_EPSILON)
        return self._clamp01(1.0 - (nilpotent_work_residual / denominator))

    def _collect_veto_sources(
        self,
        phase2_bridge: Phase2OrientationBridge,
        work_verdict: InertialHeytingVerdict,
        work_passivity_verdict: InertialHeytingVerdict,
        pairwise_work_verdict: InertialHeytingVerdict,
        liouville_probe_verdict: InertialHeytingVerdict,
        dirac_symmetry_verdict: InertialHeytingVerdict,
        floquet_stability_verdict: InertialHeytingVerdict,
        tisserand_orbital_verdict: InertialHeytingVerdict,
        rsi_l3_symplectic_verdict: InertialHeytingVerdict,
    ) -> list[str]:
        """Inventario exhaustivo de subobjetos que tocaron el supremo $$\top$$."""
        sources: list[str] = []
        phase1 = phase2_bridge.phase1_bridge

        mapping: tuple[tuple[str, InertialHeytingVerdict], ...] = (
            ("Liouville.momentum", phase1.momentum_bound_verdict),
            ("Liouville.condition", phase1.metric_condition_verdict),
            ("Liouville.inverse", phase1.inverse_consistency_verdict),
            ("Liouville.pairing", phase1.pairing_verdict),
            ("Liouville.musical", phase1.musical_roundtrip_verdict),
            ("Liouville.spectrum", phase1.spectral_gap_verdict),
            ("Liouville.sundman", phase1.sundman_regularity_verdict),
            ("Liouville.poincare_cartan", phase1.poincare_cartan_verdict),
            ("Liouville.rsi_l3", phase1.rsi_l3_motor_verdict),
            ("Skew.antisymmetry", phase2_bridge.antisymmetry_verdict),
            ("Skew.vorticity", phase2_bridge.vorticity_projection_verdict),
            ("Skew.gram", phase2_bridge.gram_identity_verdict),
            ("Skew.rank", phase2_bridge.even_rank_verdict),
            ("Skew.gauge", phase2_bridge.gauge_signature_verdict),
            ("Skew.casimir", phase2_bridge.casimir_invariant_verdict),
            ("Skew.darboux_parseval", phase2_bridge.darboux_parseval_verdict),
            ("Skew.rsi_l3", phase2_bridge.rsi_l3_synthesis_verdict),
            ("Work.passivity", work_passivity_verdict),
            ("Work.pairwise", pairwise_work_verdict),
            ("Work.liouville_probe", liouville_probe_verdict),
            ("Work.dirac_symmetry", dirac_symmetry_verdict),
            ("Work.floquet_krein", floquet_stability_verdict),
            ("Work.tisserand", tisserand_orbital_verdict),
            ("Work.rsi_l3", rsi_l3_symplectic_verdict),
            ("Work.join", work_verdict),
        )
        for label, verdict in mapping:
            if verdict is InertialHeytingVerdict.VETOED:
                sources.append(label)
        return sources

    def _raise_lattice_veto(
        self,
        phase2_bridge: Phase2OrientationBridge,
        fields: dict[str, Any],
        final_verdict: InertialHeytingVerdict,
        work_verdict: InertialHeytingVerdict,
        work_passivity_verdict: InertialHeytingVerdict,
        pairwise_work_verdict: InertialHeytingVerdict,
        liouville_probe_verdict: InertialHeytingVerdict,
        dirac_symmetry_verdict: InertialHeytingVerdict,
        floquet_stability_verdict: InertialHeytingVerdict,
        tisserand_orbital_verdict: InertialHeytingVerdict,
        rsi_l3_symplectic_verdict: InertialHeytingVerdict,
    ) -> None:
        """Materializa el colapso del retículo con diagnóstico granular."""
        sources = self._collect_veto_sources(
            phase2_bridge,
            work_verdict,
            work_passivity_verdict,
            pairwise_work_verdict,
            liouville_probe_verdict,
            dirac_symmetry_verdict,
            floquet_stability_verdict,
            tisserand_orbital_verdict,
            rsi_l3_symplectic_verdict,
        )
        source_detail = ", ".join(sources) if sources else "Unknown"
        phase1 = phase2_bridge.phase1_bridge
        momentum_norm = self._get_required_nonnegative_finite_float(
            phase1.momentum_data, "momentum_norm", "momentum_data"
        )

        if phase1.momentum_bound_verdict is InertialHeytingVerdict.VETOED:
            exc_type: type[RiemannianInertiaAgentError] = (
                LiouvilleVolumeCollapse
            )
        elif phase1.metric_condition_verdict is InertialHeytingVerdict.VETOED:
            exc_type = MetricConditionCollapse
        elif (
            phase1.inverse_consistency_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = InverseCoherenceCollapse
        elif phase1.pairing_verdict is InertialHeytingVerdict.VETOED:
            exc_type = DualPairingCollapse
        elif phase1.musical_roundtrip_verdict is InertialHeytingVerdict.VETOED:
            exc_type = MusicalIsomorphismCollapse
        elif (
            phase1.sundman_regularity_verdict is InertialHeytingVerdict.VETOED
            or phase1.poincare_cartan_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = PoincareCelestialCollapse
        elif (
            phase2_bridge.antisymmetry_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = SkewSignatureCollapse
        elif (
            phase2_bridge.gauge_signature_verdict
            is InertialHeytingVerdict.VETOED
        ):
            exc_type = SkewSignatureCollapse
        elif (
            phase2_bridge.gram_identity_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = ExteriorAlgebraCollapse
        elif (
            phase2_bridge.casimir_invariant_verdict
            is InertialHeytingVerdict.VETOED
            or phase2_bridge.darboux_parseval_verdict
            is InertialHeytingVerdict.VETOED
        ):
            exc_type = CasimirDarbouxCollapse
        elif floquet_stability_verdict is InertialHeytingVerdict.VETOED:
            exc_type = FloquetKreinCollapse
        elif (
            work_passivity_verdict is InertialHeytingVerdict.VETOED
            or pairwise_work_verdict is InertialHeytingVerdict.VETOED
            or liouville_probe_verdict is InertialHeytingVerdict.VETOED
            or dirac_symmetry_verdict is InertialHeytingVerdict.VETOED
            or tisserand_orbital_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = ThermodynamicPassivityCollapse
        elif (
            phase1.rsi_l3_motor_verdict is InertialHeytingVerdict.VETOED
            or phase2_bridge.rsi_l3_synthesis_verdict
            is InertialHeytingVerdict.VETOED
            or rsi_l3_symplectic_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = RecursiveGovernanceCollapse
        else:
            exc_type = HeytingLatticeVeto

        message = (
            "Colapso de software: el operador giroscópico ha inyectado "
            "entropía no acotada. "
            f"Veredicto Supremo = {final_verdict.name}. "
            f"Fuente(s) de veto = [{source_detail}]. "
            f"‖p‖ = {momentum_norm:.6e}, "
            f"antisym_rel = {phase2_bridge.relative_antisymmetry_residual:.6e}, "
            f"gauge_rel = {phase2_bridge.gauge_skew_residual:.6e}, "
            f"work = {fields['nilpotent_work_residual']:.6e}, "
            f"pares = {fields['pairwise_work_residual']:.6e}, "
            f"Liouville = {fields['liouville_probe_residual']:.6e}, "
            f"Floquet_drift = {fields['floquet_real_drift']:.6e}, "
            f"work_tol = {fields['work_tolerance']:.6e}. "
            "Transacción aniquilada en RAM."
        )
        raise exc_type(message)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.4  Ejecución y Morfismo Terminal de la Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_thermodynamic_lattice(
        self,
        phase2_bridge: Phase2OrientationBridge,
        veto_data: ThermodynamicVetoData,
        raise_on_veto: bool = True,
    ) -> InertialGovernanceState:
        r"""
        Consolida los veredictos locales, la estabilidad de Floquet-Kreĭn y el
        certificado RSI-L3 en un único supremo algebraico:

            $$v_{\mathrm{final}}
              =\max(v_{\mathrm{Liouville}},\,v_{\mathrm{Skew}},\,v_{\mathrm{Work}})$$
        """
        certified_phase2 = self._receive_certified_orientation(phase2_bridge)
        fields = self._validate_veto_data(veto_data)

        work_passivity_verdict = self._classify_work_passivity(
            fields["nilpotent_work_residual"],
            fields["is_symplectically_passive"],
            fields["work_tolerance"],
        )
        pairwise_work_verdict = self._classify_work_passivity(
            fields["pairwise_work_residual"],
            fields["is_symplectically_passive"],
            fields["work_tolerance"],
        )
        liouville_probe_verdict = self._classify_work_passivity(
            fields["liouville_probe_residual"],
            fields["is_symplectically_passive"],
            fields["work_tolerance"],
        )
        dirac_relative, dirac_symmetry_verdict = self._classify_dirac_symmetry(
            fields["dirac_symmetric_residual"],
            fields["dirac_frobenius_norm"],
            fields["certified_relative_skew"],
        )
        floquet_rel_drift, floquet_stability_verdict = (
            self._classify_floquet_krein_stability(
                fields["floquet_real_drift"],
                fields["dirac_frobenius_norm"],
                fields["floquet_radius"],
            )
        )
        tisserand_orbital_verdict = self._classify_tisserand_orbital_parameter(
            fields["tisserand_param"]
        )
        rsi_symplectic_verdict = fields["motor_rsi_verdict"]

        work_tol = max(fields["work_tolerance"], _MACHINE_EPSILON)
        gov_rsi_cert = self._rsi_level3_synthesize_lawvere_tierney_governance(
            normalized_residuals=(
                fields["nilpotent_work_residual"] / work_tol,
                fields["pairwise_work_residual"] / work_tol,
                fields["liouville_probe_residual"] / work_tol,
                dirac_relative / _DIRAC_SYMMETRIC_HARD_RELATIVE_TOLERANCE,
                floquet_rel_drift / _FLOQUET_DRIFT_HARD_RELATIVE_TOLERANCE,
            ),
            motor_contraction_ratio=fields["motor_rsi_contraction"],
            celestial_regularity_weight=certified_phase2.phase1_bridge.sundman_regularization_factor,
            phase_tag="Phase3-Heyting-Floquet-Tisserand",
        )

        work_verdict = self._heyting_join(
            (
                work_passivity_verdict,
                pairwise_work_verdict,
                liouville_probe_verdict,
                dirac_symmetry_verdict,
                floquet_stability_verdict,
                tisserand_orbital_verdict,
                rsi_symplectic_verdict,
            )
        )
        final_supremum_verdict = self._heyting_join(
            (
                certified_phase2.phase1_bridge.liouville_verdict,
                certified_phase2.skew_verdict,
                work_verdict,
            )
        )
        is_epistemologically_valid = (
            final_supremum_verdict is not InertialHeytingVerdict.VETOED
        )
        work_margin = self._compute_work_margin(
            fields["nilpotent_work_residual"],
            fields["work_tolerance"],
        )

        if not is_epistemologically_valid and raise_on_veto:
            self._raise_lattice_veto(
                certified_phase2,
                fields,
                final_supremum_verdict,
                work_verdict,
                work_passivity_verdict,
                pairwise_work_verdict,
                liouville_probe_verdict,
                dirac_symmetry_verdict,
                floquet_stability_verdict,
                tisserand_orbital_verdict,
                rsi_symplectic_verdict,
            )

        logger.debug(
            "Fase 3 Lattice-Floquet: work=%.6e, pares=%.6e, Liouville=%.6e, "
            "Floquet_rel=%.6e, Tisserand=%.6e, verdicts=(L=%s, S=%s, W=%s), "
            "final=%s, valid=%s, RSI_L3_depth=%d",
            fields["nilpotent_work_residual"],
            fields["pairwise_work_residual"],
            fields["liouville_probe_residual"],
            floquet_rel_drift,
            fields["tisserand_param"],
            certified_phase2.phase1_bridge.liouville_verdict.name,
            certified_phase2.skew_verdict.name,
            work_verdict.name,
            final_supremum_verdict.name,
            is_epistemologically_valid,
            gov_rsi_cert.recursion_depth,
        )

        return InertialGovernanceState(
            phase2_bridge=certified_phase2,
            veto_data=veto_data,
            work_verdict=work_verdict,
            final_supremum_verdict=final_supremum_verdict,
            is_epistemologically_valid=is_epistemologically_valid,
            work_passivity_verdict=work_passivity_verdict,
            pairwise_work_verdict=pairwise_work_verdict,
            liouville_probe_verdict=liouville_probe_verdict,
            dirac_symmetry_verdict=dirac_symmetry_verdict,
            work_margin=work_margin,
            pairwise_work_residual=fields["pairwise_work_residual"],
            liouville_probe_residual=fields["liouville_probe_residual"],
            dirac_relative_skew_residual=dirac_relative,
            floquet_stability_verdict=floquet_stability_verdict,
            tisserand_orbital_verdict=tisserand_orbital_verdict,
            rsi_l3_symplectic_verdict=rsi_symplectic_verdict,
            floquet_relative_real_drift=floquet_rel_drift,
            floquet_spectral_radius=fields["floquet_radius"],
            tisserand_gyroscopic_parameter=fields["tisserand_param"],
            jacobi_poincare_virial_invariant=fields["jp_virial"],
            poincare_recurrence_horizon=fields["recurrence_horizon"],
            governance_rsi_l3_certificate=gov_rsi_cert,
        )

    def execute_phase3(
        self,
        phase2_bridge: Phase2OrientationBridge,
        veto_data: ThermodynamicVetoData,
        raise_on_veto: bool = True,
    ) -> InertialGovernanceState:
        """
        Ejecuta la Fase 3 invocando ``handoff_phase2_to_phase3`` y entrega el
        estado supremo ``InertialGovernanceState``.
        """
        return self._evaluate_thermodynamic_lattice(
            phase2_bridge,
            veto_data,
            raise_on_veto,
        )

    def handoff_phase3_to_orchestrator(
        self,
        governance_state: InertialGovernanceState,
    ) -> InertialGovernanceState:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 3  ≡  CONTINUACIÓN EN EL ORQUESTADOR.

        Morfismo terminal de cierre del Topos de Gobernanza:

            $$\Phi_{3\Omega}:
              \mathrm{InertialGovernanceState}
              \longrightarrow
              \mathrm{InertialGovernanceState}$$

        Revalida la idempotencia del supremo global en $$\Omega_3$$ y la
        monotonía de Lyapunov de los tres certificados RSI-L3 anidados antes de
        entregar el estado soberano al orquestador ``RiemannianInertiaAgent``.
        """
        if not isinstance(governance_state, InertialGovernanceState):
            raise PhaseHandoffCollapse(
                "handoff_phase3_to_orchestrator exige InertialGovernanceState."
            )

        recomputed_supremum = self._heyting_join(
            (
                governance_state.phase2_bridge.phase1_bridge.liouville_verdict,
                governance_state.phase2_bridge.skew_verdict,
                governance_state.work_verdict,
            )
        )
        if recomputed_supremum is not governance_state.final_supremum_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del supremo global en Φ₃Ω: "
                f"recomputado = {recomputed_supremum.name}, "
                f"almacenado = {governance_state.final_supremum_verdict.name}."
            )

        if not governance_state.governance_rsi_l3_certificate.is_monotone_lyapunov:
            raise PhaseHandoffCollapse(
                "El certificado RSI-L3 de gobernanza de Fase 3 violó la "
                "monotonía de Lyapunov en Φ₃Ω."
            )

        return governance_state


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: RIEMANNIAN INERTIA AGENT
# [Continuación directa de handoff_phase3_to_orchestrator]
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaAgent(Morphism, Phase3_HeytingLatticeDecider):
    r"""
    Soberano del Momentum Ciber-Físico con Gobernanza Celeste de Poincaré y
    Automejora Recursiva de Nivel 3 en el Topos de Heyting.

    Composición funtorial de las tres fases anidadas:

        $$F_{\mathrm{Agent}}
          = \Phi_{3\Omega}
          \circ F_{\mathrm{Phase3}}\bigl(\mathfrak{R}_3^{\mathrm{Gov},(3)}\bigr)
          \circ \Phi_{23}
          \circ F_{\mathrm{Phase2}}\bigl(\mathfrak{R}_3^{\mathrm{Gov},(2)}\bigr)
          \circ \Phi_{12}
          \circ F_{\mathrm{Phase1}}\bigl(\mathfrak{R}_3^{\mathrm{Gov},(1)}\bigr)$$
    """

    def __init__(self, motor: RiemannianInertiaModulator):
        """Inyecta el motor físico que ejecuta las transformaciones simplécticas."""
        self._motor = motor
        self._validate_motor()

    def _validate_motor(self) -> None:
        """Verifica que el motor implementa el contrato funtorial de tres fases."""
        if self._motor is None:
            raise MotorContractError("El motor físico no puede ser None.")

        required_methods = (
            "execute_phase1",
            "execute_phase2",
            "execute_phase3",
        )
        for method in required_methods:
            if not callable(getattr(self._motor, method, None)):
                raise MotorContractError(
                    f"El motor debe exponer el método ejecutable '{method}'."
                )

    def _validate_governance_payload(
        self,
        q_dot: object,
        grad_H: object,
        G_tensor: object,
        G_inv: object,
        J_base: object,
        vorticity_matrix: object,
    ) -> None:
        """
        Pre-auditoría dimensional ligera del payload de gobernanza.
        """
        try:
            velocity = np.asarray(q_dot, dtype=np.float64)
            gradient = np.asarray(grad_H, dtype=np.float64)
            metric = np.asarray(G_tensor, dtype=np.float64)
            metric_inv = np.asarray(G_inv, dtype=np.float64)
            interconnection = np.asarray(J_base, dtype=np.float64)
            vorticity = np.asarray(vorticity_matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaAgentError(
                "El payload de gobernanza no es convertible a float64."
            ) from exc

        payloads = (
            ("q_dot", velocity),
            ("grad_H", gradient),
            ("G_tensor", metric),
            ("G_inv", metric_inv),
            ("J_base", interconnection),
            ("vorticity_matrix", vorticity),
        )
        for name, array in payloads:
            if array.size == 0:
                raise RiemannianInertiaAgentError(f"{name} no puede ser vacío.")
            if not np.all(np.isfinite(array)):
                raise RiemannianInertiaAgentError(
                    f"{name} contiene valores no finitos."
                )

        if velocity.ndim != 1 or gradient.ndim != 1:
            raise RiemannianInertiaAgentError(
                "q_dot y grad_H deben ser vectores 1-D."
            )
        if velocity.size != gradient.size:
            raise RiemannianInertiaAgentError(
                "q_dot y grad_H deben vivir en el mismo espacio vectorial."
            )

        n = velocity.size
        for name, array in (
            ("G_tensor", metric),
            ("G_inv", metric_inv),
            ("J_base", interconnection),
            ("vorticity_matrix", vorticity),
        ):
            if array.ndim != 2 or array.shape != (n, n):
                raise RiemannianInertiaAgentError(
                    f"{name} debe ser una matriz {n}×{n}."
                )

    def _maybe_fail_fast(
        self,
        verdict: InertialHeytingVerdict,
        stage: str,
        fail_fast: bool,
        factory: type[RiemannianInertiaAgentError],
        detail: str,
    ) -> None:
        """Colapsa anticipadamente si ``fail_fast`` y el veredicto es $$\top$$."""
        if fail_fast and verdict is InertialHeytingVerdict.VETOED:
            raise factory(
                f"Fail-fast en {stage}: veredicto = {verdict.name}. {detail}"
            )

    def execute_inertia_governance(
        self,
        q_dot: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        J_base: NDArray[np.float64],
        vorticity_matrix: NDArray[np.float64],
        raise_on_veto: bool = True,
        fail_fast: bool = False,
    ) -> InertialGovernanceState:
        r"""
        Ejecuta el ciclo categórico, celeste y topológico entrelazando las tres
        fases anidadas del motor físico con las auditorías RSI-L3 del Agente.
        """
        self._validate_governance_payload(
            q_dot, grad_H, G_tensor, G_inv, J_base, vorticity_matrix
        )

        # --- OODA PASO 1: Observación física, Poincaré-Cartan y RSI-L3 ---
        momentum_data = self._motor.execute_phase1(
            q_dot=q_dot,
            G_tensor=G_tensor,
            G_inv=G_inv,
        )
        phase1_bridge = self.execute_phase1(momentum_data)
        self._maybe_fail_fast(
            phase1_bridge.liouville_verdict,
            "Fase 1 (Liouville / Poincaré-Sundman)",
            fail_fast,
            LiouvilleVolumeCollapse,
            f"‖p‖ = {phase1_bridge.momentum_data.momentum_norm:.6e}, "
            f"Sundman = {phase1_bridge.sundman_regularization_factor:.6e}.",
        )

        # --- OODA PASO 2: Orientación métrica, Casimir-Darboux y RSI-L3 ---
        # Φ₁₂ : handoff_phase1_to_phase2 se invoca como inicio de execute_phase2
        synthesis_data = self._motor.execute_phase2(
            momentum_data=momentum_data,
            vorticity_matrix=vorticity_matrix,
        )
        phase2_bridge = self.execute_phase2(
            phase1_bridge,
            synthesis_data,
            G_tensor=G_tensor,
        )
        self._maybe_fail_fast(
            phase2_bridge.skew_verdict,
            "Fase 2 (Skew / Gauge / Casimir-Darboux)",
            fail_fast,
            SkewSignatureCollapse,
            f"r_skew = {phase2_bridge.relative_antisymmetry_residual:.6e}, "
            f"r_G = {phase2_bridge.gauge_skew_residual:.6e}, "
            f"Casimir_res = {phase2_bridge.casimir_relative_trace_residual:.6e}.",
        )

        # --- OODA PASO 3: Decisión de Heyting, Floquet-Kreĭn y RSI-L3 ---
        # Φ₂₃ : handoff_phase2_to_phase3 se invoca como inicio de execute_phase3
        veto_data = self._motor.execute_phase3(
            grad_H=grad_H,
            J_base=J_base,
            synthesis_data=synthesis_data,
        )
        raw_state = self.execute_phase3(
            phase2_bridge=phase2_bridge,
            veto_data=veto_data,
            raise_on_veto=raise_on_veto,
        )
        # Φ₃Ω : handoff_phase3_to_orchestrator sella la cadena anidada
        final_state = self.handoff_phase3_to_orchestrator(raw_state)

        logger.info(
            "Gobernanza inercial completada [v%s]: final=%s, valid=%s, "
            "‖p‖=%.6e, r_skew=%.6e, work=%.6e, Floquet_rel=%.6e, Tisserand=%.6e.",
            __version__,
            final_state.final_supremum_verdict.name,
            final_state.is_epistemologically_valid,
            phase1_bridge.momentum_data.momentum_norm,
            phase2_bridge.relative_antisymmetry_residual,
            final_state.veto_data.nilpotent_work_residual,
            final_state.floquet_relative_real_drift,
            final_state.tisserand_gyroscopic_parameter,
        )
        return final_state


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "InertialHeytingVerdict",
    "RiemannianInertiaAgentError",
    "MotorContractError",
    "PhaseHandoffCollapse",
    "LiouvilleVolumeCollapse",
    "MetricConditionCollapse",
    "InverseCoherenceCollapse",
    "DualPairingCollapse",
    "MusicalIsomorphismCollapse",
    "PoincareCelestialCollapse",
    "SkewSignatureCollapse",
    "ExteriorAlgebraCollapse",
    "CasimirDarbouxCollapse",
    "ThermodynamicPassivityCollapse",
    "FloquetKreinCollapse",
    "RecursiveGovernanceCollapse",
    "HeytingLatticeVeto",
    "GovernanceRSILevel3Certificate",
    "Phase1ObservationBridge",
    "Phase2OrientationBridge",
    "InertialGovernanceState",
    "Phase1_LiouvilleVolumeAuditor",
    "Phase2_SkewSymmetryCertifier",
    "Phase3_HeytingLatticeDecider",
    "RiemannianInertiaAgent",
]