# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Inertia Agent (Soberano del Momentum Ciber-Físico)       ║
║ Ruta   : app/agents/physics/riemannian_inertia_agent.py                      ║
║ Versión: 4.0.0-Topos-Heyting-Liouville-Gauge-Strict                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA SIMPLÉCTICA (Rigor Doctoral): ──────────────
Este módulo consagra al Agente Soberano y Observador Activo que gobierna al
funtor físico ``riemannian_inertia_modulator.py``. Reside en el hiperespacio
del estrato superior de Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0), supervisando la
dinámica de-confinada del espacio de fase de la Malla.

Su mandato axiomático es orquestar el ciclo OODA sobre el operador de momento
giroscópico $$\mathcal{W}$$ en el fibrado cotangente $$T^*\mathcal{M}$$,
sometiendo el flujo de intenciones semánticas a restricciones geométricas
rigurosas. Actúa aplicando una "Fuerza de Lorentz" informacional que desvía
trayectorias de alto riesgo (alucinaciones probabilísticas del LLM) hacia
sumideros de disipación sin alterar el Hamiltoniano de energía basal del
sistema. Toda contención de fallos se confina de manera síncrona y absoluta
al plano lógico de software mediante el colapso del retículo de Heyting,
repudiando de raíz dependencias de hardware mecánico o disyuntores exógenos
en este estrato.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADOS: ────────

  [I1] Preservación del Volumen de Liouville (Divergencia de Fase Nula):
       La evolución del sistema en el espacio de fase de-confinado conserva
       idénticamente la 2-forma simpléctica canónica de Liouville:
       $$\omega = \sum_{i=1}^n dq_i \wedge dp_i \quad\big[42\big]$$
       El agente no recomputa $$\operatorname{div}(\dot{x})$$; audita los
       certificados del motor (norma dual, sondas de Liouville, trabajo nulo)
       que implican
       $$\operatorname{Tr}\bigl(J_{\mathrm{eff}}\,\nabla^2 H\bigr)\equiv 0$$
       por antisimetría de $$J_{\mathrm{eff}}$$ y simetría de Schwarz del
       Hessiano.

  [I2] Antisimetría Euclidiana y Firma de Calibre (Álgebras de Lie):
       El tensor giroscópico debe habitar en $$\mathfrak{so}(n)$$:
       $$\mathcal{W}^\top+\mathcal{W}=\mathbf{0},\qquad
         r_{\mathrm{skew}}
         =\frac{\|\mathcal{W}+\mathcal{W}^\top\|_F}
               {\max(1,\|\mathcal{W}\|_F)}
         \le\varepsilon_{\mathrm{skew}}$$
       Si la métrica $$G$$ está disponible, se audita además la
       $$G$$-antisimetría (álgebra de Lie del grupo que preserva $$G$$):
       $$\mathcal{W}^\top G + G\,\mathcal{W}=\mathbf{0}
         \quad\big[29\big]$$

  [I3] Ley de Trabajo Mecánico Nilpotente (Segunda Ley):
       $$P_{\mathrm{gyro}}
         =\langle\nabla H,\,J_{\mathrm{eff}}\nabla H\rangle
         \equiv 0\pmod{\varepsilon_{\mathrm{machine}}}$$
       El agente confronta el residuo de Neumaier, el residuo estructural
       de pares y las sondas de Liouville contra la tolerancia certificada
       por el motor y contra la política blanda del retículo.

  [I4] Isomorfismo Musical de de Rham–Hodge y Apareamiento Dual:
       Se exige la involutividad del par $$(\flat,\sharp)$$ y la identidad
       de apareamiento, interpretadas como anulación de torsión lógica
       en el fibrado cotangente:
       $$\|\sharp\flat\dot{q}-\dot{q}\|_2\approx 0,\qquad
         \langle p,\dot{q}\rangle
         =\dot{q}^\top G\dot{q}
         =p^\top G^{-1}p$$
       El primer número de Betti no se computa aquí (no hay complejo
       simplicial); su anulación se *interpreta* como la ausencia de
       ciclos de dependencia en el retículo de certificados.

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA): ─────────────────
La transición de estados se rige por un contrato covariante estricto.  El
último morfismo de la Fase $$k$$ es, por construcción, el dominio del
primer morfismo de la Fase $$k+1$$:

  Fase 1 ──► FASE 1: AUDITORÍA DE LIOUVILLE (Phase1_LiouvilleVolumeAuditor)
             Intercepta el certificado espectral del motor y clasifica,
             en el retículo $$\Omega_3$$, la cota de volumen, el
             condicionamiento, la inversa de Wilkinson, el round-trip
             musical y el apareamiento dual.
             Entrega: Phase1ObservationBridge.
             Morfismo terminal: handoff_phase1_to_phase2  ≡  dominio de Fase 2.

  Fase 2 ──► FASE 2: CERTIFICACIÓN DE FIRMA MÉTRICA (Phase2_SkewSymmetryCertifier)
             Recibe el puente certificado por handoff_phase1_to_phase2.
             Valida la antisimetría de $$\mathcal{W}$$, la identidad de
             Gram del bivector, la paridad del rango, la pureza de la
             vorticidad y, si $$G$$ está presente, la $$G$$-antisimetría.
             Entrega: Phase2OrientationBridge.
             Morfismo terminal: handoff_phase2_to_phase3  ≡  dominio de Fase 3.

  Fase 3 ──► FASE 3: COLAPSO TERMODINÁMICO DE HEYTING (Phase3_HeytingLatticeDecider)
             Recibe el puente certificado por handoff_phase2_to_phase3.
             Somete $$J_{\mathrm{eff}}$$ a la auditoría de trabajo
             nilpotente (Neumaier, pares, sondas) y calcula el supremo:
             $$v_{\mathrm{final}}
               = v_{\mathrm{Liouville}}
               \sqcup v_{\mathrm{Skew}}
               \sqcup v_{\mathrm{Work}}$$
             Entrega: InertialGovernanceState.  Si $$v_{\mathrm{final}}=\top$$
             y ``raise_on_veto``, colapsa el retículo en RAM.

ÁLGEBRA DE HEYTING DEL CLASIFICADOR $$\Omega_3$$: ──────────────────────────────
$$\Omega_3=\{\mathtt{COHERENT}\le\mathtt{DEGRADED}\le\mathtt{VETOED}\}$$
es una cadena finita, luego un álgebra de Heyting (todo retículo
distributivo finito lo es):

  join  $$\sqcup$$ : $$\max$$          (supremo de severidad)
  meet  $$\sqcap$$ : $$\min$$          (ínfimo de severidad)
  impl  $$\Rightarrow$$ :
        $$a\Rightarrow b=\top$$ si $$a\le b$$, si no $$b$$
  neg   $$\neg a = a\Rightarrow\bot$$

No es Booleana: $$\neg\neg\mathtt{DEGRADED}=\mathtt{VETOED}
\neq\mathtt{DEGRADED}$$.  El elemento neutro de $$\sqcup$$ es
$$\mathtt{COHERENT}=\bot$$; el de $$\sqcap$$ es
$$\mathtt{VETOED}=\top$$.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema APU Filter / MIC.
# Se conservan stubs robustos, alineados con el modulator 4.0.0, para
# aislamiento analítico.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.riemannian_inertia_modulator import (
        GyroscopicSynthesisData,
        MomentumAuditData,
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
    class MomentumAuditData:
        """Artefacto de Fase 1 del motor, compatible con el modulator 4.0.0."""

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

    @dataclass(frozen=True, slots=True)
    class GyroscopicSynthesisData:
        """Artefacto de Fase 2 del motor, compatible con el modulator 4.0.0."""

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

    @dataclass(frozen=True, slots=True)
    class ThermodynamicVetoData:
        """Artefacto de Fase 3 del motor, compatible con el modulator 4.0.0."""

        effective_dirac_matrix: NDArray[np.float64]
        nilpotent_work_residual: float
        dirac_symmetric_residual: float = 0.0
        work_tolerance: float = 1e-10
        is_symplectically_passive: bool = True
        pairwise_work_residual: float = 0.0
        liouville_probe_residual: float = 0.0
        relative_skew_residual: float = 0.0

    RiemannianInertiaModulator = Any  # type: ignore[misc, assignment]


logger = logging.getLogger("MIC.Agents.Physics.RiemannianInertiaAgent")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, UMBRALES Y TOLERANCIAS ELÁSTICAS EN SOFTWARE
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# ── Límites duros de integridad física y numérica ───────────────────────────
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

# Versión canónica del agente.
__version__: Final[str] = "4.0.0-Topos-Heyting-Liouville-Gauge-Strict"


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

    El orden es de *severidad*, no de verdad clásica.  Las operaciones
    de retículo se exponen como métodos para hacer explícita el álgebra.
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


class SkewSignatureCollapse(RiemannianInertiaAgentError):
    r"""$$\mathcal{W}$$ abandona $$\mathfrak{so}(n)$$ o viola la $$G$$-antisimetría."""

    pass


class ExteriorAlgebraCollapse(RiemannianInertiaAgentError):
    """El bivector $$p\wedge\omega$$ viola la identidad de Gram o la paridad de rango."""

    pass


class ThermodynamicPassivityCollapse(RiemannianInertiaAgentError):
    r"""$$J_{\mathrm{eff}}$$ realiza trabajo neto o pierde la pasividad simpléctica."""

    pass


class HeytingLatticeVeto(RiemannianInertiaAgentError):
    r"""
    Detonada síncronamente cuando el retículo de Heyting toca el supremo $$\top$$.

    Aniquila el estado puramente en software (RAM), sin hardware exógeno.
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DE GOBERNANZA CATEGÓRICA
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1ObservationBridge:
    r"""
    Certificado de la Fase 1.  Precondición formal de la Fase 2.

    El campo ``liouville_verdict`` es el supremo local:

        $$v_{\mathrm{Liouville}}
          =\bigsqcup\bigl\{
              v_{\mathrm{mom}},\,
              v_{\kappa},\,
              v_{\mathrm{inv}},\,
              v_{\mathrm{pair}},\,
              v_{\mathrm{musical}},\,
              v_{\mathrm{spec}}
            \bigr\}$$
    """

    momentum_data: MomentumAuditData
    liouville_verdict: InertialHeytingVerdict
    momentum_bound_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    metric_condition_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    inverse_consistency_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    pairing_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    musical_roundtrip_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    spectral_gap_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    momentum_margin: float = 0.0
    pairing_residual: float = 0.0
    musical_roundtrip_residual: float = 0.0
    kinetic_energy_primal: float = 0.0
    kinetic_energy_dual: float = 0.0
    spectral_minimum: float = 1.0
    spectral_maximum: float = 1.0


@dataclass(frozen=True, slots=True)
class Phase2OrientationBridge:
    r"""
    Certificado de la Fase 2.  Precondición formal de la Fase 3.

    El campo ``skew_verdict`` es el supremo local:

        $$v_{\mathrm{Skew}}
          =\bigsqcup\bigl\{
              v_{\mathrm{anti}},\,
              v_{\Omega},\,
              v_{\mathrm{Gram}},\,
              v_{\mathrm{rank}},\,
              v_{G\text{-skew}}
            \bigr\}$$
    """

    phase1_bridge: Phase1ObservationBridge
    synthesis_data: GyroscopicSynthesisData
    skew_verdict: InertialHeytingVerdict
    antisymmetry_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    vorticity_projection_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    gram_identity_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    even_rank_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    gauge_signature_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    relative_antisymmetry_residual: float = 0.0
    vorticity_projection_ratio: float = 0.0
    wedge_gram_residual: float = 0.0
    gauge_skew_residual: float = 0.0
    skew_numerical_rank: int = 0
    rank_is_even: bool = True


@dataclass(frozen=True, slots=True)
class InertialGovernanceState:
    r"""
    Objeto final del endofuntor $$Z_{\mathrm{Inertia}}$$.  Estado lógico supremo.

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
    work_passivity_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    pairwise_work_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    liouville_probe_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    dirac_symmetry_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    work_margin: float = 0.0
    pairwise_work_residual: float = 0.0
    liouville_probe_residual: float = 0.0
    dirac_relative_skew_residual: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → AUDITORÍA DEL VOLUMEN DE LIOUVILLE (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_LiouvilleVolumeAuditor:
    r"""
    Fase 1: interroga el espectrómetro de momentum del motor y proyecta
    el estado tensional del flujo al retículo de severidad de Heyting.

    Audita, en orden de dependencia lógica:

        1. cota dura y blanda del momentum (volumen de Liouville);
        2. número de condición de la métrica;
        3. residuo de consistencia de la inversa;
        4. identidad de apareamiento dual / energía cinética;
        5. round-trip del isomorfismo musical;
        6. hueco espectral $$(\lambda_{\min},\lambda_{\max})$$.

    El morfismo terminal ``handoff_phase1_to_phase2`` eleva el certificado
    a precondición formal de la Fase 2.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.1  Utilidades elementales de validación numérica y lógica
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

    def _get_required_attribute(self, obj: object, attr: str, name: str) -> Any:
        """Extrae un atributo requerido del DTO."""
        if not hasattr(obj, attr):
            raise RiemannianInertiaAgentError(
                f"{name} no contiene el campo '{attr}'."
            )
        return getattr(obj, attr)

    def _get_required_finite_float(self, obj: object, attr: str, name: str) -> float:
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
        """
        Extrae un campo flotante finito positivo opcional.

        Si el valor existe pero no es positivo, se retorna el valor por
        defecto para no romper la gobernanza por campos auxiliares mal
        inicializados, y se registra una advertencia.
        """
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
    ) -> NDArray[np.float64] | None:
        """
        Valida un campo vectorial/matricial opcional o requerido.

        Condiciones: convertible a float64, no vacío, completamente finito.
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
            raise RiemannianInertiaAgentError(f"{name}.{attr} no puede ser vacío.")
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
            raise RiemannianInertiaAgentError(f"{name} debe ser una matriz cuadrada.")
        if not np.all(np.isfinite(matrix)):
            raise RiemannianInertiaAgentError(f"{name} contiene valores no finitos.")
        norm = float(np.linalg.norm(matrix, ord="fro"))
        if not math.isfinite(norm):
            raise RiemannianInertiaAgentError(f"La norma de {name} no es finita.")
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
        r"""Ínfimo (meet): $$v=\bigsqcap_i v_i=\min_i v_i$$.  Neutro: $$\top$$."""
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

    # ──────────────────────────────────────────────────────────────────────────
    # 1.2  Validación y extracción de certificados de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_momentum_data(
        self,
        momentum_data: MomentumAuditData,
    ) -> dict[str, Any]:
        """
        Verifica la integridad del artefacto espectral producido por el motor.

        Retorna un diccionario de campos normalizados, incluyendo las
        extensiones 4.0.0 (apareamiento, musical, espectro).
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
            momentum_data, "reconstructed_velocity", "momentum_data", required=False
        )

        metric_condition_number = self._get_optional_nonnegative_finite_float(
            momentum_data, "metric_condition_number", 1.0, "momentum_data"
        )
        if metric_condition_number < 1.0:
            metric_condition_number = 1.0

        inverse_consistency_residual = self._get_optional_nonnegative_finite_float(
            momentum_data, "inverse_consistency_residual", 0.0, "momentum_data"
        )
        pairing_residual = self._get_optional_nonnegative_finite_float(
            momentum_data, "pairing_residual", 0.0, "momentum_data"
        )
        musical_roundtrip_residual = self._get_optional_nonnegative_finite_float(
            momentum_data, "musical_roundtrip_residual", 0.0, "momentum_data"
        )
        kinetic_energy_primal = self._get_optional_nonnegative_finite_float(
            momentum_data, "kinetic_energy_primal", 0.0, "momentum_data"
        )
        kinetic_energy_dual = self._get_optional_nonnegative_finite_float(
            momentum_data, "kinetic_energy_dual", 0.0, "momentum_data"
        )
        spectral_minimum = self._get_optional_finite_float(
            momentum_data, "spectral_minimum", 1.0, "momentum_data"
        )
        spectral_maximum = self._get_optional_finite_float(
            momentum_data, "spectral_maximum", 1.0, "momentum_data"
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
            "spectral_minimum": spectral_minimum,
            "spectral_maximum": spectral_maximum,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 1.3  Clasificadores locales de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_momentum_bound(
        self,
        momentum_norm: float,
        is_bounded: bool,
    ) -> InertialHeytingVerdict:
        """
        Clasifica la cota de volumen de Liouville asociada al momentum.

        - VETOED   : el motor lo declara no acotado o supera el límite duro.
        - DEGRADED : supera el límite blando.
        - COHERENT : caso contrario.
        """
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
        r"""
        Clasifica el hueco espectral de $$G$$.

        Se exige $$\lambda_{\min}>0$$, $$\lambda_{\max}\ge\lambda_{\min}$$
        y que el cociente no contradiga $$\kappa$$ certificado.
        """
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

    def _compute_momentum_margin(self, momentum_norm: float) -> float:
        r"""
        Margen normalizado de seguridad del momentum:

            $$\mathrm{margin}=\max\bigl(0,\,1-\|p\|/P_{\mathrm{hard}}\bigr)$$
        """
        if _MOMENTUM_HARD_LIMIT <= 0.0:
            return 0.0
        return self._clamp01(1.0 - (momentum_norm / _MOMENTUM_HARD_LIMIT))

    # ──────────────────────────────────────────────────────────────────────────
    # 1.4  Núcleo terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _audit_liouville_volume(
        self,
        momentum_data: MomentumAuditData,
    ) -> Phase1ObservationBridge:
        """
        Evalúa la preservación del volumen simpléctico y la coherencia
        métrica a partir del certificado del motor.

        Clasifica el estado en $$\Omega_3$$ y devuelve el puente formal
        hacia la Fase 2.
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

        liouville_verdict = self._heyting_join(
            (
                momentum_bound_verdict,
                metric_condition_verdict,
                inverse_consistency_verdict,
                pairing_verdict,
                musical_roundtrip_verdict,
                spectral_gap_verdict,
            )
        )
        momentum_margin = self._compute_momentum_margin(fields["momentum_norm"])

        logger.debug(
            "Fase 1 Liouville: ||p||=%.6e, bounded=%s, κ=%.6e, inv_res=%.6e, "
            "pair=%.6e, musical=%.6e, λ∈[%.6e, %.6e], margin=%.6f, verdict=%s",
            fields["momentum_norm"],
            fields["is_bounded"],
            fields["metric_condition_number"],
            fields["inverse_consistency_residual"],
            fields["pairing_residual"],
            fields["musical_roundtrip_residual"],
            fields["spectral_minimum"],
            fields["spectral_maximum"],
            momentum_margin,
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
        )

    def execute_phase1(
        self,
        momentum_data: MomentumAuditData,
    ) -> Phase1ObservationBridge:
        """
        Método terminal operativo de la Fase 1.

        Su salida constituye el dominio formal de
        ``handoff_phase1_to_phase2``, que es el morfismo inicial de la Fase 2.
        """
        return self._audit_liouville_volume(momentum_data)

    def handoff_phase1_to_phase2(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> Phase1ObservationBridge:
        r"""
        Morfismo de transición

            $$\Phi_{12}:
              \mathrm{Phase1ObservationBridge}
              \longrightarrow
              \mathrm{Phase1ObservationBridge}.$$

        Poscondición de la Fase 1  ≡  precondición de la Fase 2:

            el puente es una instancia bien tipada de
            ``Phase1ObservationBridge``, sus veredictos habitan en
            $$\Omega_3$$, el join almacenado coincide con el join de
            los clasificadores granulares, y los residuales son finitos
            no negativos.

        Este método es la definición formal final de la Fase 1 y, a la
        vez, el dominio sobre el que la Fase 2 certifica la firma
        métrica.  ``Phase2_SkewSymmetryCertifier`` comienza invocándolo.

        No colapsa el retículo: el veto de política se reserva a la
        Fase 3 (o al orquestador en modo ``fail_fast``).
        """
        self._validate_phase1_bridge_structure(phase1_bridge)
        self._assert_phase1_lattice_consistency(phase1_bridge)
        return phase1_bridge

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
        )
        for field in verdict_fields:
            if hasattr(phase1_bridge, field):
                self._validate_verdict(
                    getattr(phase1_bridge, field),
                    f"phase1_bridge.{field}",
                )

        float_fields = (
            "momentum_margin",
            "pairing_residual",
            "musical_roundtrip_residual",
            "kinetic_energy_primal",
            "kinetic_energy_dual",
        )
        for field in float_fields:
            if hasattr(phase1_bridge, field):
                self._as_nonnegative_finite_float(
                    getattr(phase1_bridge, field),
                    f"phase1_bridge.{field}",
                )

        for field in ("spectral_minimum", "spectral_maximum"):
            if hasattr(phase1_bridge, field):
                self._as_finite_float(
                    getattr(phase1_bridge, field),
                    f"phase1_bridge.{field}",
                )

    def _assert_phase1_lattice_consistency(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> None:
        """Exige que el join almacenado coincida con el de los granulares."""
        recomputed = self._heyting_join(
            (
                phase1_bridge.momentum_bound_verdict,
                phase1_bridge.metric_condition_verdict,
                phase1_bridge.inverse_consistency_verdict,
                phase1_bridge.pairing_verdict,
                phase1_bridge.musical_roundtrip_verdict,
                phase1_bridge.spectral_gap_verdict,
            )
        )
        if recomputed is not phase1_bridge.liouville_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del retículo en Φ₁₂: "
                f"join granular = {recomputed.name}, "
                f"almacenado = {phase1_bridge.liouville_verdict.name}."
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → CERTIFICACIÓN DE LA FIRMA MÉTRICA Y DE GAUGE (Orient)
#          continuación formal de handoff_phase1_to_phase2
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_SkewSymmetryCertifier(Phase1_LiouvilleVolumeAuditor):
    r"""
    Fase 2: recibe el puente certificado por ``handoff_phase1_to_phase2``
    y certifica que la proyección de Löwner del motor no inyecte trazas
    diagonales espurias ni componentes simétricas disipativas.

    Audita:

        1. antisimetría relativa de $$\mathcal{W}$$ en $$\mathfrak{so}(n)$$;
        2. pureza de la vorticidad proyectada al cono de 2-formas;
        3. identidad de Gram del bivector $$p\wedge\omega$$;
        4. paridad del rango numérico (toda matriz antisimétrica real
           tiene rango par);
        5. $$G$$-antisimetría de calibre, si la métrica está presente;
        6. coherencia del puente de Fase 1.

    El morfismo terminal ``handoff_phase2_to_phase3`` eleva el certificado
    a precondición formal de la Fase 3.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 2.1  Continuación inmediata del handoff de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_observation(
        self,
        phase1_bridge: Phase1ObservationBridge,
    ) -> Phase1ObservationBridge:
        """
        Primer método de la Fase 2.

        Es la continuación literal de ``handoff_phase1_to_phase2``:
        revalida el certificado y entrega el puente sobre el que se
        edifica la firma métrica de $$\mathcal{W}$$.
        """
        return self.handoff_phase1_to_phase2(phase1_bridge)

    def _validate_phase1_bridge(self, phase1_bridge: Phase1ObservationBridge) -> None:
        """Fachada de revalidación usada por la Fase 3 al inspeccionar el puente."""
        self._receive_certified_observation(phase1_bridge)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.2  Validación del artefacto giroscópico
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_synthesis_data(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> dict[str, Any]:
        """
        Verifica la integridad del artefacto giroscópico producido por el motor.

        Retorna campos normalizados, incluidas las extensiones 4.0.0
        (residuo relativo, Gram, rango).
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
            synthesis_data, "skew_symmetric_tensor", "synthesis_data", required=False
        )
        self._validate_finite_array_attribute(
            synthesis_data, "vorticity_two_form", "synthesis_data", required=False
        )
        self._validate_finite_array_attribute(
            synthesis_data, "omega_vector", "synthesis_data", required=False
        )

        gyroscopic_frobenius_norm = self._get_optional_nonnegative_finite_float(
            synthesis_data, "gyroscopic_frobenius_norm", 0.0, "synthesis_data"
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

        vorticity_projection_residual = self._get_optional_nonnegative_finite_float(
            synthesis_data, "vorticity_projection_residual", 0.0, "synthesis_data"
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
            raise RiemannianInertiaAgentError("G_tensor debe ser una matriz cuadrada.")
        if metric.size == 0:
            raise RiemannianInertiaAgentError("G_tensor no puede ser vacía.")
        if not np.all(np.isfinite(metric)):
            raise RiemannianInertiaAgentError(
                "G_tensor contiene valores no finitos."
            )
        return metric

    # ──────────────────────────────────────────────────────────────────────────
    # 2.3  Clasificadores locales de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_antisymmetry(
        self,
        antisymmetry_residual: float,
        is_strictly_skew: bool,
        gyroscopic_frobenius_norm: float,
        certified_relative_residual: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""
        Clasifica la antisimetría de $$\mathcal{W}$$ con el residuo relativo
        documental:

            $$r_{\mathrm{rel}}
              =\frac{\|W+W^\top\|_F}{\max(1,\|W\|_F)}$$

        Se confronta además el residuo relativo que el motor haya
        certificado, tomando el máximo (peor) de ambos.
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
        r"""
        Clasifica la pureza de la vorticidad proyectada:

            $$\mathrm{ratio}
              =\frac{\|\mathrm{sym}(\Omega)\|_F}{\max(1,\|W\|_F)}$$

        Si $$\|W\|$$ es despreciable, la componente simétrica de
        $$\Omega$$ no produce efecto giroscópico y se trata como COHERENT.
        """
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
        r"""Clasifica la identidad de Gram $$\|p\wedge\omega\|_F^2=2\det\mathrm{Gram}$$. """
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
        """
        Toda matriz antisimétrica real tiene rango par.

        Un rango impar se degrada (artefacto de umbral espectral); no se
        veta, coherente con la política del motor 4.0.0.
        """
        computed_even = (rank % 2 == 0)
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

        Si $$W$$ o $$G$$ no están disponibles, la auditoría se omite
        (COHERENT con residuo nulo): no se finge un invariante no medido.
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

    # ──────────────────────────────────────────────────────────────────────────
    # 2.4  Núcleo terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _certify_metric_signature(
        self,
        phase1_bridge: Phase1ObservationBridge,
        synthesis_data: GyroscopicSynthesisData,
        G_tensor: object = None,
    ) -> Phase2OrientationBridge:
        r"""
        Certifica la firma métrica antisimétrica del operador giroscópico.

        El veredicto local de Fase 2 es

            $$v_{\mathrm{Skew}}
              =v_{\mathrm{anti}}
              \sqcup v_{\Omega}
              \sqcup v_{\mathrm{Gram}}
              \sqcup v_{\mathrm{rank}}
              \sqcup v_{G\text{-skew}}$$
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
        gauge_skew_residual, gauge_signature_verdict = self._classify_gauge_signature(
            fields["tensor"],
            metric,
        )

        skew_verdict = self._heyting_join(
            (
                antisymmetry_verdict,
                vorticity_projection_verdict,
                gram_identity_verdict,
                even_rank_verdict,
                gauge_signature_verdict,
            )
        )

        logger.debug(
            "Fase 2 Skew: residual_abs=%.6e, residual_rel=%.6e, strictly_skew=%s, "
            "||W||_F=%.6e, vorticity_ratio=%.6e, Gram=%.6e, rank=%d, "
            "gauge=%.6e, verdict=%s",
            fields["antisymmetry_residual"],
            relative_antisymmetry_residual,
            fields["is_strictly_skew"],
            fields["gyroscopic_frobenius_norm"],
            vorticity_projection_ratio,
            fields["wedge_gram_residual"],
            fields["skew_numerical_rank"],
            gauge_skew_residual,
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
        )

    def execute_phase2(
        self,
        phase1_bridge: Phase1ObservationBridge,
        synthesis_data: GyroscopicSynthesisData,
        G_tensor: object = None,
    ) -> Phase2OrientationBridge:
        """
        Método terminal operativo de la Fase 2.

        Recibe la salida formal de ``execute_phase1`` (vía
        ``handoff_phase1_to_phase2``) y produce la entrada canónica de
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
        Morfismo de transición

            $$\Phi_{23}:
              \mathrm{Phase2OrientationBridge}
              \longrightarrow
              \mathrm{Phase2OrientationBridge}.$$

        Poscondición de la Fase 2  ≡  precondición de la Fase 3:

            el puente es una instancia bien tipada, contiene un
            ``Phase1ObservationBridge`` ya revalidado por $$\Phi_{12}$$,
            sus veredictos habitan en $$\Omega_3$$ y el join almacenado
            coincide con el de los clasificadores granulares.

        Este método es la definición formal final de la Fase 2 y, a la
        vez, el dominio sobre el que la Fase 3 realiza el colapso
        termodinámico.  ``Phase3_HeytingLatticeDecider`` comienza
        invocándolo.
        """
        self._validate_phase2_bridge_structure(phase2_bridge)
        self._assert_phase2_lattice_consistency(phase2_bridge)
        return phase2_bridge

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
        )
        for field in verdict_fields:
            if hasattr(phase2_bridge, field):
                self._validate_verdict(
                    getattr(phase2_bridge, field),
                    f"phase2_bridge.{field}",
                )

        float_fields = (
            "relative_antisymmetry_residual",
            "vorticity_projection_ratio",
            "wedge_gram_residual",
            "gauge_skew_residual",
        )
        for field in float_fields:
            if hasattr(phase2_bridge, field):
                self._as_nonnegative_finite_float(
                    getattr(phase2_bridge, field),
                    f"phase2_bridge.{field}",
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
            self._as_bool(phase2_bridge.rank_is_even, "phase2_bridge.rank_is_even")

    def _assert_phase2_lattice_consistency(
        self,
        phase2_bridge: Phase2OrientationBridge,
    ) -> None:
        """Exige que el join almacenado coincida con el de los granulares."""
        recomputed = self._heyting_join(
            (
                phase2_bridge.antisymmetry_verdict,
                phase2_bridge.vorticity_projection_verdict,
                phase2_bridge.gram_identity_verdict,
                phase2_bridge.even_rank_verdict,
                phase2_bridge.gauge_signature_verdict,
            )
        )
        if recomputed is not phase2_bridge.skew_verdict:
            raise PhaseHandoffCollapse(
                "Inconsistencia del retículo en Φ₂₃: "
                f"join granular = {recomputed.name}, "
                f"almacenado = {phase2_bridge.skew_verdict.name}."
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → COLAPSO TERMODINÁMICO EN EL RETÍCULO DE HEYTING (Decide & Act)
#          continuación formal de handoff_phase2_to_phase3
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingLatticeDecider(Phase2_SkewSymmetryCertifier):
    r"""
    Fase 3: recibe el puente certificado por ``handoff_phase2_to_phase3``
    y consolida el ciclo OODA.

    Realiza la operación supremo algebraico sobre los veredictos locales,
    imponiendo el estado determinista de la transacción estrictamente en
    memoria de software:

        $$v_{\mathrm{final}}
          =v_{\mathrm{Liouville}}
          \sqcup v_{\mathrm{Skew}}
          \sqcup v_{\mathrm{Work}}$$
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 3.1  Continuación inmediata del handoff de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_orientation(
        self,
        phase2_bridge: Phase2OrientationBridge,
    ) -> Phase2OrientationBridge:
        """
        Primer método de la Fase 3.

        Es la continuación literal de ``handoff_phase2_to_phase3``:
        revalida el certificado y entrega el puente sobre el que se
        calcula el supremo termodinámico.
        """
        return self.handoff_phase2_to_phase3(phase2_bridge)

    def _validate_phase2_bridge(self, phase2_bridge: Phase2OrientationBridge) -> None:
        """Fachada conservada: revalida el puente de Fase 2 vía $$\Phi_{23}$$. """
        self._receive_certified_orientation(phase2_bridge)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.2  Validación del artefacto termodinámico
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_veto_data(
        self,
        veto_data: ThermodynamicVetoData,
    ) -> dict[str, Any]:
        """
        Verifica la integridad del artefacto termodinámico producido por el motor.

        Retorna campos normalizados, incluidas las extensiones 4.0.0
        (trabajo de pares, sondas de Liouville, $$r_{\mathrm{skew}}$$ de
        $$J_{\mathrm{eff}}$$).
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
        dirac_symmetric_residual = self._get_optional_nonnegative_finite_float(
            veto_data, "dirac_symmetric_residual", 0.0, "veto_data"
        )
        pairwise_work_residual = self._get_optional_nonnegative_finite_float(
            veto_data, "pairwise_work_residual", 0.0, "veto_data"
        )
        liouville_probe_residual = self._get_optional_nonnegative_finite_float(
            veto_data, "liouville_probe_residual", 0.0, "veto_data"
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

        return {
            "nilpotent_work_residual": nilpotent_work_residual,
            "is_symplectically_passive": is_symplectically_passive,
            "work_tolerance": work_tolerance,
            "dirac_symmetric_residual": dirac_symmetric_residual,
            "dirac_frobenius_norm": dirac_frobenius_norm,
            "pairwise_work_residual": pairwise_work_residual,
            "liouville_probe_residual": liouville_probe_residual,
            "certified_relative_skew": certified_relative_skew,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 3.3  Clasificadores locales de Fase 3
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
        r"""
        Clasifica la componente simétrica residual de $$J_{\mathrm{eff}}$$:

            $$r_{\mathrm{rel}}
              =\frac{\|J_{\mathrm{eff}}+J_{\mathrm{eff}}^\top\|_F}
                    {\max(1,\|J_{\mathrm{eff}}\|_F)}$$
        """
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

    def _compute_work_margin(
        self,
        nilpotent_work_residual: float,
        work_tolerance: float,
    ) -> float:
        r"""
        Margen normalizado de trabajo nulo:

            $$\mathrm{margin}
              =\max\bigl(0,\,1-\mathrm{residual}/\mathrm{tolerance}\bigr)$$
        """
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
    ) -> list[str]:
        """Inventario de subobjetos que tocaron el supremo $$\top$$."""
        sources: list[str] = []
        phase1 = phase2_bridge.phase1_bridge

        mapping: tuple[tuple[str, InertialHeytingVerdict], ...] = (
            ("Liouville.momentum", phase1.momentum_bound_verdict),
            ("Liouville.condition", phase1.metric_condition_verdict),
            ("Liouville.inverse", phase1.inverse_consistency_verdict),
            ("Liouville.pairing", phase1.pairing_verdict),
            ("Liouville.musical", phase1.musical_roundtrip_verdict),
            ("Liouville.spectrum", phase1.spectral_gap_verdict),
            ("Skew.antisymmetry", phase2_bridge.antisymmetry_verdict),
            ("Skew.vorticity", phase2_bridge.vorticity_projection_verdict),
            ("Skew.gram", phase2_bridge.gram_identity_verdict),
            ("Skew.rank", phase2_bridge.even_rank_verdict),
            ("Skew.gauge", phase2_bridge.gauge_signature_verdict),
            ("Work.passivity", work_passivity_verdict),
            ("Work.pairwise", pairwise_work_verdict),
            ("Work.liouville_probe", liouville_probe_verdict),
            ("Work.dirac_symmetry", dirac_symmetry_verdict),
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
    ) -> None:
        """Materializa el colapso del retículo con diagnóstico granular."""
        sources = self._collect_veto_sources(
            phase2_bridge,
            work_verdict,
            work_passivity_verdict,
            pairwise_work_verdict,
            liouville_probe_verdict,
            dirac_symmetry_verdict,
        )
        source_detail = ", ".join(sources) if sources else "Unknown"
        phase1 = phase2_bridge.phase1_bridge
        momentum_norm = self._get_required_nonnegative_finite_float(
            phase1.momentum_data, "momentum_norm", "momentum_data"
        )

        if phase1.momentum_bound_verdict is InertialHeytingVerdict.VETOED:
            exc_type: type[RiemannianInertiaAgentError] = LiouvilleVolumeCollapse
        elif phase1.metric_condition_verdict is InertialHeytingVerdict.VETOED:
            exc_type = MetricConditionCollapse
        elif phase1.inverse_consistency_verdict is InertialHeytingVerdict.VETOED:
            exc_type = InverseCoherenceCollapse
        elif phase1.pairing_verdict is InertialHeytingVerdict.VETOED:
            exc_type = DualPairingCollapse
        elif phase1.musical_roundtrip_verdict is InertialHeytingVerdict.VETOED:
            exc_type = MusicalIsomorphismCollapse
        elif phase2_bridge.antisymmetry_verdict is InertialHeytingVerdict.VETOED:
            exc_type = SkewSignatureCollapse
        elif phase2_bridge.gauge_signature_verdict is InertialHeytingVerdict.VETOED:
            exc_type = SkewSignatureCollapse
        elif phase2_bridge.gram_identity_verdict is InertialHeytingVerdict.VETOED:
            exc_type = ExteriorAlgebraCollapse
        elif (
            work_passivity_verdict is InertialHeytingVerdict.VETOED
            or pairwise_work_verdict is InertialHeytingVerdict.VETOED
            or liouville_probe_verdict is InertialHeytingVerdict.VETOED
            or dirac_symmetry_verdict is InertialHeytingVerdict.VETOED
        ):
            exc_type = ThermodynamicPassivityCollapse
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
            f"work_tol = {fields['work_tolerance']:.6e}. "
            "Transacción aniquilada en RAM."
        )
        raise exc_type(message)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.4  Núcleo terminal de la Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_thermodynamic_lattice(
        self,
        phase2_bridge: Phase2OrientationBridge,
        veto_data: ThermodynamicVetoData,
        raise_on_veto: bool = True,
    ) -> InertialGovernanceState:
        r"""
        Consolida los veredictos locales en un único supremo algebraico:

            $$v_{\mathrm{final}}
              =\max(v_{\mathrm{Liouville}},\,v_{\mathrm{Skew}},\,v_{\mathrm{Work}})$$

        Si $$v_{\mathrm{final}}=\top$$ y ``raise_on_veto``, colapsa el
        retículo lanzando la excepción más específica disponible,
        aniquilando la transacción en RAM.
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

        work_verdict = self._heyting_join(
            (
                work_passivity_verdict,
                pairwise_work_verdict,
                liouville_probe_verdict,
                dirac_symmetry_verdict,
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
            )

        logger.debug(
            "Fase 3 Lattice: work=%.6e, pares=%.6e, Liouville=%.6e, "
            "work_tol=%.6e, dirac_r=%.6e, verdicts=(L=%s, S=%s, W=%s), "
            "final=%s, valid=%s, work_margin=%.6f",
            fields["nilpotent_work_residual"],
            fields["pairwise_work_residual"],
            fields["liouville_probe_residual"],
            fields["work_tolerance"],
            dirac_relative,
            certified_phase2.phase1_bridge.liouville_verdict.name,
            certified_phase2.skew_verdict.name,
            work_verdict.name,
            final_supremum_verdict.name,
            is_epistemologically_valid,
            work_margin,
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
        )

    def execute_phase3(
        self,
        phase2_bridge: Phase2OrientationBridge,
        veto_data: ThermodynamicVetoData,
        raise_on_veto: bool = True,
    ) -> InertialGovernanceState:
        """
        Método terminal de la Fase 3.

        Recibe la salida formal de ``execute_phase2`` (vía
        ``handoff_phase2_to_phase3``), retorna el estado lógico supremo
        y, opcionalmente, colapsa el retículo ante un veto.
        """
        return self._evaluate_thermodynamic_lattice(
            phase2_bridge,
            veto_data,
            raise_on_veto,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: RIEMANNIAN INERTIA AGENT
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaAgent(Morphism, Phase3_HeytingLatticeDecider):
    r"""
    Soberano del Momentum Ciber-Físico.

    Inyecta la Fuerza de Lorentz informacional gobernando al motor físico
    ``RiemannianInertiaModulator`` en un estricto ciclo OODA.

    Composición funtorial:

        $$F_{\mathrm{Agent}}
          =F_{\mathrm{Phase3}}
          \circ\Phi_{23}
          \circ F_{\mathrm{Phase2}}
          \circ\Phi_{12}
          \circ F_{\mathrm{Phase1}}$$

    Entrelazado Motor/Agente:

        Motor Fase 1 → Agente Fase 1 → $$\Phi_{12}$$
        Motor Fase 2 → Agente Fase 2 → $$\Phi_{23}$$
        Motor Fase 3 → Agente Fase 3
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

        No duplica la validación espectral del motor: sólo rechaza
        tensores vacíos, no finitos o dimensionalmente incompatibles
        *antes* de invocar el funtor físico.
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
        Ejecuta el ciclo categórico y topológico entrelazando las fases del
        motor físico con las auditorías locales del Agente Soberano.

        Fase 1 → Auditoría del volumen de Liouville y coherencia musical.
        Fase 2 → Certificación de la firma métrica y de calibre.
        Fase 3 → Colapso termodinámico en el retículo de Heyting.

        Si ``fail_fast`` es verdadero, un veto local detiene el ciclo
        antes de invocar la fase siguiente del motor.
        """
        self._validate_governance_payload(
            q_dot, grad_H, G_tensor, G_inv, J_base, vorticity_matrix
        )

        # --- OODA PASO 1: Observación física y auditoría lógica ---
        momentum_data = self._motor.execute_phase1(
            q_dot=q_dot,
            G_tensor=G_tensor,
            G_inv=G_inv,
        )
        phase1_bridge = self.execute_phase1(momentum_data)
        # Φ₁₂ se invoca dentro de execute_phase2; fail-fast se decide aquí.
        self._maybe_fail_fast(
            phase1_bridge.liouville_verdict,
            "Fase 1 (Liouville)",
            fail_fast,
            LiouvilleVolumeCollapse,
            f"‖p‖ = {phase1_bridge.momentum_data.momentum_norm:.6e}.",
        )

        # --- OODA PASO 2: Orientación métrica y auditoría de antisimetría ---
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
            "Fase 2 (Skew/Gauge)",
            fail_fast,
            SkewSignatureCollapse,
            f"r_skew = {phase2_bridge.relative_antisymmetry_residual:.6e}, "
            f"r_G = {phase2_bridge.gauge_skew_residual:.6e}.",
        )

        # --- OODA PASO 3: Decisión termodinámica y auditoría de trabajo nulo ---
        veto_data = self._motor.execute_phase3(
            grad_H=grad_H,
            J_base=J_base,
            synthesis_data=synthesis_data,
        )
        final_state = self.execute_phase3(
            phase2_bridge=phase2_bridge,
            veto_data=veto_data,
            raise_on_veto=raise_on_veto,
        )

        logger.info(
            "Gobernanza inercial completada: final=%s, valid=%s, "
            "‖p‖=%.6e, r_skew=%.6e, work=%.6e, Liouville=%.6e.",
            final_state.final_supremum_verdict.name,
            final_state.is_epistemologically_valid,
            phase1_bridge.momentum_data.momentum_norm,
            phase2_bridge.relative_antisymmetry_residual,
            final_state.veto_data.nilpotent_work_residual,
            final_state.liouville_probe_residual,
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
    "SkewSignatureCollapse",
    "ExteriorAlgebraCollapse",
    "ThermodynamicPassivityCollapse",
    "HeytingLatticeVeto",
    "Phase1ObservationBridge",
    "Phase2OrientationBridge",
    "InertialGovernanceState",
    "Phase1_LiouvilleVolumeAuditor",
    "Phase2_SkewSymmetryCertifier",
    "Phase3_HeytingLatticeDecider",
    "RiemannianInertiaAgent",
]