# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Inertia Agent (Soberano del Momentum Ciber-Físico)       ║
║ Ruta   : app/agents/physics/riemannian_inertia_agent.py                      ║
║ Versión: 3.1.0-Topos-Heyting-Symplectic-Pure-Software-Strict                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA SIMPLÉCTICA (Rigor Doctoral): ──────────────
Este módulo consagra al Agente Soberano y Observador Activo que gobierna al 
funtor físico `riemannian_inertia_modulator.py`. Reside en el hiperespacio 
del estrato superior de Sabiduría ($$V_{\mathbb{W}}$$, Nivel 0), supervisando la 
dinámica de-confinada del espacio de fase de la Malla.

Su mandato axiomático es orquestar el ciclo OODA sobre el operador de momento 
giroscópico $$\mathcal{W}$$ en el fibrado cotangente $$T^*\mathcal{M}$$, 
sometiendo el flujo de intenciones semánticas a restricciones geométricas rigurosas. 
Actúa aplicando una "Fuerza de Lorentz" informacional que desvía trayectorias de 
alto riesgo (alucinaciones probabilísticas del LLM) hacia sumideros de disipación 
sin alterar el Hamiltoniano de energía basal del sistema. Toda contención de 
fallos se confina de manera síncrona y absoluta al plano lógico de software 
mediante el colapso del retículo de Heyting, repudiando de raíz dependencias 
de hardware mecánico o disyuntores exógenos en este estrato.

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES CONSERVATIVAS PRESERVADOS: ────────

  [I1] Preservación del Volumen de Liouville (Divergencia de Fase Nula):
       La evolución del sistema en el espacio de fase de-confinado conserva
       idénticamente la 2-forma simpléctica canónica de Liouville:
       $$\omega = \sum_{i=1}^n dq_i \wedge dp_i \quad\big[42\big]$$
       Esto exige que el flujo inducido por el operador efectivo de Dirac 
       $$J_{\mathrm{eff}}(x) = J(x) + \mathcal{W}_{\mathrm{proj}}(p)$$ sea 
       estrictamente libre de divergencia en la variedad simpléctica $$\mathcal{M}$$:
       $$\operatorname{div}(\dot{x}) = \sum_{k=1}^{2n} \frac{\partial \dot{x}_k}{\partial x_k} = \operatorname{Tr}\left( J_{\mathrm{eff}}(x) \frac{\partial^2 H}{\partial x^2} \right) + \operatorname{Tr}\left( \frac{\partial J_{\mathrm{eff}}(x)}{\partial x} \operatorname{diag}(\nabla H(x)) \right) \equiv 0 \pmod{\varepsilon_{\mathrm{machine}}}$$
       El primer término se anula por la simetría del Hessiano de Schwarz y la 
       antisimetría estricta de $$J_{\mathrm{eff}}$$; el segundo término se anula 
       puesto que la diagonal del tensor giroscópico $$\mathcal{W}_{\mathrm{proj}}(p)$$ 
       se construye mediante el producto exterior de-confinado:
       $$\mathcal{W}_{ii}(p) \equiv 0 \implies \frac{\partial \mathcal{W}_{ii}(p)}{\partial p_i} \equiv 0$$

  [I2] Antisimetría Métrica y Firma de Calibre (Álgebra de Lie $$\mathfrak{so}(n)$$):
       El tensor giroscópico de Lorentz $$\mathcal{W}_{\mathrm{proj}}(p)$$ debe habitar 
       estrictamente en el álgebra de Lie del cono antisimétrico w.r.t. el tensor 
       métrico de Riemann $$G_{\mu\nu}$$:
       $$\mathcal{W}_{\mathrm{proj}}^\top G_{\mu\nu} + G_{\mu\nu} \mathcal{W}_{\mathrm{proj}} = \mathbf{0} \quad\big[29\big]$$
       El agente audita este invariante mediante el residuo relativo de Frobenius:
       $$r_{\mathrm{skew}} = \frac{\|\mathcal{W}_{\mathrm{proj}} + \mathcal{W}_{\mathrm{proj}}^\top\|_F}{\max(1.0, \|\mathcal{W}_{\mathrm{proj}}\|_F)} \le \varepsilon_{\mathrm{skew}} \quad\big[29\big]$$

  [I3] Ley de Trabajo Mecánico Nilpotente (Segunda Ley):
       Para garantizar la pasividad estricta de la Unidad de Punto Flotante (FPU), 
       la fuerza inercial inyectada debe ser ortogonal al flujo del gradiente 
       del Hamiltoniano de energía, asegurando un trabajo mecánico neto nulo:
       $$P_{\mathrm{gyro}} = \langle \nabla_p H, \mathcal{W}_{\mathrm{proj}}(p) \nabla_p H \rangle_G \equiv 0 \pmod{\varepsilon_{\mathrm{machine}}} \quad\big[29, 36\big]$$
       Este cálculo se evalúa mediante sumación compensada de Kahan en la Fase 3 
       para expurgar la deriva de redondeo de la mantisa de coma flotante (IEEE-754).

  [I4] Isomorfismo de de Rham-Hodge y Anulación de Torsión:
       El transporte paralelo del momentum covariante a lo largo de las curvas de 
       decisión se subordina a la conexión única de Levi-Civita libre de torsión 
       ($$T(X,Y) = 0$$) compatible con la métrica:
       $$\nabla_\gamma G_{\mu\nu} = 0 \quad\big[41\big]$$
       Se exige que el primer número de Betti de de Rham sea nulo para aniquilar 
       los "socavones lógicos" (ciclos de dependencias circulares) en la Malla:
       $$\beta_1 \equiv \dim H^1_{\mathrm{dR}}(K; \mathbb{F}_2) = 0 \quad\big[33\big]$$

ESTRUCTURA DE TRES FASES ANIDADAS (Composición Funtorial OODA): ─────────────────
La transición de estados se rige por un contrato covariante estricto que compone 
secuencialmente tres morfismos encadenados por sus DTOs inmutables de handoff formal:

  Fase 1 ──► FASE 1: AUDITORÍA DE LIOUVILLE (Phase1_LiouvilleVolumeAuditor)
             Intercepta el covector de momentum $$p_\mu = G_{\mu\nu}\dot{q}^\nu$$ y 
             audita la cota del volumen en el espacio de fase mediante la norma dual:
             $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} \le P_{\max} \quad\big[29\big]$$
             Entrega: Phase1ObservationBridge como precondición formal de la Fase 2.

  Fase 2 ──► FASE 2: CERTIFICACIÓN DE FIRMA MÉTRICA (Phase2_SkewSymmetryCertifier)
             Valida la antisimetría del operador de Lorentz $$\mathcal{W}_{\mathrm{proj}}$$ y la 
             idempotencia del proyector en la variedad de Grassmann [4, 6].
             Entrega: Phase2OrientationBridge como precondición formal de la Fase 3.

  Fase 3 ──► FASE 3: COLAPSO TERMODINÁMICO DE HEYTING (Phase3_HeytingLatticeDecider)
             Somete la matriz de Dirac efectiva $$J_{\mathrm{eff}} = J + \mathcal{W}_{\mathrm{proj}}$$ a la 
             auditoría de trabajo nilpotente en la FPU mediante sumación de Kahan.
             Entrega: InertialGovernanceState como certificado terminal del funtor.
"""

import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final, Iterable
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema APU Filter.
# Se conservan stubs robustos para aislamiento analítico.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.physics.riemannian_inertia_modulator import (
        MomentumAuditData,
        GyroscopicSynthesisData,
        ThermodynamicVetoData,
        RiemannianInertiaModulator,
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
        """Artefacto mínimo de Fase 1 para ejecución autónoma."""

        covariant_momentum: NDArray[np.float64]
        momentum_norm: float
        is_bounded: bool
        metric_condition_number: float = 1.0
        inverse_consistency_residual: float = 0.0

    @dataclass(frozen=True, slots=True)
    class GyroscopicSynthesisData:
        """Artefacto mínimo de Fase 2 para ejecución autónoma."""

        skew_symmetric_tensor: NDArray[np.float64]
        antisymmetry_residual: float
        vorticity_projection_residual: float = 0.0
        gyroscopic_frobenius_norm: float = 0.0
        is_strictly_skew: bool = True

    @dataclass(frozen=True, slots=True)
    class ThermodynamicVetoData:
        """Artefacto mínimo de Fase 3 para ejecución autónoma."""

        effective_dirac_matrix: NDArray[np.float64]
        nilpotent_work_residual: float
        dirac_symmetric_residual: float = 0.0
        work_tolerance: float = 1e-10
        is_symplectically_passive: bool = True

    RiemannianInertiaModulator = Any  # type: ignore


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

# ── Umbrales blandos de degradación lógica ──────────────────────────────────
_MOMENTUM_SOFT_LIMIT: Final[float] = 1e6
_CONDITION_SOFT_MAX: Final[float] = 1e6
_INVERSE_RESIDUAL_SOFT_MAX: Final[float] = 1e-7
_SKEW_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-9
_VORTICITY_PROJECTION_SOFT_RATIO: Final[float] = 1e-3
_WORK_SOFT_ABSOLUTE_TOLERANCE: Final[float] = 1e-10
_DIRAC_SYMMETRIC_SOFT_RELATIVE_TOLERANCE: Final[float] = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DE HEYTING Y JERARQUÍA DE EXCEPCIONES LÓGICO-MATEMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════
class InertialHeytingVerdict(IntEnum):
    r"""
    Clasificador de subobjetos en el Topos de la Inercia Riemanniana.

    Álgebra de Boole acotada o Retículo de Heyting Ω₃:

        $$\Omega_3 = \{\text{COHERENT}, \text{DEGRADED}, \text{VETOED}\}$$

    El orden parcial cumple con la estructura:

        $$\text{COHERENT} \le \text{DEGRADED} \le \text{VETOED}$$

    El supremo (join $\sqcup$) se implementa mediante el máximo entero.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


class RiemannianInertiaAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de Inercia Riemanniana."""

    pass


class LiouvilleVolumeCollapse(RiemannianInertiaAgentError):
    r"""
    Detonada si el momentum desgarra asintóticamente la variedad simpléctica.
    Se define cuando la energía cinética en $T^*M$ dada por la norma del momentum
    $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu}$$
    excede la cota elástica de Liouville.
    """

    pass


class HeytingLatticeVeto(RiemannianInertiaAgentError):
    r"""
    Detonada síncronamente cuando el retículo de Heyting toca el supremo $\top$.

    Aniquila el estado puramente en software (RAM), sin hardware exógeno.
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. DTOs INMUTABLES DE GOBERNANZA CATEGÓRICA
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1ObservationBridge:
    r"""
    Certificado de la Fase 1. Precondición formal de la Fase 2.

    El campo `liouville_verdict` contiene el supremo local de Fase 1:

        $$v_{\text{Liouville}} = v_{\text{momentum}} \sqcup v_{\text{metric\_condition}} \sqcup v_{\text{inverse\_consistency}}$$
    """

    momentum_data: MomentumAuditData
    liouville_verdict: InertialHeytingVerdict

    # Extensiones granulares de auditoría (compatibles por defecto).
    momentum_bound_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    metric_condition_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    inverse_consistency_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    momentum_margin: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase2OrientationBridge:
    r"""
    Certificado de la Fase 2. Precondición formal de la Fase 3.

    El campo `skew_verdict` contiene el supremo local de Fase 2:

        $$v_{\text{Skew}} = v_{\text{antisymmetry}} \sqcup v_{\text{vorticity\_projection}}$$
    """

    phase1_bridge: Phase1ObservationBridge
    synthesis_data: GyroscopicSynthesisData
    skew_verdict: InertialHeytingVerdict

    # Extensiones granulares de auditoría (compatibles por defecto).
    antisymmetry_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    vorticity_projection_verdict: InertialHeytingVerdict = (
        InertialHeytingVerdict.COHERENT
    )
    relative_antisymmetry_residual: float = 0.0
    vorticity_projection_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class InertialGovernanceState:
    r"""
    Objeto final del endofuntor Z_Inertia. Estado lógico supremo.

    Contiene:

        - los puentes de Fase 1 y Fase 2;
        - el veredicto termodinámico local de Fase 3;
        - el supremo global:

              $$v_{\text{final}} = v_{\text{Liouville}} \sqcup v_{\text{Skew}} \sqcup v_{\text{Work}}$$

        - la validez epistemológica de la transacción.
    """

    phase2_bridge: Phase2OrientationBridge
    veto_data: ThermodynamicVetoData
    work_verdict: InertialHeytingVerdict
    final_supremum_verdict: InertialHeytingVerdict
    is_epistemologically_valid: bool

    # Extensiones granulares de auditoría (compatibles por defecto).
    work_passivity_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    dirac_symmetry_verdict: InertialHeytingVerdict = InertialHeytingVerdict.COHERENT
    work_margin: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → AUDITORÍA DEL VOLUMEN DE LIOUVILLE (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_LiouvilleVolumeAuditor:
    r"""
    Fase 1: Interroga los datos del espectrómetro de momentum y proyecta
    el estado tensional del flujo al retículo de severidad de Heyting.

    Esta fase audita:
        1. cota dura y blanda del momentum;
        2. número de condición de la métrica;
        3. residuo de consistencia de la inversa métrica.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades elementales de validación numérica y lógica
    # ──────────────────────────────────────────────────────────────────────────
    def _as_finite_float(self, value: object, name: str) -> float:
        """Convierte a float finito, rechazando booleanos y valores no finitos."""
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

    def _get_required_attribute(self, obj: object, attr: str, name: str) -> Any:
        """Extrae un atributo requerido del DTO."""
        if not hasattr(obj, attr):
            raise RiemannianInertiaAgentError(f"{name} no contiene el campo '{attr}'.")
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
        inicializados.
        """
        if not hasattr(obj, attr):
            return default

        value = getattr(obj, attr)
        if value is None:
            return default

        try:
            result = self._as_positive_finite_float(value, f"{name}.{attr}")
        except RiemannianInertiaAgentError:
            return default

        return result

    def _validate_finite_array_attribute(
        self,
        obj: object,
        attr: str,
        name: str,
        required: bool = False,
    ) -> NDArray[np.float64] | None:
        """
        Valida un campo vectorial/matricial opcional o requerido.

        Condiciones:
            - convertible a float64;
            - no vacío;
            - completamente finito.
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

    def _validate_verdict(
        self,
        verdict: object,
        name: str,
    ) -> InertialHeytingVerdict:
        """Valida que un veredicto pertenezca al retículo Ω₃."""
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
        Operación supremo (join) en el retículo de Heyting:

            $$v = \bigsqcup_i v_i = \max_i v_i$$

        El elemento neutro del join para una colección vacía es el elemento
        mínimo del retículo: COHERENT.
        """
        values = tuple(
            self._validate_verdict(verdict, f"verdict[{idx}]").value
            for idx, verdict in enumerate(verdicts)
        )

        if not values:
            return InertialHeytingVerdict.COHERENT

        return InertialHeytingVerdict(max(values))

    def _clamp01(self, value: float) -> float:
        """Proyecta un valor real al intervalo [0, 1]."""
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    # ──────────────────────────────────────────────────────────────────────────
    # Validación y extracción de certificados de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_momentum_data(
        self,
        momentum_data: MomentumAuditData,
    ) -> tuple[float, bool, float, float]:
        """
        Verifica la integridad del artefacto espectral producido por el motor.

        Retorna:
            momentum_norm,
            is_bounded,
            metric_condition_number,
            inverse_consistency_residual.
        """
        if not isinstance(momentum_data, MomentumAuditData):
            raise RiemannianInertiaAgentError(
                "momentum_data debe ser una instancia de MomentumAuditData."
            )

        momentum_norm = self._get_required_nonnegative_finite_float(
            momentum_data,
            "momentum_norm",
            "momentum_data",
        )

        is_bounded = self._get_required_bool(
            momentum_data,
            "is_bounded",
            "momentum_data",
        )

        # Campo vectorial opcional, pero si existe debe ser íntegro.
        self._validate_finite_array_attribute(
            momentum_data,
            "covariant_momentum",
            "momentum_data",
            required=False,
        )

        metric_condition_number = self._get_optional_nonnegative_finite_float(
            momentum_data,
            "metric_condition_number",
            1.0,
            "momentum_data",
        )

        # El número de condición no debe ser menor que 1 en una métrica válida.
        # Se tolera ruido de mantisa y se normaliza al valor mínimo teórico.
        if metric_condition_number < 1.0:
            metric_condition_number = 1.0

        inverse_consistency_residual = self._get_optional_nonnegative_finite_float(
            momentum_data,
            "inverse_consistency_residual",
            0.0,
            "momentum_data",
        )

        return (
            momentum_norm,
            is_bounded,
            metric_condition_number,
            inverse_consistency_residual,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Clasificadores locales de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_momentum_bound(
        self,
        momentum_norm: float,
        is_bounded: bool,
    ) -> InertialHeytingVerdict:
        """
        Clasifica la cota de volumen de Liouville asociada al momentum.

        - VETOED   : si el motor lo declara no acotado o si supera el límite duro.
        - DEGRADED : si supera el límite blando.
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
        """
        Clasifica la calidad numérica de la métrica según su número de condición.
        """
        if metric_condition_number > _CONDITION_HARD_MAX:
            return InertialHeytingVerdict.VETOED

        if metric_condition_number > _CONDITION_SOFT_MAX:
            return InertialHeytingVerdict.DEGRADED

        return InertialHeytingVerdict.COHERENT

    def _classify_inverse_consistency(
        self,
        inverse_consistency_residual: float,
    ) -> InertialHeytingVerdict:
        """
        Clasifica la consistencia de la inversa métrica G^{-1}.
        """
        if inverse_consistency_residual > _INVERSE_RESIDUAL_HARD_MAX:
            return InertialHeytingVerdict.VETOED

        if inverse_consistency_residual > _INVERSE_RESIDUAL_SOFT_MAX:
            return InertialHeytingVerdict.DEGRADED

        return InertialHeytingVerdict.COHERENT

    def _compute_momentum_margin(self, momentum_norm: float) -> float:
        """
        Margen normalizado de seguridad del momentum respecto al límite duro:

            $$margin = \max(0, 1 - \frac{\|p\|}{P_{\text{hard}}})$$
        """
        if _MOMENTUM_HARD_LIMIT <= 0.0:
            return 0.0

        return self._clamp01(1.0 - (momentum_norm / _MOMENTUM_HARD_LIMIT))

    # ──────────────────────────────────────────────────────────────────────────
    # Método nuclear de Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _audit_liouville_volume(
        self,
        momentum_data: MomentumAuditData,
    ) -> Phase1ObservationBridge:
        """
        Evalúa la preservación del volumen simpléctico a partir del certificado
        de momentum y de la calidad métrica.

        Clasifica el estado en el retículo Ω₃ y devuelve el puente formal
        hacia la Fase 2.
        """
        (
            momentum_norm,
            is_bounded,
            metric_condition_number,
            inverse_consistency_residual,
        ) = self._validate_momentum_data(momentum_data)

        momentum_bound_verdict = self._classify_momentum_bound(
            momentum_norm,
            is_bounded,
        )

        metric_condition_verdict = self._classify_metric_condition(
            metric_condition_number
        )

        inverse_consistency_verdict = self._classify_inverse_consistency(
            inverse_consistency_residual
        )

        liouville_verdict = self._heyting_join(
            (
                momentum_bound_verdict,
                metric_condition_verdict,
                inverse_consistency_verdict,
            )
        )

        momentum_margin = self._compute_momentum_margin(momentum_norm)

        logger.debug(
            "Fase 1 Liouville: ||p||=%.6e, bounded=%s, κ=%.6e, inv_res=%.6e, "
            "margin=%.6f, verdict=%s",
            momentum_norm,
            is_bounded,
            metric_condition_number,
            inverse_consistency_residual,
            momentum_margin,
            liouville_verdict.name,
        )

        return Phase1ObservationBridge(
            momentum_data=momentum_data,
            liouville_verdict=liouville_verdict,
            momentum_bound_verdict=momentum_bound_verdict,
            metric_condition_verdict=metric_condition_verdict,
            inverse_consistency_verdict=inverse_consistency_verdict,
            momentum_margin=momentum_margin,
        )

    def execute_phase1(
        self,
        momentum_data: MomentumAuditData,
    ) -> Phase1ObservationBridge:
        """
        Método terminal de la Fase 1.

        Su salida es la precondición formal de la Fase 2.
        """
        return self._audit_liouville_volume(momentum_data)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → CERTIFICACIÓN DE LA FIRMA MÉTRICA Y DE GAUGE (Orient)
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_SkewSymmetryCertifier(Phase1_LiouvilleVolumeAuditor):
    r"""
    Fase 2: Certifica que la proyección de Löwner en el Motor no inyecte
    trazas diagonales espurias ni componentes simétricas disipativas.

    Audita:
        1. antisimetría relativa de W;
        2. pureza de la vorticidad proyectada al cono de 2-formas;
        3. coherencia del puente de Fase 1.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Validación de puentes y artefactos de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_phase1_bridge(self, phase1_bridge: Phase1ObservationBridge) -> None:
        """Valida estructuralmente el certificado de Fase 1."""
        if not isinstance(phase1_bridge, Phase1ObservationBridge):
            raise RiemannianInertiaAgentError(
                "phase1_bridge debe ser una instancia de Phase1ObservationBridge."
            )

        self._validate_momentum_data(phase1_bridge.momentum_data)
        self._validate_verdict(
            phase1_bridge.liouville_verdict,
            "phase1_bridge.liouville_verdict",
        )

        # Veredictos extensionales opcionales.
        optional_verdict_fields = (
            "momentum_bound_verdict",
            "metric_condition_verdict",
            "inverse_consistency_verdict",
        )

        for field in optional_verdict_fields:
            if hasattr(phase1_bridge, field):
                self._validate_verdict(
                    getattr(phase1_bridge, field),
                    f"phase1_bridge.{field}",
                )

        if hasattr(phase1_bridge, "momentum_margin"):
            self._as_nonnegative_finite_float(
                phase1_bridge.momentum_margin,
                "phase1_bridge.momentum_margin",
            )

    def _validate_synthesis_data(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> tuple[float, bool, float, float]:
        """
        Verifica la integridad del artefacto giroscópico producido por el motor.

        Retorna:
            antisymmetry_residual,
            is_strictly_skew,
            gyroscopic_frobenius_norm,
            vorticity_projection_residual.
        """
        if not isinstance(synthesis_data, GyroscopicSynthesisData):
            raise RiemannianInertiaAgentError(
                "synthesis_data debe ser una instancia de GyroscopicSynthesisData."
            )

        antisymmetry_residual = self._get_required_nonnegative_finite_float(
            synthesis_data,
            "antisymmetry_residual",
            "synthesis_data",
        )

        is_strictly_skew = self._get_required_bool(
            synthesis_data,
            "is_strictly_skew",
            "synthesis_data",
        )

        # Campo tensorial opcional, pero si existe debe ser íntegro.
        self._validate_finite_array_attribute(
            synthesis_data,
            "skew_symmetric_tensor",
            "synthesis_data",
            required=False,
        )

        gyroscopic_frobenius_norm = self._get_optional_nonnegative_finite_float(
            synthesis_data,
            "gyroscopic_frobenius_norm",
            0.0,
            "synthesis_data",
        )

        vorticity_projection_residual = self._get_optional_nonnegative_finite_float(
            synthesis_data,
            "vorticity_projection_residual",
            0.0,
            "synthesis_data",
        )

        return (
            antisymmetry_residual,
            is_strictly_skew,
            gyroscopic_frobenius_norm,
            vorticity_projection_residual,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Clasificadores locales de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_antisymmetry(
        self,
        antisymmetry_residual: float,
        is_strictly_skew: bool,
        gyroscopic_frobenius_norm: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""
        Clasifica la antisimetría del operador giroscópico usando un residuo
        relativo:

            $$r_{\text{rel}} = \frac{\|W + W^T\|_F}{\max(1, \|W\|_F)}$$

        - VETOED   : si no es estrictamente antisimétrico o si r_rel > duro.
        - DEGRADED : si r_rel > blando.
        - COHERENT : caso contrario.
        """
        scale = max(1.0, gyroscopic_frobenius_norm)
        relative_residual = antisymmetry_residual / scale

        if not is_strictly_skew:
            return relative_residual, InertialHeytingVerdict.VETOED

        if relative_residual > _SKEW_HARD_RELATIVE_TOLERANCE:
            return relative_residual, InertialHeytingVerdict.VETOED

        if relative_residual > _SKEW_SOFT_RELATIVE_TOLERANCE:
            return relative_residual, InertialHeytingVerdict.DEGRADED

        return relative_residual, InertialHeytingVerdict.COHERENT

    def _classify_vorticity_projection(
        self,
        vorticity_projection_residual: float,
        gyroscopic_frobenius_norm: float,
    ) -> tuple[float, InertialHeytingVerdict]:
        r"""
        Clasifica la pureza de la vorticidad proyectada.

        Se usa el cociente:

            $$ratio = \frac{\|\text{sym}(\Omega)\|_F}{\max(1, \|W\|_F)}$$

        Si ||W|| es despreciable, la componente simétrica de Ω no produce
        efecto giroscópico relevante y se trata como COHERENT.
        """
        if gyroscopic_frobenius_norm <= _MACHINE_EPSILON:
            return 0.0, InertialHeytingVerdict.COHERENT

        scale = max(1.0, gyroscopic_frobenius_norm)
        ratio = vorticity_projection_residual / scale

        if ratio > _VORTICITY_PROJECTION_HARD_RATIO:
            return ratio, InertialHeytingVerdict.VETOED

        if ratio > _VORTICITY_PROJECTION_SOFT_RATIO:
            return ratio, InertialHeytingVerdict.DEGRADED

        return ratio, InertialHeytingVerdict.COHERENT

    # ──────────────────────────────────────────────────────────────────────────
    # Método nuclear de Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _certify_metric_signature(
        self,
        phase1_bridge: Phase1ObservationBridge,
        synthesis_data: GyroscopicSynthesisData,
    ) -> Phase2OrientationBridge:
        """
        Certifica la firma métrica antisimétrica del operador giroscópico.

        El veredicto local de Fase 2 es:

            $$v_{\text{Skew}} = v_{\text{antisymmetry}} \sqcup v_{\text{vorticity\_projection}}$$
        """
        self._validate_phase1_bridge(phase1_bridge)

        (
            antisymmetry_residual,
            is_strictly_skew,
            gyroscopic_frobenius_norm,
            vorticity_projection_residual,
        ) = self._validate_synthesis_data(synthesis_data)

        (
            relative_antisymmetry_residual,
            antisymmetry_verdict,
        ) = self._classify_antisymmetry(
            antisymmetry_residual,
            is_strictly_skew,
            gyroscopic_frobenius_norm,
        )

        (
            vorticity_projection_ratio,
            vorticity_projection_verdict,
        ) = self._classify_vorticity_projection(
            vorticity_projection_residual,
            gyroscopic_frobenius_norm,
        )

        skew_verdict = self._heyting_join(
            (
                antisymmetry_verdict,
                vorticity_projection_verdict,
            )
        )

        logger.debug(
            "Fase 2 Skew: residual_abs=%.6e, residual_rel=%.6e, strictly_skew=%s, "
            "||W||_F=%.6e, vorticity_ratio=%.6e, verdict=%s",
            antisymmetry_residual,
            relative_antisymmetry_residual,
            is_strictly_skew,
            gyroscopic_frobenius_norm,
            vorticity_projection_ratio,
            skew_verdict.name,
        )

        return Phase2OrientationBridge(
            phase1_bridge=phase1_bridge,
            synthesis_data=synthesis_data,
            skew_verdict=skew_verdict,
            antisymmetry_verdict=antisymmetry_verdict,
            vorticity_projection_verdict=vorticity_projection_verdict,
            relative_antisymmetry_residual=relative_antisymmetry_residual,
            vorticity_projection_ratio=vorticity_projection_ratio,
        )

    def execute_phase2(
        self,
        phase1_bridge: Phase1ObservationBridge,
        synthesis_data: GyroscopicSynthesisData,
    ) -> Phase2OrientationBridge:
        """
        Método terminal de la Fase 2.

        Recibe la salida formal de `execute_phase1` y produce la precondición
        formal de la Fase 3.
        """
        return self._certify_metric_signature(phase1_bridge, synthesis_data)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → COLAPSO TERMODINÁMICO EN EL RETÍCULO DE HEYTING (Decide & Act)
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_HeytingLatticeDecider(Phase2_SkewSymmetryCertifier):
    r"""
    Fase 3: Consolida el ciclo OODA.

    Realiza la operación supremo algebraico sobre los veredictos locales,
    imponiendo el estado determinista de la transacción estrictamente en
    memoria de software.

        $$v_{\text{final}} = v_{\text{Liouville}} \sqcup v_{\text{Skew}} \sqcup v_{\text{Work}}$$
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Validación de puentes y artefactos de Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_phase2_bridge(self, phase2_bridge: Phase2OrientationBridge) -> None:
        """Valida estructuralmente el certificado de Fase 2."""
        if not isinstance(phase2_bridge, Phase2OrientationBridge):
            raise RiemannianInertiaAgentError(
                "phase2_bridge debe ser una instancia de Phase2OrientationBridge."
            )

        self._validate_phase1_bridge(phase2_bridge.phase1_bridge)
        self._validate_synthesis_data(phase2_bridge.synthesis_data)
        self._validate_verdict(
            phase2_bridge.skew_verdict,
            "phase2_bridge.skew_verdict",
        )

        optional_verdict_fields = (
            "antisymmetry_verdict",
            "vorticity_projection_verdict",
        )

        for field in optional_verdict_fields:
            if hasattr(phase2_bridge, field):
                self._validate_verdict(
                    getattr(phase2_bridge, field),
                    f"phase2_bridge.{field}",
                )

        optional_float_fields = (
            "relative_antisymmetry_residual",
            "vorticity_projection_ratio",
        )

        for field in optional_float_fields:
            if hasattr(phase2_bridge, field):
                self._as_nonnegative_finite_float(
                    getattr(phase2_bridge, field),
                    f"phase2_bridge.{field}",
                )

    def _validate_veto_data(
        self,
        veto_data: ThermodynamicVetoData,
    ) -> tuple[float, bool, float, float, float]:
        """
        Verifica la integridad del artefacto termodinámico producido por el motor.

        Retorna:
            nilpotent_work_residual,
            is_symplectically_passive,
            work_tolerance,
            dirac_symmetric_residual,
            dirac_frobenius_norm.
        """
        if not isinstance(veto_data, ThermodynamicVetoData):
            raise RiemannianInertiaAgentError(
                "veto_data debe ser una instancia de ThermodynamicVetoData."
            )

        nilpotent_work_residual = self._get_required_nonnegative_finite_float(
            veto_data,
            "nilpotent_work_residual",
            "veto_data",
        )

        is_symplectically_passive = self._get_required_bool(
            veto_data,
            "is_symplectically_passive",
            "veto_data",
        )

        work_tolerance = self._get_optional_positive_finite_float(
            veto_data,
            "work_tolerance",
            _WORK_SOFT_ABSOLUTE_TOLERANCE,
            "veto_data",
        )

        dirac_symmetric_residual = self._get_optional_nonnegative_finite_float(
            veto_data,
            "dirac_symmetric_residual",
            0.0,
            "veto_data",
        )

        effective_dirac_matrix = self._validate_finite_array_attribute(
            veto_data,
            "effective_dirac_matrix",
            "veto_data",
            required=False,
        )

        if effective_dirac_matrix is None:
            dirac_frobenius_norm = 0.0
        else:
            dirac_frobenius_norm = self._matrix_frobenius_norm(
                effective_dirac_matrix,
                "veto_data.effective_dirac_matrix",
            )

        return (
            nilpotent_work_residual,
            is_symplectically_passive,
            work_tolerance,
            dirac_symmetric_residual,
            dirac_frobenius_norm,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Clasificadores locales de Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _classify_work_passivity(
        self,
        nilpotent_work_residual: float,
        is_symplectically_passive: bool,
        work_tolerance: float,
    ) -> InertialHeytingVerdict:
        r"""
        Clasifica la pasividad simpléctica del operador efectivo.

        - VETOED   : si el motor no certifica pasividad o si el residuo excede
                     la tolerancia certificada.
        - DEGRADED : si el residuo supera la política blanda absoluta.
        - COHERENT : caso contrario.
        """
        if not is_symplectically_passive:
            return InertialHeytingVerdict.VETOED

        if nilpotent_work_residual > work_tolerance:
            return InertialHeytingVerdict.VETOED

        if nilpotent_work_residual > _WORK_SOFT_ABSOLUTE_TOLERANCE:
            return InertialHeytingVerdict.DEGRADED

        return InertialHeytingVerdict.COHERENT

    def _classify_dirac_symmetry(
        self,
        dirac_symmetric_residual: float,
        dirac_frobenius_norm: float,
    ) -> InertialHeytingVerdict:
        r"""
        Clasifica la componente simétrica residual de J_eff.

        Se usa el residuo relativo:

            $$r_{\text{rel}} = \frac{\|J_{\text{eff}} + J_{\text{eff}}^T\|_F}{\max(1, \|J_{\text{eff}}\|_F)}$$
        """
        scale = max(1.0, dirac_frobenius_norm)
        relative_residual = dirac_symmetric_residual / scale

        if relative_residual > _DIRAC_SYMMETRIC_HARD_RELATIVE_TOLERANCE:
            return InertialHeytingVerdict.VETOED

        if relative_residual > _DIRAC_SYMMETRIC_SOFT_RELATIVE_TOLERANCE:
            return InertialHeytingVerdict.DEGRADED

        return InertialHeytingVerdict.COHERENT

    def _compute_work_margin(
        self,
        nilpotent_work_residual: float,
        work_tolerance: float,
    ) -> float:
        """
        Margen normalizado de trabajo nulo respecto a la tolerancia certificada:

            $$margin = \max(0, 1 - \frac{residual}{tolerance})$$
        """
        denominator = max(work_tolerance, _MACHINE_EPSILON)
        return self._clamp01(1.0 - (nilpotent_work_residual / denominator))

    # ──────────────────────────────────────────────────────────────────────────
    # Método nuclear de Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_thermodynamic_lattice(
        self,
        phase2_bridge: Phase2OrientationBridge,
        veto_data: ThermodynamicVetoData,
        raise_on_veto: bool = True,
    ) -> InertialGovernanceState:
        """
        Consolida los veredictos locales en un único supremo algebraico:

            $$v_{\text{final}} = \max(v_{\text{Liouville}}, v_{\text{Skew}}, v_{\text{Work}})$$

        Si v_final == VETOED y `raise_on_veto` es verdadero, colapsa el
        retículo lanzando HeytingLatticeVeto, aniquilando la transacción en RAM.
        """
        self._validate_phase2_bridge(phase2_bridge)

        (
            nilpotent_work_residual,
            is_symplectically_passive,
            work_tolerance,
            dirac_symmetric_residual,
            dirac_frobenius_norm,
        ) = self._validate_veto_data(veto_data)

        work_passivity_verdict = self._classify_work_passivity(
            nilpotent_work_residual,
            is_symplectically_passive,
            work_tolerance,
        )

        dirac_symmetry_verdict = self._classify_dirac_symmetry(
            dirac_symmetric_residual,
            dirac_frobenius_norm,
        )

        work_verdict = self._heyting_join(
            (
                work_passivity_verdict,
                dirac_symmetry_verdict,
            )
        )

        final_supremum_verdict = self._heyting_join(
            (
                phase2_bridge.phase1_bridge.liouville_verdict,
                phase2_bridge.skew_verdict,
                work_verdict,
            )
        )

        is_epistemologically_valid = (
            final_supremum_verdict != InertialHeytingVerdict.VETOED
        )

        work_margin = self._compute_work_margin(
            nilpotent_work_residual,
            work_tolerance,
        )

        if not is_epistemologically_valid and raise_on_veto:
            veto_sources = []

            if (
                phase2_bridge.phase1_bridge.liouville_verdict
                == InertialHeytingVerdict.VETOED
            ):
                veto_sources.append("Liouville")

            if phase2_bridge.skew_verdict == InertialHeytingVerdict.VETOED:
                veto_sources.append("Skew")

            if work_verdict == InertialHeytingVerdict.VETOED:
                veto_sources.append("Work")

            source_detail = ", ".join(veto_sources) if veto_sources else "Unknown"

            raise HeytingLatticeVeto(
                "Colapso de software: El operador giroscópico ha inyectado entropía "
                "no acotada. "
                f"Veredicto Supremo = {final_supremum_verdict.name}. "
                f"Fuente(s) de veto = [{source_detail}]. "
                f"||p|| = {phase2_bridge.phase1_bridge.momentum_data.momentum_norm:.6e}, "
                f"antisym_rel = {phase2_bridge.relative_antisymmetry_residual:.6e}, "
                f"work_residual = {nilpotent_work_residual:.6e}, "
                f"work_tolerance = {work_tolerance:.6e}. "
                "Transacción aniquilada en RAM."
            )

        logger.debug(
            "Fase 3 Lattice: work=%.6e, work_tol=%.6e, dirac_sym=%.6e, "
            "verdicts=(L=%s, S=%s, W=%s), final=%s, valid=%s, work_margin=%.6f",
            nilpotent_work_residual,
            work_tolerance,
            dirac_symmetric_residual,
            phase2_bridge.phase1_bridge.liouville_verdict.name,
            phase2_bridge.skew_verdict.name,
            work_verdict.name,
            final_supremum_verdict.name,
            is_epistemologically_valid,
            work_margin,
        )

        return InertialGovernanceState(
            phase2_bridge=phase2_bridge,
            veto_data=veto_data,
            work_verdict=work_verdict,
            final_supremum_verdict=final_supremum_verdict,
            is_epistemologically_valid=is_epistemologically_valid,
            work_passivity_verdict=work_passivity_verdict,
            dirac_symmetry_verdict=dirac_symmetry_verdict,
            work_margin=work_margin,
        )

    def execute_phase3(
        self,
        phase2_bridge: Phase2OrientationBridge,
        veto_data: ThermodynamicVetoData,
        raise_on_veto: bool = True,
    ) -> InertialGovernanceState:
        """
        Método terminal de la Fase 3.

        Recibe la salida formal de `execute_phase2`, retorna el estado lógico
        supremo y, opcionalmente, colapsa el retículo ante un veto.
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
    `RiemannianInertiaModulator` en un estricto ciclo OODA.

    Composición funtorial:

        $$F_{\text{Agent}} = F_{\text{Phase3}} \circ F_{\text{Phase2}} \circ F_{\text{Phase1}}$$

    Composición de Flujo:
        Motor Fase 1 → Agente Fase 1
        Motor Fase 2 → Agente Fase 2
        Motor Fase 3 → Agente Fase 3
    """

    def __init__(self, motor: RiemannianInertiaModulator):
        """
        Inyecta el motor físico que ejecuta las transformaciones simplécticas.
        """
        self._motor = motor
        self._validate_motor()

    def _validate_motor(self) -> None:
        """Verifica que el motor proporcionado implementa las fases ejecutables."""
        if self._motor is None:
            raise RiemannianInertiaAgentError("El motor físico no puede ser None.")

        required_methods = (
            "execute_phase1",
            "execute_phase2",
            "execute_phase3",
        )

        for method in required_methods:
            if not callable(getattr(self._motor, method, None)):
                raise RiemannianInertiaAgentError(
                    f"El motor debe exponer el método ejecutable '{method}'."
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
    ) -> InertialGovernanceState:
        r"""
        Ejecuta el ciclo categórico y topológico entrelazando las fases del
        Motor físico con las auditorías locales del Agente Soberano.

        Fase 1 → Auditoría del volumen de Liouville.
        Fase 2 → Certificación de la firma métrica.
        Fase 3 → Colapso termodinámico en el retículo de Heyting.
        """
        # --- OODA PASO 1: Observación física y auditoría lógica ---
        momentum_data = self._motor.execute_phase1(
            q_dot=q_dot,
            G_tensor=G_tensor,
            G_inv=G_inv,
        )
        phase1_bridge = self.execute_phase1(momentum_data)

        # --- OODA PASO 2: Orientación métrica y auditoría de antisimetría ---
        synthesis_data = self._motor.execute_phase2(
            momentum_data=momentum_data,
            vorticity_matrix=vorticity_matrix,
        )
        phase2_bridge = self.execute_phase2(phase1_bridge, synthesis_data)

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
            "Gobernanza inercial completada: final=%s, valid=%s",
            final_state.final_supremum_verdict.name,
            final_state.is_epistemologically_valid,
        )

        return final_state


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "InertialHeytingVerdict",
    "RiemannianInertiaAgentError",
    "LiouvilleVolumeCollapse",
    "HeytingLatticeVeto",
    "Phase1ObservationBridge",
    "Phase2OrientationBridge",
    "InertialGovernanceState",
    "Phase1_LiouvilleVolumeAuditor",
    "Phase2_SkewSymmetryCertifier",
    "Phase3_HeytingLatticeDecider",
    "RiemannianInertiaAgent",
]