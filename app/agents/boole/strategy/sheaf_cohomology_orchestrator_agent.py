# -*- coding: utf-8 -*-

r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sheaf Cohomology Orchestrator Agent (Custodio de la Holonomía)      ║
║ Ruta   : app/agents/boole/strategy/sheaf_cohomology_orchestrator_agent.py    ║
║ Versión: 2.0.0-Categorical-Krylov-Hodge-Strict                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna al `sheaf_cohomology_orchestrator.py` en el estrato
STRATEGY.

Su mandato axiomático es:

    1. Auditar el operador cofrontera δ.
    2. Prohibir el ensamblaje explícito del Laplaciano L = δᵀδ cuando el número
       de condición explota.
    3. Acotar termodinámicamente la proyección de Hodge-Helmholtz sobre ker(δ).

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación del Veto Cohomológico:
    Extrae el rango de δ vía SVD y computa la dimensión de las obstrucciones:

        dim H¹(G; F) = dim(C¹) - rank(δ) = 0.

    Bajo el supuesto operacional de este orquestador, δ es el operador
    cofrontera efectivo cuyo cokernel mide la primera cohomología del haz
    celular. Si se dispone del complejo completo δ⁰, δ¹, la dimensión exacta es:

        H¹ = ker(δ¹) / im(δ⁰).

    Último método de Fase 1:
        _certify_cohomological_veto_axiom(...)

    Dicho método retorna un certificado `CohomologicalVetoData`, el cual se
    convierte en el objeto inicial de la Fase 2.

Fase 2 → Regulación del Espectro de Krylov y Energía de Dirichlet:
    Audita:

        E(x) = ||δ x||₂² ≤ ε_frustration

    y veta el cómputo si:

        κ(L) = κ(δ)² > κ_max.

    Primer método de Fase 2:
        _audit_krylov_spectral_stability(..., veto_audit)

    Este método es la continuación formal de Fase 1: recibe el certificado
    cohomológico y lo propaga como invariante inicial del control espectral.

Fase 3 → Imposición del Límite Isoperimétrico de Hodge-Helmholtz:
    Aplica una cota Lipschitz sobre la sanación geométrica para preservar la
    masa inercial:

        ||x - x*||₂ ≤ Δ_inertia.

    Además verifica que la proyección no incremente la energía de Dirichlet.

    Primer método de Fase 3:
        _enforce_isoperimetric_hodge_projection(..., spectral_audit)

    Este método continúa formalmente la Fase 2: recibe el certificado espectral
    y verifica que la corrección geométrica sea termodinámicamente admisible.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, List, Optional

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Strategy.SheafCohomologyAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y LÍMITES TERMODINÁMICOS
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SVD_TOLERANCE: Final[float] = 1e-10
_MAX_CONDITION_NUMBER_L: Final[float] = 1e15
_FRUSTRATION_TOLERANCE: Final[float] = 1e-2
_INERTIA_DELTA_MAX: Final[float] = 5.0
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS
# ═══════════════════════════════════════════════════════════════════════════════
class SheafCohomologyAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Holonomía Global."""
    pass


class TopologicalBifurcationError(SheafCohomologyAgentError):
    r"""Detonada si dim H¹ > 0. Dependencias circulares mutantes insalvables."""
    pass


class SpectralComputationError(SheafCohomologyAgentError):
    r"""Detonada si κ(L) > κ_max. Peligro de colapso en la Unidad de Punto Flotante."""
    pass


class DirichletFrustrationError(SheafCohomologyAgentError):
    r"""Detonada si la energía de Dirichlet excede la frustración térmica admisible."""
    pass


class HomologicalInconsistencyError(SheafCohomologyAgentError):
    r"""Detonada si ||x - x*||₂ > Δ_inertia o si la proyección aumenta la energía."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class CohomologicalVetoData:
    r"""
    Artefacto de Fase 1.
    Certificado de anulación de obstrucciones globales.

    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    """
    dim_C0: int
    dim_C1: int
    delta_rank: int
    h1_dimension: int
    svd_tolerance: float
    max_singular_value: float
    min_nonzero_singular_value: float
    is_topologically_coherent: bool


@dataclass(frozen=True, slots=True)
class KrylovSpectralData:
    r"""
    Artefacto de Fase 2.
    Certificado termodinámico y condicionamiento espectral.

    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    """
    dirichlet_energy: float
    frustration_tolerance: float
    delta_condition_number: float
    laplacian_condition_number: float
    is_frustration_bounded: bool
    is_spectrally_stable: bool


@dataclass(frozen=True, slots=True)
class HodgeProjectionData:
    r"""
    Artefacto de Fase 3.
    Certificado de Lipschitz para el consenso de Hodge.
    """
    projection_distance: float
    relative_projection_distance: float
    inertia_delta_max: float
    original_dirichlet_energy: float
    projected_dirichlet_energy: float
    energy_reduction_ratio: float
    is_isoperimetrically_bounded: bool
    is_energy_non_increasing: bool
    verified_by_delta: bool


@dataclass(frozen=True, slots=True)
class SheafGovernanceState:
    r"""
    Objeto final del endofuntor Z_SheafAgent.
    """
    veto_audit: CohomologicalVetoData
    spectral_audit: KrylovSpectralData
    hodge_audit: HodgeProjectionData
    is_epistemologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico para evitar que singularidades aritméticas
    contaminen los invariantes topológicos y espectrales.
    """

    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Convierte un objeto a arreglo float64, rechazando:
            - Objetos complejos.
            - Valores NaN.
            - Valores infinitos.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} no puede interpretarse como arreglo numérico.") from exc

        if np.iscomplexobj(raw):
            raise TypeError(f"{name} debe ser real; se rechazó entrada compleja.")

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico real convertible a float64.") from exc

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores NaN o infinitos.")

        return arr

    @classmethod
    def _as_finite_matrix(cls, name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")

        return arr

    @classmethod
    def _as_finite_vector(cls, name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Valida un vector real finito.

        Acepta:
            - Vectores 1D.
            - Vectores columna (n, 1).
            - Vectores fila (1, n).
            - Escalares, interpretados como vector de dimensión 1.
            - Vectores vacíos, cuando el espacio subyacente tiene dimensión cero.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(f"{name} debe ser un vector 1D, fila, columna o escalar.")

        return arr

    @staticmethod
    def _vector_norm(v: NDArray[np.float64]) -> float:
        r"""
        Norma euclidiana numéricamente segura.
        """
        if v.size == 0:
            return 0.0

        value = float(la.norm(v, ord=2))
        return value if math.isfinite(value) else math.inf

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        r"""
        Norma de Frobenius numéricamente segura.
        """
        if A.size == 0:
            return 0.0

        value = float(la.norm(A, ord="fro"))
        return value if math.isfinite(value) else math.inf

    @staticmethod
    def _squared_norm_from_vector(y: NDArray[np.float64]) -> float:
        r"""
        Calcula ||y||₂² de forma segura.

        Si el producto interno arroja un negativo pequeño por ruido numérico,
        se proyecta al cono positivo. Si la negatividad es significativa, se
        denuncia la inconsistencia.
        """
        if y.size == 0:
            return 0.0

        value = float(np.dot(y, y))

        if not math.isfinite(value):
            raise DirichletFrustrationError(
                "La energía cuadrática ||δx||₂² no es finita; "
                "posible desbordamiento numérico."
            )

        tolerance = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(value))

        if value < -tolerance:
            raise DirichletFrustrationError(
                "Energía cuadrática negativa detectada; inconsistencia numérica grave."
            )

        return max(0.0, value)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN AXIOMÁTICA DEL VETO COHOMOLÓGICO                    ║
# ║                                                                             ║
# ║   Exige:                                                                    ║
# ║       dim H¹(G; F) = 0                                                      ║
# ║                                                                             ║
# ║   mediante extracción del rango efectivo de δ vía SVD.                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_CohomologicalVetoCertifier(_FiniteNumericalGuard):
    r"""
    Evalúa la matriz del operador cofrontera δ: C⁰ → C¹.

    Asegura que las dependencias inter-agente formen un consenso libre de
    vórtices cohomológicos.

    En el modelo operacional de este orquestador, se asume que la primera
    cohomología queda caracterizada por el cokernel de δ:

        dim H¹ = dim(C¹) - rank(δ).

    Si se dispone de un complejo cochain completo:

        C⁰ --δ⁰--> C¹ --δ¹--> C²,

    la definición exacta es:

        H¹ = ker(δ¹) / im(δ⁰).
    """

    def _singular_spectrum(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Calcula el espectro singular de una matriz de forma segura.

        Retorna un arreglo 1D con valores singulares finitos.
        Si la matriz tiene dimensión mínima cero, retorna un espectro vacío.
        """
        if A.size == 0 or min(A.shape) == 0:
            return np.empty(0, dtype=np.float64)

        try:
            singular_values = la.svdvals(A)
        except np.linalg.LinAlgError as exc:
            raise SpectralComputationError(
                "SVD no convergió al auditar el operador cofrontera δ."
            ) from exc

        singular_values = np.asarray(singular_values, dtype=np.float64)

        if not np.all(np.isfinite(singular_values)):
            raise SpectralComputationError(
                "El espectro singular de δ contiene valores NaN o infinitos."
            )

        return singular_values

    def _certify_cohomological_veto_axiom(
        self,
        coboundary_operator_delta: NDArray[np.float64],
    ) -> CohomologicalVetoData:
        r"""
        Último método de la Fase 1.

        Computa rank(δ) y define:

            dim H¹(G; F) = dim(C¹) - rank(δ).

        Si existe divergencia de cocadenas cerradas que no son exactas, se
        detona el veto topológico.

        Este método retorna un certificado `CohomologicalVetoData`, el cual
        constituye el objeto inicial de la Fase 2.
        """
        delta = self._as_finite_matrix(
            "coboundary_operator_delta",
            coboundary_operator_delta,
        )

        dim_C1, dim_C0 = delta.shape

        singular_values = self._singular_spectrum(delta)

        if singular_values.size > 0:
            sigma_max = float(singular_values[0])

            if sigma_max == 0.0:
                svd_tolerance = _SVD_TOLERANCE
            else:
                svd_tolerance = max(
                    _SVD_TOLERANCE,
                    _NUMERICAL_SAFETY_FACTOR
                    * _MACHINE_EPSILON
                    * max(delta.shape)
                    * sigma_max,
                )

            effective_rank = int(np.count_nonzero(singular_values > svd_tolerance))

            nonzero_singular_values = singular_values[singular_values > svd_tolerance]

            if nonzero_singular_values.size > 0:
                sigma_min_nonzero = float(nonzero_singular_values[-1])
            else:
                sigma_min_nonzero = 0.0
        else:
            sigma_max = 0.0
            sigma_min_nonzero = 0.0
            svd_tolerance = _SVD_TOLERANCE
            effective_rank = 0

        h1_dimension = int(dim_C1 - effective_rank)

        if h1_dimension < 0:
            # Defensivo: el rango SVD no debería exceder dim(C¹).
            logger.warning(
                "dim H¹ calculada fue negativa (%d); se proyecta a 0 por consistencia.",
                h1_dimension,
            )
            h1_dimension = 0

        if h1_dimension > 0:
            raise TopologicalBifurcationError(
                "Fractura homológica global. "
                f"El complejo posee dim H¹(G; F) = {h1_dimension} > 0. "
                "Existen dependencias circulares insalvables en la malla agéntica."
            )

        return CohomologicalVetoData(
            dim_C0=int(dim_C0),
            dim_C1=int(dim_C1),
            delta_rank=effective_rank,
            h1_dimension=h1_dimension,
            svd_tolerance=float(svd_tolerance),
            max_singular_value=float(sigma_max),
            min_nonzero_singular_value=float(sigma_min_nonzero),
            is_topologically_coherent=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: REGULACIÓN DEL ESPECTRO DE KRYLOV Y ENERGÍA DE DIRICHLET          ║
# ║                                                                             ║
# ║   Audita:                                                                   ║
# ║       E(x) = ||δx||₂² ≤ ε_frustration                                       ║
# ║                                                                             ║
# ║   y:                                                                        ║
# ║       κ(L) = κ(δ)² ≤ κ_max.                                                 ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 1.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_KrylovSpectralAuditor(Phase1_CohomologicalVetoCertifier):
    r"""
    Restringe la instanciación matricial explícita del Laplaciano L y mide la
    frustración térmica:

        E(x) = ||δx||₂²,

    generada por desalineaciones del estado agéntico.

    Esta fase hereda de Fase 1 y su primer método recibe explícitamente el
    certificado cohomológico emitido por:

        Phase1_CohomologicalVetoCertifier._certify_cohomological_veto_axiom(...)

    De este modo, la Fase 2 no es autónoma: está anidada funcionalmente en la
    Fase 1.
    """

    def _audit_krylov_spectral_stability(
        self,
        coboundary_operator_delta: NDArray[np.float64],
        x_state: NDArray[np.float64],
        veto_audit: Optional[CohomologicalVetoData] = None,
    ) -> KrylovSpectralData:
        r"""
        Primer método de la Fase 2.

        Continuación formal del último método de Fase 1.

        Mide la energía de disipación y el condicionamiento numérico sin
        ensamblar explícitamente el Laplaciano:

            L = δᵀδ.

        Si `veto_audit` es provisto:
            - Verifica que la Fase 1 haya certificado coherencia topológica.
            - Exige consistencia dimensional con δ.
            - Reutiliza el espectro singular certificado para calcular κ(δ).

        Si `veto_audit` no es provisto:
            - Ejecuta internamente la Fase 1 para obtener el certificado.

        Retorna:
            KrylovSpectralData, certificado que sirve como objeto inicial de
            la Fase 3.
        """
        delta = self._as_finite_matrix(
            "coboundary_operator_delta",
            coboundary_operator_delta,
        )

        x = self._as_finite_vector("x_state", x_state)

        if x.size != delta.shape[1]:
            raise ValueError(
                "Dimensión inconsistente: x_state pertenece a C⁰ y debe tener "
                f"dim={delta.shape[1]}, pero se recibió dim={x.size}."
            )

        # Continuación formal de Fase 1:
        # El certificado cohomológico restringe el dominio espectral.
        if veto_audit is None:
            veto_audit = self._certify_cohomological_veto_axiom(delta)
        else:
            if not veto_audit.is_topologically_coherent:
                raise TopologicalBifurcationError(
                    "La Fase 2 no puede iniciarse: la Fase 1 no certificó coherencia topológica."
                )

            if veto_audit.dim_C0 != delta.shape[1] or veto_audit.dim_C1 != delta.shape[0]:
                raise ValueError(
                    "El certificado cohomológico no coincide con la dimensión de δ. "
                    f"Certificado: C⁰={veto_audit.dim_C0}, C¹={veto_audit.dim_C1}; "
                    f"δ actual: C⁰={delta.shape[1]}, C¹={delta.shape[0]}."
                )

            if veto_audit.h1_dimension != 0:
                raise TopologicalBifurcationError(
                    "La Fase 2 no puede iniciarse: dim H¹ > 0 en el certificado de Fase 1."
                )

        # 1. Energía de Dirichlet (frustración térmica).
        delta_x = delta @ x

        if not np.all(np.isfinite(delta_x)):
            raise SpectralComputationError(
                "La aplicación del operador cofrontera δx produjo valores no finitos."
            )

        dirichlet_energy = self._squared_norm_from_vector(delta_x)

        frustration_tolerance = max(
            _FRUSTRATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(dirichlet_energy)),
        )

        if dirichlet_energy > frustration_tolerance:
            raise DirichletFrustrationError(
                "Frustración térmica inadmisible en la malla agéntica. "
                f"E(x) = ||δx||₂² = {dirichlet_energy:.6e} > "
                f"ε_frustration = {frustration_tolerance:.6e}."
            )

        # 2. Número de condición derivado κ(L) = κ(δ)².
        if veto_audit.dim_C1 == 0:
            # Sin restricciones 1-cocadenas: operador trivialmente estable.
            kappa_delta = 1.0
            kappa_L = 1.0
        else:
            sigma_max = veto_audit.max_singular_value
            sigma_min = veto_audit.min_nonzero_singular_value

            if sigma_max <= 0.0 or sigma_min <= 0.0:
                raise SpectralComputationError(
                    "El espectro certificado de δ es degenerado: "
                    f"σ_max={sigma_max:.6e}, σ_min={sigma_min:.6e}."
                )

            kappa_delta = sigma_max / sigma_min

            if not math.isfinite(kappa_delta):
                raise SpectralComputationError(
                    "El número de condición κ(δ) no es finito."
                )

            kappa_L = float(kappa_delta * kappa_delta)

            if not math.isfinite(kappa_L):
                raise SpectralComputationError(
                    "El número de condición κ(L) = κ(δ)² no es finito."
                )

        condition_tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, _MAX_CONDITION_NUMBER_L)
        )

        if kappa_L > _MAX_CONDITION_NUMBER_L + condition_tolerance:
            raise SpectralComputationError(
                "Peligro de colapso FPU (IEEE 754). "
                f"El condicionamiento espectral proyectado es κ(L) = {kappa_L:.6e} > "
                f"κ_max = {_MAX_CONDITION_NUMBER_L:.6e}. "
                "Se veta el cálculo de espectro explícito para preservar la estabilidad numérica."
            )

        return KrylovSpectralData(
            dirichlet_energy=float(dirichlet_energy),
            frustration_tolerance=float(frustration_tolerance),
            delta_condition_number=float(kappa_delta),
            laplacian_condition_number=float(kappa_L),
            is_frustration_bounded=True,
            is_spectrally_stable=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: IMPOSICIÓN DEL LÍMITE ISOPERIMÉTRICO DE HODGE-HELMHOLTZ           ║
# ║                                                                             ║
# ║   Garantiza:                                                                ║
# ║       ||x - x*||₂ ≤ Δ_inertia                                               ║
# ║                                                                             ║
# ║   y que la proyección no incremente la energía de Dirichlet.                ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 2.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_IsoperimetricHodgeProjector(Phase2_KrylovSpectralAuditor):
    r"""
    Regula la magnitud geométrica de la corrección del flujo.

    La Malla no puede alucinar una curación termodinámica que viole el
    presupuesto inercial físico.

    Esta fase hereda de Fase 2 y su primer método recibe explícitamente el
    certificado espectral emitido por:

        Phase2_KrylovSpectralAuditor._audit_krylov_spectral_stability(...)

    De este modo, la Fase 3 está anidada funcionalmente en la Fase 2.
    """

    def _enforce_isoperimetric_hodge_projection(
        self,
        x_original: NDArray[np.float64],
        x_projected: NDArray[np.float64],
        spectral_audit: Optional[KrylovSpectralData] = None,
        coboundary_operator_delta: Optional[NDArray[np.float64]] = None,
    ) -> HodgeProjectionData:
        r"""
        Primer método de la Fase 3.

        Continuación formal de Fase 2.

        Verifica la condición Lipschitz acotada termodinámicamente:

            ||x - x*||₂ ≤ Δ_inertia.

        Si se provee `coboundary_operator_delta`, además verifica:

            ||δ x*||₂² ≤ ε_frustration,

        y que la proyección no aumente la energía de Dirichlet:

            ||δ x*||₂² ≤ ||δ x||₂².

        Si se provee `spectral_audit`, se exige consistencia entre la energía
        original certificada en Fase 2 y la energía recalculada sobre x_original.
        """
        x0 = self._as_finite_vector("x_original", x_original)
        x1 = self._as_finite_vector("x_projected", x_projected)

        if x0.shape != x1.shape:
            raise ValueError(
                "x_original y x_projected deben tener la misma dimensión."
            )

        # Continuación formal de Fase 2:
        # El certificado espectral restringe el dominio geométrico.
        if spectral_audit is not None:
            if not spectral_audit.is_spectrally_stable:
                raise SpectralComputationError(
                    "La Fase 3 no puede iniciarse: la Fase 2 no certificó estabilidad espectral."
                )

            if not spectral_audit.is_frustration_bounded:
                raise DirichletFrustrationError(
                    "La Fase 3 no puede iniciarse: la Fase 2 no acotó la frustración térmica."
                )

        delta: Optional[NDArray[np.float64]] = None

        if coboundary_operator_delta is not None:
            delta = self._as_finite_matrix(
                "coboundary_operator_delta",
                coboundary_operator_delta,
            )

            if x0.size != delta.shape[1] or x1.size != delta.shape[1]:
                raise ValueError(
                    "Dimensión inconsistente entre x_original/x_projected y δ. "
                    f"δ espera dim={delta.shape[1]}, pero x_original dim={x0.size} "
                    f"y x_projected dim={x1.size}."
                )

        displacement_vector = x0 - x1

        if not np.all(np.isfinite(displacement_vector)):
            raise HomologicalInconsistencyError(
                "El vector de sanación topológica x - x* contiene valores no finitos."
            )

        projection_distance = self._vector_norm(displacement_vector)

        if not math.isfinite(projection_distance):
            raise HomologicalInconsistencyError(
                "La distancia de proyección ||x - x*||₂ no es finita."
            )

        norm_original = self._vector_norm(x0)
        norm_projected = self._vector_norm(x1)

        if not math.isfinite(norm_original) or not math.isfinite(norm_projected):
            raise HomologicalInconsistencyError(
                "Las normas de x_original o x_projected no son finitas."
            )

        relative_projection_distance = projection_distance / max(1.0, norm_original)

        distance_numerical_tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, norm_original, norm_projected)
        )

        inertia_limit = _INERTIA_DELTA_MAX + distance_numerical_tolerance

        if projection_distance > inertia_limit:
            raise HomologicalInconsistencyError(
                "Violación del principio de conservación financiera. "
                f"La proyección de Hodge exige un desplazamiento ||x - x*||₂ = "
                f"{projection_distance:.6f} > Δ_inertia = {_INERTIA_DELTA_MAX:.6f}. "
                "La sanación topológica requeriría recursos irreales ajenos a la física del proyecto."
            )

        verified_by_delta = delta is not None

        original_energy = 0.0
        projected_energy = 0.0
        energy_reduction_ratio = 0.0
        is_energy_non_increasing = True

        if verified_by_delta:
            assert delta is not None

            delta_x0 = delta @ x0
            delta_x1 = delta @ x1

            if not np.all(np.isfinite(delta_x0)) or not np.all(np.isfinite(delta_x1)):
                raise HomologicalInconsistencyError(
                    "La evaluación de δ sobre x_original o x_projected produjo valores no finitos."
                )

            original_energy = self._squared_norm_from_vector(delta_x0)
            projected_energy = self._squared_norm_from_vector(delta_x1)

            energy_numerical_tolerance = (
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(1.0, abs(original_energy), abs(projected_energy))
            )

            # Consistencia con el certificado de Fase 2, si existe.
            if spectral_audit is not None:
                consistency_tolerance = max(
                    1e-12,
                    _NUMERICAL_SAFETY_FACTOR
                    * _MACHINE_EPSILON
                    * max(
                        1.0,
                        abs(original_energy),
                        abs(spectral_audit.dirichlet_energy),
                    ),
                )

                if abs(original_energy - spectral_audit.dirichlet_energy) > consistency_tolerance:
                    raise HomologicalInconsistencyError(
                        "Inconsistencia energética entre Fase 2 y Fase 3. "
                        f"Energía certificada en Fase 2 = {spectral_audit.dirichlet_energy:.6e}, "
                        f"energía recalculada en Fase 3 = {original_energy:.6e}."
                    )

            # La proyección de Hodge no debe incrementar la energía de Dirichlet.
            is_energy_non_increasing = projected_energy <= (
                original_energy + energy_numerical_tolerance
            )

            if not is_energy_non_increasing:
                raise HomologicalInconsistencyError(
                    "La proyección de Hodge incrementó la energía de Dirichlet. "
                    f"E(x*) = {projected_energy:.6e} > E(x) = {original_energy:.6e}."
                )

            # La proyección debe permanecer dentro de la frustración admisible.
            frustration_limit = _FRUSTRATION_TOLERANCE

            if spectral_audit is not None:
                frustration_limit = max(
                    frustration_limit,
                    spectral_audit.frustration_tolerance,
                )

            projected_frustration_tolerance = max(
                frustration_limit,
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(1.0, abs(original_energy), abs(projected_energy)),
            )

            if projected_energy > projected_frustration_tolerance:
                raise HomologicalInconsistencyError(
                    "La proyección de Hodge no redujo la frustración a un nivel admisible. "
                    f"E(x*) = {projected_energy:.6e} > "
                    f"ε_frustration efectivo = {projected_frustration_tolerance:.6e}."
                )

            if original_energy > energy_numerical_tolerance:
                energy_reduction_ratio = projected_energy / original_energy
            else:
                energy_reduction_ratio = 0.0

            if not math.isfinite(energy_reduction_ratio):
                raise HomologicalInconsistencyError(
                    "La razón de reducción energética no es finita."
                )

        return HodgeProjectionData(
            projection_distance=float(projection_distance),
            relative_projection_distance=float(relative_projection_distance),
            inertia_delta_max=float(_INERTIA_DELTA_MAX),
            original_dirichlet_energy=float(original_energy),
            projected_dirichlet_energy=float(projected_energy),
            energy_reduction_ratio=float(energy_reduction_ratio),
            is_isoperimetrically_bounded=True,
            is_energy_non_increasing=bool(is_energy_non_increasing),
            verified_by_delta=bool(verified_by_delta),
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SHEAF COHOMOLOGY ORCHESTRATOR AGENT                  ║
# ║                                                                             ║
# ║   Endofuntor Z_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SheafCohomologyOrchestratorAgent(Morphism, Phase3_IsoperimetricHodgeProjector):
    r"""
    El Custodio de la Holonomía Global en el estrato STRATEGY.

    Impone la soberanía de la cohomología celular para discriminar entre ruido
    térmico subsanable vía Hodge y contradicciones epistemológicas globales:

        dim H¹ > 0.
    """

    def execute_sheaf_cohomology_governance(
        self,
        coboundary_operator_delta: NDArray[np.float64],
        x_state: NDArray[np.float64],
        x_projected_consensus: NDArray[np.float64],
    ) -> SheafGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁ : Certificación cohomológica.
            Φ₂ : Regulación espectral y energética de Krylov-Dirichlet.
            Φ₃ : Proyección isoperimétrica de Hodge-Helmholtz.

        Parámetros:
            coboundary_operator_delta:
                Operador cofrontera δ: C⁰ → C¹.

            x_state:
                Estado original x ∈ C⁰.

            x_projected_consensus:
                Estado proyectado x* ∈ C⁰, presuntamente en ker(δ).

        Retorna:
            SheafGovernanceState con los tres certificados y validez
            epistemológica final.
        """
        # Fase 1: Certificación axiomática del veto cohomológico.
        veto_audit = self._certify_cohomological_veto_axiom(
            coboundary_operator_delta
        )

        # Fase 2: Regulación del espectro de Krylov y energía de Dirichlet.
        spectral_audit = self._audit_krylov_spectral_stability(
            coboundary_operator_delta,
            x_state,
            veto_audit=veto_audit,
        )

        # Fase 3: Imposición del límite isoperimétrico de sanación.
        hodge_audit = self._enforce_isoperimetric_hodge_projection(
            x_state,
            x_projected_consensus,
            spectral_audit=spectral_audit,
            coboundary_operator_delta=coboundary_operator_delta,
        )

        is_epistemologically_valid = bool(
            veto_audit.is_topologically_coherent
            and spectral_audit.is_spectrally_stable
            and spectral_audit.is_frustration_bounded
            and hodge_audit.is_isoperimetrically_bounded
            and hodge_audit.is_energy_non_increasing
        )

        if not is_epistemologically_valid:
            raise SheafCohomologyAgentError(
                "La composición funtorial no autorizó la validez epistemológica del haz."
            )

        logger.info(
            "Gobernanza cohomológica del haz certificada. "
            "dim H¹: %d | κ(L): %.6e | E(x): %.6e | "
            "||x-x*||₂: %.6f | E(x*): %.6e | verified_by_δ: %s",
            veto_audit.h1_dimension,
            spectral_audit.laplacian_condition_number,
            spectral_audit.dirichlet_energy,
            hodge_audit.projection_distance,
            hodge_audit.projected_dirichlet_energy,
            hodge_audit.verified_by_delta,
        )

        return SheafGovernanceState(
            veto_audit=veto_audit,
            spectral_audit=spectral_audit,
            hodge_audit=hodge_audit,
            is_epistemologically_valid=is_epistemologically_valid,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "SheafCohomologyAgentError",
    "TopologicalBifurcationError",
    "SpectralComputationError",
    "DirichletFrustrationError",
    "HomologicalInconsistencyError",
    "CohomologicalVetoData",
    "KrylovSpectralData",
    "HodgeProjectionData",
    "SheafGovernanceState",
    "Phase1_CohomologicalVetoCertifier",
    "Phase2_KrylovSpectralAuditor",
    "Phase3_IsoperimetricHodgeProjector",
    "SheafCohomologyOrchestratorAgent",
]