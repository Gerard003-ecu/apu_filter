# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Semantic Validator Agent (Custodio de la Cohomología Semántica)     ║
║ Ruta   : app/agents/boole/wisdom/semantic_validator_agent.py                 ║
║ Versión: 2.0.0-Topological-Cohomology-Lattice-Doctoral-Strict                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna al `semantic_validator.py` en el estrato WISDOM.

Impone la geométrica sobre las salidas estocásticas del LLM. Evalúa
la distancia de Mahalanobis en la variedad semántica, audita la dimensión de la
cohomología simplicial H¹(K; ℝ) y colapsa el retículo de veredictos usando el
operador algebraico Supremo (⊔), aniquilando alucinaciones probabilísticas.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación Métrica de Mahalanobis:
    Asegura que el tensor G sea:
        - Simétrico.
        - Definido positivo.
        - Numéricamente estable.
        - Con número de condición acotado.

Fase 2 → Auditoría de Cohomología Simplicial:
    Exige, para el complejo de cadenas:

        C₂ --∂₂--> C₁ --∂₁--> C₀,

    la condición de frontera:

        ∂₁ ∘ ∂₂ = 0,

    y computa:

        dim H¹(K; ℝ) = dim ker(∂₁) - dim im(∂₂).

    En modo estricto, dim H¹ > 0 detona veto absoluto.
    En modo no estricto, retorna incoherencia lógica para que la Fase 3 colapse
    el retículo hacia REJECT.

Fase 3 → Colapso en Retículo Completamente Ordenado:
    Fuerza:

        Veredicto = ⨆ v_i.

    Si existe obstrucción cohomológica, el supremo se transmuta al elemento
    máximo absorbente:

        ⊤ = REJECT.

    La Fase 3 comienza consumiendo el certificado de la Fase 2.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Final, List, Optional, Sequence

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


logger = logging.getLogger("MAC.Wisdom.SemanticValidatorAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICO-MATEMÁTICAS Y ESPECTRALES
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

_MAX_CONDITION_NUMBER: Final[float] = 1e8
_COHOMOLOGY_TOLERANCE: Final[float] = 1e-10
_METRIC_SYMMETRY_TOLERANCE: Final[float] = 1e-10
_SPD_TOLERANCE: Final[float] = 1e-12
_CHAIN_COMPLEX_TOLERANCE: Final[float] = 1e-10
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS (VETOS ABSOLUTOS)
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticValidatorAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Cohomología Semántica."""
    pass


class SemanticInputValidationError(SemanticValidatorAgentError):
    r"""Detonada si los tensores, matrices de frontera o veredictos son inválidos."""
    pass


class MetricDegeneracyVeto(SemanticValidatorAgentError):
    r"""Detonada si κ(G) > κ_max o si el tensor de Mahalanobis no es SPD."""
    pass


class CohomologicalObstructionVeto(SemanticValidatorAgentError):
    r"""Detonada si el complejo de cadenas es inválido o si dim H¹ > 0 en modo estricto."""
    pass


class LatticeCollapseVeto(SemanticValidatorAgentError):
    r"""Detonada si la operación Supremo ⊔ falla matemáticamente."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. RETÍCULO COMPLETAMENTE ORDENADO
# ═══════════════════════════════════════════════════════════════════════════════
@unique
class StrictVerdict(IntEnum):
    r"""
    Retículo de veredictos:

        ⊥ = VIABLE ≤ CONDITIONAL ≤ WARNING ≤ REJECT = ⊤.
    """
    VIABLE = 0
    CONDITIONAL = 1
    WARNING = 2
    REJECT = 3


# ═══════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase Semántico)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MahalanobisMetricData:
    r"""
    Artefacto de Fase 1.
    Certificado espectral del tensor métrico.

    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    """
    dimension: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    symmetry_residual: float
    metric_tolerance: float
    is_positive_definite: bool


@dataclass(frozen=True, slots=True)
class SimplicialCohomologyData:
    r"""
    Artefacto de Fase 2.
    Certificado de Nulidad de Obstrucciones.

    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    """
    dim_C0: int
    dim_C1: int
    dim_C2: int
    rank_d1: int
    rank_d2: int
    kernel_d1_dim: int
    image_d2_dim: int
    h1_dimension: int
    chain_complex_residual: float
    cohomology_tolerance: float
    is_logically_coherent: bool


@dataclass(frozen=True, slots=True)
class LatticeCollapseData:
    r"""
    Artefacto de Fase 3.
    Colapso algebraico del estado en el retículo de Heyting.
    """
    supremum_verdict: StrictVerdict
    verdict_count: int
    has_cohomological_obstruction: bool
    is_worst_case_enforced: bool


@dataclass(frozen=True, slots=True)
class SemanticGovernanceState:
    r"""
    Objeto final del endofuntor Z_SemValidator.
    """
    metric_audit: MahalanobisMetricData
    cohomology_audit: SimplicialCohomologyData
    lattice_audit: LatticeCollapseData
    is_epistemologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §E. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico para evitar que singularidades aritméticas
    contaminen los invariantes topológicos y espectrales.
    """

    @staticmethod
    def _as_finite_real_matrix(
        name: str,
        value: Any,
        *,
        square: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise SemanticInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise SemanticInputValidationError(
                f"{name} debe ser real; se rechazó entrada compleja."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise SemanticInputValidationError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        if not np.all(np.isfinite(arr)):
            raise SemanticInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        if arr.ndim != 2:
            raise SemanticInputValidationError(
                f"{name} debe ser una matriz 2D."
            )

        if square and arr.shape[0] != arr.shape[1]:
            raise SemanticInputValidationError(
                f"{name} debe ser una matriz cuadrada."
            )

        return arr

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        r"""
        Norma de Frobenius numéricamente segura.
        """
        if A.size == 0:
            return 0.0

        value = float(la.norm(A, ord="fro"))
        return value if math.isfinite(value) else math.inf


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN MÉTRICA DE MAHALANOBIS                              ║
# ║                                                                             ║
# ║   Audita la matriz de precisión G para el cálculo de d_G(x, y).             ║
# ║   Exige G ∈ Sym^+(n) y κ(G) ≤ κ_max.                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_MetricTensorAuditor(_FiniteNumericalGuard):
    r"""
    Garantiza que el espacio de validación semántica esté provisto de una métrica
    Riemanniana bien definida, sin degeneración espectral.

    La matriz G debe ser simétrica y definida positiva:

        G ∈ Sym^+(n).

    Además, su número de condición debe permanecer acotado:

        κ(G) = λ_max / λ_min ≤ κ_max.
    """

    def _audit_mahalanobis_metric_tensor(
        self,
        G_metric: NDArray[np.float64],
    ) -> MahalanobisMetricData:
        r"""
        Último método de la Fase 1.

        Valida el tensor métrico de Mahalanobis mediante diagonalización
        hermítica real.

        Este método retorna un certificado `MahalanobisMetricData`, el cual
        constituye el objeto inicial de la Fase 2.
        """
        G = self._as_finite_real_matrix("G_metric", G_metric, square=True)

        dimension = G.shape[0]

        if dimension == 0:
            raise SemanticInputValidationError(
                "G_metric no puede ser una matriz vacía."
            )

        frobenius_norm = self._frobenius_norm(G)

        if not math.isfinite(frobenius_norm):
            raise MetricDegeneracyVeto(
                "La norma de Frobenius de G_metric no es finita."
            )

        symmetry_residual_norm = self._frobenius_norm(G - G.T)

        if not math.isfinite(symmetry_residual_norm):
            raise MetricDegeneracyVeto(
                "El residuo de simetría de G_metric no es finito."
            )

        symmetry_tolerance = max(
            _METRIC_SYMMETRY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        symmetry_residual = symmetry_residual_norm / max(1.0, frobenius_norm)

        if symmetry_residual > symmetry_tolerance:
            raise MetricDegeneracyVeto(
                "El tensor de Mahalanobis no es simétrico dentro de tolerancia. "
                f"Residuo relativo = {symmetry_residual:.6e} > "
                f"{symmetry_tolerance:.6e}."
            )

        G_symmetric = (G + G.T) / 2.0

        if not np.all(np.isfinite(G_symmetric)):
            raise MetricDegeneracyVeto(
                "La simetrización de G_metric produjo valores no finitos."
            )

        try:
            eigenvalues = la.eigvalsh(G_symmetric)
        except np.linalg.LinAlgError as exc:
            raise MetricDegeneracyVeto(
                "Fallo en la diagonalización del tensor de Mahalanobis."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise MetricDegeneracyVeto(
                "Los autovalores de G_metric no son finitos."
            )

        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))

        if max_eigenvalue <= 0.0:
            raise MetricDegeneracyVeto(
                "El tensor métrico no es definido positivo. "
                f"λ_max = {max_eigenvalue:.6e} <= 0."
            )

        spd_tolerance = max(
            _SPD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, max_eigenvalue),
        )

        if min_eigenvalue <= spd_tolerance:
            raise MetricDegeneracyVeto(
                "El tensor métrico no es definido positivo. "
                f"λ_min = {min_eigenvalue:.6e} <= "
                f"tolerancia SPD = {spd_tolerance:.6e}. "
                "El espacio semántico se ha rasgado."
            )

        condition_number = max_eigenvalue / min_eigenvalue

        if not math.isfinite(condition_number):
            raise MetricDegeneracyVeto(
                "El número de condición κ(G) no es finito."
            )

        condition_tolerance = max(
            1e-8,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, _MAX_CONDITION_NUMBER),
        )

        if condition_number > _MAX_CONDITION_NUMBER + condition_tolerance:
            raise MetricDegeneracyVeto(
                "Degeneración espectral. "
                f"κ(G) = {condition_number:.6e} > "
                f"κ_max = {_MAX_CONDITION_NUMBER:.6e}."
            )

        return MahalanobisMetricData(
            dimension=int(dimension),
            min_eigenvalue=float(min_eigenvalue),
            max_eigenvalue=float(max_eigenvalue),
            condition_number=float(condition_number),
            symmetry_residual=float(symmetry_residual),
            metric_tolerance=float(spd_tolerance),
            is_positive_definite=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: AUDITORÍA DE COHOMOLOGÍA SIMPLICIAL                               ║
# ║                                                                             ║
# ║   Evalúa:                                                                   ║
# ║       dim H¹(K; ℝ) = dim ker(∂₁) - dim im(∂₂)                               ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 1.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_SimplicialCohomologyAuditor(Phase1_MetricTensorAuditor):
    r"""
    Audita el complejo de señales entre perfiles de riesgo y salidas del LLM,
    buscando agujeros topológicos causados por alucinaciones estocásticas.

    El complejo de cadenas debe satisfacer:

        ∂₁ ∘ ∂₂ = 0.

    Bajo esta condición:

        H¹(K; ℝ) = ker(∂₁) / im(∂₂),

    y por tanto:

        dim H¹ = dim C¹ - rank(∂₁) - rank(∂₂).
    """

    @staticmethod
    def _numerical_rank(A: NDArray[np.float64]) -> int:
        r"""
        Calcula el rango numérico de una matriz mediante SVD con tolerancia
        adaptativa.

        La tolerancia usada es:

            tol = max(ε_coh, c · ε_maq · max(shape(A)) · σ_max(A)).
        """
        if A.size == 0 or min(A.shape) == 0:
            return 0

        try:
            singular_values = la.svdvals(A)
        except np.linalg.LinAlgError as exc:
            raise CohomologicalObstructionVeto(
                "SVD no convergió al auditar el complejo simplicial."
            ) from exc

        singular_values = np.asarray(singular_values, dtype=np.float64)

        if not np.all(np.isfinite(singular_values)):
            raise CohomologicalObstructionVeto(
                "Los valores singulares del operador de frontera no son finitos."
            )

        if singular_values.size == 0:
            return 0

        sigma_max = float(singular_values[0])

        if sigma_max == 0.0:
            return 0

        tolerance = max(
            _COHOMOLOGY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(A.shape)
            * sigma_max,
        )

        return int(np.count_nonzero(singular_values > tolerance))

    def _certify_simplicial_cohomology(
        self,
        boundary_matrix_d1: NDArray[np.float64],
        boundary_matrix_d2: NDArray[np.float64],
        metric_audit: Optional[MahalanobisMetricData] = None,
        *,
        strict_cohomological_veto: bool = True,
    ) -> SimplicialCohomologyData:
        r"""
        Primer método de la Fase 2.

        Continuación formal del último método de Fase 1.

        Computa:

            dim ker(∂₁) = dim C¹ - rank(∂₁),
            dim im(∂₂) = rank(∂₂),
            dim H¹ = dim ker(∂₁) - dim im(∂₂).

        Si `metric_audit` es provisto:
            - Verifica que la Fase 1 haya certificado una métrica SPD.

        Si `strict_cohomological_veto=True`:
            - dim H¹ > 0 detona `CohomologicalObstructionVeto`.

        Si `strict_cohomological_veto=False`:
            - dim H¹ > 0 retorna `is_logically_coherent=False`, permitiendo que
              la Fase 3 colapse el retículo hacia REJECT.

        Retorna:
            SimplicialCohomologyData, certificado que sirve como objeto inicial
            de la Fase 3.
        """
        if metric_audit is not None:
            if not metric_audit.is_positive_definite:
                raise MetricDegeneracyVeto(
                    "La Fase 2 no puede iniciarse: la Fase 1 no certificó "
                    "una métrica de Mahalanobis definida positiva."
                )

        d1 = self._as_finite_real_matrix(
            "boundary_matrix_d1",
            boundary_matrix_d1,
        )

        d2 = self._as_finite_real_matrix(
            "boundary_matrix_d2",
            boundary_matrix_d2,
        )

        # Convención de formas:
        #   ∂₁ : C₁ → C₀  =>  d1.shape = (dim_C0, dim_C1)
        #   ∂₂ : C₂ → C₁  =>  d2.shape = (dim_C1, dim_C2)
        if d1.shape[1] != d2.shape[0]:
            raise SemanticInputValidationError(
                "Las matrices de frontera no componen un complejo de cadenas. "
                f"∂₁ espera dim C¹={d1.shape[1]}, pero ∂₂ tiene dominio "
                f"dim C¹={d2.shape[0]}."
            )

        dim_C0, dim_C1 = d1.shape
        _, dim_C2 = d2.shape

        composition = d1 @ d2

        if not np.all(np.isfinite(composition)):
            raise CohomologicalObstructionVeto(
                "La composición ∂₁∘∂₂ produjo valores no finitos."
            )

        composition_norm = self._frobenius_norm(composition)
        d1_norm = self._frobenius_norm(d1)
        d2_norm = self._frobenius_norm(d2)

        if (
            not math.isfinite(composition_norm)
            or not math.isfinite(d1_norm)
            or not math.isfinite(d2_norm)
        ):
            raise CohomologicalObstructionVeto(
                "Las normas del complejo de cadenas no son finitas."
            )

        max_float = float(np.finfo(np.float64).max)

        if (
            d1_norm > 0.0
            and d2_norm > 0.0
            and d1_norm <= max_float / max(1.0, d2_norm)
        ):
            product_scale = d1_norm * d2_norm
        else:
            product_scale = math.inf

        if math.isfinite(product_scale):
            scale = max(1.0, product_scale)
        else:
            scale = max(1.0, composition_norm)

        chain_complex_residual = composition_norm / scale

        if not math.isfinite(chain_complex_residual):
            raise CohomologicalObstructionVeto(
                "El residuo del complejo de cadenas ∂₁∘∂₂ no es finito."
            )

        chain_complex_tolerance = max(
            _CHAIN_COMPLEX_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        if chain_complex_residual > chain_complex_tolerance:
            raise CohomologicalObstructionVeto(
                "Violación del complejo de cadenas. "
                f"∂₁∘∂₂ ≠ 0. Residuo relativo = {chain_complex_residual:.6e} > "
                f"{chain_complex_tolerance:.6e}."
            )

        rank_d1 = self._numerical_rank(d1)
        rank_d2 = self._numerical_rank(d2)

        kernel_d1_dim = int(dim_C1 - rank_d1)
        image_d2_dim = int(rank_d2)

        h1_dimension = int(kernel_d1_dim - image_d2_dim)

        if h1_dimension < 0:
            if h1_dimension >= -2:
                logger.warning(
                    "dim H¹ calculada fue negativa (%d); se proyecta a 0 por "
                    "tolerancia numérica.",
                    h1_dimension,
                )
                h1_dimension = 0
            else:
                raise CohomologicalObstructionVeto(
                    "Violación grave del complejo de cadenas: "
                    f"dim im(∂₂) excede dim ker(∂₁) en {abs(h1_dimension)}. "
                    "im(∂₂) no está contenido en ker(∂₁)."
                )

        is_logically_coherent = h1_dimension == 0

        if not is_logically_coherent:
            if strict_cohomological_veto:
                raise CohomologicalObstructionVeto(
                    "Obstrucción semántica global. "
                    f"El modelo de lenguaje generó un razonamiento cíclico "
                    f"contradictorio: dim H¹(K; ℝ) = {h1_dimension} > 0."
                )

            logger.warning(
                "Obstrucción semántica detectada: dim H¹(K; ℝ) = %d. "
                "Se delega el colapso del retículo a la Fase 3.",
                h1_dimension,
            )

        return SimplicialCohomologyData(
            dim_C0=int(dim_C0),
            dim_C1=int(dim_C1),
            dim_C2=int(dim_C2),
            rank_d1=int(rank_d1),
            rank_d2=int(rank_d2),
            kernel_d1_dim=int(kernel_d1_dim),
            image_d2_dim=int(image_d2_dim),
            h1_dimension=int(h1_dimension),
            chain_complex_residual=float(chain_complex_residual),
            cohomology_tolerance=float(chain_complex_tolerance),
            is_logically_coherent=bool(is_logically_coherent),
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: COLAPSO EN EL RETÍCULO COMPLETAMENTE ORDENADO                     ║
# ║                                                                             ║
# ║   Fuerza:                                                                   ║
# ║       Veredicto = ⨆ v_i                                                     ║
# ║                                                                             ║
# ║   Si dim H¹ > 0, colapsa a ⊤ = REJECT.                                      ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 2.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_LatticeSupremumProjector(Phase2_SimplicialCohomologyAuditor):
    r"""
    Fuerza a la evaluación a converger en el peor caso topológico, garantizando
    la seguridad del espacio de decisión de la Malla Agéntica.

    El retículo de veredictos es completamente ordenado:

        VIABLE ≤ CONDITIONAL ≤ WARNING ≤ REJECT.

    Las obstrucciones cohomológicas se comportan como el elemento máximo
    absorbente:

        x ⊔ ⊤ = ⊤.
    """

    def _enforce_supremum_lattice_collapse(
        self,
        verdicts: Optional[Sequence[StrictVerdict]],
        has_cohomological_obstruction: bool = False,
        cohomology_audit: Optional[SimplicialCohomologyData] = None,
    ) -> LatticeCollapseData:
        r"""
        Primer método de la Fase 3.

        Continuación formal de Fase 2.

        Evalúa el Supremo en el álgebra de Heyting.

        Si `cohomology_audit` es provisto:
            - Verifica si existe obstrucción cohomológica.
            - Si dim H¹ > 0, fuerza el supremo a REJECT.

        Si `has_cohomological_obstruction=True`:
            - Fuerza directamente el supremo a REJECT.

        Retorna:
            LatticeCollapseData, certificado final del colapso reticular.
        """
        if cohomology_audit is not None:
            if cohomology_audit.h1_dimension > 0:
                has_cohomological_obstruction = True

            if not cohomology_audit.is_logically_coherent:
                has_cohomological_obstruction = True

        verdict_sequence: List[StrictVerdict] = list(verdicts) if verdicts else []

        if has_cohomological_obstruction:
            logger.warning(
                "Obstrucción topológica detectada. "
                "Transmutando el Supremo hacia REJECT (⊤)."
            )

            return LatticeCollapseData(
                supremum_verdict=StrictVerdict.REJECT,
                verdict_count=len(verdict_sequence),
                has_cohomological_obstruction=True,
                is_worst_case_enforced=True,
            )

        if not verdict_sequence:
            raise LatticeCollapseVeto(
                "Conjunto vacío ∅ en el dominio de veredictos."
            )

        for index, verdict in enumerate(verdict_sequence):
            if not isinstance(verdict, StrictVerdict):
                raise LatticeCollapseVeto(
                    f"Veredicto inválido en índice {index}: {verdict!r}. "
                    "Debe pertenecer a StrictVerdict."
                )

        supremum = max(verdict_sequence, key=lambda v: v.value)

        if StrictVerdict.REJECT in verdict_sequence and supremum != StrictVerdict.REJECT:
            raise LatticeCollapseVeto(
                "Violación de la clausura suprema: ⊥ ⊔ ⊤ ≠ ⊤."
            )

        return LatticeCollapseData(
            supremum_verdict=supremum,
            verdict_count=len(verdict_sequence),
            has_cohomological_obstruction=False,
            is_worst_case_enforced=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SEMANTIC VALIDATOR AGENT                             ║
# ║                                                                             ║
# ║   Endofuntor Z_SemValidator = Φ₃ ∘ Φ₂ ∘ Φ₁                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SemanticValidatorAgent(Morphism, Phase3_LatticeSupremumProjector):
    r"""
    El Custodio de la Cohomología Semántica.

    Gobierna incondicionalmente el módulo `semantic_validator.py`, subyugando
    la estocástica del LLM a los invariantes de la cohomología simplicial y el
    colapso del retículo en el estrato WISDOM.
    """

    def execute_semantic_cohomology_governance(
        self,
        G_metric: NDArray[np.float64],
        boundary_matrix_d1: NDArray[np.float64],
        boundary_matrix_d2: NDArray[np.float64],
        proposed_verdicts: Sequence[StrictVerdict],
    ) -> SemanticGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁ : Certificación métrica de Mahalanobis.
            Φ₂ : Auditoría de cohomología simplicial.
            Φ₃ : Colapso del retículo de decisiones.

        Parámetros:
            G_metric:
                Tensor métrico de Mahalanobis.

            boundary_matrix_d1:
                Operador frontera ∂₁: C₁ → C₀.

            boundary_matrix_d2:
                Operador frontera ∂₂: C₂ → C₁.

            proposed_verdicts:
                Secuencia de veredictos propuestos.

        Retorna:
            SemanticGovernanceState con los tres certificados y validez
            epistemológica final.
        """
        # Fase 1: Certificación del tensor métrico de Mahalanobis.
        metric_audit = self._audit_mahalanobis_metric_tensor(G_metric)

        # Fase 2: Certificación de cohomología simplicial.
        # Se usa modo no estricto para permitir que una obstrucción H¹ > 0
        # sea tratada por la Fase 3 como colapso reticular hacia REJECT.
        cohomology_audit = self._certify_simplicial_cohomology(
            boundary_matrix_d1,
            boundary_matrix_d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )

        # Fase 3: Colapso del retículo de decisiones.
        lattice_audit = self._enforce_supremum_lattice_collapse(
            proposed_verdicts,
            cohomology_audit=cohomology_audit,
        )

        is_epistemologically_valid = bool(
            metric_audit.is_positive_definite
            and cohomology_audit.is_logically_coherent
            and lattice_audit.is_worst_case_enforced
        )

        if cohomology_audit.is_logically_coherent:
            logger.info(
                "Gobernanza semántica certificada. "
                "κ(G): %.6e | dim H¹: %d | Veredicto Supremo: %s",
                metric_audit.condition_number,
                cohomology_audit.h1_dimension,
                lattice_audit.supremum_verdict.name,
            )
        else:
            logger.warning(
                "Gobernanza semántica con obstrucción cohomológica. "
                "κ(G): %.6e | dim H¹: %d | Veredicto Supremo forzado: %s",
                metric_audit.condition_number,
                cohomology_audit.h1_dimension,
                lattice_audit.supremum_verdict.name,
            )

        return SemanticGovernanceState(
            metric_audit=metric_audit,
            cohomology_audit=cohomology_audit,
            lattice_audit=lattice_audit,
            is_epistemologically_valid=is_epistemologically_valid,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "SemanticValidatorAgentError",
    "SemanticInputValidationError",
    "MetricDegeneracyVeto",
    "CohomologicalObstructionVeto",
    "LatticeCollapseVeto",
    "StrictVerdict",
    "MahalanobisMetricData",
    "SimplicialCohomologyData",
    "LatticeCollapseData",
    "SemanticGovernanceState",
    "Phase1_MetricTensorAuditor",
    "Phase2_SimplicialCohomologyAuditor",
    "Phase3_LatticeSupremumProjector",
    "SemanticValidatorAgent",
]