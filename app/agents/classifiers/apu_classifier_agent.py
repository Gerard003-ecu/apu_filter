# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : APU Classifier Agent (Custodio de la Partición Ontológica)          ║
║ Ruta   : app/agents/classifiers/apu_classifier_agent.py                      ║
║ Versión: 5.0.0-Lebesgue-Affine-Homology-Strict                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TEORÍA DE LA MEDIDA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna a `apu_classifier.py`. Transmuta un motor de reglas
lógicas en un particionador topológico sobre el simplejo de probabilidad Δ².

Garantiza que la clasificación ontológica:

1. No posea espacios nulos:
       μ(Δ² \ ⋃ R_i) ≤ ε.

2. Preserve el difeomorfismo afín de escala:
       ||c - M_scale p||_∞ ≲ ε_affine,
   donde M_scale = 100 · I₃.

3. Respete la ortogonalidad de los centroides en las “Islas de Datos”:
       <C_isla, e_mo> = 0  ∧  <C_isla, e_eq> = 0.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Cobertura Espacial (Medida de Lebesgue).
Fase 2 → Certificación de Difeomorfismo Afín (Contrato de Escala).
Fase 3 → Topología de Centroides (Homología Estructural).

El último método de la Fase 1 es el handoff formal que inaugura la Fase 2.
El último método de la Fase 2 es el handoff formal que inaugura la Fase 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Tuple

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
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Classifiers.OntologyAgent")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, MEDIDAS Y TOLERANCIAS
# ═══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancia para vacíos de integración / medida de Lebesgue.
_LEBESGUE_MEASURE_TOLERANCE: Final[float] = 1e-7

# Tolerancia base para el contrato afín ||c - M p||_∞.
_AFFINE_ISOMORPHISM_TOLERANCE: Final[float] = 1e-12

# Tolerancia estricta del producto interno para centroides.
_ORTHOGONAL_CENTROID_TOLERANCE: Final[float] = 1e-14

# Tolerancia para la suma barycéntrica en Δ².
_SIMPLEX_SUM_TOLERANCE: Final[float] = 1e-12

# Dimensión del simplejo 3D: Suministro, Mano de Obra, Equipo.
_VECTOR_DIMENSION: Final[int] = 3

# Operador de escala: M_scale = 100 · I₃.
_SCALE_MATRIX: Final[NDArray[np.float64]] = 100.0 * np.eye(
    _VECTOR_DIMENSION,
    dtype=np.float64,
)

# Norma operacional inducida de M_scale (σ_max = 100).
_SCALE_OPERATOR_NORM: Final[float] = 100.0

# Bases canónicas del espacio de costos.
_E_SUM: Final[NDArray[np.float64]] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_E_MO: Final[NDArray[np.float64]] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_E_EQ: Final[NDArray[np.float64]] = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ONTOLÓGICAS
# ═══════════════════════════════════════════════════════════════════════════════

class APUClassifierAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Partición Ontológica."""
    pass


class DomainIntegrityViolationError(APUClassifierAgentError):
    """Detonada cuando un vector, escalar o bandera viola su contrato de dominio."""
    pass


class SimplexMembershipViolationError(APUClassifierAgentError):
    r"""Detonada cuando p ∉ Δ² o su suma barycéntrica excede la tolerancia."""
    pass


class LebesgueMeasureViolationError(APUClassifierAgentError):
    r"""Detonada si μ(Uncovered) > ε. Existen vacíos ontológicos en Δ²."""
    pass


class ScaleInvarianceCollapseError(APUClassifierAgentError):
    r"""Detonada si se pierde el isomorfismo afín entre proporciones y porcentajes."""
    pass


class TopologicalCentroidAnomalyVeto(APUClassifierAgentError):
    r"""Detonada si un centroide tipo “Isla” posee componentes no ortogonales."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio Ontológico)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class VectorDomainCertificate:
    r"""Certificado de integridad material de un vector en R³."""
    name: str
    dimension: int
    l1_norm: float
    l2_norm: float
    linf_norm: float
    is_finite: bool


@dataclass(frozen=True, slots=True)
class LebesgueAuditData:
    r"""Artefacto de Fase 1. Certificado de partición exhaustiva en Δ²."""
    uncovered_measure: float
    measure_tolerance: float
    is_partition_exhaustive: bool


@dataclass(frozen=True, slots=True)
class ScaleIsomorphismData:
    r"""Artefacto de Fase 2. Certificado del difeomorfismo afín normado."""
    residual_infinity_norm: float
    condition_number: float
    spectral_deviation: float
    affine_tolerance: float
    is_scale_isomorphic: bool


@dataclass(frozen=True, slots=True)
class CentroidTopologyData:
    r"""Artefacto de Fase 3. Certificado de ortogonalidad del producto interno."""
    inner_product_mo: float
    inner_product_eq: float
    projection_norm: float
    basis_orthogonality_deviation: float
    topology_tolerance: float
    is_structurally_orthogonal: bool


@dataclass(frozen=True, slots=True)
class Phase1LebesgueHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.

    Este objeto es la continuación material de la auditoría de Lebesgue y el
    prefijo de entrada obligatorio del certificador afín.
    """
    lebesgue_audit: LebesgueAuditData
    p_certified: NDArray[np.float64]
    c_certified: NDArray[np.float64]
    p_domain: VectorDomainCertificate
    c_domain: VectorDomainCertificate


@dataclass(frozen=True, slots=True)
class Phase2ScaleHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.

    Este objeto transporta el certificado afín y el centroide ya saneado para
    la verificación de homología estructural.
    """
    phase1_handoff: Phase1LebesgueHandoff
    scale_audit: ScaleIsomorphismData
    centroid_certified: NDArray[np.float64]
    centroid_domain: VectorDomainCertificate
    is_isolated_island: bool


@dataclass(frozen=True, slots=True)
class OntologicalPartitionState:
    r"""Objeto final del endofuntor Z_Classifier."""
    lebesgue_audit: LebesgueAuditData
    scale_audit: ScaleIsomorphismData
    centroid_audit: CentroidTopologyData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DE COBERTURA ESPACIAL Y MEDIDA DE LEBESGUE              ║
# ║   Evalúa μ(Δ² \ ⋃ R_i) ≤ ε y sanea el dominio vectorial inicial.           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_LebesgueMeasureAuditor:
    r"""
    Garantiza axiomáticamente que la unión de las regiones de las reglas de
    clasificación constituya un recubrimiento completo del espacio de estados.

    Además, sanea los vectores p y c antes de entregarlos a la Fase 2.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Tolerancia adaptativa
    # ─────────────────────────────────────────────────────────────────────────
    def _adaptive_tolerance(
        self,
        base_tolerance: float,
        reference: Any,
    ) -> float:
        r"""
        Construye una tolerancia numéricamente consciente:

            tol = max(tol_base, κ · ε_máquina · dim · escala)

        donde la escala se estima por norma L∞ del objeto de referencia.
        """
        if isinstance(reference, np.ndarray):
            if reference.size == 0:
                scale = 1.0
            else:
                scale = max(
                    1.0,
                    float(la.norm(reference.ravel(), ord=np.inf)),
                )
        else:
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        return max(
            float(base_tolerance),
            10.0 * _MACHINE_EPSILON * _VECTOR_DIMENSION * scale,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción y finitud vectorial
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_vector(
        self,
        name: str,
        vector: Any,
    ) -> NDArray[np.float64]:
        r"""
        Materializa un vector en R³ con dtype float64 y verifica finitud
        absoluta: prohibido NaN, +∞ o -∞.
        """
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"Vector ontológico '{name}' no puede materializarse como "
                f"NDArray[np.float64]."
            ) from exc

        if arr.size != _VECTOR_DIMENSION:
            raise DomainIntegrityViolationError(
                f"Vector ontológico '{name}' debe pertenecer a R^3. "
                f"Se recibieron {arr.size} componentes."
            )

        arr = arr.reshape(_VECTOR_DIMENSION)

        if not np.all(np.isfinite(arr)):
            raise DomainIntegrityViolationError(
                f"Vector ontológico '{name}' contiene componentes no finitas. "
                f"NaN o infinitos rompen la completitud del espacio de estados."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Certificado de dominio vectorial
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_vector_domain(
        self,
        name: str,
        vector: Any,
    ) -> Tuple[NDArray[np.float64], VectorDomainCertificate]:
        r"""
        Extiende la coerción vectorial emitiendo un certificado normativo
        L¹, L² y L∞.
        """
        arr = self._coerce_finite_vector(name, vector)

        l1_norm = float(la.norm(arr, ord=1))
        l2_norm = float(la.norm(arr, ord=2))
        linf_norm = float(la.norm(arr, ord=np.inf))

        certificate = VectorDomainCertificate(
            name=name,
            dimension=int(arr.size),
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            linf_norm=linf_norm,
            is_finite=True,
        )

        return arr, certificate

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. No negatividad y acotamiento superior
    # ─────────────────────────────────────────────────────────────────────────
    def _assert_nonnegative_and_bounded(
        self,
        name: str,
        arr: NDArray[np.float64],
        upper_bound: Any = None,
    ) -> None:
        r"""
        Verifica que un vector sea numéricamente no negativo y, si se indica,
        que no exceda una cota superior física.
        """
        tol = self._adaptive_tolerance(_AFFINE_ISOMORPHISM_TOLERANCE, arr)

        if np.any(arr < -tol):
            raise DomainIntegrityViolationError(
                f"Vector ontológico '{name}' posee componentes negativas "
                f"incompatibles con su semántica de costo."
            )

        if upper_bound is not None:
            bound = float(upper_bound)
            if np.any(arr > bound + tol):
                raise DomainIntegrityViolationError(
                    f"Vector ontológico '{name}' excede la cota superior "
                    f"física {bound:.6f}."
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Coerción de bandera ontológica
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_ontological_flag(
        self,
        name: str,
        value: Any,
    ) -> bool:
        r"""
        Exige tipado estricto para banderas ontológicas. Solo se admiten
        bool o np.bool_.
        """
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        raise DomainIntegrityViolationError(
            f"Bandera ontológica '{name}' debe ser bool o np.bool_. "
            f"Se recibió {type(value)!r}."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Validación de medida de Lebesgue
    # ─────────────────────────────────────────────────────────────────────────
    def _validate_measure_ratio(
        self,
        uncovered_area_ratio: Any,
    ) -> float:
        r"""
        Valida que la razón de área descubierta sea un escalar finito en
        [0, 1], dentro de tolerancia numérica.
        """
        try:
            value = float(uncovered_area_ratio)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                "La medida de Lebesgue descubierta debe ser un escalar "
                "numérico finito."
            ) from exc

        if not np.isfinite(value):
            raise DomainIntegrityViolationError(
                "La medida de Lebesgue descubierta no es finita."
            )

        tol = self._adaptive_tolerance(_LEBESGUE_MEASURE_TOLERANCE, value)

        if value < -tol:
            raise DomainIntegrityViolationError(
                f"Medida de Lebesgue negativa: {value:.6e}. "
                f"Una medida exterior no puede ser negativa."
            )

        if value > 1.0 + tol:
            raise DomainIntegrityViolationError(
                f"Medida de Lebesgue normalizada excede 1: {value:.6e}."
            )

        # Saneamiento de ruido numérico en fronteras.
        if value < 0.0:
            value = 0.0
        if value > 1.0:
            value = 1.0

        return value

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. Certificación de pertenencia al simplejo Δ²
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_probability_simplex(
        self,
        p_vector: Any,
    ) -> NDArray[np.float64]:
        r"""
        Certifica que p ∈ Δ²:

            p_i ≥ 0,   Σ p_i = 1.

        Se admite renormalización únicamente si la desviación es infinitesimal.
        """
        p = self._coerce_finite_vector("p_vector", p_vector)
        tol = self._adaptive_tolerance(_SIMPLEX_SUM_TOLERANCE, p)

        if np.any(p < -tol):
            raise SimplexMembershipViolationError(
                "Vector de proporciones p posee componentes negativas "
                "incompatibles con Δ²."
            )

        # Eliminación de ruido negativo infinitesimal.
        p_clean = np.where(p < 0.0, 0.0, p)
        total = float(np.sum(p_clean))

        if not np.isfinite(total) or total <= tol:
            raise SimplexMembershipViolationError(
                "Vector de proporciones p tiene masa total nula o no finita. "
                "No existe medida barycéntrica válida."
            )

        if abs(total - 1.0) > tol:
            raise SimplexMembershipViolationError(
                f"Vector de proporciones p no suma 1 dentro de tolerancia. "
                f"Suma observada: {total:.12e}, tolerancia: {tol:.6e}."
            )

        p_certified = p_clean / total

        # Verificación final de la renormalización.
        certified_sum = float(np.sum(p_certified))
        if abs(certified_sum - 1.0) > tol:
            raise SimplexMembershipViolationError(
                f"Renormalización de p falló. Suma final: "
                f"{certified_sum:.12e}, tolerancia: {tol:.6e}."
            )

        return p_certified

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8. Auditoría de cobertura de Lebesgue
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_lebesgue_measure_coverage(
        self,
        uncovered_area_ratio: float,
    ) -> LebesgueAuditData:
        r"""
        Verifica la integral de la función indicatriz del complemento del
        recubrimiento:

            μ(Δ² \ ⋃ R_i) ≤ ε.
        """
        measure = self._validate_measure_ratio(uncovered_area_ratio)

        if measure > _LEBESGUE_MEASURE_TOLERANCE:
            raise LebesgueMeasureViolationError(
                f"Fractura Ontológica: La Medida de Lebesgue del conjunto "
                f"descubierto es μ = {measure:.6e} > "
                f"{_LEBESGUE_MEASURE_TOLERANCE:.6e}. "
                f"Existen vectores de costo que colapsarán en un vacío "
                f"determinista."
            )

        return LebesgueAuditData(
            uncovered_measure=measure,
            measure_tolerance=_LEBESGUE_MEASURE_TOLERANCE,
            is_partition_exhaustive=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.9. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_audit_and_handoff_to_phase2(
        self,
        uncovered_area_ratio: float,
        p_vector: NDArray[np.float64],
        c_vector: NDArray[np.float64],
    ) -> Phase1LebesgueHandoff:
        r"""
        Último método de la Fase 1.

        Su definición formal es la continuación directa de la Fase 2:
        entrega p_certified, c_certified y el certificado de Lebesgue como
        prefijo obligatorio del contrato afín.
        """
        # 1. Auditoría de medida.
        lebesgue_audit = self._audit_lebesgue_measure_coverage(
            uncovered_area_ratio
        )

        # 2. Certificación de p ∈ Δ².
        p_certified = self._certify_probability_simplex(p_vector)
        _, p_domain = self._certify_vector_domain(
            "p_vector_certified",
            p_certified,
        )

        # 3. Certificación material de c.
        c_certified, c_domain = self._certify_vector_domain(
            "c_vector",
            c_vector,
        )

        # 4. c debe ser un vector de porcentajes no negativo y acotado por 100.
        self._assert_nonnegative_and_bounded(
            "c_vector",
            c_certified,
            upper_bound=100.0,
        )

        logger.debug(
            "Fase 1 completada. μ_descubierto=%.6e | ||p||_∞=%.6e | "
            "||c||_∞=%.6e.",
            lebesgue_audit.uncovered_measure,
            p_domain.linf_norm,
            c_domain.linf_norm,
        )

        return Phase1LebesgueHandoff(
            lebesgue_audit=lebesgue_audit,
            p_certified=p_certified,
            c_certified=c_certified,
            p_domain=p_domain,
            c_domain=c_domain,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE DIFEOMORFISMO AFÍN Y CONTRATO DE ESCALA          ║
# ║   Exige ||c - M_scale p||_∞ ≲ ε_affine.                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_ScaleIsomorphismCertifier(Phase1_LebesgueMeasureAuditor):
    r"""
    Audita que el paso del subespacio de proporciones p al hiperespacio de
    porcentajes c opere como un proyector isométrico perfecto:

        c = M_scale p,   M_scale = 100 · I₃.

    Esta fase hereda la auditoría de Lebesgue y el saneamiento del dominio.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Tolerancia del contrato afín
    # ─────────────────────────────────────────────────────────────────────────
    def _scale_contract_tolerance(
        self,
        p_vector: NDArray[np.float64],
        c_vector: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la tolerancia efectiva del contrato afín.

        Si p se admite con tolerancia δ_simplex, entonces M_scale p puede
        inducir un error de hasta ||M_scale|| · δ_simplex. Por tanto:

            ε_affine_eff = max(ε_affine_base, ||M_scale|| · δ_simplex).
        """
        simplex_tol = self._adaptive_tolerance(_SIMPLEX_SUM_TOLERANCE, p_vector)

        base = max(
            _AFFINE_ISOMORPHISM_TOLERANCE,
            _SCALE_OPERATOR_NORM * simplex_tol,
        )

        return self._adaptive_tolerance(base, c_vector)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Certificado espectral del operador de escala
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_scale_operator_spectrum(self) -> Tuple[float, float]:
        r"""
        Certifica que el operador M_scale sea espectralmente sano:

            σ_i(M_scale) = 100,   κ₂(M_scale) = 1.

        Esto evita anisotropías, singularidades y desbordamientos funcionales.
        """
        singular_values = la.svdvals(_SCALE_MATRIX)

        if singular_values.size != _VECTOR_DIMENSION:
            raise ScaleInvarianceCollapseError(
                "El operador de escala no posee espectro completo en R³."
            )

        s_max = float(np.max(singular_values))
        s_min = float(np.min(singular_values))

        if s_min <= _MACHINE_EPSILON:
            raise ScaleInvarianceCollapseError(
                "El operador de escala es numéricamente singular."
            )

        condition_number = s_max / s_min
        spectral_deviation = float(np.max(np.abs(singular_values - 100.0)))

        spectral_tol = self._adaptive_tolerance(
            max(
                _AFFINE_ISOMORPHISM_TOLERANCE,
                _SCALE_OPERATOR_NORM * _SIMPLEX_SUM_TOLERANCE,
            ),
            _SCALE_MATRIX,
        )

        if condition_number - 1.0 > spectral_tol:
            raise ScaleInvarianceCollapseError(
                f"Operador de escala con número de condición inaceptable: "
                f"kappa={condition_number:.12e}, tol={spectral_tol:.6e}."
            )

        if spectral_deviation > spectral_tol:
            raise ScaleInvarianceCollapseError(
                f"Operador de escala con desviación espectral inaceptable: "
                f"max|σ_i - 100|={spectral_deviation:.6e}, "
                f"tol={spectral_tol:.6e}."
            )

        return condition_number, spectral_deviation

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Certificación afín desde vectores ya certificados
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_scale_isomorphism_from_certified_vectors(
        self,
        p_certified: NDArray[np.float64],
        c_certified: NDArray[np.float64],
    ) -> ScaleIsomorphismData:
        r"""
        Verifica el contrato afín sobre vectores previamente saneados:

            ||c - M_scale p||_∞ ≤ ε_affine_eff.
        """
        p = self._certify_probability_simplex(p_certified)
        c = self._coerce_finite_vector("c_certified", c_certified)

        self._assert_nonnegative_and_bounded(
            "c_certified",
            c,
            upper_bound=100.0,
        )

        condition_number, spectral_deviation = (
            self._certify_scale_operator_spectrum()
        )

        c_expected = _SCALE_MATRIX @ p
        residual = c - c_expected
        residual_norm = float(la.norm(residual, ord=np.inf))

        affine_tol = self._scale_contract_tolerance(p, c)

        if residual_norm > affine_tol:
            raise ScaleInvarianceCollapseError(
                f"Pérdida de Biyectividad Escalar. La transformación indujo "
                f"un residuo en norma L∞ = {residual_norm:.6e} > "
                f"{affine_tol:.6e}. "
                f"Peligro de activación de reglas de negocio erróneas por "
                f"desbordamiento FPU."
            )

        return ScaleIsomorphismData(
            residual_infinity_norm=residual_norm,
            condition_number=condition_number,
            spectral_deviation=spectral_deviation,
            affine_tolerance=affine_tol,
            is_scale_isomorphic=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Wrapper público/retrocompatible de certificación afín
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_scale_isomorphism(
        self,
        p_vector: NDArray[np.float64],
        c_vector: NDArray[np.float64],
    ) -> ScaleIsomorphismData:
        r"""
        Certifica el difeomorfismo afín desde vectores crudos.

        Este método conserva compatibilidad con la signatura original de
        Fase 2.
        """
        p = self._certify_probability_simplex(p_vector)
        c = self._coerce_finite_vector("c_vector", c_vector)

        self._assert_nonnegative_and_bounded(
            "c_vector",
            c,
            upper_bound=100.0,
        )

        return self._certify_scale_isomorphism_from_certified_vectors(p, c)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_certify_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1LebesgueHandoff,
        centroid_C: NDArray[np.float64],
        is_isolated_island: bool,
    ) -> Phase2ScaleHandoff:
        r"""
        Último método de la Fase 2.

        Su definición formal es la continuación directa de la Fase 3:
        entrega el certificado afín y el centroide saneado como prefijo
        obligatorio del verificador topológico.
        """
        if not isinstance(phase1_handoff, Phase1LebesgueHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1LebesgueHandoff como prefijo formal."
            )

        scale_audit = self._certify_scale_isomorphism_from_certified_vectors(
            phase1_handoff.p_certified,
            phase1_handoff.c_certified,
        )

        centroid_certified, centroid_domain = self._certify_vector_domain(
            "centroid_C",
            centroid_C,
        )

        self._assert_nonnegative_and_bounded(
            "centroid_C",
            centroid_certified,
            upper_bound=None,
        )

        is_island = self._coerce_ontological_flag(
            "is_isolated_island",
            is_isolated_island,
        )

        logger.debug(
            "Fase 2 completada. Residuo L∞=%.6e | kappa=%.12e | "
            "||centroide||_∞=%.6e.",
            scale_audit.residual_infinity_norm,
            scale_audit.condition_number,
            centroid_domain.linf_norm,
        )

        return Phase2ScaleHandoff(
            phase1_handoff=phase1_handoff,
            scale_audit=scale_audit,
            centroid_certified=centroid_certified,
            centroid_domain=centroid_domain,
            is_isolated_island=is_island,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: EVALUACIÓN DE CENTROIDES TOPOLÓGICOS Y HOMOLOGÍA ESTRUCTURAL      ║
# ║   Exige <C_isla, e_mo> = 0 ∧ <C_isla, e_eq> = 0.                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_CentroidTopologyEnforcer(Phase2_ScaleIsomorphismCertifier):
    r"""
    Somete la geometría de las clases declaradas como “Suministro Puro”
    (Islas) a la verificación ortogonal covariante.

    Esta fase hereda la certificación de Lebesgue y el contrato afín.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Certificado de ortonormalidad de la base canónica
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_canonical_basis_orthogonality(self) -> float:
        r"""
        Verifica que la base canónica {e_sum, e_mo, e_eq} sea ortonormal:

            B Bᵀ = I₃.

        Retorna la desviación infinita de la Gram matrix.
        """
        basis = np.vstack((_E_SUM, _E_MO, _E_EQ))
        gram = basis @ basis.T
        identity = np.eye(_VECTOR_DIMENSION, dtype=np.float64)

        deviation = float(la.norm(gram - identity, ord=np.inf))
        tol = self._adaptive_tolerance(_ORTHOGONAL_CENTROID_TOLERANCE, basis)

        if deviation > tol:
            raise TopologicalCentroidAnomalyVeto(
                f"Base canónica corrupta: desviación de ortonormalidad "
                f"{deviation:.6e} > {tol:.6e}."
            )

        return deviation

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Verificación topológica de centroides
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_centroid_topology(
        self,
        centroid_C: NDArray[np.float64],
        is_isolated_island: bool,
    ) -> CentroidTopologyData:
        r"""
        Verifica el producto interno del centroide de clase C_k contra las
        bases canónicas de Mano de Obra y Equipos.

        Se asume:

            C_k = [Sum, MO, Eq]ᵀ.

        Si is_isolated_island == True, se exige:

            |<C_k, e_mo>| ≤ tol,
            |<C_k, e_eq>| ≤ tol,
            ||Proy_{span(e_mo,e_eq)} C_k||₂ ≤ tol.
        """
        centroid = self._coerce_finite_vector("centroid_C", centroid_C)

        self._assert_nonnegative_and_bounded(
            "centroid_C",
            centroid,
            upper_bound=None,
        )

        is_island = self._coerce_ontological_flag(
            "is_isolated_island",
            is_isolated_island,
        )

        basis_deviation = self._certify_canonical_basis_orthogonality()

        dot_mo = float(np.dot(centroid, _E_MO))
        dot_eq = float(np.dot(centroid, _E_EQ))
        projection_norm = float(np.hypot(dot_mo, dot_eq))

        tol = self._adaptive_tolerance(_ORTHOGONAL_CENTROID_TOLERANCE, centroid)

        if is_island:
            if (
                abs(dot_mo) > tol
                or abs(dot_eq) > tol
                or projection_norm > tol
            ):
                raise TopologicalCentroidAnomalyVeto(
                    f"Contradicción en el Complejo Simplicial. Un APU fue "
                    f"rotulado como 'Isla de Suministro', pero su centroide "
                    f"C_k exhibe componentes no ortogonales: "
                    f"<C,e_mo>={dot_mo:.6e}, <C,e_eq>={dot_eq:.6e}, "
                    f"||proy||₂={projection_norm:.6e}, tol={tol:.6e}."
                )

        # La propiedad estructural se satisface vacuamente si no es isla,
        # o efectivamente si la proyección es ortogonalmente nula.
        is_structurally_orthogonal = bool(
            (not is_island) or (projection_norm <= tol)
        )

        return CentroidTopologyData(
            inner_product_mo=dot_mo,
            inner_product_eq=dot_eq,
            projection_norm=projection_norm,
            basis_orthogonality_deviation=basis_deviation,
            topology_tolerance=tol,
            is_structurally_orthogonal=is_structurally_orthogonal,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2ScaleHandoff,
    ) -> OntologicalPartitionState:
        r"""
        Último método de la Fase 3.

        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal OntologicalPartitionState.
        """
        if not isinstance(phase2_handoff, Phase2ScaleHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2ScaleHandoff como prefijo formal."
            )

        centroid_audit = self._enforce_centroid_topology(
            phase2_handoff.centroid_certified,
            phase2_handoff.is_isolated_island,
        )

        state = OntologicalPartitionState(
            lebesgue_audit=phase2_handoff.phase1_handoff.lebesgue_audit,
            scale_audit=phase2_handoff.scale_audit,
            centroid_audit=centroid_audit,
            is_epistemologically_valid=True,
        )

        logger.info(
            "Partición Ontológica (APU Classifier) certificada con éxito. "
            "Vacío μ=%.6e | Residuo L∞=%.6e | kappa=%.12e | "
            "Ortogonalidad=%s.",
            state.lebesgue_audit.uncovered_measure,
            state.scale_audit.residual_infinity_norm,
            state.scale_audit.condition_number,
            str(centroid_audit.is_structurally_orthogonal),
        )

        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: APU CLASSIFIER AGENT                                 ║
# ║   Endofuntor Z_Classifier = Φ₃ ∘ Φ₂ ∘ Φ₁                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class APUClassifierAgent(Morphism, Phase3_CentroidTopologyEnforcer):
    r"""
    El Custodio de la Partición Ontológica.

    Gobierna el módulo `apu_classifier.py`, blindando las reglas vectorizadas y
    la asignación categórica contra vacíos estocásticos y desgarros topológicos.
    """

    def execute_ontological_partition_governance(
        self,
        uncovered_area_ratio: float,
        p_vector: NDArray[np.float64],
        c_vector: NDArray[np.float64],
        centroid_C: NDArray[np.float64],
        is_isolated_island: bool,
    ) -> OntologicalPartitionState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁: LebesgueMeasureAuditor
            Φ₂: ScaleIsomorphismCertifier
            Φ₃: CentroidTopologyEnforcer

        Retorna el objeto terminal OntologicalPartitionState.
        """
        phase1_handoff = self._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_area_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )

        phase2_handoff = self._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=is_isolated_island,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )

    def __call__(
        self,
        uncovered_area_ratio: float,
        p_vector: NDArray[np.float64],
        c_vector: NDArray[np.float64],
        centroid_C: NDArray[np.float64],
        is_isolated_island: bool,
    ) -> OntologicalPartitionState:
        r"""Alias invocable del endofuntor de gobierno ontológico."""
        return self.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_area_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=is_isolated_island,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "APUClassifierAgentError",
    "DomainIntegrityViolationError",
    "SimplexMembershipViolationError",
    "LebesgueMeasureViolationError",
    "ScaleInvarianceCollapseError",
    "TopologicalCentroidAnomalyVeto",
    "VectorDomainCertificate",
    "LebesgueAuditData",
    "ScaleIsomorphismData",
    "CentroidTopologyData",
    "Phase1LebesgueHandoff",
    "Phase2ScaleHandoff",
    "OntologicalPartitionState",
    "Phase1_LebesgueMeasureAuditor",
    "Phase2_ScaleIsomorphismCertifier",
    "Phase3_CentroidTopologyEnforcer",
    "APUClassifierAgent",
]