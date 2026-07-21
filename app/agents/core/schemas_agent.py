# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Schemas Agent (Custodio de los Invariantes Estructurales)           ║
║ Ruta   : app/agents/core/schemas_agent.py                                    ║
║ Versión: 2.0.0-Topological-Bipartite-Thermodynamic-Strict                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DEL RIESGO (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el módulo `schemas.py`. Abandona la validación de tipos
escalares pasiva para erigirse como un proyector geométrico que somete el estado
financiero a un complejo simplicial restringido.

Impone:

1. No-degeneración de la variedad financiera:
       Q ⪰ 0, P ⪰ 0, V ⪰ 0.

2. Conservación de energía financiera bajo producto de Hadamard:
       ||V - (Q ⊙ P)||_∞ ≤ ε_abs + ε_rel · escala.

3. Saturación dimensional física:
       Q ∈ [0, 10^6], P ∈ [0, 10^9], Rend ∈ [0, 10^3].

4. Idempotencia estricta de normalización:
       f(f(x)) = f(x).

5. Estabilidad termodinámica estructural:
       H_norm = -Σ p_i ln(p_i) / ln(n) ≥ 0.1,
       D(a) = |categorias| / 5 > 0.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación de Geometría Bipartita y Conservación.
Fase 2 → Retractos de Deformación, Saturación Dimensional e Idempotencia.
Fase 3 → Auditoría de Termodinámica Estructural.

El último método de la Fase 1 es el handoff formal que inaugura la Fase 2.
El último método de la Fase 2 es el handoff formal que inaugura la Fase 3.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Final, FrozenSet, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
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


logger = logging.getLogger("MIC.Core.SchemasAgent")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICO-MATEMÁTICAS Y COTAS DIMENSIONALES
# ═══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancias híbridas para fricción numérica IEEE 754.
_EPSILON_ABS: Final[float] = 1e-10
_EPSILON_REL: Final[float] = 1e-6

# Tolerancia base para no negatividad y fronteras físicas.
_NONNEGATIVITY_TOLERANCE: Final[float] = 1e-12

# Hipercubo de acotación física (saturación dimensional).
_MAX_Q: Final[float] = 1e6       # Límite logístico.
_MAX_P: Final[float] = 1e9       # Límite de capitalización.
_MAX_REND: Final[float] = 1e3    # Límite termodinámico del trabajo.

# Umbral mínimo de entropía normalizada para evitar SPOF / pirámide invertida.
_ENTROPY_MIN_THRESHOLD: Final[float] = 0.1

# Cardinalidad de referencia para diversidad categórica normalizada.
_CATEGORY_CARDINALITY_REFERENCE: Final[float] = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS (VETOS ABSOLUTOS)
# ═══════════════════════════════════════════════════════════════════════════════

class SchemasAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de Invariantes Estructurales."""
    pass


class DomainIntegrityViolationError(SchemasAgentError):
    """Detonada cuando un vector, escalar, conjunto o función viola su dominio."""
    pass


class BipartiteDegeneracyError(SchemasAgentError):
    r"""
    Detonada si se viola la conservación de energía financiera:

        ||V - (Q ⊙ P)||_∞ > ε,

    o si existe energía financiera negativa (antimateria económica).
    """
    pass


class DimensionalSaturationError(SchemasAgentError):
    r"""
    Detonada si se desborda el hipercubo físico o falla el axioma de
    idempotencia:

        f(f(x)) ≠ f(x).
    """
    pass


class StructuralThermodynamicError(SchemasAgentError):
    r"""
    Detonada si H_norm → 0 o D(a) → 0, evidenciando una pirámide invertida,
    un Punto de Fallo Único (SPOF) o degeneración categórica.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ArrayDomainCertificate:
    r"""Certificado material de un vector financiero en R^n."""
    name: str
    size: int
    l1_norm: float
    l2_norm: float
    linf_norm: float
    is_finite: bool


@dataclass(frozen=True, slots=True)
class BipartiteGeometryData:
    r"""Artefacto de Fase 1. Certificado de conservación de energía financiera."""
    max_residual_error: float
    dynamic_tolerance: float
    is_energy_conserved: bool


@dataclass(frozen=True, slots=True)
class DimensionalSaturationData:
    r"""Artefacto de Fase 2. Certificado de retracto, acotación e idempotencia."""
    max_Q_observed: float
    max_P_observed: float
    Rend_val: float
    is_idempotent: bool
    is_physically_bounded: bool


@dataclass(frozen=True, slots=True)
class StructuralThermodynamicsData:
    r"""Artefacto de Fase 3. Certificado de entropía y diversidad categórica."""
    shannon_entropy_norm: float
    categorical_diversity: float
    entropy_threshold: float
    is_thermodynamically_stable: bool


@dataclass(frozen=True, slots=True)
class Phase1GeometryHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.

    Este objeto es la continuación material de la geometría bipartita y el
    prefijo de entrada obligatorio de la Fase 2.
    """
    geometry_audit: BipartiteGeometryData
    V_certified: NDArray[np.float64]
    Q_certified: NDArray[np.float64]
    P_certified: NDArray[np.float64]
    V_domain: ArrayDomainCertificate
    Q_domain: ArrayDomainCertificate
    P_domain: ArrayDomainCertificate


@dataclass(frozen=True, slots=True)
class Phase2SaturationHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.

    Este objeto transporta la certificación dimensional, la idempotencia del
    retracto y el conjunto categórico saneado para la auditoría termodinámica.
    """
    phase1_handoff: Phase1GeometryHandoff
    saturation_audit: DimensionalSaturationData
    categories_certified: FrozenSet[str]


@dataclass(frozen=True, slots=True)
class StructuralInvariantState:
    r"""Objeto final del endofuntor Z_Schemas."""
    geometry_audit: BipartiteGeometryData
    saturation_audit: DimensionalSaturationData
    thermo_audit: StructuralThermodynamicsData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: GEOMETRÍA BIPARTITA Y CONSERVACIÓN DE ENERGÍA                     ║
# ║   Exige Q ⪰ 0, P ⪰ 0, V ⪰ 0 y ||V - (Q ⊙ P)||_∞ ≤ ε.                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_BipartiteGeometryCertifier:
    r"""
    Garantiza que la inyección de costos no produzca "antimateria" financiera
    y verifica el producto de Hadamard para la conservación exacta del valor.
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

            tol = max(tol_base, κ · ε_máquina · tamaño · escala)

        donde la escala se estima por norma L∞ del objeto de referencia.
        """
        if isinstance(reference, np.ndarray):
            size = max(1, int(reference.size))
            if reference.size == 0:
                scale = 1.0
            else:
                scale = max(
                    1.0,
                    float(la.norm(reference.ravel(), ord=np.inf)),
                )
        else:
            size = 1
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        return max(
            float(base_tolerance),
            10.0 * _MACHINE_EPSILON * size * scale,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción de escalares finitos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_scalar(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Materializa un escalar float64 y verifica finitud absoluta.
        """
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"El escalar ontológico '{name}' no puede materializarse "
                f"como float."
            ) from exc

        if not np.isfinite(scalar):
            raise DomainIntegrityViolationError(
                f"El escalar ontológico '{name}' no es finito."
            )

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción de vectores finitos
    # ─────────────────────────────────────────────────────────────────────────
    def _canonicalize_finite_vector(
        self,
        name: str,
        vector: Any,
    ) -> NDArray[np.float64]:
        r"""
        Materializa un vector float64 unidimensional y verifica finitud
        absoluta. Se canonicaliza cualquier forma a vector fila 1D.
        """
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' no puede materializarse "
                f"como NDArray[np.float64]."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)

        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' está vacío. "
                f"Un espacio financiero degenerado no puede ser certificado."
            )

        if not np.all(np.isfinite(arr)):
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' contiene componentes no "
                f"finitas. NaN o infinitos rompen la variedad de datos."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Certificado de dominio vectorial
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_array_domain(
        self,
        name: str,
        arr: NDArray[np.float64],
    ) -> ArrayDomainCertificate:
        r"""
        Emite certificado normativo L¹, L² y L∞ sobre un vector ya saneado.
        """
        l1_norm = float(la.norm(arr, ord=1))
        l2_norm = float(la.norm(arr, ord=2))
        linf_norm = float(la.norm(arr, ord=np.inf))

        return ArrayDomainCertificate(
            name=name,
            size=int(arr.size),
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            linf_norm=linf_norm,
            is_finite=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. No-negatividad absoluta de energía financiera
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_nonnegative_energy_vector(
        self,
        name: str,
        arr: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Verifica que un vector financiero sea no negativo. Se admite ruido
        negativo infinitesimal y se proyecta a cero.
        """
        tol = self._adaptive_tolerance(_NONNEGATIVITY_TOLERANCE, arr)

        if np.any(arr < -tol):
            raise BipartiteDegeneracyError(
                f"Detectada antimateria económica en '{name}': "
                f"componentes negativas incompatibles con Q ⪰ 0, P ⪰ 0, V ⪰ 0."
            )

        return np.where(arr < 0.0, 0.0, arr)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Tolerancia dinámica de conservación
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_dynamic_conservation_tolerance(
        self,
        V: NDArray[np.float64],
        QP: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la tolerancia híbrida de conservación:

            ε = ε_abs + ε_rel · max(1, ||V||∞, ||Q⊙P||∞)
                + κ · ε_máquina · n · escala.
        """
        max_V = float(np.max(np.abs(V))) if V.size > 0 else 0.0
        max_QP = float(np.max(np.abs(QP))) if QP.size > 0 else 0.0
        scale = max(1.0, max_V, max_QP)
        size = max(1, int(V.size))

        return (
            _EPSILON_ABS
            + _EPSILON_REL * scale
            + 10.0 * _MACHINE_EPSILON * size * scale
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. Certificación interna de arreglos bipartitos
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_bipartite_arrays(
        self,
        V_array: NDArray[np.float64],
        Q_array: NDArray[np.float64],
        P_array: NDArray[np.float64],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        BipartiteGeometryData,
        ArrayDomainCertificate,
        ArrayDomainCertificate,
        ArrayDomainCertificate,
    ]:
        r"""
        Sanea V, Q, P y certifica la ley de conservación:

            V = Q ⊙ P.

        Retorna los vectores certificados, el audit y los certificados de
        dominio.
        """
        V = self._canonicalize_finite_vector("V_array", V_array)
        Q = self._canonicalize_finite_vector("Q_array", Q_array)
        P = self._canonicalize_finite_vector("P_array", P_array)

        if not (V.size == Q.size == P.size):
            raise BipartiteDegeneracyError(
                f"Geometría bipartita incompatible: "
                f"|V|={V.size}, |Q|={Q.size}, |P|={P.size}. "
                f"El producto de Hadamard exige igualdad dimensional."
            )

        V = self._certify_nonnegative_energy_vector("V_array", V)
        Q = self._certify_nonnegative_energy_vector("Q_array", Q)
        P = self._certify_nonnegative_energy_vector("P_array", P)

        QP_hadamard = Q * P

        if not np.all(np.isfinite(QP_hadamard)):
            raise BipartiteDegeneracyError(
                "El producto de Hadamard Q⊙P no es finito. "
                "Existe desbordamiento aritmético o singularidad material."
            )

        residual_vector = np.abs(V - QP_hadamard)

        if not np.all(np.isfinite(residual_vector)):
            raise BipartiteDegeneracyError(
                "El residuo de conservación V - (Q⊙P) no es finito."
            )

        max_residual = float(np.max(residual_vector))
        dynamic_tolerance = self._compute_dynamic_conservation_tolerance(
            V,
            QP_hadamard,
        )

        if max_residual > dynamic_tolerance:
            raise BipartiteDegeneracyError(
                f"Fractura en la Conservación de Energía Financiera. "
                f"Residuo máximo ||V - (Q⊙P)||_∞ = "
                f"{max_residual:.6e} > {dynamic_tolerance:.6e}."
            )

        geometry_audit = BipartiteGeometryData(
            max_residual_error=max_residual,
            dynamic_tolerance=dynamic_tolerance,
            is_energy_conserved=True,
        )

        V_domain = self._certify_array_domain("V_array_certified", V)
        Q_domain = self._certify_array_domain("Q_array_certified", Q)
        P_domain = self._certify_array_domain("P_array_certified", P)

        return (
            V,
            Q,
            P,
            geometry_audit,
            V_domain,
            Q_domain,
            P_domain,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8. Wrapper público / retrocompatible de conservación bipartita
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_bipartite_dag_and_conservation(
        self,
        V_array: NDArray[np.float64],
        Q_array: NDArray[np.float64],
        P_array: NDArray[np.float64],
    ) -> BipartiteGeometryData:
        r"""
        Aplica la ley de conservación evaluando el residuo bajo la norma del
        supremo. Conserva la signatura original de Fase 1.
        """
        (
            _V,
            _Q,
            _P,
            geometry_audit,
            _V_domain,
            _Q_domain,
            _P_domain,
        ) = self._certify_bipartite_arrays(V_array, Q_array, P_array)

        return geometry_audit

    # ─────────────────────────────────────────────────────────────────────────
    # 1.9. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_certify_and_handoff_to_phase2(
        self,
        V_array: NDArray[np.float64],
        Q_array: NDArray[np.float64],
        P_array: NDArray[np.float64],
    ) -> Phase1GeometryHandoff:
        r"""
        Último método de la Fase 1.

        Su definición formal es la continuación directa de la Fase 2:
        entrega V, Q, P certificados y el audit de conservación como prefijo
        obligatorio del retracto dimensional.
        """
        (
            V_certified,
            Q_certified,
            P_certified,
            geometry_audit,
            V_domain,
            Q_domain,
            P_domain,
        ) = self._certify_bipartite_arrays(V_array, Q_array, P_array)

        logger.debug(
            "Fase 1 completada. Residuo máximo=%.6e | tolerancia=%.6e | "
            "n=%d.",
            geometry_audit.max_residual_error,
            geometry_audit.dynamic_tolerance,
            V_certified.size,
        )

        return Phase1GeometryHandoff(
            geometry_audit=geometry_audit,
            V_certified=V_certified,
            Q_certified=Q_certified,
            P_certified=P_certified,
            V_domain=V_domain,
            Q_domain=Q_domain,
            P_domain=P_domain,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: RETRACTOS DE DEFORMACIÓN Y SATURACIÓN DIMENSIONAL                 ║
# ║   Evalúa Q ∈ [0, 10^6], P ∈ [0, 10^9], Rend ∈ [0, 10^3] y f(f(x)) = f(x).   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_DimensionalSaturationEnforcer(Phase1_BipartiteGeometryCertifier):
    r"""
    Aplica el hipercubo de acotación física y verifica que las funciones de
    normalización sean proyectores ortogonales matemáticamente perfectos.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Acotación física de escalares
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_scalar_physical_bounds(
        self,
        Q_val: float,
        P_val: float,
        Rend_val: float,
    ) -> Tuple[float, float, float]:
        r"""
        Certifica que los escalares Q, P y Rend pertenezcan al hipercubo
        físico. Sanea ruido infinitesimal en fronteras.
        """
        Q = self._coerce_finite_scalar("Q_val", Q_val)
        P = self._coerce_finite_scalar("P_val", P_val)
        Rend = self._coerce_finite_scalar("Rend_val", Rend_val)

        Q_tol = self._adaptive_tolerance(_NONNEGATIVITY_TOLERANCE, Q)
        P_tol = self._adaptive_tolerance(_NONNEGATIVITY_TOLERANCE, P)
        Rend_tol = self._adaptive_tolerance(_NONNEGATIVITY_TOLERANCE, Rend)

        if Q < -Q_tol or Q > _MAX_Q + Q_tol:
            raise DimensionalSaturationError(
                f"Cantidad Q={Q:.6e} escapa del límite logístico "
                f"[0, {_MAX_Q:.6e}] dentro de tolerancia {Q_tol:.6e}."
            )

        if P < -P_tol or P > _MAX_P + P_tol:
            raise DimensionalSaturationError(
                f"Precio P={P:.6e} escapa del límite de capitalización "
                f"[0, {_MAX_P:.6e}] dentro de tolerancia {P_tol:.6e}."
            )

        if Rend < -Rend_tol or Rend > _MAX_REND + Rend_tol:
            raise DimensionalSaturationError(
                f"Rendimiento Rend={Rend:.6e} escapa del límite termodinámico "
                f"[0, {_MAX_REND:.6e}] dentro de tolerancia {Rend_tol:.6e}."
            )

        # Proyección frontera de ruido infinitesimal.
        if Q < 0.0:
            Q = 0.0
        if Q > _MAX_Q:
            Q = _MAX_Q

        if P < 0.0:
            P = 0.0
        if P > _MAX_P:
            P = _MAX_P

        if Rend < 0.0:
            Rend = 0.0
        if Rend > _MAX_REND:
            Rend = _MAX_REND

        return Q, P, Rend

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Acotación física de vectores completos
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_vector_physical_bounds(
        self,
        name: str,
        arr: NDArray[np.float64],
        upper_bound: float,
    ) -> None:
        r"""
        Verifica que todas las componentes de un vector estén dentro del
        intervalo físico [0, upper_bound].
        """
        tol = self._adaptive_tolerance(_NONNEGATIVITY_TOLERANCE, arr)
        bound = float(upper_bound)

        if np.any(arr < -tol):
            raise DimensionalSaturationError(
                f"Vector '{name}' posee componentes negativas incompatibles "
                f"con el hipercubo físico."
            )

        if np.any(arr > bound + tol):
            raise DimensionalSaturationError(
                f"Vector '{name}' excede la cota física {bound:.6e} "
                f"dentro de tolerancia {tol:.6e}."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Comparación segura de idempotencia
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_idempotence_equality(
        self,
        left: Any,
        right: Any,
    ) -> bool:
        r"""
        Compara de forma segura f(x) y f(f(x)), evitando ambigüedades de
        verdad en arreglos de NumPy.
        """
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return bool(np.array_equal(left, right))
            except Exception:
                return False

        try:
            return bool(left == right)
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Certificación de idempotencia del normalizador
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_normalizer_idempotence(
        self,
        normalizer_func: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        r"""
        Verifica el retracto topológico del operador normalizador:

            f(f(x)) = f(x).

        Si el operador no puede aplicarse o no es idempotente, se veta la
        constitución de datos.
        """
        if not callable(normalizer_func):
            raise DimensionalSaturationError(
                "normalizer_func debe ser un operador callable."
            )

        if not isinstance(test_string, str):
            raise DomainIntegrityViolationError(
                "test_string debe ser una cadena de prueba para el operador "
                "normalizador."
            )

        try:
            f_x = normalizer_func(test_string)
        except Exception as exc:
            raise DimensionalSaturationError(
                "El operador normalizador falló al evaluar f(x)."
            ) from exc

        try:
            f_f_x = normalizer_func(f_x)
        except Exception as exc:
            raise DimensionalSaturationError(
                "El operador normalizador falló al evaluar f(f(x))."
            ) from exc

        if not self._safe_idempotence_equality(f_x, f_f_x):
            raise DimensionalSaturationError(
                "Ruptura Funtorial. El operador no es idempotente: "
                "f(f(x)) ≠ f(x)."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. Wrapper público / retrocompatible de saturación e idempotencia
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_dimensional_saturation_and_idempotence(
        self,
        Q_val: float,
        P_val: float,
        Rend_val: float,
        normalizer_func: Callable[[Any], Any],
        test_string: str,
    ) -> DimensionalSaturationData:
        r"""
        Verifica saturación escalar y retracto topológico. Conserva la
        signatura original de Fase 2.
        """
        Q, P, Rend = self._certify_scalar_physical_bounds(
            Q_val,
            P_val,
            Rend_val,
        )

        self._certify_normalizer_idempotence(normalizer_func, test_string)

        return DimensionalSaturationData(
            max_Q_observed=Q,
            max_P_observed=P,
            Rend_val=Rend,
            is_idempotent=True,
            is_physically_bounded=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6. Saneamiento del conjunto categórico
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_categories_set(
        self,
        categories_set: Any,
    ) -> FrozenSet[str]:
        r"""
        Certifica que las categorías sean un conjunto finito de cadenas no
        vacías. Rechaza strings planos iterables para evitar falsos conjuntos.
        """
        if categories_set is None:
            raise StructuralThermodynamicError(
                "Degeneración Categórica. El conjunto de categorías es nulo."
            )

        if isinstance(categories_set, str):
            raise DomainIntegrityViolationError(
                "categories_set no debe ser una cadena plana. "
                "Se requiere un conjunto de categorías."
            )

        try:
            raw_categories = (
                categories_set
                if isinstance(categories_set, (set, frozenset))
                else set(categories_set)
            )
        except TypeError as exc:
            raise DomainIntegrityViolationError(
                "categories_set no puede interpretarse como un conjunto."
            ) from exc

        cleaned: set[str] = set()

        for item in raw_categories:
            if item is None:
                raise StructuralThermodynamicError(
                    "Degeneración Categórica. Existe una categoría nula."
                )

            text = item if isinstance(item, str) else str(item)
            text = text.strip()

            if not text:
                raise StructuralThermodynamicError(
                    "Degeneración Categórica. Existe una categoría vacía."
                )

            cleaned.add(text)

        if not cleaned:
            raise StructuralThermodynamicError(
                "Degeneración Categórica. El nodo exhibe un monotipo logístico "
                "o un conjunto vacío D(a)=0."
            )

        return frozenset(cleaned)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.7. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_enforce_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1GeometryHandoff,
        Rend_val: float,
        normalizer_func: Callable[[Any], Any],
        test_string: str,
        categories_set: Any,
    ) -> Phase2SaturationHandoff:
        r"""
        Último método de la Fase 2.

        Su definición formal es la continuación directa de la Fase 3:
        entrega la saturación dimensional certificada y las categorías saneadas
        como prefijo obligatorio de la auditoría termodinámica.
        """
        if not isinstance(phase1_handoff, Phase1GeometryHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1GeometryHandoff como prefijo formal."
            )

        # Saturación dimensional completa sobre los arreglos certificados.
        self._certify_vector_physical_bounds(
            "Q_array",
            phase1_handoff.Q_certified,
            _MAX_Q,
        )
        self._certify_vector_physical_bounds(
            "P_array",
            phase1_handoff.P_certified,
            _MAX_P,
        )

        max_Q_observed = float(np.max(phase1_handoff.Q_certified))
        max_P_observed = float(np.max(phase1_handoff.P_certified))

        # Certificación escalar de los observables máximos y de Rend.
        (
            max_Q_sanitized,
            max_P_sanitized,
            Rend_sanitized,
        ) = self._certify_scalar_physical_bounds(
            max_Q_observed,
            max_P_observed,
            Rend_val,
        )

        # Retracto de deformación: idempotencia estricta.
        self._certify_normalizer_idempotence(normalizer_func, test_string)

        # Saneamiento categórico para la Fase 3.
        categories_certified = self._coerce_categories_set(categories_set)

        saturation_audit = DimensionalSaturationData(
            max_Q_observed=max_Q_sanitized,
            max_P_observed=max_P_sanitized,
            Rend_val=Rend_sanitized,
            is_idempotent=True,
            is_physically_bounded=True,
        )

        logger.debug(
            "Fase 2 completada. max(Q)=%.6e | max(P)=%.6e | Rend=%.6e | "
            "categorias=%d.",
            saturation_audit.max_Q_observed,
            saturation_audit.max_P_observed,
            saturation_audit.Rend_val,
            len(categories_certified),
        )

        return Phase2SaturationHandoff(
            phase1_handoff=phase1_handoff,
            saturation_audit=saturation_audit,
            categories_certified=categories_certified,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: AUDITORÍA DE TERMODINÁMICA ESTRUCTURAL                            ║
# ║   Computa H_norm = -Σ p_i ln(p_i) / ln(n) y D(a).                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_StructuralThermodynamicAuditor(Phase2_DimensionalSaturationEnforcer):
    r"""
    Impide la instanciación de "Pirámides Invertidas" evaluando la entropía
    de la energía financiera y la diversidad ortogonal de los suministros.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Entropía de Shannon normalizada
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_shannon_entropy_normalized(
        self,
        V_array: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la entropía normalizada de Shannon sobre la energía financiera:

            p_i = V_i / Σ V_j,
            H = -Σ p_i ln(p_i),
            H_norm = H / ln(n).

        Veta SPOF absoluto cuando n = 1 o cuando H_norm < 0.1.
        """
        V = self._canonicalize_finite_vector("V_array", V_array)
        V = self._certify_nonnegative_energy_vector("V_array", V)

        n = int(V.size)

        if n == 0:
            raise StructuralThermodynamicError(
                "Vacío topológico. No existe energía financiera para auditar."
            )

        if n == 1:
            raise StructuralThermodynamicError(
                "Singularidad Logística (SPOF). Nodo dependiente de un único "
                "insumo masivo (H=0)."
            )

        total_v = float(np.sum(V))
        energy_tol = max(
            _EPSILON_ABS,
            self._adaptive_tolerance(_MACHINE_EPSILON, V),
        )

        if not np.isfinite(total_v) or total_v <= energy_tol:
            raise StructuralThermodynamicError(
                "Energía financiera total nula o no finita. "
                "La variedad termodinámica es degenerada."
            )

        p_array = V / total_v

        if not np.all(np.isfinite(p_array)):
            raise StructuralThermodynamicError(
                "La distribución de energía financiera p_i no es finita."
            )

        p_safe = p_array[p_array > _MACHINE_EPSILON]

        if p_safe.size == 0:
            raise StructuralThermodynamicError(
                "La distribución de energía financiera carece de masa "
                "positiva computable."
            )

        H = -float(np.sum(p_safe * np.log(p_safe)))

        if not np.isfinite(H):
            raise StructuralThermodynamicError(
                "La entropía de Shannon no es finita."
            )

        H_norm = float(H / math.log(n))

        if not np.isfinite(H_norm):
            raise StructuralThermodynamicError(
                "La entropía normalizada no es finita."
            )

        # Saneamiento de ruido numérico en [0, 1].
        if H_norm < -1e-12 or H_norm > 1.0 + 1e-12:
            raise StructuralThermodynamicError(
                f"Entropía normalizada fuera del intervalo [0, 1]: "
                f"H_norm={H_norm:.6e}."
            )

        if H_norm < 0.0:
            H_norm = 0.0
        if H_norm > 1.0:
            H_norm = 1.0

        if H_norm < _ENTROPY_MIN_THRESHOLD:
            raise StructuralThermodynamicError(
                f"Pirámide Invertida. Concentración anómala de energía "
                f"térmica en el nodo. Entropía H_norm={H_norm:.6f} < "
                f"{_ENTROPY_MIN_THRESHOLD:.2f} (SPOF inminente)."
            )

        return H_norm

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Diversidad categórica
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_categorical_diversity(
        self,
        categories_set: Any,
    ) -> float:
        r"""
        Calcula la diversidad categórica:

            D(a) = |tipos únicos| / 5.

        Veta monotipos logísticos cuando D(a) = 0.
        """
        categories = self._coerce_categories_set(categories_set)

        diversity = float(len(categories) / _CATEGORY_CARDINALITY_REFERENCE)

        if not np.isfinite(diversity) or diversity <= _MACHINE_EPSILON:
            raise StructuralThermodynamicError(
                "Degeneración Categórica. El nodo exhibe un monotipo "
                "logístico D(a)=0."
            )

        return diversity

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Wrapper público / retrocompatible de termodinámica estructural
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_structural_entropy_and_diversity(
        self,
        V_array: NDArray[np.float64],
        categories_set: Any,
    ) -> StructuralThermodynamicsData:
        r"""
        Evalúa entropía de Shannon y diversidad categórica. Conserva la
        signatura original de Fase 3.
        """
        H_norm = self._compute_shannon_entropy_normalized(V_array)
        D_a = self._compute_categorical_diversity(categories_set)

        return StructuralThermodynamicsData(
            shannon_entropy_norm=H_norm,
            categorical_diversity=D_a,
            entropy_threshold=_ENTROPY_MIN_THRESHOLD,
            is_thermodynamically_stable=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2SaturationHandoff,
    ) -> StructuralInvariantState:
        r"""
        Último método de la Fase 3.

        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal StructuralInvariantState.
        """
        if not isinstance(phase2_handoff, Phase2SaturationHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2SaturationHandoff como prefijo formal."
            )

        thermo_audit = self._audit_structural_entropy_and_diversity(
            phase2_handoff.phase1_handoff.V_certified,
            phase2_handoff.categories_certified,
        )

        state = StructuralInvariantState(
            geometry_audit=phase2_handoff.phase1_handoff.geometry_audit,
            saturation_audit=phase2_handoff.saturation_audit,
            thermo_audit=thermo_audit,
            is_epistemologically_valid=True,
        )

        logger.info(
            "Constitución de Datos (Schemas) certificada categóricamente. "
            "Δ(Q⊙P-V)=%.6e | H_norm=%.6f | Diversidad=%.6f.",
            state.geometry_audit.max_residual_error,
            state.thermo_audit.shannon_entropy_norm,
            state.thermo_audit.categorical_diversity,
        )

        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SCHEMAS AGENT                                        ║
# ║   Endofuntor Z_Schemas = Φ₃ ∘ Φ₂ ∘ Φ₁                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class SchemasAgent(Morphism, Phase3_StructuralThermodynamicAuditor):
    r"""
    El Custodio de los Invariantes Estructurales.

    Transmuta la declaración pasiva de atributos de Python en un proyector
    geométrico que veta la instanciación de cualquier clase de datos que
    viole la física y la topología del presupuesto.
    """

    def execute_structural_invariant_governance(
        self,
        V_array: NDArray[np.float64],
        Q_array: NDArray[np.float64],
        P_array: NDArray[np.float64],
        Rend_val: float,
        normalizer_func: Callable[[Any], Any],
        test_string: str,
        categories_set: Any,
    ) -> StructuralInvariantState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁: BipartiteGeometryCertifier
            Φ₂: DimensionalSaturationEnforcer
            Φ₃: StructuralThermodynamicAuditor
        """
        phase1_handoff = self._phase1_certify_and_handoff_to_phase2(
            V_array=V_array,
            Q_array=Q_array,
            P_array=P_array,
        )

        phase2_handoff = self._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            Rend_val=Rend_val,
            normalizer_func=normalizer_func,
            test_string=test_string,
            categories_set=categories_set,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )

    def __call__(
        self,
        V_array: NDArray[np.float64],
        Q_array: NDArray[np.float64],
        P_array: NDArray[np.float64],
        Rend_val: float,
        normalizer_func: Callable[[Any], Any],
        test_string: str,
        categories_set: Any,
    ) -> StructuralInvariantState:
        r"""Alias invocable del endofuntor de gobierno estructural."""
        return self.execute_structural_invariant_governance(
            V_array=V_array,
            Q_array=Q_array,
            P_array=P_array,
            Rend_val=Rend_val,
            normalizer_func=normalizer_func,
            test_string=test_string,
            categories_set=categories_set,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "SchemasAgentError",
    "DomainIntegrityViolationError",
    "BipartiteDegeneracyError",
    "DimensionalSaturationError",
    "StructuralThermodynamicError",
    "ArrayDomainCertificate",
    "BipartiteGeometryData",
    "DimensionalSaturationData",
    "StructuralThermodynamicsData",
    "Phase1GeometryHandoff",
    "Phase2SaturationHandoff",
    "StructuralInvariantState",
    "Phase1_BipartiteGeometryCertifier",
    "Phase2_DimensionalSaturationEnforcer",
    "Phase3_StructuralThermodynamicAuditor",
    "SchemasAgent",
]