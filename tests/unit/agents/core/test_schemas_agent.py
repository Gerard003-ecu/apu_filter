# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Schemas Agent (Suite de Validación de Invariantes Estructurales)          ║
║  Ruta   : tests/unit/agents/core/test_schemas_agent.py                                   ║
║  Versión: 2.0.0-Topological-Bipartite-Thermodynamic-Strict                               ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ARQUITECTURA DE PRUEBAS (Composición Funtorial Φ₃ ∘ Φ₂ ∘ Φ₁):                           ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Este módulo de pruebas implementa una batería exhaustiva que valida la integridad       ║
║  topológica, geométrica y termodinámica del endofuntor Z_Schemas.                        ║
║                                                                                          ║
║  FASE 1 → Geometría Bipartita y Conservación de Energía                                  ║
║           Valida: Q ⪰ 0, P ⪰ 0, V ⪰ 0 y ||V - (Q ⊙ P)||_∞ ≤ ε                            ║
║                                                                                          ║
║  FASE 2 → Retractos de Deformación y Saturación Dimensional                              ║
║           Valida: Hipercubo físico [0, 10^6], [0, 10^9], [0, 10^3] y f(f(x)) = f(x)      ║
║                                                                                          ║
║  FASE 3 → Auditoría de Termodinámica Estructural                                         ║
║           Valida: H_norm ≥ 0.1 y D(a) > 0 (anti-SPOF / anti-pirámide invertida)          ║
║                                                                                          ║
║  COBERTURA DE EXCEPCIONES TOPOLÓGICAS:                                                   ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  • DomainIntegrityViolationError    → Violaciones de dominio ontológico                  ║
║  • BipartiteDegeneracyError         → Violaciones de conservación energética             ║
║  • DimensionalSaturationError       → Desbordamiento del hipercubo físico                ║
║  • StructuralThermodynamicError     → Colapso entrópico / degeneración categórica        ║
║                                                                                          ║
║  EJECUCIÓN:                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  $ pytest tests/unit/agents/core/test_schemas_agent.py -v --cov=app.agents.core          ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIONES Y CONFIGURACIÓN DEL ENTORNO DE PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════════════════════

import pytest
import numpy as np
import math
from typing import Any, Callable
from numpy.typing import NDArray

# Importación del módulo bajo prueba
from app.agents.core.schemas_agent import (
    # Excepciones Topológicas
    SchemasAgentError,
    DomainIntegrityViolationError,
    BipartiteDegeneracyError,
    DimensionalSaturationError,
    StructuralThermodynamicError,
    # Estructuras Inmutables (DTOs)
    ArrayDomainCertificate,
    BipartiteGeometryData,
    DimensionalSaturationData,
    StructuralThermodynamicsData,
    Phase1GeometryHandoff,
    Phase2SaturationHandoff,
    StructuralInvariantState,
    # Fases Anidadas
    Phase1_BipartiteGeometryCertifier,
    Phase2_DimensionalSaturationEnforcer,
    Phase3_StructuralThermodynamicAuditor,
    # Orquestador Supremo
    SchemasAgent,
    # Constantes Físico-Matemáticas
    _MACHINE_EPSILON,
    _EPSILON_ABS,
    _EPSILON_REL,
    _NONNEGATIVITY_TOLERANCE,
    _MAX_Q,
    _MAX_P,
    _MAX_REND,
    _ENTROPY_MIN_THRESHOLD,
    _CATEGORY_CARDINALITY_REFERENCE,
)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §A. FIXTURES Y UTILITARIOS DE PRUEBA (Infraestructura Categórica)
# ═══════════════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def schemas_agent() -> SchemasAgent:
    """
    Fixture: Instancia del Orquestador Supremo SchemasAgent.
    Retorna el endofuntor completo para pruebas de integración.
    """
    return SchemasAgent()


@pytest.fixture
def phase1_certifier() -> Phase1_BipartiteGeometryCertifier:
    """
    Fixture: Instancia de Phase1_BipartiteGeometryCertifier.
    Para pruebas unitarias de la Fase 1.
    """
    return Phase1_BipartiteGeometryCertifier()


@pytest.fixture
def phase2_enforcer() -> Phase2_DimensionalSaturationEnforcer:
    """
    Fixture: Instancia de Phase2_DimensionalSaturationEnforcer.
    Para pruebas unitarias de la Fase 2.
    """
    return Phase2_DimensionalSaturationEnforcer()


@pytest.fixture
def phase3_auditor() -> Phase3_StructuralThermodynamicAuditor:
    """
    Fixture: Instancia de Phase3_StructuralThermodynamicAuditor.
    Para pruebas unitarias de la Fase 3.
    """
    return Phase3_StructuralThermodynamicAuditor()


@pytest.fixture
def valid_arrays() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Fixture: Vectores financieros válidos que satisfacen V = Q ⊙ P.
    Genera arrays que cumplen la conservación de energía financiera.
    """
    Q = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    P = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    V = Q * P  # Producto de Hadamard exacto
    return V, Q, P


@pytest.fixture
def valid_categories() -> frozenset[str]:
    """
    Fixture: Conjunto de categorías válido para diversidad D(a) > 0.
    """
    return frozenset(["categoria_1", "categoria_2", "categoria_3"])


@pytest.fixture
def idempotent_normalizer() -> Callable[[Any], Any]:
    """
    Fixture: Función normalizadora idempotente f(f(x)) = f(x).
    """
    def normalizer(x: Any) -> str:
        return str(x).strip().lower()
    return normalizer


@pytest.fixture
def test_string() -> str:
    """
    Fixture: Cadena de prueba para validación de idempotencia.
    """
    return "  TEST_STRING  "


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 1: GEOMETRÍA BIPARTITA Y CONSERVACIÓN DE ENERGÍA
#   Valida: Q ⪰ 0, P ⪰ 0, V ⪰ 0 y ||V - (Q ⊙ P)||_∞ ≤ ε
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase1_BipartiteGeometryCertifier:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 1: GEOMETRÍA BIPARTITA Y CONSERVACIÓN DE ENERGÍA                                ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida los invariantes topológicos de la variedad financiera.  ║
    ║  Cada método prueba un axioma específico del §1 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.1. Pruebas de Tolerancia Adaptativa (Método: _adaptive_tolerance)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_adaptive_tolerance_with_ndarray_reference(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia adaptativa con referencia de tipo NDArray.
        VALIDA: El cálculo de escala mediante norma L∞ del vector.
        """
        reference = np.array([1e6, 2e6, 3e6], dtype=np.float64)
        tolerance = phase1_certifier._adaptive_tolerance(
            base_tolerance=_NONNEGATIVITY_TOLERANCE,
            reference=reference,
        )
        assert isinstance(tolerance, float)
        assert tolerance >= _NONNEGATIVITY_TOLERANCE
        assert np.isfinite(tolerance)

    def test_adaptive_tolerance_with_scalar_reference(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia adaptativa con referencia escalar.
        VALIDA: El manejo correcto de escalares float.
        """
        reference = 1e9
        tolerance = phase1_certifier._adaptive_tolerance(
            base_tolerance=_NONNEGATIVITY_TOLERANCE,
            reference=reference,
        )
        assert isinstance(tolerance, float)
        assert tolerance >= _NONNEGATIVITY_TOLERANCE

    def test_adaptive_tolerance_with_empty_array(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia adaptativa con array vacío (caso borde).
        VALIDA: El manejo defensivo de arrays de tamaño cero.
        """
        reference = np.array([], dtype=np.float64)
        tolerance = phase1_certifier._adaptive_tolerance(
            base_tolerance=_NONNEGATIVITY_TOLERANCE,
            reference=reference,
        )
        assert isinstance(tolerance, float)
        assert np.isfinite(tolerance)

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.2. Pruebas de Coerción de Escalares (Método: _coerce_finite_scalar)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_finite_scalar_valid(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Coerción de escalar finito válido.
        VALIDA: Conversión correcta a float64.
        """
        result = phase1_certifier._coerce_finite_scalar("test_scalar", 42.0)
        assert isinstance(result, float)
        assert result == 42.0
        assert np.isfinite(result)

    def test_coerce_finite_scalar_from_int(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Coerción de entero a float.
        VALIDA: Conversión implícita de tipos numéricos.
        """
        result = phase1_certifier._coerce_finite_scalar("test_int", 100)
        assert isinstance(result, float)
        assert result == 100.0

    def test_coerce_finite_scalar_nan_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Coerción de NaN debe lanzar DomainIntegrityViolationError.
        VALIDA: §1. No-degeneración de la Variedad Financiera.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_scalar("nan_scalar", np.nan)
        assert "no es finito" in str(exc_info.value).lower()

    def test_coerce_finite_scalar_inf_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Coerción de infinito debe lanzar DomainIntegrityViolationError.
        VALIDA: Protección contra singularidades numéricas.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_scalar("inf_scalar", np.inf)
        assert "no es finito" in str(exc_info.value).lower()

    def test_coerce_finite_scalar_invalid_type_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Coerción de tipo no convertible debe lanzar excepción.
        VALIDA: Integridad del dominio ontológico.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_scalar("invalid_scalar", {"key": "value"})
        assert "no puede materializarse" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.3. Pruebas de Coerción de Vectores (Método: _canonicalize_finite_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_canonicalize_finite_vector_valid(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Canonicalización de vector válido.
        VALIDA: Conversión a NDArray[np.float64] unidimensional.
        """
        input_data = [1.0, 2.0, 3.0]
        result = phase1_certifier._canonicalize_finite_vector("test_vector", input_data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.ndim == 1
        assert result.size == 3
        assert np.allclose(result, np.array(input_data, dtype=np.float64))

    def test_canonicalize_finite_vector_from_numpy(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Canonicalización desde array NumPy existente.
        VALIDA: Reshape correcto a vector 1D.
        """
        input_data = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        result = phase1_certifier._canonicalize_finite_vector("test_vector", input_data)
        assert result.ndim == 1
        assert result.size == 3

    def test_canonicalize_finite_vector_scalar_reshape(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Escalar debe convertirse en vector de tamaño 1.
        VALIDA: Manejo de caso degenerate ndim=0.
        """
        input_data = np.array(42.0, dtype=np.float64)
        result = phase1_certifier._canonicalize_finite_vector("test_scalar_vector", input_data)
        assert result.ndim == 1
        assert result.size == 1
        assert result[0] == 42.0

    def test_canonicalize_finite_vector_empty_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Vector vacío debe lanzar DomainIntegrityViolationError.
        VALIDA: §1. No-degeneración de la Variedad Financiera.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._canonicalize_finite_vector("empty_vector", np.array([], dtype=np.float64))
        assert "vacío" in str(exc_info.value).lower()

    def test_canonicalize_finite_vector_nan_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Vector con NaN debe lanzar DomainIntegrityViolationError.
        VALIDA: Integridad de la variedad de datos.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._canonicalize_finite_vector("nan_vector", np.array([1.0, np.nan, 3.0]))
        assert "no finitas" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.4. Pruebas de Certificado de Dominio (Método: _certify_array_domain)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_array_domain_valid(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Certificado de dominio vectorial válido.
        VALIDA: Cálculo correcto de normas L¹, L², L∞.
        """
        arr = np.array([3.0, 4.0], dtype=np.float64)
        cert = phase1_certifier._certify_array_domain("test_array", arr)
        assert isinstance(cert, ArrayDomainCertificate)
        assert cert.name == "test_array"
        assert cert.size == 2
        assert cert.l1_norm == 7.0  # |3| + |4|
        assert np.isclose(cert.l2_norm, 5.0)  # sqrt(9 + 16)
        assert cert.linf_norm == 4.0  # max(|3|, |4|)
        assert cert.is_finite is True

    def test_array_domain_certificate_immutability(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: ArrayDomainCertificate es inmutable (frozen dataclass).
        VALIDA: Integridad estructural del DTO.
        """
        arr = np.array([1.0, 2.0], dtype=np.float64)
        cert = phase1_certifier._certify_array_domain("test_array", arr)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cert.size = 999  # type: ignore

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.5. Pruebas de No-Negatividad (Método: _certify_nonnegative_energy_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_nonnegative_energy_vector_positive(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Vector positivo pasa validación sin modificaciones.
        VALIDA: §1. Positividad Estricta.
        """
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = phase1_certifier._certify_nonnegative_energy_vector("positive_vector", arr)
        assert np.allclose(result, arr)

    def test_certify_nonnegative_energy_vector_zero(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Vector con ceros es válido.
        VALIDA: Frontera inferior del ortante positivo.
        """
        arr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        result = phase1_certifier._certify_nonnegative_energy_vector("zero_vector", arr)
        assert np.allclose(result, arr)

    def test_certify_nonnegative_energy_vector_small_negative_projected(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Negativo infinitesimal se proyecta a cero.
        VALIDA: Tolerancia de ruido numérico IEEE 754.
        """
        arr = np.array([-1e-15, 1.0, 2.0], dtype=np.float64)
        result = phase1_certifier._certify_nonnegative_energy_vector("small_negative", arr)
        assert result[0] == 0.0  # Proyectado a cero
        assert np.allclose(result[1:], arr[1:])

    def test_certify_nonnegative_energy_vector_large_negative_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Negativo significativo lanza BipartiteDegeneracyError.
        VALIDA: §1. No-degeneración (antimateria económica).
        """
        arr = np.array([-1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(BipartiteDegeneracyError) as exc_info:
            phase1_certifier._certify_nonnegative_energy_vector("negative_vector", arr)
        assert "antimateria económica" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.6. Pruebas de Tolerancia Dinámica (Método: _compute_dynamic_conservation_tolerance)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_compute_dynamic_conservation_tolerance_valid(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Cálculo de tolerancia dinámica híbrida.
        VALIDA: ε = ε_abs + ε_rel · escala + κ · ε_máquina · n · escala
        """
        V = np.array([1e6, 2e6, 3e6], dtype=np.float64)
        QP = np.array([1e6, 2e6, 3e6], dtype=np.float64)
        tolerance = phase1_certifier._compute_dynamic_conservation_tolerance(V, QP)
        assert isinstance(tolerance, float)
        assert tolerance >= _EPSILON_ABS
        assert np.isfinite(tolerance)

    def test_compute_dynamic_conservation_tolerance_empty_arrays(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia con arrays vacíos (caso defensivo).
        VALIDA: Manejo de tamaño cero.
        """
        V = np.array([], dtype=np.float64)
        QP = np.array([], dtype=np.float64)
        tolerance = phase1_certifier._compute_dynamic_conservation_tolerance(V, QP)
        assert isinstance(tolerance, float)
        assert np.isfinite(tolerance)

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.7. Pruebas de Certificación Bipartita Completa (Método: _certify_bipartite_arrays)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_bipartite_arrays_valid_conservation(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """
        PRUEBA: Certificación bipartita con conservación exacta V = Q ⊙ P.
        VALIDA: §2. Conservación de la Energía Financiera.
        """
        V, Q, P = valid_arrays
        result = phase1_certifier._certify_bipartite_arrays(V, Q, P)
        assert len(result) == 7  # V, Q, P, geometry_audit, V_domain, Q_domain, P_domain
        V_cert, Q_cert, P_cert, geometry_audit, V_domain, Q_domain, P_domain = result
        assert geometry_audit.is_energy_conserved is True
        assert geometry_audit.max_residual_error < geometry_audit.dynamic_tolerance

    def test_certify_bipartite_arrays_dimension_mismatch_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Dimensiones incompatibles lanzan BipartiteDegeneracyError.
        VALIDA: Igualdad dimensional para producto de Hadamard.
        """
        V = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        Q = np.array([1.0, 2.0], dtype=np.float64)  # Dimensión diferente
        P = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(BipartiteDegeneracyError) as exc_info:
            phase1_certifier._certify_bipartite_arrays(V, Q, P)
        assert "geometría bipartita incompatible" in str(exc_info.value).lower()

    def test_certify_bipartite_arrays_conservation_violation_raises(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Violación de conservación V ≠ Q ⊙ P lanza excepción.
        VALIDA: §2. Conservación de la Energía Financiera.
        """
        Q = np.array([100.0, 200.0], dtype=np.float64)
        P = np.array([10.0, 20.0], dtype=np.float64)
        V = np.array([5000.0, 5000.0], dtype=np.float64)  # V ≠ Q ⊙ P
        with pytest.raises(BipartiteDegeneracyError) as exc_info:
            phase1_certifier._certify_bipartite_arrays(V, Q, P)
        assert "fractura en la conservación" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.8. Pruebas de Handoff Fase 1 → Fase 2 (Método: _phase1_certify_and_handoff_to_phase2)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase1_certify_and_handoff_to_phase2_valid(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 1 a Fase 2.
        VALIDA: Continuidad funtorial Φ₁ → Φ₂.
        """
        V, Q, P = valid_arrays
        handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(V, Q, P)
        assert isinstance(handoff, Phase1GeometryHandoff)
        assert isinstance(handoff.geometry_audit, BipartiteGeometryData)
        assert isinstance(handoff.V_certified, np.ndarray)
        assert isinstance(handoff.Q_certified, np.ndarray)
        assert isinstance(handoff.P_certified, np.ndarray)
        assert isinstance(handoff.V_domain, ArrayDomainCertificate)
        assert isinstance(handoff.Q_domain, ArrayDomainCertificate)
        assert isinstance(handoff.P_domain, ArrayDomainCertificate)
        # Validación de handoff: datos certificados deben ser finitos
        assert np.all(np.isfinite(handoff.V_certified))
        assert np.all(np.isfinite(handoff.Q_certified))
        assert np.all(np.isfinite(handoff.P_certified))

    def test_phase1_geometry_handoff_immutability(
        self,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """
        PRUEBA: Phase1GeometryHandoff es inmutable.
        VALIDA: Integridad del artefacto de handoff.
        """
        V, Q, P = valid_arrays
        handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(V, Q, P)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            handoff.geometry_audit = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 2: RETRACTOS DE DEFORMACIÓN Y SATURACIÓN DIMENSIONAL
#   Valida: Hipercubo físico y f(f(x)) = f(x)
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase2_DimensionalSaturationEnforcer:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 2: RETRACTOS DE DEFORMACIÓN Y SATURACIÓN DIMENSIONAL                            ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida los límites físicos y la idempotencia de operadores.    ║
    ║  Cada método prueba un axioma específico del §3 y §4 del módulo principal.            ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.1. Pruebas de Acotación Escalar (Método: _certify_scalar_physical_bounds)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_scalar_physical_bounds_valid(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Escalares dentro del hipercubo físico.
        VALIDA: Q ∈ [0, 10^6], P ∈ [0, 10^9], Rend ∈ [0, 10^3]
        """
        Q, P, Rend = phase2_enforcer._certify_scalar_physical_bounds(
            Q_val=500.0,
            P_val=1e6,
            Rend_val=100.0,
        )
        assert Q == 500.0
        assert P == 1e6
        assert Rend == 100.0

    def test_certify_scalar_physical_bounds_at_limits(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Escalares en los límites exactos del hipercubo.
        VALIDA: Fronteras inclusivas del dominio físico.
        """
        Q, P, Rend = phase2_enforcer._certify_scalar_physical_bounds(
            Q_val=_MAX_Q,
            P_val=_MAX_P,
            Rend_val=_MAX_REND,
        )
        assert Q == _MAX_Q
        assert P == _MAX_P
        assert Rend == _MAX_REND

    def test_certify_scalar_physical_bounds_zero(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Escalares en cero (frontera inferior).
        VALIDA: Límite inferior del ortante positivo.
        """
        Q, P, Rend = phase2_enforcer._certify_scalar_physical_bounds(
            Q_val=0.0,
            P_val=0.0,
            Rend_val=0.0,
        )
        assert Q == 0.0
        assert P == 0.0
        assert Rend == 0.0

    def test_certify_scalar_physical_bounds_q_exceeded_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Q excede _MAX_Q lanza DimensionalSaturationError.
        VALIDA: §3. Saturación Dimensional Física.
        """
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_scalar_physical_bounds(
                Q_val=_MAX_Q + 1e6,  # Excede límite logístico
                P_val=1e6,
                Rend_val=100.0,
            )
        assert "escapa del límite logístico" in str(exc_info.value).lower()

    def test_certify_scalar_physical_bounds_p_exceeded_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: P excede _MAX_P lanza DimensionalSaturationError.
        VALIDA: Límite de capitalización.
        """
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_scalar_physical_bounds(
                Q_val=500.0,
                P_val=_MAX_P + 1e9,  # Excede límite de capitalización
                Rend_val=100.0,
            )
        assert "escapa del límite de capitalización" in str(exc_info.value).lower()

    def test_certify_scalar_physical_bounds_rend_exceeded_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Rend excede _MAX_REND lanza DimensionalSaturationError.
        VALIDA: Límite termodinámico del trabajo.
        """
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_scalar_physical_bounds(
                Q_val=500.0,
                P_val=1e6,
                Rend_val=_MAX_REND + 1e3,  # Excede límite termodinámico
            )
        assert "escapa del límite termodinámico" in str(exc_info.value).lower()

    def test_certify_scalar_physical_bounds_negative_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Escalares negativos lanzan DimensionalSaturationError.
        VALIDA: Positividad estricta.
        """
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_scalar_physical_bounds(
                Q_val=-1.0,
                P_val=1e6,
                Rend_val=100.0,
            )
        assert "escapa del límite" in str(exc_info.value).lower()

    def test_certify_scalar_physical_bounds_small_negative_projected(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Negativo infinitesimal se proyecta a cero.
        VALIDA: Tolerancia de ruido numérico en fronteras.
        """
        Q, P, Rend = phase2_enforcer._certify_scalar_physical_bounds(
            Q_val=-1e-15,
            P_val=-1e-15,
            Rend_val=-1e-15,
        )
        assert Q == 0.0
        assert P == 0.0
        assert Rend == 0.0

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.2. Pruebas de Acotación Vectorial (Método: _certify_vector_physical_bounds)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_vector_physical_bounds_valid(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Vector dentro de cotas físicas.
        VALIDA: Todas las componentes en [0, upper_bound].
        """
        arr = np.array([1.0, 500.0, 999999.0], dtype=np.float64)
        phase2_enforcer._certify_vector_physical_bounds("test_vector", arr, _MAX_Q)
        # No debe lanzar excepción

    def test_certify_vector_physical_bounds_negative_component_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Componente negativa en vector lanza excepción.
        VALIDA: Hipercubo físico.
        """
        arr = np.array([1.0, -500.0, 3.0], dtype=np.float64)
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_vector_physical_bounds("test_vector", arr, _MAX_Q)
        assert "componentes negativas" in str(exc_info.value).lower()

    def test_certify_vector_physical_bounds_exceeds_bound_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Componente excede cota superior lanza excepción.
        VALIDA: Saturación dimensional.
        """
        arr = np.array([1.0, _MAX_Q + 1e6, 3.0], dtype=np.float64)
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_vector_physical_bounds("test_vector", arr, _MAX_Q)
        assert "excede la cota física" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.3. Pruebas de Idempotencia (Método: _safe_idempotence_equality)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_safe_idempotence_equality_scalars_equal(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Comparación segura de escalares iguales.
        VALIDA: Función de comparación robusta.
        """
        result = phase2_enforcer._safe_idempotence_equality(42, 42)
        assert result is True

    def test_safe_idempotence_equality_scalars_different(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Comparación segura de escalares diferentes.
        VALIDA: Detección de desigualdad.
        """
        result = phase2_enforcer._safe_idempotence_equality(42, 43)
        assert result is False

    def test_safe_idempotence_equality_arrays_equal(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Comparación segura de arrays iguales.
        VALIDA: np.array_equal para NDArray.
        """
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = phase2_enforcer._safe_idempotence_equality(arr1, arr2)
        assert result is True

    def test_safe_idempotence_equality_arrays_different(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Comparación segura de arrays diferentes.
        VALIDA: Detección de diferencia en NDArray.
        """
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        arr2 = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        result = phase2_enforcer._safe_idempotence_equality(arr1, arr2)
        assert result is False

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.4. Pruebas de Idempotencia del Normalizador (Método: _certify_normalizer_idempotence)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_normalizer_idempotence_valid(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: Normalizador idempotente f(f(x)) = f(x).
        VALIDA: §4. Retractos de Deformación e Idempotencia.
        """
        # No debe lanzar excepción
        phase2_enforcer._certify_normalizer_idempotence(idempotent_normalizer, test_string)

    def test_certify_normalizer_idempotence_non_callable_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        test_string: str,
    ) -> None:
        """
        PRUEBA: Normalizador no callable lanza DimensionalSaturationError.
        VALIDA: Validación de tipo del operador.
        """
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_normalizer_idempotence("not_callable", test_string)
        assert "debe ser un operador callable" in str(exc_info.value).lower()

    def test_certify_normalizer_idempotence_non_string_test_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        idempotent_normalizer: Callable[[Any], Any],
    ) -> None:
        """
        PRUEBA: test_string no es cadena lanza DomainIntegrityViolationError.
        VALIDA: Integridad del dominio de prueba.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_enforcer._certify_normalizer_idempotence(idempotent_normalizer, 123)
        assert "debe ser una cadena" in str(exc_info.value).lower()

    def test_certify_normalizer_idempotence_non_idempotent_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Normalizador no idempotente lanza DimensionalSaturationError.
        VALIDA: §4. f(f(x)) = f(x).
        """
        def non_idempotent(x: Any) -> str:
            return str(x) + "_modified"
        with pytest.raises(DimensionalSaturationError) as exc_info:
            phase2_enforcer._certify_normalizer_idempotence(non_idempotent, "test")
        assert "no es idempotente" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.5. Pruebas de Conjunto Categórico (Método: _coerce_categories_set)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_categories_set_valid_frozenset(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Conjunto de categorías válido (frozenset).
        VALIDA: Conversión y saneamiento correcto.
        """
        result = phase2_enforcer._coerce_categories_set(valid_categories)
        assert isinstance(result, frozenset)
        assert len(result) == len(valid_categories)
        assert all(isinstance(cat, str) for cat in result)
        assert all(len(cat) > 0 for cat in result)

    def test_coerce_categories_set_from_list(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Conversión de lista a frozenset.
        VALIDA: Flexibilidad de entrada.
        """
        categories_list = ["cat1", "cat2", "cat3"]
        result = phase2_enforcer._coerce_categories_set(categories_list)
        assert isinstance(result, frozenset)
        assert len(result) == 3

    def test_coerce_categories_set_from_set(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Conversión de set a frozenset.
        VALIDA: Inmutabilización del resultado.
        """
        categories_set = {"cat1", "cat2", "cat3"}
        result = phase2_enforcer._coerce_categories_set(categories_set)
        assert isinstance(result, frozenset)

    def test_coerce_categories_set_none_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: categories_set=None lanza StructuralThermodynamicError.
        VALIDA: §5. Estabilidad Termodinámica Estructural.
        """
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase2_enforcer._coerce_categories_set(None)
        assert "degeneración categórica" in str(exc_info.value).lower()

    def test_coerce_categories_set_string_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: categories_set como string plano lanza DomainIntegrityViolationError.
        VALIDA: Rechazo de iterables falsos.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_enforcer._coerce_categories_set("single_string")
        assert "no debe ser una cadena plana" in str(exc_info.value).lower()

    def test_coerce_categories_set_empty_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Conjunto vacío lanza StructuralThermodynamicError.
        VALIDA: D(a) > 0.
        """
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase2_enforcer._coerce_categories_set(set())
        assert "conjunto vacío" in str(exc_info.value).lower()

    def test_coerce_categories_set_with_empty_string_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Categoría con string vacío lanza StructuralThermodynamicError.
        VALIDA: Saneamiento de categorías.
        """
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase2_enforcer._coerce_categories_set({"cat1", "", "cat3"})
        assert "categoría vacía" in str(exc_info.value).lower()

    def test_coerce_categories_set_with_none_item_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Categoría None lanza StructuralThermodynamicError.
        VALIDA: Integridad de elementos del conjunto.
        """
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase2_enforcer._coerce_categories_set({"cat1", None, "cat3"})
        assert "categoría nula" in str(exc_info.value).lower()

    def test_coerce_categories_set_strips_whitespace(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
    ) -> None:
        """
        PRUEBA: Categorías con whitespace se limpian.
        VALIDA: Saneamiento de strings.
        """
        result = phase2_enforcer._coerce_categories_set({"  cat1  ", "cat2"})
        assert "cat1" in result
        assert "  cat1  " not in result

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.6. Pruebas de Handoff Fase 2 → Fase 3 (Método: _phase2_enforce_and_handoff_to_phase3)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase2_enforce_and_handoff_to_phase3_valid(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        valid_categories: frozenset[str],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 2 a Fase 3.
        VALIDA: Continuidad funtorial Φ₂ → Φ₃.
        """
        V, Q, P = valid_arrays
        phase1_handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(V, Q, P)
        handoff = phase2_enforcer._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            Rend_val=100.0,
            normalizer_func=idempotent_normalizer,
            test_string=test_string,
            categories_set=valid_categories,
        )
        assert isinstance(handoff, Phase2SaturationHandoff)
        assert isinstance(handoff.phase1_handoff, Phase1GeometryHandoff)
        assert isinstance(handoff.saturation_audit, DimensionalSaturationData)
        assert isinstance(handoff.categories_certified, frozenset)
        assert handoff.saturation_audit.is_idempotent is True
        assert handoff.saturation_audit.is_physically_bounded is True

    def test_phase2_handoff_invalid_phase1_handoff_raises(
        self,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        valid_categories: frozenset[str],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: Handoff de Fase 1 inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_enforcer._phase2_enforce_and_handoff_to_phase3(
                phase1_handoff=None,  # type: ignore
                Rend_val=100.0,
                normalizer_func=idempotent_normalizer,
                test_string=test_string,
                categories_set=valid_categories,
            )
        assert "exige un phase1geometryhandoff" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 3: AUDITORÍA DE TERMODINÁMICA ESTRUCTURAL
#   Valida: H_norm ≥ 0.1 y D(a) > 0
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase3_StructuralThermodynamicAuditor:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 3: AUDITORÍA DE TERMODINÁMICA ESTRUCTURAL                                       ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la estabilidad termodinámica y diversidad categórica.   ║
    ║  Cada método prueba un axioma específico del §5 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.1. Pruebas de Entropía de Shannon (Método: _compute_shannon_entropy_normalized)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_compute_shannon_entropy_normalized_uniform_distribution(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Entropía normalizada con distribución uniforme (máxima entropía).
        VALIDA: H_norm ≈ 1.0 para distribución uniforme.
        """
        V = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        H_norm = phase3_auditor._compute_shannon_entropy_normalized(V)
        assert isinstance(H_norm, float)
        assert np.isclose(H_norm, 1.0, atol=1e-6)  # Máxima entropía
        assert 0.0 <= H_norm <= 1.0

    def test_compute_shannon_entropy_normalized_non_uniform(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Entropía normalizada con distribución no uniforme.
        VALIDA: H_norm < 1.0 pero ≥ 0.1.
        """
        V = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        H_norm = phase3_auditor._compute_shannon_entropy_normalized(V)
        assert isinstance(H_norm, float)
        assert H_norm >= _ENTROPY_MIN_THRESHOLD
        assert H_norm < 1.0

    def test_compute_shannon_entropy_normalized_single_element_raises(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Vector de un elemento lanza StructuralThermodynamicError (SPOF).
        VALIDA: §5. Estabilidad Termodinámica (anti-SPOF).
        """
        V = np.array([100.0], dtype=np.float64)
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase3_auditor._compute_shannon_entropy_normalized(V)
        assert "spof" in str(exc_info.value).lower()

    def test_compute_shannon_entropy_normalized_empty_raises(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Vector vacío lanza StructuralThermodynamicError.
        VALIDA: Vacío topológico.
        """
        V = np.array([], dtype=np.float64)
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase3_auditor._compute_shannon_entropy_normalized(V)
        assert "vacío topológico" in str(exc_info.value).lower()

    def test_compute_shannon_entropy_normalized_zero_total_raises(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Energía total cero lanza StructuralThermodynamicError.
        VALIDA: Variedad termodinámica degenerada.
        """
        V = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase3_auditor._compute_shannon_entropy_normalized(V)
        assert "energía financiera total nula" in str(exc_info.value).lower()

    def test_compute_shannon_entropy_normalized_low_entropy_raises(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Entropía por debajo del umbral lanza StructuralThermodynamicError.
        VALIDA: §5. Pirámide Invertida (H_norm < 0.1).
        """
        # Distribución altamente concentrada (pirámide invertida)
        V = np.array([1e10, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase3_auditor._compute_shannon_entropy_normalized(V)
        assert "pirámide invertida" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.2. Pruebas de Diversidad Categórica (Método: _compute_categorical_diversity)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_compute_categorical_diversity_valid(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Diversidad categórica válida D(a) > 0.
        VALIDA: §5. Diversidad Categórica.
        """
        D_a = phase3_auditor._compute_categorical_diversity(valid_categories)
        assert isinstance(D_a, float)
        assert D_a > 0
        assert np.isfinite(D_a)
        # D(a) = |categorías| / 5
        expected = len(valid_categories) / _CATEGORY_CARDINALITY_REFERENCE
        assert np.isclose(D_a, expected)

    def test_compute_categorical_diversity_single_category_valid(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Una categoría es válida (D(a) = 0.2).
        VALIDA: Mínimo de diversidad aceptable.
        """
        categories = frozenset(["single_category"])
        D_a = phase3_auditor._compute_categorical_diversity(categories)
        assert np.isclose(D_a, 0.2)  # 1 / 5

    def test_compute_categorical_diversity_empty_raises(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Conjunto vacío lanza StructuralThermodynamicError.
        VALIDA: D(a) > 0 (monotipo logístico).
        """
        with pytest.raises(StructuralThermodynamicError) as exc_info:
            phase3_auditor._compute_categorical_diversity(frozenset())
        assert "monotipo logístico" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.3. Pruebas de Auditoría Completa (Método: _audit_structural_entropy_and_diversity)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_audit_structural_entropy_and_diversity_valid(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Auditoría termodinámica completa válida.
        VALIDA: H_norm ≥ 0.1 y D(a) > 0.
        """
        V, Q, P = valid_arrays
        audit = phase3_auditor._audit_structural_entropy_and_diversity(V, valid_categories)
        assert isinstance(audit, StructuralThermodynamicsData)
        assert audit.shannon_entropy_norm >= _ENTROPY_MIN_THRESHOLD
        assert audit.categorical_diversity > 0
        assert audit.entropy_threshold == _ENTROPY_MIN_THRESHOLD
        assert audit.is_thermodynamically_stable is True

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.4. Pruebas de Finalización Funtorial (Método: _phase3_finalize_from_phase2_handoff)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase3_finalize_from_phase2_handoff_valid(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
        phase2_enforcer: Phase2_DimensionalSaturationEnforcer,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        valid_categories: frozenset[str],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: Finalización funtorial completa Φ₃ ∘ Φ₂ ∘ Φ₁.
        VALIDA: Composición de las tres fases.
        """
        V, Q, P = valid_arrays
        phase1_handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(V, Q, P)
        phase2_handoff = phase2_enforcer._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            Rend_val=100.0,
            normalizer_func=idempotent_normalizer,
            test_string=test_string,
            categories_set=valid_categories,
        )
        state = phase3_auditor._phase3_finalize_from_phase2_handoff(phase2_handoff)
        assert isinstance(state, StructuralInvariantState)
        assert isinstance(state.geometry_audit, BipartiteGeometryData)
        assert isinstance(state.saturation_audit, DimensionalSaturationData)
        assert isinstance(state.thermo_audit, StructuralThermodynamicsData)
        assert state.is_epistemologically_valid is True

    def test_phase3_handoff_invalid_phase2_handoff_raises(
        self,
        phase3_auditor: Phase3_StructuralThermodynamicAuditor,
    ) -> None:
        """
        PRUEBA: Handoff de Fase 2 inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_auditor._phase3_finalize_from_phase2_handoff(
                phase2_handoff=None  # type: ignore
            )
        assert "exige un phase2saturationhandoff" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   ORQUESTADOR SUPREMO: SCHEMASAGENT (Pruebas de Integración)
#   Valida: Endofuntor Z_Schemas = Φ₃ ∘ Φ₂ ∘ Φ₁
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestSchemasAgent_Integration:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  ORQUESTADOR SUPREMO: SCHEMASAGENT                                                    ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Pruebas de integración que validan el endofuntor completo Z_Schemas.                 ║
    ║  Estas pruebas aseguran que la composición Φ₃ ∘ Φ₂ ∘ Φ₁ funciona correctamente.       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_schemas_agent_execute_structural_invariant_governance_valid(
        self,
        schemas_agent: SchemasAgent,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        valid_categories: frozenset[str],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: Ejecución completa del gobierno de invariantes estructurales.
        VALIDA: Endofuntor Z_Schemas con datos válidos.
        """
        V, Q, P = valid_arrays
        state = schemas_agent.execute_structural_invariant_governance(
            V_array=V,
            Q_array=Q,
            P_array=P,
            Rend_val=100.0,
            normalizer_func=idempotent_normalizer,
            test_string=test_string,
            categories_set=valid_categories,
        )
        assert isinstance(state, StructuralInvariantState)
        assert state.is_epistemologically_valid is True
        assert state.geometry_audit.is_energy_conserved is True
        assert state.saturation_audit.is_physically_bounded is True
        assert state.saturation_audit.is_idempotent is True
        assert state.thermo_audit.is_thermodynamically_stable is True

    def test_schemas_agent_call_alias_valid(
        self,
        schemas_agent: SchemasAgent,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        valid_categories: frozenset[str],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: Alias invocable __call__ del endofuntor.
        VALIDA: Sintaxis alternativa de ejecución.
        """
        V, Q, P = valid_arrays
        state = schemas_agent(
            V_array=V,
            Q_array=Q,
            P_array=P,
            Rend_val=100.0,
            normalizer_func=idempotent_normalizer,
            test_string=test_string,
            categories_set=valid_categories,
        )
        assert isinstance(state, StructuralInvariantState)
        assert state.is_epistemologically_valid is True

    def test_schemas_agent_bipartite_degeneracy_error(
        self,
        schemas_agent: SchemasAgent,
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Violación de conservación energética lanza BipartiteDegeneracyError.
        VALIDA: Propagación de excepciones de Fase 1.
        """
        V = np.array([5000.0, 5000.0], dtype=np.float64)  # V ≠ Q ⊙ P
        Q = np.array([100.0, 200.0], dtype=np.float64)
        P = np.array([10.0, 20.0], dtype=np.float64)
        with pytest.raises(BipartiteDegeneracyError):
            schemas_agent(
                V_array=V,
                Q_array=Q,
                P_array=P,
                Rend_val=100.0,
                normalizer_func=idempotent_normalizer,
                test_string=test_string,
                categories_set=valid_categories,
            )

    def test_schemas_agent_dimensional_saturation_error(
        self,
        schemas_agent: SchemasAgent,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Saturación dimensional lanza DimensionalSaturationError.
        VALIDA: Propagación de excepciones de Fase 2.
        """
        V, Q, P = valid_arrays
        Q_exceeded = Q * 1e7  # Excede _MAX_Q
        with pytest.raises(DimensionalSaturationError):
            schemas_agent(
                V_array=V,
                Q_array=Q_exceeded,
                P_array=P,
                Rend_val=100.0,
                normalizer_func=idempotent_normalizer,
                test_string=test_string,
                categories_set=valid_categories,
            )

    def test_schemas_agent_structural_thermodynamic_error_spof(
        self,
        schemas_agent: SchemasAgent,
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: SPOF (un solo elemento) lanza StructuralThermodynamicError.
        VALIDA: Propagación de excepciones de Fase 3.
        """
        V = np.array([100.0], dtype=np.float64)  # SPOF
        Q = np.array([10.0], dtype=np.float64)
        P = np.array([10.0], dtype=np.float64)
        with pytest.raises(StructuralThermodynamicError):
            schemas_agent(
                V_array=V,
                Q_array=Q,
                P_array=P,
                Rend_val=100.0,
                normalizer_func=idempotent_normalizer,
                test_string=test_string,
                categories_set=valid_categories,
            )

    def test_schemas_agent_structural_thermodynamic_error_low_entropy(
        self,
        schemas_agent: SchemasAgent,
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Entropía baja (pirámide invertida) lanza StructuralThermodynamicError.
        VALIDA: §5. Estabilidad Termodinámica Estructural.
        """
        V = np.array([1e10, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)  # Pirámide invertida
        Q = np.array([1e5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        P = np.array([1e5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(StructuralThermodynamicError):
            schemas_agent(
                V_array=V,
                Q_array=Q,
                P_array=P,
                Rend_val=100.0,
                normalizer_func=idempotent_normalizer,
                test_string=test_string,
                categories_set=valid_categories,
            )

    def test_schemas_agent_domain_integrity_violation_error(
        self,
        schemas_agent: SchemasAgent,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        test_string: str,
        valid_categories: frozenset[str],
    ) -> None:
        """
        PRUEBA: Normalizador no callable lanza DomainIntegrityViolationError.
        VALIDA: Integridad del dominio ontológico.
        """
        V, Q, P = valid_arrays
        with pytest.raises(DimensionalSaturationError):  # DimensionalSaturationError para non-callable
            schemas_agent(
                V_array=V,
                Q_array=Q,
                P_array=P,
                Rend_val=100.0,
                normalizer_func="not_callable",  # type: ignore
                test_string=test_string,
                categories_set=valid_categories,
            )

    def test_schemas_agent_categories_none_raises(
        self,
        schemas_agent: SchemasAgent,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        idempotent_normalizer: Callable[[Any], Any],
        test_string: str,
    ) -> None:
        """
        PRUEBA: categories_set=None lanza StructuralThermodynamicError.
        VALIDA: Degeneración categórica.
        """
        V, Q, P = valid_arrays
        with pytest.raises(StructuralThermodynamicError):
            schemas_agent(
                V_array=V,
                Q_array=Q,
                P_array=P,
                Rend_val=100.0,
                normalizer_func=idempotent_normalizer,
                test_string=test_string,
                categories_set=None,
            )

    def test_schemas_agent_inheritance_chain(
        self,
        schemas_agent: SchemasAgent,
    ) -> None:
        """
        PRUEBA: Cadena de herencia del SchemasAgent.
        VALIDA: Arquitectura de fases anidadas.
        """
        assert isinstance(schemas_agent, SchemasAgent)
        assert isinstance(schemas_agent, Phase3_StructuralThermodynamicAuditor)
        assert isinstance(schemas_agent, Phase2_DimensionalSaturationEnforcer)
        assert isinstance(schemas_agent, Phase1_BipartiteGeometryCertifier)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Z. PRUEBAS DE ESTRUCTURAS DE DATOS (Data Classes)
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestDataStructures:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE ESTRUCTURAS DE DATOS INMUTABLES                                           ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida la integridad de todos los DTOs del espacio de fase.                          ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_array_domain_certificate_creation(self) -> None:
        """
        PRUEBA: Creación de ArrayDomainCertificate.
        VALIDA: Estructura inmutable del certificado.
        """
        cert = ArrayDomainCertificate(
            name="test",
            size=5,
            l1_norm=10.0,
            l2_norm=5.0,
            linf_norm=3.0,
            is_finite=True,
        )
        assert cert.name == "test"
        assert cert.size == 5
        assert cert.is_finite is True

    def test_bipartite_geometry_data_creation(self) -> None:
        """
        PRUEBA: Creación de BipartiteGeometryData.
        VALIDA: Artefacto de Fase 1.
        """
        audit = BipartiteGeometryData(
            max_residual_error=1e-10,
            dynamic_tolerance=1e-8,
            is_energy_conserved=True,
        )
        assert audit.is_energy_conserved is True
        assert audit.max_residual_error < audit.dynamic_tolerance

    def test_dimensional_saturation_data_creation(self) -> None:
        """
        PRUEBA: Creación de DimensionalSaturationData.
        VALIDA: Artefacto de Fase 2.
        """
        audit = DimensionalSaturationData(
            max_Q_observed=500.0,
            max_P_observed=1e6,
            Rend_val=100.0,
            is_idempotent=True,
            is_physically_bounded=True,
        )
        assert audit.is_idempotent is True
        assert audit.is_physically_bounded is True

    def test_structural_thermodynamics_data_creation(self) -> None:
        """
        PRUEBA: Creación de StructuralThermodynamicsData.
        VALIDA: Artefacto de Fase 3.
        """
        audit = StructuralThermodynamicsData(
            shannon_entropy_norm=0.5,
            categorical_diversity=0.6,
            entropy_threshold=_ENTROPY_MIN_THRESHOLD,
            is_thermodynamically_stable=True,
        )
        assert audit.is_thermodynamically_stable is True
        assert audit.shannon_entropy_norm >= _ENTROPY_MIN_THRESHOLD

    def test_structural_invariant_state_creation(
        self,
        valid_arrays: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        phase1_certifier: Phase1_BipartiteGeometryCertifier,
    ) -> None:
        """
        PRUEBA: Creación de StructuralInvariantState (objeto final).
        VALIDA: Estado epistemológico completo del endofuntor.
        """
        V, Q, P = valid_arrays
        handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(V, Q, P)
        geometry_audit = handoff.geometry_audit
        saturation_audit = DimensionalSaturationData(
            max_Q_observed=500.0,
            max_P_observed=1e6,
            Rend_val=100.0,
            is_idempotent=True,
            is_physically_bounded=True,
        )
        thermo_audit = StructuralThermodynamicsData(
            shannon_entropy_norm=0.5,
            categorical_diversity=0.6,
            entropy_threshold=_ENTROPY_MIN_THRESHOLD,
            is_thermodynamically_stable=True,
        )
        state = StructuralInvariantState(
            geometry_audit=geometry_audit,
            saturation_audit=saturation_audit,
            thermo_audit=thermo_audit,
            is_epistemologically_valid=True,
        )
        assert state.is_epistemologically_valid is True
        assert isinstance(state.geometry_audit, BipartiteGeometryData)
        assert isinstance(state.saturation_audit, DimensionalSaturationData)
        assert isinstance(state.thermo_audit, StructuralThermodynamicsData)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §∞. PRUEBAS DE CONSTANTES FÍSICO-MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhysicalMathematicalConstants:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE CONSTANTES FÍSICO-MATEMÁTICAS                                             ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida que las constantes del módulo tengan valores correctos y consistentes.        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_machine_epsilon_value(self) -> None:
        """
        PRUEBA: Valor de _MACHINE_EPSILON.
        VALIDA: Precisión de float64 de NumPy.
        """
        assert _MACHINE_EPSILON == float(np.finfo(np.float64).eps)
        assert _MACHINE_EPSILON > 0
        assert _MACHINE_EPSILON < 1e-15

    def test_epsilon_abs_value(self) -> None:
        """
        PRUEBA: Valor de _EPSILON_ABS.
        VALIDA: Tolerancia absoluta híbrida.
        """
        assert _EPSILON_ABS == 1e-10
        assert _EPSILON_ABS > 0

    def test_epsilon_rel_value(self) -> None:
        """
        PRUEBA: Valor de _EPSILON_REL.
        VALIDA: Tolerancia relativa híbrida.
        """
        assert _EPSILON_REL == 1e-6
        assert _EPSILON_REL > 0

    def test_max_q_value(self) -> None:
        """
        PRUEBA: Valor de _MAX_Q.
        VALIDA: Límite logístico.
        """
        assert _MAX_Q == 1e6
        assert _MAX_Q > 0

    def test_max_p_value(self) -> None:
        """
        PRUEBA: Valor de _MAX_P.
        VALIDA: Límite de capitalización.
        """
        assert _MAX_P == 1e9
        assert _MAX_P > 0

    def test_max_rend_value(self) -> None:
        """
        PRUEBA: Valor de _MAX_REND.
        VALIDA: Límite termodinámico del trabajo.
        """
        assert _MAX_REND == 1e3
        assert _MAX_REND > 0

    def test_entropy_min_threshold_value(self) -> None:
        """
        PRUEBA: Valor de _ENTROPY_MIN_THRESHOLD.
        VALIDA: Umbral anti-SPOF.
        """
        assert _ENTROPY_MIN_THRESHOLD == 0.1
        assert 0.0 <= _ENTROPY_MIN_THRESHOLD <= 1.0

    def test_category_cardinality_reference_value(self) -> None:
        """
        PRUEBA: Valor de _CATEGORY_CARDINALITY_REFERENCE.
        VALIDA: Cardinalidad de referencia para diversidad.
        """
        assert _CATEGORY_CARDINALITY_REFERENCE == 5.0
        assert _CATEGORY_CARDINALITY_REFERENCE > 0


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Ω. EJECUCIÓN DIRECTA (Para debugging)
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución directa para debugging fuera de pytest.
    Uso: python tests/unit/agents/core/test_schemas_agent.py
    """
    import sys
    import os
    
    # Agregar el directorio raíz al path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    
    pytest.main([__file__, "-v", "--tb=short"])