# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Telemetry Schemas Agent (Suite de Validación de Espacio de Fase Tensorial)║
║  Ruta   : tests/unit/agents/core/test_telemetry_schemas_agent.py                         ║
║  Versión: 2.0.0-Tensorial-Orthogonal-Fixpoint-Doctoral-Strict-Nested                     ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ARQUITECTURA DE PRUEBAS (Composición Funtorial Φ₃ ∘ Φ₂ ∘ Φ₁):                           ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Este módulo de pruebas implementa una batería exhaustiva que valida la integridad       ║
║  Riemanniana, ortogonal y diferencial del endofuntor Z_Tensorial.                        ║
║                                                                                          ║
║  FASE 1 → Certificación de Variedad Riemanniana                                          ║
║           Valida: G = Gᵀ, G ≻ 0, κ(G) ≤ 10¹²                                             ║
║                                                                                          ║
║  FASE 2 → Descomposición Ortogonal de Subespacios                                        ║
║           Valida: ⟨v_i, v_j⟩_G = δ_ij y Matriz de Gram bien condicionada                 ║
║                                                                                          ║
║  FASE 3 → Imposición de Inmutabilidad y Punto Fijo                                       ║
║           Valida: ∇_τ Ψ = 0 (derivada covariante nula)                                   ║
║                                                                                          ║
║  COBERTURA DE EXCEPCIONES TENSORIALES:                                                   ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  • DomainIntegrityViolationError        → Violaciones de dominio ontológico              ║
║  • MetricManifoldDegeneracyError        → Métrica no PSD o mal condicionada              ║
║  • NonOrthogonalSubspaceError           → Covarianza espuria entre subespacios           ║
║  • PhaseSpaceCorruptionError            → Mutación temporal del tensor (∇_τ Ψ > 0)       ║
║                                                                                          ║
║  EJECUCIÓN:                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  $ pytest tests/unit/agents/core/test_telemetry_schemas_agent.py -v                      ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIONES Y CONFIGURACIÓN DEL ENTORNO DE PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════════════════════

import pytest
import numpy as np
import math
from typing import Any, Optional
from numpy.typing import NDArray

# Importación del módulo bajo prueba
from app.agents.core.telemetry_schemas_agent import (
    # Excepciones Tensoriales
    TelemetrySchemasAgentError,
    DomainIntegrityViolationError,
    MetricManifoldDegeneracyError,
    NonOrthogonalSubspaceError,
    PhaseSpaceCorruptionError,
    # Estructuras Inmutables (DTOs)
    MetricManifoldData,
    OrthogonalDecompositionData,
    FixpointVerificationData,
    Phase1MetricHandoff,
    Phase2OrthogonalityHandoff,
    TensorialPhaseSpaceState,
    # Fases Anidadas
    Phase1_RiemannianMetricCertifier,
    Phase2_OrthogonalDecompositionCertifier,
    Phase3_TensorImmutabilityEnforcer,
    # Orquestador Supremo
    TelemetrySchemasAgent,
    # Constantes Matemáticas y de Tolerancia
    _MACHINE_EPSILON,
    _ORTHOGONALITY_TOLERANCE,
    _FIXPOINT_TOLERANCE,
    _METRIC_SYMMETRY_TOLERANCE,
    _MAX_METRIC_CONDITION_NUMBER,
    _SUBSPACE_NORM_TOLERANCE,
    # Métrica por defecto
    G_PHYSICS,
)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §A. FIXTURES Y UTILITARIOS DE PRUEBA (Infraestructura Categórica)
# ═══════════════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def schemas_agent() -> TelemetrySchemasAgent:
    """
    Fixture: Instancia del Arquitecto del Espacio de Fase Tensorial.
    Retorna el endofuntor completo para pruebas de integración.
    """
    return TelemetrySchemasAgent()


@pytest.fixture
def phase1_certifier() -> Phase1_RiemannianMetricCertifier:
    """
    Fixture: Instancia de Phase1_RiemannianMetricCertifier.
    Para pruebas unitarias de la Fase 1.
    """
    return Phase1_RiemannianMetricCertifier()


@pytest.fixture
def phase2_certifier() -> Phase2_OrthogonalDecompositionCertifier:
    """
    Fixture: Instancia de Phase2_OrthogonalDecompositionCertifier.
    Para pruebas unitarias de la Fase 2.
    """
    return Phase2_OrthogonalDecompositionCertifier()


@pytest.fixture
def phase3_enforcer() -> Phase3_TensorImmutabilityEnforcer:
    """
    Fixture: Instancia de Phase3_TensorImmutabilityEnforcer.
    Para pruebas unitarias de la Fase 3.
    """
    return Phase3_TensorImmutabilityEnforcer()


@pytest.fixture
def valid_metric_4x4() -> NDArray[np.float64]:
    """
    Fixture: Matriz métrica G válida (4x4, simétrica, definida positiva).
    """
    # Matriz simétrica definida positiva
    A = np.array([
        [2.0, 0.1, 0.0, 0.0],
        [0.1, 2.0, 0.1, 0.0],
        [0.0, 0.1, 2.0, 0.1],
        [0.0, 0.0, 0.1, 2.0],
    ], dtype=np.float64)
    return A @ A.T  # Garantiza simetría y PSD


@pytest.fixture
def valid_subspace_vectors_4d() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Fixture: Cuatro vectores de subespacio válidos (dimensión 4).
    Orthogonales entre sí en espacio Euclidiano.
    """
    v_physics = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    v_topology = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    v_control = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    v_thermo = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return v_physics, v_topology, v_control, v_thermo


@pytest.fixture
def valid_state_vectors_4d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fixture: Dos vectores de estado idénticos (punto fijo válido).
    Ψ(t₀) = Ψ(t₁) para ∇_τ Ψ = 0.
    """
    psi = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    return psi.copy(), psi.copy()


@pytest.fixture
def identity_matrix_4x4() -> NDArray[np.float64]:
    """
    Fixture: Matriz identidad 4x4 (métrica Euclidiana trivial).
    """
    return np.eye(4, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 1: CERTIFICACIÓN DE VARIEDAD RIEMANNIANA
#   Valida: G = Gᵀ, G ≻ 0, κ(G) ≤ 10¹²
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase1_RiemannianMetricCertifier:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 1: CERTIFICACIÓN DE VARIEDAD RIEMANNIANA                                        ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la geometría Riemanniana del tensor métrico G.          ║
    ║  Cada método prueba un axioma específico del §1 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.1. Pruebas de Tolerancia Adaptativa (Método: _adaptive_tolerance)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_adaptive_tolerance_with_ndarray_reference(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia adaptativa con referencia de tipo NDArray.
        VALIDA: El cálculo de escala mediante norma L∞ del vector.
        """
        reference = np.array([1e6, 2e6, 3e6, 4e6], dtype=np.float64)
        tolerance = phase1_certifier._adaptive_tolerance(
            base_tolerance=_SUBSPACE_NORM_TOLERANCE,
            reference=reference,
        )
        assert isinstance(tolerance, float)
        assert tolerance >= _SUBSPACE_NORM_TOLERANCE
        assert np.isfinite(tolerance)

    def test_adaptive_tolerance_with_scalar_reference(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia adaptativa con referencia escalar.
        VALIDA: El manejo correcto de escalares float.
        """
        reference = 1e9
        tolerance = phase1_certifier._adaptive_tolerance(
            base_tolerance=_SUBSPACE_NORM_TOLERANCE,
            reference=reference,
        )
        assert isinstance(tolerance, float)
        assert tolerance >= _SUBSPACE_NORM_TOLERANCE

    def test_adaptive_tolerance_with_empty_array(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Tolerancia adaptativa con array vacío (caso borde).
        VALIDA: El manejo defensivo de arrays de tamaño cero.
        """
        reference = np.array([], dtype=np.float64)
        tolerance = phase1_certifier._adaptive_tolerance(
            base_tolerance=_SUBSPACE_NORM_TOLERANCE,
            reference=reference,
        )
        assert isinstance(tolerance, float)
        assert np.isfinite(tolerance)

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.2. Pruebas de Coerción de Vectores (Método: _coerce_finite_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_finite_vector_valid(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Coerción de vector válido.
        VALIDA: Conversión a NDArray[np.float64] unidimensional.
        """
        input_data = [1.0, 2.0, 3.0, 4.0]
        result = phase1_certifier._coerce_finite_vector("test_vector", input_data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.ndim == 1
        assert result.size == 4
        assert np.all(np.isfinite(result))

    def test_coerce_finite_vector_with_expected_dim(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Coerción con dimensión esperada válida.
        VALIDA: Verificación de dimensionalidad.
        """
        input_data = [1.0, 2.0, 3.0, 4.0]
        result = phase1_certifier._coerce_finite_vector(
            "test_vector",
            input_data,
            expected_dim=4,
        )
        assert result.size == 4

    def test_coerce_finite_vector_dimension_mismatch_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Dimensión incompatible lanza DomainIntegrityViolationError.
        VALIDA: Validación estricta de dimensionalidad.
        """
        input_data = [1.0, 2.0, 3.0]
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_vector(
                "test_vector",
                input_data,
                expected_dim=4,
            )
        assert "dimensión" in str(exc_info.value).lower()

    def test_coerce_finite_vector_empty_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Vector vacío lanza DomainIntegrityViolationError.
        VALIDA: No-degeneración del espacio vectorial.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_vector("test_empty", np.array([], dtype=np.float64))
        assert "vacío" in str(exc_info.value).lower()

    def test_coerce_finite_vector_nan_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Vector con NaN lanza DomainIntegrityViolationError.
        VALIDA: Finitud absoluta de componentes.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_vector(
                "test_nan",
                np.array([1.0, np.nan, 3.0, 4.0]),
            )
        assert "no finitas" in str(exc_info.value).lower()

    def test_coerce_finite_vector_inf_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Vector con infinito lanza DomainIntegrityViolationError.
        VALIDA: Finitud absoluta de componentes.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_vector(
                "test_inf",
                np.array([1.0, np.inf, 3.0, 4.0]),
            )
        assert "no finitas" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.3. Pruebas de Coerción de Matrices (Método: _coerce_finite_square_matrix)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_finite_square_matrix_valid(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Coerción de matriz cuadrada válida.
        VALIDA: Conversión a NDArray[np.float64] 2-D cuadrada.
        """
        result = phase1_certifier._coerce_finite_square_matrix("test_matrix", valid_metric_4x4)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.ndim == 2
        assert result.shape[0] == result.shape[1]
        assert result.shape[0] == 4
        assert np.all(np.isfinite(result))

    def test_coerce_finite_square_matrix_non_square_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Matriz no cuadrada lanza DomainIntegrityViolationError.
        VALIDA: Exigencia de cuadratura.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_square_matrix(
                "test_rect",
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            )
        assert "cuadrada" in str(exc_info.value).lower()

    def test_coerce_finite_square_matrix_empty_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Matriz vacía lanza DomainIntegrityViolationError.
        VALIDA: No-degeneración matricial.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_square_matrix(
                "test_empty",
                np.array([]).reshape(0, 0),
            )
        assert "vacía" in str(exc_info.value).lower()

    def test_coerce_finite_square_matrix_nan_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Matriz con NaN lanza DomainIntegrityViolationError.
        VALIDA: Finitud absoluta de entradas.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._coerce_finite_square_matrix(
                "test_nan",
                np.array([[1.0, np.nan], [0.0, 1.0]]),
            )
        assert "no finitas" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.4. Pruebas de Certificación de Métrica Riemanniana (Método: _certify_riemannian_metric)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_riemannian_metric_valid(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Certificación de métrica Riemanniana válida.
        VALIDA: §1. G = Gᵀ, G ≻ 0, κ(G) ≤ 10¹².
        """
        G_certified, metric_audit = phase1_certifier._certify_riemannian_metric(valid_metric_4x4)
        assert isinstance(metric_audit, MetricManifoldData)
        assert metric_audit.dimension == 4
        assert metric_audit.is_positive_definite is True
        assert metric_audit.metric_condition_number < _MAX_METRIC_CONDITION_NUMBER
        assert np.allclose(G_certified, G_certified.T)  # Simetría

    def test_certify_riemannian_metric_none_uses_default(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Métrica None usa G_PHYSICS por defecto.
        VALIDA: Fallback Euclidiano.
        """
        G_certified, metric_audit = phase1_certifier._certify_riemannian_metric(None)
        assert isinstance(metric_audit, MetricManifoldData)
        assert metric_audit.is_positive_definite is True
        assert np.array_equal(G_certified, G_PHYSICS)

    def test_certify_riemannian_metric_asymmetric_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Métrica asimétrica lanza MetricManifoldDegeneracyError.
        VALIDA: §1. Simetría estricta G = Gᵀ.
        """
        G_asym = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],  # Asimetría significativa
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        with pytest.raises(MetricManifoldDegeneracyError) as exc_info:
            phase1_certifier._certify_riemannian_metric(G_asym)
        assert "simétrica" in str(exc_info.value).lower() or "simetría" in str(exc_info.value).lower()

    def test_certify_riemannian_metric_not_psd_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Métrica no definida positiva lanza MetricManifoldDegeneracyError.
        VALIDA: §1. G ≻ 0 (positividad estricta).
        """
        G_not_psd = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        with pytest.raises(MetricManifoldDegeneracyError) as exc_info:
            phase1_certifier._certify_riemannian_metric(G_not_psd)
        assert "definida positiva" in str(exc_info.value).lower() or "λ_min" in str(exc_info.value)

    def test_certify_riemannian_metric_ill_conditioned_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Métrica mal condicionada lanza MetricManifoldDegeneracyError.
        VALIDA: §1. κ(G) ≤ 10¹².
        """
        # Matriz con número de condición muy alto
        G_ill = np.array([
            [1e15, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        with pytest.raises(MetricManifoldDegeneracyError) as exc_info:
            phase1_certifier._certify_riemannian_metric(G_ill)
        assert "condicionada" in str(exc_info.value).lower() or "κ" in str(exc_info.value)

    def test_certify_riemannian_metric_dimension_mismatch_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Dimensión incompatible lanza DomainIntegrityViolationError.
        VALIDA: Consistencia dimensional esperada.
        """
        G_3x3 = np.eye(3, dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._certify_riemannian_metric(G_3x3, expected_dim=4)
        assert "dimensión" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.5. Pruebas de Certificación de Vectores de Subespacio (Método: _certify_subspace_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_subspace_vector_valid(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
    ) -> None:
        """
        PRUEBA: Certificación de vector de subespacio válido.
        VALIDA: Norma Euclidiana no degenerada.
        """
        v_physics, _, _, _ = valid_subspace_vectors_4d
        result = phase1_certifier._certify_subspace_vector("V_physics", v_physics, dimension=4)
        assert isinstance(result, np.ndarray)
        assert result.size == 4
        assert np.all(np.isfinite(result))

    def test_certify_subspace_vector_null_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Vector nulo lanza DomainIntegrityViolationError.
        VALIDA: No-degeneración de subespacio.
        """
        v_null = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._certify_subspace_vector("V_null", v_null, dimension=4)
        assert "nulo" in str(exc_info.value).lower() or "degenerado" in str(exc_info.value).lower()

    def test_certify_subspace_vector_near_null_raises(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
    ) -> None:
        """
        PRUEBA: Vector casi nulo lanza DomainIntegrityViolationError.
        VALIDA: Tolerancia de norma mínima.
        """
        v_near_null = np.array([1e-15, 1e-15, 1e-15, 1e-15], dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_certifier._certify_subspace_vector("V_near_null", v_near_null, dimension=4)
        assert "degenerado" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.6. Pruebas de Handoff Fase 1 → Fase 2 (Método: _phase1_certify_and_handoff_to_phase2)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase1_certify_and_handoff_to_phase2_valid(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 1 a Fase 2.
        VALIDA: Continuidad funtorial Φ₁ → Φ₂.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            G_metric=valid_metric_4x4,
        )
        assert isinstance(handoff, Phase1MetricHandoff)
        assert isinstance(handoff.metric_audit, MetricManifoldData)
        assert isinstance(handoff.G_certified, np.ndarray)
        assert isinstance(handoff.V_physics_certified, np.ndarray)
        assert isinstance(handoff.V_topology_certified, np.ndarray)
        assert isinstance(handoff.V_control_certified, np.ndarray)
        assert isinstance(handoff.V_thermo_certified, np.ndarray)
        # Validación de handoff: datos certificados deben ser finitos
        assert np.all(np.isfinite(handoff.G_certified))
        assert np.all(np.isfinite(handoff.V_physics_certified))

    def test_phase1_metric_handoff_immutability(
        self,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Phase1MetricHandoff es inmutable.
        VALIDA: Integridad del artefacto de handoff.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            G_metric=valid_metric_4x4,
        )
        with pytest.raises((AttributeError, TypeError)):
            handoff.metric_audit = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 2: CERTIFICACIÓN DE LA DESCOMPOSICIÓN ORTOGONAL
#   Valida: ⟨v_i, v_j⟩_G = δ_ij
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase2_OrthogonalDecompositionCertifier:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 2: CERTIFICACIÓN DE LA DESCOMPOSICIÓN ORTOGONAL                                 ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la ortogonalidad Riemanniana de los subespacios.        ║
    ║  Cada método prueba un axioma específico del §2 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.1. Pruebas de Producto Interno Covariante (Método: _metric_inner_product)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_metric_inner_product_valid(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Producto interno covariante calculado correctamente.
        VALIDA: ⟨u, v⟩_G = uᵀ G v.
        """
        u = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = phase2_certifier._metric_inner_product(u, v, valid_metric_4x4)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0  # Producto de vector consigo mismo

    def test_metric_inner_product_orthogonal_vectors(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Producto interno de vectores ortogonales es cero.
        VALIDA: ⟨e_i, e_j⟩ = 0 para i ≠ j.
        """
        u = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        result = phase2_certifier._metric_inner_product(u, v, identity_matrix_4x4)
        assert result == 0.0

    def test_metric_inner_product_non_finite_raises(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Producto interno no finito lanza DomainIntegrityViolationError.
        VALIDA: Finitud del producto interno.
        """
        u = np.array([np.inf, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._metric_inner_product(u, u, valid_metric_4x4)
        assert "no finito" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.2. Pruebas de Norma Riemanniana (Método: _metric_subspace_norm)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_metric_subspace_norm_valid(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Norma Riemanniana calculada correctamente.
        VALIDA: ||v||_G = sqrt(vᵀ G v).
        """
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = phase2_certifier._metric_subspace_norm("test_vector", v, valid_metric_4x4)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_metric_subspace_norm_degenerate_raises(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Norma degenerada lanza DomainIntegrityViolationError.
        VALIDA: Positividad estricta de norma.
        """
        v = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._metric_subspace_norm("test_null", v, valid_metric_4x4)
        assert "degenerada" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.3. Pruebas de Normalización Riemanniana (Método: _normalize_subspace_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_normalize_subspace_vector_valid(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Normalización Riemanniana de vector válido.
        VALIDA: ||v̂||_G = 1.
        """
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        result = phase2_certifier._normalize_subspace_vector("test_vector", v, valid_metric_4x4)
        assert isinstance(result, np.ndarray)
        assert result.size == 4
        # Verificar que la norma normalizada es 1
        norm = phase2_certifier._metric_subspace_norm("normalized", result, valid_metric_4x4)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_normalize_subspace_vector_non_finite_raises(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Normalización produce componentes no finitas lanza excepción.
        VALIDA: Finitud de vector normalizado.
        """
        v = np.array([np.inf, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._normalize_subspace_vector("test_inf", v, valid_metric_4x4)
        assert "no finitas" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.4. Pruebas de Certificación de Descomposición Ortogonal (Método: _certify_orthogonal_decomposition_from_certified_vectors)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_orthogonal_decomposition_valid(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Certificación de descomposición ortogonal válida.
        VALIDA: §2. ⟨v_i, v_j⟩_G = δ_ij.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        audit, basis_matrix = phase2_certifier._certify_orthogonal_decomposition_from_certified_vectors(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            G_metric=identity_matrix_4x4,
        )
        assert isinstance(audit, OrthogonalDecompositionData)
        assert audit.is_strictly_orthogonal is True
        assert audit.off_diagonal_norm < _ORTHOGONALITY_TOLERANCE
        assert basis_matrix.shape == (4, 4)

    def test_certify_orthogonal_decomposition_non_orthogonal_raises(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Vectores no ortogonales lanza NonOrthogonalSubspaceError.
        VALIDA: §2. Independencia lineal absoluta.
        """
        # Vectores no ortogonales (comparten componentes)
        v1 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        v2 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)  # Mismo que v1
        v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        v4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        with pytest.raises(NonOrthogonalSubspaceError) as exc_info:
            phase2_certifier._certify_orthogonal_decomposition_from_certified_vectors(
                V_physics=v1,
                V_topology=v2,
                V_control=v3,
                V_thermo=v4,
                G_metric=identity_matrix_4x4,
            )
        assert "ortogonal" in str(exc_info.value).lower() or "covarianza" in str(exc_info.value).lower()

    def test_certify_orthogonal_decomposition_gram_ill_conditioned_raises(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
    ) -> None:
        """
        PRUEBA: Matriz de Gram mal condicionada lanza NonOrthogonalSubspaceError.
        VALIDA: κ(Gram) < 10¹².
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        # Métrica mal condicionada
        G_ill = np.array([
            [1e15, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        with pytest.raises(NonOrthogonalSubspaceError) as exc_info:
            phase2_certifier._certify_orthogonal_decomposition_from_certified_vectors(
                V_physics=v_physics,
                V_topology=v_topology,
                V_control=v_control,
                V_thermo=v_thermo,
                G_metric=G_ill,
            )
        assert "condicionada" in str(exc_info.value).lower() or "κ" in str(exc_info.value)

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.5. Pruebas de Wrapper Público (Método: _certify_orthogonal_decomposition)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_orthogonal_decomposition_valid(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Wrapper público de certificación ortogonal.
        VALIDA: Compatibilidad retroactiva.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        audit = phase2_certifier._certify_orthogonal_decomposition(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            G_metric=identity_matrix_4x4,
        )
        assert isinstance(audit, OrthogonalDecompositionData)
        assert audit.is_strictly_orthogonal is True

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.6. Pruebas de Handoff Fase 2 → Fase 3 (Método: _phase2_certify_and_handoff_to_phase3)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase2_certify_and_handoff_to_phase3_valid(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 2 a Fase 3.
        VALIDA: Continuidad funtorial Φ₂ → Φ₃.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        phase1_handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            G_metric=valid_metric_4x4,
        )
        handoff = phase2_certifier._phase2_certify_and_handoff_to_phase3(phase1_handoff)
        assert isinstance(handoff, Phase2OrthogonalityHandoff)
        assert isinstance(handoff.phase1_handoff, Phase1MetricHandoff)
        assert isinstance(handoff.orthogonality_audit, OrthogonalDecompositionData)
        assert isinstance(handoff.basis_matrix, np.ndarray)
        assert handoff.orthogonality_audit.is_strictly_orthogonal is True

    def test_phase2_handoff_invalid_phase1_handoff_raises(
        self,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
    ) -> None:
        """
        PRUEBA: Handoff de Fase 1 inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._phase2_certify_and_handoff_to_phase3(
                phase1_handoff=None  # type: ignore
            )
        assert "phase1metrichandoff" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 3: IMPOSICIÓN DE INMUTABILIDAD Y PUNTO FIJO
#   Valida: ∇_τ Ψ = 0
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase3_TensorImmutabilityEnforcer:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 3: IMPOSICIÓN DE INMUTABILIDAD Y PUNTO FIJO                                     ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la inmutabilidad temporal del tensor de estado.         ║
    ║  Cada método prueba un axioma específico del §3 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.1. Pruebas de Coerción de Vectores de Estado (Método: _coerce_state_vector)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_state_vector_valid(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
    ) -> None:
        """
        PRUEBA: Coerción de vector de estado válido.
        VALIDA: Conversión a NDArray[np.float64] con dimensión esperada.
        """
        psi = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = phase3_enforcer._coerce_state_vector("Psi", psi, dimension=4)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.size == 4

    def test_coerce_state_vector_dimension_mismatch_raises(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
    ) -> None:
        """
        PRUEBA: Dimensión incompatible lanza DomainIntegrityViolationError.
        VALIDA: Consistencia dimensional con métrica.
        """
        psi = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Dimensión 3
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_enforcer._coerce_state_vector("Psi", psi, dimension=4)
        assert "dimensión" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.2. Pruebas de Norma Riemanniana de Estado (Método: _riemannian_state_norm)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_riemannian_state_norm_valid(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Norma Riemanniana de estado calculada correctamente.
        VALIDA: ||Ψ||_G = sqrt(Ψᵀ G Ψ).
        """
        psi = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = phase3_enforcer._riemannian_state_norm("Psi", psi, valid_metric_4x4)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_riemannian_state_norm_negative_quadratic_raises(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
    ) -> None:
        """
        PRUEBA: Forma cuadrática negativa lanza PhaseSpaceCorruptionError.
        VALIDA: Positividad de norma Riemanniana.
        """
        # Métrica no PSD para forzar forma cuadrática negativa
        G_negative = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(PhaseSpaceCorruptionError) as exc_info:
            phase3_enforcer._riemannian_state_norm("Psi", psi, G_negative)
        assert "negativa" in str(exc_info.value).lower() or "cuadrática" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.3. Pruebas de Punto Fijo Interno (Método: _enforce_tensor_immutability_and_fixpoint_internal)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_enforce_tensor_immutability_and_fixpoint_internal_valid(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Punto fijo válido (Ψ(t₀) = Ψ(t₁)).
        VALIDA: §3. ∇_τ Ψ = 0.
        """
        psi_t0, psi_t1 = valid_state_vectors_4d
        result = phase3_enforcer._enforce_tensor_immutability_and_fixpoint_internal(
            Psi_t0=psi_t0,
            Psi_t1=psi_t1,
            G_metric=identity_matrix_4x4,
        )
        assert isinstance(result, FixpointVerificationData)
        assert result.is_fixed_point is True
        assert result.covariant_derivative_norm < _FIXPOINT_TOLERANCE

    def test_enforce_tensor_immutability_and_fixpoint_internal_corruption_raises(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Corrupción temporal (Ψ(t₀) ≠ Ψ(t₁)) lanza PhaseSpaceCorruptionError.
        VALIDA: §3. Inmutabilidad Tensorial.
        """
        psi_t0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        psi_t1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)  # Diferente
        with pytest.raises(PhaseSpaceCorruptionError) as exc_info:
            phase3_enforcer._enforce_tensor_immutability_and_fixpoint_internal(
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
                G_metric=identity_matrix_4x4,
            )
        assert "corrupción" in str(exc_info.value).lower() or "punto fijo" in str(exc_info.value).lower()

    def test_enforce_tensor_immutability_and_fixpoint_internal_dimension_mismatch_raises(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Dimensión incompatible entre Ψ(t₀) y Ψ(t₁) lanza excepción.
        VALIDA: Consistencia dimensional temporal.
        """
        psi_t0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        psi_t1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Dimensión diferente
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_enforcer._enforce_tensor_immutability_and_fixpoint_internal(
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
                G_metric=identity_matrix_4x4,
            )
        assert "dimensión" in str(exc_info.value).lower() or "incompatible" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.4. Pruebas de Wrapper Público (Método: _enforce_tensor_immutability_and_fixpoint)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_enforce_tensor_immutability_and_fixpoint_valid(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Wrapper público de inmutabilidad.
        VALIDA: Compatibilidad retroactiva.
        """
        psi_t0, psi_t1 = valid_state_vectors_4d
        result = phase3_enforcer._enforce_tensor_immutability_and_fixpoint(
            Psi_t0=psi_t0,
            Psi_t1=psi_t1,
            G_metric=identity_matrix_4x4,
        )
        assert isinstance(result, FixpointVerificationData)
        assert result.is_fixed_point is True

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.5. Pruebas de Finalización Funtorial (Método: _phase3_finalize_from_phase2_handoff)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase3_finalize_from_phase2_handoff_valid(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        phase1_certifier: Phase1_RiemannianMetricCertifier,
        phase2_certifier: Phase2_OrthogonalDecompositionCertifier,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Finalización funtorial completa Φ₃ ∘ Φ₂ ∘ Φ₁.
        VALIDA: Composición de las tres fases.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        psi_t0, psi_t1 = valid_state_vectors_4d
        phase1_handoff = phase1_certifier._phase1_certify_and_handoff_to_phase2(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            G_metric=valid_metric_4x4,
        )
        phase2_handoff = phase2_certifier._phase2_certify_and_handoff_to_phase3(phase1_handoff)
        state = phase3_enforcer._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            Psi_t0=psi_t0,
            Psi_t1=psi_t1,
        )
        assert isinstance(state, TensorialPhaseSpaceState)
        assert isinstance(state.orthogonality_audit, OrthogonalDecompositionData)
        assert isinstance(state.fixpoint_audit, FixpointVerificationData)
        assert isinstance(state.metric_audit, MetricManifoldData)
        assert state.is_epistemologically_valid is True
        assert state.fixpoint_audit.is_fixed_point is True

    def test_phase3_handoff_invalid_phase2_handoff_raises(
        self,
        phase3_enforcer: Phase3_TensorImmutabilityEnforcer,
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """
        PRUEBA: Handoff de Fase 2 inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de prefijo formal.
        """
        psi_t0, psi_t1 = valid_state_vectors_4d
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_enforcer._phase3_finalize_from_phase2_handoff(
                phase2_handoff=None,  # type: ignore
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
            )
        assert "phase2orthogonalityhandoff" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   ORQUESTADOR SUPREMO: TELEMETRYSCHEMASAGENT (Pruebas de Integración)
#   Valida: Endofuntor Z_Tensorial = Φ₃ ∘ Φ₂ ∘ Φ₁
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestTelemetrySchemasAgent_Integration:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  ORQUESTADOR SUPREMO: TELEMETRYSCHEMASAGENT                                           ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Pruebas de integración que validan el endofuntor completo Z_Tensorial.               ║
    ║  Estas pruebas aseguran que la composición Φ₃ ∘ Φ₂ ∘ Φ₁ funciona correctamente.       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_telemetry_schemas_agent_execute_tensorial_phase_space_governance_valid(
        self,
        schemas_agent: TelemetrySchemasAgent,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Ejecución completa del gobierno de espacio de fase tensorial.
        VALIDA: Endofuntor Z_Tensorial con datos válidos.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        psi_t0, psi_t1 = valid_state_vectors_4d
        state = schemas_agent.execute_tensorial_phase_space_governance(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            Psi_t0=psi_t0,
            Psi_t1=psi_t1,
            G_metric=valid_metric_4x4,
        )
        assert isinstance(state, TensorialPhaseSpaceState)
        assert state.is_epistemologically_valid is True
        assert state.orthogonality_audit.is_strictly_orthogonal is True
        assert state.fixpoint_audit.is_fixed_point is True
        assert isinstance(state.metric_audit, MetricManifoldData)

    def test_telemetry_schemas_agent_call_alias_valid(
        self,
        schemas_agent: TelemetrySchemasAgent,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Alias invocable __call__ del endofuntor.
        VALIDA: Sintaxis alternativa de ejecución.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        psi_t0, psi_t1 = valid_state_vectors_4d
        state = schemas_agent(
            V_physics=v_physics,
            V_topology=v_topology,
            V_control=v_control,
            V_thermo=v_thermo,
            Psi_t0=psi_t0,
            Psi_t1=psi_t1,
            G_metric=valid_metric_4x4,
        )
        assert isinstance(state, TensorialPhaseSpaceState)
        assert state.is_epistemologically_valid is True

    def test_telemetry_schemas_agent_metric_manifold_degeneracy_error(
        self,
        schemas_agent: TelemetrySchemasAgent,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """
        PRUEBA: Métrica degenerada lanza MetricManifoldDegeneracyError.
        VALIDA: Propagación de excepciones de Fase 1.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        psi_t0, psi_t1 = valid_state_vectors_4d
        G_not_psd = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        with pytest.raises(MetricManifoldDegeneracyError):
            schemas_agent(
                V_physics=v_physics,
                V_topology=v_topology,
                V_control=v_control,
                V_thermo=v_thermo,
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
                G_metric=G_not_psd,
            )

    def test_telemetry_schemas_agent_non_orthogonal_subspace_error(
        self,
        schemas_agent: TelemetrySchemasAgent,
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Subespacios no ortogonales lanza NonOrthogonalSubspaceError.
        VALIDA: Propagación de excepciones de Fase 2.
        """
        psi_t0, psi_t1 = valid_state_vectors_4d
        # Vectores no ortogonales
        v1 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        v2 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        v4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        with pytest.raises(NonOrthogonalSubspaceError):
            schemas_agent(
                V_physics=v1,
                V_topology=v2,
                V_control=v3,
                V_thermo=v4,
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
                G_metric=identity_matrix_4x4,
            )

    def test_telemetry_schemas_agent_phase_space_corruption_error(
        self,
        schemas_agent: TelemetrySchemasAgent,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Corrupción temporal lanza PhaseSpaceCorruptionError.
        VALIDA: Propagación de excepciones de Fase 3.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        psi_t0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        psi_t1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)  # Diferente
        with pytest.raises(PhaseSpaceCorruptionError):
            schemas_agent(
                V_physics=v_physics,
                V_topology=v_topology,
                V_control=v_control,
                V_thermo=v_thermo,
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
                G_metric=identity_matrix_4x4,
            )

    def test_telemetry_schemas_agent_domain_integrity_violation_error(
        self,
        schemas_agent: TelemetrySchemasAgent,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_state_vectors_4d: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """
        PRUEBA: Violación de integridad de dominio lanza DomainIntegrityViolationError.
        VALIDA: Validación de tipos de entrada.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        psi_t0, psi_t1 = valid_state_vectors_4d
        # Vector nulo
        v_null = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(DomainIntegrityViolationError):
            schemas_agent(
                V_physics=v_null,
                V_topology=v_topology,
                V_control=v_control,
                V_thermo=v_thermo,
                Psi_t0=psi_t0,
                Psi_t1=psi_t1,
                G_metric=None,
            )

    def test_telemetry_schemas_agent_inheritance_chain(
        self,
        schemas_agent: TelemetrySchemasAgent,
    ) -> None:
        """
        PRUEBA: Cadena de herencia del TelemetrySchemasAgent.
        VALIDA: Arquitectura de fases anidadas.
        """
        assert isinstance(schemas_agent, TelemetrySchemasAgent)
        assert isinstance(schemas_agent, Phase3_TensorImmutabilityEnforcer)
        assert isinstance(schemas_agent, Phase2_OrthogonalDecompositionCertifier)
        assert isinstance(schemas_agent, Phase1_RiemannianMetricCertifier)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Z. PRUEBAS DE ESTRUCTURAS DE DATOS (Data Classes)
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestDataStructures:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE ESTRUCTURAS DE DATOS INMUTABLES                                           ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida la integridad de todos los DTOs del espacio tensorial.                        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_metric_manifold_data_creation(
        self,
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de MetricManifoldData.
        VALIDA: Estructura inmutable del certificado Riemanniano.
        """
        audit = MetricManifoldData(
            dimension=4,
            metric_condition_number=10.0,
            symmetry_deviation=1e-15,
            min_eigenvalue=0.5,
            is_positive_definite=True,
        )
        assert audit.dimension == 4
        assert audit.is_positive_definite is True

    def test_orthogonal_decomposition_data_creation(
        self,
        identity_matrix_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de OrthogonalDecompositionData.
        VALIDA: Artefacto de Fase 2.
        """
        audit = OrthogonalDecompositionData(
            gram_matrix=identity_matrix_4x4,
            off_diagonal_norm=0.0,
            is_strictly_orthogonal=True,
        )
        assert audit.is_strictly_orthogonal is True
        assert audit.off_diagonal_norm == 0.0

    def test_fixpoint_verification_data_creation(
        self,
    ) -> None:
        """
        PRUEBA: Creación de FixpointVerificationData.
        VALIDA: Artefacto de Fase 3.
        """
        audit = FixpointVerificationData(
            covariant_derivative_norm=0.0,
            is_fixed_point=True,
        )
        assert audit.is_fixed_point is True
        assert audit.covariant_derivative_norm == 0.0

    def test_phase1_metric_handoff_creation(
        self,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de Phase1MetricHandoff.
        VALIDA: Puente funtorial Φ₁ → Φ₂.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        metric_audit = MetricManifoldData(
            dimension=4,
            metric_condition_number=10.0,
            symmetry_deviation=1e-15,
            min_eigenvalue=0.5,
            is_positive_definite=True,
        )
        handoff = Phase1MetricHandoff(
            metric_audit=metric_audit,
            G_certified=valid_metric_4x4,
            V_physics_certified=v_physics,
            V_topology_certified=v_topology,
            V_control_certified=v_control,
            V_thermo_certified=v_thermo,
        )
        assert isinstance(handoff.metric_audit, MetricManifoldData)
        assert np.array_equal(handoff.G_certified, valid_metric_4x4)

    def test_phase2_orthogonality_handoff_creation(
        self,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de Phase2OrthogonalityHandoff.
        VALIDA: Puente funtorial Φ₂ → Φ₃.
        """
        v_physics, v_topology, v_control, v_thermo = valid_subspace_vectors_4d
        metric_audit = MetricManifoldData(
            dimension=4,
            metric_condition_number=10.0,
            symmetry_deviation=1e-15,
            min_eigenvalue=0.5,
            is_positive_definite=True,
        )
        phase1_handoff = Phase1MetricHandoff(
            metric_audit=metric_audit,
            G_certified=valid_metric_4x4,
            V_physics_certified=v_physics,
            V_topology_certified=v_topology,
            V_control_certified=v_control,
            V_thermo_certified=v_thermo,
        )
        ortho_audit = OrthogonalDecompositionData(
            gram_matrix=np.eye(4, dtype=np.float64),
            off_diagonal_norm=0.0,
            is_strictly_orthogonal=True,
        )
        handoff = Phase2OrthogonalityHandoff(
            phase1_handoff=phase1_handoff,
            orthogonality_audit=ortho_audit,
            basis_matrix=np.eye(4, dtype=np.float64),
        )
        assert isinstance(handoff.phase1_handoff, Phase1MetricHandoff)
        assert isinstance(handoff.orthogonality_audit, OrthogonalDecompositionData)

    def test_tensorial_phase_space_state_creation(
        self,
        valid_subspace_vectors_4d: tuple[NDArray[np.float64], ...],
        valid_metric_4x4: NDArray[np.float64],
    ) -> None:
        """
        PRUEBA: Creación de TensorialPhaseSpaceState (objeto final).
        VALIDA: Estado epistemológico completo del endofuntor.
        """
        metric_audit = MetricManifoldData(
            dimension=4,
            metric_condition_number=10.0,
            symmetry_deviation=1e-15,
            min_eigenvalue=0.5,
            is_positive_definite=True,
        )
        ortho_audit = OrthogonalDecompositionData(
            gram_matrix=np.eye(4, dtype=np.float64),
            off_diagonal_norm=0.0,
            is_strictly_orthogonal=True,
        )
        fixpoint_audit = FixpointVerificationData(
            covariant_derivative_norm=0.0,
            is_fixed_point=True,
        )
        state = TensorialPhaseSpaceState(
            orthogonality_audit=ortho_audit,
            fixpoint_audit=fixpoint_audit,
            is_epistemologically_valid=True,
            metric_audit=metric_audit,
        )
        assert state.is_epistemologically_valid is True
        assert isinstance(state.orthogonality_audit, OrthogonalDecompositionData)
        assert isinstance(state.fixpoint_audit, FixpointVerificationData)
        assert isinstance(state.metric_audit, MetricManifoldData)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §∞. PRUEBAS DE CONSTANTES MATEMÁTICAS Y DE TOLERANCIA
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestMathematicalToleranceConstants:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE CONSTANTES MATEMÁTICAS Y DE TOLERANCIA                                    ║
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

    def test_orthogonality_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _ORTHOGONALITY_TOLERANCE.
        VALIDA: Tolerancia de ortogonalidad.
        """
        assert _ORTHOGONALITY_TOLERANCE == 1e-10
        assert _ORTHOGONALITY_TOLERANCE > 0

    def test_fixpoint_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _FIXPOINT_TOLERANCE.
        VALIDA: Tolerancia de punto fijo.
        """
        assert _FIXPOINT_TOLERANCE == 1e-12
        assert _FIXPOINT_TOLERANCE > 0

    def test_metric_symmetry_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _METRIC_SYMMETRY_TOLERANCE.
        VALIDA: Tolerancia de simetría métrica.
        """
        assert _METRIC_SYMMETRY_TOLERANCE == 1e-12
        assert _METRIC_SYMMETRY_TOLERANCE > 0

    def test_max_metric_condition_number_value(self) -> None:
        """
        PRUEBA: Valor de _MAX_METRIC_CONDITION_NUMBER.
        VALIDA: Número de condición máximo admisible.
        """
        assert _MAX_METRIC_CONDITION_NUMBER == 1e12
        assert _MAX_METRIC_CONDITION_NUMBER > 0

    def test_subspace_norm_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _SUBSPACE_NORM_TOLERANCE.
        VALIDA: Norma mínima de subespacio.
        """
        assert _SUBSPACE_NORM_TOLERANCE == 1e-12
        assert _SUBSPACE_NORM_TOLERANCE > 0

    def test_g_physics_default_metric(self) -> None:
        """
        PRUEBA: G_PHYSICS es matriz identidad 4x4.
        VALIDA: Métrica Euclidiana por defecto.
        """
        assert isinstance(G_PHYSICS, np.ndarray)
        assert G_PHYSICS.shape == (4, 4)
        assert np.array_equal(G_PHYSICS, np.eye(4, dtype=np.float64))


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Ω. EJECUCIÓN DIRECTA (Para debugging)
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución directa para debugging fuera de pytest.
    Uso: python tests/unit/agents/core/test_telemetry_schemas_agent.py
    """
    import sys
    import os
    
    # Agregar el directorio raíz al path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    
    pytest.main([__file__, "-v", "--tb=short"])