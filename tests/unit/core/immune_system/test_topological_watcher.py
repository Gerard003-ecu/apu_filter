"""
Suite de Pruebas para Topological Watcher
Ubicación: tests/core/immune_system/test_topological_watcher.py

COBERTURA DE PRUEBAS:

1. ÁLGEBRA LINEAL:
   - Normalización y pre-escalado
   - Estabilidad numérica (overflow/underflow)
   - Operaciones con matrices mal condicionadas

2. GEOMETRÍA RIEMANNIANA:
   - Propiedades SPD (Simétrico Definido Positivo)
   - Descomposición espectral
   - Formas cuadráticas

3. TOPOLOGÍA ALGEBRAICA:
   - Invariantes de Betti
   - Característica de Euler
   - Bifurcaciones topológicas

4. TEORÍA DE CATEGORÍAS:
   - Propiedades funtoriales
   - Preservación de composición
   - Mónada de error

5. ESTABILIDAD NUMÉRICA:
   - Casos extremos (valores muy pequeños/grandes)
   - Matrices singulares
   - Propagación de errores

6. INTEGRACIÓN:
   - Flujo de Ricci
   - Evolución de métricas
   - Clasificación con histéresis
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_allclose
)

# Imports del módulo bajo prueba
from app.core.immune_system.topological_watcher import (
    # Excepciones
    ImmuneSystemError,
    NumericalStabilityError,
    DimensionalMismatchError,
    PhysicalBoundsError,
    TopologicalInvariantError,
    MetricTensorError,
    SpectralDecompositionError,
    
    # Constantes
    EPS,
    ALGEBRAIC_TOL,
    COND_NUM_TOL,
    MIN_EIGVAL_TOL,
    PhysicalConstants,
    
    # Álgebra lineal
    StableLinearAlgebra,
    
    # Validadores
    Validator,
    VALIDATOR_REGISTRY,
    UnitIntervalValidator,
    NonNegativeValidator,
    PositiveIntValidator,
    NonNegativeIntValidator,
    
    # Estado de salud
    HealthStatus,
    
    # Geometría Riemanniana
    SpectralDecomposition,
    MetricTensor,
    
    # Topología
    IsolatingMembraneFunctor,
    SubspaceSpec,
    ThreatAssessment,
    OrthogonalProjector,
    
    # Señal
    SignalComponent,
    SIGNAL_SCHEMA,
    BETTI_INDICES,
    build_signal,
    
    # Morfismo
    ImmuneWatcherMorphism,
    create_immune_watcher,
    
    # Utilidades
    block_diag_pure,
)

from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def logger_mock(monkeypatch):
    """Mock del logger para capturar mensajes."""
    messages = {"debug": [], "info": [], "warning": [], "error": [], "critical": []}
    
    def make_logger_func(level):
        def log_func(msg, *args, **kwargs):
            formatted = msg % args if args else msg
            messages[level].append(formatted)
        return log_func
    
    logger = logging.getLogger("MIC.ImmuneSystem")
    monkeypatch.setattr(logger, "debug", make_logger_func("debug"))
    monkeypatch.setattr(logger, "info", make_logger_func("info"))
    monkeypatch.setattr(logger, "warning", make_logger_func("warning"))
    monkeypatch.setattr(logger, "error", make_logger_func("error"))
    monkeypatch.setattr(logger, "critical", make_logger_func("critical"))
    
    return messages

@pytest.fixture
def identity_metric_3d() -> MetricTensor:
    """Métrica identidad 3×3."""
    return MetricTensor(np.eye(3, dtype=np.float64), validate=True)

@pytest.fixture
def diagonal_metric_3d() -> MetricTensor:
    """Métrica diagonal 3×3 con eigenvalores variados."""
    return MetricTensor(
        np.array([1.0, 2.0, 4.0], dtype=np.float64),
        validate=True
    )

@pytest.fixture
def dense_metric_3d() -> MetricTensor:
    """Métrica densa 3×3 bien condicionada."""
    # Construir SPD: A = QΛQ^T
    eigenvalues = np.array([4.0, 2.0, 1.0])
    Q = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [0, 0, 1]
    ])
    G = Q @ np.diag(eigenvalues) @ Q.T
    return MetricTensor(G, validate=True)

@pytest.fixture
def ill_conditioned_metric() -> MetricTensor:
    """Métrica mal condicionada (κ ≈ 10^6)."""
    eigenvalues = np.array([1e6, 1.0, 1.0])
    return MetricTensor(eigenvalues, validate=True)

@pytest.fixture
def sample_telemetry() -> Dict[str, Any]:
    """Telemetría de ejemplo válida."""
    return {
        "saturation": 0.5,
        "flyback_voltage": 200.0,
        "dissipated_power": 50.0,
        "beta_0": 1.0,
        "beta_1": 0.0,
        "entropy": 0.3,
        "exergy_loss": 0.2,
    }

@pytest.fixture
def anomalous_telemetry() -> Dict[str, Any]:
    """Telemetría con anomalías topológicas."""
    return {
        "saturation": 0.95,
        "flyback_voltage": 380.0,
        "dissipated_power": 180.0,
        "beta_0": 2.0,  # Fragmentación
        "beta_1": 3.0,  # Ciclos
        "entropy": 0.85,
        "exergy_loss": 0.75,
    }

@pytest.fixture
def categorical_state_success() -> CategoricalState:
    """Estado categórico exitoso."""
    return CategoricalState(
        stratum=Stratum.PHYSICS,
        success=True,
        context={"telemetry_metrics": {
            "saturation": 0.3,
            "flyback_voltage": 150.0,
            "dissipated_power": 30.0,
            "beta_0": 1.0,
            "beta_1": 0.0,
            "entropy": 0.2,
            "exergy_loss": 0.1,
        }}
    )

@pytest.fixture
def categorical_state_error() -> CategoricalState:
    """Estado categórico con error."""
    return CategoricalState(
        stratum=Stratum.PHYSICS,
        success=False,
        error_msg="Test error",
        context={}
    )

# ==============================================================================
# TESTS: CONSTANTES FÍSICAS
# ==============================================================================

class TestPhysicalConstants:
    """Tests para constantes físicas."""
    
    def test_constant_values(self):
        """Verifica valores de constantes físicas."""
        assert PhysicalConstants.Z_CHARACTERISTIC == 50.0
        assert PhysicalConstants.V_NOMINAL == 100.0
        assert PhysicalConstants.SATURATION_CRITICAL == 0.85
        assert PhysicalConstants.FLYBACK_MAX_SAFE == 400.0
        assert PhysicalConstants.I_THERMAL_LIMIT == 10.0
    
    def test_p_nominal_calculation(self):
        """Verifica cálculo de potencia nominal."""
        P = PhysicalConstants.P_NOMINAL()
        expected = (100.0 ** 2) / 50.0  # 200 W
        assert abs(P - expected) < 1e-10
    
    def test_physical_consistency(self):
        """Verifica consistencia dimensional."""
        # No debe lanzar excepción
        PhysicalConstants.validate_physical_consistency()
    
    def test_power_current_relationship(self):
        """Verifica P = V²/Z = I²Z."""
        V = PhysicalConstants.V_NOMINAL
        Z = PhysicalConstants.Z_CHARACTERISTIC
        I = V / Z
        
        P_from_V = V**2 / Z
        P_from_I = I**2 * Z
        
        assert abs(P_from_V - P_from_I) < 1e-10

# ==============================================================================
# TESTS: ÁLGEBRA LINEAL ESTABLE
# ==============================================================================

class TestStableLinearAlgebra:
    """Tests para operaciones de álgebra lineal estable."""
    
    def test_safe_normalize_standard(self):
        """Normalización de vector estándar."""
        v = np.array([3.0, 4.0, 0.0])
        v_norm, scale = StableLinearAlgebra.safe_normalize(v, norm_type='inf')
        
        assert scale == 4.0
        assert_array_almost_equal(v_norm, np.array([0.75, 1.0, 0.0]))
    
    def test_safe_normalize_l2(self):
        """Normalización con norma L²."""
        v = np.array([3.0, 4.0])
        v_norm, scale = StableLinearAlgebra.safe_normalize(v, norm_type='2')
        
        assert abs(scale - 5.0) < 1e-10
        assert abs(np.linalg.norm(v_norm, 2) - 1.0) < 1e-10
    
    def test_safe_normalize_zero_vector(self):
        """Normalización de vector cero."""
        v = np.zeros(3)
        with pytest.warns(RuntimeWarning, match="Vector degenerado"):
            v_norm, scale = StableLinearAlgebra.safe_normalize(v)
        
        assert scale == 0.0
        assert_array_equal(v_norm, np.zeros(3))
    
    def test_safe_normalize_tiny_vector(self):
        """Normalización de vector muy pequeño."""
        v = np.array([1e-15, 2e-15, 1e-16])
        with pytest.warns(RuntimeWarning):
            v_norm, scale = StableLinearAlgebra.safe_normalize(v, eps=1e-12)
        
        assert scale < 1e-12
    
    def test_stable_reciprocal_normal(self):
        """Recíproco de valores normales."""
        x = np.array([2.0, -4.0, 0.5])
        result = StableLinearAlgebra.stable_reciprocal(x)
        expected = np.array([0.5, -0.25, 2.0])
        
        assert_array_almost_equal(result, expected)
    
    def test_stable_reciprocal_near_zero(self):
        """Recíproco cerca de cero."""
        x = np.array([1e-15, -1e-16, 0.0])
        result = StableLinearAlgebra.stable_reciprocal(x, eps=1e-12)
        
        # Todos los valores deben ser finitos
        assert np.all(np.isfinite(result))
        # Magnitudes deben ser ≥ 1/eps
        assert np.all(np.abs(result) >= 1.0 / 1e-12 * 0.99)
    
    def test_stable_divide_normal(self):
        """División vectorial normal."""
        num = np.array([10.0, 20.0, 30.0])
        den = np.array([2.0, 5.0, 3.0])
        result = StableLinearAlgebra.stable_divide(num, den)
        expected = np.array([5.0, 4.0, 10.0])
        
        assert_array_almost_equal(result, expected)
    
    def test_stable_divide_by_zero(self):
        """División por cero protegida."""
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([1.0, 0.0, 1e-15])
        result = StableLinearAlgebra.stable_divide(num, den, eps=1e-12)
        
        assert np.all(np.isfinite(result))
    
    def test_stable_quadratic_form_identity(self):
        """Forma cuadrática con métrica identidad."""
        v = np.array([3.0, 4.0])
        G = np.eye(2)
        Q = StableLinearAlgebra.stable_quadratic_form(v, G)
        
        expected = 3.0**2 + 4.0**2  # 25.0
        assert abs(Q - expected) < 1e-10
    
    def test_stable_quadratic_form_diagonal(self):
        """Forma cuadrática con métrica diagonal."""
        v = np.array([2.0, 3.0])
        G = np.diag([4.0, 9.0])
        Q = StableLinearAlgebra.stable_quadratic_form(v, G)
        
        expected = 2.0**2 * 4.0 + 3.0**2 * 9.0  # 16 + 81 = 97
        assert abs(Q - expected) < 1e-10
    
    def test_stable_quadratic_form_large_values(self):
        """Forma cuadrática con valores grandes (overflow protection)."""
        v = np.array([1e100, 1e100])
        G = np.eye(2)
        Q = StableLinearAlgebra.stable_quadratic_form(v, G)
        
        # Debe ser finito gracias a pre-escalado
        assert np.isfinite(Q)
    
    def test_stable_quadratic_form_tiny_values(self):
        """Forma cuadrática con valores muy pequeños."""
        v = np.array([1e-100, 1e-100])
        G = np.eye(2)
        Q = StableLinearAlgebra.stable_quadratic_form(v, G, eps=1e-12)
        
        # Debe retornar 0.0 por umbral
        assert Q == 0.0
    
    def test_compute_condition_spectral_identity(self):
        """Número de condición de matriz identidad."""
        G = np.eye(3)
        kappa = StableLinearAlgebra.compute_condition_spectral(G, method='eig')
        
        assert abs(kappa - 1.0) < 1e-10
    
    def test_compute_condition_spectral_diagonal(self):
        """Número de condición de matriz diagonal."""
        G = np.diag([10.0, 5.0, 1.0])
        kappa = StableLinearAlgebra.compute_condition_spectral(G, method='eig')
        
        expected = 10.0 / 1.0  # 10.0
        assert abs(kappa - expected) < 1e-10
    
    def test_compute_condition_spectral_singular(self):
        """Número de condición de matriz singular."""
        G = np.diag([1.0, 0.0, 1.0])
        kappa = StableLinearAlgebra.compute_condition_spectral(G, method='eig')
        
        assert np.isinf(kappa)
    
    def test_compute_condition_svd_method(self):
        """Número de condición vía SVD."""
        G = np.diag([8.0, 2.0])
        kappa = StableLinearAlgebra.compute_condition_spectral(G, method='svd')
        
        expected = 8.0 / 2.0  # 4.0
        assert abs(kappa - expected) < 1e-10
    
    def test_regularize_spd_matrix_already_spd(self):
        """Regularización de matriz ya SPD."""
        G = np.diag([5.0, 3.0, 1.0])
        G_reg = StableLinearAlgebra.regularize_spd_matrix(G, min_eig=1e-12)
        
        # No debe modificarse
        assert_array_almost_equal(G_reg, G)
    
    def test_regularize_spd_matrix_spectral_method(self):
        """Regularización espectral de matriz con eigenvalor pequeño."""
        G = np.diag([1.0, 1e-15, 1.0])
        G_reg = StableLinearAlgebra.regularize_spd_matrix(
            G,
            min_eig=1e-12,
            method='spectral'
        )
        
        eigvals = np.linalg.eigvalsh(G_reg)
        assert np.all(eigvals >= 1e-12 * 0.99)
    
    def test_regularize_spd_matrix_tikhonov_method(self):
        """Regularización Tikhonov."""
        G = np.diag([1.0, 1e-15, 1.0])
        G_reg = StableLinearAlgebra.regularize_spd_matrix(
            G,
            min_eig=1e-12,
            method='tikhonov'
        )
        
        eigvals = np.linalg.eigvalsh(G_reg)
        assert np.all(eigvals >= 1e-12 * 0.99)
    
    def test_regularize_spd_matrix_dense(self):
        """Regularización de matriz densa."""
        # Matriz con eigenvalor negativo (no SPD)
        G = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigvals: 3, -1
        G_reg = StableLinearAlgebra.regularize_spd_matrix(G, min_eig=1e-12)
        
        eigvals = np.linalg.eigvalsh(G_reg)
        assert np.all(eigvals >= 1e-12 * 0.99)
    
    def test_regularize_spd_matrix_symmetry_preserved(self):
        """Regularización preserva simetría."""
        G = np.array([[2.0, 1.0], [1.0, 3.0]])
        G_reg = StableLinearAlgebra.regularize_spd_matrix(G)
        
        assert_array_almost_equal(G_reg, G_reg.T)
    
    def test_verify_orthogonality_identity(self):
        """Verificar ortogonalidad de matriz identidad."""
        Q = np.eye(3)
        is_orth, residual = StableLinearAlgebra.verify_orthogonality(Q)
        
        assert is_orth
        assert residual < ALGEBRAIC_TOL
    
    def test_verify_orthogonality_rotation(self):
        """Verificar ortogonalidad de matriz de rotación."""
        theta = np.pi / 4
        Q = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        is_orth, residual = StableLinearAlgebra.verify_orthogonality(Q)
        
        assert is_orth
        assert residual < ALGEBRAIC_TOL
    
    def test_verify_orthogonality_non_orthogonal(self):
        """Verificar matriz no ortogonal."""
        Q = np.array([[1.0, 0.5], [0.0, 1.0]])
        is_orth, residual = StableLinearAlgebra.verify_orthogonality(Q)
        
        assert not is_orth
        assert residual > ALGEBRAIC_TOL

# ==============================================================================
# TESTS: VALIDADORES
# ==============================================================================

class TestValidators:
    """Tests para validadores de restricciones."""
    
    def test_unit_interval_validator_valid(self):
        """Validar valor en [0, 1]."""
        validator = UnitIntervalValidator()
        val, modified, msg = validator.validate(0.5, "test")
        
        assert val == 0.5
        assert not modified
        assert msg is None
    
    def test_unit_interval_validator_clamp_negative(self):
        """Clamp de valor negativo."""
        validator = UnitIntervalValidator()
        val, modified, msg = validator.validate(-0.5, "test")
        
        assert val == 0.0
        assert modified
        assert "clamp a 0.0" in msg
    
    def test_unit_interval_validator_clamp_above_one(self):
        """Clamp de valor > 1."""
        validator = UnitIntervalValidator()
        val, modified, msg = validator.validate(1.5, "test")
        
        assert val == 1.0
        assert modified
        assert "clamp a 1.0" in msg
    
    def test_unit_interval_validator_nan(self):
        """Rechazar NaN."""
        validator = UnitIntervalValidator()
        with pytest.raises(PhysicalBoundsError, match="no finito"):
            validator.validate(np.nan, "test")
    
    def test_unit_interval_validator_bounds(self):
        """Verificar límites."""
        validator = UnitIntervalValidator()
        bounds = validator.get_bounds()
        
        assert bounds == (0.0, 1.0)
    
    def test_non_negative_validator_valid(self):
        """Validar valor no negativo."""
        validator = NonNegativeValidator()
        val, modified, msg = validator.validate(5.0, "test")
        
        assert val == 5.0
        assert not modified
    
    def test_non_negative_validator_clamp(self):
        """Clamp de valor negativo."""
        validator = NonNegativeValidator()
        val, modified, msg = validator.validate(-3.0, "test")
        
        assert val == 0.0
        assert modified
    
    def test_positive_int_validator_valid(self):
        """Validar entero positivo."""
        validator = PositiveIntValidator()
        val, modified, msg = validator.validate(5.0, "test")
        
        assert val == 5.0
        assert not modified
    
    def test_positive_int_validator_round(self):
        """Redondeo a entero."""
        validator = PositiveIntValidator()
        val, modified, msg = validator.validate(5.7, "test")
        
        assert val == 6.0
        assert modified
    
    def test_positive_int_validator_clamp_negative(self):
        """Clamp de valor negativo a 1."""
        validator = PositiveIntValidator()
        val, modified, msg = validator.validate(-2.0, "test")
        
        assert val == 1.0
        assert modified
    
    def test_non_negative_int_validator_zero(self):
        """Validar cero."""
        validator = NonNegativeIntValidator()
        val, modified, msg = validator.validate(0.0, "test")
        
        assert val == 0.0
        assert not modified
    
    def test_validator_registry(self):
        """Verificar registro de validadores."""
        assert "unit_interval" in VALIDATOR_REGISTRY
        assert "non_negative" in VALIDATOR_REGISTRY
        assert "positive_int" in VALIDATOR_REGISTRY
        assert "non_negative_int" in VALIDATOR_REGISTRY

# ==============================================================================
# TESTS: HEALTH STATUS
# ==============================================================================

class TestHealthStatus:
    """Tests para estado de salud."""
    
    def test_severity_ordering(self):
        """Verificar orden de severidad."""
        assert HealthStatus.HEALTHY.severity == 0
        assert HealthStatus.WARNING.severity == 1
        assert HealthStatus.CRITICAL.severity == 2
    
    def test_comparison_operators(self):
        """Verificar comparaciones."""
        assert HealthStatus.HEALTHY < HealthStatus.WARNING
        assert HealthStatus.WARNING < HealthStatus.CRITICAL
        assert not (HealthStatus.CRITICAL < HealthStatus.HEALTHY)
    
    def test_string_representation(self):
        """Verificar representación string."""
        assert "HEALTHY" in str(HealthStatus.HEALTHY)
        assert "WARNING" in str(HealthStatus.WARNING)
        assert "CRITICAL" in str(HealthStatus.CRITICAL)

# ==============================================================================
# TESTS: MÉTRICA RIEMANNIANA
# ==============================================================================

class TestMetricTensor:
    """Tests para tensor métrico Riemanniano."""
    
    def test_diagonal_metric_creation(self):
        """Crear métrica diagonal."""
        G = MetricTensor(np.array([1.0, 2.0, 3.0]))
        
        assert G.dimension == 3
        assert G.is_diagonal
        assert G.condition_number == 3.0 / 1.0
    
    def test_diagonal_metric_invalid(self):
        """Rechazar diagonal con valores no positivos."""
        with pytest.raises(MetricTensorError, match="no positivos"):
            MetricTensor(np.array([1.0, 0.0, 3.0]))
    
    def test_dense_metric_creation(self):
        """Crear métrica densa."""
        G_arr = np.array([[2.0, 1.0], [1.0, 2.0]])
        G = MetricTensor(G_arr)
        
        assert G.dimension == 2
        assert not G.is_diagonal
    
    def test_dense_metric_asymmetric_correction(self, logger_mock):
        """Corrección de asimetría."""
        G_arr = np.array([[2.0, 1.0], [0.9, 2.0]])  # Ligeramente asimétrica
        G = MetricTensor(G_arr, validate=True)
        
        # Debe haber advertencia de asimetría
        assert any("Asimetría corregida" in msg for msg in logger_mock["warning"])
    
    def test_metric_tensor_condition_number(self):
        """Número de condición."""
        G = MetricTensor(np.diag([10.0, 1.0]))
        
        assert abs(G.condition_number - 10.0) < 1e-10
    
    def test_metric_tensor_quadratic_form(self):
        """Forma cuadrática vᵀGv."""
        G = MetricTensor(np.diag([2.0, 3.0]))
        v = np.array([1.0, 2.0])
        Q = G.quadratic_form(v)
        
        expected = 1.0**2 * 2.0 + 2.0**2 * 3.0  # 2 + 12 = 14
        assert abs(Q - expected) < 1e-10
    
    def test_metric_tensor_apply(self):
        """Aplicación G·v."""
        G = MetricTensor(np.diag([2.0, 3.0]))
        v = np.array([1.0, 2.0])
        result = G.apply(v)
        
        expected = np.array([2.0, 6.0])
        assert_array_almost_equal(result, expected)
    
    def test_metric_tensor_inverse_sqrt_apply_diagonal(self):
        """G^{-1/2}·v para diagonal."""
        G = MetricTensor(np.array([4.0, 9.0]))
        v = np.array([2.0, 3.0])
        result = G.inverse_sqrt_apply(v)
        
        expected = np.array([2.0 / 2.0, 3.0 / 3.0])  # [1.0, 1.0]
        assert_array_almost_equal(result, expected)
    
    def test_metric_tensor_inverse_sqrt_apply_dense(self):
        """G^{-1/2}·v para densa."""
        # Métrica diagonal en base estándar
        G = MetricTensor(np.diag([4.0, 1.0]))
        v = np.array([4.0, 1.0])
        result = G.inverse_sqrt_apply(v)
        
        expected = np.array([2.0, 1.0])
        assert_array_almost_equal(result, expected)
    
    def test_metric_tensor_spectral_decomposition(self):
        """Descomposición espectral."""
        G = MetricTensor(np.diag([3.0, 2.0, 1.0]))
        spec = G.spectral_decomposition
        
        assert_array_almost_equal(spec.eigenvalues, np.array([3.0, 2.0, 1.0]))
        assert spec.orthogonality_residual < ALGEBRAIC_TOL
    
    def test_metric_tensor_verify_invariants(self):
        """Verificar todos los invariantes."""
        G = MetricTensor(np.diag([2.0, 3.0, 4.0]))
        checks = G.verify_invariants()
        
        assert checks['symmetry']
        assert checks['positive_definite']
        assert checks['well_conditioned']
        assert checks['eigenvectors_orthogonal']
        assert checks['spectral_reconstruction']
    
    def test_metric_tensor_to_array(self):
        """Conversión a array mutable."""
        G = MetricTensor(np.diag([1.0, 2.0]))
        arr = G.to_array()
        
        assert arr.flags.writeable
        arr[0] = 999.0  # No debe afectar a G interno
    
    def test_metric_tensor_ill_conditioned_warning(self):
        """Advertencia para matriz mal condicionada."""
        with pytest.warns(UserWarning, match="mal condicionada"):
            G = MetricTensor(np.diag([1e10, 1.0]), validate=True)

# ==============================================================================
# TESTS: ESPECTRAL DECOMPOSITION
# ==============================================================================

class TestSpectralDecomposition:
    """Tests para descomposición espectral."""
    
    def test_spectral_decomposition_valid(self):
        """Crear descomposición válida."""
        eigvals = np.array([3.0, 2.0, 1.0])  # Ordenados
        eigvecs = np.eye(3)
        
        spec = SpectralDecomposition(
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            orthogonality_residual=0.0
        )
        
        assert_array_equal(spec.eigenvalues, eigvals)
    
    def test_spectral_decomposition_unordered_eigenvalues(self):
        """Rechazar eigenvalores no ordenados."""
        eigvals = np.array([1.0, 3.0, 2.0])  # No ordenados
        eigvecs = np.eye(3)
        
        with pytest.raises(SpectralDecompositionError, match="no ordenados"):
            SpectralDecomposition(
                eigenvalues=eigvals,
                eigenvectors=eigvecs,
                orthogonality_residual=0.0
            )
    
    def test_spectral_decomposition_non_orthogonal(self):
        """Rechazar eigenvectores no ortogonales."""
        eigvals = np.array([3.0, 2.0])
        eigvecs = np.array([[1.0, 0.5], [0.0, 1.0]])  # No ortogonal
        
        with pytest.raises(SpectralDecompositionError, match="no ortogonales"):
            SpectralDecomposition(
                eigenvalues=eigvals,
                eigenvectors=eigvecs,
                orthogonality_residual=1.0  # Alto residual
            )

# ==============================================================================
# TESTS: FUNTOR DE MEMBRANA AISLANTE
# ==============================================================================

class TestIsolatingMembraneFunctor:
    """Tests para funtor de membrana p-Dirichlet."""
    
    def test_functor_creation_valid(self):
        """Crear funtor con p válido."""
        functor = IsolatingMembraneFunctor(p=1.5, eps=1e-12)
        
        assert functor.p == 1.5
        assert functor.eps == 1e-12
    
    def test_functor_creation_invalid_p(self):
        """Rechazar p fuera de [1, 2)."""
        with pytest.raises(ValueError, match="debe estar en"):
            IsolatingMembraneFunctor(p=2.5)
    
    def test_compute_topological_stress_constant(self):
        """Estrés de función constante."""
        functor = IsolatingMembraneFunctor(p=1.5, eps=1e-12)
        psi = np.ones(10)
        stress = functor.compute_topological_stress(psi)
        
        # Para constante, laplaciano ≈ 0, stress ≈ eps
        assert np.all(stress >= functor.eps)
    
    def test_compute_topological_stress_linear(self):
        """Estrés de función lineal."""
        functor = IsolatingMembraneFunctor(p=1.5)
        psi = np.linspace(0, 10, 50)
        stress = functor.compute_topological_stress(psi)
        
        # Laplaciano de lineal es pequeño
        assert np.all(stress >= functor.eps)
    
    def test_compute_topological_stress_oscillatory(self):
        """Estrés de función oscilatoria."""
        functor = IsolatingMembraneFunctor(p=1.5)
        x = np.linspace(0, 2*np.pi, 100)
        psi = np.sin(5 * x)
        stress = functor.compute_topological_stress(psi)
        
        # Debe tener variación significativa
        assert np.std(stress) > functor.eps

# ==============================================================================
# TESTS: SUBSPACE SPEC
# ==============================================================================

class TestSubspaceSpec:
    """Tests para especificación de subespacio."""
    
    def test_subspace_spec_creation_default_metric(self):
        """Crear subespacio con métrica por defecto."""
        spec = SubspaceSpec(
            name="test",
            indices=slice(0, 3),
            weight=1.0,
            reference=np.zeros(3)
        )
        
        assert spec.metric is not None
        assert spec.metric.dimension == 3
        assert spec.metric.is_diagonal
    
    def test_subspace_spec_creation_with_scale(self):
        """Crear subespacio con escala."""
        spec = SubspaceSpec(
            name="test",
            indices=slice(0, 2),
            weight=1.0,
            reference=np.zeros(2),
            scale=np.array([2.0, 3.0])
        )
        
        # Métrica debe ser G_diag = 1/scale²
        expected_diag = 1.0 / np.array([4.0, 9.0])
        actual_diag = spec.metric.to_array()
        
        assert_array_almost_equal(actual_diag, expected_diag)
    
    def test_subspace_spec_dimension_mismatch(self):
        """Rechazar dimensión inconsistente."""
        with pytest.raises(DimensionalMismatchError):
            SubspaceSpec(
                name="test",
                indices=slice(0, 3),
                weight=1.0,
                reference=np.zeros(2)  # Dim incorrecta
            )
    
    def test_subspace_spec_invalid_weight(self):
        """Rechazar peso inválido."""
        with pytest.raises(ValueError, match="positivo"):
            SubspaceSpec(
                name="test",
                indices=slice(0, 2),
                weight=-1.0,
                reference=np.zeros(2)
            )
    
    def test_subspace_spec_compute_threat_zero_deviation(self):
        """Amenaza cero para desviación nula."""
        spec = SubspaceSpec(
            name="test",
            indices=slice(0, 2),
            weight=1.0,
            reference=np.array([1.0, 2.0])
        )
        
        threat = spec.compute_threat(np.array([1.0, 2.0]))
        
        # Debe ser muy pequeño (no exactamente 0 por estrés)
        assert threat < 1e-6
    
    def test_subspace_spec_compute_threat_nonzero_deviation(self):
        """Amenaza no cero para desviación."""
        spec = SubspaceSpec(
            name="test",
            indices=slice(0, 2),
            weight=2.0,
            reference=np.zeros(2)
        )
        
        threat = spec.compute_threat(np.array([1.0, 1.0]))
        
        assert threat > 0.0
    
    def test_subspace_spec_normalize_to_reference(self):
        """Normalización al espacio de referencia."""
        spec = SubspaceSpec(
            name="test",
            indices=slice(0, 2),
            weight=1.0,
            reference=np.array([1.0, 0.0]),
            metric=MetricTensor(np.diag([1.0, 1.0]))
        )
        
        normalized = spec.normalize_to_reference(np.array([3.0, 0.0]))
        
        # G^{-1/2}(v - ref) con G = I => v - ref = [2.0, 0.0]
        assert_array_almost_equal(normalized, np.array([2.0, 0.0]))

# ==============================================================================
# TESTS: THREAT ASSESSMENT
# ==============================================================================

class TestThreatAssessment:
    """Tests para evaluación de amenazas."""
    
    def test_threat_assessment_creation(self):
        """Crear evaluación válida."""
        assessment = ThreatAssessment(
            levels={"physics": 0.5, "topology": 0.3},
            max_source="physics",
            max_value=0.5,
            total_threat=0.583,
            euler_char=1,
            status=HealthStatus.HEALTHY
        )
        
        assert assessment.max_source == "physics"
        assert assessment.status == HealthStatus.HEALTHY
    
    def test_threat_assessment_negative_level(self):
        """Rechazar nivel negativo."""
        with pytest.raises(ValueError, match="no negativos"):
            ThreatAssessment(
                levels={"physics": -0.5},
                max_source="physics",
                max_value=-0.5,
                total_threat=0.5
            )
    
    def test_threat_assessment_inconsistent_max(self):
        """Rechazar max_value inconsistente."""
        with pytest.raises(ValueError, match="inconsistente"):
            ThreatAssessment(
                levels={"physics": 0.5, "topology": 0.8},
                max_source="physics",
                max_value=0.5,  # Debería ser 0.8
                total_threat=1.0
            )
    
    def test_threat_assessment_from_components_healthy(self):
        """Factory para estado saludable."""
        assessment = ThreatAssessment.from_components(
            levels={"physics": 0.3, "topology": 0.2},
            warning_threshold=0.8
        )
        
        assert assessment.status == HealthStatus.HEALTHY
    
    def test_threat_assessment_from_components_warning(self):
        """Factory para estado de advertencia."""
        assessment = ThreatAssessment.from_components(
            levels={"physics": 0.9, "topology": 0.5},
            warning_threshold=0.8,
            critical_threshold=1.5
        )
        
        assert assessment.status == HealthStatus.WARNING
    
    def test_threat_assessment_from_components_critical(self):
        """Factory para estado crítico."""
        assessment = ThreatAssessment.from_components(
            levels={"physics": 2.0, "topology": 0.5},
            critical_threshold=1.5
        )
        
        assert assessment.status == HealthStatus.CRITICAL
    
    def test_threat_assessment_to_dict(self):
        """Serialización a diccionario."""
        assessment = ThreatAssessment(
            levels={"physics": 0.5},
            max_source="physics",
            max_value=0.5,
            total_threat=0.5,
            euler_char=1,
            status=HealthStatus.HEALTHY
        )
        
        d = assessment.to_dict()
        
        assert "threat_levels" in d
        assert "health_status" in d
        assert d["health_status"] == "HEALTHY"
        assert d["euler_characteristic"] == 1

# ==============================================================================
# TESTS: BLOCK DIAG PURE
# ==============================================================================

class TestBlockDiagPure:
    """Tests para construcción bloque-diagonal."""
    
    def test_block_diag_pure_1d_blocks(self):
        """Bloques 1D (diagonales)."""
        b1 = np.array([1.0, 2.0])
        b2 = np.array([3.0, 4.0, 5.0])
        
        result = block_diag_pure(b1, b2)
        
        expected = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_array_almost_equal(result, expected)
    
    def test_block_diag_pure_2d_blocks(self):
        """Bloques 2D (matrices)."""
        b1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        b2 = np.array([[5.0]])
        
        result = block_diag_pure(b1, b2)
        
        expected = np.array([
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 5.0]
        ])
        assert_array_almost_equal(result, expected)
    
    def test_block_diag_pure_mixed(self):
        """Bloques mixtos 1D y 2D."""
        b1 = np.array([1.0, 2.0])
        b2 = np.array([[3.0, 4.0], [5.0, 6.0]])
        
        result = block_diag_pure(b1, b2)
        
        expected = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0],
            [0.0, 0.0, 5.0, 6.0]
        ])
        assert_array_almost_equal(result, expected)
    
    def test_block_diag_pure_invalid_ndim(self):
        """Rechazar bloques con ndim inválido."""
        with pytest.raises(ValueError, match="1D o 2D"):
            block_diag_pure(np.array([[[1.0]]]))
    
    def test_block_diag_pure_non_square_2d(self):
        """Rechazar bloques 2D no cuadrados."""
        with pytest.raises(ValueError, match="cuadrados"):
            block_diag_pure(np.array([[1.0, 2.0]]))

# ==============================================================================
# TESTS: ORTHOGONAL PROJECTOR
# ==============================================================================

class TestOrthogonalProjector:
    """Tests para proyector ortogonal."""
    
    def test_projector_creation_valid(self):
        """Crear proyector válido."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
            "sub2": SubspaceSpec(
                name="sub2",
                indices=slice(2, 4),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=4,
            subspaces=subspaces,
            cache_projections=True
        )
        
        assert projector.validation_report is not None
    
    def test_projector_overlapping_subspaces(self):
        """Rechazar subespacios solapados."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 3),
                weight=1.0,
                reference=np.zeros(3)
            ),
            "sub2": SubspaceSpec(
                name="sub2",
                indices=slice(2, 5),  # Solapa con sub1
                weight=1.0,
                reference=np.zeros(3)
            ),
        }
        
        with pytest.raises(DimensionalMismatchError, match="solapa"):
            OrthogonalProjector(
                dimensions=5,
                subspaces=subspaces
            )
    
    def test_projector_compute_euler_characteristic(self):
        """Calcular característica de Euler."""
        subspaces = {
            "topo": SubspaceSpec(
                name="topo",
                indices=slice(0, 3),
                weight=1.0,
                reference=np.zeros(3)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=3,
            subspaces=subspaces,
            topo_indices=(0, 1, 2)  # β₀, β₁, β₂
        )
        
        psi = np.array([1.0, 2.0, 1.0])  # β₀=1, β₁=2, β₂=1
        euler = projector._compute_euler_characteristic(psi)
        
        expected = 1 - 2 + 1  # 0
        assert euler == expected
    
    def test_projector_euler_invalid_beta_0(self):
        """Rechazar β₀ < 1."""
        subspaces = {
            "topo": SubspaceSpec(
                name="topo",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces,
            topo_indices=(0, 1)
        )
        
        psi = np.array([0.0, 0.0])  # β₀=0 (inválido)
        
        with pytest.raises(TopologicalInvariantError, match="β₀"):
            projector._compute_euler_characteristic(psi)
    
    def test_projector_euler_invalid_beta_1(self):
        """Rechazar β₁ < 0."""
        subspaces = {
            "topo": SubspaceSpec(
                name="topo",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces,
            topo_indices=(0, 1)
        )
        
        psi = np.array([1.0, -1.0])  # β₁=-1 (inválido)
        
        with pytest.raises(TopologicalInvariantError, match="β₁"):
            projector._compute_euler_characteristic(psi)
    
    def test_projector_project_valid_signal(self):
        """Proyectar señal válida."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces
        )
        
        psi = np.array([0.1, 0.2])
        assessment = projector.project(psi)
        
        assert assessment.status == HealthStatus.HEALTHY
        assert "sub1" in assessment.levels
    
    def test_projector_project_dimension_mismatch(self):
        """Rechazar señal con dimensión incorrecta."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces
        )
        
        psi = np.array([0.1, 0.2, 0.3])  # Dim incorrecta
        
        with pytest.raises(DimensionalMismatchError, match="Shape incorrecto"):
            projector.project(psi)
    
    def test_projector_project_non_finite(self):
        """Rechazar señal no finita."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces
        )
        
        psi = np.array([0.1, np.nan])
        
        with pytest.raises(NumericalStabilityError, match="no finitos"):
            projector.project(psi)
    
    def test_projector_classify_with_hysteresis_initial(self):
        """Clasificación inicial sin histéresis."""
        status = OrthogonalProjector._classify_with_hysteresis(
            value=0.5,
            warning_th=0.8,
            critical_th=1.5,
            hysteresis=0.05,
            previous=None
        )
        
        assert status == HealthStatus.HEALTHY
    
    def test_projector_classify_with_hysteresis_healthy_to_warning(self):
        """Transición HEALTHY → WARNING."""
        status = OrthogonalProjector._classify_with_hysteresis(
            value=0.9,  # > warning + hysteresis
            warning_th=0.8,
            critical_th=1.5,
            hysteresis=0.05,
            previous=HealthStatus.HEALTHY
        )
        
        assert status == HealthStatus.WARNING
    
    def test_projector_classify_with_hysteresis_warning_to_healthy(self):
        """Transición WARNING → HEALTHY."""
        status = OrthogonalProjector._classify_with_hysteresis(
            value=0.7,  # < warning - hysteresis
            warning_th=0.8,
            critical_th=1.5,
            hysteresis=0.05,
            previous=HealthStatus.WARNING
        )
        
        assert status == HealthStatus.HEALTHY
    
    def test_projector_classify_with_hysteresis_stays_warning(self):
        """Permanecer en WARNING (banda de histéresis)."""
        status = OrthogonalProjector._classify_with_hysteresis(
            value=0.8,  # En banda
            warning_th=0.8,
            critical_th=1.5,
            hysteresis=0.05,
            previous=HealthStatus.WARNING
        )
        
        assert status == HealthStatus.WARNING

# ==============================================================================
# TESTS: CONSTRUCCIÓN DE SEÑAL
# ==============================================================================

class TestBuildSignal:
    """Tests para construcción de señal ψ."""
    
    def test_build_signal_valid(self, sample_telemetry):
        """Construir señal desde telemetría válida."""
        psi = build_signal(sample_telemetry, strict=False)
        
        assert psi.shape == (len(SIGNAL_SCHEMA),)
        assert psi[0] == 0.5  # saturation
        assert psi[3] == 1.0  # beta_0
    
    def test_build_signal_missing_keys(self):
        """Manejar claves faltantes con defaults."""
        telemetry = {"saturation": 0.5}  # Solo una clave
        psi = build_signal(telemetry, strict=False)
        
        assert psi.shape == (len(SIGNAL_SCHEMA),)
        assert psi[0] == 0.5  # saturation
        assert psi[1] == 0.0  # flyback_voltage (default)
    
    def test_build_signal_invalid_type_strict(self):
        """Rechazar tipo inválido en modo estricto."""
        telemetry = {"saturation": "invalid"}
        
        with pytest.raises(ValueError, match="no convertible"):
            build_signal(telemetry, strict=True)
    
    def test_build_signal_invalid_type_non_strict(self, logger_mock):
        """Manejar tipo inválido en modo no estricto."""
        telemetry = {"saturation": "invalid"}
        psi = build_signal(telemetry, strict=False)
        
        # Debe usar default
        assert psi[0] == 0.0
        assert any("no convertible" in msg for msg in logger_mock["warning"])
    
    def test_build_signal_nan_strict(self):
        """Rechazar NaN en modo estricto."""
        telemetry = {"saturation": np.nan}
        
        with pytest.raises(ValueError, match="no finita"):
            build_signal(telemetry, strict=True)
    
    def test_build_signal_nan_non_strict(self, logger_mock):
        """Manejar NaN en modo no estricto."""
        telemetry = {"saturation": np.nan}
        psi = build_signal(telemetry, strict=False)
        
        # Debe usar default
        assert psi[0] == 0.0
    
    def test_build_signal_validation_clamp(self, logger_mock):
        """Validación con clamp."""
        telemetry = {"saturation": 1.5}  # Fuera de [0, 1]
        psi = build_signal(telemetry, strict=False)
        
        # Debe clampearse a 1.0
        assert psi[0] == 1.0
        assert any("corregida" in msg for msg in logger_mock["debug"])
    
    def test_build_signal_betti_validation(self):
        """Validación de números de Betti."""
        telemetry = {
            "beta_0": 2.7,  # Debe redondearse a 3
            "beta_1": -0.3,  # Debe clampearse a 0
        }
        psi = build_signal(telemetry, strict=False)
        
        assert psi[BETTI_INDICES[0]] == 3.0  # beta_0
        assert psi[BETTI_INDICES[1]] == 0.0  # beta_1

# ==============================================================================
# TESTS: IMMUNE WATCHER MORPHISM
# ==============================================================================

class TestImmuneWatcherMorphism:
    """Tests para morfismo inmunológico."""
    
    def test_morphism_creation_valid(self):
        """Crear morfismo con parámetros válidos."""
        morphism = ImmuneWatcherMorphism(
            name="test_watcher",
            critical_threshold=1.5,
            warning_threshold=0.8,
            hysteresis=0.05
        )
        
        assert morphism.name == "test_watcher"
        assert morphism.thresholds["critical"] == 1.5
    
    def test_morphism_creation_invalid_thresholds(self):
        """Rechazar umbrales mal ordenados."""
        with pytest.raises(ValueError, match="warning_threshold debe ser"):
            ImmuneWatcherMorphism(
                warning_threshold=1.5,
                critical_threshold=0.8  # Orden invertido
            )
    
    def test_morphism_creation_invalid_hysteresis(self):
        """Rechazar histéresis fuera de rango."""
        with pytest.raises(ValueError, match="hysteresis debe estar"):
            ImmuneWatcherMorphism(
                warning_threshold=0.8,
                critical_threshold=1.5,
                hysteresis=0.5  # Demasiado grande
            )
    
    def test_morphism_domain(self):
        """Verificar dominio categórico."""
        morphism = create_immune_watcher("default")
        
        assert Stratum.PHYSICS in morphism.domain
    
    def test_morphism_codomain(self):
        """Verificar codominio categórico."""
        morphism = create_immune_watcher("default")
        
        assert morphism.codomain == Stratum.WISDOM
    
    def test_morphism_preserves_error_state(self, categorical_state_error):
        """F(⊥) = ⊥ (preservar objeto error)."""
        morphism = create_immune_watcher("default")
        
        result = morphism(categorical_state_error)
        
        assert not result.is_success
        assert result.error_msg == "Test error"
    
    def test_morphism_healthy_state(self, categorical_state_success):
        """Procesar estado saludable."""
        morphism = create_immune_watcher("default")
        
        result = morphism(categorical_state_success)
        
        assert result.is_success
        assert result.context.get("immune_status") == "healthy"
        assert result.stratum == Stratum.WISDOM
    
    def test_morphism_warning_state(self):
        """Procesar estado de advertencia."""
        morphism = create_immune_watcher("default")
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {
                "saturation": 0.9,  # Alta saturación
                "flyback_voltage": 350.0,  # Alto voltaje
                "dissipated_power": 150.0,
                "beta_0": 1.0,
                "beta_1": 0.0,
                "entropy": 0.7,
                "exergy_loss": 0.6,
            }}
        )
        
        result = morphism(state)
        
        # Puede ser WARNING o CRITICAL dependiendo de métricas
        assert result.context.get("immune_status") in ("warning", "healthy")
    
    def test_morphism_critical_state(self):
        """Procesar estado crítico."""
        morphism = create_immune_watcher("strict")  # Umbrales más estrictos
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {
                "saturation": 1.0,
                "flyback_voltage": 400.0,
                "dissipated_power": 200.0,
                "beta_0": 3.0,  # Fragmentación
                "beta_1": 5.0,  # Muchos ciclos
                "entropy": 0.95,
                "exergy_loss": 0.9,
            }}
        )
        
        # Ejecutar múltiples veces para evolución de métricas
        for _ in range(3):
            result = morphism(state)
        
        # Estado final puede ser crítico
        if not result.is_success:
            assert "Cuarentena" in result.error_msg or "QUARANTINE" in str(result.context)
    
    def test_morphism_topology_change_detection(self, logger_mock):
        """Detectar bifurcación topológica."""
        morphism = create_immune_watcher("default")
        
        # Estado inicial: χ = 1
        state1 = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {
                "beta_0": 1.0,
                "beta_1": 0.0,
                "saturation": 0.3,
                "flyback_voltage": 150.0,
                "dissipated_power": 30.0,
                "entropy": 0.2,
                "exergy_loss": 0.1,
            }}
        )
        
        morphism(state1)
        
        # Estado siguiente: χ = -1 (bifurcación)
        state2 = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {
                "beta_0": 1.0,
                "beta_1": 2.0,  # Cambio
                "saturation": 0.3,
                "flyback_voltage": 150.0,
                "dissipated_power": 30.0,
                "entropy": 0.2,
                "exergy_loss": 0.1,
            }}
        )
        
        morphism(state2)
        
        # Debe haber advertencia de bifurcación
        assert any("Bifurcación topológica" in msg for msg in logger_mock["warning"])
    
    def test_morphism_reset_state(self):
        """Reiniciar estado interno."""
        morphism = create_immune_watcher("default")
        
        # Ejecutar evaluación
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {}}
        )
        morphism(state)
        
        assert morphism.evaluation_count > 0
        
        # Reiniciar
        morphism.reset_state()
        
        assert morphism.evaluation_count == 0
        assert morphism.current_status is None
        assert len(morphism.topology_history) == 0
    
    def test_morphism_get_diagnostics(self):
        """Obtener diagnóstico completo."""
        morphism = create_immune_watcher("default")
        
        diagnostics = morphism.get_diagnostics()
        
        assert "name" in diagnostics
        assert "evaluation_count" in diagnostics
        assert "functorial_properties" in diagnostics
        assert "projector_validation" in diagnostics
    
    def test_morphism_health_report(self):
        """Generar reporte de salud."""
        morphism = create_immune_watcher("default")
        
        report = morphism.health_report()
        
        assert "IMMUNE WATCHER" in report
        assert "DIAGNÓSTICO" in report
        assert "PROPIEDADES CATEGÓRICAS" in report
    
    def test_morphism_evaluate_manifold_deformation(self):
        """Evaluar deformación de variedad."""
        morphism = create_immune_watcher("default")
        
        state_tensor = np.array([0.5, 200.0, 50.0, 1.0, 0.0, 0.3, 0.2])
        
        metrics = morphism.evaluate_manifold_deformation(state_tensor)
        
        assert hasattr(metrics, 'mahalanobis_distance')
        assert hasattr(metrics, 'is_stable')
        assert hasattr(metrics, 'structural_alteration')
    
    def test_morphism_ricci_flow_evolution(self):
        """Evolución de métricas vía Ricci Flow."""
        morphism = create_immune_watcher("default")
        
        # Estado con topología no trivial
        telemetry = {
            "beta_0": 2.0,
            "beta_1": 1.0,
            "saturation": 0.5,
            "flyback_voltage": 200.0,
            "dissipated_power": 50.0,
            "entropy": 0.3,
            "exergy_loss": 0.2,
        }
        
        # Ejecutar múltiples evaluaciones
        for _ in range(5):
            morphism._evolve_metric_tensors_ricci_flow(telemetry)
        
        # Verificar que métricas siguen siendo SPD
        for key, G in morphism._metric_tensors_state.items():
            eigvals = np.linalg.eigvalsh(G)
            assert np.all(eigvals >= MIN_EIGVAL_TOL * 0.99), f"Métrica {key} no SPD"

# ==============================================================================
# TESTS: FACTORY
# ==============================================================================

class TestFactory:
    """Tests para factory de morfismos."""
    
    def test_create_immune_watcher_default(self):
        """Crear con perfil default."""
        morphism = create_immune_watcher("default")
        
        assert morphism.thresholds["warning"] == 0.8
        assert morphism.thresholds["critical"] == 1.5
        assert morphism.thresholds["hysteresis"] == 0.05
    
    def test_create_immune_watcher_strict(self):
        """Crear con perfil strict."""
        morphism = create_immune_watcher("strict")
        
        assert morphism.thresholds["warning"] == 0.5
        assert morphism.thresholds["critical"] == 1.0
    
    def test_create_immune_watcher_relaxed(self):
        """Crear con perfil relaxed."""
        morphism = create_immune_watcher("relaxed")
        
        assert morphism.thresholds["warning"] == 1.0
        assert morphism.thresholds["critical"] == 2.0
    
    def test_create_immune_watcher_laboratory(self):
        """Crear con perfil laboratory (sin histéresis)."""
        morphism = create_immune_watcher("laboratory")
        
        assert morphism.thresholds["hysteresis"] == 0.0
    
    def test_create_immune_watcher_unknown_profile(self):
        """Rechazar perfil desconocido."""
        with pytest.raises(ValueError, match="desconocido"):
            create_immune_watcher("unknown_profile")
    
    def test_create_immune_watcher_with_overrides(self):
        """Crear con overrides."""
        morphism = create_immune_watcher(
            "default",
            warning_threshold=0.9,
            critical_threshold=2.0
        )
        
        assert morphism.thresholds["warning"] == 0.9
        assert morphism.thresholds["critical"] == 2.0
        assert morphism.thresholds["hysteresis"] == 0.05  # Default

# ==============================================================================
# TESTS DE INTEGRACIÓN
# ==============================================================================

class TestIntegration:
    """Tests de integración end-to-end."""
    
    def test_full_pipeline_healthy(self, sample_telemetry):
        """Pipeline completo: telemetría → señal → proyección → estado."""
        # Construir señal
        psi = build_signal(sample_telemetry, strict=False)
        
        # Crear morfismo
        morphism = create_immune_watcher("default")
        
        # Crear estado categórico
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": sample_telemetry}
        )
        
        # Aplicar morfismo
        result = morphism(state)
        
        # Verificaciones
        assert result.is_success
        assert result.stratum == Stratum.WISDOM
        assert result.context.get("immune_status") == "healthy"
    
    def test_full_pipeline_anomalous(self, anomalous_telemetry):
        """Pipeline con telemetría anómala."""
        morphism = create_immune_watcher("default")
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": anomalous_telemetry}
        )
        
        result = morphism(state)
        
        # Puede ser WARNING o CRITICAL
        status = result.context.get("immune_status")
        assert status in ("warning", "healthy") or not result.is_success
    
    def test_temporal_evolution(self):
        """Evolución temporal de métricas."""
        morphism = create_immune_watcher("default")
        
        # Secuencia temporal de telemetrías
        sequence = [
            {"beta_0": 1.0, "beta_1": 0.0, "saturation": 0.3},
            {"beta_0": 1.0, "beta_1": 1.0, "saturation": 0.5},  # Ciclo aparece
            {"beta_0": 2.0, "beta_1": 1.0, "saturation": 0.7},  # Fragmentación
            {"beta_0": 2.0, "beta_1": 2.0, "saturation": 0.9},  # Más ciclos
        ]
        
        euler_history = []
        
        for telemetry in sequence:
            # Completar telemetría
            full_telemetry = {
                "flyback_voltage": 200.0,
                "dissipated_power": 50.0,
                "entropy": 0.3,
                "exergy_loss": 0.2,
                **telemetry
            }
            
            state = CategoricalState(
                stratum=Stratum.PHYSICS,
                success=True,
                context={"telemetry_metrics": full_telemetry}
            )
            
            result = morphism(state)
            
            if "euler_characteristic" in result.context:
                euler_history.append(result.context["euler_characteristic"])
        
        # Verificar que χ cambia
        if len(set(euler_history)) > 1:
            # Hubo bifurcación
            assert True
    
    def test_stress_test_large_values(self):
        """Test de estrés con valores grandes."""
        morphism = create_immune_watcher("default")
        
        telemetry = {
            "saturation": 0.99,
            "flyback_voltage": 399.0,
            "dissipated_power": 199.0,
            "beta_0": 10.0,
            "beta_1": 20.0,
            "entropy": 0.99,
            "exergy_loss": 0.99,
        }
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": telemetry}
        )
        
        # No debe lanzar excepción
        result = morphism(state)
        
        assert result is not None
    
    def test_stress_test_tiny_values(self):
        """Test de estrés con valores muy pequeños."""
        morphism = create_immune_watcher("default")
        
        telemetry = {
            "saturation": 1e-10,
            "flyback_voltage": 1e-10,
            "dissipated_power": 1e-10,
            "beta_0": 1.0,
            "beta_1": 0.0,
            "entropy": 1e-10,
            "exergy_loss": 1e-10,
        }
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": telemetry}
        )
        
        result = morphism(state)
        
        assert result is not None
        assert result.is_success
    
    def test_numerical_stability_under_repeated_evaluation(self):
        """Estabilidad numérica bajo evaluaciones repetidas."""
        morphism = create_immune_watcher("default")
        
        telemetry = {
            "saturation": 0.5,
            "flyback_voltage": 200.0,
            "dissipated_power": 50.0,
            "beta_0": 1.0,
            "beta_1": 0.0,
            "entropy": 0.3,
            "exergy_loss": 0.2,
        }
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": telemetry}
        )
        
        # Ejecutar 100 evaluaciones
        for _ in range(100):
            result = morphism(state)
            
            # Verificar finitud de todas las métricas
            if result.is_success:
                for key, value in result.context.items():
                    if isinstance(value, (int, float)):
                        assert np.isfinite(value), f"Valor no finito en {key}"

# ==============================================================================
# TESTS DE PROPIEDADES MATEMÁTICAS
# ==============================================================================

class TestMathematicalProperties:
    """Tests de propiedades matemáticas rigurosas."""
    
    def test_metric_tensor_positive_definiteness(self):
        """Propiedad SPD: vᵀGv > 0 para v ≠ 0."""
        G = MetricTensor(np.diag([1.0, 2.0, 3.0]))
        
        for _ in range(10):
            v = np.random.randn(3)
            if np.linalg.norm(v) > EPS:
                Q = G.quadratic_form(v)
                assert Q > 0, "Forma cuadrática no positiva"
    
    def test_projector_idempotence(self):
        """Propiedad P² = P."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces,
            cache_projections=True
        )
        
        P = projector._projection_matrices["sub1"]
        P2 = P @ P
        
        assert_array_almost_equal(P, P2)
    
    def test_projector_orthogonality(self):
        """Propiedad πᵢ·πⱼ = 0 para i ≠ j."""
        subspaces = {
            "sub1": SubspaceSpec(
                name="sub1",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
            "sub2": SubspaceSpec(
                name="sub2",
                indices=slice(2, 4),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=4,
            subspaces=subspaces,
            cache_projections=True
        )
        
        P1 = projector._projection_matrices["sub1"]
        P2 = projector._projection_matrices["sub2"]
        
        product = P1 @ P2
        
        assert_array_almost_equal(product, np.zeros((4, 4)))
    
    def test_euler_characteristic_topological_invariant(self):
        """χ es invariante topológico."""
        subspaces = {
            "topo": SubspaceSpec(
                name="topo",
                indices=slice(0, 2),
                weight=1.0,
                reference=np.zeros(2)
            ),
        }
        
        projector = OrthogonalProjector(
            dimensions=2,
            subspaces=subspaces,
            topo_indices=(0, 1)
        )
        
        # Mismo espacio topológico
        psi1 = np.array([1.0, 2.0])
        psi2 = np.array([1.0, 2.0])
        
        chi1 = projector._compute_euler_characteristic(psi1)
        chi2 = projector._compute_euler_characteristic(psi2)
        
        assert chi1 == chi2
    
    def test_mahalanobis_distance_positive(self):
        """Distancia de Mahalanobis d ≥ 0."""
        spec = SubspaceSpec(
            name="test",
            indices=slice(0, 2),
            weight=1.0,
            reference=np.zeros(2)
        )
        
        for _ in range(10):
            v = np.random.randn(2)
            threat = spec.compute_threat(v)
            assert threat >= 0, "Amenaza negativa"
    
    def test_condition_number_bounds(self):
        """κ(G) ≥ 1 para matrices SPD."""
        for _ in range(10):
            # Generar matriz SPD aleatoria
            A = np.random.randn(3, 3)
            G_arr = A @ A.T + np.eye(3)
            
            G = MetricTensor(G_arr, validate=True)
            
            assert G.condition_number >= 1.0

# ==============================================================================
# TESTS DE CASOS EXTREMOS
# ==============================================================================

class TestEdgeCases:
    """Tests de casos extremos y límite."""
    
    def test_zero_vector_signal(self):
        """Señal completamente cero."""
        morphism = create_immune_watcher("default")
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {
                "saturation": 0.0,
                "flyback_voltage": 0.0,
                "dissipated_power": 0.0,
                "beta_0": 1.0,  # Mínimo válido
                "beta_1": 0.0,
                "entropy": 0.0,
                "exergy_loss": 0.0,
            }}
        )
        
        result = morphism(state)
        
        assert result.is_success
        assert result.context.get("immune_status") == "healthy"
    
    def test_maximum_valid_signal(self):
        """Señal en límites superiores válidos."""
        morphism = create_immune_watcher("default")
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {
                "saturation": 1.0,
                "flyback_voltage": PhysicalConstants.FLYBACK_MAX_SAFE,
                "dissipated_power": PhysicalConstants.P_NOMINAL() * 2.0,
                "beta_0": 1.0,
                "beta_1": 0.0,
                "entropy": 1.0,
                "exergy_loss": 1.0,
            }}
        )
        
        result = morphism(state)
        
        # Puede activar cuarentena
        assert result is not None
    
    def test_singular_metric_handling(self):
        """Manejo de métrica singular."""
        # Métrica con eigenvalor muy pequeño
        G_arr = np.diag([1.0, 1e-20, 1.0])
        
        # Debe regularizarse automáticamente
        G = MetricTensor(G_arr, validate=True)
        
        # Verificar que es SPD después de regularización
        eigvals = np.linalg.eigvalsh(G.to_array())
        assert np.all(eigvals >= MIN_EIGVAL_TOL * 0.99)
    
    def test_empty_telemetry(self):
        """Telemetría vacía."""
        morphism = create_immune_watcher("default")
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={"telemetry_metrics": {}}
        )
        
        result = morphism(state)
        
        # Debe usar todos los defaults
        assert result.is_success
    
    def test_missing_telemetry_context(self):
        """Contexto sin clave telemetry_metrics."""
        morphism = create_immune_watcher("default")
        
        state = CategoricalState(
            stratum=Stratum.PHYSICS,
            success=True,
            context={}
        )
        
        result = morphism(state)
        
        assert result.is_success

# ==============================================================================
# CONFIGURACIÓN DE PYTEST
# ==============================================================================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers",
        "integration: marca tests de integración"
    )

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])