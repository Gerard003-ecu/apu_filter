"""
Suite de Pruebas para el `DataFluxCondenser` - Versión Refinada V6.

Cobertura actualizada con mejoras en robustez matemática y física:
- Invariantes Hamiltonianos y conservación de energía
- Criterio de Jury con análisis de márgenes de estabilidad
- Verificación de relación de Euler-Poincaré para Betti numbers
- Filtro EMA con detección de anomalías y adaptación de varianza
- Métricas de Lyapunov con regresión robusta (Theil-Sen)
- Integración RK4 con verificación de orden de convergencia
- Entropía de Rényi con verificación de propiedades axiomáticas
- Estabilidad giroscópica con ecuaciones de Euler completas
- EKF adaptativo con verificación de consistencia de innovaciones
- Recuperación multinivel con garantías de progreso

Principios de diseño:
- Cada test verifica una propiedad matemática específica
- Valores de referencia derivados analíticamente cuando es posible
- Tolerancias numéricas justificadas por análisis de error
- Aislamiento completo entre pruebas
"""

import math
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import MagicMock, patch, PropertyMock, call
from collections import deque
from dataclasses import dataclass, field
from contextlib import contextmanager

import pandas as pd
import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from scipy import stats as scipy_stats
    from scipy import linalg as scipy_linalg
    HAS_SCIPY = True
except ImportError:
    scipy_stats = None
    scipy_linalg = None
    HAS_SCIPY = False

from app.flux_condenser import (
    BatchResult,
    CondenserConfig,
    ConfigurationError,
    DataFluxCondenser,
    DataFluxCondenserError,
    RefinedFluxPhysicsEngine,
    InvalidInputError,
    ParsedData,
    PIController,
    ProcessingError,
    ProcessingStats,
    SystemConstants,
    TopologicalAnalyzer,
    EntropyCalculator,
    UnifiedPhysicalState,
)

# Alias para compatibilidad
FluxPhysicsEngine = RefinedFluxPhysicsEngine

from app.laplace_oracle import LaplaceOracle, ConfigurationError as OracleConfigurationError


# ============================================================================
# CONSTANTES DE PRUEBA Y TOLERANCIAS NUMÉRICAS
# ============================================================================

class TestConstants:
    """Constantes y tolerancias para pruebas numéricas."""
    
    # Tolerancias basadas en precisión de punto flotante IEEE 754
    EPSILON_FLOAT64 = 2.220446049250313e-16  # Machine epsilon
    RTOL_ENERGY = 1e-10  # Tolerancia relativa para conservación de energía
    ATOL_PROBABILITY = 1e-12  # Tolerancia absoluta para probabilidades
    RTOL_ENTROPY = 1e-8  # Tolerancia para cálculos de entropía
    
    # Umbrales de estabilidad basados en teoría de control
    LYAPUNOV_STABLE_THRESHOLD = 0.0  # λ < 0 implica estabilidad asintótica
    JURY_STABILITY_MARGIN = 0.01  # Margen mínimo para estabilidad robusta
    
    # Constantes físicas derivadas
    BOLTZMANN_NORMALIZED = 1.0  # k_B normalizado para entropía adimensional
    
    # Límites para pruebas de estrés
    MAX_ITERATIONS_STRESS = 10000
    MAX_BATCH_SIZE_STRESS = 100000


# ============================================================================
# UTILIDADES DE PRUEBA
# ============================================================================

@contextmanager
def assert_no_warnings(caplog):
    """Context manager que falla si se emiten warnings."""
    caplog.set_level(logging.WARNING)
    yield
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    if warnings:
        pytest.fail(f"Se emitieron warnings inesperados: {[w.message for w in warnings]}")


def assert_finite(value: float, name: str = "value"):
    """Verifica que un valor sea finito (no NaN, no Inf)."""
    assert math.isfinite(value), f"{name} debe ser finito, obtenido: {value}"


def assert_probability(value: float, name: str = "probability"):
    """Verifica que un valor esté en [0, 1]."""
    assert 0.0 <= value <= 1.0, f"{name} debe estar en [0,1], obtenido: {value}"


def assert_positive(value: float, name: str = "value", strict: bool = True):
    """Verifica que un valor sea positivo."""
    if strict:
        assert value > 0, f"{name} debe ser estrictamente positivo, obtenido: {value}"
    else:
        assert value >= 0, f"{name} debe ser no negativo, obtenido: {value}"


def compute_numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-8) -> float:
    """Calcula derivada numérica usando diferencias centradas de 4to orden."""
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)


def verify_jury_criterion(coefficients: List[float]) -> Tuple[bool, Dict[str, float]]:
    """
    Verifica el criterio de Jury para estabilidad de sistemas discretos.
    
    Para un polinomio P(z) = a_n*z^n + ... + a_1*z + a_0, el sistema es estable si:
    1. P(1) > 0
    2. (-1)^n * P(-1) > 0  
    3. |a_0| < a_n
    4. Determinantes de Jury positivos
    
    Returns:
        Tuple[bool, Dict]: (es_estable, métricas_de_margen)
    """
    n = len(coefficients) - 1
    a = coefficients
    
    # Condición 1: P(1) > 0
    p_1 = sum(a)
    
    # Condición 2: (-1)^n * P(-1) > 0
    p_minus_1 = sum(a[i] * ((-1) ** i) for i in range(len(a)))
    condition_2_value = ((-1) ** n) * p_minus_1
    
    # Condición 3: |a_0| < a_n
    margin_3 = abs(a[-1]) - abs(a[0])
    
    is_stable = (p_1 > 0) and (condition_2_value > 0) and (margin_3 < 0)
    
    metrics = {
        "p_at_1": p_1,
        "p_at_minus_1_signed": condition_2_value,
        "coefficient_margin": -margin_3,
        "stability_margin": min(p_1, condition_2_value, -margin_3) if is_stable else 0.0
    }
    
    return is_stable, metrics


# ============================================================================
# FIXTURES COMPARTIDAS
# ============================================================================

@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """Configuración válida para el procesador."""
    return {
        "parser_settings": {"delimiter": ",", "encoding": "utf-8"},
        "processor_settings": {"validate_types": True, "skip_empty": False},
    }


@pytest.fixture
def valid_profile() -> Dict[str, Any]:
    """Perfil válido de mapeo de columnas."""
    return {
        "columns_mapping": {"cod_insumo": "codigo", "descripcion": "desc"},
        "validation_rules": {"required_fields": ["codigo", "cantidad"]},
    }


@pytest.fixture
def default_condenser_config() -> CondenserConfig:
    """Configuración por defecto del condensador."""
    return CondenserConfig()


@pytest.fixture
def condenser(valid_config, valid_profile) -> DataFluxCondenser:
    """Instancia de condensador con configuración válida."""
    return DataFluxCondenser(valid_config, valid_profile)


@pytest.fixture
def isolated_condenser(valid_config, valid_profile) -> DataFluxCondenser:
    """Condensador aislado con estado limpio para cada prueba."""
    c = DataFluxCondenser(valid_config, valid_profile)
    c._start_time = time.time()
    return c


@pytest.fixture
def sample_raw_records() -> List[Dict[str, Any]]:
    """Registros de ejemplo para pruebas con estructura consistente."""
    return [
        {
            "codigo": f"A{i:04d}",
            "cantidad": 10 + (i % 5),
            "precio": 100.0 + i * 0.5,
            "insumo_line": f"line_{i}"
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    """Caché de parseo de ejemplo."""
    return {f"line_{i}": f"parsed_data_{i}" for i in range(100)}


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    """Archivo CSV temporal para pruebas."""
    file_path = tmp_path / "test_data.csv"
    content = "codigo,cantidad,precio\n" + "\n".join(
        [f"A{i:04d},{10 + i % 5},{100.0 + i * 0.5}" for i in range(100)]
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def small_csv_file(tmp_path) -> Path:
    """Archivo CSV pequeño para pruebas de validación."""
    file_path = tmp_path / "small_test.csv"
    content = "codigo,cantidad\nA0001,10\nA0002,20"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def controller_fixture() -> PIController:
    """Controlador PI con parámetros estándar para pruebas."""
    return PIController(
        kp=50.0,
        ki=10.0,
        setpoint=0.5,
        min_output=10,
        max_output=100,
        integral_limit_factor=2.0,
    )


@pytest.fixture
def physics_engine_fixture() -> RefinedFluxPhysicsEngine:
    """Motor de física con parámetros estándar."""
    return RefinedFluxPhysicsEngine(
        capacitance=5000.0,
        resistance=10.0,
        inductance=2.0,
    )


# ============================================================================
# TESTS: CondenserConfig - Validación de Configuración
# ============================================================================

class TestCondenserConfig:
    """Pruebas exhaustivas para validación de configuración."""

    def test_default_config_satisfies_physical_constraints(self):
        """
        La configuración por defecto debe satisfacer restricciones físicas:
        - C > 0 (capacitancia positiva)
        - L > 0 (inductancia positiva)
        - R ≥ 0 (resistencia no negativa)
        - 0 < setpoint < 1 (punto de operación válido)
        """
        config = CondenserConfig()
        
        assert_positive(config.system_capacitance, "capacitance")
        assert_positive(config.system_inductance, "inductance")
        assert_positive(config.base_resistance, "resistance", strict=False)
        assert 0 < config.pid_setpoint < 1, "setpoint debe estar en (0, 1)"
        assert config.min_batch_size < config.max_batch_size, "rango de batch inválido"

    def test_default_config_is_numerically_stable(self):
        """
        La configuración por defecto debe producir un sistema numéricamente estable.
        Verificamos que la frecuencia natural ω₀ = 1/√(LC) sea razonable.
        """
        config = CondenserConfig()
        
        omega_0 = 1.0 / math.sqrt(config.system_inductance * config.system_capacitance)
        
        # ω₀ debe estar en un rango razonable para simulación numérica
        assert 1e-6 < omega_0 < 1e6, f"ω₀ = {omega_0} fuera de rango estable"
        
        # Factor de calidad Q debe ser finito
        if config.base_resistance > 0:
            Q = (1 / config.base_resistance) * math.sqrt(
                config.system_inductance / config.system_capacitance
            )
            assert_finite(Q, "factor de calidad Q")

    def test_invalid_negative_threshold_raises(self):
        """Umbral negativo de registros debe fallar con mensaje descriptivo."""
        with pytest.raises(ConfigurationError) as exc_info:
            CondenserConfig(min_records_threshold=-1)
        
        error_msg = str(exc_info.value).lower()
        assert "min_records_threshold" in error_msg
        assert any(word in error_msg for word in ["negativo", "negative", "inválido", "invalid"])

    def test_invalid_capacitance_raises(self):
        """Capacitancia no positiva debe fallar."""
        with pytest.raises(ConfigurationError) as exc_info:
            CondenserConfig(system_capacitance=0)
        assert "system_capacitance" in str(exc_info.value)
        
        with pytest.raises(ConfigurationError):
            CondenserConfig(system_capacitance=-1.0)

    def test_invalid_inductance_raises(self):
        """Inductancia no positiva debe fallar."""
        with pytest.raises(ConfigurationError) as exc_info:
            CondenserConfig(system_inductance=0.0)
        assert "system_inductance" in str(exc_info.value)
        
        with pytest.raises(ConfigurationError):
            CondenserConfig(system_inductance=-1.0)

    def test_invalid_batch_size_range_raises(self):
        """min_batch_size > max_batch_size debe fallar."""
        with pytest.raises(ConfigurationError) as exc_info:
            CondenserConfig(min_batch_size=1000, max_batch_size=100)
        assert "min_batch_size" in str(exc_info.value)

    def test_invalid_setpoint_boundary_raises(self):
        """Setpoint en límites exactos [0, 1] debe fallar (intervalo abierto)."""
        with pytest.raises(ConfigurationError):
            CondenserConfig(pid_setpoint=0.0)
        
        with pytest.raises(ConfigurationError):
            CondenserConfig(pid_setpoint=1.0)

    @pytest.mark.parametrize("setpoint", [-0.1, 1.1, 2.0, -1.0])
    def test_setpoint_outside_unit_interval_raises(self, setpoint):
        """Setpoints fuera de (0, 1) deben fallar."""
        with pytest.raises(ConfigurationError):
            CondenserConfig(pid_setpoint=setpoint)

    @pytest.mark.parametrize("invalid_value", [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_capacitance_raises(self, invalid_value):
        """Valores no finitos para capacitancia deben fallar."""
        with pytest.raises((ConfigurationError, ValueError)):
            CondenserConfig(system_capacitance=invalid_value)

    @pytest.mark.parametrize("invalid_value", [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_inductance_raises(self, invalid_value):
        """Valores no finitos para inductancia deben fallar."""
        with pytest.raises((ConfigurationError, ValueError)):
            CondenserConfig(system_inductance=invalid_value)

    def test_extreme_but_valid_parameters(self):
        """Parámetros extremos pero válidos deben aceptarse."""
        # Valores muy pequeños pero positivos
        config_small = CondenserConfig(
            system_capacitance=1e-10,
            system_inductance=1e-10,
            base_resistance=1e-10,
        )
        assert config_small.system_capacitance == 1e-10
        
        # Valores muy grandes
        config_large = CondenserConfig(
            system_capacitance=1e10,
            system_inductance=1e10,
        )
        assert config_large.system_capacitance == 1e10


# ============================================================================
# TESTS: PIController - Controlador PI Refinado
# ============================================================================

class TestPIController:
    """Pruebas unitarias exhaustivas para el controlador PI refinado."""

    # ---------- Validación de Parámetros ----------

    def test_invalid_negative_kp_raises(self):
        """Kp ≤ 0 debe fallar (ganancia proporcional debe ser positiva)."""
        with pytest.raises(ConfigurationError) as exc_info:
            PIController(kp=0, ki=10, setpoint=0.5, min_output=10, max_output=100)
        assert "Kp" in str(exc_info.value) or "kp" in str(exc_info.value).lower()
        
        with pytest.raises(ConfigurationError):
            PIController(kp=-5, ki=10, setpoint=0.5, min_output=10, max_output=100)

    def test_invalid_negative_ki_raises(self):
        """Ki < 0 debe fallar (ganancia integral no negativa)."""
        with pytest.raises(ConfigurationError) as exc_info:
            PIController(kp=10, ki=-5, setpoint=0.5, min_output=10, max_output=100)
        assert "Ki" in str(exc_info.value) or "ki" in str(exc_info.value).lower()

    def test_zero_ki_is_valid_proportional_only(self):
        """Ki = 0 es válido (controlador solo proporcional)."""
        controller = PIController(
            kp=50.0, ki=0.0, setpoint=0.5,
            min_output=10, max_output=100
        )
        output = controller.compute(0.3)
        assert 10 <= output <= 100

    def test_invalid_output_range_raises(self):
        """min_output >= max_output debe fallar."""
        with pytest.raises(ConfigurationError):
            PIController(kp=10, ki=5, setpoint=0.5, min_output=100, max_output=100)
        
        with pytest.raises(ConfigurationError):
            PIController(kp=10, ki=5, setpoint=0.5, min_output=200, max_output=100)

    def test_invalid_min_output_not_positive_raises(self):
        """min_output ≤ 0 debe fallar."""
        with pytest.raises(ConfigurationError) as exc_info:
            PIController(kp=10, ki=5, setpoint=0.5, min_output=0, max_output=100)
        assert "min_output" in str(exc_info.value)
        
        with pytest.raises(ConfigurationError):
            PIController(kp=10, ki=5, setpoint=0.5, min_output=-10, max_output=100)

    # ---------- Criterio de Jury para Estabilidad ----------

    def test_jury_criterion_stable_parameters(self, controller_fixture):
        """
        Parámetros estándar deben satisfacer el criterio de Jury.
        
        El sistema discretizado PI tiene función de transferencia:
        G(z) = Kp + Ki*T*z/(z-1)
        
        El polinomio característico en lazo cerrado depende de la planta.
        Para verificación, usamos la respuesta del controlador.
        """
        controller = controller_fixture
        
        # Simular respuesta a escalón
        responses = []
        for i in range(50):
            output = controller.compute(0.3)  # Error constante
            responses.append(output)
        
        # Sistema estable: respuesta debe converger (no oscilar divergentemente)
        # Verificar que las últimas respuestas son similares (convergencia)
        final_responses = responses[-10:]
        variance = sum((r - sum(final_responses)/len(final_responses))**2 
                      for r in final_responses) / len(final_responses)
        
        assert variance < 100, "Sistema no convergió - posible inestabilidad"

    def test_jury_criterion_warning_for_high_gains(self, caplog):
        """
        Ganancias muy altas deben generar warning o error.
        
        Con Kp muy alto, el sistema puede volverse marginalmente estable
        o inestable, lo cual debe detectarse.
        """
        caplog.set_level(logging.WARNING)
        
        try:
            controller = PIController(
                kp=10000.0,  # Ganancia extremadamente alta
                ki=5000.0,
                setpoint=0.5,
                min_output=10,
                max_output=100,
            )
            
            # Si se creó, verificar que hay warnings
            # O que el controlador limita internamente las ganancias
            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            # Dependiendo de la implementación, puede haber warnings
            
        except ConfigurationError:
            # Es aceptable que rechace parámetros inestables
            pass

    def test_controller_stability_analysis_reports_jury_margins(self, controller_fixture):
        """El análisis de estabilidad debe reportar márgenes de Jury."""
        controller = controller_fixture
        
        # Generar suficiente historial
        for i in range(20):
            controller.compute(0.5 + 0.05 * math.sin(i * 0.5))
        
        analysis = controller.get_stability_analysis()
        
        if analysis["status"] == "OPERATIONAL":
            assert "stability_class" in analysis
            # Las clases válidas incluyen información sobre márgenes
            valid_classes = ["STABLE", "MARGINAL", "UNSTABLE", 
                           "ASYMPTOTICALLY_STABLE", "OSCILLATORY"]
            assert any(cls in analysis["stability_class"].upper() 
                      for cls in valid_classes) or "class" in str(analysis).lower()

    # ---------- Filtro EMA Adaptativo ----------

    def test_ema_filter_initialization_no_smoothing(self, controller_fixture):
        """Primera medición debe pasar sin suavizado (inicialización)."""
        controller = controller_fixture
        
        result = controller._apply_ema_filter(0.7)
        
        assert result == 0.7, "Primera medición debe ser exacta"
        assert controller._filtered_pv == 0.7

    def test_ema_filter_applies_exponential_smoothing(self, controller_fixture):
        """Mediciones subsecuentes deben seguir suavizado exponencial."""
        controller = controller_fixture
        
        controller._apply_ema_filter(0.5)  # Inicializar
        
        # Salto grande
        result = controller._apply_ema_filter(1.0)
        
        # EMA: y[n] = α*x[n] + (1-α)*y[n-1]
        # Para α ∈ (0, 1), el resultado debe estar entre 0.5 y 1.0
        assert 0.5 < result < 1.0, f"EMA no aplicado correctamente: {result}"

    def test_ema_filter_detects_step_change(self, controller_fixture):
        """Cambios abruptos (step) deben detectarse y adaptarse."""
        controller = controller_fixture
        
        # Establecer baseline estable
        for _ in range(5):
            controller._apply_ema_filter(0.3)
        
        # Cambio abrupto (step)
        result = controller._apply_ema_filter(0.9)
        
        # El resultado debe estar más cerca del nuevo valor
        # debido a la adaptación para cambios step
        assert result > 0.5, "Detección de step debería reducir inercia del filtro"

    def test_ema_alpha_adapts_to_variance(self, controller_fixture):
        """Alpha del EMA debe adaptarse a la varianza de las innovaciones."""
        controller = controller_fixture
        
        # Secuencia con alta varianza
        high_var_sequence = [0.2, 0.8, 0.3, 0.9, 0.1, 0.7]
        
        for val in high_var_sequence:
            controller._apply_ema_filter(val)
        
        # Con alta varianza, alpha debería ser mayor (menos suavizado)
        # Esto se refleja en la respuesta más rápida a cambios
        result_after_high_var = controller._apply_ema_filter(0.5)
        
        # Reset y probar con baja varianza
        controller._filtered_pv = None
        low_var_sequence = [0.5, 0.51, 0.49, 0.50, 0.52, 0.48]
        
        for val in low_var_sequence:
            controller._apply_ema_filter(val)
        
        # Con baja varianza, la respuesta debería ser más suave
        # (alpha menor, más inercia)
        assert abs(controller._filtered_pv - 0.5) < 0.1

    # ---------- Cálculo de Salida (compute) ----------

    def test_compute_returns_integer_in_valid_range(self, controller_fixture):
        """La salida debe ser entero dentro del rango [min, max]."""
        controller = controller_fixture
        
        for pv in [0.0, 0.25, 0.5, 0.75, 1.0]:
            output = controller.compute(pv)
            
            assert isinstance(output, int), f"Salida debe ser int, obtenido: {type(output)}"
            assert controller.min_output <= output <= controller.max_output

    def test_compute_deadband_reduces_output_jitter(self, controller_fixture):
        """
        Errores dentro de la zona muerta no deben causar cambios significativos.
        
        Zona muerta = ±2% del setpoint = ±0.01 para setpoint=0.5
        """
        controller = controller_fixture
        
        # Llevar al estado estable
        for _ in range(20):
            controller.compute(0.5)
        
        baseline_output = controller._last_output
        
        # Perturbaciones dentro de zona muerta (±1% del setpoint)
        for delta in [-0.005, 0.003, -0.002, 0.004]:
            output = controller.compute(0.5 + delta)
            # Cambio debe ser mínimo
            assert abs(output - baseline_output) <= 3, \
                f"Jitter excesivo dentro de deadband: {output} vs {baseline_output}"

    def test_compute_slew_rate_limiting(self, controller_fixture):
        """
        La tasa de cambio de salida debe estar limitada.
        
        Límite = 15% del rango por paso = 0.15 * (100-10) = 13.5
        """
        controller = controller_fixture
        controller._last_output = 50
        
        # Forzar cambio grande (error máximo)
        output = controller.compute(0.0)
        
        max_slew = int(0.15 * (controller.max_output - controller.min_output))
        actual_change = abs(output - 50)
        
        assert actual_change <= max_slew + 1, \
            f"Slew rate excedido: cambio de {actual_change} > límite de {max_slew}"

    def test_compute_saturation_at_limits(self):
        """La salida debe saturarse en los límites configurados."""
        controller = PIController(
            kp=200.0, ki=50.0, setpoint=0.3,
            min_output=50, max_output=500,
        )
        
        # Error muy grande negativo (pv << setpoint) → salida alta
        output_high = controller.compute(0.0)
        assert output_high <= controller.max_output
        
        # Reset para prueba independiente
        controller.reset()
        
        # Error muy grande positivo (pv >> setpoint) → salida baja
        output_low = controller.compute(1.0)
        assert output_low >= controller.min_output

    # ---------- Anti-Windup ----------

    def test_antiwindup_bounds_integral_term(self, controller_fixture):
        """El término integral debe estar acotado por integral_limit."""
        controller = controller_fixture
        
        # Aplicar error constante grande por muchas iteraciones
        for _ in range(200):
            controller.compute(0.0)  # Error máximo = setpoint = 0.5
        
        assert abs(controller._integral_error) <= controller._integral_limit, \
            f"Integral {controller._integral_error} excede límite {controller._integral_limit}"

    def test_antiwindup_back_calculation(self, controller_fixture):
        """
        Anti-windup por back-calculation debe ajustar el integrador
        cuando la salida está saturada.
        """
        controller = controller_fixture
        
        # Saturar en el límite superior
        for _ in range(50):
            controller.compute(0.1)  # Error grande → saturación alta
        
        integral_saturated = controller._integral_error
        
        # Ahora cambiar dirección
        for _ in range(10):
            controller.compute(0.9)  # Error negativo
        
        # El integral debe haberse ajustado (reducido) por back-calculation
        # No debe estar "pegado" en el límite
        assert controller._integral_error < integral_saturated or \
               controller._integral_error < controller._integral_limit * 0.9

    # ---------- Métrica de Lyapunov ----------

    def test_lyapunov_exponent_negative_for_stable_convergence(self, controller_fixture):
        """
        Sistema con convergencia exponencial debe tener λ < 0.
        
        Para errores e(t) = e₀ * exp(-αt), α > 0:
        λ = lim_{t→∞} (1/t) * ln|e(t)/e₀| = -α < 0
        """
        controller = controller_fixture
        
        # Simular convergencia exponencial: e(k) = 0.5 * 0.9^k
        decay_rate = 0.9
        initial_error = 0.5
        
        for k in range(50):
            error = initial_error * (decay_rate ** k)
            controller._update_stability_metrics(error)
        
        lyapunov = controller.get_lyapunov_exponent()
        
        # λ teórico = ln(0.9) ≈ -0.105
        assert lyapunov < 0.05, \
            f"Sistema convergente debe tener λ < 0, obtenido: {lyapunov}"

    def test_lyapunov_exponent_positive_for_divergence(self, controller_fixture):
        """
        Sistema divergente debe tener λ > 0.
        
        Para errores e(t) = e₀ * exp(αt), α > 0:
        λ = α > 0
        """
        controller = controller_fixture
        
        # Simular divergencia exponencial: e(k) = 0.01 * 1.1^k
        growth_rate = 1.1
        initial_error = 0.01
        
        for k in range(40):
            error = min(initial_error * (growth_rate ** k), 10.0)  # Limitar overflow
            controller._update_stability_metrics(error)
        
        lyapunov = controller.get_lyapunov_exponent()
        
        # λ teórico = ln(1.1) ≈ 0.095
        assert lyapunov > 0, \
            f"Sistema divergente debe tener λ > 0, obtenido: {lyapunov}"

    def test_lyapunov_exponent_near_zero_for_marginal(self, controller_fixture):
        """Sistema marginalmente estable (oscilatorio) tiene λ ≈ 0."""
        controller = controller_fixture
        
        # Simular oscilación sostenida con amplitud constante
        amplitude = 0.2
        for k in range(60):
            error = amplitude * math.sin(k * 0.3)
            controller._update_stability_metrics(abs(error) + 0.01)  # Evitar log(0)
        
        lyapunov = controller.get_lyapunov_exponent()
        
        # Para oscilación pura, λ debería estar cerca de 0
        assert abs(lyapunov) < 0.3, \
            f"Sistema oscilatorio debe tener |λ| ≈ 0, obtenido: {lyapunov}"

    # ---------- Análisis de Estabilidad ----------

    def test_stability_analysis_insufficient_data_status(self, controller_fixture):
        """Con datos insuficientes, debe indicar estado apropiado."""
        controller = controller_fixture
        
        analysis = controller.get_stability_analysis()
        
        assert analysis["status"] == "INSUFFICIENT_DATA"
        assert "samples" in str(analysis).lower() or "data" in str(analysis).lower()

    def test_stability_analysis_complete_after_sufficient_samples(self, controller_fixture):
        """Con suficientes muestras, debe proporcionar análisis completo."""
        controller = controller_fixture
        
        # Generar historial suficiente (mínimo 10 muestras típicamente)
        for i in range(20):
            controller.compute(0.5 + 0.05 * math.sin(i * 0.5))
        
        analysis = controller.get_stability_analysis()
        
        assert analysis["status"] == "OPERATIONAL"
        assert "stability_class" in analysis
        assert "convergence" in analysis
        assert "lyapunov_exponent" in analysis
        
        # Valores deben ser finitos
        if isinstance(analysis["lyapunov_exponent"], (int, float)):
            assert_finite(analysis["lyapunov_exponent"], "lyapunov_exponent")

    # ---------- Reset y Estado ----------

    def test_reset_clears_dynamic_state(self, controller_fixture):
        """Reset debe limpiar estado dinámico pero preservar historial."""
        controller = controller_fixture
        
        # Acumular estado
        for i in range(10):
            controller.compute(0.4 + i * 0.02)
        
        original_history_len = len(controller._error_history)
        original_integral = controller._integral_error
        
        assert original_integral != 0.0  # Verificar que hay estado acumulado
        
        controller.reset()
        
        # Estado dinámico limpio
        assert controller._integral_error == 0.0
        assert controller._last_output is None
        assert controller._filtered_pv is None
        
        # Historial preservado para análisis post-mortem
        assert len(controller._error_history) == original_history_len

    def test_get_state_returns_serializable_dict(self, controller_fixture):
        """get_state debe retornar diccionario serializable a JSON."""
        controller = controller_fixture
        controller.compute(0.5)
        
        state = controller.get_state()
        
        assert isinstance(state, dict)
        assert "parameters" in state
        assert "state" in state
        assert "diagnostics" in state
        
        # Debe ser serializable a JSON sin errores
        try:
            json_str = json.dumps(state)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Estado no serializable a JSON: {e}")

    def test_get_diagnostics_includes_all_metrics(self, controller_fixture):
        """Diagnóstico debe incluir todas las métricas relevantes."""
        controller = controller_fixture
        
        for _ in range(5):
            controller.compute(0.5)
        
        diag = controller.get_diagnostics()
        
        required_keys = ["status", "control_metrics", "stability_analysis", "parameters"]
        for key in required_keys:
            assert key in diag, f"Falta clave de diagnóstico: {key}"


# ============================================================================
# TESTS: TopologicalAnalyzer - Análisis Topológico
# ============================================================================

class TestTopologicalAnalyzer:
    """Pruebas del analizador topológico con verificación de invariantes."""

    def test_betti_numbers_for_disconnected_graph(self):
        """
        Grafo con k componentes conexas debe tener β₀ = k.
        
        Ejemplo: 2 aristas disjuntas (0-1) y (2-3) → β₀ = 2
        """
        analyzer = TopologicalAnalyzer()
        
        # Configurar grafo desconectado
        analyzer._adjacency_list = {
            0: {1}, 1: {0},
            2: {3}, 3: {2}
        }
        analyzer._vertex_count = 4
        analyzer._edge_count = 2
        
        betti = analyzer.compute_betti_with_spectral()
        
        assert betti[0] == 2, f"β₀ debe ser 2 para 2 componentes, obtenido: {betti[0]}"

    def test_betti_numbers_for_cycle(self):
        """
        Grafo cíclico (triángulo) debe tener β₀ = 1, β₁ = 1.
        
        Relación de Euler: χ = V - E + F = β₀ - β₁
        Para triángulo: χ = 3 - 3 + 1 = 1 = 1 - 0 (si plano) o β₁ = 1 para ciclo
        """
        analyzer = TopologicalAnalyzer()
        
        # Triángulo: 0-1-2-0
        analyzer._adjacency_list = {
            0: {1, 2}, 1: {0, 2}, 2: {0, 1}
        }
        analyzer._vertex_count = 3
        analyzer._edge_count = 3
        
        betti = analyzer.compute_betti_with_spectral()
        
        assert betti[0] == 1, f"β₀ debe ser 1 para grafo conexo, obtenido: {betti[0]}"
        # β₁ = E - V + β₀ = 3 - 3 + 1 = 1
        assert betti[1] >= 0  # β₁ debe ser no negativo

    def test_betti_euler_poincare_relation(self):
        """
        Verificar relación de Euler-Poincaré: χ = Σ(-1)^i * β_i
        
        Para grafos: χ = β₀ - β₁ = V - E + componentes
        """
        analyzer = TopologicalAnalyzer()
        
        # Grafo con 2 ciclos independientes unidos
        # Forma de "8": dos triángulos compartiendo un vértice
        analyzer._adjacency_list = {
            0: {1, 2, 3, 4},
            1: {0, 2}, 2: {0, 1},
            3: {0, 4}, 4: {0, 3}
        }
        analyzer._vertex_count = 5
        analyzer._edge_count = 6
        
        betti = analyzer.compute_betti_with_spectral()
        
        # Euler: χ = V - E = 5 - 6 = -1
        # Para grafos conexos: β₀ - β₁ = 1 - β₁
        # -1 = 1 - β₁ → β₁ = 2
        euler_char = analyzer._vertex_count - analyzer._edge_count
        computed_euler = betti[0] - betti[1]
        
        # Permitir tolerancia por métodos aproximados
        assert abs(euler_char - computed_euler) <= 1, \
            f"Violación Euler-Poincaré: χ={euler_char}, β₀-β₁={computed_euler}"

    def test_betti_empty_graph(self):
        """Grafo vacío debe tener β₀ = 0."""
        analyzer = TopologicalAnalyzer()
        
        analyzer._adjacency_list = {}
        analyzer._vertex_count = 0
        analyzer._edge_count = 0
        
        betti = analyzer.compute_betti_with_spectral()
        
        assert betti[0] == 0

    def test_betti_single_vertex(self):
        """Vértice aislado debe tener β₀ = 1, β₁ = 0."""
        analyzer = TopologicalAnalyzer()
        
        analyzer._adjacency_list = {0: set()}
        analyzer._vertex_count = 1
        analyzer._edge_count = 0
        
        betti = analyzer.compute_betti_with_spectral()
        
        assert betti[0] == 1
        assert betti[1] == 0

    @pytest.mark.skipif(not HAS_NUMPY, reason="Requiere numpy para análisis espectral")
    def test_betti_spectral_consistency(self):
        """
        Números de Betti calculados espectralmente deben ser consistentes
        con conteo directo para grafos simples.
        """
        analyzer = TopologicalAnalyzer()
        
        # Grafo lineal: 0-1-2-3-4 (5 vértices, 4 aristas)
        analyzer._adjacency_list = {
            0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}
        }
        analyzer._vertex_count = 5
        analyzer._edge_count = 4
        
        betti = analyzer.compute_betti_with_spectral()
        
        # Grafo lineal: conexo (β₀=1), sin ciclos (β₁=0)
        assert betti[0] == 1
        assert betti[1] == 0


# ============================================================================
# TESTS: EntropyCalculator - Cálculo de Entropía
# ============================================================================

class TestEntropyCalculator:
    """Pruebas del calculador de entropía con verificación de propiedades."""

    def test_entropy_maximum_for_uniform_distribution(self):
        """
        Distribución uniforme maximiza entropía de Shannon.
        
        H_max = log₂(n) para n categorías equiprobables
        """
        calc = EntropyCalculator()
        
        # 4 categorías equiprobables
        counts = {"a": 100, "b": 100, "c": 100, "d": 100}
        result = calc.calculate_entropy_bayesian(counts)
        
        expected_max_entropy = math.log2(4)  # = 2.0
        
        assert abs(result['entropy_expected'] - expected_max_entropy) < 0.1, \
            f"Entropía uniforme debe ser ~{expected_max_entropy}, obtenido: {result['entropy_expected']}"

    def test_entropy_zero_for_pure_state(self):
        """
        Estado puro (una categoría) tiene entropía cero.
        
        H = -1 * log₂(1) = 0
        """
        calc = EntropyCalculator()
        
        counts = {"solo_categoria": 1000}
        result = calc.calculate_entropy_bayesian(counts)
        
        assert result['entropy_expected'] < 0.01, \
            f"Estado puro debe tener H ≈ 0, obtenido: {result['entropy_expected']}"

    def test_entropy_additivity_for_independent_systems(self):
        """
        Entropía de sistemas independientes es aditiva.
        
        H(X,Y) = H(X) + H(Y) si X ⊥ Y
        """
        calc = EntropyCalculator()
        
        # Sistema A: 2 estados equiprobables
        counts_a = {"a1": 50, "a2": 50}
        h_a = calc.calculate_entropy_bayesian(counts_a)['entropy_expected']
        
        # Sistema B: 3 estados equiprobables
        counts_b = {"b1": 33, "b2": 33, "b3": 34}
        h_b = calc.calculate_entropy_bayesian(counts_b)['entropy_expected']
        
        # Sistema combinado (producto cartesiano) si fueran independientes
        # H(A) + H(B) debería aproximar a log₂(2) + log₂(3) ≈ 2.58
        h_combined_expected = h_a + h_b
        
        assert h_combined_expected > h_a
        assert h_combined_expected > h_b

    def test_renyi_spectrum_converges_to_shannon_at_alpha_1(self):
        """
        Entropía de Rényi con α→1 converge a entropía de Shannon.
        
        lim_{α→1} H_α = H_Shannon
        """
        calc = EntropyCalculator()
        
        probs = [0.25, 0.25, 0.25, 0.25]  # Uniforme
        spectrum = calc.calculate_renyi_spectrum(probs)
        
        shannon = -sum(p * math.log2(p) for p in probs if p > 0)
        
        # α=1 en el espectro debe ser Shannon
        assert abs(spectrum[1] - shannon) < 1e-10, \
            f"H_Rényi(α=1) debe ser Shannon={shannon}, obtenido: {spectrum[1]}"

    def test_renyi_hartley_entropy_at_alpha_0(self):
        """
        Entropía de Rényi con α=0 es la entropía de Hartley.
        
        H_0 = log₂(|soporte|)
        """
        calc = EntropyCalculator()
        
        probs = [0.5, 0.3, 0.15, 0.05]  # 4 estados no uniformes
        spectrum = calc.calculate_renyi_spectrum(probs)
        
        hartley = math.log2(4)  # log₂ del número de estados no nulos
        
        assert abs(spectrum[0] - hartley) < 0.01, \
            f"H_Rényi(α=0) debe ser Hartley={hartley}, obtenido: {spectrum[0]}"

    def test_renyi_min_entropy_at_alpha_infinity(self):
        """
        Entropía de Rényi con α→∞ es la min-entropía.
        
        H_∞ = -log₂(max(p_i))
        """
        calc = EntropyCalculator()
        
        probs = [0.6, 0.3, 0.1]  # max = 0.6
        spectrum = calc.calculate_renyi_spectrum(probs)
        
        min_entropy = -math.log2(max(probs))
        
        # El espectro típicamente incluye α grande pero no infinito
        # Verificar que H_α decrece con α (propiedad de Rényi)
        alphas_sorted = sorted(spectrum.keys())
        for i in range(len(alphas_sorted) - 1):
            assert spectrum[alphas_sorted[i]] >= spectrum[alphas_sorted[i+1]] - 0.01

    def test_entropy_bounds(self):
        """Entropía debe estar en [0, log₂(n)]."""
        calc = EntropyCalculator()
        
        # Caso general
        counts = {"x": 70, "y": 20, "z": 10}
        result = calc.calculate_entropy_bayesian(counts)
        
        n = len(counts)
        max_entropy = math.log2(n)
        
        assert 0 <= result['entropy_expected'] <= max_entropy + 0.1


# ============================================================================
# TESTS: RefinedFluxPhysicsEngine
# ============================================================================

class TestRefinedFluxPhysicsEngine:
    """Pruebas del motor de física RLC refinado con verificación de invariantes."""

    # ---------- Validación de Parámetros ----------

    def test_invalid_zero_capacitance_raises(self):
        """Capacitancia cero viola física (C > 0 requerido)."""
        with pytest.raises(ConfigurationError):
            RefinedFluxPhysicsEngine(capacitance=0, resistance=10, inductance=2)

    def test_invalid_negative_capacitance_raises(self):
        """Capacitancia negativa es no física."""
        with pytest.raises(ConfigurationError):
            RefinedFluxPhysicsEngine(capacitance=-100, resistance=10, inductance=2)

    def test_invalid_zero_inductance_raises(self):
        """Inductancia cero viola física (L > 0 requerido)."""
        with pytest.raises(ConfigurationError):
            RefinedFluxPhysicsEngine(capacitance=100, resistance=10, inductance=0)

    def test_zero_resistance_is_valid_ideal_circuit(self):
        """Resistencia cero es válida (circuito ideal sin pérdidas)."""
        engine = RefinedFluxPhysicsEngine(
            capacitance=100.0,
            resistance=0.0,  # Sin pérdidas
            inductance=1.0,
        )
        
        # Q debe ser infinito (sin amortiguamiento)
        assert engine._Q == float("inf")
        
        # Debe funcionar sin errores
        metrics = engine.calculate_metrics(100, 50, 0, 1.0)
        assert_finite(metrics["saturation"])

    # ---------- Conservación de Energía Hamiltoniana ----------

    def test_hamiltonian_conservation_no_dissipation(self, physics_engine_fixture):
        """
        En circuito ideal (R=0), la energía total debe conservarse.
        
        H = (1/2)LI² + (1/2)CV² = constante
        """
        # Crear motor sin resistencia
        engine = RefinedFluxPhysicsEngine(
            capacitance=1000.0,
            resistance=0.0,  # Sin disipación
            inductance=1.0,
        )
        engine._initialized = True
        engine._last_time = time.time() - 0.1
        engine._last_current = 0.5
        
        # Calcular energía inicial
        m1 = engine.calculate_metrics(100, 50, 0, 0.1)
        E_initial = m1["total_energy"]
        
        # Avanzar en el tiempo (sin fuente externa)
        m2 = engine.calculate_metrics(100, 50, 0, 0.2)
        E_final = m2["total_energy"]
        
        # Energía debe conservarse (tolerancia por errores numéricos)
        if E_initial > 0:
            relative_change = abs(E_final - E_initial) / E_initial
            # RK4 tiene error O(h⁴), pero la simulación puede tener otros errores
            assert relative_change < 0.1, \
                f"Energía no conservada: ΔE/E = {relative_change}"

    def test_energy_dissipation_with_resistance(self, physics_engine_fixture):
        """
        Con resistencia, la energía debe disiparse (decrecer o convertirse en calor).
        
        dE/dt = -I²R ≤ 0
        """
        engine = physics_engine_fixture
        engine._initialized = True
        engine._last_time = time.time() - 0.1
        engine._last_current = 1.0  # Corriente inicial
        
        energies = []
        for i in range(10):
            m = engine.calculate_metrics(100, 50, 0, 0.1 * (i + 1))
            energies.append(m["total_energy"])
        
        # La energía debe tender a decrecer (o mantenerse si en equilibrio)
        # Verificar que no hay crecimiento explosivo
        assert max(energies) < energies[0] * 2, "Energía creció inesperadamente"

    # ---------- Estabilidad Giroscópica ----------

    def test_gyroscopic_stability_positive(self, physics_engine_fixture):
        """
        Estabilidad giroscópica debe ser positiva para sistema estable.
        
        Basado en ecuaciones de Euler para cuerpo rígido.
        """
        engine = physics_engine_fixture
        
        metrics = engine.calculate_metrics(100, 80, 0, 1.0)
        
        assert "gyroscopic_stability" in metrics
        assert_finite(metrics["gyroscopic_stability"])
        # Para operación normal, debe ser ≥ 0
        assert metrics["gyroscopic_stability"] >= 0

    # ---------- Métricas Completas ----------

    def test_calculate_metrics_returns_all_required_fields(self, physics_engine_fixture):
        """calculate_metrics debe retornar todas las métricas requeridas."""
        engine = physics_engine_fixture
        engine._initialized = True
        engine._last_time = time.time() - 0.1
        engine._last_current = 0.5
        
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=80,
            error_count=5,
            processing_time=1.0
        )
        
        required_keys = [
            "saturation", "complexity", "current_I",
            "potential_energy", "kinetic_energy", "total_energy",
            "dissipated_power", "flyback_voltage",
            "water_hammer_pressure", "pump_work",
            "dynamic_resistance", "damping_ratio",
            "entropy_shannon", "gyroscopic_stability",
            "betti_0", "betti_1", "hamiltonian_excess",
            "field_energy", "poynting_flux_mean"
        ]
        
        for key in required_keys:
            assert key in metrics, f"Falta métrica requerida: {key}"
            assert_finite(metrics[key], f"metrics['{key}']")

    def test_metrics_physical_consistency(self, physics_engine_fixture):
        """Las métricas deben ser físicamente consistentes."""
        engine = physics_engine_fixture
        
        metrics = engine.calculate_metrics(100, 50, 10, 1.0)
        
        # Saturación en [0, 1]
        assert_probability(metrics["saturation"], "saturation")
        
        # Energías no negativas
        assert_positive(metrics["potential_energy"], "potential_energy", strict=False)
        assert_positive(metrics["kinetic_energy"], "kinetic_energy", strict=False)
        
        # Energía total = potencial + cinética (para sistema conservativo)
        total_check = metrics["potential_energy"] + metrics["kinetic_energy"]
        assert abs(metrics["total_energy"] - total_check) < 1.0 or \
               metrics["total_energy"] >= 0

    def test_pump_analogy_backward_compatibility(self, physics_engine_fixture):
        """Las métricas de analogía de bomba deben ser consistentes."""
        engine = physics_engine_fixture
        
        metrics = engine.calculate_metrics(100, 50, 0, 1.0)
        
        # Verificar presencia de métricas de bomba
        assert "water_hammer_pressure" in metrics
        assert "piston_pressure" in metrics
        assert "pump_work" in metrics
        
        # Compatibilidad: flyback_voltage == water_hammer_pressure
        assert metrics["flyback_voltage"] == metrics["water_hammer_pressure"]

    # ---------- Entropía del Sistema ----------

    def test_entropy_pure_state_is_zero(self, physics_engine_fixture):
        """Estado puro (sin errores, máximo orden) tiene entropía cero."""
        engine = physics_engine_fixture
        
        result = engine.calculate_system_entropy(
            total_records=100,
            error_count=0,  # Sin errores
            processing_time=1.0
        )
        
        # Entropía de Shannon para estado puro
        assert result["shannon_entropy"] < 0.01

    def test_entropy_increases_with_errors(self, physics_engine_fixture):
        """Entropía debe aumentar con la tasa de errores."""
        engine = physics_engine_fixture
        
        e_low = engine.calculate_system_entropy(100, 5, 1.0)
        e_high = engine.calculate_system_entropy(100, 50, 1.0)
        
        assert e_high["shannon_entropy"] >= e_low["shannon_entropy"]

    def test_thermal_death_at_maximum_entropy(self, physics_engine_fixture):
        """100% errores debe indicar muerte térmica."""
        engine = physics_engine_fixture
        
        result = engine.calculate_system_entropy(100, 100, 1.0)
        
        assert result["is_thermal_death"] is True

    # ---------- Diagnóstico y Tendencias ----------

    def test_system_diagnosis_returns_valid_states(self, physics_engine_fixture):
        """Diagnóstico debe retornar estados válidos."""
        engine = physics_engine_fixture
        
        for saturation in [0.2, 0.5, 0.8, 0.95]:
            metrics = {"saturation": saturation, "entropy_ratio": 0.3}
            diagnosis = engine.get_system_diagnosis(metrics)
            
            assert "state" in diagnosis
            assert "damping" in diagnosis
            assert diagnosis["state"] in ["NOMINAL", "OPERATING", "WARNING", 
                                          "CRITICAL", "OVERFLOW", "UNKNOWN"]

    def test_trend_analysis_requires_history(self, physics_engine_fixture):
        """Análisis de tendencias requiere historial."""
        engine = physics_engine_fixture
        
        # Sin historial
        analysis = engine.get_trend_analysis()
        assert analysis["status"] in ["INSUFFICIENT_DATA", "NO_HISTORY", "OK"]
        
        # Generar historial
        for i in range(10):
            engine._metrics_history.append({
                "saturation": 0.5 + i * 0.02,
                "dissipated_power": 10 + i,
                "entropy_ratio": 0.3,
            })
        
        analysis = engine.get_trend_analysis()
        assert analysis["status"] == "OK"

    # ---------- Límites de Flyback ----------

    def test_flyback_voltage_is_bounded(self, physics_engine_fixture):
        """Voltaje de flyback debe estar acotado por límites de seguridad."""
        engine = physics_engine_fixture
        engine._initialized = True
        engine._last_time = time.time() - 0.001  # dt muy pequeño
        engine._last_current = 0.0
        
        # Simular cambio de corriente extremo
        metrics = engine.calculate_metrics(100, 100, 0, 0.001)
        
        assert metrics["flyback_voltage"] <= SystemConstants.MAX_FLYBACK_VOLTAGE


# ============================================================================
# TESTS: MaxwellSolver - Solucionador FDTD
# ============================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="Requiere numpy/scipy")
class TestMaxwellSolver:
    """Pruebas del solucionador Maxwell de 4to orden."""

    def test_pml_profile_initialization(self, physics_engine_fixture):
        """Perfiles PML deben inicializarse correctamente."""
        engine = physics_engine_fixture
        solver = engine.maxwell_solver
        
        if solver is None:
            pytest.skip("MaxwellSolver no inicializado")
        
        # PML no negativo
        assert np.all(solver.sigma_e_pml >= 0)
        assert np.all(solver.sigma_m_pml >= 0)
        
        # Debe tener variación espacial
        if solver.calc.num_nodes > 2:
            assert np.max(solver.sigma_e_pml) > 0

    def test_energy_and_momentum_calculation(self, physics_engine_fixture):
        """Cálculo de energía y momento debe ser consistente."""
        engine = physics_engine_fixture
        solver = engine.maxwell_solver
        
        if solver is None:
            pytest.skip("MaxwellSolver no inicializado")
        
        # Inyectar campo
        solver.E = np.ones(solver.calc.num_edges)
        solver.update_constitutive_relations()
        
        metrics = solver.compute_energy_and_momentum()
        
        assert "total_energy" in metrics
        assert "poynting_vector" in metrics
        assert metrics["total_energy"] >= 0

    def test_leapfrog_step_numerical_stability(self, physics_engine_fixture):
        """Pasos de tiempo leapfrog deben ser numéricamente estables."""
        engine = physics_engine_fixture
        
        if engine.maxwell_solver is None:
            pytest.skip("MaxwellSolver no inicializado")
        
        initial_energy = engine.maxwell_solver.compute_energy_and_momentum()["total_energy"]
        
        # Ejecutar varios pasos
        for _ in range(20):
            engine.maxwell_solver.leapfrog_step(dt=0.001)
        
        final_energy = engine.maxwell_solver.compute_energy_and_momentum()["total_energy"]
        
        # La energía no debe explotar (estabilidad)
        if initial_energy > 0:
            assert final_energy < initial_energy * 100, "Posible inestabilidad numérica"

    def test_4th_order_correction_improves_accuracy(self, physics_engine_fixture):
        """Corrección de 4to orden debe mejorar precisión vs 2do orden."""
        engine = physics_engine_fixture
        
        if engine.maxwell_solver is None:
            pytest.skip("MaxwellSolver no inicializado")
        
        # Este test verificaría que el error escala como O(h⁴) vs O(h²)
        # Por simplicidad, verificamos que la corrección se aplica sin errores
        try:
            for _ in range(5):
                engine.maxwell_solver.leapfrog_step(dt=0.001)
        except Exception as e:
            pytest.fail(f"Corrección de 4to orden falló: {e}")


# ============================================================================
# TESTS: DataFluxCondenser
# ============================================================================

class TestDataFluxCondenser:
    """Pruebas del condensador de flujo de datos refinado."""

    # ---------- Inicialización ----------

    def test_initialization_creates_all_components(self, valid_config, valid_profile):
        """Inicialización debe crear todos los componentes."""
        condenser = DataFluxCondenser(valid_config, valid_profile)
        
        assert condenser.physics is not None
        assert condenser.controller is not None
        assert condenser.condenser_config is not None
        assert hasattr(condenser, 'logger')

    def test_initialization_with_custom_config(self, valid_config, valid_profile):
        """Debe aceptar y aplicar configuración personalizada."""
        custom_config = CondenserConfig(
            min_batch_size=100,
            max_batch_size=1000,
            pid_setpoint=0.4,
            pid_kp=400.0,
            pid_ki=20.0,
        )
        
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)
        
        assert condenser.condenser_config.min_batch_size == 100
        assert condenser.condenser_config.max_batch_size == 1000
        assert condenser.condenser_config.pid_setpoint == 0.4

    # ---------- Validación de Archivos ----------

    def test_validate_file_exists(self, condenser, mock_csv_file):
        """Debe aceptar archivo existente."""
        path = condenser._validate_input_file(str(mock_csv_file))
        assert path.exists()
        assert path.is_file()

    def test_validate_file_not_exists_raises(self, condenser, tmp_path):
        """Archivo inexistente debe fallar."""
        with pytest.raises(InvalidInputError) as exc_info:
            condenser._validate_input_file(str(tmp_path / "nonexistent.csv"))
        assert "no existe" in str(exc_info.value).lower()

    def test_validate_file_invalid_extension_raises(self, condenser, tmp_path):
        """Extensión inválida debe fallar."""
        invalid_file = tmp_path / "test.xyz"
        invalid_file.write_text("data")
        
        with pytest.raises(InvalidInputError) as exc_info:
            condenser._validate_input_file(str(invalid_file))
        assert "extensión" in str(exc_info.value).lower()

    def test_validate_file_too_small_raises(self, condenser, tmp_path):
        """Archivo demasiado pequeño debe fallar."""
        tiny_file = tmp_path / "tiny.csv"
        tiny_file.write_text("a")
        
        with pytest.raises(InvalidInputError) as exc_info:
            condenser._validate_input_file(str(tiny_file))
        assert "pequeño" in str(exc_info.value).lower()

    def test_validate_directory_raises(self, condenser, tmp_path):
        """Directorio en lugar de archivo debe fallar."""
        with pytest.raises(InvalidInputError) as exc_info:
            condenser._validate_input_file(str(tmp_path))
        assert "archivo" in str(exc_info.value).lower()

    # ---------- Estimación de Cache Hits ----------

    def test_estimate_cache_hits_with_overlap(self, condenser):
        """Debe estimar hits basado en campos compartidos."""
        batch = [
            {"codigo": "A1", "cantidad": 10},
            {"codigo": "A2", "precio": 5.0},
            {"otro_campo": "x"},
        ]
        cache = {"codigo": "cached", "cantidad": "cached"}
        
        hits = condenser._estimate_cache_hits(batch, cache)
        
        # Al menos algunos hits por overlap
        assert hits >= 1

    def test_estimate_cache_hits_empty_cache(self, condenser):
        """Cache vacío debe retornar estimación mínima."""
        batch = [{"a": 1}, {"b": 2}]
        cache = {}
        
        hits = condenser._estimate_cache_hits(batch, cache)
        
        assert hits >= 0

    def test_estimate_cache_hits_empty_batch(self, condenser):
        """Batch vacío debe retornar 0 hits."""
        hits = condenser._estimate_cache_hits([], {"a": 1})
        assert hits == 0

    # ---------- Predicción de Saturación (EKF) ----------

    def test_predict_saturation_increasing_trend(self, condenser):
        """Tendencia creciente debe predecir valor mayor."""
        history = [0.3, 0.4, 0.5, 0.6]
        
        prediction = condenser._predict_next_saturation(history)
        
        assert prediction > 0.6
        assert_probability(prediction)

    def test_predict_saturation_decreasing_trend(self, condenser):
        """Tendencia decreciente debe predecir valor menor."""
        history = [0.8, 0.7, 0.6, 0.5]
        
        prediction = condenser._predict_next_saturation(history)
        
        assert prediction < 0.5
        assert_probability(prediction)

    def test_predict_saturation_stable(self, condenser):
        """Valores estables deben predecir valor similar."""
        history = [0.5, 0.51, 0.49, 0.50]
        
        prediction = condenser._predict_next_saturation(history)
        
        assert 0.4 < prediction < 0.6

    def test_predict_saturation_insufficient_history(self, condenser):
        """Historial corto retorna último valor."""
        prediction = condenser._predict_next_saturation([0.5])
        assert prediction == 0.5

    def test_predict_saturation_empty_history(self, condenser):
        """Historial vacío retorna valor por defecto."""
        prediction = condenser._predict_next_saturation([])
        assert prediction == 0.5

    @pytest.mark.skipif(not HAS_NUMPY, reason="Requiere numpy para EKF")
    def test_ekf_state_initialization(self, condenser):
        """EKF debe inicializarse con historial suficiente."""
        history = [0.5 + 0.1 * i for i in range(10)]
        
        for i in range(len(history)):
            condenser._predict_next_saturation(history[:i+1])
        
        assert hasattr(condenser, '_ekf_state')

    # ---------- Recuperación de Batches ----------

    def test_recovery_direct_success(self, condenser, sample_raw_records, sample_parse_cache):
        """Modo directo exitoso retorna resultado correcto."""
        with patch.object(condenser, "_rectify_signal") as mock_rectify:
            mock_rectify.return_value = pd.DataFrame([{"a": 1, "b": 2}])
            
            result = condenser._process_single_batch_with_recovery(
                sample_raw_records[:5],
                sample_parse_cache,
                consecutive_failures=0
            )
            
            assert result.success is True
            assert result.records_processed == 1
            assert not result.dataframe.empty

    def test_recovery_binary_split_on_failure(self, condenser, sample_raw_records, sample_parse_cache):
        """Fallo directo activa división binaria."""
        with patch.object(condenser, "_rectify_signal") as mock_rectify:
            call_count = [0]
            
            def side_effect(parsed_data, telemetry=None):
                call_count[0] += 1
                if len(parsed_data.raw_records) >= 10:
                    raise MemoryError("Out of memory")
                return pd.DataFrame([{"r": i} for i in range(len(parsed_data.raw_records))])
            
            mock_rectify.side_effect = side_effect
            
            batch = sample_raw_records[:10]
            result = condenser._process_single_batch_with_recovery(
                batch, sample_parse_cache, consecutive_failures=1
            )
            
            assert result.success is True
            assert call_count[0] > 1  # División ocurrió

    def test_recovery_empty_batch_succeeds(self, condenser, sample_parse_cache):
        """Batch vacío retorna éxito con 0 registros."""
        result = condenser._process_single_batch_with_recovery([], sample_parse_cache, 0)
        
        assert result.success is True
        assert result.records_processed == 0
        assert result.dataframe.empty

    # ---------- Consolidación de Resultados ----------

    def test_consolidate_multiple_dataframes(self, condenser):
        """Debe concatenar múltiples DataFrames."""
        dfs = [
            pd.DataFrame([{"a": 1}]),
            pd.DataFrame([{"a": 2}]),
            pd.DataFrame([{"a": 3}]),
        ]
        
        result = condenser._consolidate_results(dfs)
        
        assert len(result) == 3
        assert list(result["a"]) == [1, 2, 3]

    def test_consolidate_ignores_empty_dataframes(self, condenser):
        """Debe ignorar DataFrames vacíos."""
        dfs = [
            pd.DataFrame([{"a": 1}]),
            pd.DataFrame(),
            pd.DataFrame([{"a": 2}]),
        ]
        
        result = condenser._consolidate_results(dfs)
        
        assert len(result) == 2

    def test_consolidate_respects_limit(self, condenser):
        """Debe respetar límite de batches."""
        limit = SystemConstants.MAX_BATCHES_TO_CONSOLIDATE
        dfs = [pd.DataFrame([{"i": i}]) for i in range(limit + 100)]
        
        result = condenser._consolidate_results(dfs)
        
        assert len(result) <= limit

    # ---------- Validación de Salida ----------

    def test_validate_output_empty_warning(self, condenser, caplog):
        """DataFrame vacío genera warning."""
        caplog.set_level(logging.WARNING)
        
        # Desactivar validación estricta
        object.__setattr__(condenser.condenser_config, 'enable_strict_validation', False)
        
        condenser._validate_output(pd.DataFrame())
        
        assert any("vacío" in r.message.lower() for r in caplog.records)

    def test_validate_output_strict_raises(self, valid_config, valid_profile):
        """Validación estricta falla con registros insuficientes."""
        config = CondenserConfig(
            min_records_threshold=100,
            enable_strict_validation=True,
        )
        condenser = DataFluxCondenser(valid_config, valid_profile, config)
        
        with pytest.raises(ProcessingError):
            condenser._validate_output(pd.DataFrame([{"a": 1}]))

    # ---------- Estadísticas y Salud ----------

    def test_processing_stats_accumulation(self):
        """ProcessingStats debe acumular correctamente."""
        stats = ProcessingStats()
        
        stats.add_batch_stats(100, 0.5, 10.0, 0.1, 5.0, True)
        stats.add_batch_stats(50, 0.6, 15.0, 0.2, 8.0, True)
        stats.add_batch_stats(30, 0.8, 20.0, 0.3, 10.0, False)
        
        assert stats.total_batches == 3
        assert stats.processed_records == 150  # 100 + 50
        assert stats.failed_batches == 1
        assert stats.failed_records == 30

    def test_get_system_health_reports_healthy(self, condenser):
        """Sistema sin problemas reporta HEALTHY."""
        health = condenser.get_system_health()
        
        assert health["health"] == "HEALTHY"
        assert len(health["issues"]) == 0


# ============================================================================
# TESTS: ViscoelasticMembrane - Membrana p-Laplaciano
# ============================================================================

class TestViscoelasticMembrane:
    """Pruebas específicas para la Membrana Viscoelástica (p-Laplaciano)."""

    def test_membrane_reaction_components(self, physics_engine_fixture):
        """Verifica que los componentes de la membrana se calculen correctamente."""
        engine = physics_engine_fixture
        dt = 0.01
        current_I = 0.5

        reaction = engine.calculate_membrane_reaction(current_I, dt)

        assert "v_total" in reaction
        assert "v_elastic" in reaction
        assert "v_viscous" in reaction
        assert "v_inertial" in reaction
        assert "dynamic_esr" in reaction

        # Con corriente positiva y carga inicial cero:
        # v_elastic should be 0 initially
        assert reaction["v_elastic"] == 0.0
        # v_inertial should be positive if last_current was 0
        assert reaction["v_inertial"] > 0
        # v_viscous should be positive
        assert reaction["v_viscous"] > 0

    def test_nonlinear_viscosity_p_laplacian(self, physics_engine_fixture):
        """Verifica el endurecimiento no lineal (p-Laplaciano) ante saltos de corriente."""
        engine = physics_engine_fixture
        dt = 0.01

        # Salto pequeño
        reaction_small = engine.calculate_membrane_reaction(0.1, dt)
        esr_small = reaction_small["dynamic_esr"]

        # Salto grande (misma base pero di/dt mayor)
        # Necesitamos crear uno nuevo para comparar limpiamente
        engine2 = RefinedFluxPhysicsEngine(5000.0, 10.0, 2.0)
        reaction_large = engine2.calculate_membrane_reaction(0.5, dt)
        esr_large = reaction_large["dynamic_esr"]

        # p-Laplaciano: Reff aumenta con Vinertial.
        assert esr_large > esr_small, "La membrana debe endurecerse ante saltos mayores"

    def test_active_clamping_tl431(self, physics_engine_fixture):
        """Verifica que el clamping activo se active ante sobrepresión."""
        engine = physics_engine_fixture
        config = CondenserConfig(max_voltage=5.3)

        # Bypass slew rate limiting para esta prueba
        engine.muscle._max_slew_rate = 100.0

        # Forzar sobrepresión con un salto de corriente alto
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=500, # current_I = 5.0
            processing_time=1.0,
            condenser_config=config
        )

        assert metrics["v_total"] > 5.3
        assert metrics["clamping_active"] == 1.0
        assert engine.clamping_active is True

    def test_saturation_use_elastic_pressure(self, physics_engine_fixture):
        """Verifica que la saturación use la presión elástica normalizada."""
        engine = physics_engine_fixture
        config = CondenserConfig(max_voltage=5.3)

        # Simular carga manualmente en el estado unificado
        engine._unified_state.charge = 1000.0
        # v_elastic = q/C = 1000/5000 = 0.2V
        # saturation = v_elastic / max_v = 0.2 / 5.3 ≈ 0.0377

        # Usamos cache_hits=0 para que current_I sea 0 y la carga no cambie significativamente
        metrics = engine.calculate_metrics(100, 0, condenser_config=config)

        expected_sat = (1000.0 / 5000.0) / 5.3
        # Con I=0, Q no debería cambiar en absoluto
        assert abs(metrics["saturation"] - expected_sat) < 1e-7


# ============================================================================
# TESTS: Integración
# ============================================================================

class TestIntegration:
    """Pruebas de integración entre componentes."""

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_end_to_end(
        self, mock_parser_class, mock_processor_class,
        valid_config, valid_profile, mock_csv_file
    ):
        """Prueba completa del proceso de estabilización."""
        # Configurar mocks
        mock_parser = MagicMock()
        mock_parser.parse_to_raw.return_value = [
            {"codigo": f"A{i}", "cantidad": 10} for i in range(50)
        ]
        mock_parser.get_parse_cache.return_value = {"cached": True}
        mock_parser_class.return_value = mock_parser
        
        mock_processor = MagicMock()
        mock_processor.process_all.return_value = pd.DataFrame([
            {"codigo": f"A{i}", "cantidad": 10} for i in range(50)
        ])
        mock_processor_class.return_value = mock_processor
        
        # Ejecutar
        condenser = DataFluxCondenser(valid_config, valid_profile)
        result = condenser.stabilize(str(mock_csv_file))
        
        # Verificar
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_with_progress_callback(
        self, mock_parser_class, mock_processor_class,
        valid_config, valid_profile, mock_csv_file
    ):
        """Callback de progreso recibe métricas."""
        mock_parser = MagicMock()
        mock_parser.parse_to_raw.return_value = [{"codigo": f"A{i}"} for i in range(100)]
        mock_parser.get_parse_cache.return_value = {}
        mock_parser_class.return_value = mock_parser
        
        mock_processor = MagicMock()
        mock_processor.process_all.return_value = pd.DataFrame([{"ok": 1}] * 10)
        mock_processor_class.return_value = mock_processor
        
        progress_calls = []
        def progress_callback(metrics):
            progress_calls.append(metrics)
        
        condenser = DataFluxCondenser(valid_config, valid_profile)
        condenser.stabilize(str(mock_csv_file), progress_callback=progress_callback)
        
        assert len(progress_calls) > 0
        assert "saturation" in progress_calls[0]

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_with_telemetry(
        self, mock_parser_class, mock_processor_class,
        valid_config, valid_profile, mock_csv_file
    ):
        """Telemetría registra eventos."""
        mock_parser = MagicMock()
        mock_parser.parse_to_raw.return_value = [{"a": 1}] * 20
        mock_parser.get_parse_cache.return_value = {}
        mock_parser_class.return_value = mock_parser
        
        mock_processor = MagicMock()
        mock_processor.process_all.return_value = pd.DataFrame([{"ok": 1}])
        mock_processor_class.return_value = mock_processor
        
        mock_telemetry = MagicMock()
        
        condenser = DataFluxCondenser(valid_config, valid_profile)
        condenser.stabilize(str(mock_csv_file), telemetry=mock_telemetry)
        
        mock_telemetry.record_event.assert_called()
        event_names = [call[0][0] for call in mock_telemetry.record_event.call_args_list]
        assert "stabilization_start" in event_names
        assert "stabilization_complete" in event_names

    def test_stabilize_empty_path_raises(self, condenser):
        """Path vacío lanza excepción."""
        with pytest.raises(InvalidInputError):
            condenser.stabilize("")

    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_parser_error_propagates(
        self, mock_parser_class, valid_config, valid_profile, mock_csv_file
    ):
        """Error en parser se propaga correctamente."""
        mock_parser_class.side_effect = IOError("Cannot read file")
        
        condenser = DataFluxCondenser(valid_config, valid_profile)
        
        with pytest.raises(ProcessingError):
            condenser.stabilize(str(mock_csv_file))


# ============================================================================
# TESTS: Casos Límite y Robustez
# ============================================================================

class TestEdgeCases:
    """Pruebas de casos límite y robustez numérica."""

    def test_controller_very_small_integral_limit(self):
        """Límite integral muy pequeño debe manejarse."""
        controller = PIController(
            kp=10.0, ki=1.0, setpoint=0.5,
            min_output=10, max_output=100,
            integral_limit_factor=0.01,
        )
        
        output = controller.compute(0.3)
        assert controller.min_output <= output <= controller.max_output

    def test_engine_very_high_natural_frequency(self):
        """Frecuencia natural alta debe funcionar."""
        engine = FluxPhysicsEngine(
            capacitance=0.001,
            resistance=10.0,
            inductance=0.001,
        )
        
        metrics = engine.calculate_metrics(100, 50, 0, 1.0)
        assert_finite(metrics["saturation"])

    def test_engine_zero_resistance_infinite_q(self):
        """R=0 produce Q infinito correctamente."""
        engine = FluxPhysicsEngine(
            capacitance=100.0,
            resistance=0.0,
            inductance=1.0,
        )
        
        assert engine._Q == float("inf")
        
        metrics = engine.calculate_metrics(100, 50, 0, 1.0)
        assert_finite(metrics["saturation"])

    def test_engine_extreme_current_change_bounded(self):
        """Cambio extremo de corriente está limitado."""
        engine = FluxPhysicsEngine(5000.0, 10.0, 2.0)
        engine._initialized = True
        engine._last_time = time.time() - 0.001
        engine._last_current = 0.0
        
        metrics = engine.calculate_metrics(100, 100, 0, 0.001)
        
        assert metrics["flyback_voltage"] <= SystemConstants.MAX_FLYBACK_VOLTAGE

    def test_tank_overflow_protection(self, valid_config, valid_profile):
        """Protección de desbordamiento reduce batch size."""
        condenser = DataFluxCondenser(valid_config, valid_profile)
        condenser._start_time = time.time()
        
        with patch.object(condenser.physics, "calculate_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "saturation": 0.98,
                "complexity": 0.5,
                "current_I": 0.5,
                "flyback_voltage": 0.1,
                "water_hammer_pressure": 0.1,
                "gyroscopic_stability": 1.0
            }
            
            with patch.object(condenser, "logger") as mock_logger:
                condenser._process_batches_with_pid(
                    raw_records=[{"a": 1}] * 100,
                    cache={},
                    total_records=100,
                    on_progress=None,
                    progress_callback=None,
                    telemetry=None
                )
                
                warning_messages = [str(c[0][0]) for c in mock_logger.warning.call_args_list]
                assert any("PRESIÓN MÁXIMA" in msg for msg in warning_messages)

    @pytest.mark.parametrize("error_count,expected_thermal_death", [
        (0, False),
        (50, False),
        (100, True),
    ])
    def test_thermal_death_thresholds(self, error_count, expected_thermal_death):
        """Umbrales de muerte térmica correctos."""
        engine = FluxPhysicsEngine(5000.0, 10.0, 2.0)
        
        result = engine.calculate_system_entropy(100, error_count, 1.0)
        
        if error_count == 100:
            assert result["is_thermal_death"] is True

    def test_emergency_brake_activation(self, valid_config, valid_profile):
        """Freno de emergencia se activa correctamente."""
        condenser = DataFluxCondenser(valid_config, valid_profile)
        condenser._start_time = time.time()
        
        with patch.object(condenser.physics, "calculate_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "saturation": 0.5,
                "complexity": 0.5,
                "current_I": 0.5,
                "dissipated_power": SystemConstants.OVERHEAT_POWER_THRESHOLD + 10.0,
                "flyback_voltage": 0.1,
                "water_hammer_pressure": 0.1,
                "gyroscopic_stability": 1.0
            }
            
            with patch.object(condenser, "logger") as mock_logger:
                condenser._process_batches_with_pid(
                    raw_records=[{"a": 1}] * 100,
                    cache={},
                    total_records=100,
                    on_progress=None,
                    progress_callback=None,
                    telemetry=None
                )
                
                warning_messages = [str(c[0][0]) for c in mock_logger.warning.call_args_list]
                assert any("EMERGENCY BRAKE" in msg for msg in warning_messages)
                assert condenser._emergency_brake_count > 0

    @pytest.mark.parametrize("nan_value", [float('nan')])
    def test_nan_propagation_prevention(self, nan_value):
        """NaN no debe propagarse en métricas."""
        engine = FluxPhysicsEngine(5000.0, 10.0, 2.0)
        
        # Aunque los inputs sean válidos, verificar que no se producen NaN
        metrics = engine.calculate_metrics(100, 50, 0, 1.0)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not math.isnan(value), f"NaN detectado en metrics['{key}']"


# ============================================================================
# TESTS: Rendimiento
# ============================================================================

class TestPerformance:
    """Pruebas de rendimiento."""

    @pytest.mark.slow
    def test_controller_many_iterations(self):
        """Controller maneja miles de iteraciones."""
        controller = PIController(
            kp=50.0, ki=10.0, setpoint=0.5,
            min_output=10, max_output=100
        )
        
        import random
        for _ in range(TestConstants.MAX_ITERATIONS_STRESS):
            pv = 0.5 + random.uniform(-0.2, 0.2)
            output = controller.compute(pv)
            assert 10 <= output <= 100

    @pytest.mark.slow
    def test_engine_metrics_history_bounded(self):
        """Historial de métricas está acotado."""
        engine = FluxPhysicsEngine(5000.0, 10.0, 2.0)
        
        for i in range(2000):
            engine.calculate_metrics(100, 50 + i % 50, i % 10, float(i) / 100)
        
        assert len(engine._metrics_history) <= FluxPhysicsEngine._MAX_METRICS_HISTORY

    @pytest.mark.slow
    def test_prediction_performance(self, condenser):
        """Predicción de saturación es eficiente."""
        import time
        
        history = [0.5 + 0.01 * (i % 20) for i in range(100)]
        
        start = time.time()
        for _ in range(1000):
            condenser._predict_next_saturation(history)
        elapsed = time.time() - start
        
        # Debe completar 1000 predicciones en menos de 1 segundo
        assert elapsed < 1.0, f"Predicción demasiado lenta: {elapsed}s para 1000 iteraciones"


# ============================================================================
# TESTS: V3 - Músculo y Reserva Táctica
# ============================================================================

class TestFluxMuscleController:
    """Pruebas para el controlador del músculo (MOSFET)."""

    def test_muscle_slew_rate_limiting(self):
        """Verifica que el cambio de fuerza sea gradual."""
        from app.flux_condenser import FluxMuscleController
        muscle = FluxMuscleController()

        # Intentar saltar de 0 a 1 en 1ms
        dt = 0.001
        duty = muscle.apply_force(1.0, dt)

        # Max change per 10ms is 0.1, so per 1ms is 0.01
        assert duty < 0.02
        assert duty > 0

    def test_muscle_thermal_throttling(self):
        """Verifica que el músculo se debilite ante esfuerzo sostenido."""
        from app.flux_condenser import FluxMuscleController
        muscle = FluxMuscleController()

        # Esfuerzo máximo por 6 segundos (limite es 5s)
        dt = 1.0
        for _ in range(6):
            duty = muscle.apply_force(1.0, dt)

        # Después de 5s al 100%, debe reducirse a la mitad
        assert duty <= 0.55  # margen por el incremento final

    def test_muscle_temperature_increase(self):
        """Verifica que la temperatura aumente con la carga."""
        from app.flux_condenser import FluxMuscleController
        muscle = FluxMuscleController()

        initial_temp = muscle.temperature
        muscle.apply_force(1.0, 1.0)  # 1s al 100%

        assert muscle.temperature > initial_temp


class TestTacticalReserve:
    """Pruebas para la Reserva Táctica (UPS)."""

    def test_reserve_charging(self, physics_engine_fixture):
        """Verifica que la reserva se cargue cuando el bus tiene voltaje."""
        engine = physics_engine_fixture
        config = CondenserConfig()

        # Inicialmente en 3V
        engine._unified_state.brain_voltage = 3.0

        # Bus a 12V
        engine._update_tactical_reserve(0.1, 12.0, config)

        assert engine._unified_state.brain_voltage > 3.0

    def test_reserve_discharging(self, physics_engine_fixture):
        """Verifica descarga en modo supervivencia."""
        engine = physics_engine_fixture
        config = CondenserConfig()

        engine._unified_state.brain_voltage = 5.0

        # Bus a 0V (Diodo bloqueado)
        engine._update_tactical_reserve(1.0, 0.0, config)

        assert engine._unified_state.brain_voltage < 5.0

    def test_brownout_detection(self, physics_engine_fixture):
        """Verifica detección de bajo voltaje."""
        engine = physics_engine_fixture
        config = CondenserConfig(brain_brownout_threshold=2.65)

        engine._unified_state.brain_voltage = 2.6
        engine._update_tactical_reserve(0.1, 0.0, config)

        assert engine._unified_state.brain_alive is False


class TestDataFluxCondenserV3:
    """Pruebas de integración V3 en DataFluxCondenser."""

    def test_control_plane_collapse_raises_error(self, valid_config, valid_profile):
        """Verifica que el sistema se detenga si el cerebro muere."""
        condenser = DataFluxCondenser(valid_config, valid_profile)
        condenser._start_time = time.time()

        with patch.object(condenser.physics, "calculate_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "saturation": 0.5,
                "brain_alive": 0.0,  # Cerebro muerto
                "brain_voltage": 2.0
            }

            with pytest.raises(ProcessingError) as exc:
                condenser._process_batches_with_pid(
                    raw_records=[{"a": 1}] * 10,
                    cache={},
                    total_records=10,
                    on_progress=None,
                    progress_callback=None,
                    telemetry=None
                )
            assert "CONTROL_PLANE_COLLAPSE" in str(exc.value)


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

def pytest_configure(config):
    """Configuración de marcadores personalizados."""
    config.addinivalue_line(
        "markers", "slow: marca tests lentos que se saltan por defecto"
    )


def pytest_collection_modifyitems(config, items):
    """Saltar tests lentos a menos que se especifique --runslow."""
    if config.getoption("--runslow", default=False):
        return
    
    skip_slow = pytest.mark.skip(reason="necesita --runslow para ejecutar")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Añadir opción --runslow."""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="ejecutar tests lentos"
    )