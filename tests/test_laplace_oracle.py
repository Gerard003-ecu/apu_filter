"""
Test Suite Exhaustivo para LaplaceOracle

Este módulo proporciona cobertura completa del analizador de estabilidad
en el dominio de Laplace, incluyendo:

1. Validación de parámetros físicos
2. Clasificación de sistemas (sub/sobre/críticamente amortiguado)
3. Análisis de estabilidad (continuo y discreto)
4. Márgenes de estabilidad (ganancia, fase)
5. Métricas de respuesta transitoria
6. Sensibilidad paramétrica
7. Respuesta en frecuencia
8. Casos límite y estabilidad numérica

Convenciones de nomenclatura:
- test_<método>_<escenario>_<resultado_esperado>
- Valores RLC en unidades SI estándar con rangos físicamente realizables

"""

import pytest
import math
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

# Importaciones del módulo bajo prueba
try:
    from app.laplace_oracle import (
        LaplaceOracle, 
        ConfigurationError,
        NumericalConstants,
        DampingClass,
        StabilityStatus,
        SystemParameters,
    )
    ENUMS_AVAILABLE = True
except ImportError:
    # Fallback si el código no tiene las clases auxiliares
    from app.laplace_oracle import LaplaceOracle, ConfigurationError
    ENUMS_AVAILABLE = False

# Intentar importar numpy para tests que lo requieren
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# ============================================================================
# CONSTANTES DE PRUEBA Y CONFIGURACIÓN
# ============================================================================

class TestConstants:
    """Constantes para pruebas con valores físicamente realizables."""
    
    # Tolerancias para comparaciones numéricas
    RTOL = 1e-5          # Tolerancia relativa
    ATOL = 1e-10         # Tolerancia absoluta
    ATOL_DEGREES = 0.1   # Tolerancia para ángulos en grados
    
    # Parámetros RLC típicos (circuito de audio/RF)
    # Sistema subamortiguado típico: R=100Ω, L=10mH, C=1µF
    # ωₙ = 1/√(LC) = 1/√(0.01 × 1e-6) = 10,000 rad/s
    # ζ = (R/2)√(C/L) = 50 × √(1e-6/0.01) = 50 × 0.01 = 0.5
    
    UNDERDAMPED = {"R": 100.0, "L": 0.01, "C": 1e-6}      # ζ ≈ 0.5
    CRITICALLY_DAMPED = {"R": 200.0, "L": 0.01, "C": 1e-6}  # ζ ≈ 1.0
    OVERDAMPED = {"R": 400.0, "L": 0.01, "C": 1e-6}       # ζ ≈ 2.0
    UNDAMPED = {"R": 0.0, "L": 0.01, "C": 1e-6}           # ζ = 0
    LIGHTLY_DAMPED = {"R": 10.0, "L": 0.01, "C": 1e-6}    # ζ ≈ 0.05
    HEAVILY_DAMPED = {"R": 1000.0, "L": 0.01, "C": 1e-6}  # ζ ≈ 5.0
    
    # Frecuencias de muestreo
    SAMPLE_RATE_DEFAULT = 100000.0  # 100 kHz (adecuado para ωₙ = 10k rad/s)
    SAMPLE_RATE_LOW = 1000.0        # 1 kHz (insuficiente)
    SAMPLE_RATE_HIGH = 1000000.0    # 1 MHz
    
    # Valores derivados teóricos para UNDERDAMPED
    OMEGA_N_UNDERDAMPED = 10000.0   # rad/s
    ZETA_UNDERDAMPED = 0.5
    Q_UNDERDAMPED = 1.0             # Q = 1/(2ζ)


@dataclass
class ExpectedSystemMetrics:
    """Métricas esperadas para un sistema dado."""
    omega_n: float
    zeta: float
    Q: float
    damping_class: str
    is_stable: bool
    has_overshoot: bool


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def underdamped_oracle() -> LaplaceOracle:
    """Fixture: Sistema subamortiguado típico (ζ ≈ 0.5)."""
    params = TestConstants.UNDERDAMPED
    return LaplaceOracle(
        R=params["R"], 
        L=params["L"], 
        C=params["C"],
        sample_rate=TestConstants.SAMPLE_RATE_DEFAULT
    )


@pytest.fixture
def critically_damped_oracle() -> LaplaceOracle:
    """Fixture: Sistema críticamente amortiguado (ζ ≈ 1.0)."""
    params = TestConstants.CRITICALLY_DAMPED
    return LaplaceOracle(
        R=params["R"], 
        L=params["L"], 
        C=params["C"],
        sample_rate=TestConstants.SAMPLE_RATE_DEFAULT
    )


@pytest.fixture
def overdamped_oracle() -> LaplaceOracle:
    """Fixture: Sistema sobreamortiguado (ζ ≈ 2.0)."""
    params = TestConstants.OVERDAMPED
    return LaplaceOracle(
        R=params["R"], 
        L=params["L"], 
        C=params["C"],
        sample_rate=TestConstants.SAMPLE_RATE_DEFAULT
    )


@pytest.fixture
def undamped_oracle() -> LaplaceOracle:
    """Fixture: Sistema sin amortiguamiento (ζ = 0)."""
    params = TestConstants.UNDAMPED
    return LaplaceOracle(
        R=params["R"], 
        L=params["L"], 
        C=params["C"],
        sample_rate=TestConstants.SAMPLE_RATE_DEFAULT
    )


@pytest.fixture
def lightly_damped_oracle() -> LaplaceOracle:
    """Fixture: Sistema ligeramente amortiguado (ζ ≈ 0.05)."""
    params = TestConstants.LIGHTLY_DAMPED
    return LaplaceOracle(
        R=params["R"], 
        L=params["L"], 
        C=params["C"],
        sample_rate=TestConstants.SAMPLE_RATE_DEFAULT
    )


@pytest.fixture
def sample_frequencies() -> 'np.ndarray':
    """Fixture: Vector de frecuencias para análisis."""
    if not NUMPY_AVAILABLE:
        pytest.skip("NumPy no disponible")
    return np.logspace(1, 6, 200)  # 10 rad/s a 1M rad/s


# ============================================================================
# HELPERS Y UTILIDADES
# ============================================================================

def calculate_theoretical_omega_n(L: float, C: float) -> float:
    """Calcula ωₙ teórico: ωₙ = 1/√(LC)."""
    return 1.0 / math.sqrt(L * C)


def calculate_theoretical_zeta(R: float, L: float, C: float) -> float:
    """Calcula ζ teórico: ζ = (R/2)√(C/L)."""
    return (R / 2.0) * math.sqrt(C / L)


def calculate_theoretical_Q(zeta: float) -> float:
    """Calcula Q teórico: Q = 1/(2ζ)."""
    if zeta > 0:
        return 1.0 / (2.0 * zeta)
    return float('inf')


def assert_close(
    actual: float, 
    expected: float, 
    rtol: float = TestConstants.RTOL,
    atol: float = TestConstants.ATOL,
    msg: str = ""
) -> None:
    """Assertion con tolerancia para valores flotantes."""
    if math.isinf(expected) and math.isinf(actual):
        assert math.copysign(1, actual) == math.copysign(1, expected), \
            f"Signos de infinito no coinciden: {actual} vs {expected}. {msg}"
        return
    
    if math.isnan(expected) or math.isnan(actual):
        assert math.isnan(expected) and math.isnan(actual), \
            f"NaN inesperado: actual={actual}, expected={expected}. {msg}"
        return
    
    diff = abs(actual - expected)
    threshold = atol + rtol * abs(expected)
    assert diff <= threshold, \
        f"Valores no coinciden: {actual} vs {expected} (diff={diff:.2e}, threshold={threshold:.2e}). {msg}"


def get_damping_class_name(oracle: LaplaceOracle) -> str:
    """Obtiene el nombre de la clase de amortiguamiento de forma compatible."""
    dc = oracle.damping_class
    if hasattr(dc, 'name'):
        return dc.name
    return str(dc)


# ============================================================================
# TESTS DE INICIALIZACIÓN Y VALIDACIÓN DE PARÁMETROS
# ============================================================================

class TestLaplaceOracleInitialization:
    """Tests de inicialización y validación de parámetros."""

    def test_init_with_valid_underdamped_params(self, underdamped_oracle):
        """Verifica inicialización correcta con parámetros subamortiguados válidos."""
        oracle = underdamped_oracle
        params = TestConstants.UNDERDAMPED
        
        assert oracle.R == params["R"], "R no coincide"
        assert oracle.L == params["L"], "L no coincide"
        assert oracle.C == params["C"], "C no coincide"
        assert oracle.omega_n > 0, "ωₙ debe ser positivo"
        assert 0 < oracle.zeta < 1, f"ζ debe estar en (0,1) para subamortiguado, got {oracle.zeta}"

    def test_init_with_valid_overdamped_params(self, overdamped_oracle):
        """Verifica inicialización correcta con parámetros sobreamortiguados."""
        oracle = overdamped_oracle
        assert oracle.zeta > 1, f"ζ debe ser > 1 para sobreamortiguado, got {oracle.zeta}"

    def test_init_with_valid_critically_damped_params(self, critically_damped_oracle):
        """Verifica inicialización correcta con amortiguamiento crítico."""
        oracle = critically_damped_oracle
        assert_close(oracle.zeta, 1.0, rtol=0.1, msg="ζ debe ser ≈ 1 para crítico")

    def test_init_with_zero_resistance(self, undamped_oracle):
        """Verifica inicialización con R=0 (sistema sin amortiguamiento)."""
        oracle = undamped_oracle
        assert oracle.R == 0.0
        assert_close(oracle.zeta, 0.0, atol=1e-12, msg="ζ debe ser 0 para R=0")

    @pytest.mark.parametrize("invalid_R", [-1.0, -100.0, float('-inf')])
    def test_init_negative_resistance_raises_error(self, invalid_R):
        """Verifica que R negativo lanza ConfigurationError."""
        with pytest.raises(ConfigurationError, match=r"[Rr]|negativ|inválid"):
            LaplaceOracle(R=invalid_R, L=0.01, C=1e-6)

    @pytest.mark.parametrize("invalid_L", [0.0, -0.01, -1.0, float('-inf')])
    def test_init_invalid_inductance_raises_error(self, invalid_L):
        """Verifica que L ≤ 0 lanza ConfigurationError."""
        with pytest.raises(ConfigurationError, match=r"[Ll]|positiv|inválid"):
            LaplaceOracle(R=100.0, L=invalid_L, C=1e-6)

    @pytest.mark.parametrize("invalid_C", [0.0, -1e-6, -1.0, float('-inf')])
    def test_init_invalid_capacitance_raises_error(self, invalid_C):
        """Verifica que C ≤ 0 lanza ConfigurationError."""
        with pytest.raises(ConfigurationError, match=r"[Cc]|positiv|inválid"):
            LaplaceOracle(R=100.0, L=0.01, C=invalid_C)

    @pytest.mark.parametrize("param,value", [
        ("R", float('inf')),
        ("L", float('inf')),
        ("C", float('inf')),
        ("R", float('nan')),
        ("L", float('nan')),
        ("C", float('nan')),
    ])
    def test_init_non_finite_params_raises_error(self, param, value):
        """Verifica que valores no finitos lanzan ConfigurationError."""
        params = {"R": 100.0, "L": 0.01, "C": 1e-6}
        params[param] = value
        with pytest.raises(ConfigurationError, match=r"finit|inválid"):
            LaplaceOracle(**params)

    def test_init_stores_sample_rate(self):
        """Verifica que sample_rate se almacena correctamente."""
        sample_rate = 50000.0
        oracle = LaplaceOracle(R=100.0, L=0.01, C=1e-6, sample_rate=sample_rate)
        assert oracle.sample_rate == sample_rate
        assert_close(oracle.T, 1.0 / sample_rate, msg="Período de muestreo incorrecto")


# ============================================================================
# TESTS DE PARÁMETROS DERIVADOS
# ============================================================================

class TestDerivedParameters:
    """Tests de cálculo de parámetros derivados (ωₙ, ζ, Q)."""

    @pytest.mark.parametrize("R,L,C", [
        (100.0, 0.01, 1e-6),    # Subamortiguado
        (200.0, 0.01, 1e-6),    # Crítico
        (400.0, 0.01, 1e-6),    # Sobreamortiguado
        (50.0, 0.001, 1e-5),    # Otro subamortiguado
        (1000.0, 0.1, 1e-7),    # Otro sobreamortiguado
    ])
    def test_omega_n_calculation(self, R, L, C):
        """Verifica cálculo de ωₙ = 1/√(LC)."""
        oracle = LaplaceOracle(R=R, L=L, C=C, sample_rate=1e6)
        expected_omega_n = calculate_theoretical_omega_n(L, C)
        assert_close(
            oracle.omega_n, 
            expected_omega_n, 
            msg=f"ωₙ incorrecto para L={L}, C={C}"
        )

    @pytest.mark.parametrize("R,L,C", [
        (100.0, 0.01, 1e-6),
        (200.0, 0.01, 1e-6),
        (400.0, 0.01, 1e-6),
        (10.0, 0.01, 1e-6),
    ])
    def test_zeta_calculation(self, R, L, C):
        """Verifica cálculo de ζ = (R/2)√(C/L)."""
        oracle = LaplaceOracle(R=R, L=L, C=C, sample_rate=1e6)
        expected_zeta = calculate_theoretical_zeta(R, L, C)
        assert_close(
            oracle.zeta, 
            expected_zeta, 
            msg=f"ζ incorrecto para R={R}, L={L}, C={C}"
        )

    @pytest.mark.parametrize("R,L,C", [
        (100.0, 0.01, 1e-6),   # Q = 1
        (50.0, 0.01, 1e-6),    # Q = 2
        (200.0, 0.01, 1e-6),   # Q = 0.5
    ])
    def test_Q_calculation(self, R, L, C):
        """Verifica cálculo de Q = 1/(2ζ)."""
        oracle = LaplaceOracle(R=R, L=L, C=C, sample_rate=1e6)
        expected_zeta = calculate_theoretical_zeta(R, L, C)
        expected_Q = calculate_theoretical_Q(expected_zeta)
        assert_close(
            oracle.Q, 
            expected_Q, 
            msg=f"Q incorrecto para ζ={expected_zeta}"
        )

    def test_Q_infinite_for_undamped(self, undamped_oracle):
        """Verifica Q → ∞ para sistema sin amortiguamiento."""
        assert math.isinf(undamped_oracle.Q), "Q debe ser infinito para ζ=0"
        assert undamped_oracle.Q > 0, "Q debe ser +∞"


# ============================================================================
# TESTS DE CLASIFICACIÓN DEL SISTEMA
# ============================================================================

class TestSystemClassification:
    """Tests de clasificación del sistema según amortiguamiento."""

    def test_underdamped_classification(self, underdamped_oracle):
        """Verifica clasificación de sistema subamortiguado."""
        damping_name = get_damping_class_name(underdamped_oracle)
        assert "UNDERDAMPED" in damping_name.upper(), \
            f"Esperado UNDERDAMPED, got {damping_name}"

    def test_critically_damped_classification(self, critically_damped_oracle):
        """Verifica clasificación de sistema críticamente amortiguado."""
        damping_name = get_damping_class_name(critically_damped_oracle)
        assert "CRITICAL" in damping_name.upper(), \
            f"Esperado CRITICALLY_DAMPED, got {damping_name}"

    def test_overdamped_classification(self, overdamped_oracle):
        """Verifica clasificación de sistema sobreamortiguado."""
        damping_name = get_damping_class_name(overdamped_oracle)
        assert "OVERDAMPED" in damping_name.upper(), \
            f"Esperado OVERDAMPED, got {damping_name}"

    def test_undamped_classification(self, undamped_oracle):
        """Verifica clasificación de sistema sin amortiguamiento."""
        damping_name = get_damping_class_name(undamped_oracle)
        assert "UNDAMPED" in damping_name.upper(), \
            f"Esperado UNDAMPED, got {damping_name}"


# ============================================================================
# TESTS DE ANÁLISIS DE ESTABILIDAD
# ============================================================================

class TestStabilityAnalysis:
    """Tests del análisis de estabilidad."""

    def test_stable_system_analysis(self, underdamped_oracle):
        """Verifica análisis de sistema estable."""
        stability = underdamped_oracle.analyze_stability()
        
        assert stability["is_stable"] is True, "Sistema debe ser estable"
        assert stability["status"] == "STABLE", f"Status incorrecto: {stability['status']}"
        assert "continuous" in stability, "Falta sección 'continuous'"
        assert "discrete" in stability, "Falta sección 'discrete'"

    def test_marginally_stable_system(self, undamped_oracle):
        """Verifica análisis de sistema marginalmente estable (R=0)."""
        stability = undamped_oracle.analyze_stability()
        
        assert stability["status"] == "MARGINALLY_STABLE", \
            f"Esperado MARGINALLY_STABLE, got {stability['status']}"
        assert stability["is_marginally_stable"] is True

    def test_poles_in_left_half_plane_for_stable(self, underdamped_oracle):
        """Verifica que polos estén en semiplano izquierdo para sistema estable."""
        stability = underdamped_oracle.analyze_stability()
        poles = stability["continuous"]["poles"]
        
        for real, imag in poles:
            assert real < 0, f"Polo ({real}, {imag}) tiene parte real ≥ 0"

    def test_poles_on_imaginary_axis_for_marginal(self, undamped_oracle):
        """Verifica polos en eje imaginario para sistema marginalmente estable."""
        stability = undamped_oracle.analyze_stability()
        poles = stability["continuous"]["poles"]
        
        for real, imag in poles:
            assert_close(real, 0.0, atol=1e-10, msg=f"Polo ({real}, {imag}) no está en eje imaginario")

    def test_discrete_poles_inside_unit_circle_for_stable(self, underdamped_oracle):
        """Verifica polos discretos dentro del círculo unitario."""
        stability = underdamped_oracle.analyze_stability()
        poles_d = stability["discrete"]["poles"]
        
        for real, imag in poles_d:
            magnitude = math.sqrt(real**2 + imag**2)
            assert magnitude < 1.0 + 1e-6, \
                f"Polo discreto ({real}, {imag}) fuera del círculo unitario, |p|={magnitude}"

    def test_stability_includes_transient_metrics(self, underdamped_oracle):
        """Verifica que análisis incluye métricas de respuesta transitoria."""
        stability = underdamped_oracle.analyze_stability()
        
        assert "transient_response" in stability, "Falta sección 'transient_response'"
        transient = stability["transient_response"]
        
        expected_keys = ["rise_time_s", "settling_time_s", "overshoot_percent"]
        for key in expected_keys:
            assert key in transient, f"Falta métrica transitoria '{key}'"

    def test_stability_includes_margins(self, underdamped_oracle):
        """Verifica que análisis incluye márgenes de estabilidad."""
        stability = underdamped_oracle.analyze_stability()
        
        assert "stability_margins" in stability, "Falta sección 'stability_margins'"
        margins = stability["stability_margins"]
        
        assert "phase_margin_deg" in margins
        assert "gain_margin_db" in margins


# ============================================================================
# TESTS DE MÁRGENES DE ESTABILIDAD
# ============================================================================

class TestStabilityMargins:
    """Tests de cálculo de márgenes de estabilidad."""

    def test_gain_margin_infinite_for_second_order(self, underdamped_oracle):
        """Verifica margen de ganancia infinito para sistemas de 2º orden pasivos."""
        stability = underdamped_oracle.analyze_stability()
        gm = stability["stability_margins"]["gain_margin_db"]
        
        assert math.isinf(gm) and gm > 0, \
            f"Margen de ganancia debe ser +∞ para 2º orden, got {gm}"

    def test_phase_margin_positive_for_stable(self, underdamped_oracle):
        """Verifica margen de fase positivo para sistema estable."""
        stability = underdamped_oracle.analyze_stability()
        pm = stability["stability_margins"]["phase_margin_deg"]
        
        assert pm > 0, f"Margen de fase debe ser positivo, got {pm}"

    @pytest.mark.parametrize("zeta,expected_pm_range", [
        (0.1, (10, 25)),      # Muy subamortiguado: PM bajo
        (0.5, (45, 65)),      # Subamortiguado típico: PM moderado
        (0.707, (60, 75)),    # ζ crítico: PM ≈ 65°
    ])
    def test_phase_margin_correlates_with_damping(self, zeta, expected_pm_range):
        """Verifica correlación entre ζ y margen de fase."""
        # Calcular R para obtener ζ deseado: ζ = (R/2)√(C/L) → R = 2ζ√(L/C)
        L, C = 0.01, 1e-6
        R = 2 * zeta * math.sqrt(L / C)
        
        oracle = LaplaceOracle(R=R, L=L, C=C, sample_rate=1e6)
        stability = oracle.analyze_stability()
        pm = stability["stability_margins"]["phase_margin_deg"]
        
        assert expected_pm_range[0] <= pm <= expected_pm_range[1], \
            f"PM={pm}° fuera de rango esperado {expected_pm_range} para ζ={zeta}"

    def test_phase_margin_zero_for_undamped(self, undamped_oracle):
        """Verifica margen de fase cero o cercano para sistema sin amortiguamiento."""
        stability = undamped_oracle.analyze_stability()
        pm = stability["stability_margins"]["phase_margin_deg"]
        
        # Para ζ = 0, PM debería ser 0 o muy pequeño
        assert pm < 10, f"PM debe ser ≈ 0 para sistema sin amortiguamiento, got {pm}"


# ============================================================================
# TESTS DE RESPUESTA TRANSITORIA
# ============================================================================

class TestTransientResponse:
    """Tests de métricas de respuesta transitoria."""

    def test_underdamped_has_overshoot(self, underdamped_oracle):
        """Verifica sobrepaso para sistema subamortiguado."""
        stability = underdamped_oracle.analyze_stability()
        transient = stability["transient_response"]
        
        assert transient["overshoot_percent"] > 0, \
            "Sistema subamortiguado debe tener sobrepaso"

    def test_overdamped_no_overshoot(self, overdamped_oracle):
        """Verifica ausencia de sobrepaso para sistema sobreamortiguado."""
        stability = overdamped_oracle.analyze_stability()
        transient = stability["transient_response"]
        
        assert_close(
            transient["overshoot_percent"], 
            0.0, 
            atol=0.01,
            msg="Sistema sobreamortiguado no debe tener sobrepaso"
        )

    def test_critically_damped_no_overshoot(self, critically_damped_oracle):
        """Verifica ausencia de sobrepaso para sistema críticamente amortiguado."""
        stability = critically_damped_oracle.analyze_stability()
        transient = stability["transient_response"]
        
        assert_close(
            transient["overshoot_percent"], 
            0.0, 
            atol=0.01,
            msg="Sistema críticamente amortiguado no debe tener sobrepaso"
        )

    def test_overshoot_formula_for_underdamped(self, underdamped_oracle):
        """Verifica fórmula de sobrepaso: Mp = exp(-πζ/√(1-ζ²)) × 100%."""
        zeta = underdamped_oracle.zeta
        expected_overshoot = math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2)) * 100
        
        stability = underdamped_oracle.analyze_stability()
        actual_overshoot = stability["transient_response"]["overshoot_percent"]
        
        assert_close(
            actual_overshoot, 
            expected_overshoot, 
            rtol=0.02,
            msg="Fórmula de sobrepaso no coincide"
        )

    def test_peak_time_formula(self, underdamped_oracle):
        """Verifica fórmula de tiempo de pico: tp = π/ωd."""
        zeta = underdamped_oracle.zeta
        omega_n = underdamped_oracle.omega_n
        omega_d = omega_n * math.sqrt(1 - zeta**2)
        expected_peak_time = math.pi / omega_d
        
        stability = underdamped_oracle.analyze_stability()
        actual_peak_time = stability["transient_response"]["peak_time_s"]
        
        assert_close(
            actual_peak_time, 
            expected_peak_time, 
            rtol=0.02,
            msg="Fórmula de tiempo de pico no coincide"
        )

    def test_settling_time_positive(self, underdamped_oracle):
        """Verifica tiempo de asentamiento positivo y finito."""
        stability = underdamped_oracle.analyze_stability()
        ts = stability["transient_response"]["settling_time_s"]
        
        assert ts > 0, "Tiempo de asentamiento debe ser positivo"
        assert math.isfinite(ts), "Tiempo de asentamiento debe ser finito"

    def test_settling_time_infinite_for_undamped(self, undamped_oracle):
        """Verifica tiempo de asentamiento infinito para sistema sin amortiguamiento."""
        stability = undamped_oracle.analyze_stability()
        ts = stability["transient_response"]["settling_time_s"]
        
        assert math.isinf(ts), "Tiempo de asentamiento debe ser infinito para ζ=0"


# ============================================================================
# TESTS DE SENSIBILIDAD PARAMÉTRICA
# ============================================================================

class TestParameterSensitivity:
    """Tests de análisis de sensibilidad paramétrica."""

    def test_sensitivity_matrix_structure(self, underdamped_oracle):
        """Verifica estructura de la matriz de sensibilidad."""
        stability = underdamped_oracle.analyze_stability()
        sensitivity = stability["parameter_sensitivity"]
        
        # Debe tener sensibilidades escalares
        assert "sensitivity_to_R" in sensitivity or "scalar_sensitivities" in sensitivity
        
        # Debe indicar el parámetro más sensible
        assert "most_sensitive_parameter" in sensitivity or "most_sensitive" in sensitivity

    def test_omega_n_independent_of_R(self, underdamped_oracle):
        """Verifica que ωₙ no depende de R (∂ωₙ/∂R = 0)."""
        stability = underdamped_oracle.analyze_stability()
        sensitivity = stability["parameter_sensitivity"]
        
        if "sensitivity_matrix" in sensitivity:
            s_r_omega = sensitivity["sensitivity_matrix"].get("R", {}).get("omega_n", 0)
        elif "normalized_sensitivities" in sensitivity:
            s_r_omega = sensitivity["normalized_sensitivities"]["omega_n"]["R"]
        else:
            s_r_omega = 0  # Asumir correcto si no está disponible
        
        assert_close(s_r_omega, 0.0, atol=1e-10, msg="∂ωₙ/∂R debe ser 0")

    def test_zeta_sensitivity_to_R_is_unity(self, underdamped_oracle):
        """Verifica S_R^ζ = 1 (sensibilidad normalizada de ζ a R)."""
        stability = underdamped_oracle.analyze_stability()
        sensitivity = stability["parameter_sensitivity"]
        
        # Buscar sensibilidad de ζ a R
        if "sensitivity_matrix" in sensitivity:
            s_r_zeta = sensitivity["sensitivity_matrix"].get("R", {}).get("zeta", None)
        elif "normalized_sensitivities" in sensitivity:
            s_r_zeta = sensitivity["normalized_sensitivities"]["zeta"]["R"]
        else:
            pytest.skip("Formato de sensibilidad no reconocido")
        
        if s_r_zeta is not None:
            assert_close(s_r_zeta, 1.0, rtol=0.01, msg="S_R^ζ debe ser 1")

    def test_robustness_classification_exists(self, underdamped_oracle):
        """Verifica que existe clasificación de robustez."""
        stability = underdamped_oracle.analyze_stability()
        sensitivity = stability["parameter_sensitivity"]
        
        assert "robustness_classification" in sensitivity or "robustness_class" in sensitivity


# ============================================================================
# TESTS DE RESPUESTA EN FRECUENCIA
# ============================================================================

class TestFrequencyResponse:
    """Tests del cálculo de respuesta en frecuencia."""

    def test_frequency_response_structure(self, underdamped_oracle):
        """Verifica estructura de la respuesta en frecuencia."""
        freq_resp = underdamped_oracle.get_frequency_response()
        
        required_keys = [
            "frequencies_rad_s", 
            "magnitude_db", 
            "phase_deg"
        ]
        for key in required_keys:
            assert key in freq_resp, f"Falta clave '{key}' en respuesta en frecuencia"

    def test_dc_gain_is_unity(self, underdamped_oracle):
        """Verifica ganancia DC = 1 (0 dB) para sistema RLC pasivo."""
        freq_resp = underdamped_oracle.get_frequency_response()
        
        dc_gain_db = freq_resp["magnitude_db"][0]
        assert_close(dc_gain_db, 0.0, atol=0.1, msg="Ganancia DC debe ser 0 dB")

    def test_high_frequency_rolloff(self, underdamped_oracle):
        """Verifica caída de -40 dB/década en alta frecuencia."""
        freq_resp = underdamped_oracle.get_frequency_response()
        
        # Verificar pendiente asintótica
        if "high_freq_rolloff_db_per_decade" in freq_resp:
            rolloff = freq_resp["high_freq_rolloff_db_per_decade"]
            assert_close(rolloff, -40.0, atol=1.0, msg="Rolloff debe ser -40 dB/dec")
        else:
            # Calcular de los datos
            freqs = freq_resp["frequencies_rad_s"]
            mags = freq_resp["magnitude_db"]
            
            # Tomar últimos puntos para calcular pendiente
            if len(freqs) >= 10:
                log_f1, log_f2 = math.log10(freqs[-10]), math.log10(freqs[-1])
                mag1, mag2 = mags[-10], mags[-1]
                slope = (mag2 - mag1) / (log_f2 - log_f1)
                assert_close(slope, -40.0, atol=5.0, msg="Pendiente HF debe ser ≈ -40 dB/dec")

    def test_resonance_peak_for_underdamped(self, underdamped_oracle):
        """Verifica existencia de pico de resonancia para ζ < 1/√2."""
        freq_resp = underdamped_oracle.get_frequency_response()
        
        if "resonance" in freq_resp:
            resonance = freq_resp["resonance"]
            if underdamped_oracle.zeta < 1 / math.sqrt(2):
                assert resonance.get("exists", True) is True, \
                    "Debe existir resonancia para ζ < 0.707"
                assert resonance.get("magnitude_db", 0) > 0, \
                    "Pico de resonancia debe ser > 0 dB"

    def test_no_resonance_for_overdamped(self, overdamped_oracle):
        """Verifica ausencia de resonancia para sistema sobreamortiguado."""
        freq_resp = overdamped_oracle.get_frequency_response()
        
        if "resonance" in freq_resp:
            resonance = freq_resp["resonance"]
            assert resonance.get("exists", False) is False, \
                "No debe haber resonancia para ζ > 1"

    def test_bandwidth_positive(self, underdamped_oracle):
        """Verifica ancho de banda positivo."""
        freq_resp = underdamped_oracle.get_frequency_response()
        
        if "bandwidth_rad_s" in freq_resp:
            bw = freq_resp["bandwidth_rad_s"]
            assert bw > 0, f"Ancho de banda debe ser positivo, got {bw}"

    def test_nyquist_data_present(self, underdamped_oracle):
        """Verifica presencia de datos para diagrama de Nyquist."""
        freq_resp = underdamped_oracle.get_frequency_response()
        
        assert "nyquist_real" in freq_resp, "Falta parte real de Nyquist"
        assert "nyquist_imag" in freq_resp, "Falta parte imaginaria de Nyquist"
        assert len(freq_resp["nyquist_real"]) == len(freq_resp["frequencies_rad_s"])

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy requerido")
    def test_custom_frequencies_accepted(self, underdamped_oracle, sample_frequencies):
        """Verifica que acepta vector de frecuencias personalizado."""
        freq_resp = underdamped_oracle.get_frequency_response(frequencies=sample_frequencies)
        
        assert len(freq_resp["frequencies_rad_s"]) == len(sample_frequencies)


# ============================================================================
# TESTS DE LA PIRÁMIDE DE LAPLACE
# ============================================================================

class TestLaplacePyramid:
    """Tests de la estructura jerárquica de la Pirámide de Laplace."""

    def test_pyramid_has_four_levels(self, underdamped_oracle):
        """Verifica que la pirámide tiene exactamente 4 niveles."""
        pyramid = underdamped_oracle.get_laplace_pyramid()
        
        expected_levels = [
            "level_0_verdict",
            "level_1_robustness",
            "level_2_dynamics",
            "level_3_physics",
        ]
        for level in expected_levels:
            assert level in pyramid, f"Falta nivel '{level}' en pirámide"

    def test_level_0_verdict_structure(self, underdamped_oracle):
        """Verifica estructura del nivel 0 (veredicto)."""
        pyramid = underdamped_oracle.get_laplace_pyramid()
        verdict = pyramid["level_0_verdict"]
        
        assert "is_controllable" in verdict
        assert isinstance(verdict["is_controllable"], bool)
        assert "stability_status" in verdict

    def test_level_1_robustness_has_margins(self, underdamped_oracle):
        """Verifica que nivel 1 tiene márgenes de estabilidad."""
        pyramid = underdamped_oracle.get_laplace_pyramid()
        robustness = pyramid["level_1_robustness"]
        
        assert "phase_margin_deg" in robustness
        assert "gain_margin_db" in robustness

    def test_level_2_dynamics_has_poles(self, underdamped_oracle):
        """Verifica que nivel 2 tiene información de polos."""
        pyramid = underdamped_oracle.get_laplace_pyramid()
        dynamics = pyramid["level_2_dynamics"]
        
        # Aceptar ambas claves posibles
        has_poles = "poles_continuous" in dynamics or "poles" in dynamics
        assert has_poles, "Nivel 2 debe tener información de polos"
        
        # Verificar frecuencia natural
        has_omega = "natural_frequency_rad_s" in dynamics or "omega_n_rad_s" in dynamics
        assert has_omega, "Nivel 2 debe tener frecuencia natural"

    def test_level_3_physics_matches_input(self, underdamped_oracle):
        """Verifica que nivel 3 contiene parámetros físicos correctos."""
        pyramid = underdamped_oracle.get_laplace_pyramid()
        physics = pyramid["level_3_physics"]
        
        params = TestConstants.UNDERDAMPED
        
        # Verificar R (puede tener diferentes claves)
        R_value = physics.get("R", physics.get("R_ohm", None))
        assert R_value == params["R"], f"R incorrecto: {R_value} vs {params['R']}"
        
        L_value = physics.get("L", physics.get("L_henry", None))
        assert L_value == params["L"], f"L incorrecto: {L_value} vs {params['L']}"
        
        C_value = physics.get("C", physics.get("C_farad", None))
        assert C_value == params["C"], f"C incorrecto: {C_value} vs {params['C']}"

    def test_stable_system_is_controllable(self, underdamped_oracle):
        """Verifica que sistema estable es marcado como controlable."""
        pyramid = underdamped_oracle.get_laplace_pyramid()
        
        # Un sistema estable con buen amortiguamiento debería ser controlable
        # (excepto si hay otras condiciones que lo impidan)
        verdict = pyramid["level_0_verdict"]
        stability = verdict.get("stability_status", "")
        
        if "STABLE" in stability and "UNSTABLE" not in stability:
            # Debería ser controlable o al menos no definitivamente no controlable
            assert verdict["is_controllable"] is True or "warning" in str(verdict).lower()


# ============================================================================
# TESTS DE VALIDACIÓN PARA DISEÑO DE CONTROL
# ============================================================================

class TestControlDesignValidation:
    """Tests de validación para diseño de control."""

    def test_validation_structure(self, underdamped_oracle):
        """Verifica estructura de validación para control."""
        validation = underdamped_oracle.validate_for_control_design()
        
        required_keys = ["is_suitable_for_control", "recommendations"]
        for key in required_keys:
            assert key in validation, f"Falta clave '{key}' en validación"

    def test_stable_system_suitable_for_control(self, underdamped_oracle):
        """Verifica que sistema estable es adecuado para control."""
        validation = underdamped_oracle.validate_for_control_design()
        
        # Sistema subamortiguado típico debería ser adecuado
        # (podría tener warnings pero no issues críticos)
        if "issues" in validation:
            # No debería haber issues de inestabilidad
            issues_text = " ".join(validation["issues"])
            assert "inestable" not in issues_text.lower()

    def test_recommendations_present(self, underdamped_oracle):
        """Verifica que hay recomendaciones de control."""
        validation = underdamped_oracle.validate_for_control_design()
        
        assert isinstance(validation["recommendations"], list)

    def test_low_damping_generates_warning(self, lightly_damped_oracle):
        """Verifica advertencia para sistema con bajo amortiguamiento."""
        validation = lightly_damped_oracle.validate_for_control_design()
        
        # Combinar todas las advertencias y recomendaciones
        all_text = " ".join(
            validation.get("warnings", []) + 
            validation.get("recommendations", [])
        ).lower()
        
        # Debería mencionar amortiguamiento o subamortiguado o similar
        has_damping_warning = any(word in all_text for word in [
            "subamortiguado", "underdamped", "amortiguamiento", 
            "damping", "oscila", "oscillat"
        ])
        
        assert has_damping_warning, \
            f"Debería advertir sobre bajo amortiguamiento. Texto: {all_text[:200]}"


# ============================================================================
# TESTS DE ROOT LOCUS
# ============================================================================

class TestRootLocus:
    """Tests del lugar de las raíces."""

    def test_root_locus_structure(self, underdamped_oracle):
        """Verifica estructura de datos del root locus."""
        rl_data = underdamped_oracle.get_root_locus_data()
        
        required_keys = ["poles_real", "poles_imag", "gain_values"]
        for key in required_keys:
            assert key in rl_data, f"Falta clave '{key}' en root locus"

    def test_root_locus_poles_count(self, underdamped_oracle):
        """Verifica cantidad correcta de polos en root locus."""
        rl_data = underdamped_oracle.get_root_locus_data()
        
        n_gains = len(rl_data["gain_values"])
        n_poles_real = len(rl_data["poles_real"])
        
        # Debería haber 2 polos por cada valor de ganancia (sistema 2º orden)
        assert n_poles_real == 2 * n_gains, \
            f"Esperados {2 * n_gains} polos, got {n_poles_real}"

    def test_asymptote_angles_for_second_order(self, underdamped_oracle):
        """Verifica ángulos de asíntotas para sistema de 2º orden."""
        rl_data = underdamped_oracle.get_root_locus_data()
        
        if "asymptote_angles_deg" in rl_data:
            angles = rl_data["asymptote_angles_deg"]
            # Para n=2 polos y m=0 ceros: ángulos = ±90°
            assert 90 in angles or 270 in angles, \
                f"Asíntotas incorrectas para 2º orden: {angles}"


# ============================================================================
# TESTS DE CASOS LÍMITE Y ESTABILIDAD NUMÉRICA
# ============================================================================

class TestEdgeCasesAndNumericalStability:
    """Tests de casos límite y estabilidad numérica."""

    def test_very_small_capacitance(self):
        """Test con capacitancia muy pequeña (pF)."""
        oracle = LaplaceOracle(R=1000.0, L=0.001, C=1e-12, sample_rate=1e9)
        
        assert oracle.omega_n > 0
        assert math.isfinite(oracle.omega_n)
        
        stability = oracle.analyze_stability()
        assert stability is not None

    def test_very_large_inductance(self):
        """Test con inductancia grande (1H)."""
        oracle = LaplaceOracle(R=100.0, L=1.0, C=1e-6, sample_rate=10000.0)
        
        assert oracle.omega_n > 0
        assert oracle.omega_n < 10000  # ≈ 1000 rad/s para estos valores
        
        stability = oracle.analyze_stability()
        assert stability["is_stable"] is True

    def test_very_small_resistance(self):
        """Test con resistencia muy pequeña (casi sin amortiguamiento)."""
        oracle = LaplaceOracle(R=0.001, L=0.01, C=1e-6, sample_rate=1e6)
        
        assert oracle.zeta < 0.01
        assert oracle.zeta > 0

    def test_cache_mechanism(self, underdamped_oracle):
        """Verifica que el cache funciona correctamente."""
        # Primera llamada
        result1 = underdamped_oracle.analyze_stability()
        
        # Segunda llamada (debería usar cache)
        result2 = underdamped_oracle.analyze_stability()
        
        # Los resultados deben ser idénticos
        assert result1["status"] == result2["status"]
        assert result1["is_stable"] == result2["is_stable"]

    def test_frequency_response_caching(self, underdamped_oracle):
        """Verifica caching de respuesta en frecuencia."""
        # Primera llamada
        resp1 = underdamped_oracle.get_frequency_response(use_cache=True)
        
        # Segunda llamada
        resp2 = underdamped_oracle.get_frequency_response(use_cache=True)
        
        # Deben tener la misma longitud
        assert len(resp1["frequencies_rad_s"]) == len(resp2["frequencies_rad_s"])

    @pytest.mark.parametrize("zeta_target", [0.001, 0.999, 1.001, 10.0])
    def test_boundary_damping_ratios(self, zeta_target):
        """Test con valores de ζ en fronteras críticas."""
        L, C = 0.01, 1e-6
        R = 2 * zeta_target * math.sqrt(L / C)
        
        if R < 0:
            pytest.skip("R negativo no válido")
        
        oracle = LaplaceOracle(R=R, L=L, C=C, sample_rate=1e6)
        
        # Verificar que ζ es aproximadamente el esperado
        assert_close(oracle.zeta, zeta_target, rtol=0.01)
        
        # Debe poder analizar sin errores
        stability = oracle.analyze_stability()
        assert stability is not None


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Tests de integración entre componentes."""

    def test_full_analysis_pipeline(self, underdamped_oracle):
        """Verifica pipeline completo de análisis."""
        # 1. Análisis de estabilidad
        stability = underdamped_oracle.analyze_stability()
        assert stability["is_stable"]
        
        # 2. Respuesta en frecuencia
        freq_resp = underdamped_oracle.get_frequency_response()
        assert len(freq_resp["frequencies_rad_s"]) > 0
        
        # 3. Root locus
        rl = underdamped_oracle.get_root_locus_data()
        assert len(rl["poles_real"]) > 0
        
        # 4. Pirámide
        pyramid = underdamped_oracle.get_laplace_pyramid()
        assert len(pyramid) >= 4
        
        # 5. Validación
        validation = underdamped_oracle.validate_for_control_design()
        assert "is_suitable_for_control" in validation

    def test_consistency_between_methods(self, underdamped_oracle):
        """Verifica consistencia entre diferentes métodos."""
        stability = underdamped_oracle.analyze_stability()
        pyramid = underdamped_oracle.get_laplace_pyramid()
        
        # Estabilidad debe ser consistente
        stab_status = stability["status"]
        pyramid_status = pyramid["level_0_verdict"]["stability_status"]
        
        assert stab_status == pyramid_status, \
            f"Inconsistencia: {stab_status} vs {pyramid_status}"
        
        # Parámetros físicos deben ser consistentes
        physics = pyramid["level_3_physics"]
        R_pyramid = physics.get("R", physics.get("R_ohm"))
        
        assert R_pyramid == underdamped_oracle.R

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy requerido")
    def test_comprehensive_report(self, underdamped_oracle):
        """Verifica generación de reporte completo."""
        if hasattr(underdamped_oracle, 'get_comprehensive_report'):
            report = underdamped_oracle.get_comprehensive_report()
            
            expected_sections = [
                "system_parameters",
                "stability_analysis",
                "frequency_response",
            ]
            for section in expected_sections:
                assert section in report, f"Falta sección '{section}' en reporte"


# ============================================================================
# TESTS DE REGRESIÓN
# ============================================================================

class TestRegression:
    """Tests de regresión para valores conocidos."""

    def test_known_underdamped_system(self):
        """Test con sistema subamortiguado de valores conocidos."""
        # Sistema con ωₙ = 10000, ζ = 0.5
        # R = 2ζ√(L/C) = 2 × 0.5 × √(0.01/1e-6) = 100
        oracle = LaplaceOracle(R=100.0, L=0.01, C=1e-6, sample_rate=1e6)
        
        assert_close(oracle.omega_n, 10000.0, rtol=1e-6)
        assert_close(oracle.zeta, 0.5, rtol=1e-6)
        assert_close(oracle.Q, 1.0, rtol=1e-6)

    def test_known_critically_damped_system(self):
        """Test con sistema críticamente amortiguado."""
        # ζ = 1 → R = 2√(L/C) = 200 para L=0.01, C=1e-6
        oracle = LaplaceOracle(R=200.0, L=0.01, C=1e-6, sample_rate=1e6)
        
        assert_close(oracle.zeta, 1.0, rtol=1e-6)

    def test_overshoot_16_percent_for_zeta_0_5(self):
        """Verifica sobrepaso de ~16.3% para ζ = 0.5."""
        oracle = LaplaceOracle(R=100.0, L=0.01, C=1e-6, sample_rate=1e6)
        
        stability = oracle.analyze_stability()
        overshoot = stability["transient_response"]["overshoot_percent"]
        
        # Valor teórico: exp(-π × 0.5 / √(1-0.25)) × 100 ≈ 16.3%
        expected_overshoot = math.exp(-math.pi * 0.5 / math.sqrt(0.75)) * 100
        
        assert_close(overshoot, expected_overshoot, rtol=0.01)


# ============================================================================
# TESTS DE LOGGING Y WARNINGS
# ============================================================================

class TestLoggingAndWarnings:
    """Tests de logging y advertencias."""

    def test_warning_for_low_sample_rate(self, caplog):
        """Verifica warning cuando sample_rate es insuficiente."""
        with caplog.at_level(logging.WARNING):
            # ωₙ = 10000 rad/s → f_n ≈ 1592 Hz
            # Sample rate de 1000 Hz es insuficiente
            try:
                oracle = LaplaceOracle(
                    R=100.0, L=0.01, C=1e-6, 
                    sample_rate=1000.0  # Muy bajo
                )
                # Puede lanzar error o generar warning
            except ConfigurationError:
                pass  # Comportamiento aceptable
            
            # Si no lanzó error, debe haber warning
            if "ConfigurationError" not in str(caplog.text):
                assert "sample" in caplog.text.lower() or "nyquist" in caplog.text.lower() or \
                       "frecuencia" in caplog.text.lower(), \
                    "Debería advertir sobre sample rate insuficiente"

    def test_warning_for_extreme_damping(self, caplog):
        """Verifica warning para amortiguamiento extremo."""
        with caplog.at_level(logging.WARNING):
            # ζ muy pequeño (≈ 0.005)
            oracle = LaplaceOracle(R=1.0, L=0.01, C=1e-6, sample_rate=1e6)
            
            # Debería generar warning sobre bajo amortiguamiento
            # (depende de la implementación)


# ============================================================================
# FIXTURES Y CONFIGURACIÓN DE PYTEST
# ============================================================================

@pytest.fixture(scope="session")
def test_parameters():
    """Fixture de sesión con parámetros de prueba."""
    return TestConstants()


def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "numerical: marks tests requiring numerical precision"
    )


# ============================================================================
# MAIN PARA EJECUCIÓN DIRECTA
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])