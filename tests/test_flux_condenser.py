"""
Suite de Pruebas para el `DataFluxCondenser` - Versión Refinada V4.

Cobertura actualizada para las mejoras implementadas:
- Integración RK4 en FluxPhysicsEngine (antes RK2)
- PIController con ganancia adaptativa y estabilidad mejorada
- Entropía de Tsallis y correcciones estadísticas
- Predicción de saturación y recuperación en DataFluxCondenser
"""

import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

try:
    import numpy as np
except ImportError:
    np = None

from app.flux_condenser import (
    CondenserConfig,
    DataFluxCondenser,
    FluxPhysicsEngine,
    PIController,
    ProcessingStats,
)

# ==================== FIXTURES ====================


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
def sample_raw_records() -> List[Dict[str, Any]]:
    """Registros de ejemplo para pruebas."""
    return [
        {"codigo": f"A{i}", "cantidad": 10, "precio": 100.0, "insumo_line": f"line_{i}"}
        for i in range(100)
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    """Caché de parseo de ejemplo."""
    return {f"line_{i}": "data" for i in range(100)}


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    """Archivo CSV temporal para pruebas."""
    file_path = tmp_path / "test_data.csv"
    content = "codigo,cantidad,precio\n" + "\n".join([f"A{i},10,100.0" for i in range(100)])
    file_path.write_text(content)
    return file_path


# ==================== TESTS: PIController Refinado ====================


class TestPIController:
    """Pruebas unitarias para el controlador PI refinado V4."""

    @pytest.fixture
    def controller(self) -> PIController:
        """Controlador básico para pruebas."""
        return PIController(
            kp=50.0,
            ki=10.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
            integral_limit_factor=2.0,
        )

    def test_adaptive_integral_gain_on_windup(self, controller):
        """
        La ganancia integral debe reducirse cuando se detecta windup.
        Windup: Error pequeño pero saturación frecuente.
        """
        # Simular condiciones de windup: error constante pequeño y salida saturada
        initial_ki = controller.Ki

        # Ejecutar ciclo de detección
        # Error pequeño (0.02) y salida saturada (True)
        for _ in range(5):
            controller._adapt_integral_gain(0.02, True)

        # Ki adaptativa debe haberse reducido
        assert controller._ki_adaptive < initial_ki
        assert controller._ki_adaptive == initial_ki * 0.5

        # Simular recuperación: error cambia o no saturación
        for _ in range(5):
            controller._adapt_integral_gain(0.1, False)

        # Ki adaptativa debe restaurarse
        assert controller._ki_adaptive == initial_ki

    def test_ema_filter_adaptive_alpha(self, controller):
        """
        Alpha del filtro EMA debe adaptarse a la varianza del error.
        """
        # Caso 1: Señal estable (baja varianza) -> Alpha debería estabilizarse
        # en un valor, pero el adaptativo depende de la inversa de varianza.
        # Si varianza ~ 0, alpha sube para confiar más en medición (o bajar para suavizar menos?)
        # Ver lógica: adaptive_alpha = 0.1 + 0.4 / (1.0 + 5.0 * normalized_var)
        # Si var -> 0, alpha -> 0.1 + 0.4 = 0.5 (Reacción rápida)
        # Si var -> 1, alpha -> 0.1 + 0.4/6 ~ 0.16 (Mayor suavizado)

        # Resetear historial
        controller._error_history.clear()

        # Generar historial de bajo ruido
        for _ in range(10):
            controller._error_history.append(0.01)  # Varianza 0

        # Forzar actualización de alpha llamando a _apply_ema_filter
        controller._apply_ema_filter(0.5)
        stable_alpha = controller._ema_alpha

        # Caso 2: Señal ruidosa (alta varianza) -> Alpha bajo
        controller._error_history.clear()
        values = [0.1, 0.9, 0.2, 0.8, 0.1]  # Alta varianza
        for v in values:
            controller._error_history.append(v)

        controller._apply_ema_filter(0.5)
        noisy_alpha = controller._ema_alpha

        # Alpha en ruido (var alta) debe ser menor (más filtrado) que en estable (var baja)
        assert noisy_alpha < stable_alpha

    def test_lyapunov_stability_detection(self, controller):
        """
        Debe calcular exponente de Lyapunov y detectar inestabilidad.
        """
        # Simular divergencia (inestabilidad)
        # Error creciente exponencialmente
        errors = [0.01 * (1.5**i) for i in range(15)]

        # Inyectar errores en el cálculo
        controller._last_error = errors[0]
        for e in errors[1:]:
            controller._update_lyapunov_metric(e)
            controller._last_error = e

        lyapunov = controller.get_lyapunov_exponent()
        # Exponente positivo indica caos/divergencia
        assert lyapunov > 0.0

    def test_output_smoothing(self, controller):
        """La salida no debe cambiar bruscamente."""
        # Establecer salida inicial
        controller._last_output = 50

        # Forzar un cambio brusco en la entrada
        # El slew rate limit es 15% del rango (90 * 0.15 = 13.5)
        # Cambio bruto sería > 20
        output = controller.compute(0.0)

        change = abs(output - 50)
        # Ajustado a 15% (13.5 redondeado -> 14)
        assert change <= 15


# ==================== TESTS: FluxPhysicsEngine Refinado ====================


class TestFluxPhysicsEngine:
    """Pruebas del motor de física RLC refinado V4."""

    @pytest.fixture
    def engine(self) -> FluxPhysicsEngine:
        """Motor de física con parámetros por defecto."""
        return FluxPhysicsEngine(
            capacitance=5000.0,
            resistance=10.0,
            inductance=2.0,
        )

    def test_rk4_integration_stability(self, engine):
        """
        El integrador RK4 debe ser estable incluso con pasos grandes.
        Renombrado de test_rk2_integration_stability
        """
        # Paso de tiempo grande que podría desestabilizar Euler
        dt = 0.1
        current_I = 1.0

        # Evolucionar estado usando RK4
        Q, I = engine._evolve_state_rk4(current_I, dt)

        assert math.isfinite(Q)
        assert math.isfinite(I)
        # La energía no debe explotar
        energy = 0.5 * engine.L * I**2 + 0.5 * engine.C * (Q / engine.C) ** 2
        assert energy < 1e6  # Límite razonable

    def test_nonlinear_damping_at_high_energy(self, engine):
        """
        Debe aplicar amortiguamiento extra a alta energía.
        """
        # Inyectar estado de alta energía manualmente
        engine._state = [10000.0, 100.0]  # Q alto, I alto

        # Usar RK4
        engine._evolve_state_rk4(1.0, 0.01)

        # Verificar que se activó el factor de amortiguamiento no lineal
        assert engine._nonlinear_damping_factor < 1.0

    def test_tsallis_entropy_calculation(self, engine):
        """
        Cálculo de entropía de Tsallis y correcciones.
        """
        # Caso: distribución uniforme (máxima entropía)
        res = engine.calculate_system_entropy(100, 50, 1.0)

        assert "tsallis_entropy" in res
        assert "shannon_entropy_corrected" in res
        assert "kl_divergence" in res

        # KL Divergence debe ser 0 para uniforme (p=0.5)
        assert math.isclose(res["kl_divergence"], 0.0, abs_tol=1e-9)

        # Caso: distribución sesgada (baja entropía)
        res_skewed = engine.calculate_system_entropy(100, 0, 1.0)
        # KL debe ser positiva
        assert res_skewed["kl_divergence"] > 0

    def test_gyroscopic_stability_precession(self, engine):
        """
        Estabilidad giroscópica debe detectar precesión (cambio de eje).
        """
        # Inicializar
        engine.calculate_gyroscopic_stability(0.5)

        # Cambio brusco de "eje" (media móvil de corriente)
        # Si la corriente oscila violentamente, la estabilidad baja
        stabilities = []
        for i in range(10):
            val = 0.5 + 0.4 * ((-1) ** i)  # Oscilación fuerte
            s = engine.calculate_gyroscopic_stability(val)
            stabilities.append(s)

        # La estabilidad debería degradarse
        assert stabilities[-1] < 1.0


# ==================== TESTS: DataFluxCondenser Refinado ====================


class TestDataFluxCondenserRefined:
    """Pruebas para DataFluxCondenser con nuevas capacidades."""

    def test_predict_next_saturation(self, condenser):
        """
        Debe predecir saturación futura basada en tendencia.
        """
        # Tendencia creciente
        history = [0.5, 0.6, 0.7, 0.8]
        pred = condenser._predict_next_saturation(history)

        # Debe predecir > 0.8
        assert pred > 0.8
        assert pred <= 1.0

    def test_estimate_cache_hits_logic(self, condenser):
        """
        Estimación de cache hits basada en muestreo.
        """
        batch = [{"a": 1}, {"b": 2}, {"a": 3}]
        cache = {"a": "cached"}  # Solo campo 'a' está en cache

        hits = condenser._estimate_cache_hits(batch, cache)

        # 2 de 3 registros tienen 'a'
        # Estimación para 3 registros: int(2/3 * 3) = 2
        assert hits == 2

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_recovery_mode_splits_batch(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        sample_raw_records,
        sample_parse_cache,
    ):
        """
        Modo de recuperación debe dividir batch grande en sub-lotes.
        """
        # Mock processor que falla la primera vez pero pasa en sub-lotes
        # Esto es complejo de mockear, probaremos la lógica de _process_single_batch_with_recovery directamente

        # Preparar mocks para _rectify_signal
        with patch.object(condenser, "_rectify_signal") as mock_rectify:
            # Simular: llamada con batch completo falla, llamadas con sub-batch funcionan
            def side_effect(parsed_data):
                if len(parsed_data.raw_records) == 10:  # Batch completo
                    raise MemoryError("Out of memory")
                return pd.DataFrame([{"res": 1}] * len(parsed_data.raw_records))

            mock_rectify.side_effect = side_effect

            batch = sample_raw_records[:10]
            # consecutive_failures > 0 activa modo recuperación
            result = condenser._process_single_batch_with_recovery(
                batch, sample_parse_cache, consecutive_failures=1
            )

            assert result.success is True
            assert result.records_processed == 10
            # Debe haber llamado rectificación múltiples veces para los sub-lotes
            assert mock_rectify.call_count > 1

    def test_enhanced_stats_include_physics(self, condenser):
        """Las estadísticas deben incluir diagnóstico físico y de salud."""
        stats = ProcessingStats()
        metrics = {"saturation": 0.5, "entropy_ratio": 0.1}

        enhanced = condenser._enhance_stats_with_diagnostics(stats, metrics)

        assert "physics_diagnosis" in enhanced
        assert "system_health" in enhanced
        assert "current_metrics" in enhanced
        assert enhanced["efficiency"] == 0.0  # No processed records yet
