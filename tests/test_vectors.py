"""
test_vectors.py — Suite de Pruebas para el Transmisor MIC
==========================================================

Cobertura de Tests:
───────────────────────────────────────────────────────────────────────────
  1. FUNCIONES MATEMÁTICAS
     · _validate_finite: detección de NaN/Inf
     · _sigmoid: propiedades de función sigmoidal
     · _clamp: restricción a intervalos
  
  2. PHYSICSSTATE — Estado Físico
     · Invariantes I1-I4 (normalización, termodinámica, acotamiento, finitud)
     · Métricas derivadas: ECI, RSF
     · Casos límite y valores extremos
  
  3. TOPOLOGYSTATE — Estado Topológico
     · Números de Betti (β₀, β₁, β₂)
     · Característica de Euler-Poincaré
     · Coherencia topológica logarítmica
     · Complejidad topológica y densidad de defectos
  
  4. WISDOMSTATE — Veredicto Semántico
     · Validación de VerdictCode
     · Narrative no vacío
  
  5. VECTORESTADO — Espacio Producto Ω
     · Coherencia global física-topológica
     · Consistencia de veredicto
     · Serialización JSON determinista
  
  6. PROTOCOLO SERIAL
     · Context manager puerto pasivo
     · Detección de beacon
     · Envío JSON y ACK
     · Backoff con jitter
  
  7. PROPERTY-BASED TESTING
     · Invariantes universales con Hypothesis

Fundamentos Matemáticos Verificados:
───────────────────────────────────────────────────────────────────────────
  · Espacio de estados: Ω = Φ × Τ × Σ
  · Invariantes físicos: saturación ∈ [0,1], disipación ≥ 0, estabilidad ∈ [0,1]
  · Invariantes topológicos: βᵢ ∈ ℤ≥0, χ = β₀ - β₁ + β₂
  · Función sigmoidal: σ(x) = 1/(1 + e^(-k(x-c)))
  · ECI = sat × gyro × (diss/diss_ref)
  · RSF = √(sat² + (diss/diss_ref)²) / √2
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import MagicMock, Mock, patch, PropertyMock, call

import pytest

# Intentar importar hypothesis
try:
    from hypothesis import given, settings, strategies as st, assume, example
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="hypothesis not installed")(func)
        return decorator
    def example(*args, **kwargs):
        return lambda f: f
    settings = lambda **kwargs: lambda f: f
    class st:
        @staticmethod
        def floats(*args, **kwargs): return None
        @staticmethod
        def integers(*args, **kwargs): return None
        @staticmethod
        def text(*args, **kwargs): return None
        @staticmethod
        def sampled_from(elements): return None
    def composite(f): return f

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ══════════════════════════════════════════════════════════════════════════════

from test_vector import (
    # Constantes
    PUERTO,
    BAUDIOS,
    TIMEOUT_LECTURA,
    BEACON_KEYWORDS,
    TIMEOUT_BEACON,
    PAUSA_POST_BEACON,
    TIMEOUT_ACK,
    MAX_REINTENTOS,
    BACKOFF_BASE,
    JITTER_MAX,
    DISSIPATION_REFERENCE,
    SATURATION_HIGH_THRESHOLD,
    DISSIPATION_HIGH_THRESHOLD,
    STABILITY_MIN_REQUIRED,
    COHERENCE_SIGMOID_STEEPNESS,
    COHERENCE_SIGMOID_CENTER,
    BETA_1_MAX_REFERENCIA,
    EULER_CHAR_WARNING_THRESHOLD,
    TOPOLOGICAL_COMPLEXITY_CRITICAL,
    
    # Funciones matemáticas
    _validate_finite,
    _sigmoid,
    _clamp,
    
    # Enumeraciones
    VerdictCode,
    
    # Dataclasses
    PhysicsState,
    TopologyState,
    WisdomState,
    VectorEstado,
    
    # Funciones del protocolo
    puerto_serial_pasivo,
    _es_beacon,
    _esperar_beacon,
    _enviar_json,
    _esperar_ack,
    _construir_vector,
    _calcular_backoff_con_jitter,
    _ejecutar_ciclo_pasivo,
    enviar_vector_estado,
)


# ══════════════════════════════════════════════════════════════════════════════
# ESPECIFICACIÓN DEL DOMINIO (GROUND TRUTH)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhysicsSpec:
    """Especificación de un estado físico válido."""
    saturation: float
    dissipated_power: float
    gyroscopic_stability: float
    description: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Verifica si cumple todos los invariantes."""
        return (
            0.0 <= self.saturation <= 1.0 and
            self.dissipated_power >= 0.0 and
            0.0 <= self.gyroscopic_stability <= 1.0 and
            all(math.isfinite(v) for v in [
                self.saturation, self.dissipated_power, self.gyroscopic_stability
            ])
        )
    
    @property
    def expected_eci(self) -> float:
        """ECI esperado según la fórmula."""
        return (
            self.saturation *
            self.gyroscopic_stability *
            (self.dissipated_power / DISSIPATION_REFERENCE)
        )
    
    @property
    def expected_rsf(self) -> float:
        """RSF esperado según la fórmula."""
        norm_diss = min(1.0, self.dissipated_power / DISSIPATION_REFERENCE)
        raw = math.sqrt(self.saturation**2 + norm_diss**2)
        return raw / math.sqrt(2.0)


@dataclass(frozen=True)
class TopologySpec:
    """Especificación de un estado topológico válido."""
    beta_0: int
    beta_1: int
    beta_2: int
    pyramid_stability: float
    description: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Verifica si cumple todos los invariantes."""
        return (
            isinstance(self.beta_0, int) and self.beta_0 >= 1 and
            isinstance(self.beta_1, int) and self.beta_1 >= 0 and
            isinstance(self.beta_2, int) and self.beta_2 >= 0 and
            0.0 <= self.pyramid_stability <= 1.0 and
            math.isfinite(self.pyramid_stability)
        )
    
    @property
    def expected_euler(self) -> int:
        """χ = β₀ - β₁ + β₂"""
        return self.beta_0 - self.beta_1 + self.beta_2
    
    @property
    def expected_complexity(self) -> float:
        """C = [β₁/(1+β₁)] × (1 - pyramid_stability)"""
        betti_factor = self.beta_1 / (1.0 + self.beta_1)
        instability_factor = 1.0 - self.pyramid_stability
        return betti_factor * instability_factor
    
    @property
    def expected_defect_density(self) -> float:
        """ρ = β₁ / (β₀ × (1 + β₂))"""
        denominator = self.beta_0 * (1 + self.beta_2)
        if denominator == 0:
            return float("inf")
        return self.beta_1 / denominator


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES DE ESTADOS VÁLIDOS E INVÁLIDOS
# ══════════════════════════════════════════════════════════════════════════════

# Estados físicos válidos
VALID_PHYSICS_SPECS = [
    PhysicsSpec(0.0, 0.0, 0.0, "mínimos"),
    PhysicsSpec(1.0, 0.0, 1.0, "máximos normalizados"),
    PhysicsSpec(0.5, 50.0, 0.5, "medio"),
    PhysicsSpec(0.85, 65.0, 0.4, "típico de fiebre estructural"),
    PhysicsSpec(0.1, 1000.0, 0.9, "alta disipación"),
    PhysicsSpec(0.99, 99.9, 0.01, "límites cercanos"),
]

# Estados físicos inválidos
INVALID_PHYSICS_SPECS = [
    PhysicsSpec(-0.1, 0.0, 0.5, "saturación negativa"),
    PhysicsSpec(1.1, 0.0, 0.5, "saturación > 1"),
    PhysicsSpec(0.5, -1.0, 0.5, "disipación negativa"),
    PhysicsSpec(0.5, 0.0, -0.1, "estabilidad negativa"),
    PhysicsSpec(0.5, 0.0, 1.1, "estabilidad > 1"),
    PhysicsSpec(float('nan'), 0.0, 0.5, "saturación NaN"),
    PhysicsSpec(0.5, float('inf'), 0.5, "disipación infinita"),
    PhysicsSpec(0.5, 0.0, float('-inf'), "estabilidad -infinita"),
]

# Estados topológicos válidos
VALID_TOPOLOGY_SPECS = [
    TopologySpec(1, 0, 0, 1.0, "mínimo válido"),
    TopologySpec(1, 0, 0, 0.0, "estabilidad cero sin ciclos"),
    TopologySpec(5, 10, 2, 0.5, "complejo moderado"),
    TopologySpec(1, 442, 3, 0.69, "típico de prueba"),
    TopologySpec(100, 0, 50, 0.8, "muchas componentes y cavidades"),
]

# Estados topológicos inválidos
INVALID_TOPOLOGY_SPECS = [
    TopologySpec(0, 0, 0, 1.0, "β₀ = 0 inválido"),
    TopologySpec(-1, 0, 0, 1.0, "β₀ negativo"),
    TopologySpec(1, -1, 0, 1.0, "β₁ negativo"),
    TopologySpec(1, 0, -1, 1.0, "β₂ negativo"),
    TopologySpec(1, 0, 0, -0.1, "estabilidad negativa"),
    TopologySpec(1, 0, 0, 1.1, "estabilidad > 1"),
    TopologySpec(1, 0, 0, float('nan'), "estabilidad NaN"),
]


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def valid_physics_state() -> PhysicsState:
    """Estado físico válido estándar."""
    return PhysicsState(
        saturation=0.5,
        dissipated_power=50.0,
        gyroscopic_stability=0.8,
    )


@pytest.fixture
def valid_topology_state() -> TopologyState:
    """Estado topológico válido estándar."""
    return TopologyState(
        beta_0=1,
        beta_1=10,
        beta_2=0,
        pyramid_stability=0.9,
    )


@pytest.fixture
def valid_wisdom_state() -> WisdomState:
    """Estado de sabiduría válido estándar."""
    return WisdomState(
        verdict_code=VerdictCode.ADVERTENCIA,
        narrative="Sistema en estado de vigilancia",
    )


@pytest.fixture
def valid_vector_estado(
    valid_physics_state: PhysicsState,
    valid_topology_state: TopologyState,
    valid_wisdom_state: WisdomState,
) -> VectorEstado:
    """Vector de estado completo válido."""
    return VectorEstado(
        type="state_update",
        physics=valid_physics_state,
        topology=valid_topology_state,
        wisdom=valid_wisdom_state,
    )


@pytest.fixture
def mock_serial():
    """Mock del puerto serial para pruebas."""
    mock = MagicMock()
    mock.is_open = True
    mock.in_waiting = 0
    mock.readline.return_value = b""
    mock.write.return_value = 100
    return mock


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: FUNCIONES MATEMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateFinite:
    """Tests para _validate_finite."""

    def test_valid_zero(self) -> None:
        """Cero es finito."""
        _validate_finite(0.0, "test")  # No debe lanzar

    def test_valid_positive(self) -> None:
        """Números positivos finitos son válidos."""
        _validate_finite(1.0, "test")
        _validate_finite(1e10, "test")
        _validate_finite(1e-10, "test")

    def test_valid_negative(self) -> None:
        """Números negativos finitos son válidos."""
        _validate_finite(-1.0, "test")
        _validate_finite(-1e10, "test")

    def test_nan_raises(self) -> None:
        """NaN debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no es finito"):
            _validate_finite(float('nan'), "test_value")

    def test_positive_inf_raises(self) -> None:
        """Infinito positivo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no es finito"):
            _validate_finite(float('inf'), "test_value")

    def test_negative_inf_raises(self) -> None:
        """Infinito negativo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="no es finito"):
            _validate_finite(float('-inf'), "test_value")

    def test_error_message_includes_name(self) -> None:
        """El mensaje de error incluye el nombre del valor."""
        with pytest.raises(ValueError, match="my_variable"):
            _validate_finite(float('nan'), "my_variable")


class TestSigmoid:
    """Tests para _sigmoid."""

    def test_center_equals_half(self) -> None:
        """σ(c) = 0.5 para cualquier steepness."""
        for k in [0.1, 1.0, 10.0, 100.0]:
            for c in [-10.0, 0.0, 5.0, 100.0]:
                result = _sigmoid(c, steepness=k, center=c)
                assert result == pytest.approx(0.5, abs=1e-10)

    def test_monotonically_increasing(self) -> None:
        """σ es estrictamente creciente."""
        for k in [1.0, 10.0]:
            for c in [0.0, 5.0]:
                values = [_sigmoid(x, k, c) for x in range(-10, 11)]
                for i in range(len(values) - 1):
                    assert values[i] < values[i + 1]

    def test_asymptotes(self) -> None:
        """lim_{x→-∞} σ = 0 y lim_{x→+∞} σ = 1."""
        # Valores muy negativos → 0
        assert _sigmoid(-100, steepness=1.0, center=0.0) < 1e-10
        
        # Valores muy positivos → 1
        assert _sigmoid(100, steepness=1.0, center=0.0) > 1 - 1e-10

    def test_range_is_zero_to_one(self) -> None:
        """σ(x) ∈ (0, 1) para todo x finito."""
        for x in [-1000, -10, -1, 0, 1, 10, 1000]:
            result = _sigmoid(x, steepness=1.0, center=0.0)
            assert 0.0 < result < 1.0

    def test_steepness_effect(self) -> None:
        """Mayor steepness → transición más abrupta."""
        # Con steepness bajo, la transición es gradual
        low_k = _sigmoid(0.1, steepness=1.0, center=0.0)
        
        # Con steepness alto, x > c debería dar valor cercano a 1
        high_k = _sigmoid(0.1, steepness=100.0, center=0.0)
        
        assert high_k > low_k

    def test_overflow_protection_large_negative(self) -> None:
        """Protección contra overflow para exponentes muy negativos."""
        # Esto causaría exp(700+) sin protección
        result = _sigmoid(-1000, steepness=1.0, center=0.0)
        assert result == 0.0
        assert math.isfinite(result)

    def test_overflow_protection_large_positive(self) -> None:
        """Protección contra overflow para exponentes muy positivos."""
        result = _sigmoid(1000, steepness=1.0, center=0.0)
        assert result == 1.0
        assert math.isfinite(result)

    @pytest.mark.parametrize("x", [-100, -10, -1, 0, 1, 10, 100])
    def test_always_finite(self, x: float) -> None:
        """El resultado siempre es finito."""
        for k in [0.01, 1.0, 100.0]:
            for c in [-50, 0, 50]:
                result = _sigmoid(x, k, c)
                assert math.isfinite(result)


class TestClamp:
    """Tests para _clamp."""

    def test_value_in_range_unchanged(self) -> None:
        """Valor dentro del rango no cambia."""
        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_value_below_clamped_to_low(self) -> None:
        """Valor bajo el mínimo se clampea al mínimo."""
        assert _clamp(-5.0, 0.0, 10.0) == 0.0
        assert _clamp(-100.0, -10.0, 10.0) == -10.0

    def test_value_above_clamped_to_high(self) -> None:
        """Valor sobre el máximo se clampea al máximo."""
        assert _clamp(15.0, 0.0, 10.0) == 10.0
        assert _clamp(100.0, -10.0, 10.0) == 10.0

    def test_boundary_values(self) -> None:
        """Valores en los límites exactos no cambian."""
        assert _clamp(0.0, 0.0, 10.0) == 0.0
        assert _clamp(10.0, 0.0, 10.0) == 10.0

    def test_single_point_range(self) -> None:
        """Rango de un solo punto."""
        assert _clamp(-5.0, 5.0, 5.0) == 5.0
        assert _clamp(5.0, 5.0, 5.0) == 5.0
        assert _clamp(10.0, 5.0, 5.0) == 5.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PHYSICSSTATE
# ══════════════════════════════════════════════════════════════════════════════

class TestPhysicsStateValidation:
    """Tests de validación de PhysicsState."""

    @pytest.mark.parametrize(
        "spec",
        VALID_PHYSICS_SPECS,
        ids=lambda s: s.description,
    )
    def test_valid_states_accepted(self, spec: PhysicsSpec) -> None:
        """Estados válidos se construyen sin error."""
        state = PhysicsState(
            saturation=spec.saturation,
            dissipated_power=spec.dissipated_power,
            gyroscopic_stability=spec.gyroscopic_stability,
        )
        assert state.saturation == spec.saturation
        assert state.dissipated_power == spec.dissipated_power
        assert state.gyroscopic_stability == spec.gyroscopic_stability

    @pytest.mark.parametrize(
        "spec",
        INVALID_PHYSICS_SPECS,
        ids=lambda s: s.description,
    )
    def test_invalid_states_rejected(self, spec: PhysicsSpec) -> None:
        """Estados inválidos lanzan ValueError."""
        with pytest.raises(ValueError):
            PhysicsState(
                saturation=spec.saturation,
                dissipated_power=spec.dissipated_power,
                gyroscopic_stability=spec.gyroscopic_stability,
            )

    def test_saturation_below_zero_rejected(self) -> None:
        """Saturación < 0 es rechazada."""
        with pytest.raises(ValueError, match="saturation"):
            PhysicsState(saturation=-0.001, dissipated_power=0, gyroscopic_stability=0.5)

    def test_saturation_above_one_rejected(self) -> None:
        """Saturación > 1 es rechazada."""
        with pytest.raises(ValueError, match="saturation"):
            PhysicsState(saturation=1.001, dissipated_power=0, gyroscopic_stability=0.5)

    def test_negative_dissipation_rejected(self) -> None:
        """Disipación negativa viola Segunda Ley."""
        with pytest.raises(ValueError, match="Segunda Ley"):
            PhysicsState(saturation=0.5, dissipated_power=-1, gyroscopic_stability=0.5)

    def test_nan_in_any_field_rejected(self) -> None:
        """NaN en cualquier campo es rechazado."""
        for field_name in ["saturation", "dissipated_power", "gyroscopic_stability"]:
            kwargs = {
                "saturation": 0.5,
                "dissipated_power": 50.0,
                "gyroscopic_stability": 0.5,
            }
            kwargs[field_name] = float('nan')
            
            with pytest.raises(ValueError, match="no es finito"):
                PhysicsState(**kwargs)


class TestPhysicsStateDerivedMetrics:
    """Tests de métricas derivadas de PhysicsState."""

    @pytest.mark.parametrize(
        "spec",
        VALID_PHYSICS_SPECS,
        ids=lambda s: s.description,
    )
    def test_energy_consistency_index_formula(self, spec: PhysicsSpec) -> None:
        """ECI = sat × gyro × (diss / diss_ref)"""
        state = PhysicsState(
            saturation=spec.saturation,
            dissipated_power=spec.dissipated_power,
            gyroscopic_stability=spec.gyroscopic_stability,
        )
        
        expected = spec.expected_eci
        assert state.energy_consistency_index == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize(
        "spec",
        VALID_PHYSICS_SPECS,
        ids=lambda s: s.description,
    )
    def test_regime_stress_factor_formula(self, spec: PhysicsSpec) -> None:
        """RSF = √(sat² + (diss/diss_ref)²) / √2"""
        state = PhysicsState(
            saturation=spec.saturation,
            dissipated_power=spec.dissipated_power,
            gyroscopic_stability=spec.gyroscopic_stability,
        )
        
        expected = spec.expected_rsf
        assert state.regime_stress_factor == pytest.approx(expected, rel=1e-10)

    def test_eci_range(self) -> None:
        """ECI puede ser cualquier valor ≥ 0."""
        # ECI mínimo
        state_min = PhysicsState(saturation=0, dissipated_power=0, gyroscopic_stability=0)
        assert state_min.energy_consistency_index == 0.0
        
        # ECI puede ser > 1 con alta disipación
        state_high = PhysicsState(saturation=1, dissipated_power=200, gyroscopic_stability=1)
        assert state_high.energy_consistency_index > 1.0

    def test_rsf_range(self) -> None:
        """RSF ∈ [0, 1] (posiblemente > 1 con disipación extrema)."""
        # RSF mínimo
        state_min = PhysicsState(saturation=0, dissipated_power=0, gyroscopic_stability=0.5)
        assert state_min.regime_stress_factor == 0.0
        
        # RSF con valores máximos normalizados
        state_max = PhysicsState(saturation=1, dissipated_power=100, gyroscopic_stability=0.5)
        assert state_max.regime_stress_factor == pytest.approx(1.0, rel=1e-6)

    def test_rsf_clamped_dissipation(self) -> None:
        """RSF normaliza disipación con min(1, diss/ref)."""
        # Alta disipación se clampea a 1 en la fórmula
        state = PhysicsState(saturation=0, dissipated_power=1000, gyroscopic_stability=0.5)
        # RSF = √(0² + 1²) / √2 = 1/√2 ≈ 0.707
        expected = 1.0 / math.sqrt(2.0)
        assert state.regime_stress_factor == pytest.approx(expected, rel=1e-6)


class TestPhysicsStateImmutability:
    """Tests de inmutabilidad de PhysicsState."""

    def test_frozen_dataclass(self, valid_physics_state: PhysicsState) -> None:
        """PhysicsState es inmutable (frozen)."""
        with pytest.raises(AttributeError):
            valid_physics_state.saturation = 0.9


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: TOPOLOGYSTATE
# ══════════════════════════════════════════════════════════════════════════════

class TestTopologyStateValidation:
    """Tests de validación de TopologyState."""

    @pytest.mark.parametrize(
        "spec",
        VALID_TOPOLOGY_SPECS,
        ids=lambda s: s.description,
    )
    def test_valid_states_accepted(self, spec: TopologySpec) -> None:
        """Estados topológicos válidos se construyen sin error."""
        state = TopologyState(
            beta_0=spec.beta_0,
            beta_1=spec.beta_1,
            beta_2=spec.beta_2,
            pyramid_stability=spec.pyramid_stability,
        )
        assert state.beta_0 == spec.beta_0
        assert state.euler_characteristic == spec.expected_euler

    @pytest.mark.parametrize(
        "spec",
        INVALID_TOPOLOGY_SPECS,
        ids=lambda s: s.description,
    )
    def test_invalid_states_rejected(self, spec: TopologySpec) -> None:
        """Estados inválidos lanzan ValueError."""
        with pytest.raises(ValueError):
            TopologyState(
                beta_0=spec.beta_0,
                beta_1=spec.beta_1,
                beta_2=spec.beta_2,
                pyramid_stability=spec.pyramid_stability,
            )

    def test_beta_0_must_be_at_least_one(self) -> None:
        """β₀ ≥ 1 para estructuras no vacías."""
        with pytest.raises(ValueError, match="beta_0"):
            TopologyState(beta_0=0, beta_1=0, beta_2=0, pyramid_stability=1.0)

    def test_negative_betti_numbers_rejected(self) -> None:
        """Números de Betti negativos son rechazados."""
        with pytest.raises(ValueError, match="beta"):
            TopologyState(beta_0=1, beta_1=-1, beta_2=0, pyramid_stability=1.0)

    def test_non_integer_betti_rejected(self) -> None:
        """Números de Betti deben ser enteros."""
        with pytest.raises(ValueError, match="entero"):
            TopologyState(beta_0=1.5, beta_1=0, beta_2=0, pyramid_stability=1.0)


class TestTopologyStateEulerCharacteristic:
    """Tests de la característica de Euler-Poincaré."""

    @pytest.mark.parametrize(
        "b0, b1, b2, expected_chi",
        [
            (1, 0, 0, 1),   # Punto: χ = 1
            (1, 1, 0, 0),   # Círculo: χ = 0
            (1, 2, 1, 0),   # Toro: χ = 0
            (1, 0, 1, 2),   # Esfera: χ = 2
            (2, 0, 0, 2),   # Dos puntos: χ = 2
            (1, 10, 0, -9), # Muchos ciclos: χ < 0
        ],
        ids=["punto", "circulo", "toro", "esfera", "dos_puntos", "muchos_ciclos"],
    )
    def test_euler_formula(
        self, b0: int, b1: int, b2: int, expected_chi: int
    ) -> None:
        """χ = β₀ - β₁ + β₂"""
        state = TopologyState(
            beta_0=b0,
            beta_1=b1,
            beta_2=b2,
            pyramid_stability=1.0,
        )
        assert state.euler_characteristic == expected_chi

    def test_euler_is_integer(self, valid_topology_state: TopologyState) -> None:
        """χ siempre es entero."""
        assert isinstance(valid_topology_state.euler_characteristic, int)


class TestTopologyStateComplexity:
    """Tests de complejidad topológica."""

    def test_complexity_zero_when_no_cycles(self) -> None:
        """C = 0 cuando β₁ = 0."""
        state = TopologyState(beta_0=1, beta_1=0, beta_2=0, pyramid_stability=0.5)
        assert state.topological_complexity == 0.0

    def test_complexity_zero_when_fully_stable(self) -> None:
        """C = 0 cuando pyramid_stability = 1."""
        state = TopologyState(beta_0=1, beta_1=100, beta_2=0, pyramid_stability=1.0)
        assert state.topological_complexity == 0.0

    def test_complexity_increases_with_cycles(self) -> None:
        """C aumenta con β₁ (a estabilidad fija)."""
        stability = 0.5
        
        complexities = []
        for b1 in [0, 10, 100, 1000]:
            state = TopologyState(beta_0=1, beta_1=b1, beta_2=0, pyramid_stability=stability)
            complexities.append(state.topological_complexity)
        
        # Debe ser monótonamente creciente
        for i in range(len(complexities) - 1):
            assert complexities[i] < complexities[i + 1]

    def test_complexity_decreases_with_stability(self) -> None:
        """C disminuye con pyramid_stability (a β₁ fijo)."""
        beta_1 = 50
        
        complexities = []
        for stability in [0.0, 0.25, 0.5, 0.75, 1.0]:
            state = TopologyState(beta_0=1, beta_1=beta_1, beta_2=0, pyramid_stability=stability)
            complexities.append(state.topological_complexity)
        
        # Debe ser monótonamente decreciente
        for i in range(len(complexities) - 1):
            assert complexities[i] > complexities[i + 1]

    def test_complexity_bounded(self) -> None:
        """C ∈ [0, 1)."""
        for b1 in [0, 1, 10, 100, 10000]:
            for stability in [0.0, 0.5, 1.0]:
                state = TopologyState(beta_0=1, beta_1=b1, beta_2=0, pyramid_stability=stability)
                assert 0.0 <= state.topological_complexity < 1.0


class TestTopologyStateDefectDensity:
    """Tests de densidad de defectos homológicos."""

    def test_defect_density_formula(self) -> None:
        """ρ = β₁ / (β₀ × (1 + β₂))"""
        state = TopologyState(beta_0=2, beta_1=10, beta_2=4, pyramid_stability=0.5)
        # ρ = 10 / (2 × 5) = 1.0
        expected = 10 / (2 * (1 + 4))
        assert state.homological_defect_density == pytest.approx(expected)

    def test_defect_density_zero_when_no_cycles(self) -> None:
        """ρ = 0 cuando β₁ = 0."""
        state = TopologyState(beta_0=5, beta_1=0, beta_2=10, pyramid_stability=0.5)
        assert state.homological_defect_density == 0.0

    def test_defect_density_increases_with_cycles(self) -> None:
        """ρ aumenta con β₁."""
        densities = []
        for b1 in [0, 10, 50, 100]:
            state = TopologyState(beta_0=1, beta_1=b1, beta_2=0, pyramid_stability=0.5)
            densities.append(state.homological_defect_density)
        
        for i in range(len(densities) - 1):
            assert densities[i] < densities[i + 1]


class TestTopologyStateCoherence:
    """Tests de coherencia topológica."""

    def test_high_beta1_requires_low_stability(self) -> None:
        """Con β₁ alto, se permite estabilidad baja."""
        # Con β₁ = 999 (cerca de referencia 1000), stability ≈ 0 permitido
        state = TopologyState(beta_0=1, beta_1=999, beta_2=0, pyramid_stability=0.01)
        # Debe construirse sin error
        assert state.beta_1 == 999

    def test_incoherence_detected(self) -> None:
        """Incoherencia: β₁ bajo con estabilidad muy baja."""
        # Con β₁ = 1, se requiere estabilidad cercana a 1
        # log(2) / log(1001) ≈ 0.1, lower_bound ≈ 0.9
        with pytest.raises(ValueError, match="Incoherencia topológica"):
            TopologyState(beta_0=1, beta_1=1, beta_2=0, pyramid_stability=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: WISDOMSTATE
# ══════════════════════════════════════════════════════════════════════════════

class TestWisdomStateValidation:
    """Tests de validación de WisdomState."""

    @pytest.mark.parametrize(
        "verdict",
        list(VerdictCode),
        ids=lambda v: v.name,
    )
    def test_all_verdict_codes_accepted(self, verdict: VerdictCode) -> None:
        """Todos los VerdictCode son válidos."""
        state = WisdomState(
            verdict_code=verdict,
            narrative="Test narrative",
        )
        assert state.verdict_code == verdict

    def test_empty_narrative_rejected(self) -> None:
        """Narrativa vacía es rechazada."""
        with pytest.raises(ValueError, match="narrative"):
            WisdomState(verdict_code=VerdictCode.OPTIMO, narrative="")

    def test_whitespace_narrative_rejected(self) -> None:
        """Narrativa solo espacios es rechazada."""
        with pytest.raises(ValueError, match="narrative"):
            WisdomState(verdict_code=VerdictCode.OPTIMO, narrative="   ")

    def test_invalid_verdict_type_rejected(self) -> None:
        """Tipo de veredicto inválido es rechazado."""
        with pytest.raises(ValueError, match="VerdictCode"):
            WisdomState(verdict_code=99, narrative="Test")

    def test_string_verdict_rejected(self) -> None:
        """String en lugar de VerdictCode es rechazado."""
        with pytest.raises(ValueError):
            WisdomState(verdict_code="OPTIMO", narrative="Test")


class TestVerdictCodeOrdering:
    """Tests de ordenamiento de VerdictCode."""

    def test_ordering_total(self) -> None:
        """VerdictCode tiene orden total."""
        assert VerdictCode.OPTIMO < VerdictCode.ADVERTENCIA
        assert VerdictCode.ADVERTENCIA < VerdictCode.FIEBRE_ESTRUCTURAL
        assert VerdictCode.FIEBRE_ESTRUCTURAL < VerdictCode.COLAPSO_INMINENTE

    def test_values_are_sequential(self) -> None:
        """Los valores son secuenciales 0, 1, 2, 3."""
        assert VerdictCode.OPTIMO.value == 0
        assert VerdictCode.ADVERTENCIA.value == 1
        assert VerdictCode.FIEBRE_ESTRUCTURAL.value == 2
        assert VerdictCode.COLAPSO_INMINENTE.value == 3


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: VECTORESTADO
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorEstadoCreation:
    """Tests de creación de VectorEstado."""

    def test_valid_vector_created(self, valid_vector_estado: VectorEstado) -> None:
        """Vector válido se crea correctamente."""
        assert valid_vector_estado.type == "state_update"
        assert isinstance(valid_vector_estado.physics, PhysicsState)
        assert isinstance(valid_vector_estado.topology, TopologyState)
        assert isinstance(valid_vector_estado.wisdom, WisdomState)

    def test_validate_integrity_passes_for_valid(
        self, valid_vector_estado: VectorEstado
    ) -> None:
        """validate_integrity no lanza para vector válido."""
        valid_vector_estado.validate_integrity()  # No debe lanzar


class TestVectorEstadoPhysicsTopologyCoherence:
    """Tests de coherencia física-topológica en VectorEstado."""

    def test_high_stress_requires_high_stability(self) -> None:
        """Estrés alto requiere estabilidad combinada alta."""
        # Configuración de alto estrés
        physics = PhysicsState(
            saturation=0.95,  # Alto
            dissipated_power=150.0,  # Alto
            gyroscopic_stability=0.3,  # Bajo
        )
        
        topology = TopologyState(
            beta_0=1,
            beta_1=10,
            beta_2=0,
            pyramid_stability=0.2,  # Bajo
        )
        
        wisdom = WisdomState(
            verdict_code=VerdictCode.COLAPSO_INMINENTE,
            narrative="Sistema en colapso",
        )
        
        # Estabilidad combinada = 0.3 + 0.2 = 0.5 < requerida
        with pytest.raises(ValueError, match="Incoherencia física-topológica"):
            vector = VectorEstado(
                type="test",
                physics=physics,
                topology=topology,
                wisdom=wisdom,
            )
            vector.validate_integrity()

    def test_low_stress_allows_low_stability(self) -> None:
        """Estrés bajo permite estabilidad combinada baja."""
        physics = PhysicsState(
            saturation=0.1,  # Bajo
            dissipated_power=5.0,  # Bajo
            gyroscopic_stability=0.2,  # Bajo
        )
        
        topology = TopologyState(
            beta_0=1,
            beta_1=0,
            beta_2=0,
            pyramid_stability=0.1,  # Bajo
        )
        
        wisdom = WisdomState(
            verdict_code=VerdictCode.OPTIMO,
            narrative="Sistema óptimo",
        )
        
        # Debe aceptarse porque el estrés es bajo
        vector = VectorEstado(
            type="test",
            physics=physics,
            topology=topology,
            wisdom=wisdom,
        )
        vector.validate_integrity()  # No debe lanzar


class TestVectorEstadoSerialization:
    """Tests de serialización de VectorEstado."""

    def test_to_dict_structure(self, valid_vector_estado: VectorEstado) -> None:
        """to_dict produce estructura correcta."""
        d = valid_vector_estado.to_dict()
        
        assert "type" in d
        assert "physics" in d
        assert "topology" in d
        assert "wisdom" in d

    def test_to_dict_includes_derived_metrics(
        self, valid_vector_estado: VectorEstado
    ) -> None:
        """to_dict incluye métricas derivadas."""
        d = valid_vector_estado.to_dict()
        
        # Métricas físicas derivadas
        assert "energy_consistency_index" in d["physics"]
        assert "regime_stress_factor" in d["physics"]
        
        # Métricas topológicas derivadas
        assert "euler_characteristic" in d["topology"]
        assert "topological_complexity" in d["topology"]
        assert "homological_defect_density" in d["topology"]

    def test_to_json_is_valid_json(self, valid_vector_estado: VectorEstado) -> None:
        """to_json produce JSON válido."""
        json_str = valid_vector_estado.to_json()
        
        # Debe ser parseable
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert parsed["type"] == valid_vector_estado.type

    def test_to_json_is_deterministic(self, valid_vector_estado: VectorEstado) -> None:
        """to_json produce el mismo resultado cada vez (sort_keys=True)."""
        json1 = valid_vector_estado.to_json()
        json2 = valid_vector_estado.to_json()
        json3 = valid_vector_estado.to_json()
        
        assert json1 == json2 == json3

    def test_verdict_code_serialized_as_int(
        self, valid_vector_estado: VectorEstado
    ) -> None:
        """verdict_code se serializa como int."""
        d = valid_vector_estado.to_dict()
        
        assert isinstance(d["wisdom"]["verdict_code"], int)
        assert "verdict_name" in d["wisdom"]

    def test_summary_is_string(self, valid_vector_estado: VectorEstado) -> None:
        """summary produce string legible."""
        summary = valid_vector_estado.summary
        
        assert isinstance(summary, str)
        assert "type=" in summary
        assert "sat=" in summary
        assert "verdict=" in summary


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: DETECCIÓN DE BEACON
# ══════════════════════════════════════════════════════════════════════════════

class TestBeaconDetection:
    """Tests para _es_beacon."""

    @pytest.mark.parametrize(
        "line",
        [
            "SENTINEL ready",
            "System SENTINEL initialized",
            "READY to receive",
            "sentinel",  # Case insensitive
            "ready",     # Case insensitive
            "[INFO] SENTINEL marker detected",
        ],
    )
    def test_beacon_detected(self, line: str) -> None:
        """Líneas con keywords son detectadas."""
        assert _es_beacon(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            "System booting...",
            "Initializing modules",
            "",
            "Ready? Not quite",  # Parcial no cuenta
            "SENTIN",  # Incompleto
        ],
    )
    def test_non_beacon_rejected(self, line: str) -> None:
        """Líneas sin keywords no son beacon."""
        assert _es_beacon(line) is False

    def test_empty_line_not_beacon(self) -> None:
        """Línea vacía no es beacon."""
        assert _es_beacon("") is False
        assert _es_beacon("   ") is False


class TestEsperarBeacon:
    """Tests para _esperar_beacon con mock serial."""

    def test_beacon_found_returns_true(self, mock_serial: MagicMock) -> None:
        """Retorna True cuando se encuentra beacon."""
        mock_serial.readline.side_effect = [
            b"Booting...\n",
            b"SENTINEL ready\n",
        ]
        
        with patch("time.monotonic") as mock_time:
            mock_time.side_effect = [0.0, 0.1, 0.2]
            
            result = _esperar_beacon(mock_serial)
        
        assert result is True

    def test_timeout_returns_false(self, mock_serial: MagicMock) -> None:
        """Retorna False cuando timeout sin beacon."""
        mock_serial.readline.return_value = b"No beacon here\n"
        
        with patch("time.monotonic") as mock_time:
            # Simular timeout
            mock_time.side_effect = [
                0.0,    # Start
                30.0,   # Durante
                61.0,   # Después de timeout
            ]
            
            result = _esperar_beacon(mock_serial)
        
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: BACKOFF CON JITTER
# ══════════════════════════════════════════════════════════════════════════════

class TestBackoffConJitter:
    """Tests para _calcular_backoff_con_jitter."""

    def test_base_case_intento_1(self) -> None:
        """Intento 1: base_delay = BACKOFF_BASE^1."""
        with patch("random.uniform", return_value=0.0):  # Sin jitter
            result = _calcular_backoff_con_jitter(1)
        
        assert result == pytest.approx(BACKOFF_BASE ** 1)

    def test_exponential_growth(self) -> None:
        """El backoff crece exponencialmente."""
        with patch("random.uniform", return_value=0.0):
            delays = [_calcular_backoff_con_jitter(i) for i in range(1, 5)]
        
        for i in range(len(delays) - 1):
            ratio = delays[i + 1] / delays[i]
            assert ratio == pytest.approx(BACKOFF_BASE, rel=1e-6)

    def test_jitter_range(self) -> None:
        """El jitter está en el rango [-JITTER_MAX, +JITTER_MAX]."""
        results = []
        for _ in range(1000):
            result = _calcular_backoff_con_jitter(1)
            results.append(result)
        
        base = BACKOFF_BASE ** 1
        min_expected = base * (1 - JITTER_MAX)
        max_expected = base * (1 + JITTER_MAX)
        
        assert min(results) >= min_expected * 0.99  # Tolerancia
        assert max(results) <= max_expected * 1.01

    def test_jitter_adds_randomness(self) -> None:
        """El jitter produce variación en los resultados."""
        results = set()
        for _ in range(100):
            result = _calcular_backoff_con_jitter(1)
            results.add(round(result, 4))
        
        # Debe haber variación (más de un valor único)
        assert len(results) > 1

    def test_always_positive(self) -> None:
        """El resultado siempre es positivo."""
        for intento in range(1, 10):
            for _ in range(100):
                result = _calcular_backoff_con_jitter(intento)
                assert result > 0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CONSTRUIR VECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TestConstruirVector:
    """Tests para _construir_vector."""

    def test_returns_valid_vector(self) -> None:
        """Retorna un VectorEstado válido."""
        vector = _construir_vector()
        
        assert isinstance(vector, VectorEstado)
        assert vector.type == "state_update"

    def test_vector_passes_integrity_check(self) -> None:
        """El vector pasa validación de integridad."""
        vector = _construir_vector()
        
        # No debe lanzar
        vector.validate_integrity()

    def test_vector_is_serializable(self) -> None:
        """El vector es serializable a JSON."""
        vector = _construir_vector()
        json_str = vector.to_json()
        
        # Debe ser parseable
        parsed = json.loads(json_str)
        assert "type" in parsed


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PROTOCOLO SERIAL COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class TestPuertoSerialPasivo:
    """Tests para puerto_serial_pasivo context manager."""

    def test_opens_with_correct_settings(self) -> None:
        """Abre puerto con configuración pasiva correcta."""
        with patch("serial.Serial") as MockSerial:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            MockSerial.return_value = mock_instance
            
            with puerto_serial_pasivo(PUERTO, BAUDIOS, TIMEOUT_LECTURA) as ser:
                pass
            
            MockSerial.assert_called_once_with(
                PUERTO,
                BAUDIOS,
                timeout=TIMEOUT_LECTURA,
                dsrdtr=False,
                rtscts=False,
            )

    def test_closes_on_exit(self) -> None:
        """Cierra puerto al salir del contexto."""
        with patch("serial.Serial") as MockSerial:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            MockSerial.return_value = mock_instance
            
            with puerto_serial_pasivo(PUERTO, BAUDIOS, TIMEOUT_LECTURA):
                pass
            
            mock_instance.close.assert_called_once()

    def test_closes_on_exception(self) -> None:
        """Cierra puerto incluso si hay excepción."""
        with patch("serial.Serial") as MockSerial:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            MockSerial.return_value = mock_instance
            
            try:
                with puerto_serial_pasivo(PUERTO, BAUDIOS, TIMEOUT_LECTURA):
                    raise RuntimeError("Test error")
            except RuntimeError:
                pass
            
            mock_instance.close.assert_called_once()


class TestEnviarJson:
    """Tests para _enviar_json."""

    def test_sends_json_with_newline(self, mock_serial: MagicMock) -> None:
        """Envía JSON terminado con newline."""
        vector = _construir_vector()
        
        _enviar_json(mock_serial, vector)
        
        # Verificar que se llamó write
        assert mock_serial.write.called
        
        # Verificar que termina con newline
        written = mock_serial.write.call_args[0][0]
        assert written.endswith(b"\n")

    def test_resets_input_buffer(self, mock_serial: MagicMock) -> None:
        """Limpia buffer de entrada antes de enviar."""
        vector = _construir_vector()
        
        _enviar_json(mock_serial, vector)
        
        mock_serial.reset_input_buffer.assert_called_once()

    def test_flushes_after_write(self, mock_serial: MagicMock) -> None:
        """Hace flush después de escribir."""
        vector = _construir_vector()
        
        _enviar_json(mock_serial, vector)
        
        mock_serial.flush.assert_called()


class TestEsperarAck:
    """Tests para _esperar_ack."""

    def test_ack_found_returns_true(self, mock_serial: MagicMock) -> None:
        """Retorna True cuando se encuentra ACK."""
        mock_serial.in_waiting = 10
        mock_serial.readline.return_value = b"ACK received\n"
        
        with patch("time.monotonic") as mock_time:
            mock_time.side_effect = [0.0, 0.1, 0.2]
            
            result = _esperar_ack(mock_serial)
        
        assert result is True

    def test_no_ack_returns_false(self, mock_serial: MagicMock) -> None:
        """Retorna False cuando no hay ACK."""
        mock_serial.in_waiting = 10
        mock_serial.readline.return_value = b"Some other response\n"
        
        with patch("time.monotonic") as mock_time:
            mock_time.side_effect = [0.0, 2.0, 6.0]  # Timeout
            
            result = _esperar_ack(mock_serial)
        
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: PROPERTY-BASED
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBased:
    """Tests basados en propiedades con Hypothesis."""

    @given(
        sat=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        diss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        gyro=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_valid_physics_always_accepted(
        self, sat: float, diss: float, gyro: float
    ) -> None:
        """Cualquier combinación válida de parámetros es aceptada."""
        state = PhysicsState(
            saturation=sat,
            dissipated_power=diss,
            gyroscopic_stability=gyro,
        )
        assert state.saturation == sat

    @given(
        sat=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        diss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        gyro=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_eci_always_non_negative(
        self, sat: float, diss: float, gyro: float
    ) -> None:
        """ECI ≥ 0 siempre."""
        state = PhysicsState(
            saturation=sat,
            dissipated_power=diss,
            gyroscopic_stability=gyro,
        )
        assert state.energy_consistency_index >= 0

    @given(
        sat=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        diss=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        gyro=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_rsf_in_valid_range(
        self, sat: float, diss: float, gyro: float
    ) -> None:
        """RSF ∈ [0, ~1.5] aproximadamente."""
        state = PhysicsState(
            saturation=sat,
            dissipated_power=diss,
            gyroscopic_stability=gyro,
        )
        # RSF puede ser > 1 con disipación extrema, pero acotado
        assert 0.0 <= state.regime_stress_factor <= 2.0

    @given(
        b0=st.integers(min_value=1, max_value=100),
        b1=st.integers(min_value=0, max_value=1000),
        b2=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=100)
    def test_euler_characteristic_formula(
        self, b0: int, b1: int, b2: int
    ) -> None:
        """χ = β₀ - β₁ + β₂ siempre."""
        # Usar estabilidad alta para evitar coherencia check
        state = TopologyState(
            beta_0=b0,
            beta_1=b1,
            beta_2=b2,
            pyramid_stability=1.0,
        )
        
        expected_chi = b0 - b1 + b2
        assert state.euler_characteristic == expected_chi

    @given(x=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_sigmoid_range(self, x: float) -> None:
        """σ(x) ∈ (0, 1) para todo x finito."""
        result = _sigmoid(x, steepness=1.0, center=0.0)
        assert 0.0 < result < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: THREAD SAFETY
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Tests de seguridad en hilos."""

    def test_concurrent_physics_state_creation(self) -> None:
        """Crear PhysicsState concurrentemente es seguro."""
        results: List[PhysicsState] = []
        errors: List[Exception] = []
        lock = threading.Lock()
        
        def create_state(i: int):
            try:
                state = PhysicsState(
                    saturation=i / 100.0,
                    dissipated_power=float(i),
                    gyroscopic_stability=0.5,
                )
                with lock:
                    results.append(state)
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = [threading.Thread(target=create_state, args=(i,)) for i in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors
        assert len(results) == 100

    def test_concurrent_vector_serialization(self) -> None:
        """Serializar VectorEstado concurrentemente es seguro."""
        vector = _construir_vector()
        results: List[str] = []
        errors: List[Exception] = []
        lock = threading.Lock()
        
        def serialize():
            try:
                json_str = vector.to_json()
                with lock:
                    results.append(json_str)
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = [threading.Thread(target=serialize) for _ in range(50)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors
        assert len(results) == 50
        # Todos deben ser iguales (determinístico)
        assert len(set(results)) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: RENDIMIENTO
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Tests de rendimiento básico."""

    def test_physics_state_creation_is_fast(self) -> None:
        """Crear PhysicsState es rápido."""
        import time
        
        start = time.perf_counter()
        
        for _ in range(10000):
            PhysicsState(saturation=0.5, dissipated_power=50.0, gyroscopic_stability=0.8)
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"10000 creaciones tomaron {elapsed:.3f}s"

    def test_vector_serialization_is_fast(self) -> None:
        """Serializar VectorEstado es rápido."""
        import time
        
        vector = _construir_vector()
        
        start = time.perf_counter()
        
        for _ in range(1000):
            _ = vector.to_json()
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.5, f"1000 serializaciones tomaron {elapsed:.3f}s"

    def test_sigmoid_is_fast(self) -> None:
        """Calcular sigmoid es rápido."""
        import time
        
        start = time.perf_counter()
        
        for x in range(-1000, 1001):
            _sigmoid(x / 10.0, steepness=10.0, center=0.85)
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"2001 cálculos de sigmoid tomaron {elapsed:.3f}s"


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Tests de constantes del módulo."""

    def test_dissipation_reference_positive(self) -> None:
        """DISSIPATION_REFERENCE > 0."""
        assert DISSIPATION_REFERENCE > 0

    def test_timeout_values_positive(self) -> None:
        """Todos los timeouts son positivos."""
        assert TIMEOUT_LECTURA > 0
        assert TIMEOUT_BEACON > 0
        assert TIMEOUT_ACK > 0

    def test_max_reintentos_reasonable(self) -> None:
        """MAX_REINTENTOS está en rango razonable."""
        assert 1 <= MAX_REINTENTOS <= 10

    def test_backoff_base_greater_than_one(self) -> None:
        """BACKOFF_BASE > 1 para crecimiento exponencial."""
        assert BACKOFF_BASE > 1.0

    def test_jitter_max_is_fraction(self) -> None:
        """JITTER_MAX ∈ (0, 1) es fracción razonable."""
        assert 0.0 < JITTER_MAX < 1.0

    def test_beacon_keywords_not_empty(self) -> None:
        """BEACON_KEYWORDS contiene al menos un keyword."""
        assert len(BEACON_KEYWORDS) > 0
        assert all(isinstance(kw, str) for kw in BEACON_KEYWORDS)

    def test_beta_1_max_referencia_positive(self) -> None:
        """BETA_1_MAX_REFERENCIA > 0."""
        assert BETA_1_MAX_REFERENCIA > 0


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
        "--durations=10",
    ])