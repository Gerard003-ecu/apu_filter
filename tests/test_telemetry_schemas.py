"""
Suite de pruebas para el módulo de Telemetry Schemas.

Valida:
1. Inmutabilidad de los esquemas (frozen dataclasses).
2. Valores por defecto correctos.
3. Tipado correcto de los campos.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, is_dataclass
import pytest

from app.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
)


class TestPhysicsMetrics:
    """Pruebas para PhysicsMetrics."""

    def test_physics_metrics_defaults(self):
        """Valida los valores por defecto."""
        metrics = PhysicsMetrics()
        assert metrics.saturation == 0.0
        assert metrics.pressure == 0.0
        assert metrics.kinetic_energy == 0.0
        assert metrics.potential_energy == 0.0
        assert metrics.flyback_voltage == 0.0
        assert metrics.dissipated_power == 0.0
        assert metrics.gyroscopic_stability == 1.0
        assert metrics.poynting_flux == 0.0

    def test_physics_metrics_immutability(self):
        """Valida que la clase sea inmutable."""
        metrics = PhysicsMetrics()
        with pytest.raises(FrozenInstanceError):
            metrics.saturation = 0.5

    def test_physics_metrics_initialization(self):
        """Valida la inicialización con valores personalizados."""
        metrics = PhysicsMetrics(
            saturation=0.8,
            flyback_voltage=1.2
        )
        assert metrics.saturation == 0.8
        assert metrics.flyback_voltage == 1.2
        assert metrics.gyroscopic_stability == 1.0  # Default


class TestTopologicalMetrics:
    """Pruebas para TopologicalMetrics."""

    def test_topological_metrics_defaults(self):
        """Valida los valores por defecto."""
        metrics = TopologicalMetrics()
        assert metrics.beta_0 == 1
        assert metrics.beta_1 == 0
        assert metrics.beta_2 == 0
        assert metrics.euler_characteristic == 1
        assert metrics.fiedler_value == 1.0
        assert metrics.spectral_gap == 0.0
        assert metrics.pyramid_stability == 1.0
        assert metrics.structural_entropy == 0.0

    def test_topological_metrics_immutability(self):
        """Valida que la clase sea inmutable."""
        metrics = TopologicalMetrics()
        with pytest.raises(FrozenInstanceError):
            metrics.beta_0 = 2

    def test_topological_metrics_initialization(self):
        """Valida la inicialización con valores personalizados."""
        metrics = TopologicalMetrics(
            beta_0=5,
            beta_1=2
        )
        assert metrics.beta_0 == 5
        assert metrics.beta_1 == 2
        assert metrics.beta_2 == 0  # Default


class TestControlMetrics:
    """Pruebas para ControlMetrics."""

    def test_control_metrics_defaults(self):
        """Valida los valores por defecto."""
        metrics = ControlMetrics()
        assert metrics.poles_real == []
        assert metrics.is_stable is True
        assert metrics.phase_margin_deg == 45.0
        assert metrics.gain_margin_db == float('inf')
        assert metrics.damping_ratio == 0.707
        assert metrics.natural_frequency == 0.0
        assert metrics.lyapunov_exponent == -1.0

    def test_control_metrics_immutability(self):
        """Valida que la clase sea inmutable."""
        metrics = ControlMetrics()
        with pytest.raises(FrozenInstanceError):
            metrics.is_stable = False

        # La lista poles_real es mutable si se accede, pero el campo no se puede reasignar
        with pytest.raises(FrozenInstanceError):
            metrics.poles_real = [1.0]

    def test_control_metrics_initialization(self):
        """Valida la inicialización con valores personalizados."""
        metrics = ControlMetrics(
            poles_real=[-1.0, -2.0],
            is_stable=False
        )
        assert metrics.poles_real == [-1.0, -2.0]
        assert metrics.is_stable is False
        assert metrics.phase_margin_deg == 45.0  # Default


class TestThermodynamicMetrics:
    """Pruebas para ThermodynamicMetrics."""

    def test_thermodynamic_metrics_defaults(self):
        """Valida los valores por defecto."""
        metrics = ThermodynamicMetrics()
        assert metrics.system_temperature == 25.0
        assert metrics.entropy == 0.0
        assert metrics.exergy == 1.0
        assert metrics.heat_capacity == 0.5

    def test_thermodynamic_metrics_immutability(self):
        """Valida que la clase sea inmutable."""
        metrics = ThermodynamicMetrics()
        with pytest.raises(FrozenInstanceError):
            metrics.system_temperature = 30.0

    def test_thermodynamic_metrics_initialization(self):
        """Valida la inicialización con valores personalizados."""
        metrics = ThermodynamicMetrics(
            system_temperature=50.0,
            entropy=0.8
        )
        assert metrics.system_temperature == 50.0
        assert metrics.entropy == 0.8
        assert metrics.exergy == 1.0  # Default


class TestSchemaInteroperability:
    """Pruebas de interoperabilidad y conversión."""

    def test_all_schemas_are_dataclasses(self):
        """Valida que todos sean dataclasses."""
        assert is_dataclass(PhysicsMetrics)
        assert is_dataclass(TopologicalMetrics)
        assert is_dataclass(ControlMetrics)
        assert is_dataclass(ThermodynamicMetrics)

    def test_serialization_to_dict(self):
        """Valida que se puedan convertir a diccionario."""
        metrics = PhysicsMetrics(saturation=0.5)
        data = asdict(metrics)
        assert isinstance(data, dict)
        assert data["saturation"] == 0.5
        assert data["gyroscopic_stability"] == 1.0
