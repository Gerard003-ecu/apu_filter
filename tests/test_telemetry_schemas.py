"""
Tests para Telemetry Schemas (Validación del Espacio Vectorial de Estado)
========================================================================

Suite de pruebas para asegurar los invariantes matemáticos y físicos
definidos en `app.telemetry_schemas`.

Cobertura:
1. PhysicsMetrics: Invariantes de energía, saturación y fábrica RLC.
2. TopologicalMetrics: Invariantes de Betti, Euler y polinomios de Poincaré.
3. ControlMetrics: Estabilidad, polos y respuesta en frecuencia.
4. ThermodynamicMetrics: Leyes de la termodinámica financiera.
"""

import math
import pytest
from app.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
)


# =============================================================================
# 1. PhysicsMetrics Tests
# =============================================================================

class TestPhysicsMetrics:
    """Pruebas para el subespacio físico."""

    def test_invariants_valid(self):
        """Verifica que valores válidos no lancen excepción."""
        pm = PhysicsMetrics(
            saturation=0.5,
            kinetic_energy=10.0,
            potential_energy=5.0,
            dissipated_power=1.0,
            gyroscopic_stability=0.9
        )
        assert pm.saturation == 0.5
        assert pm.total_energy == 15.0

    def test_saturation_bounds(self):
        """Saturation debe estar en [0, 1]."""
        with pytest.raises(ValueError, match="Saturation"):
            PhysicsMetrics(saturation=1.1)
        with pytest.raises(ValueError, match="Saturation"):
            PhysicsMetrics(saturation=-0.1)

    def test_energy_non_negative(self):
        """Energías deben ser no negativas."""
        with pytest.raises(ValueError, match="Energies"):
            PhysicsMetrics(kinetic_energy=-1.0)
        with pytest.raises(ValueError, match="Energies"):
            PhysicsMetrics(potential_energy=-1.0)

    def test_efficiency_calculation(self):
        """Eficiencia = Ep / Etotal."""
        pm = PhysicsMetrics(kinetic_energy=10.0, potential_energy=10.0)
        assert pm.efficiency == 0.5  # 10 / 20

    def test_efficiency_zero_energy(self):
        """Eficiencia es 1.0 si energía total es 0 (sistema ocioso)."""
        pm = PhysicsMetrics(kinetic_energy=0.0, potential_energy=0.0)
        assert pm.efficiency == 1.0

    def test_energy_density(self):
        """Densidad = Etotal / Pressure."""
        pm = PhysicsMetrics(
            kinetic_energy=10.0,
            potential_energy=10.0,
            pressure=2.0
        )
        assert pm.energy_density == 10.0  # 20 / 2

    def test_from_components_factory(self):
        """Fábrica RLC calcula correctamente las energías."""
        # L=2, I=3 => Ek = 0.5 * 2 * 3^2 = 9
        # C=4, V=2 => Ep = 0.5 * 4 * 2^2 = 8
        pm = PhysicsMetrics.from_components(
            saturation=0.5,
            pressure=1.0,
            inductance=2.0,
            current=3.0,
            capacitance=4.0,
            voltage=2.0,
            resistance=1.0
        )
        assert pm.kinetic_energy == 9.0
        assert pm.potential_energy == 8.0
        assert pm.dissipated_power == 9.0  # R*I^2 = 1*9


# =============================================================================
# 2. TopologicalMetrics Tests
# =============================================================================

class TestTopologicalMetrics:
    """Pruebas para el subespacio topológico."""

    def test_euler_auto_calculation(self):
        """Euler se calcula automáticamente si es None."""
        # χ = β0 - β1 + β2 = 1 - 0 + 0 = 1
        tm = TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0)
        assert tm.euler_characteristic == 1

    def test_euler_consistency_check(self):
        """Si se provee Euler, debe ser consistente."""
        with pytest.raises(ValueError, match="Euler characteristic"):
            TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0, euler_characteristic=5)

    def test_betti_non_negative(self):
        """Números de Betti no negativos."""
        with pytest.raises(ValueError, match="Betti numbers"):
            TopologicalMetrics(beta_0=-1)

    def test_connectivity_properties(self):
        """Propiedades derivadas de conectividad."""
        # Conectado, sin ciclos
        tm = TopologicalMetrics(beta_0=1, beta_1=0)
        assert tm.is_connected is True
        assert tm.has_cycles is False
        assert tm.is_simply_connected is True

        # Desconectado
        tm_dis = TopologicalMetrics(beta_0=2)
        assert tm_dis.is_connected is False

        # Con ciclos
        tm_cyc = TopologicalMetrics(beta_0=1, beta_1=1)
        assert tm_cyc.has_cycles is True
        assert tm_cyc.is_simply_connected is False

    def test_poincare_polynomial(self):
        """Polinomio de Poincaré P(t)."""
        # P(t) = 1 + 2t + 3t^2
        tm = TopologicalMetrics(beta_0=1, beta_1=2, beta_2=3)
        assert tm.poincare_polynomial(0) == 1
        assert tm.poincare_polynomial(1) == 6  # Suma de Betti (complejidad)
        assert tm.poincare_polynomial(-1) == 2 # Euler: 1 - 2 + 3 = 2


# =============================================================================
# 3. ControlMetrics Tests
# =============================================================================

class TestControlMetrics:
    """Pruebas para el subespacio de control."""

    def test_stability_inference(self):
        """Estabilidad se infiere de polos."""
        # Polos estables (parte real negativa)
        cm = ControlMetrics(poles_real=(-1.0, -2.0))
        assert cm.is_stable is True

        # Polos inestables
        cm_unstable = ControlMetrics(poles_real=(1.0, -2.0))
        assert cm_unstable.is_stable is False

    def test_damping_properties(self):
        """Propiedades de amortiguamiento."""
        # Subamortiguado
        cm = ControlMetrics(damping_ratio=0.5)
        assert cm.is_underdamped is True
        assert cm.is_overdamped is False

        # Crítico
        cm_crit = ControlMetrics(damping_ratio=1.0)
        assert cm_crit.is_critically_damped is True

        # Sobreamortiguado
        cm_over = ControlMetrics(damping_ratio=1.5)
        assert cm_over.is_overdamped is True

    def test_settling_time(self):
        """Tiempo de asentamiento ts = 4 / (zeta * wn)."""
        cm = ControlMetrics(damping_ratio=0.5, natural_frequency=2.0)
        # ts = 4 / (0.5 * 2) = 4 / 1 = 4
        assert cm.settling_time == 4.0

    def test_from_poles_factory(self):
        """Fábrica desde polos complejos."""
        # s = -3 ± 4j
        # wn = sqrt(3^2 + 4^2) = 5
        # zeta = -(-3) / 5 = 0.6
        poles = [complex(-3, 4), complex(-3, -4)]
        cm = ControlMetrics.from_poles(poles)

        assert cm.natural_frequency == 5.0
        assert cm.damping_ratio == 0.6
        assert cm.is_stable is True


# =============================================================================
# 4. ThermodynamicMetrics Tests
# =============================================================================

class TestThermodynamicMetrics:
    """Pruebas para el subespacio termodinámico."""

    def test_non_negative_invariants(self):
        """Valores físicos no negativos."""
        with pytest.raises(ValueError, match="negative"):
            ThermodynamicMetrics(system_temperature=-5.0)
        with pytest.raises(ValueError, match="negative"):
            ThermodynamicMetrics(entropy=-0.1)

    def test_free_energy(self):
        """F = U - TS."""
        # U=100, T=10, S=2 => F = 100 - 20 = 80
        tm = ThermodynamicMetrics(
            exergy=100.0,
            system_temperature=10.0,
            entropy=2.0
        )
        assert tm.free_energy == 80.0

    def test_thermal_efficiency(self):
        """Eficiencia Carnot = 1 - T0/T."""
        # T=100, T0=25 => eff = 1 - 0.25 = 0.75
        tm = ThermodynamicMetrics(
            system_temperature=100.0,
            reference_temperature=25.0
        )
        assert tm.thermal_efficiency == 0.75

    def test_from_entropy_and_temp(self):
        """Fábrica calcula exergía como Cv*T."""
        tm = ThermodynamicMetrics.from_entropy_and_temp(
            temperature=100.0,
            entropy=0.5,
            heat_capacity=2.0
        )
        # Exergy = 2.0 * 100.0 = 200.0
        assert tm.exergy == 200.0
