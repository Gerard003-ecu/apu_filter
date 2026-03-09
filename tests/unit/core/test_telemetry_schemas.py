"""
Suite de Pruebas: Telemetry Schemas
===================================

Pruebas exhaustivas para validar la correctitud matemática y la robustez
de los subespacios métricos del sistema APU Filter.

Estructura de Pruebas:
----------------------
1. TestPhysicsMetrics: Validaciones físicas y modelo RLC
2. TestTopologicalMetrics: Invariantes homológicos y Euler-Poincaré
3. TestControlMetrics: Teoría de control y estabilidad
4. TestThermodynamicMetrics: Potenciales termodinámicos y eficiencias
5. TestSystemStateVector: Composición y evaluación de salud
6. TestNumericalEdgeCases: Casos límite numéricos
7. TestImmutability: Garantías de inmutabilidad
8. TestSerialization: Exportación a diccionarios

Ejecutar con:
    pytest test_telemetry_schemas.py -v --tb=short
    pytest test_telemetry_schemas.py -v --cov=telemetry_schemas --cov-report=term-missing

Requisitos:
    pip install pytest pytest-cov
"""

import math
import pytest
from typing import List, Tuple, Any
from dataclasses import FrozenInstanceError

# Importar el módulo bajo prueba
from app.telemetry_schemas import (
    PhysicsMetrics,
    TopologicalMetrics,
    ControlMetrics,
    ThermodynamicMetrics,
    SystemStateVector,
    EPSILON_ABSOLUTE,
    EPSILON_RELATIVE,
    EPSILON_FREQUENCY,
    SETTLING_TIME_CRITERION,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDOS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_physics() -> PhysicsMetrics:
    """Métricas físicas con valores por defecto."""
    return PhysicsMetrics()


@pytest.fixture
def energetic_physics() -> PhysicsMetrics:
    """Métricas físicas con energía significativa."""
    return PhysicsMetrics(
        saturation=0.75,
        pressure=2.5,
        kinetic_energy=100.0,
        potential_energy=50.0,
        dissipated_power=10.0,
        effective_capacitance=2.0,
        gyroscopic_stability=1.5,
    )


@pytest.fixture
def default_topology() -> TopologicalMetrics:
    """Topología de grafo conexo ideal (árbol)."""
    return TopologicalMetrics()


@pytest.fixture
def complex_topology() -> TopologicalMetrics:
    """Topología con ciclos y cavidades."""
    return TopologicalMetrics(
        beta_0=1,
        beta_1=3,
        beta_2=1,
        fiedler_value=0.5,
        spectral_gap=0.3,
    )


@pytest.fixture
def stable_control() -> ControlMetrics:
    """Sistema de control estable subamortiguado."""
    return ControlMetrics(
        poles_real=(-1.0, -2.0, -5.0),
        damping_ratio=0.707,
        natural_frequency=2.0,
        phase_margin_deg=60.0,
    )


@pytest.fixture
def unstable_control() -> ControlMetrics:
    """Sistema de control inestable."""
    return ControlMetrics(
        poles_real=(0.5, -1.0),
        damping_ratio=0.3,
        natural_frequency=1.0,
    )


@pytest.fixture
def default_thermo() -> ThermodynamicMetrics:
    """Métricas termodinámicas típicas."""
    return ThermodynamicMetrics(
        system_temperature=400.0,
        entropy=0.5,
        internal_energy=100.0,
        heat_capacity=2.0,
        reference_temperature=300.0,
    )


@pytest.fixture
def healthy_state(
    energetic_physics: PhysicsMetrics,
    default_topology: TopologicalMetrics,
    stable_control: ControlMetrics,
    default_thermo: ThermodynamicMetrics,
) -> SystemStateVector:
    """Vector de estado saludable completo."""
    return SystemStateVector(
        physics=energetic_physics,
        topology=default_topology,
        control=stable_control,
        thermodynamics=default_thermo,
        epoch=1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TESTS DE PHYSICS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhysicsMetricsValidation:
    """Pruebas de validación de invariantes físicos."""
    
    def test_default_values_are_valid(self, default_physics: PhysicsMetrics):
        """Los valores por defecto deben ser válidos."""
        assert default_physics.saturation == 0.0
        assert default_physics.total_energy == 0.0
        assert default_physics.gyroscopic_stability == 1.0
    
    @pytest.mark.parametrize("saturation", [-0.1, -1.0, 1.1, 2.0, 100.0])
    def test_saturation_out_of_range_raises(self, saturation: float):
        """Saturación fuera de [0, 1] debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Saturación"):
            PhysicsMetrics(saturation=saturation)
    
    @pytest.mark.parametrize("saturation", [0.0, 0.5, 1.0, 0.001, 0.999])
    def test_saturation_valid_range(self, saturation: float):
        """Saturación en [0, 1] debe ser aceptada."""
        phys = PhysicsMetrics(saturation=saturation)
        assert phys.saturation == saturation
    
    def test_negative_kinetic_energy_raises(self):
        """Energía cinética negativa viola E_k = ½LI² ≥ 0."""
        with pytest.raises(ValueError, match="cinética"):
            PhysicsMetrics(kinetic_energy=-0.001)
    
    def test_negative_potential_energy_raises(self):
        """Energía potencial negativa viola E_p = ½CV² ≥ 0."""
        with pytest.raises(ValueError, match="potencial"):
            PhysicsMetrics(potential_energy=-1.0)
    
    def test_negative_dissipated_power_raises(self):
        """Potencia disipada negativa viola P = RI² ≥ 0."""
        with pytest.raises(ValueError, match="disipada"):
            PhysicsMetrics(dissipated_power=-0.5)
    
    def test_zero_effective_capacitance_raises(self):
        """Capacitancia efectiva debe ser estrictamente positiva."""
        with pytest.raises(ValueError, match="Capacitancia"):
            PhysicsMetrics(effective_capacitance=0.0)
    
    def test_negative_effective_capacitance_raises(self):
        """Capacitancia negativa es físicamente imposible."""
        with pytest.raises(ValueError, match="Capacitancia"):
            PhysicsMetrics(effective_capacitance=-1.0)
    
    def test_negative_gyroscopic_stability_raises(self):
        """Estabilidad giroscópica negativa no tiene sentido físico."""
        with pytest.raises(ValueError, match="giroscópica"):
            PhysicsMetrics(gyroscopic_stability=-0.1)
    
    def test_negative_hamiltonian_excess_raises(self):
        """Exceso hamiltoniano negativo viola conservación de energía."""
        with pytest.raises(ValueError, match="hamiltoniano"):
            PhysicsMetrics(hamiltonian_excess=-1.0)


class TestPhysicsMetricsProperties:
    """Pruebas de propiedades calculadas."""
    
    def test_total_energy_is_sum(self, energetic_physics: PhysicsMetrics):
        """E_total = E_k + E_p."""
        assert energetic_physics.total_energy == 150.0  # 100 + 50
    
    def test_potential_ratio_calculation(self):
        """ρ_p = E_p / E_total."""
        phys = PhysicsMetrics(kinetic_energy=30.0, potential_energy=70.0)
        assert math.isclose(phys.potential_ratio, 0.7, rel_tol=1e-9)
    
    def test_kinetic_ratio_complement(self):
        """ρ_k = 1 - ρ_p."""
        phys = PhysicsMetrics(kinetic_energy=30.0, potential_energy=70.0)
        assert math.isclose(phys.kinetic_ratio, 0.3, rel_tol=1e-9)
        assert math.isclose(phys.potential_ratio + phys.kinetic_ratio, 1.0)
    
    def test_potential_ratio_zero_energy(self, default_physics: PhysicsMetrics):
        """Con E_total = 0, ρ_p debe ser 1.0 (estado de reposo)."""
        assert default_physics.potential_ratio == 1.0
    
    def test_energy_density_formula(self):
        """u = E_total / C_eff."""
        phys = PhysicsMetrics(
            kinetic_energy=10.0,
            potential_energy=10.0,
            effective_capacitance=4.0,
        )
        assert math.isclose(phys.energy_density, 5.0)  # 20 / 4
    
    def test_dissipation_efficiency_formula(self):
        """η = E_total / (E_total + P_dis)."""
        phys = PhysicsMetrics(
            kinetic_energy=80.0,
            potential_energy=20.0,
            dissipated_power=25.0,
        )
        # η = 100 / (100 + 25) = 100/125 = 0.8
        assert math.isclose(phys.dissipation_efficiency, 0.8, rel_tol=1e-9)
    
    def test_dissipation_efficiency_no_dissipation(self):
        """Sin disipación, η = 1.0."""
        phys = PhysicsMetrics(kinetic_energy=50.0, dissipated_power=0.0)
        assert phys.dissipation_efficiency == 1.0
    
    def test_dissipation_efficiency_zero_energy_zero_dissipation(self):
        """Con E = 0 y P_dis = 0, η = 1.0."""
        phys = PhysicsMetrics()
        assert phys.dissipation_efficiency == 1.0
    
    def test_quality_factor_formula(self):
        """Q = E_total / P_dis."""
        phys = PhysicsMetrics(
            kinetic_energy=50.0,
            potential_energy=50.0,
            dissipated_power=10.0,
        )
        assert math.isclose(phys.quality_factor, 10.0)  # 100 / 10
    
    def test_quality_factor_no_dissipation_is_infinite(self):
        """Sin disipación, Q = ∞."""
        phys = PhysicsMetrics(kinetic_energy=100.0, dissipated_power=0.0)
        assert phys.quality_factor == float('inf')


class TestPhysicsMetricsFactory:
    """Pruebas del método de fábrica from_rlc_parameters."""
    
    def test_rlc_factory_kinetic_energy(self):
        """E_k = ½·L·I²."""
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.5, pressure=1.0,
            inductance=2.0, current=3.0,
            capacitance=1.0, voltage=0.0,
            resistance=0.0,
        )
        # E_k = 0.5 * 2 * 9 = 9
        assert math.isclose(phys.kinetic_energy, 9.0)
    
    def test_rlc_factory_potential_energy(self):
        """E_p = ½·C·V²."""
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.5, pressure=1.0,
            inductance=0.0, current=0.0,
            capacitance=0.02, voltage=100.0,
            resistance=0.0,
        )
        # E_p = 0.5 * 0.02 * 10000 = 100
        assert math.isclose(phys.potential_energy, 100.0)
    
    def test_rlc_factory_flyback_voltage(self):
        """V_fb = L·dI/dt."""
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.5, pressure=1.0,
            inductance=0.5, current=0.0,
            capacitance=1.0, voltage=0.0,
            resistance=0.0, di_dt=10.0,
        )
        # V_fb = 0.5 * 10 = 5
        assert math.isclose(phys.flyback_voltage, 5.0)
    
    def test_rlc_factory_dissipated_power(self):
        """P_dis = R·I²."""
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.5, pressure=1.0,
            inductance=1.0, current=4.0,
            capacitance=1.0, voltage=0.0,
            resistance=5.0,
        )
        # P_dis = 5 * 16 = 80
        assert math.isclose(phys.dissipated_power, 80.0)
    
    def test_rlc_factory_negative_inductance_raises(self):
        """L < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Inductancia"):
            PhysicsMetrics.from_rlc_parameters(
                saturation=0.5, pressure=1.0,
                inductance=-1.0, current=1.0,
                capacitance=1.0, voltage=1.0,
                resistance=1.0,
            )
    
    def test_rlc_factory_zero_capacitance_raises(self):
        """C = 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Capacitancia"):
            PhysicsMetrics.from_rlc_parameters(
                saturation=0.5, pressure=1.0,
                inductance=1.0, current=1.0,
                capacitance=0.0, voltage=1.0,
                resistance=1.0,
            )
    
    def test_rlc_factory_negative_resistance_raises(self):
        """R < 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Resistencia"):
            PhysicsMetrics.from_rlc_parameters(
                saturation=0.5, pressure=1.0,
                inductance=1.0, current=1.0,
                capacitance=1.0, voltage=1.0,
                resistance=-1.0,
            )
    
    def test_rlc_factory_sets_effective_capacitance(self):
        """La fábrica debe establecer effective_capacitance = C."""
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.5, pressure=1.0,
            inductance=1.0, current=1.0,
            capacitance=0.05, voltage=10.0,
            resistance=1.0,
        )
        assert phys.effective_capacitance == 0.05
    
    def test_rlc_factory_negative_current_squared(self):
        """Corriente negativa debe dar energía positiva (I²)."""
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.5, pressure=1.0,
            inductance=2.0, current=-3.0,
            capacitance=1.0, voltage=0.0,
            resistance=0.0,
        )
        assert math.isclose(phys.kinetic_energy, 9.0)  # 0.5 * 2 * 9


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TESTS DE TOPOLOGICAL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopologicalMetricsValidation:
    """Pruebas de validación de invariantes topológicos."""
    
    def test_default_is_connected_tree(self, default_topology: TopologicalMetrics):
        """Por defecto: grafo conexo sin ciclos (árbol)."""
        assert default_topology.beta_0 == 1
        assert default_topology.beta_1 == 0
        assert default_topology.beta_2 == 0
        assert default_topology.euler_characteristic == 1
    
    @pytest.mark.parametrize("beta_idx,beta_val", [
        (0, -1), (1, -1), (2, -1),
        (0, -100), (1, -5), (2, -2),
    ])
    def test_negative_betti_numbers_raise(self, beta_idx: int, beta_val: int):
        """Números de Betti negativos son matemáticamente imposibles."""
        kwargs = {f"beta_{beta_idx}": beta_val}
        with pytest.raises(ValueError, match=f"β_{beta_idx}"):
            TopologicalMetrics(**kwargs)
    
    def test_euler_poincare_auto_calculation(self):
        """χ se auto-calcula si no se especifica."""
        topo = TopologicalMetrics(beta_0=3, beta_1=2, beta_2=1)
        # χ = 3 - 2 + 1 = 2
        assert topo.euler_characteristic == 2
    
    def test_euler_poincare_explicit_correct(self):
        """χ explícito correcto debe ser aceptado."""
        topo = TopologicalMetrics(
            beta_0=2, beta_1=1, beta_2=0,
            euler_characteristic=1,  # 2 - 1 + 0 = 1 ✓
        )
        assert topo.euler_characteristic == 1
    
    def test_euler_poincare_explicit_incorrect_raises(self):
        """χ explícito incorrecto debe lanzar ValueError."""
        with pytest.raises(ValueError, match="Euler-Poincaré"):
            TopologicalMetrics(
                beta_0=2, beta_1=1, beta_2=0,
                euler_characteristic=5,  # 2 - 1 + 0 = 1 ≠ 5
            )
    
    def test_negative_fiedler_value_raises(self):
        """λ₂ < 0 viola que el Laplaciano es semidefinido positivo."""
        with pytest.raises(ValueError, match="Fiedler"):
            TopologicalMetrics(fiedler_value=-0.1)
    
    def test_negative_spectral_gap_raises(self):
        """Gap espectral negativo no tiene sentido."""
        with pytest.raises(ValueError, match="espectral"):
            TopologicalMetrics(spectral_gap=-0.5)
    
    def test_negative_pyramid_stability_raises(self):
        """Estabilidad piramidal negativa no tiene sentido."""
        with pytest.raises(ValueError, match="piramidal"):
            TopologicalMetrics(pyramid_stability=-1.0)
    
    def test_negative_structural_entropy_raises(self):
        """Entropía estructural negativa viola definición de entropía."""
        with pytest.raises(ValueError, match="estructural"):
            TopologicalMetrics(structural_entropy=-0.001)


class TestTopologicalMetricsProperties:
    """Pruebas de propiedades topológicas derivadas."""
    
    @pytest.mark.parametrize("beta_0,expected", [
        (1, True), (2, False), (0, False), (5, False),
    ])
    def test_is_connected(self, beta_0: int, expected: bool):
        """Conexo ⟺ β₀ = 1."""
        topo = TopologicalMetrics(beta_0=beta_0)
        assert topo.is_connected == expected
    
    @pytest.mark.parametrize("beta_1,expected", [
        (0, False), (1, True), (5, True),
    ])
    def test_has_cycles(self, beta_1: int, expected: bool):
        """Tiene ciclos ⟺ β₁ > 0."""
        topo = TopologicalMetrics(beta_1=beta_1)
        assert topo.has_cycles == expected
    
    @pytest.mark.parametrize("beta_2,expected", [
        (0, False), (1, True), (3, True),
    ])
    def test_has_cavities(self, beta_2: int, expected: bool):
        """Tiene cavidades ⟺ β₂ > 0."""
        topo = TopologicalMetrics(beta_2=beta_2)
        assert topo.has_cavities == expected
    
    @pytest.mark.parametrize("beta_0,beta_1,expected", [
        (1, 0, True),   # Conexo sin ciclos
        (1, 1, False),  # Conexo con ciclos
        (2, 0, False),  # Disconexo sin ciclos
        (2, 2, False),  # Disconexo con ciclos
    ])
    def test_is_simply_connected(self, beta_0: int, beta_1: int, expected: bool):
        """Simplemente conexo ⟺ β₀ = 1 ∧ β₁ = 0."""
        topo = TopologicalMetrics(beta_0=beta_0, beta_1=beta_1)
        assert topo.is_simply_connected == expected
    
    @pytest.mark.parametrize("beta_1,beta_2,expected", [
        (0, 0, True),   # Árbol
        (1, 0, False),  # Con ciclos
        (0, 1, False),  # Con cavidades
        (2, 1, False),  # Con ambos
    ])
    def test_is_acyclic(self, beta_1: int, beta_2: int, expected: bool):
        """Acíclico ⟺ β₁ = 0 ∧ β₂ = 0."""
        topo = TopologicalMetrics(beta_1=beta_1, beta_2=beta_2)
        assert topo.is_acyclic == expected
    
    def test_betti_vector_is_tuple(self, complex_topology: TopologicalMetrics):
        """betti_vector devuelve tupla inmutable."""
        bv = complex_topology.betti_vector
        assert isinstance(bv, tuple)
        assert bv == (1, 3, 1)
    
    def test_total_betti_number(self):
        """Σβᵢ = β₀ + β₁ + β₂."""
        topo = TopologicalMetrics(beta_0=2, beta_1=5, beta_2=3)
        assert topo.total_betti_number == 10
    
    @pytest.mark.parametrize("betti,expected_dim", [
        ((1, 0, 0), 0),   # Solo componentes
        ((1, 2, 0), 1),   # Con ciclos
        ((1, 0, 1), 2),   # Con cavidades
        ((2, 3, 4), 2),   # Todo
        ((0, 0, 0), -1),  # Espacio vacío
    ])
    def test_homological_dimension(
        self,
        betti: Tuple[int, int, int],
        expected_dim: int,
    ):
        """Dimensión homológica = max{i : βᵢ > 0}."""
        topo = TopologicalMetrics(beta_0=betti[0], beta_1=betti[1], beta_2=betti[2])
        assert topo.homological_dimension == expected_dim
    
    def test_cyclomatic_complexity(self):
        """M = β₁ - β₀ + 1."""
        topo = TopologicalMetrics(beta_0=1, beta_1=5)
        # M = 5 - 1 + 1 = 5
        assert topo.cyclomatic_complexity == 5
    
    def test_cyclomatic_complexity_tree(self, default_topology: TopologicalMetrics):
        """Árbol: M = 0 - 1 + 1 = 0."""
        assert default_topology.cyclomatic_complexity == 0


class TestPoincarePolynomial:
    """Pruebas del polinomio de Poincaré P(t) = β₀ + β₁t + β₂t²."""
    
    def test_poincare_at_one_equals_total_betti(self):
        """P(1) = Σβᵢ."""
        topo = TopologicalMetrics(beta_0=2, beta_1=3, beta_2=4)
        assert topo.poincare_polynomial(1.0) == 9.0
        assert topo.poincare_polynomial(1.0) == topo.total_betti_number
    
    def test_poincare_at_minus_one_equals_euler(self):
        """P(-1) = χ."""
        topo = TopologicalMetrics(beta_0=2, beta_1=3, beta_2=4)
        # χ = 2 - 3 + 4 = 3
        assert topo.poincare_polynomial(-1.0) == 3.0
        assert topo.poincare_polynomial(-1.0) == topo.euler_characteristic
    
    def test_poincare_at_zero_equals_beta_zero(self):
        """P(0) = β₀."""
        topo = TopologicalMetrics(beta_0=5, beta_1=3, beta_2=2)
        assert topo.poincare_polynomial(0.0) == 5.0
    
    def test_poincare_polynomial_formula(self):
        """P(t) = β₀ + β₁t + β₂t² para t arbitrario."""
        topo = TopologicalMetrics(beta_0=1, beta_1=2, beta_2=3)
        t = 2.5
        expected = 1 + 2 * 2.5 + 3 * 2.5**2  # 1 + 5 + 18.75 = 24.75
        assert math.isclose(topo.poincare_polynomial(t), expected)
    
    def test_poincare_polynomial_log_valid(self):
        """log(P(t)) para P(t) > 0."""
        topo = TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0)
        # P(t) = 1 para todo t → log(1) = 0
        assert topo.poincare_polynomial_log(0.0) == 0.0
        assert topo.poincare_polynomial_log(10.0) == 0.0
    
    def test_poincare_polynomial_log_returns_none_when_negative(self):
        """log(P(t)) = None si P(t) ≤ 0."""
        topo = TopologicalMetrics(beta_0=0, beta_1=1, beta_2=0)
        # P(t) = t, P(-1) = -1 < 0
        assert topo.poincare_polynomial_log(-1.0) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TESTS DE CONTROL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestControlMetricsValidation:
    """Pruebas de validación de parámetros de control."""
    
    def test_default_values(self):
        """Valores por defecto representan sistema estable nominal."""
        ctrl = ControlMetrics()
        assert ctrl.is_stable is True
        assert ctrl.poles_real == ()
        assert ctrl.damping_ratio == 0.707
    
    def test_negative_damping_ratio_raises(self):
        """ζ < 0 no tiene sentido físico."""
        with pytest.raises(ValueError, match="amortiguamiento"):
            ControlMetrics(damping_ratio=-0.1)
    
    def test_negative_natural_frequency_raises(self):
        """ω_n < 0 no tiene sentido físico."""
        with pytest.raises(ValueError, match="natural"):
            ControlMetrics(natural_frequency=-1.0)
    
    def test_list_poles_converted_to_tuple(self):
        """Lista de polos debe convertirse a tupla (inmutabilidad)."""
        ctrl = ControlMetrics(poles_real=[-1.0, -2.0])
        assert isinstance(ctrl.poles_real, tuple)
        assert ctrl.poles_real == (-1.0, -2.0)
    
    def test_stability_inferred_from_poles_stable(self):
        """is_stable = True si todos los polos tienen σ < 0."""
        ctrl = ControlMetrics(poles_real=(-0.5, -1.0, -10.0))
        assert ctrl.is_stable is True
    
    def test_stability_inferred_from_poles_unstable(self):
        """is_stable = False si algún polo tiene σ ≥ 0."""
        ctrl = ControlMetrics(poles_real=(-1.0, 0.1))
        assert ctrl.is_stable is False
    
    def test_stability_inferred_from_poles_marginal(self):
        """Polo en σ = 0 implica sistema no estable (marginalmente estable)."""
        ctrl = ControlMetrics(poles_real=(-1.0, 0.0))
        assert ctrl.is_stable is False
    
    def test_stability_empty_poles_defaults_stable(self):
        """Sin polos especificados, asumir estable."""
        ctrl = ControlMetrics(poles_real=())
        assert ctrl.is_stable is True
    
    def test_stability_explicit_consistent_accepted(self):
        """is_stable explícito consistente con polos debe ser aceptado."""
        ctrl = ControlMetrics(poles_real=(-1.0, -2.0), is_stable=True)
        assert ctrl.is_stable is True
    
    def test_stability_explicit_inconsistent_raises(self):
        """is_stable=True con polos positivos debe lanzar error."""
        with pytest.raises(ValueError, match="Inconsistencia"):
            ControlMetrics(poles_real=(0.5, -1.0), is_stable=True)


class TestControlMetricsPolesProperties:
    """Pruebas de propiedades relacionadas con polos."""
    
    def test_dominant_pole_is_maximum_real(self):
        """Polo dominante = max(σ) (más cercano al eje jω)."""
        ctrl = ControlMetrics(poles_real=(-5.0, -1.0, -0.2, -3.0))
        assert ctrl.dominant_pole == -0.2
    
    def test_dominant_pole_empty_is_negative_infinity(self):
        """Sin polos, σ_dom = -∞."""
        ctrl = ControlMetrics(poles_real=())
        assert ctrl.dominant_pole == -float('inf')
    
    def test_fastest_pole_is_minimum_real(self):
        """Polo más rápido = min(σ)."""
        ctrl = ControlMetrics(poles_real=(-5.0, -1.0, -0.2, -3.0))
        assert ctrl.fastest_pole == -5.0
    
    def test_fastest_pole_empty_is_positive_infinity(self):
        """Sin polos, σ_fast = +∞."""
        ctrl = ControlMetrics(poles_real=())
        assert ctrl.fastest_pole == float('inf')
    
    def test_stability_margin_formula(self):
        """Margen de estabilidad = -σ_dom."""
        ctrl = ControlMetrics(poles_real=(-0.5, -2.0))
        assert ctrl.stability_margin == 0.5  # -(-0.5)


class TestControlMetricsDampingClassification:
    """Pruebas de clasificación de amortiguamiento."""
    
    def test_undamped_zero_zeta(self):
        """ζ = 0 → UNDAMPED."""
        ctrl = ControlMetrics(damping_ratio=0.0)
        assert ctrl.damping_category == "UNDAMPED"
    
    @pytest.mark.parametrize("zeta", [0.1, 0.5, 0.707, 0.99])
    def test_underdamped(self, zeta: float):
        """0 < ζ < 1 → UNDERDAMPED."""
        ctrl = ControlMetrics(damping_ratio=zeta)
        assert ctrl.is_underdamped is True
        assert ctrl.damping_category == "UNDERDAMPED"
    
    @pytest.mark.parametrize("zeta", [1.0, 0.999, 1.001])
    def test_critically_damped(self, zeta: float):
        """ζ ≈ 1 → CRITICALLY_DAMPED."""
        ctrl = ControlMetrics(damping_ratio=zeta)
        # Verificar que reconoce valores muy cercanos a 1
        if abs(zeta - 1.0) < 1e-3:
            assert ctrl.is_critically_damped is True
    
    @pytest.mark.parametrize("zeta", [1.1, 2.0, 5.0])
    def test_overdamped(self, zeta: float):
        """ζ > 1 → OVERDAMPED."""
        ctrl = ControlMetrics(damping_ratio=zeta)
        assert ctrl.is_overdamped is True
        assert ctrl.damping_category == "OVERDAMPED"


class TestControlMetricsDynamicProperties:
    """Pruebas de propiedades dinámicas."""
    
    def test_damped_frequency_underdamped(self):
        """ω_d = ω_n·√(1-ζ²) para ζ < 1."""
        ctrl = ControlMetrics(damping_ratio=0.6, natural_frequency=10.0)
        # ω_d = 10 * sqrt(1 - 0.36) = 10 * sqrt(0.64) = 10 * 0.8 = 8
        assert math.isclose(ctrl.damped_frequency, 8.0)
    
    def test_damped_frequency_critically_damped(self):
        """ω_d = 0 para ζ = 1."""
        ctrl = ControlMetrics(damping_ratio=1.0, natural_frequency=10.0)
        assert ctrl.damped_frequency == 0.0
    
    def test_damped_frequency_overdamped(self):
        """ω_d = 0 para ζ > 1."""
        ctrl = ControlMetrics(damping_ratio=2.0, natural_frequency=10.0)
        assert ctrl.damped_frequency == 0.0
    
    def test_settling_time_underdamped(self):
        """t_s ≈ 4/(ζ·ω_n) para ζ < 1."""
        ctrl = ControlMetrics(damping_ratio=0.5, natural_frequency=4.0)
        # t_s = 4 / (0.5 * 4) = 4 / 2 = 2
        assert math.isclose(ctrl.settling_time, 2.0)
    
    def test_settling_time_critically_damped(self):
        """t_s ≈ 4.75/ω_n para ζ = 1."""
        ctrl = ControlMetrics(damping_ratio=1.0, natural_frequency=2.0)
        assert math.isclose(ctrl.settling_time, 4.75 / 2.0)
    
    def test_settling_time_overdamped(self):
        """t_s = 4/|σ_dom| para ζ > 1."""
        ctrl = ControlMetrics(damping_ratio=2.0, natural_frequency=2.0)
        # σ_dom = ω_n·(ζ - √(ζ²-1)) = 2·(2 - √3) ≈ 2·(2 - 1.732) ≈ 0.536
        # t_s = 4 / 0.536 ≈ 7.46
        expected_sigma = 2.0 * (2.0 - math.sqrt(3.0))
        expected_ts = 4.0 / expected_sigma
        assert math.isclose(ctrl.settling_time, expected_ts, rel_tol=1e-6)
    
    def test_settling_time_zero_frequency_is_infinite(self):
        """t_s = ∞ si ω_n = 0."""
        ctrl = ControlMetrics(natural_frequency=0.0)
        assert ctrl.settling_time == float('inf')
    
    def test_rise_time_formula(self):
        """t_r ≈ (1 + 1.1ζ + 1.4ζ²) / ω_n."""
        ctrl = ControlMetrics(damping_ratio=0.707, natural_frequency=5.0)
        zeta = 0.707
        expected = (1.0 + 1.1 * zeta + 1.4 * zeta**2) / 5.0
        assert math.isclose(ctrl.rise_time, expected)
    
    def test_rise_time_zero_frequency_is_infinite(self):
        """t_r = ∞ si ω_n = 0."""
        ctrl = ControlMetrics(natural_frequency=0.0)
        assert ctrl.rise_time == float('inf')
    
    def test_peak_overshoot_underdamped(self):
        """M_p = exp(-πζ/√(1-ζ²))·100% para ζ < 1."""
        ctrl = ControlMetrics(damping_ratio=0.5)
        # M_p = exp(-π·0.5/√0.75)·100 = exp(-1.814)·100 ≈ 16.3%
        discriminant = math.sqrt(1 - 0.5**2)
        expected = 100.0 * math.exp(-math.pi * 0.5 / discriminant)
        assert math.isclose(ctrl.peak_overshoot, expected, rel_tol=1e-6)
    
    def test_peak_overshoot_critically_damped(self):
        """M_p = 0% para ζ ≥ 1."""
        ctrl = ControlMetrics(damping_ratio=1.0)
        assert ctrl.peak_overshoot == 0.0
    
    def test_peak_overshoot_overdamped(self):
        """M_p = 0% para ζ > 1."""
        ctrl = ControlMetrics(damping_ratio=1.5)
        assert ctrl.peak_overshoot == 0.0
    
    def test_peak_overshoot_undamped(self):
        """M_p = 100% para ζ = 0 (oscilación pura)."""
        ctrl = ControlMetrics(damping_ratio=0.0)
        assert ctrl.peak_overshoot == 100.0


class TestControlMetricsChaos:
    """Pruebas de propiedades de dinámica caótica."""
    
    def test_is_chaotic_positive_lyapunov(self):
        """λ_L > 0 → sistema caótico."""
        ctrl = ControlMetrics(lyapunov_exponent=0.1)
        assert ctrl.is_chaotic is True
    
    def test_is_chaotic_negative_lyapunov(self):
        """λ_L < 0 → sistema estable."""
        ctrl = ControlMetrics(lyapunov_exponent=-0.5)
        assert ctrl.is_chaotic is False
    
    def test_is_chaotic_zero_lyapunov(self):
        """λ_L = 0 → bifurcación, no caos."""
        ctrl = ControlMetrics(lyapunov_exponent=0.0)
        assert ctrl.is_chaotic is False
    
    def test_is_chaotic_none_lyapunov(self):
        """λ_L = None → no caótico por defecto."""
        ctrl = ControlMetrics(lyapunov_exponent=None)
        assert ctrl.is_chaotic is False
    
    def test_lyapunov_time_positive_exponent(self):
        """τ_L = 1/λ_L para λ_L > 0."""
        ctrl = ControlMetrics(lyapunov_exponent=0.25)
        assert ctrl.lyapunov_time == 4.0
    
    def test_lyapunov_time_non_positive_is_infinite(self):
        """τ_L = ∞ para λ_L ≤ 0."""
        ctrl = ControlMetrics(lyapunov_exponent=-1.0)
        assert ctrl.lyapunov_time == float('inf')
    
    def test_lyapunov_time_none_is_infinite(self):
        """τ_L = ∞ si λ_L = None."""
        ctrl = ControlMetrics(lyapunov_exponent=None)
        assert ctrl.lyapunov_time == float('inf')


class TestControlMetricsFactory:
    """Pruebas del método de fábrica from_poles."""
    
    def test_from_poles_empty(self):
        """Sin polos, retorna defaults."""
        ctrl = ControlMetrics.from_poles([])
        assert ctrl.poles_real == ()
        assert ctrl.is_stable is True
    
    def test_from_poles_complex_conjugate_pair(self):
        """Par conjugado: s = -1 ± 2j → ω_n = √5, ζ = 1/√5."""
        poles = [complex(-1, 2), complex(-1, -2)]
        ctrl = ControlMetrics.from_poles(poles)
        
        expected_wn = math.sqrt(5)  # |s| = √(1 + 4)
        expected_zeta = 1.0 / expected_wn  # -(-1) / √5
        
        assert math.isclose(ctrl.natural_frequency, expected_wn, rel_tol=1e-9)
        assert math.isclose(ctrl.damping_ratio, expected_zeta, rel_tol=1e-9)
        assert ctrl.is_stable is True
    
    def test_from_poles_real_stable(self):
        """Polo real estable: s = -3 → ω_n = 3, ζ = 1."""
        poles = [complex(-3, 0)]
        ctrl = ControlMetrics.from_poles(poles)
        
        assert math.isclose(ctrl.natural_frequency, 3.0)
        assert math.isclose(ctrl.damping_ratio, 1.0)
        assert ctrl.is_stable is True
    
    def test_from_poles_real_unstable(self):
        """Polo real inestable: s = 2 → ω_n = 2, ζ = 0."""
        poles = [complex(2, 0)]
        ctrl = ControlMetrics.from_poles(poles)
        
        assert math.isclose(ctrl.natural_frequency, 2.0)
        assert math.isclose(ctrl.damping_ratio, 0.0)
        assert ctrl.is_stable is False
    
    def test_from_poles_dominant_selection(self):
        """El polo dominante es max(Re(s))."""
        # Polos: -0.1, -2, -5 → dominante = -0.1
        poles = [complex(-2, 0), complex(-0.1, 0), complex(-5, 0)]
        ctrl = ControlMetrics.from_poles(poles)
        
        # ω_n y ζ se calculan del polo dominante (-0.1)
        assert math.isclose(ctrl.natural_frequency, 0.1)
        assert math.isclose(ctrl.damping_ratio, 1.0)
    
    def test_from_poles_extracts_real_parts(self):
        """poles_real debe contener partes reales de todos los polos."""
        poles = [complex(-1, 2), complex(-3, 0), complex(0.5, -1)]
        ctrl = ControlMetrics.from_poles(poles)
        
        assert ctrl.poles_real == (-1.0, -3.0, 0.5)
    
    def test_from_poles_with_kwargs(self):
        """Parámetros adicionales deben pasarse correctamente."""
        poles = [complex(-1, 0)]
        ctrl = ControlMetrics.from_poles(
            poles,
            phase_margin_deg=90.0,
            gain_margin_db=20.0,
        )
        
        assert ctrl.phase_margin_deg == 90.0
        assert ctrl.gain_margin_db == 20.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TESTS DE THERMODYNAMIC METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestThermodynamicMetricsValidation:
    """Pruebas de validación de parámetros termodinámicos."""
    
    def test_default_values(self):
        """Valores por defecto representan sistema en equilibrio."""
        thermo = ThermodynamicMetrics()
        assert thermo.system_temperature == 300.0
        assert thermo.entropy == 0.0
        assert thermo.internal_energy == 0.0
    
    def test_negative_system_temperature_raises(self):
        """T < 0 viola el tercer principio."""
        with pytest.raises(ValueError, match="[Tt]emperatura"):
            ThermodynamicMetrics(system_temperature=-1.0)
    
    def test_negative_reference_temperature_raises(self):
        """T₀ < 0 viola el tercer principio."""
        with pytest.raises(ValueError, match="referencia"):
            ThermodynamicMetrics(reference_temperature=-10.0)
    
    def test_negative_entropy_raises(self):
        """S < 0 viola el segundo principio."""
        with pytest.raises(ValueError, match="[Ee]ntropía"):
            ThermodynamicMetrics(entropy=-0.1)
    
    def test_negative_heat_capacity_raises(self):
        """C_v < 0 viola estabilidad térmica."""
        with pytest.raises(ValueError, match="[Cc]alorífica"):
            ThermodynamicMetrics(heat_capacity=-1.0)
    
    def test_zero_heat_capacity_allowed(self):
        """C_v = 0 debe ser permitido (límite)."""
        thermo = ThermodynamicMetrics(heat_capacity=0.0)
        assert thermo.heat_capacity == 0.0


class TestThermodynamicMetricsPotentials:
    """Pruebas de potenciales termodinámicos."""
    
    def test_helmholtz_free_energy(self, default_thermo: ThermodynamicMetrics):
        """F = U - T·S."""
        # F = 100 - 400*0.5 = 100 - 200 = -100
        assert math.isclose(default_thermo.helmholtz_free_energy, -100.0)
    
    def test_gibbs_free_energy(self):
        """G = F + PV = U - TS + PV."""
        thermo = ThermodynamicMetrics(
            system_temperature=300.0,
            entropy=1.0,
            internal_energy=500.0,
            pressure_volume=100.0,
        )
        # F = 500 - 300*1 = 200
        # G = 200 + 100 = 300
        assert math.isclose(thermo.gibbs_free_energy, 300.0)
    
    def test_enthalpy(self):
        """H = U + PV."""
        thermo = ThermodynamicMetrics(
            internal_energy=100.0,
            pressure_volume=50.0,
        )
        assert math.isclose(thermo.enthalpy, 150.0)
    
    def test_exergy(self):
        """Ex = U - T₀·(S - S₀)."""
        thermo = ThermodynamicMetrics(
            internal_energy=200.0,
            entropy=2.0,
            reference_temperature=300.0,
            reference_entropy=0.5,
        )
        # Ex = 200 - 300*(2 - 0.5) = 200 - 300*1.5 = 200 - 450 = -250
        assert math.isclose(thermo.exergy, -250.0)
    
    def test_entropic_penalty(self):
        """Penalización entrópica = T·S."""
        thermo = ThermodynamicMetrics(
            system_temperature=350.0,
            entropy=2.0,
        )
        assert math.isclose(thermo.entropic_penalty, 700.0)


class TestThermodynamicMetricsEfficiencies:
    """Pruebas de eficiencias termodinámicas."""
    
    def test_carnot_efficiency_typical(self):
        """η_C = 1 - T₀/T."""
        thermo = ThermodynamicMetrics(
            system_temperature=400.0,
            reference_temperature=300.0,
        )
        # η = 1 - 300/400 = 0.25
        assert math.isclose(thermo.carnot_efficiency, 0.25)
    
    def test_carnot_efficiency_equilibrium(self):
        """η_C = 0 si T = T₀ (equilibrio térmico)."""
        thermo = ThermodynamicMetrics(
            system_temperature=300.0,
            reference_temperature=300.0,
        )
        assert thermo.carnot_efficiency == 0.0
    
    def test_carnot_efficiency_heat_pump(self):
        """η_C < 0 si T < T₀ (bomba de calor)."""
        thermo = ThermodynamicMetrics(
            system_temperature=200.0,
            reference_temperature=300.0,
        )
        # η = 1 - 300/200 = 1 - 1.5 = -0.5
        assert math.isclose(thermo.carnot_efficiency, -0.5)
    
    def test_carnot_efficiency_zero_temperature(self):
        """η_C = 0 si T = 0 (tercer principio)."""
        thermo = ThermodynamicMetrics(
            system_temperature=0.0,
            reference_temperature=300.0,
        )
        assert thermo.carnot_efficiency == 0.0
    
    def test_carnot_efficiency_cold_sink_zero(self):
        """η_C = 1 si T₀ = 0 (máxima eficiencia teórica)."""
        thermo = ThermodynamicMetrics(
            system_temperature=500.0,
            reference_temperature=0.0,
        )
        assert thermo.carnot_efficiency == 1.0
    
    def test_exergetic_efficiency_typical(self):
        """η_ex = Ex / U."""
        thermo = ThermodynamicMetrics(
            internal_energy=100.0,
            entropy=0.5,
            reference_temperature=200.0,
            reference_entropy=0.2,
        )
        # Ex = 100 - 200*(0.5 - 0.2) = 100 - 60 = 40
        # η_ex = 40/100 = 0.4
        assert math.isclose(thermo.exergetic_efficiency, 0.4)
    
    def test_exergetic_efficiency_zero_energy(self):
        """η_ex = 1.0 si U = 0."""
        thermo = ThermodynamicMetrics(internal_energy=0.0)
        assert thermo.exergetic_efficiency == 1.0


class TestThermodynamicMetricsFactory:
    """Pruebas de métodos de fábrica."""
    
    def test_from_temperature_and_entropy_internal_energy(self):
        """U = C_v · T."""
        thermo = ThermodynamicMetrics.from_temperature_and_entropy(
            temperature=500.0,
            entropy=1.0,
            heat_capacity=3.0,
        )
        # U = 3 * 500 = 1500
        assert math.isclose(thermo.internal_energy, 1500.0)
    
    def test_from_temperature_and_entropy_preserves_params(self):
        """Parámetros pasados deben preservarse."""
        thermo = ThermodynamicMetrics.from_temperature_and_entropy(
            temperature=400.0,
            entropy=2.0,
            heat_capacity=1.5,
            reference_temperature=290.0,
        )
        assert thermo.system_temperature == 400.0
        assert thermo.entropy == 2.0
        assert thermo.heat_capacity == 1.5
        assert thermo.reference_temperature == 290.0
    
    def test_from_financial_analogy_mapping(self):
        """Mapeo de métricas financieras a termodinámicas."""
        thermo = ThermodynamicMetrics.from_financial_analogy(
            volatility=0.30,
            uncertainty=1.5,
            total_capital=1e6,
            market_temperature=0.20,
        )
        assert thermo.system_temperature == 0.30
        assert thermo.entropy == 1.5
        assert thermo.internal_energy == 1e6
        assert thermo.reference_temperature == 0.20


class TestThermodynamicMetricsDerivedProperties:
    """Pruebas de propiedades derivadas adicionales."""
    
    def test_specific_heat_ratio_typical(self):
        """γ = 1 + T/(C_v + T)."""
        thermo = ThermodynamicMetrics(
            system_temperature=300.0,
            heat_capacity=200.0,
        )
        # γ = 1 + 300/500 = 1.6
        assert math.isclose(thermo.specific_heat_ratio, 1.6)
    
    def test_specific_heat_ratio_zero_capacity(self):
        """γ = 1.0 si C_v = 0."""
        thermo = ThermodynamicMetrics(
            system_temperature=300.0,
            heat_capacity=0.0,
        )
        assert thermo.specific_heat_ratio == 1.0
    
    def test_thermal_diffusivity_typical(self):
        """α = T/C_v."""
        thermo = ThermodynamicMetrics(
            system_temperature=400.0,
            heat_capacity=100.0,
        )
        assert math.isclose(thermo.thermal_diffusivity, 4.0)
    
    def test_thermal_diffusivity_zero_capacity_infinite(self):
        """α = ∞ si C_v ≈ 0."""
        thermo = ThermodynamicMetrics(
            system_temperature=300.0,
            heat_capacity=0.0,
        )
        assert thermo.thermal_diffusivity == float('inf')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TESTS DE SYSTEM STATE VECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemStateVectorComposition:
    """Pruebas de composición del vector de estado."""
    
    def test_default_composition(self):
        """Estado por defecto usa métricas por defecto."""
        state = SystemStateVector()
        assert isinstance(state.physics, PhysicsMetrics)
        assert isinstance(state.topology, TopologicalMetrics)
        assert isinstance(state.control, ControlMetrics)
        assert isinstance(state.thermodynamics, ThermodynamicMetrics)
    
    def test_custom_composition(self, healthy_state: SystemStateVector):
        """Estado personalizado preserva componentes."""
        assert healthy_state.physics.saturation == 0.75
        assert healthy_state.topology.beta_0 == 1
        assert healthy_state.control.is_stable is True
        assert healthy_state.thermodynamics.system_temperature == 400.0
    
    def test_epoch_and_timestamp(self):
        """Metadatos temporales se preservan."""
        import time
        ts = time.time()
        state = SystemStateVector(timestamp=ts, epoch=42)
        assert state.timestamp == ts
        assert state.epoch == 42


class TestSystemStateVectorHealth:
    """Pruebas de evaluación de salud del sistema."""
    
    def test_healthy_state_all_criteria_met(self, healthy_state: SystemStateVector):
        """Estado saludable cumple todos los criterios."""
        assert healthy_state.is_healthy is True
    
    def test_unhealthy_unstable_control(
        self,
        energetic_physics: PhysicsMetrics,
        default_topology: TopologicalMetrics,
        unstable_control: ControlMetrics,
        default_thermo: ThermodynamicMetrics,
    ):
        """Sistema inestable → no saludable."""
        state = SystemStateVector(
            physics=energetic_physics,
            topology=default_topology,
            control=unstable_control,
            thermodynamics=default_thermo,
        )
        assert state.is_healthy is False
    
    def test_unhealthy_disconnected_topology(
        self,
        energetic_physics: PhysicsMetrics,
        stable_control: ControlMetrics,
        default_thermo: ThermodynamicMetrics,
    ):
        """Grafo disconexo → no saludable."""
        disconnected_topo = TopologicalMetrics(beta_0=3)
        state = SystemStateVector(
            physics=energetic_physics,
            topology=disconnected_topo,
            control=stable_control,
            thermodynamics=default_thermo,
        )
        assert state.is_healthy is False
    
    def test_unhealthy_negative_carnot(
        self,
        energetic_physics: PhysicsMetrics,
        default_topology: TopologicalMetrics,
        stable_control: ControlMetrics,
    ):
        """η_C ≤ 0 → no saludable."""
        cold_thermo = ThermodynamicMetrics(
            system_temperature=200.0,
            reference_temperature=300.0,  # T₀ > T → η_C < 0
        )
        state = SystemStateVector(
            physics=energetic_physics,
            topology=default_topology,
            control=stable_control,
            thermodynamics=cold_thermo,
        )
        assert state.is_healthy is False
    
    def test_health_vector_structure(self, healthy_state: SystemStateVector):
        """health_vector es tupla de 4 booleanos."""
        hv = healthy_state.health_vector
        assert isinstance(hv, tuple)
        assert len(hv) == 4
        assert all(isinstance(x, bool) for x in hv)


class TestSystemStateVectorSerialization:
    """Pruebas de serialización del estado."""
    
    def test_to_dict_keys(self, healthy_state: SystemStateVector):
        """to_dict contiene todas las claves esperadas."""
        d = healthy_state.to_dict()
        expected_keys = {
            'physics', 'topology', 'control', 'thermodynamics',
            'timestamp', 'epoch', 'is_healthy',
        }
        assert set(d.keys()) == expected_keys
    
    def test_to_dict_nested_structure(self, healthy_state: SystemStateVector):
        """Subespacios se serializan como diccionarios anidados."""
        d = healthy_state.to_dict()
        assert isinstance(d['physics'], dict)
        assert 'kinetic_energy' in d['physics']
        assert isinstance(d['topology'], dict)
        assert 'beta_0' in d['topology']
    
    def test_to_dict_preserves_values(self, healthy_state: SystemStateVector):
        """Valores se preservan en la serialización."""
        d = healthy_state.to_dict()
        assert d['physics']['saturation'] == 0.75
        assert d['epoch'] == 1
        assert d['is_healthy'] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TESTS DE CASOS LÍMITE NUMÉRICOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumericalEdgeCases:
    """Pruebas de comportamiento en casos límite numéricos."""
    
    def test_physics_near_zero_energy(self):
        """Energías cercanas a cero deben manejarse correctamente."""
        phys = PhysicsMetrics(
            kinetic_energy=1e-15,
            potential_energy=1e-15,
        )
        # No debe dividir por cero
        assert phys.potential_ratio >= 0.0
        assert phys.potential_ratio <= 1.0
    
    def test_physics_large_energy_values(self):
        """Energías grandes no deben causar overflow."""
        phys = PhysicsMetrics(
            kinetic_energy=1e15,
            potential_energy=1e15,
        )
        assert math.isfinite(phys.total_energy)
        assert math.isfinite(phys.potential_ratio)
    
    def test_control_very_small_damping(self):
        """ζ muy pequeño debe comportarse correctamente."""
        ctrl = ControlMetrics(damping_ratio=1e-10, natural_frequency=1.0)
        assert ctrl.is_underdamped is True
        assert math.isclose(ctrl.damped_frequency, 1.0, rel_tol=1e-9)
        # Overshoot cercano a 100%
        assert ctrl.peak_overshoot > 99.0
    
    def test_control_very_large_frequency(self):
        """ω_n grande no debe causar problemas."""
        ctrl = ControlMetrics(damping_ratio=0.5, natural_frequency=1e6)
        assert math.isfinite(ctrl.settling_time)
        assert ctrl.settling_time > 0
    
    def test_topology_large_betti_numbers(self):
        """Números de Betti grandes deben manejarse."""
        topo = TopologicalMetrics(beta_0=1000, beta_1=5000, beta_2=2000)
        assert topo.total_betti_number == 8000
        assert topo.euler_characteristic == -2000  # 1000 - 5000 + 2000
    
    def test_thermo_temperature_near_zero(self):
        """T cercano a 0 debe manejarse (tercer principio)."""
        thermo = ThermodynamicMetrics(
            system_temperature=1e-13,  # Menor que EPSILON_ABSOLUTE (1e-12)
            reference_temperature=300.0,
        )
        # η_C = 0 porque T ≈ 0 (T < EPSILON_ABSOLUTE)
        assert thermo.carnot_efficiency == 0.0
    
    def test_thermo_large_temperature_difference(self):
        """Gran diferencia T - T₀ debe manejarse."""
        thermo = ThermodynamicMetrics(
            system_temperature=1e6,
            reference_temperature=300.0,
        )
        # η_C ≈ 1
        assert thermo.carnot_efficiency > 0.999
    
    def test_inf_values_in_quality_factor(self):
        """Q = ∞ cuando P_dis = 0 es un valor válido."""
        phys = PhysicsMetrics(
            kinetic_energy=100.0,
            dissipated_power=0.0,
        )
        assert phys.quality_factor == float('inf')
        assert math.isinf(phys.quality_factor)


class TestNumericalStability:
    """Pruebas de estabilidad numérica."""
    
    def test_physics_ratio_sum_is_one(self):
        """ρ_p + ρ_k = 1 numéricamente."""
        for _ in range(100):
            import random
            ke = random.uniform(0, 1000)
            pe = random.uniform(0, 1000)
            phys = PhysicsMetrics(kinetic_energy=ke, potential_energy=pe)
            total = phys.potential_ratio + phys.kinetic_ratio
            assert math.isclose(total, 1.0, rel_tol=1e-9)
    
    def test_topology_euler_poincare_consistency(self):
        """P(-1) = χ numéricamente."""
        for b0 in range(5):
            for b1 in range(5):
                for b2 in range(5):
                    topo = TopologicalMetrics(beta_0=b0, beta_1=b1, beta_2=b2)
                    p_minus_one = topo.poincare_polynomial(-1.0)
                    assert p_minus_one == float(topo.euler_characteristic)
    
    def test_control_damped_frequency_discriminant(self):
        """ω_d² + σ² = ω_n² para sistemas subamortiguados."""
        for zeta in [0.1, 0.3, 0.5, 0.707, 0.9]:
            wn = 10.0
            ctrl = ControlMetrics(damping_ratio=zeta, natural_frequency=wn)
            wd = ctrl.damped_frequency
            sigma = zeta * wn
            reconstructed_wn = math.sqrt(wd**2 + sigma**2)
            assert math.isclose(reconstructed_wn, wn, rel_tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TESTS DE INMUTABILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestImmutability:
    """Pruebas de garantías de inmutabilidad (frozen dataclasses)."""
    
    def test_physics_immutable(self, energetic_physics: PhysicsMetrics):
        """PhysicsMetrics no permite modificación de atributos."""
        with pytest.raises(FrozenInstanceError):
            energetic_physics.saturation = 0.5  # type: ignore
    
    def test_topology_immutable(self, default_topology: TopologicalMetrics):
        """TopologicalMetrics no permite modificación de atributos."""
        with pytest.raises(FrozenInstanceError):
            default_topology.beta_0 = 5  # type: ignore
    
    def test_control_immutable(self, stable_control: ControlMetrics):
        """ControlMetrics no permite modificación de atributos."""
        with pytest.raises(FrozenInstanceError):
            stable_control.damping_ratio = 2.0  # type: ignore
    
    def test_thermo_immutable(self, default_thermo: ThermodynamicMetrics):
        """ThermodynamicMetrics no permite modificación de atributos."""
        with pytest.raises(FrozenInstanceError):
            default_thermo.entropy = 100.0  # type: ignore
    
    def test_state_vector_immutable(self, healthy_state: SystemStateVector):
        """SystemStateVector no permite modificación de atributos."""
        with pytest.raises(FrozenInstanceError):
            healthy_state.epoch = 999  # type: ignore
    
    def test_poles_tuple_is_immutable(self, stable_control: ControlMetrics):
        """poles_real es tupla inmutable."""
        with pytest.raises(TypeError):
            stable_control.poles_real[0] = 0.0  # type: ignore
    
    def test_betti_vector_is_immutable(self, complex_topology: TopologicalMetrics):
        """betti_vector es tupla inmutable."""
        bv = complex_topology.betti_vector
        with pytest.raises(TypeError):
            bv[0] = 999  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TESTS DE SERIALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestSerialization:
    """Pruebas de métodos to_dict() para serialización."""
    
    def test_physics_to_dict_complete(self, energetic_physics: PhysicsMetrics):
        """PhysicsMetrics.to_dict() contiene todos los campos."""
        d = energetic_physics.to_dict()
        expected_fields = {
            'saturation', 'pressure', 'kinetic_energy', 'potential_energy',
            'flyback_voltage', 'dissipated_power', 'effective_capacitance',
            'gyroscopic_stability', 'poynting_flux', 'hamiltonian_excess',
        }
        assert expected_fields.issubset(set(d.keys()))
    
    def test_topology_to_dict_complete(self, complex_topology: TopologicalMetrics):
        """TopologicalMetrics.to_dict() contiene todos los campos."""
        d = complex_topology.to_dict()
        expected_fields = {
            'beta_0', 'beta_1', 'beta_2', 'euler_characteristic',
            'fiedler_value', 'spectral_gap', 'pyramid_stability',
            'structural_entropy',
        }
        assert expected_fields.issubset(set(d.keys()))
    
    def test_control_to_dict_complete(self, stable_control: ControlMetrics):
        """ControlMetrics.to_dict() contiene todos los campos."""
        d = stable_control.to_dict()
        expected_fields = {
            'poles_real', 'is_stable', 'phase_margin_deg', 'gain_margin_db',
            'damping_ratio', 'natural_frequency', 'lyapunov_exponent',
        }
        assert expected_fields.issubset(set(d.keys()))
    
    def test_thermo_to_dict_complete(self, default_thermo: ThermodynamicMetrics):
        """ThermodynamicMetrics.to_dict() contiene todos los campos."""
        d = default_thermo.to_dict()
        expected_fields = {
            'system_temperature', 'entropy', 'internal_energy',
            'heat_capacity', 'pressure_volume', 'reference_temperature',
            'reference_entropy',
        }
        assert expected_fields.issubset(set(d.keys()))
    
    def test_serialization_roundtrip_physics(self, energetic_physics: PhysicsMetrics):
        """Serialización preserva valores para reconstrucción."""
        d = energetic_physics.to_dict()
        reconstructed = PhysicsMetrics(**d)
        assert reconstructed == energetic_physics
    
    def test_serialization_roundtrip_topology(self, complex_topology: TopologicalMetrics):
        """Serialización preserva valores para reconstrucción."""
        d = complex_topology.to_dict()
        reconstructed = TopologicalMetrics(**d)
        assert reconstructed == complex_topology
    
    def test_serialization_roundtrip_control(self, stable_control: ControlMetrics):
        """Serialización preserva valores para reconstrucción."""
        d = stable_control.to_dict()
        reconstructed = ControlMetrics(**d)
        assert reconstructed == stable_control
    
    def test_serialization_roundtrip_thermo(self, default_thermo: ThermodynamicMetrics):
        """Serialización preserva valores para reconstrucción."""
        d = default_thermo.to_dict()
        reconstructed = ThermodynamicMetrics(**d)
        assert reconstructed == default_thermo


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TESTS DE REPRESENTACIÓN (__str__)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStringRepresentation:
    """Pruebas de representación en cadena para depuración."""
    
    def test_physics_str_format(self, energetic_physics: PhysicsMetrics):
        """PhysicsMetrics.__str__() tiene formato esperado."""
        s = str(energetic_physics)
        assert "Physics(" in s
        assert "E=" in s
        assert "sat=" in s
    
    def test_topology_str_format(self, complex_topology: TopologicalMetrics):
        """TopologicalMetrics.__str__() tiene formato esperado."""
        s = str(complex_topology)
        assert "Topology(" in s
        assert "β=" in s
        assert "χ=" in s
    
    def test_control_str_stable_format(self, stable_control: ControlMetrics):
        """ControlMetrics.__str__() muestra STABLE."""
        s = str(stable_control)
        assert "Control(" in s
        assert "STABLE" in s
        assert "ζ=" in s
    
    def test_control_str_unstable_format(self, unstable_control: ControlMetrics):
        """ControlMetrics.__str__() muestra UNSTABLE."""
        s = str(unstable_control)
        assert "UNSTABLE" in s
    
    def test_control_str_chaotic_format(self):
        """ControlMetrics.__str__() muestra CHAOTIC cuando aplica."""
        ctrl = ControlMetrics(lyapunov_exponent=0.5)
        s = str(ctrl)
        assert "CHAOTIC" in s
    
    def test_thermo_str_format(self, default_thermo: ThermodynamicMetrics):
        """ThermodynamicMetrics.__str__() tiene formato esperado."""
        s = str(default_thermo)
        assert "Thermo(" in s
        assert "T=" in s
        assert "η_C=" in s
    
    def test_state_vector_str_multiline(self, healthy_state: SystemStateVector):
        """SystemStateVector.__str__() es multilínea."""
        s = str(healthy_state)
        lines = s.split('\n')
        assert len(lines) >= 4
        assert "SystemState" in s
        assert "HEALTHY" in s or "DEGRADED" in s


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TESTS DE INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Pruebas de integración entre componentes."""
    
    def test_full_pipeline_healthy_system(self):
        """Pipeline completo para sistema saludable."""
        # 1. Crear métricas físicas desde RLC
        phys = PhysicsMetrics.from_rlc_parameters(
            saturation=0.6, pressure=1.5,
            inductance=0.1, current=2.0,
            capacitance=0.01, voltage=50.0,
            resistance=5.0,
        )
        
        # 2. Crear topología de grafo conexo
        topo = TopologicalMetrics(
            beta_0=1, beta_1=2, beta_2=0,
            fiedler_value=0.8,
        )
        
        # 3. Crear control desde polos complejos
        ctrl = ControlMetrics.from_poles([
            complex(-2, 5),
            complex(-2, -5),
            complex(-10, 0),
        ])
        
        # 4. Crear termodinámica
        thermo = ThermodynamicMetrics.from_temperature_and_entropy(
            temperature=400.0,
            entropy=0.8,
            heat_capacity=2.5,
            reference_temperature=290.0,
        )
        
        # 5. Componer estado
        state = SystemStateVector(
            physics=phys,
            topology=topo,
            control=ctrl,
            thermodynamics=thermo,
            epoch=1,
        )
        
        # 6. Verificar salud
        assert state.is_healthy is True
        
        # 7. Serializar y verificar roundtrip
        state_dict = state.to_dict()
        assert state_dict['is_healthy'] is True
        
        # Reconstruir subespacios
        phys_reconstructed = PhysicsMetrics(**state_dict['physics'])
        assert phys_reconstructed.total_energy == phys.total_energy
    
    def test_degraded_system_detection(self):
        """Detección de sistema degradado."""
        # Sistema con múltiples problemas
        state = SystemStateVector(
            physics=PhysicsMetrics(
                kinetic_energy=0.1,  # Baja energía
                potential_energy=0.1,
                dissipated_power=10.0,  # Alta disipación → Q bajo
            ),
            topology=TopologicalMetrics(
                beta_0=3,  # Disconexo
                beta_1=5,  # Muchos ciclos
            ),
            control=ControlMetrics(
                poles_real=(0.1, -0.5),  # Inestable
                lyapunov_exponent=0.3,  # Caótico
            ),
            thermodynamics=ThermodynamicMetrics(
                system_temperature=100.0,
                reference_temperature=300.0,  # T < T₀
            ),
        )
        
        assert state.is_healthy is False
        
        hv = state.health_vector
        assert hv[0] is False  # Q < 1
        assert hv[1] is False  # No conexo
        assert hv[2] is False  # No estable
        assert hv[3] is False  # η_C < 0
    
    def test_mathematical_consistency_across_components(self):
        """Consistencia matemática entre componentes."""
        # Crear un sistema donde las propiedades derivadas
        # deben ser consistentes entre sí
        
        wn = 5.0
        zeta = 0.6
        
        ctrl = ControlMetrics(
            damping_ratio=zeta,
            natural_frequency=wn,
        )
        
        # Verificar relaciones matemáticas
        wd = ctrl.damped_frequency
        sigma = zeta * wn
        
        # ω_n² = ω_d² + σ²
        wn_squared_check = wd**2 + sigma**2
        assert math.isclose(wn_squared_check, wn**2, rel_tol=1e-9)
        
        # t_s ≈ 4/σ para subamortiguado
        ts = ctrl.settling_time
        ts_check = 4.0 / sigma
        assert math.isclose(ts, ts_check, rel_tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configuración de marcadores personalizados."""
    config.addinivalue_line(
        "markers", "slow: marca tests que toman más tiempo"
    )
    config.addinivalue_line(
        "markers", "numerical: tests de estabilidad numérica"
    )


if __name__ == "__main__":
    # Ejecutar con pytest
    pytest.main([__file__, "-v", "--tb=short"])