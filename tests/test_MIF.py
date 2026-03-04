"""
Suite de Pruebas para la Matriz de Interacción de Frontera (MIF) v2.0

Arquitectura de Tests:
    ├── TestPhysicalConstants      - Constantes físicas fundamentales
    ├── TestDataclassMetrics       - Estructuras de datos tipadas
    ├── TestMIFRegistry            - Registro y evaluación de co-vectores
    ├── TestLogisticalTurbulence   - Modelo RLC (Co-vector 1)
    ├── TestTopologicalPerturbation - Muerte de nodos (Co-vector 2)
    ├── TestThermalRadiation       - Radiación térmica (Co-vector 3)
    ├── TestCohomologicalAudit     - Auditoría institucional (Co-vector 4)
    ├── TestCascadeEvaluation      - Evaluación en cascada
    └── TestIntegration            - Tests de integración end-to-end

Propiedades verificadas:
    - Invariantes físicos (conservación de energía, leyes de Kirchhoff)
    - Desigualdades espectrales (Cheeger, Fiedler)
    - Estabilidad BIBO y márgenes de fase
    - Consistencia termodinámica (primera y segunda ley)
    - Correctitud topológica (números de Betti, conectividad)
"""

import math
import pytest
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Importaciones del módulo bajo prueba
from app.mif_module import (
    # Constantes y enumeraciones
    PHYS,
    PhysicalConstants,
    SeverityLevel,
    
    # Dataclasses
    RLCMetrics,
    TopologyMetrics,
    ThermalMetrics,
    AuditFinding,
    EvaluationRecord,
    
    # Registry
    MIFRegistry,
    
    # Co-vectores
    inject_logistical_turbulence,
    perturb_topological_manifold,
    apply_thermal_radiation,
    audit_cohomological_isomorphism,
    
    # Funciones auxiliares
    _classify_damping,
    _success,
    _physics_error,
    _topology_error,
    _logic_error,
    
    # Inicialización
    initialize_mif,
)

from app.schemas import Stratum
from app.adapters.mic_vectors import VectorResultStatus


# =============================================================================
# FIXTURES GLOBALES
# =============================================================================

@pytest.fixture
def empty_context() -> Dict[str, Any]:
    """Contexto vacío para pruebas básicas."""
    return {}


@pytest.fixture
def standard_context() -> Dict[str, Any]:
    """Contexto estándar con parámetros típicos."""
    return {
        "viscoelastic_resistance": 10.0,
        "viscoelastic_capacitance": 0.001,
        "critical_voltage_threshold": 100.0,
        "critical_energy_ratio": 10.0,
        "max_quality_factor": 50.0,
        "fiedler_threshold": 0.1,
        "pyramid_stability_threshold": 0.6,
        "survival_ratio_threshold": 0.5,
        "max_connected_components": 1,
        "base_system_temperature": 298.0,
        "ambient_temperature": 298.0,
        "fever_threshold": 323.0,
        "critical_threshold": 373.0,
        "cooling_coefficient": 0.5,
        "radiative_area": 1.0,
        "financial_emissivity": 0.1,
        "stability_margin": 1e-6,
        "min_pole_margin": 0.1,
    }


@pytest.fixture
def simple_digraph() -> nx.DiGraph:
    """Grafo dirigido simple para pruebas básicas."""
    G = nx.DiGraph()
    G.add_nodes_from([
        ("A", {"type": "INSUMO"}),
        ("B", {"type": "INSUMO"}),
        ("C", {"type": "APU"}),
        ("D", {"type": "APU"}),
        ("E", {"type": "OUTPUT"}),
    ])
    G.add_edges_from([
        ("A", "C"), ("B", "C"),
        ("C", "D"), ("D", "E"),
    ])
    return G


@pytest.fixture
def fragile_digraph() -> nx.DiGraph:
    """Grafo con un nodo crítico (cuello de botella)."""
    G = nx.DiGraph()
    G.add_nodes_from([
        ("S1", {"type": "INSUMO"}),
        ("S2", {"type": "INSUMO"}),
        ("S3", {"type": "INSUMO"}),
        ("HUB", {"type": "APU"}),  # Nodo crítico
        ("D1", {"type": "OUTPUT"}),
        ("D2", {"type": "OUTPUT"}),
    ])
    G.add_edges_from([
        ("S1", "HUB"), ("S2", "HUB"), ("S3", "HUB"),
        ("HUB", "D1"), ("HUB", "D2"),
    ])
    return G


@pytest.fixture
def cyclic_digraph() -> nx.DiGraph:
    """Grafo con ciclos (β₁ > 0)."""
    G = nx.DiGraph()
    G.add_nodes_from([
        ("A", {"type": "INSUMO"}),
        ("B", {"type": "APU"}),
        ("C", {"type": "APU"}),
        ("D", {"type": "OUTPUT"}),
    ])
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "D"),
        ("C", "B"),  # Ciclo B ↔ C
    ])
    return G


@pytest.fixture
def valid_telemetry_passport() -> Dict[str, Any]:
    """Pasaporte de telemetría válido (sistema estable, acíclico)."""
    return {
        "laplace_poles": [-1.0 + 0j, -2.0 + 1j, -2.0 - 1j, -5.0 + 0j],
        "betti_1": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@pytest.fixture
def unstable_telemetry_passport() -> Dict[str, Any]:
    """Pasaporte con polos inestables."""
    return {
        "laplace_poles": [0.5 + 2j, 0.5 - 2j, -1.0],  # Polos en RHP
        "betti_1": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@pytest.fixture
def mif_registry() -> MIFRegistry:
    """Registro MIF inicializado."""
    return initialize_mif()


# =============================================================================
# TESTS: CONSTANTES FÍSICAS
# =============================================================================

class TestPhysicalConstants:
    """Verificación de constantes físicas fundamentales."""

    def test_stefan_boltzmann_constant_value(self):
        """σ debe ser 5.670374419×10⁻⁸ W/(m²·K⁴)."""
        expected = 5.670374419e-8
        assert abs(PHYS.STEFAN_BOLTZMANN - expected) < 1e-15

    def test_boltzmann_constant_value(self):
        """k_B debe ser 1.380649×10⁻²³ J/K."""
        expected = 1.380649e-23
        assert abs(PHYS.BOLTZMANN - expected) < 1e-30

    def test_planck_constant_value(self):
        """h debe ser 6.62607015×10⁻³⁴ J·s."""
        expected = 6.62607015e-34
        assert abs(PHYS.PLANCK - expected) < 1e-42

    def test_constants_are_immutable(self):
        """Las constantes deben ser inmutables (frozen dataclass)."""
        with pytest.raises(AttributeError):
            PHYS.STEFAN_BOLTZMANN = 0.0

    def test_epsilon_is_positive_and_small(self):
        """ε debe ser positivo y suficientemente pequeño."""
        assert PHYS.EPSILON > 0
        assert PHYS.EPSILON < 1e-6


# =============================================================================
# TESTS: DATACLASSES DE MÉTRICAS
# =============================================================================

class TestRLCMetrics:
    """Tests para la estructura RLCMetrics."""

    def test_creation_with_valid_values(self):
        """Debe crear métricas RLC con valores válidos."""
        metrics = RLCMetrics(
            v_flyback_volts=50.0,
            stored_energy_joules=0.5,
            power_dissipated_watts=250.0,
            damping_ratio_zeta=0.707,
            absorption_capacity_joules=1.25,
            resonant_frequency_hz=159.15,
            quality_factor=0.707,
            bandwidth_hz=225.0,
            time_constant_seconds=0.01,
            severity=SeverityLevel.NOMINAL,
        )
        assert metrics.v_flyback_volts == 50.0
        assert metrics.severity == SeverityLevel.NOMINAL

    def test_to_dict_returns_valid_structure(self):
        """to_dict() debe retornar diccionario con claves esperadas."""
        metrics = RLCMetrics(
            v_flyback_volts=100.0,
            stored_energy_joules=1.0,
            power_dissipated_watts=1000.0,
            damping_ratio_zeta=1.0,
            absorption_capacity_joules=5.0,
            resonant_frequency_hz=100.0,
            quality_factor=0.5,
            bandwidth_hz=200.0,
            time_constant_seconds=0.01,
            severity=SeverityLevel.WARNING,
        )
        d = metrics.to_dict()
        
        assert "v_flyback_volts" in d
        assert "severity" in d
        assert d["severity"] == "WARNING"
        assert isinstance(d["v_flyback_volts"], float)

    def test_to_dict_rounds_values(self):
        """to_dict() debe redondear valores apropiadamente."""
        metrics = RLCMetrics(
            v_flyback_volts=100.123456789,
            stored_energy_joules=1.0,
            power_dissipated_watts=1000.0,
            damping_ratio_zeta=1.0,
            absorption_capacity_joules=5.0,
            resonant_frequency_hz=100.0,
            quality_factor=0.5,
            bandwidth_hz=200.0,
            time_constant_seconds=0.01,
            severity=SeverityLevel.NOMINAL,
        )
        d = metrics.to_dict()
        
        # Debe estar redondeado a 6 decimales
        assert d["v_flyback_volts"] == 100.123457


class TestTopologyMetrics:
    """Tests para la estructura TopologyMetrics."""

    def test_creation_with_graph_properties(self):
        """Debe almacenar propiedades topológicas correctamente."""
        metrics = TopologyMetrics(
            fiedler_value=0.5,
            cheeger_lower_bound=0.25,
            cheeger_upper_bound=1.0,
            cheeger_estimate=0.5,
            pyramid_stability=0.8,
            betti_0=1,
            betti_1=0,
            surviving_nodes=10,
            survival_ratio=0.9,
            nodes_killed=1,
            graph_diameter=4,
            graph_radius=2,
            max_betweenness=0.3,
            avg_clustering=0.6,
            base_width=5,
            structure_load=3,
        )
        assert metrics.betti_0 == 1
        assert metrics.betti_1 == 0
        assert metrics.graph_diameter == 4

    def test_to_dict_handles_none_values(self):
        """to_dict() debe manejar valores None correctamente."""
        metrics = TopologyMetrics(
            fiedler_value=0.0,
            cheeger_lower_bound=0.0,
            cheeger_upper_bound=0.0,
            cheeger_estimate=0.0,
            pyramid_stability=0.0,
            betti_0=3,
            betti_1=0,
            surviving_nodes=3,
            survival_ratio=0.3,
            nodes_killed=7,
            graph_diameter=None,  # Grafo desconectado
            graph_radius=None,
            max_betweenness=0.0,
            avg_clustering=0.0,
            base_width=0,
            structure_load=0,
        )
        d = metrics.to_dict()
        
        assert d["graph_diameter"] is None
        assert d["graph_radius"] is None


class TestAuditFinding:
    """Tests para hallazgos de auditoría."""

    def test_finding_creation(self):
        """Debe crear hallazgo con todos los campos."""
        finding = AuditFinding(
            criterion="BIBO_STABILITY",
            passed=True,
            severity="NOMINAL",
            detail="Sistema estable.",
        )
        assert finding.criterion == "BIBO_STABILITY"
        assert finding.passed is True
        assert finding.timestamp is not None

    def test_finding_to_dict(self):
        """to_dict() debe incluir timestamp."""
        finding = AuditFinding(
            criterion="TEST",
            passed=False,
            severity="CRITICAL",
            detail="Fallo de prueba.",
        )
        d = finding.to_dict()
        
        assert "timestamp" in d
        assert d["passed"] is False


# =============================================================================
# TESTS: MIF REGISTRY
# =============================================================================

class TestMIFRegistryCreation:
    """Tests de creación e inicialización del registro."""

    def test_empty_registry_creation(self):
        """Debe crear registro vacío sin errores."""
        registry = MIFRegistry()
        assert len(registry.registered_names) == 0
        assert len(registry.evaluation_log) == 0

    def test_initialize_mif_creates_four_covectors(self):
        """initialize_mif() debe registrar exactamente 4 co-vectores."""
        mif = initialize_mif()
        assert len(mif.registered_names) == 4

    def test_initialize_mif_registers_correct_names(self):
        """Debe registrar los nombres correctos."""
        mif = initialize_mif()
        expected_names = {
            "turbulence_shock",
            "chaos_node_death",
            "inflationary_radiation",
            "institutional_audit",
        }
        assert set(mif.registered_names) == expected_names

    def test_initialize_mif_assigns_correct_strata(self):
        """Debe asignar estratos correctos a cada co-vector."""
        mif = initialize_mif()
        strata = mif.registered_strata
        
        assert strata["turbulence_shock"] == "PHYSICS"
        assert strata["chaos_node_death"] == "TACTICS"
        assert strata["inflationary_radiation"] == "STRATEGY"
        assert strata["institutional_audit"] == "WISDOM"


class TestMIFRegistryRegistration:
    """Tests de registro de co-vectores."""

    def test_register_valid_covector(self):
        """Debe registrar co-vector válido sin errores."""
        registry = MIFRegistry()
        handler = lambda **kwargs: {"success": True}
        
        registry.register_covector("test_covector", Stratum.PHYSICS, handler)
        
        assert "test_covector" in registry.registered_names

    def test_register_with_empty_name_raises(self):
        """Nombre vacío debe lanzar ValueError."""
        registry = MIFRegistry()
        handler = lambda **kwargs: {"success": True}
        
        with pytest.raises(ValueError, match="no puede ser vacío"):
            registry.register_covector("", Stratum.PHYSICS, handler)

    def test_register_with_whitespace_name_raises(self):
        """Nombre solo espacios debe lanzar ValueError."""
        registry = MIFRegistry()
        handler = lambda **kwargs: {"success": True}
        
        with pytest.raises(ValueError, match="no puede ser vacío"):
            registry.register_covector("   ", Stratum.PHYSICS, handler)

    def test_register_with_non_string_name_raises(self):
        """Nombre no-string debe lanzar TypeError."""
        registry = MIFRegistry()
        handler = lambda **kwargs: {"success": True}
        
        with pytest.raises(TypeError, match="debe ser str"):
            registry.register_covector(123, Stratum.PHYSICS, handler)

    def test_register_with_non_callable_handler_raises(self):
        """Handler no-callable debe lanzar TypeError."""
        registry = MIFRegistry()
        
        with pytest.raises(TypeError, match="debe ser callable"):
            registry.register_covector("test", Stratum.PHYSICS, "not_a_function")

    def test_register_with_invalid_stratum_raises(self):
        """Stratum inválido debe lanzar TypeError."""
        registry = MIFRegistry()
        handler = lambda **kwargs: {"success": True}
        
        with pytest.raises(TypeError, match="debe ser Stratum"):
            registry.register_covector("test", "PHYSICS", handler)

    def test_re_registration_warns_and_overwrites(self):
        """Re-registrar debe advertir y sobrescribir."""
        registry = MIFRegistry()
        handler1 = lambda **kwargs: {"success": True, "version": 1}
        handler2 = lambda **kwargs: {"success": True, "version": 2}
        
        registry.register_covector("test", Stratum.PHYSICS, handler1)
        
        with pytest.warns(match=None):  # Puede o no advertir según logging
            registry.register_covector("test", Stratum.TACTICS, handler2)
        
        # Debe haber sobrescrito
        result = registry.evaluate_impact("test", {}, {"context": True})
        assert result.get("version") == 2

    def test_unregister_existing_covector(self):
        """Debe eliminar co-vector existente."""
        registry = MIFRegistry()
        handler = lambda **kwargs: {"success": True}
        
        registry.register_covector("test", Stratum.PHYSICS, handler)
        assert "test" in registry.registered_names
        
        result = registry.unregister_covector("test")
        assert result is True
        assert "test" not in registry.registered_names

    def test_unregister_nonexistent_returns_false(self):
        """Eliminar co-vector inexistente debe retornar False."""
        registry = MIFRegistry()
        result = registry.unregister_covector("nonexistent")
        assert result is False


class TestMIFRegistryEvaluation:
    """Tests de evaluación de co-vectores."""

    def test_evaluate_unknown_covector_raises(self):
        """Evaluar co-vector desconocido debe lanzar ValueError."""
        registry = MIFRegistry()
        
        with pytest.raises(ValueError, match="desconocido"):
            registry.evaluate_impact("unknown", {}, {})

    def test_evaluate_returns_success_structure(self, mif_registry, standard_context):
        """Evaluación exitosa debe tener estructura correcta."""
        result = mif_registry.evaluate_impact(
            "turbulence_shock",
            {"inductance_L": 0.01, "target_current_drop": 1.0},
            standard_context,
        )
        
        assert "success" in result
        assert "_mif_stratum" in result
        assert "_mif_covector" in result

    def test_evaluate_logs_evaluation(self, mif_registry, standard_context):
        """Cada evaluación debe registrarse en el log."""
        initial_log_len = len(mif_registry.evaluation_log)
        
        mif_registry.evaluate_impact(
            "turbulence_shock",
            {"inductance_L": 0.01, "target_current_drop": 1.0},
            standard_context,
        )
        
        assert len(mif_registry.evaluation_log) == initial_log_len + 1
        
        last_entry = mif_registry.evaluation_log[-1]
        assert last_entry["covector"] == "turbulence_shock"
        assert "timestamp" in last_entry
        assert "duration_ms" in last_entry

    def test_evaluate_with_incompatible_signature_returns_error(self):
        """Handler con firma incompatible debe retornar error."""
        registry = MIFRegistry()
        
        # Handler que espera argumentos específicos
        def strict_handler(required_arg: str, context: Dict) -> Dict:
            return {"success": True}
        
        registry.register_covector("strict", Stratum.PHYSICS, strict_handler)
        
        # Payload sin el argumento requerido
        result = registry.evaluate_impact("strict", {}, {})
        
        assert result["success"] is False
        assert "Firma incompatible" in result.get("error", "")

    def test_evaluate_with_handler_exception_returns_error(self):
        """Handler que lanza excepción debe retornar error."""
        registry = MIFRegistry()
        
        def failing_handler(**kwargs) -> Dict:
            raise RuntimeError("Error intencional")
        
        registry.register_covector("failing", Stratum.PHYSICS, failing_handler)
        
        result = registry.evaluate_impact("failing", {}, {})
        
        assert result["success"] is False
        assert "RuntimeError" in result.get("error", "")


# =============================================================================
# TESTS: CO-VECTOR 1 — TURBULENCIA LOGÍSTICA (RLC)
# =============================================================================

class TestLogisticalTurbulenceValidation:
    """Tests de validación de entradas para modelo RLC."""

    def test_negative_inductance_returns_error(self, empty_context):
        """Inductancia negativa debe fallar."""
        result = inject_logistical_turbulence(
            inductance_L=-0.1,
            target_current_drop=1.0,
            context=empty_context,
        )
        
        assert result["success"] is False
        assert result["status"] == VectorResultStatus.PHYSICS_ERROR.value
        assert "positiva" in result["error"].lower()

    def test_zero_inductance_returns_error(self, empty_context):
        """Inductancia cero debe fallar."""
        result = inject_logistical_turbulence(
            inductance_L=0.0,
            target_current_drop=1.0,
            context=empty_context,
        )
        
        assert result["success"] is False

    def test_non_numeric_inductance_returns_error(self, empty_context):
        """Inductancia no numérica debe fallar."""
        result = inject_logistical_turbulence(
            inductance_L="invalid",
            target_current_drop=1.0,
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "numérica" in result["error"].lower()

    def test_zero_current_drop_returns_success(self, empty_context):
        """Sin cambio de corriente, sistema estacionario."""
        result = inject_logistical_turbulence(
            inductance_L=0.1,
            target_current_drop=0.0,
            context=empty_context,
        )
        
        assert result["success"] is True
        assert result["metrics"]["v_flyback_volts"] == 0.0

    def test_negative_resistance_returns_error(self, empty_context):
        """Resistencia negativa debe fallar."""
        context = {**empty_context, "viscoelastic_resistance": -10.0}
        
        result = inject_logistical_turbulence(
            inductance_L=0.1,
            target_current_drop=5.0,
            context=context,
        )
        
        assert result["success"] is False
        assert "Resistencia" in result["error"]


class TestLogisticalTurbulencePhysics:
    """Tests de física del modelo RLC."""

    def test_flyback_voltage_calculation(self, standard_context):
        """V_flyback = L × |di/dt| debe ser correcto."""
        L = 0.1  # 100 mH
        di_dt = 10.0  # 10 A/s
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=di_dt,
            context=standard_context,
        )
        
        expected_v = L * abs(di_dt)  # 1.0 V
        assert abs(result["metrics"]["v_flyback_volts"] - expected_v) < 1e-6

    def test_stored_energy_calculation(self, standard_context):
        """E = ½ × L × I² debe ser correcto."""
        L = 0.1  # 100 mH
        I = 5.0  # 5 A
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=I,
            context=standard_context,
        )
        
        expected_E = 0.5 * L * I ** 2  # 1.25 J
        assert abs(result["metrics"]["stored_energy_joules"] - expected_E) < 1e-6

    def test_damping_ratio_calculation(self):
        """ζ = R / (2√(L/C)) debe ser correcto."""
        L = 0.1   # 100 mH
        R = 10.0  # 10 Ω
        C = 0.001 # 1 mF
        
        context = {
            "viscoelastic_resistance": R,
            "viscoelastic_capacitance": C,
            "critical_voltage_threshold": 1000.0,  # Alto para evitar fallo
        }
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=1.0,
            context=context,
        )
        
        # ζ = R / (2 × √(L/C)) = 10 / (2 × √(100)) = 10 / 20 = 0.5
        expected_zeta = R / (2.0 * math.sqrt(L / C))
        assert abs(result["metrics"]["damping_ratio_zeta"] - expected_zeta) < 1e-6

    def test_resonant_frequency_calculation(self):
        """f₀ = 1/(2π√(LC)) debe ser correcto."""
        L = 0.1   # 100 mH
        C = 0.001 # 1 mF
        
        context = {
            "viscoelastic_resistance": 10.0,
            "viscoelastic_capacitance": C,
            "critical_voltage_threshold": 1000.0,
        }
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=1.0,
            context=context,
        )
        
        # f₀ = 1 / (2π√(LC)) = 1 / (2π × √(0.0001)) = 1 / (2π × 0.01) ≈ 15.915 Hz
        expected_f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        assert abs(result["metrics"]["resonant_frequency_hz"] - expected_f0) < 0.001

    def test_quality_factor_relationship(self):
        """Q = 1/(2ζ) debe cumplirse."""
        L = 0.1
        R = 5.0
        C = 0.01
        
        context = {
            "viscoelastic_resistance": R,
            "viscoelastic_capacitance": C,
            "critical_voltage_threshold": 1000.0,
        }
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=1.0,
            context=context,
        )
        
        zeta = result["metrics"]["damping_ratio_zeta"]
        Q = result["metrics"]["quality_factor"]
        
        if zeta > 0 and Q > 0:
            assert abs(Q - 1.0 / (2.0 * zeta)) < 1e-4

    def test_bandwidth_calculation(self):
        """BW = R/(2πL) debe ser correcto."""
        L = 0.1
        R = 10.0
        
        context = {
            "viscoelastic_resistance": R,
            "viscoelastic_capacitance": 0.001,
            "critical_voltage_threshold": 1000.0,
        }
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=1.0,
            context=context,
        )
        
        expected_bw = R / (2.0 * math.pi * L)  # ≈ 15.915 Hz
        assert abs(result["metrics"]["bandwidth_hz"] - expected_bw) < 0.001

    def test_time_constant_calculation(self):
        """τ = L/R debe ser correcto."""
        L = 0.5
        R = 10.0
        
        context = {
            "viscoelastic_resistance": R,
            "viscoelastic_capacitance": 0.001,
            "critical_voltage_threshold": 1000.0,
        }
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=1.0,
            context=context,
        )
        
        expected_tau = L / R  # 0.05 s
        assert abs(result["metrics"]["time_constant_seconds"] - expected_tau) < 1e-6


class TestLogisticalTurbulenceSeverity:
    """Tests de clasificación de severidad."""

    def test_nominal_severity_on_low_stress(self, standard_context):
        """Baja perturbación debe dar severidad NOMINAL."""
        result = inject_logistical_turbulence(
            inductance_L=0.001,  # 1 mH (pequeña)
            target_current_drop=0.1,  # Pequeño cambio
            context=standard_context,
        )
        
        assert result["success"] is True
        assert result["metrics"]["severity"] == "NOMINAL"

    def test_critical_severity_on_voltage_breach(self, empty_context):
        """Superar V_breakdown debe dar severidad CRITICAL."""
        context = {
            **empty_context,
            "viscoelastic_resistance": 1.0,
            "viscoelastic_capacitance": 0.001,
            "critical_voltage_threshold": 10.0,  # Umbral bajo
        }
        
        result = inject_logistical_turbulence(
            inductance_L=1.0,  # 1 H (grande)
            target_current_drop=100.0,  # V_flyback = 100 V >> 10 V
            context=context,
        )
        
        assert result["success"] is False
        assert "CRITICAL" in result["metrics"]["severity"] or "CATASTROPHIC" in result["metrics"]["severity"]

    def test_catastrophic_severity_on_multiple_breaches(self, empty_context):
        """Múltiples brechas deben dar CATASTROPHIC."""
        context = {
            **empty_context,
            "viscoelastic_resistance": 0.1,  # Baja resistencia
            "viscoelastic_capacitance": 1e-6,  # Muy poca capacitancia
            "critical_voltage_threshold": 10.0,
            "critical_energy_ratio": 0.1,  # Umbral muy bajo
        }
        
        result = inject_logistical_turbulence(
            inductance_L=1.0,
            target_current_drop=100.0,
            context=context,
        )
        
        assert result["success"] is False
        assert result["metrics"]["severity"] == "CATASTROPHIC"


class TestDampingClassification:
    """Tests para la función _classify_damping."""

    @pytest.mark.parametrize("zeta,expected", [
        (0.1, "muy subamortiguado"),
        (0.4, "muy subamortiguado"),
        (0.5, "subamortiguado"),
        (0.7, "subamortiguado"),
        (0.99, "subamortiguado"),
        (1.0, "crítico"),
        (1.005, "crítico"),
        (1.5, "sobreamortiguado"),
        (2.5, "muy sobreamortiguado"),
        (float('inf'), "circuito RL puro"),
    ])
    def test_damping_classification(self, zeta, expected):
        """Debe clasificar correctamente el amortiguamiento."""
        result = _classify_damping(zeta)
        assert result == expected


# =============================================================================
# TESTS: CO-VECTOR 2 — MUERTE DE NODOS (TOPOLOGÍA)
# =============================================================================

class TestTopologicalPerturbationValidation:
    """Tests de validación de entradas para perturbación topológica."""

    def test_non_digraph_returns_error(self, empty_context):
        """Tipo de grafo incorrecto debe fallar."""
        G = nx.Graph()  # No es DiGraph
        G.add_edge("A", "B")
        
        result = perturb_topological_manifold(
            graph=G,
            nodes_to_kill=["A"],
            context=empty_context,
        )
        
        assert result["success"] is False
        assert result["status"] == VectorResultStatus.TOPOLOGY_ERROR.value
        assert "DiGraph" in result["error"]

    def test_empty_graph_returns_error(self, empty_context):
        """Grafo vacío debe fallar."""
        G = nx.DiGraph()
        
        result = perturb_topological_manifold(
            graph=G,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "vacío" in result["error"].lower()

    def test_invalid_nodes_to_kill_type_returns_error(self, simple_digraph, empty_context):
        """nodes_to_kill debe ser lista/tupla/set."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill="A",  # String, no lista
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "lista" in result["error"].lower()

    def test_phantom_nodes_are_filtered(self, simple_digraph, empty_context):
        """Nodos inexistentes deben filtrarse con advertencia."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=["A", "PHANTOM", "GHOST"],
            context=empty_context,
        )
        
        # Debe haber eliminado solo "A"
        assert result["metrics"]["nodes_killed"] == 1


class TestTopologicalPerturbationMetrics:
    """Tests de métricas topológicas."""

    def test_survival_ratio_calculation(self, simple_digraph, empty_context):
        """ρ = n_surviving / n_original debe ser correcto."""
        n_original = simple_digraph.number_of_nodes()  # 5
        
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=["A"],
            context=empty_context,
        )
        
        expected_ratio = (n_original - 1) / n_original  # 4/5 = 0.8
        assert abs(result["metrics"]["survival_ratio"] - expected_ratio) < 1e-4

    def test_betti_0_counts_components(self, simple_digraph, empty_context):
        """β₀ debe contar componentes conexas."""
        # Sin matar nodos, debe haber 1 componente
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        assert result["metrics"]["betti_0"] == 1

    def test_betti_1_cyclomatic_number(self, cyclic_digraph, empty_context):
        """β₁ = m - n + c debe contar ciclos independientes."""
        result = perturb_topological_manifold(
            graph=cyclic_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        # El grafo tiene un ciclo B↔C
        assert result["metrics"]["betti_1"] >= 1

    def test_fiedler_value_positive_for_connected(self, simple_digraph, empty_context):
        """λ₂ debe ser > 0 para grafo conexo."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        if result["metrics"]["betti_0"] == 1:
            assert result["metrics"]["fiedler_value"] > 0

    def test_cheeger_bounds_satisfy_inequality(self, simple_digraph, empty_context):
        """λ₂/2 ≤ h(G) ≤ √(2×d_max×λ₂) debe cumplirse."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        metrics = result["metrics"]
        
        if metrics["betti_0"] == 1:
            # La estimación debe estar entre las cotas
            assert metrics["cheeger_lower_bound"] <= metrics["cheeger_estimate"]
            assert metrics["cheeger_estimate"] <= metrics["cheeger_upper_bound"]

    def test_pyramid_stability_formula(self, simple_digraph, empty_context):
        """Ψ = tanh(base/carga) debe ser correcto."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        base = result["metrics"]["base_width"]  # 2 INSUMOs
        load = result["metrics"]["structure_load"]  # 2 APUs
        
        if load > 0:
            expected_psi = math.tanh(base / load)
            assert abs(result["metrics"]["pyramid_stability"] - expected_psi) < 1e-6


class TestTopologicalPerturbationFailures:
    """Tests de condiciones de fallo."""

    def test_total_extinction_fails(self, simple_digraph, empty_context):
        """Eliminar todos los nodos debe fallar."""
        all_nodes = list(simple_digraph.nodes())
        
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=all_nodes,
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "total" in result["error"].lower() or "extinción" in result["error"].lower()

    def test_single_node_survival_fails(self, simple_digraph, empty_context):
        """Solo 1 nodo sobreviviente debe fallar."""
        all_but_one = list(simple_digraph.nodes())[:-1]
        
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=all_but_one,
            context=empty_context,
        )
        
        assert result["success"] is False

    def test_fragmentation_detection(self, fragile_digraph, empty_context):
        """Matar el HUB debe fragmentar el grafo."""
        result = perturb_topological_manifold(
            graph=fragile_digraph,
            nodes_to_kill=["HUB"],
            context=empty_context,
        )
        
        # Sin HUB, el grafo se fragmenta en 5 componentes (cada nodo aislado)
        assert result["metrics"]["betti_0"] > 1
        assert result["success"] is False
        assert "FRAGMENTACIÓN" in result["error"]

    def test_low_survival_ratio_fails(self):
        """Bajo ratio de supervivencia debe fallar."""
        G = nx.DiGraph()
        for i in range(20):
            G.add_node(f"N{i}", type="INSUMO")
        for i in range(19):
            G.add_edge(f"N{i}", f"N{i+1}")
        
        # Matar 15 de 20 nodos (supervivencia = 25%)
        to_kill = [f"N{i}" for i in range(15)]
        
        result = perturb_topological_manifold(
            graph=G,
            nodes_to_kill=to_kill,
            context={"survival_ratio_threshold": 0.5},
        )
        
        assert result["success"] is False
        assert "EXTINCIÓN" in result["error"]


# =============================================================================
# TESTS: CO-VECTOR 3 — RADIACIÓN TÉRMICA
# =============================================================================

class TestThermalRadiationValidation:
    """Tests de validación de entradas para modelo térmico."""

    def test_negative_liquidity_returns_error(self, empty_context):
        """Liquidez negativa debe fallar."""
        result = apply_thermal_radiation(
            heat_shock_Q=100.0,
            liquidity_L=-1.0,
            fixed_contracts_F=1.0,
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "negativa" in result["error"].lower()

    def test_negative_contracts_returns_error(self, empty_context):
        """Contratos negativos debe fallar."""
        result = apply_thermal_radiation(
            heat_shock_Q=100.0,
            liquidity_L=1.0,
            fixed_contracts_F=-1.0,
            context=empty_context,
        )
        
        assert result["success"] is False

    def test_non_numeric_inputs_return_error(self, empty_context):
        """Entradas no numéricas deben fallar."""
        result = apply_thermal_radiation(
            heat_shock_Q="hot",
            liquidity_L=1.0,
            fixed_contracts_F=1.0,
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "numérico" in result["error"].lower()

    def test_zero_inertia_with_shock_fails(self, empty_context):
        """Inercia cero con choque térmico debe fallar."""
        result = apply_thermal_radiation(
            heat_shock_Q=100.0,
            liquidity_L=0.0,
            fixed_contracts_F=0.0,
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "HIPOTERMIA" in result["error"]


class TestThermalRadiationPhysics:
    """Tests de física termodinámica."""

    def test_delta_T_calculation(self, empty_context):
        """ΔT = Q/I debe ser correcto."""
        Q = 100.0
        L = 10.0
        F = 5.0
        I = L * F  # 50.0
        
        result = apply_thermal_radiation(
            heat_shock_Q=Q,
            liquidity_L=L,
            fixed_contracts_F=F,
            context=empty_context,
        )
        
        expected_delta_T = Q / I  # 2.0 K
        assert abs(result["metrics"]["delta_T_raw_K"] - expected_delta_T) < 1e-4

    def test_newtonian_cooling_reduces_temperature(self):
        """Enfriamiento newtoniano debe reducir ΔT efectivo."""
        context = {
            "cooling_coefficient": 1.0,
            "radiative_area": 1.0,
            "base_system_temperature": 300.0,
            "ambient_temperature": 298.0,
        }
        
        result = apply_thermal_radiation(
            heat_shock_Q=50.0,
            liquidity_L=5.0,
            fixed_contracts_F=5.0,
            context=context,
        )
        
        # Con enfriamiento, ΔT_eff < ΔT_raw
        assert result["metrics"]["delta_T_effective_K"] < result["metrics"]["delta_T_raw_K"]

    def test_stefan_boltzmann_radiation(self):
        """Pérdida radiativa debe seguir σεA(T⁴-T∞⁴)."""
        context = {
            "base_system_temperature": 350.0,  # Sistema caliente
            "ambient_temperature": 298.0,
            "financial_emissivity": 0.5,
            "radiative_area": 2.0,
            "cooling_coefficient": 0.0,  # Sin enfriamiento convectivo
        }
        
        result = apply_thermal_radiation(
            heat_shock_Q=100.0,
            liquidity_L=10.0,
            fixed_contracts_F=10.0,
            context=context,
        )
        
        # Debe haber pérdida radiativa no nula
        assert result["metrics"]["radiative_loss_W"] > 0

    def test_time_constant_calculation(self):
        """τ = I/(h×A) debe ser correcto."""
        h = 2.0
        A = 3.0
        L = 12.0
        F = 5.0
        I = L * F  # 60.0
        
        context = {
            "cooling_coefficient": h,
            "radiative_area": A,
        }
        
        result = apply_thermal_radiation(
            heat_shock_Q=10.0,
            liquidity_L=L,
            fixed_contracts_F=F,
            context=context,
        )
        
        expected_tau = I / (h * A)  # 60 / 6 = 10.0 s
        assert abs(result["metrics"]["time_constant_tau_s"] - expected_tau) < 0.01

    def test_equilibrium_temperature_calculation(self):
        """T_eq = T∞ + Q/(h×A) debe ser correcto."""
        Q = 50.0
        h = 5.0
        A = 2.0
        T_inf = 298.0
        
        context = {
            "cooling_coefficient": h,
            "radiative_area": A,
            "ambient_temperature": T_inf,
        }
        
        result = apply_thermal_radiation(
            heat_shock_Q=Q,
            liquidity_L=10.0,
            fixed_contracts_F=10.0,
            context=context,
        )
        
        expected_T_eq = T_inf + Q / (h * A)  # 298 + 50/10 = 303 K
        assert abs(result["metrics"]["equilibrium_temperature_K"] - expected_T_eq) < 0.1


class TestThermalRadiationThresholds:
    """Tests de umbrales de temperatura."""

    def test_fever_threshold_triggers_failure(self, empty_context):
        """Superar umbral de fiebre debe fallar."""
        context = {
            **empty_context,
            "base_system_temperature": 298.0,
            "fever_threshold": 323.0,
        }
        
        result = apply_thermal_radiation(
            heat_shock_Q=500.0,  # Choque grande
            liquidity_L=1.0,
            fixed_contracts_F=1.0,  # Inercia baja = 1.0
            context=context,
        )
        
        # T_new = 298 + 500/1 = 798 K >> 323 K
        assert result["success"] is False
        # Puede ser fiebre, ebullición o fusión según el valor

    def test_boiling_threshold_triggers_failure(self, empty_context):
        """Superar umbral de ebullición debe fallar con mensaje específico."""
        context = {
            **empty_context,
            "base_system_temperature": 298.0,
            "fever_threshold": 300.0,
            "critical_threshold": 350.0,
        }
        
        result = apply_thermal_radiation(
            heat_shock_Q=100.0,
            liquidity_L=1.0,
            fixed_contracts_F=1.0,
            context=context,
        )
        
        # T_new = 298 + 100 = 398 K > 350 K
        assert result["success"] is False
        assert "EBULLICIÓN" in result["error"] or "FUSIÓN" in result["error"]

    def test_below_thresholds_succeeds(self, standard_context):
        """Debajo de todos los umbrales debe tener éxito."""
        result = apply_thermal_radiation(
            heat_shock_Q=1.0,  # Choque mínimo
            liquidity_L=100.0,
            fixed_contracts_F=100.0,  # Alta inercia
            context=standard_context,
        )
        
        assert result["success"] is True


# =============================================================================
# TESTS: CO-VECTOR 4 — AUDITORÍA COHOMOLÓGICA
# =============================================================================

class TestCohomologicalAuditValidation:
    """Tests de validación de entradas para auditoría."""

    def test_empty_passport_returns_error(self, empty_context):
        """Pasaporte vacío debe fallar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport={},
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "vacío" in result["error"].lower()

    def test_none_passport_returns_error(self, empty_context):
        """Pasaporte None debe fallar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport=None,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        assert result["success"] is False

    def test_non_dict_passport_returns_error(self, empty_context):
        """Pasaporte no-dict debe fallar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport=["laplace_poles", "betti_1"],
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "dict" in result["error"].lower()

    def test_empty_schema_returns_error(self, valid_telemetry_passport, empty_context):
        """Schema vacío debe fallar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport=valid_telemetry_passport,
            target_schema="",
            context=empty_context,
        )
        
        assert result["success"] is False


class TestCohomologicalAuditStability:
    """Tests de análisis de estabilidad BIBO."""

    def test_stable_poles_pass(self, valid_telemetry_passport, empty_context):
        """Polos en LHP deben pasar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport=valid_telemetry_passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        # Buscar hallazgo de estabilidad
        stability_findings = [
            f for f in result["audit_findings"]
            if "BIBO" in f["criterion"] or "STABILITY" in f["criterion"]
        ]
        
        # Debe haber al menos un hallazgo de estabilidad que pase
        passed_stability = [f for f in stability_findings if f["passed"]]
        assert len(passed_stability) > 0

    def test_unstable_poles_fail(self, unstable_telemetry_passport, empty_context):
        """Polos en RHP deben fallar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport=unstable_telemetry_passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "VETO BANCARIO" in result["error"] or "RHP" in result["error"]

    def test_marginal_poles_warn(self, empty_context):
        """Polos sobre eje imaginario deben advertir."""
        passport = {
            "laplace_poles": [0j, 2j, -2j],  # Sobre el eje imaginario
            "betti_1": 0,
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        marginal_findings = [
            f for f in result["audit_findings"]
            if "MARGINAL" in f["criterion"]
        ]
        
        assert len(marginal_findings) > 0

    def test_missing_poles_warns(self, empty_context):
        """Polos ausentes deben generar advertencia."""
        passport = {
            "betti_1": 0,
            # Sin laplace_poles
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        absent_findings = [
            f for f in result["audit_findings"]
            if "PRESENT" in f["criterion"] and "LAPLACE" in f["criterion"]
        ]
        
        assert len(absent_findings) > 0


class TestCohomologicalAuditTopology:
    """Tests de análisis topológico (β₁)."""

    def test_acyclic_structure_passes(self, valid_telemetry_passport, empty_context):
        """β₁ = 0 debe pasar."""
        result = audit_cohomological_isomorphism(
            telemetry_passport=valid_telemetry_passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        betti_findings = [
            f for f in result["audit_findings"]
            if "BETTI" in f["criterion"]
        ]
        
        # Al menos uno debe pasar (acyclicidad)
        passed = [f for f in betti_findings if f["passed"]]
        assert len(passed) > 0

    def test_cyclic_structure_fails(self, empty_context):
        """β₁ > 0 debe fallar."""
        passport = {
            "laplace_poles": [-1.0, -2.0],
            "betti_1": 3,  # Ciclos
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="SECOP_II",
            context=empty_context,
        )
        
        assert result["success"] is False
        assert "VETO ESTATAL" in result["error"] or "β₁" in result["error"]

    def test_negative_betti_fails(self, empty_context):
        """β₁ negativo debe fallar (inválido matemáticamente)."""
        passport = {
            "laplace_poles": [-1.0],
            "betti_1": -1,
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        assert result["success"] is False

    def test_missing_betti_warns(self, empty_context):
        """β₁ ausente debe generar advertencia."""
        passport = {
            "laplace_poles": [-1.0, -2.0],
            # Sin betti_1
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="BIM_2026",
            context=empty_context,
        )
        
        absent_findings = [
            f for f in result["audit_findings"]
            if "BETTI_1_PRESENT" in f["criterion"]
        ]
        
        assert len(absent_findings) > 0


class TestCohomologicalAuditCompleteness:
    """Tests de completitud del pasaporte."""

    def test_complete_passport_passes(self, empty_context):
        """Pasaporte completo debe pasar verificación de completitud."""
        passport = {
            "laplace_poles": [-1.0],
            "betti_1": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }
        
        context = {
            **empty_context,
            "required_passport_fields": ["laplace_poles", "betti_1", "timestamp", "version"],
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="BIM_2026",
            context=context,
        )
        
        completeness_findings = [
            f for f in result["audit_findings"]
            if "COMPLETENESS" in f["criterion"]
        ]
        
        assert any(f["passed"] for f in completeness_findings)

    def test_incomplete_passport_warns(self, empty_context):
        """Pasaporte incompleto debe generar advertencia."""
        passport = {
            "laplace_poles": [-1.0],
            "betti_1": 0,
            # Faltan timestamp y version
        }
        
        context = {
            **empty_context,
            "required_passport_fields": ["laplace_poles", "betti_1", "timestamp", "version"],
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="BIM_2026",
            context=context,
        )
        
        completeness_findings = [
            f for f in result["audit_findings"]
            if "COMPLETENESS" in f["criterion"]
        ]
        
        assert any(not f["passed"] for f in completeness_findings)


# =============================================================================
# TESTS: EVALUACIÓN EN CASCADA
# =============================================================================

class TestCascadeEvaluation:
    """Tests para evaluación en cascada de co-vectores."""

    def test_empty_cascade_succeeds(self, mif_registry, empty_context):
        """Cascada vacía debe tener éxito trivial."""
        result = mif_registry.evaluate_cascade(
            covector_names=[],
            payload={},
            context=empty_context,
        )
        
        assert result["_cascade_success"] is True
        assert result["_cascade_empty"] is True

    def test_single_covector_cascade(self, mif_registry, standard_context):
        """Cascada de un solo co-vector debe funcionar."""
        result = mif_registry.evaluate_cascade(
            covector_names=["turbulence_shock"],
            payload={
                "inductance_L": 0.01,
                "target_current_drop": 1.0,
            },
            context=standard_context,
        )
        
        assert "turbulence_shock" in result
        assert result["_cascade_completed_steps"] == 1

    def test_cascade_fail_fast_stops_on_failure(self, mif_registry, empty_context):
        """fail_fast=True debe detener en primer fallo."""
        # Primer co-vector fallará (inductancia negativa)
        result = mif_registry.evaluate_cascade(
            covector_names=["turbulence_shock", "inflationary_radiation"],
            payload={
                "inductance_L": -0.1,  # Inválido
                "target_current_drop": 1.0,
                "heat_shock_Q": 10.0,
                "liquidity_L": 10.0,
                "fixed_contracts_F": 10.0,
            },
            context=empty_context,
            fail_fast=True,
        )
        
        assert result["_cascade_success"] is False
        assert result["_cascade_halted_at"] == "turbulence_shock"
        assert "inflationary_radiation" not in result

    def test_cascade_continue_on_failure(self, mif_registry, empty_context):
        """fail_fast=False debe continuar tras fallos."""
        result = mif_registry.evaluate_cascade(
            covector_names=["turbulence_shock", "inflationary_radiation"],
            payload={
                "inductance_L": -0.1,  # Inválido para primer co-vector
                "target_current_drop": 1.0,
                "heat_shock_Q": 1.0,
                "liquidity_L": 100.0,
                "fixed_contracts_F": 100.0,
            },
            context=empty_context,
            fail_fast=False,
        )
        
        # Ambos deben haberse evaluado
        assert "turbulence_shock" in result
        assert "inflationary_radiation" in result
        assert result["_cascade_completed_steps"] == 2

    def test_cascade_propagates_metrics(self, mif_registry, standard_context):
        """Métricas deben propagarse al contexto para siguientes co-vectores."""
        registry = MIFRegistry()
        
        # Handler que agrega métricas al resultado
        def first_handler(**kwargs):
            return {
                "success": True,
                "metrics": {"computed_value": 42},
            }
        
        # Handler que usa las métricas del anterior
        def second_handler(context, **kwargs):
            value = context.get("computed_value", 0)
            return {
                "success": True,
                "received_value": value,
            }
        
        registry.register_covector("first", Stratum.PHYSICS, first_handler)
        registry.register_covector("second", Stratum.TACTICS, second_handler)
        
        result = registry.evaluate_cascade(
            covector_names=["first", "second"],
            payload={},
            context={},
            propagate_metrics=True,
        )
        
        assert result["second"]["received_value"] == 42


# =============================================================================
# TESTS: FUNCIONES AUXILIARES
# =============================================================================

class TestHelperFunctions:
    """Tests para funciones auxiliares."""

    def test_success_template(self):
        """_success debe crear estructura correcta."""
        result = _success("Test message", {"key": "value"})
        
        assert result["success"] is True
        assert result["status"] == VectorResultStatus.SUCCESS.value
        assert result["payload"]["message"] == "Test message"
        assert result["metrics"]["key"] == "value"

    def test_physics_error_template(self):
        """_physics_error debe crear estructura correcta."""
        result = _physics_error("Physical violation")
        
        assert result["success"] is False
        assert result["status"] == VectorResultStatus.PHYSICS_ERROR.value
        assert result["error"] == "Physical violation"

    def test_topology_error_template(self):
        """_topology_error debe crear estructura correcta."""
        result = _topology_error("Topology broken")
        
        assert result["success"] is False
        assert result["status"] == VectorResultStatus.TOPOLOGY_ERROR.value

    def test_logic_error_template(self):
        """_logic_error debe crear estructura correcta."""
        result = _logic_error("Logic violated")
        
        assert result["success"] is False
        assert result["status"] == VectorResultStatus.LOGIC_ERROR.value


# =============================================================================
# TESTS: INTEGRACIÓN END-TO-END
# =============================================================================

class TestIntegrationEndToEnd:
    """Tests de integración completos."""

    def test_full_mif_initialization_and_evaluation(self):
        """Ciclo completo: inicialización → evaluación → logs."""
        mif = initialize_mif()
        
        assert len(mif.registered_names) == 4
        
        # Evaluar cada co-vector con entradas válidas
        contexts = {
            "turbulence_shock": {
                "viscoelastic_resistance": 10.0,
                "viscoelastic_capacitance": 0.001,
                "critical_voltage_threshold": 1000.0,
            },
            "chaos_node_death": {
                "fiedler_threshold": 0.01,
                "pyramid_stability_threshold": 0.3,
                "survival_ratio_threshold": 0.3,
            },
            "inflationary_radiation": {
                "base_system_temperature": 298.0,
                "fever_threshold": 500.0,
            },
            "institutional_audit": {
                "required_passport_fields": ["laplace_poles", "betti_1"],
            },
        }
        
        payloads = {
            "turbulence_shock": {
                "inductance_L": 0.01,
                "target_current_drop": 1.0,
            },
            "chaos_node_death": {
                "graph": nx.DiGraph([("A", "B"), ("B", "C")]),
                "nodes_to_kill": [],
            },
            "inflationary_radiation": {
                "heat_shock_Q": 10.0,
                "liquidity_L": 100.0,
                "fixed_contracts_F": 100.0,
            },
            "institutional_audit": {
                "telemetry_passport": {
                    "laplace_poles": [-1.0, -2.0],
                    "betti_1": 0,
                },
                "target_schema": "TEST_SCHEMA",
            },
        }
        
        results = {}
        for name in mif.registered_names:
            result = mif.evaluate_impact(
                name,
                payloads[name],
                contexts[name],
            )
            results[name] = result
            
            # Cada resultado debe tener estructura mínima
            assert "success" in result
            assert "_mif_stratum" in result
        
        # Log debe tener 4 entradas
        assert len(mif.evaluation_log) == 4

    def test_realistic_supply_chain_scenario(self):
        """Escenario realista: cadena de suministro bajo estrés."""
        mif = initialize_mif()
        
        # Construir grafo de cadena de suministro
        supply_chain = nx.DiGraph()
        supply_chain.add_nodes_from([
            ("Proveedor_A", {"type": "INSUMO"}),
            ("Proveedor_B", {"type": "INSUMO"}),
            ("Proveedor_C", {"type": "INSUMO"}),
            ("Almacen_Central", {"type": "APU"}),
            ("Distribucion_Norte", {"type": "APU"}),
            ("Distribucion_Sur", {"type": "APU"}),
            ("Cliente_Final", {"type": "OUTPUT"}),
        ])
        supply_chain.add_edges_from([
            ("Proveedor_A", "Almacen_Central"),
            ("Proveedor_B", "Almacen_Central"),
            ("Proveedor_C", "Almacen_Central"),
            ("Almacen_Central", "Distribucion_Norte"),
            ("Almacen_Central", "Distribucion_Sur"),
            ("Distribucion_Norte", "Cliente_Final"),
            ("Distribucion_Sur", "Cliente_Final"),
        ])
        
        # Evaluar cascada: shock logístico → muerte de nodo → radiación térmica → auditoría
        cascade_result = mif.evaluate_cascade(
            covector_names=[
                "turbulence_shock",
                "chaos_node_death",
                "inflationary_radiation",
                "institutional_audit",
            ],
            payload={
                # Para turbulence_shock
                "inductance_L": 0.1,
                "target_current_drop": 5.0,
                # Para chaos_node_death
                "graph": supply_chain,
                "nodes_to_kill": ["Proveedor_B"],  # Matar un proveedor
                # Para inflationary_radiation
                "heat_shock_Q": 50.0,
                "liquidity_L": 100.0,
                "fixed_contracts_F": 50.0,
                # Para institutional_audit
                "telemetry_passport": {
                    "laplace_poles": [-0.5, -1.0, -2.0],
                    "betti_1": 0,
                },
                "target_schema": "ISO_31000",
            },
            context={
                "viscoelastic_resistance": 10.0,
                "viscoelastic_capacitance": 0.01,
                "critical_voltage_threshold": 100.0,
                "fiedler_threshold": 0.05,
                "pyramid_stability_threshold": 0.5,
                "survival_ratio_threshold": 0.5,
                "base_system_temperature": 298.0,
                "fever_threshold": 350.0,
                "critical_threshold": 400.0,
            },
            fail_fast=True,
        )
        
        # La cascada debe haber completado todos los pasos
        assert cascade_result["_cascade_completed_steps"] == 4


class TestEdgeCasesAndBoundaries:
    """Tests de casos límite y fronteras."""

    def test_very_large_inductance(self, standard_context):
        """Inductancia muy grande debe manejarse correctamente."""
        result = inject_logistical_turbulence(
            inductance_L=1e6,  # 1 MH
            target_current_drop=0.001,  # Pequeño cambio
            context=standard_context,
        )
        
        # V = L × di/dt = 1e6 × 0.001 = 1000 V
        assert result["metrics"]["v_flyback_volts"] == pytest.approx(1000.0)

    def test_very_small_values(self, standard_context):
        """Valores muy pequeños no deben causar errores numéricos."""
        result = inject_logistical_turbulence(
            inductance_L=1e-12,  # 1 pH
            target_current_drop=1e-12,
            context=standard_context,
        )
        
        assert result["success"] is True
        assert math.isfinite(result["metrics"]["v_flyback_volts"])

    def test_graph_with_self_loops(self, empty_context):
        """Grafo con self-loops debe manejarse."""
        G = nx.DiGraph()
        G.add_nodes_from([("A", {"type": "INSUMO"}), ("B", {"type": "APU"})])
        G.add_edge("A", "B")
        G.add_edge("A", "A")  # Self-loop
        
        result = perturb_topological_manifold(
            graph=G,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        assert "success" in result

    def test_complex_poles_parsing(self, empty_context):
        """Polos complejos en diferentes formatos deben parsearse."""
        passport = {
            "laplace_poles": [
                -1.0,             # Real
                -2.0 + 3.0j,      # Complejo
                "-3+4j",          # String
                complex(-5, 6),   # complex()
            ],
            "betti_1": 0,
        }
        
        result = audit_cohomological_isomorphism(
            telemetry_passport=passport,
            target_schema="TEST",
            context=empty_context,
        )
        
        # Debe haber procesado sin error de parseo
        assert "PARSEABLE" not in str([f["criterion"] for f in result["audit_findings"] if not f["passed"]])


# =============================================================================
# TESTS: PROPIEDADES MATEMÁTICAS INVARIANTES
# =============================================================================

class TestMathematicalInvariants:
    """Tests de propiedades matemáticas que deben cumplirse siempre."""

    def test_energy_conservation_rlc(self, standard_context):
        """La energía almacenada no puede exceder el trabajo realizado."""
        L = 0.5
        I = 10.0
        
        result = inject_logistical_turbulence(
            inductance_L=L,
            target_current_drop=I,
            context=standard_context,
        )
        
        E_stored = result["metrics"]["stored_energy_joules"]
        
        # E = ½LI² es la fórmula exacta, debe coincidir
        expected = 0.5 * L * I ** 2
        assert E_stored == pytest.approx(expected, rel=1e-6)

    def test_cheeger_inequality_always_holds(self, simple_digraph, empty_context):
        """La desigualdad de Cheeger debe cumplirse para cualquier grafo conexo."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        metrics = result["metrics"]
        
        if metrics["betti_0"] == 1:  # Grafo conexo
            h_lower = metrics["cheeger_lower_bound"]
            h_est = metrics["cheeger_estimate"]
            h_upper = metrics["cheeger_upper_bound"]
            
            assert h_lower <= h_est <= h_upper or h_lower == h_upper == h_est == 0

    def test_fiedler_value_bounds(self, simple_digraph, empty_context):
        """0 ≤ λ₂ ≤ n para cualquier grafo."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        n = result["metrics"]["surviving_nodes"]
        fiedler = result["metrics"]["fiedler_value"]
        
        assert 0 <= fiedler <= n

    def test_betti_numbers_non_negative(self, simple_digraph, empty_context):
        """Los números de Betti siempre deben ser ≥ 0."""
        result = perturb_topological_manifold(
            graph=simple_digraph,
            nodes_to_kill=[],
            context=empty_context,
        )
        
        assert result["metrics"]["betti_0"] >= 0
        assert result["metrics"]["betti_1"] >= 0

    def test_thermal_inertia_positivity(self, empty_context):
        """La inercia térmica debe ser no negativa cuando L,F ≥ 0."""
        for L in [0, 1, 10, 100]:
            for F in [0, 1, 10, 100]:
                if L > 0 and F > 0:
                    result = apply_thermal_radiation(
                        heat_shock_Q=1.0,
                        liquidity_L=L,
                        fixed_contracts_F=F,
                        context=empty_context,
                    )
                    
                    I = result["metrics"]["thermal_inertia"]
                    assert I >= 0
                    assert I == pytest.approx(L * F, rel=1e-4)


# =============================================================================
# TESTS: RENDIMIENTO Y ESTABILIDAD
# =============================================================================

class TestPerformance:
    """Tests de rendimiento básico."""

    def test_large_graph_handling(self, empty_context):
        """Grafos grandes deben procesarse en tiempo razonable."""
        import time
        
        # Crear grafo grande
        n_nodes = 1000
        G = nx.DiGraph()
        for i in range(n_nodes):
            G.add_node(f"N{i}", type="INSUMO" if i < n_nodes // 2 else "APU")
        
        # Añadir aristas (grafo casi completo sería O(n²), usamos sparse)
        for i in range(n_nodes - 1):
            G.add_edge(f"N{i}", f"N{i+1}")
        
        start = time.perf_counter()
        
        result = perturb_topological_manifold(
            graph=G,
            nodes_to_kill=[f"N{i}" for i in range(0, 100, 10)],  # Matar 10 nodos
            context=empty_context,
        )
        
        elapsed = time.perf_counter() - start
        
        assert result["success"] is True or result["success"] is False  # Debe completar
        assert elapsed < 5.0  # Menos de 5 segundos

    def test_repeated_evaluations_stable(self, mif_registry, standard_context):
        """Evaluaciones repetidas deben dar resultados consistentes."""
        results = []
        
        for _ in range(10):
            result = mif_registry.evaluate_impact(
                "turbulence_shock",
                {"inductance_L": 0.1, "target_current_drop": 5.0},
                standard_context,
            )
            results.append(result["metrics"]["v_flyback_volts"])
        
        # Todos los resultados deben ser idénticos (determinístico)
        assert all(v == results[0] for v in results)


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Parar en primer fallo
        "--durations=10",  # Mostrar 10 tests más lentos
    ])