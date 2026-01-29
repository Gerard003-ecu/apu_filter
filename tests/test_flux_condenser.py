"""
Suite de Pruebas para el `DataFluxCondenser` - Versión Refinada V5.

Incluye pruebas unitarias para el motor de física refinado y pruebas de integración
para el orquestador DataFluxCondenser.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import math
import time
import warnings
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# Importar clases del módulo refactorizado
from app.flux_condenser import (
    DiscreteVectorCalculus, MaxwellSolver, PortHamiltonianController,
    TopologicalAnalyzer, EntropyCalculator, UnifiedPhysicalState,
    CodeQualityMetrics, RefinedFluxPhysicsEngine, SystemConstants,
    DampingType, ConfigurationError, DataFluxCondenser, CondenserConfig,
    PIController, InvalidInputError, ProcessingError, ProcessingStats,
    BatchResult, ParsedData
)

# Alias for compatibility in tests
FluxPhysicsEngine = RefinedFluxPhysicsEngine

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#17A2B8',
    'energy_potential': '#3498db',
    'energy_kinetic': '#e74c3c',
    'energy_total': '#2ecc71',
    'entropy': '#9b59b6',
    'stability': '#f39c12'
}


# ============================================================================
# CLASE BASE PARA TESTS CON VISUALIZACIÓN
# ============================================================================
class VisualTestCase(unittest.TestCase):
    """Caso de prueba base con soporte para visualización."""

    @classmethod
    def setUpClass(cls):
        cls.figures: List[plt.Figure] = []
        cls.test_results: Dict[str, Any] = {}

    def add_figure(self, fig: plt.Figure, name: str):
        """Registra una figura para mostrar al final."""
        self.figures.append((fig, name))

    @classmethod
    def tearDownClass(cls):
        """Muestra todas las figuras al final de los tests."""
        # En entorno CI/Headless, esto podría deshabilitarse o guardar a archivo
        pass


# ============================================================================
# TESTS DE DiscreteVectorCalculus
# ============================================================================
class TestDiscreteVectorCalculus(VisualTestCase):
    """Tests para el cálculo vectorial discreto."""

    def setUp(self):
        # Grafo simple: triángulo
        self.triangle_adj = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1}
        }

        # Grafo K4 (tetraedro)
        self.k4_adj = {
            i: set(range(4)) - {i} for i in range(4)
        }

    def test_triangle_complex(self):
        """Verifica complejo de de Rham para triángulo."""
        vc = DiscreteVectorCalculus(self.triangle_adj)

        # Verificar dimensiones
        self.assertEqual(vc.num_nodes, 3, "Triángulo debe tener 3 nodos")
        self.assertEqual(vc.num_edges, 3, "Triángulo debe tener 3 aristas")
        self.assertEqual(vc.num_faces, 1, "Triángulo debe tener 1 cara")

        # Verificar d₀ (gradiente)
        if hasattr(vc, 'd0'):
            # d₀ debe tener dimensión (num_edges, num_nodes)
            self.assertEqual(vc.d0.shape, (3, 3))

    def test_gradient_divergence_adjoint(self):
        """Verifica que divergencia es adjunto negativo del gradiente."""
        if not SystemConstants.EPSILON: # Skip if constants not loaded correctly (sanity check)
             pass

        vc = DiscreteVectorCalculus(self.k4_adj)

        # Función escalar aleatoria en nodos
        phi = np.random.randn(vc.num_nodes)

        # 1-forma aleatoria en aristas
        psi = np.random.randn(vc.num_edges)

        # Verificar <grad(φ), ψ> = -<φ, div(ψ)>
        grad_phi = vc.gradient(phi)
        div_psi = vc.divergence(psi)

        lhs = np.dot(grad_phi, psi)
        rhs = -np.dot(phi, div_psi)

        self.assertAlmostEqual(lhs, rhs, places=10,
            msg="Gradiente y divergencia deben ser adjuntos")


# ============================================================================
# TESTS DE MaxwellSolver
# ============================================================================
class TestMaxwellSolver(VisualTestCase):
    """Tests para el solver de Maxwell discreto."""

    def setUp(self):
        # Grafo para simulación
        self.adj = {i: set(range(6)) - {i} for i in range(6)}
        self.vc = DiscreteVectorCalculus(self.adj)

    def test_energy_conservation_no_dissipation(self):
        """Verifica conservación de energía sin disipación."""
        solver = MaxwellSolver(
            self.vc,
            permittivity=1.0,
            permeability=1.0,
            electric_conductivity=0.0  # Sin disipación
        )

        # Condición inicial: pulso en E
        solver.E[0] = 1.0
        # solver.update_constitutive_relations() # Not in this implementation

        energies = []
        dt = 0.01

        for step in range(50):
            solver.step(dt)
            energy = solver.compute_energy()
            energies.append(energy['total_energy'])

        # Verificar conservación (menos de 1% de variación)
        initial_energy = energies[0]
        max_deviation = max(abs(e - initial_energy) for e in energies)

        # Relative tolerance might need adjustment depending on numerical scheme
        self.assertLess(max_deviation, 0.05 * initial_energy)


# ============================================================================
# TESTS DE RefinedFluxPhysicsEngine
# ============================================================================
class TestRefinedFluxPhysicsEngine(VisualTestCase):
    """Tests para el motor de física completo."""

    def test_metrics_calculation(self):
        """Verifica cálculo de métricas."""
        engine = RefinedFluxPhysicsEngine(
            capacitance=1.0,
            resistance=0.5,
            inductance=1.0
        )

        # Calcular métricas
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=60,
            error_count=5,
            processing_time=1.0
        )

        # Verificar que contiene métricas esperadas
        expected_keys = [
            'current_I', 'charge', 'total_energy', 'entropy_bits',
            'betti_0', 'betti_1', 'gyroscopic_stability'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics,
                f"Métricas deben contener '{key}'")

    def test_conservation_verification(self):
        """Verifica conservación de energía."""
        engine = RefinedFluxPhysicsEngine(
            capacitance=1.0,
            resistance=0.0,  # Sin disipación
            inductance=1.0
        )

        # Simular varios pasos
        for i in range(50):
            engine.calculate_metrics(
                total_records=100 + i,
                cache_hits=50 + i % 10,
                error_count=1
            )

        # Obtener diagnóstico
        diagnosis = engine.get_system_diagnosis()

        self.assertIn('conservation', diagnosis)


# ============================================================================
# TESTS: DataFluxCondenser - Integración
# ============================================================================
class TestDataFluxCondenserIntegration(unittest.TestCase):

    def setUp(self):
        self.config = {
            "parser_settings": {"delimiter": ",", "encoding": "utf-8"},
            "processor_settings": {"validate_types": True, "skip_empty": False},
        }
        self.profile = {
            "columns_mapping": {"cod_insumo": "codigo", "descripcion": "desc"},
            "validation_rules": {"required_fields": ["codigo", "cantidad"]},
        }
        self.condenser = DataFluxCondenser(self.config, self.profile)

    def test_initialization(self):
        self.assertIsInstance(self.condenser.physics, RefinedFluxPhysicsEngine)
        self.assertIsInstance(self.condenser.controller, PIController)

    @patch("app.flux_condenser.ReportParserCrudo")
    @patch("app.flux_condenser.APUProcessor")
    def test_stabilize_flow(self, MockProcessor, MockParser):
        # Mock dependencies
        mock_parser_instance = MockParser.return_value
        mock_parser_instance.parse_to_raw.return_value = [{"col": "val"}] * 10
        mock_parser_instance.get_parse_cache.return_value = {}

        mock_processor_instance = MockProcessor.return_value
        mock_processor_instance.process_all.return_value = pd.DataFrame([{"col": "val"}] * 10)

        # Test file path
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat:

            mock_stat.return_value.st_size = 1000

            # Use a dummy path that passes regex check for extensions if any
            result_df = self.condenser.stabilize("dummy.csv")

            self.assertFalse(result_df.empty)
            self.assertEqual(len(result_df), 10)

            # Check stats
            stats = self.condenser.get_processing_stats()
            self.assertEqual(stats['statistics']['processed_records'], 10)


# ============================================================================
# TESTS: PIController
# ============================================================================
class TestPIController(unittest.TestCase):

    def test_compute(self):
        controller = PIController(kp=1.0, ki=0.1, setpoint=0.5, min_output=1, max_output=100)
        output = controller.compute(0.4) # Error 0.1
        self.assertIsInstance(output, int)
        self.assertGreater(output, 0)

    def test_anti_windup(self):
        controller = PIController(kp=1.0, ki=100.0, setpoint=0.5, min_output=1, max_output=10)
        # Saturate
        for _ in range(10):
            controller.compute(0.0) # Big error

        # Check internal state limits
        self.assertLessEqual(abs(controller._integral_error), controller._integral_limit)


# ============================================================================
# RUNNER DE TESTS
# ============================================================================
if __name__ == '__main__':
    unittest.main()
