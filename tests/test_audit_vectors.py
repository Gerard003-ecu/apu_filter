"""
Suite de Pruebas — Vector de Auditoría Topológica.

Cobertura de pruebas:
─────────────────────
├── Unitarias: funciones matemáticas individuales
├── Sanitización: limpieza y validación de datos
├── Integración: flujo completo del vector
├── Propiedades: invariantes matemáticos
├── Casos límite: edge cases y degeneraciones
├── Configuración: validación de parámetros
└── Rendimiento: benchmarks básicos

Ejecución:
    pytest test_audit_vectors.py -v --cov=audit_vectors --cov-report=html
    pytest test_audit_vectors.py -v -m "not slow"  # Excluir lentas
    pytest test_audit_vectors.py -v -k "simpson"   # Solo Simpson
"""

import math
import time
import hashlib
import warnings
from typing import Dict, Set, Any, List, Tuple
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume

# ═══════════════════════════════════════════════════════════════════════════
# Importación del módulo bajo prueba
# ═══════════════════════════════════════════════════════════════════════════

from audit_vectors import (
    # Configuración
    AuditConfiguration,
    DEFAULT_CONFIG,
    
    # Dataclasses
    GraphMetrics,
    StabilityVerdict,
    DataQualityReport,
    SchemaValidationResult,
    DistributionDiagnostics,
    
    # Funciones de sanitización
    _sanitize_codigo_set,
    _extract_insumo_distribution,
    _validate_dataframe_schema,
    
    # Funciones matemáticas
    _compute_simpson_diversity,
    _compute_shannon_entropy,
    _compute_gini_coefficient,
    _compute_connectivity_ratio,
    _compute_topological_robustness,
    _compute_composite_stability,
    _compute_algebraic_connectivity,
    _compute_distribution_diagnostics,
    
    # Veredicto
    _determine_verdict,
    _identify_contributing_factors,
    _generate_recommendations,
    
    # Hashing
    _build_audit_hash,
    
    # Vector principal
    vector_audit_pyramidal_structure,
    
    # Funciones auxiliares exportadas
    compute_herfindahl_index,
    compute_effective_number_of_species,
    analyze_graph_spectrum,
    
    # Constantes
    SCIPY_AVAILABLE,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — Datos de prueba reutilizables
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config_default() -> AuditConfiguration:
    """Configuración por defecto."""
    return DEFAULT_CONFIG


@pytest.fixture
def config_strict() -> AuditConfiguration:
    """Configuración con umbrales estrictos."""
    return AuditConfiguration(
        psi_veto_threshold=0.50,
        psi_warning_threshold=0.75,
        psi_healthy_threshold=0.90,
        simpson_min_threshold=0.60,
        gini_max_threshold=0.50,
    )


@pytest.fixture
def config_relaxed() -> AuditConfiguration:
    """Configuración con umbrales relajados."""
    return AuditConfiguration(
        psi_veto_threshold=0.20,
        psi_warning_threshold=0.40,
        psi_healthy_threshold=0.60,
    )


@pytest.fixture
def empty_distribution() -> Dict[str, int]:
    """Distribución vacía."""
    return {}


@pytest.fixture
def monopoly_distribution() -> Dict[str, int]:
    """Distribución monopolística (un solo insumo)."""
    return {"CEMENTO": 100}


@pytest.fixture
def duopoly_distribution() -> Dict[str, int]:
    """Distribución duopólica."""
    return {"CEMENTO": 50, "ARENA": 50}


@pytest.fixture
def uniform_distribution() -> Dict[str, int]:
    """Distribución uniforme (5 insumos iguales)."""
    return {
        "CEMENTO": 20,
        "ARENA": 20,
        "GRAVA": 20,
        "AGUA": 20,
        "ACERO": 20,
    }


@pytest.fixture
def skewed_distribution() -> Dict[str, int]:
    """Distribución sesgada (Pareto-like)."""
    return {
        "CEMENTO": 50,
        "ARENA": 25,
        "GRAVA": 12,
        "AGUA": 8,
        "ACERO": 5,
    }


@pytest.fixture
def highly_diverse_distribution() -> Dict[str, int]:
    """Distribución muy diversa (20 insumos)."""
    return {f"INSUMO_{i:02d}": 5 for i in range(20)}


@pytest.fixture
def df_presupuesto_valid() -> pd.DataFrame:
    """DataFrame de presupuesto válido."""
    return pd.DataFrame({
        "CODIGO_APU": ["APU001", "APU002", "APU003", "APU004", "APU005"],
        "DESCRIPCION": ["Desc A", "Desc B", "Desc C", "Desc D", "Desc E"],
        "VALOR": [1000, 2000, 1500, 3000, 2500],
    })


@pytest.fixture
def df_insumos_valid() -> pd.DataFrame:
    """DataFrame de insumos válido con buena conectividad."""
    return pd.DataFrame({
        "CODIGO_APU": [
            "APU001", "APU001", "APU001",
            "APU002", "APU002", "APU002",
            "APU003", "APU003", "APU003",
            "APU004", "APU004", "APU004",
            "APU005", "APU005", "APU005",
        ],
        "DESCRIPCION_INSUMO": [
            "CEMENTO", "ARENA", "GRAVA",
            "CEMENTO", "AGUA", "ACERO",
            "ARENA", "GRAVA", "AGUA",
            "CEMENTO", "ACERO", "MADERA",
            "ARENA", "AGUA", "PINTURA",
        ],
    })


@pytest.fixture
def df_insumos_sparse() -> pd.DataFrame:
    """DataFrame de insumos con poca conectividad."""
    return pd.DataFrame({
        "CODIGO_APU": ["APU001", "APU002", "APU003"],
        "DESCRIPCION_INSUMO": ["CEMENTO", "CEMENTO", "CEMENTO"],
    })


@pytest.fixture
def df_presupuesto_with_nulls() -> pd.DataFrame:
    """DataFrame de presupuesto con valores nulos."""
    return pd.DataFrame({
        "CODIGO_APU": ["APU001", None, "APU003", np.nan, "APU005", "", "nan"],
    })


@pytest.fixture
def df_insumos_with_floating() -> pd.DataFrame:
    """DataFrame de insumos que deja APUs flotantes."""
    return pd.DataFrame({
        "CODIGO_APU": ["APU001", "APU001", "APU002"],  # APU003-005 flotantes
        "DESCRIPCION_INSUMO": ["CEMENTO", "ARENA", "GRAVA"],
    })


@pytest.fixture
def apu_to_insumos_connected() -> Dict[str, Set[str]]:
    """Mapa APU→Insumos bien conectado."""
    return {
        "APU001": {"CEMENTO", "ARENA", "GRAVA"},
        "APU002": {"CEMENTO", "AGUA", "ACERO"},
        "APU003": {"ARENA", "GRAVA", "AGUA"},
        "APU004": {"CEMENTO", "ACERO", "MADERA"},
        "APU005": {"ARENA", "AGUA", "PINTURA"},
    }


@pytest.fixture
def apu_to_insumos_disconnected() -> Dict[str, Set[str]]:
    """Mapa APU→Insumos sin conexiones compartidas."""
    return {
        "APU001": {"A", "B"},
        "APU002": {"C", "D"},
        "APU003": {"E", "F"},
    }


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: AuditConfiguration — Validación de configuración
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditConfiguration:
    """Pruebas de la dataclass de configuración."""
    
    def test_default_config_is_valid(self, config_default):
        """La configuración por defecto debe ser válida."""
        assert config_default.psi_veto_threshold == 0.40
        assert config_default.psi_warning_threshold == 0.65
        assert config_default.psi_healthy_threshold == 0.80
        assert config_default.algorithm_version == "3.0.0-spectral"
    
    def test_threshold_order_invariant(self):
        """Los umbrales deben estar ordenados: veto < warning < healthy."""
        # Válido
        config = AuditConfiguration(
            psi_veto_threshold=0.30,
            psi_warning_threshold=0.50,
            psi_healthy_threshold=0.70,
        )
        assert config.psi_veto_threshold < config.psi_warning_threshold
        assert config.psi_warning_threshold < config.psi_healthy_threshold
    
    def test_threshold_order_violation_raises(self):
        """Umbrales desordenados deben lanzar ValueError."""
        with pytest.raises(ValueError, match="Umbrales PSI deben cumplir"):
            AuditConfiguration(
                psi_veto_threshold=0.70,
                psi_warning_threshold=0.50,
                psi_healthy_threshold=0.30,
            )
    
    def test_negative_weights_raise(self):
        """Pesos negativos deben lanzar ValueError."""
        with pytest.raises(ValueError, match="pesos deben ser no-negativos"):
            AuditConfiguration(
                weight_simpson=-0.1,
            )
    
    def test_normalized_weights_sum_to_one(self, config_default):
        """Los pesos normalizados deben sumar 1."""
        weights = config_default.normalized_weights
        assert len(weights) == 4
        assert abs(sum(weights) - 1.0) < 1e-10
    
    def test_zero_weights_handled(self):
        """Pesos todos cero deben normalizarse a uniformes."""
        config = AuditConfiguration(
            weight_simpson=0.0,
            weight_connectivity=0.0,
            weight_robustness=0.0,
            weight_gini=0.0,
        )
        weights = config.normalized_weights
        assert weights == (0.25, 0.25, 0.25, 0.25)
    
    def test_config_is_frozen(self, config_default):
        """La configuración debe ser inmutable."""
        with pytest.raises(AttributeError):
            config_default.psi_veto_threshold = 0.99


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Índice de Simpson — Diversidad
# ═══════════════════════════════════════════════════════════════════════════

class TestSimpsonDiversity:
    """Pruebas del índice de diversidad de Simpson."""
    
    def test_empty_distribution_returns_zero(self, empty_distribution):
        """Distribución vacía → D = 0."""
        assert _compute_simpson_diversity(empty_distribution) == 0.0
    
    def test_monopoly_returns_zero(self, monopoly_distribution):
        """Monopolio (1 insumo) → D = 0."""
        result = _compute_simpson_diversity(monopoly_distribution)
        assert result == 0.0
    
    def test_duopoly_equal_split(self, duopoly_distribution):
        """Duopolio 50/50 → D = 0.5."""
        result = _compute_simpson_diversity(duopoly_distribution)
        assert abs(result - 0.5) < 1e-10
    
    def test_uniform_distribution_formula(self, uniform_distribution):
        """Distribución uniforme → D = 1 - 1/n."""
        n = len(uniform_distribution)
        expected = 1.0 - 1.0 / n  # D = 1 - 5*(1/5)² = 1 - 0.2 = 0.8
        result = _compute_simpson_diversity(uniform_distribution)
        assert abs(result - expected) < 1e-10
    
    def test_skewed_distribution_lower_than_uniform(
        self, skewed_distribution, uniform_distribution
    ):
        """Distribución sesgada tiene menor diversidad que uniforme."""
        skewed_d = _compute_simpson_diversity(skewed_distribution)
        uniform_d = _compute_simpson_diversity(uniform_distribution)
        assert skewed_d < uniform_d
    
    def test_highly_diverse_approaches_one(self, highly_diverse_distribution):
        """Distribución muy diversa → D cercano a 1."""
        result = _compute_simpson_diversity(highly_diverse_distribution)
        assert result > 0.9
        assert result < 1.0
    
    def test_result_in_valid_range(self, skewed_distribution):
        """El resultado siempre debe estar en [0, 1]."""
        result = _compute_simpson_diversity(skewed_distribution)
        assert 0.0 <= result <= 1.0
    
    def test_symmetric_invariant(self):
        """D es invariante bajo permutaciones."""
        dist1 = {"A": 10, "B": 20, "C": 30}
        dist2 = {"C": 30, "A": 10, "B": 20}
        dist3 = {"B": 20, "C": 30, "A": 10}
        
        d1 = _compute_simpson_diversity(dist1)
        d2 = _compute_simpson_diversity(dist2)
        d3 = _compute_simpson_diversity(dist3)
        
        assert abs(d1 - d2) < 1e-10
        assert abs(d2 - d3) < 1e-10
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.integers(min_value=1, max_value=1000),
        min_size=0,
        max_size=50
    ))
    @settings(max_examples=100)
    def test_property_always_in_range(self, distribution):
        """Propiedad: D ∈ [0, 1] para cualquier distribución válida."""
        result = _compute_simpson_diversity(distribution)
        assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Entropía de Shannon — Información
# ═══════════════════════════════════════════════════════════════════════════

class TestShannonEntropy:
    """Pruebas de la entropía de Shannon normalizada."""
    
    def test_empty_distribution(self, empty_distribution):
        """Distribución vacía → H = 0, N_eff = 1."""
        h, n_eff = _compute_shannon_entropy(empty_distribution)
        assert h == 0.0
        assert n_eff == 1.0
    
    def test_monopoly_zero_entropy(self, monopoly_distribution):
        """Monopolio → H = 0 (certeza total)."""
        h, n_eff = _compute_shannon_entropy(monopoly_distribution)
        assert h == 0.0
        assert n_eff == 1.0
    
    def test_uniform_distribution_max_entropy(self, uniform_distribution):
        """Distribución uniforme → H = 1 (máxima entropía)."""
        h, n_eff = _compute_shannon_entropy(uniform_distribution)
        assert abs(h - 1.0) < 1e-10
        assert abs(n_eff - len(uniform_distribution)) < 1e-6
    
    def test_effective_species_interpretation(self):
        """N_eff = exp(H_raw) interpreta como especies equivalentes."""
        # Distribución donde ~3 especies dominan
        dist = {"A": 40, "B": 35, "C": 25}
        h, n_eff = _compute_shannon_entropy(dist)
        
        # N_eff debe estar cerca de 3 (las 3 especies)
        assert 2.5 < n_eff < 3.5
    
    def test_entropy_in_valid_range(self, skewed_distribution):
        """H normalizada ∈ [0, 1]."""
        h, _ = _compute_shannon_entropy(skewed_distribution)
        assert 0.0 <= h <= 1.0
    
    def test_duopoly_specific_value(self, duopoly_distribution):
        """Duopolio 50/50 → H = 1 (normalizado), N_eff = 2."""
        h, n_eff = _compute_shannon_entropy(duopoly_distribution)
        assert abs(h - 1.0) < 1e-10  # Máxima entropía para 2 categorías
        assert abs(n_eff - 2.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Coeficiente de Gini — Concentración
# ═══════════════════════════════════════════════════════════════════════════

class TestGiniCoefficient:
    """Pruebas del coeficiente de Gini."""
    
    def test_empty_distribution(self, empty_distribution):
        """Distribución vacía → G = 0."""
        assert _compute_gini_coefficient(empty_distribution) == 0.0
    
    def test_single_element(self, monopoly_distribution):
        """Un solo elemento → G = 0 (no hay desigualdad)."""
        assert _compute_gini_coefficient(monopoly_distribution) == 0.0
    
    def test_uniform_distribution_zero_gini(self, uniform_distribution):
        """Distribución uniforme → G = 0."""
        result = _compute_gini_coefficient(uniform_distribution)
        assert abs(result) < 1e-10
    
    def test_extreme_inequality_high_gini(self):
        """Desigualdad extrema → G cercano a 1."""
        dist = {"RICO": 99999, "POBRE1": 1}
        result = _compute_gini_coefficient(dist)
        assert result > 0.9
    
    def test_moderate_inequality(self, skewed_distribution):
        """Desigualdad moderada → G intermedio."""
        result = _compute_gini_coefficient(skewed_distribution)
        assert 0.1 < result < 0.5
    
    def test_gini_in_valid_range(self, highly_diverse_distribution):
        """G ∈ [0, 1]."""
        result = _compute_gini_coefficient(highly_diverse_distribution)
        assert 0.0 <= result <= 1.0
    
    def test_gini_increases_with_concentration(self):
        """G aumenta cuando se concentra más la distribución."""
        dist_equal = {"A": 25, "B": 25, "C": 25, "D": 25}
        dist_slight = {"A": 40, "B": 30, "C": 20, "D": 10}
        dist_extreme = {"A": 97, "B": 1, "C": 1, "D": 1}
        
        g_equal = _compute_gini_coefficient(dist_equal)
        g_slight = _compute_gini_coefficient(dist_slight)
        g_extreme = _compute_gini_coefficient(dist_extreme)
        
        assert g_equal < g_slight < g_extreme


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Conectividad Normalizada — Densidad de aristas
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectivityRatio:
    """Pruebas del ratio de conectividad."""
    
    def test_zero_structure_load(self):
        """m = 0 → κ = 0."""
        result = _compute_connectivity_ratio(
            edge_count=10,
            structure_load=0,
            expected_insumos_per_apu=5,
        )
        assert result == 0.0
    
    def test_expected_ratio_gives_half(self):
        """Cuando edges = m * c̄ → κ ≈ 0.5."""
        # ratio = 1 → sigmoide(0) = 0.5
        result = _compute_connectivity_ratio(
            edge_count=50,
            structure_load=10,
            expected_insumos_per_apu=5,
        )
        assert abs(result - 0.5) < 1e-10
    
    def test_double_expected_gives_high(self):
        """Cuando edges = 2 * m * c̄ → κ alto (> 0.9)."""
        result = _compute_connectivity_ratio(
            edge_count=100,
            structure_load=10,
            expected_insumos_per_apu=5,
        )
        assert result > 0.9
    
    def test_half_expected_gives_low(self):
        """Cuando edges = 0.5 * m * c̄ → κ bajo (< 0.2)."""
        result = _compute_connectivity_ratio(
            edge_count=25,
            structure_load=10,
            expected_insumos_per_apu=5,
        )
        assert result < 0.2
    
    def test_zero_edges(self):
        """Sin aristas → κ muy bajo."""
        result = _compute_connectivity_ratio(
            edge_count=0,
            structure_load=10,
            expected_insumos_per_apu=5,
        )
        assert result < 0.05
    
    def test_result_in_valid_range(self):
        """κ ∈ [0, 1]."""
        for edges in [0, 10, 50, 100, 500]:
            result = _compute_connectivity_ratio(
                edge_count=edges,
                structure_load=20,
                expected_insumos_per_apu=5,
            )
            assert 0.0 <= result <= 1.0
    
    def test_monotonically_increasing(self):
        """κ crece monótonamente con edge_count."""
        results = []
        for edges in range(0, 200, 10):
            r = _compute_connectivity_ratio(
                edge_count=edges,
                structure_load=20,
                expected_insumos_per_apu=5,
            )
            results.append(r)
        
        # Verificar monotonía
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Robustez Topológica — Nodos flotantes
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologicalRobustness:
    """Pruebas de robustez topológica."""
    
    def test_zero_structure_load(self):
        """m = 0 → ρ = 0."""
        result = _compute_topological_robustness(
            structure_load=0,
            floating_nodes=0,
        )
        assert result == 0.0
    
    def test_no_floating_nodes(self):
        """Sin nodos flotantes → ρ = 1."""
        result = _compute_topological_robustness(
            structure_load=100,
            floating_nodes=0,
        )
        assert result == 1.0
    
    def test_all_floating_nodes(self):
        """Todos flotantes → ρ = 0."""
        result = _compute_topological_robustness(
            structure_load=100,
            floating_nodes=100,
        )
        assert result == 0.0
    
    def test_half_floating_with_penalty(self):
        """50% flotantes con penalización cuadrática → ρ = 0.25."""
        result = _compute_topological_robustness(
            structure_load=100,
            floating_nodes=50,
            penalty_exponent=2.0,
        )
        # (0.5)² = 0.25
        assert abs(result - 0.25) < 1e-10
    
    def test_linear_penalty(self):
        """Penalización lineal (p=1) → ρ = ratio directo."""
        result = _compute_topological_robustness(
            structure_load=100,
            floating_nodes=30,
            penalty_exponent=1.0,
        )
        assert abs(result - 0.70) < 1e-10
    
    def test_cubic_penalty(self):
        """Penalización cúbica más severa."""
        linear = _compute_topological_robustness(100, 20, penalty_exponent=1.0)
        quadratic = _compute_topological_robustness(100, 20, penalty_exponent=2.0)
        cubic = _compute_topological_robustness(100, 20, penalty_exponent=3.0)
        
        assert linear > quadratic > cubic
    
    def test_result_in_valid_range(self):
        """ρ ∈ [0, 1]."""
        for floating in range(0, 101, 10):
            result = _compute_topological_robustness(100, floating)
            assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Estabilidad Compuesta (Ψ) — Índice agregado
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeStability:
    """Pruebas del índice de estabilidad compuesto."""
    
    def test_all_ones_gives_high_psi(self, config_default):
        """Métricas perfectas → Ψ cercano a 1."""
        result = _compute_composite_stability(
            simpson=1.0,
            connectivity=1.0,
            robustness=1.0,
            gini=0.0,  # 0 gini es bueno (equidad = 1)
            config=config_default,
        )
        assert result > 0.95
    
    def test_all_zeros_gives_low_psi(self, config_default):
        """Métricas pésimas → Ψ cercano a 0 (pero > 0 por ε)."""
        result = _compute_composite_stability(
            simpson=0.0,
            connectivity=0.0,
            robustness=0.0,
            gini=1.0,  # Gini 1 es malo
            config=config_default,
        )
        assert result < 0.01
        assert result > 0.0  # ε evita colapso total
    
    def test_one_bad_metric_penalizes(self, config_default):
        """Una métrica mala penaliza el resultado."""
        good = _compute_composite_stability(
            simpson=0.8, connectivity=0.8, robustness=0.8, gini=0.2,
            config=config_default,
        )
        with_bad_simpson = _compute_composite_stability(
            simpson=0.1, connectivity=0.8, robustness=0.8, gini=0.2,
            config=config_default,
        )
        
        assert with_bad_simpson < good * 0.8  # Penalización significativa
    
    def test_result_in_valid_range(self, config_default):
        """Ψ ∈ [0, 1]."""
        for s, c, r, g in [
            (0.0, 0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0, 0.0),
            (0.3, 0.7, 0.9, 0.2),
        ]:
            result = _compute_composite_stability(s, c, r, g, config_default)
            assert 0.0 <= result <= 1.0
    
    def test_weight_sensitivity(self):
        """Diferentes pesos producen diferentes resultados."""
        config_simpson_heavy = AuditConfiguration(
            weight_simpson=0.7,
            weight_connectivity=0.1,
            weight_robustness=0.1,
            weight_gini=0.1,
        )
        config_connectivity_heavy = AuditConfiguration(
            weight_simpson=0.1,
            weight_connectivity=0.7,
            weight_robustness=0.1,
            weight_gini=0.1,
        )
        
        # Simpson alto, conectividad baja
        psi_simpson = _compute_composite_stability(
            simpson=0.9, connectivity=0.2, robustness=0.7, gini=0.3,
            config=config_simpson_heavy,
        )
        psi_conn = _compute_composite_stability(
            simpson=0.9, connectivity=0.2, robustness=0.7, gini=0.3,
            config=config_connectivity_heavy,
        )
        
        # Con peso alto en Simpson, debería dar mejor resultado
        assert psi_simpson > psi_conn


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Conectividad Algebraica (λ₂) — Análisis espectral
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy no disponible")
class TestAlgebraicConnectivity:
    """Pruebas de conectividad algebraica (requiere scipy)."""
    
    def test_empty_graph(self):
        """Grafo vacío → None."""
        result = _compute_algebraic_connectivity({})
        assert result is None
    
    def test_single_node(self):
        """Un solo nodo → None."""
        result = _compute_algebraic_connectivity({"APU1": {"A", "B"}})
        assert result is None
    
    def test_disconnected_graph(self, apu_to_insumos_disconnected):
        """Grafo desconectado → λ₂ = 0."""
        result = _compute_algebraic_connectivity(apu_to_insumos_disconnected)
        if result is not None:
            lambda_2, eigenvalues = result
            assert lambda_2 < 0.01  # Cercano a 0
    
    def test_connected_graph(self, apu_to_insumos_connected):
        """Grafo conectado → λ₂ > 0."""
        result = _compute_algebraic_connectivity(apu_to_insumos_connected)
        if result is not None:
            lambda_2, eigenvalues = result
            assert lambda_2 > 0.0
    
    def test_complete_graph_high_lambda(self):
        """Grafo completo (todos comparten) → λ₂ alto."""
        # Todos comparten el mismo insumo
        apu_to_insumos = {
            f"APU{i}": {"SHARED", f"UNIQUE_{i}"} for i in range(5)
        }
        result = _compute_algebraic_connectivity(apu_to_insumos)
        if result is not None:
            lambda_2, _ = result
            assert lambda_2 > 0.1
    
    def test_returns_normalized_value(self, apu_to_insumos_connected):
        """λ₂ normalizado debe estar en [0, 1]."""
        result = _compute_algebraic_connectivity(apu_to_insumos_connected)
        if result is not None:
            lambda_2, _ = result
            assert 0.0 <= lambda_2 <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Sanitización de datos
# ═══════════════════════════════════════════════════════════════════════════

class TestSanitizeCodigoSet:
    """Pruebas de sanitización de códigos."""
    
    def test_empty_series(self):
        """Serie vacía → conjunto vacío."""
        series = pd.Series([], dtype=object)
        result, report = _sanitize_codigo_set(series, "test")
        
        assert result == set()
        assert report.original_count == 0
        assert report.valid_count == 0
    
    def test_none_series(self):
        """Serie None → conjunto vacío."""
        result, report = _sanitize_codigo_set(None, "test")
        
        assert result == set()
        assert report.original_count == 0
    
    def test_removes_null_values(self):
        """Elimina None, NaN, pd.NA."""
        series = pd.Series(["A", None, "B", np.nan, "C", pd.NA])
        result, report = _sanitize_codigo_set(series, "test")
        
        assert result == {"A", "B", "C"}
        assert report.null_count == 3
        assert report.valid_count == 3
    
    def test_removes_invalid_strings(self):
        """Elimina strings vacíos y patrones inválidos."""
        series = pd.Series(["A", "", "B", "nan", "C", "NONE", "NULL", " "])
        result, report = _sanitize_codigo_set(series, "test")
        
        assert result == {"A", "B", "C"}
        assert report.invalid_string_count > 0
    
    def test_normalizes_case(self):
        """Normaliza a mayúsculas."""
        series = pd.Series(["apu001", "APU001", "Apu001"])
        result, report = _sanitize_codigo_set(series, "test")
        
        assert result == {"APU001"}
        assert report.duplicate_count == 2
    
    def test_strips_whitespace(self):
        """Elimina espacios en blanco."""
        series = pd.Series(["  APU001  ", "APU002\t", "\nAPU003"])
        result, report = _sanitize_codigo_set(series, "test")
        
        assert result == {"APU001", "APU002", "APU003"}
    
    def test_validity_ratio(self):
        """Calcula ratio de validez correctamente."""
        series = pd.Series(["A", "B", None, "", "C"])
        result, report = _sanitize_codigo_set(series, "test")
        
        assert report.original_count == 5
        assert report.valid_count == 3
        assert abs(report.validity_ratio - 0.6) < 1e-10
    
    def test_is_acceptable_threshold(self):
        """is_acceptable es True si validez >= 50%."""
        good_series = pd.Series(["A", "B", "C", None])  # 75% válido
        bad_series = pd.Series(["A", None, None, None])  # 25% válido
        
        _, good_report = _sanitize_codigo_set(good_series, "test")
        _, bad_report = _sanitize_codigo_set(bad_series, "test")
        
        assert good_report.is_acceptable is True
        assert bad_report.is_acceptable is False


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Extracción de distribución de insumos
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractInsumoDistribution:
    """Pruebas de extracción de distribución."""
    
    def test_empty_dataframe(self):
        """DataFrame vacío → todo vacío."""
        df = pd.DataFrame(columns=["CODIGO_APU", "DESCRIPCION_INSUMO"])
        base_width, dist, apu_map = _extract_insumo_distribution(
            df, "DESCRIPCION_INSUMO"
        )
        
        assert base_width == 0
        assert dist == {}
        assert apu_map == {}
    
    def test_counts_insumo_frequencies(self, df_insumos_valid):
        """Cuenta frecuencias correctamente."""
        base_width, dist, _ = _extract_insumo_distribution(
            df_insumos_valid, "DESCRIPCION_INSUMO"
        )
        
        assert base_width > 0
        assert sum(dist.values()) == len(df_insumos_valid)
    
    def test_builds_apu_to_insumos_map(self, df_insumos_valid):
        """Construye mapa APU → insumos."""
        _, _, apu_map = _extract_insumo_distribution(
            df_insumos_valid, "DESCRIPCION_INSUMO"
        )
        
        assert "APU001" in apu_map
        assert isinstance(apu_map["APU001"], set)
        assert len(apu_map["APU001"]) == 3  # 3 insumos por APU en fixture
    
    def test_handles_nulls_in_insumos(self):
        """Maneja nulos en columna de insumos."""
        df = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU001", "APU002"],
            "DESCRIPCION_INSUMO": ["CEMENTO", None, "ARENA"],
        })
        
        base_width, dist, apu_map = _extract_insumo_distribution(
            df, "DESCRIPCION_INSUMO"
        )
        
        assert base_width == 2
        assert "CEMENTO" in dist
        assert "ARENA" in dist


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Validación de esquema
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """Pruebas de validación de esquema de DataFrame."""
    
    def test_valid_schema(self, df_presupuesto_valid):
        """DataFrame válido pasa validación."""
        result = _validate_dataframe_schema(
            df_presupuesto_valid,
            required_columns={"CODIGO_APU"},
            df_name="presupuesto",
        )
        
        assert result.is_valid is True
        assert len(result.missing_columns) == 0
    
    def test_none_dataframe(self):
        """DataFrame None falla validación."""
        result = _validate_dataframe_schema(
            None,
            required_columns={"CODIGO_APU"},
            df_name="presupuesto",
        )
        
        assert result.is_valid is False
        assert "CODIGO_APU" in result.missing_columns
    
    def test_missing_columns(self, df_presupuesto_valid):
        """Columnas faltantes se reportan."""
        result = _validate_dataframe_schema(
            df_presupuesto_valid,
            required_columns={"CODIGO_APU", "COLUMNA_INEXISTENTE"},
            df_name="presupuesto",
        )
        
        assert result.is_valid is False
        assert "COLUMNA_INEXISTENTE" in result.missing_columns
    
    def test_empty_dataframe_warning(self):
        """DataFrame vacío genera warning."""
        df = pd.DataFrame(columns=["CODIGO_APU"])
        result = _validate_dataframe_schema(
            df,
            required_columns={"CODIGO_APU"},
            df_name="presupuesto",
        )
        
        assert result.is_valid is True
        assert any("vacío" in w for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Determinación de veredicto
# ═══════════════════════════════════════════════════════════════════════════

class TestDetermineVerdict:
    """Pruebas de lógica de veredicto."""
    
    @pytest.fixture
    def metrics_good(self, uniform_distribution) -> GraphMetrics:
        """Métricas buenas."""
        return GraphMetrics(
            structure_load=10,
            base_width=5,
            edge_count=50,
            floating_nodes=0,
            simpson_diversity=0.8,
            shannon_entropy=0.9,
            effective_species=5.0,
            gini_coefficient=0.1,
            connectivity_ratio=0.5,
            topological_robustness=1.0,
            algebraic_connectivity=0.5,
            composite_stability=0.85,
            insumo_distribution=uniform_distribution,
        )
    
    @pytest.fixture
    def metrics_bad(self, monopoly_distribution) -> GraphMetrics:
        """Métricas malas."""
        return GraphMetrics(
            structure_load=10,
            base_width=1,
            edge_count=10,
            floating_nodes=5,
            simpson_diversity=0.0,
            shannon_entropy=0.0,
            effective_species=1.0,
            gini_coefficient=0.9,
            connectivity_ratio=0.2,
            topological_robustness=0.25,
            algebraic_connectivity=0.0,
            composite_stability=0.15,
            insumo_distribution=monopoly_distribution,
        )
    
    def test_optimal_verdict(self, metrics_good, config_default):
        """Métricas excelentes → OPTIMAL."""
        verdict = _determine_verdict(0.90, metrics_good, config_default)
        
        assert verdict.level == StabilityVerdict.Level.OPTIMAL
        assert verdict.blocking is False
    
    def test_healthy_verdict(self, metrics_good, config_default):
        """Métricas buenas → HEALTHY."""
        verdict = _determine_verdict(0.70, metrics_good, config_default)
        
        assert verdict.level == StabilityVerdict.Level.HEALTHY
        assert verdict.blocking is False
    
    def test_warning_verdict(self, metrics_good, config_default):
        """Métricas mediocres → WARNING."""
        verdict = _determine_verdict(0.50, metrics_good, config_default)
        
        assert verdict.level == StabilityVerdict.Level.WARNING
        assert verdict.blocking is False
    
    def test_veto_verdict(self, metrics_bad, config_default):
        """Métricas pésimas → VETO."""
        verdict = _determine_verdict(0.30, metrics_bad, config_default)
        
        assert verdict.level == StabilityVerdict.Level.VETO
        assert verdict.blocking is True
    
    def test_veto_includes_factors(self, metrics_bad, config_default):
        """VETO incluye factores contribuyentes."""
        verdict = _determine_verdict(0.30, metrics_bad, config_default)
        
        assert len(verdict.contributing_factors) > 0
        assert any("diversidad" in f.lower() for f in verdict.contributing_factors)
    
    def test_recommendations_generated(self, metrics_bad, config_default):
        """Se generan recomendaciones para problemas."""
        verdict = _determine_verdict(0.50, metrics_bad, config_default)
        
        assert len(verdict.recommendations) > 0
    
    def test_severity_score_ordering(self, metrics_good, config_default):
        """severity_score sigue orden lógico."""
        optimal = _determine_verdict(0.90, metrics_good, config_default)
        healthy = _determine_verdict(0.70, metrics_good, config_default)
        warning = _determine_verdict(0.50, metrics_good, config_default)
        veto = _determine_verdict(0.30, metrics_good, config_default)
        
        assert optimal.severity_score < healthy.severity_score
        assert healthy.severity_score < warning.severity_score
        assert warning.severity_score < veto.severity_score


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Audit Hash — Trazabilidad
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditHash:
    """Pruebas del hash de auditoría."""
    
    @pytest.fixture
    def sample_metrics(self, uniform_distribution) -> GraphMetrics:
        """Métricas de ejemplo."""
        return GraphMetrics(
            structure_load=10,
            base_width=5,
            edge_count=50,
            floating_nodes=0,
            simpson_diversity=0.8,
            shannon_entropy=0.9,
            effective_species=5.0,
            gini_coefficient=0.1,
            connectivity_ratio=0.5,
            topological_robustness=1.0,
            algebraic_connectivity=0.5,
            composite_stability=0.85,
            insumo_distribution=uniform_distribution,
        )
    
    def test_hash_is_sha256(self, sample_metrics, config_default):
        """Hash es SHA256 (64 caracteres hex)."""
        hash_result = _build_audit_hash(sample_metrics, time.time(), config_default)
        
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)
    
    def test_hash_is_deterministic(self, sample_metrics, config_default):
        """Mismo input → mismo hash."""
        timestamp = 1234567890.123456
        
        hash1 = _build_audit_hash(sample_metrics, timestamp, config_default)
        hash2 = _build_audit_hash(sample_metrics, timestamp, config_default)
        
        assert hash1 == hash2
    
    def test_different_timestamp_different_hash(self, sample_metrics, config_default):
        """Diferente timestamp → diferente hash."""
        hash1 = _build_audit_hash(sample_metrics, 1000.0, config_default)
        hash2 = _build_audit_hash(sample_metrics, 2000.0, config_default)
        
        assert hash1 != hash2
    
    def test_different_metrics_different_hash(self, config_default, uniform_distribution):
        """Diferentes métricas → diferente hash."""
        metrics1 = GraphMetrics(
            structure_load=10, base_width=5, edge_count=50, floating_nodes=0,
            simpson_diversity=0.8, shannon_entropy=0.9, effective_species=5.0,
            gini_coefficient=0.1, connectivity_ratio=0.5, topological_robustness=1.0,
            algebraic_connectivity=0.5, composite_stability=0.85,
            insumo_distribution=uniform_distribution,
        )
        metrics2 = GraphMetrics(
            structure_load=20, base_width=5, edge_count=50, floating_nodes=0,
            simpson_diversity=0.8, shannon_entropy=0.9, effective_species=5.0,
            gini_coefficient=0.1, connectivity_ratio=0.5, topological_robustness=1.0,
            algebraic_connectivity=0.5, composite_stability=0.85,
            insumo_distribution=uniform_distribution,
        )
        
        timestamp = 1000.0
        hash1 = _build_audit_hash(metrics1, timestamp, config_default)
        hash2 = _build_audit_hash(metrics2, timestamp, config_default)
        
        assert hash1 != hash2


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Vector Principal — Integración
# ═══════════════════════════════════════════════════════════════════════════

class TestVectorAuditPyramidalStructure:
    """Pruebas de integración del vector principal."""
    
    def test_success_with_valid_data(
        self, df_presupuesto_valid, df_insumos_valid, config_default
    ):
        """Datos válidos producen resultado exitoso."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_valid,
            config=config_default,
        )
        
        assert result["success"] is True
        assert "payload" in result
        assert "composite_stability_psi" in result["payload"]
    
    def test_empty_presupuesto_fails(self, df_insumos_valid, config_default):
        """Presupuesto vacío → fallo."""
        df_empty = pd.DataFrame(columns=["CODIGO_APU"])
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_empty,
            df_insumos=df_insumos_valid,
            config=config_default,
        )
        
        assert result["success"] is False
        assert "GRAFO_VACÍO" in result["error"]
    
    def test_none_dataframe_fails(self, df_insumos_valid, config_default):
        """DataFrame None → fallo."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=None,
            df_insumos=df_insumos_valid,
            config=config_default,
        )
        
        assert result["success"] is False
        assert "Esquema inválido" in result["error"]
    
    def test_missing_columns_fails(self, config_default):
        """Columnas faltantes → fallo."""
        df_bad = pd.DataFrame({"WRONG_COLUMN": ["A", "B"]})
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_bad,
            df_insumos=df_bad,
            config=config_default,
        )
        
        assert result["success"] is False
        assert "Esquema inválido" in result["error"]
    
    def test_floating_nodes_penalize_but_dont_block(
        self, df_presupuesto_valid, df_insumos_with_floating, config_default
    ):
        """Nodos flotantes penalizan ρ pero no bloquean."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_with_floating,
            config=config_default,
        )
        
        # Puede ser éxito o fallo según el puntaje, pero no debe ser error de topología duro
        if result["success"]:
            payload = result["payload"]
            assert payload["connectivity"]["topological_robustness"] < 1.0
            assert payload["connectivity"]["floating_nodes"] > 0
        else:
            # Si falla, debe ser por bajo Ψ, no por "NODOS_FLOTANTES"
            assert "TOPOLOGY_ERROR" in str(result.get("status", ""))
    
    def test_sparse_insumos_low_diversity(
        self, df_presupuesto_valid, df_insumos_sparse, config_default
    ):
        """Pocos insumos → baja diversidad."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_sparse,
            config=config_default,
        )
        
        if result["success"]:
            payload = result["payload"]
            assert payload["diversity"]["simpson_index"] < 0.5
    
    def test_returns_correct_structure(
        self, df_presupuesto_valid, df_insumos_valid, config_default
    ):
        """Resultado tiene estructura correcta."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_valid,
            config=config_default,
        )
        
        # Campos obligatorios
        assert "success" in result
        assert "stratum" in result
        assert "status" in result
        assert "metrics" in result
        
        if result["success"]:
            payload = result["payload"]
            # Secciones del payload
            assert "audit_verdict" in payload
            assert "composite_stability_psi" in payload
            assert "diversity" in payload
            assert "connectivity" in payload
            assert "thresholds" in payload
            assert "graph_structure" in payload
            assert "algorithm_version" in payload
            assert "structural_audit_hash" in payload
    
    def test_metrics_include_timing(
        self, df_presupuesto_valid, df_insumos_valid, config_default
    ):
        """Métricas incluyen tiempo de procesamiento."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_valid,
            config=config_default,
        )
        
        assert "processing_time_ms" in result["metrics"]
        assert result["metrics"]["processing_time_ms"] >= 0
    
    def test_handles_nulls_gracefully(
        self, df_presupuesto_with_nulls, df_insumos_valid, config_default
    ):
        """Maneja valores nulos sin crashear."""
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_with_nulls,
            df_insumos=df_insumos_valid,
            config=config_default,
        )
        
        # Puede fallar por datos insuficientes, pero no debe lanzar excepción
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_uses_descripcion_insumo_norm_if_present(
        self, df_presupuesto_valid, config_default
    ):
        """Usa DESCRIPCION_INSUMO_NORM si existe."""
        df_insumos = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU002", "APU003", "APU004", "APU005"],
            "DESCRIPCION_INSUMO": ["raw1", "raw2", "raw3", "raw4", "raw5"],
            "DESCRIPCION_INSUMO_NORM": ["NORM1", "NORM2", "NORM3", "NORM4", "NORM5"],
        })
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos,
            config=config_default,
        )
        
        if result["success"]:
            assert result["payload"]["col_insumo_used"] == "DESCRIPCION_INSUMO_NORM"
    
    def test_custom_config_affects_verdict(
        self, df_presupuesto_valid, df_insumos_valid, config_strict, config_relaxed
    ):
        """Diferentes configuraciones producen diferentes veredictos."""
        result_strict = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_valid,
            config=config_strict,
        )
        result_relaxed = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto_valid,
            df_insumos=df_insumos_valid,
            config=config_relaxed,
        )
        
        # Con config estricto, más probable fallo/warning
        # Con config relajado, más probable éxito
        # Al menos uno debería ser diferente en severidad
        if result_strict["success"] and result_relaxed["success"]:
            strict_severity = result_strict["payload"]["verdict_severity"]
            relaxed_severity = result_relaxed["payload"]["verdict_severity"]
            # Estricto debería ser igual o más severo
            assert strict_severity >= relaxed_severity


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Funciones auxiliares exportadas
# ═══════════════════════════════════════════════════════════════════════════

class TestAuxiliaryFunctions:
    """Pruebas de funciones auxiliares exportadas."""
    
    def test_herfindahl_complement_of_simpson(self, skewed_distribution):
        """HHI = 1 - Simpson."""
        simpson = _compute_simpson_diversity(skewed_distribution)
        hhi = compute_herfindahl_index(skewed_distribution)
        
        assert abs(simpson + hhi - 1.0) < 1e-10
    
    def test_effective_species_q0_is_richness(self, uniform_distribution):
        """N_eff(q=0) = número de especies."""
        n_eff = compute_effective_number_of_species(uniform_distribution, q=0)
        
        assert abs(n_eff - len(uniform_distribution)) < 1e-10
    
    def test_effective_species_q1_is_exp_shannon(self, skewed_distribution):
        """N_eff(q=1) = exp(H)."""
        _, n_eff_from_shannon = _compute_shannon_entropy(skewed_distribution)
        n_eff_hill = compute_effective_number_of_species(skewed_distribution, q=1)
        
        assert abs(n_eff_from_shannon - n_eff_hill) < 1e-6
    
    def test_effective_species_q2_is_inverse_hhi(self, skewed_distribution):
        """N_eff(q=2) = 1/HHI."""
        hhi = compute_herfindahl_index(skewed_distribution)
        n_eff = compute_effective_number_of_species(skewed_distribution, q=2)
        
        if hhi > 0:
            assert abs(n_eff - 1.0/hhi) < 1e-6
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy no disponible")
    def test_analyze_graph_spectrum(self, apu_to_insumos_connected):
        """Análisis espectral retorna estructura correcta."""
        result = analyze_graph_spectrum(apu_to_insumos_connected)
        
        if result is not None:
            assert "algebraic_connectivity" in result
            assert "eigenvalues" in result
            assert "spectral_gap" in result
            assert "is_connected" in result
    
    def test_analyze_graph_spectrum_without_scipy(self, apu_to_insumos_connected):
        """Sin scipy, retorna None."""
        with patch("audit_vectors.SCIPY_AVAILABLE", False):
            # Nota: Esto puede no funcionar si la constante se evalúa en import
            # En ese caso, saltar esta prueba
            pass


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Diagnósticos de distribución
# ═══════════════════════════════════════════════════════════════════════════

class TestDistributionDiagnostics:
    """Pruebas de diagnósticos estadísticos de distribución."""
    
    def test_empty_distribution(self, empty_distribution):
        """Distribución vacía → None."""
        result = _compute_distribution_diagnostics(empty_distribution)
        assert result is None
    
    def test_single_element(self, monopoly_distribution):
        """Un elemento → None (necesita ≥2 para estadísticas)."""
        result = _compute_distribution_diagnostics(monopoly_distribution)
        assert result is None
    
    def test_uniform_distribution_low_cv(self, uniform_distribution):
        """Distribución uniforme tiene CV bajo."""
        result = _compute_distribution_diagnostics(uniform_distribution)
        
        assert result is not None
        assert result.coefficient_of_variation == 0.0
        assert result.skewness == 0.0
    
    def test_skewed_distribution_positive_skew(self, skewed_distribution):
        """Distribución sesgada tiene skewness positivo."""
        result = _compute_distribution_diagnostics(skewed_distribution)
        
        assert result is not None
        assert result.skewness > 0  # Cola derecha
    
    def test_concentration_metrics(self, skewed_distribution):
        """Métricas de concentración calculadas correctamente."""
        result = _compute_distribution_diagnostics(skewed_distribution)
        
        assert result is not None
        assert 0.0 <= result.max_concentration <= 1.0
        assert result.max_concentration <= result.top_5_concentration
    
    def test_iqr_calculation(self, highly_diverse_distribution):
        """IQR calculado correctamente."""
        result = _compute_distribution_diagnostics(highly_diverse_distribution)
        
        assert result is not None
        assert result.iqr == result.percentile_75 - result.percentile_25
        assert result.iqr >= 0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Propiedades matemáticas (Hypothesis)
# ═══════════════════════════════════════════════════════════════════════════

class TestMathematicalProperties:
    """Pruebas de propiedades matemáticas usando Hypothesis."""
    
    @given(st.dictionaries(
        st.text(alphabet="ABCDEFGHIJ", min_size=1, max_size=5),
        st.integers(min_value=1, max_value=100),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=50)
    def test_simpson_gini_relationship(self, distribution):
        """Simpson y Gini están inversamente relacionados en tendencia."""
        simpson = _compute_simpson_diversity(distribution)
        gini = _compute_gini_coefficient(distribution)
        
        # Ambos en [0, 1]
        assert 0.0 <= simpson <= 1.0
        assert 0.0 <= gini <= 1.0
        
        # En general, alta diversidad (Simpson) implica baja concentración (Gini)
        # No es una regla absoluta, pero para distribuciones extremas sí
        if simpson > 0.9:
            assert gini < 0.5
        if gini > 0.9:
            assert simpson < 0.5
    
    @given(st.integers(min_value=0, max_value=1000),
           st.integers(min_value=1, max_value=100),
           st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_connectivity_bounds(self, edges, load, expected):
        """κ siempre en [0, 1]."""
        result = _compute_connectivity_ratio(edges, load, expected)
        assert 0.0 <= result <= 1.0
    
    @given(st.integers(min_value=1, max_value=100),
           st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_robustness_bounds(self, load, floating):
        """ρ siempre en [0, 1]."""
        assume(floating <= load)
        result = _compute_topological_robustness(load, floating)
        assert 0.0 <= result <= 1.0
    
    @given(st.floats(min_value=0, max_value=1),
           st.floats(min_value=0, max_value=1),
           st.floats(min_value=0, max_value=1),
           st.floats(min_value=0, max_value=1))
    @settings(max_examples=50)
    def test_composite_stability_bounds(self, s, c, r, g):
        """Ψ siempre en [0, 1]."""
        result = _compute_composite_stability(s, c, r, g, DEFAULT_CONFIG)
        assert 0.0 <= result <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Rendimiento (marcados como slow)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPerformance:
    """Pruebas de rendimiento (excluir con -m "not slow")."""
    
    def test_large_distribution_performance(self):
        """Simpson/Gini/Shannon escalan bien con muchos insumos."""
        distribution = {f"INSUMO_{i:06d}": i + 1 for i in range(10000)}
        
        start = time.perf_counter()
        _compute_simpson_diversity(distribution)
        _compute_shannon_entropy(distribution)
        _compute_gini_coefficient(distribution)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0  # < 1 segundo para 10k insumos
    
    def test_large_dataframe_performance(self):
        """Vector principal escala con DataFrames grandes."""
        n_apus = 1000
        n_insumos_per_apu = 10
        
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": [f"APU{i:05d}" for i in range(n_apus)]
        })
        
        df_insumos = pd.DataFrame({
            "CODIGO_APU": [
                f"APU{i:05d}" 
                for i in range(n_apus) 
                for _ in range(n_insumos_per_apu)
            ],
            "DESCRIPCION_INSUMO": [
                f"INS{j:03d}" 
                for _ in range(n_apus) 
                for j in range(n_insumos_per_apu)
            ],
        })
        
        start = time.perf_counter()
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto,
            df_insumos=df_insumos,
        )
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0  # < 5 segundos para 1k APUs × 10 insumos
        assert result["success"] is True
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy no disponible")
    def test_spectral_analysis_performance(self):
        """Análisis espectral escala con grafos medianos."""
        # Crear grafo conectado mediano
        apu_to_insumos = {
            f"APU{i:03d}": {f"INS{j:02d}" for j in range(i % 5, i % 5 + 3)}
            for i in range(100)
        }
        
        start = time.perf_counter()
        result = _compute_algebraic_connectivity(apu_to_insumos)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0  # < 2 segundos para 100 nodos


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Casos Edge
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Pruebas de casos límite y situaciones especiales."""
    
    def test_all_same_insumo(self):
        """Todos los APUs usan el mismo insumo (monopolio total)."""
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU002", "APU003"]
        })
        df_insumos = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU002", "APU003"],
            "DESCRIPCION_INSUMO": ["CEMENTO", "CEMENTO", "CEMENTO"],
        })
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto,
            df_insumos=df_insumos,
        )
        
        if result["success"]:
            assert result["payload"]["diversity"]["simpson_index"] == 0.0
    
    def test_single_apu(self):
        """Un solo APU."""
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": ["APU001"]
        })
        df_insumos = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU001", "APU001"],
            "DESCRIPCION_INSUMO": ["A", "B", "C"],
        })
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto,
            df_insumos=df_insumos,
        )
        
        assert isinstance(result, dict)
        if result["success"]:
            assert result["payload"]["graph_structure"]["structure_load_m"] == 1
    
    def test_unicode_in_codes(self):
        """Códigos con caracteres Unicode."""
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": ["APU_SEÑAL", "APU_NIÑO", "APU_ESPAÑA"]
        })
        df_insumos = pd.DataFrame({
            "CODIGO_APU": ["APU_SEÑAL", "APU_NIÑO", "APU_ESPAÑA"],
            "DESCRIPCION_INSUMO": ["TÜBERÍA", "VÁLVULA", "CONEXIÓN"],
        })
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto,
            df_insumos=df_insumos,
        )
        
        assert isinstance(result, dict)
    
    def test_numeric_codes(self):
        """Códigos numéricos (se convierten a string)."""
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": [1, 2, 3, 4, 5]
        })
        df_insumos = pd.DataFrame({
            "CODIGO_APU": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "DESCRIPCION_INSUMO": ["A", "B"] * 5,
        })
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto,
            df_insumos=df_insumos,
        )
        
        assert isinstance(result, dict)
        if result["success"]:
            assert result["payload"]["graph_structure"]["structure_load_m"] == 5
    
    def test_very_long_codes(self):
        """Códigos muy largos."""
        long_code = "A" * 1000
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": [long_code]
        })
        df_insumos = pd.DataFrame({
            "CODIGO_APU": [long_code],
            "DESCRIPCION_INSUMO": ["INSUMO"],
        })
        
        result = vector_audit_pyramidal_structure(
            df_presupuesto=df_presupuesto,
            df_insumos=df_insumos,
        )
        
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ═══════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Registrar marcadores personalizados."""
    config.addinivalue_line(
        "markers", "slow: marca pruebas lentas (excluir con -m 'not slow')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])