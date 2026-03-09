"""
Suite de Pruebas Exhaustivas para el Módulo Business Agent
==========================================================

Esta suite valida exhaustivamente:

1. **Estructuras de Datos Inmutables**
   - Validación de invariantes en construcción
   - Inmutabilidad (frozen dataclasses)
   - Serialización/deserialización

2. **Enumeraciones y Clasificaciones**
   - Parsing de strings
   - Mapeos de severidad y riesgo

3. **Validación de DataFrames**
   - Existencia y no-vacuidad
   - Esquema y tipos de columnas
   - Integridad referencial
   - Detección de outliers

4. **Construcción Topológica**
   - Métricas de Betti
   - Coherencia estructural
   - Cotas de viabilidad

5. **Álgebra de Decisiones**
   - Normalización de vectores
   - Media geométrica ponderada
   - Combinaciones convexas
   - Factores de calidad

6. **Estrategias de Pivote**
   - Monopolio Coberturado
   - Opción de Espera
   - Cuarentena Topológica

7. **Risk Challenger**
   - Auditoría adversarial
   - Emisión de vetos
   - Excepciones laterales
   - Integración con MIC

8. **Composición de Reportes**
   - Álgebra multicriterio
   - Generación de narrativas

9. **BusinessAgent (Pipeline Completo)**
   - Evaluación end-to-end
   - Propagación de errores
   - Casos límite

Convenciones:
─────────────
- test_<componente>_<comportamiento>_<condición>
- Fixtures: <dominio>_<característica>
- Marcadores: @pytest.mark.{unit, integration, algebraic, slow}
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import FrozenInstanceError, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# =============================================================================

from app.business_agent import (
    # Configuración y Constantes
    RiskChallengerThresholds,
    DecisionWeights,
    FinancialDefaults,
    NumericalConstants,
    # Excepciones
    BusinessAgentError,
    ConfigurationError,
    ValidationError,
    TopologicalAnomalyError,
    MICHierarchyViolationError,
    FinancialProjectionError,
    SynthesisError,
    # Enumeraciones
    RiskClassification,
    VetoSeverity,
    PivotType,
    # Estructuras de Datos
    FinancialParameters,
    BettiNumbers,
    TopologicalMetricsBundle,
    ValidationResult,
    VetoRecord,
    LateralExceptionRecord,
    # Componentes
    DataFrameValidator,
    TopologyBuilder,
    DecisionAlgebra,
    ReportComposer,
    RiskChallenger,
    BusinessAgent,
    # Estrategias
    PivotStrategy,
    MonopolioCoberturadoStrategy,
    OpcionEsperaStrategy,
    CuarentenaTopologicaStrategy,
    # Factories
    create_business_agent,
    evaluate_project,
)

# Schemas (para columnas)
try:
    from app.constants import ColumnNames
except ImportError:
    class ColumnNames:
        CODIGO_APU = "codigo_apu"
        DESCRIPCION_APU = "descripcion_apu"
        VALOR_TOTAL = "valor_total"
        DESCRIPCION_INSUMO = "descripcion_insumo"
        CANTIDAD_APU = "cantidad_apu"
        COSTO_INSUMO_EN_APU = "costo_insumo_en_apu"

# Stratum
try:
    from app.schemas import Stratum
except ImportError:
    from enum import IntEnum
    class Stratum(IntEnum):
        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3


# =============================================================================
# CONSTANTES DE PRUEBA
# =============================================================================

class TestConstants:
    """Constantes centralizadas para pruebas."""
    
    # Estabilidad
    STABILITY_CRITICAL: float = 0.50
    STABILITY_LOW: float = 0.70
    STABILITY_SUBOPTIMAL: float = 0.80
    STABILITY_NORMAL: float = 0.90
    STABILITY_EXCELLENT: float = 0.98
    
    # Financieros
    INVESTMENT_SMALL: float = 100_000.0
    INVESTMENT_MEDIUM: float = 1_000_000.0
    INVESTMENT_LARGE: float = 10_000_000.0
    
    # Topológicos
    DEFAULT_N_NODES: int = 50
    DEFAULT_N_EDGES: int = 75
    
    # Tolerancias
    EPSILON: float = 1e-9
    FLOAT_TOLERANCE: float = 1e-6


# =============================================================================
# MARCADORES PERSONALIZADOS
# =============================================================================

def pytest_configure(config):
    """Registra marcadores personalizados."""
    config.addinivalue_line("markers", "unit: Tests unitarios rápidos")
    config.addinivalue_line("markers", "integration: Tests de integración")
    config.addinivalue_line("markers", "algebraic: Tests de propiedades algebraicas")
    config.addinivalue_line("markers", "slow: Tests lentos")
    config.addinivalue_line("markers", "regression: Tests de regresión")


# =============================================================================
# FACTORIES Y HELPERS
# =============================================================================

class DataFrameFactory:
    """Factory para creación de DataFrames de prueba."""
    
    @staticmethod
    def create_budget_df(
        n_rows: int = 10,
        *,
        include_totals: bool = True,
        negative_values: bool = False,
        missing_columns: Optional[List[str]] = None,
        null_ratio: float = 0.0
    ) -> pd.DataFrame:
        """Crea DataFrame de presupuesto."""
        data = {
            ColumnNames.CODIGO_APU: [f"APU-{i:03d}" for i in range(n_rows)],
            ColumnNames.DESCRIPCION_APU: [f"Descripción APU {i}" for i in range(n_rows)],
        }
        
        if include_totals:
            values = np.random.uniform(1000, 100000, n_rows)
            if negative_values:
                values[0] = -1000
            data[ColumnNames.VALOR_TOTAL] = values
        
        df = pd.DataFrame(data)
        
        # Eliminar columnas si se solicita
        if missing_columns:
            for col in missing_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
        
        # Introducir nulos
        if null_ratio > 0:
            mask = np.random.random(n_rows) < null_ratio
            df.loc[mask, ColumnNames.DESCRIPCION_APU] = None
        
        return df
    
    @staticmethod
    def create_detail_df(
        n_rows: int = 50,
        *,
        apu_codes: Optional[List[str]] = None,
        missing_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Crea DataFrame de detalle de APUs."""
        if apu_codes is None:
            apu_codes = [f"APU-{i:03d}" for i in range(10)]
        
        data = {
            ColumnNames.CODIGO_APU: np.random.choice(apu_codes, n_rows),
            ColumnNames.DESCRIPCION_INSUMO: [f"Insumo {i}" for i in range(n_rows)],
            ColumnNames.CANTIDAD_APU: np.random.uniform(1, 100, n_rows),
            ColumnNames.COSTO_INSUMO_EN_APU: np.random.uniform(10, 1000, n_rows),
        }
        
        df = pd.DataFrame(data)
        
        if missing_columns:
            for col in missing_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
        
        return df
    
    @staticmethod
    def create_empty_df() -> pd.DataFrame:
        """Crea DataFrame vacío."""
        return pd.DataFrame()
    
    @staticmethod
    def create_orphan_detail_df(budget_df: pd.DataFrame) -> pd.DataFrame:
        """Crea detalle con códigos huérfanos (no en presupuesto)."""
        orphan_codes = ["ORPHAN-001", "ORPHAN-002", "ORPHAN-003"]
        return DataFrameFactory.create_detail_df(n_rows=20, apu_codes=orphan_codes)


class MockFactory:
    """Factory para creación de mocks."""
    
    @staticmethod
    def create_mock_mic(
        *,
        project_success: bool = True,
        financial_results: Optional[Dict[str, Any]] = None
    ) -> MagicMock:
        """Crea mock de MICRegistry."""
        mic = MagicMock()
        
        default_results = {
            "npv": 500_000.0,
            "irr": 0.15,
            "payback_period": 3.5,
            "var_95": 50_000.0,
            "sharpe_ratio": 1.2,
        }
        
        results = financial_results or default_results
        
        def mock_project_intent(service_name: str, payload: Dict, context: Dict) -> Dict:
            if service_name == "financial_analysis":
                return {"success": project_success, "results": results}
            elif service_name == "lateral_thinking_pivot":
                pivot_type = payload.get("pivot_type", "")
                return {
                    "success": project_success,
                    "payload": {
                        "penalty_relief": 0.1,
                        "reasoning": f"Pivote {pivot_type} aprobado por MIC"
                    }
                }
            return {"success": False, "error": "Unknown service"}
        
        mic.project_intent = Mock(side_effect=mock_project_intent)
        return mic
    
    @staticmethod
    def create_mock_graph(
        n_nodes: int = 50,
        n_edges: int = 75,
        connected: bool = True
    ) -> MagicMock:
        """Crea mock de grafo NetworkX."""
        graph = MagicMock()
        graph.number_of_nodes.return_value = n_nodes
        graph.number_of_edges.return_value = n_edges
        
        undirected = MagicMock()
        graph.to_undirected.return_value = undirected
        
        # Mock para nx.is_connected
        return graph
    
    @staticmethod
    def create_mock_graph_builder(
        graph: Optional[MagicMock] = None
    ) -> MagicMock:
        """Crea mock de BudgetGraphBuilder."""
        builder = MagicMock()
        builder.build.return_value = graph or MockFactory.create_mock_graph()
        return builder
    
    @staticmethod
    def create_mock_topology_analyzer(
        *,
        betti: Optional[BettiNumbers] = None,
        stability: float = 0.9
    ) -> MagicMock:
        """Crea mock de BusinessTopologicalAnalyzer."""
        analyzer = MagicMock()
        
        betti_result = betti or BettiNumbers(beta_0=1, beta_1=0, beta_2=0)
        analyzer.calculate_betti_numbers.return_value = betti_result
        analyzer.calculate_pyramid_stability.return_value = stability
        
        # Mock para generate_executive_report
        mock_report = MagicMock()
        mock_report.integrity_score = 75.0
        mock_report.waste_alerts = []
        mock_report.circular_risks = []
        mock_report.complexity_level = "Media"
        mock_report.financial_risk_level = "Moderado"
        mock_report.details = {}
        mock_report.strategic_narrative = "Narrativa de prueba"
        
        analyzer.generate_executive_report.return_value = mock_report
        analyzer.analyze_thermal_flow.return_value = {"system_temperature": 0.3}
        
        return analyzer
    
    @staticmethod
    def create_mock_telemetry() -> MagicMock:
        """Crea mock de TelemetryContext."""
        telemetry = MagicMock()
        telemetry.record_error = MagicMock()
        return telemetry
    
    @staticmethod
    def create_mock_translator() -> MagicMock:
        """Crea mock de SemanticTranslator."""
        translator = MagicMock()
        
        mock_report = MagicMock()
        mock_report.raw_narrative = "Narrativa estratégica generada"
        
        translator.compose_strategic_narrative.return_value = mock_report
        return translator


class MetricsFactory:
    """Factory para creación de métricas de prueba."""
    
    @staticmethod
    def create_betti(
        beta_0: int = 1,
        beta_1: int = 0,
        beta_2: int = 0
    ) -> BettiNumbers:
        """Crea números de Betti."""
        return BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=beta_2)
    
    @staticmethod
    def create_topology_bundle(
        *,
        beta_0: int = 1,
        beta_1: int = 0,
        stability: float = 0.9,
        n_nodes: int = 50,
        n_edges: int = 75,
        connected: bool = True
    ) -> TopologicalMetricsBundle:
        """Crea bundle de métricas topológicas."""
        return TopologicalMetricsBundle(
            betti=BettiNumbers(beta_0=beta_0, beta_1=beta_1, beta_2=0),
            pyramid_stability=stability,
            n_nodes=n_nodes,
            n_edges=n_edges,
            is_connected=connected,
            persistence_diagram=None
        )
    
    @staticmethod
    def create_financial_params(
        *,
        investment: float = 1_000_000.0,
        periods: int = 5,
        cash_flow_ratio: float = 0.3,
        std_dev_ratio: float = 0.15,
        volatility: float = 0.2
    ) -> FinancialParameters:
        """Crea parámetros financieros."""
        cash_flows = tuple(investment * cash_flow_ratio for _ in range(periods))
        return FinancialParameters(
            initial_investment=investment,
            cash_flows=cash_flows,
            cost_std_dev=investment * std_dev_ratio,
            project_volatility=volatility
        )
    
    @staticmethod
    def create_financial_metrics(
        *,
        npv: float = 500_000.0,
        irr: float = 0.15,
        payback: float = 3.5,
        var_95: float = 50_000.0,
        sharpe: float = 1.2
    ) -> Dict[str, Any]:
        """Crea métricas financieras."""
        return {
            "npv": npv,
            "irr": irr,
            "payback_period": payback,
            "var_95": var_95,
            "sharpe_ratio": sharpe,
            "initial_investment": 1_000_000.0,
        }
    
    @staticmethod
    def create_thermal_metrics(
        *,
        temperature: float = 0.3,
        entropy: float = 0.5,
        heat_capacity: float = 0.6
    ) -> Dict[str, Any]:
        """Crea métricas térmicas."""
        return {
            "system_temperature": temperature,
            "entropy": entropy,
            "heat_capacity": heat_capacity,
        }


class ContextFactory:
    """Factory para creación de contextos de evaluación."""
    
    @staticmethod
    def create_evaluation_context(
        *,
        df_presupuesto: Optional[pd.DataFrame] = None,
        df_merged: Optional[pd.DataFrame] = None,
        validated_strata: Optional[Set[str]] = None,
        initial_investment: float = 1_000_000.0
    ) -> Dict[str, Any]:
        """Crea contexto de evaluación completo."""
        if df_presupuesto is None:
            df_presupuesto = DataFrameFactory.create_budget_df()
        if df_merged is None:
            apu_codes = df_presupuesto[ColumnNames.CODIGO_APU].tolist()
            df_merged = DataFrameFactory.create_detail_df(apu_codes=apu_codes)
        if validated_strata is None:
            validated_strata = {"PHYSICS", "TACTICS"}
        
        return {
            "df_presupuesto": df_presupuesto,
            "df_merged": df_merged,
            "validated_strata": validated_strata,
            "initial_investment": initial_investment,
            "session_id": "test-session-001",
        }
    
    @staticmethod
    def create_minimal_context() -> Dict[str, Any]:
        """Crea contexto mínimo."""
        return {
            "df_presupuesto": DataFrameFactory.create_budget_df(n_rows=5),
            "df_merged": DataFrameFactory.create_detail_df(n_rows=10),
            "validated_strata": {"PHYSICS", "TACTICS"},
        }


class ConfigFactory:
    """Factory para creación de configuraciones."""
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """Crea configuración por defecto."""
        return {
            "financial_config": {
                "risk_free_rate": 0.03,
                "discount_rate": 0.10,
                "market_return": 0.08,
            },
            "default_financial_params": {
                "initial_investment": 1_000_000.0,
                "cash_flow_ratio": 0.30,
                "cash_flow_periods": 5,
                "cost_std_dev_ratio": 0.15,
                "project_volatility": 0.20,
            },
            "decision_weights": {
                "topology": 0.40,
                "finance": 0.40,
                "thermodynamics": 0.20,
            },
        }
    
    @staticmethod
    def create_strict_config() -> Dict[str, Any]:
        """Crea configuración estricta."""
        config = ConfigFactory.create_default_config()
        config["risk_challenger_config"] = {
            "critical_stability": 0.80,
            "warning_stability": 0.90,
            "integrity_penalty_veto": 0.40,
        }
        return config
    
    @staticmethod
    def create_invalid_config() -> Dict[str, Any]:
        """Crea configuración inválida."""
        return {
            "financial_config": {
                "risk_free_rate": -0.05,  # Inválido: negativo
            }
        }


# =============================================================================
# FIXTURES
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: DataFrames
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_budget_df() -> pd.DataFrame:
    """DataFrame de presupuesto válido."""
    return DataFrameFactory.create_budget_df(n_rows=10)


@pytest.fixture
def valid_detail_df(valid_budget_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame de detalle válido (consistente con presupuesto)."""
    apu_codes = valid_budget_df[ColumnNames.CODIGO_APU].tolist()
    return DataFrameFactory.create_detail_df(n_rows=50, apu_codes=apu_codes)


@pytest.fixture
def empty_budget_df() -> pd.DataFrame:
    """DataFrame de presupuesto vacío."""
    return DataFrameFactory.create_empty_df()


@pytest.fixture
def budget_with_negatives() -> pd.DataFrame:
    """DataFrame con valores negativos."""
    return DataFrameFactory.create_budget_df(n_rows=10, negative_values=True)


@pytest.fixture
def budget_missing_columns() -> pd.DataFrame:
    """DataFrame sin columnas requeridas."""
    return DataFrameFactory.create_budget_df(
        n_rows=10,
        missing_columns=[ColumnNames.CODIGO_APU]
    )


@pytest.fixture
def orphan_detail_df(valid_budget_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame de detalle con códigos huérfanos."""
    return DataFrameFactory.create_orphan_detail_df(valid_budget_df)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Mocks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_mic() -> MagicMock:
    """Mock de MICRegistry con comportamiento exitoso."""
    return MockFactory.create_mock_mic(project_success=True)


@pytest.fixture
def mock_mic_failing() -> MagicMock:
    """Mock de MICRegistry que falla."""
    return MockFactory.create_mock_mic(project_success=False)


@pytest.fixture
def mock_graph() -> MagicMock:
    """Mock de grafo NetworkX."""
    return MockFactory.create_mock_graph()


@pytest.fixture
def mock_graph_builder(mock_graph: MagicMock) -> MagicMock:
    """Mock de BudgetGraphBuilder."""
    return MockFactory.create_mock_graph_builder(mock_graph)


@pytest.fixture
def mock_topology_analyzer() -> MagicMock:
    """Mock de BusinessTopologicalAnalyzer."""
    return MockFactory.create_mock_topology_analyzer()


@pytest.fixture
def mock_telemetry() -> MagicMock:
    """Mock de TelemetryContext."""
    return MockFactory.create_mock_telemetry()


@pytest.fixture
def mock_translator() -> MagicMock:
    """Mock de SemanticTranslator."""
    return MockFactory.create_mock_translator()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Métricas
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_betti() -> BettiNumbers:
    """Números de Betti limpios (conexo, sin ciclos)."""
    return MetricsFactory.create_betti(beta_0=1, beta_1=0)


@pytest.fixture
def cyclic_betti() -> BettiNumbers:
    """Números de Betti con ciclos."""
    return MetricsFactory.create_betti(beta_0=1, beta_1=3)


@pytest.fixture
def fragmented_betti() -> BettiNumbers:
    """Números de Betti fragmentados."""
    return MetricsFactory.create_betti(beta_0=4, beta_1=0)


@pytest.fixture
def stable_topology_bundle() -> TopologicalMetricsBundle:
    """Bundle topológico estable."""
    return MetricsFactory.create_topology_bundle(stability=0.95)


@pytest.fixture
def unstable_topology_bundle() -> TopologicalMetricsBundle:
    """Bundle topológico inestable."""
    return MetricsFactory.create_topology_bundle(stability=0.50, beta_1=5)


@pytest.fixture
def viable_financial_metrics() -> Dict[str, Any]:
    """Métricas financieras viables."""
    return MetricsFactory.create_financial_metrics(npv=500_000.0)


@pytest.fixture
def negative_npv_metrics() -> Dict[str, Any]:
    """Métricas financieras con VPN negativo."""
    return MetricsFactory.create_financial_metrics(npv=-200_000.0)


@pytest.fixture
def thermal_cold() -> Dict[str, Any]:
    """Métricas térmicas frías."""
    return MetricsFactory.create_thermal_metrics(temperature=0.1)


@pytest.fixture
def thermal_hot() -> Dict[str, Any]:
    """Métricas térmicas calientes."""
    return MetricsFactory.create_thermal_metrics(temperature=0.9, heat_capacity=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Configuración
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Configuración por defecto."""
    return ConfigFactory.create_default_config()


@pytest.fixture
def strict_config() -> Dict[str, Any]:
    """Configuración estricta."""
    return ConfigFactory.create_strict_config()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Contextos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_context(valid_budget_df: pd.DataFrame, valid_detail_df: pd.DataFrame) -> Dict[str, Any]:
    """Contexto de evaluación válido."""
    return ContextFactory.create_evaluation_context(
        df_presupuesto=valid_budget_df,
        df_merged=valid_detail_df
    )


@pytest.fixture
def context_missing_strata(valid_budget_df: pd.DataFrame, valid_detail_df: pd.DataFrame) -> Dict[str, Any]:
    """Contexto sin estratos validados."""
    return ContextFactory.create_evaluation_context(
        df_presupuesto=valid_budget_df,
        df_merged=valid_detail_df,
        validated_strata=set()  # Sin estratos
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Componentes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def validator() -> DataFrameValidator:
    """Validador de DataFrames."""
    return DataFrameValidator()


@pytest.fixture
def default_thresholds() -> RiskChallengerThresholds:
    """Umbrales por defecto del Risk Challenger."""
    return RiskChallengerThresholds()


@pytest.fixture
def default_weights() -> DecisionWeights:
    """Pesos de decisión por defecto."""
    return DecisionWeights()


@pytest.fixture
def risk_challenger(mock_mic: MagicMock) -> RiskChallenger:
    """Risk Challenger con MIC mock."""
    return RiskChallenger(mic=mock_mic)


@pytest.fixture
def business_agent(
    default_config: Dict[str, Any],
    mock_mic: MagicMock,
    mock_telemetry: MagicMock,
    mock_graph_builder: MagicMock,
    mock_topology_analyzer: MagicMock,
    mock_translator: MagicMock
) -> BusinessAgent:
    """BusinessAgent completo con mocks."""
    return BusinessAgent(
        config=default_config,
        mic=mock_mic,
        telemetry=mock_telemetry,
        graph_builder=mock_graph_builder,
        topology_analyzer=mock_topology_analyzer,
        semantic_translator=mock_translator
    )


# =============================================================================
# TESTS: ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================

@pytest.mark.unit
class TestRiskChallengerThresholds:
    """Pruebas para RiskChallengerThresholds."""
    
    def test_default_values(self):
        """Valores por defecto son sensatos."""
        thresholds = RiskChallengerThresholds()
        
        assert thresholds.critical_stability == 0.70
        assert thresholds.warning_stability == 0.85
        assert thresholds.coherence_minimum == 0.60
        assert thresholds.cycle_density_limit == 0.33
        assert thresholds.integrity_penalty_veto == 0.30
        assert thresholds.integrity_penalty_warn == 0.15
    
    def test_custom_values(self):
        """Acepta valores personalizados."""
        thresholds = RiskChallengerThresholds(
            critical_stability=0.80,
            warning_stability=0.95
        )
        
        assert thresholds.critical_stability == 0.80
        assert thresholds.warning_stability == 0.95
    
    @pytest.mark.parametrize("field,invalid_value", [
        ("critical_stability", -0.1),
        ("critical_stability", 1.5),
        ("warning_stability", -0.5),
        ("integrity_penalty_veto", 2.0),
    ])
    def test_invalid_threshold_raises_error(self, field: str, invalid_value: float):
        """Valores fuera de [0, 1] lanzan error."""
        kwargs = {field: invalid_value}
        
        with pytest.raises(ValueError, match="fuera de rango"):
            RiskChallengerThresholds(**kwargs)
    
    def test_is_frozen(self):
        """Es inmutable (frozen dataclass)."""
        thresholds = RiskChallengerThresholds()
        
        with pytest.raises(FrozenInstanceError):
            thresholds.critical_stability = 0.99


@pytest.mark.unit
class TestDecisionWeights:
    """Pruebas para DecisionWeights."""
    
    def test_default_values(self):
        """Valores por defecto."""
        weights = DecisionWeights()
        
        assert weights.topology == 0.40
        assert weights.finance == 0.40
        assert weights.thermodynamics == 0.20
    
    def test_normalized_property(self):
        """Propiedad normalized retorna pesos normalizados a suma 1."""
        weights = DecisionWeights(topology=1.0, finance=1.0, thermodynamics=1.0)
        normalized = weights.normalized
        
        total = normalized.topology + normalized.finance + normalized.thermodynamics
        assert abs(total - 1.0) < TestConstants.FLOAT_TOLERANCE
    
    def test_normalized_zero_weights(self):
        """Pesos cero resultan en distribución uniforme."""
        weights = DecisionWeights(topology=0.0, finance=0.0, thermodynamics=0.0)
        normalized = weights.normalized
        
        assert normalized.topology == pytest.approx(1/3, rel=1e-5)
        assert normalized.finance == pytest.approx(1/3, rel=1e-5)
        assert normalized.thermodynamics == pytest.approx(1/3, rel=1e-5)
    
    def test_to_tuple(self):
        """to_tuple retorna tupla correcta."""
        weights = DecisionWeights(topology=0.5, finance=0.3, thermodynamics=0.2)
        result = weights.to_tuple()
        
        assert result == (0.5, 0.3, 0.2)
    
    def test_negative_weight_raises_error(self):
        """Pesos negativos lanzan error."""
        with pytest.raises(ValueError, match="no puede ser negativo"):
            DecisionWeights(topology=-0.1)
    
    def test_is_frozen(self):
        """Es inmutable."""
        weights = DecisionWeights()
        
        with pytest.raises(FrozenInstanceError):
            weights.topology = 0.99


@pytest.mark.unit
class TestFinancialParameters:
    """Pruebas para FinancialParameters."""
    
    def test_valid_construction(self):
        """Construcción válida."""
        params = FinancialParameters(
            initial_investment=1_000_000.0,
            cash_flows=(300_000.0, 300_000.0, 300_000.0),
            cost_std_dev=100_000.0,
            project_volatility=0.2
        )
        
        assert params.initial_investment == 1_000_000.0
        assert len(params.cash_flows) == 3
        assert params.periods == 3
        assert params.total_cash_flow == 900_000.0
    
    def test_zero_investment_raises_error(self):
        """Inversión cero lanza error."""
        with pytest.raises(ValueError, match="positiva"):
            FinancialParameters(
                initial_investment=0.0,
                cash_flows=(100.0,),
                cost_std_dev=10.0,
                project_volatility=0.1
            )
    
    def test_negative_investment_raises_error(self):
        """Inversión negativa lanza error."""
        with pytest.raises(ValueError, match="positiva"):
            FinancialParameters(
                initial_investment=-1000.0,
                cash_flows=(100.0,),
                cost_std_dev=10.0,
                project_volatility=0.1
            )
    
    def test_negative_std_dev_raises_error(self):
        """Desviación estándar negativa lanza error."""
        with pytest.raises(ValueError, match="negativa"):
            FinancialParameters(
                initial_investment=1000.0,
                cash_flows=(100.0,),
                cost_std_dev=-10.0,
                project_volatility=0.1
            )
    
    @pytest.mark.parametrize("volatility", [-0.1, 1.5, 2.0])
    def test_invalid_volatility_raises_error(self, volatility: float):
        """Volatilidad fuera de [0, 1] lanza error."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            FinancialParameters(
                initial_investment=1000.0,
                cash_flows=(100.0,),
                cost_std_dev=10.0,
                project_volatility=volatility
            )
    
    def test_to_dict(self):
        """Serialización a diccionario."""
        params = MetricsFactory.create_financial_params(investment=1_000_000.0, periods=3)
        result = params.to_dict()
        
        assert "initial_investment" in result
        assert "cash_flows" in result
        assert "periods" in result
        assert result["periods"] == 3
    
    def test_is_frozen(self):
        """Es inmutable."""
        params = MetricsFactory.create_financial_params()
        
        with pytest.raises(FrozenInstanceError):
            params.initial_investment = 999.0


@pytest.mark.unit
class TestBettiNumbers:
    """Pruebas para BettiNumbers."""
    
    def test_default_values(self):
        """Valores por defecto."""
        betti = BettiNumbers()
        
        assert betti.beta_0 == 1
        assert betti.beta_1 == 0
        assert betti.beta_2 == 0
    
    def test_euler_characteristic(self):
        """Característica de Euler χ = β₀ - β₁ + β₂."""
        betti = BettiNumbers(beta_0=2, beta_1=3, beta_2=1)
        
        assert betti.euler_characteristic == 2 - 3 + 1
    
    def test_is_connected(self):
        """is_connected es True cuando β₀ = 1."""
        connected = BettiNumbers(beta_0=1)
        disconnected = BettiNumbers(beta_0=3)
        
        assert connected.is_connected is True
        assert disconnected.is_connected is False
    
    def test_has_cycles(self):
        """has_cycles es True cuando β₁ > 0."""
        no_cycles = BettiNumbers(beta_0=1, beta_1=0)
        with_cycles = BettiNumbers(beta_0=1, beta_1=2)
        
        assert no_cycles.has_cycles is False
        assert with_cycles.has_cycles is True
    
    def test_from_dict(self):
        """Creación desde diccionario."""
        data = {"beta_0": 2, "beta_1": 1, "beta_2": 0}
        betti = BettiNumbers.from_dict(data)
        
        assert betti.beta_0 == 2
        assert betti.beta_1 == 1
    
    def test_from_dict_with_defaults(self):
        """Creación desde diccionario parcial usa defaults."""
        data = {"beta_0": 3}
        betti = BettiNumbers.from_dict(data)
        
        assert betti.beta_0 == 3
        assert betti.beta_1 == 0
    
    def test_to_dict(self):
        """Serialización a diccionario."""
        betti = BettiNumbers(beta_0=1, beta_1=2, beta_2=0)
        result = betti.to_dict()
        
        assert result == {"beta_0": 1, "beta_1": 2, "beta_2": 0}


@pytest.mark.unit
class TestTopologicalMetricsBundle:
    """Pruebas para TopologicalMetricsBundle."""
    
    def test_structural_coherence_perfect(self):
        """Coherencia perfecta cuando β₀=1, β₁=0, Ψ=1."""
        bundle = TopologicalMetricsBundle(
            betti=BettiNumbers(beta_0=1, beta_1=0),
            pyramid_stability=1.0,
            n_nodes=50,
            n_edges=75,
            is_connected=True
        )
        
        # Coherencia debe ser muy cercana a 1.0
        assert bundle.structural_coherence > 0.95
    
    def test_structural_coherence_fragmented(self):
        """Coherencia baja cuando hay múltiples componentes."""
        bundle = TopologicalMetricsBundle(
            betti=BettiNumbers(beta_0=5, beta_1=0),  # 5 componentes
            pyramid_stability=0.9,
            n_nodes=50,
            n_edges=40,
            is_connected=False
        )
        
        # Coherencia penalizada por fragmentación
        assert bundle.structural_coherence < 0.5
    
    def test_structural_coherence_cyclic(self):
        """Coherencia reducida cuando hay ciclos."""
        bundle = TopologicalMetricsBundle(
            betti=BettiNumbers(beta_0=1, beta_1=10),  # Muchos ciclos
            pyramid_stability=0.9,
            n_nodes=50,
            n_edges=60,
            is_connected=True
        )
        
        # Coherencia penalizada por ciclos
        assert bundle.structural_coherence < 0.7
    
    def test_cycle_density(self):
        """cycle_density es β₁/n."""
        bundle = TopologicalMetricsBundle(
            betti=BettiNumbers(beta_0=1, beta_1=5),
            pyramid_stability=0.9,
            n_nodes=100,
            n_edges=150,
            is_connected=True
        )
        
        assert bundle.cycle_density == 5 / 100
    
    def test_to_dict(self):
        """Serialización completa."""
        bundle = MetricsFactory.create_topology_bundle()
        result = bundle.to_dict()
        
        assert "betti_numbers" in result
        assert "pyramid_stability" in result
        assert "structural_coherence" in result
        assert "cycle_density" in result
        assert "euler_characteristic" in result


@pytest.mark.unit
class TestValidationResult:
    """Pruebas para ValidationResult."""
    
    def test_success_factory(self):
        """Factory success crea resultado válido."""
        result = ValidationResult.success({"key": "value"})
        
        assert result.is_valid is True
        assert result.message == "Validación exitosa"
        assert result.diagnostics == {"key": "value"}
    
    def test_failure_factory(self):
        """Factory failure crea resultado inválido."""
        result = ValidationResult.failure("Error de prueba", {"error_code": 123})
        
        assert result.is_valid is False
        assert result.message == "Error de prueba"
        assert result.diagnostics["error_code"] == 123


# =============================================================================
# TESTS: ENUMERACIONES
# =============================================================================

@pytest.mark.unit
class TestRiskClassification:
    """Pruebas para RiskClassification."""
    
    @pytest.mark.parametrize("input_str,expected", [
        ("LOW", RiskClassification.SAFE),
        ("low", RiskClassification.SAFE),
        ("BAJO", RiskClassification.SAFE),
        ("SAFE", RiskClassification.SAFE),
        ("MODERATE", RiskClassification.MODERATE),
        ("MEDIUM", RiskClassification.MODERATE),
        ("HIGH", RiskClassification.HIGH),
        ("ALTO", RiskClassification.HIGH),
        ("CRITICAL", RiskClassification.CRITICAL),
        ("SEVERE", RiskClassification.CRITICAL),
        ("UNKNOWN_VALUE", RiskClassification.UNKNOWN),
        ("", RiskClassification.UNKNOWN),
    ])
    def test_from_string(self, input_str: str, expected: RiskClassification):
        """Parsing de string a clasificación."""
        assert RiskClassification.from_string(input_str) == expected
    
    def test_from_non_string(self):
        """Entrada no-string retorna UNKNOWN."""
        assert RiskClassification.from_string(123) == RiskClassification.UNKNOWN  # type: ignore
        assert RiskClassification.from_string(None) == RiskClassification.UNKNOWN  # type: ignore


@pytest.mark.unit
class TestVetoSeverity:
    """Pruebas para VetoSeverity."""
    
    def test_all_values_exist(self):
        """Todos los valores existen."""
        assert VetoSeverity.CRITICO.value == "CRÍTICO"
        assert VetoSeverity.SEVERO.value == "SEVERO"
        assert VetoSeverity.MODERADO.value == "MODERADO"
        assert VetoSeverity.LEVE.value == "LEVE"


@pytest.mark.unit
class TestPivotType:
    """Pruebas para PivotType."""
    
    def test_all_pivot_types_exist(self):
        """Todos los tipos de pivote existen."""
        assert PivotType.MONOPOLIO_COBERTURADO.value == "MONOPOLIO_COBERTURADO"
        assert PivotType.OPCION_ESPERA.value == "OPCION_ESPERA"
        assert PivotType.CUARENTENA_TOPOLOGICA.value == "CUARENTENA_TOPOLOGICA"


# =============================================================================
# TESTS: EXCEPCIONES
# =============================================================================

@pytest.mark.unit
class TestExceptions:
    """Pruebas para jerarquía de excepciones."""
    
    def test_business_agent_error_base(self):
        """BusinessAgentError es la base."""
        error = BusinessAgentError("Test error", {"key": "value"})
        
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}
        assert error.timestamp > 0
    
    def test_business_agent_error_to_dict(self):
        """to_dict serializa correctamente."""
        error = BusinessAgentError("Test", {"code": 123})
        result = error.to_dict()
        
        assert result["error"] == "Test"
        assert result["error_type"] == "BusinessAgentError"
        assert result["details"]["code"] == 123
    
    def test_all_exceptions_inherit_from_base(self):
        """Todas las excepciones heredan de BusinessAgentError."""
        exceptions = [
            ConfigurationError,
            ValidationError,
            TopologicalAnomalyError,
            MICHierarchyViolationError,
            FinancialProjectionError,
            SynthesisError,
        ]
        
        for exc_class in exceptions:
            error = exc_class("Test")
            assert isinstance(error, BusinessAgentError)


# =============================================================================
# TESTS: DATAFRAME VALIDATOR
# =============================================================================

@pytest.mark.unit
class TestDataFrameValidator:
    """Pruebas para DataFrameValidator."""
    
    def test_validate_valid_dataframes(
        self,
        validator: DataFrameValidator,
        valid_budget_df: pd.DataFrame,
        valid_detail_df: pd.DataFrame
    ):
        """DataFrames válidos pasan validación."""
        result = validator.validate(valid_budget_df, valid_detail_df)
        
        assert result.is_valid is True
        assert "Validación exitosa" in result.message
    
    def test_validate_none_budget_fails(self, validator: DataFrameValidator):
        """Presupuesto None falla."""
        result = validator.validate(None, pd.DataFrame({"a": [1]}))
        
        assert result.is_valid is False
        assert "None" in result.message
    
    def test_validate_none_detail_fails(
        self,
        validator: DataFrameValidator,
        valid_budget_df: pd.DataFrame
    ):
        """Detalle None falla."""
        result = validator.validate(valid_budget_df, None)
        
        assert result.is_valid is False
        assert "None" in result.message
    
    def test_validate_empty_budget_fails(
        self,
        validator: DataFrameValidator,
        empty_budget_df: pd.DataFrame
    ):
        """Presupuesto vacío falla."""
        result = validator.validate(empty_budget_df, pd.DataFrame({"a": [1]}))
        
        assert result.is_valid is False
        assert "vacío" in result.message
    
    def test_validate_missing_required_column(
        self,
        validator: DataFrameValidator,
        budget_missing_columns: pd.DataFrame,
        valid_detail_df: pd.DataFrame
    ):
        """Columna requerida faltante falla."""
        result = validator.validate(budget_missing_columns, valid_detail_df)
        
        assert result.is_valid is False
        assert "no encontrada" in result.message or "FAIL" in str(result.diagnostics)
    
    def test_validate_detects_duplicates(
        self,
        validator: DataFrameValidator,
        valid_detail_df: pd.DataFrame
    ):
        """Detecta códigos duplicados."""
        # Crear presupuesto con duplicados
        df = DataFrameFactory.create_budget_df(n_rows=10)
        df.loc[5, ColumnNames.CODIGO_APU] = df.loc[0, ColumnNames.CODIGO_APU]
        
        result = validator.validate(df, valid_detail_df)
        
        # Debe pasar pero con diagnóstico de duplicados
        assert "duplicated_codes" in result.diagnostics.get("duplicate_analysis", {})
    
    def test_validate_detects_orphan_codes(
        self,
        validator: DataFrameValidator,
        valid_budget_df: pd.DataFrame,
        orphan_detail_df: pd.DataFrame
    ):
        """Detecta códigos huérfanos en detalle."""
        result = validator.validate(valid_budget_df, orphan_detail_df)
        
        # Pasa pero con warnings
        warnings = result.diagnostics.get("warnings", [])
        assert any("sin referencia" in w for w in warnings) or len(warnings) > 0
    
    def test_validate_detects_outliers(self, validator: DataFrameValidator):
        """Detecta outliers en valores."""
        # Crear presupuesto con outlier extremo
        df = DataFrameFactory.create_budget_df(n_rows=20)
        df.loc[0, ColumnNames.VALOR_TOTAL] = 1e9  # Outlier extremo
        
        detail = DataFrameFactory.create_detail_df(
            apu_codes=df[ColumnNames.CODIGO_APU].tolist()
        )
        
        result = validator.validate(df, detail)
        
        dist_analysis = result.diagnostics.get("distribution_analysis", {})
        assert dist_analysis.get("outlier_count", 0) >= 1
    
    def test_validate_legacy_column_mapping(self, validator: DataFrameValidator):
        """Mapea columnas legacy correctamente."""
        # Usar nombres legacy
        df_budget = pd.DataFrame({
            "item": ["A001", "A002"],  # Legacy para codigo_apu
            "descripcion": ["Desc 1", "Desc 2"],  # Legacy
            "total": [1000, 2000],  # Legacy para valor_total
        })
        
        df_detail = pd.DataFrame({
            "codigo": ["A001", "A001"],  # Legacy
            "desc_insumo": ["Insumo 1", "Insumo 2"],  # Legacy
            "cantidad": [10, 20],  # Legacy
            "costo": [100, 200],  # Legacy
        })
        
        result = validator.validate(df_budget, df_detail)
        
        # Debe encontrar las columnas vía mapeo legacy
        assert result.is_valid is True or "no encontrada" not in result.message


# =============================================================================
# TESTS: DECISION ALGEBRA
# =============================================================================

@pytest.mark.unit
@pytest.mark.algebraic
class TestDecisionAlgebra:
    """Pruebas para DecisionAlgebra."""
    
    def test_normalize_to_sphere_unit_vector(self):
        """Normalización de vector a esfera unitaria."""
        v = np.array([3.0, 4.0])  # Norma = 5
        
        result = DecisionAlgebra.normalize_to_sphere(v)
        
        assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-6)
        assert result[0] == pytest.approx(0.6, rel=1e-6)
        assert result[1] == pytest.approx(0.8, rel=1e-6)
    
    def test_normalize_to_sphere_zero_vector(self):
        """Vector cero se convierte en vector uniforme."""
        v = np.array([0.0, 0.0, 0.0])
        
        result = DecisionAlgebra.normalize_to_sphere(v)
        
        assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-6)
        # Uniforme: cada componente = 1/√3
        expected = 1.0 / np.sqrt(3)
        assert all(r == pytest.approx(expected, rel=1e-6) for r in result)
    
    def test_normalize_to_sphere_near_zero(self):
        """Vector casi-cero se maneja correctamente."""
        v = np.array([1e-15, 1e-15])
        
        result = DecisionAlgebra.normalize_to_sphere(v)
        
        assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-6)
    
    def test_weighted_geometric_mean_equal_weights(self):
        """Media geométrica con pesos iguales."""
        factors = [2.0, 8.0]  # √(2*8) = 4
        
        result = DecisionAlgebra.weighted_geometric_mean(factors)
        
        assert result == pytest.approx(4.0, rel=1e-6)
    
    def test_weighted_geometric_mean_custom_weights(self):
        """Media geométrica con pesos personalizados."""
        factors = [2.0, 4.0]
        weights = [1.0, 3.0]  # 2^(1/4) * 4^(3/4) ≈ 3.36
        
        result = DecisionAlgebra.weighted_geometric_mean(factors, weights)
        
        expected = (2.0 ** 0.25) * (4.0 ** 0.75)
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_weighted_geometric_mean_zero_factor(self):
        """Factor cero con peso positivo resulta en 0."""
        factors = [0.0, 2.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        
        result = DecisionAlgebra.weighted_geometric_mean(factors, weights)
        
        assert result == 0.0
    
    def test_weighted_geometric_mean_empty_factors(self):
        """Lista vacía retorna 0."""
        result = DecisionAlgebra.weighted_geometric_mean([])
        
        assert result == 0.0
    
    def test_weighted_geometric_mean_dimension_mismatch(self):
        """Dimensiones distintas lanzan error."""
        with pytest.raises(ValueError, match="no coinciden"):
            DecisionAlgebra.weighted_geometric_mean([1.0, 2.0], [1.0])
    
    def test_convex_combination(self):
        """Combinación convexa de vectores."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        
        weights = DecisionWeights(topology=0.5, finance=0.3, thermodynamics=0.2)
        
        result = DecisionAlgebra.convex_combination([v1, v2, v3], weights)
        
        assert result[0] == pytest.approx(0.5, rel=1e-6)
        assert result[1] == pytest.approx(0.3, rel=1e-6)
        assert result[2] == pytest.approx(0.2, rel=1e-6)
    
    def test_compute_quality_factors(self, stable_topology_bundle: TopologicalMetricsBundle):
        """Cálculo de factores de calidad."""
        financial = MetricsFactory.create_financial_metrics(npv=500_000.0)
        
        topo_q, fin_q, thermo_q = DecisionAlgebra.compute_quality_factors(
            topo_bundle=stable_topology_bundle,
            financial_metrics=financial,
            entropy=0.3,
            exergy=0.7,
            initial_investment=1_000_000.0
        )
        
        # Todos deben estar en [0, 1]
        assert 0.0 <= topo_q <= 1.0
        assert 0.0 <= fin_q <= 1.0
        assert 0.0 <= thermo_q <= 1.0
        
        # Topología estable debe tener alta calidad
        assert topo_q > 0.8
    
    def test_compute_quality_factors_negative_npv(
        self,
        stable_topology_bundle: TopologicalMetricsBundle
    ):
        """Factores con VPN negativo."""
        financial = MetricsFactory.create_financial_metrics(npv=-500_000.0)
        
        _, fin_q, _ = DecisionAlgebra.compute_quality_factors(
            topo_bundle=stable_topology_bundle,
            financial_metrics=financial,
            entropy=0.5,
            exergy=0.5,
            initial_investment=1_000_000.0
        )
        
        # Calidad financiera baja pero >= 0
        assert 0.0 <= fin_q < 0.5


# =============================================================================
# TESTS: ESTRATEGIAS DE PIVOTE
# =============================================================================

@pytest.mark.unit
class TestMonopolioCoberturadoStrategy:
    """Pruebas para MonopolioCoberturadoStrategy."""
    
    @pytest.fixture
    def strategy(self) -> MonopolioCoberturadoStrategy:
        return MonopolioCoberturadoStrategy()
    
    def test_applies_when_conditions_met(self, strategy: MonopolioCoberturadoStrategy):
        """Aplica cuando todas las condiciones se cumplen."""
        applies, reason = strategy.evaluate(
            stability=0.65,  # < 0.70
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={"system_temperature": 10.0, "heat_capacity": 0.8},  # Frío, alta inercia
            financial_metrics={},
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is True
        assert "inercia" in reason.lower()
    
    def test_not_applies_high_stability(self, strategy: MonopolioCoberturadoStrategy):
        """No aplica con alta estabilidad."""
        applies, _ = strategy.evaluate(
            stability=0.90,  # > 0.70
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={"system_temperature": 10.0, "heat_capacity": 0.8},
            financial_metrics={},
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is False
    
    def test_not_applies_hot_system(self, strategy: MonopolioCoberturadoStrategy):
        """No aplica con sistema caliente."""
        applies, _ = strategy.evaluate(
            stability=0.60,
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={"system_temperature": 30.0, "heat_capacity": 0.8},  # Caliente
            financial_metrics={},
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is False
    
    def test_not_applies_low_inertia(self, strategy: MonopolioCoberturadoStrategy):
        """No aplica con baja inercia financiera."""
        applies, _ = strategy.evaluate(
            stability=0.60,
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={"system_temperature": 10.0, "heat_capacity": 0.3},  # Baja inercia
            financial_metrics={},
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is False


@pytest.mark.unit
class TestOpcionEsperaStrategy:
    """Pruebas para OpcionEsperaStrategy."""
    
    @pytest.fixture
    def strategy(self) -> OpcionEsperaStrategy:
        return OpcionEsperaStrategy(npv_multiplier=1.5)
    
    def test_applies_when_conditions_met(self, strategy: OpcionEsperaStrategy):
        """Aplica cuando riesgo es HIGH y opción de espera supera umbral."""
        applies, reason = strategy.evaluate(
            stability=0.8,
            financial_class=RiskClassification.HIGH,
            thermal_metrics={},
            financial_metrics={
                "npv": 100_000.0,
                "real_options": {"wait_option_value": 200_000.0}  # > 150k
            },
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is True
        assert "opción de espera" in reason.lower()
    
    def test_not_applies_wrong_risk_class(self, strategy: OpcionEsperaStrategy):
        """No aplica si riesgo no es HIGH."""
        applies, _ = strategy.evaluate(
            stability=0.8,
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={},
            financial_metrics={
                "npv": 100_000.0,
                "real_options": {"wait_option_value": 200_000.0}
            },
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is False
    
    def test_not_applies_low_option_value(self, strategy: OpcionEsperaStrategy):
        """No aplica si valor de opción es bajo."""
        applies, _ = strategy.evaluate(
            stability=0.8,
            financial_class=RiskClassification.HIGH,
            thermal_metrics={},
            financial_metrics={
                "npv": 100_000.0,
                "real_options": {"wait_option_value": 50_000.0}  # < 150k
            },
            synergy_risk={},
            beta_1=0
        )
        
        assert applies is False


@pytest.mark.unit
class TestCuarentenaTopologicaStrategy:
    """Pruebas para CuarentenaTopologicaStrategy."""
    
    @pytest.fixture
    def strategy(self) -> CuarentenaTopologicaStrategy:
        return CuarentenaTopologicaStrategy()
    
    def test_applies_cycles_without_synergy(self, strategy: CuarentenaTopologicaStrategy):
        """Aplica cuando hay ciclos pero sin sinergia."""
        applies, reason = strategy.evaluate(
            stability=0.8,
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={},
            financial_metrics={},
            synergy_risk={"synergy_detected": False},
            beta_1=3  # Ciclos presentes
        )
        
        assert applies is True
        assert "confinados" in reason.lower()
    
    def test_not_applies_with_synergy(self, strategy: CuarentenaTopologicaStrategy):
        """No aplica cuando hay sinergia multiplicativa."""
        applies, reason = strategy.evaluate(
            stability=0.8,
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={},
            financial_metrics={},
            synergy_risk={"synergy_detected": True},
            beta_1=3
        )
        
        assert applies is False
        assert "sinergia" in reason.lower()
    
    def test_not_applies_no_cycles(self, strategy: CuarentenaTopologicaStrategy):
        """No aplica cuando no hay ciclos."""
        applies, _ = strategy.evaluate(
            stability=0.8,
            financial_class=RiskClassification.MODERATE,
            thermal_metrics={},
            financial_metrics={},
            synergy_risk={"synergy_detected": False},
            beta_1=0  # Sin ciclos
        )
        
        assert applies is False


# =============================================================================
# TESTS: RISK CHALLENGER
# =============================================================================

@pytest.mark.unit
class TestRiskChallenger:
    """Pruebas para RiskChallenger."""
    
    def test_challenge_stable_report_unchanged(self, risk_challenger: RiskChallenger):
        """Reporte estable no cambia."""
        report = MagicMock()
        report.integrity_score = 85.0
        report.financial_risk_level = "MODERATE"
        report.waste_alerts = []
        report.circular_risks = []
        report.complexity_level = "Media"
        report.strategic_narrative = "Test"
        report.details = {
            "pyramid_stability": 0.95,
            "topological_invariants": {
                "betti_numbers": {"beta_0": 1, "beta_1": 0},
                "n_nodes": 50
            }
        }
        
        result = risk_challenger.challenge_verdict(report)
        
        # Debe ser igual (sin vetos)
        assert result.integrity_score >= report.integrity_score * 0.9
    
    def test_challenge_unstable_emits_veto(self, mock_mic: MagicMock):
        """Reporte inestable emite veto."""
        # MIC que rechaza pivotes
        mock_mic.project_intent.return_value = {"success": False}
        challenger = RiskChallenger(mic=mock_mic)
        
        report = MagicMock()
        report.integrity_score = 80.0
        report.financial_risk_level = "MODERATE"
        report.waste_alerts = []
        report.circular_risks = []
        report.complexity_level = "Media"
        report.strategic_narrative = "Test"
        report.details = {
            "pyramid_stability": 0.50,  # Inestable
            "topological_invariants": {
                "betti_numbers": {"beta_0": 1, "beta_1": 0},
                "n_nodes": 50
            }
        }
        
        result = challenger.challenge_verdict(report)
        
        # Integridad debe reducirse
        assert result.integrity_score < report.integrity_score
        # Debe haber acta de veto
        assert "challenger_verdict" in result.details or "challenger_applied" in result.details
    
    def test_challenge_with_lateral_exception(self, mock_mic: MagicMock):
        """Pivote aprobado por MIC emite excepción lateral."""
        # MIC que aprueba pivotes
        mock_mic.project_intent.return_value = {
            "success": True,
            "payload": {
                "penalty_relief": 0.1,
                "reasoning": "Pivote aprobado"
            }
        }
        challenger = RiskChallenger(mic=mock_mic)
        
        report = MagicMock()
        report.integrity_score = 70.0
        report.financial_risk_level = "HIGH"  # Para activar OPCION_ESPERA
        report.waste_alerts = []
        report.circular_risks = []
        report.complexity_level = "Alta"
        report.strategic_narrative = "Test"
        report.details = {
            "pyramid_stability": 0.60,  # Inestable para MONOPOLIO
            "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8},
            "financial_metrics": {
                "npv": 100_000.0,
                "real_options": {"wait_option_value": 200_000.0}
            },
            "topological_invariants": {
                "betti_numbers": {"beta_0": 1, "beta_1": 0},
                "n_nodes": 50
            }
        }
        
        result = challenger.challenge_verdict(report)
        
        # Debe haber excepción lateral
        assert "lateral_thinking_applied" in result.details or "lateral_exception" in result.details
    
    def test_challenge_cycle_penalty(self, mock_mic: MagicMock):
        """Ciclos generan penalización."""
        mock_mic.project_intent.return_value = {"success": False}  # No aprueba cuarentena
        
        challenger = RiskChallenger(
            thresholds=RiskChallengerThresholds(cycle_density_limit=0.05),
            mic=mock_mic
        )
        
        report = MagicMock()
        report.integrity_score = 80.0
        report.financial_risk_level = "SAFE"
        report.waste_alerts = []
        report.circular_risks = []
        report.complexity_level = "Media"
        report.strategic_narrative = "Test"
        report.details = {
            "pyramid_stability": 0.95,
            "synergy_risk": {"synergy_detected": True},  # Cuarentena no aplica
            "topological_invariants": {
                "betti_numbers": {"beta_0": 1, "beta_1": 10},  # Muchos ciclos
                "n_nodes": 50  # 10/50 = 0.2 > 0.05
            }
        }
        
        result = challenger.challenge_verdict(report)
        
        # Debe aplicar penalización por ciclos
        assert "cycle_penalty" in result.details.get("penalties_applied", []) or \
               result.integrity_score < report.integrity_score


# =============================================================================
# TESTS: TOPOLOGY BUILDER
# =============================================================================

@pytest.mark.unit
class TestTopologyBuilder:
    """Pruebas para TopologyBuilder."""
    
    def test_build_success(
        self,
        mock_graph_builder: MagicMock,
        mock_topology_analyzer: MagicMock,
        mock_telemetry: MagicMock,
        valid_budget_df: pd.DataFrame,
        valid_detail_df: pd.DataFrame
    ):
        """Construcción exitosa retorna grafo y bundle."""
        # Configurar mock de nx.is_connected
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            mock_nx.number_connected_components.return_value = 1
            
            builder = TopologyBuilder(
                graph_builder=mock_graph_builder,
                analyzer=mock_topology_analyzer,
                telemetry=mock_telemetry
            )
            
            graph, bundle = builder.build(valid_budget_df, valid_detail_df)
        
        assert graph is not None
        assert isinstance(bundle, TopologicalMetricsBundle)
        assert bundle.betti.beta_0 == 1
    
    def test_build_no_graph_builder_raises(
        self,
        mock_telemetry: MagicMock,
        valid_budget_df: pd.DataFrame,
        valid_detail_df: pd.DataFrame
    ):
        """Sin graph_builder lanza error."""
        builder = TopologyBuilder(
            graph_builder=None,
            analyzer=None,
            telemetry=mock_telemetry
        )
        
        with pytest.raises(RuntimeError, match="GraphBuilder no configurado"):
            builder.build(valid_budget_df, valid_detail_df)
    
    def test_build_empty_graph_raises(
        self,
        mock_topology_analyzer: MagicMock,
        mock_telemetry: MagicMock,
        valid_budget_df: pd.DataFrame,
        valid_detail_df: pd.DataFrame
    ):
        """Grafo vacío lanza TopologicalAnomalyError."""
        empty_graph = MagicMock()
        empty_graph.number_of_nodes.return_value = 0
        
        mock_builder = MagicMock()
        mock_builder.build.return_value = empty_graph
        
        builder = TopologyBuilder(
            graph_builder=mock_builder,
            analyzer=mock_topology_analyzer,
            telemetry=mock_telemetry
        )
        
        with pytest.raises(TopologicalAnomalyError, match="no tiene vértices"):
            builder.build(valid_budget_df, valid_detail_df)


# =============================================================================
# TESTS: REPORT COMPOSER
# =============================================================================

@pytest.mark.unit
class TestReportComposer:
    """Pruebas para ReportComposer."""
    
    def test_compose_success(
        self,
        mock_topology_analyzer: MagicMock,
        mock_translator: MagicMock,
        mock_graph: MagicMock,
        stable_topology_bundle: TopologicalMetricsBundle,
        viable_financial_metrics: Dict[str, Any],
        thermal_cold: Dict[str, Any]
    ):
        """Composición exitosa genera reporte."""
        composer = ReportComposer(
            analyzer=mock_topology_analyzer,
            translator=mock_translator,
            weights=DecisionWeights()
        )
        
        report = composer.compose(
            graph=mock_graph,
            topo_bundle=stable_topology_bundle,
            financial_metrics=viable_financial_metrics,
            thermal_metrics=thermal_cold,
            entropy=0.3,
            exergy=0.7
        )
        
        assert report is not None
        assert report.integrity_score > 0
        assert "decision_algebra" in report.details
    
    def test_compose_no_analyzer_raises(
        self,
        mock_graph: MagicMock,
        stable_topology_bundle: TopologicalMetricsBundle
    ):
        """Sin analyzer lanza SynthesisError."""
        composer = ReportComposer(analyzer=None)
        
        with pytest.raises(SynthesisError, match="no configurado"):
            composer.compose(
                graph=mock_graph,
                topo_bundle=stable_topology_bundle,
                financial_metrics={},
                thermal_metrics={},
            )


# =============================================================================
# TESTS: BUSINESS AGENT
# =============================================================================

@pytest.mark.integration
class TestBusinessAgent:
    """Pruebas de integración para BusinessAgent."""
    
    def test_init_valid_config(
        self,
        default_config: Dict[str, Any],
        mock_mic: MagicMock,
        mock_telemetry: MagicMock
    ):
        """Inicialización con configuración válida."""
        agent = BusinessAgent(
            config=default_config,
            mic=mock_mic,
            telemetry=mock_telemetry
        )
        
        assert agent.config == default_config
        assert agent.mic == mock_mic
    
    def test_init_invalid_config_raises(self, mock_mic: MagicMock):
        """Configuración inválida lanza error."""
        invalid_config = ConfigFactory.create_invalid_config()
        
        with pytest.raises(ConfigurationError):
            BusinessAgent(config=invalid_config, mic=mock_mic)
    
    def test_init_non_dict_config_raises(self, mock_mic: MagicMock):
        """Config no-dict lanza error."""
        with pytest.raises(ConfigurationError, match="diccionario"):
            BusinessAgent(config="not a dict", mic=mock_mic)  # type: ignore
    
    def test_evaluate_project_success(
        self,
        business_agent: BusinessAgent,
        valid_context: Dict[str, Any]
    ):
        """Evaluación exitosa retorna reporte."""
        # Mock para nx
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            mock_nx.number_connected_components.return_value = 1
            
            report = business_agent.evaluate_project(valid_context)
        
        assert report is not None
        assert hasattr(report, 'integrity_score')
    
    def test_evaluate_project_none_presupuesto(
        self,
        business_agent: BusinessAgent
    ):
        """Presupuesto None retorna None."""
        context = {"df_presupuesto": None, "df_merged": pd.DataFrame({"a": [1]})}
        
        report = business_agent.evaluate_project(context)
        
        assert report is None
    
    def test_evaluate_project_empty_presupuesto(
        self,
        business_agent: BusinessAgent
    ):
        """Presupuesto vacío retorna reporte mínimo (no None)."""
        context = {
            "df_presupuesto": pd.DataFrame(),
            "df_merged": pd.DataFrame({"a": [1]}),
            "validated_strata": {"PHYSICS", "TACTICS"}
        }
        
        report = business_agent.evaluate_project(context)
        
        # Debe retornar reporte con score 0 según implementación
        assert report is not None
        assert report.integrity_score == 0.0
    
    def test_evaluate_project_missing_strata_raises(
        self,
        business_agent: BusinessAgent,
        valid_context: Dict[str, Any]
    ):
        """Estratos faltantes causan error en proyección financiera."""
        # Quitar estratos validados
        context = {**valid_context, "validated_strata": set()}
        
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            mock_nx.number_connected_components.return_value = 1
            
            # Debe fallar en fase financiera
            report = business_agent.evaluate_project(context)
        
        # Retorna None porque la fase financiera falla
        assert report is None
    
    def test_evaluate_uses_df_final_if_present(
        self,
        business_agent: BusinessAgent,
        valid_budget_df: pd.DataFrame,
        valid_detail_df: pd.DataFrame
    ):
        """Usa df_final si está presente en lugar de df_presupuesto."""
        df_final = valid_budget_df.copy()
        df_final["extra_column"] = "marker"
        
        context = {
            "df_final": df_final,
            "df_presupuesto": valid_budget_df,  # Se ignora
            "df_merged": valid_detail_df,
            "validated_strata": {"PHYSICS", "TACTICS"}
        }
        
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            
            report = business_agent.evaluate_project(context)
        
        # No falla aunque df_final tenga columna extra
        assert report is not None
    
    def test_properties_accessible(self, business_agent: BusinessAgent):
        """Propiedades del agente son accesibles."""
        assert business_agent.config is not None
        assert business_agent.mic is not None
        assert business_agent.telemetry is not None
        assert business_agent.risk_challenger is not None


@pytest.mark.integration
class TestBusinessAgentIntegration:
    """Pruebas de integración avanzadas."""
    
    def test_full_pipeline_viable_project(
        self,
        default_config: Dict[str, Any],
        mock_mic: MagicMock,
        mock_telemetry: MagicMock
    ):
        """Pipeline completo para proyecto viable."""
        # Crear agente con mocks
        mock_graph = MockFactory.create_mock_graph(n_nodes=50, n_edges=75)
        mock_builder = MockFactory.create_mock_graph_builder(mock_graph)
        mock_analyzer = MockFactory.create_mock_topology_analyzer(
            betti=BettiNumbers(beta_0=1, beta_1=0),
            stability=0.95
        )
        mock_translator = MockFactory.create_mock_translator()
        
        agent = BusinessAgent(
            config=default_config,
            mic=mock_mic,
            telemetry=mock_telemetry,
            graph_builder=mock_builder,
            topology_analyzer=mock_analyzer,
            semantic_translator=mock_translator
        )
        
        context = ContextFactory.create_evaluation_context(
            initial_investment=1_000_000.0
        )
        
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            mock_nx.number_connected_components.return_value = 1
            
            report = agent.evaluate_project(context)
        
        assert report is not None
        assert report.integrity_score > 50.0  # Proyecto viable
    
    def test_full_pipeline_unstable_project(
        self,
        default_config: Dict[str, Any],
        mock_mic: MagicMock,
        mock_telemetry: MagicMock
    ):
        """Pipeline completo para proyecto inestable."""
        # MIC que rechaza pivotes
        mock_mic.project_intent.side_effect = lambda name, payload, ctx: (
            {"success": True, "results": {"npv": 500000, "irr": 0.15}}
            if name == "financial_analysis"
            else {"success": False}
        )
        
        mock_graph = MockFactory.create_mock_graph()
        mock_builder = MockFactory.create_mock_graph_builder(mock_graph)
        mock_analyzer = MockFactory.create_mock_topology_analyzer(
            betti=BettiNumbers(beta_0=1, beta_1=5),  # Ciclos
            stability=0.50  # Inestable
        )
        
        agent = BusinessAgent(
            config=default_config,
            mic=mock_mic,
            telemetry=mock_telemetry,
            graph_builder=mock_builder,
            topology_analyzer=mock_analyzer
        )
        
        context = ContextFactory.create_evaluation_context()
        
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            
            report = agent.evaluate_project(context)
        
        assert report is not None
        # Debe tener penalizaciones
        assert "challenger_applied" in report.details or "challenger_verdict" in report.details


# =============================================================================
# TESTS: FACTORIES Y FUNCIONES DE CONVENIENCIA
# =============================================================================

@pytest.mark.unit
class TestFactoryFunctions:
    """Pruebas para funciones factory."""
    
    def test_create_business_agent(
        self,
        default_config: Dict[str, Any],
        mock_mic: MagicMock
    ):
        """create_business_agent crea instancia correctamente."""
        agent = create_business_agent(default_config, mock_mic)
        
        assert isinstance(agent, BusinessAgent)
        assert agent.config == default_config
    
    def test_evaluate_project_function(
        self,
        default_config: Dict[str, Any],
        mock_mic: MagicMock
    ):
        """evaluate_project como función standalone."""
        context = ContextFactory.create_minimal_context()
        
        # Mockeamos las dependencias internas
        with patch.object(BusinessAgent, 'evaluate_project') as mock_eval:
            mock_eval.return_value = MagicMock(integrity_score=75.0)
            
            result = evaluate_project(context, default_config, mock_mic)
        
        # La función debe crear agente y llamar evaluate_project
        assert result is not None


# =============================================================================
# TESTS: EDGE CASES Y BOUNDARY CONDITIONS
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Pruebas de casos límite."""
    
    def test_extreme_stability_values(self, risk_challenger: RiskChallenger):
        """Valores extremos de estabilidad."""
        for stability in [0.0, 0.001, 0.999, 1.0]:
            report = MagicMock()
            report.integrity_score = 80.0
            report.financial_risk_level = "SAFE"
            report.waste_alerts = []
            report.circular_risks = []
            report.complexity_level = "Media"
            report.strategic_narrative = "Test"
            report.details = {
                "pyramid_stability": stability,
                "topological_invariants": {"betti_numbers": {"beta_0": 1, "beta_1": 0}}
            }
            
            # No debe lanzar excepción
            result = risk_challenger.challenge_verdict(report)
            assert result is not None
    
    def test_extreme_betti_numbers(self):
        """Números de Betti extremos."""
        # Muchos componentes
        betti_fragmented = BettiNumbers(beta_0=100, beta_1=0)
        assert betti_fragmented.is_connected is False
        
        # Muchos ciclos
        betti_cyclic = BettiNumbers(beta_0=1, beta_1=1000)
        assert betti_cyclic.has_cycles is True
        assert betti_cyclic.euler_characteristic == 1 - 1000
    
    def test_zero_nodes_topology_bundle(self):
        """Bundle con cero nodos."""
        bundle = TopologicalMetricsBundle(
            betti=BettiNumbers(),
            pyramid_stability=1.0,
            n_nodes=0,
            n_edges=0,
            is_connected=False
        )
        
        # cycle_density no debe dividir por cero
        assert bundle.cycle_density == 0.0
        # structural_coherence debe manejarlo
        assert 0.0 <= bundle.structural_coherence <= 1.0
    
    def test_nan_npv_handling(self, stable_topology_bundle: TopologicalMetricsBundle):
        """NaN en NPV se maneja correctamente."""
        financial = {"npv": float('nan'), "irr": 0.15}
        
        # compute_quality_factors debe manejar NaN
        _, fin_q, _ = DecisionAlgebra.compute_quality_factors(
            topo_bundle=stable_topology_bundle,
            financial_metrics=financial,
            entropy=0.5,
            exergy=0.5
        )
        
        # Debe retornar valor válido (0.5 por tanh(0))
        assert 0.0 <= fin_q <= 1.0
    
    def test_inf_npv_handling(self, stable_topology_bundle: TopologicalMetricsBundle):
        """Infinito en NPV se maneja correctamente."""
        financial = {"npv": float('inf'), "irr": 0.15}
        
        _, fin_q, _ = DecisionAlgebra.compute_quality_factors(
            topo_bundle=stable_topology_bundle,
            financial_metrics=financial,
            entropy=0.5,
            exergy=0.5
        )
        
        # tanh(inf) = 1.0, normalizado a (1+1)/2 = 1.0
        assert fin_q == pytest.approx(1.0, rel=0.01)


# =============================================================================
# TESTS: PROPIEDADES ALGEBRAICAS
# =============================================================================

@pytest.mark.algebraic
class TestAlgebraicProperties:
    """Pruebas de propiedades algebraicas del sistema."""
    
    def test_structural_coherence_bounded(self):
        """Coherencia estructural siempre en [0, 1]."""
        test_cases = [
            {"beta_0": 1, "beta_1": 0, "stability": 1.0},
            {"beta_0": 10, "beta_1": 5, "stability": 0.5},
            {"beta_0": 1, "beta_1": 100, "stability": 0.1},
            {"beta_0": 100, "beta_1": 100, "stability": 0.0},
        ]
        
        for case in test_cases:
            bundle = TopologicalMetricsBundle(
                betti=BettiNumbers(beta_0=case["beta_0"], beta_1=case["beta_1"]),
                pyramid_stability=case["stability"],
                n_nodes=100,
                n_edges=150,
                is_connected=case["beta_0"] == 1
            )
            
            coherence = bundle.structural_coherence
            assert 0.0 <= coherence <= 1.0, f"Coherence {coherence} out of bounds for {case}"
    
    def test_geometric_mean_identity(self):
        """Media geométrica de factores iguales es el mismo factor."""
        factor = 5.0
        factors = [factor, factor, factor]
        
        result = DecisionAlgebra.weighted_geometric_mean(factors)
        
        assert result == pytest.approx(factor, rel=1e-6)
    
    def test_normalization_idempotent(self):
        """Normalizar un vector normalizado no cambia nada."""
        v = np.array([0.6, 0.8])  # Ya normalizado
        
        result1 = DecisionAlgebra.normalize_to_sphere(v)
        result2 = DecisionAlgebra.normalize_to_sphere(result1)
        
        assert np.allclose(result1, result2)
    
    def test_decision_weights_normalization_idempotent(self):
        """Normalizar pesos ya normalizados es idempotente."""
        weights = DecisionWeights(topology=0.4, finance=0.4, thermodynamics=0.2)
        
        norm1 = weights.normalized
        norm2 = norm1.normalized
        
        assert norm1.topology == pytest.approx(norm2.topology)
        assert norm1.finance == pytest.approx(norm2.finance)
        assert norm1.thermodynamics == pytest.approx(norm2.thermodynamics)


# =============================================================================
# TESTS: REGRESIÓN
# =============================================================================

@pytest.mark.regression
class TestRegression:
    """Tests de regresión para bugs conocidos."""
    
    def test_regression_empty_details_in_report(self, risk_challenger: RiskChallenger):
        """
        Regresión: details=None no debe causar error.
        
        Bug: challenge_verdict fallaba con details=None.
        """
        report = MagicMock()
        report.integrity_score = 80.0
        report.financial_risk_level = "SAFE"
        report.waste_alerts = []
        report.circular_risks = []
        report.complexity_level = "Media"
        report.strategic_narrative = "Test"
        report.details = None  # None en lugar de dict
        
        # No debe lanzar excepción
        result = risk_challenger.challenge_verdict(report)
        assert result is not None
    
    def test_regression_list_validated_strata(
        self,
        business_agent: BusinessAgent,
        valid_context: Dict[str, Any]
    ):
        """
        Regresión: validated_strata como lista en lugar de set.
        
        Bug: JSON deserializa sets como listas, causando error.
        """
        context = {**valid_context, "validated_strata": ["PHYSICS", "TACTICS"]}
        
        with patch('app.business_agent.nx') as mock_nx:
            mock_nx.is_connected.return_value = True
            
            # No debe fallar por tipo de validated_strata
            report = business_agent.evaluate_project(context)
        
        assert report is not None


# =============================================================================
# TESTS: PERFORMANCE (Opcional)
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Tests de rendimiento."""
    
    def test_validation_performance(self, validator: DataFrameValidator):
        """La validación debe ser rápida para DataFrames grandes."""
        import time
        
        # DataFrames grandes
        df_budget = DataFrameFactory.create_budget_df(n_rows=10000)
        apu_codes = df_budget[ColumnNames.CODIGO_APU].tolist()
        df_detail = DataFrameFactory.create_detail_df(n_rows=50000, apu_codes=apu_codes)
        
        start = time.monotonic()
        result = validator.validate(df_budget, df_detail)
        elapsed = time.monotonic() - start
        
        assert result.is_valid is True
        assert elapsed < 5.0, f"Validación tomó {elapsed:.2f}s (esperado < 5s)"
    
    def test_coherence_calculation_performance(self):
        """Cálculo de coherencia debe ser rápido."""
        import time
        
        bundle = MetricsFactory.create_topology_bundle(
            n_nodes=10000,
            n_edges=50000,
            beta_1=100
        )
        
        start = time.monotonic()
        for _ in range(1000):
            _ = bundle.structural_coherence
        elapsed = time.monotonic() - start
        
        assert elapsed < 1.0, f"1000 cálculos tomaron {elapsed:.2f}s"


# =============================================================================
# UTILIDADES DE PRUEBA
# =============================================================================

def assert_report_valid(report: Any) -> None:
    """Verifica que un reporte tenga estructura válida."""
    assert report is not None
    assert hasattr(report, 'integrity_score')
    assert hasattr(report, 'waste_alerts')
    assert hasattr(report, 'circular_risks')
    assert hasattr(report, 'details')
    assert 0 <= report.integrity_score <= 100


def assert_bundle_valid(bundle: TopologicalMetricsBundle) -> None:
    """Verifica que un bundle topológico sea válido."""
    assert bundle.betti.beta_0 >= 0
    assert bundle.betti.beta_1 >= 0
    assert 0 <= bundle.pyramid_stability <= 1
    assert bundle.n_nodes >= 0
    assert 0 <= bundle.structural_coherence <= 1


def create_mock_report(
    *,
    integrity: float = 80.0,
    stability: float = 0.9,
    beta_1: int = 0,
    financial_risk: str = "MODERATE"
) -> MagicMock:
    """Helper para crear mock de ConstructionRiskReport."""
    report = MagicMock()
    report.integrity_score = integrity
    report.financial_risk_level = financial_risk
    report.waste_alerts = []
    report.circular_risks = []
    report.complexity_level = "Media"
    report.strategic_narrative = "Test narrative"
    report.details = {
        "pyramid_stability": stability,
        "topological_invariants": {
            "betti_numbers": {"beta_0": 1, "beta_1": beta_1},
            "n_nodes": 50,
        },
        "thermal_metrics": {"system_temperature": 0.3},
        "financial_metrics": {"npv": 500000},
    }
    return report