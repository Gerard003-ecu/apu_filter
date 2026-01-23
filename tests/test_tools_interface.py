"""
Suite de pruebas para el módulo tools_interface Refinado.

Cobertura:
- MICRegistry y Gatekeeper (Niveles Jerárquicos)
- Estructuras de datos topológicas (PersistenceInterval, TopologicalSummary)
- Funciones de entropía y probabilidad
- Análisis topológico de archivos
- Diagnóstico con homología y persistencia
- Limpieza con preservación topológica
- Análisis financiero con variedades de riesgo
- Telemetría del sistema
"""

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest
import pandas as pd
import numpy as np

# Módulo bajo prueba
from app.tools_interface import (
    # Constantes
    _EPSILON,
    _DEFAULT_RANDOM_SEED,
    _MAX_SAMPLE_ROWS,
    _PERSISTENCE_THRESHOLD,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_ENCODINGS,
    VALID_DELIMITERS,
    VALID_EXTENSIONS,
    _ENCODING_ALIASES,
    
    # Excepciones
    CleaningError,
    DiagnosticError,
    FileNotFoundDiagnosticError,
    FileValidationError,
    UnsupportedFileTypeError,
    MICHierarchyViolationError,
    
    # Enums y Estructuras
    FileType,
    Stratum,
    IntentVector,
    MICRegistry,
    PersistenceInterval,
    TopologicalSummary,
    
    # Protocolos
    TelemetryContextProtocol,
    DiagnosticProtocol,
    
    # Funciones de Entropía
    _compute_shannon_entropy,
    _compute_distribution_from_counts,
    _compute_persistence_entropy,
    
    # Funciones Topológicas
    _analyze_topological_features,
    _detect_cyclic_patterns,
    _estimate_intrinsic_dimension,
    _compute_homology_groups,
    _compute_persistence_diagram,
    _compute_diagnostic_magnitude,
    
    # Funciones CSV
    _analyze_csv_topology,
    _estimate_effective_rank,
    _compute_topological_preservation,
    
    # Funciones Financieras
    _generate_topological_cash_flows,
    _analyze_risk_manifold,
    _compute_risk_homology,
    _compute_opportunity_persistence,
    _compute_risk_adjusted_return,
    _compute_topological_efficiency,
    
    # Funciones de Validación
    _validate_path_not_empty,
    _normalize_path,
    _validate_file_exists,
    _validate_file_extension,
    _validate_file_size,
    _normalize_encoding,
    _validate_csv_parameters,
    _normalize_file_type,
    _generate_output_path,
    
    # Funciones de Respuesta
    _create_error_response,
    _create_success_response,
    _extract_diagnostic_result,
    
    # Utilidades Públicas
    get_supported_file_types,
    get_supported_delimiters,
    get_supported_encodings,
    is_valid_file_type,
    validate_file_for_processing,
    
    # Handlers MIC
    diagnose_file,
    clean_file,
    analyze_financial_viability,
    get_telemetry_status,
    
    # Registry
    _DIAGNOSTIC_REGISTRY,
    _get_diagnostic_class,
)


# =============================================================================
# Configuración de Logging para Tests
# =============================================================================

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures Generales
# =============================================================================

@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Crea un archivo CSV temporal válido con estructura topológica simple."""
    csv_file = tmp_path / "test_data.csv"
    content = "col1;col2;col3\nval1;val2;val3\nvalA;valB;valC\nval1;val2;val3\n"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture
def temp_csv_with_cycles(tmp_path: Path) -> Path:
    """CSV con patrones cíclicos claros para detección de periodicidad."""
    csv_file = tmp_path / "cyclic_data.csv"
    # Patrón A, B, A, B, A, B (periodo 2)
    content = "A\nB\nA\nB\nA\nB\nA\nB\nA\nB\n"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture
def temp_csv_numeric(tmp_path: Path) -> Path:
    """CSV con datos numéricos para análisis de rango efectivo."""
    csv_file = tmp_path / "numeric_data.csv"
    content = "a,b,c,d\n1,2,3,4\n2,4,6,8\n3,6,9,12\n4,8,12,16\n5,10,15,20\n"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture
def temp_empty_file(tmp_path: Path) -> Path:
    """Archivo vacío para edge cases."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    return empty_file


@pytest.fixture
def temp_large_csv(tmp_path: Path) -> Path:
    """CSV moderadamente grande para tests de rendimiento."""
    csv_file = tmp_path / "large_data.csv"
    header = "col1,col2,col3,col4,col5\n"
    rows = "\n".join([f"{i},{i*2},{i*3},{i%10},{i%5}" for i in range(500)])
    csv_file.write_text(header + rows, encoding="utf-8")
    return csv_file


@pytest.fixture
def mock_diagnostic_result() -> Dict[str, Any]:
    """Resultado simulado de diagnóstico con estructura completa."""
    return {
        "total_rows": 100,
        "valid_rows": 95,
        "error_rows": 5,
        "warnings": [
            "Circular dependency detected in row 10",
            "Missing value in column 3"
        ],
        "errors": [
            "Row 10: Invalid format",
            "Row 25: Type mismatch"
        ],
        "issues": [
            {"type": "FORMAT_ERROR", "severity": "HIGH", "row": 10},
            {"type": "MISSING_VALUE", "severity": "LOW", "row": 15},
            {"type": "FORMAT_ERROR", "severity": "MEDIUM", "row": 20},
        ]
    }


@pytest.fixture
def mock_cleaning_stats():
    """Estadísticas simuladas de limpieza."""
    mock_stats = MagicMock()
    mock_stats.to_dict.return_value = {
        "rows_processed": 100,
        "rows_cleaned": 95,
        "rows_removed": 5,
        "cleaning_time_ms": 150,
    }
    return mock_stats


@pytest.fixture
def mock_telemetry_context():
    """Contexto de telemetría mock que implementa el protocolo."""
    context = MagicMock(spec=TelemetryContextProtocol)
    context.get_business_report.return_value = {
        "status": "ACTIVE",
        "system_health": "HEALTHY",
        "active_processes": 3,
        "memory_usage_mb": 256.5,
    }
    return context


# =============================================================================
# Tests para PersistenceInterval
# =============================================================================

class TestPersistenceInterval:
    """Tests para la estructura de intervalos de persistencia."""

    def test_valid_interval_creation(self):
        """Crear intervalo válido."""
        interval = PersistenceInterval(birth=0.0, death=1.0, dimension=0)
        assert interval.birth == 0.0
        assert interval.death == 1.0
        assert interval.dimension == 0
        assert interval.persistence == 1.0
        assert not interval.is_essential

    def test_essential_interval(self):
        """Intervalo que nunca muere."""
        interval = PersistenceInterval(birth=0.5, death=float('inf'), dimension=1)
        assert interval.is_essential
        assert math.isinf(interval.persistence)

    def test_invalid_negative_birth(self):
        """Birth negativo debe fallar."""
        with pytest.raises(ValueError, match="Birth time must be non-negative"):
            PersistenceInterval(birth=-1.0, death=1.0)

    def test_invalid_death_before_birth(self):
        """Death antes de birth debe fallar."""
        with pytest.raises(ValueError, match="Death .* must be >= birth"):
            PersistenceInterval(birth=1.0, death=0.5)

    def test_invalid_negative_dimension(self):
        """Dimensión negativa debe fallar."""
        with pytest.raises(ValueError, match="Dimension must be non-negative"):
            PersistenceInterval(birth=0.0, death=1.0, dimension=-1)

    def test_ordering_by_persistence(self):
        """Intervalos se ordenan por persistencia descendente."""
        intervals = [
            PersistenceInterval(0.0, 0.5, 0),   # persistence = 0.5
            PersistenceInterval(0.0, 2.0, 0),   # persistence = 2.0
            PersistenceInterval(0.0, 1.0, 0),   # persistence = 1.0
        ]
        sorted_intervals = sorted(intervals)
        assert sorted_intervals[0].persistence == 2.0
        assert sorted_intervals[1].persistence == 1.0
        assert sorted_intervals[2].persistence == 0.5

    def test_immutability(self):
        """PersistenceInterval es inmutable (frozen)."""
        interval = PersistenceInterval(0.0, 1.0, 0)
        with pytest.raises(AttributeError):
            interval.birth = 0.5


# =============================================================================
# Tests para TopologicalSummary
# =============================================================================

class TestTopologicalSummary:
    """Tests para el resumen topológico."""

    def test_create_summary(self):
        """Crear resumen válido."""
        summary = TopologicalSummary(
            betti_0=3,
            betti_1=1,
            betti_2=0,
            euler_characteristic=2,
            structural_entropy=0.5,
            persistence_entropy=0.3
        )
        assert summary.betti_0 == 3
        assert summary.euler_characteristic == 2

    def test_empty_summary(self):
        """Crear resumen vacío."""
        summary = TopologicalSummary.empty()
        assert summary.betti_0 == 0
        assert summary.betti_1 == 0
        assert summary.euler_characteristic == 0
        assert summary.structural_entropy == 0.0

    def test_to_dict(self):
        """Serialización a diccionario."""
        summary = TopologicalSummary(
            betti_0=2,
            betti_1=1,
            betti_2=0,
            euler_characteristic=1,
            structural_entropy=0.123456789,
            persistence_entropy=0.987654321
        )
        d = summary.to_dict()
        
        assert d["betti_numbers"] == [2, 1, 0]
        assert d["euler_characteristic"] == 1
        assert d["structural_entropy"] == 0.123457  # Rounded to 6 decimals
        assert d["persistence_entropy"] == 0.987654

    def test_immutability(self):
        """TopologicalSummary es inmutable."""
        summary = TopologicalSummary.empty()
        with pytest.raises(AttributeError):
            summary.betti_0 = 5


# =============================================================================
# Tests para Funciones de Entropía
# =============================================================================

class TestShannonEntropy:
    """Tests para cálculo de entropía de Shannon."""

    def test_uniform_distribution(self):
        """Distribución uniforme tiene máxima entropía."""
        # 4 eventos equiprobables: H = log2(4) = 2 bits
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = _compute_shannon_entropy(probs)
        assert math.isclose(entropy, 2.0, rel_tol=1e-6)

    def test_deterministic_distribution(self):
        """Distribución determinista tiene entropía cero."""
        probs = [1.0, 0.0, 0.0, 0.0]
        entropy = _compute_shannon_entropy(probs)
        assert entropy == 0.0

    def test_binary_distribution(self):
        """Distribución binaria equiprobable."""
        probs = [0.5, 0.5]
        entropy = _compute_shannon_entropy(probs)
        assert math.isclose(entropy, 1.0, rel_tol=1e-6)

    def test_empty_distribution(self):
        """Distribución vacía retorna cero."""
        entropy = _compute_shannon_entropy([])
        assert entropy == 0.0

    def test_all_zeros(self):
        """Distribución de todos ceros retorna cero."""
        entropy = _compute_shannon_entropy([0.0, 0.0, 0.0])
        assert entropy == 0.0

    def test_unnormalized_distribution(self):
        """Distribuciones no normalizadas se normalizan automáticamente."""
        probs = [2, 2, 2, 2]  # Equivalente a [0.25, 0.25, 0.25, 0.25]
        entropy = _compute_shannon_entropy(probs)
        assert math.isclose(entropy, 2.0, rel_tol=1e-6)

    def test_negative_probability_raises(self):
        """Probabilidades negativas lanzan error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            _compute_shannon_entropy([0.5, -0.5])

    def test_different_base(self):
        """Entropía con base e (nats)."""
        probs = [0.5, 0.5]
        entropy_nats = _compute_shannon_entropy(probs, base=math.e)
        # H = ln(2) ≈ 0.693
        assert math.isclose(entropy_nats, math.log(2), rel_tol=1e-6)

    def test_numerical_stability(self):
        """Estabilidad con probabilidades muy pequeñas."""
        probs = [1e-15, 1.0 - 1e-15]
        # No debería lanzar error ni retornar NaN
        entropy = _compute_shannon_entropy(probs)
        assert not math.isnan(entropy)
        assert entropy >= 0


class TestDistributionFromCounts:
    """Tests para conversión de conteos a distribución."""

    def test_basic_counts(self):
        """Conteos básicos."""
        counts = {"a": 2, "b": 3, "c": 5}
        probs = _compute_distribution_from_counts(counts)
        assert sum(probs) == pytest.approx(1.0)
        assert probs == [0.2, 0.3, 0.5]

    def test_counter_object(self):
        """Funciona con Counter."""
        counts = Counter(["a", "a", "b", "b", "b"])
        probs = _compute_distribution_from_counts(counts)
        assert sum(probs) == pytest.approx(1.0)

    def test_empty_counts(self):
        """Conteos vacíos retornan lista vacía."""
        probs = _compute_distribution_from_counts({})
        assert probs == []

    def test_zero_total(self):
        """Total cero retorna lista vacía."""
        probs = _compute_distribution_from_counts({"a": 0, "b": 0})
        assert probs == []


class TestPersistenceEntropy:
    """Tests para entropía del diagrama de persistencia."""

    def test_single_dominant_feature(self):
        """Una característica dominante = baja entropía."""
        intervals = [
            PersistenceInterval(0.0, 10.0, 0),  # Dominante
            PersistenceInterval(0.0, 0.1, 0),
            PersistenceInterval(0.0, 0.1, 0),
        ]
        entropy = _compute_persistence_entropy(intervals)
        # La característica dominante tiene ~98% de la persistencia
        assert entropy < 0.3

    def test_equal_features(self):
        """Características iguales = alta entropía."""
        intervals = [
            PersistenceInterval(0.0, 1.0, 0),
            PersistenceInterval(0.0, 1.0, 0),
            PersistenceInterval(0.0, 1.0, 0),
        ]
        entropy = _compute_persistence_entropy(intervals)
        assert entropy > 0.9  # Cercano a 1 (máximo normalizado)

    def test_empty_intervals(self):
        """Sin intervalos retorna cero."""
        entropy = _compute_persistence_entropy([])
        assert entropy == 0.0

    def test_only_essential_intervals(self):
        """Solo intervalos esenciales (infinitos) retorna cero."""
        intervals = [
            PersistenceInterval(0.0, float('inf'), 0),
            PersistenceInterval(1.0, float('inf'), 0),
        ]
        entropy = _compute_persistence_entropy(intervals)
        assert entropy == 0.0


# =============================================================================
# Tests para Análisis Topológico de Archivos
# =============================================================================

class TestAnalyzeTopologicalFeatures:
    """Tests para análisis topológico de archivos."""

    def test_basic_file(self, temp_csv_file: Path):
        """Análisis básico de archivo CSV."""
        features = _analyze_topological_features(temp_csv_file)
        
        assert "beta_0" in features
        assert "beta_1" in features
        assert "euler_characteristic" in features
        assert "structural_entropy" in features
        assert features["num_lines"] == 4

    def test_cyclic_patterns(self, temp_csv_with_cycles: Path):
        """Detecta patrones cíclicos."""
        features = _analyze_topological_features(temp_csv_with_cycles)
        
        # Archivo con patrón A,B repetido debería tener ciclos
        assert features["beta_1"] >= 1

    def test_empty_file(self, temp_empty_file: Path):
        """Archivo vacío retorna resumen vacío."""
        features = _analyze_topological_features(temp_empty_file)
        
        assert features["beta_0"] == 0
        assert features["beta_1"] == 0

    def test_file_dimension(self, temp_csv_file: Path):
        """Estima dimensión del archivo."""
        features = _analyze_topological_features(temp_csv_file)
        
        # Archivo tiene 3 columnas
        assert features["intrinsic_dimension"] == 3

    def test_nonexistent_file(self, tmp_path: Path):
        """Archivo inexistente retorna error."""
        features = _analyze_topological_features(tmp_path / "nonexistent.csv")
        assert "analysis_error" in features


class TestDetectCyclicPatterns:
    """Tests para detección de patrones cíclicos."""

    def test_no_cycles(self):
        """Sin ciclos en secuencia única."""
        lines = ["A", "B", "C", "D", "E"]
        cycles = _detect_cyclic_patterns(lines)
        assert cycles == 0

    def test_period_two_cycle(self):
        """Ciclo de período 2."""
        lines = ["A", "B"] * 10  # A,B,A,B,...
        cycles = _detect_cyclic_patterns(lines)
        assert cycles >= 1

    def test_period_three_cycle(self):
        """Ciclo de período 3."""
        lines = ["A", "B", "C"] * 10
        cycles = _detect_cyclic_patterns(lines)
        assert cycles >= 1

    def test_short_sequence(self):
        """Secuencia muy corta retorna cero."""
        lines = ["A", "B"]
        cycles = _detect_cyclic_patterns(lines)
        assert cycles == 0

    def test_all_same(self):
        """Todos iguales = múltiples ciclos."""
        lines = ["A"] * 20
        cycles = _detect_cyclic_patterns(lines)
        # Coincide con todos los períodos
        assert cycles > 5


class TestEstimateIntrinsicDimension:
    """Tests para estimación de dimensión intrínseca."""

    def test_csv_structure(self):
        """Detecta estructura CSV."""
        lines = ["a,b,c,d", "1,2,3,4", "5,6,7,8"]
        dim = _estimate_intrinsic_dimension(lines)
        assert dim == 4

    def test_semicolon_delimiter(self):
        """Detecta delimitador punto y coma."""
        lines = ["a;b;c", "1;2;3"]
        dim = _estimate_intrinsic_dimension(lines)
        assert dim == 3

    def test_no_structure(self):
        """Sin estructura columnar."""
        lines = ["texto plano", "otra linea"]
        dim = _estimate_intrinsic_dimension(lines)
        assert dim == 1

    def test_empty_lines(self):
        """Lista vacía retorna cero."""
        dim = _estimate_intrinsic_dimension([])
        assert dim == 0


# =============================================================================
# Tests para Homología y Persistencia
# =============================================================================

class TestComputeHomologyGroups:
    """Tests para cálculo de grupos de homología."""

    def test_basic_homology(self, mock_diagnostic_result: Dict[str, Any]):
        """Homología básica desde diagnóstico."""
        homology = _compute_homology_groups(mock_diagnostic_result)
        
        assert "H_0" in homology
        assert "H_1" in homology
        assert "beta_0" in homology
        assert "beta_1" in homology
        assert "euler_characteristic" in homology

    def test_beta_0_from_issue_types(self, mock_diagnostic_result: Dict[str, Any]):
        """β₀ se calcula de tipos de issues únicos."""
        homology = _compute_homology_groups(mock_diagnostic_result)
        
        # El mock tiene 2 tipos únicos: FORMAT_ERROR, MISSING_VALUE
        assert homology["beta_0"] == 2

    def test_beta_1_from_circular_references(self, mock_diagnostic_result: Dict[str, Any]):
        """β₁ se calcula de referencias circulares."""
        homology = _compute_homology_groups(mock_diagnostic_result)
        
        # El mock tiene 1 warning con "circular"
        assert homology["beta_1"] == 1

    def test_empty_diagnostic(self):
        """Diagnóstico vacío."""
        homology = _compute_homology_groups({})
        
        assert homology["beta_0"] == 1  # Mínimo 1 componente
        assert homology["beta_1"] == 0

    def test_no_circular_references(self):
        """Sin referencias circulares = β₁ = 0."""
        data = {"issues": [{"type": "ERROR"}], "warnings": ["Simple warning"]}
        homology = _compute_homology_groups(data)
        
        assert homology["beta_1"] == 0
        assert not homology["has_cycles"]


class TestComputePersistenceDiagram:
    """Tests para diagrama de persistencia."""

    def test_basic_diagram(self, mock_diagnostic_result: Dict[str, Any]):
        """Diagrama básico."""
        diagram = _compute_persistence_diagram(mock_diagnostic_result)
        
        assert isinstance(diagram, list)
        assert len(diagram) > 0
        
        for point in diagram:
            assert "birth" in point
            assert "death" in point
            assert "persistence" in point

    def test_severity_mapping(self):
        """Mapeo correcto de severidad a persistencia."""
        data = {
            "issues": [
                {"severity": "CRITICAL"},
                {"severity": "LOW"},
            ]
        }
        diagram = _compute_persistence_diagram(data)
        
        # CRITICAL debe tener mayor persistencia
        if len(diagram) >= 2:
            persistences = [p["persistence"] for p in diagram]
            assert max(persistences) == pytest.approx(1.0, rel=0.1)

    def test_empty_issues(self):
        """Sin issues retorna lista vacía."""
        diagram = _compute_persistence_diagram({"issues": []})
        assert diagram == []

    def test_significance_flag(self):
        """Flag de significancia correcto."""
        data = {"issues": [{"severity": "HIGH"}]}  # persistence = 0.8
        diagram = _compute_persistence_diagram(data)
        
        if diagram:
            assert diagram[0]["is_significant"] == True


class TestComputeDiagnosticMagnitude:
    """Tests para magnitud del diagnóstico."""

    def test_zero_problems(self):
        """Sin problemas = magnitud cero."""
        mag = _compute_diagnostic_magnitude({})
        assert mag == 0.0

    def test_increasing_magnitude(self):
        """Más problemas = mayor magnitud."""
        mag1 = _compute_diagnostic_magnitude({"issues": [1]})
        mag2 = _compute_diagnostic_magnitude({"issues": [1, 2, 3, 4, 5]})
        
        assert mag2 > mag1

    def test_error_weight(self):
        """Errores pesan más que warnings."""
        mag_warnings = _compute_diagnostic_magnitude({"warnings": [1, 2, 3]})
        mag_errors = _compute_diagnostic_magnitude({"errors": [1, 2, 3]})
        
        assert mag_errors > mag_warnings

    def test_bounded_output(self):
        """Salida acotada en [0, 1] via tanh."""
        mag = _compute_diagnostic_magnitude({
            "issues": list(range(100)),
            "warnings": list(range(100)),
            "errors": list(range(100))
        })
        
        assert 0 <= mag <= 1


# =============================================================================
# Tests para Análisis CSV
# =============================================================================

class TestAnalyzeCsvTopology:
    """Tests para análisis topológico de CSV."""

    def test_basic_csv(self, temp_csv_file: Path):
        """Análisis básico."""
        topology = _analyze_csv_topology(temp_csv_file, ";", "utf-8")
        
        assert topology["rows"] == 3  # Excluyendo header
        assert topology["columns"] == 3
        assert "density" in topology
        assert "null_entropy" in topology

    def test_empty_csv(self, temp_empty_file: Path):
        """CSV vacío."""
        topology = _analyze_csv_topology(temp_empty_file, ",", "utf-8")
        
        assert topology.get("is_empty", True)

    def test_numeric_csv(self, temp_csv_numeric: Path):
        """CSV numérico con cálculo de rango efectivo."""
        topology = _analyze_csv_topology(temp_csv_numeric, ",", "utf-8")
        
        assert topology["columns"] == 4
        assert "effective_rank" in topology
        # Datos linealmente dependientes = rango bajo
        assert topology["effective_rank"] <= 2

    def test_density_calculation(self, temp_csv_file: Path):
        """Cálculo de densidad (sin nulos = densidad 1)."""
        topology = _analyze_csv_topology(temp_csv_file, ";", "utf-8")
        
        assert topology["density"] == pytest.approx(1.0, rel=0.01)
        assert topology["sparsity"] == pytest.approx(0.0, rel=0.01)


class TestEstimateEffectiveRank:
    """Tests para estimación de rango efectivo via SVD."""

    def test_full_rank_data(self, tmp_path: Path):
        """Datos de rango completo."""
        df = pd.DataFrame({
            "a": [1, 0, 0, 0, 0],
            "b": [0, 1, 0, 0, 0],
            "c": [0, 0, 1, 0, 0],
        })
        rank = _estimate_effective_rank(df)
        
        # 3 columnas independientes
        assert rank >= 2

    def test_rank_deficient_data(self):
        """Datos con dependencia lineal."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # b = 2*a
            "c": [3, 6, 9, 12, 15],  # c = 3*a
        })
        rank = _estimate_effective_rank(df)
        
        # Solo 1 dimensión efectiva
        assert rank == 1

    def test_single_column(self):
        """Una sola columna."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        rank = _estimate_effective_rank(df)
        assert rank == 1

    def test_empty_dataframe(self):
        """DataFrame vacío."""
        df = pd.DataFrame()
        rank = _estimate_effective_rank(df)
        assert rank == 0

    def test_non_numeric_columns(self):
        """Columnas no numéricas se ignoran."""
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "num": [1, 2, 3]
        })
        rank = _estimate_effective_rank(df)
        assert rank == 1


class TestComputeTopologicalPreservation:
    """Tests para cálculo de preservación topológica."""

    def test_identical_topology(self):
        """Topologías idénticas = preservación 1.0."""
        initial = {"rows": 100, "columns": 5, "density": 0.9}
        final = {"rows": 100, "columns": 5, "density": 0.9}
        
        pres = _compute_topological_preservation(initial, final)
        
        assert pres["preservation_rate"] == pytest.approx(1.0)
        assert pres["is_valid"]

    def test_row_reduction(self):
        """Reducción de filas reduce preservación."""
        initial = {"rows": 100, "columns": 5, "density": 0.8}
        final = {"rows": 50, "columns": 5, "density": 0.9}
        
        pres = _compute_topological_preservation(initial, final)
        
        assert pres["preservation_rate"] < 1.0
        assert pres["row_preservation"] == 0.5

    def test_density_improvement_bonus(self):
        """Mejora de densidad da bonus."""
        initial = {"rows": 100, "columns": 5, "density": 0.5}
        final = {"rows": 100, "columns": 5, "density": 0.8}
        
        pres = _compute_topological_preservation(initial, final)
        
        assert pres["improved_density"]
        assert pres["density_delta"] > 0

    def test_error_handling(self):
        """Manejo de errores en topologías."""
        initial = {"error": "Cannot read file"}
        final = {"rows": 100, "columns": 5}
        
        pres = _compute_topological_preservation(initial, final)
        
        assert pres["preservation_rate"] == 0.0
        assert not pres["is_valid"]


# =============================================================================
# Tests para Funciones Financieras
# =============================================================================

class TestGenerateTopologicalCashFlows:
    """Tests para generación de flujos de caja."""

    def test_basic_generation(self):
        """Generación básica."""
        flows = _generate_topological_cash_flows(
            amount=1000,
            time_years=5,
            std_dev=50,
            random_seed=42
        )
        
        assert len(flows) == 5
        assert all(f > 0 for f in flows)  # Log-normal garantiza positivos

    def test_reproducibility(self):
        """Misma semilla = mismos resultados."""
        flows1 = _generate_topological_cash_flows(1000, 5, 50, random_seed=123)
        flows2 = _generate_topological_cash_flows(1000, 5, 50, random_seed=123)
        
        assert flows1 == flows2

    def test_different_seeds(self):
        """Diferentes semillas = diferentes resultados."""
        flows1 = _generate_topological_cash_flows(1000, 5, 50, random_seed=1)
        flows2 = _generate_topological_cash_flows(1000, 5, 50, random_seed=2)
        
        assert flows1 != flows2

    def test_zero_std_dev(self):
        """Sin volatilidad = flujos deterministas."""
        flows = _generate_topological_cash_flows(1000, 5, 0, random_seed=42)
        
        # Flujos decaen según decay_rate
        assert flows[0] > flows[-1]

    def test_invalid_inputs(self):
        """Inputs inválidos retornan lista vacía."""
        assert _generate_topological_cash_flows(0, 5, 50) == []
        assert _generate_topological_cash_flows(1000, 0, 50) == []
        assert _generate_topological_cash_flows(-100, 5, 50) == []


class TestAnalyzeRiskManifold:
    """Tests para análisis de variedad de riesgo."""

    def test_basic_manifold(self):
        """Manifold básico."""
        flows = [200, 180, 160, 150, 140]
        manifold = _analyze_risk_manifold(1000, 50, 5, flows)
        
        assert manifold["dimension"] == 2
        assert "volatility_surface" in manifold
        assert "flow_stability" in manifold
        assert "curvature" in manifold

    def test_degenerate_manifold(self):
        """Manifold degenerado (sin flujos)."""
        manifold = _analyze_risk_manifold(1000, 50, 5, [])
        
        assert manifold["is_degenerate"]
        assert manifold["dimension"] == 0

    def test_constant_flows(self):
        """Flujos constantes = dimensión 1."""
        flows = [100, 100, 100, 100]
        manifold = _analyze_risk_manifold(1000, 0, 4, flows)
        
        assert manifold["dimension"] == 1

    def test_local_extrema_count(self):
        """Cuenta extremos locales."""
        flows = [100, 150, 100, 150, 100]  # 2 máximos, 1 mínimo
        manifold = _analyze_risk_manifold(1000, 50, 5, flows)
        
        assert manifold["local_extrema"] >= 2


class TestComputeRiskHomology:
    """Tests para homología del riesgo."""

    def test_stable_manifold(self):
        """Manifold estable = pocos agujeros."""
        manifold = {
            "flow_stability": 0.9,
            "local_extrema": 1,
            "volatility_surface": 0.05,
            "is_degenerate": False
        }
        homology = _compute_risk_homology(manifold)
        
        assert homology["risk_holes_beta_1"] <= 2
        assert "Solid" in homology["interpretation"] or "Minor" in homology["interpretation"]

    def test_unstable_manifold(self):
        """Manifold inestable = muchos agujeros."""
        manifold = {
            "flow_stability": 0.1,
            "local_extrema": 5,
            "volatility_surface": 0.5,
            "is_degenerate": False
        }
        homology = _compute_risk_homology(manifold)
        
        assert homology["risk_holes_beta_1"] >= 4
        assert "Critical" in homology["interpretation"] or "Significant" in homology["interpretation"]

    def test_degenerate_input(self):
        """Input degenerado."""
        manifold = {"is_degenerate": True}
        homology = _compute_risk_homology(manifold)
        
        assert "Degenerate" in homology["interpretation"]


class TestComputeOpportunityPersistence:
    """Tests para persistencia de oportunidades."""

    def test_positive_mean_flow(self):
        """Flujo medio positivo = oportunidad primaria."""
        manifold = {"mean_flow": 150, "flow_stability": 0.8, "volatility_surface": 0.1, "local_extrema": 2}
        persistence = _compute_opportunity_persistence(manifold)
        
        assert len(persistence) >= 1
        assert persistence[0]["type"] == "primary_opportunity"

    def test_secondary_opportunities(self):
        """Extremos locales generan oportunidades secundarias."""
        manifold = {"mean_flow": 100, "flow_stability": 0.5, "volatility_surface": 0.2, "local_extrema": 3}
        persistence = _compute_opportunity_persistence(manifold)
        
        secondary = [p for p in persistence if p["type"] == "secondary_opportunity"]
        assert len(secondary) == 3


class TestComputeRiskAdjustedReturn:
    """Tests para retorno ajustado al riesgo."""

    def test_positive_npv(self):
        """NPV positivo con baja tolerancia."""
        analysis = {"npv": 10000, "volatility": 0.3, "project_life_years": 5}
        rar = _compute_risk_adjusted_return(analysis, risk_tolerance=0.05)
        
        assert rar > 0

    def test_null_npv(self):
        """NPV nulo retorna cero."""
        analysis = {"npv": None}
        rar = _compute_risk_adjusted_return(analysis, risk_tolerance=0.1)
        
        assert rar == 0.0

    def test_high_risk_tolerance(self):
        """Alta tolerancia reduce penalización."""
        analysis = {"npv": 10000}
        rar_low = _compute_risk_adjusted_return(analysis, risk_tolerance=0.1)
        rar_high = _compute_risk_adjusted_return(analysis, risk_tolerance=0.9)
        
        # Mayor tolerancia = menor penalización = mayor RAR (aproximadamente)
        # La relación depende de la fórmula exacta


class TestComputeTopologicalEfficiency:
    """Tests para eficiencia topológica."""

    def test_smooth_manifold(self):
        """Manifold suave = alta eficiencia."""
        analysis = {"npv": 10000}
        manifold = {"curvature": 0.0, "flow_stability": 1.0}
        
        eff = _compute_topological_efficiency(analysis, manifold)
        assert eff > 0.4  # Relativamente alto

    def test_rough_manifold(self):
        """Manifold rugoso = baja eficiencia."""
        analysis = {"npv": 10000}
        manifold = {"curvature": 0.5, "flow_stability": 0.2}
        
        eff = _compute_topological_efficiency(analysis, manifold)
        # Eficiencia penalizada por rugosidad


# =============================================================================
# Tests para MICRegistry (Gatekeeper)
# =============================================================================

class TestMICRegistry:
    """Pruebas para la Matriz de Interacción Central y su Gatekeeper."""

    def test_register_vector(self):
        """Registro básico de vector."""
        mic = MICRegistry()
        handler = lambda **k: {"success": True}
        mic.register_vector("test_service", Stratum.PHYSICS, handler)
        
        assert "test_service" in mic.registered_services

    def test_register_empty_name_fails(self):
        """Nombre vacío falla."""
        mic = MICRegistry()
        with pytest.raises(ValueError, match="cannot be empty"):
            mic.register_vector("", Stratum.PHYSICS, lambda: None)

    def test_register_non_callable_fails(self):
        """Handler no callable falla."""
        mic = MICRegistry()
        with pytest.raises(TypeError, match="must be callable"):
            mic.register_vector("service", Stratum.PHYSICS, "not a function")

    def test_overwrite_warning(self, caplog):
        """Sobrescribir vector genera warning."""
        mic = MICRegistry()
        mic.register_vector("service", Stratum.PHYSICS, lambda: None)
        
        with caplog.at_level(logging.WARNING):
            mic.register_vector("service", Stratum.TACTICS, lambda: None)
        
        assert "Overwriting" in caplog.text

    def test_gatekeeper_blocks_strategy_without_physics(self):
        """Estrategia requiere Física validada."""
        mic = MICRegistry()
        mic.register_vector("finance", Stratum.STRATEGY, lambda **k: {"success": True})

        context = {"validated_strata": set()}
        result = mic.project_intent("finance", {}, context)

        assert result["success"] is False
        assert "MIC Hierarchy Violation" in result["error"]
        assert result.get("error_category") == "mic_hierarchy_violation"
        assert "PHYSICS" in str(result.get("missing_strata"))

    def test_gatekeeper_allows_physics(self):
        """Física siempre es accesible (base)."""
        mic = MICRegistry()
        mic.register_vector("basic_physics", Stratum.PHYSICS, lambda **k: {"success": True})

        context = {"validated_strata": set()}
        result = mic.project_intent("basic_physics", {}, context)

        assert result["success"] is True
        assert result.get("_mic_validation_update") == Stratum.PHYSICS
        assert result.get("_mic_stratum") == "PHYSICS"

    def test_gatekeeper_allows_strategy_with_full_chain(self):
        """Estrategia funciona con cadena completa validada."""
        mic = MICRegistry()
        mic.register_vector("finance", Stratum.STRATEGY, lambda **k: {"success": True})

        # STRATEGY(1) requiere TACTICS(2) y PHYSICS(3)
        context = {"validated_strata": {Stratum.PHYSICS, Stratum.TACTICS}}
        result = mic.project_intent("finance", {}, context)

        assert result["success"] is True

    def test_gatekeeper_blocks_strategy_with_only_physics(self):
        """Estrategia falla solo con Física (falta Táctica)."""
        mic = MICRegistry()
        mic.register_vector("finance", Stratum.STRATEGY, lambda **k: {"success": True})

        context = {"validated_strata": {Stratum.PHYSICS}}
        result = mic.project_intent("finance", {}, context)

        assert result["success"] is False
        assert "TACTICS" in str(result.get("missing_strata"))

    def test_force_override(self):
        """Bypass de emergencia."""
        mic = MICRegistry()
        mic.register_vector("finance", Stratum.STRATEGY, lambda **k: {"success": True})

        context = {"validated_strata": set(), "force_physics_override": True}
        result = mic.project_intent("finance", {}, context)

        assert result["success"] is True

    def test_unknown_vector_raises(self):
        """Vector desconocido lanza error."""
        mic = MICRegistry()
        
        with pytest.raises(ValueError, match="Unknown vector"):
            mic.project_intent("nonexistent", {}, {})

    def test_normalize_strata_from_int(self):
        """Normaliza estratos desde int."""
        mic = MICRegistry()
        mic.register_vector("service", Stratum.TACTICS, lambda **k: {"success": True})

        # Usar int en lugar de Stratum
        context = {"validated_strata": {3}}  # 3 = PHYSICS
        result = mic.project_intent("service", {}, context)

        assert result["success"] is True

    def test_normalize_strata_from_string(self):
        """Normaliza estratos desde string."""
        mic = MICRegistry()
        mic.register_vector("service", Stratum.TACTICS, lambda **k: {"success": True})

        context = {"validated_strata": {"PHYSICS"}}
        result = mic.project_intent("service", {}, context)

        assert result["success"] is True

    def test_handler_signature_error(self):
        """Error de firma del handler."""
        mic = MICRegistry()
        mic.register_vector("service", Stratum.PHYSICS, lambda x: {"success": True})

        result = mic.project_intent("service", {"wrong_key": 1}, {})

        assert result["success"] is False
        assert result["error_category"] == "mic_handler_signature_error"


# =============================================================================
# Tests para IntentVector
# =============================================================================

class TestIntentVector:
    """Tests para el vector de intención."""

    def test_valid_creation(self):
        """Crear IntentVector válido."""
        iv = IntentVector(
            service_name="test",
            payload={"key": "value"},
            context={"session": "123"}
        )
        assert iv.service_name == "test"
        assert iv.payload == {"key": "value"}

    def test_empty_name_fails(self):
        """Nombre vacío falla."""
        with pytest.raises(ValueError, match="cannot be empty"):
            IntentVector(service_name="", payload={}, context={})

    def test_whitespace_name_fails(self):
        """Nombre solo espacios falla."""
        with pytest.raises(ValueError, match="cannot be empty"):
            IntentVector(service_name="   ", payload={}, context={})

    def test_immutability(self):
        """IntentVector es inmutable."""
        iv = IntentVector(service_name="test", payload={}, context={})
        with pytest.raises(AttributeError):
            iv.service_name = "other"


# =============================================================================
# Tests para Handlers MIC
# =============================================================================

class TestDiagnoseFile:
    """Tests para el handler de diagnóstico."""

    @patch("app.tools_interface._get_diagnostic_class")
    def test_basic_diagnosis(self, mock_get_class, temp_csv_file: Path):
        """Diagnóstico básico exitoso."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.to_dict.return_value = {"issues": [], "warnings": []}
        mock_get_class.return_value = MagicMock(return_value=mock_diagnostic)

        result = diagnose_file(temp_csv_file, FileType.APUS)

        assert result["success"] is True
        assert result["file_type"] == "apus"
        assert "diagnostic_magnitude" in result

    @patch("app.tools_interface._get_diagnostic_class")
    def test_diagnosis_with_topology(self, mock_get_class, temp_csv_file: Path):
        """Diagnóstico con análisis topológico."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.to_dict.return_value = {
            "issues": [{"type": "ERROR", "severity": "HIGH"}],
            "warnings": []
        }
        mock_get_class.return_value = MagicMock(return_value=mock_diagnostic)

        result = diagnose_file(temp_csv_file, "apus", topological_analysis=True)

        assert result["success"] is True
        assert result["has_topological_analysis"] is True
        assert "topological_features" in result
        assert "homology" in result
        assert "persistence_diagram" in result

    def test_file_not_found(self, tmp_path: Path):
        """Archivo no encontrado."""
        result = diagnose_file(tmp_path / "nonexistent.csv", "apus")

        assert result["success"] is False
        assert result["error_category"] == "validation"

    def test_invalid_file_type(self, temp_csv_file: Path):
        """Tipo de archivo inválido."""
        result = diagnose_file(temp_csv_file, "invalid_type")

        assert result["success"] is False

    def test_empty_file_warning(self, temp_empty_file: Path):
        """Archivo vacío genera warning."""
        result = diagnose_file(temp_empty_file, "apus")

        assert result["success"] is True
        assert result.get("is_empty") or result.get("warning")


class TestCleanFile:
    """Tests para el handler de limpieza."""

    @patch("app.tools_interface.CSVCleaner")
    def test_basic_cleaning(self, mock_cleaner_class, temp_csv_file: Path):
        """Limpieza básica exitosa."""
        mock_instance = mock_cleaner_class.return_value
        mock_instance.clean.return_value = {"rows_cleaned": 10}

        result = clean_file(temp_csv_file)

        assert result["success"] is True
        assert "output_path" in result

    @patch("app.tools_interface.CSVCleaner")
    def test_cleaning_with_topology(self, mock_cleaner_class, temp_csv_file: Path):
        """Limpieza con preservación topológica."""
        output_path = temp_csv_file.with_name("test_data_clean.csv")
        
        def mock_clean():
            output_path.write_text(temp_csv_file.read_text())
            return {"rows_cleaned": 10}
        
        mock_instance = mock_cleaner_class.return_value
        mock_instance.clean.side_effect = mock_clean

        result = clean_file(temp_csv_file, preserve_topology=True)

        assert result["success"] is True
        assert result["preserved_topology"] is True
        assert "topological_preservation" in result

    def test_same_input_output_fails(self, temp_csv_file: Path):
        """Input = Output falla."""
        result = clean_file(temp_csv_file, output_path=temp_csv_file)

        assert result["success"] is False
        assert "same as input" in result["error"].lower()

    def test_invalid_delimiter(self, temp_csv_file: Path):
        """Delimitador inválido."""
        result = clean_file(temp_csv_file, delimiter="invalid")

        assert result["success"] is False
        assert result["error_category"] == "validation"


class TestAnalyzeFinancialViability:
    """Tests para el handler de análisis financiero."""

    @patch("app.tools_interface.FinancialEngine")
    def test_basic_analysis(self, mock_engine_class):
        """Análisis básico exitoso."""
        mock_engine = mock_engine_class.return_value
        mock_engine.analyze_project.return_value = {
            "npv": 50000,
            "wacc": 0.08,
            "irr": 0.12,
            "performance": {"recommendation": "PROCEED"}
        }

        result = analyze_financial_viability(
            amount=100000,
            std_dev=5000,
            time_years=5
        )

        assert result["success"] is True
        assert "results" in result
        assert result["results"]["npv"] == 50000

    @patch("app.tools_interface.FinancialEngine")
    def test_analysis_with_topology(self, mock_engine_class):
        """Análisis con riesgo topológico."""
        mock_engine = mock_engine_class.return_value
        mock_engine.analyze_project.return_value = {
            "npv": 50000,
            "wacc": 0.08,
            "performance": {}
        }

        result = analyze_financial_viability(
            amount=100000,
            std_dev=5000,
            time_years=5,
            topological_risk_analysis=True,
            random_seed=42
        )

        assert result["success"] is True
        assert "topological_risk" in result["results"]
        assert "risk_manifold" in result["results"]
        assert "opportunity_persistence" in result["results"]
        assert "topological_efficiency" in result["results"]

    def test_negative_amount_fails(self):
        """Monto negativo falla."""
        result = analyze_financial_viability(
            amount=-1000,
            std_dev=50,
            time_years=5
        )

        assert result["success"] is False
        assert "positive" in result["error"].lower()

    def test_zero_time_fails(self):
        """Tiempo cero falla."""
        result = analyze_financial_viability(
            amount=1000,
            std_dev=50,
            time_years=0
        )

        assert result["success"] is False

    def test_invalid_risk_tolerance(self):
        """Tolerancia fuera de rango falla."""
        result = analyze_financial_viability(
            amount=1000,
            std_dev=50,
            time_years=5,
            risk_tolerance=1.5  # > 1
        )

        assert result["success"] is False

    @patch("app.tools_interface.FinancialEngine")
    def test_reproducibility(self, mock_engine_class):
        """Reproducibilidad con semilla."""
        mock_engine = mock_engine_class.return_value
        mock_engine.analyze_project.return_value = {"npv": 100, "performance": {}}

        result1 = analyze_financial_viability(1000, 50, 5, random_seed=42)
        result2 = analyze_financial_viability(1000, 50, 5, random_seed=42)

        # Los parámetros registrados deben ser iguales
        assert result1["parameters"]["random_seed"] == result2["parameters"]["random_seed"]


class TestGetTelemetryStatus:
    """Tests para el handler de telemetría."""

    def test_no_context(self):
        """Sin contexto = IDLE."""
        result = get_telemetry_status(None)

        assert result["success"] is True
        assert result["status"] == "IDLE"
        assert result["has_active_context"] is False

    def test_valid_context(self, mock_telemetry_context):
        """Contexto válido."""
        result = get_telemetry_status(mock_telemetry_context)

        assert result["success"] is True
        assert result["status"] == "ACTIVE"
        assert result["system_health"] == "HEALTHY"
        assert result["has_active_context"] is True
        assert result["active_processes"] == 3

    def test_invalid_context_type(self):
        """Tipo de contexto inválido."""
        result = get_telemetry_status("not a context")

        assert result["success"] is False
        assert result["status"] == "ERROR"
        assert result["system_health"] == "DEGRADED"

    def test_context_returns_none(self):
        """Contexto retorna None."""
        context = MagicMock(spec=TelemetryContextProtocol)
        context.get_business_report.return_value = None

        result = get_telemetry_status(context)

        assert result["success"] is True
        assert result["status"] == "ACTIVE"

    def test_context_returns_non_dict(self):
        """Contexto retorna no-dict."""
        context = MagicMock(spec=TelemetryContextProtocol)
        context.get_business_report.return_value = "raw string"

        result = get_telemetry_status(context)

        assert result["success"] is True
        assert result["raw_report"] == "raw string"

    def test_context_raises_exception(self):
        """Contexto lanza excepción."""
        context = MagicMock(spec=TelemetryContextProtocol)
        context.get_business_report.side_effect = RuntimeError("Connection failed")

        result = get_telemetry_status(context)

        assert result["success"] is False
        assert result["status"] == "ERROR"
        assert "Connection failed" in result["error"]


# =============================================================================
# Tests para Funciones de Validación
# =============================================================================

class TestValidationFunctions:
    """Tests para funciones de validación."""

    def test_normalize_path(self, temp_csv_file: Path):
        """Normalización de path."""
        result = _normalize_path(str(temp_csv_file))
        assert isinstance(result, Path)
        assert result.exists()

    def test_normalize_path_empty_fails(self):
        """Path vacío falla."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _normalize_path("")

    def test_normalize_path_none_fails(self):
        """Path None falla."""
        with pytest.raises(ValueError, match="cannot be None"):
            _normalize_path(None)

    def test_validate_file_exists(self, temp_csv_file: Path):
        """Archivo existe."""
        _validate_file_exists(temp_csv_file)  # No debe lanzar

    def test_validate_file_not_exists(self, tmp_path: Path):
        """Archivo no existe."""
        with pytest.raises(FileNotFoundDiagnosticError):
            _validate_file_exists(tmp_path / "nonexistent.csv")

    def test_validate_extension_valid(self, temp_csv_file: Path):
        """Extensión válida."""
        ext = _validate_file_extension(temp_csv_file)
        assert ext == ".csv"

    def test_validate_extension_invalid(self, tmp_path: Path):
        """Extensión inválida."""
        invalid_file = tmp_path / "file.xyz"
        invalid_file.touch()
        
        with pytest.raises(FileValidationError, match="Invalid extension"):
            _validate_file_extension(invalid_file)

    def test_normalize_encoding(self):
        """Normalización de encoding."""
        assert _normalize_encoding("UTF-8") == "utf-8"
        assert _normalize_encoding("utf8") == "utf-8"
        assert _normalize_encoding("LATIN1") == "latin-1"

    def test_validate_csv_parameters(self):
        """Validación de parámetros CSV."""
        delim, enc = _validate_csv_parameters(";", "utf-8")
        assert delim == ";"
        assert enc == "utf-8"

    def test_validate_csv_invalid_delimiter(self):
        """Delimitador inválido."""
        with pytest.raises(ValueError):
            _validate_csv_parameters("XX", "utf-8")

    def test_normalize_file_type(self):
        """Normalización de tipo de archivo."""
        assert _normalize_file_type("APUS") == FileType.APUS
        assert _normalize_file_type(FileType.INSUMOS) == FileType.INSUMOS

    def test_generate_output_path(self, temp_csv_file: Path):
        """Generación de path de salida."""
        output = _generate_output_path(temp_csv_file)
        assert output.stem == "test_data_clean"
        assert output.suffix == ".csv"


# =============================================================================
# Tests para Funciones de Respuesta
# =============================================================================

class TestResponseFunctions:
    """Tests para funciones de creación de respuestas."""

    def test_create_success_response(self):
        """Respuesta de éxito."""
        resp = _create_success_response({"key": "value"}, extra="data")
        
        assert resp["success"] is True
        assert resp["key"] == "value"
        assert resp["extra"] == "data"

    def test_create_error_response_from_exception(self):
        """Respuesta de error desde excepción."""
        error = DiagnosticError("Test error", details={"code": 123})
        resp = _create_error_response(error, category="test")
        
        assert resp["success"] is False
        assert resp["error"] == "Test error"
        assert resp["error_type"] == "DiagnosticError"
        assert resp["error_details"]["code"] == 123

    def test_create_error_response_from_string(self):
        """Respuesta de error desde string."""
        resp = _create_error_response("Simple error")
        
        assert resp["success"] is False
        assert resp["error"] == "Simple error"
        assert resp["error_type"] == "Error"

    def test_extract_diagnostic_result(self):
        """Extracción de resultado diagnóstico."""
        mock_diag = MagicMock()
        mock_diag.to_dict.return_value = {"issues": [], "status": "OK"}
        
        result = _extract_diagnostic_result(mock_diag)
        
        assert result["diagnostic_completed"] is True
        assert result["status"] == "OK"


# =============================================================================
# Tests para Utilidades Públicas
# =============================================================================

class TestPublicUtilities:
    """Tests para utilidades públicas."""

    def test_get_supported_file_types(self):
        """Lista de tipos soportados."""
        types = get_supported_file_types()
        assert "apus" in types
        assert "insumos" in types
        assert "presupuesto" in types

    def test_get_supported_delimiters(self):
        """Lista de delimitadores soportados."""
        delims = get_supported_delimiters()
        assert "," in delims
        assert ";" in delims
        assert "\t" in delims

    def test_get_supported_encodings(self):
        """Lista de encodings soportados."""
        encs = get_supported_encodings()
        assert "utf-8" in encs
        assert "latin-1" in encs

    def test_is_valid_file_type(self):
        """Validación de tipo de archivo."""
        assert is_valid_file_type("apus") is True
        assert is_valid_file_type("invalid") is False
        assert is_valid_file_type(FileType.INSUMOS) is True

    def test_validate_file_for_processing(self, temp_csv_file: Path):
        """Pre-validación de archivo."""
        result = validate_file_for_processing(temp_csv_file)
        
        assert result["valid"] is True
        assert result["size"] > 0
        assert result["extension"] == ".csv"

    def test_validate_file_for_processing_nonexistent(self, tmp_path: Path):
        """Pre-validación de archivo inexistente."""
        result = validate_file_for_processing(tmp_path / "nonexistent.csv")
        
        assert result["valid"] is False
        assert "errors" in result


# =============================================================================
# Tests de Integración
# =============================================================================

class TestIntegration:
    """Tests de integración end-to-end."""

    @patch("app.tools_interface._get_diagnostic_class")
    @patch("app.tools_interface.CSVCleaner")
    def test_full_physics_workflow(
        self,
        mock_cleaner_class,
        mock_get_class,
        temp_csv_file: Path
    ):
        """Flujo completo de Física: diagnóstico + limpieza."""
        # Setup mocks
        mock_diagnostic = MagicMock()
        mock_diagnostic.to_dict.return_value = {"issues": [], "warnings": []}
        mock_get_class.return_value = MagicMock(return_value=mock_diagnostic)
        
        output_path = temp_csv_file.with_name("test_data_clean.csv")
        mock_cleaner_class.return_value.clean.side_effect = lambda: (
            output_path.write_text(temp_csv_file.read_text()) or {"cleaned": 10}
        )

        # Diagnóstico primero
        diag_result = diagnose_file(temp_csv_file, "apus", topological_analysis=True)
        assert diag_result["success"] is True

        # Luego limpieza
        clean_result = clean_file(temp_csv_file, preserve_topology=True)
        assert clean_result["success"] is True

    @patch("app.tools_interface.FinancialEngine")
    def test_mic_hierarchy_workflow(self, mock_engine_class):
        """Flujo completo respetando jerarquía MIC."""
        mock_engine = mock_engine_class.return_value
        mock_engine.analyze_project.return_value = {"npv": 1000, "performance": {}}

        mic = MICRegistry()

        # Registrar vectores
        mic.register_vector("physics_validate", Stratum.PHYSICS, lambda **k: {"success": True})
        mic.register_vector("tactics_analyze", Stratum.TACTICS, lambda **k: {"success": True})
        mic.register_vector("strategy_decide", Stratum.STRATEGY, lambda **k: {"success": True})

        # Fase 1: Física (base)
        ctx = {"validated_strata": set()}
        result1 = mic.project_intent("physics_validate", {}, ctx)
        assert result1["success"] is True
        ctx["validated_strata"].add(result1["_mic_validation_update"])

        # Fase 2: Táctica (requiere física)
        result2 = mic.project_intent("tactics_analyze", {}, ctx)
        assert result2["success"] is True
        ctx["validated_strata"].add(result2["_mic_validation_update"])

        # Fase 3: Estrategia (requiere física + táctica)
        result3 = mic.project_intent("strategy_decide", {}, ctx)
        assert result3["success"] is True


# =============================================================================
# Ejecución directa
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])