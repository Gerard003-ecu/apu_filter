"""
Suite de Pruebas — Matriz de Interacción Central (MIC) y Tools Interface.

Cobertura de pruebas:
─────────────────────
├── Configuración: MICConfiguration y validaciones
├── Estructuras Topológicas: PersistenceInterval, BettiNumbers, TopologicalSummary
├── IntentVector: inmutabilidad y propiedades
├── Cache: TTLCache con TTL, evicción, estadísticas
├── Métricas: LatencyHistogram, MICMetrics
├── Análisis Espectral: SpectralGraphMetrics
├── Transiciones: StratumTransitionMatrix
├── MICRegistry: registro, proyección, Command Pattern
├── Excepciones: jerarquía y serialización
├── Validación de Archivos: permisos, extensiones, tamaños
├── Funciones Matemáticas: entropía, persistencia, homología
├── Singleton: get_global_mic, reset_global_mic
├── Diagnóstico: diagnose_file con análisis topológico
├── Propiedades: invariantes matemáticos (Hypothesis)
├── Edge Cases: casos límite y degeneraciones
└── Rendimiento: benchmarks básicos

Ejecución:
    pytest test_tools_interface.py -v --cov=tools_interface --cov-report=html
    pytest test_tools_interface.py -v -m "not slow"  # Excluir lentas
    pytest test_tools_interface.py -v -k "cache"     # Solo cache
"""

import gc
import hashlib
import math
import os
import stat
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# ═══════════════════════════════════════════════════════════════════════════
# Importación del módulo bajo prueba
# ═══════════════════════════════════════════════════════════════════════════

from app.adapters.tools_interface import (
    # Configuración
    MICConfiguration,
    DEFAULT_MIC_CONFIG,
    
    # Tipos y Enums
    FileType,
    Stratum,
    ProjectionResult,
    DiagnosticResult,
    CacheStats,
    LatencyStats,
    
    # Estructuras topológicas
    PersistenceInterval,
    BettiNumbers,
    TopologicalSummary,
    IntentVector,
    
    # Excepciones
    MICException,
    FileNotFoundDiagnosticError,
    UnsupportedFileTypeError,
    FileValidationError,
    FilePermissionError as MICFilePermissionError,
    CleaningError,
    MICHierarchyViolationError,
    TimeoutError as MICTimeoutError,
    
    # Cache y métricas
    TTLCache,
    CacheEntry,
    LatencyHistogram,
    MICMetrics,
    
    # Análisis
    SpectralGraphMetrics,
    StratumTransitionMatrix,
    
    # Core
    MICRegistry,
    
    # Commands
    ProjectionContext,
    ProjectionCommand,
    CacheCheckCommand,
    ResolutionCommand,
    NormalizationCommand,
    ValidationCommand,
    ExecutionCommand,
    
    # Funciones
    diagnose_file,
    get_global_mic,
    reset_global_mic,
    register_core_vectors,
    get_supported_file_types,
    get_supported_delimiters,
    get_supported_encodings,
    validate_file_for_processing,
    
    # Funciones matemáticas
    compute_shannon_entropy,
    compute_persistence_entropy,
    distribution_from_counts,
    analyze_topological_features,
    compute_homology_from_diagnostic,
    compute_persistence_diagram,
    compute_diagnostic_magnitude,
    detect_cyclic_patterns,
    estimate_intrinsic_dimension,
    
    # Validación
    normalize_path,
    validate_file_exists,
    validate_file_permissions,
    validate_file_extension,
    validate_file_size,
    normalize_encoding,
    normalize_file_type,
    
    # Constantes
    SUPPORTED_ENCODINGS,
    VALID_DELIMITERS,
    VALID_EXTENSIONS,
    _SEVERITY_WEIGHTS,
    
    # Helpers
    _jaccard_similarity,
    _tokenize_line,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES — Datos de prueba reutilizables
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_config() -> MICConfiguration:
    """Configuración por defecto."""
    return DEFAULT_MIC_CONFIG


@pytest.fixture
def strict_config() -> MICConfiguration:
    """Configuración estricta."""
    return MICConfiguration(
        max_file_size_bytes=10 * 1024 * 1024,  # 10 MB
        cache_ttl_seconds=60.0,
        persistence_threshold=0.05,
        cycle_similarity_threshold=0.90,
        strict_encoding_validation=True,
    )


@pytest.fixture
def relaxed_config() -> MICConfiguration:
    """Configuración relajada."""
    return MICConfiguration(
        max_file_size_bytes=500 * 1024 * 1024,  # 500 MB
        cache_ttl_seconds=600.0,
        persistence_threshold=0.001,
        cycle_similarity_threshold=0.50,
    )


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Archivo CSV temporal válido."""
    file_path = tmp_path / "test_data.csv"
    content = """CODIGO_APU,DESCRIPCION,VALOR
APU001,Excavación manual,1500.00
APU002,Concreto 3000 psi,2500.00
APU003,Acero de refuerzo,3200.00
APU004,Mampostería,1800.00
APU005,Pintura interior,900.00
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def temp_empty_file(tmp_path: Path) -> Path:
    """Archivo vacío temporal."""
    file_path = tmp_path / "empty.csv"
    file_path.touch()
    return file_path


@pytest.fixture
def temp_large_csv(tmp_path: Path) -> Path:
    """Archivo CSV grande para pruebas de rendimiento."""
    file_path = tmp_path / "large_data.csv"
    lines = ["COL1,COL2,COL3,COL4,COL5"]
    for i in range(1000):
        lines.append(f"VAL{i},DATA{i},{i*10},{i*100},{i*1000}")
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


@pytest.fixture
def temp_cyclic_csv(tmp_path: Path) -> Path:
    """Archivo CSV con patrones cíclicos."""
    file_path = tmp_path / "cyclic_data.csv"
    pattern = ["A,B,C", "D,E,F", "G,H,I"]
    lines = ["COL1,COL2,COL3"]
    for _ in range(50):
        lines.extend(pattern)
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


@pytest.fixture
def mic_registry() -> MICRegistry:
    """Instancia limpia de MICRegistry."""
    return MICRegistry()


@pytest.fixture
def mic_with_vectors() -> MICRegistry:
    """MICRegistry con vectores registrados."""
    mic = MICRegistry()
    
    # Registrar vectores de prueba
    mic.register_vector(
        "test_physics",
        Stratum.PHYSICS,
        lambda **kw: {"success": True, "data": "physics", **kw}
    )
    mic.register_vector(
        "test_tactics",
        Stratum.TACTICS,
        lambda **kw: {"success": True, "data": "tactics", **kw}
    )
    mic.register_vector(
        "test_strategy",
        Stratum.STRATEGY,
        lambda **kw: {"success": True, "data": "strategy", **kw}
    )
    mic.register_vector(
        "test_wisdom",
        Stratum.WISDOM,
        lambda **kw: {"success": True, "data": "wisdom", **kw}
    )
    
    return mic


@pytest.fixture
def sample_diagnostic_data() -> Dict[str, Any]:
    """Datos diagnósticos de ejemplo."""
    return {
        "issues": [
            {"type": "MISSING_VALUE", "severity": "HIGH", "column": "VALOR"},
            {"type": "DUPLICATE_ROW", "severity": "MEDIUM", "row": 5},
            {"type": "FORMAT_ERROR", "severity": "LOW", "column": "FECHA"},
        ],
        "warnings": [
            "Algunas filas tienen valores vacíos",
            "circular dependency detected in column references",
        ],
        "errors": [],
    }


@pytest.fixture
def sample_diagnostic_critical() -> Dict[str, Any]:
    """Datos diagnósticos con errores críticos."""
    return {
        "issues": [
            {"type": "SCHEMA_VIOLATION", "severity": "CRITICAL"},
            {"type": "DATA_CORRUPTION", "severity": "CRITICAL"},
        ],
        "warnings": [],
        "errors": ["File encoding error", "Malformed CSV structure"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: MICConfiguration — Validación de configuración
# ═══════════════════════════════════════════════════════════════════════════

class TestMICConfiguration:
    """Pruebas de la configuración de la MIC."""
    
    def test_default_config_is_valid(self, default_config: MICConfiguration):
        """La configuración por defecto debe ser válida."""
        assert default_config.max_file_size_bytes == 100 * 1024 * 1024
        assert default_config.cache_ttl_seconds == 300.0
        assert default_config.cache_max_size == 128
        assert default_config.algorithm_version == "4.0.0-topological"
    
    def test_config_immutability(self, default_config: MICConfiguration):
        """La configuración debe ser inmutable."""
        with pytest.raises(AttributeError):
            default_config.max_file_size_bytes = 999
    
    def test_invalid_max_file_size_raises(self):
        """max_file_size_bytes <= 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="max_file_size_bytes debe ser > 0"):
            MICConfiguration(max_file_size_bytes=0)
        
        with pytest.raises(ValueError, match="max_file_size_bytes debe ser > 0"):
            MICConfiguration(max_file_size_bytes=-100)
    
    def test_invalid_cache_ttl_raises(self):
        """cache_ttl_seconds <= 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="cache_ttl_seconds debe ser > 0"):
            MICConfiguration(cache_ttl_seconds=0)
    
    def test_invalid_cycle_similarity_raises(self):
        """cycle_similarity_threshold fuera de (0, 1] debe lanzar ValueError."""
        with pytest.raises(ValueError, match="cycle_similarity_threshold"):
            MICConfiguration(cycle_similarity_threshold=0)
        
        with pytest.raises(ValueError, match="cycle_similarity_threshold"):
            MICConfiguration(cycle_similarity_threshold=1.5)
    
    def test_invalid_persistence_threshold_raises(self):
        """persistence_threshold fuera de (0, 1) debe lanzar ValueError."""
        with pytest.raises(ValueError, match="persistence_threshold"):
            MICConfiguration(persistence_threshold=0)
        
        with pytest.raises(ValueError, match="persistence_threshold"):
            MICConfiguration(persistence_threshold=1.0)
    
    def test_custom_config_values(self):
        """Configuración personalizada acepta valores válidos."""
        config = MICConfiguration(
            max_file_size_bytes=50 * 1024 * 1024,
            cache_ttl_seconds=120.0,
            cache_max_size=256,
            persistence_threshold=0.05,
            cycle_similarity_threshold=0.75,
        )
        assert config.max_file_size_bytes == 50 * 1024 * 1024
        assert config.cache_max_size == 256


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Stratum — Jerarquía DIKW
# ═══════════════════════════════════════════════════════════════════════════

class TestStratum:
    """Pruebas del enum Stratum."""
    
    def test_stratum_values(self):
        """Los valores numéricos deben ser correctos."""
        assert Stratum.WISDOM.value == 0
        assert Stratum.OMEGA.value == 1
        assert Stratum.STRATEGY.value == 2
        assert Stratum.TACTICS.value == 3
        assert Stratum.PHYSICS.value == 4
    
    def test_base_stratum(self):
        """base_stratum debe retornar PHYSICS."""
        assert Stratum.base_stratum() == Stratum.PHYSICS
    
    def test_apex_stratum(self):
        """apex_stratum debe retornar WISDOM."""
        assert Stratum.apex_stratum() == Stratum.WISDOM
    
    def test_requires_physics(self):
        """PHYSICS no requiere ningún prerrequisito."""
        assert Stratum.PHYSICS.requires() == frozenset()
    
    def test_requires_tactics(self):
        """TACTICS requiere PHYSICS."""
        required = Stratum.TACTICS.requires()
        assert Stratum.PHYSICS in required
        assert len(required) == 1
    
    def test_requires_strategy(self):
        """STRATEGY requiere PHYSICS y TACTICS."""
        required = Stratum.STRATEGY.requires()
        assert Stratum.PHYSICS in required
        assert Stratum.TACTICS in required
        assert len(required) == 2
    
    def test_requires_omega(self):
        """OMEGA requiere PHYSICS, TACTICS y STRATEGY."""
        required = Stratum.OMEGA.requires()
        assert Stratum.PHYSICS in required
        assert Stratum.TACTICS in required
        assert Stratum.STRATEGY in required
        assert len(required) == 3

    def test_requires_wisdom(self):
        """WISDOM requiere todos los demás estratos."""
        required = Stratum.WISDOM.requires()
        assert Stratum.PHYSICS in required
        assert Stratum.TACTICS in required
        assert Stratum.STRATEGY in required
        assert Stratum.OMEGA in required
        assert len(required) == 4
    
    def test_ordered_bottom_up(self):
        """Orden de base a cúspide."""
        order = Stratum.ordered_bottom_up()
        assert order == [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.OMEGA, Stratum.WISDOM]
    
    def test_ordered_top_down(self):
        """Orden de cúspide a base."""
        order = Stratum.ordered_top_down()
        assert order == [Stratum.WISDOM, Stratum.OMEGA, Stratum.STRATEGY, Stratum.TACTICS, Stratum.PHYSICS]


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: FileType — Tipos de archivo
# ═══════════════════════════════════════════════════════════════════════════

class TestFileType:
    """Pruebas del enum FileType."""
    
    def test_file_type_values(self):
        """Los valores deben ser strings correctos."""
        assert FileType.APUS.value == "apus"
        assert FileType.INSUMOS.value == "insumos"
        assert FileType.PRESUPUESTO.value == "presupuesto"
    
    def test_values_method(self):
        """values() retorna lista de valores."""
        values = FileType.values()
        assert "apus" in values
        assert "insumos" in values
        assert "presupuesto" in values
        assert len(values) == 3
    
    def test_from_string_valid(self):
        """from_string parsea strings válidos."""
        assert FileType.from_string("apus") == FileType.APUS
        assert FileType.from_string("APUS") == FileType.APUS
        assert FileType.from_string("  Apus  ") == FileType.APUS
        assert FileType.from_string("insumos") == FileType.INSUMOS
        assert FileType.from_string("presupuesto") == FileType.PRESUPUESTO
    
    def test_from_string_invalid_raises_value_error(self):
        """from_string con valor inválido lanza ValueError."""
        with pytest.raises(ValueError, match="no es válido"):
            FileType.from_string("invalid_type")
    
    def test_from_string_non_string_raises_type_error(self):
        """from_string con no-string lanza TypeError."""
        with pytest.raises(TypeError, match="Se esperaba str"):
            FileType.from_string(123)
        
        with pytest.raises(TypeError):
            FileType.from_string(None)
    
    def test_file_type_is_str(self):
        """FileType hereda de str para serialización JSON."""
        assert isinstance(FileType.APUS, str)
        assert FileType.APUS == "apus"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: PersistenceInterval — Intervalos de persistencia
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistenceInterval:
    """Pruebas de intervalos de persistencia topológica."""
    
    def test_valid_interval_creation(self):
        """Crear intervalo válido."""
        iv = PersistenceInterval(birth=0.0, death=1.0, dimension=0)
        assert iv.birth == 0.0
        assert iv.death == 1.0
        assert iv.dimension == 0
    
    def test_persistence_calculation(self):
        """persistence = death - birth."""
        iv = PersistenceInterval(birth=0.5, death=1.5)
        assert iv.persistence == 1.0
    
    def test_essential_interval(self):
        """Intervalos esenciales tienen death = inf."""
        iv = PersistenceInterval.essential(birth=0.0, dimension=1)
        assert iv.is_essential is True
        assert math.isinf(iv.death)
        assert math.isinf(iv.persistence)
    
    def test_midpoint_finite(self):
        """Midpoint de intervalo finito."""
        iv = PersistenceInterval(birth=0.0, death=2.0)
        assert iv.midpoint == 1.0
    
    def test_midpoint_essential(self):
        """Midpoint de intervalo esencial es birth."""
        iv = PersistenceInterval.essential(birth=0.5)
        assert iv.midpoint == 0.5
    
    def test_negative_birth_raises(self):
        """birth < 0 lanza ValueError."""
        with pytest.raises(ValueError, match="birth debe ser ≥ 0"):
            PersistenceInterval(birth=-1.0, death=1.0)
    
    def test_death_less_than_birth_raises(self):
        """death < birth (finito) lanza ValueError."""
        with pytest.raises(ValueError, match="death.*debe ser ≥ birth"):
            PersistenceInterval(birth=2.0, death=1.0)
    
    def test_negative_dimension_raises(self):
        """dimension < 0 lanza ValueError."""
        with pytest.raises(ValueError, match="dimension debe ser ≥ 0"):
            PersistenceInterval(birth=0.0, death=1.0, dimension=-1)
    
    def test_ordering_by_persistence(self):
        """Ordenamiento por persistencia descendente."""
        iv1 = PersistenceInterval(birth=0.0, death=1.0)  # pers = 1.0
        iv2 = PersistenceInterval(birth=0.0, death=2.0)  # pers = 2.0
        iv3 = PersistenceInterval(birth=0.0, death=0.5)  # pers = 0.5
        
        sorted_intervals = sorted([iv1, iv2, iv3])
        assert sorted_intervals[0].persistence == 2.0
        assert sorted_intervals[1].persistence == 1.0
        assert sorted_intervals[2].persistence == 0.5
    
    def test_essential_precedes_finite(self):
        """Intervalos esenciales preceden a finitos."""
        essential = PersistenceInterval.essential(birth=1.0)
        finite = PersistenceInterval(birth=0.0, death=100.0)  # pers = 100
        
        sorted_intervals = sorted([finite, essential])
        assert sorted_intervals[0].is_essential is True
    
    def test_to_dict(self):
        """Serialización a diccionario."""
        iv = PersistenceInterval(birth=0.5, death=1.5, dimension=1)
        d = iv.to_dict()
        
        assert d["birth"] == 0.5
        assert d["death"] == 1.5
        assert d["persistence"] == 1.0
        assert d["dimension"] == 1
        assert d["is_essential"] is False
        assert d["midpoint"] == 1.0
    
    def test_immutability(self):
        """Intervalo es inmutable (frozen)."""
        iv = PersistenceInterval(birth=0.0, death=1.0)
        with pytest.raises(AttributeError):
            iv.birth = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: BettiNumbers — Números de Betti
# ═══════════════════════════════════════════════════════════════════════════

class TestBettiNumbers:
    """Pruebas de números de Betti."""
    
    def test_valid_creation(self):
        """Crear números de Betti válidos."""
        betti = BettiNumbers(beta_0=3, beta_1=1, beta_2=0)
        assert betti.beta_0 == 3
        assert betti.beta_1 == 1
        assert betti.beta_2 == 0
    
    def test_euler_characteristic(self):
        """χ = β₀ - β₁ + β₂."""
        betti = BettiNumbers(beta_0=5, beta_1=2, beta_2=1)
        assert betti.euler_characteristic == 5 - 2 + 1  # = 4
    
    def test_total_rank(self):
        """Rango total = suma de Betti."""
        betti = BettiNumbers(beta_0=3, beta_1=2, beta_2=1)
        assert betti.total_rank == 6
    
    def test_is_connected(self):
        """Conexo cuando β₀ = 1."""
        connected = BettiNumbers(beta_0=1, beta_1=0, beta_2=0)
        disconnected = BettiNumbers(beta_0=3, beta_1=0, beta_2=0)
        
        assert connected.is_connected is True
        assert disconnected.is_connected is False
    
    def test_has_cycles(self):
        """Ciclos cuando β₁ > 0."""
        with_cycles = BettiNumbers(beta_0=1, beta_1=2, beta_2=0)
        no_cycles = BettiNumbers(beta_0=1, beta_1=0, beta_2=0)
        
        assert with_cycles.has_cycles is True
        assert no_cycles.has_cycles is False
    
    def test_zero_factory(self):
        """BettiNumbers.zero() crea (0, 0, 0)."""
        zero = BettiNumbers.zero()
        assert zero.beta_0 == 0
        assert zero.beta_1 == 0
        assert zero.beta_2 == 0
    
    def test_point_factory(self):
        """BettiNumbers.point() crea (1, 0, 0)."""
        point = BettiNumbers.point()
        assert point.beta_0 == 1
        assert point.beta_1 == 0
        assert point.beta_2 == 0
    
    def test_negative_betti_raises(self):
        """Valores negativos lanzan ValueError."""
        with pytest.raises(ValueError, match="beta_0"):
            BettiNumbers(beta_0=-1, beta_1=0, beta_2=0)
        
        with pytest.raises(ValueError, match="beta_1"):
            BettiNumbers(beta_0=1, beta_1=-1, beta_2=0)
    
    def test_non_integer_betti_raises(self):
        """Valores no enteros lanzan ValueError."""
        with pytest.raises(ValueError, match="beta_0"):
            BettiNumbers(beta_0=1.5, beta_1=0, beta_2=0)
    
    def test_to_dict(self):
        """Serialización a diccionario."""
        betti = BettiNumbers(beta_0=2, beta_1=1, beta_2=0)
        d = betti.to_dict()
        
        assert d["beta_0"] == 2
        assert d["beta_1"] == 1
        assert d["beta_2"] == 0
        assert d["betti_numbers"] == [2, 1, 0]
        assert d["euler_characteristic"] == 1
        assert d["total_rank"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: TopologicalSummary — Resumen topológico
# ═══════════════════════════════════════════════════════════════════════════

class TestTopologicalSummary:
    """Pruebas del resumen topológico."""
    
    def test_valid_creation(self):
        """Crear resumen válido."""
        betti = BettiNumbers(beta_0=5, beta_1=2, beta_2=0)
        summary = TopologicalSummary(
            betti=betti,
            structural_entropy=2.5,
            persistence_entropy=0.75,
            intrinsic_dimension=4,
        )
        
        assert summary.betti == betti
        assert summary.structural_entropy == 2.5
        assert summary.persistence_entropy == 0.75
        assert summary.intrinsic_dimension == 4
    
    def test_empty_factory(self):
        """TopologicalSummary.empty() crea resumen vacío."""
        empty = TopologicalSummary.empty()
        
        assert empty.betti.beta_0 == 0
        assert empty.structural_entropy == 0.0
        assert empty.persistence_entropy == 0.0
        assert empty.intrinsic_dimension == 0
    
    def test_negative_structural_entropy_raises(self):
        """structural_entropy < 0 lanza ValueError."""
        with pytest.raises(ValueError, match="structural_entropy"):
            TopologicalSummary(
                betti=BettiNumbers.zero(),
                structural_entropy=-1.0,
                persistence_entropy=0.5,
            )
    
    def test_persistence_entropy_out_of_range_raises(self):
        """persistence_entropy fuera de [0, 1] lanza ValueError."""
        with pytest.raises(ValueError, match="persistence_entropy"):
            TopologicalSummary(
                betti=BettiNumbers.zero(),
                structural_entropy=0.5,
                persistence_entropy=1.5,
            )
    
    def test_negative_dimension_raises(self):
        """intrinsic_dimension < 0 lanza ValueError."""
        with pytest.raises(ValueError, match="intrinsic_dimension"):
            TopologicalSummary(
                betti=BettiNumbers.zero(),
                structural_entropy=0.0,
                persistence_entropy=0.0,
                intrinsic_dimension=-1,
            )
    
    def test_to_dict(self):
        """Serialización incluye todos los campos."""
        betti = BettiNumbers(beta_0=3, beta_1=1, beta_2=0)
        summary = TopologicalSummary(
            betti=betti,
            structural_entropy=1.5,
            persistence_entropy=0.8,
            intrinsic_dimension=5,
        )
        d = summary.to_dict()
        
        assert d["beta_0"] == 3
        assert d["structural_entropy"] == 1.5
        assert d["persistence_entropy"] == 0.8
        assert d["intrinsic_dimension"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: IntentVector — Vector de intención
# ═══════════════════════════════════════════════════════════════════════════

class TestIntentVector:
    """Pruebas del vector de intención."""
    
    def test_valid_creation(self):
        """Crear vector válido."""
        iv = IntentVector(
            service_name="test_service",
            payload={"key": "value"},
            context={"user": "admin"},
        )
        
        assert iv.service_name == "test_service"
        assert iv.payload == {"key": "value"}
        assert iv.context == {"user": "admin"}
    
    def test_empty_service_name_raises(self):
        """Nombre de servicio vacío lanza ValueError."""
        with pytest.raises(ValueError, match="service_name no puede estar vacío"):
            IntentVector(service_name="")
        
        with pytest.raises(ValueError):
            IntentVector(service_name="   ")
    
    def test_default_payload_and_context(self):
        """Payload y context son dicts vacíos por defecto."""
        iv = IntentVector(service_name="test")
        assert iv.payload == {}
        assert iv.context == {}
    
    def test_payload_hash_deterministic(self):
        """payload_hash es determinístico."""
        iv1 = IntentVector(service_name="test", payload={"a": 1, "b": 2})
        iv2 = IntentVector(service_name="test", payload={"b": 2, "a": 1})
        
        # El hash debe ser el mismo para el mismo contenido
        assert iv1.payload_hash == iv2.payload_hash
    
    def test_payload_hash_different_for_different_payloads(self):
        """payload_hash difiere para payloads diferentes."""
        iv1 = IntentVector(service_name="test", payload={"a": 1})
        iv2 = IntentVector(service_name="test", payload={"a": 2})
        
        assert iv1.payload_hash != iv2.payload_hash
    
    def test_norm_calculation(self):
        """Norma = sqrt(|payload| + |context|)."""
        iv = IntentVector(
            service_name="test",
            payload={"a": 1, "b": 2, "c": 3},
            context={"x": 1},
        )
        expected_norm = math.sqrt(3 + 1)
        assert iv.norm == expected_norm
    
    def test_with_context_creates_new_vector(self):
        """with_context crea nuevo vector sin mutar el original."""
        original = IntentVector(
            service_name="test",
            payload={"key": "value"},
            context={"original": True},
        )
        
        extended = original.with_context(new_key="new_value")
        
        # Original no cambió
        assert "new_key" not in original.context
        # Extendido tiene ambos
        assert extended.context["original"] is True
        assert extended.context["new_key"] == "new_value"
    
    def test_immutability(self):
        """IntentVector es inmutable."""
        iv = IntentVector(service_name="test")
        with pytest.raises(AttributeError):
            iv.service_name = "other"


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: TTLCache — Cache con TTL
# ═══════════════════════════════════════════════════════════════════════════

class TestTTLCache:
    """Pruebas del cache con TTL."""
    
    def test_set_and_get(self):
        """Almacenar y recuperar valores."""
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        cache.set("key1", "value1")
        
        assert cache.get("key1") == "value1"
    
    def test_get_nonexistent_returns_none(self):
        """get de clave inexistente retorna None."""
        cache: TTLCache[str] = TTLCache()
        assert cache.get("nonexistent") is None
    
    def test_contains_operator(self):
        """Operador 'in' funciona correctamente."""
        cache: TTLCache[str] = TTLCache()
        cache.set("exists", "value")
        
        assert "exists" in cache
        assert "not_exists" not in cache
    
    def test_ttl_expiration(self):
        """Valores expiran después del TTL."""
        cache: TTLCache[str] = TTLCache(ttl_seconds=0.1)
        cache.set("key", "value")
        
        assert cache.get("key") == "value"
        
        time.sleep(0.15)
        
        assert cache.get("key") is None
    
    def test_lru_eviction(self):
        """Evicción LRU cuando se alcanza max_size."""
        cache: TTLCache[int] = TTLCache(ttl_seconds=60.0, max_size=3)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Acceder a 'a' para que no sea el LRU
        cache.get("a")
        
        # Agregar 'd' debería evictar 'b' (LRU)
        cache.set("d", 4)
        
        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache
        assert "d" in cache
    
    def test_update_existing_key(self):
        """Actualizar clave existente mueve al final del LRU."""
        cache: TTLCache[int] = TTLCache(ttl_seconds=60.0, max_size=3)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Actualizar 'a'
        cache.set("a", 100)
        
        # Agregar 'd' debería evictar 'b' (ahora LRU)
        cache.set("d", 4)
        
        assert cache.get("a") == 100
        assert "b" not in cache
    
    def test_get_or_compute(self):
        """get_or_compute ejecuta función solo si no está en cache."""
        cache: TTLCache[int] = TTLCache()
        call_count = 0
        
        def compute():
            nonlocal call_count
            call_count += 1
            return 42
        
        result1 = cache.get_or_compute("key", compute)
        result2 = cache.get_or_compute("key", compute)
        
        assert result1 == 42
        assert result2 == 42
        assert call_count == 1  # Solo se llamó una vez
    
    def test_clear_returns_count(self):
        """clear() retorna número de entradas eliminadas."""
        cache: TTLCache[str] = TTLCache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        
        count = cache.clear()
        assert count == 3
        assert cache.size == 0
    
    def test_prune_expired(self):
        """prune_expired elimina solo entradas expiradas."""
        cache: TTLCache[str] = TTLCache(ttl_seconds=0.1)
        cache.set("a", "1")
        
        time.sleep(0.15)
        
        cache.set("b", "2")  # Esta no está expirada
        
        pruned = cache.prune_expired()
        
        assert pruned == 1
        assert "a" not in cache
        assert "b" in cache
    
    def test_size_property(self):
        """size retorna número de entradas."""
        cache: TTLCache[str] = TTLCache()
        assert cache.size == 0
        
        cache.set("a", "1")
        cache.set("b", "2")
        assert cache.size == 2
    
    def test_hit_rate(self):
        """hit_rate calcula correctamente."""
        cache: TTLCache[str] = TTLCache()
        cache.set("exists", "value")
        
        cache.get("exists")  # Hit
        cache.get("exists")  # Hit
        cache.get("missing")  # Miss
        
        assert cache.hit_rate == 2 / 3
    
    def test_hit_rate_no_queries(self):
        """hit_rate es 0 sin consultas."""
        cache: TTLCache[str] = TTLCache()
        assert cache.hit_rate == 0.0
    
    def test_stats(self):
        """stats retorna CacheStats completo."""
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0, max_size=10)
        cache.set("a", "1")
        cache.get("a")
        cache.get("missing")
        
        stats = cache.stats
        
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["ttl_seconds"] == 60.0
        assert "evictions" in stats
        assert "expirations" in stats
    
    def test_thread_safety(self):
        """Cache es thread-safe."""
        cache: TTLCache[int] = TTLCache(max_size=100)
        errors = []
        
        def writer(n: int):
            try:
                for i in range(100):
                    cache.set(f"key_{n}_{i}", i)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for _ in range(100):
                    cache.get("key_0_50")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer, args=(i,))
            for i in range(5)
        ]
        threads.extend([
            threading.Thread(target=reader)
            for _ in range(5)
        ])
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: LatencyHistogram — Histograma de latencias
# ═══════════════════════════════════════════════════════════════════════════

class TestLatencyHistogram:
    """Pruebas del histograma de latencias."""
    
    def test_record_and_stats(self):
        """Registrar latencias y obtener estadísticas."""
        hist = LatencyHistogram(max_size=100)
        
        for latency in [10, 20, 30, 40, 50]:
            hist.record(latency)
        
        stats = hist.get_stats()
        
        assert stats["count"] == 5
        assert stats["mean_ms"] == 30.0
        assert stats["median_ms"] == 30.0
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 50.0
    
    def test_empty_histogram_stats(self):
        """Histograma vacío tiene stats en cero."""
        hist = LatencyHistogram()
        stats = hist.get_stats()
        
        assert stats["count"] == 0
        assert stats["mean_ms"] == 0.0
        assert stats["median_ms"] == 0.0
    
    def test_percentiles(self):
        """Cálculo de percentiles."""
        hist = LatencyHistogram()
        
        # 100 valores de 1 a 100
        for i in range(1, 101):
            hist.record(float(i))
        
        stats = hist.get_stats()
        
        assert 90 <= stats["p95_ms"] <= 100
        assert 95 <= stats["p99_ms"] <= 100
    
    def test_context_manager_measure(self):
        """Context manager mide latencia automáticamente."""
        hist = LatencyHistogram()
        
        with hist.measure():
            time.sleep(0.05)  # 50ms
        
        stats = hist.get_stats()
        assert stats["count"] == 1
        assert stats["mean_ms"] >= 45  # Al menos ~45ms
    
    def test_buffer_circular(self):
        """Buffer circular no excede max_size."""
        hist = LatencyHistogram(max_size=10)
        
        for i in range(100):
            hist.record(float(i))
        
        stats = hist.get_stats()
        assert stats["count"] == 100  # Total registrado
        # Pero solo los últimos 10 están en el buffer
        assert stats["min_ms"] >= 90  # Los últimos 10 son 90-99
    
    def test_reset(self):
        """reset limpia el histograma."""
        hist = LatencyHistogram()
        hist.record(100)
        hist.record(200)
        
        hist.reset()
        
        stats = hist.get_stats()
        assert stats["count"] == 0
    
    def test_thread_safety(self):
        """Histograma es thread-safe."""
        hist = LatencyHistogram(max_size=1000)
        errors = []
        
        def record_many():
            try:
                for _ in range(100):
                    hist.record(1.0)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: MICMetrics — Métricas agregadas
# ═══════════════════════════════════════════════════════════════════════════

class TestMICMetrics:
    """Pruebas de métricas de la MIC."""
    
    def test_initial_state(self):
        """Estado inicial tiene contadores en cero."""
        metrics = MICMetrics()
        
        assert metrics.projections == 0
        assert metrics.cache_hits == 0
        assert metrics.violations == 0
        assert metrics.errors == 0
        assert metrics.timeouts == 0
    
    def test_record_projection(self):
        """record_projection incrementa contadores."""
        metrics = MICMetrics()
        
        metrics.record_projection(Stratum.PHYSICS)
        metrics.record_projection(Stratum.PHYSICS)
        metrics.record_projection(Stratum.TACTICS)
        
        assert metrics.projections == 3
        assert metrics.projections_by_stratum["PHYSICS"] == 2
        assert metrics.projections_by_stratum["TACTICS"] == 1
    
    def test_record_error(self):
        """record_error incrementa contadores por categoría."""
        metrics = MICMetrics()
        
        metrics.record_error("validation")
        metrics.record_error("validation")
        metrics.record_error("execution")
        
        assert metrics.errors == 3
        assert metrics.errors_by_category["validation"] == 2
        assert metrics.errors_by_category["execution"] == 1
    
    def test_to_dict(self):
        """to_dict serializa todas las métricas."""
        metrics = MICMetrics()
        metrics.record_projection(Stratum.PHYSICS)
        metrics.record_error("test")
        
        d = metrics.to_dict()
        
        assert "counters" in d
        assert d["counters"]["projections"] == 1
        assert d["counters"]["errors"] == 1
        assert "projections_by_stratum" in d
        assert "errors_by_category" in d
        assert "latency" in d


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Excepciones — Jerarquía y serialización
# ═══════════════════════════════════════════════════════════════════════════

class TestExceptions:
    """Pruebas de la jerarquía de excepciones."""
    
    def test_mic_exception_base(self):
        """MICException tiene atributos correctos."""
        exc = MICException(
            "Test error",
            details={"key": "value"},
            category="test",
        )
        
        assert str(exc) == "Test error"
        assert exc.details == {"key": "value"}
        assert exc.category == "test"
        assert exc.timestamp > 0
    
    def test_mic_exception_to_dict(self):
        """to_dict serializa la excepción."""
        exc = MICException("Test error", category="test_cat")
        d = exc.to_dict()
        
        assert d["error"] == "Test error"
        assert d["error_type"] == "MICException"
        assert d["error_category"] == "test_cat"
        assert "timestamp" in d
    
    def test_file_not_found_diagnostic_error(self):
        """FileNotFoundDiagnosticError incluye path."""
        exc = FileNotFoundDiagnosticError("/path/to/file.csv")
        
        assert "/path/to/file.csv" in str(exc)
        assert exc.details["path"] == "/path/to/file.csv"
        assert exc.category == "validation"
    
    def test_unsupported_file_type_error(self):
        """UnsupportedFileTypeError incluye opciones."""
        exc = UnsupportedFileTypeError("xyz", ["apus", "insumos"])
        
        assert "xyz" in str(exc)
        assert exc.details["file_type"] == "xyz"
        assert exc.details["available_types"] == ["apus", "insumos"]
    
    def test_file_validation_error(self):
        """FileValidationError acepta kwargs adicionales."""
        exc = FileValidationError(
            "Invalid extension",
            extension=".xyz",
            expected=[".csv", ".txt"],
        )
        
        assert exc.details["extension"] == ".xyz"
        assert exc.details["expected"] == [".csv", ".txt"]
    
    def test_file_permission_error(self):
        """FilePermissionError incluye operación."""
        exc = MICFilePermissionError("/path/to/file", "read")
        
        assert "Permission denied" in str(exc)
        assert exc.details["path"] == "/path/to/file"
        assert exc.details["operation"] == "read"
    
    def test_hierarchy_violation_error(self):
        """MICHierarchyViolationError incluye estratos."""
        exc = MICHierarchyViolationError(
            target_stratum=Stratum.STRATEGY,
            missing_strata={Stratum.PHYSICS, Stratum.TACTICS},
            validated_strata=set(),
        )
        
        assert "STRATEGY" in str(exc)
        assert exc.target_stratum == Stratum.STRATEGY
        assert Stratum.PHYSICS in exc.missing_strata
        assert exc.details["missing_strata"] == ["PHYSICS", "TACTICS"]
    
    def test_timeout_error(self):
        """TimeoutError incluye tiempos."""
        exc = MICTimeoutError("operation_x", 5.0, 7.5)
        
        assert "operation_x" in str(exc)
        assert "7.50s" in str(exc)
        assert exc.details["timeout_seconds"] == 5.0
        assert exc.details["elapsed_seconds"] == 7.5


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Validación de Archivos
# ═══════════════════════════════════════════════════════════════════════════

class TestFileValidation:
    """Pruebas de validación de archivos."""
    
    def test_normalize_path_valid(self, tmp_path: Path):
        """Normaliza path válido."""
        file_path = tmp_path / "test.csv"
        file_path.touch()
        
        result = normalize_path(str(file_path))
        
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_normalize_path_none_raises(self):
        """Path None lanza ValueError."""
        with pytest.raises(ValueError, match="Path no puede ser None"):
            normalize_path(None)
    
    def test_normalize_path_empty_raises(self):
        """Path vacío lanza ValueError."""
        with pytest.raises(ValueError, match="Path no puede estar vacío"):
            normalize_path("")
        
        with pytest.raises(ValueError):
            normalize_path("   ")
    
    def test_validate_file_exists_success(self, temp_csv_file: Path):
        """Archivo existente pasa validación."""
        validate_file_exists(temp_csv_file)  # No debe lanzar
    
    def test_validate_file_exists_not_found(self, tmp_path: Path):
        """Archivo inexistente lanza FileNotFoundDiagnosticError."""
        with pytest.raises(FileNotFoundDiagnosticError):
            validate_file_exists(tmp_path / "nonexistent.csv")
    
    def test_validate_file_exists_directory(self, tmp_path: Path):
        """Directorio lanza FileValidationError."""
        with pytest.raises(FileValidationError, match="no apunta a un archivo"):
            validate_file_exists(tmp_path)
    
    def test_validate_file_extension_valid(self, temp_csv_file: Path):
        """Extensión .csv es válida."""
        ext = validate_file_extension(temp_csv_file)
        assert ext == ".csv"
    
    def test_validate_file_extension_invalid(self, tmp_path: Path):
        """Extensión inválida lanza FileValidationError."""
        invalid_file = tmp_path / "test.xyz"
        invalid_file.touch()
        
        with pytest.raises(FileValidationError, match="Extensión no soportada"):
            validate_file_extension(invalid_file)
    
    def test_validate_file_size_valid(self, temp_csv_file: Path):
        """Archivo dentro del límite pasa."""
        size, is_empty = validate_file_size(temp_csv_file)
        
        assert size > 0
        assert is_empty is False
    
    def test_validate_file_size_empty(self, temp_empty_file: Path):
        """Archivo vacío retorna is_empty=True."""
        size, is_empty = validate_file_size(temp_empty_file)
        
        assert size == 0
        assert is_empty is True
    
    def test_validate_file_size_exceeds_limit(self, temp_csv_file: Path):
        """Archivo que excede límite lanza FileValidationError."""
        with pytest.raises(FileValidationError, match="excede el límite"):
            validate_file_size(temp_csv_file, max_size=10)  # 10 bytes
    
    def test_normalize_encoding_standard(self):
        """Codificaciones estándar se normalizan."""
        assert normalize_encoding("utf-8") == "utf-8"
        assert normalize_encoding("UTF-8") == "utf-8"
        assert normalize_encoding("latin-1") == "latin-1"
    
    def test_normalize_encoding_aliases(self):
        """Aliases se convierten a forma canónica."""
        assert normalize_encoding("utf8") == "utf-8"
        assert normalize_encoding("latin1") == "latin-1"
        assert normalize_encoding("iso88591") == "iso-8859-1"
    
    def test_normalize_encoding_unknown(self):
        """Codificación desconocida retorna utf-8."""
        assert normalize_encoding("unknown_encoding") == "utf-8"
    
    def test_normalize_encoding_empty(self):
        """Codificación vacía retorna utf-8."""
        assert normalize_encoding("") == "utf-8"
        assert normalize_encoding("   ") == "utf-8"
    
    def test_normalize_file_type_enum(self):
        """FileType se retorna sin cambios."""
        result = normalize_file_type(FileType.APUS)
        assert result == FileType.APUS
    
    def test_normalize_file_type_string(self):
        """String se convierte a FileType."""
        result = normalize_file_type("insumos")
        assert result == FileType.INSUMOS
    
    def test_validate_file_for_processing_valid(self, temp_csv_file: Path):
        """Validación completa de archivo válido."""
        result = validate_file_for_processing(temp_csv_file)
        
        assert result["valid"] is True
        assert result["extension"] == ".csv"
        assert result["is_empty"] is False
        assert "size" in result


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Funciones Matemáticas
# ═══════════════════════════════════════════════════════════════════════════

class TestShannonEntropy:
    """Pruebas de entropía de Shannon."""
    
    def test_empty_distribution(self):
        """Distribución vacía tiene entropía 0."""
        assert compute_shannon_entropy([]) == 0.0
    
    def test_single_element(self):
        """Un solo elemento tiene entropía 0."""
        assert compute_shannon_entropy([1.0]) == 0.0
    
    def test_uniform_distribution(self):
        """Distribución uniforme tiene entropía máxima."""
        # 4 elementos uniformes: log₂(4) = 2
        entropy = compute_shannon_entropy([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy - 2.0) < 1e-10
    
    def test_biased_distribution(self):
        """Distribución sesgada tiene entropía menor que uniforme."""
        uniform_entropy = compute_shannon_entropy([0.25, 0.25, 0.25, 0.25])
        biased_entropy = compute_shannon_entropy([0.7, 0.1, 0.1, 0.1])
        
        assert biased_entropy < uniform_entropy
    
    def test_auto_normalization(self):
        """Distribución se normaliza automáticamente."""
        # Suma = 10, no 1
        entropy = compute_shannon_entropy([5, 5])
        assert abs(entropy - 1.0) < 1e-10  # log₂(2) = 1
    
    def test_different_bases(self):
        """Diferentes bases producen diferentes valores."""
        probs = [0.5, 0.5]
        
        bits = compute_shannon_entropy(probs, base=2.0)
        nats = compute_shannon_entropy(probs, base=math.e)
        
        assert abs(bits - 1.0) < 1e-10  # log₂(2) = 1
        assert abs(nats - math.log(2)) < 1e-10
    
    def test_negative_probability_raises(self):
        """Probabilidad negativa lanza ValueError."""
        with pytest.raises(ValueError, match="no pueden ser negativas"):
            compute_shannon_entropy([0.5, -0.5])
    
    def test_invalid_base_raises(self):
        """Base <= 1 lanza ValueError."""
        with pytest.raises(ValueError, match="base debe ser > 1"):
            compute_shannon_entropy([0.5, 0.5], base=1.0)
    
    def test_entropy_always_non_negative(self):
        """Entropía siempre es >= 0."""
        for _ in range(100):
            probs = np.random.dirichlet(np.ones(5))
            entropy = compute_shannon_entropy(probs.tolist())
            assert entropy >= 0


class TestDistributionFromCounts:
    """Pruebas de conversión de conteos a distribución."""
    
    def test_empty_counts(self):
        """Conteos vacíos retornan lista vacía."""
        assert distribution_from_counts({}) == []
    
    def test_zero_total(self):
        """Total cero retorna lista vacía."""
        assert distribution_from_counts({"a": 0, "b": 0}) == []
    
    def test_valid_counts(self):
        """Conteos válidos se convierten a probabilidades."""
        result = distribution_from_counts({"a": 3, "b": 1})
        
        assert len(result) == 2
        assert sum(result) == 1.0
        assert 0.75 in result
        assert 0.25 in result
    
    def test_counter_input(self):
        """Acepta Counter como input."""
        counts = Counter(["a", "a", "b"])
        result = distribution_from_counts(counts)
        
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-10


class TestPersistenceEntropy:
    """Pruebas de entropía de persistencia."""
    
    def test_empty_intervals(self):
        """Sin intervalos retorna 0."""
        assert compute_persistence_entropy([]) == 0.0
    
    def test_all_essential(self):
        """Solo intervalos esenciales retorna 0."""
        intervals = [
            PersistenceInterval.essential(0.0),
            PersistenceInterval.essential(1.0),
        ]
        assert compute_persistence_entropy(intervals) == 0.0
    
    def test_uniform_persistence(self):
        """Persistencias uniformes dan entropía 1."""
        intervals = [
            PersistenceInterval(birth=0.0, death=1.0),
            PersistenceInterval(birth=0.0, death=1.0),
            PersistenceInterval(birth=0.0, death=1.0),
        ]
        entropy = compute_persistence_entropy(intervals)
        assert abs(entropy - 1.0) < 1e-10
    
    def test_entropy_in_range(self):
        """Entropía siempre en [0, 1]."""
        intervals = [
            PersistenceInterval(birth=0.0, death=1.0),
            PersistenceInterval(birth=0.0, death=2.0),
            PersistenceInterval(birth=0.0, death=0.5),
        ]
        entropy = compute_persistence_entropy(intervals)
        assert 0.0 <= entropy <= 1.0


class TestTopologicalAnalysis:
    """Pruebas de análisis topológico."""
    
    def test_analyze_features_valid_file(self, temp_csv_file: Path):
        """Análisis de archivo válido produce resumen."""
        summary = analyze_topological_features(temp_csv_file)
        
        assert summary.betti.beta_0 >= 1
        assert summary.structural_entropy >= 0
        assert summary.intrinsic_dimension >= 1
    
    def test_analyze_features_nonexistent_file(self, tmp_path: Path):
        """Archivo inexistente retorna resumen vacío."""
        summary = analyze_topological_features(tmp_path / "nonexistent.csv")
        
        assert summary == TopologicalSummary.empty()
    
    def test_estimate_intrinsic_dimension_csv(self):
        """Dimensión intrínseca = número de columnas."""
        lines = [
            "A,B,C,D,E",
            "1,2,3,4,5",
            "6,7,8,9,10",
        ]
        dim = estimate_intrinsic_dimension(lines)
        assert dim == 5
    
    def test_estimate_intrinsic_dimension_empty(self):
        """Líneas vacías retornan 0."""
        assert estimate_intrinsic_dimension([]) == 0
    
    def test_detect_cyclic_patterns_no_cycles(self):
        """Sin patrones cíclicos retorna 0."""
        lines = ["a", "b", "c", "d", "e", "f", "g", "h"]
        cycles = detect_cyclic_patterns(lines)
        assert cycles == 0
    
    def test_detect_cyclic_patterns_with_cycles(self, temp_cyclic_csv: Path):
        """Detecta patrones cíclicos."""
        with open(temp_cyclic_csv, "r") as f:
            lines = f.readlines()
        
        cycles = detect_cyclic_patterns(lines)
        assert cycles > 0
    
    def test_jaccard_similarity_identical(self):
        """Conjuntos idénticos tienen similitud 1."""
        tokens = frozenset({"a", "b", "c"})
        assert _jaccard_similarity(tokens, tokens) == 1.0
    
    def test_jaccard_similarity_disjoint(self):
        """Conjuntos disjuntos tienen similitud 0."""
        a = frozenset({"a", "b"})
        b = frozenset({"c", "d"})
        assert _jaccard_similarity(a, b) == 0.0
    
    def test_jaccard_similarity_partial(self):
        """Conjuntos con solapamiento parcial."""
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        # Intersección: {b, c} = 2
        # Unión: {a, b, c, d} = 4
        assert _jaccard_similarity(a, b) == 0.5
    
    def test_tokenize_line(self):
        """Tokeniza línea correctamente."""
        tokens = _tokenize_line("  a,b;c\td  ")
        assert tokens == frozenset({"a", "b", "c", "d"})


class TestHomologyAndPersistence:
    """Pruebas de cálculo de homología y diagramas de persistencia."""
    
    def test_compute_homology_basic(self, sample_diagnostic_data: Dict):
        """Homología básica de datos diagnósticos."""
        homology = compute_homology_from_diagnostic(sample_diagnostic_data)
        
        assert "H_0" in homology
        assert "beta_0" in homology
        assert homology["beta_0"] >= 1
    
    def test_compute_homology_with_cycles(self, sample_diagnostic_data: Dict):
        """Detecta ciclos en warnings."""
        # El fixture tiene "circular dependency"
        homology = compute_homology_from_diagnostic(sample_diagnostic_data)
        assert homology["beta_1"] >= 1
    
    def test_compute_persistence_diagram(self, sample_diagnostic_data: Dict):
        """Diagrama de persistencia de issues."""
        intervals = compute_persistence_diagram(sample_diagnostic_data)
        
        assert len(intervals) > 0
        for iv in intervals:
            assert isinstance(iv, PersistenceInterval)
            assert iv.persistence > 0
    
    def test_compute_persistence_diagram_empty(self):
        """Sin issues retorna lista vacía."""
        intervals = compute_persistence_diagram({"issues": []})
        assert intervals == []
    
    def test_compute_diagnostic_magnitude_basic(self, sample_diagnostic_data: Dict):
        """Magnitud diagnóstica en [0, 1]."""
        magnitude = compute_diagnostic_magnitude(sample_diagnostic_data)
        assert 0.0 <= magnitude <= 1.0
    
    def test_compute_diagnostic_magnitude_critical(self, sample_diagnostic_critical: Dict):
        """Errores críticos dan mayor magnitud."""
        critical_mag = compute_diagnostic_magnitude(sample_diagnostic_critical)
        normal_mag = compute_diagnostic_magnitude({"issues": [], "warnings": [], "errors": []})
        
        assert critical_mag > normal_mag


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: SpectralGraphMetrics
# ═══════════════════════════════════════════════════════════════════════════

class TestSpectralGraphMetrics:
    """Pruebas de análisis espectral."""
    
    def test_empty_registry(self, mic_registry: MICRegistry):
        """Registro vacío produce métricas vacías."""
        analyzer = SpectralGraphMetrics(mic_registry)
        metrics = analyzer.compute_spectral_metrics()
        
        assert metrics["n_services"] == 0
        assert metrics["is_connected"] is False
    
    def test_single_service(self, mic_registry: MICRegistry):
        """Un solo servicio produce grafo trivial."""
        mic_registry.register_vector(
            "test", Stratum.PHYSICS, lambda **kw: {"success": True}
        )
        
        analyzer = SpectralGraphMetrics(mic_registry)
        metrics = analyzer.compute_spectral_metrics()
        
        assert metrics["n_services"] == 1
    
    def test_connected_services(self, mic_with_vectors: MICRegistry):
        """Servicios con dependencias forman grafo conectado."""
        analyzer = SpectralGraphMetrics(mic_with_vectors)
        metrics = analyzer.compute_spectral_metrics()
        
        assert metrics["n_services"] == 4
        assert "algebraic_connectivity" in metrics
        assert "spectral_radius" in metrics
        assert "spectral_energy" in metrics
    
    def test_adjacency_matrix_shape(self, mic_with_vectors: MICRegistry):
        """Matriz de adyacencia tiene forma correcta."""
        analyzer = SpectralGraphMetrics(mic_with_vectors)
        A = analyzer.build_adjacency_matrix()
        
        n = mic_with_vectors.dimension
        assert A.shape == (n, n)
    
    def test_laplacian_symmetric(self, mic_with_vectors: MICRegistry):
        """Laplaciana es simétrica."""
        analyzer = SpectralGraphMetrics(mic_with_vectors)
        L = analyzer.build_laplacian()
        
        assert np.allclose(L, L.T)


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: StratumTransitionMatrix
# ═══════════════════════════════════════════════════════════════════════════

class TestStratumTransitionMatrix:
    """Pruebas de matriz de transición entre estratos."""
    
    def test_build_matrix_shape(self):
        """Matriz tiene forma n×n donde n = número de estratos."""
        stm = StratumTransitionMatrix()
        counts = {s: 1 for s in Stratum}
        T = stm.build(counts)
        
        n = len(list(Stratum))
        assert T.shape == (n, n)
    
    def test_matrix_is_stochastic(self):
        """Filas de la matriz suman 1."""
        stm = StratumTransitionMatrix()
        counts = {s: 2 for s in Stratum}
        T = stm.build(counts)
        
        row_sums = T.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
    
    def test_wisdom_is_absorbing(self):
        """WISDOM es estado absorbente (T[wisdom, wisdom] = 1)."""
        stm = StratumTransitionMatrix()
        counts = {s: 1 for s in Stratum}
        T = stm.build(counts)
        
        wisdom_idx = stm._idx[Stratum.WISDOM]
        assert T[wisdom_idx, wisdom_idx] == 1.0
    
    def test_transition_direction(self):
        """Transiciones van de base a cúspide."""
        stm = StratumTransitionMatrix()
        counts = {s: 1 for s in Stratum}
        T = stm.build(counts)
        
        physics_idx = stm._idx[Stratum.PHYSICS]
        tactics_idx = stm._idx[Stratum.TACTICS]
        
        # PHYSICS puede transicionar a TACTICS
        # (PHYSICS es prerrequisito de TACTICS)
        assert T[physics_idx, tactics_idx] > 0
    
    def test_stationary_distribution(self):
        """Distribución estacionaria suma 1."""
        stm = StratumTransitionMatrix()
        counts = {s: 1 for s in Stratum}
        
        stationary = stm.stationary_distribution(counts)
        
        total = sum(stationary.values())
        assert abs(total - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: MICRegistry — Core
# ═══════════════════════════════════════════════════════════════════════════

class TestMICRegistry:
    """Pruebas del registro central MIC."""
    
    def test_initial_state(self, mic_registry: MICRegistry):
        """Estado inicial vacío."""
        assert mic_registry.dimension == 0
        assert mic_registry.registered_services == []
    
    def test_register_vector(self, mic_registry: MICRegistry):
        """Registrar vector aumenta dimensión."""
        mic_registry.register_vector(
            "test_service",
            Stratum.PHYSICS,
            lambda **kw: {"success": True}
        )
        
        assert mic_registry.dimension == 1
        assert "test_service" in mic_registry.registered_services
    
    def test_register_empty_name_raises(self, mic_registry: MICRegistry):
        """Nombre vacío lanza ValueError."""
        with pytest.raises(ValueError, match="service_name no puede estar vacío"):
            mic_registry.register_vector("", Stratum.PHYSICS, lambda: {})
    
    def test_register_invalid_stratum_raises(self, mic_registry: MICRegistry):
        """Stratum inválido lanza TypeError."""
        with pytest.raises(TypeError, match="stratum debe ser Stratum"):
            mic_registry.register_vector("test", "PHYSICS", lambda: {})
    
    def test_register_non_callable_raises(self, mic_registry: MICRegistry):
        """Handler no callable lanza TypeError."""
        with pytest.raises(TypeError, match="handler debe ser callable"):
            mic_registry.register_vector("test", Stratum.PHYSICS, "not_callable")
    
    def test_unregister_vector(self, mic_with_vectors: MICRegistry):
        """Eliminar vector reduce dimensión."""
        initial_dim = mic_with_vectors.dimension
        
        result = mic_with_vectors.unregister_vector("test_physics")
        
        assert result is True
        assert mic_with_vectors.dimension == initial_dim - 1
    
    def test_unregister_nonexistent(self, mic_registry: MICRegistry):
        """Eliminar vector inexistente retorna False."""
        result = mic_registry.unregister_vector("nonexistent")
        assert result is False
    
    def test_is_registered(self, mic_with_vectors: MICRegistry):
        """is_registered funciona correctamente."""
        assert mic_with_vectors.is_registered("test_physics") is True
        assert mic_with_vectors.is_registered("nonexistent") is False
    
    def test_get_stratum(self, mic_with_vectors: MICRegistry):
        """get_stratum retorna estrato correcto."""
        assert mic_with_vectors.get_stratum("test_physics") == Stratum.PHYSICS
        assert mic_with_vectors.get_stratum("test_tactics") == Stratum.TACTICS
        assert mic_with_vectors.get_stratum("nonexistent") is None
    
    def test_get_services_by_stratum(self, mic_with_vectors: MICRegistry):
        """get_services_by_stratum filtra correctamente."""
        physics_services = mic_with_vectors.get_services_by_stratum(Stratum.PHYSICS)
        
        assert "test_physics" in physics_services
        assert "test_tactics" not in physics_services
    
    def test_get_stratum_hierarchy(self, mic_with_vectors: MICRegistry):
        """get_stratum_hierarchy retorna estructura completa."""
        hierarchy = mic_with_vectors.get_stratum_hierarchy()
        
        assert "PHYSICS" in hierarchy
        assert "TACTICS" in hierarchy
        assert "STRATEGY" in hierarchy
        assert "WISDOM" in hierarchy
    
    def test_project_intent_success(self, mic_with_vectors: MICRegistry):
        """Proyección exitosa con contexto válido."""
        result = mic_with_vectors.project_intent(
            service_name="test_physics",
            payload={"key": "value"},
            context={"validated_strata": set()},  # PHYSICS no requiere nada
        )
        
        assert result["success"] is True
        assert result["_mic_stratum"] == "PHYSICS"
    
    def test_project_intent_missing_prerequisites(self, mic_with_vectors: MICRegistry):
        """Proyección falla sin prerrequisitos."""
        result = mic_with_vectors.project_intent(
            service_name="test_strategy",
            payload={},
            context={"validated_strata": set()},  # STRATEGY requiere PHYSICS y TACTICS
        )
        
        assert result["success"] is False
        assert "error_category" in result
        assert result["error_category"] == "hierarchy_violation"
    
    def test_project_intent_with_prerequisites(self, mic_with_vectors: MICRegistry):
        """Proyección exitosa con prerrequisitos."""
        result = mic_with_vectors.project_intent(
            service_name="test_strategy",
            payload={},
            context={"validated_strata": {Stratum.PHYSICS, Stratum.TACTICS}},
        )
        
        assert result["success"] is True
        assert result["_mic_stratum"] == "STRATEGY"
    
    def test_project_intent_unknown_service(self, mic_with_vectors: MICRegistry):
        """Proyección a servicio desconocido falla."""
        result = mic_with_vectors.project_intent(
            service_name="nonexistent",
            payload={},
            context={},
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_project_intent_with_cache(self, mic_with_vectors: MICRegistry):
        """Cache funciona en proyecciones."""
        payload = {"key": "value"}
        context = {"validated_strata": set()}
        
        # Primera llamada
        result1 = mic_with_vectors.project_intent(
            service_name="test_physics",
            payload=payload,
            context=context,
            use_cache=True,
        )
        
        # Segunda llamada (debería usar cache)
        result2 = mic_with_vectors.project_intent(
            service_name="test_physics",
            payload=payload,
            context=context,
            use_cache=True,
        )
        
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Verificar que hubo hit de cache
        metrics = mic_with_vectors.metrics
        assert metrics["counters"]["cache_hits"] >= 1
    
    def test_project_intent_force_override(self, mic_with_vectors: MICRegistry):
        """force_physics_override bypasea validación."""
        result = mic_with_vectors.project_intent(
            service_name="test_strategy",
            payload={},
            context={
                "validated_strata": set(),
                "force_physics_override": True,
            },
        )
        
        assert result["success"] is True
    
    def test_clear_cache(self, mic_with_vectors: MICRegistry):
        """clear_cache limpia el cache."""
        mic_with_vectors.project_intent(
            service_name="test_physics",
            payload={},
            context={},
            use_cache=True,
        )
        
        count = mic_with_vectors.clear_cache()
        assert count >= 0
    
    def test_spectral_analysis(self, mic_with_vectors: MICRegistry):
        """spectral_analysis retorna métricas."""
        metrics = mic_with_vectors.spectral_analysis()
        
        assert "algebraic_connectivity" in metrics
        assert "n_services" in metrics
    
    def test_stratum_statistics(self, mic_with_vectors: MICRegistry):
        """stratum_statistics retorna estadísticas."""
        stats = mic_with_vectors.stratum_statistics()
        
        assert "counts_by_stratum" in stats
        assert "distribution" in stats
        assert "stratum_entropy" in stats
        assert "total_services" in stats
    
    def test_metrics_property(self, mic_with_vectors: MICRegistry):
        """metrics retorna métricas completas."""
        mic_with_vectors.project_intent(
            service_name="test_physics",
            payload={},
            context={},
        )
        
        metrics = mic_with_vectors.metrics
        
        assert "counters" in metrics
        assert "cache" in metrics


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Command Pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestProjectionCommands:
    """Pruebas de comandos de proyección."""
    
    def test_projection_context_creation(self):
        """Crear contexto de proyección."""
        ctx = ProjectionContext(
            service_name="test",
            payload={"key": "value"},
            context={"ctx_key": "ctx_value"},
            use_cache=True,
        )
        
        assert ctx.service_name == "test"
        assert ctx.use_cache is True
        assert ctx.cache_key is None  # Aún no computado
    
    def test_normalization_command(self):
        """NormalizationCommand normaliza estratos."""
        cmd = NormalizationCommand()
        ctx = ProjectionContext(
            service_name="test",
            payload={},
            context={"validated_strata": ["PHYSICS", "TACTICS"]},
            use_cache=False,
        )
        
        result = cmd.execute(ctx)
        
        assert result is None  # Continúa al siguiente comando
        assert Stratum.PHYSICS in ctx.validated_strata
        assert Stratum.TACTICS in ctx.validated_strata
    
    def test_normalization_command_with_integers(self):
        """NormalizationCommand acepta enteros."""
        cmd = NormalizationCommand()
        ctx = ProjectionContext(
            service_name="test",
            payload={},
            context={"validated_strata": [4, 3]},  # PHYSICS=4, TACTICS=3
            use_cache=False,
        )
        
        cmd.execute(ctx)
        
        assert Stratum.PHYSICS in ctx.validated_strata
        assert Stratum.TACTICS in ctx.validated_strata


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Singleton Global
# ═══════════════════════════════════════════════════════════════════════════

class TestGlobalMIC:
    """Pruebas del singleton global."""
    
    def setup_method(self):
        """Resetear MIC global antes de cada test."""
        reset_global_mic()
    
    def teardown_method(self):
        """Resetear MIC global después de cada test."""
        reset_global_mic()
    
    def test_get_global_mic_creates_instance(self):
        """get_global_mic crea instancia."""
        mic = get_global_mic()
        
        assert isinstance(mic, MICRegistry)
        assert mic.dimension > 0  # Tiene vectores registrados
    
    def test_get_global_mic_returns_same_instance(self):
        """Llamadas sucesivas retornan la misma instancia."""
        mic1 = get_global_mic()
        mic2 = get_global_mic()
        
        assert mic1 is mic2
    
    def test_reset_global_mic(self):
        """reset_global_mic limpia la instancia."""
        mic1 = get_global_mic()
        reset_global_mic()
        mic2 = get_global_mic()
        
        assert mic1 is not mic2
    
    def test_force_reinit(self):
        """force_reinit crea nueva instancia."""
        mic1 = get_global_mic()
        mic2 = get_global_mic(force_reinit=True)
        
        assert mic1 is not mic2
    
    def test_custom_config(self):
        """Configuración personalizada se aplica."""
        custom_config = MICConfiguration(
            cache_ttl_seconds=120.0,
            cache_max_size=64,
        )
        
        mic = get_global_mic(mic_config=custom_config)
        
        assert mic.config.cache_ttl_seconds == 120.0
        assert mic.config.cache_max_size == 64


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: API Pública
# ═══════════════════════════════════════════════════════════════════════════

class TestPublicAPI:
    """Pruebas de funciones de API pública."""
    
    def test_get_supported_file_types(self):
        """Retorna tipos de archivo soportados."""
        types = get_supported_file_types()
        
        assert "apus" in types
        assert "insumos" in types
        assert "presupuesto" in types
    
    def test_get_supported_delimiters(self):
        """Retorna delimitadores soportados."""
        delimiters = get_supported_delimiters()
        
        assert "," in delimiters
        assert ";" in delimiters
        assert "\t" in delimiters
    
    def test_get_supported_encodings(self):
        """Retorna codificaciones soportadas."""
        encodings = get_supported_encodings()
        
        assert "utf-8" in encodings
        assert "latin-1" in encodings


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Propiedades Matemáticas (Hypothesis)
# ═══════════════════════════════════════════════════════════════════════════

class TestMathematicalProperties:
    """Pruebas de propiedades matemáticas con Hypothesis."""
    
    @given(st.lists(
        st.floats(min_value=0.01, max_value=1.0),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_entropy_non_negative(self, probabilities: List[float]):
        """Entropía siempre >= 0."""
        # Normalizar
        total = sum(probabilities)
        if total > 0:
            normalized = [p / total for p in probabilities]
            entropy = compute_shannon_entropy(normalized)
            assert entropy >= 0
    
    @given(st.integers(min_value=0, max_value=100),
           st.integers(min_value=0, max_value=100),
           st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_betti_euler_formula(self, b0: int, b1: int, b2: int):
        """χ = β₀ - β₁ + β₂ siempre se cumple."""
        betti = BettiNumbers(beta_0=b0, beta_1=b1, beta_2=b2)
        assert betti.euler_characteristic == b0 - b1 + b2
    
    @given(st.floats(min_value=0.0, max_value=10.0),
           st.floats(min_value=0.0, max_value=10.0))
    @settings(max_examples=50)
    def test_persistence_interval_invariants(self, birth: float, delta: float):
        """Invariantes de PersistenceInterval."""
        death = birth + delta
        iv = PersistenceInterval(birth=birth, death=death)
        
        assert math.isclose(iv.persistence, delta, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(iv.midpoint, (birth + death) / 2, rel_tol=1e-9, abs_tol=1e-9)
        assert not iv.is_essential


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Casos Edge
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Pruebas de casos límite."""
    
    def test_very_small_probabilities(self):
        """Probabilidades muy pequeñas no causan overflow."""
        tiny_probs = [1e-100, 1e-100, 1e-100]
        # No debe lanzar excepción
        entropy = compute_shannon_entropy(tiny_probs)
        assert entropy >= 0
    
    def test_single_stratum_registry(self, mic_registry: MICRegistry):
        """Registro con un solo estrato funciona."""
        mic_registry.register_vector("only", Stratum.PHYSICS, lambda **kw: {"success": True})
        
        hierarchy = mic_registry.get_stratum_hierarchy()
        assert len(hierarchy["PHYSICS"]) == 1
        assert len(hierarchy["TACTICS"]) == 0
    
    def test_handler_with_exception(self, mic_registry: MICRegistry):
        """Handler que lanza excepción se maneja gracefully."""
        def failing_handler(**kwargs):
            raise RuntimeError("Intentional failure")
        
        mic_registry.register_vector("failing", Stratum.PHYSICS, failing_handler)
        
        result = mic_registry.project_intent(
            service_name="failing",
            payload={},
            context={},
        )
        
        assert result["success"] is False
        assert "RuntimeError" in result.get("error_type", "")
    
    def test_handler_wrong_signature(self, mic_registry: MICRegistry):
        """Handler con firma incorrecta se maneja."""
        def wrong_signature(required_arg):  # No acepta **kwargs
            return {"success": True}
        
        mic_registry.register_vector("wrong_sig", Stratum.PHYSICS, wrong_signature)
        
        result = mic_registry.project_intent(
            service_name="wrong_sig",
            payload={},  # No pasa required_arg
            context={},
        )
        
        assert result["success"] is False
        assert result["error_category"] == "handler_signature_error"
    
    def test_unicode_in_payload(self, mic_registry: MICRegistry):
        """Payload con Unicode funciona."""
        mic_registry.register_vector(
            "unicode_test",
            Stratum.PHYSICS,
            lambda **kw: {"success": True, **kw}
        )
        
        result = mic_registry.project_intent(
            service_name="unicode_test",
            payload={"名前": "値", "emoji": "🎉"},
            context={},
        )
        
        assert result["success"] is True
    
    def test_very_deep_hierarchy(self, mic_registry: MICRegistry):
        """Todas las dependencias de WISDOM se validan."""
        mic_registry.register_vector(
            "wisdom_service",
            Stratum.WISDOM,
            lambda **kw: {"success": True}
        )
        
        # Sin validación de prerrequisitos
        result_fail = mic_registry.project_intent(
            service_name="wisdom_service",
            payload={},
            context={"validated_strata": set()},
        )
        assert result_fail["success"] is False
        
        # Con todos los prerrequisitos
        result_success = mic_registry.project_intent(
            service_name="wisdom_service",
            payload={},
            context={"validated_strata": {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.OMEGA}},
        )
        assert result_success["success"] is True


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Rendimiento (marcados como slow)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPerformance:
    """Pruebas de rendimiento."""
    
    def test_cache_performance_under_load(self):
        """Cache mantiene rendimiento bajo carga."""
        cache: TTLCache[int] = TTLCache(max_size=1000)
        
        start = time.perf_counter()
        for i in range(10000):
            cache.set(f"key_{i}", i)
        write_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for i in range(10000):
            cache.get(f"key_{i % 1000}")  # Solo los últimos 1000 existen
        read_time = time.perf_counter() - start
        
        assert write_time < 1.0  # < 1s para 10k escrituras
        assert read_time < 0.5   # < 0.5s para 10k lecturas
    
    def test_large_mic_registry(self):
        """Registro grande mantiene rendimiento."""
        mic = MICRegistry()
        
        start = time.perf_counter()
        for i in range(100):
            mic.register_vector(
                f"service_{i}",
                Stratum.PHYSICS,
                lambda **kw: {"success": True}
            )
        register_time = time.perf_counter() - start
        
        assert register_time < 1.0  # < 1s para 100 registros
        assert mic.dimension == 100
    
    def test_concurrent_projections(self, mic_with_vectors: MICRegistry):
        """Proyecciones concurrentes son thread-safe."""
        errors = []
        results = []
        
        def project():
            try:
                result = mic_with_vectors.project_intent(
                    service_name="test_physics",
                    payload={"thread_id": threading.current_thread().name},
                    context={},
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(project) for _ in range(100)]
            for future in as_completed(futures):
                future.result()
        
        assert len(errors) == 0
        assert len(results) == 100
        assert all(r["success"] for r in results)
    
    def test_large_file_topological_analysis(self, temp_large_csv: Path):
        """Análisis topológico de archivo grande es eficiente."""
        start = time.perf_counter()
        summary = analyze_topological_features(temp_large_csv)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0  # < 2s para ~1000 líneas
        assert summary.intrinsic_dimension >= 1
    
    def test_cycle_detection_performance(self):
        """Detección de ciclos escala bien."""
        lines = [f"line_{i % 10}" for i in range(1000)]
        
        start = time.perf_counter()
        cycles = detect_cyclic_patterns(lines)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0  # < 1s para 1000 líneas


# ═══════════════════════════════════════════════════════════════════════════
# TESTS: Thread Safety
# ═══════════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Pruebas de seguridad en hilos."""
    
    def test_mic_registry_concurrent_registration(self):
        """Registro concurrente es thread-safe."""
        mic = MICRegistry()
        errors = []
        
        def register_service(n: int):
            try:
                mic.register_vector(
                    f"service_{n}",
                    Stratum.PHYSICS,
                    lambda **kw: {"success": True, "n": n}
                )
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=register_service, args=(i,))
            for i in range(50)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert mic.dimension == 50
    
    def test_metrics_concurrent_updates(self):
        """Métricas soportan actualizaciones concurrentes."""
        metrics = MICMetrics()
        errors = []
        
        def update_metrics():
            try:
                for _ in range(100):
                    metrics.record_projection(Stratum.PHYSICS)
                    metrics.record_error("test")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=update_metrics) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert metrics.projections == 1000
        assert metrics.errors == 1000


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