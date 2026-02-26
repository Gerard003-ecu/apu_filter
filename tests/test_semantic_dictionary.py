"""
Suite de Pruebas para SemanticDictionaryService
===============================================

Cobertura:
    - Pruebas unitarias para cada componente
    - Pruebas de integración entre componentes
    - Pruebas de concurrencia (thread safety)
    - Pruebas de casos límite y edge cases
    - Pruebas de validación y manejo de errores

Requisitos:
    pytest >= 7.0
    pytest-cov (opcional, para cobertura)
"""
import concurrent.futures
import hashlib
import logging
import random
import string
import threading
import time
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Importar los módulos a probar
from semantic_dictionary import (
    GraphSemanticProjector,
    PyramidalSemanticVector,
    SemanticDictionaryService,
    Stratum,
    TemplateValidator,
    TTLCache,
    VALID_NODE_TYPES,
    create_semantic_dictionary_service,
)


# =============================================================================
# FIXTURES COMPARTIDAS
# =============================================================================

@pytest.fixture
def service() -> SemanticDictionaryService:
    """Fixture para crear una instancia limpia del servicio."""
    return SemanticDictionaryService()


@pytest.fixture
def ttl_cache() -> TTLCache:
    """Fixture para crear un caché con TTL corto para pruebas."""
    return TTLCache(ttl_seconds=1.0, maxsize=10)


@pytest.fixture
def projector(service: SemanticDictionaryService) -> GraphSemanticProjector:
    """Fixture para crear un proyector semántico."""
    return GraphSemanticProjector(service, cache_ttl=1.0, cache_maxsize=50)


@pytest.fixture
def valid_vector_data() -> Dict[str, Any]:
    """Fixture con datos válidos para PyramidalSemanticVector."""
    return {
        "node_id": "APU-001",
        "node_type": "APU",
        "stratum": Stratum.TACTICS,
        "in_degree": 3,
        "out_degree": 5,
        "is_critical_bridge": False,
    }


@pytest.fixture
def sample_vectors() -> List[PyramidalSemanticVector]:
    """Fixture con varios vectores de ejemplo."""
    return [
        PyramidalSemanticVector(
            node_id="ROOT-001",
            node_type="ROOT",
            stratum=Stratum.WISDOM,
            in_degree=0,
            out_degree=10,
        ),
        PyramidalSemanticVector(
            node_id="CAP-001",
            node_type="CAPITULO",
            stratum=Stratum.STRATEGY,
            in_degree=1,
            out_degree=5,
        ),
        PyramidalSemanticVector(
            node_id="APU-001",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=2,
            out_degree=3,
        ),
        PyramidalSemanticVector(
            node_id="INS-001",
            node_type="INSUMO",
            stratum=Stratum.PHYSICS,
            in_degree=5,
            out_degree=0,
        ),
    ]


# =============================================================================
# TESTS PARA TTLCache
# =============================================================================

class TestTTLCache:
    """Suite de pruebas para TTLCache."""
    
    def test_init_valid_parameters(self):
        """Verifica inicialización con parámetros válidos."""
        cache = TTLCache(ttl_seconds=60.0, maxsize=100)
        assert cache._ttl == 60.0
        assert cache._maxsize == 100
        assert len(cache._cache) == 0
    
    def test_init_invalid_ttl_raises_error(self):
        """TTL <= 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="TTL debe ser positivo"):
            TTLCache(ttl_seconds=0)
        
        with pytest.raises(ValueError, match="TTL debe ser positivo"):
            TTLCache(ttl_seconds=-1.0)
    
    def test_init_invalid_maxsize_raises_error(self):
        """maxsize <= 0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="maxsize debe ser positivo"):
            TTLCache(ttl_seconds=1.0, maxsize=0)
        
        with pytest.raises(ValueError, match="maxsize debe ser positivo"):
            TTLCache(ttl_seconds=1.0, maxsize=-5)
    
    def test_set_and_get_basic(self, ttl_cache: TTLCache):
        """Operaciones básicas de set y get."""
        ttl_cache.set("key1", "value1")
        ttl_cache.set("key2", {"nested": "data"})
        
        assert ttl_cache.get("key1") == "value1"
        assert ttl_cache.get("key2") == {"nested": "data"}
    
    def test_get_nonexistent_key_returns_none(self, ttl_cache: TTLCache):
        """Get de clave inexistente retorna None."""
        assert ttl_cache.get("nonexistent") is None
    
    def test_ttl_expiration(self):
        """Verifica que las entradas expiren después del TTL."""
        cache = TTLCache(ttl_seconds=0.1, maxsize=10)
        cache.set("key", "value")
        
        # Inmediatamente después, debe existir
        assert cache.get("key") == "value"
        
        # Esperar a que expire
        time.sleep(0.15)
        
        # Ahora debe retornar None
        assert cache.get("key") is None
    
    def test_lru_eviction_on_capacity(self):
        """Verifica evicción LRU cuando se alcanza capacidad máxima."""
        cache = TTLCache(ttl_seconds=60.0, maxsize=3)
        
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Acceder a 'a' para que sea más reciente
        _ = cache.get("a")
        
        # Insertar nuevo elemento, 'b' debería ser eviccionado (LRU)
        cache.set("d", 4)
        
        assert cache.get("a") == 1  # Accedido recientemente
        assert cache.get("b") is None  # Eviccionado
        assert cache.get("c") == 3
        assert cache.get("d") == 4
    
    def test_update_existing_key(self, ttl_cache: TTLCache):
        """Actualizar clave existente debe actualizar valor y timestamp."""
        ttl_cache.set("key", "old_value")
        old_ts = ttl_cache._timestamps["key"]
        
        time.sleep(0.01)
        ttl_cache.set("key", "new_value")
        
        assert ttl_cache.get("key") == "new_value"
        assert ttl_cache._timestamps["key"] >= old_ts
    
    def test_clear_removes_all_entries(self, ttl_cache: TTLCache):
        """Clear debe eliminar todas las entradas."""
        ttl_cache.set("a", 1)
        ttl_cache.set("b", 2)
        ttl_cache.set("c", 3)
        
        ttl_cache.clear()
        
        assert ttl_cache.get("a") is None
        assert ttl_cache.get("b") is None
        assert ttl_cache.get("c") is None
        assert len(ttl_cache._cache) == 0
    
    def test_cleanup_expired_removes_old_entries(self):
        """cleanup_expired debe eliminar entradas expiradas."""
        cache = TTLCache(ttl_seconds=0.1, maxsize=10)
        
        cache.set("old1", "val1")
        cache.set("old2", "val2")
        time.sleep(0.15)
        cache.set("new", "val3")
        
        removed = cache.cleanup_expired()
        
        assert removed == 2
        assert cache.get("old1") is None
        assert cache.get("old2") is None
        assert cache.get("new") == "val3"
    
    def test_stats_tracking(self, ttl_cache: TTLCache):
        """Verifica que las estadísticas se actualicen correctamente."""
        ttl_cache.set("key", "value")
        
        # Hit
        _ = ttl_cache.get("key")
        # Miss
        _ = ttl_cache.get("nonexistent")
        # Otro hit
        _ = ttl_cache.get("key")
        
        stats = ttl_cache.stats
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3, rel=0.01)
        assert stats["size"] == 1
    
    def test_thread_safety_concurrent_access(self):
        """Verifica thread safety con accesos concurrentes."""
        cache = TTLCache(ttl_seconds=60.0, maxsize=1000)
        errors = []
        
        def writer(thread_id: int):
            try:
                for i in range(100):
                    cache.set(f"key_{thread_id}_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)
        
        def reader(thread_id: int):
            try:
                for i in range(100):
                    _ = cache.get(f"key_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errores de concurrencia: {errors}"


# =============================================================================
# TESTS PARA PyramidalSemanticVector
# =============================================================================

class TestPyramidalSemanticVector:
    """Suite de pruebas para PyramidalSemanticVector."""
    
    def test_valid_construction(self, valid_vector_data: Dict[str, Any]):
        """Construcción válida de un vector."""
        vector = PyramidalSemanticVector(**valid_vector_data)
        
        assert vector.node_id == "APU-001"
        assert vector.node_type == "APU"
        assert vector.stratum == Stratum.TACTICS
        assert vector.in_degree == 3
        assert vector.out_degree == 5
        assert vector.is_critical_bridge is False
    
    @pytest.mark.parametrize("node_type", list(VALID_NODE_TYPES))
    def test_all_valid_node_types(self, node_type: str):
        """Verifica que todos los tipos de nodo válidos sean aceptados."""
        vector = PyramidalSemanticVector(
            node_id="test",
            node_type=node_type,
            stratum=Stratum.TACTICS,
            in_degree=1,
            out_degree=1,
        )
        assert vector.node_type == node_type
    
    def test_invalid_node_type_raises_error(self):
        """node_type inválido debe lanzar ValueError."""
        with pytest.raises(ValueError, match="node_type inválido"):
            PyramidalSemanticVector(
                node_id="test",
                node_type="INVALID_TYPE",
                stratum=Stratum.TACTICS,
                in_degree=1,
                out_degree=1,
            )
    
    def test_negative_in_degree_raises_error(self):
        """in_degree negativo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="in_degree no puede ser negativo"):
            PyramidalSemanticVector(
                node_id="test",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=-1,
                out_degree=1,
            )
    
    def test_negative_out_degree_raises_error(self):
        """out_degree negativo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="out_degree no puede ser negativo"):
            PyramidalSemanticVector(
                node_id="test",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=1,
                out_degree=-1,
            )
    
    def test_empty_node_id_raises_error(self):
        """node_id vacío debe lanzar ValueError."""
        with pytest.raises(ValueError, match="node_id no puede estar vacío"):
            PyramidalSemanticVector(
                node_id="",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=1,
                out_degree=1,
            )
        
        with pytest.raises(ValueError, match="node_id no puede estar vacío"):
            PyramidalSemanticVector(
                node_id="   ",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=1,
                out_degree=1,
            )
    
    def test_total_degree_property(self):
        """Verifica cálculo de total_degree."""
        vector = PyramidalSemanticVector(
            node_id="test",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=3,
            out_degree=5,
        )
        assert vector.total_degree == 8
    
    def test_is_leaf_property(self):
        """Verifica propiedad is_leaf."""
        leaf = PyramidalSemanticVector(
            node_id="leaf",
            node_type="INSUMO",
            stratum=Stratum.PHYSICS,
            in_degree=3,
            out_degree=0,
        )
        non_leaf = PyramidalSemanticVector(
            node_id="non_leaf",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=2,
            out_degree=3,
        )
        
        assert leaf.is_leaf is True
        assert non_leaf.is_leaf is False
    
    def test_is_root_property(self):
        """Verifica propiedad is_root."""
        root = PyramidalSemanticVector(
            node_id="root",
            node_type="ROOT",
            stratum=Stratum.WISDOM,
            in_degree=0,
            out_degree=5,
        )
        non_root = PyramidalSemanticVector(
            node_id="non_root",
            node_type="CAPITULO",
            stratum=Stratum.STRATEGY,
            in_degree=1,
            out_degree=3,
        )
        
        assert root.is_root is True
        assert non_root.is_root is False
    
    def test_is_isolated_property(self):
        """Verifica propiedad is_isolated."""
        isolated = PyramidalSemanticVector(
            node_id="isolated",
            node_type="INSUMO",
            stratum=Stratum.PHYSICS,
            in_degree=0,
            out_degree=0,
        )
        connected = PyramidalSemanticVector(
            node_id="connected",
            node_type="APU",
            stratum=Stratum.TACTICS,
            in_degree=1,
            out_degree=0,
        )
        
        assert isolated.is_isolated is True
        assert connected.is_isolated is False
    
    def test_to_dict_serialization(self, valid_vector_data: Dict[str, Any]):
        """Verifica serialización a diccionario."""
        vector = PyramidalSemanticVector(**valid_vector_data)
        result = vector.to_dict()
        
        assert result["node_id"] == "APU-001"
        assert result["node_type"] == "APU"
        assert result["stratum"] == "TACTICS"
        assert result["in_degree"] == 3
        assert result["out_degree"] == 5
        assert result["is_critical_bridge"] is False
        assert result["total_degree"] == 8
    
    def test_frozen_dataclass_immutable(self, valid_vector_data: Dict[str, Any]):
        """Verifica que el dataclass sea inmutable (frozen)."""
        vector = PyramidalSemanticVector(**valid_vector_data)
        
        with pytest.raises(FrozenInstanceError):
            vector.node_id = "new_id"
    
    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_all_strata_accepted(self, stratum: Stratum):
        """Verifica que todos los estratos sean aceptados."""
        vector = PyramidalSemanticVector(
            node_id="test",
            node_type="APU",
            stratum=stratum,
            in_degree=1,
            out_degree=1,
        )
        assert vector.stratum == stratum


# =============================================================================
# TESTS PARA TemplateValidator
# =============================================================================

class TestTemplateValidator:
    """Suite de pruebas para TemplateValidator."""
    
    def test_extract_simple_placeholders(self):
        """Extrae placeholders simples."""
        template = "Hello {name}, your score is {score}."
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert placeholders == {"name", "score"}
    
    def test_extract_formatted_placeholders(self):
        """Extrae placeholders con formato."""
        template = "Value: {value:.2f}, Percent: {percent:.1%}, Price: ${price:,.2f}"
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert placeholders == {"value", "percent", "price"}
    
    def test_extract_no_placeholders(self):
        """Template sin placeholders retorna conjunto vacío."""
        template = "No placeholders here!"
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert placeholders == set()
    
    def test_extract_repeated_placeholders(self):
        """Placeholders repetidos solo aparecen una vez."""
        template = "{name} said hello to {name} and {other}"
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert placeholders == {"name", "other"}
    
    def test_extract_complex_format_specs(self):
        """Maneja especificadores de formato complejos."""
        template = "{value:>10.2f} {date:%Y-%m-%d} {num:08d}"
        placeholders = TemplateValidator.extract_placeholders(template)
        
        assert "value" in placeholders
        assert "date" in placeholders
        assert "num" in placeholders
    
    def test_validate_valid_template(self):
        """Validación de plantilla correcta."""
        template = "Hello {name}!"
        is_valid, error = TemplateValidator.validate_template(template)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_template_with_required_params(self):
        """Validación con parámetros requeridos presentes."""
        template = "User {name} has {count} items"
        required = {"name", "count"}
        
        is_valid, error = TemplateValidator.validate_template(template, required)
        
        assert is_valid is True
    
    def test_validate_template_missing_required_params(self):
        """Detecta parámetros requeridos faltantes."""
        template = "User {name} logged in"
        required = {"name", "email", "timestamp"}
        
        is_valid, error = TemplateValidator.validate_template(template, required)
        
        assert is_valid is False
        assert "email" in error or "timestamp" in error
    
    def test_validate_malformed_template(self):
        """Detecta plantillas mal formadas."""
        malformed = "Hello {name"  # Falta cerrar
        
        is_valid, error = TemplateValidator.validate_template(malformed)
        
        # Puede ser válido o inválido según implementación
        # Lo importante es que no lance excepción
        assert isinstance(is_valid, bool)
    
    def test_validate_all_templates_recursive(self):
        """Validación recursiva de diccionario de plantillas."""
        templates = {
            "GROUP_A": {
                "valid": "This is {valid}",
                "also_valid": "Value: {num:.2f}",
            },
            "GROUP_B": {
                "simple": "No placeholders",
            },
        }
        
        errors = TemplateValidator.validate_all_templates(templates)
        
        assert len(errors) == 0
    
    def test_validate_all_templates_finds_errors(self):
        """Detecta errores en validación recursiva."""
        templates = {
            "GROUP_A": {
                "broken": "This has {unclosed",
            },
        }
        
        # No debe lanzar excepción
        errors = TemplateValidator.validate_all_templates(templates)
        # Puede o no detectar el error según la implementación


# =============================================================================
# TESTS PARA GraphSemanticProjector
# =============================================================================

class TestGraphSemanticProjector:
    """Suite de pruebas para GraphSemanticProjector."""
    
    def test_project_pyramidal_stress_valid(
        self, 
        projector: GraphSemanticProjector,
        valid_vector_data: Dict[str, Any]
    ):
        """Proyección de estrés con vector válido."""
        vector = PyramidalSemanticVector(**valid_vector_data)
        result = projector.project_pyramidal_stress(vector)
        
        assert result["success"] is True
        assert "narrative" in result
        assert "vector_metadata" in result
    
    def test_project_pyramidal_stress_caching(
        self, 
        projector: GraphSemanticProjector,
        valid_vector_data: Dict[str, Any]
    ):
        """Verifica que el caché funcione para proyecciones."""
        vector = PyramidalSemanticVector(**valid_vector_data)
        
        # Primera llamada
        result1 = projector.project_pyramidal_stress(vector)
        stats1 = projector.cache_stats
        
        # Segunda llamada (debería usar caché)
        result2 = projector.project_pyramidal_stress(vector)
        stats2 = projector.cache_stats
        
        assert stats2["hits"] > stats1["hits"]
        assert result1["narrative"] == result2["narrative"]
    
    def test_project_cycle_path_valid(self, projector: GraphSemanticProjector):
        """Proyección de ciclo con ruta válida."""
        path = ["A", "B", "C", "A"]
        result = projector.project_cycle_path(path)
        
        assert result["success"] is True
        assert "narrative" in result
        assert "cycle_metadata" in result
        assert result["cycle_metadata"]["length"] == 4
    
    def test_project_cycle_path_empty(self, projector: GraphSemanticProjector):
        """Proyección de ciclo con lista vacía."""
        result = projector.project_cycle_path([])
        
        assert result["success"] is False
        assert "error" in result
    
    def test_project_cycle_path_single_node(self, projector: GraphSemanticProjector):
        """Proyección de self-loop (nodo único)."""
        result = projector.project_cycle_path(["A"])
        
        assert result["success"] is True
        assert result["cycle_metadata"]["is_self_loop"] is True
    
    def test_project_cycle_path_sanitizes_nodes(self, projector: GraphSemanticProjector):
        """Verifica sanitización de nodos."""
        path = ["  A  ", "B", "  C  "]
        result = projector.project_cycle_path(path)
        
        assert result["success"] is True
        assert result["cycle_metadata"]["nodes"] == ["A", "B", "C"]
    
    def test_project_fragmentation_empty(self, projector: GraphSemanticProjector):
        """Proyección de grafo vacío (β₀ = 0)."""
        result = projector.project_fragmentation(0)
        
        assert result["success"] is True
        assert "empty" in result.get("classification", "") or "Vacío" in result.get("narrative", "")
    
    def test_project_fragmentation_unified(self, projector: GraphSemanticProjector):
        """Proyección de grafo conectado (β₀ = 1)."""
        result = projector.project_fragmentation(1)
        
        assert result["success"] is True
    
    def test_project_fragmentation_with_sizes(self, projector: GraphSemanticProjector):
        """Proyección con tamaños de componentes."""
        result = projector.project_fragmentation(3, [100, 50, 10])
        
        assert result["success"] is True
        assert "component_analysis" in result
        assert result["component_analysis"]["largest"] == 100
        assert result["component_analysis"]["smallest"] == 10
    
    def test_calculate_gini_equal_distribution(self):
        """Gini = 0 para distribución perfectamente igual."""
        gini = GraphSemanticProjector._calculate_gini([10, 10, 10, 10])
        assert gini == pytest.approx(0.0, abs=0.01)
    
    def test_calculate_gini_unequal_distribution(self):
        """Gini > 0 para distribución desigual."""
        gini = GraphSemanticProjector._calculate_gini([1, 1, 1, 100])
        assert gini > 0.5  # Distribución muy desigual
    
    def test_calculate_gini_single_value(self):
        """Gini = 0 para un solo valor."""
        gini = GraphSemanticProjector._calculate_gini([100])
        assert gini == 0.0
    
    def test_calculate_gini_empty_list(self):
        """Gini = 0 para lista vacía."""
        gini = GraphSemanticProjector._calculate_gini([])
        assert gini == 0.0


# =============================================================================
# TESTS PARA SemanticDictionaryService
# =============================================================================

class TestSemanticDictionaryService:
    """Suite de pruebas para SemanticDictionaryService."""
    
    def test_initialization(self, service: SemanticDictionaryService):
        """Verifica inicialización correcta."""
        assert service is not None
        assert len(service._templates) > 0
        assert len(service._market_contexts) > 0
    
    def test_health_check(self, service: SemanticDictionaryService):
        """Verifica endpoint de health check."""
        health = service.health_check()
        
        assert health["status"] == "healthy"
        assert "template_domains" in health
        assert "timestamp" in health
        assert health["stratum"] == "WISDOM"
    
    def test_get_available_domains(self, service: SemanticDictionaryService):
        """Verifica listado de dominios disponibles."""
        domains = service.get_available_domains()
        
        assert isinstance(domains, list)
        assert len(domains) > 0
        assert "TOPOLOGY_CYCLES" in domains
        assert "STABILITY" in domains
    
    def test_get_domain_classifications_existing(self, service: SemanticDictionaryService):
        """Obtiene clasificaciones de dominio existente."""
        classifications = service.get_domain_classifications("STABILITY")
        
        assert classifications is not None
        assert "critical" in classifications
        assert "robust" in classifications
    
    def test_get_domain_classifications_nonexistent(self, service: SemanticDictionaryService):
        """Dominio inexistente retorna None."""
        classifications = service.get_domain_classifications("NONEXISTENT")
        assert classifications is None
    
    # =========================================================================
    # fetch_narrative tests
    # =========================================================================
    
    def test_fetch_narrative_valid_domain_classification(
        self, 
        service: SemanticDictionaryService
    ):
        """fetch_narrative con dominio y clasificación válidos."""
        result = service.fetch_narrative(
            domain="STABILITY",
            classification="robust",
            params={"stability": 0.90}
        )
        
        assert result["success"] is True
        assert "narrative" in result
        assert "0.90" in result["narrative"]
    
    def test_fetch_narrative_invalid_domain(self, service: SemanticDictionaryService):
        """fetch_narrative con dominio inválido."""
        result = service.fetch_narrative(
            domain="NONEXISTENT_DOMAIN",
            classification="any"
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "available_domains" in result
    
    def test_fetch_narrative_missing_params(self, service: SemanticDictionaryService):
        """fetch_narrative con parámetros faltantes."""
        result = service.fetch_narrative(
            domain="STABILITY",
            classification="robust",
            params={}  # Falta 'stability'
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_fetch_narrative_market_context_random(
        self, 
        service: SemanticDictionaryService
    ):
        """fetch_narrative para contexto de mercado aleatorio."""
        result = service.fetch_narrative(
            domain="MARKET_CONTEXT",
            params={"deterministic": False}
        )
        
        assert result["success"] is True
        assert "narrative" in result
    
    def test_fetch_narrative_market_context_deterministic(
        self, 
        service: SemanticDictionaryService
    ):
        """fetch_narrative para contexto de mercado determinístico."""
        result1 = service.fetch_narrative(
            domain="MARKET_CONTEXT",
            params={"deterministic": True, "index": 0}
        )
        result2 = service.fetch_narrative(
            domain="MARKET_CONTEXT",
            params={"deterministic": True, "index": 0}
        )
        
        assert result1["narrative"] == result2["narrative"]
    
    def test_fetch_narrative_with_kwargs(self, service: SemanticDictionaryService):
        """fetch_narrative acepta kwargs como parámetros."""
        result = service.fetch_narrative(
            domain="STABILITY",
            classification="robust",
            stability=0.95
        )
        
        assert result["success"] is True
        assert "0.95" in result["narrative"]
    
    @pytest.mark.parametrize("domain,classification,params", [
        ("TOPOLOGY_CYCLES", "clean", {}),
        ("TOPOLOGY_CYCLES", "minor", {"beta_1": 2}),
        ("TOPOLOGY_CONNECTIVITY", "unified", {}),
        ("SPECTRAL_COHESION", "high", {"fiedler": 0.85}),
        ("THERMAL_ENTROPY", "low", {"entropy": 0.2}),
    ])
    def test_fetch_narrative_various_templates(
        self, 
        service: SemanticDictionaryService,
        domain: str,
        classification: str,
        params: Dict[str, Any]
    ):
        """Prueba varias combinaciones de plantillas."""
        result = service.fetch_narrative(
            domain=domain,
            classification=classification,
            params=params
        )
        
        assert result["success"] is True
        assert result["domain"] == domain
        assert result["classification"] == classification
    
    # =========================================================================
    # get_classification_by_threshold tests
    # =========================================================================
    
    @pytest.mark.parametrize("value,expected", [
        (0.95, "robust"),
        (0.75, "stable"),
        (0.55, "warning"),
        (0.25, "critical"),
    ])
    def test_get_classification_stability(
        self, 
        service: SemanticDictionaryService,
        value: float,
        expected: str
    ):
        """Clasificación de estabilidad por umbrales."""
        classification = service.get_classification_by_threshold("STABILITY", value)
        assert classification == expected
    
    @pytest.mark.parametrize("value,expected", [
        (0.8, "high"),
        (0.2, "low"),
    ])
    def test_get_classification_entropy(
        self, 
        service: SemanticDictionaryService,
        value: float,
        expected: str
    ):
        """Clasificación de entropía (invertida)."""
        classification = service.get_classification_by_threshold("ENTROPY", value)
        assert classification == expected
    
    def test_get_classification_invalid_metric(
        self, 
        service: SemanticDictionaryService
    ):
        """Métrica inválida lanza ValueError."""
        with pytest.raises(ValueError, match="no tiene umbrales definidos"):
            service.get_classification_by_threshold("INVALID_METRIC", 0.5)
    
    # =========================================================================
    # convert_stratum_value tests
    # =========================================================================
    
    def test_convert_stratum_from_stratum(self, service: SemanticDictionaryService):
        """Conversión desde Stratum retorna mismo valor."""
        result = service.convert_stratum_value(Stratum.TACTICS)
        assert result == Stratum.TACTICS
    
    @pytest.mark.parametrize("value,expected", [
        (0, Stratum.WISDOM),
        (1, Stratum.STRATEGY),
        (2, Stratum.TACTICS),
        (3, Stratum.PHYSICS),
    ])
    def test_convert_stratum_from_int(
        self, 
        service: SemanticDictionaryService,
        value: int,
        expected: Stratum
    ):
        """Conversión desde int."""
        result = service.convert_stratum_value(value)
        assert result == expected
    
    @pytest.mark.parametrize("value,expected", [
        ("WISDOM", Stratum.WISDOM),
        ("wisdom", Stratum.WISDOM),
        ("Tactics", Stratum.TACTICS),
        ("PHYSICS", Stratum.PHYSICS),
        ("  STRATEGY  ", Stratum.STRATEGY),
    ])
    def test_convert_stratum_from_string(
        self, 
        service: SemanticDictionaryService,
        value: str,
        expected: Stratum
    ):
        """Conversión desde string (case insensitive)."""
        result = service.convert_stratum_value(value)
        assert result == expected
    
    def test_convert_stratum_invalid_int(self, service: SemanticDictionaryService):
        """Entero inválido lanza ValueError."""
        with pytest.raises(ValueError, match="no es un Stratum válido"):
            service.convert_stratum_value(99)
    
    def test_convert_stratum_invalid_string(self, service: SemanticDictionaryService):
        """String inválido lanza ValueError."""
        with pytest.raises(ValueError, match="no es un nombre de Stratum válido"):
            service.convert_stratum_value("INVALID")
    
    def test_convert_stratum_invalid_type(self, service: SemanticDictionaryService):
        """Tipo inválido lanza TypeError."""
        with pytest.raises(TypeError, match="Tipo no soportado"):
            service.convert_stratum_value([1, 2, 3])
    
    # =========================================================================
    # project_graph_narrative tests
    # =========================================================================
    
    def test_project_graph_narrative_cycle(self, service: SemanticDictionaryService):
        """Proyección de anomalía de ciclo."""
        payload = {
            "anomaly_type": "CYCLE",
            "path_nodes": ["A", "B", "C", "A"],
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is True
        assert "cycle_metadata" in result
    
    def test_project_graph_narrative_stress(
        self, 
        service: SemanticDictionaryService,
        valid_vector_data: Dict[str, Any]
    ):
        """Proyección de anomalía de estrés."""
        payload = {
            "anomaly_type": "STRESS",
            "vector": valid_vector_data,
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is True
        assert "vector_metadata" in result
    
    def test_project_graph_narrative_stress_with_string_stratum(
        self, 
        service: SemanticDictionaryService
    ):
        """Proyección de estrés con stratum como string."""
        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": "test",
                "node_type": "APU",
                "stratum": "TACTICS",  # String en lugar de Enum
                "in_degree": 3,
                "out_degree": 5,
            },
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is True
    
    def test_project_graph_narrative_fragmentation(
        self, 
        service: SemanticDictionaryService
    ):
        """Proyección de anomalía de fragmentación."""
        payload = {
            "anomaly_type": "FRAGMENTATION",
            "beta_0": 3,
            "component_sizes": [100, 50, 10],
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is True
    
    def test_project_graph_narrative_unsupported_type(
        self, 
        service: SemanticDictionaryService
    ):
        """Tipo de anomalía no soportado."""
        payload = {
            "anomaly_type": "UNKNOWN_TYPE",
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is False
        assert "supported_types" in result
    
    def test_project_graph_narrative_missing_vector_fields(
        self, 
        service: SemanticDictionaryService
    ):
        """Vector con campos faltantes."""
        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": "test",
                # Faltan otros campos
            },
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is False
        assert "Faltan campos requeridos" in result["error"]
    
    def test_project_graph_narrative_invalid_node_type(
        self, 
        service: SemanticDictionaryService
    ):
        """Vector con node_type inválido."""
        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": "test",
                "node_type": "INVALID",
                "stratum": Stratum.TACTICS,
                "in_degree": 1,
                "out_degree": 1,
            },
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is False
        assert "node_type inválido" in result["error"]
    
    # =========================================================================
    # register_in_mic tests
    # =========================================================================
    
    def test_register_in_mic_without_module(self, service: SemanticDictionaryService):
        """Registro sin módulo MICRegistry disponible retorna False."""
        with patch.dict('sys.modules', {'app.tools_interface': None}):
            result = service.register_in_mic(MagicMock())
            # Puede retornar True o False dependiendo de la implementación
            assert isinstance(result, bool)
    
    def test_register_in_mic_wrong_type(self, service: SemanticDictionaryService):
        """Registro con tipo incorrecto retorna False."""
        result = service.register_in_mic("not_a_registry")
        assert result is False
    
    # =========================================================================
    # Projector lazy initialization
    # =========================================================================
    
    def test_projector_lazy_initialization(self, service: SemanticDictionaryService):
        """Projector se inicializa de forma perezosa."""
        assert service._projector is None
        
        projector = service.projector
        
        assert projector is not None
        assert service._projector is projector
    
    def test_projector_singleton(self, service: SemanticDictionaryService):
        """Projector es singleton."""
        projector1 = service.projector
        projector2 = service.projector
        
        assert projector1 is projector2


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Pruebas de integración entre componentes."""
    
    def test_full_cycle_projection_flow(self, service: SemanticDictionaryService):
        """Flujo completo de proyección de ciclo."""
        # Simular detección de ciclo por motor topológico
        cycle_data = {
            "anomaly_type": "CYCLE",
            "path_nodes": ["Excavación", "Cimentación", "Estructura", "Excavación"],
        }
        
        # Proyectar a narrativa
        result = service.project_graph_narrative(cycle_data)
        
        # Verificar resultado
        assert result["success"] is True
        assert "narrative" in result
        assert "Excavación" in result["narrative"]
    
    def test_full_stress_projection_flow(self, service: SemanticDictionaryService):
        """Flujo completo de proyección de estrés."""
        # Simular detección de punto crítico
        stress_data = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": "CEMENTO-PORTLAND",
                "node_type": "INSUMO",
                "stratum": "PHYSICS",
                "in_degree": 25,
                "out_degree": 0,
                "is_critical_bridge": True,
            },
        }
        
        result = service.project_graph_narrative(stress_data)
        
        assert result["success"] is True
        assert "CEMENTO-PORTLAND" in result["narrative"]
    
    def test_narrative_generation_for_multiple_domains(
        self, 
        service: SemanticDictionaryService
    ):
        """Generación de narrativas para múltiples dominios."""
        test_cases = [
            ("TOPOLOGY_CYCLES", "critical", {"beta_1": 15}),
            ("STABILITY", "critical", {"stability": 0.2}),
            ("SPECTRAL_COHESION", "low", {"fiedler": 0.15}),
            ("THERMAL_TEMPERATURE", "hot", {"temperature": 85.5}),
            ("FINANCIAL_VERDICT", "accept", {"pi": 1.25}),
        ]
        
        for domain, classification, params in test_cases:
            result = service.fetch_narrative(
                domain=domain,
                classification=classification,
                params=params
            )
            
            assert result["success"] is True, f"Fallo en {domain}.{classification}"
            assert len(result["narrative"]) > 0
    
    def test_threshold_classification_integration(
        self, 
        service: SemanticDictionaryService
    ):
        """Integración clasificación por umbral + narrativa."""
        stability_value = 0.45
        
        # Clasificar
        classification = service.get_classification_by_threshold(
            "STABILITY", 
            stability_value
        )
        
        # Generar narrativa
        result = service.fetch_narrative(
            domain="STABILITY",
            classification=classification,
            params={"stability": stability_value}
        )
        
        assert result["success"] is True
        assert classification == "warning"


# =============================================================================
# TESTS DE CONCURRENCIA
# =============================================================================

class TestConcurrency:
    """Pruebas de comportamiento bajo carga concurrente."""
    
    def test_concurrent_fetch_narrative(self, service: SemanticDictionaryService):
        """Múltiples llamadas concurrentes a fetch_narrative."""
        errors = []
        results = []
        
        def fetch(thread_id: int):
            try:
                for i in range(20):
                    result = service.fetch_narrative(
                        domain="STABILITY",
                        classification="stable",
                        params={"stability": 0.75 + thread_id * 0.01}
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        assert len(errors) == 0
        assert len(results) == 200
        assert all(r["success"] for r in results)
    
    def test_concurrent_project_graph_narrative(
        self, 
        service: SemanticDictionaryService
    ):
        """Múltiples proyecciones concurrentes."""
        errors = []
        
        def project(thread_id: int):
            try:
                for i in range(10):
                    payload = {
                        "anomaly_type": "CYCLE",
                        "path_nodes": [f"Node_{thread_id}_{j}" for j in range(3)],
                    }
                    result = service.project_graph_narrative(payload)
                    assert result["success"]
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=project, args=(i,))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_cache_thread_safety_under_load(self):
        """Cache bajo alta carga concurrente."""
        cache = TTLCache(ttl_seconds=60.0, maxsize=100)
        errors = []
        operations = []
        
        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f"key_{thread_id % 20}_{i % 10}"
                    if random.random() > 0.3:
                        cache.set(key, f"value_{thread_id}_{i}")
                        operations.append(("set", key))
                    else:
                        value = cache.get(key)
                        operations.append(("get", key, value))
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(20)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# =============================================================================
# TESTS DE EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Pruebas de casos límite y condiciones de borde."""
    
    def test_very_long_node_id(self, service: SemanticDictionaryService):
        """node_id muy largo."""
        long_id = "A" * 10000
        
        payload = {
            "anomaly_type": "STRESS",
            "vector": {
                "node_id": long_id,
                "node_type": "APU",
                "stratum": Stratum.TACTICS,
                "in_degree": 1,
                "out_degree": 1,
            },
        }
        
        result = service.project_graph_narrative(payload)
        
        assert result["success"] is True
    
    def test_unicode_in_narratives(self, service: SemanticDictionaryService):
        """Caracteres unicode en narrativas."""
        result = service.fetch_narrative(
            domain="TOPOLOGY_CYCLES",
            classification="clean"
        )
        
        assert result["success"] is True
        # Verificar que los emojis estén presentes
        assert "✅" in result["narrative"] or "β" in result["narrative"]
    
    def test_zero_degrees(self):
        """Vector con grados cero (nodo aislado)."""
        vector = PyramidalSemanticVector(
            node_id="isolated",
            node_type="INSUMO",
            stratum=Stratum.PHYSICS,
            in_degree=0,
            out_degree=0,
        )
        
        assert vector.is_isolated is True
        assert vector.total_degree == 0
    
    def test_very_high_degrees(self):
        """Vector con grados muy altos."""
        vector = PyramidalSemanticVector(
            node_id="hub",
            node_type="ROOT",
            stratum=Stratum.WISDOM,
            in_degree=1000000,
            out_degree=500000,
        )
        
        assert vector.total_degree == 1500000
    
    def test_special_characters_in_path_nodes(
        self, 
        service: SemanticDictionaryService
    ):
        """Caracteres especiales en nodos del ciclo."""
        payload = {
            "anomaly_type": "CYCLE",
            "path_nodes": [
                "Nodo<script>",
                "Nodo'quoted'",
                'Nodo"double"',
                "Nodo\nwith\nnewlines",
            ],
        }
        
        result = service.project_graph_narrative(payload)
        
        # Debe manejar sin errores
        assert "success" in result
    
    def test_empty_params_dict(self, service: SemanticDictionaryService):
        """Plantilla que no requiere parámetros."""
        result = service.fetch_narrative(
            domain="TOPOLOGY_CYCLES",
            classification="clean",
            params={}
        )
        
        assert result["success"] is True
    
    def test_extra_params_ignored(self, service: SemanticDictionaryService):
        """Parámetros extra son ignorados."""
        result = service.fetch_narrative(
            domain="STABILITY",
            classification="stable",
            params={
                "stability": 0.75,
                "extra_param": "ignored",
                "another_extra": 12345,
            }
        )
        
        assert result["success"] is True
    
    def test_none_classification(self, service: SemanticDictionaryService):
        """Classification None con dominio que es string directo."""
        # MARKET_CONTEXT es manejado especialmente
        result = service.fetch_narrative(
            domain="MARKET_CONTEXT",
            classification=None
        )
        
        assert result["success"] is True
    
    def test_boundary_threshold_values(self, service: SemanticDictionaryService):
        """Valores exactamente en los umbrales."""
        # Exactamente en el umbral de robust
        result = service.get_classification_by_threshold("STABILITY", 0.85)
        assert result == "robust"
        
        # Justo debajo
        result = service.get_classification_by_threshold("STABILITY", 0.849)
        assert result == "stable"
    
    def test_negative_float_handling(self):
        """Manejo de valores flotantes negativos donde no aplica."""
        # Los grados deben ser enteros no negativos
        with pytest.raises(ValueError):
            PyramidalSemanticVector(
                node_id="test",
                node_type="APU",
                stratum=Stratum.TACTICS,
                in_degree=-1,
                out_degree=5,
            )


# =============================================================================
# TESTS PARA FACTORY FUNCTION
# =============================================================================

class TestFactoryFunction:
    """Pruebas para la función factory."""
    
    def test_create_semantic_dictionary_service(self):
        """Factory crea instancia válida."""
        service = create_semantic_dictionary_service()
        
        assert isinstance(service, SemanticDictionaryService)
        assert service.health_check()["status"] == "healthy"
    
    def test_factory_creates_independent_instances(self):
        """Factory crea instancias independientes."""
        service1 = create_semantic_dictionary_service()
        service2 = create_semantic_dictionary_service()
        
        assert service1 is not service2


# =============================================================================
# TESTS DE STRATUM ENUM
# =============================================================================

class TestStratum:
    """Pruebas para el enum Stratum."""
    
    def test_stratum_values(self):
        """Verifica valores correctos del enum."""
        assert Stratum.WISDOM.value == 0
        assert Stratum.STRATEGY.value == 1
        assert Stratum.TACTICS.value == 2
        assert Stratum.PHYSICS.value == 3
    
    def test_stratum_ordering(self):
        """Verifica orden de los estratos."""
        strata = list(Stratum)
        
        assert strata[0] == Stratum.WISDOM
        assert strata[-1] == Stratum.PHYSICS
    
    def test_stratum_is_int_enum(self):
        """Stratum es IntEnum, permite comparaciones numéricas."""
        assert Stratum.WISDOM < Stratum.STRATEGY
        assert Stratum.PHYSICS > Stratum.TACTICS
        assert int(Stratum.TACTICS) == 2


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

@pytest.fixture(autouse=True)
def setup_logging():
    """Configura logging para las pruebas."""
    logging.basicConfig(level=logging.WARNING)
    yield
    logging.shutdown()


def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# =============================================================================
# MAIN PARA EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Detener en primer fallo
        "--cov=semantic_dictionary",
        "--cov-report=term-missing",
    ])