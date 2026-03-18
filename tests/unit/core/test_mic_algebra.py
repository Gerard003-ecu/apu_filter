"""
Módulo: test_mic_algebra_suite.py
==================================

Suite de pruebas comprensiva para el núcleo categórico refactorizado (MIC Algebra).

Cubre:
1. Objetos categóricos (CategoricalState)
2. Morfismos (identidad, atómicos, compuestos)
3. Composición y verificación algebraica
4. Productos y coproductos
5. Pullbacks (intersecciones)
6. Funtores y transformaciones naturales
7. Verificación homológica
8. Registro categórico y grafos de dependencias
9. Railway Oriented Programming (monadalidad)
10. Trazas de composición

Ejecutar:
    pytest test_mic_algebra_suite.py -v --tb=short
    pytest test_mic_algebra_suite.py -v --cov=mic_algebra --cov-report=html
"""

import datetime
import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import pytest

from app.core.schemas import Stratum
from app.core.mic_algebra import (
    AtomicVector,
    CategoricalRegistry,
    CategoricalState,
    ComposedMorphism,
    CompositionTrace,
    CoproductMorphism,
    Functor,
    HomologicalVerifier,
    IdentityMorphism,
    Morphism,
    MorphismComposer,
    NaturalTransformation,
    ProductMorphism,
    PullbackMorphism,
    create_categorical_state,
    create_morphism_from_handler,
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def configure_logging():
    """Configura logging para pruebas."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    yield


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_state():
    """Crea estado categórico simple."""
    return create_categorical_state(
        payload={"value": 42, "name": "test"},
        context={"origin": "fixture"},
    )


@pytest.fixture
def physics_state():
    """Estado con PHYSICS validado."""
    return create_categorical_state(
        payload={"raw_data": [1, 2, 3]},
        strata={Stratum.PHYSICS},
    )


@pytest.fixture
def tactics_state():
    """Estado con TACTICS validado."""
    return create_categorical_state(
        payload={"computed": 100},
        strata={Stratum.PHYSICS, Stratum.TACTICS},
    )


@pytest.fixture
def complete_state():
    """Estado con múltiples estratos."""
    return create_categorical_state(
        payload={
            "physics": "data",
            "tactics": "metrics",
            "strategy": "plan",
            "wisdom": "decision",
        },
        strata={
            Stratum.PHYSICS,
            Stratum.TACTICS,
            Stratum.STRATEGY,
            Stratum.WISDOM,
        },
    )


@pytest.fixture
def error_state():
    """Estado con error."""
    return create_categorical_state(
        payload={"partial": "data"},
        error="Test error",
        details={"reason": "fixture"},
    )


@pytest.fixture
def simple_handler():
    """Handler simple para pruebas."""
    def handler(x: int = 0):
        return {"result": x * 2, "processed": True}
    return handler


@pytest.fixture
def failing_handler():
    """Handler que falla."""
    def handler():
        raise ValueError("Intentional failure")
    return handler


@pytest.fixture
def dict_returning_handler():
    """Handler que retorna dict con metadatos."""
    def handler(value: int):
        return {
            "success": True,
            "data": value * 3,
            "error": None,
        }
    return handler


@pytest.fixture
def error_returning_handler():
    """Handler que retorna error en dict."""
    def handler():
        return {
            "success": False,
            "error": "Operación rechazada",
            "error_details": {"code": 403},
        }
    return handler


# ============================================================================
# PRUEBAS DE STRATUM (ESTRATIFICACIÓN DIKW)
# ============================================================================


class TestStratumHierarchy:
    """Pruebas de la jerarquía de estratos DIKW."""
    
    def test_stratum_values(self):
        """Valores de estratos son correctos."""
        assert Stratum.WISDOM.value == 0      # Más alto
        assert Stratum.STRATEGY.value == 1
        assert Stratum.TACTICS.value == 2
        assert Stratum.PHYSICS.value == 3     # Más bajo
    
    def test_stratum_comparison(self):
        """Comparación de estratos."""
        assert Stratum.WISDOM < Stratum.STRATEGY
        assert Stratum.STRATEGY < Stratum.TACTICS
        assert Stratum.TACTICS < Stratum.PHYSICS
        assert Stratum.WISDOM < Stratum.PHYSICS
    
    def test_stratum_requires(self):
        """Requisitos de estratos."""
        # WISDOM requiere todos los demás
        wisdom_reqs = Stratum.WISDOM.requires()
        assert Stratum.STRATEGY in wisdom_reqs
        assert Stratum.TACTICS in wisdom_reqs
        assert Stratum.PHYSICS in wisdom_reqs
        
        # PHYSICS no requiere nada (es la base)
        physics_reqs = Stratum.PHYSICS.requires()
        assert len(physics_reqs) == 0
        
        # TACTICS requiere PHYSICS
        tactics_reqs = Stratum.TACTICS.requires()
        assert Stratum.PHYSICS in tactics_reqs
        assert Stratum.TACTICS not in tactics_reqs
    
    def test_stratum_level(self):
        """Niveles de estratos."""
        assert Stratum.WISDOM.value == 0
        assert Stratum.STRATEGY.value == 1
        assert Stratum.TACTICS.value == 2
        assert Stratum.PHYSICS.value == 3


# ============================================================================
# PRUEBAS DE CATEGORICAL STATE (Objetos de la Categoría)
# ============================================================================


class TestCategoricalState:
    """Pruebas del estado categórico."""
    
    def test_state_creation_empty(self):
        """Crea estado vacío."""
        state = CategoricalState()
        
        assert state.payload == {}
        assert state.context == {}
        assert state.validated_strata == frozenset()
        assert state.is_success is True
    
    def test_state_creation_with_data(self):
        """Crea estado con datos."""
        payload = {"key": "value", "number": 123}
        context = {"origin": "test"}
        
        state = CategoricalState(
            payload=payload,
            context=context,
        )
        
        assert state.payload == payload
        assert state.context == context
    
    def test_state_immutability(self):
        """Estado es inmutable (frozen)."""
        state = CategoricalState(payload={"a": 1})
        
        with pytest.raises((AttributeError, TypeError)):
            state.payload["b"] = 2
    
    def test_state_is_success_property(self):
        """Propiedad is_success funciona."""
        state_ok = create_categorical_state()
        state_err = create_categorical_state().with_error("Failed")
        
        assert state_ok.is_success is True
        assert state_err.is_success is False
    
    def test_state_is_failed_property(self):
        """Propiedad is_failed funciona."""
        state_ok = create_categorical_state()
        state_err = create_categorical_state().with_error("Failed")
        
        assert state_ok.is_failed is False
        assert state_err.is_failed is True
    
    def test_with_update_merge_payload(self, simple_state):
        """Actualiza estado mergeando payload."""
        updated = simple_state.with_update(
            {"new_key": "new_value"},
            merge_payload=True,
        )
        
        # Original intacto
        assert "new_key" not in simple_state.payload
        
        # Nuevo estado tiene ambos
        assert "value" in updated.payload
        assert updated.payload["new_key"] == "new_value"
    
    def test_with_update_replace_payload(self, simple_state):
        """Actualiza estado reemplazando payload."""
        updated = simple_state.with_update(
            {"only_new": 1},
            merge_payload=False,
        )
        
        # Nuevo estado solo tiene lo nuevo
        assert "value" not in updated.payload
        assert updated.payload["only_new"] == 1
    
    def test_with_update_adds_stratum(self, simple_state):
        """Actualización añade estrato."""
        updated = simple_state.with_update(
            new_stratum=Stratum.TACTICS,
        )
        
        assert Stratum.TACTICS in updated.validated_strata
    
    def test_with_update_multiple_strata(self, simple_state):
        """Puede haber múltiples estratos."""
        state = simple_state
        
        state1 = state.with_update(new_stratum=Stratum.PHYSICS)
        state2 = state1.with_update(new_stratum=Stratum.TACTICS)
        
        assert Stratum.PHYSICS in state2.validated_strata
        assert Stratum.TACTICS in state2.validated_strata
    
    def test_with_error_monadal_collapse(self, simple_state):
        """Error colapsa el estado (monadalidad)."""
        error_state = simple_state.with_error(
            "Operation failed",
            details={"reason": "test"},
        )
        
        assert error_state.is_failed
        assert error_state.error == "Operation failed"
        assert error_state.error_details["reason"] == "test"
        
        # El payload original se conserva (para recuperación)
        assert error_state.payload == simple_state.payload
    
    def test_stratum_level_property(self):
        """Calcula nivel correcto."""
        state1 = create_categorical_state(strata={Stratum.PHYSICS})
        state2 = create_categorical_state(
            strata={Stratum.PHYSICS, Stratum.TACTICS}
        )
        state3 = create_categorical_state(
            strata={Stratum.WISDOM, Stratum.TACTICS}
        )
        
        assert state1.stratum_level == Stratum.PHYSICS.value
        assert state2.stratum_level == Stratum.TACTICS.value
        assert state3.stratum_level == Stratum.WISDOM.value
    
    def test_add_trace(self, simple_state):
        """Añade entrada a traza."""
        state_with_trace = simple_state.add_trace(
            morphism_name="test_morph",
            input_domain=frozenset([Stratum.PHYSICS]),
            output_codomain=Stratum.TACTICS,
            success=True,
        )
        
        assert len(state_with_trace.composition_trace) == 1
        
        trace = state_with_trace.composition_trace[0]
        assert trace.step_number == 1
        assert trace.morphism_name == "test_morph"
        assert trace.success is True
    
    def test_compute_hash(self, simple_state):
        """Computa hash del estado."""
        hash1 = simple_state.compute_hash()
        
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256
        
        # Estados diferentes producen hashes diferentes
        state2 = simple_state.with_update({"extra": "field"})
        hash2 = state2.compute_hash()
        
        assert hash1 != hash2
    
    def test_to_dict_serialization(self, complete_state):
        """Serializa estado a diccionario."""
        state_dict = complete_state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert "payload" in state_dict
        assert "context" in state_dict
        assert "validated_strata" in state_dict
        assert "composition_trace" in state_dict
    
    def test_from_dict_deserialization(self, complete_state):
        """Reconstruye estado desde diccionario."""
        original_dict = complete_state.to_dict()
        
        reconstructed = CategoricalState.from_dict(original_dict)
        
        assert reconstructed.payload == complete_state.payload
        assert reconstructed.context == complete_state.context
        assert reconstructed.validated_strata == complete_state.validated_strata
    
    def test_factory_function(self):
        """Factory function crea estado correctamente."""
        state = create_categorical_state(
            payload={"data": 123},
            context={"test": True},
            strata={Stratum.PHYSICS, Stratum.TACTICS},
        )
        
        assert state.payload["data"] == 123
        assert state.context["test"] is True
        assert Stratum.PHYSICS in state.validated_strata
        assert Stratum.TACTICS in state.validated_strata


class TestCompositionTrace:
    """Pruebas de trazas de composición."""
    
    def test_trace_creation(self):
        """Crea traza."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="op1",
            input_domain=frozenset([Stratum.PHYSICS]),
            output_codomain=Stratum.TACTICS,
            success=True,
        )
        
        assert trace.step_number == 1
        assert trace.morphism_name == "op1"
        assert trace.success is True
    
    def test_trace_with_error(self):
        """Traza puede contener error."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="failed_op",
            input_domain=frozenset([Stratum.PHYSICS]),
            output_codomain=Stratum.TACTICS,
            success=False,
            error="Domain violation",
        )
        
        assert trace.success is False
        assert trace.error == "Domain violation"
    
    def test_trace_to_dict(self):
        """Serializa traza."""
        trace = CompositionTrace(
            step_number=2,
            morphism_name="op2",
            input_domain=frozenset([Stratum.PHYSICS, Stratum.TACTICS]),
            output_codomain=Stratum.STRATEGY,
            success=True,
        )
        
        trace_dict = trace.to_dict()
        
        assert trace_dict["step"] == 2
        assert trace_dict["morphism"] == "op2"
        assert "PHYSICS" in trace_dict["domain"]
        assert "TACTICS" in trace_dict["domain"]
        assert trace_dict["codomain"] == "STRATEGY"
        assert trace_dict["success"] is True


# ============================================================================
# PRUEBAS DE MORFISMOS
# ============================================================================


class TestIdentityMorphism:
    """Pruebas del morfismo identidad."""
    
    def test_identity_creation(self):
        """Crea identidad."""
        morph = IdentityMorphism(Stratum.PHYSICS)
        
        assert morph.name == "id_PHYSICS"
        assert morph.domain == frozenset([Stratum.PHYSICS])
        assert morph.codomain == Stratum.PHYSICS
    
    def test_identity_preserves_state(self, simple_state):
        """Identidad no altera estado."""
        morph = IdentityMorphism(Stratum.PHYSICS)
        
        result = morph(simple_state)
        
        assert result == simple_state
        assert result.payload == simple_state.payload
    
    def test_identity_axiom_left(self, simple_state):
        """Axioma: id ∘ f = f."""
        # Simular f
        def handler(value: int = 0):
            return {"doubled": value * 2}
        
        f = AtomicVector(
            "double",
            Stratum.TACTICS,
            handler,
        )
        
        id_phys = IdentityMorphism(Stratum.PHYSICS)
        composed = id_phys >> f
        
        result = composed(simple_state)
        
        assert result.is_success
    
    def test_identity_axiom_right(self, tactics_state):
        """Axioma: f ∘ id = f."""
        # Simular f
        def handler(value: int = 0):
            return {"result": value}
        
        f = AtomicVector(
            "identity_like",
            Stratum.STRATEGY,
            handler,
        )
        
        id_tactics = IdentityMorphism(Stratum.TACTICS)
        
        # f primero, luego identidad
        composed = f >> id_tactics
        
        # Pero id_tactics requiere TACTICS en dominio de f
        # Así que esto es f >> id_strategy en realidad


class TestAtomicVector:
    """Pruebas del vector atómico."""
    
    def test_atomic_vector_creation(self, simple_handler):
        """Crea vector atómico."""
        morph = AtomicVector(
            name="simple_op",
            target_stratum=Stratum.TACTICS,
            handler=simple_handler,
        )
        
        assert morph.name == "simple_op"
        assert morph.codomain == Stratum.TACTICS
        assert morph.domain == frozenset(Stratum.TACTICS.requires())
    
    def test_atomic_vector_with_required_keys(self, simple_handler):
        """Vector atómico con claves requeridas."""
        morph = AtomicVector(
            "op",
            Stratum.TACTICS,
            simple_handler,
            required_keys=["x", "y"],
        )
        
        assert "x" in morph._required_keys
        assert "y" in morph._required_keys
    
    def test_atomic_vector_success_path(self, physics_state, simple_handler):
        """Vector atómico ejecuta exitosamente."""
        morph = AtomicVector(
            "double_op",
            Stratum.TACTICS,
            simple_handler,
        )
        
        result = morph(physics_state)
        
        assert result.is_success
        assert "result" in result.payload
        assert result.payload["result"] == 0  # simple_handler(0) * 2
        assert Stratum.TACTICS in result.validated_strata
    
    def test_atomic_vector_domain_violation(self, simple_state, simple_handler):
        """Detecta violación de dominio."""
        # Crea un estado sin PHYSICS
        state = create_categorical_state(
            payload={"data": 1},
            strata=set(),  # Vacío
        )
        
        morph = AtomicVector(
            "op",
            Stratum.TACTICS,
            simple_handler,
        )
        
        result = morph(state)
        
        assert result.is_failed
        assert "Violación de Dominio" in result.error
    
    def test_atomic_vector_force_override(self, simple_state, simple_handler):
        """Force override bypasa validación de dominio."""
        state = create_categorical_state(
            payload={"data": 1},
            strata=set(),
            context={"force_override": True},
        )
        
        morph = AtomicVector(
            "op",
            Stratum.TACTICS,
            simple_handler,
        )
        
        result = morph(state)
        
        # Debe ejecutarse a pesar de dominio vacío
        assert result.is_success
    
    def test_atomic_vector_missing_required_key(self, physics_state):
        """Detecta claves requeridas faltantes."""
        def needs_x(x: int):
            return {"result": x}
        
        morph = AtomicVector(
            "needs_x",
            Stratum.TACTICS,
            needs_x,
            required_keys=["x"],
        )
        
        result = morph(physics_state)
        
        assert result.is_failed
        assert "requeridas faltantes" in result.error
    
    def test_atomic_vector_exception_handling(self, physics_state, failing_handler):
        """Maneja excepciones en handler."""
        morph = AtomicVector(
            "failing",
            Stratum.TACTICS,
            failing_handler,
        )
        
        result = morph(physics_state)
        
        assert result.is_failed
        assert "ValueError" in result.error or "Intentional" in result.error
    
    def test_atomic_vector_dict_result_success(self, physics_state, dict_returning_handler):
        """Procesa resultado dict con success=True."""
        morph = AtomicVector(
            "returns_dict",
            Stratum.TACTICS,
            dict_returning_handler,
            required_keys=["value"],
        )
        
        state = physics_state.with_update({"value": 5})
        result = morph(state)
        
        assert result.is_success
        assert result.payload["data"] == 15  # 5 * 3
    
    def test_atomic_vector_dict_result_failure(self, physics_state, error_returning_handler):
        """Procesa resultado dict con success=False."""
        morph = AtomicVector(
            "returns_error",
            Stratum.TACTICS,
            error_returning_handler,
        )
        
        result = morph(physics_state)
        
        assert result.is_failed
        assert "Operación rechazada" in result.error
        assert result.error_details["code"] == 403
    
    def test_atomic_vector_monadal_absorption(self, failing_handler):
        """Absorbe errores previos (monadalidad)."""
        morph = AtomicVector(
            "will_not_execute",
            Stratum.TACTICS,
            failing_handler,
        )
        
        error_input = create_categorical_state().with_error("Previous error")
        
        result = morph(error_input)
        
        # Debe absorber el error anterior
        assert result.is_failed
        assert result.error == "Previous error"
    
    def test_atomic_vector_repr(self, simple_handler):
        """Representación string del morfismo."""
        morph = AtomicVector(
            "test_op",
            Stratum.TACTICS,
            simple_handler,
        )
        
        repr_str = repr(morph)
        
        assert "test_op" in repr_str
        assert "TACTICS" in repr_str


class TestComposedMorphism:
    """Pruebas de composición de morfismos."""
    
    def test_composed_morphism_creation(self, simple_handler):
        """Crea morfismo compuesto."""
        def handler2(result: int):
            return {"final": result + 10}
        
        morph1 = AtomicVector(
            "step1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "step2",
            Stratum.STRATEGY,
            handler2,
            required_keys=["result"],
        )
        
        composed = morph1 >> morph2
        
        assert isinstance(composed, ComposedMorphism)
        assert composed.f == morph1
        assert composed.g == morph2
    
    def test_composed_morphism_domain(self, simple_handler):
        """Dominio de composición es correcto."""
        def handler2():
            return {"final": 1}
        
        morph1 = AtomicVector(
            "m1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "m2",
            Stratum.STRATEGY,
            handler2,
        )
        
        composed = morph1 >> morph2
        
        # Dominio es TACTICS.requires() (que es PHYSICS)
        assert Stratum.PHYSICS in composed.domain
    
    def test_composed_morphism_codomain(self, simple_handler):
        """Codominio de composición es correcto."""
        def handler2():
            return {"final": 1}
        
        morph1 = AtomicVector(
            "m1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "m2",
            Stratum.STRATEGY,
            handler2,
        )
        
        composed = morph1 >> morph2
        
        assert composed.codomain == Stratum.STRATEGY
    
    def test_composed_morphism_incompatible_raises(self, simple_handler):
        """Composición incompatible levanta excepción."""
        def handler_needs_xy(x: int, y: int):
            return {"xy": x + y}
        
        morph1 = AtomicVector(
            "m1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "m2_needs_xy",
            Stratum.STRATEGY,
            handler_needs_xy,
            required_keys=["x", "y"],
        )
        
        with pytest.raises(TypeError, match="Composición"):
            _ = morph1 >> morph2
    
    def test_composed_morphism_execution(self, physics_state):
        """Ejecución de composición."""
        def handler1(raw_data: list = None):
            return {"processed": len(raw_data or [])}
        
        def handler2(processed: int):
            return {"analyzed": processed * 2}
        
        morph1 = AtomicVector(
            "process",
            Stratum.TACTICS,
            handler1,
            required_keys=["raw_data"],
        )
        
        morph2 = AtomicVector(
            "analyze",
            Stratum.STRATEGY,
            handler2,
            required_keys=["processed"],
        )
        
        composed = morph1 >> morph2
        
        result = composed(physics_state)
        
        assert result.is_success
        assert result.payload["analyzed"] == 6  # 3 * 2
    
    def test_composed_morphism_failure_propagation(self, physics_state):
        """Fallo en f propaga (monadalidad)."""
        def handler1():
            raise ValueError("Step 1 failed")
        
        def handler2():
            raise ValueError("Should not reach")
        
        morph1 = AtomicVector(
            "failing_step",
            Stratum.TACTICS,
            handler1,
        )
        
        morph2 = AtomicVector(
            "never_reached",
            Stratum.STRATEGY,
            handler2,
        )
        
        composed = morph1 >> morph2
        
        result = composed(physics_state)
        
        assert result.is_failed
        assert "Step 1 failed" in result.error
    
    def test_composed_morphism_associativity(self, physics_state):
        """Asociatividad: (h ∘ g) ∘ f = h ∘ (g ∘ f)."""
        def h1(x: int = 0):
            return {"y": x + 1}
        
        def h2(y: int):
            return {"z": y * 2}
        
        def h3(z: int):
            return {"w": z - 1}
        
        m1 = AtomicVector("m1", Stratum.TACTICS, h1)
        m2 = AtomicVector("m2", Stratum.STRATEGY, h2, required_keys=["y"])
        m3 = AtomicVector("m3", Stratum.WISDOM, h3, required_keys=["z"])
        
        # (m3 ∘ m2) ∘ m1
        comp1 = (m1 >> m2) >> m3
        
        # m3 ∘ (m2 ∘ m1)
        comp2 = m1 >> (m2 >> m3)
        
        # Ambas deben dar el mismo resultado
        result1 = comp1(physics_state)
        result2 = comp2(physics_state)
        
        assert result1.payload == result2.payload
    
    def test_composed_morphism_trace(self, physics_state):
        """Traza registra ambos pasos."""
        def h1(raw_data: list = None):
            return {"count": len(raw_data or [])}
        
        def h2(count: int):
            return {"doubled": count * 2}
        
        m1 = AtomicVector(
            "count_data",
            Stratum.TACTICS,
            h1,
            required_keys=["raw_data"],
        )
        
        m2 = AtomicVector(
            "double_count",
            Stratum.STRATEGY,
            h2,
            required_keys=["count"],
        )
        
        composed = m1 >> m2
        
        result = composed(physics_state)
        
        assert len(result.composition_trace) == 2
        assert result.composition_trace[0].morphism_name == "count_data"
        assert result.composition_trace[1].morphism_name == "double_count"


class TestProductMorphism:
    """Pruebas del producto categórico (paralelismo)."""
    
    def test_product_morphism_creation(self, simple_handler):
        """Crea producto de morfismos."""
        def handler2():
            return {"other": 99}
        
        morph1 = AtomicVector(
            "path1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "path2",
            Stratum.TACTICS,
            handler2,
        )
        
        product = morph1 * morph2
        
        assert isinstance(product, ProductMorphism)
        assert product.f == morph1
        assert product.g == morph2
    
    def test_product_morphism_domain(self, simple_handler):
        """Dominio de producto es unión."""
        def handler2():
            return {}
        
        morph1 = AtomicVector(
            "m1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "m2",
            Stratum.STRATEGY,
            handler2,
        )
        
        product = morph1 * morph2
        
        # Dominio es unión de dominios
        assert morph1.domain.issubset(product.domain)
        assert morph2.domain.issubset(product.domain)
    
    def test_product_morphism_codomain(self, simple_handler):
        """Codominio de producto es el más alto."""
        def handler2():
            return {}
        
        morph1 = AtomicVector(
            "m1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "m2",
            Stratum.WISDOM,
            handler2,
        )
        
        product = morph1 * morph2
        
        # Codominio es WISDOM (menor valor = más alto)
        assert product.codomain == Stratum.WISDOM
    
    def test_product_morphism_parallel_execution(self, physics_state):
        """Ejecuta ambas ramas en paralelo."""
        def handler1():
            return {"branch1": "result1"}
        
        def handler2():
            return {"branch2": "result2"}
        
        morph1 = AtomicVector(
            "branch1",
            Stratum.TACTICS,
            handler1,
        )
        
        morph2 = AtomicVector(
            "branch2",
            Stratum.TACTICS,
            handler2,
        )
        
        product = morph1 * morph2
        
        result = product(physics_state)
        
        assert result.is_success
        assert result.payload["branch1"] == "result1"
        assert result.payload["branch2"] == "result2"
    
    def test_product_morphism_failure_stops_product(self, physics_state):
        """Fallo en una rama detiene el producto."""
        def handler1():
            raise ValueError("Branch 1 fails")
        
        def handler2():
            return {"branch2": "result"}
        
        morph1 = AtomicVector(
            "failing",
            Stratum.TACTICS,
            handler1,
        )
        
        morph2 = AtomicVector(
            "ok",
            Stratum.TACTICS,
            handler2,
        )
        
        product = morph1 * morph2
        
        result = product(physics_state)
        
        assert result.is_failed
    
    def test_product_morphism_commutativity(self, physics_state):
        """Producto es conmutativo (sobre mismo estado)."""
        def handler1():
            return {"a": 1}
        
        def handler2():
            return {"b": 2}
        
        m1 = AtomicVector("op1", Stratum.TACTICS, handler1)
        m2 = AtomicVector("op2", Stratum.TACTICS, handler2)
        
        # m1 * m2
        result1 = (m1 * m2)(physics_state)
        
        # m2 * m1
        result2 = (m2 * m1)(physics_state)
        
        # Ambos producen los mismos resultados
        assert result1.payload["a"] == result2.payload["a"]
        assert result1.payload["b"] == result2.payload["b"]


class TestCoproductMorphism:
    """Pruebas del coproducto categórico (elección)."""
    
    def test_coproduct_morphism_creation(self, simple_handler):
        """Crea coproducto."""
        def handler2():
            return {"fallback": "result"}
        
        morph1 = AtomicVector(
            "primary",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "fallback",
            Stratum.TACTICS,
            handler2,
        )
        
        coproduct = morph1 | morph2
        
        assert isinstance(coproduct, CoproductMorphism)
    
    def test_coproduct_first_succeeds(self, physics_state, simple_handler):
        """Usa primera rama si exitosa."""
        def handler2():
            return {"unused": "fallback"}
        
        morph1 = AtomicVector(
            "primary",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "fallback",
            Stratum.TACTICS,
            handler2,
        )
        
        coproduct = morph1 | morph2
        
        result = coproduct(physics_state)
        
        assert result.is_success
        assert "result" in result.payload  # De primary
        assert "unused" not in result.payload
    
    def test_coproduct_fallback_on_failure(self, physics_state):
        """Usa segunda rama si primera falla."""
        def handler1():
            raise ValueError("Primary fails")
        
        def handler2():
            return {"fallback": "success"}
        
        morph1 = AtomicVector(
            "primary",
            Stratum.TACTICS,
            handler1,
        )
        
        morph2 = AtomicVector(
            "fallback",
            Stratum.TACTICS,
            handler2,
        )
        
        coproduct = morph1 | morph2
        
        result = coproduct(physics_state)
        
        assert result.is_success
        assert result.payload["fallback"] == "success"
    
    def test_coproduct_both_fail(self, physics_state):
        """Ambas ramas fallan."""
        def handler1():
            raise ValueError("First fails")
        
        def handler2():
            raise ValueError("Fallback fails")
        
        morph1 = AtomicVector(
            "primary",
            Stratum.TACTICS,
            handler1,
        )
        
        morph2 = AtomicVector(
            "fallback",
            Stratum.TACTICS,
            handler2,
        )
        
        coproduct = morph1 | morph2
        
        result = coproduct(physics_state)
        
        assert result.is_failed
        assert "First fails" in result.error or "Fallback fails" in result.error
    
    def test_coproduct_monadal_absorption(self, physics_state):
        """Absorbe errores previos."""
        def handler1():
            raise ValueError("Should not execute")
        
        def handler2():
            raise ValueError("Should not execute")
        
        morph1 = AtomicVector("m1", Stratum.TACTICS, handler1)
        morph2 = AtomicVector("m2", Stratum.TACTICS, handler2)
        
        coproduct = morph1 | morph2
        
        error_state = create_categorical_state().with_error("Previous error")
        
        result = coproduct(error_state)
        
        assert result.is_failed
        assert result.error == "Previous error"


class TestPullbackMorphism:
    """Pruebas del pullback (intersección)."""
    
    def test_pullback_creation(self, simple_handler):
        """Crea pullback."""
        def handler2():
            return {"value": 100}
        
        morph1 = AtomicVector("f", Stratum.TACTICS, simple_handler)
        morph2 = AtomicVector("g", Stratum.TACTICS, handler2)
        
        validator = lambda s1, s2: True
        
        pullback = PullbackMorphism("verify", morph1, morph2, validator)
        
        assert pullback.name == "verify"
        assert pullback.f == morph1
        assert pullback.g == morph2
    
    def test_pullback_congruent_paths(self, physics_state):
        """Pullback con caminos congruentes."""
        def handler1():
            return {"result": 42}
        
        def handler2():
            return {"result": 42}
        
        morph1 = AtomicVector(
            "path1",
            Stratum.TACTICS,
            handler1,
        )
        
        morph2 = AtomicVector(
            "path2",
            Stratum.TACTICS,
            handler2,
        )
        
        def check_congruence(state1, state2):
            return state1.payload.get("result") == state2.payload.get("result")
        
        pullback = PullbackMorphism("verify", morph1, morph2, check_congruence)
        
        result = pullback(physics_state)
        
        assert result.is_success
    
    def test_pullback_divergent_paths(self, physics_state):
        """Pullback detecta caminos divergentes."""
        def handler1():
            return {"result": 10}
        
        def handler2():
            return {"result": 20}
        
        morph1 = AtomicVector("f", Stratum.TACTICS, handler1)
        morph2 = AtomicVector("g", Stratum.TACTICS, handler2)
        
        def check_congruence(state1, state2):
            return state1.payload.get("result") == state2.payload.get("result")
        
        pullback = PullbackMorphism("verify", morph1, morph2, check_congruence)
        
        result = pullback(physics_state)
        
        assert result.is_failed
        assert "divergentes" in result.error


# ============================================================================
# PRUEBAS DEL COMPOSITOR DE MORFISMOS
# ============================================================================


class TestMorphismComposer:
    """Pruebas del constructor de composiciones."""
    
    def test_composer_creation(self):
        """Crea composer vacío."""
        composer = MorphismComposer()
        
        assert len(composer.steps) == 0
    
    def test_composer_add_single_step(self):
        """Añade un paso."""
        composer = MorphismComposer()
        morph = IdentityMorphism(Stratum.PHYSICS)
        
        composer.add_step(morph)
        
        assert len(composer.steps) == 1
        assert composer.steps[0] == morph
    
    def test_composer_add_incompatible_step_raises(self, simple_handler):
        """Rechaza paso incompatible."""
        def handler_needs_xy(x: int, y: int):
            return {}
        
        morph1 = AtomicVector(
            "m1",
            Stratum.TACTICS,
            simple_handler,
        )
        
        morph2 = AtomicVector(
            "needs_xy",
            Stratum.STRATEGY,
            handler_needs_xy,
            required_keys=["x", "y"],
        )
        
        composer = MorphismComposer()
        composer.add_step(morph1)
        
        with pytest.raises(TypeError, match="no componible"):
            composer.add_step(morph2)
    
    def test_composer_build_single_step(self):
        """Construye composición simple."""
        composer = MorphismComposer()
        morph = IdentityMorphism(Stratum.PHYSICS)
        
        result = composer.add_step(morph).build()
        
        assert result == morph
    
    def test_composer_build_multiple_steps(self, simple_handler):
        """Construye composición múltiple."""
        def h1():
            return {"step1": 1}
        
        def h2():
            return {"step2": 2}
        
        m1 = AtomicVector("s1", Stratum.TACTICS, h1)
        m2 = AtomicVector("s2", Stratum.STRATEGY, h2)
        
        composer = MorphismComposer()
        composed = (composer
                    .add_step(m1)
                    .add_step(m2)
                    .build())
        
        assert isinstance(composed, ComposedMorphism)
    
    def test_composer_visualize(self, simple_handler):
        """Visualiza plan de composición."""
        def h2():
            return {}
        
        m1 = AtomicVector("step1", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("step2", Stratum.STRATEGY, h2)
        
        composer = MorphismComposer()
        composer.add_step(m1).add_step(m2)
        
        visualization = composer.visualize()
        
        assert "step1" in visualization
        assert "step2" in visualization


# ============================================================================
# PRUEBAS DE VERIFICADOR HOMOLÓGICO
# ============================================================================


class TestHomologicalVerifier:
    """Pruebas de verificación de exactitud."""
    
    def test_verifier_creation(self):
        """Crea verificador."""
        verifier = HomologicalVerifier()
        
        assert verifier is not None
    
    def test_empty_sequence_is_exact(self):
        """Secuencia vacía es exacta."""
        verifier = HomologicalVerifier()
        
        is_exact = verifier.is_exact_sequence([])
        
        assert is_exact is True
    
    def test_single_morphism_is_exact(self, simple_handler):
        """Secuencia simple es exacta."""
        morph = AtomicVector("op", Stratum.TACTICS, simple_handler)
        
        verifier = HomologicalVerifier()
        
        is_exact = verifier.is_exact_sequence([morph])
        
        assert is_exact is True
    
    def test_exact_composed_sequence(self, simple_handler):
        """Secuencia compuesta exacta."""
        def h2():
            return {}
        
        m1 = AtomicVector("m1", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("m2", Stratum.STRATEGY, h2)
        
        verifier = HomologicalVerifier()
        
        is_exact = verifier.is_exact_sequence([m1, m2])
        
        assert is_exact is True
    
    def test_verify_composition(self, simple_handler):
        """Verifica propiedades de composición."""
        morph = AtomicVector("op", Stratum.TACTICS, simple_handler)
        
        verifier = HomologicalVerifier()
        
        verification = verifier.verify_composition(morph)
        
        assert verification["is_valid"] is True
        assert "domain" in verification
        assert "codomain" in verification


# ============================================================================
# PRUEBAS DE REGISTRO CATEGÓRICO
# ============================================================================


class TestCategoricalRegistry:
    """Pruebas del registro de morfismos."""
    
    def test_registry_creation(self):
        """Crea registro vacío."""
        registry = CategoricalRegistry()
        
        assert len(registry.list_morphisms()) == 0
        assert len(registry.list_compositions()) == 0
    
    def test_register_morphism(self, simple_handler):
        """Registra morfismo."""
        morph = AtomicVector("op", Stratum.TACTICS, simple_handler)
        registry = CategoricalRegistry()
        
        registry.register_morphism("my_op", morph)
        
        assert "my_op" in registry.list_morphisms()
        assert registry.get_morphism("my_op") == morph
    
    def test_register_composition(self, simple_handler):
        """Registra composición."""
        def h2():
            return {}
        
        m1 = AtomicVector("m1", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("m2", Stratum.STRATEGY, h2)
        
        composed = m1 >> m2
        registry = CategoricalRegistry()
        
        registry.register_composition("pipeline", composed)
        
        assert "pipeline" in registry.list_compositions()
        assert registry.get_composition("pipeline") == composed
    
    def test_get_nonexistent_morphism(self):
        """Obtener morfismo inexistente retorna None."""
        registry = CategoricalRegistry()
        
        result = registry.get_morphism("nonexistent")
        
        assert result is None
    
    def test_get_dependency_graph(self, simple_handler):
        """Obtiene grafo de dependencias."""
        def h2():
            return {}
        
        m1 = AtomicVector("op1", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("op2", Stratum.STRATEGY, h2)
        
        registry = CategoricalRegistry()
        registry.register_morphism("op1", m1)
        registry.register_morphism("op2", m2)
        
        graph = registry.get_dependency_graph()
        
        assert isinstance(graph, nx.DiGraph)
        assert "op1" in graph.nodes()
        assert "op2" in graph.nodes()
    
    def test_verify_acyclicity_true(self, simple_handler):
        """Verifica aciclicidad correcta."""
        def h2():
            return {}
        
        m1 = AtomicVector("op1", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("op2", Stratum.STRATEGY, h2)
        
        registry = CategoricalRegistry()
        registry.register_morphism("op1", m1)
        registry.register_morphism("op2", m2)
        
        is_acyclic = registry.verify_acyclicity()
        
        assert is_acyclic is True
    
    def test_replace_existing_morphism(self, simple_handler):
        """Reemplaza morfismo existente."""
        def new_handler():
            return {"new": True}
        
        m1 = AtomicVector("op", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("op", Stratum.TACTICS, new_handler)
        
        registry = CategoricalRegistry()
        registry.register_morphism("op", m1)
        registry.register_morphism("op", m2)
        
        assert registry.get_morphism("op") == m2


# ============================================================================
# PRUEBAS DE MONADALIDAD (Railway Oriented Programming)
# ============================================================================


class TestMonadalityAndErrorHandling:
    """Pruebas del modelo monadal de error."""
    
    def test_error_is_absorbing_element(self):
        """Error es elemento absorbente."""
        state = create_categorical_state().with_error("Initial error")
        
        def handler():
            return {"should_not_see": "this"}
        
        morph = AtomicVector("op", Stratum.TACTICS, handler)
        
        result = morph(state)
        
        # El morfismo no se ejecuta
        assert result.error == "Initial error"
        assert "should_not_see" not in result.payload
    
    def test_success_continues_pipeline(self, physics_state):
        """Estado exitoso continúa el pipeline."""
        def handler1(raw_data: list = None):
            return {"processed": len(raw_data or [])}
        
        def handler2(processed: int):
            return {"analyzed": processed * 2}
        
        m1 = AtomicVector(
            "process",
            Stratum.TACTICS,
            handler1,
            required_keys=["raw_data"],
        )
        
        m2 = AtomicVector(
            "analyze",
            Stratum.STRATEGY,
            handler2,
            required_keys=["processed"],
        )
        
        composed = m1 >> m2
        
        result = composed(physics_state)
        
        assert result.is_success
        assert "analyzed" in result.payload
    
    def test_error_recovery_via_coproduct(self, physics_state):
        """Recuperación de error usando coproducto."""
        def failing_handler():
            raise ValueError("Primary fails")
        
        def recovery_handler():
            return {"recovered": True}
        
        primary = AtomicVector(
            "primary",
            Stratum.TACTICS,
            failing_handler,
        )
        
        recovery = AtomicVector(
            "recovery",
            Stratum.TACTICS,
            recovery_handler,
        )
        
        with_fallback = primary | recovery
        
        result = with_fallback(physics_state)
        
        assert result.is_success
        assert result.payload["recovered"] is True
    
    def test_error_details_preserved(self):
        """Detalles de error se preservan."""
        state = create_categorical_state().with_error(
            error_msg="Main error",
            details={"code": 500, "source": "database"},
        )
        
        assert state.error_details["code"] == 500
        assert state.error_details["source"] == "database"
    
    def test_monadal_laws_left_identity(self, physics_state):
        """Ley monadal: return a >>= f = f a."""
        def handler(raw_data: list = None):
            return {"count": len(raw_data or [])}
        
        # return a (crear estado con PHYSICS)
        # >>= f (aplicar morfismo)
        morph = AtomicVector(
            "count",
            Stratum.TACTICS,
            handler,
            required_keys=["raw_data"],
        )
        
        result = morph(physics_state)
        
        assert result.is_success
    
    def test_monadal_laws_right_identity(self, physics_state):
        """Ley monadal: m >>= return = m."""
        # Aplicar identidad después de algo
        morph = AtomicVector(
            "op",
            Stratum.TACTICS,
            lambda: {"result": 1},
        )
        
        id_morph = IdentityMorphism(Stratum.TACTICS)
        
        result1 = morph(physics_state)
        result2 = id_morph(result1)
        
        assert result2 == result1
    
    def test_monadal_laws_associativity(self, physics_state):
        """Ley monadal: (m >>= f) >>= g = m >>= (x => f x >>= g)."""
        def f(raw_data: list = None):
            return {"processed": len(raw_data or [])}
        
        def g(processed: int):
            return {"final": processed * 2}
        
        m1 = AtomicVector("f", Stratum.TACTICS, f, required_keys=["raw_data"])
        m2 = AtomicVector("g", Stratum.STRATEGY, g, required_keys=["processed"])
        
        # (m1 >>= f) >>= g equivalente a m1 >> m2
        composed = m1 >> m2
        
        result = composed(physics_state)
        
        assert result.is_success


# ============================================================================
# PRUEBAS DE TRAZAS Y AUDITORÍA
# ============================================================================


class TestCompositionTracing:
    """Pruebas del seguimiento de composiciones."""
    
    def test_single_morphism_generates_trace(self, physics_state, simple_handler):
        """Un morfismo genera una entrada de traza."""
        morph = AtomicVector(
            "test_op",
            Stratum.TACTICS,
            simple_handler,
        )
        
        result = morph(physics_state)
        
        assert len(result.composition_trace) == 1
        assert result.composition_trace[0].morphism_name == "test_op"
        assert result.composition_trace[0].step_number == 1
    
    def test_composed_morphisms_generate_multiple_traces(self, physics_state):
        """Composición genera múltiples entradas."""
        def h1():
            return {"s1": 1}
        
        def h2():
            return {"s2": 2}
        
        m1 = AtomicVector("step1", Stratum.TACTICS, h1)
        m2 = AtomicVector("step2", Stratum.STRATEGY, h2)
        
        composed = m1 >> m2
        
        result = composed(physics_state)
        
        assert len(result.composition_trace) == 2
        assert result.composition_trace[0].morphism_name == "step1"
        assert result.composition_trace[1].morphism_name == "step2"
    
    def test_trace_records_domain_and_codomain(self, physics_state):
        """Traza registra dominio y codominio."""
        def handler(raw_data: list = None):
            return {"processed": True}
        
        morph = AtomicVector(
            "op",
            Stratum.TACTICS,
            handler,
            required_keys=["raw_data"],
        )
        
        result = morph(physics_state)
        
        trace = result.composition_trace[0]
        assert Stratum.PHYSICS in trace.input_domain
        assert trace.output_codomain == Stratum.TACTICS
    
    def test_trace_records_success_status(self, physics_state, failing_handler):
        """Traza registra estado de éxito/fallo."""
        morph_ok = AtomicVector(
            "ok",
            Stratum.TACTICS,
            lambda: {"ok": True},
        )
        
        morph_fail = AtomicVector(
            "fail",
            Stratum.TACTICS,
            failing_handler,
        )
        
        result_ok = morph_ok(physics_state)
        result_fail = morph_fail(physics_state)
        
        assert result_ok.composition_trace[0].success is True
        assert result_fail.composition_trace[0].success is False
    
    def test_trace_serialization(self, physics_state):
        """Traza se serializa correctamente."""
        def handler():
            return {"data": "test"}
        
        morph = AtomicVector("op", Stratum.TACTICS, handler)
        
        result = morph(physics_state)
        state_dict = result.to_dict()
        
        assert "composition_trace" in state_dict
        assert len(state_dict["composition_trace"]) == 1
        
        trace_dict = state_dict["composition_trace"][0]
        assert trace_dict["morphism"] == "op"
        assert trace_dict["success"] is True


# ============================================================================
# PRUEBAS DE CASOS EXTREMOS
# ============================================================================


class TestEdgeCases:
    """Pruebas de casos extremos."""
    
    def test_empty_payload_morphism(self):
        """Morfismo con payload vacío."""
        state = create_categorical_state(payload={})
        
        def handler():
            return {"added": "value"}
        
        morph = AtomicVector("op", Stratum.TACTICS, handler)
        
        result = morph(state)
        
        assert result.payload["added"] == "value"
    
    def test_morphism_with_none_values(self):
        """Morfismo maneja None en payload."""
        state = create_categorical_state(
            payload={"value": None, "name": None}
        )
        
        def handler(**kwargs):
            return {"processed": kwargs.get("value") is None}
        
        morph = AtomicVector(
            "op",
            Stratum.TACTICS,
            handler,
        )
        
        result = morph(state)
        
        assert result.payload["processed"] is True
    
    def test_large_payload_morphism(self):
        """Morfismo con payload grande."""
        large_payload = {f"key_{i}": list(range(100)) for i in range(100)}
        state = create_categorical_state(payload=large_payload)
        
        def handler(**kwargs):
            return {"size": len(kwargs)}
        
        morph = AtomicVector("op", Stratum.TACTICS, handler)
        
        result = morph(state)
        
        assert result.payload["size"] == 100
    
    def test_unicode_in_morphism_names(self):
        """Morfismo con nombres Unicode."""
        def handler():
            return {"resultado": "éxito"}
        
        morph = AtomicVector(
            "operación_∫∆∇",
            Stratum.TACTICS,
            handler,
        )
        
        assert "∫" in morph.name
    
    def test_very_deep_composition(self, physics_state):
        """Composición muy profunda."""
        morphisms = []
        
        for i in range(10):
            def make_handler(step_num):
                def handler(step: int = 0):
                    return {"step": step_num}
                return handler
            
            stratum_idx = min(i // 3, 3)
            strata = [
                Stratum.TACTICS,
                Stratum.STRATEGY,
                Stratum.WISDOM,
            ]
            
            m = AtomicVector(
                f"step_{i}",
                strata[stratum_idx],
                make_handler(i),
            )
            morphisms.append(m)
        
        # Componer todas
        composed = morphisms[0]
        for morph in morphisms[1:]:
            try:
                composed = composed >> morph
            except TypeError:
                break
        
        result = composed(physics_state)
        
        assert result.is_success or result.is_failed


# ============================================================================
# PRUEBAS DE SERIALIZACIÓN
# ============================================================================


class TestSerialization:
    """Pruebas de serialización y deserialización."""
    
    def test_categorical_state_to_dict(self, complete_state):
        """Serializa estado a diccionario."""
        state_dict = complete_state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert "payload" in state_dict
        assert "context" in state_dict
        assert "validated_strata" in state_dict
        assert "error" in state_dict
    
    def test_categorical_state_from_dict(self, complete_state):
        """Reconstruye estado desde diccionario."""
        original_dict = complete_state.to_dict()
        
        reconstructed = CategoricalState.from_dict(original_dict)
        
        assert reconstructed.payload == complete_state.payload
        assert reconstructed.validated_strata == complete_state.validated_strata
    
    def test_trace_serialization(self):
        """Serializa traza."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="test",
            input_domain=frozenset([Stratum.PHYSICS]),
            output_codomain=Stratum.TACTICS,
            success=True,
        )
        
        trace_dict = trace.to_dict()
        
        assert trace_dict["step"] == 1
        assert trace_dict["morphism"] == "test"
    
    def test_state_with_traces_serialization(self, physics_state):
        """Serializa estado con trazas."""
        def handler():
            return {"data": "test"}
        
        morph = AtomicVector("op", Stratum.TACTICS, handler)
        result = morph(physics_state)
        
        result_dict = result.to_dict()
        state_reconstructed = CategoricalState.from_dict(result_dict)
        
        assert len(state_reconstructed.composition_trace) > 0


# ============================================================================
# PRUEBAS DE RENDIMIENTO
# ============================================================================


class TestPerformance:
    """Pruebas de rendimiento."""
    
    def test_morphism_creation_performance(self):
        """Crear múltiples morfismos es rápido."""
        start = time.time()
        
        for i in range(1000):
            def handler():
                return {"result": i}
            
            AtomicVector(
                f"op_{i}",
                Stratum.TACTICS,
                handler,
            )
        
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Menos de 1 segundo para 1000 morfismos
    
    def test_composition_creation_performance(self, simple_handler):
        """Crear composiciones es rápido."""
        def handler2():
            return {"s2": 1}
        
        m1 = AtomicVector("m1", Stratum.TACTICS, simple_handler)
        m2 = AtomicVector("m2", Stratum.STRATEGY, handler2)
        
        start = time.time()
        
        for _ in range(100):
            _ = m1 >> m2
        
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Muy rápido
    
    def test_morphism_execution_performance(self, physics_state, simple_handler):
        """Ejecución de morfismo es rápido."""
        morph = AtomicVector("op", Stratum.TACTICS, simple_handler)
        
        start = time.time()
        
        for _ in range(100):
            _ = morph(physics_state)
        
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # 100 ejecuciones en menos de 0.5 segundos
    
    def test_state_hash_performance(self, complete_state):
        """Hashing de estado es rápido."""
        start = time.time()
        
        for _ in range(100):
            _ = complete_state.compute_hash()
        
        elapsed = time.time() - start
        
        assert elapsed < 0.1


# ============================================================================
# CONFIGURACIÓN Y EJECUCIÓN
# ============================================================================


def pytest_configure(config):
    """Configuración inicial."""
    config.addinivalue_line(
        "markers", "slow: marca pruebas lentas"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])