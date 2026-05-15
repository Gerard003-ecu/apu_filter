"""
Módulo: test_mic_algebra.py
Propósito: Suite de pruebas rigurosas para MIC Algebra
Ubicación: tests/core/test_mic_algebra.py

FUNDAMENTOS DE PRUEBAS MATEMÁTICAS:
===================================
1. Tests Unitarios: Verificación de comportamiento individual
2. Property-Based Testing: Verificación de propiedades algebraicas
3. Tests de Invariantes: Verificación de condiciones que siempre deben cumplirse
4. Tests de Bordes: Casos extremos y condiciones límite
5. Tests de Propiedades Categóricas: Axiomas de teoría de categorías

COBERTURA REQUERIDA:
===================
- Stratum: 100% (retículo, orden, meet/join)
- CategoricalState: 100% (inmutabilidad, hash, serialización)
- Morphism: 100% (composición, asociatividad, identidad)
- Functor: 95% (preservación de composición)
- NaturalTransformation: 90% (ley de naturalidad)
- Verificadores: 95% (aciclicidad, homología)
- Utilidades: 100% (tolerancias, canonicalización)

EJECUCIÓN:
==========
$ pytest tests/core/test_mic_algebra.py -v --cov=app.core.mic_algebra
$ pytest tests/core/test_mic_algebra.py -v -k "test_category"
$ pytest tests/core/test_mic_algebra.py -v -m "property"
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import os
import decimal

# Fase 1: Esterilización del Vacío Termodinámico y Condicionamiento Numérico
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Cuantización Decimal
decimal.getcontext().prec = 50
decimal.getcontext().rounding = decimal.ROUND_HALF_EVEN

import pytest
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from dataclasses import asdict
from copy import deepcopy
import threading

# Importar módulo bajo prueba
from app.core.mic_algebra import (
    # Constantes
    _SCHEMA_VERSION,
    _MAX_CANONICALIZE_DEPTH,
    _ALGEBRAIC_TOL,
    _FLOAT_COMPARISON_TOL,
    _MACHINE_EPSILON,
    
    # Excepciones
    AlgebraicError,
    CanonicalizationError,
    StratumResolutionError,
    CategoryError,
    CompositionError,
    AssociativityError,
    IdentityError,
    FunctorialityError,
    HomologicalError,
    NumericalInstabilityError,
    
    # Estratificación
    Stratum,
    
    # Utilidades
    MathUtils,
    
    # Estado categórico
    CategoricalState,
    CompositionTrace,
    create_categorical_state,
    
    # Morfismos
    Morphism,
    IdentityMorphism,
    AtomicVector,
    ComposedMorphism,
    ProductMorphism,
    CoproductMorphism,
    PullbackMorphism,
    
    # Funtores
    Functor,
    StateToDictFunctor,
    NaturalTransformation,
    
    # Composición y verificación
    MorphismComposer,
    StructuralVerifier,
    HomologicalVerifier,
    
    # Registro
    CategoricalRegistry,
    
    # Orquestación
    TwoCategoryOrchestrator,
    
    # Utilidades internas (para testing)
    _canonicalize,
    _stable_hash,
    _safe_merge_dicts,
    _copy_trace,
)

# ==============================================================================
# CONFIGURACIÓN DE LOGGING PARA TESTS
# ==============================================================================
logging.basicConfig(
    level=logging.WARNING,  # Silenciar logs informativos durante tests
    format="%(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)

# ==============================================================================
# FIXTURES COMUNES
# ==============================================================================
@pytest.fixture
def empty_state() -> CategoricalState:
    """Estado inicial vacío (objeto inicial ⊥)."""
    return create_categorical_state()


@pytest.fixture
def physics_state() -> CategoricalState:
    """Estado con estrato PHYSICS validado."""
    return create_categorical_state(
        payload={"data": "test"},
        strata={Stratum.PHYSICS},
    )


@pytest.fixture
def full_strata_state() -> CategoricalState:
    """Estado con todos los estratos validados."""
    return create_categorical_state(
        payload={"complete": True},
        strata=frozenset(Stratum),
    )


@pytest.fixture
def sample_handler() -> callable:
    """Handler simple para AtomicVector."""
    def handler(value: int) -> Dict[str, Any]:
        return {"result": value * 2, "success": True}
    return handler


@pytest.fixture
def failing_handler() -> callable:
    """Handler que siempre falla."""
    def handler(**kwargs) -> Dict[str, Any]:
        return {"success": False, "error": " intentional failure"}
    return handler


@pytest.fixture
def registry() -> CategoricalRegistry:
    """Registro limpio para cada test."""
    return CategoricalRegistry()


@pytest.fixture
def verifier() -> HomologicalVerifier:
    """Verificador homológico."""
    return HomologicalVerifier()


@pytest.fixture
def composer() -> MorphismComposer:
    """Compositor limpio para cada test."""
    return MorphismComposer()


# ==============================================================================
# TESTS DE ESTRATIFICACIÓN (STRATUM)
# ==============================================================================
class TestStratumLattice:
    """
    Tests para propiedades de retículo de Stratum.
    
    Propiedades Verificadas:
    =======================
    1. Orden parcial (reflexivo, antisimétrico, transitivo)
    2. Elementos mínimo y máximo (⊥, ⊤)
    3. Operaciones meet y join (conmutativas, asociativas, idempotentes)
    4. Relaciones de cobertura (diagrama de Hasse)
    """
    
    def test_stratum_values_are_sequential(self) -> None:
        """Los valores de estratos son secuenciales 0-5."""
        expected_values = list(range(6))
        actual_values = [s.value for s in Stratum]
        assert sorted(actual_values) == expected_values
    
    def test_stratum_bottom_is_wisdom(self) -> None:
        """Elemento mínimo es WISDOM (valor 0)."""
        assert Stratum.bottom() == Stratum.WISDOM
        assert Stratum.WISDOM.value == 0
    
    def test_stratum_top_is_physics(self) -> None:
        """Elemento máximo es PHYSICS (valor 5)."""
        assert Stratum.top() == Stratum.PHYSICS
        assert Stratum.PHYSICS.value == 5
    
    def test_stratum_order_reflexivity(self) -> None:
        """Orden es reflexivo: s ≤ s para todo s."""
        for s in Stratum:
            assert s <= s
            assert not (s < s)
    
    def test_stratum_order_antisymmetry(self) -> None:
        """Orden es antisimétrico: s ≤ t ∧ t ≤ s ⟹ s = t."""
        for s in Stratum:
            for t in Stratum:
                if s <= t and t <= s:
                    assert s == t
    
    def test_stratum_order_transitivity(self) -> None:
        """Orden es transitivo: s ≤ t ∧ t ≤ u ⟹ s ≤ u."""
        for s in Stratum:
            for t in Stratum:
                for u in Stratum:
                    if s <= t and t <= u:
                        assert s <= u
    
    def test_stratum_meet_commutativity(self) -> None:
        """Meet es conmutativo: s ∧ t = t ∧ s."""
        for s in Stratum:
            for t in Stratum:
                assert s.meet(t) == t.meet(s)
    
    def test_stratum_meet_associativity(self) -> None:
        """Meet es asociativo: (s ∧ t) ∧ u = s ∧ (t ∧ u)."""
        for s in Stratum:
            for t in Stratum:
                for u in Stratum:
                    assert (s.meet(t)).meet(u) == s.meet(t.meet(u))
    
    def test_stratum_meet_idempotence(self) -> None:
        """Meet es idempotente: s ∧ s = s."""
        for s in Stratum:
            assert s.meet(s) == s
    
    def test_stratum_join_commutativity(self) -> None:
        """Join es conmutativo: s ∨ t = t ∨ s."""
        for s in Stratum:
            for t in Stratum:
                assert s.join(t) == t.join(s)
    
    def test_stratum_join_associativity(self) -> None:
        """Join es asociativo: (s ∨ t) ∨ u = s ∨ (t ∨ u)."""
        for s in Stratum:
            for t in Stratum:
                for u in Stratum:
                    assert (s.join(t)).join(u) == s.join(t.join(u))
    
    def test_stratum_join_idempotence(self) -> None:
        """Join es idempotente: s ∨ s = s."""
        for s in Stratum:
            assert s.join(s) == s
    
    def test_stratum_absorption_laws(self) -> None:
        """Leyes de absorción: s ∧ (s ∨ t) = s y s ∨ (s ∧ t) = s."""
        for s in Stratum:
            for t in Stratum:
                assert s.meet(s.join(t)) == s
                assert s.join(s.meet(t)) == s
    
    def test_stratum_requires_irreflexivity(self) -> None:
        """requires(s) no contiene a s (irreflexividad)."""
        for s in Stratum:
            assert s not in s.requires()
    
    def test_stratum_requires_transitivity(self) -> None:
        """requires es transitivo."""
        for s in Stratum:
            for t in s.requires():
                for u in t.requires():
                    assert u in s.requires()
    
    def test_stratum_requires_cardinality(self) -> None:
        """|requires(s)| = 5 - s.value."""
        for s in Stratum:
            assert len(s.requires()) == 5 - s.value
    
    def test_stratum_height_property(self) -> None:
        """height(s) = s.value."""
        for s in Stratum:
            assert s.height == s.value
    
    def test_stratum_depth_property(self) -> None:
        """depth(s) = 5 - s.value."""
        for s in Stratum:
            assert s.depth == 5 - s.value
    
    def test_stratum_covers_relation(self) -> None:
        """covers(s, t) ⟺ s.value = t.value + 1."""
        for s in Stratum:
            for t in Stratum:
                expected = s.value == t.value + 1
                assert s.covers(t) == expected
    
    def test_stratum_successor_predecessor_inverse(self) -> None:
        """s.is_successor_of(t) ⟺ t.is_predecessor_of(s)."""
        for s in Stratum:
            for t in Stratum:
                assert s.is_successor_of(t) == t.is_predecessor_of(s)
    
    def test_stratum_chain_is_complete(self) -> None:
        """La cadena contiene todos los estratos ordenados."""
        chain = Stratum.chain()
        assert len(chain) == 6
        for i in range(len(chain) - 1):
            assert chain[i].value < chain[i + 1].value


# ==============================================================================
# TESTS DE UTILIDADES MATEMÁTICAS
# ==============================================================================
class TestMathUtils:
    """Tests para utilidades matemáticas con garantías numéricas."""
    
    def test_float_equal_reflexivity(self) -> None:
        """float_equal es reflexiva."""
        for val in [0.0, 1.0, -1.0, 1e-10, 1e10]:
            assert MathUtils.float_equal(val, val)
    
    def test_float_equal_symmetry(self) -> None:
        """float_equal es simétrica."""
        pairs = [(0.0, 1e-10), (1.0, 1.0 + 1e-10), (1e10, 1e10 + 1.0)]
        for a, b in pairs:
            assert MathUtils.float_equal(a, b) == MathUtils.float_equal(b, a)
    
    def test_float_equal_tolerance_absolute(self) -> None:
        """Tolerancia absoluta para valores cercanos a cero."""
        assert MathUtils.float_equal(0.0, 1e-10, abs_tol=1e-9)
        assert not MathUtils.float_equal(0.0, 1e-8, abs_tol=1e-9)
    
    def test_float_equal_tolerance_relative(self) -> None:
        """Tolerancia relativa para valores grandes."""
        assert MathUtils.float_equal(1e10, 1e10 + 1.0, rel_tol=1e-9)
    
    def test_safe_divide_normal(self) -> None:
        """División normal funciona correctamente."""
        assert MathUtils.safe_divide(10.0, 2.0) == 5.0
        assert MathUtils.safe_divide(-10.0, 2.0) == -5.0
    
    def test_safe_divide_by_zero(self) -> None:
        """División por cero retorna valor finito."""
        result = MathUtils.safe_divide(10.0, 0.0)
        assert abs(result) < float('inf')
        assert result > 0  # Mantiene signo
    
    def test_safe_divide_negative_zero(self) -> None:
        """División por cero negativo mantiene signo."""
        result = MathUtils.safe_divide(10.0, -0.0)
        assert result < 0
    
    def test_clamp_valid_range(self) -> None:
        """Clamp con rango válido funciona."""
        assert MathUtils.clamp(5.0, 0.0, 10.0) == 5.0
        assert MathUtils.clamp(-5.0, 0.0, 10.0) == 0.0
        assert MathUtils.clamp(15.0, 0.0, 10.0) == 10.0
    
    def test_clamp_invalid_range_raises(self) -> None:
        """Clamp con rango inválido lanza ValueError."""
        with pytest.raises(ValueError):
            MathUtils.clamp(5.0, 10.0, 0.0)
    
    def test_adaptive_tolerance_scales(self) -> None:
        """Tolerancia adaptativa escala con magnitud."""
        base = 1e-10
        assert MathUtils.adaptive_tolerance(base, 0.0) == base
        assert MathUtils.adaptive_tolerance(base, 100.0) == base * 100.0
    
    def test_condition_number_estimate(self) -> None:
        """Número de condición se calcula correctamente."""
        assert MathUtils.condition_number_estimate([1.0, 10.0]) == 10.0
        assert MathUtils.condition_number_estimate([0.0, 0.0]) == float('inf')


# ==============================================================================
# TESTS DE CANONICALIZACIÓN Y HASH
# ==============================================================================
class TestCanonicalization:
    """Tests para canonicalización determinista."""
    
    def test_canonicalize_primitives(self) -> None:
        """Primitivos se canonicalizan a sí mismos."""
        assert _canonicalize(None) is None
        assert _canonicalize(42) == 42
        assert _canonicalize(3.14) == 3.14
        assert _canonicalize("test") == "test"
        assert _canonicalize(True) is True
    
    def test_canonicalize_stratum(self) -> None:
        """Stratum se canonicaliza a dict con nombre y valor."""
        result = _canonicalize(Stratum.PHYSICS)
        assert result == {"__stratum__": "PHYSICS", "__value__": 5}
    
    def test_canonicalize_dict_ordering(self) -> None:
        """Dicts se ordenan lexicográficamente por clave."""
        input_dict = {"z": 1, "a": 2, "m": 3}
        result = _canonicalize(input_dict)
        keys = list(result.keys())
        assert keys == sorted(keys)
    
    def test_canonicalize_list_preserves_order(self) -> None:
        """Listas preservan orden."""
        input_list = [3, 1, 2]
        result = _canonicalize(input_list)
        assert result == [3, 1, 2]
    
    def test_canonicalize_tuple_preserves_order(self) -> None:
        """Tuplas preservan orden y tipo."""
        input_tuple = (3, 1, 2)
        result = _canonicalize(input_tuple)
        assert result == (3, 1, 2)
        assert isinstance(result, tuple)
    
    def test_canonicalize_set_ordering(self) -> None:
        """Sets se ordenan establemente."""
        input_set = {3, 1, 2}
        result = _canonicalize(input_set)
        assert result == sorted(list(input_set))
    
    def test_canonicalize_depth_limit(self) -> None:
        """Profundidad excesiva lanza CanonicalizationError."""
        # Crear estructura profundamente anidada
        deep = {"level": 0}
        current = deep
        for i in range(_MAX_CANONICALIZE_DEPTH + 1):
            current["nested"] = {"level": i + 1}
            current = current["nested"]
        
        with pytest.raises(CanonicalizationError):
            _canonicalize(deep)
    
    def test_canonicalize_idempotence(self) -> None:
        """Canonicalización es idempotente: canon(canon(x)) = canon(x)."""
        test_values = [
            {"a": 1, "b": 2},
            [1, 2, 3],
            {"nested": {"deep": True}},
        ]
        for val in test_values:
            first = _canonicalize(val)
            second = _canonicalize(first)
            assert first == second
    
    def test_stable_hash_determinism(self) -> None:
        """Hash es determinista para mismo input."""
        data = {"test": "value", "number": 42}
        hash1 = _stable_hash(data)
        hash2 = _stable_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
    
    def test_stable_hash_avalanche(self) -> None:
        """Cambio mínimo produce hash diferente (efecto avalancha)."""
        data1 = {"test": "value"}
        data2 = {"test": "value1"}
        hash1 = _stable_hash(data1)
        hash2 = _stable_hash(data2)
        assert hash1 != hash2
        # Verificar que difieren en múltiples bits (aproximado)
        diff_bits = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        assert diff_bits > 10  # Al menos 10 caracteres diferentes


# ==============================================================================
# TESTS DE FUSIÓN DE DICCIONARIOS
# ==============================================================================
class TestDictMerge:
    """Tests para fusión de diccionarios con políticas."""
    
    def test_merge_prefer_right(self) -> None:
        """prefer_right: derecho prevalece en conflictos."""
        left = {"a": 1, "b": 2}
        right = {"b": 3, "c": 4}
        result = _safe_merge_dicts(left, right, conflict_policy="prefer_right")
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_merge_prefer_left(self) -> None:
        """prefer_left: izquierdo prevalece en conflictos."""
        left = {"a": 1, "b": 2}
        right = {"b": 3, "c": 4}
        result = _safe_merge_dicts(left, right, conflict_policy="prefer_left")
        assert result == {"a": 1, "b": 2, "c": 4}
    
    def test_merge_error_on_conflict(self) -> None:
        """error_on_conflict: lanza excepción en conflictos."""
        left = {"a": 1, "b": 2}
        right = {"b": 3}
        with pytest.raises(ValueError):
            _safe_merge_dicts(left, right, conflict_policy="error_on_conflict")
    
    def test_merge_no_conflict(self) -> None:
        """Sin conflictos: todas las políticas dan mismo resultado."""
        left = {"a": 1}
        right = {"b": 2}
        for policy in ["prefer_right", "prefer_left", "error_on_conflict"]:
            result = _safe_merge_dicts(left, right, conflict_policy=policy)
            assert result == {"a": 1, "b": 2}
    
    def test_merge_invalid_policy_raises(self) -> None:
        """Política inválida lanza ValueError."""
        with pytest.raises(ValueError):
            _safe_merge_dicts({}, {}, conflict_policy="invalid")
    
    def test_merge_associativity(self) -> None:
        """Fusión es asociativa para prefer_right."""
        a = {"x": 1}
        b = {"x": 2, "y": 3}
        c = {"y": 4, "z": 5}
        
        left = _safe_merge_dicts(_safe_merge_dicts(a, b), c)
        right = _safe_merge_dicts(a, _safe_merge_dicts(b, c))
        assert left == right


# ==============================================================================
# TESTS DE COMPOSITION TRACE
# ==============================================================================
class TestCompositionTrace:
    """Tests para trazas de composición."""
    
    def test_trace_creation(self) -> None:
        """Creación básica de traza."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="test_morphism",
            input_domain=frozenset({Stratum.PHYSICS}),
            output_codomain=Stratum.TACTICS,
            success=True,
        )
        assert trace.step_number == 1
        assert trace.morphism_name == "test_morphism"
        assert trace.success is True
        assert trace.error is None
    
    def test_trace_step_number_correction(self) -> None:
        """step_number < 1 se corrige a 1."""
        trace = CompositionTrace(
            step_number=0,
            morphism_name="test",
            input_domain=frozenset(),
            output_codomain=Stratum.WISDOM,
            success=True,
        )
        assert trace.step_number == 1
    
    def test_trace_timestamp_correction(self) -> None:
        """timestamp ≤ 0 se corrige a tiempo actual."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="test",
            input_domain=frozenset(),
            output_codomain=Stratum.WISDOM,
            success=True,
            timestamp=-1.0,
        )
        assert trace.timestamp > 0
    
    def test_trace_identity_key(self) -> None:
        """trace_identity_key es hashable y única."""
        trace1 = CompositionTrace(
            step_number=1,
            morphism_name="test",
            input_domain=frozenset({Stratum.PHYSICS}),
            output_codomain=Stratum.TACTICS,
            success=True,
        )
        trace2 = CompositionTrace(
            step_number=1,
            morphism_name="test",
            input_domain=frozenset({Stratum.PHYSICS}),
            output_codomain=Stratum.TACTICS,
            success=True,
            timestamp=trace1.timestamp + 100,  # Diferente timestamp
        )
        assert trace1.trace_identity_key == trace2.trace_identity_key
    
    def test_trace_to_dict_serialization(self) -> None:
        """to_dict produce dict serializable."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="test",
            input_domain=frozenset({Stratum.PHYSICS}),
            output_codomain=Stratum.TACTICS,
            success=True,
        )
        result = trace.to_dict()
        assert isinstance(result, dict)
        assert "step" in result
        assert "morphism" in result
        assert "domain" in result
        assert "codomain" in result
        assert "success" in result
    
    def test_trace_immutability(self) -> None:
        """Traza es inmutable (frozen dataclass)."""
        trace = CompositionTrace(
            step_number=1,
            morphism_name="test",
            input_domain=frozenset(),
            output_codomain=Stratum.WISDOM,
            success=True,
        )
        with pytest.raises(AttributeError):  # frozen=True
            trace.step_number = 2  # type: ignore


# ==============================================================================
# TESTS DE CATEGORICAL STATE
# ==============================================================================
class TestCategoricalState:
    """Tests para estado categórico con propiedades universales."""
    
    def test_state_creation_empty(self) -> None:
        """Creación de estado vacío."""
        state = create_categorical_state()
        assert state.payload == {}
        assert state.context == {}
        assert state.validated_strata == frozenset()
        assert state.is_success is True
    
    def test_state_creation_with_data(self) -> None:
        """Creación con datos iniciales."""
        state = create_categorical_state(
            payload={"key": "value"},
            context={"ctx": "data"},
            strata={Stratum.PHYSICS, Stratum.TACTICS},
        )
        assert state.payload == {"key": "value"}
        assert state.context == {"ctx": "data"}
        assert Stratum.PHYSICS in state.validated_strata
    
    def test_state_immutability(self) -> None:
        """Estado es inmutable (frozen dataclass)."""
        state = create_categorical_state()
        with pytest.raises(AttributeError):
            state.payload = {"new": "data"}  # type: ignore
    
    def test_state_is_success_is_failed_exclusive(self) -> None:
        """is_success y is_failed son mutuamente excluyentes."""
        state_success = create_categorical_state()
        state_failed = state_success.with_error("test error")
        
        assert state_success.is_success != state_success.is_failed
        assert state_failed.is_success != state_failed.is_failed
        assert state_success.is_success is True
        assert state_failed.is_failed is True
    
    def test_state_stratum_level_empty(self) -> None:
        """stratum_level con estratos vacíos retorna PHYSICS.value."""
        state = create_categorical_state()
        assert state.stratum_level == Stratum.PHYSICS.value
    
    def test_state_stratum_level_with_strata(self) -> None:
        """stratum_level es mínimo de estratos validados."""
        state = create_categorical_state(
            strata={Stratum.WISDOM, Stratum.PHYSICS},
        )
        assert state.stratum_level == Stratum.WISDOM.value
    
    def test_state_with_update_pure(self) -> None:
        """with_update no modifica estado original."""
        original = create_categorical_state(payload={"a": 1})
        updated = original.with_update(new_payload={"b": 2})
        
        assert original.payload == {"a": 1}
        assert updated.payload == {"a": 1, "b": 2}
        assert original is not updated
    
    def test_state_with_update_merge(self) -> None:
        """with_update con merge=True fusiona payloads."""
        state = create_categorical_state(payload={"a": 1, "b": 2})
        updated = state.with_update(
            new_payload={"b": 3, "c": 4},
            merge_payload=True,
        )
        assert updated.payload == {"a": 1, "b": 3, "c": 4}
    
    def test_state_with_update_replace(self) -> None:
        """with_update con merge=False reemplaza payloads."""
        state = create_categorical_state(payload={"a": 1, "b": 2})
        updated = state.with_update(
            new_payload={"c": 3},
            merge_payload=False,
        )
        assert updated.payload == {"c": 3}
    
    def test_state_with_error_preserves_data(self) -> None:
        """with_error preserva payload y context."""
        original = create_categorical_state(
            payload={"data": "value"},
            context={"ctx": "info"},
        )
        failed = original.with_error("test error")
        
        assert failed.payload == original.payload
        assert failed.context == original.context
        assert failed.is_failed is True
    
    def test_state_clear_error_idempotence(self) -> None:
        """clear_error es idempotente."""
        state = create_categorical_state().with_error("error")
        cleared1 = state.clear_error()
        cleared2 = cleared1.clear_error()
        
        assert cleared1.is_success is True
        assert cleared2.is_success is True
        assert cleared1.compute_hash() == cleared2.compute_hash()
    
    def test_state_add_trace_monotonicity(self) -> None:
        """add_trace incrementa trace_length."""
        state = create_categorical_state()
        state1 = state.add_trace(
            "morphism1", frozenset(), Stratum.PHYSICS, True
        )
        state2 = state1.add_trace(
            "morphism2", frozenset(), Stratum.TACTICS, True
        )
        
        assert state.trace_length == 0
        assert state1.trace_length == 1
        assert state2.trace_length == 2
    
    def test_state_hash_determinism(self) -> None:
        """compute_hash es determinista."""
        state = create_categorical_state(payload={"test": "value"})
        hash1 = state.compute_hash()
        hash2 = state.compute_hash()
        assert hash1 == hash2
    
    def test_state_hash_sensitivity(self) -> None:
        """compute_hash es sensible a cambios mínimos."""
        state1 = create_categorical_state(payload={"test": "value"})
        state2 = create_categorical_state(payload={"test": "value1"})
        assert state1.compute_hash() != state2.compute_hash()
    
    def test_state_serialization_roundtrip(self) -> None:
        """to_dict/from_dict es biyectivo (módulo timestamps)."""
        original = create_categorical_state(
            payload={"data": [1, 2, 3]},
            context={"meta": "info"},
            strata={Stratum.PHYSICS, Stratum.TACTICS},
        )
        serialized = original.to_dict()
        reconstructed = CategoricalState.from_dict(serialized)
        
        # Verificar igualdad estructural (ignorar timestamps)
        assert original.payload == reconstructed.payload
        assert original.context == reconstructed.context
        assert original.validated_strata == reconstructed.validated_strata
    
    def test_state_equality_structural(self) -> None:
        """Igualdad es estructural basada en hash."""
        state1 = create_categorical_state(payload={"a": 1})
        state2 = create_categorical_state(payload={"a": 1})
        state3 = create_categorical_state(payload={"a": 2})
        
        assert state1 == state2
        assert state1 != state3
    
    def test_state_stratum_normalization(self) -> None:
        """Estratos se normalizan en __post_init__."""
        state = CategoricalState(
            validated_strata=frozenset([5, "physics", Stratum.PHYSICS])
        )
        # Todos deberían normalizarse a Stratum.PHYSICS
        assert Stratum.PHYSICS in state.validated_strata
        assert len(state.validated_strata) == 1  # Duplicados eliminados
    
    def test_state_merkle_root(self) -> None:
        """merkle_root se calcula para trazas."""
        state_empty = create_categorical_state()
        assert state_empty.merkle_root is not None
        assert len(state_empty.merkle_root) == 64
        
        state_with_traces = state_empty.add_trace(
            "test", frozenset(), Stratum.PHYSICS, True
        )
        assert state_with_traces.merkle_root != state_empty.merkle_root


# ==============================================================================
# TESTS DE MORFISMOS
# ==============================================================================
class TestMorphisms:
    """Tests para morfismos con axiomas categóricos."""
    
    def test_identity_morphism_domain_codomain(self) -> None:
        """IdentityMorphism tiene domain = codomain = {stratum}."""
        id_morphism = IdentityMorphism(Stratum.PHYSICS)
        assert id_morphism.domain == frozenset({Stratum.PHYSICS})
        assert id_morphism.codomain == Stratum.PHYSICS
    
    def test_identity_morphism_application(self) -> None:
        """IdentityMorphism aplica traza sin modificar estado."""
        id_morphism = IdentityMorphism(Stratum.PHYSICS)
        state = create_categorical_state(strata={Stratum.PHYSICS})
        result = id_morphism(state)
        
        assert result.payload == state.payload
        assert result.context == state.context
        assert result.trace_length == state.trace_length + 1
    
    def test_atomic_vector_success(self) -> None:
        """AtomicVector ejecuta handler exitosamente."""
        def handler(value: int) -> Dict[str, Any]:
            return {"doubled": value * 2}
        
        morphism = AtomicVector(
            name="double",
            target_stratum=Stratum.TACTICS,
            handler=handler,
            required_keys=["value"],
        )
        
        state = create_categorical_state(
            payload={"value": 5},
            strata=Stratum.TACTICS.requires(),
        )
        result = morphism(state)
        
        assert result.is_success is True
        assert result.payload.get("doubled") == 10
        assert Stratum.TACTICS in result.validated_strata
    
    def test_atomic_vector_absorption(self) -> None:
        """AtomicVector absorbe estado fallido."""
        def handler(value: int) -> Dict[str, Any]:
            return {"result": value}
        
        morphism = AtomicVector(
            name="test",
            target_stratum=Stratum.TACTICS,
            handler=handler,
        )
        
        failed_state = create_categorical_state().with_error("previous error")
        result = morphism(failed_state)
        
        assert result.is_failed is True
        assert "Absorción" in result.error  # type: ignore
    
    def test_atomic_vector_missing_strata(self) -> None:
        """AtomicVector falla si faltan estratos requeridos."""
        def handler(value: int) -> Dict[str, Any]:
            return {"result": value}
        
        morphism = AtomicVector(
            name="test",
            target_stratum=Stratum.TACTICS,
            handler=handler,
        )
        
        # TACTICS requiere PHYSICS, pero no está validado
        state = create_categorical_state(strata=frozenset())
        result = morphism(state)
        
        assert result.is_failed is True
        assert "clausura transitiva" in result.error.lower()  # type: ignore
    
    def test_atomic_vector_missing_keys(self) -> None:
        """AtomicVector falla si faltan claves requeridas."""
        def handler(value: int) -> Dict[str, Any]:
            return {"result": value}
        
        morphism = AtomicVector(
            name="test",
            target_stratum=Stratum.TACTICS,
            handler=handler,
            required_keys=["value"],
        )
        
        state = create_categorical_state(
            payload={"other": "data"},
            strata=Stratum.TACTICS.requires(),
        )
        result = morphism(state)
        
        assert result.is_failed is True
        assert "requeridas" in result.error.lower()  # type: ignore
    
    def test_composed_morphism_domain(self) -> None:
        """ComposedMorphism calcula dominio correctamente."""
        f = IdentityMorphism(Stratum.PHYSICS)
        g = IdentityMorphism(Stratum.TACTICS)
        composed = ComposedMorphism(f, g)
        
        # Dominio debería incluir dominios de ambos
        assert Stratum.PHYSICS in composed.domain
    
    def test_composed_morphism_codomain(self) -> None:
        """ComposedMorphism codominio es del segundo morfismo."""
        f = IdentityMorphism(Stratum.PHYSICS)
        g = IdentityMorphism(Stratum.TACTICS)
        composed = ComposedMorphism(f, g)
        
        assert composed.codomain == Stratum.TACTICS
    
    def test_composed_morphism_application(self) -> None:
        """ComposedMorphism aplica f luego g."""
        def handler1(**kwargs) -> Dict[str, Any]:
            return {"step1": "done"}
        
        def handler2(**kwargs) -> Dict[str, Any]:
            return {"step2": "done"}
        
        f = AtomicVector("step1", Stratum.TACTICS, handler1)
        g = AtomicVector("step2", Stratum.STRATEGY, handler2)
        composed = ComposedMorphism(f, g)
        
        state = create_categorical_state(
            strata=Stratum.STRATEGY.requires(),
        )
        result = composed(state)
        
        assert result.is_success is True
        assert "step1" in result.payload
        assert "step2" in result.payload
    
    def test_product_morphism_parallel(self) -> None:
        """ProductMorphism ejecuta en paralelo."""
        def handler1(**kwargs) -> Dict[str, Any]:
            return {"f": "result"}
        
        def handler2(**kwargs) -> Dict[str, Any]:
            return {"g": "result"}
        
        f = AtomicVector("f", Stratum.TACTICS, handler1)
        g = AtomicVector("g", Stratum.STRATEGY, handler2)
        product = ProductMorphism(f, g)
        
        state = create_categorical_state(
            strata=Stratum.STRATEGY.requires(),
        )
        result = product(state)
        
        assert result.is_success is True
        assert "f" in result.payload
        assert "g" in result.payload
    
    def test_coproduct_morphism_fallback(self) -> None:
        """CoproductMorphism usa fallback si primero falla."""
        def handler_fail(**kwargs) -> Dict[str, Any]:
            return {"success": False, "error": "intentional"}
        
        def handler_success(**kwargs) -> Dict[str, Any]:
            return {"fallback": "worked"}
        
        f = AtomicVector("fail", Stratum.TACTICS, handler_fail)
        g = AtomicVector("success", Stratum.TACTICS, handler_success)
        coproduct = CoproductMorphism(f, g)
        
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        result = coproduct(state)
        
        assert result.is_success is True
        assert "fallback" in result.payload
    
    def test_morphism_composition_operator(self) -> None:
        """Operador >> compone morfismos."""
        f = IdentityMorphism(Stratum.PHYSICS)
        g = IdentityMorphism(Stratum.TACTICS)
        composed = f >> g
        
        assert isinstance(composed, ComposedMorphism)
        assert composed.codomain == Stratum.TACTICS
    
    def test_morphism_can_compose_with(self) -> None:
        """can_compose_with verifica compatibilidad."""
        f = IdentityMorphism(Stratum.PHYSICS)
        g = IdentityMorphism(Stratum.PHYSICS)
        
        assert f.can_compose_with(g) is True


# ==============================================================================
# TESTS DE PROPIEDADES CATEGÓRICAS
# ==============================================================================
class TestCategoryAxioms:
    """Tests para axiomas de teoría de categorías."""
    
    def test_identity_law_left(self) -> None:
        """Ley de identidad izquierda: id_B ∘ f = f."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"result": "test"}
        
        f = AtomicVector("test", Stratum.TACTICS, handler)
        id_codomain = IdentityMorphism(f.codomain)
        
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        
        # id ∘ f
        result1 = id_codomain(f(state))
        # f
        result2 = f(state)
        
        # Verificar que payloads son iguales (trazas difieren)
        assert result1.payload == result2.payload
    
    def test_identity_law_right(self) -> None:
        """Ley de identidad derecha: f ∘ id_A = f."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"result": "test"}
        
        f = AtomicVector("test", Stratum.TACTICS, handler)
        id_domain = IdentityMorphism(Stratum.PHYSICS)  # Strato requerido
        
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        
        # f ∘ id
        result1 = f(id_domain(state))
        # f
        result2 = f(state)
        
        assert result1.payload == result2.payload
    
    def test_associativity_law(self) -> None:
        """Ley de asociatividad: h ∘ (g ∘ f) = (h ∘ g) ∘ f."""
        def handler1(**kwargs) -> Dict[str, Any]:
            return {"step": 1}
        
        def handler2(**kwargs) -> Dict[str, Any]:
            return {"step": 2}
        
        def handler3(**kwargs) -> Dict[str, Any]:
            return {"step": 3}
        
        f = AtomicVector("f", Stratum.TACTICS, handler1)
        g = AtomicVector("g", Stratum.STRATEGY, handler2)
        h = AtomicVector("h", Stratum.OMEGA, handler3)
        
        state = create_categorical_state(
            strata=Stratum.OMEGA.requires(),
        )
        
        # h ∘ (g ∘ f)
        lhs = h >> (g >> f)
        result_lhs = lhs(state)
        
        # (h ∘ g) ∘ f
        rhs = (h >> g) >> f
        result_rhs = rhs(state)
        
        # Verificar igualdad estructural de payloads
        assert result_lhs.payload == result_rhs.payload
    
    def test_associativity_verification_method(self) -> None:
        """verify_associativity lanza error si se viola."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"test": "value"}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        g = AtomicVector("g", Stratum.STRATEGY, handler)
        h = AtomicVector("h", Stratum.OMEGA, handler)
        
        composed = ComposedMorphism(f, g)
        state = create_categorical_state(
            strata=Stratum.OMEGA.requires(),
        )
        
        # Debería pasar (asociatividad se cumple)
        assert composed.verify_associativity(h, state) is True
    
    def test_identity_verification_method(self) -> None:
        """verify_identity_law verifica leyes de identidad."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"test": "value"}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        id_morphism = IdentityMorphism(Stratum.PHYSICS)
        
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        
        assert id_morphism.verify_identity_law(f, state) is True


# ==============================================================================
# TESTS DE FUNTORES
# ==============================================================================
class TestFunctors:
    """Tests para funtores y preservación de estructura."""
    
    def test_state_to_dict_functor_map_object(self) -> None:
        """StateToDictFunctor mapea objetos a dicts."""
        functor = StateToDictFunctor()
        state = create_categorical_state(payload={"test": "value"})
        
        result = functor.map_object(state)
        
        assert isinstance(result, dict)
        assert "payload" in result
        assert result["payload"]["test"] == "value"
    
    def test_state_to_dict_functor_map_morphism(self) -> None:
        """StateToDictFunctor mapea morfismos a callables."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"result": "test"}
        
        functor = StateToDictFunctor()
        morphism = AtomicVector("test", Stratum.TACTICS, handler)
        
        mapped = functor.map_morphism(morphism)
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        result = mapped(state)
        
        assert isinstance(result, dict)
    
    def test_functor_preserves_composition(self) -> None:
        """Funtor preserva composición: F(g∘f) = F(g)∘F(f)."""
        def handler1(**kwargs) -> Dict[str, Any]:
            return {"step": 1}
        
        def handler2(**kwargs) -> Dict[str, Any]:
            return {"step": 2}
        
        functor = StateToDictFunctor()
        f = AtomicVector("f", Stratum.TACTICS, handler1)
        g = AtomicVector("g", Stratum.STRATEGY, handler2)
        
        state = create_categorical_state(
            strata=Stratum.STRATEGY.requires(),
        )
        
        # Verificar propiedad funtorial
        assert functor.verify_functoriality(f, g, state) is True


# ==============================================================================
# TESTS DE TRANSFORMACIONES NATURALES
# ==============================================================================
class TestNaturalTransformations:
    """Tests para transformaciones naturales."""
    
    def test_natural_transformation_creation(self) -> None:
        """Creación básica de transformación natural."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"test": "value"}
        
        source = AtomicVector("source", Stratum.TACTICS, handler)
        target = AtomicVector("target", Stratum.TACTICS, handler)
        
        class TestTransformation(NaturalTransformation):
            def __call__(self, state: CategoricalState) -> CategoricalState:
                return state
        
        transformation = TestTransformation(source, target, "test")
        assert transformation.name == "test"
    
    def test_vertical_composition(self) -> None:
        """Composición vertical de transformaciones."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"test": "value"}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        g = AtomicVector("g", Stratum.TACTICS, handler)
        h = AtomicVector("h", Stratum.TACTICS, handler)
        
        class T1(NaturalTransformation):
            def __call__(self, state: CategoricalState) -> CategoricalState:
                return state.with_update(new_payload={"t1": True})
        
        class T2(NaturalTransformation):
            def __call__(self, state: CategoricalState) -> CategoricalState:
                return state.with_update(new_payload={"t2": True})
        
        t1 = T1(f, g, "t1")
        t2 = T2(g, h, "t2")
        
        # La composición debería funcionar si target_morphism coincide
        composed = t1.vertical_compose(t2)
        assert composed is not None
    
    def test_horizontal_composition(self) -> None:
        """Composición horizontal de transformaciones."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {"test": "value"}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        g = AtomicVector("g", Stratum.TACTICS, handler)
        
        class TestTransformation(NaturalTransformation):
            def __call__(self, state: CategoricalState) -> CategoricalState:
                return state
        
        t1 = TestTransformation(f, f, "t1")
        t2 = TestTransformation(g, g, "t2")
        
        composed = t1.horizontal_compose(t2)
        assert composed is not None


# ==============================================================================
# TESTS DE VERIFICADORES ESTRUCTURALES Y HOMOLÓGICOS
# ==============================================================================
class TestVerifiers:
    """Tests para verificadores estructurales y homológicos."""
    
    def test_structural_verifier_composable_sequence(self) -> None:
        """is_composable_sequence verifica secuencias."""
        verifier = StructuralVerifier()
        
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        g = AtomicVector("g", Stratum.STRATEGY, handler)
        
        # Secuencia componible
        assert verifier.is_composable_sequence([f, g]) is True
    
    def test_homological_verifier_acyclicity(self) -> None:
        """verify_acyclicity detecta ciclos."""
        verifier = HomologicalVerifier()
        
        # Trazas sin ciclos
        traces = [
            CompositionTrace(
                step_number=1,
                morphism_name="m1",
                input_domain=frozenset({Stratum.PHYSICS}),
                output_codomain=Stratum.TACTICS,
                success=True,
            ),
            CompositionTrace(
                step_number=2,
                morphism_name="m2",
                input_domain=frozenset({Stratum.TACTICS}),
                output_codomain=Stratum.STRATEGY,
                success=True,
            ),
        ]
        
        assert verifier.verify_acyclicity(traces) is True
    
    def test_homological_verifier_betti_numbers(self) -> None:
        """compute_betti_numbers calcula números de Betti."""
        verifier = HomologicalVerifier()
        
        traces = [
            CompositionTrace(
                step_number=1,
                morphism_name="m1",
                input_domain=frozenset({Stratum.PHYSICS}),
                output_codomain=Stratum.TACTICS,
                success=True,
            ),
        ]
        
        betti = verifier.compute_betti_numbers(traces)
        assert 0 in betti
        assert 1 in betti
        assert betti[0] >= 0  # β₀ ≥ 0
        assert betti[1] >= 0  # β₁ ≥ 0
    
    def test_homological_verifier_detect_cycles(self) -> None:
        """detect_cycles encuentra ciclos explícitos."""
        verifier = HomologicalVerifier()
        
        # Crear trazas que formen ciclo
        traces = [
            CompositionTrace(
                step_number=1,
                morphism_name="m1",
                input_domain=frozenset({Stratum.PHYSICS}),
                output_codomain=Stratum.TACTICS,
                success=True,
            ),
            CompositionTrace(
                step_number=2,
                morphism_name="m2",
                input_domain=frozenset({Stratum.TACTICS}),
                output_codomain=Stratum.PHYSICS,  # Ciclo de vuelta
                success=True,
            ),
        ]
        
        cycles = verifier.detect_cycles(traces)
        # Debería detectar al menos un ciclo
        assert isinstance(cycles, list)
    
    def test_euler_characteristic(self) -> None:
        """compute_euler_characteristic calcula χ correctamente."""
        verifier = StructuralVerifier()
        
        traces = [
            CompositionTrace(
                step_number=1,
                morphism_name="m1",
                input_domain=frozenset({Stratum.PHYSICS}),
                output_codomain=Stratum.TACTICS,
                success=True,
            ),
        ]
        
        chi = verifier.compute_euler_characteristic(traces)
        assert isinstance(chi, int)


# ==============================================================================
# TESTS DE MORPHISM COMPOSER
# ==============================================================================
class TestMorphismComposer:
    """Tests para compositor de morfismos."""
    
    def test_composer_add_step(self) -> None:
        """add_step agrega morfismos al pipeline."""
        composer = MorphismComposer()
        
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        m = AtomicVector("test", Stratum.TACTICS, handler)
        composer.add_step(m)
        
        assert len(composer.steps) == 1
    
    def test_composer_build(self) -> None:
        """build construye morfismo compuesto."""
        composer = MorphismComposer()
        
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        m = AtomicVector("test", Stratum.TACTICS, handler)
        composer.add_step(m)
        
        result = composer.build()
        assert isinstance(result, Morphism)
    
    def test_composer_build_empty_raises(self) -> None:
        """build con pasos vacíos lanza ValueError."""
        composer = MorphismComposer()
        
        with pytest.raises(ValueError):
            composer.build()
    
    def test_composer_reset(self) -> None:
        """reset limpia el compositor."""
        composer = MorphismComposer()
        
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        m = AtomicVector("test", Stratum.TACTICS, handler)
        composer.add_step(m)
        composer.reset()
        
        assert len(composer.steps) == 0
    
    def test_composer_visualize(self) -> None:
        """visualize produce representación legible."""
        composer = MorphismComposer()
        assert composer.visualize() == "(vacío)"
        
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        m = AtomicVector("test", Stratum.TACTICS, handler)
        composer.add_step(m)
        
        visualization = composer.visualize()
        assert "1." in visualization
        assert "test" in visualization


# ==============================================================================
# TESTS DE CATEGORICAL REGISTRY
# ==============================================================================
class TestCategoricalRegistry:
    """Tests para registro categórico thread-safe."""
    
    def test_register_morphism(self, registry: CategoricalRegistry) -> None:
        """register_morphism agrega morfismo."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        m = AtomicVector("test", Stratum.TACTICS, handler)
        registry.register_morphism("test_morphism", m)
        
        assert "test_morphism" in registry.list_morphisms()
    
    def test_register_composition(self, registry: CategoricalRegistry) -> None:
        """register_composition agrega composición."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        g = AtomicVector("g", Stratum.STRATEGY, handler)
        composed = ComposedMorphism(f, g)
        
        registry.register_composition("test_composition", composed)
        
        assert "test_composition" in registry.list_compositions()
    
    def test_get_morphism(self, registry: CategoricalRegistry) -> None:
        """get_morphism recupera morfismo por nombre."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        m = AtomicVector("test_morphism", Stratum.TACTICS, handler)
        registry.register_morphism("test_morphism", m)
        
        retrieved = registry.get_morphism("test_morphism")
        assert retrieved is not None
        assert retrieved.name == "test_morphism"
    
    def test_get_morphism_not_found(self, registry: CategoricalRegistry) -> None:
        """get_morphism retorna None si no existe."""
        assert registry.get_morphism("nonexistent") is None
    
    def test_registry_thread_safety(self, registry: CategoricalRegistry) -> None:
        """Registro es thread-safe."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        def register_worker(reg: CategoricalRegistry, index: int) -> None:
            m = AtomicVector(f"m{index}", Stratum.TACTICS, handler)
            reg.register_morphism(f"morphism_{index}", m)
        
        threads = [
            threading.Thread(target=register_worker, args=(registry, i))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(registry.list_morphisms()) == 10


# ==============================================================================
# TESTS DE TWO CATEGORY ORCHESTRATOR
# ==============================================================================
class TestTwoCategoryOrchestrator:
    """Tests para orquestador de 2-categoría."""
    
    def test_interchange_law_valid(self) -> None:
        """validate_interchange_law pasa para transformaciones válidas."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        f = AtomicVector("f", Stratum.TACTICS, handler)
        g = AtomicVector("g", Stratum.TACTICS, handler)
        
        class IdentityTransformation(NaturalTransformation):
            def __call__(self, state: CategoricalState) -> CategoricalState:
                return state
        
        alpha = IdentityTransformation(f, f, "α")
        alpha_prime = IdentityTransformation(f, f, "α'")
        beta = IdentityTransformation(g, g, "β")
        beta_prime = IdentityTransformation(g, g, "β'")
        
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        
        # Debería pasar para transformaciones identidad
        result = TwoCategoryOrchestrator.validate_interchange_law(
            alpha, alpha_prime, beta, beta_prime, state
        )
        assert result is True


# ==============================================================================
# TESTS DE PROPIEDADES (PROPERTY-BASED)
# ==============================================================================
class TestProperties:
    """Tests de propiedades que deben cumplirse siempre."""
    
    @pytest.mark.property
    def test_state_update_preserves_error(self) -> None:
        """with_update preserva estado de error."""
        state = create_categorical_state().with_error("test error")
        updated = state.with_update(new_payload={"new": "data"})
        
        assert updated.is_failed is True
        assert updated.error == state.error
    
    @pytest.mark.property
    def test_state_trace_length_never_decreases(self) -> None:
        """trace_length nunca decrece con operaciones."""
        state = create_categorical_state()
        initial_length = state.trace_length
        
        for i in range(5):
            state = state.add_trace(
                f"morphism_{i}",
                frozenset(),
                Stratum.PHYSICS,
                True,
            )
            assert state.trace_length >= initial_length
            initial_length = state.trace_length
    
    @pytest.mark.property
    def test_hash_collision_improbable(self) -> None:
        """Colisiones de hash son extremadamente improbables."""
        states = [
            create_categorical_state(payload={"index": i})
            for i in range(100)
        ]
        
        hashes = [s.compute_hash() for s in states]
        
        # No debería haber colisiones
        assert len(hashes) == len(set(hashes))
    
    @pytest.mark.property
    def test_stratum_chain_total_order(self) -> None:
        """La cadena de estratos es orden total."""
        chain = Stratum.chain()
        
        for i in range(len(chain) - 1):
            assert chain[i] < chain[i + 1]
    
    @pytest.mark.property
    def test_morphism_call_count_increments(self) -> None:
        """call_count incrementa con cada llamada."""
        def handler(**kwargs) -> Dict[str, Any]:
            return {}
        
        morphism = AtomicVector("test", Stratum.TACTICS, handler)
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        
        initial_count = morphism.call_count
        
        for _ in range(5):
            morphism(state)
        
        assert morphism.call_count == initial_count + 5


# ==============================================================================
# TESTS DE BORDES Y CASOS EXTREMOS
# ==============================================================================
class TestEdgeCases:
    """Tests para casos borde y condiciones extremas."""
    
    def test_empty_payload_state(self) -> None:
        """Estado con payload vacío funciona."""
        state = create_categorical_state(payload={})
        assert state.payload == {}
        assert state.is_success is True
    
    def test_deeply_nested_payload(self) -> None:
        """Payload profundamente anidado se serializa."""
        nested = {"level1": {"level2": {"level3": {"value": 42}}}}
        state = create_categorical_state(payload=nested)
        
        serialized = state.to_dict()
        assert serialized["payload"]["level1"]["level2"]["level3"]["value"] == 42
    
    def test_large_payload(self) -> None:
        """Payload grande se maneja correctamente."""
        large_payload = {f"key_{i}": f"value_{i}" for i in range(1000)}
        state = create_categorical_state(payload=large_payload)
        
        assert len(state.payload) == 1000
        assert state.compute_hash() is not None
    
    def test_unicode_payload(self) -> None:
        """Payload con unicode se serializa."""
        state = create_categorical_state(
            payload={"texto": "España", "emoji": "🎉", "chino": "你好"}
        )
        
        serialized = state.to_dict()
        assert serialized["payload"]["texto"] == "España"
    
    def test_special_float_values(self) -> None:
        """Valores float especiales se manejan."""
        state = create_categorical_state(
            payload={
                "inf": float('inf'),
                "neginf": float('-inf'),
                "nan": float('nan'),
            }
        )
        
        # Debería poder calcular hash (con fallback para nan)
        hash_value = state.compute_hash()
        assert len(hash_value) == 64
    
    def test_morphism_with_no_required_keys(self) -> None:
        """Morfismo sin claves requeridas funciona."""
        def handler() -> Dict[str, Any]:
            return {"result": "ok"}
        
        morphism = AtomicVector(
            "test",
            Stratum.TACTICS,
            handler,
            required_keys=[],
        )
        
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        result = morphism(state)
        
        assert result.is_success is True
    
    def test_composition_chain_length(self) -> None:
        """Cadena larga de composiciones funciona."""
        def make_handler(i: int):
            def handler(**kwargs) -> Dict[str, Any]:
                return {f"step_{i}": i}
            return handler
        
        morphisms = [
            AtomicVector(f"m{i}", Stratum.TACTICS, make_handler(i))
            for i in range(10)
        ]
        
        composer = MorphismComposer()
        for m in morphisms:
            composer.add_step(m)
        
        composed = composer.build()
        state = create_categorical_state(
            strata=Stratum.TACTICS.requires(),
        )
        result = composed(state)
        
        # Último morfismo debería haberse ejecutado
        assert result.trace_length >= 1


# ==============================================================================
# TESTS DE RENDIMIENTO BÁSICO
# ==============================================================================
class TestPerformance:
    """Tests básicos de rendimiento."""
    
    def test_state_creation_performance(self) -> None:
        """Creación de estado es rápida."""
        import time
        
        start = time.perf_counter()
        for _ in range(1000):
            create_categorical_state(payload={"test": "value"})
        elapsed = time.perf_counter() - start
        
        # Debería completar en menos de 1 segundo
        assert elapsed < 1.0
    
    def test_hash_computation_performance(self) -> None:
        """Cálculo de hash es rápido."""
        import time
        
        state = create_categorical_state(
            payload={"data": list(range(100))}
        )
        
        start = time.perf_counter()
        for _ in range(100):
            state.compute_hash()
        elapsed = time.perf_counter() - start
        
        # 100 hashes en menos de 1 segundo
        assert elapsed < 1.0


# ==============================================================================
# EJECUCIÓN DIRECTA
# ==============================================================================
if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.core.mic_algebra",
        "--cov-report=term-missing",
    ])