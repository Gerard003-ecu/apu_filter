"""
Tests para MIC Hierarchy y Gatekeeper Logic
=============================================

Verifica que la Matriz de Interacción Central (MIC) imponga correctamente
la regla "Physics before Strategy" siguiendo la pirámide DIKW.

Modelo Formal — Estructura de Orden:

  Sea P = {PHYSICS, TACTICS, STRATEGY, WISDOM} el conjunto de estratos.
  Definimos la relación de orden estricto (<) sobre P:
  
    PHYSICS < TACTICS < STRATEGY < WISDOM
    
  Esta relación induce:
    1. Un orden total (cadena) sobre P
    2. Una función de requisitos req: P → 𝒫(P)
       donde req(s) = {t ∈ P | t < s}

  Diagrama de Hasse:
  
         WISDOM
            │
         STRATEGY
            │
         TACTICS
            │
         PHYSICS
            
  (Estructura de cadena lineal, caso degenerado de lattice)

Propiedades Algebraicas del Orden:

  ┌─────┬────────────────────────────────────────────────────────────────────┐
  │ A₁  │ Irreflexividad: ∀ s ∈ P, s ∉ req(s)                               │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ A₂  │ Antisimetría: s₁ ∈ req(s₂) → s₂ ∉ req(s₁)                         │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ A₃  │ Transitividad: req(WISDOM) ⊇ req(STRATEGY) ⊇ req(TACTICS)         │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ A₄  │ Totalidad: ∀ s₁ ≠ s₂, s₁ ∈ req(s₂) ∨ s₂ ∈ req(s₁)                │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ A₅  │ Minimalidad: |req(PHYSICS)| = 0 (elemento mínimo ⊥)               │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ A₆  │ Maximalidad: req(WISDOM) = P \\ {WISDOM} (elemento máximo ⊤)      │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ A₇  │ Bien fundado: no existen cadenas descendentes infinitas           │
  └─────┴────────────────────────────────────────────────────────────────────┘

Reglas del Gatekeeper:

  ┌─────┬────────────────────────────────────────────────────────────────────┐
  │ G₁  │ Vector de estrato s ejecutable ↔ req(s) ⊆ validated_strata        │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ G₂  │ force_physics_override = True → bypass universal de G₁            │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ G₃  │ Vector desconocido → ValueError                                   │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ G₄  │ Excepción en handler → {success: False}, sin actualización        │
  ├─────┼────────────────────────────────────────────────────────────────────┤
  │ G₅  │ Resultado siempre contiene 'success' (bool)                       │
  └─────┴────────────────────────────────────────────────────────────────────┘

Pirámide DIKW (Data-Information-Knowledge-Wisdom):

  El modelo DIKW es una jerarquía epistémica donde cada nivel
  construye sobre el anterior:
  
    - PHYSICS (Data): Hechos crudos, mediciones, observaciones
    - TACTICS (Information): Datos contextualizados, patrones
    - STRATEGY (Knowledge): Información aplicada, modelos
    - WISDOM (Wisdom): Conocimiento con juicio, decisiones
    
  La regla "Physics before Strategy" formaliza que no se pueden
  tomar decisiones estratégicas sin validar primero los fundamentos
  físicos/empíricos del sistema.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, FrozenSet
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from app.schemas import Stratum


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN DE DOMINIO
# =============================================================================

class MICErrorMessages:
    """
    Mensajes de error estándar del sistema MIC.
    
    Centralizar mensajes permite:
      - Consistencia en assertions
      - Facilidad de internacionalización
      - Mantenimiento simplificado
    """
    UNKNOWN_VECTOR = "Unknown vector"
    HIERARCHY_VIOLATION = "MIC Hierarchy Violation"
    PERMISSION_ERROR = "PermissionError"
    REQUIRES_PRIOR_VALIDATION = "requires prior validation"
    HANDLER_FAILED = "Handler execution failed"


class MICResultKeys:
    """Claves estándar del diccionario de resultado MIC."""
    SUCCESS = "success"
    ERROR = "error"
    ERROR_TYPE = "error_type"
    REQUIRED_STRATA = "required_strata"
    VALIDATION_UPDATE = "_mic_validation_update"
    STRATUM = "stratum"


class MICContextKeys:
    """Claves del contexto de ejecución MIC."""
    VALIDATED_STRATA = "validated_strata"
    FORCE_OVERRIDE = "force_physics_override"


class DIKWPyramid:
    """
    Estructura formal de la pirámide DIKW.
    
    Define la semántica del orden y los requisitos de cada nivel.
    """
    # Orden canónico de la pirámide (de base a cúspide)
    ORDER: List[Stratum] = [
        Stratum.PHYSICS,
        Stratum.TACTICS,
        Stratum.STRATEGY,
        Stratum.WISDOM,
    ]
    
    # Requisitos de cada estrato (pre-calculados)
    REQUIREMENTS: Dict[Stratum, Set[Stratum]] = {
        Stratum.PHYSICS: set(),
        Stratum.TACTICS: {Stratum.PHYSICS},
        Stratum.STRATEGY: {Stratum.PHYSICS, Stratum.TACTICS},
        Stratum.WISDOM: {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY},
    }
    
    @classmethod
    def get_level(cls, stratum: Stratum) -> int:
        """Retorna el nivel (0-indexed) del estrato en la pirámide."""
        return cls.ORDER.index(stratum)
    
    @classmethod
    def is_below(cls, s1: Stratum, s2: Stratum) -> bool:
        """Verifica si s1 < s2 en el orden de la pirámide."""
        return cls.get_level(s1) < cls.get_level(s2)
    
    @classmethod
    def get_requirements(cls, stratum: Stratum) -> Set[Stratum]:
        """Retorna req(stratum) = {t | t < stratum}."""
        return cls.REQUIREMENTS.get(stratum, set())
    
    @classmethod
    def minimum(cls) -> Stratum:
        """Retorna el elemento mínimo (⊥) de la pirámide."""
        return cls.ORDER[0]
    
    @classmethod
    def maximum(cls) -> Stratum:
        """Retorna el elemento máximo (⊤) de la pirámide."""
        return cls.ORDER[-1]


# =============================================================================
# IMPORTACIONES CONDICIONALES
# =============================================================================

try:
    from app.tools_interface import MICRegistry, IntentVector
    HAS_MIC = True
except ImportError:
    HAS_MIC = False
    MICRegistry = None
    IntentVector = None

try:
    from app.telemetry import TelemetryContext, StepStatus
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False
    TelemetryContext = None
    StepStatus = None


# =============================================================================
# HELPERS DE VALIDACIÓN
# =============================================================================

def validate_mic_result_structure(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida la estructura del resultado MIC.
    
    Invariantes:
      (R₁) 'success' siempre presente y es bool
      (R₂) Si success=True → puede contener '_mic_validation_update'
      (R₃) Si success=False → contiene 'error' y 'error_type'
      
    Returns:
        Dict con:
          - is_valid: bool
          - violations: List[str]
          - success_value: bool | None
    """
    violations = []
    
    # R₁: success obligatorio
    if MICResultKeys.SUCCESS not in result:
        violations.append(f"Falta clave '{MICResultKeys.SUCCESS}'")
        return {
            "is_valid": False,
            "violations": violations,
            "success_value": None,
        }
    
    if not isinstance(result[MICResultKeys.SUCCESS], bool):
        violations.append(
            f"'{MICResultKeys.SUCCESS}' no es bool: "
            f"{type(result[MICResultKeys.SUCCESS])}"
        )
    
    success = result[MICResultKeys.SUCCESS]
    
    if not success:
        # R₃: claves de error
        if MICResultKeys.ERROR not in result:
            violations.append(f"Falta '{MICResultKeys.ERROR}' en resultado fallido")
        if MICResultKeys.ERROR_TYPE not in result:
            violations.append(f"Falta '{MICResultKeys.ERROR_TYPE}' en resultado fallido")
    
    return {
        "is_valid": len(violations) == 0,
        "violations": violations,
        "success_value": success,
    }


def assert_mic_success(
    result: Dict[str, Any],
    expected_stratum: Optional[Stratum] = None,
    context: str = "",
):
    """
    Aserción compuesta para resultado exitoso de MIC.
    """
    prefix = f"[{context}] " if context else ""
    
    assert result[MICResultKeys.SUCCESS] is True, (
        f"{prefix}Esperado success=True, obtenido: {result}"
    )
    
    if expected_stratum is not None:
        assert MICResultKeys.VALIDATION_UPDATE in result, (
            f"{prefix}Falta '_mic_validation_update' en resultado exitoso"
        )
        assert result[MICResultKeys.VALIDATION_UPDATE] == expected_stratum, (
            f"{prefix}Estrato incorrecto: "
            f"esperado {expected_stratum}, "
            f"obtenido {result[MICResultKeys.VALIDATION_UPDATE]}"
        )


def assert_mic_hierarchy_violation(
    result: Dict[str, Any],
    expected_required: Optional[Set[Stratum]] = None,
    context: str = "",
):
    """
    Aserción compuesta para violación de jerarquía MIC.
    """
    prefix = f"[{context}] " if context else ""
    
    assert result[MICResultKeys.SUCCESS] is False, (
        f"{prefix}Esperado success=False por violación de jerarquía"
    )
    assert result[MICResultKeys.ERROR_TYPE] == MICErrorMessages.PERMISSION_ERROR, (
        f"{prefix}error_type debe ser PermissionError"
    )
    assert MICErrorMessages.HIERARCHY_VIOLATION in result[MICResultKeys.ERROR], (
        f"{prefix}Error debe mencionar 'MIC Hierarchy Violation'"
    )
    
    if expected_required is not None:
        assert MICResultKeys.REQUIRED_STRATA in result
        actual_required = set(result[MICResultKeys.REQUIRED_STRATA])
        expected_names = {s.name for s in expected_required}
        assert actual_required == expected_names, (
            f"{prefix}Estratos requeridos incorrectos: "
            f"esperado {expected_names}, obtenido {actual_required}"
        )


def verify_order_properties(registry) -> Dict[str, Any]:
    """
    Verifica todas las propiedades algebraicas del orden.
    
    Returns:
        Dict con resultados de verificación de A₁-A₇
    """
    results = {}
    
    # A₁: Irreflexividad
    a1_violations = []
    for s in Stratum:
        if s in registry.get_required_strata(s):
            a1_violations.append(s.name)
    results["A1_irreflexive"] = len(a1_violations) == 0
    results["A1_violations"] = a1_violations
    
    # A₂: Antisimetría
    a2_violations = []
    for s1 in Stratum:
        for s2 in Stratum:
            if s1 != s2:
                req1 = registry.get_required_strata(s1)
                req2 = registry.get_required_strata(s2)
                if s2 in req1 and s1 in req2:
                    a2_violations.append((s1.name, s2.name))
    results["A2_antisymmetric"] = len(a2_violations) == 0
    results["A2_violations"] = a2_violations
    
    # A₃: Transitividad (cadena de inclusión)
    order = DIKWPyramid.ORDER
    a3_valid = True
    for i in range(len(order) - 1):
        lower = registry.get_required_strata(order[i])
        higher = registry.get_required_strata(order[i + 1])
        if not lower.issubset(higher):
            a3_valid = False
            break
    results["A3_transitive"] = a3_valid
    
    # A₄: Totalidad
    a4_valid = True
    for i, s1 in enumerate(list(Stratum)):
        for s2 in list(Stratum)[i + 1:]:
            req1 = registry.get_required_strata(s1)
            req2 = registry.get_required_strata(s2)
            if s1 not in req2 and s2 not in req1:
                a4_valid = False
                break
    results["A4_total"] = a4_valid
    
    # A₅: Minimalidad
    results["A5_minimum"] = (
        registry.get_required_strata(Stratum.PHYSICS) == set()
    )
    
    # A₆: Maximalidad
    expected_max = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
    results["A6_maximum"] = (
        registry.get_required_strata(Stratum.WISDOM) == expected_max
    )
    
    # A₇: Bien fundado (trivialmente verdadero para conjunto finito)
    results["A7_well_founded"] = True
    
    results["all_valid"] = all([
        results["A1_irreflexive"],
        results["A2_antisymmetric"],
        results["A3_transitive"],
        results["A4_total"],
        results["A5_minimum"],
        results["A6_maximum"],
        results["A7_well_founded"],
    ])
    
    return results


# =============================================================================
# MOCK DEL REGISTRY
# =============================================================================

class MockMICRegistry:
    """
    Mock fiel del MICRegistry para tests.

    Implementa la semántica de jerarquía DIKW como orden parcial
    estricto sobre Stratum con las reglas del gatekeeper.
    
    Thread-safety: El registry usa un lock para operaciones de registro
    pero las proyecciones son lock-free (lectura).
    """

    def __init__(self):
        self._vectors: Dict[str, Tuple[Stratum, Callable]] = {}
        self._lock = threading.Lock()

    def register_vector(
        self,
        name: str,
        stratum: Stratum,
        handler: Callable,
    ) -> None:
        """
        Registra un vector con su estrato y handler.
        
        Thread-safe mediante lock.
        """
        if name is None:
            raise ValueError("Vector name cannot be None")
        if stratum is None:
            raise ValueError("Stratum cannot be None")
        if handler is None:
            raise ValueError("Handler cannot be None")
            
        with self._lock:
            self._vectors[name] = (stratum, handler)

    def project_intent(
        self,
        vector_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Proyecta una intención a través del vector especificado.
        
        Aplica las reglas del gatekeeper G₁-G₅.
        """
        # G₃: Vector desconocido
        if vector_name not in self._vectors:
            raise ValueError(f"{MICErrorMessages.UNKNOWN_VECTOR}: {vector_name}")

        stratum, handler = self._vectors[vector_name]

        # G₁: Verificar permisos de jerarquía
        if not self._check_hierarchy_permission(stratum, context):
            required = DIKWPyramid.get_requirements(stratum)
            return {
                MICResultKeys.SUCCESS: False,
                MICResultKeys.ERROR_TYPE: MICErrorMessages.PERMISSION_ERROR,
                MICResultKeys.ERROR: (
                    f"{MICErrorMessages.HIERARCHY_VIOLATION}: {stratum.name} "
                    f"{MICErrorMessages.REQUIRES_PRIOR_VALIDATION}"
                ),
                MICResultKeys.REQUIRED_STRATA: [s.name for s in required],
            }

        # Ejecutar handler
        try:
            result = handler(**payload)
            
            # Validar que el handler retornó un dict con 'success'
            if not isinstance(result, dict):
                return {
                    MICResultKeys.SUCCESS: False,
                    MICResultKeys.ERROR_TYPE: "TypeError",
                    MICResultKeys.ERROR: "Handler must return a dict",
                }
            
            if MICResultKeys.SUCCESS not in result:
                result[MICResultKeys.SUCCESS] = True
            
            # Agregar actualización de validación
            result[MICResultKeys.VALIDATION_UPDATE] = stratum
            return result
            
        except Exception as e:
            # G₄: Excepción en handler
            return {
                MICResultKeys.SUCCESS: False,
                MICResultKeys.ERROR_TYPE: type(e).__name__,
                MICResultKeys.ERROR: str(e),
            }

    def _check_hierarchy_permission(
        self,
        stratum: Stratum,
        context: Dict[str, Any],
    ) -> bool:
        """
        Verifica si el estrato tiene permiso para ejecutarse.
        
        Implementa G₁ y G₂.
        """
        # G₂: Override universal
        if context.get(MICContextKeys.FORCE_OVERRIDE, False):
            return True
        
        # G₁: Verificar requisitos
        validated = context.get(MICContextKeys.VALIDATED_STRATA, set())
        if validated is None:
            validated = set()
        
        required = DIKWPyramid.get_requirements(stratum)
        return required.issubset(validated)

    def get_required_strata(self, stratum: Stratum) -> Set[Stratum]:
        """Retorna los estratos requeridos para el estrato dado."""
        return DIKWPyramid.get_requirements(stratum)

    def is_registered(self, vector_name: str) -> bool:
        """Verifica si un vector está registrado."""
        return vector_name in self._vectors

    def get_vector_stratum(self, vector_name: str) -> Optional[Stratum]:
        """Retorna el estrato de un vector, o None si no existe."""
        if vector_name in self._vectors:
            return self._vectors[vector_name][0]
        return None

    def get_registered_vectors(self) -> List[str]:
        """Retorna lista de nombres de vectores registrados."""
        return list(self._vectors.keys())

    def clear(self) -> None:
        """Limpia todos los vectores registrados."""
        with self._lock:
            self._vectors.clear()


# =============================================================================
# HANDLERS MOCK
# =============================================================================

def _physics_handler(val: int) -> Dict[str, Any]:
    """Handler de nivel PHYSICS."""
    return {
        MICResultKeys.SUCCESS: True,
        "val": val,
        MICResultKeys.STRATUM: "PHYSICS",
    }


def _tactics_handler(items: list) -> Dict[str, Any]:
    """Handler de nivel TACTICS."""
    return {
        MICResultKeys.SUCCESS: True,
        "items": items,
        MICResultKeys.STRATUM: "TACTICS",
    }


def _strategy_handler(amount: float) -> Dict[str, Any]:
    """Handler de nivel STRATEGY."""
    return {
        MICResultKeys.SUCCESS: True,
        "amount": amount,
        MICResultKeys.STRATUM: "STRATEGY",
    }


def _wisdom_handler(decision: str) -> Dict[str, Any]:
    """Handler de nivel WISDOM."""
    return {
        MICResultKeys.SUCCESS: True,
        "decision": decision,
        MICResultKeys.STRATUM: "WISDOM",
    }


def _failing_handler(**kwargs) -> Dict[str, Any]:
    """Handler que siempre falla."""
    raise RuntimeError(MICErrorMessages.HANDLER_FAILED)


def _slow_handler(delay: float = 0.1) -> Dict[str, Any]:
    """Handler con latencia artificial."""
    time.sleep(delay)
    return {MICResultKeys.SUCCESS: True, "delayed": True}


def _malformed_handler(**kwargs) -> str:
    """Handler que retorna tipo incorrecto."""
    return "not_a_dict"


def _no_success_handler(**kwargs) -> Dict[str, Any]:
    """Handler que retorna dict sin 'success'."""
    return {"data": "value"}


# =============================================================================
# FACTORIES Y FIXTURES
# =============================================================================

def _build_registry() -> MockMICRegistry:
    """Construye el registry con vectores mock registrados."""
    registry_class = MICRegistry if HAS_MIC else MockMICRegistry
    registry = registry_class()
    
    registry.register_vector("mock_physics", Stratum.PHYSICS, _physics_handler)
    registry.register_vector("mock_tactics", Stratum.TACTICS, _tactics_handler)
    registry.register_vector("mock_strategy", Stratum.STRATEGY, _strategy_handler)
    registry.register_vector("mock_wisdom", Stratum.WISDOM, _wisdom_handler)
    
    return registry


def _build_context(
    validated: Optional[Set[Stratum]] = None,
    force_override: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Factory para contextos de ejecución MIC.

    Args:
        validated: Conjunto de estratos validados (None → set vacío)
        force_override: Si True, activa bypass de jerarquía
        extra_fields: Campos adicionales para el contexto
        
    Returns:
        Contexto estructurado para project_intent
    """
    ctx = {
        MICContextKeys.VALIDATED_STRATA: validated if validated is not None else set()
    }
    
    if force_override:
        ctx[MICContextKeys.FORCE_OVERRIDE] = True
    
    if extra_fields:
        ctx.update(extra_fields)
    
    return ctx


# Mapeo estrato → (vector_name, payload) para tests parametrizados
STRATUM_VECTOR_MAP: Dict[Stratum, Tuple[str, Dict[str, Any]]] = {
    Stratum.PHYSICS: ("mock_physics", {"val": 1}),
    Stratum.TACTICS: ("mock_tactics", {"items": []}),
    Stratum.STRATEGY: ("mock_strategy", {"amount": 1.0}),
    Stratum.WISDOM: ("mock_wisdom", {"decision": "test"}),
}


@pytest.fixture
def mic() -> MockMICRegistry:
    """Registry MIC con vectores mock registrados."""
    return _build_registry()


@pytest.fixture
def mic_empty() -> MockMICRegistry:
    """Registry MIC vacío (sin vectores)."""
    return MockMICRegistry()


@pytest.fixture
def mic_failing() -> MockMICRegistry:
    """Registry con un handler que siempre falla."""
    registry = MockMICRegistry()
    registry.register_vector("failing_vector", Stratum.PHYSICS, _failing_handler)
    return registry


@pytest.fixture
def mic_slow() -> MockMICRegistry:
    """Registry con handler de latencia."""
    registry = MockMICRegistry()
    registry.register_vector("slow_vector", Stratum.PHYSICS, _slow_handler)
    return registry


# =============================================================================
# TEST: REGISTRO DE VECTORES
# =============================================================================

class TestMICRegistryBasics:
    """
    Pruebas de registro y consulta de vectores en el MIC.
    
    Verifica la semántica básica del registry como contenedor
    de vectores mapeados a estratos.
    """

    def test_registered_vectors_found(self, mic):
        """Todos los vectores registrados son localizables."""
        expected_vectors = ["mock_physics", "mock_tactics", "mock_strategy", "mock_wisdom"]
        
        for name in expected_vectors:
            assert mic.is_registered(name), f"Vector '{name}' no encontrado"

    def test_unregistered_vector_not_found(self, mic):
        """Vector no registrado retorna False."""
        assert not mic.is_registered("nonexistent_vector")
        assert not mic.is_registered("")
        assert not mic.is_registered("MOCK_PHYSICS")  # Case sensitive

    @pytest.mark.parametrize(
        "vector_name, expected_stratum",
        [
            ("mock_physics", Stratum.PHYSICS),
            ("mock_tactics", Stratum.TACTICS),
            ("mock_strategy", Stratum.STRATEGY),
            ("mock_wisdom", Stratum.WISDOM),
        ],
        ids=["physics", "tactics", "strategy", "wisdom"],
    )
    def test_vector_stratum_mapping(self, mic, vector_name, expected_stratum):
        """Cada vector reporta su estrato correcto."""
        assert mic.get_vector_stratum(vector_name) == expected_stratum

    def test_unknown_vector_stratum_returns_none(self, mic):
        """Estrato de vector desconocido → None."""
        assert mic.get_vector_stratum("unknown") is None
        assert mic.get_vector_stratum("") is None

    def test_unknown_vector_raises_valueerror(self, mic):
        """Regla G₃: proyectar vector desconocido → ValueError."""
        with pytest.raises(ValueError, match=MICErrorMessages.UNKNOWN_VECTOR):
            mic.project_intent("unknown_vector", {}, _build_context())

    def test_empty_registry_raises_for_any_vector(self, mic_empty):
        """Registry vacío siempre lanza ValueError."""
        with pytest.raises(ValueError):
            mic_empty.project_intent("any_vector", {}, _build_context())

    def test_duplicate_registration_overwrites(self):
        """
        Re-registrar un vector con el mismo nombre sobrescribe
        el handler anterior (semántica de dict).
        """
        registry = MockMICRegistry()

        def handler_v1(val: int):
            return {MICResultKeys.SUCCESS: True, "version": 1, "val": val}

        def handler_v2(val: int):
            return {MICResultKeys.SUCCESS: True, "version": 2, "val": val}

        registry.register_vector("test_v", Stratum.PHYSICS, handler_v1)
        registry.register_vector("test_v", Stratum.PHYSICS, handler_v2)

        result = registry.project_intent("test_v", {"val": 1}, _build_context())

        assert result["version"] == 2

    def test_reregistration_with_different_stratum(self):
        """
        Re-registrar un vector con stratum diferente actualiza el estrato.
        """
        registry = MockMICRegistry()
        registry.register_vector("migrating_v", Stratum.PHYSICS, _physics_handler)

        assert registry.get_vector_stratum("migrating_v") == Stratum.PHYSICS

        registry.register_vector("migrating_v", Stratum.TACTICS, _physics_handler)

        assert registry.get_vector_stratum("migrating_v") == Stratum.TACTICS

    def test_register_with_none_name_raises(self):
        """Registrar con nombre None lanza ValueError."""
        registry = MockMICRegistry()
        with pytest.raises(ValueError, match="name"):
            registry.register_vector(None, Stratum.PHYSICS, _physics_handler)

    def test_register_with_none_stratum_raises(self):
        """Registrar con stratum None lanza ValueError."""
        registry = MockMICRegistry()
        with pytest.raises(ValueError, match="[Ss]tratum"):
            registry.register_vector("test", None, _physics_handler)

    def test_register_with_none_handler_raises(self):
        """Registrar con handler None lanza ValueError."""
        registry = MockMICRegistry()
        with pytest.raises(ValueError, match="[Hh]andler"):
            registry.register_vector("test", Stratum.PHYSICS, None)

    def test_get_registered_vectors(self, mic):
        """Obtener lista de vectores registrados."""
        vectors = mic.get_registered_vectors()
        
        assert len(vectors) == 4
        assert set(vectors) == {
            "mock_physics", "mock_tactics", 
            "mock_strategy", "mock_wisdom"
        }

    def test_clear_removes_all_vectors(self, mic):
        """Clear elimina todos los vectores."""
        assert len(mic.get_registered_vectors()) == 4
        
        mic.clear()
        
        assert len(mic.get_registered_vectors()) == 0
        assert not mic.is_registered("mock_physics")


# =============================================================================
# TEST: PERMISOS DE JERARQUÍA
# =============================================================================

class TestMICHierarchyPermissions:
    """
    Pruebas de la regla G₁: vector de estrato s ejecutable
    ↔ req(s) ⊆ validated_strata.

    Verifica cada estrato individualmente y la matriz completa.
    """

    # ── PHYSICS: ELEMENTO MÍNIMO (req = ∅) ─────────────────────────────

    def test_physics_always_allowed_empty_context(self, mic):
        """
        PHYSICS es la base de la pirámide: req(PHYSICS) = ∅.
        Se ejecuta sin validaciones previas.
        """
        result = mic.project_intent(
            "mock_physics", {"val": 42}, _build_context()
        )

        assert_mic_success(result, Stratum.PHYSICS, "PHYSICS con contexto vacío")
        assert result["val"] == 42

    def test_physics_allowed_with_any_validated(self, mic):
        """PHYSICS permitido sin importar qué estratos estén validados."""
        # Incluso con estratos "superiores" validados
        result = mic.project_intent(
            "mock_physics",
            {"val": 1},
            _build_context(validated={Stratum.WISDOM}),
        )

        assert result[MICResultKeys.SUCCESS] is True

    # ── TACTICS: req = {PHYSICS} ───────────────────────────────────────

    def test_tactics_blocked_without_physics(self, mic):
        """TACTICS sin PHYSICS → violación de jerarquía."""
        result = mic.project_intent(
            "mock_tactics", {"items": ["a"]}, _build_context()
        )

        assert_mic_hierarchy_violation(
            result,
            expected_required={Stratum.PHYSICS},
            context="TACTICS sin validación",
        )

    def test_tactics_allowed_with_physics(self, mic):
        """TACTICS con PHYSICS validado → permitido."""
        result = mic.project_intent(
            "mock_tactics",
            {"items": ["a", "b"]},
            _build_context(validated={Stratum.PHYSICS}),
        )

        assert_mic_success(result, Stratum.TACTICS)
        assert result["items"] == ["a", "b"]

    def test_tactics_blocked_with_wrong_stratum(self, mic):
        """TACTICS con STRATEGY validado pero no PHYSICS → bloqueado."""
        result = mic.project_intent(
            "mock_tactics",
            {"items": []},
            _build_context(validated={Stratum.STRATEGY}),
        )

        assert result[MICResultKeys.SUCCESS] is False

    # ── STRATEGY: req = {PHYSICS, TACTICS} ─────────────────────────────

    def test_strategy_blocked_without_any(self, mic):
        """STRATEGY sin validaciones → bloqueado."""
        result = mic.project_intent(
            "mock_strategy", {"amount": 100.0}, _build_context()
        )

        assert result[MICResultKeys.SUCCESS] is False

    def test_strategy_blocked_with_only_physics(self, mic):
        """STRATEGY con solo PHYSICS (falta TACTICS) → bloqueado."""
        result = mic.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            _build_context(validated={Stratum.PHYSICS}),
        )

        assert result[MICResultKeys.SUCCESS] is False

    def test_strategy_blocked_with_only_tactics(self, mic):
        """STRATEGY con solo TACTICS (falta PHYSICS) → bloqueado."""
        result = mic.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            _build_context(validated={Stratum.TACTICS}),
        )

        assert result[MICResultKeys.SUCCESS] is False

    def test_strategy_allowed_with_physics_and_tactics(self, mic):
        """STRATEGY con PHYSICS + TACTICS → permitido."""
        result = mic.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            _build_context(validated={Stratum.PHYSICS, Stratum.TACTICS}),
        )

        assert_mic_success(result, Stratum.STRATEGY)
        assert result["amount"] == 100.0

    # ── WISDOM: req = {PHYSICS, TACTICS, STRATEGY} ─────────────────────

    def test_wisdom_blocked_without_strategy(self, mic):
        """WISDOM sin STRATEGY → bloqueado."""
        result = mic.project_intent(
            "mock_wisdom",
            {"decision": "proceed"},
            _build_context(validated={Stratum.PHYSICS, Stratum.TACTICS}),
        )

        assert result[MICResultKeys.SUCCESS] is False

    def test_wisdom_blocked_with_gap_in_chain(self, mic):
        """WISDOM con PHYSICS + STRATEGY pero sin TACTICS → bloqueado."""
        result = mic.project_intent(
            "mock_wisdom",
            {"decision": "proceed"},
            _build_context(validated={Stratum.PHYSICS, Stratum.STRATEGY}),
        )

        assert result[MICResultKeys.SUCCESS] is False

    def test_wisdom_allowed_with_full_chain(self, mic):
        """WISDOM con la cadena completa → permitido."""
        result = mic.project_intent(
            "mock_wisdom",
            {"decision": "proceed"},
            _build_context(validated={
                Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY,
            }),
        )

        assert_mic_success(result, Stratum.WISDOM)
        assert result["decision"] == "proceed"

    # ── MATRIZ COMPLETA DE PERMISOS ────────────────────────────────────

    @pytest.mark.parametrize(
        "target, validated, expected_success",
        [
            # PHYSICS: siempre permitido (elemento mínimo)
            (Stratum.PHYSICS, set(), True),
            (Stratum.PHYSICS, {Stratum.TACTICS}, True),
            (Stratum.PHYSICS, {Stratum.STRATEGY, Stratum.WISDOM}, True),
            (Stratum.PHYSICS, {Stratum.PHYSICS}, True),  # Idempotente

            # TACTICS: necesita PHYSICS
            (Stratum.TACTICS, set(), False),
            (Stratum.TACTICS, {Stratum.PHYSICS}, True),
            (Stratum.TACTICS, {Stratum.STRATEGY}, False),
            (Stratum.TACTICS, {Stratum.WISDOM}, False),
            (Stratum.TACTICS, {Stratum.PHYSICS, Stratum.WISDOM}, True),

            # STRATEGY: necesita PHYSICS + TACTICS
            (Stratum.STRATEGY, set(), False),
            (Stratum.STRATEGY, {Stratum.PHYSICS}, False),
            (Stratum.STRATEGY, {Stratum.TACTICS}, False),
            (Stratum.STRATEGY, {Stratum.PHYSICS, Stratum.TACTICS}, True),
            (Stratum.STRATEGY, {Stratum.PHYSICS, Stratum.WISDOM}, False),
            (Stratum.STRATEGY, {Stratum.TACTICS, Stratum.WISDOM}, False),

            # WISDOM: necesita todo (elemento máximo)
            (Stratum.WISDOM, set(), False),
            (Stratum.WISDOM, {Stratum.PHYSICS}, False),
            (Stratum.WISDOM, {Stratum.PHYSICS, Stratum.TACTICS}, False),
            (Stratum.WISDOM, {Stratum.PHYSICS, Stratum.STRATEGY}, False),
            (Stratum.WISDOM, {Stratum.TACTICS, Stratum.STRATEGY}, False),
            (Stratum.WISDOM, {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}, True),
        ],
        ids=[
            "phys_empty", "phys_with_tactics", "phys_with_extras", "phys_idempotent",
            "tact_empty", "tact_with_phys", "tact_wrong_strat", "tact_wrong_wisd",
            "tact_phys_wisd",
            "strat_empty", "strat_only_phys", "strat_only_tact",
            "strat_phys_tact", "strat_phys_wisd", "strat_tact_wisd",
            "wisd_empty", "wisd_only_phys", "wisd_phys_tact",
            "wisd_phys_strat", "wisd_tact_strat", "wisd_full_chain",
        ],
    )
    def test_permission_matrix(self, mic, target, validated, expected_success):
        """
        Tabla de verdad exhaustiva: (target, validated) → allowed.
        Verifica la regla G₁ en todo el espacio de combinaciones relevantes.
        """
        vector_name, payload = STRATUM_VECTOR_MAP[target]
        result = mic.project_intent(
            vector_name, payload, _build_context(validated=validated)
        )

        validated_str = f"{{{', '.join(s.name for s in validated)}}}" if validated else "∅"
        
        assert result[MICResultKeys.SUCCESS] is expected_success, (
            f"Fallo en matriz de permisos:\n"
            f"  target = {target.name}\n"
            f"  validated = {validated_str}\n"
            f"  req(target) = {{{', '.join(s.name for s in DIKWPyramid.get_requirements(target))}}}\n"
            f"  expected = {'allowed' if expected_success else 'blocked'}\n"
            f"  actual = {'allowed' if result[MICResultKeys.SUCCESS] else 'blocked'}"
        )


# =============================================================================
# TEST: LÓGICA DEL GATEKEEPER
# =============================================================================

class TestMICGatekeeperLogic:
    """
    Pruebas de la lógica del gatekeeper: override, validación
    de actualizaciones, y contenido de errores.
    """

    # ── OVERRIDE (Regla G₂) ────────────────────────────────────────────

    def test_override_bypasses_strategy(self, mic):
        """Override permite STRATEGY sin validaciones previas."""
        result = mic.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            _build_context(force_override=True),
        )

        assert result[MICResultKeys.SUCCESS] is True

    def test_override_bypasses_wisdom(self, mic):
        """Override permite WISDOM sin ninguna validación."""
        result = mic.project_intent(
            "mock_wisdom",
            {"decision": "emergency"},
            _build_context(force_override=True),
        )

        assert result[MICResultKeys.SUCCESS] is True

    @pytest.mark.parametrize(
        "target",
        list(Stratum),
        ids=[s.name for s in Stratum],
    )
    def test_override_works_for_all_strata(self, mic, target):
        """Regla G₂: override es universal sobre todos los estratos."""
        vector_name, payload = STRATUM_VECTOR_MAP[target]
        result = mic.project_intent(
            vector_name, payload, _build_context(force_override=True)
        )

        assert result[MICResultKeys.SUCCESS] is True, (
            f"Override falló para {target.name}"
        )

    def test_override_false_does_not_bypass(self, mic):
        """force_override=False no activa bypass."""
        ctx = _build_context(force_override=False)
        
        result = mic.project_intent("mock_strategy", {"amount": 1.0}, ctx)
        
        assert result[MICResultKeys.SUCCESS] is False

    def test_override_with_empty_validated_works(self, mic):
        """Override con validated vacío funciona."""
        ctx = {
            MICContextKeys.VALIDATED_STRATA: set(),
            MICContextKeys.FORCE_OVERRIDE: True,
        }
        
        result = mic.project_intent("mock_wisdom", {"decision": "x"}, ctx)
        
        assert result[MICResultKeys.SUCCESS] is True

    # ── ACTUALIZACIÓN DE VALIDACIÓN ────────────────────────────────────

    def test_success_returns_validation_update(self, mic):
        """Ejecución exitosa incluye el estrato como actualización."""
        result = mic.project_intent(
            "mock_physics", {"val": 1}, _build_context()
        )

        assert MICResultKeys.VALIDATION_UPDATE in result
        assert result[MICResultKeys.VALIDATION_UPDATE] == Stratum.PHYSICS

    def test_permission_failure_no_validation_update(self, mic):
        """Fallo por permisos → sin actualización de validación."""
        result = mic.project_intent(
            "mock_strategy", {"amount": 100.0}, _build_context()
        )

        assert result[MICResultKeys.SUCCESS] is False
        assert result.get(MICResultKeys.VALIDATION_UPDATE) is None

    @pytest.mark.parametrize("stratum", list(Stratum), ids=[s.name for s in Stratum])
    def test_validation_update_matches_stratum(self, mic, stratum):
        """La actualización siempre corresponde al estrato del vector."""
        vector_name, payload = STRATUM_VECTOR_MAP[stratum]
        
        # Usar override para garantizar éxito
        result = mic.project_intent(
            vector_name, payload, _build_context(force_override=True)
        )
        
        assert result[MICResultKeys.VALIDATION_UPDATE] == stratum

    # ── CONTENIDO DEL ERROR ────────────────────────────────────────────

    def test_error_includes_required_strata(self, mic):
        """
        Error de jerarquía incluye los estratos requeridos
        para que el caller pueda corregir la secuencia.
        """
        result = mic.project_intent(
            "mock_strategy", {"amount": 100.0}, _build_context()
        )

        assert result[MICResultKeys.SUCCESS] is False
        assert MICResultKeys.REQUIRED_STRATA in result
        
        required = set(result[MICResultKeys.REQUIRED_STRATA])
        assert "PHYSICS" in required
        assert "TACTICS" in required

    def test_error_message_is_descriptive(self, mic):
        """El mensaje de error es descriptivo y menciona la violación."""
        result = mic.project_intent(
            "mock_wisdom", {"decision": "x"}, _build_context()
        )
        
        error_msg = result[MICResultKeys.ERROR]
        
        assert "WISDOM" in error_msg or "wisdom" in error_msg.lower()
        assert MICErrorMessages.HIERARCHY_VIOLATION in error_msg


# =============================================================================
# TEST: PROPIEDADES ALGEBRAICAS DEL ORDEN
# =============================================================================

class TestMICAlgebraicProperties:
    """
    Pruebas de las propiedades algebraicas del orden parcial
    inducido por la pirámide DIKW.

    El conjunto (P = Stratum, <) forma una cadena total (orden lineal):
      PHYSICS < TACTICS < STRATEGY < WISDOM

    Propiedades verificadas: A₁–A₇ del docstring de módulo.
    """

    @pytest.fixture
    def registry(self) -> MockMICRegistry:
        return MockMICRegistry()

    # ── A₁: IRREFLEXIVIDAD ────────────────────────────────────────────

    @pytest.mark.parametrize("stratum", list(Stratum), ids=[s.name for s in Stratum])
    def test_irreflexive_no_self_requirement(self, registry, stratum):
        """A₁: ∀ s ∈ P, s ∉ req(s). Ningún estrato se requiere a sí mismo."""
        required = registry.get_required_strata(stratum)
        assert stratum not in required, (
            f"{stratum.name} se requiere a sí mismo (viola irreflexividad)"
        )

    # ── A₂: ANTISIMETRÍA ──────────────────────────────────────────────

    def test_antisymmetric(self, registry):
        """A₂: s₁ ∈ req(s₂) → s₂ ∉ req(s₁). No hay dependencias circulares."""
        for s1 in Stratum:
            for s2 in Stratum:
                if s1 != s2:
                    req1 = registry.get_required_strata(s1)
                    req2 = registry.get_required_strata(s2)
                    if s2 in req1:
                        assert s1 not in req2, (
                            f"Antisimetría violada: {s1.name} ↔ {s2.name}"
                        )

    def test_no_circular_dependencies(self, registry):
        """
        Verificación explícita de ausencia de ciclos.
        
        Para cada estrato, seguimos la cadena de requisitos
        y verificamos que nunca volvemos al origen.
        """
        for start in Stratum:
            visited = set()
            to_visit = list(registry.get_required_strata(start))
            
            while to_visit:
                current = to_visit.pop()
                if current == start:
                    pytest.fail(f"Ciclo detectado desde {start.name}")
                if current not in visited:
                    visited.add(current)
                    to_visit.extend(registry.get_required_strata(current))

    # ── A₃: TRANSITIVIDAD ─────────────────────────────────────────────

    def test_transitive_chain_inclusion(self, registry):
        """
        A₃: req(WISDOM) ⊇ req(STRATEGY) ⊇ req(TACTICS) ⊇ req(PHYSICS).
        Inclusión de cadena monotónica creciente.
        """
        req_phys = registry.get_required_strata(Stratum.PHYSICS)
        req_tact = registry.get_required_strata(Stratum.TACTICS)
        req_strat = registry.get_required_strata(Stratum.STRATEGY)
        req_wisd = registry.get_required_strata(Stratum.WISDOM)

        assert req_phys.issubset(req_tact), "req(PHYSICS) ⊄ req(TACTICS)"
        assert req_tact.issubset(req_strat), "req(TACTICS) ⊄ req(STRATEGY)"
        assert req_strat.issubset(req_wisd), "req(STRATEGY) ⊄ req(WISDOM)"

    def test_transitive_closure_correct(self, registry):
        """
        La clausura transitiva de los requisitos es correcta.
        
        Si PHYSICS ∈ req(TACTICS) y TACTICS ∈ req(STRATEGY),
        entonces PHYSICS ∈ req(STRATEGY).
        """
        # PHYSICS está en requisitos de todos los superiores
        for stratum in [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
            assert Stratum.PHYSICS in registry.get_required_strata(stratum), (
                f"PHYSICS no está en req({stratum.name}) - falla transitividad"
            )

    # ── A₄: TOTALIDAD (CADENA) ────────────────────────────────────────

    def test_total_order_comparability(self, registry):
        """
        A₄: ∀ s₁ ≠ s₂, s₁ ∈ req(s₂) ∨ s₂ ∈ req(s₁).
        Todo par de estratos es comparable (orden total).
        """
        strata = list(Stratum)
        for i, s1 in enumerate(strata):
            for s2 in strata[i + 1:]:
                req1 = registry.get_required_strata(s1)
                req2 = registry.get_required_strata(s2)
                
                comparable = s1 in req2 or s2 in req1
                assert comparable, (
                    f"Par incomparable: {s1.name}, {s2.name}\n"
                    f"La pirámide DIKW requiere orden total"
                )

    def test_order_is_linear(self, registry):
        """
        Verificación directa de que el orden es una cadena lineal.
        
        Esto significa que existe una única secuencia
        s₀ < s₁ < s₂ < s₃ que ordena todos los elementos.
        """
        expected_order = [
            Stratum.PHYSICS,
            Stratum.TACTICS,
            Stratum.STRATEGY,
            Stratum.WISDOM,
        ]
        
        # Verificar que cada elemento tiene requisitos exactos
        for i, stratum in enumerate(expected_order):
            expected_req = set(expected_order[:i])
            actual_req = registry.get_required_strata(stratum)
            
            assert actual_req == expected_req, (
                f"Requisitos incorrectos para {stratum.name}:\n"
                f"  esperado: {{{', '.join(s.name for s in expected_req)}}}\n"
                f"  actual: {{{', '.join(s.name for s in actual_req)}}}"
            )

    # ── A₅: MINIMALIDAD ───────────────────────────────────────────────

    def test_physics_is_minimum(self, registry):
        """A₅: |req(PHYSICS)| = 0. Elemento mínimo de la cadena."""
        assert registry.get_required_strata(Stratum.PHYSICS) == set()

    def test_physics_in_all_higher_requirements(self, registry):
        """Corolario de A₅: PHYSICS ∈ req(s) para todo s > PHYSICS."""
        for stratum in [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
            assert Stratum.PHYSICS in registry.get_required_strata(stratum), (
                f"PHYSICS ausente en req({stratum.name})"
            )

    def test_only_physics_has_empty_requirements(self, registry):
        """PHYSICS es el único estrato con requisitos vacíos."""
        for stratum in Stratum:
            req = registry.get_required_strata(stratum)
            if stratum == Stratum.PHYSICS:
                assert req == set()
            else:
                assert len(req) > 0, (
                    f"{stratum.name} tiene requisitos vacíos pero no es PHYSICS"
                )

    # ── A₆: MAXIMALIDAD ───────────────────────────────────────────────

    def test_wisdom_is_maximum(self, registry):
        """A₆: req(WISDOM) = P \\ {WISDOM}. Elemento máximo."""
        expected = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
        assert registry.get_required_strata(Stratum.WISDOM) == expected

    def test_wisdom_requires_all_others(self, registry):
        """WISDOM requiere todos los demás estratos."""
        wisdom_req = registry.get_required_strata(Stratum.WISDOM)
        all_others = set(Stratum) - {Stratum.WISDOM}
        
        assert wisdom_req == all_others

    def test_nothing_requires_wisdom(self, registry):
        """Ningún estrato requiere WISDOM (es el máximo)."""
        for stratum in Stratum:
            req = registry.get_required_strata(stratum)
            assert Stratum.WISDOM not in req, (
                f"{stratum.name} requiere WISDOM (viola maximalidad)"
            )

    # ── A₇: BIEN FUNDADO ──────────────────────────────────────────────

    def test_well_founded_finite(self, registry):
        """
        A₇: El orden es bien fundado.
        
        Para un conjunto finito, esto es trivialmente cierto
        si no hay ciclos (verificado en A₂).
        """
        # Verificamos que podemos alcanzar el mínimo desde cualquier punto
        for stratum in Stratum:
            visited = {stratum}
            current_level = registry.get_required_strata(stratum)
            
            while current_level:
                next_level = set()
                for s in current_level:
                    visited.add(s)
                    next_level.update(registry.get_required_strata(s))
                current_level = next_level - visited
            
            # Siempre debemos llegar a PHYSICS o empezar desde él
            if stratum != Stratum.PHYSICS:
                assert Stratum.PHYSICS in visited

    # ── CARDINALIDAD MONÓTONA ──────────────────────────────────────────

    def test_cardinality_strictly_increasing(self, registry):
        """
        |req(s)| es estrictamente creciente a lo largo de la cadena.
        Esto induce una biyección con {0, 1, 2, 3}.
        """
        cardinalities = [
            (stratum, len(registry.get_required_strata(stratum)))
            for stratum in Stratum
        ]

        sorted_by_card = sorted(cardinalities, key=lambda x: x[1])
        expected_order = [
            Stratum.PHYSICS, Stratum.TACTICS,
            Stratum.STRATEGY, Stratum.WISDOM,
        ]

        actual_order = [s for s, _ in sorted_by_card]
        assert actual_order == expected_order, (
            f"Orden por cardinalidad incorrecto:\n"
            f"  esperado: {[s.name for s in expected_order]}\n"
            f"  actual: {[s.name for s in actual_order]}"
        )

        # Estrictamente creciente (sin empates)
        cards = [c for _, c in sorted_by_card]
        for i in range(len(cards) - 1):
            assert cards[i] < cards[i + 1], (
                f"Cardinalidades no estrictamente crecientes: {cards}"
            )

    # ── VERIFICACIÓN COMPLETA ─────────────────────────────────────────

    def test_all_algebraic_properties(self, registry):
        """Verificación compuesta de todas las propiedades."""
        results = verify_order_properties(registry)
        
        assert results["all_valid"], (
            f"Propiedades algebraicas violadas:\n"
            f"  A₁ (irreflexivo): {results['A1_irreflexive']}\n"
            f"  A₂ (antisimétrico): {results['A2_antisymmetric']}\n"
            f"  A₃ (transitivo): {results['A3_transitive']}\n"
            f"  A₄ (total): {results['A4_total']}\n"
            f"  A₅ (mínimo): {results['A5_minimum']}\n"
            f"  A₆ (máximo): {results['A6_maximum']}\n"
            f"  A₇ (bien fundado): {results['A7_well_founded']}"
        )


# =============================================================================
# TEST: MANEJO DE ERRORES (Regla G₄)
# =============================================================================

class TestMICErrorHandling:
    """
    Pruebas de manejo de errores en la ejecución de handlers.

    Regla G₄: si el handler lanza una excepción, el resultado
    es {success: False} con tipo y mensaje del error.
    """

    def test_handler_exception_caught(self, mic_failing):
        """Excepción en handler se captura y empaqueta en el resultado."""
        result = mic_failing.project_intent(
            "failing_vector", {}, _build_context()
        )

        assert result[MICResultKeys.SUCCESS] is False
        assert result[MICResultKeys.ERROR_TYPE] == "RuntimeError"
        assert MICErrorMessages.HANDLER_FAILED in result[MICResultKeys.ERROR]

    def test_handler_exception_no_validation_update(self, mic_failing):
        """Excepción en handler → sin actualización de validación."""
        result = mic_failing.project_intent(
            "failing_vector", {}, _build_context()
        )

        assert result.get(MICResultKeys.VALIDATION_UPDATE) is None

    @pytest.mark.parametrize(
        "exception_type, message",
        [
            (ValueError, "Invalid value"),
            (TypeError, "Type mismatch"),
            (KeyError, "Missing key"),
            (AttributeError, "No attribute"),
            (IndexError, "Index out of range"),
            (ZeroDivisionError, "Division by zero"),
        ],
        ids=["ValueError", "TypeError", "KeyError", 
             "AttributeError", "IndexError", "ZeroDivisionError"],
    )
    def test_various_exceptions_caught(self, exception_type, message):
        """Cualquier tipo de excepción es capturada."""
        registry = MockMICRegistry()
        
        def raising_handler(**kwargs):
            raise exception_type(message)
        
        registry.register_vector("raising", Stratum.PHYSICS, raising_handler)
        
        result = registry.project_intent("raising", {}, _build_context())
        
        assert result[MICResultKeys.SUCCESS] is False
        assert result[MICResultKeys.ERROR_TYPE] == exception_type.__name__
        assert message in result[MICResultKeys.ERROR]

    def test_missing_payload_key_does_not_crash(self, mic):
        """
        Payload sin la clave esperada por el handler.
        El TypeError se captura, no se propaga.
        """
        result = mic.project_intent(
            "mock_physics", {}, _build_context()  # Falta 'val'
        )

        assert MICResultKeys.SUCCESS in result
        if result[MICResultKeys.SUCCESS] is False:
            assert MICResultKeys.ERROR in result

    def test_malformed_handler_result(self):
        """Handler que retorna tipo incorrecto."""
        registry = MockMICRegistry()
        registry.register_vector("malformed", Stratum.PHYSICS, _malformed_handler)
        
        result = registry.project_intent("malformed", {}, _build_context())
        
        assert result[MICResultKeys.SUCCESS] is False

    def test_handler_without_success_key(self):
        """Handler que retorna dict sin 'success'."""
        registry = MockMICRegistry()
        registry.register_vector("no_success", Stratum.PHYSICS, _no_success_handler)
        
        result = registry.project_intent("no_success", {}, _build_context())
        
        # Debe tener success (se agrega automáticamente o falla)
        assert MICResultKeys.SUCCESS in result

    def test_result_always_has_success_key(self, mic):
        """
        Invariante G₅: todo resultado contiene 'success' (bool).
        """
        # Éxito
        ok = mic.project_intent(
            "mock_physics", {"val": 1}, _build_context()
        )
        assert isinstance(ok[MICResultKeys.SUCCESS], bool)

        # Fallo por permisos
        fail = mic.project_intent(
            "mock_strategy", {"amount": 1.0}, _build_context()
        )
        assert isinstance(fail[MICResultKeys.SUCCESS], bool)


# =============================================================================
# TEST: FLUJOS SECUENCIALES
# =============================================================================

class TestMICFlowScenarios:
    """
    Pruebas de flujos end-to-end que simulan el uso real
    de la pirámide DIKW.
    """

    def test_sequential_validation_flow(self, mic):
        """
        Flujo canónico: PHYSICS → TACTICS → STRATEGY → WISDOM.
        Cada paso habilita el siguiente.
        """
        validated = set()

        for stratum in DIKWPyramid.ORDER:
            vector_name, payload = STRATUM_VECTOR_MAP[stratum]
            result = mic.project_intent(
                vector_name, payload, _build_context(validated=validated)
            )

            assert result[MICResultKeys.SUCCESS] is True, (
                f"Fallo en {stratum.name} con validated="
                f"{{{', '.join(s.name for s in validated)}}}"
            )
            validated.add(result[MICResultKeys.VALIDATION_UPDATE])

        expected = set(Stratum)
        assert validated == expected

    def test_reverse_order_fails_except_physics(self, mic):
        """Intentar en orden inverso: todo falla excepto PHYSICS."""
        results = {}
        
        for stratum in reversed(DIKWPyramid.ORDER):
            vector_name, payload = STRATUM_VECTOR_MAP[stratum]
            result = mic.project_intent(
                vector_name, payload, _build_context()
            )
            results[stratum] = result[MICResultKeys.SUCCESS]

        assert results[Stratum.WISDOM] is False
        assert results[Stratum.STRATEGY] is False
        assert results[Stratum.TACTICS] is False
        assert results[Stratum.PHYSICS] is True  # Siempre permitido

    def test_skip_level_blocked(self, mic):
        """Saltar TACTICS e ir directo a STRATEGY → bloqueado."""
        result = mic.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            _build_context(validated={Stratum.PHYSICS}),
        )

        assert result[MICResultKeys.SUCCESS] is False

    def test_emergency_override_flow(self, mic):
        """
        Flujo de emergencia: WISDOM directamente con override.
        """
        result = mic.project_intent(
            "mock_wisdom",
            {"decision": "emergency_override"},
            _build_context(force_override=True),
        )

        assert result[MICResultKeys.SUCCESS] is True

    def test_partial_progress_resumes(self, mic):
        """
        Progreso parcial: PHYSICS validado, se retoma desde TACTICS.
        """
        # Ya validamos PHYSICS
        validated = {Stratum.PHYSICS}

        # Retomar desde TACTICS
        result = mic.project_intent(
            "mock_tactics",
            {"items": ["resumed"]},
            _build_context(validated=validated),
        )

        assert result[MICResultKeys.SUCCESS] is True

    def test_can_reexecute_validated_stratum(self, mic):
        """
        Podemos re-ejecutar un estrato ya validado.
        """
        validated = {Stratum.PHYSICS}
        
        # Re-ejecutar PHYSICS (ya validado)
        result = mic.project_intent(
            "mock_physics",
            {"val": 99},
            _build_context(validated=validated),
        )
        
        assert result[MICResultKeys.SUCCESS] is True
        assert result["val"] == 99

    def test_full_pipeline_with_metrics(self, mic):
        """
        Pipeline completo acumulando resultados.
        """
        validated = set()
        metrics = {}
        
        for stratum in DIKWPyramid.ORDER:
            vector_name, payload = STRATUM_VECTOR_MAP[stratum]
            result = mic.project_intent(
                vector_name, payload, _build_context(validated=validated)
            )
            
            assert result[MICResultKeys.SUCCESS] is True
            validated.add(result[MICResultKeys.VALIDATION_UPDATE])
            metrics[stratum] = result.get(MICResultKeys.STRATUM)
        
        # Verificar que cada resultado reportó su estrato
        assert metrics[Stratum.PHYSICS] == "PHYSICS"
        assert metrics[Stratum.TACTICS] == "TACTICS"


# =============================================================================
# TEST: DETERMINISMO E IDEMPOTENCIA
# =============================================================================

class TestMICDeterminism:
    """
    Pruebas de determinismo: misma entrada → misma salida,
    e idempotencia de ejecuciones repetidas.
    """

    def test_same_input_same_output(self, mic):
        """Determinismo: f(x) = f(x) para todo x."""
        ctx = _build_context(validated={Stratum.PHYSICS})
        payload = {"items": ["x", "y"]}

        r1 = mic.project_intent("mock_tactics", dict(payload), dict(ctx))
        r2 = mic.project_intent("mock_tactics", dict(payload), dict(ctx))

        assert r1[MICResultKeys.SUCCESS] == r2[MICResultKeys.SUCCESS]
        assert r1["items"] == r2["items"]

    def test_physics_idempotent(self, mic):
        """
        Ejecutar PHYSICS dos veces produce resultados equivalentes.
        """
        ctx = _build_context()
        payload = {"val": 42}

        r1 = mic.project_intent("mock_physics", dict(payload), dict(ctx))
        r2 = mic.project_intent("mock_physics", dict(payload), dict(ctx))

        assert r1[MICResultKeys.SUCCESS] == r2[MICResultKeys.SUCCESS] is True
        assert r1["val"] == r2["val"] == 42

    def test_repeated_execution_all_strata(self, mic):
        """Ejecución repetida para todos los estratos."""
        for stratum in Stratum:
            vector_name, payload = STRATUM_VECTOR_MAP[stratum]
            ctx = _build_context(force_override=True)
            
            results = [
                mic.project_intent(vector_name, dict(payload), dict(ctx))
                for _ in range(3)
            ]
            
            # Todos deben ser exitosos e iguales
            for r in results:
                assert r[MICResultKeys.SUCCESS] is True
            
            assert all(
                r[MICResultKeys.VALIDATION_UPDATE] == stratum
                for r in results
            )

    def test_context_immutability(self, mic):
        """
        project_intent NO debe mutar el contexto de entrada.
        """
        original_validated = {Stratum.PHYSICS}
        ctx = _build_context(validated=original_validated.copy())
        ctx_snapshot = deepcopy(ctx)

        mic.project_intent("mock_tactics", {"items": []}, ctx)

        assert ctx == ctx_snapshot
        assert ctx[MICContextKeys.VALIDATED_STRATA] == {Stratum.PHYSICS}

    def test_payload_immutability(self, mic):
        """
        project_intent NO debe mutar el payload de entrada.
        """
        payload = {"val": 100}
        payload_snapshot = deepcopy(payload)
        
        mic.project_intent("mock_physics", payload, _build_context())
        
        assert payload == payload_snapshot


# =============================================================================
# TEST: CASOS LÍMITE
# =============================================================================

class TestMICEdgeCases:
    """Pruebas de casos límite y entradas anómalas."""

    def test_extra_validated_strata_ignored(self, mic):
        """
        Estratos extra en validated_strata no interfieren.
        """
        # WISDOM extra no afecta TACTICS (que solo necesita PHYSICS)
        ctx = _build_context(validated={Stratum.PHYSICS, Stratum.WISDOM})

        result = mic.project_intent("mock_tactics", {"items": []}, ctx)

        assert result[MICResultKeys.SUCCESS] is True

    def test_empty_context_dict(self, mic):
        """
        Contexto completamente vacío (sin 'validated_strata').
        Solo PHYSICS debe estar permitido.
        """
        result_physics = mic.project_intent("mock_physics", {"val": 1}, {})
        result_tactics = mic.project_intent("mock_tactics", {"items": []}, {})

        assert result_physics[MICResultKeys.SUCCESS] is True
        assert result_tactics[MICResultKeys.SUCCESS] is False

    def test_validated_strata_none(self, mic):
        """
        validated_strata = None: tratado como set vacío.
        """
        ctx = {MICContextKeys.VALIDATED_STRATA: None}
        
        result_physics = mic.project_intent("mock_physics", {"val": 1}, ctx)
        result_tactics = mic.project_intent("mock_tactics", {"items": []}, ctx)
        
        assert result_physics[MICResultKeys.SUCCESS] is True
        assert result_tactics[MICResultKeys.SUCCESS] is False

    def test_payload_with_extra_keys(self, mic):
        """
        Payload con claves adicionales no esperadas.
        """
        result = mic.project_intent(
            "mock_physics",
            {"val": 1, "unexpected": "extra", "another": 123},
            _build_context(),
        )

        # Puede fallar por TypeError, pero debe retornar resultado válido
        assert MICResultKeys.SUCCESS in result

    def test_large_validated_set(self, mic):
        """
        validated con todos los estratos: WISDOM debe estar permitido.
        """
        all_strata = set(Stratum)
        result = mic.project_intent(
            "mock_wisdom",
            {"decision": "saturated"},
            _build_context(validated=all_strata),
        )

        assert result[MICResultKeys.SUCCESS] is True

    def test_payload_empty_dict(self, mic):
        """Payload vacío puede causar error en handler."""
        result = mic.project_intent(
            "mock_physics",
            {},  # Falta 'val'
            _build_context(),
        )
        
        # Debe manejar sin crash
        assert MICResultKeys.SUCCESS in result

    def test_vector_name_empty_string(self, mic):
        """Vector con nombre vacío."""
        with pytest.raises(ValueError):
            mic.project_intent("", {}, _build_context())

    def test_context_with_extra_fields(self, mic):
        """Contexto con campos adicionales ignorados."""
        ctx = _build_context(
            validated={Stratum.PHYSICS},
            extra_fields={"custom_field": "value", "another": 123}
        )
        
        result = mic.project_intent("mock_tactics", {"items": []}, ctx)
        
        assert result[MICResultKeys.SUCCESS] is True


# =============================================================================
# TEST: CONCURRENCIA Y THREAD-SAFETY
# =============================================================================

class TestMICConcurrency:
    """
    Pruebas de comportamiento bajo concurrencia.
    """

    def test_concurrent_reads_safe(self, mic):
        """Lecturas concurrentes son seguras."""
        results = []
        
        def reader(vector_name):
            for _ in range(10):
                is_reg = mic.is_registered(vector_name)
                stratum = mic.get_vector_stratum(vector_name)
                results.append((is_reg, stratum))
        
        threads = [
            threading.Thread(target=reader, args=(name,))
            for name in ["mock_physics", "mock_tactics", "mock_strategy"]
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Todos los resultados deben ser válidos
        for is_reg, stratum in results:
            assert isinstance(is_reg, bool)

    def test_concurrent_projections(self, mic):
        """Proyecciones concurrentes funcionan correctamente."""
        results = {}
        
        def worker(stratum):
            vector_name, payload = STRATUM_VECTOR_MAP[stratum]
            ctx = _build_context(force_override=True)
            
            for i in range(5):
                result = mic.project_intent(vector_name, payload, ctx)
                results[(stratum, i)] = result[MICResultKeys.SUCCESS]
        
        threads = [
            threading.Thread(target=worker, args=(stratum,))
            for stratum in Stratum
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Todas las ejecuciones deben ser exitosas
        assert all(results.values())

    def test_concurrent_registrations(self):
        """Registros concurrentes son thread-safe."""
        registry = MockMICRegistry()
        
        def register_vectors(prefix):
            for i in range(10):
                registry.register_vector(
                    f"{prefix}_v{i}",
                    Stratum.PHYSICS,
                    _physics_handler
                )
        
        threads = [
            threading.Thread(target=register_vectors, args=(f"t{i}",))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Debe haber exactamente 50 vectores
        assert len(registry.get_registered_vectors()) == 50


# =============================================================================
# TEST: RENDIMIENTO Y STRESS
# =============================================================================

class TestMICPerformance:
    """Pruebas de rendimiento bajo carga."""

    def test_high_volume_projections(self, mic):
        """Alto volumen de proyecciones."""
        ctx = _build_context(force_override=True)
        
        start = time.perf_counter()
        
        for _ in range(1000):
            mic.project_intent("mock_physics", {"val": 1}, ctx)
        
        elapsed = time.perf_counter() - start
        
        # Debe completarse en tiempo razonable
        assert elapsed < 1.0, f"1000 proyecciones tardaron {elapsed:.2f}s"

    def test_large_registry(self):
        """Registry con muchos vectores."""
        registry = MockMICRegistry()
        
        # Registrar 1000 vectores
        for i in range(1000):
            def handler(val=i):
                return {MICResultKeys.SUCCESS: True, "val": val}
            registry.register_vector(f"vector_{i}", Stratum.PHYSICS, handler)
        
        # Verificar que todos están registrados
        assert len(registry.get_registered_vectors()) == 1000
        
        # Proyectar algunos
        for i in [0, 500, 999]:
            result = registry.project_intent(
                f"vector_{i}",
                {},
                _build_context()
            )
            assert result[MICResultKeys.SUCCESS] is True

    def test_projection_latency_bounded(self, mic_slow):
        """La latencia de proyección está acotada por el handler."""
        ctx = _build_context()
        
        start = time.perf_counter()
        result = mic_slow.project_intent("slow_vector", {"delay": 0.05}, ctx)
        elapsed = time.perf_counter() - start
        
        assert result[MICResultKeys.SUCCESS] is True
        # La latencia debe ser cercana al delay del handler
        assert 0.04 < elapsed < 0.2


# =============================================================================
# TEST: INTEGRACIÓN CON TELEMETRÍA
# =============================================================================

@pytest.mark.skipif(not HAS_TELEMETRY, reason="Telemetry not available")
class TestMICIntegrationWithTelemetry:
    """
    Pruebas de integración entre MIC y el sistema de telemetría.
    """

    @pytest.fixture
    def telemetry_ctx(self):
        return TelemetryContext()

    def test_mic_execution_traced_in_span(self, mic, telemetry_ctx):
        """La ejecución MIC dentro de un span se registra."""
        with telemetry_ctx.span("mic_physics_execution"):
            result = mic.project_intent(
                "mock_physics", {"val": 42}, _build_context()
            )

        assert result[MICResultKeys.SUCCESS] is True
        assert len(telemetry_ctx.root_spans) == 1
        assert telemetry_ctx.root_spans[0].name == "mic_physics_execution"

    def test_mic_failure_recorded_in_span(self, mic, telemetry_ctx):
        """Fallo de MIC se puede registrar como error en el span."""
        with telemetry_ctx.span("mic_strategy_attempt") as span:
            result = mic.project_intent(
                "mock_strategy", {"amount": 100.0}, _build_context()
            )

            if not result[MICResultKeys.SUCCESS]:
                span.metrics["mic_error"] = result.get(MICResultKeys.ERROR, "Unknown")

        root = telemetry_ctx.root_spans[0]
        assert "mic_error" in root.metrics
        assert MICErrorMessages.HIERARCHY_VIOLATION in root.metrics["mic_error"]

    def test_sequential_flow_produces_multiple_spans(self, mic, telemetry_ctx):
        """
        Flujo secuencial completo genera un span por estrato.
        """
        validated = set()

        for stratum in [Stratum.PHYSICS, Stratum.TACTICS]:
            vector_name, payload = STRATUM_VECTOR_MAP[stratum]
            with telemetry_ctx.span(f"mic_{stratum.name.lower()}"):
                result = mic.project_intent(
                    vector_name, payload,
                    _build_context(validated=validated),
                )
                if result[MICResultKeys.SUCCESS]:
                    validated.add(result[MICResultKeys.VALIDATION_UPDATE])

        assert len(telemetry_ctx.root_spans) == 2
        assert telemetry_ctx.root_spans[0].name == "mic_physics"
        assert telemetry_ctx.root_spans[1].name == "mic_tactics"

    def test_nested_spans_for_dependent_strata(self, mic, telemetry_ctx):
        """
        Spans anidados para representar dependencias.
        """
        with telemetry_ctx.span("pipeline") as outer:
            validated = set()
            
            for stratum in DIKWPyramid.ORDER:
                vector_name, payload = STRATUM_VECTOR_MAP[stratum]
                with telemetry_ctx.span(f"stratum_{stratum.name.lower()}"):
                    result = mic.project_intent(
                        vector_name, payload,
                        _build_context(validated=validated)
                    )
                    if result[MICResultKeys.SUCCESS]:
                        validated.add(stratum)
        
        # Un span raíz con 4 hijos
        assert len(telemetry_ctx.root_spans) == 1
        pipeline_span = telemetry_ctx.root_spans[0]
        assert pipeline_span.name == "pipeline"
        assert len(pipeline_span.children) == 4


# =============================================================================
# TEST: INVARIANTES ESTRUCTURALES DEL RESULTADO
# =============================================================================

class TestMICResultInvariants:
    """
    Pruebas de invariantes estructurales del resultado MIC.
    """

    @pytest.mark.parametrize(
        "scenario",
        ["success", "permission_failure", "handler_exception", "unknown_vector"],
        ids=["success", "perm_fail", "handler_exc", "unknown"],
    )
    def test_result_structure_valid(self, mic, mic_failing, scenario):
        """
        Todo resultado tiene estructura válida según las reglas.
        """
        if scenario == "success":
            result = mic.project_intent(
                "mock_physics", {"val": 1}, _build_context()
            )
        elif scenario == "permission_failure":
            result = mic.project_intent(
                "mock_strategy", {"amount": 1.0}, _build_context()
            )
        elif scenario == "handler_exception":
            result = mic_failing.project_intent(
                "failing_vector", {}, _build_context()
            )
        else:  # unknown_vector
            try:
                mic.project_intent("unknown", {}, _build_context())
                pytest.fail("Debería haber lanzado ValueError")
            except ValueError:
                return  # Correcto
        
        validation = validate_mic_result_structure(result)
        assert validation["is_valid"], (
            f"Estructura inválida para {scenario}:\n"
            f"  Violaciones: {validation['violations']}"
        )

    def test_success_result_complete(self, mic):
        """Resultado exitoso tiene todos los campos esperados."""
        result = mic.project_intent(
            "mock_physics", {"val": 42}, _build_context()
        )
        
        assert result[MICResultKeys.SUCCESS] is True
        assert MICResultKeys.VALIDATION_UPDATE in result
        assert result[MICResultKeys.VALIDATION_UPDATE] == Stratum.PHYSICS

    def test_failure_result_complete(self, mic):
        """Resultado fallido tiene todos los campos de error."""
        result = mic.project_intent(
            "mock_wisdom", {"decision": "x"}, _build_context()
        )
        
        assert result[MICResultKeys.SUCCESS] is False
        assert MICResultKeys.ERROR in result
        assert MICResultKeys.ERROR_TYPE in result
        assert MICResultKeys.REQUIRED_STRATA in result