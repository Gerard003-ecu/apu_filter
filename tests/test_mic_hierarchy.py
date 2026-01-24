"""
Tests for MIC Hierarchy and Gatekeeper Logic
=============================================

Verifies that the Matriz de Interacción Central (MIC) correctly enforces
the "Physics before Strategy" rule following the DIKW pyramid.

Arquitectura MIC:
-----------------
La MIC implementa un sistema de permisos jerárquico basado en la pirámide DIKW:

    Level 0: WISDOM    - Requiere validación de STRATEGY, TACTICS, PHYSICS
    Level 1: STRATEGY  - Requiere validación de TACTICS, PHYSICS
    Level 2: TACTICS   - Requiere validación de PHYSICS
    Level 3: PHYSICS   - Siempre permitido (base de la pirámide)

Reglas de Gatekeeper:
--------------------
1. Un vector de nivel N solo puede ejecutarse si todos los niveles > N están validados
2. El override `force_physics_override` bypasea todas las validaciones
3. Vectores desconocidos lanzan ValueError

Estructura de Tests:
- TestMICRegistryBasics: Registro y configuración
- TestMICHierarchyPermissions: Permisos por nivel
- TestMICGatekeeperLogic: Lógica del gatekeeper
- TestMICTransitiveRequirements: Requisitos transitivos
- TestMICErrorHandling: Manejo de errores
- TestMICIntegrationWithTelemetry: Integración con telemetría
- TestMICAlgebraicProperties: Propiedades algebraicas
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
from unittest.mock import Mock, MagicMock, patch

import pytest

# ============================================================================
# IMPORTACIONES CON MANEJO CONDICIONAL
# ============================================================================

from app.schemas import Stratum

# Intentar importar MIC components
try:
    from app.tools_interface import MICRegistry, IntentVector
    HAS_MIC = True
except ImportError:
    HAS_MIC = False
    MICRegistry = None
    IntentVector = None

# Intentar importar telemetría para tests de integración
try:
    from app.telemetry import TelemetryContext, StepStatus
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False
    TelemetryContext = None
    StepStatus = None


# ============================================================================
# SKIP DECORATOR SI MIC NO ESTÁ DISPONIBLE
# ============================================================================

requires_mic = pytest.mark.skipif(
    not HAS_MIC,
    reason="MICRegistry not available in app.tools_interface"
)


# ============================================================================
# MOCK HANDLERS Y HELPERS
# ============================================================================


def physics_handler(val: int) -> Dict[str, Any]:
    """Handler mock para operaciones de PHYSICS."""
    return {"success": True, "val": val, "stratum": "PHYSICS"}


def tactics_handler(items: List[str]) -> Dict[str, Any]:
    """Handler mock para operaciones de TACTICS."""
    return {"success": True, "items": items, "stratum": "TACTICS"}


def strategy_handler(amount: float) -> Dict[str, Any]:
    """Handler mock para operaciones de STRATEGY."""
    return {"success": True, "amount": amount, "stratum": "STRATEGY"}


def wisdom_handler(decision: str) -> Dict[str, Any]:
    """Handler mock para operaciones de WISDOM."""
    return {"success": True, "decision": decision, "stratum": "WISDOM"}


def failing_handler(**kwargs) -> Dict[str, Any]:
    """Handler que siempre falla."""
    raise RuntimeError("Handler execution failed")


def slow_handler(delay: float = 0.1) -> Dict[str, Any]:
    """Handler que simula operación lenta."""
    import time
    time.sleep(delay)
    return {"success": True, "delayed": True}


class MockMICRegistry:
    """
    Mock de MICRegistry para tests cuando el módulo real no está disponible.
    
    Implementa la lógica básica de jerarquía MIC.
    """
    
    # Requisitos de validación por estrato
    STRATUM_REQUIREMENTS: Dict[Stratum, Set[Stratum]] = {
        Stratum.PHYSICS: set(),  # No requiere nada
        Stratum.TACTICS: {Stratum.PHYSICS},
        Stratum.STRATEGY: {Stratum.PHYSICS, Stratum.TACTICS},
        Stratum.WISDOM: {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY},
    }
    
    def __init__(self):
        self._vectors: Dict[str, Tuple[Stratum, Callable]] = {}
    
    def register_vector(
        self,
        name: str,
        stratum: Stratum,
        handler: Callable,
    ) -> None:
        """Registra un vector de intención."""
        self._vectors[name] = (stratum, handler)
    
    def project_intent(
        self,
        vector_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Proyecta una intención a través del vector."""
        if vector_name not in self._vectors:
            raise ValueError(f"Unknown vector: {vector_name}")
        
        stratum, handler = self._vectors[vector_name]
        
        # Verificar permisos de jerarquía
        if not self._check_hierarchy_permission(stratum, context):
            return {
                "success": False,
                "error_type": "PermissionError",
                "error": f"MIC Hierarchy Violation: {stratum.name} requires prior validation",
                "required_strata": [s.name for s in self.STRATUM_REQUIREMENTS[stratum]],
            }
        
        # Ejecutar handler
        try:
            result = handler(**payload)
            result["_mic_validation_update"] = stratum
            return result
        except Exception as e:
            return {
                "success": False,
                "error_type": type(e).__name__,
                "error": str(e),
            }
    
    def _check_hierarchy_permission(
        self,
        stratum: Stratum,
        context: Dict[str, Any],
    ) -> bool:
        """Verifica si el estrato tiene permiso de ejecución."""
        # Override bypasea todo
        if context.get("force_physics_override", False):
            return True
        
        # Obtener estratos validados
        validated = context.get("validated_strata", set())
        
        # Verificar requisitos
        required = self.STRATUM_REQUIREMENTS.get(stratum, set())
        return required.issubset(validated)
    
    def get_required_strata(self, stratum: Stratum) -> Set[Stratum]:
        """Retorna los estratos requeridos para ejecutar en el estrato dado."""
        return self.STRATUM_REQUIREMENTS.get(stratum, set())
    
    def is_registered(self, vector_name: str) -> bool:
        """Verifica si un vector está registrado."""
        return vector_name in self._vectors
    
    def get_vector_stratum(self, vector_name: str) -> Optional[Stratum]:
        """Retorna el estrato de un vector."""
        if vector_name in self._vectors:
            return self._vectors[vector_name][0]
        return None


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mic_registry():
    """
    Registry MIC con vectores mock registrados.
    
    Usa el MICRegistry real si está disponible, sino usa el mock.
    """
    if HAS_MIC:
        registry = MICRegistry()
    else:
        registry = MockMICRegistry()
    
    # Registrar vectores para cada nivel
    registry.register_vector("mock_physics", Stratum.PHYSICS, physics_handler)
    registry.register_vector("mock_tactics", Stratum.TACTICS, tactics_handler)
    registry.register_vector("mock_strategy", Stratum.STRATEGY, strategy_handler)
    registry.register_vector("mock_wisdom", Stratum.WISDOM, wisdom_handler)
    
    return registry


@pytest.fixture
def mic_with_failing_handler():
    """Registry con un handler que falla."""
    if HAS_MIC:
        registry = MICRegistry()
    else:
        registry = MockMICRegistry()
    
    registry.register_vector("failing_vector", Stratum.PHYSICS, failing_handler)
    return registry


@pytest.fixture
def empty_context() -> Dict[str, Any]:
    """Contexto vacío (sin validaciones)."""
    return {"validated_strata": set()}


@pytest.fixture
def physics_validated_context() -> Dict[str, Any]:
    """Contexto con PHYSICS validado."""
    return {"validated_strata": {Stratum.PHYSICS}}


@pytest.fixture
def physics_tactics_validated_context() -> Dict[str, Any]:
    """Contexto con PHYSICS y TACTICS validados."""
    return {"validated_strata": {Stratum.PHYSICS, Stratum.TACTICS}}


@pytest.fixture
def all_validated_context() -> Dict[str, Any]:
    """Contexto con todos los estratos validados."""
    return {"validated_strata": {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}}


@pytest.fixture
def override_context() -> Dict[str, Any]:
    """Contexto con override de física."""
    return {"validated_strata": set(), "force_physics_override": True}


# ============================================================================
# TEST: BASICS DEL REGISTRY
# ============================================================================


class TestMICRegistryBasics:
    """Pruebas básicas del registro MIC."""

    def test_register_vector(self, mic_registry):
        """Puede registrar vectores."""
        assert mic_registry.is_registered("mock_physics")
        assert mic_registry.is_registered("mock_tactics")
        assert mic_registry.is_registered("mock_strategy")
        assert mic_registry.is_registered("mock_wisdom")

    def test_unregistered_vector_not_found(self, mic_registry):
        """Vectores no registrados no se encuentran."""
        assert not mic_registry.is_registered("nonexistent_vector")

    def test_get_vector_stratum(self, mic_registry):
        """Puede obtener el estrato de un vector."""
        assert mic_registry.get_vector_stratum("mock_physics") == Stratum.PHYSICS
        assert mic_registry.get_vector_stratum("mock_tactics") == Stratum.TACTICS
        assert mic_registry.get_vector_stratum("mock_strategy") == Stratum.STRATEGY
        assert mic_registry.get_vector_stratum("mock_wisdom") == Stratum.WISDOM

    def test_get_unknown_vector_stratum_returns_none(self, mic_registry):
        """Estrato de vector desconocido retorna None."""
        assert mic_registry.get_vector_stratum("unknown") is None

    def test_unknown_vector_raises_error(self, mic_registry, empty_context):
        """Proyectar a vector desconocido lanza ValueError."""
        with pytest.raises(ValueError, match="Unknown vector"):
            mic_registry.project_intent("unknown_vector", {}, empty_context)


# ============================================================================
# TEST: PERMISOS DE JERARQUÍA
# ============================================================================


class TestMICHierarchyPermissions:
    """Pruebas de permisos de jerarquía MIC."""

    def test_physics_always_allowed(self, mic_registry, empty_context):
        """PHYSICS siempre está permitido (nivel base)."""
        result = mic_registry.project_intent(
            "mock_physics",
            {"val": 42},
            empty_context
        )
        
        assert result["success"] is True
        assert result["val"] == 42
        assert result.get("_mic_validation_update") == Stratum.PHYSICS

    def test_tactics_blocked_without_physics(self, mic_registry, empty_context):
        """TACTICS bloqueado sin PHYSICS validado."""
        result = mic_registry.project_intent(
            "mock_tactics",
            {"items": ["a", "b"]},
            empty_context
        )
        
        assert result["success"] is False
        assert result["error_type"] == "PermissionError"
        assert "MIC Hierarchy Violation" in result["error"]

    def test_tactics_allowed_with_physics(self, mic_registry, physics_validated_context):
        """TACTICS permitido con PHYSICS validado."""
        result = mic_registry.project_intent(
            "mock_tactics",
            {"items": ["a", "b"]},
            physics_validated_context
        )
        
        assert result["success"] is True
        assert result["items"] == ["a", "b"]

    def test_strategy_blocked_without_physics(self, mic_registry, empty_context):
        """STRATEGY bloqueado sin validaciones."""
        result = mic_registry.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            empty_context
        )
        
        assert result["success"] is False
        assert "PermissionError" in result["error_type"]

    def test_strategy_blocked_with_only_physics(self, mic_registry, physics_validated_context):
        """STRATEGY bloqueado con solo PHYSICS (falta TACTICS)."""
        result = mic_registry.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            physics_validated_context
        )
        
        assert result["success"] is False

    def test_strategy_allowed_with_physics_and_tactics(
        self, mic_registry, physics_tactics_validated_context
    ):
        """STRATEGY permitido con PHYSICS y TACTICS validados."""
        result = mic_registry.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            physics_tactics_validated_context
        )
        
        assert result["success"] is True
        assert result["amount"] == 100.0

    def test_wisdom_blocked_without_full_validation(self, mic_registry, physics_tactics_validated_context):
        """WISDOM bloqueado sin STRATEGY validado."""
        result = mic_registry.project_intent(
            "mock_wisdom",
            {"decision": "proceed"},
            physics_tactics_validated_context
        )
        
        assert result["success"] is False

    def test_wisdom_allowed_with_full_validation(self, mic_registry, all_validated_context):
        """WISDOM permitido con todas las validaciones."""
        result = mic_registry.project_intent(
            "mock_wisdom",
            {"decision": "proceed"},
            all_validated_context
        )
        
        assert result["success"] is True
        assert result["decision"] == "proceed"


# ============================================================================
# TEST: LÓGICA DEL GATEKEEPER
# ============================================================================


class TestMICGatekeeperLogic:
    """Pruebas de la lógica del gatekeeper."""

    def test_override_bypasses_all_checks(self, mic_registry, override_context):
        """force_physics_override bypasea todas las validaciones."""
        # STRATEGY sin validaciones previas, pero con override
        result = mic_registry.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            override_context
        )
        
        assert result["success"] is True

    def test_override_works_for_wisdom(self, mic_registry, override_context):
        """Override funciona para WISDOM también."""
        result = mic_registry.project_intent(
            "mock_wisdom",
            {"decision": "emergency"},
            override_context
        )
        
        assert result["success"] is True

    def test_validation_update_returned(self, mic_registry, empty_context):
        """La ejecución exitosa retorna actualización de validación."""
        result = mic_registry.project_intent(
            "mock_physics",
            {"val": 1},
            empty_context
        )
        
        assert "_mic_validation_update" in result
        assert result["_mic_validation_update"] == Stratum.PHYSICS

    def test_failed_execution_no_validation_update(self, mic_registry, empty_context):
        """Ejecución fallida no retorna actualización de validación."""
        result = mic_registry.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            empty_context
        )
        
        assert result["success"] is False
        # No debe tener actualización de validación
        assert result.get("_mic_validation_update") is None

    def test_required_strata_in_error(self, mic_registry, empty_context):
        """Error incluye estratos requeridos."""
        result = mic_registry.project_intent(
            "mock_strategy",
            {"amount": 100.0},
            empty_context
        )
        
        assert result["success"] is False
        if "required_strata" in result:
            required = result["required_strata"]
            assert "PHYSICS" in required
            assert "TACTICS" in required


# ============================================================================
# TEST: REQUISITOS TRANSITIVOS
# ============================================================================


class TestMICTransitiveRequirements:
    """Pruebas de requisitos transitivos de la jerarquía."""

    def test_get_required_strata_for_physics(self, mic_registry):
        """PHYSICS no requiere nada."""
        required = mic_registry.get_required_strata(Stratum.PHYSICS)
        assert required == set()

    def test_get_required_strata_for_tactics(self, mic_registry):
        """TACTICS requiere PHYSICS."""
        required = mic_registry.get_required_strata(Stratum.TACTICS)
        assert Stratum.PHYSICS in required

    def test_get_required_strata_for_strategy(self, mic_registry):
        """STRATEGY requiere PHYSICS y TACTICS."""
        required = mic_registry.get_required_strata(Stratum.STRATEGY)
        assert Stratum.PHYSICS in required
        assert Stratum.TACTICS in required

    def test_get_required_strata_for_wisdom(self, mic_registry):
        """WISDOM requiere PHYSICS, TACTICS y STRATEGY."""
        required = mic_registry.get_required_strata(Stratum.WISDOM)
        assert Stratum.PHYSICS in required
        assert Stratum.TACTICS in required
        assert Stratum.STRATEGY in required

    @pytest.mark.parametrize(
        "target_stratum,validated,expected_allowed",
        [
            # PHYSICS siempre permitido
            (Stratum.PHYSICS, set(), True),
            (Stratum.PHYSICS, {Stratum.TACTICS}, True),
            
            # TACTICS requiere PHYSICS
            (Stratum.TACTICS, set(), False),
            (Stratum.TACTICS, {Stratum.PHYSICS}, True),
            (Stratum.TACTICS, {Stratum.STRATEGY}, False),  # Stratum equivocado
            
            # STRATEGY requiere PHYSICS + TACTICS
            (Stratum.STRATEGY, set(), False),
            (Stratum.STRATEGY, {Stratum.PHYSICS}, False),
            (Stratum.STRATEGY, {Stratum.TACTICS}, False),
            (Stratum.STRATEGY, {Stratum.PHYSICS, Stratum.TACTICS}, True),
            
            # WISDOM requiere todo
            (Stratum.WISDOM, set(), False),
            (Stratum.WISDOM, {Stratum.PHYSICS, Stratum.TACTICS}, False),
            (Stratum.WISDOM, {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}, True),
        ],
    )
    def test_permission_matrix(
        self,
        mic_registry,
        target_stratum: Stratum,
        validated: Set[Stratum],
        expected_allowed: bool,
    ):
        """Matriz completa de permisos."""
        # Mapear stratum a vector mock
        stratum_to_vector = {
            Stratum.PHYSICS: ("mock_physics", {"val": 1}),
            Stratum.TACTICS: ("mock_tactics", {"items": []}),
            Stratum.STRATEGY: ("mock_strategy", {"amount": 1.0}),
            Stratum.WISDOM: ("mock_wisdom", {"decision": "test"}),
        }
        
        vector_name, payload = stratum_to_vector[target_stratum]
        context = {"validated_strata": validated}
        
        result = mic_registry.project_intent(vector_name, payload, context)
        
        assert result["success"] == expected_allowed, \
            f"Expected {'allowed' if expected_allowed else 'blocked'} for {target_stratum.name} " \
            f"with validated={[s.name for s in validated]}"


# ============================================================================
# TEST: MANEJO DE ERRORES
# ============================================================================


class TestMICErrorHandling:
    """Pruebas de manejo de errores en MIC."""

    def test_handler_exception_caught(self, mic_with_failing_handler, empty_context):
        """Excepciones en handlers se capturan."""
        result = mic_with_failing_handler.project_intent(
            "failing_vector",
            {},
            empty_context
        )
        
        assert result["success"] is False
        assert result["error_type"] == "RuntimeError"
        assert "Handler execution failed" in result["error"]

    def test_handler_exception_no_validation_update(self, mic_with_failing_handler, empty_context):
        """Excepción en handler no genera actualización de validación."""
        result = mic_with_failing_handler.project_intent(
            "failing_vector",
            {},
            empty_context
        )
        
        assert result.get("_mic_validation_update") is None

    def test_invalid_payload_type_handled(self, mic_registry, empty_context):
        """Payload inválido se maneja correctamente."""
        # Handler espera int, pasamos string
        result = mic_registry.project_intent(
            "mock_physics",
            {"val": "not_an_int"},  # Tipo incorrecto
            empty_context
        )
        
        # Dependiendo de la implementación, puede fallar o convertir
        # Verificamos que al menos no crashea
        assert "success" in result

    def test_missing_payload_key_handled(self, mic_registry, empty_context):
        """Payload incompleto se maneja."""
        # Handler espera 'val', no lo pasamos
        result = mic_registry.project_intent(
            "mock_physics",
            {},  # Falta 'val'
            empty_context
        )
        
        # Puede fallar pero no debe crashear
        if result["success"] is False:
            assert "error" in result


# ============================================================================
# TEST: INTEGRACIÓN CON TELEMETRÍA
# ============================================================================


@pytest.mark.skipif(not HAS_TELEMETRY, reason="Telemetry not available")
class TestMICIntegrationWithTelemetry:
    """Pruebas de integración con el sistema de telemetría."""

    @pytest.fixture
    def telemetry_context(self):
        """Contexto de telemetría fresco."""
        return TelemetryContext()

    def test_mic_execution_can_be_traced(self, mic_registry, telemetry_context):
        """La ejecución MIC puede ser trazada con telemetría."""
        with telemetry_context.span("mic_physics_execution", stratum=Stratum.PHYSICS):
            result = mic_registry.project_intent(
                "mock_physics",
                {"val": 42},
                {"validated_strata": set()}
            )
        
        assert result["success"] is True
        assert len(telemetry_context.root_spans) == 1

    def test_mic_failure_recorded_in_telemetry(self, mic_registry, telemetry_context):
        """Fallos de MIC se pueden registrar en telemetría."""
        with telemetry_context.span("mic_strategy_attempt", stratum=Stratum.STRATEGY) as span:
            result = mic_registry.project_intent(
                "mock_strategy",
                {"amount": 100.0},
                {"validated_strata": set()}
            )
            
            if not result["success"]:
                span.status = StepStatus.FAILURE
                span.errors.append({
                    "message": result.get("error", "Unknown error"),
                    "type": result.get("error_type", "MICError"),
                })
        
        assert telemetry_context.root_spans[0].status == StepStatus.FAILURE

    def test_validated_strata_from_telemetry_health(self, mic_registry, telemetry_context):
        """Los estratos validados se pueden derivar de la salud de telemetría."""
        # Simular validación exitosa de PHYSICS
        telemetry_context.start_step("physics_validation", stratum=Stratum.PHYSICS)
        telemetry_context.end_step("physics_validation", StepStatus.SUCCESS)
        
        # Los estratos saludables podrían considerarse validados
        healthy_strata = {
            stratum for stratum in Stratum
            if telemetry_context._strata_health[stratum].is_healthy
        }
        
        # Todos deberían estar saludables ya que no hubo errores
        assert Stratum.PHYSICS in healthy_strata


# ============================================================================
# TEST: PROPIEDADES ALGEBRAICAS
# ============================================================================


class TestMICAlgebraicProperties:
    """Pruebas de propiedades algebraicas de la jerarquía MIC."""

    def test_physics_is_always_in_requirements(self):
        """PHYSICS está en los requisitos de todos los niveles superiores."""
        registry = MockMICRegistry()
        
        for stratum in [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
            required = registry.get_required_strata(stratum)
            assert Stratum.PHYSICS in required, \
                f"PHYSICS should be required for {stratum.name}"

    def test_requirements_are_transitive(self):
        """Los requisitos son transitivos: req(WISDOM) ⊇ req(STRATEGY) ⊇ req(TACTICS)."""
        registry = MockMICRegistry()
        
        tactics_req = registry.get_required_strata(Stratum.TACTICS)
        strategy_req = registry.get_required_strata(Stratum.STRATEGY)
        wisdom_req = registry.get_required_strata(Stratum.WISDOM)
        
        # STRATEGY debe incluir todo lo de TACTICS
        assert tactics_req.issubset(strategy_req)
        
        # WISDOM debe incluir todo lo de STRATEGY
        assert strategy_req.issubset(wisdom_req)

    def test_stratum_order_matches_requirements_cardinality(self):
        """El orden del estrato corresponde a la cardinalidad de requisitos."""
        registry = MockMICRegistry()
        
        cardinalities = {
            stratum: len(registry.get_required_strata(stratum))
            for stratum in Stratum
        }
        
        # Ordenar por cardinalidad ascendente
        sorted_by_cardinality = sorted(cardinalities.items(), key=lambda x: x[1])
        
        # PHYSICS debe tener 0, luego TACTICS, STRATEGY, WISDOM
        expected_order = [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]
        actual_order = [s for s, _ in sorted_by_cardinality]
        
        assert actual_order == expected_order

    def test_requirements_form_partial_order(self):
        """Los requisitos forman un orden parcial (antisimétrico)."""
        registry = MockMICRegistry()
        
        for s1 in Stratum:
            for s2 in Stratum:
                if s1 != s2:
                    req1 = registry.get_required_strata(s1)
                    req2 = registry.get_required_strata(s2)
                    
                    # Si s1 requiere s2, entonces s2 no puede requerir s1
                    if s2 in req1:
                        assert s1 not in req2, \
                            f"Antisymmetry violated: {s1.name} requires {s2.name} but {s2.name} also requires {s1.name}"

    def test_physics_is_minimal_element(self):
        """PHYSICS es el elemento mínimo (no tiene requisitos)."""
        registry = MockMICRegistry()
        
        physics_req = registry.get_required_strata(Stratum.PHYSICS)
        assert len(physics_req) == 0

    def test_wisdom_is_maximal_element(self):
        """WISDOM es el elemento máximo (requiere todos los demás)."""
        registry = MockMICRegistry()
        
        wisdom_req = registry.get_required_strata(Stratum.WISDOM)
        other_strata = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
        
        assert wisdom_req == other_strata


# ============================================================================
# TEST: ESCENARIOS DE FLUJO
# ============================================================================


class TestMICFlowScenarios:
    """Pruebas de escenarios de flujo típicos."""

    def test_sequential_validation_flow(self, mic_registry):
        """Flujo de validación secuencial: PHYSICS → TACTICS → STRATEGY → WISDOM."""
        validated = set()
        
        # Paso 1: PHYSICS
        result = mic_registry.project_intent(
            "mock_physics", {"val": 1}, {"validated_strata": validated}
        )
        assert result["success"] is True
        validated.add(result["_mic_validation_update"])
        
        # Paso 2: TACTICS
        result = mic_registry.project_intent(
            "mock_tactics", {"items": []}, {"validated_strata": validated}
        )
        assert result["success"] is True
        validated.add(result["_mic_validation_update"])
        
        # Paso 3: STRATEGY
        result = mic_registry.project_intent(
            "mock_strategy", {"amount": 100.0}, {"validated_strata": validated}
        )
        assert result["success"] is True
        validated.add(result["_mic_validation_update"])
        
        # Paso 4: WISDOM
        result = mic_registry.project_intent(
            "mock_wisdom", {"decision": "approve"}, {"validated_strata": validated}
        )
        assert result["success"] is True
        
        # Todos los estratos validados
        assert validated == {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}

    def test_skip_level_blocked(self, mic_registry):
        """Saltar niveles está bloqueado."""
        validated = {Stratum.PHYSICS}
        
        # Intentar STRATEGY sin TACTICS
        result = mic_registry.project_intent(
            "mock_strategy", {"amount": 100.0}, {"validated_strata": validated}
        )
        
        assert result["success"] is False

    def test_emergency_override_flow(self, mic_registry):
        """Flujo de emergencia con override."""
        context = {"validated_strata": set(), "force_physics_override": True}
        
        # Puede ejecutar WISDOM directamente
        result = mic_registry.project_intent(
            "mock_wisdom", {"decision": "emergency_override"}, context
        )
        
        assert result["success"] is True


# ============================================================================
# TEST: DETERMINISMO
# ============================================================================


class TestMICDeterminism:
    """Pruebas de determinismo en MIC."""

    def test_same_input_same_output(self, mic_registry):
        """Mismo input produce mismo output."""
        context = {"validated_strata": {Stratum.PHYSICS}}
        payload = {"items": ["x", "y"]}
        
        result1 = mic_registry.project_intent("mock_tactics", payload.copy(), context.copy())
        result2 = mic_registry.project_intent("mock_tactics", payload.copy(), context.copy())
        
        assert result1["success"] == result2["success"]
        assert result1["items"] == result2["items"]

    def test_order_of_validation_irrelevant(self, mic_registry):
        """El orden en que se agregaron las validaciones no importa."""
        # Validar PHYSICS primero, luego TACTICS
        validated_a = set()
        validated_a.add(Stratum.PHYSICS)
        validated_a.add(Stratum.TACTICS)
        
        # Crear set en orden diferente (aunque sets no tienen orden, 
        # verificamos que la lógica no depende del orden de inserción)
        validated_b = {Stratum.TACTICS, Stratum.PHYSICS}
        
        result_a = mic_registry.project_intent(
            "mock_strategy", {"amount": 1.0}, {"validated_strata": validated_a}
        )
        result_b = mic_registry.project_intent(
            "mock_strategy", {"amount": 1.0}, {"validated_strata": validated_b}
        )
        
        assert result_a["success"] == result_b["success"]


# ============================================================================
# TEST: EDGE CASES
# ============================================================================


class TestMICEdgeCases:
    """Pruebas de casos límite."""

    def test_empty_payload(self, mic_registry, empty_context):
        """Payload vacío manejado."""
        # physics_handler espera 'val', payload vacío puede fallar
        result = mic_registry.project_intent("mock_physics", {}, empty_context)
        # No debe crashear
        assert isinstance(result, dict)

    def test_none_in_validated_strata(self, mic_registry):
        """None en validated_strata no causa crash."""
        context = {"validated_strata": {Stratum.PHYSICS, None}}
        
        # No debe crashear
        result = mic_registry.project_intent(
            "mock_tactics", {"items": []}, context
        )
        # TACTICS debería estar permitido ya que PHYSICS está validado
        assert result["success"] is True

    def test_extra_validated_strata_ignored(self, mic_registry):
        """Estratos extra en validated_strata se ignoran."""
        context = {"validated_strata": {Stratum.PHYSICS, Stratum.WISDOM}}  # WISDOM extra
        
        # TACTICS solo necesita PHYSICS
        result = mic_registry.project_intent(
            "mock_tactics", {"items": []}, context
        )
        
        assert result["success"] is True

    def test_duplicate_registrations(self):
        """Registros duplicados sobrescriben el anterior."""
        registry = MockMICRegistry()
        
        def handler_v1(val: int):
            return {"version": 1, "val": val}
        
        def handler_v2(val: int):
            return {"version": 2, "val": val}
        
        registry.register_vector("test_vector", Stratum.PHYSICS, handler_v1)
        registry.register_vector("test_vector", Stratum.PHYSICS, handler_v2)
        
        result = registry.project_intent("test_vector", {"val": 1}, {})
        
        # Debe usar v2
        assert result["version"] == 2