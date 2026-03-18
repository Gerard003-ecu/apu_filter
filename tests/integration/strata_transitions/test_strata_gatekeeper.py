"""
Integration Tests para el Gatekeeper Categórico de la MIC
=========================================================

Fundamentos Matemáticos
-----------------------
1. Teoría de Orden y Clausura Transitiva:
   - Estratos forman una cadena total: PHYSICS > TACTICS > STRATEGY > WISDOM
   - Clausura transitiva: si S es el objetivo, requiere todos los estratos s < S
   - Prerrequisitos(S) = {s ∈ Stratum : s.value > S.value}

2. Propiedades del Gatekeeper:
   - Seguridad: nunca permite proyección sin clausura completa
   - Vivacidad: siempre permite proyección con clausura completa
   - Determinismo: misma entrada → mismo resultado
   - Pureza: no muta el contexto de entrada

3. Monotonía del Conjunto Faltante:
   - Sea Missing(V, T) = Prerrequisitos(T) - V
   - Si V₁ ⊆ V₂, entonces Missing(V₂, T) ⊆ Missing(V₁, T)
   - Antimonotonía: más validados → menos faltantes

4. Álgebra de Contextos:
   - Contextos forman un semirretículo bajo unión de estratos validados
   - La proyección es una función parcial sobre contextos

Contrato Probado
----------------
- La proyección a un estrato objetivo requiere clausura transitiva
  de todos los estratos inferiores necesarios.
- Un salto inválido debe ser bloqueado antes de ejecutar el handler.
- Al añadir estratos validados, la violación jerárquica solo puede
  disminuir, nunca empeorar.
- El gatekeeper no muta el contexto del llamador.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeVar
from unittest.mock import Mock, call

import pytest

from app.core.schemas import Stratum
from app.adapters.tools_interface import MICRegistry


# =============================================================================
# CONSTANTES DEL DOMINIO MATEMÁTICO
# =============================================================================

# Nombres de servicios para pruebas
_PHYSICS_SERVICE: Final[str] = "physics_service"
_TACTICS_SERVICE: Final[str] = "tactics_service"
_STRATEGY_SERVICE: Final[str] = "strategy_service"
_WISDOM_SERVICE: Final[str] = "wisdom_service"

# Mapeo de estrato a nombre de servicio
_SERVICE_FOR_STRATUM: Final[dict[Stratum, str]] = {
    Stratum.PHYSICS: _PHYSICS_SERVICE,
    Stratum.TACTICS: _TACTICS_SERVICE,
    Stratum.STRATEGY: _STRATEGY_SERVICE,
    Stratum.WISDOM: _WISDOM_SERVICE,
}

# Orden de estratos por valor (de mayor a menor, base a cúspide)
_STRATUM_ORDER: Final[tuple[Stratum, ...]] = (
    Stratum.PHYSICS,    # valor 3
    Stratum.TACTICS,    # valor 2
    Stratum.STRATEGY,   # valor 1
    Stratum.WISDOM,     # valor 0
)

# Prerrequisitos para cada estrato (clausura transitiva)
_PREREQUISITES: Final[dict[Stratum, frozenset[Stratum]]] = {
    Stratum.PHYSICS: frozenset(),
    Stratum.TACTICS: frozenset({Stratum.PHYSICS}),
    Stratum.STRATEGY: frozenset({Stratum.PHYSICS, Stratum.TACTICS}),
    Stratum.WISDOM: frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}),
}

# Categoría de error para violación jerárquica
_HIERARCHY_VIOLATION: Final[str] = "hierarchy_violation"


# =============================================================================
# TIPOS AUXILIARES
# =============================================================================

T = TypeVar("T")
Context = dict[str, set[Stratum]]
ProjectionResult = dict[str, Any]


@dataclass(frozen=True)
class TransitiveClosureSpec:
    """Especificación de clausura transitiva para un estrato."""
    
    target: Stratum
    prerequisites: frozenset[Stratum]
    
    @classmethod
    def for_stratum(cls, stratum: Stratum) -> "TransitiveClosureSpec":
        """Construye la especificación para un estrato dado."""
        return cls(
            target=stratum,
            prerequisites=_PREREQUISITES[stratum],
        )
    
    def missing_from(self, validated: frozenset[Stratum]) -> frozenset[Stratum]:
        """Calcula los prerrequisitos faltantes dado un conjunto validado."""
        return self.prerequisites - validated
    
    def is_satisfied_by(self, validated: frozenset[Stratum]) -> bool:
        """Verifica si la clausura está satisfecha."""
        return self.prerequisites.issubset(validated)


@dataclass(frozen=True)
class GatekeeperInvariant:
    """Invariante verificable del gatekeeper."""
    
    name: str
    description: str
    check: Callable[[ProjectionResult, Context, Stratum], bool]


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def compute_prerequisites(target: Stratum) -> frozenset[Stratum]:
    """
    Calcula los prerrequisitos (clausura transitiva) para un estrato objetivo.
    
    Prerrequisitos(T) = {s ∈ Stratum : s.value > T.value}
    
    Args:
        target: Estrato objetivo.
    
    Returns:
        Conjunto de estratos prerrequisitos.
    """
    target_value = target.value
    return frozenset(s for s in Stratum if s.value > target_value)


def compute_missing(
    validated: frozenset[Stratum],
    target: Stratum,
) -> frozenset[Stratum]:
    """
    Calcula los estratos faltantes para alcanzar un objetivo.
    
    Missing(V, T) = Prerrequisitos(T) - V
    
    Args:
        validated: Estratos ya validados.
        target: Estrato objetivo.
    
    Returns:
        Conjunto de estratos faltantes.
    """
    prerequisites = compute_prerequisites(target)
    return prerequisites - validated


def is_projection_valid(
    validated: frozenset[Stratum],
    target: Stratum,
) -> bool:
    """
    Determina si una proyección es válida.
    
    Una proyección es válida si y solo si la clausura transitiva está satisfecha.
    
    Args:
        validated: Estratos ya validados.
        target: Estrato objetivo.
    
    Returns:
        True si la proyección es válida.
    """
    return compute_missing(validated, target) == frozenset()


def generate_all_context_combinations() -> list[frozenset[Stratum]]:
    """
    Genera todas las 2^4 = 16 combinaciones posibles de contexto.
    
    Returns:
        Lista de todos los posibles conjuntos de estratos validados.
    """
    combinations = [frozenset()]  # Conjunto vacío
    for r in range(1, len(Stratum) + 1):
        for combo in itertools.combinations(Stratum, r):
            combinations.append(frozenset(combo))
    return combinations


def generate_valid_projection_pairs() -> list[tuple[frozenset[Stratum], Stratum]]:
    """
    Genera todos los pares (contexto, objetivo) donde la proyección es válida.
    
    Returns:
        Lista de pares (validated, target) válidos.
    """
    pairs = []
    for validated in generate_all_context_combinations():
        for target in Stratum:
            if is_projection_valid(validated, target):
                pairs.append((validated, target))
    return pairs


def generate_invalid_projection_pairs() -> list[tuple[frozenset[Stratum], Stratum]]:
    """
    Genera todos los pares (contexto, objetivo) donde la proyección es inválida.
    
    Returns:
        Lista de pares (validated, target) inválidos.
    """
    pairs = []
    for validated in generate_all_context_combinations():
        for target in Stratum:
            if not is_projection_valid(validated, target):
                pairs.append((validated, target))
    return pairs


# =============================================================================
# FUNCIONES DE CONSTRUCCIÓN Y EXTRACCIÓN
# =============================================================================

def build_context(*validated_strata: Stratum) -> Context:
    """
    Construye un contexto mínimo y explícito para la proyección.
    
    Args:
        validated_strata: Estratos ya validados.
    
    Returns:
        Contexto con el conjunto de estratos validados.
    """
    return {"validated_strata": set(validated_strata)}


def build_context_from_set(validated: frozenset[Stratum]) -> Context:
    """
    Construye un contexto a partir de un frozenset.
    
    Args:
        validated: Conjunto de estratos validados.
    
    Returns:
        Contexto con el conjunto de estratos validados.
    """
    return {"validated_strata": set(validated)}


def normalize_stratum_name(value: Any) -> str:
    """
    Normaliza enums o strings a nombre de estrato.
    
    Args:
        value: Valor a normalizar.
    
    Returns:
        Nombre del estrato como string.
    """
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def missing_strata_from(result: ProjectionResult) -> set[str]:
    """
    Extrae y normaliza los estratos faltantes del error del gatekeeper.
    
    Args:
        result: Resultado de la proyección.
    
    Returns:
        Conjunto de nombres de estratos faltantes.
    """
    details = result.get("error_details", {})
    raw_missing = details.get("missing_strata", [])
    return {normalize_stratum_name(item) for item in raw_missing}


def missing_strata_as_set(result: ProjectionResult) -> frozenset[Stratum]:
    """
    Extrae los estratos faltantes como frozenset de Stratum.
    
    Args:
        result: Resultado de la proyección.
    
    Returns:
        Conjunto de estratos faltantes.
    """
    names = missing_strata_from(result)
    return frozenset(Stratum[name] for name in names if name)


def target_stratum_from(result: ProjectionResult) -> str:
    """
    Extrae y normaliza el estrato objetivo reportado por el gatekeeper.
    
    Args:
        result: Resultado de la proyección.
    
    Returns:
        Nombre del estrato objetivo.
    """
    details = result.get("error_details", {})
    raw_target = details.get("target_stratum")
    return normalize_stratum_name(raw_target)


def is_success(result: ProjectionResult) -> bool:
    """Verifica si el resultado indica éxito."""
    return result.get("success") is True


def is_hierarchy_violation(result: ProjectionResult) -> bool:
    """Verifica si el resultado indica violación jerárquica."""
    return result.get("error_category") == _HIERARCHY_VIOLATION


def make_mock_handler(result_label: str) -> Mock:
    """
    Crea un handler observable que retorna una respuesta exitosa.
    
    Args:
        result_label: Etiqueta para identificar el resultado.
    
    Returns:
        Mock que retorna un diccionario de éxito.
    """
    return Mock(
        side_effect=lambda **kwargs: {
            "success": True,
            "result": result_label,
            "payload": dict(kwargs),
        }
    )


def make_failing_handler(error_message: str) -> Mock:
    """
    Crea un handler que lanza una excepción.
    
    Args:
        error_message: Mensaje de error.
    
    Returns:
        Mock que lanza RuntimeError.
    """
    return Mock(side_effect=RuntimeError(error_message))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mic() -> MICRegistry:
    """Registro fresco por prueba para evitar contaminación de servicios."""
    return MICRegistry()


@pytest.fixture
def mic_with_all_services(mic: MICRegistry) -> MICRegistry:
    """Registro con servicios para todos los estratos."""
    for stratum in Stratum:
        service_name = _SERVICE_FOR_STRATUM[stratum]
        handler = make_mock_handler(f"{stratum.name.lower()} completed")
        mic.register_vector(
            service_name=service_name,
            stratum=stratum,
            handler=handler,
        )
    return mic


@pytest.fixture
def physics_context() -> Context:
    """Contexto con solo PHYSICS validado."""
    return build_context(Stratum.PHYSICS)


@pytest.fixture
def tactics_context() -> Context:
    """Contexto con PHYSICS y TACTICS validados."""
    return build_context(Stratum.PHYSICS, Stratum.TACTICS)


@pytest.fixture
def strategy_context() -> Context:
    """Contexto con PHYSICS, TACTICS y STRATEGY validados."""
    return build_context(Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY)


@pytest.fixture
def full_context() -> Context:
    """Contexto con todos los estratos validados."""
    return build_context(*Stratum)


@pytest.fixture
def empty_context() -> Context:
    """Contexto sin estratos validados."""
    return build_context()


# =============================================================================
# PRUEBAS DE PROPIEDADES DE CLAUSURA TRANSITIVA
# =============================================================================

class TestTransitiveClosureProperties:
    """Pruebas de propiedades de la clausura transitiva."""

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_prerequisites_match_specification(self, stratum: Stratum) -> None:
        """Verifica que los prerrequisitos calculados coinciden con la especificación."""
        computed = compute_prerequisites(stratum)
        specified = _PREREQUISITES[stratum]
        
        assert computed == specified, (
            f"Para {stratum.name}: "
            f"calculado={[s.name for s in computed]}, "
            f"especificado={[s.name for s in specified]}"
        )

    def test_physics_has_no_prerequisites(self) -> None:
        """PHYSICS no tiene prerrequisitos (es la base)."""
        prereqs = compute_prerequisites(Stratum.PHYSICS)
        assert prereqs == frozenset()

    def test_tactics_requires_only_physics(self) -> None:
        """TACTICS requiere solo PHYSICS."""
        prereqs = compute_prerequisites(Stratum.TACTICS)
        assert prereqs == frozenset({Stratum.PHYSICS})

    def test_strategy_requires_physics_and_tactics(self) -> None:
        """STRATEGY requiere PHYSICS y TACTICS."""
        prereqs = compute_prerequisites(Stratum.STRATEGY)
        assert prereqs == frozenset({Stratum.PHYSICS, Stratum.TACTICS})

    def test_wisdom_requires_all_lower_strata(self) -> None:
        """WISDOM requiere todos los estratos inferiores."""
        prereqs = compute_prerequisites(Stratum.WISDOM)
        expected = frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY})
        assert prereqs == expected

    def test_prerequisites_are_transitive(self) -> None:
        """
        Verifica transitividad: si A requiere B y B requiere C, entonces A requiere C.
        """
        for stratum in Stratum:
            prereqs = compute_prerequisites(stratum)
            for prereq in prereqs:
                prereq_prereqs = compute_prerequisites(prereq)
                assert prereq_prereqs.issubset(prereqs), (
                    f"{stratum.name} requiere {prereq.name}, "
                    f"pero no todos sus prerrequisitos: {prereq_prereqs - prereqs}"
                )

    def test_prerequisites_have_higher_values(self) -> None:
        """Verifica que todos los prerrequisitos tienen valores mayores."""
        for stratum in Stratum:
            prereqs = compute_prerequisites(stratum)
            for prereq in prereqs:
                assert prereq.value > stratum.value, (
                    f"Prerrequisito {prereq.name} debe tener valor mayor que "
                    f"{stratum.name}: {prereq.value} > {stratum.value}"
                )

    def test_prerequisites_form_downward_closure(self) -> None:
        """
        Verifica que los prerrequisitos forman una clausura descendente.
        
        Si S requiere T, entonces S requiere todos los que T requiere.
        """
        for stratum in Stratum:
            prereqs = compute_prerequisites(stratum)
            if prereqs:
                # El prerrequisito mínimo (máximo valor) debe tener todos los demás
                max_prereq = max(prereqs, key=lambda s: s.value)
                max_prereqs = compute_prerequisites(max_prereq)
                
                # Todos los prerrequisitos del mínimo deben estar incluidos
                assert max_prereqs.issubset(prereqs)


# =============================================================================
# PRUEBAS DE MONOTONÍA DEL CONJUNTO FALTANTE
# =============================================================================

class TestMissingStrataMonotonicity:
    """Pruebas de antimonotonía del conjunto de estratos faltantes."""

    def test_missing_decreases_with_more_validated(self) -> None:
        """
        Verifica antimonotonía: más validados → menos faltantes.
        
        Si V₁ ⊆ V₂, entonces Missing(V₂, T) ⊆ Missing(V₁, T)
        """
        target = Stratum.STRATEGY
        
        v1 = frozenset()
        v2 = frozenset({Stratum.PHYSICS})
        v3 = frozenset({Stratum.PHYSICS, Stratum.TACTICS})
        
        m1 = compute_missing(v1, target)
        m2 = compute_missing(v2, target)
        m3 = compute_missing(v3, target)
        
        # Verificar contención
        assert m3.issubset(m2), f"Missing debe decrecer: {m3} ⊄ {m2}"
        assert m2.issubset(m1), f"Missing debe decrecer: {m2} ⊄ {m1}"

    @pytest.mark.parametrize("target", list(Stratum))
    def test_antimonotonicity_for_all_targets(self, target: Stratum) -> None:
        """Verifica antimonotonía para todos los estratos objetivo."""
        all_contexts = generate_all_context_combinations()
        
        for v1 in all_contexts:
            for v2 in all_contexts:
                if v1.issubset(v2):
                    m1 = compute_missing(v1, target)
                    m2 = compute_missing(v2, target)
                    
                    assert m2.issubset(m1), (
                        f"Para target={target.name}, "
                        f"V1={[s.name for s in v1]}, V2={[s.name for s in v2]}: "
                        f"Missing(V2)={[s.name for s in m2]} ⊄ Missing(V1)={[s.name for s in m1]}"
                    )

    def test_empty_context_has_all_prerequisites_missing(self) -> None:
        """Contexto vacío tiene todos los prerrequisitos como faltantes."""
        empty = frozenset()
        
        for stratum in Stratum:
            missing = compute_missing(empty, stratum)
            prereqs = compute_prerequisites(stratum)
            
            assert missing == prereqs, (
                f"Para {stratum.name}: missing={[s.name for s in missing]}, "
                f"prereqs={[s.name for s in prereqs]}"
            )

    def test_full_context_has_no_missing(self) -> None:
        """Contexto completo no tiene faltantes."""
        full = frozenset(Stratum)
        
        for stratum in Stratum:
            missing = compute_missing(full, stratum)
            assert missing == frozenset(), (
                f"Para {stratum.name}: should have no missing, got {missing}"
            )


# =============================================================================
# PRUEBAS BÁSICAS DEL GATEKEEPER
# =============================================================================

class TestStrataGatekeeperBasic:
    """Pruebas básicas del gatekeeper de estratos."""

    def test_physics_to_strategy_transitive_closure_violation(
        self,
        mic: MICRegistry,
    ) -> None:
        """
        PHYSICS por sí solo no habilita STRATEGY.
        STRATEGY requiere clausura transitiva: TACTICS y PHYSICS.
        """
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_STRATEGY_SERVICE,
            payload={"some_arg": 1},
            context=build_context(Stratum.PHYSICS),
            use_cache=False,
        )

        assert not is_success(result)
        assert is_hierarchy_violation(result)
        assert target_stratum_from(result) == "STRATEGY"

        missing = missing_strata_from(result)
        assert "TACTICS" in missing
        assert "PHYSICS" not in missing

        handler.assert_not_called()

    def test_empty_context_reports_all_missing_for_strategy(
        self,
        mic: MICRegistry,
    ) -> None:
        """
        Sin estratos validados, STRATEGY debe reportar todos los prerrequisitos.
        """
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_STRATEGY_SERVICE,
            payload={"some_arg": 1},
            context=build_context(),
            use_cache=False,
        )

        assert not is_success(result)
        assert is_hierarchy_violation(result)
        assert target_stratum_from(result) == "STRATEGY"

        missing = missing_strata_from(result)
        assert "TACTICS" in missing
        assert "PHYSICS" in missing

        handler.assert_not_called()

    def test_strategy_succeeds_with_complete_closure(
        self,
        mic: MICRegistry,
    ) -> None:
        """
        STRATEGY debe ejecutarse cuando TACTICS y PHYSICS están validados.
        """
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_STRATEGY_SERVICE,
            payload={"some_arg": 1},
            context=build_context(Stratum.PHYSICS, Stratum.TACTICS),
            use_cache=False,
        )

        assert is_success(result)
        assert result.get("result") == "strategy completed"
        handler.assert_called_once()

    def test_tactics_requires_physics(
        self,
        mic: MICRegistry,
    ) -> None:
        """TACTICS no puede ejecutarse sin PHYSICS validado."""
        handler = make_mock_handler("tactics completed")
        mic.register_vector(
            service_name=_TACTICS_SERVICE,
            stratum=Stratum.TACTICS,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_TACTICS_SERVICE,
            payload={"x": 1},
            context=build_context(),
            use_cache=False,
        )

        assert not is_success(result)
        assert is_hierarchy_violation(result)
        assert target_stratum_from(result) == "TACTICS"
        assert missing_strata_from(result) == {"PHYSICS"}

        handler.assert_not_called()

    def test_wisdom_requires_complete_chain(
        self,
        mic: MICRegistry,
    ) -> None:
        """WISDOM exige la cadena completa inferior."""
        handler = make_mock_handler("wisdom completed")
        mic.register_vector(
            service_name=_WISDOM_SERVICE,
            stratum=Stratum.WISDOM,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_WISDOM_SERVICE,
            payload={"summary": True},
            context=build_context(Stratum.PHYSICS, Stratum.TACTICS),
            use_cache=False,
        )

        assert not is_success(result)
        assert is_hierarchy_violation(result)
        assert target_stratum_from(result) == "WISDOM"

        missing = missing_strata_from(result)
        assert "STRATEGY" in missing
        assert "TACTICS" not in missing
        assert "PHYSICS" not in missing

        handler.assert_not_called()


# =============================================================================
# PRUEBAS DE MONOTONÍA DEL GATEKEEPER
# =============================================================================

class TestGatekeeperMonotonicity:
    """Pruebas de monotonía del comportamiento del gatekeeper."""

    @pytest.mark.parametrize(
        ("validated", "expected_missing"),
        [
            (tuple(), {"PHYSICS", "TACTICS"}),
            ((Stratum.PHYSICS,), {"TACTICS"}),
        ],
    )
    def test_missing_shrinks_monotonically_for_strategy(
        self,
        mic: MICRegistry,
        validated: tuple[Stratum, ...],
        expected_missing: set[str],
    ) -> None:
        """Al aumentar la información validada, el conjunto de faltantes se contrae."""
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_STRATEGY_SERVICE,
            payload={"some_arg": 1},
            context=build_context(*validated),
            use_cache=False,
        )

        assert not is_success(result)
        assert missing_strata_from(result) == expected_missing
        handler.assert_not_called()

    @pytest.mark.parametrize("target", [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM])
    def test_missing_shrinks_as_validated_grows(
        self,
        mic: MICRegistry,
        target: Stratum,
    ) -> None:
        """Verifica que el conjunto faltante decrece monótonamente."""
        handler = make_mock_handler(f"{target.name.lower()} completed")
        mic.register_vector(
            service_name=_SERVICE_FOR_STRATUM[target],
            stratum=target,
            handler=handler,
        )
        
        # Construir secuencia creciente de contextos
        contexts = [
            frozenset(),
            frozenset({Stratum.PHYSICS}),
            frozenset({Stratum.PHYSICS, Stratum.TACTICS}),
            frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}),
        ]
        
        previous_missing: set[str] | None = None
        
        for validated in contexts:
            result = mic.project_intent(
                service_name=_SERVICE_FOR_STRATUM[target],
                payload={"x": 1},
                context=build_context_from_set(validated),
                use_cache=False,
            )
            
            if is_success(result):
                current_missing: set[str] = set()
            else:
                current_missing = missing_strata_from(result)
            
            if previous_missing is not None:
                assert current_missing.issubset(previous_missing), (
                    f"Para target={target.name}, validated={[s.name for s in validated]}: "
                    f"missing={current_missing} ⊄ previous={previous_missing}"
                )
            
            previous_missing = current_missing


# =============================================================================
# PRUEBAS DE TRANSICIONES VÁLIDAS
# =============================================================================

class TestValidTransitions:
    """Pruebas de transiciones válidas entre estratos."""

    def test_physics_projection_always_succeeds(
        self,
        mic: MICRegistry,
    ) -> None:
        """PHYSICS no tiene prerrequisitos, siempre debe tener éxito."""
        handler = make_mock_handler("physics completed")
        mic.register_vector(
            service_name=_PHYSICS_SERVICE,
            stratum=Stratum.PHYSICS,
            handler=handler,
        )

        # Incluso con contexto vacío
        result = mic.project_intent(
            service_name=_PHYSICS_SERVICE,
            payload={"data": [1, 2, 3]},
            context=build_context(),
            use_cache=False,
        )

        assert is_success(result)
        handler.assert_called_once()

    def test_tactics_succeeds_after_physics(
        self,
        mic: MICRegistry,
    ) -> None:
        """TACTICS debe tener éxito después de PHYSICS."""
        handler = make_mock_handler("tactics completed")
        mic.register_vector(
            service_name=_TACTICS_SERVICE,
            stratum=Stratum.TACTICS,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_TACTICS_SERVICE,
            payload={"x": 42},
            context=build_context(Stratum.PHYSICS),
            use_cache=False,
        )

        assert is_success(result)
        assert result.get("result") == "tactics completed"
        handler.assert_called_once()

    def test_complete_chain_allows_wisdom(
        self,
        mic: MICRegistry,
    ) -> None:
        """La cadena completa permite proyección a WISDOM."""
        handler = make_mock_handler("wisdom completed")
        mic.register_vector(
            service_name=_WISDOM_SERVICE,
            stratum=Stratum.WISDOM,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_WISDOM_SERVICE,
            payload={"final": True},
            context=build_context(Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY),
            use_cache=False,
        )

        assert is_success(result)
        assert result.get("result") == "wisdom completed"
        handler.assert_called_once()


# =============================================================================
# PRUEBAS EXHAUSTIVAS
# =============================================================================

class TestExhaustiveProjections:
    """Pruebas exhaustivas de todas las combinaciones contexto-objetivo."""

    @pytest.mark.parametrize(
        "validated,target",
        generate_valid_projection_pairs(),
        ids=lambda x: (
            f"{[s.name for s in x[0]]}→{x[1].name}"
            if isinstance(x, tuple) else str(x)
        ),
    )
    def test_valid_projections_succeed(
        self,
        mic: MICRegistry,
        validated: frozenset[Stratum],
        target: Stratum,
    ) -> None:
        """Todas las proyecciones válidas deben tener éxito."""
        handler = make_mock_handler(f"{target.name.lower()} completed")
        mic.register_vector(
            service_name=_SERVICE_FOR_STRATUM[target],
            stratum=target,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_SERVICE_FOR_STRATUM[target],
            payload={"test": True},
            context=build_context_from_set(validated),
            use_cache=False,
        )

        assert is_success(result), (
            f"Proyección válida falló: validated={[s.name for s in validated]}, "
            f"target={target.name}, result={result}"
        )
        handler.assert_called_once()

    @pytest.mark.parametrize(
        "validated,target",
        generate_invalid_projection_pairs(),
        ids=lambda x: (
            f"{[s.name for s in x[0]]}→{x[1].name}"
            if isinstance(x, tuple) else str(x)
        ),
    )
    def test_invalid_projections_fail(
        self,
        mic: MICRegistry,
        validated: frozenset[Stratum],
        target: Stratum,
    ) -> None:
        """Todas las proyecciones inválidas deben fallar."""
        handler = make_mock_handler(f"{target.name.lower()} completed")
        mic.register_vector(
            service_name=_SERVICE_FOR_STRATUM[target],
            stratum=target,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_SERVICE_FOR_STRATUM[target],
            payload={"test": True},
            context=build_context_from_set(validated),
            use_cache=False,
        )

        assert not is_success(result), (
            f"Proyección inválida tuvo éxito: validated={[s.name for s in validated]}, "
            f"target={target.name}"
        )
        assert is_hierarchy_violation(result)
        handler.assert_not_called()

    @pytest.mark.parametrize(
        "validated,target",
        generate_invalid_projection_pairs(),
    )
    def test_missing_strata_match_computation(
        self,
        mic: MICRegistry,
        validated: frozenset[Stratum],
        target: Stratum,
    ) -> None:
        """Los estratos faltantes reportados coinciden con el cálculo."""
        handler = make_mock_handler(f"{target.name.lower()} completed")
        mic.register_vector(
            service_name=_SERVICE_FOR_STRATUM[target],
            stratum=target,
            handler=handler,
        )

        result = mic.project_intent(
            service_name=_SERVICE_FOR_STRATUM[target],
            payload={"test": True},
            context=build_context_from_set(validated),
            use_cache=False,
        )

        expected_missing = compute_missing(validated, target)
        actual_missing = missing_strata_as_set(result)

        assert actual_missing == expected_missing, (
            f"Para validated={[s.name for s in validated]}, target={target.name}: "
            f"expected={[s.name for s in expected_missing]}, "
            f"actual={[s.name for s in actual_missing]}"
        )


# =============================================================================
# PRUEBAS DE PROPIEDADES DE SEGURIDAD
# =============================================================================

class TestGatekeeperSafetyProperties:
    """Pruebas de propiedades de seguridad del gatekeeper."""

    def test_handler_never_called_on_invalid_projection(
        self,
        mic: MICRegistry,
    ) -> None:
        """El handler nunca se invoca si la proyección es inválida."""
        handler = make_mock_handler("should not be called")
        mic.register_vector(
            service_name=_WISDOM_SERVICE,
            stratum=Stratum.WISDOM,
            handler=handler,
        )

        # Probar con varios contextos insuficientes
        insufficient_contexts = [
            frozenset(),
            frozenset({Stratum.PHYSICS}),
            frozenset({Stratum.PHYSICS, Stratum.TACTICS}),
            frozenset({Stratum.TACTICS, Stratum.STRATEGY}),  # Falta PHYSICS
        ]

        for validated in insufficient_contexts:
            result = mic.project_intent(
                service_name=_WISDOM_SERVICE,
                payload={"test": True},
                context=build_context_from_set(validated),
                use_cache=False,
            )
            
            assert not is_success(result)

        # El handler nunca debe haberse llamado
        handler.assert_not_called()

    def test_context_not_mutated_on_failure(
        self,
        mic: MICRegistry,
    ) -> None:
        """Un veto jerárquico no debe mutar el contexto del caller."""
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        context = build_context(Stratum.PHYSICS)
        original = set(context["validated_strata"])

        result = mic.project_intent(
            service_name=_STRATEGY_SERVICE,
            payload={"some_arg": 1},
            context=context,
            use_cache=False,
        )

        assert not is_success(result)
        assert context["validated_strata"] == original
        handler.assert_not_called()

    def test_context_not_mutated_on_success(
        self,
        mic: MICRegistry,
    ) -> None:
        """Una proyección exitosa tampoco debe mutar el contexto del caller."""
        handler = make_mock_handler("tactics completed")
        mic.register_vector(
            service_name=_TACTICS_SERVICE,
            stratum=Stratum.TACTICS,
            handler=handler,
        )

        context = build_context(Stratum.PHYSICS)
        original = set(context["validated_strata"])

        result = mic.project_intent(
            service_name=_TACTICS_SERVICE,
            payload={"x": 1},
            context=context,
            use_cache=False,
        )

        assert is_success(result)
        assert context["validated_strata"] == original


# =============================================================================
# PRUEBAS DE DETERMINISMO E IDEMPOTENCIA
# =============================================================================

class TestGatekeeperDeterminism:
    """Pruebas de determinismo del gatekeeper."""

    def test_same_input_same_output(
        self,
        mic: MICRegistry,
    ) -> None:
        """La misma entrada produce la misma salida."""
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        context = build_context(Stratum.PHYSICS)
        payload = {"arg": 1}

        results = [
            mic.project_intent(
                service_name=_STRATEGY_SERVICE,
                payload=payload,
                context=build_context(Stratum.PHYSICS),  # Contexto fresco
                use_cache=False,
            )
            for _ in range(5)
        ]

        # Todos los resultados deben ser iguales en estructura
        for result in results:
            assert not is_success(result)
            assert is_hierarchy_violation(result)
            assert missing_strata_from(result) == {"TACTICS"}

    def test_projection_is_idempotent_in_effect(
        self,
        mic: MICRegistry,
    ) -> None:
        """Proyecciones repetidas con el mismo contexto tienen el mismo efecto."""
        handler = make_mock_handler("tactics completed")
        mic.register_vector(
            service_name=_TACTICS_SERVICE,
            stratum=Stratum.TACTICS,
            handler=handler,
        )

        # Múltiples proyecciones exitosas
        for i in range(3):
            result = mic.project_intent(
                service_name=_TACTICS_SERVICE,
                payload={"iteration": i},
                context=build_context(Stratum.PHYSICS),
                use_cache=False,
            )
            assert is_success(result)

        # El handler se llamó 3 veces
        assert handler.call_count == 3


# =============================================================================
# PRUEBAS DE COMPOSICIÓN DE PROYECCIONES
# =============================================================================

class TestProjectionComposition:
    """Pruebas de composición de proyecciones secuenciales."""

    def test_sequential_valid_projections(
        self,
        mic_with_all_services: MICRegistry,
    ) -> None:
        """Secuencia de proyecciones válidas tiene éxito."""
        mic = mic_with_all_services
        validated: set[Stratum] = set()

        # Construir la cadena de abajo hacia arriba
        for stratum in _STRATUM_ORDER:
            context = build_context(*validated)
            
            result = mic.project_intent(
                service_name=_SERVICE_FOR_STRATUM[stratum],
                payload={"step": stratum.name},
                context=context,
                use_cache=False,
            )
            
            assert is_success(result), (
                f"Proyección a {stratum.name} falló con contexto "
                f"{[s.name for s in validated]}"
            )
            
            # Simular validación del estrato
            validated.add(stratum)

    def test_skipping_stratum_always_fails(
        self,
        mic: MICRegistry,
    ) -> None:
        """Saltar un estrato siempre falla."""
        handler = make_mock_handler("strategy completed")
        mic.register_vector(
            service_name=_STRATEGY_SERVICE,
            stratum=Stratum.STRATEGY,
            handler=handler,
        )

        # Intentar saltar de vacío a STRATEGY
        result = mic.project_intent(
            service_name=_STRATEGY_SERVICE,
            payload={"skip_attempt": True},
            context=build_context(),
            use_cache=False,
        )

        assert not is_success(result)
        handler.assert_not_called()


# =============================================================================
# PRUEBAS DE INVARIANTES GLOBALES
# =============================================================================

class TestGlobalInvariants:
    """Pruebas de invariantes globales del sistema."""

    def test_prerequisites_are_subset_of_all_strata(self) -> None:
        """Los prerrequisitos siempre son subconjunto del espacio de estratos."""
        all_strata = frozenset(Stratum)
        
        for stratum in Stratum:
            prereqs = compute_prerequisites(stratum)
            assert prereqs.issubset(all_strata)

    def test_no_stratum_is_its_own_prerequisite(self) -> None:
        """Ningún estrato es su propio prerrequisito."""
        for stratum in Stratum:
            prereqs = compute_prerequisites(stratum)
            assert stratum not in prereqs

    def test_prerequisite_chain_is_acyclic(self) -> None:
        """La cadena de prerrequisitos es acíclica."""
        for stratum in Stratum:
            # Only trace paths to verify it is acyclic, tracking the full path to detect cycles
            visited_paths = set()
            to_check = [(stratum,)]
            
            while to_check:
                current_path = to_check.pop()
                current_node = current_path[-1]

                prereqs = compute_prerequisites(current_node)
                for prereq in prereqs:
                    if prereq in current_path:
                        pytest.fail(f"Ciclo detectado involucrando {prereq.name} en el camino {current_path}")
                    to_check.append(current_path + (prereq,))

    def test_total_valid_projections_count(self) -> None:
        """Verifica el conteo de proyecciones válidas."""
        valid_pairs = generate_valid_projection_pairs()
        invalid_pairs = generate_invalid_projection_pairs()
        
        total_pairs = 16 * 4  # 2^4 contextos × 4 objetivos
        
        assert len(valid_pairs) + len(invalid_pairs) == total_pairs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])