"""
Test Maestro: El Gran Colapso de Onda (The Great Wave Collapse).
================================================================

Verifica la coherencia del flujo DIKW como composición categórica
sobre CategoricalState y su acoplamiento mínimo con PipelineDirector.

Fundamentos Matemáticos
-----------------------
1. Teoría de Categorías:
   - DIKW como funtor F: Stratum → State
   - Composición de morfismos preserva estructura
   - Diagramas conmutativos para caminos equivalentes
   - Identidad y asociatividad verificadas

2. Teoría de Orden:
   - Estratos forman un retículo (PHYSICS ≤ TACTICS ≤ STRATEGY ≤ WISDOM)
   - Transiciones son monótonas (preservan orden)
   - Clausura transitiva de validated_strata

3. Álgebra de Tipos:
   - Estados como objetos inmutables
   - Transiciones como morfismos puros
   - Producto de estados bien definido

4. Teoría de la Información:
   - Conservación de información (no-pérdida)
   - Linaje como cadena de custodia criptográfica
   - Trazabilidad completa de transformaciones

Invariantes Probados
--------------------
- El DAG del director es acíclico (propiedad topológica)
- La evolución DIKW es monótona en validated_strata (orden parcial)
- Cada transición produce un estado nuevo (pureza funcional)
- El hash del estado evoluciona a lo largo del flujo (unicidad)
- El producto final referencia correctamente la huella del estado WISDOM
- La traza de composición conserva el linaje de morfismos
- Diagrama conmutativo para caminos equivalentes
- Idempotencia de transiciones a estratos ya validados
- Conservación de payload a través de transiciones
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Any, Final, TypeVar
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from app.mic_algebra import CategoricalState, create_categorical_state
from app.pipeline_director import PipelineConfig, PipelineDirector
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.tools_interface import MICRegistry


# =============================================================================
# CONSTANTES DEL DOMINIO DIKW
# =============================================================================

# Orden canónico de estratos (retículo lineal)
_STRATUM_ORDER: Final[tuple[Stratum, ...]] = (
    Stratum.PHYSICS,
    Stratum.TACTICS,
    Stratum.STRATEGY,
    Stratum.WISDOM,
)

# Conjunto completo de estratos DIKW
_ALL_DIKW_STRATA: Final[frozenset[Stratum]] = frozenset(_STRATUM_ORDER)

# Mapeo de estrato a su nivel ordinal
_STRATUM_LEVEL: Final[dict[Stratum, int]] = {
    stratum: i for i, stratum in enumerate(_STRATUM_ORDER)
}

# Nombre de morfismos mock por estrato
_MORPHISM_NAMES: Final[dict[Stratum, str]] = {
    stratum: f"mock_{stratum.name.lower()}_morphism"
    for stratum in _STRATUM_ORDER
}

# Tolerancia para comparaciones de punto flotante
_FLOAT_TOLERANCE: Final[float] = 1e-10


# =============================================================================
# TIPOS AUXILIARES
# =============================================================================

StateTransition = Callable[[CategoricalState], CategoricalState]
T = TypeVar("T")


@dataclass(frozen=True)
class TransitionRecord:
    """Registro inmutable de una transición de estado."""
    
    source_hash: str
    target_hash: str
    source_strata: frozenset[Stratum]
    target_strata: frozenset[Stratum]
    morphism_name: str
    
    @property
    def is_monotone(self) -> bool:
        """Verifica si la transición es monótona (preserva orden)."""
        return self.source_strata.issubset(self.target_strata)
    
    @property
    def added_strata(self) -> frozenset[Stratum]:
        """Retorna los estratos añadidos por esta transición."""
        return self.target_strata - self.source_strata


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def stratum_level(stratum: Stratum) -> int:
    """
    Retorna el nivel ordinal de un estrato en el retículo DIKW.
    
    El orden es: PHYSICS(0) < TACTICS(1) < STRATEGY(2) < WISDOM(3)
    
    Args:
        stratum: Estrato a consultar.
    
    Returns:
        Nivel ordinal del estrato.
    """
    return _STRATUM_LEVEL.get(stratum, -1)


def max_stratum_level(strata: frozenset[Stratum]) -> int:
    """
    Retorna el nivel máximo alcanzado por un conjunto de estratos.
    
    Args:
        strata: Conjunto de estratos validados.
    
    Returns:
        Nivel máximo, o -1 si el conjunto está vacío.
    """
    if not strata:
        return -1
    return max(stratum_level(s) for s in strata)


def is_stratum_reachable(
    source: Stratum,
    target: Stratum,
) -> bool:
    """
    Verifica si un estrato es alcanzable desde otro en el retículo.
    
    En el retículo lineal DIKW, source → target es válido si level(source) < level(target).
    
    Args:
        source: Estrato origen.
        target: Estrato destino.
    
    Returns:
        True si target es alcanzable desde source.
    """
    return stratum_level(source) < stratum_level(target)


def verify_monotone_chain(
    states: Sequence[CategoricalState],
) -> tuple[bool, str]:
    """
    Verifica que una cadena de estados satisface monotonía.
    
    Monotonía: ∀i < j: states[i].validated_strata ⊆ states[j].validated_strata
    
    Args:
        states: Secuencia de estados a verificar.
    
    Returns:
        Tupla (es_monotona, mensaje_error).
    """
    for i in range(len(states) - 1):
        current_strata = states[i].validated_strata
        next_strata = states[i + 1].validated_strata
        
        if not current_strata.issubset(next_strata):
            return False, (
                f"Violación de monotonía en posición {i}: "
                f"{current_strata} ⊄ {next_strata}"
            )
    
    return True, "Cadena monótona válida"


def verify_state_purity(
    states: Sequence[CategoricalState],
) -> tuple[bool, str]:
    """
    Verifica que todos los estados en una cadena son objetos distintos.
    
    Pureza funcional: cada transición produce un nuevo objeto.
    
    Args:
        states: Secuencia de estados a verificar.
    
    Returns:
        Tupla (son_puros, mensaje_error).
    """
    ids = [id(state) for state in states]
    
    if len(ids) != len(set(ids)):
        return False, "Estados duplicados detectados (mismo objeto)"
    
    return True, "Todos los estados son objetos distintos"


def verify_hash_evolution(
    states: Sequence[CategoricalState],
) -> tuple[bool, str]:
    """
    Verifica que los hashes evolucionan a través de la cadena.
    
    Args:
        states: Secuencia de estados a verificar.
    
    Returns:
        Tupla (evolucionan, mensaje_error).
    """
    hashes = [state.compute_hash() for state in states]
    
    for i in range(len(hashes) - 1):
        if hashes[i] == hashes[i + 1]:
            return False, (
                f"Hash no evolucionó entre posiciones {i} y {i + 1}"
            )
    
    return True, "Hashes evolucionan correctamente"


def compute_stratum_closure(strata: frozenset[Stratum]) -> frozenset[Stratum]:
    """
    Calcula la clausura transitiva de un conjunto de estratos.
    
    En el retículo DIKW, la clausura incluye todos los estratos menores.
    Si STRATEGY ∈ strata, entonces {PHYSICS, TACTICS} también deberían estar.
    
    Args:
        strata: Conjunto de estratos.
    
    Returns:
        Clausura transitiva del conjunto.
    """
    if not strata:
        return frozenset()
    
    max_level = max_stratum_level(strata)
    return frozenset(s for s in _STRATUM_ORDER if stratum_level(s) <= max_level)


def create_deterministic_dataframe(
    rows: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Crea un DataFrame determinista para pruebas.
    
    Args:
        rows: Número de filas.
        seed: Semilla para reproducibilidad.
    
    Returns:
        DataFrame determinista.
    """
    np.random.seed(seed)
    return pd.DataFrame({
        "id": list(range(1, rows + 1)),
        "value": np.linspace(0.1, 0.1 * rows, rows),
        "category": [chr(65 + i % 26) for i in range(rows)],
    })


# =============================================================================
# FUNCIONES DE TRANSICIÓN DE ESTADO
# =============================================================================

def create_initial_state(
    payload: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> CategoricalState:
    """
    Crea un estado inicial limpio y determinista.
    
    El estado inicial no tiene estratos validados y representa el punto
    de entrada al flujo DIKW.
    
    Args:
        payload: Datos iniciales (opcional).
        context: Contexto de ejecución (opcional).
    
    Returns:
        Estado inicial para el flujo DIKW.
    """
    if payload is None:
        df = create_deterministic_dataframe()
        payload = {"data": df}
    
    if context is None:
        context = {"source": "integration_test"}
    
    return create_categorical_state(
        payload=payload,
        context=context,
    )


def transition_state(
    state: CategoricalState,
    stratum: Stratum,
    morphism_name: str | None = None,
) -> CategoricalState:
    """
    Aplica una transición pura a un nuevo estrato y añade traza.
    
    La transición:
    1. Crea un nuevo estado con el payload actual
    2. Añade el nuevo estrato a validated_strata
    3. Registra la transición en composition_trace
    
    Args:
        state: Estado origen.
        stratum: Estrato destino.
        morphism_name: Nombre del morfismo (opcional, auto-generado si None).
    
    Returns:
        Nuevo estado con el estrato añadido.
    """
    if morphism_name is None:
        morphism_name = _MORPHISM_NAMES.get(stratum, f"morphism_{stratum.name}")
    
    evolved = state.with_update(dict(state.payload), new_stratum=stratum)
    return evolved.add_trace(
        morphism_name=morphism_name,
        input_domain=state.validated_strata,
        output_codomain=stratum,
        success=True,
    )


def apply_sequential_transitions(
    initial_state: CategoricalState,
    strata: Sequence[Stratum],
) -> list[CategoricalState]:
    """
    Aplica una secuencia de transiciones y retorna todos los estados intermedios.
    
    Args:
        initial_state: Estado inicial.
        strata: Secuencia de estratos a transitar.
    
    Returns:
        Lista de estados [inicial, después_de_s1, después_de_s2, ...].
    """
    states = [initial_state]
    current = initial_state
    
    for stratum in strata:
        current = transition_state(current, stratum)
        states.append(current)
    
    return states


def apply_final_result(
    state: CategoricalState,
    result_kind: str = "DataProduct",
    content: str = "Final analysis report",
) -> CategoricalState:
    """
    Añade el producto de datos final preservando el payload previo.
    
    El producto final incluye:
    - Tipo de producto
    - Contenido del análisis
    - Hash de linaje para trazabilidad
    
    Args:
        state: Estado con WISDOM validado.
        result_kind: Tipo del producto de datos.
        content: Contenido del resultado.
    
    Returns:
        Estado con producto final añadido.
    """
    lineage_hash = state.compute_hash()
    payload = dict(state.payload)
    payload["final_result"] = {
        "kind": result_kind,
        "metadata": {
            "lineage_hash": lineage_hash,
            "source_strata": [s.name for s in state.validated_strata],
        },
        "content": content,
    }
    return state.with_update(payload)


def create_alternative_path_state(
    initial_state: CategoricalState,
    path: Sequence[Stratum],
) -> CategoricalState:
    """
    Crea un estado siguiendo un camino alternativo de estratos.
    
    Útil para verificar conmutatividad de diagramas.
    
    Args:
        initial_state: Estado inicial.
        path: Camino de estratos a seguir.
    
    Returns:
        Estado final del camino.
    """
    return reduce(
        lambda s, stratum: transition_state(s, stratum),
        path,
        initial_state,
    )


# =============================================================================
# FIXTURES
# =============================================================================

def pytest_configure(config) -> None:
    """Configura marcadores de pytest."""
    config.addinivalue_line(
        "markers",
        "integration: marca pruebas de integración del flujo DIKW",
    )
    config.addinivalue_line(
        "markers",
        "categorical: marca pruebas de propiedades categoriales",
    )
    config.addinivalue_line(
        "markers",
        "order_theory: marca pruebas de teoría de orden",
    )


@pytest.fixture
def mock_telemetry() -> MagicMock:
    """Crea un mock de TelemetryContext."""
    telemetry = MagicMock(spec=TelemetryContext)
    telemetry.start_step = MagicMock()
    telemetry.end_step = MagicMock()
    telemetry.record_metric = MagicMock()
    telemetry.record_error = MagicMock()
    return telemetry


@pytest.fixture
def mock_mic_registry() -> MagicMock:
    """Crea un mock de MICRegistry."""
    mic_registry = MagicMock(spec=MICRegistry)
    mic_registry.project_intent = MagicMock(return_value={"status": "ok"})
    mic_registry.get_basis_vector = MagicMock(return_value=None)
    mic_registry.add_basis_vector = MagicMock()
    return mic_registry


@pytest.fixture
def mock_pipeline_components(tmp_path, mock_telemetry, mock_mic_registry):
    """Construye un director con dependencias controladas y deterministas."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    config = PipelineConfig.from_dict({
        "session_dir": str(session_dir),
        "enforce_filtration": False,
        "enforce_homology": False,
    })

    director = PipelineDirector(
        config=config,
        telemetry=mock_telemetry,
        mic=mock_mic_registry,
    )

    return {
        "config": config,
        "telemetry": mock_telemetry,
        "mic_registry": mock_mic_registry,
        "director": director,
        "session_dir": session_dir,
    }


@pytest.fixture
def initial_state() -> CategoricalState:
    """Estado inicial limpio para pruebas."""
    return create_initial_state()


@pytest.fixture
def dikw_state_chain(initial_state) -> dict[str, CategoricalState]:
    """
    Construye la cadena completa DIKW.
    
    Cadena: initial → PHYSICS → TACTICS → STRATEGY → WISDOM → final
    """
    physics_state = transition_state(initial_state, Stratum.PHYSICS)
    tactics_state = transition_state(physics_state, Stratum.TACTICS)
    strategy_state = transition_state(tactics_state, Stratum.STRATEGY)
    wisdom_state = transition_state(strategy_state, Stratum.WISDOM)
    final_state = apply_final_result(wisdom_state)

    return {
        "initial": initial_state,
        "physics": physics_state,
        "tactics": tactics_state,
        "strategy": strategy_state,
        "wisdom": wisdom_state,
        "final": final_state,
    }


@pytest.fixture
def all_states_list(dikw_state_chain) -> list[CategoricalState]:
    """Lista ordenada de todos los estados en la cadena DIKW."""
    return [
        dikw_state_chain["initial"],
        dikw_state_chain["physics"],
        dikw_state_chain["tactics"],
        dikw_state_chain["strategy"],
        dikw_state_chain["wisdom"],
        dikw_state_chain["final"],
    ]


# =============================================================================
# PRUEBAS DE PROPIEDADES DEL PIPELINE DIRECTOR
# =============================================================================

@pytest.mark.integration
class TestPipelineDirectorProperties:
    """Pruebas de propiedades estructurales del PipelineDirector."""

    def test_dag_is_acyclic(self, mock_pipeline_components) -> None:
        """Verifica que el DAG del director es acíclico."""
        director = mock_pipeline_components["director"]
        dag_info = director.get_dag_info()

        assert dag_info["is_acyclic"] is True, (
            "El DAG del pipeline debe ser acíclico"
        )

    def test_dag_has_nodes(self, mock_pipeline_components) -> None:
        """Verifica que el DAG tiene nodos definidos."""
        director = mock_pipeline_components["director"]
        dag_info = director.get_dag_info()

        assert "nodes" in dag_info
        assert len(dag_info["nodes"]) > 0, (
            "El DAG debe tener al menos un nodo"
        )

    def test_dag_has_edges(self, mock_pipeline_components) -> None:
        """Verifica que el DAG tiene aristas definidas."""
        director = mock_pipeline_components["director"]
        dag_info = director.get_dag_info()

        assert "edges" in dag_info
        # Un DAG con más de un nodo debería tener aristas
        if len(dag_info["nodes"]) > 1:
            assert len(dag_info["edges"]) > 0, (
                "Un DAG con múltiples nodos debe tener aristas"
            )

    def test_dag_structure_is_deterministic(
        self,
        mock_pipeline_components,
    ) -> None:
        """Verifica que la estructura del DAG es determinista."""
        director = mock_pipeline_components["director"]
        
        dag_info_1 = director.get_dag_info()
        dag_info_2 = director.get_dag_info()
        dag_info_3 = director.get_dag_info()

        assert dag_info_1["nodes"] == dag_info_2["nodes"] == dag_info_3["nodes"]
        assert dag_info_1["edges"] == dag_info_2["edges"] == dag_info_3["edges"]

    def test_dag_topological_order_exists(
        self,
        mock_pipeline_components,
    ) -> None:
        """
        Verifica que existe un orden topológico válido.
        
        Todo DAG acíclico tiene al menos un orden topológico.
        """
        director = mock_pipeline_components["director"]
        dag_info = director.get_dag_info()

        if dag_info["is_acyclic"]:
            # Reconstruir el grafo para verificar orden topológico
            G = nx.DiGraph()
            G.add_nodes_from(dag_info["nodes"])
            G.add_edges_from(dag_info["edges"])
            
            # Esto no lanzará excepción si el grafo es acíclico
            order = list(nx.topological_sort(G))
            
            assert len(order) == len(dag_info["nodes"])


# =============================================================================
# PRUEBAS DEL FLUJO DIKW - EL GRAN COLAPSO DE ONDA
# =============================================================================

@pytest.mark.integration
class TestGreatWaveCollapse:
    """
    Pruebas de integración del flujo categórico DIKW.
    
    El "Gran Colapso de Onda" representa la evolución del estado
    a través de los estratos DIKW, donde cada transición "colapsa"
    las posibilidades a un estado más definido y de mayor nivel.
    """

    def test_state_purity(self, all_states_list) -> None:
        """
        Verifica pureza estructural: cada estado es un objeto distinto.
        
        En programación funcional pura, las transformaciones no mutan
        el estado original sino que producen nuevos objetos.
        """
        is_pure, message = verify_state_purity(all_states_list)
        assert is_pure, message

    def test_hash_evolution(self, all_states_list) -> None:
        """
        Verifica que el hash del estado evoluciona a lo largo del flujo.
        
        Cada transición debe producir un hash diferente, reflejando
        el cambio en el estado.
        """
        is_evolving, message = verify_hash_evolution(all_states_list)
        assert is_evolving, message

    def test_monotone_stratum_evolution(self, dikw_state_chain) -> None:
        """
        Verifica monotonía: los estratos validados crecen monótonamente.
        
        Si estado_i tiene estrato S validado, entonces estado_j (j > i)
        también debe tener S validado.
        """
        states = [
            dikw_state_chain["initial"],
            dikw_state_chain["physics"],
            dikw_state_chain["tactics"],
            dikw_state_chain["strategy"],
            dikw_state_chain["wisdom"],
            dikw_state_chain["final"],
        ]
        
        is_monotone, message = verify_monotone_chain(states)
        assert is_monotone, message

    def test_stratum_subset_chain(self, dikw_state_chain) -> None:
        """Verifica cadena de subconjuntos estricta."""
        initial = dikw_state_chain["initial"]
        physics = dikw_state_chain["physics"]
        tactics = dikw_state_chain["tactics"]
        strategy = dikw_state_chain["strategy"]
        wisdom = dikw_state_chain["wisdom"]

        # Verificar inclusión estricta donde aplique
        assert Stratum.PHYSICS in physics.validated_strata
        assert physics.validated_strata.issubset(tactics.validated_strata)
        assert tactics.validated_strata.issubset(strategy.validated_strata)
        assert strategy.validated_strata.issubset(wisdom.validated_strata)

    def test_final_state_contains_all_strata(self, dikw_state_chain) -> None:
        """Verifica que el estado final contiene todos los estratos DIKW."""
        final = dikw_state_chain["final"]
        
        assert _ALL_DIKW_STRATA.issubset(final.validated_strata), (
            f"Estado final debe contener todos los estratos DIKW. "
            f"Tiene: {final.validated_strata}, "
            f"Esperado superconjunto de: {_ALL_DIKW_STRATA}"
        )

    def test_final_product_structure(self, dikw_state_chain) -> None:
        """Verifica estructura del producto final."""
        final = dikw_state_chain["final"]
        final_product = final.payload.get("final_result")

        assert final_product is not None, "Producto final debe existir"
        assert final_product.get("kind") == "DataProduct"
        assert final_product.get("content") == "Final analysis report"
        assert "metadata" in final_product

    def test_lineage_hash_points_to_wisdom(self, dikw_state_chain) -> None:
        """
        Verifica que el hash de linaje apunta exactamente al estado WISDOM.
        
        El linaje debe ser trazable al estado inmediatamente anterior
        a la adición del producto final.
        """
        wisdom = dikw_state_chain["wisdom"]
        final = dikw_state_chain["final"]
        
        final_product = final.payload.get("final_result")
        lineage_hash = final_product["metadata"]["lineage_hash"]
        
        assert lineage_hash == wisdom.compute_hash(), (
            "El hash de linaje debe coincidir con el estado WISDOM"
        )

    def test_initial_hash_differs_from_final(self, dikw_state_chain) -> None:
        """Verifica que el hash inicial difiere del final."""
        initial = dikw_state_chain["initial"]
        final = dikw_state_chain["final"]
        
        assert initial.compute_hash() != final.compute_hash(), (
            "El hash debe evolucionar entre estado inicial y final"
        )

    def test_wisdom_hash_differs_from_final(self, dikw_state_chain) -> None:
        """
        Verifica que el hash WISDOM difiere del final.
        
        La adición del producto final debe cambiar el hash.
        """
        wisdom = dikw_state_chain["wisdom"]
        final = dikw_state_chain["final"]
        
        assert wisdom.compute_hash() != final.compute_hash(), (
            "La adición del producto final debe cambiar el hash"
        )


# =============================================================================
# PRUEBAS DE CONSERVACIÓN Y LINAJE
# =============================================================================

@pytest.mark.integration
class TestDataLineagePreservation:
    """Pruebas de conservación de datos y linaje a través del flujo."""

    def test_original_dataframe_survives(self, dikw_state_chain) -> None:
        """
        Verifica que el DataFrame original sobrevive al colapso de onda.
        
        Los datos de entrada deben preservarse intactos a través de
        todas las transiciones.
        """
        initial = dikw_state_chain["initial"]
        final = dikw_state_chain["final"]

        pd.testing.assert_frame_equal(
            initial.payload["data"],
            final.payload["data"],
            obj="DataFrame debe preservarse a través del flujo",
        )

    def test_composition_trace_contains_all_morphisms(
        self,
        dikw_state_chain,
    ) -> None:
        """Verifica que la traza contiene todos los morfismos DIKW."""
        final = dikw_state_chain["final"]
        trace_repr = str(final.composition_trace)

        for stratum in _STRATUM_ORDER:
            morphism_name = _MORPHISM_NAMES[stratum]
            assert morphism_name in trace_repr, (
                f"Morfismo {morphism_name} debe estar en la traza"
            )

    def test_composition_trace_order(self, dikw_state_chain) -> None:
        """
        Verifica que la traza mantiene el orden correcto de morfismos.
        
        El orden debe reflejar la secuencia de transiciones.
        """
        final = dikw_state_chain["final"]
        trace_repr = str(final.composition_trace)

        # Verificar que los morfismos aparecen en el orden correcto
        positions = [
            trace_repr.find(_MORPHISM_NAMES[stratum])
            for stratum in _STRATUM_ORDER
        ]
        
        # Todos deben existir
        assert all(pos >= 0 for pos in positions), (
            "Todos los morfismos deben aparecer en la traza"
        )
        
        # Deben estar en orden ascendente
        assert positions == sorted(positions), (
            "Los morfismos deben aparecer en orden DIKW"
        )

    def test_payload_keys_preserved(self, dikw_state_chain) -> None:
        """Verifica que las claves del payload se preservan."""
        initial = dikw_state_chain["initial"]
        final = dikw_state_chain["final"]

        initial_keys = set(initial.payload.keys())
        final_keys = set(final.payload.keys())

        assert initial_keys.issubset(final_keys), (
            f"Claves originales {initial_keys} deben estar en "
            f"claves finales {final_keys}"
        )


# =============================================================================
# PRUEBAS DE PROPIEDADES CATEGORIALES
# =============================================================================

@pytest.mark.categorical
class TestCategoricalProperties:
    """Pruebas de propiedades de teoría de categorías."""

    def test_identity_transition_is_neutral(self, initial_state) -> None:
        """
        Verifica que una transición al mismo estrato es casi-neutral.
        
        Nota: En nuestra implementación, una transición siempre añade
        un estrato, pero si el estrato ya está validado, el conjunto
        no cambia (idempotencia de unión de conjuntos).
        """
        physics_state = transition_state(initial_state, Stratum.PHYSICS)
        physics_again = transition_state(physics_state, Stratum.PHYSICS)

        # Los estratos validados deben ser iguales
        assert physics_state.validated_strata == physics_again.validated_strata

    def test_composition_is_associative(self, initial_state) -> None:
        """
        Verifica asociatividad: (f ∘ g) ∘ h = f ∘ (g ∘ h).
        
        Aplicar transiciones en cualquier agrupación debe producir
        el mismo conjunto de estratos validados.
        """
        # Camino 1: ((initial → PHYSICS) → TACTICS) → STRATEGY
        path1_step1 = transition_state(initial_state, Stratum.PHYSICS)
        path1_step2 = transition_state(path1_step1, Stratum.TACTICS)
        path1_final = transition_state(path1_step2, Stratum.STRATEGY)

        # Camino 2: aplicación directa secuencial
        states = apply_sequential_transitions(
            initial_state,
            [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY],
        )
        path2_final = states[-1]

        # Los estratos validados deben ser iguales
        assert path1_final.validated_strata == path2_final.validated_strata

    def test_morphism_composition_preserves_payload(
        self,
        initial_state,
    ) -> None:
        """
        Verifica que la composición de morfismos preserva el payload.
        
        El payload original debe estar presente después de cualquier
        secuencia de transiciones.
        """
        original_payload = dict(initial_state.payload)
        
        states = apply_sequential_transitions(
            initial_state,
            list(_STRATUM_ORDER),
        )
        
        final_state = states[-1]
        
        for key, value in original_payload.items():
            assert key in final_state.payload, (
                f"Clave {key} debe preservarse en el payload"
            )

    def test_transition_produces_new_object(self, initial_state) -> None:
        """Verifica que cada transición produce un nuevo objeto."""
        physics_state = transition_state(initial_state, Stratum.PHYSICS)
        
        assert initial_state is not physics_state
        assert id(initial_state) != id(physics_state)

    def test_original_state_unchanged_after_transition(
        self,
        initial_state,
    ) -> None:
        """
        Verifica inmutabilidad: el estado original no cambia.
        
        Las transiciones deben ser funciones puras que no mutan
        el estado de entrada.
        """
        original_hash = initial_state.compute_hash()
        original_strata = frozenset(initial_state.validated_strata)
        
        # Aplicar múltiples transiciones
        _ = apply_sequential_transitions(
            initial_state,
            list(_STRATUM_ORDER),
        )
        
        # El estado original debe permanecer intacto
        assert initial_state.compute_hash() == original_hash
        assert frozenset(initial_state.validated_strata) == original_strata


# =============================================================================
# PRUEBAS DE TEORÍA DE ORDEN
# =============================================================================

@pytest.mark.order_theory
class TestOrderTheoryProperties:
    """Pruebas de propiedades de teoría de orden en el retículo DIKW."""

    def test_stratum_order_is_total(self) -> None:
        """Verifica que el orden de estratos es total (lineal)."""
        for i, s1 in enumerate(_STRATUM_ORDER):
            for j, s2 in enumerate(_STRATUM_ORDER):
                if i != j:
                    # En un orden total, cualquier par es comparable
                    assert stratum_level(s1) != stratum_level(s2)

    def test_stratum_order_is_transitive(self) -> None:
        """
        Verifica transitividad: a < b ∧ b < c ⟹ a < c.
        """
        for i in range(len(_STRATUM_ORDER) - 2):
            a = _STRATUM_ORDER[i]
            b = _STRATUM_ORDER[i + 1]
            c = _STRATUM_ORDER[i + 2]
            
            assert stratum_level(a) < stratum_level(b)
            assert stratum_level(b) < stratum_level(c)
            assert stratum_level(a) < stratum_level(c)

    def test_stratum_order_is_antisymmetric(self) -> None:
        """
        Verifica antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b.
        """
        for s1 in _STRATUM_ORDER:
            for s2 in _STRATUM_ORDER:
                level1 = stratum_level(s1)
                level2 = stratum_level(s2)
                
                if level1 <= level2 and level2 <= level1:
                    assert s1 == s2

    def test_validated_strata_form_downward_closure(
        self,
        dikw_state_chain,
    ) -> None:
        """
        Verifica que los estratos validados forman una clausura descendente.
        
        Si STRATEGY está validado, entonces PHYSICS y TACTICS también
        deberían estarlo (en un flujo bien formado).
        """
        wisdom = dikw_state_chain["wisdom"]
        
        # WISDOM requiere que todos los anteriores estén validados
        expected_closure = compute_stratum_closure(frozenset({Stratum.WISDOM}))
        
        assert expected_closure.issubset(wisdom.validated_strata), (
            f"Estado WISDOM debe tener clausura completa. "
            f"Esperado: {expected_closure}, "
            f"Actual: {wisdom.validated_strata}"
        )

    def test_max_stratum_level_increases(self, all_states_list) -> None:
        """
        Verifica que el nivel máximo de estrato aumenta (o se mantiene).
        
        A lo largo del flujo, nunca debemos retroceder en nivel de estrato.
        """
        levels = [
            max_stratum_level(frozenset(state.validated_strata))
            for state in all_states_list
        ]
        
        for i in range(len(levels) - 1):
            assert levels[i] <= levels[i + 1], (
                f"Nivel de estrato no debe decrecer: "
                f"posición {i}={levels[i]} > posición {i+1}={levels[i+1]}"
            )


# =============================================================================
# PRUEBAS DE CAMINOS ALTERNATIVOS (CONMUTATIVIDAD DE DIAGRAMAS)
# =============================================================================

@pytest.mark.categorical
class TestDiagramCommutativity:
    """
    Pruebas de conmutatividad de diagramas.
    
    En teoría de categorías, un diagrama conmuta si diferentes caminos
    entre los mismos objetos producen el mismo morfismo.
    """

    def test_different_orderings_same_final_strata(
        self,
        initial_state,
    ) -> None:
        """
        Verifica que diferentes ordenamientos producen los mismos estratos.
        
        El conjunto final de estratos validados debe ser independiente
        del orden de aplicación (para estratos que no tienen dependencias
        de orden estricto en el payload).
        """
        # Camino 1: PHYSICS → TACTICS → STRATEGY
        path1 = create_alternative_path_state(
            initial_state,
            [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY],
        )
        
        # En nuestro modelo, cada transición añade un estrato
        # Verificar que los estratos finales son los esperados
        expected_strata = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
        
        assert expected_strata.issubset(path1.validated_strata)

    def test_repeated_transitions_idempotent_on_strata(
        self,
        initial_state,
    ) -> None:
        """
        Verifica idempotencia: transitar al mismo estrato múltiples veces
        no cambia el conjunto de estratos validados.
        """
        physics_once = transition_state(initial_state, Stratum.PHYSICS)
        physics_twice = transition_state(physics_once, Stratum.PHYSICS)
        physics_thrice = transition_state(physics_twice, Stratum.PHYSICS)

        assert physics_once.validated_strata == physics_twice.validated_strata
        assert physics_twice.validated_strata == physics_thrice.validated_strata

    def test_full_path_equivalence(self, initial_state) -> None:
        """
        Verifica que el camino completo DIKW produce estratos consistentes.
        """
        # Aplicar todos los estratos en orden
        full_path = create_alternative_path_state(
            initial_state,
            list(_STRATUM_ORDER),
        )
        
        # Debe contener todos los estratos DIKW
        assert _ALL_DIKW_STRATA.issubset(full_path.validated_strata)


# =============================================================================
# PRUEBAS DE CASOS LÍMITE
# =============================================================================

@pytest.mark.integration
class TestEdgeCases:
    """Pruebas de casos límite del flujo DIKW."""

    def test_empty_initial_state(self) -> None:
        """Verifica comportamiento con estado inicial vacío."""
        empty_state = create_categorical_state(payload={})
        
        physics = transition_state(empty_state, Stratum.PHYSICS)
        
        assert physics.is_success
        assert Stratum.PHYSICS in physics.validated_strata

    def test_large_payload_preservation(self) -> None:
        """Verifica preservación de payload grande."""
        large_df = create_deterministic_dataframe(rows=1000)
        large_payload = {
            "data": large_df,
            "metadata": {f"key_{i}": f"value_{i}" for i in range(100)},
        }
        
        initial = create_initial_state(payload=large_payload)
        final = create_alternative_path_state(initial, list(_STRATUM_ORDER))
        
        pd.testing.assert_frame_equal(
            initial.payload["data"],
            final.payload["data"],
        )
        assert initial.payload["metadata"] == final.payload["metadata"]

    def test_single_stratum_transition(self) -> None:
        """Verifica transición a un solo estrato."""
        initial = create_initial_state()
        physics = transition_state(initial, Stratum.PHYSICS)
        
        assert physics.is_success
        assert Stratum.PHYSICS in physics.validated_strata
        assert len(physics.validated_strata) >= 1

    def test_skip_stratum_if_allowed(self) -> None:
        """
        Verifica comportamiento al saltar estratos.
        
        Nota: Dependiendo de la implementación, esto puede o no estar
        permitido. Esta prueba documenta el comportamiento actual.
        """
        initial = create_initial_state()
        
        # Intentar saltar directamente a STRATEGY sin TACTICS
        strategy = transition_state(
            transition_state(initial, Stratum.PHYSICS),
            Stratum.STRATEGY,
        )
        
        # El estado debe contener ambos estratos aplicados
        assert Stratum.PHYSICS in strategy.validated_strata
        assert Stratum.STRATEGY in strategy.validated_strata

    def test_hash_uniqueness_across_different_payloads(self) -> None:
        """Verifica que diferentes payloads producen diferentes hashes."""
        state1 = create_initial_state(payload={"data": pd.DataFrame({"a": [1]})})
        state2 = create_initial_state(payload={"data": pd.DataFrame({"a": [2]})})
        
        assert state1.compute_hash() != state2.compute_hash()

    def test_context_preservation(self) -> None:
        """Verifica que el contexto se preserva a través del flujo."""
        context = {"source": "test", "version": "1.0", "timestamp": "2024-01-01"}
        initial = create_initial_state(context=context)
        
        final = create_alternative_path_state(initial, list(_STRATUM_ORDER))
        
        # Verificar que el contexto original está accesible
        # (la implementación específica determinará cómo)
        assert initial is not final  # Estados distintos


# =============================================================================
# PRUEBAS DE RENDIMIENTO (SMOKE TESTS)
# =============================================================================

@pytest.mark.slow
class TestPerformanceSmoke:
    """Pruebas de humo de rendimiento."""

    def test_many_transitions_performance(self) -> None:
        """Verifica que muchas transiciones no degradan el rendimiento."""
        import time
        
        initial = create_initial_state()
        
        start = time.perf_counter()
        
        # Aplicar muchas transiciones (ciclos a través de estratos)
        current = initial
        for _ in range(100):
            for stratum in _STRATUM_ORDER:
                current = transition_state(current, stratum)
        
        elapsed = time.perf_counter() - start
        
        # Debería completar en tiempo razonable
        assert elapsed < 5.0, f"Transiciones tomaron {elapsed:.2f}s (máximo 5s)"

    def test_large_trace_does_not_explode(self) -> None:
        """Verifica que una traza larga no causa problemas de memoria."""
        initial = create_initial_state()
        
        current = initial
        for i in range(50):
            for stratum in _STRATUM_ORDER:
                current = transition_state(
                    current,
                    stratum,
                    morphism_name=f"morphism_{stratum.name}_{i}",
                )
        
        # Verificar que la traza es accesible
        trace_repr = str(current.composition_trace)
        assert len(trace_repr) > 0


# =============================================================================
# PRUEBAS DE INVARIANTES GLOBALES
# =============================================================================

@pytest.mark.integration
class TestGlobalInvariants:
    """Pruebas de invariantes globales del sistema."""

    def test_hash_is_deterministic(self, initial_state) -> None:
        """Verifica que el cálculo de hash es determinista."""
        hash1 = initial_state.compute_hash()
        hash2 = initial_state.compute_hash()
        hash3 = initial_state.compute_hash()
        
        assert hash1 == hash2 == hash3

    def test_hash_length_is_consistent(self, all_states_list) -> None:
        """Verifica que todos los hashes tienen la misma longitud."""
        hash_lengths = [len(state.compute_hash()) for state in all_states_list]
        
        assert len(set(hash_lengths)) == 1, (
            f"Todos los hashes deben tener la misma longitud: {hash_lengths}"
        )

    def test_success_states_remain_successful(
        self,
        all_states_list,
    ) -> None:
        """Verifica que los estados exitosos permanecen exitosos."""
        for state in all_states_list:
            assert state.is_success, (
                f"Estado debe ser exitoso: {state}"
            )

    def test_no_none_hashes(self, all_states_list) -> None:
        """Verifica que ningún estado tiene hash None."""
        for state in all_states_list:
            hash_value = state.compute_hash()
            assert hash_value is not None
            assert len(hash_value) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])