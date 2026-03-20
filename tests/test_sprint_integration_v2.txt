"""
Suite de integración: Extracción topológica de anomalías (Sprint Integration).

Fundamentación matemática:
──────────────────────────

1. Ciclos en grafos dirigidos y homología simplicial:
   Sea G = (V, E) un grafo dirigido. Un ciclo c es una secuencia
   (v₀, v₁, ..., vₖ) donde:
       - (vᵢ, vᵢ₊₁) ∈ E  para i ∈ {0, ..., k-1}
       - v₀ = vₖ  (condición de cierre)
       - k ≥ 1    (al menos una arista)

   En el contexto del presupuesto, un ciclo representa una dependencia
   circular ("socavón lógico") — un elemento no trivial del primer
   grupo de homología H₁(G; ℤ).

   Referencia: [1] Hatcher, A. "Algebraic Topology", Cambridge, 2002.

2. Operador frontera y cierre topológico:
   Para una 1-cadena c = Σᵢ (vᵢ, vᵢ₊₁), el operador frontera es:

       ∂₁(c) = Σᵢ [vᵢ₊₁ - vᵢ]

   Un ciclo cerrado satisface ∂₁(c) = 0 (la suma telescópica cancela
   cuando v₀ = vₖ). Esto es equivalente a ker(∂₁) ⊃ im(∂₂) en la
   secuencia exacta de homología.

3. Extracción como funtor:
   La función extract_anomaly_data actúa como un funtor:

       F: CycleSpace → AnomalyData

   que mapea el espacio de ciclos (representados como strings) al
   espacio discreto de conjuntos de nodos y aristas, preservando:
   - Unicidad (I₂): F mapea a conjuntos, eliminando duplicados
   - Determinismo: F(x) = F(x) para todo x (función pura)
   - Inmutabilidad (I₅): F no modifica su argumento

4. Idempotencia del funtor de extracción:
   Una función f es idempotente si f(f(x)) = f(x) para todo x.
   En nuestro caso, como F: Dict → AnomalyData y el dominio/codominio
   difieren, la idempotencia se interpreta como:
       F(x₁) = F(x₂) cuando x₁ ≡ x₂ (inputs equivalentes)

Invariantes del sistema verificados:
    (I₁) ∀ entrada: type(F(entrada)) == AnomalyData (nunca None, nunca excepción)
    (I₂) ∀ resultado: nodes_in_cycles es conjunto (sin duplicados)
    (I₃) Si cycles = ∅ → nodes_in_cycles = ∅
    (I₄) ∀ cadena "A → B → C": {A, B, C} ⊆ nodes_in_cycles
    (I₅) Input original permanece inmutable tras F(input)
    (I₆) ∂₁(c) = 0 para todo ciclo cerrado c

Referencias:
    [1] Hatcher, A. "Algebraic Topology", Cambridge University Press, 2002.
    [2] Diestel, R. "Graph Theory", 5th Ed., Springer, 2017.
"""

from __future__ import annotations

import re
import pytest
from typing import Any, Dict, List, Optional, Set, Tuple
from copy import deepcopy

from app.adapters.topology_viz import extract_anomaly_data, AnomalyData


# =============================================================================
# CONSTANTES
# =============================================================================

# Patrón regex para normalización de separadores de ciclo.
# Captura variantes: ->, →, =>, ⟶ con whitespace arbitrario.
_SEPARATOR_PATTERN: re.Pattern = re.compile(r'\s*(?:->|→|=>|⟶)\s*')

# Separador canónico tras normalización.
_CANONICAL_SEPARATOR: str = " -> "

# Límite de tiempo para tests de rendimiento [segundos].
# Conservador para CI en hardware heterogéneo.
_PERFORMANCE_TIME_LIMIT: float = 2.0


# =============================================================================
# FUNCIONES AUXILIARES DE PARSING Y VALIDACIÓN
# =============================================================================


def _parse_cycle_string(cycle_str: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parsea una cadena de ciclo y extrae nodos y aristas dirigidas.

    La normalización procede en dos pasos:
    1. Reemplazar todos los separadores reconocidos por el canónico " -> "
    2. Dividir por el separador canónico y limpiar whitespace

    Parameters
    ----------
    cycle_str : str
        Cadena en formato "A -> B -> C -> A" (o variantes de separador).

    Returns
    -------
    Tuple[List[str], List[Tuple[str, str]]]
        - nodes: Lista de nodos en orden de aparición (incluye repetición
          del nodo inicial si el ciclo es cerrado).
        - edges: Lista de aristas dirigidas (source, target).

    Examples
    --------
    >>> _parse_cycle_string("A -> B -> C -> A")
    (['A', 'B', 'C', 'A'], [('A', 'B'), ('B', 'C'), ('C', 'A')])

    >>> _parse_cycle_string("")
    ([], [])
    """
    if not cycle_str or not cycle_str.strip():
        return [], []

    normalized: str = _SEPARATOR_PATTERN.sub(_CANONICAL_SEPARATOR, cycle_str)
    nodes: List[str] = [n.strip() for n in normalized.split(_CANONICAL_SEPARATOR) if n.strip()]
    edges: List[Tuple[str, str]] = [
        (nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)
    ]

    return nodes, edges


def _validate_cycle_closure(cycle_str: str) -> Dict[str, Any]:
    """
    Valida si una cadena de ciclo es topológicamente cerrada.

    Un ciclo cerrado satisface v₀ = vₖ, lo cual implica ∂₁(c) = 0
    (la suma telescópica del operador frontera cancela).

    Parameters
    ----------
    cycle_str : str
        Cadena de ciclo a validar.

    Returns
    -------
    Dict[str, Any]
        - "is_closed": bool — True si v₀ = vₖ
        - "first_node": Optional[str] — v₀
        - "last_node": Optional[str] — vₖ
        - "is_self_loop": bool — True si k=1 y v₀ = v₁
        - "num_edges": int — número de aristas k
        - "boundary_is_zero": bool — True si ∂₁(c) = 0 (equivalente a is_closed)
    """
    nodes, edges = _parse_cycle_string(cycle_str)

    if len(nodes) < 2:
        return {
            "is_closed": False,
            "first_node": nodes[0] if nodes else None,
            "last_node": nodes[0] if nodes else None,
            "is_self_loop": False,
            "num_edges": 0,
            "boundary_is_zero": False,
        }

    is_closed: bool = nodes[0] == nodes[-1]

    return {
        "is_closed": is_closed,
        "first_node": nodes[0],
        "last_node": nodes[-1],
        "is_self_loop": len(nodes) == 2 and is_closed,
        "num_edges": len(edges),
        "boundary_is_zero": is_closed,
    }


def _compute_boundary_operator(edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Calcula el operador frontera ∂₁ sobre una 1-cadena.

    Para una cadena c = Σᵢ (vᵢ, vᵢ₊₁), el operador frontera es:

        ∂₁(c) = Σᵢ [vᵢ₊₁ - vᵢ]

    Esto se implementa como un contador: cada arista (u, v) contribuye
    +1 a v y -1 a u. Si ∂₁(c) = 0, todos los contadores son cero.

    Parameters
    ----------
    edges : List[Tuple[str, str]]
        Lista de aristas dirigidas (source, target).

    Returns
    -------
    Dict[str, int]
        Mapa nodo → valor del operador frontera.
        Si todos los valores son 0, la cadena es un ciclo (∂₁ = 0).
    """
    boundary: Dict[str, int] = {}
    for source, target in edges:
        boundary[source] = boundary.get(source, 0) - 1
        boundary[target] = boundary.get(target, 0) + 1
    return boundary


def _extract_expected_nodes(cycle_strings: List[str]) -> Set[str]:
    """
    Calcula el conjunto de nodos únicos esperado para una lista de ciclos.

    V = ⋃ᵢ V(cᵢ) donde V(cᵢ) son los nodos del i-ésimo ciclo.

    Parameters
    ----------
    cycle_strings : List[str]
        Lista de cadenas de ciclo.

    Returns
    -------
    Set[str]
        Unión de todos los nodos únicos.
    """
    all_nodes: Set[str] = set()
    for cycle in cycle_strings:
        nodes, _ = _parse_cycle_string(cycle)
        all_nodes.update(n for n in nodes if n)
    return all_nodes


def _make_analysis_result(
    cycles: Any = None,
    include_details: bool = True,
    additional_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Factory para construir analysis_result con estructura controlada.

    Estructura del resultado:
        {
            "details": {
                "cycles": <cycles>,
                ...additional_details
            }
        }

    Parameters
    ----------
    cycles : Any
        Valor del campo 'cycles'. None = omitir la clave.
    include_details : bool
        Si False, retorna dict sin clave 'details'.
    additional_details : Optional[Dict]
        Campos adicionales para el dict 'details'.

    Returns
    -------
    Dict[str, Any]
        Diccionario estructurado para testing.
    """
    if not include_details:
        return {}

    details: Dict[str, Any] = {}
    if cycles is not None:
        details["cycles"] = cycles
    if additional_details:
        details.update(additional_details)

    return {"details": details} if details else {"details": {}}


def _assert_anomaly_data_valid(
    result: Any,
    expected_nodes: Optional[Set[str]] = None,
    context: str = "",
) -> None:
    """
    Aserción compuesta que verifica la estructura de AnomalyData.

    Verifica los invariantes I₁ (tipo), I₂ (unicidad) y opcionalmente
    I₄ (nodos esperados).

    Parameters
    ----------
    result : Any
        Resultado de extract_anomaly_data a validar.
    expected_nodes : Optional[Set[str]]
        Si se provee, verifica inclusión exacta de nodos.
    context : str
        Descripción del contexto para mensajes de error.
    """
    prefix = f"[{context}] " if context else ""

    # I₁: Tipo correcto
    assert isinstance(result, AnomalyData), (
        f"{prefix}Invariante I₁ violado: esperado AnomalyData, "
        f"obtenido {type(result).__name__}."
    )

    # Atributo requerido
    assert hasattr(result, "nodes_in_cycles"), (
        f"{prefix}AnomalyData carece del atributo 'nodes_in_cycles'."
    )

    # I₂: Unicidad
    nodes_list = list(result.nodes_in_cycles)
    nodes_set = set(nodes_list)
    assert len(nodes_list) == len(nodes_set), (
        f"{prefix}Invariante I₂ violado: nodos duplicados encontrados. "
        f"Lista: {nodes_list}, Conjunto: {nodes_set}."
    )

    # I₄: Nodos esperados
    if expected_nodes is not None:
        actual_nodes = set(result.nodes_in_cycles)
        assert actual_nodes == expected_nodes, (
            f"{prefix}Nodos incorrectos:\n"
            f"  Esperados:  {sorted(expected_nodes)}\n"
            f"  Obtenidos:  {sorted(actual_nodes)}\n"
            f"  Faltantes:  {sorted(expected_nodes - actual_nodes)}\n"
            f"  Sobrantes:  {sorted(actual_nodes - expected_nodes)}"
        )


# =============================================================================
# TEST SUITE 1: FORMATO F₁ — LISTA DE STRINGS
# =============================================================================


class TestExtractAnomalyDataListFormat:
    """
    Tests para extract_anomaly_data con formato F₁ (lista de strings).

    Cada test verifica una configuración topológica específica del
    grafo de dependencias.

    Formato F₁:
        {"details": {"cycles": ["A -> B -> A", "C -> D -> C"]}}
    """

    def test_two_disjoint_cycles(self) -> None:
        """
        Dos ciclos disjuntos: V₁ ∩ V₂ = ∅.

        Ciclo₁: A → B → A   (V₁ = {A, B})
        Ciclo₂: C → D → C   (V₂ = {C, D})

        V(G) = V₁ ∪ V₂ = {A, B, C, D}
        """
        cycles = ["A -> B -> A", "C -> D -> C"]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))
        _assert_anomaly_data_valid(result, {"A", "B", "C", "D"}, "disjoint")

    def test_single_cycle(self) -> None:
        """
        Un único ciclo: Alpha → Beta → Alpha.

        |V| = 2, |E| = 2.
        """
        cycles = ["Alpha -> Beta -> Alpha"]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))
        _assert_anomaly_data_valid(result, {"Alpha", "Beta"}, "single")

    def test_long_cycle_4_nodes(self) -> None:
        """
        Ciclo simple de 4 nodos: A → B → C → D → A.

        |V| = 4, |E| = 4. Verifica que el parser no asuma longitud fija.
        """
        cycles = ["A -> B -> C -> D -> A"]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))
        _assert_anomaly_data_valid(result, {"A", "B", "C", "D"}, "4-cycle")

    def test_self_loop(self) -> None:
        """
        Auto-lazo: A → A (ciclo de longitud 1).

        Es el ciclo más corto posible. Topológicamente válido:
        la arista (A, A) ∈ E forma un 1-ciclo con ∂₁((A,A)) = A - A = 0.
        """
        cycles = ["A -> A"]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))
        _assert_anomaly_data_valid(result, {"A"}, "self-loop")

        validation = _validate_cycle_closure("A -> A")
        assert validation["is_self_loop"] is True
        assert validation["boundary_is_zero"] is True

    def test_shared_nodes_between_cycles(self) -> None:
        """
        Ciclos con intersección no vacía: V₁ ∩ V₂ = {B}.

        Ciclo₁: A → B → A
        Ciclo₂: B → C → B

        V(G) = {A, B, C}. El nodo B aparece exactamente una vez (I₂).
        """
        cycles = ["A -> B -> A", "B -> C -> B"]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))
        _assert_anomaly_data_valid(result, {"A", "B", "C"}, "shared-node")

    def test_hexagonal_cycle(self) -> None:
        """
        Ciclo hexagonal: A → B → C → D → E → F → A.

        |V| = 6, |E| = 6. Verifica escalamiento con ciclos largos.
        """
        cycles = ["A -> B -> C -> D -> E -> F -> A"]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))
        _assert_anomaly_data_valid(
            result,
            {"A", "B", "C", "D", "E", "F"},
            "hexagon",
        )

    @pytest.mark.parametrize(
        "cycle_strings, expected_nodes",
        [
            (
                ["W -> X -> Y -> Z -> W"],
                {"W", "X", "Y", "Z"},
            ),
            (
                ["A -> B -> A", "C -> D -> E -> C"],
                {"A", "B", "C", "D", "E"},
            ),
            (
                ["N1 -> N2 -> N3 -> N1", "N3 -> N4 -> N5 -> N3"],
                {"N1", "N2", "N3", "N4", "N5"},
            ),
        ],
        ids=["square", "mixed-lengths", "shared-vertex-triangles"],
    )
    def test_parametrized_topologies(
        self, cycle_strings: List[str], expected_nodes: Set[str],
    ) -> None:
        """
        Verificación parametrizada de múltiples configuraciones topológicas.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=cycle_strings)
        )
        _assert_anomaly_data_valid(result, expected_nodes)


# =============================================================================
# TEST SUITE 2: FORMATO F₂ — DICT LEGACY
# =============================================================================


class TestExtractAnomalyDataDictLegacy:
    """
    Tests para formato F₂ (retrocompatibilidad).

    Formato F₂:
        {"details": {"cycles": {"list": ["X -> Y -> X"]}}}

    Este formato existe por retrocompatibilidad con versiones anteriores
    del pipeline. Solo la clave 'list' debe procesarse.
    """

    def test_basic_legacy_format(self) -> None:
        """Formato legacy con un ciclo simple."""
        result = extract_anomaly_data(
            _make_analysis_result(cycles={"list": ["X -> Y -> X"]})
        )
        _assert_anomaly_data_valid(result, {"X", "Y"}, "legacy-basic")

    def test_multiple_cycles_legacy(self) -> None:
        """Formato legacy con múltiples ciclos."""
        result = extract_anomaly_data(
            _make_analysis_result(
                cycles={"list": ["A -> B -> A", "C -> D -> C"]}
            )
        )
        _assert_anomaly_data_valid(result, {"A", "B", "C", "D"}, "legacy-multi")

    def test_empty_list_legacy(self) -> None:
        """
        Formato legacy con lista vacía: {"list": []}.

        Invariante I₃: sin ciclos → nodes_in_cycles = ∅.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles={"list": []})
        )
        _assert_anomaly_data_valid(result, set(), "legacy-empty")

    def test_extra_keys_ignored(self) -> None:
        """
        Claves adicionales en el dict legacy se ignoran.

        Solo 'list' contribuye nodos.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles={
                "list": ["A -> B -> A"],
                "metadata": "ignored",
                "count": 1,
            })
        )
        _assert_anomaly_data_valid(result, {"A", "B"}, "legacy-extra-keys")

    def test_missing_list_key(self) -> None:
        """
        Dict sin clave 'list'. Debe degradar graciosamente.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles={"other_key": ["A -> B -> A"]})
        )
        _assert_anomaly_data_valid(result, context="legacy-missing-list")


# =============================================================================
# TEST SUITE 3: DEGRADACIÓN GRACIOSA (FORMATO F₃)
# =============================================================================


class TestExtractAnomalyDataGracefulDegradation:
    """
    Tests de degradación graciosa ante inputs inválidos.

    Invariante I₁: extract_anomaly_data NUNCA lanza excepción no
    capturada y SIEMPRE retorna AnomalyData.

    Invariante I₃: inputs sin ciclos válidos → nodes_in_cycles = ∅.
    """

    @pytest.mark.parametrize(
        "invalid_cycles",
        [123, 3.14, True, False, None, "not a list", set(), (), (1, 2, 3)],
        ids=[
            "int", "float", "bool-true", "bool-false", "none",
            "plain-string", "empty-set", "empty-tuple", "int-tuple",
        ],
    )
    def test_invalid_types_produce_empty_result(
        self, invalid_cycles: Any,
    ) -> None:
        """
        Tipos no soportados en 'cycles' producen AnomalyData con ∅.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=invalid_cycles)
        )
        _assert_anomaly_data_valid(result, context=f"invalid-{type(invalid_cycles).__name__}")
        assert len(result.nodes_in_cycles) == 0, (
            f"Tipo inválido {type(invalid_cycles).__name__} no debe producir nodos."
        )

    def test_missing_details_key(self) -> None:
        """analysis_result sin clave 'details' → ∅."""
        result = extract_anomaly_data({})
        _assert_anomaly_data_valid(result, set(), "no-details")

    def test_details_without_cycles(self) -> None:
        """'details' presente pero sin 'cycles' → ∅."""
        result = extract_anomaly_data({"details": {}})
        _assert_anomaly_data_valid(result, set(), "no-cycles-key")

    def test_details_is_none(self) -> None:
        """details = None → ∅."""
        result = extract_anomaly_data({"details": None})
        _assert_anomaly_data_valid(result, set(), "none-details")

    def test_empty_cycles_list(self) -> None:
        """
        cycles = [] (lista vacía). Invariante I₃.
        """
        result = extract_anomaly_data(_make_analysis_result(cycles=[]))
        _assert_anomaly_data_valid(result, set(), "empty-list")

    def test_list_of_empty_strings(self) -> None:
        """
        Lista de strings vacíos. No debe producir nodos fantasma.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["", "", ""])
        )
        _assert_anomaly_data_valid(result, set(), "empty-strings")

    def test_mixed_valid_and_invalid_elements(self) -> None:
        """
        Lista mezclando strings válidos con tipos inválidos.

        Solo los strings válidos deben contribuir nodos.
        Los elementos no-string se ignoran silenciosamente.
        """
        result = extract_anomaly_data(
            _make_analysis_result(
                cycles=["A -> B -> A", 123, "C -> D -> C", None]
            )
        )
        assert "A" in result.nodes_in_cycles
        assert "B" in result.nodes_in_cycles
        assert "C" in result.nodes_in_cycles
        assert "D" in result.nodes_in_cycles


# =============================================================================
# TEST SUITE 4: ROBUSTEZ DEL PARSER
# =============================================================================


class TestCycleParserRobustness:
    """
    Tests de robustez del parser de cadenas de ciclo.

    Verifica tolerancia a whitespace, caracteres especiales,
    nombres largos y Unicode.
    """

    @pytest.mark.parametrize(
        "cycle_variant, expected_nodes",
        [
            ("A -> B -> A", {"A", "B"}),
            ("A->B->A", {"A", "B"}),
            ("A  ->  B  ->  A", {"A", "B"}),
            (" A -> B -> A ", {"A", "B"}),
            ("A   ->B->   A", {"A", "B"}),
        ],
        ids=["canonical", "no-spaces", "extra-spaces", "external-spaces", "irregular"],
    )
    def test_whitespace_tolerance(
        self, cycle_variant: str, expected_nodes: Set[str],
    ) -> None:
        """
        Todas las variantes de whitespace representan el mismo ciclo A → B → A.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=[cycle_variant])
        )
        _assert_anomaly_data_valid(result, expected_nodes, "whitespace")

    def test_empty_string_intercalated(self) -> None:
        """
        Strings vacíos intercalados con ciclos válidos se ignoran.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["", "A -> B -> A", ""])
        )
        _assert_anomaly_data_valid(result, {"A", "B"}, "intercalated-empty")
        assert "" not in result.nodes_in_cycles

    @pytest.mark.parametrize(
        "node_name",
        [
            "node-1", "node_1", "node.1", "node:1",
            "node/path", "UPPERCASE", "MixedCase", "123numeric",
        ],
        ids=[
            "hyphens", "underscores", "dots", "colons",
            "slashes", "uppercase", "mixed-case", "numeric-prefix",
        ],
    )
    def test_special_characters_in_node_names(self, node_name: str) -> None:
        """
        Nombres de nodo con caracteres especiales no deben fragmentarse.
        """
        cycle = f"{node_name} -> other -> {node_name}"
        result = extract_anomaly_data(_make_analysis_result(cycles=[cycle]))

        assert node_name in result.nodes_in_cycles, (
            f"Nodo '{node_name}' no fue extraído correctamente. "
            f"Nodos obtenidos: {result.nodes_in_cycles}"
        )
        assert "other" in result.nodes_in_cycles

    def test_very_long_node_names(self) -> None:
        """
        Nombres de nodo de 1000 caracteres. No debe truncar.
        """
        long_name = "X" * 1000
        cycle = f"{long_name} -> Y -> {long_name}"
        result = extract_anomaly_data(_make_analysis_result(cycles=[cycle]))

        assert long_name in result.nodes_in_cycles
        assert "Y" in result.nodes_in_cycles


# =============================================================================
# TEST SUITE 5: INVARIANTE I₁ — TIPO DE RETORNO
# =============================================================================


class TestReturnTypeInvariant:
    """
    Verificación exhaustiva del invariante I₁: extract_anomaly_data
    SIEMPRE retorna AnomalyData, independientemente del input.
    """

    @pytest.mark.parametrize(
        "input_data",
        [
            {},
            {"details": None},
            {"details": {}},
            {"details": {"cycles": []}},
            {"details": {"cycles": ["A -> B -> A"]}},
            {"details": {"cycles": {"list": ["X -> Y -> X"]}}},
            {"details": {"cycles": 999}},
            {"details": {"cycles": ["", "   "]}},
            {"details": {"other_key": "value"}},
            {"unrelated": "data"},
        ],
        ids=[
            "empty-dict", "none-details", "empty-details", "empty-cycles",
            "valid-F1", "valid-F2", "invalid-type", "empty-strings",
            "no-cycles-key", "no-details-key",
        ],
    )
    def test_always_returns_anomaly_data(self, input_data: Dict) -> None:
        """
        Invariante I₁: type(result) == AnomalyData para todo input.
        """
        result = extract_anomaly_data(input_data)
        assert isinstance(result, AnomalyData), (
            f"Invariante I₁ violado: esperado AnomalyData, "
            f"obtenido {type(result).__name__} para input {input_data}"
        )


# =============================================================================
# TEST SUITE 6: INVARIANTE I₂ — UNICIDAD DE NODOS
# =============================================================================


class TestNodeUniquenessInvariant:
    """
    Verificación del invariante I₂: nodes_in_cycles no contiene duplicados.
    """

    def test_overlapping_cycles_no_duplicates(self) -> None:
        """
        Triángulo completo con cada arista como ciclo.

        Todos los nodos aparecen en múltiples ciclos, pero el resultado
        debe contener cada nodo exactamente una vez.
        """
        cycles = [
            "A -> B -> A",
            "B -> C -> B",
            "C -> A -> C",
            "A -> B -> C -> A",
        ]
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))

        nodes_list = list(result.nodes_in_cycles)
        assert len(nodes_list) == len(set(nodes_list)), (
            f"Duplicados en nodes_in_cycles: {nodes_list}"
        )
        assert set(nodes_list) == {"A", "B", "C"}

    def test_same_cycle_repeated(self) -> None:
        """
        El mismo ciclo listado 10 veces. Resultado idéntico a una vez.
        """
        cycles = ["A -> B -> A"] * 10
        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))

        assert set(result.nodes_in_cycles) == {"A", "B"}
        assert len(result.nodes_in_cycles) == 2


# =============================================================================
# TEST SUITE 7: INVARIANTE I₅ — INMUTABILIDAD DEL INPUT
# =============================================================================


class TestInputImmutabilityInvariant:
    """
    Verificación del invariante I₅: extract_anomaly_data no modifica su input.
    """

    def test_list_format_immutability(self) -> None:
        """
        Input con formato F₁ permanece intacto tras la extracción.
        """
        original_cycles = ["A -> B -> A", "C -> D -> C"]
        input_data = {"details": {"cycles": original_cycles.copy()}}
        snapshot = deepcopy(input_data)

        _ = extract_anomaly_data(input_data)

        assert input_data == snapshot, (
            "Invariante I₅ violado: el input fue mutado.\n"
            f"  Antes:   {snapshot}\n"
            f"  Después: {input_data}"
        )

    def test_dict_format_immutability(self) -> None:
        """
        Input con formato F₂ permanece intacto tras la extracción.
        """
        input_data = {
            "details": {
                "cycles": {
                    "list": ["X -> Y -> X"],
                    "metadata": {"count": 1},
                }
            }
        }
        snapshot = deepcopy(input_data)

        _ = extract_anomaly_data(input_data)

        assert input_data == snapshot, (
            "Invariante I₅ violado: el input fue mutado (formato legacy)."
        )


# =============================================================================
# TEST SUITE 8: INVARIANTE I₆ — OPERADOR FRONTERA ∂₁
# =============================================================================


class TestBoundaryOperatorInvariant:
    """
    Verificación del invariante I₆: ∂₁(c) = 0 para ciclos cerrados.

    El operador frontera sobre una 1-cadena c = Σᵢ (vᵢ, vᵢ₊₁) es:

        ∂₁(c) = Σᵢ [vᵢ₊₁ - vᵢ]

    Para un ciclo cerrado (v₀ = vₖ), la suma telescópica cancela
    y ∂₁(c) = 0. Este es el criterio algebraico de cierre.
    """

    @pytest.mark.parametrize(
        "cycle_str",
        [
            "A -> B -> A",
            "A -> B -> C -> A",
            "A -> B -> C -> D -> E -> F -> A",
            "X -> X",
        ],
        ids=["digon", "triangle", "hexagon", "self-loop"],
    )
    def test_closed_cycles_have_zero_boundary(self, cycle_str: str) -> None:
        """
        Para ciclos cerrados, ∂₁(c) = 0: la suma de las contribuciones
        de cada arista al operador frontera es el vector nulo.
        """
        nodes, edges = _parse_cycle_string(cycle_str)
        boundary = _compute_boundary_operator(edges)

        # Verificar que todos los valores del boundary son 0
        for node, value in boundary.items():
            assert value == 0, (
                f"∂₁ ≠ 0 para ciclo '{cycle_str}': "
                f"∂₁({node}) = {value}. "
                f"Boundary completo: {boundary}."
            )

    @pytest.mark.parametrize(
        "path_str",
        [
            "A -> B -> C",
            "X -> Y",
        ],
        ids=["open-path-3", "open-path-2"],
    )
    def test_open_paths_have_nonzero_boundary(self, path_str: str) -> None:
        """
        Para caminos abiertos (v₀ ≠ vₖ), ∂₁(c) ≠ 0.

        ∂₁(A → B → C) = C - A ≠ 0.
        """
        nodes, edges = _parse_cycle_string(path_str)
        boundary = _compute_boundary_operator(edges)

        # Al menos un nodo tiene valor no nulo
        nonzero_values = {n: v for n, v in boundary.items() if v != 0}
        assert len(nonzero_values) > 0, (
            f"Camino abierto '{path_str}' reporta ∂₁ = 0, "
            f"pero debería tener boundary no nulo. "
            f"Boundary: {boundary}."
        )

    def test_boundary_operator_consistency_with_closure_validation(self) -> None:
        """
        Verifica que ∂₁(c) = 0 ⟺ is_closed para múltiples cadenas.
        """
        test_cases = [
            ("A -> B -> A", True),
            ("A -> B -> C -> A", True),
            ("A -> B -> C", False),
            ("X -> X", True),
            ("P -> Q", False),
        ]

        for cycle_str, expected_closed in test_cases:
            nodes, edges = _parse_cycle_string(cycle_str)
            boundary = _compute_boundary_operator(edges)
            boundary_is_zero = all(v == 0 for v in boundary.values())

            validation = _validate_cycle_closure(cycle_str)

            assert boundary_is_zero == expected_closed, (
                f"Inconsistencia para '{cycle_str}': "
                f"∂₁=0 es {boundary_is_zero}, esperado {expected_closed}."
            )
            assert validation["is_closed"] == expected_closed, (
                f"Inconsistencia entre ∂₁ y validate_cycle_closure "
                f"para '{cycle_str}'."
            )
            assert validation["boundary_is_zero"] == boundary_is_zero


# =============================================================================
# TEST SUITE 9: DETERMINISMO E IDEMPOTENCIA
# =============================================================================


class TestDeterminismAndIdempotence:
    """
    Verifica que extract_anomaly_data es una función pura:
    determinista y sin efectos secundarios observables.
    """

    def test_determinism_same_input(self) -> None:
        """
        La misma entrada produce el mismo resultado en múltiples llamadas.
        """
        input_data = _make_analysis_result(cycles=["X -> Y -> Z -> X"])

        results = [extract_anomaly_data(input_data) for _ in range(5)]
        first_nodes = set(results[0].nodes_in_cycles)

        for i, result in enumerate(results[1:], start=2):
            assert set(result.nodes_in_cycles) == first_nodes, (
                f"Llamada #{i} produjo nodos distintos: "
                f"{set(result.nodes_in_cycles)} vs {first_nodes}."
            )

    def test_order_independence(self) -> None:
        """
        El orden de los ciclos en la lista no afecta el conjunto de nodos.

        La extracción es una operación de conjunto (unión), que es
        conmutativa y asociativa.
        """
        order_1 = ["A -> B -> A", "C -> D -> C"]
        order_2 = ["C -> D -> C", "A -> B -> A"]

        result_1 = extract_anomaly_data(_make_analysis_result(cycles=order_1))
        result_2 = extract_anomaly_data(_make_analysis_result(cycles=order_2))

        assert set(result_1.nodes_in_cycles) == set(result_2.nodes_in_cycles), (
            "El orden de los ciclos afectó el resultado. "
            f"Orden 1: {set(result_1.nodes_in_cycles)}, "
            f"Orden 2: {set(result_2.nodes_in_cycles)}."
        )


# =============================================================================
# TEST SUITE 10: EXTRACCIÓN DE ARISTAS (CONDICIONAL)
# =============================================================================


_HAS_EDGES_ATTRIBUTE: bool = hasattr(AnomalyData, "edges_in_cycles")


@pytest.mark.skipif(
    not _HAS_EDGES_ATTRIBUTE,
    reason="AnomalyData no expone edges_in_cycles",
)
class TestExtractAnomalyEdges:
    """
    Tests de extracción de aristas dirigidas.

    Solo se ejecutan si AnomalyData implementa edges_in_cycles.

    Para un ciclo A → B → C → A, las aristas son:
        E = {(A,B), (B,C), (C,A)}

    Las aristas son dirigidas: (A,B) ≠ (B,A).
    """

    def test_simple_cycle_edges(self) -> None:
        """
        Ciclo A → B → C → A genera exactamente 3 aristas dirigidas.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["A -> B -> C -> A"])
        )
        edges = set(result.edges_in_cycles)

        assert edges == {("A", "B"), ("B", "C"), ("C", "A")}, (
            f"Aristas incorrectas: {edges}"
        )

    def test_multiple_cycles_edges(self) -> None:
        """
        Aristas de múltiples ciclos se combinan.
        """
        result = extract_anomaly_data(
            _make_analysis_result(
                cycles=["A -> B -> A", "C -> D -> E -> C"]
            )
        )
        edges = set(result.edges_in_cycles)

        assert ("A", "B") in edges
        assert ("B", "A") in edges
        assert ("C", "D") in edges
        assert ("D", "E") in edges
        assert ("E", "C") in edges

    def test_empty_cycles_no_edges(self) -> None:
        """Sin ciclos → sin aristas."""
        result = extract_anomaly_data(_make_analysis_result(cycles=[]))
        assert len(result.edges_in_cycles) == 0

    def test_self_loop_edge(self) -> None:
        """Auto-lazo A → A genera arista (A, A)."""
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["A -> A"])
        )
        assert ("A", "A") in result.edges_in_cycles

    def test_edges_are_directed(self) -> None:
        """
        Aristas dirigidas: ciclo A → B → A contiene (A,B) Y (B,A).
        Son aristas distintas.
        """
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["A -> B -> A"])
        )
        edges = set(result.edges_in_cycles)

        assert ("A", "B") in edges
        assert ("B", "A") in edges
        assert len(edges) == 2


# =============================================================================
# TEST SUITE 11: RENDIMIENTO
# =============================================================================


@pytest.mark.stress
class TestExtractionPerformance:
    """
    Tests de rendimiento bajo carga.

    Estos tests verifican que la extracción escale linealmente
    con el número de ciclos y nodos, sin explosión combinatoria.
    """

    def test_many_small_cycles(self) -> None:
        """
        100 ciclos pequeños (stress en número de ciclos).

        Complejidad esperada: O(n · k) donde n = num_cycles, k = avg_length.
        """
        num_cycles = 100
        cycles = [
            f"Node_{i} -> Node_{i + 1} -> Node_{i}"
            for i in range(num_cycles)
        ]

        result = extract_anomaly_data(_make_analysis_result(cycles=cycles))

        assert isinstance(result, AnomalyData)
        assert len(result.nodes_in_cycles) > 0

    def test_single_long_cycle(self) -> None:
        """
        Un único ciclo de 100 nodos (stress en longitud).
        """
        num_nodes = 100
        node_names = [f"N{i}" for i in range(num_nodes)]
        cycle = " -> ".join(node_names + [node_names[0]])

        result = extract_anomaly_data(_make_analysis_result(cycles=[cycle]))

        assert len(result.nodes_in_cycles) == num_nodes

    def test_extraction_completes_in_bounded_time(self) -> None:
        """
        500 ciclos se extraen en menos de _PERFORMANCE_TIME_LIMIT segundos.
        """
        import time

        cycles = [f"A{i} -> B{i} -> A{i}" for i in range(500)]
        input_data = _make_analysis_result(cycles=cycles)

        start = time.perf_counter()
        result = extract_anomaly_data(input_data)
        elapsed = time.perf_counter() - start

        assert elapsed < _PERFORMANCE_TIME_LIMIT, (
            f"Extracción tardó {elapsed:.3f}s, "
            f"límite: {_PERFORMANCE_TIME_LIMIT}s. "
            f"Posible explosión combinatoria en el parser."
        )
        assert isinstance(result, AnomalyData)


# =============================================================================
# TEST SUITE 12: INTEGRACIÓN CON AnomalyData
# =============================================================================


class TestAnomalyDataStructure:
    """
    Tests que verifican la estructura y usabilidad del objeto AnomalyData.
    """

    def test_has_required_attribute(self) -> None:
        """AnomalyData expone nodes_in_cycles como atributo iterable."""
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["A -> B -> A"])
        )
        assert hasattr(result, "nodes_in_cycles")
        assert hasattr(result.nodes_in_cycles, "__iter__")

    def test_nodes_are_strings(self) -> None:
        """Todos los nodos extraídos son instancias de str."""
        result = extract_anomaly_data(
            _make_analysis_result(
                cycles=["Node1 -> Node2 -> Node1", "A -> B -> C -> A"]
            )
        )
        for node in result.nodes_in_cycles:
            assert isinstance(node, str), (
                f"Nodo no es str: {node!r} (tipo: {type(node).__name__})"
            )

    def test_nodes_usable_in_set_operations(self) -> None:
        """nodes_in_cycles soporta operaciones de conjunto estándar."""
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["A -> B -> C -> A"])
        )
        nodes_set = set(result.nodes_in_cycles)

        assert "A" in nodes_set
        assert nodes_set & {"A", "B"} == {"A", "B"}
        assert len(nodes_set - {"Z"}) == len(nodes_set)

    def test_has_string_representation(self) -> None:
        """AnomalyData tiene representación str/repr legible."""
        result = extract_anomaly_data(
            _make_analysis_result(cycles=["A -> B -> A"])
        )
        assert len(str(result)) > 0
        assert len(repr(result)) > 0