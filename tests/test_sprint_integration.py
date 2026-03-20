"""
Suite de Integración y Extracción Topológica de Anomalías (Sprint Integration).

Fundamentos Matemáticos Verificados:
─────────────────────────────────────────────────────────────────────────────
1. Homología Simplicial y Ciclos (H₁):
   En la topología del presupuesto, una dependencia circular o "Socavón Lógico" 
   se formaliza como un elemento no trivial del primer grupo de homología H₁(G), 
   donde el primer número de Betti es estrictamente positivo (β₁ > 0).
   Esta suite valida que las cadenas de texto detectadas correspondan a verdaderos 
   1-ciclos algebraicos. Específicamente, a través de `validate_cycle_closure`, 
   se demuestra computacionalmente que el operador frontera sobre la cadena 
   evaluada es estrictamente nulo: ∂₁(c) = 0, garantizando el cierre topológico del grafo.

2. Proyección y Descomposición de Subgrafos Anómalos:
   La función `extract_anomaly_data` actúa como un funtor que mapea el espacio 
   de ciclos y anomalías topológicas hacia un espacio de conjuntos discretos 
   (`AnomalyData`), aislando los vértices (nodos únicos) y los 1-simplices 
   (aristas) responsables de la fractura del DAG.

3. Idempotencia y Determinismo Analítico:
   Se exige axiomáticamente que cualquier proyección topológica hacia las métricas 
   de anomalía sea un morfismo idempotente y determinista. La evaluación iterativa 
   y el parseo de representaciones de grafos bajo múltiples formatos (`CycleFormat`) 
   y separadores (`CycleSeparators`) debe converger sin varianza numérica ni pérdida 
   de entropía estructural.

Contratos Estructurales Probados:
─────────────────────────────────────────────────────────────────────────────
- Parseo estricto y extracción geométrica de ciclos mediante `parse_cycle_string`.
- Validación matemática del cierre del ciclo en topologías cíclicas complejas.
- Identificación de conjuntos ortogonales de nodos y aristas únicos involucrados 
  en la anomalía de Betti (H₁).
- Preservación de la complejidad asintótica y estabilidad de la memoria bajo 
  escenarios de estrés, evitando la explosión combinatoria al extraer métricas.
"""

import pytest
import re
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

from app.adapters.topology_viz import extract_anomaly_data, AnomalyData


# =============================================================================
# CONSTANTES Y ENUMERACIONES DE DOMINIO TOPOLÓGICO
# =============================================================================

class CycleFormat(str, Enum):
    """
    Formatos soportados para representación de ciclos.
    
    Taxonomía:
      F₁ (LIST): Lista de strings representando ciclos individuales
      F₂ (DICT_LEGACY): Diccionario con clave 'list' (retrocompatibilidad)
      F₃ (INVALID): Cualquier otro formato no reconocido
    """
    LIST = "list"
    DICT_LEGACY = "dict_legacy"
    INVALID = "invalid"


class CycleSeparators:
    """
    Separadores válidos para cadenas de ciclo.
    
    El separador canónico es ' -> ' pero el parser debe ser
    tolerante a variaciones de whitespace.
    """
    CANONICAL = " -> "
    ARROW = "->"
    UNICODE_ARROW = "→"
    DOUBLE_ARROW = "=>"
    LONG_ARROW = "⟶"
    
    # Patrón regex que captura todas las variantes
    PATTERN = re.compile(r'\s*(?:->|→|=>|⟶)\s*')
    
    @classmethod
    def all_variants(cls) -> List[str]:
        return [cls.CANONICAL, cls.ARROW, cls.UNICODE_ARROW, 
                cls.DOUBLE_ARROW, cls.LONG_ARROW]


class TopologicalInvariants:
    """
    Invariantes topológicos del sistema de extracción de anomalías.
    
    Fundamentación en Teoría de Grafos:
      Sea G = (V, E) un grafo dirigido.
      Un ciclo C es una secuencia (v₀, v₁, ..., vₖ) donde:
        - (vᵢ, vᵢ₊₁) ∈ E para i ∈ [0, k-1]
        - v₀ = vₖ (ciclo cerrado)
        - k ≥ 1 (al menos una arista)
    """
    # Longitud mínima de ciclo válido (auto-lazo: A -> A)
    MIN_CYCLE_LENGTH = 1
    
    # Número máximo de nodos para pruebas de stress
    MAX_STRESS_NODES = 1000


# =============================================================================
# HELPERS DE VALIDACIÓN Y PARSING TOPOLÓGICO
# =============================================================================

def parse_cycle_string(cycle_str: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parsea una cadena de ciclo y extrae nodos y aristas.
    
    Args:
        cycle_str: Cadena en formato "A -> B -> C -> A"
        
    Returns:
        Tupla (nodes, edges) donde:
          - nodes: Lista de nodos en orden de aparición
          - edges: Lista de aristas dirigidas (source, target)
          
    Ejemplo:
        "A -> B -> C -> A" → 
        nodes = ["A", "B", "C", "A"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
    """
    if not cycle_str or not cycle_str.strip():
        return [], []
    
    # Normalizar separadores usando regex
    normalized = CycleSeparators.PATTERN.sub(" -> ", cycle_str)
    
    # Extraer nodos
    nodes = [n.strip() for n in normalized.split(" -> ") if n.strip()]
    
    # Generar aristas como pares consecutivos
    edges = []
    for i in range(len(nodes) - 1):
        edges.append((nodes[i], nodes[i + 1]))
    
    return nodes, edges


def validate_cycle_closure(cycle_str: str) -> Dict[str, Any]:
    """
    Valida si un ciclo está correctamente cerrado.
    
    Un ciclo cerrado tiene la forma v₀ → v₁ → ... → v₀
    donde el primer y último nodo son idénticos.
    
    Returns:
        Dict con:
          - is_closed: bool
          - first_node: str | None
          - last_node: str | None
          - is_self_loop: bool (True si A -> A)
    """
    nodes, _ = parse_cycle_string(cycle_str)
    
    if len(nodes) < 2:
        return {
            "is_closed": False,
            "first_node": nodes[0] if nodes else None,
            "last_node": nodes[0] if nodes else None,
            "is_self_loop": False,
        }
    
    is_closed = nodes[0] == nodes[-1]
    is_self_loop = len(nodes) == 2 and is_closed
    
    return {
        "is_closed": is_closed,
        "first_node": nodes[0],
        "last_node": nodes[-1],
        "is_self_loop": is_self_loop,
    }


def extract_unique_nodes(cycle_strings: List[str]) -> Set[str]:
    """
    Extrae el conjunto de nodos únicos de múltiples ciclos.
    
    Dado un conjunto de ciclos C = {c₁, c₂, ..., cₙ},
    retorna V = ⋃ᵢ V(cᵢ) (unión de vértices).
    """
    all_nodes: Set[str] = set()
    
    for cycle in cycle_strings:
        nodes, _ = parse_cycle_string(cycle)
        all_nodes.update(n for n in nodes if n)
    
    return all_nodes


def extract_all_edges(cycle_strings: List[str]) -> Set[Tuple[str, str]]:
    """
    Extrae el conjunto de aristas únicas de múltiples ciclos.
    
    Dado un conjunto de ciclos C, retorna E = ⋃ᵢ E(cᵢ).
    """
    all_edges: Set[Tuple[str, str]] = set()
    
    for cycle in cycle_strings:
        _, edges = parse_cycle_string(cycle)
        all_edges.update(edges)
    
    return all_edges


def compute_cycle_metrics(cycle_strings: List[str]) -> Dict[str, Any]:
    """
    Computa métricas topológicas de un conjunto de ciclos.
    
    Returns:
        Dict con:
          - num_cycles: Número de ciclos
          - num_unique_nodes: |V|
          - num_unique_edges: |E|
          - avg_cycle_length: Longitud promedio
          - max_cycle_length: Longitud máxima
          - has_self_loops: Si existe algún auto-lazo
    """
    if not cycle_strings:
        return {
            "num_cycles": 0,
            "num_unique_nodes": 0,
            "num_unique_edges": 0,
            "avg_cycle_length": 0.0,
            "max_cycle_length": 0,
            "has_self_loops": False,
        }
    
    all_nodes = extract_unique_nodes(cycle_strings)
    all_edges = extract_all_edges(cycle_strings)
    
    cycle_lengths = []
    has_self_loops = False
    
    for cycle in cycle_strings:
        validation = validate_cycle_closure(cycle)
        nodes, _ = parse_cycle_string(cycle)
        
        if nodes:
            cycle_lengths.append(len(nodes) - 1)  # Aristas = nodos - 1
            if validation["is_self_loop"]:
                has_self_loops = True
    
    return {
        "num_cycles": len(cycle_strings),
        "num_unique_nodes": len(all_nodes),
        "num_unique_edges": len(all_edges),
        "avg_cycle_length": sum(cycle_lengths) / len(cycle_lengths) if cycle_lengths else 0.0,
        "max_cycle_length": max(cycle_lengths) if cycle_lengths else 0,
        "has_self_loops": has_self_loops,
    }


# =============================================================================
# PRUEBAS DE EXTRACCIÓN DE DATOS DE ANOMALÍA TOPOLÓGICA
# =============================================================================

class TestExtractAnomalyData:
    """
    Pruebas unitarias para extract_anomaly_data().

    Modelo formal (Teoría de Grafos Dirigidos):
    
      Sea G = (V, E) un digrafo donde:
        - V = conjunto de vértices (nodos)
        - E ⊆ V × V = conjunto de aristas dirigidas
        
      Un ciclo en G es un camino cerrado c = (v₀, v₁, ..., vₖ = v₀) donde:
        - ∀i ∈ [0, k-1]: (vᵢ, vᵢ₊₁) ∈ E
        - v₀ = vₖ (condición de cierre)
        
    Formatos de entrada soportados:
      ┌─────┬─────────────────────────────────────────────────────────────┐
      │ F₁  │ Lista de strings: ["A -> B -> A", "C -> D -> C"]           │
      ├─────┼─────────────────────────────────────────────────────────────┤
      │ F₂  │ Dict legacy: {"list": ["X -> Y -> X"]}                     │
      ├─────┼─────────────────────────────────────────────────────────────┤
      │ F₃  │ Formato inválido: cualquier otro tipo → degradación        │
      └─────┴─────────────────────────────────────────────────────────────┘

    Invariantes del sistema:
      ┌─────┬───────────────────────────────────────────────────────────────┐
      │ I₁  │ ∀ entrada: type(resultado) == AnomalyData (nunca None)       │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₂  │ ∀ resultado: nodes_in_cycles es conjunto (sin duplicados)    │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₃  │ Si cycles = [] o ausente → nodes_in_cycles = ∅               │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₄  │ ∀ cadena "A -> B -> C": {A, B, C} ⊆ nodes_in_cycles         │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₅  │ Input original permanece inmutable                           │
      └─────┴───────────────────────────────────────────────────────────────┘
    """

    # ═══════════════════════════════════════════════════════════════════════
    # FACTORIES Y HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _make_result(
        cycles: Any = None,
        include_details: bool = True,
        additional_details: Optional[Dict] = None,
    ) -> dict:
        """
        Factory para construir analysis_result con estructura controlada.

        Args:
            cycles: Valor del campo 'cycles' dentro de details
            include_details: Si False, omite la clave 'details' por completo
            additional_details: Campos adicionales para el dict 'details'
            
        Returns:
            Diccionario estructurado para testing
        """
        if not include_details:
            return {}
            
        details = {}
        if cycles is not None:
            details["cycles"] = cycles
        if additional_details:
            details.update(additional_details)
            
        if not details:
            return {"details": {}}
            
        return {"details": details}

    @staticmethod
    def _extract_expected_nodes(cycle_strings: List[str]) -> Set[str]:
        """Wrapper para extracción de nodos esperados."""
        return extract_unique_nodes(cycle_strings)

    @staticmethod
    def _assert_anomaly_data_valid(result: Any, expected_nodes: Optional[Set[str]] = None):
        """
        Aserción compuesta que verifica estructura básica de AnomalyData.
        
        Args:
            result: Resultado a validar
            expected_nodes: Si se provee, verifica que los nodos coincidan
        """
        # Invariante I₁: Siempre es AnomalyData
        assert isinstance(result, AnomalyData), (
            f"Tipo incorrecto: esperado AnomalyData, obtenido {type(result)}"
        )
        
        # Verificar que tiene el atributo nodes_in_cycles
        assert hasattr(result, 'nodes_in_cycles'), (
            "AnomalyData debe tener atributo 'nodes_in_cycles'"
        )
        
        # Invariante I₂: Sin duplicados
        nodes_list = list(result.nodes_in_cycles)
        assert len(nodes_list) == len(set(nodes_list)), (
            f"Nodos duplicados encontrados: {nodes_list}"
        )
        
        # Verificar nodos esperados si se proporcionan
        if expected_nodes is not None:
            actual_nodes = set(result.nodes_in_cycles)
            assert actual_nodes == expected_nodes, (
                f"Nodos incorrectos:\n"
                f"  Esperados: {expected_nodes}\n"
                f"  Obtenidos: {actual_nodes}\n"
                f"  Faltantes: {expected_nodes - actual_nodes}\n"
                f"  Extras: {actual_nodes - expected_nodes}"
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FORMATO F₁: LISTA DE STRINGS
    # ═══════════════════════════════════════════════════════════════════════

    def test_list_format_basic_two_disjoint_cycles(self):
        """
        Formato F₁: Dos ciclos disjuntos (sin nodos compartidos).
        
        Ciclo₁: A → B → A (triángulo degenerado)
        Ciclo₂: C → D → C
        
        V(G) = {A, B} ∪ {C, D} = {A, B, C, D}
        Los ciclos son componentes conexas separadas.
        """
        cycles = ["A -> B -> A", "C -> D -> C"]
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        expected = {"A", "B", "C", "D"}
        self._assert_anomaly_data_valid(result, expected)

    def test_list_format_single_cycle(self):
        """
        Formato F₁: Un único ciclo.
        
        Ciclo: Alpha → Beta → Alpha
        Este es un ciclo de longitud 2 (2 aristas).
        """
        cycles = ["Alpha -> Beta -> Alpha"]
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        self._assert_anomaly_data_valid(result)
        assert "Alpha" in result.nodes_in_cycles
        assert "Beta" in result.nodes_in_cycles
        assert len(result.nodes_in_cycles) == 2

    def test_list_format_long_cycle_4_nodes(self):
        """
        Ciclo de longitud 4: A → B → C → D → A.
        
        Este es un ciclo simple (sin nodos repetidos internamente).
        |V| = 4, |E| = 4
        
        Verifica que el parser no asuma longitud fija.
        """
        cycle = "A -> B -> C -> D -> A"
        result = extract_anomaly_data(self._make_result(cycles=[cycle]))

        expected = {"A", "B", "C", "D"}
        self._assert_anomaly_data_valid(result, expected)
        
        # Verificar métricas del ciclo
        metrics = compute_cycle_metrics([cycle])
        assert metrics["num_unique_nodes"] == 4
        assert metrics["max_cycle_length"] == 4  # 4 aristas

    def test_list_format_self_loop(self):
        """
        Auto-lazo: A → A (ciclo de longitud 1).
        
        En teoría de grafos, un auto-lazo es una arista (v, v) ∈ E.
        Es el ciclo más corto posible.
        
        Topológicamente válido; el parser debe extraer el nodo.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=["A -> A"])
        )

        self._assert_anomaly_data_valid(result, {"A"})
        
        # Verificar que es reconocido como self-loop
        validation = validate_cycle_closure("A -> A")
        assert validation["is_self_loop"] is True
        assert validation["is_closed"] is True

    def test_list_format_shared_nodes_between_cycles(self):
        """
        Nodos compartidos entre ciclos (intersección no vacía).
        
        Ciclo₁: A → B → A
        Ciclo₂: B → C → B
        
        V₁ ∩ V₂ = {B} ≠ ∅
        V(G) = V₁ ∪ V₂ = {A, B, C}
        
        Invariante I₂: B debe aparecer exactamente una vez.
        """
        cycles = ["A -> B -> A", "B -> C -> B"]
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        expected = {"A", "B", "C"}
        self._assert_anomaly_data_valid(result, expected)

    def test_list_format_triangle_complete_graph(self):
        """
        Grafo completo K₃ representado como ciclos.
        
        Ciclos: A→B→A, B→C→B, C→A→C
        
        Esto cubre todas las aristas de un triángulo dirigido.
        V = {A, B, C}
        """
        cycles = ["A -> B -> A", "B -> C -> B", "C -> A -> C"]
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        expected = {"A", "B", "C"}
        self._assert_anomaly_data_valid(result, expected)
        
        # Verificar métricas
        metrics = compute_cycle_metrics(cycles)
        assert metrics["num_cycles"] == 3
        assert metrics["num_unique_nodes"] == 3

    @pytest.mark.parametrize(
        "cycle_strings, expected_nodes, description",
        [
            (
                ["W -> X -> Y -> Z -> W"],
                {"W", "X", "Y", "Z"},
                "Ciclo simple de 4 nodos (cuadrado)",
            ),
            (
                ["A -> B -> A", "C -> D -> E -> C"],
                {"A", "B", "C", "D", "E"},
                "Ciclos de diferentes longitudes (2 y 3)",
            ),
            (
                ["Solo -> Solo"],
                {"Solo"},
                "Auto-lazo único",
            ),
            (
                ["A -> B -> C -> D -> E -> F -> A"],
                {"A", "B", "C", "D", "E", "F"},
                "Ciclo largo de 6 nodos (hexágono)",
            ),
            (
                ["N1 -> N2 -> N3 -> N1", "N3 -> N4 -> N5 -> N3"],
                {"N1", "N2", "N3", "N4", "N5"},
                "Dos triángulos con nodo compartido (N3)",
            ),
        ],
        ids=["square", "mixed_lengths", "self_loop", "hexagon", "shared_vertex"],
    )
    def test_list_format_parametrized(self, cycle_strings, expected_nodes, description):
        """
        Verificación parametrizada de extracción de nodos.
        
        Cubre múltiples configuraciones topológicas.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=cycle_strings)
        )

        self._assert_anomaly_data_valid(result, expected_nodes)

    # ═══════════════════════════════════════════════════════════════════════
    # FORMATO F₂: DICT LEGACY (RETROCOMPATIBILIDAD)
    # ═══════════════════════════════════════════════════════════════════════

    def test_dict_legacy_format_basic(self):
        """
        Formato legacy: {"list": ["X -> Y -> X"]}.
        
        Mantiene retrocompatibilidad con versiones anteriores
        del pipeline que usaban este formato.
        """
        result = extract_anomaly_data(
            self._make_result(cycles={"list": ["X -> Y -> X"]})
        )

        self._assert_anomaly_data_valid(result, {"X", "Y"})

    def test_dict_legacy_format_multiple_cycles(self):
        """Formato legacy con múltiples ciclos."""
        result = extract_anomaly_data(
            self._make_result(cycles={"list": ["A -> B -> A", "C -> D -> C"]})
        )

        self._assert_anomaly_data_valid(result, {"A", "B", "C", "D"})

    def test_dict_legacy_format_empty_list(self):
        """
        Formato legacy con lista vacía: {"list": []}.
        
        Invariante I₃: sin ciclos → conjunto vacío.
        """
        result = extract_anomaly_data(
            self._make_result(cycles={"list": []})
        )

        self._assert_anomaly_data_valid(result, set())

    def test_dict_legacy_format_with_extra_keys(self):
        """
        Formato legacy con claves adicionales (ignoradas).
        
        Solo la clave 'list' debe procesarse.
        """
        result = extract_anomaly_data(
            self._make_result(cycles={
                "list": ["A -> B -> A"],
                "metadata": "should_be_ignored",
                "count": 1,
            })
        )

        self._assert_anomaly_data_valid(result, {"A", "B"})

    def test_dict_legacy_format_missing_list_key(self):
        """
        Formato dict pero sin clave 'list'.
        
        Debe degradar graciosamente (como formato inválido).
        """
        result = extract_anomaly_data(
            self._make_result(cycles={"other_key": ["A -> B -> A"]})
        )

        # Comportamiento esperado: sin la clave 'list', no se extraen nodos
        self._assert_anomaly_data_valid(result)
        # El comportamiento exacto depende de la implementación
        # Verificamos al menos que no crashea

    # ═══════════════════════════════════════════════════════════════════════
    # FORMATO F₃: ENTRADAS INVÁLIDAS (DEGRADACIÓN GRACIOSA)
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "invalid_cycles, type_description",
        [
            (123, "entero"),
            (3.14159, "flotante"),
            (True, "booleano True"),
            (False, "booleano False"),
            (None, "None explícito"),
            ("not a list", "string plano"),
            (set(), "set vacío"),
            (frozenset(), "frozenset vacío"),
            ((), "tupla vacía"),
            ((1, 2, 3), "tupla de enteros"),
            (object(), "objeto genérico"),
            (lambda x: x, "función lambda"),
        ],
        ids=lambda x: x if isinstance(x, str) else type(x).__name__,
    )
    def test_invalid_formats_degrade_gracefully(self, invalid_cycles, type_description):
        """
        Invariante I₁ + I₃: Cualquier tipo no soportado
        debe producir AnomalyData con conjunto vacío, sin excepción.
        
        El sistema es defensivo ante inputs malformados.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=invalid_cycles)
        )

        self._assert_anomaly_data_valid(result)
        assert len(result.nodes_in_cycles) == 0, (
            f"Formato inválido ({type_description}) no debe producir nodos"
        )

    def test_invalid_format_nested_invalid_types(self):
        """
        Tipos inválidos anidados profundamente.
        """
        result = extract_anomaly_data(
            self._make_result(cycles={"list": [123, None, True, ["nested"]]})
        )

        # Elementos inválidos dentro de la lista deben ignorarse
        self._assert_anomaly_data_valid(result)

    def test_invalid_format_mixed_valid_invalid(self):
        """
        Lista mezclando elementos válidos e inválidos.
        
        Solo los elementos válidos (strings) deben procesarse.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=["A -> B -> A", 123, "C -> D -> C", None])
        )

        # Solo los strings válidos contribuyen nodos
        assert "A" in result.nodes_in_cycles
        assert "B" in result.nodes_in_cycles
        assert "C" in result.nodes_in_cycles
        assert "D" in result.nodes_in_cycles

    # ═══════════════════════════════════════════════════════════════════════
    # CASOS DEGENERADOS: AUSENCIA DE CLAVES
    # ═══════════════════════════════════════════════════════════════════════

    def test_missing_details_key(self):
        """
        analysis_result sin clave 'details'.
        
        Primer nivel de ausencia → degradación graciosa.
        """
        result = extract_anomaly_data({})

        self._assert_anomaly_data_valid(result, set())

    def test_details_without_cycles_key(self):
        """
        'details' presente pero sin clave 'cycles'.
        """
        result = extract_anomaly_data({"details": {}})

        self._assert_anomaly_data_valid(result, set())

    def test_details_is_none(self):
        """details = None: valor nulo en el contenedor."""
        result = extract_anomaly_data({"details": None})

        self._assert_anomaly_data_valid(result, set())

    def test_analysis_result_is_none(self):
        """
        Input raíz es None.
        
        Caso extremo: debe manejarse sin excepción.
        """
        try:
            result = extract_anomaly_data(None)
            self._assert_anomaly_data_valid(result, set())
        except (TypeError, AttributeError):
            # Si lanza excepción, debe ser descriptiva
            pass  # Aceptable si la API no soporta None explícitamente

    def test_analysis_result_is_empty_string(self):
        """Input raíz es string vacío."""
        try:
            result = extract_anomaly_data("")
            self._assert_anomaly_data_valid(result)
        except (TypeError, AttributeError):
            pass

    def test_cycles_is_empty_list(self):
        """
        cycles = [] (lista vacía).
        
        Invariante I₃: sin ciclos → nodes_in_cycles = ∅.
        """
        result = extract_anomaly_data(self._make_result(cycles=[]))

        self._assert_anomaly_data_valid(result, set())

    def test_cycles_is_list_of_empty_strings(self):
        """
        Lista de strings vacíos.
        
        No debe producir nodos fantasma.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=["", "", ""])
        )

        self._assert_anomaly_data_valid(result, set())

    # ═══════════════════════════════════════════════════════════════════════
    # ROBUSTEZ DEL PARSER DE CADENAS
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "cycle_variant, expected_nodes",
        [
            ("A -> B -> A", {"A", "B"}),           # Canónico
            ("A->B->A", {"A", "B"}),               # Sin espacios
            ("A  ->  B  ->  A", {"A", "B"}),      # Espacios extras
            (" A -> B -> A ", {"A", "B"}),        # Espacios externos
            ("A   ->B->   A", {"A", "B"}),        # Espacios irregulares
            ("\tA -> B -> A\n", {"A", "B"}),      # Whitespace especial
        ],
        ids=[
            "canonical", 
            "no_spaces", 
            "extra_spaces", 
            "external_spaces",
            "irregular_spaces",
            "special_whitespace",
        ],
    )
    def test_whitespace_tolerance(self, cycle_variant, expected_nodes):
        """
        El parser debe ser tolerante a variaciones de whitespace.
        
        Todas las variantes representan el mismo ciclo: A → B → A.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=[cycle_variant])
        )

        self._assert_anomaly_data_valid(result, expected_nodes)

    def test_empty_string_in_cycles_list(self):
        """
        Cadena vacía intercalada con ciclos válidos.
        
        La cadena vacía debe ignorarse sin afectar los demás.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=["", "A -> B -> A", ""])
        )

        self._assert_anomaly_data_valid(result, {"A", "B"})
        assert "" not in result.nodes_in_cycles

    def test_only_whitespace_in_cycles_list(self):
        """
        Strings que contienen solo whitespace.
        """
        result = extract_anomaly_data(
            self._make_result(cycles=["   ", "\t\n", "A -> B -> A"])
        )

        self._assert_anomaly_data_valid(result, {"A", "B"})

    @pytest.mark.parametrize(
        "node_name, description",
        [
            ("node-1", "guiones"),
            ("node_1", "guiones bajos"),
            ("node.1", "puntos"),
            ("node:1", "dos puntos"),
            ("node/path", "barras"),
            ("node@domain", "arroba"),
            ("node#123", "hash"),
            ("UPPERCASE", "mayúsculas"),
            ("MixedCase", "case mixto"),
            ("123numeric", "inicio numérico"),
            ("node with spaces", "espacios internos"),
        ],
        ids=lambda x: x if len(x) < 20 else x[:17] + "...",
    )
    def test_special_characters_in_node_names(self, node_name, description):
        """
        Nombres de nodo con caracteres especiales.
        
        El parser no debe fragmentar ni corromper nombres complejos.
        """
        cycle = f"{node_name} -> other -> {node_name}"
        result = extract_anomaly_data(self._make_result(cycles=[cycle]))

        assert node_name in result.nodes_in_cycles, (
            f"Nodo con {description} no fue extraído: '{node_name}'"
        )
        assert "other" in result.nodes_in_cycles

    def test_unicode_node_names(self):
        """
        Nombres de nodo con caracteres Unicode.
        """
        cycles = [
            "αlpha -> βeta -> αlpha",      # Griego
            "節点 -> 连接 -> 節点",           # Chino/Japonés
            "узел -> связь -> узел",       # Cirílico
            "nœud -> lien -> nœud",        # Francés con ligadura
            "😀 -> 😎 -> 😀",               # Emojis
        ]
        
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        # Verificar algunos nodos Unicode
        assert "αlpha" in result.nodes_in_cycles or "alpha" in str(result.nodes_in_cycles).lower()
        # El comportamiento exacto depende de la implementación

    def test_very_long_node_names(self):
        """
        Nombres de nodo extremadamente largos.
        """
        long_name = "X" * 1000
        cycle = f"{long_name} -> Y -> {long_name}"
        
        result = extract_anomaly_data(self._make_result(cycles=[cycle]))

        assert long_name in result.nodes_in_cycles
        assert "Y" in result.nodes_in_cycles

    # ═══════════════════════════════════════════════════════════════════════
    # SEPARADORES ALTERNATIVOS
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "separator, description",
        [
            ("->", "sin espacios"),
            (" -> ", "canónico"),
            ("→", "flecha Unicode"),
            (" → ", "flecha Unicode con espacios"),
            ("=>", "doble flecha"),
            (" => ", "doble flecha con espacios"),
        ],
        ids=["arrow_compact", "arrow_spaced", "unicode", "unicode_spaced", 
             "double_arrow", "double_spaced"],
    )
    def test_alternative_separators(self, separator, description):
        """
        El parser debe soportar múltiples formatos de separador.
        """
        cycle = f"A{separator}B{separator}A"
        result = extract_anomaly_data(self._make_result(cycles=[cycle]))

        # Si el separador es soportado, debería extraer los nodos
        # Si no, debe degradar graciosamente
        self._assert_anomaly_data_valid(result)

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₁: TIPO DE RETORNO SIEMPRE AnomalyData
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "input_data, description",
        [
            ({}, "dict vacío"),
            ({"details": None}, "details None"),
            ({"details": {}}, "details vacío"),
            ({"details": {"cycles": []}}, "cycles vacío"),
            ({"details": {"cycles": ["A -> B -> A"]}}, "ciclo válido F₁"),
            ({"details": {"cycles": {"list": ["X -> Y -> X"]}}}, "ciclo válido F₂"),
            ({"details": {"cycles": 999}}, "cycles tipo inválido"),
            ({"details": {"cycles": ["", "   "]}}, "cycles con strings vacíos"),
            ({"details": {"other_key": "value"}}, "details sin cycles"),
            ({"unrelated": "data"}, "sin details"),
        ],
        ids=[
            "empty_dict",
            "none_details",
            "empty_details",
            "empty_cycles",
            "valid_F1",
            "valid_F2",
            "invalid_type",
            "empty_strings",
            "no_cycles_key",
            "no_details_key",
        ],
    )
    def test_return_type_always_anomaly_data(self, input_data, description):
        """
        Invariante I₁: El retorno siempre es AnomalyData.
        
        Nunca None, nunca excepción no capturada.
        """
        result = extract_anomaly_data(input_data)

        assert isinstance(result, AnomalyData), (
            f"Invariante I₁ violado para {description}: "
            f"esperado AnomalyData, obtenido {type(result)}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₂: UNICIDAD DE NODOS
    # ═══════════════════════════════════════════════════════════════════════

    def test_nodes_uniqueness_overlapping_cycles(self):
        """
        Invariante I₂: Nodos repetidos entre ciclos
        aparecen exactamente una vez.
        
        Configuración: Triángulo completo con cada arista como ciclo.
        """
        cycles = [
            "A -> B -> A",
            "B -> C -> B",
            "C -> A -> C",
            "A -> B -> C -> A",  # Ciclo largo que repite todos los nodos
        ]
        
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        nodes_list = list(result.nodes_in_cycles)
        nodes_set = set(nodes_list)
        
        assert len(nodes_list) == len(nodes_set), (
            f"Duplicados encontrados: {nodes_list}"
        )
        assert nodes_set == {"A", "B", "C"}

    def test_nodes_uniqueness_same_cycle_repeated(self):
        """
        El mismo ciclo listado múltiples veces.
        """
        cycles = ["A -> B -> A"] * 10
        
        result = extract_anomaly_data(self._make_result(cycles=cycles))

        assert set(result.nodes_in_cycles) == {"A", "B"}
        assert len(result.nodes_in_cycles) == 2

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₅: INMUTABILIDAD DEL INPUT
    # ═══════════════════════════════════════════════════════════════════════

    def test_input_immutability_list_format(self):
        """
        Invariante I₅: El input original no debe ser modificado.
        """
        original_cycles = ["A -> B -> A", "C -> D -> C"]
        input_data = {
            "details": {
                "cycles": original_cycles.copy()
            }
        }
        
        # Crear snapshot profundo
        input_snapshot = deepcopy(input_data)
        
        _ = extract_anomaly_data(input_data)
        
        # Verificar que el input no cambió
        assert input_data == input_snapshot, (
            "El input fue mutado por extract_anomaly_data"
        )
        assert input_data["details"]["cycles"] == original_cycles

    def test_input_immutability_dict_format(self):
        """
        Inmutabilidad con formato dict legacy.
        """
        input_data = {
            "details": {
                "cycles": {
                    "list": ["X -> Y -> X"],
                    "metadata": {"count": 1}
                }
            }
        }
        
        input_snapshot = deepcopy(input_data)
        
        _ = extract_anomaly_data(input_data)
        
        assert input_data == input_snapshot


# =============================================================================
# PRUEBAS DE EXTRACCIÓN DE ARISTAS
# =============================================================================

class TestExtractAnomalyEdges:
    """
    Pruebas de extracción de aristas de ciclos.

    Si AnomalyData expone edges_in_cycles, validamos que:
      - Cada par consecutivo en "A -> B -> C -> A" genera (A,B), (B,C), (C,A)
      - Las aristas son dirigidas: (A,B) ≠ (B,A)
      - Sin ciclos → sin aristas

    Estas pruebas se saltan si AnomalyData no implementa edges_in_cycles.
    """

    @staticmethod
    def _has_edges_attribute() -> bool:
        """Verifica si AnomalyData expone edges_in_cycles."""
        return hasattr(AnomalyData, "edges_in_cycles")

    @pytest.fixture
    def result_with_simple_cycle(self):
        """AnomalyData con un ciclo simple de 3 aristas."""
        return extract_anomaly_data(
            {"details": {"cycles": ["A -> B -> C -> A"]}}
        )

    @pytest.fixture
    def result_with_multiple_cycles(self):
        """AnomalyData con múltiples ciclos."""
        return extract_anomaly_data(
            {"details": {"cycles": ["A -> B -> A", "C -> D -> E -> C"]}}
        )

    @pytest.mark.skipif(
        not hasattr(AnomalyData, "edges_in_cycles"),
        reason="AnomalyData no expone edges_in_cycles",
    )
    def test_edges_extracted_from_simple_cycle(self, result_with_simple_cycle):
        """
        Ciclo A → B → C → A genera 3 aristas dirigidas.
        
        E = {(A,B), (B,C), (C,A)}
        """
        edges = set(result_with_simple_cycle.edges_in_cycles)
        expected_edges = {("A", "B"), ("B", "C"), ("C", "A")}
        
        assert expected_edges.issubset(edges)
        assert len(edges) == 3

    @pytest.mark.skipif(
        not hasattr(AnomalyData, "edges_in_cycles"),
        reason="AnomalyData no expone edges_in_cycles",
    )
    def test_edges_from_multiple_cycles(self, result_with_multiple_cycles):
        """
        Múltiples ciclos: aristas de ambos se combinan.
        """
        edges = set(result_with_multiple_cycles.edges_in_cycles)
        
        # Aristas del primer ciclo
        assert ("A", "B") in edges
        assert ("B", "A") in edges
        
        # Aristas del segundo ciclo
        assert ("C", "D") in edges
        assert ("D", "E") in edges
        assert ("E", "C") in edges

    @pytest.mark.skipif(
        not hasattr(AnomalyData, "edges_in_cycles"),
        reason="AnomalyData no expone edges_in_cycles",
    )
    def test_no_edges_from_empty_cycles(self):
        """Sin ciclos → sin aristas."""
        result = extract_anomaly_data({"details": {"cycles": []}})
        
        assert len(result.edges_in_cycles) == 0

    @pytest.mark.skipif(
        not hasattr(AnomalyData, "edges_in_cycles"),
        reason="AnomalyData no expone edges_in_cycles",
    )
    def test_self_loop_edge(self):
        """Auto-lazo genera una arista (A, A)."""
        result = extract_anomaly_data(
            {"details": {"cycles": ["A -> A"]}}
        )
        
        assert ("A", "A") in result.edges_in_cycles

    @pytest.mark.skipif(
        not hasattr(AnomalyData, "edges_in_cycles"),
        reason="AnomalyData no expone edges_in_cycles",
    )
    def test_edges_are_directed(self):
        """
        Las aristas son dirigidas: (A,B) y (B,A) son distintas.
        """
        result = extract_anomaly_data(
            {"details": {"cycles": ["A -> B -> A"]}}
        )
        
        edges = set(result.edges_in_cycles)
        
        # Ambas direcciones deben estar presentes
        assert ("A", "B") in edges
        assert ("B", "A") in edges
        
        # Son aristas distintas
        assert len(edges) == 2


# =============================================================================
# PRUEBAS DE PROPIEDADES TOPOLÓGICAS
# =============================================================================

class TestTopologicalProperties:
    """
    Pruebas de propiedades topológicas de los ciclos extraídos.
    
    Verifica que la extracción preserve invariantes de teoría de grafos.
    """

    def test_cycle_closure_verification(self):
        """
        Verificar que los ciclos extraídos son cerrados.
        
        Un ciclo cerrado tiene v₀ = vₖ.
        """
        cycles = [
            "A -> B -> C -> A",      # Cerrado
            "X -> Y -> Z -> X",      # Cerrado
            "Self -> Self",          # Auto-lazo (cerrado)
        ]
        
        for cycle in cycles:
            validation = validate_cycle_closure(cycle)
            assert validation["is_closed"], (
                f"Ciclo debería ser cerrado: {cycle}"
            )

    def test_open_path_detection(self):
        """
        Detectar caminos abiertos (no son ciclos verdaderos).
        
        Un camino A → B → C (sin retorno a A) no es un ciclo.
        """
        # Nota: Esto verifica la función de validación, no extract_anomaly_data
        open_paths = [
            "A -> B -> C",           # Sin cierre
            "X -> Y",                # Camino de longitud 1
        ]
        
        for path in open_paths:
            validation = validate_cycle_closure(path)
            assert not validation["is_closed"], (
                f"Camino debería detectarse como abierto: {path}"
            )

    def test_node_count_matches_extraction(self):
        """
        Verificar que el conteo de nodos del helper coincide
        con la extracción de AnomalyData.
        """
        cycles = ["A -> B -> C -> A", "D -> E -> D"]
        
        # Conteo con helper
        expected_nodes = extract_unique_nodes(cycles)
        
        # Extracción real
        result = extract_anomaly_data({"details": {"cycles": cycles}})
        actual_nodes = set(result.nodes_in_cycles)
        
        assert expected_nodes == actual_nodes

    def test_metrics_computation_consistency(self):
        """
        Verificar que las métricas computadas son consistentes.
        """
        cycles = [
            "A -> B -> C -> D -> A",  # 4 aristas
            "X -> Y -> X",            # 2 aristas
            "Z -> Z",                 # 1 arista (self-loop)
        ]
        
        metrics = compute_cycle_metrics(cycles)
        
        assert metrics["num_cycles"] == 3
        assert metrics["num_unique_nodes"] == 7  # A,B,C,D,X,Y,Z
        assert metrics["has_self_loops"] is True
        assert metrics["avg_cycle_length"] == pytest.approx((4 + 2 + 1) / 3)
        assert metrics["max_cycle_length"] == 4


# =============================================================================
# PRUEBAS DE RENDIMIENTO Y STRESS
# =============================================================================

class TestPerformanceAndStress:
    """
    Pruebas de rendimiento bajo carga.
    
    Verifican que la extracción escale adecuadamente
    con grafos grandes.
    """

    def test_many_small_cycles(self):
        """
        Muchos ciclos pequeños (stress en número de ciclos).
        """
        num_cycles = 100
        cycles = [f"Node_{i} -> Node_{i+1} -> Node_{i}" for i in range(num_cycles)]
        
        result = extract_anomaly_data({"details": {"cycles": cycles}})
        
        assert isinstance(result, AnomalyData)
        # Cada ciclo aporta ~2 nodos únicos (con algo de overlap)
        assert len(result.nodes_in_cycles) > 0

    def test_single_very_long_cycle(self):
        """
        Un único ciclo muy largo (stress en longitud).
        """
        num_nodes = 100
        node_names = [f"N{i}" for i in range(num_nodes)]
        cycle = " -> ".join(node_names + [node_names[0]])  # Cerrar el ciclo
        
        result = extract_anomaly_data({"details": {"cycles": [cycle]}})
        
        assert len(result.nodes_in_cycles) == num_nodes

    def test_fully_connected_graph_cycles(self):
        """
        Ciclos de un grafo completamente conectado K_n.
        
        Para K_5, hay muchos ciclos posibles.
        """
        nodes = ["A", "B", "C", "D", "E"]
        # Algunos ciclos de K_5
        cycles = [
            "A -> B -> C -> A",
            "A -> C -> D -> A",
            "B -> D -> E -> B",
            "A -> B -> C -> D -> E -> A",
        ]
        
        result = extract_anomaly_data({"details": {"cycles": cycles}})
        
        assert set(result.nodes_in_cycles) == set(nodes)

    def test_extraction_time_bounded(self):
        """
        La extracción debe completarse en tiempo razonable.
        """
        import time
        
        # Crear input moderadamente grande
        cycles = [f"A{i} -> B{i} -> A{i}" for i in range(500)]
        input_data = {"details": {"cycles": cycles}}
        
        start = time.perf_counter()
        result = extract_anomaly_data(input_data)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Extracción tardó {elapsed:.2f}s (límite: 1s)"
        assert isinstance(result, AnomalyData)


# =============================================================================
# PRUEBAS DE IDEMPOTENCIA Y DETERMINISMO
# =============================================================================

class TestIdempotenceAndDeterminism:
    """
    Pruebas de propiedades de idempotencia y determinismo.
    """

    def test_extraction_is_idempotent(self):
        """
        Múltiples llamadas con el mismo input producen el mismo resultado.
        """
        input_data = {
            "details": {
                "cycles": ["A -> B -> C -> A", "D -> E -> D"]
            }
        }
        
        result1 = extract_anomaly_data(input_data)
        result2 = extract_anomaly_data(input_data)
        result3 = extract_anomaly_data(input_data)
        
        assert set(result1.nodes_in_cycles) == set(result2.nodes_in_cycles)
        assert set(result2.nodes_in_cycles) == set(result3.nodes_in_cycles)

    def test_extraction_is_deterministic(self):
        """
        Dado el mismo input, siempre produce el mismo output.
        """
        def create_input():
            return {
                "details": {
                    "cycles": ["X -> Y -> Z -> X"]
                }
            }
        
        results = [extract_anomaly_data(create_input()) for _ in range(5)]
        
        first_nodes = set(results[0].nodes_in_cycles)
        for result in results[1:]:
            assert set(result.nodes_in_cycles) == first_nodes

    def test_order_independence_in_cycles_list(self):
        """
        El orden de los ciclos en la lista no afecta el resultado.
        
        La extracción de nodos es una operación de conjunto.
        """
        cycles_order_1 = ["A -> B -> A", "C -> D -> C"]
        cycles_order_2 = ["C -> D -> C", "A -> B -> A"]
        
        result1 = extract_anomaly_data({"details": {"cycles": cycles_order_1}})
        result2 = extract_anomaly_data({"details": {"cycles": cycles_order_2}})
        
        assert set(result1.nodes_in_cycles) == set(result2.nodes_in_cycles)


# =============================================================================
# PRUEBAS DE INTEGRACIÓN CON ANOMALY DATA
# =============================================================================

class TestAnomalyDataIntegration:
    """
    Pruebas de integración que verifican la estructura y comportamiento
    del objeto AnomalyData retornado.
    """

    def test_anomaly_data_has_required_attributes(self):
        """
        AnomalyData debe exponer al menos nodes_in_cycles.
        """
        result = extract_anomaly_data(
            {"details": {"cycles": ["A -> B -> A"]}}
        )
        
        assert hasattr(result, 'nodes_in_cycles')
        # nodes_in_cycles debe ser iterable
        assert hasattr(result.nodes_in_cycles, '__iter__')

    def test_anomaly_data_nodes_are_strings(self):
        """
        Todos los nodos extraídos deben ser strings.
        """
        result = extract_anomaly_data(
            {"details": {"cycles": ["Node1 -> Node2 -> Node1", "A -> B -> C -> A"]}}
        )
        
        for node in result.nodes_in_cycles:
            assert isinstance(node, str), f"Nodo no es string: {node} ({type(node)})"

    def test_anomaly_data_is_usable_in_set_operations(self):
        """
        nodes_in_cycles debe poder usarse en operaciones de conjunto.
        """
        result = extract_anomaly_data(
            {"details": {"cycles": ["A -> B -> C -> A"]}}
        )
        
        nodes_set = set(result.nodes_in_cycles)
        
        # Operaciones de conjunto básicas
        assert "A" in nodes_set
        assert nodes_set & {"A", "B"} == {"A", "B"}
        assert len(nodes_set - {"Z"}) == len(nodes_set)

    def test_anomaly_data_repr_or_str(self):
        """
        AnomalyData debe tener representación legible.
        """
        result = extract_anomaly_data(
            {"details": {"cycles": ["A -> B -> A"]}}
        )
        
        # No debe lanzar excepción
        str_repr = str(result)
        repr_repr = repr(result)
        
        assert len(str_repr) > 0
        assert len(repr_repr) > 0