import pytest
import threading
import time
from typing import Any, Dict, List, Optional, Set
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from enum import Enum
from collections import deque

from app.core.telemetry import TelemetryContext
from app.core.telemetry_narrative import TelemetryNarrator


# =============================================================================
# CONSTANTES Y ENUMERACIONES DE DOMINIO
# =============================================================================

class VerdictCode(str, Enum):
    """
    Códigos de veredicto del sistema de telemetría.
    
    Semántica:
      - APPROVED: Ejecución exitosa sin anomalías
      - REJECTED_PHYSICS: Fallo por violación de invariantes físicos/lógicos
      - REJECTED_TIMEOUT: Fallo por exceder límites temporales
      - REJECTED_RESOURCE: Fallo por agotamiento de recursos
    """
    APPROVED = "APPROVED"
    REJECTED_PHYSICS = "REJECTED_PHYSICS"
    REJECTED_TIMEOUT = "REJECTED_TIMEOUT"
    REJECTED_RESOURCE = "REJECTED_RESOURCE"


class SpanStatus(str, Enum):
    """
    Estados posibles de un span en la jerarquía.
    
    Ordenamiento de severidad: OK < WARNING < CRITICO
    """
    OK = "OK"
    WARNING = "WARNING"
    CRITICO = "CRITICO"
    
    @classmethod
    def max_severity(cls, *statuses: "SpanStatus") -> "SpanStatus":
        """Retorna el status de mayor severidad."""
        severity_order = [cls.OK, cls.WARNING, cls.CRITICO]
        max_idx = max(severity_order.index(s) for s in statuses if s in severity_order)
        return severity_order[max_idx]


class ReportKeys:
    """Claves obligatorias del reporte de telemetría."""
    VERDICT_CODE = "verdict_code"
    PHASES = "phases"
    FORENSIC_EVIDENCE = "forensic_evidence"
    NARRATIVE = "narrative"
    CAUSALITY_CHAIN = "causality_chain"
    
    @classmethod
    def required_keys(cls) -> Set[str]:
        return {cls.VERDICT_CODE, cls.PHASES, cls.FORENSIC_EVIDENCE, cls.NARRATIVE}


# =============================================================================
# HELPERS DE VALIDACIÓN DE ESTRUCTURA DE ÁRBOL
# =============================================================================

def validate_tree_structure(root_spans: List[Any]) -> Dict[str, Any]:
    """
    Valida que los spans formen un bosque de árboles bien formado.
    
    Invariantes verificados:
      (T₁) Cada nodo tiene exactamente un padre (excepto raíces)
      (T₂) No existen ciclos en el grafo
      (T₃) Todas las duraciones son no negativas
      (T₄) duration(padre) ≥ max(duration(hijo)) para todo hijo
      
    Args:
        root_spans: Lista de spans raíz del bosque
        
    Returns:
        Diccionario con métricas de validación:
          - is_valid: bool
          - total_nodes: int
          - max_depth: int
          - violations: List[str]
    """
    violations = []
    total_nodes = 0
    max_depth = 0
    visited_ids = set()
    
    def traverse(span, depth: int, parent_duration: Optional[float] = None):
        nonlocal total_nodes, max_depth
        
        span_id = id(span)
        
        # (T₂) Detección de ciclos
        if span_id in visited_ids:
            violations.append(f"Ciclo detectado en span '{span.name}'")
            return
        visited_ids.add(span_id)
        
        total_nodes += 1
        max_depth = max(max_depth, depth)
        
        # (T₃) Duración no negativa
        if hasattr(span, 'duration') and span.duration is not None:
            if span.duration < 0:
                violations.append(
                    f"Duración negativa en '{span.name}': {span.duration}"
                )
            
            # (T₄) Monotonía temporal padre ≥ hijo
            if parent_duration is not None and span.duration > parent_duration:
                violations.append(
                    f"Violación de monotonía: hijo '{span.name}' "
                    f"(d={span.duration}) > padre (d={parent_duration})"
                )
        
        # Recursión sobre hijos
        for child in getattr(span, 'children', []):
            traverse(child, depth + 1, getattr(span, 'duration', None))
    
    for root in root_spans:
        traverse(root, depth=1)
    
    return {
        "is_valid": len(violations) == 0,
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "num_trees": len(root_spans),
        "violations": violations,
    }


def compute_tree_metrics(root_spans: List[Any]) -> Dict[str, Any]:
    """
    Computa métricas estructurales del bosque de spans.
    
    Métricas:
      - branching_factor: Factor de ramificación promedio
      - leaf_count: Número de hojas (spans sin hijos)
      - internal_count: Número de nodos internos
      - depth_distribution: Histograma de profundidades
    """
    leaf_count = 0
    internal_count = 0
    total_children = 0
    depth_distribution: Dict[int, int] = {}
    
    def traverse(span, depth: int):
        nonlocal leaf_count, internal_count, total_children
        
        depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        children = getattr(span, 'children', [])
        
        if not children:
            leaf_count += 1
        else:
            internal_count += 1
            total_children += len(children)
            for child in children:
                traverse(child, depth + 1)
    
    for root in root_spans:
        traverse(root, depth=1)
    
    branching_factor = total_children / internal_count if internal_count > 0 else 0.0
    
    return {
        "leaf_count": leaf_count,
        "internal_count": internal_count,
        "branching_factor": branching_factor,
        "depth_distribution": depth_distribution,
    }


def find_span_by_name(root_spans: List[Any], name: str) -> Optional[Any]:
    """Búsqueda BFS de un span por nombre en el bosque."""
    queue = deque(root_spans)
    while queue:
        span = queue.popleft()
        if getattr(span, 'name', None) == name:
            return span
        queue.extend(getattr(span, 'children', []))
    return None


def collect_all_spans(root_spans: List[Any]) -> List[Any]:
    """Recolecta todos los spans del bosque en orden BFS."""
    result = []
    queue = deque(root_spans)
    while queue:
        span = queue.popleft()
        result.append(span)
        queue.extend(getattr(span, 'children', []))
    return result


# =============================================================================
# PRUEBAS DE ESTRUCTURA JERÁRQUICA DE SPANS
# =============================================================================

class TestSpanHierarchy:
    """
    Pruebas sobre la estructura de árbol enraizado que forman los spans.

    Modelo formal (Teoría de Grafos):
    
      Sea G = (V, E) el grafo dirigido donde:
        - V = conjunto de spans
        - E = {(p, c) | c es hijo directo de p}
      
      G debe satisfacer:
        (G₁) G es un bosque (colección de árboles disjuntos)
        (G₂) ∀v ∈ V \ {raíces}: ∃! p ∈ V tal que (p, v) ∈ E
        (G₃) G es acíclico (no existen caminos v → ... → v)
        
    Invariantes del modelo:
      ┌─────┬───────────────────────────────────────────────────────────────┐
      │ I₁  │ |root_spans| = k (número de árboles en el bosque)             │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₂  │ ∀ span s: duration(s) ≥ 0                                     │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₃  │ ∀ span s con hijos: duration(s) ≥ max(duration(cᵢ))          │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₄  │ Las métricas se registran en el span activo al momento       │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₅  │ El orden de inserción de hijos se preserva                   │
      └─────┴───────────────────────────────────────────────────────────────┘
    """

    @pytest.fixture
    def ctx(self) -> TelemetryContext:
        """Contexto de telemetría aislado por test."""
        return TelemetryContext()

    # ═══════════════════════════════════════════════════════════════════════
    # ESTRUCTURA BÁSICA: ÁRBOL MÍNIMO
    # ═══════════════════════════════════════════════════════════════════════

    def test_single_root_span(self, ctx):
        """
        Árbol trivial: único nodo (raíz sin hijos).
        
        Estructura: R
        |V| = 1, |E| = 0
        """
        with ctx.span("Solitary Root"):
            pass

        assert len(ctx.root_spans) == 1
        assert ctx.root_spans[0].name == "Solitary Root"
        assert ctx.root_spans[0].children == []
        
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"]
        assert validation["total_nodes"] == 1
        assert validation["max_depth"] == 1

    def test_single_root_with_one_child(self, ctx):
        """
        Árbol mínimo no trivial: raíz con un hijo.
        
        Estructura: R → C
        |V| = 2, |E| = 1
        
        Verifica nombres, parentesco y conteo.
        """
        with ctx.span("Root Phase") as root:
            with ctx.span("Child Operation") as child:
                child.metrics["processed"] = 100

        assert len(ctx.root_spans) == 1
        assert ctx.root_spans[0].name == "Root Phase"
        assert len(ctx.root_spans[0].children) == 1
        assert ctx.root_spans[0].children[0].name == "Child Operation"
        assert ctx.root_spans[0].children[0].metrics["processed"] == 100
        
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"]
        assert validation["total_nodes"] == 2

    def test_root_with_multiple_children(self, ctx):
        """
        Árbol de profundidad 2 con múltiples hijos.
        
        Estructura:
              R
            / | \
           A  B  C
           
        Factor de ramificación = 3
        """
        with ctx.span("Root"):
            with ctx.span("Child_A"):
                pass
            with ctx.span("Child_B"):
                pass
            with ctx.span("Child_C"):
                pass

        root = ctx.root_spans[0]
        assert len(root.children) == 3
        
        metrics = compute_tree_metrics(ctx.root_spans)
        assert metrics["branching_factor"] == 3.0
        assert metrics["leaf_count"] == 3
        assert metrics["internal_count"] == 1

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₂: DURACIÓN NO NEGATIVA
    # ═══════════════════════════════════════════════════════════════════════

    def test_duration_non_negative_simple(self, ctx):
        """
        Invariante I₂: toda duración es no negativa.
        
        Para cualquier span s: duration(s) ∈ [0, ∞)
        """
        with ctx.span("Root"):
            with ctx.span("Child"):
                pass

        root = ctx.root_spans[0]
        assert root.duration >= 0, f"Duración raíz negativa: {root.duration}"
        assert root.children[0].duration >= 0, (
            f"Duración hijo negativa: {root.children[0].duration}"
        )

    def test_duration_non_negative_deep_tree(self, ctx):
        """
        Invariante I₂ en árbol profundo: todas las duraciones ≥ 0.
        """
        with ctx.span("Level_1"):
            with ctx.span("Level_2"):
                with ctx.span("Level_3"):
                    with ctx.span("Level_4"):
                        pass

        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"], (
            f"Violaciones encontradas: {validation['violations']}"
        )

    def test_zero_duration_span_allowed(self, ctx):
        """
        Duración cero es válida (operación instantánea).
        
        En el límite: lim(Δt→0) duration = 0 es aceptable.
        """
        with ctx.span("Instant Operation"):
            pass  # Operación sin trabajo

        root = ctx.root_spans[0]
        # duration puede ser 0 o muy pequeño, pero no negativo
        assert root.duration >= 0

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₃: MONOTONÍA TEMPORAL
    # ═══════════════════════════════════════════════════════════════════════

    def test_parent_duration_geq_child_duration(self, ctx):
        """
        Invariante I₃ (monotonía temporal):
        
        El span padre abarca temporalmente al hijo, por lo tanto:
          duration(padre) ≥ duration(hijo)
          
        Esto se deriva del hecho de que:
          - start(padre) ≤ start(hijo)
          - end(hijo) ≤ end(padre)
        """
        with ctx.span("Parent"):
            with ctx.span("Child"):
                time.sleep(0.01)  # Trabajo en el hijo

        parent = ctx.root_spans[0]
        child = parent.children[0]

        assert parent.duration >= child.duration, (
            f"Violación de monotonía: "
            f"padre ({parent.duration}) < hijo ({child.duration})"
        )

    def test_parent_duration_geq_max_children(self, ctx):
        """
        Invariante I₃ generalizado:
          duration(padre) ≥ max({duration(c) | c ∈ children})
        """
        with ctx.span("Parent") as parent_span:
            with ctx.span("Short_Child"):
                time.sleep(0.005)
            with ctx.span("Long_Child"):
                time.sleep(0.02)
            with ctx.span("Medium_Child"):
                time.sleep(0.01)

        parent = ctx.root_spans[0]
        max_child_duration = max(c.duration for c in parent.children)

        assert parent.duration >= max_child_duration, (
            f"padre ({parent.duration}) < max_hijo ({max_child_duration})"
        )

    def test_monotonicity_in_deep_chain(self, ctx):
        """
        Monotonía en cadena: duration(L₁) ≥ duration(L₂) ≥ ... ≥ duration(Lₙ)
        """
        depth = 5
        
        def build_chain(level):
            if level > depth:
                return
            with ctx.span(f"Level_{level}"):
                time.sleep(0.002 * (depth - level + 1))  # Trabajo decreciente
                build_chain(level + 1)
        
        build_chain(1)
        
        # Verificar monotonía descendente
        current = ctx.root_spans[0]
        prev_duration = current.duration
        
        while current.children:
            child = current.children[0]
            assert prev_duration >= child.duration, (
                f"Violación en cadena: {current.name} ({prev_duration}) < "
                f"{child.name} ({child.duration})"
            )
            prev_duration = child.duration
            current = child

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₅: ORDEN DE INSERCIÓN PRESERVADO
    # ═══════════════════════════════════════════════════════════════════════

    def test_siblings_preserve_insertion_order(self, ctx):
        """
        Invariante I₅: el orden de inserción de hermanos se preserva.
        
        Si spans A, B, C se crean en ese orden, entonces:
          children[0] = A, children[1] = B, children[2] = C
        """
        sibling_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]

        with ctx.span("Root"):
            for name in sibling_names:
                with ctx.span(name):
                    pass

        root = ctx.root_spans[0]

        assert len(root.children) == len(sibling_names)
        for i, expected_name in enumerate(sibling_names):
            actual_name = root.children[i].name
            assert actual_name == expected_name, (
                f"Orden incorrecto en posición {i}: "
                f"esperado '{expected_name}', obtenido '{actual_name}'"
            )

    @pytest.mark.parametrize(
        "num_siblings",
        [1, 2, 5, 10, 50],
        ids=lambda n: f"n={n}",
    )
    def test_sibling_order_parametric(self, ctx, num_siblings):
        """Orden de inserción preservado para N hermanos."""
        names = [f"Sibling_{i}" for i in range(num_siblings)]

        with ctx.span("Root"):
            for name in names:
                with ctx.span(name):
                    pass

        root = ctx.root_spans[0]
        assert len(root.children) == num_siblings
        
        for i, expected in enumerate(names):
            assert root.children[i].name == expected

    # ═══════════════════════════════════════════════════════════════════════
    # TOPOLOGÍA: PROFUNDIDAD ARBITRARIA
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "depth", 
        [1, 3, 5, 10, 20],
        ids=lambda d: f"depth_{d}"
    )
    def test_nested_depth_chain(self, ctx, depth):
        """
        Cadena lineal de profundidad variable: R → C₁ → C₂ → … → Cₙ.
        
        Verifica que la estructura se preserve a profundidad arbitraria.
        Este es un "path graph" P_n en terminología de grafos.
        """
        def build_chain(level):
            if level > depth:
                return
            with ctx.span(f"Level_{level}"):
                build_chain(level + 1)

        build_chain(1)

        # Validar estructura de árbol
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"], f"Violaciones: {validation['violations']}"
        assert validation["max_depth"] == depth
        assert validation["total_nodes"] == depth

        # Navegar desde la raíz hasta la hoja
        current = ctx.root_spans[0]
        for level in range(1, depth + 1):
            assert current.name == f"Level_{level}"
            if level < depth:
                assert len(current.children) == 1, (
                    f"Nivel {level} debe tener exactamente 1 hijo"
                )
                current = current.children[0]
            else:
                assert len(current.children) == 0, (
                    f"Nivel {depth} (hoja) no debe tener hijos"
                )

    def test_deep_recursion_stress(self, ctx):
        """
        Stress test: profundidad extrema para verificar que no hay
        desbordamiento de pila en la implementación.
        """
        max_depth = 100  # Ajustar según límites de recursión
        
        def build_deep_chain(level):
            if level > max_depth:
                return
            with ctx.span(f"Deep_{level}"):
                build_deep_chain(level + 1)
        
        # No debe lanzar RecursionError
        build_deep_chain(1)
        
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["max_depth"] == max_depth

    # ═══════════════════════════════════════════════════════════════════════
    # TOPOLOGÍA: BOSQUE (MÚLTIPLES RAÍCES)
    # ═══════════════════════════════════════════════════════════════════════

    def test_multiple_root_spans_form_forest(self, ctx):
        """
        Invariante I₁: k raíces independientes forman un bosque.
        
        Un bosque F es una colección de árboles disjuntos:
          F = T₁ ∪ T₂ ∪ ... ∪ Tₖ  donde Tᵢ ∩ Tⱼ = ∅ para i ≠ j
        """
        root_names = ["Pipeline_A", "Pipeline_B", "Pipeline_C"]

        for name in root_names:
            with ctx.span(name):
                pass

        assert len(ctx.root_spans) == len(root_names)
        
        for i, expected_name in enumerate(root_names):
            assert ctx.root_spans[i].name == expected_name
            assert ctx.root_spans[i].children == []

        validation = validate_tree_structure(ctx.root_spans)
        assert validation["num_trees"] == len(root_names)

    def test_forest_with_varied_subtrees(self, ctx):
        """
        Bosque con árboles de diferentes tamaños y profundidades.
        
        T₁: profundidad 1 (solo raíz)
        T₂: profundidad 2 (raíz + 2 hijos)
        T₃: profundidad 3 (cadena lineal)
        """
        # T₁: Solo raíz
        with ctx.span("Tree_1_Root"):
            pass

        # T₂: Raíz con 2 hijos
        with ctx.span("Tree_2_Root"):
            with ctx.span("T2_Child_A"):
                pass
            with ctx.span("T2_Child_B"):
                pass

        # T₃: Cadena de profundidad 3
        with ctx.span("Tree_3_Root"):
            with ctx.span("T3_Level_2"):
                with ctx.span("T3_Level_3"):
                    pass

        assert len(ctx.root_spans) == 3
        
        # Verificar estructura de cada árbol
        t1 = ctx.root_spans[0]
        assert len(t1.children) == 0

        t2 = ctx.root_spans[1]
        assert len(t2.children) == 2

        t3 = ctx.root_spans[2]
        assert len(t3.children) == 1
        assert len(t3.children[0].children) == 1

    @pytest.mark.parametrize(
        "num_trees",
        [1, 5, 10, 50],
        ids=lambda n: f"trees={n}",
    )
    def test_forest_size_parametric(self, ctx, num_trees):
        """Bosque con N árboles independientes."""
        for i in range(num_trees):
            with ctx.span(f"Tree_{i}"):
                pass

        assert len(ctx.root_spans) == num_trees
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["num_trees"] == num_trees

    # ═══════════════════════════════════════════════════════════════════════
    # CASO DEGENERADO: CONTEXTO VACÍO
    # ═══════════════════════════════════════════════════════════════════════

    def test_empty_context_has_no_spans(self, ctx):
        """
        Bosque vacío: F = ∅.
        
        Un contexto sin spans registrados debe tener:
          - root_spans = []
          - Sin métricas globales (o métricas vacías)
        """
        assert len(ctx.root_spans) == 0
        
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"]
        assert validation["total_nodes"] == 0

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₄: MÉTRICAS EN SPAN ACTIVO
    # ═══════════════════════════════════════════════════════════════════════

    def test_metrics_attached_to_active_span(self, ctx):
        """
        Invariante I₄: las métricas se registran en el span
        que está activo al momento de la llamada.
        
        Principio de localidad: una métrica pertenece al contexto
        de ejecución donde se emite, no a contextos ancestrales.
        """
        with ctx.span("Outer") as outer:
            outer.metrics["outer_key"] = "outer_val"
            outer.metrics["shared_key"] = "from_outer"
            
            with ctx.span("Inner") as inner:
                inner.metrics["inner_key"] = "inner_val"
                inner.metrics["shared_key"] = "from_inner"  # Sobrescribe en Inner

        root = ctx.root_spans[0]
        inner_span = root.children[0]

        # Verificar aislamiento de métricas
        assert "outer_key" in root.metrics
        assert "inner_key" not in root.metrics
        
        assert "inner_key" in inner_span.metrics
        assert "outer_key" not in inner_span.metrics
        
        # Verificar independencia de claves compartidas
        assert root.metrics["shared_key"] == "from_outer"
        assert inner_span.metrics["shared_key"] == "from_inner"

    def test_metrics_isolation_between_siblings(self, ctx):
        """
        Métricas aisladas entre hermanos (siblings).
        
        Spans al mismo nivel no comparten métricas entre sí.
        """
        with ctx.span("Parent"):
            with ctx.span("Sibling_A") as a:
                a.metrics["a_metric"] = 100
            with ctx.span("Sibling_B") as b:
                b.metrics["b_metric"] = 200

        parent = ctx.root_spans[0]
        sibling_a = parent.children[0]
        sibling_b = parent.children[1]

        assert "a_metric" in sibling_a.metrics
        assert "b_metric" not in sibling_a.metrics
        
        assert "b_metric" in sibling_b.metrics
        assert "a_metric" not in sibling_b.metrics

    def test_metric_types_preserved(self, ctx):
        """
        Los tipos de valores de métricas se preservan correctamente.
        """
        with ctx.span("TypeTest") as span:
            span.metrics["int_val"] = 42
            span.metrics["float_val"] = 3.14159
            span.metrics["str_val"] = "hello"
            span.metrics["bool_val"] = True
            span.metrics["list_val"] = [1, 2, 3]
            span.metrics["dict_val"] = {"nested": "value"}
            span.metrics["none_val"] = None

        root = ctx.root_spans[0]
        
        assert root.metrics["int_val"] == 42
        assert isinstance(root.metrics["int_val"], int)
        
        assert root.metrics["float_val"] == pytest.approx(3.14159)
        assert isinstance(root.metrics["float_val"], float)
        
        assert root.metrics["str_val"] == "hello"
        assert root.metrics["bool_val"] is True
        assert root.metrics["list_val"] == [1, 2, 3]
        assert root.metrics["dict_val"] == {"nested": "value"}
        assert root.metrics["none_val"] is None

    # ═══════════════════════════════════════════════════════════════════════
    # AISLAMIENTO ENTRE CONTEXTOS
    # ═══════════════════════════════════════════════════════════════════════

    def test_contexts_are_isolated(self):
        """
        Dos TelemetryContext independientes no comparten estado.
        
        Cada contexto es un universo aislado; operaciones en uno
        no afectan al otro. Esto es crítico para:
          - Ejecución paralela de pipelines
          - Testing aislado
          - Multi-tenancy
        """
        ctx_a = TelemetryContext()
        ctx_b = TelemetryContext()

        with ctx_a.span("Span_A"):
            pass

        with ctx_b.span("Span_B"):
            pass

        # Verificar aislamiento
        assert len(ctx_a.root_spans) == 1
        assert ctx_a.root_spans[0].name == "Span_A"

        assert len(ctx_b.root_spans) == 1
        assert ctx_b.root_spans[0].name == "Span_B"
        
        # Verificar que no hay contaminación cruzada
        assert not any(s.name == "Span_B" for s in ctx_a.root_spans)
        assert not any(s.name == "Span_A" for s in ctx_b.root_spans)

    def test_concurrent_contexts_isolated(self):
        """
        Contextos usados concurrentemente permanecen aislados.
        """
        results = {}
        
        def worker(ctx_id: int):
            ctx = TelemetryContext()
            with ctx.span(f"Context_{ctx_id}_Root"):
                with ctx.span(f"Context_{ctx_id}_Child"):
                    time.sleep(0.01)  # Simular trabajo
            results[ctx_id] = ctx

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verificar que cada contexto tiene exactamente sus spans
        for ctx_id, ctx in results.items():
            assert len(ctx.root_spans) == 1
            assert ctx.root_spans[0].name == f"Context_{ctx_id}_Root"
            assert len(ctx.root_spans[0].children) == 1

    # ═══════════════════════════════════════════════════════════════════════
    # NOMBRES DE SPANS: CASOS ESPECIALES
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "span_name, description",
        [
            ("", "nombre vacío"),
            ("   ", "solo espacios"),
            ("Nombre con espacios", "espacios internos"),
            ("Nombre_con_guiones_bajos", "guiones bajos"),
            ("Nombre-con-guiones", "guiones medios"),
            ("Nombre.con.puntos", "puntos"),
            ("Nombre/con/barras", "barras"),
            ("Ñoño_con_tildes_áéíóú", "caracteres Unicode"),
            ("名前", "caracteres no-ASCII (japonés)"),
            ("A" * 1000, "nombre muy largo"),
            ("123_numeric_start", "inicia con número"),
        ],
        ids=lambda x: x if len(x) < 30 else x[:27] + "...",
    )
    def test_span_name_special_characters(self, ctx, span_name, description):
        """
        Los nombres de spans deben soportar caracteres especiales.
        """
        with ctx.span(span_name):
            pass

        assert len(ctx.root_spans) == 1
        assert ctx.root_spans[0].name == span_name

    def test_duplicate_span_names_allowed(self, ctx):
        """
        Spans con nombres duplicados son permitidos (nodos distintos).
        
        La identidad de un span no está determinada solo por su nombre,
        sino por su posición en el árbol.
        """
        with ctx.span("Duplicate"):
            with ctx.span("Duplicate"):
                pass
        with ctx.span("Duplicate"):
            pass

        assert len(ctx.root_spans) == 2
        assert ctx.root_spans[0].name == "Duplicate"
        assert ctx.root_spans[0].children[0].name == "Duplicate"
        assert ctx.root_spans[1].name == "Duplicate"


# =============================================================================
# PRUEBAS DE PROPAGACIÓN DE ERRORES Y NARRATIVA FORENSE
# =============================================================================

class TestErrorPropagationAndForensics:
    """
    Pruebas sobre la propagación de errores a través del árbol de spans
    y su captura en evidencia forense.

    Modelo de propagación (semántica de excepciones):
    
      Sea s un span donde ocurre un error ε.
      Sea path(s) = [r, n₁, n₂, ..., nₖ, s] el camino desde la raíz r hasta s.
      
      Propagación:
        ∀ nᵢ ∈ path(s): status(nᵢ) := max_severity(status(nᵢ), CRITICO)
        
      La evidencia forense E es un conjunto de registros:
        E = {(source, message, level) | error capturado en algún span}
        
    Invariantes de propagación:
      (P₁) Si error en hijo → status(padre) = CRITICO
      (P₂) |forensic_evidence| ≥ |errores distintos capturados|
      (P₃) Cada error tiene source = nombre del span donde ocurrió
    """

    @pytest.fixture
    def ctx(self) -> TelemetryContext:
        return TelemetryContext()

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    # ═══════════════════════════════════════════════════════════════════════
    # PROPAGACIÓN BÁSICA DE ERRORES
    # ═══════════════════════════════════════════════════════════════════════

    def test_error_in_child_marks_parent_critical(self, ctx, narrator):
        """
        Invariante P₁: Error en hijo → padre es CRITICO.
        
        Estructura:
          Data Loading (raíz)
            ├── Read CSV (ok)
            └── Validate Schema (ERROR)
            
        El error en 'Validate Schema' debe propagarse a 'Data Loading'.
        """
        try:
            with ctx.span("Data Loading"):
                with ctx.span("Read CSV"):
                    pass
                with ctx.span("Validate Schema"):
                    raise ValueError("Invalid Column: 'Price'")
        except ValueError:
            pass

        report = narrator.summarize_execution(ctx)

        assert report[ReportKeys.VERDICT_CODE] == VerdictCode.REJECTED_PHYSICS.value
        assert len(report[ReportKeys.PHASES]) == 1
        assert report[ReportKeys.PHASES][0]["name"] == "Data Loading"
        assert report[ReportKeys.PHASES][0]["status"] == SpanStatus.CRITICO.value

    def test_error_in_deeply_nested_span_propagates(self, ctx, narrator):
        """
        Error en profundidad N burbujea hasta la raíz.
        
        Cadena: Root → Level_1 → Level_2 → Level_3 → ERROR
        Todos los ancestros deben reflejar el error.
        """
        try:
            with ctx.span("Root"):
                with ctx.span("Level_1"):
                    with ctx.span("Level_2"):
                        with ctx.span("Level_3"):
                            raise IOError("Disk full")
        except IOError:
            pass

        report = narrator.summarize_execution(ctx)

        assert report[ReportKeys.PHASES][0]["status"] == SpanStatus.CRITICO.value
        
        # Verificar que el error se atribuye al span correcto
        evidence = report[ReportKeys.FORENSIC_EVIDENCE]
        assert any(e["source"] == "Level_3" for e in evidence)

    def test_error_does_not_affect_sibling_branches(self, ctx, narrator):
        """
        Error en una rama no afecta a ramas hermanas ya completadas.
        
        Estructura:
          Root
            ├── Branch_A (completo, ok)
            └── Branch_B (ERROR)
            
        Branch_A debe mantener su status independiente.
        """
        try:
            with ctx.span("Root"):
                with ctx.span("Branch_A") as a:
                    a.metrics["completed"] = True
                with ctx.span("Branch_B"):
                    raise RuntimeError("Failure in B")
        except RuntimeError:
            pass

        root = ctx.root_spans[0]
        branch_a = root.children[0]
        
        # Branch_A debe tener sus métricas intactas
        assert branch_a.metrics["completed"] is True

    # ═══════════════════════════════════════════════════════════════════════
    # EVIDENCIA FORENSE
    # ═══════════════════════════════════════════════════════════════════════

    def test_forensic_evidence_captures_error_source(self, ctx, narrator):
        """
        Invariante P₃: La evidencia forense identifica el span fuente.
        
        Campos requeridos por entrada de evidencia:
          - source: nombre del span donde ocurrió el error
          - message: mensaje de la excepción
        """
        error_message = "Null pointer in column 'Amount'"
        
        try:
            with ctx.span("Pipeline"):
                with ctx.span("Transform"):
                    raise RuntimeError(error_message)
        except RuntimeError:
            pass

        report = narrator.summarize_execution(ctx)
        evidence = report[ReportKeys.FORENSIC_EVIDENCE]

        assert len(evidence) > 0
        
        transform_evidence = [e for e in evidence if e["source"] == "Transform"]
        assert len(transform_evidence) >= 1, (
            f"No se encontró evidencia del span 'Transform'. "
            f"Evidencia disponible: {evidence}"
        )
        
        assert any(error_message in e["message"] for e in transform_evidence)

    def test_multiple_errors_all_captured(self, ctx, narrator):
        """
        Invariante P₂: Múltiples errores generan múltiples entradas.
        
        Cada error en un span distinto debe tener su propia entrada
        en la evidencia forense.
        """
        try:
            with ctx.span("Pipeline"):
                try:
                    with ctx.span("Step_1"):
                        raise ValueError("Error in Step_1")
                except ValueError:
                    pass  # Capturado pero registrado
                    
                try:
                    with ctx.span("Step_2"):
                        raise TypeError("Error in Step_2")
                except TypeError:
                    pass
                    
                try:
                    with ctx.span("Step_3"):
                        raise KeyError("Error in Step_3")
                except KeyError:
                    pass
        except Exception:
            pass

        report = narrator.summarize_execution(ctx)
        evidence = report[ReportKeys.FORENSIC_EVIDENCE]

        sources = {e["source"] for e in evidence}
        
        assert "Step_1" in sources, "Falta evidencia de Step_1"
        assert "Step_2" in sources, "Falta evidencia de Step_2"
        assert "Step_3" in sources, "Falta evidencia de Step_3"

    def test_nested_exceptions_captured(self, ctx, narrator):
        """
        Excepciones anidadas (chained) deben capturarse correctamente.
        """
        try:
            with ctx.span("Outer"):
                try:
                    with ctx.span("Inner"):
                        raise ValueError("Original error")
                except ValueError as e:
                    raise RuntimeError("Wrapper error") from e
        except RuntimeError:
            pass

        report = narrator.summarize_execution(ctx)
        evidence = report[ReportKeys.FORENSIC_EVIDENCE]

        # Debe haber al menos una entrada
        assert len(evidence) >= 1
        
        # Verificar que algún mensaje captura información del error
        messages = " ".join(e.get("message", "") for e in evidence)
        assert "error" in messages.lower()

    # ═══════════════════════════════════════════════════════════════════════
    # EJECUCIÓN EXITOSA (CASO BASE)
    # ═══════════════════════════════════════════════════════════════════════

    def test_successful_execution_no_evidence(self, ctx, narrator):
        """
        Ejecución limpia: sin errores → sin evidencia forense.
        
        Invariante: |errors| = 0 ⟹ |forensic_evidence| = 0
        """
        with ctx.span("Clean Pipeline"):
            with ctx.span("Step A"):
                pass
            with ctx.span("Step B"):
                pass
            with ctx.span("Step C"):
                pass

        report = narrator.summarize_execution(ctx)

        assert report[ReportKeys.VERDICT_CODE] != VerdictCode.REJECTED_PHYSICS.value
        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) == 0

    def test_successful_nested_execution(self, ctx, narrator):
        """
        Ejecución exitosa con estructura compleja.
        """
        with ctx.span("Root"):
            with ctx.span("Branch_A"):
                with ctx.span("Leaf_A1"):
                    pass
                with ctx.span("Leaf_A2"):
                    pass
            with ctx.span("Branch_B"):
                with ctx.span("Leaf_B1"):
                    pass

        report = narrator.summarize_execution(ctx)
        
        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) == 0
        
        # Todas las fases deben ser exitosas
        for phase in report[ReportKeys.PHASES]:
            assert phase["status"] != SpanStatus.CRITICO.value

    # ═══════════════════════════════════════════════════════════════════════
    # TIPOS DE EXCEPCIONES
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "exception_type, message",
        [
            (ValueError, "Invalid value provided"),
            (TypeError, "Type mismatch in operation"),
            (KeyError, "Missing required key"),
            (IndexError, "Index out of bounds"),
            (IOError, "IO operation failed"),
            (RuntimeError, "Runtime condition violated"),
            (AttributeError, "Missing attribute"),
            (ZeroDivisionError, "Division by zero"),
            (MemoryError, "Out of memory"),
            (AssertionError, "Assertion failed"),
        ],
        ids=lambda x: x.__name__ if isinstance(x, type) else str(x)[:20],
    )
    def test_different_exception_types_captured(
        self, ctx, narrator, exception_type, message
    ):
        """
        El sistema forense debe capturar cualquier tipo de excepción.
        
        No debe haber filtrado por tipo; todas las excepciones
        son relevantes para análisis post-mortem.
        """
        try:
            with ctx.span("Fault Injection"):
                raise exception_type(message)
        except exception_type:
            pass

        report = narrator.summarize_execution(ctx)

        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) > 0
        assert any(
            message in e["message"] 
            for e in report[ReportKeys.FORENSIC_EVIDENCE]
        )

    def test_custom_exception_captured(self, ctx, narrator):
        """
        Excepciones personalizadas (user-defined) también se capturan.
        """
        class CustomBusinessException(Exception):
            """Excepción de negocio personalizada."""
            pass

        try:
            with ctx.span("Business Logic"):
                raise CustomBusinessException("Business rule violated")
        except CustomBusinessException:
            pass

        report = narrator.summarize_execution(ctx)
        
        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) > 0


# =============================================================================
# PRUEBAS DEL MODO LEGACY (RETROCOMPATIBILIDAD)
# =============================================================================

class TestLegacyFallback:
    """
    Pruebas del modo legacy (start_step / end_step / record_error).

    El TelemetryContext debe soportar la API antigua y traducirla
    internamente al modelo jerárquico de spans para que el
    TelemetryNarrator pueda generar reportes coherentes.
    
    Invariantes del modo legacy:
      (L₁) start_step(name) + end_step(name) ≈ with ctx.span(name)
      (L₂) record_error genera entrada en forensic_evidence
      (L₃) El modo legacy es documentado en la narrativa
    """

    @pytest.fixture
    def ctx(self) -> TelemetryContext:
        return TelemetryContext()

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    # ═══════════════════════════════════════════════════════════════════════
    # OPERACIONES BÁSICAS LEGACY
    # ═══════════════════════════════════════════════════════════════════════

    def test_legacy_step_creates_span(self, ctx, narrator):
        """
        start_step + end_step deben crear un span equivalente.
        """
        ctx.start_step("Legacy Step")
        ctx.end_step("Legacy Step")

        # Debe haber alguna representación del paso
        report = narrator.summarize_execution(ctx)
        assert ReportKeys.VERDICT_CODE in report

    def test_legacy_error_produces_forensic_evidence(self, ctx, narrator):
        """
        Invariante L₂: record_error genera exactamente una entrada.
        """
        ctx.start_step("Legacy Step 1")
        ctx.end_step("Legacy Step 1")
        ctx.record_error("Legacy Step 2", "Something went wrong")

        report = narrator.summarize_execution(ctx)

        assert report[ReportKeys.VERDICT_CODE] == VerdictCode.REJECTED_PHYSICS.value
        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) >= 1

    def test_legacy_multiple_errors(self, ctx, narrator):
        """
        Múltiples record_error generan múltiples entradas de evidencia.
        """
        ctx.record_error("Step_A", "Error A")
        ctx.record_error("Step_B", "Error B")
        ctx.record_error("Step_C", "Error C")

        report = narrator.summarize_execution(ctx)
        evidence = report[ReportKeys.FORENSIC_EVIDENCE]

        assert len(evidence) >= 3

    def test_legacy_narrative_mentions_compatibility(self, ctx, narrator):
        """
        Invariante L₃: La narrativa documenta el modo legacy.
        
        Para trazabilidad, es importante saber que se usó
        la API antigua vs la API de spans.
        """
        ctx.start_step("Old Step")
        ctx.end_step("Old Step")
        ctx.record_error("Failing Step", "Timeout")

        report = narrator.summarize_execution(ctx)

        narrative_lower = report[ReportKeys.NARRATIVE].lower()
        causality_lower = [
            c.lower() for c in report.get(ReportKeys.CAUSALITY_CHAIN, [])
        ]

        legacy_documented = (
            "legacy" in narrative_lower
            or "compatibilidad" in narrative_lower
            or "retrocompat" in narrative_lower
            or any("legacy" in c for c in causality_lower)
        )

        assert legacy_documented, (
            "La narrativa debe documentar el modo legacy para trazabilidad.\n"
            f"Narrativa recibida: {report[ReportKeys.NARRATIVE]}"
        )

    def test_legacy_without_errors_no_evidence(self, ctx, narrator):
        """
        Modo legacy sin errores: ejecución limpia.
        """
        ctx.start_step("Step A")
        ctx.end_step("Step A")
        ctx.start_step("Step B")
        ctx.end_step("Step B")

        report = narrator.summarize_execution(ctx)

        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) == 0

    # ═══════════════════════════════════════════════════════════════════════
    # CASOS DEGENERADOS LEGACY
    # ═══════════════════════════════════════════════════════════════════════

    def test_legacy_start_without_end(self, ctx, narrator):
        """
        start_step sin end_step correspondiente.
        
        El sistema debe manejar este caso gracefully,
        ya sea auto-cerrando o marcando como incompleto.
        """
        ctx.start_step("Orphan Step")
        # Sin end_step

        # No debe lanzar excepción
        report = narrator.summarize_execution(ctx)
        assert ReportKeys.VERDICT_CODE in report

    def test_legacy_end_without_start(self, ctx, narrator):
        """
        end_step sin start_step previo.
        
        Debe manejarse sin crash, posiblemente como no-op.
        """
        # No debe lanzar excepción
        try:
            ctx.end_step("Phantom Step")
            report = narrator.summarize_execution(ctx)
            assert ReportKeys.VERDICT_CODE in report
        except Exception as e:
            # Si lanza excepción, debe ser descriptiva
            assert "step" in str(e).lower() or "start" in str(e).lower()

    def test_legacy_duplicate_start(self, ctx, narrator):
        """
        Dos start_step consecutivos para el mismo nombre.
        """
        ctx.start_step("Duplicate")
        ctx.start_step("Duplicate")  # ¿Reinicia o apila?
        ctx.end_step("Duplicate")
        ctx.end_step("Duplicate")

        report = narrator.summarize_execution(ctx)
        # Solo verificamos que no crashee
        assert ReportKeys.VERDICT_CODE in report

    def test_legacy_mismatched_names(self, ctx, narrator):
        """
        end_step con nombre diferente al start_step.
        """
        ctx.start_step("Started")
        ctx.end_step("Different")  # Nombre no coincide

        report = narrator.summarize_execution(ctx)
        assert ReportKeys.VERDICT_CODE in report


# =============================================================================
# PRUEBAS DEL MODO MIXTO (SPANS + LEGACY)
# =============================================================================

class TestMixedMode:
    """
    Pruebas de coexistencia entre spans jerárquicos y API legacy.

    Cuando se mezclan ambos modos dentro del mismo contexto,
    las métricas legacy deben registrarse tanto en el ámbito global
    como en el span activo al momento del registro.
    
    Invariantes del modo mixto:
      (M₁) Métricas legacy dentro de span → existen en global + span
      (M₂) Métricas legacy fuera de span → solo en global
      (M₃) La estructura jerárquica de spans no se altera por ops legacy
    """

    @pytest.fixture
    def ctx(self) -> TelemetryContext:
        return TelemetryContext()

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    def test_legacy_metrics_inside_span(self, ctx):
        """
        Invariante M₁: Métricas legacy dentro de un span
        existen en ambos ámbitos: global y span activo.
        """
        with ctx.span("Hybrid Phase"):
            ctx.start_step("Legacy Inside Span")
            ctx.record_metric("legacy", "count", 50)
            ctx.end_step("Legacy Inside Span")

        assert len(ctx.root_spans) == 1
        assert ctx.metrics["legacy.count"] == 50
        assert ctx.root_spans[0].metrics["legacy.count"] == 50

    def test_legacy_metrics_outside_span_only_global(self, ctx):
        """
        Invariante M₂: Métricas legacy registradas fuera de spans
        solo existen en el ámbito global.
        """
        ctx.record_metric("global", "total", 200)

        with ctx.span("After Metric"):
            pass

        assert ctx.metrics["global.total"] == 200
        assert "global.total" not in ctx.root_spans[0].metrics

    def test_mixed_mode_span_structure_preserved(self, ctx):
        """
        Invariante M₃: Operaciones legacy no alteran estructura de spans.
        """
        with ctx.span("Outer"):
            ctx.start_step("Legacy")
            ctx.end_step("Legacy")
            with ctx.span("Inner"):
                pass

        root = ctx.root_spans[0]

        assert root.name == "Outer"
        assert len(root.children) == 1
        assert root.children[0].name == "Inner"

    def test_error_in_legacy_inside_span_bubbles(self, ctx, narrator):
        """
        Error registrado con API legacy dentro de un span
        debe reflejarse en la evidencia forense.
        """
        with ctx.span("Container"):
            ctx.record_error("Legacy Sub-Step", "Connection refused")

        report = narrator.summarize_execution(ctx)

        assert len(report[ReportKeys.FORENSIC_EVIDENCE]) >= 1
        assert any(
            "Connection refused" in e["message"]
            for e in report[ReportKeys.FORENSIC_EVIDENCE]
        )

    def test_mixed_metrics_aggregation(self, ctx):
        """
        Métricas de ambos modos se agregan correctamente.
        """
        # Métrica global legacy
        ctx.record_metric("global", "counter", 10)

        with ctx.span("Phase_1") as span:
            # Métrica en span
            span.metrics["span_counter"] = 20
            # Métrica legacy dentro de span
            ctx.record_metric("legacy", "value", 30)

        # Verificar agregación
        assert ctx.metrics["global.counter"] == 10
        assert ctx.metrics["legacy.value"] == 30
        
        root = ctx.root_spans[0]
        assert root.metrics["span_counter"] == 20
        assert root.metrics["legacy.value"] == 30

    def test_complex_mixed_mode_scenario(self, ctx, narrator):
        """
        Escenario complejo mezclando spans y legacy.
        """
        ctx.start_step("Pre-Process")
        ctx.record_metric("pre", "count", 100)
        ctx.end_step("Pre-Process")

        with ctx.span("Main Pipeline"):
            ctx.record_metric("main", "input_size", 1000)
            
            with ctx.span("Transform"):
                ctx.start_step("Legacy Transform Sub")
                ctx.record_metric("transform", "rows", 500)
                ctx.end_step("Legacy Transform Sub")

            with ctx.span("Validate"):
                pass

        ctx.start_step("Post-Process")
        ctx.end_step("Post-Process")

        report = narrator.summarize_execution(ctx)
        
        # Debe tener fases del pipeline
        assert len(report[ReportKeys.PHASES]) >= 1
        
        # Métricas globales deben existir
        assert ctx.metrics["pre.count"] == 100
        assert ctx.metrics["main.input_size"] == 1000


# =============================================================================
# PRUEBAS DE INVARIANTES GLOBALES DEL REPORTE
# =============================================================================

class TestReportInvariants:
    """
    Pruebas que validan invariantes estructurales del reporte
    generado por TelemetryNarrator.

    Invariantes del reporte:
      ┌─────┬───────────────────────────────────────────────────────────────┐
      │ R₁  │ report contiene: verdict_code, phases, forensic_evidence,    │
      │     │ narrative (claves obligatorias)                              │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ R₂  │ |phases| = |root_spans| (biyección fases ↔ raíces)          │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ R₃  │ Si ∃ error → verdict_code ∈ {REJECTED_*}                     │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ R₄  │ Si ∄ error → forensic_evidence = []                          │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ R₅  │ narrative es string no vacío                                 │
      └─────┴───────────────────────────────────────────────────────────────┘
    """

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE R₁: ESQUEMA COMPLETO
    # ═══════════════════════════════════════════════════════════════════════

    def test_report_schema_completeness(self, narrator):
        """
        Invariante R₁: todas las claves obligatorias presentes.
        """
        ctx = TelemetryContext()
        with ctx.span("Probe"):
            pass

        report = narrator.summarize_execution(ctx)

        required_keys = ReportKeys.required_keys()
        actual_keys = set(report.keys())
        missing = required_keys - actual_keys

        assert required_keys.issubset(actual_keys), (
            f"Claves faltantes: {missing}"
        )

    def test_report_schema_with_errors(self, narrator):
        """
        Esquema completo incluso con errores presentes.
        """
        ctx = TelemetryContext()
        try:
            with ctx.span("Faulty"):
                raise ValueError("Test error")
        except ValueError:
            pass

        report = narrator.summarize_execution(ctx)

        required_keys = ReportKeys.required_keys()
        assert required_keys.issubset(set(report.keys()))

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE R₂: BIYECCIÓN FASES ↔ RAÍCES
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize("num_roots", [1, 2, 3, 5, 10])
    def test_phases_count_matches_root_spans(self, narrator, num_roots):
        """
        Invariante R₂: |phases| = |root_spans|.
        
        Cada árbol raíz del bosque corresponde a exactamente una fase.
        """
        ctx = TelemetryContext()

        for i in range(num_roots):
            with ctx.span(f"Phase_{i}"):
                pass

        report = narrator.summarize_execution(ctx)

        assert len(report[ReportKeys.PHASES]) == num_roots
        assert len(report[ReportKeys.PHASES]) == len(ctx.root_spans)

    def test_phase_names_match_root_names(self, narrator):
        """
        Los nombres de fases deben coincidir con los nombres de raíces.
        """
        ctx = TelemetryContext()
        root_names = ["Alpha", "Beta", "Gamma"]

        for name in root_names:
            with ctx.span(name):
                pass

        report = narrator.summarize_execution(ctx)
        phase_names = [p["name"] for p in report[ReportKeys.PHASES]]

        assert phase_names == root_names

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE R₃ y R₄: CONSISTENCIA ERROR ↔ VEREDICTO
    # ═══════════════════════════════════════════════════════════════════════

    def test_error_implies_rejected_verdict(self, narrator):
        """
        Invariante R₃: presencia de error → veredicto de rechazo.
        """
        ctx = TelemetryContext()
        try:
            with ctx.span("Failing"):
                raise RuntimeError("Forced failure")
        except RuntimeError:
            pass

        report = narrator.summarize_execution(ctx)

        assert report[ReportKeys.VERDICT_CODE] in [
            VerdictCode.REJECTED_PHYSICS.value,
            VerdictCode.REJECTED_TIMEOUT.value,
            VerdictCode.REJECTED_RESOURCE.value,
        ]

    def test_no_error_implies_empty_evidence(self, narrator):
        """
        Invariante R₄: sin errores → evidencia vacía.
        """
        ctx = TelemetryContext()
        with ctx.span("Clean"):
            pass

        report = narrator.summarize_execution(ctx)

        assert report[ReportKeys.FORENSIC_EVIDENCE] == []

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE R₅: NARRATIVA NO VACÍA
    # ═══════════════════════════════════════════════════════════════════════

    def test_narrative_is_non_empty_string(self, narrator):
        """
        Invariante R₅: narrative es string no vacío.
        """
        ctx = TelemetryContext()
        with ctx.span("Documented"):
            pass

        report = narrator.summarize_execution(ctx)

        assert isinstance(report[ReportKeys.NARRATIVE], str)
        assert len(report[ReportKeys.NARRATIVE]) > 0

    def test_narrative_with_errors_is_descriptive(self, narrator):
        """
        La narrativa con errores debe ser descriptiva del fallo.
        """
        ctx = TelemetryContext()
        try:
            with ctx.span("Critical Failure"):
                raise IOError("Catastrophic disk failure")
        except IOError:
            pass

        report = narrator.summarize_execution(ctx)
        narrative = report[ReportKeys.NARRATIVE]

        # La narrativa debe mencionar el fallo de alguna forma
        assert len(narrative) > 20, "Narrativa demasiado corta para ser descriptiva"

    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXTO VACÍO
    # ═══════════════════════════════════════════════════════════════════════

    def test_empty_context_report_is_valid(self, narrator):
        """
        Contexto vacío produce un reporte válido con esquema completo.
        """
        ctx = TelemetryContext()
        report = narrator.summarize_execution(ctx)

        assert ReportKeys.VERDICT_CODE in report
        assert report[ReportKeys.PHASES] == []
        assert report[ReportKeys.FORENSIC_EVIDENCE] == []


# =============================================================================
# PRUEBAS DE RENDIMIENTO Y STRESS
# =============================================================================

class TestPerformanceAndStress:
    """
    Pruebas de rendimiento y comportamiento bajo carga.
    
    Estas pruebas verifican que el sistema de telemetría escale
    adecuadamente y no degrade bajo condiciones de uso intensivo.
    """

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    def test_wide_tree_many_siblings(self):
        """
        Árbol ancho: muchos hermanos (factor de ramificación alto).
        """
        ctx = TelemetryContext()
        num_children = 100

        with ctx.span("Wide Root"):
            for i in range(num_children):
                with ctx.span(f"Child_{i}"):
                    pass

        root = ctx.root_spans[0]
        assert len(root.children) == num_children
        
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"]

    def test_deep_tree_stress(self):
        """
        Árbol profundo: cadena larga sin desbordamiento.
        """
        ctx = TelemetryContext()
        depth = 50

        def build_chain(level):
            if level > depth:
                return
            with ctx.span(f"Level_{level}"):
                build_chain(level + 1)

        build_chain(1)

        validation = validate_tree_structure(ctx.root_spans)
        assert validation["is_valid"]
        assert validation["max_depth"] == depth

    def test_many_roots_stress(self):
        """
        Bosque con muchos árboles independientes.
        """
        ctx = TelemetryContext()
        num_trees = 200

        for i in range(num_trees):
            with ctx.span(f"Tree_{i}"):
                pass

        assert len(ctx.root_spans) == num_trees
        
        validation = validate_tree_structure(ctx.root_spans)
        assert validation["num_trees"] == num_trees

    def test_high_metric_volume(self):
        """
        Alto volumen de métricas en un span.
        """
        ctx = TelemetryContext()
        num_metrics = 500

        with ctx.span("Metric Heavy") as span:
            for i in range(num_metrics):
                span.metrics[f"metric_{i}"] = i * 1.5

        root = ctx.root_spans[0]
        assert len(root.metrics) == num_metrics

    def test_report_generation_time_bounded(self, narrator):
        """
        La generación del reporte debe completarse en tiempo razonable.
        """
        ctx = TelemetryContext()
        
        # Crear estructura compleja
        for tree in range(10):
            with ctx.span(f"Tree_{tree}"):
                for branch in range(5):
                    with ctx.span(f"Branch_{branch}"):
                        for leaf in range(5):
                            with ctx.span(f"Leaf_{leaf}"):
                                pass

        import time
        start = time.perf_counter()
        report = narrator.summarize_execution(ctx)
        elapsed = time.perf_counter() - start

        # Debe completarse en menos de 1 segundo
        assert elapsed < 1.0, f"Generación de reporte tardó {elapsed:.2f}s"
        assert ReportKeys.VERDICT_CODE in report


# =============================================================================
# PRUEBAS DE IDEMPOTENCIA Y DETERMINISMO
# =============================================================================

class TestIdempotenceAndDeterminism:
    """
    Pruebas que verifican propiedades de idempotencia y determinismo.
    
    - Idempotencia: operaciones repetidas producen el mismo resultado
    - Determinismo: dado el mismo input, siempre el mismo output
    """

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    def test_summarize_is_idempotent(self, narrator):
        """
        Llamar summarize_execution múltiples veces produce el mismo resultado.
        """
        ctx = TelemetryContext()
        with ctx.span("Test"):
            pass

        report1 = narrator.summarize_execution(ctx)
        report2 = narrator.summarize_execution(ctx)
        report3 = narrator.summarize_execution(ctx)

        assert report1 == report2 == report3

    def test_summarize_does_not_mutate_context(self, narrator):
        """
        summarize_execution no debe mutar el contexto de telemetría.
        """
        ctx = TelemetryContext()
        with ctx.span("Immutable Test") as span:
            span.metrics["value"] = 42

        original_span_count = len(ctx.root_spans)
        original_metric_value = ctx.root_spans[0].metrics["value"]

        _ = narrator.summarize_execution(ctx)
        _ = narrator.summarize_execution(ctx)

        assert len(ctx.root_spans) == original_span_count
        assert ctx.root_spans[0].metrics["value"] == original_metric_value

    def test_deterministic_tree_structure(self):
        """
        La estructura del árbol es determinista dado el mismo código.
        """
        def create_tree():
            ctx = TelemetryContext()
            with ctx.span("A"):
                with ctx.span("B"):
                    pass
                with ctx.span("C"):
                    pass
            return ctx

        ctx1 = create_tree()
        ctx2 = create_tree()

        # Misma estructura
        assert len(ctx1.root_spans) == len(ctx2.root_spans)
        assert ctx1.root_spans[0].name == ctx2.root_spans[0].name
        assert len(ctx1.root_spans[0].children) == len(ctx2.root_spans[0].children)

        for c1, c2 in zip(ctx1.root_spans[0].children, ctx2.root_spans[0].children):
            assert c1.name == c2.name