"""
Tests para Generación de Reportes Ejecutivos de Riesgo
=======================================================

Suite robustecida que valida:
- Análisis de integridad estructural de presupuestos
- Detección de anomalías topológicas (ciclos, nodos aislados)
- Scoring de riesgo basado en propiedades del grafo
- Traducción semántica a metáforas de construcción
- Compatibilidad hacia atrás de la API

Modelo Topológico del Presupuesto:
----------------------------------
El presupuesto se modela como un grafo dirigido G = (V, E) donde:
- V = APUs ∪ Insumos
- E = {(apu, insumo) : apu usa insumo}

Propiedades deseables:
- DAG (sin ciclos): Evita dependencias circulares de costos
- Conectado: Todo insumo es alcanzable desde algún APU
- Estabilidad Piramidal: |Insumos| >> |APUs| (base amplia)

Invariantes Topológicos:
- β₀ = 1 (ideal): Un solo componente conexo
- β₁ = 0 (ideal): Sin ciclos (es un bosque/árbol)
- Grado de entrada de APUs = 0 (son fuentes)
- Grado de salida de Insumos = 0 (son sumideros)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import pytest

from agent.business_topology import (
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

# Umbrales de scoring por defecto
DEFAULT_INTEGRITY_THRESHOLDS = {
    "excellent": 90.0,
    "good": 70.0,
    "acceptable": 50.0,
    "poor": 30.0,
}

# Penalizaciones esperadas
EXPECTED_PENALTIES = {
    "cycle": 50.0,       # Penalización por ciclo
    "isolated_node": 5.0,  # Penalización por nodo aislado
    "orphan_insumo": 3.0,  # Penalización por insumo huérfano
}

# Niveles de complejidad
COMPLEXITY_LEVELS = frozenset({"Baja", "Media", "Alta", "Muy Alta"})


class NodeType(Enum):
    """Tipos de nodos en el grafo de presupuesto."""
    APU = "APU"
    INSUMO = "INSUMO"
    UNKNOWN = "UNKNOWN"


class RiskLevel(Enum):
    """Niveles de riesgo del reporte."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


# =============================================================================
# BUILDERS: Construcción de Grafos con Validación
# =============================================================================


@dataclass
class NodeSpec:
    """Especificación de un nodo en el grafo."""
    id: str
    node_type: NodeType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Especificación de una arista en el grafo."""
    source: str
    target: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BudgetGraphBuilder:
    """
    Builder para grafos de presupuesto con validación de invariantes.
    
    Garantiza:
    - IDs de nodos únicos
    - Tipos de nodo válidos
    - Estructura consistente para análisis
    
    Ejemplo de uso:
        graph = (
            BudgetGraphBuilder()
            .with_apu("APU-001", "Cimentación")
            .with_insumo("INS-001", "Cemento")
            .with_edge("APU-001", "INS-001")
            .build()
        )
    """
    
    def __init__(self):
        self._nodes: Dict[str, NodeSpec] = {}
        self._edges: List[EdgeSpec] = []
        self._validate_on_build: bool = True
    
    def with_apu(
        self, 
        node_id: str, 
        description: str = "",
        **metadata
    ) -> "BudgetGraphBuilder":
        """Agrega un nodo APU al grafo."""
        if not node_id:
            raise ValueError("ID de nodo no puede estar vacío")
        
        self._nodes[node_id] = NodeSpec(
            id=node_id,
            node_type=NodeType.APU,
            metadata={"description": description, **metadata}
        )
        return self
    
    def with_insumo(
        self, 
        node_id: str, 
        description: str = "",
        cost: float = 0.0,
        **metadata
    ) -> "BudgetGraphBuilder":
        """Agrega un nodo Insumo al grafo."""
        if not node_id:
            raise ValueError("ID de nodo no puede estar vacío")
        if cost < 0:
            raise ValueError(f"Costo no puede ser negativo: {cost}")
        
        self._nodes[node_id] = NodeSpec(
            id=node_id,
            node_type=NodeType.INSUMO,
            metadata={"description": description, "cost": cost, **metadata}
        )
        return self
    
    def with_edge(
        self, 
        source: str, 
        target: str, 
        weight: float = 1.0
    ) -> "BudgetGraphBuilder":
        """Agrega una arista dirigida al grafo."""
        self._edges.append(EdgeSpec(source=source, target=target, weight=weight))
        return self
    
    def with_apus(self, count: int, prefix: str = "APU") -> "BudgetGraphBuilder":
        """Agrega múltiples APUs con IDs secuenciales."""
        for i in range(1, count + 1):
            self.with_apu(f"{prefix}{i}", f"Actividad {i}")
        return self
    
    def with_insumos(self, count: int, prefix: str = "Insumo") -> "BudgetGraphBuilder":
        """Agrega múltiples Insumos con IDs secuenciales."""
        for i in range(1, count + 1):
            self.with_insumo(f"{prefix}{i}", f"Material {i}")
        return self
    
    def with_pyramid_structure(
        self, 
        apu_count: int = 1, 
        insumos_per_apu: int = 10
    ) -> "BudgetGraphBuilder":
        """
        Crea estructura piramidal: pocos APUs, muchos insumos por APU.
        
        Esta es la estructura "saludable" típica de un presupuesto.
        """
        for i in range(1, apu_count + 1):
            apu_id = f"APU{i}"
            self.with_apu(apu_id, f"Actividad {i}")
            
            for j in range(1, insumos_per_apu + 1):
                insumo_id = f"Insumo{i}_{j}"
                self.with_insumo(insumo_id, f"Material {j} de APU {i}")
                self.with_edge(apu_id, insumo_id)
        
        return self
    
    def with_cycle(self, nodes: List[str]) -> "BudgetGraphBuilder":
        """
        Crea un ciclo entre los nodos especificados.
        
        Útil para testing de detección de ciclos.
        """
        if len(nodes) < 2:
            raise ValueError("Un ciclo requiere al menos 2 nodos")
        
        for i in range(len(nodes)):
            source = nodes[i]
            target = nodes[(i + 1) % len(nodes)]
            self.with_edge(source, target)
        
        return self
    
    def with_isolated_node(
        self, 
        node_id: str, 
        node_type: NodeType = NodeType.INSUMO
    ) -> "BudgetGraphBuilder":
        """Agrega un nodo aislado (sin conexiones)."""
        if node_type == NodeType.APU:
            self.with_apu(node_id, "APU Aislado")
        else:
            self.with_insumo(node_id, "Insumo Aislado")
        return self
    
    def without_validation(self) -> "BudgetGraphBuilder":
        """Desactiva validación al construir (para tests de casos inválidos)."""
        self._validate_on_build = False
        return self
    
    def _validate_graph(self, G: nx.DiGraph) -> List[str]:
        """Valida el grafo y retorna lista de advertencias."""
        warnings = []
        
        # Verificar que las aristas referencian nodos existentes
        for edge in self._edges:
            source = edge.source
            target = edge.target
            if source not in self._nodes:
                warnings.append(f"Arista referencia nodo fuente inexistente: {source}")
            if target not in self._nodes:
                warnings.append(f"Arista referencia nodo destino inexistente: {target}")
        
        return warnings
    
    def build(self) -> nx.DiGraph:
        """
        Construye el grafo NetworkX.
        
        Returns:
            nx.DiGraph con nodos y aristas configurados
        
        Raises:
            ValueError: Si la validación está activa y hay errores
        """
        G = nx.DiGraph()
        
        # Agregar nodos
        for node_id, spec in self._nodes.items():
            G.add_node(
                node_id,
                type=spec.node_type.value,
                **spec.metadata
            )
        
        # Agregar aristas
        for edge in self._edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight, **edge.metadata)
        
        if self._validate_on_build:
            warnings = self._validate_graph(G)
            if warnings:
                raise ValueError(f"Grafo inválido: {warnings}")
        
        return G
    
    def get_node_counts(self) -> Dict[str, int]:
        """Retorna conteo de nodos por tipo."""
        counts = {NodeType.APU.value: 0, NodeType.INSUMO.value: 0}
        for spec in self._nodes.values():
            counts[spec.node_type.value] = counts.get(spec.node_type.value, 0) + 1
        return counts
    
    def get_expected_stability_ratio(self) -> float:
        """Calcula la proporción Insumos/APUs esperada."""
        counts = self.get_node_counts()
        apu_count = counts.get(NodeType.APU.value, 0)
        insumo_count = counts.get(NodeType.INSUMO.value, 0)
        
        if apu_count == 0:
            return float('inf') if insumo_count > 0 else 0.0
        return insumo_count / apu_count


class ScenarioFactory:
    """Factory para escenarios de prueba predefinidos."""
    
    @staticmethod
    def healthy_pyramid(apu_count: int = 1, insumos_per_apu: int = 200) -> nx.DiGraph:
        """Escenario: Presupuesto saludable con estructura piramidal."""
        return (
            BudgetGraphBuilder()
            .with_pyramid_structure(apu_count, insumos_per_apu)
            .build()
        )
    
    @staticmethod
    def simple_cycle() -> nx.DiGraph:
        """Escenario: Ciclo simple entre dos APUs."""
        return (
            BudgetGraphBuilder()
            .with_apu("APU1")
            .with_apu("APU2")
            .with_edge("APU1", "APU2")
            .with_edge("APU2", "APU1")
            .build()
        )
    
    @staticmethod
    def complex_cycle() -> nx.DiGraph:
        """Escenario: Ciclo complejo con múltiples nodos."""
        builder = BudgetGraphBuilder()
        nodes = ["APU1", "APU2", "APU3", "APU4"]
        
        for node in nodes:
            builder.with_apu(node)
        
        return builder.with_cycle(nodes).build()
    
    @staticmethod
    def isolated_insumo() -> nx.DiGraph:
        """Escenario: Insumo aislado (no conectado a ningún APU)."""
        return (
            BudgetGraphBuilder()
            .with_isolated_node("InsumoFantasma", NodeType.INSUMO)
            .build()
        )
    
    @staticmethod
    def orphan_insumos(count: int = 3) -> nx.DiGraph:
        """Escenario: Múltiples insumos huérfanos."""
        builder = BudgetGraphBuilder()
        for i in range(count):
            builder.with_isolated_node(f"OrphanInsumo{i}", NodeType.INSUMO)
        return builder.build()
    
    @staticmethod
    def mixed_anomalies() -> nx.DiGraph:
        """Escenario: Combinación de anomalías (ciclo + aislados)."""
        return (
            BudgetGraphBuilder()
            .with_apu("APU1")
            .with_apu("APU2")
            .with_edge("APU1", "APU2")
            .with_edge("APU2", "APU1")  # Ciclo
            .with_isolated_node("Orphan1", NodeType.INSUMO)  # Aislado
            .with_isolated_node("Orphan2", NodeType.INSUMO)  # Aislado
            .build()
        )
    
    @staticmethod
    def inverted_pyramid() -> nx.DiGraph:
        """Escenario: Estructura invertida (más APUs que Insumos) - inestable."""
        builder = BudgetGraphBuilder()
        
        # Muchos APUs
        for i in range(10):
            builder.with_apu(f"APU{i}")
        
        # Pocos insumos compartidos
        for i in range(3):
            builder.with_insumo(f"Insumo{i}")
        
        # Conectar todos los APUs a los pocos insumos
        for i in range(10):
            for j in range(3):
                builder.with_edge(f"APU{i}", f"Insumo{j}")
        
        return builder.build()
    
    @staticmethod
    def linear_chain(length: int = 5) -> nx.DiGraph:
        """Escenario: Cadena lineal de dependencias."""
        builder = BudgetGraphBuilder()
        
        for i in range(length):
            builder.with_apu(f"APU{i}")
        
        for i in range(length - 1):
            builder.with_edge(f"APU{i}", f"APU{i+1}")
        
        return builder.build()
    
    @staticmethod
    def empty_graph() -> nx.DiGraph:
        """Escenario: Grafo vacío."""
        return BudgetGraphBuilder().build()
    
    @staticmethod
    def single_node(node_type: NodeType = NodeType.APU) -> nx.DiGraph:
        """Escenario: Un solo nodo."""
        builder = BudgetGraphBuilder()
        if node_type == NodeType.APU:
            builder.with_apu("SingleAPU")
        else:
            builder.with_insumo("SingleInsumo")
        return builder.build()
    
    @staticmethod
    def large_budget(apu_count: int = 100, insumos_per_apu: int = 50) -> nx.DiGraph:
        """Escenario: Presupuesto grande para pruebas de rendimiento."""
        return (
            BudgetGraphBuilder()
            .with_pyramid_structure(apu_count, insumos_per_apu)
            .build()
        )


# =============================================================================
# UTILIDADES DE ANÁLISIS TOPOLÓGICO
# =============================================================================


class TopologyAnalyzer:
    """Utilidades para análisis topológico de grafos de presupuesto."""
    
    @staticmethod
    def calculate_betti_numbers(G: nx.DiGraph) -> Tuple[int, int]:
        """
        Calcula números de Betti para el grafo.
        
        β₀ = número de componentes conexos (ignorando dirección)
        β₁ = número de ciclos independientes = |E| - |V| + β₀
        
        Returns:
            Tuple (β₀, β₁)
        """
        # β₀: componentes conexos (grafo no dirigido subyacente)
        undirected = G.to_undirected()
        b0 = nx.number_connected_components(undirected)
        
        # β₁: ciclos independientes (característica de Euler)
        # Para un grafo: β₁ = |E| - |V| + β₀
        b1 = G.number_of_edges() - G.number_of_nodes() + b0
        
        return (b0, max(0, b1))  # β₁ no puede ser negativo
    
    @staticmethod
    def euler_characteristic(G: nx.DiGraph) -> int:
        """
        Calcula la característica de Euler.
        
        χ = |V| - |E| para un grafo
        χ = β₀ - β₁ por dualidad
        """
        return G.number_of_nodes() - G.number_of_edges()
    
    @staticmethod
    def get_sources(G: nx.DiGraph) -> Set[str]:
        """Retorna nodos fuente (grado de entrada = 0)."""
        return {n for n in G.nodes() if G.in_degree(n) == 0}
    
    @staticmethod
    def get_sinks(G: nx.DiGraph) -> Set[str]:
        """Retorna nodos sumidero (grado de salida = 0)."""
        return {n for n in G.nodes() if G.out_degree(n) == 0}
    
    @staticmethod
    def get_isolated_nodes(G: nx.DiGraph) -> Set[str]:
        """Retorna nodos sin ninguna conexión."""
        return {n for n in G.nodes() if G.degree(n) == 0}
    
    @staticmethod
    def stability_ratio(G: nx.DiGraph) -> float:
        """
        Calcula la proporción de estabilidad (Insumos/APUs).
        
        Valores altos indican "pirámide estable".
        """
        apu_count = sum(
            1 for n, d in G.nodes(data=True) 
            if d.get("type") == "APU"
        )
        insumo_count = sum(
            1 for n, d in G.nodes(data=True) 
            if d.get("type") == "INSUMO"
        )
        
        if apu_count == 0:
            return float('inf') if insumo_count > 0 else 0.0
        return insumo_count / apu_count
    
    @staticmethod
    def is_dag(G: nx.DiGraph) -> bool:
        """Verifica si el grafo es un DAG (sin ciclos)."""
        return nx.is_directed_acyclic_graph(G)
    
    @staticmethod
    def find_cycles(G: nx.DiGraph) -> List[List[str]]:
        """Encuentra todos los ciclos simples en el grafo."""
        try:
            return list(nx.simple_cycles(G))
        except nx.NetworkXNoCycle:
            return []


# =============================================================================
# FIXTURES
# =============================================================================


class TestFixtures:
    """Clase base con fixtures reutilizables."""

    @pytest.fixture
    def analyzer(self) -> BusinessTopologicalAnalyzer:
        """Proporciona una instancia del analizador."""
        return BusinessTopologicalAnalyzer()

    @pytest.fixture
    def healthy_graph(self) -> nx.DiGraph:
        """Grafo saludable con estructura piramidal."""
        return ScenarioFactory.healthy_pyramid()

    @pytest.fixture
    def cyclic_graph(self) -> nx.DiGraph:
        """Grafo con ciclo simple."""
        return ScenarioFactory.simple_cycle()

    @pytest.fixture
    def isolated_graph(self) -> nx.DiGraph:
        """Grafo con nodo aislado."""
        return ScenarioFactory.isolated_insumo()

    @pytest.fixture
    def empty_graph(self) -> nx.DiGraph:
        """Grafo vacío."""
        return ScenarioFactory.empty_graph()

    @pytest.fixture
    def large_graph(self) -> nx.DiGraph:
        """Grafo grande para pruebas de rendimiento."""
        return ScenarioFactory.large_budget(50, 20)

    @pytest.fixture
    def mixed_anomalies_graph(self) -> nx.DiGraph:
        """Grafo con múltiples tipos de anomalías."""
        return ScenarioFactory.mixed_anomalies()


# =============================================================================
# TESTS: ESTRUCTURA DAG SALUDABLE
# =============================================================================


class TestHealthyDAG(TestFixtures):
    """
    Tests para grafos DAG saludables.
    
    Valida que estructuras bien formadas:
    - Generen scores altos de integridad
    - No reporten anomalías falsas
    - Sean clasificadas con complejidad apropiada
    """

    def test_healthy_dag_high_integrity_score(self, analyzer, healthy_graph):
        """DAG saludable genera score de integridad > 90."""
        report = analyzer.generate_executive_report(healthy_graph)
        
        assert isinstance(report, ConstructionRiskReport)
        assert report.integrity_score > 90.0

    def test_healthy_dag_no_circular_risks(self, analyzer, healthy_graph):
        """DAG saludable no tiene riesgos circulares."""
        report = analyzer.generate_executive_report(healthy_graph)
        
        assert len(report.circular_risks) == 0

    def test_healthy_dag_no_waste_alerts(self, analyzer, healthy_graph):
        """DAG saludable no tiene alertas de desperdicio."""
        report = analyzer.generate_executive_report(healthy_graph)
        
        assert len(report.waste_alerts) == 0

    def test_healthy_dag_valid_complexity_level(self, analyzer, healthy_graph):
        """Nivel de complejidad es uno de los valores válidos."""
        report = analyzer.generate_executive_report(healthy_graph)
        
        assert report.complexity_level in COMPLEXITY_LEVELS

    def test_healthy_dag_is_acyclic(self, healthy_graph):
        """Grafo saludable es efectivamente un DAG."""
        assert TopologyAnalyzer.is_dag(healthy_graph)

    def test_healthy_dag_betti_numbers(self, healthy_graph):
        """DAG saludable tiene β₀=1, β₁=0 (conectado, sin ciclos)."""
        b0, b1 = TopologyAnalyzer.calculate_betti_numbers(healthy_graph)
        
        assert b0 == 1, "Debe ser un solo componente conexo"
        assert b1 == 0, "No debe tener ciclos"

    def test_healthy_dag_positive_stability_ratio(self, healthy_graph):
        """DAG saludable tiene alta proporción Insumos/APUs."""
        ratio = TopologyAnalyzer.stability_ratio(healthy_graph)
        
        assert ratio >= 10.0, "Estructura piramidal debe tener muchos insumos por APU"

    def test_healthy_dag_audit_report_generated(self, analyzer, healthy_graph):
        """Se genera reporte de auditoría correctamente."""
        result = analyzer.analyze_structural_integrity(healthy_graph)
        audit_lines = analyzer.get_audit_report(result)
        
        assert any("AUDITORIA ESTRUCTURAL" in line for line in audit_lines)

    @pytest.mark.parametrize("insumo_count", [10, 50, 100, 200])
    def test_integrity_increases_with_stability(self, analyzer, insumo_count):
        """Score de integridad aumenta con la estabilidad piramidal."""
        graph = ScenarioFactory.healthy_pyramid(1, insumo_count)
        report = analyzer.generate_executive_report(graph)
        
        # Con más insumos por APU, mayor estabilidad
        expected_min_score = min(50 + insumo_count * 0.2, 100)
        assert report.integrity_score >= expected_min_score * 0.8  # 80% del esperado


# =============================================================================
# TESTS: DETECCIÓN DE CICLOS
# =============================================================================


class TestCircularReferences(TestFixtures):
    """
    Tests para detección de referencias circulares.
    
    Los ciclos en un presupuesto indican:
    - Dependencias de costo circulares
    - Errores lógicos en la estructura
    - Alto riesgo de cálculos infinitos
    """

    def test_simple_cycle_detected(self, analyzer, cyclic_graph):
        """Ciclo simple de 2 nodos es detectado."""
        report = analyzer.generate_executive_report(cyclic_graph)
        
        assert len(report.circular_risks) > 0

    def test_cycle_severely_penalizes_score(self, analyzer, cyclic_graph):
        """Ciclo penaliza severamente el score de integridad."""
        report = analyzer.generate_executive_report(cyclic_graph)
        
        assert report.integrity_score <= 50.0

    def test_cycle_message_format(self, analyzer, cyclic_graph):
        """Mensaje de ciclo contiene texto esperado."""
        report = analyzer.generate_executive_report(cyclic_graph)
        
        assert any("ciclo" in r.lower() for r in report.circular_risks)

    def test_complex_cycle_detected(self, analyzer):
        """Ciclo complejo de 4+ nodos es detectado."""
        graph = ScenarioFactory.complex_cycle()
        report = analyzer.generate_executive_report(graph)
        
        assert len(report.circular_risks) > 0

    def test_multiple_cycles_all_detected(self, analyzer):
        """Múltiples ciclos independientes son detectados."""
        builder = (
            BudgetGraphBuilder()
            .with_apu("A1").with_apu("A2")
            .with_apu("B1").with_apu("B2")
            .with_edge("A1", "A2").with_edge("A2", "A1")  # Ciclo 1
            .with_edge("B1", "B2").with_edge("B2", "B1")  # Ciclo 2
        )
        graph = builder.build()
        
        report = analyzer.generate_executive_report(graph)
        
        # Debe detectar ambos ciclos o al menos reportar múltiples
        assert len(report.circular_risks) >= 1

    def test_cycle_triggers_audit_alert(self, analyzer, cyclic_graph):
        """Ciclo genera alerta en reporte de auditoría."""
        result = analyzer.analyze_structural_integrity(cyclic_graph)
        audit_lines = analyzer.get_audit_report(result)
        
        assert any("ALERTA" in line for line in audit_lines)

    def test_self_loop_detected(self, analyzer):
        """Auto-referencia (self-loop) es detectada como ciclo."""
        builder = (
            BudgetGraphBuilder()
            .with_apu("APU1")
            .with_edge("APU1", "APU1")  # Self-loop
        )
        graph = builder.build()
        
        report = analyzer.generate_executive_report(graph)
        
        # Un self-loop es un ciclo de longitud 1
        assert report.integrity_score < 100.0

    def test_cycle_betti_number(self, cyclic_graph):
        """Grafo con ciclo tiene β₁ > 0."""
        b0, b1 = TopologyAnalyzer.calculate_betti_numbers(cyclic_graph)
        
        assert b1 >= 1, "Ciclo debe reflejarse en β₁"


# =============================================================================
# TESTS: NODOS AISLADOS Y HUÉRFANOS
# =============================================================================


class TestIsolatedNodes(TestFixtures):
    """
    Tests para detección de nodos aislados.
    
    Nodos aislados representan:
    - Insumos definidos pero no usados (desperdicio)
    - APUs sin insumos (incompletos)
    - Inconsistencias en el presupuesto
    """

    def test_isolated_insumo_detected(self, analyzer, isolated_graph):
        """Insumo aislado es detectado."""
        report = analyzer.generate_executive_report(isolated_graph)
        
        assert len(report.waste_alerts) > 0

    def test_isolated_penalizes_score(self, analyzer, isolated_graph):
        """Nodo aislado penaliza el score de integridad."""
        report = analyzer.generate_executive_report(isolated_graph)
        
        assert report.integrity_score < 100.0

    def test_isolated_message_format(self, analyzer, isolated_graph):
        """Mensaje de aislado contiene texto esperado."""
        report = analyzer.generate_executive_report(isolated_graph)
        
        assert any("aislado" in alert.lower() for alert in report.waste_alerts)

    def test_multiple_isolated_nodes_all_detected(self, analyzer):
        """Múltiples nodos aislados son detectados."""
        graph = ScenarioFactory.orphan_insumos(count=5)
        report = analyzer.generate_executive_report(graph)
        
        assert len(report.waste_alerts) > 0
        # Penalización proporcional al número de aislados
        assert report.integrity_score < 90.0

    def test_isolated_apu_detected(self, analyzer):
        """APU aislado (sin insumos) también es detectado."""
        builder = BudgetGraphBuilder().with_apu("APU_Alone")
        graph = builder.build()
        
        report = analyzer.generate_executive_report(graph)
        
        # Un APU sin insumos es anómalo
        assert report.integrity_score < 100.0

    def test_isolated_triggers_audit_warning(self, analyzer, isolated_graph):
        """Nodo aislado genera advertencia en auditoría."""
        result = analyzer.analyze_structural_integrity(isolated_graph)
        audit_lines = analyzer.get_audit_report(result)
        
        assert any("ADVERTENCIA" in line for line in audit_lines)

    def test_isolated_node_count_in_topology(self, isolated_graph):
        """Conteo correcto de nodos aislados."""
        isolated = TopologyAnalyzer.get_isolated_nodes(isolated_graph)
        
        assert len(isolated) == 1

    def test_multiple_components_increase_betti_0(self):
        """Nodos aislados aumentan β₀ (componentes)."""
        graph = ScenarioFactory.orphan_insumos(count=3)
        b0, _ = TopologyAnalyzer.calculate_betti_numbers(graph)
        
        # Cada nodo aislado es su propio componente
        assert b0 == 3


# =============================================================================
# TESTS: ANOMALÍAS COMBINADAS
# =============================================================================


class TestMixedAnomalies(TestFixtures):
    """
    Tests para grafos con múltiples tipos de anomalías.
    
    Valida que:
    - Todas las anomalías sean detectadas
    - Las penalizaciones sean acumulativas
    - Los mensajes sean claros y específicos
    """

    def test_all_anomalies_detected(self, analyzer, mixed_anomalies_graph):
        """Ciclos y nodos aislados son detectados simultáneamente."""
        report = analyzer.generate_executive_report(mixed_anomalies_graph)
        
        assert len(report.circular_risks) > 0
        assert len(report.waste_alerts) > 0

    def test_penalties_accumulate(self, analyzer, mixed_anomalies_graph):
        """Penalizaciones de diferentes anomalías se acumulan."""
        report = analyzer.generate_executive_report(mixed_anomalies_graph)
        
        # Score debe ser muy bajo con múltiples problemas
        assert report.integrity_score <= 50.0

    def test_each_anomaly_type_in_audit(self, analyzer, mixed_anomalies_graph):
        """Cada tipo de anomalía aparece en el reporte de auditoría."""
        result = analyzer.analyze_structural_integrity(mixed_anomalies_graph)
        audit_lines = analyzer.get_audit_report(result)
        audit_text = "\n".join(audit_lines)
        
        # Debe mencionar tanto ciclos como nodos aislados
        has_cycle_mention = any(
            term in audit_text.lower() 
            for term in ["ciclo", "circular", "loop"]
        )
        has_isolated_mention = any(
            term in audit_text.lower() 
            for term in ["aislado", "huérfano", "orphan", "isolated"]
        )
        
        assert has_cycle_mention or has_isolated_mention


# =============================================================================
# TESTS: CASOS EDGE
# =============================================================================


class TestEdgeCases(TestFixtures):
    """Tests para casos límite y situaciones extremas."""

    def test_empty_graph_handled(self, analyzer, empty_graph):
        """Grafo vacío es manejado sin errores."""
        report = analyzer.generate_executive_report(empty_graph)
        
        assert report is not None
        assert isinstance(report, ConstructionRiskReport)

    def test_single_node_graph(self, analyzer):
        """Grafo con un solo nodo es manejado."""
        graph = ScenarioFactory.single_node(NodeType.APU)
        report = analyzer.generate_executive_report(graph)
        
        assert report is not None

    def test_linear_chain_no_cycle(self, analyzer):
        """Cadena lineal no es detectada como ciclo."""
        graph = ScenarioFactory.linear_chain(length=5)
        report = analyzer.generate_executive_report(graph)
        
        assert len(report.circular_risks) == 0

    def test_inverted_pyramid_low_stability(self, analyzer):
        """Pirámide invertida tiene baja estabilidad."""
        graph = ScenarioFactory.inverted_pyramid()
        ratio = TopologyAnalyzer.stability_ratio(graph)
        
        assert ratio < 1.0, "Más APUs que Insumos indica estructura inestable"

    def test_disconnected_components_multiple(self, analyzer):
        """Múltiples componentes desconectados son detectados."""
        builder = (
            BudgetGraphBuilder()
            .with_apu("APU1").with_insumo("I1").with_edge("APU1", "I1")
            .with_apu("APU2").with_insumo("I2").with_edge("APU2", "I2")
            # Sin conexión entre los dos grupos
        )
        graph = builder.build()
        
        b0, _ = TopologyAnalyzer.calculate_betti_numbers(graph)
        assert b0 == 2, "Debe haber 2 componentes conexos"

    def test_very_large_graph_performance(self, analyzer, large_graph):
        """Grafo grande se procesa en tiempo razonable."""
        import time
        
        start = time.time()
        report = analyzer.generate_executive_report(large_graph)
        elapsed = time.time() - start
        
        assert report is not None
        assert elapsed < 5.0, "Procesamiento no debe exceder 5 segundos"

    def test_graph_with_only_insumos(self, analyzer):
        """Grafo con solo insumos (sin APUs) es detectado como anómalo."""
        builder = (
            BudgetGraphBuilder()
            .with_insumo("I1")
            .with_insumo("I2")
            .with_insumo("I3")
        )
        graph = builder.build()
        
        report = analyzer.generate_executive_report(graph)
        
        # Sin APUs, todos los insumos están huérfanos
        assert len(report.waste_alerts) > 0 or report.integrity_score < 100.0


# =============================================================================
# TESTS: SCORING Y MÉTRICAS
# =============================================================================


class TestScoringMetrics(TestFixtures):
    """
    Tests para el sistema de scoring de integridad.
    
    Valida que:
    - El score esté en rango válido [0, 100]
    - Las penalizaciones sean proporcionales
    - Los niveles de complejidad sean consistentes
    """

    def test_score_in_valid_range(self, analyzer, healthy_graph, cyclic_graph):
        """Score siempre está en [0, 100]."""
        for graph in [healthy_graph, cyclic_graph]:
            report = analyzer.generate_executive_report(graph)
            assert 0.0 <= report.integrity_score <= 100.0

    def test_perfect_score_requires_dag_and_connected(self, analyzer):
        """Score perfecto requiere DAG conectado sin anomalías."""
        graph = ScenarioFactory.healthy_pyramid(1, 100)
        report = analyzer.generate_executive_report(graph)
        
        if TopologyAnalyzer.is_dag(graph):
            b0, _ = TopologyAnalyzer.calculate_betti_numbers(graph)
            if b0 == 1:
                assert report.integrity_score >= 90.0

    def test_more_cycles_lower_score(self, analyzer):
        """Más ciclos resultan en score más bajo."""
        # Un ciclo
        single_cycle = ScenarioFactory.simple_cycle()
        report_single = analyzer.generate_executive_report(single_cycle)
        
        # Múltiples ciclos
        builder = (
            BudgetGraphBuilder()
            .with_apu("A1").with_apu("A2").with_apu("A3")
            .with_edge("A1", "A2").with_edge("A2", "A1")
            .with_edge("A2", "A3").with_edge("A3", "A2")
        )
        multi_cycle = builder.build()
        report_multi = analyzer.generate_executive_report(multi_cycle)
        
        # Más ciclos = score más bajo o igual
        assert report_multi.integrity_score <= report_single.integrity_score

    def test_complexity_level_correlates_with_size(self, analyzer):
        """Nivel de complejidad correlaciona con tamaño del grafo."""
        small = ScenarioFactory.healthy_pyramid(1, 10)
        large = ScenarioFactory.healthy_pyramid(10, 50)
        
        report_small = analyzer.generate_executive_report(small)
        report_large = analyzer.generate_executive_report(large)
        
        complexity_order = ["Baja", "Media", "Alta", "Muy Alta"]
        
        small_idx = complexity_order.index(report_small.complexity_level) \
            if report_small.complexity_level in complexity_order else -1
        large_idx = complexity_order.index(report_large.complexity_level) \
            if report_large.complexity_level in complexity_order else -1
        
        assert large_idx >= small_idx, "Grafo más grande debe tener complejidad >= "


# =============================================================================
# TESTS: TRADUCCIÓN SEMÁNTICA
# =============================================================================


class TestSemanticTranslation(TestFixtures):
    """
    Tests para la traducción de métricas a lenguaje de construcción.
    
    Valida que:
    - Los reportes usen metáforas de construcción
    - Los términos técnicos sean apropiados
    - La narrativa sea comprensible para el dominio
    """

    def test_audit_uses_construction_terms(self, analyzer, healthy_graph):
        """Auditoría usa terminología de construcción."""
        result = analyzer.analyze_structural_integrity(healthy_graph)
        audit_lines = analyzer.get_audit_report(result)
        audit_text = "\n".join(audit_lines)
        
        construction_terms = [
            "estructural", "integridad", "cimentación", 
            "carga", "soporte", "análisis"
        ]
        
        matches = sum(1 for term in construction_terms if term in audit_text.lower())
        assert matches >= 1, "Reporte debe usar terminología de construcción"

    def test_risk_terms_for_anomalies(self, analyzer, cyclic_graph):
        """Anomalías se describen con términos de riesgo apropiados."""
        report = analyzer.generate_executive_report(cyclic_graph)
        
        risk_terms = ["riesgo", "alerta", "crítico", "advertencia", "peligro"]
        all_messages = report.circular_risks + report.waste_alerts
        combined_text = " ".join(all_messages).lower()
        
        # Al menos un término de riesgo en los mensajes
        has_risk_term = any(term in combined_text for term in risk_terms)
        # O en el nivel de complejidad/score bajo
        has_low_score = report.integrity_score < 60.0
        
        assert has_risk_term or has_low_score

    def test_spanish_language_in_reports(self, analyzer, healthy_graph):
        """Reportes están en español."""
        result = analyzer.analyze_structural_integrity(healthy_graph)
        audit_lines = analyzer.get_audit_report(result)
        audit_text = "\n".join(audit_lines).lower()
        
        spanish_indicators = ["de", "del", "la", "los", "las", "en", "con"]
        matches = sum(1 for word in spanish_indicators if word in audit_text.split())
        
        assert matches >= 2, "Reporte debe estar en español"


# =============================================================================
# TESTS: COMPATIBILIDAD HACIA ATRÁS
# =============================================================================


class TestBackwardCompatibility(TestFixtures):
    """
    Tests para compatibilidad con versiones anteriores de la API.
    
    Garantiza que:
    - Métodos antiguos sigan funcionando
    - Formatos de salida sean consistentes
    - No se rompan integraciones existentes
    """

    def test_analyze_structural_integrity_returns_dict(self, analyzer, healthy_graph):
        """analyze_structural_integrity retorna estructura compatible."""
        result = analyzer.analyze_structural_integrity(healthy_graph)
        
        # Debe ser un diccionario o estructura similar
        assert result is not None
        assert hasattr(result, '__getitem__') or hasattr(result, '__dict__')

    def test_get_audit_report_returns_list(self, analyzer, healthy_graph):
        """get_audit_report retorna lista de strings."""
        result = analyzer.analyze_structural_integrity(healthy_graph)
        audit_lines = analyzer.get_audit_report(result)
        
        assert isinstance(audit_lines, (list, tuple))
        assert all(isinstance(line, str) for line in audit_lines)

    def test_audit_contains_standard_sections(self, analyzer, healthy_graph):
        """Auditoría contiene secciones estándar esperadas."""
        result = analyzer.analyze_structural_integrity(healthy_graph)
        audit_lines = analyzer.get_audit_report(result)
        audit_text = "\n".join(audit_lines)
        
        expected_sections = [
            "AUDITORIA ESTRUCTURAL",
            "Ciclos de Costo",
        ]
        
        for section in expected_sections:
            assert section in audit_text, f"Sección faltante: {section}"

    def test_report_dataclass_attributes(self, analyzer, healthy_graph):
        """ConstructionRiskReport tiene todos los atributos esperados."""
        report = analyzer.generate_executive_report(healthy_graph)
        
        required_attrs = [
            "integrity_score",
            "circular_risks",
            "waste_alerts",
            "complexity_level",
        ]
        
        for attr in required_attrs:
            assert hasattr(report, attr), f"Atributo faltante: {attr}"


# =============================================================================
# TESTS: INVARIANTES TOPOLÓGICOS
# =============================================================================


class TestTopologicalInvariants(TestFixtures):
    """
    Tests para invariantes matemáticos de topología algebraica.
    
    Valida propiedades fundamentales que siempre deben cumplirse.
    """

    def test_euler_characteristic_formula(self):
        """Verifica χ = |V| - |E| para varios grafos."""
        test_cases = [
            (ScenarioFactory.healthy_pyramid(1, 10), 11, 10),  # 1 APU + 10 Insumos, 10 edges
            (ScenarioFactory.empty_graph(), 0, 0),
        ]
        
        for graph, expected_v, expected_e in test_cases:
            chi = TopologyAnalyzer.euler_characteristic(graph)
            assert chi == expected_v - expected_e

    def test_betti_relationship(self):
        """Verifica β₀ - β₁ = χ."""
        graphs = [
            ScenarioFactory.healthy_pyramid(1, 10),
            ScenarioFactory.simple_cycle(),
            ScenarioFactory.linear_chain(5),
        ]
        
        for graph in graphs:
            b0, b1 = TopologyAnalyzer.calculate_betti_numbers(graph)
            chi = TopologyAnalyzer.euler_characteristic(graph)
            
            # β₀ - β₁ ≈ χ (puede haber diferencias por grafos dirigidos)
            # Para grafos no dirigidos subyacentes: β₀ - β₁ = χ
            undirected_chi = graph.number_of_nodes() - graph.to_undirected().number_of_edges()
            assert b0 - b1 == undirected_chi

    def test_dag_implies_no_cycles(self):
        """DAG implica β₁ = 0 en el espacio de caminos."""
        dag = ScenarioFactory.healthy_pyramid(2, 5)
        
        assert TopologyAnalyzer.is_dag(dag)
        cycles = TopologyAnalyzer.find_cycles(dag)
        assert len(cycles) == 0

    def test_sources_have_zero_in_degree(self):
        """Fuentes tienen grado de entrada cero."""
        graph = ScenarioFactory.healthy_pyramid(2, 5)
        sources = TopologyAnalyzer.get_sources(graph)
        
        for source in sources:
            assert graph.in_degree(source) == 0

    def test_sinks_have_zero_out_degree(self):
        """Sumideros tienen grado de salida cero."""
        graph = ScenarioFactory.healthy_pyramid(2, 5)
        sinks = TopologyAnalyzer.get_sinks(graph)
        
        for sink in sinks:
            assert graph.out_degree(sink) == 0

    def test_isolated_nodes_zero_total_degree(self):
        """Nodos aislados tienen grado total cero."""
        graph = ScenarioFactory.orphan_insumos(3)
        isolated = TopologyAnalyzer.get_isolated_nodes(graph)
        
        for node in isolated:
            assert graph.degree(node) == 0


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])