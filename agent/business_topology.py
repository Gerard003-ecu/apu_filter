import logging
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from app.constants import ColumnNames
from app.telemetry import TelemetryContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopologicalMetrics:
    """
    Métricas Topológicas Invariantes para el Grafo de Negocio.

    Attributes:
        beta_0 (int): Número de componentes conexas (fragmentación).
        beta_1 (int): Número de ciclos independientes (complejidad de bucles).
        euler_characteristic (int): Característica de Euler (beta_0 - beta_1).
        euler_efficiency (float): Eficiencia topológica normalizada (0.0 - 1.0).
    """

    beta_0: int
    beta_1: int
    euler_characteristic: int
    euler_efficiency: float = 1.0

    @property
    def is_connected(self) -> bool:
        """Determina si el grafo está conectado (tiene una sola componente)."""
        return self.beta_0 == 1

    @property
    def is_simply_connected(self) -> bool:
        """Determina si el grafo es simplemente conexo (conexo y sin ciclos)."""
        return self.beta_0 == 1 and self.beta_1 == 0


@dataclass
class ConstructionRiskReport:
    """
    Reporte Ejecutivo de Riesgos de Construcción.

    Attributes:
        integrity_score (float): Puntuación de integridad (0-100).
        waste_alerts (List[str]): Alertas de posible desperdicio (nodos aislados).
        circular_risks (List[str]): Riesgos de cálculo circular (ciclos).
        complexity_level (str): Nivel de complejidad (Baja, Media, Alta).
        details (Dict[str, Any]): Metadatos para serialización y visualización.
        financial_risk_level (Optional[str]): Nivel de riesgo financiero ('Bajo', 'Medio', 'Alto', 'CATÁSTROFICO').
        strategic_narrative (Optional[str]): Narrativa estratégica para decisores (La Voz del Consejo).
    """

    integrity_score: float
    waste_alerts: List[str]
    circular_risks: List[str]
    complexity_level: str
    details: Dict[str, Any] = field(default_factory=dict)
    financial_risk_level: Optional[str] = None
    strategic_narrative: Optional[str] = None


class BudgetGraphBuilder:
    """
    Construye el Grafo del Presupuesto (Topología de Negocio) Versión 2 con estructura Piramidal.
    Adopta la lógica de 'Upsert' y manejo jerárquico de la Propuesta 2.
    """

    def __init__(self):
        """Inicializa el constructor del grafo."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ROOT_NODE = "PROYECTO_TOTAL"

    def _sanitize_code(self, value: Any) -> str:
        """Sanitiza el código o identificador asegurando una cadena limpia y normalizada."""
        if pd.isna(value) or value is None:
            return ""
        sanitized = str(value).strip()
        sanitized = " ".join(sanitized.split())
        return sanitized

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convierte un valor a float de manera segura con soporte para formatos locales."""
        if pd.isna(value) or value is None:
            return default
        try:
            if isinstance(value, (int, float)):
                return float(value)
            str_value = str(value).strip()
            if "," in str_value and "." in str_value:
                if str_value.rfind(",") < str_value.rfind("."):
                    str_value = str_value.replace(",", "")
                else:
                    str_value = str_value.replace(".", "").replace(",", ".")
            elif "," in str_value and "." not in str_value:
                parts = str_value.split(",")
                if len(parts) == 2 and len(parts[1]) <= 2:
                    str_value = str_value.replace(",", ".")
                else:
                    str_value = str_value.replace(",", "")
            return float(str_value)
        except (ValueError, TypeError, AttributeError):
            return default

    def _create_node_attributes(
        self,
        node_type: str,
        level: int,
        source: str = "generated",
        idx: int = -1,
        inferred: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        attrs = {
            "type": node_type,
            "level": level,
            "source": source,
            "original_index": idx,
            "inferred": inferred,
        }
        attrs.update(kwargs)
        return attrs

    def _create_apu_attributes(
        self, row: pd.Series, source: str, idx: int, inferred: bool
    ) -> Dict[str, Any]:
        attrs = self._create_node_attributes(
            node_type="APU", level=2, source=source, idx=idx, inferred=inferred
        )
        if not inferred:
            attrs["description"] = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_APU))
            attrs["quantity"] = self._safe_float(row.get(ColumnNames.CANTIDAD_PRESUPUESTO))
        return attrs

    def _create_insumo_attributes(
        self, row: pd.Series, insumo_desc: str, source: str, idx: int
    ) -> Dict[str, Any]:
        return self._create_node_attributes(
            node_type="INSUMO",
            level=3,
            source=source,
            idx=idx,
            description=insumo_desc,
            tipo_insumo=self._sanitize_code(row.get(ColumnNames.TIPO_INSUMO)),
            unit_cost=self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU)),
        )

    def _upsert_edge(
        self, G: nx.DiGraph, u: str, v: str, unit_cost: float, quantity: float, idx: int
    ) -> bool:
        """Inserta o actualiza una arista aplicando agregación de cantidades y costos (Upsert)."""
        total_cost = unit_cost * quantity

        if G.has_edge(u, v):
            edge = G[u][v]
            edge["quantity"] += quantity
            edge["total_cost"] += total_cost
            edge["occurrence_count"] += 1
            if "original_indices" not in edge:
                edge["original_indices"] = []
            edge["original_indices"].append(idx)
            return False

        G.add_edge(
            u,
            v,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            occurrence_count=1,
            original_indices=[idx],
        )
        return True

    def _compute_graph_statistics(self, G: nx.DiGraph) -> Dict[str, int]:
        stats = {
            "chapter_count": 0,
            "apu_count": 0,
            "insumo_count": 0,
            "inferred_count": 0,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }

        type_counters = {
            "CAPITULO": "chapter_count",
            "APU": "apu_count",
            "INSUMO": "insumo_count",
        }

        for _, data in G.nodes(data=True):
            node_type = data.get("type")
            if node_type in type_counters:
                stats[type_counters[node_type]] += 1
                if node_type == "APU" and data.get("inferred", False):
                    stats["inferred_count"] += 1

        return stats

    def _process_presupuesto_row(
        self, G: nx.DiGraph, row: pd.Series, idx: int, chapter_cols: List[str]
    ) -> None:
        apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
        if not apu_code:
            return

        # Crear nodo APU (Nivel 2)
        attrs = self._create_apu_attributes(
            row, source="presupuesto", idx=idx, inferred=False
        )
        G.add_node(apu_code, **attrs)

        # Buscar y establecer jerarquía de capítulo
        chapter_name = None
        for col in chapter_cols:
            val = self._sanitize_code(row.get(col))
            if val:
                chapter_name = val
                break

        if chapter_name:
            if chapter_name not in G:
                G.add_node(
                    chapter_name,
                    type="CAPITULO",
                    level=1,
                    description=f"Capítulo: {chapter_name}",
                )
                G.add_edge(self.ROOT_NODE, chapter_name, relation="CONTAINS")
            G.add_edge(chapter_name, apu_code, relation="CONTAINS")
        else:
            G.add_edge(self.ROOT_NODE, apu_code, relation="CONTAINS")

    def _process_apu_detail_row(self, G: nx.DiGraph, row: pd.Series, idx: int) -> None:
        apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
        insumo_desc = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_INSUMO))

        if not apu_code or not insumo_desc:
            return

        # Inferir APU si no existe
        if apu_code not in G:
            attrs = self._create_apu_attributes(
                row, source="detail", idx=idx, inferred=True
            )
            G.add_node(apu_code, **attrs)
            G.add_edge(self.ROOT_NODE, apu_code, relation="CONTAINS_INFERRED")

        # Crear o reutilizar nodo Insumo
        if insumo_desc not in G:
            attrs = self._create_insumo_attributes(
                row, insumo_desc, source="detail", idx=idx
            )
            G.add_node(insumo_desc, **attrs)

        # Establecer relación APU -> Insumo con agregación
        qty = self._safe_float(row.get(ColumnNames.CANTIDAD_APU))
        cost = self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU))
        self._upsert_edge(G, apu_code, insumo_desc, cost, qty, idx)

    def build(
        self, presupuesto_df: pd.DataFrame, apus_detail_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Construye un grafo dirigido piramidal representando la topología del presupuesto.
        """
        G = nx.DiGraph(name="BudgetTopology")
        self.logger.info("Iniciando construcción del Grafo Piramidal de Presupuesto...")

        # Nivel 0: Nodo Raíz
        G.add_node(self.ROOT_NODE, type="ROOT", level=0, description="Proyecto Completo")

        # Columnas candidatas para identificar capítulos
        chapter_cols = ["CAPITULO", "CATEGORIA", "TITULO"]

        # Niveles 1 y 2: Procesar Presupuesto
        if presupuesto_df is not None and not presupuesto_df.empty:
            available_chapter_cols = [
                c for c in chapter_cols if c in presupuesto_df.columns
            ]
            for idx, row in presupuesto_df.iterrows():
                self._process_presupuesto_row(G, row, idx, available_chapter_cols)

        # Nivel 3: Procesar Detalle de APUs (Insumos)
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                self._process_apu_detail_row(G, row, idx)

        stats = self._compute_graph_statistics(G)
        self.logger.info(f"Grafo Piramidal construido: {stats}")
        return G


class BusinessTopologicalAnalyzer:
    """
    Analizador de topología de negocio V2 con Telemetría Granular.
    Fusión Estratégica:
    - Motor Matemático: Propuesta 1 (El Cerebro Forense)
    - Narrativa: Propuesta 2 (La Voz del Consejo)
    """

    def __init__(
        self, telemetry: Optional[TelemetryContext] = None, max_cycles: int = 100
    ):
        self.telemetry = telemetry
        self.max_cycles = max_cycles
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_euler_efficiency(self, graph: nx.DiGraph) -> float:
        """Calcula la Eficiencia de Euler normalizada mediante decaimiento exponencial (Propuesta 1)."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes <= 1:
            return 1.0

        min_edges = n_nodes - 1
        excess_edges = max(0, n_edges - min_edges)
        efficiency = np.exp(-excess_edges / n_nodes) if n_nodes > 0 else 1.0
        return round(efficiency, 4)

    def calculate_betti_numbers(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """Calcula métricas topológicas invariantes (Números de Betti)."""
        if graph.number_of_nodes() == 0:
            return TopologicalMetrics(0, 0, 0, 1.0)

        # Usar MultiGraph para preservar todas las aristas y calcular correctamente Betti_1
        undirected = nx.MultiGraph()
        undirected.add_nodes_from(graph.nodes(data=True))
        undirected.add_edges_from(graph.edges(data=True))

        beta_0 = nx.number_connected_components(undirected)
        n_edges = undirected.number_of_edges()
        n_nodes = undirected.number_of_nodes()

        beta_1 = max(0, n_edges - n_nodes + beta_0)
        euler_char = beta_0 - beta_1
        efficiency = self.calculate_euler_efficiency(graph)

        return TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            euler_characteristic=euler_char,
            euler_efficiency=efficiency,
        )

    def calculate_pyramid_stability(self, graph: nx.DiGraph) -> float:
        """Calcula el Índice de Estabilidad Piramidal (Ψ) con robustez mejorada (Propuesta 1)."""
        nodes_data = graph.nodes(data=True)
        num_apus = sum(1 for _, d in nodes_data if d.get("type") == "APU")
        num_insumos = sum(1 for _, d in nodes_data if d.get("type") == "INSUMO")

        if num_apus == 0 or num_insumos == 0:
            return 0.0

        base_ratio = num_insumos / num_apus
        ratio_term = np.log10(1 + base_ratio)
        density = nx.density(graph)
        density_penalty = 1.0 - min(density, 0.99)
        connectivity_factor = 1.0 if nx.is_directed_acyclic_graph(graph) else 0.7

        stability = ratio_term * density_penalty * connectivity_factor
        return round(stability, 3)

    def audit_integration_homology(
        self, graph_a: nx.DiGraph, graph_b: nx.DiGraph
    ) -> Dict[str, Any]:
        """Ejecuta el Test de Mayer-Vietoris riguroso (Propuesta 1)."""
        metrics_a = self.calculate_betti_numbers(graph_a)
        metrics_b = self.calculate_betti_numbers(graph_b)
        graph_union = nx.compose(graph_a, graph_b)
        metrics_union = self.calculate_betti_numbers(graph_union)

        nodes_a = set(graph_a.nodes())
        nodes_b = set(graph_b.nodes())
        common_nodes = nodes_a.intersection(nodes_b)

        graph_intersection = nx.DiGraph()
        if common_nodes:
            graph_intersection.add_nodes_from(common_nodes)
            for u, v in graph_a.edges():
                if u in common_nodes and v in common_nodes:
                    graph_intersection.add_edge(u, v)
            for u, v in graph_b.edges():
                if u in common_nodes and v in common_nodes:
                    graph_intersection.add_edge(u, v)

        metrics_intersection = self.calculate_betti_numbers(graph_intersection)
        delta = len(common_nodes) - metrics_intersection.beta_0 + 1

        emergent_theoretical = (
            metrics_a.beta_1 + metrics_b.beta_1 - metrics_intersection.beta_1 + delta
        )
        emergent_observed = metrics_union.beta_1 - (
            metrics_a.beta_1 + metrics_b.beta_1
        )
        discrepancy = abs(emergent_observed - emergent_theoretical)

        narrative = self._generate_mayer_vietoris_narrative(emergent_observed, discrepancy)

        verdict = "CLEAN_MERGE"
        if discrepancy <= 1:
            if emergent_observed > 0:
                verdict = "INTEGRATION_CONFLICT"
            elif emergent_observed < 0:
                verdict = "TOPOLOGY_SIMPLIFIED"
        else:
            verdict = "INCONSISTENT_TOPOLOGY"

        return {
            "status": verdict,
            "delta_beta_1_observed": emergent_observed,
            "delta_beta_1_theoretical": emergent_theoretical,
            "discrepancy": discrepancy,
            "details": {
                "beta_1_A": metrics_a.beta_1,
                "beta_1_B": metrics_b.beta_1,
                "beta_1_Union": metrics_union.beta_1,
                "common_nodes_count": len(common_nodes),
            },
            "narrative": narrative,
        }

    def _generate_mayer_vietoris_narrative(self, observed: int, discrepancy: float) -> str:
        if discrepancy > 1:
            return f"⚠️ Discrepancia topológica detectada (Δ={discrepancy}). Revisar superposición de componentes."
        if observed > 0:
            return f"🚨 ALERTA MAYER-VIETORIS: La fusión generó {observed} nuevos ciclos de dependencia. Conflicto de interfaz detectado."
        if observed < 0:
            return f"✅ Fusión simplificó la estructura. Se eliminaron {abs(observed)} ciclos redundantes."
        return "✅ Fusión topológicamente neutra: sin riesgos estructurales nuevos."

    def _get_raw_cycles(self, graph: nx.DiGraph) -> Tuple[List[List[str]], bool]:
        """Obtiene los ciclos crudos con algoritmo Johnson optimizado (Propuesta 1)."""
        cycles = []
        truncated = False
        try:
            cycle_generator = nx.simple_cycles(graph)
            max_cycle_length = 10
            for count, cycle in enumerate(cycle_generator):
                if count >= self.max_cycles:
                    truncated = True
                    self.logger.warning(f"Truncado de ciclos en {self.max_cycles}")
                    break
                if len(cycle) <= max_cycle_length:
                    cycles.append(cycle)
        except Exception as e:
            self.logger.error(f"Error en detección de ciclos: {e}")

        cycles.sort(key=len)
        return cycles, truncated

    def _detect_cycles(self, graph: nx.DiGraph) -> Tuple[List[str], bool]:
        """
        Detecta y formatea ciclos en el grafo (Compatibilidad hacia atrás).
        """
        raw_cycles, truncated = self._get_raw_cycles(graph)
        formatted_cycles = [" → ".join(map(str, c + [c[0]])) for c in raw_cycles]
        return formatted_cycles, truncated

    def detect_risk_synergy(
        self, graph: nx.DiGraph, raw_cycles: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """Detecta Sinergia de Riesgo por 'Betweenness Centrality' (Propuesta 1)."""
        if raw_cycles is None:
            raw_cycles, _ = self._get_raw_cycles(graph)

        if len(raw_cycles) < 2:
            return {
                "synergy_detected": False,
                "shared_nodes": [],
                "intersecting_cycles_count": 0,
                "bridge_nodes": [],
                "synergy_score": 0.0,
            }

        try:
            betweenness = nx.betweenness_centrality(graph, normalized=True)
            threshold = np.percentile(list(betweenness.values()), 75) if betweenness else 0.0
        except:
            betweenness = {}
            threshold = 0.0

        critical_nodes = {n for n, c in betweenness.items() if c >= threshold}

        synergy_pairs = []
        bridge_nodes = set()
        cycle_sets = [set(c) for c in raw_cycles]

        for i in range(len(cycle_sets)):
            for j in range(i + 1, len(cycle_sets)):
                intersection = cycle_sets[i].intersection(cycle_sets[j])
                if len(intersection) >= 2:
                    critical_intersection = intersection.intersection(critical_nodes)
                    if critical_intersection:
                        synergy_pairs.append((i, j))
                        bridge_nodes.update(critical_intersection)

        synergy_score = 0.0
        if synergy_pairs:
            total_pairs = len(cycle_sets) * (len(cycle_sets) - 1) / 2
            synergy_score = min(
                1.0, len(synergy_pairs) / total_pairs * len(bridge_nodes)
            )

        return {
            "synergy_detected": len(synergy_pairs) > 0,
            "shared_nodes": list(bridge_nodes),
            "intersecting_cycles_count": len(synergy_pairs),
            "bridge_nodes": list(bridge_nodes),
            "synergy_score": round(synergy_score, 3),
        }

    def _compute_connectivity_analysis(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calcula métricas de conectividad avanzadas (Propuesta 1)."""
        if graph.number_of_nodes() == 0:
            return {
                "is_dag": True,
                "num_wcc": 0,
                "is_weakly_connected": True,
                "num_scc": 0,
                "num_non_trivial_scc": 0,
                "scc_sizes": [],
                "non_trivial_scc": [],
                "articulation_points": [],
                "average_clustering": 0.0,
            }

        undirected = graph.to_undirected()
        scc = list(nx.strongly_connected_components(graph))
        non_trivial_scc = [c for c in scc if len(c) > 1]
        articulation_points = list(nx.articulation_points(undirected))

        try:
            avg_clustering = nx.average_clustering(undirected)
        except:
            avg_clustering = 0.0

        return {
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "num_wcc": nx.number_weakly_connected_components(graph),
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "num_scc": len(non_trivial_scc),
            "num_non_trivial_scc": len(non_trivial_scc), # Alias for compat
            "scc_sizes": [len(c) for c in non_trivial_scc],
            "non_trivial_scc": [list(c) for c in non_trivial_scc],
            "articulation_points": articulation_points,
            "average_clustering": round(avg_clustering, 4),
        }

    def _classify_anomalous_nodes(self, graph: nx.DiGraph) -> Dict[str, List[Dict[str, Any]]]:
        """Clasifica nodos anómalos."""
        result = {"isolated_nodes": [], "orphan_insumos": [], "empty_apus": []}
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        for node, data in graph.nodes(data=True):
            node_type = data.get("type")
            if node_type == "ROOT": continue

            in_deg = in_degrees.get(node, 0)
            out_deg = out_degrees.get(node, 0)

            node_info = {
                "id": node,
                "type": node_type,
                "description": data.get("description", ""),
                "inferred": data.get("inferred", False),
                "in_degree": in_deg,
                "out_degree": out_deg,
            }

            is_isolated = in_deg == 0 and out_deg == 0
            if is_isolated:
                result["isolated_nodes"].append(node_info)
                if node_type == "INSUMO": result["orphan_insumos"].append(node_info)
            elif node_type == "INSUMO" and in_deg == 0:
                result["orphan_insumos"].append(node_info)
            elif node_type == "APU" and out_deg == 0:
                result["empty_apus"].append(node_info)
        return result

    def _identify_critical_resources(self, graph: nx.DiGraph, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identifica recursos (insumos) críticos por centralidad de grado."""
        resources = []
        for node, data in graph.nodes(data=True):
            if data.get("type") == "INSUMO":
                degree = graph.in_degree(node)
                if degree > 0:
                    resources.append({
                        "id": node,
                        "in_degree": degree,
                        "description": data.get("description", "")
                    })
        resources.sort(key=lambda x: x["in_degree"], reverse=True)
        return resources[:top_n]

    def _interpret_topology(self, metrics: TopologicalMetrics) -> Dict[str, str]:
        """Genera interpretaciones semánticas (Compatibilidad hacia atrás)."""
        connectivity_status = "Espacio conexo" if metrics.is_connected else "Espacio fragmentado"
        cycle_status = "Estructura acíclica" if metrics.beta_1 == 0 else "Complejidad cíclica presente"

        return {
            "beta_0": f"{metrics.beta_0} componente(s) conexa(s). {connectivity_status}.",
            "beta_1": f"{metrics.beta_1} ciclo(s) independiente(s). {cycle_status}.",
            "euler": f"Característica de Euler: χ = {metrics.euler_characteristic}",
            "efficiency": f"Eficiencia de Euler: {metrics.euler_efficiency:.2%}",
        }

    def generate_executive_report(
        self, graph: nx.DiGraph, financial_metrics: Optional[Dict[str, Any]] = None
    ) -> ConstructionRiskReport:
        """
        Genera reporte de riesgos con modelo de scoring bayesiano (Propuesta 1).
        Inyecta la narrativa de la Propuesta 2.
        """
        metrics = self.calculate_betti_numbers(graph)
        raw_cycles, _ = self._get_raw_cycles(graph)
        cycles = [" → ".join(c + [c[0]]) for c in raw_cycles]

        synergy = self.detect_risk_synergy(graph, raw_cycles)
        anomalies = self._classify_anomalous_nodes(graph)
        pyramid_stability = self.calculate_pyramid_stability(graph)
        connectivity = self._compute_connectivity_analysis(graph)

        # Scoring Bayesiano
        density = nx.density(graph) if graph else 0.0
        euler_factor = metrics.euler_efficiency
        density_factor = 1.0 - min(density, 0.99)
        stability_factor = min(pyramid_stability / 3.0, 1.0)

        weights = {"euler": 0.4, "density": 0.3, "stability": 0.3}
        integrity_score = 100.0 * (
            weights["euler"] * euler_factor
            + weights["density"] * density_factor
            + weights["stability"] * stability_factor
        )

        # Penalizaciones
        penalty_multiplier = 1.0
        if metrics.beta_1 > 0:
            penalty_multiplier -= min(0.5, metrics.beta_1 * 0.1)
        if synergy["synergy_detected"]:
            penalty_multiplier -= min(0.3, synergy["synergy_score"] * 0.5)

        iso_count = len(anomalies["isolated_nodes"])
        orphan_count = len(anomalies["orphan_insumos"])
        penalty_multiplier -= min(0.2, (iso_count + orphan_count) * 0.05)

        integrity_score *= max(0.0, penalty_multiplier)
        integrity_score = round(max(0.0, min(100.0, integrity_score)), 1)

        # Complejidad
        complexity_score = (
            0.4 * (metrics.beta_1 / max(1, graph.number_of_nodes()))
            + 0.3 * density
            + 0.3 * (1.0 - metrics.euler_efficiency)
        )
        if complexity_score > 0.3:
            complexity_level = "Alta (Crítica)"
        elif complexity_score > 0.15:
            complexity_level = "Media"
        else:
            complexity_level = "Baja"

        # Alertas y Riesgos (Listas)
        waste_alerts = []
        if iso_count > 0: waste_alerts.append(f"🚨 {iso_count} nodos aislados detectados.")
        if orphan_count > 0: waste_alerts.append(f"⚠️ {orphan_count} insumos huérfanos.")
        if metrics.euler_efficiency < 0.6: waste_alerts.append(f"⚠️ Baja eficiencia topológica ({metrics.euler_efficiency:.2f}).")

        circular_risks = []
        if metrics.beta_1 > 0: circular_risks.append(f"🚨 CRÍTICO: {metrics.beta_1} ciclo(s) de dependencia.")
        if synergy["synergy_detected"]: circular_risks.append(f"🚨 RIESGO SISTÉMICO: Sinergia detectada (score: {synergy['synergy_score']:.2f}).")

        # Riesgo Financiero
        financial_risk = None
        if financial_metrics:
            volatility = financial_metrics.get("volatility", 0.0)
            roi = financial_metrics.get("roi", 0.0)
            if roi < 0: financial_risk = "CRÍTICO"
            elif volatility > 0.25: financial_risk = "ALTO"
            elif volatility > 0.15: financial_risk = "MEDIO"
            else: financial_risk = "BAJO"

            if (metrics.beta_1 > 2 or synergy["synergy_detected"]) and financial_risk in ["ALTO", "MEDIO"]:
                financial_risk = "CATÁSTROFICO"

        # Inyección de Narrativa (Propuesta 2)
        strategic_narrative = self._generate_strategic_narrative(
            metrics, synergy, pyramid_stability, financial_risk
        )

        return ConstructionRiskReport(
            integrity_score=integrity_score,
            waste_alerts=waste_alerts,
            circular_risks=circular_risks,
            complexity_level=complexity_level,
            financial_risk_level=financial_risk,
            strategic_narrative=strategic_narrative,
            details={
                "metrics": asdict(metrics),
                "cycles": cycles,
                "anomalies": anomalies,
                "synergy_risk": synergy,
                "connectivity": connectivity,
                "pyramid_stability": pyramid_stability,
                "density": density,
            },
        )

    def _generate_strategic_narrative(
        self,
        metrics: TopologicalMetrics,
        synergy: Dict[str, Any],
        stability: float,
        financial_risk: Optional[str],
    ) -> str:
        """
        Genera una narrativa estratégica con el tono del 'Consejo de Sabios' (Propuesta 2).
        Integra los conceptos de 'El Intérprete Diplomático'.
        """
        narrative_parts = []

        # 1. Análisis Estructural (La Base)
        if stability > 2.0:
            narrative_parts.append("🏗️ ESTRUCTURA ANTISÍSMICA: La pirámide presupuestaria posee una base robusta y bien distribuida.")
        elif stability > 1.0:
            narrative_parts.append("✅ CIMENTACIÓN ESTABLE: La relación entre insumos y APUs es adecuada para soportar la carga del proyecto.")
        else:
            narrative_parts.append("⚠️ RIESGO DE COLAPSO (PIRÁMIDE INVERTIDA): La base de recursos es insuficiente para la complejidad de los APUs definidos.")

        # 2. Integridad Lógica (Topología)
        if metrics.beta_1 == 0:
            narrative_parts.append("La trazabilidad de cargas es limpia (Acíclica).")
        else:
            narrative_parts.append(f"⛔ SOCAVONES LÓGICOS DETECTADOS: Existen {metrics.beta_1} ciclos de dependencia que comprometen la integridad del cálculo.")

        # 3. Sinergia de Riesgo (Efecto Dominó)
        if synergy.get("synergy_detected"):
            narrative_parts.append(f"☣️ RIESGO DE CONTAGIO: Se detectó una 'Sinergia de Riesgo' en {synergy.get('intersecting_cycles_count', 0)} puntos críticos. Un fallo en un insumo clave podría desencadenar un efecto dominó.")

        # 4. Veredicto Financiero (El Oráculo)
        if financial_risk:
            if financial_risk in ["CRÍTICO", "CATÁSTROFICO"]:
                narrative_parts.append(f"💀 ALERTA DE VIABILIDAD: El perfil de riesgo financiero es {financial_risk}, agravado por la estructura topológica.")
            elif financial_risk == "ALTO":
                narrative_parts.append("📉 PRECAUCIÓN FINANCIERA: Alta volatilidad detectada en los componentes críticos.")
            elif financial_risk == "BAJO":
                narrative_parts.append("💰 SALUD FINANCIERA: Los indicadores económicos respaldan la viabilidad técnica.")

        return " ".join(narrative_parts)

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Wrapper de análisis compatible con el pipeline actual y telemetría."""
        report = self.generate_executive_report(graph)
        metrics = TopologicalMetrics(**report.details["metrics"])

        flat_results = {
            "business.integrity_score": report.integrity_score,
            "business.pyramid_stability": report.details["pyramid_stability"],
            "business.betti_b0": metrics.beta_0,
            "business.betti_b1": metrics.beta_1,
            "business.euler_characteristic": metrics.euler_characteristic,
            "business.euler_efficiency": metrics.euler_efficiency,
            "business.cycles_count": len(report.details["cycles"]),
            "business.synergy_detected": 1 if report.details["synergy_risk"]["synergy_detected"] else 0,
            "business.is_dag": 1 if report.details["connectivity"]["is_dag"] else 0,
            "business.isolated_count": len(report.details["anomalies"]["isolated_nodes"]),
            "business.orphan_insumos_count": len(report.details["anomalies"]["orphan_insumos"]),
            "business.empty_apus_count": len(report.details["anomalies"]["empty_apus"]),
            "details": {
                "executive_report": asdict(report),
                "topology": {"betti_numbers": asdict(metrics)},
                "cycles": {"list": report.details["cycles"]},
                "connectivity": report.details["connectivity"],
                "anomalies": report.details["anomalies"],
                "critical_resources": self._identify_critical_resources(graph),
                "graph_summary": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "density": report.details["density"],
                    "pyramid_stability": report.details["pyramid_stability"]
                }
            }
        }

        # Emisión de Telemetría
        if self.telemetry:
            for k, v in flat_results.items():
                if isinstance(v, (int, float)):
                    self.telemetry.record_metric(k, v)

        return flat_results

    def get_audit_report(self, analysis_result_or_graph: Any) -> List[str]:
        """Genera un reporte ASCII art profesional."""
        if isinstance(analysis_result_or_graph, nx.DiGraph):
            analysis = self.analyze_structural_integrity(analysis_result_or_graph)
        else:
            analysis = analysis_result_or_graph

        report_dict = analysis.get("details", {}).get("executive_report", {})
        if not report_dict:
             return ["Error: No se pudo generar el reporte."]

        lines = []
        lines.append("┌──────────────────────────────────────────────────┐")
        lines.append("│      AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO       │")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append(f"│ PUNTUACIÓN DE INTEGRIDAD: {report_dict.get('integrity_score', 0):>6.1f} / 100.0          │")
        lines.append(f"│ Nivel de Complejidad:     {report_dict.get('complexity_level', ''):<23}│")
        lines.append("├──────────────────────────────────────────────────┤")

        metrics = report_dict.get("details", {}).get("metrics", {})
        lines.append("│ [MÉTRICAS TÉCNICAS]                              │")
        lines.append(f"│ Ciclos de Costo:           {metrics.get('beta_1', 0):<22}│")
        lines.append(f"│ Eficiencia de Euler:       {metrics.get('euler_efficiency', 0.0):<22.2f}│")
        lines.append("└──────────────────────────────────────────────────┘")

        if report_dict.get('circular_risks'):
            lines.append("│ [ALERTA CRÍTICA] Referencias circulares detectadas! │")
            for risk in report_dict['circular_risks']:
                 wrapped_lines = textwrap.wrap(risk, width=44)
                 for line in wrapped_lines:
                      lines.append(f"│ ❌ {line:<44} │")

        waste_alerts = report_dict.get('waste_alerts', [])
        anomalies = analysis.get("details", {}).get("anomalies", {})
        iso_count = len(anomalies.get("isolated_nodes", []))
        orphan_count = len(anomalies.get("orphan_insumos", []))
        empty_count = len(anomalies.get("empty_apus", []))

        if waste_alerts or iso_count > 0 or orphan_count > 0 or empty_count > 0:
            lines.append("├──────────────────────────────────────────────────┤")
            lines.append("│ [POSIBLE DESPERDICIO / ALERTAS]                  │")
            if iso_count > 0: lines.append(f"│ ⚠ Recursos Fantasma (Sin uso): {iso_count:<18}│")
            if empty_count > 0: lines.append(f"│ ⚠ APUs Vacíos:          {empty_count:<25}│")
            for alert in waste_alerts:
                wrapped_lines = textwrap.wrap(alert, width=44)
                for line in wrapped_lines:
                    lines.append(f"│ ⚠ {line:<44} │")
            lines.append("└──────────────────────────────────────────────────┘")

        return lines
