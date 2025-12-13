import networkx as nx
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from app.telemetry import TelemetryContext
from app.constants import ColumnNames

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TopologicalMetrics:
    """Invariant Topological Metrics for Business Graph."""
    beta_0: int  # Number of connected components (fragmentation)
    beta_1: int  # Number of independent cycles (loop complexity)
    euler_characteristic: int     # Euler Characteristic (beta_0 - beta_1)

    @property
    def is_connected(self) -> bool:
        return self.beta_0 == 1

    @property
    def is_simply_connected(self) -> bool:
        return self.beta_0 == 1 and self.beta_1 == 0

class BudgetGraphBuilder:
    """Construye el Grafo del Presupuesto (Business Topology) Version 2."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _sanitize_code(self, value: Any) -> str:
        if pd.isna(value) or value is None:
            return ""
        return str(value).strip()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def _create_apu_attributes(self, row: pd.Series, source: str, idx: int, inferred: bool) -> Dict[str, Any]:
        attrs = {
            "type": "APU",
            "source": source,
            "original_index": idx,
            "inferred": inferred
        }
        if not inferred:
            attrs["description"] = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_APU))
            attrs["quantity"] = self._safe_float(row.get(ColumnNames.CANTIDAD_PRESUPUESTO))
        return attrs

    def _create_insumo_attributes(self, row: pd.Series, insumo_desc: str, source: str, idx: int) -> Dict[str, Any]:
        return {
            "type": "INSUMO",
            "description": insumo_desc,
            "tipo_insumo": self._sanitize_code(row.get(ColumnNames.TIPO_INSUMO)),
            "unit_cost": self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU)),
            "source": source,
            "original_index": idx
        }

    def _upsert_edge(self, G: nx.DiGraph, u: str, v: str, unit_cost: float, quantity: float, idx: int) -> bool:
        """
        Inserta o actualiza una arista acumulando cantidades y costos.
        Retorna True si es una nueva arista, False si se actualizó.
        """
        total_cost = unit_cost * quantity
        is_new = False

        if G.has_edge(u, v):
            edge_data = G[u][v]
            # Accumulate
            new_qty = edge_data['quantity'] + quantity
            new_total = edge_data['total_cost'] + total_cost

            # Update attributes
            G[u][v]['quantity'] = new_qty
            G[u][v]['total_cost'] = new_total
            G[u][v]['occurrence_count'] += 1
            if 'original_indices' in G[u][v]:
                G[u][v]['original_indices'].append(idx)
        else:
            is_new = True
            G.add_edge(
                u,
                v,
                quantity=quantity,
                unit_cost=unit_cost,
                total_cost=total_cost,
                occurrence_count=1,
                original_indices=[idx]
            )
        return is_new

    def _compute_graph_statistics(self, G: nx.DiGraph) -> Dict[str, int]:
        apu_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'APU']
        insumo_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'INSUMO']
        inferred_apus = [n for n in apu_nodes if G.nodes[n].get('inferred', False)]

        return {
            "apu_count": len(apu_nodes),
            "insumo_count": len(insumo_nodes),
            "inferred_count": len(inferred_apus),
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges()
        }

    def build(self, presupuesto_df: pd.DataFrame, apus_detail_df: pd.DataFrame) -> nx.DiGraph:
        """
        Construye un grafo dirigido optimizado.
        """
        G = nx.DiGraph(name="BudgetTopology")
        self.logger.info("Iniciando construcción del Budget Graph V2...")

        # 1. Agregar nodos APU desde presupuesto
        if presupuesto_df is not None and not presupuesto_df.empty:
            for idx, row in presupuesto_df.iterrows():
                apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
                if not apu_code:
                    continue

                attrs = self._create_apu_attributes(row, source="presupuesto", idx=idx, inferred=False)
                # Use add_node which updates attributes if exists
                G.add_node(apu_code, **attrs)

        # 2. Procesar detalle de APUs
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                apu_code = self._sanitize_code(row.get(ColumnNames.CODIGO_APU))
                insumo_desc = self._sanitize_code(row.get(ColumnNames.DESCRIPCION_INSUMO))

                if not apu_code or not insumo_desc:
                    continue

                # Ensure APU node exists (inference)
                if apu_code not in G:
                    attrs = self._create_apu_attributes(row, source="detail", idx=idx, inferred=True)
                    G.add_node(apu_code, **attrs)

                # Ensure Insumo node exists
                # Using description as ID for consistency with V1
                insumo_id = insumo_desc
                if insumo_id not in G:
                    attrs = self._create_insumo_attributes(row, insumo_desc, source="detail", idx=idx)
                    G.add_node(insumo_id, **attrs)

                # Upsert Edge
                qty = self._safe_float(row.get(ColumnNames.CANTIDAD_APU))
                cost = self._safe_float(row.get(ColumnNames.COSTO_INSUMO_EN_APU))

                self._upsert_edge(G, apu_code, insumo_id, cost, qty, idx)

        stats = self._compute_graph_statistics(G)
        self.logger.info(f"Grafo construido: {stats}")
        return G

class BusinessTopologicalAnalyzer:
    """Analizador de topología de negocio V2 con Telemetría Granular."""

    def __init__(self, telemetry: Optional[TelemetryContext] = None, max_cycles: int = 100):
        self.telemetry = telemetry
        self.max_cycles = max_cycles
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_betti_numbers(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """
        Calcula métricas topológicas invariantes (Betti Numbers).
        """
        if graph.number_of_nodes() == 0:
            return TopologicalMetrics(0, 0, 0)

        # Convert to MultiGraph to preserve edge multiplicity logic if needed,
        # but typical Betti calculation on abstract simplicial complex from graph
        # usually considers 1-skeleton.
        # However, for cycle detection in business logic, parallel edges matter?
        # Usually not for Betti_1 of the underlying graph unless they form distinct cycles.
        # But let's follow the standard graph topology definition where multigraphs have distinct edges.
        # If A->B and B->A, that's a cycle.

        undirected = nx.MultiGraph()
        undirected.add_nodes_from(graph.nodes(data=True))
        undirected.add_edges_from(graph.edges(data=True))

        # Beta 0: Componentes Conexas
        beta_0 = nx.number_connected_components(undirected)

        # Counts
        n_edges = undirected.number_of_edges()
        n_nodes = undirected.number_of_nodes()

        # Beta 1: Ciclos
        # Euler Characteristic chi = V - E = beta_0 - beta_1 (for 1D complex)
        # So beta_1 = beta_0 - V + E
        beta_1 = max(0, beta_0 - n_nodes + n_edges)

        chi = n_nodes - n_edges # Wait, definition varies.
        # Usually Chi = Sum((-1)^k * beta_k). For graph: beta_0 - beta_1.
        # And Chi = V - E.
        # So beta_0 - beta_1 = V - E => beta_1 = beta_0 - V + E.
        # Let's use Chi = beta_0 - beta_1 as the "Topological Euler Characteristic".

        chi = beta_0 - beta_1

        return TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            euler_characteristic=chi
        )

    # Alias for backward compatibility if needed, but the proposal doesn't use it.
    # We will keep it but delegating.
    def calculate_metrics(self, graph: nx.DiGraph) -> TopologicalMetrics:
        return self.calculate_betti_numbers(graph)

    def _detect_cycles(self, graph: nx.DiGraph) -> Tuple[List[str], bool]:
        """Detecta ciclos y retorna una lista de representaciones string y un flag si fue truncado."""
        cycles = []
        truncated = False
        try:
            raw_cycles = list(nx.simple_cycles(graph))
            if len(raw_cycles) > self.max_cycles:
                raw_cycles = raw_cycles[:self.max_cycles]
                truncated = True

            for cycle in raw_cycles:
                # Add start node to end to close the loop in representation
                cycle_repr = cycle + [cycle[0]]
                cycles.append(" → ".join(map(str, cycle_repr)))
        except Exception as e:
            self.logger.error(f"Error detecting cycles: {e}")
        return cycles, truncated

    def _classify_anomalous_nodes(self, graph: nx.DiGraph) -> Dict[str, List[Dict[str, Any]]]:
        """Clasifica nodos en categorías anómalas."""
        isolated_nodes = []
        orphan_insumos = []
        empty_apus = []

        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        for node, data in graph.nodes(data=True):
            node_info = {
                "id": node,
                "type": data.get("type"),
                "description": data.get("description", ""),
                "inferred": data.get("inferred", False),
                "in_degree": in_degrees.get(node, 0),
                "out_degree": out_degrees.get(node, 0)
            }

            is_isolated = (in_degrees.get(node, 0) == 0) and (out_degrees.get(node, 0) == 0)

            if is_isolated:
                isolated_nodes.append(node_info)

            if data.get("type") == "INSUMO" and in_degrees.get(node, 0) == 0 and not is_isolated:
                # Insumo that is not isolated but has no incoming edges?
                # Wait, Insumos usually come from APUs.
                # If it's connected to something (out > 0) but in=0? Insumos usually don't have out edges in this model (APU->Insumo).
                # The prompt test says "orphans detected" -> in_degree = 0.
                orphan_insumos.append(node_info)
            elif data.get("type") == "INSUMO" and is_isolated:
                 # Isolated insumos are also orphans
                 orphan_insumos.append(node_info)

            if data.get("type") == "APU" and out_degrees.get(node, 0) == 0:
                empty_apus.append(node_info)

        return {
            "isolated_nodes": isolated_nodes,
            "orphan_insumos": orphan_insumos,
            "empty_apus": empty_apus
        }

    def _identify_critical_resources(self, graph: nx.DiGraph, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identifica recursos críticos (alto in-degree)."""
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

    def _compute_connectivity_analysis(self, graph: nx.DiGraph) -> Dict[str, Any]:
        return {
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "num_wcc": nx.number_weakly_connected_components(graph),
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "num_non_trivial_scc": len([c for c in nx.strongly_connected_components(graph) if len(c) > 1]),
            "non_trivial_scc": [list(c) for c in nx.strongly_connected_components(graph) if len(c) > 1]
        }

    def _interpret_topology(self, metrics: TopologicalMetrics) -> Dict[str, str]:
        return {
            "beta_0": f"{metrics.beta_0} componente(s) conexa(s). {'Espacio conexo' if metrics.is_connected else 'Espacio fragmentado'}.",
            "beta_1": f"{metrics.beta_1} ciclo(s) independiente(s). {'Estructura acíclica' if metrics.beta_1 == 0 else 'Complejidad ciclica presente'}.",
            "euler": f"Característica de Euler: {metrics.euler_characteristic}"
        }

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Ejecuta análisis completo y emite telemetría."""
        metrics = self.calculate_betti_numbers(graph)
        cycles, cycles_truncated = self._detect_cycles(graph)
        anomalies = self._classify_anomalous_nodes(graph)
        critical = self._identify_critical_resources(graph)
        connectivity = self._compute_connectivity_analysis(graph)
        interpretation = self._interpret_topology(metrics)

        # Structure for report
        details = {
            "topology": {
                "betti_numbers": asdict(metrics),
                "interpretation": interpretation
            },
            "cycles": {
                "count": len(cycles),
                "list": cycles,
                "truncated": cycles_truncated
            },
            "connectivity": connectivity,
            "anomalies": anomalies,
            "critical_resources": critical,
            "graph_summary": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph)
            }
        }

        # Flat metrics for telemetry/convenience
        flat_results = {
            "business.betti_b0": metrics.beta_0,
            "business.betti_b1": metrics.beta_1,
            "business.euler_characteristic": metrics.euler_characteristic,
            "business.cycles_count": len(cycles),
            "business.is_dag": 1 if connectivity["is_dag"] else 0,
            "business.isolated_count": len(anomalies["isolated_nodes"]),
            "business.orphan_insumos_count": len(anomalies["orphan_insumos"]),
            "business.empty_apus_count": len(anomalies["empty_apus"]),
            "details": details
        }

        # Telemetry
        if self.telemetry:
            for k, v in flat_results.items():
                if isinstance(v, (int, float)):
                    self.telemetry.record_metric(k, v)

        return flat_results

    def analyze(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Backward compatibility wrapper for analyze."""
        new_result = self.analyze_structural_integrity(graph)

        # Map to old structure:
        # metrics: {beta_0, beta_1, chi, density, is_dag}
        # anomalies: {cycles, isolates_count, orphan_insumos_count, empty_apus_count}

        metrics = TopologicalMetrics(
            beta_0=new_result["business.betti_b0"],
            beta_1=new_result["business.betti_b1"],
            euler_characteristic=new_result["business.euler_characteristic"]
        )
        # Add old fields that were in V1 but not strict topological invariants
        metrics_dict = asdict(metrics)
        metrics_dict["density"] = new_result["details"]["graph_summary"]["density"]
        metrics_dict["is_dag"] = bool(new_result["business.is_dag"])
        # Chi was aliased to euler_characteristic
        metrics_dict["chi"] = metrics.euler_characteristic

        anomalies = {
            "cycles": new_result["details"]["cycles"]["list"],
            "isolates_count": new_result["business.isolated_count"],
            "orphan_insumos_count": new_result["business.orphan_insumos_count"],
            "empty_apus_count": new_result["business.empty_apus_count"]
        }

        return {
            "metrics": metrics_dict,
            "anomalies": anomalies
        }

    def get_audit_report(self, analysis_result_or_graph: Any) -> List[str]:
        """Genera un reporte ASCII art profesional."""

        if isinstance(analysis_result_or_graph, nx.DiGraph):
            res = self.analyze_structural_integrity(analysis_result_or_graph)
        else:
            # If it's the old structure from `analyze` wrapper, we need to adapt or just fail/warn.
            # But the test calls `analyze_structural_integrity` then passes result here.
            # If someone passes the result of `analyze` (backward compat), it has 'metrics' and 'anomalies'.
            if "details" not in analysis_result_or_graph and "metrics" in analysis_result_or_graph:
                 # Reconstruct minimal details for reporting
                 m = analysis_result_or_graph["metrics"]
                 a = analysis_result_or_graph["anomalies"]
                 res = {
                     "details": {
                         "topology": {"betti_numbers": m},
                         "anomalies": {
                             "isolated_nodes": [{}] * a.get("isolates_count", 0),
                             "orphan_insumos": [{}] * a.get("orphan_insumos_count", 0),
                             "empty_apus": [{}] * a.get("empty_apus_count", 0)
                         },
                         "cycles": {"count": len(a.get("cycles", [])), "list": a.get("cycles", [])},
                         "connectivity": {"is_dag": m.get("is_dag", True)},
                         "graph_summary": {"density": m.get("density", 0.0)}
                     }
                 }
            else:
                res = analysis_result_or_graph

        details = res.get("details", {})
        m = details.get("topology", {}).get("betti_numbers", {})
        a = details.get("anomalies", {})
        c = details.get("cycles", {})
        conn = details.get("connectivity", {})
        summ = details.get("graph_summary", {})

        # Handle fallback if 'details' not present (e.g. if old analyze was called)
        if not details:
            # Fallback logic could go here
            pass

        lines = []
        lines.append("┌──────────────────────────────────────────────────┐")
        lines.append("│      REPORTE DE TOPOLOGÍA DE NEGOCIO (V2)        │")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append("│ [ANÁLISIS TOPOLÓGICO]                            │")
        lines.append(f"│ β₀ (Componentes):       {m.get('beta_0', 0):<25}│")
        lines.append(f"│ β₁ (Ciclos):            {m.get('beta_1', 0):<25}│")
        lines.append(f"│ χ  (Euler):             {m.get('euler_characteristic', 0):<25}│")
        lines.append(f"│ Densidad:               {summ.get('density', 0):.4f}                   │")
        lines.append("├──────────────────────────────────────────────────┤")

        if not conn.get('is_dag', True) or c.get('count', 0) > 0:
            lines.append("│ [ALERTAS CRÍTICAS]                               │")
            lines.append(f"│ ❌ CICLOS DETECTADOS:    {c.get('count', 0):<25}│")
            for i, cycle in enumerate(c.get('list', [])[:3]):
                # cycle string is already formatted
                # Truncate if too long for box
                c_str = cycle
                if len(c_str) > 38:
                    c_str = c_str[:35] + "..."
                lines.append(f"│    {i+1}. {c_str:<38} │")
        else:
            lines.append("│ [ESTADO]                                         │")
            lines.append("│ ✅ Grafo Acíclico (DAG) - Estructura Sólida      │")

        iso_count = len(a.get('isolated_nodes', []))
        orphan_count = len(a.get('orphan_insumos', []))
        empty_count = len(a.get('empty_apus', []))

        if iso_count > 0 or orphan_count > 0 or empty_count > 0:
            lines.append("├──────────────────────────────────────────────────┤")
            lines.append("│ [ADVERTENCIAS]                                   │")
            if orphan_count:
                lines.append(f"│ ⚠ Insumos Huérfanos:    {orphan_count:<25}│")
            if empty_count:
                lines.append(f"│ ⚠ APUs Vacíos:          {empty_count:<25}│")
            if iso_count:
                 lines.append(f"│ ⚠ Nodos Aislados:       {iso_count:<25}│")

        lines.append("└──────────────────────────────────────────────────┘")

        return lines
