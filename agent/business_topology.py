import networkx as nx
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from app.telemetry import TelemetryContext
from app.constants import ColumnNames

logger = logging.getLogger(__name__)

@dataclass
class TopologicalMetrics:
    """Invariant Topological Metrics for Business Graph."""
    beta_0: int  # Number of connected components (fragmentation)
    beta_1: int  # Number of independent cycles (loop complexity)
    chi: int     # Euler Characteristic (beta_0 - beta_1)
    density: float
    is_dag: bool

class BudgetGraphBuilder:
    """Construye el Grafo del Presupuesto (Business Topology) Version 2."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _upsert_edge(self, G: nx.DiGraph, u: str, v: str, quantity: float, unit_cost: float, idx: int) -> None:
        """
        Inserta o actualiza una arista acumulando cantidades y costos.
        Maneja la lógica de 'agregar o actualizar' de manera limpia.
        """
        total_cost = unit_cost * quantity

        if G.has_edge(u, v):
            edge_data = G[u][v]
            # Accumulate
            new_qty = edge_data['quantity'] + quantity
            new_total = edge_data['total_cost'] + total_cost

            # Update attributes
            G[u][v]['quantity'] = new_qty
            G[u][v]['total_cost'] = new_total
            # We keep the original index of the first occurrence or append?
            # Usually keeping the first is enough for tracing, or we could append to a list.
            # Here we just update numerical values.
        else:
            G.add_edge(
                u,
                v,
                quantity=quantity,
                unit_cost=unit_cost,
                total_cost=total_cost,
                original_index=idx
            )

    def build(self, presupuesto_df: pd.DataFrame, apus_detail_df: pd.DataFrame) -> nx.DiGraph:
        """
        Construye un grafo dirigido optimizado.
        """
        G = nx.DiGraph()
        self.logger.info("Iniciando construcción del Budget Graph V2...")

        # 1. Agregar nodos APU desde presupuesto
        if presupuesto_df is not None and not presupuesto_df.empty:
            for idx, row in presupuesto_df.iterrows():
                apu_code = str(row.get(ColumnNames.CODIGO_APU, "")).strip()
                if not apu_code:
                    continue

                # Use setdefault equivalent for nodes to avoid overwriting if needed,
                # but add_node updates attributes if exists.
                G.add_node(
                    apu_code,
                    type="APU",
                    description=row.get(ColumnNames.DESCRIPCION_APU, ""),
                    quantity=row.get(ColumnNames.CANTIDAD_PRESUPUESTO, 0.0),
                    source="presupuesto",
                    original_index=idx
                )

        # 2. Procesar detalle de APUs
        if apus_detail_df is not None and not apus_detail_df.empty:
            for idx, row in apus_detail_df.iterrows():
                apu_code = str(row.get(ColumnNames.CODIGO_APU, "")).strip()
                insumo_desc = str(row.get(ColumnNames.DESCRIPCION_INSUMO, "")).strip()

                if not apu_code or not insumo_desc:
                    continue

                # Ensure APU node exists (inference)
                if apu_code not in G:
                    G.add_node(apu_code, type="APU", inferred=True, source="detail")

                # Ensure Insumo node exists
                # Using description as ID for consistency with V1, though Code is better if available
                insumo_id = insumo_desc
                if insumo_id not in G:
                    G.add_node(
                        insumo_id,
                        type="INSUMO",
                        description=insumo_desc,
                        tipo_insumo=row.get(ColumnNames.TIPO_INSUMO, ""),
                        source="detail"
                    )

                # Upsert Edge
                qty = float(row.get(ColumnNames.CANTIDAD_APU, 0.0))
                cost = float(row.get(ColumnNames.COSTO_INSUMO_EN_APU, 0.0))

                self._upsert_edge(G, apu_code, insumo_id, qty, cost, idx)

        self.logger.info(f"Grafo construido: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        return G

class BusinessTopologicalAnalyzer:
    """Analizador de topología de negocio V2 con Telemetría Granular."""

    def __init__(self, telemetry: Optional[TelemetryContext] = None):
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_metrics(self, graph: nx.DiGraph) -> TopologicalMetrics:
        """
        Calcula métricas topológicas invariantes.
        β1 = |E| - |V| + β0
        """
        if graph.number_of_nodes() == 0:
            return TopologicalMetrics(0, 0, 0, 0.0, True)

        # Convert to MultiGraph to preserve edge multiplicity (e.g., A->B and B->A should count as 2 edges)
        # This ensures that reciprocal cycles contribute to Beta 1.
        # Note: nx.MultiGraph(digraph) collapses reciprocal edges if not added explicitly via add_edges_from
        undirected = nx.MultiGraph()
        undirected.add_nodes_from(graph.nodes(data=True))
        undirected.add_edges_from(graph.edges(data=True))

        # Beta 0: Componentes Conexas (calculated on standard Graph to avoid MultiGraph overhead for connectivity)
        # Note: number_connected_components works on MultiGraph too, but let's be explicit.
        beta_0 = nx.number_connected_components(graph.to_undirected())

        # Counts
        n_edges = undirected.number_of_edges()
        n_nodes = undirected.number_of_nodes()

        # Beta 1: Ciclos (Euler-Poincaré formula rearrangement)
        # Chi (Euler Characteristic) = V - E = beta_0 - beta_1
        # Thus: beta_1 = beta_0 - (V - E) = beta_0 - V + E
        beta_1 = max(0, beta_0 - n_nodes + n_edges)

        # Euler Characteristic
        chi = beta_0 - beta_1

        # Density
        density = nx.density(graph)

        # DAG Check
        is_dag = nx.is_directed_acyclic_graph(graph)

        return TopologicalMetrics(
            beta_0=beta_0,
            beta_1=beta_1,
            chi=chi,
            density=density,
            is_dag=is_dag
        )

    def analyze(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Ejecuta análisis completo y emite telemetría."""
        metrics = self.calculate_metrics(graph)

        # Anomalies
        isolates = list(nx.isolates(graph))
        cycles = list(nx.simple_cycles(graph)) if not metrics.is_dag else []

        # Specific Anomalies
        orphan_insumos = [n for n in isolates if graph.nodes[n].get('type') == 'INSUMO']
        empty_apus = [n for n in isolates if graph.nodes[n].get('type') == 'APU']

        results = {
            "metrics": asdict(metrics),
            "anomalies": {
                "cycles": cycles,
                "isolates_count": len(isolates),
                "orphan_insumos_count": len(orphan_insumos),
                "empty_apus_count": len(empty_apus)
            }
        }

        # Telemetry
        if self.telemetry:
            self.telemetry.record_metric("business.beta_0", metrics.beta_0)
            self.telemetry.record_metric("business.beta_1", metrics.beta_1)
            self.telemetry.record_metric("business.is_dag", 1 if metrics.is_dag else 0)
            self.telemetry.record_metric("business.graph_density", metrics.density)

        return results

    def get_audit_report(self, graph: nx.DiGraph) -> str:
        """Genera un reporte ASCII art profesional."""
        results = self.analyze(graph)
        m = results['metrics']
        a = results['anomalies']

        lines = []
        lines.append("┌──────────────────────────────────────────────────┐")
        lines.append("│      REPORTE DE TOPOLOGÍA DE NEGOCIO (V2)        │")
        lines.append("├──────────────────────────────────────────────────┤")
        lines.append("│ [INVARIANTES]                                    │")
        lines.append(f"│ β₀ (Componentes):       {m['beta_0']:<25}│")
        lines.append(f"│ β₁ (Ciclos):            {m['beta_1']:<25}│")
        lines.append(f"│ χ  (Euler):             {m['chi']:<25}│")
        lines.append(f"│ Densidad:               {m['density']:.4f}                   │")
        lines.append("├──────────────────────────────────────────────────┤")

        if not m['is_dag'] or a['cycles']:
            lines.append("│ [ALERTAS CRÍTICAS]                               │")
            lines.append(f"│ ❌ CICLOS DETECTADOS:    {len(a['cycles']):<25}│")
            for i, cycle in enumerate(a['cycles'][:3]):
                c_str = " -> ".join(cycle[:3]) + "..."
                lines.append(f"│    {i+1}. {c_str:<38} │")
        else:
            lines.append("│ [ESTADO]                                         │")
            lines.append("│ ✅ Grafo Acíclico (DAG) - Estructura Sana        │")

        if a['isolates_count'] > 0:
            lines.append("├──────────────────────────────────────────────────┤")
            lines.append("│ [ADVERTENCIAS]                                   │")
            if a['orphan_insumos_count']:
                lines.append(f"│ ⚠ Insumos Huérfanos:    {a['orphan_insumos_count']:<25}│")
            if a['empty_apus_count']:
                lines.append(f"│ ⚠ APUs Vacíos:          {a['empty_apus_count']:<25}│")

        lines.append("└──────────────────────────────────────────────────┘")

        return "\n".join(lines)
