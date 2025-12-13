import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
import logging
from dataclasses import dataclass
from app.telemetry import TelemetryContext
from app.constants import ColumnNames

logger = logging.getLogger(__name__)

class BudgetGraphBuilder:
    """Construye el Grafo del Presupuesto (Business Topology)."""

    def build(self, presupuesto_df: pd.DataFrame, apus_detail_df: pd.DataFrame) -> nx.DiGraph:
        """
        Construye un grafo dirigido donde:
        - Nodos: APUs (Actividades) e Insumos (Recursos).
        - Aristas: APU -> Insumo (relación "contiene").
        """
        G = nx.DiGraph()

        logger.info("Building Budget Graph...")

        # 1. Agregar Nodos APU desde presupuesto
        if presupuesto_df is not None and not presupuesto_df.empty:
            for _, row in presupuesto_df.iterrows():
                apu_code = str(row.get(ColumnNames.CODIGO_APU, "")).strip()
                if not apu_code:
                    continue

                # Evitar sobrescribir si ya existe (aunque aquí es la fuente primaria)
                if apu_code not in G:
                    G.add_node(apu_code, type="APU",
                               description=row.get(ColumnNames.DESCRIPCION_APU),
                               quantity=row.get(ColumnNames.CANTIDAD_PRESUPUESTO))

        # 2. Agregar Nodos Insumo y Aristas desde apus_detail
        if apus_detail_df is not None and not apus_detail_df.empty:
            for _, row in apus_detail_df.iterrows():
                apu_code = str(row.get(ColumnNames.CODIGO_APU, "")).strip()
                insumo_desc = str(row.get(ColumnNames.DESCRIPCION_INSUMO, "")).strip()

                if not apu_code or not insumo_desc:
                    continue

                # Asegurar que el APU existe (si no estaba en presupuesto, se agrega como nodo inferido)
                if apu_code not in G:
                    G.add_node(apu_code, type="APU", inferred=True)

                # Nodo Insumo
                # Usamos la descripción como ID por simplicidad y consistencia con DataMerger
                insumo_id = insumo_desc

                if insumo_id not in G:
                     G.add_node(insumo_id, type="INSUMO",
                                tipo_insumo=row.get(ColumnNames.TIPO_INSUMO),
                                costo=row.get(ColumnNames.COSTO_INSUMO_EN_APU))

                # Arista APU -> Insumo
                # Peso podría ser costo total o cantidad. Usaremos costo para análisis de valor.
                costo = row.get(ColumnNames.COSTO_INSUMO_EN_APU, 0)
                cantidad = row.get(ColumnNames.CANTIDAD_APU, 0)

                G.add_edge(apu_code, insumo_id,
                           cantidad=cantidad,
                           costo=costo)

        logger.info(f"Budget Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

class BusinessTopologicalAnalyzer:
    """Analizador de la topología de negocio."""

    def __init__(self, telemetry: Optional[TelemetryContext] = None):
        self.telemetry = telemetry

    def analyze_structural_integrity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Calcula Betti numbers y detecta anomalías.
        """

        # Beta 1: Ciclos (Dependencias circulares)
        # nx.simple_cycles es costoso para grafos muy grandes, pero el presupuesto suele ser un árbol.
        # Si hay ciclos, es un error grave.
        try:
            # Limitamos la búsqueda si es demasiado grande, aunque para presupuestos debería ser manejable.
            # simple_cycles retorna un iterador.
            cycles = list(nx.simple_cycles(graph))
            b1 = len(cycles)
        except Exception as e:
            logger.error(f"Error calculating cycles: {e}")
            cycles = []
            b1 = -1

        # Beta 0: Componentes conexas
        # Para grafos dirigidos, usamos weakly connected components para ver islas.
        try:
            b0 = nx.number_weakly_connected_components(graph)
        except Exception:
            b0 = 0

        # Identificar nodos aislados (Grado 0)
        # Ojo: en DiGraph, degree es in_degree + out_degree.
        isolates = list(nx.isolates(graph))

        # Clasificar isolados
        orphan_resources = []
        empty_activities = []

        for node in isolates:
            node_data = graph.nodes[node]
            node_type = node_data.get("type")
            if node_type == "INSUMO":
                orphan_resources.append(node)
            elif node_type == "APU":
                empty_activities.append(node)

        # Detectar insumos que no son usados por ningun APU (in_degree = 0) pero NO son APUs
        # Un insumo debería tener in_degree > 0 (ser consumido por alguien)
        # Excluyendo los que ya identificamos como totalmente aislados.
        unused_resources = []
        for node, in_deg in graph.in_degree():
             node_data = graph.nodes[node]
             if node_data.get("type") == "INSUMO" and in_deg == 0:
                 if node not in orphan_resources:
                     unused_resources.append(node)

        # Unir a orphan_resources para el reporte (insumos sin uso)
        orphan_resources.extend(unused_resources)
        orphan_resources = list(set(orphan_resources))

        # Descripción de dependencias circulares
        circular_dependencies = []
        for cycle in cycles:
            circular_dependencies.append(" -> ".join(cycle))

        metrics = {
            "business.b1_cycles": b1,
            "business.b0_components": b0,
            "business.circular_dependencies_count": len(cycles),
            "business.orphan_resources_count": len(orphan_resources),
            "business.empty_activities_count": len(empty_activities),
            "details": {
                "circular_dependencies": circular_dependencies,
                "orphan_resources": orphan_resources,
                "empty_activities": empty_activities
            }
        }

        # Inject into telemetry
        if self.telemetry:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.telemetry.record_metric("business_topology", k, v)

        return metrics

    def get_audit_report(self, analysis_result: Dict[str, Any]) -> List[str]:
        report = []

        cycles = analysis_result.get("details", {}).get("circular_dependencies", [])
        if cycles:
            report.append(f"Alerta: Se detectó una dependencia circular entre {len(cycles)} elementos. Esto causará errores de cálculo.")
            for c in cycles[:3]: # Mostrar solo los primeros 3
                report.append(f"  - Ciclo: {c}")
            if len(cycles) > 3:
                report.append(f"  - ... y {len(cycles)-3} más.")

        orphans = analysis_result.get("details", {}).get("orphan_resources", [])
        if orphans:
            report.append(f"Aviso: Hay {len(orphans)} insumos en la lista que no se utilizan en ninguna actividad.")

        empty_apus = analysis_result.get("details", {}).get("empty_activities", [])
        if empty_apus:
             report.append(f"Aviso: Hay {len(empty_apus)} actividades sin insumos definidos.")

        if not report:
            report.append("Su presupuesto es topológicamente sólido.")

        return report
